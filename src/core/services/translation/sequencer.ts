/**
 * Per-utterance sequencing for a text stage that turns a live interim/final text stream into another
 * one (translation today). It replaces the old single `#id` gate, which dropped a FINAL translation
 * whenever any newer event (even an interim the backend ignored) arrived while it was in flight, so
 * TTS and the VRChat chatbox, which act on finals only, silently lost sentences.
 *
 * Model (per stream, e.g. `stt` and `textfield`, which are independent):
 * - An utterance is a run of interims closed by one final. Interims never re-open a closed one.
 * - Interims are latest-wins: at most one request per utterance is in flight; newer interims replace
 *   the queued one, and a result is shown only if it is newer than the last one shown. The final
 *   aborts the utterance's interim work.
 * - Finals are never dropped by sequencing: every final result is published, in utterance order. A
 *   final that finishes early waits for earlier finals; interims of a later utterance are held (latest
 *   only) until every earlier utterance is settled. The published stream therefore stays well formed
 *   for consumers without ids (OBS captions, caption elements matching "the first interim"): for each
 *   utterance in order, interims with rising revision, then at most one final.
 * - Every job has a deadline, so a hung backend delays later utterances by at most that long. A final's
 *   deadline grows with its length (see `finalDeadlineMs`). A final that fails or times out is
 *   reported through `onError` and its utterance is closed without output.
 */

/** Published by Service_STT.tryCancelSentence (stt/index.ts) when a sentence is cut off mid-interim. */
export const CANCEL_MARKER = "[...]";

export function isCancelMarker(value: string): boolean {
  return value.trim() === CANCEL_MARKER;
}

export type SequenceKind = "interim" | "final";

export interface SequencerJob<E> {
  readonly stream: string;
  /** Utterance number within the stream, starting at 0. */
  readonly utterance: number;
  /** Rises with every event of the utterance; the final has the highest. */
  readonly revision: number;
  readonly kind: SequenceKind;
  readonly event: E;
  /** Aborted when the job is superseded, times out, or the sequencer is reset. */
  readonly signal: AbortSignal;
}

/** Rejection reason of a job that exceeded its deadline. */
export class SequencerTimeoutError extends Error {
  constructor(readonly timeoutMs: number) {
    super(`timed out after ${Math.round(timeoutMs / 100) / 10} s`);
    this.name = "SequencerTimeoutError";
  }
}

/** Abort reason for jobs cancelled by `reset()` or superseded by a newer event; never reported. */
class SequencerCancelled extends Error {
  constructor() {
    super("cancelled");
    this.name = "AbortError";
  }
}

export interface TranslationSequencerOptions<E> {
  /** Runs one job: resolve with the output text, reject on failure. Should honour `job.signal`. */
  run(job: SequencerJob<E>): Promise<string>;
  /** Receives outputs in order (see the module docs). */
  publish(job: SequencerJob<E>, value: string): void;
  /** A job failed or timed out. Not called for superseded or reset jobs. */
  onError?(job: SequencerJob<E>, error: unknown): void;
  /**
   * Deadlines per kind (ms), fixed or computed from the event. Defaults: interim 15 s, final
   * `finalDeadlineMs(event.value)` (30 s for caption-length text, more for long pasted text).
   */
  timeoutMs?: Partial<Record<SequenceKind, number | ((event: E) => number)>>;
  /**
   * An interim that arrives within this many ms after a final, while no utterance is open, and whose
   * text is a prefix of that final's text (ignoring case, punctuation and spacing, which finals often
   * add) is a late duplicate (STT can publish an interim after its final) and is ignored instead of
   * opening a new utterance that would never be finalized. 0 = off.
   */
  staleInterimMs?: number;
  now?: () => number;
}

type FinalState = "open" | "pending" | "ready" | "dropped";

interface Utterance<E> {
  readonly id: number;
  revision: number;
  /** Revision last published (-1 = nothing published for this utterance yet). */
  shown: number;
  final: FinalState;
  finalOut?: { job: SequencerJob<E>; value: string };
  inFlight?: { revision: number; controller: AbortController };
  /** Latest interim waiting for the in-flight one to settle. */
  queued?: { revision: number; event: E };
  /** Latest interim result waiting for earlier utterances to settle. */
  held?: { job: SequencerJob<E>; value: string };
}

interface StreamState<E> {
  nextUtterance: number;
  open?: Utterance<E>;
  /** Unsettled utterances in order; the head is the only one allowed to publish. */
  queue: Utterance<E>[];
  lastClosed?: { text: string; at: number };
}

/** Deadline of an interim (ms): a partial that takes longer is useless anyway. */
export const INTERIM_TIMEOUT_MS = 15_000;
/** Base deadline of a final (ms). */
export const FINAL_TIMEOUT_MS = 30_000;
/** Extra final deadline per character (ms), as the Rust LibreTranslate timeout adds (translate.rs). */
export const FINAL_TIMEOUT_MS_PER_CHAR = 10;
/** Upper bound of a final's deadline (ms): 10 s above Rust's 60 s cap for one LibreTranslate call. */
export const FINAL_TIMEOUT_MAX_MS = 70_000;

/**
 * Deadline for a final of `text`: 30 s plus 10 ms per character, capped at 70 s. It stays 10 s above
 * the backend's own per-request limit (LibreTranslate: 20 s + 10 ms/char up to 60 s; Azure: 20 s), so
 * a long pasted paragraph on a slow CPU server is not cut off by the sequencer first.
 */
export function finalDeadlineMs(text: string): number {
  const chars = [...text].length;
  return Math.min(FINAL_TIMEOUT_MAX_MS, FINAL_TIMEOUT_MS + FINAL_TIMEOUT_MS_PER_CHAR * chars);
}

type Deadline<E> = number | ((event: E) => number);

/**
 * Comparison form of a line for the late-interim check: lower case, apostrophes dropped ("I'm" =
 * "Im"), other punctuation and symbols turned into spaces, whitespace collapsed. Cloud STT finals add
 * commas and full stops that their interims lack ("yes i think" vs "Yes, I think so.").
 */
export function normalizeForMatch(s: string): string {
  return s
    .normalize("NFKC")
    .toLowerCase()
    .replace(/['‘’ʼ]/g, "")
    .replace(/[\p{P}\p{S}]+/gu, " ")
    .replace(/\s+/g, " ")
    .trim();
}

export class TranslationSequencer<E extends { value: string }> {
  #streams = new Map<string, StreamState<E>>();
  #controllers = new Set<AbortController>();
  #epoch = 0;
  readonly #timeouts: Record<SequenceKind, Deadline<E>>;
  readonly #staleMs: number;
  readonly #now: () => number;

  constructor(private readonly options: TranslationSequencerOptions<E>) {
    this.#timeouts = {
      interim: options.timeoutMs?.interim ?? INTERIM_TIMEOUT_MS,
      final: options.timeoutMs?.final ?? ((event: E) => finalDeadlineMs(event.value)),
    };
    this.#staleMs = options.staleInterimMs ?? 1_500;
    this.#now = options.now ?? (() => Date.now());
  }

  /** Utterances whose output is not settled yet (all streams). */
  get pendingUtterances(): number {
    let n = 0;
    for (const s of this.#streams.values()) n += s.queue.length;
    return n;
  }

  /** Jobs currently running (all streams). */
  get runningJobs(): number {
    return this.#controllers.size;
  }

  pushInterim(stream: string, event: E): void {
    const s = this.#stream(stream);
    if (!s.open && this.#isLateInterim(s, event)) return;
    const u = s.open ?? this.#openUtterance(s);
    const revision = ++u.revision;
    if (u.inFlight) {
      u.queued = { revision, event };
      return;
    }
    this.#dispatchInterim(stream, s, u, revision, event);
  }

  pushFinal(stream: string, event: E): void {
    const s = this.#stream(stream);
    const u = s.open ?? this.#openUtterance(s);
    this.#close(s, u, event);
    const controller = this.#newController();
    const job = this.#job(stream, u, ++u.revision, "final", event, controller);
    u.final = "pending";
    const epoch = this.#epoch;
    this.#execute(job, controller).then(
      (value) => {
        if (epoch !== this.#epoch) return;
        u.final = "ready";
        u.finalOut = { job, value };
        this.#flush(s);
      },
      (error: unknown) => {
        if (epoch !== this.#epoch) return;
        u.final = "dropped";
        this.options.onError?.(job, error);
        this.#flush(s);
      },
    );
  }

  /**
   * Closes the open utterance of `stream` because its sentence was cancelled. If anything was already
   * published for it, `marker` is published as its final so consumers replace the partial; otherwise
   * it closes silently. Nothing happens when no utterance is open.
   */
  cancel(stream: string, event: E, marker: string): void {
    const s = this.#streams.get(stream);
    const u = s?.open;
    if (!s || !u) return;
    this.#close(s, u, event);
    if (u.shown >= 0) {
      const controller = new AbortController();
      u.final = "ready";
      u.finalOut = { job: this.#job(stream, u, ++u.revision, "final", event, controller), value: marker };
    } else {
      u.final = "dropped";
    }
    this.#flush(s);
  }

  /** Aborts every running job and forgets all streams; late results are ignored. */
  reset(): void {
    this.#epoch++;
    const cancelled = new SequencerCancelled();
    for (const c of this.#controllers) c.abort(cancelled);
    this.#controllers.clear();
    this.#streams.clear();
  }

  #stream(stream: string): StreamState<E> {
    let s = this.#streams.get(stream);
    if (!s) {
      s = { nextUtterance: 0, queue: [] };
      this.#streams.set(stream, s);
    }
    return s;
  }

  #openUtterance(s: StreamState<E>): Utterance<E> {
    const u: Utterance<E> = { id: s.nextUtterance++, revision: -1, shown: -1, final: "open" };
    s.open = u;
    s.queue.push(u);
    return u;
  }

  #isLateInterim(s: StreamState<E>, event: E): boolean {
    const last = s.lastClosed;
    if (!last || this.#staleMs <= 0 || this.#now() - last.at > this.#staleMs) return false;
    const text = normalizeForMatch(event.value);
    return text.length > 0 && normalizeForMatch(last.text).startsWith(text);
  }

  /** Ends interim work of `u`; the caller sets its final state. */
  #close(s: StreamState<E>, u: Utterance<E>, event: E): void {
    if (s.open === u) s.open = undefined;
    u.inFlight?.controller.abort(new SequencerCancelled());
    u.inFlight = undefined;
    u.queued = undefined;
    u.held = undefined;
    s.lastClosed = { text: event.value, at: this.#now() };
  }

  #dispatchInterim(stream: string, s: StreamState<E>, u: Utterance<E>, revision: number, event: E): void {
    const controller = this.#newController();
    const job = this.#job(stream, u, revision, "interim", event, controller);
    u.inFlight = { revision, controller };
    const epoch = this.#epoch;
    const settle = (ok: boolean, valueOrError: unknown) => {
      if (epoch !== this.#epoch) return;
      if (u.inFlight?.controller === controller) u.inFlight = undefined;
      if (u.final === "open") {
        if (ok) this.#offerInterim(s, u, job, valueOrError as string);
        else if (!(controller.signal.reason instanceof SequencerCancelled)) this.options.onError?.(job, valueOrError);
        const next = u.queued;
        if (next && !u.inFlight) {
          u.queued = undefined;
          this.#dispatchInterim(stream, s, u, next.revision, next.event);
        }
      }
    };
    this.#execute(job, controller).then(
      (value) => settle(true, value),
      (error: unknown) => settle(false, error),
    );
  }

  #offerInterim(s: StreamState<E>, u: Utterance<E>, job: SequencerJob<E>, value: string): void {
    if (job.revision <= u.shown) return;
    if (s.queue[0] === u) {
      u.shown = job.revision;
      this.options.publish(job, value);
    } else if (!u.held || u.held.job.revision < job.revision) {
      u.held = { job, value };
    }
  }

  /** Publishes everything that became publishable at the head of the stream. */
  #flush(s: StreamState<E>): void {
    while (s.queue.length > 0) {
      const head = s.queue[0];
      if (head.final === "ready") {
        s.queue.shift();
        const out = head.finalOut!;
        this.options.publish(out.job, out.value);
        continue;
      }
      if (head.final === "dropped") {
        s.queue.shift();
        continue;
      }
      if (head.final === "open" && head.held) {
        const held = head.held;
        head.held = undefined;
        this.#offerInterim(s, head, held.job, held.value);
      }
      break;
    }
  }

  #newController(): AbortController {
    const c = new AbortController();
    this.#controllers.add(c);
    return c;
  }

  #job(
    stream: string,
    u: Utterance<E>,
    revision: number,
    kind: SequenceKind,
    event: E,
    controller: AbortController,
  ): SequencerJob<E> {
    return { stream, utterance: u.id, revision, kind, event, signal: controller.signal };
  }

  /** Runs `job` under its deadline; settles exactly once, on the first of result / abort / timeout. */
  #execute(job: SequencerJob<E>, controller: AbortController): Promise<string> {
    const deadline = this.#timeouts[job.kind];
    const timeoutMs = typeof deadline === "function" ? deadline(job.event) : deadline;
    return new Promise<string>((resolve, reject) => {
      let done = false;
      const finish = (settle: () => void) => {
        if (done) return;
        done = true;
        clearTimeout(timer);
        controller.signal.removeEventListener("abort", onAbort);
        this.#controllers.delete(controller);
        settle();
      };
      const onAbort = () => finish(() => reject(controller.signal.reason));
      const timer = setTimeout(() => controller.abort(new SequencerTimeoutError(timeoutMs)), timeoutMs);
      if (controller.signal.aborted) {
        onAbort();
        return;
      }
      controller.signal.addEventListener("abort", onAbort, { once: true });
      let result: Promise<string>;
      try {
        result = Promise.resolve(this.options.run(job));
      } catch (e) {
        result = Promise.reject(e);
      }
      result.then(
        (value) => finish(() => resolve(value)),
        (error: unknown) => finish(() => reject(error)),
      );
    });
  }
}
