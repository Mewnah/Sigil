/**
 * TTS sentence pipeline (P2-G): splits each final into sentences ({@link splitSentences}), synthesizes them in
 * order a few sentences ahead of playback, and plays them through the unified Rust TTS sink, with stop and skip.
 *
 * - Sentence 1 is synthesized alone first, so audio starts as soon as it is ready (streaming backends start after
 *   the sink's 200 ms prebuffer); sentence k+1 synthesizes while sentence k plays. Synthesis runs one sentence at a
 *   time (local engines share one CPU/GPU; servers queue anyway) and at most `ahead` sentences ahead of the one
 *   playing.
 * - Each sentence is one sink utterance. Begins, ends and cancels go through one ordered control chain, so the sink
 *   sees them in sentence order and a cancel can never overtake its begin; playback order is the begin order,
 *   never the order synthesis happened to finish.
 * - Backends implement ITTSService v2 (`types.ts`): `synthesizeToSink` (Rust engines push into the begun
 *   utterance themselves) or `synthesizePcm` (chunks pushed here over binary IPC). Self-playing backends (Web
 *   Speech, SAPI) do not use the pipeline.
 * - Skip cancels the sentence playing now; stop cancels everything queued, synthesizing and playing. Cancelling
 *   aborts the sentence's `signal`; aborted synthesis never reports an error, and the next sentence may start
 *   synthesizing at once (it does not wait for the aborted call to settle). A failed sentence is reported through
 *   `onError` and skipped; the next one plays. A sentence the sink dropped by itself (stall, output failure) is
 *   reported once, by its terminal sink event.
 * - Utterance events from the sink drive the state (playing / done / cancelled / error). A missed event cannot
 *   wedge the pipeline: while sentences are waiting on playback, the sink's status is checked periodically and
 *   sentences it no longer knows are dropped.
 */
import type { RustTtsSinkStatus, RustTtsSinkTarget, RustTtsSinkUtteranceEvent } from "@/types";
import { isSinkCancelled, type SinkPcm } from "../sound/sink";
import { clampedOr } from "./num";
import type { TTS_State } from "./schema";
import { splitSentences, type SentenceOptions } from "./sentence";
import { ITTSService, TtsSinkContext, errorMessage, isAbortError, ttsOutputMode } from "./types";

/** The sink calls the pipeline needs (`TtsSinkClient` in the app; a fake in tests). */
export interface TtsSinkApi {
  ready(): Promise<void>;
  begin(id: string, target: RustTtsSinkTarget): Promise<void>;
  push(id: string, pcm: SinkPcm): Promise<void>;
  end(id: string): Promise<void>;
  cancel(id: string | null): Promise<void>;
  status(): Promise<RustTtsSinkStatus>;
  onUtterance(handler: (event: RustTtsSinkUtteranceEvent) => void): () => void;
}

/** What the UI shows (Service_TTS mirrors it into its serviceState). */
export interface TtsPipelineState {
  /** Sentences queued, synthesizing or playing. */
  pending: number;
  /** A sentence is audible right now. */
  playing: boolean;
}

export interface TtsPipelineOptions {
  sink: TtsSinkApi;
  /** Route, device and volume for the sentences of one `speak()` call (read at that time). */
  target: () => RustTtsSinkTarget;
  /** One sentence failed (synthesis or audio output). */
  onError: (message: string) => void;
  onChange?: (state: TtsPipelineState) => void;
  /** Sentences synthesized ahead of the one playing (default 2). */
  ahead?: number;
  sentences?: Partial<SentenceOptions>;
  /** Period of the missed-event check (ms). */
  reconcileMs?: number;
}

type JobState = "waiting" | "synthesizing" | "synthesized" | "playing";

interface Job {
  id: string;
  text: string;
  target: RustTtsSinkTarget;
  /** Language tag of the text this sentence came from (`TtsSinkContext.lang`); undefined = untagged. */
  lang: string | undefined;
  state: JobState;
  controller: AbortController;
  /** `begin_utterance` completed. */
  begun: boolean;
  cancelled: boolean;
  /** No longer tracked (done, cancelled or failed). */
  finished: boolean;
}

const DEFAULT_AHEAD = 2;
const DEFAULT_RECONCILE_MS = 3_000;

/**
 * Sink target for the active backend: the top-level route, the backend's `device` (else the app's output device,
 * else the system default) and its `volume` (0..2) as the sink gain when the backend has one.
 */
export function ttsSinkTarget(data: TTS_State, appOutputDevice: string): RustTtsSinkTarget {
  const slice = (data as unknown as Record<string, unknown>)[data.backend];
  const rec = slice && typeof slice === "object" ? (slice as { device?: unknown; volume?: unknown }) : {};
  const device = typeof rec.device === "string" ? rec.device.trim() : "";
  const volume = "volume" in rec ? clampedOr(rec.volume, 1, 0, 2) : 1;
  return { route: data.route, device: device || (appOutputDevice ?? "").trim(), volume };
}

export class TtsPipeline {
  readonly mode: "sink" | "pcm";
  #jobs: Job[] = [];
  #control: Promise<void> = Promise.resolve();
  /**
   * The sentence being synthesized (one at a time). Released as soon as that sentence is cancelled or finished, not
   * when its synthesis call settles: a backend slow to notice the abort must not hold up the next sentence.
   */
  #busy: Job | null = null;
  #disposed = false;
  #seq = 0;
  readonly #session = Math.random().toString(36).slice(2, 8);
  readonly #unsubscribe: () => void;
  #reconcileTimer: ReturnType<typeof setInterval> | null = null;
  #lastState = "";

  constructor(
    private readonly service: ITTSService,
    private readonly opts: TtsPipelineOptions,
  ) {
    const mode = ttsOutputMode(service);
    if (mode === "selfPlaying") throw new Error("this backend plays by itself and cannot use the sentence pipeline");
    this.mode = mode;
    this.#unsubscribe = opts.sink.onUtterance((event) => this.#onEvent(event));
  }

  get state(): TtsPipelineState {
    return { pending: this.#jobs.length, playing: this.#jobs.some((j) => j.state === "playing") };
  }

  /**
   * Split `text` into sentences and queue them after everything already queued. `lang` is the text's language tag
   * (the TextEvent's `lang`), passed to sink engines with each sentence. Returns how many.
   */
  speak(text: string, lang?: string | null): number {
    if (this.#disposed) return 0;
    const sentences = splitSentences(text, this.opts.sentences);
    if (!sentences.length) return 0;
    const target = this.opts.target();
    const tag = typeof lang === "string" && lang.trim() ? lang.trim() : undefined;
    for (const sentence of sentences) {
      this.#jobs.push({
        id: `tts-${this.#session}-${this.#seq++}`,
        text: sentence,
        target: { ...target },
        lang: tag,
        state: "waiting",
        controller: new AbortController(),
        begun: false,
        cancelled: false,
        finished: false,
      });
    }
    this.#changed();
    this.#pump();
    return sentences.length;
  }

  /** Skip the sentence playing now (else the first one still being prepared); the next one follows. */
  skip(): void {
    const job = this.#jobs.find((j) => j.state === "playing") ?? this.#jobs[0];
    if (job) this.#cancel(job);
  }

  /** Stop speaking: cancel every queued, synthesizing and playing sentence. The pipeline stays usable. */
  stop(): void {
    for (const job of [...this.#jobs]) this.#cancel(job);
  }

  /** Stop and release (event subscription, timers). */
  dispose(): void {
    this.stop();
    this.#disposed = true;
    this.#unsubscribe();
    this.#stopReconcile();
  }

  #changed() {
    const state = this.state;
    const key = `${state.pending}|${state.playing}`;
    if (key === this.#lastState) return;
    this.#lastState = key;
    this.opts.onChange?.(state);
  }

  /** Run `fn` after every earlier control call (begin / end / cancel keep sentence order in the sink). */
  #controlCall(fn: () => Promise<void>): Promise<void> {
    const run = this.#control.then(fn);
    this.#control = run.catch(() => {});
    return run;
  }

  #pump() {
    if (this.#disposed || this.#busy) return;
    const ahead = Math.max(0, this.opts.ahead ?? DEFAULT_AHEAD);
    const inFlight = this.#jobs.filter((j) => j.state !== "waiting").length;
    if (inFlight > ahead) return;
    const next = this.#jobs.find((j) => j.state === "waiting");
    if (!next) return;
    next.state = "synthesizing";
    this.#busy = next;
    void this.#run(next).finally(() => {
      if (this.#busy === next) this.#busy = null;
      this.#pump();
    });
  }

  async #run(job: Job): Promise<void> {
    const { sink } = this.opts;
    const signal = job.controller.signal;
    try {
      await this.#controlCall(async () => {
        if (job.cancelled) return;
        await sink.ready();
        await sink.begin(job.id, job.target);
        job.begun = true;
      });
    } catch (error) {
      if (!job.cancelled) this.#fail(job, error);
      return;
    }
    if (job.cancelled || !job.begun) return;
    try {
      if (this.mode === "sink") {
        await this.service.synthesizeToSink!(
          job.text,
          { utteranceId: job.id, target: job.target, lang: job.lang },
          signal,
        );
      } else {
        for await (const chunk of this.service.synthesizePcm!(job.text, signal)) {
          if (signal.aborted) break;
          if (!chunk.samples.length) continue;
          await sink.push(job.id, chunk);
        }
      }
      if (job.cancelled || signal.aborted) return;
      if (job.state === "synthesizing") job.state = "synthesized";
      await this.#controlCall(() => sink.end(job.id));
      this.#ensureReconcile();
    } catch (error) {
      if (job.cancelled || signal.aborted || isAbortError(error)) return;
      if (isSinkCancelled(error)) {
        // The sink dropped it (its stall timeout, an output error): the sink reports that with the utterance's
        // terminal event, which may arrive after this rejection. Keep the sentence until that event (or the
        // reconcile pass, should it be lost) so the error is reported exactly once. Should the utterance still be
        // alive (an engine that gave up by itself), cancelling it ends it with a `cancelled` event; for one the sink
        // already dropped the cancel is a no-op.
        if (job.state === "synthesizing") job.state = "synthesized";
        void this.#controlCall(() => sink.cancel(job.id)).catch(() => {});
        this.#ensureReconcile();
        return;
      }
      this.#fail(job, error);
    }
  }

  #fail(job: Job, error: unknown) {
    if (job.finished) return;
    this.opts.onError(errorMessage(error));
    this.#cancel(job);
  }

  #cancel(job: Job) {
    if (job.finished) return;
    job.cancelled = true;
    job.controller.abort();
    // Chained after a begin still in flight; nothing to cancel when that begin was skipped.
    void this.#controlCall(async () => {
      if (job.begun) await this.opts.sink.cancel(job.id);
    }).catch(() => {});
    this.#finish(job);
  }

  #finish(job: Job) {
    if (job.finished) return;
    job.finished = true;
    job.controller.abort();
    // Its synthesis (if still running) settles in the background; the next sentence may start now.
    if (this.#busy === job) this.#busy = null;
    this.#jobs = this.#jobs.filter((j) => j !== job);
    if (!this.#jobs.length) this.#stopReconcile();
    this.#changed();
    this.#pump();
  }

  #onEvent(event: RustTtsSinkUtteranceEvent) {
    const job = this.#jobs.find((j) => j.id === event.id);
    if (!job) return;
    switch (event.state) {
      case "playing":
        if (job.state !== "waiting") {
          job.state = "playing";
          this.#changed();
        }
        break;
      case "done":
      case "cancelled":
        this.#finish(job);
        break;
      case "error":
        job.cancelled = true;
        this.opts.onError(event.error || "audio output failed");
        this.#finish(job);
        break;
      default:
        break;
    }
  }

  #ensureReconcile() {
    if (this.#reconcileTimer || this.#disposed) return;
    this.#reconcileTimer = setInterval(() => void this.#reconcile(), this.opts.reconcileMs ?? DEFAULT_RECONCILE_MS);
  }

  #stopReconcile() {
    if (this.#reconcileTimer) clearInterval(this.#reconcileTimer);
    this.#reconcileTimer = null;
  }

  /** Drop sentences waiting on playback that the sink no longer knows (a missed done event). */
  async #reconcile(): Promise<void> {
    const candidates = this.#jobs.filter((j) => j.begun && (j.state === "synthesized" || j.state === "playing"));
    if (!candidates.length) return;
    let status: RustTtsSinkStatus;
    try {
      status = await this.opts.sink.status();
    } catch {
      return;
    }
    const known = new Set<string>(status.queued);
    if (status.playing) known.add(status.playing);
    for (const job of candidates) {
      if (!job.finished && !known.has(job.id)) this.#finish(job);
    }
  }
}

// -------------------------------------------------------------------------------------------------
// Fallback chain (P6-H): a cloud voice with the local voice behind it
// -------------------------------------------------------------------------------------------------

/**
 * Why a cloud TTS request failed. The Rust cloud plugins (`edge-tts`, `gemini-tts`) reject with a tag in square
 * brackets, `cloud-tts:` plus the kind, optionally followed by `:partial` and `:retry=` plus milliseconds, then the
 * message (Rust `edge_tts::cloud`; parsed by `FAILURE_TAG` below). Spelled out in words because Tailwind's class
 * scanner reads the bracketed form as an arbitrary property and breaks `pnpm build`.
 */
export type TtsFailureKind =
  | "network"
  | "rate_limited"
  | "unavailable"
  | "stalled"
  | "auth"
  | "rejected"
  | "no_audio"
  | "other";

export const TTS_FAILURE_KINDS: readonly TtsFailureKind[] = [
  "network",
  "rate_limited",
  "unavailable",
  "stalled",
  "auth",
  "rejected",
  "no_audio",
  "other",
];

/** A parsed cloud failure. */
export interface TtsFailure {
  kind: TtsFailureKind;
  /** Audio of the sentence was already played (or queued) before it failed. */
  partial: boolean;
  /** When the service said to try again (ms), if it did. */
  retryAfterMs: number | null;
  /** The readable message (without the tag). */
  message: string;
  /** The error carried the cloud tag (an untagged error is not the service's: sink, IPC). */
  tagged: boolean;
}

const FAILURE_TAG = /^\s*\[cloud-tts:([a-z_]+)((?::[a-z]+(?:=\d+)?)*)\]\s*/;

/** Parse a cloud plugin's error (see {@link TtsFailure}); anything else is an untagged `other`. */
export function parseTtsFailure(error: unknown): TtsFailure {
  const text = errorMessage(error);
  const m = FAILURE_TAG.exec(text);
  if (!m) return { kind: "other", partial: false, retryAfterMs: null, message: text, tagged: false };
  const kind = (TTS_FAILURE_KINDS as readonly string[]).includes(m[1]) ? (m[1] as TtsFailureKind) : "other";
  let partial = false;
  let retryAfterMs: number | null = null;
  for (const flag of m[2].split(":").filter(Boolean)) {
    if (flag === "partial") partial = true;
    else if (flag.startsWith("retry=")) {
      const ms = Number(flag.slice("retry=".length));
      if (Number.isFinite(ms) && ms >= 0) retryAfterMs = ms;
    }
  }
  return { kind, partial, retryAfterMs, message: text.slice(m[0].length).trim() || text, tagged: true };
}

/** How long the fallback chain bypasses a failing service (ms). */
export interface FallbackBreakerOptions {
  /** First pause after a network failure, 5xx or stall; doubles with each failure in a row. */
  baseOpenMs: number;
  /** Longest such pause. */
  maxOpenMs: number;
  /** Pause after a 429 without a retry hint. */
  rateLimitMs: number;
  /** Pause after a refused key / voice / model (settings problems), and the cap of any retry hint. */
  hardOpenMs: number;
}

export const DEFAULT_FALLBACK_BREAKER: FallbackBreakerOptions = {
  baseOpenMs: 15_000,
  maxOpenMs: 5 * 60_000,
  rateLimitMs: 60_000,
  hardOpenMs: 10 * 60_000,
};

/** What a failure did to the breaker. */
export interface BreakerVerdict {
  /** The service was healthy until now: tell the user once. */
  opened: boolean;
  /** Sentences go to the fallback until then (epoch ms); 0 = not open (a failure of one sentence only). */
  openUntil: number;
}

/**
 * Circuit breaker of the fallback chain (pure; the clock is injectable). While open, sentences go straight to the
 * fallback, so an outage costs nothing after its first sentence. When the pause ends, the next sentence tries the
 * service again (half-open): success closes the breaker, another failure re-opens it for longer.
 *
 * - Network failures, 5xx, stalls and unknown failures: 15 s, then 30 s, 60 s ... up to 5 min (or the service's retry
 *   hint when longer).
 * - 429 / quota: the service's retry hint, else 60 s.
 * - A refused key, voice or model: 10 min (the settings must change; restarting the backend resets it).
 * - A sentence the service could not say (`no_audio`): that sentence only.
 */
export class FallbackBreaker {
  #openUntil = 0;
  #failures = 0;
  #healthy = true;

  constructor(
    private readonly opts: FallbackBreakerOptions = DEFAULT_FALLBACK_BREAKER,
    private readonly now: () => number = Date.now,
  ) {}

  /** Where the next sentence goes. */
  route(): "primary" | "fallback" {
    return this.now() < this.#openUntil ? "fallback" : "primary";
  }

  /** The service failed at least once and has not succeeded since. */
  get degraded(): boolean {
    return !this.#healthy;
  }

  get openUntil(): number {
    return this.#openUntil;
  }

  /** The service spoke a sentence. Returns true when it recovered from failures. */
  success(): boolean {
    const recovered = !this.#healthy;
    this.#healthy = true;
    this.#failures = 0;
    this.#openUntil = 0;
    return recovered;
  }

  failure(f: Pick<TtsFailure, "kind" | "retryAfterMs">): BreakerVerdict {
    if (f.kind === "no_audio") return { opened: false, openUntil: 0 };
    const o = this.opts;
    const hint = f.retryAfterMs != null ? Math.min(f.retryAfterMs, o.hardOpenMs) : null;
    this.#failures++;
    let pause: number;
    switch (f.kind) {
      case "rate_limited":
        pause = hint ?? o.rateLimitMs;
        break;
      case "auth":
      case "rejected":
        pause = o.hardOpenMs;
        break;
      default:
        pause = Math.max(Math.min(o.baseOpenMs * 2 ** Math.min(this.#failures - 1, 20), o.maxOpenMs), hint ?? 0);
        break;
    }
    const opened = this.#healthy;
    this.#healthy = false;
    this.#openUntil = this.now() + pause;
    return { opened, openUntil: this.#openUntil };
  }

  reset(): void {
    this.#openUntil = 0;
    this.#failures = 0;
    this.#healthy = true;
  }
}

/** Speaks one sentence into a begun sink utterance (ITTSService v2 `synthesizeToSink` shape). */
export type TtsSinkSpeaker = (text: string, sink: TtsSinkContext, signal: AbortSignal) => Promise<void>;

/** A sentence that did not go to the service as planned. */
export interface TtsFallbackEvent {
  /** Why (`null` = the breaker was already open: the service was not asked). */
  failure: TtsFailure | null;
  /** This failure opened the breaker (the first failure after the service worked): tell the user. */
  opened: boolean;
  /** The fallback spoke the sentence (false: part of it already played, or no fallback is available). */
  spokenByFallback: boolean;
}

export interface TtsFallbackChainOptions {
  /** The cloud voice. Rejects with a tagged failure ({@link parseTtsFailure}). */
  primary: TtsSinkSpeaker;
  /**
   * The local voice, or `null` when there is none (switched off, not downloaded, failed to load). May wait for a
   * model that is still loading.
   */
  fallback: () => Promise<TtsSinkSpeaker | null>;
  breaker?: FallbackBreaker;
  /** A sentence failed on the service, or went to the fallback while the breaker is open (see {@link TtsFallbackEvent}). */
  onFallback?: (event: TtsFallbackEvent) => void;
  /** The service spoke again after failing. */
  onRecovered?: () => void;
}

/**
 * The fallback chain of a cloud TTS backend (P6-H): each sentence goes to the service; when the service fails (no
 * connection, 429, 5xx, a stall, a refused request) the same sentence is spoken by the local voice into the same sink
 * utterance at once, and the breaker keeps the next sentences on the local voice until the service is worth trying
 * again. The first audio of a failed sentence therefore comes from the local voice right after the failure (a stall
 * is detected by the plugin's short timeout), and every later sentence of an outage starts on the local voice
 * without waiting.
 *
 * - A failure after part of the sentence already played is not spoken again (that would repeat words); the breaker
 *   still opens, so the next sentence uses the local voice.
 * - Stop / skip (the signal) never fall back. Errors that do not come from the service (the sink, IPC) and failures
 *   without a fallback are thrown, so the pipeline reports them.
 * - With the breaker open and no fallback available, the service is still asked (there is nothing better to do).
 */
export class TtsFallbackChain {
  readonly breaker: FallbackBreaker;

  constructor(private readonly opts: TtsFallbackChainOptions) {
    this.breaker = opts.breaker ?? new FallbackBreaker();
  }

  async synthesize(text: string, sink: TtsSinkContext, signal: AbortSignal): Promise<void> {
    if (signal.aborted) return;
    if (this.breaker.route() === "fallback") {
      const fallback = await this.opts.fallback();
      if (signal.aborted) return;
      if (fallback) {
        this.opts.onFallback?.({ failure: null, opened: false, spokenByFallback: true });
        await fallback(text, sink, signal);
        return;
      }
    }
    try {
      await this.opts.primary(text, sink, signal);
    } catch (error) {
      if (signal.aborted || isAbortError(error) || isSinkCancelled(error)) throw error;
      const failure = parseTtsFailure(error);
      if (!failure.tagged) throw error;
      const { opened } = this.breaker.failure(failure);
      if (failure.partial) {
        this.opts.onFallback?.({ failure, opened, spokenByFallback: false });
        return;
      }
      const fallback = await this.opts.fallback();
      if (signal.aborted) return;
      if (!fallback) {
        this.opts.onFallback?.({ failure, opened, spokenByFallback: false });
        throw new Error(failure.message);
      }
      this.opts.onFallback?.({ failure, opened, spokenByFallback: true });
      await fallback(text, sink, signal);
      return;
    }
    if (!signal.aborted && this.breaker.success()) this.opts.onRecovered?.();
  }
}
