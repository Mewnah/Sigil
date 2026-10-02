/**
 * Smart Turn commit gate (P6-C): sends a sentence on only once the speaker finished their turn.
 *
 * The local speech pipeline marks each final with Smart Turn's verdict (`TextEvent.turnEnd`): `false` means the
 * speaker paused mid-thought. With the gate on, such finals are held and joined with what follows; the joined text is
 * committed by the next final whose verdict is `true` or unknown, after `holdMs` without any speech (interims re-arm
 * the wait), or once it grows past `maxChars`. Finals without a verdict (Smart Turn off, cloud STT, typed text) pass
 * straight through, so the gate never delays them. Captions keep streaming from partials; only one-shot outputs
 * (the chatbox, and text-to-speech in `Service_TTS`, wired by P6-I0) wait for the turn.
 *
 * STT's cancel marker (`[...]`, a sentence cut off by mute or stop) is never committed: it only releases what is held.
 *
 * Pure logic: no `window`, no Tauri. Generic over the consumer, so text-to-speech can reuse it.
 */
import { TextEvent, TextEventType } from "@/types";
import { isCancelMarker } from "@/core/services/translation/sequencer";
import { joinFragments, MAX_HELD_CHARS, TURN_HOLD_MS } from "@/core/services/translation/tracker";

export interface TurnCommitClock {
  /** Monotonic milliseconds (`performance.now()`), never the wall clock. */
  now(): number;
  setTimeout(fn: () => void, ms: number): unknown;
  clearTimeout(handle: unknown): void;
}

const systemClock: TurnCommitClock = {
  now: () => (typeof performance !== "undefined" ? performance.now() : Date.now()),
  setTimeout: (fn, ms) => setTimeout(fn, ms),
  clearTimeout: (h) => clearTimeout(h as ReturnType<typeof setTimeout>),
};

export interface TurnCommitOptions {
  /** Hold unfinished turns (else every final is committed at once). Read on every event. */
  enabled(): boolean;
  /** Longest wait for the turn to end after the last speech event (ms; default 2500, like translation). */
  holdMs?(): number;
  /** Held text is committed once it is this long (characters; default 280). */
  maxChars?(): number;
  /** A committed line: `value` is the joined text; `segmentId` that of its first final; `tags` merged. */
  commit(event: TextEvent): void;
  clock?: TurnCommitClock;
}

/** Most tags kept on a joined line (`TextEventSchema` allows 8). */
const MAX_TAGS = 8;

export class TurnCommitGate {
  readonly #options: TurnCommitOptions;
  readonly #clock: TurnCommitClock;
  #held: TextEvent[] = [];
  #timer: unknown = null;

  constructor(options: TurnCommitOptions) {
    this.#options = options;
    this.#clock = options.clock ?? systemClock;
  }

  /** Whether finals are waiting for the end of the turn. */
  get holding(): boolean {
    return this.#held.length > 0;
  }

  /** The held text so far ("" when nothing is held). */
  get heldText(): string {
    return this.#join(this.#held);
  }

  /** A final line of the source. */
  final(event: TextEvent): void {
    const value = (event.value ?? "").trim();
    if (!value) return;
    if (isCancelMarker(value)) {
      this.flush();
      return;
    }
    const ev: TextEvent = { ...event, type: TextEventType.final, value };
    if (!this.#options.enabled()) {
      this.flush();
      this.#options.commit(ev);
      return;
    }
    const max = this.#maxChars();
    // Joined with this final the held text would be too long: what is held goes first, as its own line.
    if (this.#held.length > 0 && this.#join([...this.#held, ev]).length > max) this.flush();
    this.#held.push(ev);
    if (ev.turnEnd === false && this.#holdMs() > 0 && this.#join(this.#held).length < max) {
      this.#arm();
      return;
    }
    this.flush();
  }

  /** An interim line: the speaker is still talking, so a held turn keeps waiting. */
  interim(event: TextEvent): void {
    if (this.#held.length === 0) return;
    const value = (event.value ?? "").trim();
    if (!value || isCancelMarker(value)) return;
    this.#arm();
  }

  /** A complete message from elsewhere (typed text): what is held goes first, then it, without waiting. */
  passThrough(event: TextEvent): void {
    const value = (event.value ?? "").trim();
    if (!value || isCancelMarker(value)) return;
    this.flush();
    this.#options.commit({ ...event, type: TextEventType.final, value });
  }

  /** Commits what is held now. */
  flush(): void {
    this.#disarm();
    if (this.#held.length === 0) return;
    const held = this.#held;
    this.#held = [];
    const last = held[held.length - 1];
    const tags = [...new Set(held.flatMap((e) => e.tags ?? []))].slice(0, MAX_TAGS);
    const out: TextEvent = { ...last, type: TextEventType.final, value: this.#join(held) };
    const first = held[0].segmentId;
    if (first !== undefined) out.segmentId = first;
    else delete out.segmentId;
    if (tags.length > 0) out.tags = tags;
    else delete out.tags;
    this.#options.commit(out);
  }

  /** Drops what is held without committing it (target disabled or disposed). */
  reset(): void {
    this.#disarm();
    this.#held = [];
  }

  #join(events: TextEvent[]): string {
    let out = "";
    for (const e of events) out = joinFragments(out, e.value, e.lang ?? "");
    return out;
  }

  #holdMs(): number {
    const ms = this.#options.holdMs?.() ?? TURN_HOLD_MS;
    return Number.isFinite(ms) ? Math.max(0, ms) : TURN_HOLD_MS;
  }

  #maxChars(): number {
    const n = this.#options.maxChars?.() ?? MAX_HELD_CHARS;
    return Number.isFinite(n) && n > 0 ? n : MAX_HELD_CHARS;
  }

  #arm(): void {
    this.#disarm();
    this.#timer = this.#clock.setTimeout(() => {
      this.#timer = null;
      this.flush();
    }, this.#holdMs());
  }

  #disarm(): void {
    if (this.#timer !== null) {
      this.#clock.clearTimeout(this.#timer);
      this.#timer = null;
    }
  }
}
