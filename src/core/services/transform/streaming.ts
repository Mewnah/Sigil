/**
 * Rate limit for streamed partial replies: a model streams a token every 10–50 ms, and every publish fans out to
 * PeerJS clients, the LAN WebSocket and OBS. The first partial goes out at once, then at most one per `intervalMs`,
 * always ending with the latest text (a trailing publish) unless the line finished first (`close`).
 */
export class PartialThrottle {
  #lastAt = Number.NEGATIVE_INFINITY;
  #lastText = "";
  #pending: string | null = null;
  #timer: ReturnType<typeof setTimeout> | undefined;
  #closed = false;

  constructor(
    private readonly publish: (text: string) => void,
    private readonly intervalMs = 100,
    private readonly now: () => number = () => Date.now(),
  ) {}

  push(text: string): void {
    if (this.#closed || !text || text === this.#lastText) return;
    const wait = this.#lastAt + this.intervalMs - this.now();
    if (wait <= 0 && this.#timer === undefined) {
      this.#emit(text);
      return;
    }
    this.#pending = text;
    if (this.#timer === undefined) {
      this.#timer = setTimeout(() => {
        this.#timer = undefined;
        const next = this.#pending;
        this.#pending = null;
        if (next !== null && !this.#closed && next !== this.#lastText) this.#emit(next);
      }, Math.max(0, wait));
    }
  }

  /** The line is done (final, failure or stop): nothing more is published. */
  close(): void {
    this.#closed = true;
    this.#pending = null;
    if (this.#timer !== undefined) clearTimeout(this.#timer);
    this.#timer = undefined;
  }

  #emit(text: string) {
    this.#lastAt = this.now();
    this.#lastText = text;
    try {
      this.publish(text);
    } catch (e) {
      console.error("[Transform] partial publish failed:", e);
    }
  }
}
