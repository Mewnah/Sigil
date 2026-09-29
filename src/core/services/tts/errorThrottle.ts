/**
 * Rate limit for repeated TTS error toasts. When a server is down, every caption line fails the same
 * way, and one toast per line would bury the UI. Every error still goes to the system log.
 */
export class ErrorThrottle {
  #lastShownAt = Number.NEGATIVE_INFINITY;
  #shownAt = new Map<string, number>();

  /**
   * @param repeatMs how long the same message stays suppressed after it was shown
   * @param minGapMs minimum time between any two toasts
   * @param now clock (injectable for tests)
   */
  constructor(
    private readonly repeatMs = 15_000,
    private readonly minGapMs = 3_000,
    private readonly now: () => number = () => Date.now(),
  ) {}

  /** Whether to show `message` now; records it as shown when it returns true. */
  shouldShow(message: string): boolean {
    const t = this.now();
    if (t - this.#lastShownAt < this.minGapMs) return false;
    const last = this.#shownAt.get(message);
    if (last !== undefined && t - last < this.repeatMs) return false;

    this.#lastShownAt = t;
    this.#shownAt.set(message, t);
    if (this.#shownAt.size > 64) {
      for (const [m, at] of this.#shownAt) {
        if (t - at >= this.repeatMs) this.#shownAt.delete(m);
      }
    }
    return true;
  }

  /** Forget history, e.g. when the service restarts, so the first new error is shown. */
  reset(): void {
    this.#lastShownAt = Number.NEGATIVE_INFINITY;
    this.#shownAt.clear();
  }
}
