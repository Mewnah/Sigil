/**
 * Delivers results in the order their slots were reserved, whatever order they complete in. AI
 * replies take anywhere from 0.3 to 15 s, so sentence 2 can finish before sentence 1; publishing in
 * completion order made TTS speak and the chatbox post lines out of order and scrambled the context
 * history.
 *
 * Every reserved slot must be resolved exactly once (callers guarantee this with a deadline and an
 * abort on stop), otherwise every later slot waits behind it.
 */
export class OrderedQueue<T> {
  #reserved = 0;
  #delivered = 0;
  #ready = new Map<number, T>();
  #draining = false;

  constructor(
    private readonly deliver: (item: T, seq: number) => void,
    private readonly onDeliverError: (error: unknown) => void = (e) => console.error("[OrderedQueue] delivery failed:", e),
  ) {}

  /** Slots reserved but not delivered yet. */
  get pending(): number {
    return this.#reserved - this.#delivered;
  }

  reserve(): number {
    return this.#reserved++;
  }

  /** Resolves `seq`; ignored when it is unknown or already resolved. */
  resolve(seq: number, item: T): void {
    if (!Number.isInteger(seq) || seq < this.#delivered || seq >= this.#reserved || this.#ready.has(seq)) return;
    this.#ready.set(seq, item);
    // A delivery that resolves another slot synchronously is picked up by the loop already running.
    if (this.#draining) return;
    this.#draining = true;
    try {
      while (this.#ready.has(this.#delivered)) {
        const next = this.#delivered++;
        const ready = this.#ready.get(next) as T;
        this.#ready.delete(next);
        try {
          this.deliver(ready, next);
        } catch (e) {
          this.onDeliverError(e);
        }
      }
    } finally {
      this.#draining = false;
    }
  }
}
