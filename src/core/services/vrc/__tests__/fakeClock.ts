/** Manual clock for the VRChat pacing / turn / tag timers (tests only). */
export class FakeClock {
  #now = 0;
  #next = 1;
  #timers = new Map<number, { at: number; fn: () => void }>();

  now = (): number => this.#now;

  setTimeout = (fn: () => void, ms: number): unknown => {
    const id = this.#next++;
    this.#timers.set(id, { at: this.#now + Math.max(0, ms), fn });
    return id;
  };

  clearTimeout = (handle: unknown): void => {
    this.#timers.delete(handle as number);
  };

  /** Timers waiting. */
  get pending(): number {
    return this.#timers.size;
  }

  /** Advances time by `ms`, firing due timers in order (timers they arm fire too when due). */
  advance(ms: number): void {
    const end = this.#now + ms;
    for (;;) {
      let due: [number, { at: number; fn: () => void }] | undefined;
      for (const entry of this.#timers) {
        if (entry[1].at <= end && (!due || entry[1].at < due[1].at)) due = entry;
      }
      if (!due) break;
      this.#timers.delete(due[0]);
      this.#now = Math.max(this.#now, due[1].at);
      due[1].fn();
    }
    this.#now = end;
  }
}
