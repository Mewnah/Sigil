import { TextEventType, type CaptionEventSource, type TextEvent } from "@/types";
import { CaptionModel, type CaptionModelOptions } from "./captionModel";
import { acceptsTopic, normalizeSources } from "./config";

/**
 * Feeds one caption display's {@link CaptionModel} from the caption topics (P6-B) and runs its clear timer. The pubsub
 * and the clock are injected, so it is tested in node with a fake bus and fake timers.
 */

/** Subscribe / unsubscribe on the caption topics (`Service_PubSub.subscribeText` with empty events). */
export interface CaptionBus {
  subscribe(source: CaptionEventSource, fn: (event: TextEvent | undefined, topic: string | undefined) => void): string;
  unsubscribe(token: string): void;
}

export interface FeedClock {
  now(): number;
  setTimeout(fn: () => void, ms: number): unknown;
  clearTimeout(handle: unknown): void;
}

export const systemClock: FeedClock = {
  now: () => (typeof performance !== "undefined" ? performance.now() : Date.now()),
  setTimeout: (fn, ms) => setTimeout(fn, ms),
  clearTimeout: (h) => clearTimeout(h as ReturnType<typeof setTimeout>),
};

/** How long a test caption stays. */
export const TEST_CAPTION_MS = 6000;

export class CaptionFeed {
  readonly model: CaptionModel;
  readonly #bus: CaptionBus;
  readonly #clock: FeedClock;
  readonly #onChange: () => void;
  #tokens: string[] = [];
  #sourcesKey = "";
  #expiryTimer: unknown = null;
  #active = false;

  constructor(options: CaptionModelOptions, bus: CaptionBus, onChange: () => void, clock: FeedClock = systemClock) {
    this.#bus = bus;
    this.#clock = clock;
    this.#onChange = onChange;
    this.model = new CaptionModel(options, () => clock.now());
  }

  /** Listen to `sources` with `options` (re-subscribing only when the sources changed). */
  configure(sources: readonly CaptionEventSource[], options: CaptionModelOptions): void {
    const normalized = normalizeSources(sources);
    const key = normalized.join(",");
    let changed = this.model.setOptions(options);
    if (!this.#active || key !== this.#sourcesKey) {
      this.#unsubscribeAll();
      // Lines of a source no longer shown must not linger.
      if (this.#active) changed = this.model.clear() || changed;
      this.#active = true;
      this.#sourcesKey = key;
      for (const source of normalized) {
        this.#tokens.push(this.#bus.subscribe(source, (event, topic) => this.#onEvent(source, event, topic)));
      }
    }
    this.#scheduleExpiry();
    if (changed) this.#onChange();
  }

  /** Stop listening and forget every line. */
  stop(): void {
    this.#unsubscribeAll();
    this.#active = false;
    this.#sourcesKey = "";
    this.#clearExpiry();
    if (this.model.clear()) this.#onChange();
  }

  /** Show a test line for {@link TEST_CAPTION_MS} (whatever the clear timer says). */
  showTest(text: string, lang?: string): void {
    const key = this.model.push(text, "test", lang);
    this.#onChange();
    this.#clock.setTimeout(() => {
      if (this.model.removeKey(key)) this.#onChange();
    }, TEST_CAPTION_MS);
  }

  #onEvent(subscribed: CaptionEventSource, event: TextEvent | undefined, topic: string | undefined): void {
    if (!event || typeof event !== "object") return;
    const actual = topic ?? subscribed;
    if (!acceptsTopic(subscribed, actual)) return;
    if (event.type !== TextEventType.final && event.type !== TextEventType.interim) return;
    if (this.model.apply(actual, event)) {
      this.#scheduleExpiry();
      this.#onChange();
    }
  }

  #unsubscribeAll(): void {
    for (const token of this.#tokens) this.#bus.unsubscribe(token);
    this.#tokens = [];
  }

  #clearExpiry(): void {
    if (this.#expiryTimer !== null) this.#clock.clearTimeout(this.#expiryTimer);
    this.#expiryTimer = null;
  }

  #scheduleExpiry(): void {
    this.#clearExpiry();
    const at = this.model.nextExpiry();
    if (at === null) return;
    const delay = Math.max(0, at - this.#clock.now()) + 5;
    this.#expiryTimer = this.#clock.setTimeout(() => {
      this.#expiryTimer = null;
      if (this.model.expire()) this.#onChange();
      this.#scheduleExpiry();
    }, delay);
  }
}
