/**
 * Client-side request pacing for rate-limited cloud STT (Groq's free plan). Pure: every method takes `now` (ms), so
 * the limits are unit-tested without timers.
 *
 * Groq's free plan for whisper-large-v3(-turbo) allows 20 requests per minute, 2,000 per day and 7,200 audio seconds
 * per hour, and bills every request as at least 10 s of audio (console.groq.com/docs/rate-limits, checked 2026-09).
 * The per-minute and per-day caps are enforced exactly with sliding windows; audio seconds are paced with a token
 * bucket sized so no rolling hour can exceed the hourly budget, which spreads the budget out instead of spending it
 * in 40 minutes and then stalling for 20.
 *
 * Because of the 10 s billing minimum, one request per short utterance bills two to three times the real audio and
 * uses up the 2,000 requests/day within about three hours of steady talking. So once requests run ahead of a pace
 * that makes the daily cap last a long session (`dailyPaceHours`), the limiter asks the utterance pipeline to hold
 * short utterances briefly and merge them (`minBatchSeconds`). Light talkers keep one request per utterance and no
 * added latency.
 */

/** A pacing policy: how long until a request carrying `audioSeconds` of audio may start. */
export interface RequestLimiter {
  /** Milliseconds until a request with `audioSeconds` of audio may start (0 = now). */
  waitMs(now: number, audioSeconds: number): number;
  /** Records a request that starts at `now`. */
  record(now: number, audioSeconds: number): void;
  /**
   * Audio a request should carry at `now` before it is sent (requests shorter than a billing minimum cost as much
   * as the minimum): the pipeline holds shorter batches briefly so more utterances can join them. Absent or 0 = send
   * every utterance at once.
   */
  minBatchSeconds?(now: number): number;
}

/**
 * Exact sliding-window limiter: the total weight recorded in any window of `windowMs` never exceeds `limit`
 * (an entry recorded at `t` counts until `t + windowMs`).
 */
export class SlidingWindowLimiter {
  #entries: { at: number; weight: number }[] = [];

  constructor(readonly windowMs: number, readonly limit: number) {
    if (!(windowMs > 0) || !(limit > 0)) throw new RangeError("SlidingWindowLimiter: windowMs and limit must be > 0");
  }

  #prune(now: number): void {
    let drop = 0;
    while (drop < this.#entries.length && this.#entries[drop].at + this.windowMs <= now) drop++;
    if (drop > 0) this.#entries.splice(0, drop);
  }

  /** Weight counted in the window ending at `now`. */
  used(now: number): number {
    this.#prune(now);
    return this.#entries.reduce((sum, e) => sum + e.weight, 0);
  }

  /** Milliseconds until `weight` fits. A weight above `limit` fits only in an empty window. */
  waitMs(now: number, weight = 1): number {
    const need = Math.min(Math.max(weight, 0), this.limit);
    let used = this.used(now);
    if (used + need <= this.limit) return 0;
    let i = 0;
    while (i < this.#entries.length && used + need > this.limit) {
      used -= this.#entries[i].weight;
      i++;
    }
    return Math.max(0, this.#entries[i - 1].at + this.windowMs - now);
  }

  record(now: number, weight = 1): void {
    this.#prune(now);
    // Keep entries ordered even if a caller's clock steps back a little.
    const last = this.#entries[this.#entries.length - 1];
    this.#entries.push({ at: last && last.at > now ? last.at : now, weight });
  }
}

/**
 * Token bucket: holds at most `capacity` tokens and refills at `refillPerSecond`. Over any interval of length T it
 * lets through at most `capacity + refillPerSecond * T` tokens. `take` may go into debt (a cost above the balance),
 * which later waits repay, so the bound still holds.
 */
export class TokenBucket {
  #tokens: number;
  #last: number;

  /** `minTokens` bounds the debt `take` can build up (default: unbounded). */
  constructor(
    readonly capacity: number,
    readonly refillPerSecond: number,
    now: number,
    initialTokens = capacity,
    readonly minTokens = Number.NEGATIVE_INFINITY,
  ) {
    if (!(capacity > 0) || !(refillPerSecond > 0)) throw new RangeError("TokenBucket: capacity and refill must be > 0");
    this.#tokens = Math.min(capacity, initialTokens);
    this.#last = now;
  }

  #refill(now: number): void {
    if (now <= this.#last) return;
    this.#tokens = Math.min(this.capacity, this.#tokens + ((now - this.#last) / 1000) * this.refillPerSecond);
    this.#last = now;
  }

  available(now: number): number {
    this.#refill(now);
    return this.#tokens;
  }

  /** Milliseconds until `cost` tokens are available (a cost above `capacity` waits for a full bucket). */
  waitMs(now: number, cost: number): number {
    const need = Math.min(Math.max(cost, 0), this.capacity);
    const have = this.available(now);
    // Tolerance for float refill arithmetic: waiting the returned time must always be enough.
    if (have >= need - 1e-6) return 0;
    return Math.ceil(((need - have) / this.refillPerSecond) * 1000);
  }

  take(now: number, cost: number): void {
    this.#refill(now);
    this.#tokens = Math.max(this.minTokens, this.#tokens - Math.max(cost, 0));
  }
}

export interface GroqPlanLimits {
  requestsPerMinute: number;
  requestsPerDay: number;
  audioSecondsPerHour: number;
  /** Every request is billed as at least this much audio. */
  minBilledSeconds: number;
}

/** Groq free plan, whisper-large-v3 and whisper-large-v3-turbo. */
export const GROQ_FREE_PLAN_LIMITS: Readonly<GroqPlanLimits> = Object.freeze({
  requestsPerMinute: 20,
  requestsPerDay: 2000,
  audioSecondsPerHour: 7200,
  minBilledSeconds: 10,
});

export interface GroqLimiterOptions {
  /** Added to each window so clock skew and network jitter cannot squeeze an extra request into Groq's window. */
  windowMarginMs?: number;
  /**
   * Requests that may go out back to back before per-minute pacing starts. A plain 20-per-minute window lets a
   * fast talker spend all 20 in the first 40 s and then shows no captions for 20 s; pacing the rest keeps waits
   * to a few seconds (and queued utterances are merged meanwhile).
   */
  requestBurst?: number;
  /** Audio seconds that may be spent in a burst before pacing starts (the bucket capacity). */
  audioBurstSeconds?: number;
  /** Fraction of the hourly audio budget used (headroom for requests an earlier app session already made). */
  audioBudgetFraction?: number;
  /**
   * Continuous talking the daily request cap should last (default 8 h). Requests beyond that pace make the
   * pipeline merge short utterances up to the billing minimum.
   */
  dailyPaceHours?: number;
  /** Requests sent one per utterance before the daily pace applies (default 100). */
  paceBurstRequests?: number;
}

/** Audio seconds Groq bills for a request carrying `audioSeconds` of audio. */
export function billedAudioSeconds(audioSeconds: number, minBilledSeconds: number): number {
  return Math.max(minBilledSeconds, Number.isFinite(audioSeconds) ? audioSeconds : 0);
}

/**
 * Paces requests so the free plan's per-minute, per-day and per-hour audio limits are never exceeded.
 *
 * - Requests per minute: a token bucket (burst `requestBurst`, then an even pace) sized so any window of
 *   60 s + margin holds at most `requestsPerMinute`; the exact sliding window stays as a hard backstop.
 * - Audio per hour: a token bucket over billed seconds, sized the same way for a rolling hour.
 * - Requests per day: an exact sliding window.
 */
export class GroqRateLimiter implements RequestLimiter {
  /** `null` when the per-minute limit is too small to pace (the sliding window alone applies). */
  readonly requests: TokenBucket | null;
  readonly perMinute: SlidingWindowLimiter;
  readonly perDay: SlidingWindowLimiter;
  readonly audio: TokenBucket;
  /**
   * Daily request pace: every request takes a token, refilled so the daily cap lasts `dailyPaceHours`. While it is
   * empty, short utterances are merged up to the billing minimum (see {@link minBatchSeconds}).
   */
  readonly pace: TokenBucket;

  constructor(now: number, readonly limits: GroqPlanLimits = GROQ_FREE_PLAN_LIMITS, options: GroqLimiterOptions = {}) {
    const margin = options.windowMarginMs ?? 2_000;
    const minuteWindowS = (60_000 + margin) / 1000;
    // capacity + rate * window == limit: no window of 60 s + margin can hold more than the limit.
    const requestBurst = Math.min(Math.max(1, options.requestBurst ?? 5), limits.requestsPerMinute - 1);
    this.requests =
      requestBurst >= 1 ? new TokenBucket(requestBurst, (limits.requestsPerMinute - requestBurst) / minuteWindowS, now) : null;
    this.perMinute = new SlidingWindowLimiter(60_000 + margin, limits.requestsPerMinute);
    this.perDay = new SlidingWindowLimiter(86_400_000 + margin, limits.requestsPerDay);
    const fraction = options.audioBudgetFraction ?? 0.97;
    const hourly = limits.audioSecondsPerHour * fraction;
    // Bounded so capacity + rate * 3600 s == hourly budget: no rolling hour can exceed it.
    const audioBurst = Math.min(options.audioBurstSeconds ?? 900, hourly / 2);
    this.audio = new TokenBucket(audioBurst, (hourly - audioBurst) / 3600, now);
    const paceHours = Math.max(1, options.dailyPaceHours ?? 8);
    const paceBurst = Math.min(Math.max(1, options.paceBurstRequests ?? 100), limits.requestsPerDay / 2);
    // Debt is bounded so a talkative stretch does not keep merging on for hours after the talking slowed down.
    this.pace = new TokenBucket(paceBurst, (limits.requestsPerDay - paceBurst) / (paceHours * 3600), now, paceBurst, -paceBurst);
  }

  /** The billing minimum while requests run ahead of the daily pace, else 0 (no merging, no added latency). */
  minBatchSeconds(now: number): number {
    return this.pace.available(now) >= 1 ? 0 : Math.max(0, this.limits.minBilledSeconds);
  }

  waitMs(now: number, audioSeconds: number): number {
    const billed = billedAudioSeconds(audioSeconds, this.limits.minBilledSeconds);
    return Math.max(
      this.requests?.waitMs(now, 1) ?? 0,
      this.perMinute.waitMs(now),
      this.perDay.waitMs(now),
      this.audio.waitMs(now, billed),
    );
  }

  record(now: number, audioSeconds: number): void {
    this.pace.take(now, 1);
    this.requests?.take(now, 1);
    this.perMinute.record(now);
    this.perDay.record(now);
    this.audio.take(now, billedAudioSeconds(audioSeconds, this.limits.minBilledSeconds));
  }
}

/**
 * `Retry-After` as milliseconds from `now`: delta-seconds or an HTTP date. `null` when absent or unparsable
 * (browsers hide the header unless the server exposes it via CORS).
 */
export function parseRetryAfterMs(header: string | null | undefined, now: number): number | null {
  const value = header?.trim();
  if (!value) return null;
  if (/^\d+(\.\d+)?$/.test(value)) return Math.round(parseFloat(value) * 1000);
  const at = Date.parse(value);
  if (Number.isNaN(at)) return null;
  return Math.max(0, at - now);
}
