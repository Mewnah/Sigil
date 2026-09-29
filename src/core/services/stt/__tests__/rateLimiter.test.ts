import { describe, expect, it } from "vitest";
import {
  billedAudioSeconds,
  GROQ_FREE_PLAN_LIMITS,
  GroqRateLimiter,
  parseRetryAfterMs,
  SlidingWindowLimiter,
  TokenBucket,
} from "../rateLimiter";

/** Max number of timestamps inside any half-open window of `windowMs`. */
function maxInWindow(times: number[], windowMs: number): number {
  let best = 0;
  let lo = 0;
  for (let hi = 0; hi < times.length; hi++) {
    while (times[hi] - times[lo] >= windowMs) lo++;
    best = Math.max(best, hi - lo + 1);
  }
  return best;
}

/** Drives `limiter` greedily: every request starts as soon as it is allowed. */
function flood(limiter: { waitMs(now: number, s: number): number; record(now: number, s: number): void }, count: number, audioSeconds: number, start = 0) {
  const times: number[] = [];
  let now = start;
  for (let i = 0; i < count; i++) {
    const wait = limiter.waitMs(now, audioSeconds);
    now += wait;
    expect(limiter.waitMs(now, audioSeconds)).toBe(0);
    limiter.record(now, audioSeconds);
    times.push(now);
  }
  return times;
}

describe("SlidingWindowLimiter", () => {
  it("allows a burst up to the limit, then waits for the oldest entry to leave the window", () => {
    const l = new SlidingWindowLimiter(60_000, 3);
    l.record(0);
    l.record(1_000);
    l.record(2_000);
    expect(l.waitMs(2_500)).toBe(57_500);
    expect(l.waitMs(60_000)).toBe(0);
    expect(l.used(60_000)).toBe(2);
  });

  it("never lets more than `limit` into any window under a flood", () => {
    const l = new SlidingWindowLimiter(60_000, 20);
    const times = flood(l, 200, 1);
    expect(maxInWindow(times, 60_000)).toBe(20);
  });

  it("handles weights", () => {
    const l = new SlidingWindowLimiter(1_000, 10);
    l.record(0, 6);
    expect(l.waitMs(0, 4)).toBe(0);
    expect(l.waitMs(0, 5)).toBe(1_000);
    // Heavier than the limit: allowed only once the window is empty.
    expect(l.waitMs(500, 50)).toBe(500);
  });
});

describe("TokenBucket", () => {
  it("spends, refills and waits for the missing tokens", () => {
    const b = new TokenBucket(10, 2, 0);
    b.take(0, 10);
    expect(b.available(0)).toBe(0);
    expect(b.waitMs(0, 4)).toBe(2_000);
    expect(b.available(1_000)).toBe(2);
    expect(b.available(100_000)).toBe(10);
  });

  it("never passes more than capacity + rate * T in any window", () => {
    const b = new TokenBucket(100, 1, 0);
    const times = flood({ waitMs: (n, c) => b.waitMs(n, c), record: (n, c) => b.take(n, c) }, 500, 10);
    // In any 1000 s window at most 100 + 1000 = 1100 tokens = 110 requests of 10.
    expect(maxInWindow(times, 1_000_000)).toBeLessThanOrEqual(110);
  });

  it("bounds the debt when a floor is set", () => {
    const b = new TokenBucket(10, 1, 0, 10, -10);
    b.take(0, 1_000);
    expect(b.available(0)).toBe(-10);
    expect(b.waitMs(0, 1)).toBe(11_000);
  });
});

describe("GroqRateLimiter (free plan)", () => {
  it("bills at least 10 s per request", () => {
    expect(billedAudioSeconds(2, 10)).toBe(10);
    expect(billedAudioSeconds(25, 10)).toBe(25);
    expect(billedAudioSeconds(Number.NaN, 10)).toBe(10);
  });

  it("never exceeds 20 requests in any minute, even with short utterances back to back", () => {
    const g = new GroqRateLimiter(0);
    const times = flood(g, 400, 1.5);
    expect(maxInWindow(times, 60_000)).toBeLessThanOrEqual(GROQ_FREE_PLAN_LIMITS.requestsPerMinute);
    // Safety margin: not even 20 in a slightly longer window measured from Groq's side.
    expect(maxInWindow(times, 61_000)).toBeLessThanOrEqual(20);
  });

  it("keeps every rolling hour under the audio budget (billed seconds)", () => {
    const g = new GroqRateLimiter(0);
    // 1,500 requests (under the 2,000/day cap) of 1.5 s audio, each billed as 10 s: about 2.5 hours of pacing.
    const times = flood(g, 1_500, 1.5);
    let worst = 0;
    let lo = 0;
    for (let hi = 0; hi < times.length; hi++) {
      while (times[hi] - times[lo] >= 3_600_000) lo++;
      worst = Math.max(worst, (hi - lo + 1) * 10);
    }
    expect(worst).toBeLessThanOrEqual(GROQ_FREE_PLAN_LIMITS.audioSecondsPerHour);
    // It paces instead of stalling: the sustained gap stays in seconds, not minutes.
    const gaps = times.slice(1).map((t, i) => t - times[i]);
    expect(Math.max(...gaps)).toBeLessThan(10_000);
  });

  it("asks for merging (the billing minimum) only once requests run ahead of the daily pace", () => {
    const g = new GroqRateLimiter(0);
    // Light use: no merging, so no added caption latency.
    expect(g.minBatchSeconds(0)).toBe(0);
    let now = 0;
    for (let i = 0; i < 99; i++) g.record((now += 100), 2);
    expect(g.minBatchSeconds(now)).toBe(0);
    // The burst (100 requests) is spent: short utterances are merged up to the 10 s minimum from now on.
    g.record((now += 100), 2);
    expect(g.minBatchSeconds(now)).toBe(GROQ_FREE_PLAN_LIMITS.minBilledSeconds);
    // The pace refills so the daily cap lasts 8 h of talking: (2,000 - 100) / 8 h, one request every ~15 s.
    expect(g.minBatchSeconds(now + 16_000)).toBe(0);
  });

  it("recovers from a long talkative stretch within a bounded time (the pace debt is capped)", () => {
    const g = new GroqRateLimiter(0, GROQ_FREE_PLAN_LIMITS, { dailyPaceHours: 8, paceBurstRequests: 100 });
    let now = 0;
    // Far more requests than the pace allows (record does not enforce waits).
    for (let i = 0; i < 3_000; i++) g.record((now += 100), 10);
    const perSecond = (GROQ_FREE_PLAN_LIMITS.requestsPerDay - 100) / (8 * 3600);
    // From the debt floor (-100) back to one token at the refill rate: about 26 minutes, not hours.
    const recoverMs = Math.ceil((101 / perSecond) * 1000);
    expect(g.minBatchSeconds(now + recoverMs - 60_000)).toBe(10);
    expect(g.minBatchSeconds(now + recoverMs + 1)).toBe(0);
  });

  it("caps requests per day", () => {
    const g = new GroqRateLimiter(0, { ...GROQ_FREE_PLAN_LIMITS, requestsPerDay: 5, audioSecondsPerHour: 1e9 });
    const times = flood(g, 6, 1);
    expect(times[5] - times[0]).toBeGreaterThanOrEqual(86_400_000);
  });
});

describe("parseRetryAfterMs", () => {
  it("parses delta-seconds and HTTP dates", () => {
    expect(parseRetryAfterMs("7", 0)).toBe(7_000);
    expect(parseRetryAfterMs("1.5", 0)).toBe(1_500);
    const now = Date.parse("Mon, 28 Sep 2026 10:00:00 GMT");
    expect(parseRetryAfterMs("Mon, 28 Sep 2026 10:00:30 GMT", now)).toBe(30_000);
    expect(parseRetryAfterMs("Mon, 28 Sep 2026 09:00:00 GMT", now)).toBe(0);
  });

  it("returns null when missing or unreadable", () => {
    expect(parseRetryAfterMs(null, 0)).toBeNull();
    expect(parseRetryAfterMs("", 0)).toBeNull();
    expect(parseRetryAfterMs("soon", 0)).toBeNull();
  });
});
