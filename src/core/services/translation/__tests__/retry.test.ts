import { afterEach, describe, expect, it, vi } from "vitest";
import { abortableDelay, raceWithSignal } from "../abort";
import { isRetryableTranslationError, withOneRetry } from "../retry";
import { SequencerTimeoutError } from "../sequencer";
import { isRetryableStatus, TranslationError } from "../types";

afterEach(() => {
  vi.useRealTimers();
});

describe("withOneRetry", () => {
  it("retries a fast retryable failure once", async () => {
    vi.useFakeTimers();
    const attempt = vi.fn().mockRejectedValueOnce(new Error("ECONNRESET")).mockResolvedValueOnce("ok");
    const p = withOneRetry(attempt, new AbortController().signal, { delayMs: 300 });
    await vi.advanceTimersByTimeAsync(300);
    await expect(p).resolves.toBe("ok");
    expect(attempt).toHaveBeenCalledTimes(2);
  });

  it("gives up after the second failure", async () => {
    const attempt = vi.fn().mockRejectedValue(new TranslationError("429", { status: 429 }));
    await expect(withOneRetry(attempt, new AbortController().signal, { delayMs: 0 })).rejects.toThrow("429");
    expect(attempt).toHaveBeenCalledTimes(2);
  });

  it("does not retry errors marked non-retryable", async () => {
    const attempt = vi.fn().mockRejectedValue(new TranslationError("bad key", { status: 401, retryable: false }));
    await expect(withOneRetry(attempt, new AbortController().signal, { delayMs: 0 })).rejects.toThrow("bad key");
    expect(attempt).toHaveBeenCalledTimes(1);
  });

  it("does not retry a slow failure", async () => {
    let now = 0;
    const attempt = vi.fn().mockImplementationOnce(async () => {
      now += 6_000;
      throw new Error("slow");
    });
    await expect(
      withOneRetry(attempt, new AbortController().signal, { delayMs: 0, retryWithinMs: 5_000, now: () => now }),
    ).rejects.toThrow("slow");
    expect(attempt).toHaveBeenCalledTimes(1);
  });

  it("does not retry once aborted, and an abort during the pause cancels the retry", async () => {
    const aborted = new AbortController();
    aborted.abort(new Error("stop"));
    const attempt = vi.fn().mockRejectedValue(new Error("x"));
    await expect(withOneRetry(attempt, aborted.signal)).rejects.toThrow("x");
    expect(attempt).toHaveBeenCalledTimes(1);

    vi.useFakeTimers();
    const c = new AbortController();
    const attempt2 = vi.fn().mockRejectedValueOnce(new Error("first")).mockResolvedValue("never");
    const p = withOneRetry(attempt2, c.signal, { delayMs: 1_000 });
    const settled = expect(p).rejects.toThrow("stopped");
    await vi.advanceTimersByTimeAsync(10);
    c.abort(new Error("stopped"));
    await settled;
    expect(attempt2).toHaveBeenCalledTimes(1);
  });
});

describe("retry classification", () => {
  it("classifies statuses and errors", () => {
    expect(isRetryableStatus(429)).toBe(true);
    expect(isRetryableStatus(503)).toBe(true);
    expect(isRetryableStatus(401)).toBe(false);
    expect(isRetryableStatus(400)).toBe(false);
    expect(isRetryableTranslationError(new SequencerTimeoutError(1_000))).toBe(false);
    expect(isRetryableTranslationError(new DOMException("Aborted", "AbortError"))).toBe(false);
    expect(isRetryableTranslationError("LibreTranslate translation: HTTP 502")).toBe(true);
  });
});

describe("abort helpers", () => {
  it("raceWithSignal settles with the promise, or with the abort reason", async () => {
    await expect(raceWithSignal(Promise.resolve(1), new AbortController().signal)).resolves.toBe(1);
    const c = new AbortController();
    const never = new Promise<number>(() => {});
    const p = raceWithSignal(never, c.signal);
    c.abort(new Error("gone"));
    await expect(p).rejects.toThrow("gone");
    const pre = new AbortController();
    pre.abort();
    await expect(raceWithSignal(Promise.resolve(1), pre.signal)).rejects.toBeDefined();
  });

  it("abortableDelay resolves after the delay", async () => {
    vi.useFakeTimers();
    const done = vi.fn();
    void abortableDelay(100, new AbortController().signal).then(done);
    await vi.advanceTimersByTimeAsync(99);
    expect(done).not.toHaveBeenCalled();
    await vi.advanceTimersByTimeAsync(1);
    expect(done).toHaveBeenCalled();
  });
});
