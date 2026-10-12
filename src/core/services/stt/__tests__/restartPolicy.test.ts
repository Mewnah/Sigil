import { describe, expect, it } from "vitest";
import {
  isSlowRestart,
  MAX_RESTARTS,
  networkFailureCanRetry,
  networkFailureNeedsPause,
  RESTART_DELAYS_MS,
  restartCanFix,
  restartDelayMs,
  SLOW_RESTART_MS,
  startFailureCanRetry,
} from "../restartPolicy";

describe("restart ladder", () => {
  it("waits the fast delays, then a steady slow pace that never ends", () => {
    expect(RESTART_DELAYS_MS.map((_, i) => restartDelayMs(i + 1))).toEqual([...RESTART_DELAYS_MS]);
    for (const attempt of [MAX_RESTARTS + 1, MAX_RESTARTS + 2, 100, 10_000]) {
      expect(restartDelayMs(attempt)).toBe(SLOW_RESTART_MS);
      expect(isSlowRestart(attempt)).toBe(true);
    }
    expect(isSlowRestart(MAX_RESTARTS)).toBe(false);
    // The slow pace is about 30-60 s: a streamer in VR is back in seconds once the fault passes, and a fault that
    // lasts does not make the PC reload its models in a loop.
    expect(SLOW_RESTART_MS).toBeGreaterThanOrEqual(30_000);
    expect(SLOW_RESTART_MS).toBeLessThanOrEqual(60_000);
  });
});

describe("what a failed start or a failed HTTP / cloud session retries", () => {
  it("retries a busy microphone and a failed engine load, not a request nothing can run", () => {
    expect(startFailureCanRetry("Could not open the audio device 'USB'", null, "auto")).toBe(true);
    expect(startFailureCanRetry("runtime: model load failed", null, "cpu")).toBe(true);
    expect(startFailureCanRetry("out_of_memory: VRAM", null, "auto")).toBe(true);
    expect(startFailureCanRetry("out_of_memory: VRAM", null, "gpu")).toBe(false);
    expect(startFailureCanRetry("invalid_input: bad options", null, "auto")).toBe(false);
    expect(startFailureCanRetry("not_installed: stt/x", null, "auto")).toBe(false);
    // A missing model with no catalog id (a bring-your-own file nobody chose) would loop for ever: a restart installs nothing.
    expect(startFailureCanRetry("not_installed: Model whisper/custom is bring-your-own: choose its .bin file", null, "auto")).toBe(false);
    expect(startFailureCanRetry("start_pipeline failed: not_installed: the model file is missing", null, "auto")).toBe(false);
    expect(restartCanFix("engine_error", "not_installed: custom model", null, "auto")).toBe(false);
    expect(startFailureCanRetry("no speech engine for this request (language xx)", null, "auto")).toBe(false);
    // Another start took the pipeline over: retrying would fight it.
    expect(startFailureCanRetry("cancelled", null, "auto")).toBe(false);
    // It keeps the rules of a session restart for everything else.
    expect(restartCanFix("", "no speech engine for this request (language xx)", null, "auto")).toBe(true);
  });

  it("retries a network failure, a lost microphone and a rate limit; a rejected key and a bad request need the user", () => {
    expect(networkFailureCanRetry("engine_error", "5 utterances failed in a row; last error: network: connection refused")).toBe(true);
    expect(networkFailureCanRetry("capture_error", "no input device named 'USB'")).toBe(true);
    expect(networkFailureCanRetry("", "Could not open the audio device")).toBe(true);
    expect(networkFailureCanRetry("engine_error", "invalid_input: HTTP transcription: HTTP 401 Unauthorized")).toBe(false);
    expect(networkFailureCanRetry("engine_error", "unsupported: language xx")).toBe(false);
    // A rate limit clears with time (Groq's per-minute window): it is retried, slowly.
    expect(networkFailureCanRetry("rate_limited", "5 utterances failed in a row; last error: rate_limited: slow down")).toBe(true);
    expect(networkFailureNeedsPause("rate_limited")).toBe(true);
    expect(networkFailureNeedsPause("engine_error")).toBe(false);
    expect(networkFailureNeedsPause("capture_error")).toBe(false);
    // An unreadable reply (a login page in the way) is a network-class failure; a rejected request still is not.
    expect(networkFailureCanRetry("engine_error", "5 utterances failed in a row; last error: network: the server returned a response that is not JSON (failed 4 times)")).toBe(true);
    expect(networkFailureCanRetry("not_installed", "stt/x")).toBe(false);
    expect(networkFailureCanRetry("something_new", "boom")).toBe(false);
  });
});
