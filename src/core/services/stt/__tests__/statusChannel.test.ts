import { describe, expect, it, vi } from "vitest";
import {
  clampProgress,
  createScopedReceiver,
  EMPTY_STT_STATUS,
  MAX_STATUS_LENGTH,
  normalizeSttStatus,
  type ScopedReceiverSinks,
} from "../statusChannel";

function sinks(overrides: Partial<ScopedReceiverSinks> = {}) {
  const s = {
    isCurrent: vi.fn(() => true),
    start: vi.fn(),
    stop: vi.fn(),
    interim: vi.fn(),
    final: vi.fn(),
    status: vi.fn(),
    alive: vi.fn(),
    muted: vi.fn(() => false),
    ...overrides,
  };
  return s;
}

describe("STT status channel", () => {
  it("routes status to the UI only, never to the transcript sinks", () => {
    const s = sinks();
    const r = createScopedReceiver(s);
    r.onStatus("Downloading ggml-base.en.bin: 45%", { progress: 45 });
    r.onStatus("Loading the Whisper model…");
    r.onStatus("Rate limit: next request in 3 s", { level: "warning" });
    r.onStatus("");
    expect(s.interim).not.toHaveBeenCalled();
    expect(s.final).not.toHaveBeenCalled();
    expect(s.status).toHaveBeenCalledTimes(4);
    expect(s.status).toHaveBeenNthCalledWith(1, { text: "Downloading ggml-base.en.bin: 45%", progress: 45, level: "info" });
    expect(s.status).toHaveBeenNthCalledWith(3, expect.objectContaining({ level: "warning" }));
    expect(s.status).toHaveBeenLastCalledWith(EMPTY_STT_STATUS);
  });

  it("status and interims keep the connecting watchdog alive; finals do not need to", () => {
    const s = sinks();
    const r = createScopedReceiver(s);
    r.onStatus("Downloading…");
    r.onInterim("hello");
    expect(s.alive).toHaveBeenCalledTimes(2);
    r.onFinal("hello there");
    expect(s.alive).toHaveBeenCalledTimes(2);
    expect(s.interim).toHaveBeenCalledWith("hello", undefined);
    expect(s.final).toHaveBeenCalledWith("hello there", undefined);
  });

  it("forwards the engine's segment metadata with the text", () => {
    const s = sinks();
    const r = createScopedReceiver(s);
    r.onInterim("hel", { segmentId: "mic:1:2", revision: 0 });
    r.onFinal("Hello.", { segmentId: "mic:1:2", revision: 1, lang: "en" });
    expect(s.interim).toHaveBeenCalledWith("hel", { segmentId: "mic:1:2", revision: 0 });
    expect(s.final).toHaveBeenCalledWith("Hello.", { segmentId: "mic:1:2", revision: 1, lang: "en" });
  });

  it("forwards a silent microphone to the UI only, with the device trimmed, and drops it from a stale session", () => {
    let current = true;
    const s = sinks({ isCurrent: vi.fn(() => current), inputSilence: vi.fn() });
    const r = createScopedReceiver(s);
    r.onInputSilence?.(true, "  Headset Mic ");
    r.onInputSilence?.(true);
    r.onInputSilence?.(false);
    expect(s.inputSilence).toHaveBeenNthCalledWith(1, true, "Headset Mic");
    expect(s.inputSilence).toHaveBeenNthCalledWith(2, true, "");
    expect(s.inputSilence).toHaveBeenNthCalledWith(3, false, "");
    // Not text, not status, and not a sign of life for the connecting watchdog.
    for (const fn of [s.interim, s.final, s.status, s.alive]) expect(fn).not.toHaveBeenCalled();
    current = false;
    r.onInputSilence?.(true, "late");
    expect(s.inputSilence).toHaveBeenCalledTimes(3);
  });

  it("a service without the silent-microphone sink ignores the report", () => {
    const r = createScopedReceiver(sinks());
    expect(() => r.onInputSilence?.(true, "Headset Mic")).not.toThrow();
  });

  it("drops every report from a stale session", () => {
    let current = true;
    const s = sinks({ isCurrent: vi.fn(() => current) });
    const r = createScopedReceiver(s);
    current = false;
    r.onStart();
    r.onStop("boom");
    r.onInterim("late words");
    r.onFinal("late final");
    r.onStatus("late status", { progress: 50 });
    for (const fn of [s.start, s.stop, s.interim, s.final, s.status, s.alive]) expect(fn).not.toHaveBeenCalled();
    // A stale engine asking whether to do work is told it is muted, so it skips billed requests.
    expect(r.isMuted()).toBe(true);
  });

  it("reports mute from the service while current", () => {
    let muted = false;
    const r = createScopedReceiver(sinks({ muted: () => muted }));
    expect(r.isMuted()).toBe(false);
    muted = true;
    expect(r.isMuted()).toBe(true);
  });

  it("normalizes status lines and progress", () => {
    expect(normalizeSttStatus("  a \n  b  ")).toEqual({ text: "a b", progress: null, level: "info" });
    expect(normalizeSttStatus("   ", { progress: 40, level: "warning" })).toEqual(EMPTY_STT_STATUS);
    expect(normalizeSttStatus("x".repeat(1000)).text.length).toBe(MAX_STATUS_LENGTH);
    expect(clampProgress(-1)).toBeNull();
    expect(clampProgress(Number.NaN)).toBeNull();
    expect(clampProgress("50")).toBeNull();
    expect(clampProgress(150)).toBe(100);
    expect(clampProgress(12.5)).toBe(12.5);
  });

  it("keeps a positive expiry for transient lines only", () => {
    expect(normalizeSttStatus("skipped", { level: "warning", expireMs: 15_000 })).toEqual({
      text: "skipped",
      progress: null,
      level: "warning",
      expireMs: 15_000,
    });
    expect(normalizeSttStatus("x", { expireMs: 0 })).not.toHaveProperty("expireMs");
    expect(normalizeSttStatus("x", { expireMs: Number.NaN })).not.toHaveProperty("expireMs");
    expect(normalizeSttStatus("", { expireMs: 5_000 })).toEqual(EMPTY_STT_STATUS);
  });
});
