import { describe, expect, it } from "vitest";
import {
  OutcomeDeduper,
  RateLimiter,
  SeqGate,
  componentKey,
  epLabel,
  formatBytes,
  logEntryToSystemLog,
  outcomeToSystemLog,
  packProgressFraction,
  tagForTarget,
} from "../format";
import type { RustEpOutcomeEntry, RustPackProgress } from "../types";

const entry = (over: Partial<RustEpOutcomeEntry>): RustEpOutcomeEntry => ({
  component: "vc",
  requested: "cuda",
  active: "cuda",
  reason: null,
  at_unix_ms: 1,
  ...over,
});

describe("diagnostics format helpers", () => {
  it("labels execution providers", () => {
    expect(epLabel("cuda")).toBe("CUDA");
    expect(epLabel("GPU")).toBe("CUDA");
    expect(epLabel("dml")).toBe("DirectML");
    expect(epLabel("cpu")).toBe("CPU");
    expect(epLabel(undefined)).toBe("CPU");
    expect(epLabel("tensorrt")).toBe("tensorrt");
  });

  it("maps components to i18n keys", () => {
    expect(componentKey("vc.pitch")).toBe("vc_pitch");
    expect(componentKey("stt.vad")).toBe("stt_vad");
    expect(componentKey("stt.turn")).toBe("stt_turn");
    expect(componentKey("future.thing")).toBeNull();
  });

  it("reports fallbacks truthfully as warnings with the reason", () => {
    const line = outcomeToSystemLog(entry({ active: "dml", reason: "cudnn64_9.dll was not found" }));
    expect(line).toEqual({
      tag: "GPU",
      level: "warning",
      message:
        "Voice changer model: CUDA is unavailable, running on DirectML. Reason: cudnn64_9.dll was not found",
    });
  });

  it("reports granted GPU sessions, voice-changer CPU sessions, and skips CPU-only helpers", () => {
    expect(outcomeToSystemLog(entry({ requested: "dml", active: "dml" }))?.level).toBe("success");
    expect(outcomeToSystemLog(entry({ requested: "cpu", active: "cpu" }))?.message).toMatch(/CPU/);
    expect(outcomeToSystemLog(entry({ component: "stt.vad", requested: "cpu", active: "cpu" }))).toBeNull();
  });

  it("de-duplicates identical outcomes per component within the window", () => {
    const d = new OutcomeDeduper(1000);
    const e = entry({ active: "cpu", reason: "x" });
    expect(d.accept(e, 0)).toBe(true);
    expect(d.accept(e, 500)).toBe(false);
    expect(d.accept(entry({ component: "vc.pitch", active: "cpu", reason: "x" }), 500)).toBe(true);
    expect(d.accept(entry({ active: "dml", reason: "y" }), 600)).toBe(true);
    expect(d.accept(entry({ active: "dml", reason: "y" }), 1700)).toBe(true);
  });

  it("accepts each Rust log sequence number once", () => {
    const g = new SeqGate();
    expect([3, 1, 4, 4, 0, 0, 5].map((s) => g.accept(s))).toEqual([true, false, true, false, true, true, true]);
  });

  it("derives system-log tags from tracing targets", () => {
    expect(tagForTarget("sigil::services::voice_changer::worker")).toBe("Voice changer");
    expect(tagForTarget("sigil::services::neural_audio::chunk_processor")).toBe("Voice changer");
    expect(tagForTarget("sigil::services::neural_audio::ort_session")).toBe("GPU");
    expect(tagForTarget("sigil::ort")).toBe("GPU");
    expect(tagForTarget("ort::session")).toBe("GPU");
    expect(tagForTarget("sigil::services::whisper")).toBe("STT");
    expect(tagForTarget("sigil::services::windows_tts")).toBe("TTS");
    expect(tagForTarget("sigil::panic")).toBe("Crash");
    expect(tagForTarget("tauri::manager")).toBe("Sigil");
    expect(logEntryToSystemLog({ seq: 1, at_unix_ms: 0, level: "error", target: "sigil::panic", message: "boom" })).toEqual({
      tag: "Crash",
      level: "error",
      message: "boom",
    });
  });

  it("formats download sizes", () => {
    expect(formatBytes(1_434_973_784)).toBe("1.4 GB");
    expect(formatBytes(549_755_186)).toBe("550 MB");
    expect(formatBytes(3_521_238)).toBe("3.5 MB");
    expect(formatBytes(812_000)).toBe("812 KB");
    expect(formatBytes(-1)).toBe("?");
  });

  it("computes monotonic install progress", () => {
    const p = (over: Partial<RustPackProgress>): RustPackProgress => ({
      phase: "download",
      archive: null,
      archive_index: 0,
      archive_count: 4,
      done_bytes: 0,
      total_bytes: 100,
      message: null,
      ...over,
    });
    const seq = [
      p({ done_bytes: 0 }),
      p({ done_bytes: 50 }),
      p({ done_bytes: 100 }),
      p({ phase: "extract", archive_index: 0, done_bytes: 100 }),
      p({ phase: "extract", archive_index: 3, done_bytes: 100 }),
      p({ phase: "finalize", done_bytes: 100 }),
      p({ phase: "done", done_bytes: 100 }),
    ].map(packProgressFraction);
    for (let i = 1; i < seq.length; i++) expect(seq[i]).toBeGreaterThanOrEqual(seq[i - 1]);
    expect(seq[seq.length - 1]).toBe(1);
    expect(packProgressFraction(null)).toBe(0);
  });

  it("rate-limits file-log forwarding per window", () => {
    const r = new RateLimiter(2, 1000);
    expect([r.allow(0), r.allow(10), r.allow(20), r.allow(1000), r.allow(1001)]).toEqual([true, true, false, true, true]);
  });
});
