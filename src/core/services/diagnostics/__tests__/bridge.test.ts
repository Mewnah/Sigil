import { describe, expect, it, vi } from "vitest";
import type { RustEpOutcomeEntry, RustEpProbeReport, RustLogEntry } from "../types";

type Handler = (ev: { payload: unknown }) => void;
const handlers = new Map<string, Handler>();

vi.mock("@tauri-apps/api/event", () => ({
  listen: async (name: string, cb: Handler) => {
    handlers.set(name, cb);
    return () => handlers.delete(name);
  },
}));
vi.mock("@tauri-apps/api/core", () => ({ invoke: async () => undefined }));

let releaseBackfill: ((entries: RustLogEntry[]) => void) | null = null;
const probe = {
  adapters: [{ index: 0, name: "NVIDIA GeForce RTX 4090", software: false }],
  recommended_ep: "dml",
  notes: ["The CUDA 12 / cuDNN 9 runtime is not installed."],
} as unknown as RustEpProbeReport;

vi.mock("../api", async (importOriginal) => {
  const actual = await importOriginal<typeof import("../api")>();
  return {
    ...actual,
    diagnosticsApi: {
      getRecentLogEntries: () => new Promise<RustLogEntry[]>((resolve) => (releaseBackfill = resolve)),
      getEpOutcomes: async () => [],
      getEpProbe: async () => probe,
      getNvidiaPackStatus: async () => ({ installed: false }),
      installNvidiaPack: async () => ({ installed: true, label: "cuda-12.9.1_cudnn-9.10.2", download_bytes: 1 }),
      getLogInfo: async () => ({ file: "C:/logs/sigil.log" }),
      getLastDeepEpProbe: async () => null,
    },
  };
});

const log = (seq: number, message: string, level: "warn" | "error" = "warn"): RustLogEntry => ({
  seq,
  at_unix_ms: seq,
  level,
  target: "sigil::services::voice_changer::worker",
  message,
});

describe("diagnostics bridge", () => {
  it("delivers backfilled and live Rust log lines once, in order, and reports EP changes", async () => {
    const { startDiagnosticsBridge, diagnosticsState, stopDiagnosticsBridge } = await import("../bridge");
    const lines: Array<[string, string, string, boolean | undefined]> = [];
    const started = startDiagnosticsBridge((tag, message, level, options) => lines.push([tag, message, level, options?.fileLog]));

    // Listeners attach first; a live line arriving before the backfill resolves must wait for it.
    await vi.waitFor(() => expect(releaseBackfill).not.toBeNull());
    expect(handlers.has("diagnostics:log")).toBe(true);
    handlers.get("diagnostics:log")!({ payload: log(5, "live five") });
    releaseBackfill!([log(3, "early three"), log(5, "live five", "warn")]);
    await started;

    const logged = lines.filter(([tag]) => tag === "Voice changer").map(([, m]) => m);
    expect(logged).toEqual(["early three", "live five"]);
    expect(lines.every(([, , , fileLog]) => fileLog === false)).toBe(true); // already in the Rust file log

    handlers.get("diagnostics:log")!({ payload: log(5, "live five") });
    handlers.get("diagnostics:log")!({ payload: log(6, "boom", "error") });
    const after = lines.filter(([tag]) => tag === "Voice changer");
    expect(after.map(([, m, level]) => `${m}:${level}`)).toEqual(["early three:warning", "live five:warning", "boom:error"]);

    // The startup probe is summarized once.
    expect(lines.filter(([tag, m]) => tag === "GPU" && m.includes("best available: DirectML"))).toHaveLength(1);
    expect(diagnosticsState.logInfo?.file).toBe("C:/logs/sigil.log");

    // EP outcomes update the store every time but the system log only on change.
    const outcome: RustEpOutcomeEntry = {
      component: "vc",
      requested: "cuda",
      active: "dml",
      reason: "cudnn64_9.dll (CUDA 12 / cuDNN 9 runtime) was not found",
      at_unix_ms: 10,
    };
    handlers.get("diagnostics:ep_outcome")!({ payload: outcome });
    handlers.get("diagnostics:ep_outcome")!({ payload: { ...outcome, at_unix_ms: 11 } });
    expect(diagnosticsState.outcomes.vc.at_unix_ms).toBe(11);
    const gpu = lines.filter(([tag, m]) => tag === "GPU" && m.startsWith("Voice changer model"));
    expect(gpu).toEqual([
      [
        "GPU",
        "Voice changer model: CUDA is unavailable, running on DirectML. Reason: cudnn64_9.dll (CUDA 12 / cuDNN 9 runtime) was not found",
        "warning",
        false,
      ],
    ]);

    // Installing the NVIDIA pack drops the GPU test run before it; Rust re-emits the light probe
    // and only a changed recommendation is logged again.
    const { installNvidiaPack } = await import("../bridge");
    diagnosticsState.deepProbe = { ...probe, deep: true } as RustEpProbeReport;
    await installNvidiaPack();
    expect(diagnosticsState.pack?.installed).toBe(true);
    expect(diagnosticsState.deepProbe).toBeNull();
    const nowCuda = { ...probe, recommended_ep: "cuda", notes: [] } as RustEpProbeReport;
    handlers.get("diagnostics:probe")!({ payload: nowCuda });
    handlers.get("diagnostics:probe")!({ payload: nowCuda });
    expect(diagnosticsState.probe?.recommended_ep).toBe("cuda");
    expect(lines.filter(([tag, m]) => tag === "GPU" && m === "Best available is now CUDA.")).toHaveLength(1);
    expect(lines.filter(([tag, m]) => tag === "GPU" && m.includes("best available:"))).toHaveLength(1);

    stopDiagnosticsBridge();
    expect(handlers.size).toBe(0);
  });
});
