/**
 * Autotune, "Benchmark this PC" and live recording (`bench.ts`) against mocked IPC: the commands and
 * arguments sent, the guards (engine running / no voice), progress, results and errors.
 */
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";

type Handler = (ev: { payload: unknown }) => void;
const handlers = new Map<string, Handler>();
const invoke = vi.fn();

vi.mock("@tauri-apps/api/core", () => ({ invoke: (...args: unknown[]) => invoke(...args) }));
vi.mock("@tauri-apps/api/event", () => ({
    listen: async (name: string, cb: Handler) => {
        handlers.set(name, cb);
        return () => handlers.delete(name);
    },
}));
vi.mock("@/utils/devLog", () => ({ devLog: () => {} }));

import { VoiceBench, type VoiceBenchHost } from "../bench";
import { BENCH_IPC, VC_COMMANDS, VC_EVENTS } from "../ipc";
import { VoiceChangerStateSchema } from "../schema";

const emit = (name: string, payload: unknown) => handlers.get(name)?.({ payload });

let running = false;
let settings = VoiceChangerStateSchema.parse({});
const applyTuning = vi.fn(async () => true);
const host: VoiceBenchHost = { settings: () => settings, isRunning: () => running, applyTuning };

const result = { gpu: "RTX 4090", created_unix_ms: 1, params: { block_ms: 40 }, rtf_p99: 0.3, block_ms_p95: 3, notes: [] };

beforeEach(() => {
    handlers.clear();
    invoke.mockReset();
    applyTuning.mockClear();
    running = false;
    settings = { ...VoiceChangerStateSchema.parse({}), rvcSlotDir: "C:/m/alto" };
});

afterEach(() => {
    vi.useRealTimers();
});

describe("autotune", () => {
    it("loads the stored result, tunes, shows progress and applies the result", async () => {
        let finish: (v: unknown) => void = () => {};
        invoke.mockImplementation(async (cmd: string) => {
            if (cmd === VC_COMMANDS.getAutotune) return { ...result, gpu: "stored" };
            if (cmd === VC_COMMANDS.autotune) return new Promise((r) => (finish = r));
            return null;
        });
        const b = new VoiceBench(host);
        await b.init();
        expect(b.state.autotune.result?.gpu).toBe("stored");
        const p = b.runAutotune(true);
        expect(invoke).toHaveBeenCalledWith(VC_COMMANDS.autotune, { request: { quick: true } });
        expect(b.state.autotune.running).toBe(true);
        emit(VC_EVENTS.autotuneProgress, { stage: "sweeping", fraction: 0.5, candidate: "cuda 40 ms", message: "" });
        expect(b.state.autotune.progress?.candidate).toBe("cuda 40 ms");
        finish(result);
        expect((await p)?.gpu).toBe("RTX 4090");
        expect(b.state.autotune).toMatchObject({ running: false, progress: null, error: null, applied: false });
        expect(await b.applyAutotune()).toBe(true);
        expect(applyTuning).toHaveBeenCalledWith({ block_ms: 40 });
        expect(b.state.autotune.applied).toBe(true);
    });

    it("does not tune while the voice changer runs, and reports a refusal", async () => {
        invoke.mockImplementation(async (cmd: string) => {
            if (cmd === VC_COMMANDS.getAutotune) return null;
            if (cmd === VC_COMMANDS.autotune) throw "no voice selected: pick a voice before tuning";
            return null;
        });
        const b = new VoiceBench(host);
        await b.init();
        running = true;
        expect(await b.runAutotune(false)).toBeNull();
        expect(invoke).not.toHaveBeenCalledWith(VC_COMMANDS.autotune, expect.anything());
        running = false;
        expect(await b.runAutotune(false)).toBeNull();
        expect(b.state.autotune.error).toContain("no voice selected");
    });
});

describe("benchmark this PC", () => {
    it("runs the paced spec of the selected voice and keeps its summary", async () => {
        invoke.mockImplementation(async (cmd: string) => {
            if (cmd === BENCH_IPC.run) return { engine: { execution_provider_active: "cuda" }, runs: [{ status: "passed", metrics: { block_ms_p95: 4, rtf_p99: 0.35 } }] };
            return null;
        });
        const b = new VoiceBench(host);
        const s = await b.runBenchmark();
        const call = invoke.mock.calls.find(([c]) => c === BENCH_IPC.run)!;
        expect((call[1] as { spec: { engine: { slot_dir: string } } }).spec.engine.slot_dir).toBe("C:/m/alto");
        expect(s).toMatchObject({ status: "passed", blockP95: 4, rtfP99: 0.35, epActive: "cuda" });
        expect(b.state.benchmark).toMatchObject({ running: false, error: null });
    });

    it("needs a voice and a stopped engine", async () => {
        const b = new VoiceBench(host);
        settings = VoiceChangerStateSchema.parse({});
        expect(await b.runBenchmark()).toBeNull();
        settings = { ...settings, rvcSlotDir: "C:/m/alto" };
        running = true;
        expect(await b.runBenchmark()).toBeNull();
        expect(invoke).not.toHaveBeenCalled();
    });
});

describe("live recording", () => {
    it("needs the engine running, then stops with its summary", async () => {
        invoke.mockImplementation(async (cmd: string) => {
            if (cmd === VC_COMMANDS.liveRecordStop) return { seconds: 30, ticks: 700, report_dir: "C:/r/live-1", metrics: { latency_ms_p50: 70 } };
            return null;
        });
        const b = new VoiceBench(host);
        expect(await b.startLiveRecord()).toBe(false);
        running = true;
        expect(await b.startLiveRecord(90)).toBe(true);
        expect(invoke).toHaveBeenCalledWith(VC_COMMANDS.liveRecordStart, { request: { max_seconds: 90 } });
        expect(b.state.liveRecord).toMatchObject({ recording: true, maxSeconds: 90 });
        const s = await b.stopLiveRecord();
        expect(s?.report_dir).toBe("C:/r/live-1");
        expect(b.state.liveRecord).toMatchObject({ recording: false, startedAt: null, error: null });
        b.dispose();
    });

    it("collects a recording that ended because the voice changer stopped", async () => {
        vi.useFakeTimers();
        invoke.mockImplementation(async (cmd: string) => {
            if (cmd === VC_COMMANDS.liveRecordStop) return { seconds: 5, ticks: 100, report_dir: "C:/r/live-2", metrics: {} };
            return null;
        });
        const b = new VoiceBench(host);
        running = true;
        await b.startLiveRecord(600);
        running = false;
        b.onEngineStopped();
        await vi.advanceTimersByTimeAsync(3000);
        expect(invoke).toHaveBeenCalledWith(VC_COMMANDS.liveRecordStop);
        expect(b.state.liveRecord.summary?.report_dir).toBe("C:/r/live-2");
        b.dispose();
    });

    it("runs through an engine restart (Rust keeps recording through a stop under 2 s)", async () => {
        vi.useFakeTimers();
        invoke.mockImplementation(async (cmd: string) => {
            if (cmd === VC_COMMANDS.liveRecordStop) return { seconds: 120, ticks: 9000, report_dir: "C:/r/live-3", metrics: {} };
            return null;
        });
        const stops = () => invoke.mock.calls.filter(([c]) => c === VC_COMMANDS.liveRecordStop).length;
        const b = new VoiceBench(host);
        running = true;
        await b.startLiveRecord(120);
        // A settings change restarts the engine: stop (from here, then Rust's event), 80 ms, start.
        b.onEngineStopped();
        await vi.advanceTimersByTimeAsync(50);
        b.onEngineStopped();
        await vi.advanceTimersByTimeAsync(80);
        b.onEngineStarted();
        await vi.advanceTimersByTimeAsync(10_000);
        expect(stops()).toBe(0);
        expect(b.state.liveRecord.recording).toBe(true);
        // Rust ends it at max_seconds; it is collected then.
        await vi.advanceTimersByTimeAsync(120_000);
        expect(stops()).toBe(1);
        expect(b.state.liveRecord).toMatchObject({ recording: false, summary: { report_dir: "C:/r/live-3" } });
        b.dispose();
    });

    it("a start later than the grace does not keep the recording (Rust ended it): it is collected", async () => {
        vi.useFakeTimers();
        invoke.mockImplementation(async (cmd: string) => {
            if (cmd === VC_COMMANDS.liveRecordStop) return { seconds: 40, ticks: 3000, report_dir: "C:/r/live-4", metrics: {} };
            return null;
        });
        const b = new VoiceBench(host);
        running = true;
        await b.startLiveRecord(600);
        b.onEngineStopped();
        await vi.advanceTimersByTimeAsync(2_200);
        b.onEngineStarted();
        await vi.advanceTimersByTimeAsync(500);
        expect(invoke).toHaveBeenCalledWith(VC_COMMANDS.liveRecordStop);
        expect(b.state.liveRecord).toMatchObject({ recording: false, summary: { report_dir: "C:/r/live-4" } });
        // A start with no recording (or no pending stop) does nothing.
        b.onEngineStarted();
        b.dispose();
    });
});
