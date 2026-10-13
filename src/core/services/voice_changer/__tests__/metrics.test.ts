import { describe, expect, it } from "vitest";

import {
    assessMetrics,
    epDisplayName,
    effectiveLatencyBudgetMs,
    epStatus,
    formatChunkMs,
    formatMs,
    isVoiceChangerEpComponent,
    normalizeMetrics,
    raisedOutputBuffer,
    type VoiceChangerMetrics,
} from "../metrics";
import { findSlot, normalizeSlots, sameSlotPath } from "../slots";

const sample = (over: Partial<VoiceChangerMetrics> = {}): VoiceChangerMetrics => ({
    buffer_ms: 40,
    process_ms: 4,
    chunk_ms: 10.7,
    process_p99_ms: 6,
    backlog_ms: 5,
    dropped_ms: 0,
    underruns: 0,
    ep: { requested: "cuda", active: "cuda", reason: null },
    process_p50_ms: null,
    process_p95_ms: null,
    e2e_ms: null,
    vram_mb: null,
    governor_level: null,
    levers: null,
    cushion_ms: null,
    buffer_prefill_ms: null,
    buffer_max_queue_ms: null,
    engine_block_ms: null,
    call_p50_ms: null,
    call_p99_ms: null,
    ...over,
});

describe("metrics normalization", () => {
    it("an older backend's reply has null P0 fields", () => {
        expect(normalizeMetrics({ buffer_ms: 12.5, process_ms: 3, chunk_ms: 10 })).toEqual({
            buffer_ms: 12.5,
            process_ms: 3,
            chunk_ms: 10,
            process_p99_ms: null,
            backlog_ms: null,
            dropped_ms: null,
            underruns: null,
            ep: null,
            process_p50_ms: null,
            process_p95_ms: null,
            e2e_ms: null,
            vram_mb: null,
            governor_level: null,
            levers: null,
            cushion_ms: null,
            buffer_prefill_ms: null,
            buffer_max_queue_ms: null,
            engine_block_ms: null,
            call_p50_ms: null,
            call_p99_ms: null,
        });
    });

    it("reads the model-call percentiles of a long-block voice and drops nonsense", () => {
        const read = (v: unknown) => {
            const m = normalizeMetrics({ buffer_ms: 1, process_ms: 2, chunk_ms: 10, call_p50_ms: v, call_p99_ms: v });
            return [m?.call_p50_ms, m?.call_p99_ms];
        };
        expect(read(150)).toEqual([150, 150]);
        expect(read(null)).toEqual([null, null]);
        expect(read(undefined)).toEqual([null, null]);
        expect(read("150")).toEqual([null, null]);
        expect(read(Number.NaN)).toEqual([null, null]);
    });

    it("reads the engine block the playing voice converts per call and drops nonsense", () => {
        const read = (v: unknown) => normalizeMetrics({ buffer_ms: 1, process_ms: 2, chunk_ms: 10, engine_block_ms: v })?.engine_block_ms;
        expect(read(160)).toBe(160);
        expect(read(20)).toBe(20);
        expect(read(null)).toBeNull();
        expect(read(undefined)).toBeNull();
        expect(read("160")).toBeNull();
        expect(read(Number.NaN)).toBeNull();
        expect(read(-5)).toBe(0);
    });

    it("reads the output buffer in force and drops nonsense", () => {
        const m = normalizeMetrics({ buffer_ms: 1, process_ms: 2, chunk_ms: 10, buffer_prefill_ms: 110, buffer_max_queue_ms: 136 });
        expect([m?.buffer_prefill_ms, m?.buffer_max_queue_ms]).toEqual([110, 136]);
        const bad = normalizeMetrics({ buffer_ms: 1, process_ms: 2, chunk_ms: 10, buffer_prefill_ms: "110", buffer_max_queue_ms: null });
        expect([bad?.buffer_prefill_ms, bad?.buffer_max_queue_ms]).toEqual([null, null]);
    });

    it("tells a buffer the voice raised from the stored one", () => {
        const inForce = (prefill: number | null, queue: number | null) => sample({ buffer_prefill_ms: prefill, buffer_max_queue_ms: queue });
        // The Low latency template under a CPU voice's 110 / 136 ms floor, and the Stable one (queue kept).
        expect(raisedOutputBuffer(inForce(110, 136), 35, 70)).toEqual({ prefillMs: 110, maxQueueMs: 136 });
        expect(raisedOutputBuffer(inForce(110, 150), 80, 150)).toEqual({ prefillMs: 110, maxQueueMs: 150 });
        // Only the queue raised.
        expect(raisedOutputBuffer(inForce(110, 136), 120, 130)).toEqual({ prefillMs: 110, maxQueueMs: 136 });
        // Nothing raised: the in-force numbers are the stored ones (also when Rust clamped the stored queue up).
        expect(raisedOutputBuffer(inForce(35, 70), 35, 70)).toBeNull();
        expect(raisedOutputBuffer(inForce(110, 136), 110, 136)).toBeNull();
        expect(raisedOutputBuffer(inForce(60, 65), 60, 65)).toBeNull();
        expect(raisedOutputBuffer(inForce(40, 45), 40, 41)).toBeNull();
        expect(raisedOutputBuffer(inForce(300, 600), 400, 900)).toBeNull();
        // Stopped, or an older backend: no claim.
        expect(raisedOutputBuffer(null, 35, 70)).toBeNull();
        expect(raisedOutputBuffer(inForce(null, null), 35, 70)).toBeNull();
        expect(raisedOutputBuffer(inForce(110, null), 35, 70)).toBeNull();
    });

    it("reads the output cushion the queue holds (P3B-H) and drops nonsense", () => {
        expect(normalizeMetrics({ buffer_ms: 1, process_ms: 2, chunk_ms: 10, cushion_ms: 17.5 })?.cushion_ms).toBe(17.5);
        expect(normalizeMetrics({ buffer_ms: 1, process_ms: 2, chunk_ms: 10, cushion_ms: null })?.cushion_ms).toBeNull();
        expect(normalizeMetrics({ buffer_ms: 1, process_ms: 2, chunk_ms: 10, cushion_ms: "17" })?.cushion_ms).toBeNull();
        expect(normalizeMetrics({ buffer_ms: 1, process_ms: 2, chunk_ms: 10, cushion_ms: -3 })?.cushion_ms).toBe(0);
    });

    it("reads the P3 fields and normalizes the levers in use", () => {
        const m = normalizeMetrics({
            buffer_ms: 1,
            process_ms: 2,
            chunk_ms: 10,
            process_p50_ms: 3.5,
            process_p95_ms: 5,
            e2e_ms: 61,
            vram_mb: 540,
            governor_level: 1,
            levers: { cuda_graphs: true, partial_decode: "yes", accelerator: "tensorrt_rtx", precision: "FP16", notes: ["x", 3] },
        })!;
        expect([m.process_p50_ms, m.process_p95_ms, m.e2e_ms, m.vram_mb, m.governor_level]).toEqual([3.5, 5, 61, 540, 1]);
        expect(m.levers).toMatchObject({ cuda_graphs: true, partial_decode: false, governor: false, accelerator: "tensorrt_rtx", precision: "fp16", notes: ["x"] });
        expect(normalizeMetrics({ buffer_ms: 1, process_ms: 2, chunk_ms: 3, levers: "bogus" })!.levers).toBeNull();
    });

    it("keeps the P0 fields and normalizes the EP", () => {
        const m = normalizeMetrics({
            buffer_ms: 1,
            process_ms: 2,
            chunk_ms: 3,
            process_p99_ms: 4,
            backlog_ms: 5,
            dropped_ms: 6,
            underruns: 7,
            ep: { requested: "CUDA", active: "cpu", reason: "  cudnn64_9.dll not found " },
        });
        expect(m).toMatchObject({ process_p99_ms: 4, backlog_ms: 5, dropped_ms: 6, underruns: 7 });
        expect(m?.ep).toEqual({ requested: "cuda", active: "cpu", reason: "cudnn64_9.dll not found" });
    });

    it("rejects garbage without throwing", () => {
        expect(normalizeMetrics(null)).toBeNull();
        expect(normalizeMetrics("x")).toBeNull();
        const m = normalizeMetrics({ buffer_ms: Number.NaN, process_ms: -1, chunk_ms: "10", ep: { active: 1 } });
        expect(m).toMatchObject({ buffer_ms: 0, process_ms: 0, chunk_ms: 0, ep: null });
        expect(normalizeMetrics({ ep: { requested: "dml", active: "dml", reason: "" } })?.ep?.reason).toBeNull();
    });
});

describe("EP status", () => {
    it("tells a granted EP from a fallback", () => {
        expect(epStatus(null)).toBeNull();
        expect(epStatus({ requested: "dml", active: "dml", reason: null })).toEqual({ kind: "granted", active: "dml" });
        expect(epStatus({ requested: "cuda", active: "dml", reason: "CUDA runtime missing" })).toEqual({
            kind: "fallback",
            requested: "cuda",
            active: "dml",
            reason: "CUDA runtime missing",
        });
    });

    it("names EPs and recognizes voice-changer components", () => {
        expect(epDisplayName("cpu")).toBe("CPU");
        expect(epDisplayName("CUDA")).toBe("CUDA");
        expect(epDisplayName("dml")).toBe("DirectML");
        expect(epDisplayName("tensorrt")).toBe("TensorRT");
        expect(epDisplayName("openvino")).toBe("openvino");
        expect(isVoiceChangerEpComponent("vc")).toBe(true);
        expect(isVoiceChangerEpComponent("vc.pitch")).toBe(true);
        expect(isVoiceChangerEpComponent("vcx")).toBe(false);
        expect(isVoiceChangerEpComponent("tts.kitten")).toBe(false);
    });
});

describe("metrics assessment", () => {
    it("is starting until the block length is known", () => {
        expect(assessMetrics(sample({ chunk_ms: 0, process_ms: 0 }), 300).starting).toBe(true);
        expect(assessMetrics(sample({ chunk_ms: 0, process_ms: 0 }), 300).realtime).toBe("ok");
    });

    it("is starting while the model loads: rates published, no block processed yet", () => {
        // What Rust reports during a slow RVC / CUDA load: rates (so the block length) are already set,
        // the per-run counters are zero, nothing is queued for output and there is no EP yet.
        const loading = sample({
            chunk_ms: 10.7,
            process_ms: 0,
            process_p99_ms: 0,
            buffer_ms: 0,
            backlog_ms: 0,
            dropped_ms: 0,
            underruns: 0,
            ep: null,
        });
        expect(assessMetrics(loading, 300)).toMatchObject({ starting: true, realtime: "ok", backlogHigh: false });
        // An older backend without p99.
        expect(assessMetrics({ ...loading, process_p99_ms: null }, 300).starting).toBe(true);
        // The first converted block ends it.
        expect(assessMetrics({ ...loading, process_ms: 0.4, process_p99_ms: 0.4 }, 300).starting).toBe(false);
        // Something already reached the output queue: running, not starting.
        expect(assessMetrics({ ...loading, buffer_ms: 5 }, 300).starting).toBe(false);
        expect(assessMetrics(sample(), 300).starting).toBe(false);
    });

    it("separates a slow average from slow spikes", () => {
        expect(assessMetrics(sample(), 300).realtime).toBe("ok");
        expect(assessMetrics(sample({ process_p99_ms: 12 }), 300).realtime).toBe("spikes");
        expect(assessMetrics(sample({ process_ms: 11, process_p99_ms: 15 }), 300).realtime).toBe("slow");
        expect(assessMetrics(sample({ process_p99_ms: null }), 300).realtime).toBe("ok");
    });

    it("judges a voice that converts long blocks by its block, not by the 10 ms device tick", () => {
        // The Streamer on the processor, 160 ms chunks: the tick that calls the model takes ~100 ms, the ticks between
        // cost almost nothing, and the moving average spikes right after each call.
        const burst = (over: Partial<VoiceChangerMetrics> = {}) =>
            sample({ engine_block_ms: 160, chunk_ms: 10.67, process_ms: 22, process_p50_ms: 0.4, process_p95_ms: 98, process_p99_ms: 112, ...over });
        // Healthy: calls finish well inside the block. Judged by the device tick this said "slow" after every call.
        expect(assessMetrics(sample({ chunk_ms: 10.67, process_ms: 22, process_p95_ms: 98, process_p99_ms: 112 }), 300).realtime).toBe("slow");
        expect(assessMetrics(burst(), 300).realtime).toBe("ok");
        // The typical call takes the whole block or more: it falls behind.
        expect(assessMetrics(burst({ process_p95_ms: 160 }), 300).realtime).toBe("slow");
        expect(assessMetrics(burst({ process_p95_ms: 175, process_p99_ms: 190 }), 300).realtime).toBe("slow");
        // Only the slowest calls overrun: spikes.
        expect(assessMetrics(burst({ process_p95_ms: 120, process_p99_ms: 165 }), 300).realtime).toBe("spikes");
        // An older backend without the percentiles: the p99 / the average stand in.
        expect(assessMetrics(burst({ process_p95_ms: null, process_p99_ms: 120 }), 300).realtime).toBe("ok");
        expect(assessMetrics(burst({ process_p95_ms: null, process_p99_ms: 170 }), 300).realtime).toBe("slow");
        // A block no longer than one device tick (the app's 20 ms against a 21 ms buffer), no engine block (a
        // bring-your-own voice, a backend without it): the per-tick rules.
        expect(assessMetrics(sample({ engine_block_ms: 10.7, process_ms: 11, process_p99_ms: 15 }), 300).realtime).toBe("slow");
        expect(assessMetrics(sample({ engine_block_ms: 10, process_p99_ms: 12 }), 300).realtime).toBe("spikes");
        expect(assessMetrics(sample({ engine_block_ms: null, process_ms: 11, process_p99_ms: 15 }), 300).realtime).toBe("slow");
        // Still starting while nothing was converted.
        expect(assessMetrics(burst({ process_ms: 0, process_p99_ms: 0, process_p95_ms: 0, buffer_ms: 0 }), 300).starting).toBe(true);
    });

    it("judges a long block by its model calls when the backend times them: 10 % overrunning calls read 'ok' by the ticks and 'spikes' by the calls", () => {
        // The 4-core Streamer on the processor: 200 ms blocks on 10.7 ms ticks. One tick in nineteen runs the model, so
        // the tick p95 sits among the quickest calls and the tick p99 below the block; 10 % of the calls take 230 ms.
        const ticks = sample({ engine_block_ms: 200, chunk_ms: 10.67, process_ms: 8, process_p50_ms: 0.3, process_p95_ms: 40, process_p99_ms: 150 });
        expect(assessMetrics(ticks, 300)).toMatchObject({ realtime: "ok", byCalls: false });
        const calls = { ...ticks, call_p50_ms: 150, call_p99_ms: 230 };
        expect(assessMetrics(calls, 300)).toMatchObject({ realtime: "spikes", byCalls: true });
        // Healthy calls: the median and the p99 both inside the block.
        expect(assessMetrics({ ...ticks, call_p50_ms: 130, call_p99_ms: 190 }, 300)).toMatchObject({ realtime: "ok", byCalls: true });
        // The typical call takes the whole block: it falls behind for good.
        expect(assessMetrics({ ...ticks, call_p50_ms: 205, call_p99_ms: 260 }, 300).realtime).toBe("slow");
        // One without the other, or no long block: the per-tick judgement stays.
        expect(assessMetrics({ ...ticks, call_p50_ms: 150, call_p99_ms: null }, 300)).toMatchObject({ realtime: "ok", byCalls: false });
        expect(assessMetrics({ ...calls, engine_block_ms: null }, 300).byCalls).toBe(false);
        expect(assessMetrics({ ...calls, engine_block_ms: 10.5 }, 300).byCalls).toBe(false);
    });

    it("keeps the latency budget at least one engine block plus two device buffers, as Rust does", () => {
        const long = sample({ engine_block_ms: 120, chunk_ms: 10.67 });
        // Stored 100 ms: raised to 120 + 21.3.
        expect(effectiveLatencyBudgetMs(100, long)).toBeCloseTo(141.34, 2);
        expect(effectiveLatencyBudgetMs(300, long)).toBe(300);
        expect(effectiveLatencyBudgetMs(100, sample({ engine_block_ms: null, chunk_ms: 10.67 }))).toBe(100);
        expect(effectiveLatencyBudgetMs(100, null)).toBe(100);
        expect(effectiveLatencyBudgetMs(5, sample({ engine_block_ms: null, chunk_ms: 10.67 }))).toBe(10.67);
        // A backlog of one call's worth (~130 ms) is no warning against a 100 ms budget with 120 ms blocks, nor is 112 ms.
        expect(assessMetrics(sample({ ...long, backlog_ms: 112 }), 100).backlogHigh).toBe(false);
        // It is once it nears the budget in force.
        expect(assessMetrics(sample({ ...long, backlog_ms: 115 }), 100).backlogHigh).toBe(true);
        expect(assessMetrics(sample({ engine_block_ms: null, chunk_ms: 10.67, backlog_ms: 85 }), 100).backlogHigh).toBe(true);
    });

    it("formats a chunk in whole ms when it is one", () => {
        expect([20, 80, 100, 160, 200, 19.98, 20.03].map(formatChunkMs)).toEqual(["20", "80", "100", "160", "200", "20", "20"]);
        expect(formatChunkMs(10.67)).toBe("10.7");
        expect(formatChunkMs(83.3)).toBe("83.3");
        expect(formatChunkMs(Number.NaN)).toBe("–");
    });

    it("flags a backlog near the budget (never below one block)", () => {
        expect(assessMetrics(sample({ backlog_ms: 239 }), 300).backlogHigh).toBe(false);
        expect(assessMetrics(sample({ backlog_ms: 240 }), 300).backlogHigh).toBe(true);
        // A block longer than the budget: the backlog may hold one block.
        expect(assessMetrics(sample({ chunk_ms: 200, backlog_ms: 150 }), 100).backlogHigh).toBe(false);
        expect(assessMetrics(sample({ backlog_ms: null }), 300).backlogHigh).toBe(false);
    });

    it("reports drops and underruns only while they grow", () => {
        const prev = sample({ dropped_ms: 40, underruns: 2 });
        expect(assessMetrics(sample({ dropped_ms: 40, underruns: 2 }), 300, prev)).toMatchObject({
            droppingNow: false,
            underrunNow: false,
        });
        expect(assessMetrics(sample({ dropped_ms: 60, underruns: 3 }), 300, prev)).toMatchObject({
            droppingNow: true,
            underrunNow: true,
        });
        // A new run restarts the counters: not a drop.
        expect(assessMetrics(sample({ dropped_ms: 0 }), 300, prev).droppingNow).toBe(false);
        expect(assessMetrics(sample({ dropped_ms: 60 }), 300, null).droppingNow).toBe(false);
    });

    it("formats milliseconds", () => {
        expect(formatMs(3.14159)).toBe("3.1");
        expect(formatMs(250.4)).toBe("250");
        expect(formatMs(Number.NaN)).toBe("–");
    });
});

describe("voice slots", () => {
    it("keeps only well-formed rows", () => {
        expect(normalizeSlots(null)).toEqual([]);
        expect(
            normalizeSlots([
                { id: "alto", path: "C:\\m\\alto", has_manifest: true },
                { id: "bass", path: "C:\\m\\bass" },
                { id: "bad", path: " " },
                { path: "C:\\m\\noid" },
                "x",
            ]),
        ).toEqual([
            { id: "alto", path: "C:\\m\\alto", has_manifest: true },
            { id: "bass", path: "C:\\m\\bass", has_manifest: false },
        ]);
    });

    it("matches folders regardless of separators, trailing slashes and case", () => {
        expect(sameSlotPath("C:\\Voices\\Alto", "c:/voices/alto/")).toBe(true);
        expect(sameSlotPath(" C:\\voices\\alto\\ ", "C:\\voices\\alto")).toBe(true);
        expect(sameSlotPath("C:\\voices\\alto", "C:\\voices\\alto2")).toBe(false);
        expect(sameSlotPath("", "")).toBe(false);
        const slots = normalizeSlots([
            { id: "alto", path: "C:\\m\\alto", has_manifest: false },
            { id: "bass", path: "C:\\m\\bass", has_manifest: false },
        ]);
        expect(findSlot(slots, "c:/m/bass")?.id).toBe("bass");
        expect(findSlot(slots, "D:\\elsewhere")).toBeUndefined();
    });
});
