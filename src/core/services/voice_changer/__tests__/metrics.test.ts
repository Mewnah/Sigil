import { describe, expect, it } from "vitest";

import {
    assessMetrics,
    epDisplayName,
    epStatus,
    formatMs,
    isVoiceChangerEpComponent,
    normalizeMetrics,
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
        });
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
