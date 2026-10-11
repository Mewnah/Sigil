/**
 * The choices the UI offers (`options.ts`): every value this build runs, and a stored value it cannot
 * run stays visible with what runs instead (no silent remapping), mirroring Rust's stand-in, base-rung
 * and lever-applicability rules (`params.rs`, `capabilities.rs`, `denoise::effective_kind`).
 */
import { describe, expect, it } from "vitest";

import capabilitiesRs from "../../../../../src-tauri/src/services/voice_changer/capabilities.rs?raw";
import { EXECUTION_PROVIDERS, VC_LEVERS } from "../contract";
import { BASELINE_CAPABILITIES, normalizeEffectiveLevers, type VcCapabilities } from "../ipc";
import {
    denoiserOptions,
    effectiveDenoiser,
    executionProviderOptions,
    generalLeverNotes,
    gpuPrioritySupport,
    leverApplies,
    leverViews,
    nvidiaAdapterPresent,
    pitchExtractorOptions,
    precisionOptions,
    profileOptions,
    STREAMER_FEW_CORES_MAX,
    STREAMER_KEEPS_UP_BLOCK_MS,
    STREAMER_KEEPS_UP_BLOCK_MS_FEW,
    streamerFewCoresNote,
    streamerKeepsUpBlockMs,
} from "../options";

const caps = (over: Partial<VcCapabilities> = {}): VcCapabilities => ({ ...BASELINE_CAPABILITIES, ...over });

describe("offered choices", () => {
    it("F0 methods: the build's, plus a stored one it cannot run, marked with its stand-in", () => {
        const base = pitchExtractorOptions(caps(), "fcpe", "cuda");
        expect(base.map((o) => o.value)).toEqual(["fcpe", "rmvpe", "dio"]);
        expect(base.every((o) => o.available && o.runsAs === null)).toBe(true);

        const stored = pitchExtractorOptions(caps(), "crepe", "cuda");
        expect(stored.map((o) => o.value)).toEqual(["fcpe", "rmvpe", "dio", "crepe"]);
        expect(stored.find((o) => o.value === "crepe")).toMatchObject({ available: false, runsAs: "rmvpe" });
        // On the CPU the stand-in rmvpe itself runs as fcpe (the GPU-only rule).
        expect(pitchExtractorOptions(caps(), "crepe", "cpu").find((o) => o.value === "crepe")?.runsAs).toBe("fcpe");
        expect(pitchExtractorOptions(caps(), "rmvpe", "cpu").find((o) => o.value === "rmvpe")).toMatchObject({
            available: true,
            runsAs: "fcpe",
            gpuOnly: true,
        });

        const full = caps({ pitch_extractors: ["fcpe", "rmvpe", "dio", "swiftf0", "crepe", "pesto", "fusion"] });
        expect(pitchExtractorOptions(full, "fusion", "dml").map((o) => o.value)).toEqual(["fcpe", "rmvpe", "dio", "swiftf0", "crepe", "pesto", "fusion"]);
        expect(pitchExtractorOptions(full, "fusion", "dml").every((o) => o.available)).toBe(true);
    });

    it("execution providers: an accelerator the build cannot run builds on its base rung", () => {
        expect(executionProviderOptions(caps(), "cuda").map((o) => o.value)).toEqual([...EXECUTION_PROVIDERS]);
        const trt = executionProviderOptions(caps(), "tensorrt_rtx").find((o) => o.value === "tensorrt_rtx");
        expect(trt).toMatchObject({ available: false, runsAs: "cuda" });
        const hp = executionProviderOptions(caps({ rvc_execution_providers: ["cpu", "cuda", "dml", "dml_hp"] }), "cpu");
        expect(hp.map((o) => o.value)).toEqual(["cpu", "cuda", "dml", "dml_hp"]);
        expect(hp.find((o) => o.value === "dml_hp")).toMatchObject({ available: true, runsAs: null });
    });

    it("presets: Studio-TRT only with TensorRT-RTX, or stored (then Quality on CUDA)", () => {
        expect(profileOptions(caps(), "streamer").map((o) => o.value)).toEqual(["streamer", "low_latency", "quality"]);
        expect(profileOptions(caps(), "studio_trt").find((o) => o.value === "studio_trt")).toMatchObject({ available: false, runsAs: "cuda" });
        const withTrt = caps({ rvc_presets: ["streamer", "low_latency", "quality", "studio_trt"] });
        expect(profileOptions(withTrt, "quality").find((o) => o.value === "studio_trt")).toMatchObject({ available: true, runsAs: null });
    });

    it("precision and denoisers follow the build, with Rust's denoiser stand-in", () => {
        expect(precisionOptions(caps(), "auto").map((o) => o.value)).toEqual(["auto", "fp32"]);
        expect(precisionOptions(caps(), "fp16").find((o) => o.value === "fp16")).toMatchObject({ available: false, runsAs: "fp32" });
        const c = caps({ denoisers: ["off", "gtcrn", "dpdfnet"] });
        expect(denoiserOptions(c, "off").map((o) => o.value)).toEqual(["off", "gtcrn", "dpdfnet"]);
        expect(denoiserOptions(c, "dfn3").find((o) => o.value === "dfn3")).toMatchObject({ available: false, runsAs: "dpdfnet" });
        expect(effectiveDenoiser("dfn3", ["off"])).toBe("off");
        expect(effectiveDenoiser("gtcrn", ["off"])).toBe("off");
        expect(effectiveDenoiser("gtcrn", ["off", "gtcrn"])).toBe("gtcrn");
    });
});

describe("levers", () => {
    it("apply on the rungs Rust's lever_applies names", () => {
        // The rules as written in capabilities.rs (GPU retrieval / IOBinding need a GPU, CUDA graphs CUDA).
        expect(capabilitiesRs).toContain("Lever::GpuKnn | Lever::IoBinding => base != \"cpu\"");
        expect(capabilitiesRs).toContain("Lever::CudaGraphs => base == \"cuda\"");
        for (const base of EXECUTION_PROVIDERS) {
            expect(leverApplies("gpu_knn", base)).toBe(base !== "cpu");
            expect(leverApplies("io_binding", base)).toBe(base !== "cpu");
            expect(leverApplies("cuda_graphs", base)).toBe(base === "cuda");
            expect(leverApplies("gpu_priority", base)).toBe(base !== "cpu");
            for (const l of ["partial_decode", "anchored_noise", "nsf_phase0", "f0_cache", "governor"] as const) {
                expect(leverApplies(l, base), `${l} ${base}`).toBe(true);
            }
        }
    });

    it("GPU priority applies only where its queue runs (Rust: CUDA stream on an NVIDIA adapter, the DirectML queue)", () => {
        // The rules as written in capabilities.rs, which the derivation below mirrors.
        const rs = capabilitiesRs.replace(/\s+/g, " ");
        expect(rs).toContain('Lever::GpuPriority => match base { "cuda" => gpu::cuda_stream::available(), "dml" => gpu::dml_queue::available(), _ => false, }');
        expect(rs).toContain("Lever::GpuPriority => gpu::cuda_stream::available() || gpu::dml_queue::available()");
        expect(rs).toContain("Accelerator::DmlHighPriority => cfg!(windows) && gpu::dml_queue::available()");

        const withDmlQueue = ["cpu", "cuda", "dml", "dml_hp"] as VcCapabilities["rvc_execution_providers"];
        // AMD / Intel PC: the DirectML queue, no NVIDIA adapter (system info).
        const amd = gpuPrioritySupport(caps({ levers: ["gpu_priority"], rvc_execution_providers: withDmlQueue }), false);
        expect(amd).toEqual({ cuda: false, dml: true });
        expect(leverApplies("gpu_priority", "cuda", amd)).toBe(false);
        expect(leverApplies("gpu_priority", "dml", amd)).toBe(true);
        expect(leverApplies("gpu_priority", "cpu", amd)).toBe(false);
        // Offered without the DirectML queue: only the CUDA stream can be the reason.
        expect(gpuPrioritySupport(caps({ levers: ["gpu_priority"] }), null)).toEqual({ cuda: true, dml: false });
        // TensorRT-RTX needs an NVIDIA adapter too.
        const trt = caps({ levers: ["gpu_priority"], rvc_execution_providers: [...withDmlQueue, "tensorrt_rtx"] });
        expect(gpuPrioritySupport(trt, null)).toEqual({ cuda: true, dml: true });
        // Not offered: neither queue runs.
        expect(gpuPrioritySupport(caps(), true)).toEqual({ cuda: false, dml: false });
        // Unknown adapter: not reported as off.
        const unknown = gpuPrioritySupport(caps({ levers: ["gpu_priority"], rvc_execution_providers: withDmlQueue }), null);
        expect(unknown).toEqual({ cuda: null, dml: true });
        expect(leverApplies("gpu_priority", "cuda", unknown)).toBe(true);

        // The panel: "not for this EP" on CUDA without an NVIDIA adapter.
        const amdCaps = caps({ levers: ["gpu_priority"], rvc_execution_providers: withDmlQueue });
        expect(leverViews(["gpu_priority"], amdCaps, "cuda", null, false)[0]).toMatchObject({ available: true, applies: false });
        expect(leverViews(["gpu_priority"], amdCaps, "dml_hp", null, false)[0]).toMatchObject({ available: true, applies: true });
        expect(leverViews(["gpu_priority"], amdCaps, "cuda", null, true)[0]).toMatchObject({ applies: true });
    });

    it("an NVIDIA adapter as cuda_stream sees it: hardware NVIDIA in the DXGI list", () => {
        expect(nvidiaAdapterPresent(null)).toBeNull();
        expect(nvidiaAdapterPresent({ gpus: [], gpu_probe: "missing" })).toBeNull();
        expect(nvidiaAdapterPresent({ gpus: [{ vendor_id: 0x1002, software: false }] })).toBe(false);
        expect(nvidiaAdapterPresent({ gpus: [{ vendor_id: 0x10de, software: true }] })).toBe(false);
        expect(nvidiaAdapterPresent({ gpus: [{ vendor_id: 0x8086, software: false }, { vendor_id: 0x10de, software: false }] })).toBe(true);
    });

    it("report availability, applicability and what the running engine uses", () => {
        const c = caps({ levers: ["partial_decode", "cuda_graphs", "governor"] });
        const stopped = leverViews(VC_LEVERS, c, "dml", null);
        expect(stopped.find((v) => v.lever === "cuda_graphs")).toMatchObject({ available: true, applies: false, inUse: null });
        expect(stopped.find((v) => v.lever === "gpu_knn")).toMatchObject({ available: false, applies: true });
        const running = normalizeEffectiveLevers({
            partial_decode: true,
            governor: false,
            accelerator: "none",
            precision: "fp16",
            notes: ["governor: not available in this build", "tensorrt_rtx: not available in this build; running on cuda"],
        });
        const views = leverViews(VC_LEVERS, c, "tensorrt_rtx", running);
        expect(views.find((v) => v.lever === "partial_decode")).toMatchObject({ inUse: true, note: null, applies: true });
        expect(views.find((v) => v.lever === "governor")).toMatchObject({ inUse: false, note: "not available in this build" });
        expect(generalLeverNotes(running, VC_LEVERS)).toEqual(["tensorrt_rtx: not available in this build; running on cuda"]);
        expect(generalLeverNotes(null, VC_LEVERS)).toEqual([]);
    });
});

describe("the Streamer on few physical cores (vc-cpu-rerun)", () => {
    const base = { profile: "streamer", ep: "", physicalCores: 6, blockMs: 0 } as const;

    it("warns on 1 to 6 physical cores with the Streamer on the processor and Block size on Auto", () => {
        expect(STREAMER_FEW_CORES_MAX).toBe(6);
        for (const cores of [1, 2, 4, 5]) expect(streamerFewCoresNote({ ...base, physicalCores: cores })).toEqual({ cores, blockMs: 200 });
        expect(streamerFewCoresNote({ ...base, physicalCores: 6 })).toEqual({ cores: 6, blockMs: 160 });
        // The 8-core dev rig kept up under the game stand-in; 7 is not measured and gets no note.
        for (const cores of [7, 8, 12, 16]) expect(streamerFewCoresNote({ ...base, physicalCores: cores })).toBeNull();
    });

    it("says nothing while the core count is unknown", () => {
        for (const cores of [null, undefined, 0, -1, Number.NaN]) {
            expect(streamerFewCoresNote({ ...base, physicalCores: cores })).toBeNull();
        }
    });

    it("follows the rung the preset runs on: an unset EP is the preset's CPU, a GPU rung is not the Streamer's problem", () => {
        expect(streamerFewCoresNote({ ...base, ep: "cpu" })).toEqual({ cores: 6, blockMs: 160 });
        for (const ep of ["cuda", "dml", "tensorrt_rtx"] as const) expect(streamerFewCoresNote({ ...base, ep })).toBeNull();
        for (const profile of ["low_latency", "quality", "studio_trt"] as const) {
            // Their unset EP is a GPU rung; on an explicit CPU they are not the Streamer's shape (no note).
            expect(streamerFewCoresNote({ ...base, profile })).toBeNull();
            expect(streamerFewCoresNote({ ...base, profile, ep: "cpu" })).toBeNull();
        }
    });

    it("hides once Block size is long enough to keep up on a 6-core PC (160 ms, measured)", () => {
        expect(STREAMER_KEEPS_UP_BLOCK_MS).toBe(160);
        for (const blockMs of [0, 20, 60, 120, 159]) expect(streamerFewCoresNote({ ...base, blockMs })).toEqual({ cores: 6, blockMs: 160 });
        for (const blockMs of [160, 200, 400]) expect(streamerFewCoresNote({ ...base, blockMs })).toBeNull();
    });

    it("does not call 160 ms enough on 4 or 5 cores, where only 200 ms was measured", () => {
        // Benchmark: 160 ms was run on 6 cores only; 4 cores were run at 200 ms (nothing dropped under the game stand-in),
        // 5 cores on no longer block. A 4-core PC at 160 ms would lose the note with no measurement behind the all-clear.
        expect(STREAMER_KEEPS_UP_BLOCK_MS_FEW).toBe(200);
        for (const cores of [1, 2, 3, 4, 5]) {
            expect(streamerKeepsUpBlockMs(cores)).toBe(200);
            for (const blockMs of [0, 120, 160, 199]) expect(streamerFewCoresNote({ ...base, physicalCores: cores, blockMs })).toEqual({ cores, blockMs: 200 });
            for (const blockMs of [200, 240]) expect(streamerFewCoresNote({ ...base, physicalCores: cores, blockMs })).toBeNull();
        }
        expect(streamerKeepsUpBlockMs(6)).toBe(160);
    });
});
