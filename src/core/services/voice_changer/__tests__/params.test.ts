/**
 * The TS voice-changer lists, defaults and ranges against the frozen Rust <-> TS contract
 * (`src-tauri/src/services/voice_changer/params_contract.json`, which Rust's `contract_tests.rs`
 * checks against the Rust allow-lists), plus the params mapping in both directions.
 */
import { describe, expect, it } from "vitest";

import contract from "../../../../../src-tauri/src/services/voice_changer/params_contract.json";
import { normalizeChunkFramesValue, VOICE_CHANGER_CHUNK_FRAMES } from "../chunkFrames";
import {
    AUX_MIC_GATES,
    BASELINE_DENOISERS,
    BASELINE_PITCH_EXTRACTORS,
    BASELINE_PRECISIONS,
    BASELINE_RVC_PROFILES,
    clampLatencyBudgetMs,
    clampOutputBufferMs,
    DENOISERS,
    effectivePitchExtractor,
    effectiveRvcEp,
    EXECUTION_PROVIDER_ALIASES,
    EXECUTION_PROVIDER_BASE,
    EXECUTION_PROVIDERS,
    F0_RANGES,
    GPU_ONLY_PITCH_EXTRACTORS,
    LATENCY_PRESETS,
    LEVER_MODES,
    normalizeAuxMicGate,
    normalizeDenoiser,
    normalizeExecutionProvider,
    normalizeF0Range,
    normalizeLatencyPreset,
    normalizeLeverMode,
    normalizeMode,
    normalizeOnnxExecutionProvider,
    normalizePitchExtractor,
    normalizePrecision,
    normalizeRvcProfile,
    normalizeVocoderOversample,
    normalizeWorkerPriority,
    PITCH_EXTRACTOR_STAND_IN,
    PRECISIONS,
    profileDefaultEp,
    RVC_EXECUTION_PROVIDERS,
    rvcBaseEp,
    VC_LEVERS,
    OUTPUT_BUFFER_LOW_LATENCY,
    OUTPUT_BUFFER_STABLE,
    OUTPUT_MAX_QUEUE_MS,
    PITCH_EXTRACTOR_CPU_FALLBACK,
    PITCH_EXTRACTOR_LEGACY_ALIASES,
    PITCH_EXTRACTORS,
    RVC_PROFILES,
    VC_ENUM_DEFAULTS,
    VC_MODES,
    VC_NUMBER_RANGES,
    VOCODER_OVERSAMPLES,
    WORKER_PRIORITIES,
} from "../contract";
import { normalizeMetrics } from "../metrics";
import {
    adoptAdjustedParams,
    applyRustParams,
    LEVER_STATE_KEYS,
    RUST_PARAM_KEYS,
    sameParamValue,
    toRustParams,
    type VoiceChangerParamsRust,
} from "../params";
import {
    AuxMicGateSchema,
    DenoiserSchema,
    ExecutionProviderSchema,
    F0RangeSchema,
    LatencyPresetSchema,
    LeverModeSchema,
    PitchExtractorSchema,
    PrecisionSchema,
    RvcExecutionProviderSchema,
    RvcProfileSchema,
    VoiceChangerModeSchema,
    VoiceChangerState,
    VoiceChangerStateSchema,
    WorkerPrioritySchema,
} from "../schema";

type EnumSpec = {
    values: (string | number)[];
    default: string | number;
    fallback?: string | number;
    aliases?: Record<string, string>;
    legacy_aliases?: Record<string, string>;
    gpu_only?: string[];
    cpu_fallback?: string;
    baseline?: string[];
    stand_in?: Record<string, string>;
    base?: Record<string, string>;
    default_ep?: Record<string, string>;
};

const enums = contract.enums as unknown as Record<string, EnumSpec>;
const numbers = contract.numbers as unknown as Record<string, Record<string, number | boolean | string>>;

const defaults = (): VoiceChangerState => VoiceChangerStateSchema.parse({});

/**
 * For a string enum: listed values (any case, padded) map to themselves, aliases to their target and
 * anything else to the fallback, exactly as the Rust normalizer does.
 */
function checkStringEnum(name: string, list: readonly string[], normalize: (s: unknown) => string, schemaOptions: readonly string[]) {
    const spec = enums[name];
    expect(list, `${name}: TS list`).toEqual(spec.values);
    expect(schemaOptions, `${name}: zod options`).toEqual(spec.values);
    for (const v of spec.values as string[]) {
        expect(normalize(v), `${name}: ${v}`).toBe(v);
        expect(normalize(` ${v.toUpperCase()} `), `${name}: ${v} (upper case, padded)`).toBe(v);
    }
    for (const [alias, target] of Object.entries({ ...spec.aliases, ...spec.legacy_aliases })) {
        expect(normalize(alias), `${name}: alias ${alias}`).toBe(target);
    }
    const fallback = spec.fallback ?? spec.default;
    expect(normalize("definitely-not-a-value"), `${name}: fallback`).toBe(fallback);
    expect(normalize(undefined), `${name}: missing`).toBe(fallback);
}

describe("contract parity with Rust (params_contract.json)", () => {
    it("string enums list and normalize exactly like Rust", () => {
        checkStringEnum("mode", VC_MODES, normalizeMode, VoiceChangerModeSchema.options);
        checkStringEnum("latency_preset", LATENCY_PRESETS, normalizeLatencyPreset, LatencyPresetSchema.options);
        checkStringEnum("rvc_preset", RVC_PROFILES, normalizeRvcProfile, RvcProfileSchema.options);
        checkStringEnum("pitch_extractor", PITCH_EXTRACTORS, normalizePitchExtractor, PitchExtractorSchema.options);
        checkStringEnum("worker_priority", WORKER_PRIORITIES, normalizeWorkerPriority, WorkerPrioritySchema.options);
        checkStringEnum("aux_mic_gate", AUX_MIC_GATES, normalizeAuxMicGate, AuxMicGateSchema.options);
        checkStringEnum(
            "rvc_execution_provider",
            RVC_EXECUTION_PROVIDERS,
            (s) => normalizeExecutionProvider(s) || "cpu",
            RvcExecutionProviderSchema.options,
        );
        expect(enums.onnx_execution_provider.values).toEqual([...EXECUTION_PROVIDERS]);
        expect(ExecutionProviderSchema.options).toEqual([...EXECUTION_PROVIDERS]);
        checkStringEnum("precision", PRECISIONS, normalizePrecision, PrecisionSchema.options);
        checkStringEnum("lever_mode", LEVER_MODES, normalizeLeverMode, LeverModeSchema.options);
        checkStringEnum("denoiser", DENOISERS, normalizeDenoiser, DenoiserSchema.options);
        checkStringEnum("f0_range", F0_RANGES, normalizeF0Range, F0RangeSchema.options);
    });

    it("P3: levers, baselines, stand-ins, base rungs and preset EPs are the contract's", () => {
        expect([...VC_LEVERS]).toEqual(contract.levers);
        expect(Object.keys(LEVER_STATE_KEYS)).toEqual(contract.levers);
        expect([...BASELINE_PITCH_EXTRACTORS]).toEqual(enums.pitch_extractor.baseline);
        expect([...BASELINE_RVC_PROFILES]).toEqual(enums.rvc_preset.baseline);
        expect([...BASELINE_PRECISIONS]).toEqual(enums.precision.baseline);
        expect([...BASELINE_DENOISERS]).toEqual(enums.denoiser.baseline);
        expect([...EXECUTION_PROVIDERS]).toEqual(enums.rvc_execution_provider.baseline);
        expect(PITCH_EXTRACTOR_STAND_IN).toEqual(enums.pitch_extractor.stand_in);
        expect(EXECUTION_PROVIDER_BASE).toEqual(enums.rvc_execution_provider.base);
        for (const [ep, base] of Object.entries(enums.rvc_execution_provider.base ?? {})) {
            expect(rvcBaseEp(ep), ep).toBe(base);
            expect(normalizeOnnxExecutionProvider(ep), ep).toBe(base);
        }
        for (const [profile, ep] of Object.entries(enums.rvc_preset.default_ep ?? {})) {
            expect(profileDefaultEp(normalizeRvcProfile(profile)), profile).toBe(ep);
        }
        expect(normalizeOnnxExecutionProvider("")).toBe("cpu");
    });

    it("legacy and alias maps are the contract's, and the UI never offers an alias", () => {
        expect(PITCH_EXTRACTOR_LEGACY_ALIASES).toEqual(enums.pitch_extractor.legacy_aliases);
        expect(EXECUTION_PROVIDER_ALIASES).toEqual(enums.rvc_execution_provider.aliases);
        expect([...GPU_ONLY_PITCH_EXTRACTORS]).toEqual(enums.pitch_extractor.gpu_only);
        expect(PITCH_EXTRACTOR_CPU_FALLBACK).toBe(enums.pitch_extractor.cpu_fallback);
        for (const alias of Object.keys(PITCH_EXTRACTOR_LEGACY_ALIASES)) {
            expect(PitchExtractorSchema.options as readonly string[]).not.toContain(alias);
        }
        // An empty RVC EP means "the profile's default", as in Rust.
        expect(normalizeExecutionProvider("")).toBe("");
        expect(normalizeExecutionProvider("   ")).toBe("");
    });

    it("enum defaults match", () => {
        expect(VC_ENUM_DEFAULTS.mode).toBe(enums.mode.default);
        expect(VC_ENUM_DEFAULTS.latencyPreset).toBe(enums.latency_preset.default);
        expect(VC_ENUM_DEFAULTS.rvcProfile).toBe(enums.rvc_preset.default);
        expect(VC_ENUM_DEFAULTS.pitchExtractor).toBe(enums.pitch_extractor.default);
        expect(VC_ENUM_DEFAULTS.rvcExecutionProvider).toBe(enums.rvc_execution_provider.default);
        expect(VC_ENUM_DEFAULTS.onnxExecutionProvider).toBe(enums.onnx_execution_provider.default);
        expect(VC_ENUM_DEFAULTS.vocoderOversample).toBe(enums.vocoder_oversample.default);
        expect(VC_ENUM_DEFAULTS.workerPriority).toBe(enums.worker_priority.default);
        expect(VC_ENUM_DEFAULTS.auxMicGate).toBe(enums.aux_mic_gate.default);
        expect(VC_ENUM_DEFAULTS.precision).toBe(enums.precision.default);
        expect(VC_ENUM_DEFAULTS.leverMode).toBe(enums.lever_mode.default);
        expect(VC_ENUM_DEFAULTS.denoiser).toBe(enums.denoiser.default);
        expect(VC_ENUM_DEFAULTS.f0Range).toBe(enums.f0_range.default);
    });

    it("vocoder oversample list and fallback match", () => {
        expect([...VOCODER_OVERSAMPLES]).toEqual(enums.vocoder_oversample.values);
        for (const v of VOCODER_OVERSAMPLES) expect(normalizeVocoderOversample(v)).toBe(v);
        expect(normalizeVocoderOversample(3)).toBe(enums.vocoder_oversample.fallback);
        expect(normalizeVocoderOversample("8")).toBe(enums.vocoder_oversample.fallback);
    });

    it("chunk sizes match, and TS only ever sends a size Rust keeps", () => {
        expect([...VOICE_CHANGER_CHUNK_FRAMES]).toEqual(contract.chunk_frames.values);
        expect(defaults().chunkFrames).toBe(contract.chunk_frames.default);
        const allowed = new Set(contract.chunk_frames.values);
        for (const n of [0, 1, 161, 500, 511, 513, 999_999, -3, Number.NaN]) {
            const sent = toRustParams({ ...defaults(), chunkFrames: n as VoiceChangerState["chunkFrames"] }).chunk_frames;
            expect(allowed.has(sent), `chunk ${n} -> ${sent}`).toBe(true);
        }
        expect(normalizeChunkFramesValue(512)).toBe(512);
    });

    it("every numeric range and default matches", () => {
        const ranges = VC_NUMBER_RANGES as Record<string, { min: number; max?: number; default: number }>;
        for (const [key, spec] of Object.entries(numbers)) {
            if (key === "output_max_queue_ms") continue; // relative lower bound, checked below
            const r = ranges[key];
            expect(r, `range for ${key}`).toBeDefined();
            expect(r.min, `${key}.min`).toBe(spec.min);
            expect(r.max, `${key}.max`).toBe(spec.max);
            expect(r.default, `${key}.default`).toBe(spec.default);
        }
        // No TS-only range either.
        expect(Object.keys(ranges).sort()).toEqual(Object.keys(numbers).filter((k) => k !== "output_max_queue_ms").sort());
        expect(OUTPUT_MAX_QUEUE_MS.minAbovePrefill).toBe(numbers.output_max_queue_ms.min_above_prefill);
        expect(OUTPUT_MAX_QUEUE_MS.max).toBe(numbers.output_max_queue_ms.max);
        expect(OUTPUT_MAX_QUEUE_MS.default).toBe(numbers.output_max_queue_ms.default);
        expect(OUTPUT_BUFFER_STABLE).toEqual({
            prefillMs: numbers.output_prefill_ms.stable_preset,
            maxQueueMs: numbers.output_max_queue_ms.stable_preset,
        });
        expect(OUTPUT_BUFFER_LOW_LATENCY).toEqual({
            prefillMs: numbers.output_prefill_ms.default,
            maxQueueMs: numbers.output_max_queue_ms.default,
        });
    });

    it("clamps mirror Rust", () => {
        const b = numbers.latency_budget_ms;
        expect(clampLatencyBudgetMs(0)).toBe(b.min);
        expect(clampLatencyBudgetMs(1e9)).toBe(b.max);
        expect(clampLatencyBudgetMs(Number.NaN)).toBe(b.default);
        expect(clampLatencyBudgetMs(450)).toBe(450);
        expect(clampOutputBufferMs(1, 2)).toEqual({ prefillMs: 5, maxQueueMs: 10 });
        expect(clampOutputBufferMs(300, 100)).toEqual({ prefillMs: 300, maxQueueMs: 305 });
        expect(clampOutputBufferMs(35, 9000)).toEqual({ prefillMs: 35, maxQueueMs: 600 });
    });

    it("the schema defaults are the contract defaults on the wire", () => {
        const sent = toRustParams(defaults()) as unknown as Record<string, unknown>;
        for (const [key, spec] of Object.entries(numbers)) {
            expect(sent[key], `default ${key}`).toBe(spec.default);
        }
        expect(sent.mode).toBe(enums.mode.default);
        expect(sent.latency_preset).toBe(enums.latency_preset.default);
        expect(sent.rvc_preset).toBe(enums.rvc_preset.default);
        expect(sent.pitch_extractor).toBe(enums.pitch_extractor.default);
        expect(sent.rvc_execution_provider).toBe(enums.rvc_execution_provider.default);
        expect(sent.onnx_execution_provider).toBe(enums.onnx_execution_provider.default);
        expect(sent.vocoder_oversample).toBe(enums.vocoder_oversample.default);
        expect(sent.worker_priority).toBe(enums.worker_priority.default);
        expect(sent.aux_mic_gate).toBe(enums.aux_mic_gate.default);
        expect(sent.chunk_frames).toBe(contract.chunk_frames.default);
        expect(sent.input_device).toBe("");
        expect(sent.rvc_consent_acknowledged).toBe(false);
        expect(sent.slot_id).toBe("");
        expect(sent.precision).toBe(enums.precision.default);
        expect(sent.f0_range).toBe(enums.f0_range.default);
        expect(sent.denoiser).toBe(enums.denoiser.default);
        for (const lever of VC_LEVERS) expect(sent[lever], lever).toBe(enums.lever_mode.default);
    });

    it("every P3 params field is sent and hydrated under its contract name", () => {
        const sent = toRustParams(defaults());
        for (const key of Object.keys(contract.params_p3_new_fields)) {
            expect(RUST_PARAM_KEYS, `key list has ${key}`).toContain(key);
            expect(sent[key as keyof VoiceChangerParamsRust], `sent ${key}`).not.toBeUndefined();
        }
    });

    it("every P0 params field is sent and hydrated under its contract name", () => {
        const sent = toRustParams(defaults());
        for (const key of Object.keys(contract.params_p0_new_fields)) {
            expect(RUST_PARAM_KEYS, `key list has ${key}`).toContain(key);
            expect(sent[key as keyof VoiceChangerParamsRust], `sent ${key}`).not.toBeUndefined();
        }
        // Everything sent is in the key list (the adopt step compares all of it).
        expect(Object.keys(sent).sort()).toEqual([...RUST_PARAM_KEYS].sort());
    });

    it("every metrics field of the contract is read", () => {
        const raw: Record<string, unknown> = { buffer_ms: 1, process_ms: 2, chunk_ms: 10 };
        const m = normalizeMetrics(raw) as unknown as Record<string, unknown>;
        for (const key of [
            ...Object.keys(contract.metrics_existing_fields),
            ...Object.keys(contract.metrics_p0_new_fields),
            ...Object.keys(contract.metrics_p3_new_fields),
        ]) {
            expect(key in m, `metrics field ${key}`).toBe(true);
        }
    });
});

describe("RVC effective EP and F0 (mirror of Rust effective_rvc_ep / effective_pitch_extractor)", () => {
    it("an unset EP runs the profile default; an explicit one wins", () => {
        expect(effectiveRvcEp("", "low_latency")).toBe("cuda");
        expect(effectiveRvcEp("", "quality")).toBe("cuda");
        expect(effectiveRvcEp("", "streamer")).toBe("cpu");
        expect(effectiveRvcEp("cpu", "quality")).toBe("cpu");
        expect(effectiveRvcEp("dml", "streamer")).toBe("dml");
        // P3: Studio-TRT asks for TensorRT-RTX, which builds on the CUDA rung.
        expect(effectiveRvcEp("", "studio_trt")).toBe("tensorrt_rtx");
        expect(rvcBaseEp(effectiveRvcEp("", "studio_trt"))).toBe("cuda");
        expect(rvcBaseEp("dml_hp")).toBe("dml");
    });

    it("a GPU-only F0 runs as fcpe on the CPU only, and says so", () => {
        expect(effectivePitchExtractor("rmvpe", "cpu")).toEqual({ extractor: "fcpe", fellBack: true });
        expect(effectivePitchExtractor("RMVPE", "dml")).toEqual({ extractor: "rmvpe", fellBack: false });
        expect(effectivePitchExtractor("dio", "cpu")).toEqual({ extractor: "dio", fellBack: false });
        expect(effectivePitchExtractor("fcpe", "cuda")).toEqual({ extractor: "fcpe", fellBack: false });
        expect(effectivePitchExtractor("rmvpe", "tensorrt_rtx")).toEqual({ extractor: "rmvpe", fellBack: false });
    });

    it("P3: a method this build cannot run runs its stand-in (then the GPU-only rule), as in Rust", () => {
        // Not in the baseline (default `available`): the stand-in runs.
        expect(effectivePitchExtractor("crepe", "cuda")).toEqual({ extractor: "rmvpe", fellBack: true });
        expect(effectivePitchExtractor("crepe", "cpu")).toEqual({ extractor: "fcpe", fellBack: true });
        expect(effectivePitchExtractor("swiftf0", "cpu")).toEqual({ extractor: "fcpe", fellBack: true });
        expect(effectivePitchExtractor("fusion", "dml_hp")).toEqual({ extractor: "rmvpe", fellBack: true });
        // Once the capabilities list it, it runs itself.
        const withSwift = [...BASELINE_PITCH_EXTRACTORS, "swiftf0"];
        expect(effectivePitchExtractor("swiftf0", "cpu", withSwift)).toEqual({ extractor: "swiftf0", fellBack: false });
    });
});

/** A state with every field away from its default. */
function customState(): VoiceChangerState {
    return {
        ...defaults(),
        enabled: false,
        pitch: 4.5,
        formant: -0.25,
        vocoderWindowMs: 52,
        vocoderOversample: 16,
        chunkFrames: 1024,
        inputGain: 1.5,
        outputGain: 0.75,
        outputPrefillMs: 40,
        outputMaxQueueMs: 90,
        bufferTruncateEnabled: true,
        bufferTruncateRms: 0.001,
        bufferTruncateChunks: 12,
        inputDevice: "Microphone (USB)",
        outputDevice: "CABLE Input",
        monitorOutputDevice: "Headphones",
        monitorGain: 0.5,
        mode: "rvc",
        sidecarWsUrl: "ws://127.0.0.1:8765",
        onnxModelPath: "C:\\models\\a.onnx",
        onnxExecutionProvider: "dml",
        latencyPreset: "custom",
        rvcSlotDir: "C:\\voices\\alto",
        rvcHubertPath: "C:\\base\\hubert.onnx",
        rvcFcpePath: "C:\\base\\fcpe.onnx",
        rvcRmvpePath: "C:\\base\\rmvpe.onnx",
        rvcProfile: "quality",
        rvcExecutionProvider: "cuda",
        pitchExtractor: "rmvpe",
        extraContextMs: 192,
        crossfadeMs: 8,
        indexRate: 0.5,
        protect: 0.2,
        speakerId: 3,
        gpuDeviceIndex: 1,
        silenceSkip: false,
        vramCapMb: 1536,
        latencyBudgetMs: 450,
        auxMicGate: "duck",
        workerPriority: "protect_voice",
        rvcConsentAcknowledged: true,
        slotId: "alto",
        blockMs: 40,
        precision: "fp16",
        f0Range: "singing",
        rmsMix: 0.25,
        formantSemitones: -3,
        noiseScale: 0.5,
        silenceDb: -50,
        partialDecode: "on",
        anchoredNoise: "off",
        nsfPhase0: "on",
        f0Cache: "off",
        gpuKnn: "on",
        ioBinding: "on",
        cudaGraphs: "off",
        gpuPriority: "on",
        governor: "off",
        denoiser: "gtcrn",
        hotSlots: 3,
        vramBudgetMb: 2048,
    };
}

describe("params mapping", () => {
    it("hydrating what TS sends gives back the same state", () => {
        const s = customState();
        expect(applyRustParams(defaults(), toRustParams(s))).toEqual(s);
    });

    it("hydrates every field, including the input device and the P0 settings", () => {
        const p = toRustParams(customState());
        const got = applyRustParams(defaults(), p);
        expect(got.inputDevice).toBe("Microphone (USB)");
        expect(got.latencyBudgetMs).toBe(450);
        expect(got.auxMicGate).toBe("duck");
        expect(got.workerPriority).toBe("protect_voice");
        expect(got.rvcConsentAcknowledged).toBe(true);
    });

    it("normalizes like Rust on hydration: crepe is its own value, an unset RVC EP is the profile default", () => {
        const base = toRustParams(defaults());
        expect(applyRustParams(defaults(), { ...base, pitch_extractor: "crepe" }).pitchExtractor).toBe("crepe");
        expect(applyRustParams(defaults(), { ...base, pitch_extractor: "harvest" }).pitchExtractor).toBe("fcpe");
        expect(applyRustParams(defaults(), { ...base, rvc_execution_provider: "TensorRT_RTX" }).rvcExecutionProvider).toBe(
            "tensorrt_rtx",
        );
        expect(applyRustParams(defaults(), { ...base, onnx_execution_provider: "dml_hp" }).onnxExecutionProvider).toBe("dml");
        expect(applyRustParams(defaults(), { ...base, cuda_graphs: "ON", denoiser: "rnnoise" })).toMatchObject({
            cudaGraphs: "on",
            denoiser: "off",
        });
        expect(
            applyRustParams(defaults(), { ...base, rvc_execution_provider: "", rvc_preset: "low_latency" }).rvcExecutionProvider,
        ).toBe("cuda");
        expect(applyRustParams(defaults(), { ...base, rvc_execution_provider: "GPU" }).rvcExecutionProvider).toBe("cuda");
        expect(applyRustParams(defaults(), { ...base, onnx_execution_provider: "" }).onnxExecutionProvider).toBe("cpu");
        expect(applyRustParams(defaults(), { ...base, worker_priority: "realtime" }).workerPriority).toBe("auto");
        expect(applyRustParams(defaults(), { ...base, vocoder_oversample: 3 }).vocoderOversample).toBe(8);
    });

    it("an older backend without the P0 fields keeps the current values", () => {
        const prev = customState();
        const old = toRustParams(defaults()) as Partial<VoiceChangerParamsRust>;
        delete old.latency_budget_ms;
        delete old.aux_mic_gate;
        delete old.worker_priority;
        delete old.input_device;
        delete old.rvc_consent_acknowledged;
        for (const key of Object.keys(contract.params_p3_new_fields)) delete old[key as keyof VoiceChangerParamsRust];
        const got = applyRustParams(prev, { ...old, worker_priority: null, governor: null });
        expect(got.latencyBudgetMs).toBe(prev.latencyBudgetMs);
        expect(got.governor).toBe(prev.governor);
        expect(got.rmsMix).toBe(prev.rmsMix);
        expect(got.slotId).toBe(prev.slotId);
        expect(got.hotSlots).toBe(prev.hotSlots);
        expect(got.auxMicGate).toBe(prev.auxMicGate);
        expect(got.workerPriority).toBe(prev.workerPriority);
        expect(got.inputDevice).toBe(prev.inputDevice);
        expect(got.rvcConsentAcknowledged).toBe(prev.rvcConsentAcknowledged);
    });

    it("a malformed reply cannot put non-finite numbers into the state", () => {
        const prev = customState();
        const bad = { ...toRustParams(prev), pitch_semitones: Number.NaN, index_rate: null, vram_cap_mb: "x" } as unknown as VoiceChangerParamsRust;
        const got = applyRustParams(prev, bad);
        expect(got.pitch).toBe(prev.pitch);
        expect(got.indexRate).toBe(prev.indexRate);
        expect(got.vramCapMb).toBe(prev.vramCapMb);
    });

    it("sends only what Rust can deserialize, without clamping values Rust would report back", () => {
        const s = {
            ...defaults(),
            pitch: Number.NaN,
            speakerId: -2,
            gpuDeviceIndex: 1.6,
            vramCapMb: 1000.4,
            bufferTruncateChunks: Number.POSITIVE_INFINITY,
            vocoderWindowMs: 44.6,
            outputPrefillMs: 2,
            latencyBudgetMs: 50,
        } as VoiceChangerState;
        const sent = toRustParams(s);
        expect(sent.pitch_semitones).toBe(0);
        expect(sent.speaker_id).toBe(0);
        expect(sent.gpu_device_index).toBe(2);
        expect(sent.vram_cap_mb).toBe(1000);
        expect(sent.buffer_truncate_chunks).toBe(32);
        expect(sent.vocoder_window_ms).toBe(45);
        // Out-of-range but finite: Rust clamps and its reply brings the value back into the UI.
        expect(sent.output_prefill_ms).toBe(2);
        expect(sent.latency_budget_ms).toBe(50);
        expect(JSON.parse(JSON.stringify(sent))).toEqual(sent);
    });
});

describe("adopting Rust's effective params", () => {
    it("takes over values Rust clamped", () => {
        const cur = { ...defaults(), outputPrefillMs: 2, latencyBudgetMs: 50 };
        const sent = toRustParams(cur);
        const effective = { ...sent, output_prefill_ms: 5, output_max_queue_ms: 70, latency_budget_ms: 100 };
        const { patch, adjusted } = adoptAdjustedParams(cur, sent, effective);
        expect(adjusted.sort()).toEqual(["latency_budget_ms", "output_prefill_ms"]);
        expect(patch).toEqual({ outputPrefillMs: 5, latencyBudgetMs: 100 });
    });

    it("takes over a voice's own settings applied on entering it", () => {
        const cur = { ...customState(), rvcSlotDir: "C:\\voices\\bass" };
        const sent = toRustParams(cur);
        const effective = { ...sent, pitch_semitones: -5, protect: 0.4, pitch_extractor: "dio", index_rate: 0.75 };
        const { patch } = adoptAdjustedParams(cur, sent, effective);
        expect(patch).toEqual({ pitch: -5, protect: 0.4, pitchExtractor: "dio", indexRate: 0.75 });
    });

    it("never overwrites a field the user changed after the send", () => {
        const sentState = customState();
        const sent = toRustParams(sentState);
        // The user keeps dragging pitch and typing a budget while the reply is on its way.
        const cur = { ...sentState, pitch: 7, latencyBudgetMs: 900 };
        const effective = { ...sent, pitch_semitones: -5, latency_budget_ms: 600, protect: 0.1 };
        const { patch, adjusted } = adoptAdjustedParams(cur, sent, effective);
        expect(adjusted).toEqual(["protect"]);
        expect(patch).toEqual({ protect: 0.1 });
    });

    it("ignores f32 round-trip noise, `enabled` and fields an older backend leaves out", () => {
        const cur = { ...defaults(), indexRate: 1 / 3, enabled: true };
        const sent = toRustParams(cur);
        const effective: Partial<VoiceChangerParamsRust> = {
            ...sent,
            index_rate: Math.fround(1 / 3),
            enabled: false,
            latency_budget_ms: null,
        };
        delete effective.worker_priority;
        expect(adoptAdjustedParams(cur, sent, effective)).toEqual({ patch: {}, adjusted: [] });
    });

    it("an unset RVC EP in the reply shows the profile default instead of silently becoming CPU", () => {
        const cur = { ...defaults(), rvcProfile: "quality" as const, rvcExecutionProvider: "cuda" as const };
        const sent = toRustParams(cur);
        const { adjusted } = adoptAdjustedParams(cur, sent, { ...sent, rvc_execution_provider: "" });
        // "" is Rust for "profile default" = cuda here: same as shown, but still reported as adjusted.
        expect(adjusted).toEqual(["rvc_execution_provider"]);
        expect(applyRustParams(cur, { ...sent, rvc_execution_provider: "" }).rvcExecutionProvider).toBe("cuda");
    });

    it("compares numbers with a tolerance and everything else exactly", () => {
        expect(sameParamValue(0.33, Math.fround(0.33))).toBe(true);
        expect(sameParamValue(0.33, 0.34)).toBe(false);
        expect(sameParamValue(300, 300)).toBe(true);
        expect(sameParamValue(Number.NaN, Number.NaN)).toBe(false);
        expect(sameParamValue("cpu", "cpu")).toBe(true);
        expect(sameParamValue("cpu", "CPU")).toBe(false);
        expect(sameParamValue(null, undefined)).toBe(true);
        expect(sameParamValue(false, 0)).toBe(false);
    });
});
