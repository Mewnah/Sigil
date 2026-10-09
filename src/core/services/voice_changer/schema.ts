import { z } from "zod";

import { zSafe } from "@/utils";

import { normalizeChunkFramesValue } from "./chunkFrames";
import {
    AUTO_MS,
    AUX_MIC_GATES,
    DENOISERS,
    DEVICE_BLOCKS,
    EXECUTION_PROVIDERS,
    F0_RANGES,
    LATENCY_PRESETS,
    LEVER_MODES,
    OUTPUT_CUSHIONS,
    PITCH_EXTRACTORS,
    PRECISIONS,
    RVC_EXECUTION_PROVIDERS,
    RVC_PROFILES,
    VC_ENUM_DEFAULTS,
    VC_MODES,
    VC_NUMBER_RANGES,
    WORKER_PRIORITIES,
} from "./contract";

/**
 * In-memory shape of the voice-changer settings. Not persisted by the TS side (no `patchService`):
 * Rust owns `app_data/voice-changer/settings.json` and the service hydrates from
 * `get_voice_changer_params`. Enum lists and ranges come from `contract.ts` (checked against the
 * Rust allow-lists by `__tests__/params.test.ts`).
 */
export const VoiceChangerModeSchema = z.enum(VC_MODES);

export const LatencyPresetSchema = z.enum(LATENCY_PRESETS);

/** RVC quality/latency profile (Streamer / Low latency / Quality). */
export const RvcProfileSchema = z.enum(RVC_PROFILES);

/** F0 methods (P3 added swiftf0 / crepe / pesto / fusion; the UI offers the baseline plus the capabilities). */
export const PitchExtractorSchema = z.enum(PITCH_EXTRACTORS);

/** EP ladder rungs (ONNX chunk mode). */
export const ExecutionProviderSchema = z.enum(EXECUTION_PROVIDERS);

/** RVC EPs: the rungs plus the P3 accelerators (`tensorrt_rtx`, `dml_hp`). */
export const RvcExecutionProviderSchema = z.enum(RVC_EXECUTION_PROVIDERS);

export const PrecisionSchema = z.enum(PRECISIONS);

export const LeverModeSchema = z.enum(LEVER_MODES);

export const DenoiserSchema = z.enum(DENOISERS);

export const F0RangeSchema = z.enum(F0_RANGES);

export const WorkerPrioritySchema = z.enum(WORKER_PRIORITIES);

export const AuxMicGateSchema = z.enum(AUX_MIC_GATES);

export const OutputCushionSchema = z.enum(OUTPUT_CUSHIONS);

export const DeviceBlockSchema = z.enum(DEVICE_BLOCKS);

const R = VC_NUMBER_RANGES;

export const VoiceChangerStateSchema = z.object({
    enabled: zSafe(z.boolean(), false),
    pitch: zSafe(z.number(), R.pitch_semitones.default), // semitones, ±24
    formant: zSafe(z.number(), R.formant_shift.default), // -1.0 to +1.0
    /** FFT window length (ms). Higher = usually better quality, more latency and CPU. */
    vocoderWindowMs: zSafe(z.number().min(R.vocoder_window_ms.min).max(R.vocoder_window_ms.max), R.vocoder_window_ms.default),
    /** Phase vocoder oversampling: 1 = lowest CPU (more artifacts), 32 = heaviest, often cleaner. */
    vocoderOversample: zSafe(
        z.union([z.literal(1), z.literal(2), z.literal(4), z.literal(8), z.literal(16), z.literal(32)]),
        VC_ENUM_DEFAULTS.vocoderOversample,
    ),
    /** Frames per processing chunk (per channel). Changing while running restarts the voice changer. */
    chunkFrames: zSafe(
        z.coerce.number().transform((n) => normalizeChunkFramesValue(n)),
        512,
    ),
    inputGain: zSafe(z.number().min(R.input_gain.min).max(R.input_gain.max), R.input_gain.default),
    outputGain: zSafe(z.number().min(R.output_gain.min).max(R.output_gain.max), R.output_gain.default),
    outputPrefillMs: zSafe(z.number(), R.output_prefill_ms.default),
    outputMaxQueueMs: zSafe(z.number(), 70),
    bufferTruncateEnabled: zSafe(z.boolean(), false),
    bufferTruncateRms: zSafe(z.number(), R.buffer_truncate_rms.default),
    bufferTruncateChunks: zSafe(
        z.number().min(R.buffer_truncate_chunks.min).max(R.buffer_truncate_chunks.max),
        R.buffer_truncate_chunks.default,
    ),
    /** cpal input device name; `""` = system default. Persisted by Rust (`input_device`). */
    inputDevice: zSafe(z.string(), ""),
    outputDevice: zSafe(z.string(), "default"),
    /** Second output for headphones (streamers). Empty = off; `"default"` = system default output device. */
    monitorOutputDevice: zSafe(z.string(), ""),
    monitorGain: zSafe(z.number().min(R.monitor_gain.min).max(R.monitor_gain.max), R.monitor_gain.default),
    /** `classic` = built-in phase vocoder; `sidecar` / `onnx` / `rvc` = converter/model first, then shared post-DSP. */
    mode: zSafe(VoiceChangerModeSchema, VC_ENUM_DEFAULTS.mode),
    /** WebSocket URL for sidecar mode, e.g. ws://127.0.0.1:8765 */
    sidecarWsUrl: zSafe(z.string(), ""),
    /** Path to ONNX model: float32 `[1, chunk_samples]` in/out (see scripts/neural-sidecar). */
    onnxModelPath: zSafe(z.string(), ""),
    /** Execution provider for ONNX chunk mode only (RVC has its own `rvcExecutionProvider`). */
    onnxExecutionProvider: zSafe(ExecutionProviderSchema, VC_ENUM_DEFAULTS.onnxExecutionProvider),
    /** Buffer templates; `custom` when prefill/max ms edited manually. */
    latencyPreset: zSafe(LatencyPresetSchema, VC_ENUM_DEFAULTS.latencyPreset),
    /** Directory containing BYO RVC slot files (`gen.onnx`, optional `added.index`). */
    rvcSlotDir: zSafe(z.string(), ""),
    rvcHubertPath: zSafe(z.string(), ""),
    rvcFcpePath: zSafe(z.string(), ""),
    rvcRmvpePath: zSafe(z.string(), ""),
    /** Streamer / Low latency / Quality — maps to Rust `rvc_preset`. */
    rvcProfile: zSafe(RvcProfileSchema, VC_ENUM_DEFAULTS.rvcProfile),
    /** Execution provider for RVC (HuBERT, pitch extractor, generator) — maps to Rust `rvc_execution_provider`. */
    rvcExecutionProvider: zSafe(RvcExecutionProviderSchema, VC_ENUM_DEFAULTS.rvcExecutionProvider),
    pitchExtractor: zSafe(PitchExtractorSchema, VC_ENUM_DEFAULTS.pitchExtractor),
    /** Read when the RVC harness is built (restart to apply). Streamer 64 / Low latency 96 / Quality 192. */
    extraContextMs: zSafe(z.number().min(R.extra_context_ms.min).max(R.extra_context_ms.max), R.extra_context_ms.default),
    /** Read when the RVC harness is built (restart to apply). */
    crossfadeMs: zSafe(z.number().min(R.crossfade_ms.min).max(R.crossfade_ms.max), R.crossfade_ms.default),
    indexRate: zSafe(z.number().min(R.index_rate.min).max(R.index_rate.max), R.index_rate.default),
    protect: zSafe(z.number().min(R.protect.min).max(R.protect.max), R.protect.default),
    speakerId: zSafe(z.number().int().min(R.speaker_id.min), R.speaker_id.default),
    gpuDeviceIndex: zSafe(z.number().int().min(R.gpu_device_index.min), R.gpu_device_index.default),
    silenceSkip: zSafe(z.boolean(), true),
    vramCapMb: zSafe(z.number().int().min(R.vram_cap_mb.min).max(R.vram_cap_mb.max), R.vram_cap_mb.default),
    /**
     * Most captured audio (ms) that may wait for conversion; older audio is dropped instead of queued.
     * Hot: applies while running, in every mode (also the WebSocket read timeout).
     */
    latencyBudgetMs: zSafe(
        z.number().min(R.latency_budget_ms.min).max(R.latency_budget_ms.max),
        R.latency_budget_ms.default,
    ),
    /** The microphone while aux (TTS) audio plays through the voice changer. Hot. */
    auxMicGate: zSafe(AuxMicGateSchema, VC_ENUM_DEFAULTS.auxMicGate),
    /** Worker thread priority (auto / protect game FPS / protect voice). Hot. */
    workerPriority: zSafe(WorkerPrioritySchema, VC_ENUM_DEFAULTS.workerPriority),
    /** The RVC voice-permission notice was acknowledged (stored by Rust, not in the webview). */
    rvcConsentAcknowledged: zSafe(z.boolean(), false),

    // ---- P3: Sigil Voice Engine (P3-S0; Rust names in `params.ts`, ranges in `contract.ts`) ----
    /** Slot-store id of the selected voice (`""` = a folder chosen directly, `rvcSlotDir`). */
    slotId: zSafe(z.string(), ""),
    /** SVE block (ms); 0 = derived from `chunkFrames`. Restart. */
    blockMs: zSafe(z.number().min(R.block_ms.min).max(R.block_ms.max), R.block_ms.default),
    /** SVE lookahead (ms; 0 = off): template voices decode this much future context into every
     *  returned sample (cleaner joins and voicing) at exactly that much latency. Restart. */
    lookaheadMs: zSafe(z.number().min(R.lookahead_ms.min).max(R.lookahead_ms.max), R.lookahead_ms.default),
    precision: zSafe(PrecisionSchema, VC_ENUM_DEFAULTS.precision),
    /** Per voice. Restart. */
    f0Range: zSafe(F0RangeSchema, VC_ENUM_DEFAULTS.f0Range),
    /** Loudness envelope mix (1 = off). Hot, per voice. */
    rmsMix: zSafe(z.number().min(R.rms_mix.min).max(R.rms_mix.max), R.rms_mix.default),
    /** True formant shift of the RVC path (semitones). Hot, per voice. */
    formantSemitones: zSafe(
        z.number().min(R.formant_semitones.min).max(R.formant_semitones.max),
        R.formant_semitones.default,
    ),
    /** Flow prior noise scale of template voices. Hot. */
    noiseScale: zSafe(z.number().min(R.noise_scale.min).max(R.noise_scale.max), R.noise_scale.default),
    /** Silence-gate threshold (RMS dBFS). Hot. */
    silenceDb: zSafe(z.number().min(R.silence_db.min).max(R.silence_db.max), R.silence_db.default),
    /** Engine levers (`auto` / `on` / `off` each; Rust names in `VC_LEVERS`). Restart. */
    partialDecode: zSafe(LeverModeSchema, VC_ENUM_DEFAULTS.leverMode),
    anchoredNoise: zSafe(LeverModeSchema, VC_ENUM_DEFAULTS.leverMode),
    nsfPhase0: zSafe(LeverModeSchema, VC_ENUM_DEFAULTS.leverMode),
    f0Cache: zSafe(LeverModeSchema, VC_ENUM_DEFAULTS.leverMode),
    gpuKnn: zSafe(LeverModeSchema, VC_ENUM_DEFAULTS.leverMode),
    ioBinding: zSafe(LeverModeSchema, VC_ENUM_DEFAULTS.leverMode),
    cudaGraphs: zSafe(LeverModeSchema, VC_ENUM_DEFAULTS.leverMode),
    gpuPriority: zSafe(LeverModeSchema, VC_ENUM_DEFAULTS.leverMode),
    governor: zSafe(LeverModeSchema, VC_ENUM_DEFAULTS.leverMode),
    denoiser: zSafe(DenoiserSchema, VC_ENUM_DEFAULTS.denoiser),
    hotSlots: zSafe(z.number().int().min(R.hot_slots.min).max(R.hot_slots.max), R.hot_slots.default),
    vramBudgetMb: zSafe(
        z.number().int().min(R.vram_budget_mb.min).max(R.vram_budget_mb.max),
        R.vram_budget_mb.default,
    ),

    // ---- P5: voice families (P5-S0; Rust name `reference_voice_id`) ----
    /** Voice-library id of a zero-shot voice's reference (`""` = none). Per voice; switch. */
    referenceVoiceId: zSafe(z.string(), ""),

    // ---- P3B: voice-engine optimization (P3B-S0; Rust names in `params.ts`; no UI until P3B-H) ----
    /** Window-independent content normalisation lever (P3B-D). Restart. */
    contentNorm: zSafe(LeverModeSchema, VC_ENUM_DEFAULTS.leverMode),
    /** Cross-tick F0 decoding lever (P3B-F). Restart. */
    f0Decode: zSafe(LeverModeSchema, VC_ENUM_DEFAULTS.leverMode),
    /** `fixed` prefill or `adaptive` cushion (P3B-H). Restart. */
    outputCushion: zSafe(OutputCushionSchema, VC_ENUM_DEFAULTS.outputCushion),
    /** SOLA search window (ms); `AUTO_MS` (-1) = automatic (P3B-H). Restart. */
    solaSearchMs: zSafe(
        z.number().refine((n) => n === AUTO_MS || (n >= R.sola_search_ms.min && n <= R.sola_search_ms.max)),
        R.sola_search_ms.default,
    ),
    /** `fixed` device block or `model_divisor` (template voices; P3B-H). Restart. */
    deviceBlock: zSafe(DeviceBlockSchema, VC_ENUM_DEFAULTS.deviceBlock),
    /** Content-encoder left context (ms); `AUTO_MS` (-1) = the preset's (P3B-D). Restart. */
    contentContextMs: zSafe(
        z.number().refine((n) => n === AUTO_MS || (n >= R.content_context_ms.min && n <= R.content_context_ms.max)),
        R.content_context_ms.default,
    ),
});

export type VoiceChangerState = z.infer<typeof VoiceChangerStateSchema>;
export type RvcProfile = z.infer<typeof RvcProfileSchema>;
export type PitchExtractor = z.infer<typeof PitchExtractorSchema>;
export type ExecutionProvider = z.infer<typeof ExecutionProviderSchema>;
export type RvcExecutionProvider = z.infer<typeof RvcExecutionProviderSchema>;
