/**
 * TS mirror of the frozen P0 Rust <-> TS voice-changer contract
 * (`src-tauri/src/services/voice_changer/params_contract.json`).
 *
 * Every list, default and range here must equal the Rust allow-lists: `__tests__/params.test.ts`
 * checks this file against the same JSON the Rust `contract_tests.rs` reads, so an option added on
 * one side only fails CI instead of being remapped silently at runtime.
 *
 * The normalizers mirror Rust's exactly (trimmed, case-insensitive; unknown values map to the
 * contract fallback). Pure module: no Tauri, DOM or `window`.
 *
 * P3 (P3-S0): the Sigil Voice Engine values. An enum's `BASELINE_*` list is what every build runs;
 * the rest run once their workstream lands, and `voice_changer_get_capabilities` (`ipc.ts`) says
 * which. The UI offers the baseline plus the capabilities; until then a value runs its stand-in
 * (`PITCH_EXTRACTOR_STAND_IN`) or base rung (`EXECUTION_PROVIDER_BASE`), exactly as in Rust.
 */

export const VC_MODES = ["classic", "onnx", "sidecar", "rvc"] as const;
export type VcMode = (typeof VC_MODES)[number];

export const LATENCY_PRESETS = ["low_latency", "stable", "custom"] as const;
export type LatencyPreset = (typeof LATENCY_PRESETS)[number];

/** RVC profiles. `studio_trt` (P3): Quality shapes on TensorRT-RTX, offered once the capabilities list it. */
export const RVC_PROFILES = ["streamer", "low_latency", "quality", "studio_trt"] as const;
export type RvcProfileId = (typeof RVC_PROFILES)[number];
export const BASELINE_RVC_PROFILES: readonly RvcProfileId[] = ["streamer", "low_latency", "quality"];

/**
 * F0 methods. P3 added `swiftf0`, a real `crepe` (CREPE-tiny; it was an alias of rmvpe until P3),
 * `pesto` and `fusion`; while a build cannot run one it runs its stand-in.
 */
export const PITCH_EXTRACTORS = ["fcpe", "rmvpe", "dio", "swiftf0", "crepe", "pesto", "fusion"] as const;
export type PitchExtractorId = (typeof PITCH_EXTRACTORS)[number];
/** F0 methods every build runs. */
export const BASELINE_PITCH_EXTRACTORS: readonly PitchExtractorId[] = ["fcpe", "rmvpe", "dio"];
/** No legacy aliases since P3 (`crepe` is a value of its own). */
export const PITCH_EXTRACTOR_LEGACY_ALIASES: Readonly<Record<string, PitchExtractorId>> = {};
/** What an F0 method runs as while this build cannot run it (Rust `pitch_stand_in`). */
export const PITCH_EXTRACTOR_STAND_IN: Readonly<Partial<Record<PitchExtractorId, PitchExtractorId>>> = {
    swiftf0: "fcpe",
    crepe: "rmvpe",
    pesto: "fcpe",
    fusion: "rmvpe",
};
/** Extractors that need a GPU execution provider; on the CPU EP they run as `PITCH_EXTRACTOR_CPU_FALLBACK`. */
export const GPU_ONLY_PITCH_EXTRACTORS: readonly PitchExtractorId[] = ["rmvpe"];
export const PITCH_EXTRACTOR_CPU_FALLBACK: PitchExtractorId = "fcpe";

/** EP ladder rungs (ONNX chunk mode offers only these). */
export const EXECUTION_PROVIDERS = ["cpu", "cuda", "dml"] as const;
export type ExecutionProviderId = (typeof EXECUTION_PROVIDERS)[number];
/** RVC EPs: the rungs plus the P3 accelerators (`tensorrt_rtx` on CUDA, `dml_hp` on DirectML). */
export const RVC_EXECUTION_PROVIDERS = ["cpu", "cuda", "dml", "tensorrt_rtx", "dml_hp"] as const;
export type RvcExecutionProviderId = (typeof RVC_EXECUTION_PROVIDERS)[number];
export const EXECUTION_PROVIDER_ALIASES: Readonly<Record<string, ExecutionProviderId>> = { gpu: "cuda" };
/** The EP ladder rung each RVC EP builds its sessions on (Rust `base_ep`). */
export const EXECUTION_PROVIDER_BASE: Readonly<Record<RvcExecutionProviderId, ExecutionProviderId>> = {
    cpu: "cpu",
    cuda: "cuda",
    dml: "dml",
    tensorrt_rtx: "cuda",
    dml_hp: "dml",
};

export const VOCODER_OVERSAMPLES = [1, 2, 4, 8, 16, 32] as const;
export type VocoderOversampleValue = (typeof VOCODER_OVERSAMPLES)[number];

/** Worker thread priority (Rust `rt_thread::WorkerPriority`). */
export const WORKER_PRIORITIES = ["auto", "protect_game", "protect_voice"] as const;
export type WorkerPriority = (typeof WORKER_PRIORITIES)[number];

/** What happens to the microphone while aux (TTS) audio plays through the voice changer. */
export const AUX_MIC_GATES = ["none", "duck", "mute"] as const;
export type AuxMicGate = (typeof AUX_MIC_GATES)[number];

/** P3: graph precision of template voices (`auto` = the template's recommendation per graph and device). */
export const PRECISIONS = ["auto", "fp32", "fp16"] as const;
export type Precision = (typeof PRECISIONS)[number];
export const BASELINE_PRECISIONS: readonly Precision[] = ["auto", "fp32"];

/** P3: mode of every engine lever (`auto` = the preset's benchmark-decided default). */
export const LEVER_MODES = ["auto", "on", "off"] as const;
export type LeverMode = (typeof LEVER_MODES)[number];

/** P3: the switchable engine levers (Rust `engine::options::Lever`), in contract order. Each is a param of its own. */
export const VC_LEVERS = [
    "partial_decode",
    "anchored_noise",
    "nsf_phase0",
    "f0_cache",
    "gpu_knn",
    "io_binding",
    "cuda_graphs",
    "gpu_priority",
    "governor",
] as const;
export type VcLever = (typeof VC_LEVERS)[number];

/** P3: input denoiser on the voice-changer path only. */
export const DENOISERS = ["off", "gtcrn", "dfn3", "dpdfnet"] as const;
export type Denoiser = (typeof DENOISERS)[number];
export const BASELINE_DENOISERS: readonly Denoiser[] = ["off"];

/** P3: F0 search range (per voice). */
export const F0_RANGES = ["speech", "singing"] as const;
export type F0Range = (typeof F0_RANGES)[number];

export const VC_ENUM_DEFAULTS = {
    mode: "classic",
    latencyPreset: "low_latency",
    rvcProfile: "streamer",
    pitchExtractor: "fcpe",
    rvcExecutionProvider: "cpu",
    onnxExecutionProvider: "cpu",
    vocoderOversample: 8,
    workerPriority: "auto",
    auxMicGate: "none",
    precision: "auto",
    leverMode: "auto",
    denoiser: "off",
    f0Range: "speech",
} as const satisfies {
    mode: VcMode;
    latencyPreset: LatencyPreset;
    rvcProfile: RvcProfileId;
    pitchExtractor: PitchExtractorId;
    rvcExecutionProvider: ExecutionProviderId;
    onnxExecutionProvider: ExecutionProviderId;
    vocoderOversample: VocoderOversampleValue;
    workerPriority: WorkerPriority;
    auxMicGate: AuxMicGate;
    precision: Precision;
    leverMode: LeverMode;
    denoiser: Denoiser;
    f0Range: F0Range;
};

export interface NumberRange {
    min: number;
    /** Absent = unbounded above. */
    max?: number;
    default: number;
}

/** Numeric params, keyed by their Rust (snake_case) name. Rust clamps to these on `set`. */
export const VC_NUMBER_RANGES = {
    pitch_semitones: { min: -24, max: 24, default: 0 },
    formant_shift: { min: -1, max: 1, default: 0 },
    vocoder_window_ms: { min: 30, max: 60, default: 45 },
    input_gain: { min: 0, max: 4, default: 1 },
    output_gain: { min: 0, max: 4, default: 1 },
    monitor_gain: { min: 0, max: 4, default: 1 },
    output_prefill_ms: { min: 5, max: 300, default: 35 },
    buffer_truncate_rms: { min: 1e-8, max: 1, default: 0.0005 },
    buffer_truncate_chunks: { min: 1, max: 500, default: 32 },
    extra_context_ms: { min: 0, max: 2000, default: 64 },
    crossfade_ms: { min: 0, max: 200, default: 20 },
    index_rate: { min: 0, max: 1, default: 0 },
    protect: { min: 0, max: 0.5, default: 0.33 },
    speaker_id: { min: 0, default: 0 },
    gpu_device_index: { min: 0, default: 0 },
    vram_cap_mb: { min: 256, max: 8192, default: 768 },
    latency_budget_ms: { min: 100, max: 2000, default: 300 },
    // P3 (P3-S0)
    block_ms: { min: 0, max: 200, default: 0 },
    // P3-I0
    lookahead_ms: { min: 0, max: 80, default: 0 },
    rms_mix: { min: 0, max: 1, default: 1 },
    formant_semitones: { min: -12, max: 12, default: 0 },
    noise_scale: { min: 0, max: 1, default: 0.66666 },
    silence_db: { min: -90, max: -10, default: -42 },
    hot_slots: { min: 1, max: 8, default: 1 },
    vram_budget_mb: { min: 0, max: 65536, default: 0 },
} as const satisfies Record<string, NumberRange>;

/** `output_max_queue_ms` is clamped to `[prefill + MIN_ABOVE_PREFILL, MAX]` (Rust `clamp_output_buffer_ms`). */
export const OUTPUT_MAX_QUEUE_MS = { minAbovePrefill: 5, max: 600, default: 70 } as const;

/** Output buffer templates (Rust `PREFILL_MS_*` / `MAX_QUEUE_MS_*`). */
export const OUTPUT_BUFFER_LOW_LATENCY = { prefillMs: 35, maxQueueMs: 70 } as const;
export const OUTPUT_BUFFER_STABLE = { prefillMs: 80, maxQueueMs: 150 } as const;

function norm(s: unknown): string {
    return typeof s === "string" ? s.trim().toLowerCase() : "";
}

function pick<T extends string>(values: readonly T[], s: unknown, fallback: T): T {
    const v = norm(s);
    return (values as readonly string[]).includes(v) ? (v as T) : fallback;
}

/** Voice-changer mode; unknown values are `classic`. */
export function normalizeMode(s: unknown): VcMode {
    return pick(VC_MODES, s, VC_ENUM_DEFAULTS.mode);
}

/** Rust `normalize_latency_preset`: anything unknown is `low_latency`. */
export function normalizeLatencyPreset(s: unknown): LatencyPreset {
    return pick(LATENCY_PRESETS, s, VC_ENUM_DEFAULTS.latencyPreset);
}

/** Rust `normalize_rvc_preset`: anything unknown is `streamer`. */
export function normalizeRvcProfile(s: unknown): RvcProfileId {
    return pick(RVC_PROFILES, s, VC_ENUM_DEFAULTS.rvcProfile);
}

/** Rust `normalize_pitch_extractor`: `crepe` is stored as `rmvpe`; anything unknown is `fcpe`. */
export function normalizePitchExtractor(s: unknown): PitchExtractorId {
    const v = norm(s);
    const alias = PITCH_EXTRACTOR_LEGACY_ALIASES[v];
    if (alias) return alias;
    return pick(PITCH_EXTRACTORS, v, VC_ENUM_DEFAULTS.pitchExtractor);
}

/**
 * Rust `normalize_execution_provider`: `gpu` is `cuda`, `""` stays `""` (= the RVC profile's default
 * EP), the P3 accelerators are kept, anything else unknown is `cpu`.
 */
export function normalizeExecutionProvider(s: unknown): RvcExecutionProviderId | "" {
    const v = norm(s);
    if (v === "") return "";
    const alias = EXECUTION_PROVIDER_ALIASES[v];
    if (alias) return alias;
    return pick(RVC_EXECUTION_PROVIDERS, v, "cpu");
}

/** ONNX chunk mode EP: a ladder rung only (an accelerator runs as its base rung, `""` as the CPU). */
export function normalizeOnnxExecutionProvider(s: unknown): ExecutionProviderId {
    const v = normalizeExecutionProvider(s);
    return v === "" ? "cpu" : EXECUTION_PROVIDER_BASE[v];
}

/** Rust `base_ep`: the EP ladder rung an RVC EP builds its sessions on. */
export function rvcBaseEp(ep: unknown): ExecutionProviderId {
    const v = normalizeExecutionProvider(ep);
    return v === "" ? "cpu" : EXECUTION_PROVIDER_BASE[v];
}

/** Rust `normalize_precision`: anything unknown is `auto`. */
export function normalizePrecision(s: unknown): Precision {
    return pick(PRECISIONS, s, VC_ENUM_DEFAULTS.precision);
}

/** Rust `normalize_lever_mode`: anything unknown is `auto`. */
export function normalizeLeverMode(s: unknown): LeverMode {
    return pick(LEVER_MODES, s, VC_ENUM_DEFAULTS.leverMode);
}

/** Rust `normalize_denoiser`: anything unknown is `off`. */
export function normalizeDenoiser(s: unknown): Denoiser {
    return pick(DENOISERS, s, VC_ENUM_DEFAULTS.denoiser);
}

/** Rust `normalize_f0_range`: anything unknown is `speech`. */
export function normalizeF0Range(s: unknown): F0Range {
    return pick(F0_RANGES, s, VC_ENUM_DEFAULTS.f0Range);
}

/** Rust `WorkerPriority::from_str_lossy`: anything unknown is `auto`. */
export function normalizeWorkerPriority(s: unknown): WorkerPriority {
    return pick(WORKER_PRIORITIES, s, VC_ENUM_DEFAULTS.workerPriority);
}

/** Rust `AuxMicGate::from_str_lossy`: anything unknown is `none`. */
export function normalizeAuxMicGate(s: unknown): AuxMicGate {
    return pick(AUX_MIC_GATES, s, VC_ENUM_DEFAULTS.auxMicGate);
}

/** Rust `set_voice_changer_params`: an unsupported oversample factor is 8. */
export function normalizeVocoderOversample(n: unknown): VocoderOversampleValue {
    return (VOCODER_OVERSAMPLES as readonly unknown[]).includes(n)
        ? (n as VocoderOversampleValue)
        : VC_ENUM_DEFAULTS.vocoderOversample;
}

/**
 * RVC profile default EP (Rust `params::preset_default_ep`): Streamer runs on the CPU, the GPU profiles on
 * CUDA, Studio-TRT asks for TensorRT-RTX.
 */
export function profileDefaultEp(profile: RvcProfileId): RvcExecutionProviderId {
    if (profile === "streamer") return "cpu";
    return profile === "studio_trt" ? "tensorrt_rtx" : "cuda";
}

/**
 * Rust `requested_rvc_ep`: the stored EP, or the profile default only when it is unset (`""`). May be an
 * accelerator; `rvcBaseEp` gives the rung its sessions build on.
 */
export function effectiveRvcEp(ep: RvcExecutionProviderId | "", profile: RvcProfileId): RvcExecutionProviderId {
    return ep === "" ? profileDefaultEp(profile) : ep;
}

/**
 * Rust `effective_pitch_extractor`: the F0 extractor a run actually uses on `ep`, and whether it fell back.
 * A method not in `available` (default: the baseline; pass the capabilities' list) runs its stand-in, and a
 * GPU-only extractor on the CPU rung runs as fcpe. The stored choice is never rewritten.
 */
export function effectivePitchExtractor(
    pitchExtractor: unknown,
    ep: RvcExecutionProviderId,
    available: readonly string[] = BASELINE_PITCH_EXTRACTORS,
): { extractor: PitchExtractorId; fellBack: boolean } {
    const chosen = normalizePitchExtractor(pitchExtractor);
    const runnable = available.includes(chosen) ? chosen : (PITCH_EXTRACTOR_STAND_IN[chosen] ?? chosen);
    const run =
        rvcBaseEp(ep) === "cpu" && GPU_ONLY_PITCH_EXTRACTORS.includes(runnable) ? PITCH_EXTRACTOR_CPU_FALLBACK : runnable;
    return { extractor: run, fellBack: run !== chosen };
}

/** Clamp to `[min, max]`; a non-finite value is `fallback`. */
export function clampFinite(n: unknown, min: number, max: number, fallback: number): number {
    if (typeof n !== "number" || !Number.isFinite(n)) return fallback;
    return Math.min(max, Math.max(min, n));
}

/** Rust `clamp_latency_budget_ms`: clamp to 100-2000 ms; a non-finite value is the 300 ms default. */
export function clampLatencyBudgetMs(ms: unknown): number {
    const r = VC_NUMBER_RANGES.latency_budget_ms;
    return clampFinite(ms, r.min, r.max, r.default);
}

/** Rust `clamp_output_buffer_ms`: prefill 5-300 ms, max queue at least prefill + 5 and at most 600 ms. */
export function clampOutputBufferMs(prefillMs: number, maxQueueMs: number): { prefillMs: number; maxQueueMs: number } {
    const r = VC_NUMBER_RANGES.output_prefill_ms;
    const prefill = clampFinite(prefillMs, r.min, r.max, r.default);
    const maxQueue = clampFinite(
        maxQueueMs,
        prefill + OUTPUT_MAX_QUEUE_MS.minAbovePrefill,
        OUTPUT_MAX_QUEUE_MS.max,
        Math.max(OUTPUT_MAX_QUEUE_MS.default, prefill + OUTPUT_MAX_QUEUE_MS.minAbovePrefill),
    );
    return { prefillMs: prefill, maxQueueMs: maxQueue };
}
