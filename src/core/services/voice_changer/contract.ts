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
 */

export const VC_MODES = ["classic", "onnx", "sidecar", "rvc"] as const;
export type VcMode = (typeof VC_MODES)[number];

export const LATENCY_PRESETS = ["low_latency", "stable", "custom"] as const;
export type LatencyPreset = (typeof LATENCY_PRESETS)[number];

export const RVC_PROFILES = ["streamer", "low_latency", "quality"] as const;
export type RvcProfileId = (typeof RVC_PROFILES)[number];

/** F0 extractors the engine runs. `crepe` is a legacy alias of `rmvpe` and is never offered. */
export const PITCH_EXTRACTORS = ["fcpe", "rmvpe", "dio"] as const;
export type PitchExtractorId = (typeof PITCH_EXTRACTORS)[number];
export const PITCH_EXTRACTOR_LEGACY_ALIASES: Readonly<Record<string, PitchExtractorId>> = { crepe: "rmvpe" };
/** Extractors that need a GPU execution provider; on the CPU EP they run as `PITCH_EXTRACTOR_CPU_FALLBACK`. */
export const GPU_ONLY_PITCH_EXTRACTORS: readonly PitchExtractorId[] = ["rmvpe"];
export const PITCH_EXTRACTOR_CPU_FALLBACK: PitchExtractorId = "fcpe";

export const EXECUTION_PROVIDERS = ["cpu", "cuda", "dml"] as const;
export type ExecutionProviderId = (typeof EXECUTION_PROVIDERS)[number];
export const EXECUTION_PROVIDER_ALIASES: Readonly<Record<string, ExecutionProviderId>> = { gpu: "cuda" };

export const VOCODER_OVERSAMPLES = [1, 2, 4, 8, 16, 32] as const;
export type VocoderOversampleValue = (typeof VOCODER_OVERSAMPLES)[number];

/** Worker thread priority (Rust `rt_thread::WorkerPriority`). */
export const WORKER_PRIORITIES = ["auto", "protect_game", "protect_voice"] as const;
export type WorkerPriority = (typeof WORKER_PRIORITIES)[number];

/** What happens to the microphone while aux (TTS) audio plays through the voice changer. */
export const AUX_MIC_GATES = ["none", "duck", "mute"] as const;
export type AuxMicGate = (typeof AUX_MIC_GATES)[number];

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
 * EP), anything else unknown is `cpu`.
 */
export function normalizeExecutionProvider(s: unknown): ExecutionProviderId | "" {
    const v = norm(s);
    if (v === "") return "";
    const alias = EXECUTION_PROVIDER_ALIASES[v];
    if (alias) return alias;
    return pick(EXECUTION_PROVIDERS, v, "cpu");
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

/** RVC profile default EP (Rust `RvcPresetProfile::default_ep`): Streamer runs on the CPU, the GPU profiles on CUDA. */
export function profileDefaultEp(profile: RvcProfileId): ExecutionProviderId {
    return profile === "streamer" ? "cpu" : "cuda";
}

/** Rust `effective_rvc_ep`: the stored EP, or the profile default only when it is unset (`""`). */
export function effectiveRvcEp(ep: ExecutionProviderId | "", profile: RvcProfileId): ExecutionProviderId {
    return ep === "" ? profileDefaultEp(profile) : ep;
}

/**
 * Rust `effective_pitch_extractor`: the F0 extractor a run actually uses on `ep`, and whether it fell
 * back (a GPU-only extractor on the CPU EP runs as fcpe). The stored choice is never rewritten.
 */
export function effectivePitchExtractor(
    pitchExtractor: unknown,
    ep: ExecutionProviderId,
): { extractor: PitchExtractorId; fellBack: boolean } {
    const p = normalizePitchExtractor(pitchExtractor);
    if (ep === "cpu" && GPU_ONLY_PITCH_EXTRACTORS.includes(p)) {
        return { extractor: PITCH_EXTRACTOR_CPU_FALLBACK, fellBack: true };
    }
    return { extractor: p, fellBack: false };
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
