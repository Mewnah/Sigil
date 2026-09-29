/**
 * Voice-changer params between the TS state (camelCase) and the Rust IPC struct
 * `VoiceChangerParams` (snake_case, `get_voice_changer_params` / `set_voice_changer_params`).
 *
 * - `toRustParams`: what `syncParams` sends. It never silently changes a value Rust would accept;
 *   it only makes the payload deserializable (non-finite numbers -> defaults, integers rounded,
 *   unsigned fields non-negative). Rust clamps and normalizes, and returns the effective params.
 * - `applyRustParams`: hydration (every field), normalized with the contract rules.
 * - `adoptAdjustedParams`: after a `set`, take over the values Rust changed (clamps, a voice's own
 *   `slot.json` applied on entering it), without clobbering fields the user edited meanwhile.
 *
 * Pure module: no Tauri, DOM or `window`.
 */
import { normalizeChunkFramesValue } from "./chunkFrames";
import {
    clampFinite,
    clampLatencyBudgetMs,
    effectiveRvcEp,
    normalizeAuxMicGate,
    normalizeExecutionProvider,
    normalizeLatencyPreset,
    normalizeMode,
    normalizePitchExtractor,
    normalizeRvcProfile,
    normalizeVocoderOversample,
    normalizeWorkerPriority,
    VC_NUMBER_RANGES as R,
    type VcMode,
} from "./contract";
import type { VoiceChangerState } from "./schema";

/** Rust `VoiceChangerParams` (serde snake_case). The P0 fields are optional on `set` (absent = keep). */
export interface VoiceChangerParamsRust {
    enabled: boolean;
    pitch_semitones: number;
    formant_shift: number;
    output_device: string;
    vocoder_window_ms: number;
    vocoder_oversample: number;
    chunk_frames: number;
    input_gain: number;
    output_gain: number;
    output_prefill_ms: number;
    output_max_queue_ms: number;
    buffer_truncate_enabled: boolean;
    buffer_truncate_rms: number;
    buffer_truncate_chunks: number;
    mode: VcMode;
    sidecar_ws_url: string;
    onnx_model_path: string;
    /** ONNX chunk mode only. */
    onnx_execution_provider: string;
    /** RVC mode only (HuBERT, pitch extractor, generator); `""` = the profile's default EP. */
    rvc_execution_provider: string;
    latency_preset: string;
    monitor_output_device: string;
    monitor_gain: number;
    rvc_slot_dir: string;
    rvc_hubert_path: string;
    rvc_fcpe_path: string;
    rvc_rmvpe_path: string;
    rvc_preset: string;
    pitch_extractor: string;
    extra_context_ms: number;
    crossfade_ms: number;
    index_rate: number;
    protect: number;
    speaker_id: number;
    gpu_device_index: number;
    silence_skip: boolean;
    vram_cap_mb: number;
    /** P0: most captured audio (ms) that may wait for conversion (100-2000). */
    latency_budget_ms?: number | null;
    /** P0: `none` / `duck` / `mute`. */
    aux_mic_gate?: string | null;
    /** P0: `auto` / `protect_game` / `protect_voice`. */
    worker_priority?: string | null;
    /** P0: cpal input device name; `""` = system default. */
    input_device?: string | null;
    /** P0: the RVC voice-permission notice was acknowledged. */
    rvc_consent_acknowledged?: boolean | null;
}

export type RustParamKey = keyof VoiceChangerParamsRust;

/** Every field `toRustParams` sends, in a stable order (also what `adoptAdjustedParams` compares). */
export const RUST_PARAM_KEYS: readonly RustParamKey[] = [
    "enabled",
    "pitch_semitones",
    "formant_shift",
    "output_device",
    "vocoder_window_ms",
    "vocoder_oversample",
    "chunk_frames",
    "input_gain",
    "output_gain",
    "output_prefill_ms",
    "output_max_queue_ms",
    "buffer_truncate_enabled",
    "buffer_truncate_rms",
    "buffer_truncate_chunks",
    "mode",
    "sidecar_ws_url",
    "onnx_model_path",
    "onnx_execution_provider",
    "rvc_execution_provider",
    "latency_preset",
    "monitor_output_device",
    "monitor_gain",
    "rvc_slot_dir",
    "rvc_hubert_path",
    "rvc_fcpe_path",
    "rvc_rmvpe_path",
    "rvc_preset",
    "pitch_extractor",
    "extra_context_ms",
    "crossfade_ms",
    "index_rate",
    "protect",
    "speaker_id",
    "gpu_device_index",
    "silence_skip",
    "vram_cap_mb",
    "latency_budget_ms",
    "aux_mic_gate",
    "worker_priority",
    "input_device",
    "rvc_consent_acknowledged",
];

/** A finite number, else `fallback` (JSON has no NaN/Infinity: they would fail Rust's deserialize). */
function finite(n: unknown, fallback: number): number {
    return typeof n === "number" && Number.isFinite(n) ? n : fallback;
}

/** A finite, rounded, non-negative integer (Rust `u32` / `i64` fields reject fractions and negatives). */
function uint(n: unknown, fallback: number): number {
    return Math.max(0, Math.round(finite(n, fallback)));
}

function str(s: unknown): string {
    return typeof s === "string" ? s : "";
}

/** The IPC params for `set_voice_changer_params` (snake_case). */
export function toRustParams(vc: VoiceChangerState): VoiceChangerParamsRust {
    return {
        enabled: !!vc.enabled,
        pitch_semitones: finite(vc.pitch, R.pitch_semitones.default),
        formant_shift: finite(vc.formant, R.formant_shift.default),
        output_device: str(vc.outputDevice),
        vocoder_window_ms: uint(vc.vocoderWindowMs, R.vocoder_window_ms.default),
        vocoder_oversample: normalizeVocoderOversample(vc.vocoderOversample),
        chunk_frames: normalizeChunkFramesValue(finite(vc.chunkFrames, 512)),
        input_gain: finite(vc.inputGain, R.input_gain.default),
        output_gain: finite(vc.outputGain, R.output_gain.default),
        output_prefill_ms: finite(vc.outputPrefillMs, R.output_prefill_ms.default),
        output_max_queue_ms: finite(vc.outputMaxQueueMs, 70),
        buffer_truncate_enabled: !!vc.bufferTruncateEnabled,
        buffer_truncate_rms: finite(vc.bufferTruncateRms, R.buffer_truncate_rms.default),
        buffer_truncate_chunks: uint(vc.bufferTruncateChunks, R.buffer_truncate_chunks.default),
        mode: normalizeMode(vc.mode),
        sidecar_ws_url: str(vc.sidecarWsUrl),
        onnx_model_path: str(vc.onnxModelPath),
        onnx_execution_provider: str(vc.onnxExecutionProvider),
        rvc_execution_provider: str(vc.rvcExecutionProvider),
        latency_preset: str(vc.latencyPreset),
        monitor_output_device: str(vc.monitorOutputDevice),
        monitor_gain: finite(vc.monitorGain, R.monitor_gain.default),
        rvc_slot_dir: str(vc.rvcSlotDir),
        rvc_hubert_path: str(vc.rvcHubertPath),
        rvc_fcpe_path: str(vc.rvcFcpePath),
        rvc_rmvpe_path: str(vc.rvcRmvpePath),
        rvc_preset: str(vc.rvcProfile),
        pitch_extractor: str(vc.pitchExtractor),
        extra_context_ms: finite(vc.extraContextMs, R.extra_context_ms.default),
        crossfade_ms: finite(vc.crossfadeMs, R.crossfade_ms.default),
        index_rate: finite(vc.indexRate, R.index_rate.default),
        protect: finite(vc.protect, R.protect.default),
        speaker_id: uint(vc.speakerId, R.speaker_id.default),
        gpu_device_index: uint(vc.gpuDeviceIndex, R.gpu_device_index.default),
        silence_skip: !!vc.silenceSkip,
        vram_cap_mb: uint(vc.vramCapMb, R.vram_cap_mb.default),
        latency_budget_ms: finite(vc.latencyBudgetMs, R.latency_budget_ms.default),
        aux_mic_gate: str(vc.auxMicGate),
        worker_priority: str(vc.workerPriority),
        input_device: str(vc.inputDevice),
        rvc_consent_acknowledged: !!vc.rvcConsentAcknowledged,
    };
}

type Patch = Partial<VoiceChangerState>;

/**
 * Per Rust field: the TS state it sets. `p` is the whole Rust object (the RVC EP needs the profile);
 * `prev` is the current state, kept when a value is absent or not a finite number.
 */
const FROM_RUST: { [K in RustParamKey]: (p: Partial<VoiceChangerParamsRust>, prev: VoiceChangerState) => Patch } = {
    enabled: (p, prev) => ({ enabled: typeof p.enabled === "boolean" ? p.enabled : prev.enabled }),
    pitch_semitones: (p, prev) => ({
        pitch: clampFinite(p.pitch_semitones, R.pitch_semitones.min, R.pitch_semitones.max, prev.pitch),
    }),
    formant_shift: (p, prev) => ({
        formant: clampFinite(p.formant_shift, R.formant_shift.min, R.formant_shift.max, prev.formant),
    }),
    output_device: (p, prev) => ({ outputDevice: typeof p.output_device === "string" ? p.output_device : prev.outputDevice }),
    vocoder_window_ms: (p, prev) => ({
        vocoderWindowMs: Math.round(
            clampFinite(p.vocoder_window_ms, R.vocoder_window_ms.min, R.vocoder_window_ms.max, prev.vocoderWindowMs),
        ),
    }),
    vocoder_oversample: (p) => ({ vocoderOversample: normalizeVocoderOversample(p.vocoder_oversample) }),
    chunk_frames: (p, prev) => ({ chunkFrames: normalizeChunkFramesValue(finite(p.chunk_frames, prev.chunkFrames)) }),
    input_gain: (p, prev) => ({ inputGain: clampFinite(p.input_gain, R.input_gain.min, R.input_gain.max, prev.inputGain) }),
    output_gain: (p, prev) => ({
        outputGain: clampFinite(p.output_gain, R.output_gain.min, R.output_gain.max, prev.outputGain),
    }),
    output_prefill_ms: (p, prev) => ({ outputPrefillMs: finite(p.output_prefill_ms, prev.outputPrefillMs) }),
    output_max_queue_ms: (p, prev) => ({ outputMaxQueueMs: finite(p.output_max_queue_ms, prev.outputMaxQueueMs) }),
    buffer_truncate_enabled: (p, prev) => ({
        bufferTruncateEnabled:
            typeof p.buffer_truncate_enabled === "boolean" ? p.buffer_truncate_enabled : prev.bufferTruncateEnabled,
    }),
    buffer_truncate_rms: (p, prev) => ({ bufferTruncateRms: finite(p.buffer_truncate_rms, prev.bufferTruncateRms) }),
    buffer_truncate_chunks: (p, prev) => ({
        bufferTruncateChunks: Math.round(
            clampFinite(
                p.buffer_truncate_chunks,
                R.buffer_truncate_chunks.min,
                R.buffer_truncate_chunks.max,
                prev.bufferTruncateChunks,
            ),
        ),
    }),
    mode: (p) => ({ mode: normalizeMode(p.mode) }),
    sidecar_ws_url: (p, prev) => ({ sidecarWsUrl: typeof p.sidecar_ws_url === "string" ? p.sidecar_ws_url : prev.sidecarWsUrl }),
    onnx_model_path: (p, prev) => ({
        onnxModelPath: typeof p.onnx_model_path === "string" ? p.onnx_model_path : prev.onnxModelPath,
    }),
    // ONNX chunk mode has no profile: an unset EP runs on the CPU.
    onnx_execution_provider: (p) => ({ onnxExecutionProvider: normalizeExecutionProvider(p.onnx_execution_provider) || "cpu" }),
    // Unset (`""`) runs the profile default; show that instead of pretending it is the CPU.
    rvc_execution_provider: (p) => ({
        rvcExecutionProvider: effectiveRvcEp(
            normalizeExecutionProvider(p.rvc_execution_provider),
            normalizeRvcProfile(p.rvc_preset),
        ),
    }),
    latency_preset: (p) => ({ latencyPreset: normalizeLatencyPreset(p.latency_preset) }),
    monitor_output_device: (p) => ({
        monitorOutputDevice: typeof p.monitor_output_device === "string" ? p.monitor_output_device : "",
    }),
    monitor_gain: (p, prev) => ({
        monitorGain: clampFinite(p.monitor_gain, R.monitor_gain.min, R.monitor_gain.max, prev.monitorGain),
    }),
    rvc_slot_dir: (p, prev) => ({ rvcSlotDir: typeof p.rvc_slot_dir === "string" ? p.rvc_slot_dir : prev.rvcSlotDir }),
    rvc_hubert_path: (p, prev) => ({
        rvcHubertPath: typeof p.rvc_hubert_path === "string" ? p.rvc_hubert_path : prev.rvcHubertPath,
    }),
    rvc_fcpe_path: (p, prev) => ({ rvcFcpePath: typeof p.rvc_fcpe_path === "string" ? p.rvc_fcpe_path : prev.rvcFcpePath }),
    rvc_rmvpe_path: (p, prev) => ({
        rvcRmvpePath: typeof p.rvc_rmvpe_path === "string" ? p.rvc_rmvpe_path : prev.rvcRmvpePath,
    }),
    rvc_preset: (p) => ({ rvcProfile: normalizeRvcProfile(p.rvc_preset) }),
    pitch_extractor: (p) => ({ pitchExtractor: normalizePitchExtractor(p.pitch_extractor) }),
    extra_context_ms: (p, prev) => ({
        extraContextMs: clampFinite(p.extra_context_ms, R.extra_context_ms.min, R.extra_context_ms.max, prev.extraContextMs),
    }),
    crossfade_ms: (p, prev) => ({
        crossfadeMs: clampFinite(p.crossfade_ms, R.crossfade_ms.min, R.crossfade_ms.max, prev.crossfadeMs),
    }),
    index_rate: (p, prev) => ({ indexRate: clampFinite(p.index_rate, R.index_rate.min, R.index_rate.max, prev.indexRate) }),
    protect: (p, prev) => ({ protect: clampFinite(p.protect, R.protect.min, R.protect.max, prev.protect) }),
    speaker_id: (p, prev) => ({ speakerId: uint(p.speaker_id, prev.speakerId) }),
    gpu_device_index: (p, prev) => ({ gpuDeviceIndex: uint(p.gpu_device_index, prev.gpuDeviceIndex) }),
    silence_skip: (p, prev) => ({ silenceSkip: typeof p.silence_skip === "boolean" ? p.silence_skip : prev.silenceSkip }),
    vram_cap_mb: (p, prev) => ({
        vramCapMb: Math.round(clampFinite(p.vram_cap_mb, R.vram_cap_mb.min, R.vram_cap_mb.max, prev.vramCapMb)),
    }),
    // P0 fields: absent (older backend) keeps the current value.
    latency_budget_ms: (p, prev) => ({
        latencyBudgetMs: p.latency_budget_ms == null ? prev.latencyBudgetMs : clampLatencyBudgetMs(p.latency_budget_ms),
    }),
    aux_mic_gate: (p, prev) => ({ auxMicGate: p.aux_mic_gate == null ? prev.auxMicGate : normalizeAuxMicGate(p.aux_mic_gate) }),
    worker_priority: (p, prev) => ({
        workerPriority: p.worker_priority == null ? prev.workerPriority : normalizeWorkerPriority(p.worker_priority),
    }),
    input_device: (p, prev) => ({ inputDevice: typeof p.input_device === "string" ? p.input_device : prev.inputDevice }),
    rvc_consent_acknowledged: (p, prev) => ({
        rvcConsentAcknowledged:
            typeof p.rvc_consent_acknowledged === "boolean" ? p.rvc_consent_acknowledged : prev.rvcConsentAcknowledged,
    }),
};

/** The state after hydrating every field of `p` (as `get_voice_changer_params` returns it) onto `prev`. */
export function applyRustParams(prev: VoiceChangerState, p: Partial<VoiceChangerParamsRust>): VoiceChangerState {
    let next: VoiceChangerState = { ...prev };
    for (const key of RUST_PARAM_KEYS) next = { ...next, ...FROM_RUST[key](p, next) };
    return next;
}

/** Relative tolerance for numbers: Rust stores several params as `f32`, so e.g. 1/3 comes back as 0.33333334. */
const NUMBER_TOLERANCE = 1e-6;

/** Whether two IPC values are the same param value (numbers within the f32 round-trip tolerance). */
export function sameParamValue(a: unknown, b: unknown): boolean {
    if (typeof a === "number" && typeof b === "number") {
        if (a === b) return true;
        if (!Number.isFinite(a) || !Number.isFinite(b)) return false;
        return Math.abs(a - b) <= NUMBER_TOLERANCE * Math.max(1, Math.abs(a), Math.abs(b));
    }
    return (a ?? null) === (b ?? null);
}

/** Fields never adopted from a `set` reply: `enabled` follows the started / stopped events. */
const NOT_ADOPTED: ReadonlySet<RustParamKey> = new Set<RustParamKey>(["enabled"]);

/**
 * After `set_voice_changer_params(sent)` returned `effective`: the state patch that takes over
 * every value Rust changed (clamped, normalized, or a voice's own `slot.json` applied on entering
 * it), plus the keys it changed. A field the user has edited since `sent` was built (`current`
 * differs from `sent`) is left alone: that edit wins and its own sync follows.
 */
export function adoptAdjustedParams(
    current: VoiceChangerState,
    sent: VoiceChangerParamsRust,
    effective: Partial<VoiceChangerParamsRust>,
): { patch: Patch; adjusted: RustParamKey[] } {
    const now = toRustParams(current);
    let next = current;
    const adjusted: RustParamKey[] = [];
    for (const key of RUST_PARAM_KEYS) {
        if (NOT_ADOPTED.has(key) || !(key in effective)) continue;
        const eff = effective[key];
        if (eff == null) continue; // a P0 field an older backend does not report
        if (!sameParamValue(now[key], sent[key])) continue; // edited since the send
        if (sameParamValue(eff, sent[key])) continue;
        const patch = FROM_RUST[key](effective, next);
        next = { ...next, ...patch };
        adjusted.push(key);
    }
    const patch: Patch = {};
    for (const [k, v] of Object.entries(next) as [keyof VoiceChangerState, unknown][]) {
        if (!Object.is(v, current[k])) (patch as Record<string, unknown>)[k] = v;
    }
    return { patch, adjusted };
}
