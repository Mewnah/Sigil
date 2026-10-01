/**
 * What a params change does to a running voice changer (P3-G), mirroring how the Rust worker reads
 * each param (`src-tauri/src/services/voice_changer/worker.rs`):
 *
 * - `hot`: read while running (every block, or on change), e.g. pitch, gains, the latency budget,
 *   index rate / protect / speaker, and the P3 voice controls (`params_contract.json`
 *   `params_p3_hot_fields`: rms_mix, formant_semitones, noise_scale, silence_db).
 * - `switch`: RVC mode swaps the voice without a restart when the selection changes (voice folder,
 *   F0 method, F0 range): the incoming voice loads in the background and is crossfaded in over one
 *   block (`VoiceSet`, plan T9), so these never restart the engine.
 * - `restart`: read only when the engine starts (devices, block size, the mode's model / bridge,
 *   the RVC engine shape: EP, preset, precision, block, context, levers, hot slots, VRAM, denoiser).
 * - `idle`: belongs to another mode than the one running (it applies when that mode starts).
 *
 * Pure module: no Tauri, DOM or `window`.
 */
import type { VcMode } from "./contract";
import { RUST_PARAM_KEYS, sameParamValue, type RustParamKey, type VoiceChangerParamsRust } from "./params";

export type ParamApply = "hot" | "switch" | "restart" | "idle";

/** How a param applies, by the mode it belongs to (`all` = every mode). */
type Rule = { modes: "all" | readonly VcMode[]; apply: "hot" | "switch" | "restart" };

const ALL_HOT: Rule = { modes: "all", apply: "hot" };
const ALL_RESTART: Rule = { modes: "all", apply: "restart" };
const RVC_HOT: Rule = { modes: ["rvc"], apply: "hot" };
const RVC_SWITCH: Rule = { modes: ["rvc"], apply: "switch" };
const RVC_RESTART: Rule = { modes: ["rvc"], apply: "restart" };

/** Every Rust param (the `Record` makes a new param a type error until it has a rule). */
const RULES: Record<RustParamKey, Rule> = {
    // `enabled` follows start / stop, never a sync.
    enabled: ALL_HOT,
    pitch_semitones: ALL_HOT,
    formant_shift: ALL_HOT,
    output_device: ALL_RESTART,
    vocoder_window_ms: { modes: ["classic"], apply: "hot" },
    vocoder_oversample: { modes: ["classic"], apply: "hot" },
    chunk_frames: ALL_RESTART,
    input_gain: ALL_HOT,
    output_gain: ALL_HOT,
    output_prefill_ms: ALL_HOT,
    output_max_queue_ms: ALL_HOT,
    buffer_truncate_enabled: ALL_HOT,
    buffer_truncate_rms: ALL_HOT,
    buffer_truncate_chunks: ALL_HOT,
    mode: ALL_RESTART,
    sidecar_ws_url: { modes: ["sidecar"], apply: "restart" },
    onnx_model_path: { modes: ["onnx"], apply: "restart" },
    onnx_execution_provider: { modes: ["onnx"], apply: "restart" },
    rvc_execution_provider: RVC_RESTART,
    latency_preset: ALL_HOT,
    monitor_output_device: ALL_RESTART,
    monitor_gain: ALL_HOT,
    rvc_slot_dir: RVC_SWITCH,
    rvc_hubert_path: RVC_RESTART,
    rvc_fcpe_path: RVC_RESTART,
    rvc_rmvpe_path: RVC_RESTART,
    rvc_preset: RVC_RESTART,
    pitch_extractor: RVC_SWITCH,
    extra_context_ms: RVC_RESTART,
    crossfade_ms: RVC_RESTART,
    index_rate: RVC_HOT,
    protect: RVC_HOT,
    speaker_id: RVC_HOT,
    gpu_device_index: RVC_RESTART,
    silence_skip: RVC_RESTART,
    vram_cap_mb: RVC_RESTART,
    latency_budget_ms: ALL_HOT,
    aux_mic_gate: ALL_HOT,
    worker_priority: ALL_HOT,
    input_device: ALL_RESTART,
    rvc_consent_acknowledged: ALL_HOT,
    // P3
    slot_id: RVC_SWITCH,
    block_ms: RVC_RESTART,
    // Engine shape (the plan's windows and context).
    lookahead_ms: RVC_RESTART,
    precision: RVC_RESTART,
    f0_range: RVC_SWITCH,
    rms_mix: RVC_HOT,
    formant_semitones: RVC_HOT,
    noise_scale: RVC_HOT,
    silence_db: RVC_HOT,
    partial_decode: RVC_RESTART,
    anchored_noise: RVC_RESTART,
    nsf_phase0: RVC_RESTART,
    f0_cache: RVC_RESTART,
    gpu_knn: RVC_RESTART,
    io_binding: RVC_RESTART,
    cuda_graphs: RVC_RESTART,
    gpu_priority: RVC_RESTART,
    governor: RVC_RESTART,
    // The device-rate denoiser stage is built at start in every mode.
    denoiser: ALL_RESTART,
    hot_slots: RVC_RESTART,
    vram_budget_mb: RVC_RESTART,
};

/** How a change of `key` applies while the engine runs in `mode`. */
export function paramApply(key: RustParamKey, mode: VcMode): ParamApply {
    const rule = RULES[key];
    if (!rule) return "restart";
    if (rule.modes !== "all" && !rule.modes.includes(mode)) return "idle";
    return rule.apply;
}

/** The params whose value differs between `a` and `b` (numbers within the f32 round-trip tolerance). */
export function changedParamKeys(a: VoiceChangerParamsRust, b: VoiceChangerParamsRust): RustParamKey[] {
    return RUST_PARAM_KEYS.filter((k) => k !== "enabled" && !sameParamValue(a[k], b[k]));
}

/**
 * Whether a running engine must restart for `changed` to take effect: a mode change always does
 * (the new mode is built from scratch); otherwise any change whose rule is `restart` in `mode`.
 */
export function restartNeeded(changed: readonly RustParamKey[], mode: VcMode): boolean {
    return changed.some((k) => paramApply(k, mode) === "restart");
}
