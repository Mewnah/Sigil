import { invoke } from "@tauri-apps/api/core";

import { listen, UnlistenFn } from "@tauri-apps/api/event";

import i18n from "i18next";

import { proxy } from "valtio";

import { toast } from "react-toastify";

import { effectiveCrossfadeMs, normalizeChunkFramesValue } from "./chunkFrames";
import {
    ExecutionProvider,
    PitchExtractor,
    RvcProfile,
    VoiceChangerState,
    VoiceChangerStateSchema,
} from "./schema";

import { devLog } from "@/utils/devLog";

export type VocoderOversample = 1 | 2 | 4 | 8 | 16 | 32;

export interface VoiceChangerMetrics {
    buffer_ms: number;
    process_ms: number;
    chunk_ms: number;
}

/** Emitted when ONNX or sidecar conversion fails and audio falls back to dry input. */
export interface VoiceChangerConversionErrorPayload {
    code: string;
    message: string;
}

/** Template output buffer (ms) — matches Rust `PREFILL_MS_*` / `MAX_QUEUE_MS_*`. */
export const OUTPUT_BUFFER_LOW_LATENCY = { prefillMs: 35, maxQueueMs: 70 };
export const OUTPUT_BUFFER_STABLE = { prefillMs: 80, maxQueueMs: 150 };

interface VoiceChangerParamsRust {
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
    mode: "classic" | "onnx" | "sidecar" | "rvc";
    sidecar_ws_url: string;
    onnx_model_path: string;
    /** ONNX chunk mode only. */
    onnx_execution_provider: string;
    /** RVC mode only (HuBERT, pitch extractor, generator). */
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
}

interface VoiceChangerServiceState {
    voiceChanger: VoiceChangerState;
    isRunning: boolean;
}

/**
 * Values the RVC harness reads only at construction; changing them needs an engine restart. Stored as set (not
 * clamped): the crossfade clamp depends on the capture rate, which is only known once the engine runs.
 */
interface HarnessParams {
    extraContextMs: number;
    crossfadeMs: number;
    chunkFrames: number;
}

function normalizeChunkFrames(n: number): VoiceChangerState["chunkFrames"] {
    return normalizeChunkFramesValue(n);
}

function normalizeLatencyPreset(s: string): VoiceChangerState["latencyPreset"] {
    if (s === "stable") return "stable";
    if (s === "custom") return "custom";
    return "low_latency";
}

function normalizeMode(s: string): VoiceChangerState["mode"] {
    if (s === "sidecar" || s === "onnx" || s === "rvc") return s;
    return "classic";
}

function normalizeRvcProfile(s: string): RvcProfile {
    if (s === "low_latency" || s === "quality") return s;
    return "streamer";
}

function normalizePitchExtractor(s: string): PitchExtractor {
    const p = s.toLowerCase();
    if (p === "rmvpe" || p === "crepe" || p === "dio") return p;
    return "fcpe";
}

function normalizeExecutionProvider(s: string | undefined): ExecutionProvider {
    return s === "cuda" || s === "dml" ? s : "cpu";
}

function harnessParamsOf(vc: VoiceChangerState): HarnessParams {
    return {
        extraContextMs: vc.extraContextMs,
        crossfadeMs: vc.crossfadeMs,
        chunkFrames: vc.chunkFrames,
    };
}

export class VoiceChangerService {
    public state = proxy<VoiceChangerServiceState>({
        voiceChanger: VoiceChangerStateSchema.parse({}),
        isRunning: false,
    });

    private unlistenStart?: UnlistenFn;
    private unlistenStop?: UnlistenFn;
    private unlistenError?: UnlistenFn;

    /** Harness params the running engine was last started (or restarted) with; null when unknown. */
    private appliedHarness: HarnessParams | null = null;

    async init() {
        devLog("[VoiceChanger] Initializing service");

        this.unlistenStart = await listen("voice_changer:started", () => {
            this.state.isRunning = true;
            this.state.voiceChanger.enabled = true;
        });

        this.unlistenStop = await listen("voice_changer:stopped", () => {
            this.state.isRunning = false;
            this.state.voiceChanger.enabled = false;
        });

        this.unlistenError = await listen<VoiceChangerConversionErrorPayload>("voice_changer:error", (ev) => {
            const { code, message } = ev.payload;
            const msg = message?.trim() || "";
            if (code === "onnx_infer") {
                toast.warning(i18n.t("voice_changer.toast_error_onnx", { message: msg }));
            } else if (code === "sidecar_length") {
                toast.warning(i18n.t("voice_changer.toast_error_sidecar_length", { message: msg }));
            } else if (code === "rvc_load" || code === "rvc_convert") {
                toast.warning(i18n.t("voice_changer.toast_error_rvc", { message: msg }));
            } else if (code === "rvc_warning") {
                toast.warning(i18n.t("voice_changer.toast_warning_rvc", { message: msg }));
            } else if (code === "worker_panic") {
                toast.error(i18n.t("voice_changer.toast_error_worker_panic", { message: msg }));
            } else {
                toast.warning(i18n.t("voice_changer.toast_error_sidecar", { message: msg }));
            }
        });

        try {
            this.state.isRunning = await invoke<boolean>("plugin:voice-changer|is_voice_changer_running");
        } catch {
            this.state.isRunning = false;
        }

        try {
            const p = await invoke<VoiceChangerParamsRust>("plugin:voice-changer|get_voice_changer_params");
            const vc = this.state.voiceChanger;
            vc.pitch = p.pitch_semitones;
            vc.formant = p.formant_shift;
            vc.outputDevice = p.output_device;
            vc.monitorOutputDevice = p.monitor_output_device ?? "";
            vc.monitorGain = Math.min(4, Math.max(0, p.monitor_gain ?? 1));
            vc.enabled = p.enabled;
            vc.vocoderWindowMs = Math.min(60, Math.max(30, Math.round(p.vocoder_window_ms)));
            const o = p.vocoder_oversample;
            vc.vocoderOversample =
                o === 1 || o === 2 || o === 4 || o === 8 || o === 16 || o === 32 ? o : 8;
            vc.chunkFrames = normalizeChunkFrames(p.chunk_frames);
            vc.inputGain = Math.min(4, Math.max(0, p.input_gain));
            vc.outputGain = Math.min(4, Math.max(0, p.output_gain));
            vc.outputPrefillMs = p.output_prefill_ms;
            vc.outputMaxQueueMs = p.output_max_queue_ms;
            vc.bufferTruncateEnabled = p.buffer_truncate_enabled;
            vc.bufferTruncateRms = p.buffer_truncate_rms;
            vc.bufferTruncateChunks = Math.min(500, Math.max(1, Math.round(p.buffer_truncate_chunks)));
            vc.mode = normalizeMode(p.mode);
            vc.sidecarWsUrl = p.sidecar_ws_url || "";
            vc.onnxModelPath = p.onnx_model_path || "";
            vc.onnxExecutionProvider = normalizeExecutionProvider(p.onnx_execution_provider);
            vc.rvcExecutionProvider = normalizeExecutionProvider(p.rvc_execution_provider);
            vc.latencyPreset = normalizeLatencyPreset(p.latency_preset);
            vc.rvcSlotDir = p.rvc_slot_dir || "";
            vc.rvcHubertPath = p.rvc_hubert_path || "";
            vc.rvcFcpePath = p.rvc_fcpe_path || "";
            vc.rvcRmvpePath = p.rvc_rmvpe_path || "";
            vc.rvcProfile = normalizeRvcProfile(p.rvc_preset || "");
            vc.pitchExtractor = normalizePitchExtractor(p.pitch_extractor || "");
            vc.extraContextMs = Math.min(2000, Math.max(0, p.extra_context_ms ?? 64));
            vc.crossfadeMs = Math.min(200, Math.max(0, p.crossfade_ms ?? 20));
            vc.indexRate = Math.min(1, Math.max(0, p.index_rate ?? 0));
            vc.protect = Math.min(0.5, Math.max(0, p.protect ?? 0.33));
            vc.speakerId = Math.max(0, Math.round(p.speaker_id ?? 0));
            vc.gpuDeviceIndex = Math.max(0, Math.round(p.gpu_device_index ?? 0));
            vc.silenceSkip = p.silence_skip ?? true;
            vc.vramCapMb = Math.min(8192, Math.max(256, Math.round(p.vram_cap_mb ?? 768)));
            // Already running (e.g. webview reload): the live harness was built from these params.
            if (this.state.isRunning) this.appliedHarness = harnessParamsOf(vc);
        } catch {
            /* keep schema defaults */
        }
    }

    async getMetrics(): Promise<VoiceChangerMetrics | null> {
        try {
            return await invoke<VoiceChangerMetrics>("plugin:voice-changer|get_voice_changer_metrics");
        } catch {
            return null;
        }
    }

    async start() {
        if (this.state.isRunning) return;

        this.state.voiceChanger.enabled = true;
        // Snapshot taken together with the params sent below, so later edits compare against what the harness got.
        const harness = harnessParamsOf(this.state.voiceChanger);
        try {
            await this.syncParams();

            await invoke("plugin:voice-changer|start_voice_changer", {
                inputDevice: this.state.voiceChanger.inputDevice || null,
            });
            this.appliedHarness = harness;
            devLog("[VoiceChanger] Started");
        } catch (error) {
            this.state.voiceChanger.enabled = false;
            console.error("[VoiceChanger] Failed to start:", error);
            throw error;
        }
    }

    async stop() {
        if (!this.state.isRunning) return;

        try {
            await invoke("plugin:voice-changer|stop_voice_changer");
            this.state.isRunning = false;
            this.state.voiceChanger.enabled = false;
            devLog("[VoiceChanger] Stopped");
        } catch (error) {
            console.error("[VoiceChanger] Failed to stop:", error);
        }
    }

    async toggle() {
        if (this.state.isRunning) {
            await this.stop();
        } else {
            await this.start();
        }
    }

    /** Apply low_latency / stable buffer template and sync. */
    async applyLatencyPreset(preset: "low_latency" | "stable") {
        const vc = this.state.voiceChanger;
        vc.latencyPreset = preset;
        if (preset === "stable") {
            vc.outputPrefillMs = OUTPUT_BUFFER_STABLE.prefillMs;
            vc.outputMaxQueueMs = OUTPUT_BUFFER_STABLE.maxQueueMs;
        } else {
            vc.outputPrefillMs = OUTPUT_BUFFER_LOW_LATENCY.prefillMs;
            vc.outputMaxQueueMs = OUTPUT_BUFFER_LOW_LATENCY.maxQueueMs;
        }
        await this.syncParams();
    }

    /**
     * Apply RVC Streamer / Low latency / Quality profile (RVC EP + F0 + index + extra context + VRAM).
     * Extra context follows the plan's minimal / small / higher split (HuBERT re-encodes it every tick).
     * Structural fields restart the engine when running.
     */
    async applyRvcProfile(profile: RvcProfile) {
        const vc = this.state.voiceChanger;
        vc.rvcProfile = profile;
        if (profile === "streamer") {
            vc.rvcExecutionProvider = "cpu";
            vc.pitchExtractor = "fcpe";
            vc.indexRate = 0;
            vc.extraContextMs = 64;
        } else if (profile === "low_latency") {
            vc.rvcExecutionProvider = "cuda";
            vc.pitchExtractor = "rmvpe";
            vc.indexRate = 0;
            vc.extraContextMs = 96;
            vc.vramCapMb = 768;
        } else {
            vc.rvcExecutionProvider = "cuda";
            vc.pitchExtractor = "rmvpe";
            vc.indexRate = 0.5;
            vc.extraContextMs = 192;
            vc.vramCapMb = 1536;
        }
        await this.syncParamsAndRestartIfRunning();
    }

    async listModelSlots(): Promise<{ id: string; path: string; has_manifest: boolean }[]> {
        try {
            return await invoke("plugin:voice-changer|list_voice_changer_model_slots");
        } catch {
            return [];
        }
    }

    async captureDebug(): Promise<Record<string, unknown> | null> {
        try {
            return await invoke("plugin:voice-changer|capture_voice_changer_debug");
        } catch (error) {
            console.error("[VoiceChanger] Debug capture failed:", error);
            return null;
        }
    }

    /** Mark buffer as custom (call when user edits ms fields). */
    markBufferCustom() {
        this.state.voiceChanger.latencyPreset = "custom";
    }

    /** Restart if running so chunk size / buffer caps take effect. */
    async restartIfRunning() {
        if (!this.state.isRunning) return;
        await this.stop();
        await new Promise(r => setTimeout(r, 80));
        await this.start();
    }

    /** Sync Rust params then restart the engine when running (devices, mode, chunk size, bridges). */
    async syncParamsAndRestartIfRunning() {
        await this.syncParams();
        await this.restartIfRunning();
    }

    /**
     * Commit extra context / crossfade (call on slider release or field blur). The RVC harness reads them only
     * at construction, so restart once when they differ from what the running engine was started with; a
     * repeated commit of the same value is a no-op. When stopped, only sync (the next start applies them).
     */
    async commitHarnessParams() {
        const cur = harnessParamsOf(this.state.voiceChanger);
        const applied = this.appliedHarness;
        if (
            this.state.isRunning &&
            applied &&
            applied.extraContextMs === cur.extraContextMs &&
            applied.chunkFrames === cur.chunkFrames
        ) {
            // Compared as the harness sees them: crossfade beyond one chunk is clamped in Rust, so e.g. 20 -> 30 ms
            // at 512 frames is not a change and must not restart the engine. The chunk length comes from the
            // running engine's capture rate (`chunk_ms`), falling back to 48 kHz when metrics are unavailable.
            const chunkMs = (await this.getMetrics())?.chunk_ms;
            if (
                effectiveCrossfadeMs(applied.crossfadeMs, cur.chunkFrames, chunkMs) ===
                effectiveCrossfadeMs(cur.crossfadeMs, cur.chunkFrames, chunkMs)
            ) {
                return;
            }
        }
        await this.syncParamsAndRestartIfRunning();
    }

    async syncParams() {
        try {
            await invoke("plugin:voice-changer|set_voice_changer_params", {
                params: {
                    enabled: this.state.voiceChanger.enabled,
                    pitch_semitones: this.state.voiceChanger.pitch,
                    formant_shift: this.state.voiceChanger.formant,
                    output_device: this.state.voiceChanger.outputDevice,
                    vocoder_window_ms: this.state.voiceChanger.vocoderWindowMs,
                    vocoder_oversample: this.state.voiceChanger.vocoderOversample,
                    chunk_frames: this.state.voiceChanger.chunkFrames,
                    input_gain: this.state.voiceChanger.inputGain,
                    output_gain: this.state.voiceChanger.outputGain,
                    output_prefill_ms: this.state.voiceChanger.outputPrefillMs,
                    output_max_queue_ms: this.state.voiceChanger.outputMaxQueueMs,
                    buffer_truncate_enabled: this.state.voiceChanger.bufferTruncateEnabled,
                    buffer_truncate_rms: this.state.voiceChanger.bufferTruncateRms,
                    buffer_truncate_chunks: this.state.voiceChanger.bufferTruncateChunks,
                    mode: this.state.voiceChanger.mode,
                    sidecar_ws_url: this.state.voiceChanger.sidecarWsUrl,
                    onnx_model_path: this.state.voiceChanger.onnxModelPath,
                    onnx_execution_provider: this.state.voiceChanger.onnxExecutionProvider,
                    rvc_execution_provider: this.state.voiceChanger.rvcExecutionProvider,
                    latency_preset: this.state.voiceChanger.latencyPreset,
                    monitor_output_device: this.state.voiceChanger.monitorOutputDevice,
                    monitor_gain: this.state.voiceChanger.monitorGain,
                    rvc_slot_dir: this.state.voiceChanger.rvcSlotDir,
                    rvc_hubert_path: this.state.voiceChanger.rvcHubertPath,
                    rvc_fcpe_path: this.state.voiceChanger.rvcFcpePath,
                    rvc_rmvpe_path: this.state.voiceChanger.rvcRmvpePath,
                    rvc_preset: this.state.voiceChanger.rvcProfile,
                    pitch_extractor: this.state.voiceChanger.pitchExtractor,
                    extra_context_ms: this.state.voiceChanger.extraContextMs,
                    crossfade_ms: this.state.voiceChanger.crossfadeMs,
                    index_rate: this.state.voiceChanger.indexRate,
                    protect: this.state.voiceChanger.protect,
                    speaker_id: this.state.voiceChanger.speakerId,
                    gpu_device_index: this.state.voiceChanger.gpuDeviceIndex,
                    silence_skip: this.state.voiceChanger.silenceSkip,
                    vram_cap_mb: this.state.voiceChanger.vramCapMb,
                },
            });
        } catch (error) {
            console.error("[VoiceChanger] Failed to sync params:", error);
        }
    }

    async setPitch(pitch: number) {
        this.state.voiceChanger.pitch = Math.max(-24, Math.min(24, pitch));
        await this.syncParams();
    }

    async setFormant(formant: number) {
        this.state.voiceChanger.formant = Math.max(-1, Math.min(1, formant));
        await this.syncParams();
    }

    async setVocoderWindowMs(ms: number) {
        this.state.voiceChanger.vocoderWindowMs = Math.min(60, Math.max(30, Math.round(ms)));
        await this.syncParams();
    }

    async setVocoderOversample(o: VocoderOversample) {
        this.state.voiceChanger.vocoderOversample = o;
        await this.syncParams();
    }

    dispose() {
        this.unlistenStart?.();
        this.unlistenStop?.();
        this.unlistenError?.();
    }
}
