import { invoke } from "@tauri-apps/api/core";

import { listen, UnlistenFn } from "@tauri-apps/api/event";

import { proxy } from "valtio";

import { normalizeChunkFramesValue } from "./chunkFrames";
import { VoiceChangerState, VoiceChangerStateSchema } from "./schema";

import { devLog } from "@/utils/devLog";



export interface VoiceChangerPreset {

    id: string;

    name: string;

    pitch: number;

    formant: number;

}



export type VocoderOversample = 1 | 2 | 4 | 8 | 16 | 32;



export interface VoiceChangerMetrics {

    buffer_ms: number;

    process_ms: number;

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

    mode: "classic" | "onnx" | "sidecar";

    sidecar_ws_url: string;

    onnx_model_path: string;

    onnx_execution_provider: string;

    latency_preset: string;

    monitor_output_device: string;

    monitor_gain: number;

}



interface VoiceChangerServiceState {

    voiceChanger: VoiceChangerState;

    isRunning: boolean;

    presets: VoiceChangerPreset[];

}



function normalizeChunkFrames(n: number): VoiceChangerState["chunkFrames"] {
    return normalizeChunkFramesValue(n);
}



function normalizeLatencyPreset(s: string): VoiceChangerState["latencyPreset"] {

    if (s === "stable") return "stable";

    if (s === "custom") return "custom";

    return "low_latency";

}



export class VoiceChangerService {

    public state = proxy<VoiceChangerServiceState>({

        voiceChanger: VoiceChangerStateSchema.parse({}),

        isRunning: false,

        presets: [],

    });



    private unlistenStart?: UnlistenFn;

    private unlistenStop?: UnlistenFn;



    async init() {

        devLog("[VoiceChanger] Initializing service");



        try {

            const presets = await invoke<VoiceChangerPreset[]>("plugin:voice-changer|get_voice_changer_presets");

            this.state.presets = presets;

        } catch (error) {

            console.error("[VoiceChanger] Failed to load presets:", error);

        }



        this.unlistenStart = await listen("voice_changer:started", () => {

            this.state.isRunning = true;

            this.state.voiceChanger.enabled = true;

        });



        this.unlistenStop = await listen("voice_changer:stopped", () => {

            this.state.isRunning = false;

            this.state.voiceChanger.enabled = false;

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

            vc.mode = p.mode === "sidecar" || p.mode === "onnx" ? p.mode : "classic";

            vc.sidecarWsUrl = p.sidecar_ws_url || "";

            vc.onnxModelPath = p.onnx_model_path || "";

            const ep = p.onnx_execution_provider;

            vc.onnxExecutionProvider = ep === "cuda" || ep === "dml" ? ep : "cpu";

            vc.latencyPreset = normalizeLatencyPreset(p.latency_preset);

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

        try {

            await this.syncParams();



            await invoke("plugin:voice-changer|start_voice_changer", {

                inputDevice: this.state.voiceChanger.inputDevice || null,

            });

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

        await this.restartIfRunning();

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

                    latency_preset: this.state.voiceChanger.latencyPreset,

                    monitor_output_device: this.state.voiceChanger.monitorOutputDevice,

                    monitor_gain: this.state.voiceChanger.monitorGain,

                },

            });

        } catch (error) {

            console.error("[VoiceChanger] Failed to sync params:", error);

        }

    }



    async applyPreset(presetId: string) {

        const preset = this.state.presets.find(p => p.id === presetId);

        if (!preset) return;



        this.state.voiceChanger.pitch = preset.pitch;

        this.state.voiceChanger.formant = preset.formant;

        this.state.voiceChanger.preset = presetId;



        try {

            await invoke("plugin:voice-changer|apply_voice_changer_preset", {

                presetId,

            });

        } catch (error) {

            console.error("[VoiceChanger] Failed to apply preset:", error);

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

    }

}


