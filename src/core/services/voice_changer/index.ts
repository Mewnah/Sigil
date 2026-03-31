import { invoke } from "@tauri-apps/api/core";
import { listen, UnlistenFn } from "@tauri-apps/api/event";
import { proxy } from "valtio";
import { VoiceChangerState, VoiceChangerStateSchema } from "./schema";
import { devLog } from "@/utils/devLog";

export interface VoiceChangerPreset {
    id: string;
    name: string;
    pitch: number;
    formant: number;
}

export type VocoderOversample = 4 | 8 | 16 | 32;

interface VoiceChangerParamsRust {
    enabled: boolean;
    pitch_semitones: number;
    formant_shift: number;
    output_device: string;
    vocoder_window_ms: number;
    vocoder_oversample: number;
}

interface VoiceChangerServiceState {
    voiceChanger: VoiceChangerState;
    isRunning: boolean;
    presets: VoiceChangerPreset[];
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

        // Fetch presets from backend
        try {
            const presets = await invoke<VoiceChangerPreset[]>("plugin:voice-changer|get_voice_changer_presets");
            this.state.presets = presets;
        } catch (error) {
            console.error("[VoiceChanger] Failed to load presets:", error);
        }

        // Listen for start/stop events
        this.unlistenStart = await listen("voice_changer:started", () => {
            this.state.isRunning = true;
            this.state.voiceChanger.enabled = true;
        });

        this.unlistenStop = await listen("voice_changer:stopped", () => {
            this.state.isRunning = false;
            this.state.voiceChanger.enabled = false;
        });

        // Check if already running
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
            vc.enabled = p.enabled;
            vc.vocoderWindowMs = Math.min(60, Math.max(30, Math.round(p.vocoder_window_ms)));
            const o = p.vocoder_oversample;
            vc.vocoderOversample = o === 4 || o === 8 || o === 16 || o === 32 ? o : 8;
        } catch {
            /* keep schema defaults */
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
        this.state.voiceChanger.pitch = Math.max(-12, Math.min(12, pitch));
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
