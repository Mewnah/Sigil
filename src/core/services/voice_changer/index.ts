import { invoke } from "@tauri-apps/api/core";
import { listen, UnlistenFn } from "@tauri-apps/api/event";
import { proxy } from "valtio";
import { VoiceChangerState, VoiceChangerStateSchema } from "./schema";

export interface VoiceChangerPreset {
    id: string;
    name: string;
    pitch: number;
    formant: number;
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
        console.log("[VoiceChanger] Initializing service");

        // Fetch presets from backend
        try {
            const presets = await invoke<VoiceChangerPreset[]>("plugin:voice_changer|get_voice_changer_presets");
            this.state.presets = presets;
        } catch (error) {
            console.error("[VoiceChanger] Failed to load presets:", error);
        }

        // Listen for start/stop events
        this.unlistenStart = await listen("voice_changer:started", () => {
            this.state.isRunning = true;
        });

        this.unlistenStop = await listen("voice_changer:stopped", () => {
            this.state.isRunning = false;
        });

        // Check if already running
        try {
            this.state.isRunning = await invoke<boolean>("plugin:voice_changer|is_voice_changer_running");
        } catch {
            this.state.isRunning = false;
        }
    }

    async start() {
        if (this.state.isRunning) return;

        try {
            // Update backend params before starting
            await this.syncParams();

            await invoke("plugin:voice_changer|start_voice_changer", {
                inputDevice: this.state.voiceChanger.inputDevice || null,
            });
            console.log("[VoiceChanger] Started");
        } catch (error) {
            console.error("[VoiceChanger] Failed to start:", error);
            throw error;
        }
    }

    async stop() {
        if (!this.state.isRunning) return;

        try {
            await invoke("plugin:voice_changer|stop_voice_changer");
            console.log("[VoiceChanger] Stopped");
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
            await invoke("plugin:voice_changer|set_voice_changer_params", {
                params: {
                    enabled: this.state.voiceChanger.enabled,
                    pitch_semitones: this.state.voiceChanger.pitch,
                    formant_shift: this.state.voiceChanger.formant,
                    output_device: this.state.voiceChanger.outputDevice,
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
            await invoke("plugin:voice_changer|apply_voice_changer_preset", {
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

    async setEnabled(enabled: boolean) {
        this.state.voiceChanger.enabled = enabled;
        await invoke("plugin:voice_changer|set_voice_changer_enabled", { enabled });
    }

    dispose() {
        this.unlistenStart?.();
        this.unlistenStop?.();
    }
}
