import { ISTTReceiver, ISpeechRecognitionService } from "../types";
import { STT_State } from "../schema";
import { invoke } from "@tauri-apps/api/core";
import { listen, UnlistenFn } from "@tauri-apps/api/event";

// Available Vosk models - now fetched from Rust backend when available
export const VOSK_MODELS = [
    { id: "vosk-model-small-en-us-0.15", name: "English (US)", size: "40 MB" },
    { id: "vosk-model-small-en-in-0.4", name: "English (India)", size: "36 MB" },
    { id: "vosk-model-small-cn-0.22", name: "Chinese", size: "42 MB" },
    { id: "vosk-model-small-ru-0.22", name: "Russian", size: "45 MB" },
    { id: "vosk-model-small-de-0.15", name: "German", size: "45 MB" },
    { id: "vosk-model-small-fr-0.22", name: "French", size: "41 MB" },
    { id: "vosk-model-small-es-0.42", name: "Spanish", size: "39 MB" },
    { id: "vosk-model-small-pt-0.3", name: "Portuguese", size: "31 MB" },
    { id: "vosk-model-small-it-0.22", name: "Italian", size: "48 MB" },
    { id: "vosk-model-small-ja-0.22", name: "Japanese", size: "48 MB" },
    { id: "vosk-model-small-ko-0.22", name: "Korean", size: "82 MB" },
];

// Default CDN for Vosk models
const MODEL_CDN_BASE = "https://alphacephei.com/vosk/models";

interface AudioChunkPayload {
    samples: number[];
    sample_rate: number;
    channels: number;
}

export class STT_VoskService implements ISpeechRecognitionService {
    private model: any = null;
    private recognizer: any = null;
    private isRunning = false;
    private unlistenAudio?: UnlistenFn;

    constructor(private readonly receiver: ISTTReceiver) { }

    async start(params: STT_State) {
        try {
            console.log("[Vosk] Starting service...");
            this.receiver.onInterim("Loading Vosk...");

            // Determine model URL
            const modelId = params.vosk.model || "vosk-model-small-en-us-0.15";
            let modelUrl = params.vosk.modelUrl || `${MODEL_CDN_BASE}/${modelId}.zip`;

            // Try to check if model is already downloaded via Rust backend
            try {
                const downloadedModels = await invoke<string[]>("plugin:vosk_stt|list_downloaded_vosk_models");
                if (downloadedModels.includes(modelId)) {
                    console.log(`[Vosk] Model ${modelId} already downloaded via Rust backend`);
                }
            } catch {
                // Rust backend not available, use CDN directly
            }

            console.log(`[Vosk] Loading model: ${modelId}`);
            this.receiver.onInterim(`Downloading model: ${modelId}...`);

            // Use vosk-browser WASM for transcription (most compatible)
            const { createModel, KaldiRecognizer } = await import("vosk-browser");

            // Load the model (this may take a while on first load)
            this.model = await createModel(modelUrl);
            console.log("[Vosk] Model loaded successfully");

            // Create recognizer with sample rate (Vosk works best at 16000)
            const sampleRate = 16000;
            this.recognizer = new KaldiRecognizer(this.model, sampleRate);

            // Set up result handlers
            this.recognizer.on("result", (message: any) => {
                const text = message.result?.text || "";
                if (text.trim()) {
                    console.log("[Vosk] Final result:", text);
                    this.receiver.onFinal(text);
                }
            });

            this.recognizer.on("partialresult", (message: any) => {
                const partial = message.result?.partial || "";
                if (partial.trim()) {
                    console.log("[Vosk] Partial result:", partial);
                    this.receiver.onInterim(partial);
                }
            });

            // Set up Rust audio capture listener
            console.log("[Vosk] Setting up Rust audio capture...");
            this.unlistenAudio = await listen<AudioChunkPayload>("audio:chunk", (event) => {
                if (!this.isRunning || !this.recognizer) return;

                const { samples } = event.payload;

                // Convert f32 samples to Int16Array for Vosk
                const int16Data = new Int16Array(samples.length);
                for (let i = 0; i < samples.length; i++) {
                    int16Data[i] = Math.max(-32768, Math.min(32767, Math.round(samples[i] * 32768)));
                }

                this.recognizer.acceptWaveform(int16Data);
            });

            // Start Rust audio capture with device selection
            const deviceName = params.vosk.device || window.ApiServer.state.audioInputDevice || "";
            console.log("[Vosk] Starting Rust audio capture with device:", deviceName || "default");

            await invoke("plugin:audio|start_audio_capture", {
                deviceName: deviceName || null,
                sampleRate: sampleRate,
            });

            this.isRunning = true;
            this.receiver.onStart();
            console.log("[Vosk] Recording started via Rust audio capture");

        } catch (error) {
            console.error("[Vosk] Error starting:", error);
            this.receiver.onStop(String(error));
        }
    }

    async stop() {
        if (!this.isRunning) return;

        console.log("[Vosk] Stopping...");
        this.isRunning = false;

        // Stop Rust audio capture
        try {
            await invoke("plugin:audio|stop_audio_capture");
        } catch (error) {
            console.error("[Vosk] Error stopping audio capture:", error);
        }

        // Clean up event listener
        if (this.unlistenAudio) {
            this.unlistenAudio();
            this.unlistenAudio = undefined;
        }

        // Clean up recognizer
        if (this.recognizer) {
            this.recognizer.remove();
            this.recognizer = null;
        }

        this.receiver.onStop();
        console.log("[Vosk] Stopped");
    }

    dispose() {
        this.stop();

        // Clean up model
        if (this.model) {
            this.model.terminate();
            this.model = null;
        }
    }
}
