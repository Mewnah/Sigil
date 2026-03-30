import { ISTTReceiver, ISpeechRecognitionService } from "../types";
import { STT_State } from "../schema";
import { invoke } from "@tauri-apps/api/core";
import { listen, UnlistenFn } from "@tauri-apps/api/event";
import { BaseDirectory } from "@tauri-apps/api/path";
import { readFile } from "@tauri-apps/plugin-fs";

/** Mirrors Rust `VoskModel` for `get_vosk_models`. */
export type VoskCatalogModel = {
    id: string;
    name: string;
    language: string;
    size: string;
    url: string;
};

/** Inspector fallback if `invoke` fails (e.g. non-Tauri dev). */
export const VOSK_MODELS_FALLBACK: VoskCatalogModel[] = [
    {
        id: "vosk-model-small-en-us-0.15",
        name: "English (US)",
        language: "en",
        size: "40 MB",
        url: "https://alphacephei.com/vosk/models/vosk-model-small-en-us-0.15.zip",
    },
];

interface AudioChunkPayload {
    samples: number[];
    sample_rate: number;
    channels: number;
}

export class STT_VoskService implements ISpeechRecognitionService {
    private model: unknown = null;
    private recognizer: {
        remove(): void;
        acceptWaveform(data: Int16Array): void;
        on(event: string, cb: (msg: unknown) => void): void;
    } | null = null;
    private isRunning = false;
    private unlistenAudio?: UnlistenFn;

    constructor(private readonly receiver: ISTTReceiver) { }

    async start(params: STT_State) {
        try {
            console.log("[Vosk] Starting service...");
            this.receiver.onInterim("Loading Vosk...");

            const modelId = params.vosk.model || "vosk-model-small-en-us-0.15";
            const customUrl = params.vosk.modelUrl?.trim() || null;

            this.receiver.onInterim(`Ensuring model on disk: ${modelId}...`);

            const relPath = await invoke<string>("plugin:vosk-stt|download_vosk_model", {
                model_id: modelId,
                url_override: customUrl,
            });

            const bytes = await readFile(relPath, { baseDir: BaseDirectory.AppData });
            const blob = new Blob([bytes], { type: "application/zip" });
            const objectUrl = URL.createObjectURL(blob);

            console.log(`[Vosk] Loading model from local zip (${bytes.byteLength} bytes)`);
            this.receiver.onInterim(`Loading Vosk model (${modelId})...`);

            const { createModel, KaldiRecognizer } = await import("vosk-browser");

            try {
                this.model = await createModel(objectUrl);
            } finally {
                URL.revokeObjectURL(objectUrl);
            }

            console.log("[Vosk] Model loaded successfully");

            const sampleRate = 16000;
            this.recognizer = new KaldiRecognizer(this.model as never, sampleRate);

            this.recognizer.on("result", (msg: unknown) => {
                const message = msg as { result?: { text?: string } };
                const text = message.result?.text || "";
                if (text.trim()) {
                    console.log("[Vosk] Final result:", text);
                    this.receiver.onFinal(text);
                }
            });

            this.recognizer.on("partialresult", (msg: unknown) => {
                const message = msg as { result?: { partial?: string } };
                const partial = message.result?.partial || "";
                if (partial.trim()) {
                    console.log("[Vosk] Partial result:", partial);
                    this.receiver.onInterim(partial);
                }
            });

            console.log("[Vosk] Setting up Rust audio capture...");
            this.unlistenAudio = await listen<AudioChunkPayload>("audio:chunk", (event) => {
                if (!this.isRunning || !this.recognizer) return;

                const { samples } = event.payload;
                const int16Data = new Int16Array(samples.length);
                for (let i = 0; i < samples.length; i++) {
                    int16Data[i] = Math.max(-32768, Math.min(32767, Math.round(samples[i] * 32768)));
                }

                this.recognizer.acceptWaveform(int16Data);
            });

            const deviceName = params.vosk.device || window.ApiServer.state.audioInputDevice || "";
            console.log("[Vosk] Starting Rust audio capture with device:", deviceName || "default");

            await invoke("plugin:audio|start_audio_capture", {
                deviceName: deviceName || null,
                sampleRate,
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

        try {
            await invoke("plugin:audio|stop_audio_capture");
        } catch (error) {
            console.error("[Vosk] Error stopping audio capture:", error);
        }

        if (this.unlistenAudio) {
            this.unlistenAudio();
            this.unlistenAudio = undefined;
        }

        if (this.recognizer) {
            this.recognizer.remove();
            this.recognizer = null;
        }

        this.receiver.onStop();
        console.log("[Vosk] Stopped");
    }

    dispose() {
        void this.stop();

        if (this.model) {
            (this.model as { terminate(): void }).terminate();
            this.model = null;
        }
    }
}
