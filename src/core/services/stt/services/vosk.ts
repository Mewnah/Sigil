import { ISTTReceiver, ISpeechRecognitionService } from "../types";
import { STT_State } from "../schema";
import { invoke } from "@tauri-apps/api/core";
import { listen, UnlistenFn } from "@tauri-apps/api/event";
import { BaseDirectory } from "@tauri-apps/api/path";
import { readFile } from "@tauri-apps/plugin-fs";
import { devLog } from "@/utils/devLog";

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

/** Loaded model from `vosk-browser` `createModel()` — `KaldiRecognizer` is a getter, not a module export. */
type VoskLoadedModel = {
    KaldiRecognizer: new (sampleRate: number, grammar?: string) => VoskRecognizer;
    terminate(): void;
};

type VoskRecognizer = {
    remove(): void;
    acceptWaveformFloat(buffer: Float32Array, sampleRate: number): void;
    on(event: string, cb: (msg: unknown) => void): void;
};

function toMonoFloat32(samples: number[], channels: number): Float32Array {
    if (channels <= 1) {
        return new Float32Array(samples);
    }
    const n = Math.floor(samples.length / channels);
    const out = new Float32Array(n);
    for (let i = 0; i < n; i++) {
        let sum = 0;
        for (let c = 0; c < channels; c++) {
            sum += samples[i * channels + c]!;
        }
        out[i] = sum / channels;
    }
    return out;
}

export class STT_VoskService implements ISpeechRecognitionService {
    private model: VoskLoadedModel | null = null;
    private recognizer: VoskRecognizer | null = null;
    private isRunning = false;
    private unlistenAudio?: UnlistenFn;

    constructor(private readonly receiver: ISTTReceiver) { }

    #ensureRecognizer(sampleRate: number) {
        if (this.recognizer || !this.model) return;
        const Rec = this.model.KaldiRecognizer;
        this.recognizer = new Rec(sampleRate);

        this.recognizer.on("result", (msg: unknown) => {
            const message = msg as { result?: { text?: string } };
            const text = message.result?.text || "";
            if (text.trim()) {
                devLog("[Vosk] Final result:", text);
                this.receiver.onFinal(text);
            }
        });

        this.recognizer.on("partialresult", (msg: unknown) => {
            const message = msg as { result?: { partial?: string } };
            const partial = message.result?.partial || "";
            if (partial.trim()) {
                devLog("[Vosk] Partial result:", partial);
                this.receiver.onInterim(partial);
            }
        });
    }

    async start(params: STT_State) {
        try {
            devLog("[Vosk] Starting service...");
            this.receiver.onInterim("Loading Vosk...");

            const modelId = params.vosk.model || "vosk-model-small-en-us-0.15";
            const customUrl = params.vosk.modelUrl?.trim() || null;

            this.receiver.onInterim(`Ensuring model on disk: ${modelId}...`);

            const relPath = await invoke<string>("plugin:vosk-stt|download_vosk_model", {
                modelId,
                urlOverride: customUrl,
            });

            const bytes = await readFile(relPath, { baseDir: BaseDirectory.AppData });
            const blob = new Blob([bytes], { type: "application/zip" });
            const objectUrl = URL.createObjectURL(blob);

            devLog(`[Vosk] Loading model from local zip (${bytes.byteLength} bytes)`);
            this.receiver.onInterim(`Loading Vosk model (${modelId})...`);

            const { createModel } = await import("vosk-browser");

            try {
                this.model = (await createModel(objectUrl)) as VoskLoadedModel;
            } finally {
                URL.revokeObjectURL(objectUrl);
            }

            devLog("[Vosk] Model loaded successfully");

            const deviceName = params.vosk.device || window.ApiServer.state.audioInputDevice || "";
            devLog("[Vosk] Setting up Rust audio capture...");

            this.unlistenAudio = await listen<AudioChunkPayload>("audio:chunk", (event) => {
                if (!this.isRunning || !this.model) return;

                const { samples, sample_rate, channels } = event.payload;
                this.#ensureRecognizer(sample_rate);
                if (!this.recognizer) return;

                const floats = toMonoFloat32(samples, channels);
                this.recognizer.acceptWaveformFloat(floats, sample_rate);
            });

            devLog("[Vosk] Starting Rust audio capture with device:", deviceName || "default");

            // Set before capture starts so the first `audio:chunk` is not dropped (Rust uses device native rate).
            this.isRunning = true;

            await invoke("plugin:audio|start_audio_capture", {
                deviceName: deviceName || null,
                sampleRate: 16000,
            });

            this.receiver.onStart();
            devLog("[Vosk] Recording started via Rust audio capture");

        } catch (error) {
            console.error("[Vosk] Error starting:", error);
            this.isRunning = false;
            if (this.unlistenAudio) {
                this.unlistenAudio();
                this.unlistenAudio = undefined;
            }
            this.receiver.onStop(String(error));
        }
    }

    async stop() {
        if (!this.isRunning) return;

        devLog("[Vosk] Stopping...");
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
        devLog("[Vosk] Stopped");
    }

    dispose() {
        void this.stop();

        if (this.model) {
            this.model.terminate();
            this.model = null;
        }
    }
}
