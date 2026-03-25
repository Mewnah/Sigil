import { invoke } from "@tauri-apps/api/core";
import { listen, UnlistenFn } from "@tauri-apps/api/event";
import { ISTTReceiver, ISpeechRecognitionService } from "../types";
import { STT_State } from "../schema";

interface AudioChunkPayload {
    samples: number[];
    sample_rate: number;
    channels: number;
}

export class STT_MoonshineService implements ISpeechRecognitionService {
    private isRunning = false;
    private unlistenAudio?: UnlistenFn;
    private audioBuffer: Float32Array[] = [];
    private processingInterval?: ReturnType<typeof setInterval>;

    constructor(private readonly receiver: ISTTReceiver) { }

    async start(params: STT_State) {
        try {
            console.log("[Moonshine] Starting service...");
            this.receiver.onInterim("Connecting to Moonshine...");

            const config = params.moonshine;

            // Set endpoint in backend
            await invoke("plugin:moonshine_stt|set_moonshine_endpoint", {
                endpoint: config.endpoint,
            });

            // Check availability
            const isAvailable = await invoke<boolean>("plugin:moonshine_stt|check_moonshine_availability");
            if (!isAvailable) {
                const setupInstructions = [
                    `Moonshine server not found at ${config.endpoint}`,
                    "",
                    "To start Moonshine, run:",
                    "docker run -p 8090:8090 useful-sensors/moonshine-onnx-server",
                    "",
                    "Or use the tiny model for faster startup:",
                    "docker run -p 8090:8090 useful-sensors/moonshine-onnx-server:tiny",
                ].join("\n");

                this.receiver.onStop(setupInstructions);
                return;
            }

            console.log("[Moonshine] Server connected");

            // Set up Rust audio capture listener
            this.unlistenAudio = await listen<AudioChunkPayload>("audio:chunk", (event) => {
                if (!this.isRunning) return;
                const { samples } = event.payload;
                this.audioBuffer.push(new Float32Array(samples));
            });

            // Start audio capture
            const deviceName = config.device || window.ApiServer.state.audioInputDevice || "";
            await invoke("plugin:audio|start_audio_capture", {
                deviceName: deviceName || null,
                sampleRate: 16000,
            });

            // Process audio chunks periodically (every 500ms for near-real-time)
            this.processingInterval = setInterval(async () => {
                await this.processAudioBuffer(config.language);
            }, 500);

            this.isRunning = true;
            this.receiver.onStart();
            console.log("[Moonshine] Recording started");

        } catch (error) {
            console.error("[Moonshine] Error starting:", error);
            this.receiver.onStop(String(error));
        }
    }

    private async processAudioBuffer(language: string) {
        if (this.audioBuffer.length === 0) return;

        // Combine all chunks
        const totalLength = this.audioBuffer.reduce((acc, chunk) => acc + chunk.length, 0);
        if (totalLength < 4000) return; // At least 250ms at 16kHz for better accuracy

        const combined = new Float32Array(totalLength);
        let offset = 0;
        for (const chunk of this.audioBuffer) {
            combined.set(chunk, offset);
            offset += chunk.length;
        }
        this.audioBuffer = [];

        // Show interim status while processing
        this.receiver.onInterim("...");

        // Convert to base64-encoded WAV for API
        const audioBase64 = this.floatToBase64Wav(combined, 16000);

        try {
            const text = await invoke<string>("plugin:moonshine_stt|moonshine_transcribe", {
                audioBase64,
                language: language || null,
            });

            if (text.trim()) {
                console.log("[Moonshine] Result:", text);
                this.receiver.onFinal(text);
            }
        } catch (error) {
            console.error("[Moonshine] Transcription error:", error);
            // Don't spam errors, just log
        }
    }

    private floatToBase64Wav(samples: Float32Array, sampleRate: number): string {
        // Create WAV header + data
        const numChannels = 1;
        const bitsPerSample = 16;
        const byteRate = sampleRate * numChannels * (bitsPerSample / 8);
        const blockAlign = numChannels * (bitsPerSample / 8);
        const dataSize = samples.length * (bitsPerSample / 8);
        const headerSize = 44;
        const totalSize = headerSize + dataSize;

        const buffer = new ArrayBuffer(totalSize);
        const view = new DataView(buffer);

        // RIFF header
        this.writeString(view, 0, "RIFF");
        view.setUint32(4, totalSize - 8, true);
        this.writeString(view, 8, "WAVE");

        // fmt chunk
        this.writeString(view, 12, "fmt ");
        view.setUint32(16, 16, true); // chunk size
        view.setUint16(20, 1, true); // PCM
        view.setUint16(22, numChannels, true);
        view.setUint32(24, sampleRate, true);
        view.setUint32(28, byteRate, true);
        view.setUint16(32, blockAlign, true);
        view.setUint16(34, bitsPerSample, true);

        // data chunk
        this.writeString(view, 36, "data");
        view.setUint32(40, dataSize, true);

        // Convert float samples to int16
        let offset = 44;
        for (let i = 0; i < samples.length; i++) {
            const sample = Math.max(-1, Math.min(1, samples[i]));
            const int16 = sample < 0 ? sample * 0x8000 : sample * 0x7FFF;
            view.setInt16(offset, int16, true);
            offset += 2;
        }

        // Convert to base64
        const bytes = new Uint8Array(buffer);
        let binary = "";
        for (let i = 0; i < bytes.length; i++) {
            binary += String.fromCharCode(bytes[i]);
        }
        return btoa(binary);
    }

    private writeString(view: DataView, offset: number, str: string) {
        for (let i = 0; i < str.length; i++) {
            view.setUint8(offset + i, str.charCodeAt(i));
        }
    }

    async stop() {
        if (!this.isRunning) return;

        console.log("[Moonshine] Stopping...");
        this.isRunning = false;

        if (this.processingInterval) {
            clearInterval(this.processingInterval);
            this.processingInterval = undefined;
        }

        try {
            await invoke("plugin:audio|stop_audio_capture");
        } catch (error) {
            console.error("[Moonshine] Error stopping audio capture:", error);
        }

        if (this.unlistenAudio) {
            this.unlistenAudio();
            this.unlistenAudio = undefined;
        }

        this.audioBuffer = [];
        this.receiver.onStop();
        console.log("[Moonshine] Stopped");
    }

    dispose() {
        this.stop();
    }
}
