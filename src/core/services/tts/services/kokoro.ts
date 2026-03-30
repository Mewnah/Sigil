import { invoke } from "@tauri-apps/api/core";
import { ITTSReceiver, ITTSService } from "../types";
import { TTS_State } from "../schema";

export class TTS_KokoroService implements ITTSService {
    #receiver: ITTSReceiver;
    #endpoint: string = "http://localhost:8880";
    #voice: string = "af_bella";
    #speed: number = 1.0;
    #audioContext: AudioContext | null = null;
    #isRunning: boolean = false;

    constructor(receiver: ITTSReceiver) {
        this.#receiver = receiver;
    }

    async start(data: TTS_State): Promise<void> {
        const config = data.kokoro;
        this.#endpoint = config.endpoint;
        this.#voice = config.voice;
        this.#speed = parseFloat(config.speed);

        try {
            // Set endpoint in backend
            await invoke("plugin:kokoro-tts|set_kokoro_endpoint", {
                endpoint: this.#endpoint,
            });

            // Check availability
            const isAvailable = await invoke<boolean>("plugin:kokoro-tts|check_kokoro_availability");
            if (!isAvailable) {
                this.#receiver.onStop(`Kokoro TTS not available at ${this.#endpoint}. Start with: docker run -p 8880:8880 ghcr.io/remsky/kokoro-fastapi-cpu`);
                return;
            }

            this.#audioContext = new AudioContext();
            this.#isRunning = true;
            this.#receiver.onStart();
            console.log("[Kokoro] TTS service started");
        } catch (error) {
            console.error("[Kokoro] Failed to start:", error);
            this.#receiver.onStop(String(error));
        }
    }

    stop(): void {
        this.#isRunning = false;
        if (this.#audioContext) {
            this.#audioContext.close();
            this.#audioContext = null;
        }
        console.log("[Kokoro] TTS service stopped");
    }

    async play(text: string): Promise<void> {
        if (!this.#isRunning || !text.trim()) return;

        try {
            // Get audio bytes from backend
            const audioBytes = await invoke<number[]>("plugin:kokoro-tts|kokoro_speak", {
                text,
                voice: this.#voice,
                speed: this.#speed,
            });

            if (!audioBytes || audioBytes.length === 0) return;

            // Convert to Uint8Array and play
            const uint8Array = new Uint8Array(audioBytes);
            await this.playAudio(uint8Array);
        } catch (error) {
            console.error("[Kokoro] Speech synthesis error:", error);
        }
    }

    private async playAudio(audioData: Uint8Array): Promise<void> {
        if (!this.#audioContext) {
            this.#audioContext = new AudioContext();
        }

        try {
            const audioBuffer = await this.#audioContext.decodeAudioData(audioData.buffer.slice(0) as ArrayBuffer);
            const source = this.#audioContext.createBufferSource();
            source.buffer = audioBuffer;
            source.connect(this.#audioContext.destination);
            source.start(0);
        } catch (error) {
            console.error("[Kokoro] Audio playback error:", error);
        }
    }

    dispose(): void {
        this.stop();
    }

    // Static helper to get available voices
    static async getVoices(endpoint: string = "http://localhost:8880"): Promise<{ name: string }[]> {
        try {
            await invoke("plugin:kokoro-tts|set_kokoro_endpoint", { endpoint });
            return await invoke<{ name: string }[]>("plugin:kokoro-tts|get_kokoro_voices");
        } catch {
            return [];
        }
    }
}
