import { invoke } from "@tauri-apps/api/tauri";
import { ITTSReceiver, ITTSService } from "../types";
import { TTS_State } from "../schema";

interface MeloSpeaker {
    id: string;
    name: string;
    language: string;
}

export class TTS_MeloService implements ITTSService {
    #receiver: ITTSReceiver;
    #endpoint: string = "http://localhost:8888";
    #speakerId: string = "";
    #speed: number = 1.0;
    #audioContext: AudioContext | null = null;
    #isRunning: boolean = false;

    constructor(receiver: ITTSReceiver) {
        this.#receiver = receiver;
    }

    async start(data: TTS_State): Promise<void> {
        const config = data.melo;
        this.#endpoint = config.endpoint;
        this.#speakerId = config.speaker;
        this.#speed = parseFloat(config.speed);

        try {
            // Set endpoint in backend
            await invoke("plugin:melo_tts|set_melo_endpoint", {
                endpoint: this.#endpoint,
            });

            // Check availability
            const isAvailable = await invoke<boolean>("plugin:melo_tts|check_melo_availability");
            if (!isAvailable) {
                this.#receiver.onStop(
                    `MeloTTS not available at ${this.#endpoint}. Start with: docker run -p 8888:8888 myshell-ai/melotts`
                );
                return;
            }

            this.#audioContext = new AudioContext();
            this.#isRunning = true;
            this.#receiver.onStart();
            console.log("[MeloTTS] Service started");
        } catch (error) {
            console.error("[MeloTTS] Failed to start:", error);
            this.#receiver.onStop(String(error));
        }
    }

    stop(): void {
        this.#isRunning = false;
        if (this.#audioContext) {
            this.#audioContext.close();
            this.#audioContext = null;
        }
        console.log("[MeloTTS] Service stopped");
    }

    async play(text: string): Promise<void> {
        if (!this.#isRunning || !text.trim()) return;

        try {
            // Get audio bytes from backend (WAV format)
            const audioBytes = await invoke<number[]>("plugin:melo_tts|melo_speak", {
                text,
                speakerId: this.#speakerId || null,
                speed: this.#speed,
            });

            if (!audioBytes || audioBytes.length === 0) return;

            // Convert to Uint8Array and play
            const uint8Array = new Uint8Array(audioBytes);
            await this.playAudio(uint8Array);
        } catch (error) {
            console.error("[MeloTTS] Speech synthesis error:", error);
        }
    }

    private async playAudio(audioData: Uint8Array): Promise<void> {
        if (!this.#audioContext) {
            this.#audioContext = new AudioContext();
        }

        try {
            const audioBuffer = await this.#audioContext.decodeAudioData(audioData.buffer.slice(0));
            const source = this.#audioContext.createBufferSource();
            source.buffer = audioBuffer;
            source.connect(this.#audioContext.destination);
            source.start(0);
        } catch (error) {
            console.error("[MeloTTS] Audio playback error:", error);
        }
    }

    dispose(): void {
        this.stop();
    }

    // Static helper to get available speakers
    static async getSpeakers(endpoint: string = "http://localhost:8888"): Promise<MeloSpeaker[]> {
        try {
            await invoke("plugin:melo_tts|set_melo_endpoint", { endpoint });
            return await invoke<MeloSpeaker[]>("plugin:melo_tts|get_melo_speakers");
        } catch {
            return [];
        }
    }
}
