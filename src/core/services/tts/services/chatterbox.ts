import { invoke } from "@tauri-apps/api/core";
import { ITTSReceiver, ITTSService } from "../types";
import { TTS_State } from "../schema";

export class TTS_ChatterboxService implements ITTSService {
    #receiver: ITTSReceiver;
    #endpoint: string = "http://localhost:5555";
    #voice: string = "default";
    #speed: number = 1.0;
    #exaggeration: number = 0.5;
    #audioContext: AudioContext | null = null;
    #isRunning: boolean = false;

    constructor(receiver: ITTSReceiver) {
        this.#receiver = receiver;
    }

    async start(data: TTS_State): Promise<void> {
        const config = data.chatterbox;
        this.#endpoint = config.endpoint;
        this.#voice = config.voice;
        this.#speed = parseFloat(config.speed);
        this.#exaggeration = parseFloat(config.exaggeration);

        try {
            await invoke("plugin:chatterbox-tts|set_chatterbox_endpoint", {
                endpoint: this.#endpoint,
            });

            const isAvailable = await invoke<boolean>("plugin:chatterbox-tts|check_chatterbox_availability");
            if (!isAvailable) {
                this.#receiver.onStop(
                    `Chatterbox not available at ${this.#endpoint}. Start with: docker compose up -d`
                );
                return;
            }

            this.#audioContext = new AudioContext();
            this.#isRunning = true;
            this.#receiver.onStart();
            console.log("[Chatterbox] Service started");
        } catch (error) {
            console.error("[Chatterbox] Failed to start:", error);
            this.#receiver.onStop(String(error));
        }
    }

    stop(): void {
        this.#isRunning = false;
        if (this.#audioContext) {
            this.#audioContext.close();
            this.#audioContext = null;
        }
    }

    async play(text: string): Promise<void> {
        if (!this.#isRunning || !text.trim()) return;

        try {
            const audioBytes = await invoke<number[]>("plugin:chatterbox-tts|chatterbox_speak", {
                text,
                voice: this.#voice,
                speed: this.#speed,
                exaggeration: this.#exaggeration,
            });

            if (!audioBytes || audioBytes.length === 0) return;

            const uint8Array = new Uint8Array(audioBytes);
            await this.playAudio(uint8Array);
        } catch (error) {
            console.error("[Chatterbox] Speech error:", error);
        }
    }

    private async playAudio(audioData: Uint8Array): Promise<void> {
        if (!this.#audioContext) this.#audioContext = new AudioContext();

        try {
            const audioBuffer = await this.#audioContext.decodeAudioData(audioData.buffer.slice(0) as ArrayBuffer);
            const source = this.#audioContext.createBufferSource();
            source.buffer = audioBuffer;
            source.connect(this.#audioContext.destination);
            source.start(0);
        } catch (error) {
            console.error("[Chatterbox] Playback error:", error);
        }
    }

    dispose(): void {
        this.stop();
    }

    static async getVoices(endpoint: string = "http://localhost:5555"): Promise<{ id: string; name: string }[]> {
        try {
            await invoke("plugin:chatterbox-tts|set_chatterbox_endpoint", { endpoint });
            return await invoke<{ id: string; name: string }[]>("plugin:chatterbox-tts|get_chatterbox_voices");
        } catch {
            return [];
        }
    }
}
