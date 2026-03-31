import { invoke } from "@tauri-apps/api/core";
import { ITTSReceiver, ITTSService } from "../types";
import { TTS_State } from "../schema";
import { devLog } from "@/utils/devLog";

export class TTS_FishSpeechService implements ITTSService {
    #receiver: ITTSReceiver;
    #endpoint: string = "http://localhost:8080";
    #referenceId: string = "";
    #audioContext: AudioContext | null = null;
    #isRunning: boolean = false;

    constructor(receiver: ITTSReceiver) {
        this.#receiver = receiver;
    }

    async start(data: TTS_State): Promise<void> {
        const config = data.fishSpeech;
        this.#endpoint = config.endpoint;
        this.#referenceId = config.referenceId;

        try {
            await invoke("plugin:fish-speech|set_fish_endpoint", {
                endpoint: this.#endpoint,
            });

            const isAvailable = await invoke<boolean>("plugin:fish-speech|check_fish_availability");
            if (!isAvailable) {
                this.#receiver.onStop(
                    `Fish Speech not available at ${this.#endpoint}. Start with: docker run -p 8080:8080 fishaudio/fish-speech`
                );
                return;
            }

            this.#audioContext = new AudioContext();
            this.#isRunning = true;
            this.#receiver.onStart();
            devLog("[Fish Speech] Service started");
        } catch (error) {
            console.error("[Fish Speech] Failed to start:", error);
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
            const audioBytes = await invoke<number[]>("plugin:fish-speech|fish_speak", {
                text,
                referenceId: this.#referenceId || null,
            });

            if (!audioBytes || audioBytes.length === 0) return;

            const uint8Array = new Uint8Array(audioBytes);
            await this.playAudio(uint8Array);
        } catch (error) {
            console.error("[Fish Speech] Speech error:", error);
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
            console.error("[Fish Speech] Playback error:", error);
        }
    }

    dispose(): void {
        this.stop();
    }

    static async getVoices(endpoint: string = "http://localhost:8080"): Promise<{ id: string; name: string }[]> {
        try {
            await invoke("plugin:fish-speech|set_fish_endpoint", { endpoint });
            return await invoke<{ id: string; name: string }[]>("plugin:fish-speech|get_fish_voices");
        } catch {
            return [];
        }
    }
}
