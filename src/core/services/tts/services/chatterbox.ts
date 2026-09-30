import { invoke } from "@tauri-apps/api/core";
import { ITTSReceiver } from "../types";
import { TTS_State } from "../schema";
import { devLog } from "@/utils/devLog";
import { OpenAiSpeechBackend, OpenAiSpeechSettings, chatterboxSpeechSettings } from "./openaiSpeech";

/**
 * Chatterbox HTTP server (backend `chatterbox`, legacy settings kept): speech goes through the OpenAI-compatible
 * streaming client (`{endpoint}/v1`, model `chatterbox`, WAV, `exaggeration` as an extra member, see
 * {@link chatterboxSpeechSettings}); the server check stays on the `chatterbox-tts` plugin.
 */
export class TTS_ChatterboxService extends OpenAiSpeechBackend {
    #endpoint = "http://localhost:5555";

    constructor(receiver: ITTSReceiver) {
        super(receiver, "Chatterbox");
    }

    protected settings(state: TTS_State): OpenAiSpeechSettings | string {
        this.#endpoint = state.chatterbox.endpoint.trim();
        return chatterboxSpeechSettings(state.chatterbox);
    }

    protected async check(): Promise<string | null> {
        await invoke("plugin:chatterbox-tts|set_chatterbox_endpoint", { endpoint: this.#endpoint });
        const isAvailable = await invoke<boolean>("plugin:chatterbox-tts|check_chatterbox_availability", {
            endpoint: this.#endpoint,
        });
        if (!isAvailable) {
            return `Chatterbox not available at ${this.#endpoint}. Start with: docker compose up -d`;
        }
        devLog("[Chatterbox] Service started");
        return null;
    }

    /** Voices on the server at `endpoint`. Rejects when the server cannot be reached. */
    static async getVoices(endpoint: string = "http://localhost:5555"): Promise<{ id: string; name: string }[]> {
        return await invoke<{ id: string; name: string }[]>("plugin:chatterbox-tts|get_chatterbox_voices", {
            endpoint: endpoint.trim(),
        });
    }
}
