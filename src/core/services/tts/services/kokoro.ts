import { invoke } from "@tauri-apps/api/core";
import { ITTSReceiver } from "../types";
import { TTS_State } from "../schema";
import { devLog } from "@/utils/devLog";
import { OpenAiSpeechBackend, OpenAiSpeechSettings, kokoroSpeechSettings } from "./openaiSpeech";

/**
 * Kokoro-FastAPI (backend `kokoro`, legacy settings kept): speech goes through the OpenAI-compatible streaming client
 * (`{endpoint}/v1`, model `kokoro`, raw 24 kHz PCM streamed into the TTS sink, see {@link kokoroSpeechSettings}); the
 * server check and the voice list stay on the `kokoro-tts` plugin.
 */
export class TTS_KokoroService extends OpenAiSpeechBackend {
    #endpoint = "http://localhost:8880";

    constructor(receiver: ITTSReceiver) {
        super(receiver, "Kokoro");
    }

    protected settings(state: TTS_State): OpenAiSpeechSettings | string {
        this.#endpoint = state.kokoro.endpoint.trim();
        return kokoroSpeechSettings(state.kokoro);
    }

    protected async check(): Promise<string | null> {
        // Keep the plugin's stored default in sync; the voice list also passes the endpoint per call.
        await invoke("plugin:kokoro-tts|set_kokoro_endpoint", { endpoint: this.#endpoint });
        const isAvailable = await invoke<boolean>("plugin:kokoro-tts|check_kokoro_availability", {
            endpoint: this.#endpoint,
        });
        if (!isAvailable) {
            return `Kokoro TTS not available at ${this.#endpoint}. Start with: docker run -p 8880:8880 ghcr.io/remsky/kokoro-fastapi-cpu`;
        }
        devLog("[Kokoro] TTS service started");
        return null;
    }

    /** Voices on the server at `endpoint`. Rejects when the server cannot be reached. */
    static async getVoices(endpoint: string = "http://localhost:8880"): Promise<{ name: string }[]> {
        return await invoke<{ name: string }[]>("plugin:kokoro-tts|get_kokoro_voices", { endpoint: endpoint.trim() });
    }
}
