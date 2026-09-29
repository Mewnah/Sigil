import { invoke } from "@tauri-apps/api/core";
import { ITTSReceiver } from "../types";
import { TTS_State } from "../schema";
import { devLog } from "@/utils/devLog";
import { ClipTtsBackend } from "../clipBackend";
import { positiveOr } from "../num";

/** Kokoro-FastAPI (OpenAI-style `/v1/audio/speech`) through the `kokoro-tts` plugin. */
export class TTS_KokoroService extends ClipTtsBackend {
    #endpoint = "http://localhost:8880";
    #voice = "af_bella";
    #speed = 1.0;

    constructor(receiver: ITTSReceiver) {
        super(receiver, "Kokoro");
    }

    protected async connect(data: TTS_State): Promise<string | null> {
        const config = data.kokoro;
        this.#endpoint = config.endpoint.trim();
        this.#voice = config.voice;
        this.#speed = positiveOr(config.speed, 1);

        // Keep the plugin's stored default in sync; every call below also passes the endpoint.
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

    protected synthesize(text: string) {
        return invoke<number[]>("plugin:kokoro-tts|kokoro_speak", {
            text,
            voice: this.#voice,
            speed: this.#speed,
            endpoint: this.#endpoint,
        });
    }

    /** Voices on the server at `endpoint`. Rejects when the server cannot be reached. */
    static async getVoices(endpoint: string = "http://localhost:8880"): Promise<{ name: string }[]> {
        return await invoke<{ name: string }[]>("plugin:kokoro-tts|get_kokoro_voices", { endpoint: endpoint.trim() });
    }
}
