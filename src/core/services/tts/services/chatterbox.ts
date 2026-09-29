import { invoke } from "@tauri-apps/api/core";
import { ITTSReceiver } from "../types";
import { TTS_State } from "../schema";
import { devLog } from "@/utils/devLog";
import { ClipTtsBackend } from "../clipBackend";
import { finiteOr, positiveOr } from "../num";

/** Chatterbox TTS HTTP server through the `chatterbox-tts` plugin. */
export class TTS_ChatterboxService extends ClipTtsBackend {
    #endpoint = "http://localhost:5555";
    #voice = "default";
    #speed = 1.0;
    #exaggeration = 0.5;

    constructor(receiver: ITTSReceiver) {
        super(receiver, "Chatterbox");
    }

    protected async connect(data: TTS_State): Promise<string | null> {
        const config = data.chatterbox;
        this.#endpoint = config.endpoint.trim();
        this.#voice = config.voice;
        this.#speed = positiveOr(config.speed, 1);
        this.#exaggeration = finiteOr(config.exaggeration, 0.5);

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

    protected synthesize(text: string) {
        return invoke<number[]>("plugin:chatterbox-tts|chatterbox_speak", {
            text,
            voice: this.#voice,
            speed: this.#speed,
            exaggeration: this.#exaggeration,
            endpoint: this.#endpoint,
        });
    }

    /** Voices on the server at `endpoint`. Rejects when the server cannot be reached. */
    static async getVoices(endpoint: string = "http://localhost:5555"): Promise<{ id: string; name: string }[]> {
        return await invoke<{ id: string; name: string }[]>("plugin:chatterbox-tts|get_chatterbox_voices", {
            endpoint: endpoint.trim(),
        });
    }
}
