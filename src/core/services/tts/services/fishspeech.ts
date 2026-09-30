import { invoke } from "@tauri-apps/api/core";
import type { RustTtsSpeakReport } from "@/types";
import { ITTSReceiver } from "../types";
import { TTS_State } from "../schema";
import { devLog } from "@/utils/devLog";
import { HttpSinkBackend } from "./openaiSpeech";

/** Optional per-call credentials; blank values are sent as `null` (not sent to the server). */
function orNull(value: string): string | null {
    const v = value.trim();
    return v ? v : null;
}

/**
 * Fish Speech through the `fish-speech` plugin: a self-hosted fish-speech server, or Fish Audio's
 * hosted API (`https://api.fish.audio`) with an API key and a `model` such as `s2.1-pro-free`. Each
 * sentence's WAV streams into the TTS sink (`fish_speak_to_sink`).
 */
export class TTS_FishSpeechService extends HttpSinkBackend {
    #endpoint = "http://localhost:8080";
    #referenceId = "";
    #apiKey: string | null = null;
    #model: string | null = null;

    constructor(receiver: ITTSReceiver) {
        super(receiver, "Fish Speech");
    }

    protected async connect(data: TTS_State): Promise<string | null> {
        const config = data.fishSpeech;
        this.#endpoint = config.endpoint.trim();
        this.#referenceId = config.referenceId;
        this.#apiKey = orNull(config.apiKey);
        this.#model = orNull(config.model);

        await invoke("plugin:fish-speech|set_fish_endpoint", { endpoint: this.#endpoint });
        // Rejects (instead of returning false) when the server answers but refuses the key.
        const isAvailable = await invoke<boolean>("plugin:fish-speech|check_fish_availability", {
            endpoint: this.#endpoint,
            apiKey: this.#apiKey,
        });
        if (!isAvailable) {
            return `Fish Speech not available at ${this.#endpoint}. Start with: docker run -p 8080:8080 fishaudio/fish-speech`;
        }
        devLog("[Fish Speech] Service started");
        return null;
    }

    protected speak(text: string, utteranceId: string) {
        return invoke<RustTtsSpeakReport>("plugin:fish-speech|fish_speak_to_sink", {
            text,
            referenceId: this.#referenceId || null,
            endpoint: this.#endpoint,
            apiKey: this.#apiKey,
            model: this.#model,
            utteranceId,
        });
    }

    /** Voices from the server at `endpoint` (with `apiKey` for the hosted API). Rejects on failure. */
    static async getVoices(endpoint: string = "http://localhost:8080", apiKey = ""): Promise<{ id: string; name: string }[]> {
        return await invoke<{ id: string; name: string }[]>("plugin:fish-speech|get_fish_voices", {
            endpoint: endpoint.trim(),
            apiKey: orNull(apiKey),
        });
    }
}
