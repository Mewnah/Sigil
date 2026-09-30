import { invoke } from "@tauri-apps/api/core";
import type { RustTtsSpeakReport } from "@/types";
import { ITTSReceiver } from "../types";
import { TTS_State } from "../schema";
import { devLog } from "@/utils/devLog";
import { positiveOr } from "../num";
import { HttpSinkBackend } from "./openaiSpeech";

interface MeloSpeaker {
    id: string;
    name: string;
    language: string;
}

/**
 * MeloTTS HTTP API through the `melo-tts` plugin: each sentence's WAV streams into the TTS sink
 * (`melo_speak_to_sink`), so the output device, the route and stop / skip apply.
 */
export class TTS_MeloService extends HttpSinkBackend {
    #endpoint = "http://localhost:8888";
    #speakerId = "";
    #speed = 1.0;

    constructor(receiver: ITTSReceiver) {
        super(receiver, "MeloTTS");
    }

    protected async connect(data: TTS_State): Promise<string | null> {
        const config = data.melo;
        this.#endpoint = config.endpoint.trim();
        this.#speakerId = config.speaker.trim();
        this.#speed = positiveOr(config.speed, 1);

        await invoke("plugin:melo-tts|set_melo_endpoint", { endpoint: this.#endpoint });
        const isAvailable = await invoke<boolean>("plugin:melo-tts|check_melo_availability", {
            endpoint: this.#endpoint,
        });
        if (!isAvailable) {
            return `MeloTTS not available at ${this.#endpoint}. Start with: docker run -p 8888:8888 myshell-ai/melotts`;
        }
        devLog("[MeloTTS] Service started");
        return null;
    }

    protected speak(text: string, utteranceId: string) {
        return invoke<RustTtsSpeakReport>("plugin:melo-tts|melo_speak_to_sink", {
            text,
            speakerId: this.#speakerId || null,
            speed: this.#speed,
            endpoint: this.#endpoint,
            utteranceId,
        });
    }

    /** Speakers on the server at `endpoint`. Rejects when the server cannot be reached. */
    static async getSpeakers(endpoint: string = "http://localhost:8888"): Promise<MeloSpeaker[]> {
        return await invoke<MeloSpeaker[]>("plugin:melo-tts|get_melo_speakers", { endpoint: endpoint.trim() });
    }
}
