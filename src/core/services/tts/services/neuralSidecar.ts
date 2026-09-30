import { invoke } from "@tauri-apps/api/core";
import type { RustTtsSpeakReport } from "@/types";
import { ITTSReceiver } from "../types";
import { TTS_State } from "../schema";
import { devLog } from "@/utils/devLog";
import { HttpSinkBackend } from "./openaiSpeech";

/**
 * User-run HTTP speech server (`GET /health`, `POST /speech`) through the `neural-sidecar-tts` plugin. Each sentence
 * streams into the TTS sink (`neural_sidecar_tts_speak_to_sink`, which asks the server for a streamed WAV); the
 * top-level route decides whether it plays on the device, into the voice changer, or both.
 */
export class TTS_NeuralSidecarService extends HttpSinkBackend {
  #endpoint = "";
  #voice = "default";

  constructor(receiver: ITTSReceiver) {
    super(receiver, "HTTP speech server");
  }

  protected async connect(data: TTS_State): Promise<string | null> {
    const c = data.neuralSidecar;
    this.#endpoint = c.endpoint.trim();
    this.#voice = c.voice.trim() || "default";
    await invoke("plugin:neural-sidecar-tts|set_neural_sidecar_tts_endpoint", {
      endpoint: this.#endpoint,
    });
    const ok = await invoke<boolean>("plugin:neural-sidecar-tts|check_neural_sidecar_tts", {
      endpoint: this.#endpoint,
    });
    if (!ok) {
      return `Neural sidecar TTS not reachable at ${this.#endpoint}. Run scripts/neural-sidecar/tts_sidecar.py or your server.`;
    }
    devLog("[HTTP speech server] started");
    return null;
  }

  protected speak(text: string, utteranceId: string) {
    return invoke<RustTtsSpeakReport>("plugin:neural-sidecar-tts|neural_sidecar_tts_speak_to_sink", {
      text,
      voice: this.#voice,
      endpoint: this.#endpoint,
      utteranceId,
    });
  }
}
