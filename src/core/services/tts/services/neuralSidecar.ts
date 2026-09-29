import { invoke } from "@tauri-apps/api/core";
import { ITTSReceiver } from "../types";
import { TTS_State } from "../schema";
import { devLog } from "@/utils/devLog";
import { ClipTtsBackend } from "../clipBackend";

/** User-run HTTP speech server (`GET /health`, `POST /speech`) through the `neural-sidecar-tts` plugin. */
export class TTS_NeuralSidecarService extends ClipTtsBackend {
  #endpoint = "";

  constructor(receiver: ITTSReceiver) {
    super(receiver, "HTTP speech server");
  }

  protected async connect(data: TTS_State): Promise<string | null> {
    const c = data.neuralSidecar;
    this.#endpoint = c.endpoint.trim();
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

  protected async synthesize(text: string) {
    const data = window.ApiServer.state.services.tts.data.neuralSidecar;
    if (data.routeThroughVoiceChanger && window.ApiServer.voiceChanger.state.isRunning) {
      await invoke("plugin:neural-sidecar-tts|neural_sidecar_tts_speak_into_voice_changer", {
        text,
        voice: data.voice,
        endpoint: this.#endpoint,
      });
      return null;
    }
    return invoke<number[]>("plugin:neural-sidecar-tts|neural_sidecar_tts_speak", {
      text,
      voice: data.voice,
      endpoint: this.#endpoint,
    });
  }
}
