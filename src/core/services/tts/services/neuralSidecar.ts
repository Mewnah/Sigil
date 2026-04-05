import { invoke } from "@tauri-apps/api/core";
import { ITTSReceiver, ITTSService } from "../types";
import { TTS_State } from "../schema";
import { devLog } from "@/utils/devLog";

export class TTS_NeuralSidecarService implements ITTSService {
  #receiver: ITTSReceiver;
  #running = false;
  #audioContext: AudioContext | null = null;

  constructor(receiver: ITTSReceiver) {
    this.#receiver = receiver;
  }

  start(data: TTS_State): void {
    const c = data.neuralSidecar;
    void (async () => {
      try {
        await invoke("plugin:neural-sidecar-tts|set_neural_sidecar_tts_endpoint", {
          endpoint: c.endpoint,
        });
        const ok = await invoke<boolean>("plugin:neural-sidecar-tts|check_neural_sidecar_tts");
        if (!ok) {
          this.#receiver.onStop(
            `Neural sidecar TTS not reachable at ${c.endpoint}. Run scripts/neural-sidecar/tts_sidecar.py or your server.`,
          );
          return;
        }
        this.#audioContext = new AudioContext();
        this.#running = true;
        this.#receiver.onStart();
        devLog("[HTTP speech server] started");
      } catch (e) {
        console.error(e);
        this.#receiver.onStop(String(e));
      }
    })();
  }

  stop(): void {
    this.#running = false;
    void this.#audioContext?.close();
    this.#audioContext = null;
  }

  async play(text: string): Promise<void> {
    if (!this.#running || !text.trim()) return;
    const data = window.ApiServer.state.services.tts.data.neuralSidecar;
    try {
      if (data.routeThroughVoiceChanger && window.ApiServer.voiceChanger.state.isRunning) {
        await invoke("plugin:neural-sidecar-tts|neural_sidecar_tts_speak_into_voice_changer", {
          text,
          voice: data.voice,
        });
        return;
      }

      const bytes = await invoke<number[]>("plugin:neural-sidecar-tts|neural_sidecar_tts_speak", {
        text,
        voice: data.voice,
      });
      if (!bytes?.length) return;

      const u8 = new Uint8Array(bytes);
      if (!this.#audioContext) this.#audioContext = new AudioContext();
      const buf = await this.#audioContext.decodeAudioData(u8.buffer.slice(0) as ArrayBuffer);
      const src = this.#audioContext.createBufferSource();
      src.buffer = buf;
      src.connect(this.#audioContext.destination);
      src.start(0);
    } catch (e) {
      console.error("[NeuralSidecar TTS] play", e);
    }
  }

  dispose(): void {
    this.stop();
  }
}
