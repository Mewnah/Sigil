import { invoke } from "@tauri-apps/api/core";
import { ITTSReceiver, ITTSService } from "../types";
import { TTS_State } from "../schema";
import { devLog } from "@/utils/devLog";

export class TTS_NeuralOnnxService implements ITTSService {
  #receiver: ITTSReceiver;
  #running = false;
  #audioContext: AudioContext | null = null;

  constructor(receiver: ITTSReceiver) {
    this.#receiver = receiver;
  }

  start(data: TTS_State): void {
    const c = data.neuralOnnx;
    void (async () => {
      try {
        await invoke("plugin:neural-onnx-tts|neural_onnx_tts_set_model_source", {
          source: c.modelSource,
          localDir:
            c.modelSource === "local" && c.localModelDir.trim()
              ? c.localModelDir.trim()
              : null,
        });
        await invoke("plugin:neural-onnx-tts|neural_onnx_tts_set_local_package_format", {
          format: c.localPackageFormat,
        });
        if (c.modelSource === "hub") {
          await invoke("plugin:neural-onnx-tts|neural_onnx_tts_set_hub_id", { hubId: c.hubId });
        }
        await invoke("plugin:neural-onnx-tts|neural_onnx_tts_set_voice_speed", {
          voice: c.voice,
          speed: parseFloat(c.speed) || 1,
        });
        await invoke("plugin:neural-onnx-tts|neural_onnx_tts_load_model", {
          executionProvider: c.executionProvider,
        });
        this.#audioContext = new AudioContext();
        this.#running = true;
        this.#receiver.onStart();
        devLog("[Local ONNX speech] started");
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
    devLog("[Local ONNX speech] stopped");
  }

  async play(text: string): Promise<void> {
    if (!this.#running || !text.trim()) return;
    const data = window.ApiServer.state.services.tts.data.neuralOnnx;
    try {
      if (data.routeThroughVoiceChanger && window.ApiServer.voiceChanger.state.isRunning) {
        await invoke("plugin:neural-onnx-tts|neural_onnx_tts_speak_into_voice_changer", { text });
        return;
      }

      const bytes = await invoke<number[]>("plugin:neural-onnx-tts|neural_onnx_tts_speak", { text });
      if (!bytes?.length) return;

      const u8 = new Uint8Array(bytes);
      if (!this.#audioContext) this.#audioContext = new AudioContext();
      const buf = await this.#audioContext.decodeAudioData(u8.buffer.slice(0) as ArrayBuffer);
      const src = this.#audioContext.createBufferSource();
      src.buffer = buf;
      src.connect(this.#audioContext.destination);
      src.start(0);
    } catch (e) {
      console.error("[NeuralOnnx TTS] play", e);
    }
  }

  dispose(): void {
    this.stop();
  }
}
