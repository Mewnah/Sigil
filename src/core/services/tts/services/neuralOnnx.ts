import { invoke } from "@tauri-apps/api/core";
import { ITTSReceiver } from "../types";
import { TTS_State } from "../schema";
import { devLog } from "@/utils/devLog";
import { ClipTtsBackend } from "../clipBackend";
import { positiveOr } from "../num";

/** Local ONNX speech (`neural-onnx-tts` plugin), played in the webview or mixed into the voice changer. */
export class TTS_NeuralOnnxService extends ClipTtsBackend {
  constructor(receiver: ITTSReceiver) {
    super(receiver, "Local ONNX speech");
  }

  protected async connect(data: TTS_State): Promise<string | null> {
    const c = data.neuralOnnx;
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
      speed: positiveOr(c.speed, 1),
    });
    await invoke("plugin:neural-onnx-tts|neural_onnx_tts_load_model", {
      executionProvider: c.executionProvider,
    });
    devLog("[Local ONNX speech] started");
    return null;
  }

  protected async synthesize(text: string) {
    const data = window.ApiServer.state.services.tts.data.neuralOnnx;
    if (data.routeThroughVoiceChanger && window.ApiServer.voiceChanger.state.isRunning) {
      await invoke("plugin:neural-onnx-tts|neural_onnx_tts_speak_into_voice_changer", { text });
      return null;
    }
    return invoke<number[]>("plugin:neural-onnx-tts|neural_onnx_tts_speak", { text });
  }
}
