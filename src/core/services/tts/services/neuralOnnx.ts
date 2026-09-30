import i18n from "i18next";
import { positiveOr } from "../num";
import { TTS_State } from "../schema";
import { ITTSReceiver } from "../types";
import { LOCAL_TTS_HUB_PREFIX, LocalTtsBackend, LocalTtsVoiceSettings, normalizeHubRepo } from "./localNeural";

/**
 * The legacy `neuralOnnx` settings as a local-tts request (P2-H: "neuralOnnx maps onto localNeural"), or an error
 * message to refuse Start with:
 * - Hub mode: `hub:<org>/<name>` (a KittenML 0.8 repo runs as its catalog Kitten package once installed; other repos
 *   download from the Hub as before, see Rust `local_tts::models`).
 * - Local mode: the speech package folder (`speech_package_v1`) or an official sherpa-onnx Kokoro / Kitten / Piper
 *   package folder.
 * The execution-provider hint is not used (the local engines run on the processor, as Kitten always did).
 */
export function neuralOnnxSettings(c: TTS_State["neuralOnnx"]): LocalTtsVoiceSettings | string {
  let model: string;
  if (c.modelSource === "local") {
    model = c.localModelDir.trim();
    if (!model) return i18n.t("tts.local_neural_folder_missing");
  } else {
    const repo = normalizeHubRepo(c.hubId);
    if (!repo) return i18n.t("tts.neural_onnx_hub_invalid", { repo: c.hubId.trim() });
    model = `${LOCAL_TTS_HUB_PREFIX}${repo}`;
  }
  return {
    model,
    // "" = the model's default voice (Jasper for KittenML models).
    voice: c.voice.trim(),
    speed: Math.min(2, Math.max(0.5, positiveOr(c.speed, 1))),
    language: null,
    threads: null,
  };
}

/**
 * Local ONNX speech (backend `neuralOnnx`): Kitten from a Hugging Face repo or a local speech package, now spoken by
 * the in-process `local-tts` engine through the sentence pipeline and the TTS sink (so the output device and the
 * top-level route apply, and stop / skip cut in at once). Settings keep their persisted keys.
 */
export class TTS_NeuralOnnxService extends LocalTtsBackend {
  constructor(receiver: ITTSReceiver) {
    super(receiver, "Local ONNX speech");
  }

  protected settings(state: TTS_State): LocalTtsVoiceSettings | string {
    return neuralOnnxSettings(state.neuralOnnx);
  }
}
