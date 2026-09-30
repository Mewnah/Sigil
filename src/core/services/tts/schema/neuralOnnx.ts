import { zSafe, zStringNumber } from "@/utils";
import { z } from "zod";

/** Local ONNX speech (`neural-onnx-tts` plugin; Kitten via Hub or a local speech package). Maps onto `localNeural` in P2-H. */
export const TTS_NeuralOnnx_Schema = z.object({
  /** `hub` = download from Hugging Face; `local` = folder matching supported package (see docs/OPERATOR_REFERENCE.md#local-speech-package-onnx). */
  modelSource: zSafe(z.enum(["hub", "local"]), "hub"),
  /** On-disk layout when `modelSource` is `local`. Only `speech_package_v1` is implemented today. */
  localPackageFormat: zSafe(z.enum(["speech_package_v1"]), "speech_package_v1"),
  localModelDir: zSafe(z.coerce.string(), ""),
  /** Default Hugging Face repo id for Hub mode (example model). */
  hubId: zSafe(z.coerce.string(), "KittenML/kitten-tts-mini-0.8"),
  voice: zSafe(z.coerce.string(), "Jasper"),
  speed: zSafe(zStringNumber(), "1.0"),
  /** Hint for ORT / GPU (`cpu`, `cuda`, `dml`). Actual EPs depend on the bundled ONNX Runtime. */
  executionProvider: zSafe(z.enum(["cpu", "cuda", "dml"]), "cpu"),
  device: zSafe(z.coerce.string(), ""),
  /** Mix TTS output into the live voice changer input (start VC first). */
  routeThroughVoiceChanger: zSafe(z.coerce.boolean(), false),
}).default({});
