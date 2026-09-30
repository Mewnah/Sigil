import { zSafe, zStringNumber } from "@/utils";
import { z } from "zod";

/** `model` value that means "the folder in `folder`" instead of a catalog id. */
export const LOCAL_NEURAL_FOLDER = "folder";

/** Inference threads: `auto` = half the physical cores within 2-4 (Rust `local_tts::models::effective_threads`). */
export const LOCAL_NEURAL_THREADS = ["auto", "2", "3", "4"] as const;

/**
 * Settings of the `localNeural` backend: in-process Kokoro / Kokoro v1.1-zh / Piper / Kitten through the Rust
 * `local-tts` plugin (`RustLocalTtsRequest` in src/types.ts). Owned by P2-H.
 *
 * - `model`: a catalog id (`tts/...`, see `src-tauri/model-catalog/tts-local.json`), "" = Kokoro v1.0 (the default
 *   local voice), or {@link LOCAL_NEURAL_FOLDER} to use `folder` (an official sherpa-onnx Kokoro / Kitten / Piper
 *   package, or a `speech_package_v1` folder).
 * - `voice`: a voice id of the model ("" = the model's default, e.g. Kokoro's `af_heart`).
 * - `device` / `volume`: the TTS sink target of this backend (the route is the top-level `route`).
 */
export const TTS_LocalNeural_Schema = z.object({
  model: zSafe(z.coerce.string(), ""),
  voice: zSafe(z.coerce.string(), ""),
  speed: zSafe(zStringNumber(), "1.0"),
  /** Output device (cpal name); "" = the system default output. */
  device: zSafe(z.coerce.string(), ""),
  /** Sink gain 0..2. */
  volume: zSafe(zStringNumber(), "1.0"),
  /** Absolute folder of a local voice package (used when `model` is {@link LOCAL_NEURAL_FOLDER}). */
  folder: zSafe(z.coerce.string(), ""),
  threads: zSafe(z.enum(LOCAL_NEURAL_THREADS), "auto"),
}).default({});
