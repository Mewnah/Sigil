import { zSafe, zStringNumber } from "@/utils";
import { z } from "zod";

/** Where the `sbv2` engine runs: the processor (default), DirectML or CUDA (`ort_session`'s EP ladder). */
export const SBV2_DEVICES = ["cpu", "dml", "cuda"] as const;
export type Sbv2Device = (typeof SBV2_DEVICES)[number];

/** Processor threads: automatic, or 1-8. */
export const SBV2_THREADS = ["auto", "1", "2", "3", "4", "6", "8"] as const;

/**
 * Settings of the `sbv2` TTS backend (P6-G): Style-Bert-VITS2 JP-Extra and AivisSpeech `.aivmx` voices on Sigil's own
 * ONNX Runtime (`plugin:tts-helpers|tts_helper_start { engine: "sbv2" }`, voices imported with `sbv2_import_model`).
 * `device` / `volume` are the TTS sink target (the route is the top-level `route`). Persisted values are never renamed
 * or removed.
 */
export const TTS_Sbv2_Schema = z.object({
  /** Imported model id (`plugin:tts-helpers|sbv2_list_models`); "" = none chosen. */
  model: zSafe(z.coerce.string(), ""),
  /** Speaker name ("" = the model's first speaker). */
  speaker: zSafe(z.coerce.string(), ""),
  /** Style name ("Neutral" or unknown = the neutral style). */
  style: zSafe(z.coerce.string(), "Neutral"),
  /** Style strength (Style-Bert-VITS2's style weight; 1 = as trained, 0-10). */
  styleWeight: zSafe(zStringNumber(), "1.0"),
  /** Speaking rate (0.5-2). */
  speed: zSafe(zStringNumber(), "1.0"),
  /** Tempo variation (SDP ratio, 0-1; Style-Bert-VITS2's default 0.2). */
  sdpRatio: zSafe(zStringNumber(), "0.2"),
  /** Noise (0-1.5; default 0.6). */
  noise: zSafe(zStringNumber(), "0.6"),
  /** Phoneme-length noise (0-1.5; default 0.8). */
  noiseW: zSafe(zStringNumber(), "0.8"),
  /** Execution provider: the processor by default. */
  executionProvider: zSafe(z.enum(SBV2_DEVICES), "cpu"),
  /** Processor threads ("auto" = half the physical cores within 2-4). */
  threads: zSafe(z.enum(SBV2_THREADS), "auto"),
  /** Output device (cpal name); "" = the system default output. */
  device: zSafe(z.coerce.string(), ""),
  /** Sink gain 0..2. */
  volume: zSafe(zStringNumber(), "1.0"),
}).default({});

export type TTS_Sbv2_State = z.infer<typeof TTS_Sbv2_Schema>;
