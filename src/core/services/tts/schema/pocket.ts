import { zSafe, zStringNumber } from "@/utils";
import { z } from "zod";

/** Inference threads (`auto` = the engine's choice, two). */
export const POCKET_THREADS = ["auto", "1", "2", "3", "4"] as const;

/**
 * Settings of the `pocket` TTS backend (P6-I): Kyutai Pocket TTS, voice cloning on the CPU, in process
 * (`plugin:local-tts|pocket_tts_speak`). `device` / `volume` are the TTS sink target (the route is the top-level
 * `route`). Persisted values are never renamed or removed.
 */
export const TTS_Pocket_Schema = z.object({
  /** Catalog id (`model-catalog/tts-cloning.json`); "" = the default model (int8). */
  model: zSafe(z.coerce.string(), ""),
  /** Voice-library voice to clone ("" = the model's built-in voice). */
  referenceVoiceId: zSafe(z.coerce.string(), ""),
  /** Unused: Pocket speaks English (kept from the P6-S0 scaffold). */
  language: zSafe(z.coerce.string(), "auto"),
  /** Unused: Pocket has no rate control (kept from the P6-S0 scaffold). */
  speed: zSafe(zStringNumber(), "1.0"),
  threads: zSafe(z.enum(POCKET_THREADS), "auto"),
  /** Sampling temperature of the voice (P6-I): lower is steadier, higher more varied. */
  temperature: zSafe(zStringNumber(), "0.7"),
  /** Output device (cpal name); "" = the system default output. */
  device: zSafe(z.coerce.string(), ""),
  /** Sink gain 0..2. */
  volume: zSafe(zStringNumber(), "1.0"),
}).default({});
