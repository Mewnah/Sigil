import { zSafe, zStringNumber } from "@/utils";
import { z } from "zod";

/** Where the model runs (`auto` = CUDA on NVIDIA cards, DirectML on other cards with 2 GB or more, else the CPU). */
export const CHATTERBOX_LOCAL_DEVICES = ["auto", "cuda", "dml", "cpu"] as const;
/** CPU threads (`auto` = four). */
export const CHATTERBOX_LOCAL_THREADS = ["auto", "1", "2", "3", "4", "6", "8"] as const;
/** Decoder steps (`auto` = 10 on a graphics card, 6 on the CPU). */
export const CHATTERBOX_LOCAL_QUALITY = ["auto", "fast", "best"] as const;

/**
 * Settings of the `chatterboxLocal` TTS backend (P6-I): Chatterbox Multilingual V3 on Sigil's ONNX Runtime, the
 * in-process voice-cloning fallback in 23 languages (`plugin:local-tts|chatterbox_local_speak`). Not the
 * `chatterbox` backend (a user-run HTTP server). `device` / `volume` are the TTS sink target (the route is the
 * top-level `route`). Persisted values are never renamed or removed.
 */
export const TTS_ChatterboxLocal_Schema = z.object({
  /** Catalog id (`model-catalog/tts-cloning.json`); "" = the default model. */
  model: zSafe(z.coerce.string(), ""),
  /** Voice-library voice to clone (required: the model has no built-in voice). */
  referenceVoiceId: zSafe(z.coerce.string(), ""),
  /** A language tag the model speaks, or `auto` (the caption's language, else the script, else English). */
  language: zSafe(z.coerce.string(), "auto"),
  /** Emotion exaggeration 0.25..2 (0.5 = neutral). */
  exaggeration: zSafe(zStringNumber(), "0.5"),
  /** Classifier-free guidance weight 0..1 (0 when the voice speaks another language than the text). */
  cfgWeight: zSafe(zStringNumber(), "0.5"),
  /** Unused: the model has no rate control (kept from the P6-S0 scaffold). */
  speed: zSafe(zStringNumber(), "1.0"),
  /** Sampling temperature 0.05..2 (P6-I). */
  temperature: zSafe(zStringNumber(), "0.8"),
  /** Execution provider (P6-I). */
  ep: zSafe(z.enum(CHATTERBOX_LOCAL_DEVICES), "auto"),
  /** CPU threads (P6-I). */
  threads: zSafe(z.enum(CHATTERBOX_LOCAL_THREADS), "auto"),
  /** Decoder quality (P6-I): `fast` = 6 steps, `best` = 10. */
  quality: zSafe(z.enum(CHATTERBOX_LOCAL_QUALITY), "auto"),
  /** Output device (cpal name); "" = the system default output. */
  device: zSafe(z.coerce.string(), ""),
  /** Sink gain 0..2. */
  volume: zSafe(zStringNumber(), "1.0"),
}).default({});
