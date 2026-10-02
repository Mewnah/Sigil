import { zSafe, zStringNumber } from "@/utils";
import { z } from "zod";

/**
 * Model size: `auto` follows the resource governor (1.7B when it fits, 0.6B while the voice changer runs on the GPU of a
 * card under 16 GB, 0.6B on the processor); `1.7b` / `0.6b` are fixed choices.
 */
export const QWEN3_TTS_MODELS = ["auto", "1.7b", "0.6b"] as const;
/** Runtime pack variant: `auto` = CUDA 13 on NVIDIA when its pack is installed, else Vulkan, the processor without a GPU. */
export const QWEN3_TTS_VARIANTS = ["auto", "vulkan", "cuda13", "cpu"] as const;
/**
 * `preset`: one of the 9 built-in voices, with an optional style / emotion instruction (1.7B); `clone`: a voice from the
 * voice library; `design`: a voice described in words (1.7B).
 */
export const QWEN3_TTS_MODES = ["preset", "clone", "design"] as const;

export type Qwen3TtsModel = (typeof QWEN3_TTS_MODELS)[number];
export type Qwen3TtsVariant = (typeof QWEN3_TTS_VARIANTS)[number];
export type Qwen3TtsMode = (typeof QWEN3_TTS_MODES)[number];

/**
 * Settings of the `qwen3` TTS backend (P6-E): Qwen3-TTS through Sigil's qwentts.cpp server helper
 * (`plugin:tts-helpers|tts_helper_start { engine: "qwen3" }`), spoken through the OpenAI-compatible client into the TTS
 * sink. `model`, `variant` and `mode` decide what the helper loads (a change reloads it before the next sentence); the
 * others, `language` included, apply per sentence (each request names its language). `device` / `volume` are the TTS
 * sink target (the route is the top-level `route`). Persisted values are never renamed or removed.
 */
export const TTS_Qwen3_Schema = z.object({
  model: zSafe(z.enum(QWEN3_TTS_MODELS), "auto"),
  variant: zSafe(z.enum(QWEN3_TTS_VARIANTS), "auto"),
  mode: zSafe(z.enum(QWEN3_TTS_MODES), "preset"),
  /** Built-in voice (`preset`; "" = Ryan). */
  voice: zSafe(z.coerce.string(), ""),
  /** Voice-library voice (`clone`). */
  referenceVoiceId: zSafe(z.coerce.string(), ""),
  /** Style / emotion instruction of a built-in voice (`preset`, 1.7B only). */
  instruction: zSafe(z.coerce.string(), ""),
  /** The voice described in words (`design`). */
  voiceDescription: zSafe(z.coerce.string(), ""),
  /** Language of the text: `auto` = the caption's language (else the model guesses), or a BCP-47 tag. */
  language: zSafe(z.coerce.string(), "auto"),
  /**
   * Sampling seed: "" = a new one per sentence (preset and clone), a fixed design seed in `design` (steadier voice from
   * sentence to sentence); a whole number makes every sentence reproducible.
   */
  seed: zSafe(z.coerce.string(), ""),
  /** Unused: the server has no time stretch (kept: persisted values are never removed). */
  speed: zSafe(zStringNumber(), "1.0"),
  /** Output device (cpal name); "" = the system default output. */
  device: zSafe(z.coerce.string(), ""),
  /** Sink gain 0..2. */
  volume: zSafe(zStringNumber(), "1.0"),
}).default({});

export type TTS_Qwen3_State = z.infer<typeof TTS_Qwen3_Schema>;
