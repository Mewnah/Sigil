import { zSafe, zStringNumber } from "@/utils";
import { z } from "zod";

/** Engines of this backend (`RustTtsHelperEngine` values). Persisted: add ids, never rename them. */
export const GGML_TTS_ENGINES = ["magpie", "breeze", "omnivoice", "s2"] as const;
/**
 * Runtime pack variant: `auto` picks an installed GPU pack the card runs (CUDA on NVIDIA, else Vulkan), the processor
 * without a GPU. Each engine has packs for some of these (Rust refuses the others with a reason).
 */
export const GGML_TTS_VARIANTS = ["auto", "cuda13", "cuda", "vulkan", "cpu"] as const;
/** OmniVoice refinement steps: `high` = 32 (its default), `fast` = 16. */
export const GGML_TTS_QUALITIES = ["high", "fast"] as const;

export type GgmlTtsEngine = (typeof GGML_TTS_ENGINES)[number];
export type GgmlTtsVariant = (typeof GGML_TTS_VARIANTS)[number];
export type GgmlTtsQuality = (typeof GGML_TTS_QUALITIES)[number];

/**
 * Settings of the `ggmlTts` TTS backend (P6-F): the experimental ggml TTS helpers (NVIDIA Magpie through NeMo-Speech.cpp,
 * Breeze TTS 2 through Breeze-TTS-2.cpp, OmniVoice through CrispASR, Fish S2 Pro through s2.cpp), started by the Rust
 * `tts-helpers` plugin (`plugin:tts-helpers|tts_helper_start`) and spoken through the OpenAI-compatible client into the
 * TTS sink. `engine`, `model`, `variant` and the launch switches (`quality`, `lowVram`) decide what the helper loads (a
 * change reloads it before the next sentence); the others apply per sentence. `device` / `volume` are the TTS sink
 * target (the route is the top-level `route`). Persisted values are never renamed or removed.
 */
export const TTS_GgmlTts_Schema = z.object({
  engine: zSafe(z.enum(GGML_TTS_ENGINES), "magpie"),
  /** The engine's model ("" = its default: Magpie `v2602`, Breeze `q8_0`, OmniVoice `q8_0`, S2 `q4_k_m`). */
  model: zSafe(z.coerce.string(), ""),
  /** Speaker / preset voice ("" = default; Magpie: John, Sofia, Aria, Jason or Leo). */
  voice: zSafe(z.coerce.string(), ""),
  /** Voice-library voice for engines that clone ("" = none: Breeze designs a voice, OmniVoice and S2 pick one). */
  referenceVoiceId: zSafe(z.coerce.string(), ""),
  /** Language of the text: `auto` = the caption's language, or a BCP-47 tag (Magpie, OmniVoice). */
  language: zSafe(z.coerce.string(), "auto"),
  /** Unused: none of these servers stretches time without changing the pitch (kept: persisted values stay). */
  speed: zSafe(zStringNumber(), "1.0"),
  /** Runtime pack variant. */
  variant: zSafe(z.enum(GGML_TTS_VARIANTS), "auto"),
  /** Breeze: the voice described or directed in words ("" = its default, "Speak clearly and naturally."). */
  instruction: zSafe(z.coerce.string(), ""),
  /** Breeze: classifier-free guidance, 1-3 (2-3 makes inline events such as `(laugh)` work). */
  cfgScale: zSafe(zStringNumber(), "1.0"),
  /** Sampling seed ("" = a fixed default, so a designed or automatic voice stays the same from sentence to sentence). */
  seed: zSafe(z.coerce.string(), ""),
  /** OmniVoice: refinement steps. */
  quality: zSafe(z.enum(GGML_TTS_QUALITIES), "high"),
  /** Fish S2 Pro: part of the model and the codec on the processor (cards under 10 GB). */
  lowVram: zSafe(z.coerce.boolean(), false),
  /** OmniVoice: the user confirmed they have the consent of the person whose voice is cloned (CrispASR requires it). */
  cloneConsent: zSafe(z.coerce.boolean(), false),
  /** OmniVoice: the user tells their audience the voice is AI, so CrispASR skips its spoken notice before cloned speech. */
  discloseAi: zSafe(z.coerce.boolean(), false),
  /** Output device (cpal name); "" = the system default output. */
  device: zSafe(z.coerce.string(), ""),
  /** Sink gain 0..2. */
  volume: zSafe(zStringNumber(), "1.0"),
}).default({});

export type TTS_GgmlTts_State = z.infer<typeof TTS_GgmlTts_Schema>;
