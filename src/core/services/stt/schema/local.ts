import { zSafe } from "@/utils";
import { z } from "zod";

/**
 * Settings of the `local` backend: Sigil's in-process speech pipeline (Rust `speech` plugin) with automatic model
 * routing by language and hardware (P2-E). Sent to Rust as `RustSpeechPipelineConfig` / `RustSpeechRouteRequest`
 * (see src/types.ts) by `services/local.ts`; the Silero VAD settings are the shared `stt.vad` slice.
 */
export const STT_Local_Schema = z.object({
  /** Capture device (cpal name); "" = the default microphone. */
  device: zSafe(z.coerce.string(), ""),
  /** "auto" (spoken language ID) or a language code (normalized by the router: `ja-JP` → `ja`). */
  language: zSafe(z.coerce.string(), "auto"),
  /** Hardware tier for the router: auto ("Best for my PC"), cpu or gpu. */
  tier: zSafe(z.enum(["auto", "cpu", "gpu"]), "auto"),
  /**
   * Re-decode each closed utterance with the final model (replaces the partial under the same segment). The router
   * still runs one pass where the PC cannot afford two (few cores, the voice changer on the CPU).
   */
  twoPass: zSafe(z.coerce.boolean(), true),
  /** Smart Turn end-of-turn hints (commit translation / chatbox / TTS sooner; never shortens captions). */
  smartTurn: zSafe(z.coerce.boolean(), false),
  /** Comma- or newline-separated names and terms to bias recognition towards. */
  hotwords: zSafe(z.coerce.string(), ""),
  /** Whole-utterance hallucination filter and repeat-loop removal on finals (Rust `speech_core::hallucination`). */
  hallucinationFilter: zSafe(z.coerce.boolean(), true),
  /**
   * P6-L A/B (P6-S0): drop utterances the FireRedVAD gate hears as music or singing before any engine sees them
   * (`RustSpeechPipelineConfig.musicGate`). Off by default; needs its model installed.
   */
  musicGate: zSafe(z.coerce.boolean(), false),
}).default({});
