import { zSafe, zStringNumber } from "@/utils";
import { z } from "zod";

/**
 * Silero VAD settings shared by the engines whose audio Sigil segments in Rust (Whisper, Moonshine, HTTP / Groq).
 * Strings (form inputs); `sttOptions.buildVadConfig` validates them and falls back to the Rust defaults.
 */
export const STT_Vad_Schema = z.object({
  /** Speech-start probability (0.05-0.95); the end threshold is derived (0.15 lower). */
  threshold: zSafe(zStringNumber(), "0.5"),
  /** Silence that closes an utterance (100-5000 ms). */
  minSilenceMs: zSafe(zStringNumber(), "500"),
  /** Audio kept before the detected speech start (0-2000 ms). */
  preRollMs: zSafe(zStringNumber(), "400"),
  /** Shorter speech bursts are ignored (0-5000 ms). */
  minSpeechMs: zSafe(zStringNumber(), "250"),
  /** Longer speech is split here (2000-60000 ms). */
  maxUtteranceMs: zSafe(zStringNumber(), "18000"),
}).default({});
