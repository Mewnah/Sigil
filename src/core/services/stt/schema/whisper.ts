import { zSafe, zStringNumber } from "@/utils";
import { z } from "zod";

/** Local Whisper (whisper-rs) settings. */
export const STT_Whisper_Schema = z.object({
  device: zSafe(z.coerce.string(), ""),
  model: zSafe(z.coerce.string(), "base.en"),
  language: zSafe(z.coerce.string(), "en"),
  /** Legacy RMS gate settings (pre-Silero). Kept so old settings parse; no longer sent or shown. */
  vadEnabled: zSafe(z.coerce.boolean(), true),
  silenceThresholdDb: zSafe(zStringNumber(), "-40"),
  silenceDurationMs: zSafe(zStringNumber(), "1500"),
  minChunkDurationMs: zSafe(zStringNumber(), "1000"),
  /** When true, `localModelPath` is copied into app data instead of downloading from Hugging Face. */
  useLocalModel: zSafe(z.coerce.boolean(), false),
  localModelPath: zSafe(z.coerce.string(), ""),
  /** Inference threads; "0" = automatic (about half the cores, at most 4, so games keep their CPU). */
  threads: zSafe(zStringNumber(), "0"),
  /** Decoder prompt: style / vocabulary hint (ignored by presets that do not take prompts). */
  initialPrompt: zSafe(z.coerce.string(), ""),
  /** Comma- or newline-separated names and terms to bias recognition towards. */
  hotwords: zSafe(z.coerce.string(), ""),
  /** Drop an utterance whose no-speech probability is above this (0.05-1; 1 disables). */
  noSpeechThreshold: zSafe(zStringNumber(), "0.6"),
  /** Curated whole-utterance hallucination phrases and repeat-loop removal. */
  hallucinationFilter: zSafe(z.coerce.boolean(), true),
  /** Utterances that may wait for the model before the oldest is skipped (1-10). */
  maxBacklog: zSafe(zStringNumber(), "3"),
}).default({});
