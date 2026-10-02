import { zSafe, zStringNumber } from "@/utils";
import { z } from "zod";

/**
 * Settings of the `gemini` TTS backend (P6-H): Google Gemini TTS on the user's own key (`gemini-tts` plugin; the key
 * goes in the `x-goog-api-key` header only), with the local Kokoro voice as fallback (`pipeline.ts`
 * `TtsFallbackChain`). `device` / `volume` are the TTS sink target (the route is the top-level `route`). Persisted
 * values are never renamed or removed.
 */
export const TTS_Gemini_Schema = z.object({
  /** The user's Gemini API key (sent only in the `x-goog-api-key` header). "" = not set up. */
  apiKey: zSafe(z.coerce.string(), ""),
  /** Model id (editable: preview names change). */
  model: zSafe(z.coerce.string(), "gemini-3.8-flash-tts"),
  /** Prebuilt voice name. */
  voice: zSafe(z.coerce.string(), "Kore"),
  /** Natural-language style instruction ("" = none). */
  style: zSafe(z.coerce.string(), ""),
  /** Speak with the local voice on 429 / 503 / stalls (the pipeline's fallback chain). */
  fallbackToLocal: zSafe(z.coerce.boolean(), true),
  /** Seconds without audio before the local voice takes over (1-30; used while the local voice is ready). */
  fallbackAfter: zSafe(zStringNumber(), "3"),
  /** API base URL ("" = Google's); for a proxy in front of the Generative Language API. */
  baseUrl: zSafe(z.coerce.string(), ""),
  /** Output device (cpal name); "" = the system default output. */
  device: zSafe(z.coerce.string(), ""),
  /** Sink gain 0..2. */
  volume: zSafe(zStringNumber(), "1.0"),
}).default({});
