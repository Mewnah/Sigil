import { zSafe, zStringNumber } from "@/utils";
import { z } from "zod";

/**
 * Settings of the `edge` TTS backend (P6-H): Microsoft Edge Read-aloud voices, keyless (`edge-tts` plugin), with the
 * local Kokoro voice as fallback (`pipeline.ts` `TtsFallbackChain`). `device` / `volume` are the TTS sink target (the
 * route is the top-level `route`). Persisted values are never renamed or removed.
 */
export const TTS_Edge_Schema = z.object({
  /** Voice short name (`plugin:edge-tts|edge_tts_list_voices`). */
  voice: zSafe(z.coerce.string(), "en-US-AvaMultilingualNeural"),
  /** 1 = normal (0.5-2). */
  rate: zSafe(zStringNumber(), "1.0"),
  /** Pitch offset in Hz (-50..50). */
  pitchHz: zSafe(zStringNumber(), "0"),
  /** Speak with the local voice when the service fails or stalls (the pipeline's fallback chain). */
  fallbackToLocal: zSafe(z.coerce.boolean(), true),
  /** Seconds without audio before the local voice takes over (1-30; used while the local voice is ready). */
  fallbackAfter: zSafe(zStringNumber(), "3"),
  /** Output device (cpal name); "" = the system default output. */
  device: zSafe(z.coerce.string(), ""),
  /** Sink gain 0..2. */
  volume: zSafe(zStringNumber(), "1.0"),
}).default({});
