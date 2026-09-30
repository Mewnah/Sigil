import { zSafe, zStringNumber } from "@/utils";
import { z } from "zod";

/**
 * Kokoro-FastAPI server settings (legacy backend). Speech goes through the OpenAI-compatible streaming client
 * (P2-I, `services/kokoro.ts`): `{endpoint}/v1`, model `kokoro`, 24 kHz PCM into the TTS sink.
 */
export const TTS_Kokoro_Schema = z.object({
  endpoint: zSafe(z.coerce.string(), "http://localhost:8880"),
  voice: zSafe(z.coerce.string(), "af_bella"),
  speed: zSafe(zStringNumber(), "1.0"),
  /** Output device (cpal name); "" = the system default output. */
  device: zSafe(z.coerce.string(), ""),
  /** Sink gain 0..2 (P2-I). */
  volume: zSafe(zStringNumber(), "1.0"),
}).default({});
