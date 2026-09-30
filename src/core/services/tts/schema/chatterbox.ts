import { zSafe, zStringNumber } from "@/utils";
import { z } from "zod";

/**
 * Chatterbox server settings (legacy backend). Speech goes through the OpenAI-compatible streaming client (P2-I,
 * `services/chatterbox.ts`): `{endpoint}/v1`, model `chatterbox`, WAV, `exaggeration` as an extra request member.
 */
export const TTS_Chatterbox_Schema = z.object({
  endpoint: zSafe(z.coerce.string(), "http://localhost:5555"),
  voice: zSafe(z.coerce.string(), "default"),
  speed: zSafe(zStringNumber(), "1.0"),
  exaggeration: zSafe(zStringNumber(), "0.5"),
  /** Output device (cpal name); "" = the system default output. */
  device: zSafe(z.coerce.string(), ""),
  /** Sink gain 0..2 (P2-I). */
  volume: zSafe(zStringNumber(), "1.0"),
}).default({});
