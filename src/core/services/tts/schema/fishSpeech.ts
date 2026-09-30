import { zSafe, zStringNumber } from "@/utils";
import { z } from "zod";

/**
 * Fish Speech (self-hosted `api_server.py` or Fish Audio's hosted API) settings. Speech streams into the TTS sink
 * through `fish_speak_to_sink` (P2-I).
 */
export const TTS_FishSpeech_Schema = z.object({
  endpoint: zSafe(z.coerce.string(), "http://localhost:8080"),
  referenceId: zSafe(z.coerce.string(), ""),
  /** Optional Bearer key: Fish Audio's hosted API, or a self-hosted `api_server.py --api-key`. */
  apiKey: zSafe(z.coerce.string(), ""),
  /** Optional `model` header, e.g. `s2.1-pro-free` on the hosted API; blank = server default. */
  model: zSafe(z.coerce.string(), ""),
  /** Output device (cpal name); "" = the system default output. */
  device: zSafe(z.coerce.string(), ""),
  /** Sink gain 0..2 (P2-I). */
  volume: zSafe(zStringNumber(), "1.0"),
}).default({});
