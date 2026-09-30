import { zSafe, zStringNumber } from "@/utils";
import { z } from "zod";

/** MeloTTS server settings. Speech streams into the TTS sink through `melo_speak_to_sink` (P2-I). */
export const TTS_Melo_Schema = z.object({
  endpoint: zSafe(z.coerce.string(), "http://localhost:8888"),
  speaker: zSafe(z.coerce.string(), ""),
  speed: zSafe(zStringNumber(), "1.0"),
  /** Output device (cpal name); "" = the system default output. */
  device: zSafe(z.coerce.string(), ""),
  /** Sink gain 0..2 (P2-I). */
  volume: zSafe(zStringNumber(), "1.0"),
}).default({});
