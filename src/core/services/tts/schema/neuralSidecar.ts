import { zSafe, zStringNumber } from "@/utils";
import { z } from "zod";

/**
 * User-run HTTP speech server (`/speech`, `/health`) settings. Speech streams into the TTS sink through
 * `neural_sidecar_tts_speak_to_sink` (P2-I).
 */
export const TTS_NeuralSidecar_Schema = z.object({
  endpoint: zSafe(z.coerce.string(), "http://127.0.0.1:8766"),
  voice: zSafe(z.coerce.string(), "default"),
  /** Output device (cpal name); "" = the system default output. */
  device: zSafe(z.coerce.string(), ""),
  /**
   * Legacy (kept so saved settings parse): replaced by the top-level `route`, which `migrateTtsRoute` set from this
   * flag once. No longer read.
   */
  routeThroughVoiceChanger: zSafe(z.coerce.boolean(), false),
  /** Sink gain 0..2 (P2-I). */
  volume: zSafe(zStringNumber(), "1.0"),
}).default({});
