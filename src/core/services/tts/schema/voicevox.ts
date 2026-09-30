import { zSafe, zStringNumber } from "@/utils";
import { z } from "zod";
import { VOICEVOX_DEFAULT_HOST, VOICEVOX_HOST_PRESETS, migrateVoicevoxSettings } from "../voicevoxPresets";

/** VOICEVOX-compatible engine settings (VOICEVOX, AivisSpeech, custom host). */
export const TTS_Voicevox_Schema = z.preprocess(migrateVoicevoxSettings, z.object({
  /**
   * Engine preset: `voicevox` (:50021), `aivisspeech` (:10101) or `custom` (edit `host`). Settings
   * saved before presets existed get it inferred from `host`.
   */
  hostPreset: zSafe(z.enum(VOICEVOX_HOST_PRESETS), "voicevox"),
  host: zSafe(z.coerce.string(), VOICEVOX_DEFAULT_HOST),
  speaker: zSafe(z.coerce.string(), "0"),
  speedScale: zSafe(zStringNumber(), "1.0"),
  pitchScale: zSafe(zStringNumber(), "0.0"),
  intonationScale: zSafe(zStringNumber(), "1.0"),
  volumeScale: zSafe(zStringNumber(), "1.0"),
  device: zSafe(z.coerce.string(), ""),
})).default({});
