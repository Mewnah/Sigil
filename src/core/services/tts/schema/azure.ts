import { zSafe, zStringNumber } from "@/utils";
import { z } from "zod";
import { normalizeAzureVoiceName } from "../ssml";

/** Azure Speech synthesis settings. */
export const TTS_Azure_Schema = z.object({
  device: zSafe(z.coerce.string(), ""),
  language: zSafe(z.coerce.string(), "English (United States)"),
  /** Voice short name; a trailing comma from old voice-list data is dropped on load. */
  voice: zSafe(z.coerce.string().transform(normalizeAzureVoiceName), ""),
  voiceStyle: zSafe(z.coerce.string(), ""),
  voiceRole: zSafe(z.coerce.string(), ""),
  voiceVolume: zSafe(z.coerce.string(), "default"),
  voiceRate: zSafe(z.coerce.string(), "default"),
  voicePitch: zSafe(z.coerce.string(), "default"),
  voiceRange: zSafe(z.coerce.string(), "default"),
  volume: zSafe(zStringNumber(), "1"),
  rate: zSafe(zStringNumber(), "1"),
  key: zSafe(z.coerce.string(), ""),
  location: zSafe(z.coerce.string(), ""),
}).default({});
