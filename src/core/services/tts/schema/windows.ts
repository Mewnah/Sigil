import { zSafe, zStringNumber } from "@/utils";
import { z } from "zod";

/** Windows SAPI (`windows-tts` plugin) settings. */
export const TTS_Windows_Schema = z.object({
  device: zSafe(z.coerce.string(), ""),
  voice: zSafe(z.coerce.string(), ""),
  volume: zSafe(zStringNumber(), "1"),
  rate: zSafe(zStringNumber(), "1"),
}).default({});
