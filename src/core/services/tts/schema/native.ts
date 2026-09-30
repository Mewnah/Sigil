import { zSafe, zStringNumber } from "@/utils";
import { z } from "zod";

/** Web Speech synthesis settings. */
export const TTS_Native_Schema = z.object({
  voice: zSafe(z.coerce.string(), ""),
  pitch: zSafe(zStringNumber(), "1"),
  rate: zSafe(zStringNumber(), "1"),
  volume: zSafe(zStringNumber(), "1"),
}).default({});
