import { zSafe, zStringNumber } from "@/utils";
import { z } from "zod";

/** Uberduck settings. */
export const TTS_Uberduck_Schema = z.object({
  api_key: zSafe(z.coerce.string(), ""),
  secret_key: zSafe(z.coerce.string(), ""),
  device: zSafe(z.coerce.string(), ""),
  voice: zSafe(z.coerce.string(), ""),
  volume: zSafe(zStringNumber(), "1"),
  rate: zSafe(zStringNumber(), "1"),
}).default({});
