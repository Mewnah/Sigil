import { zSafe, zStringNumber } from "@/utils";
import { z } from "zod";

/** Azure Speech settings. */
export const STT_Azure_Schema = z.object({
  device: zSafe(z.coerce.string(), "default"),
  language_group: zSafe(z.coerce.string(), "English"),
  language: zSafe(z.coerce.string(), "en-US"),
  secondary_language_group: zSafe(z.coerce.string(), ""),
  secondary_language: zSafe(z.coerce.string(), ""),
  use_secondary_language: zSafe(z.coerce.boolean(), true),
  key: zSafe(z.coerce.string(), ""),
  location: zSafe(z.coerce.string(), ""),
  profanity: zSafe(z.coerce.string(), "masked"),
  silenceTimeout: zSafe(zStringNumber(), "20"),
  /** Silence (ms) that ends a phrase; lower = faster finals. Valid 100-5000; 0 = service default. */
  segmentationSilenceMs: zSafe(zStringNumber(), "500"),
  interim: zSafe(z.coerce.boolean(), true),
}).default({});
