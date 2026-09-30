import { zSafe } from "@/utils";
import { z } from "zod";
import { normalizeDeepgramLanguage } from "../stt_data";

/** Deepgram settings. */
export const STT_Deepgram_Schema = z.object({
  device: zSafe(z.coerce.string(), "default"),
  language_group: zSafe(z.coerce.string(), "English"),
  /** Deepgram `language=`; legacy `zn-CN` / `zn-TW` (typo) load as `zh-CN` / `zh-TW`. */
  language: zSafe(z.coerce.string().transform(normalizeDeepgramLanguage), "en-US"),
  /** Legacy (pre-Nova) quality tier; unused once `model` is set. */
  tier: zSafe(z.coerce.string(), ""),
  /** Deepgram model id sent as `model=` (e.g. nova-3, nova-2). */
  model: zSafe(z.coerce.string(), "nova-3"),
  key: zSafe(z.coerce.string(), ""),
  punctuate: zSafe(z.coerce.boolean(), true),
  profanity: zSafe(z.coerce.boolean(), true),
  interim: zSafe(z.coerce.boolean(), true),
}).default({});
