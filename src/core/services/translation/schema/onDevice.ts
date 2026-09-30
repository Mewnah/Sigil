import { zSafe } from "@/utils";
import { z } from "zod";

/** WebView Translator API. Legacy per-slice language pair (BCP-47). */
export const Translation_OnDevice_Schema = z.object({
  languageFrom: zSafe(z.coerce.string(), "en"),
  language: zSafe(z.coerce.string(), "es"),
  autoDetect: zSafe(z.coerce.boolean(), false),
  interim: zSafe(z.coerce.boolean(), true),
}).default({});
