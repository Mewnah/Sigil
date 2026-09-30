import { zSafe } from "@/utils";
import { z } from "zod";

/** Azure Translator (renderer fetch). Legacy per-slice language pair (Azure codes; blank source = auto-detect). */
export const Translation_Azure_Schema = z.object({
  key: zSafe(z.coerce.string(), ""),
  location: zSafe(z.coerce.string(), ""),
  languageFrom: zSafe(z.coerce.string(), "en"),
  language: zSafe(z.coerce.string(), "en"),
  profanity: zSafe(z.enum(["Deleted", "Marked", "NoAction"]), "Marked"),
  interim: zSafe(z.coerce.boolean(), true)
}).default({});
