import { zSafe } from "@/utils";
import { z } from "zod";

/** LibreTranslate-compatible server (Rust `translate` plugin; also LTEngine). Legacy per-slice language pair. */
export const Translation_LibreTranslate_Schema = z.object({
  endpoint: zSafe(z.coerce.string(), "http://localhost:5000"),
  apiKey: zSafe(z.coerce.string(), ""),
  languageFrom: zSafe(z.coerce.string(), "en"),
  language: zSafe(z.coerce.string(), "es"),
  autoDetect: zSafe(z.coerce.boolean(), false),
  interim: zSafe(z.coerce.boolean(), true),
}).default({});
