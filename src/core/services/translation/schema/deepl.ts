import { zSafe } from "@/utils";
import { z } from "zod";

/** DeepL API Free / Pro (`plugin:translate|deepl_*`). The language pair is the shared one. */
export const Translation_Deepl_Schema = z.object({
  apiKey: zSafe(z.coerce.string(), ""),
  /** `auto` = Free for keys ending in `:fx`, else Pro. */
  plan: zSafe(z.enum(["auto", "free", "pro"]), "auto"),
  formality: zSafe(z.enum(["default", "more", "less", "prefer_more", "prefer_less"]), "default"),
  /** `default` sends nothing. */
  modelType: zSafe(z.enum(["default", "latency_optimized", "quality_optimized", "prefer_quality_optimized"]), "default"),
  /** Send earlier lines as DeepL `context` (not billed). */
  sendContext: zSafe(z.coerce.boolean(), true),
  /** Off by default: every interim counts against the monthly character quota. */
  interim: zSafe(z.coerce.boolean(), false),
}).default({});
