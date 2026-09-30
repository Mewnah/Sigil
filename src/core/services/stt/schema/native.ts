import { zSafe } from "@/utils";
import { z } from "zod";

/** Web Speech (`native` / `browser`) settings. */
export const STT_Native_Schema = z.object({
  language_group: zSafe(z.coerce.string(), "English"),
  language: zSafe(z.coerce.string(), "en-US"),
}).default({});
