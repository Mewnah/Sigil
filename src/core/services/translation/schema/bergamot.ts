import { zSafe } from "@/utils";
import { z } from "zod";

/** Mozilla Bergamot CPU helper (`plugin:mt-local|bergamot_*`). The language pair is the shared one. */
export const Translation_Bergamot_Schema = z.object({
  /** Helper worker threads. */
  threads: zSafe(z.coerce.number().int().min(1).max(4), 1),
  /** Cheap enough to re-translate partials. */
  interim: zSafe(z.coerce.boolean(), true),
}).default({});
