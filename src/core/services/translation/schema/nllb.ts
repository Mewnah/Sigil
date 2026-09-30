import { zSafe } from "@/utils";
import { z } from "zod";

/** Catalog ids of the NLLB models (Rust `mt_local::NLLB_DEFAULT_MODEL` / `NLLB_LARGE_MODEL`, P4-D's `mt-nllb.json`). */
export const NLLB_MODELS = ["mt/nllb-200-distilled-600m-int8", "mt/nllb-200-distilled-1.3b-int8"] as const;

/** NLLB-200 distilled on Sigil's ORT (`plugin:mt-local|nllb_*`). The language pair is the shared one. */
export const Translation_Nllb_Schema = z.object({
  /** Catalog id (one of `NLLB_MODELS`). */
  model: zSafe(z.coerce.string(), NLLB_MODELS[0]),
  /** CPU background profile: 2 intra-op threads. */
  threads: zSafe(z.coerce.number().int().min(1).max(4), 2),
  /** 1 = greedy. */
  beam: zSafe(z.coerce.number().int().min(1).max(4), 1),
  /** Off by default: CPU cost while a game runs. */
  interim: zSafe(z.coerce.boolean(), false),
}).default({});
