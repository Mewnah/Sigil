import { zSafe } from "@/utils";
import { z } from "zod";
import { keepObjects, TRANSLATION_ENGINE_IDS, Translation_Backends, TranslationEngineId } from "./backends";

const zEngineId = z.enum(TRANSLATION_ENGINE_IDS);

/** `interimEngine` values. */
export const TRANSLATION_INTERIM_ENGINES = ["auto", ...TRANSLATION_ENGINE_IDS] as const;
export type TranslationInterimEngine = (typeof TRANSLATION_INTERIM_ENGINES)[number];

/** Default chain: quality first (the local LLM), then the zero-VRAM CPU fallback, then the long tail. */
export const TRANSLATION_DEFAULT_CHAIN: readonly TranslationEngineId[] = [
  Translation_Backends.llm,
  Translation_Backends.bergamot,
  Translation_Backends.nllb,
];

/** Known engine ids in order, without duplicates (unknown or repeated entries are dropped, not the whole list). */
export function cleanChain(value: unknown): unknown {
  if (!Array.isArray(value)) return value;
  const known = new Set<string>(TRANSLATION_ENGINE_IDS);
  const out: string[] = [];
  for (const v of value) if (typeof v === "string" && known.has(v) && !out.includes(v)) out.push(v);
  return out;
}

const zChain = z.preprocess(cleanChain, z.array(zEngineId).min(1).max(TRANSLATION_ENGINE_IDS.length));

/** A chain for one language pair (canonical BCP-47; `from: ""` = any source). */
export const Translation_PairOverride_Schema = z.object({
  from: zSafe(z.coerce.string(), ""),
  to: zSafe(z.coerce.string(), ""),
  chain: zSafe(zChain, () => [...TRANSLATION_DEFAULT_CHAIN]),
});

/**
 * Router (`auto` backend, P4-C `router.ts`): tries the chain in order per line, skipping engines that are not
 * configured, not installed, do not support the pair, or whose circuit breaker is open; a failed or timed-out
 * attempt falls through to the next engine within `attemptTimeoutMs`. The language pair is the shared one.
 */
export const Translation_Router_Schema = z.object({
  chain: zSafe(zChain, () => [...TRANSLATION_DEFAULT_CHAIN]),
  /** Per-pair chains; the first match wins, else `chain`. */
  pairOverrides: zSafe(z.preprocess(keepObjects, z.array(Translation_PairOverride_Schema).max(50)), () => []),
  /** Engine for interims: `auto` = the cheapest ready local engine for the pair (Bergamot when installed), else the chain. */
  interimEngine: zSafe(z.enum(TRANSLATION_INTERIM_ENGINES), "auto"),
  /**
   * Budget of one engine attempt before falling through, for caption-length text (a long final gets proportionally
   * more, `routerAttemptBudgetMs`); also the longest wait for a member that is still starting.
   */
  attemptTimeoutMs: zSafe(z.coerce.number().int().min(500).max(60_000), 6000),
  /** When a cloud engine fails, also try local engines that are not in the chain. */
  cloudFallback: zSafe(z.coerce.boolean(), true),
  interim: zSafe(z.coerce.boolean(), true),
}).default({});
