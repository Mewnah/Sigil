import { canonicalLanguage } from "./languages";
import { TRANSLATION_DEFAULT_CHAIN, Translation_Backends, Translation_State, TranslationEngineId } from "./schema";
import type { TranslationRequest } from "./types";

/**
 * Built-in routes of the translation router (`auto` backend), by tier. They sit between the user's own
 * `auto.pairOverrides` (always first) and `auto.chain`.
 *
 * Why (MT-Europe, `docs/benchmarks/SPEECH_BENCHMARKS.md`, "European pairs on a processor-only PC"): the default chain
 * puts the bundled Hy-MT2 1.8B first. It is the better translator into Japanese, Korean and Chinese on every tier, and on
 * a graphics card it answers in 50-120 ms for half a core. On the processor it takes 3.7-5 cores for 0.4 s a line, and on
 * four European languages NLLB-200 600M with beam 4 translates better: chrF2 over both directions with English, paired
 * bootstrap, 95 % interval above zero for French (+6.0), Portuguese (+4.8), Dutch (+3.6) and Spanish (+2.8). German
 * (+1.7, interval -0.6 to +3.9) and Italian (+0.8) are ties and stay on the AI model. The price is time: 1.6 s a casual
 * caption line (p95 2.4 s; the AI model 0.4 s), 1.8 cores instead of 3.7, and the same 2.6 GB.
 * So while the bundled translator runs on the processor (`TranslationEngineHost.translatorOnCpu`), those pairs go to NLLB
 * beam 4 first, with the rest of the default chain behind it.
 *
 * Rules, all of them so that nothing changes for a user who chose something:
 * - only the `cpu` tier; a graphics-card translator, a cloud provider or a custom server keeps `auto.chain` (on a
 *   graphics card the AI model costs 0.5 core for 0.1 s, so the same gain would not be worth the processor time);
 * - only while `auto.chain` is the shipped default (a chain you edited is yours), and never over a matching
 *   `pairOverrides` entry, so a rule of your own for the pair turns the route off;
 * - only the directions that were measured, and only when the line's source language is known: the router is given
 *   the pair the pipeline resolved (your Source language setting, else the language the speech engine reported, else a
 *   guess from the text), so with Source language on automatic the speech engine's language decides;
 * - a line that carries a glossary term or the background text keeps the AI model first: NLLB uses neither, and the
 *   user asked for them ({@link LineNeeds});
 * - the rolling context (the earlier lines of the stream, on by default) is read by the AI model only, and the route
 *   does not give it up for it: NLLB translates each line alone. The campaign scored the AI model without context too
 *   (the bench has no document-ordered European set), so the gain above is for a line translated alone; what the
 *   context adds to the AI model on these pairs is not measured. Turn the route off with a pair rule of your own;
 * - the engines are the default chain's, so nothing new is started or downloaded: a PC without the NLLB model falls
 *   through to the AI model as before.
 */

export type TranslationTier = "cpu" | "default";

/** One built-in route. */
export interface TierRoute {
  /** The tier it applies to. */
  readonly tier: TranslationTier;
  /** `from>to` pairs of base language codes (`en>fr`) it covers. */
  readonly directions: readonly string[];
  /** Engines in order. */
  readonly chain: readonly TranslationEngineId[];
  /** Beam width asked of NLLB for finals (`TranslationRequest.beam`). */
  readonly nllbBeam: number;
  /**
   * NLLB's attempt budget for a final is the router's budget times this. Beam 4 takes about 50 ms per source character at
   * the 95th percentile (the budget is 50 ms per character): 1 line in 480 of the campaign overran it, 18 reached 80 %.
   */
  readonly nllbBudgetFactor: number;
}

/** Languages whose pair with English NLLB beam 4 won on the processor (95 % interval above zero); de and it tied. */
export const NLLB_BEAM4_EUROPEAN = ["fr", "pt", "nl", "es"] as const;

function withEnglish(languages: readonly string[]): string[] {
  return languages.flatMap((l) => [`en>${l}`, `${l}>en`]);
}

const B = Translation_Backends;

/** The shipped routes. */
export const TIER_ROUTES: readonly TierRoute[] = [
  {
    tier: "cpu",
    directions: withEnglish(NLLB_BEAM4_EUROPEAN),
    chain: [B.nllb, B.llm, B.bergamot],
    nllbBeam: 4,
    nllbBudgetFactor: 1.5,
  },
];

/** The base language of a code (`pt-BR` → `pt`); "" when unknown. */
function baseLanguage(code: string): string {
  return canonicalLanguage(code).split("-")[0].toLowerCase();
}

/** Whether `chain` is the shipped default (same engines, same order). */
export function isDefaultChain(chain: readonly TranslationEngineId[]): boolean {
  return chain.length === TRANSLATION_DEFAULT_CHAIN.length && chain.every((id, i) => id === TRANSLATION_DEFAULT_CHAIN[i]);
}

/** What a line carries that NLLB (the first engine of a built-in route) does not use. */
export interface LineNeeds {
  /** The line has glossary terms (`TranslationRequest.glossary`). */
  glossary?: boolean;
  /** The context settings have background text (`TranslationRequest.background`). */
  background?: boolean;
}

/** What `request` carries that a built-in route would drop (see {@link LineNeeds}). */
export function lineNeeds(request: Pick<TranslationRequest, "glossary" | "background">): LineNeeds {
  return { glossary: (request.glossary?.length ?? 0) > 0, background: (request.background ?? "").trim() !== "" };
}

/**
 * The built-in route for `from` → `to` on `tier`, or null (the pair has none, the chain is not the default one, or the
 * line carries a glossary term or the background text, which only the AI model uses).
 */
export function tierRouteFor(
  auto: Translation_State["auto"],
  from: string,
  to: string,
  tier: TranslationTier,
  needs: LineNeeds = {},
): TierRoute | null {
  if (tier === "default" || !isDefaultChain(auto.chain)) return null;
  if (needs.glossary || needs.background) return null;
  const source = baseLanguage(from);
  const target = baseLanguage(to);
  if (!source || !target) return null;
  const direction = `${source}>${target}`;
  return TIER_ROUTES.find((r) => r.tier === tier && r.directions.includes(direction)) ?? null;
}
