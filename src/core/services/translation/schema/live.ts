import { zSafe } from "@/utils";
import { z } from "zod";
import { keepObjects } from "./backends";

/*
 * Settings that apply live, to the next line, without restarting translation (the inspector shows them outside
 * `Inspector.Deactivatable`): glossary, rolling context and the interim (partial caption) policy. The pipeline
 * (P4-C) reads them per line.
 */

/** One glossary term; the prompt builder sends only entries whose `source` occurs in the line. */
export const Translation_GlossaryEntry_Schema = z.object({
  source: zSafe(z.coerce.string(), ""),
  target: zSafe(z.coerce.string(), ""),
  /** Canonical BCP-47 source language this entry applies to; "" = any. */
  from: zSafe(z.coerce.string(), ""),
  /** Canonical BCP-47 target language; "" = any. */
  to: zSafe(z.coerce.string(), ""),
  caseSensitive: zSafe(z.coerce.boolean(), false),
  enabled: zSafe(z.coerce.boolean(), true),
});

export const TRANSLATION_GLOSSARY_MAX = 1000;

export const Translation_Glossary_Schema = zSafe(
  z.preprocess(keepObjects, z.array(Translation_GlossaryEntry_Schema).max(TRANSLATION_GLOSSARY_MAX)),
  () => [],
);

/** Rolling context for LLM prompts (Hy-MT2 context block, DeepL `context`). */
export const Translation_Context_Schema = z.object({
  enabled: zSafe(z.coerce.boolean(), true),
  /** Earlier final (source, translation) pairs sent with each line. */
  turns: zSafe(z.coerce.number().int().min(0).max(10), 3),
  /** Free-text background (topic, names, speaking style) sent with every LLM request. */
  background: zSafe(z.coerce.string(), ""),
}).default({});

/** How interim (partial) translations are published. */
export const Translation_Partials_Schema = z.object({
  /** LocalAgreement-2: publish only what two consecutive interim translations agree on (no flicker). */
  localAgreement: zSafe(z.coerce.boolean(), true),
  /** Also publish the not-yet-agreed tail after the committed prefix (`TextEvent.stableLength` marks the split). */
  showTentative: zSafe(z.coerce.boolean(), false),
  /** Minimum gap between interim requests of one line (latest wins). */
  throttleMs: zSafe(z.coerce.number().int().min(0).max(5000), 300),
  /**
   * Smart Turn (`TextEvent.turnEnd`): a final the speaker has not finished (`turnEnd: false`) is held and joined with
   * the next sentence into one translated line, committed when the turn ends. Finals without a verdict (Smart Turn
   * off) are committed at once either way.
   */
  turnEndCommit: zSafe(z.coerce.boolean(), true),
  /** Longest wait for the end of a turn Smart Turn said was not finished (ms, after the last speech event). */
  turnHoldMs: zSafe(z.coerce.number().int().min(500).max(10_000), 2500),
}).default({});
