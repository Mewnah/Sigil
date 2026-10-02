/**
 * Reverse translation of other players (P6-A): a second translation pipeline instance (P4-C's
 * `createTranslationPipeline`) that translates what other players say *into* the user's language and publishes on
 * `others.translation`. It shares the Translation tab's engine, glossary, context and partial settings; only the
 * language pair is its own (`pair` override), so engines validate, warm up and route for the reversed pair.
 */
import { canonicalLanguage } from "@/core/services/translation/languages";

export interface OthersPairInput {
  /** The other players' language setting (`auto` or a code). */
  othersLanguage: string | null | undefined;
  /** The Others `translateTo` setting ("" = the user's own language). */
  translateTo: string | null | undefined;
  /** The main translation's source language (the user's language; "" = taken from STT). */
  mainSource: string | null | undefined;
  /** The user's STT language setting (`auto` or a code), when known. */
  sttLanguage: string | null | undefined;
  /** The app's UI language (`en`, `ja`, `zh-CN`, ...). */
  appLanguage: string | null | undefined;
}

/**
 * The reverse pair: `from` = the other players' language ("" = detect per line: the STT event's `lang`, else the
 * engine guesses), `to` = `translateTo`, else the user's own language as the main translation knows it (its source),
 * else the user's STT language, else the app language, else English.
 *
 * `from` is always a string (never undefined): an undefined side would make the pipeline use the main translation's
 * `sourceLanguage`, which is the user's language, the wrong way round.
 */
export function othersTranslationPair(input: OthersPairInput): { from: string; to: string } {
  const from = canonicalLanguage(input.othersLanguage);
  const to =
    canonicalLanguage(input.translateTo) ||
    canonicalLanguage(input.mainSource) ||
    canonicalLanguage(input.sttLanguage) ||
    canonicalLanguage(input.appLanguage) ||
    "en";
  return { from, to };
}
