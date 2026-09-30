/**
 * Glossary filtering (research §3.2): only the terms that occur in the current line are sent to the engine (Hy-MT2's
 * terminology block, the generic prompt's glossary, Sakura's term table), so prompts stay short and the model is not
 * distracted by unrelated terms. Owned by P4-C.
 */
import { canonicalLanguage, UNSPACED_SCRIPT_RE } from "./languages";
import type { TranslationGlossaryTerm } from "./types";

/** A stored glossary entry (`Translation_GlossaryEntry`; structural so tests can pass plain objects). */
export interface GlossaryEntryLike {
  source: string;
  target: string;
  from: string;
  to: string;
  caseSensitive: boolean;
  enabled: boolean;
}

/** Most terms sent with one line. */
export const GLOSSARY_MAX_TERMS_PER_LINE = 24;

/** Whether an entry's language (`from` or `to`; "" = any) applies to the line's language (canonical, "" = unknown). */
export function glossaryLanguageMatches(entryLang: string, lineLang: string): boolean {
  const raw = entryLang.trim();
  if (!raw || !lineLang) return true;
  const entry = canonicalLanguage(raw) || raw.toLowerCase();
  return entry === lineLang || entry.toLowerCase() === lineLang.toLowerCase();
}

const WORD_CHAR = /[\p{L}\p{N}\p{M}]/u;
/** Hangul: Korean attaches particles and endings straight to the word (`민지가`, `서울에서`). */
const HANGUL_RE = /\p{Script=Hangul}/u;
/**
 * Languages that attach case endings and suffixes to a word without a space or an apostrophe (Turkish `Sigilde`,
 * Finnish `Sigilissä`): in their text a term only needs a word boundary before it.
 */
const SUFFIXING_LANGUAGES = new Set(["ko", "tr", "az", "fi", "et", "hu", "kk", "uz", "ky", "tk", "tt", "ba", "mn", "eu", "ka", "ta", "te", "kn", "ml"]);

/** A term edge needs a word boundary when its character belongs to a spaced script (not Han, kana, Thai, ...). */
function needsBoundary(ch: string | undefined): boolean {
  return ch !== undefined && WORD_CHAR.test(ch) && !UNSPACED_SCRIPT_RE.test(ch);
}

/**
 * Whether `term` occurs in `text` as a whole word (`cat` is not found in `category`); terms in unspaced scripts
 * (`東京`) match anywhere. A Hangul term, or any term in a line whose language (`lang`, canonical; "" = unknown)
 * attaches suffixes (Korean, Turkish, Finnish, Hungarian, ...), needs a boundary only before it, so `민지` is found
 * in `민지가`. Case-insensitive unless `caseSensitive`; both sides are NFKC-normalized (full-width letters and digits
 * match their ASCII forms).
 */
export function termOccurs(text: string, term: string, caseSensitive = false, lang = ""): boolean {
  const norm = (s: string) => (caseSensitive ? s.normalize("NFKC") : s.normalize("NFKC").toLowerCase());
  const hay = norm(text);
  const needle = norm(term.trim());
  if (!needle) return false;
  const chars = [...needle];
  const last = chars[chars.length - 1];
  const suffixing = SUFFIXING_LANGUAGES.has((canonicalLanguage(lang) || lang).split("-")[0].toLowerCase());
  const startBoundary = needsBoundary(chars[0]);
  const endBoundary = needsBoundary(last) && !HANGUL_RE.test(last) && !suffixing;
  for (let at = hay.indexOf(needle); at >= 0; at = hay.indexOf(needle, at + 1)) {
    const before = at > 0 ? [...hay.slice(0, at)].pop() : undefined;
    const after = [...hay.slice(at + needle.length)][0];
    if (startBoundary && before !== undefined && WORD_CHAR.test(before)) continue;
    if (endBoundary && after !== undefined && WORD_CHAR.test(after)) continue;
    return true;
  }
  return false;
}

/**
 * Glossary terms for one line: enabled entries with a source and a target, whose languages match the pair and
 * whose source occurs in `text`; the user's order is kept, duplicates (same source) are dropped, at most `limit`.
 */
export function filterGlossary(
  entries: readonly GlossaryEntryLike[] | null | undefined,
  text: string,
  from: string,
  to: string,
  limit = GLOSSARY_MAX_TERMS_PER_LINE,
): TranslationGlossaryTerm[] {
  if (!entries?.length || !text.trim() || limit <= 0) return [];
  const out: TranslationGlossaryTerm[] = [];
  const seen = new Set<string>();
  for (const e of entries) {
    if (!e || !e.enabled) continue;
    const source = String(e.source ?? "").trim();
    const target = String(e.target ?? "").trim();
    if (!source || !target) continue;
    if (!glossaryLanguageMatches(String(e.from ?? ""), from) || !glossaryLanguageMatches(String(e.to ?? ""), to)) continue;
    const key = source.normalize("NFKC").toLowerCase();
    if (seen.has(key)) continue;
    if (!termOccurs(text, source, e.caseSensitive, from)) continue;
    seen.add(key);
    out.push({ source, target });
    if (out.length >= limit) break;
  }
  return out;
}
