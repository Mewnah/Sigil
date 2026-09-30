/**
 * Sentence splitter for the TTS pipeline (P2-G).
 *
 * Time to first audio depends on synthesizing the first sentence alone, so long finals are cut at sentence
 * boundaries before synthesis. The rules, in order:
 * 1. Hard boundaries: line breaks, and sentence-final punctuation of every script Sigil supports: `. ! ? …`
 *    (followed by whitespace, CJK text or the end), and wherever they stand the full-width / CJK `。 ！ ？ ｡ ．`,
 *    Devanagari `। ॥`, Arabic `؟ ۔`, Armenian `։`, Ethiopic `።`, Myanmar `။`. Closing quotes and brackets right
 *    after the punctuation stay with the sentence (`「はい。」`, `"Yes!"`); a closing corner bracket ends a line of
 *    dialogue when the next one opens (`「はい」「いいえ」`); a period does not split abbreviations (`Dr.`,
 *    `e.g.`, `No.` before a number), initials (`J. K.`; the pronoun `I` only next to another initial) or numbers
 *    (`3.14`).
 * 2. Fragments without letters or digits (`...`, `!?`, `👍`) join the previous sentence, so their prosody is kept
 *    but nothing silent is synthesized; symbols before the first sentence join it. Text of symbols only (an emoji
 *    reply) is one sentence; text of punctuation only has nothing to say.
 * 3. Short fragments are merged with the next one (below `minLength`), so replies like `Hi!` do not become a
 *    separate request with an unnatural pause.
 * 4. Long sentences are capped (`maxLength`, and a lower `firstMaxLength` for the first one, so the first audio
 *    comes quickly): cut after the last clause punctuation (`, ; : 、 ， ； ： —`) in the second half of the
 *    window, else at the last space there, else hard (CJK text without punctuation), never inside a surrogate
 *    pair.
 *
 * Lengths are *weights*, not characters: a CJK / kana / Hangul character counts as 2 (about a syllable), every
 * other character as 1.
 */

export interface SentenceOptions {
  /** Fragments lighter than this are merged with the next one. */
  minLength: number;
  /** Heaviest sentence after the first. */
  maxLength: number;
  /** Heaviest first sentence (kept short for a quick first audio). */
  firstMaxLength: number;
}

export const DEFAULT_SENTENCE_OPTIONS: SentenceOptions = {
  minLength: 12,
  maxLength: 240,
  firstMaxLength: 120,
};

/** Punctuation that ends a sentence wherever it stands (no space needed after it). */
const CJK_TERMINATORS = new Set(["。", "！", "？", "｡", "．", "।", "॥", "؟", "۔", "։", "።", "။", "‼", "⁇", "⁈", "⁉"]);
/** Punctuation that ends a sentence when whitespace, CJK text or the end follows. */
const LATIN_TERMINATORS = new Set([".", "!", "?", "…"]);
/** Closers that stay with the sentence they close. */
const CLOSERS = new Set(['"', "'", "”", "’", "»", ")", "]", "}", "」", "』", "）", "】", "〉", "》", "〕", "］", "｝", "〞"]);
/** Closers after which no space is needed when two fragments are glued back together. */
const WIDE_CLOSERS = new Set(["」", "』", "）", "】", "〉", "》", "〕", "］", "｝", "〞"]);
/** Corner brackets that end a line of dialogue when another one opens (or at the end). */
const DIALOGUE_CLOSERS = new Set(["」", "』"]);
const DIALOGUE_OPENERS = new Set(["「", "『"]);
/** Clause punctuation for cutting long sentences. */
const CLAUSE_BREAKS = new Set([",", ";", ":", "、", "，", "；", "：", "—", "–", "،", "؛"]);

/**
 * Common abbreviations (lower case, without the final period) that do not end a sentence. Words that also end
 * sentences as ordinary words ("no", "I") are handled in {@link isNonTerminalPeriod} instead.
 */
const ABBREVIATIONS = new Set([
  "mr", "mrs", "ms", "dr", "prof", "sr", "jr", "st", "mt", "vs", "etc", "e.g", "i.e", "cf", "vol", "fig",
  "inc", "ltd", "co", "corp", "approx", "dept", "est", "jan", "feb", "mar", "apr", "jun", "jul", "aug", "sep",
  "sept", "oct", "nov", "dec", "u.s", "u.k", "a.m", "p.m", "ca", "gen", "col", "lt", "sgt", "capt", "rev",
]);
/** Abbreviations that are only abbreviations before a number ("No. 5", "Nos. 3 and 4"). */
const NUMBER_ABBREVIATIONS = new Set(["no", "nos", "nr"]);

const LETTER_OR_DIGIT = /[\p{L}\p{N}]/u;
/** Symbols (emoji, currency, math): something an engine may read aloud even without letters or digits. */
const SYMBOL = /\p{S}/u;
const DIGIT = /\p{N}/u;
const INITIAL = /^\p{Lu}$/u;
const WIDE = /[\p{Script=Han}\p{Script=Hiragana}\p{Script=Katakana}\p{Script=Hangul}]/u;
const WHITESPACE = /\s/u;

/** Weight of one character (see the module docs). */
function charWeight(ch: string): number {
  return WIDE.test(ch) ? 2 : 1;
}

/** Weight of a string (see the module docs). */
export function textWeight(text: string): number {
  let w = 0;
  for (const ch of text) w += charWeight(ch);
  return w;
}

/** The whitespace-delimited token after `chars[i]` on the same line ("" at the end or a line break). */
function tokenAfter(chars: string[], i: number): string {
  let k = i + 1;
  while (k < chars.length && chars[k] !== "\n" && WHITESPACE.test(chars[k])) k++;
  let end = k;
  while (end < chars.length && !WHITESPACE.test(chars[end])) end++;
  return chars.slice(k, end).join("");
}

/** The whitespace-delimited token that ends before `chars[j]` (skipping the whitespace at and before `j`). */
function tokenBefore(chars: string[], j: number): string {
  let k = j;
  while (k >= 0 && chars[k] !== "\n" && WHITESPACE.test(chars[k])) k--;
  const end = k + 1;
  while (k >= 0 && !WHITESPACE.test(chars[k])) k--;
  return chars.slice(k + 1, end).join("");
}

/** An initial with its period ("K."). */
const INITIAL_WITH_PERIOD = /^\p{Lu}\.$/u;

/** Whether the lone period at `chars[i]` belongs to an abbreviation, an initial or a number. */
function isNonTerminalPeriod(chars: string[], i: number): boolean {
  const next = chars[i + 1];
  // "3.14", "example.com": something other than whitespace or a closer follows directly.
  if (next !== undefined && !WHITESPACE.test(next) && !CLOSERS.has(next) && !WIDE.test(next)) return true;
  let j = i - 1;
  while (j >= 0 && !WHITESPACE.test(chars[j]) && !CLOSERS.has(chars[j]) && chars[j] !== "(") j--;
  const word = chars.slice(j + 1, i).join("");
  if (!word) return false;
  // "J. K. Rowling": a single upper-case letter. The pronoun "I" ends sentences ("So do I."), so it counts as an
  // initial only next to another initial ("I. M. Pei", "J. I. Smith").
  if (INITIAL.test(word)) {
    if (word !== "I") return true;
    return INITIAL_WITH_PERIOD.test(tokenAfter(chars, i)) || INITIAL_WITH_PERIOD.test(tokenBefore(chars, j));
  }
  const lower = word.toLowerCase();
  // "No. 5" is an abbreviation, "I said no." is not.
  if (NUMBER_ABBREVIATIONS.has(lower)) return DIGIT.test(Array.from(tokenAfter(chars, i))[0] ?? "");
  return ABBREVIATIONS.has(lower);
}

/** Split at hard boundaries (rule 1). Fragments keep their punctuation and are trimmed. */
function splitHard(text: string): string[] {
  const chars = Array.from(text.replace(/\r\n?/g, "\n"));
  const n = chars.length;
  const out: string[] = [];
  let start = 0;
  const cut = (end: number) => {
    const piece = chars.slice(start, end).join("").trim();
    if (piece) out.push(piece);
    start = end;
  };
  for (let i = 0; i < n; i++) {
    const ch = chars[i];
    if (ch === "\n") {
      cut(i + 1);
      continue;
    }
    if (CJK_TERMINATORS.has(ch) || LATIN_TERMINATORS.has(ch)) {
      // A run of terminators ("?!", "...", "。。") ends the sentence together.
      let end = i + 1;
      while (end < n && (CJK_TERMINATORS.has(chars[end]) || LATIN_TERMINATORS.has(chars[end]))) end++;
      const runHasCjk = chars.slice(i, end).some((c) => CJK_TERMINATORS.has(c));
      if (!runHasCjk && end === i + 1 && ch === "." && isNonTerminalPeriod(chars, i)) continue;
      // Closing quotes / brackets stay with the sentence.
      while (end < n && CLOSERS.has(chars[end])) end++;
      if (!runHasCjk) {
        const next = chars[end];
        if (next !== undefined && !WHITESPACE.test(next) && !WIDE.test(next) && !DIALOGUE_OPENERS.has(next)) {
          // Latin punctuation glued to the next word ("wait...what"): not a boundary.
          i = end - 1;
          continue;
        }
      }
      cut(end);
      i = end - 1;
      continue;
    }
    if (DIALOGUE_CLOSERS.has(ch)) {
      let k = i + 1;
      while (k < n && chars[k] !== "\n" && WHITESPACE.test(chars[k])) k++;
      if (k >= n || DIALOGUE_OPENERS.has(chars[k])) cut(i + 1);
    }
  }
  cut(n);
  return out;
}

/** Glue two fragments back together, with a space only where the script uses spaces. */
function join(a: string, b: string): string {
  if (!a) return b;
  if (!b) return a;
  const last = Array.from(a).pop() ?? "";
  const first = Array.from(b)[0] ?? "";
  const tight =
    WIDE.test(last) ||
    WIDE.test(first) ||
    CJK_TERMINATORS.has(last) ||
    WIDE_CLOSERS.has(last) ||
    CJK_TERMINATORS.has(first) ||
    DIALOGUE_OPENERS.has(first);
  return tight ? a + b : `${a} ${b}`;
}

/** Cut `chars` once so the head weighs at most `max`; returns [head, rest] (both trimmed). */
function cutOnce(chars: string[], max: number): [string, string[]] {
  let w = 0;
  let limit = 0;
  while (limit < chars.length && w + charWeight(chars[limit]) <= max) {
    w += charWeight(chars[limit]);
    limit++;
  }
  limit = Math.max(1, limit);
  let cutAt = -1;
  // After clause punctuation in the second half of the window...
  for (let k = limit - 1; k >= Math.floor(limit / 2); k--) {
    if (CLAUSE_BREAKS.has(chars[k])) {
      cutAt = k + 1;
      break;
    }
  }
  // ...else at a space there...
  if (cutAt < 0) {
    for (let k = limit - 1; k >= Math.floor(limit / 2); k--) {
      if (WHITESPACE.test(chars[k])) {
        cutAt = k;
        break;
      }
    }
  }
  // ...else hard.
  if (cutAt <= 0) cutAt = limit;
  const head = chars.slice(0, cutAt).join("").trim();
  const rest = Array.from(chars.slice(cutAt).join("").trimStart());
  return [head, rest];
}

/** Cut `sentence` into pieces: the first no heavier than `firstMax`, the others no heavier than `max`. */
function capLength(sentence: string, firstMax: number, max: number): string[] {
  const out: string[] = [];
  let chars = Array.from(sentence);
  let limit = firstMax;
  while (chars.length && textWeight(chars.join("")) > limit) {
    const [head, rest] = cutOnce(chars, limit);
    if (head) out.push(head);
    chars = rest;
    limit = max;
  }
  const rest = chars.join("").trim();
  if (rest) out.push(rest);
  return out;
}

/**
 * Split `text` into sentences for synthesis (see the module docs). Returns `[]` for text without anything to
 * speak (whitespace and punctuation only). Joining the result reproduces the text up to whitespace, except for
 * punctuation before the first sentence.
 */
export function splitSentences(text: string, options: Partial<SentenceOptions> = {}): string[] {
  const opts = { ...DEFAULT_SENTENCE_OPTIONS, ...options };
  const minLength = Math.max(0, opts.minLength);
  const maxLength = Math.max(8, opts.maxLength);
  const firstMax = Math.max(8, Math.min(maxLength, opts.firstMaxLength));

  // Rule 2: fragments without letters or digits join the previous sentence. Before the first one, symbols (emoji)
  // are carried into it and bare punctuation is dropped.
  const fragments: string[] = [];
  let lead = "";
  for (const piece of splitHard(text)) {
    if (!LETTER_OR_DIGIT.test(piece)) {
      if (fragments.length) fragments[fragments.length - 1] = join(fragments[fragments.length - 1], piece);
      else if (lead || SYMBOL.test(piece)) lead = join(lead, piece);
      continue;
    }
    fragments.push(join(lead, piece));
    lead = "";
  }
  // Nothing but emoji / symbols ("👍", "❤️🔥"): one sentence, as before the pipeline (engines that read symbols,
  // like Azure reading emoji names, still do). Punctuation alone ("?!", "...") has nothing to say.
  if (!fragments.length && lead) fragments.push(lead);

  // Rule 3: merge light fragments into the next one (a light last one into the previous).
  const merged: string[] = [];
  let pending = "";
  for (const piece of fragments) {
    const candidate = join(pending, piece);
    if (textWeight(candidate) < minLength) {
      pending = candidate;
      continue;
    }
    merged.push(candidate);
    pending = "";
  }
  if (pending) {
    const lastIndex = merged.length - 1;
    if (lastIndex >= 0 && textWeight(join(merged[lastIndex], pending)) <= maxLength) {
      merged[lastIndex] = join(merged[lastIndex], pending);
    } else {
      merged.push(pending);
    }
  }

  // Rule 4: cap lengths, the very first sentence tighter.
  const out: string[] = [];
  for (const sentence of merged) {
    out.push(...capLength(sentence, out.length === 0 ? firstMax : maxLength, maxLength));
  }
  return out;
}
