/**
 * LocalAgreement-2 commit policy for translating live partial captions (research §3.2): every interim translation
 * is a new hypothesis for the whole line; the longest common prefix of the last two hypotheses (in whole tokens) is
 * committed, and the committed prefix only ever grows, so captions built from it never flicker. The STT final
 * flushes the line with the engine's translation of the complete sentence (it may revise the committed text once,
 * because it is translated from the full source).
 *
 * Tokens: whitespace-delimited words (with their trailing whitespace) for spaced scripts, `Intl.Segmenter` words for
 * scripts without spaces (Chinese, Japanese, Thai, ...). Tokens always concatenate back to the exact input, so
 * committed prefixes are exact prefixes of the hypotheses. Owned by P4-C.
 */
import { isUnspaced, UNSPACED_SCRIPT_RE } from "./languages";

type Segmenter = { segment(text: string): Iterable<{ segment: string }> };
type SegmenterCtor = new (locale?: string, options?: { granularity?: "grapheme" | "word" | "sentence" }) => Segmenter;

const segmenters = new Map<string, Segmenter | null>();

function wordSegmenter(locale: string): Segmenter | null {
  const key = locale || "und";
  if (segmenters.has(key)) return segmenters.get(key) ?? null;
  let seg: Segmenter | null = null;
  const Ctor = (Intl as unknown as { Segmenter?: SegmenterCtor }).Segmenter;
  if (typeof Ctor === "function") {
    try {
      seg = new Ctor(locale || undefined, { granularity: "word" });
    } catch {
      try {
        seg = new Ctor(undefined, { granularity: "word" });
      } catch {
        seg = null;
      }
    }
  }
  segmenters.set(key, seg);
  return seg;
}

/** Words with their trailing whitespace; leading whitespace is its own token. */
function spacedTokens(text: string): string[] {
  const tokens: string[] = [];
  const lead = /^\s+/.exec(text);
  if (lead) tokens.push(lead[0]);
  const re = /\S+\s*/g;
  let m: RegExpExecArray | null;
  while ((m = re.exec(text)) !== null) tokens.push(m[0]);
  return tokens;
}

/**
 * Tokens of `text` for agreement, in the target language `lang` (canonical BCP-47, "" = unknown: decided by the
 * text's script). `tokens.join("") === text` always holds.
 */
export function tokenize(text: string, lang = ""): string[] {
  if (!text) return [];
  const unspaced = lang ? isUnspaced(lang) : UNSPACED_SCRIPT_RE.test(text);
  if (!unspaced) return spacedTokens(text);
  const seg = wordSegmenter(lang);
  // No Intl.Segmenter: code points (safe, just more conservative for agreement).
  if (!seg) return Array.from(text);
  const out: string[] = [];
  for (const s of seg.segment(text)) out.push(s.segment);
  return out;
}

/**
 * Longest common prefix of `a` and `b` in whole tokens (compared without their trailing whitespace, so a sentence's
 * last word agrees with the same word followed by more text). The result is taken from `b`'s tokens, so it is an
 * exact prefix of `b` (the newer hypothesis).
 */
export function agreedPrefix(a: string, b: string, lang = ""): string {
  if (!a || !b) return "";
  const ta = tokenize(a, lang);
  const tb = tokenize(b, lang);
  let out = "";
  for (let i = 0; i < ta.length && i < tb.length; i++) {
    if (ta[i].trimEnd() !== tb[i].trimEnd()) break;
    out += tb[i];
  }
  return out;
}

/**
 * Drops the last token of a growing (streamed) text, which may still be incomplete: "I want to go to the sto" →
 * "I want to go to the ". Text that ends in whitespace keeps its last word.
 */
export function completeTokens(text: string, lang = ""): string {
  if (!text) return "";
  if (/\s$/.test(text)) return text;
  const tokens = tokenize(text, lang);
  tokens.pop();
  return tokens.join("");
}

export interface LocalAgreementUpdate {
  /** Committed prefix after this hypothesis (never shorter than before). */
  committed: string;
  /** The hypothesis beyond the committed prefix; "" when the hypothesis does not extend the committed prefix. */
  tentative: string;
  /** The committed prefix grew. */
  grew: boolean;
}

/** LocalAgreement-2 state of one line. */
export class LocalAgreement {
  #committed = "";
  #previous: string | null = null;

  /** @param lang Target language (canonical BCP-47; "" = decided by script). */
  constructor(readonly lang = "") {}

  /** Committed prefix so far (may end in whitespace; trim it for display). */
  get committed(): string {
    return this.#committed;
  }

  /** Last hypothesis seen. */
  get previous(): string | null {
    return this.#previous;
  }

  /** Feeds the next complete hypothesis for the whole line. */
  update(hypothesis: string): LocalAgreementUpdate {
    const before = this.#committed;
    if (this.#previous !== null) {
      const agreed = agreedPrefix(this.#previous, hypothesis, this.lang);
      // Never retract: a shorter or diverging agreement leaves the committed prefix as it is.
      if (agreed.length > before.length && agreed.startsWith(before)) this.#committed = agreed;
    }
    this.#previous = hypothesis;
    const committed = this.#committed;
    const tentative = hypothesis.startsWith(committed) ? hypothesis.slice(committed.length) : "";
    return { committed, tentative, grew: committed.length > before.length };
  }

  reset(): void {
    this.#committed = "";
    this.#previous = null;
  }
}
