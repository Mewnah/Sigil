import { describe, expect, it } from "vitest";
import { agreedPrefix, completeTokens, LocalAgreement, tokenize } from "../localAgreement";

/** Small deterministic PRNG (mulberry32) so property tests are reproducible. */
function rng(seed: number): () => number {
  let a = seed >>> 0;
  return () => {
    a = (a + 0x6d2b79f5) >>> 0;
    let t = a;
    t = Math.imul(t ^ (t >>> 15), t | 1);
    t ^= t + Math.imul(t ^ (t >>> 7), t | 61);
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}

const EN_WORDS = ["I", "want", "to", "go", "the", "store", "tomorrow,", "maybe.", "we", "can", "meet", "there", "at", "noon"];
const JA_WORDS = ["今日", "は", "東京", "に", "行き", "ます", "。", "明日", "東北", "で", "会い", "ましょう"];

function randomSentence(r: () => number, words: readonly string[], sep: string, len: number): string {
  const out: string[] = [];
  for (let i = 0; i < len; i++) out.push(words[Math.floor(r() * words.length)]);
  return out.join(sep);
}

/** A hypothesis that keeps a random-length prefix of the previous one and continues randomly. */
function nextHypothesis(r: () => number, prev: string, words: readonly string[], sep: string, lang: string): string {
  const tokens = tokenize(prev, lang);
  const keep = Math.floor(r() * (tokens.length + 1));
  const kept = tokens.slice(0, keep).join("");
  const tail = randomSentence(r, words, sep, 1 + Math.floor(r() * 4));
  return kept && sep && !/\s$/.test(kept) ? `${kept}${sep}${tail}` : `${kept}${tail}`;
}

describe("tokenize", () => {
  it("round-trips any text exactly, spaced or not", () => {
    const r = rng(7);
    const samples = [
      "",
      " leading and  double  spaces ",
      "Hello, world! How are you?",
      "今日は東京に行きます。明日は？",
      "สวัสดีครับ ผมชื่อ",
      "mixed 日本語 and English 123",
      "emoji 👍🏽 and ZWJ 👩‍👩‍👧 text",
    ];
    for (let i = 0; i < 200; i++) {
      samples.push(randomSentence(r, [...EN_WORDS, ...JA_WORDS, " ", "  ", "\n"], r() < 0.5 ? " " : "", 1 + Math.floor(r() * 12)));
    }
    for (const s of samples) {
      for (const lang of ["", "en", "ja", "th", "ko"]) expect(tokenize(s, lang).join(""), `${lang}:${s}`).toBe(s);
    }
  });

  it("splits spaced scripts into words with their trailing space", () => {
    expect(tokenize("I want to go", "en")).toEqual(["I ", "want ", "to ", "go"]);
  });

  it("splits CJK into words with Intl.Segmenter", () => {
    const tokens = tokenize("今日は東京に行きます。", "ja");
    expect(tokens.join("")).toBe("今日は東京に行きます。");
    expect(tokens).toContain("東京");
  });
});

describe("agreedPrefix", () => {
  it("agrees on whole words only", () => {
    expect(agreedPrefix("I want to go", "I want to go to the store", "en")).toBe("I want to go ");
    expect(agreedPrefix("I want to go home", "I want to go to the store", "en")).toBe("I want to go ");
    expect(agreedPrefix("I want to go,", "I want to go to the store", "en")).toBe("I want to ");
    expect(agreedPrefix("I wanted", "I want", "en")).toBe("I ");
    expect(agreedPrefix("", "anything", "en")).toBe("");
  });

  it("uses word segmentation for CJK (never half a word)", () => {
    // 東京 and 東北 share the character 東, but not the word.
    expect(agreedPrefix("今日は東京に行きます", "今日は東北に行きます", "ja")).toBe("今日は");
    expect(agreedPrefix("我们明天去北京", "我们明天去上海", "zh")).toBe("我们明天去");
  });

  it("detects unspaced text by script when the language is unknown", () => {
    expect(agreedPrefix("今日は東京に行きます", "今日は東北に行きます", "")).toBe("今日は");
  });
});

describe("completeTokens", () => {
  it("holds back a possibly incomplete last word", () => {
    expect(completeTokens("I want to go to the sto", "en")).toBe("I want to go to the ");
    expect(completeTokens("I want ", "en")).toBe("I want ");
    expect(completeTokens("word", "en")).toBe("");
    expect(completeTokens("", "en")).toBe("");
  });
});

describe("LocalAgreement-2", () => {
  it("commits the prefix two consecutive hypotheses agree on", () => {
    const la = new LocalAgreement("en");
    expect(la.update("I want").committed).toBe("");
    const u = la.update("I want to go");
    expect(u.committed).toBe("I want ");
    expect(u.grew).toBe(true);
    expect(u.tentative).toBe("to go");
    expect(la.update("I want to go there").committed).toBe("I want to go ");
  });

  it("never retracts committed text when a later hypothesis diverges", () => {
    const la = new LocalAgreement("en");
    la.update("We should leave now");
    la.update("We should leave now please");
    expect(la.committed).toBe("We should leave now ");
    const u = la.update("You must go");
    expect(u.committed).toBe("We should leave now ");
    expect(u.grew).toBe(false);
    expect(u.tentative).toBe("");
  });

  it("property: the committed prefix is monotonic and always agreed by two hypotheses (en and ja)", () => {
    for (const [lang, words, sep] of [
      ["en", EN_WORDS, " "],
      ["ja", JA_WORDS, ""],
    ] as const) {
      for (let seed = 1; seed <= 200; seed++) {
        const r = rng(seed * 31 + lang.length);
        const la = new LocalAgreement(lang);
        let hyp = randomSentence(r, words, sep, 1 + Math.floor(r() * 5));
        let previous: string | null = null;
        let committed = "";
        for (let step = 0; step < 25; step++) {
          const u = la.update(hyp);
          // Monotonic: every committed prefix extends the previous one.
          expect(u.committed.startsWith(committed), `${lang}#${seed}/${step}`).toBe(true);
          if (u.grew) {
            // A growth is agreed by the last two hypotheses.
            expect(previous).not.toBeNull();
            expect(hyp.startsWith(u.committed)).toBe(true);
            expect(previous!.startsWith(u.committed.trimEnd())).toBe(true);
          }
          // What is shown with a tentative tail is the hypothesis itself (or just the committed part).
          if (u.tentative) expect(u.committed + u.tentative).toBe(hyp);
          committed = u.committed;
          previous = hyp;
          hyp = nextHypothesis(r, hyp, words, sep, lang);
        }
      }
    }
  });
});
