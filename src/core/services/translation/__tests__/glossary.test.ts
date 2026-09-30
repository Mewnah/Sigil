import { describe, expect, it } from "vitest";
import { filterGlossary, glossaryLanguageMatches, GlossaryEntryLike, termOccurs } from "../glossary";

const entry = (source: string, target: string, patch: Partial<GlossaryEntryLike> = {}): GlossaryEntryLike => ({
  source,
  target,
  from: "",
  to: "",
  caseSensitive: false,
  enabled: true,
  ...patch,
});

describe("termOccurs", () => {
  it("matches whole words in spaced scripts", () => {
    expect(termOccurs("I love my cat", "cat")).toBe(true);
    expect(termOccurs("a category of things", "cat")).toBe(false);
    expect(termOccurs("cats are nice", "cat")).toBe(false);
    expect(termOccurs("Welcome to New York!", "new york")).toBe(true);
    expect(termOccurs("VRChat, again", "VRChat")).toBe(true);
  });

  it("matches CJK terms anywhere", () => {
    expect(termOccurs("今日は東京に行きます", "東京")).toBe(true);
    expect(termOccurs("シジルを使います", "シジル")).toBe(true);
  });

  it("honours case sensitivity and normalizes full-width forms", () => {
    expect(termOccurs("I use sigil daily", "Sigil", true)).toBe(false);
    expect(termOccurs("I use Sigil daily", "Sigil", true)).toBe(true);
    expect(termOccurs("ＶＲＣｈａｔ is fun", "VRChat")).toBe(true);
  });

  it("finds a later whole-word occurrence after an embedded one", () => {
    expect(termOccurs("category then cat", "cat")).toBe(true);
  });

  it("ignores empty terms", () => {
    expect(termOccurs("anything", "  ")).toBe(false);
  });

  it("finds Korean terms with particles attached, but not inside another word", () => {
    expect(termOccurs("민지가 왔어요", "민지")).toBe(true);
    expect(termOccurs("서울에서 만나요", "서울")).toBe(true);
    expect(termOccurs("김민지가 왔어요", "민지")).toBe(false); // still needs a boundary before the term
  });

  it("allows suffixes after a term in suffixing languages only", () => {
    expect(termOccurs("Sigilde yayın yapıyorum", "Sigil", false, "tr")).toBe(true);
    expect(termOccurs("Sigilissä on bugi", "Sigil", false, "fi")).toBe(true);
    expect(termOccurs("Sigilde yayın yapıyorum", "Sigil")).toBe(false); // unknown language: whole words
    expect(termOccurs("a category of things", "cat", false, "en")).toBe(false);
  });
});

describe("glossaryLanguageMatches", () => {
  it("treats blank entry languages and unknown line languages as any", () => {
    expect(glossaryLanguageMatches("", "ja")).toBe(true);
    expect(glossaryLanguageMatches("ja", "")).toBe(true);
    expect(glossaryLanguageMatches("ja-JP", "ja")).toBe(true);
    expect(glossaryLanguageMatches("ja", "en")).toBe(false);
    expect(glossaryLanguageMatches("zh", "zh-Hant")).toBe(false);
  });
});

describe("filterGlossary", () => {
  const entries = [
    entry("Sigil", "シジル"),
    entry("VRChat", "VRChat", { to: "ja" }),
    entry("cat", "猫", { from: "en" }),
    entry("disabled", "x", { enabled: false }),
    entry("", "empty source"),
    entry("sigil", "duplicate"),
    entry("stream", "配信", { to: "zh" }),
  ];

  it("keeps enabled terms of the line in the user's order", () => {
    expect(filterGlossary(entries, "Sigil works in VRChat with my cat", "en", "ja")).toEqual([
      { source: "Sigil", target: "シジル" },
      { source: "VRChat", target: "VRChat" },
      { source: "cat", target: "猫" },
    ]);
  });

  it("filters by language pair", () => {
    expect(filterGlossary(entries, "my cat on stream", "fr", "ja")).toEqual([]);
    expect(filterGlossary(entries, "my cat on stream", "en", "zh")).toEqual([
      { source: "cat", target: "猫" },
      { source: "stream", target: "配信" },
    ]);
  });

  it("limits the number of terms", () => {
    const many = Array.from({ length: 50 }, (_, i) => entry(`term${i}`, `t${i}`));
    const text = many.map((e) => e.source).join(" ");
    expect(filterGlossary(many, text, "en", "ja")).toHaveLength(24);
    expect(filterGlossary(many, text, "en", "ja", 3)).toHaveLength(3);
  });

  it("sends Korean names followed by particles", () => {
    expect(filterGlossary([entry("민지", "Minji")], "민지가 왔어요", "ko", "en")).toEqual([{ source: "민지", target: "Minji" }]);
    expect(filterGlossary([entry("Sigil", "Sigil")], "Sigilde yayın", "tr", "en")).toEqual([{ source: "Sigil", target: "Sigil" }]);
  });

  it("handles missing input", () => {
    expect(filterGlossary(undefined, "text", "en", "ja")).toEqual([]);
    expect(filterGlossary(entries, "  ", "en", "ja")).toEqual([]);
  });
});
