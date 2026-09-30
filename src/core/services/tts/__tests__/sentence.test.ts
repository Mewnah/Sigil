import { describe, expect, it } from "vitest";
import { DEFAULT_SENTENCE_OPTIONS, splitSentences, textWeight } from "../sentence";

const raw = (text: string) => splitSentences(text, { minLength: 0 });
const LONE_SURROGATE = /[\uD800-\uDBFF](?![\uDC00-\uDFFF])|(?<![\uD800-\uDBFF])[\uDC00-\uDFFF]/;

describe("splitSentences", () => {
  it("splits English at sentence ends and merges short fragments into the next one", () => {
    expect(splitSentences("Hello there, how are you today? I am fine. Thanks for asking!")).toEqual([
      "Hello there, how are you today?",
      "I am fine. Thanks for asking!",
    ]);
    expect(raw("One. Two! Three? Four…")).toEqual(["One.", "Two!", "Three?", "Four…"]);
  });

  it("splits Japanese and Chinese at full-width punctuation without spaces", () => {
    expect(splitSentences("今日はいい天気ですね。散歩に行きましょうか？それとも家にいますか。")).toEqual([
      "今日はいい天気ですね。",
      "散歩に行きましょうか？",
      "それとも家にいますか。",
    ]);
    expect(raw("你好！今天天气很好。我们去散步吧？")).toEqual(["你好！", "今天天气很好。", "我们去散步吧？"]);
    // Half-width punctuation before CJK text also ends a sentence.
    expect(raw("Really?本当ですか。")).toEqual(["Really?", "本当ですか。"]);
  });

  it("keeps closing quotes and brackets with their sentence and splits dialogue lines", () => {
    expect(raw("「はい。」「いいえ。」")).toEqual(["「はい。」", "「いいえ。」"]);
    expect(raw("「はい」「いいえ」")).toEqual(["「はい」", "「いいえ」"]);
    expect(raw("「こんにちは」と言った。そして帰った。")).toEqual(["「こんにちは」と言った。", "そして帰った。"]);
    expect(raw('He said "Stop!" Then he left.')).toEqual(['He said "Stop!"', "Then he left."]);
    // Merged back without an artificial space.
    expect(splitSentences("「はい。」「いいえ。」")).toEqual(["「はい。」「いいえ。」"]);
  });

  it("does not split abbreviations, initials or numbers", () => {
    expect(splitSentences("Dr. Smith paid $3.50 for it, e.g. a coffee. Then he left the shop.")).toEqual([
      "Dr. Smith paid $3.50 for it, e.g. a coffee.",
      "Then he left the shop.",
    ]);
    expect(splitSentences("J. K. Rowling wrote it. Everyone read it twice.")).toEqual([
      "J. K. Rowling wrote it.",
      "Everyone read it twice.",
    ]);
    expect(raw("Visit example.com today. Version 2.0 is out.")).toEqual(["Visit example.com today.", "Version 2.0 is out."]);
    expect(raw("Read No. 5 first. Then nos. 3 and 4.")).toEqual(["Read No. 5 first.", "Then nos. 3 and 4."]);
    expect(raw("I. M. Pei designed it. J. I. Smith built it.")).toEqual(["I. M. Pei designed it.", "J. I. Smith built it."]);
  });

  it("ends sentences at the word 'no' and the pronoun 'I' (review finding: they were merged)", () => {
    expect(raw("I said no. We are leaving now.")).toEqual(["I said no.", "We are leaving now."]);
    expect(raw("So do I. Then we go home.")).toEqual(["So do I.", "Then we go home."]);
    expect(raw("No. No. I will not.")).toEqual(["No.", "No.", "I will not."]);
  });

  it("keeps punctuation-only fragments with the previous sentence and drops text with nothing to say", () => {
    expect(raw("Hello there my friend. ... ! Next sentence here.")).toEqual([
      "Hello there my friend. ... !",
      "Next sentence here.",
    ]);
    expect(splitSentences("Wait... what?! Really?!")).toEqual(["Wait... what?! Really?!"]);
    expect(splitSentences("")).toEqual([]);
    expect(splitSentences("   \n\n  ")).toEqual([]);
    expect(splitSentences("?!… 。")).toEqual([]);
    expect(splitSentences("...")).toEqual([]);
  });

  it("passes emoji-only text through whole (review finding: it was dropped, so nothing was said)", () => {
    expect(splitSentences("👍")).toEqual(["👍"]);
    const heartOnFire = "❤️‍\u{1F525}";
    expect(splitSentences(` ${heartOnFire} `)).toEqual([heartOnFire]);
    expect(splitSentences("🎉🎉!")).toEqual(["🎉🎉!"]);
    expect(splitSentences("👍\n😂")).toEqual(["👍 😂"]);
    // Emoji before the first sentence stay with it; bare punctuation there is still dropped.
    expect(raw("👍\nGreat job everyone. See you soon.")).toEqual(["👍 Great job everyone.", "See you soon."]);
    expect(raw("?! Great job everyone.")).toEqual(["Great job everyone."]);
    expect(raw("Great job everyone. 👏👏")).toEqual(["Great job everyone. 👏👏"]);
  });

  it("treats line breaks as boundaries", () => {
    expect(splitSentences("Line one is here\r\nLine two is here\n\nand three")).toEqual([
      "Line one is here",
      "Line two is here and three",
    ]);
  });

  it("splits other scripts: Devanagari, Arabic, Armenian", () => {
    expect(raw("नमस्ते दोस्तों। आप कैसे हैं? मैं ठीक हूँ।")).toEqual(["नमस्ते दोस्तों।", "आप कैसे हैं?", "मैं ठीक हूँ।"]);
    expect(raw("مرحبا بكم جميعا؟ كيف حالكم اليوم.")).toEqual(["مرحبا بكم جميعا؟", "كيف حالكم اليوم."]);
    expect(raw("Բարեւ։ Ինչպես ես։")).toEqual(["Բարեւ։", "Ինչպես ես։"]);
  });

  it("caps long sentences at clause punctuation, the first one tighter for a quick first audio", () => {
    const clause = "this clause keeps going for a while";
    const long = Array.from({ length: 12 }, () => clause).join(", ") + ".";
    const out = splitSentences(long);
    expect(out.length).toBeGreaterThan(2);
    expect(textWeight(out[0])).toBeLessThanOrEqual(DEFAULT_SENTENCE_OPTIONS.firstMaxLength);
    for (const s of out.slice(1)) expect(textWeight(s)).toBeLessThanOrEqual(DEFAULT_SENTENCE_OPTIONS.maxLength);
    // Cut after a comma, and nothing is lost.
    expect(out[0].endsWith(",")).toBe(true);
    expect(out.join(" ").replace(/\s+/g, " ")).toBe(long);
  });

  it("cuts unpunctuated CJK text hard at the weight limit and never inside a surrogate pair", () => {
    const kanji = "漢".repeat(200);
    const out = splitSentences(kanji);
    expect(out.map((s) => s.length)).toEqual([60, 120, 20]);
    expect(out.join("")).toBe(kanji);

    const emoji = "x😀".repeat(80);
    const pieces = splitSentences(emoji, { firstMaxLength: 21, maxLength: 33, minLength: 0 });
    expect(pieces.join("")).toBe(emoji);
    for (const p of pieces) expect(LONE_SURROGATE.test(p)).toBe(false);
  });

  it("weighs CJK, kana and Hangul characters double", () => {
    expect(textWeight("abc")).toBe(3);
    expect(textWeight("日本語")).toBe(6);
    expect(textWeight("ひらがなカタカナ한글")).toBe(20);
    expect(textWeight("😀")).toBe(1);
  });
});
