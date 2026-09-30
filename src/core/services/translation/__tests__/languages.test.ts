import { describe, expect, it } from "vitest";
import { TextEventSchema, TextEventType } from "@/types";
import fixture from "./languages.fixture.json";
import {
  canonicalLanguage,
  guessLanguage,
  isCjk,
  isUnspaced,
  languageInfo,
  languageName,
  sameLanguage,
  toAzure,
  toFlores,
  toIso6391,
  TRANSLATION_LANGUAGES,
} from "../languages";

describe("TRANSLATION_LANGUAGES table", () => {
  it("has unique canonical codes that TextEvent.lang accepts", () => {
    const codes = TRANSLATION_LANGUAGES.map((l) => l.code);
    expect(new Set(codes).size).toBe(codes.length);
    for (const code of codes) {
      const ev = TextEventSchema.parse({ type: TextEventType.final, value: "x", lang: code });
      expect(ev.lang, code).toBe(code);
    }
  });

  it("uses well-formed FLORES-200, ISO 639-1 and Azure codes", () => {
    for (const l of TRANSLATION_LANGUAGES) {
      if (l.flores !== null) expect(l.flores, l.code).toMatch(/^[a-z]{3}_[A-Z][a-z]{3}$/);
      if (l.iso6391 !== null) expect(l.iso6391, l.code).toMatch(/^[a-z]{2}$/);
      if (l.azure !== null) expect(l.azure, l.code).toMatch(/^[a-z]{2,3}(-[A-Za-z]{2,4})?$/);
      expect(l.name.trim(), l.code).not.toBe("");
    }
  });

  it("covers every NLLB-200 language that Sigil can name (about 200)", () => {
    expect(TRANSLATION_LANGUAGES.filter((l) => l.flores !== null).length).toBeGreaterThanOrEqual(195);
  });

  it("matches the shared fixture read by Rust mt_local/flores.rs", () => {
    const table = TRANSLATION_LANGUAGES.map(({ code, flores, azure, iso6391 }) => ({ code, flores, azure, iso6391 }));
    expect(fixture.languages).toEqual(table);
  });
});

describe("canonicalLanguage", () => {
  it.each([
    ["EN_us", "en"],
    ["en-GB", "en"],
    ["ja-JP", "ja"],
    ["zh", "zh"],
    ["zh-CN", "zh"],
    ["zh-Hans", "zh"],
    ["zh-Hans-SG", "zh"],
    ["zh-TW", "zh-Hant"],
    ["zh-HK", "zh-Hant"],
    ["zh-Hant-TW", "zh-Hant"],
    ["cmn", "zh"],
    ["yue", "yue"],
    ["pt-BR", "pt"],
    ["pt-PT", "pt-PT"],
    ["no", "nb"],
    ["fil", "tl"],
    ["iw", "he"],
    ["jw", "jv"],
    ["jpn_Jpan", "ja"],
    ["zho_Hant", "zh-Hant"],
    ["jpn", "ja"],
    ["arb", "ar"],
    ["pes_Arab", "fa"],
    ["sr-Latn", "sr"],
    ["mn-Cyrl", "mn"],
    ["arz", "arz"],
  ])("%s → %s", (input, expected) => {
    expect(canonicalLanguage(input)).toBe(expected);
  });

  it("returns '' for empty, auto, und and unknown codes", () => {
    for (const v of ["", "  ", "auto", "AUTO", "und", "xx", "klingon", null, undefined]) {
      expect(canonicalLanguage(v)).toBe("");
    }
  });

  it("is idempotent on canonical codes", () => {
    for (const l of TRANSLATION_LANGUAGES) expect(canonicalLanguage(l.code)).toBe(l.code);
  });

  it("maps to each engine's codes", () => {
    expect(toFlores("ja")).toBe("jpn_Jpan");
    expect(toFlores("zh-TW")).toBe("zho_Hant");
    expect(toFlores("xx")).toBeNull();
    expect(toAzure("zh")).toBe("zh-Hans");
    expect(toAzure("tl")).toBe("fil");
    expect(toAzure("mn")).toBe("mn-Cyrl");
    expect(toAzure("ace")).toBeNull();
    expect(toIso6391("pt-BR")).toBe("pt");
    expect(toIso6391("yue")).toBeNull();
    expect(toIso6391("arz")).toBeNull();
    expect(languageInfo("jpn_Jpan")?.code).toBe("ja");
    expect(languageName("ja")).toBe("Japanese");
    expect(languageName("zh-Hant")).toBe("Chinese (Traditional)");
    expect(languageName("xx")).toBe("xx");
    expect(sameLanguage("zh-CN", "zh-Hans")).toBe(true);
    expect(sameLanguage("zh", "zh-TW")).toBe(false);
    expect(sameLanguage("", "")).toBe(false);
  });

  it("knows which languages are written without spaces", () => {
    expect(isCjk("ja-JP")).toBe(true);
    expect(isCjk("ko")).toBe(true);
    expect(isCjk("en")).toBe(false);
    expect(isUnspaced("ja")).toBe(true);
    expect(isUnspaced("th")).toBe(true);
    expect(isUnspaced("zh-TW")).toBe(true);
    expect(isUnspaced("ko")).toBe(false);
    expect(isUnspaced("en")).toBe(false);
  });
});

describe("guessLanguage", () => {
  it.each([
    ["今日は東京に行きます", "ja"],
    ["カタカナだけ", "ja"],
    ["我们明天去北京", "zh"],
    ["안녕하세요 여러분", "ko"],
    ["Привет, как дела?", "ru"],
    ["Привіт, як справи? Їжак", "uk"],
    ["مرحبا كيف حالك", "ar"],
    ["سلام، چطوری؟ خیلی خوبم", "fa"],
    ["یہ ایک جملہ ہے", "ur"],
    ["สวัสดีครับ", "th"],
    ["नमस्ते दोस्तों", "hi"],
    ["Γεια σου", "el"],
    ["שלום לכולם", "he"],
    ["Hello there", ""],
    ["Bonjour tout le monde", ""],
    ["12345 !!!", ""],
    ["", ""],
  ])("%s → %s", (text, expected) => {
    expect(guessLanguage(text)).toBe(expected);
  });
});
