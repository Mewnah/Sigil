import { describe, expect, it } from "vitest";
import {
  KEEP_SNOOZE_MS,
  LanguageStreak,
  MIN_SPEECH_WEIGHT,
  NO_DECISION,
  ONBOARDING_AUTO,
  STREAK_FINALS,
  decisionFor,
  isWeakSystemLanguage,
  languageDisplayName,
  lockOutcome,
  lockableCode,
  lockableLanguages,
  mayAskAgain,
  onboardingLanguageDefault,
  pickSystemLanguage,
  planDetectedOffer,
  planOnboardingSttSave,
  planSystemHint,
  speechWeight,
  systemLanguageTags,
  type DetectedFinal,
  type LanguageDecision,
} from "../languageSuggestion";
import { routedLanguages } from "../router";
import { Service_STT_Schema, STT_Backends } from "../schema";
import { WHISPER_LANGUAGES } from "../whisperModels";

const NOW = 1_800_000_000_000;
const FRENCH = "Je voudrais essayer ce nouveau micro ce soir";
const ENGLISH = "I would like to try this new microphone tonight";

let nextSegment = 0;
const final = (lang: string | null, text = FRENCH, extra: Partial<DetectedFinal> = {}): DetectedFinal => ({
  text,
  lang,
  segmentId: `local-x:1:${++nextSegment}`,
  ...extra,
});

function streakOf(...finals: DetectedFinal[]): LanguageStreak {
  const s = new LanguageStreak();
  for (const f of finals) s.note(f);
  return s;
}

describe("the documented rule", () => {
  it("is three counting finals in a row, each with about three words of speech, and a week of silence after Keep automatic", () => {
    // docs/OPERATOR_REFERENCE.md ("Set your language") and docs/manual/01-audio.md state these numbers.
    expect(STREAK_FINALS).toBe(3);
    expect(MIN_SPEECH_WEIGHT).toBe(12);
    expect(KEEP_SNOOZE_MS).toBe(7 * 24 * 60 * 60 * 1000);
  });
});

describe("speechWeight", () => {
  it("counts letters, and counts Chinese / Japanese / Korean characters twice", () => {
    expect(speechWeight("Hello there, 123!")).toBe(10);
    expect(speechWeight("")).toBe(0);
    expect(speechWeight("...")).toBe(0);
    expect(speechWeight("你好大家好")).toBe(10);
    expect(speechWeight("你好大家好吗")).toBe(MIN_SPEECH_WEIGHT);
    expect(speechWeight("안녕하세요")).toBe(10);
    expect(speechWeight("こんにちは")).toBe(10);
  });

  it("makes a one-word or two-word final too small and a short sentence big enough", () => {
    expect(speechWeight("okay thanks")).toBeLessThan(MIN_SPEECH_WEIGHT);
    expect(speechWeight("yeah")).toBeLessThan(MIN_SPEECH_WEIGHT);
    expect(speechWeight("thank you very much")).toBeGreaterThanOrEqual(MIN_SPEECH_WEIGHT);
    expect(speechWeight("ありがとうございます")).toBeGreaterThanOrEqual(MIN_SPEECH_WEIGHT);
  });
});

describe("LanguageStreak", () => {
  it("reaches the suggestion threshold after the same language on consecutive finals", () => {
    const s = new LanguageStreak();
    for (let i = 1; i <= STREAK_FINALS; i++) {
      expect(s.note(final("fr"))).toBe(true);
      expect(s.current).toEqual({ language: "fr", count: i });
    }
  });

  it("restarts the run on a counting final of another language", () => {
    const s = streakOf(final("fr"), final("fr"), final("en", ENGLISH));
    expect(s.current).toEqual({ language: "en", count: 1 });
    s.note(final("fr"));
    expect(s.current).toEqual({ language: "fr", count: 1 });
  });

  it("normalizes the language tags it is given", () => {
    const s = streakOf(final("fr-CA"), final("FR"), final("fra"));
    expect(s.current).toEqual({ language: "fr", count: 3 });
  });

  it("neither counts nor breaks on finals that are too short, without a language, or music", () => {
    const s = streakOf(final("fr"), final("fr"));
    expect(s.note(final("en", "yeah"))).toBe(false);
    expect(s.note(final(null))).toBe(false);
    expect(s.note(final("ja", FRENCH, { tags: ["event:bgm"] }))).toBe(false);
    expect(s.note(final("fr", FRENCH, { tags: ["emotion:happy"] }))).toBe(true);
    expect(s.current).toEqual({ language: "fr", count: 3 });
  });

  it("counts a segment once, however many finals report it", () => {
    const s = new LanguageStreak();
    const a = final("fr");
    expect(s.note(a)).toBe(true);
    expect(s.note({ ...a })).toBe(false);
    expect(s.current).toEqual({ language: "fr", count: 1 });
    // Finals without a segment id (engines without segments) are each a detection.
    const t = streakOf({ text: FRENCH, lang: "fr" }, { text: FRENCH, lang: "fr" });
    expect(t.current?.count).toBe(2);
  });

  it("starts from zero after reset", () => {
    const s = streakOf(final("fr"), final("fr"));
    s.reset();
    expect(s.current).toBeNull();
    s.note(final("fr"));
    expect(s.current).toEqual({ language: "fr", count: 1 });
  });
});

describe("the question's memory", () => {
  const at = (outcome: LanguageDecision["outcome"], ago = 0): LanguageDecision => ({ outcome, language: "fr", at: NOW - ago });

  it("is asked when never answered, a week after Keep automatic, and never after Lock or Don't ask again", () => {
    expect(mayAskAgain(undefined, NOW)).toBe(true);
    expect(mayAskAgain(NO_DECISION, NOW)).toBe(true);
    expect(mayAskAgain(at("kept", 1000), NOW)).toBe(false);
    expect(mayAskAgain(at("kept", KEEP_SNOOZE_MS - 1), NOW)).toBe(false);
    expect(mayAskAgain(at("kept", KEEP_SNOOZE_MS), NOW)).toBe(true);
    expect(mayAskAgain(at("locked", 10 * KEEP_SNOOZE_MS), NOW)).toBe(false);
    expect(mayAskAgain(at("never", 10 * KEEP_SNOOZE_MS), NOW)).toBe(false);
  });

  it("stores an answer with its language and time", () => {
    expect(decisionFor("locked", "fr", NOW)).toEqual({ outcome: "locked", language: "fr", at: NOW });
    expect(decisionFor("never", null, NOW)).toEqual({ outcome: "never", language: "", at: NOW });
  });

  it("defaults to unanswered in a new profile, and tolerates a corrupt value", () => {
    expect(Service_STT_Schema.parse({}).local.languageDecision).toEqual(NO_DECISION);
    expect(Service_STT_Schema.parse({ local: { languageDecision: "nonsense" } }).local.languageDecision).toEqual(NO_DECISION);
    expect(Service_STT_Schema.parse({ local: { languageDecision: { outcome: "kept", language: "fr", at: 5 } } }).local.languageDecision).toEqual({
      outcome: "kept",
      language: "fr",
      at: 5,
    });
    expect(Service_STT_Schema.parse({ local: { languageDecision: { outcome: "maybe" } } }).local.languageDecision.outcome).toBe("none");
  });
});

describe("the session suggestion", () => {
  const base = {
    backend: STT_Backends.local,
    language: "auto",
    decision: NO_DECISION as LanguageDecision,
    now: NOW,
    streak: { language: "fr", count: STREAK_FINALS },
    offeredThisSession: false,
  };

  it("offers the detected language once the streak is long enough", () => {
    expect(planDetectedOffer(base)).toEqual({ show: true, language: "fr", source: "detected" });
    expect(planDetectedOffer({ ...base, streak: { language: "fr", count: STREAK_FINALS - 1 } })).toEqual({ show: false, reason: "no_streak" });
    expect(planDetectedOffer({ ...base, streak: null })).toEqual({ show: false, reason: "no_streak" });
  });

  it("only ever asks a profile on the automatic backend and the automatic language", () => {
    expect(planDetectedOffer({ ...base, backend: STT_Backends.whisper })).toEqual({ show: false, reason: "not_local" });
    expect(planDetectedOffer({ ...base, language: "fr" })).toEqual({ show: false, reason: "language_set" });
    expect(planDetectedOffer({ ...base, language: "" }).show).toBe(true);
    expect(planDetectedOffer({ ...base, language: "AUTO" }).show).toBe(true);
  });

  it("respects the answer: never after Lock or Don't ask again, a week after Keep automatic, one offer per session", () => {
    expect(planDetectedOffer({ ...base, decision: decisionFor("never", null, NOW - 1) })).toEqual({ show: false, reason: "answered" });
    expect(planDetectedOffer({ ...base, decision: decisionFor("locked", "fr", NOW - 1) })).toEqual({ show: false, reason: "answered" });
    expect(planDetectedOffer({ ...base, decision: decisionFor("kept", "fr", NOW - 1000) })).toEqual({ show: false, reason: "snoozed" });
    expect(planDetectedOffer({ ...base, decision: decisionFor("kept", "fr", NOW - KEEP_SNOOZE_MS) }).show).toBe(true);
    expect(planDetectedOffer({ ...base, offeredThisSession: true })).toEqual({ show: false, reason: "already_offered" });
  });

  it("offers only a language the panel can be locked to", () => {
    expect(planDetectedOffer({ ...base, streak: { language: "xx", count: 9 } })).toEqual({ show: false, reason: "not_lockable" });
    expect(planDetectedOffer({ ...base, streak: { language: "ru", count: 9 } })).toMatchObject({ show: true, language: "ru" });
  });
});

describe("the hint for a profile that never answered (the Windows language)", () => {
  const base = { backend: STT_Backends.local, language: "auto", decision: NO_DECISION as LanguageDecision, now: NOW, system: "fr" as string | null };

  it("offers the Windows language only to a profile that never answered", () => {
    expect(planSystemHint(base)).toEqual({ show: true, language: "fr", source: "system" });
    for (const outcome of ["locked", "kept", "never"] as const)
      expect(planSystemHint({ ...base, decision: decisionFor(outcome, "fr", NOW - 10 * KEEP_SNOOZE_MS) })).toEqual({ show: false, reason: "answered" });
  });

  it("stays quiet without a routable Windows language, on another backend, or once a language is set", () => {
    expect(planSystemHint({ ...base, system: null })).toEqual({ show: false, reason: "no_system_language" });
    expect(planSystemHint({ ...base, system: "xx" })).toEqual({ show: false, reason: "no_system_language" });
    expect(planSystemHint({ ...base, backend: STT_Backends.vosk })).toEqual({ show: false, reason: "not_local" });
    expect(planSystemHint({ ...base, language: "de" })).toEqual({ show: false, reason: "language_set" });
  });

  it("never offers English as a guess: an English Windows says nothing about what the user speaks", () => {
    // A Brazilian or Korean gamer on English Windows must not get a one-click Lock to English; the session suggestion
    // (three English sentences in a row) is the evidence-based way to English.
    for (const system of ["en", "en-US", "en-GB"])
      expect(planSystemHint({ ...base, system })).toEqual({ show: false, reason: "no_system_language" });
  });
});

describe("languages", () => {
  it("lets a profile lock to every routed language and every Whisper language the router has another code for", () => {
    const lockable = lockableLanguages();
    for (const code of routedLanguages()) expect(lockable.has(code)).toBe(true);
    for (const code of WHISPER_LANGUAGES.filter((c) => c !== "no" && c !== "tl")) expect(lockable.has(code)).toBe(true);
    expect(lockable.has("no")).toBe(false);
    expect(lockable.has("tl")).toBe(false);
    for (const code of ["en", "fr", "ru", "ja", "ko", "zh", "de", "es", "pt", "nb", "fil"]) expect(lockable.has(code)).toBe(true);
    expect(lockable.has("auto")).toBe(false);
  });

  it("maps a tag to its lockable code", () => {
    expect(lockableCode("fr-CA")).toBe("fr");
    expect(lockableCode("zh-Hant-TW")).toBe("zh");
    expect(lockableCode("nb-NO")).toBe("nb");
    expect(lockableCode("no")).toBe("nb");
    expect(lockableCode("tl-PH")).toBe("fil");
    expect(lockableCode("auto")).toBeNull();
    expect(lockableCode("")).toBeNull();
    expect(lockableCode("xx")).toBeNull();
    expect(lockableCode(null)).toBeNull();
  });

  it("names a language in the UI language", () => {
    expect(languageDisplayName("fr", "en")).toBe("French");
    expect(languageDisplayName("fr", "fr")).toBe("français");
    expect(languageDisplayName("zz-not-a-code", "en")).toBeTruthy();
    expect(languageDisplayName("fr", "")).toBe("French");
  });
});

describe("the Windows language", () => {
  it("prefers the display language and falls back to an input language", () => {
    expect(pickSystemLanguage(["fr-FR", "en-US"])).toBe("fr");
    expect(pickSystemLanguage(["xx-XX", "de-DE", "en-US"])).toBe("de");
    expect(pickSystemLanguage(["xx-XX", "yy"])).toBeNull();
    expect(pickSystemLanguage([])).toBeNull();
    expect(pickSystemLanguage([null, undefined, "ja"])).toBe("ja");
  });

  it("treats English as no evidence of the spoken language, and looks past it", () => {
    expect(isWeakSystemLanguage("en")).toBe(true);
    expect(isWeakSystemLanguage("en-US")).toBe(true);
    expect(isWeakSystemLanguage("fr")).toBe(false);
    expect(isWeakSystemLanguage("")).toBe(false);
    expect(isWeakSystemLanguage(null)).toBe(false);
    // An English-only list gives nothing; English first with another routable language listed gives that language.
    expect(pickSystemLanguage(["en-US"])).toBeNull();
    expect(pickSystemLanguage(["en-GB", "en"])).toBeNull();
    expect(pickSystemLanguage(["en-US", "pt-BR"])).toBe("pt");
    expect(pickSystemLanguage(["en-US", "xx-XX", "ja-JP"])).toBe("ja");
  });

  it("reads the list the webview reports, display language first", () => {
    expect(systemLanguageTags({ language: "fr-FR", languages: ["fr-FR", "en-US"] })).toEqual(["fr-FR", "en-US"]);
    expect(systemLanguageTags({ language: "de-DE", languages: [] })).toEqual(["de-DE"]);
    expect(systemLanguageTags({ language: "", languages: ["ja"] })).toEqual(["ja"]);
    expect(systemLanguageTags(undefined)).toEqual(typeof navigator === "undefined" ? [] : systemLanguageTags(navigator));
  });
});

describe("onboarding: What language do you speak?", () => {
  it("starts on the Windows language when Sigil can route it", () => {
    expect(onboardingLanguageDefault(["fr-FR", "en-US"])).toBe("fr");
    expect(onboardingLanguageDefault(["ru-RU"])).toBe("ru");
    expect(onboardingLanguageDefault(["pt-BR"])).toBe("pt");
    expect(onboardingLanguageDefault(["xx-XX", "ko-KR"])).toBe("ko");
  });

  it("starts on Detect automatically when it cannot, and never invents a language", () => {
    expect(onboardingLanguageDefault(["xx-XX"])).toBe(ONBOARDING_AUTO);
    expect(onboardingLanguageDefault([])).toBe(ONBOARDING_AUTO);
  });

  it("starts on English when Windows is in English (the user's call; they confirm it with Continue)", () => {
    expect(onboardingLanguageDefault(["en-US"])).toBe("en");
    // The display language comes first, English included.
    expect(onboardingLanguageDefault(["en-US", "pt-BR"])).toBe("en");
    expect(onboardingLanguageDefault(["xx-XX"])).toBe(ONBOARDING_AUTO);
    expect(planOnboardingSttSave("local", "en", NOW)).toEqual({
      backend: STT_Backends.local,
      localLanguage: "en",
      decision: { outcome: "locked", language: "en", at: NOW },
    });
  });

  it("stores the chosen language as a locked answer", () => {
    expect(planOnboardingSttSave("local", "fr", NOW)).toEqual({
      backend: STT_Backends.local,
      localLanguage: "fr",
      decision: { outcome: "locked", language: "fr", at: NOW },
    });
    expect(planOnboardingSttSave("local", "fr-CA", NOW).localLanguage).toBe("fr");
  });

  it("keeps Detect automatically as an explicit choice (the session suggestion waits a week)", () => {
    const save = planOnboardingSttSave("local", ONBOARDING_AUTO, NOW);
    expect(save).toEqual({ backend: STT_Backends.local, localLanguage: "auto", decision: { outcome: "kept", language: "", at: NOW } });
    expect(planOnboardingSttSave("local", "not-a-language", NOW).localLanguage).toBe("auto");
  });

  it("does not ask the language question for the engines that have their own language settings", () => {
    expect(planOnboardingSttSave("whisper", "fr", NOW)).toEqual({ backend: STT_Backends.whisper });
    expect(planOnboardingSttSave("vosk", "fr", NOW)).toEqual({ backend: STT_Backends.vosk });
  });
});

describe("what a lock does", () => {
  it("restarts a running session only when the new route needs no download", () => {
    expect(lockOutcome(true, true)).toBe("restarted");
    expect(lockOutcome(true, false)).toBe("download");
    expect(lockOutcome(true, null)).toBe("next_start");
  });

  it("applies at the next Start otherwise, and says when that Start downloads", () => {
    expect(lockOutcome(false, true)).toBe("next_start");
    expect(lockOutcome(false, null)).toBe("next_start");
    expect(lockOutcome(false, false)).toBe("download");
  });
});
