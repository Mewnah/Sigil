import { isAutoLanguage, normalizeLanguage, routedLanguages } from "./router";
import { STT_Backends } from "./schema";
import { WHISPER_LANGUAGES } from "./whisperModels";

/**
 * "Lock captions to your language?" (pure; no `window`, no Tauri, no UI).
 *
 * Why it exists: the P7 screening (docs/benchmarks/SPEECH_BENCHMARKS.md, "Auto mode") found that a new install, which
 * runs the `local` backend on language `auto`, scores 1.4 to 3.6 times worse than a set language on the European
 * languages (ru 22.0 against 6.1 % WER, fr 17.3 against 8.8, es 13.1 against 6.1, de 11.5 against 6.7, pt 5.5 against 3.9) and holds about
 * 1 GB more memory (2756 against 1765 MB), while its language ID named the right language for 105 of 108 utterances.
 * Detection is therefore good enough to ask with, and never good enough to decide with: a language is only ever set
 * by the user's own answer. Three sources of that answer, all of them visible and all of them keeping "Detect
 * automatically" one click away:
 * 1. onboarding: the language question of the STT step, preselected from the Windows language;
 * 2. the STT panel: the same hint for profiles that never answered (the wizard is optional);
 * 3. the session: after {@link STREAK_FINALS} consistent detections in a row, a non-modal suggestion.
 *
 * What the pipeline reports (and what it does not): every final carries `lang` (the language the segment was decoded
 * in: the language-ID result of that utterance, or the engine's own detection). It reports no language probability
 * and no speech duration to the UI. So "confident" is built from what can be measured here: agreement across
 * consecutive utterances, and enough speech in each (a one-word final is a poor sample for language ID).
 */

// ---------------------------------------------------------------------------------------------------------------
// Constants

/** Consecutive counting finals in the same language that make a suggestion. */
export const STREAK_FINALS = 3;

/**
 * A final counts when {@link speechWeight} of its text reaches this: 12 letters (about three words), or 6 Chinese /
 * Japanese / Korean characters. Shorter finals ("yeah", "okay thanks") are neither counted nor do they break a
 * streak: language ID on a fraction of a second is unreliable either way.
 */
export const MIN_SPEECH_WEIGHT = 12;

/** "Keep automatic" silences the session suggestion for this long (a week); "Don't ask again" is forever. */
export const KEEP_SNOOZE_MS = 7 * 24 * 60 * 60 * 1000;

/** Audio event tags that mark a final as not being the user's plain speech (music: its language says nothing). */
const IGNORED_TAGS: ReadonlySet<string> = new Set(["event:bgm"]);

// ---------------------------------------------------------------------------------------------------------------
// Languages

/** Router codes standing for a code the Whisper list or a Windows tag spells differently. */
const LOCK_ALIASES: Readonly<Record<string, string>> = { no: "nb", nn: "nb", tl: "fil" };

/**
 * Languages a profile can be locked to: every language the routing table has a row for, plus the ones Whisper handles
 * (the same set the STT panel's language select offers). Normalized codes (`nb`, `fil`), no `auto`.
 */
export function lockableLanguages(): Set<string> {
  const codes = new Set(routedLanguages());
  for (const code of WHISPER_LANGUAGES) if (!(code in LOCK_ALIASES)) codes.add(code);
  return codes;
}

/** A language tag as a lockable code (`fr-CA` → `fr`, `nb-NO` → `nb`), or null when the tag is not lockable. */
export function lockableCode(tag: string | null | undefined, lockable: ReadonlySet<string> = lockableLanguages()): string | null {
  const n = normalizeLanguage(tag);
  if (!n) return null;
  const code = LOCK_ALIASES[n] ?? n;
  return lockable.has(code) ? code : null;
}

/**
 * Windows languages that say nothing about what the user speaks. English is the display language of a very large share
 * of PCs whose owner speaks something else (a Brazilian or Korean gamer on English Windows is the common case), so
 * "Your Windows language is English" must never become a one-click Lock on a guess. English is found by listening
 * instead ({@link planDetectedOffer}: three English sentences in a row), which is evidence rather than a default.
 */
const WEAK_SYSTEM_LANGUAGES: ReadonlySet<string> = new Set(["en"]);

/** Whether the Windows language `code` is too common a display language to count as the language spoken. */
export function isWeakSystemLanguage(code: string | null | undefined): boolean {
  const n = normalizeLanguage(code);
  return n !== null && WEAK_SYSTEM_LANGUAGES.has(n);
}

/**
 * The first lockable, non-English language of `tags`: a weak prior for the language the user speaks, offered as a
 * question (never applied by itself). Pass the Windows list in its own order: the display language comes first, then
 * whatever other preferred languages the webview lists, so a non-English display language wins and a listed second
 * language (say Japanese after English) is the fallback. `null` when the list holds no language Sigil can route other
 * than English, which includes an English-only PC.
 */
export function pickSystemLanguage(tags: readonly (string | null | undefined)[], lockable: ReadonlySet<string> = lockableLanguages()): string | null {
  for (const tag of tags) {
    const code = lockableCode(tag, lockable);
    if (code && !isWeakSystemLanguage(code)) return code;
  }
  return null;
}

/**
 * The Windows language list as the webview reports it (`navigator.languages`: the display language first; the
 * webview lists the other preferred languages after it only when Windows exposes them to it, and the keyboard layouts
 * are not read). Empty when the environment exposes none.
 */
export function systemLanguageTags(nav: Pick<Navigator, "language" | "languages"> | undefined = typeof navigator === "undefined" ? undefined : navigator): string[] {
  if (!nav) return [];
  const list = Array.isArray(nav.languages) ? [...nav.languages] : [];
  if (nav.language && !list.includes(nav.language)) list.unshift(nav.language);
  return list.filter((t): t is string => typeof t === "string" && t !== "");
}

/** The Windows language of this PC when it is lockable. */
export function systemLanguage(): string | null {
  return pickSystemLanguage(systemLanguageTags());
}

/** Name of a language code in the UI language `locale` (`Intl.DisplayNames`); the code itself when it has none. */
export function languageDisplayName(code: string, locale = "en"): string {
  try {
    return new Intl.DisplayNames([locale.replace("_", "-") || "en"], { type: "language" }).of(code) ?? code;
  } catch {
    return code;
  }
}

// ---------------------------------------------------------------------------------------------------------------
// The answer that is kept

export type LanguageDecisionOutcome = "none" | "locked" | "kept" | "never";

/** `stt.local.languageDecision`. */
export interface LanguageDecision {
  outcome: LanguageDecisionOutcome;
  language: string;
  at: number;
}

export const NO_DECISION: Readonly<LanguageDecision> = Object.freeze({ outcome: "none", language: "", at: 0 });

/** The decision to store for an answer given at `now`. */
export function decisionFor(outcome: Exclude<LanguageDecisionOutcome, "none">, language: string | null, now: number): LanguageDecision {
  return { outcome, language: language ?? "", at: now };
}

/** Whether the question may be asked again after `decision`: never after `locked` / `never`; a week after `kept`. */
export function mayAskAgain(decision: LanguageDecision | null | undefined, now: number): boolean {
  switch (decision?.outcome ?? "none") {
    case "none":
      return true;
    case "kept":
      return now - (decision?.at ?? 0) >= KEEP_SNOOZE_MS;
    default:
      return false;
  }
}

// ---------------------------------------------------------------------------------------------------------------
// Counting detections

/**
 * How much speech a final holds: its letters, with Chinese / Japanese / Korean characters counted twice (one such
 * character carries about as much as two letters). Punctuation, digits and spaces do not count.
 */
export function speechWeight(text: string): number {
  const letters = text.match(/\p{L}/gu)?.length ?? 0;
  const dense = text.match(/[\p{Script=Han}\p{Script=Hiragana}\p{Script=Katakana}\p{Script=Hangul}]/gu)?.length ?? 0;
  return letters + dense;
}

/** One final as the streak sees it (`SttTextMeta` of the engine receiver, plus the text). */
export interface DetectedFinal {
  text: string;
  /** `TextEvent.lang` of the final. */
  lang?: string | null;
  /** Published segment id: a second final of the same segment is not a second detection. */
  segmentId?: string | null;
  tags?: readonly string[] | null;
}

export interface LanguageStreakState {
  language: string;
  /** Consecutive counting finals in this language. */
  count: number;
}

/**
 * The run of consecutive finals in one language. A final in another language starts a new run; finals without a
 * language, with too little speech, tagged as music, or repeating the previous segment change nothing.
 */
export class LanguageStreak {
  #language: string | null = null;
  #count = 0;
  #lastSegment: string | null = null;

  /** Feeds one final; true when it counted as a detection. */
  note(final: DetectedFinal): boolean {
    const lang = normalizeLanguage(final.lang);
    if (!lang) return false;
    if (final.tags?.some((t) => IGNORED_TAGS.has(t))) return false;
    if (speechWeight(final.text) < MIN_SPEECH_WEIGHT) return false;
    const segment = final.segmentId ?? null;
    if (segment !== null && segment === this.#lastSegment) return false;
    this.#lastSegment = segment;
    if (lang === this.#language) this.#count++;
    else {
      this.#language = lang;
      this.#count = 1;
    }
    return true;
  }

  get current(): LanguageStreakState | null {
    return this.#language === null ? null : { language: this.#language, count: this.#count };
  }

  reset(): void {
    this.#language = null;
    this.#count = 0;
    this.#lastSegment = null;
  }
}

// ---------------------------------------------------------------------------------------------------------------
// Deciding whether to ask

/** What the question is based on: what the session heard, or the Windows language. */
export type LanguageOfferSource = "detected" | "system";

export type OfferRefusal =
  | "not_local"
  | "language_set"
  | "answered"
  | "snoozed"
  | "already_offered"
  | "no_streak"
  | "not_lockable"
  | "no_system_language";

export type OfferPlan = { show: true; language: string; source: LanguageOfferSource } | { show: false; reason: OfferRefusal };

export interface OfferInput {
  backend: string;
  /** `stt.local.language`. */
  language: string;
  decision: LanguageDecision | null | undefined;
  now: number;
  lockable?: ReadonlySet<string>;
}

/** Whether a profile can be asked at all: the automatic backend, on a language that is still automatic. */
export function isAskable(backend: string, language: string): boolean {
  return backend === STT_Backends.local && isAutoLanguage(language);
}

function baseRefusal(input: OfferInput): OfferRefusal | null {
  if (input.backend !== STT_Backends.local) return "not_local";
  if (!isAutoLanguage(input.language)) return "language_set";
  return null;
}

/**
 * The session suggestion: ask when the local backend runs on `auto`, the question was not answered for good (or its
 * week of silence is over), nothing was offered this session yet, and the session heard {@link STREAK_FINALS}
 * counting finals in one lockable language in a row.
 */
export function planDetectedOffer(input: OfferInput & { streak: LanguageStreakState | null; offeredThisSession: boolean }): OfferPlan {
  const refusal = baseRefusal(input);
  if (refusal) return { show: false, reason: refusal };
  if (!mayAskAgain(input.decision, input.now))
    return { show: false, reason: input.decision?.outcome === "kept" ? "snoozed" : "answered" };
  if (input.offeredThisSession) return { show: false, reason: "already_offered" };
  const { streak } = input;
  if (!streak || streak.count < STREAK_FINALS) return { show: false, reason: "no_streak" };
  const language = lockableCode(streak.language, input.lockable);
  return language ? { show: true, language, source: "detected" } : { show: false, reason: "not_lockable" };
}

/**
 * The hint for a profile that never answered: the Windows language, offered in the STT panel only (never as a toast,
 * never after any answer). `system` is the language {@link systemLanguage} found. A weak one (English: the display
 * language of PCs whose owner speaks something else) is not offered; the session suggestion covers English with evidence.
 */
export function planSystemHint(input: OfferInput & { system: string | null }): OfferPlan {
  const refusal = baseRefusal(input);
  if (refusal) return { show: false, reason: refusal };
  if ((input.decision?.outcome ?? "none") !== "none") return { show: false, reason: "answered" };
  const language = lockableCode(input.system, input.lockable);
  return language && !isWeakSystemLanguage(language) ? { show: true, language, source: "system" } : { show: false, reason: "no_system_language" };
}

// ---------------------------------------------------------------------------------------------------------------
// Onboarding

/** The wizard's choice for "What language do you speak?": a lockable code, or `auto`. */
export const ONBOARDING_AUTO = "auto";

/**
 * What the language question starts on: the Windows language when Sigil can route it and it says something about the
 * speaker (not English), else "Detect automatically". The user sees the choice and confirms it with Continue.
 */
export function onboardingLanguageDefault(tags: readonly (string | null | undefined)[], lockable?: ReadonlySet<string>): string {
  return pickSystemLanguage(tags, lockable) ?? ONBOARDING_AUTO;
}

export type OnboardingSttChoice = "local" | "whisper" | "vosk";

export interface OnboardingSttSave {
  backend: STT_Backends;
  /** `stt.local.language` to store; absent: leave it (the language question only applies to the automatic engine). */
  localLanguage?: string;
  /** `stt.local.languageDecision` to store; absent: leave it. */
  decision?: LanguageDecision;
}

/**
 * What the STT step of the wizard stores. A chosen language is stored as locked, an explicit "Detect automatically"
 * as kept (the suggestion then waits a week); the question is not asked for Whisper and Vosk, which have their own
 * language settings.
 */
export function planOnboardingSttSave(choice: OnboardingSttChoice, language: string, now: number): OnboardingSttSave {
  if (choice === "vosk") return { backend: STT_Backends.vosk };
  if (choice === "whisper") return { backend: STT_Backends.whisper };
  const code = lockableCode(language);
  if (!code) return { backend: STT_Backends.local, localLanguage: "auto", decision: decisionFor("kept", null, now) };
  return { backend: STT_Backends.local, localLanguage: code, decision: decisionFor("locked", code, now) };
}

// ---------------------------------------------------------------------------------------------------------------
// Locking

/**
 * What a lock did, for the one line of feedback:
 * - `restarted`: captions were running and every model of the new route is installed, so they restarted at once;
 * - `next_start`: it applies the next time Start is pressed;
 * - `download`: the new route needs models that are not installed yet, so the next Start downloads them first (and
 *   asks, like any Start). A running session is never restarted into a download.
 */
export type LockOutcome = "restarted" | "next_start" | "download";

/** The outcome of a lock from what is known: is a session running, and are the new route's models all installed. */
export function lockOutcome(running: boolean, routeReady: boolean | null): LockOutcome {
  if (routeReady === false) return "download";
  return running && routeReady === true ? "restarted" : "next_start";
}
