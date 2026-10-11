import { proxy } from "valtio";
import { isAutoLanguage, normalizeLanguage } from "./router";
import { STT_Backends, type STT_State } from "./schema";
import {
  LanguageStreak,
  decisionFor,
  lockOutcome,
  lockableCode,
  planDetectedOffer,
  type DetectedFinal,
  type LanguageOfferSource,
  type LockOutcome,
} from "./languageSuggestion";
import type { SttTextMeta } from "./types";

/**
 * The in-session "Detected French. Lock captions to French?" suggestion (the rules are in languageSuggestion.ts).
 * Holds the state the UI shows (an offer, then the one line of feedback after a lock), counts the engine's finals,
 * and applies the user's answer: Lock (sets `stt.local.language`, restarts captions only when that is safe), Keep
 * automatic and Don't ask again. Every answer is persisted in `stt.local.languageDecision`.
 *
 * Nothing here ever sets a language by itself: `lock` runs on a click. Dependencies are injected so the logic runs
 * in tests without Tauri.
 */

/** How long the feedback line after a lock stays up. */
export const RESULT_VISIBLE_MS = 12_000;

/** How long a lock waits for the router before it stops asking and applies the language at the next Start. */
export const ROUTE_CHECK_TIMEOUT_MS = 5_000;

export interface LanguageOfferDeps {
  data(): Pick<STT_State, "backend" | "local">;
  patch(fn: (stt: STT_State) => void): void;
  now(): number;
  /** A session of the local backend is running (connected). */
  running(): boolean;
  /**
   * Whether the route for `language` needs no download (every model installed and an engine in the plan), asked of the
   * Rust router with the stored settings; `null` when that cannot be told.
   */
  routeReady(language: string): Promise<boolean | null>;
  /** Starts the session again with the stored settings; an automatic start, so it asks nothing. */
  restart(): void;
  /** A lock was applied (for the system log). */
  log(language: string, outcome: LockOutcome): void;
}

export interface LanguageOfferView {
  /** The language being offered (normalized), or null. */
  language: string | null;
  source: LanguageOfferSource | null;
  /** A lock is being applied. */
  busy: boolean;
  /** What the last lock did; cleared after {@link RESULT_VISIBLE_MS}. */
  result: { language: string; outcome: LockOutcome } | null;
}

export class LanguageOffer {
  readonly view: LanguageOfferView = proxy<LanguageOfferView>({ language: null, source: null, busy: false, result: null });

  readonly #deps: LanguageOfferDeps;
  readonly #streak = new LanguageStreak();
  /** One suggestion per session, whatever the answer. */
  #offered = false;
  #resultTimer: ReturnType<typeof setTimeout> | null = null;

  constructor(deps: LanguageOfferDeps) {
    this.#deps = deps;
  }

  /**
   * A new session starts: its detections are counted from zero and it may make one suggestion. An offer nobody
   * answered was made from the last session's speech (a guest, a stream someone else read aloud): it is taken down
   * without an answer, so a click on it can never lock a language nobody has spoken since.
   */
  sessionStarted(): void {
    this.#streak.reset();
    this.#offered = false;
    this.#withdraw();
    this.settingsChanged();
  }

  /**
   * One final of the engine of the current session. While an offer is up, a counting final in another language takes
   * it down (still no answer is stored): the evidence it rests on has ended. The session does not offer again; the
   * next one counts from zero.
   */
  noteFinal(text: string, meta?: SttTextMeta): void {
    const { backend, local } = this.#deps.data();
    if (backend !== STT_Backends.local || !isAutoLanguage(local.language)) return;
    const final: DetectedFinal = { text, lang: meta?.lang, segmentId: meta?.segmentId, tags: meta?.tags };
    if (!this.#streak.note(final)) return;
    if (this.view.language !== null) {
      const heard = this.#streak.current;
      if (!heard || lockableCode(heard.language) !== this.view.language) this.#withdraw();
      return;
    }
    if (this.#offered) return;
    const plan = planDetectedOffer({
      backend,
      language: local.language,
      decision: local.languageDecision,
      now: this.#deps.now(),
      streak: this.#streak.current,
      offeredThisSession: this.#offered,
    });
    if (!plan.show) return;
    this.#offered = true;
    this.view.language = plan.language;
    this.view.source = plan.source;
  }

  /** The language or backend settings changed: an offer for a profile that is no longer on auto is withdrawn. */
  settingsChanged(): void {
    const { backend, local } = this.#deps.data();
    if (this.view.language && (backend !== STT_Backends.local || !isAutoLanguage(local.language))) this.#withdraw();
  }

  /**
   * Lock captions to `language` (the user pressed Lock): stores the language and the answer. A running session
   * restarts at once only when the router says every model of the new route is installed (a restart must never turn
   * into a download or a question while the user is live); otherwise the language applies at the next Start, which
   * asks before it downloads anything. Returns what happened (the UI says it), or null when nothing was locked.
   */
  async lock(language: string): Promise<LockOutcome | null> {
    const code = lockableCode(language) ?? normalizeLanguage(language);
    if (!code || this.view.busy) return null;
    this.view.busy = true;
    try {
      const ready = await this.#routeReady(code);
      const now = this.#deps.now();
      const outcome = lockOutcome(this.#deps.running(), ready);
      // The feedback line goes up before the offer comes down and the language changes, so a toast showing the
      // offer turns into the feedback line without ever being taken down in between.
      this.#showResult(code, outcome);
      this.#withdraw();
      this.#deps.patch((stt) => {
        stt.local.language = code;
        stt.local.languageDecision = decisionFor("locked", code, now);
      });
      this.#deps.log(code, outcome);
      if (outcome === "restarted") this.#deps.restart();
      return outcome;
    } finally {
      this.view.busy = false;
    }
  }

  /** "Keep automatic": automatic detection stays; the suggestion is silent for a week. */
  keepAutomatic(language?: string | null): void {
    this.#answer("kept", language);
  }

  /** "Don't ask again": automatic detection stays and the question is never asked again. */
  neverAsk(language?: string | null): void {
    this.#answer("never", language);
  }

  /** The feedback line was closed. */
  dismissResult(): void {
    this.#clearResult();
  }

  dispose(): void {
    this.#clearResultTimer();
  }

  /** The router's answer, or null (cannot be told) when it fails or takes too long. */
  async #routeReady(code: string): Promise<boolean | null> {
    let timer: ReturnType<typeof setTimeout> | undefined;
    try {
      return await Promise.race([
        this.#deps.routeReady(code),
        new Promise<null>((resolve) => {
          timer = setTimeout(() => resolve(null), ROUTE_CHECK_TIMEOUT_MS);
        }),
      ]);
    } catch {
      return null;
    } finally {
      clearTimeout(timer);
    }
  }

  #answer(outcome: "kept" | "never", language?: string | null): void {
    const code = lockableCode(language ?? this.view.language) ?? null;
    const now = this.#deps.now();
    this.#deps.patch((stt) => {
      stt.local.languageDecision = decisionFor(outcome, code, now);
    });
    this.#withdraw();
  }

  #withdraw(): void {
    this.view.language = null;
    this.view.source = null;
  }

  #showResult(language: string, outcome: LockOutcome): void {
    this.#clearResultTimer();
    this.view.result = { language, outcome };
    this.#resultTimer = setTimeout(() => {
      this.#resultTimer = null;
      this.view.result = null;
    }, RESULT_VISIBLE_MS);
  }

  #clearResult(): void {
    this.#clearResultTimer();
    this.view.result = null;
  }

  #clearResultTimer(): void {
    if (this.#resultTimer !== null) {
      clearTimeout(this.#resultTimer);
      this.#resultTimer = null;
    }
  }
}
