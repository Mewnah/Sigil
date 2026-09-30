import i18n from "i18next";
import { raceWithSignal } from "./abort";
import { canonicalLanguage, sameLanguage } from "./languages";
import { Translation_Backends, Translation_State, TranslationEngineId } from "./schema";
import {
  ITranslationEngine,
  TranslationEngineHost,
  TranslationError,
  TranslationKind,
  TranslationRequest,
  TranslationStatusUpdate,
} from "./types";

/**
 * Router engine (`auto` backend, research §3.2 "language-pair router with a fallback chain"). For each line it tries
 * the pair's chain (`auto.pairOverrides`, first match, else `auto.chain`) in order:
 *
 * - Members are the other engines, created from the host's constructors. `start` starts every member of the
 *   configured pair's chain concurrently and resolves as soon as one is ready (the rest keep starting; members of
 *   other pairs' chains start on first use). A member whose start failed is retried after `ROUTER_START_RETRY_MS`.
 * - Every attempt is bounded (a hung engine is aborted and the next one tried), so the chain falls through within
 *   one budget per member: `auto.attemptTimeoutMs` for caption-length text, proportionally more for a long final
 *   (`routerAttemptBudgetMs`). Waiting for a member that is still starting is bounded by `attemptTimeoutMs` too, and
 *   is never charged to its breaker (a slow model load is not a failing engine).
 * - A member still starting is skipped while a later member is ready (early lines use the fallback instead of
 *   waiting for a model to load); members that answer `supports(from, to) === false` or whose slice has
 *   `interim: false` (interims only) are skipped.
 * - Circuit breakers per member and pair, separate for interims and finals (an interim that times out in a busy
 *   server's queue must not lock the engine out of finals): `BREAKER_THRESHOLD` failures in a row (or one
 *   non-retryable failure) open one for 15 s, doubling up to 5 min; after the cooldown one probe request decides.
 *   An attempt timeout is an ordinary (retryable) failure. Interims also skip a member whose final breaker is open.
 * - Finals are never dropped just because breakers are open or a member is still starting: members passed over for
 *   that reason are tried after the rest of the chain failed (a forced probe) before the line is given up.
 * - `auto.cloudFallback`: when a cloud engine (DeepL, Azure, a cloud LLM) failed, the local engines that are not in
 *   the chain (Bergamot, NLLB) are tried after it.
 * - Interims: `auto.interimEngine` (`auto` = the cheapest ready local engine, Bergamot first), then the chain; at
 *   most two attempts, since the next partial supersedes them anyway.
 * - The member that produced a line is reported through `request.onEngine`; a final produced by a fallback shows
 *   a warning status until the chain's first engine answers again.
 *
 * Owned by P4-C.
 */

/** A member whose start failed is started again after this long (e.g. once the user installed its model). */
export const ROUTER_START_RETRY_MS = 60_000;
export const BREAKER_THRESHOLD = 2;
export const BREAKER_BASE_COOLDOWN_MS = 15_000;
export const BREAKER_MAX_COOLDOWN_MS = 300_000;
/** Attempts per interim (the next partial supersedes it anyway). */
export const INTERIM_MAX_ATTEMPTS = 2;
/** Text length (characters) `attemptTimeoutMs` is meant for; a longer final gets proportionally more time. */
export const ROUTER_CAPTION_CHARS = 120;
/** Longest budget of one final attempt (the schema's `attemptTimeoutMs` maximum). */
export const ROUTER_MAX_ATTEMPT_MS = 60_000;

/** Local engines tried after a cloud failure when they are not in the chain. */
export const LOCAL_FALLBACKS: readonly TranslationEngineId[] = [Translation_Backends.bergamot, Translation_Backends.nllb];
/** Interim engines for `interimEngine: "auto"`, cheapest first. */
const INTERIM_COST_ORDER: readonly TranslationEngineId[] = [
  Translation_Backends.bergamot,
  Translation_Backends.onDevice,
  Translation_Backends.nllb,
];

/** Engines that send text to a cloud service (their failures trigger `cloudFallback`). */
export function isCloudEngine(id: TranslationEngineId, state: Pick<Translation_State, "llm">): boolean {
  switch (id) {
    case Translation_Backends.deepl:
    case Translation_Backends.azure:
      return true;
    case Translation_Backends.llm:
      return state.llm.provider !== "bundled" && state.llm.provider !== "custom";
    default:
      return false;
  }
}

/** The chain for a pair: the first matching `pairOverrides` entry (blank side = any), else `chain`. */
export function chainForPair(auto: Translation_State["auto"], from: string, to: string): TranslationEngineId[] {
  for (const o of auto.pairOverrides) {
    const fromOk = !o.from.trim() || (from !== "" && sameLanguage(o.from, from));
    const toOk = !o.to.trim() || sameLanguage(o.to, to);
    if (fromOk && toOk && o.chain.length) return [...o.chain];
  }
  return [...auto.chain];
}

/** i18n name of an engine for messages. */
export function engineLabel(id: Translation_Backends | TranslationEngineId): string {
  const key = `transl.backend_${id === Translation_Backends.onDevice ? "ondevice" : id}`;
  return i18n.t(key) || id;
}

/**
 * Budget of one attempt: `attemptTimeoutMs` for interims and caption-length finals; a final longer than
 * `ROUTER_CAPTION_CHARS` gets proportionally more (an LLM's generation time grows with the text), up to
 * `ROUTER_MAX_ATTEMPT_MS`.
 */
export function routerAttemptBudgetMs(attemptTimeoutMs: number, kind: TranslationKind, text: string): number {
  if (kind === "interim") return attemptTimeoutMs;
  const scaled = Math.round((attemptTimeoutMs * [...text].length) / ROUTER_CAPTION_CHARS);
  return Math.max(attemptTimeoutMs, Math.min(ROUTER_MAX_ATTEMPT_MS, scaled));
}

/**
 * Longest time the router can spend on one line: every candidate may wait `attemptTimeoutMs` for its start and then
 * use its attempt budget (the pipeline's sequencer deadline must not cut the chain short).
 */
export function routerLineBudgetMs(auto: Translation_State["auto"], kind: TranslationKind, text: string): number {
  const chains = [auto.chain, ...auto.pairOverrides.map((o) => o.chain)];
  const longest = Math.max(1, ...chains.map((c) => c.length));
  const members = kind === "interim" ? INTERIM_MAX_ATTEMPTS : longest + (auto.cloudFallback ? LOCAL_FALLBACKS.length : 0);
  return members * (auto.attemptTimeoutMs + routerAttemptBudgetMs(auto.attemptTimeoutMs, kind, text));
}

/** Rejection reason of an attempt that exceeded its budget (`phase: "start"`: the member was still starting). */
export class RouterAttemptTimeoutError extends Error {
  constructor(
    readonly engine: TranslationEngineId,
    readonly timeoutMs: number,
    readonly phase: "start" | "translate" = "translate",
  ) {
    super(i18n.t("transl.error_timeout", { seconds: Math.round(timeoutMs / 100) / 10 }) || `timed out after ${timeoutMs} ms`);
    this.name = "RouterAttemptTimeoutError";
  }
}

/** Consecutive-failure circuit breaker with a doubling cooldown and a single half-open probe. */
export class CircuitBreaker {
  #failures = 0;
  #openUntil = 0;
  #cooldown: number;
  #probing = false;

  constructor(
    private readonly threshold = BREAKER_THRESHOLD,
    private readonly baseCooldownMs = BREAKER_BASE_COOLDOWN_MS,
    private readonly maxCooldownMs = BREAKER_MAX_COOLDOWN_MS,
  ) {
    this.#cooldown = baseCooldownMs;
  }

  get state(): "closed" | "open" | "half-open" {
    if (this.#openUntil === 0) return "closed";
    return this.#probing ? "half-open" : "open";
  }

  /** Whether requests are held back now (open and cooling down, or a probe is out); no side effects. */
  blocked(now: number): boolean {
    return this.#openUntil !== 0 && (now < this.#openUntil || this.#probing);
  }

  /** Whether a request may go out now (after the cooldown, one probe at a time). */
  allow(now: number): boolean {
    if (this.#openUntil === 0) return true;
    if (now < this.#openUntil || this.#probing) return false;
    this.#probing = true;
    return true;
  }

  success(): void {
    this.#failures = 0;
    this.#openUntil = 0;
    this.#probing = false;
    this.#cooldown = this.baseCooldownMs;
  }

  /** A failed request; `severe` (not worth repeating: bad key, unsupported pair) opens the breaker at once. */
  failure(now: number, severe = false): void {
    const wasOpen = this.#openUntil !== 0;
    this.#probing = false;
    this.#failures++;
    if (severe || wasOpen || this.#failures >= this.threshold) {
      this.#openUntil = now + this.#cooldown;
      this.#cooldown = Math.min(this.maxCooldownMs, this.#cooldown * 2);
    }
  }

  /** The request was cancelled by the caller: neither success nor failure. */
  release(): void {
    this.#probing = false;
  }
}

interface Member {
  readonly id: TranslationEngineId;
  engine: ITranslationEngine | null;
  starting: Promise<void> | null;
  ready: boolean;
  startError: unknown;
  startFailedAt: number;
  lifetime: AbortController;
}

function messageOf(e: unknown): string {
  const text = e instanceof Error ? e.message : e === undefined || e === null ? "" : String(e);
  const trimmed = text.trim() || i18n.t("common.error_no_details") || "no details";
  return trimmed.length > 200 ? `${trimmed.slice(0, 197)}…` : trimmed;
}

function describe(failures: readonly { id: TranslationEngineId; error: unknown }[]): string {
  return failures.map((f) => `${engineLabel(f.id)}: ${messageOf(f.error)}`).join("; ");
}

/** Whether a failure is worth repeating; an attempt timeout is (a busy server), so it only counts toward the threshold. */
function isRetryable(e: unknown): boolean {
  if (e instanceof TranslationError) return e.retryable;
  return true;
}

function breakerKey(kind: TranslationKind, id: TranslationEngineId, from: string, to: string): string {
  return `${kind}|${id}|${from}|${to}`;
}

type AttemptResult = { ok: true; text: string } | { ok: false; error: unknown };

function missingOf(errors: readonly unknown[]): string[] {
  const out: string[] = [];
  for (const e of errors) {
    if (e instanceof TranslationError) for (const id of e.missing) if (!out.includes(id)) out.push(id);
  }
  return out;
}

export interface RouterOptions {
  now?: () => number;
}

export class Translation_RouterService implements ITranslationEngine {
  #host: TranslationEngineHost | null = null;
  #state: Translation_State | null = null;
  #members = new Map<TranslationEngineId, Member>();
  #breakers = new Map<string, CircuitBreaker>();
  /** Fallback warning currently shown ("" = none). */
  #notice = "";
  readonly #now: () => number;

  constructor(options: RouterOptions = {}) {
    this.#now = options.now ?? (() => Date.now());
  }

  attach(host: TranslationEngineHost): void {
    this.#host = host;
  }

  async start(state: Translation_State, signal: AbortSignal): Promise<void> {
    if (!this.#host) {
      throw new TranslationError(i18n.t("transl.error_backend_unavailable"), { retryable: false });
    }
    this.#state = state;
    const to = canonicalLanguage(state.targetLanguage);
    if (!to) throw new TranslationError(i18n.t("transl.error_target_missing"), { retryable: false });
    const from = canonicalLanguage(state.sourceLanguage);
    const chain = chainForPair(state.auto, from, to);
    const extra = state.auto.interimEngine !== "auto" ? [state.auto.interimEngine] : [];

    const failures: { id: TranslationEngineId; error: unknown }[] = [];
    await new Promise<void>((resolve, reject) => {
      let pending = chain.length;
      const onAbort = () => reject(signal.reason ?? new DOMException("Aborted", "AbortError"));
      if (signal.aborted) {
        onAbort();
        return;
      }
      signal.addEventListener("abort", onAbort, { once: true });
      const settleFailure = (id: TranslationEngineId, error: unknown) => {
        failures.push({ id, error });
        if (--pending === 0) {
          signal.removeEventListener("abort", onAbort);
          reject(
            new TranslationError(i18n.t("transl.router_error_none_started", { errors: describe(failures) }), {
              retryable: false,
              missing: missingOf(failures.map((f) => f.error)),
            }),
          );
        }
      };
      for (const id of chain) {
        this.#ensureStarted(this.#member(id)).then(
          () => {
            signal.removeEventListener("abort", onAbort);
            resolve();
          },
          (error: unknown) => settleFailure(id, error),
        );
      }
      for (const id of extra) {
        if (!chain.includes(id)) this.#ensureStarted(this.#member(id)).catch(() => {});
      }
    });
  }

  async translate(request: TranslationRequest, signal: AbortSignal): Promise<string> {
    const state = this.#state;
    if (!state || !this.#host) throw new TranslationError(i18n.t("transl.error_not_running"), { retryable: false });
    const to = canonicalLanguage(request.to ?? state.targetLanguage);
    const from = canonicalLanguage(request.from ?? state.sourceLanguage);
    if (!to) throw new TranslationError(i18n.t("transl.error_target_missing"), { retryable: false });

    const kind = request.kind;
    const chain = chainForPair(state.auto, from, to);
    const candidates = this.#candidates(kind, chain, from, to);
    const startWaitMs = state.auto.attemptTimeoutMs;
    const budgetMs = routerAttemptBudgetMs(state.auto.attemptTimeoutMs, kind, request.text);
    const memberRequest: TranslationRequest = { ...request, from, to, onEngine: undefined };
    const failures: { id: TranslationEngineId; error: unknown }[] = [];
    let cloudFailed = false;
    let attempts = 0;
    /** First chain member that can serve this pair: the one a fallback warning names. */
    let preferred: TranslationEngineId | undefined;
    /** Final candidates passed over only for an open breaker or a slow start: tried last instead of dropping the line. */
    const deferred: TranslationEngineId[] = [];

    for (let i = 0; i < candidates.length; i++) {
      const { id, afterCloudFailure } = candidates[i];
      if (afterCloudFailure && !cloudFailed) continue;
      if (kind === "interim" && attempts >= INTERIM_MAX_ATTEMPTS) break;
      const m = this.#member(id);
      if (!this.#usable(m, from, to)) continue;
      if (!afterCloudFailure) preferred ??= id;
      // Do not wait for an engine that is still loading while a later one can answer now.
      if (!m.ready && m.starting && candidates.slice(i + 1).some((c) => this.#readyFor(kind, c.id, from, to, c.afterCloudFailure))) {
        if (kind === "final") deferred.push(id);
        continue;
      }
      const breaker = this.#breaker(kind, id, from, to);
      if (this.#blocked(kind, id, from, to) || !breaker.allow(this.#now())) {
        if (kind === "final") deferred.push(id);
        // A cloud engine whose breaker is open has been failing: its local fallbacks apply.
        if (isCloudEngine(id, state)) cloudFailed = true;
        continue;
      }

      attempts++;
      const result = await this.#try(m, breaker, memberRequest, signal, startWaitMs, budgetMs);
      if (result.ok) return this.#answered(request, id, preferred, result.text);
      failures.push({ id, error: result.error });
      if (isCloudEngine(id, state)) cloudFailed = true;
    }

    // Finals are never dropped for an open breaker or a slow start alone: probe those members anyway.
    for (const id of deferred) {
      const m = this.#member(id);
      if (!this.#usable(m, from, to)) continue;
      const result = await this.#try(m, this.#breaker(kind, id, from, to), memberRequest, signal, startWaitMs, budgetMs);
      if (result.ok) return this.#answered(request, id, preferred, result.text);
      failures.push({ id, error: result.error });
    }

    const pair = `${from || "?"} → ${to}`;
    if (failures.length === 0) {
      throw new TranslationError(i18n.t("transl.router_error_no_engine", { pair }), { retryable: false });
    }
    throw new TranslationError(i18n.t("transl.router_error_all_failed", { errors: describe(failures) }), {
      // Repeating a chain that timed out would only time out again; other transient failures may clear.
      retryable: failures.some((f) => isRetryable(f.error) && !(f.error instanceof RouterAttemptTimeoutError)),
      missing: missingOf(failures.map((f) => f.error)),
    });
  }

  stop(): void {
    for (const m of this.#members.values()) this.#stopMember(m);
    this.#members.clear();
    this.#breakers.clear();
    this.#state = null;
    this.#notice = "";
  }

  /**
   * Whether the pair's chain can translate `from` → `to`: false only when every chain member has settled (started,
   * or failed to start and not due for a retry) and none of the started ones supports the pair. A member still
   * starting, or not started yet (another pair's chain), may still support it.
   */
  supports(from: string, to: string): boolean {
    const state = this.#state;
    if (!state) return true;
    let unsettled = false;
    for (const id of chainForPair(state.auto, from, to)) {
      const m = this.#members.get(id);
      if (m?.ready && m.engine) {
        if (m.engine.supports?.(from, to) !== false) return true;
      } else if (!m || m.starting || !this.#startFailedRecently(m)) {
        unsettled = true;
      }
    }
    return unsettled;
  }

  // --- internals ------------------------------------------------------------------------------------------------

  #member(id: TranslationEngineId): Member {
    let m = this.#members.get(id);
    if (!m) {
      m = { id, engine: null, starting: null, ready: false, startError: undefined, startFailedAt: 0, lifetime: new AbortController() };
      this.#members.set(id, m);
    }
    return m;
  }

  #breaker(kind: TranslationKind, id: TranslationEngineId, from: string, to: string): CircuitBreaker {
    const key = breakerKey(kind, id, from, to);
    let b = this.#breakers.get(key);
    if (!b) {
      b = new CircuitBreaker();
      this.#breakers.set(key, b);
    }
    return b;
  }

  /** Whether the member's breaker for `kind` holds requests back; interims also stay off a member failing finals. */
  #blocked(kind: TranslationKind, id: TranslationEngineId, from: string, to: string): boolean {
    const now = this.#now();
    if (this.#breakers.get(breakerKey(kind, id, from, to))?.blocked(now)) return true;
    return kind === "interim" && (this.#breakers.get(breakerKey("final", id, from, to))?.blocked(now) ?? false);
  }

  /** Ordered attempts for one line. */
  #candidates(
    kind: TranslationRequest["kind"],
    chain: readonly TranslationEngineId[],
    from: string,
    to: string,
  ): { id: TranslationEngineId; afterCloudFailure: boolean }[] {
    const state = this.#state!;
    const out: { id: TranslationEngineId; afterCloudFailure: boolean }[] = [];
    const push = (id: TranslationEngineId, afterCloudFailure = false, chosen = false) => {
      if (id === (Translation_Backends.auto as string)) return;
      if (out.some((c) => c.id === id)) return;
      // Engines whose slice turns interims off (quota, CPU) take none, unless picked as the interim engine.
      if (kind === "interim" && !chosen && state[id]?.interim === false) return;
      out.push({ id, afterCloudFailure });
    };
    if (kind === "interim") {
      const pick = state.auto.interimEngine;
      if (pick !== "auto") push(pick, false, true);
      else {
        const cheap = INTERIM_COST_ORDER.find((id) => this.#readyFor(kind, id, from, to, false) && state[id]?.interim !== false);
        if (cheap) push(cheap);
      }
    }
    for (const id of chain) push(id);
    if (state.auto.cloudFallback && kind === "final") for (const id of LOCAL_FALLBACKS) push(id, true);
    return out;
  }

  /** Member exists, is ready, supports the pair and its breaker for `kind` does not hold requests back. */
  #readyFor(kind: TranslationKind, id: TranslationEngineId, from: string, to: string, afterCloudFailure: boolean): boolean {
    if (afterCloudFailure) return false;
    const m = this.#members.get(id);
    if (!m?.ready || !m.engine) return false;
    if (m.engine.supports?.(from, to) === false) return false;
    return !this.#blocked(kind, id, from, to);
  }

  #usable(m: Member, from: string, to: string): boolean {
    if (m.ready && m.engine) return m.engine.supports?.(from, to) !== false;
    if (m.starting) return true;
    return !this.#startFailedRecently(m);
  }

  /** The member's last start failed less than `ROUTER_START_RETRY_MS` ago (it is not retried before then). */
  #startFailedRecently(m: Member): boolean {
    return m.startError !== undefined && this.#now() - m.startFailedAt < ROUTER_START_RETRY_MS;
  }

  #ensureStarted(m: Member): Promise<void> {
    if (m.ready) return Promise.resolve();
    if (m.starting) return m.starting;
    if (this.#startFailedRecently(m)) return Promise.reject(m.startError);
    const host = this.#host;
    const state = this.#state;
    const Engine = host?.engines[m.id];
    if (!host || !state || !Engine) return Promise.reject(new TranslationError(i18n.t("transl.error_not_running"), { retryable: false }));

    const engine = new Engine();
    m.engine = engine;
    m.lifetime = new AbortController();
    m.startError = undefined;
    const lifetime = m.lifetime.signal;
    engine.attach?.({ engines: host.engines, status: (u) => this.#memberStatus(u) });
    const starting = Promise.resolve()
      .then(() => engine.start(state, lifetime))
      .then(
        () => {
          if (m.engine !== engine) return;
          m.ready = true;
          m.starting = null;
        },
        (error: unknown) => {
          if (m.engine === engine) {
            m.starting = null;
            m.engine = null;
            m.startError = error;
            m.startFailedAt = this.#now();
            try {
              engine.stop();
            } catch {
              // ignore
            }
          }
          throw error;
        },
      );
    m.starting = starting;
    starting.catch(() => {}); // callers may have given up waiting
    return starting;
  }

  /**
   * One attempt on member `m`: waits (at most `startWaitMs`) for a member still starting, then translates within
   * `budgetMs`. Only the translation's outcome goes to `breaker`; a start that fails or is still pending is not the
   * engine failing a request (start failures are tracked by the member). Rejects only when `signal` aborts.
   */
  async #try(
    m: Member,
    breaker: CircuitBreaker,
    request: TranslationRequest,
    signal: AbortSignal,
    startWaitMs: number,
    budgetMs: number,
  ): Promise<AttemptResult> {
    if (!m.ready) {
      try {
        await this.#bounded(signal, startWaitMs, (s) => raceWithSignal(this.#ensureStarted(m), s), () => new RouterAttemptTimeoutError(m.id, startWaitMs, "start"));
      } catch (error) {
        breaker.release();
        if (signal.aborted) throw signal.reason ?? error;
        return { ok: false, error };
      }
    }
    try {
      const engine = m.ready ? m.engine : null;
      if (!engine) throw new TranslationError(i18n.t("transl.error_not_running"), { retryable: false });
      const text = await this.#bounded(
        signal,
        budgetMs,
        (s) => raceWithSignal(engine.translate(request, s), s),
        () => new RouterAttemptTimeoutError(m.id, budgetMs),
      );
      breaker.success();
      return { ok: true, text };
    } catch (error) {
      if (signal.aborted) {
        breaker.release();
        throw signal.reason ?? error;
      }
      breaker.failure(this.#now(), !isRetryable(error));
      return { ok: false, error };
    }
  }

  /** Runs `work` with a child signal that aborts with the caller's `signal` or with `timeout()` after `ms`. */
  async #bounded<T>(signal: AbortSignal, ms: number, work: (signal: AbortSignal) => Promise<T>, timeout: () => Error): Promise<T> {
    if (signal.aborted) throw signal.reason ?? new DOMException("Aborted", "AbortError");
    const child = new AbortController();
    const onAbort = () => child.abort(signal.reason ?? new DOMException("Aborted", "AbortError"));
    signal.addEventListener("abort", onAbort, { once: true });
    const timer = setTimeout(() => child.abort(timeout()), ms);
    try {
      return await work(child.signal);
    } finally {
      clearTimeout(timer);
      signal.removeEventListener("abort", onAbort);
    }
  }

  /** `id` answered the line: report it and show (or clear) the fallback warning. */
  #answered(request: TranslationRequest, id: TranslationEngineId, preferred: TranslationEngineId | undefined, text: string): string {
    request.onEngine?.(id);
    if (request.kind === "final") this.#noteFinal(id, preferred);
    return text;
  }

  #stopMember(m: Member): void {
    m.lifetime.abort();
    const engine = m.engine;
    m.engine = null;
    m.ready = false;
    m.starting = null;
    try {
      engine?.stop();
    } catch (e) {
      console.error("[Translation] router member stop failed:", e);
    }
  }

  #memberStatus(update: TranslationStatusUpdate): void {
    if (!update.text && this.#notice) {
      this.#host?.status({ text: this.#notice, level: "warning" });
      return;
    }
    this.#host?.status(update);
  }

  /** A final came from `id`: show (or clear) the fallback warning. */
  #noteFinal(id: TranslationEngineId, preferred: TranslationEngineId | undefined): void {
    if (!preferred) return;
    const notice =
      id === preferred ? "" : i18n.t("transl.router_status_fallback", { engine: engineLabel(id), preferred: engineLabel(preferred) }) || "";
    if (notice === this.#notice) return;
    this.#notice = notice;
    this.#host?.status(notice ? { text: notice, level: "warning" } : { text: "" });
  }
}
