import type { RustModelEntryStatus, RustMtEngineStatus, RustMtError, RustMtPair, RustMtRequest, RustMtResponse } from "@/types";
import { invoke } from "@tauri-apps/api/core";
import { listen, UnlistenFn } from "@tauri-apps/api/event";
import i18n from "i18next";
import { raceWithSignal } from "../abort";
import { canonicalLanguage, guessLanguage, languageName, sameLanguage } from "../languages";
import { Translation_State } from "../schema";
import { ITranslationEngine, TranslationEngineHost, TranslationError, TranslationRequest } from "../types";

/** Catalog id of the helper runtime pack (Rust `process_host::recipes::BERGAMOT_RUNTIME_ID`). */
export const BERGAMOT_RUNTIME_ID = "runtime/bergamot";
/** Catalog id prefix of the direction models (`mt-bergamot/en_ja`, `model-catalog/mt-bergamot.json`). */
export const BERGAMOT_MODEL_PREFIX = "mt-bergamot/";
/** The language every Mozilla direction goes to or from; other pairs pivot through it. */
export const BERGAMOT_PIVOT = "en";
/** Refresh the installed directions this long after the last model-store change (installs come in bursts). */
export const BERGAMOT_REFRESH_DELAY_MS = 500;

/** Rejection value of the `plugin:mt-local|bergamot_*` commands (Rust `mt_local::MtError`). */
function isMtError(e: unknown): e is RustMtError {
  return (
    typeof e === "object" &&
    e !== null &&
    typeof (e as { kind?: unknown }).kind === "string" &&
    typeof (e as { message?: unknown }).message === "string"
  );
}

function messageOf(e: unknown): string {
  if (isMtError(e)) return e.message.trim();
  if (e instanceof Error) return e.message.trim();
  return e === undefined || e === null ? "" : String(e).trim();
}

/** "Japanese → English" for messages (English names; codes for unknown languages). */
export function bergamotPairLabel(from: string, to: string): string {
  const src = from ? languageName(from) : "?";
  return `${src} → ${languageName(to)}`;
}

/**
 * A user-facing `TranslationError` for a failed Bergamot command. `not_installed` names the catalog ids to download
 * (`missing`: the direction models, or the helper pack `runtime/bergamot`), pair and input problems never succeed on a
 * retry, `busy` and helper failures do.
 */
export function bergamotError(e: unknown, from = "", to = ""): TranslationError {
  const detail = messageOf(e) || i18n.t("common.error_no_details") || "no details";
  const failed = () => i18n.t("transl.bergamot_error_failed", { error: detail }) || detail;
  if (!isMtError(e)) return new TranslationError(failed(), { retryable: true });
  switch (e.kind) {
    case "not_installed": {
      const missing = e.missing?.length ? [...e.missing] : [];
      const message = missing.includes(BERGAMOT_RUNTIME_ID)
        ? i18n.t("transl.bergamot_error_runtime_missing")
        : i18n.t("transl.bergamot_error_not_installed", { pair: bergamotPairLabel(from, to) });
      return new TranslationError(message || detail, { retryable: false, missing });
    }
    case "unsupported_pair":
      return new TranslationError(i18n.t("transl.bergamot_error_pair", { error: detail }) || detail, { retryable: false });
    case "invalid_input":
    case "not_implemented":
      return new TranslationError(failed(), { retryable: false });
    case "cancelled":
      return new TranslationError(i18n.t("transl.bergamot_error_cancelled") || detail, { retryable: false });
    case "busy":
    default:
      return new TranslationError(failed(), { retryable: true });
  }
}

/** The key of a direction: canonical codes, lower-cased (`zh-hant>en`). */
function pairKey(source: string, target: string): string {
  return `${source.toLowerCase()}>${target.toLowerCase()}`;
}

/**
 * The directions `bergamot_pairs` reported (direct ones and English pivots, with install state), queried with
 * canonical codes. Codes Bergamot does not distinguish fall back to their base language (`pt-PT` → `pt`,
 * `en-GB` → `en`); Traditional Chinese never falls back to the Simplified models. Mirrors Rust
 * `mt_local/bergamot/catalog.rs`.
 */
export class BergamotPairs {
  readonly #byPair = new Map<string, RustMtPair>();
  readonly #sources = new Set<string>();
  readonly #targets = new Set<string>();

  constructor(pairs: readonly RustMtPair[]) {
    for (const p of pairs) {
      const source = canonicalLanguage(p.source) || p.source;
      const target = canonicalLanguage(p.target) || p.target;
      this.#byPair.set(pairKey(source, target), p);
      this.#sources.add(source.toLowerCase());
      this.#targets.add(target.toLowerCase());
    }
  }

  get size(): number {
    return this.#byPair.size;
  }

  /** The code Bergamot uses for a canonical code on the given side, or "" when it has none. */
  #lang(code: string, side: Set<string>): string {
    const c = canonicalLanguage(code).toLowerCase();
    if (!c) return "";
    if (side.has(c)) return c;
    if (c.startsWith("zh-hant")) return "";
    const base = c.split("-")[0];
    return side.has(base) ? base : "";
  }

  /** Whether any direction translates from `code`. */
  knowsSource(code: string): boolean {
    return this.#lang(code, this.#sources) !== "";
  }

  /** Whether any direction translates into `code`. */
  knowsTarget(code: string): boolean {
    return this.#lang(code, this.#targets) !== "";
  }

  /** The direction (direct or pivot) for a pair of canonical codes; undefined when the catalog has none. */
  find(from: string, to: string): RustMtPair | undefined {
    const s = this.#lang(from, this.#sources);
    const t = this.#lang(to, this.#targets);
    if (!s || !t) return undefined;
    return this.#byPair.get(pairKey(s, t));
  }

  /** Whether the pair can be translated right now (its models are installed). */
  installed(from: string, to: string): boolean {
    return this.find(from, to)?.installed === true;
  }

  /** Whether some installed direction (direct or through English) translates into `to`, from any source. */
  reachesInstalled(to: string): boolean {
    const t = this.#lang(to, this.#targets);
    if (!t) return false;
    for (const p of this.#byPair.values()) {
      if (p.installed && (canonicalLanguage(p.target) || p.target).toLowerCase() === t) return true;
    }
    return false;
  }

  /**
   * Catalog ids the pair needs. With an unknown source only the leg into the target is known (`en_<target>`), and
   * nothing for an English target.
   */
  neededIds(from: string, to: string): string[] {
    if (from) return this.find(from, to)?.ids.slice() ?? [];
    if (canonicalLanguage(to) === BERGAMOT_PIVOT) return [];
    return this.find(BERGAMOT_PIVOT, to)?.ids.slice() ?? [];
  }
}

interface BergamotConfig {
  threads: number;
}

/**
 * Bergamot engine (`bergamot` backend) over `plugin:mt-local|bergamot_*` (Rust `mt_local/bergamot.rs`, the
 * `sigil-bergamot.exe` helper running Mozilla's Firefox Translations models on the processor, no graphics memory).
 * About 57 languages to and from English; other pairs pivot through English. Cheap enough to re-translate partials,
 * so the router prefers it for interims.
 *
 * - `start` lists the directions (`bergamot_pairs`), rejects a configured pair Bergamot cannot translate, and names
 *   the missing downloads (`missing`: the direction models, or the helper pack `runtime/bergamot`) without
 *   downloading anything. It then starts the helper (`bergamot_start`) and loads the configured pair's models with a
 *   short warm-up line, so the first caption does not wait for them. A missing direction model is also shown as a
 *   warning status (a router member's start failure is otherwise silent); a missing helper pack is not, since the
 *   pack is built locally and a router whose chain lists Bergamot simply falls through to the next engine.
 * - `supports()` answers from the installed directions and stays current: model-store changes (`models:state`)
 *   refresh them while the engine runs. An unknown source (`""`) is supported when an installed direction reaches
 *   the target, since `translate` guesses the source.
 * - `translate` sends canonical codes; an unknown source is guessed from the script (`guessLanguage`), since
 *   Bergamot has no language detection. Aborted requests are cancelled in Rust (`bergamot_cancel`).
 * - `stop` stops the helper once no other engine instance (a router member, a second pipeline) uses it.
 *
 * Owned by P4-E.
 */
export class Translation_BergamotService implements ITranslationEngine {
  /** Engine instances that started the helper and have not stopped: the last one to stop stops it. */
  static readonly #users = new Set<Translation_BergamotService>();

  #host: TranslationEngineHost | null = null;
  #config: BergamotConfig | null = null;
  #pairs: BergamotPairs | null = null;
  #source = "";
  #target = "";
  #stopped = false;
  #counter = 0;
  #unlisten: UnlistenFn | null = null;
  #refreshTimer: ReturnType<typeof setTimeout> | null = null;

  attach(host: TranslationEngineHost): void {
    this.#host = host;
  }

  async start(state: Translation_State, signal: AbortSignal): Promise<void> {
    const target = canonicalLanguage(state.targetLanguage);
    if (!target) throw new TranslationError(i18n.t("transl.error_target_missing"), { retryable: false });
    const source = canonicalLanguage(state.sourceLanguage);
    const config: BergamotConfig = { threads: state.bergamot.threads };

    Translation_BergamotService.#users.add(this);
    this.#host?.status({ text: i18n.t("transl.bergamot_status_starting") || "", level: "info", progress: null });
    let pairs: BergamotPairs;
    try {
      pairs = new BergamotPairs(await raceWithSignal(invoke<RustMtPair[]>("plugin:mt-local|bergamot_pairs"), signal));
      if (!pairs.knowsTarget(target)) {
        throw new TranslationError(i18n.t("transl.bergamot_error_language", { language: languageName(target) }), { retryable: false });
      }
      if (source && !sameLanguage(source, target) && !pairs.find(source, target)) {
        const language = pairs.knowsSource(source) ? bergamotPairLabel(source, target) : languageName(source);
        throw new TranslationError(i18n.t("transl.bergamot_error_language", { language }), { retryable: false });
      }
      const missing = pairs.neededIds(source, target).filter((id) => !this.#idInstalled(pairs, id));
      if (missing.length) {
        throw new TranslationError(i18n.t("transl.bergamot_error_not_installed", { pair: bergamotPairLabel(source, target) }), {
          retryable: false,
          missing,
        });
      }
      // Automatic source: every line is translated from the language guessed for it, so at least one direction
      // into the target must be installed (into English, no single model is implied).
      if (!source && !pairs.reachesInstalled(target)) {
        throw new TranslationError(i18n.t("transl.bergamot_unknown_source"), { retryable: false });
      }
      await raceWithSignal(invoke<RustMtEngineStatus>("plugin:mt-local|bergamot_start", { threads: config.threads }), signal);
      await this.#warmUp(source, target, config, signal);
    } catch (e) {
      this.#host?.status({ text: "" });
      if (signal.aborted) {
        this.#release();
        throw signal.reason ?? e;
      }
      const error = e instanceof TranslationError ? e : bergamotError(e, source, target);
      // Name missing direction models (small downloads) in the status; the error itself carries every missing id.
      if (error.missing.length && !error.missing.includes(BERGAMOT_RUNTIME_ID)) {
        this.#host?.status({ text: error.message, level: "warning", missing: error.missing });
      }
      this.#release();
      throw error;
    }
    this.#host?.status({ text: "" });
    // Stopped (or superseded) while starting: give the helper back.
    if (signal.aborted || this.#stopped) {
      this.#release();
      throw signal.reason ?? new DOMException("Aborted", "AbortError");
    }
    this.#config = config;
    this.#pairs = pairs;
    this.#source = source;
    this.#target = target;
    await this.#watchModels();
  }

  async translate(request: TranslationRequest, signal: AbortSignal): Promise<string> {
    const config = this.#config;
    if (!config || !this.#pairs) throw new TranslationError(i18n.t("transl.error_not_running"), { retryable: false });
    const target = canonicalLanguage(request.to ?? this.#target) || this.#target;
    let source = canonicalLanguage(request.from ?? this.#source);
    if (source && sameLanguage(source, target)) return request.text;
    // Bergamot cannot detect the language: guess it from the script (Japanese, Chinese, Cyrillic, Arabic, ...).
    if (!source) source = guessLanguage(request.text);
    if (!source) throw new TranslationError(i18n.t("transl.bergamot_error_source_unknown"), { retryable: false });
    if (sameLanguage(source, target)) return request.text;
    const requestId = request.requestId?.trim() || `bergamot-${++this.#counter}`;
    const payload: RustMtRequest = {
      text: request.text,
      source,
      target,
      kind: request.kind,
      requestId,
      threads: config.threads,
    };
    const onAbort = () => {
      void invoke("plugin:mt-local|bergamot_cancel", { requestId }).catch(() => {});
    };
    signal.addEventListener("abort", onAbort, { once: true });
    try {
      const response = await raceWithSignal(invoke<RustMtResponse>("plugin:mt-local|bergamot_translate", { request: payload }), signal);
      return response.text;
    } catch (e) {
      if (signal.aborted) throw signal.reason ?? e;
      const error = bergamotError(e, source, target);
      if (error.missing.length) {
        this.#host?.status({ text: error.message, level: "warning", missing: error.missing });
        this.#scheduleRefresh();
      }
      throw error;
    } finally {
      signal.removeEventListener("abort", onAbort);
    }
  }

  stop(): void {
    this.#stopped = true;
    this.#config = null;
    this.#pairs = null;
    if (this.#refreshTimer !== null) clearTimeout(this.#refreshTimer);
    this.#refreshTimer = null;
    const unlisten = this.#unlisten;
    this.#unlisten = null;
    unlisten?.();
    this.#release();
  }

  supports(from: string, to: string): boolean {
    const pairs = this.#pairs;
    if (!pairs) return false;
    if (from && sameLanguage(from, to)) return true;
    // Unknown source: `translate` guesses it from the script, so any installed direction into `to` may serve.
    if (!from) return pairs.reachesInstalled(to);
    return pairs.installed(from, to);
  }

  /** Every direction the catalog offers (direct and English pivots), with install state; [] when unavailable. */
  static async getPairs(): Promise<RustMtPair[]> {
    try {
      return await invoke<RustMtPair[]>("plugin:mt-local|bergamot_pairs");
    } catch {
      return [];
    }
  }

  #idInstalled(pairs: BergamotPairs, id: string): boolean {
    // A direct pair's install state is its model's; pivots report both legs together, so look the leg up directly.
    const m = /^mt-bergamot\/([a-z-]+)_([a-z-]+)$/.exec(id);
    return m ? pairs.installed(m[1], m[2]) : false;
  }

  /** Load the configured pair's models now (a short line), so the first caption does not wait for them. */
  async #warmUp(source: string, target: string, config: BergamotConfig, signal: AbortSignal): Promise<void> {
    let from = source;
    if (!from) {
      if (target === BERGAMOT_PIVOT) return; // unknown source into English: nothing to preload yet
      from = BERGAMOT_PIVOT;
    }
    if (sameLanguage(from, target)) return;
    const request: RustMtRequest = { text: "OK.", source: from, target, kind: "final", requestId: "bergamot-warmup", threads: config.threads };
    await raceWithSignal(invoke<RustMtResponse>("plugin:mt-local|bergamot_translate", { request }), signal);
  }

  async #watchModels(): Promise<void> {
    try {
      const unlisten = await listen<RustModelEntryStatus>("models:state", (event) => {
        const id = event.payload?.id ?? "";
        if (id === BERGAMOT_RUNTIME_ID || id.startsWith(BERGAMOT_MODEL_PREFIX)) this.#scheduleRefresh();
      });
      if (this.#stopped) unlisten();
      else this.#unlisten = unlisten;
    } catch {
      // No event bus (tests, web preview): the installed directions refresh after a not_installed error instead.
    }
  }

  #scheduleRefresh(): void {
    if (this.#stopped) return;
    if (this.#refreshTimer !== null) clearTimeout(this.#refreshTimer);
    this.#refreshTimer = setTimeout(() => {
      this.#refreshTimer = null;
      void invoke<RustMtPair[]>("plugin:mt-local|bergamot_pairs").then(
        (pairs) => {
          if (!this.#stopped && this.#config) this.#pairs = new BergamotPairs(pairs);
        },
        () => {},
      );
    }, BERGAMOT_REFRESH_DELAY_MS);
  }

  /** Forget this instance; the last user stops the helper. */
  #release(): void {
    if (!Translation_BergamotService.#users.delete(this)) return;
    if (Translation_BergamotService.#users.size === 0) void invoke("plugin:mt-local|bergamot_stop").catch(() => {});
  }
}
