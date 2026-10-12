import type { TranslationEngineId, Translation_State } from "./schema";

export type TranslationKind = "interim" | "final";

/** An earlier final line of the same stream and its translation (rolling context). */
export interface TranslationContextPair {
  source: string;
  target: string;
}

/** A glossary term already filtered to the line (its `source` occurs in `text`). */
export interface TranslationGlossaryTerm {
  source: string;
  target: string;
}

/*
 * Request fields added by P4-S0 (frozen; all optional, so the P0 engines keep working unchanged). The pipeline /
 * router (P4-C) fills them; an engine that ignores one must still translate correctly.
 */
export interface TranslationRequest {
  /** Source text, non-empty. */
  text: string;
  /** Interims are live partials that may be superseded; finals must be delivered. */
  kind: TranslationKind;
  /**
   * Canonical lower-case BCP-47 source language (`translation/languages.ts`); "" or absent = unknown (the engine
   * detects it or uses its own setting). Absent together with `to` = the engine's own slice settings (the legacy
   * single-backend path).
   */
  from?: string;
  /** Canonical BCP-47 target language; absent = the engine's own slice setting. */
  to?: string;
  /** Earlier final pairs of this stream, oldest first (LLM prompts, DeepL `context`); absent = none. */
  context?: TranslationContextPair[];
  /** Glossary terms that occur in `text`; absent = none. */
  glossary?: TranslationGlossaryTerm[];
  /** Free-text background from the context settings (LLM prompts). */
  background?: string;
  /** Utterance id (`TextEvent.segmentId`, or a local counter) plus revision, e.g. `mic:1:4#2`: logs and `*_cancel`. */
  requestId?: string;
  /**
   * Streaming engines (LLM) call this with the translation so far, in order, before resolving. Optional: the
   * pipeline decides what to publish from it (never more than a final's committed text).
   */
  onPartial?: (text: string) => void;
  /** Router (P4-C): called with the chain member whose result is returned, before `translate` resolves. */
  onEngine?: (engine: TranslationEngineId) => void;
  /**
   * Router (`routingTable.ts`): decoding beam width a built-in route asks of the engine for a final (NLLB; the
   * larger of this and the engine's own setting is used). Engines without beam search ignore it; absent = the
   * engine's own setting.
   */
  beam?: number;
}

/** Engine preparation status for the UI (model download or load, server start, a fallback in use); never a caption. */
export interface TranslationStatusUpdate {
  /** "" clears the status line. */
  text: string;
  level?: "info" | "warning";
  /** 0-100 while downloading / loading; null or absent = no progress bar. */
  progress?: number | null;
  /** Catalog ids the engine needs installed (the inspector offers them in Models & Runtimes). */
  missing?: string[];
}

/** What the host (the translation pipeline) gives an engine before `start`. */
export interface TranslationEngineHost {
  /** Constructors of the chainable engines (the router builds its chain from them). */
  readonly engines: Readonly<Record<TranslationEngineId, ITranslationEngineConstructor>>;
  /** Reports preparation status; the pipeline shows the latest one. */
  status(update: TranslationStatusUpdate): void;
  /**
   * The bundled llama-server translator runs on the processor right now (`interimPolicy.bundledTranslatorOnCpu`,
   * read per line). The router's CPU-tier routes (`routingTable.ts`) apply only while this is true; absent = false.
   */
  translatorOnCpu?(): boolean;
  /**
   * Whether the model store reports catalog entry `id` installed (`window.ApiServer.models`), read when a member's start is due.
   * The router does not start a member again after its start failed on a missing model (`TranslationError.missing`) that this
   * reports not installed, until every missing id is installed: a retry on a timer posts the "model missing" status card again
   * each time. A start that failed on a model this already reports installed (the store is stale) is retried on the timer
   * (`ROUTER_START_RETRY_MS`), like an absent `modelInstalled`.
   */
  modelInstalled?(id: string): boolean;
}

/**
 * One translation backend. Engines never publish anything themselves: the service sequences their
 * results (see `sequencer.ts`), so every method either resolves with a value or throws.
 *
 * - `start` validates settings and connects. It rejects with a user-facing message when the backend
 *   cannot be used, and must honour `signal` (aborted when the user stops or restarts while starting):
 *   anything it created after the abort has to be released before it rejects.
 * - `translate` resolves with the translation or throws (a `TranslationError` when the failure is
 *   known not to be worth a retry). It must honour `signal`: interims are aborted when superseded.
 * - `stop` releases resources. It may be called while `start` or `translate` calls are pending.
 */
export interface ITranslationEngine {
  start(state: Translation_State, signal: AbortSignal): Promise<void>;
  translate(request: TranslationRequest, signal: AbortSignal): Promise<string>;
  stop(): void;
  /**
   * Optional (P4-S0): whether the started engine can translate `from` → `to` (canonical BCP-47; `from` may be "" =
   * unknown). The router skips engines that answer false; absent = assume yes. Must be cheap and synchronous
   * (answer from data loaded in `start`).
   */
  supports?(from: string, to: string): boolean;
  /** Optional (P4-C): receives the host before `start` (router chain, status reporting). */
  attach?(host: TranslationEngineHost): void;
}

export interface ITranslationEngineConstructor {
  new (): ITranslationEngine;
}

/** A failed translation with a user-facing message. */
export class TranslationError extends Error {
  /** HTTP status when the backend answered with one. */
  readonly status?: number;
  /** Whether repeating the same request can succeed (network hiccup, 429, 5xx). */
  readonly retryable: boolean;
  /** Catalog ids to install before the engine can work (`not_installed`); [] otherwise. */
  readonly missing: string[];

  constructor(message: string, options: { status?: number; retryable?: boolean; missing?: string[] } = {}) {
    super(message);
    this.name = "TranslationError";
    this.status = options.status;
    this.retryable = options.retryable ?? true;
    this.missing = options.missing ?? [];
  }
}

/** Retryable HTTP statuses: rate limits, timeouts and server-side failures. */
export function isRetryableStatus(status: number): boolean {
  return status === 408 || status === 429 || status >= 500;
}
