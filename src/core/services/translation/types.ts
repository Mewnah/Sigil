import { Translation_State } from "./schema";

export type TranslationKind = "interim" | "final";

export interface TranslationRequest {
  /** Source text, non-empty. */
  text: string;
  /** Interims are live partials that may be superseded; finals must be delivered. */
  kind: TranslationKind;
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

  constructor(message: string, options: { status?: number; retryable?: boolean } = {}) {
    super(message);
    this.name = "TranslationError";
    this.status = options.status;
    this.retryable = options.retryable ?? true;
  }
}

/** Retryable HTTP statuses: rate limits, timeouts and server-side failures. */
export function isRetryableStatus(status: number): boolean {
  return status === 408 || status === 429 || status >= 500;
}
