import { abortableDelay } from "./abort";
import { TranslationError } from "./types";

export interface RetryOptions {
  /** Pause before the retry (ms). */
  delayMs?: number;
  /** Retry only when the first attempt failed within this many ms (slow failures are timeouts). */
  retryWithinMs?: number;
  now?: () => number;
}

/** Whether a failed translation is worth one more attempt. */
export function isRetryableTranslationError(error: unknown): boolean {
  if (error instanceof TranslationError) return error.retryable;
  if (error instanceof Error && (error.name === "AbortError" || error.name === "SequencerTimeoutError")) return false;
  return true;
}

/**
 * Runs `attempt`, and once more after a short pause when the first failure was fast and retryable
 * (connection reset, 429, 5xx). Used for finals only: they feed TTS and the chatbox, so one transient
 * error should not lose a sentence, while interims are superseded by the next partial anyway.
 */
export async function withOneRetry<T>(
  attempt: () => Promise<T>,
  signal: AbortSignal,
  options: RetryOptions = {},
): Promise<T> {
  const { delayMs = 500, retryWithinMs = 5_000, now = () => Date.now() } = options;
  const startedAt = now();
  try {
    return await attempt();
  } catch (error) {
    if (signal.aborted || !isRetryableTranslationError(error) || now() - startedAt > retryWithinMs) throw error;
    await abortableDelay(delayMs, signal);
    return attempt();
  }
}
