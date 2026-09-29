import { TTS_State } from "./schema";

/**
 * `onStop` error a backend reports when required settings (voice, key, device, ...) are missing.
 * Service_TTS turns it into a localized "needs setup" toast instead of an error.
 */
export const TTS_OPTIONS_MISSING = "Options missing";

/** How a backend reports to Service_TTS. Each backend instance gets its own receiver. */
export interface ITTSReceiver {
  /** The backend is ready to speak. */
  onStart(): void;
  /**
   * The backend stopped. With `error` it failed to start or hit a fatal error (bad key, lost
   * session), and the service returns to disconnected. Called without an error from `stop()`.
   */
  onStop(error?: string): void;
  /**
   * One utterance failed (network hiccup, server error, unplayable audio or missing device), but
   * the backend keeps running. Reported to the user and the system log, with throttling.
   */
  onError(error: string): void;
}

export interface ITTSService {
  /** Validate settings and connect. Report `onStart()` or `onStop(error)` when done (sync or async). */
  start(params: TTS_State): void | Promise<void>;
  /**
   * Stop speaking and release resources: cancel pending synthesis, silence queued audio and call
   * `onStop()`. Must be safe to call more than once and while `start` is still in progress.
   */
  stop(): void;
  /** Speak one final text. Must not throw; failures go to `onError`. */
  play(value: string): void | Promise<void>;
  /** Final cleanup after `stop()`. */
  dispose(): void;
}

export interface ITTSServiceConstructor {
  new (receiver: ITTSReceiver): ITTSService;
}

/** Readable message from an unknown thrown value (Tauri invoke rejects with plain strings). */
export function errorMessage(error: unknown): string {
  if (typeof error === "string") return error;
  if (error instanceof Error) return error.message || error.name;
  if (error && typeof error === "object" && "message" in error && typeof (error as { message: unknown }).message === "string") {
    return (error as { message: string }).message;
  }
  try {
    return JSON.stringify(error) ?? String(error);
  } catch {
    return String(error);
  }
}

/** True for errors from an aborted fetch/AbortController (not a failure worth reporting). */
export function isAbortError(error: unknown): boolean {
  return (
    !!error &&
    typeof error === "object" &&
    "name" in error &&
    (error as { name: unknown }).name === "AbortError"
  );
}
