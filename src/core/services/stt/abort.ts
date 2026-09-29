import { EventCallback, listen, UnlistenFn } from "@tauri-apps/api/event";

/** Rejects when `signal` is aborted (DOM `AbortError`). */
export function untilAborted(signal: AbortSignal): Promise<never> {
  return new Promise((_, reject) => {
    if (signal.aborted) {
      reject(new DOMException("Aborted", "AbortError"));
      return;
    }
    signal.addEventListener("abort", () => reject(new DOMException("Aborted", "AbortError")), { once: true });
  });
}

/**
 * Like `untilAborted`, plus a `cleanup` that removes the abort listener. Per-call races on a long-lived signal
 * must call it, or every call leaves one listener (and its closure) on the signal until it aborts.
 */
function abortRejection(signal: AbortSignal): { promise: Promise<never>; cleanup: () => void } {
  let onAbort: (() => void) | undefined;
  const promise = new Promise<never>((_, reject) => {
    if (signal.aborted) {
      reject(new DOMException("Aborted", "AbortError"));
      return;
    }
    onAbort = () => reject(new DOMException("Aborted", "AbortError"));
    signal.addEventListener("abort", onAbort, { once: true });
  });
  return {
    promise,
    cleanup: () => {
      if (onAbort) signal.removeEventListener("abort", onAbort);
    },
  };
}

/** Races `p` against abort; if `signal` is null/undefined, returns `p`. */
export async function raceWithAbort<T>(p: Promise<T>, signal: AbortSignal | null | undefined): Promise<T> {
  if (!signal) return p;
  const a = abortRejection(signal);
  try {
    return await Promise.race([p, a.promise]);
  } finally {
    a.cleanup();
  }
}

/**
 * Tauri `listen()` raced against abort. When the abort wins, the listener that registers afterwards is removed
 * as soon as it resolves instead of leaking (its `UnlistenFn` would otherwise be lost with the rejected race).
 */
export async function listenWithAbort<T>(
  event: string,
  handler: EventCallback<T>,
  signal: AbortSignal | null | undefined
): Promise<UnlistenFn> {
  const registering = listen<T>(event, handler);
  try {
    return await raceWithAbort(registering, signal);
  } catch (e) {
    void registering.then((unlisten) => unlisten(), () => {});
    throw e;
  }
}

export function isAbortError(e: unknown): boolean {
  return e instanceof DOMException && e.name === "AbortError";
}

/** Races `p` against an optional abort signal and a timeout; clears the timer when any branch settles. */
export async function raceWithTimeoutAndAbort<T>(
  p: Promise<T>,
  ms: number,
  timeoutLabel: string,
  signal: AbortSignal | null | undefined
): Promise<T> {
  let t: ReturnType<typeof setTimeout> | undefined;
  const timeout = new Promise<never>((_, reject) => {
    t = setTimeout(() => reject(new Error(`${timeoutLabel} (${ms / 1000}s)`)), ms);
  });
  const racers: Promise<T | never>[] = [p, timeout];
  const a = signal ? abortRejection(signal) : undefined;
  if (a) racers.push(a.promise);
  try {
    return await Promise.race(racers);
  } finally {
    if (t !== undefined) clearTimeout(t);
    a?.cleanup();
  }
}
