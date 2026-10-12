/**
 * Restart policy shared by every STT engine that restarts itself after a failure: the built-in engine (`local.ts`),
 * and Service_STT for the HTTP and cloud engines (OpenAI-compatible, Groq, Deepgram, Azure). Pure (no `window`, no
 * Tauri).
 *
 * A session that stops with a failure a restart can fix is restarted after 1, 2, 5, 10 and 20 s (the fast ladder). When
 * those are used up it keeps trying every {@link SLOW_RESTART_MS} while the user still wants it running (Start was
 * pressed, Stop was not): a streamer in VR cannot press Start again, and a fault that lasts minutes (the microphone
 * held by another app, the connection down) ends by itself. A restart that fails at start never ran, so it never
 * resets the count; a session that ran {@link STABLE_RUN_MS} begins a new one. Errors a restart cannot fix
 * ({@link restartCanFix}) still stop with their message.
 */
import type { RustSpeechRoutePlan, RustSpeechTier } from "@/types";

/** Waits before the fast automatic restarts after a pipeline error. */
export const RESTART_DELAYS_MS: readonly number[] = [1_000, 2_000, 5_000, 10_000, 20_000];
/** Failed restarts in a row after which the retries slow down to {@link SLOW_RESTART_MS}. */
export const MAX_RESTARTS = RESTART_DELAYS_MS.length;
/** Wait between restarts once the fast ladder is used up. */
export const SLOW_RESTART_MS = 45_000;
/**
 * A session that failed after running this long (since its last successful start) begins a new restart budget. A
 * restart that fails at start never ran, so it never resets the budget.
 */
export const STABLE_RUN_MS = 60_000;

/** Wait before restart number `attempt` (1 = the first restart of a failing streak). */
export function restartDelayMs(attempt: number): number {
  return attempt > MAX_RESTARTS ? SLOW_RESTART_MS : RESTART_DELAYS_MS[Math.max(0, attempt - 1)];
}

/** Whether restart number `attempt` is past the fast ladder (the session retries at the slow pace). */
export function isSlowRestart(attempt: number): boolean {
  return attempt > MAX_RESTARTS;
}

/** `speech:status` codes of a fatal error that a restart can fix. */
const RETRYABLE_CODES: ReadonlySet<string> = new Set([
  "capture_error",
  "capture_gap",
  "engine_error",
  "engine_fallback",
  "backlog_dropped",
  "rate_limited",
  "",
]);
/**
 * Rust engine error kinds (`AsrErrorKind`) that lead a fatal detail or a start error as `<kind>: message`
 * (`worker.rs` fatals them at once, `is_fatal`): the same setup fails the same way on every start.
 */
const UNFIXABLE_KIND = /^\s*(invalid_input|unsupported|out_of_memory):/;
/**
 * A missing model, runtime pack or file (`not_installed: <why>`, with or without a catalog id): the router and the
 * model store own installs, and a restart installs nothing. Without this a bring-your-own model that was never chosen
 * would be restarted for ever.
 */
const NOT_INSTALLED = /\bnot_installed:/;
/** Start errors that no restart changes: nothing can run this request, or another start took the pipeline over. */
const UNFIXABLE_START = /no speech engine for this request|^\s*cancelled\s*$/i;

/** Catalog id named by a Rust `not_installed: <id>[: why]` error, if any. */
export function notInstalledId(error: string): string | null {
  const m = /not_installed:\s*([A-Za-z0-9._-]+(?:\/[A-Za-z0-9._-]+)+)/.exec(error);
  return m ? m[1] : null;
}

/**
 * Whether restarting the pipeline can fix a fatal error: its `speech:status` code (`""` for a failed start) is one a
 * restart can fix, and its detail does not start with an engine error kind that fails on every start
 * (`invalid_input`, `unsupported`). `out_of_memory` is worth a restart only when the next start can land on a lighter
 * setup: the plan has a CPU fallback, or the tier is automatic (the router plans again with the VRAM the resource
 * governor grants now). Restarts of a failing streak never end, but they slow down after {@link MAX_RESTARTS}.
 */
export function restartCanFix(
  code: string,
  detail: string,
  plan: Pick<RustSpeechRoutePlan, "fallback"> | null,
  tier: RustSpeechTier | undefined,
): boolean {
  if (code === "not_installed" || !RETRYABLE_CODES.has(code)) return false;
  if (NOT_INSTALLED.test(detail) || notInstalledId(detail)) return false;
  const kind = UNFIXABLE_KIND.exec(detail)?.[1];
  if (!kind) return true;
  if (kind === "out_of_memory") return !!plan?.fallback || (tier ?? "auto") === "auto";
  return false;
}

/**
 * Whether a `start_pipeline` that failed (`detail`: the command's error) is worth another try: a busy microphone, an
 * engine that failed to load right now. Not a missing model, a request nothing can run, or a start another one
 * cancelled.
 */
export function startFailureCanRetry(
  detail: string,
  plan: Pick<RustSpeechRoutePlan, "fallback"> | null,
  tier: RustSpeechTier | undefined,
): boolean {
  return !UNFIXABLE_START.test(detail) && restartCanFix("", detail, plan, tier);
}

/**
 * Whether a failure of an HTTP or cloud engine passes by itself (a network error, a busy microphone, a rate limit
 * that clears when its window ends) or needs the user (rejected key or request). The same rules as the built-in
 * engine's. A rate limit is retried too, at the slow pace ({@link networkFailureNeedsPause}): Groq's per-minute limit
 * ends by itself, and a used-up daily quota only costs one quiet attempt every {@link SLOW_RESTART_MS}.
 */
export function networkFailureCanRetry(code: string, detail: string): boolean {
  return !UNFIXABLE_START.test(detail) && restartCanFix(code, detail, null, undefined);
}

/**
 * Whether the webview does not know the network to be down (`navigator.onLine`; unknown counts as up). A refused
 * connection while it is up may be a rejected key or a wrong region, so engines report it `unproven`
 * (`SttStopOptions.unproven`); while it is down the refusal is the outage and retries go on.
 */
export function networkLooksUp(): boolean {
  return typeof navigator === "undefined" || navigator.onLine !== false;
}

/** Whether a retryable failure needs a long pause first (a rate limit: trying again at once only extends it). */
export function networkFailureNeedsPause(code: string): boolean {
  return code === "rate_limited";
}
