import type { ReactNode } from "react";
import { ErrorBoundary } from "react-error-boundary";

/**
 * Self-recovery of the `/client` page (the OBS browser source). It is on stream for hours with no
 * one to refresh it, and a panel or an error text on stream is worse than nothing: when its React
 * tree throws (or start-up fails for a reason a retry cannot fix in place) the page stays
 * transparent and reloads itself. Reloads are rate limited, so a failure that survives a reload
 * cannot spin: each one inside the last [`RELOAD_WINDOW_MS`] doubles the wait. A reload only
 * happens while the page's own address answers: if Sigil is down at that moment a reload would
 * land on the browser's error page, which has no script left to try again.
 */

/** Wait before the first reload; doubled for every reload this tab already made in the window. */
export const RELOAD_BASE_DELAY_MS = 4_000;
/** The wait never grows past this. */
export const RELOAD_MAX_DELAY_MS = 120_000;
/** Reloads older than this no longer count. */
export const RELOAD_WINDOW_MS = 10 * 60_000;
/** Wait when the reloads of this tab cannot be counted (no session storage, or it refuses writes). */
export const RELOAD_UNCOUNTED_DELAY_MS = 30_000;
/** A probe of the page's address that takes longer than this counts as unreachable. */
export const RELOAD_PROBE_TIMEOUT_MS = 5_000;
/** Pause before the next probe while the address does not answer. */
export const RELOAD_PROBE_RETRY_MS = 5_000;

const STORAGE_KEY = "sigil.client.reloads";

/** What a reload needs from its surroundings (replaced in tests). */
export interface ReloadEnv {
  now(): number;
  /** Per-tab storage, or null when the browser refuses it. */
  storage: Pick<Storage, "getItem" | "setItem"> | null;
  setTimeout(fn: () => void, ms: number): unknown;
  reload(): void;
  /** Whether this page's own address answers right now (a probe that fails or times out is false). */
  reachable(): Promise<boolean>;
}

/** Wait for the next reload after `recent` earlier ones in the window; null: unknown. */
export function reloadDelayMs(recent: number | null): number {
  if (recent === null) return RELOAD_UNCOUNTED_DELAY_MS;
  return Math.min(RELOAD_MAX_DELAY_MS, RELOAD_BASE_DELAY_MS * 2 ** Math.min(recent, 16));
}

async function probeOwnAddress(): Promise<boolean> {
  const controller = new AbortController();
  const timer = window.setTimeout(() => controller.abort(), RELOAD_PROBE_TIMEOUT_MS);
  try {
    // The page itself: the same URL (token and all) a reload would load.
    const response = await fetch(window.location.href, { cache: "no-store", signal: controller.signal });
    return response.ok;
  } catch {
    return false;
  } finally {
    window.clearTimeout(timer);
  }
}

function browserEnv(): ReloadEnv {
  let storage: ReloadEnv["storage"] = null;
  try {
    storage = window.sessionStorage;
  } catch {
    /* blocked */
  }
  return {
    now: () => Date.now(),
    storage,
    setTimeout: (fn, ms) => window.setTimeout(fn, ms),
    reload: () => window.location.reload(),
    reachable: probeOwnAddress,
  };
}

/** Reload times (ms) of this tab inside the window, or null when they cannot be read. */
function recentReloads(env: ReloadEnv): number[] | null {
  if (!env.storage) return null;
  try {
    const raw = JSON.parse(env.storage.getItem(STORAGE_KEY) ?? "[]");
    const now = env.now();
    return Array.isArray(raw)
      ? raw.filter((t): t is number => typeof t === "number" && t <= now && now - t < RELOAD_WINDOW_MS)
      : [];
  } catch {
    return null;
  }
}

let reloadPending = false;

/** Reload once the page's address answers; until then probe again every few seconds. */
function reloadWhenReachable(env: ReloadEnv) {
  env
    .reachable()
    .catch(() => false)
    .then((reachable) => {
      if (reachable) {
        reloadPending = false;
        env.reload();
        return;
      }
      env.setTimeout(() => reloadWhenReachable(env), RELOAD_PROBE_RETRY_MS);
    });
}

/**
 * Reload this page after a delay that grows with the reloads this tab already made in the last
 * ten minutes (4 s, 8 s, ... up to 2 min), and return that delay. The reload waits for the page's
 * address to answer. A second call while one reload is pending does nothing and returns null.
 */
export function scheduleClientReload(reason: string, error?: unknown, env: ReloadEnv = browserEnv()): number | null {
  if (reloadPending) return null;
  const recent = recentReloads(env);
  let delay = reloadDelayMs(recent === null ? null : recent.length);
  if (recent !== null) {
    try {
      env.storage!.setItem(STORAGE_KEY, JSON.stringify([...recent, env.now()]));
    } catch {
      // Reads work but writes are refused (a full quota, a private-mode storage): the next load
      // would count no reloads and wait the shortest time again, for ever. Treat it as no storage.
      delay = Math.max(delay, RELOAD_UNCOUNTED_DELAY_MS);
    }
  }
  console.error(`[client] ${reason}; reloading in ${Math.round(delay / 1000)} s`, error);
  reloadPending = true;
  env.setTimeout(() => reloadWhenReachable(env), delay);
  return delay;
}

/**
 * The /client view is rendered inside this. A render error unmounts a React tree for good, which
 * left the OBS source blank until someone refreshed it: show nothing and reload instead.
 */
export function ClientErrorBoundary({ children }: { children: ReactNode }) {
  return (
    <ErrorBoundary
      fallbackRender={() => null}
      onError={(error) => {
        scheduleClientReload("the page crashed", error);
      }}
    >
      {children}
    </ErrorBoundary>
  );
}
