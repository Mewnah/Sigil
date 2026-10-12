/**
 * First-run decision: does this launch show the setup wizard, and who may connect to the web server?
 *
 * Every profile written before the wizard was mounted has `onboardingComplete` false, so that flag alone would show a
 * "first run" wizard to people who have used Sigil for months. The decision is therefore taken once, at the first
 * launch that knows about it (`firstRunDecided`):
 * - no settings file at all: a new install, even when a `settings.bak` is still lying around (the loader never reads
 *   the backup of a file that is gone, and the Rust side calls it a new install too). `onboardingComplete` stays
 *   false, so the wizard shows (and the STT and TTS defaults treat the profile as fresh), and `lanAccess.mode`
 *   becomes `localhost` ("This PC only"): the web server listens on 127.0.0.1, so the first launch never raises the
 *   Windows Firewall prompt. Turning on LAN access in Settings is one switch away;
 * - anything else (a profile that loaded, was recovered from its backup, or could not be read): an existing user.
 *   `onboardingComplete` becomes true and the wizard never appears, and `lanAccess` stays as the profile has it (a
 *   profile written before `lanAccess` existed gets the schema default `lan`, as it always had).
 *
 * The Rust web plugin takes the same decision for the address it binds before this code runs
 * (`startup_lan_access` in src-tauri/src/services/web/auth.rs: no settings file means `localhost`), so the server
 * already listens on 127.0.0.1 when this pushes `localhost` and nothing rebinds. The two sides cannot see each other, so
 * their cases are one table, src-tauri/src/services/web/startup_lan_cases.json, which the tests of both read (a file that
 * is missing, empty, damaged, unreadable, recovered from its backup, with or without the key, with a byte order mark).
 *
 * Quitting in the middle of the wizard leaves `onboardingComplete` false and `firstRunDecided` true, so it shows again
 * (and a later switch to LAN access is the user's choice: it is not decided again).
 * Pure: `Service_State.init` applies it between parsing the settings and creating the state proxy.
 */

import type { LanAccessMode } from "@/shared/services/peer/lanAccess";

/** The mode a new install starts with: this PC only. */
export const NEW_INSTALL_LAN_MODE: LanAccessMode = "localhost";

/** The settings the decision reads and writes. */
export interface FirstRunState {
  firstRunDecided: boolean;
  onboardingComplete: boolean;
  lanAccess: { mode: LanAccessMode };
}

/** How the settings file was found (a subset of `LoadResult` of `diagnostics/atomicJsonFile.ts`). */
export interface FirstRunLoad {
  source: "primary" | "backup" | "none";
  problems: readonly string[];
}

/**
 * True only when there was no settings file and nothing went wrong looking for one. A damaged or unreadable file
 * (which `AtomicJsonFile` reports as `source: "none"` plus a problem) belongs to an existing user.
 */
export function isFreshProfile(load: FirstRunLoad): boolean {
  return load.source === "none" && load.problems.length === 0;
}

/**
 * Takes the first-run decision once. Returns true when it changed anything (the caller then saves the settings).
 * A state that already has `firstRunDecided` is left exactly as it is.
 */
export function applyFirstRunDecision(state: FirstRunState, fresh: boolean): boolean {
  if (state.firstRunDecided) return false;
  state.firstRunDecided = true;
  if (fresh) state.lanAccess.mode = NEW_INSTALL_LAN_MODE;
  else state.onboardingComplete = true;
  return true;
}
