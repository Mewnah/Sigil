/**
 * First-run decision: does this launch show the setup wizard?
 *
 * Every profile written before the wizard was mounted has `onboardingComplete` false, so that flag alone would show a
 * "first run" wizard to people who have used Sigil for months. The decision is therefore taken once, at the first
 * launch that knows about it (`firstRunDecided`):
 * - no settings file at all: a new install. `onboardingComplete` stays false, so the wizard shows (and the STT and TTS
 *   defaults treat the profile as fresh);
 * - anything else (a profile that loaded, was recovered from its backup, or could not be read): an existing user.
 *   `onboardingComplete` becomes true and the wizard never appears.
 *
 * Quitting in the middle of the wizard leaves `onboardingComplete` false and `firstRunDecided` true, so it shows again.
 * Pure: `Service_State.init` applies it between parsing the settings and creating the state proxy.
 */

/** The two settings the decision reads and writes. */
export interface FirstRunState {
  firstRunDecided: boolean;
  onboardingComplete: boolean;
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
  if (!fresh) state.onboardingComplete = true;
  return true;
}
