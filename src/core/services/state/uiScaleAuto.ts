/**
 * Interface size Auto: which profiles move to it. The decision is taken once, at the first launch that knows about it
 * (`uiScaleAutoDecided`), like the first-run decision (`firstRun.ts`):
 * - a profile that had picked S, L or XL (`uiScale` is not 1) keeps that size: Auto is turned off for it;
 * - any other profile, the default M (1) and a new install alike, uses Auto. Where the monitor needs no larger interface
 *   (1080p or 1440p, or a 4K monitor at 150% Windows scale or more) Auto is 1, so nothing changes for those users.
 *
 * Nothing is decided again afterwards: a user who later picks Auto or a size keeps it. Pure: `Service_State.init`
 * applies it between parsing the settings and creating the state proxy.
 */

/** The settings the decision reads and writes. */
export interface UiScaleAutoState {
  uiScale: number;
  uiScaleAuto: boolean;
  uiScaleAutoDecided: boolean;
}

/** M, the size a profile has when it never picked one. */
const DEFAULT_MANUAL_SCALE = 1;
/** Tolerance for "is 1" (a hand-edited 1.0 or a value that went through a slider). */
const SAME_SCALE_EPSILON = 0.001;

/**
 * Takes the Auto decision once. Returns true when it changed anything (the caller then saves the settings). A state that
 * already has `uiScaleAutoDecided` is left exactly as it is.
 */
export function applyUiScaleAutoDecision(state: UiScaleAutoState): boolean {
  if (state.uiScaleAutoDecided) return false;
  state.uiScaleAutoDecided = true;
  state.uiScaleAuto = Math.abs(state.uiScale - DEFAULT_MANUAL_SCALE) <= SAME_SCALE_EPSILON;
  return true;
}
