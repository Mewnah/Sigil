import { STT_Backends } from "./schema";
import { whisperLanguageFromBcp47 } from "./whisperModels";

/**
 * One-time desktop default migration. `native` (Web Speech) was the default backend but cannot work in the
 * desktop webview (it fails with a network error), so desktop profiles on it are moved once:
 * - a new install (no finished onboarding) starts on `local` (P2: Sigil picks and runs the speech models itself;
 *   the first Start asks before downloading them);
 * - an existing profile is offered `local` once (never a silent switch); when it declines (or already saw the
 *   offer) it moves to local Whisper behind the Silero VAD gate (P0), whose model downloads on the first Start.
 *
 * Chrome and Edge (the microphone page, the same Web Speech engines in a browser window) work, so a profile on them
 * keeps working whatever it answers: it gets the same one-time offer of the built-in engine, with no fallback, and
 * "Keep" records `localOfferShown` and is never asked again. It is judged on `localOfferShown` alone, not on
 * `desktopDefaultMigrated`, because that flag was already set for those profiles by the earlier "mark" step.
 * Pure: `Service_STT.init` applies the plan.
 */

export interface DesktopDefaultMigrationInput {
  /** Running in the Tauri desktop app (the web build keeps Web Speech, which works in browsers). */
  isDesktop: boolean;
  /** `stt.desktopDefaultMigrated`. */
  alreadyMigrated: boolean;
  backend: string;
  /** Web Speech language of the native backend (BCP-47, e.g. "ja-JP"). */
  nativeLanguage?: string;
  /** Current Whisper settings: kept when the user already chose them. */
  whisperModel?: string;
  whisperLanguage?: string;
  /** `navigator.hardwareConcurrency` (logical processors). */
  logicalCores?: number;
  /** No finished onboarding yet (`onboardingComplete` false): a new install. */
  freshInstall?: boolean;
  /** `stt.localOfferShown`: the one-time offer of the `local` backend was already shown. */
  localOfferShown?: boolean;
}

/** Which engine the one-time "switch to the built-in engine?" prompt is about; it picks the wording and the buttons. */
export type LocalOfferKind = "native" | "chrome" | "edge";

export type DesktopDefaultMigration =
  /** Nothing to do and nothing to record (web build, or already migrated). */
  | { action: "none" }
  /** Record that the migration ran, change nothing (the user is on a working backend). */
  | { action: "mark" }
  /** Switch to Whisper (model / language only when the Whisper settings are still the defaults). */
  | WhisperMigration
  /** New install: switch to the `local` backend (its language stays "auto"). */
  | { action: "local"; backend: STT_Backends.local }
  /** Existing profile on `native` (does not work in the desktop webview): offer `local` once; apply `fallback` (Whisper) when declined. */
  | { action: "offer"; kind: "native"; fallback: WhisperMigration }
  /** Existing profile on Chrome or Edge (works): offer `local` once; declining changes nothing. */
  | { action: "offer"; kind: "chrome" | "edge"; fallback: null };

export type WhisperMigration = { action: "migrate"; backend: STT_Backends.whisper; whisperModel?: string; whisperLanguage?: string };

/** Schema defaults of `stt.whisper.model` / `language`: treated as "never chosen". */
const WHISPER_DEFAULT_MODEL = "base.en";
const WHISPER_DEFAULT_LANGUAGE = "en";

/**
 * Physical cores from the logical count. Browsers expose only logical processors; desktop CPUs with 8+ threads
 * almost always run 2 threads per core, smaller counts are usually one per core. Errs low, which only picks the
 * lighter model.
 */
export function estimatePhysicalCores(logicalCores: number | undefined): number {
  const logical = Number.isFinite(logicalCores) && (logicalCores as number) > 0 ? Math.floor(logicalCores as number) : 4;
  return logical >= 8 ? Math.floor(logical / 2) : logical;
}

/**
 * small (Q5_1) with 6+ physical cores, else base (Q5_1): fast enough next to a game on CPU. English uses the `.en`
 * build, which is more accurate for English at these sizes.
 */
export function pickDesktopWhisperModel(physicalCores: number, language: string): string {
  const size = physicalCores >= 6 ? "small" : "base";
  return language === "en" ? `${size}.en-q5_1` : `${size}-q5_1`;
}

export function planDesktopDefaultMigration(input: DesktopDefaultMigrationInput): DesktopDefaultMigration {
  if (!input.isDesktop) return { action: "none" };
  if (input.backend === STT_Backends.chrome || input.backend === STT_Backends.edge) {
    // A working engine: offer the built-in one once to an existing profile, never switch it silently. A new install
    // (onboarding not finished) that is on one has just chosen it, and an offer already answered is not repeated.
    if (!input.freshInstall && !input.localOfferShown) return { action: "offer", kind: input.backend, fallback: null };
    return input.alreadyMigrated ? { action: "none" } : { action: "mark" };
  }
  if (input.alreadyMigrated) return { action: "none" };
  if (input.backend !== STT_Backends.native && input.backend !== STT_Backends.browser) return { action: "mark" };
  if (input.freshInstall) return { action: "local", backend: STT_Backends.local };
  const fallback = planWhisperMigration(input);
  return input.localOfferShown ? fallback : { action: "offer", kind: "native", fallback };
}

/** The prompt kind for the backend a profile is on, or null when no offer applies to it (`offerLocalBackend`). */
export function localOfferKindOf(backend: string): LocalOfferKind | null {
  if (backend === STT_Backends.chrome) return "chrome";
  if (backend === STT_Backends.edge) return "edge";
  return backend === STT_Backends.native || backend === STT_Backends.browser ? "native" : null;
}

/** P0 migration of a profile on `native` to local Whisper (small / base Q5_1 by core count). */
export function planWhisperMigration(input: DesktopDefaultMigrationInput): WhisperMigration {
  const whisperUntouched =
    (input.whisperModel ?? WHISPER_DEFAULT_MODEL) === WHISPER_DEFAULT_MODEL &&
    (input.whisperLanguage ?? WHISPER_DEFAULT_LANGUAGE) === WHISPER_DEFAULT_LANGUAGE;
  if (!whisperUntouched) return { action: "migrate", backend: STT_Backends.whisper };

  const language = whisperLanguageFromBcp47(input.nativeLanguage || "en");
  return {
    action: "migrate",
    backend: STT_Backends.whisper,
    whisperModel: pickDesktopWhisperModel(estimatePhysicalCores(input.logicalCores), language),
    whisperLanguage: language,
  };
}

/**
 * Backends removed from the product. A desktop profile still on one moves to the built-in engine (`local`, automatic)
 * once at launch, with a notice; Moonshine server mode is the first. The web build is untouched (it has no in-process
 * engine to move to). Pure: `Service_STT.init` applies the plan.
 */
export type RetiredBackendMigration = { action: "none" } | { action: "local" };

export function planRetiredBackendMigration(input: { isDesktop: boolean; backend: string }): RetiredBackendMigration {
  return input.isDesktop && input.backend === STT_Backends.moonshine ? { action: "local" } : { action: "none" };
}
