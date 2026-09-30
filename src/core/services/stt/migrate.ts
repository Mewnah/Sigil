import { STT_Backends } from "./schema";
import { whisperLanguageFromBcp47 } from "./whisperModels";

/**
 * One-time desktop default migration. `native` (Web Speech) was the default backend but cannot work in the
 * desktop webview (it fails with a network error), so desktop profiles on it are moved once:
 * - a new install (no finished onboarding) starts on `local` (P2: Sigil picks and runs the speech models itself;
 *   the first Start asks before downloading them);
 * - an existing profile is offered `local` once (never a silent switch); when it declines (or already saw the
 *   offer) it moves to local Whisper behind the Silero VAD gate (P0), whose model downloads on the first Start.
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

export type DesktopDefaultMigration =
  /** Nothing to do and nothing to record (web build, or already migrated). */
  | { action: "none" }
  /** Record that the migration ran, change nothing (the user is on a working backend). */
  | { action: "mark" }
  /** Switch to Whisper (model / language only when the Whisper settings are still the defaults). */
  | WhisperMigration
  /** New install: switch to the `local` backend (its language stays "auto"). */
  | { action: "local"; backend: STT_Backends.local }
  /** Existing profile: offer `local` once; apply `fallback` (Whisper) when the offer is declined. */
  | { action: "offer"; fallback: WhisperMigration };

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
  if (!input.isDesktop || input.alreadyMigrated) return { action: "none" };
  if (input.backend !== STT_Backends.native && input.backend !== STT_Backends.browser) return { action: "mark" };
  if (input.freshInstall) return { action: "local", backend: STT_Backends.local };
  const fallback = planWhisperMigration(input);
  return input.localOfferShown ? fallback : { action: "offer", fallback };
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
