import { TTS_Backends } from "./schema";

/**
 * P2 desktop default for TTS: new desktop installs speak with the in-process Kokoro voice (`localNeural`, the local
 * TTS default of the speech-stack plan); existing profiles are never switched silently: profiles still on one of
 * the old default backends (Web Speech, Windows SAPI) get a one-time offer instead. Nothing downloads until the
 * user agrees in the Kokoro panel (`<ModelRequirement>`). Pure: `Service_TTS.init` applies the plan.
 *
 * Only people who use TTS are asked (most users only run captions): a profile that shows no sign of TTS use
 * ({@link profileUsesTts}) waits, unmarked, and is asked at the first launch after it starts TTS.
 */

export interface TtsDesktopDefaultInput {
  /** Running in the Tauri desktop app (`localNeural` needs the Rust `local-tts` plugin). */
  isDesktop: boolean;
  /** `tts.desktopDefaultMigrated`. */
  alreadyMigrated: boolean;
  /** No finished onboarding yet (`onboardingComplete` false): a new install. */
  freshInstall: boolean;
  backend: string;
  /** `tts.localOfferShown`. */
  localOfferShown: boolean;
  /** The profile uses TTS ({@link profileUsesTts}). */
  usesTts: boolean;
}

export type TtsDesktopDefault =
  /** Nothing to do and nothing to record (web build, or already done). */
  | { action: "none" }
  /** Record that the default was handled, change nothing. */
  | { action: "mark" }
  /** New install: switch to `localNeural` (and record it). */
  | { action: "local" }
  /** Existing profile on an old default backend: record it, then ask once whether to switch. */
  | { action: "offer" }
  /** Existing profile on an old default backend that has not used TTS: record nothing, decide again next launch. */
  | { action: "wait" };

/** Backends a profile may be on without having chosen them (the schema default and the old desktop pick). */
const OLD_DEFAULT_BACKENDS: ReadonlySet<string> = new Set([TTS_Backends.native, TTS_Backends.windows]);

export function planTtsDesktopDefault(input: TtsDesktopDefaultInput): TtsDesktopDefault {
  if (!input.isDesktop || input.alreadyMigrated) return { action: "none" };
  if (input.freshInstall && input.backend === TTS_Backends.native) return { action: "local" };
  if (OLD_DEFAULT_BACKENDS.has(input.backend) && !input.localOfferShown)
    return input.usesTts ? { action: "offer" } : { action: "wait" };
  return { action: "mark" };
}

/** The TTS settings {@link profileUsesTts} reads. */
export interface TtsUsageSignals {
  everStarted: boolean;
  autoStart: boolean;
  backend: string;
  native?: { voice?: string };
  windows?: { voice?: string };
}

/**
 * Whether a profile uses TTS: it has started at least once since Sigil started recording that (`everStarted`), it
 * starts with the app (`autoStart`), or a voice was picked for its Web Speech / SAPI backend (profiles from before
 * `everStarted` existed show their use that way).
 */
export function profileUsesTts(data: TtsUsageSignals): boolean {
  if (data.everStarted || data.autoStart) return true;
  if (data.backend === TTS_Backends.native) return !!data.native?.voice;
  if (data.backend === TTS_Backends.windows) return !!data.windows?.voice;
  return false;
}
