import type { RustTtsSinkTarget } from "@/types";
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

/**
 * One block of synthesized audio (ITTSService v2 PCM contract): f32 samples in [-1, 1], interleaved when
 * `channels > 1`. Chunks of one sentence arrive in order; the sink resamples any rate to the device.
 */
export interface PcmChunk {
  samples: Float32Array;
  sampleRate: number;
  channels: number;
}

/**
 * Where a sentence goes when an in-process (Rust) engine speaks it straight into the TTS sink
 * ({@link ITTSService.synthesizeToSink}). The pipeline has already begun the sink utterance `utteranceId` with
 * `target` (in sentence order) and ends or cancels it itself; the engine only pushes audio into it.
 */
export interface TtsSinkContext {
  /** Id of the begun `tts_sink` utterance for this sentence. */
  utteranceId: string;
  /** Route / device / volume the utterance was begun with (informational for the engine). */
  target: RustTtsSinkTarget;
  /**
   * Language of the text this sentence came from (the TextEvent's `lang`: lower-case BCP-47 base), when the
   * publisher tagged it. Engines may prefer it over guessing from the script.
   */
  lang?: string | null;
}

/**
 * A TTS backend.
 *
 * # ITTSService v2: the PCM contract (frozen by P2-S0; change only through an INTEGRATE step)
 * A backend produces audio in one of three ways; the sentence pipeline (P2-G, `pipeline.ts`) picks the first one the
 * backend implements, in this order:
 * 1. {@link synthesizeToSink}: an in-process Rust engine (local-tts, openai-speech) speaks one sentence straight into
 *    the begun sink utterance (no PCM crosses IPC).
 * 2. {@link synthesizePcm}: the backend yields {@link PcmChunk}s for one sentence; the pipeline pushes them into the sink
 *    utterance with the binary `plugin:tts-sink|push_pcm` command.
 * 3. Legacy {@link play}: the backend plays by itself. Backends that can only do this set {@link selfPlaying} (Web
 *    Speech; SAPI until P6), and the pipeline passes them whole finals without sentence splitting or sink routing.
 *
 * Rules for 1 and 2:
 * - One call per sentence. The pipeline may run the calls for the next N sentences concurrently (synthesize ahead);
 *   playback order is the sink's begin order, never the completion order.
 * - `signal` aborts on stop / skip / flush: stop producing audio promptly, then settle — `synthesizePcm` returns (or
 *   throws an `AbortError`), `synthesizeToSink` resolves or rejects; an aborted call never reports `onError`.
 * - Failures of one sentence reject / throw with a readable message; the pipeline reports it through `onError` and
 *   carries on with the next sentence. Fatal failures still call `onStop(error)`.
 * - Output device, volume and route (device / voice changer / both) are the sink's job: v2 backends never open audio
 *   outputs themselves.
 * - `start` / `stop` / `dispose` keep their v1 meaning; `play` stays for the legacy path and for backends not yet
 *   migrated (the pipeline wraps them with an adapter).
 */
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
  /** v2: the backend plays audio itself (only {@link play} works; no sentence pipeline, no sink routing). */
  readonly selfPlaying?: boolean;
  /**
   * v2: synthesize one sentence into PCM chunks, in order (see the rules above). Named `synthesizePcm` (not the
   * plan's `synthesize`) because the retired clip backends had a protected `synthesize(text)` hook.
   */
  synthesizePcm?(text: string, signal: AbortSignal): AsyncIterable<PcmChunk>;
  /** v2: speak one sentence straight into the begun sink utterance `sink.utteranceId` (Rust engines). */
  synthesizeToSink?(text: string, sink: TtsSinkContext, signal: AbortSignal): Promise<void>;
}

/** How the sentence pipeline drives a backend (the first capability it implements; see {@link ITTSService}). */
export type TtsOutputMode = "sink" | "pcm" | "selfPlaying";

/** The output mode the pipeline uses for `service`. */
export function ttsOutputMode(service: ITTSService): TtsOutputMode {
  if (typeof service.synthesizeToSink === "function") return "sink";
  if (typeof service.synthesizePcm === "function") return "pcm";
  return "selfPlaying";
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
