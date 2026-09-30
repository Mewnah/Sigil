import { STT_State } from "./schema";

/**
 * Rust `audio:chunk`: interleaved f32 from the shared capture hub. `start_audio_capture({ sampleRate: 16000 })`
 * delivers 16 kHz mono; any other request delivers the device's own rate and channel count. Every payload states
 * its own `sample_rate` / `channels`, so downmix with `audioConvert.ts` when `channels > 1`.
 */
export interface RustAudioChunkPayload {
  samples: number[];
  sample_rate: number;
  channels: number;
}

/**
 * Revision metadata of a transcript, published on the `TextEvent` (`segmentId`, `revision`, `lang`, `turnEnd`; see
 * `TextEventSchema` in `@/types`). Set by engines that know segments (the `local` speech pipeline); every field is
 * optional and consumers must work without them.
 */
export interface SttTextMeta {
  /** Stable id of one spoken segment (`<pipelineId>-<process nonce>:<epoch>:<segmentId>`): interims and the final share it. */
  segmentId?: string;
  /** Rises with every interim and final of the segment; the highest wins. */
  revision?: number;
  /** Lower-case base language of the text (`en`, `ja`, `yue`, ...). */
  lang?: string;
  /** Smart Turn: the speaker probably finished their turn after this final (commit hint only). */
  turnEnd?: boolean;
}

/** Severity of an engine status line (styling only; errors that end a session go through `onStop`). */
export type SttStatusLevel = "info" | "warning";

export interface SttStatusOptions {
  /** 0-100 for a determinate progress bar; `null` / omitted = no bar. */
  progress?: number | null;
  level?: SttStatusLevel;
  /**
   * Transient line (e.g. "some speech was skipped"): cleared after this many ms unless a newer status replaced it.
   * Omitted = shown until the engine replaces or clears it.
   */
  expireMs?: number;
}

/**
 * Callbacks an engine uses to report to `Service_STT`. Transcript text and engine status are separate channels:
 * only `onInterim` / `onFinal` are ever published (captions, translation, chatbox, TTS); `onStatus` only reaches
 * the UI (`serviceState.statusText` / `progress`).
 */
export interface ISTTReceiver {
  onStart(): void;
  onStop(error?: string): void;
  /**
   * A partial hypothesis of recognized speech. Only transcript text: never status, progress or placeholders. `""`
   * abandons the open sentence (nothing is published). `meta` (speech-pipeline engines) is published with the text.
   */
  onInterim(value: string, meta?: SttTextMeta): void;
  /** A final transcript. Only transcript text. `meta` (speech-pipeline engines) is published with the text. */
  onFinal(value: string, meta?: SttTextMeta): void;
  /**
   * Engine status for the UI (connecting, model download / load progress, transient errors, rate-limit waits).
   * Never published. `""` clears it. While the session is connecting, every call also proves the engine is alive
   * (re-arms the connecting watchdog), so long first-time downloads are not cut off.
   */
  onStatus(text: string, options?: SttStatusOptions): void;
  /**
   * True while STT is muted: engines may skip work that would only be thrown away (e.g. billed requests). A pending
   * unmute (the user unmuted while an interim sentence was open) is not muted here: engines keep working so the
   * sentence can end, and engines without interims never get stuck waiting for one.
   */
  isMuted(): boolean;
}

export interface ISTTService {
  start(params: STT_State): void | Promise<void>;
  stop(): void | Promise<void>;
  dispose(): void | Promise<void>;
  /** Wired by Service_STT before each `start()`; engines may race awaits against this signal. */
  startAbortSignal?: AbortSignal | null;
  /**
   * Wired by Service_STT before each `start()`: true when nobody pressed Start just now (auto-start at launch).
   * Engines must not ask questions then (such as download consent); they stop with a message instead.
   */
  startAutomatic?: boolean;
  /**
   * Publish now, through the receiver, the finals of finished sentences the engine still holds back (the `local`
   * backend holds a first-pass final while it waits for the two-pass replacement, showing it as an interim).
   * Service_STT calls it synchronously right before it closes the open sentence with the "[...]" marker (mute, stop,
   * a new start), so a finished sentence is never replaced by the marker. Optional: engines that publish finals as
   * soon as they have them have nothing to commit.
   */
  commitPending?(): void;
}

export interface ISTTServiceConstructor {
  new (receiver: ISTTReceiver): ISTTService;
}

export interface ISpeechRecognitionService {
  start(params: STT_State): void | Promise<void>;
  stop(): void | Promise<void>;
  dispose(): void | Promise<void>;
  /** Wired by Service_STT before each `start()`; engines may race awaits against this signal. */
  startAbortSignal?: AbortSignal | null;
}

export const enum SttMuteState {
  muted,
  pendingUnmute,
  unmuted,
}
