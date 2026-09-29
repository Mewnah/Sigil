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
  /** A partial hypothesis of recognized speech. Only transcript text: never status, progress or placeholders. */
  onInterim(value: string): void;
  /** A final transcript. Only transcript text. */
  onFinal(value: string): void;
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
