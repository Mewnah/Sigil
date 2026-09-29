import { STT_State } from "./schema";

/**
 * Rust `audio:chunk`: interleaved f32 at the capture DEVICE's own `sample_rate` / `channels` (the rate passed to
 * `start_audio_capture` is not applied). Convert with `audioConvert.ts` before assuming mono 16 kHz.
 */
export interface RustAudioChunkPayload {
  samples: number[];
  sample_rate: number;
  channels: number;
}

export interface ISTTReceiver {
  onStart(): void;
  onStop(error?: string): void;
  onInterim(value: string): void;
  onFinal(value: string): void;
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
