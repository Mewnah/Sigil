import z from "zod";
import { ElementType } from "./client/elements/schema";
import { Services } from "./services-registry";

export type MappedGroupDictionary<Options = any> = Record<string, ([string] | [string, string] | [string, string, Options])[]>

export interface IServiceInterface {
  init(): void;
}

export const enum ServiceNetworkState {
  disconnected = "disconnected",
  connecting = "connecting",
  connected = "connected",
  error = "error"
}

export enum TextEventType {
  final,
  interim,
}

const TextEventTypeSchema = z.nativeEnum(TextEventType);

export type BaseEvent<Data = unknown> = {
  topic: string,
  data?: Data
}

export type PartialWithRequired<T, K extends keyof T> = Pick<T, K> & Partial<T>;

export const TextEventSchema = z.object({
  type: TextEventTypeSchema,
  value: z.string().default(""),
  emotes: z.record(z.number().or(z.string()), z.string()).default({}),
  textFieldType: z.enum(["textField", "twitchChat", "kickChat"]).nullish(),
  id: z.number().optional()
});

export type TextEvent = z.infer<typeof TextEventSchema>;

export enum TextEventSource {
  any = "text",
  textfield = "text.textfield",
  stt = "text.stt",
  translation = "text.translation",
  transform = "text.transform",
  transform_source = "text.transform_source",
  transform_raw = "text.transform_raw",
}

export const TextEventSourceSchema = z.nativeEnum(TextEventSource);

export const ObsCaptionEnvelopeSchema = z.object({
  v: z.literal(1),
  source: TextEventSourceSchema,
  type: z.enum(["final", "interim"]),
  value: z.string(),
  at: z.number(),
});

export type ObsCaptionEnvelope = z.infer<typeof ObsCaptionEnvelopeSchema>;

export type InspectorTabPath = {
  tab: Services | ElementType | "settings" | "integrations" | "scenes" | "files" | "fonts" | "project" | "elements",
  value?: string
}

/* ------------------------------------------------------------------------------------------------
 * Frozen Rust <-> TS contracts (P0-S0). Change only through an INTEGRATE step, together with the
 * Rust side named in each comment.
 * ---------------------------------------------------------------------------------------------- */

/**
 * VAD segmentation settings sent as `vad` to `plugin:audio|start_utterance_capture`
 * (Rust `speech_core::VadConfig`, snake_case; every field optional, Rust fills defaults).
 */
export interface RustVadConfig {
  /** Speech starts at this frame probability (default 0.5). */
  threshold?: number;
  /** Speech ends after `min_silence_ms` below this (default 0.35). */
  neg_threshold?: number;
  /** Audio kept before the detected start (default 400). */
  pre_roll_ms?: number;
  /** Silence that closes an utterance (default 500). */
  min_silence_ms?: number;
  /** Shorter speech is discarded (default 250), except the voiced remainder of a forced split. */
  min_speech_ms?: number;
  /** Forced close / split length (default 18000). */
  max_utterance_ms?: number;
}

/**
 * Payload of the Tauri event `audio:utterance` (Rust `speech_core::UtteranceEvent`). Metadata only:
 * fetch the PCM of an `utterance` once with
 * `new Float32Array(await invoke<ArrayBuffer>("plugin:audio|take_utterance", { captureId, seq }))`
 * (little-endian f32 mono at `sample_rate`). Ignore events whose `capture_id` is not the id returned
 * by `plugin:audio|start_utterance_capture`.
 */
export type RustUtteranceEvent =
  | { kind: "speech_start"; capture_id: number; seq: number; start_ms: number }
  | {
      kind: "utterance";
      capture_id: number;
      seq: number;
      start_ms: number;
      end_ms: number;
      sample_rate: number;
      samples: number;
      forced: boolean;
    }
  | { kind: "dropped"; capture_id: number; seq: number; reason: string }
  | { kind: "error"; capture_id: number; message: string };

/**
 * Rust `whisper::WhisperDecodeOptions`: the optional `decode` argument of `plugin:whisper|start_recording`
 * (snake_case like `RustVadConfig`). Every field is optional; omitted fields use the Rust defaults.
 */
export interface RustWhisperDecodeOptions {
  /** Decoder threads (default: half the logical CPUs, 1–4). */
  n_threads?: number;
  /** Context prompt, tokenized once per model. */
  initial_prompt?: string;
  /** Words Whisper should favour; passed together with the initial prompt. */
  hotwords?: string[];
  /** Utterances whose no-speech probability is above this are dropped (default 0.6; 1 disables). */
  no_speech_threshold?: number;
  /** Curated hallucination / repeat filter (default true). */
  hallucination_filter?: boolean;
  /** Utterances waiting for the decoder before the oldest is dropped (1..10, default 3). */
  max_backlog?: number;
}

/** Payload of the Tauri event `whisper:speech_start` (the Silero gate confirmed speech). */
export interface RustWhisperSpeechStart {
  seq: number;
  start_ms: number;
}

/** Payload of `whisper:utterance_dropped`: the decoder fell behind and skipped the oldest waiting utterance. */
export interface RustWhisperUtteranceDropped {
  seq: number;
  audio_ms: number;
  reason: "backlog" | (string & {});
}

/** Payload of `whisper:capture_error`: the input device failed and the recording ended. */
export interface RustWhisperCaptureError {
  message: string;
}

/**
 * Execution provider requested vs. actually in use (Rust `diagnostics::EpOutcome`). EP ids are
 * lower-case: "cpu" | "cuda" | "dml" (later "tensorrt", "nvrtx"). `reason` explains a fallback.
 */
export interface RustEpOutcome {
  requested: string;
  active: string;
  reason: string | null;
}

/** Row of `plugin:diagnostics|get_ep_outcomes` / event `diagnostics:ep_outcome`. */
export interface RustEpOutcomeEntry extends RustEpOutcome {
  /** "vc", "vc.content", "vc.pitch", "vc.onnx", "tts.kitten", "stt.vad", ... */
  component: string;
  at_unix_ms: number;
}
