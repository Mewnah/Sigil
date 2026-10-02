import { zSafe, zStringNumber } from "@/utils";
import { z } from "zod";

/**
 * `transcribe`: captions only (`text.stt`). `translate` (preview): Gemini Live Translate from the same audio, publishing
 * `text.stt` and `text.translation` (the translation service keeps its own settings; this mode is a combined backend).
 */
export const GEMINI_LIVE_MODES = ["transcribe", "translate"] as const;
export type GeminiLiveMode = (typeof GEMINI_LIVE_MODES)[number];

/**
 * Transcript style of Gemini Transcribe Live (`inputAudioTranscription.mode`): `verbatim` keeps every word as spoken
 * (the service default); `smart` removes filler words and formats numbers and punctuation.
 */
export const GEMINI_LIVE_FORMATTING = ["verbatim", "smart"] as const;

/** API version of the Live endpoint and its ephemeral-token endpoint (preview services move between them). */
export const GEMINI_LIVE_API_VERSIONS = ["v1beta", "v1alpha"] as const;

/** `rotateMinutes` bounds: sessions end after 10 minutes, and a rotation needs some time before that. */
export const GEMINI_LIVE_ROTATE_MIN = 1;
export const GEMINI_LIVE_ROTATE_MAX = 9.75;
export const GEMINI_LIVE_ROTATE_DEFAULT = 9.5;

/**
 * Settings of the `geminiLive` STT backend (P6-S0 scaffold; owned by P6-K): Gemini 3.5 Transcribe Live over a webview
 * WebSocket on the user's own key, rotating sessions every `rotateMinutes` with a short overlap and word dedupe so
 * captions never gap. Model ids are editable (preview names change). A missing key never blocks anything else.
 *
 * - `apiKey`: "" = use the translation settings' Gemini key (the shared LLM client's provider key).
 * - `device`: capture device (cpal name, "" = the default microphone), as the `local` backend.
 *
 * P6-K additions (new keys only): `interim`, `formatting`, `vocabulary`, `echoTargetLanguage`, `apiVersion`.
 */
export const STT_GeminiLive_Schema = z.object({
  apiKey: zSafe(z.coerce.string(), ""),
  model: zSafe(z.coerce.string(), "gemini-3.5-transcribe-live"),
  mode: zSafe(z.enum(GEMINI_LIVE_MODES), "transcribe"),
  translateModel: zSafe(z.coerce.string(), "gemini-3.5-live-translate-preview"),
  /** Spoken language (`auto` = detect). */
  language: zSafe(z.coerce.string(), "auto"),
  /** Target language of `translate` mode. */
  translateTo: zSafe(z.coerce.string(), "en"),
  device: zSafe(z.coerce.string(), ""),
  rotateMinutes: zSafe(zStringNumber(), "9.5"),
  /** Publish partial captions while a sentence is being spoken (finals are always published). */
  interim: zSafe(z.coerce.boolean(), true),
  /** Transcribe mode: `verbatim` or `smart` ({@link GEMINI_LIVE_FORMATTING}). */
  formatting: zSafe(z.enum(GEMINI_LIVE_FORMATTING), "verbatim"),
  /** Transcribe mode: names and terms to favour, separated by commas, semicolons or new lines (up to 1,000). */
  vocabulary: zSafe(z.coerce.string(), ""),
  /** Translate mode: speech already in the target language is repeated, so every line gets a translation caption. */
  echoTargetLanguage: zSafe(z.coerce.boolean(), true),
  apiVersion: zSafe(z.enum(GEMINI_LIVE_API_VERSIONS), "v1beta"),
}).default({});
