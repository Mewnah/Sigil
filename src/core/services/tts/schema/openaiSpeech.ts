import { zSafe, zStringNumber } from "@/utils";
import { z } from "zod";

/**
 * Server presets of the `openaiSpeech` backend (values in `services/openaiSpeech.ts` → `OPENAI_SPEECH_PRESETS`).
 * Persisted: add ids, never rename or remove them.
 */
export const OPENAI_SPEECH_PRESET_IDS = ["kokoroFastapi", "chatterbox", "voxtral", "indexTts", "voxcpm2", "qwentts", "custom"] as const;
export type OpenAiSpeechPresetId = (typeof OPENAI_SPEECH_PRESET_IDS)[number];

/** `response_format` values (Rust `openai_speech::SpeechFormat`). */
export const OPENAI_SPEECH_FORMATS = ["pcm", "wav"] as const;
export type OpenAiSpeechFormat = (typeof OPENAI_SPEECH_FORMATS)[number];

/**
 * Settings of the `openaiSpeech` backend: any OpenAI-compatible `/v1/audio/speech` server through the Rust
 * `openai-speech` plugin (`RustOpenAiSpeechRequest` in src/types.ts), streaming into the TTS sink. Owned by P2-I.
 *
 * - `preset`: the server template last applied in the panel (it fills URL, model, voice, format, rate, stream, extra);
 *   the fields stay editable, and `custom` changes nothing.
 * - `baseUrl`: including the API prefix; a URL without a path gets `/v1` (Rust `speech_base`).
 * - `apiKey`: optional, sent only in the `Authorization: Bearer` header; a preset that switches to a server with a
 *   different origin clears it (`applyOpenAiSpeechPreset`).
 * - `model` / `voice`: blank = not sent (the server's default).
 * - `format` / `pcmSampleRate`: raw 16-bit PCM (lowest latency; the rate must match the server) or WAV.
 * - `stream`: ask the server to stream (`"stream": true`; Rust never sends it to OpenAI's own API, which streams
 *   by default and has no such member).
 * - `extra`: a JSON object of server-specific request members (text; "" = none).
 * - `timeout`: seconds to the first audio, up to 600 (0 = automatic); the gap between two pieces of streamed audio
 *   is limited to 110 s either way (the TTS sink drops a sentence whose audio stops for 120 s).
 * - `device` / `volume`: the TTS sink target (the route is the top-level `route`).
 */
export const TTS_OpenAiSpeech_Schema = z.object({
  preset: zSafe(z.enum(OPENAI_SPEECH_PRESET_IDS), "kokoroFastapi"),
  /** Base URL including the API prefix. */
  baseUrl: zSafe(z.coerce.string(), "http://127.0.0.1:8880/v1"),
  /** Optional Bearer key (sent only in the Authorization header). */
  apiKey: zSafe(z.coerce.string(), ""),
  model: zSafe(z.coerce.string(), "kokoro"),
  voice: zSafe(z.coerce.string(), "af_heart"),
  speed: zSafe(zStringNumber(), "1.0"),
  format: zSafe(z.enum(OPENAI_SPEECH_FORMATS), "pcm"),
  pcmSampleRate: zSafe(zStringNumber(), "24000"),
  stream: zSafe(z.coerce.boolean(), true),
  extra: zSafe(z.coerce.string(), ""),
  timeout: zSafe(zStringNumber(), "0"),
  /** Output device (cpal name); "" = the system default output. */
  device: zSafe(z.coerce.string(), ""),
  /** Sink gain 0..2. */
  volume: zSafe(zStringNumber(), "1.0"),
}).default({});
