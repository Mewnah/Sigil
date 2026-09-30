import { zSafe } from "@/utils";
import { z } from "zod";

/** `openai_audio` presets: a user-run OpenAI-compatible server, or Groq's hosted Whisper (free plan, rate limited). */
export enum OpenAiAudioPreset {
  custom = "custom",
  groq = "groq",
}

/**
 * OpenAI-compatible `/v1/audio/transcriptions` settings (custom server or the Groq preset). Requests are sent by the
 * Rust speech pipeline's HTTP engine (keys only in the `Authorization` header), one per VAD utterance.
 */
export const STT_OpenAiAudio_Schema = z.object({
  device: zSafe(z.coerce.string(), ""),
  /** `custom` uses baseUrl/apiKey/model below; `groq` uses Groq's endpoint with groqApiKey/groqModel. */
  preset: zSafe(z.nativeEnum(OpenAiAudioPreset), OpenAiAudioPreset.custom),
  baseUrl: zSafe(z.coerce.string(), "http://127.0.0.1:8000/v1"),
  apiKey: zSafe(z.coerce.string(), ""),
  /** Must match the id configured on your local server (e.g. speaches); not sent to OpenAI cloud unless you point baseUrl there. */
  model: zSafe(z.coerce.string(), ""),
  language: zSafe(z.coerce.string(), ""),
  /** Groq API key (kept apart from `apiKey` so switching presets never sends one server's key to the other). */
  groqApiKey: zSafe(z.coerce.string(), ""),
  groqModel: zSafe(z.coerce.string(), "whisper-large-v3-turbo"),
  /** Optional `prompt` (vocabulary / spelling hint) sent with every request. */
  prompt: zSafe(z.coerce.string(), ""),
}).default({});
