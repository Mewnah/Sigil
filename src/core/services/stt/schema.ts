import { zSafe, zStringNumber } from "@/utils";
import { z } from "zod";

export enum STT_Backends {
  native = "native",
  chrome = "chrome",
  edge = "edge",
  browser = "browser",
  azure = "azure",
  deepgram = "deepgram",
  whisper = "whisper",
  vosk = "vosk",
  moonshine = "moonshine",
  /** Local server: POST /v1/audio/transcriptions (same request shape as OpenAI; e.g. speaches, LocalAI). */
  openai_audio = "openai_audio",
}

export const zodSTT_Backends = z.nativeEnum(STT_Backends);

export const Service_STT_Schema = z.object({
  backend: zSafe(zodSTT_Backends, STT_Backends.native),
  autoStart: zSafe(z.coerce.boolean(), false),
  uwu: zSafe(z.coerce.boolean(), false),
  stopWithStream: zSafe(z.coerce.boolean(), false),
  replaceWords: zSafe(z.record(z.coerce.string(), z.coerce.string()), {}),
  replaceWordsIgnoreCase: zSafe(z.coerce.boolean(), false),
  replaceWordsPreserveCase: zSafe(z.coerce.boolean(), false),
  native: z.object({
    language_group: zSafe(z.coerce.string(), "English"),
    language: zSafe(z.coerce.string(), "en-US"),
  }).default({}),
  azure: z.object({
    device: zSafe(z.coerce.string(), "default"),
    language_group: zSafe(z.coerce.string(), "English"),
    language: zSafe(z.coerce.string(), "en-US"),
    secondary_language_group: zSafe(z.coerce.string(), ""),
    secondary_language: zSafe(z.coerce.string(), ""),
    use_secondary_language: zSafe(z.coerce.boolean(), true),
    key: zSafe(z.coerce.string(), ""),
    location: zSafe(z.coerce.string(), ""),
    profanity: zSafe(z.coerce.string(), "masked"),
    silenceTimeout: zSafe(zStringNumber(), "20"),
    interim: zSafe(z.coerce.boolean(), true),
  }).default({}),
  whisper: z.object({
    device: zSafe(z.coerce.string(), ""),
    model: zSafe(z.coerce.string(), "base.en"),
    language: zSafe(z.coerce.string(), "en"),
    vadEnabled: zSafe(z.coerce.boolean(), true),
    silenceThresholdDb: zSafe(zStringNumber(), "-40"),
    silenceDurationMs: zSafe(zStringNumber(), "1500"),
    minChunkDurationMs: zSafe(zStringNumber(), "1000"),
  }).default({}),
  deepgram: z.object({
    device: zSafe(z.coerce.string(), "default"),
    language_group: zSafe(z.coerce.string(), "English"),
    language: zSafe(z.coerce.string(), "en-US"),
    tier: zSafe(z.coerce.string(), ""),
    key: zSafe(z.coerce.string(), ""),
    punctuate: zSafe(z.coerce.boolean(), true),
    profanity: zSafe(z.coerce.boolean(), true),
    interim: zSafe(z.coerce.boolean(), true),
  }).default({}),
  vosk: z.object({
    device: zSafe(z.coerce.string(), ""),
    model: zSafe(z.coerce.string(), "vosk-model-small-en-us-0.15"),
    modelUrl: zSafe(z.coerce.string(), ""),
  }).default({}),
  moonshine: z.object({
    device: zSafe(z.coerce.string(), ""),
    endpoint: zSafe(z.coerce.string(), "http://localhost:8090"),
    language: zSafe(z.coerce.string(), "en"),
  }).default({}),
  openai_audio: z.object({
    device: zSafe(z.coerce.string(), ""),
    baseUrl: zSafe(z.coerce.string(), "http://127.0.0.1:8000/v1"),
    apiKey: zSafe(z.coerce.string(), ""),
    /** Must match the id configured on your local server (e.g. speaches); not sent to OpenAI cloud unless you point baseUrl there. */
    model: zSafe(z.coerce.string(), ""),
    language: zSafe(z.coerce.string(), ""),
  }).default({}),
}).default({});

export type STT_State = z.infer<typeof Service_STT_Schema>;

