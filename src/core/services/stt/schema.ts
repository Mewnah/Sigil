import { zSafe, zStringNumber } from "@/utils";
import { z } from "zod";
import { normalizeDeepgramLanguage } from "./stt_data";

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

/** `openai_audio` presets: a user-run OpenAI-compatible server, or Groq's hosted Whisper (free plan, rate limited). */
export enum OpenAiAudioPreset {
  custom = "custom",
  groq = "groq",
}

export const Service_STT_Schema = z.object({
  backend: zSafe(zodSTT_Backends, STT_Backends.native),
  /**
   * Set once the one-time desktop default migration ran (`native`, which cannot work in the desktop webview, is
   * switched to local Whisper). Never reset, so a later deliberate choice of `native` is kept.
   */
  desktopDefaultMigrated: zSafe(z.coerce.boolean(), false),
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
    /** Silence (ms) that ends a phrase; lower = faster finals. Valid 100-5000; 0 = service default. */
    segmentationSilenceMs: zSafe(zStringNumber(), "500"),
    interim: zSafe(z.coerce.boolean(), true),
  }).default({}),
  /**
   * Silero VAD settings shared by the engines whose audio Sigil segments in Rust (Whisper, Moonshine, HTTP / Groq).
   * Strings (form inputs); `sttOptions.buildVadConfig` validates them and falls back to the Rust defaults.
   */
  vad: z.object({
    /** Speech-start probability (0.05-0.95); the end threshold is derived (0.15 lower). */
    threshold: zSafe(zStringNumber(), "0.5"),
    /** Silence that closes an utterance (100-5000 ms). */
    minSilenceMs: zSafe(zStringNumber(), "500"),
    /** Audio kept before the detected speech start (0-2000 ms). */
    preRollMs: zSafe(zStringNumber(), "400"),
    /** Shorter speech bursts are ignored (0-5000 ms). */
    minSpeechMs: zSafe(zStringNumber(), "250"),
    /** Longer speech is split here (2000-60000 ms). */
    maxUtteranceMs: zSafe(zStringNumber(), "18000"),
  }).default({}),
  whisper: z.object({
    device: zSafe(z.coerce.string(), ""),
    model: zSafe(z.coerce.string(), "base.en"),
    language: zSafe(z.coerce.string(), "en"),
    /** Legacy RMS gate settings (pre-Silero). Kept so old settings parse; no longer sent or shown. */
    vadEnabled: zSafe(z.coerce.boolean(), true),
    silenceThresholdDb: zSafe(zStringNumber(), "-40"),
    silenceDurationMs: zSafe(zStringNumber(), "1500"),
    minChunkDurationMs: zSafe(zStringNumber(), "1000"),
    /** When true, `localModelPath` is copied into app data instead of downloading from Hugging Face. */
    useLocalModel: zSafe(z.coerce.boolean(), false),
    localModelPath: zSafe(z.coerce.string(), ""),
    /** Inference threads; "0" = automatic (about half the cores, at most 4, so games keep their CPU). */
    threads: zSafe(zStringNumber(), "0"),
    /** Decoder prompt: style / vocabulary hint (ignored by presets that do not take prompts). */
    initialPrompt: zSafe(z.coerce.string(), ""),
    /** Comma- or newline-separated names and terms to bias recognition towards. */
    hotwords: zSafe(z.coerce.string(), ""),
    /** Drop an utterance whose no-speech probability is above this (0.05-1; 1 disables). */
    noSpeechThreshold: zSafe(zStringNumber(), "0.6"),
    /** Curated whole-utterance hallucination phrases and repeat-loop removal. */
    hallucinationFilter: zSafe(z.coerce.boolean(), true),
    /** Utterances that may wait for the model before the oldest is skipped (1-10). */
    maxBacklog: zSafe(zStringNumber(), "3"),
  }).default({}),
  deepgram: z.object({
    device: zSafe(z.coerce.string(), "default"),
    language_group: zSafe(z.coerce.string(), "English"),
    /** Deepgram `language=`; legacy `zn-CN` / `zn-TW` (typo) load as `zh-CN` / `zh-TW`. */
    language: zSafe(z.coerce.string().transform(normalizeDeepgramLanguage), "en-US"),
    /** Legacy (pre-Nova) quality tier; unused once `model` is set. */
    tier: zSafe(z.coerce.string(), ""),
    /** Deepgram model id sent as `model=` (e.g. nova-3, nova-2). */
    model: zSafe(z.coerce.string(), "nova-3"),
    key: zSafe(z.coerce.string(), ""),
    punctuate: zSafe(z.coerce.boolean(), true),
    profanity: zSafe(z.coerce.boolean(), true),
    interim: zSafe(z.coerce.boolean(), true),
  }).default({}),
  vosk: z.object({
    device: zSafe(z.coerce.string(), ""),
    model: zSafe(z.coerce.string(), "vosk-model-small-en-us-0.15"),
    modelUrl: zSafe(z.coerce.string(), ""),
    /** When set, this zip is copied into app data instead of downloading from the catalog URL. */
    useLocalZip: zSafe(z.coerce.boolean(), false),
    localZipPath: zSafe(z.coerce.string(), ""),
  }).default({}),
  moonshine: z.object({
    device: zSafe(z.coerce.string(), ""),
    endpoint: zSafe(z.coerce.string(), "http://localhost:8090"),
    language: zSafe(z.coerce.string(), "en"),
  }).default({}),
  openai_audio: z.object({
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
  }).default({}),
}).default({});

export type STT_State = z.infer<typeof Service_STT_Schema>;

