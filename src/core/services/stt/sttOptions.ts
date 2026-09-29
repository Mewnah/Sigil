import type { RustVadConfig, RustWhisperDecodeOptions } from "@/types";
import { OpenAiAudioPreset, type STT_State } from "./schema";

/**
 * Persisted STT settings → the option objects sent to Rust and to HTTP engines. Pure (unit-tested): settings are
 * form strings, so everything is parsed, range-checked and dropped when unusable (Rust then uses its default).
 */

function toNumber(value: unknown): number | null {
  if (typeof value === "number") return Number.isFinite(value) ? value : null;
  const text = String(value ?? "").trim();
  if (!text) return null;
  const n = Number(text);
  return Number.isFinite(n) ? n : null;
}

function clampInt(value: unknown, min: number, max: number): number | undefined {
  const n = toNumber(value);
  return n === null ? undefined : Math.min(max, Math.max(min, Math.round(n)));
}

function clampFloat(value: unknown, min: number, max: number): number | undefined {
  const n = toNumber(value);
  return n === null ? undefined : Math.min(max, Math.max(min, Math.round(n * 1000) / 1000));
}

/** Gap between the speech-start and speech-end thresholds (Silero's reference 0.5 / 0.35). */
export const VAD_HYSTERESIS = 0.15;

/** `stt.vad` → Rust `VadConfig` (snake_case; omitted fields use the Rust defaults). */
export function buildVadConfig(vad: Partial<STT_State["vad"]> | undefined): RustVadConfig {
  const out: RustVadConfig = {};
  const threshold = clampFloat(vad?.threshold, 0.05, 0.95);
  if (threshold !== undefined) {
    out.threshold = threshold;
    out.neg_threshold = Math.max(0.01, Math.round((threshold - VAD_HYSTERESIS) * 1000) / 1000);
  }
  const minSilence = clampInt(vad?.minSilenceMs, 100, 5_000);
  if (minSilence !== undefined) out.min_silence_ms = minSilence;
  const preRoll = clampInt(vad?.preRollMs, 0, 2_000);
  if (preRoll !== undefined) out.pre_roll_ms = preRoll;
  const minSpeech = clampInt(vad?.minSpeechMs, 0, 5_000);
  if (minSpeech !== undefined) out.min_speech_ms = minSpeech;
  const maxUtterance = clampInt(vad?.maxUtteranceMs, 2_000, 60_000);
  if (maxUtterance !== undefined) out.max_utterance_ms = maxUtterance;
  return out;
}

/** Rust `whisper::WhisperDecodeOptions` (the `decode` argument of `plugin:whisper|start_recording`); see `@/types`. */
export type { RustWhisperDecodeOptions };

export const MAX_HOTWORDS = 50;
const MAX_HOTWORD_LENGTH = 64;

/** Hotwords from a comma / semicolon / newline separated list (also full-width and ideographic commas). */
export function parseHotwords(text: unknown): string[] {
  const seen = new Set<string>();
  const out: string[] = [];
  for (const raw of String(text ?? "").split(/[,;\n\r、，；]+/)) {
    const word = raw.replace(/\s+/g, " ").trim().slice(0, MAX_HOTWORD_LENGTH);
    const key = word.toLowerCase();
    if (!word || seen.has(key)) continue;
    seen.add(key);
    out.push(word);
    if (out.length >= MAX_HOTWORDS) break;
  }
  return out;
}

/** `stt.whisper` → Rust decode options. */
export function buildWhisperDecodeOptions(whisper: Partial<STT_State["whisper"]> | undefined): RustWhisperDecodeOptions {
  const out: RustWhisperDecodeOptions = {};
  const threads = clampInt(whisper?.threads, 0, 16);
  if (threads !== undefined && threads > 0) out.n_threads = threads;
  const prompt = String(whisper?.initialPrompt ?? "").trim();
  if (prompt) out.initial_prompt = prompt;
  const hotwords = parseHotwords(whisper?.hotwords);
  if (hotwords.length > 0) out.hotwords = hotwords;
  const noSpeech = clampFloat(whisper?.noSpeechThreshold, 0.05, 1);
  if (noSpeech !== undefined) out.no_speech_threshold = noSpeech;
  out.hallucination_filter = whisper?.hallucinationFilter !== false;
  const backlog = clampInt(whisper?.maxBacklog, 1, 10);
  if (backlog !== undefined) out.max_backlog = backlog;
  return out;
}

/**
 * Azure `Speech_SegmentationSilenceTimeoutMs` from `stt.azure.segmentationSilenceMs`: the silence that ends a phrase
 * (Azure accepts 100-5000 ms). `null` (0, blank or out of range) leaves the service default.
 */
export function azureSegmentationSilenceMs(value: unknown): number | null {
  const n = toNumber(value);
  if (n === null) return null;
  const ms = Math.round(n);
  return ms >= 100 && ms <= 5000 ? ms : null;
}

/** Whisper non-speech markers such as `[MUSIC]`, `[BLANK_AUDIO]` or `(water splashing)`. */
const WHISPER_NOISE_MARKER_RE = /\[[^\]]*\]|\([^)]*\)/g;

/** Removes Whisper non-speech markers and tidies the whitespace they leave. */
export function stripWhisperNoiseMarkers(text: string): string {
  return text
    .replace(WHISPER_NOISE_MARKER_RE, " ")
    .replace(/\s+([,.!?;:。、！？])/g, "$1")
    .replace(/\s+/g, " ")
    .trim();
}

// ---------------------------------------------------------------------------------------------------------------
// OpenAI-compatible transcription (`openai_audio`): local servers and the Groq preset.

export const GROQ_BASE_URL = "https://api.groq.com/openai/v1";
export const GROQ_MODELS = ["whisper-large-v3-turbo", "whisper-large-v3"] as const;
export const GROQ_DEFAULT_MODEL = GROQ_MODELS[0];
export const GROQ_KEYS_URL = "https://console.groq.com/keys";

/** True for Groq's API host (the preset, or a custom base URL typed by hand), so its rate limits always apply. */
export function isGroqUrl(url: string): boolean {
  try {
    const host = new URL(url.trim()).hostname.toLowerCase();
    return host === "api.groq.com" || host.endsWith(".groq.com");
  } catch {
    return false;
  }
}

/** Primary ISO-639 subtag (`en-US` → `en`); empty for blank or "auto". */
export function normalizeTranscriptionLanguage(language: string | undefined): string {
  const primary = String(language ?? "").trim().split(/[-_]/)[0].toLowerCase();
  if (!primary || primary === "auto") return "";
  return /^[a-z]{2,3}$/.test(primary) ? primary : "";
}

export interface ResolvedOpenAiAudio {
  preset: OpenAiAudioPreset;
  /** Base URL without a trailing slash. */
  baseUrl: string;
  apiKey: string;
  model: string;
  language: string;
  prompt: string;
  /** Groq's free-plan limits apply (pace requests, merge short utterances). */
  rateLimited: boolean;
}

export function resolveOpenAiAudioConfig(config: Partial<STT_State["openai_audio"]>): ResolvedOpenAiAudio {
  const preset = config.preset === OpenAiAudioPreset.groq ? OpenAiAudioPreset.groq : OpenAiAudioPreset.custom;
  const groq = preset === OpenAiAudioPreset.groq;
  const baseUrl = (groq ? GROQ_BASE_URL : String(config.baseUrl ?? "")).trim().replace(/\/+$/, "");
  return {
    preset,
    baseUrl,
    apiKey: String((groq ? config.groqApiKey : config.apiKey) ?? "").trim(),
    model: String((groq ? config.groqModel || GROQ_DEFAULT_MODEL : config.model) ?? "").trim(),
    language: normalizeTranscriptionLanguage(config.language),
    prompt: String(config.prompt ?? "").trim(),
    rateLimited: groq || isGroqUrl(baseUrl),
  };
}

export function transcriptionsUrl(baseUrl: string): string {
  return `${baseUrl.trim().replace(/\/+$/, "")}/audio/transcriptions`;
}
