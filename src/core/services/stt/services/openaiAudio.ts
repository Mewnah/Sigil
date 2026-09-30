import i18n from "i18next";
import type { RustSpeechEngineSpec } from "@/types";
import { OpenAiAudioPreset, STT_State } from "../schema";
import { normalizeEndpointUrl, RustUtteranceService, type PipelineEngineSetup } from "../rustCapturePolling";
import { resolveOpenAiAudioConfig, stripWhisperNoiseMarkers, type ResolvedOpenAiAudio } from "../sttOptions";

/**
 * `openai_audio` settings → the Rust HTTP engine (`speech::engines::http`), or the message to stop with. Pure.
 *
 * - Groq preset: Rust uses Groq's endpoint, paces requests to the free plan (20 / min, 2,000 / day, 7,200 audio s /
 *   hour, 10 s billing minimum) with one limiter for the whole app run, and merges short utterances when the daily
 *   pace asks for it. The key is required.
 * - Custom server: base URL and model are required; the key is optional. A Groq URL typed by hand is paced too.
 *
 * The key only ever travels in `options.apiKey` (Rust puts it in the `Authorization` header; never logged).
 */
export function openAiAudioSetup(config: ResolvedOpenAiAudio): PipelineEngineSetup | string {
  const groq = config.preset === OpenAiAudioPreset.groq;
  const options: Record<string, unknown> = {};
  if (groq) {
    if (!config.apiKey) return i18n.t("stt.error_groq_key_missing");
    options.preset = "groq";
    options.model = config.model;
  } else {
    if (!config.baseUrl) return i18n.t("stt.error_http_base_url");
    const baseUrl = normalizeEndpointUrl(config.baseUrl);
    if (!baseUrl) return i18n.t("stt.error_http_base_url_invalid", { url: config.baseUrl });
    if (!config.model) return i18n.t("stt.error_http_model");
    // Groq's API typed as a custom server: it rejects every request without a key.
    if (config.rateLimited && !config.apiKey) return i18n.t("stt.error_groq_key_missing");
    options.preset = "openai";
    options.baseUrl = baseUrl;
    options.model = config.model;
  }
  if (config.apiKey) options.apiKey = config.apiKey;
  const engine: RustSpeechEngineSpec = { engine: "http", model: "", options };
  return {
    engine,
    language: config.language || "auto",
    prompt: config.prompt,
  };
}

/**
 * OpenAI-compatible `POST {base}/audio/transcriptions`: a user-run server (speaches, LocalAI, faster-whisper-server,
 * whisper.cpp server; on this PC or the LAN) or the Groq preset (whisper-large-v3-turbo on the free plan). One request
 * per VAD utterance, sent by Sigil's Rust speech pipeline; see {@link RustUtteranceService}.
 */
export class STT_OpenAI_AudioService extends RustUtteranceService {
  protected readonly logTag = "[HTTP STT]";
  #config: ResolvedOpenAiAudio | null = null;

  protected engineLabel(): string {
    return this.#config?.rateLimited ? "Groq" : i18n.t("stt.provider_openai_audio");
  }

  protected inputDevice(params: STT_State): string {
    return params.openai_audio.device;
  }

  protected prepare(params: STT_State): PipelineEngineSetup | string {
    const config = resolveOpenAiAudioConfig(params.openai_audio);
    this.#config = config;
    return openAiAudioSetup(config);
  }

  protected postProcess(text: string): string {
    return stripWhisperNoiseMarkers(text);
  }
}
