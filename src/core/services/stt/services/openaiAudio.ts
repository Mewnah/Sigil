import i18n from "i18next";
import { floatSamplesToWavBlob } from "../encodeWav";
import { OpenAiAudioPreset, STT_State } from "../schema";
import { fetchWithTimeout } from "@/utils/fetchWithTimeout";
import { isAbortError, raceWithTimeoutAndAbort } from "../abort";
import { GroqRateLimiter, parseRetryAfterMs, type RequestLimiter } from "../rateLimiter";
import { RUST_CAPTURE_SAMPLE_RATE, RustUtteranceService, truncateForUi } from "../rustCapturePolling";
import {
  resolveOpenAiAudioConfig,
  stripWhisperNoiseMarkers,
  transcriptionsUrl,
  type ResolvedOpenAiAudio,
} from "../sttOptions";
import { TranscriptionError, httpTranscriptionError } from "../utterancePipeline";

/** Avoid a hung `fetch` when a local server stops responding (large local models can be slow). */
const LOCAL_STT_FETCH_TIMEOUT_MS = 120_000;
/** Groq answers in well under a second; a minute means the request is lost. */
const CLOUD_STT_FETCH_TIMEOUT_MS = 60_000;
const MAX_ERROR_DETAIL = 160;

/**
 * One limiter for every Groq session in this app run: stopping and restarting STT must not reset the request
 * history, or a quick restart could exceed the per-minute limit.
 */
let groqLimiter: GroqRateLimiter | null = null;
function sharedGroqLimiter(): GroqRateLimiter {
  groqLimiter ??= new GroqRateLimiter(Date.now());
  return groqLimiter;
}

/** Reads the whole body as UTF-8 through `reader`, so the caller keeps the handle it needs to cancel the read. */
async function readBodyText(reader: ReadableStreamDefaultReader<Uint8Array>): Promise<string> {
  const decoder = new TextDecoder();
  let text = "";
  for (;;) {
    const { done, value } = await reader.read();
    if (done) break;
    text += decoder.decode(value, { stream: true });
  }
  return text + decoder.decode();
}

/** Error message from an OpenAI-style error body (`{ error: { message } }` or `{ detail }`), else the raw text. */
function errorDetail(raw: string): string {
  try {
    const json = JSON.parse(raw) as { error?: { message?: string } | string; detail?: unknown; message?: unknown };
    const msg =
      (typeof json.error === "object" ? json.error?.message : json.error) ??
      (typeof json.detail === "string" ? json.detail : undefined) ??
      (typeof json.message === "string" ? json.message : undefined);
    if (msg) return truncateForUi(String(msg), MAX_ERROR_DETAIL);
  } catch {
    /* not JSON */
  }
  return truncateForUi(raw, MAX_ERROR_DETAIL);
}

/**
 * OpenAI-compatible `POST {base}/audio/transcriptions`: a user-run server (speaches, LocalAI, faster-whisper-server,
 * whisper.cpp server) or the Groq preset (whisper-large-v3-turbo, paced to the free plan's limits). One request
 * per VAD utterance; see {@link RustUtteranceService}.
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

  protected async prepare(params: STT_State): Promise<string | null> {
    const config = resolveOpenAiAudioConfig(params.openai_audio);
    this.#config = config;
    if (config.preset === OpenAiAudioPreset.groq) {
      if (!config.apiKey) return i18n.t("stt.error_groq_key_missing");
      return null;
    }
    if (!config.baseUrl) return i18n.t("stt.error_http_base_url");
    try {
      new URL(config.baseUrl);
    } catch {
      return i18n.t("stt.error_http_base_url_invalid", { url: config.baseUrl });
    }
    if (!config.model) return i18n.t("stt.error_http_model");
    return null;
  }

  protected limiter(): RequestLimiter | null {
    return this.#config?.rateLimited ? sharedGroqLimiter() : null;
  }

  protected postProcess(text: string): string {
    return stripWhisperNoiseMarkers(text);
  }

  protected async transcribe(samples: Float32Array, signal: AbortSignal): Promise<string> {
    const config = this.#config;
    if (!config) throw new TranscriptionError("not configured", "fatal");
    const timeoutMs = config.rateLimited ? CLOUD_STT_FETCH_TIMEOUT_MS : LOCAL_STT_FETCH_TIMEOUT_MS;

    const form = new FormData();
    form.append("file", floatSamplesToWavBlob(samples, RUST_CAPTURE_SAMPLE_RATE), "audio.wav");
    form.append("model", config.model);
    form.append("response_format", "json");
    form.append("temperature", "0");
    if (config.language) form.append("language", config.language);
    if (config.prompt) form.append("prompt", config.prompt);

    // The key travels only in the Authorization header (never in the URL or in logs).
    const headers: HeadersInit = {};
    if (config.apiKey) headers.Authorization = `Bearer ${config.apiKey}`;

    let res: Response;
    try {
      res = await fetchWithTimeout(transcriptionsUrl(config.baseUrl), {
        method: "POST",
        body: form,
        headers,
        signal,
        timeoutMs,
      });
    } catch (error) {
      // `signal` aborts only on teardown; any other abort is the timeout. Network errors are worth a retry.
      if (isAbortError(error) && !signal.aborted)
        throw new TranscriptionError(`request timed out after ${timeoutMs / 1000}s`, "retryable");
      if (isAbortError(error)) throw error;
      throw new TranscriptionError(error instanceof Error ? error.message : String(error), "retryable");
    }

    // fetchWithTimeout stops watching `signal` and its timer once headers arrive, so bound the body read here:
    // a server that stalls mid-body must not pin the pipeline past stop() or forever. On abort or timeout the
    // body is cancelled too, so the connection and the pending read are released instead of left running.
    const reader = res.body?.getReader();
    let raw: string;
    try {
      raw = await raceWithTimeoutAndAbort(
        reader ? readBodyText(reader) : res.text(),
        timeoutMs,
        "response body timed out",
        signal,
      );
    } catch (error) {
      void reader?.cancel().catch(() => {});
      if (isAbortError(error)) throw error;
      throw new TranscriptionError(error instanceof Error ? error.message : String(error), "retryable");
    }
    if (!res.ok) {
      throw httpTranscriptionError(res.status, errorDetail(raw), parseRetryAfterMs(res.headers.get("retry-after"), Date.now()));
    }
    let json: { text?: unknown; error?: { message?: string } };
    try {
      json = (JSON.parse(raw) ?? {}) as typeof json;
    } catch {
      throw new TranscriptionError("server returned a non-JSON response", "fatal");
    }
    if (json.error?.message) throw new TranscriptionError(json.error.message, "fatal");
    return typeof json.text === "string" ? json.text : "";
  }
}
