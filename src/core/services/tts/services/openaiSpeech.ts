import { invoke } from "@tauri-apps/api/core";
import i18n from "i18next";
import type { RustOpenAiSpeechRequest, RustTtsSinkTarget, RustTtsSpeakReport } from "@/types";
import { isSinkCancelled } from "../../sound/sink";
import { clampedOr, finiteOr } from "../num";
import { TTS_State } from "../schema";
import type { OpenAiSpeechFormat, OpenAiSpeechPresetId } from "../schema/openaiSpeech";
import { ITTSReceiver, ITTSService, TtsSinkContext, errorMessage } from "../types";

// -------------------------------------------------------------------------------------------------
// Presets
// -------------------------------------------------------------------------------------------------

/** A server template of the `openaiSpeech` backend. */
export interface OpenAiSpeechPreset {
  baseUrl: string;
  /** "" = not sent (the server's default model). */
  model: string;
  /** "" = not sent (the server's default voice). */
  voice: string;
  format: OpenAiSpeechFormat;
  /** Rate of the server's raw PCM (only used with `format: "pcm"`). */
  pcmSampleRate: number;
  /**
   * Ask the server to stream. Off for servers that answer whole clips or reject the member (Rust never sends it to
   * OpenAI's own API).
   */
  stream: boolean;
  /** Server-specific request members. */
  extra?: Record<string, unknown>;
  /** How to start the server (shown in the panel; not translated). */
  command?: string;
}

/**
 * Server presets (P2-I). Ports and formats are the servers' documented defaults:
 * - Kokoro-FastAPI: `/v1/audio/speech`, raw 24 kHz PCM streamed, voices at `/v1/audio/voices`.
 * - Chatterbox (OpenAI-compatible Chatterbox servers, Sigil's old default port 5555): WAV, whole clips.
 * - Voxtral-4B-TTS, IndexTTS-2.5 and VoxCPM2 on vLLM-Omni (`vllm serve <model> --omni`, port 8000): Voxtral and
 *   VoxCPM2 stream raw PCM (24 kHz / 48 kHz); IndexTTS-2.5 answers 22.05 kHz WAV (streaming it is untested).
 * - qwentts.cpp `tts-server` (Qwen3-TTS): raw 24 kHz PCM streamed, voices at `/v1/audio/voices`.
 */
export const OPENAI_SPEECH_PRESETS: Readonly<Record<Exclude<OpenAiSpeechPresetId, "custom">, OpenAiSpeechPreset>> = {
  kokoroFastapi: {
    baseUrl: "http://127.0.0.1:8880/v1",
    model: "kokoro",
    voice: "af_heart",
    format: "pcm",
    pcmSampleRate: 24_000,
    stream: true,
    command: "docker run -p 8880:8880 ghcr.io/remsky/kokoro-fastapi-cpu",
  },
  chatterbox: {
    baseUrl: "http://127.0.0.1:5555/v1",
    model: "chatterbox",
    voice: "default",
    format: "wav",
    pcmSampleRate: 24_000,
    stream: false,
    extra: { exaggeration: 0.5 },
  },
  voxtral: {
    baseUrl: "http://127.0.0.1:8000/v1",
    model: "mistralai/Voxtral-4B-TTS-2603",
    voice: "casual_male",
    format: "pcm",
    pcmSampleRate: 24_000,
    stream: true,
    command: "vllm serve mistralai/Voxtral-4B-TTS-2603 --omni",
  },
  indexTts: {
    baseUrl: "http://127.0.0.1:8000/v1",
    model: "IndexTeam/IndexTTS-2.5",
    voice: "",
    format: "wav",
    pcmSampleRate: 22_050,
    stream: false,
    command: "vllm serve IndexTeam/IndexTTS-2.5 --omni",
  },
  voxcpm2: {
    baseUrl: "http://127.0.0.1:8000/v1",
    model: "openbmb/VoxCPM2",
    voice: "",
    format: "pcm",
    pcmSampleRate: 48_000,
    stream: true,
    command: "vllm serve openbmb/VoxCPM2 --omni",
  },
  qwentts: {
    baseUrl: "http://127.0.0.1:8080/v1",
    model: "",
    voice: "",
    format: "pcm",
    pcmSampleRate: 24_000,
    stream: true,
    command: "tts-server --port 8080",
  },
};

/** The preset `id`, or `null` for `custom`. */
export function openAiSpeechPreset(id: OpenAiSpeechPresetId): OpenAiSpeechPreset | null {
  return id === "custom" ? null : OPENAI_SPEECH_PRESETS[id] ?? null;
}

/**
 * Origin (scheme, host, port) of a server URL, or `null` when it is not an http(s) URL. A URL without a scheme is
 * http, as in Rust (`http_client::normalize_base_url`).
 */
export function serverOrigin(url: string | null | undefined): string | null {
  const u = (url ?? "").trim();
  if (!u) return null;
  try {
    const parsed = new URL(u.includes("://") ? u : `http://${u}`);
    return parsed.protocol === "http:" || parsed.protocol === "https:" ? parsed.origin : null;
  } catch {
    return null;
  }
}

/**
 * The settings slice after choosing preset `id`: its server fields replace the current ones (`extra` too, since it
 * is server-specific); speed, device, volume and timeout stay. The API key stays only when the preset's server has
 * the same origin as the current URL: a key is never sent to a different server. `custom` only records the choice.
 */
export function applyOpenAiSpeechPreset(
  current: TTS_State["openaiSpeech"],
  id: OpenAiSpeechPresetId,
): TTS_State["openaiSpeech"] {
  const preset = openAiSpeechPreset(id);
  if (!preset) return { ...current, preset: id };
  const origin = serverOrigin(current.baseUrl);
  const sameServer = origin !== null && origin === serverOrigin(preset.baseUrl);
  return {
    ...current,
    preset: id,
    apiKey: sameServer ? current.apiKey : "",
    baseUrl: preset.baseUrl,
    model: preset.model,
    voice: preset.voice,
    format: preset.format,
    pcmSampleRate: String(preset.pcmSampleRate),
    stream: preset.stream,
    extra: preset.extra ? JSON.stringify(preset.extra) : "",
  };
}

// -------------------------------------------------------------------------------------------------
// Requests
// -------------------------------------------------------------------------------------------------

/** Everything but the text of an `openai_speech_speak` request. */
export type OpenAiSpeechSettings = Omit<RustOpenAiSpeechRequest, "input">;

/** Automatic timeout (Rust picks it from the server's location and the text length). */
const AUTOMATIC_TIMEOUT = 0;

/**
 * The `extra` text as a request object: `null` when blank, `undefined` when it is not a JSON object (the panel shows
 * the error; Start refuses).
 */
export function parseExtra(text: string | null | undefined): Record<string, unknown> | null | undefined {
  const trimmed = (text ?? "").trim();
  if (!trimmed) return null;
  try {
    const value: unknown = JSON.parse(trimmed);
    return value && typeof value === "object" && !Array.isArray(value) ? (value as Record<string, unknown>) : undefined;
  } catch {
    return undefined;
  }
}

/** The request settings of the `openaiSpeech` backend, or a localized reason to refuse Start. */
export function openAiSpeechSettings(c: TTS_State["openaiSpeech"]): OpenAiSpeechSettings | string {
  const baseUrl = (c.baseUrl ?? "").trim();
  if (!baseUrl) return i18n.t("tts.openai_speech_url_missing");
  const extra = parseExtra(c.extra);
  if (extra === undefined) return i18n.t("tts.openai_speech_extra_invalid");
  const format: OpenAiSpeechFormat = c.format === "wav" ? "wav" : "pcm";
  return {
    baseUrl,
    apiKey: (c.apiKey ?? "").trim(),
    model: (c.model ?? "").trim(),
    voice: (c.voice ?? "").trim(),
    responseFormat: format,
    speed: clampedOr(c.speed, 1, 0.25, 4),
    stream: !!c.stream,
    pcmSampleRate: Math.round(clampedOr(c.pcmSampleRate, 24_000, 8_000, 192_000)),
    extra,
    timeoutMs: Math.round(clampedOr(c.timeout, 0, 0, 600) * 1000),
  };
}

/**
 * `endpoint` as an OpenAI API base: `http://localhost:8880` → `http://localhost:8880/v1` (legacy Kokoro and
 * Chatterbox settings store the server root; one that already ends in `/v1` is kept).
 */
export function legacyV1Base(endpoint: string): string {
  const e = (endpoint ?? "").trim().replace(/\/+$/, "");
  if (!e) return "";
  return /\/v1$/i.test(e) ? e : `${e}/v1`;
}

/** Legacy `kokoro` settings mapped onto the OpenAI-compatible client (Kokoro-FastAPI's streamed 24 kHz PCM). */
export function kokoroSpeechSettings(c: TTS_State["kokoro"]): OpenAiSpeechSettings {
  return {
    baseUrl: legacyV1Base(c.endpoint),
    apiKey: "",
    model: "kokoro",
    voice: (c.voice ?? "").trim() || "af_bella",
    responseFormat: "pcm",
    speed: clampedOr(c.speed, 1, 0.25, 4),
    stream: true,
    pcmSampleRate: 24_000,
    extra: null,
    timeoutMs: AUTOMATIC_TIMEOUT,
  };
}

/**
 * Legacy `chatterbox` settings mapped onto the OpenAI-compatible client: WAV (the old client asked for MP3; every
 * Chatterbox server answers WAV, and an MP3 answer is still decoded), `exaggeration` as an extra member, no `stream`
 * member (the old request had none).
 */
export function chatterboxSpeechSettings(c: TTS_State["chatterbox"]): OpenAiSpeechSettings {
  const exaggeration = finiteOr(c.exaggeration, 0.5);
  return {
    baseUrl: legacyV1Base(c.endpoint),
    apiKey: "",
    model: "chatterbox",
    voice: (c.voice ?? "").trim() || "default",
    responseFormat: "wav",
    speed: clampedOr(c.speed, 1, 0.25, 4),
    stream: false,
    pcmSampleRate: 24_000,
    extra: exaggeration >= 0 ? { exaggeration } : null,
    timeoutMs: AUTOMATIC_TIMEOUT,
  };
}

/** Speak `text` with `settings` into the begun sink utterance `utteranceId`. */
export function speakOpenAi(settings: OpenAiSpeechSettings, text: string, utteranceId: string): Promise<RustTtsSpeakReport> {
  const request: RustOpenAiSpeechRequest = { ...settings, input: text };
  return invoke<RustTtsSpeakReport>("plugin:openai-speech|openai_speech_speak", { request, utteranceId });
}

/** Whether a server answers at `baseUrl`; rejects with a readable reason when it refuses the key. */
export function checkOpenAiServer(baseUrl: string, apiKey: string): Promise<boolean> {
  return invoke<boolean>("plugin:openai-speech|openai_speech_check", { baseUrl: baseUrl.trim(), apiKey: apiKey.trim() || null });
}

/** Voice ids the server at `baseUrl` lists (`[]` when it lists none). Rejects when it cannot be reached. */
export async function listOpenAiVoices(baseUrl: string, apiKey: string): Promise<string[]> {
  const list = await invoke<string[]>("plugin:openai-speech|openai_speech_list_voices", {
    baseUrl: baseUrl.trim(),
    apiKey: apiKey.trim() || null,
  });
  return Array.isArray(list) ? list : [];
}

/** Fire-and-forget IPC: never throws, synchronously or later. */
function send(command: string, args?: Record<string, unknown>): void {
  try {
    void Promise.resolve(invoke(command, args)).catch(() => {});
  } catch {
    // Without Tauri (web build, tests) there is nothing to tell.
  }
}

/**
 * Stop the HTTP speech stream of `utteranceId`, or every one (`null`). Covers every backend on the shared Rust
 * streaming core: OpenAI-compatible, Kokoro, Chatterbox, MeloTTS, Fish Speech and the HTTP speech server.
 */
export function cancelHttpSpeech(utteranceId: string | null): void {
  send("plugin:openai-speech|openai_speech_cancel", { utteranceId });
}

/**
 * Speak once into a new sink utterance on `target` (the panels' "Test voice"). `speak` receives the utterance id and
 * pushes the audio; resolves once it is queued. A stop / skip meanwhile is not an error.
 */
export async function previewToSink(target: RustTtsSinkTarget, speak: (utteranceId: string) => Promise<unknown>): Promise<void> {
  const sink = window.ApiServer.sound.sink;
  const id = `tts-preview-${Date.now().toString(36)}-${Math.random().toString(36).slice(2, 8)}`;
  await sink.ready();
  await sink.begin(id, target);
  try {
    await speak(id);
    await sink.end(id);
  } catch (error) {
    await sink.cancel(id).catch(() => {});
    if (!isSinkCancelled(error)) throw error;
  }
}

// -------------------------------------------------------------------------------------------------
// Backends
// -------------------------------------------------------------------------------------------------

/**
 * Base of the backends that speak through a user-run HTTP speech server and Rust's streaming core (ITTSService v2
 * `synthesizeToSink`): OpenAI-compatible servers (and the legacy Kokoro / Chatterbox settings mapped onto them),
 * MeloTTS, Fish Speech and the HTTP speech server. Rust streams each sentence straight into the sink utterance the
 * sentence pipeline began, decoding the response as it arrives: the first audio plays before the server finished,
 * the output device and the route (device / voice changer / both) apply, and no audio crosses IPC.
 *
 * - `start` runs `connect()` (settings, server check) and reports `onStart`, or `onStop(reason)`. A Stop during
 *   `connect()` wins: the late result is dropped.
 * - Stop / skip abort the sentence's signal, which cancels the Rust request at once (the sink cancel also ends it
 *   within ~25 ms); an aborted sentence never reports an error. A failed sentence rejects with `"<label>: <reason>"`.
 * - `stop` cancels every running request and reports `onStop()`.
 */
export abstract class HttpSinkBackend implements ITTSService {
  #running = false;
  #stopped = false;

  constructor(
    protected readonly bindings: ITTSReceiver,
    /** Prefix of error messages, e.g. "Kokoro". */
    protected readonly label: string,
  ) {}

  /** Read the settings and check the server. Resolve with a reason to refuse Start, or `null`. */
  protected abstract connect(state: TTS_State): Promise<string | null>;

  /** Speak one sentence into the begun sink utterance `utteranceId` (resolves once the audio is queued). */
  protected abstract speak(text: string, utteranceId: string): Promise<unknown>;

  get running(): boolean {
    return this.#running;
  }

  async start(state: TTS_State): Promise<void> {
    try {
      const refusal = await this.connect(state);
      if (this.#stopped) return;
      if (refusal) {
        this.bindings.onStop(refusal);
        return;
      }
      this.#running = true;
      this.bindings.onStart();
    } catch (error) {
      if (this.#stopped) return;
      this.bindings.onStop(errorMessage(error));
    }
  }

  /** v2: Rust streams the sentence into the begun sink utterance `sink.utteranceId`. */
  async synthesizeToSink(text: string, sink: TtsSinkContext, signal: AbortSignal): Promise<void> {
    if (!this.#running || signal.aborted || !text.trim()) return;
    const onAbort = () => cancelHttpSpeech(sink.utteranceId);
    signal.addEventListener("abort", onAbort, { once: true });
    try {
      await this.speak(text, sink.utteranceId);
    } catch (error) {
      // Stop / skip cancel the utterance; Rust then answers "cancelled".
      if (signal.aborted || isSinkCancelled(error) || !this.#running) return;
      throw new Error(`${this.label}: ${errorMessage(error)}`);
    } finally {
      signal.removeEventListener("abort", onAbort);
    }
  }

  /**
   * Legacy whole-line path. Unused: a backend with `synthesizeToSink` always speaks through the sentence pipeline
   * (Service_TTS creates it for every non-self-playing backend).
   */
  play(_value: string): void {}

  stop(): void {
    this.#halt();
    this.bindings.onStop();
  }

  dispose(): void {
    this.#halt();
  }

  #halt() {
    const wasActive = !this.#stopped;
    this.#stopped = true;
    this.#running = false;
    if (wasActive) cancelHttpSpeech(null);
  }
}

/**
 * Base of the backends on the Rust `openai-speech` client: the settings are read and checked at start, then every
 * sentence is one `openai_speech_speak` request. With {@link liveSettings}, each sentence re-reads the settings (the
 * panel keeps voice and speed editable while speaking); otherwise the start-time settings are used.
 */
export abstract class OpenAiSpeechBackend extends HttpSinkBackend {
  #settings: OpenAiSpeechSettings | null = null;

  /** Re-read the settings for every sentence (voice / speed changes apply to the next sentence). */
  protected readonly liveSettings: boolean = false;

  /** The request settings, or a reason to refuse Start. */
  protected abstract settings(state: TTS_State): OpenAiSpeechSettings | string;

  /** Check the server; resolve with a reason to refuse Start, or `null`. */
  protected async check(settings: OpenAiSpeechSettings): Promise<string | null> {
    const ok = await checkOpenAiServer(settings.baseUrl, settings.apiKey ?? "");
    return ok ? null : i18n.t("tts.openai_speech_unreachable", { url: settings.baseUrl });
  }

  protected async connect(state: TTS_State): Promise<string | null> {
    const settings = this.settings(state);
    if (typeof settings === "string") return settings;
    this.#settings = settings;
    return this.check(settings);
  }

  protected async speak(text: string, utteranceId: string): Promise<unknown> {
    let settings = this.#settings;
    if (this.liveSettings) {
      const live = this.settings(window.ApiServer.state.services.tts.data);
      if (typeof live === "string") throw new Error(live);
      settings = live;
    }
    if (!settings) throw new Error("not started");
    return speakOpenAi(settings, text, utteranceId);
  }
}

/**
 * Any OpenAI-compatible `/v1/audio/speech` server (backend `openaiSpeech`): Kokoro-FastAPI, Chatterbox servers,
 * Voxtral-4B-TTS / IndexTTS-2.5 / VoxCPM2 on vLLM-Omni, qwentts.cpp's `tts-server`, or any other, streaming through
 * the Rust `openai-speech` plugin into the TTS sink. Voice and speed changes apply to the next sentence.
 */
export class TTS_OpenAISpeechService extends OpenAiSpeechBackend {
  protected readonly liveSettings = true;

  constructor(bindings: ITTSReceiver) {
    super(bindings, "Speech server");
  }

  protected settings(state: TTS_State): OpenAiSpeechSettings | string {
    return openAiSpeechSettings(state.openaiSpeech);
  }
}
