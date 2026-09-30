/**
 * Shared OpenAI-compatible LLM client contract (P4-S0; frozen: change only through INTEGRATE, members may be
 * added). Implemented by P4-A in `client.ts` / `providers.ts` / `stream.ts` / `sanitize.ts`; used by the
 * transform service (P4-A), the translation LLM engine (P4-C) and the llama-server host module (P4-B).
 *
 * - Providers: the bundled llama-server helper (`bundled`, endpoint from `@/core/services/llm_server`), Gemini's
 *   OpenAI endpoint (key in the `x-goog-api-key` header), OpenRouter, OpenAI and any custom OpenAI-compatible
 *   server (Ollama, LM Studio, vLLM, a LAN llama-server).
 * - Transports: `webview` = `fetch` from the renderer (CSP allows 127.0.0.1 and the cloud providers); `rust` =
 *   `plugin:llm-server|llm_proxy_chat` (LAN hosts and anything CORS / CSP would block), streamed through a Tauri
 *   Channel (wire types `RustLlmProxy*` in `@/types`). `auto` picks `rust` for non-loopback custom hosts; loopback
 *   hosts (`bundled`, `custom` on `http://localhost` / `http://127.0.0.1`) use `webview` with `rust` as the
 *   automatic fallback when the browser cannot reach them (CORS: Ollama and LM Studio reject the app's origin by
 *   default). Outside Tauri (web mode) everything uses `webview`.
 * - Keys only ever travel in headers, never in URLs or logs (`redactKeys` before anything is shown).
 * - Output is sanitized before callers see it: reasoning (`<think>` blocks, including template-opened ones) and
 *   stray special tokens are stripped (`LlmResult.text`, `LlmDelta.text`).
 * - Every call honours `signal`; an abort rejects with the signal's reason (an `AbortError` DOMException by
 *   default), never with an `LlmError`.
 */

/** Where requests go. Persisted in settings: add values, never rename or remove them. */
export const LLM_PROVIDERS = ["bundled", "gemini", "openrouter", "openai", "custom"] as const;
export type LlmProviderId = (typeof LLM_PROVIDERS)[number];

/**
 * "default" sends nothing (the model's own behaviour; the only safe value for models without reasoning); the rest
 * map to `reasoning_effort` (OpenAI, Gemini's OpenAI endpoint, llama.cpp, vLLM) or `reasoning.effort` (OpenRouter).
 * Same values as the transform schema's `TRANSFORM_REASONING_EFFORTS`.
 */
export const LLM_REASONING_EFFORTS = ["default", "none", "minimal", "low", "medium", "high"] as const;
export type LlmReasoningEffort = (typeof LLM_REASONING_EFFORTS)[number];

/** Transport setting (persisted); `auto` resolves per endpoint (see `resolveLlmEndpoint`). */
export const LLM_TRANSPORTS = ["auto", "webview", "rust"] as const;
export type LlmTransportChoice = (typeof LLM_TRANSPORTS)[number];
/** Resolved transport of one endpoint. */
export type LlmTransport = Exclude<LlmTransportChoice, "auto">;

/** Default base URLs (OpenAI-compatible prefixes). */
export const LLM_DEFAULT_URLS: Readonly<Record<Exclude<LlmProviderId, "bundled" | "custom">, string>> = {
  gemini: "https://generativelanguage.googleapis.com/v1beta/openai",
  openrouter: "https://openrouter.ai/api/v1",
  openai: "https://api.openai.com/v1",
};

/** User-level connection settings of one consumer (translation `llm` slice, transform), before resolution. */
export interface LlmConnectionSettings {
  provider: LlmProviderId;
  /** Model name for remote providers (ignored for `bundled`, whose served alias comes from `bundled`). */
  model: string;
  /** "" = none (custom local servers). */
  apiKey: string;
  /** Base URL for `custom`; for the others "" = the provider default (`LLM_DEFAULT_URLS`). */
  baseUrl: string;
  transport: LlmTransportChoice;
  /** Required for `bundled`: the running llama-server (`LlmServerLease.endpoint` from `@/core/services/llm_server`). */
  bundled?: { baseUrl: string; apiKey: string; model: string } | null;
}

/** A fully resolved endpoint: everything a client needs to send a request. */
export interface LlmEndpoint {
  provider: LlmProviderId;
  /** OpenAI-compatible base URL including the API prefix, without a trailing slash (`http://127.0.0.1:5123/v1`). */
  baseUrl: string;
  /** "" = no auth header. */
  apiKey: string;
  /** How the key is sent: `bearer` = `Authorization: Bearer <key>`; `x-goog-api-key` for Gemini. */
  keyHeader: "bearer" | "x-goog-api-key";
  model: string;
  /** Extra headers (OpenRouter attribution). Never contains the key. */
  headers: Record<string, string>;
  transport: LlmTransport;
  /**
   * Transport to retry with when `transport` cannot reach the server at all (the browser's fetch fails, or a
   * loopback server refuses the app's Origin with 403). Set by `resolveLlmEndpoint` for `auto` on loopback hosts
   * inside Tauri; the client then remembers the switch per origin.
   */
  fallbackTransport?: LlmTransport;
}

export interface LlmMessage {
  role: "system" | "user" | "assistant";
  content: string;
}

/** Sampling knobs; `undefined` = not sent (server default). llama.cpp accepts all of them. */
export interface LlmSampling {
  temperature?: number;
  topP?: number;
  topK?: number;
  minP?: number;
  repetitionPenalty?: number;
  seed?: number;
  stop?: string[];
}

/** `POST {baseUrl}/chat/completions`. */
export interface LlmChatRequest extends LlmSampling {
  messages: LlmMessage[];
  /** Sent as `max_completion_tokens` to OpenAI and `max_tokens` elsewhere. */
  maxTokens?: number;
  reasoningEffort?: LlmReasoningEffort;
  /** Stream with SSE; defaults to `true` when `onDelta` is given. */
  stream?: boolean;
  /** Extra JSON members merged into the body last (e.g. llama.cpp `chat_template_kwargs`). */
  extra?: Record<string, unknown>;
  /** Overall deadline for this call; the caller's `signal` still wins. */
  timeoutMs?: number;
}

/**
 * Raw completion for models driven by a plain prompt (Seed-X, MADLAD): llama-server `POST /completion` at the
 * server root (`baseUrl` minus `/v1`), or `POST {baseUrl}/completions` on other OpenAI-compatible servers.
 */
export interface LlmCompletionRequest extends LlmSampling {
  prompt: string;
  maxTokens?: number;
  stream?: boolean;
  extra?: Record<string, unknown>;
  timeoutMs?: number;
}

export type LlmFinishReason = "stop" | "length" | "content_filter" | "unknown";

export interface LlmResult {
  /** Reply with reasoning and special tokens removed: what callers publish. */
  text: string;
  /** Reply as received (reasoning included); for logs (after `redactKeys`) and tests only. */
  raw: string;
  finishReason: LlmFinishReason;
  /** Model reported by the server (or the requested one). */
  model: string;
  usage?: { promptTokens: number; completionTokens: number };
  /** ms from sending to the first content token (streaming only). */
  firstTokenMs?: number;
  /** ms from sending to the end of the reply. */
  totalMs: number;
}

/** Streaming progress. */
export interface LlmDelta {
  /** Sanitized text so far (an unclosed `<think>` block hides everything after it). */
  text: string;
  /** Raw text received in this chunk. */
  delta: string;
}

export interface LlmCallOptions {
  signal: AbortSignal;
  /** Called for every streamed chunk, in order, before the promise resolves. */
  onDelta?: (delta: LlmDelta) => void;
}

/** One client per resolved endpoint (cheap to create; holds no connection). */
export interface ILlmClient {
  readonly endpoint: LlmEndpoint;
  chat(request: LlmChatRequest, options: LlmCallOptions): Promise<LlmResult>;
  complete(request: LlmCompletionRequest, options: LlmCallOptions): Promise<LlmResult>;
  /** `GET {baseUrl}/models` ids; rejects with an `LlmError` when unreachable. */
  listModels(signal: AbortSignal): Promise<string[]>;
}

export type LlmErrorKind =
  /** Missing key / URL / model (a start-time problem, not a per-line one). */
  | "not_configured"
  /** 401 / 403. */
  | "unauthorized"
  /** 429. */
  | "rate_limited"
  | "timeout"
  /** Connection refused / reset / DNS / proxy transport failure. */
  | "network"
  /** 5xx. */
  | "server"
  /** Other 4xx (unknown model, bad parameter). */
  | "bad_request"
  /** `finish_reason: length` with no usable text. */
  | "truncated"
  /** A successful reply with no text after sanitizing. */
  | "empty"
  /** The endpoint cannot do this (e.g. raw completion on a provider without it). */
  | "unsupported";

/** A failed LLM call with a message that is safe to show (keys redacted). */
export class LlmError extends Error {
  readonly kind: LlmErrorKind;
  /** HTTP status when the server answered with one. */
  readonly status?: number;
  /** Whether repeating the same request can succeed. */
  readonly retryable: boolean;
  /** How long the server asked to wait before retrying (`Retry-After`, Gemini's `RetryInfo`), when it said. */
  readonly retryAfterMs?: number;

  constructor(kind: LlmErrorKind, message: string, options: { status?: number; retryable?: boolean; retryAfterMs?: number } = {}) {
    super(message);
    this.name = "LlmError";
    this.kind = kind;
    this.status = options.status;
    this.retryable = options.retryable ?? (kind === "rate_limited" || kind === "timeout" || kind === "network" || kind === "server");
    if (options.retryAfterMs !== undefined) this.retryAfterMs = options.retryAfterMs;
  }
}
