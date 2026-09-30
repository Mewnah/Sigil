import { cleanLlmOutput, redactKeys, visibleStreamText } from "./sanitize";
import {
  LlmPayload,
  parseChatPayload,
  parseCompletionPayload,
  readErrorObject,
  readErrorPayload,
  SSE_DONE,
  SseParser,
  tryParseJson,
} from "./stream";
import {
  abortReason,
  createRustTransport,
  createWebviewTransport,
  LlmHttpRequest,
  LlmHttpResponse,
  LlmHttpTransport,
  LlmTransportError,
} from "./transport";
import {
  ILlmClient,
  LlmCallOptions,
  LlmChatRequest,
  LlmCompletionRequest,
  LlmDelta,
  LlmEndpoint,
  LlmError,
  LlmFinishReason,
  LlmResult,
  LlmSampling,
  LlmTransport,
} from "./types";
import { urlOrigin } from "./providers";

/**
 * OpenAI-compatible client for one resolved endpoint (see `types.ts` for the contract). Plain `fetch` / Rust proxy,
 * no SDK:
 * - `chat`: `POST {baseUrl}/chat/completions`, streamed (SSE) when `stream` / `onDelta` asks for it;
 * - `complete`: raw prompts (Seed-X, MADLAD): llama-server's native `POST /completion` for the bundled server,
 *   `POST {baseUrl}/completions` elsewhere (not offered by Gemini);
 * - `listModels`: `GET {baseUrl}/models`.
 *
 * A server that refuses a streamed request (HTTP 400 about `stream`, before any text) gets the same request once more
 * without streaming, and this client stops asking it to stream (`withStreamFallback`): the reply then arrives whole,
 * without deltas.
 *
 * Provider differences handled here: `max_completion_tokens` for OpenAI (its reasoning models reject
 * `max_tokens`), OpenRouter's `reasoning: { effort }`, Gemini 3's lack of `none` effort (sent as `minimal`),
 * llama.cpp's `chat_template_kwargs.enable_thinking` and `repeat_penalty`, and sampling knobs only where the
 * provider accepts them. Keys travel only in headers and are redacted from every error message.
 */

/** Deadline of `listModels` (the connection check in settings panels). */
export const LLM_LIST_MODELS_TIMEOUT_MS = 10_000;
/** A stream that sends nothing for this long is treated as a timeout (the Rust proxy uses the same stall limit). */
export const LLM_STALL_TIMEOUT_MS = 30_000;
/** Error bodies are read up to this size for their message. */
const ERROR_BODY_LIMIT = 64 * 1024;
/** Whole (non-streamed) replies are capped at this size (the Rust proxy caps at 8 MB as well). */
const BODY_LIMIT = 8 * 1024 * 1024;

export interface LlmClientDeps {
  /** Transports by name (tests; defaults: webview `fetch` and the Rust proxy). */
  transports?: Partial<Record<LlmTransport, LlmHttpTransport>>;
  /** Clock for `firstTokenMs` / `totalMs` (tests). */
  now?: () => number;
  /** Stall limit override (tests). */
  stallTimeoutMs?: number;
}

/**
 * Origins that needed the Rust proxy after the webview could not reach them (CORS / Origin checks of local
 * servers); later requests to them skip the failing browser attempt. Process-wide, reset by `resetLlmTransportMemory`.
 */
const proxiedOrigins = new Set<string>();

export function resetLlmTransportMemory(): void {
  proxiedOrigins.clear();
}

export function createLlmClient(endpoint: LlmEndpoint, deps: LlmClientDeps = {}): ILlmClient {
  const transports: Record<LlmTransport, LlmHttpTransport> = {
    webview: deps.transports?.webview ?? createWebviewTransport(),
    rust: deps.transports?.rust ?? createRustTransport(),
  };
  const now = deps.now ?? (() => (typeof performance !== "undefined" ? performance.now() : Date.now()));
  const stallTimeoutMs = deps.stallTimeoutMs ?? LLM_STALL_TIMEOUT_MS;
  const secrets = endpoint.apiKey ? [endpoint.apiKey] : [];
  const redact = (text: string) => redactKeys(text, secrets);

  function headers(json: boolean): Record<string, string> {
    const h: Record<string, string> = { Accept: json ? "application/json, text/event-stream" : "application/json", ...endpoint.headers };
    if (json) h["Content-Type"] = "application/json";
    if (endpoint.apiKey) {
      if (endpoint.keyHeader === "x-goog-api-key") h["x-goog-api-key"] = endpoint.apiKey;
      else h.Authorization = `Bearer ${endpoint.apiKey}`;
    }
    return h;
  }

  /** Sends `request` on the endpoint's transport, falling back to `fallbackTransport` when the first cannot reach the server. */
  async function send(request: LlmHttpRequest, signal: AbortSignal): Promise<LlmHttpResponse> {
    const origin = urlOrigin(request.url);
    const fallback = endpoint.fallbackTransport && endpoint.fallbackTransport !== endpoint.transport ? endpoint.fallbackTransport : undefined;
    if (fallback && proxiedOrigins.has(origin)) return viaTransport(fallback, request, signal);
    let response: LlmHttpResponse;
    try {
      response = await viaTransport(endpoint.transport, request, signal);
    } catch (e) {
      if (signal.aborted || !fallback || !(e instanceof LlmTransportError) || !e.beforeResponse) throw e;
      return retryVia(fallback, origin, request, signal, e);
    }
    // Ollama answers a browser Origin it does not allow with a bare 403; the proxy sends no Origin.
    if (fallback && endpoint.transport === "webview" && response.status === 403) {
      const refusal = await readText(response, signal, ERROR_BODY_LIMIT).catch(() => "");
      response.cancel();
      try {
        return await retryVia(fallback, origin, request, signal, null);
      } catch (e) {
        // No proxy here: the browser's answer is the one to report.
        if (e instanceof LlmTransportError && e.unavailable) throw httpError(403, refusal, redact);
        throw e;
      }
    }
    return response;
  }

  async function retryVia(fallback: LlmTransport, origin: string, request: LlmHttpRequest, signal: AbortSignal, first: LlmTransportError | null) {
    try {
      const response = await viaTransport(fallback, request, signal);
      if (origin) proxiedOrigins.add(origin);
      return response;
    } catch (e) {
      // The proxy is not available (web mode, older build): report what the first transport saw.
      if (first && e instanceof LlmTransportError && e.unavailable) throw first;
      throw e;
    }
  }

  function viaTransport(transport: LlmTransport, request: LlmHttpRequest, signal: AbortSignal) {
    return transports[transport](request, signal);
  }

  /**
   * Runs one request with the caller's signal, an optional overall deadline and the stall guard, mapping every
   * failure to an `LlmError` (aborts by the caller reject with the signal's reason). `touch` (re)arms the stall
   * guard: callers call it once the response has started and for every chunk, so a slow prefill is bounded only by
   * the deadline while a stream that goes silent is cut off.
   */
  async function run<T>(
    signal: AbortSignal,
    timeoutMs: number | undefined,
    body: (inner: AbortSignal, touch: () => void) => Promise<T>,
  ): Promise<T> {
    if (signal.aborted) throw abortReason(signal);
    const inner = new AbortController();
    const onAbort = () => inner.abort(abortReason(signal));
    signal.addEventListener("abort", onAbort, { once: true });
    const state = { expired: false };
    const expire = (error: LlmError) => {
      if (inner.signal.aborted) return;
      state.expired = true;
      inner.abort(error);
    };
    const deadline =
      timeoutMs && timeoutMs > 0
        ? setTimeout(() => expire(new LlmError("timeout", `no reply within ${Math.round(timeoutMs / 100) / 10} s`)), timeoutMs)
        : undefined;
    let stall: ReturnType<typeof setTimeout> | undefined;
    const touch = () => {
      if (stall !== undefined) clearTimeout(stall);
      stall = setTimeout(() => expire(new LlmError("timeout", `the server sent nothing for ${Math.round(stallTimeoutMs / 1000)} s`)), stallTimeoutMs);
    };
    try {
      return await body(inner.signal, touch);
    } catch (e) {
      if (signal.aborted) throw abortReason(signal);
      if (state.expired) throw inner.signal.reason;
      throw toLlmError(e, redact);
    } finally {
      if (deadline !== undefined) clearTimeout(deadline);
      if (stall !== undefined) clearTimeout(stall);
      signal.removeEventListener("abort", onAbort);
      // Releases a body the caller stopped reading (errors, early returns).
      if (!inner.signal.aborted) inner.abort(new DOMException("done", "AbortError"));
    }
  }

  async function failForStatus(response: LlmHttpResponse, signal: AbortSignal): Promise<never> {
    const text = await readText(response, signal, ERROR_BODY_LIMIT).catch(() => "");
    throw httpError(response.status, text, redact, response.headers);
  }

  /** Streams SSE `data` payloads (or one whole JSON reply when the server ignored `stream`). */
  async function consume(
    response: LlmHttpResponse,
    signal: AbortSignal,
    touch: () => void,
    stream: boolean,
    parse: (json: unknown) => LlmPayload,
    onPayload: (payload: LlmPayload) => void,
  ): Promise<void> {
    const contentType = response.headers["content-type"] ?? "";
    const sse = stream && !/application\/json/i.test(contentType);
    if (!sse) {
      const text = await readText(response, signal, BODY_LIMIT, touch);
      const json = tryParseJson(text);
      if (json === undefined) throw new LlmError("server", notOpenAiReply(text), { retryable: false });
      const payload = parse(json);
      if (payload.error) throw payloadError(payload.error, redact);
      onPayload(payload);
      return;
    }
    const parser = new SseParser();
    let sawPayload = false;
    let raw = "";
    const handle = (events: ReturnType<SseParser["push"]>): boolean => {
      for (const ev of events) {
        const data = ev.data.trim();
        if (data === SSE_DONE) return true;
        if (!data) continue;
        const json = tryParseJson(data);
        if (json === undefined) {
          if (ev.event === "error") throw new LlmError("server", redact(data), { retryable: true });
          continue;
        }
        const payload = parse(json);
        if (payload.error || ev.event === "error") {
          throw payloadError(payload.error ?? readErrorObject(json) ?? { message: data }, redact);
        }
        sawPayload = true;
        onPayload(payload);
      }
      return false;
    };
    for await (const chunk of response.body) {
      touch();
      if (!sawPayload && raw.length < ERROR_BODY_LIMIT) raw += chunk;
      if (handle(parser.push(chunk))) {
        response.cancel();
        return;
      }
    }
    handle(parser.end());
    if (!sawPayload) {
      // Not SSE after all (some servers ignore `stream` and send one JSON body with a text/plain type).
      const json = tryParseJson(raw.trim());
      if (json === undefined) {
        if (raw.trim()) throw new LlmError("server", notOpenAiReply(raw), { retryable: false });
        return;
      }
      const payload = parse(json);
      if (payload.error) throw payloadError(payload.error, redact);
      onPayload(payload);
    }
  }

  /** Accumulates payloads into an `LlmResult`, reporting deltas as they arrive. */
  function accumulator(requestedModel: string, started: number, options: LlmCallOptions) {
    let raw = "";
    let finishReason: LlmFinishReason | undefined;
    let model = requestedModel;
    let usage: LlmResult["usage"];
    let firstTokenMs: number | undefined;
    let deltaFailed = false;
    return {
      add(payload: LlmPayload, streaming: boolean) {
        if (payload.model) model = payload.model;
        if (payload.finishReason) finishReason = payload.finishReason;
        if (payload.usage) usage = payload.usage;
        if (!payload.content) return;
        raw += payload.content;
        if (!streaming) return;
        if (firstTokenMs === undefined) firstTokenMs = Math.max(0, now() - started);
        if (!options.onDelta || options.signal.aborted) return;
        const delta: LlmDelta = { text: visibleStreamText(raw), delta: payload.content };
        try {
          options.onDelta(delta);
        } catch (e) {
          // A consumer bug must not cost the line.
          if (!deltaFailed) console.warn("[LLM] onDelta handler failed:", e);
          deltaFailed = true;
        }
      },
      finish(): LlmResult {
        const text = cleanLlmOutput(raw);
        if (!text) {
          if (finishReason === "length") {
            throw new LlmError("truncated", "the reply hit the output token limit before any text (finish_reason=length)", { retryable: false });
          }
          if (finishReason === "content_filter") {
            throw new LlmError("empty", "the provider's content filter blocked the reply", { retryable: false });
          }
          throw new LlmError("empty", `no text returned by ${model}`, { retryable: false });
        }
        return {
          text,
          raw,
          finishReason: finishReason ?? "unknown",
          model,
          ...(usage ? { usage } : {}),
          ...(firstTokenMs !== undefined ? { firstTokenMs } : {}),
          totalMs: Math.max(0, now() - started),
        };
      },
    };
  }

  /** Routes whose server refused a streamed request (`refusesStreaming`): later calls on this client go plain. */
  const streamRefused = { chat: false, complete: false };

  /**
   * Runs `call` streamed when asked, and once more without streaming when the server refuses the stream itself
   * before any text arrived (OpenAI answers streamed requests for some models of unverified organizations with
   * HTTP 400; some OpenAI-compatible servers reject `stream`). The refusal is remembered for this client, so later
   * calls go out plain in a single request. The retry gets what is left of `timeoutMs`; the caller's signal still
   * wins.
   */
  async function withStreamFallback(
    route: keyof typeof streamRefused,
    wantStream: boolean,
    timeoutMs: number | undefined,
    options: LlmCallOptions,
    call: (stream: boolean, timeoutMs: number | undefined, options: LlmCallOptions) => Promise<LlmResult>,
  ): Promise<LlmResult> {
    if (!wantStream || streamRefused[route]) return call(false, timeoutMs, options);
    const startedAt = Date.now();
    let delivered = false;
    const onDelta = options.onDelta;
    const tracked: LlmCallOptions = onDelta
      ? {
          ...options,
          onDelta: (delta) => {
            delivered = true;
            onDelta(delta);
          },
        }
      : options;
    try {
      return await call(true, timeoutMs, tracked);
    } catch (e) {
      if (delivered || options.signal.aborted || !refusesStreaming(e)) throw e;
      streamRefused[route] = true;
      console.warn(`[LLM] ${endpoint.model}: the server refused a streamed request, retrying without streaming`);
      const left = timeoutMs && timeoutMs > 0 ? Math.max(1, timeoutMs - (Date.now() - startedAt)) : timeoutMs;
      return call(false, left, options);
    }
  }

  const client: ILlmClient = {
    endpoint,

    async chat(request: LlmChatRequest, options: LlmCallOptions): Promise<LlmResult> {
      const wantStream = request.stream ?? options.onDelta !== undefined;
      return withStreamFallback("chat", wantStream, request.timeoutMs, options, (stream, timeoutMs, callOptions) => {
        const body = buildChatBody(endpoint, request, stream);
        return run(callOptions.signal, timeoutMs, async (signal, touch) => {
          const started = now();
          const response = await send(
            { url: `${endpoint.baseUrl}/chat/completions`, method: "POST", headers: headers(true), body: JSON.stringify(body), stream, timeoutMs },
            signal,
          );
          touch();
          if (response.status < 200 || response.status >= 300) await failForStatus(response, signal);
          const acc = accumulator(endpoint.model, started, callOptions);
          await consume(response, signal, touch, stream, parseChatPayload, (p) => acc.add(p, stream));
          return acc.finish();
        });
      });
    },

    async complete(request: LlmCompletionRequest, options: LlmCallOptions): Promise<LlmResult> {
      if (endpoint.provider === "gemini") {
        throw new LlmError("unsupported", "Gemini has no raw completion endpoint; use a chat prompt format instead", { retryable: false });
      }
      const wantStream = request.stream ?? options.onDelta !== undefined;
      const native = endpoint.provider === "bundled";
      const url = native ? `${serverRoot(endpoint.baseUrl)}/completion` : `${endpoint.baseUrl}/completions`;
      return withStreamFallback("complete", wantStream, request.timeoutMs, options, (stream, timeoutMs, callOptions) => {
        const body = native ? buildNativeCompletionBody(request, stream) : buildCompletionBody(endpoint, request, stream);
        return run(callOptions.signal, timeoutMs, async (signal, touch) => {
          const started = now();
          const response = await send({ url, method: "POST", headers: headers(true), body: JSON.stringify(body), stream, timeoutMs }, signal);
          touch();
          if (response.status < 200 || response.status >= 300) await failForStatus(response, signal);
          const acc = accumulator(endpoint.model, started, callOptions);
          await consume(response, signal, touch, stream, parseCompletionPayload, (p) => acc.add(p, stream));
          return acc.finish();
        });
      });
    },

    async listModels(signal: AbortSignal): Promise<string[]> {
      return run(signal, LLM_LIST_MODELS_TIMEOUT_MS, async (inner, touch) => {
        const response = await send({ url: `${endpoint.baseUrl}/models`, method: "GET", headers: headers(false), stream: false, timeoutMs: LLM_LIST_MODELS_TIMEOUT_MS }, inner);
        touch();
        if (response.status < 200 || response.status >= 300) await failForStatus(response, inner);
        const text = await readText(response, inner, BODY_LIMIT, touch);
        const json = tryParseJson(text);
        if (json === undefined) throw new LlmError("server", notOpenAiReply(text), { retryable: false });
        return modelIds(json, endpoint.provider === "gemini");
      });
    },
  };
  return client;
}

// ---------------------------------------------------------------------------------------------------------------
// Request bodies (exported for tests)

/** Providers that accept llama.cpp / vLLM style sampling extensions (`top_k`, `min_p`, repetition penalty). */
function acceptsExtendedSampling(endpoint: LlmEndpoint): boolean {
  return endpoint.provider === "bundled" || endpoint.provider === "custom" || endpoint.provider === "openrouter";
}

function applySampling(body: Record<string, unknown>, endpoint: LlmEndpoint, s: LlmSampling) {
  if (isNum(s.temperature)) body.temperature = s.temperature;
  if (isNum(s.topP)) body.top_p = s.topP;
  if (isNum(s.seed)) body.seed = Math.round(s.seed);
  if (s.stop && s.stop.length > 0) body.stop = s.stop.slice(0, 4);
  if (!acceptsExtendedSampling(endpoint)) return;
  if (isNum(s.topK)) body.top_k = Math.round(s.topK);
  if (isNum(s.minP)) body.min_p = s.minP;
  if (isNum(s.repetitionPenalty)) {
    if (endpoint.provider === "bundled") body.repeat_penalty = s.repetitionPenalty;
    else body.repetition_penalty = s.repetitionPenalty;
  }
}

function isNum(v: unknown): v is number {
  return typeof v === "number" && Number.isFinite(v);
}

function tokenLimit(v: number | undefined): number | undefined {
  return isNum(v) && v > 0 ? Math.max(1, Math.round(v)) : undefined;
}

/** Gemini 2.x models accept `none` (thinking off); Gemini 3+ cannot turn thinking off, `minimal` is the lowest. */
function geminiEffort(model: string, effort: string): string {
  if (effort !== "none") return effort;
  return /(^|\/)gemini-(1|2)(\.|-)/i.test(model) ? "none" : "minimal";
}

/** `POST /chat/completions` body. */
export function buildChatBody(endpoint: LlmEndpoint, request: LlmChatRequest, stream: boolean): Record<string, unknown> {
  const body: Record<string, unknown> = {
    model: endpoint.model,
    messages: request.messages.map((m) => ({ role: m.role, content: m.content })),
  };
  const maxTokens = tokenLimit(request.maxTokens);
  if (maxTokens !== undefined) {
    if (endpoint.provider === "openai") body.max_completion_tokens = maxTokens;
    else body.max_tokens = maxTokens;
  }
  applySampling(body, endpoint, request);

  const effort = request.reasoningEffort;
  if (effort && effort !== "default") {
    switch (endpoint.provider) {
      case "openrouter":
        body.reasoning = { effort };
        break;
      case "gemini":
        body.reasoning_effort = geminiEffort(endpoint.model, effort);
        break;
      case "bundled":
        // llama-server: Qwen3-style templates switch thinking through the template kwargs.
        if (effort === "none") body.chat_template_kwargs = { enable_thinking: false };
        else body.reasoning_effort = effort;
        break;
      default:
        body.reasoning_effort = effort;
    }
  }
  if (stream) body.stream = true;
  if (endpoint.provider === "bundled") body.cache_prompt = true;
  return { ...body, ...(request.extra ?? {}) };
}

/** OpenAI legacy `POST /completions` body. */
export function buildCompletionBody(endpoint: LlmEndpoint, request: LlmCompletionRequest, stream: boolean): Record<string, unknown> {
  const body: Record<string, unknown> = { model: endpoint.model, prompt: request.prompt };
  const maxTokens = tokenLimit(request.maxTokens);
  if (maxTokens !== undefined) body.max_tokens = maxTokens;
  applySampling(body, endpoint, request);
  if (stream) body.stream = true;
  return { ...body, ...(request.extra ?? {}) };
}

/** llama-server native `POST /completion` body. */
export function buildNativeCompletionBody(request: LlmCompletionRequest, stream: boolean): Record<string, unknown> {
  const body: Record<string, unknown> = { prompt: request.prompt, cache_prompt: true, stream };
  const maxTokens = tokenLimit(request.maxTokens);
  if (maxTokens !== undefined) body.n_predict = maxTokens;
  if (isNum(request.temperature)) body.temperature = request.temperature;
  if (isNum(request.topP)) body.top_p = request.topP;
  if (isNum(request.topK)) body.top_k = Math.round(request.topK);
  if (isNum(request.minP)) body.min_p = request.minP;
  if (isNum(request.repetitionPenalty)) body.repeat_penalty = request.repetitionPenalty;
  if (isNum(request.seed)) body.seed = Math.round(request.seed);
  if (request.stop && request.stop.length > 0) body.stop = request.stop;
  return { ...body, ...(request.extra ?? {}) };
}

/** `http://127.0.0.1:5123/v1` → `http://127.0.0.1:5123` (llama-server's native routes live at the root). */
export function serverRoot(baseUrl: string): string {
  return baseUrl.replace(/\/v1$/i, "");
}

// ---------------------------------------------------------------------------------------------------------------
// Replies and errors

/**
 * Whether `model` (as the user typed it) is among a server's listed `models`: exact, as the last path segment
 * (`models/gemini-…`, `org/model`), or with Ollama's implicit `:latest` tag on either side (Ollama lists
 * `llama3:latest` and serves it for `llama3`). Another tag is a different model (`llama3:8b` ≠ `llama3`).
 */
export function modelListed(models: readonly string[], model: string): boolean {
  const want = model.trim();
  if (!want) return false;
  const untagged = (id: string) => id.replace(/:latest$/i, "");
  const wantBase = untagged(want);
  return models.some((m) => {
    if (m === want || m.endsWith(`/${want}`)) return true;
    const base = untagged(m);
    return base === wantBase || base.endsWith(`/${wantBase}`);
  });
}

/** Ids of a `/models` reply: OpenAI `{ data: [{ id }] }`, Ollama-native `{ models: [{ name }] }` or a bare list. */
export function modelIds(json: unknown, stripModelsPrefix: boolean): string[] {
  const obj = (typeof json === "object" && json !== null ? json : {}) as { data?: unknown; models?: unknown };
  const list: unknown[] = Array.isArray(obj.data) ? obj.data : Array.isArray(obj.models) ? obj.models : Array.isArray(json) ? json : [];
  const ids = new Set<string>();
  for (const item of list) {
    const rec = (typeof item === "object" && item !== null ? item : {}) as Record<string, unknown>;
    const pick = [item, rec.id, rec.name, rec.model].find((v): v is string => typeof v === "string" && v.trim() !== "");
    const id = (pick ?? "").trim();
    const clean = stripModelsPrefix ? id.replace(/^models\//, "") : id;
    if (clean) ids.add(clean);
  }
  return [...ids];
}

async function readText(response: LlmHttpResponse, signal: AbortSignal, limit: number, touch?: () => void): Promise<string> {
  let text = "";
  for await (const chunk of response.body) {
    if (signal.aborted) throw abortReason(signal);
    touch?.();
    text += chunk;
    if (text.length > limit) {
      response.cancel();
      if (limit === ERROR_BODY_LIMIT) return text.slice(0, limit);
      throw new LlmError("server", `the reply is larger than ${Math.round(limit / 1024 / 1024)} MB`, { retryable: false });
    }
  }
  return text;
}

function notOpenAiReply(text: string): string {
  const looksHtml = /^\s*</.test(text);
  return looksHtml
    ? "the server answered with a web page, not an OpenAI-compatible reply (check the URL: it usually ends in /v1)"
    : "the server's reply is not OpenAI-compatible JSON (check the URL: it usually ends in /v1)";
}

/** Message of an error body (`{ error: { message } }`, `{ error: "..." }`, `{ detail }`, or the text itself). */
export function errorBodyMessage(text: string): string {
  const trimmed = text.trim();
  if (!trimmed) return "";
  let json = tryParseJson(trimmed);
  // Gemini's OpenAI endpoint wraps its error object in an array.
  if (Array.isArray(json)) json = json[0];
  if (typeof json === "object" && json !== null) {
    const rec = json as Record<string, unknown>;
    const fromError = readErrorPayload(rec)?.message;
    if (fromError) return fromError;
    if (typeof rec.detail === "string" && rec.detail.trim()) return rec.detail.trim();
    if (typeof rec.message === "string" && rec.message.trim()) return rec.message.trim();
  }
  if (/^\s*</.test(trimmed)) return ""; // an HTML error page says nothing useful
  return trimmed.length > 300 ? `${trimmed.slice(0, 300)}…` : trimmed;
}

/**
 * How long the server asked to wait: `Retry-After` (seconds or an HTTP date), else Gemini's
 * `google.rpc.RetryInfo.retryDelay` ("41s") in the error body. Capped at 1 hour.
 */
export function retryAfterMs(headers: Record<string, string>, bodyText: string, nowMs: number = Date.now()): number | undefined {
  const header = headers["retry-after"]?.trim();
  let ms: number | undefined;
  if (header) {
    if (/^\d+(\.\d+)?$/.test(header)) ms = Number(header) * 1000;
    else {
      const at = Date.parse(header);
      if (Number.isFinite(at)) ms = Math.max(0, at - nowMs);
    }
  }
  if (ms === undefined) {
    const delay = /"retryDelay"\s*:\s*"(\d+(?:\.\d+)?)s"/.exec(bodyText);
    if (delay) ms = Number(delay[1]) * 1000;
  }
  return ms === undefined || !Number.isFinite(ms) ? undefined : Math.min(3_600_000, Math.round(ms));
}

/** Maps an HTTP status (and the server's message) to an `LlmError`. */
export function httpError(
  status: number,
  bodyText: string,
  redact: (s: string) => string = (s) => redactKeys(s),
  headers: Record<string, string> = {},
): LlmError {
  const detail = redact(errorBodyMessage(bodyText));
  const message = detail ? `HTTP ${status}: ${detail}` : `HTTP ${status}`;
  if (status === 401 || status === 403) return new LlmError("unauthorized", message, { status });
  if (status === 429 || status === 503) {
    const wait = retryAfterMs(headers, bodyText);
    if (status === 429) return new LlmError("rate_limited", message, { status, retryAfterMs: wait });
    return new LlmError("server", message, { status, retryAfterMs: wait });
  }
  if (status === 408) return new LlmError("timeout", message, { status });
  if (status >= 500) return new LlmError("server", message, { status });
  if (status === 404) {
    return new LlmError("bad_request", detail ? message : `HTTP 404: not found (check the URL and the model name)`, { status });
  }
  return new LlmError("bad_request", message, { status });
}

/**
 * Whether a failed streamed call was refused because of the streaming itself: a 400/422 whose message is about
 * `stream` (OpenAI: "Your organization must be verified to stream this model"; servers without SSE: "stream is not
 * supported"). Other 4xx (unknown model, bad key) fail the same way without streaming, so they are not retried.
 */
export function refusesStreaming(e: unknown): boolean {
  return (
    e instanceof LlmError &&
    e.kind === "bad_request" &&
    (e.status === 400 || e.status === 422) &&
    /\bstream/i.test(e.message)
  );
}

function payloadError(error: { message: string; status?: number }, redact: (s: string) => string): LlmError {
  if (error.status !== undefined) return httpError(error.status, JSON.stringify({ error: { message: error.message } }), redact);
  return new LlmError("server", redact(error.message), { retryable: true });
}

/** Any failure inside a call (transport errors, bugs) as an `LlmError` with a redacted message. */
function toLlmError(e: unknown, redact: (s: string) => string): LlmError {
  if (e instanceof LlmError) {
    const message = redact(e.message);
    return message === e.message ? e : new LlmError(e.kind, message, { status: e.status, retryable: e.retryable });
  }
  if (e instanceof LlmTransportError) {
    return new LlmError("network", redact(e.message || "the server could not be reached"), { retryable: !e.unavailable });
  }
  if (e instanceof TypeError) return new LlmError("network", redact(e.message || "the server could not be reached"));
  const message = e instanceof Error ? e.message : typeof e === "object" && e !== null && "message" in e ? String((e as { message: unknown }).message) : String(e);
  return new LlmError("network", redact(message));
}
