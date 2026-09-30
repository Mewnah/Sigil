/**
 * Streaming support of the shared LLM client: an incremental Server-Sent Events parser (chunks can split anywhere,
 * including inside a line or a CRLF pair) and parsers for the JSON payloads of the three reply shapes the client
 * reads:
 * - OpenAI Chat Completions chunks (`choices[0].delta.content`), streamed or whole;
 * - OpenAI legacy Completions (`choices[0].text`);
 * - llama-server's native `/completion` (`content`, `stop`, `stopped_limit`, ...), used for raw prompts (Seed-X).
 *
 * Errors inside a 200 stream are reported as payloads with `error` (OpenRouter sends `data: {"error": ...}`,
 * llama-server an `error:` line). Pure functions: no window, no Tauri (tested in `__tests__/stream.test.ts`).
 */
import type { LlmFinishReason } from "./types";
import { contentText } from "./sanitize";

export interface SseEvent {
  /** `event:` field; "message" when absent. llama-server's non-standard `error:` lines arrive as event "error". */
  event: string;
  data: string;
}

/**
 * Incremental SSE parser (WHATWG event-stream rules): `push` text as it arrives and receive the complete events;
 * `end` flushes a final event that was not followed by a blank line. Comments (`: keep-alive`, OpenRouter's
 * `: OPENROUTER PROCESSING`) and `id` / `retry` fields are ignored.
 */
export class SseParser {
  #buffer = "";
  #data: string[] = [];
  #event = "";
  #hasData = false;

  push(chunk: string): SseEvent[] {
    this.#buffer += chunk;
    const events: SseEvent[] = [];
    for (;;) {
      const nl = this.#buffer.search(/[\r\n]/);
      if (nl < 0) break;
      // A CR at the very end may be the first half of a CRLF split across chunks: wait for more.
      if (this.#buffer[nl] === "\r" && nl === this.#buffer.length - 1) break;
      const line = this.#buffer.slice(0, nl);
      const skip = this.#buffer[nl] === "\r" && this.#buffer[nl + 1] === "\n" ? 2 : 1;
      this.#buffer = this.#buffer.slice(nl + skip);
      this.#line(line, events);
    }
    return events;
  }

  end(): SseEvent[] {
    const events: SseEvent[] = [];
    if (this.#buffer !== "") {
      this.#line(this.#buffer.replace(/\r$/, ""), events);
      this.#buffer = "";
    }
    this.#dispatch(events);
    return events;
  }

  #line(line: string, events: SseEvent[]) {
    if (line === "") {
      this.#dispatch(events);
      return;
    }
    if (line.startsWith(":")) return;
    const colon = line.indexOf(":");
    const field = colon < 0 ? line : line.slice(0, colon);
    let value = colon < 0 ? "" : line.slice(colon + 1);
    if (value.startsWith(" ")) value = value.slice(1);
    switch (field) {
      case "data":
        this.#data.push(value);
        this.#hasData = true;
        break;
      case "event":
        this.#event = value;
        break;
      case "error":
        // llama-server: `error: {json}` (not a standard SSE field).
        this.#event = "error";
        this.#data.push(value);
        this.#hasData = true;
        break;
      default:
        break;
    }
  }

  #dispatch(events: SseEvent[]) {
    if (this.#hasData) events.push({ event: this.#event || "message", data: this.#data.join("\n") });
    this.#data = [];
    this.#event = "";
    this.#hasData = false;
  }
}

/** One parsed payload (a stream chunk or a whole reply). */
export interface LlmPayload {
  /** Reply text carried by this payload (a delta when streaming, the whole reply otherwise). */
  content: string;
  finishReason?: LlmFinishReason;
  model?: string;
  usage?: { promptTokens: number; completionTokens: number };
  /** Server-reported failure inside a successful HTTP response. */
  error?: { message: string; status?: number };
}

type Json = Record<string, unknown>;

function isObject(v: unknown): v is Json {
  return typeof v === "object" && v !== null && !Array.isArray(v);
}

export function mapFinishReason(value: unknown): LlmFinishReason | undefined {
  if (typeof value !== "string" || value === "") return undefined;
  switch (value) {
    case "stop":
    case "eos":
    case "end_turn":
    case "stop_sequence":
    case "tool_calls":
      return "stop";
    case "length":
    case "max_tokens":
    case "limit":
      return "length";
    case "content_filter":
    case "safety":
      return "content_filter";
    default:
      return "unknown";
  }
}

function readUsage(v: unknown): LlmPayload["usage"] {
  if (!isObject(v)) return undefined;
  const prompt = Number(v.prompt_tokens);
  const completion = Number(v.completion_tokens);
  if (!Number.isFinite(prompt) && !Number.isFinite(completion)) return undefined;
  return { promptTokens: Number.isFinite(prompt) ? prompt : 0, completionTokens: Number.isFinite(completion) ? completion : 0 };
}

/** `{ error: { message, code } }` (OpenAI, OpenRouter, llama-server) or `{ error: "text" }` (Ollama). */
export function readErrorPayload(v: unknown): LlmPayload["error"] {
  if (!isObject(v)) return undefined;
  const err = v.error;
  if (typeof err === "string" && err.trim()) return { message: err.trim() };
  return isObject(err) ? readErrorObject(err) : undefined;
}

/** An error object itself: `{ message, code | status }` (llama-server's `error:` stream lines carry it bare). */
export function readErrorObject(err: unknown): LlmPayload["error"] {
  if (!isObject(err)) return undefined;
  const message = typeof err.message === "string" && err.message.trim() ? err.message.trim() : JSON.stringify(err);
  const code = typeof err.code === "number" ? err.code : typeof err.status === "number" ? err.status : Number(err.code);
  return { message, status: Number.isInteger(code) && code >= 400 && code < 600 ? code : undefined };
}

/** A Chat Completions reply or stream chunk. */
export function parseChatPayload(json: unknown): LlmPayload {
  if (!isObject(json)) return { content: "" };
  const error = readErrorPayload(json);
  const choice = Array.isArray(json.choices) && isObject(json.choices[0]) ? (json.choices[0] as Json) : undefined;
  const delta = choice && isObject(choice.delta) ? choice.delta : undefined;
  const message = choice && isObject(choice.message) ? choice.message : undefined;
  const content = contentText(delta ? delta.content : message ? message.content : choice?.text);
  return {
    content,
    finishReason: mapFinishReason(choice?.finish_reason),
    model: typeof json.model === "string" && json.model ? json.model : undefined,
    usage: readUsage(json.usage),
    // OpenRouter reports a mid-stream failure as a chunk with both `error` and a `finish_reason: "error"` choice.
    error,
  };
}

/** A legacy Completions (`choices[0].text`) or llama-server `/completion` reply or stream chunk. */
export function parseCompletionPayload(json: unknown): LlmPayload {
  if (!isObject(json)) return { content: "" };
  const error = readErrorPayload(json);
  if (Array.isArray(json.choices)) {
    const choice = isObject(json.choices[0]) ? (json.choices[0] as Json) : undefined;
    return {
      content: typeof choice?.text === "string" ? choice.text : "",
      finishReason: mapFinishReason(choice?.finish_reason),
      model: typeof json.model === "string" && json.model ? json.model : undefined,
      usage: readUsage(json.usage),
      error,
    };
  }
  if (error && typeof json.content !== "string") return { content: "", error };
  // llama-server native: `stop` marks the last chunk; `stop_type` ("eos" | "limit" | "word") on newer builds.
  let finishReason: LlmFinishReason | undefined;
  if (json.stop === true || typeof json.stop_type === "string") {
    if (json.stopped_limit === true || json.stop_type === "limit") finishReason = "length";
    else if (json.stopped_eos === true || json.stopped_word === true || json.stop_type === "eos" || json.stop_type === "word") finishReason = "stop";
    else finishReason = mapFinishReason(json.stop_type) ?? "stop";
  }
  const predicted = Number(json.tokens_predicted);
  const evaluated = Number(json.tokens_evaluated);
  return {
    content: typeof json.content === "string" ? json.content : "",
    finishReason,
    model: typeof json.model === "string" && json.model ? json.model : undefined,
    usage:
      finishReason && Number.isFinite(predicted)
        ? { promptTokens: Number.isFinite(evaluated) ? evaluated : 0, completionTokens: predicted }
        : undefined,
  };
}

/** Parses JSON, or returns `undefined` for a malformed payload. */
export function tryParseJson(text: string): unknown {
  try {
    return JSON.parse(text);
  } catch {
    return undefined;
  }
}

/** The SSE `data` value that ends an OpenAI stream. */
export const SSE_DONE = "[DONE]";
