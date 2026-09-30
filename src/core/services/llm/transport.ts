/**
 * HTTP transports of the shared LLM client. Both deliver the same shape: the status and headers once the response
 * starts, then the body as text chunks (decoded UTF-8) in order.
 * - `webview`: `fetch` from the renderer (streams through `ReadableStream`).
 * - `rust`: `plugin:llm-server|llm_proxy_chat`, which performs the request in Rust and streams `response` /
 *   `chunk` / `done` / `error` events through a Tauri Channel (no CORS, no CSP; used for LAN hosts). Aborting calls
 *   `llm_proxy_cancel`.
 *
 * Failures before a response exists reject with `LlmTransportError` (the client may retry on another transport);
 * aborts reject with the signal's reason.
 */
import { Channel, invoke } from "@tauri-apps/api/core";
import type { RustLlmProxyEvent, RustLlmProxyRequest, RustLlmServerError } from "@/types";

export interface LlmHttpRequest {
  url: string;
  method: "GET" | "POST";
  /** Includes the auth header; never logged. */
  headers: Record<string, string>;
  /** JSON body text. */
  body?: string;
  /** The reply is SSE (the Rust proxy forwards it in chunks instead of one piece). */
  stream: boolean;
  /** Overall deadline the Rust proxy applies too (the client enforces its own). */
  timeoutMs?: number;
}

export interface LlmHttpResponse {
  status: number;
  /** Lower-case header names. */
  headers: Record<string, string>;
  /** Body text in arrival order; rejects with the abort reason or an `LlmTransportError`. */
  body: AsyncIterable<string>;
  /** Stops reading the body (releases the connection); safe to call more than once. */
  cancel(): void;
}

export type LlmHttpTransport = (request: LlmHttpRequest, signal: AbortSignal) => Promise<LlmHttpResponse>;

/** The request never got a response (connection refused, DNS, CORS, proxy failure) or its body broke off. */
export class LlmTransportError extends Error {
  constructor(
    message: string,
    /** True when no status line was received, so another transport may still reach the server. */
    readonly beforeResponse: boolean,
    /** The transport is not available at all here (the Rust proxy outside Tauri or not built). */
    readonly unavailable = false,
  ) {
    super(message);
    this.name = "LlmTransportError";
  }
}

export function abortReason(signal: AbortSignal): unknown {
  return signal.reason ?? new DOMException("The operation was aborted.", "AbortError");
}

/** `fetch` from the webview. */
export function createWebviewTransport(fetchImpl?: typeof fetch): LlmHttpTransport {
  return async (request, signal) => {
    const doFetch = fetchImpl ?? globalThis.fetch;
    if (typeof doFetch !== "function") throw new LlmTransportError("fetch is not available", true, true);
    if (signal.aborted) throw abortReason(signal);
    let res: Response;
    try {
      res = await doFetch(request.url, {
        method: request.method,
        headers: request.headers,
        body: request.method === "POST" ? request.body : undefined,
        signal,
        // Never send cookies to model servers; never serve a cached completion.
        credentials: "omit",
        cache: "no-store",
      });
    } catch (e) {
      if (signal.aborted) throw abortReason(signal);
      // Browsers say only "Failed to fetch" for refused connections, DNS failures and CORS rejections alike.
      throw new LlmTransportError(e instanceof Error ? e.message : String(e), true);
    }
    const headers: Record<string, string> = {};
    res.headers.forEach((value, key) => {
      headers[key.toLowerCase()] = value;
    });
    const reader = res.body?.getReader();
    let cancelled = false;
    const cancel = () => {
      if (cancelled) return;
      cancelled = true;
      reader?.cancel().catch(() => {});
    };
    async function* body(): AsyncGenerator<string> {
      if (!reader) {
        const text = await res.text();
        if (text) yield text;
        return;
      }
      const decoder = new TextDecoder("utf-8");
      try {
        for (;;) {
          let chunk: ReadableStreamReadResult<Uint8Array>;
          try {
            chunk = await reader.read();
          } catch (e) {
            if (signal.aborted) throw abortReason(signal);
            throw new LlmTransportError(e instanceof Error ? e.message : String(e), false);
          }
          if (chunk.done) break;
          const text = decoder.decode(chunk.value, { stream: true });
          if (text) yield text;
        }
        const tail = decoder.decode();
        if (tail) yield tail;
      } finally {
        cancel();
      }
    }
    return { status: res.status, headers, body: body(), cancel };
  };
}

/** Pull-based queue fed by Channel callbacks. */
class ChunkQueue implements AsyncIterable<string> {
  #items: string[] = [];
  #ended = false;
  #error: { value: unknown } | null = null;
  #wake: (() => void) | null = null;

  push(item: string) {
    if (this.#ended || this.#error) return;
    this.#items.push(item);
    this.#notify();
  }
  end() {
    this.#ended = true;
    this.#notify();
  }
  fail(error: unknown) {
    if (this.#ended || this.#error) return;
    this.#error = { value: error };
    this.#notify();
  }
  get settled() {
    return this.#ended || this.#error !== null;
  }
  #notify() {
    const wake = this.#wake;
    this.#wake = null;
    wake?.();
  }
  async *[Symbol.asyncIterator](): AsyncGenerator<string> {
    for (;;) {
      if (this.#items.length > 0) {
        yield this.#items.shift() as string;
        continue;
      }
      if (this.#error) throw this.#error.value;
      if (this.#ended) return;
      await new Promise<void>((resolve) => (this.#wake = resolve));
    }
  }
}

let proxyRequestCounter = 0;

function proxyRequestId(): string {
  proxyRequestCounter += 1;
  const random = Math.random().toString(36).slice(2, 10);
  return `llm-${Date.now().toString(36)}-${proxyRequestCounter}-${random}`;
}

/** How long to wait for the terminal channel event after the command resolved (delivery order is not guaranteed). */
const PROXY_END_GRACE_MS = 3_000;

function isRustError(e: unknown): e is RustLlmServerError {
  return typeof e === "object" && e !== null && typeof (e as RustLlmServerError).kind === "string" && typeof (e as RustLlmServerError).message === "string";
}

/** `plugin:llm-server|llm_proxy_chat` (Tauri only). */
export function createRustTransport(): LlmHttpTransport {
  return (request, signal) =>
    new Promise<LlmHttpResponse>((resolve, reject) => {
      if (signal.aborted) {
        reject(abortReason(signal));
        return;
      }
      const requestId = proxyRequestId();
      const queue = new ChunkQueue();
      let responded = false;
      let finished = false;
      let graceTimer: ReturnType<typeof setTimeout> | undefined;

      const cleanup = () => {
        finished = true;
        if (graceTimer !== undefined) clearTimeout(graceTimer);
        signal.removeEventListener("abort", onAbort);
      };
      const failBeforeResponse = (error: unknown) => {
        cleanup();
        reject(error);
      };
      const cancel = () => {
        if (finished) return;
        cleanup();
        queue.end();
        invoke("plugin:llm-server|llm_proxy_cancel", { requestId }).catch(() => {});
      };
      function onAbort() {
        const reason = abortReason(signal);
        if (!finished) invoke("plugin:llm-server|llm_proxy_cancel", { requestId }).catch(() => {});
        if (!responded) {
          failBeforeResponse(reason);
          return;
        }
        cleanup();
        queue.fail(reason);
      }

      let channel: Channel<RustLlmProxyEvent>;
      try {
        channel = new Channel<RustLlmProxyEvent>((event) => {
          // After done / error / abort / cancel nothing more is delivered.
          if (finished) return;
          switch (event.kind) {
            case "response":
              if (responded) return;
              responded = true;
              resolve({
                status: event.status,
                headers: Object.fromEntries(Object.entries(event.headers ?? {}).map(([k, v]) => [k.toLowerCase(), v])),
                body: queue,
                cancel,
              });
              return;
            case "chunk":
              queue.push(event.data);
              return;
            case "done":
              if (!responded) {
                failBeforeResponse(new LlmTransportError("the proxy ended without a response", true));
                return;
              }
              cleanup();
              queue.end();
              return;
            case "error": {
              const message = event.message || "proxy error";
              if (signal.aborted) return; // onAbort already settled everything.
              if (!responded) {
                failBeforeResponse(new LlmTransportError(message, true));
                return;
              }
              cleanup();
              queue.fail(new LlmTransportError(message, false));
              return;
            }
          }
        });
      } catch (e) {
        // No Tauri IPC here (web mode).
        failBeforeResponse(new LlmTransportError(e instanceof Error ? e.message : String(e), true, true));
        return;
      }

      signal.addEventListener("abort", onAbort, { once: true });
      const wire: RustLlmProxyRequest = {
        requestId,
        url: request.url,
        method: request.method,
        headers: request.headers,
        body: request.body ?? null,
        stream: request.stream,
        timeoutMs: request.timeoutMs ?? null,
      };
      invoke("plugin:llm-server|llm_proxy_chat", { request: wire, onEvent: channel }).then(
        () => {
          if (finished) return;
          // The command resolves after its last event, but channel messages can be delivered after the reply.
          graceTimer = setTimeout(() => {
            if (finished) return;
            if (!responded) {
              failBeforeResponse(new LlmTransportError("the proxy ended without a response", true));
              return;
            }
            cleanup();
            queue.end();
          }, PROXY_END_GRACE_MS);
        },
        (e: unknown) => {
          if (finished || signal.aborted) return;
          const unavailable = isRustError(e) && e.kind === "not_implemented";
          const message = isRustError(e) ? e.message || e.kind : e instanceof Error ? e.message : String(e);
          const error = new LlmTransportError(message, !responded, unavailable);
          if (!responded) {
            failBeforeResponse(error);
            return;
          }
          cleanup();
          queue.fail(error);
        },
      );
    });
}
