import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";

vi.mock("@tauri-apps/api/core", () => ({ invoke: vi.fn(), isTauri: () => false, Channel: class {} }));

import { buildChatBody, createLlmClient, httpError, modelListed, refusesStreaming, resetLlmTransportMemory, retryAfterMs } from "../client";
import { createWebviewTransport, LlmHttpResponse, LlmHttpTransport, LlmTransportError } from "../transport";
import { LlmDelta, LlmEndpoint, LlmError } from "../types";

const enc = new TextEncoder();

function endpoint(patch: Partial<LlmEndpoint> = {}): LlmEndpoint {
  return {
    provider: "custom",
    baseUrl: "http://127.0.0.1:8080/v1",
    apiKey: "",
    keyHeader: "bearer",
    model: "m",
    headers: {},
    transport: "webview",
    ...patch,
  };
}

interface FetchCall {
  url: string;
  init: RequestInit;
  body: Record<string, unknown>;
  push(text: string | Uint8Array): void;
  close(): void;
  cancelled: boolean;
}

/** A fetch whose response body the test feeds; aborting the request errors the body like a browser does. */
function controlledFetch(response: { status?: number; headers?: Record<string, string>; chunks?: (string | Uint8Array)[] } = {}) {
  const calls: FetchCall[] = [];
  const fetchMock = vi.fn(async (url: string, init: RequestInit) => {
    const signal = init.signal as AbortSignal;
    if (signal?.aborted) throw signal.reason;
    let ctl!: ReadableStreamDefaultController<Uint8Array>;
    const call: FetchCall = {
      url,
      init,
      body: init.body ? JSON.parse(String(init.body)) : {},
      push: (t) => ctl.enqueue(typeof t === "string" ? enc.encode(t) : t),
      close: () => ctl.close(),
      cancelled: false,
    };
    const stream = new ReadableStream<Uint8Array>({
      start(c) {
        ctl = c;
      },
      cancel() {
        call.cancelled = true;
      },
    });
    signal?.addEventListener("abort", () => {
      call.cancelled = true;
      try {
        ctl.error(signal.reason);
      } catch {
        /* already closed */
      }
    });
    calls.push(call);
    if (response.chunks) {
      for (const c of response.chunks) call.push(c);
      call.close();
    }
    return new Response(stream, {
      status: response.status ?? 200,
      headers: response.headers ?? { "content-type": "text/event-stream" },
    });
  });
  return { fetchMock, calls, transport: createWebviewTransport(fetchMock as unknown as typeof fetch) };
}

const sse = (payload: unknown) => `data: ${JSON.stringify(payload)}\n\n`;
const delta = (content: string, extra: Record<string, unknown> = {}) => sse({ choices: [{ delta: { content }, ...extra }], model: "served-model" });
const json = { "content-type": "application/json" };
const live = () => new AbortController().signal;
const tick = () => new Promise<void>((r) => setTimeout(r, 0));

async function until(cond: () => boolean) {
  for (let i = 0; i < 200 && !cond(); i++) await tick();
  expect(cond()).toBe(true);
}

beforeEach(() => {
  resetLlmTransportMemory();
});

afterEach(() => {
  vi.useRealTimers();
});

describe("chat request", () => {
  it("posts OpenAI-compatible JSON with the key only in the header", async () => {
    const f = controlledFetch({ headers: json, chunks: [JSON.stringify({ choices: [{ message: { content: "<think>x</think> Hewwo" }, finish_reason: "stop" }], usage: { prompt_tokens: 3, completion_tokens: 4 }, model: "real" })] });
    const client = createLlmClient(endpoint({ apiKey: "sk-secret-123", baseUrl: "http://127.0.0.1:8080/v1" }), { transports: { webview: f.transport } });
    const result = await client.chat({ messages: [{ role: "user", content: "hello" }], maxTokens: 300, temperature: 0.2 }, { signal: live() });

    expect(result).toMatchObject({ text: "Hewwo", raw: "<think>x</think> Hewwo", finishReason: "stop", model: "real", usage: { promptTokens: 3, completionTokens: 4 } });
    expect(result.firstTokenMs).toBeUndefined();
    const call = f.calls[0];
    expect(call.url).toBe("http://127.0.0.1:8080/v1/chat/completions");
    expect(call.init.method).toBe("POST");
    expect(call.init.credentials).toBe("omit");
    expect((call.init.headers as Record<string, string>).Authorization).toBe("Bearer sk-secret-123");
    expect(call.url).not.toContain("sk-secret");
    expect(call.body).toEqual({ model: "m", messages: [{ role: "user", content: "hello" }], max_tokens: 300, temperature: 0.2 });
  });

  it("sends no auth header without a key, and Gemini's key as x-goog-api-key", async () => {
    const reply = JSON.stringify({ choices: [{ message: { content: "ok" } }] });
    const a = controlledFetch({ headers: json, chunks: [reply] });
    await createLlmClient(endpoint(), { transports: { webview: a.transport } }).chat({ messages: [] }, { signal: live() });
    const ha = a.calls[0].init.headers as Record<string, string>;
    expect(ha.Authorization).toBeUndefined();
    expect(ha["x-goog-api-key"]).toBeUndefined();

    const g = controlledFetch({ headers: json, chunks: [reply] });
    const gemini = endpoint({ provider: "gemini", baseUrl: "https://generativelanguage.googleapis.com/v1beta/openai", apiKey: "AIzaXYZ", keyHeader: "x-goog-api-key" });
    await createLlmClient(gemini, { transports: { webview: g.transport } }).chat({ messages: [] }, { signal: live() });
    const hg = g.calls[0].init.headers as Record<string, string>;
    expect(hg["x-goog-api-key"]).toBe("AIzaXYZ");
    expect(hg.Authorization).toBeUndefined();
  });

  it("maps token limits, reasoning and sampling per provider", () => {
    const req = { messages: [], maxTokens: 512, reasoningEffort: "none" as const, topK: 40, minP: 0.05, repetitionPenalty: 1.1, seed: 7.2, stop: ["\n\n"] };
    const openai = buildChatBody(endpoint({ provider: "openai" }), req, false);
    expect(openai).toMatchObject({ max_completion_tokens: 512, reasoning_effort: "none", seed: 7, stop: ["\n\n"] });
    expect(openai).not.toHaveProperty("max_tokens");
    expect(openai).not.toHaveProperty("top_k"); // OpenAI rejects unknown parameters
    expect(openai).not.toHaveProperty("stream");

    const or = buildChatBody(endpoint({ provider: "openrouter" }), { ...req, reasoningEffort: "low" }, true);
    expect(or).toMatchObject({ max_tokens: 512, reasoning: { effort: "low" }, top_k: 40, min_p: 0.05, repetition_penalty: 1.1, stream: true });
    expect(or).not.toHaveProperty("reasoning_effort");

    const gemini3 = buildChatBody(endpoint({ provider: "gemini", model: "gemini-3.5-flash-lite" }), req, false);
    expect(gemini3.reasoning_effort).toBe("minimal"); // Gemini 3 cannot turn thinking off
    expect(gemini3).not.toHaveProperty("top_k");
    expect(buildChatBody(endpoint({ provider: "gemini", model: "gemini-2.5-flash-lite" }), req, false).reasoning_effort).toBe("none");
    expect(buildChatBody(endpoint({ provider: "gemini", model: "models/gemini-2.0-flash" }), req, false).reasoning_effort).toBe("none");

    const bundled = buildChatBody(endpoint({ provider: "bundled" }), req, true);
    expect(bundled).toMatchObject({ chat_template_kwargs: { enable_thinking: false }, repeat_penalty: 1.1, cache_prompt: true, top_k: 40 });
    expect(bundled).not.toHaveProperty("reasoning_effort");
    expect(buildChatBody(endpoint({ provider: "bundled" }), { ...req, reasoningEffort: "high" }, false).reasoning_effort).toBe("high");

    const none = buildChatBody(endpoint(), { messages: [], reasoningEffort: "default" }, false);
    expect(none).not.toHaveProperty("reasoning_effort");
    expect(none).not.toHaveProperty("max_tokens");
    // `extra` is merged last.
    expect(buildChatBody(endpoint(), { messages: [], maxTokens: 10, extra: { max_tokens: 99, n_probs: 1 } }, false)).toMatchObject({ max_tokens: 99, n_probs: 1 });
  });
});

describe("chat streaming", () => {
  it("delivers sanitized deltas in order, across arbitrary chunk and UTF-8 boundaries", async () => {
    const body = [delta("<think>plan"), delta("</think>"), delta("こんにちは"), delta(" world", { finish_reason: "stop" }), "data: [DONE]\n\n"].join("");
    const bytes = enc.encode(body);
    // Split the byte stream every 7 bytes: lines, JSON and multi-byte characters all break mid-way.
    const pieces: Uint8Array[] = [];
    for (let i = 0; i < bytes.length; i += 7) pieces.push(bytes.slice(i, i + 7));
    const f = controlledFetch({ chunks: pieces });
    let clock = 1000;
    const client = createLlmClient(endpoint(), { transports: { webview: f.transport }, now: () => (clock += 10) });
    const deltas: LlmDelta[] = [];
    const result = await client.chat({ messages: [{ role: "user", content: "hi" }] }, { signal: live(), onDelta: (d) => deltas.push(d) });

    expect(f.calls[0].body.stream).toBe(true);
    expect(deltas.map((d) => d.delta)).toEqual(["<think>plan", "</think>", "こんにちは", " world"]);
    expect(deltas.map((d) => d.text)).toEqual(["", "", "こんにちは", "こんにちは world"]);
    expect(result).toMatchObject({ text: "こんにちは world", finishReason: "stop", model: "served-model" });
    expect(result.firstTokenMs).toBeGreaterThan(0);
    expect(result.totalMs).toBeGreaterThanOrEqual(result.firstTokenMs!);
  });

  it("rejects with the caller's abort reason mid-stream, cancels the request and sends no more deltas", async () => {
    const f = controlledFetch();
    const controller = new AbortController();
    const deltas: string[] = [];
    const pending = createLlmClient(endpoint(), { transports: { webview: f.transport } }).chat(
      { messages: [] },
      { signal: controller.signal, onDelta: (d) => deltas.push(d.text) },
    );
    await until(() => f.calls.length === 1);
    f.calls[0].push(delta("Hel"));
    await until(() => deltas.length === 1);

    const reason = new Error("user stopped");
    controller.abort(reason);
    await expect(pending).rejects.toBe(reason);
    expect(f.calls[0].cancelled).toBe(true);
    expect((f.calls[0].init.signal as AbortSignal).aborted).toBe(true);
    try {
      f.calls[0].push(delta("lo"));
    } catch {
      /* the stream is already errored */
    }
    await tick();
    expect(deltas).toEqual(["Hel"]);
  });

  it("rejects with an AbortError DOMException when aborted without a reason", async () => {
    const f = controlledFetch();
    const controller = new AbortController();
    const pending = createLlmClient(endpoint(), { transports: { webview: f.transport } }).chat({ messages: [] }, { signal: controller.signal, onDelta: () => {} });
    await until(() => f.calls.length === 1);
    controller.abort();
    const err = await pending.catch((e) => e);
    expect(err).not.toBeInstanceOf(LlmError);
    expect(err.name).toBe("AbortError");
  });

  it("does not send anything when the signal is already aborted", async () => {
    const f = controlledFetch();
    const controller = new AbortController();
    const reason = new Error("gone");
    controller.abort(reason);
    await expect(createLlmClient(endpoint(), { transports: { webview: f.transport } }).chat({ messages: [] }, { signal: controller.signal })).rejects.toBe(reason);
    expect(f.fetchMock).not.toHaveBeenCalled();
  });

  it("times out with an LlmError when its own deadline passes, aborting the request", async () => {
    vi.useFakeTimers();
    const f = controlledFetch();
    const pending = createLlmClient(endpoint(), { transports: { webview: f.transport } })
      .chat({ messages: [], timeoutMs: 5_000 }, { signal: live(), onDelta: () => {} })
      .catch((e) => e);
    await vi.advanceTimersByTimeAsync(1);
    f.calls[0].push(delta("slow"));
    await vi.advanceTimersByTimeAsync(5_000);
    const err = await pending;
    expect(err).toBeInstanceOf(LlmError);
    expect(err.kind).toBe("timeout");
    expect(err.retryable).toBe(true);
    expect(f.calls[0].cancelled).toBe(true);
  });

  it("cuts off a stream that goes silent", async () => {
    vi.useFakeTimers();
    const f = controlledFetch();
    const pending = createLlmClient(endpoint(), { transports: { webview: f.transport }, stallTimeoutMs: 2_000 })
      .chat({ messages: [] }, { signal: live(), onDelta: () => {} })
      .catch((e) => e);
    await vi.advanceTimersByTimeAsync(1);
    for (let i = 0; i < 5; i++) {
      f.calls[0].push(delta("x"));
      await vi.advanceTimersByTimeAsync(1_500); // chunks keep the stream alive
    }
    await vi.advanceTimersByTimeAsync(2_000);
    const err = await pending;
    expect(err).toBeInstanceOf(LlmError);
    expect(err.kind).toBe("timeout");
    expect(err.message).toContain("sent nothing");
  });

  it("accepts a whole JSON reply from a server that ignored stream", async () => {
    const f = controlledFetch({ headers: json, chunks: [JSON.stringify({ choices: [{ message: { content: "whole" }, finish_reason: "stop" }] })] });
    const deltas: string[] = [];
    const result = await createLlmClient(endpoint(), { transports: { webview: f.transport } }).chat({ messages: [] }, { signal: live(), onDelta: (d) => deltas.push(d.text) });
    expect(result.text).toBe("whole");
    expect(deltas).toEqual(["whole"]);
  });

  it("keeps streaming when a delta handler throws", async () => {
    const warn = vi.spyOn(console, "warn").mockImplementation(() => {});
    const f = controlledFetch({ chunks: [delta("a"), delta("b"), "data: [DONE]\n\n"] });
    const result = await createLlmClient(endpoint(), { transports: { webview: f.transport } }).chat(
      { messages: [] },
      {
        signal: live(),
        onDelta: () => {
          throw new Error("consumer bug");
        },
      },
    );
    expect(result.text).toBe("ab");
    expect(warn).toHaveBeenCalledTimes(1);
  });

  it("stops reading at [DONE]", async () => {
    const f = controlledFetch();
    const pending = createLlmClient(endpoint(), { transports: { webview: f.transport } }).chat({ messages: [] }, { signal: live(), onDelta: () => {} });
    await until(() => f.calls.length === 1);
    f.calls[0].push(delta("done") + "data: [DONE]\n\n");
    await expect(pending).resolves.toMatchObject({ text: "done" });
    expect(f.calls[0].cancelled).toBe(true);
  });
});

describe("chat errors", () => {
  async function failWith(status: number, body: string, ep = endpoint()): Promise<LlmError> {
    const f = controlledFetch({ status, headers: json, chunks: [body] });
    return createLlmClient(ep, { transports: { webview: f.transport } })
      .chat({ messages: [] }, { signal: live() })
      .then(
        () => {
          throw new Error(`expected HTTP ${status} to fail`);
        },
        (e: unknown) => e as LlmError,
      );
  }

  it("classifies HTTP failures and redacts keys from their messages", async () => {
    const unauthorized = await failWith(401, JSON.stringify({ error: { message: "Incorrect API key provided: sk-proj-abc123456789xyz" } }), endpoint({ apiKey: "sk-proj-abc123456789xyz" }));
    expect(unauthorized).toBeInstanceOf(LlmError);
    expect(unauthorized.kind).toBe("unauthorized");
    expect(unauthorized.status).toBe(401);
    expect(unauthorized.retryable).toBe(false);
    expect(unauthorized.message).not.toContain("abc123456789xyz");

    const limited = await failWith(429, JSON.stringify({ error: { message: "Resource exhausted" } }));
    expect(limited).toMatchObject({ kind: "rate_limited", status: 429, retryable: true });
    expect(limited.message).toBe("HTTP 429: Resource exhausted");

    expect(await failWith(503, "upstream down")).toMatchObject({ kind: "server", retryable: true });
    expect(await failWith(400, JSON.stringify({ error: "model 'x' not found" }))).toMatchObject({ kind: "bad_request", message: "HTTP 400: model 'x' not found" });
    const notFound = await failWith(404, "<html>Not Found</html>");
    expect(notFound.kind).toBe("bad_request");
    expect(notFound.message).toContain("check the URL");
    // Gemini's OpenAI endpoint wraps errors in an array.
    const gemini = await failWith(400, JSON.stringify([{ error: { code: 400, message: "API key not valid.", status: "INVALID_ARGUMENT" } }]));
    expect(gemini.message).toBe("HTTP 400: API key not valid.");
  });

  it("reports a web page where JSON was expected", async () => {
    const err = await failWith(200, "<!doctype html><title>Login</title>");
    expect(err).toMatchObject({ kind: "server", retryable: false });
    expect(err.message).toContain("web page");
  });

  it("reports errors sent inside a stream", async () => {
    const f = controlledFetch({ chunks: [delta("par"), sse({ error: { message: "Provider returned error", code: 502 }, choices: [{ delta: { content: "" }, finish_reason: "error" }] })] });
    const err = await createLlmClient(endpoint(), { transports: { webview: f.transport } })
      .chat({ messages: [] }, { signal: live(), onDelta: () => {} })
      .catch((e) => e);
    expect(err).toMatchObject({ kind: "server", status: 502 });
    const llama = controlledFetch({ chunks: ['error: {"code":400,"message":"context too long"}\n\n'] });
    const err2 = await createLlmClient(endpoint(), { transports: { webview: llama.transport } })
      .chat({ messages: [] }, { signal: live(), onDelta: () => {} })
      .catch((e) => e);
    expect(err2).toMatchObject({ kind: "bad_request", status: 400 });
  });

  it("tells a reply that ran out of tokens while thinking from an empty one", async () => {
    const cut = controlledFetch({ chunks: [delta("<think>hmm", { finish_reason: "length" }), "data: [DONE]\n\n"] });
    const truncated = await createLlmClient(endpoint(), { transports: { webview: cut.transport } })
      .chat({ messages: [] }, { signal: live(), onDelta: () => {} })
      .catch((e) => e);
    expect(truncated).toMatchObject({ kind: "truncated", retryable: false });

    const blank = controlledFetch({ headers: json, chunks: [JSON.stringify({ choices: [{ message: { content: "  " }, finish_reason: "stop" }] })] });
    const empty = await createLlmClient(endpoint(), { transports: { webview: blank.transport } })
      .chat({ messages: [] }, { signal: live() })
      .catch((e) => e);
    expect(empty).toMatchObject({ kind: "empty" });
  });

  it("maps an unreachable server to a network error", async () => {
    const refused: LlmHttpTransport = async () => {
      throw new LlmTransportError("Failed to fetch", true);
    };
    const err = await createLlmClient(endpoint(), { transports: { webview: refused } })
      .chat({ messages: [] }, { signal: live() })
      .catch((e) => e);
    expect(err).toMatchObject({ kind: "network", retryable: true, message: "Failed to fetch" });
  });

  it("reads how long a rate-limited provider asks to wait", async () => {
    const f = controlledFetch({ status: 429, headers: { ...json, "retry-after": "7" }, chunks: [JSON.stringify({ error: { message: "slow down" } })] });
    const err = await createLlmClient(endpoint(), { transports: { webview: f.transport } })
      .chat({ messages: [] }, { signal: live() })
      .catch((e) => e);
    expect(err).toMatchObject({ kind: "rate_limited", retryAfterMs: 7_000 });
    // Gemini: google.rpc.RetryInfo in the body.
    const gemini = JSON.stringify([
      { error: { code: 429, message: "quota", status: "RESOURCE_EXHAUSTED", details: [{ "@type": "type.googleapis.com/google.rpc.RetryInfo", retryDelay: "41s" }] } },
    ]);
    expect(retryAfterMs({}, gemini)).toBe(41_000);
    expect(retryAfterMs({ "retry-after": "Wed, 21 Oct 2015 07:28:10 GMT" }, "", Date.parse("Wed, 21 Oct 2015 07:28:00 GMT"))).toBe(10_000);
    expect(retryAfterMs({}, "no hint")).toBeUndefined();
    expect(retryAfterMs({ "retry-after": "999999" }, "")).toBe(3_600_000);
    expect(httpError(400, "").retryAfterMs).toBeUndefined();
  });

  it("maps HTTP statuses", () => {
    expect(httpError(403, "").kind).toBe("unauthorized");
    expect(httpError(408, "").kind).toBe("timeout");
    expect(httpError(422, "{}").kind).toBe("bad_request");
    expect(httpError(500, "").message).toBe("HTTP 500");
  });
});

function httpResponse(status: number, chunks: string[], headers: Record<string, string> = json): LlmHttpResponse & { cancelled: boolean } {
  const r = {
    status,
    headers,
    cancelled: false,
    body: (async function* () {
      for (const c of chunks) yield c;
    })(),
    cancel() {
      r.cancelled = true;
    },
  };
  return r;
}

describe("transport fallback", () => {
  const okReply = JSON.stringify({ choices: [{ message: { content: "ok" } }] });
  const auto = endpoint({ baseUrl: "http://localhost:11434/v1", transport: "webview", fallbackTransport: "rust" });

  it("retries through the Rust proxy when the webview cannot reach a local server, then remembers it", async () => {
    const webview = vi.fn<LlmHttpTransport>(async () => {
      throw new LlmTransportError("Failed to fetch", true);
    });
    const rust = vi.fn<LlmHttpTransport>(async () => httpResponse(200, [okReply]));
    const client = createLlmClient(auto, { transports: { webview, rust } });
    await expect(client.chat({ messages: [] }, { signal: live() })).resolves.toMatchObject({ text: "ok" });
    expect(webview).toHaveBeenCalledTimes(1);
    expect(rust).toHaveBeenCalledTimes(1);

    await client.chat({ messages: [] }, { signal: live() });
    expect(webview).toHaveBeenCalledTimes(1); // the origin now goes straight to the proxy
    expect(rust).toHaveBeenCalledTimes(2);

    resetLlmTransportMemory();
    await client.chat({ messages: [] }, { signal: live() });
    expect(webview).toHaveBeenCalledTimes(2);
  });

  it("retries a 403 through the proxy (Ollama refuses unknown browser origins)", async () => {
    const refused = httpResponse(403, [""]);
    const webview = vi.fn<LlmHttpTransport>(async () => refused);
    const rust = vi.fn<LlmHttpTransport>(async () => httpResponse(200, [okReply]));
    await expect(createLlmClient(auto, { transports: { webview, rust } }).chat({ messages: [] }, { signal: live() })).resolves.toMatchObject({ text: "ok" });
    expect(refused.cancelled).toBe(true);
    expect(rust).toHaveBeenCalledTimes(1);
  });

  it("reports the first failure when the proxy is not available", async () => {
    const webview: LlmHttpTransport = async () => {
      throw new LlmTransportError("Failed to fetch", true);
    };
    const rust: LlmHttpTransport = async () => {
      throw new LlmTransportError("not implemented", true, true);
    };
    const err = await createLlmClient(auto, { transports: { webview, rust } })
      .chat({ messages: [] }, { signal: live() })
      .catch((e) => e);
    expect(err).toMatchObject({ kind: "network", message: "Failed to fetch" });
  });

  it("never falls back without a fallback transport, nor after an abort", async () => {
    const rust = vi.fn<LlmHttpTransport>(async () => httpResponse(200, [okReply]));
    const webview: LlmHttpTransport = async () => {
      throw new LlmTransportError("Failed to fetch", true);
    };
    await createLlmClient(endpoint(), { transports: { webview, rust } })
      .chat({ messages: [] }, { signal: live() })
      .catch(() => {});
    expect(rust).not.toHaveBeenCalled();

    const controller = new AbortController();
    const hanging: LlmHttpTransport = (_req, signal) =>
      new Promise((_resolve, reject) => signal.addEventListener("abort", () => reject(signal.reason)));
    const pending = createLlmClient(auto, { transports: { webview: hanging, rust } }).chat({ messages: [] }, { signal: controller.signal });
    controller.abort(new Error("stop"));
    await expect(pending).rejects.toThrow("stop");
    expect(rust).not.toHaveBeenCalled();
  });

  it("uses the Rust transport directly when the endpoint says so", async () => {
    const webview = vi.fn<LlmHttpTransport>();
    const rust = vi.fn<LlmHttpTransport>(async () => httpResponse(200, [okReply]));
    await createLlmClient(endpoint({ baseUrl: "http://192.168.1.5:1234/v1", transport: "rust" }), { transports: { webview, rust } }).chat({ messages: [] }, { signal: live() });
    expect(webview).not.toHaveBeenCalled();
    expect(rust.mock.calls[0][0]).toMatchObject({ url: "http://192.168.1.5:1234/v1/chat/completions", method: "POST", stream: false });
  });
});

describe("stream refusal", () => {
  // OpenAI's answer to a streamed request for some models of an unverified organization.
  const refusal = JSON.stringify({
    error: {
      message: "Your organization must be verified to stream this model. Please go to: https://platform.openai.com/settings/organization/general and click on Verify Organization.",
      type: "invalid_request_error",
      param: "stream",
      code: "unsupported_value",
    },
  });
  const okReply = JSON.stringify({ choices: [{ message: { content: "Hewwo" }, finish_reason: "stop" }] });
  const openai = endpoint({ provider: "openai", baseUrl: "https://api.openai.com/v1", apiKey: "sk-123456", model: "o3" });
  const bodyOf = (t: ReturnType<typeof vi.fn<LlmHttpTransport>>, i: number) => JSON.parse(String(t.mock.calls[i][0].body));

  beforeEach(() => {
    vi.spyOn(console, "warn").mockImplementation(() => {});
  });
  afterEach(() => {
    vi.restoreAllMocks();
  });

  it("retries a refused stream once without streaming, then stops asking this client's server to stream", async () => {
    const webview = vi.fn<LlmHttpTransport>(async (req) => (req.stream ? httpResponse(400, [refusal]) : httpResponse(200, [okReply])));
    const client = createLlmClient(openai, { transports: { webview } });
    const deltas: string[] = [];
    const request = { messages: [{ role: "user" as const, content: "hello" }], timeoutMs: 5_000 };
    await expect(client.chat(request, { signal: live(), onDelta: (d) => deltas.push(d.text) })).resolves.toMatchObject({ text: "Hewwo" });
    expect(webview).toHaveBeenCalledTimes(2);
    expect(webview.mock.calls[0][0].stream).toBe(true);
    expect(bodyOf(webview, 0).stream).toBe(true);
    expect(webview.mock.calls[1][0].stream).toBe(false);
    expect(bodyOf(webview, 1)).not.toHaveProperty("stream");
    // The retry gets what is left of the deadline, not a fresh one.
    expect(webview.mock.calls[1][0].timeoutMs).toBeGreaterThan(0);
    expect(webview.mock.calls[1][0].timeoutMs).toBeLessThanOrEqual(5_000);
    expect(deltas).toEqual([]); // the reply arrived whole

    // Remembered for this client: the next line is one plain request.
    await expect(client.chat(request, { signal: live(), onDelta: (d) => deltas.push(d.text) })).resolves.toMatchObject({ text: "Hewwo" });
    expect(webview).toHaveBeenCalledTimes(3);
    expect(webview.mock.calls[2][0].stream).toBe(false);

    // A client for other settings tries streaming again.
    await createLlmClient(openai, { transports: { webview } }).chat(request, { signal: live(), onDelta: () => {} });
    expect(webview.mock.calls[3][0].stream).toBe(true);
  });

  it("does the same for raw completions", async () => {
    const webview = vi.fn<LlmHttpTransport>(async (req) =>
      req.stream ? httpResponse(400, [JSON.stringify({ error: "stream is not supported" })]) : httpResponse(200, [JSON.stringify({ content: "Bonjour", stop: true })]),
    );
    const client = createLlmClient(endpoint({ provider: "bundled", baseUrl: "http://127.0.0.1:5123/v1" }), { transports: { webview } });
    await expect(client.complete({ prompt: "p" }, { signal: live(), onDelta: () => {} })).resolves.toMatchObject({ text: "Bonjour" });
    expect(webview).toHaveBeenCalledTimes(2);
    expect(bodyOf(webview, 1).stream).toBe(false);
  });

  it("does not retry other 4xx, a stream that already delivered text, or an aborted call", async () => {
    const unknownModel = vi.fn<LlmHttpTransport>(async () => httpResponse(400, [JSON.stringify({ error: "model 'x' not found" })]));
    await expect(createLlmClient(openai, { transports: { webview: unknownModel } }).chat({ messages: [], stream: true }, { signal: live() })).rejects.toMatchObject({
      kind: "bad_request",
      status: 400,
    });
    expect(unknownModel).toHaveBeenCalledTimes(1);

    const broken = vi.fn<LlmHttpTransport>(async () =>
      httpResponse(200, [delta("Hew"), sse({ error: { message: "stream interrupted", code: 400 } })], { "content-type": "text/event-stream" }),
    );
    const deltas: string[] = [];
    await expect(
      createLlmClient(openai, { transports: { webview: broken } }).chat({ messages: [] }, { signal: live(), onDelta: (d) => deltas.push(d.text) }),
    ).rejects.toMatchObject({ kind: "bad_request" });
    expect(deltas).toEqual(["Hew"]);
    expect(broken).toHaveBeenCalledTimes(1);

    const controller = new AbortController();
    const abortingRefusal = vi.fn<LlmHttpTransport>(async () => {
      controller.abort(new Error("stop"));
      return httpResponse(400, [refusal]);
    });
    await expect(createLlmClient(openai, { transports: { webview: abortingRefusal } }).chat({ messages: [], stream: true }, { signal: controller.signal })).rejects.toThrow("stop");
    expect(abortingRefusal).toHaveBeenCalledTimes(1);
  });

  it("recognizes a refusal of the stream itself", () => {
    expect(refusesStreaming(httpError(400, refusal))).toBe(true);
    expect(refusesStreaming(httpError(422, JSON.stringify({ detail: "stream: extra fields not permitted" })))).toBe(true);
    expect(refusesStreaming(httpError(400, JSON.stringify({ error: "model 'x' not found" })))).toBe(false);
    expect(refusesStreaming(httpError(401, JSON.stringify({ error: "stream key invalid" })))).toBe(false);
    expect(refusesStreaming(httpError(500, "stream failed"))).toBe(false);
    expect(refusesStreaming(httpError(400, "upstream error"))).toBe(false);
    expect(refusesStreaming(new Error("stream"))).toBe(false);
  });
});

describe("raw completion", () => {
  it("uses llama-server's native /completion for the bundled server", async () => {
    const f = controlledFetch({
      chunks: [sse({ content: "Bon", stop: false }), sse({ content: "jour</s>", stop: false }), sse({ content: "", stop: true, stopped_eos: true, tokens_predicted: 3, tokens_evaluated: 12 })],
    });
    const client = createLlmClient(endpoint({ provider: "bundled", baseUrl: "http://127.0.0.1:5123/v1", apiKey: "k-123456" }), { transports: { webview: f.transport } });
    const deltas: string[] = [];
    const result = await client.complete(
      { prompt: "Translate the following English sentence into French:\nHello <fr>", maxTokens: 64, temperature: 0, stop: ["\n"] },
      { signal: live(), onDelta: (d) => deltas.push(d.text) },
    );
    expect(f.calls[0].url).toBe("http://127.0.0.1:5123/completion");
    expect(f.calls[0].body).toMatchObject({ prompt: expect.stringContaining("<fr>"), n_predict: 64, temperature: 0, stop: ["\n"], stream: true, cache_prompt: true });
    expect(result).toMatchObject({ text: "Bonjour", finishReason: "stop", usage: { promptTokens: 12, completionTokens: 3 } });
    expect(deltas).toEqual(["Bon", "Bonjour"]);
  });

  it("uses /completions on other servers and refuses on Gemini", async () => {
    const f = controlledFetch({ headers: json, chunks: [JSON.stringify({ choices: [{ text: "Hallo", finish_reason: "length" }] })] });
    const result = await createLlmClient(endpoint(), { transports: { webview: f.transport } }).complete({ prompt: "p", maxTokens: 8 }, { signal: live() });
    expect(f.calls[0].url).toBe("http://127.0.0.1:8080/v1/completions");
    expect(f.calls[0].body).toEqual({ model: "m", prompt: "p", max_tokens: 8 });
    expect(result).toMatchObject({ text: "Hallo", finishReason: "length" });

    const gemini = createLlmClient(endpoint({ provider: "gemini" }), { transports: { webview: f.transport } });
    await expect(gemini.complete({ prompt: "p" }, { signal: live() })).rejects.toMatchObject({ kind: "unsupported" });
  });
});

describe("modelListed", () => {
  it("matches exact ids, path suffixes and Ollama's implicit :latest tag", () => {
    expect(modelListed(["llama3:latest", "qwen3:8b"], "llama3")).toBe(true);
    expect(modelListed(["llama3"], "llama3:latest")).toBe(true);
    expect(modelListed(["library/llama3:latest"], "llama3")).toBe(true);
    expect(modelListed(["llama3:8b"], "llama3")).toBe(false); // another tag is another model
    expect(modelListed(["qwen3:8b"], "qwen3:4b")).toBe(false);
    expect(modelListed(["models/gemini-3.5-flash-lite"], "gemini-3.5-flash-lite")).toBe(true);
    expect(modelListed(["google/gemini-2.5-flash-lite"], "google/gemini-2.5-flash-lite")).toBe(true);
    expect(modelListed(["gpt-4o-mini"], " gpt-4o-mini ")).toBe(true);
    expect(modelListed(["gpt-4o-mini"], "gpt-4o")).toBe(false);
    expect(modelListed(["a"], " ")).toBe(false);
  });
});

describe("listModels", () => {
  it("reads OpenAI, Gemini and Ollama model lists", async () => {
    const openai = controlledFetch({ headers: json, chunks: [JSON.stringify({ object: "list", data: [{ id: "a" }, { id: "b" }, { id: "a" }] })] });
    const client = createLlmClient(endpoint({ apiKey: "sk-123456" }), { transports: { webview: openai.transport } });
    await expect(client.listModels(live())).resolves.toEqual(["a", "b"]);
    expect(openai.calls[0].url).toBe("http://127.0.0.1:8080/v1/models");
    expect(openai.calls[0].init.method).toBe("GET");
    expect((openai.calls[0].init.headers as Record<string, string>).Authorization).toBe("Bearer sk-123456");

    const gemini = controlledFetch({ headers: json, chunks: [JSON.stringify({ data: [{ id: "models/gemini-3.5-flash-lite" }] })] });
    await expect(createLlmClient(endpoint({ provider: "gemini" }), { transports: { webview: gemini.transport } }).listModels(live())).resolves.toEqual(["gemini-3.5-flash-lite"]);

    const ollama = controlledFetch({ headers: json, chunks: [JSON.stringify({ models: [{ name: "llama3.2:latest" }] })] });
    await expect(createLlmClient(endpoint(), { transports: { webview: ollama.transport } }).listModels(live())).resolves.toEqual(["llama3.2:latest"]);
  });

  it("rejects with an LlmError when the server is unreachable or refuses the key", async () => {
    const f = controlledFetch({ status: 401, headers: json, chunks: [JSON.stringify({ error: { message: "bad key" } })] });
    await expect(createLlmClient(endpoint(), { transports: { webview: f.transport } }).listModels(live())).rejects.toMatchObject({ kind: "unauthorized" });
  });
});
