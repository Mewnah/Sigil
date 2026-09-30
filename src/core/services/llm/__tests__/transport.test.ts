import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import type { RustLlmProxyEvent } from "@/types";

type Handler = (event: RustLlmProxyEvent) => void;

const tauri = vi.hoisted(() => ({
  channels: [] as { onmessage: (event: never) => void }[],
  failChannel: false,
  invoke: undefined as unknown as ReturnType<typeof import("vitest").vi.fn>,
}));

vi.mock("@tauri-apps/api/core", async () => {
  const { vi: v } = await import("vitest");
  tauri.invoke = v.fn();
  class Channel {
    onmessage: (event: never) => void;
    constructor(onmessage: (event: never) => void) {
      if (tauri.failChannel) throw new Error("window.__TAURI_INTERNALS__ is undefined");
      this.onmessage = onmessage;
      tauri.channels.push(this);
    }
  }
  return { Channel, invoke: (...args: unknown[]) => tauri.invoke(...args), isTauri: () => true };
});

import { createLlmClient } from "../client";
import { createRustTransport, LlmHttpRequest, LlmTransportError } from "../transport";
import { LlmEndpoint } from "../types";

const request: LlmHttpRequest = {
  url: "http://192.168.1.20:11434/v1/chat/completions",
  method: "POST",
  headers: { Authorization: "Bearer k" },
  body: '{"model":"m"}',
  stream: true,
  timeoutMs: 15_000,
};

/** The pending `llm_proxy_chat` call: its channel and a way to settle the command. */
interface ProxyCall {
  args: { request: Record<string, unknown>; onEvent: { onmessage: Handler } };
  emit: Handler;
  resolve(): void;
  reject(e: unknown): void;
}
let proxyCalls: ProxyCall[];
let cancels: string[];

beforeEach(() => {
  tauri.channels = [];
  tauri.failChannel = false;
  proxyCalls = [];
  cancels = [];
  tauri.invoke.mockImplementation((cmd: string, args: Record<string, unknown>) => {
    if (cmd === "plugin:llm-server|llm_proxy_cancel") {
      cancels.push(String((args as { requestId: string }).requestId));
      return Promise.resolve();
    }
    if (cmd === "plugin:llm-server|llm_proxy_chat") {
      return new Promise<void>((resolve, reject) => {
        const a = args as ProxyCall["args"];
        proxyCalls.push({ args: a, emit: (e) => a.onEvent.onmessage(e), resolve, reject });
      });
    }
    return Promise.reject(new Error(`unexpected ${cmd}`));
  });
});

afterEach(() => {
  vi.useRealTimers();
});

const tick = () => new Promise<void>((r) => setTimeout(r, 0));

async function collect(body: AsyncIterable<string>): Promise<string[]> {
  const out: string[] = [];
  for await (const c of body) out.push(c);
  return out;
}

describe("Rust proxy transport", () => {
  it("sends the frozen wire request and streams response, chunks and done in order", async () => {
    const pending = createRustTransport()(request, new AbortController().signal);
    await tick();
    const call = proxyCalls[0];
    expect(call.args.request).toMatchObject({
      url: request.url,
      method: "POST",
      headers: { Authorization: "Bearer k" },
      body: '{"model":"m"}',
      stream: true,
      timeoutMs: 15_000,
    });
    expect(String(call.args.request.requestId)).toMatch(/^llm-/);

    call.emit({ kind: "response", status: 200, headers: { "Content-Type": "text/event-stream" } });
    const response = await pending;
    expect(response.status).toBe(200);
    expect(response.headers).toEqual({ "content-type": "text/event-stream" });
    call.emit({ kind: "chunk", data: "data: a\n\n" });
    call.emit({ kind: "chunk", data: "data: b\n\n" });
    call.emit({ kind: "done" });
    call.resolve();
    await expect(collect(response.body)).resolves.toEqual(["data: a\n\n", "data: b\n\n"]);
    expect(cancels).toEqual([]);
  });

  it("still delivers events that arrive after the command resolved, and ends a body with no done event after a grace period", async () => {
    vi.useFakeTimers();
    const pending = createRustTransport()(request, new AbortController().signal);
    await vi.advanceTimersByTimeAsync(0);
    const call = proxyCalls[0];
    call.resolve();
    await vi.advanceTimersByTimeAsync(10);
    call.emit({ kind: "response", status: 200, headers: {} });
    const response = await pending;
    call.emit({ kind: "chunk", data: "late" });
    const body = collect(response.body);
    await vi.advanceTimersByTimeAsync(3_000);
    await expect(body).resolves.toEqual(["late"]);
  });

  it("rejects before a response with a transport error the client may retry elsewhere", async () => {
    const pending = createRustTransport()(request, new AbortController().signal);
    await tick();
    proxyCalls[0].emit({ kind: "error", message: "error sending request: connection refused" });
    const err = await pending.catch((e) => e);
    expect(err).toBeInstanceOf(LlmTransportError);
    expect(err.beforeResponse).toBe(true);
    expect(err.message).toContain("connection refused");
  });

  it("reports a proxy that is not built yet as unavailable", async () => {
    const pending = createRustTransport()(request, new AbortController().signal);
    await tick();
    proxyCalls[0].reject({ kind: "not_implemented", message: "llm-server: not implemented yet", missing: [] });
    const err = await pending.catch((e) => e);
    expect(err).toBeInstanceOf(LlmTransportError);
    expect(err.unavailable).toBe(true);
  });

  it("reports no Tauri IPC (web mode) as unavailable", async () => {
    tauri.failChannel = true;
    const err = await createRustTransport()(request, new AbortController().signal).catch((e) => e);
    expect(err).toBeInstanceOf(LlmTransportError);
    expect(err.unavailable).toBe(true);
  });

  it("fails the body when the stream breaks off after the response", async () => {
    const pending = createRustTransport()(request, new AbortController().signal);
    await tick();
    const call = proxyCalls[0];
    call.emit({ kind: "response", status: 200, headers: {} });
    const response = await pending;
    call.emit({ kind: "chunk", data: "x" });
    call.emit({ kind: "error", message: "stalled for 30 s" });
    const err = await collect(response.body).catch((e) => e);
    expect(err).toBeInstanceOf(LlmTransportError);
    expect(err.beforeResponse).toBe(false);
  });

  it("cancels the proxied request on abort, before and after the response", async () => {
    const before = new AbortController();
    const p1 = createRustTransport()(request, before.signal);
    await tick();
    const reason = new Error("stop");
    before.abort(reason);
    await expect(p1).rejects.toBe(reason);
    expect(cancels).toEqual([String(proxyCalls[0].args.request.requestId)]);

    const after = new AbortController();
    const p2 = createRustTransport()(request, after.signal);
    await tick();
    proxyCalls[1].emit({ kind: "response", status: 200, headers: {} });
    const response = await p2;
    const body = collect(response.body).catch((e) => e);
    after.abort(reason);
    await expect(body).resolves.toBe(reason);
    expect(cancels).toEqual([String(proxyCalls[0].args.request.requestId), String(proxyCalls[1].args.request.requestId)]);
    // The proxy's own "cancelled" error after the abort changes nothing.
    proxyCalls[1].emit({ kind: "error", message: "cancelled" });
  });

  it("cancels when the reader stops early", async () => {
    const pending = createRustTransport()(request, new AbortController().signal);
    await tick();
    proxyCalls[0].emit({ kind: "response", status: 200, headers: {} });
    const response = await pending;
    response.cancel();
    response.cancel();
    expect(cancels).toHaveLength(1);
  });
});

describe("LLM client over the Rust proxy", () => {
  const lan: LlmEndpoint = {
    provider: "custom",
    baseUrl: "http://192.168.1.20:11434/v1",
    apiKey: "",
    keyHeader: "bearer",
    model: "qwen3:4b",
    headers: {},
    transport: "rust",
  };

  it("streams a chat reply", async () => {
    const deltas: string[] = [];
    const pending = createLlmClient(lan).chat({ messages: [{ role: "user", content: "hi" }] }, { signal: new AbortController().signal, onDelta: (d) => deltas.push(d.text) });
    await tick();
    const call = proxyCalls[0];
    expect(JSON.parse(String(call.args.request.body))).toMatchObject({ model: "qwen3:4b", stream: true });
    call.emit({ kind: "response", status: 200, headers: { "content-type": "text/event-stream" } });
    call.emit({ kind: "chunk", data: 'data: {"choices":[{"delta":{"content":"Hel"}}]}\n\ndata: {"choices":[{"delta":{"con' });
    call.emit({ kind: "chunk", data: 'tent":"lo"},"finish_reason":"stop"}]}\n\ndata: [DONE]\n\n' });
    call.emit({ kind: "done" });
    call.resolve();
    await expect(pending).resolves.toMatchObject({ text: "Hello", finishReason: "stop" });
    expect(deltas).toEqual(["Hel", "Hello"]);
  });

  it("aborts the proxied request when the caller aborts mid-stream", async () => {
    const controller = new AbortController();
    const pending = createLlmClient(lan).chat({ messages: [] }, { signal: controller.signal, onDelta: () => {} });
    await tick();
    const call = proxyCalls[0];
    call.emit({ kind: "response", status: 200, headers: { "content-type": "text/event-stream" } });
    call.emit({ kind: "chunk", data: 'data: {"choices":[{"delta":{"content":"Hel"}}]}\n\n' });
    await tick();
    const reason = new Error("stopped");
    controller.abort(reason);
    await expect(pending).rejects.toBe(reason);
    expect(cancels).toEqual([String(call.args.request.requestId)]);
  });
});
