/**
 * Acceptance (P4-A): "Transform streams and aborts correctly." Drives the real Service_Transform with the real
 * Transform_LLMService and shared LLM client, over the real webview transport with a fetch whose body the test feeds.
 */
import i18n from "i18next";
import { proxy } from "valtio";
import { afterEach, beforeAll, beforeEach, describe, expect, it, vi } from "vitest";
import { ServiceNetworkState, TextEventSource, TextEventType } from "@/types";

vi.mock("@tauri-apps/api/core", () => ({ invoke: vi.fn(), isTauri: () => false, Channel: class {} }));
vi.mock("react-toastify", () => ({
  toast: Object.assign(vi.fn(), { error: vi.fn(), info: vi.fn(), warning: vi.fn(), success: vi.fn() }),
}));
vi.mock("../../systemLog", () => ({ pushSystemLog: vi.fn() }));
vi.mock("@/utils/devLog", () => ({ devLog: () => {} }));
// Line ids start at a random per-process base in the app; pin it so the expected ids stay readable.
vi.mock("@/utils/processNonce", async (importOriginal) => ({ ...(await importOriginal<typeof import("@/utils/processNonce")>()), PROCESS_ID_BASE: 0 }));

import { toast } from "react-toastify";
import { resetLlmTransportMemory } from "@/core/services/llm/client";
import { createWebviewTransport } from "@/core/services/llm/transport";
import Service_Transform, { TRANSFORM_PARTIAL_INTERVAL_MS, TransformEngineMap } from "../index";
import { Service_Transform_Schema, Transform_Backends, Transform_State } from "../schema";
import { Transform_LLMService } from "../services/llm";
import { PartialThrottle } from "../streaming";

const enc = new TextEncoder();

interface FetchCall {
  body: Record<string, unknown>;
  signal: AbortSignal;
  push(text: string): void;
  close(): void;
  cancelled: boolean;
}

let calls: FetchCall[];
/** Answer streamed requests like OpenAI does for some models of unverified organizations (HTTP 400). */
let refuseStreams = false;
const STREAM_REFUSAL = JSON.stringify({
  error: { message: "Your organization must be verified to stream this model.", type: "invalid_request_error", param: "stream", code: "unsupported_value" },
});

const fetchMock = async (_url: string, init: RequestInit) => {
  const signal = init.signal as AbortSignal;
  if (signal.aborted) throw signal.reason;
  if (refuseStreams && JSON.parse(String(init.body)).stream === true) {
    return new Response(STREAM_REFUSAL, { status: 400, headers: { "content-type": "application/json" } });
  }
  let ctl!: ReadableStreamDefaultController<Uint8Array>;
  const call: FetchCall = {
    body: JSON.parse(String(init.body)),
    signal,
    push: (t) => ctl.enqueue(enc.encode(t)),
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
  signal.addEventListener("abort", () => {
    call.cancelled = true;
    try {
      ctl.error(signal.reason);
    } catch {
      /* closed */
    }
  });
  calls.push(call);
  const streaming = call.body.stream === true;
  return new Response(stream, { status: 200, headers: { "content-type": streaming ? "text/event-stream" : "application/json" } });
};

class TestLlm extends Transform_LLMService {
  constructor() {
    super({ transports: { webview: createWebviewTransport(fetchMock as unknown as typeof fetch) } }, { inTauri: false });
  }
}

const engines = { [Transform_Backends.openai]: TestLlm, [Transform_Backends.dummy]: TestLlm } as unknown as TransformEngineMap;

type Handler = (e?: { type: TextEventType; value: string; id?: number }) => void;
type Published = { topic: string; type: TextEventType; value: string; id?: number };
let sttHandler: Handler;
let published: Published[];

function makeService(patch: (s: Transform_State) => void = () => {}) {
  const data = proxy(Service_Transform_Schema.parse({}));
  data.autoStart = false;
  data.provider = "openrouter";
  data.openRouterKey = "sk-or-test-key";
  patch(data);
  published = [];
  const transformState = { data };
  vi.stubGlobal("window", {
    ApiServer: {
      state: { services: { transform: transformState } },
      patchService: vi.fn((_key: string, fn: (s: typeof transformState) => void) => fn(transformState)),
    },
    ApiShared: {
      pubsub: {
        subscribe: vi.fn(() => "sub"),
        unsubscribe: vi.fn(),
        subscribeText: vi.fn((_source: string, fn: Handler) => {
          sttHandler = fn;
          return "sub-stt";
        }),
        publishText: vi.fn((topic: string, e: Published) => published.push({ ...e, topic })),
      },
    },
  });
  const service = new Service_Transform(engines);
  service.init();
  return { service, data };
}

const sse = (content: string) => `data: ${JSON.stringify({ choices: [{ delta: { content } }] })}\n\n`;
const on = (topic: TextEventSource) =>
  published
    .filter((p) => p.topic === topic)
    .map((p) => `${p.type === TextEventType.final ? "final" : "interim"}:${p.value}${p.id === undefined ? "" : `#${p.id}`}`);
const stt = (value: string) => sttHandler({ type: TextEventType.final, value });

beforeAll(async () => {
  await i18n.init({ lng: "en", resources: { en: { translation: {} } } });
});

beforeEach(() => {
  calls = [];
  refuseStreams = false;
  resetLlmTransportMemory();
  vi.mocked(toast).mockClear();
  vi.spyOn(console, "warn").mockImplementation(() => {});
});

afterEach(() => {
  vi.useRealTimers();
  vi.unstubAllGlobals();
});

async function startedService(patch?: (s: Transform_State) => void) {
  const s = makeService(patch);
  s.service.start();
  await vi.advanceTimersByTimeAsync(0);
  expect(s.service.serviceState.status).toBe(ServiceNetworkState.connected);
  return s;
}

describe("Service_Transform streaming (acceptance)", () => {
  it("shows the reply growing on the synced captions, then delivers one final everywhere", async () => {
    vi.useFakeTimers();
    await startedService();
    stt("hello there");
    await vi.advanceTimersByTimeAsync(0);
    expect(calls).toHaveLength(1);
    const call = calls[0];
    expect(call.body.stream).toBe(true);

    call.push(sse("<think>pun</think>"));
    await vi.advanceTimersByTimeAsync(0);
    call.push(sse("Hewwo"));
    await vi.advanceTimersByTimeAsync(0);
    call.push(sse(" thewe"));
    await vi.advanceTimersByTimeAsync(0);
    // Throttled: the first partial went out at once, the next waits for the interval.
    expect(on(TextEventSource.transform_source)).toEqual(["interim:hello there#0", "interim:Hewwo#0"]);
    await vi.advanceTimersByTimeAsync(TRANSFORM_PARTIAL_INTERVAL_MS);
    expect(on(TextEventSource.transform_source).at(-1)).toBe("interim:Hewwo thewe#0");
    expect(on(TextEventSource.transform)).toEqual([]); // finals only

    call.push(sse(" owo") + "data: [DONE]\n\n");
    await vi.advanceTimersByTimeAsync(0);
    expect(on(TextEventSource.transform)).toEqual(["final:Hewwo thewe owo"]);
    expect(on(TextEventSource.transform_source).at(-1)).toBe("final:Hewwo thewe owo#0");
    expect(on(TextEventSource.transform_raw)).toEqual(["interim:hello there ...#0", "final:hello there#0"]);
    // Nothing trails after the final.
    const count = published.length;
    await vi.advanceTimersByTimeAsync(1_000);
    expect(published).toHaveLength(count);
    expect(published.map((p) => p.value).join(" ")).not.toContain("pun");
  });

  it("stop mid-stream aborts the HTTP request and closes the captions with the original, silently", async () => {
    vi.useFakeTimers();
    const { service } = await startedService();
    stt("pending line");
    await vi.advanceTimersByTimeAsync(0);
    const call = calls[0];
    call.push(sse("Pend"));
    await vi.advanceTimersByTimeAsync(0);

    service.stop();
    await vi.advanceTimersByTimeAsync(0);
    expect(call.signal.aborted).toBe(true);
    expect(call.cancelled).toBe(true);
    expect(on(TextEventSource.transform)).toEqual([]);
    expect(on(TextEventSource.transform_source).at(-1)).toBe("final:pending line#0");
    expect(on(TextEventSource.transform_raw).at(-1)).toBe("final:pending line#0");
    expect(vi.mocked(toast)).not.toHaveBeenCalled();
    expect(service.pendingLines).toBe(0);
    const count = published.length;
    await vi.advanceTimersByTimeAsync(1_000);
    expect(published).toHaveLength(count);
  });

  it("aborts the request at the configured timeout and sends the original text", async () => {
    vi.useFakeTimers();
    await startedService((s) => (s.timeoutMs = 4_000));
    stt("slow line");
    await vi.advanceTimersByTimeAsync(0);
    const call = calls[0];
    call.push(sse("Sl"));
    await vi.advanceTimersByTimeAsync(3_999);
    expect(call.signal.aborted).toBe(false);
    await vi.advanceTimersByTimeAsync(1);
    expect(call.signal.aborted).toBe(true);
    expect(call.cancelled).toBe(true);
    expect(on(TextEventSource.transform)).toEqual(["final:slow line"]);
    expect(on(TextEventSource.transform_source).at(-1)).toBe("final:slow line#0");
    expect(vi.mocked(toast)).toHaveBeenCalledWith("transform.error_timeout", expect.anything());
  });

  it("keeps sentence order when a later line's stream finishes first", async () => {
    vi.useFakeTimers();
    await startedService();
    stt("one");
    stt("two");
    await vi.advanceTimersByTimeAsync(0);
    const [first, second] = calls;
    second.push(sse("TWO") + "data: [DONE]\n\n");
    await vi.advanceTimersByTimeAsync(0);
    expect(on(TextEventSource.transform)).toEqual([]);
    expect(on(TextEventSource.transform_source)).toContain("interim:TWO#1");
    // Only the newest line streams: the older one's reply shows up as its final.
    first.push(sse("ON"));
    await vi.advanceTimersByTimeAsync(TRANSFORM_PARTIAL_INTERVAL_MS);
    expect(on(TextEventSource.transform_source)).not.toContain("interim:ON#0");
    first.push(sse("E") + "data: [DONE]\n\n");
    await vi.advanceTimersByTimeAsync(0);
    expect(on(TextEventSource.transform)).toEqual(["final:ONE", "final:TWO"]);
  });

  it("stops a line's partials once the next sentence starts, so captions that ignore ids never flip between lines", async () => {
    vi.useFakeTimers();
    await startedService();
    stt("hello there");
    await vi.advanceTimersByTimeAsync(0);
    const first = calls[0];
    first.push(sse("Hewwo"));
    await vi.advanceTimersByTimeAsync(0);
    first.push(sse(" th")); // throttled: waits for the interval
    await vi.advanceTimersByTimeAsync(0);
    // The speaker goes on: STT interims of the next sentence share text.transform_source.
    sttHandler({ type: TextEventType.interim, value: "and then" });
    await vi.advanceTimersByTimeAsync(TRANSFORM_PARTIAL_INTERVAL_MS);
    first.push(sse("ewe"));
    await vi.advanceTimersByTimeAsync(TRANSFORM_PARTIAL_INTERVAL_MS);
    sttHandler({ type: TextEventType.interim, value: "and then we" });
    await vi.advanceTimersByTimeAsync(TRANSFORM_PARTIAL_INTERVAL_MS);
    expect(on(TextEventSource.transform_source)).toEqual(["interim:hello there#0", "interim:Hewwo#0", "interim:and then#1", "interim:and then we#1"]);

    // The reply still arrives, whole, as the line's final.
    first.push(sse(" owo") + "data: [DONE]\n\n");
    await vi.advanceTimersByTimeAsync(0);
    expect(on(TextEventSource.transform_source).at(-1)).toBe("final:Hewwo thewe owo#0");
    expect(on(TextEventSource.transform)).toEqual(["final:Hewwo thewe owo"]);

    // The next line is the newest one now, and streams until the sentence after it starts.
    stt("and then we left");
    await vi.advanceTimersByTimeAsync(0);
    calls[1].push(sse("And"));
    await vi.advanceTimersByTimeAsync(0);
    expect(on(TextEventSource.transform_source).at(-1)).toBe("interim:And#1");
    const count = published.length;
    sttHandler({ type: TextEventType.interim, value: "so" });
    calls[1].push(sse(" then"));
    await vi.advanceTimersByTimeAsync(TRANSFORM_PARTIAL_INTERVAL_MS);
    expect(published.slice(count).map((p) => p.value)).toEqual(["so"]);
  });

  it("retries without streaming when the provider refuses to stream, and keeps sending plain requests", async () => {
    vi.useFakeTimers();
    refuseStreams = true;
    await startedService((s) => {
      s.provider = "openai";
      s.openaiKey = "sk-test-123456";
      s.openaiModel = "o3";
    });
    stt("hello there");
    await vi.advanceTimersByTimeAsync(0);
    expect(calls).toHaveLength(1); // the refused streamed request is answered before a body stream exists
    expect(calls[0].body).not.toHaveProperty("stream");
    calls[0].push(JSON.stringify({ choices: [{ message: { content: "Hewwo thewe" }, finish_reason: "stop" }] }));
    calls[0].close();
    await vi.advanceTimersByTimeAsync(0);
    expect(on(TextEventSource.transform)).toEqual(["final:Hewwo thewe"]);
    expect(on(TextEventSource.transform_source).at(-1)).toBe("final:Hewwo thewe#0");
    expect(vi.mocked(toast)).not.toHaveBeenCalled();

    stt("again");
    await vi.advanceTimersByTimeAsync(0);
    expect(calls).toHaveLength(2);
    expect(calls[1].body).not.toHaveProperty("stream");
  });

  it("pairs partials with their line by id only, keeping the segment revision for the final", async () => {
    vi.useFakeTimers();
    await startedService();
    sttHandler({ type: TextEventType.final, value: "segmented", segmentId: "mic:1:7", revision: 4, lang: "en" } as never);
    await vi.advanceTimersByTimeAsync(0);
    calls[0].push(sse("SEG"));
    await vi.advanceTimersByTimeAsync(0);
    calls[0].push(sse("MENTED") + "data: [DONE]\n\n");
    await vi.advanceTimersByTimeAsync(0);
    const source = published.filter((p) => p.topic === TextEventSource.transform_source) as (Published & { segmentId?: string; revision?: number })[];
    const partial = source.find((p) => p.value === "SEG")!;
    expect(partial).toMatchObject({ type: TextEventType.interim, id: 0 });
    expect(partial.segmentId).toBeUndefined();
    expect(partial.revision).toBeUndefined();
    const final = source.at(-1)!;
    expect(final).toMatchObject({ type: TextEventType.final, value: "SEGMENTED", id: 0, segmentId: "mic:1:7", revision: 4 });
  });

  it("sends a plain request and publishes no partials with streaming off", async () => {
    vi.useFakeTimers();
    await startedService((s) => (s.stream = false));
    stt("plain");
    await vi.advanceTimersByTimeAsync(0);
    const call = calls[0];
    expect(call.body).not.toHaveProperty("stream");
    call.push(JSON.stringify({ choices: [{ message: { content: "PLAIN" }, finish_reason: "stop" }] }));
    call.close();
    await vi.advanceTimersByTimeAsync(0);
    expect(on(TextEventSource.transform_source)).toEqual(["interim:plain#0", "final:PLAIN#0"]);
    expect(on(TextEventSource.transform)).toEqual(["final:PLAIN"]);
  });

  it("falls back to the original text when the stream breaks with a provider error", async () => {
    vi.useFakeTimers();
    await startedService();
    stt("broken");
    await vi.advanceTimersByTimeAsync(0);
    calls[0].push(sse("Bro") + `data: ${JSON.stringify({ error: { message: "Provider returned error", code: 429 } })}\n\n`);
    await vi.advanceTimersByTimeAsync(0);
    expect(on(TextEventSource.transform)).toEqual(["final:broken"]);
    expect(on(TextEventSource.transform_source).at(-1)).toBe("final:broken#0");
    expect(vi.mocked(toast)).toHaveBeenCalledWith("transform.error_rate_limited", expect.anything());
  });
});

describe("PartialThrottle", () => {
  it("publishes the first partial at once, then at most one per interval, always ending on the latest", () => {
    vi.useFakeTimers();
    const out: string[] = [];
    const t = new PartialThrottle((s) => out.push(s), 100, () => Date.now());
    t.push("a");
    t.push("ab");
    t.push("abc");
    expect(out).toEqual(["a"]);
    vi.advanceTimersByTime(99);
    expect(out).toEqual(["a"]);
    vi.advanceTimersByTime(1);
    expect(out).toEqual(["a", "abc"]);
    vi.advanceTimersByTime(500);
    t.push("abcd");
    expect(out).toEqual(["a", "abc", "abcd"]);
  });

  it("skips empty and repeated text, and publishes nothing after close", () => {
    vi.useFakeTimers();
    const out: string[] = [];
    const t = new PartialThrottle((s) => out.push(s), 100, () => Date.now());
    t.push("");
    t.push("x");
    vi.advanceTimersByTime(200);
    t.push("x");
    expect(out).toEqual(["x"]);
    t.push("xy");
    t.push("xyz");
    t.close();
    vi.advanceTimersByTime(1_000);
    t.push("late");
    expect(out).toEqual(["x", "xy"]);
  });

  it("survives a publisher that throws", () => {
    const error = vi.spyOn(console, "error").mockImplementation(() => {});
    const t = new PartialThrottle(() => {
      throw new Error("pubsub down");
    });
    expect(() => t.push("a")).not.toThrow();
    expect(error).toHaveBeenCalled();
    t.close();
  });
});
