import i18n from "i18next";
import { proxy } from "valtio";
import { afterEach, beforeAll, beforeEach, describe, expect, it, vi } from "vitest";
import { ServiceNetworkState, TextEventSource, TextEventType } from "@/types";

vi.mock("@tauri-apps/api/core", () => ({ invoke: vi.fn(), isTauri: () => false }));
vi.mock("react-toastify", () => ({
  toast: Object.assign(vi.fn(), { error: vi.fn(), info: vi.fn(), warning: vi.fn(), success: vi.fn() }),
}));
vi.mock("../../systemLog", () => ({ pushSystemLog: vi.fn() }));
vi.mock("@/utils/devLog", () => ({ devLog: () => {} }));
// Line ids start at a random per-process base in the app; pin it so the expected ids stay readable.
vi.mock("@/utils/processNonce", async (importOriginal) => ({ ...(await importOriginal<typeof import("@/utils/processNonce")>()), PROCESS_ID_BASE: 0 }));

import { toast } from "react-toastify";
import Service_Transform, { TRANSFORM_TIMEOUT_MS, TransformEngineMap } from "../index";
import { STREAM_OFFLINE_CONFIRM_MS } from "../../stt/streamStop";
import { OrderedQueue } from "../ordering";
import { Service_Transform_Schema, Transform_Backends, Transform_State } from "../schema";
import {
  ChatTurn,
  ITransformEngine,
  TransformNotConfiguredError,
  TransformRequest,
  TransformRequestError,
} from "../types";

describe("OrderedQueue", () => {
  it("delivers in reservation order whatever the completion order", () => {
    const out: string[] = [];
    const q = new OrderedQueue<string>((v) => out.push(v));
    const a = q.reserve();
    const b = q.reserve();
    const c = q.reserve();
    q.resolve(c, "c");
    q.resolve(b, "b");
    expect(out).toEqual([]);
    expect(q.pending).toBe(3);
    q.resolve(a, "a");
    expect(out).toEqual(["a", "b", "c"]);
    expect(q.pending).toBe(0);
  });

  it("ignores unknown and repeated resolutions", () => {
    const out: string[] = [];
    const q = new OrderedQueue<string>((v) => out.push(v));
    const a = q.reserve();
    q.resolve(a, "a");
    q.resolve(a, "again");
    q.resolve(5, "unknown");
    q.resolve(-1, "negative");
    expect(out).toEqual(["a"]);
  });

  it("keeps delivering after a delivery throws", () => {
    const out: string[] = [];
    const errors: unknown[] = [];
    const q = new OrderedQueue<string>((v) => {
      if (v === "bad") throw new Error("consumer failed");
      out.push(v);
    }, (e) => errors.push(e));
    const a = q.reserve();
    const b = q.reserve();
    q.resolve(b, "ok");
    q.resolve(a, "bad");
    expect(out).toEqual(["ok"]);
    expect(errors).toHaveLength(1);
  });

  it("handles a delivery that resolves another slot synchronously", () => {
    const out: string[] = [];
    let q!: OrderedQueue<string>;
    let later = -1;
    q = new OrderedQueue<string>((v) => {
      out.push(v);
      if (v === "first") q.resolve(later, "second");
    });
    const first = q.reserve();
    later = q.reserve();
    q.resolve(first, "first");
    expect(out).toEqual(["first", "second"]);
  });
});

// ---------------------------------------------------------------------------------------------
// Service_Transform

type Call = { request: TransformRequest; signal: AbortSignal; resolve: (v: string) => void; reject: (e: unknown) => void };

class FakeEngine implements ITransformEngine {
  static instances: FakeEngine[] = [];
  static startImpl: () => Promise<void> = async () => {};
  calls: Call[] = [];
  stopped = 0;
  constructor() {
    FakeEngine.instances.push(this);
  }
  start() {
    return FakeEngine.startImpl();
  }
  transform(request: TransformRequest, signal: AbortSignal) {
    return new Promise<string>((resolve, reject) => this.calls.push({ request, signal, resolve, reject }));
  }
  stop() {
    this.stopped++;
  }
  call(text: string) {
    const c = this.calls.find((x) => x.request.text === text);
    if (!c) throw new Error(`no call for "${text}"`);
    return c;
  }
}

const engines = Object.fromEntries(Object.values(Transform_Backends).map((b) => [b, FakeEngine])) as unknown as TransformEngineMap;

type Handler = (e?: { type: TextEventType; value: string; id?: number }) => void;
type Published = { topic: string; type: TextEventType; value: string; id?: number };
let sttHandler: Handler;
let streamEnded: () => void;
let streamStarted: () => void;
let published: Published[];

function makeService(patch: (s: Transform_State) => void = () => {}, twitch?: { state: object }) {
  const data = proxy(Service_Transform_Schema.parse({}));
  data.autoStart = false;
  patch(data);
  published = [];
  const transformState = { data };
  vi.stubGlobal("window", {
    ApiServer: {
      state: { services: { transform: transformState } },
      patchService: vi.fn((_key: string, fn: (s: typeof transformState) => void) => fn(transformState)),
      ...(twitch ? { twitch } : {}),
    },
    ApiShared: {
      pubsub: {
        subscribe: vi.fn((topic: string, fn: () => void) => {
          if (topic === "stream.on_ended") streamEnded = fn;
          if (topic === "stream.on_started") streamStarted = fn;
          return `sub-${topic}`;
        }),
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

const flush = () => new Promise<void>((r) => setTimeout(r, 0));
const engine = () => FakeEngine.instances[FakeEngine.instances.length - 1];
const stt = (type: TextEventType, value: string) => sttHandler({ type, value });
const on = (topic: TextEventSource) =>
  published
    .filter((p) => p.topic === topic)
    .map((p) => `${p.type === TextEventType.final ? "final" : "interim"}:${p.value}${p.id === undefined ? "" : `#${p.id}`}`);

async function started(patch?: (s: Transform_State) => void, twitch?: { state: object }) {
  const s = makeService(patch, twitch);
  s.service.start();
  await flush();
  expect(s.service.serviceState.status).toBe(ServiceNetworkState.connected);
  return s;
}

beforeAll(async () => {
  await i18n.init({ lng: "en", resources: { en: { translation: {} } } });
});

beforeEach(() => {
  FakeEngine.instances = [];
  FakeEngine.startImpl = async () => {};
  vi.mocked(toast).mockClear();
  // Failed lines are logged with console.warn by design; keep test output readable.
  vi.spyOn(console, "warn").mockImplementation(() => {});
});

afterEach(() => {
  vi.useRealTimers();
  vi.unstubAllGlobals();
});

describe("Service_Transform ordering", () => {
  it("publishes AI lines in sentence order even when replies arrive out of order", async () => {
    await started();
    stt(TextEventType.final, "one");
    stt(TextEventType.final, "two");
    engine().call("two").resolve("TWO");
    await flush();
    expect(on(TextEventSource.transform)).toEqual([]);
    engine().call("one").resolve("ONE");
    await flush();
    expect(on(TextEventSource.transform)).toEqual(["final:ONE", "final:TWO"]);
    expect(on(TextEventSource.transform_raw)).toEqual([
      "interim:one ...#0",
      "interim:two ...#1",
      "final:one#0",
      "final:two#1",
    ]);
    expect(on(TextEventSource.transform_source).filter((s) => s.startsWith("final"))).toEqual(["final:ONE#0", "final:TWO#1"]);
  });

  it("builds context history in sentence order, not completion order", async () => {
    await started((s) => (s.contextHistory = true));
    stt(TextEventType.final, "a");
    stt(TextEventType.final, "b");
    engine().call("b").resolve("B");
    engine().call("a").resolve("A");
    await flush();
    stt(TextEventType.final, "c");
    const expected: ChatTurn[] = [
      { role: "user", content: "a" },
      { role: "assistant", content: "A" },
      { role: "user", content: "b" },
      { role: "assistant", content: "B" },
    ];
    expect(engine().call("c").request.history).toEqual(expected);
  });

  it("sends no history when context history is off", async () => {
    await started();
    stt(TextEventType.final, "a");
    engine().call("a").resolve("A");
    await flush();
    stt(TextEventType.final, "b");
    expect(engine().call("b").request.history).toEqual([]);
  });
});

describe("Service_Transform failures", () => {
  it("never publishes a '[AI Rate Limited]' literal: the original text goes out and a notice is shown", async () => {
    await started();
    stt(TextEventType.final, "hello there");
    engine().call("hello there").reject(new TransformRequestError("rate_limited", "429 Too Many Requests", 429));
    await flush();
    const everything = published.map((p) => p.value);
    expect(everything.some((v) => v.includes("Rate Limited"))).toBe(false);
    expect(on(TextEventSource.transform)).toEqual(["final:hello there"]);
    expect(on(TextEventSource.transform_raw).at(-1)).toBe("final:hello there#0");
    expect(vi.mocked(toast)).toHaveBeenCalledWith("transform.error_rate_limited", expect.anything());
  });

  it("aborts the request when it times out, then sends the original", async () => {
    vi.useFakeTimers();
    const s = makeService();
    s.service.start();
    await vi.advanceTimersByTimeAsync(0);
    stt(TextEventType.final, "slow");
    const call = engine().call("slow");
    await vi.advanceTimersByTimeAsync(TRANSFORM_TIMEOUT_MS - 1);
    expect(call.signal.aborted).toBe(false);
    await vi.advanceTimersByTimeAsync(1);
    expect(call.signal.aborted).toBe(true);
    expect(on(TextEventSource.transform)).toEqual(["final:slow"]);
    expect(vi.mocked(toast)).toHaveBeenCalledWith("transform.error_timeout", expect.anything());
  });

  it("throttles repeated notices", async () => {
    await started();
    for (const line of ["a", "b", "c"]) {
      stt(TextEventType.final, line);
      engine().call(line).reject(new TransformRequestError("rate_limited", "429", 429));
      await flush();
    }
    expect(vi.mocked(toast)).toHaveBeenCalledTimes(1);
  });

  it("does not add failed lines to the context history", async () => {
    await started((s) => (s.contextHistory = true));
    stt(TextEventType.final, "a");
    engine().call("a").reject(new TransformRequestError("request", "500", 500));
    await flush();
    stt(TextEventType.final, "b");
    expect(engine().call("b").request.history).toEqual([]);
  });
});

describe("Service_Transform with AI off", () => {
  it("finalizes transform_raw (Sync timing captions) and mirrors STT", async () => {
    makeService();
    stt(TextEventType.interim, "hel");
    stt(TextEventType.final, "hello");
    await flush();
    expect(on(TextEventSource.transform_raw)).toEqual(["interim:hel#0", "final:hello#0"]);
    expect(on(TextEventSource.transform_source)).toEqual(["interim:hel#0", "final:hello#0"]);
    expect(on(TextEventSource.transform)).toEqual([]);
  });

  it("stop mid-line aborts the request and closes the synced captions without an AI line", async () => {
    const { service } = await started();
    stt(TextEventType.final, "pending");
    const call = engine().call("pending");
    service.stop();
    expect(call.signal.aborted).toBe(true);
    await flush();
    expect(on(TextEventSource.transform)).toEqual([]);
    expect(on(TextEventSource.transform_raw).at(-1)).toBe("final:pending#0");
    expect(vi.mocked(toast)).not.toHaveBeenCalled();
    expect(service.pendingLines).toBe(0);
  });

  it("keeps order between a line cut off by stop and the lines after it", async () => {
    const { service } = await started();
    stt(TextEventType.final, "first");
    service.stop();
    stt(TextEventType.final, "second");
    await flush();
    expect(on(TextEventSource.transform_raw).filter((s) => s.startsWith("final"))).toEqual(["final:first#0", "final:second#1"]);
  });
});

describe("Service_Transform lifecycle", () => {
  it("does not send the STT cancel marker to the AI", async () => {
    await started();
    stt(TextEventType.interim, "I was");
    stt(TextEventType.final, "[...]");
    await flush();
    expect(engine().calls).toEqual([]);
    expect(on(TextEventSource.transform)).toEqual([]);
    expect(on(TextEventSource.transform_source).at(-1)).toBe("final:[...]#0");
  });

  it("rewrites the retired OpenRouter default that existing installs have stored", () => {
    // The whole state is persisted with its defaults, so the old default survives a schema change.
    const legacy = makeService((s) => (s.openRouterModel = "google/gemini-1.5-flash"));
    expect(legacy.data.openRouterModel).toBe("google/gemini-2.5-flash-lite");
    expect(vi.mocked(window.ApiServer.patchService)).toHaveBeenCalledTimes(1);

    const chosen = makeService((s) => (s.openRouterModel = "anthropic/claude-haiku-4.5"));
    expect(chosen.data.openRouterModel).toBe("anthropic/claude-haiku-4.5");
    expect(vi.mocked(window.ApiServer.patchService)).not.toHaveBeenCalled();
  });

  it("stops when the stream stays offline if stopWithStream is on, and starts again when it comes back", async () => {
    const { service } = await started((s) => (s.stopWithStream = true));
    vi.useFakeTimers();
    streamEnded();
    // One empty reply is not an ended stream: nothing stops before the offline state has lasted.
    await vi.advanceTimersByTimeAsync(STREAM_OFFLINE_CONFIRM_MS - 1);
    expect(service.serviceState.status).toBe(ServiceNetworkState.connected);
    await vi.advanceTimersByTimeAsync(1);
    expect(service.serviceState.status).toBe(ServiceNetworkState.disconnected);
    streamStarted();
    vi.useRealTimers();
    await flush();
    expect(service.serviceState.status).toBe(ServiceNetworkState.connected);
  });

  it("starts again on Twitch's real live state, the only thing that says the stream is back", async () => {
    // The host's Twitch service: a valtio proxy whose liveStatus flips with every poll. Nothing publishes stream.on_started.
    const twitch = { state: proxy({ liveStatus: ServiceNetworkState.connected }) };
    const { service } = await started((s) => (s.stopWithStream = true), twitch);
    vi.useFakeTimers();
    twitch.state.liveStatus = ServiceNetworkState.disconnected;
    streamEnded();
    await vi.advanceTimersByTimeAsync(STREAM_OFFLINE_CONFIRM_MS);
    expect(service.serviceState.status).toBe(ServiceNetworkState.disconnected);
    twitch.state.liveStatus = ServiceNetworkState.connected;
    await vi.advanceTimersByTimeAsync(0);
    vi.useRealTimers();
    await flush();
    expect(service.serviceState.status).toBe(ServiceNetworkState.connected);
    expect(FakeEngine.instances).toHaveLength(2);
  });

  it("keeps running when the stream is back before the offline state is confirmed", async () => {
    const { service } = await started((s) => (s.stopWithStream = true));
    vi.useFakeTimers();
    streamEnded();
    await vi.advanceTimersByTimeAsync(STREAM_OFFLINE_CONFIRM_MS / 2);
    streamStarted();
    await vi.advanceTimersByTimeAsync(STREAM_OFFLINE_CONFIRM_MS * 2);
    expect(service.serviceState.status).toBe(ServiceNetworkState.connected);
    expect(FakeEngine.instances).toHaveLength(1);
  });

  it("does not start again when the option was switched off meanwhile", async () => {
    const { service, data } = await started((s) => (s.stopWithStream = true));
    vi.useFakeTimers();
    streamEnded();
    await vi.advanceTimersByTimeAsync(STREAM_OFFLINE_CONFIRM_MS);
    expect(service.serviceState.status).toBe(ServiceNetworkState.disconnected);
    data.stopWithStream = false;
    streamStarted();
    expect(service.serviceState.status).toBe(ServiceNetworkState.disconnected);
  });

  it("does not start again what the user started and stopped by hand after the stream stopped it", async () => {
    const { service } = await started((s) => (s.stopWithStream = true));
    vi.useFakeTimers();
    streamEnded();
    await vi.advanceTimersByTimeAsync(STREAM_OFFLINE_CONFIRM_MS);
    expect(service.serviceState.status).toBe(ServiceNetworkState.disconnected);
    service.start();
    service.stop();
    streamStarted();
    expect(service.serviceState.status).toBe(ServiceNetworkState.disconnected);
  });

  it("keeps running when the stream ends if stopWithStream is off", async () => {
    const { service } = await started();
    vi.useFakeTimers();
    streamEnded();
    await vi.advanceTimersByTimeAsync(STREAM_OFFLINE_CONFIRM_MS * 2);
    expect(service.serviceState.status).toBe(ServiceNetworkState.connected);
  });

  it("an unconfigured provider stays silent on auto start but explains a manual start", async () => {
    FakeEngine.startImpl = async () => {
      throw new TransformNotConfiguredError("transform.error_key_missing");
    };
    const auto = makeService((s) => (s.autoStart = true));
    await flush();
    expect(auto.service.serviceState.status).toBe(ServiceNetworkState.disconnected);
    expect(vi.mocked(toast)).not.toHaveBeenCalled();

    auto.service.start();
    await flush();
    expect(vi.mocked(toast)).toHaveBeenCalledWith("transform.error_start_failed", expect.objectContaining({ type: "error" }));
  });

  it("a stop during start wins", async () => {
    let finish!: () => void;
    FakeEngine.startImpl = () => new Promise<void>((r) => (finish = r));
    const { service } = makeService();
    service.start();
    service.stop();
    finish();
    await flush();
    expect(service.serviceState.status).toBe(ServiceNetworkState.disconnected);
  });
});
