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

import { toast } from "react-toastify";
import { pushSystemLog } from "../../systemLog";
import Service_Translation, { TranslationEngineMap } from "../index";
import { Service_Translation_Schema, Translation_Backends, Translation_State } from "../schema";
import { CANCEL_MARKER } from "../sequencer";
import { ITranslationEngine, TranslationError, TranslationRequest } from "../types";

type Call = { request: TranslationRequest; signal: AbortSignal; resolve: (v: string) => void; reject: (e: unknown) => void };

class FakeEngine implements ITranslationEngine {
  static instances: FakeEngine[] = [];
  static startImpl: (signal: AbortSignal) => Promise<void> = async () => {};
  calls: Call[] = [];
  stopped = 0;
  constructor() {
    FakeEngine.instances.push(this);
  }
  start(_state: Translation_State, signal: AbortSignal) {
    return FakeEngine.startImpl(signal);
  }
  translate(request: TranslationRequest, signal: AbortSignal) {
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

const engines = Object.fromEntries(Object.values(Translation_Backends).map((b) => [b, FakeEngine])) as unknown as TranslationEngineMap;

type Handler = (e?: { type: TextEventType; value: string }) => void;
let handlers: Record<string, Handler>;
let published: { topic: string; type: TextEventType; value: string }[];

function makeService(patch: (s: Translation_State) => void = () => {}) {
  const data = proxy(Service_Translation_Schema.parse({}));
  data.inputField = true;
  patch(data);
  handlers = {};
  published = [];
  vi.stubGlobal("window", {
    ApiServer: { state: { services: { translation: { data } } } },
    ApiShared: {
      pubsub: {
        subscribe: vi.fn(() => "sub"),
        unsubscribe: vi.fn(),
        subscribeText: vi.fn((source: string, fn: Handler) => {
          handlers[source] = fn;
          return `sub-${source}`;
        }),
        publishText: vi.fn((topic: string, e: { type: TextEventType; value: string }) =>
          published.push({ topic, type: e.type, value: e.value })),
      },
    },
  });
  vi.stubGlobal("fetch", vi.fn(async () => new Response("{}")));
  const service = new Service_Translation(engines);
  service.init();
  return { service, data };
}

const flush = () => new Promise<void>((r) => setTimeout(r, 0));
const stt = (type: TextEventType, value: string) => handlers[TextEventSource.stt]({ type, value });
const typed = (type: TextEventType, value: string) => handlers[TextEventSource.textfield]({ type, value });
const engine = () => FakeEngine.instances[FakeEngine.instances.length - 1];
const out = () => published.map((p) => `${p.type === TextEventType.final ? "final" : "interim"}:${p.value}`);

async function started(patch?: (s: Translation_State) => void) {
  const s = makeService(patch);
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
  vi.mocked(pushSystemLog).mockClear();
});

afterEach(() => {
  vi.unstubAllGlobals();
});

describe("Service_Translation", () => {
  it("publishes a final while an interim from the next utterance is in flight (acceptance)", async () => {
    await started();
    stt(TextEventType.interim, "good");
    stt(TextEventType.final, "good morning");
    stt(TextEventType.interim, "how");
    engine().call("good").resolve("buen");
    await flush();
    expect(engine().call("good").signal.aborted).toBe(true); // superseded by the final
    engine().call("good morning").resolve("buenos días");
    await flush();
    expect(out()).toEqual(["final:buenos días"]);
    expect(published.every((p) => p.topic === TextEventSource.translation)).toBe(true);
    engine().call("how").resolve("cómo");
    await flush();
    expect(out()).toEqual(["final:buenos días", "interim:cómo"]);
  });

  it("does not translate the STT cancel marker", async () => {
    await started();
    stt(TextEventType.interim, "I was");
    engine().call("I was").resolve("Yo estaba");
    await flush();
    stt(TextEventType.final, CANCEL_MARKER);
    expect(engine().calls.map((c) => c.request.text)).toEqual(["I was"]);
    expect(out()).toEqual(["interim:Yo estaba", `final:${CANCEL_MARKER}`]);
  });

  it("skips interims when the backend's interim option is off", async () => {
    await started((s) => (s.azure.interim = false));
    stt(TextEventType.interim, "part");
    stt(TextEventType.final, "part whole");
    expect(engine().calls.map((c) => [c.request.kind, c.request.text])).toEqual([["final", "part whole"]]);
  });

  it("sequences typed text separately from speech", async () => {
    await started();
    stt(TextEventType.final, "spoken");
    typed(TextEventType.final, "typed");
    engine().call("typed").resolve("TYPED");
    await flush();
    expect(out()).toEqual(["final:TYPED"]);
  });

  it("retries a final once after a transient failure", async () => {
    vi.useFakeTimers();
    try {
      await (async () => {
        const s = makeService();
        s.service.start();
        await vi.advanceTimersByTimeAsync(0);
        stt(TextEventType.final, "hello");
        engine().call("hello").reject(new TranslationError("HTTP 503", { status: 503 }));
        await vi.advanceTimersByTimeAsync(600);
        expect(engine().calls).toHaveLength(2);
        engine().calls[1].resolve("hola");
        await vi.advanceTimersByTimeAsync(0);
        expect(out()).toEqual(["final:hola"]);
      })();
    } finally {
      vi.useRealTimers();
    }
  });

  it("reports a failed final with a throttled notice and keeps running", async () => {
    const { service } = await started();
    stt(TextEventType.final, "one");
    engine().call("one").reject(new TranslationError("bad key", { status: 401, retryable: false }));
    await flush();
    stt(TextEventType.final, "two");
    engine().call("two").reject(new TranslationError("bad key", { status: 401, retryable: false }));
    await flush();
    expect(service.serviceState.status).toBe(ServiceNetworkState.connected);
    expect(vi.mocked(toast)).toHaveBeenCalledTimes(1);
    expect(vi.mocked(pushSystemLog)).toHaveBeenCalledTimes(2); // every lost final is logged
    expect(out()).toEqual([]);
  });

  it("ignores text until connected", async () => {
    let finishStart!: () => void;
    FakeEngine.startImpl = () => new Promise<void>((r) => (finishStart = r));
    const { service } = makeService();
    service.start();
    stt(TextEventType.final, "too early");
    expect(engine().calls).toEqual([]);
    finishStart();
    await flush();
    expect(service.serviceState.status).toBe(ServiceNetworkState.connected);
  });

  it("stays stopped when stop() lands while start is still probing (start/stop race)", async () => {
    let finishStart!: () => void;
    FakeEngine.startImpl = () => new Promise<void>((r) => (finishStart = r));
    const { service } = makeService();
    service.start();
    expect(service.serviceState.status).toBe(ServiceNetworkState.connecting);
    service.stop();
    finishStart();
    await flush();
    expect(service.serviceState.status).toBe(ServiceNetworkState.disconnected);
    expect(engine().stopped).toBe(1);
  });

  it("aborts the start signal on stop", async () => {
    let seen!: AbortSignal;
    FakeEngine.startImpl = (signal) => {
      seen = signal;
      return new Promise<void>(() => {});
    };
    const { service } = makeService();
    service.start();
    service.stop();
    expect(seen.aborted).toBe(true);
  });

  it("a restart supersedes an older start", async () => {
    const finishers: (() => void)[] = [];
    FakeEngine.startImpl = () => new Promise<void>((r) => finishers.push(r));
    const { service } = makeService();
    service.start();
    service.start();
    finishers[1]();
    await flush();
    expect(service.serviceState.status).toBe(ServiceNetworkState.connected);
    finishers[0](); // stale: must not touch the state
    await flush();
    expect(service.serviceState.status).toBe(ServiceNetworkState.connected);
    expect(FakeEngine.instances[0].stopped).toBe(1);
  });

  it("surfaces a start failure", async () => {
    FakeEngine.startImpl = async () => {
      throw new Error("server down");
    };
    const { service } = makeService();
    service.start();
    await flush();
    expect(service.serviceState.status).toBe(ServiceNetworkState.disconnected);
    expect(service.serviceState.error).toBe("server down");
    expect(vi.mocked(toast)).toHaveBeenCalledWith("server down", expect.objectContaining({ type: "error" }));
  });

  it("explains a start failure whose error has no message instead of failing silently", async () => {
    // What an engine throws when i18n.t() ran before i18next was ready: new Error(undefined).
    FakeEngine.startImpl = async () => {
      throw new Error(undefined);
    };
    const { service } = makeService();
    service.start();
    await flush();
    expect(service.serviceState.status).toBe(ServiceNetworkState.disconnected);
    expect(service.serviceState.error).toBe("transl.error_start_no_details");
    expect(vi.mocked(toast)).toHaveBeenCalledWith("transl.error_start_no_details", expect.objectContaining({ type: "error" }));
    expect(vi.mocked(pushSystemLog)).toHaveBeenCalledWith("Translation", "transl.error_start_no_details", "error");
  });

  it("drops in-flight lines on stop and publishes nothing late", async () => {
    const { service } = await started();
    stt(TextEventType.final, "pending");
    const call = engine().call("pending");
    service.stop();
    expect(call.signal.aborted).toBe(true);
    call.resolve("late");
    await flush();
    expect(out()).toEqual([]);
    expect(vi.mocked(toast)).not.toHaveBeenCalled();
  });

  it("auto-starts from settings", async () => {
    const { service } = makeService((s) => (s.autoStart = true));
    await flush();
    expect(service.serviceState.status).toBe(ServiceNetworkState.connected);
  });
});
