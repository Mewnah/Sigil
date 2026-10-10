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
vi.mock("@tauri-apps/plugin-dialog", () => ({ ask: vi.fn(async () => false) }));

import { ask } from "@tauri-apps/plugin-dialog";
import { toast } from "react-toastify";
import { pushSystemLog } from "../../systemLog";
import Service_TTS, { TTSBackendMap } from "../index";
import { Service_TTS_Schema, TTS_Backends, TTS_State } from "../schema";
import { ITTSReceiver, ITTSService, TTS_OPTIONS_MISSING } from "../types";

type TextHandler = (e?: { type: TextEventType; value: string }) => void;

class FakeBackend implements ITTSService {
  static instances: FakeBackend[] = [];
  static startImpl: (self: FakeBackend) => void | Promise<void> = () => {};
  started = 0;
  stopped = 0;
  disposed = 0;
  played: string[] = [];
  constructor(public receiver: ITTSReceiver) {
    FakeBackend.instances.push(this);
  }
  start(_state: TTS_State) {
    this.started++;
    return FakeBackend.startImpl(this);
  }
  stop() {
    this.stopped++;
    this.receiver.onStop();
  }
  play(value: string) {
    this.played.push(value);
  }
  dispose() {
    this.disposed++;
  }
}

const fakeMap = Object.fromEntries(Object.values(TTS_Backends).map((b) => [b, FakeBackend])) as unknown as TTSBackendMap;

let subscriptions: { source: string; fn: TextHandler }[];

function installWindow(data: TTS_State) {
  subscriptions = [];
  vi.stubGlobal("window", {
    ApiServer: {
      state: { services: { tts: { data } }, audioOutputDevice: "" },
      sound: { enqueueVoiceClip: vi.fn(async () => {}), clearVoiceClips: vi.fn() },
      voiceChanger: { state: { isRunning: false } },
      patchService: (_key: "tts", fn: (s: { data: TTS_State }) => void) => fn({ data }),
    },
    ApiShared: {
      pubsub: {
        subscribe: vi.fn(() => "stream-sub"),
        unsubscribe: vi.fn(),
        subscribeText: vi.fn((source: string, fn: TextHandler) => {
          subscriptions.push({ source, fn });
          return `sub-${subscriptions.length}`;
        }),
      },
    },
  });
}

function makeService(overrides: Partial<TTS_State> = {}) {
  const data = proxy({ ...Service_TTS_Schema.parse({}), ...overrides });
  installWindow(data);
  const service = new Service_TTS(fakeMap);
  service.updateReplacementsCache();
  return { service, data };
}

const last = () => FakeBackend.instances[FakeBackend.instances.length - 1];

beforeAll(async () => {
  await i18n.init({ lng: "en", resources: { en: { translation: {} } } });
});

beforeEach(() => {
  FakeBackend.instances = [];
  FakeBackend.startImpl = () => {};
  vi.mocked(toast).mockClear();
  vi.mocked(pushSystemLog).mockClear();
});

afterEach(() => {
  vi.unstubAllGlobals();
});

describe("Service_TTS lifecycle", () => {
  it("start -> connecting -> connected; stop -> disconnected, backend stopped and disposed once", () => {
    const { service } = makeService();
    service.start();
    expect(service.serviceState.status).toBe(ServiceNetworkState.connecting);
    last().receiver.onStart();
    expect(service.serviceState.status).toBe(ServiceNetworkState.connected);

    service.stop();
    expect(service.serviceState.status).toBe(ServiceNetworkState.disconnected);
    expect(last().stopped).toBe(1);
    expect(last().disposed).toBe(1);
  });

  it("stop sets the status even when the backend never reports onStop", () => {
    const { service } = makeService();
    service.start();
    last().receiver.onStart();
    last().stop = function () {
      this.stopped++; // a backend that forgets onStop()
    };
    service.stop();
    expect(service.serviceState.status).toBe(ServiceNetworkState.disconnected);
  });

  it("stop while connecting cancels; a late onStart from that instance is ignored", () => {
    const { service } = makeService();
    service.start();
    const pending = last();
    service.stop();
    expect(service.serviceState.status).toBe(ServiceNetworkState.disconnected);
    pending.receiver.onStart();
    expect(service.serviceState.status).toBe(ServiceNetworkState.disconnected);
  });

  it("a replaced instance cannot stop or error the new one", () => {
    const { service } = makeService();
    service.start();
    const old = last();
    service.start();
    const current = last();
    expect(old).not.toBe(current);
    current.receiver.onStart();

    old.receiver.onStop("late failure from the old instance");
    old.receiver.onError("late error");
    expect(service.serviceState.status).toBe(ServiceNetworkState.connected);
    expect(service.serviceState.error).toBe("");
    expect(toast).not.toHaveBeenCalled();
    expect(pushSystemLog).not.toHaveBeenCalled();
  });

  it("a backend failing on its own goes to disconnected with the error, is disposed, and toasts", () => {
    const { service } = makeService();
    service.start();
    const backend = last();
    backend.receiver.onStop("Server not reachable");
    expect(service.serviceState.status).toBe(ServiceNetworkState.disconnected);
    expect(service.serviceState.error).toBe("Server not reachable");
    expect(backend.disposed).toBe(1);
    expect(toast).toHaveBeenCalledWith("toasts.tts_stopped_error", expect.objectContaining({ type: "error" }));
  });

  it("missing options show the localized setup toast", () => {
    const { service } = makeService();
    FakeBackend.startImpl = (self) => self.receiver.onStop(TTS_OPTIONS_MISSING);
    service.start();
    expect(service.serviceState.status).toBe(ServiceNetworkState.disconnected);
    expect(toast).toHaveBeenCalledWith("toasts.tts_setup_required", expect.objectContaining({ type: "warning" }));
  });

  it("a start that throws or rejects ends disconnected with the error", async () => {
    const { service } = makeService();
    FakeBackend.startImpl = () => {
      throw new Error("boom");
    };
    service.start();
    expect(service.serviceState.status).toBe(ServiceNetworkState.disconnected);
    expect(service.serviceState.error).toBe("boom");

    FakeBackend.startImpl = () => Promise.reject("rejected");
    service.start();
    await Promise.resolve();
    await Promise.resolve();
    expect(service.serviceState.status).toBe(ServiceNetworkState.disconnected);
    expect(service.serviceState.error).toBe("rejected");
  });

  it("clears the previous error on start", () => {
    const { service } = makeService();
    service.start();
    last().receiver.onStop("bad key");
    expect(service.serviceState.error).toBe("bad key");
    service.start();
    expect(service.serviceState.error).toBe("");
  });

  it("onError logs every failure but throttles toasts", () => {
    const { service } = makeService();
    service.start();
    last().receiver.onStart();
    last().receiver.onError("Kokoro: HTTP 500");
    last().receiver.onError("Kokoro: HTTP 500");
    expect(pushSystemLog).toHaveBeenCalledTimes(2);
    expect(pushSystemLog).toHaveBeenCalledWith("TTS", "Kokoro: HTTP 500", "error");
    expect(toast).toHaveBeenCalledTimes(1);
    expect(toast).toHaveBeenCalledWith("toasts.tts_play_failed", expect.objectContaining({ type: "error" }));
    expect(service.serviceState.status).toBe(ServiceNetworkState.connected);
  });
});

describe("Service_TTS play", () => {
  it("speaks only while connected, after word replacements", () => {
    const { service, data } = makeService();
    data.replaceWords = { lol: "laughing" };
    service.updateReplacementsCache();

    service.play("before start");
    service.start();
    service.play("while connecting");
    last().receiver.onStart();
    service.play("lol ok");
    service.play("   ");
    expect(last().played).toEqual(["laughing ok"]);

    const backend = last();
    service.stop();
    service.play("after stop");
    expect(backend.played).toEqual(["laughing ok"]);
  });

  it("reports a rejected play as an utterance error, not a stop", async () => {
    const { service } = makeService();
    service.start();
    const backend = last();
    backend.receiver.onStart();
    backend.play = () => Promise.reject(new Error("network"));
    service.play("hi");
    await Promise.resolve();
    await Promise.resolve();
    expect(pushSystemLog).toHaveBeenCalledWith("TTS", "network", "error");
    expect(service.serviceState.status).toBe(ServiceNetworkState.connected);
  });
});

describe("Service_TTS keyboard input dedupe", () => {
  const textfieldSubs = () => subscriptions.filter((s) => s.source === TextEventSource.textfield);

  it.each([TextEventSource.textfield, TextEventSource.any])(
    "source %s already covers typed text: one subscription, spoken once",
    async (source) => {
      const { service } = makeService({ source, inputField: true });
      await service.init();
      // Only the source subscription; no extra textfield subscription.
      expect(subscriptions).toHaveLength(1);
      expect(subscriptions[0].source).toBe(source);
      service.dispose();
    },
  );

  it("source stt plus keyboard input subscribes to both", async () => {
    const { service } = makeService({ source: TextEventSource.stt, inputField: true });
    await service.init();
    expect(subscriptions.map((s) => s.source)).toEqual([TextEventSource.stt, TextEventSource.textfield]);

    service.start();
    last().receiver.onStart();
    textfieldSubs()[0].fn({ type: TextEventType.final, value: "typed" });
    textfieldSubs()[0].fn({ type: TextEventType.interim, value: "ignored" });
    expect(last().played).toEqual(["typed"]);
    service.dispose();
  });

  it("keyboard input off: no textfield subscription", async () => {
    const { service } = makeService({ source: TextEventSource.stt, inputField: false });
    await service.init();
    expect(textfieldSubs()).toHaveLength(0);
    service.dispose();
  });
});

describe("Service_TTS Smart Turn commit (P6-C gate, wired by P6-I0)", () => {
  const sttSub = () => subscriptions.find((s) => s.source === TextEventSource.stt)!;
  type TurnEvent = { type: TextEventType; value: string; turnEnd?: boolean; lang?: string };
  const say = (e: TurnEvent) => (sttSub().fn as unknown as (e: TurnEvent) => void)(e);

  async function running(overrides: Partial<TTS_State>) {
    const made = makeService({ source: TextEventSource.stt, inputField: false, ...overrides });
    await made.service.init();
    made.service.start();
    last().receiver.onStart();
    return made;
  }

  it("off (default): every final is spoken at once, and STT's cancel marker never", async () => {
    const { service } = await running({});
    say({ type: TextEventType.final, value: "I think", turnEnd: false, lang: "en" });
    say({ type: TextEventType.final, value: "[...]" });
    expect(last().played).toEqual(["I think"]);
    service.dispose();
  });

  it("on: an unfinished turn waits and is spoken joined with the rest", async () => {
    const { service } = await running({ commitOnTurnEnd: true });
    say({ type: TextEventType.final, value: "I think", turnEnd: false, lang: "en" });
    say({ type: TextEventType.interim, value: "we should" });
    expect(last().played).toEqual([]);
    say({ type: TextEventType.final, value: "we should go.", turnEnd: true, lang: "en" });
    expect(last().played).toEqual(["I think we should go."]);
    service.dispose();
  });

  it("on: held text is spoken after the hold time without speech, and dropped by stop", async () => {
    vi.useFakeTimers();
    try {
      const { service, data } = await running({ commitOnTurnEnd: true, turnHoldMs: "800" });
      say({ type: TextEventType.final, value: "so", turnEnd: false, lang: "en" });
      vi.advanceTimersByTime(700);
      expect(last().played).toEqual([]);
      vi.advanceTimersByTime(200);
      expect(last().played).toEqual(["so"]);

      say({ type: TextEventType.final, value: "never mind", turnEnd: false, lang: "en" });
      const backend = last();
      service.stop();
      vi.advanceTimersByTime(5_000);
      expect(backend.played).toEqual(["so"]);
      expect(data.commitOnTurnEnd).toBe(true);
      service.dispose();
    } finally {
      vi.useRealTimers();
    }
  });
});

describe("Service_TTS desktop default (P2)", () => {
  function makeDesktop(overrides: Partial<TTS_State> = {}, root: { onboardingComplete: boolean; isApp?: boolean } = { onboardingComplete: true }) {
    const { service, data } = makeService(overrides);
    const w = window as unknown as { Config: unknown; ApiServer: Record<string, unknown> };
    w.Config = { isApp: () => root.isApp !== false };
    const state = w.ApiServer.state as Record<string, unknown>;
    state.onboardingComplete = root.onboardingComplete;
    w.ApiServer.patchService = (_key: "tts", fn: (s: { data: TTS_State }) => void) => fn({ data });
    w.ApiServer.changeTab = vi.fn();
    return { service, data, changeTab: w.ApiServer.changeTab as ReturnType<typeof vi.fn> };
  }

  beforeEach(() => {
    vi.mocked(ask).mockClear();
  });

  it("starts a new install on the local Kokoro voice without asking", async () => {
    const { service, data } = makeDesktop({}, { onboardingComplete: false });
    await service.init();
    expect(data.backend).toBe(TTS_Backends.localNeural);
    expect(data.desktopDefaultMigrated).toBe(true);
    await service.runStartupOffer();
    expect(ask).not.toHaveBeenCalled();
    expect(FakeBackend.instances).toHaveLength(0);
    service.dispose();
  });

  it("offers Kokoro once to an existing Web Speech profile and switches on yes", async () => {
    const { service, data, changeTab } = makeDesktop({ everStarted: true });
    await service.init();
    expect(data.backend).toBe(TTS_Backends.native);
    expect(data.desktopDefaultMigrated).toBe(true);
    vi.mocked(ask).mockResolvedValueOnce(true);
    await service.runStartupOffer();
    expect(ask).toHaveBeenCalledTimes(1);
    expect(data.localOfferShown).toBe(true);
    expect(data.backend).toBe(TTS_Backends.localNeural);
    expect(changeTab).toHaveBeenCalledWith({ tab: "tts" });
    // Once per launch, and never again on later launches.
    await service.runStartupOffer();
    expect(ask).toHaveBeenCalledTimes(1);
    service.dispose();
  });

  it("keeps the backend when the offer is declined, and never asks again", async () => {
    const { service, data } = makeDesktop({ backend: TTS_Backends.windows, everStarted: true });
    await service.init();
    await service.runStartupOffer();
    expect(ask).toHaveBeenCalledTimes(1);
    expect(data.backend).toBe(TTS_Backends.windows);
    expect(data.localOfferShown).toBe(true);
    const again = makeDesktop({ ...data });
    await again.service.init();
    await again.service.runStartupOffer();
    expect(ask).toHaveBeenCalledTimes(1);
    service.dispose();
    again.service.dispose();
  });

  it("stops a running session before switching", async () => {
    const { service, data } = makeDesktop({ autoStart: true });
    await service.init();
    last().receiver.onStart();
    vi.mocked(ask).mockResolvedValueOnce(true);
    await service.runStartupOffer();
    expect(service.serviceState.status).toBe(ServiceNetworkState.disconnected);
    expect(data.backend).toBe(TTS_Backends.localNeural);
    service.dispose();
  });

  it("does not ask a profile that never used TTS, and asks at the launch after its first start", async () => {
    const { service, data } = makeDesktop();
    await service.init();
    await service.runStartupOffer();
    expect(ask).not.toHaveBeenCalled();
    expect(data.desktopDefaultMigrated).toBe(false);
    expect(data.localOfferShown).toBe(false);
    expect(data.everStarted).toBe(false);

    // A start that never comes up records nothing.
    FakeBackend.startImpl = () => { throw new Error("no voice"); };
    service.start();
    expect(data.everStarted).toBe(false);
    FakeBackend.startImpl = () => {};
    service.start();
    last().receiver.onStart();
    expect(data.everStarted).toBe(true);
    service.dispose();

    const next = makeDesktop({ ...data });
    await next.service.init();
    await next.service.runStartupOffer();
    expect(ask).toHaveBeenCalledTimes(1);
    expect(data.localOfferShown).toBe(false); // the first launch's data; the next launch recorded its own
    expect(next.data.localOfferShown).toBe(true);
    next.service.dispose();
  });

  it("does nothing in the web build", async () => {
    const { service, data } = makeDesktop({}, { onboardingComplete: false, isApp: false });
    await service.init();
    expect(data.backend).toBe(TTS_Backends.native);
    expect(data.desktopDefaultMigrated).toBe(false);
    service.dispose();
  });
});
