/**
 * P6-J: Windows SAPI through the TTS sink (`synthesizeToSink`), and the microphone policy the TTS service mirrors from
 * the voice changer into the sink.
 */
import i18n from "i18next";
import { proxy } from "valtio";
import { afterEach, beforeAll, beforeEach, describe, expect, it, vi } from "vitest";

vi.mock("@tauri-apps/api/core", () => ({ invoke: vi.fn(), isTauri: () => false }));
vi.mock("react-toastify", () => ({
  toast: Object.assign(vi.fn(), { error: vi.fn(), info: vi.fn(), warning: vi.fn(), success: vi.fn() }),
}));
vi.mock("../../systemLog", () => ({ pushSystemLog: vi.fn() }));
vi.mock("@/utils/devLog", () => ({ devLog: () => {} }));
vi.mock("@tauri-apps/plugin-dialog", () => ({ ask: vi.fn(async () => false) }));

import { invoke } from "@tauri-apps/api/core";
import Service_TTS, { MicPolicySync, TTSBackendMap, ttsMicPolicy } from "../index";
import { Service_TTS_Schema, TTS_Backends, TTS_State } from "../schema";
import { TTS_WindowsService, windowsSinkRequest } from "../services/windows";
import { ITTSReceiver, ITTSService, TTS_OPTIONS_MISSING, TtsSinkContext, ttsOutputMode } from "../types";

const invokeMock = vi.mocked(invoke);

function receiver() {
  return { onStart: vi.fn(), onStop: vi.fn(), onError: vi.fn() } satisfies ITTSReceiver;
}

const flush = async (n = 10) => {
  for (let i = 0; i < n; i++) await Promise.resolve();
};

const sinkCtx = (utteranceId = "tts-a-1"): TtsSinkContext => ({
  utteranceId,
  target: { route: "voiceChanger", device: "", volume: 1 },
  lang: "en",
});

let data: TTS_State;

beforeAll(async () => {
  await i18n.init({ lng: "en", resources: { en: { translation: {} } } });
});

beforeEach(() => {
  invokeMock.mockReset();
  data = proxy(Service_TTS_Schema.parse({}));
  vi.stubGlobal("window", { ApiServer: { state: { services: { tts: { data } } } } });
});

afterEach(() => {
  vi.unstubAllGlobals();
});

describe("Windows SAPI through the TTS sink", () => {
  it("is a sink backend, so the sentence pipeline and the route apply", () => {
    const win: ITTSService = new TTS_WindowsService(receiver());
    expect(ttsOutputMode(win)).toBe("sink");
    expect(win.selfPlaying).toBeFalsy();
  });

  it("maps the settings onto the Rust request (rate clamped, voice trimmed)", () => {
    expect(windowsSinkRequest({ device: "", voice: " v ", volume: "1", rate: "9" }, "Hi.")).toEqual({
      voice: "v",
      text: "Hi.",
      rate: 5,
    });
    expect(windowsSinkRequest({ device: "", voice: "v", volume: "1", rate: "abc" }, "x").rate).toBe(1);
    expect(windowsSinkRequest({ device: "", voice: "v", volume: "1", rate: "0" }, "x").rate).toBe(0.1);
  });

  it("starts with a voice alone (the device is optional now: the sink's default)", () => {
    const r = receiver();
    new TTS_WindowsService(r).start(data);
    expect(r.onStop).toHaveBeenCalledWith(TTS_OPTIONS_MISSING);

    data.windows.voice = "HKEY\\Voices\\Tokens\\Zira";
    const ok = receiver();
    new TTS_WindowsService(ok).start(data);
    expect(ok.onStart).toHaveBeenCalled();
    expect(ok.onStop).not.toHaveBeenCalled();
  });

  it("renders one sentence into the begun sink utterance", async () => {
    Object.assign(data.windows, { voice: "zira", rate: "2" });
    invokeMock.mockResolvedValue({ sampleRate: 16_000, frames: 16_000, firstAudioMs: 20, totalMs: 40 });
    const win = new TTS_WindowsService(receiver());
    win.start(data);
    await win.synthesizeToSink("Hello there.", sinkCtx("tts-x-7"), new AbortController().signal);
    expect(invokeMock).toHaveBeenCalledWith("plugin:windows-tts|speak_to_sink", {
      request: { voice: "zira", text: "Hello there.", rate: 2 },
      utteranceId: "tts-x-7",
    });
    // Never the legacy device path.
    expect(invokeMock.mock.calls.some(([c]) => c === "plugin:windows-tts|speak")).toBe(false);
  });

  it("does nothing before start, after stop, for empty text or an aborted signal", async () => {
    data.windows.voice = "v";
    const win = new TTS_WindowsService(receiver());
    await win.synthesizeToSink("x", sinkCtx(), new AbortController().signal);
    win.start(data);
    await win.synthesizeToSink("   ", sinkCtx(), new AbortController().signal);
    const aborted = new AbortController();
    aborted.abort();
    await win.synthesizeToSink("x", sinkCtx(), aborted.signal);
    expect(invokeMock).not.toHaveBeenCalled();
  });

  it("cancels the render on abort and treats a cancelled answer as no error", async () => {
    data.windows.voice = "v";
    let reject!: (e: unknown) => void;
    invokeMock.mockImplementation(((cmd: string) =>
      cmd === "plugin:windows-tts|speak_to_sink"
        ? new Promise((_, rej) => {
            reject = rej;
          })
        : Promise.resolve()) as never);
    const win = new TTS_WindowsService(receiver());
    win.start(data);
    const ctl = new AbortController();
    const speaking = win.synthesizeToSink("A long sentence.", sinkCtx("tts-s-3"), ctl.signal);
    await flush();
    ctl.abort();
    expect(invokeMock).toHaveBeenCalledWith("plugin:windows-tts|cancel_speak_to_sink", { utteranceId: "tts-s-3" });
    reject("cancelled");
    await expect(speaking).resolves.toBeUndefined();

    // A cancelled answer without an abort (the sink cancelled it: stop / skip / flush) is not an error either.
    invokeMock.mockRejectedValueOnce("cancelled");
    await expect(win.synthesizeToSink("x", sinkCtx("tts-s-4"), new AbortController().signal)).resolves.toBeUndefined();
  });

  it("reports a failed sentence with a readable message", async () => {
    data.windows.voice = "v";
    invokeMock.mockRejectedValueOnce("The selected Windows voice is not installed");
    const win = new TTS_WindowsService(receiver());
    win.start(data);
    await expect(win.synthesizeToSink("x", sinkCtx(), new AbortController().signal)).rejects.toThrow(
      "Windows speech: The selected Windows voice is not installed",
    );
  });

  it("stop cancels everything the SAPI thread still has, once", () => {
    data.windows.voice = "v";
    invokeMock.mockResolvedValue(undefined);
    const r = receiver();
    const win = new TTS_WindowsService(r);
    win.start(data);
    win.stop();
    win.dispose();
    const cancels = invokeMock.mock.calls.filter(([c]) => c === "plugin:windows-tts|cancel_speak_to_sink");
    expect(cancels).toEqual([["plugin:windows-tts|cancel_speak_to_sink", { utteranceId: null }]]);
    expect(r.onStop).toHaveBeenCalledTimes(1);
  });
});

describe("microphone policy while TTS plays into the voice changer", () => {
  it("maps the voice changer's auxMicGate values", () => {
    expect(ttsMicPolicy("mute")).toBe("mute");
    expect(ttsMicPolicy("duck")).toBe("duck");
    expect(ttsMicPolicy("none")).toBe("none");
    expect(ttsMicPolicy("loud")).toBe("none");
    expect(ttsMicPolicy(undefined)).toBe("none");
  });

  it("sends each change once, in order, and again after a failed send", async () => {
    const sent: string[] = [];
    let fail = false;
    const sync = new MicPolicySync(async (p) => {
      await Promise.resolve();
      if (fail) throw new Error("no sink");
      sent.push(p);
    });
    await sync.update("mute");
    await sync.update("mute");
    expect(sent).toEqual(["mute"]);
    // Two quick changes arrive in order.
    void sync.update("duck");
    await sync.update("none");
    expect(sent).toEqual(["mute", "duck", "none"]);
    // A failed send is retried on the next update with the same value.
    fail = true;
    await sync.update("mute");
    expect(sync.sent).toBeNull();
    fail = false;
    await sync.update("mute");
    expect(sent).toEqual(["mute", "duck", "none", "mute"]);
    expect(sync.sent).toBe("mute");
  });

  it("the TTS service mirrors the voice changer's setting into the sink on the desktop", async () => {
    class Idle implements ITTSService {
      start() {}
      stop() {}
      play() {}
      dispose() {}
    }
    const vcState = proxy({ voiceChanger: { auxMicGate: "none" }, isRunning: false });
    const tts = proxy(Service_TTS_Schema.parse({ backend: TTS_Backends.windows, desktopDefaultMigrated: true }));
    vi.stubGlobal("window", {
      Config: { isApp: () => true },
      ApiServer: {
        state: { services: { tts: { data: tts } }, audioOutputDevice: "", onboardingComplete: true },
        voiceChanger: { state: vcState },
        patchService: (_k: string, fn: (s: { data: TTS_State }) => void) => fn({ data: tts }),
      },
      ApiShared: {
        pubsub: { subscribe: vi.fn(() => "s"), unsubscribe: vi.fn(), subscribeText: vi.fn(() => "t") },
      },
    });
    invokeMock.mockResolvedValue(undefined);
    const backends = Object.fromEntries(Object.values(TTS_Backends).map((b) => [b, Idle])) as unknown as TTSBackendMap;
    const service = new Service_TTS(backends);
    await service.init();
    await flush();
    const policies = () =>
      invokeMock.mock.calls.filter(([c]) => c === "plugin:tts-sink|set_mic_policy").map(([, a]) => (a as { policy: string }).policy);
    expect(policies()).toEqual(["none"]);

    vcState.voiceChanger.auxMicGate = "mute";
    await flush(20);
    expect(policies()).toEqual(["none", "mute"]);
    vcState.isRunning = true; // unrelated changes send nothing new
    await flush(20);
    expect(policies()).toEqual(["none", "mute"]);

    service.dispose();
    vcState.voiceChanger.auxMicGate = "duck";
    await flush(20);
    expect(policies(), "no longer followed after dispose").toEqual(["none", "mute"]);
  });

  it("does not follow the voice changer outside the desktop app", async () => {
    const vcState = proxy({ voiceChanger: { auxMicGate: "mute" } });
    const tts = proxy(Service_TTS_Schema.parse({}));
    vi.stubGlobal("window", {
      ApiServer: { state: { services: { tts: { data: tts } }, audioOutputDevice: "" }, voiceChanger: { state: vcState } },
      ApiShared: { pubsub: { subscribe: vi.fn(() => "s"), unsubscribe: vi.fn(), subscribeText: vi.fn(() => "t") } },
    });
    const service = new Service_TTS({} as TTSBackendMap);
    await service.init();
    await flush();
    expect(invokeMock.mock.calls.some(([c]) => c === "plugin:tts-sink|set_mic_policy")).toBe(false);
    service.dispose();
  });
});
