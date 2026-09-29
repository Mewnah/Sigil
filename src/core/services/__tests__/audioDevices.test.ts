import { proxy } from "valtio";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import { ServiceNetworkState, type RustAudioDevicesChanged, type RustAudioStreamEvent } from "@/types";

const invokeMock = vi.fn();
const handlers = new Map<string, (e: { payload: unknown }) => void>();

vi.mock("@tauri-apps/api/core", () => ({ invoke: (...args: unknown[]) => invokeMock(...args), isTauri: () => true }));
vi.mock("@tauri-apps/api/event", () => ({
  listen: vi.fn(async (name: string, cb: (e: { payload: unknown }) => void) => {
    handlers.set(name, cb);
    return () => handlers.delete(name);
  }),
}));
vi.mock("react-toastify", () => ({
  toast: Object.assign(vi.fn(), {
    error: vi.fn(),
    info: vi.fn(),
    warning: vi.fn(),
    success: vi.fn(),
    update: vi.fn(),
    dismiss: vi.fn(),
  }),
}));
vi.mock("@/core/services/systemLog", () => ({ pushSystemLog: vi.fn() }));

import { toast } from "react-toastify";
import { STT_Backends } from "../stt/schema";
import { SttMuteState } from "../stt/types";
import Service_AudioDevices, {
  AUDIO_DEVICE_EVENTS,
  AUDIO_STREAM_CLOSED_EVENT,
  lostAny,
  planSttRestart,
  sttCaptureClass,
  webAudioInputPresent,
  type SttRestartInput,
} from "../audioDevices";

/** Web Audio inputs the fake webview lists (`deviceId`s). */
let webInputs: string[] | null = null;

function stubWebAudio() {
  vi.stubGlobal("navigator", {
    mediaDevices: {
      enumerateDevices: vi.fn(async () =>
        (webInputs ?? []).map((deviceId) => ({ deviceId, kind: "audioinput", label: deviceId, groupId: "" })),
      ),
    },
  });
}

const view = (inputs: string[], defaultInput: string | null, outputs = ["Speakers"]) => ({
  inputs,
  outputs,
  defaultInput,
  defaultOutput: outputs[0] ?? null,
});

function snapshot(inputs: string[], defaultInput: string | null, reason = "snapshot"): RustAudioDevicesChanged {
  return {
    reason,
    inputs: inputs.map((n) => ({ id: n, name: n })),
    outputs: [{ id: "Speakers", name: "Speakers" }],
    default_input: defaultInput,
    default_output: "Speakers",
  };
}

describe("pure policy", () => {
  it("classifies STT backends by who captures the audio", () => {
    expect(sttCaptureClass(STT_Backends.whisper)).toBe("hub");
    expect(sttCaptureClass(STT_Backends.vosk)).toBe("hub");
    expect(sttCaptureClass(STT_Backends.openai_audio)).toBe("hub");
    expect(sttCaptureClass(STT_Backends.moonshine)).toBe("hub");
    expect(sttCaptureClass(STT_Backends.azure)).toBe("webview");
    expect(sttCaptureClass(STT_Backends.deepgram)).toBe("webview");
    expect(sttCaptureClass(STT_Backends.chrome)).toBe("external");
    expect(sttCaptureClass("bogus")).toBe("none");
  });

  it("detects lost devices with duplicates counted", () => {
    expect(lostAny(["a", "b"], ["b", "a"])).toBe(false);
    expect(lostAny(["a", "a"], ["a"])).toBe(true);
    expect(lostAny(["a"], ["a", "b"])).toBe(false);
  });

  const base: SttRestartInput = {
    backend: STT_Backends.whisper,
    status: ServiceNetworkState.connected,
    error: "",
    source: null,
    followsDefaultMic: true,
    failedWhileSourceMissing: false,
    before: view(["Mic"], "Mic"),
    after: view(["Mic", "USB Mic"], "USB Mic"),
  };

  it("never restarts a running hub-captured session (the hub follows devices itself)", () => {
    expect(planSttRestart(base)).toBe("none");
    expect(planSttRestart({ ...base, before: base.after, after: base.before })).toBe("none");
  });

  it("retries a hub backend whose start failed on a missing device once it appears", () => {
    const failed = {
      ...base,
      status: ServiceNetworkState.disconnected,
      error: "Input device 'USB Mic' not found",
      source: "USB Mic",
      failedWhileSourceMissing: true,
    };
    expect(planSttRestart(failed)).toBe("restart");
    // Not when the failure had another cause, the device was already there, or it is still missing.
    expect(planSttRestart({ ...failed, failedWhileSourceMissing: false })).toBe("none");
    expect(planSttRestart({ ...failed, before: failed.after })).toBe("none");
    expect(planSttRestart({ ...failed, after: view(["Mic"], "Mic") })).toBe("none");
    expect(planSttRestart({ ...failed, error: "" })).toBe("none");
    // A default-input source that had no device at all.
    expect(
      planSttRestart({ ...failed, source: null, before: view([], null), after: view(["Mic"], "Mic") }),
    ).toBe("restart");
  });

  it("restarts browser-captured backends when their microphone changes", () => {
    const azure = { ...base, backend: STT_Backends.azure };
    expect(planSttRestart(azure)).toBe("restart");
    expect(planSttRestart({ ...azure, status: ServiceNetworkState.disconnected })).toBe("none");
    expect(planSttRestart({ ...azure, after: view(["Mic", "USB Mic"], "Mic") })).toBe("none");
    expect(planSttRestart({ ...base, backend: STT_Backends.chrome })).toBe("none");
  });

  it("restarts a pinned browser-captured device only when that device itself changes", () => {
    const pinned: SttRestartInput = {
      ...base,
      backend: STT_Backends.azure,
      followsDefaultMic: false,
      before: view(["Headset", "Webcam Mic"], "Headset"),
      after: view(["Headset"], "Headset"),
    };
    // An unrelated device (a webcam mic) was unplugged: the pinned headset is still listed.
    expect(planSttRestart({ ...pinned, webDeviceBefore: true, webDeviceAfter: true })).toBe("none");
    // Presence unknown (no Web Audio list): never restart a working session on a guess.
    expect(planSttRestart(pinned)).toBe("none");
    expect(planSttRestart({ ...pinned, webDeviceBefore: null, webDeviceAfter: false })).toBe("none");
    // Its own device went away while connected: reopen (Deepgram falls back to the default microphone).
    expect(planSttRestart({ ...pinned, webDeviceBefore: true, webDeviceAfter: false })).toBe("restart");
    expect(
      planSttRestart({ ...pinned, webDeviceBefore: true, webDeviceAfter: false, after: view([], null) }),
    ).toBe("none");
    // It came back: a session running on the fallback moves back to it...
    expect(planSttRestart({ ...pinned, webDeviceBefore: false, webDeviceAfter: true })).toBe("restart");
    // ...and a start that failed without it is retried.
    const failed = {
      ...pinned,
      status: ServiceNetworkState.disconnected,
      error: "ConnectionFailure",
      failedWhileSourceMissing: true,
      webDeviceBefore: false,
      webDeviceAfter: true,
    };
    expect(planSttRestart(failed)).toBe("restart");
    expect(planSttRestart({ ...failed, failedWhileSourceMissing: false })).toBe("none");
    expect(planSttRestart({ ...failed, webDeviceBefore: true, webDeviceAfter: false })).toBe("none");
  });

  it("reads Web Audio presence defensively", async () => {
    vi.stubGlobal("navigator", undefined);
    expect(await webAudioInputPresent("x")).toBeNull();
    stubWebAudio();
    webInputs = ["a", "b"];
    expect(await webAudioInputPresent("b")).toBe(true);
    expect(await webAudioInputPresent("c")).toBe(false);
    // Ids are hidden before microphone permission: unknown.
    webInputs = ["", ""];
    expect(await webAudioInputPresent("b")).toBeNull();
    vi.unstubAllGlobals();
  });
});

describe("Service_AudioDevices", () => {
  let stt: {
    serviceState: { status: ServiceNetworkState; error: string; muted: SttMuteState };
    start: ReturnType<typeof vi.fn>;
  };
  let sttData: { backend: STT_Backends; whisper: { device: string }; azure: { device: string } } & Record<string, unknown>;
  let current: RustAudioDevicesChanged;

  beforeEach(() => {
    vi.useFakeTimers();
    handlers.clear();
    invokeMock.mockReset();
    vi.mocked(toast.dismiss).mockClear();
    vi.mocked(toast.success).mockClear();
    webInputs = null;
    stubWebAudio();
    current = snapshot(["Mic"], "Mic");
    invokeMock.mockImplementation(async (cmd: string) => {
      if (cmd === "plugin:audio|get_device_snapshot") return current;
      return undefined;
    });
    stt = {
      serviceState: proxy({ status: ServiceNetworkState.disconnected, error: "", muted: SttMuteState.unmuted }),
      start: vi.fn(),
    };
    sttData = {
      backend: STT_Backends.whisper,
      whisper: { device: "" },
      vosk: { device: "" },
      moonshine: { device: "" },
      openai_audio: { device: "" },
      azure: { device: "default" },
      deepgram: { device: "default" },
    };
    (globalThis as { window?: unknown }).window = {
      Config: { isApp: () => true },
      ApiServer: {
        stt,
        state: {
          audioInputDevice: "",
          audioInputDeviceNative: "",
          services: { stt: { data: sttData } },
        },
      },
    };
  });

  afterEach(() => {
    vi.useRealTimers();
    vi.unstubAllGlobals();
    delete (globalThis as { window?: unknown }).window;
  });

  async function started() {
    const service = new Service_AudioDevices();
    await service.init();
    return service;
  }

  function fire<T>(event: string, payload: T) {
    const handler = handlers.get(event);
    expect(handler, event).toBeDefined();
    handler!({ payload });
  }

  it("loads the devices and follows device events", async () => {
    const service = await started();
    expect(service.state.inputs.map((d) => d.name)).toEqual(["Mic"]);
    expect(service.state.defaultInput).toBe("Mic");
    const revision = service.state.revision;
    fire(AUDIO_DEVICE_EVENTS.devicesChanged, snapshot(["Mic", "USB Mic"], "USB Mic", "added"));
    expect(service.state.inputs.map((d) => d.name)).toEqual(["Mic", "USB Mic"]);
    expect(service.state.defaultInput).toBe("USB Mic");
    expect(service.state.revision).toBe(revision + 1);
    service.dispose();
  });

  it("drives the Rust mute gate from the STT hard mute only", async () => {
    const service = await started();
    const gateCalls = () => invokeMock.mock.calls.filter(([cmd]) => cmd === "plugin:audio|set_capture_mute");
    expect(gateCalls()).toEqual([["plugin:audio|set_capture_mute", { group: "stt", muted: false }]]);
    stt.serviceState.muted = SttMuteState.muted;
    await vi.advanceTimersByTimeAsync(0);
    expect(gateCalls().at(-1)).toEqual(["plugin:audio|set_capture_mute", { group: "stt", muted: true }]);
    // A pending unmute already counts as unmuted for engines.
    stt.serviceState.muted = SttMuteState.pendingUnmute;
    await vi.advanceTimersByTimeAsync(0);
    expect(gateCalls().at(-1)).toEqual(["plugin:audio|set_capture_mute", { group: "stt", muted: false }]);
    stt.serviceState.muted = SttMuteState.unmuted;
    await vi.advanceTimersByTimeAsync(0);
    expect(gateCalls()).toHaveLength(3);
    service.dispose();
  });

  it("reports stream outages and recoveries", async () => {
    const service = await started();
    const ev: RustAudioStreamEvent = { source: "USB Mic", epoch: 3, message: "unplugged" };
    fire(AUDIO_DEVICE_EVENTS.streamError, ev);
    expect(service.state.failing["USB Mic"]).toEqual({ epoch: 3, message: "unplugged" });
    expect(toast.warning).toHaveBeenCalled();
    fire(AUDIO_DEVICE_EVENTS.streamRecovered, { ...ev, message: null });
    expect(service.state.failing["USB Mic"]).toBeUndefined();
    expect(toast.dismiss).toHaveBeenCalledWith("audio-stream-USB Mic");
    expect(toast.success).toHaveBeenCalled();
    service.dispose();
  });

  it("ends an outage whose source closed before it recovered", async () => {
    const service = await started();
    fire(AUDIO_DEVICE_EVENTS.streamError, { source: "USB Mic", epoch: 3, message: "unplugged" });
    // A close of an older epoch never ends a newer outage.
    fire(AUDIO_STREAM_CLOSED_EVENT, { source: "USB Mic", epoch: 2, message: null });
    expect(service.state.failing["USB Mic"]).toEqual({ epoch: 3, message: "unplugged" });
    expect(toast.dismiss).not.toHaveBeenCalled();
    // STT stopped (or switched device) while the mic was unplugged: the hub closed the source.
    fire(AUDIO_STREAM_CLOSED_EVENT, { source: "USB Mic", epoch: 3, message: null });
    expect(service.state.failing["USB Mic"]).toBeUndefined();
    expect(toast.dismiss).toHaveBeenCalledWith("audio-stream-USB Mic");
    expect(toast.success).not.toHaveBeenCalled();
    service.dispose();
  });

  it("restarts a pinned browser-captured STT only when its own microphone goes away or comes back", async () => {
    sttData.backend = STT_Backends.azure;
    sttData.azure.device = "web-headset";
    webInputs = ["web-mic", "web-headset", "web-webcam"];
    current = snapshot(["Mic", "Headset", "Webcam Mic"], "Mic");
    stt.serviceState.status = ServiceNetworkState.connected;
    const service = await started();

    // An unrelated webcam microphone is unplugged: the Azure session keeps running.
    webInputs = ["web-mic", "web-headset"];
    fire(AUDIO_DEVICE_EVENTS.devicesChanged, snapshot(["Mic", "Headset"], "Mic", "removed"));
    await vi.advanceTimersByTimeAsync(1_000);
    expect(stt.start).not.toHaveBeenCalled();

    // The headset itself is unplugged.
    webInputs = ["web-mic"];
    fire(AUDIO_DEVICE_EVENTS.devicesChanged, snapshot(["Mic"], "Mic", "removed"));
    await vi.advanceTimersByTimeAsync(1_000);
    expect(stt.start).toHaveBeenCalledTimes(1);

    // Azure cannot open the missing device; once it is back, STT starts again.
    stt.serviceState.error = "ConnectionFailure";
    stt.serviceState.status = ServiceNetworkState.disconnected;
    await vi.advanceTimersByTimeAsync(5_000);
    webInputs = ["web-mic", "web-headset"];
    fire(AUDIO_DEVICE_EVENTS.devicesChanged, snapshot(["Mic", "Headset"], "Mic", "added"));
    await vi.advanceTimersByTimeAsync(1_000);
    expect(stt.start).toHaveBeenCalledTimes(2);
    service.dispose();
  });

  it("restarts STT once the missing device of a failed start is plugged in", async () => {
    sttData.whisper.device = "USB Mic";
    const service = await started();
    // The start fails because the device is missing.
    stt.serviceState.status = ServiceNetworkState.connecting;
    await vi.advanceTimersByTimeAsync(0);
    stt.serviceState.error = "Input device 'USB Mic' not found";
    stt.serviceState.status = ServiceNetworkState.disconnected;
    await vi.advanceTimersByTimeAsync(0);
    fire(AUDIO_DEVICE_EVENTS.devicesChanged, snapshot(["Mic", "USB Mic"], "Mic", "added"));
    expect(stt.start).not.toHaveBeenCalled();
    await vi.advanceTimersByTimeAsync(1_000);
    expect(stt.start).toHaveBeenCalledTimes(1);
    service.dispose();
  });

  it("restarts a browser-captured STT on a default microphone change, rate limited", async () => {
    sttData.backend = STT_Backends.azure;
    stt.serviceState.status = ServiceNetworkState.connected;
    const service = await started();
    fire(AUDIO_DEVICE_EVENTS.devicesChanged, snapshot(["Mic", "USB Mic"], "USB Mic", "added"));
    await vi.advanceTimersByTimeAsync(1_000);
    expect(stt.start).toHaveBeenCalledTimes(1);
    fire(AUDIO_DEVICE_EVENTS.devicesChanged, snapshot(["Mic", "USB Mic"], "Mic", "default_changed"));
    await vi.advanceTimersByTimeAsync(1_000);
    expect(stt.start).toHaveBeenCalledTimes(1);
    service.dispose();
  });

  it("does nothing outside the desktop app", async () => {
    (globalThis as { window?: { Config: { isApp: () => boolean } } }).window!.Config.isApp = () => false;
    const service = await started();
    expect(invokeMock).not.toHaveBeenCalled();
    expect(handlers.size).toBe(0);
  });
});
