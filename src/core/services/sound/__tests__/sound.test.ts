import i18n from "i18next";
import { afterEach, beforeAll, beforeEach, describe, expect, it, vi } from "vitest";

vi.mock("@tauri-apps/api/core", () => ({ invoke: vi.fn() }));

import { invoke } from "@tauri-apps/api/core";
import Service_Sound, { sanitizeVoiceClipOptions } from "../index";

const invokeMock = vi.mocked(invoke);

const flush = async (n = 10) => {
  for (let i = 0; i < n; i++) await Promise.resolve();
};

beforeAll(async () => {
  await i18n.init({
    lng: "en",
    // Same as the app (src/i18n.ts): React escapes, so i18next must not HTML-escape values.
    interpolation: { escapeValue: false },
    resources: {
      en: {
        translation: {
          toasts: {
            sound_playback_failed: "Could not play audio: {{error}}",
            sound_playback_failed_device: 'Could not play audio on "{{device}}": {{error}}',
          },
        },
      },
    },
  });
});

beforeEach(() => {
  invokeMock.mockReset();
  vi.stubGlobal("window", { ApiServer: { state: { audioOutputDevice: "App Output" } } });
});

afterEach(() => {
  vi.unstubAllGlobals();
});

describe("sanitizeVoiceClipOptions", () => {
  it("replaces NaN and non-positive rates (a 0 speed would never finish playing)", () => {
    expect(sanitizeVoiceClipOptions({ device_name: "", volume: Number.NaN, rate: 0 }, "")).toEqual({
      device_name: "",
      volume: 1,
      rate: 1,
    });
    expect(sanitizeVoiceClipOptions({ device_name: "X", volume: -1, rate: 100 }, "")).toEqual({
      device_name: "X",
      volume: 0,
      rate: 10,
    });
  });

  it("keeps volume 0 (mute) and falls back to the app's output device", () => {
    expect(sanitizeVoiceClipOptions({ device_name: "", volume: 0, rate: 1 }, "App Output")).toEqual({
      device_name: "App Output",
      volume: 0,
      rate: 1,
    });
  });
});

describe("Service_Sound voice clips", () => {
  it("plays clips one at a time in order, with the device fallback", async () => {
    const finish: (() => void)[] = [];
    invokeMock.mockImplementation(() => new Promise<void>((res) => finish.push(res)) as never);
    const sound = new Service_Sound();
    const first = sound.enqueueVoiceClip(new Uint8Array([1, 2]).buffer, { device_name: "", volume: 1, rate: 1 });
    const second = sound.enqueueVoiceClip(new Uint8Array([3]).buffer, { device_name: "Headphones", volume: 0.5, rate: 1 });

    await flush();
    expect(invokeMock).toHaveBeenCalledTimes(1);
    expect(invokeMock).toHaveBeenLastCalledWith("plugin:audio|play_async", {
      data: { data: [1, 2], device_name: "App Output", volume: 1, rate: 1 },
    });
    finish[0]();
    await first;
    await flush();
    expect(invokeMock).toHaveBeenCalledTimes(2);
    expect(invokeMock).toHaveBeenLastCalledWith("plugin:audio|play_async", {
      data: { data: [3], device_name: "Headphones", volume: 0.5, rate: 1 },
    });
    finish[1]();
    await second;
  });

  it("surfaces playback errors instead of swallowing them, and keeps playing the queue", async () => {
    invokeMock.mockRejectedValueOnce("Output device 'Gone' not found").mockResolvedValueOnce(undefined as never);
    const sound = new Service_Sound();
    const failed = sound.enqueueVoiceClip(new ArrayBuffer(4), { device_name: "Gone", volume: 1, rate: 1 });
    const next = sound.enqueueVoiceClip(new ArrayBuffer(4), { device_name: "", volume: 1, rate: 1 });
    await expect(failed).rejects.toThrow(`Could not play audio on "Gone": Output device 'Gone' not found`);
    await expect(next).resolves.toBeUndefined();
    expect(invokeMock).toHaveBeenCalledTimes(2);
  });

  it("clearVoiceClips drops clips that have not started (they resolve, not reject)", async () => {
    const finish: (() => void)[] = [];
    invokeMock.mockImplementation(() => new Promise<void>((res) => finish.push(res)) as never);
    const sound = new Service_Sound();
    const playing = sound.enqueueVoiceClip(new ArrayBuffer(1), { device_name: "", volume: 1, rate: 1 });
    const queued = sound.enqueueVoiceClip(new ArrayBuffer(1), { device_name: "", volume: 1, rate: 1 });
    await flush();
    expect(sound.pendingVoiceClips).toBe(1);
    sound.clearVoiceClips();
    await expect(queued).resolves.toBeUndefined();
    expect(sound.pendingVoiceClips).toBe(0);
    finish[0]();
    await playing;
    await flush();
    expect(invokeMock).toHaveBeenCalledTimes(1);
  });

  it("ignores empty clips", async () => {
    const sound = new Service_Sound();
    await sound.enqueueVoiceClip(new ArrayBuffer(0), { device_name: "", volume: 1, rate: 1 });
    expect(invokeMock).not.toHaveBeenCalled();
  });
});
