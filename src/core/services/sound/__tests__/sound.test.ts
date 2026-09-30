import i18n from "i18next";
import { afterEach, beforeAll, beforeEach, describe, expect, it, vi } from "vitest";

vi.mock("@tauri-apps/api/core", () => ({ invoke: vi.fn() }));
vi.mock("@tauri-apps/api/event", () => ({ listen: vi.fn(async () => () => {}) }));

import type { InvokeArgs, InvokeOptions } from "@tauri-apps/api/core";
import { TTS_SINK_HEADERS } from "@/types";
import { deviceFamily, isDefaultOutputName, matchOutputDevice, migrateOutputDeviceName } from "../devices";
import Service_Sound, { sanitizeVoiceClipOptions } from "../index";
import { decodeAudioClip, decodeWav, isWav, pcm16leToFloat32 } from "../pcm";
import { TtsSinkClient, isFinalSinkState, isSinkCancelled } from "../sink";
import { FakeSinkTransport, flush, makeWav } from "./fakeSink";

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
      route: "device",
    });
    expect(sanitizeVoiceClipOptions({ device_name: "X", volume: -1, rate: 100, route: "both" }, "")).toEqual({
      device_name: "X",
      volume: 0,
      rate: 10,
      route: "both",
    });
  });

  it("keeps volume 0 (mute), caps the gain at 2 and falls back to the app's output device", () => {
    expect(sanitizeVoiceClipOptions({ device_name: "", volume: 0, rate: 1 }, "App Output")).toEqual({
      device_name: "App Output",
      volume: 0,
      rate: 1,
      route: "device",
    });
    expect(sanitizeVoiceClipOptions({ device_name: " Headphones ", volume: 4, rate: 1 }, "App Output").volume).toBe(2);
  });
});

describe("WAV and PCM decoding", () => {
  it("decodes integer PCM of every width, float, and WAVE_FORMAT_EXTENSIBLE", () => {
    const s16 = decodeWav(makeWav([16_384, -16_384, 0, 32_767], { rate: 24_000, channels: 2 }));
    expect(s16.sampleRate).toBe(24_000);
    expect(s16.channels).toBe(2);
    expect(Array.from(s16.samples)).toEqual([0.5, -0.5, 0, 32_767 / 32_768]);
    expect(Array.from(decodeWav(makeWav([128, 255, 0], { bits: 8 })).samples)).toEqual([0, 127 / 128, -1]);
    expect(Array.from(decodeWav(makeWav([4_194_304, -8_388_608], { bits: 24 })).samples)).toEqual([0.5, -1]);
    expect(Array.from(decodeWav(makeWav([1_073_741_824], { bits: 32 })).samples)).toEqual([0.5]);
    expect(Array.from(decodeWav(makeWav([0.25, -1], { bits: 32, float: true })).samples)).toEqual([0.25, -1]);
    expect(Array.from(decodeWav(makeWav([0.5], { bits: 64, float: true })).samples)).toEqual([0.5]);
    expect(Array.from(decodeWav(makeWav([16_384], { extensible: true })).samples)).toEqual([0.5]);
    // Chunks before `data` (odd-sized, word-aligned) are skipped.
    expect(Array.from(decodeWav(makeWav([-16_384], { extraChunk: true })).samples)).toEqual([-0.5]);
  });

  it("reads a streamed WAV whose data size runs past the end", () => {
    const wav = makeWav([16_384, 16_384]);
    const dv = new DataView(wav);
    dv.setUint32(40, 0xffffffff, true);
    expect(Array.from(decodeWav(new Uint8Array(wav)).samples)).toEqual([0.5, 0.5]);
  });

  it("refuses what it cannot decode with a readable error", () => {
    expect(isWav(new Uint8Array([1, 2, 3]))).toBe(false);
    expect(() => decodeWav(new Uint8Array(16))).toThrow("not a WAV file");
    const adpcm = makeWav([1, 2]);
    new DataView(adpcm).setUint16(20, 2, true);
    expect(() => decodeWav(adpcm)).toThrow("format 2 is not supported");
    const noData = makeWav([]).slice(0, 36);
    new DataView(noData).setUint32(4, 28, true);
    expect(() => decodeWav(noData)).toThrow("no data chunk");
  });

  it("converts 16-bit PCM split at odd byte boundaries exactly", () => {
    const whole = new Uint8Array(new Int16Array([1_000, -32_768, 32_767, -1]).buffer);
    const expected = Array.from(pcm16leToFloat32(whole).samples);
    const parts = [whole.slice(0, 3), whole.slice(3, 4), whole.slice(4, 7), whole.slice(7)];
    let carry: number | null = null;
    const got: number[] = [];
    for (const part of parts) {
      const r = pcm16leToFloat32(part, carry);
      carry = r.carry;
      got.push(...r.samples);
    }
    expect(carry).toBeNull();
    expect(got).toEqual(expected);
    expect(expected).toEqual([1_000 / 32_768, -1, 32_767 / 32_768, -1 / 32_768]);
  });

  it("compressed clips need the webview decoder", async () => {
    await expect(decodeAudioClip(new Uint8Array([0xff, 0xfb, 0x90, 0x00]))).rejects.toThrow("cannot be decoded here");
    const pcm = await decodeAudioClip(makeWav([16_384], { rate: 16_000 }));
    expect(pcm.sampleRate).toBe(16_000);
  });
});

describe("output device names", () => {
  const names = [
    "Speakers (Realtek(R) Audio)",
    "CABLE Input (VB-Audio Virtual Cable)",
    "2- Headphones (USB Audio)",
    "Headset Earphone (HyperX Cloud)",
    "Headset Earphone (Other)",
  ];

  it("matches like the Rust sink: exact, case, renumbering, truncated SAPI names; never guesses", () => {
    expect(matchOutputDevice(" Speakers (Realtek(R) Audio) ", names)).toBe(0);
    expect(matchOutputDevice("speakers (realtek(r) audio)", names)).toBe(0);
    expect(matchOutputDevice("Headphones (USB Audio)", names)).toBe(2);
    expect(matchOutputDevice("CABLE Input (VB-Audio Virtual C", names)).toBe(1);
    expect(matchOutputDevice("Headset Earphone (", names)).toBeNull();
    expect(matchOutputDevice("CABLE", names)).toBeNull();
    expect(matchOutputDevice("Missing device", names)).toBeNull();
    expect(matchOutputDevice("", names)).toBeNull();
    expect(deviceFamily("3- Speakers (2- USB)")).toBe("speakers (usb)");
  });

  it("migrates old SAPI names to cpal names and leaves the rest alone", () => {
    expect(migrateOutputDeviceName("CABLE Input (VB-Audio Virtual C", names)).toBe("CABLE Input (VB-Audio Virtual Cable)");
    expect(migrateOutputDeviceName("Speakers (Realtek(R) Audio)", names)).toBe("Speakers (Realtek(R) Audio)");
    expect(migrateOutputDeviceName("Unplugged DAC", names)).toBe("Unplugged DAC");
    expect(migrateOutputDeviceName("", names)).toBe("");
    expect(isDefaultOutputName(" Default ")).toBe(true);
    expect(isDefaultOutputName("Speakers")).toBe(false);
  });
});

describe("TtsSinkClient", () => {
  it("pushes PCM as a binary f32 body with the x-sigil headers, split into bounded pushes in order", async () => {
    const fake = new FakeSinkTransport();
    const client = new TtsSinkClient(fake);
    await client.begin("u1", { route: "device", device: "", volume: 1 });
    const long = new Float32Array(48_000 * 2 * 31 + 2).map((_, i) => (i % 100) / 100);
    await client.push("u1", { samples: long, sampleRate: 48_000, channels: 2 });
    const pushes = fake.calls.filter((c) => c.cmd.endsWith("push_pcm"));
    expect(pushes.length).toBe(2);
    expect(pushes[0].args).toBeInstanceOf(Uint8Array);
    expect(pushes[0].headers).toEqual({
      [TTS_SINK_HEADERS.utterance]: "u1",
      [TTS_SINK_HEADERS.sampleRate]: "48000",
      [TTS_SINK_HEADERS.channels]: "2",
    });
    const got = fake.utterances.get("u1")!.pushes.map((p) => p.samples);
    expect(got.reduce((n, p) => n + p.length, 0)).toBe(long.length);
    expect(got[0].length % 2).toBe(0);
    let at = 0;
    let same = true;
    for (const part of got) {
      for (let i = 0; i < part.length; i++) same &&= part[i] === long[at + i];
      at += part.length;
    }
    expect(same).toBe(true);
    // Nothing to send: no IPC.
    await client.push("u1", { samples: new Float32Array(1), sampleRate: 48_000, channels: 2 });
    expect(fake.calls.filter((c) => c.cmd.endsWith("push_pcm")).length).toBe(2);
  });

  it("fans utterance events out through one listener", async () => {
    const fake = new FakeSinkTransport();
    const client = new TtsSinkClient(fake);
    const a = vi.fn();
    const b = vi.fn();
    const offA = client.onUtterance(a);
    client.onUtterance(b);
    await client.ready();
    expect(fake.listenCalls).toBe(1);
    fake.play("x");
    expect(a).toHaveBeenCalledTimes(1);
    expect(b).toHaveBeenCalledTimes(1);
    offA();
    fake.done("x");
    expect(a).toHaveBeenCalledTimes(1);
    expect(b).toHaveBeenCalledTimes(2);
    expect(isSinkCancelled("cancelled")).toBe(true);
    expect(isSinkCancelled(new Error("cancelled"))).toBe(true);
    expect(isSinkCancelled("audio output: gone")).toBe(false);
    expect(["done", "cancelled", "error"].every((s) => isFinalSinkState(s as never))).toBe(true);
    expect(isFinalSinkState("playing")).toBe(false);
  });

  it("flushes what an earlier webview session left in the sink on the first ready(), once", async () => {
    const fake = new FakeSinkTransport();
    const client = new TtsSinkClient(fake);
    client.onUtterance(() => {});
    await client.ready();
    await client.ready();
    await client.begin("u1", { route: "device", device: "", volume: 1 });
    await client.ready();
    expect(fake.commands()).toEqual(["flush_sink", "begin_utterance"]);
    expect(fake.utterances.get("u1")?.state).toBe("queued");
  });
});

describe("Service_Sound voice clips (through the TTS sink)", () => {
  function setup() {
    const fake = new FakeSinkTransport();
    const sound = new Service_Sound(new TtsSinkClient(fake));
    return { fake, sound };
  }

  it("decodes a WAV clip, plays it on the device fallback and resolves when it finished playing", async () => {
    const { fake, sound } = setup();
    let settled = false;
    const clip = sound
      .enqueueVoiceClip(makeWav([16_384, -16_384], { rate: 24_000 }), { device_name: "", volume: 0.5, rate: 1 })
      .then(() => (settled = true));
    const second = sound.enqueueVoiceClip(makeWav([8_192]), { device_name: "Headphones", volume: 1, rate: 2 });
    await flush(40);
    expect(fake.order.length).toBe(2);
    const [first, next] = fake.order;
    expect(fake.utterances.get(first)?.target).toEqual({ route: "device", device: "App Output", volume: 0.5 });
    expect(fake.audio(first)).toEqual([0.5, -0.5]);
    expect(fake.utterances.get(first)?.pushes[0].sampleRate).toBe(24_000);
    expect(fake.utterances.get(first)?.ended).toBe(true);
    // Speed 2: the samples are played at twice the rate.
    expect(fake.utterances.get(next)?.pushes[0].sampleRate).toBe(48_000);
    expect(fake.utterances.get(next)?.target.device).toBe("Headphones");
    expect(settled).toBe(false);
    fake.done(first);
    await clip;
    expect(settled).toBe(true);
    fake.done(next);
    await second;
    expect(sound.pendingVoiceClips).toBe(0);
    // PCM directly.
    const pcm = sound.enqueuePcm({ samples: new Float32Array([0.1, 0.2]), sampleRate: 48_000, channels: 1 }, {
      device_name: "",
      volume: 1,
      rate: 1,
    });
    await flush(40);
    const third = fake.order[2];
    expect(fake.audio(third).map((v) => Math.round(v * 10) / 10)).toEqual([0.1, 0.2]);
    fake.done(third);
    await pcm;
  });

  it("surfaces playback errors (with the device) and keeps playing the queue", async () => {
    const { fake, sound } = setup();
    fake.failNext.set("plugin:tts-sink|begin_utterance", "audio output: Output device 'Gone' not found");
    const failed = sound.enqueueVoiceClip(makeWav([1]), { device_name: "Gone", volume: 1, rate: 1 });
    const next = sound.enqueueVoiceClip(makeWav([1]), { device_name: "", volume: 1, rate: 1 });
    await expect(failed).rejects.toThrow(`Could not play audio on "Gone": audio output: Output device 'Gone' not found`);
    await flush(40);
    fake.done(fake.order[0]);
    await expect(next).resolves.toBeUndefined();
    // An error while playing (device unplugged) rejects too.
    const lost = sound.enqueueVoiceClip(makeWav([1]), { device_name: "USB", volume: 1, rate: 1 });
    await flush(40);
    fake.fail(fake.order[1], "audio output: the device was removed");
    await expect(lost).rejects.toThrow(`Could not play audio on "USB": audio output: the device was removed`);
    // Undecodable audio.
    await expect(sound.enqueueVoiceClip(new Uint8Array([9, 9, 9]).buffer, { device_name: "", volume: 1, rate: 1 })).rejects.toThrow(
      "Could not play audio on \"App Output\": this audio format cannot be decoded here",
    );
  });

  it("clearVoiceClips stops the clip playing, queued ones and those still decoding (they resolve)", async () => {
    const { fake, sound } = setup();
    const playing = sound.enqueueVoiceClip(makeWav([1, 2]), { device_name: "", volume: 1, rate: 1 });
    const queued = sound.enqueueVoiceClip(makeWav([3]), { device_name: "", volume: 1, rate: 1 });
    await flush(40);
    expect(sound.pendingVoiceClips).toBe(2);
    const decoding = sound.enqueueVoiceClip(makeWav([4]), { device_name: "", volume: 1, rate: 1 });
    sound.clearVoiceClips();
    await expect(playing).resolves.toBeUndefined();
    await expect(queued).resolves.toBeUndefined();
    await expect(decoding).resolves.toBeUndefined();
    await flush(40);
    expect(sound.pendingVoiceClips).toBe(0);
    expect(fake.order.length).toBe(2);
    expect(fake.commands().filter((c) => c === "cancel_utterance").length).toBe(2);
  });

  it("clearVoiceClips during a begin still in flight cancels that clip once it exists (review finding)", async () => {
    let release!: () => void;
    /** Holds `begin_utterance` until released, so the stop's cancel reaches the sink before the begin. */
    class GatedBegin extends FakeSinkTransport {
      gate = new Promise<void>((resolve) => (release = resolve));
      async invoke<T>(cmd: string, args?: InvokeArgs, options?: InvokeOptions): Promise<T> {
        if (cmd === "plugin:tts-sink|begin_utterance") await this.gate;
        return super.invoke<T>(cmd, args, options);
      }
    }
    const fake = new GatedBegin();
    const sound = new Service_Sound(new TtsSinkClient(fake));
    const clip = sound.enqueuePcm({ samples: new Float32Array([0.5]), sampleRate: 48_000, channels: 1 }, {
      device_name: "",
      volume: 1,
      rate: 1,
    });
    await flush(40);
    // The begin is in flight: the stop's cancel finds no such utterance in the sink yet (a no-op there).
    sound.clearVoiceClips();
    await expect(clip).resolves.toBeUndefined();
    release();
    await flush(40);
    const [id] = fake.order;
    expect(fake.utterances.get(id)?.state).toBe("cancelled");
    expect(fake.calls.filter((c) => c.cmd.endsWith("push_pcm"))).toEqual([]);
    expect(fake.commands().filter((c) => c === "cancel_utterance").length).toBe(2);
  });

  it("ignores empty clips", async () => {
    const { fake, sound } = setup();
    await sound.enqueueVoiceClip(new ArrayBuffer(0), { device_name: "", volume: 1, rate: 1 });
    await sound.enqueuePcm({ samples: new Float32Array(0), sampleRate: 48_000, channels: 1 }, { device_name: "", volume: 1, rate: 1 });
    expect(fake.calls.length).toBe(0);
  });
});
