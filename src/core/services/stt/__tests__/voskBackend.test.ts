import { beforeAll, beforeEach, describe, expect, it, vi } from "vitest";

const h = vi.hoisted(() => {
  class FakeRecognizer {
    handlers = new Map<string, (msg: unknown) => void>();
    fed: { length: number; rate: number }[] = [];
    removed = false;
    constructor(public sampleRate: number) {
      h.recognizers.push(this);
    }
    on(event: string, cb: (msg: unknown) => void) {
      this.handlers.set(event, cb);
    }
    acceptWaveformFloat(buffer: Float32Array, rate: number) {
      this.fed.push({ length: buffer.length, rate });
    }
    remove() {
      this.removed = true;
    }
  }
  return {
    recognizers: [] as InstanceType<typeof FakeRecognizer>[],
    FakeRecognizer,
    terminated: 0,
    listeners: new Map<string, (event: { payload: unknown }) => void>(),
    invoke: null as unknown as ReturnType<typeof vi.fn>,
  };
});

vi.mock("@tauri-apps/api/core", () => ({
  invoke: (cmd: string, args?: unknown) => h.invoke(cmd, args),
  convertFileSrc: (p: string) => `asset://${p}`,
  isTauri: () => true,
}));
vi.mock("@tauri-apps/api/event", () => ({
  listen: vi.fn(async (event: string, handler: (event: { payload: unknown }) => void) => {
    h.listeners.set(event, handler);
    return () => h.listeners.delete(event);
  }),
}));
vi.mock("@tauri-apps/api/path", () => ({ appDataDir: async () => "C:/appdata", join: async (...p: string[]) => p.join("/") }));
vi.mock("@/utils/devLog", () => ({ devLog: () => {} }));
vi.mock("@/utils/rustAudioInput", () => ({ resolveRustInputDeviceForInvoke: (d: string) => (d ? d : null) }));
vi.mock("../loadVoskModel", () => ({
  loadVoskModel: async () => ({
    KaldiRecognizer: h.FakeRecognizer,
    terminate: () => {
      h.terminated++;
    },
  }),
}));

import { resampleLinear, toMonoFloat32 } from "../audioConvert";
import { Service_STT_Schema } from "../schema";
import { STT_VoskService, VOSK_SAMPLE_RATE, VoskTranscript } from "../services/vosk";
import type { ISTTReceiver } from "../types";
import { initSttTestI18n } from "./sttTestI18n";

function receiver() {
  return {
    onStart: vi.fn(),
    onStop: vi.fn(),
    onInterim: vi.fn(),
    onFinal: vi.fn(),
    onStatus: vi.fn(),
    isMuted: vi.fn(() => false),
  } satisfies ISTTReceiver;
}

beforeAll(async () => {
  await initSttTestI18n();
});

beforeEach(() => {
  h.recognizers.length = 0;
  h.terminated = 0;
  h.listeners.clear();
  h.invoke = vi.fn(async (cmd: string) => (cmd === "plugin:vosk-stt|download_vosk_model" ? "vosk-models/m.zip" : undefined));
});

describe("VoskTranscript", () => {
  it("publishes a partial once until it changes, and a result without words abandons it", () => {
    const interim = vi.fn();
    const final = vi.fn();
    const t = new VoskTranscript({ interim, final });
    t.partial("");
    t.partial("hello");
    t.partial("hello");
    t.partial(" hello ");
    t.partial("hello there");
    expect(interim.mock.calls).toEqual([["hello"], ["hello there"]]);
    t.result("hello there");
    expect(final).toHaveBeenCalledWith("hello there");
    t.partial("uh");
    t.result("");
    expect(interim).toHaveBeenLastCalledWith("");
    t.result(""); // nothing open any more
    expect(interim).toHaveBeenCalledTimes(4);
    t.partial("hello"); // a new utterance repeats the text: published again
    expect(interim).toHaveBeenLastCalledWith("hello");
  });
});

describe("audio conversion", () => {
  it("downmixes and resamples", () => {
    expect(Array.from(toMonoFloat32([1, 0, 0.5, 0.5], 2))).toEqual([0.5, 0.5]);
    const same = new Float32Array([1, 2, 3]);
    expect(resampleLinear(same, 16_000, 16_000)).toBe(same);
    const down = resampleLinear(new Float32Array(480).map((_, i) => i), 48_000, 16_000);
    expect(down.length).toBe(160);
    expect(down[1]).toBeCloseTo(3);
    const up = resampleLinear(new Float32Array([0, 1]), 8_000, 16_000);
    expect(Array.from(up)).toEqual([0, 0.5, 1, 1]);
  });
});

describe("STT_VoskService", () => {
  it("is fed 16 kHz mono by the capture hub through audio:chunk, and publishes deduplicated partials", async () => {
    const r = receiver();
    const svc = new STT_VoskService(r);
    await svc.start(Service_STT_Schema.parse({ vosk: { device: "Microphone (USB)" } }));
    expect(r.onStart).toHaveBeenCalledTimes(1);
    expect(h.invoke).toHaveBeenCalledWith("plugin:audio|start_audio_capture", { deviceName: "Microphone (USB)", sampleRate: 16_000 });
    const rec = h.recognizers[0];
    expect(rec.sampleRate).toBe(VOSK_SAMPLE_RATE);

    const chunk = h.listeners.get("audio:chunk")!;
    chunk({ payload: { samples: new Array(320).fill(0.1), sample_rate: 16_000, channels: 1 } });
    // A payload in another format (not what the hub sends for 16 kHz, but tolerated) is converted.
    chunk({ payload: { samples: new Array(960 * 2).fill(0.1), sample_rate: 48_000, channels: 2 } });
    chunk({ payload: { samples: [], sample_rate: 16_000, channels: 1 } });
    expect(rec.fed).toEqual([
      { length: 320, rate: 16_000 },
      { length: 320, rate: 16_000 },
    ]);

    const partial = rec.handlers.get("partialresult")!;
    partial({ result: { partial: "hi" } });
    partial({ result: { partial: "hi" } });
    rec.handlers.get("result")!({ result: { text: "hi there" } });
    expect(r.onInterim.mock.calls).toEqual([["hi"]]);
    expect(r.onFinal).toHaveBeenCalledWith("hi there");

    await svc.stop();
    expect(h.invoke).toHaveBeenCalledWith("plugin:audio|stop_audio_capture", undefined);
    expect(rec.removed).toBe(true);
    expect(h.terminated).toBe(1);
    expect(h.listeners.has("audio:chunk")).toBe(false);
  });

  it("a failed start stops with a readable message", async () => {
    h.invoke = vi.fn(async (cmd: string) => {
      if (cmd === "plugin:vosk-stt|download_vosk_model") throw "HTTP 404 for the model URL";
      return undefined;
    });
    const r = receiver();
    await new STT_VoskService(r).start(Service_STT_Schema.parse({}));
    expect(r.onStop).toHaveBeenCalledWith("Vosk could not start: HTTP 404 for the model URL");
    expect(r.onStart).not.toHaveBeenCalled();
  });
});
