import i18n from "i18next";
import { proxy } from "valtio";
import { afterEach, beforeAll, beforeEach, describe, expect, it, vi } from "vitest";

vi.mock("@tauri-apps/api/core", () => ({ invoke: vi.fn(), isTauri: () => false }));
vi.mock("@/utils/devLog", () => ({ devLog: () => {} }));

/**
 * Azure SDK stand-in: records SSML and lets the test deliver results. `connect` decides how the
 * connection check in start() ends; in the webview a rejected key or unknown region fails the
 * WebSocket handshake with close code 1006, which the SDK reports as this string.
 */
const azure = vi.hoisted(() => ({
  instances: [] as { ssml: string[]; cb: ((r: unknown) => void)[]; err: ((e: string) => void)[]; closed: boolean }[],
  connect: "ok" as "ok" | "fail" | "pending",
  connectError: "",
  connections: [] as { cb: () => void; err: (e: string) => void }[],
}));
vi.mock("microsoft-cognitiveservices-speech-sdk", async (importOriginal) => {
  const actual = await importOriginal<typeof import("microsoft-cognitiveservices-speech-sdk")>();
  const FakeConnection = {
    fromSynthesizer: () => ({
      openConnection(cb: () => void, err: (e: string) => void) {
        azure.connections.push({ cb, err });
        // The SDK settles through a promise, never synchronously.
        if (azure.connect === "ok") void Promise.resolve().then(cb);
        else if (azure.connect === "fail") void Promise.resolve().then(() => err(azure.connectError));
      },
      close() {},
    }),
  };
  class FakeSynthesizer {
    rec = { ssml: [] as string[], cb: [] as ((r: unknown) => void)[], err: [] as ((e: string) => void)[], closed: false };
    constructor() {
      azure.instances.push(this.rec);
    }
    speakSsmlAsync(ssml: string, cb: (r: unknown) => void, err: (e: string) => void) {
      this.rec.ssml.push(ssml);
      this.rec.cb.push(cb);
      this.rec.err.push(err);
    }
    close() {
      this.rec.closed = true;
    }
  }
  return {
    ...actual,
    SpeechConfig: { fromSubscription: () => ({}) },
    SpeechSynthesizer: FakeSynthesizer,
    Connection: FakeConnection,
    CancellationDetails: {
      fromResult: (r: { errorDetails: string; code: number }) => ({ errorDetails: r.errorDetails, ErrorCode: r.code }),
    },
  };
});

import { invoke } from "@tauri-apps/api/core";
import { CancellationErrorCode, ResultReason } from "microsoft-cognitiveservices-speech-sdk";
import { TTS_BACKENDS } from "../index";
import { Service_TTS_Schema, TTS_Backends, TTS_State } from "../schema";
import { AZURE_CONNECT_TIMEOUT_MS, TTS_AzureService } from "../services/azure";
import { TTS_FishSpeechService } from "../services/fishspeech";
import { TTS_KokoroService } from "../services/kokoro";
import { TTS_NativeService } from "../services/native";
import { TTS_UberduckService } from "../services/uberduck";
import { TTS_VoicevoxService, fetchVoicevoxSpeakers } from "../services/voicevox";
import { TTS_WindowsService } from "../services/windows";
import { ITTSReceiver, TTS_OPTIONS_MISSING } from "../types";

const invokeMock = vi.mocked(invoke);

function receiver() {
  return { onStart: vi.fn(), onStop: vi.fn(), onError: vi.fn() } satisfies ITTSReceiver;
}

const flush = async (n = 10) => {
  for (let i = 0; i < n; i++) await Promise.resolve();
};

/** Minimal Web Audio for the clip queue. */
class FakeAudioContext {
  static started: number[] = [];
  state = "running";
  currentTime = 0;
  destination = {};
  async decodeAudioData(data: ArrayBuffer) {
    return { duration: data.byteLength / 1000 };
  }
  createBufferSource() {
    return {
      buffer: null,
      onended: null,
      connect() {},
      disconnect() {},
      start: (when: number) => FakeAudioContext.started.push(when),
      stop() {},
    };
  }
  async resume() {}
  async close() {
    this.state = "closed";
  }
}

let data: TTS_State;
let sound: { enqueueVoiceClip: ReturnType<typeof vi.fn>; clearVoiceClips: ReturnType<typeof vi.fn> };

function installWindow(extra: Record<string, unknown> = {}) {
  data = proxy(Service_TTS_Schema.parse({}));
  sound = { enqueueVoiceClip: vi.fn(async () => {}), clearVoiceClips: vi.fn() };
  vi.stubGlobal("window", {
    ApiServer: {
      state: { services: { tts: { data } }, audioOutputDevice: "Speakers (App default)" },
      sound,
      voiceChanger: { state: { isRunning: false } },
    },
    ...extra,
  });
}

beforeAll(async () => {
  // Same interpolation setting as src/i18n.ts (URLs and hosts must not be HTML-escaped).
  await i18n.init({ lng: "en", resources: { en: { translation: {} } }, interpolation: { escapeValue: false } });
});

beforeEach(() => {
  invokeMock.mockReset();
  azure.instances = [];
  azure.connect = "ok";
  azure.connectError = "";
  azure.connections = [];
  FakeAudioContext.started = [];
  vi.stubGlobal("AudioContext", FakeAudioContext);
  installWindow();
});

afterEach(() => {
  vi.unstubAllGlobals();
});

describe("every backend", () => {
  it.each(Object.values(TTS_Backends))("%s: stop() reports onStop() once, without an error", (backend) => {
    const r = receiver();
    const instance = new TTS_BACKENDS[backend](r);
    instance.stop();
    expect(r.onStop).toHaveBeenCalledTimes(1);
    expect(r.onStop).toHaveBeenCalledWith();
    // dispose after stop must be safe and silent.
    instance.dispose();
    expect(r.onStop).toHaveBeenCalledTimes(1);
  });
});

describe("HTTP clip backends (Kokoro as representative)", () => {
  it("passes the endpoint per call and plays clips in order", async () => {
    data.kokoro.endpoint = " http://localhost:8880 ";
    data.kokoro.speed = "abc";
    invokeMock.mockImplementation(async (cmd, args) => {
      if (cmd.endsWith("check_kokoro_availability")) return true;
      if (cmd.endsWith("kokoro_speak")) return Array.from({ length: (args as { text: string }).text.length * 100 }, () => 0);
      return undefined;
    });
    const r = receiver();
    const kokoro = new TTS_KokoroService(r);
    await kokoro.start(data);
    expect(r.onStart).toHaveBeenCalledTimes(1);
    expect(invokeMock).toHaveBeenCalledWith("plugin:kokoro-tts|check_kokoro_availability", { endpoint: "http://localhost:8880" });

    kokoro.play("four");
    kokoro.play("two");
    await flush(30);
    expect(invokeMock).toHaveBeenCalledWith("plugin:kokoro-tts|kokoro_speak", {
      text: "four",
      voice: "af_bella",
      speed: 1, // junk speed falls back instead of NaN
      endpoint: "http://localhost:8880",
    });
    // 0.4 s clip, then the next clip right after it: no overlap.
    expect(FakeAudioContext.started).toEqual([0, 0.4]);
  });

  it("stop during connect wins over the late result", async () => {
    let answer!: (v: boolean) => void;
    invokeMock.mockImplementation(async (cmd) => {
      if (cmd.endsWith("check_kokoro_availability")) return new Promise<boolean>((res) => (answer = res));
      return undefined;
    });
    const r = receiver();
    const kokoro = new TTS_KokoroService(r);
    const starting = kokoro.start(data);
    await flush();
    kokoro.stop();
    answer(true);
    await starting;
    expect(r.onStart).not.toHaveBeenCalled();
    expect(r.onStop).toHaveBeenCalledTimes(1);
  });

  it("a failed line is reported with onError and the backend keeps running", async () => {
    invokeMock.mockImplementation(async (cmd) => {
      if (cmd.endsWith("check_kokoro_availability")) return true;
      if (cmd.endsWith("kokoro_speak")) throw "Kokoro synthesis: HTTP 500";
      return undefined;
    });
    const r = receiver();
    const kokoro = new TTS_KokoroService(r);
    await kokoro.start(data);
    kokoro.play("hello");
    await flush(30);
    expect(r.onError).toHaveBeenCalledWith("Kokoro: Kokoro synthesis: HTTP 500");
    expect(r.onStop).not.toHaveBeenCalled();
    expect(kokoro.running).toBe(true);
  });

  it("audio synthesized after Stop is never played", async () => {
    let finish!: (v: number[]) => void;
    invokeMock.mockImplementation(async (cmd) => {
      if (cmd.endsWith("check_kokoro_availability")) return true;
      if (cmd.endsWith("kokoro_speak")) return new Promise<number[]>((res) => (finish = res));
      return undefined;
    });
    const r = receiver();
    const kokoro = new TTS_KokoroService(r);
    await kokoro.start(data);
    kokoro.play("hello");
    await flush(30);
    kokoro.stop();
    finish(new Array(500).fill(0));
    await flush(30);
    expect(FakeAudioContext.started).toEqual([]);
    expect(r.onError).not.toHaveBeenCalled();
  });

  it("getVoices rejects instead of hiding a failure behind an empty list", async () => {
    invokeMock.mockRejectedValue("Kokoro voice list: connection refused");
    await expect(TTS_KokoroService.getVoices("http://localhost:8880")).rejects.toBe("Kokoro voice list: connection refused");
  });
});

describe("Fish Speech", () => {
  it("passes the API key and model through, and blanks as null", async () => {
    invokeMock.mockImplementation(async (cmd) => (cmd.endsWith("check_fish_availability") ? true : [1, 2, 3]));
    data.fishSpeech.endpoint = "https://api.fish.audio";
    data.fishSpeech.apiKey = " sk-test ";
    data.fishSpeech.model = "s2.1-pro-free";
    data.fishSpeech.referenceId = "default";
    const r = receiver();
    const fish = new TTS_FishSpeechService(r);
    await fish.start(data);
    expect(invokeMock).toHaveBeenCalledWith("plugin:fish-speech|check_fish_availability", {
      endpoint: "https://api.fish.audio",
      apiKey: "sk-test",
    });
    fish.play("hello");
    await flush(30);
    expect(invokeMock).toHaveBeenCalledWith("plugin:fish-speech|fish_speak", {
      text: "hello",
      referenceId: "default",
      endpoint: "https://api.fish.audio",
      apiKey: "sk-test",
      model: "s2.1-pro-free",
    });

    await TTS_FishSpeechService.getVoices("http://localhost:8080", "  ");
    expect(invokeMock).toHaveBeenLastCalledWith("plugin:fish-speech|get_fish_voices", {
      endpoint: "http://localhost:8080",
      apiKey: null,
    });
  });

  it("a rejected key stops the backend with the server's reason", async () => {
    invokeMock.mockImplementation(async (cmd) => {
      if (cmd.endsWith("check_fish_availability")) throw "Fish Speech rejected the API key (HTTP 401 Unauthorized).";
      return undefined;
    });
    const r = receiver();
    await new TTS_FishSpeechService(r).start(data);
    expect(r.onStop).toHaveBeenCalledWith("Fish Speech rejected the API key (HTTP 401 Unauthorized).");
  });
});

describe("Azure", () => {
  /** What SDK 1.48 reports in the webview for a rejected key or an unknown region (see SynthesisAdapterBase). */
  const HANDSHAKE_1006 = `Unable to contact server. StatusCode: 1006,
                    wss://westeurope.tts.speech.microsoft.com/cognitiveservices/websocket/v1 Reason: `;

  function setupAzure() {
    Object.assign(data.azure, { key: "k", location: "westeurope", voice: "zh-CN-XiaoyiNeural,", device: "" });
    const r = receiver();
    return { r, service: new TTS_AzureService(r) };
  }

  async function startAzure() {
    const { r, service } = setupAzure();
    await service.start(data);
    return { r, service, synth: azure.instances[0] };
  }

  it("only needs key, location and voice (device is optional)", async () => {
    const { r } = await startAzure();
    expect(r.onStart).toHaveBeenCalled();

    installWindow();
    Object.assign(data.azure, { key: "", location: "x", voice: "en-US-JennyNeural" });
    const r2 = receiver();
    await new TTS_AzureService(r2).start(data);
    expect(r2.onStop).toHaveBeenCalledWith(TTS_OPTIONS_MISSING);
  });

  it("reports onStart only once the connection with this key and region is open", async () => {
    azure.connect = "pending";
    const { r, service } = setupAzure();
    const started = service.start(data);
    await flush();
    expect(azure.connections).toHaveLength(1);
    expect(r.onStart).not.toHaveBeenCalled();
    azure.connections[0].cb();
    await started;
    expect(r.onStart).toHaveBeenCalledTimes(1);
    expect(r.onStop).not.toHaveBeenCalled();
  });

  it("a rejected key or unknown region fails Start instead of every line", async () => {
    i18n.addResource("en", "translation", "tts.azure_connect_failed", "bad key or region {{region}} | {{error}}");
    azure.connect = "fail";
    azure.connectError = HANDSHAKE_1006;
    const { r, synth } = await startAzure();
    expect(r.onStart).not.toHaveBeenCalled();
    expect(r.onStop).toHaveBeenCalledTimes(1);
    expect(r.onStop).toHaveBeenCalledWith(
      "Azure: bad key or region westeurope | Unable to contact server. StatusCode: 1006, " +
      "wss://westeurope.tts.speech.microsoft.com/cognitiveservices/websocket/v1 Reason:",
    );
    expect(synth.closed).toBe(true);
  });

  it("a connection that never opens times out", async () => {
    vi.useFakeTimers();
    try {
      azure.connect = "pending";
      const { r, service } = setupAzure();
      const started = service.start(data);
      await vi.advanceTimersByTimeAsync(AZURE_CONNECT_TIMEOUT_MS - 1);
      expect(r.onStop).not.toHaveBeenCalled();
      await vi.advanceTimersByTimeAsync(1);
      await started;
      expect(r.onStart).not.toHaveBeenCalled();
      expect(r.onStop).toHaveBeenCalledWith("Azure: tts.azure_connect_timeout");
      expect(azure.instances[0].closed).toBe(true);
    } finally {
      vi.useRealTimers();
    }
  });

  it("stop during the connection check wins over a late connect", async () => {
    azure.connect = "pending";
    const { r, service } = setupAzure();
    const started = service.start(data);
    await flush();
    service.stop();
    await started;
    azure.connections[0].cb();
    await flush();
    expect(r.onStart).not.toHaveBeenCalled();
    expect(r.onStop).toHaveBeenCalledTimes(1);
    expect(r.onStop).toHaveBeenCalledWith();
    expect(azure.instances[0].closed).toBe(true);
  });

  it("speaks text containing '&' with escaped SSML in the voice's language", async () => {
    const { service, synth } = await startAzure();
    service.play("Tom & Jerry <3");
    expect(synth.ssml[0]).toContain("Tom &amp; Jerry &lt;3");
    expect(synth.ssml[0]).toContain('xml:lang="zh-CN"');
    expect(synth.ssml[0]).toContain('<voice name="zh-CN-XiaoyiNeural">');
  });

  it("plays completed audio through Service_Sound with sane options", async () => {
    const { service, synth } = await startAzure();
    data.azure.volume = "NaN";
    service.play("hi");
    const audio = new ArrayBuffer(8);
    synth.cb[0]({ reason: ResultReason.SynthesizingAudioCompleted, audioData: audio });
    expect(sound.enqueueVoiceClip).toHaveBeenCalledWith(audio, { volume: 1, rate: 1, device_name: "" });
  });

  it("after a verified connection, a failed line (dropped connection, throttling) only fails that line", async () => {
    const { r, service, synth } = await startAzure();
    service.play("one");
    service.play("two");
    synth.cb[0]({ reason: ResultReason.Canceled, errorDetails: HANDSHAKE_1006, code: CancellationErrorCode.ConnectionFailure });
    expect(r.onError).toHaveBeenCalledWith(expect.stringContaining("Azure: Unable to contact server. StatusCode: 1006"));
    synth.cb[1]({ reason: ResultReason.Canceled, errorDetails: "Too many requests", code: CancellationErrorCode.TooManyRequests });
    expect(r.onError).toHaveBeenCalledWith("Azure: Too many requests");
    expect(r.onStop).not.toHaveBeenCalled();
    expect(synth.closed).toBe(false);
  });

  it("a cancellation marked as an auth failure stops the service", async () => {
    // SDK 1.48 raises this only for recognition; kept in case synthesis starts reporting it.
    const { r, service, synth } = await startAzure();
    service.play("one");
    synth.cb[0]({ reason: ResultReason.Canceled, errorDetails: "", code: CancellationErrorCode.AuthenticationFailure });
    expect(r.onStop).toHaveBeenCalledWith("Azure: AuthenticationFailure");
    expect(synth.closed).toBe(true);
  });

  it("stop closes the synthesizer, drops queued clips and ignores late results", async () => {
    const { r, service, synth } = await startAzure();
    service.play("late");
    service.stop();
    expect(synth.closed).toBe(true);
    expect(sound.clearVoiceClips).toHaveBeenCalled();
    synth.cb[0]({ reason: ResultReason.SynthesizingAudioCompleted, audioData: new ArrayBuffer(8) });
    expect(sound.enqueueVoiceClip).not.toHaveBeenCalled();
    expect(r.onStop).toHaveBeenCalledTimes(1);
  });
});

describe("VOICEVOX / AivisSpeech", () => {
  function mockEngine() {
    const bodies: unknown[] = [];
    const urls: string[] = [];
    const fetchMock = vi.fn(async (url: string, init?: RequestInit) => {
      urls.push(url);
      if (url.endsWith("/version")) return new Response('"1.0.0"');
      if (url.includes("/audio_query")) return new Response(JSON.stringify({ speedScale: 1, pitchScale: 0.1, accent_phrases: [] }));
      if (url.includes("/synthesis")) {
        bodies.push(JSON.parse(String(init?.body)));
        return new Response(new Uint8Array([1, 2, 3, 4]));
      }
      return new Response("", { status: 404 });
    });
    vi.stubGlobal("fetch", fetchMock);
    return { bodies, urls };
  }

  it("connects to the AivisSpeech preset and keeps 0 values in the query", async () => {
    const { bodies, urls } = mockEngine();
    Object.assign(data.voicevox, {
      hostPreset: "aivisspeech",
      host: "http://localhost:10101/",
      speaker: "888753760",
      pitchScale: "0",
      intonationScale: "0",
      volumeScale: "0",
      device: "Headphones",
    });
    const r = receiver();
    const vv = new TTS_VoicevoxService(r);
    await vv.start(data);
    expect(r.onStart).toHaveBeenCalled();
    expect(urls[0]).toBe("http://localhost:10101/version");

    vv.play("こんにちは");
    await flush(40);
    expect(urls[1]).toBe(`http://localhost:10101/audio_query?text=${encodeURIComponent("こんにちは")}&speaker=888753760`);
    expect(bodies[0]).toMatchObject({ pitchScale: 0, intonationScale: 0, volumeScale: 0, speedScale: 1 });
    expect(sound.enqueueVoiceClip).toHaveBeenCalledWith(expect.any(ArrayBuffer), { volume: 1, rate: 1, device_name: "Headphones" });
  });

  it("refuses a LAN http host with an explanation instead of a CSP failure", async () => {
    const fetchMock = vi.fn();
    vi.stubGlobal("fetch", fetchMock);
    data.voicevox.host = "http://192.168.1.20:50021";
    const r = receiver();
    await new TTS_VoicevoxService(r).start(data);
    expect(fetchMock).not.toHaveBeenCalled();
    expect(r.onStop).toHaveBeenCalledWith("tts.voicevox_host_error_remote_http");
  });

  it("reports a stopped engine as not running", async () => {
    vi.stubGlobal("fetch", vi.fn(async () => {
      throw new TypeError("Failed to fetch");
    }));
    const r = receiver();
    await new TTS_VoicevoxService(r).start(data);
    expect(r.onStop).toHaveBeenCalledWith("tts.voicevox_not_running");
  });

  it("stop aborts synthesis in flight and drops queued clips", async () => {
    let aborted = false;
    vi.stubGlobal("fetch", vi.fn(async (url: string, init?: RequestInit) => {
      if (url.endsWith("/version")) return new Response("1");
      return new Promise<Response>((_, reject) => {
        init?.signal?.addEventListener("abort", () => {
          aborted = true;
          reject(new DOMException("Aborted", "AbortError"));
        });
      });
    }));
    const r = receiver();
    const vv = new TTS_VoicevoxService(r);
    await vv.start(data);
    vv.play("one");
    vv.play("two");
    await flush(20);
    vv.stop();
    await flush(20);
    expect(aborted).toBe(true);
    expect(sound.clearVoiceClips).toHaveBeenCalled();
    expect(sound.enqueueVoiceClip).not.toHaveBeenCalled();
    expect(r.onError).not.toHaveBeenCalled();
    expect(r.onStop).toHaveBeenCalledTimes(1);
  });

  it("a failed line goes to onError and the next line still plays", async () => {
    let calls = 0;
    vi.stubGlobal("fetch", vi.fn(async (url: string) => {
      if (url.endsWith("/version")) return new Response("1");
      if (url.includes("/audio_query")) {
        calls++;
        return calls === 1 ? new Response("", { status: 422 }) : new Response("{}");
      }
      return new Response(new Uint8Array([1]));
    }));
    const r = receiver();
    const vv = new TTS_VoicevoxService(r);
    await vv.start(data);
    vv.play("bad");
    vv.play("good");
    await flush(60);
    expect(r.onError).toHaveBeenCalledWith("VOICEVOX: tts.voicevox_query_failed");
    expect(sound.enqueueVoiceClip).toHaveBeenCalledTimes(1);
  });

  it("a line that hits the synthesis timeout is reported, and the next line still plays", async () => {
    vi.useFakeTimers();
    try {
      let queries = 0;
      vi.stubGlobal("fetch", vi.fn((url: string, init?: RequestInit) => {
        if (url.endsWith("/version")) return Promise.resolve(new Response("1"));
        if (url.includes("/audio_query") && ++queries === 1) {
          // A hung engine: never answers, rejects only when fetchWithTimeout aborts it.
          return new Promise<Response>((_, reject) => {
            init?.signal?.addEventListener("abort", () => reject(new DOMException("Aborted", "AbortError")));
          });
        }
        return Promise.resolve(new Response(url.includes("/audio_query") ? "{}" : new Uint8Array([1])));
      }));
      const r = receiver();
      const vv = new TTS_VoicevoxService(r);
      await vv.start(data);
      vv.play("hangs");
      vv.play("next");
      await vi.advanceTimersByTimeAsync(119_999);
      expect(r.onError).not.toHaveBeenCalled();
      expect(sound.enqueueVoiceClip).not.toHaveBeenCalled();
      await vi.advanceTimersByTimeAsync(1);
      await flush(40);
      expect(r.onError).toHaveBeenCalledTimes(1);
      expect(r.onError).toHaveBeenCalledWith("VOICEVOX: tts.voicevox_timeout");
      expect(sound.enqueueVoiceClip).toHaveBeenCalledTimes(1);
      expect(r.onStop).not.toHaveBeenCalled();
    } finally {
      vi.useRealTimers();
    }
  });
});

describe("fetchVoicevoxSpeakers", () => {
  it("lists speakers from the normalized host", async () => {
    const speakers = [{ name: "四国めたん", speaker_uuid: "u", styles: [{ name: "ノーマル", id: 2 }] }];
    const fetchMock = vi.fn(async (_url: string) => new Response(JSON.stringify(speakers)));
    vi.stubGlobal("fetch", fetchMock);
    await expect(fetchVoicevoxSpeakers("localhost:10101/")).resolves.toEqual(speakers);
    expect(fetchMock.mock.calls[0][0]).toBe("http://localhost:10101/speakers");
  });

  it("explains an engine that is not running instead of returning an empty list", async () => {
    vi.stubGlobal("fetch", vi.fn(async () => {
      throw new TypeError("Failed to fetch");
    }));
    await expect(fetchVoicevoxSpeakers("http://localhost:50021")).rejects.toThrow("tts.voicevox_not_running");
    await expect(fetchVoicevoxSpeakers("http://10.0.0.2:50021")).rejects.toThrow("tts.voicevox_host_error_remote_http");
  });

  it("treats a non-list answer as no speakers", async () => {
    vi.stubGlobal("fetch", vi.fn(async () => new Response("{}")));
    await expect(fetchVoicevoxSpeakers("http://localhost:50021")).resolves.toEqual([]);
  });
});

describe("Native (Web Speech)", () => {
  class FakeUtterance {
    voice: unknown = null;
    lang = "";
    pitch = 1;
    rate = 1;
    volume = 1;
    onend: (() => void) | null = null;
    onerror: ((e: { error: string }) => void) | null = null;
    constructor(public text: string) {}
  }

  function installSpeech(voices: { voiceURI: string; lang: string }[]) {
    const spoken: FakeUtterance[] = [];
    const listeners = new Set<() => void>();
    const synth = {
      getVoices: () => voices,
      speak: (u: FakeUtterance) => spoken.push(u),
      cancel: vi.fn(),
      addEventListener: (_: string, fn: () => void) => listeners.add(fn),
      removeEventListener: (_: string, fn: () => void) => listeners.delete(fn),
    };
    (window as unknown as { speechSynthesis: unknown }).speechSynthesis = synth;
    vi.stubGlobal("SpeechSynthesisUtterance", FakeUtterance);
    return { spoken, synth, listeners };
  }

  it("uses a fresh utterance per line with NaN-safe settings, and cancels on stop", async () => {
    const { spoken, synth } = installSpeech([{ voiceURI: "Haruka", lang: "ja-JP" }]);
    Object.assign(data.native, { voice: "Haruka", pitch: "abc", rate: "0", volume: "" });
    const r = receiver();
    const native = new TTS_NativeService(r);
    await native.start(data);
    expect(r.onStart).toHaveBeenCalled();

    native.play("one");
    native.play("two");
    expect(spoken).toHaveLength(2);
    expect(spoken[0]).not.toBe(spoken[1]);
    expect(spoken.map((u) => u.text)).toEqual(["one", "two"]);
    expect(spoken[0]).toMatchObject({ pitch: 1, rate: 0.1, volume: 1, lang: "ja-JP" });

    // cancel() interrupts: not an error.
    spoken[0].onerror?.({ error: "interrupted" });
    native.stop();
    expect(synth.cancel).toHaveBeenCalled();
    expect(r.onError).not.toHaveBeenCalled();
  });

  it("does not hang in connecting when the voice list never loads", async () => {
    vi.useFakeTimers();
    try {
      installSpeech([]);
      data.native.voice = "Haruka";
      const r = receiver();
      const starting = new TTS_NativeService(r).start(data);
      await vi.advanceTimersByTimeAsync(5_000);
      await starting;
      expect(r.onStop).toHaveBeenCalledWith("tts.native_voice_missing");
    } finally {
      vi.useRealTimers();
    }
  });
});

describe("Windows (SAPI)", () => {
  it("sends finite volume/rate and reports a failed line", async () => {
    Object.assign(data.windows, { voice: "v", device: "d", volume: "NaN", rate: "" });
    invokeMock.mockRejectedValue("Unable to update rate");
    const r = receiver();
    const win = new TTS_WindowsService(r);
    win.start(data);
    win.play("hi");
    await flush();
    expect(invokeMock).toHaveBeenCalledWith("plugin:windows-tts|speak", {
      data: { voice: "v", device: "d", volume: 1, rate: 1, value: "hi" },
    });
    expect(r.onError).toHaveBeenCalledWith("Windows speech: Unable to update rate");
  });
});

describe("Uberduck", () => {
  it("speaks lines one after another and drops queued lines on stop", async () => {
    Object.assign(data.uberduck, { api_key: "key", voice: "uuid", volume: "0", device: "" });
    const finishers: (() => void)[] = [];
    invokeMock.mockImplementation(() => new Promise<void>((res) => finishers.push(res)) as never);
    const r = receiver();
    const ub = new TTS_UberduckService(r);
    ub.start(data);
    ub.play("one");
    ub.play("two");
    await flush();
    expect(invokeMock).toHaveBeenCalledTimes(1);
    expect(invokeMock.mock.calls[0][1]).toMatchObject({
      data: { text: "one", volume: 0, device_name: "Speakers (App default)" },
    });
    ub.stop();
    finishers[0]();
    await flush();
    expect(invokeMock).toHaveBeenCalledTimes(1);
  });
});
