import i18n from "i18next";
import { proxy } from "valtio";
import { afterEach, beforeAll, beforeEach, describe, expect, it, vi } from "vitest";

vi.mock("@tauri-apps/api/core", () => ({ invoke: vi.fn(), isTauri: () => false }));
vi.mock("@/utils/devLog", () => ({ devLog: () => {} }));

import { invoke } from "@tauri-apps/api/core";
import type { RustTtsSinkTarget } from "@/types";
import { Service_TTS_Schema, TTS_Backends, TTS_State } from "../schema";
import { OPENAI_SPEECH_PRESET_IDS } from "../schema/openaiSpeech";
import { TTS_BACKENDS } from "../index";
import { TTS_ChatterboxService } from "../services/chatterbox";
import { TTS_FishSpeechService } from "../services/fishspeech";
import { TTS_KokoroService } from "../services/kokoro";
import { TTS_MeloService } from "../services/melo";
import { TTS_NeuralSidecarService } from "../services/neuralSidecar";
import {
  OPENAI_SPEECH_PRESETS,
  TTS_OpenAISpeechService,
  applyOpenAiSpeechPreset,
  chatterboxSpeechSettings,
  kokoroSpeechSettings,
  legacyV1Base,
  openAiSpeechSettings,
  parseExtra,
  previewToSink,
  serverOrigin,
} from "../services/openaiSpeech";
import { ITTSReceiver, ITTSService, TtsSinkContext, ttsOutputMode } from "../types";

const invokeMock = vi.mocked(invoke);

function receiver() {
  return { onStart: vi.fn(), onStop: vi.fn(), onError: vi.fn() } satisfies ITTSReceiver;
}

const flush = async (n = 10) => {
  for (let i = 0; i < n; i++) await Promise.resolve();
};

const TARGET: RustTtsSinkTarget = { route: "device", device: "", volume: 1 };

function sinkCtx(id = "tts-1"): TtsSinkContext {
  return { utteranceId: id, target: TARGET };
}

let data: TTS_State;
let sink: { ready: ReturnType<typeof vi.fn>; begin: ReturnType<typeof vi.fn>; end: ReturnType<typeof vi.fn>; cancel: ReturnType<typeof vi.fn> };

beforeAll(async () => {
  await i18n.init({ lng: "en", resources: { en: { translation: {} } }, interpolation: { escapeValue: false } });
});

beforeEach(() => {
  invokeMock.mockReset();
  data = proxy(Service_TTS_Schema.parse({}));
  sink = {
    ready: vi.fn(async () => {}),
    begin: vi.fn(async () => {}),
    end: vi.fn(async () => {}),
    cancel: vi.fn(async () => {}),
  };
  vi.stubGlobal("window", {
    ApiServer: {
      state: { services: { tts: { data } }, audioOutputDevice: "" },
      sound: { sink },
      voiceChanger: { state: { isRunning: false } },
    },
  });
});

afterEach(() => {
  vi.unstubAllGlobals();
});

/** Calls of `cmd` (full command name). */
function calls(cmd: string) {
  return invokeMock.mock.calls.filter(([c]) => c === cmd).map(([, args]) => args as Record<string, unknown>);
}

/** Answer the server checks with `true`, speak commands with a report, everything else with undefined. */
function answerOk() {
  invokeMock.mockImplementation(async (cmd) => {
    if (cmd.includes("check")) return true;
    if (cmd.includes("speak")) return { sampleRate: 24_000, frames: 100, firstAudioMs: 5, totalMs: 9 };
    return undefined;
  });
}

// -------------------------------------------------------------------------------------------------
// Presets and settings (pure)
// -------------------------------------------------------------------------------------------------

describe("OpenAI-compatible presets", () => {
  it("every preset is a complete, valid server template", () => {
    for (const id of OPENAI_SPEECH_PRESET_IDS) {
      if (id === "custom") continue;
      const p = OPENAI_SPEECH_PRESETS[id];
      expect(p.baseUrl, id).toMatch(/^http:\/\/127\.0\.0\.1:\d+\/v1$/);
      expect(["pcm", "wav"], id).toContain(p.format);
      expect(p.pcmSampleRate, id).toBeGreaterThanOrEqual(8_000);
      const applied = applyOpenAiSpeechPreset(data.openaiSpeech, id);
      expect(typeof openAiSpeechSettings(applied), id).toBe("object");
    }
    // The research's server targets are all there.
    expect(Object.keys(OPENAI_SPEECH_PRESETS).sort()).toEqual(["chatterbox", "indexTts", "kokoroFastapi", "qwentts", "voxcpm2", "voxtral"]);
    expect(OPENAI_SPEECH_PRESETS.kokoroFastapi.baseUrl).toContain(":8880");
    expect(OPENAI_SPEECH_PRESETS.chatterbox.baseUrl).toContain(":5555");
    expect(OPENAI_SPEECH_PRESETS.voxcpm2.pcmSampleRate).toBe(48_000);
  });

  it("applying a preset replaces the server fields and keeps key (same server), speed, device, volume and timeout", () => {
    const current = {
      ...data.openaiSpeech,
      baseUrl: "http://127.0.0.1:8000/v1",
      apiKey: "k",
      speed: "1.3",
      device: "Speakers",
      volume: "0.7",
      timeout: "30",
      extra: '{"a":1}',
    };
    const voxtral = applyOpenAiSpeechPreset(current, "voxtral");
    expect(voxtral).toMatchObject({
      preset: "voxtral",
      baseUrl: "http://127.0.0.1:8000/v1",
      model: "mistralai/Voxtral-4B-TTS-2603",
      voice: "casual_male",
      format: "pcm",
      pcmSampleRate: "24000",
      stream: true,
      extra: "",
      apiKey: "k",
      speed: "1.3",
      device: "Speakers",
      volume: "0.7",
      timeout: "30",
    });
    expect(JSON.parse(applyOpenAiSpeechPreset(current, "chatterbox").extra)).toEqual({ exaggeration: 0.5 });
    // Custom only records the choice.
    expect(applyOpenAiSpeechPreset(voxtral, "custom")).toEqual({ ...voxtral, preset: "custom" });
    // Another model on the same vLLM server keeps the key.
    expect(applyOpenAiSpeechPreset(voxtral, "voxcpm2").apiKey).toBe("k");
  });

  it("a preset that switches to another server clears the API key (it is never sent to a different server)", () => {
    // Review finding: an OpenAI key entered under Custom went to 127.0.0.1:8880 after picking Kokoro-FastAPI.
    const cloud = { ...data.openaiSpeech, preset: "custom" as const, baseUrl: "https://api.openai.com/v1", apiKey: "sk-live" };
    for (const id of OPENAI_SPEECH_PRESET_IDS) {
      if (id === "custom") continue;
      expect(applyOpenAiSpeechPreset(cloud, id).apiKey, id).toBe("");
    }
    expect(applyOpenAiSpeechPreset(cloud, "custom").apiKey).toBe("sk-live");
    // Same host, other port; a blank or unparsable current URL: cleared too.
    const kokoro = { ...data.openaiSpeech, baseUrl: "http://127.0.0.1:8880/v1", apiKey: "k" };
    expect(applyOpenAiSpeechPreset(kokoro, "voxtral").apiKey).toBe("");
    expect(applyOpenAiSpeechPreset({ ...kokoro, baseUrl: "" }, "kokoroFastapi").apiKey).toBe("");
    expect(applyOpenAiSpeechPreset({ ...kokoro, baseUrl: "http://" }, "kokoroFastapi").apiKey).toBe("");
    // The same server written differently (no scheme, trailing slash, other path) keeps it.
    expect(applyOpenAiSpeechPreset({ ...kokoro, baseUrl: "127.0.0.1:8880/" }, "kokoroFastapi").apiKey).toBe("k");
    expect(applyOpenAiSpeechPreset({ ...kokoro, baseUrl: "http://127.0.0.1:8880/api/v1" }, "kokoroFastapi").apiKey).toBe("k");
    expect(serverOrigin("HTTPS://Example.com:443/v1")).toBe("https://example.com");
    expect(serverOrigin("ftp://example.com")).toBeNull();
  });

  it("defaults point at Kokoro-FastAPI", () => {
    expect(data.openaiSpeech).toMatchObject({ preset: "kokoroFastapi", baseUrl: "http://127.0.0.1:8880/v1", model: "kokoro", voice: "af_heart", format: "pcm" });
    // A saved slice with a junk preset / format falls back instead of resetting the backend.
    const parsed = Service_TTS_Schema.parse({ backend: "openaiSpeech", openaiSpeech: { preset: "nope", format: "mp3", baseUrl: "http://gpu-box:8000/v1" } });
    expect(parsed.backend).toBe(TTS_Backends.openaiSpeech);
    expect(parsed.openaiSpeech).toMatchObject({ preset: "kokoroFastapi", format: "pcm", baseUrl: "http://gpu-box:8000/v1" });
  });
});

describe("request settings", () => {
  it("parses the extra JSON object", () => {
    expect(parseExtra("")).toBeNull();
    expect(parseExtra("   ")).toBeNull();
    expect(parseExtra('{"lang_code":"j"}')).toEqual({ lang_code: "j" });
    expect(parseExtra("[1]")).toBeUndefined();
    expect(parseExtra('"x"')).toBeUndefined();
    expect(parseExtra("{nope")).toBeUndefined();
  });

  it("validates and clamps the openaiSpeech settings", () => {
    const c = {
      ...data.openaiSpeech,
      baseUrl: " http://gpu-box:8000/v1 ",
      apiKey: " sk ",
      model: " m ",
      voice: " v ",
      speed: "9",
      format: "wav" as const,
      pcmSampleRate: "abc",
      stream: false,
      extra: '{"x": 1}',
      timeout: "5000",
    };
    expect(openAiSpeechSettings(c)).toEqual({
      baseUrl: "http://gpu-box:8000/v1",
      apiKey: "sk",
      model: "m",
      voice: "v",
      responseFormat: "wav",
      speed: 4,
      stream: false,
      pcmSampleRate: 24_000,
      extra: { x: 1 },
      timeoutMs: 600_000,
    });
    expect(openAiSpeechSettings({ ...c, baseUrl: "  " })).toBe("tts.openai_speech_url_missing");
    expect(openAiSpeechSettings({ ...c, extra: "[]" })).toBe("tts.openai_speech_extra_invalid");
    expect(openAiSpeechSettings({ ...c, timeout: "0", speed: "", pcmSampleRate: "48000" })).toMatchObject({ timeoutMs: 0, speed: 1, pcmSampleRate: 48_000 });
  });

  it("maps legacy Kokoro and Chatterbox settings onto the client", () => {
    expect(legacyV1Base(" http://localhost:8880/ ")).toBe("http://localhost:8880/v1");
    expect(legacyV1Base("http://localhost:8880/v1")).toBe("http://localhost:8880/v1");
    expect(legacyV1Base("https://tts.example/kokoro")).toBe("https://tts.example/kokoro/v1");
    expect(legacyV1Base("")).toBe("");

    const kokoro = { ...data.kokoro, endpoint: " http://localhost:8880 ", voice: " ", speed: "abc" };
    expect(kokoroSpeechSettings(kokoro)).toEqual({
      baseUrl: "http://localhost:8880/v1",
      apiKey: "",
      model: "kokoro",
      voice: "af_bella",
      responseFormat: "pcm",
      speed: 1,
      stream: true,
      pcmSampleRate: 24_000,
      extra: null,
      timeoutMs: 0,
    });

    const chatterbox = { ...data.chatterbox, voice: "alice", speed: "1.2", exaggeration: "0.8" };
    expect(chatterboxSpeechSettings(chatterbox)).toMatchObject({
      baseUrl: "http://localhost:5555/v1",
      model: "chatterbox",
      voice: "alice",
      responseFormat: "wav",
      speed: 1.2,
      stream: false,
      extra: { exaggeration: 0.8 },
    });
    expect(chatterboxSpeechSettings({ ...chatterbox, voice: "", exaggeration: "junk" })).toMatchObject({ voice: "default", extra: { exaggeration: 0.5 } });
  });

  it("legacy saved settings keep parsing and gain a sink volume", () => {
    const saved = {
      backend: "kokoro",
      kokoro: { endpoint: "http://192.168.1.5:8880", voice: "jf_alpha", speed: "1.1", device: "Headphones" },
      chatterbox: { endpoint: "http://localhost:5555", voice: "me", speed: "1.0", exaggeration: "0.3", device: "" },
      melo: { endpoint: "http://localhost:8888", speaker: "EN-US", speed: "1.0", device: "" },
      fishSpeech: { endpoint: "https://api.fish.audio", referenceId: "abc", apiKey: "k", model: "s2.1-pro-free", device: "" },
      neuralSidecar: { endpoint: "http://127.0.0.1:8766", voice: "default", device: "", routeThroughVoiceChanger: true },
    };
    const parsed = Service_TTS_Schema.parse(saved);
    expect(parsed.backend).toBe(TTS_Backends.kokoro);
    // Numeric strings are normalized by zStringNumber ("1.0" → "1"); everything else is kept as saved.
    expect(parsed.kokoro).toEqual({ ...saved.kokoro, volume: "1" });
    expect(parsed.chatterbox).toEqual({ ...saved.chatterbox, speed: "1", volume: "1" });
    expect(parsed.melo).toEqual({ ...saved.melo, speed: "1", volume: "1" });
    expect(parsed.fishSpeech).toEqual({ ...saved.fishSpeech, volume: "1" });
    expect(parsed.neuralSidecar).toEqual({ ...saved.neuralSidecar, volume: "1" });
  });
});

// -------------------------------------------------------------------------------------------------
// Backends
// -------------------------------------------------------------------------------------------------

describe("HTTP sink backends", () => {
  const all: [string, new (r: ITTSReceiver) => ITTSService][] = [
    ["openaiSpeech", TTS_OpenAISpeechService],
    ["kokoro", TTS_KokoroService],
    ["chatterbox", TTS_ChatterboxService],
    ["melo", TTS_MeloService],
    ["fishSpeech", TTS_FishSpeechService],
    ["neuralSidecar", TTS_NeuralSidecarService],
  ];

  it.each(all)("%s speaks into the sink (no clip playback, no JSON audio)", (backend, Backend) => {
    expect(TTS_BACKENDS[backend as TTS_Backends]).toBe(Backend);
    expect(ttsOutputMode(new Backend(receiver()))).toBe("sink");
  });

  it("Kokoro: legacy settings speak through openai_speech_speak", async () => {
    answerOk();
    data.kokoro.endpoint = " http://localhost:8880 ";
    data.kokoro.speed = "abc";
    const r = receiver();
    const kokoro = new TTS_KokoroService(r);
    await kokoro.start(data);
    expect(r.onStart).toHaveBeenCalledTimes(1);
    expect(invokeMock).toHaveBeenCalledWith("plugin:kokoro-tts|set_kokoro_endpoint", { endpoint: "http://localhost:8880" });
    expect(invokeMock).toHaveBeenCalledWith("plugin:kokoro-tts|check_kokoro_availability", { endpoint: "http://localhost:8880" });

    await kokoro.synthesizeToSink("Hello there.", sinkCtx("tts-a-0"), new AbortController().signal);
    expect(calls("plugin:openai-speech|openai_speech_speak")).toEqual([
      {
        request: {
          baseUrl: "http://localhost:8880/v1",
          apiKey: "",
          model: "kokoro",
          input: "Hello there.",
          voice: "af_bella",
          responseFormat: "pcm",
          speed: 1, // junk speed falls back instead of NaN
          stream: true,
          pcmSampleRate: 24_000,
          extra: null,
          timeoutMs: 0,
        },
        utteranceId: "tts-a-0",
      },
    ]);
    // No clip commands any more.
    expect(invokeMock.mock.calls.some(([c]) => c === "plugin:kokoro-tts|kokoro_speak")).toBe(false);
  });

  it("Kokoro: an unreachable server refuses Start with a hint", async () => {
    invokeMock.mockImplementation(async (cmd) => (cmd.endsWith("check_kokoro_availability") ? false : undefined));
    const r = receiver();
    await new TTS_KokoroService(r).start(data);
    expect(r.onStart).not.toHaveBeenCalled();
    expect(r.onStop).toHaveBeenCalledWith(expect.stringContaining("Kokoro TTS not available at http://localhost:8880"));
  });

  it("Stop during connect wins over the late result", async () => {
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
    expect(r.onStop).toHaveBeenCalledWith();
    // Stop cancels every running HTTP speech stream.
    expect(invokeMock).toHaveBeenCalledWith("plugin:openai-speech|openai_speech_cancel", { utteranceId: null });
  });

  it("a failed sentence rejects with the backend's label and the backend keeps running", async () => {
    invokeMock.mockImplementation(async (cmd) => {
      if (cmd.includes("check")) return true;
      if (cmd.endsWith("openai_speech_speak")) throw "Speech request: HTTP 500 Internal Server Error";
      return undefined;
    });
    const r = receiver();
    const kokoro = new TTS_KokoroService(r);
    await kokoro.start(data);
    await expect(kokoro.synthesizeToSink("hello", sinkCtx(), new AbortController().signal)).rejects.toThrow(
      "Kokoro: Speech request: HTTP 500 Internal Server Error",
    );
    expect(r.onStop).not.toHaveBeenCalled();
    expect(kokoro.running).toBe(true);
  });

  it("stop / skip cancel the Rust request and are never reported as errors", async () => {
    let finish!: (v: unknown) => void;
    let reject!: (e: unknown) => void;
    invokeMock.mockImplementation(async (cmd) => {
      if (cmd.includes("check")) return true;
      if (cmd.endsWith("openai_speech_speak")) return new Promise((res, rej) => ((finish = res), (reject = rej)));
      return undefined;
    });
    const kokoro = new TTS_KokoroService(receiver());
    await kokoro.start(data);
    const controller = new AbortController();
    const speaking = kokoro.synthesizeToSink("hello", sinkCtx("tts-x-3"), controller.signal);
    await flush();
    controller.abort();
    expect(invokeMock).toHaveBeenCalledWith("plugin:openai-speech|openai_speech_cancel", { utteranceId: "tts-x-3" });
    reject("cancelled");
    await expect(speaking).resolves.toBeUndefined();

    // A sink cancel (the pipeline cancelled the utterance) is not an error either.
    const second = kokoro.synthesizeToSink("again", sinkCtx("tts-x-4"), new AbortController().signal);
    await flush();
    reject("cancelled");
    await expect(second).resolves.toBeUndefined();
    void finish;
  });

  it("an aborted or blank sentence makes no request", async () => {
    answerOk();
    const kokoro = new TTS_KokoroService(receiver());
    await kokoro.start(data);
    const aborted = new AbortController();
    aborted.abort();
    await kokoro.synthesizeToSink("hello", sinkCtx(), aborted.signal);
    await kokoro.synthesizeToSink("   ", sinkCtx(), new AbortController().signal);
    expect(calls("plugin:openai-speech|openai_speech_speak")).toHaveLength(0);
  });

  it("Chatterbox: legacy settings map onto the client with exaggeration", async () => {
    answerOk();
    Object.assign(data.chatterbox, { voice: "me", exaggeration: "0.25" });
    const chatterbox = new TTS_ChatterboxService(receiver());
    await chatterbox.start(data);
    expect(invokeMock).toHaveBeenCalledWith("plugin:chatterbox-tts|check_chatterbox_availability", { endpoint: "http://localhost:5555" });
    await chatterbox.synthesizeToSink("Hi.", sinkCtx("c-1"), new AbortController().signal);
    const [call] = calls("plugin:openai-speech|openai_speech_speak");
    expect(call.utteranceId).toBe("c-1");
    expect(call.request).toMatchObject({
      baseUrl: "http://localhost:5555/v1",
      model: "chatterbox",
      voice: "me",
      responseFormat: "wav",
      stream: false,
      extra: { exaggeration: 0.25 },
      input: "Hi.",
    });
  });

  it("OpenAI-compatible: checks the server, then applies voice and speed changes to the next sentence", async () => {
    answerOk();
    Object.assign(data.openaiSpeech, { baseUrl: "http://gpu-box:8000/v1", apiKey: " sk ", model: "openbmb/VoxCPM2", voice: "", pcmSampleRate: "48000" });
    const r = receiver();
    const service = new TTS_OpenAISpeechService(r);
    await service.start(data);
    expect(r.onStart).toHaveBeenCalledTimes(1);
    expect(invokeMock).toHaveBeenCalledWith("plugin:openai-speech|openai_speech_check", { baseUrl: "http://gpu-box:8000/v1", apiKey: "sk" });

    await service.synthesizeToSink("One.", sinkCtx("o-1"), new AbortController().signal);
    data.openaiSpeech.voice = "narrator";
    data.openaiSpeech.speed = "1.5";
    await service.synthesizeToSink("Two.", sinkCtx("o-2"), new AbortController().signal);
    const [first, second] = calls("plugin:openai-speech|openai_speech_speak");
    expect(first.request).toMatchObject({ voice: "", speed: 1, pcmSampleRate: 48_000, apiKey: "sk", input: "One." });
    expect(second.request).toMatchObject({ voice: "narrator", speed: 1.5, input: "Two." });

    // Settings broken while running: that sentence fails, readable.
    data.openaiSpeech.extra = "{oops";
    await expect(service.synthesizeToSink("Three.", sinkCtx("o-3"), new AbortController().signal)).rejects.toThrow(
      "Speech server: tts.openai_speech_extra_invalid",
    );
  });

  it("OpenAI-compatible: refuses to start without a URL, with a bad extra, or with no server", async () => {
    data.openaiSpeech.baseUrl = " ";
    const r1 = receiver();
    await new TTS_OpenAISpeechService(r1).start(data);
    expect(r1.onStop).toHaveBeenCalledWith("tts.openai_speech_url_missing");
    expect(invokeMock).not.toHaveBeenCalled();

    data.openaiSpeech.baseUrl = "http://127.0.0.1:8880/v1";
    data.openaiSpeech.extra = "[1,2]";
    const r2 = receiver();
    await new TTS_OpenAISpeechService(r2).start(data);
    expect(r2.onStop).toHaveBeenCalledWith("tts.openai_speech_extra_invalid");

    data.openaiSpeech.extra = "";
    invokeMock.mockResolvedValue(false);
    const r3 = receiver();
    await new TTS_OpenAISpeechService(r3).start(data);
    expect(r3.onStop).toHaveBeenCalledWith("tts.openai_speech_unreachable");

    invokeMock.mockRejectedValue("The speech server at http://127.0.0.1:8880/v1 rejected the API key (HTTP 401 Unauthorized). Check the key.");
    const r4 = receiver();
    await new TTS_OpenAISpeechService(r4).start(data);
    expect(r4.onStop).toHaveBeenCalledWith(expect.stringContaining("rejected the API key"));
  });

  it("MeloTTS streams each sentence with its own request shape", async () => {
    answerOk();
    Object.assign(data.melo, { endpoint: " http://localhost:8888 ", speaker: " EN-US ", speed: "0" });
    const melo = new TTS_MeloService(receiver());
    await melo.start(data);
    await melo.synthesizeToSink("Hi.", sinkCtx("m-1"), new AbortController().signal);
    expect(calls("plugin:melo-tts|melo_speak_to_sink")).toEqual([
      { text: "Hi.", speakerId: "EN-US", speed: 1, endpoint: "http://localhost:8888", utteranceId: "m-1" },
    ]);
  });

  it("Fish Speech passes the API key and model through, and blanks as null", async () => {
    answerOk();
    Object.assign(data.fishSpeech, { endpoint: "https://api.fish.audio", apiKey: " sk-test ", model: "s2.1-pro-free", referenceId: "default" });
    const fish = new TTS_FishSpeechService(receiver());
    await fish.start(data);
    expect(invokeMock).toHaveBeenCalledWith("plugin:fish-speech|check_fish_availability", { endpoint: "https://api.fish.audio", apiKey: "sk-test" });
    await fish.synthesizeToSink("hello", sinkCtx("f-1"), new AbortController().signal);
    expect(calls("plugin:fish-speech|fish_speak_to_sink")).toEqual([
      { text: "hello", referenceId: "default", endpoint: "https://api.fish.audio", apiKey: "sk-test", model: "s2.1-pro-free", utteranceId: "f-1" },
    ]);

    await TTS_FishSpeechService.getVoices("http://localhost:8080", "  ");
    expect(invokeMock).toHaveBeenLastCalledWith("plugin:fish-speech|get_fish_voices", { endpoint: "http://localhost:8080", apiKey: null });
  });

  it("Fish Speech: a rejected key stops the backend with the server's reason", async () => {
    invokeMock.mockImplementation(async (cmd) => {
      if (cmd.endsWith("check_fish_availability")) throw "Fish Speech rejected the API key (HTTP 401 Unauthorized).";
      return undefined;
    });
    const r = receiver();
    await new TTS_FishSpeechService(r).start(data);
    expect(r.onStop).toHaveBeenCalledWith("Fish Speech rejected the API key (HTTP 401 Unauthorized).");
  });

  it("HTTP speech server streams into the sink; the route replaces the old voice-changer command", async () => {
    answerOk();
    Object.assign(data.neuralSidecar, { voice: "  ", routeThroughVoiceChanger: true });
    const sidecar = new TTS_NeuralSidecarService(receiver());
    await sidecar.start(data);
    await sidecar.synthesizeToSink("Hi.", sinkCtx("s-1"), new AbortController().signal);
    expect(calls("plugin:neural-sidecar-tts|neural_sidecar_tts_speak_to_sink")).toEqual([
      { text: "Hi.", voice: "default", endpoint: "http://127.0.0.1:8766", utteranceId: "s-1" },
    ]);
    expect(invokeMock.mock.calls.some(([c]) => String(c).includes("speak_into_voice_changer"))).toBe(false);
  });

  it("getVoices rejects instead of hiding a failure behind an empty list", async () => {
    invokeMock.mockRejectedValue("Kokoro voice list: connection refused");
    await expect(TTS_KokoroService.getVoices("http://localhost:8880")).rejects.toBe("Kokoro voice list: connection refused");
  });
});

describe("previewToSink", () => {
  it("begins, speaks and ends one utterance", async () => {
    const speak = vi.fn(async () => {});
    await previewToSink(TARGET, speak);
    const id = sink.begin.mock.calls[0][0] as string;
    expect(id).toMatch(/^tts-preview-/);
    expect(sink.begin).toHaveBeenCalledWith(id, TARGET);
    expect(speak).toHaveBeenCalledWith(id);
    expect(sink.end).toHaveBeenCalledWith(id);
    expect(sink.cancel).not.toHaveBeenCalled();
  });

  it("cancels the utterance on failure; a stop is not an error", async () => {
    await expect(previewToSink(TARGET, async () => Promise.reject("cancelled"))).resolves.toBeUndefined();
    expect(sink.cancel).toHaveBeenCalledTimes(1);
    await expect(previewToSink(TARGET, async () => Promise.reject("Speech request: HTTP 404"))).rejects.toBe("Speech request: HTTP 404");
    expect(sink.cancel).toHaveBeenCalledTimes(2);
    expect(sink.end).not.toHaveBeenCalled();
  });
});
