import i18n from "i18next";
import { proxy } from "valtio";
import { afterEach, beforeAll, beforeEach, describe, expect, it, vi } from "vitest";

vi.mock("@tauri-apps/api/core", () => ({ invoke: vi.fn(), isTauri: () => false }));

import { invoke } from "@tauri-apps/api/core";
import { Service_TTS_Schema, TTS_State } from "../schema";
import {
  GPT_SOVITS_DEFAULT_URL,
  TTS_GptSovitsService,
  gptSovitsSettings,
  gsvLanguageOf,
  isLocalServer,
  sentenceLanguage,
} from "../services/gptSovits";
import type { ITTSReceiver, TtsSinkContext } from "../types";

// P6-G: the gptSovits backend (a user-run GPT-SoVITS api_v2 server through the Rust gpt-sovits plugin).

const invokeMock = vi.mocked(invoke);

function receiver() {
  return { onStart: vi.fn(), onStop: vi.fn(), onError: vi.fn() } satisfies ITTSReceiver;
}

let data: TTS_State;

beforeAll(async () => {
  await i18n.init({ lng: "en", resources: { en: { translation: {} } }, interpolation: { escapeValue: false } });
});

beforeEach(() => {
  invokeMock.mockReset();
  data = proxy(Service_TTS_Schema.parse({ gptSovits: { referenceVoiceId: "alice-1a2b3c" } }));
  vi.stubGlobal("window", { ApiServer: { state: { services: { tts: { data } }, audioOutputDevice: "" } } });
});

afterEach(() => {
  vi.unstubAllGlobals();
});

function calls(cmd: string) {
  return invokeMock.mock.calls.filter(([c]) => c === cmd).map(([, args]) => args as Record<string, unknown>);
}

const SINK: TtsSinkContext = { utteranceId: "u1", target: { route: "device", device: "", volume: 1 }, lang: "ja" };

describe("GPT-SoVITS settings", () => {
  it("maps caption languages onto GPT-SoVITS codes", () => {
    expect(gsvLanguageOf("ja")).toBe("ja");
    expect(gsvLanguageOf("zh-CN")).toBe("zh");
    expect(gsvLanguageOf("zh-HK")).toBe("yue");
    expect(gsvLanguageOf("zh_MO")).toBe("yue");
    expect(gsvLanguageOf("zh-yue")).toBe("yue");
    expect(gsvLanguageOf("yue")).toBe("yue");
    // Traditional Chinese (zh-TW / zh-HK / zh-MO captions all canonicalize to zh-Hant) is Mandarin, as in Rust.
    expect(gsvLanguageOf("zh-Hant")).toBe("zh");
    expect(gsvLanguageOf("zh-TW")).toBe("zh");
    expect(gsvLanguageOf("zh-Hant-TW")).toBe("zh");
    expect(sentenceLanguage("auto", "zh-Hant")).toBe("zh");
    expect(gsvLanguageOf("en-US")).toBe("en");
    expect(gsvLanguageOf("fr")).toBeNull();
    expect(gsvLanguageOf(null)).toBeNull();
    expect(sentenceLanguage("auto", "ja")).toBe("ja");
    expect(sentenceLanguage("auto", "fr")).toBe("auto");
    expect(sentenceLanguage("all_ja", "en")).toBe("all_ja");
    expect(sentenceLanguage("", null)).toBe("auto");
  });

  it("knows which servers can read the voice library", () => {
    expect(isLocalServer(GPT_SOVITS_DEFAULT_URL)).toBe(true);
    expect(isLocalServer("localhost:9880")).toBe(true);
    expect(isLocalServer("http://[::1]:9880")).toBe(true);
    expect(isLocalServer("http://192.168.1.20:9880")).toBe(false);
    expect(isLocalServer("http://gpu-box:9880")).toBe(false);
    expect(isLocalServer("not a url")).toBe(false);
  });

  it("uses a library voice on this PC and the server path elsewhere", () => {
    const local = gptSovitsSettings({ ...data.gptSovits, refAudioPath: "C:/refs/x.wav", streamingMode: "fast", speed: "9", timeout: "12.5" });
    expect(local).toMatchObject({
      baseUrl: GPT_SOVITS_DEFAULT_URL,
      referenceVoiceId: "alice-1a2b3c",
      refAudioPath: "",
      streaming: true,
      streamingMode: 3,
      speed: 2,
      timeoutMs: 12_500,
      textLang: "auto",
      promptLang: "auto",
      extra: null,
    });
    const remote = gptSovitsSettings({ ...data.gptSovits, baseUrl: "http://192.168.1.20:9880", refAudioPath: " D:/refs/mao.wav " });
    expect(remote).toMatchObject({ referenceVoiceId: "", refAudioPath: "D:/refs/mao.wav" });
    // A remote server with only a library voice has no reference it can read.
    expect(gptSovitsSettings({ ...data.gptSovits, baseUrl: "http://192.168.1.20:9880" })).toBe("tts.gpt_sovits_no_reference");
    expect(gptSovitsSettings({ ...data.gptSovits, referenceVoiceId: "" })).toBe("tts.gpt_sovits_no_reference");
    expect(gptSovitsSettings({ ...data.gptSovits, extra: "[1]" })).toBe("tts.openai_speech_extra_invalid");
    expect(gptSovitsSettings({ ...data.gptSovits, extra: '{"top_k": 5}' })).toMatchObject({ extra: { top_k: 5 } });
    expect(gptSovitsSettings({ ...data.gptSovits, streamingMode: "fragments" })).toMatchObject({ streamingMode: 0 });
  });

  it("defaults are safe and saved junk falls back", () => {
    const d = Service_TTS_Schema.parse({}).gptSovits;
    expect(d).toMatchObject({ baseUrl: GPT_SOVITS_DEFAULT_URL, streaming: true, streamingMode: "fragments", textLang: "auto", timeout: "0" });
    expect(Service_TTS_Schema.parse({ gptSovits: { streamingMode: "turbo" } }).gptSovits.streamingMode).toBe("fragments");
  });
});

describe("TTS_GptSovitsService", () => {
  it("checks the server and switches the weights at start", async () => {
    data.gptSovits.gptWeights = "GPT_weights_v2Pro/mao.ckpt";
    invokeMock.mockImplementation(async (cmd) => (cmd === "plugin:gpt-sovits|gpt_sovits_check" ? true : undefined));
    const r = receiver();
    await new TTS_GptSovitsService(r).start(data);
    expect(r.onStart).toHaveBeenCalled();
    expect(calls("plugin:gpt-sovits|gpt_sovits_check")).toEqual([{ baseUrl: GPT_SOVITS_DEFAULT_URL }]);
    expect(calls("plugin:gpt-sovits|gpt_sovits_set_weights")).toEqual([
      { baseUrl: GPT_SOVITS_DEFAULT_URL, gptWeights: "GPT_weights_v2Pro/mao.ckpt", sovitsWeights: "" },
    ]);
  });

  it("refuses to start when the server is down, is another server, or the weights fail", async () => {
    invokeMock.mockResolvedValue(false);
    const down = receiver();
    await new TTS_GptSovitsService(down).start(data);
    expect(down.onStop).toHaveBeenCalledWith("tts.gpt_sovits_unreachable");

    invokeMock.mockRejectedValue("The server at http://127.0.0.1:9880/ answers, but it is not GPT-SoVITS api_v2");
    const other = receiver();
    await new TTS_GptSovitsService(other).start(data);
    expect(other.onStop.mock.calls[0][0]).toContain("not GPT-SoVITS api_v2");

    data.gptSovits.sovitsWeights = "SoVITS_weights/x.pth";
    invokeMock.mockImplementation(async (cmd) => {
      if (cmd === "plugin:gpt-sovits|gpt_sovits_check") return true;
      throw "Switching the SoVITS weights failed (HTTP 400): change sovits weight failed: No such file";
    });
    const weights = receiver();
    await new TTS_GptSovitsService(weights).start(data);
    expect(weights.onStop).toHaveBeenCalledWith("tts.gpt_sovits_weights_failed");
    expect(weights.onStart).not.toHaveBeenCalled();
  });

  it("speaks with the caption's language and cancels on abort", async () => {
    invokeMock.mockImplementation(async (cmd) => {
      if (cmd === "plugin:gpt-sovits|gpt_sovits_check") return true;
      if (cmd === "plugin:gpt-sovits|gpt_sovits_speak") return { sampleRate: 32000, frames: 10, firstAudioMs: 1, totalMs: 2 };
      return undefined;
    });
    const svc = new TTS_GptSovitsService(receiver());
    await svc.start(data);
    await svc.synthesizeToSink("今日はいい天気ですね。", SINK, new AbortController().signal);
    const speak = calls("plugin:gpt-sovits|gpt_sovits_speak");
    expect(speak).toHaveLength(1);
    expect(speak[0]).toMatchObject({
      utteranceId: "u1",
      request: { text: "今日はいい天気ですね。", textLang: "ja", referenceVoiceId: "alice-1a2b3c", streaming: true, streamingMode: 0 },
    });
    // An explicit language wins over the caption's.
    data.gptSovits.textLang = "all_ja";
    await svc.synthesizeToSink("OK", { ...SINK, utteranceId: "u2", lang: "en" }, new AbortController().signal);
    expect(calls("plugin:gpt-sovits|gpt_sovits_speak")[1]).toMatchObject({ request: { textLang: "all_ja" } });

    // Abort cancels the stream; the "cancelled" answer is not an error.
    const ctl = new AbortController();
    invokeMock.mockImplementation(async (cmd) => {
      if (cmd === "plugin:gpt-sovits|gpt_sovits_speak") {
        ctl.abort();
        throw "cancelled";
      }
      return undefined;
    });
    await expect(svc.synthesizeToSink("はい。", { ...SINK, utteranceId: "u3" }, ctl.signal)).resolves.toBeUndefined();
    expect(calls("plugin:openai-speech|openai_speech_cancel")).toContainEqual({ utteranceId: "u3" });

    invokeMock.mockImplementation(async () => {
      throw "GPT-SoVITS synthesis: HTTP 400 Bad Request: tts failed";
    });
    await expect(svc.synthesizeToSink("はい。", { ...SINK, utteranceId: "u4" }, new AbortController().signal)).rejects.toThrow(
      "GPT-SoVITS: GPT-SoVITS synthesis: HTTP 400",
    );
  });
});
