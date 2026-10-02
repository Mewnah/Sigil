import i18n from "i18next";
import { proxy } from "valtio";
import { afterEach, beforeAll, beforeEach, describe, expect, it, vi } from "vitest";

vi.mock("@tauri-apps/api/core", () => ({ invoke: vi.fn(), isTauri: vi.fn(() => true) }));
vi.mock("@tauri-apps/api/event", () => ({ listen: vi.fn(async () => () => {}) }));
vi.mock("@/utils/devLog", () => ({ devLog: () => {} }));

import { invoke, isTauri } from "@tauri-apps/api/core";
import { listen } from "@tauri-apps/api/event";
import type { RustTtsHelperEndpoint, RustTtsHelperStatus } from "@/types";
import { Service_TTS_Schema, TTS_State } from "../schema";
import { TTS_Qwen3_Schema } from "../schema/qwen3";
import { TTS_BACKENDS } from "../index";
import {
  NVIDIA_RUNTIME_ID,
  QWEN3_DESIGN_SEED,
  QWEN3_RUNTIME_IDS,
  QWEN3_SPEAKERS,
  QWEN3_TOKENIZER_ID,
  TTS_Qwen3Service,
  noteQwen3Report,
  planQwen3,
  qwen3ConfigKey,
  qwen3HelperConfig,
  qwen3Language,
  qwen3SettingsProblem,
  qwen3SizeOf,
  qwen3SpeechSettings,
  qwen3TalkerId,
  qwen3Ui,
  type Qwen3Recommendation,
} from "../services/qwen3";
import { ITTSReceiver, TTS_OPTIONS_MISSING, TtsSinkContext, ttsOutputMode } from "../types";

const invokeMock = vi.mocked(invoke);
const isTauriMock = vi.mocked(isTauri);
const listenMock = vi.mocked(listen);

const ALL = [
  "tts/qwen3-tts-1.7b",
  "tts/qwen3-tts-1.7b-base",
  "tts/qwen3-tts-1.7b-voicedesign",
  "tts/qwen3-tts-0.6b",
  "tts/qwen3-tts-0.6b-base",
  QWEN3_TOKENIZER_ID,
];

const rec = (choice: string, alternatives: string[] = [], device = "gpu"): Qwen3Recommendation => ({ choice, alternatives, device, reason_code: "fits" });
const has = (ids: string[]) => (id: string) => ids.includes(id);
const settings = (patch: Partial<TTS_State["qwen3"]> = {}) => ({ ...TTS_Qwen3_Schema.parse({}), ...patch });

beforeAll(async () => {
  await i18n.init({ lng: "en", resources: { en: { translation: {} } }, interpolation: { escapeValue: false } });
});

describe("plan (mirror of Rust tts_helpers::qwen::plan)", () => {
  it("maps size and mode to the talker checkpoints", () => {
    expect(qwen3TalkerId("1.7b", "preset")).toBe("tts/qwen3-tts-1.7b");
    expect(qwen3TalkerId("1.7b", "clone")).toBe("tts/qwen3-tts-1.7b-base");
    expect(qwen3TalkerId("1.7b", "design")).toBe("tts/qwen3-tts-1.7b-voicedesign");
    expect(qwen3TalkerId("0.6b", "preset")).toBe("tts/qwen3-tts-0.6b");
    expect(qwen3TalkerId("0.6b", "clone")).toBe("tts/qwen3-tts-0.6b-base");
    expect(qwen3TalkerId("0.6b", "design")).toBeNull();
    expect(qwen3SizeOf("tts/qwen3-tts-0.6b-base")).toBe("0.6b");
    expect(qwen3SizeOf("tts/qwen3-tts-1.7b-voicedesign")).toBe("1.7b");
    expect(qwen3SizeOf("tts/kokoro-v1.0")).toBeNull();
  });

  it("follows the governor for auto and prefers installed alternatives", () => {
    const env = { installed: has([...ALL, QWEN3_RUNTIME_IDS.vulkan]), cudaUsable: false };
    let p = planQwen3(settings(), { ...env, recommendation: rec("tts/qwen3-tts-1.7b", ["tts/qwen3-tts-0.6b"]) });
    expect([p.size, p.talker, p.variant, p.instructions]).toEqual(["1.7b", "tts/qwen3-tts-1.7b", "vulkan", true]);
    expect(p.ids).toEqual(["tts/qwen3-tts-1.7b", QWEN3_TOKENIZER_ID, QWEN3_RUNTIME_IDS.vulkan]);
    p = planQwen3(settings({ mode: "clone" }), { ...env, recommendation: rec("tts/qwen3-tts-0.6b") });
    expect([p.size, p.talker, p.instructions]).toEqual(["0.6b", "tts/qwen3-tts-0.6b-base", false]);
    // 1.7B recommended, only 0.6B installed and accepted: 0.6B.
    const small = { installed: has(["tts/qwen3-tts-0.6b", QWEN3_TOKENIZER_ID]), cudaUsable: false };
    expect(planQwen3(settings(), { ...small, recommendation: rec("tts/qwen3-tts-1.7b", ["tts/qwen3-tts-0.6b"]) }).size).toBe("0.6b");
    expect(planQwen3(settings(), { ...small, recommendation: rec("tts/qwen3-tts-1.7b") }).size).toBe("1.7b");
    // Explicit sizes and design.
    expect(planQwen3(settings({ model: "1.7b" }), { ...env, recommendation: rec("tts/qwen3-tts-0.6b") }).size).toBe("1.7b");
    const design = planQwen3(settings({ mode: "design", model: "0.6b" }), { ...env, recommendation: rec("tts/qwen3-tts-0.6b") });
    expect([design.size, design.talker, design.designForcedLarge, design.instructions]).toEqual(["1.7b", "tts/qwen3-tts-1.7b-voicedesign", true, false]);
  });

  it("picks the runtime pack like Rust", () => {
    const r = rec("tts/qwen3-tts-1.7b");
    const run = (installed: string[], cudaUsable: boolean, variant: "auto" | "vulkan" | "cuda13" | "cpu" = "auto", recommendation = r) =>
      planQwen3(settings({ variant }), { recommendation, installed: has(installed), cudaUsable }).variant;
    expect(run([QWEN3_RUNTIME_IDS.cuda13, NVIDIA_RUNTIME_ID, QWEN3_RUNTIME_IDS.vulkan], true)).toBe("cuda13");
    expect(run([QWEN3_RUNTIME_IDS.cuda13, QWEN3_RUNTIME_IDS.vulkan], true)).toBe("vulkan");
    expect(run([QWEN3_RUNTIME_IDS.cuda13], true)).toBe("cuda13");
    // The NVIDIA runtime is installed already (voice changer): the CUDA server pack is the one to add.
    expect(run([NVIDIA_RUNTIME_ID], true)).toBe("cuda13");
    expect(run([NVIDIA_RUNTIME_ID], false)).toBe("vulkan");
    expect(run([QWEN3_RUNTIME_IDS.cuda13, NVIDIA_RUNTIME_ID], false)).toBe("vulkan");
    expect(run([], true)).toBe("vulkan");
    expect(run([], false, "auto", rec("tts/qwen3-tts-0.6b", [], "cpu"))).toBe("cpu");
    expect(run([], true, "cpu")).toBe("cpu");
    expect(run([QWEN3_RUNTIME_IDS.cuda13, NVIDIA_RUNTIME_ID], true, "vulkan")).toBe("vulkan");
    // The CUDA pack needs the NVIDIA runtime too (Rust lists it last among the missing ids).
    const ids = (installed: string[], cudaUsable: boolean) => planQwen3(settings({}), { recommendation: r, installed: has(installed), cudaUsable }).ids;
    expect(ids([NVIDIA_RUNTIME_ID], true)).toEqual(["tts/qwen3-tts-1.7b", QWEN3_TOKENIZER_ID, QWEN3_RUNTIME_IDS.cuda13, NVIDIA_RUNTIME_ID]);
    expect(ids([], false)).toEqual(["tts/qwen3-tts-1.7b", QWEN3_TOKENIZER_ID, QWEN3_RUNTIME_IDS.vulkan]);
  });

  it("maps languages like Rust and falls back to auto", () => {
    for (const [tag, name] of [
      ["en", "english"],
      ["EN-us", "english"],
      ["ja", "japanese"],
      ["zh-Hant", "chinese"],
      ["yue", "chinese"],
      ["pt_BR", "portuguese"],
      ["Japanese", "japanese"],
      ["auto", "auto"],
      ["", "auto"],
      ["ar", "auto"],
    ] as const) {
      expect(qwen3Language(tag), tag).toBe(name);
    }
    expect(qwen3Language(null)).toBe("auto");
  });
});

describe("requests", () => {
  const endpoint = (model: string): RustTtsHelperEndpoint => ({ engine: "qwen3", baseUrl: "http://127.0.0.1:5123/v1", apiKey: "k".repeat(64), model, pcmSampleRate: 24_000 });

  it("sends style instructions only to the 1.7B built-in voices", () => {
    const s = settings({ instruction: "Speak in a happy, cheerful tone." });
    const big = qwen3SpeechSettings(endpoint("tts/qwen3-tts-1.7b"), s, "ja", "serena");
    expect(big).toMatchObject({ baseUrl: "http://127.0.0.1:5123/v1", voice: "serena", responseFormat: "pcm", stream: true, pcmSampleRate: 24_000, speed: 1, model: "" });
    expect(big.extra).toEqual({ language: "japanese", instructions: "Speak in a happy, cheerful tone." });
    expect(qwen3SpeechSettings(endpoint("tts/qwen3-tts-0.6b"), s, "ja", "serena").extra).toEqual({ language: "japanese" });
  });

  it("speaks a cloned voice without instructions and a designed voice with its description and a fixed seed", () => {
    const clone = qwen3SpeechSettings(endpoint("tts/qwen3-tts-1.7b-base"), settings({ mode: "clone", instruction: "ignored" }), "en", "sigil-alice-1a2b3c");
    expect(clone.voice).toBe("sigil-alice-1a2b3c");
    expect(clone.extra).toEqual({ language: "english" });
    const design = qwen3SpeechSettings(endpoint("tts/qwen3-tts-1.7b-voicedesign"), settings({ mode: "design", voiceDescription: "  Old wizard, raspy and slow.  " }), null, "");
    expect(design.extra).toEqual({ language: "auto", instructions: "Old wizard, raspy and slow.", seed: QWEN3_DESIGN_SEED });
    const seeded = qwen3SpeechSettings(endpoint("tts/qwen3-tts-1.7b"), settings({ seed: "42", language: "de" }), "ja", "ryan");
    expect(seeded.extra).toEqual({ language: "german", seed: 42 });
    expect(qwen3SpeechSettings(endpoint("tts/qwen3-tts-1.7b"), settings({ seed: "abc" }), "en", "ryan").extra).toEqual({ language: "english" });
  });

  it("names what is missing and keys the launch", () => {
    expect(qwen3SettingsProblem(settings())).toBeNull();
    expect(qwen3SettingsProblem(settings({ mode: "clone" }))).toBe("tts.qwen3_clone_voice_missing");
    expect(qwen3SettingsProblem(settings({ mode: "design" }))).toBe("tts.qwen3_design_description_missing");
    const c = qwen3HelperConfig(settings({ language: "ja-JP", voice: "sohee" }));
    expect(c).toEqual({ model: "auto", variant: "auto", mode: "preset", language: "japanese" });
    // Voice, language, instruction and seed apply per sentence: they never reload the helper.
    expect(qwen3ConfigKey(qwen3HelperConfig(settings({ voice: "ryan", instruction: "x", seed: "1" })))).toBe(qwen3ConfigKey(qwen3HelperConfig(settings())));
    expect(qwen3ConfigKey(qwen3HelperConfig(settings({ language: "ja" })))).toBe(qwen3ConfigKey(qwen3HelperConfig(settings({ language: "auto" }))));
    expect(qwen3ConfigKey(qwen3HelperConfig(settings({ mode: "clone" })))).not.toBe(qwen3ConfigKey(qwen3HelperConfig(settings())));
    expect(qwen3ConfigKey(qwen3HelperConfig(settings({ model: "0.6b" })))).not.toBe(qwen3ConfigKey(qwen3HelperConfig(settings())));
    expect(qwen3ConfigKey(qwen3HelperConfig(settings({ variant: "cpu" })))).not.toBe(qwen3ConfigKey(qwen3HelperConfig(settings())));
  });

  it("keeps the speaker list and the report stats", () => {
    expect(QWEN3_SPEAKERS.map((s) => s.id)).toEqual(["ryan", "aiden", "vivian", "serena", "uncle_fu", "dylan", "eric", "ono_anna", "sohee"]);
    expect(noteQwen3Report({ sampleRate: 24_000, frames: 36_000, firstAudioMs: 182.4, totalMs: 1500 })).toMatchObject({ firstAudioMs: 182, audioSec: 1.5, totalMs: 1500 });
    expect(qwen3Ui.last?.audioSec).toBe(1.5);
    expect(noteQwen3Report({ sampleRate: 0, frames: 0, firstAudioMs: 0, totalMs: 0 })).toBeNull();
  });
});

// -------------------------------------------------------------------------------------------------
// The backend
// -------------------------------------------------------------------------------------------------

const ENDPOINT: RustTtsHelperEndpoint = { engine: "qwen3", baseUrl: "http://127.0.0.1:5123/v1", apiKey: "a".repeat(64), model: "tts/qwen3-tts-1.7b", pcmSampleRate: 24_000 };
const REPORT = { sampleRate: 24_000, frames: 24_000, firstAudioMs: 150, totalMs: 900 };

let data: TTS_State;
let recommendations: Record<string, Qwen3Recommendation | undefined>;
let helperListener: ((e: { payload: RustTtsHelperStatus }) => void) | null;

function receiver() {
  return { onStart: vi.fn(), onStop: vi.fn(), onError: vi.fn() } satisfies ITTSReceiver;
}

const flush = async (n = 20) => {
  for (let i = 0; i < n; i++) await Promise.resolve();
};

function calls(cmd: string) {
  return invokeMock.mock.calls.filter(([c]) => c === cmd).map(([, args]) => args as Record<string, unknown>);
}

const sinkCtx = (id: string, lang: string | null = "en"): TtsSinkContext => ({ utteranceId: id, target: { route: "device", device: "", volume: 1 }, lang });

/** Default IPC answers: the helper starts, voices register, sentences speak. */
function answer(overrides: Record<string, (args: Record<string, unknown>) => unknown> = {}) {
  invokeMock.mockImplementation(async (cmd: string, args?: unknown) => {
    const a = (args ?? {}) as Record<string, unknown>;
    if (overrides[cmd]) return overrides[cmd](a);
    switch (cmd) {
      case "plugin:tts-helpers|tts_helper_start":
        return ENDPOINT;
      case "plugin:tts-helpers|qwen_tts_prepare_clone":
        return { voiceId: a.voiceId, voice: `sigil-${a.voiceId}`, cached: true };
      case "plugin:openai-speech|openai_speech_speak":
        return REPORT;
      default:
        return undefined;
    }
  });
}

beforeEach(() => {
  invokeMock.mockReset();
  isTauriMock.mockReturnValue(true);
  helperListener = null;
  listenMock.mockImplementation((async (_event: string, cb: (e: { payload: RustTtsHelperStatus }) => void) => {
    helperListener = cb;
    return () => {
      helperListener = null;
    };
  }) as unknown as typeof listen);
  data = proxy(Service_TTS_Schema.parse({ backend: "qwen3" }));
  recommendations = proxy({ qwen3_tts: rec("tts/qwen3-tts-1.7b", ["tts/qwen3-tts-0.6b"]) });
  vi.stubGlobal("window", {
    ApiServer: {
      state: { services: { tts: { data } }, audioOutputDevice: "" },
      resources: { state: { recommendations } },
      models: { state: { states: {} } },
      sysinfo: { state: { info: null } },
    },
  });
  answer();
});

afterEach(() => {
  vi.unstubAllGlobals();
  vi.useRealTimers();
});

describe("TTS_Qwen3Service", () => {
  it("is a sink backend registered as qwen3", () => {
    expect(TTS_BACKENDS.qwen3).toBe(TTS_Qwen3Service);
    expect(ttsOutputMode(new TTS_Qwen3Service(receiver()))).toBe("sink");
  });

  it("starts the helper and streams sentences into the sink", async () => {
    const r = receiver();
    const s = new TTS_Qwen3Service(r);
    await s.start(data);
    expect(r.onStart).toHaveBeenCalledTimes(1);
    expect(calls("plugin:tts-helpers|tts_helper_start")).toEqual([{ engine: "qwen3", config: { model: "auto", variant: "auto", mode: "preset", language: "auto" } }]);

    data.qwen3.voice = "sohee";
    data.qwen3.instruction = "Speak calmly and gently, at a relaxed pace.";
    await s.synthesizeToSink("こんにちは。", sinkCtx("u1", "ja"), new AbortController().signal);
    const [speak] = calls("plugin:openai-speech|openai_speech_speak");
    expect(speak.utteranceId).toBe("u1");
    expect(speak.request).toMatchObject({
      baseUrl: ENDPOINT.baseUrl,
      apiKey: ENDPOINT.apiKey,
      input: "こんにちは。",
      voice: "sohee",
      responseFormat: "pcm",
      extra: { language: "japanese", instructions: "Speak calmly and gently, at a relaxed pace." },
    });
    // Voice and instruction changes did not reload the helper.
    expect(calls("plugin:tts-helpers|tts_helper_start")).toHaveLength(1);
    expect(qwen3Ui.last).toMatchObject({ firstAudioMs: 150, audioSec: 1 });

    // Neither does a language change: the next sentence names the new language in its request.
    data.qwen3.language = "de";
    await s.synthesizeToSink("Hallo.", sinkCtx("u2", "ja"), new AbortController().signal);
    expect(calls("plugin:tts-helpers|tts_helper_start")).toHaveLength(1);
    expect(calls("plugin:openai-speech|openai_speech_speak")[1].request).toMatchObject({ input: "Hallo.", extra: { language: "german" } });
  });

  it("refuses without the desktop app or a voice to clone", async () => {
    isTauriMock.mockReturnValue(false);
    let r = receiver();
    await new TTS_Qwen3Service(r).start(data);
    expect(r.onStop).toHaveBeenCalledWith("tts.qwen3_desktop_only");
    isTauriMock.mockReturnValue(true);
    data.qwen3.mode = "clone";
    r = receiver();
    await new TTS_Qwen3Service(r).start(data);
    expect(r.onStop.mock.calls[0][0]).toContain(TTS_OPTIONS_MISSING);
    expect(calls("plugin:tts-helpers|tts_helper_start")).toHaveLength(0);
  });

  it("registers the library voice at start and before each sentence", async () => {
    data.qwen3.mode = "clone";
    data.qwen3.referenceVoiceId = "alice-1a2b3c";
    const r = receiver();
    const s = new TTS_Qwen3Service(r);
    await s.start(data);
    expect(r.onStart).toHaveBeenCalled();
    expect(calls("plugin:tts-helpers|qwen_tts_prepare_clone")).toEqual([{ voiceId: "alice-1a2b3c" }]);
    await s.synthesizeToSink("Hello.", sinkCtx("u1"), new AbortController().signal);
    expect(calls("plugin:tts-helpers|qwen_tts_prepare_clone")).toHaveLength(2);
    expect(calls("plugin:openai-speech|openai_speech_speak")[0].request).toMatchObject({ voice: "sigil-alice-1a2b3c", extra: { language: "english" } });
  });

  it("reports a failed start with Rust's reason", async () => {
    answer({ "plugin:tts-helpers|tts_helper_start": () => Promise.reject("Qwen3-TTS needs downloads first: tts/qwen3-tts-1.7b") });
    const r = receiver();
    await new TTS_Qwen3Service(r).start(data);
    expect(r.onStart).not.toHaveBeenCalled();
    expect(r.onStop).toHaveBeenCalledWith("tts.qwen3_start_failed");
  });

  it("reloads the helper when the model settings change, then finds it again after a crash", async () => {
    const r = receiver();
    const s = new TTS_Qwen3Service(r);
    await s.start(data);
    data.qwen3.mode = "design";
    data.qwen3.voiceDescription = "Old wizard.";
    await s.synthesizeToSink("One.", sinkCtx("u1"), new AbortController().signal);
    const starts = calls("plugin:tts-helpers|tts_helper_start");
    expect(starts).toHaveLength(2);
    expect(starts[1].config).toMatchObject({ mode: "design" });
    // The helper restarts by itself (crash): the next sentence asks for the endpoint again.
    helperListener?.({ payload: { engine: "qwen3", state: "starting", model: "tts/qwen3-tts-1.7b-voicedesign", error: null } });
    await s.synthesizeToSink("Two.", sinkCtx("u2"), new AbortController().signal);
    expect(calls("plugin:tts-helpers|tts_helper_start")).toHaveLength(3);
    await s.synthesizeToSink("Three.", sinkCtx("u3"), new AbortController().signal);
    expect(calls("plugin:tts-helpers|tts_helper_start")).toHaveLength(3);
    // A permanent failure is reported, and the next sentence tries again.
    helperListener?.({ payload: { engine: "qwen3", state: "failed", model: null, error: "exit code 3" } });
    expect(r.onError).toHaveBeenCalledWith("tts.qwen3_helper_failed");
  });

  it("reloads once the sentence streaming from the old helper ends, never waiting for the ones queued behind it", async () => {
    vi.useFakeTimers();
    let finishFirst: (v: unknown) => void = () => {};
    let speaks = 0;
    answer({
      "plugin:openai-speech|openai_speech_speak": () => {
        speaks += 1;
        return speaks === 1 ? new Promise((res) => (finishFirst = res)) : REPORT;
      },
    });
    const r = receiver();
    const s = new TTS_Qwen3Service(r);
    await s.start(data);
    const one = s.synthesizeToSink("One.", sinkCtx("u1"), new AbortController().signal);
    await flush();
    expect(speaks).toBe(1);
    // The mode changes while "One." streams; the pipeline already asks for the next two sentences.
    data.qwen3.mode = "design";
    data.qwen3.voiceDescription = "Old wizard.";
    const two = s.synthesizeToSink("Two.", sinkCtx("u2"), new AbortController().signal);
    const three = s.synthesizeToSink("Three.", sinkCtx("u3"), new AbortController().signal);
    await vi.advanceTimersByTimeAsync(1_000);
    // Nothing reloads while "One." still streams from the loaded helper.
    expect(calls("plugin:tts-helpers|tts_helper_start")).toHaveLength(1);
    finishFirst(REPORT);
    await one;
    // The reload follows within a poll interval, not after the 30 s bound.
    await vi.advanceTimersByTimeAsync(100);
    await Promise.all([two, three]);
    const starts = calls("plugin:tts-helpers|tts_helper_start");
    expect(starts).toHaveLength(2);
    expect(starts[1].config).toMatchObject({ mode: "design" });
    const requests = calls("plugin:openai-speech|openai_speech_speak").map((c) => c.request as { input: string; extra: Record<string, unknown> });
    expect(requests.map((q) => q.input)).toEqual(["One.", "Two.", "Three."]);
    expect(requests[1].extra).toMatchObject({ instructions: "Old wizard." });
    expect(r.onError).not.toHaveBeenCalled();
    s.stop();
  });

  it("fails one sentence on a connection error and starts the helper again for the next", async () => {
    const r = receiver();
    const s = new TTS_Qwen3Service(r);
    await s.start(data);
    let fail = true;
    answer({
      "plugin:openai-speech|openai_speech_speak": () => {
        if (fail) {
          fail = false;
          return Promise.reject("Speech request failed: connection refused");
        }
        return REPORT;
      },
    });
    await expect(s.synthesizeToSink("One.", sinkCtx("u1"), new AbortController().signal)).rejects.toThrow("Qwen3-TTS: Speech request failed: connection refused");
    await s.synthesizeToSink("Two.", sinkCtx("u2"), new AbortController().signal);
    expect(calls("plugin:tts-helpers|tts_helper_start")).toHaveLength(2);
  });

  it("swaps the size when the governor changes its mind and the other size is installed", async () => {
    vi.useFakeTimers();
    const states: Record<string, { state: string }> = {};
    for (const id of [...ALL, QWEN3_RUNTIME_IDS.vulkan]) states[id] = { state: "installed" };
    (window as unknown as { ApiServer: { models: { state: { states: unknown } } } }).ApiServer.models.state.states = states;
    const r = receiver();
    const s = new TTS_Qwen3Service(r);
    await s.start(data);
    expect(calls("plugin:tts-helpers|tts_helper_start")).toHaveLength(1);
    // The voice changer started on an 8 GB card.
    recommendations.qwen3_tts = rec("tts/qwen3-tts-0.6b");
    await flush();
    await vi.advanceTimersByTimeAsync(2_500);
    await flush();
    expect(calls("plugin:tts-helpers|tts_helper_start")).toHaveLength(2);
    expect(qwen3Ui.swapBlocked).toBeNull();
    s.stop();
  });

  it("keeps the loaded size when the recommended one is not downloaded", async () => {
    vi.useFakeTimers();
    const r = receiver();
    const s = new TTS_Qwen3Service(r);
    await s.start(data);
    recommendations.qwen3_tts = rec("tts/qwen3-tts-0.6b");
    await flush();
    await vi.advanceTimersByTimeAsync(2_500);
    await flush();
    expect(calls("plugin:tts-helpers|tts_helper_start")).toHaveLength(1);
    expect(qwen3Ui.swapBlocked).toBe("tts/qwen3-tts-0.6b");
    s.stop();
  });

  it("stops the helper once and ignores a start that finishes after stop", async () => {
    let release: (v: RustTtsHelperEndpoint) => void = () => {};
    answer({ "plugin:tts-helpers|tts_helper_start": () => new Promise<RustTtsHelperEndpoint>((res) => (release = res)) });
    const r = receiver();
    const s = new TTS_Qwen3Service(r);
    const started = s.start(data);
    s.stop();
    expect(r.onStop).toHaveBeenCalledTimes(1);
    expect(r.onStop).toHaveBeenCalledWith();
    expect(calls("plugin:tts-helpers|tts_helper_stop")).toEqual([{ engine: "qwen3" }]);
    release(ENDPOINT);
    await started;
    expect(r.onStart).not.toHaveBeenCalled();
    s.dispose();
    expect(r.onStop).toHaveBeenCalledTimes(1);
    expect(calls("plugin:tts-helpers|tts_helper_stop")).toHaveLength(1);
    // Nothing speaks after a stop.
    await s.synthesizeToSink("x", sinkCtx("u9"), new AbortController().signal);
    expect(calls("plugin:openai-speech|openai_speech_speak")).toHaveLength(0);
  });

  it("cancels the stream when the sentence is skipped and never reports an aborted sentence", async () => {
    let finish: (v: unknown) => void = () => {};
    answer({ "plugin:openai-speech|openai_speech_speak": () => new Promise((_, rej) => (finish = rej)) });
    const r = receiver();
    const s = new TTS_Qwen3Service(r);
    await s.start(data);
    const ac = new AbortController();
    const p = s.synthesizeToSink("Long sentence.", sinkCtx("u1"), ac.signal);
    await flush();
    ac.abort();
    expect(calls("plugin:openai-speech|openai_speech_cancel")).toEqual([{ utteranceId: "u1" }]);
    finish("cancelled");
    await expect(p).resolves.toBeUndefined();
  });
});
