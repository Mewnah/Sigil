import i18n from "i18next";
import { proxy } from "valtio";
import { afterEach, beforeAll, beforeEach, describe, expect, it, vi } from "vitest";

vi.mock("@tauri-apps/api/core", () => ({ invoke: vi.fn(), isTauri: vi.fn(() => true) }));
vi.mock("@tauri-apps/api/event", () => ({ listen: vi.fn(async () => () => {}) }));
vi.mock("@/utils/devLog", () => ({ devLog: () => {} }));

import { invoke, isTauri } from "@tauri-apps/api/core";
import { listen } from "@tauri-apps/api/event";
import type { RustTtsHelperEndpoint, RustTtsHelperStatus } from "@/types";
import catalogFragment from "../../../../../src-tauri/model-catalog/tts-ggml-experimental.json";
import { Service_TTS_Schema, TTS_State } from "../schema";
import { GGML_TTS_ENGINES, TTS_GgmlTts_Schema, type TTS_GgmlTts_State } from "../schema/ggmlTts";
import { TTS_BACKENDS } from "../index";
import {
  GGML_ENGINES,
  GGML_LOCAL_ONLY_IDS,
  GGML_PREVIEW_IDLE_MS,
  MAGPIE_SPEAKERS,
  NVIDIA_RUNTIME_ID,
  OMNIVOICE_CONSENT_TEXT,
  OMNIVOICE_DISCLOSURE_TEXT,
  SerialGate,
  TTS_GgmlTtsService,
  ggmlEngine,
  ggmlHelperConfig,
  ggmlLanguageAfterChange,
  ggmlLanguageChoices,
  ggmlLanguageUsable,
  ggmlLaunchKey,
  ggmlModel,
  ggmlPreviewIdleEngines,
  ggmlRetryable,
  ggmlSeed,
  ggmlSentenceLanguage,
  ggmlSettingsProblem,
  ggmlSpeechSettings,
  ggmlUi,
  languageBase,
  noteGgmlReport,
  planGgml,
  previewGgml,
  startGgmlHelper,
  stopGgmlHelper,
  type GgmlGpu,
} from "../services/ggmlTts";
import { ITTSReceiver, TTS_OPTIONS_MISSING, TtsSinkContext, ttsOutputMode } from "../types";

const invokeMock = vi.mocked(invoke);
const isTauriMock = vi.mocked(isTauri);
const listenMock = vi.mocked(listen);

const settings = (patch: Partial<TTS_GgmlTts_State> = {}): TTS_GgmlTts_State => ({ ...TTS_GgmlTts_Schema.parse({}), ...patch });
const has = (ids: string[]) => (id: string) => ids.includes(id);
const NVIDIA: GgmlGpu = { anyGpu: true, cuda: true };
const AMD: GgmlGpu = { anyGpu: true, cuda: false };
const NONE: GgmlGpu = { anyGpu: false, cuda: false };

beforeAll(async () => {
  await i18n.init({ lng: "en", resources: { en: { translation: {} } }, interpolation: { escapeValue: false } });
});

describe("schema", () => {
  it("defaults off and repairs bad values", () => {
    const s = TTS_GgmlTts_Schema.parse({});
    expect(s).toMatchObject({ engine: "magpie", model: "", variant: "auto", quality: "high", lowVram: false, cloneConsent: false, discloseAi: false });
    const bad = TTS_GgmlTts_Schema.parse({ engine: "zonos", variant: "metal", quality: "ultra" });
    expect([bad.engine, bad.variant, bad.quality]).toEqual(["magpie", "auto", "high"]);
    expect(Service_TTS_Schema.parse({ backend: "ggmlTts" }).ggmlTts.engine).toBe("magpie");
  });
});

describe("engine facts (mirror of Rust tts_helpers::{magpie,breeze,omnivoice,s2})", () => {
  const catalog = new Map((catalogFragment.entries as { id: string; status: string; requires?: string[]; files: { size: number; sha256?: string }[] }[]).map((e) => [e.id, e]));

  it("names only catalog entries, and every published one is pinned", () => {
    for (const engine of GGML_TTS_ENGINES) {
      const facts = GGML_ENGINES[engine];
      expect(facts.id).toBe(engine);
      for (const id of [...facts.packs.map((p) => p.runtime), ...facts.models.flatMap((m) => [m.id, ...m.companions])]) {
        expect(catalog.has(id), id).toBe(true);
      }
      // A model's companions are its catalog `requires`, so <ModelRequirement> and Rust agree.
      for (const m of facts.models) {
        for (const c of m.companions) expect(catalog.get(m.id)?.requires ?? [], `${m.id} requires ${c}`).toContain(c);
      }
    }
    for (const id of GGML_LOCAL_ONLY_IDS) expect(catalog.get(id)?.status, id).toBe("unpublished");
    for (const e of catalog.values()) {
      if (e.status !== "published") continue;
      for (const f of e.files) expect(!!f.sha256 && f.size > 0, e.id).toBe(true);
    }
  });

  it("picks models by key, falling back to the engine's default", () => {
    expect(ggmlModel(GGML_ENGINES.magpie, "").key).toBe("v2602");
    expect(ggmlModel(GGML_ENGINES.magpie, "V2607").key).toBe("v2607");
    expect(ggmlModel(GGML_ENGINES.breeze, "v2607").key).toBe("q8_0");
    expect(ggmlModel(GGML_ENGINES.s2, "tts/s2-pro-q8_0").key).toBe("q8_0");
    expect(ggmlEngine("nope").id).toBe("magpie");
  });
});

describe("plan (mirror of Rust launch::pick_pack)", () => {
  const crisp = GGML_ENGINES.omnivoice.packs;
  it("follows the same pack order as Rust", () => {
    const s = settings({ engine: "omnivoice" });
    expect(planGgml(s, { installed: has([]), gpu: NVIDIA }).pack.variant).toBe("vulkan");
    expect(planGgml(s, { installed: has([]), gpu: NONE }).pack.variant).toBe("cpu");
    expect(planGgml(s, { installed: has([NVIDIA_RUNTIME_ID]), gpu: NVIDIA }).pack.variant).toBe("cuda13");
    expect(planGgml(s, { installed: has([crisp[1].runtime, NVIDIA_RUNTIME_ID]), gpu: NVIDIA }).pack.variant).toBe("vulkan");
    expect(planGgml(s, { installed: has([crisp[0].runtime, crisp[1].runtime, NVIDIA_RUNTIME_ID]), gpu: NVIDIA }).pack.variant).toBe("cuda13");
    expect(planGgml(s, { installed: has([crisp[0].runtime, NVIDIA_RUNTIME_ID]), gpu: AMD }).pack.variant).toBe("vulkan");
  });

  it("lists every download, with the NVIDIA runtime for CUDA 13, and refuses impossible choices", () => {
    let p = planGgml(settings({ engine: "s2", variant: "cuda13" }), { installed: has([]), gpu: NVIDIA });
    expect(p.ids).toEqual(["tts/s2-pro-q4_k_m", "tts/s2-pro-tokenizer", "runtime/s2-server-cuda13", NVIDIA_RUNTIME_ID]);
    expect(p.refused).toBeNull();
    p = planGgml(settings({ engine: "magpie", model: "v2607" }), { installed: has([]), gpu: AMD });
    expect(p.ids).toEqual(["tts/magpie-tts-357m-v2607", "tts/magpie-tts-357m-v2607-tokenizer", "tts/nemo-nano-codec-22khz", "runtime/nemo-speech-vulkan"]);
    expect(planGgml(settings({ engine: "breeze", variant: "cuda13" }), { installed: has([]), gpu: NVIDIA }).refused).toBe("tts.ggml_variant_missing");
    expect(planGgml(settings({ engine: "magpie", variant: "cuda" }), { installed: has([]), gpu: AMD }).refused).toBe("tts.ggml_variant_needs_cuda");
    expect(planGgml(settings({ engine: "magpie", variant: "vulkan" }), { installed: has([]), gpu: NONE }).refused).toBe("tts.ggml_variant_needs_gpu");
    // Breeze's processor variant is its Vulkan pack started with --cpu.
    p = planGgml(settings({ engine: "breeze" }), { installed: has([]), gpu: NONE });
    expect([p.pack.variant, p.pack.runtime]).toEqual(["cpu", "runtime/breeze-server-vulkan"]);
  });
});

describe("languages, voices and requests", () => {
  const endpoint = { baseUrl: "http://127.0.0.1:5000/v1", apiKey: "k".repeat(64), pcmSampleRate: 22_050 };

  it("maps languages per engine", () => {
    expect(languageBase("pt-BR")).toBe("pt");
    expect(languageBase("zh_Hant")).toBe("zh");
    expect(languageBase("auto")).toBeNull();
    const magpie = settings({ engine: "magpie" });
    expect(ggmlSentenceLanguage(magpie, "de")).toBe("de-DE");
    expect(ggmlSentenceLanguage(magpie, null)).toBe("en-US");
    expect(() => ggmlSentenceLanguage(magpie, "ko")).toThrow("tts.ggml_magpie_language_unsupported");
    expect(ggmlSentenceLanguage({ ...magpie, model: "v2607" }, "ko-KR")).toBe("ko-KR");
    expect(ggmlSentenceLanguage({ ...magpie, model: "v2607" }, "ar")).toBe("ar-MSA");
    expect(ggmlSentenceLanguage({ ...magpie, language: "fr" }, "de")).toBe("fr-FR");
    expect(() => ggmlSentenceLanguage(magpie, "ja")).toThrow();
    expect(ggmlSentenceLanguage(settings({ engine: "omnivoice" }), "sw-KE")).toBe("sw");
    expect(ggmlSentenceLanguage(settings({ engine: "omnivoice" }), null)).toBeUndefined();
    expect(ggmlSentenceLanguage(settings({ engine: "breeze" }), "en")).toBeUndefined();
    expect(ggmlLanguageChoices(settings({ engine: "magpie" }))).toEqual(["en", "es", "de", "fr", "it", "vi", "hi"]);
    expect(ggmlLanguageChoices(settings({ engine: "magpie", model: "v2607" }))).toContain("ko");
    expect(ggmlLanguageChoices(settings({ engine: "s2" }))).toBeNull();
  });

  it("checks the settings", () => {
    expect(ggmlSettingsProblem(settings())).toBeNull();
    expect(ggmlSettingsProblem(settings({ voice: "alloy" }))).toBe("tts.ggml_magpie_voice_unknown");
    expect(ggmlSettingsProblem(settings({ engine: "omnivoice", referenceVoiceId: "alice-1" }))).toBe("tts.ggml_omnivoice_consent_missing");
    expect(ggmlSettingsProblem(settings({ engine: "omnivoice", referenceVoiceId: "alice-1", cloneConsent: true }))).toBeNull();
    // Magpie does not clone: a stale library voice is ignored.
    expect(ggmlSettingsProblem(settings({ engine: "magpie", referenceVoiceId: "alice-1" }))).toBeNull();
  });

  it("resets, or reports, a language the engine or checkpoint cannot speak", () => {
    const ko = settings({ engine: "magpie", model: "v2607", language: "ko" });
    expect(ggmlLanguageUsable(ko)).toBe(true);
    expect(ggmlSettingsProblem(ko)).toBeNull();
    // v2607 -> v2602: Korean goes back to automatic (the panel), or is named as the problem (older settings).
    expect(ggmlLanguageAfterChange({ ...ko, model: "v2602" })).toBe("auto");
    expect(ggmlSettingsProblem({ ...ko, model: "v2602" })).toBe("tts.ggml_magpie_language_unsupported");
    expect(ggmlSettingsProblem(settings({ language: "ja" }))).toBe("tts.ggml_magpie_language_unsupported");
    // OmniVoice speaks any language; Breeze and S2 ignore the setting.
    const sw = settings({ engine: "omnivoice", language: "sw" });
    expect(ggmlLanguageAfterChange(sw)).toBe("sw");
    expect(ggmlLanguageAfterChange({ ...ko, engine: "omnivoice", model: "" })).toBe("ko");
    expect(ggmlLanguageAfterChange({ ...sw, engine: "magpie" })).toBe("auto");
    expect(ggmlSettingsProblem(settings({ engine: "breeze", language: "ko" }))).toBeNull();
    expect(ggmlLanguageAfterChange(settings({ engine: "s2", language: "ko" }))).toBe("ko");
    expect(ggmlLanguageAfterChange(settings({ language: "auto" }))).toBe("auto");
  });

  it("sends a failed sentence again only when nobody heard any of it", () => {
    const refused = "Speech request (http://127.0.0.1:5123/v1/audio/speech): cannot connect. Is the server running? (os error 10061)";
    const adapter502 = "Speech request: HTTP 502 Bad Gateway: the voice engine is not running or still loading (cannot connect after 90 s): refused";
    const broke = "Speech request (http://127.0.0.1:5123/v1/audio/speech) failed: connection closed before message completed";
    for (const m of [refused, adapter502]) {
      expect(ggmlRetryable(m, false), m).toBe(true);
      expect(ggmlRetryable(new Error(m), true), m).toBe(true);
    }
    // A stream that broke off may have played its start already; a whole clip had not begun.
    expect(ggmlRetryable(broke, false)).toBe(false);
    expect(ggmlRetryable(broke, true)).toBe(true);
    expect(ggmlRetryable("Speech request: HTTP 503 Service Unavailable: the voice engine is busy with another sentence", true)).toBe(false);
    expect(ggmlRetryable("Speech request: the server stopped sending audio for 20 s (stalled).", true)).toBe(false);
  });

  it("keeps per-sentence settings out of the launch key (voice and language are never launch arguments)", () => {
    const base = settings({ engine: "magpie", voice: "Sofia", language: "de" });
    const a = ggmlHelperConfig(base);
    expect(a).toEqual({ model: "v2602", variant: "auto", language: "de", voice: "Sofia", referenceVoiceId: "", quality: "", lowVram: false });
    expect(ggmlLaunchKey("magpie", a)).toBe(ggmlLaunchKey("magpie", ggmlHelperConfig({ ...base, voice: "Leo", language: "fr" })));
    expect(ggmlLaunchKey("magpie", a)).not.toBe(ggmlLaunchKey("magpie", ggmlHelperConfig({ ...base, model: "v2607" })));
    // OmniVoice copies the library voice at start; Breeze and S2 read it per request.
    const omni = settings({ engine: "omnivoice", referenceVoiceId: "alice-1", quality: "fast" });
    expect(ggmlHelperConfig(omni)).toMatchObject({ referenceVoiceId: "alice-1", quality: "fast" });
    expect(ggmlHelperConfig({ ...omni, engine: "breeze" })).toMatchObject({ referenceVoiceId: "", quality: "" });
    expect(ggmlHelperConfig(settings({ engine: "s2", lowVram: true })).lowVram).toBe(true);
    expect(ggmlHelperConfig(settings({ engine: "breeze", lowVram: true })).lowVram).toBe(false);
  });

  it("builds each engine's request", () => {
    let r = ggmlSpeechSettings(endpoint, settings({ engine: "magpie", voice: "aria" }), "es");
    expect(r).toMatchObject({ baseUrl: endpoint.baseUrl, apiKey: endpoint.apiKey, voice: "Aria", responseFormat: "pcm", speed: 1, stream: false, pcmSampleRate: 22_050, extra: { language: "es-ES" } });

    r = ggmlSpeechSettings(endpoint, settings({ engine: "breeze", instruction: "  A calm voice. ", cfgScale: "9" }), "en");
    expect(r).toMatchObject({ voice: "", stream: true, extra: { instructions: "A calm voice.", cfg_scale: 3, seed: 42 } });
    expect(r.extra).not.toHaveProperty("sigil_reference_voice");
    r = ggmlSpeechSettings(endpoint, settings({ engine: "breeze", referenceVoiceId: "alice-1", seed: "77" }), "en");
    expect(r.extra).toMatchObject({ sigil_reference_voice: "alice-1", seed: 77, cfg_scale: 1 });

    r = ggmlSpeechSettings(endpoint, settings({ engine: "omnivoice" }), "ja");
    expect(r).toMatchObject({ voice: "", stream: true, extra: { language: "ja", seed: 7 } });
    r = ggmlSpeechSettings(endpoint, settings({ engine: "omnivoice", referenceVoiceId: "alice-1", cloneConsent: true }), "en");
    expect(r.voice).toBe("alice-1");
    expect(r.extra).toMatchObject({ consent_attestation: OMNIVOICE_CONSENT_TEXT });
    expect(r.extra).not.toHaveProperty("spoken_disclaimer");
    r = ggmlSpeechSettings(endpoint, settings({ engine: "omnivoice", referenceVoiceId: "alice-1", cloneConsent: true, discloseAi: true }), "en");
    expect(r.extra).toMatchObject({ spoken_disclaimer: false, marking_attestation: OMNIVOICE_DISCLOSURE_TEXT });

    r = ggmlSpeechSettings({ ...endpoint, pcmSampleRate: 44_100 }, settings({ engine: "s2", referenceVoiceId: "bob-2" }), "en");
    expect(r).toMatchObject({ voice: "", stream: false, pcmSampleRate: 44_100, extra: { sigil_reference_voice: "bob-2" } });
    expect(ggmlSeed(settings({ engine: "s2" }))).toBeNull();
    expect(MAGPIE_SPEAKERS).toEqual(["John", "Sofia", "Aria", "Jason", "Leo"]);
  });

  it("keeps the last sentence's stats per engine", () => {
    expect(noteGgmlReport("breeze", { sampleRate: 24_000, frames: 36_000, firstAudioMs: 280.4, totalMs: 1500 })).toMatchObject({ engine: "breeze", firstAudioMs: 280, audioSec: 1.5 });
    expect(ggmlUi.last?.engine).toBe("breeze");
    expect(noteGgmlReport("breeze", { sampleRate: 0, frames: 0, firstAudioMs: 0, totalMs: 0 })).toBeNull();
  });
});

describe("SerialGate", () => {
  it("hands out turns in order and lets an aborted waiter leave", async () => {
    const gate = new SerialGate();
    const order: string[] = [];
    const first = await gate.acquire(new AbortController().signal);
    const ac = new AbortController();
    const second = gate.acquire(ac.signal);
    const third = gate.acquire(new AbortController().signal).then((release) => {
      order.push("third");
      return release;
    });
    ac.abort();
    expect(await second).toBeNull();
    await Promise.resolve();
    expect(order).toEqual([]);
    first?.();
    const r3 = await third;
    expect(order).toEqual(["third"]);
    r3?.();
    const aborted = new AbortController();
    aborted.abort();
    expect(await gate.acquire(aborted.signal)).toBeNull();
    expect(await gate.acquire(new AbortController().signal)).toBeTypeOf("function");
  });
});

// -------------------------------------------------------------------------------------------------
// The backend
// -------------------------------------------------------------------------------------------------

const ENDPOINT: RustTtsHelperEndpoint = { engine: "magpie", baseUrl: "http://127.0.0.1:5123/v1", apiKey: "a".repeat(64), model: "tts/magpie-tts-357m-v2602", pcmSampleRate: 22_050 };
const REPORT = { sampleRate: 22_050, frames: 22_050, firstAudioMs: 300, totalMs: 900 };

let data: TTS_State;
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

function answer(overrides: Record<string, (args: Record<string, unknown>) => unknown> = {}) {
  invokeMock.mockImplementation(async (cmd: string, args?: unknown) => {
    const a = (args ?? {}) as Record<string, unknown>;
    if (overrides[cmd]) return overrides[cmd](a);
    switch (cmd) {
      case "plugin:tts-helpers|tts_helper_start":
        return { ...ENDPOINT, engine: a.engine };
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
  data = proxy(Service_TTS_Schema.parse({ backend: "ggmlTts" }));
  vi.stubGlobal("window", {
    ApiServer: {
      state: { services: { tts: { data } }, audioOutputDevice: "" },
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

describe("TTS_GgmlTtsService", () => {
  it("is a sink backend registered as ggmlTts", () => {
    expect(TTS_BACKENDS.ggmlTts).toBe(TTS_GgmlTtsService);
    expect(ttsOutputMode(new TTS_GgmlTtsService(receiver()))).toBe("sink");
  });

  it("starts the helper and streams sentences in order, one at a time", async () => {
    let active = 0;
    let maxActive = 0;
    answer({
      "plugin:openai-speech|openai_speech_speak": async () => {
        active += 1;
        maxActive = Math.max(maxActive, active);
        await new Promise((r) => setTimeout(r, 5));
        active -= 1;
        return REPORT;
      },
    });
    const r = receiver();
    const s = new TTS_GgmlTtsService(r);
    await s.start(data);
    expect(r.onStart).toHaveBeenCalledTimes(1);
    expect(calls("plugin:tts-helpers|tts_helper_start")).toEqual([
      { engine: "magpie", config: { model: "v2602", variant: "auto", language: "", voice: "", referenceVoiceId: "", quality: "", lowVram: false } },
    ]);
    data.ggmlTts.voice = "Leo";
    await Promise.all([
      s.synthesizeToSink("One.", sinkCtx("u1", "de"), new AbortController().signal),
      s.synthesizeToSink("Two.", sinkCtx("u2", "fr"), new AbortController().signal),
      s.synthesizeToSink("Three.", sinkCtx("u3"), new AbortController().signal),
    ]);
    expect(maxActive).toBe(1);
    const speaks = calls("plugin:openai-speech|openai_speech_speak");
    expect(speaks.map((c) => c.utteranceId)).toEqual(["u1", "u2", "u3"]);
    expect(speaks[0].request).toMatchObject({ input: "One.", voice: "Leo", extra: { language: "de-DE" } });
    expect(speaks[1].request).toMatchObject({ extra: { language: "fr-FR" } });
    // Voice changes do not reload the helper.
    expect(calls("plugin:tts-helpers|tts_helper_start")).toHaveLength(1);
    expect(ggmlUi.last).toMatchObject({ engine: "magpie", audioSec: 1 });
  });

  it("fails a sentence its engine cannot speak, without stopping", async () => {
    const r = receiver();
    const s = new TTS_GgmlTtsService(r);
    await s.start(data);
    await expect(s.synthesizeToSink("こんにちは。", sinkCtx("u1", "ja"), new AbortController().signal)).rejects.toThrow("NVIDIA Magpie TTS: tts.ggml_magpie_language_unsupported");
    expect(r.onStop).not.toHaveBeenCalled();
    expect(calls("plugin:openai-speech|openai_speech_speak")).toHaveLength(0);
  });

  it("refuses without the desktop app, or without consent to clone", async () => {
    isTauriMock.mockReturnValue(false);
    let r = receiver();
    await new TTS_GgmlTtsService(r).start(data);
    expect(r.onStop).toHaveBeenCalledWith("tts.ggml_desktop_only");
    isTauriMock.mockReturnValue(true);
    data.ggmlTts.engine = "omnivoice";
    data.ggmlTts.referenceVoiceId = "alice-1";
    r = receiver();
    await new TTS_GgmlTtsService(r).start(data);
    expect(r.onStop.mock.calls[0][0]).toContain(TTS_OPTIONS_MISSING);
    expect(calls("plugin:tts-helpers|tts_helper_start")).toHaveLength(0);
  });

  it("reports a failed start with Rust's reason", async () => {
    answer({ "plugin:tts-helpers|tts_helper_start": () => Promise.reject("Breeze TTS 2 needs downloads first: tts/breeze-tts-2-q8_0") });
    data.ggmlTts.engine = "breeze";
    const r = receiver();
    await new TTS_GgmlTtsService(r).start(data);
    expect(r.onStart).not.toHaveBeenCalled();
    expect(r.onStop).toHaveBeenCalledWith("tts.ggml_start_failed");
  });

  it("reloads on a launch change, stops the old engine on an engine change, and finds a restarted helper", async () => {
    const r = receiver();
    const s = new TTS_GgmlTtsService(r);
    await s.start(data);
    data.ggmlTts.model = "v2607";
    await s.synthesizeToSink("One.", sinkCtx("u1"), new AbortController().signal);
    let starts = calls("plugin:tts-helpers|tts_helper_start");
    expect(starts).toHaveLength(2);
    expect(starts[1]).toMatchObject({ engine: "magpie", config: { model: "v2607" } });
    // Another engine: the old helper is stopped first.
    data.ggmlTts.engine = "breeze";
    data.ggmlTts.model = "";
    await s.synthesizeToSink("Two.", sinkCtx("u2"), new AbortController().signal);
    starts = calls("plugin:tts-helpers|tts_helper_start");
    expect(starts[2]).toMatchObject({ engine: "breeze", config: { model: "q8_0" } });
    expect(calls("plugin:tts-helpers|tts_helper_stop")).toEqual([{ engine: "magpie" }]);
    expect(calls("plugin:openai-speech|openai_speech_speak")[1].request).toMatchObject({ extra: { seed: 42, cfg_scale: 1 } });
    // A crash of another engine's helper is ignored; this engine's restart makes the next sentence look again.
    helperListener?.({ payload: { engine: "magpie", state: "starting", model: null, error: null } });
    await s.synthesizeToSink("Three.", sinkCtx("u3"), new AbortController().signal);
    expect(calls("plugin:tts-helpers|tts_helper_start")).toHaveLength(3);
    helperListener?.({ payload: { engine: "breeze", state: "starting", model: null, error: null } });
    await s.synthesizeToSink("Four.", sinkCtx("u4"), new AbortController().signal);
    expect(calls("plugin:tts-helpers|tts_helper_start")).toHaveLength(4);
    helperListener?.({ payload: { engine: "breeze", state: "failed", model: null, error: "exit code 3" } });
    expect(r.onError).toHaveBeenCalledWith("tts.ggml_helper_failed");
    s.stop();
    expect(calls("plugin:tts-helpers|tts_helper_stop").at(-1)).toEqual({ engine: "breeze" });
    expect(r.onStop).toHaveBeenCalledWith();
  });

  it("retries once after a connection error, and an aborted sentence never reports", async () => {
    const r = receiver();
    const s = new TTS_GgmlTtsService(r);
    await s.start(data);
    let fail = 1;
    answer({
      "plugin:openai-speech|openai_speech_speak": () => {
        if (fail > 0) {
          fail -= 1;
          return Promise.reject("Speech request failed: connection refused");
        }
        return REPORT;
      },
    });
    await s.synthesizeToSink("One.", sinkCtx("u1"), new AbortController().signal);
    expect(calls("plugin:tts-helpers|tts_helper_start")).toHaveLength(2);
    expect(calls("plugin:openai-speech|openai_speech_speak")).toHaveLength(2);
    const ac = new AbortController();
    answer({ "plugin:openai-speech|openai_speech_speak": () => new Promise((_, reject) => ac.signal.addEventListener("abort", () => reject("cancelled"))) });
    const p = s.synthesizeToSink("Two.", sinkCtx("u2"), ac.signal);
    await flush();
    ac.abort();
    await expect(p).resolves.toBeUndefined();
    expect(calls("plugin:openai-speech|openai_speech_cancel")).toContainEqual({ utteranceId: "u2" });
    expect(r.onError).not.toHaveBeenCalled();
  });

  it("never repeats the start of a streamed sentence that broke off, but sends a whole-clip one again", async () => {
    const broke = "Speech request (http://127.0.0.1:5123/v1/audio/speech) failed: connection closed before message completed";
    let fail = 1;
    answer({
      "plugin:openai-speech|openai_speech_speak": () => {
        if (fail > 0) {
          fail -= 1;
          return Promise.reject(broke);
        }
        return REPORT;
      },
    });
    // Breeze streams: its first part may have played already, so the sentence fails (and the helper is found again).
    data.ggmlTts.engine = "breeze";
    const r = receiver();
    const s = new TTS_GgmlTtsService(r);
    await s.start(data);
    await expect(s.synthesizeToSink("One.", sinkCtx("u1"), new AbortController().signal)).rejects.toThrow(`Breeze TTS 2: ${broke}`);
    expect(calls("plugin:openai-speech|openai_speech_speak")).toHaveLength(1);
    await s.synthesizeToSink("Two.", sinkCtx("u2"), new AbortController().signal);
    expect(calls("plugin:tts-helpers|tts_helper_start")).toHaveLength(2);
    s.stop();

    // Magpie answers a whole clip per sentence: nothing played yet, so it is sent once more.
    invokeMock.mockClear();
    fail = 1;
    data.ggmlTts.engine = "magpie";
    const m = new TTS_GgmlTtsService(receiver());
    await m.start(data);
    await m.synthesizeToSink("Three.", sinkCtx("u3"), new AbortController().signal);
    expect(calls("plugin:openai-speech|openai_speech_speak").map((c) => c.utteranceId)).toEqual(["u3", "u3"]);
    m.stop();
  });

  it("starts an engine only once its pending stop finished (a restart of text to speech)", async () => {
    const stopping: { finish?: () => void } = {};
    answer({ "plugin:tts-helpers|tts_helper_stop": () => new Promise<void>((resolve) => (stopping.finish = resolve)) });
    const first = new TTS_GgmlTtsService(receiver());
    await first.start(data);
    first.stop();
    const r = receiver();
    const second = new TTS_GgmlTtsService(r);
    const starting = second.start(data);
    await flush();
    // Rust would cancel a start that the stop overtakes: it is not sent yet.
    expect(calls("plugin:tts-helpers|tts_helper_start")).toHaveLength(1);
    expect(stopping.finish).toBeTypeOf("function");
    // Another engine does not wait for it.
    await expect(startGgmlHelper("breeze", ggmlHelperConfig(settings({ engine: "breeze" })))).resolves.toMatchObject({ engine: "breeze" });
    stopping.finish?.();
    await starting;
    const order = calls("plugin:tts-helpers|tts_helper_start").map((a) => a.engine);
    expect(order).toEqual(["magpie", "breeze", "magpie"]);
    const helperCalls = invokeMock.mock.calls.map(([c]) => c).filter((c) => c.startsWith("plugin:tts-helpers"));
    expect(helperCalls.indexOf("plugin:tts-helpers|tts_helper_stop")).toBe(1);
    expect(r.onStart).toHaveBeenCalledTimes(1);
    answer();
    second.stop();
    await stopGgmlHelper("breeze");
  });
});

describe("previews", () => {
  const target = { route: "device" as const, device: "", volume: 1 };

  beforeEach(() => {
    vi.useFakeTimers();
    const sink = { ready: vi.fn(async () => {}), begin: vi.fn(async () => {}), end: vi.fn(async () => {}), cancel: vi.fn(async () => {}) };
    (window as unknown as { ApiServer: Record<string, unknown> }).ApiServer.sound = { sink };
  });

  const stops = () => calls("plugin:tts-helpers|tts_helper_stop").map((a) => a.engine);
  const helperCalls = () =>
    invokeMock.mock.calls
      .filter(([c]) => c === "plugin:tts-helpers|tts_helper_start" || c === "plugin:tts-helpers|tts_helper_stop")
      .map(([c, a]) => `${c.endsWith("start") ? "start" : "stop"} ${(a as { engine: string }).engine}`);

  it("give every engine its own idle stop, and free another engine's helper before the next one loads", async () => {
    await previewGgml(settings({ engine: "s2" }), "Hi.", target);
    expect(ggmlPreviewIdleEngines()).toEqual(["s2"]);
    // A preview of another engine stops the S2 helper (about 7 GB of video memory) before OmniVoice loads.
    await previewGgml(settings({ engine: "omnivoice" }), "Hi.", target);
    expect(helperCalls()).toEqual(["start s2", "stop s2", "start omnivoice"]);
    expect(ggmlPreviewIdleEngines()).toEqual(["omnivoice"]);
    // Previewing the same engine again keeps its helper; the idle stop starts over.
    await previewGgml(settings({ engine: "omnivoice" }), "Hi again.", target);
    expect(stops()).toEqual(["s2"]);
    await vi.advanceTimersByTimeAsync(GGML_PREVIEW_IDLE_MS);
    expect(stops()).toEqual(["s2", "omnivoice"]);
    expect(ggmlPreviewIdleEngines()).toEqual([]);
  });

  it("stop when text to speech starts with another engine, and are taken over when it starts with theirs", async () => {
    await previewGgml(settings({ engine: "s2" }), "Hi.", target);
    await previewGgml(settings({ engine: "magpie" }), "Hi.", target);
    expect(ggmlPreviewIdleEngines()).toEqual(["magpie"]);
    invokeMock.mockClear();
    const s = new TTS_GgmlTtsService(receiver());
    await s.start(data);
    // Magpie's helper is the service's now: no idle stop is left that could unload it under the running service.
    expect(helperCalls()).toEqual(["start magpie"]);
    expect(ggmlPreviewIdleEngines()).toEqual([]);
    await vi.advanceTimersByTimeAsync(GGML_PREVIEW_IDLE_MS);
    expect(stops()).toEqual([]);
    s.stop();

    invokeMock.mockClear();
    await previewGgml(settings({ engine: "s2" }), "Hi.", target);
    const t = new TTS_GgmlTtsService(receiver());
    await t.start(data);
    expect(helperCalls()).toEqual(["start s2", "stop s2", "start magpie"]);
    t.stop();
  });
});
