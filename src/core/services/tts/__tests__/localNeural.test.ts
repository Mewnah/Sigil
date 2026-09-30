import { readFileSync } from "node:fs";
import { resolve } from "node:path";
import i18n from "i18next";
import { proxy } from "valtio";
import { afterEach, beforeAll, beforeEach, describe, expect, it, vi } from "vitest";

vi.mock("@tauri-apps/api/core", () => ({ invoke: vi.fn(), isTauri: () => false }));

import { invoke } from "@tauri-apps/api/core";
import type { RustCatalogEntry, RustModelEntryStatus } from "@/types";
import { Service_TTS_Schema, TTS_State } from "../schema";
import { LOCAL_NEURAL_FOLDER } from "../schema/localNeural";
import {
  HUB_CATALOG,
  LOCAL_TTS_DEFAULT_MODEL,
  LocalTtsModelInfo,
  TTS_LocalNeuralService,
  defaultVoiceLanguage,
  groupLocalModels,
  groupVoicesByLanguage,
  hubCatalogId,
  localNeuralModel,
  localNeuralSettings,
  localTtsState,
  missingDownloads,
  normalizeHubRepo,
  previewText,
  requirementIdsOf,
  scriptLanguage,
  threadsOf,
  unloadLocalTtsIfIdle,
  voicesFor,
} from "../services/localNeural";
import { TTS_NeuralOnnxService, neuralOnnxSettings } from "../services/neuralOnnx";
import type { ITTSReceiver } from "../types";

// P2-H: the localNeural backend (in-process Kokoro / Piper / Kitten through the Rust local-tts plugin) and the
// legacy neuralOnnx backend mapped onto it.

const invokeMock = vi.mocked(invoke);

function receiver() {
  return { onStart: vi.fn(), onStop: vi.fn(), onError: vi.fn() } satisfies ITTSReceiver;
}

const flush = async (n = 20) => {
  for (let i = 0; i < n; i++) await Promise.resolve();
};

function entry(id: string, requires: string[] = [], title = id): RustCatalogEntry {
  return {
    id,
    family: id.split("/")[0],
    kind: id.startsWith("runtime/") ? "runtime" : "model",
    version: "1",
    title,
    license: "Apache-2.0",
    status: "published",
    requires,
    files: [{ path: `${id}.tar.bz2`, size: 1000, sha256: "0".repeat(64), sources: [] }],
  } as unknown as RustCatalogEntry;
}

function status(id: string, state: RustModelEntryStatus["state"]): RustModelEntryStatus {
  return { id, state, installed_version: state === "installed" ? "1" : null, installed_path: null, bytes_on_disk: 0, adopted: false, error: null };
}

const CATALOG = [
  entry("runtime/sherpa-onnx", [], "sherpa-onnx runtime"),
  entry("tts/kokoro-v1.0", ["runtime/sherpa-onnx"], "Kokoro v1.0 (multilingual)"),
  entry("tts/kitten-mini-en-v0.8", ["runtime/sherpa-onnx"], "Kitten mini 0.8 (English)"),
];

let data: TTS_State;
let models: { load: ReturnType<typeof vi.fn>; state: { catalog: RustCatalogEntry[]; states: Record<string, RustModelEntryStatus> } };

function installWindow() {
  data = proxy(Service_TTS_Schema.parse({}));
  models = { load: vi.fn(async () => {}), state: { catalog: CATALOG, states: {} } };
  vi.stubGlobal("window", {
    ApiServer: {
      state: { services: { tts: { data } }, audioOutputDevice: "" },
      models,
      tts: { serviceState: { status: "disconnected" } },
    },
  });
}

beforeAll(async () => {
  await i18n.init({ lng: "en", resources: { en: { translation: {} } }, interpolation: { escapeValue: false } });
});

beforeEach(() => {
  invokeMock.mockReset();
  localTtsState.loaded = null;
  installWindow();
});

afterEach(() => {
  vi.unstubAllGlobals();
});

describe("localNeural settings", () => {
  it("maps the persisted slice onto a request", () => {
    const c = Service_TTS_Schema.parse({}).localNeural;
    expect(localNeuralModel(c)).toBe(LOCAL_TTS_DEFAULT_MODEL);
    expect(localNeuralSettings(c)).toEqual({ model: LOCAL_TTS_DEFAULT_MODEL, voice: "", speed: 1, language: null, threads: null });
    const custom = Service_TTS_Schema.parse({
      localNeural: { model: " tts/piper-ru_ru-irina-medium ", voice: " 0 ", speed: "9", threads: "3" },
    }).localNeural;
    expect(localNeuralSettings(custom)).toMatchObject({ model: "tts/piper-ru_ru-irina-medium", voice: "0", speed: 2, threads: 3 });
    expect(localNeuralSettings({ ...custom, speed: "abc" }).speed).toBe(1);
    expect(localNeuralSettings({ ...custom, speed: "0.1" }).speed).toBe(0.5);
    const folder = { ...custom, model: LOCAL_NEURAL_FOLDER, folder: "  C:\\Voices\\kokoro  " };
    expect(localNeuralModel(folder)).toBe("C:\\Voices\\kokoro");
    expect(localNeuralModel({ ...folder, folder: " " })).toBe("");
  });

  it("parses thread settings", () => {
    expect(threadsOf("auto")).toBeNull();
    expect(threadsOf("2")).toBe(2);
    expect(threadsOf("4")).toBe(4);
    expect(threadsOf("5")).toBeNull();
    expect(threadsOf("0")).toBeNull();
    expect(threadsOf("2.5")).toBeNull();
    expect(threadsOf(undefined)).toBeNull();
  });
});

describe("Hub repos", () => {
  it("normalizes repo ids like Rust", () => {
    expect(normalizeHubRepo("kitten-tts-mini-0.8")).toBe("KittenML/kitten-tts-mini-0.8");
    expect(normalizeHubRepo(" someone/voice_1.0 ")).toBe("someone/voice_1.0");
    expect(normalizeHubRepo("../escape")).toBeNull();
    expect(normalizeHubRepo("a/.hidden")).toBeNull();
    expect(normalizeHubRepo("a/b/c")).toBeNull();
    expect(normalizeHubRepo("")).toBeNull();
    expect(normalizeHubRepo("has space/x")).toBeNull();
  });

  it("maps KittenML 0.8 repos onto catalog packages, any case", () => {
    expect(hubCatalogId("KittenML/kitten-tts-mini-0.8")).toBe("tts/kitten-mini-en-v0.8");
    expect(hubCatalogId("kittenml/KITTEN-TTS-MINI-0.8")).toBe("tts/kitten-mini-en-v0.8");
    expect(hubCatalogId("kitten-tts-nano-0.8-fp32")).toBe("tts/kitten-nano-en-v0.8");
    expect(hubCatalogId("KittenML/kitten-tts-nano-0.2")).toBeNull();
    expect(hubCatalogId("../x")).toBeNull();
  });

  it("keeps the table in step with Rust local_tts::models::HUB_CATALOG", () => {
    const rust = readFileSync(resolve(__dirname, "../../../../../src-tauri/src/services/local_tts/models.rs"), "utf8");
    const block = rust.slice(rust.indexOf("pub const HUB_CATALOG"), rust.indexOf("];", rust.indexOf("pub const HUB_CATALOG")));
    const pairs = [...block.matchAll(/\("([^"]+)",\s*"([^"]+)"\)/g)].map((m) => [m[1], m[2]]);
    expect(pairs.length).toBeGreaterThan(0);
    expect(pairs).toEqual(HUB_CATALOG.map(([r, id]) => [r, id]));
  });

  it("names the catalog entries a request model needs", () => {
    expect(requirementIdsOf("tts/kokoro-v1.0")).toEqual(["tts/kokoro-v1.0"]);
    expect(requirementIdsOf("hub:KittenML/kitten-tts-micro-0.8")).toEqual(["tts/kitten-micro-en-v0.8"]);
    expect(requirementIdsOf("hub:someone/else")).toEqual([]);
    expect(requirementIdsOf("C:\\Voices\\kokoro")).toEqual([]);
  });
});

describe("neuralOnnx mapped onto the local engine", () => {
  it("turns the legacy settings into a local-tts request", () => {
    const c = Service_TTS_Schema.parse({}).neuralOnnx;
    expect(neuralOnnxSettings(c)).toEqual({ model: "hub:KittenML/kitten-tts-mini-0.8", voice: "Jasper", speed: 1, language: null, threads: null });
    expect(neuralOnnxSettings({ ...c, hubId: "kitten-tts-nano-0.8", voice: " ", speed: "7" })).toMatchObject({
      model: "hub:KittenML/kitten-tts-nano-0.8",
      voice: "",
      speed: 2,
    });
    expect(neuralOnnxSettings({ ...c, modelSource: "local", localModelDir: " D:\\pkg " })).toMatchObject({ model: "D:\\pkg" });
    expect(typeof neuralOnnxSettings({ ...c, modelSource: "local", localModelDir: "" })).toBe("string");
    expect(typeof neuralOnnxSettings({ ...c, hubId: "bad id" })).toBe("string");
  });

  it("starts the local engine with the mapped model and speaks into the sink", async () => {
    invokeMock.mockImplementation(async (cmd) => {
      if (cmd === "plugin:local-tts|load_local_tts") return { id: "tts/kitten-mini-en-v0.8", voices: [], installed: true };
      return { sampleRate: 24000, frames: 10, firstAudioMs: 1, totalMs: 2 };
    });
    const r = receiver();
    const s = new TTS_NeuralOnnxService(r);
    await s.start(data);
    expect(r.onStart).toHaveBeenCalledTimes(1);
    expect(invokeMock).toHaveBeenCalledWith("plugin:local-tts|load_local_tts", {
      model: "hub:KittenML/kitten-tts-mini-0.8",
      threads: null,
      voice: "Jasper",
      language: null,
    });
    await s.synthesizeToSink("Hello.", { utteranceId: "u1", target: { route: "device", device: "", volume: 1 } }, new AbortController().signal);
    expect(invokeMock).toHaveBeenCalledWith("plugin:local-tts|speak_local_tts", {
      request: { model: "hub:KittenML/kitten-tts-mini-0.8", voice: "Jasper", speed: 1, language: null, threads: null, text: "Hello." },
      utteranceId: "u1",
    });
    // The old neural-onnx-tts plugin is not used any more.
    expect(invokeMock.mock.calls.some(([cmd]) => String(cmd).includes("neural-onnx-tts"))).toBe(false);
  });

  it("a Hub repo whose catalog package is missing says what to download", async () => {
    invokeMock.mockImplementation(async (cmd) => {
      if (cmd === "plugin:local-tts|load_local_tts") throw "Kitten mini 0.8 (English) is not downloaded yet (tts/kitten-mini-en-v0.8)";
      return undefined;
    });
    const r = receiver();
    await new TTS_NeuralOnnxService(r).start(data);
    expect(r.onStart).not.toHaveBeenCalled();
    expect(r.onStop).toHaveBeenCalledWith("tts.local_neural_not_downloaded");
    // A failure for another reason keeps Rust's message.
    models.state.states = {
      "runtime/sherpa-onnx": status("runtime/sherpa-onnx", "installed"),
      "tts/kitten-mini-en-v0.8": status("tts/kitten-mini-en-v0.8", "installed"),
    };
    const r2 = receiver();
    await new TTS_NeuralOnnxService(r2).start(data);
    expect(r2.onStop).toHaveBeenCalledWith(expect.stringContaining("Local ONNX speech: Kitten mini 0.8"));
  });
});

describe("TTS_LocalNeuralService", () => {
  const installed = () => {
    models.state.states = {
      "runtime/sherpa-onnx": status("runtime/sherpa-onnx", "installed"),
      "tts/kokoro-v1.0": status("tts/kokoro-v1.0", "installed"),
    };
  };

  it("refuses to start before the model and runtime are downloaded, without loading", async () => {
    const r = receiver();
    await new TTS_LocalNeuralService(r).start(data);
    expect(r.onStop).toHaveBeenCalledWith("tts.local_neural_not_downloaded");
    expect(invokeMock).not.toHaveBeenCalledWith("plugin:local-tts|load_local_tts", expect.anything());
  });

  it("refuses a folder model without a folder", async () => {
    data.localNeural.model = LOCAL_NEURAL_FOLDER;
    const r = receiver();
    await new TTS_LocalNeuralService(r).start(data);
    expect(r.onStop).toHaveBeenCalledWith("tts.local_neural_folder_missing");
  });

  it("loads with the voice and threads, remembers the loaded info, and reports Rust errors", async () => {
    installed();
    data.localNeural.voice = "jf_alpha";
    data.localNeural.threads = "4";
    const info = { id: "tts/kokoro-v1.0", family: "kokoro", title: "Kokoro", languages: ["ja"], voices: [{ id: "jf_alpha", name: "Alpha", language: "ja" }], sampleRate: 24000, installed: true, defaultVoice: "af_heart" };
    invokeMock.mockImplementation(async (cmd) => (cmd === "plugin:local-tts|load_local_tts" ? info : undefined));
    const r = receiver();
    await new TTS_LocalNeuralService(r).start(data);
    expect(r.onStart).toHaveBeenCalledTimes(1);
    expect(invokeMock).toHaveBeenCalledWith("plugin:local-tts|load_local_tts", { model: "tts/kokoro-v1.0", threads: 4, voice: "jf_alpha", language: null });
    expect(localTtsState.loaded?.voices.map((v) => v.id)).toEqual(["jf_alpha"]);

    invokeMock.mockImplementation(async () => {
      throw "sherpa-onnx could not load the kokoro voice";
    });
    const r2 = receiver();
    await new TTS_LocalNeuralService(r2).start(data);
    expect(r2.onStop).toHaveBeenCalledWith("Local voice: sherpa-onnx could not load the kokoro voice");
  });

  it("a stop during start wins over the late load", async () => {
    installed();
    let answer!: (v: unknown) => void;
    invokeMock.mockImplementation((cmd) => (cmd === "plugin:local-tts|load_local_tts" ? new Promise((res) => (answer = res)) : Promise.resolve(undefined)));
    const r = receiver();
    const s = new TTS_LocalNeuralService(r);
    const starting = s.start(data);
    await flush();
    s.stop();
    answer({ id: "tts/kokoro-v1.0", voices: [] });
    await starting;
    expect(r.onStart).not.toHaveBeenCalled();
    expect(r.onStop).toHaveBeenCalledTimes(1);
    expect(invokeMock).toHaveBeenCalledWith("plugin:local-tts|cancel_local_tts", { utteranceId: null });
    expect(invokeMock).toHaveBeenCalledWith("plugin:local-tts|unload_local_tts", undefined);
  });

  it("stop / skip cancel the Rust job and are not errors; other failures are", async () => {
    installed();
    let rejectSpeak!: (e: unknown) => void;
    invokeMock.mockImplementation((cmd) => {
      if (cmd === "plugin:local-tts|load_local_tts") return Promise.resolve({ id: "tts/kokoro-v1.0", voices: [] });
      if (cmd === "plugin:local-tts|speak_local_tts") return new Promise((_, rej) => (rejectSpeak = rej));
      return Promise.resolve(undefined);
    });
    const s = new TTS_LocalNeuralService(receiver());
    await s.start(data);
    const ac = new AbortController();
    const speaking = s.synthesizeToSink("Hello there.", { utteranceId: "s:1", target: { route: "device", device: "", volume: 1 } }, ac.signal);
    await flush();
    ac.abort();
    expect(invokeMock).toHaveBeenCalledWith("plugin:local-tts|cancel_local_tts", { utteranceId: "s:1" });
    rejectSpeak("cancelled");
    await expect(speaking).resolves.toBeUndefined();

    const failing = s.synthesizeToSink("Boom.", { utteranceId: "s:2", target: { route: "device", device: "", volume: 1 } }, new AbortController().signal);
    await flush();
    rejectSpeak("the voice has no phonemes for this text");
    await expect(failing).rejects.toThrow("Local voice: the voice has no phonemes for this text");
    // Empty text and a stopped backend do nothing.
    invokeMock.mockClear();
    await s.synthesizeToSink("  ", { utteranceId: "s:3", target: { route: "device", device: "", volume: 1 } }, new AbortController().signal);
    s.stop();
    await s.synthesizeToSink("Late.", { utteranceId: "s:4", target: { route: "device", device: "", volume: 1 } }, new AbortController().signal);
    expect(invokeMock).not.toHaveBeenCalledWith("plugin:local-tts|speak_local_tts", expect.anything());
  });

  it("gives default-voice sentences the language their script shows (Rust picks that language's voice)", async () => {
    installed();
    invokeMock.mockImplementation(async (cmd) => (cmd === "plugin:local-tts|load_local_tts" ? { id: "tts/kokoro-v1.0", voices: [] } : undefined));
    const s = new TTS_LocalNeuralService(receiver());
    await s.start(data);
    const target = { route: "device", device: "", volume: 1 } as const;
    const languages: (string | null | undefined)[] = [];
    for (const [i, text] of ["今日はいい天気ですね。", "123!", "Hello there.", "你好，世界。"].entries()) {
      invokeMock.mockClear();
      await s.synthesizeToSink(text, { utteranceId: `u${i}`, target }, new AbortController().signal);
      const call = invokeMock.mock.calls.find(([cmd]) => cmd === "plugin:local-tts|speak_local_tts");
      languages.push((call?.[1] as { request: { language?: string | null } }).request.language);
    }
    expect(languages).toEqual(["ja", "ja", null, "zh"]);

    // The text event's language tag wins over the script guess (Latin-script languages need it).
    invokeMock.mockClear();
    await s.synthesizeToSink("Bonjour à tous.", { utteranceId: "u5", target, lang: "fr" }, new AbortController().signal);
    expect(invokeMock).toHaveBeenCalledWith("plugin:local-tts|speak_local_tts", {
      request: expect.objectContaining({ voice: "", language: "fr" }),
      utteranceId: "u5",
    });

    // A chosen voice keeps its own language (no reload per sentence).
    data.localNeural.voice = "af_heart";
    invokeMock.mockClear();
    await s.synthesizeToSink("今日はいい天気ですね。", { utteranceId: "u9", target }, new AbortController().signal);
    expect(invokeMock).toHaveBeenCalledWith("plugin:local-tts|speak_local_tts", {
      request: expect.objectContaining({ voice: "af_heart", language: null }),
      utteranceId: "u9",
    });
  });

  it("stop never throws, even when IPC is unavailable", () => {
    invokeMock.mockImplementation(() => {
      throw new Error("no Tauri");
    });
    const r = receiver();
    const s = new TTS_LocalNeuralService(r);
    expect(() => s.stop()).not.toThrow();
    expect(() => s.dispose()).not.toThrow();
    expect(r.onStop).toHaveBeenCalledTimes(1);
  });
});

describe("sentence language from the script", () => {
  it("tells Japanese, Chinese and Hindi apart and leaves Latin scripts to the model's default", () => {
    expect(scriptLanguage("今日はいい天気ですね。")).toBe("ja");
    expect(scriptLanguage("コーヒーを飲む")).toBe("ja");
    expect(scriptLanguage("iPhoneを買った。")).toBe("ja");
    expect(scriptLanguage("你好，世界。")).toBe("zh");
    expect(scriptLanguage("我用iPhone拍照。")).toBe("zh");
    expect(scriptLanguage("नमस्ते, आप कैसे हैं?")).toBe("hi");
    expect(scriptLanguage("Hello there.")).toBeNull();
    expect(scriptLanguage("¡Hola! ¿Qué tal?")).toBeNull();
    expect(scriptLanguage("I love 寿司.")).toBeNull();
    expect(scriptLanguage("Привет!")).toBeNull();
    expect(scriptLanguage("123!")).toBeUndefined();
    expect(scriptLanguage("  ")).toBeUndefined();
  });

  it("keeps the previous language for sentences without letters", () => {
    expect(defaultVoiceLanguage("123!", "ja")).toBe("ja");
    expect(defaultVoiceLanguage("123!", null)).toBeNull();
    expect(defaultVoiceLanguage("Hello.", "ja")).toBeNull();
    expect(defaultVoiceLanguage("你好。", null)).toBe("zh");
  });
});

describe("preview cleanup", () => {
  const unloads = () => invokeMock.mock.calls.filter(([cmd]) => cmd === "plugin:local-tts|unload_local_tts").length;
  const setTts = (status: string, backend: string) => {
    const w = window as unknown as { ApiServer: { tts: { serviceState: { status: string } } } };
    w.ApiServer.tts.serviceState.status = status;
    data.backend = backend as TTS_State["backend"];
  };

  it("frees the model unless a running local backend uses it", () => {
    invokeMock.mockResolvedValue(undefined);
    setTts("disconnected", "localNeural");
    unloadLocalTtsIfIdle();
    expect(unloads()).toBe(1);
    setTts("connected", "azure");
    unloadLocalTtsIfIdle();
    expect(unloads()).toBe(2);
    setTts("connected", "localNeural");
    unloadLocalTtsIfIdle();
    setTts("connecting", "neuralOnnx");
    unloadLocalTtsIfIdle();
    expect(unloads()).toBe(2);
  });
});

describe("download check", () => {
  it("lists what a model still needs, runtime included", () => {
    expect(missingDownloads(["tts/kokoro-v1.0"], CATALOG, {})?.map((e) => e.id)).toEqual(["runtime/sherpa-onnx", "tts/kokoro-v1.0"]);
    expect(
      missingDownloads(["tts/kokoro-v1.0"], CATALOG, { "runtime/sherpa-onnx": status("runtime/sherpa-onnx", "installed") })?.map((e) => e.id),
    ).toEqual(["tts/kokoro-v1.0"]);
    expect(missingDownloads(["tts/unknown"], CATALOG, {})).toBeNull();
    expect(missingDownloads([], CATALOG, {})).toBeNull();
    expect(missingDownloads(["tts/kokoro-v1.0"], [], {})).toBeNull();
  });
});

describe("panel helpers", () => {
  const model = (id: string, family: string, title: string, voices: LocalTtsModelInfo["voices"] = [], defaultVoice = ""): LocalTtsModelInfo => ({
    id,
    family,
    title,
    languages: [],
    voices,
    sampleRate: 24000,
    installed: false,
    defaultVoice,
  });

  it("picks a preview line in the voice's language", () => {
    expect(previewText("ja")).toMatch(/こんにちは/);
    expect(previewText("pt-BR")).toMatch(/Olá/);
    expect(previewText("en-GB")).toMatch(/Hello/);
    expect(previewText("sw")).toMatch(/Hello/);
    expect(previewText(null)).toMatch(/Hello/);
  });

  it("groups models by family with Piper sorted by title", () => {
    const groups = groupLocalModels([
      model("tts/piper-ru", "piper", "Russian: Irina (Piper)"),
      model("tts/kokoro-v1.0", "kokoro", "Kokoro"),
      model("tts/piper-ar", "piper", "Arabic: Kareem (Piper)"),
      model("tts/kitten", "kitten", "Kitten"),
      model("tts/new", "matcha", "Matcha"),
    ]);
    expect(groups.map((g) => g.family)).toEqual(["kokoro", "kitten", "piper", "matcha"]);
    expect(groups[2].models.map((m) => m.id)).toEqual(["tts/piper-ar", "tts/piper-ru"]);
  });

  it("groups voices by language in first-seen order", () => {
    const groups = groupVoicesByLanguage([
      { id: "af_heart", language: "en-US" },
      { id: "bf_emma", language: "en-GB" },
      { id: "af_bella", language: "en-US" },
      { id: "x", language: null },
    ]);
    expect(groups.map((g) => [g.language, g.voices.map((v) => v.id)])).toEqual([
      ["en-US", ["af_heart", "af_bella"]],
      ["en-GB", ["bf_emma"]],
      [null, ["x"]],
    ]);
  });

  it("offers the list's voices, else the loaded model's", () => {
    const heart = { id: "af_heart", name: "Heart", language: "en-US" };
    const list = [model("tts/kokoro-v1.0", "kokoro", "Kokoro", [heart], "af_heart")];
    expect(voicesFor("tts/kokoro-v1.0", list, null)).toEqual({ voices: [heart], defaultVoice: "af_heart" });
    const loaded = model("D:\\voices\\piper", "piper", "piper", [{ id: "0", name: "Irina", language: "ru" }], "0");
    expect(voicesFor("D:\\voices\\piper", list, loaded)?.voices.map((v) => v.name)).toEqual(["Irina"]);
    expect(voicesFor("D:\\other", list, loaded)).toBeNull();
    expect(voicesFor("D:\\voices\\piper", list, { ...loaded, voices: [] })).toBeNull();
  });
});
