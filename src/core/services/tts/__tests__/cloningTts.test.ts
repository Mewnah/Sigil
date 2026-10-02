import { readFileSync } from "node:fs";
import { resolve } from "node:path";
import i18n from "i18next";
import { proxy } from "valtio";
import { afterEach, beforeAll, beforeEach, describe, expect, it, vi } from "vitest";

vi.mock("@tauri-apps/api/core", () => ({ invoke: vi.fn(), isTauri: () => false }));

import { invoke } from "@tauri-apps/api/core";
import type { RustCatalogEntry, RustCloningTtsRequest, RustModelEntryStatus } from "@/types";
import { TTS_BACKENDS } from "../index";
import { Service_TTS_Schema, TTS_Backends, TTS_State } from "../schema";
import { TTS_ChatterboxLocal_Schema } from "../schema/chatterboxLocal";
import { TTS_Pocket_Schema } from "../schema/pocket";
import {
  CHATTERBOX_LOCAL_GPU_FLOW,
  CHATTERBOX_LOCAL_LANGUAGES,
  CHATTERBOX_LOCAL_MODEL,
  CHATTERBOX_PREVIEW_TEXT,
  TTS_ChatterboxLocalService,
  chatterboxLanguage,
  chatterboxOptionalIds,
  chatterboxPreview,
  chatterboxPreviewText,
  chatterboxRequest,
  chatterboxRequestLanguage,
  chatterboxSettingsProblem,
} from "../services/chatterboxLocal";
import { PREVIEW_TEXT } from "../services/localNeural";
import {
  CLONING_COMMANDS,
  POCKET_DEFAULT_MODEL,
  POCKET_MODELS,
  TTS_PocketService,
  cloningThreads,
  pocketModel,
  pocketRequest,
  unloadCloningIfIdle,
} from "../services/pocket";
import { ITTSReceiver, TtsSinkContext, ttsOutputMode } from "../types";

// P6-I: the in-process cloning backends (Pocket TTS, Chatterbox Multilingual on Sigil's ONNX Runtime).

const invokeMock = vi.mocked(invoke);
const ROOT = resolve(__dirname, "../../../../..");

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
    license: "MIT",
    status: "published",
    requires,
    files: [{ path: `${id.replace(/\//g, "_")}.bin`, size: 1000, sha256: "0".repeat(64), sources: [] }],
  } as unknown as RustCatalogEntry;
}

function status(id: string, state: RustModelEntryStatus["state"]): RustModelEntryStatus {
  return { id, state, installed_version: state === "installed" ? "1" : null, installed_path: null, bytes_on_disk: 0, adopted: false, error: null };
}

const CATALOG = [
  entry("runtime/sherpa-onnx", [], "sherpa-onnx runtime"),
  entry(POCKET_DEFAULT_MODEL, ["runtime/sherpa-onnx"], "Pocket TTS (English)"),
  entry(CHATTERBOX_LOCAL_MODEL, [], "Chatterbox Multilingual V3"),
  entry(CHATTERBOX_LOCAL_GPU_FLOW, [CHATTERBOX_LOCAL_MODEL], "Chatterbox GPU decoder"),
];

let data: TTS_State;
let models: { load: ReturnType<typeof vi.fn>; state: { catalog: RustCatalogEntry[]; states: Record<string, RustModelEntryStatus> } };
let serviceState: { status: string };

function installWindow() {
  data = proxy(Service_TTS_Schema.parse({}));
  models = { load: vi.fn(async () => {}), state: { catalog: CATALOG, states: {} } };
  serviceState = { status: "disconnected" };
  vi.stubGlobal("window", {
    ApiServer: {
      state: { services: { tts: { data } }, audioOutputDevice: "" },
      models,
      tts: { serviceState },
    },
  });
}

function installAll() {
  for (const e of CATALOG) models.state.states[e.id] = status(e.id, "installed");
}

const sink = (utteranceId: string, lang: string | null = null): TtsSinkContext => ({
  utteranceId,
  target: { route: "device", device: "", volume: 1 },
  lang,
});

beforeAll(async () => {
  await i18n.init({ lng: "en", resources: { en: { translation: {} } }, interpolation: { escapeValue: false } });
});

beforeEach(() => {
  invokeMock.mockReset();
  installWindow();
});

afterEach(() => {
  vi.unstubAllGlobals();
});

describe("settings → requests", () => {
  it("defaults are off-safe and new fields have defaults", () => {
    const p = TTS_Pocket_Schema.parse({});
    expect([p.model, p.referenceVoiceId, p.threads, p.temperature]).toEqual(["", "", "auto", "0.7"]);
    const c = TTS_ChatterboxLocal_Schema.parse({});
    expect([c.referenceVoiceId, c.language, c.ep, c.threads, c.quality, c.temperature]).toEqual(["", "auto", "auto", "auto", "auto", "0.8"]);
    // Unknown persisted values fall back instead of failing the profile.
    expect(TTS_ChatterboxLocal_Schema.parse({ ep: "tpu", quality: "ultra" })).toMatchObject({ ep: "auto", quality: "auto" });
  });

  it("builds Pocket requests", () => {
    const c = { ...TTS_Pocket_Schema.parse({}), referenceVoiceId: " alice-1a2b ", threads: "3" as const, temperature: "9" };
    const r = pocketRequest(c, "Hello.");
    expect(r).toEqual({
      model: POCKET_DEFAULT_MODEL,
      referenceVoiceId: "alice-1a2b",
      text: "Hello.",
      language: null,
      speed: 1,
      threads: 3,
      options: { temperature: 1.5 },
    } satisfies RustCloningTtsRequest);
    expect(pocketModel({ ...c, model: "tts/pocket-tts-en" })).toBe("tts/pocket-tts-en");
    expect(pocketModel({ ...c, model: "tts/unknown" })).toBe(POCKET_DEFAULT_MODEL);
    expect(pocketRequest({ ...c, temperature: "abc", threads: "auto" }, "").options).toEqual({ temperature: 0.7 });
    expect(cloningThreads("auto")).toBeNull();
    expect(cloningThreads("6")).toBeNull();
    expect(cloningThreads("6", 8)).toBe(6);
  });

  it("builds Chatterbox requests with the language, options and quality", () => {
    const c = { ...TTS_ChatterboxLocal_Schema.parse({}), referenceVoiceId: "bob", cfgWeight: "2", exaggeration: "0.1", quality: "best" as const };
    const r = chatterboxRequest(c, "Hola.", "es-MX");
    expect(r.model).toBe(CHATTERBOX_LOCAL_MODEL);
    expect(r.language).toBe("es");
    expect(r.threads).toBeNull();
    expect(r.options).toEqual({ ep: "auto", exaggeration: 0.25, cfgWeight: 1, temperature: 0.8, flowSteps: 10 });
    expect(chatterboxRequest({ ...c, quality: "auto" }, "x", null).options).not.toHaveProperty("flowSteps");
    expect(chatterboxRequest({ ...c, quality: "fast", threads: "8" }, "x", null)).toMatchObject({ threads: 8, options: { flowSteps: 6 } });
    // A fixed language wins over the caption's; auto takes the caption's when the model speaks it.
    expect(chatterboxRequestLanguage("ja", "en")).toBe("ja");
    expect(chatterboxRequestLanguage("auto", "pt-BR")).toBe("pt");
    expect(chatterboxRequestLanguage("auto", "vi")).toBeNull();
    expect(chatterboxRequestLanguage("auto", null)).toBeNull();
    expect(chatterboxLanguage("nb")).toBe("no");
    expect(chatterboxLanguage("zh-Hant")).toBe("zh");
    expect(chatterboxLanguage("iw")).toBe("he");
    expect(chatterboxSettingsProblem({ ...c, referenceVoiceId: "  " })).toBe("tts.chatterbox_local_voice_required");
    expect(chatterboxSettingsProblem(c)).toBeNull();
    expect(chatterboxOptionalIds({ ...c, ep: "cpu" })).toEqual([]);
    expect(chatterboxOptionalIds({ ...c, ep: "auto" })).toEqual([CHATTERBOX_LOCAL_GPU_FLOW]);
  });

  it("the Chatterbox preview tags the line it speaks with that line's language", () => {
    // Every language Chatterbox speaks has a sample line in that language.
    for (const lang of CHATTERBOX_LOCAL_LANGUAGES) {
      const p = chatterboxPreview(lang, "en", "");
      expect(p.language).toBe(lang);
      if (lang !== "en") expect(p.text).not.toBe(PREVIEW_TEXT.en);
      expect(chatterboxPreviewText(lang, "en")).toBe(p.text);
    }
    expect(chatterboxPreview("ko", "en", "")).toEqual({ text: CHATTERBOX_PREVIEW_TEXT.ko, language: "ko" });
    expect(chatterboxPreview("auto", "de", "")).toEqual({ text: PREVIEW_TEXT.de, language: "de" });
    expect(chatterboxPreview("auto", "he", "")).toEqual({ text: CHATTERBOX_PREVIEW_TEXT.he, language: "he" });
    expect(chatterboxPreview("auto", "pt_br", "")).toEqual({ text: PREVIEW_TEXT.pt, language: "pt" });
    // A UI language Chatterbox does not speak: the English line, tagged English.
    expect(chatterboxPreview("auto", "vi", "")).toEqual({ text: PREVIEW_TEXT.en, language: "en" });
    expect(chatterboxPreviewText("auto", "vi")).toBe(PREVIEW_TEXT.en);
    // A typed line in Auto: no tag (Rust goes by the script), whatever the UI language.
    expect(chatterboxPreview("auto", "de", "  Hello, this is my voice ")).toEqual({ text: "Hello, this is my voice", language: null });
    // A typed line with a chosen language: that language (as every sentence then).
    expect(chatterboxPreview("ja", "de", "こんにちは")).toEqual({ text: "こんにちは", language: "ja" });
  });
});

describe("mirrors of the Rust tables", () => {
  const rust = (rel: string) => readFileSync(resolve(ROOT, rel), "utf8");

  it("Chatterbox languages equal chatterbox_ort::text::LANGUAGES", () => {
    const src = rust("src-tauri/src/services/local_tts/chatterbox_ort/text.rs");
    const block = /pub const LANGUAGES: \[&str; \d+\] = \[([^\]]*)\]/.exec(src)?.[1] ?? "";
    const langs = [...block.matchAll(/"([a-z]+)"/g)].map((m) => m[1]);
    expect(langs).toEqual([...CHATTERBOX_LOCAL_LANGUAGES]);
  });

  it("model ids match the Rust constants and the catalog fragment", () => {
    const pocket = rust("src-tauri/src/services/local_tts/pocket.rs");
    expect(pocket).toContain(`pub const DEFAULT_MODEL: &str = "${POCKET_MODELS[0]}";`);
    expect(pocket).toContain(`"${POCKET_MODELS[1]}"`);
    const chatterbox = rust("src-tauri/src/services/local_tts/chatterbox_ort.rs");
    expect(chatterbox).toContain(`pub const MODEL_ID: &str = "${CHATTERBOX_LOCAL_MODEL}";`);
    expect(chatterbox).toContain(`pub const GPU_FLOW_ID: &str = "${CHATTERBOX_LOCAL_GPU_FLOW}";`);
    const catalog = JSON.parse(rust("src-tauri/model-catalog/tts-cloning.json")) as { entries: { id: string; requires?: string[] }[] };
    const ids = catalog.entries.map((e) => e.id);
    for (const id of [...POCKET_MODELS, CHATTERBOX_LOCAL_MODEL, CHATTERBOX_LOCAL_GPU_FLOW]) expect(ids).toContain(id);
    expect(catalog.entries.find((e) => e.id === POCKET_DEFAULT_MODEL)?.requires).toContain("runtime/sherpa-onnx");
    expect(catalog.entries.find((e) => e.id === CHATTERBOX_LOCAL_GPU_FLOW)?.requires).toContain(CHATTERBOX_LOCAL_MODEL);
  });

  it("the commands are registered for the local-tts plugin", () => {
    const list = rust("src-tauri/plugin-commands/local-tts.txt");
    for (const engine of Object.values(CLONING_COMMANDS)) {
      for (const cmd of Object.values(engine)) {
        expect(cmd.startsWith("plugin:local-tts|")).toBe(true);
        expect(list).toContain(cmd.slice("plugin:local-tts|".length));
      }
    }
  });
});

describe("backends", () => {
  it("are registered and speak through the sink", () => {
    expect(TTS_BACKENDS[TTS_Backends.pocket]).toBe(TTS_PocketService);
    expect(TTS_BACKENDS[TTS_Backends.chatterboxLocal]).toBe(TTS_ChatterboxLocalService);
    expect(ttsOutputMode(new TTS_PocketService(receiver()))).toBe("sink");
    expect(ttsOutputMode(new TTS_ChatterboxLocalService(receiver()))).toBe("sink");
  });

  it("start loads the model and prepares the voice, then sentences go to their utterances", async () => {
    installAll();
    data.pocket.referenceVoiceId = "alice";
    invokeMock.mockResolvedValue({ sampleRate: 24000, frames: 0, firstAudioMs: 0, totalMs: 0 });
    const r = receiver();
    const s = new TTS_PocketService(r);
    await s.start(data);
    expect(invokeMock).toHaveBeenCalledWith(CLONING_COMMANDS.pocket.speak, {
      request: expect.objectContaining({ text: "", referenceVoiceId: "alice", model: POCKET_DEFAULT_MODEL }),
      utteranceId: "",
    });
    expect(r.onStart).toHaveBeenCalledTimes(1);
    invokeMock.mockClear();
    await s.synthesizeToSink("Hello there.", sink("u1"), new AbortController().signal);
    expect(invokeMock).toHaveBeenCalledWith(CLONING_COMMANDS.pocket.speak, {
      request: expect.objectContaining({ text: "Hello there." }),
      utteranceId: "u1",
    });
    // Empty sentences are not sent.
    invokeMock.mockClear();
    await s.synthesizeToSink("  ", sink("u2"), new AbortController().signal);
    expect(invokeMock).not.toHaveBeenCalled();
  });

  it("Chatterbox refuses to start without a voice, and before the download", async () => {
    const r = receiver();
    await new TTS_ChatterboxLocalService(r).start(data);
    expect(r.onStop).toHaveBeenCalledWith("tts.chatterbox_local_voice_required");
    expect(invokeMock).not.toHaveBeenCalled();

    data.chatterboxLocal.referenceVoiceId = "bob";
    const r2 = receiver();
    await new TTS_ChatterboxLocalService(r2).start(data);
    expect(r2.onStop).toHaveBeenCalledWith("tts.cloning_not_downloaded");
    expect(invokeMock).not.toHaveBeenCalled();

    // Only the model itself is required (the graphics-card decoder is optional).
    models.state.states[CHATTERBOX_LOCAL_MODEL] = status(CHATTERBOX_LOCAL_MODEL, "installed");
    invokeMock.mockResolvedValue({ sampleRate: 0, frames: 0, firstAudioMs: 0, totalMs: 0 });
    const r3 = receiver();
    await new TTS_ChatterboxLocalService(r3).start(data);
    expect(r3.onStart).toHaveBeenCalled();
    expect(invokeMock.mock.calls[0][0]).toBe(CLONING_COMMANDS.chatterbox.speak);
  });

  it("a failing load stops with the engine's reason", async () => {
    installAll();
    data.chatterboxLocal.referenceVoiceId = "bob";
    invokeMock.mockRejectedValue("reference voice \"bob\": voice not found");
    const r = receiver();
    await new TTS_ChatterboxLocalService(r).start(data);
    expect(r.onStop).toHaveBeenCalledWith('Chatterbox: reference voice "bob": voice not found');
    expect(r.onStart).not.toHaveBeenCalled();
  });

  it("sentences carry the caption language; abort cancels; a cancelled sentence is no error", async () => {
    installAll();
    data.chatterboxLocal.referenceVoiceId = "bob";
    invokeMock.mockResolvedValue({ sampleRate: 24000, frames: 0, firstAudioMs: 0, totalMs: 0 });
    const s = new TTS_ChatterboxLocalService(receiver());
    await s.start(data);
    invokeMock.mockClear();
    await s.synthesizeToSink("Guten Tag.", sink("u1", "de"), new AbortController().signal);
    const [, args] = invokeMock.mock.calls[0] as [string, { request: RustCloningTtsRequest }];
    expect(args.request.language).toBe("de");

    // Abort while the engine works: cancel is sent for that utterance, and Rust's "cancelled" is swallowed.
    let reject: (e: unknown) => void = () => {};
    invokeMock.mockReset();
    invokeMock.mockImplementation((cmd: string) =>
      cmd === CLONING_COMMANDS.chatterbox.speak ? new Promise((_, rej) => (reject = rej)) : Promise.resolve(undefined),
    );
    const ctl = new AbortController();
    const pending = s.synthesizeToSink("Long sentence.", sink("u2"), ctl.signal);
    ctl.abort();
    expect(invokeMock).toHaveBeenCalledWith(CLONING_COMMANDS.chatterbox.cancel, { utteranceId: "u2" });
    reject("cancelled");
    await expect(pending).resolves.toBeUndefined();

    // A real failure of one sentence is reported with the label.
    invokeMock.mockReset();
    invokeMock.mockRejectedValue("the language model failed: out of memory");
    await expect(s.synthesizeToSink("Again.", sink("u3"), new AbortController().signal)).rejects.toThrow(
      "Chatterbox: the language model failed: out of memory",
    );
  });

  it("stop cancels everything and frees the model, once", async () => {
    installAll();
    invokeMock.mockResolvedValue(undefined);
    const r = receiver();
    const s = new TTS_PocketService(r);
    await s.start(data);
    invokeMock.mockClear();
    s.stop();
    await flush();
    expect(invokeMock.mock.calls).toEqual([
      [CLONING_COMMANDS.pocket.cancel, { utteranceId: null }],
      [CLONING_COMMANDS.pocket.unload, undefined],
    ]);
    expect(r.onStop).toHaveBeenCalledWith();
    invokeMock.mockClear();
    s.dispose();
    await flush();
    expect(invokeMock).not.toHaveBeenCalled();
    // Not running: sentences are ignored.
    await s.synthesizeToSink("Late.", sink("u9"), new AbortController().signal);
    expect(invokeMock).not.toHaveBeenCalled();
  });

  it("a stop during start wins over the late load result", async () => {
    installAll();
    let resolve: (v: unknown) => void = () => {};
    invokeMock.mockImplementation((cmd: string) => (cmd === CLONING_COMMANDS.pocket.speak ? new Promise((res) => (resolve = res)) : Promise.resolve()));
    const r = receiver();
    const s = new TTS_PocketService(r);
    const starting = s.start(data);
    await flush();
    s.stop();
    resolve({});
    await starting;
    expect(r.onStart).not.toHaveBeenCalled();
    expect(r.onStop).toHaveBeenCalledTimes(1);
  });

  it("previews free the model unless that backend is running", () => {
    invokeMock.mockResolvedValue(undefined);
    unloadCloningIfIdle("chatterbox", TTS_Backends.chatterboxLocal);
    expect(invokeMock).toHaveBeenCalledWith(CLONING_COMMANDS.chatterbox.unload, undefined);
    invokeMock.mockClear();
    serviceState.status = "connected";
    data.backend = TTS_Backends.chatterboxLocal;
    unloadCloningIfIdle("chatterbox", TTS_Backends.chatterboxLocal);
    expect(invokeMock).not.toHaveBeenCalled();
  });
});
