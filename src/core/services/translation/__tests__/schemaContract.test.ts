import { describe, expect, it, vi } from "vitest";

vi.mock("@tauri-apps/api/core", () => ({ invoke: vi.fn(), isTauri: () => false }));
vi.mock("react-toastify", () => ({
  toast: Object.assign(vi.fn(), { error: vi.fn(), info: vi.fn(), warning: vi.fn(), success: vi.fn() }),
}));
vi.mock("../../systemLog", () => ({ pushSystemLog: vi.fn() }));
vi.mock("@/utils/devLog", () => ({ devLog: () => {} }));

import { TextEventSchema, TextEventType } from "@/types";
import { TRANSLATION_ENGINES } from "../index";
import { canonicalLanguage, isCjk, toAzure, toFlores, toIso6391 } from "../languages";
import {
  NLLB_MODELS,
  Service_Translation_Schema,
  SHARED_PAIR_BACKENDS,
  TRANSLATION_DEFAULT_CHAIN,
  TRANSLATION_ENGINE_IDS,
  TRANSLATION_LLM_DEFAULT_MODEL,
  Translation_Backends,
} from "../schema";
import { TranslationError } from "../types";

// P4-S0 contract tests: the persisted translation settings stay migration-safe and every backend has an engine.

describe("translation schema (P4-S0 split)", () => {
  it("fills every slice with defaults from an empty object", () => {
    const s = Service_Translation_Schema.parse({});
    expect(s.backend).toBe(Translation_Backends.azure);
    expect(s.sourceLanguage).toBe("");
    expect(s.targetLanguage).toBe("en");
    expect(s.llm.provider).toBe("bundled");
    expect(s.llm.serverModel).toBe(TRANSLATION_LLM_DEFAULT_MODEL);
    expect(s.llm.promptFormat).toBe("auto");
    expect(s.deepl.interim).toBe(false);
    expect(s.bergamot.interim).toBe(true);
    expect(s.nllb.model).toBe(NLLB_MODELS[0]);
    expect(s.auto.chain).toEqual([...TRANSLATION_DEFAULT_CHAIN]);
    expect(s.auto.interimEngine).toBe("auto");
    expect(s.glossary).toEqual([]);
    expect(s.context).toEqual({ enabled: true, turns: 3, background: "" });
    expect(s.partials.localAgreement).toBe(true);
    expect(s.partials.showTentative).toBe(false);
  });

  it("keeps settings saved before the split unchanged", () => {
    const saved = {
      backend: "libretranslate",
      autoStart: true,
      inputField: false,
      azure: { key: "k", location: "westeurope", languageFrom: "", language: "ja", profanity: "Deleted", interim: false },
      libretranslate: { endpoint: "http://10.0.0.2:5000", apiKey: "x", languageFrom: "en", language: "de", autoDetect: true, interim: false },
      onDevice: { languageFrom: "en", language: "fr", autoDetect: true, interim: false },
    };
    const s = Service_Translation_Schema.parse(saved);
    expect(s.backend).toBe(Translation_Backends.libretranslate);
    expect(s.autoStart).toBe(true);
    expect(s.inputField).toBe(false);
    expect(s.azure).toEqual(saved.azure);
    expect(s.libretranslate).toEqual(saved.libretranslate);
    expect(s.onDevice).toEqual(saved.onDevice);
  });

  it("accepts the new backend values and resets unknown ones", () => {
    for (const b of ["llm", "deepl", "bergamot", "nllb", "auto"]) {
      expect(Service_Translation_Schema.parse({ backend: b }).backend).toBe(b);
    }
    expect(Service_Translation_Schema.parse({ backend: "nope" }).backend).toBe(Translation_Backends.azure);
    expect(SHARED_PAIR_BACKENDS).not.toContain(Translation_Backends.azure);
    expect(TRANSLATION_ENGINE_IDS).not.toContain(Translation_Backends.auto);
  });

  it("drops malformed glossary entries without losing the others", () => {
    const s = Service_Translation_Schema.parse({
      glossary: [{ source: "Sigil", target: "シジル" }, null, "junk", [1], { source: "VRChat", target: "VRChat", to: "ja" }],
    });
    expect(s.glossary).toEqual([
      { source: "Sigil", target: "シジル", from: "", to: "", caseSensitive: false, enabled: true },
      { source: "VRChat", target: "VRChat", from: "", to: "ja", caseSensitive: false, enabled: true },
    ]);
    expect(Service_Translation_Schema.parse({ glossary: "nope" }).glossary).toEqual([]);
  });

  it("cleans router chains entry by entry", () => {
    const s = Service_Translation_Schema.parse({
      auto: { chain: ["deepl", "nope", "deepl", "nllb", "auto"], interimEngine: "bergamot", pairOverrides: [{ to: "ja", chain: ["llm"] }, 3] },
    });
    expect(s.auto.chain).toEqual(["deepl", "nllb"]);
    expect(s.auto.interimEngine).toBe("bergamot");
    expect(s.auto.pairOverrides).toEqual([{ from: "", to: "ja", chain: ["llm"] }]);
    expect(Service_Translation_Schema.parse({ auto: { chain: [] } }).auto.chain).toEqual([...TRANSLATION_DEFAULT_CHAIN]);
    expect(Service_Translation_Schema.parse({ auto: { interimEngine: "auto2" } }).auto.interimEngine).toBe("auto");
  });

  it("clamps numeric settings back to their defaults when out of range", () => {
    const s = Service_Translation_Schema.parse({ llm: { maxTokens: 5, timeoutMs: "abc" }, nllb: { threads: 64 } });
    expect(s.llm.maxTokens).toBe(512);
    expect(s.llm.timeoutMs).toBe(15_000);
    expect(s.nllb.threads).toBe(2);
  });
});

describe("translation engine registry", () => {
  it("registers an engine for every backend; the P4 stubs refuse to start with a TranslationError", async () => {
    for (const b of Object.values(Translation_Backends)) {
      expect(TRANSLATION_ENGINES[b], b).toBeTypeOf("function");
    }
    const engine = new TRANSLATION_ENGINES[Translation_Backends.auto]();
    const started = engine.start(Service_Translation_Schema.parse({}), new AbortController().signal);
    await expect(started).rejects.toBeInstanceOf(TranslationError);
  });
});

describe("languages.ts seed", () => {
  it("canonicalizes codes and maps them per engine", () => {
    expect(canonicalLanguage("EN_us")).toBe("en");
    expect(canonicalLanguage("zh-TW")).toBe("zh-Hant");
    expect(canonicalLanguage("zh-CN")).toBe("zh");
    expect(canonicalLanguage("auto")).toBe("");
    expect(canonicalLanguage(undefined)).toBe("");
    expect(canonicalLanguage("xx")).toBe("");
    expect(toFlores("ja")).toBe("jpn_Jpan");
    expect(toFlores("zh-Hant")).toBe("zho_Hant");
    expect(toAzure("zh")).toBe("zh-Hans");
    expect(toIso6391("pt-BR")).toBe("pt");
    expect(isCjk("ja-JP")).toBe(true);
    expect(isCjk("en")).toBe(false);
  });
});

describe("TextEvent.stableLength (P4-G captions)", () => {
  it("is optional, and malformed values are dropped instead of rejecting the event", () => {
    const ok = TextEventSchema.parse({ type: TextEventType.interim, value: "こんにちは世界", segmentId: "mic:1:4", revision: 3, stableLength: 5 });
    expect(ok.stableLength).toBe(5);
    const bad = TextEventSchema.parse({ type: TextEventType.interim, value: "x", stableLength: -1 });
    expect(bad.stableLength).toBeUndefined();
    expect(TextEventSchema.parse({ type: TextEventType.final, value: "x" }).stableLength).toBeUndefined();
  });
});
