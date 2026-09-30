import i18n from "i18next";
import { beforeAll, beforeEach, describe, expect, it, vi } from "vitest";

vi.mock("@tauri-apps/api/core", () => ({ invoke: vi.fn() }));

import { invoke } from "@tauri-apps/api/core";
import { Service_Translation_Schema, Translation_State } from "../schema";
import {
  deeplContext,
  deeplError,
  deeplErrorStatus,
  deeplFormality,
  deeplSourceCode,
  deeplTargetCandidates,
  pickDeeplTarget,
  Translation_DeeplService,
} from "../services/deepl";
import { TranslationEngineHost, TranslationError, TranslationStatusUpdate } from "../types";

const invokeMock = vi.mocked(invoke);

function state(patch: (s: Translation_State) => void = () => {}): Translation_State {
  const s = Service_Translation_Schema.parse({});
  s.deepl.apiKey = " key:fx ";
  s.targetLanguage = "en";
  patch(s);
  return s;
}

const live = () => new AbortController().signal;

function host(): TranslationEngineHost & { updates: TranslationStatusUpdate[] } {
  const updates: TranslationStatusUpdate[] = [];
  return { engines: {} as TranslationEngineHost["engines"], status: (u) => updates.push(u), updates };
}

const LANGS = {
  source: [{ code: "EN" }, { code: "JA" }, { code: "DE" }, { code: "ZH" }, { code: "PT" }],
  target: [{ code: "EN-US" }, { code: "EN-GB" }, { code: "JA" }, { code: "DE" }, { code: "ZH" }, { code: "PT-BR" }],
};

function route(handlers: Record<string, (args: Record<string, unknown>) => unknown>) {
  invokeMock.mockImplementation(async (cmd: string, args?: unknown) => {
    const h = handlers[cmd];
    if (!h) return undefined;
    return h((args ?? {}) as Record<string, unknown>);
  });
}

function healthy(extra: Record<string, (args: Record<string, unknown>) => unknown> = {}) {
  route({
    "plugin:translate|deepl_usage": () => ({ characterCount: 10, characterLimit: 500_000 }),
    "plugin:translate|deepl_languages": (a) => LANGS[a.kind as "source" | "target"],
    ...extra,
  });
}

const calls = (cmd: string) => invokeMock.mock.calls.filter(([c]) => c === cmd).map(([, a]) => a as Record<string, unknown>);

beforeAll(async () => {
  await i18n.init({ lng: "en", resources: { en: { translation: {} } } });
});

beforeEach(() => {
  invokeMock.mockReset();
});

describe("DeepL language codes", () => {
  it("maps canonical codes to DeepL source codes", () => {
    expect(deeplSourceCode("en")).toBe("EN");
    expect(deeplSourceCode("zh-Hant")).toBe("ZH");
    expect(deeplSourceCode("pt-PT")).toBe("PT");
    expect(deeplSourceCode("ja")).toBe("JA");
    expect(deeplSourceCode("")).toBeNull();
    expect(deeplSourceCode("not-a-language")).toBeNull();
  });

  it("prefers regional and script target codes", () => {
    expect(deeplTargetCandidates("en")).toEqual(["EN-US", "EN-GB", "EN"]);
    expect(deeplTargetCandidates("pt")).toEqual(["PT-BR", "PT-PT", "PT"]);
    expect(deeplTargetCandidates("pt-PT")[0]).toBe("PT-PT");
    expect(deeplTargetCandidates("zh")).toEqual(["ZH-HANS", "ZH"]);
    expect(deeplTargetCandidates("zh-TW")).toEqual(["ZH-HANT"]);
    expect(deeplTargetCandidates("ko")).toEqual(["KO"]);
    expect(deeplTargetCandidates("")).toEqual([]);
    const available = new Set(["EN-GB", "ZH", "JA"]);
    expect(pickDeeplTarget("en", available)).toBe("EN-GB");
    expect(pickDeeplTarget("zh", available), "older accounts only list ZH").toBe("ZH");
    expect(pickDeeplTarget("zh-Hant", available)).toBeNull();
    expect(pickDeeplTarget("en", null)).toBe("EN-US");
  });

  it("sends only the lenient formality values", () => {
    expect(deeplFormality("default")).toBeNull();
    expect(deeplFormality("more")).toBe("prefer_more");
    expect(deeplFormality("less")).toBe("prefer_less");
    expect(deeplFormality("prefer_more")).toBe("prefer_more");
  });

  it("builds context from the last source lines", () => {
    expect(deeplContext(undefined)).toBeNull();
    expect(deeplContext([])).toBeNull();
    const pairs = ["a", "b", "c", "d"].map((s) => ({ source: s, target: s.toUpperCase() }));
    expect(deeplContext(pairs)).toBe("b\nc\nd");
    const long = deeplContext([{ source: "x".repeat(5000), target: "" }]);
    expect(long?.length).toBe(1500);
  });
});

describe("DeepL errors", () => {
  it("reads the status from the Rust message", () => {
    expect(deeplErrorStatus("DeepL translation: HTTP 456: the monthly character quota is used up")).toBe(456);
    expect(deeplErrorStatus("DeepL translation (https://x): cannot connect")).toBeUndefined();
  });

  it("classifies failures", () => {
    const auth = deeplError("DeepL translation: HTTP 403: the API key was rejected");
    expect([auth.status, auth.retryable]).toEqual([403, false]);
    const quota = deeplError("DeepL translation: HTTP 456: quota");
    expect([quota.status, quota.retryable]).toEqual([456, false]);
    expect(deeplError("DeepL translation: HTTP 429: too many").retryable).toBe(true);
    expect(deeplError("DeepL translation: HTTP 529: busy").retryable).toBe(true);
    expect(deeplError("DeepL translation: HTTP 503: unavailable").retryable).toBe(true);
    expect(deeplError("DeepL translation: HTTP 400: bad request").retryable).toBe(false);
    expect(deeplError("DeepL: invalid target language code").retryable).toBe(false);
    expect(deeplError("DeepL translation (https://api-free.deepl.com/v2/translate): no response within 10 s (timed out).").retryable).toBe(
      true,
    );
    expect(deeplError(new Error("DeepL usage: HTTP 403: x"))).toBeInstanceOf(TranslationError);
  });
});

describe("Translation_DeeplService", () => {
  it("checks the key and loads the language lists on start", async () => {
    healthy();
    const engine = new Translation_DeeplService();
    await engine.start(state((s) => (s.deepl.plan = "free")), live());
    expect(calls("plugin:translate|deepl_usage")).toEqual([{ apiKey: "key:fx", plan: "free" }]);
    expect(calls("plugin:translate|deepl_languages").map((a) => a.kind).sort()).toEqual(["source", "target"]);
    expect(engine.supports("ja", "en")).toBe(true);
    expect(engine.supports("ko", "en"), "KO is not in this account's source list").toBe(false);
    expect(engine.supports("en", "ko")).toBe(false);
    expect(engine.supports("", "de")).toBe(true);
  });

  it("refuses to start without a key, with a rejected key or a used-up quota", async () => {
    await expect(new Translation_DeeplService().start(state((s) => (s.deepl.apiKey = "  ")), live())).rejects.toMatchObject({
      retryable: false,
    });
    route({
      "plugin:translate|deepl_usage": () => {
        throw "DeepL usage: HTTP 403: the API key was rejected";
      },
    });
    await expect(new Translation_DeeplService().start(state(), live())).rejects.toMatchObject({ status: 403 });
    route({ "plugin:translate|deepl_usage": () => ({ characterCount: 500_000, characterLimit: 500_000 }) });
    await expect(new Translation_DeeplService().start(state(), live())).rejects.toMatchObject({ status: 456 });
  });

  it("starts when DeepL is unreachable and uses the built-in language lists", async () => {
    route({
      "plugin:translate|deepl_usage": () => {
        throw "DeepL usage (https://api-free.deepl.com/v2/usage): cannot connect. Is the server running? (dns)";
      },
      "plugin:translate|deepl_languages": () => {
        throw "offline";
      },
    });
    const engine = new Translation_DeeplService();
    await engine.start(state(), live());
    expect(engine.supports("ko", "en")).toBe(true);
  });

  it("warns near the end of the monthly quota", async () => {
    healthy({ "plugin:translate|deepl_usage": () => ({ characterCount: 460_000, characterLimit: 500_000 }) });
    const h = host();
    const engine = new Translation_DeeplService();
    engine.attach(h);
    await engine.start(state(), live());
    expect(h.updates).toEqual([{ text: "transl.deepl_status_quota", level: "warning" }]);
  });

  it("rejects a target DeepL does not offer", async () => {
    healthy();
    await expect(new Translation_DeeplService().start(state((s) => (s.targetLanguage = "ko")), live())).rejects.toMatchObject({
      retryable: false,
    });
  });

  it("sends the mapped request", async () => {
    healthy({ "plugin:translate|deepl_translate": () => ({ text: "Hello", detectedSource: "JA", billedCharacters: 5 }) });
    const engine = new Translation_DeeplService();
    await engine.start(
      state((s) => {
        s.deepl.formality = "more";
        s.deepl.modelType = "quality_optimized";
      }),
      live(),
    );
    const text = await engine.translate(
      { text: "こんにちは", kind: "final", from: "ja", context: [{ source: "前の行", target: "previous" }] },
      live(),
    );
    expect(text).toBe("Hello");
    expect(calls("plugin:translate|deepl_translate")[0].request).toEqual({
      text: "こんにちは",
      source: "JA",
      target: "EN-US",
      apiKey: "key:fx",
      plan: "auto",
      formality: "prefer_more",
      context: "前の行",
      modelType: "quality_optimized",
      timeoutMs: 10_000,
    });
  });

  it("lets DeepL detect unknown sources and leaves out context when disabled", async () => {
    healthy({ "plugin:translate|deepl_translate": () => ({ text: "x" }) });
    const engine = new Translation_DeeplService();
    await engine.start(state((s) => (s.deepl.sendContext = false)), live());
    await engine.translate({ text: "안녕", kind: "interim", from: "ko", context: [{ source: "a", target: "b" }] }, live());
    await engine.translate({ text: "hola", kind: "interim" }, live());
    const [ko, unknown] = calls("plugin:translate|deepl_translate").map((a) => a.request as Record<string, unknown>);
    expect(ko).toMatchObject({ source: null, context: null, formality: null, modelType: null, timeoutMs: 5_000 });
    expect(unknown.source).toBeNull();
  });

  it("returns a line already in the target language unchanged", async () => {
    healthy();
    const engine = new Translation_DeeplService();
    await engine.start(state(), live());
    await expect(engine.translate({ text: "Hi", kind: "final", from: "en" }, live())).resolves.toBe("Hi");
    expect(calls("plugin:translate|deepl_translate")).toHaveLength(0);
  });

  it("maps translation failures and stops cleanly", async () => {
    healthy({
      "plugin:translate|deepl_translate": () => {
        throw "DeepL translation: HTTP 456: the monthly character quota is used up";
      },
    });
    const engine = new Translation_DeeplService();
    await engine.start(state(), live());
    const e = await engine.translate({ text: "hola", kind: "final", from: "es" }, live()).catch((x: unknown) => x);
    expect(e).toMatchObject({ status: 456, retryable: false });
    engine.stop();
    await expect(engine.translate({ text: "hola", kind: "final" }, live())).rejects.toBeInstanceOf(TranslationError);
  });

  it("static helpers never throw", async () => {
    route({
      "plugin:translate|deepl_usage": () => {
        throw "down";
      },
      "plugin:translate|deepl_languages": () => {
        throw "down";
      },
    });
    await expect(Translation_DeeplService.getUsage("k")).resolves.toBeNull();
    await expect(Translation_DeeplService.getUsage("  ")).resolves.toBeNull();
    await expect(Translation_DeeplService.getLanguages("k", "auto", "target")).resolves.toEqual([]);
    healthy();
    await expect(Translation_DeeplService.getUsage("k", "pro")).resolves.toEqual({ characterCount: 10, characterLimit: 500_000 });
    expect(calls("plugin:translate|deepl_usage").at(-1)).toEqual({ apiKey: "k", plan: "pro" });
    expect(Translation_DeeplService.targetCode("pt")).toBe("PT-BR");
    expect(Translation_DeeplService.targetCode("en", [{ code: "EN-GB", name: "", supportsFormality: false }])).toBe("EN-GB");
  });
});
