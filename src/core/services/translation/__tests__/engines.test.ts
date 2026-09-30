import i18n from "i18next";
import { afterEach, beforeAll, beforeEach, describe, expect, it, vi } from "vitest";

vi.mock("@tauri-apps/api/core", () => ({ invoke: vi.fn() }));
vi.mock("@/utils/devLog", () => ({ devLog: () => {} }));

import { invoke } from "@tauri-apps/api/core";
import { Service_Translation_Schema, Translation_State } from "../schema";
import {
  azureHeaders,
  azurePair,
  azureTranslateUrl,
  parseAzureTranslation,
  Translation_AzureService,
} from "../services/azure";
import { isRetryableTranslationError } from "../retry";
import {
  libreErrorStatus,
  libreLanguageCode,
  libreTranslationError,
  Translation_LibreTranslateService,
} from "../services/libretranslate";
import { Translation_OnDeviceService } from "../services/onDevice";
import { TranslationError } from "../types";

const invokeMock = vi.mocked(invoke);

function state(patch: (s: Translation_State) => void = () => {}): Translation_State {
  const s = Service_Translation_Schema.parse({});
  patch(s);
  return s;
}

const live = () => new AbortController().signal;

beforeAll(async () => {
  await i18n.init({ lng: "en", resources: { en: { translation: {} } } });
});

beforeEach(() => {
  invokeMock.mockReset();
});

afterEach(() => {
  vi.unstubAllGlobals();
});

describe("Azure", () => {
  it("omits the region header for global resources and sends it when set", () => {
    expect(azureHeaders({ key: " k ", location: "  " })).toEqual({
      "Ocp-Apim-Subscription-Key": "k",
      "Content-Type": "application/json",
    });
    expect(azureHeaders({ key: "k", location: "westeurope" })["Ocp-Apim-Subscription-Region"]).toBe("westeurope");
  });

  it("builds the URL, leaving `from` out when blank so Azure auto-detects", () => {
    const url = new URL(azureTranslateUrl({ languageFrom: "", language: "ja", profanity: "Marked" }));
    expect(url.searchParams.get("api-version")).toBe("3.0");
    expect(url.searchParams.has("from")).toBe(false);
    expect(url.searchParams.get("to")).toBe("ja");
    expect(url.searchParams.get("profanityAction")).toBe("Marked");
    const withFrom = new URL(azureTranslateUrl({ languageFrom: "zh-Hans", language: "en", profanity: "NoAction" }));
    expect(withFrom.searchParams.get("from")).toBe("zh-Hans");
  });

  it("starts without a location but not without a key", async () => {
    await expect(new Translation_AzureService().start(state((s) => (s.azure.key = "k")), live())).resolves.toBeUndefined();
    await expect(new Translation_AzureService().start(state(), live())).rejects.toThrow("transl.azure_error_key_missing");
  });

  it("returns the translation", async () => {
    const fetchMock = vi.fn(async () => new Response(JSON.stringify([{ translations: [{ text: "hola", to: "es" }] }])));
    vi.stubGlobal("fetch", fetchMock);
    const az = new Translation_AzureService();
    await az.start(state((s) => { s.azure.key = "k"; s.azure.language = "es"; }), live());
    await expect(az.translate({ text: "hello", kind: "final" }, live())).resolves.toBe("hola");
    const [, init] = fetchMock.mock.calls[0] as unknown as [string, RequestInit];
    expect(JSON.parse(init.body as string)).toEqual([{ text: "hello" }]);
    expect((init.headers as Record<string, string>)["Ocp-Apim-Subscription-Region"]).toBeUndefined();
  });

  it("turns a network failure into a retryable error instead of an unhandled rejection", async () => {
    vi.stubGlobal("fetch", vi.fn(async () => { throw new TypeError("Failed to fetch"); }));
    const az = new Translation_AzureService();
    await az.start(state((s) => (s.azure.key = "k")), live());
    const err = await az.translate({ text: "x", kind: "final" }, live()).catch((e) => e);
    expect(err).toBeInstanceOf(TranslationError);
    expect(err.retryable).toBe(true);
  });

  it("reports auth failures as non-retryable with Azure's reason", async () => {
    vi.stubGlobal("fetch", vi.fn(async () => new Response(JSON.stringify({ error: { code: 401000, message: "bad key" } }), { status: 401 })));
    const az = new Translation_AzureService();
    await az.start(state((s) => (s.azure.key = "k")), live());
    const err = await az.translate({ text: "x", kind: "final" }, live()).catch((e) => e);
    expect(err).toBeInstanceOf(TranslationError);
    expect(err.status).toBe(401);
    expect(err.retryable).toBe(false);
    expect(err.message).toBe("transl.azure_error_auth");
  });

  it("marks 429 and 5xx retryable", async () => {
    vi.stubGlobal("fetch", vi.fn(async () => new Response("busy", { status: 429, statusText: "Too Many Requests" })));
    const az = new Translation_AzureService();
    await az.start(state((s) => (s.azure.key = "k")), live());
    const err = await az.translate({ text: "x", kind: "interim" }, live()).catch((e) => e);
    expect(err.retryable).toBe(true);
  });

  it("rethrows the abort reason when the caller aborts", async () => {
    vi.stubGlobal("fetch", vi.fn((_: string, init: RequestInit) => new Promise((_r, reject) => {
      init.signal?.addEventListener("abort", () => reject(new DOMException("Aborted", "AbortError")));
    })));
    const az = new Translation_AzureService();
    await az.start(state((s) => (s.azure.key = "k")), live());
    const c = new AbortController();
    const p = az.translate({ text: "x", kind: "interim" }, c.signal);
    const reason = new Error("superseded");
    c.abort(reason);
    await expect(p).rejects.toBe(reason);
  });

  it("rejects malformed bodies", () => {
    expect(() => parseAzureTranslation({})).toThrow(TranslationError);
    expect(parseAzureTranslation([{ translations: [{ text: "" }] }])).toBe("");
  });
});

describe("LibreTranslate", () => {
  const libreState = () =>
    state((s) => {
      s.libretranslate.endpoint = " http://lt:5000 ";
      s.libretranslate.apiKey = "";
      s.libretranslate.language = "de";
    });

  it("passes endpoint and key per call and never touches the global endpoint", async () => {
    invokeMock.mockImplementation(async (cmd: string) => (cmd === "plugin:translate|check_availability" ? true : "hallo"));
    const lt = new Translation_LibreTranslateService();
    await lt.start(libreState(), live());
    await expect(lt.translate({ text: "hello", kind: "final" }, live())).resolves.toBe("hallo");

    const cmds = invokeMock.mock.calls.map((c) => c[0]);
    expect(cmds).not.toContain("plugin:translate|set_endpoint");
    expect(cmds).not.toContain("plugin:translate|set_api_key");
    expect(invokeMock).toHaveBeenCalledWith("plugin:translate|check_availability", { endpoint: "http://lt:5000", apiKey: "" });
    expect(invokeMock).toHaveBeenCalledWith("plugin:translate|translate", {
      text: "hello",
      source: "en",
      target: "de",
      endpoint: "http://lt:5000",
      apiKey: "",
    });
  });

  it("fails to start when the server does not answer", async () => {
    invokeMock.mockResolvedValue(false);
    await expect(new Translation_LibreTranslateService().start(libreState(), live())).rejects.toThrow(
      "transl.libretranslate_error_unavailable",
    );
  });

  it("stops waiting for the availability probe as soon as start is aborted", async () => {
    invokeMock.mockImplementation(() => new Promise(() => {}));
    const c = new AbortController();
    const p = new Translation_LibreTranslateService().start(libreState(), c.signal);
    c.abort(new Error("stopped"));
    await expect(p).rejects.toThrow("stopped");
  });

  it("falls back to the configured source language when detection fails", async () => {
    invokeMock.mockImplementation(async (cmd: string) => {
      if (cmd === "plugin:translate|check_availability") return true;
      if (cmd === "plugin:translate|detect_language") throw "no detection";
      return "ok";
    });
    const lt = new Translation_LibreTranslateService();
    await lt.start(state((s) => { s.libretranslate.autoDetect = true; s.libretranslate.languageFrom = "fr"; }), live());
    await lt.translate({ text: "bonjour", kind: "final" }, live());
    expect(invokeMock).toHaveBeenLastCalledWith("plugin:translate|translate", expect.objectContaining({ source: "fr" }));
  });

  it("wraps Rust errors in a TranslationError", async () => {
    invokeMock.mockImplementation(async (cmd: string) => {
      if (cmd === "plugin:translate|check_availability") return true;
      throw "LibreTranslate translation: HTTP 502 Bad Gateway";
    });
    const lt = new Translation_LibreTranslateService();
    await lt.start(libreState(), live());
    const err = await lt.translate({ text: "x", kind: "final" }, live()).catch((e) => e);
    expect(err).toBeInstanceOf(TranslationError);
    expect(err.message).toContain("HTTP 502");
  });

  it("marks permanent HTTP errors non-retryable and transient ones retryable", async () => {
    let failure = "";
    invokeMock.mockImplementation(async (cmd: string) => {
      if (cmd === "plugin:translate|check_availability") return true;
      throw failure;
    });
    const lt = new Translation_LibreTranslateService();
    await lt.start(libreState(), live());
    const attempt = async (message: string) => {
      failure = message;
      return (await lt.translate({ text: "x", kind: "final" }, live()).catch((e) => e)) as TranslationError;
    };

    const forbidden = await attempt("LibreTranslate translation: HTTP 403 Forbidden: Invalid API key");
    expect(forbidden).toBeInstanceOf(TranslationError);
    expect(forbidden.status).toBe(403);
    expect(forbidden.retryable).toBe(false);
    expect(isRetryableTranslationError(forbidden)).toBe(false);
    expect((await attempt("LibreTranslate translation: HTTP 400 Bad Request: xx is not supported")).retryable).toBe(false);
    expect((await attempt("LibreTranslate translation: HTTP 429 Too Many Requests")).retryable).toBe(true);
    expect((await attempt("LibreTranslate translation: HTTP 503 Service Unavailable")).status).toBe(503);
    expect((await attempt("LibreTranslate translation (http://lt:5000/translate): cannot connect. Is the server running? (refused)")).retryable).toBe(true);
    expect((await attempt("Invalid endpoint URL 'lt:x': invalid port number")).retryable).toBe(false);
    expect((await attempt("LibreTranslate translation: unexpected response (missing field)")).retryable).toBe(false);
  });

  it("reads the status out of Rust error strings", () => {
    expect(libreErrorStatus("LibreTranslate translation: HTTP 404 Not Found")).toBe(404);
    expect(libreErrorStatus("no status here")).toBeUndefined();
    expect(libreTranslationError(new Error("LibreTranslate translation: HTTP 502 Bad Gateway")).retryable).toBe(true);
  });

  it("passes a line through when the detected language is already the target", async () => {
    invokeMock.mockImplementation(async (cmd: string) => {
      if (cmd === "plugin:translate|check_availability") return true;
      if (cmd === "plugin:translate|detect_language") return "de";
      return "should not be called";
    });
    const lt = new Translation_LibreTranslateService();
    await lt.start(state((s) => { s.libretranslate.autoDetect = true; s.libretranslate.language = "de"; }), live());
    await expect(lt.translate({ text: "Guten Morgen", kind: "final" }, live())).resolves.toBe("Guten Morgen");
    expect(invokeMock.mock.calls.map((c) => c[0])).not.toContain("plugin:translate|translate");
  });

  it("lists languages for the endpoint it is given", async () => {
    invokeMock.mockResolvedValue([{ code: "en", name: "English" }]);
    await expect(Translation_LibreTranslateService.getLanguages("http://probe:5000", " key ")).resolves.toEqual([
      { code: "en", name: "English" },
    ]);
    expect(invokeMock).toHaveBeenCalledWith("plugin:translate|get_languages", { endpoint: "http://probe:5000", apiKey: "key" });
    invokeMock.mockRejectedValue("down");
    await expect(Translation_LibreTranslateService.getLanguages("http://probe:5000")).resolves.toEqual([]);
    invokeMock.mockClear();
    await expect(Translation_LibreTranslateService.getLanguages("  ")).resolves.toEqual([]);
    expect(invokeMock).not.toHaveBeenCalled();
  });
});

describe("On-device (WebView Translator API)", () => {
  type FakeTranslator = { translate: ReturnType<typeof vi.fn>; destroy: ReturnType<typeof vi.fn> };

  function installTranslator(create: (opts: { sourceLanguage: string; targetLanguage: string }) => Promise<FakeTranslator>) {
    // The real global is an interface object: typeof "function", not "object".
    const Translator = Object.assign(function Translator() {}, { create: vi.fn(create) });
    vi.stubGlobal("Translator", Translator);
    return Translator;
  }

  const fake = (prefix = "T:"): FakeTranslator => ({
    translate: vi.fn(async (text: string) => prefix + text),
    destroy: vi.fn(),
  });

  it("starts when Translator is a function (typeof fix) and translates", async () => {
    const t = fake();
    const Translator = installTranslator(async () => t);
    const od = new Translation_OnDeviceService();
    await od.start(state((s) => { s.onDevice.languageFrom = "EN"; s.onDevice.language = "es"; }), live());
    expect(Translator.create).toHaveBeenCalledWith(expect.objectContaining({ sourceLanguage: "en", targetLanguage: "es" }));
    await expect(od.translate({ text: "hi", kind: "final" }, live())).resolves.toBe("T:hi");
  });

  it("reports a missing API as a start error", async () => {
    await expect(new Translation_OnDeviceService().start(state(), live())).rejects.toThrow("transl.ondevice_error_unavailable");
  });

  it("destroys a translator whose model finished loading after stop", async () => {
    const t = fake();
    let finish!: (v: FakeTranslator) => void;
    installTranslator(() => new Promise((r) => (finish = r)));
    const od = new Translation_OnDeviceService();
    const c = new AbortController();
    const started = od.start(state(), c.signal);
    c.abort(new Error("stopped"));
    od.stop();
    await expect(started).rejects.toThrow("stopped");
    finish(t);
    await new Promise((r) => setTimeout(r, 0));
    expect(t.destroy).toHaveBeenCalled();
  });

  it("shares one translator between concurrent calls and destroys everything on stop", async () => {
    const t = fake();
    const Translator = installTranslator(async () => t);
    const od = new Translation_OnDeviceService();
    await od.start(state((s) => (s.onDevice.autoDetect = true)), live());
    await Promise.all([
      od.translate({ text: "a", kind: "interim" }, live()),
      od.translate({ text: "ab", kind: "final" }, live()),
    ]);
    expect(Translator.create).toHaveBeenCalledTimes(1);
    od.stop();
    await new Promise((r) => setTimeout(r, 0));
    expect(t.destroy).toHaveBeenCalledTimes(1);
  });

  it("passes a line through when auto-detect finds the target language (no same-language translator)", async () => {
    const Translator = installTranslator(async (opts) => {
      if (opts.sourceLanguage === opts.targetLanguage) throw new Error("unsupported");
      return fake();
    });
    const detector = { detect: vi.fn(async (text: string) => [{ detectedLanguage: text.startsWith("Hola") ? "es" : "en" }]) };
    vi.stubGlobal("LanguageDetector", Object.assign(function LanguageDetector() {}, { create: vi.fn(async () => detector) }));
    const od = new Translation_OnDeviceService();
    await od.start(state((s) => { s.onDevice.autoDetect = true; s.onDevice.language = "es"; }), live());

    await expect(od.translate({ text: "Hola amigos", kind: "final" }, live())).resolves.toBe("Hola amigos");
    await expect(od.translate({ text: "Hello friends", kind: "final" }, live())).resolves.toBe("T:Hello friends");
    expect(Translator.create).toHaveBeenCalledTimes(1);
    expect(Translator.create).toHaveBeenCalledWith(expect.objectContaining({ sourceLanguage: "en", targetLanguage: "es" }));
  });

  it("starts without a translator when source and target are the same", async () => {
    const Translator = installTranslator(async () => {
      throw new Error("unsupported");
    });
    const od = new Translation_OnDeviceService();
    await od.start(state((s) => { s.onDevice.languageFrom = "ES"; s.onDevice.language = "es"; }), live());
    await expect(od.translate({ text: "hola", kind: "final" }, live())).resolves.toBe("hola");
    expect(Translator.create).not.toHaveBeenCalled();
  });

  it("wraps an unsupported pair in a non-retryable error and retries creation next time", async () => {
    let calls = 0;
    const Translator = installTranslator(async () => {
      calls++;
      if (calls === 1) throw new Error("unsupported");
      return fake();
    });
    const od = new Translation_OnDeviceService();
    await od.start(state((s) => (s.onDevice.autoDetect = true)), live());
    const err = await od.translate({ text: "x", kind: "final" }, live()).catch((e) => e);
    expect(err).toBeInstanceOf(TranslationError);
    expect(err.retryable).toBe(false);
    await expect(od.translate({ text: "x", kind: "final" }, live())).resolves.toBe("T:x");
    expect(Translator.create).toHaveBeenCalledTimes(2);
  });
});

describe("pipeline-set language pairs (router chains, reversed pipelines)", () => {
  it("Azure maps canonical codes, auto-detects an unknown source and refuses unknown targets", async () => {
    expect(azurePair({ languageFrom: "en", language: "es" }, {})).toEqual({ languageFrom: "en", language: "es" });
    expect(azurePair({ languageFrom: "en", language: "es" }, { from: "zh-Hant", to: "tl" })).toEqual({ languageFrom: "zh-Hant", language: "fil" });
    expect(azurePair({ languageFrom: "en", language: "es" }, { from: "", to: "ja" })).toEqual({ languageFrom: "", language: "ja" });
    expect(() => azurePair({ languageFrom: "en", language: "es" }, { from: "en", to: "ace" })).toThrow(TranslationError);

    const fetchMock = vi.fn(async () => new Response(JSON.stringify([{ translations: [{ text: "你好" }] }])));
    vi.stubGlobal("fetch", fetchMock);
    const az = new Translation_AzureService();
    await az.start(state((s) => (s.azure.key = "k")), live());
    await expect(az.translate({ text: "hello", kind: "final", from: "en", to: "zh" }, live())).resolves.toBe("你好");
    const url = new URL((fetchMock.mock.calls[0] as unknown as [string])[0]);
    expect(url.searchParams.get("from")).toBe("en");
    expect(url.searchParams.get("to")).toBe("zh-Hans");
    await expect(az.translate({ text: "hola", kind: "final", from: "es", to: "es" }, live())).resolves.toBe("hola");
    expect(fetchMock).toHaveBeenCalledTimes(1);
    expect(az.supports("en", "ja")).toBe(true);
    expect(az.supports("en", "ace")).toBe(false);
  });

  it("LibreTranslate maps canonical codes to the server's and lets the server detect an unknown source", async () => {
    expect(libreLanguageCode("zh-TW", ["en", "zt", "zh"])).toBe("zt");
    expect(libreLanguageCode("zh-Hant", ["en", "zh-Hans", "zh-Hant"])).toBe("zh-Hant");
    expect(libreLanguageCode("zh", ["en", "zh-Hans"])).toBe("zh-Hans");
    expect(libreLanguageCode("pt-BR", ["pt"])).toBe("pt");
    expect(libreLanguageCode("ko", ["en", "ja"])).toBeNull();
    expect(libreLanguageCode("ja", null)).toBe("ja");

    invokeMock.mockImplementation(async (cmd: string) => {
      if (cmd === "plugin:translate|check_availability") return true;
      if (cmd === "plugin:translate|get_languages") return [{ code: "en", name: "English" }, { code: "zt", name: "Chinese (traditional)" }, { code: "ja", name: "Japanese" }];
      return "翻譯";
    });
    const lt = new Translation_LibreTranslateService();
    await lt.start(state(), live());
    await expect(lt.translate({ text: "translation", kind: "final", from: "", to: "zh-Hant" }, live())).resolves.toBe("翻譯");
    expect(invokeMock).toHaveBeenLastCalledWith("plugin:translate|translate", expect.objectContaining({ source: "auto", target: "zt" }));
    expect(lt.supports("en", "ja")).toBe(true);
    expect(lt.supports("en", "ko")).toBe(false);
    await expect(lt.translate({ text: "x", kind: "final", from: "en", to: "ko" }, live())).rejects.toThrow("transl.error_unsupported_pair");
  });

  it("On-device uses the request pair and detects an unknown source", async () => {
    const t = { translate: vi.fn(async (text: string) => `T:${text}`), destroy: vi.fn() };
    const Translator = Object.assign(function Translator() {}, { create: vi.fn(async () => t) });
    vi.stubGlobal("Translator", Translator);
    const detector = { detect: vi.fn(async () => [{ detectedLanguage: "fr" }]) };
    vi.stubGlobal("LanguageDetector", Object.assign(function LanguageDetector() {}, { create: vi.fn(async () => detector) }));
    // The module caches its detector: use a fresh instance so earlier tests' detectors do not leak in.
    vi.resetModules();
    const { Translation_OnDeviceService: FreshOnDevice } = await import("../services/onDevice");
    const od = new FreshOnDevice();
    await od.start(state((s) => { s.onDevice.languageFrom = "en"; s.onDevice.language = "es"; }), live());
    await expect(od.translate({ text: "bonjour", kind: "final", from: "", to: "de" }, live())).resolves.toBe("T:bonjour");
    expect(Translator.create).toHaveBeenLastCalledWith(expect.objectContaining({ sourceLanguage: "fr", targetLanguage: "de" }));
    await od.translate({ text: "hi", kind: "final", from: "ja", to: "zh-Hant" }, live());
    expect(Translator.create).toHaveBeenLastCalledWith(expect.objectContaining({ sourceLanguage: "ja", targetLanguage: "zh-hant" }));
  });
});
