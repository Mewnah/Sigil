import i18n from "i18next";
import { afterEach, beforeAll, beforeEach, describe, expect, it, vi } from "vitest";

vi.mock("@tauri-apps/api/core", () => ({ invoke: vi.fn(), isTauri: () => false }));
vi.mock("@tauri-apps/api/event", () => ({ listen: vi.fn() }));
// The pipeline-level tests start Bergamot through the real translation pipeline.
vi.mock("react-toastify", () => ({
  toast: Object.assign(vi.fn(), { error: vi.fn(), info: vi.fn(), warning: vi.fn(), success: vi.fn() }),
}));
vi.mock("../../systemLog", () => ({ pushSystemLog: vi.fn() }));
vi.mock("@/utils/devLog", () => ({ devLog: () => {} }));

import { ServiceNetworkState, TextEventType, type RustMtPair } from "@/types";
import { invoke } from "@tauri-apps/api/core";
import { listen } from "@tauri-apps/api/event";
import { proxy } from "valtio";
import { createTranslationPipeline, TranslationEngineMap, TranslationOutputEvent } from "../pipeline";
import { Service_Translation_Schema, Translation_Backends, Translation_State } from "../schema";
import {
  BERGAMOT_REFRESH_DELAY_MS,
  BERGAMOT_RUNTIME_ID,
  bergamotError,
  BergamotPairs,
  Translation_BergamotService,
} from "../services/bergamot";
import { TranslationEngineHost, TranslationError, TranslationStatusUpdate } from "../types";

const invokeMock = vi.mocked(invoke);
const listenMock = vi.mocked(listen);

const pair = (source: string, target: string, installed: boolean, ids?: string[]): RustMtPair => ({
  source,
  target,
  ids: ids ?? [`mt-bergamot/${source.toLowerCase()}_${target.toLowerCase()}`],
  installed,
  pivot: (ids?.length ?? 1) === 2,
});

/** A small catalog: en↔ja installed, en→zh not, zh-Hant→en installed, en→pt installed, pivots through English. */
const PAIRS: RustMtPair[] = [
  pair("en", "ja", true),
  pair("ja", "en", true),
  pair("en", "zh", false),
  pair("zh-Hant", "en", true),
  pair("en", "pt", true),
  pair("ja", "zh", false, ["mt-bergamot/ja_en", "mt-bergamot/en_zh"]),
  pair("zh-Hant", "ja", true, ["mt-bergamot/zh-hant_en", "mt-bergamot/en_ja"]),
  pair("ja", "pt", true, ["mt-bergamot/ja_en", "mt-bergamot/en_pt"]),
];

function state(patch: (s: Translation_State) => void = () => {}): Translation_State {
  const s = Service_Translation_Schema.parse({});
  s.sourceLanguage = "en";
  s.targetLanguage = "ja";
  patch(s);
  return s;
}

const live = () => new AbortController().signal;

function host(): TranslationEngineHost & { updates: TranslationStatusUpdate[] } {
  const updates: TranslationStatusUpdate[] = [];
  return { engines: {} as TranslationEngineHost["engines"], status: (u) => updates.push(u), updates };
}

type Handler = (args: Record<string, unknown>) => unknown;

/** Routes `invoke` by command; unknown commands resolve undefined. */
function route(handlers: Record<string, Handler>) {
  invokeMock.mockImplementation(async (cmd: string, args?: unknown) => {
    const h = handlers[cmd];
    if (!h) return undefined;
    return h((args ?? {}) as Record<string, unknown>);
  });
}

const calls = (cmd: string) => invokeMock.mock.calls.filter(([c]) => c === cmd).map(([, a]) => a as Record<string, unknown>);

const translateEcho: Handler = (a) => {
  const r = a.request as { text: string; source: string; target: string };
  return { text: `[${r.source}>${r.target}] ${r.text}`, engine: "bergamot", model: `${r.source}-${r.target}`, elapsedMs: 3, pivot: false };
};

let started: Translation_BergamotService[] = [];
let modelEvents: ((e: { payload: { id: string } }) => void) | null = null;

async function startedEngine(s = state(), h = host(), pairs = PAIRS): Promise<Translation_BergamotService> {
  route({
    "plugin:mt-local|bergamot_pairs": () => pairs,
    "plugin:mt-local|bergamot_start": () => ({ engine: "bergamot", state: "ready" }),
    "plugin:mt-local|bergamot_translate": translateEcho,
  });
  const e = new Translation_BergamotService();
  e.attach(h);
  started.push(e);
  await e.start(s, live());
  return e;
}

beforeAll(async () => {
  await i18n.init({ lng: "en", resources: { en: { translation: {} } } });
});

beforeEach(() => {
  invokeMock.mockReset();
  listenMock.mockReset();
  modelEvents = null;
  listenMock.mockImplementation(async (_event, handler) => {
    modelEvents = handler as unknown as typeof modelEvents;
    return () => {
      modelEvents = null;
    };
  });
});

afterEach(() => {
  for (const e of started) e.stop();
  started = [];
  vi.useRealTimers();
});

describe("Bergamot errors", () => {
  it("names the missing downloads: direction models or the helper pack", () => {
    const models = bergamotError({ kind: "not_installed", message: "not installed", missing: ["mt-bergamot/en_ja"] }, "en", "ja");
    expect(models.retryable).toBe(false);
    expect(models.missing).toEqual(["mt-bergamot/en_ja"]);
    expect(models.message).toBe("transl.bergamot_error_not_installed");
    const runtime = bergamotError({ kind: "not_installed", message: "x", missing: [BERGAMOT_RUNTIME_ID] });
    expect(runtime.missing).toEqual([BERGAMOT_RUNTIME_ID]);
    expect(runtime.message).toBe("transl.bergamot_error_runtime_missing");
  });

  it("maps kinds to retryability", () => {
    expect(bergamotError({ kind: "unsupported_pair", message: "no", missing: [] }).retryable).toBe(false);
    expect(bergamotError({ kind: "invalid_input", message: "no", missing: [] }).retryable).toBe(false);
    expect(bergamotError({ kind: "cancelled", message: "c", missing: [] }).retryable).toBe(false);
    expect(bergamotError({ kind: "busy", message: "later", missing: [] }).retryable).toBe(true);
    expect(bergamotError({ kind: "runtime", message: "crashed", missing: [] }).retryable).toBe(true);
    expect(bergamotError(new Error("ipc")).retryable).toBe(true);
    expect(bergamotError("plain").retryable).toBe(true);
  });
});

describe("BergamotPairs", () => {
  const p = new BergamotPairs(PAIRS);

  it("finds direct and pivot directions with canonical codes", () => {
    expect(p.find("en", "ja")?.ids).toEqual(["mt-bergamot/en_ja"]);
    expect(p.find("ja", "zh-Hans")?.pivot).toBe(true);
    expect(p.find("zh-TW", "ja")?.ids).toEqual(["mt-bergamot/zh-hant_en", "mt-bergamot/en_ja"]);
    expect(p.find("en-US", "ja")?.ids).toEqual(["mt-bergamot/en_ja"]);
    expect(p.find("en", "pt-PT")?.ids, "Portugal falls back to the Portuguese model").toEqual(["mt-bergamot/en_pt"]);
    expect(p.find("en", "fr")).toBeUndefined();
    expect(p.find("", "ja")).toBeUndefined();
  });

  it("never serves Traditional Chinese with the Simplified models", () => {
    const only = new BergamotPairs([pair("en", "zh", true)]);
    expect(only.knowsTarget("zh-Hans")).toBe(true);
    expect(only.knowsTarget("zh-Hant")).toBe(false);
    expect(only.find("en", "zh-TW")).toBeUndefined();
  });

  it("reports install state and the ids a pair needs", () => {
    expect(p.installed("en", "ja")).toBe(true);
    expect(p.installed("ja", "zh")).toBe(false);
    expect(p.installed("zh-Hant", "ja")).toBe(true);
    expect(p.neededIds("ja", "zh")).toEqual(["mt-bergamot/ja_en", "mt-bergamot/en_zh"]);
    expect(p.neededIds("", "ja"), "unknown source: the leg into the target").toEqual(["mt-bergamot/en_ja"]);
    expect(p.neededIds("", "en")).toEqual([]);
    expect(p.knowsSource("zh-Hant")).toBe(true);
    expect(p.knowsSource("fr")).toBe(false);
    expect(p.size).toBe(PAIRS.length);
  });

  it("knows which targets an installed direction reaches, from any source", () => {
    expect(p.reachesInstalled("ja"), "en→ja, and zh-Hant→ja through English").toBe(true);
    expect(p.reachesInstalled("en"), "ja→en, zh-Hant→en").toBe(true);
    expect(p.reachesInstalled("pt-BR"), "en→pt, base-language fallback").toBe(true);
    expect(p.reachesInstalled("zh-Hans"), "en→zh and ja→zh are not installed").toBe(false);
    expect(p.reachesInstalled("fr")).toBe(false);
    expect(new BergamotPairs(PAIRS.map((x) => ({ ...x, installed: false }))).reachesInstalled("en")).toBe(false);
  });
});

describe("Bergamot engine start", () => {
  it("rejects a target Bergamot cannot translate before starting the helper", async () => {
    route({ "plugin:mt-local|bergamot_pairs": () => PAIRS });
    const e = new Translation_BergamotService();
    await expect(e.start(state((s) => (s.targetLanguage = "fr")), live())).rejects.toThrow("transl.bergamot_error_language");
    expect(calls("plugin:mt-local|bergamot_start")).toHaveLength(0);
  });

  it("names missing models without downloading or starting anything", async () => {
    route({ "plugin:mt-local|bergamot_pairs": () => PAIRS });
    const h = host();
    const e = new Translation_BergamotService();
    e.attach(h);
    const err = await e.start(state((s) => { s.sourceLanguage = "ja"; s.targetLanguage = "zh"; }), live()).catch((x) => x);
    expect(err).toBeInstanceOf(TranslationError);
    expect((err as TranslationError).missing).toEqual(["mt-bergamot/en_zh"]);
    expect((err as TranslationError).retryable).toBe(false);
    expect(h.updates.at(-1)).toMatchObject({ level: "warning", missing: ["mt-bergamot/en_zh"] });
    expect(calls("plugin:mt-local|bergamot_start")).toHaveLength(0);
  });

  it("reports a missing helper pack and releases the helper", async () => {
    route({
      "plugin:mt-local|bergamot_pairs": () => PAIRS,
      "plugin:mt-local|bergamot_start": () => {
        throw { kind: "not_installed", message: "not installed: runtime/bergamot", missing: [BERGAMOT_RUNTIME_ID] };
      },
    });
    const h = host();
    const e = new Translation_BergamotService();
    e.attach(h);
    const err = (await e.start(state(), live()).catch((x) => x)) as TranslationError;
    expect(err.missing).toEqual([BERGAMOT_RUNTIME_ID]);
    expect(err.message).toBe("transl.bergamot_error_runtime_missing");
    expect(calls("plugin:mt-local|bergamot_stop")).toHaveLength(1);
    // The pack is built locally: a router member without it falls through quietly instead of warning on every retry.
    expect(h.updates.filter((u) => u.level === "warning")).toEqual([]);
  });

  it("with an automatic source into English, needs at least one direction into English", async () => {
    route({ "plugin:mt-local|bergamot_pairs": () => PAIRS.map((x) => (x.target === "en" ? { ...x, installed: false } : x)) });
    const e = new Translation_BergamotService();
    const err = (await e
      .start(
        state((s) => {
          s.sourceLanguage = "";
          s.targetLanguage = "en";
        }),
        live(),
      )
      .catch((x) => x)) as TranslationError;
    expect(err).toBeInstanceOf(TranslationError);
    expect(err.message).toBe("transl.bergamot_unknown_source");
    expect(err.retryable).toBe(false);
    expect(calls("plugin:mt-local|bergamot_start")).toHaveLength(0);
  });

  it("starts the helper with the slice's threads and preloads the pair", async () => {
    const h = host();
    await startedEngine(state((s) => (s.bergamot.threads = 2)), h);
    expect(calls("plugin:mt-local|bergamot_start")).toEqual([{ threads: 2 }]);
    const warm = calls("plugin:mt-local|bergamot_translate");
    expect(warm).toHaveLength(1);
    expect(warm[0].request).toMatchObject({ source: "en", target: "ja", kind: "final", requestId: "bergamot-warmup" });
    expect(h.updates.at(-1)).toEqual({ text: "" });
    expect(listenMock).toHaveBeenCalledWith("models:state", expect.any(Function));
  });

  it("preloads English into the target when the source is the speech language", async () => {
    await startedEngine(state((s) => { s.sourceLanguage = ""; s.targetLanguage = "ja"; }));
    expect(calls("plugin:mt-local|bergamot_translate")[0].request).toMatchObject({ source: "en", target: "ja" });
  });

  it("gives the helper back when aborted while starting", async () => {
    let release!: (v: unknown) => void;
    route({
      "plugin:mt-local|bergamot_pairs": () => PAIRS,
      "plugin:mt-local|bergamot_start": () => new Promise((r) => (release = r)),
    });
    const e = new Translation_BergamotService();
    const ac = new AbortController();
    const p = e.start(state(), ac.signal);
    await vi.waitFor(() => expect(calls("plugin:mt-local|bergamot_start")).toHaveLength(1));
    ac.abort(new DOMException("stop", "AbortError"));
    await expect(p).rejects.toThrow("stop");
    release({});
    expect(calls("plugin:mt-local|bergamot_stop")).toHaveLength(1);
  });
});

describe("Bergamot engine translate", () => {
  it("sends canonical codes and the request id", async () => {
    const e = await startedEngine();
    await expect(e.translate({ text: "hello", kind: "interim", from: "en-US", to: "ja", requestId: "mic:1:2#3" }, live())).resolves.toBe(
      "[en>ja] hello",
    );
    const last = calls("plugin:mt-local|bergamot_translate").at(-1)!.request;
    expect(last).toMatchObject({ text: "hello", source: "en", target: "ja", kind: "interim", requestId: "mic:1:2#3", threads: 1 });
  });

  it("passes the same language through and guesses an unknown source from the script", async () => {
    const e = await startedEngine(state((s) => { s.sourceLanguage = ""; s.targetLanguage = "en"; }));
    const before = calls("plugin:mt-local|bergamot_translate").length;
    await expect(e.translate({ text: "hi there", kind: "final", from: "en", to: "en" }, live())).resolves.toBe("hi there");
    expect(calls("plugin:mt-local|bergamot_translate")).toHaveLength(before);
    await expect(e.translate({ text: "こんにちは", kind: "final" }, live())).resolves.toBe("[ja>en] こんにちは");
    const err = await e.translate({ text: "bonjour", kind: "final", from: "", to: "ja" }, live()).catch((x) => x);
    expect(err).toBeInstanceOf(TranslationError);
    expect((err as TranslationError).message).toBe("transl.bergamot_error_source_unknown");
    expect((err as TranslationError).retryable).toBe(false);
  });

  it("cancels an aborted request in Rust", async () => {
    const e = await startedEngine();
    route({ "plugin:mt-local|bergamot_translate": () => new Promise(() => {}) });
    const ac = new AbortController();
    const p = e.translate({ text: "slow", kind: "interim", requestId: "r-1" }, ac.signal);
    ac.abort(new DOMException("superseded", "AbortError"));
    await expect(p).rejects.toThrow("superseded");
    expect(calls("plugin:mt-local|bergamot_cancel")).toEqual([{ requestId: "r-1" }]);
  });

  it("reports models removed while running and refreshes the directions", async () => {
    vi.useFakeTimers();
    const h = host();
    const e = await startedEngine(state(), h);
    route({
      "plugin:mt-local|bergamot_translate": () => {
        throw { kind: "not_installed", message: "not installed", missing: ["mt-bergamot/en_ja"] };
      },
      "plugin:mt-local|bergamot_pairs": () => PAIRS.map((p) => ({ ...p, installed: false })),
    });
    const err = (await e.translate({ text: "x", kind: "final" }, live()).catch((x) => x)) as TranslationError;
    expect(err.missing).toEqual(["mt-bergamot/en_ja"]);
    expect(h.updates.at(-1)).toMatchObject({ level: "warning", missing: ["mt-bergamot/en_ja"] });
    await vi.advanceTimersByTimeAsync(BERGAMOT_REFRESH_DELAY_MS + 1);
    expect(e.supports("en", "ja")).toBe(false);
  });
});

describe("Bergamot engine supports and stop", () => {
  it("answers from the installed directions and follows model-store changes", async () => {
    vi.useFakeTimers();
    const e = await startedEngine();
    expect(e.supports("en", "ja")).toBe(true);
    expect(e.supports("ja", "zh")).toBe(false);
    // Unknown source: translate() guesses it, so an installed direction into the target is enough.
    expect(e.supports("", "ja")).toBe(true);
    expect(e.supports("", "en")).toBe(true);
    expect(e.supports("", "zh")).toBe(false);
    expect(e.supports("", "fr")).toBe(false);
    expect(e.supports("ja", "ja")).toBe(true);
    // en_zh gets installed: the directions are listed again (debounced) and the pivot becomes usable.
    route({ "plugin:mt-local|bergamot_pairs": () => PAIRS.map((p) => (p.target === "zh" ? { ...p, installed: true } : p)) });
    const listed = calls("plugin:mt-local|bergamot_pairs").length;
    modelEvents?.({ payload: { id: "mt-bergamot/en_zh" } });
    modelEvents?.({ payload: { id: "runtime/bergamot" } });
    modelEvents?.({ payload: { id: "stt/unrelated" } });
    await vi.advanceTimersByTimeAsync(BERGAMOT_REFRESH_DELAY_MS + 1);
    expect(calls("plugin:mt-local|bergamot_pairs")).toHaveLength(listed + 1);
    expect(e.supports("ja", "zh")).toBe(true);
    expect(e.supports("", "zh")).toBe(true);
  });

  it("stops the helper when the last engine instance stops", async () => {
    const a = await startedEngine();
    const b = await startedEngine();
    started = [];
    a.stop();
    expect(calls("plugin:mt-local|bergamot_stop")).toHaveLength(0);
    a.stop();
    expect(calls("plugin:mt-local|bergamot_stop")).toHaveLength(0);
    b.stop();
    expect(calls("plugin:mt-local|bergamot_stop")).toHaveLength(1);
    expect(a.supports("en", "ja")).toBe(false);
    await expect(a.translate({ text: "x", kind: "final" }, live())).rejects.toThrow("transl.error_not_running");
    expect(modelEvents).toBeNull();
  });

  it("lists pairs for the inspector and tolerates failures", async () => {
    route({ "plugin:mt-local|bergamot_pairs": () => PAIRS });
    await expect(Translation_BergamotService.getPairs()).resolves.toHaveLength(PAIRS.length);
    route({
      "plugin:mt-local|bergamot_pairs": () => {
        throw new Error("no ipc");
      },
    });
    await expect(Translation_BergamotService.getPairs()).resolves.toEqual([]);
  });
});

describe("Bergamot as the pipeline backend", () => {
  const engines = Object.fromEntries(
    Object.values(Translation_Backends).map((b) => [b, Translation_BergamotService]),
  ) as unknown as TranslationEngineMap;

  function pipeline(source: string, target: string) {
    const settings = proxy(Service_Translation_Schema.parse({}));
    settings.backend = Translation_Backends.bergamot;
    settings.sourceLanguage = source;
    settings.targetLanguage = target;
    settings.partials.throttleMs = 0;
    const events: TranslationOutputEvent[] = [];
    const p = createTranslationPipeline({
      name: "tr",
      settings: () => settings,
      engines,
      publish: (ev) => events.push(ev),
      toasts: false,
    });
    return { p, events };
  }

  beforeEach(() => {
    route({
      "plugin:mt-local|bergamot_pairs": () => PAIRS,
      "plugin:mt-local|bergamot_start": () => ({ engine: "bergamot", state: "ready" }),
      "plugin:mt-local|bergamot_translate": translateEcho,
    });
  });

  it.each([
    ["ja", "en"],
    ["en", "ja"],
    ["", "en"],
    ["", "ja"],
  ])("connects when selected directly (source %j, target %j)", async (source, target) => {
    const { p } = pipeline(source, target);
    p.start();
    await vi.waitFor(() => expect(p.state.status).toBe(ServiceNetworkState.connected));
    expect(p.state.error).toBe("");
    p.dispose();
  });

  it("translates lines with the configured or the guessed source", async () => {
    for (const source of ["ja", ""]) {
      const { p, events } = pipeline(source, "en");
      p.start();
      await vi.waitFor(() => expect(p.state.status).toBe(ServiceNetworkState.connected));
      p.push("stt", { type: TextEventType.final, value: "こんにちは", emotes: {} });
      await vi.waitFor(() => expect(events.map((ev) => ev.value)).toEqual(["[ja>en] こんにちは"]));
      p.dispose();
    }
  });

  it("names the missing direction model when the pair is not installed", async () => {
    route({
      "plugin:mt-local|bergamot_pairs": () => PAIRS.map((x) => (x.source === "ja" && x.target === "en" ? { ...x, installed: false } : x)),
    });
    const { p } = pipeline("ja", "en");
    p.start();
    await vi.waitFor(() => expect(p.state.status).toBe(ServiceNetworkState.disconnected));
    expect(p.state.missing).toEqual(["mt-bergamot/ja_en"]);
    expect(calls("plugin:mt-local|bergamot_start")).toHaveLength(0);
    p.dispose();
  });
});
