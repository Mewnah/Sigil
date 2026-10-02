import i18n from "i18next";
import { afterEach, beforeAll, beforeEach, describe, expect, it, vi } from "vitest";

vi.mock("@tauri-apps/api/core", () => ({ invoke: vi.fn() }));

import { invoke } from "@tauri-apps/api/core";
import { NLLB_MODELS, Service_Translation_Schema, Translation_State } from "../schema";
import { isMtLocalError, nllbError, Translation_NllbService } from "../services/nllb";
import { TranslationEngineHost, TranslationError, TranslationStatusUpdate } from "../types";

const invokeMock = vi.mocked(invoke);

function state(patch: (s: Translation_State) => void = () => {}): Translation_State {
  const s = Service_Translation_Schema.parse({});
  s.targetLanguage = "ja";
  patch(s);
  return s;
}

const live = () => new AbortController().signal;

function host(): TranslationEngineHost & { updates: TranslationStatusUpdate[] } {
  const updates: TranslationStatusUpdate[] = [];
  return { engines: {} as TranslationEngineHost["engines"], status: (u) => updates.push(u), updates };
}

/** Routes `invoke` by command; unknown commands resolve undefined. */
function route(handlers: Record<string, (args: Record<string, unknown>) => unknown>) {
  invokeMock.mockImplementation(async (cmd: string, args?: unknown) => {
    const h = handlers[cmd];
    if (!h) return undefined;
    return h((args ?? {}) as Record<string, unknown>);
  });
}

const calls = (cmd: string) => invokeMock.mock.calls.filter(([c]) => c === cmd).map(([, a]) => a as Record<string, unknown>);

beforeAll(async () => {
  await i18n.init({ lng: "en", resources: { en: { translation: {} } } });
});

// The engine counts its users process-wide (the last one to stop unloads the model), so every instance a test starts is
// stopped after it: a leftover user would keep the model "in use" for the next test.
// (`restoreMocks` restores spies before every test, so the spy is installed per test.)
const started: Translation_NllbService[] = [];
const realStart = Translation_NllbService.prototype.start;
beforeEach(() => {
  vi.spyOn(Translation_NllbService.prototype, "start").mockImplementation(function (this: Translation_NllbService, ...args) {
    started.push(this);
    return realStart.apply(this, args);
  });
});
afterEach(() => {
  for (const engine of started.splice(0)) engine.stop();
});

beforeEach(() => {
  invokeMock.mockReset();
});

describe("NLLB errors", () => {
  it("recognizes mt-local rejections", () => {
    expect(isMtLocalError({ kind: "runtime", message: "x", missing: [] })).toBe(true);
    expect(isMtLocalError(new Error("x"))).toBe(false);
    expect(isMtLocalError("x")).toBe(false);
  });

  it("maps kinds to retryability and missing models", () => {
    const missing = nllbError({ kind: "not_installed", message: "not installed", missing: [NLLB_MODELS[1]] });
    expect(missing.retryable).toBe(false);
    expect(missing.missing).toEqual([NLLB_MODELS[1]]);
    expect(nllbError({ kind: "not_installed", message: "", missing: [] }).missing).toEqual([NLLB_MODELS[0]]);
    expect(nllbError({ kind: "unsupported_pair", message: "no", missing: [] }).retryable).toBe(false);
    expect(nllbError({ kind: "invalid_input", message: "no", missing: [] }).retryable).toBe(false);
    expect(nllbError({ kind: "busy", message: "later", missing: [] }).retryable).toBe(true);
    expect(nllbError({ kind: "runtime", message: "boom", missing: [] }).retryable).toBe(true);
    expect(nllbError(new Error("ipc")).retryable).toBe(true);
    expect(nllbError({ kind: "runtime", message: "boom", missing: [] })).toBeInstanceOf(TranslationError);
  });
});

describe("Translation_NllbService", () => {
  it("loads the configured model with its threads on start", async () => {
    route({ "plugin:mt-local|nllb_load": () => ({ engine: "nllb", state: "ready" }) });
    const h = host();
    const engine = new Translation_NllbService();
    engine.attach(h);
    await engine.start(
      state((s) => {
        s.nllb.model = NLLB_MODELS[1];
        s.nllb.threads = 3;
      }),
      live(),
    );
    expect(calls("plugin:mt-local|nllb_load")).toEqual([{ model: NLLB_MODELS[1], threads: 3 }]);
    expect(h.updates.at(-1)).toEqual({ text: "" });
  });

  it("falls back to the default model for an unknown persisted id", async () => {
    route({ "plugin:mt-local|nllb_load": () => ({}) });
    await new Translation_NllbService().start(state((s) => (s.nllb.model = "mt/something-else")), live());
    expect(calls("plugin:mt-local|nllb_load")[0].model).toBe(NLLB_MODELS[0]);
  });

  it("rejects a missing model with the ids to install", async () => {
    route({
      "plugin:mt-local|nllb_load": () => {
        throw { kind: "not_installed", message: "not installed", missing: [NLLB_MODELS[0]] };
      },
    });
    const h = host();
    const engine = new Translation_NllbService();
    engine.attach(h);
    const e = await engine.start(state(), live()).catch((x: unknown) => x);
    expect(e).toBeInstanceOf(TranslationError);
    expect((e as TranslationError).missing).toEqual([NLLB_MODELS[0]]);
    expect(h.updates.some((u) => u.missing?.includes(NLLB_MODELS[0]))).toBe(true);
  });

  it("needs a target language", async () => {
    await expect(new Translation_NllbService().start(state((s) => (s.targetLanguage = "")), live())).rejects.toBeInstanceOf(
      TranslationError,
    );
    await expect(
      new Translation_NllbService().start(state((s) => (s.sourceLanguage = "not-a-language")), live()),
    ).rejects.toBeInstanceOf(TranslationError);
    expect(invokeMock).not.toHaveBeenCalled();
  });

  it("translates with canonical codes, beam for finals only, and the request id", async () => {
    route({
      "plugin:mt-local|nllb_load": () => ({}),
      "plugin:mt-local|nllb_translate": (a) => ({ text: `T:${(a.request as { text: string }).text}`, engine: "nllb" }),
    });
    const engine = new Translation_NllbService();
    await engine.start(state((s) => (s.nllb.beam = 3)), live());
    await expect(engine.translate({ text: "Hello", kind: "final", from: "en", requestId: "mic:1:2#0" }, live())).resolves.toBe(
      "T:Hello",
    );
    await expect(engine.translate({ text: "Hel", kind: "interim", from: "en" }, live())).resolves.toBe("T:Hel");
    const [final, interim] = calls("plugin:mt-local|nllb_translate").map((a) => a.request as Record<string, unknown>);
    expect(final).toMatchObject({ text: "Hello", source: "en", target: "ja", kind: "final", requestId: "mic:1:2#0", beam: 3 });
    expect(final.model).toBe(NLLB_MODELS[0]);
    expect(interim).toMatchObject({ kind: "interim", beam: 1 });
    expect(interim.requestId).toMatch(/^nllb-\d+$/);
  });

  it("guesses an unknown source from the script and passes it on", async () => {
    route({ "plugin:mt-local|nllb_load": () => ({}), "plugin:mt-local|nllb_translate": () => ({ text: "x" }) });
    const engine = new Translation_NllbService();
    await engine.start(state((s) => (s.targetLanguage = "en")), live());
    await engine.translate({ text: "Привіт, як справи?", kind: "final" }, live());
    await engine.translate({ text: "Hello there", kind: "final", from: "" }, live());
    const [uk, latin] = calls("plugin:mt-local|nllb_translate").map((a) => a.request as Record<string, unknown>);
    expect(uk.source).toBe("uk");
    expect(latin.source, "Latin text is left to Rust's guess").toBe("");
  });

  it("returns a line already in the target language unchanged", async () => {
    route({ "plugin:mt-local|nllb_load": () => ({}) });
    const engine = new Translation_NllbService();
    await engine.start(state(), live());
    await expect(engine.translate({ text: "こんにちは", kind: "final", from: "ja", to: "ja" }, live())).resolves.toBe("こんにちは");
    expect(calls("plugin:mt-local|nllb_translate")).toHaveLength(0);
  });

  it("cancels an aborted request in Rust", async () => {
    let finish: (v: unknown) => void = () => {};
    route({
      "plugin:mt-local|nllb_load": () => ({}),
      "plugin:mt-local|nllb_translate": () => new Promise((r) => (finish = r)),
      "plugin:mt-local|nllb_cancel": () => undefined,
    });
    const engine = new Translation_NllbService();
    await engine.start(state(), live());
    const ctl = new AbortController();
    const pending = engine.translate({ text: "Hello", kind: "interim", from: "en", requestId: "r1" }, ctl.signal);
    ctl.abort(new DOMException("superseded", "AbortError"));
    await expect(pending).rejects.toMatchObject({ name: "AbortError" });
    expect(calls("plugin:mt-local|nllb_cancel")).toEqual([{ requestId: "r1" }]);
    finish({ text: "late" });
  });

  it("maps Rust failures and unloads on stop", async () => {
    route({
      "plugin:mt-local|nllb_load": () => ({}),
      "plugin:mt-local|nllb_translate": () => {
        throw { kind: "unsupported_pair", message: "cannot translate ? to ja", missing: [] };
      },
      "plugin:mt-local|nllb_unload": () => undefined,
    });
    const engine = new Translation_NllbService();
    await engine.start(state(), live());
    const e = await engine.translate({ text: "x", kind: "final", from: "en" }, live()).catch((x: unknown) => x);
    expect((e as TranslationError).retryable).toBe(false);
    engine.stop();
    expect(calls("plugin:mt-local|nllb_unload")).toHaveLength(1);
    await expect(engine.translate({ text: "x", kind: "final" }, live())).rejects.toBeInstanceOf(TranslationError);
  });

  it("gives the memory back when stopped while loading", async () => {
    let loaded: (v: unknown) => void = () => {};
    route({
      "plugin:mt-local|nllb_load": () => new Promise((r) => (loaded = r)),
      "plugin:mt-local|nllb_unload": () => undefined,
    });
    const engine = new Translation_NllbService();
    const ctl = new AbortController();
    const starting = engine.start(state(), ctl.signal);
    ctl.abort();
    engine.stop();
    loaded({});
    await expect(starting).rejects.toBeDefined();
    expect(calls("plugin:mt-local|nllb_unload").length).toBeGreaterThanOrEqual(1);
  });

  it("only the latest instance unloads the shared model", async () => {
    route({ "plugin:mt-local|nllb_load": () => ({}), "plugin:mt-local|nllb_unload": () => undefined });
    const old = new Translation_NllbService();
    await old.start(state(), live());
    const next = new Translation_NllbService();
    await next.start(state(), live());
    old.stop();
    expect(calls("plugin:mt-local|nllb_unload")).toHaveLength(0);
    next.stop();
    expect(calls("plugin:mt-local|nllb_unload")).toHaveLength(1);
  });

  it("keeps the model while another pipeline still uses it (main + Other players, P6-A)", async () => {
    route({ "plugin:mt-local|nllb_load": () => ({}), "plugin:mt-local|nllb_unload": () => undefined });
    const main = new Translation_NllbService();
    await main.start(state(), live());
    const others = new Translation_NllbService();
    await others.start(state(), live());
    others.stop();
    expect(calls("plugin:mt-local|nllb_unload")).toHaveLength(0);
    others.stop();
    expect(calls("plugin:mt-local|nllb_unload")).toHaveLength(0);
    main.stop();
    expect(calls("plugin:mt-local|nllb_unload")).toHaveLength(1);
  });

  it("supports every language with a FLORES code", () => {
    const engine = new Translation_NllbService();
    expect(engine.supports("en", "ja")).toBe(true);
    expect(engine.supports("", "sw")).toBe(true);
    expect(engine.supports("en", "xx")).toBe(false);
    expect(engine.supports("xx", "en")).toBe(false);
  });
});
