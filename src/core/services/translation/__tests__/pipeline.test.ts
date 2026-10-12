import i18n from "i18next";
import { proxy } from "valtio";
import { afterEach, beforeAll, beforeEach, describe, expect, it, vi } from "vitest";
import { ServiceNetworkState, TextEvent, TextEventType } from "@/types";

vi.mock("@tauri-apps/api/core", () => ({ invoke: vi.fn(), isTauri: () => false }));
vi.mock("react-toastify", () => ({
  toast: Object.assign(vi.fn(), { error: vi.fn(), info: vi.fn(), warning: vi.fn(), success: vi.fn() }),
}));
vi.mock("../../systemLog", () => ({ pushSystemLog: vi.fn() }));
vi.mock("@/utils/devLog", () => ({ devLog: () => {} }));

import { createTranslationPipeline, TranslationEngineMap, TranslationOutputEvent, TranslationPipeline } from "../pipeline";
import { Translation_RouterService } from "../router";
import { Service_Translation_Schema, Translation_Backends, Translation_State } from "../schema";
import { CANCEL_MARKER } from "../sequencer";
import { ITranslationEngine, TranslationEngineHost, TranslationError, TranslationRequest } from "../types";
import { PROCESS_NONCE } from "@/utils/processNonce";

function rng(seed: number): () => number {
  let a = seed >>> 0;
  return () => {
    a = (a + 0x6d2b79f5) >>> 0;
    let t = a;
    t = Math.imul(t ^ (t >>> 15), t | 1);
    t ^= t + Math.imul(t ^ (t >>> 7), t | 61);
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}

type Call = { request: TranslationRequest; signal: AbortSignal; resolve: (v: string) => void; reject: (e: unknown) => void };

class FakeEngine implements ITranslationEngine {
  static instances: FakeEngine[] = [];
  static startImpl: (signal: AbortSignal, host: TranslationEngineHost | null) => Promise<void> = async () => {};
  static supportsImpl: ((from: string, to: string) => boolean) | null = null;
  calls: Call[] = [];
  host: TranslationEngineHost | null = null;
  stopped = 0;
  constructor() {
    FakeEngine.instances.push(this);
  }
  attach(host: TranslationEngineHost) {
    this.host = host;
  }
  start(_state: Translation_State, signal: AbortSignal) {
    return FakeEngine.startImpl(signal, this.host);
  }
  translate(request: TranslationRequest, signal: AbortSignal) {
    return new Promise<string>((resolve, reject) => this.calls.push({ request, signal, resolve, reject }));
  }
  supports(from: string, to: string) {
    return FakeEngine.supportsImpl ? FakeEngine.supportsImpl(from, to) : true;
  }
  stop() {
    this.stopped++;
  }
  call(text: string, kind?: "interim" | "final") {
    const c = [...this.calls].reverse().find((x) => x.request.text === text && (!kind || x.request.kind === kind));
    if (!c) throw new Error(`no call for "${text}"`);
    return c;
  }
}

const engines = Object.fromEntries(Object.values(Translation_Backends).map((b) => [b, FakeEngine])) as unknown as TranslationEngineMap;
const engine = () => FakeEngine.instances[FakeEngine.instances.length - 1];
const flush = async () => {
  for (let i = 0; i < 5; i++) await new Promise<void>((r) => setTimeout(r, 0));
};

interface Harness {
  pipeline: TranslationPipeline;
  settings: Translation_State;
  events: TranslationOutputEvent[];
  out: () => string[];
  push: (type: TextEventType, value: string, meta?: Partial<TextEvent>, stream?: string) => void;
}

interface MakeOptions {
  pair?: () => { from?: string; to?: string };
  engines?: Partial<TranslationEngineMap>;
  translatorOnCpu?: () => boolean;
  modelInstalled?: (id: string) => boolean;
}

function make(patch: (s: Translation_State) => void = () => {}, options: MakeOptions = {}): Harness {
  const settings = proxy(Service_Translation_Schema.parse({}));
  settings.backend = Translation_Backends.llm;
  settings.targetLanguage = "es";
  settings.partials.throttleMs = 0;
  patch(settings);
  const events: TranslationOutputEvent[] = [];
  const pipeline = createTranslationPipeline({
    name: "tr",
    settings: () => settings,
    engines: { ...engines, ...options.engines },
    publish: (e) => events.push(e),
    pair: options.pair,
    translatorOnCpu: options.translatorOnCpu,
    modelInstalled: options.modelInstalled,
  });
  return {
    pipeline,
    settings,
    events,
    out: () => events.map((e) => `${e.type === TextEventType.final ? "final" : "interim"}:${e.value}`),
    push: (type, value, meta = {}, stream = "stt") => pipeline.push(stream, { type, value, emotes: {}, ...meta }),
  };
}

async function started(patch?: (s: Translation_State) => void, options?: MakeOptions) {
  const h = make(patch, options);
  h.pipeline.start();
  await flush();
  expect(h.pipeline.state.status).toBe(ServiceNetworkState.connected);
  return h;
}

const I = TextEventType.interim;
const F = TextEventType.final;
const seg = (n: number, revision: number, extra: Partial<TextEvent> = {}) => ({ segmentId: `mic:1:${n}`, revision, ...extra });

beforeAll(async () => {
  await i18n.init({ lng: "en", resources: { en: { translation: {} } } });
});

beforeEach(() => {
  FakeEngine.instances = [];
  FakeEngine.startImpl = async () => {};
  FakeEngine.supportsImpl = null;
  vi.spyOn(console, "warn").mockImplementation(() => {});
});

afterEach(() => {
  vi.useRealTimers();
});

describe("LocalAgreement on interims (default)", () => {
  it("publishes only the prefix two translations agree on, growing, with ids, revisions and the target language", async () => {
    const h = await started();
    h.push(I, "good", seg(1, 0));
    engine().call("good").resolve("Buenos");
    await flush();
    expect(h.events).toEqual([]); // one hypothesis: nothing agreed yet
    h.push(I, "good morning every", seg(1, 1));
    engine().call("good morning every").resolve("Buenos días a cada");
    await flush();
    h.push(I, "good morning everyone", seg(1, 2));
    engine().call("good morning everyone").resolve("Buenos días a todos");
    await flush();
    h.push(F, "Good morning everyone.", seg(1, 3, { turnEnd: true }));
    engine().call("Good morning everyone.").resolve("Buenos días a todos.");
    await flush();
    expect(h.out()).toEqual(["interim:Buenos", "interim:Buenos días a", "final:Buenos días a todos."]);
    expect(h.events.map((e) => [e.segmentId, e.revision, e.lang, e.stableLength])).toEqual([
      ["mic:1:1", 1, "es", 6],
      ["mic:1:1", 2, "es", 13],
      ["mic:1:1", 3, "es", undefined],
    ]);
    expect(h.events[2].turnEnd).toBe(true);
  });

  it("never publishes a shorter or diverging interim (no flicker)", async () => {
    const h = await started();
    const hyps = ["We should go", "We should go now", "You must leave", "We should go now please"];
    hyps.forEach((text, i) => {
      h.push(I, `src ${i}`, seg(1, i));
    });
    // Latest wins: only "src 0" runs, then the latest queued one.
    engine().call("src 0").resolve(hyps[0]);
    await flush();
    engine().call("src 3").resolve(hyps[1]);
    await flush();
    h.push(I, "src 4", seg(1, 4));
    engine().call("src 4").resolve(hyps[2]);
    await flush();
    h.push(I, "src 5", seg(1, 5));
    engine().call("src 5").resolve(hyps[3]);
    await flush();
    expect(h.out()).toEqual(["interim:We should go"]);
  });

  it("shows the tentative tail when asked, marking the committed part with stableLength", async () => {
    const h = await started((s) => (s.partials.showTentative = true));
    h.push(I, "a", seg(1, 0));
    engine().call("a").resolve("Hola");
    await flush();
    h.push(I, "ab", seg(1, 1));
    engine().call("ab").resolve("Hola mundo");
    await flush();
    expect(h.events.map((e) => [e.value, e.stableLength])).toEqual([
      ["Hola", 0],
      ["Hola mundo", 4], // "Hola" is committed; the space and "mundo" may still change
    ]);
  });

  it("publishes every distinct interim when LocalAgreement is off", async () => {
    const h = await started((s) => (s.partials.localAgreement = false));
    h.push(I, "a", seg(1, 0));
    engine().call("a").resolve("uno");
    await flush();
    h.push(I, "b", seg(1, 1));
    engine().call("b").resolve("dos");
    await flush();
    expect(h.out()).toEqual(["interim:uno", "interim:dos"]);
    expect(h.events[0].stableLength).toBeUndefined();
  });
});

describe("throttling", () => {
  it("starts interim requests at most every throttleMs, latest wins", async () => {
    vi.useFakeTimers();
    const h = make((s) => (s.partials.throttleMs = 300));
    h.pipeline.start();
    await vi.advanceTimersByTimeAsync(0);
    h.push(I, "a", seg(1, 0));
    engine().call("a").resolve("A");
    await vi.advanceTimersByTimeAsync(10);
    h.push(I, "ab", seg(1, 1));
    h.push(I, "abc", seg(1, 2));
    await vi.advanceTimersByTimeAsync(100);
    expect(engine().calls.map((c) => c.request.text)).toEqual(["a"]);
    await vi.advanceTimersByTimeAsync(200);
    expect(engine().calls.map((c) => c.request.text)).toEqual(["a", "abc"]);
  });
});

describe("Smart Turn commit", () => {
  it("translates a turn split by a pause as one line", async () => {
    const h = await started((s) => (s.partials.localAgreement = false));
    h.push(F, "I think that", seg(1, 1, { turnEnd: false }));
    engine().call("I think that", "interim").resolve("Creo que");
    await flush();
    h.push(F, "we should go.", seg(2, 1, { turnEnd: true }));
    expect(engine().calls.filter((c) => c.request.kind === "final").map((c) => c.request.text)).toEqual(["I think that we should go."]);
    engine().call("I think that we should go.", "final").resolve("Creo que deberíamos ir.");
    await flush();
    expect(h.out()).toEqual(["interim:Creo que", "final:Creo que deberíamos ir."]);
    expect(new Set(h.events.map((e) => e.segmentId))).toEqual(new Set(["mic:1:1"]));
  });

  it("commits a held line after turnHoldMs", async () => {
    vi.useFakeTimers();
    const h = make((s) => (s.partials.turnHoldMs = 1000));
    h.pipeline.start();
    await vi.advanceTimersByTimeAsync(0);
    h.push(F, "So basically", seg(1, 1, { turnEnd: false }));
    await vi.advanceTimersByTimeAsync(999);
    expect(engine().calls.filter((c) => c.request.kind === "final")).toHaveLength(0);
    await vi.advanceTimersByTimeAsync(1);
    engine().call("So basically", "final").resolve("Así que básicamente");
    await vi.advanceTimersByTimeAsync(0);
    expect(h.out().at(-1)).toBe("final:Así que básicamente");
  });
});

describe("requests", () => {
  it("fills the pair from settings, the STT language or the script, and sends context and the line's glossary", async () => {
    const h = await started((s) => {
      s.targetLanguage = "ja";
      s.glossary = [
        { source: "Sigil", target: "シジル", from: "", to: "", caseSensitive: false, enabled: true },
        { source: "unused", target: "x", from: "", to: "", caseSensitive: false, enabled: true },
      ];
      s.context.background = "A VRChat stream.";
    });
    h.push(F, "Hello from Sigil", seg(1, 1, { lang: "en" }));
    const first = engine().call("Hello from Sigil");
    expect(first.request).toMatchObject({
      text: "Hello from Sigil",
      kind: "final",
      from: "en",
      to: "ja",
      requestId: "mic:1:1#0",
      glossary: [{ source: "Sigil", target: "シジル" }],
      background: "A VRChat stream.",
    });
    expect(first.request.context).toBeUndefined();
    first.resolve("シジルからこんにちは");
    await flush();
    h.push(F, "안녕하세요", seg(2, 1));
    const second = engine().call("안녕하세요");
    expect(second.request.from).toBe("ko"); // guessed from the script
    // The first line was en → ja; a ko source still gets it as context (same target, the source differs only when both are known).
    expect(second.request.context).toBeUndefined();
    second.resolve("こんにちは");
    await flush();
    h.push(F, "Next line", seg(3, 1, { lang: "en" }));
    expect(engine().call("Next line").request.context).toEqual([{ source: "Hello from Sigil", target: "シジルからこんにちは" }]);
  });

  it("uses the configured source language over the event's", async () => {
    const h = await started((s) => (s.sourceLanguage = "fr"));
    h.push(F, "Bonjour", seg(1, 1, { lang: "en" }));
    expect(engine().call("Bonjour").request.from).toBe("fr");
  });

  it("respects the context settings live", async () => {
    const h = await started((s) => (s.context.turns = 1));
    h.push(F, "one", seg(1, 1, { lang: "en" }));
    engine().call("one").resolve("uno");
    await flush();
    h.push(F, "two", seg(2, 1, { lang: "en" }));
    engine().call("two").resolve("dos");
    await flush();
    h.push(F, "three", seg(3, 1, { lang: "en" }));
    expect(engine().call("three").request.context).toEqual([{ source: "two", target: "dos" }]);
    h.settings.context.enabled = false;
    h.push(F, "four", seg(4, 1, { lang: "en" }));
    expect(engine().call("four").request.context).toBeUndefined();
  });

  it("leaves the legacy backends on their own slice pair, unless a pair override is given", async () => {
    const legacy = await started((s) => {
      s.backend = Translation_Backends.azure;
      s.azure.language = "zh-Hans";
    });
    legacy.push(F, "hi", seg(1, 1, { lang: "en" }));
    expect(engine().call("hi").request.from).toBeUndefined();
    expect(engine().call("hi").request.to).toBeUndefined();
    engine().call("hi").resolve("你好");
    await flush();
    expect(legacy.events[0].lang).toBe("zh");

    const reversed = await started((s) => (s.backend = Translation_Backends.azure), { pair: () => ({ from: "ja", to: "en" }) });
    reversed.push(F, "こんにちは", {}, "others");
    expect(engine().call("こんにちは").request).toMatchObject({ from: "ja", to: "en" });
  });
});

describe("streamed finals", () => {
  it("shows streamed text that extends the line, holding back the last word", async () => {
    const h = await started();
    h.push(F, "Good morning everyone", seg(1, 1));
    const call = engine().call("Good morning everyone");
    call.request.onPartial?.("Buenos dí");
    call.request.onPartial?.("Buenos días a to");
    call.request.onPartial?.("Buenos días a todos");
    call.resolve("Buenos días a todos.");
    await flush();
    expect(h.out()).toEqual(["interim:Buenos", "interim:Buenos días a", "final:Buenos días a todos."]);
  });

  it("does not show streamed text that diverges from what the line shows", async () => {
    const h = await started();
    h.push(I, "a", seg(1, 0));
    engine().call("a").resolve("Hola a todos");
    await flush();
    h.push(I, "ab", seg(1, 1));
    engine().call("ab").resolve("Hola a todos ustedes");
    await flush();
    expect(h.out()).toEqual(["interim:Hola a todos"]);
    h.push(F, "abc", seg(1, 2));
    const call = engine().call("abc", "final");
    call.request.onPartial?.("Buenas tardes a todos");
    expect(h.out()).toEqual(["interim:Hola a todos"]);
    call.request.onPartial?.("Hola a todos y bienvenidos");
    expect(h.out()).toEqual(["interim:Hola a todos", "interim:Hola a todos y"]);
  });

  it("does not shrink the interim on screen when LocalAgreement is off", async () => {
    const h = await started((s) => (s.partials.localAgreement = false));
    h.push(I, "src", seg(1, 0));
    engine().call("src").resolve("I think we should go now");
    await flush();
    h.push(F, "src done", seg(1, 1));
    const call = engine().call("src done", "final");
    call.request.onPartial?.("I think"); // shorter than what is shown: not published
    call.request.onPartial?.("I think we should go now please thank");
    call.resolve("I think we should go now, please.");
    await flush();
    expect(h.out()).toEqual([
      "interim:I think we should go now",
      "interim:I think we should go now please",
      "final:I think we should go now, please.",
    ]);
  });

  it("does not stream a later line ahead of an earlier final", async () => {
    const h = await started();
    h.push(F, "first", seg(1, 1));
    h.push(F, "second", seg(2, 1));
    engine().call("second").request.onPartial?.("segunda línea larga");
    expect(h.out()).toEqual([]);
    engine().call("first").resolve("primera");
    engine().call("second").resolve("segunda");
    await flush();
    expect(h.out()).toEqual(["final:primera", "final:segunda"]);
  });

  it("does not stream when interims are off for the backend", async () => {
    const h = await started((s) => (s.llm.interim = false));
    h.push(F, "x", seg(1, 1));
    expect(engine().call("x").request.onPartial).toBeUndefined();
  });
});

describe("lines", () => {
  it("closes a given-up segment silently and keeps going", async () => {
    const h = await started((s) => (s.partials.localAgreement = false));
    h.push(I, "lost", seg(1, 0));
    const lost = engine().call("lost");
    h.push(I, "next", seg(2, 0));
    expect(lost.signal.aborted).toBe(true);
    lost.resolve("perdido");
    engine().call("next").resolve("siguiente");
    await flush();
    expect(h.out()).toEqual(["interim:siguiente"]);
  });

  it("publishes the cancel marker only for a line that showed something", async () => {
    const h = await started((s) => (s.partials.localAgreement = false));
    h.push(I, "half", seg(1, 0));
    engine().call("half").resolve("medio");
    await flush();
    h.push(F, CANCEL_MARKER, seg(1, 1));
    await flush();
    expect(h.out()).toEqual(["interim:medio", `final:${CANCEL_MARKER}`]);

    const quiet = await started(); // LocalAgreement: one hypothesis shows nothing
    quiet.push(I, "half", seg(1, 0));
    engine().call("half").resolve("medio");
    await flush();
    quiet.push(F, CANCEL_MARKER, seg(1, 1));
    await flush();
    expect(quiet.out()).toEqual([]);
  });

  it("keeps two pipelines independent", async () => {
    const a = await started();
    const engineA = engine();
    const b = await started((s) => (s.targetLanguage = "de"));
    const engineB = engine();
    a.push(F, "hello", seg(1, 1));
    b.push(F, "hello", seg(1, 1));
    b.pipeline.stop();
    expect(engineB.stopped).toBe(1);
    expect(engineB.call("hello").signal.aborted).toBe(true);
    engineA.call("hello").resolve("hola");
    await flush();
    expect(a.out()).toEqual(["final:hola"]);
    expect(b.out()).toEqual([]);
    expect(engineA.stopped).toBe(0);
  });

  it("gives lines without a segment id a generated id carrying the process nonce", async () => {
    const h = await started((s) => (s.partials.localAgreement = false));
    h.push(F, "hello");
    engine().call("hello").resolve("hola");
    await flush();
    expect(h.out()).toEqual(["final:hola"]);
    // `<name>.<stream>.<nonce>.<epoch>:<n>`: unique across restarts and linked hosts, stream part before the ':'.
    expect(h.events[0].segmentId).toMatch(new RegExp(String.raw`^tr\.stt\.${PROCESS_NONCE}\.\d+:\d+$`));
  });
});

describe("state", () => {
  it("shows engine status, reports the engine used and the missing ids of a failed start", async () => {
    FakeEngine.startImpl = async (_signal, host) => {
      host?.status({ text: "Loading", progress: 40 });
    };
    const h = await started();
    expect(h.pipeline.state.statusText).toBe("Loading");
    expect(h.pipeline.state.progress).toBe(40);
    engine().host?.status({ text: "" });
    expect(h.pipeline.state.statusText).toBe("");
    expect(h.pipeline.state.progress).toBeNull();

    h.push(F, "x", seg(1, 1));
    const call = engine().call("x");
    call.request.onEngine?.(Translation_Backends.bergamot);
    call.resolve("equis");
    await flush();
    expect(h.pipeline.state.activeEngine).toBe("bergamot");

    FakeEngine.startImpl = async () => {
      throw new TranslationError("not installed", { missing: ["mt/hy-mt2-1.8b-q4_k_m"] });
    };
    const failed = make();
    failed.pipeline.start();
    await flush();
    expect(failed.pipeline.state.status).toBe(ServiceNetworkState.disconnected);
    expect(failed.pipeline.state.error).toBe("not installed");
    expect(failed.pipeline.state.missing).toEqual(["mt/hy-mt2-1.8b-q4_k_m"]);
  });

  it("refuses to start when the engine cannot translate into the target", async () => {
    FakeEngine.supportsImpl = (_from, to) => to !== "es";
    const h = make();
    h.pipeline.start();
    await flush();
    expect(h.pipeline.state.status).toBe(ServiceNetworkState.disconnected);
    expect(h.pipeline.state.error).toBe("transl.error_unsupported_target");
    expect(engine().stopped).toBe(1);
  });
});

/** Bergamot's `supports` semantics: installed directions only, and no answer for an unknown source. */
class BergamotLike implements ITranslationEngine {
  static installed = new Set<string>();
  static startStates: Translation_State[] = [];
  start(state: Translation_State) {
    BergamotLike.startStates.push(state);
    return Promise.resolve();
  }
  translate(request: TranslationRequest) {
    return Promise.resolve(`bergamot:${request.text}`);
  }
  stop() {}
  supports(from: string, to: string) {
    return from !== "" && BergamotLike.installed.has(`${from}>${to}`);
  }
}

/** A local LLM still loading its model. */
class LoadingEngine implements ITranslationEngine {
  start() {
    return new Promise<void>(() => {});
  }
  translate(request: TranslationRequest) {
    return Promise.resolve(`llm:${request.text}`);
  }
  stop() {}
}

describe("start check (engines that cannot detect the source)", () => {
  const bergamot = { [Translation_Backends.bergamot]: BergamotLike };

  beforeEach(() => {
    BergamotLike.installed = new Set(["en>ja"]);
    BergamotLike.startStates = [];
  });

  it("starts Bergamot with a known or an unknown source", async () => {
    for (const source of ["en", ""]) {
      await started(
        (s) => {
          s.backend = Translation_Backends.bergamot;
          s.sourceLanguage = source;
          s.targetLanguage = "ja";
        },
        { engines: bergamot },
      );
    }
  });

  it("names the problem when the pair or the target cannot be translated", async () => {
    const pair = make(
      (s) => {
        s.backend = Translation_Backends.bergamot;
        s.sourceLanguage = "fr";
        s.targetLanguage = "ja";
      },
      { engines: bergamot },
    );
    pair.pipeline.start();
    await flush();
    expect(pair.pipeline.state.status).toBe(ServiceNetworkState.disconnected);
    expect(pair.pipeline.state.error).toBe("transl.error_unsupported_pair");

    const target = make(
      (s) => {
        s.backend = Translation_Backends.bergamot;
        s.targetLanguage = "de";
      },
      { engines: bergamot },
    );
    target.pipeline.start();
    await flush();
    expect(target.pipeline.state.error).toBe("transl.error_unsupported_target");
  });

  it("starts auto while the LLM is still loading and Bergamot is ready first", async () => {
    for (const source of ["en", ""]) {
      const h = await started(
        (s) => {
          s.backend = Translation_Backends.auto;
          s.auto.chain = [Translation_Backends.llm, Translation_Backends.bergamot];
          s.sourceLanguage = source;
          s.targetLanguage = "ja";
        },
        { engines: { ...bergamot, [Translation_Backends.llm]: LoadingEngine, [Translation_Backends.auto]: Translation_RouterService } },
      );
      h.push(F, "hello", seg(1, 1, { lang: "en" }));
      await flush();
      expect(h.out()).toEqual(["final:bergamot:hello"]);
      h.pipeline.stop();
    }
  });

  it("starts engines for the pair override, not the settings' pair", async () => {
    BergamotLike.installed = new Set(["ja>en"]);
    await started(
      (s) => {
        s.backend = Translation_Backends.bergamot;
        s.sourceLanguage = "en";
        s.targetLanguage = "ja";
      },
      { engines: bergamot, pair: () => ({ from: "ja", to: "en" }) },
    );
    expect(BergamotLike.startStates.map((s) => [s.sourceLanguage, s.targetLanguage])).toEqual([["ja", "en"]]);
  });
});

describe("property: no dropped finals, ordered, interims never flicker", () => {
  const WORDS = ["good", "morning", "every", "one", "we", "should", "go", "now", "please", "thanks"];

  it("holds for random speech, random engine timing and random Smart Turn verdicts", async () => {
    vi.useFakeTimers();
    const coverage = { interims: 0, merged: 0, givenUp: 0, streamed: 0 };
    for (let seed = 1; seed <= 80; seed++) {
      FakeEngine.instances = [];
      const r = rng(seed);
      const h = make((s) => {
        s.partials.throttleMs = r() < 0.5 ? 0 : 50;
        s.partials.showTentative = r() < 0.3;
        s.partials.turnHoldMs = 800;
      });
      h.pipeline.start();
      await vi.advanceTimersByTimeAsync(0);
      const eng = engine();
      const expectedFinals: string[] = [];
      let answered = 0;

      // The engine answers pending calls in random order: finals as ⟦source⟧, interims noisily (unstable tails).
      const answer = async (n: number) => {
        for (let k = 0; k < n; k++) {
          const pending = eng.calls.filter((c) => !(c as Call & { done?: boolean }).done);
          if (!pending.length) break;
          const c = pending[Math.floor(r() * pending.length)] as Call & { done?: boolean };
          c.done = true;
          answered++;
          if (c.request.kind === "final") {
            const out = `⟦${c.request.text}⟧`;
            if (r() < 0.5 && c.request.onPartial) {
              coverage.streamed++;
              c.request.onPartial(out.slice(0, Math.floor(out.length / 2)));
            }
            c.resolve(out);
          } else {
            const noise = WORDS[Math.floor(r() * WORDS.length)];
            c.resolve(`⟦${c.request.text} ${noise}`);
          }
        }
        await vi.advanceTimersByTimeAsync(Math.floor(r() * 120));
      };

      const segments = 3 + Math.floor(r() * 8);
      for (let n = 1; n <= segments; n++) {
        const words = Array.from({ length: 1 + Math.floor(r() * 5) }, () => WORDS[Math.floor(r() * WORDS.length)]);
        let revision = 0;
        for (let w = 1; w <= words.length; w++) {
          h.push(I, words.slice(0, w).join(" "), seg(n, revision++));
          await answer(Math.floor(r() * 3));
        }
        if (r() < 0.1) {
          coverage.givenUp++;
          continue; // given up: no final ever comes
        }
        const verdict = r();
        const turnEnd = verdict < 0.3 ? false : verdict < 0.6 ? true : undefined;
        const text = words.join(" ");
        h.push(F, text, seg(n, revision++, turnEnd === undefined ? {} : { turnEnd }));
        expectedFinals.push(text);
        if (r() < 0.2) h.push(I, words[0], seg(n, revision++)); // a late interim after the final
        await answer(Math.floor(r() * 3));
      }
      // Drain: answer everything and let held turns expire.
      for (let i = 0; i < 200 && (eng.calls.some((c) => !(c as Call & { done?: boolean }).done) || i < 3); i++) {
        await answer(5);
        await vi.advanceTimersByTimeAsync(1000);
      }

      const finals = h.events.filter((e) => e.type === TextEventType.final);
      const sources = finals.map((e) => {
        expect(e.value.startsWith("⟦") && e.value.endsWith("⟧"), `seed ${seed}: ${e.value}`).toBe(true);
        return e.value.slice(1, -1);
      });
      // Every STT final is translated exactly once, in order (merged turns keep their order).
      expect(sources.join(" "), `seed ${seed}`).toBe(expectedFinals.join(" "));
      coverage.merged += expectedFinals.length - sources.length;
      coverage.interims += h.events.filter((e) => e.type === TextEventType.interim).length;

      const byLine = new Map<string, TranslationOutputEvent[]>();
      for (const e of h.events) byLine.set(e.segmentId!, [...(byLine.get(e.segmentId!) ?? []), e]);
      for (const [id, events] of byLine) {
        let committed = "";
        let lastRevision = 0;
        let finalSeen = false;
        for (const e of events) {
          expect(e.revision!, `seed ${seed} ${id}`).toBeGreaterThan(lastRevision);
          lastRevision = e.revision!;
          expect(finalSeen, `seed ${seed} ${id}: event after the final`).toBe(false);
          if (e.type === TextEventType.final) {
            finalSeen = true;
            continue;
          }
          const stable = e.value.slice(0, e.stableLength ?? e.value.length);
          expect(stable.startsWith(committed), `seed ${seed} ${id}: "${committed}" → "${stable}"`).toBe(true);
          committed = stable;
        }
      }
      expect(answered).toBeGreaterThan(0);
      h.pipeline.stop();
    }
    // The scenarios really exercise interims, merged turns, given-up segments and streamed finals.
    expect(coverage.interims).toBeGreaterThan(100);
    expect(coverage.merged).toBeGreaterThan(10);
    expect(coverage.givenUp).toBeGreaterThan(5);
    expect(coverage.streamed).toBeGreaterThan(20);
  });
});

describe("automatic interim switch (P7: off while the bundled translator runs on the processor)", () => {
  const interimCalls = () => engine().calls.filter((c) => c.request.kind === "interim");

  it("translates interims by default (the translator is not on the processor)", async () => {
    const h = await started();
    expect(h.settings.llm.interim).toBe("auto");
    h.push(I, "hello there", seg(1, 0));
    expect(interimCalls()).toHaveLength(1);
  });

  it("hands the engine the processor fact for the router's CPU-tier routes (read when asked, false without an option)", async () => {
    let onCpu = false;
    const h = await started(undefined, { translatorOnCpu: () => onCpu });
    expect(engine().host?.translatorOnCpu?.()).toBe(false);
    onCpu = true;
    expect(engine().host?.translatorOnCpu?.()).toBe(true);
    h.pipeline.stop();
    const plain = await started();
    expect(engine().host?.translatorOnCpu?.()).toBe(false);
    plain.pipeline.stop();
  });

  it("hands the engine what the model store reports (read when asked), and nothing without an option", async () => {
    const installed = new Set<string>();
    const h = await started(undefined, { modelInstalled: (id) => installed.has(id) });
    expect(engine().host?.modelInstalled?.("mt/nllb")).toBe(false);
    installed.add("mt/nllb");
    expect(engine().host?.modelInstalled?.("mt/nllb")).toBe(true);
    h.pipeline.stop();
    const plain = await started();
    expect(engine().host?.modelInstalled).toBeUndefined(); // the router then keeps its timed retry
    plain.pipeline.stop();
  });

  it("holds interims off on the processor but still translates the final", async () => {
    const h = await started(undefined, { translatorOnCpu: () => true });
    h.push(I, "hello", seg(1, 0));
    h.push(I, "hello there", seg(1, 1));
    expect(interimCalls()).toHaveLength(0);
    h.push(F, "Hello there.", seg(1, 2, { turnEnd: true }));
    expect(engine().call("Hello there.").request.onPartial).toBeUndefined();
    engine().call("Hello there.").resolve("Hola.");
    await flush();
    expect(h.out()).toEqual(["final:Hola."]);
  });

  it("a stored explicit true wins on the processor, an explicit false wins everywhere", async () => {
    const on = await started((s) => (s.llm.interim = true), { translatorOnCpu: () => true });
    on.push(I, "hello", seg(1, 0));
    expect(interimCalls()).toHaveLength(1);
    const off = await started((s) => (s.llm.interim = false), { translatorOnCpu: () => false });
    off.push(I, "hello", seg(1, 0));
    expect(interimCalls()).toHaveLength(0);
  });

  it("follows the translator while it runs (the governor's choice shows once the server is up)", async () => {
    let onCpu = false;
    const h = await started(undefined, { translatorOnCpu: () => onCpu });
    h.push(I, "one", seg(1, 0));
    expect(interimCalls()).toHaveLength(1);
    engine().call("one").resolve("uno");
    await flush();
    onCpu = true;
    h.push(I, "one two", seg(1, 1));
    await flush();
    expect(interimCalls()).toHaveLength(1);
    onCpu = false;
    h.push(I, "one two three", seg(1, 2));
    await flush();
    expect(interimCalls()).toHaveLength(2);
  });

  it("the automatic router follows the same rule when its chain uses the bundled translator", async () => {
    const withLlm = await started((s) => (s.backend = Translation_Backends.auto), { translatorOnCpu: () => true });
    withLlm.push(I, "hello", seg(1, 0));
    expect(interimCalls()).toHaveLength(0);
    const without = await started(
      (s) => {
        s.backend = Translation_Backends.auto;
        s.auto.chain = [Translation_Backends.nllb];
      },
      { translatorOnCpu: () => true },
    );
    without.push(I, "hello", seg(1, 0));
    expect(interimCalls()).toHaveLength(1);
    const explicit = await started(
      (s) => {
        s.backend = Translation_Backends.auto;
        s.auto.interim = true;
      },
      { translatorOnCpu: () => true },
    );
    explicit.push(I, "hello", seg(1, 0));
    expect(interimCalls()).toHaveLength(1);
  });
});
