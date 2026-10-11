import i18n from "i18next";
import { afterEach, beforeAll, describe, expect, it, vi } from "vitest";
import {
  BREAKER_BASE_COOLDOWN_MS,
  chainForPair,
  CircuitBreaker,
  isCloudEngine,
  ROUTER_START_RETRY_MS,
  routeForPair,
  routerAttemptBudgetMs,
  routerLineBudgetMs,
  Translation_RouterService,
} from "../router";
import { Service_Translation_Schema, TRANSLATION_ENGINE_IDS, Translation_Backends as B, Translation_State, TranslationEngineId } from "../schema";
import {
  ITranslationEngine,
  ITranslationEngineConstructor,
  TranslationEngineHost,
  TranslationError,
  TranslationRequest,
  TranslationStatusUpdate,
} from "../types";

interface Ctl {
  start?: (signal: AbortSignal) => Promise<void>;
  translate?: (request: TranslationRequest, signal: AbortSignal) => Promise<string>;
  supports?: (from: string, to: string) => boolean;
  calls: { request: TranslationRequest; signal: AbortSignal }[];
  starts: number;
  stops: number;
}

function setup(patch: (s: Translation_State) => void = () => {}) {
  const ctl = Object.fromEntries(TRANSLATION_ENGINE_IDS.map((id) => [id, { calls: [], starts: 0, stops: 0 } as Ctl])) as Record<
    TranslationEngineId,
    Ctl
  >;
  const engines = Object.fromEntries(
    TRANSLATION_ENGINE_IDS.map((id) => {
      const c = ctl[id];
      const Engine = class implements ITranslationEngine {
        start(_s: Translation_State, signal: AbortSignal) {
          c.starts++;
          return c.start ? c.start(signal) : Promise.resolve();
        }
        translate(request: TranslationRequest, signal: AbortSignal) {
          c.calls.push({ request, signal });
          return c.translate ? c.translate(request, signal) : Promise.resolve(`${id}:${request.text}`);
        }
        stop() {
          c.stops++;
        }
        supports(from: string, to: string) {
          return c.supports ? c.supports(from, to) : true;
        }
      };
      return [id, Engine];
    }),
  ) as Record<TranslationEngineId, ITranslationEngineConstructor>;
  const statuses: TranslationStatusUpdate[] = [];
  /** `onCpu.value`: the bundled translator runs on the processor (the host's `translatorOnCpu`). */
  const onCpu = { value: false };
  const host: TranslationEngineHost = { engines, status: (u) => statuses.push(u), translatorOnCpu: () => onCpu.value };
  const state = Service_Translation_Schema.parse({
    sourceLanguage: "en",
    targetLanguage: "ja",
    auto: { chain: ["llm", "bergamot", "nllb"], attemptTimeoutMs: 1000 },
  });
  patch(state);
  const router = new Translation_RouterService();
  router.attach(host);
  return { router, ctl, statuses, state, onCpu };
}

const live = () => new AbortController().signal;
const req = (text: string, kind: "final" | "interim" = "final"): TranslationRequest => ({ text, kind });
const never = () => new Promise<never>(() => {});
const fail = (message: string, retryable = true) => () => Promise.reject(new TranslationError(message, { retryable }));

beforeAll(async () => {
  await i18n.init({ lng: "en", resources: { en: { translation: {} } } });
});

afterEach(() => {
  vi.useRealTimers();
});

describe("chainForPair", () => {
  it("uses the first matching override, else the default chain", () => {
    const auto = Service_Translation_Schema.parse({
      auto: {
        chain: ["llm", "bergamot"],
        pairOverrides: [
          { from: "ja", to: "en", chain: ["deepl"] },
          { from: "", to: "zh-TW", chain: ["nllb"] },
        ],
      },
    }).auto;
    expect(chainForPair(auto, "ja", "en")).toEqual(["deepl"]);
    expect(chainForPair(auto, "ko", "zh-Hant")).toEqual(["nllb"]);
    expect(chainForPair(auto, "", "en")).toEqual(["llm", "bergamot"]); // unknown source never matches a from-specific override
    expect(chainForPair(auto, "en", "ja")).toEqual(["llm", "bergamot"]);
  });

  it("knows which engines are cloud engines", () => {
    const s = Service_Translation_Schema.parse({});
    expect(isCloudEngine(B.deepl, s)).toBe(true);
    expect(isCloudEngine(B.bergamot, s)).toBe(false);
    expect(isCloudEngine(B.llm, s)).toBe(false); // bundled
    s.llm.provider = "gemini";
    expect(isCloudEngine(B.llm, s)).toBe(true);
  });
});

describe("CircuitBreaker", () => {
  it("opens after consecutive failures, probes once after the cooldown, doubles the cooldown on a failed probe", () => {
    const b = new CircuitBreaker(2, 1000, 8000);
    expect(b.allow(0)).toBe(true);
    b.failure(0);
    expect(b.state).toBe("closed");
    b.failure(0);
    expect(b.state).toBe("open");
    expect(b.allow(999)).toBe(false);
    expect(b.allow(1000)).toBe(true); // the probe
    expect(b.allow(1000)).toBe(false); // one at a time
    b.failure(1000);
    expect(b.allow(2999)).toBe(false); // 2 s now
    expect(b.allow(3000)).toBe(true);
    b.success();
    expect(b.state).toBe("closed");
  });

  it("reports whether it holds requests back without side effects", () => {
    const b = new CircuitBreaker(1, 1000);
    expect(b.blocked(0)).toBe(false);
    b.failure(0);
    expect(b.blocked(999)).toBe(true);
    expect(b.blocked(1000)).toBe(false);
    expect(b.allow(1000)).toBe(true); // blocked() did not use up the probe
    expect(b.blocked(1000)).toBe(true); // the probe is out
  });

  it("opens at once on a severe failure and ignores cancellations", () => {
    const b = new CircuitBreaker();
    b.failure(0, true);
    expect(b.allow(1)).toBe(false);
    const c = new CircuitBreaker(1, 100);
    c.failure(0);
    expect(c.allow(100)).toBe(true);
    c.release();
    expect(c.allow(100)).toBe(true);
  });
});

describe("attempt budgets", () => {
  it("scales a long final's budget with its length and keeps interims at attemptTimeoutMs", () => {
    expect(routerAttemptBudgetMs(6000, "final", "a short caption")).toBe(6000);
    expect(routerAttemptBudgetMs(6000, "final", "x".repeat(600))).toBe(30_000);
    expect(routerAttemptBudgetMs(6000, "final", "x".repeat(100_000))).toBe(60_000);
    expect(routerAttemptBudgetMs(6000, "interim", "x".repeat(600))).toBe(6000);
  });

  it("covers every member's start wait and attempt in a line's budget", () => {
    const auto = Service_Translation_Schema.parse({ auto: { chain: ["llm", "bergamot", "nllb"], attemptTimeoutMs: 1000 } }).auto;
    // 3 chain members + 2 local fallbacks, plus half an attempt budget: a built-in route gives NLLB's attempt 1.5 times the budget
    expect(routerLineBudgetMs(auto, "final", "hi")).toBe(5 * 2000 + 500);
    expect(routerLineBudgetMs(auto, "interim", "hi")).toBe(2 * 2000);
    auto.cloudFallback = false;
    expect(routerLineBudgetMs(auto, "final", "x".repeat(600))).toBe(3 * (1000 + 5000) + 2500);
  });
});

describe("Translation_RouterService supports", () => {
  it("answers false only when every chain member has settled and none supports the pair", async () => {
    const { router, ctl, state } = setup((s) => (s.auto.chain = [B.llm, B.bergamot]));
    let finishStart!: () => void;
    ctl.llm.start = () => new Promise<void>((r) => (finishStart = r));
    ctl.llm.supports = (_from, to) => to !== "ja";
    // Bergamot: installed directions only, no answer for an unknown source.
    ctl.bergamot.supports = (from, to) => from === "en" && to === "ja";
    await router.start(state, live()); // Bergamot is ready first
    expect(router.supports("en", "ja")).toBe(true);
    expect(router.supports("", "ja")).toBe(true); // the LLM is still starting
    expect(router.supports("fr", "ja")).toBe(true);
    finishStart();
    await vi.waitFor(() => expect(router.supports("fr", "ja")).toBe(false));
    expect(router.supports("en", "ja")).toBe(true);
    expect(router.supports("fr", "de")).toBe(true);
  });
});

describe("Translation_RouterService start", () => {
  it("refuses to start without a host or a target", async () => {
    await expect(new Translation_RouterService().start(Service_Translation_Schema.parse({}), live())).rejects.toBeInstanceOf(TranslationError);
    const { router, state } = setup((s) => (s.targetLanguage = ""));
    await expect(router.start(state, live())).rejects.toThrow("transl.error_target_missing");
  });

  it("starts every chain member and resolves as soon as one is ready", async () => {
    const { router, ctl, state } = setup();
    ctl.llm.start = never; // a slow model load
    await router.start(state, live());
    expect(ctl.llm.starts).toBe(1);
    expect(ctl.bergamot.starts).toBe(1);
    expect(ctl.nllb.starts).toBe(1);
    expect(ctl.deepl.starts).toBe(0);
  });

  it("fails with every member's reason and the missing ids when nothing starts", async () => {
    const { router, ctl, state } = setup();
    ctl.llm.start = () => Promise.reject(new TranslationError("no model", { missing: ["mt/hy-mt2-1.8b-q4_k_m"] }));
    ctl.bergamot.start = () => Promise.reject(new TranslationError("no helper", { missing: ["runtime/bergamot"] }));
    ctl.nllb.start = () => Promise.reject(new Error("no nllb"));
    const err = await router.start(state, live()).catch((e) => e);
    expect(err).toBeInstanceOf(TranslationError);
    expect(err.message).toBe("transl.router_error_none_started");
    expect(err.missing).toEqual(["mt/hy-mt2-1.8b-q4_k_m", "runtime/bergamot"]);
  });

  it("aborts with the start signal", async () => {
    const { router, ctl, state } = setup();
    ctl.llm.start = ctl.bergamot.start = ctl.nllb.start = never;
    const c = new AbortController();
    const p = router.start(state, c.signal);
    c.abort(new Error("stopped"));
    await expect(p).rejects.toThrow("stopped");
  });
});

describe("Translation_RouterService translate", () => {
  it("uses the first engine and reports it", async () => {
    const { router, state } = setup();
    await router.start(state, live());
    const used: string[] = [];
    await expect(router.translate({ ...req("hi"), onEngine: (id) => used.push(id) }, live())).resolves.toBe("llm:hi");
    expect(used).toEqual(["llm"]);
  });

  it("passes the canonical pair to members", async () => {
    const { router, ctl, state } = setup((s) => (s.sourceLanguage = "EN_us"));
    await router.start(state, live());
    await router.translate(req("hi"), live());
    expect(ctl.llm.calls[0].request).toMatchObject({ from: "en", to: "ja" });
    await router.translate({ ...req("hi"), from: "ko", to: "zh-TW" }, live());
    expect(ctl.llm.calls[1].request).toMatchObject({ from: "ko", to: "zh-Hant" });
  });

  it("falls through on a failure and shows a fallback warning until the first engine answers again", async () => {
    const { router, ctl, statuses, state } = setup();
    await router.start(state, live());
    ctl.llm.translate = fail("server down");
    const used: string[] = [];
    await expect(router.translate({ ...req("a"), onEngine: (id) => used.push(id) }, live())).resolves.toBe("bergamot:a");
    expect(used).toEqual(["bergamot"]);
    expect(statuses.at(-1)).toEqual({ text: "transl.router_status_fallback", level: "warning" });
    ctl.llm.translate = undefined;
    vi.useFakeTimers();
    vi.advanceTimersByTime(BREAKER_BASE_COOLDOWN_MS);
    vi.useRealTimers();
    await expect(router.translate(req("b"), live())).resolves.toBe("llm:b");
    expect(statuses.at(-1)).toEqual({ text: "" });
  });

  it("falls through within one attempt timeout when an engine hangs (acceptance)", async () => {
    vi.useFakeTimers();
    const { router, ctl, state } = setup();
    await router.start(state, live());
    ctl.llm.translate = never;
    ctl.bergamot.translate = never;
    const p = router.translate(req("x"), live());
    let done: string | undefined;
    void p.then((v) => (done = v));
    await vi.advanceTimersByTimeAsync(999);
    expect(done).toBeUndefined();
    expect(ctl.bergamot.calls).toHaveLength(0);
    await vi.advanceTimersByTimeAsync(1);
    expect(ctl.llm.calls[0].signal.aborted).toBe(true); // the hung attempt is cancelled
    expect(ctl.bergamot.calls).toHaveLength(1);
    await vi.advanceTimersByTimeAsync(1000);
    expect(done).toBe("nllb:x"); // two hung engines, answered after two timeouts
  });

  it("opens the breaker after repeated failures and probes again after the cooldown", async () => {
    vi.useFakeTimers();
    const { router, ctl, state } = setup();
    await router.start(state, live());
    ctl.llm.translate = fail("500");
    await router.translate(req("1"), live());
    await router.translate(req("2"), live());
    expect(ctl.llm.calls).toHaveLength(2);
    await router.translate(req("3"), live());
    expect(ctl.llm.calls).toHaveLength(2); // open: skipped
    ctl.llm.translate = undefined;
    vi.advanceTimersByTime(BREAKER_BASE_COOLDOWN_MS);
    await expect(router.translate(req("4"), live())).resolves.toBe("llm:4");
    await expect(router.translate(req("5"), live())).resolves.toBe("llm:5");
  });

  it("opens the breaker at once for a non-retryable failure", async () => {
    const { router, ctl, state } = setup();
    await router.start(state, live());
    ctl.llm.translate = fail("bad key", false);
    await router.translate(req("1"), live());
    await router.translate(req("2"), live());
    expect(ctl.llm.calls).toHaveLength(1);
  });

  it("does not blame an engine for a cancelled request", async () => {
    const { router, ctl, state } = setup();
    await router.start(state, live());
    ctl.llm.translate = never;
    for (const text of ["x1", "x2"]) {
      const c = new AbortController();
      const p = router.translate(req(text), c.signal);
      await vi.waitFor(() => expect(ctl.llm.calls.some((call) => call.request.text === text)).toBe(true));
      c.abort(new Error("superseded"));
      await expect(p).rejects.toThrow("superseded");
    }
    expect(ctl.bergamot.calls).toHaveLength(0); // a cancelled line does not fall through
    ctl.llm.translate = undefined;
    await expect(router.translate(req("y"), live())).resolves.toBe("llm:y");
  });

  it("skips engines that do not support the pair, without calling that a fallback", async () => {
    const { router, ctl, statuses, state } = setup();
    await router.start(state, live());
    ctl.llm.supports = (_f, to) => to !== "ja";
    await expect(router.translate(req("x"), live())).resolves.toBe("bergamot:x");
    expect(ctl.llm.calls).toHaveLength(0);
    expect(statuses.filter((u) => u.text === "transl.router_status_fallback")).toEqual([]);
  });

  it("tries local engines outside the chain after a cloud failure (cloudFallback)", async () => {
    const { router, ctl, state } = setup((s) => (s.auto.chain = [B.deepl]));
    await router.start(state, live());
    ctl.deepl.translate = fail("456 quota");
    await expect(router.translate(req("x"), live())).resolves.toBe("bergamot:x");
    expect(ctl.bergamot.starts).toBe(1); // started on demand
  });

  it("does not use local extras without a cloud failure or with cloudFallback off", async () => {
    const off = setup((s) => {
      s.auto.chain = [B.deepl];
      s.auto.cloudFallback = false;
    });
    await off.router.start(off.state, live());
    off.ctl.deepl.translate = fail("456 quota");
    await expect(off.router.translate(req("x"), live())).rejects.toThrow("transl.router_error_all_failed");
    const local = setup((s) => (s.auto.chain = [B.llm]));
    await local.router.start(local.state, live());
    local.ctl.llm.translate = fail("down");
    await expect(local.router.translate(req("x"), live())).rejects.toThrow("transl.router_error_all_failed");
    expect(local.ctl.bergamot.starts).toBe(0);
  });

  it("uses a later ready engine instead of waiting for one still starting", async () => {
    const { router, ctl, state } = setup();
    let finishStart!: () => void;
    ctl.llm.start = () => new Promise<void>((r) => (finishStart = r));
    await router.start(state, live());
    await expect(router.translate(req("early"), live())).resolves.toBe("bergamot:early");
    finishStart();
    await Promise.resolve();
    await Promise.resolve();
    await expect(router.translate(req("later"), live())).resolves.toBe("llm:later");
  });

  it("starts the members of a pair override on first use", async () => {
    const { router, ctl, state } = setup((s) => (s.auto.pairOverrides = [{ from: "", to: "ko", chain: [B.nllb, B.deepl] }]));
    await router.start(state, live());
    expect(ctl.deepl.starts).toBe(0);
    ctl.nllb.translate = fail("x");
    await expect(router.translate({ ...req("hi"), to: "ko" }, live())).resolves.toBe("deepl:hi");
    expect(ctl.deepl.starts).toBe(1);
  });

  it("retries a failed member start after ROUTER_START_RETRY_MS", async () => {
    vi.useFakeTimers();
    const { router, ctl, state } = setup((s) => (s.auto.chain = [B.nllb, B.bergamot]));
    ctl.nllb.start = () => Promise.reject(new TranslationError("not installed", { missing: ["mt/nllb"] }));
    await router.start(state, live());
    await expect(router.translate(req("a"), live())).resolves.toBe("bergamot:a");
    expect(ctl.nllb.starts).toBe(1);
    ctl.nllb.start = undefined; // installed meanwhile
    vi.advanceTimersByTime(ROUTER_START_RETRY_MS);
    await expect(router.translate(req("b"), live())).resolves.toBe("nllb:b");
    expect(ctl.nllb.starts).toBe(2);
  });

  it("counts an attempt timeout as an ordinary failure, not a severe one", async () => {
    vi.useFakeTimers();
    const { router, ctl, state } = setup();
    await router.start(state, live());
    ctl.llm.translate = never;
    const p = router.translate(req("a"), live());
    await vi.advanceTimersByTimeAsync(1000);
    await expect(p).resolves.toBe("bergamot:a");
    ctl.llm.translate = undefined;
    await expect(router.translate(req("b"), live())).resolves.toBe("llm:b"); // one timeout does not open the breaker
  });

  it("probes a member whose breaker is open instead of dropping a final", async () => {
    const only = setup((s) => (s.auto.chain = [B.llm]));
    await only.router.start(only.state, live());
    only.ctl.llm.translate = fail("server restarting", false); // opens the breaker at once
    await expect(only.router.translate(req("1"), live())).rejects.toThrow("transl.router_error_all_failed");
    only.ctl.llm.translate = undefined; // back
    await expect(only.router.translate(req("2"), live())).resolves.toBe("llm:2");

    const chain = setup();
    await chain.router.start(chain.state, live());
    chain.ctl.llm.translate = fail("server restarting", false);
    await expect(chain.router.translate(req("1"), live())).resolves.toBe("bergamot:1");
    chain.ctl.llm.translate = undefined;
    chain.ctl.bergamot.translate = fail("busy");
    chain.ctl.nllb.translate = fail("busy");
    // LLM breaker open, the fallbacks fail: the LLM is tried anyway before the final is given up.
    await expect(chain.router.translate(req("2"), live())).resolves.toBe("llm:2");
  });

  it("gives a long final proportionally more time than attemptTimeoutMs", async () => {
    vi.useFakeTimers();
    const { router, ctl, state } = setup(); // attemptTimeoutMs 1000
    await router.start(state, live());
    const paragraph = "x".repeat(600);
    ctl.llm.translate = (r) => new Promise((resolve) => setTimeout(() => resolve(`llm:${r.text.length}`), 4000));
    const p = router.translate(req(paragraph), live());
    await vi.advanceTimersByTimeAsync(4000);
    await expect(p).resolves.toBe("llm:600");
    expect(ctl.bergamot.calls).toHaveLength(0);
  });

  it("waits for a member still starting only after the others failed, without charging the start to it", async () => {
    vi.useFakeTimers();
    const { router, ctl, state } = setup((s) => (s.auto.chain = [B.llm, B.bergamot]));
    let finishStart!: () => void;
    ctl.llm.start = () => new Promise<void>((r) => (finishStart = r));
    await router.start(state, live()); // Bergamot is ready, the LLM still loads
    ctl.bergamot.translate = fail("busy");
    ctl.llm.translate = (r) => new Promise((resolve) => setTimeout(() => resolve(`llm:${r.text}`), 900));
    // The start takes longer than attemptTimeoutMs: this line fails, and the LLM is not blamed for it.
    const first = router.translate(req("a"), live()).catch((e: unknown) => e);
    await vi.advanceTimersByTimeAsync(1000);
    expect(await first).toBeInstanceOf(TranslationError);
    // The start finishes during the wait: the translation still gets its whole budget after it.
    const second = router.translate(req("b"), live());
    await vi.advanceTimersByTimeAsync(800);
    finishStart();
    await vi.advanceTimersByTimeAsync(900);
    await expect(second).resolves.toBe("llm:b");
  });

  it("stops every member", async () => {
    const { router, ctl, state } = setup();
    await router.start(state, live());
    router.stop();
    expect(ctl.llm.stops + ctl.bergamot.stops + ctl.nllb.stops).toBe(3);
    await expect(router.translate(req("x"), live())).rejects.toThrow("transl.error_not_running");
  });
});

describe("Translation_RouterService interims", () => {
  it("sends interims to the cheapest ready local engine (Bergamot) and finals to the chain", async () => {
    const { router, ctl, state } = setup();
    await router.start(state, live());
    await expect(router.translate(req("part", "interim"), live())).resolves.toBe("bergamot:part");
    await expect(router.translate(req("whole"), live())).resolves.toBe("llm:whole");
    expect(ctl.llm.calls.map((c) => c.request.text)).toEqual(["whole"]);
  });

  it("uses a chosen interim engine and respects the slices' interim switches", async () => {
    const chosen = setup((s) => (s.auto.interimEngine = B.nllb));
    await chosen.router.start(chosen.state, live());
    expect(chosen.ctl.nllb.starts).toBe(1); // started with the chain anyway
    await expect(chosen.router.translate(req("p", "interim"), live())).resolves.toBe("nllb:p");

    const noInterim = setup((s) => {
      s.bergamot.interim = false;
      s.llm.interim = false;
    });
    await noInterim.router.start(noInterim.state, live());
    // nllb.interim is off by default too: no engine may take interims.
    await expect(noInterim.router.translate(req("p", "interim"), live())).rejects.toThrow("transl.router_error_no_engine");
  });

  it("does not let interim timeouts in a busy server's queue lock the engine out of finals (no dropped finals)", async () => {
    vi.useFakeTimers();
    const { router, ctl, state } = setup((s) => (s.llm.interim = true));
    // The zero-setup install: only the local LLM is present.
    ctl.bergamot.start = () => Promise.reject(new TranslationError("no helper", { missing: ["runtime/bergamot"] }));
    ctl.nllb.start = () => Promise.reject(new TranslationError("no model", { missing: ["mt/nllb"] }));
    await router.start(state, live());
    ctl.llm.translate = (r) => (r.kind === "interim" ? never() : Promise.resolve(`llm:${r.text}`));
    for (const text of ["p1", "p2", "p3"]) {
      const p = router.translate(req(text, "interim"), live()).catch((e: unknown) => e);
      await vi.advanceTimersByTimeAsync(1000);
      expect(await p).toBeInstanceOf(Error);
    }
    expect(ctl.llm.calls).toHaveLength(2); // the interim breaker opened after two timeouts
    await expect(router.translate(req("final line"), live())).resolves.toBe("llm:final line");
  });

  it("gives an interim at most two attempts", async () => {
    const { router, ctl, state } = setup((s) => {
      s.auto.chain = [B.llm, B.bergamot, B.nllb];
      s.nllb.interim = true;
    });
    await router.start(state, live());
    ctl.bergamot.translate = fail("x");
    ctl.llm.translate = fail("y");
    await expect(router.translate(req("p", "interim"), live())).rejects.toThrow("transl.router_error_all_failed");
    expect(ctl.nllb.calls).toHaveLength(0);
  });
});

describe("routeForPair", () => {
  const autoOf = (patch: (a: Translation_State["auto"]) => void = () => {}) => {
    const a = Service_Translation_Schema.parse({}).auto;
    patch(a);
    return a;
  };

  it("takes the built-in CPU route for a European pair, with the beam it asks of NLLB", () => {
    expect(routeForPair(autoOf(), "en", "fr", "cpu")).toEqual({ chain: [B.nllb, B.llm, B.bergamot], nllbBeam: 4, nllbBudgetFactor: 1.5 });
    expect(chainForPair(autoOf(), "pt-BR", "en", "cpu")).toEqual([B.nllb, B.llm, B.bergamot]);
  });

  it("keeps the chain on the default tier and for pairs without a route", () => {
    expect(routeForPair(autoOf(), "en", "fr")).toEqual({ chain: [B.llm, B.bergamot, B.nllb] });
    expect(routeForPair(autoOf(), "en", "fr", "default")).toEqual({ chain: [B.llm, B.bergamot, B.nllb] });
    expect(routeForPair(autoOf(), "en", "ja", "cpu")).toEqual({ chain: [B.llm, B.bergamot, B.nllb] });
    expect(routeForPair(autoOf(), "", "fr", "cpu")).toEqual({ chain: [B.llm, B.bergamot, B.nllb] }); // a source left on detect
    expect(routeForPair(autoOf(), "en", "de", "cpu")).toEqual({ chain: [B.llm, B.bergamot, B.nllb] }); // a tie
    expect(routeForPair(autoOf(), "it", "en", "cpu")).toEqual({ chain: [B.llm, B.bergamot, B.nllb] }); // a tie
  });

  it("keeps the chain for a line with glossary terms or background text on the CPU tier", () => {
    expect(routeForPair(autoOf(), "en", "fr", "cpu", { glossary: true })).toEqual({ chain: [B.llm, B.bergamot, B.nllb] });
    expect(routeForPair(autoOf(), "en", "fr", "cpu", { background: true })).toEqual({ chain: [B.llm, B.bergamot, B.nllb] });
    expect(routeForPair(autoOf(), "en", "fr", "cpu", { glossary: false, background: false }).nllbBeam).toBe(4);
  });

  it("lets a pair override of the user win, and a chain the user edited stand", () => {
    const own = autoOf((a) => (a.pairOverrides = [{ from: "", to: "fr", chain: [B.deepl] }]));
    expect(routeForPair(own, "en", "fr", "cpu")).toEqual({ chain: [B.deepl] });
    expect(routeForPair(own, "en", "es", "cpu").chain).toEqual([B.nllb, B.llm, B.bergamot]); // other pairs keep the route
    const any = autoOf((a) => (a.pairOverrides = [{ from: "", to: "", chain: [B.llm] }]));
    expect(routeForPair(any, "en", "fr", "cpu")).toEqual({ chain: [B.llm] });
    const edited = autoOf((a) => (a.chain = [B.deepl, B.llm]));
    expect(routeForPair(edited, "en", "fr", "cpu")).toEqual({ chain: [B.deepl, B.llm] });
  });
});

describe("Translation_RouterService built-in CPU-tier routes", () => {
  const pair = (from: string, to: string) => (s: Translation_State) => {
    s.sourceLanguage = from;
    s.targetLanguage = to;
  };

  it("sends a European pair to NLLB first, asking for beam 4, while the translator runs on the processor", async () => {
    const { router, ctl, state, onCpu } = setup(pair("en", "fr"));
    onCpu.value = true;
    await router.start(state, live());
    const used: string[] = [];
    await expect(router.translate({ ...req("hello"), onEngine: (id) => used.push(id) }, live())).resolves.toBe("nllb:hello");
    expect(used).toEqual(["nllb"]);
    expect(ctl.nllb.calls[0].request).toMatchObject({ from: "en", to: "fr", beam: 4 });
    expect(ctl.llm.calls).toHaveLength(0);
  });

  it("gives NLLB's beam-4 attempt on a final half again the budget, and an interim or any other member the plain one", async () => {
    vi.useFakeTimers();
    const { router, ctl, state, onCpu } = setup(pair("en", "fr"));
    onCpu.value = true;
    await router.start(state, live());
    ctl.nllb.translate = never;
    const p = router.translate(req("hello"), live());
    await vi.advanceTimersByTimeAsync(1000); // attemptTimeoutMs: the default tier's llm would be given up here
    expect(ctl.llm.calls).toHaveLength(0);
    await vi.advanceTimersByTimeAsync(500);
    await expect(p).resolves.toBe("llm:hello");
    // The long final scales like any other attempt: 600 characters get 5 s, NLLB 7.5 s.
    const long = router.translate(req("x".repeat(600)), live());
    await vi.advanceTimersByTimeAsync(7499);
    expect(ctl.llm.calls).toHaveLength(1);
    await vi.advanceTimersByTimeAsync(1);
    await expect(long).resolves.toBe(`llm:${"x".repeat(600)}`);
  });

  it("starts the same members as the default chain, so nothing new loads or downloads", async () => {
    const { router, ctl, state, onCpu } = setup(pair("en", "fr"));
    onCpu.value = true;
    await router.start(state, live());
    expect([ctl.llm.starts, ctl.bergamot.starts, ctl.nllb.starts, ctl.deepl.starts]).toEqual([1, 1, 1, 0]);
  });

  it("changes nothing off the processor: the AI model first, no beam asked", async () => {
    const { router, ctl, state, onCpu } = setup(pair("en", "fr"));
    onCpu.value = false;
    await router.start(state, live());
    await expect(router.translate(req("hello"), live())).resolves.toBe("llm:hello");
    expect(ctl.llm.calls[0].request.beam).toBeUndefined();
    expect(ctl.nllb.calls).toHaveLength(0);
  });

  it("also applies with Source language left on automatic: the language the pipeline resolved for the line decides", async () => {
    // The pipeline fills request.from from the setting, else the language the speech engine reported for the line, else a
    // guess from the text; the router sees that language, not the (empty) setting.
    const { router, ctl, state, onCpu } = setup(pair("", "fr"));
    onCpu.value = true;
    await router.start(state, live());
    await expect(router.translate({ ...req("hello"), from: "en" }, live())).resolves.toBe("nllb:hello");
    expect(ctl.nllb.calls[0].request).toMatchObject({ from: "en", to: "fr", beam: 4 });
    // Only when no language is known at all does the chain stand.
    await expect(router.translate(req("hello again"), live())).resolves.toBe("llm:hello again");
  });

  it("keeps the audience's pairs on the AI model on the processor", async () => {
    const { router, ctl, state, onCpu } = setup(pair("en", "ja"));
    onCpu.value = true;
    await router.start(state, live());
    await expect(router.translate(req("hello"), live())).resolves.toBe("llm:hello");
    await expect(router.translate({ ...req("hello"), to: "ko" }, live())).resolves.toBe("llm:hello");
    await expect(router.translate({ ...req("hello"), from: "zh", to: "en" }, live())).resolves.toBe("llm:hello");
    expect(ctl.nllb.calls).toHaveLength(0);
  });

  it("keeps the pairs NLLB only tied on (German, Italian) on the AI model on the processor", async () => {
    const { router, ctl, state, onCpu } = setup(pair("en", "de"));
    onCpu.value = true;
    await router.start(state, live());
    await expect(router.translate(req("hello"), live())).resolves.toBe("llm:hello");
    await expect(router.translate({ ...req("ciao"), from: "it", to: "en" }, live())).resolves.toBe("llm:ciao");
    expect(ctl.nllb.calls).toHaveLength(0);
  });

  it("follows the tier line by line (the server's variant becomes known after the first lines)", async () => {
    const { router, ctl, state, onCpu } = setup(pair("en", "es"));
    await router.start(state, live());
    await expect(router.translate(req("a"), live())).resolves.toBe("llm:a");
    onCpu.value = true;
    await expect(router.translate(req("b"), live())).resolves.toBe("nllb:b");
    onCpu.value = false;
    await expect(router.translate(req("c"), live())).resolves.toBe("llm:c");
    expect(ctl.nllb.calls.map((c) => c.request.beam)).toEqual([4]);
  });

  it("falls through to the AI model when the NLLB model is not installed, without a fallback warning", async () => {
    const { router, ctl, statuses, state, onCpu } = setup(pair("en", "pt"));
    onCpu.value = true;
    ctl.nllb.start = () => Promise.reject(new TranslationError("not installed", { missing: ["mt/nllb-200-distilled-600m-int8"] }));
    await router.start(state, live());
    const used: string[] = [];
    await expect(router.translate({ ...req("hello"), onEngine: (id) => used.push(id) }, live())).resolves.toBe("llm:hello");
    expect(used).toEqual(["llm"]);
    expect(statuses.some((s) => s.level === "warning" && s.text === "transl.router_status_fallback")).toBe(false);
  });

  it("does not flash a fallback warning when the retry of the NLLB start (every minute) finds the model still missing", async () => {
    vi.useFakeTimers();
    const { router, ctl, statuses, state, onCpu } = setup(pair("en", "pt"));
    onCpu.value = true;
    ctl.nllb.start = () => Promise.reject(new TranslationError("not installed", { missing: ["mt/nllb-200-distilled-600m-int8"] }));
    await router.start(state, live());
    await expect(router.translate(req("a"), live())).resolves.toBe("llm:a");
    expect(ctl.nllb.starts).toBe(1);
    // A minute later the member is due another start; it is retried first (the model may be installed by now) ...
    vi.advanceTimersByTime(ROUTER_START_RETRY_MS);
    const used: string[] = [];
    await expect(router.translate({ ...req("b"), onEngine: (id) => used.push(id) }, live())).resolves.toBe("llm:b");
    expect(ctl.nllb.starts).toBe(2);
    expect(used).toEqual(["llm"]);
    // ... and a model that is still not there is nobody's "preferred" engine: no "using X instead of Y" for a PC without it.
    expect(statuses.some((s) => s.text === "transl.router_status_fallback")).toBe(false);
    vi.advanceTimersByTime(ROUTER_START_RETRY_MS);
    await expect(router.translate(req("c"), live())).resolves.toBe("llm:c");
    expect(statuses.some((s) => s.text === "transl.router_status_fallback")).toBe(false);
  });

  it("uses NLLB as soon as a retry finds the model installed", async () => {
    vi.useFakeTimers();
    const { router, ctl, statuses, state, onCpu } = setup(pair("en", "pt"));
    onCpu.value = true;
    ctl.nllb.start = () => Promise.reject(new TranslationError("not installed", { missing: ["mt/nllb-200-distilled-600m-int8"] }));
    await router.start(state, live());
    await expect(router.translate(req("a"), live())).resolves.toBe("llm:a");
    ctl.nllb.start = undefined; // downloaded meanwhile
    vi.advanceTimersByTime(ROUTER_START_RETRY_MS);
    await expect(router.translate(req("b"), live())).resolves.toBe("nllb:b");
    expect(statuses.some((s) => s.text === "transl.router_status_fallback")).toBe(false);
  });

  it("still warns when NLLB, which is installed, fails a line after a failed start that was not about a missing model", async () => {
    vi.useFakeTimers();
    const { router, ctl, statuses, state, onCpu } = setup(pair("en", "pt"));
    onCpu.value = true;
    ctl.nllb.start = () => Promise.reject(new TranslationError("session create failed"));
    await router.start(state, live());
    await expect(router.translate(req("a"), live())).resolves.toBe("llm:a");
    vi.advanceTimersByTime(ROUTER_START_RETRY_MS);
    await expect(router.translate(req("b"), live())).resolves.toBe("llm:b");
    expect(statuses.at(-1)).toEqual({ text: "transl.router_status_fallback", level: "warning" });
  });

  it("keeps the AI model first for a line with a glossary term or the background text, which NLLB would drop", async () => {
    const { router, ctl, state, onCpu } = setup(pair("en", "fr"));
    onCpu.value = true;
    await router.start(state, live());
    await expect(router.translate({ ...req("a"), glossary: [{ source: "Mewnah", target: "Mewnah" }] }, live())).resolves.toBe("llm:a");
    await expect(router.translate({ ...req("b"), background: "A VRChat stream about horror games." }, live())).resolves.toBe("llm:b");
    expect(ctl.nllb.calls).toHaveLength(0);
    // The same stream's other lines still take the route, and blank background or an empty glossary change nothing.
    await expect(router.translate(req("c"), live())).resolves.toBe("nllb:c");
    await expect(router.translate({ ...req("d"), background: "   ", glossary: [] }, live())).resolves.toBe("nllb:d");
    expect(ctl.llm.calls.every((c) => c.request.beam === undefined)).toBe(true);
  });

  it("does not give up the route for the rolling context alone (NLLB translates each line alone; the docs say so)", async () => {
    const { router, state, onCpu } = setup(pair("en", "fr"));
    onCpu.value = true;
    await router.start(state, live());
    await expect(router.translate({ ...req("e"), context: [{ source: "hi", target: "salut" }] }, live())).resolves.toBe("nllb:e");
  });

  it("falls through to the AI model when NLLB fails a line, and shows the fallback warning", async () => {
    const { router, ctl, statuses, state, onCpu } = setup(pair("en", "nl"));
    onCpu.value = true;
    await router.start(state, live());
    ctl.nllb.translate = fail("decode failed");
    await expect(router.translate(req("hello"), live())).resolves.toBe("llm:hello");
    expect(statuses.at(-1)).toEqual({ text: "transl.router_status_fallback", level: "warning" });
  });

  it("lets a pair rule of the user turn the route off", async () => {
    const { router, ctl, state, onCpu } = setup((s) => {
      pair("en", "nl")(s);
      s.auto.pairOverrides = [{ from: "en", to: "nl", chain: [B.llm, B.nllb] }];
    });
    onCpu.value = true;
    await router.start(state, live());
    await expect(router.translate(req("hello"), live())).resolves.toBe("llm:hello");
    expect(ctl.nllb.calls).toHaveLength(0);
  });

  it("answers supports() from the route's members", async () => {
    const { router, ctl, state, onCpu } = setup(pair("en", "fr"));
    onCpu.value = true;
    ctl.nllb.supports = () => false;
    ctl.llm.supports = () => false;
    ctl.bergamot.supports = () => false;
    await router.start(state, live());
    await vi.waitFor(() => expect(router.supports("en", "fr")).toBe(false));
  });
});
