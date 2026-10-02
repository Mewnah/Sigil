import { beforeAll, beforeEach, describe, expect, it, vi } from "vitest";
import {
  OthersEventSource,
  ServiceNetworkState,
  TextEventType,
  type RustSpeechFinal,
  type RustSpeechPartial,
  type RustSpeechPipelineConfig,
  type RustSpeechPipelineStarted,
  type RustSpeechRoutePlan,
  type RustSpeechStatus,
  type TextEvent,
} from "@/types";

vi.mock("@tauri-apps/api/core", () => ({ invoke: vi.fn(async () => ""), isTauri: () => false }));
vi.mock("@tauri-apps/api/event", () => ({ listen: vi.fn(async () => () => {}) }));
vi.mock("@tauri-apps/plugin-dialog", () => ({ ask: vi.fn(async () => false), confirm: vi.fn(async () => false), message: vi.fn(async () => "Cancel") }));
vi.mock("react-toastify", () => ({
  toast: Object.assign(vi.fn(), { error: vi.fn(), info: vi.fn(), warning: vi.fn(), success: vi.fn(), dismiss: vi.fn() }),
}));
vi.mock("@/core/services/systemLog", () => ({ pushSystemLog: vi.fn() }));
vi.mock("@/utils/devLog", () => ({ devLog: () => {} }));

import i18n from "i18next";
import { initSttTestI18n, translated } from "@/core/services/stt/__tests__/sttTestI18n";
import { RESTART_DELAYS_MS, type SegmentClock } from "@/core/services/stt/services/local";
import { PROCESS_NONCE } from "@/utils/processNonce";
import {
  buildOthersPipelineConfig,
  CAPTURE_RETRY_DELAYS_MS,
  OTHERS_LATENCY_BUDGET_MS,
  OTHERS_PIPELINE_ID,
  othersRouteRequest,
  OthersSpeechDriver,
  WATCH_INTERVAL_MS,
  type OthersDriverDeps,
} from "../driver";
import { OthersPublisher, othersTextMeta, OTHERS_TRANSLATION_STREAM } from "../publish";
import { Service_Others_Schema, type Others_State } from "../schema";
import { isProcessSource, parsePlan, processExe, sourceLabel, type OthersSourcePlan } from "../source";
import { othersTranslationPair } from "../translation";

// ---------------------------------------------------------------------------------------------------------------
// Fakes

class FakeClock implements SegmentClock {
  t = 1_000_000;
  #timers: { at: number; fn: () => void; id: number }[] = [];
  #next = 1;
  now() {
    return this.t;
  }
  setTimeout(fn: () => void, ms: number) {
    const id = this.#next++;
    this.#timers.push({ at: this.t + ms, fn, id });
    return id;
  }
  clearTimeout(handle: unknown) {
    this.#timers = this.#timers.filter((x) => x.id !== handle);
  }
  async advance(ms: number) {
    const end = this.t + ms;
    for (;;) {
      const due = this.#timers.filter((x) => x.at <= end).sort((a, b) => a.at - b.at)[0];
      if (!due) break;
      this.#timers = this.#timers.filter((x) => x !== due);
      this.t = due.at;
      due.fn();
      await settle();
    }
    this.t = end;
    await settle();
  }
}

const settle = async () => {
  for (let i = 0; i < 10; i++) await new Promise<void>((r) => setTimeout(r, 0));
};

/** A session shutdown in Rust: the command returns a little after `stopped` went out. */
const shutdown = () => new Promise<void>((r) => setTimeout(r, 0));

const STREAMING_PLAN: RustSpeechRoutePlan = {
  language: "auto",
  tier: "cpu",
  streaming: { engine: "sherpa-online", model: "stt/nemotron-3.5-streaming-560ms-int8" },
  final: null,
  fallback: null,
  languageId: null,
  models: ["stt/nemotron-3.5-streaming-560ms-int8"],
  missing: [],
  vramMb: 0,
  notes: [],
};

/**
 * A small model of the Rust world: the planner (`speech::session`), the pipeline registry and the events. Like Rust's
 * `stop_session_at` / a replacing `start_session`, ending the live session emits `stopped` for its epoch before the
 * command returns (the session shutdown takes a while).
 */
class World {
  supported = true;
  running = new Set<string>(["VRChat.exe"]);
  missing: string[] = [];
  epoch = 0;
  /** Epoch of the session live on the `loopback` key (null = none). */
  live: number | null = null;
  calls: { cmd: string; args?: Record<string, unknown> }[] = [];
  starts: RustSpeechPipelineConfig[] = [];
  /** Runs inside the next `start_pipeline` (may throw to fail it). */
  beforeNextStart: (() => void) | null = null;
  /** Every program capture fails to open with this error (null = they open). */
  captureError: string | null = null;
  #handlers = new Map<string, Set<(payload: unknown) => void>>();
  clock = new FakeClock();
  logs: string[] = [];

  plan(source: string, fallback: boolean): OthersSourcePlan {
    const s = source.trim();
    if (s.toLowerCase().startsWith("process:")) {
      const exe = s.slice("process:".length).trim();
      const requested = `process:${exe}`;
      if (!this.supported)
        return { requested, source: fallback ? "loopback:default" : null, fallback: "unsupported", watch: null, running: false };
      const running = [...this.running].find((r) => r.toLowerCase() === exe.toLowerCase());
      return running
        ? { requested, source: `process:${running}`, fallback: null, watch: exe, running: true }
        : { requested, source: fallback ? "loopback:default" : null, fallback: "not_running", watch: exe, running: false };
    }
    return { requested: s, source: s, fallback: null, watch: null, running: false };
  }

  deps(): Partial<OthersDriverDeps> {
    return {
      available: () => true,
      clock: this.clock,
      log: (line) => void this.logs.push(line),
      title: (id) => `T(${id})`,
      listen: async <T,>(event: string, handler: (payload: T) => void) => {
        const set = this.#handlers.get(event) ?? new Set();
        set.add(handler as (payload: unknown) => void);
        this.#handlers.set(event, set);
        return () => set.delete(handler as (payload: unknown) => void);
      },
      invoke: async <T,>(cmd: string, args?: Record<string, unknown>): Promise<T> => {
        this.calls.push({ cmd, args });
        switch (cmd) {
          case "plugin:speech|preview_route":
            return { ...STREAMING_PLAN, missing: [...this.missing] } as T;
          case "plugin:speech|get_pipeline_status":
            return (this.live !== null ? [{ pipelineId: OTHERS_PIPELINE_ID, epoch: this.live, state: "listening" }] : []) as T;
          case "plugin:audio|list_loopback_processes":
            return {
              supported: this.supported,
              processes: [],
              windowsBuild: this.supported ? 26_200 : 19_045,
              plan: this.plan(String(args?.source ?? ""), args?.fallback !== false),
            } as T;
          case "plugin:speech|start_pipeline": {
            const config = args!.config as RustSpeechPipelineConfig;
            this.starts.push(config);
            // The new start replaces the key's session first.
            this.#endLive();
            await shutdown();
            const hook = this.beforeNextStart;
            this.beforeNextStart = null;
            hook?.();
            if (this.captureError !== null && config.source?.startsWith("process:")) throw this.captureError;
            this.epoch += 1;
            this.live = this.epoch;
            return { pipelineId: config.pipelineId!, epoch: this.epoch, plan: STREAMING_PLAN } satisfies RustSpeechPipelineStarted as T;
          }
          case "plugin:speech|stop_pipeline": {
            // An epoch-scoped stop ends exactly that session (nothing when another one is live).
            const epoch = args?.epoch;
            if (typeof epoch !== "number" || epoch === this.live) this.#endLive();
            await shutdown();
            return undefined as T;
          }
          default:
            throw new Error(`unexpected ${cmd}`);
        }
      },
    };
  }

  /** Rust `end_with_stopped`: the live session emits `stopped` and is gone. */
  #endLive() {
    if (this.live === null) return;
    const epoch = this.live;
    this.live = null;
    this.status(epoch, "stopped", "stopped");
  }

  emit(event: string, payload: unknown) {
    for (const h of [...(this.#handlers.get(event) ?? [])]) h(payload);
  }

  partial(epoch: number, segmentId: number, revision: number, text: string, pipelineId = OTHERS_PIPELINE_ID) {
    const ev: RustSpeechPartial = { pipelineId, epoch, segmentId, revision, text, lang: "ja" };
    this.emit("speech:partial", ev);
  }

  final(epoch: number, segmentId: number, revision: number, text: string, pipelineId = OTHERS_PIPELINE_ID) {
    const ev: RustSpeechFinal = {
      pipelineId,
      epoch,
      segmentId,
      revision,
      text,
      lang: "ja",
      tags: ["emotion:happy"],
      pass: "first",
      engine: "sherpa-online",
      model: "m",
      startMs: 0,
      endMs: 1200,
      forced: false,
      latencyMs: 80,
      turnEnd: null,
    };
    this.emit("speech:final", ev);
  }

  status(epoch: number, state: RustSpeechStatus["state"], code: string, detail = "") {
    const ev: RustSpeechStatus = {
      pipelineId: OTHERS_PIPELINE_ID,
      epoch,
      state,
      code,
      detail,
      progress: null,
      level: state === "error" ? "error" : "info",
      modelId: null,
      engine: null,
      device: null,
    };
    this.emit("speech:status", ev);
  }

  /** Pipeline keys every pipeline command named. */
  keys(): string[] {
    return this.calls
      .filter((c) => c.cmd.startsWith("plugin:speech|") && c.cmd !== "plugin:speech|preview_route" && c.cmd !== "plugin:speech|get_pipeline_status")
      .map((c) => String((c.args?.config as RustSpeechPipelineConfig | undefined)?.pipelineId ?? c.args?.pipelineId));
  }
}

interface Out {
  interims: [string, string | undefined][];
  finals: [string, string | undefined][];
  stops: (string | undefined)[];
}

function make(world: World, patch: Partial<Others_State> = {}) {
  const out: Out = { interims: [], finals: [], stops: [] };
  const driver = new OthersSpeechDriver(
    {
      interim: (text, meta) => void out.interims.push([text, meta.segmentId]),
      final: (text, meta) => void out.finals.push([text, meta.segmentId]),
    },
    (error) => void out.stops.push(error),
    world.deps(),
  );
  const settings: Others_State = { ...Service_Others_Schema.parse({}), ...patch };
  return { driver, out, settings };
}

const seg = (epoch: number, n: number) => `${OTHERS_PIPELINE_ID}-${PROCESS_NONCE}:${epoch}:${n}`;

beforeAll(async () => {
  await initSttTestI18n();
});

// ---------------------------------------------------------------------------------------------------------------

describe("Others settings", () => {
  it("default off, VRChat with the default-output fallback", () => {
    const s = Service_Others_Schema.parse({});
    expect(s).toMatchObject({
      autoStart: false,
      source: "process:VRChat.exe",
      language: "auto",
      tier: "auto",
      interim: true,
      translate: false,
      translateTo: "",
      fallback: true,
      twoPass: true,
      musicGate: false,
    });
    // Profiles saved by the P6-S0 scaffold keep their values and gain the new keys.
    expect(Service_Others_Schema.parse({ source: "loopback:Speakers", tier: "bogus", fallback: "nope" })).toMatchObject({
      source: "loopback:Speakers",
      tier: "auto",
    });
  });

  it("builds the loopback pipeline config, never the microphone's", () => {
    const s = Service_Others_Schema.parse({ language: "JA", tier: "gpu", twoPass: false, musicGate: true });
    const request = othersRouteRequest(s);
    expect(request).toEqual({ language: "ja", tier: "gpu", twoPass: false });
    expect(buildOthersPipelineConfig(s, request, "process:VRChat.exe")).toEqual({
      pipelineId: "loopback",
      source: "process:VRChat.exe",
      vad: {},
      language: "ja",
      route: { language: "ja", tier: "gpu", twoPass: false },
      smartTurn: false,
      hotwords: [],
      hallucinationFilter: true,
      latencyBudgetMs: OTHERS_LATENCY_BUDGET_MS,
      musicGate: true,
    });
    expect(othersRouteRequest(Service_Others_Schema.parse({})).language).toBe("auto");
  });
});

describe("source helpers", () => {
  it("recognises program sources and labels every kind", () => {
    expect(isProcessSource("process:VRChat.exe")).toBe(true);
    expect(isProcessSource(" PROCESS:x")).toBe(true);
    expect(isProcessSource("loopback:default")).toBe(false);
    expect(processExe("process: VRChat.exe ")).toBe("VRChat.exe");
    const labels = {
      program: (e: string) => `P(${e})`,
      loopbackDefault: "DEF",
      loopback: (n: string) => `L(${n})`,
      device: (n: string) => `D(${n})`,
    };
    expect(sourceLabel("process:VRChat.exe", labels)).toBe("P(VRChat.exe)");
    expect(sourceLabel("loopback:default", labels)).toBe("DEF");
    expect(sourceLabel("loopback:Speakers", labels)).toBe("L(Speakers)");
    expect(sourceLabel("CABLE Output", labels)).toBe("D(CABLE Output)");
    expect(sourceLabel("", labels)).toBe("DEF");
  });

  it("reads plans defensively", () => {
    expect(parsePlan(null)).toBeNull();
    expect(parsePlan({ requested: "process:a.exe", source: "", fallback: "weird", watch: 3, running: "yes" })).toEqual({
      requested: "process:a.exe",
      source: null,
      fallback: null,
      watch: null,
      running: false,
    });
  });
});

describe("reverse translation pair", () => {
  const base = { othersLanguage: "auto", translateTo: "", mainSource: "", sttLanguage: "auto", appLanguage: "en" };

  it("translates from their language into the user's", () => {
    expect(othersTranslationPair({ ...base, othersLanguage: "ja", mainSource: "en" })).toEqual({ from: "ja", to: "en" });
    // `auto`: detect per line (never the user's own language as the source).
    expect(othersTranslationPair({ ...base, mainSource: "de" })).toEqual({ from: "", to: "de" });
  });

  it("picks the user's language from translateTo, the main source, STT, then the app", () => {
    expect(othersTranslationPair({ ...base, translateTo: "fr", mainSource: "de" }).to).toBe("fr");
    expect(othersTranslationPair({ ...base, sttLanguage: "ko" }).to).toBe("ko");
    expect(othersTranslationPair({ ...base, appLanguage: "zh-CN" }).to).toBe("zh");
    expect(othersTranslationPair({ ...base, appLanguage: "" }).to).toBe("en");
  });
});

describe("OthersPublisher", () => {
  function harness(interim = true) {
    const published: { topic: string; event: Partial<TextEvent> }[] = [];
    const translated: { stream: string; event: TextEvent }[] = [];
    const pub = new OthersPublisher({
      publish: (topic, event) => void published.push({ topic, event }),
      translate: (stream, event) => void translated.push({ stream, event }),
      interimEnabled: () => interim,
    });
    return { pub, published, translated };
  }

  it("publishes on others.stt and feeds the reverse translation", () => {
    const h = harness();
    h.pub.interim(" こんにちは ", { segmentId: "loopback-x:1:1", revision: 1, lang: "ja" });
    h.pub.final("こんにちは。", { segmentId: "loopback-x:1:1", revision: 2, lang: "ja", tags: ["emotion:happy"] });
    expect(h.published.map((p) => p.topic)).toEqual([OthersEventSource.stt, OthersEventSource.stt]);
    expect(h.published[0].event).toEqual({ type: TextEventType.interim, value: "こんにちは", segmentId: "loopback-x:1:1", revision: 1, lang: "ja" });
    expect(h.published[1].event).toMatchObject({ type: TextEventType.final, tags: ["emotion:happy"] });
    expect(h.translated.map((t) => [t.stream, t.event.type, t.event.value])).toEqual([
      [OTHERS_TRANSLATION_STREAM, TextEventType.interim, "こんにちは"],
      [OTHERS_TRANSLATION_STREAM, TextEventType.final, "こんにちは。"],
    ]);
    // Separate objects: a caption subscriber cannot change what the translation reads.
    expect(h.published[1].event).not.toBe(h.translated[1].event);
  });

  it("drops interims when live text is off, empty text, and malformed metadata", () => {
    const h = harness(false);
    h.pub.interim("partial", { segmentId: "a:1:1", revision: 1 });
    h.pub.final("   ", {});
    h.pub.final("done", { segmentId: "bad id with spaces", revision: -1, lang: "JA!", tags: ["<b>"] });
    expect(h.published).toEqual([{ topic: OthersEventSource.stt, event: { type: TextEventType.final, value: "done" } }]);
    expect(h.translated).toHaveLength(1);
    expect(othersTextMeta({ turnEnd: true, tags: [] })).toEqual({ turnEnd: true });
  });
});

describe("OthersSpeechDriver", () => {
  let world: World;
  beforeEach(() => {
    world = new World();
  });

  it("captures the running program on the loopback key and publishes its text", async () => {
    const { driver, out, settings } = make(world);
    await driver.start(settings);
    expect(world.starts).toHaveLength(1);
    expect(world.starts[0]).toMatchObject({ pipelineId: "loopback", source: "process:VRChat.exe", musicGate: false });
    expect(driver.state).toMatchObject({ status: ServiceNetworkState.connected, phase: "running", source: "process:VRChat.exe", error: "" });
    expect(driver.state.plan?.running).toBe(true);

    world.partial(1, 1, 1, "こんに");
    world.final(1, 1, 2, "こんにちは");
    // The user's own microphone pipeline (and other epochs) are never taken.
    world.partial(1, 9, 1, "my own words", "mic");
    world.final(1, 9, 2, "my own words", "mic");
    world.final(7, 1, 2, "stale epoch");
    expect(out.interims).toEqual([["こんに", seg(1, 1)]]);
    expect(out.finals).toEqual([["こんにちは", seg(1, 1)]]);
    // Every pipeline command names the loopback key; nothing touches "mic".
    expect(new Set(world.keys())).toEqual(new Set(["loopback"]));
    driver.dispose();
  });

  it("falls back to the default output, then follows the program as it starts and exits", async () => {
    world.running.clear();
    const { driver, out, settings } = make(world);
    await driver.start(settings);
    expect(world.starts.map((c) => c.source)).toEqual(["loopback:default"]);
    expect(driver.state).toMatchObject({ phase: "running", source: "loopback:default" });
    expect(driver.state.plan?.fallback).toBe("not_running");
    expect(driver.state.statusText).toBe("VRChat.exe isn't running, so the default output is captured for now.");

    world.partial(1, 1, 1, "music video words");
    world.running.add("VRChat.exe");
    await world.clock.advance(WATCH_INTERVAL_MS);
    expect(world.starts.map((c) => c.source)).toEqual(["loopback:default", "process:VRChat.exe"]);
    expect(driver.state).toMatchObject({ phase: "running", source: "process:VRChat.exe", statusText: "" });
    // The replaced session's late events are refused.
    world.final(1, 1, 2, "late final of the old source");
    world.final(2, 2, 2, "from VRChat");
    expect(out.finals).toEqual([["from VRChat", seg(2, 2)]]);

    world.running.clear();
    await world.clock.advance(WATCH_INTERVAL_MS);
    expect(world.starts.map((c) => c.source)).toEqual(["loopback:default", "process:VRChat.exe", "loopback:default"]);
    expect(driver.state.source).toBe("loopback:default");
    // Nothing changes while the plan stays the same.
    await world.clock.advance(WATCH_INTERVAL_MS * 3);
    expect(world.starts).toHaveLength(3);
    driver.dispose();
  });

  it("waits for the program with the fallback off", async () => {
    world.running.clear();
    const { driver, out, settings } = make(world, { fallback: false });
    await driver.start(settings);
    expect(world.starts).toHaveLength(0);
    expect(driver.state).toMatchObject({ status: ServiceNetworkState.connected, phase: "waiting", source: null });
    expect(driver.state.statusText).toBe("Waiting for VRChat.exe to start…");

    world.running.add("VRChat.exe");
    await world.clock.advance(WATCH_INTERVAL_MS);
    expect(world.starts.map((c) => c.source)).toEqual(["process:VRChat.exe"]);
    expect(driver.state.phase).toBe("running");

    // The game closes: the pipeline stops and the driver waits again (no device fallback). The stop's own `stopped`
    // event is not "another start replaced the session".
    world.running.clear();
    await world.clock.advance(WATCH_INTERVAL_MS);
    expect(driver.state).toMatchObject({ status: ServiceNetworkState.connected, phase: "waiting", source: null, error: "" });
    expect(driver.state.statusText).toBe("Waiting for VRChat.exe to start…");
    expect(world.calls.filter((c) => c.cmd === "plugin:speech|stop_pipeline").at(-1)?.args).toEqual({ pipelineId: "loopback", epoch: 1 });
    expect(world.live).toBeNull();
    expect(out.stops).toEqual([]);

    // And it comes back.
    world.running.add("VRChat.exe");
    await world.clock.advance(WATCH_INTERVAL_MS);
    expect(driver.state).toMatchObject({ phase: "running", source: "process:VRChat.exe" });
    expect(world.live).toBe(2);
    driver.dispose();
  });

  it("waits when the game exits while its capture runs (fallback off), never failing on the stop's own event", async () => {
    const { driver, out, settings } = make(world, { fallback: false });
    await driver.start(settings);
    expect(driver.state).toMatchObject({ phase: "running", source: "process:VRChat.exe" });
    world.partial(1, 1, 1, "see you");
    world.running.clear();
    await world.clock.advance(WATCH_INTERVAL_MS);
    expect(driver.state).toMatchObject({ phase: "waiting", source: null, error: "" });
    expect(out.stops).toEqual([]);
    expect(world.logs).not.toContain(i18n.t("others.error_stopped_elsewhere"));
    // The ended session's late events are refused.
    world.final(1, 1, 2, "late");
    expect(out.finals).toEqual([]);
    // The watch keeps going while waiting.
    world.running.add("VRChat.exe");
    await world.clock.advance(WATCH_INTERVAL_MS);
    expect(driver.state.phase).toBe("running");
    driver.dispose();
  });

  it("stops a session left running before the start when the first plan waits for the program", async () => {
    // An earlier webview (a reload) left epoch 7 running on the loopback key.
    world.epoch = 7;
    world.live = 7;
    world.running.clear();
    const { driver, out, settings } = make(world, { fallback: false });
    await driver.start(settings);
    expect(driver.state.phase).toBe("waiting");
    expect(world.calls.filter((c) => c.cmd === "plugin:speech|stop_pipeline").map((c) => c.args)).toEqual([
      { pipelineId: "loopback", epoch: 7 },
    ]);
    expect(world.live).toBeNull();
    world.final(7, 1, 2, "from the old session");
    expect(out.finals).toEqual([]);
    expect(out.stops).toEqual([]);
    driver.dispose();
  });

  it("a new start whose plan waits stops the session of the previous start", async () => {
    const { driver, out, settings } = make(world, { fallback: false });
    await driver.start(settings);
    expect(world.live).toBe(1);
    await driver.start({ ...settings, source: "process:NotRunning.exe" });
    expect(driver.state).toMatchObject({ phase: "waiting", source: null });
    expect(world.live).toBeNull();
    expect(world.calls.filter((c) => c.cmd === "plugin:speech|stop_pipeline").at(-1)?.args).toEqual({ pipelineId: "loopback", epoch: 1 });
    expect(out.stops).toEqual([]);
    driver.dispose();
  });

  it("captures the default output on Windows without per-process capture, or fails with the fallback off", async () => {
    world.supported = false;
    const a = make(world);
    await a.driver.start(a.settings);
    expect(world.starts.map((c) => c.source)).toEqual(["loopback:default"]);
    expect(a.driver.state.statusText).toBe("This Windows version can't capture one program, so the default output is captured.");
    await a.driver.stop();

    const b = make(world, { fallback: false });
    await b.driver.start(b.settings);
    expect(b.driver.state.phase).toBe("idle");
    expect(b.driver.state.error).toContain("can't capture one program");
    expect(b.out.stops).toEqual([b.driver.state.error]);
    expect(world.starts).toHaveLength(1);
  });

  it("never starts without its models and says which are missing", async () => {
    world.missing = ["stt/nemotron-3.5-streaming-560ms-int8"];
    const { driver, out, settings } = make(world);
    await driver.start(settings);
    expect(world.starts).toHaveLength(0);
    expect(driver.state.status).toBe(ServiceNetworkState.disconnected);
    expect(out.stops[0]).toBe("Download the speech models in the Other players panel first: T(stt/nemotron-3.5-streaming-560ms-int8)");
  });

  it("plans again when the program exits between the plan and the start", async () => {
    // The start fails because the program just exited, so the new plan is the default output.
    world.beforeNextStart = () => {
      world.running.clear();
      throw "Could not capture the program's sound: VRChat.exe is not running";
    };
    const { driver, settings } = make(world);
    await driver.start(settings);
    expect(world.starts.map((c) => c.source)).toEqual(["process:VRChat.exe", "loopback:default"]);
    expect(driver.state.phase).toBe("running");
    driver.dispose();
  });

  it("captures the default output while a running program's capture cannot be opened, and retries it with back-off", async () => {
    world.captureError = "Could not capture the program's sound: 0x80070005";
    const { driver, out, settings } = make(world);
    await driver.start(settings);
    expect(world.starts.map((c) => c.source)).toEqual(["process:VRChat.exe", "loopback:default"]);
    expect(driver.state).toMatchObject({ status: ServiceNetworkState.connected, phase: "running", source: "loopback:default", error: "" });
    expect(driver.state.plan).toMatchObject({ fallback: "capture_failed", watch: "VRChat.exe", running: true });
    const note = i18n.t("others.status_fallback_capture_failed", { program: "VRChat.exe" });
    expect(translated(note)).toBe(true);
    expect(driver.state).toMatchObject({ statusText: note, statusLevel: "warning" });
    expect(world.logs.some((l) => l.includes("0x80070005") && l.includes("30"))).toBe(true);
    expect(out.stops).toEqual([]);

    // No churn: the program is not tried again before the back-off ends.
    await world.clock.advance(CAPTURE_RETRY_DELAYS_MS[0] - WATCH_INTERVAL_MS);
    expect(world.starts).toHaveLength(2);
    // Then it is; failing again backs off longer.
    await world.clock.advance(WATCH_INTERVAL_MS);
    expect(world.starts.map((c) => c.source)).toEqual(["process:VRChat.exe", "loopback:default", "process:VRChat.exe", "loopback:default"]);
    expect(driver.state).toMatchObject({ phase: "running", source: "loopback:default" });
    await world.clock.advance(CAPTURE_RETRY_DELAYS_MS[1] - WATCH_INTERVAL_MS);
    expect(world.starts).toHaveLength(4);

    // The capture works now: the program is captured at the next retry.
    world.captureError = null;
    await world.clock.advance(WATCH_INTERVAL_MS);
    expect(world.starts).toHaveLength(5);
    expect(driver.state).toMatchObject({ phase: "running", source: "process:VRChat.exe", statusText: "" });
    expect(driver.state.plan?.fallback).toBeNull();
    expect(out.stops).toEqual([]);
    driver.dispose();
    // The fake clock walks about 95 s of timers with real macrotask settles (~3.5 s of wall time), which
    // exceeded vitest's 5 s default while cargo builds loaded the machine.
  }, 20_000);

  it("tries a program again at once when it restarts after a failed capture", async () => {
    world.captureError = "Timed out starting the program capture";
    const { driver, settings } = make(world);
    await driver.start(settings);
    expect(driver.state.source).toBe("loopback:default");
    // The game exits and starts again: no back-off for the new instance.
    world.running.clear();
    await world.clock.advance(WATCH_INTERVAL_MS);
    expect(driver.state.plan?.fallback).toBe("not_running");
    expect(driver.state.statusText).toBe("VRChat.exe isn't running, so the default output is captured for now.");
    expect(world.starts).toHaveLength(2);
    world.captureError = null;
    world.running.add("VRChat.exe");
    await world.clock.advance(WATCH_INTERVAL_MS);
    expect(driver.state).toMatchObject({ phase: "running", source: "process:VRChat.exe" });
    driver.dispose();
  });

  it("fails when a running program cannot be captured and the fallback is off, or the models are the problem", async () => {
    world.captureError = "Could not capture the program's sound: 0x80070005";
    const a = make(world, { fallback: false });
    await a.driver.start(a.settings);
    expect(world.starts.map((c) => c.source)).toEqual(["process:VRChat.exe"]);
    expect(a.driver.state.phase).toBe("idle");
    expect(a.driver.state.error).toContain("0x80070005");
    expect(a.out.stops).toEqual([a.driver.state.error]);

    // A missing model is not a capture problem: the default output would fail the same way.
    world.captureError = "not_installed: stt/nemotron-3.5-streaming-560ms-int8";
    world.starts = [];
    const b = make(world);
    await b.driver.start(b.settings);
    expect(world.starts.map((c) => c.source)).toEqual(["process:VRChat.exe"]);
    expect(b.driver.state.phase).toBe("idle");
    expect(b.driver.state.plan?.fallback).toBeNull();
  });

  it("restarts after a pipeline error and plans the source again", async () => {
    const { driver, out, settings } = make(world);
    await driver.start(settings);
    world.partial(1, 1, 1, "half a sent");
    world.status(1, "error", "capture_error", "device lost");
    expect(driver.state.phase).toBe("restarting");
    expect(driver.state.status).toBe(ServiceNetworkState.connecting);
    world.running.clear();
    await world.clock.advance(RESTART_DELAYS_MS[0]);
    expect(world.starts.map((c) => c.source)).toEqual(["process:VRChat.exe", "loopback:default"]);
    expect(driver.state).toMatchObject({ phase: "running", restarts: 1 });
    expect(out.stops).toEqual([]);
    driver.dispose();
  });

  it("stops on errors a restart cannot fix", async () => {
    const { driver, out, settings } = make(world);
    await driver.start(settings);
    world.status(1, "error", "not_installed", "not_installed: stt/x");
    expect(driver.state.phase).toBe("idle");
    expect(out.stops).toHaveLength(1);
    expect(out.stops[0]).toContain("Other players' captions stopped");
  });

  it("stop ends the session, publishes what it holds and takes no later events", async () => {
    const { driver, out, settings } = make(world);
    await driver.start(settings);
    world.partial(1, 1, 1, "open");
    await driver.stop();
    expect(world.calls.at(-1)).toEqual({ cmd: "plugin:speech|stop_pipeline", args: { pipelineId: "loopback" } });
    expect(driver.state).toMatchObject({ status: ServiceNetworkState.disconnected, phase: "idle", source: null });
    expect(out.stops).toEqual([undefined]);
    world.final(1, 1, 2, "after stop");
    expect(out.finals).toEqual([]);
    // No watch keeps running after a stop.
    const before = world.calls.length;
    await world.clock.advance(WATCH_INTERVAL_MS * 2);
    expect(world.calls.length).toBe(before);
  });

  it("a new start replaces the running session without taking its events", async () => {
    const { driver, out, settings } = make(world);
    await driver.start(settings);
    await driver.start({ ...settings, language: "ja" });
    expect(world.starts.map((c) => [c.source, c.language])).toEqual([
      ["process:VRChat.exe", "auto"],
      ["process:VRChat.exe", "ja"],
    ]);
    world.final(1, 1, 2, "old");
    world.final(2, 1, 2, "new");
    expect(out.finals).toEqual([["new", seg(2, 1)]]);
    driver.dispose();
  });
});
