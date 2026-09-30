import i18n from "i18next";
import { proxy } from "valtio";
import { afterEach, beforeAll, beforeEach, describe, expect, it, vi } from "vitest";
import {
  ServiceNetworkState,
  TextEventSource,
  TextEventType,
  type RustCatalogEntry,
  type RustModelEntryState,
  type RustModelEntryStatus,
  type RustSpeechFinal,
  type RustSpeechPipelineConfig,
  type RustSpeechPipelineStarted,
  type RustSpeechRoutePlan,
  type RustSpeechRouteRequest,
  type RustSpeechStatus,
} from "@/types";

vi.mock("@tauri-apps/api/core", () => ({ invoke: vi.fn(async () => ""), isTauri: () => false }));
vi.mock("@tauri-apps/plugin-dialog", () => ({ ask: vi.fn(async () => false), confirm: vi.fn(async () => false), message: vi.fn(async () => "Cancel") }));
vi.mock("react-toastify", () => ({
  toast: Object.assign(vi.fn(), {
    error: vi.fn(),
    info: vi.fn(),
    warning: vi.fn(),
    success: vi.fn(),
    update: vi.fn(),
    dismiss: vi.fn(),
  }),
}));
vi.mock("@/core/services/systemLog", () => ({ pushSystemLog: vi.fn() }));
vi.mock("@/utils/devLog", () => ({ devLog: () => {} }));
vi.mock("@/utils/rustAudioInput", () => ({
  migrateLegacyNativeInputDevice: vi.fn(async () => {}),
  resolveRustInputDeviceForInvoke: (d: string | undefined) => (d && d.trim() ? d.trim() : null),
}));
vi.mock("@/shared/services/pubsub", () => ({ PUBSUB_STT_EXTERNAL_COMMAND: "stt.external.command" }));
vi.mock("../services/azure", () => ({ STT_AzureService: class {} }));
vi.mock("../services/deepgram", () => ({ STT_DeepgramService: class {} }));
vi.mock("../services/native", () => ({ STT_NativeService: class {} }));
vi.mock("../services/whisper", () => ({ STT_WhisperService: class {} }));
vi.mock("../services/vosk", () => ({ STT_VoskService: class {} }));
vi.mock("../services/moonshine", () => ({ STT_MoonshineService: class {} }));
vi.mock("../services/openaiAudio", () => ({ STT_OpenAI_AudioService: class {} }));
vi.mock("../services/externalMic", () => ({ STT_ExternalMicService: class {} }));

import { invoke } from "@tauri-apps/api/core";
import { ask, confirm, message } from "@tauri-apps/plugin-dialog";
import Service_STT, { connectingFallbackMs, localBackendCarryOver, textEventMeta, type SttBackendMap } from "../index";
import { Service_STT_Schema, STT_Backends, type STT_State } from "../schema";
import {
  askConsentDialog,
  BLOCKED_HOLD_MS,
  buildLocalPipelineConfig,
  buildLocalRouteRequest,
  consentTexts,
  HOLD_MARGIN_MS,
  LOCAL_PIPELINE_ID,
  localSpeechState,
  MAX_HOLD_MISSES,
  MAX_RESTARTS,
  notInstalledId,
  RESTART_DELAYS_MS,
  restartCanFix,
  SegmentPublisher,
  speechStatusLine,
  STABLE_RUN_MS,
  STALL_MS,
  startErrorMessage,
  STT_LocalService,
  TWO_PASS_BUDGET_MS,
  type ConsentChoice,
  type ConsentRequest,
  type LocalDriverDeps,
  type SegmentClock,
} from "../services/local";
import { SttMuteState, type ISTTReceiver, type ISTTService, type SttTextMeta } from "../types";
import { PROCESS_NONCE } from "@/utils/processNonce";

/** Stream part of the driver's segment ids (`mic-<nonce>:<epoch>:<n>`). */
const MIC = `mic-${PROCESS_NONCE}`;

beforeAll(async () => {
  await i18n.init({ lng: "en", resources: { en: { translation: {} } } });
});

// ---------------------------------------------------------------------------------------------------------------
// Segment publisher

class FakeClock implements SegmentClock {
  t = 1_000_000;
  timers: { at: number; fn: () => void; id: number }[] = [];
  #next = 1;
  now = () => this.t;
  setTimeout = (fn: () => void, ms: number) => {
    const id = this.#next++;
    this.timers.push({ at: this.t + ms, fn, id });
    return id;
  };
  clearTimeout = (h: unknown) => {
    this.timers = this.timers.filter((x) => x.id !== h);
  };
  advance(ms: number) {
    const end = this.t + ms;
    for (;;) {
      this.timers.sort((a, b) => a.at - b.at);
      const next = this.timers[0];
      if (!next || next.at > end) break;
      this.timers.shift();
      this.t = next.at;
      next.fn();
    }
    this.t = end;
  }
}

type Out = { kind: "interim" | "final" | "abandon"; text: string; meta: SttTextMeta };

function publisher(twoPass: boolean) {
  const clock = new FakeClock();
  const out: Out[] = [];
  const holding: boolean[] = [];
  const pub = new SegmentPublisher({
    pipelineId: "mic",
    stream: "mic",
    epoch: 3,
    twoPass,
    clock,
    sink: {
      interim: (text, meta) => out.push({ kind: "interim", text, meta }),
      final: (text, meta) => out.push({ kind: "final", text, meta }),
      abandon: (meta) => out.push({ kind: "abandon", text: "", meta }),
    },
    onHoldingChange: (h) => holding.push(h),
  });
  const partial = (segmentId: number, revision: number, text: string, lang: string | null = "en") =>
    pub.partial({ segmentId, revision, text, lang });
  const final = (segmentId: number, revision: number, text: string, extra: Partial<RustSpeechFinal> = {}) =>
    pub.final({ segmentId, revision, text, lang: "en", pass: "first", latencyMs: 0, startMs: 0, endMs: 2_000, turnEnd: null, ...extra });
  const closed = (segmentId: number, audioMs = 2_000) =>
    pub.utterance({ kind: "utterance", pipelineId: "mic", epoch: 3, segmentId, startMs: 0, endMs: audioMs, samples: audioMs * 16, forced: false });
  const summary = () => out.map((o) => `${o.kind}:${o.text}@${o.meta.segmentId}#${o.meta.revision}`);
  return { pub, clock, out, holding, partial, final, closed, summary };
}

describe("SegmentPublisher", () => {
  it("defaults the id stream to the pipeline id plus a per-process nonce", () => {
    expect(PROCESS_NONCE).toMatch(/^[0-9a-z]{6}$/);
    const out: SttTextMeta[] = [];
    const pub = new SegmentPublisher({
      pipelineId: "mic",
      epoch: 1,
      twoPass: false,
      clock: new FakeClock(),
      sink: { interim: (_t, meta) => out.push(meta), final: (_t, meta) => out.push(meta), abandon: (meta) => out.push(meta) },
    });
    pub.partial({ segmentId: 1, revision: 0, text: "hi", lang: "en" });
    expect(out.map((m) => m.segmentId)).toEqual([`mic-${PROCESS_NONCE}:1:1`]);
    expect(out[0].segmentId).toMatch(/^[A-Za-z0-9._-]+:\d+:\d+$/);
  });

  it("publishes partials as interims and a single-pass final, with segment metadata", () => {
    const p = publisher(false);
    p.partial(7, 0, "hel");
    p.partial(7, 1, "hello");
    p.partial(7, 1, "stale revision");
    p.closed(7);
    p.final(7, 2, "Hello.", { lang: "en-US", turnEnd: true });
    expect(p.summary()).toEqual(["interim:hel@mic:3:7#0", "interim:hello@mic:3:7#1", "final:Hello.@mic:3:7#2"]);
    expect(p.out[2].meta).toEqual({ segmentId: "mic:3:7", revision: 2, lang: "en", turnEnd: true });
    // Nothing more for a finished segment.
    p.partial(7, 5, "late");
    p.final(7, 6, "late final");
    expect(p.out).toHaveLength(3);
  });

  it("two-pass: the replacement is the only final; the first-pass text is shown as an interim meanwhile", () => {
    const p = publisher(true);
    p.partial(1, 0, "i scream");
    p.closed(1);
    p.final(1, 1, "I scream.");
    p.clock.advance(200);
    p.final(1, 2, "Ice cream.", { pass: "second", lang: "en" });
    expect(p.summary()).toEqual(["interim:i scream@mic:3:1#0", "interim:I scream.@mic:3:1#1", "final:Ice cream.@mic:3:1#2"]);
    expect(p.out.filter((o) => o.kind === "final")).toHaveLength(1);
  });

  it("two-pass: without a replacement the first-pass final is published after Rust's budget, and holding pauses after repeated misses", () => {
    const p = publisher(true);
    for (let seg = 1; seg <= MAX_HOLD_MISSES; seg++) {
      p.partial(seg, 0, `words ${seg}`);
      p.closed(seg, 2_000);
      p.final(seg, 1, `Words ${seg}.`);
      // budget = 1500 + 0.5 × 2000 ms of audio, plus the IPC margin.
      p.clock.advance(TWO_PASS_BUDGET_MS + 1_000 + HOLD_MARGIN_MS - 1);
      expect(p.out.at(-1)?.kind, `seg ${seg} still held`).toBe("interim");
      p.clock.advance(2);
      expect(p.out.at(-1)).toMatchObject({ kind: "final", text: `Words ${seg}.` });
    }
    expect(p.pub.holding).toBe(false);
    expect(p.holding).toEqual([false]);
    // Not holding: a first-pass final is published at once...
    p.final(9, 1, "Fast.");
    expect(p.out.at(-1)).toMatchObject({ kind: "final", text: "Fast." });
    // ...and a replacement showing up again (late, ignored as text) turns holding back on.
    p.final(9, 2, "Faster.", { pass: "second" });
    expect(p.out.at(-1)).toMatchObject({ kind: "final", text: "Fast." });
    expect(p.pub.holding).toBe(true);
    expect(p.holding).toEqual([false, true]);
  });

  it("keeps segments in order: a held sentence blocks the next one's interims, at most BLOCKED_HOLD_MS", () => {
    const p = publisher(true);
    p.partial(1, 0, "first sentence");
    p.closed(1, 8_000);
    p.final(1, 1, "First sentence.");
    p.partial(2, 0, "second");
    p.partial(2, 1, "second one");
    expect(p.summary()).toEqual(["interim:first sentence@mic:3:1#0", "interim:First sentence.@mic:3:1#1"]);
    p.clock.advance(BLOCKED_HOLD_MS);
    expect(p.summary()).toEqual([
      "interim:first sentence@mic:3:1#0",
      "interim:First sentence.@mic:3:1#1",
      "final:First sentence.@mic:3:1#1",
      "interim:second one@mic:3:2#1",
    ]);
    // The late replacement of segment 1 is not published twice.
    p.final(1, 2, "First sentence!", { pass: "second" });
    expect(p.out.filter((o) => o.kind === "final")).toHaveLength(1);
  });

  it("a replacement that lands while the next sentence waits publishes both in order", () => {
    const p = publisher(true);
    p.closed(1);
    p.final(1, 0, "One.");
    p.partial(2, 0, "tw");
    p.clock.advance(300);
    p.final(1, 1, "One!", { pass: "second" });
    expect(p.summary()).toEqual(["interim:One.@mic:3:1#0", "final:One!@mic:3:1#1", "interim:tw@mic:3:2#0"]);
  });

  it("abandons the interim of a dropped segment", () => {
    const p = publisher(false);
    p.partial(4, 0, "uh");
    p.pub.utterance({ kind: "dropped", pipelineId: "mic", epoch: 3, segmentId: 4, reason: "no_speech" });
    p.partial(5, 0, "next");
    expect(p.summary()).toEqual(["interim:uh@mic:3:4#0", "abandon:@mic:3:4#0", "interim:next@mic:3:5#0"]);
    // A dropped segment without interims publishes nothing.
    p.pub.utterance({ kind: "dropped", pipelineId: "mic", epoch: 3, segmentId: 6, reason: "muted" });
    expect(p.out).toHaveLength(3);
  });

  it("gives up on a segment whose final is late once a later one waited STALL_MS, and publishes that final later", () => {
    const p = publisher(false);
    p.partial(1, 0, "lagging");
    p.closed(1);
    p.partial(2, 0, "moving on");
    p.clock.advance(STALL_MS);
    expect(p.summary()).toEqual(["interim:lagging@mic:3:1#0", "abandon:@mic:3:1#0", "interim:moving on@mic:3:2#0"]);
    p.final(1, 1, "Lagging.");
    // Segment 2 is still open: segment 1's late final waits behind it.
    expect(p.out.filter((o) => o.kind === "final")).toHaveLength(0);
    p.final(2, 1, "Moving on.");
    expect(p.summary().slice(-2)).toEqual(["final:Moving on.@mic:3:2#1", "final:Lagging.@mic:3:1#1"]);
  });

  it("carries Smart Turn onto the final", () => {
    const p = publisher(false);
    p.partial(1, 0, "done");
    p.pub.turnEnd({ segmentId: 1 });
    p.final(1, 1, "Done.", { turnEnd: false });
    expect(p.out.at(-1)?.meta.turnEnd).toBe(true);
    p.final(2, 0, "Not yet.", { turnEnd: false });
    expect(p.out.at(-1)?.meta.turnEnd).toBe(false);
    p.final(3, 0, "Unknown.");
    expect(p.out.at(-1)?.meta).not.toHaveProperty("turnEnd");
  });

  it("flush publishes held finals and abandons open interims; dispose publishes nothing", () => {
    const p = publisher(true);
    p.closed(1);
    p.final(1, 0, "Held.");
    p.partial(2, 0, "open");
    p.pub.flush();
    // Segment 2's interim was never shown (it waited behind the held one): nothing to abandon.
    expect(p.summary()).toEqual(["interim:Held.@mic:3:1#0", "final:Held.@mic:3:1#0"]);
    p.final(3, 0, "after flush");
    expect(p.out).toHaveLength(2);

    const open = publisher(false);
    open.partial(1, 0, "open");
    open.pub.flush();
    expect(open.summary()).toEqual(["interim:open@mic:3:1#0", "abandon:@mic:3:1#0"]);

    const q = publisher(true);
    q.final(1, 0, "Held.");
    q.pub.dispose();
    q.clock.advance(60_000);
    expect(q.out.filter((o) => o.kind === "final")).toHaveLength(0);
  });

  it("commitHeld publishes held finals at the head in order and keeps the session going", () => {
    const p = publisher(true);
    p.closed(1);
    p.final(1, 1, "One.");
    p.closed(2);
    p.final(2, 1, "Two.");
    p.partial(3, 0, "thr");
    expect(p.summary()).toEqual(["interim:One.@mic:3:1#1"]);
    expect(p.pub.commitHeld()).toBe(true);
    // Both finished sentences go out with their first-pass text; the open one is not shown by the commit.
    expect(p.summary()).toEqual(["interim:One.@mic:3:1#1", "final:One.@mic:3:1#1", "final:Two.@mic:3:2#1"]);
    // A replacement arriving later is not a second final, and the session goes on.
    p.final(1, 2, "One!", { pass: "second" });
    p.partial(3, 1, "three");
    expect(p.summary().slice(3)).toEqual(["interim:three@mic:3:3#1"]);
    expect(p.pub.commitHeld()).toBe(false);
    p.final(3, 2, "Three.", { pass: "second" });
    expect(p.out.at(-1)).toMatchObject({ kind: "final", text: "Three." });
    expect(p.out.filter((o) => o.kind === "final")).toHaveLength(3);
  });

  it("does not repeat an interim whose text did not change", () => {
    const p = publisher(true);
    p.partial(1, 0, "same");
    p.closed(1);
    p.final(1, 1, "same");
    expect(p.summary()).toEqual(["interim:same@mic:3:1#0"]);
    p.final(1, 2, "Same.", { pass: "second" });
    expect(p.summary()).toEqual(["interim:same@mic:3:1#0", "final:Same.@mic:3:1#2"]);
  });
});

// ---------------------------------------------------------------------------------------------------------------
// Pure helpers

describe("local backend settings → Rust", () => {
  it("builds the route request and the pipeline config", () => {
    const local = { ...Service_STT_Schema.parse({}).local, language: "ja-JP", tier: "gpu" as const, twoPass: false, smartTurn: true, hotwords: "Sigil, VRChat, sigil" };
    const request = buildLocalRouteRequest(local);
    expect(request).toEqual({ language: "ja", tier: "gpu", twoPass: false });
    expect(buildLocalRouteRequest({ ...local, language: "nonsense-123" }).language).toBe("auto");
    const config = buildLocalPipelineConfig(local, { threshold: "0.6", minSilenceMs: "700" }, request, "USB Mic");
    expect(config).toMatchObject({
      pipelineId: LOCAL_PIPELINE_ID,
      source: "USB Mic",
      language: "ja",
      route: { language: "ja", tier: "gpu", twoPass: false },
      smartTurn: true,
      hotwords: ["Sigil", "VRChat"],
      hallucinationFilter: true,
      vad: { threshold: 0.6, min_silence_ms: 700 },
    });
    expect(buildLocalPipelineConfig(undefined, undefined, {}, null)).toMatchObject({ source: "", language: "auto", smartTurn: false, hotwords: [] });
  });

  it("maps pipeline statuses to status lines", () => {
    const title = (id: string) => `T(${id})`;
    const ev = (code: string, extra: Partial<RustSpeechStatus> = {}): RustSpeechStatus => ({
      pipelineId: "mic",
      epoch: 1,
      state: "loading",
      code,
      detail: "",
      progress: null,
      level: "info",
      modelId: null,
      engine: null,
      device: null,
      ...extra,
    });
    expect(speechStatusLine(ev("listening"), title)).toBeNull();
    expect(speechStatusLine(ev("loading_model", { modelId: "stt/x", progress: 150 }), title)).toEqual([
      "stt.local_status_loading_model",
      { progress: 100 },
    ]);
    expect(speechStatusLine(ev("backlog_dropped", { level: "warning" }), title)?.[1]).toMatchObject({ level: "warning", expireMs: expect.any(Number) });
    expect(speechStatusLine(ev("capture_gap", { level: "warning" }), title)?.[0]).toBe("stt.local_status_capture_gap");
    expect(speechStatusLine(ev("something_new", { detail: "hi", level: "warning" }), title)?.[0]).toBe("stt.local_status_warning");
    expect(speechStatusLine(ev("something_new"), title)).toBeNull();
  });

  it("reads not-installed errors and turns start errors into messages", () => {
    expect(notInstalledId("not_installed: stt/qwen3-asr-0.6b-int8: missing model files")).toBe("stt/qwen3-asr-0.6b-int8");
    expect(notInstalledId("engine_error: boom")).toBeNull();
    expect(startErrorMessage("not_installed: runtime/sherpa-onnx", (id) => id)).toBe("stt.local_error_not_installed");
    expect(startErrorMessage("no speech engine for this request (language xx)", (id) => id)).toBe("stt.local_error_no_engine");
    expect(startErrorMessage(new Error("boom"), (id) => id)).toBe("stt.local_error_start");
  });

  it("restarts only for errors a restart can fix", () => {
    const cpuFallback = { fallback: { engine: "sherpa-offline" as const, model: "stt/x", threads: null, variant: null, options: {} } };
    const none = { fallback: null };
    expect(restartCanFix("capture_error", "device lost", none, "auto")).toBe(true);
    expect(restartCanFix("engine_error", "runtime: CUDA error 700", none, "gpu")).toBe(true);
    expect(restartCanFix("engine_error", "5 utterances failed in a row; last error: runtime: boom", none, "cpu")).toBe(true);
    expect(restartCanFix("", "engine_error: still broken", none, "auto")).toBe(true);
    // Rust's non-transient kinds fail the same way on every start.
    expect(restartCanFix("engine_error", "unsupported: language xx", cpuFallback, "auto")).toBe(false);
    expect(restartCanFix("engine_error", "invalid_input: bad options", none, "auto")).toBe(false);
    expect(restartCanFix("", "unsupported: language xx", none, "auto")).toBe(false);
    // Out of memory: only when the next start can land on a lighter setup.
    expect(restartCanFix("engine_error", "out_of_memory: VRAM", none, "gpu")).toBe(false);
    expect(restartCanFix("engine_error", "out_of_memory: RAM", none, "cpu")).toBe(false);
    expect(restartCanFix("engine_error", "out_of_memory: VRAM", cpuFallback, "gpu")).toBe(true);
    expect(restartCanFix("engine_error", "out_of_memory: VRAM", none, "auto")).toBe(true);
    // Missing models and unknown codes never restart.
    expect(restartCanFix("not_installed", "stt/x", none, "auto")).toBe(false);
    expect(restartCanFix("", "not_installed: stt/x: missing files", none, "auto")).toBe(false);
    expect(restartCanFix("something_new", "boom", none, "auto")).toBe(false);
  });

  it("offers two or three choices in the consent dialog", async () => {
    const req: ConsentRequest = {
      items: [{ title: "Nemotron", bytes: 475_273_363 }, { title: "Parakeet", bytes: 629_500_000 }],
      bytes: 1_104_773_363,
      exact: true,
      smaller: { items: [{ title: "Nemotron", bytes: 475_273_363 }], bytes: 475_273_363, exact: true, live: true },
    };
    const texts = consentTexts(req);
    expect(texts.smaller).not.toBeNull();
    expect(texts.message).toContain("stt.local_consent_smaller_hint");
    expect(consentTexts({ ...req, smaller: { ...req.smaller!, live: false } }).message).toContain("stt.local_consent_smaller_hint_finals");
    vi.mocked(message).mockResolvedValueOnce(texts.smaller as string);
    expect(await askConsentDialog(req)).toBe("smaller");
    vi.mocked(message).mockResolvedValueOnce("Yes");
    expect(await askConsentDialog(req)).toBe("download");
    vi.mocked(message).mockResolvedValueOnce("Cancel");
    expect(await askConsentDialog(req)).toBe("cancel");
    vi.mocked(confirm).mockResolvedValueOnce(true);
    expect(await askConsentDialog({ ...req, smaller: null })).toBe("download");
    expect(consentTexts({ ...req, smaller: null }).smaller).toBeNull();
  });
});

// ---------------------------------------------------------------------------------------------------------------
// Driver

const NEMOTRON = "stt/nemotron-3.5-streaming-560ms-int8";
const PARAKEET = "stt/parakeet-ultra-int8";
const SHERPA = "runtime/sherpa-onnx";

function entry(id: string, bytes: number, requires: string[] = []): RustCatalogEntry {
  return {
    id,
    family: "stt",
    kind: id.startsWith("runtime/") ? "runtime" : "model",
    version: "1",
    title: id.split("/")[1],
    license: "x",
    status: "published",
    requires,
    files: [{ path: `${id}.bin`, size: bytes, sha256: "00", sources: [] }],
  };
}

const CATALOG = [entry(SHERPA, 8_000_000), entry(NEMOTRON, 475_000_000, [SHERPA]), entry(PARAKEET, 630_000_000, [SHERPA])];

const status = (id: string, state: RustModelEntryState, error: string | null = null): RustModelEntryStatus => ({
  id,
  state,
  installed_version: state === "installed" ? "1" : null,
  installed_path: null,
  bytes_on_disk: 0,
  adopted: false,
  error,
});

function makeModels(installed: string[]) {
  const state = proxy({
    catalog: CATALOG,
    states: Object.fromEntries(installed.map((id) => [id, status(id, "installed")])) as Record<string, RustModelEntryStatus>,
    progress: {},
    pending: {},
    opErrors: {} as Record<string, string>,
    loaded: false,
    error: null as string | null,
  });
  return {
    state,
    load: vi.fn(async () => {
      state.loaded = true;
    }),
    install: vi.fn(async (ids: readonly string[]) => {
      for (const id of ids) state.states[id] = status(id, "queued");
      return true;
    }),
    cancel: vi.fn(async (id: string) => {
      state.states[id] = status(id, "cancelled");
      return true;
    }),
    refreshStates: vi.fn(async () => {}),
    finish(id: string, s: RustModelEntryState = "installed", error: string | null = null) {
      state.states[id] = status(id, s, error);
    },
  };
}

/** Plans the fake router returns: two-pass en (Nemotron + Parakeet) or single pass (Nemotron alone). */
function planFor(request: RustSpeechRouteRequest | undefined, installed: Set<string>): RustSpeechRoutePlan {
  const twoPass = request?.twoPass !== false;
  const models = twoPass ? [NEMOTRON, SHERPA, PARAKEET] : [NEMOTRON, SHERPA];
  return {
    language: "en",
    tier: "cpu",
    streaming: { engine: "sherpa-online", model: NEMOTRON, threads: null, variant: null, options: { language: "en" } },
    final: twoPass ? { engine: "sherpa-offline", model: PARAKEET, threads: null, variant: null, options: { language: "en" } } : null,
    fallback: null,
    languageId: null,
    models,
    missing: models.filter((m) => !installed.has(m)),
    vramMb: 0,
    notes: [],
  };
}

interface Harness {
  svc: STT_LocalService;
  receiver: { [K in keyof ISTTReceiver]: ReturnType<typeof vi.fn> };
  models: ReturnType<typeof makeModels>;
  invoke: ReturnType<typeof vi.fn>;
  calls: (cmd: string) => Record<string, unknown>[];
  emit: (event: string, payload: unknown) => void;
  askConsent: ReturnType<typeof vi.fn>;
  patched: STT_State["local"];
  startResults: (RustSpeechPipelineStarted | Error | (() => Promise<RustSpeechPipelineStarted>))[];
  epoch: number;
}

let epochCounter = 100;

function harness(opts: { installed?: string[]; consent?: ConsentChoice; automatic?: boolean } = {}): Harness {
  const models = makeModels(opts.installed ?? [SHERPA, NEMOTRON, PARAKEET]);
  const handlers = new Map<string, (p: unknown) => void>();
  const log: { cmd: string; args: Record<string, unknown> }[] = [];
  const h = {} as Harness;
  h.startResults = [];
  h.epoch = 0;
  const installedSet = () => new Set(Object.values(models.state.states).filter((s) => s.state === "installed").map((s) => s.id));
  const invoke = vi.fn(async (cmd: string, args: Record<string, unknown> = {}) => {
    log.push({ cmd, args });
    if (cmd === "plugin:speech|preview_route") return planFor(args.request as RustSpeechRouteRequest, installedSet());
    if (cmd === "plugin:speech|start_pipeline") {
      const next = h.startResults.shift();
      if (next instanceof Error) throw next.message;
      if (typeof next === "function") return next();
      if (next) return next;
      h.epoch = ++epochCounter;
      const config = args.config as RustSpeechPipelineConfig;
      return { pipelineId: "mic", epoch: h.epoch, plan: planFor(config.route, installedSet()) };
    }
    return null;
  });
  const askConsent = vi.fn(async () => opts.consent ?? "download");
  const patched = { ...Service_STT_Schema.parse({}).local };
  const deps: Partial<LocalDriverDeps> = {
    available: () => true,
    invoke: invoke as LocalDriverDeps["invoke"],
    listen: async (event, handler) => {
      handlers.set(event, handler as (p: unknown) => void);
      return () => handlers.delete(event);
    },
    models: () => models,
    customMirror: () => false,
    askConsent,
    patchLocal: (fn) => fn(patched),
    resolveDevice: (d) => (d ? d : null),
    log: vi.fn(),
  };
  const receiver = {
    onStart: vi.fn(),
    onStop: vi.fn(),
    onInterim: vi.fn(),
    onFinal: vi.fn(),
    onStatus: vi.fn(),
    isMuted: vi.fn(() => false),
  };
  const svc = new STT_LocalService(receiver as unknown as ISTTReceiver, deps);
  svc.startAutomatic = opts.automatic === true;
  Object.assign(h, {
    svc,
    receiver,
    models,
    invoke,
    askConsent,
    patched,
    calls: (cmd: string) => log.filter((c) => c.cmd === cmd).map((c) => c.args),
    emit: (event: string, payload: unknown) => handlers.get(event)?.(payload),
  });
  return h;
}

function params(local: Partial<STT_State["local"]> = {}): STT_State {
  const p = Service_STT_Schema.parse({ backend: "local" });
  return { ...p, local: { ...p.local, language: "en", ...local } };
}

const flush = () => vi.advanceTimersByTimeAsync(0);

describe("STT_LocalService", () => {
  beforeEach(() => {
    vi.useFakeTimers();
    localSpeechState.plan = null;
    localSpeechState.restarts = 0;
  });
  afterEach(() => {
    vi.useRealTimers();
    vi.clearAllMocks();
  });

  it("starts with everything installed and publishes this session's speech with segment ids", async () => {
    const h = harness();
    await h.svc.start(params());
    expect(h.receiver.onStart).toHaveBeenCalledTimes(1);
    expect(h.askConsent).not.toHaveBeenCalled();
    expect(h.calls("plugin:speech|start_pipeline")[0].config).toMatchObject({ pipelineId: "mic", language: "en", route: { twoPass: true } });
    expect(localSpeechState.plan?.final?.model).toBe(PARAKEET);
    const e = h.epoch;
    h.emit("speech:partial", { pipelineId: "mic", epoch: e, segmentId: 1, revision: 0, text: "hello", lang: "en" });
    h.emit("speech:partial", { pipelineId: "mic", epoch: e - 1, segmentId: 1, revision: 1, text: "stale session", lang: "en" });
    h.emit("speech:partial", { pipelineId: "loopback", epoch: e, segmentId: 1, revision: 1, text: "other pipeline", lang: "en" });
    expect(h.receiver.onInterim).toHaveBeenCalledTimes(1);
    expect(h.receiver.onInterim).toHaveBeenCalledWith("hello", { segmentId: `${MIC}:${e}:1`, revision: 0, lang: "en" });
    const final = { pipelineId: "mic", epoch: e, segmentId: 1, tags: [], engine: "sherpa-offline", model: PARAKEET, startMs: 0, endMs: 900, forced: false, latencyMs: 10, turnEnd: null, lang: "en" };
    h.emit("speech:final", { ...final, revision: 1, text: "Hello", pass: "first" });
    expect(h.receiver.onFinal).not.toHaveBeenCalled(); // held for the two-pass replacement
    h.emit("speech:final", { ...final, revision: 2, text: "Hello!", pass: "second" });
    expect(h.receiver.onFinal).toHaveBeenCalledWith("Hello!", { segmentId: `${MIC}:${e}:1`, revision: 2, lang: "en" });

    const interims = h.receiver.onInterim.mock.calls.length;
    await h.svc.stop();
    expect(h.calls("plugin:speech|stop_pipeline")).toEqual([{ pipelineId: "mic" }]);
    expect(h.receiver.onStop).toHaveBeenCalledTimes(1);
    h.emit("speech:partial", { pipelineId: "mic", epoch: e, segmentId: 2, revision: 0, text: "after stop", lang: "en" });
    expect(h.receiver.onInterim).toHaveBeenCalledTimes(interims);
    expect(localSpeechState.plan).toBeNull();
  });

  it("accepts the new session's events before start_pipeline returns", async () => {
    const h = harness();
    let resolveStart!: (v: RustSpeechPipelineStarted) => void;
    h.startResults.push(() => new Promise((r) => { resolveStart = r; }));
    const started = h.svc.start(params());
    await flush();
    const epoch = ++epochCounter;
    h.emit("speech:status", { pipelineId: "mic", epoch, state: "loading", code: "loading_model", detail: "", progress: 40, level: "info", modelId: NEMOTRON, engine: null, device: null });
    expect(h.receiver.onStatus).toHaveBeenLastCalledWith("stt.local_status_loading_model", { progress: 40 });
    resolveStart({ pipelineId: "mic", epoch, plan: planFor({}, new Set(CATALOG.map((c) => c.id))) });
    await started;
    expect(h.receiver.onStart).toHaveBeenCalled();
    // Load progress delivered late does not come back once running.
    h.receiver.onStatus.mockClear();
    h.emit("speech:status", { pipelineId: "mic", epoch, state: "loading", code: "loading_model", detail: "", progress: 90, level: "info", modelId: NEMOTRON, engine: null, device: null });
    expect(h.receiver.onStatus).not.toHaveBeenCalled();
    h.emit("speech:status", { pipelineId: "mic", epoch, state: "listening", code: "listening", detail: "", progress: null, level: "info", modelId: null, engine: null, device: null });
    expect(h.receiver.onStatus).toHaveBeenCalledWith("");
    await h.svc.stop();
  });

  it("asks before downloading, downloads with progress, then starts", async () => {
    const h = harness({ installed: [] });
    const run = h.svc.start(params());
    await flush();
    expect(h.askConsent).toHaveBeenCalledTimes(1);
    const req = h.askConsent.mock.calls[0][0] as ConsentRequest;
    expect(req.items.map((i) => i.title)).toEqual(["sherpa-onnx", "nemotron-3.5-streaming-560ms-int8", "parakeet-ultra-int8"]);
    expect(req.bytes).toBe(8_000_000 + 475_000_000 + 630_000_000);
    // The single-pass setup (Nemotron alone) downloads far less: offered, with live captions.
    expect(req.smaller?.bytes).toBe(8_000_000 + 475_000_000);
    expect(req.smaller?.live).toBe(true);
    await flush();
    expect(h.models.install).toHaveBeenCalledWith([SHERPA, NEMOTRON, PARAKEET]);
    expect(h.calls("plugin:speech|start_pipeline")).toHaveLength(0);
    h.models.state.states[SHERPA] = status(SHERPA, "downloading");
    h.models.finish(SHERPA);
    h.models.state.states[NEMOTRON] = status(NEMOTRON, "downloading");
    await flush();
    expect(h.receiver.onStatus).toHaveBeenCalledWith(expect.stringMatching(/^stt\.local_status_download/), expect.anything());
    h.models.finish(NEMOTRON);
    h.models.state.states[PARAKEET] = status(PARAKEET, "downloading");
    await flush();
    h.models.finish(PARAKEET);
    await flush();
    await run;
    expect(h.calls("plugin:speech|start_pipeline")).toHaveLength(1);
    expect(h.receiver.onStart).toHaveBeenCalledTimes(1);
    await h.svc.stop();
  });

  it("stops with a message when the download is declined", async () => {
    const h = harness({ installed: [], consent: "cancel" });
    await h.svc.start(params());
    expect(h.models.install).not.toHaveBeenCalled();
    expect(h.calls("plugin:speech|start_pipeline")).toHaveLength(0);
    expect(h.receiver.onStop).toHaveBeenCalledWith("stt.local_error_declined");
  });

  it("never asks on an automatic start", async () => {
    const h = harness({ installed: [], automatic: true });
    await h.svc.start(params());
    expect(h.askConsent).not.toHaveBeenCalled();
    expect(h.models.install).not.toHaveBeenCalled();
    expect(h.receiver.onStop).toHaveBeenCalledWith("stt.local_error_models_missing");
  });

  it("the smaller setup turns two-pass off and downloads only its models", async () => {
    const h = harness({ installed: [], consent: "smaller" });
    const run = h.svc.start(params());
    await flush();
    await flush();
    expect(h.patched.twoPass).toBe(false);
    expect(h.models.install).toHaveBeenCalledWith([SHERPA, NEMOTRON]);
    h.models.finish(SHERPA);
    h.models.finish(NEMOTRON);
    await flush();
    await run;
    expect(h.calls("plugin:speech|start_pipeline")[0].config).toMatchObject({ route: { twoPass: false } });
    await h.svc.stop();
  });

  it("reports a failed download", async () => {
    const h = harness({ installed: [SHERPA, NEMOTRON] });
    const run = h.svc.start(params());
    await flush();
    await flush();
    expect(h.models.install).toHaveBeenCalledWith([PARAKEET]);
    h.models.state.states[PARAKEET] = status(PARAKEET, "downloading");
    await flush();
    h.models.finish(PARAKEET, "failed", "HTTP 404");
    await flush();
    await run;
    expect(h.receiver.onStop).toHaveBeenCalledWith("stt.local_error_download_failed");
    expect(h.calls("plugin:speech|start_pipeline")).toHaveLength(0);
  });

  it("a stop during the download cancels what this start queued and starts nothing", async () => {
    const h = harness({ installed: [SHERPA, NEMOTRON] });
    const controller = new AbortController();
    h.svc.startAbortSignal = controller.signal;
    const run = h.svc.start(params());
    await flush();
    await flush();
    h.models.state.states[PARAKEET] = status(PARAKEET, "downloading");
    await flush();
    controller.abort();
    await h.svc.stop();
    await run;
    expect(h.models.cancel).toHaveBeenCalledWith(PARAKEET);
    expect(h.calls("plugin:speech|start_pipeline")).toHaveLength(0);
    expect(h.receiver.onStart).not.toHaveBeenCalled();
  });

  it("restarts the pipeline after a retryable error, publishing what the ended session held", async () => {
    const h = harness();
    await h.svc.start(params());
    const first = h.epoch;
    h.emit("speech:utterance", { kind: "utterance", pipelineId: "mic", epoch: first, segmentId: 3, startMs: 0, endMs: 1_000, samples: 16_000, forced: false });
    h.emit("speech:final", { pipelineId: "mic", epoch: first, segmentId: 3, revision: 0, text: "Held words.", lang: "en", tags: [], pass: "first", engine: "sherpa-online", model: NEMOTRON, startMs: 0, endMs: 1_000, forced: false, latencyMs: 0, turnEnd: null });
    expect(h.receiver.onFinal).not.toHaveBeenCalled();
    h.emit("speech:status", { pipelineId: "mic", epoch: first, state: "error", code: "capture_error", detail: "device lost", progress: null, level: "error", modelId: null, engine: null, device: null });
    expect(h.receiver.onFinal).toHaveBeenCalledWith("Held words.", expect.objectContaining({ segmentId: `${MIC}:${first}:3` }));
    expect(h.receiver.onStatus).toHaveBeenLastCalledWith("stt.local_status_restarting", { level: "warning" });
    expect(h.receiver.onStop).not.toHaveBeenCalled();
    await vi.advanceTimersByTimeAsync(RESTART_DELAYS_MS[0]);
    expect(h.calls("plugin:speech|start_pipeline")).toHaveLength(2);
    const second = h.epoch;
    expect(second).toBeGreaterThan(first);
    expect(localSpeechState.restarts).toBe(1);
    h.emit("speech:partial", { pipelineId: "mic", epoch: second, segmentId: 4, revision: 0, text: "back", lang: "en" });
    expect(h.receiver.onInterim).toHaveBeenLastCalledWith("back", expect.objectContaining({ segmentId: `${MIC}:${second}:4` }));
    await h.svc.stop();
  });

  it("handles a session that failed before its start result arrived", async () => {
    const h = harness();
    let resolveStart!: (v: RustSpeechPipelineStarted) => void;
    h.startResults.push(() => new Promise((r) => { resolveStart = r; }));
    const run = h.svc.start(params());
    await flush();
    const epoch = ++epochCounter;
    h.emit("speech:status", { pipelineId: "mic", epoch, state: "error", code: "capture_error", detail: "gone", progress: null, level: "error", modelId: null, engine: null, device: null });
    resolveStart({ pipelineId: "mic", epoch, plan: planFor({}, new Set(CATALOG.map((c) => c.id))) });
    await run;
    expect(h.receiver.onStart).toHaveBeenCalledTimes(1);
    expect(h.receiver.onStatus).toHaveBeenLastCalledWith("stt.local_status_restarting", { level: "warning" });
    await vi.advanceTimersByTimeAsync(RESTART_DELAYS_MS[0]);
    expect(h.calls("plugin:speech|start_pipeline")).toHaveLength(2);
    await h.svc.stop();
  });

  it("gives up after the restart budget, and never restarts a missing model", async () => {
    const h = harness();
    await h.svc.start(params());
    for (let i = 0; i < MAX_RESTARTS; i++) h.startResults.push(new Error("engine_error: still broken"));
    h.emit("speech:status", { pipelineId: "mic", epoch: h.epoch, state: "error", code: "engine_error", detail: "boom", progress: null, level: "error", modelId: null, engine: null, device: null });
    for (const delay of RESTART_DELAYS_MS) await vi.advanceTimersByTimeAsync(delay);
    expect(h.calls("plugin:speech|start_pipeline")).toHaveLength(1 + MAX_RESTARTS);
    expect(h.receiver.onStop).toHaveBeenCalledWith("stt.local_error_gave_up");

    const g = harness();
    await g.svc.start(params());
    g.emit("speech:status", { pipelineId: "mic", epoch: g.epoch, state: "error", code: "not_installed", detail: "stt/x", progress: null, level: "error", modelId: NEMOTRON, engine: null, device: null });
    await vi.advanceTimersByTimeAsync(60_000);
    expect(g.calls("plugin:speech|start_pipeline")).toHaveLength(1);
    expect(g.receiver.onStop).toHaveBeenCalledWith("stt.local_error_not_installed");
  });

  it("counts restarts that fail at start against the budget, even after a long session", async () => {
    const h = harness();
    await h.svc.start(params());
    await vi.advanceTimersByTimeAsync(5 * 60_000);
    for (let i = 0; i < MAX_RESTARTS + 3; i++) h.startResults.push(new Error("engine_error: still broken"));
    h.emit("speech:status", { pipelineId: "mic", epoch: h.epoch, state: "error", code: "engine_error", detail: "runtime: device reset", progress: null, level: "error", modelId: null, engine: null, device: null });
    for (const delay of RESTART_DELAYS_MS) await vi.advanceTimersByTimeAsync(delay);
    await vi.advanceTimersByTimeAsync(5 * 60_000);
    expect(h.calls("plugin:speech|start_pipeline")).toHaveLength(1 + MAX_RESTARTS);
    expect(h.receiver.onStop).toHaveBeenCalledWith("stt.local_error_gave_up");
  });

  it("a session that ran STABLE_RUN_MS after a restart begins a new restart budget", async () => {
    const h = harness();
    await h.svc.start(params());
    const fail = () =>
      h.emit("speech:status", { pipelineId: "mic", epoch: h.epoch, state: "error", code: "capture_error", detail: "device lost", progress: null, level: "error", modelId: null, engine: null, device: null });
    // Two failures in a row: the second waits the second delay.
    fail();
    await vi.advanceTimersByTimeAsync(RESTART_DELAYS_MS[0]);
    expect(h.calls("plugin:speech|start_pipeline")).toHaveLength(2);
    fail();
    await vi.advanceTimersByTimeAsync(RESTART_DELAYS_MS[0]);
    expect(h.calls("plugin:speech|start_pipeline")).toHaveLength(2);
    await vi.advanceTimersByTimeAsync(RESTART_DELAYS_MS[1] - RESTART_DELAYS_MS[0]);
    expect(h.calls("plugin:speech|start_pipeline")).toHaveLength(3);
    // After a long run the next failure restarts after the first delay again.
    await vi.advanceTimersByTimeAsync(STABLE_RUN_MS);
    fail();
    await vi.advanceTimersByTimeAsync(RESTART_DELAYS_MS[0]);
    expect(h.calls("plugin:speech|start_pipeline")).toHaveLength(4);
    expect(localSpeechState.restarts).toBe(3);
    await h.svc.stop();
    expect(localSpeechState.restarts).toBe(0);
  });

  it("does not restart for engine errors that fail on every start", async () => {
    const h = harness();
    await h.svc.start(params());
    h.emit("speech:status", { pipelineId: "mic", epoch: h.epoch, state: "error", code: "engine_error", detail: "unsupported: language xx", progress: null, level: "error", modelId: null, engine: null, device: null });
    await vi.advanceTimersByTimeAsync(60_000);
    expect(h.calls("plugin:speech|start_pipeline")).toHaveLength(1);
    expect(h.receiver.onStop).toHaveBeenCalledWith("stt.local_error_failed");

    // Out of memory with a forced GPU tier and no CPU fallback: the same setup runs out again.
    const gpu = harness();
    await gpu.svc.start(params({ tier: "gpu" }));
    gpu.emit("speech:status", { pipelineId: "mic", epoch: gpu.epoch, state: "error", code: "engine_error", detail: "out_of_memory: cudaMalloc", progress: null, level: "error", modelId: null, engine: null, device: null });
    await vi.advanceTimersByTimeAsync(60_000);
    expect(gpu.calls("plugin:speech|start_pipeline")).toHaveLength(1);
    expect(gpu.receiver.onStop).toHaveBeenCalledWith("stt.local_error_failed");

    // A restart whose start fails with such an error stops at once instead of using up the budget.
    const r = harness();
    await r.svc.start(params());
    r.startResults.push(new Error("invalid_input: bad options"));
    r.emit("speech:status", { pipelineId: "mic", epoch: r.epoch, state: "error", code: "capture_error", detail: "device lost", progress: null, level: "error", modelId: null, engine: null, device: null });
    await vi.advanceTimersByTimeAsync(60_000);
    expect(r.calls("plugin:speech|start_pipeline")).toHaveLength(2);
    expect(r.receiver.onStop).toHaveBeenCalledWith("stt.local_error_start");
  });

  it("resets the restart count shown in the panel for each new session", async () => {
    localSpeechState.restarts = 4; // left by an earlier session
    const h = harness();
    await h.svc.start(params());
    expect(localSpeechState.restarts).toBe(0);
    await h.svc.stop();
  });

  it("commitPending publishes a held first-pass final at once; the later replacement is ignored", async () => {
    const h = harness();
    await h.svc.start(params());
    const e = h.epoch;
    const final = { pipelineId: "mic", epoch: e, segmentId: 1, tags: [], engine: "sherpa-online", model: NEMOTRON, startMs: 0, endMs: 2_000, forced: false, latencyMs: 0, turnEnd: null, lang: "en" };
    h.emit("speech:utterance", { kind: "utterance", pipelineId: "mic", epoch: e, segmentId: 1, startMs: 0, endMs: 2_000, samples: 32_000, forced: false });
    h.emit("speech:final", { ...final, revision: 1, text: "Hello everyone.", pass: "first" });
    expect(h.receiver.onInterim).toHaveBeenLastCalledWith("Hello everyone.", expect.objectContaining({ segmentId: `${MIC}:${e}:1`, revision: 1 }));
    expect(h.receiver.onFinal).not.toHaveBeenCalled();
    h.svc.commitPending();
    expect(h.receiver.onFinal).toHaveBeenCalledTimes(1);
    expect(h.receiver.onFinal).toHaveBeenCalledWith("Hello everyone.", { segmentId: `${MIC}:${e}:1`, revision: 1, lang: "en" });
    h.emit("speech:final", { ...final, revision: 2, text: "Hello, everyone!", pass: "second" });
    await vi.advanceTimersByTimeAsync(10_000);
    expect(h.receiver.onFinal).toHaveBeenCalledTimes(1);
    await h.svc.stop();
  });

  it("stops what a start overtaken by the stop started", async () => {
    const h = harness();
    const controller = new AbortController();
    h.svc.startAbortSignal = controller.signal;
    let resolveStart!: (v: RustSpeechPipelineStarted) => void;
    h.startResults.push(() => new Promise((r) => { resolveStart = r; }));
    const run = h.svc.start(params());
    await flush();
    controller.abort();
    const stopping = h.svc.stop();
    await flush();
    resolveStart({ pipelineId: "mic", epoch: ++epochCounter, plan: planFor({}, new Set()) });
    await stopping;
    await run;
    expect(h.calls("plugin:speech|stop_pipeline")).toHaveLength(2);
    expect(h.receiver.onStart).not.toHaveBeenCalled();
  });

  it("plans again once when the pipeline wants a model the preview did not", async () => {
    const h = harness();
    h.startResults.push(new Error(`not_installed: ${PARAKEET}`));
    await h.svc.start(params());
    expect(h.calls("plugin:speech|preview_route")).toHaveLength(2);
    expect(h.calls("plugin:speech|start_pipeline")).toHaveLength(2);
    expect(h.receiver.onStart).toHaveBeenCalledTimes(1);
    await h.svc.stop();
  });

  it("refuses to start outside the desktop app", async () => {
    const receiver = { onStart: vi.fn(), onStop: vi.fn(), onInterim: vi.fn(), onFinal: vi.fn(), onStatus: vi.fn(), isMuted: () => false };
    const svc = new STT_LocalService(receiver, { available: () => false });
    await svc.start(params());
    expect(receiver.onStop).toHaveBeenCalledWith("stt.local_error_desktop_only");
  });
});

// ---------------------------------------------------------------------------------------------------------------
// Service_STT: metadata on published text, heavy connecting window, automatic starts, the local-backend offer

class FakeEngine implements ISTTService {
  static instances: FakeEngine[] = [];
  startAbortSignal: AbortSignal | null = null;
  startAutomatic?: boolean;
  commitPending?: () => void;
  constructor(public receiver: ISTTReceiver) {
    FakeEngine.instances.push(this);
  }
  start() {}
  stop() {}
  dispose() {}
}

/** A fake `local` engine: a real SegmentPublisher (two-pass) feeding the engine's receiver, committed like the driver. */
function twoPassEngine(engine: FakeEngine, epoch = 5) {
  const pub = new SegmentPublisher({
    pipelineId: "mic",
    stream: "mic",
    epoch,
    twoPass: true,
    sink: {
      interim: (text, meta) => engine.receiver.onInterim(text, meta),
      final: (text, meta) => engine.receiver.onFinal(text, meta),
      abandon: (meta) => engine.receiver.onInterim("", meta),
    },
  });
  engine.commitPending = () => void pub.commitHeld();
  const first = (segmentId: number, revision: number, text: string) =>
    pub.final({ segmentId, revision, text, lang: "en", pass: "first", latencyMs: 0, startMs: 0, endMs: 2_000, turnEnd: null });
  const second = (segmentId: number, revision: number, text: string) =>
    pub.final({ segmentId, revision, text, lang: "en", pass: "second", latencyMs: 0, startMs: 0, endMs: 2_000, turnEnd: null });
  return { pub, first, second };
}

const fakeMap = Object.fromEntries(Object.values(STT_Backends).map((b) => [b, FakeEngine])) as SttBackendMap;

describe("Service_STT with segment metadata", () => {
  let publishText: ReturnType<typeof vi.fn>;
  let data: STT_State;

  function makeService(overrides: Partial<STT_State> = {}, isApp = true) {
    data = proxy({ ...Service_STT_Schema.parse({}), ...overrides });
    const state = { services: { stt: { data } }, onboardingComplete: true };
    publishText = vi.fn();
    vi.stubGlobal("window", {
      Config: { isApp: () => isApp },
      ApiServer: { state, patchService: (_key: "stt", fn: (s: { data: STT_State }) => void) => fn(state.services.stt) },
      ApiShared: { pubsub: { publishText, subscribe: vi.fn(() => "sub"), unsubscribe: vi.fn(), broadcastLocalWebSocketOnly: vi.fn() } },
    });
    const service = new Service_STT(fakeMap);
    service.updateReplacementsCache();
    return service;
  }

  beforeEach(() => {
    vi.useFakeTimers();
    FakeEngine.instances = [];
    vi.stubGlobal("navigator", { hardwareConcurrency: 8 });
  });
  afterEach(() => {
    vi.useRealTimers();
    vi.unstubAllGlobals();
    vi.clearAllMocks();
  });

  it("publishes segment metadata with the text, and closes an open segment with its id", async () => {
    const service = makeService({ backend: STT_Backends.local });
    service.start();
    await flush();
    const engine = FakeEngine.instances[0];
    engine.receiver.onStart();
    engine.receiver.onInterim("hel", { segmentId: "mic:5:1", revision: 0, lang: "en" });
    engine.receiver.onFinal("Hello.", { segmentId: "mic:5:1", revision: 2, lang: "en", turnEnd: true });
    engine.receiver.onInterim("wor", { segmentId: "mic:5:2", revision: 0, lang: "en" });
    await flush();
    expect(publishText).toHaveBeenNthCalledWith(1, TextEventSource.stt, { value: "hel", type: TextEventType.interim, segmentId: "mic:5:1", revision: 0, lang: "en" });
    expect(publishText).toHaveBeenNthCalledWith(2, TextEventSource.stt, {
      value: "Hello.",
      type: TextEventType.final,
      segmentId: "mic:5:1",
      revision: 2,
      lang: "en",
      turnEnd: true,
    });
    service.stop();
    await flush();
    expect(publishText).toHaveBeenLastCalledWith(TextEventSource.stt, { value: "[...]", type: TextEventType.final, segmentId: "mic:5:2", revision: 1, lang: "en" });
  });

  const finalsOf = (segmentId: string) =>
    publishText.mock.calls.filter(([, e]) => e.type === TextEventType.final && e.segmentId === segmentId).map(([, e]) => e);

  it("muting publishes a sentence held for its two-pass replacement instead of cutting it off", async () => {
    const service = makeService({ backend: STT_Backends.local });
    service.start();
    await flush();
    const engine = FakeEngine.instances[0];
    engine.receiver.onStart();
    const p = twoPassEngine(engine);
    p.pub.partial({ segmentId: 1, revision: 0, text: "hello every", lang: "en" });
    p.first(1, 1, "Hello everyone, welcome to the stream.");
    await flush();
    // Shown as an interim while it waits for the replacement.
    expect(publishText).toHaveBeenLastCalledWith(TextEventSource.stt, expect.objectContaining({ type: TextEventType.interim, revision: 1 }));
    service.setMuted(true);
    await flush();
    expect(finalsOf("mic:5:1")).toEqual([
      { value: "Hello everyone, welcome to the stream.", type: TextEventType.final, segmentId: "mic:5:1", revision: 1, lang: "en" },
    ]);
    expect(publishText.mock.calls.some(([, e]) => e.value === "[...]")).toBe(false);
    // The replacement arriving later is not a second final.
    p.second(1, 2, "Hello everyone, welcome to the stream!");
    service.setMuted(false);
    await vi.advanceTimersByTimeAsync(10_000);
    expect(finalsOf("mic:5:1")).toHaveLength(1);
  });

  it("stopping publishes a held sentence instead of the marker", async () => {
    const service = makeService({ backend: STT_Backends.local });
    service.start();
    await flush();
    const engine = FakeEngine.instances[0];
    engine.receiver.onStart();
    const p = twoPassEngine(engine);
    p.first(1, 0, "Goodbye.");
    await flush();
    service.stop();
    await flush();
    expect(finalsOf("mic:5:1")).toEqual([{ value: "Goodbye.", type: TextEventType.final, segmentId: "mic:5:1", revision: 0, lang: "en" }]);
    expect(publishText.mock.calls.some(([, e]) => e.value === "[...]")).toBe(false);
  });

  it("a held sentence muted while the UwU filter runs is still published", async () => {
    const service = makeService({ backend: STT_Backends.local, uwu: true });
    vi.mocked(invoke).mockImplementation(async (_cmd, args) => String((args as { value: string }).value).toUpperCase());
    service.start();
    await flush();
    const engine = FakeEngine.instances[0];
    engine.receiver.onStart();
    const p = twoPassEngine(engine);
    p.first(1, 0, "Held words.");
    await flush();
    service.setMuted(true);
    await flush();
    expect(finalsOf("mic:5:1")).toEqual([{ value: "HELD WORDS.", type: TextEventType.final, segmentId: "mic:5:1", revision: 0, lang: "en" }]);
    expect(publishText.mock.calls.some(([, e]) => e.value === "[...]")).toBe(false);
    vi.mocked(invoke).mockReset();
  });

  it("a segment closed by the marker gets no second final, even after an unmute", async () => {
    const service = makeService({ backend: STT_Backends.local });
    service.start();
    await flush();
    const engine = FakeEngine.instances[0];
    engine.receiver.onStart();
    const p = twoPassEngine(engine);
    p.pub.partial({ segmentId: 2, revision: 0, text: "still talk", lang: "en" });
    await flush();
    service.setMuted(true);
    await flush();
    expect(finalsOf("mic:5:2")).toEqual([{ value: "[...]", type: TextEventType.final, segmentId: "mic:5:2", revision: 1, lang: "en" }]);
    // Unmuted before the segment's own final (revision 1 too) arrives: it must not follow the marker.
    service.setMuted(false);
    expect(service.serviceState.muted).toBe(SttMuteState.unmuted);
    p.first(2, 1, "Still talking.");
    await vi.advanceTimersByTimeAsync(20_000);
    expect(finalsOf("mic:5:2")).toHaveLength(1);
    expect(publishText.mock.calls.some(([, e]) => String(e.value).startsWith("Still talking"))).toBe(false);
    // The next sentence is published normally.
    p.pub.partial({ segmentId: 3, revision: 0, text: "next", lang: "en" });
    p.second(3, 1, "Next.");
    await flush();
    expect(finalsOf("mic:5:3")).toEqual([{ value: "Next.", type: TextEventType.final, segmentId: "mic:5:3", revision: 1, lang: "en" }]);
  });

  it("drops malformed metadata and leaves events of other engines unchanged", () => {
    expect(textEventMeta(undefined)).toEqual({});
    expect(textEventMeta({ segmentId: "<x>", revision: -1, lang: "EN US", turnEnd: undefined })).toEqual({});
    expect(textEventMeta({ segmentId: "mic:1:2", revision: 3, lang: "yue", turnEnd: false })).toEqual({ segmentId: "mic:1:2", revision: 3, lang: "yue", turnEnd: false });
  });

  it("gives the local backend the long connecting window and tells engines about automatic starts", async () => {
    expect(connectingFallbackMs(STT_Backends.local)).toBe(300_000);
    expect(connectingFallbackMs(STT_Backends.azure)).toBe(45_000);
    const service = makeService({ backend: STT_Backends.local, autoStart: true, desktopDefaultMigrated: true });
    await service.init();
    await flush();
    expect(FakeEngine.instances[0].startAutomatic).toBe(true);
    // Not yet ready after 4 minutes of silence...
    await vi.advanceTimersByTimeAsync(240_000);
    expect(service.serviceState.status).toBe(ServiceNetworkState.connecting);
    service.stop();
    service.start();
    await flush();
    expect(FakeEngine.instances[1].startAutomatic).toBe(false);
  });

  it("offers the local backend once, carrying the language and microphone over", async () => {
    const service = makeService({ backend: STT_Backends.whisper, whisper: { ...Service_STT_Schema.parse({}).whisper, language: "ja", device: "USB Mic" } });
    vi.mocked(ask).mockResolvedValueOnce(true);
    expect(await service.offerLocalBackend()).toBe(true);
    expect(data.backend).toBe(STT_Backends.local);
    expect(data.local.language).toBe("ja");
    expect(data.local.device).toBe("USB Mic");
    expect(data.localOfferShown).toBe(true);
    expect(await service.offerLocalBackend()).toBe(false);
    expect(ask).toHaveBeenCalledTimes(1);

    const declined = makeService({ backend: STT_Backends.native });
    vi.mocked(ask).mockResolvedValueOnce(false);
    expect(await declined.offerLocalBackend()).toBe(false);
    expect(data.backend).toBe(STT_Backends.native);
    expect(data.localOfferShown).toBe(true);

    const web = makeService({ backend: STT_Backends.native }, false);
    expect(await web.offerLocalBackend()).toBe(false);
    expect(data.localOfferShown).toBe(false);
  });

  it("carries language only over the automatic default", () => {
    const base = Service_STT_Schema.parse({});
    expect(localBackendCarryOver({ ...base, backend: STT_Backends.native, native: { ...base.native, language: "pt-BR" } })).toEqual({ language: "pt", device: null });
    expect(localBackendCarryOver({ ...base, backend: STT_Backends.whisper, local: { ...base.local, language: "de" } }).language).toBeNull();
    expect(localBackendCarryOver({ ...base, backend: STT_Backends.whisper, whisper: { ...base.whisper, language: "auto" } }).language).toBeNull();
  });
});
