import { afterEach, beforeAll, beforeEach, describe, expect, it, vi } from "vitest";
import type {
  RustSpeechFinal,
  RustSpeechPipelineConfig,
  RustSpeechPipelineStarted,
  RustSpeechPipelineStatus,
  RustSpeechStatus,
  RustSpeechUtterance,
} from "@/types";

const listeners = new Map<string, (event: { payload: unknown }) => void>();
const invoke = vi.fn();
vi.mock("@tauri-apps/api/core", () => ({ invoke: (cmd: string, args?: unknown) => invoke(cmd, args), isTauri: () => true }));
vi.mock("@tauri-apps/api/event", () => ({
  listen: vi.fn(async (event: string, handler: (event: { payload: unknown }) => void) => {
    listeners.set(event, handler);
    return () => listeners.delete(event);
  }),
}));
vi.mock("@/utils/devLog", () => ({ devLog: () => {} }));
vi.mock("@/utils/rustAudioInput", () => ({ resolveRustInputDeviceForInvoke: (d: string) => (d ? d : null) }));

import {
  HTTP_LATENCY_BUDGET_MS,
  HTTP_MAX_BACKLOG,
  noteSttPipelineStart,
  RustUtteranceService,
  speechStatusLine,
  START_SETTLE_WAIT_MS,
  STT_PIPELINE_ID,
  type PipelineEngineSetup,
} from "../rustCapturePolling";
import { Service_STT_Schema, type STT_State } from "../schema";
import { TRANSIENT_WARNING_MS } from "../statusChannel";
import type { ISTTReceiver, SttStopOptions } from "../types";
import { initSttTestI18n, translated } from "./sttTestI18n";
import { PROCESS_NONCE } from "@/utils/processNonce";

const SETUP: PipelineEngineSetup = {
  engine: { engine: "http", model: "", options: { preset: "openai", baseUrl: "http://127.0.0.1:8000/v1", model: "small", apiKey: "sk-secret" } },
  language: "en",
  prompt: "VRChat",
};

class TestEngine extends RustUtteranceService {
  protected readonly logTag = "[Test]";
  setup: PipelineEngineSetup | string = SETUP;
  protected engineLabel() {
    return "TestSTT";
  }
  protected inputDevice(params: STT_State) {
    return params.openai_audio.device;
  }
  protected prepare() {
    return this.setup;
  }
  protected postProcess(text: string) {
    return text.replace("[noise]", "").trim();
  }
}

function receiver() {
  return {
    onStart: vi.fn(),
    onStop: vi.fn(),
    onInterim: vi.fn(),
    onFinal: vi.fn(),
    onStatus: vi.fn(),
    onInputSilence: vi.fn(),
    isMuted: vi.fn(() => false),
  } satisfies ISTTReceiver;
}

// --- fake speech plugin ------------------------------------------------------------------------------------------

let epoch = 0;
let stops = 0;
let configs: RustSpeechPipelineConfig[] = [];
let statusRows: RustSpeechPipelineStatus[] = [];
/** Replaces the default `start_pipeline` behaviour (events + result). */
let startImpl: ((config: RustSpeechPipelineConfig) => Promise<RustSpeechPipelineStarted>) | null = null;

const started = (e: number): RustSpeechPipelineStarted => ({
  pipelineId: STT_PIPELINE_ID,
  epoch: e,
  plan: {
    language: "en",
    tier: "cpu",
    streaming: null,
    final: SETUP.engine,
    fallback: null,
    languageId: null,
    models: [],
    missing: [],
    vramMb: 0,
    notes: [],
  },
});

function emit(event: string, payload: unknown) {
  listeners.get(event)?.({ payload });
}

function status(e: number, state: RustSpeechStatus["state"], code: string, detail = "", level: RustSpeechStatus["level"] = "info"): RustSpeechStatus {
  return { pipelineId: STT_PIPELINE_ID, epoch: e, state, code, detail, progress: null, level, modelId: null, engine: "http", device: null };
}

function final(e: number, segmentId: number, text: string, extra: Partial<RustSpeechFinal> = {}): RustSpeechFinal {
  return {
    pipelineId: STT_PIPELINE_ID,
    epoch: e,
    segmentId,
    revision: 0,
    text,
    lang: "en",
    tags: [],
    pass: "first",
    engine: "http",
    model: "small",
    startMs: 0,
    endMs: 1000,
    forced: false,
    latencyMs: 300,
    turnEnd: null,
    ...extra,
  };
}

const dropped = (e: number, segmentId: number, reason: string): RustSpeechUtterance => ({
  kind: "dropped",
  pipelineId: STT_PIPELINE_ID,
  epoch: e,
  segmentId,
  reason,
});

/** Like Rust: lifecycle statuses of the new epoch are emitted before the command resolves. */
async function defaultStart(): Promise<RustSpeechPipelineStarted> {
  const e = ++epoch;
  emit("speech:status", status(e, "starting", "starting"));
  emit("speech:status", status(e, "loading", "loading_model", "loading the final engine (http)"));
  emit("speech:status", status(e, "listening", "listening"));
  return started(e);
}

function deferred<T>() {
  let resolve!: (v: T) => void;
  let reject!: (e: unknown) => void;
  const promise = new Promise<T>((res, rej) => {
    resolve = res;
    reject = rej;
  });
  return { promise, resolve, reject };
}

async function startedEngine(r: ISTTReceiver, params: Partial<STT_State> = {}) {
  const engine = new TestEngine(r);
  await engine.start({ ...Service_STT_Schema.parse({}), ...params });
  return engine;
}

beforeAll(async () => {
  await initSttTestI18n();
});

beforeEach(() => {
  listeners.clear();
  epoch = 10;
  stops = 0;
  configs = [];
  statusRows = [];
  startImpl = null;
  invoke.mockReset();
  invoke.mockImplementation(async (cmd: string, args: { config?: RustSpeechPipelineConfig; pipelineId?: string; epoch?: number } | undefined) => {
    switch (cmd) {
      case "plugin:speech|start_pipeline":
        configs.push(args!.config!);
        return (startImpl ?? defaultStart)(args!.config!);
      case "plugin:speech|stop_pipeline":
        expect(args?.pipelineId).toBe(STT_PIPELINE_ID);
        // Rust: a stop that names an epoch stops only while that epoch is the key's current session.
        if (args?.epoch !== undefined && !statusRows.some((r) => r.pipelineId === STT_PIPELINE_ID && r.epoch === args.epoch))
          return undefined;
        stops++;
        return undefined;
      case "plugin:speech|get_pipeline_status":
        return statusRows;
      default:
        throw new Error(`unexpected command ${cmd}`);
    }
  });
});

afterEach(() => {
  vi.useRealTimers();
});

describe("RustUtteranceService (speech pipeline driver)", () => {
  it("starts the pipeline with the backend's engine as its utterance engine", async () => {
    const r = receiver();
    const engine = await startedEngine(r, {
      openai_audio: { ...Service_STT_Schema.parse({}).openai_audio, device: "Microphone (USB)" },
      vad: { ...Service_STT_Schema.parse({}).vad, minSilenceMs: "700" },
    });
    expect(configs).toHaveLength(1);
    const c = configs[0];
    expect(c).toMatchObject({
      pipelineId: "mic",
      source: "Microphone (USB)",
      language: "en",
      prompt: "VRChat",
      streaming: null,
      final: SETUP.engine,
      smartTurn: false,
      hallucinationFilter: true,
      latencyBudgetMs: HTTP_LATENCY_BUDGET_MS,
      maxBacklog: HTTP_MAX_BACKLOG,
      retainPcm: false,
    });
    expect(c.vad?.min_silence_ms).toBe(700);
    expect(r.onStart).toHaveBeenCalledTimes(1);
    // The connecting line is shown while starting and cleared once listening; stale start statuses never win.
    expect(translated(r.onStatus.mock.calls[0][0])).toBe(true);
    expect(r.onStatus.mock.calls[0][0]).toContain("TestSTT");
    expect(r.onStatus).toHaveBeenLastCalledWith("");
    expect(r.onStop).not.toHaveBeenCalled();
    await engine.stop();
  });

  it("publishes each segment's final once, with segment metadata, and ignores other sessions", async () => {
    const r = receiver();
    const engine = await startedEngine(r);
    const e = epoch;
    emit("speech:final", final(e, 3, " hello [noise] there ", { revision: 1, lang: "en" }));
    emit("speech:final", final(e, 3, "hello again", { revision: 2 })); // same segment: one final per segment
    emit("speech:final", final(e - 1, 4, "stale session"));
    emit("speech:final", { ...final(e, 5, "other pipeline"), pipelineId: "loopback" });
    emit("speech:final", final(e, 6, "[noise]")); // nothing left after post-processing
    emit("speech:final", final(e, 7, "second", { lang: null, turnEnd: true }));
    expect(r.onFinal.mock.calls).toEqual([
      ["hello  there", { segmentId: `mic-${PROCESS_NONCE}:${e}:3`, revision: 1, lang: "en" }],
      ["second", { segmentId: `mic-${PROCESS_NONCE}:${e}:7`, revision: 0, turnEnd: true }],
    ]);
    expect(r.onInterim).not.toHaveBeenCalled();
    await engine.stop();
  });

  it("holds events that arrive before the epoch is known and replays only this session's", async () => {
    startImpl = async () => {
      const e = ++epoch;
      emit("speech:status", status(e - 1, "stopped", "stopped")); // the previous session's last event, late
      emit("speech:status", status(e, "starting", "starting"));
      emit("speech:final", final(e - 1, 1, "old session"));
      emit("speech:status", status(e, "listening", "listening"));
      emit("speech:status", status(e, "listening", "rate_limited", "waiting 12 s for the request limit", "warning"));
      return started(e);
    };
    const r = receiver();
    const engine = await startedEngine(r);
    expect(r.onFinal).not.toHaveBeenCalled();
    expect(r.onStop).not.toHaveBeenCalled();
    expect(r.onStart).toHaveBeenCalledTimes(1);
    const warning = r.onStatus.mock.calls.find((c) => String(c[0]).includes("12 s"));
    expect(warning?.[1]).toMatchObject({ level: "warning" });
    await engine.stop();
  });

  it("status warnings and lost speech go to the status line only; merged or empty utterances stay quiet", async () => {
    const r = receiver();
    const engine = await startedEngine(r);
    const e = epoch;
    r.onStatus.mockClear();
    emit("speech:status", status(e, "listening", "rate_limited", "waiting 30 s for the request limit", "warning"));
    const [text, options] = r.onStatus.mock.calls[0];
    expect(text).toContain("30 s");
    expect(options).toMatchObject({ level: "warning" });
    expect(options.expireMs).toBeGreaterThanOrEqual(35_000);

    emit("speech:status", status(e, "listening", "engine_error", "HTTP transcription: HTTP 503; retrying (1/4)", "warning"));
    expect(r.onStatus).toHaveBeenLastCalledWith(
      expect.stringContaining("retrying (1/4)"),
      { level: "warning", expireMs: TRANSIENT_WARNING_MS },
    );
    emit("speech:status", status(e, "listening", "capture_gap", "the capture restarted", "warning"));
    expect(translated(r.onStatus.mock.lastCall?.[0])).toBe(true);

    r.onStatus.mockClear();
    emit("speech:utterance", dropped(e, 8, "merged"));
    emit("speech:utterance", dropped(e, 9, "no_speech"));
    emit("speech:utterance", dropped(e, 10, "muted"));
    emit("speech:utterance", { kind: "speech_start", pipelineId: "mic", epoch: e, segmentId: 11, startMs: 0 });
    expect(r.onStatus).not.toHaveBeenCalled();
    for (const reason of ["backlog", "latency_budget", "rate_limited", "engine_error"]) {
      emit("speech:utterance", dropped(e, 12, reason));
      const [line, opts] = r.onStatus.mock.lastCall!;
      expect(translated(line)).toBe(true);
      expect(opts).toEqual({ level: "warning", expireMs: TRANSIENT_WARNING_MS });
    }
    expect(r.onInterim).not.toHaveBeenCalled();
    expect(r.onFinal).not.toHaveBeenCalled();
    await engine.stop();
  });

  it("a silent microphone goes to the receiver's own channel, never a status line, and ends with the session", async () => {
    const r = receiver();
    const engine = await startedEngine(r);
    const e = epoch;
    r.onStatus.mockClear();
    r.onInputSilence.mockClear();
    emit("speech:status", status(e, "listening", "no_signal", "Headset Mic", "warning"));
    expect(r.onInputSilence).toHaveBeenLastCalledWith(true, "Headset Mic");
    emit("speech:status", status(e - 1, "listening", "no_signal", "Old Mic", "warning")); // another session
    expect(r.onInputSilence).toHaveBeenCalledTimes(1);
    emit("speech:status", status(e, "listening", "signal_restored", "", "info"));
    expect(r.onInputSilence).toHaveBeenLastCalledWith(false, "");
    expect(r.onStatus).not.toHaveBeenCalled();
    expect(speechStatusLine(status(e, "listening", "no_signal", "Headset Mic", "warning"), "TestSTT")).toBeUndefined();
    expect(speechStatusLine(status(e, "listening", "signal_restored"), "TestSTT")).toBeUndefined();

    emit("speech:status", status(e, "listening", "no_signal", "", "warning"));
    expect(r.onInputSilence).toHaveBeenLastCalledWith(true, "");
    await engine.stop();
    expect(r.onInputSilence).toHaveBeenLastCalledWith(false);
  });

  it("an utterance given up to the request limit stays reported as skipped, not as a wait", async () => {
    const r = receiver();
    const engine = await startedEngine(r);
    const e = epoch;
    r.onStatus.mockClear();
    // Rust's worker order when a wait would pass the budget: `dropped` / `rate_limited`, then the warning.
    emit("speech:utterance", dropped(e, 20, "rate_limited"));
    emit(
      "speech:status",
      status(e, "listening", "rate_limited", "an utterance was dropped: it would have waited past the latency budget (rate_limited: the request limit allows the next request in 431.9 s)", "warning"),
    );
    // A failed request's own error (Groq's 429 text holds "12.5s", which is not a wait of 5 s).
    emit("speech:utterance", dropped(e, 21, "rate_limited"));
    emit("speech:status", status(e, "listening", "rate_limited", "rate_limited: HTTP 429: Please try again in 7m12.5s", "warning"));
    const skipped = "Some speech was skipped because the request limit was reached.";
    expect(r.onStatus.mock.calls.map((c) => c[0])).toEqual([skipped, skipped, skipped, skipped]);
    for (const call of r.onStatus.mock.calls) expect(call[1]).toEqual({ level: "warning", expireMs: TRANSIENT_WARNING_MS });
    await engine.stop();
  });

  it("a fatal status ends the session with a message, and a pipeline Rust ended is never stopped again", async () => {
    const r = receiver();
    const engine = await startedEngine(r);
    emit("speech:status", status(epoch, "error", "engine_error", "HTTP transcription: HTTP 401 Unauthorized: Invalid API Key", "error"));
    await vi.waitFor(() => expect(r.onStop).toHaveBeenCalledTimes(1));
    const message = r.onStop.mock.calls[0][0] as string;
    expect(message).toContain("TestSTT");
    expect(message).toContain("Invalid API Key");
    expect(r.onStatus).toHaveBeenLastCalledWith("");
    // Later events of the ended session are ignored, and stopping sends nothing.
    emit("speech:final", final(epoch, 1, "late"));
    await engine.stop();
    expect(r.onFinal).not.toHaveBeenCalled();
    expect(stops).toBe(0);
    expect(r.onStop).toHaveBeenCalledTimes(1);
  });

  it("a used-up request limit names the wait in minutes", async () => {
    const r = receiver();
    await startedEngine(r);
    emit(
      "speech:status",
      status(epoch, "error", "rate_limited", "5 utterances failed in a row; last error: the request limit is used up for now (the next request would wait 42 min)", "error"),
    );
    await vi.waitFor(() => expect(r.onStop).toHaveBeenCalled());
    expect(r.onStop.mock.calls[0][0]).toContain("42");
  });

  it("a failed start tells a microphone problem from an engine problem", async () => {
    startImpl = async () => {
      const e = ++epoch;
      emit("speech:status", status(e, "error", "capture_error", "no input device named 'USB'", "error"));
      throw "no input device named 'USB'";
    };
    const r = receiver();
    const engine = await startedEngine(r);
    expect(r.onStart).not.toHaveBeenCalled();
    expect(r.onStop).toHaveBeenCalledTimes(1);
    expect(r.onStop.mock.calls[0][0]).toMatch(/^Microphone capture failed: no input device named 'USB'/);
    expect(stops).toBe(0);
    await engine.stop();
    expect(stops).toBe(0);

    startImpl = async () => {
      ++epoch;
      throw "the Moonshine server does not answer";
    };
    const r2 = receiver();
    await startedEngine(r2);
    expect(r2.onStop.mock.calls[0][0]).toBe("TestSTT could not start: the Moonshine server does not answer");
  });

  it("a fatal status that may pass (a network failure, a busy microphone, a rate limit) is flagged for a reconnect; one that needs the user is not", async () => {
    const cases: [code: string, detail: string, retry: SttStopOptions | false][] = [
      ["engine_error", "5 utterances failed in a row; last error: network: connection refused (failed 3 times)", { retry: true }],
      // A page a captive portal or a proxy answered with: the next request, on a working network, is a normal one.
      ["engine_error", "5 utterances failed in a row; last error: network: HTTP transcription: the server returned a response that is not JSON (a login page or proxy in the way?) (failed 4 times)", { retry: true }],
      ["capture_error", "no input device named 'USB'", { retry: true }],
      ["engine_error", "invalid_input: HTTP transcription: HTTP 401 Unauthorized: Invalid API Key", false],
      ["engine_error", "unsupported: no such model", false],
      // Groq's per-minute limit ends by itself; it waits the slow pace, a quick retry only extends it.
      ["rate_limited", "5 utterances failed in a row; last error: the request limit is used up for now (the next request would wait 42 min)", { retry: true, slow: true }],
      ["not_installed", "stt/whatever", false],
      // A model nobody chose (bring-your-own) is not installed by a restart, with or without a catalog id.
      ["engine_error", "not_installed: Model whisper/custom is bring-your-own: choose its .bin file", false],
    ];
    for (const [code, detail, retry] of cases) {
      const r = receiver();
      await startedEngine(r);
      emit("speech:status", status(epoch, "error", code, detail, "error"));
      await vi.waitFor(() => expect(r.onStop).toHaveBeenCalledTimes(1), { timeout: 2_000 });
      const call = r.onStop.mock.calls[0];
      if (retry) expect(call[1], `${code}: ${detail}`).toEqual(retry);
      else expect(call, `${code}: ${detail}`).toHaveLength(1);
    }
  });

  it("a failed start is flagged for a reconnect unless a restart cannot fix it", async () => {
    const cases: [error: string, retry: boolean][] = [
      ["Could not open the audio device 'USB'", true],
      ["network: the Moonshine server does not answer at http://127.0.0.1:8000", true],
      ["invalid_input: the server address is not valid", false],
      ["no speech engine for this request (language xx)", false],
      ["cancelled", false],
    ];
    for (const [error, retry] of cases) {
      startImpl = async () => {
        ++epoch;
        throw error;
      };
      const r = receiver();
      await startedEngine(r);
      expect(r.onStop, error).toHaveBeenCalledTimes(1);
      const call = r.onStop.mock.calls[0];
      if (retry) expect(call[1], error).toEqual({ retry: true });
      else expect(call, error).toHaveLength(1);
    }
  });

  it("prepare errors stop before any listener or pipeline exists", async () => {
    const r = receiver();
    const engine = new TestEngine(r);
    engine.setup = "Enter a URL.";
    await engine.start(Service_STT_Schema.parse({}));
    expect(r.onStop).toHaveBeenCalledWith("Enter a URL.");
    expect(configs).toHaveLength(0);
    expect(listeners.size).toBe(0);
  });

  it("stop stops the pipeline once, removes the listeners and ignores later events", async () => {
    const r = receiver();
    const engine = await startedEngine(r);
    expect(listeners.size).toBe(3);
    await engine.stop();
    await engine.stop();
    expect(stops).toBe(1);
    expect(listeners.size).toBe(0);
    expect(r.onStop).toHaveBeenCalledTimes(1);
    expect(r.onStop.mock.calls[0]).toEqual([undefined]);
  });

  it("a session replaced from elsewhere ends quietly without stopping the new owner", async () => {
    const r = receiver();
    const engine = await startedEngine(r);
    emit("speech:status", status(epoch, "stopped", "stopped"));
    await vi.waitFor(() => expect(r.onStop).toHaveBeenCalledWith(undefined));
    await engine.stop();
    expect(stops).toBe(0);
  });

  it("a stop during start cancels it, and a start that reached Rust after the stop is stopped when it resolves", async () => {
    const pending = deferred<RustSpeechPipelineStarted>();
    startImpl = () => pending.promise;
    const r = receiver();
    const engine = new TestEngine(r);
    const starting = engine.start(Service_STT_Schema.parse({}));
    await vi.waitFor(() => expect(configs).toHaveLength(1));
    const stopping = engine.stop();
    await vi.waitFor(() => expect(stops).toBe(1));
    pending.resolve(started(++epoch)); // Rust ran the start after the stop: a live session
    await stopping;
    await starting;
    expect(stops).toBe(2);
    expect(r.onStart).not.toHaveBeenCalled();
    expect(r.onStop).toHaveBeenCalledTimes(1);
  });

  it("a start the stop cancelled in Rust is not stopped twice", async () => {
    const pending = deferred<RustSpeechPipelineStarted>();
    startImpl = () => pending.promise;
    const r = receiver();
    const engine = new TestEngine(r);
    const starting = engine.start(Service_STT_Schema.parse({}));
    await vi.waitFor(() => expect(configs).toHaveLength(1));
    const stopping = engine.stop();
    await vi.waitFor(() => expect(stops).toBe(1));
    pending.reject("cancelled");
    await stopping;
    await starting;
    expect(stops).toBe(1);
    expect(r.onStart).not.toHaveBeenCalled();
  });

  it("gives up waiting for a hung start, then stops it later only while it still owns the pipeline key", async () => {
    vi.useFakeTimers();
    for (const owner of ["late start", "newer session"] as const) {
      stops = 0;
      const pending = deferred<RustSpeechPipelineStarted>();
      startImpl = () => pending.promise;
      const engine = new TestEngine(receiver());
      void engine.start(Service_STT_Schema.parse({}));
      await vi.advanceTimersByTimeAsync(0);
      const stopping = engine.stop();
      await vi.advanceTimersByTimeAsync(START_SETTLE_WAIT_MS + 1);
      await stopping; // the orchestrator is not blocked forever
      expect(stops).toBe(1);
      const e = ++epoch;
      statusRows = [
        {
          pipelineId: "mic",
          epoch: owner === "late start" ? e : e + 1,
          state: "listening",
          source: "",
          streaming: null,
          final: null,
          backlog: 0,
          dropped: 0,
        },
      ];
      pending.resolve(started(e));
      await vi.advanceTimersByTimeAsync(0);
      expect(stops).toBe(owner === "late start" ? 2 : 1);
      // The late stop names its session, so Rust itself refuses it once another session owns the key.
      const lateStop = invoke.mock.calls.filter(([cmd]) => cmd === "plugin:speech|stop_pipeline").at(-1)?.[1];
      expect(lateStop).toEqual({ pipelineId: STT_PIPELINE_ID, epoch: e });
    }
  });

  it("never sends the late stop once a newer start was issued (it replaces the late session and must not be cancelled)", async () => {
    vi.useFakeTimers();
    stops = 0;
    const pending = deferred<RustSpeechPipelineStarted>();
    startImpl = () => pending.promise;
    const engine = new TestEngine(receiver());
    void engine.start(Service_STT_Schema.parse({}));
    await vi.advanceTimersByTimeAsync(0);
    const stopping = engine.stop();
    await vi.advanceTimersByTimeAsync(START_SETTLE_WAIT_MS + 1);
    await stopping;
    expect(stops).toBe(1);
    const e = ++epoch;
    // Rust still reports the late session as the key's own (the newer start is queued behind it): only the newer
    // start's note tells it apart, and an epoch stop now would still stop the late session the newer start replaces.
    statusRows = [{ pipelineId: "mic", epoch: e, state: "listening", source: "", streaming: null, final: null, backlog: 0, dropped: 0 }];
    noteSttPipelineStart(); // e.g. the next backend's start, queued in Rust behind the hung one
    const sent = invoke.mock.calls.length;
    pending.resolve(started(e));
    await vi.advanceTimersByTimeAsync(0);
    expect(stops).toBe(1);
    expect(invoke.mock.calls.slice(sent).filter(([cmd]) => cmd === "plugin:speech|stop_pipeline")).toEqual([]);
  });
});
