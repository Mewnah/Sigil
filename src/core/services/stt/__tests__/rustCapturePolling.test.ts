import i18n from "i18next";
import { afterEach, beforeAll, beforeEach, describe, expect, it, vi } from "vitest";
import type { RustUtteranceEvent } from "@/types";

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
vi.mock("@/utils/rustAudioInput", () => ({ resolveRustInputDeviceForInvoke: () => null }));

import { RustUtteranceService } from "../rustCapturePolling";
import type { RequestLimiter } from "../rateLimiter";
import { Service_STT_Schema, type STT_State } from "../schema";
import { TRANSIENT_WARNING_MS } from "../statusChannel";
import type { ISTTReceiver } from "../types";

const SR = 16_000;
const CAPTURE = 7;
/** Utterance length in seconds per seq, served by the fake `take_utterance` (default 2 s). */
const lengths = new Map<number, number>();

class TestEngine extends RustUtteranceService {
  protected readonly logTag = "[Test]";
  /** Seqs of each request (every sample of an utterance carries its seq). */
  readonly requests: number[][] = [];
  limiterForTest: RequestLimiter | null = null;
  protected engineLabel() {
    return "Test";
  }
  protected async prepare(_params: STT_State) {
    return null;
  }
  protected inputDevice() {
    return "";
  }
  protected limiter() {
    return this.limiterForTest;
  }
  protected async transcribe(samples: Float32Array) {
    const seqs: number[] = [];
    for (const v of samples) if (v !== 0 && seqs[seqs.length - 1] !== v) seqs.push(v);
    this.requests.push(seqs);
    return `text ${seqs.join("+")}`;
  }
}

function receiver(muted: () => boolean = () => false) {
  return {
    onStart: vi.fn(),
    onStop: vi.fn(),
    onInterim: vi.fn(),
    onFinal: vi.fn(),
    onStatus: vi.fn(),
    isMuted: vi.fn(muted),
  } satisfies ISTTReceiver;
}

function emit(event: RustUtteranceEvent) {
  listeners.get("audio:utterance")!({ payload: event });
}

const utteranceEvent = (seq: number, seconds: number, forced = false): RustUtteranceEvent => ({
  kind: "utterance",
  capture_id: CAPTURE,
  seq,
  start_ms: 0,
  end_ms: seconds * 1000,
  sample_rate: SR,
  samples: seconds * SR,
  forced,
});

async function started(r: ISTTReceiver, limiter: RequestLimiter | null = null) {
  const engine = new TestEngine(r);
  engine.limiterForTest = limiter;
  await engine.start(Service_STT_Schema.parse({}));
  return engine;
}

beforeAll(async () => {
  await i18n.init({ lng: "en", resources: { en: { translation: {} } } });
});

beforeEach(() => {
  vi.useFakeTimers();
  listeners.clear();
  lengths.clear();
  invoke.mockReset();
  invoke.mockImplementation(async (cmd: string, args: { seq?: number } | undefined) => {
    if (cmd === "plugin:audio|start_utterance_capture") return CAPTURE;
    if (cmd === "plugin:audio|take_utterance") {
      const seq = args?.seq ?? 0;
      return new Float32Array(Math.round((lengths.get(seq) ?? 2) * SR)).fill(seq).buffer;
    }
    return undefined;
  });
});

afterEach(() => {
  vi.useRealTimers();
});

describe("RustUtteranceService", () => {
  it("sends one request per utterance and publishes the transcript as a final", async () => {
    const r = receiver();
    const engine = await started(r);
    expect(r.onStart).toHaveBeenCalledTimes(1);
    emit(utteranceEvent(1, 2));
    await vi.advanceTimersByTimeAsync(10);
    emit(utteranceEvent(2, 2));
    await vi.advanceTimersByTimeAsync(10);
    expect(engine.requests).toEqual([[1], [2]]);
    expect(r.onFinal.mock.calls).toEqual([["text 1"], ["text 2"]]);
    await engine.stop();
  });

  it("skips (never sends) utterances while muted", async () => {
    let muted = true;
    const r = receiver(() => muted);
    const engine = await started(r);
    emit(utteranceEvent(1, 2));
    await vi.advanceTimersByTimeAsync(10);
    muted = false;
    emit(utteranceEvent(2, 2));
    await vi.advanceTimersByTimeAsync(10);
    expect(engine.requests).toEqual([[2]]);
    await engine.stop();
  });

  it("a dropped utterance shows a warning that clears itself", async () => {
    const r = receiver();
    const engine = await started(r);
    emit({ kind: "dropped", capture_id: CAPTURE, seq: 3, reason: "backlog" });
    expect(r.onStatus).toHaveBeenLastCalledWith(expect.any(String), { level: "warning", expireMs: TRANSIENT_WARNING_MS });
    await engine.stop();
  });

  it("billing minimum: speech_start keeps a short batch waiting for the utterance being spoken", async () => {
    const r = receiver();
    const engine = await started(r, { waitMs: () => 0, record: () => {}, minBatchSeconds: () => 10 });
    emit({ kind: "speech_start", capture_id: CAPTURE, seq: 1, start_ms: 0 });
    emit(utteranceEvent(1, 2));
    await vi.advanceTimersByTimeAsync(500);
    emit({ kind: "speech_start", capture_id: CAPTURE, seq: 2, start_ms: 2_500 });
    await vi.advanceTimersByTimeAsync(5_000); // past the plain 3.5 s hold: seq 2 is still being spoken
    expect(engine.requests).toEqual([]);
    lengths.set(2, 3);
    emit(utteranceEvent(2, 3));
    await vi.advanceTimersByTimeAsync(10);
    expect(engine.requests).toEqual([[1, 2]]);
    expect(r.onFinal).toHaveBeenCalledWith("text 1+2");
    await engine.stop();
  });

  it("billing minimum: a muted (skipped) utterance releases the hold instead of stretching it", async () => {
    let muted = false;
    const r = receiver(() => muted);
    const engine = await started(r, { waitMs: () => 0, record: () => {}, minBatchSeconds: () => 10 });
    emit(utteranceEvent(1, 2));
    await vi.advanceTimersByTimeAsync(10);
    emit({ kind: "speech_start", capture_id: CAPTURE, seq: 2, start_ms: 2_500 });
    await vi.advanceTimersByTimeAsync(4_000);
    expect(engine.requests).toEqual([]);
    muted = true;
    emit(utteranceEvent(2, 2));
    await vi.advanceTimersByTimeAsync(10);
    expect(engine.requests).toEqual([[1]]);
    await engine.stop();
  });
});
