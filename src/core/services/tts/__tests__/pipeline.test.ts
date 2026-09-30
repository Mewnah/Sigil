import i18n from "i18next";
import { proxy } from "valtio";
import { afterEach, beforeAll, beforeEach, describe, expect, it, vi } from "vitest";

vi.mock("@tauri-apps/api/core", () => ({ invoke: vi.fn(), isTauri: () => false }));
vi.mock("@tauri-apps/api/event", () => ({ listen: vi.fn(async () => () => {}) }));
vi.mock("@/utils/devLog", () => ({ devLog: () => {} }));
vi.mock("react-toastify", () => ({
  toast: Object.assign(vi.fn(), { error: vi.fn(), info: vi.fn(), warning: vi.fn(), success: vi.fn() }),
}));
vi.mock("../../systemLog", () => ({ pushSystemLog: vi.fn() }));

/** Azure SDK stand-in whose synthesizer exposes its `synthesizing` handler (streamed chunks). */
const azure = vi.hoisted(() => ({
  synths: [] as {
    ssml: string[];
    cb: ((r: unknown) => void)[];
    self: { synthesizing?: (s: unknown, e: { result: { audioData: ArrayBuffer } }) => void };
  }[],
}));
vi.mock("microsoft-cognitiveservices-speech-sdk", async (importOriginal) => {
  const actual = await importOriginal<typeof import("microsoft-cognitiveservices-speech-sdk")>();
  class FakeSynthesizer {
    synthesizing?: (s: unknown, e: { result: { audioData: ArrayBuffer } }) => void;
    rec: (typeof azure.synths)[number];
    constructor() {
      this.rec = { ssml: [], cb: [], self: this };
      azure.synths.push(this.rec);
    }
    speakSsmlAsync(ssml: string, cb: (r: unknown) => void) {
      this.rec.ssml.push(ssml);
      this.rec.cb.push(cb);
    }
    close() {}
  }
  return {
    ...actual,
    SpeechConfig: { fromSubscription: () => ({}) },
    SpeechSynthesizer: FakeSynthesizer,
    Connection: {
      fromSynthesizer: () => ({
        openConnection: (cb: () => void) => void Promise.resolve().then(cb),
        close() {},
      }),
    },
  };
});

import { invoke } from "@tauri-apps/api/core";
import { ResultReason } from "microsoft-cognitiveservices-speech-sdk";
import { ServiceNetworkState, TextEventType, type RustTtsSinkTarget } from "@/types";
import { TtsSinkClient } from "../../sound/sink";
import { FakeSinkTransport, flush, makeWav } from "../../sound/__tests__/fakeSink";
import Service_TTS, { type TTSBackendMap } from "../index";
import { TtsPipeline, ttsSinkTarget, type TtsPipelineOptions } from "../pipeline";
import { Service_TTS_Schema, TTS_Backends, TTS_Routes, type TTS_State } from "../schema";
import { migrateTtsRoute } from "../schema/route";
import { TTS_AzureService } from "../services/azure";
import { TTS_UberduckService } from "../services/uberduck";
import { TTS_VoicevoxService } from "../services/voicevox";
import type { ITTSReceiver, ITTSService, PcmChunk, TtsSinkContext } from "../types";

const invokeMock = vi.mocked(invoke);

beforeAll(async () => {
  await i18n.init({ lng: "en", resources: { en: { translation: {} } }, interpolation: { escapeValue: false } });
});

/** Settles `synthesizePcm` per sentence from the test. */
class ScriptedPcmBackend implements ITTSService {
  calls: { text: string; signal: AbortSignal }[] = [];
  #pending = new Map<string, { resolve: (chunks: PcmChunk[]) => void; reject: (e: unknown) => void }>();

  start() {}
  stop() {}
  play() {}
  dispose() {}

  async *synthesizePcm(text: string, signal: AbortSignal): AsyncIterable<PcmChunk> {
    this.calls.push({ text, signal });
    const chunks = await new Promise<PcmChunk[]>((resolve, reject) => {
      this.#pending.set(text, { resolve, reject });
      signal.addEventListener("abort", () => reject(new DOMException("Aborted", "AbortError")), { once: true });
    });
    for (const c of chunks) yield c;
  }

  finish(text: string, ...values: number[]) {
    this.#pending.get(text)?.resolve([{ samples: new Float32Array(values.length ? values : [0.5]), sampleRate: 24_000, channels: 1 }]);
  }

  fail(text: string, error: unknown) {
    this.#pending.get(text)?.reject(error);
  }
}

function makePipeline(service: ITTSService, opts: Partial<TtsPipelineOptions> = {}) {
  const fake = new FakeSinkTransport();
  const sink = new TtsSinkClient(fake);
  const errors: string[] = [];
  const states: { pending: number; playing: boolean }[] = [];
  const target: RustTtsSinkTarget = { route: "device", device: "Speakers", volume: 1 };
  const pipeline = new TtsPipeline(service, {
    sink,
    target: () => target,
    onError: (e) => errors.push(e),
    onChange: (s) => states.push(s),
    ...opts,
  });
  return { fake, sink, pipeline, errors, states };
}

const THREE = "The first sentence is here. The second sentence is here. The third sentence is here.";
const S1 = "The first sentence is here.";
const S2 = "The second sentence is here.";
const S3 = "The third sentence is here.";

describe("TtsPipeline", () => {
  it("synthesizes sentence one alone, so its audio starts before later sentences exist", async () => {
    const backend = new ScriptedPcmBackend();
    const { fake, pipeline } = makePipeline(backend);
    expect(pipeline.speak(THREE)).toBe(3);
    await flush();
    expect(backend.calls.map((c) => c.text)).toEqual([S1]);
    // The session's first sink call drops what an earlier webview session may have left behind.
    expect(fake.commands()).toEqual(["flush_sink", "begin_utterance"]);
    backend.finish(S1, 0.25, -0.25);
    await flush(40);
    // Sentence one is pushed and ended (the sink plays it now) before sentence two is even begun.
    const [id1, id2] = fake.order;
    expect(fake.commands().slice(1, 5)).toEqual(["begin_utterance", "push_pcm", "end_utterance", "begin_utterance"]);
    expect(fake.audio(id1)).toEqual([0.25, -0.25]);
    expect(fake.utterances.get(id1)?.ended).toBe(true);
    expect(fake.utterances.get(id1)?.target).toEqual({ route: "device", device: "Speakers", volume: 1 });
    expect(backend.calls.map((c) => c.text)).toEqual([S1, S2]);
    expect(id2).toBeDefined();
  });

  it("stays at most `ahead` sentences ahead of the one playing, and begins in sentence order", async () => {
    const backend = new ScriptedPcmBackend();
    const { fake, pipeline } = makePipeline(backend, { ahead: 1 });
    pipeline.speak(THREE);
    await flush();
    backend.finish(S1);
    await flush(40);
    backend.finish(S2);
    await flush(40);
    // Sentence 1 still playing, sentence 2 synthesized: sentence 3 waits.
    expect(backend.calls.map((c) => c.text)).toEqual([S1, S2]);
    fake.play(fake.order[0]);
    fake.done(fake.order[0]);
    await flush(40);
    expect(backend.calls.map((c) => c.text)).toEqual([S1, S2, S3]);
    backend.finish(S3);
    await flush(40);
    expect(fake.order.map((id) => fake.utterances.get(id)?.pushes.length)).toEqual([1, 1, 1]);
    // Ids rise in sentence order.
    const seq = fake.order.map((id) => Number(id.split("-").pop()));
    expect(seq).toEqual([...seq].sort((a, b) => a - b));
    expect(pipeline.state.pending).toBe(2);
  });

  it("skip cancels the sentence playing now and the next one follows", async () => {
    const backend = new ScriptedPcmBackend();
    const { fake, pipeline, errors } = makePipeline(backend);
    pipeline.speak(THREE);
    await flush();
    backend.finish(S1);
    await flush(40);
    const [id1] = fake.order;
    fake.play(id1);
    expect(pipeline.state.playing).toBe(true);
    pipeline.skip();
    await flush(40);
    expect(fake.calls.some((c) => c.cmd.endsWith("cancel_utterance") && (c.args as { id: string }).id === id1)).toBe(true);
    expect(pipeline.state.pending).toBe(2);
    backend.finish(S2);
    await flush(40);
    expect(fake.utterances.get(fake.order[1])?.ended).toBe(true);
    expect(errors).toEqual([]);
  });

  it("stop cancels everything, aborts synthesis in flight without an error, and the pipeline stays usable", async () => {
    const backend = new ScriptedPcmBackend();
    const { fake, pipeline, errors, states } = makePipeline(backend);
    pipeline.speak(THREE);
    await flush();
    backend.finish(S1);
    await flush(40);
    const inFlight = backend.calls[1];
    expect(inFlight.text).toBe(S2);
    pipeline.stop();
    await flush(40);
    expect(inFlight.signal.aborted).toBe(true);
    expect(pipeline.state.pending).toBe(0);
    expect(states.at(-1)).toEqual({ pending: 0, playing: false });
    const cancelled = fake.calls.filter((c) => c.cmd.endsWith("cancel_utterance")).map((c) => (c.args as { id: string }).id);
    expect(new Set(cancelled)).toEqual(new Set(fake.order));
    expect(backend.calls.length).toBe(2);
    expect(errors).toEqual([]);
    // Usable again with fresh ids.
    pipeline.speak("A brand new sentence to say.");
    await flush();
    expect(backend.calls.at(-1)?.text).toBe("A brand new sentence to say.");
    expect(new Set(fake.order).size).toBe(fake.order.length);
  });

  it("a sentence that fails is reported and skipped; the sink's own cancel is not an error", async () => {
    const backend = new ScriptedPcmBackend();
    const { fake, pipeline, errors } = makePipeline(backend);
    pipeline.speak(THREE);
    await flush();
    backend.fail(S1, new Error("VOICEVOX: engine exploded"));
    await flush(40);
    expect(errors).toEqual(["VOICEVOX: engine exploded"]);
    expect(fake.utterances.get(fake.order[0])?.state).toBe("cancelled");
    // The next sentence goes on; its push failing with "cancelled" is not an error of ours. The utterance is still
    // alive in the sink here (as when an engine gave up by itself), so the pipeline cancels it: no error, no leak.
    expect(backend.calls.at(-1)?.text).toBe(S2);
    fake.failNext.set("plugin:tts-sink|push_pcm", "cancelled");
    backend.finish(S2);
    await flush(40);
    expect(errors).toEqual(["VOICEVOX: engine exploded"]);
    expect(fake.utterances.get(fake.order[1])?.state).toBe("cancelled");
    // An output error event is reported once and the pipeline moves on.
    backend.finish(S3);
    await flush(40);
    fake.fail(fake.order[2], "audio output: the device was removed");
    await flush(20);
    expect(errors).toEqual(["VOICEVOX: engine exploded", "audio output: the device was removed"]);
    expect(pipeline.state.pending).toBe(0);
  });

  it("reports a sink-side failure once even when the push rejection arrives before the error event", async () => {
    const backend = new ScriptedPcmBackend();
    const { fake, pipeline, errors } = makePipeline(backend);
    const text = "A sentence whose output device goes away.";
    pipeline.speak(text);
    await flush();
    const [id] = fake.order;
    // The sink dropped the utterance (output failure, stall); the push fails with "cancelled" before the
    // utterance's error event reaches the webview.
    fake.utterances.get(id)!.state = "error";
    fake.failNext.set("plugin:tts-sink|push_pcm", "cancelled");
    backend.finish(text);
    await flush(40);
    expect(errors).toEqual([]);
    expect(pipeline.state.pending).toBe(1);
    fake.emit({ id, state: "error", error: "audio output: the device was removed", playedMs: 0 });
    await flush(20);
    expect(errors).toEqual(["audio output: the device was removed"]);
    expect(pipeline.state.pending).toBe(0);
  });

  it("drops such a sentence through the sink status when its error event is lost", async () => {
    vi.useFakeTimers();
    try {
      const backend = new ScriptedPcmBackend();
      const { fake, pipeline, errors } = makePipeline(backend, { reconcileMs: 1_000 });
      const text = "A sentence whose error event is lost.";
      pipeline.speak(text);
      await vi.advanceTimersByTimeAsync(0);
      fake.utterances.get(fake.order[0])!.state = "error";
      fake.failNext.set("plugin:tts-sink|push_pcm", "cancelled");
      backend.finish(text);
      await vi.advanceTimersByTimeAsync(0);
      expect(pipeline.state.pending).toBe(1);
      await vi.advanceTimersByTimeAsync(1_000);
      expect(pipeline.state.pending).toBe(0);
      expect(errors).toEqual([]);
    } finally {
      vi.useRealTimers();
    }
  });

  it("a cancelled sentence frees the synthesis slot at once, even when its backend is slow to settle", async () => {
    const calls: string[] = [];
    const hung: ITTSService = {
      start() {},
      stop() {},
      play() {},
      dispose() {},
      async *synthesizePcm(text: string) {
        calls.push(text);
        // Ignores the abort signal and never settles (a request stuck in flight).
        if (text.startsWith("Slow")) await new Promise<never>(() => {});
        yield { samples: new Float32Array([0.5]), sampleRate: 24_000, channels: 1 };
      },
    };
    const { fake, pipeline, errors } = makePipeline(hung);
    // Skip while nothing plays yet: the synthesizing sentence is cancelled and the next one starts.
    pipeline.speak("Slow sentence that hangs forever here. Second sentence plays right away.");
    await flush(40);
    expect(calls).toEqual(["Slow sentence that hangs forever here."]);
    pipeline.skip();
    await flush(40);
    expect(calls).toEqual(["Slow sentence that hangs forever here.", "Second sentence plays right away."]);
    expect(fake.audio(fake.order[1])).toEqual([0.5]);
    expect(fake.utterances.get(fake.order[1])?.ended).toBe(true);
    // Stop speaking while a sentence hangs, then new text: synthesized at once.
    fake.done(fake.order[1]);
    pipeline.speak("Slow again and stuck in flight.");
    await flush(40);
    pipeline.stop();
    pipeline.speak("Fresh sentence after stopping.");
    await flush(40);
    expect(calls.at(-1)).toBe("Fresh sentence after stopping.");
    expect(fake.utterances.get(fake.order.at(-1)!)?.ended).toBe(true);
    expect(errors).toEqual([]);
  });

  it("drives in-process engines through synthesizeToSink with the begun utterance", async () => {
    const seen: { text: string; sink: TtsSinkContext }[] = [];
    let client!: TtsSinkClient;
    const engine: ITTSService = {
      start() {},
      stop() {},
      play() {},
      dispose() {},
      async synthesizeToSink(text, sink) {
        seen.push({ text, sink });
        // Rust would push straight into the utterance.
        await client.push(sink.utteranceId, { samples: new Float32Array([0.1]), sampleRate: 22_050, channels: 1 });
      },
    };
    const made = makePipeline(engine);
    client = made.sink;
    expect(made.pipeline.mode).toBe("sink");
    made.pipeline.speak("Only one sentence to speak here.");
    await flush(40);
    expect(seen).toEqual([
      {
        text: "Only one sentence to speak here.",
        sink: { utteranceId: made.fake.order[0], target: { route: "device", device: "Speakers", volume: 1 } },
      },
    ]);
    expect(made.fake.utterances.get(made.fake.order[0])?.ended).toBe(true);
    expect(seen[0].sink.lang).toBeUndefined();

    // A tagged text passes its language with every sentence (engines prefer it over the script guess).
    made.pipeline.speak("Hola a todos, bienvenidos al directo de esta noche. Hoy vamos a probar un juego nuevo.", "es");
    await flush(40);
    expect(seen.slice(1).map((s) => [s.text, s.sink.lang])).toEqual([
      ["Hola a todos, bienvenidos al directo de esta noche.", "es"],
      ["Hoy vamos a probar un juego nuevo.", "es"],
    ]);
  });

  it("recovers from a missed done event through the sink status", async () => {
    vi.useFakeTimers();
    try {
      const backend = new ScriptedPcmBackend();
      const { fake, pipeline } = makePipeline(backend, { reconcileMs: 1_000, ahead: 0 });
      pipeline.speak("The only sentence we will say here.");
      await vi.advanceTimersByTimeAsync(0);
      backend.finish("The only sentence we will say here.");
      await vi.advanceTimersByTimeAsync(0);
      const [id] = fake.order;
      expect(fake.utterances.get(id)?.ended).toBe(true);
      // The sink finished it but the event was lost.
      fake.utterances.get(id)!.state = "done";
      await vi.advanceTimersByTimeAsync(1_000);
      expect(pipeline.state.pending).toBe(0);
    } finally {
      vi.useRealTimers();
    }
  });

  it("refuses self-playing backends", () => {
    const selfPlaying: ITTSService = { start() {}, stop() {}, play() {}, dispose() {}, selfPlaying: true };
    expect(() => makePipeline(selfPlaying)).toThrow("plays by itself");
  });
});

describe("route setting", () => {
  it("migrates the active backend's old routeThroughVoiceChanger flag once", () => {
    expect(migrateTtsRoute({ backend: "neuralOnnx", neuralOnnx: { routeThroughVoiceChanger: true } })).toMatchObject({
      route: "voiceChanger",
    });
    expect(migrateTtsRoute({ backend: "neuralSidecar", neuralSidecar: { routeThroughVoiceChanger: true } })).toMatchObject({
      route: "voiceChanger",
    });
    // Another backend is active: it played on the device.
    expect(migrateTtsRoute({ backend: "azure", neuralOnnx: { routeThroughVoiceChanger: true } })).toMatchObject({ route: "device" });
    expect(migrateTtsRoute({ backend: "neuralOnnx", neuralOnnx: { routeThroughVoiceChanger: "true" } })).toMatchObject({
      route: "device",
    });
    // Already migrated (or chosen): untouched.
    const saved = { backend: "neuralOnnx", route: "both", neuralOnnx: { routeThroughVoiceChanger: false } };
    expect(migrateTtsRoute(saved)).toBe(saved);
    expect(migrateTtsRoute(null)).toBe(null);
    expect(migrateTtsRoute([1])).toEqual([1]);
  });

  it("is part of the persisted schema with a safe default", () => {
    expect(Service_TTS_Schema.parse({}).route).toBe(TTS_Routes.device);
    expect(Service_TTS_Schema.parse(undefined).route).toBe(TTS_Routes.device);
    expect(Service_TTS_Schema.parse({ route: "bogus" }).route).toBe(TTS_Routes.device);
    expect(Service_TTS_Schema.parse({ route: "both" }).route).toBe(TTS_Routes.both);
    const migrated = Service_TTS_Schema.parse({ backend: "neuralSidecar", neuralSidecar: { routeThroughVoiceChanger: true } });
    expect(migrated.route).toBe(TTS_Routes.voiceChanger);
    // The old flag stays readable (persisted keys are never removed).
    expect(migrated.neuralSidecar.routeThroughVoiceChanger).toBe(true);
  });

  it("builds the sink target from the route, the backend's device and volume", () => {
    const data = Service_TTS_Schema.parse({ backend: "azure", route: "both", azure: { device: " Headphones ", volume: "0.5" } });
    expect(ttsSinkTarget(data, "App Output")).toEqual({ route: "both", device: "Headphones", volume: 0.5 });
    const fallback = Service_TTS_Schema.parse({ backend: "uberduck", uberduck: { device: "", volume: "NaN" } });
    expect(ttsSinkTarget(fallback, " App Output ")).toEqual({ route: "device", device: "App Output", volume: 1 });
    // VOICEVOX has no volume field (the engine applies volumeScale): gain 1.
    const vv = Service_TTS_Schema.parse({ backend: "voicevox", voicevox: { device: "Cable" } });
    expect(ttsSinkTarget(vv, "")).toEqual({ route: "device", device: "Cable", volume: 1 });
  });
});

// ---------------------------------------------------------------------------------------------
// Service_TTS and real backends through the pipeline
// ---------------------------------------------------------------------------------------------

let data: TTS_State;
let fake: FakeSinkTransport;

function installWindow(extra: Record<string, unknown> = {}) {
  data = proxy(Service_TTS_Schema.parse({}));
  fake = new FakeSinkTransport();
  vi.stubGlobal("window", {
    ApiServer: {
      state: { services: { tts: { data } }, audioOutputDevice: "App Output" },
      sound: { sink: new TtsSinkClient(fake), enqueueVoiceClip: vi.fn(), enqueuePcm: vi.fn(), clearVoiceClips: vi.fn() },
      voiceChanger: { state: { isRunning: false } },
    },
    ApiShared: {
      pubsub: {
        subscribe: vi.fn(() => "sub"),
        unsubscribe: vi.fn(),
        subscribeText: vi.fn(() => "sub"),
      },
    },
    ...extra,
  });
}

function receiver() {
  return { onStart: vi.fn(), onStop: vi.fn(), onError: vi.fn() } satisfies ITTSReceiver;
}

beforeEach(() => {
  invokeMock.mockReset();
  azure.synths = [];
  installWindow();
});

afterEach(() => {
  vi.unstubAllGlobals();
});

describe("Service_TTS with the sentence pipeline", () => {
  it("speaks finals sentence by sentence, exposes progress, and skip / stop speaking / stop cancel", async () => {
    let backend!: ScriptedPcmBackend;
    class Backend extends ScriptedPcmBackend {
      constructor(public receiver: ITTSReceiver) {
        super();
        backend = this;
      }
      start() {
        this.receiver.onStart();
      }
    }
    const map = Object.fromEntries(Object.values(TTS_Backends).map((b) => [b, Backend])) as unknown as TTSBackendMap;
    const service = new Service_TTS(map);
    service.updateReplacementsCache();
    Object.assign(data, { backend: TTS_Backends.voicevox, route: TTS_Routes.both });
    data.voicevox.device = "Headphones";
    service.start();
    expect(service.serviceState.status).toBe(ServiceNetworkState.connected);
    service.play(THREE);
    await flush();
    expect(service.serviceState.pending).toBe(3);
    backend.finish(S1);
    await flush(40);
    expect(fake.utterances.get(fake.order[0])?.target).toEqual({ route: "both", device: "Headphones", volume: 1 });
    fake.play(fake.order[0]);
    expect(service.serviceState.speaking).toBe(true);
    service.skip();
    await flush(40);
    expect(service.serviceState.pending).toBe(2);
    service.stopSpeaking();
    await flush(20);
    expect(service.serviceState.pending).toBe(0);
    expect(service.serviceState.status).toBe(ServiceNetworkState.connected);
    service.play("Another line to say out loud.");
    await flush();
    expect(service.serviceState.pending).toBe(1);
    service.stop();
    expect(service.serviceState.pending).toBe(0);
    expect(backend.calls.at(-1)?.signal.aborted).toBe(true);
    // Finals from the source still work after a restart.
    expect(TextEventType.final).toBeDefined();
  });
});

describe("VOICEVOX through the pipeline", () => {
  it("decodes the engine's WAV into PCM for the sink, per sentence", async () => {
    const wav = makeWav([16_384, -16_384, 8_192], { rate: 24_000 });
    const urls: string[] = [];
    vi.stubGlobal("fetch", vi.fn(async (url: string) => {
      urls.push(url);
      if (url.endsWith("/version")) return new Response('"0.21.0"');
      if (url.includes("/audio_query")) return new Response(JSON.stringify({ accent_phrases: [] }));
      return new Response(wav);
    }));
    const r = receiver();
    const vv = new TTS_VoicevoxService(r);
    await vv.start(data);
    expect(r.onStart).toHaveBeenCalled();
    const { fake: sinkFake, pipeline, errors } = makePipeline(vv);
    pipeline.speak("こんにちは、今日はいい天気ですね。散歩に行きましょうか？");
    await flush(80);
    const [id1] = sinkFake.order;
    expect(sinkFake.audio(id1)).toEqual([0.5, -0.5, 0.25]);
    expect(sinkFake.utterances.get(id1)?.pushes[0]).toMatchObject({ sampleRate: 24_000, channels: 1 });
    expect(urls[1]).toContain(encodeURIComponent("こんにちは、今日はいい天気ですね。"));
    expect(errors).toEqual([]);
  });

  it("reports a broken engine answer as a readable sentence error", async () => {
    vi.stubGlobal("fetch", vi.fn(async (url: string) => {
      if (url.endsWith("/version")) return new Response("1");
      if (url.includes("/audio_query")) return new Response("{}");
      return new Response(new Uint8Array([1, 2, 3]));
    }));
    const vv = new TTS_VoicevoxService(receiver());
    await vv.start(data);
    const { pipeline, errors } = makePipeline(vv);
    pipeline.speak("A sentence the engine mangles.");
    await flush(80);
    expect(errors).toEqual(["VOICEVOX: not a WAV file"]);
  });
});

describe("Azure through the pipeline", () => {
  it("streams chunks as Azure synthesizes them (first audio before the sentence is complete)", async () => {
    Object.assign(data.azure, { key: "k", location: "westeurope", voice: "en-US-JennyNeural" });
    const r = receiver();
    const az = new TTS_AzureService(r);
    await az.start(data);
    expect(r.onStart).toHaveBeenCalled();
    const { fake: sinkFake, pipeline, errors } = makePipeline(az);
    pipeline.speak("A sentence that Azure streams in pieces.");
    await flush(40);
    const synth = azure.synths[0];
    expect(synth.ssml).toHaveLength(1);
    // Two chunks, the second split inside a 16-bit sample.
    const pcm = new Uint8Array(new Int16Array([16_384, -16_384, 8_192]).buffer);
    synth.self.synthesizing?.(null, { result: { audioData: pcm.slice(0, 3).buffer } });
    await flush(40);
    const [id] = sinkFake.order;
    expect(sinkFake.audio(id)).toEqual([0.5]);
    expect(sinkFake.utterances.get(id)?.ended).toBe(false);
    synth.self.synthesizing?.(null, { result: { audioData: pcm.slice(3).buffer } });
    synth.cb[0]({ reason: ResultReason.SynthesizingAudioCompleted, audioData: pcm.buffer });
    await flush(40);
    expect(sinkFake.audio(id)).toEqual([0.5, -0.5, 0.25]);
    expect(sinkFake.utterances.get(id)?.pushes.every((p) => p.sampleRate === 48_000 && p.channels === 1)).toBe(true);
    expect(sinkFake.utterances.get(id)?.ended).toBe(true);
    expect(errors).toEqual([]);
  });

  it("a refused line is one sentence error", async () => {
    Object.assign(data.azure, { key: "k", location: "westeurope", voice: "en-US-JennyNeural" });
    const az = new TTS_AzureService(receiver());
    await az.start(data);
    const { pipeline, errors } = makePipeline(az);
    pipeline.speak("A sentence Azure refuses to say.");
    await flush(40);
    azure.synths[0].cb[0]({ reason: ResultReason.Canceled, errorDetails: "Too many requests" });
    await flush(40);
    expect(errors).toEqual(["Azure: Too many requests"]);
  });
});

describe("Uberduck through the pipeline", () => {
  it("has Rust push each sentence straight into the begun utterance", async () => {
    Object.assign(data.uberduck, { api_key: "key", voice: "uuid", volume: "0.5" });
    invokeMock.mockResolvedValue({ sampleRate: 22_050, frames: 10, firstAudioMs: 5, totalMs: 9 } as never);
    const ub = new TTS_UberduckService(receiver());
    ub.start(data);
    const { fake: sinkFake, pipeline, errors } = makePipeline(ub);
    expect(pipeline.mode).toBe("sink");
    pipeline.speak("One sentence for the duck.");
    await flush(40);
    expect(invokeMock).toHaveBeenCalledWith("plugin:uberduck-tts|speak_to_sink", {
      data: { auth: { api_key: "key", secret_key: "" }, text: "One sentence for the duck.", voicemodel_uuid: "uuid" },
      utteranceId: sinkFake.order[0],
    });
    expect(sinkFake.utterances.get(sinkFake.order[0])?.ended).toBe(true);
    // A cancelled utterance ends Rust's work with "cancelled": not an error.
    invokeMock.mockRejectedValueOnce("cancelled");
    pipeline.speak("Another sentence for the duck.");
    await flush(40);
    expect(errors).toEqual([]);
    invokeMock.mockRejectedValueOnce("Uberduck speech: HTTP 401");
    pipeline.speak("A third sentence for the duck.");
    await flush(40);
    expect(errors).toEqual(["Uberduck: Uberduck speech: HTTP 401"]);
  });
});
