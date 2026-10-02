import { readFileSync } from "node:fs";
import { resolve } from "node:path";
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

import { invoke } from "@tauri-apps/api/core";
import { toast } from "react-toastify";
import type { RustCatalogEntry, RustModelEntryStatus, RustTtsSinkTarget } from "@/types";
import { TtsSinkClient } from "../../sound/sink";
import { FakeSinkTransport, flush } from "../../sound/__tests__/fakeSink";
import {
  DEFAULT_FALLBACK_BREAKER,
  FallbackBreaker,
  TTS_FAILURE_KINDS,
  TtsFallbackChain,
  TtsPipeline,
  parseTtsFailure,
  type TtsFallbackEvent,
  type TtsSinkSpeaker,
} from "../pipeline";
import { Service_TTS_Schema, TTS_Backends, type TTS_State } from "../schema";
import {
  EDGE_DEFAULT_VOICE,
  FALLBACK_MODEL,
  FALLBACK_RETRY_MS,
  LocalFallbackVoice,
  TTS_EdgeService,
  baseLanguage,
  cloudTimeoutMs,
  edgeRequest,
  edgeVoiceLabel,
  edgeVoiceLanguage,
  edgeVoiceLocale,
  fallbackAfterMs,
  fallbackLanguage,
  groupEdgeVoices,
  isEdgeVoiceName,
  previewEdgeVoice,
} from "../services/edge";
import {
  GEMINI_STYLE_PRESETS,
  GEMINI_TTS_MODELS,
  GEMINI_VOICES,
  TTS_GeminiService,
  geminiRequest,
  previewGeminiVoice,
} from "../services/gemini";
import { TTS_OPTIONS_MISSING, errorMessage, type ITTSReceiver, type TtsSinkContext } from "../types";

// P6-H: free and keyless cloud TTS (Edge, Gemini) with the local Kokoro voice as fallback. The acceptance criterion is
// "an outage falls back to Kokoro within 1 s": a failed request is classified by the Rust plugins at once (their mock
// tests bound that below 1 s), and these tests check that the same sentence then goes to Kokoro at once, and that
// later sentences of the outage go to Kokoro without asking the service.

const invokeMock = vi.mocked(invoke);
const toastMock = vi.mocked(toast);

const T0 = 1_000_000;

function receiver() {
  return { onStart: vi.fn(), onStop: vi.fn(), onError: vi.fn() } satisfies ITTSReceiver;
}

function ctx(id: string, lang: string | null = null): TtsSinkContext {
  return { utteranceId: id, target: { route: "device", device: "", volume: 1 }, lang };
}

const signal = () => new AbortController().signal;

function entry(id: string, requires: string[] = []): RustCatalogEntry {
  return {
    id,
    family: id.split("/")[0],
    kind: id.startsWith("runtime/") ? "runtime" : "model",
    version: "1",
    title: id,
    license: "Apache-2.0",
    status: "published",
    requires,
    files: [{ path: `${id}.tar.bz2`, size: 1000, sha256: "0".repeat(64), sources: [] }],
  } as unknown as RustCatalogEntry;
}

function status(id: string, state: RustModelEntryStatus["state"]): RustModelEntryStatus {
  return { id, state, installed_version: state === "installed" ? "1" : null, installed_path: null, bytes_on_disk: 0, adopted: false, error: null };
}

const CATALOG = [entry("runtime/sherpa-onnx"), entry(FALLBACK_MODEL, ["runtime/sherpa-onnx"])];

let data: TTS_State;
let modelStates: Record<string, RustModelEntryStatus>;

function installWindow(kokoroInstalled = true) {
  data = proxy(Service_TTS_Schema.parse({}));
  modelStates = kokoroInstalled
    ? { "runtime/sherpa-onnx": status("runtime/sherpa-onnx", "installed"), [FALLBACK_MODEL]: status(FALLBACK_MODEL, "installed") }
    : {};
  vi.stubGlobal("window", {
    ApiServer: {
      state: { services: { tts: { data } }, audioOutputDevice: "" },
      models: { load: vi.fn(async () => {}), state: { catalog: CATALOG, states: modelStates } },
      tts: { serviceState: { status: "disconnected" } },
    },
  });
}

/** Commands sent to `plugin`'s `cmd`, with their arguments. */
function calls(cmd: string): Record<string, unknown>[] {
  return invokeMock.mock.calls.filter(([c]) => c === cmd).map(([, a]) => (a ?? {}) as Record<string, unknown>);
}

/** Scripted IPC: `edge` / `gemini` decide each cloud request; local-tts and the check succeed. */
function scriptIpc(cloud: (cmd: string, args: Record<string, unknown>) => Promise<unknown>) {
  invokeMock.mockImplementation(async (cmd: string, args?: unknown) => {
    const a = (args ?? {}) as Record<string, unknown>;
    if (cmd === "plugin:edge-tts|edge_tts_speak" || cmd === "plugin:gemini-tts|gemini_tts_speak") return cloud(cmd, a);
    if (cmd === "plugin:gemini-tts|gemini_tts_check") return true;
    if (cmd === "plugin:local-tts|load_local_tts") return { id: FALLBACK_MODEL };
    if (cmd === "plugin:local-tts|speak_local_tts") return { sampleRate: 24000, frames: 2400, firstAudioMs: 120, totalMs: 300 };
    return undefined;
  });
}

beforeAll(async () => {
  await i18n.init({ lng: "en", resources: { en: { translation: {} } }, interpolation: { escapeValue: false } });
});

beforeEach(() => {
  invokeMock.mockReset();
  toastMock.mockClear();
  installWindow();
});

afterEach(() => {
  vi.useRealTimers();
  vi.unstubAllGlobals();
});

// -------------------------------------------------------------------------------------------------

describe("parseTtsFailure", () => {
  it("reads the Rust failure tag", () => {
    expect(parseTtsFailure("[cloud-tts:rate_limited:retry=21000] Gemini TTS: the free quota is used up.")).toEqual({
      kind: "rate_limited",
      partial: false,
      retryAfterMs: 21000,
      message: "Gemini TTS: the free quota is used up.",
      tagged: true,
    });
    expect(parseTtsFailure(new Error("[cloud-tts:stalled:partial] Edge voices: stalled"))).toMatchObject({ kind: "stalled", partial: true, tagged: true });
    expect(parseTtsFailure("[cloud-tts:unavailable:partial:retry=7000] x")).toMatchObject({ kind: "unavailable", partial: true, retryAfterMs: 7000 });
    // An unknown kind is still the service's failure.
    expect(parseTtsFailure("[cloud-tts:teapot] x")).toMatchObject({ kind: "other", tagged: true, message: "x" });
  });

  it("treats untagged errors as not the service's", () => {
    expect(parseTtsFailure("no TTS utterance \"x\"")).toEqual({ kind: "other", partial: false, retryAfterMs: null, message: "no TTS utterance \"x\"", tagged: false });
    expect(parseTtsFailure("cancelled").tagged).toBe(false);
  });

  it("knows every kind Rust sends", () => {
    const rust = readFileSync(resolve(__dirname, "../../../../../src-tauri/src/services/edge_tts/cloud.rs"), "utf8");
    const wire = [...rust.matchAll(/FailureKind::\w+ => "([a-z_]+)"/g)].map((m) => m[1]);
    expect(wire.length).toBe(8);
    expect([...wire].sort()).toEqual([...TTS_FAILURE_KINDS].sort());
  });
});

describe("FallbackBreaker", () => {
  it("backs off on outages and closes on success", () => {
    let now = T0;
    const b = new FallbackBreaker(DEFAULT_FALLBACK_BREAKER, () => now);
    expect(b.route()).toBe("primary");
    expect(b.failure({ kind: "unavailable", retryAfterMs: null })).toEqual({ opened: true, openUntil: T0 + 15_000 });
    expect(b.route()).toBe("fallback");
    expect(b.degraded).toBe(true);
    now += 15_000;
    // Half-open: the next sentence tries the service again; a second failure pauses longer and is not news.
    expect(b.route()).toBe("primary");
    expect(b.failure({ kind: "network", retryAfterMs: null })).toEqual({ opened: false, openUntil: now + 30_000 });
    now += 30_000;
    expect(b.failure({ kind: "stalled", retryAfterMs: null }).openUntil).toBe(now + 60_000);
    for (let i = 0; i < 10; i++) b.failure({ kind: "stalled", retryAfterMs: null });
    expect(b.openUntil).toBe(now + DEFAULT_FALLBACK_BREAKER.maxOpenMs);
    expect(b.success()).toBe(true);
    expect(b.success()).toBe(false);
    expect(b.route()).toBe("primary");
    expect(b.failure({ kind: "unavailable", retryAfterMs: null }).openUntil).toBe(now + 15_000);
  });

  it("follows the service's retry hint for 429 and pauses long for settings problems", () => {
    let now = T0;
    const b = new FallbackBreaker(DEFAULT_FALLBACK_BREAKER, () => now);
    expect(b.failure({ kind: "rate_limited", retryAfterMs: 21_000 }).openUntil).toBe(T0 + 21_000);
    b.reset();
    expect(b.failure({ kind: "rate_limited", retryAfterMs: null }).openUntil).toBe(T0 + DEFAULT_FALLBACK_BREAKER.rateLimitMs);
    b.reset();
    // A hint never exceeds the hard pause.
    expect(b.failure({ kind: "rate_limited", retryAfterMs: 99 * 3_600_000 }).openUntil).toBe(T0 + DEFAULT_FALLBACK_BREAKER.hardOpenMs);
    b.reset();
    expect(b.failure({ kind: "auth", retryAfterMs: null }).openUntil).toBe(T0 + DEFAULT_FALLBACK_BREAKER.hardOpenMs);
    b.reset();
    expect(b.failure({ kind: "rejected", retryAfterMs: null }).openUntil).toBe(T0 + DEFAULT_FALLBACK_BREAKER.hardOpenMs);
    b.reset();
    // A longer hint than the backoff wins for outages too.
    expect(b.failure({ kind: "unavailable", retryAfterMs: 40_000 }).openUntil).toBe(T0 + 40_000);
    now += 1;
  });

  it("does not open for a sentence the service could not say", () => {
    const b = new FallbackBreaker(DEFAULT_FALLBACK_BREAKER, () => T0);
    expect(b.failure({ kind: "no_audio", retryAfterMs: null })).toEqual({ opened: false, openUntil: 0 });
    expect(b.route()).toBe("primary");
    expect(b.degraded).toBe(false);
  });
});

describe("TtsFallbackChain", () => {
  function chain(opts: { primary: TtsSinkSpeaker; fallback?: TtsSinkSpeaker | null; now?: () => number }) {
    const events: TtsFallbackEvent[] = [];
    const recovered = vi.fn();
    const fallbackCalls: { text: string; id: string; at: number }[] = [];
    const fallback: TtsSinkSpeaker | null =
      opts.fallback === null
        ? null
        : async (text, sink, sig) => {
            fallbackCalls.push({ text, id: sink.utteranceId, at: performance.now() });
            await opts.fallback?.(text, sink, sig);
          };
    const c = new TtsFallbackChain({
      primary: opts.primary,
      fallback: async () => fallback,
      breaker: new FallbackBreaker(DEFAULT_FALLBACK_BREAKER, opts.now ?? (() => T0)),
      onFallback: (e) => events.push(e),
      onRecovered: recovered,
    });
    return { c, events, recovered, fallbackCalls };
  }

  it("speaks a sentence the service failed with the local voice at once, into the same utterance", async () => {
    let failedAt = 0;
    const primary = vi.fn<TtsSinkSpeaker>(async () => {
      await new Promise((r) => setTimeout(r, 30));
      failedAt = performance.now();
      throw "[cloud-tts:unavailable] Edge voices: the service is unavailable (HTTP 503).";
    });
    const { c, events, fallbackCalls } = chain({ primary });
    const started = performance.now();
    await c.synthesize("Hello there.", ctx("u1"), signal());
    expect(fallbackCalls).toHaveLength(1);
    expect(fallbackCalls[0]).toMatchObject({ text: "Hello there.", id: "u1" });
    // The local voice starts right after the failure: well within the 1 s budget of the acceptance criterion.
    expect(fallbackCalls[0].at - failedAt).toBeLessThan(100);
    expect(fallbackCalls[0].at - started).toBeLessThan(1000);
    expect(events).toEqual([{ failure: expect.objectContaining({ kind: "unavailable" }), opened: true, spokenByFallback: true }]);
  });

  it("keeps later sentences of an outage on the local voice without asking the service", async () => {
    let now = T0;
    const primary = vi.fn<TtsSinkSpeaker>(async () => {
      throw "[cloud-tts:rate_limited:retry=20000] Gemini TTS: quota.";
    });
    const { c, events, fallbackCalls, recovered } = chain({ primary, now: () => now });
    await c.synthesize("One.", ctx("u1"), signal());
    await c.synthesize("Two.", ctx("u2"), signal());
    await c.synthesize("Three.", ctx("u3"), signal());
    expect(primary).toHaveBeenCalledTimes(1);
    expect(fallbackCalls.map((f) => f.id)).toEqual(["u1", "u2", "u3"]);
    expect(events.slice(1)).toEqual([
      { failure: null, opened: false, spokenByFallback: true },
      { failure: null, opened: false, spokenByFallback: true },
    ]);
    // After the service's retry hint the next sentence tries it again; success ends the outage.
    now += 20_000;
    primary.mockImplementationOnce(async () => {});
    expect(recovered).not.toHaveBeenCalled();
    await c.synthesize("Four.", ctx("u4"), signal());
    expect(primary).toHaveBeenCalledTimes(2);
    expect(fallbackCalls).toHaveLength(3);
    expect(recovered).toHaveBeenCalledTimes(1);
    expect(c.breaker.route()).toBe("primary");
  });

  it("falls back after a stall (the plugin's short timeout)", async () => {
    const primary = vi.fn<TtsSinkSpeaker>(async () => {
      throw "[cloud-tts:stalled] Edge voices: no audio within 3 s (the service did not answer in time).";
    });
    const { c, fallbackCalls, events } = chain({ primary });
    await c.synthesize("Slow.", ctx("u1"), signal());
    expect(fallbackCalls).toHaveLength(1);
    expect(events[0].failure?.kind).toBe("stalled");
  });

  it("never repeats words: a failure after part of the sentence played opens the breaker only", async () => {
    const primary = vi.fn<TtsSinkSpeaker>(async () => {
      throw "[cloud-tts:stalled:partial] Edge voices: stalled.";
    });
    const { c, fallbackCalls, events } = chain({ primary });
    await c.synthesize("Half spoken.", ctx("u1"), signal());
    expect(fallbackCalls).toHaveLength(0);
    expect(events).toEqual([{ failure: expect.objectContaining({ partial: true }), opened: true, spokenByFallback: false }]);
    await c.synthesize("Next.", ctx("u2"), signal());
    expect(fallbackCalls.map((f) => f.id)).toEqual(["u2"]);
  });

  it("never falls back on stop / skip, the sink's own cancel or errors that are not the service's", async () => {
    const ac = new AbortController();
    const aborted = chain({
      primary: async () => {
        ac.abort();
        throw "cancelled";
      },
    });
    await expect(aborted.c.synthesize("x", ctx("u1"), ac.signal)).rejects.toBe("cancelled");
    expect(aborted.fallbackCalls).toHaveLength(0);

    const sinkCancel = chain({
      primary: async () => {
        throw "cancelled";
      },
    });
    await expect(sinkCancel.c.synthesize("x", ctx("u1"), signal())).rejects.toBe("cancelled");
    expect(sinkCancel.fallbackCalls).toHaveLength(0);
    expect(sinkCancel.c.breaker.degraded).toBe(false);

    const ipc = chain({
      primary: async () => {
        throw "no TTS utterance \"u1\"";
      },
    });
    await expect(ipc.c.synthesize("x", ctx("u1"), signal())).rejects.toBe("no TTS utterance \"u1\"");
    expect(ipc.fallbackCalls).toHaveLength(0);
    expect(ipc.c.breaker.degraded).toBe(false);

    // Already aborted: nothing runs.
    const done = new AbortController();
    done.abort();
    const idle = chain({ primary: vi.fn() });
    await idle.c.synthesize("x", ctx("u1"), done.signal);
    expect(idle.fallbackCalls).toHaveLength(0);
  });

  it("reports the failure when there is no local voice, and keeps asking the service", async () => {
    const primary = vi.fn<TtsSinkSpeaker>(async () => {
      throw "[cloud-tts:network] Edge voices: cannot reach speech.platform.bing.com.";
    });
    const { c, events } = chain({ primary, fallback: null });
    await expect(c.synthesize("x", ctx("u1"), signal())).rejects.toThrow("Edge voices: cannot reach speech.platform.bing.com.");
    expect(events).toEqual([{ failure: expect.objectContaining({ kind: "network" }), opened: true, spokenByFallback: false }]);
    // Breaker open, but nothing better to do: the service is still asked.
    await expect(c.synthesize("y", ctx("u2"), signal())).rejects.toThrow();
    expect(primary).toHaveBeenCalledTimes(2);
  });

  it("does not fall back for a sentence the service could not say (and stays closed)", async () => {
    const primary = vi.fn<TtsSinkSpeaker>(async () => {
      throw "[cloud-tts:no_audio] Gemini TTS: the model answered with text instead of audio.";
    });
    const { c, fallbackCalls } = chain({ primary });
    await c.synthesize("x", ctx("u1"), signal());
    // The local voice says it, but the service is asked for the next sentence.
    expect(fallbackCalls).toHaveLength(1);
    await c.synthesize("y", ctx("u2"), signal());
    expect(primary).toHaveBeenCalledTimes(2);
  });
});

// -------------------------------------------------------------------------------------------------

describe("cloud helpers", () => {
  it("picks the local voice's language", () => {
    expect(baseLanguage("pt-BR")).toBe("pt");
    expect(baseLanguage("  ")).toBeNull();
    expect(baseLanguage("x")).toBeNull();
    // The text event's tag wins; languages Kokoro lacks get its default voice.
    expect(fallbackLanguage("Hola", "es", "en", null)).toBe("es");
    expect(fallbackLanguage("Hallo", "de", "en", null)).toBeNull();
    // Else the script, else the cloud voice's language for Latin text.
    expect(fallbackLanguage("こんにちは", null, "en", null)).toBe("ja");
    expect(fallbackLanguage("Bonjour", null, "fr", null)).toBe("fr");
    expect(fallbackLanguage("Hallo", null, "de", null)).toBeNull();
    // No letters: keep the previous language.
    expect(fallbackLanguage("123!", null, "en", "ja")).toBe("ja");
  });

  it("validates Edge voice names like Rust", () => {
    expect(isEdgeVoiceName("en-US-AvaMultilingualNeural")).toBe(true);
    expect(isEdgeVoiceName(" zh-CN-liaoning-XiaobeiNeural ")).toBe(true);
    expect(isEdgeVoiceName("fil-PH-BlessicaNeural")).toBe(true);
    expect(isEdgeVoiceName("Microsoft Server Speech Text to Speech Voice (ja-JP, NanamiNeural)")).toBe(true);
    expect(isEdgeVoiceName("")).toBe(false);
    expect(isEdgeVoiceName("en-US-Ava'><x>Neural")).toBe(false);
    expect(isEdgeVoiceName("Ava")).toBe(false);
    expect(isEdgeVoiceName("EN-us-AvaNeural")).toBe(false);
    expect(isEdgeVoiceName("en-US-Ava")).toBe(false);
    // The service also lists voices with a script subtag (Inuktitut in Canadian syllabics and in Latin script).
    expect(isEdgeVoiceName("iu-Cans-CA-SiqiniqNeural")).toBe(true);
    expect(isEdgeVoiceName("iu-Latn-CA-TaqqiqNeural")).toBe(true);
    expect(isEdgeVoiceName("es-419-ValentinaNeural")).toBe(true);
    for (const bad of ["en-US-Neural", "iu-cans-CA-SiqiniqNeural", "iu-Cans-SiqiniqNeural", "en-US--AvaNeural", "en-US-Ava.Neural", "-US-AvaNeural", "en-Us-AvaNeural"]) {
      expect(isEdgeVoiceName(bad), bad).toBe(false);
    }
    expect(edgeVoiceLanguage("ja-JP-NanamiNeural")).toBe("ja");
    expect(edgeVoiceLanguage("iu-Cans-CA-SiqiniqNeural")).toBe("iu");
    expect(edgeVoiceLanguage("x")).toBeNull();
  });

  it("builds requests from the slices", () => {
    const edge = Service_TTS_Schema.parse({ edge: { voice: "  ", rate: "9", pitchHz: "-80.4" } }).edge;
    expect(edgeRequest(edge, "Hi.", 3000.4)).toEqual({ text: "Hi.", voice: EDGE_DEFAULT_VOICE, rate: 2, pitchHz: -50, timeoutMs: 3000 });
    const gemini = Service_TTS_Schema.parse({ gemini: { apiKey: " k ", model: " ", voice: "", style: " calm " } }).gemini;
    expect(geminiRequest(gemini, "Hi.", 0, "ja")).toEqual({
      apiKey: "k",
      baseUrl: "",
      model: GEMINI_TTS_MODELS[0].id,
      text: "Hi.",
      voice: "Kore",
      style: "calm",
      language: "ja",
      timeoutMs: 0,
    });
  });

  it("uses the short timeout only while the local voice can take over", () => {
    const c = { fallbackToLocal: true, fallbackAfter: "3" };
    expect(fallbackAfterMs(c)).toBe(3000);
    expect(fallbackAfterMs({ fallbackAfter: "0.2" })).toBe(1000);
    expect(fallbackAfterMs({ fallbackAfter: "x" })).toBe(3000);
    expect(fallbackAfterMs({ fallbackAfter: "99" })).toBe(30_000);
    expect(cloudTimeoutMs(c, { usable: true })).toBe(3000);
    expect(cloudTimeoutMs(c, { usable: false })).toBe(0);
    expect(cloudTimeoutMs(c, null)).toBe(0);
    expect(cloudTimeoutMs({ ...c, fallbackToLocal: false }, { usable: true })).toBe(0);
  });

  it("groups and labels Edge voices", () => {
    const voices = [
      { shortName: "ja-JP-NanamiNeural", friendlyName: "", locale: "ja-JP", gender: "Female" },
      { shortName: "en-US-AndrewMultilingualNeural", friendlyName: "", locale: "en-US", gender: "Male" },
      { shortName: "en-US-AvaMultilingualNeural", friendlyName: "", locale: "en-US", gender: "Female" },
      { shortName: "xx-YY-OddNeural", friendlyName: "", locale: "", gender: "" },
    ];
    const groups = groupEdgeVoices(voices, "en");
    expect(groups.map((g) => g.locale).sort()).toEqual(["en-US", "ja-JP", "xx-YY"]);
    const en = groups.find((g) => g.locale === "en-US")!;
    expect(en.voices.map((v) => v.shortName)).toEqual(["en-US-AndrewMultilingualNeural", "en-US-AvaMultilingualNeural"]);
    expect(en.label).toContain("English");
    expect(edgeVoiceLabel(voices[0])).toBe("tts.edge_voice_female");
    expect(edgeVoiceLabel(voices[3])).toBe("Odd");
    // Locales with a script subtag or a sub-region: the name is what follows the locale.
    const iu = { shortName: "iu-Cans-CA-SiqiniqNeural", friendlyName: "", locale: "iu-Cans-CA", gender: "" };
    expect(edgeVoiceLocale(iu)).toBe("iu-Cans-CA");
    expect(edgeVoiceLocale({ shortName: iu.shortName, locale: "" })).toBe("iu-Cans-CA");
    expect(edgeVoiceLabel(iu)).toBe("Siqiniq");
    expect(edgeVoiceLabel({ shortName: "zh-CN-liaoning-XiaobeiNeural", locale: "zh-CN-liaoning", gender: "" })).toBe("Xiaobei");
    expect(groupEdgeVoices([iu], "en").map((g) => g.locale)).toEqual(["iu-Cans-CA"]);
  });

  it("shows a failed test sentence without the internal failure tag", async () => {
    const sink = { ready: vi.fn(async () => {}), begin: vi.fn(async () => {}), end: vi.fn(async () => {}), cancel: vi.fn(async () => {}) };
    Object.assign(window.ApiServer, { sound: { sink } });
    const target: RustTtsSinkTarget = { route: "device", device: "", volume: 1 };
    const failed = (p: Promise<void>) => p.then(() => "resolved", (e: unknown) => errorMessage(e));

    invokeMock.mockImplementation(async () => {
      throw "[cloud-tts:network] Edge voices: cannot reach speech.platform.bing.com.";
    });
    expect(await failed(previewEdgeVoice(data.edge, "Hi.", target))).toBe("Edge voices: cannot reach speech.platform.bing.com.");
    invokeMock.mockImplementation(async () => {
      throw "[cloud-tts:auth:retry=5000] Gemini TTS: the API key was refused. Check the key in the Gemini TTS settings (HTTP 400).";
    });
    expect(await failed(previewGeminiVoice(data.gemini, "Hi.", target))).toBe(
      "Gemini TTS: the API key was refused. Check the key in the Gemini TTS settings (HTTP 400).",
    );
    // Errors that are not the service's pass unchanged; the preview utterance is dropped each time.
    invokeMock.mockImplementation(async () => {
      throw "TTS sink unavailable: no output device";
    });
    expect(await failed(previewEdgeVoice(data.edge, "Hi.", target))).toBe("TTS sink unavailable: no output device");
    expect(sink.cancel).toHaveBeenCalledTimes(3);
    expect(sink.end).not.toHaveBeenCalled();
  });

  it("defaults are off and safe", () => {
    const parsed = Service_TTS_Schema.parse({});
    expect(parsed.backend).not.toBe(TTS_Backends.edge);
    expect(parsed.backend).not.toBe(TTS_Backends.gemini);
    expect(parsed.edge).toMatchObject({ voice: EDGE_DEFAULT_VOICE, fallbackToLocal: true, fallbackAfter: "3", device: "" });
    expect(parsed.gemini).toMatchObject({ apiKey: "", model: GEMINI_TTS_MODELS[0].id, voice: "Kore", fallbackToLocal: true, baseUrl: "" });
    // Bad persisted values fall back to the defaults.
    const bad = Service_TTS_Schema.parse({ edge: { fallbackAfter: {}, voice: 5 }, gemini: { fallbackToLocal: "yes" } });
    expect(bad.edge.voice).toBe("5");
    expect(bad.gemini.fallbackToLocal).toBe(true);
  });

  it("has an English string for every dynamic key", () => {
    // The P6-H staging file was merged into English by P6-I0 (and removed); English is the source of truth now.
    const live = JSON.parse(readFileSync(resolve(__dirname, "../../../../../public/i18n/en/translation.json"), "utf8")).tts as Record<string, string>;
    const has = (k: string) => k in live;
    for (const kind of TTS_FAILURE_KINDS) expect(has(`cloud_reason_${kind}`), kind).toBe(true);
    for (const v of GEMINI_VOICES) expect(has(`gemini_voice_style_${v.style}`), v.style).toBe(true);
    for (const p of GEMINI_STYLE_PRESETS) expect(has(`gemini_style_preset_${p.key}`), p.key).toBe(true);
    for (const m of GEMINI_TTS_MODELS) if (m.label) expect(has(m.label.replace("tts.", "")), m.label).toBe(true);
  });
});

// -------------------------------------------------------------------------------------------------

describe("TTS_EdgeService", () => {
  async function started(r = receiver()) {
    const edge = new TTS_EdgeService(r);
    await edge.start(data);
    await flush(40);
    return edge;
  }

  it("starts at once and loads the local voice in the background", async () => {
    scriptIpc(async () => ({}));
    data.edge.voice = "ja-JP-NanamiNeural";
    const r = receiver();
    const edge = await started(r);
    expect(r.onStart).toHaveBeenCalledTimes(1);
    expect(calls("plugin:local-tts|load_local_tts")).toEqual([{ model: FALLBACK_MODEL, threads: null, voice: null, language: "ja" }]);
    expect(edge.local?.state).toBe("ready");
    edge.dispose();
  });

  it("refuses an invalid voice name", async () => {
    scriptIpc(async () => ({}));
    data.edge.voice = "en-US-Ava'/><x";
    const r = receiver();
    await started(r);
    expect(r.onStart).not.toHaveBeenCalled();
    expect(r.onStop).toHaveBeenCalledWith("tts.edge_voice_invalid");
    expect(calls("plugin:local-tts|unload_local_tts")).toHaveLength(1);
  });

  it("speaks through Edge with the short timeout while the local voice is ready", async () => {
    scriptIpc(async () => ({ sampleRate: 24000, frames: 100, firstAudioMs: 300, totalMs: 900 }));
    data.edge.voice = "en-GB-SoniaNeural";
    data.edge.fallbackAfter = "2.5";
    const edge = await started();
    await edge.synthesizeToSink("Good morning.", ctx("u1"), signal());
    expect(calls("plugin:edge-tts|edge_tts_speak")).toEqual([
      { request: { text: "Good morning.", voice: "en-GB-SoniaNeural", rate: 1, pitchHz: 0, timeoutMs: 2500 }, utteranceId: "u1" },
    ]);
    expect(calls("plugin:local-tts|speak_local_tts")).toHaveLength(0);
    edge.dispose();
  });

  it("an Edge outage falls back to Kokoro within 1 s, then stays on Kokoro", async () => {
    scriptIpc(async () => {
      throw "[cloud-tts:unavailable] Edge voices: the service is unavailable (HTTP 503).";
    });
    data.edge.voice = "fr-FR-DeniseNeural";
    data.edge.rate = "1.2";
    const edge = await started();
    const t = performance.now();
    await edge.synthesizeToSink("Bonjour à tous.", ctx("u1"), signal());
    expect(performance.now() - t).toBeLessThan(1000);
    expect(calls("plugin:local-tts|speak_local_tts")).toEqual([
      {
        request: { model: FALLBACK_MODEL, voice: "", speed: 1.2, language: "fr", text: "Bonjour à tous.", threads: null },
        utteranceId: "u1",
      },
    ]);
    // One notice per outage.
    expect(toastMock).toHaveBeenCalledTimes(1);
    expect(toastMock.mock.calls[0][0]).toBe("tts.cloud_fallback_active");

    await edge.synthesizeToSink("Encore une phrase.", ctx("u2"), signal());
    expect(calls("plugin:edge-tts|edge_tts_speak")).toHaveLength(1);
    expect(calls("plugin:local-tts|speak_local_tts").map((c) => c.utteranceId)).toEqual(["u1", "u2"]);
    expect(toastMock).toHaveBeenCalledTimes(1);

    // A new voice is a new setup: tried at once.
    data.edge.voice = "fr-FR-HenriNeural";
    await edge.synthesizeToSink("Et encore.", ctx("u3"), signal());
    expect(calls("plugin:edge-tts|edge_tts_speak")).toHaveLength(2);
    edge.dispose();
  });

  it("without the local voice it waits the automatic limit and reports failures", async () => {
    installWindow(false);
    scriptIpc(async () => {
      throw "[cloud-tts:network] Edge voices: cannot reach speech.platform.bing.com.";
    });
    const r = receiver();
    const edge = await started(r);
    expect(r.onStart).toHaveBeenCalled();
    expect(edge.local?.state).toBe("missing");
    await expect(edge.synthesizeToSink("Hi.", ctx("u1"), signal())).rejects.toThrow("Edge voices: cannot reach");
    expect(calls("plugin:edge-tts|edge_tts_speak")[0]).toMatchObject({ request: { timeoutMs: 0 } });
    expect(calls("plugin:local-tts|load_local_tts")).toHaveLength(0);
    expect(calls("plugin:local-tts|speak_local_tts")).toHaveLength(0);
    edge.dispose();
  });

  it("uses a Kokoro downloaded while Edge runs from the next sentence on, without a restart", async () => {
    installWindow(false);
    let fail = false;
    scriptIpc(async () => {
      if (fail) throw "[cloud-tts:network] Edge voices: cannot reach speech.platform.bing.com.";
      return {};
    });
    const edge = await started();
    expect(edge.local?.state).toBe("missing");
    await edge.synthesizeToSink("One.", ctx("u1"), signal());
    // Still not downloaded: the automatic limit, nothing loaded.
    expect(calls("plugin:edge-tts|edge_tts_speak")[0]).toMatchObject({ request: { timeoutMs: 0 } });
    expect(calls("plugin:local-tts|load_local_tts")).toHaveLength(0);

    // The user downloads Kokoro from the card in the Edge panel; then Edge goes down.
    modelStates["runtime/sherpa-onnx"] = status("runtime/sherpa-onnx", "installed");
    modelStates[FALLBACK_MODEL] = status(FALLBACK_MODEL, "installed");
    fail = true;
    const t = performance.now();
    await edge.synthesizeToSink("Two.", ctx("u2"), signal());
    expect(performance.now() - t).toBeLessThan(1000);
    // That sentence already had the short timeout, and Kokoro said it.
    expect(calls("plugin:edge-tts|edge_tts_speak")[1]).toMatchObject({ request: { timeoutMs: 3000 } });
    expect(calls("plugin:local-tts|load_local_tts")).toHaveLength(1);
    expect(calls("plugin:local-tts|speak_local_tts").map((c) => c.utteranceId)).toEqual(["u2"]);
    expect(edge.local?.state).toBe("ready");
    edge.dispose();
  });

  it("with the fallback switched off nothing local is loaded", async () => {
    scriptIpc(async () => ({}));
    data.edge.fallbackToLocal = false;
    const edge = await started();
    expect(edge.local).toBeNull();
    await edge.synthesizeToSink("Hi.", ctx("u1"), signal());
    expect(calls("plugin:edge-tts|edge_tts_speak")[0]).toMatchObject({ request: { timeoutMs: 0 } });
    expect(calls("plugin:local-tts|load_local_tts")).toHaveLength(0);
    edge.dispose();
  });

  it("skip cancels the Edge request and never falls back", async () => {
    let reject: (e: unknown) => void = () => {};
    scriptIpc(
      () =>
        new Promise((_, rej) => {
          reject = rej;
        }),
    );
    const edge = await started();
    const ac = new AbortController();
    const p = edge.synthesizeToSink("Long sentence.", ctx("u1"), ac.signal);
    await flush();
    ac.abort();
    expect(calls("plugin:edge-tts|edge_tts_cancel")).toEqual([{ utteranceId: "u1" }]);
    reject("cancelled");
    await p;
    expect(calls("plugin:local-tts|speak_local_tts")).toHaveLength(0);
    edge.dispose();
  });

  it("a sink that dropped the utterance is reported by the pipeline, not as a service failure", async () => {
    scriptIpc(async () => {
      throw "cancelled";
    });
    const edge = await started();
    await expect(edge.synthesizeToSink("Hi.", ctx("u1"), signal())).rejects.toBe("cancelled");
    expect(edge.chain.breaker.degraded).toBe(false);
    expect(calls("plugin:local-tts|speak_local_tts")).toHaveLength(0);
    edge.dispose();
  });

  it("stop cancels every request and frees the local voice", async () => {
    scriptIpc(async () => ({}));
    const r = receiver();
    const edge = await started(r);
    edge.stop();
    expect(calls("plugin:edge-tts|edge_tts_cancel")).toEqual([{ utteranceId: null }]);
    expect(calls("plugin:local-tts|cancel_local_tts")).toEqual([{ utteranceId: null }]);
    expect(calls("plugin:local-tts|unload_local_tts")).toHaveLength(1);
    expect(r.onStop).toHaveBeenCalledWith();
    // Stopped: sentences are ignored.
    await edge.synthesizeToSink("Hi.", ctx("u9"), signal());
    expect(calls("plugin:edge-tts|edge_tts_speak")).toHaveLength(0);
  });

  it("in the sentence pipeline the fallback fills the sentence's own sink utterance", async () => {
    scriptIpc(async () => {
      throw "[cloud-tts:unavailable] Edge voices: the service is unavailable (HTTP 503).";
    });
    const edge = await started();
    const fake = new FakeSinkTransport();
    const errors: string[] = [];
    const target: RustTtsSinkTarget = { route: "device", device: "Speakers", volume: 1 };
    const pipeline = new TtsPipeline(edge, { sink: new TtsSinkClient(fake), target: () => target, onError: (e) => errors.push(e) });
    const t = performance.now();
    expect(pipeline.speak("First sentence here. Second sentence here.", "en")).toBe(2);
    await flush(80);
    expect(performance.now() - t).toBeLessThan(1000);
    const [id1, id2] = fake.order;
    const local = calls("plugin:local-tts|speak_local_tts");
    expect(local.map((c) => c.utteranceId)).toEqual([id1, id2]);
    expect(local[0]).toMatchObject({ request: { text: "First sentence here.", language: "en" } });
    // Both sentences were ended normally (they play), and nothing was reported as an error.
    expect(fake.utterances.get(id1)?.ended).toBe(true);
    expect(fake.utterances.get(id2)?.ended).toBe(true);
    expect(errors).toEqual([]);
    expect(calls("plugin:edge-tts|edge_tts_speak")).toHaveLength(1);
    pipeline.dispose();
    edge.dispose();
  });
});

describe("LocalFallbackVoice", () => {
  it("tries a failed load again after a pause, and is usable at once when it does", async () => {
    let now = T0;
    let loadFails = true;
    invokeMock.mockImplementation(async (cmd: string) => {
      if (cmd === "plugin:local-tts|load_local_tts") {
        if (loadFails) throw "not enough memory to load the model";
        return { id: FALLBACK_MODEL };
      }
      return undefined;
    });
    const local = new LocalFallbackVoice({ threads: null, speed: () => 1, languageHint: () => "en" }, () => now);
    expect(await local.prepare()).toBe(false);
    expect(local.state).toBe("failed");
    expect(local.error).toBe("not enough memory to load the model");
    expect(local.usable).toBe(false);

    // Not due yet: the same failed attempt; nothing is loaded again.
    loadFails = false;
    local.observe("Hello.", "en");
    expect(await local.speaker()).toBeNull();
    expect(calls("plugin:local-tts|load_local_tts")).toHaveLength(1);

    // Due: the next sentence starts a new attempt, which counts as usable at once (the short timeout applies).
    now += FALLBACK_RETRY_MS;
    local.observe("Hello again.", "en");
    expect(local.usable).toBe(true);
    expect(await local.speaker()).not.toBeNull();
    expect(local.state).toBe("ready");
    expect(local.error).toBe("");
    expect(calls("plugin:local-tts|load_local_tts")).toHaveLength(2);
    // Ready is final: no more attempts.
    now += FALLBACK_RETRY_MS;
    local.observe("Bye.", "en");
    await local.speaker();
    expect(calls("plugin:local-tts|load_local_tts")).toHaveLength(2);
    local.dispose();
  });

  it("checks a missing download again on each sentence, without loading anything until it is there", async () => {
    installWindow(false);
    scriptIpc(async () => ({}));
    const local = new LocalFallbackVoice({ threads: null, speed: () => 1, languageHint: () => null });
    expect(await local.prepare()).toBe(false);
    expect(local.state).toBe("missing");
    local.observe("Hello.", null);
    expect(local.usable).toBe(false);
    expect(await local.speaker()).toBeNull();
    expect(calls("plugin:local-tts|load_local_tts")).toHaveLength(0);

    modelStates["runtime/sherpa-onnx"] = status("runtime/sherpa-onnx", "installed");
    modelStates[FALLBACK_MODEL] = status(FALLBACK_MODEL, "installed");
    // A failure (or the breaker's route) asks for the speaker: that finds the download too.
    expect(await local.speaker()).not.toBeNull();
    expect(local.state).toBe("ready");
    expect(calls("plugin:local-tts|load_local_tts")).toHaveLength(1);
    local.dispose();
  });
});

describe("TTS_GeminiService", () => {
  it("needs a key", async () => {
    scriptIpc(async () => ({}));
    const r = receiver();
    const gemini = new TTS_GeminiService(r);
    await gemini.start(data);
    expect(r.onStop).toHaveBeenCalledWith(TTS_OPTIONS_MISSING);
    expect(r.onStart).not.toHaveBeenCalled();
    expect(calls("plugin:gemini-tts|gemini_tts_check")).toHaveLength(0);
  });

  it("checks the key on start; an unreachable Google still starts, a refused key does not", async () => {
    data.gemini.apiKey = " AIza-key ";
    scriptIpc(async () => ({}));
    invokeMock.mockImplementation(async (cmd: string) => {
      if (cmd === "plugin:gemini-tts|gemini_tts_check") return false;
      return undefined;
    });
    const r = receiver();
    const gemini = new TTS_GeminiService(r);
    await gemini.start(data);
    expect(calls("plugin:gemini-tts|gemini_tts_check")).toEqual([{ apiKey: "AIza-key", model: GEMINI_TTS_MODELS[0].id, baseUrl: null }]);
    expect(r.onStart).toHaveBeenCalledTimes(1);
    gemini.dispose();

    invokeMock.mockImplementation(async (cmd: string) => {
      if (cmd === "plugin:gemini-tts|gemini_tts_check") throw "Gemini TTS: the API key was refused. Check the key in the Gemini TTS settings (HTTP 400).";
      return undefined;
    });
    const r2 = receiver();
    const refused = new TTS_GeminiService(r2);
    await refused.start(data);
    expect(r2.onStart).not.toHaveBeenCalled();
    expect(r2.onStop).toHaveBeenCalledWith(expect.stringContaining("API key was refused"));
  });

  it("keeps Kokoro loaded for the language being spoken, so a failure needs no reload", async () => {
    data.gemini.apiKey = "k";
    let fail = false;
    scriptIpc(async () => {
      if (fail) throw "[cloud-tts:unavailable] Gemini TTS: the service is busy or unavailable (HTTP 503).";
      return {};
    });
    const gemini = new TTS_GeminiService(receiver());
    await gemini.start(data);
    await flush(40);
    // Gemini has no voice language: Kokoro starts on its default (English).
    expect(calls("plugin:local-tts|load_local_tts").map((c) => c.language)).toEqual([null]);
    await gemini.synthesizeToSink("今日はいい天気ですね。", ctx("g1"), signal());
    // The cloud voice spoke Japanese: Kokoro was re-warmed for Japanese meanwhile (once).
    expect(calls("plugin:local-tts|load_local_tts").map((c) => c.language)).toEqual([null, "ja"]);
    await gemini.synthesizeToSink("明日も晴れるでしょう。", ctx("g2"), signal());
    expect(calls("plugin:local-tts|load_local_tts")).toHaveLength(2);
    fail = true;
    await gemini.synthesizeToSink("それではまた。", ctx("g3"), signal());
    // The fallback speaks in the language it is loaded for: no reload in the way.
    expect(calls("plugin:local-tts|speak_local_tts")).toEqual([
      { request: expect.objectContaining({ language: "ja", text: "それではまた。" }), utteranceId: "g3" },
    ]);
    expect(calls("plugin:local-tts|load_local_tts")).toHaveLength(2);
    gemini.dispose();
  });

  it("a 429 falls back to Kokoro at once and honours Google's retry delay", async () => {
    data.gemini.apiKey = "k";
    scriptIpc(async () => {
      throw "[cloud-tts:rate_limited:retry=21000] Gemini TTS: the request limit or the free quota is used up for now (HTTP 429).";
    });
    const gemini = new TTS_GeminiService(receiver());
    await gemini.start(data);
    await flush(40);
    const before = Date.now();
    await gemini.synthesizeToSink("Have a nice day.", ctx("g1", "en"), signal());
    expect(calls("plugin:gemini-tts|gemini_tts_speak")[0]).toMatchObject({
      request: { apiKey: "k", model: GEMINI_TTS_MODELS[0].id, voice: "Kore", language: "en", timeoutMs: 3000 },
      utteranceId: "g1",
    });
    expect(calls("plugin:local-tts|speak_local_tts")).toEqual([
      { request: expect.objectContaining({ text: "Have a nice day.", language: "en" }), utteranceId: "g1" },
    ]);
    const until = gemini.chain.breaker.openUntil;
    expect(until - before).toBeGreaterThanOrEqual(20_900);
    expect(until - before).toBeLessThan(22_000);
    // A corrected key is tried on the next sentence.
    data.gemini.apiKey = "k2";
    await gemini.synthesizeToSink("Again.", ctx("g2"), signal());
    expect(calls("plugin:gemini-tts|gemini_tts_speak")).toHaveLength(2);
    gemini.dispose();
  });
});
