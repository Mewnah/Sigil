/**
 * Gemini Live STT backend (P6-K): protocol, seam dedupe, the session rotator (no caption gap across a rotation or a
 * reconnect), the WebSocket connection, the ephemeral token, the Live Translate helpers and the service driver.
 * Everything runs offline: sessions, sockets, token transports, the clock and the timers are fakes.
 */
import i18n from "i18next";
import { afterEach, beforeAll, beforeEach, describe, expect, it, vi } from "vitest";

const tauri = vi.hoisted(() => ({
  audioHandler: null as ((event: { payload: unknown }) => void) | null,
}));

vi.mock("@tauri-apps/api/core", () => ({ invoke: vi.fn(async () => undefined), isTauri: () => true, Channel: class {} }));
vi.mock("@tauri-apps/api/event", () => ({
  listen: vi.fn(async (_event: string, handler: (event: { payload: unknown }) => void) => {
    tauri.audioHandler = handler;
    return () => {
      if (tauri.audioHandler === handler) tauri.audioHandler = null;
    };
  }),
}));
vi.mock("@/utils/devLog", () => ({ devLog: () => {} }));
vi.mock("@/utils/rustAudioInput", () => ({
  resolveRustInputDeviceForInvoke: (d: string | undefined) => (d && d.trim() ? d.trim() : null),
}));

import { invoke } from "@tauri-apps/api/core";
import { ServiceNetworkState, TextEventSource, TextEventType } from "@/types";
import { PROCESS_NONCE } from "@/utils/processNonce";
import type { LlmHttpRequest, LlmHttpResponse, LlmHttpTransport } from "@/core/services/llm/transport";
import { LlmTransportError } from "@/core/services/llm/transport";
import {
  buildLiveTranslateSetups,
  geminiLiveModelName,
  geminiLiveTranslatesStt,
  isLiveTranslatedStt,
  LIVE_TRANSLATE_INPUT_IDLE_MS,
  LIVE_TRANSLATE_OUTPUT_IDLE_MS,
  LIVE_TRANSLATE_STT_STREAM,
  LiveTranslateNormalizer,
  LiveTranslationPublisher,
  liveTranslateTargetCode,
  mergeTranscriptPiece,
} from "@/core/services/translation/services/geminiLive";
import type { TranslationOutputEvent } from "@/core/services/translation/pipeline";
import { Service_STT_Schema, type STT_State } from "../schema";
import {
  alignSeam,
  allowedEdits,
  AUDIO_RING_MS,
  authTokenUrl,
  BACKOFF_MS,
  buildTranscribeSetups,
  classifyClose,
  closeDetail,
  containedSeam,
  createContentHandler,
  DRAIN_MIN_MS,
  floatToPcm16,
  GEMINI_LIVE_SESSION_LIMIT_MS,
  GeminiLiveConnection,
  geminiLiveDeps,
  geminiLiveErrorMessage,
  GeminiLiveRotator,
  liveLanguageCodes,
  LiveSegmentOutput,
  LiveSessionError,
  liveSocketUrl,
  MAX_FAILURES,
  MAX_VOCABULARY,
  mintGeminiLiveToken,
  mintGeminiLiveTokenAnyVersion,
  OVERLAP_MS,
  parseDurationMs,
  parseServerMessage,
  parseVocabulary,
  PAUSE_WAIT_MS,
  pcm16ToBase64,
  PcmFramer,
  PREPARE_LEAD_MS,
  QuietTracker,
  remainderAfter,
  resolveGeminiLiveConfig,
  rotateAfterMs,
  SEAM_GRACE_MS,
  SeamGate,
  seamMaxUnits,
  SESSION_SAFETY_MS,
  seamUnits,
  STT_GeminiLiveService,
  tokenHttpError,
  TranscribeNormalizer,
  type LiveChannel,
  type LiveContentHandler,
  type LiveRotatorStatus,
  type LiveSessionCallbacks,
  type LiveSessionHandle,
  type WebSocketLike,
} from "../services/geminiLive";
import type { GeminiLiveMode } from "../schema/geminiLive";
import type { ISTTReceiver, SttTextMeta } from "../types";
import { initSttTestI18n, translated } from "./sttTestI18n";

beforeAll(async () => {
  await initSttTestI18n();
});

/** Lets promise callbacks (session opens, `then` chains) run. */
async function settle(rounds = 12): Promise<void> {
  for (let i = 0; i < rounds; i++) await Promise.resolve();
}

const KEY = "AIzaSyD-TESTKEY-0123456789abcdefghijklm";

// ---------------------------------------------------------------------------------------------------------------
// Protocol

describe("Gemini Live protocol", () => {
  it("builds the token and socket URLs without the API key", () => {
    expect(authTokenUrl("v1beta")).toBe("https://generativelanguage.googleapis.com/v1beta/auth_tokens");
    expect(authTokenUrl("v1alpha")).toBe("https://generativelanguage.googleapis.com/v1alpha/auth_tokens");
    expect(authTokenUrl("bogus")).toBe("https://generativelanguage.googleapis.com/v1beta/auth_tokens");
    const url = liveSocketUrl("v1beta", "auth_tokens/abc+def=");
    expect(url).toBe(
      "wss://generativelanguage.googleapis.com/ws/google.ai.generativelanguage.v1beta.GenerativeService.BidiGenerateContentConstrained?access_token=auth_tokens/abc%2Bdef%3D",
    );
    expect(liveSocketUrl("v1alpha", "t")).toContain(".v1alpha.GenerativeService.BidiGenerateContentConstrained?access_token=t");
    expect(url).not.toContain("key=");
  });

  it("turns the language setting into a languageCodes hint", () => {
    expect(liveLanguageCodes("auto")).toEqual([]);
    expect(liveLanguageCodes("")).toEqual([]);
    expect(liveLanguageCodes("  ")).toEqual([]);
    expect(liveLanguageCodes("en-US")).toEqual(["en-US"]);
    expect(liveLanguageCodes("ja")).toEqual(["ja"]);
    expect(liveLanguageCodes("cmn-Hans-CN")).toEqual(["cmn-Hans-CN"]);
    expect(liveLanguageCodes("not a tag")).toEqual([]);
  });

  it("parses the custom vocabulary", () => {
    expect(parseVocabulary("Sigil, VRChat;  sigil \n Kokoro、こんにちは，テスト")).toEqual(["Sigil", "VRChat", "Kokoro", "こんにちは", "テスト"]);
    expect(parseVocabulary("")).toEqual([]);
    expect(parseVocabulary("x".repeat(150))).toEqual(["x".repeat(100)]);
    const many = Array.from({ length: 1200 }, (_, i) => `term${i}`).join(",");
    expect(parseVocabulary(many)).toHaveLength(MAX_VOCABULARY);
  });

  it("builds the Transcribe Live setup (documented layout) with a fallback without the options", () => {
    const plain = buildTranscribeSetups({ model: "gemini-3.5-transcribe-live", language: "auto", formatting: "verbatim", vocabulary: [] });
    expect(plain).toEqual([
      {
        setup: {
          model: "models/gemini-3.5-transcribe-live",
          generationConfig: { responseModalities: ["TEXT"] },
          inputAudioTranscription: { languageCodes: [] },
        },
      },
    ]);
    const rich = buildTranscribeSetups({ model: "models/x", language: "ja-JP", formatting: "smart", vocabulary: ["Sigil"] });
    expect(rich).toHaveLength(2);
    expect(rich[0]).toEqual({
      setup: {
        model: "models/x",
        generationConfig: { responseModalities: ["TEXT"] },
        inputAudioTranscription: { languageCodes: ["ja-JP"], customVocabulary: ["Sigil"], mode: "SMART" },
      },
    });
    expect(rich[1]).toEqual({
      setup: { model: "models/x", generationConfig: { responseModalities: ["TEXT"] }, inputAudioTranscription: { languageCodes: ["ja-JP"] } },
    });
  });

  it("builds the Live Translate setup (documented layout first)", () => {
    expect(geminiLiveModelName(" models/gemini-3.5-live-translate-preview ")).toBe("models/gemini-3.5-live-translate-preview");
    expect(geminiLiveModelName("")).toBe("");
    expect(liveTranslateTargetCode("zh")).toBe("zh-Hans");
    expect(liveTranslateTargetCode("zh-TW")).toBe("zh-Hant");
    expect(liveTranslateTargetCode("pl")).toBe("pl");
    expect(liveTranslateTargetCode("auto")).toBe("");
    const setups = buildLiveTranslateSetups({ model: "gemini-3.5-live-translate-preview", targetLanguageCode: "pl", echoTargetLanguage: true });
    expect(setups[0]).toEqual({
      setup: {
        model: "models/gemini-3.5-live-translate-preview",
        generationConfig: {
          responseModalities: ["AUDIO"],
          inputAudioTranscription: {},
          outputAudioTranscription: {},
          translationConfig: { targetLanguageCode: "pl", echoTargetLanguage: true },
        },
      },
    });
    expect(setups[1]).toMatchObject({ setup: { inputAudioTranscription: {}, outputAudioTranscription: {} } });
  });

  it("parses server messages", () => {
    expect(parseServerMessage(JSON.stringify({ setupComplete: {} }))).toEqual([{ kind: "setupComplete" }]);
    expect(parseServerMessage(JSON.stringify({ serverContent: { interimInputTranscription: { text: "hel" } } }))).toEqual([
      { kind: "content", interimInput: "hel", turnComplete: false },
    ]);
    expect(
      parseServerMessage(JSON.stringify({ serverContent: { interimInputTranscription: { text: "Bonj", languageCode: "fr-FR" } } })),
    ).toEqual([{ kind: "content", interimInput: "Bonj", interimLanguageCode: "fr-FR", turnComplete: false }]);
    expect(parseServerMessage(JSON.stringify({ serverContent: { inputTranscription: { text: "Hello.", languageCode: "en-US" } } }))).toEqual([
      { kind: "content", input: { text: "Hello.", languageCode: "en-US" }, turnComplete: false },
    ]);
    expect(
      parseServerMessage(JSON.stringify({ serverContent: { outputTranscription: { text: "Cześć" }, turnComplete: true } })),
    ).toEqual([{ kind: "content", output: "Cześć", turnComplete: true }]);
    expect(parseServerMessage(JSON.stringify({ serverContent: { generationComplete: true } }))).toEqual([{ kind: "content", turnComplete: true }]);
    expect(parseServerMessage(JSON.stringify({ goAway: { timeLeft: "12s" } }))).toEqual([{ kind: "goAway", timeLeftMs: 12_000 }]);
    expect(parseServerMessage(JSON.stringify({ goAway: {} }))).toEqual([{ kind: "goAway", timeLeftMs: null }]);
    expect(parseServerMessage("not json")).toEqual([{ kind: "other" }]);
    expect(parseServerMessage(JSON.stringify({ usageMetadata: {} }))).toEqual([{ kind: "other" }]);
    expect(parseDurationMs("1.5s")).toBe(1500);
    expect(parseDurationMs({ seconds: 2, nanos: 500_000_000 })).toBe(2500);
    expect(parseDurationMs("soon")).toBeNull();
    expect(parseDurationMs(undefined)).toBeNull();
  });

  it("classifies closes and redacts their reasons", () => {
    expect(classifyClose(1011, "Resource has been exhausted (e.g. check quota).")).toBe("quota");
    expect(classifyClose(1008, "API key not valid. Please pass a valid API key.")).toBe("auth");
    expect(classifyClose(1007, "Request contains an invalid argument.")).toBe("config");
    expect(classifyClose(1008, "models/x is not found for API version v1beta")).toBe("config");
    expect(classifyClose(1011, "Internal error encountered.")).toBe("transient");
    expect(classifyClose(1006, "")).toBe("transient");
    expect(classifyClose(1000, "")).toBe("transient");
    expect(closeDetail(1008, `bad key ${KEY}`)).toBe("1008: bad key [redacted]");
    expect(closeDetail(1006, "")).toBe("1006");
  });

  it("maps token request failures", () => {
    expect(tokenHttpError(401, "", {}, KEY).kind).toBe("auth");
    expect(tokenHttpError(403, "", {}, KEY).kind).toBe("auth");
    const quota = tokenHttpError(429, JSON.stringify({ error: { message: "quota" } }), { "retry-after": "7" }, KEY);
    expect(quota.kind).toBe("quota");
    expect(quota.info.retryAfterMs).toBe(7000);
    expect(tokenHttpError(400, JSON.stringify({ error: { message: "API key not valid. Please pass a valid API key." } }), {}, KEY).kind).toBe("auth");
    expect(tokenHttpError(400, JSON.stringify({ error: { message: "Invalid JSON payload" } }), {}, KEY).kind).toBe("config");
    expect(tokenHttpError(500, "", {}, KEY).kind).toBe("transient");
    expect(tokenHttpError(400, JSON.stringify({ error: { message: `key ${KEY} is bad` } }), {}, KEY).message).not.toContain(KEY);
  });

  it("clamps the rotation interval", () => {
    expect(rotateAfterMs("9.5")).toBe(570_000);
    expect(rotateAfterMs("")).toBe(570_000);
    expect(rotateAfterMs("0")).toBe(570_000);
    expect(rotateAfterMs("abc")).toBe(570_000);
    expect(rotateAfterMs("0.2")).toBe(60_000);
    expect(rotateAfterMs("30")).toBe(585_000);
    expect(rotateAfterMs(5)).toBe(300_000);
  });

  it("resolves the settings into a session configuration", () => {
    const base = Service_STT_Schema.parse({}).geminiLive;
    expect(base.model).toBe("gemini-3.5-transcribe-live");
    expect(base.translateModel).toBe("gemini-3.5-live-translate-preview");
    expect(base.mode).toBe("transcribe");
    const noKey = resolveGeminiLiveConfig({ geminiLive: base }, "");
    expect(noKey).toBe(i18n.t("stt.gemini_error_no_key"));
    expect(translated(noKey)).toBe(true);

    const shared = resolveGeminiLiveConfig({ geminiLive: { ...base, language: "ja-JP", vocabulary: "Sigil" } }, ` ${KEY} `);
    if (typeof shared === "string") throw new Error(shared);
    expect(shared.apiKey).toBe(KEY);
    expect(shared.mode).toBe("transcribe");
    expect(shared.sourceLang).toBe("ja");
    expect(shared.setups).toHaveLength(2);
    expect(shared.setupFallbackDropsOptions).toBe(true);
    expect(shared.rotateAfterMs).toBe(570_000);

    const own = resolveGeminiLiveConfig({ geminiLive: { ...base, apiKey: "own-key-123456" } }, KEY);
    if (typeof own === "string") throw new Error(own);
    expect(own.apiKey).toBe("own-key-123456");
    expect(own.sourceLang).toBeNull();

    const tr = resolveGeminiLiveConfig({ geminiLive: { ...base, mode: "translate", translateTo: "zh-CN" } }, KEY);
    if (typeof tr === "string") throw new Error(tr);
    expect(tr.mode).toBe("translate");
    expect(tr.targetLang).toBe("zh");
    expect(tr.setups[0]).toMatchObject({ setup: { generationConfig: { translationConfig: { targetLanguageCode: "zh-Hans" } } } });

    expect(resolveGeminiLiveConfig({ geminiLive: { ...base, mode: "translate", translateTo: "auto" } }, KEY)).toBe(i18n.t("stt.gemini_error_target"));
    expect(resolveGeminiLiveConfig({ geminiLive: { ...base, model: " " } }, KEY)).toBe(i18n.t("stt.gemini_error_model"));
  });

  it("words fatal errors for the user", () => {
    for (const kind of ["auth", "config", "quota", "transient"] as const) {
      const text = geminiLiveErrorMessage(new LiveSessionError(kind, "detail"));
      expect(translated(text)).toBe(true);
      expect(text).toContain("detail");
    }
    expect(geminiLiveErrorMessage(new LiveSessionError("transient", "x"), true)).toBe(i18n.t("stt.gemini_error_connect", { detail: "x" }));
  });
});

// ---------------------------------------------------------------------------------------------------------------
// Audio

describe("Gemini Live audio", () => {
  it("converts samples to 16-bit PCM and base64", () => {
    expect(Array.from(floatToPcm16(new Float32Array([0, 1, -1, 2, -2, 0.5])))).toEqual([0, 32767, -32768, 32767, -32768, 16384]);
    const b64 = pcm16ToBase64(new Int16Array([1, -2]));
    expect(Array.from(Buffer.from(b64, "base64"))).toEqual([1, 0, 0xfe, 0xff]);
  });

  it("cuts the stream into 100 ms frames", () => {
    const framer = new PcmFramer();
    const frames: Int16Array[] = [];
    framer.push(new Float32Array(1000).fill(0.25), (f) => frames.push(f));
    expect(frames).toHaveLength(0);
    framer.push(new Float32Array(2500).fill(0.25), (f) => frames.push(f));
    expect(frames.map((f) => f.length)).toEqual([1600, 1600]);
    framer.reset();
    framer.push(new Float32Array(1599), (f) => frames.push(f));
    expect(frames).toHaveLength(2);
  });

  it("tracks pauses against an adaptive noise floor", () => {
    const quiet = new QuietTracker();
    const speech = new Int16Array(1600).map((_, i) => (i % 2 ? 8000 : -8000));
    const noise = new Int16Array(1600).map((_, i) => (i % 2 ? 40 : -40));
    expect(quiet.quietFor(0)).toBe(Number.POSITIVE_INFINITY);
    quiet.update(noise, 0);
    quiet.update(speech, 100);
    expect(quiet.quietFor(100)).toBe(0);
    quiet.update(noise, 200);
    quiet.update(noise, 300);
    expect(quiet.quietFor(300)).toBe(200);
  });
});

// ---------------------------------------------------------------------------------------------------------------
// Seam dedupe

describe("Gemini Live seam dedupe", () => {
  const units = (text: string) => seamUnits(text);

  it("splits text into comparable units", () => {
    expect(seamUnits("Hello, World! it's").map((u) => u.norm)).toEqual(["hello", "world", "it's"]);
    expect(seamUnits("今日はいい天気。").map((u) => u.norm)).toEqual(["今", "日", "は", "い", "い", "天", "気"]);
    expect(seamUnits("Sigilは最高").map((u) => u.norm)).toEqual(["sigil", "は", "最", "高"]);
    expect(seamUnits(" … ")).toEqual([]);
  });

  it("tolerates more differences in longer seams", () => {
    expect([1, 2, 3, 7, 8, 12, 24, 40, 80].map(allowedEdits)).toEqual([0, 0, 1, 1, 2, 3, 6, 8, 8]);
    expect(seamMaxUnits(1000, false)).toBe(8);
    expect(seamMaxUnits(8000, false)).toBe(36);
    expect(seamMaxUnits(1000, true)).toBe(16);
    expect(seamMaxUnits(0, false)).toBe(4);
    expect(seamMaxUnits(100_000, true)).toBe(80);
  });

  it("finds where the new transcript repeats the end of the old one", () => {
    const a = units("I went to the store and bought some milk");
    expect(alignSeam(a, units("bought some milk and eggs"), 20)).toEqual({ k: 3, keepLast: false, cost: 0 });
    expect(alignSeam(a, units("completely different words"), 20)).toBeNull();
    // The new session started inside a word.
    expect(alignSeam(units("we should probably go home now"), units("ably go home now and eat"), 20)?.k).toBe(4);
    // The old session was cut inside its last word: the new one's complete word is kept.
    expect(alignSeam(units("let's meet at the stat"), units("at the station tomorrow"), 20)).toEqual({ k: 3, keepLast: true, cost: 0 });
    // Small recognition differences.
    expect(alignSeam(units("the weather is really nice today in Tokyo"), units("the whether is really nice today in Tokyo and"), 20)).toEqual({
      k: 8,
      keepLast: false,
      cost: 1,
    });
    // The old session's last word was misheard: its second-to-last word anchors.
    expect(alignSeam(units("I will see you at nine"), units("see you at night tonight"), 20)).toEqual({ k: 3, keepLast: false, cost: 1 });
    // A seam never covers more than the overlap allows.
    const long = "one two three four five six seven eight nine ten eleven twelve";
    expect(alignSeam(units(long), units(`${long} thirteen`), 6)).toBeNull();
  });

  it("finds a new sentence that lies inside the old transcript", () => {
    const a = units("I think so. Let's go now.");
    expect(containedSeam(a, units("I think so."), 20)).toBe(3);
    expect(containedSeam(a, units("Let's go"), 20)).toBe(5);
    expect(containedSeam(a, units("something else"), 20)).toBe(-1);
  });

  it("cuts the repeated part off cleanly", () => {
    const text = "Milk, and eggs.";
    expect(remainderAfter(text, seamUnits(text), 1)).toBe("and eggs.");
    expect(remainderAfter(text, seamUnits(text), 0)).toBe(text);
    expect(remainderAfter(text, seamUnits(text), 3)).toBe("");
    const ja = "天気ですね。明日も晴れるかな";
    expect(remainderAfter(ja, seamUnits(ja), 5)).toBe("明日も晴れるかな");
  });

  it("gates the new session's first utterance", () => {
    const gate = new SeamGate("And then we went to the market to buy some apples and oranges too.", 30);
    expect(gate.interim("to buy some")).toBe("");
    expect(gate.interim("to buy some apples and oranges too and")).toBe("and");
    expect(gate.armed).toBe(true);
    expect(gate.final("to buy some apples and oranges too, and then home.")).toBe("and then home.");
    expect(gate.armed).toBe(false);
    expect(gate.final("anything")).toBe("anything");
  });

  it("follows the old text through sentences the new session split off", () => {
    const gate = new SeamGate("I think so. Let's go now.", 20);
    expect(gate.final("I think so.")).toBe("");
    expect(gate.armed).toBe(true);
    expect(gate.final("Let's go now. Okay?")).toBe("Okay?");
    expect(gate.armed).toBe(false);
  });

  it("passes unrelated text and disarms", () => {
    const gate = new SeamGate("see you tomorrow", 20);
    expect(gate.final("What a nice day.")).toBe("What a nice day.");
    expect(gate.armed).toBe(false);
    expect(new SeamGate("  ", 20).armed).toBe(false);
  });

  it("dedupes scripts without spaces character by character", () => {
    const gate = new SeamGate("今日はいい天気ですね。", 30);
    expect(gate.final("天気ですね。明日も晴れるかな")).toBe("明日も晴れるかな");
  });
});

// ---------------------------------------------------------------------------------------------------------------
// Normalizers and output stages

interface Recorded {
  channel: LiveChannel;
  kind: "interim" | "final";
  text: string;
  lang?: string;
}

function recordingSinks(out: Recorded[]) {
  const sink = (channel: LiveChannel) => ({
    interim: (text: string, lang?: string) => out.push(lang ? { channel, kind: "interim", text, lang } : { channel, kind: "interim", text }),
    final: (text: string, lang?: string) => out.push(lang ? { channel, kind: "final", text, lang } : { channel, kind: "final", text }),
  });
  return { source: sink("source"), translation: sink("translation") };
}

describe("Gemini Live normalizers", () => {
  it("normalizes Transcribe Live hypotheses and segments", () => {
    const out: Recorded[] = [];
    const n = new TranscribeNormalizer(recordingSinks(out).source, "ja");
    n.interim("hel");
    n.interim("hel");
    n.interim(" hello ");
    n.final("Hello.", "en-US");
    n.final("");
    n.interim("open");
    n.final("");
    n.interim("bye");
    expect(n.open).toBe(true);
    n.flushOpen();
    expect(n.open).toBe(false);
    expect(out).toEqual([
      { channel: "source", kind: "interim", text: "hel", lang: "ja" },
      { channel: "source", kind: "interim", text: "hello", lang: "ja" },
      { channel: "source", kind: "final", text: "Hello.", lang: "en" },
      { channel: "source", kind: "interim", text: "open", lang: "en" },
      { channel: "source", kind: "interim", text: "", lang: "en" },
      { channel: "source", kind: "interim", text: "bye", lang: "en" },
      { channel: "source", kind: "final", text: "bye", lang: "en" },
    ]);
  });

  it("closes a segment before the next hypothesis in the same frame", () => {
    const out: Recorded[] = [];
    const h = createContentHandler("transcribe", (ev) => out.push(ev));
    h.message({ kind: "content", input: { text: "One." }, interimInput: "two", turnComplete: false });
    expect(out.map((e) => `${e.kind}:${e.text}`)).toEqual(["final:One.", "interim:two"]);
  });

  it("takes the detected language from hypotheses too", () => {
    const out: Recorded[] = [];
    const h = createContentHandler("transcribe", (ev) => out.push(ev));
    h.message({ kind: "content", interimInput: "Bonj", interimLanguageCode: "fr-FR", turnComplete: false });
    h.message({ kind: "content", input: { text: "Bonjour." }, turnComplete: false });
    expect(out.map((e) => `${e.kind}:${e.text}:${e.lang}`)).toEqual(["interim:Bonj:fr", "final:Bonjour.:fr"]);
  });

  it("joins streamed transcription pieces", () => {
    expect(mergeTranscriptPiece("", "Hello")).toBe("Hello");
    expect(mergeTranscriptPiece("Hello", " there")).toBe("Hello there");
    expect(mergeTranscriptPiece("Hello there", "Hello there friend")).toBe("Hello there friend");
    expect(mergeTranscriptPiece("no", "no")).toBe("nono");
    expect(mergeTranscriptPiece("abc", "")).toBe("abc");
  });

  it("assembles Live Translate lines on both channels", () => {
    let now = 0;
    const out: Recorded[] = [];
    const n = new LiveTranslateNormalizer(recordingSinks(out), () => now);
    n.input("Hola", "es-ES");
    n.input(" amigo");
    n.output("Hello");
    n.output(" friend");
    expect(n.open).toBe(true);
    n.turnComplete();
    expect(n.open).toBe(false);
    expect(out).toEqual([
      { channel: "source", kind: "interim", text: "Hola", lang: "es" },
      { channel: "source", kind: "interim", text: "Hola amigo", lang: "es" },
      { channel: "translation", kind: "interim", text: "Hello" },
      { channel: "translation", kind: "interim", text: "Hello friend" },
      { channel: "source", kind: "final", text: "Hola amigo", lang: "es" },
      { channel: "translation", kind: "final", text: "Hello friend" },
    ]);
    out.length = 0;
    n.input("¿Qué tal?");
    n.output("How are");
    now += LIVE_TRANSLATE_INPUT_IDLE_MS;
    n.poll();
    expect(out.at(-1)).toEqual({ channel: "source", kind: "final", text: "¿Qué tal?", lang: "es" });
    now += LIVE_TRANSLATE_OUTPUT_IDLE_MS - LIVE_TRANSLATE_INPUT_IDLE_MS;
    n.poll();
    expect(out.at(-1)).toEqual({ channel: "translation", kind: "final", text: "How are" });
  });

  it("switches to segments when the session sends hypotheses", () => {
    const out: Recorded[] = [];
    const n = new LiveTranslateNormalizer(recordingSinks(out), () => 0);
    n.input("joined");
    n.interimInput("seg");
    n.input("Segment one.");
    n.interimInput("");
    n.input("");
    expect(out.map((e) => `${e.kind}:${e.text}`)).toEqual(["interim:joined", "final:joined", "interim:seg", "final:Segment one.", "interim:", "interim:"]);
  });

  it("publishes translated lines like the translation pipeline", () => {
    const events: TranslationOutputEvent[] = [];
    let muted = false;
    let interims = true;
    const p = new LiveTranslationPublisher({
      epoch: 7,
      lang: "en",
      publish: (e) => events.push(e),
      interims: () => interims,
      muted: () => muted,
    });
    p.interim("Hello");
    p.interim("Hello fri");
    p.interim("Hello friend, how");
    p.final("Hello friend, how are you?");
    const id = `gemini.tr.${PROCESS_NONCE}.7:1`;
    expect(events).toEqual([
      { type: TextEventType.interim, value: "Hello", segmentId: id, revision: 1, lang: "en", stableLength: 0 },
      { type: TextEventType.interim, value: "Hello fri", segmentId: id, revision: 2, lang: "en", stableLength: 5 },
      { type: TextEventType.interim, value: "Hello friend, how", segmentId: id, revision: 3, lang: "en", stableLength: 13 },
      { type: TextEventType.final, value: "Hello friend, how are you?", segmentId: id, revision: 4, lang: "en" },
    ]);
    events.length = 0;
    interims = false;
    p.interim("Second");
    p.final("Second line.");
    expect(events).toEqual([{ type: TextEventType.final, value: "Second line.", segmentId: `gemini.tr.${PROCESS_NONCE}.7:2`, revision: 1, lang: "en" }]);
    events.length = 0;
    muted = true;
    p.final("hidden");
    expect(events).toEqual([]);
    expect(id).toMatch(/^[A-Za-z0-9._:-]{1,128}$/);
  });

  it("publishes captions with segment ids and rising revisions", () => {
    const calls: [string, string, SttTextMeta | undefined][] = [];
    let interims = true;
    const o = new LiveSegmentOutput(
      "gemini-x:1",
      { interim: (t, m) => calls.push(["interim", t, m]), final: (t, m) => calls.push(["final", t, m]) },
      () => interims,
      "de",
    );
    o.interim("hal");
    o.interim("hal");
    o.interim("hallo", "de-DE");
    o.final("Hallo.", "en");
    o.interim("x");
    o.interim("");
    o.interim("");
    interims = false;
    o.interim("hidden");
    o.final("Shown.");
    expect(calls).toEqual([
      ["interim", "hal", { segmentId: "gemini-x:1:1", revision: 1, lang: "de" }],
      ["interim", "hallo", { segmentId: "gemini-x:1:1", revision: 2, lang: "de" }],
      ["final", "Hallo.", { segmentId: "gemini-x:1:1", revision: 3, lang: "en" }],
      ["interim", "x", { segmentId: "gemini-x:1:2", revision: 1, lang: "de" }],
      ["interim", "", { segmentId: "gemini-x:1:2", revision: 1, lang: "de" }],
      ["final", "Shown.", { segmentId: "gemini-x:1:3", revision: 1, lang: "de" }],
    ]);
  });

  it("tells the translation service when Live Translate already translates", () => {
    expect(geminiLiveTranslatesStt({ backend: "geminiLive", geminiLive: { mode: "translate" } })).toBe(true);
    expect(geminiLiveTranslatesStt({ backend: "geminiLive", geminiLive: { mode: "transcribe" } })).toBe(false);
    expect(geminiLiveTranslatesStt({ backend: "local", geminiLive: { mode: "translate" } })).toBe(false);
    expect(geminiLiveTranslatesStt(null)).toBe(false);
    expect(isLiveTranslatedStt({ segmentId: `${LIVE_TRANSLATE_STT_STREAM}-abc123:1:4` })).toBe(true);
    expect(isLiveTranslatedStt({ segmentId: "gemini-abc123:1:4" })).toBe(false);
    expect(isLiveTranslatedStt({ segmentId: "mic-abc123:1:4" })).toBe(false);
    expect(isLiveTranslatedStt({})).toBe(false);
    expect(isLiveTranslatedStt(undefined)).toBe(false);
  });
});

// ---------------------------------------------------------------------------------------------------------------
// Rotator harness: fake sessions (real content normalizers), a fake clock and manual timers

type Behavior = "ready" | "pending" | LiveSessionError;

class FakeSession {
  readonly frames: number[] = [];
  ended = 0;
  /** How many frames the session had got at each `audioStreamEnd`. */
  readonly endedAfter: number[] = [];
  closed = false;
  /** The rotator aborted the open (token request, socket setup). */
  aborted = false;
  sendOk = true;
  readonly content: LiveContentHandler;
  ready: () => void = () => {};
  fail: (e: unknown) => void = () => {};
  readonly handle: LiveSessionHandle;

  constructor(
    readonly id: number,
    readonly cb: LiveSessionCallbacks,
    ids: WeakMap<Int16Array, number>,
    mode: GeminiLiveMode,
    now: () => number,
  ) {
    this.content = createContentHandler(mode, (ev) => cb.event(ev), { now });
    this.handle = {
      sendAudio: (f) => {
        if (!this.sendOk || this.closed) return false;
        this.frames.push(ids.get(f) ?? -1);
        return true;
      },
      endAudio: () => {
        this.ended += 1;
        this.endedAfter.push(this.frames.length);
      },
      poll: () => this.content.poll(),
      commit: () => this.content.commit(),
      flushOpen: () => this.content.flushOpen(),
      close: () => {
        this.closed = true;
      },
    };
  }

  // Transcribe Live
  interim(text: string): void {
    this.content.message({ kind: "content", interimInput: text, turnComplete: false });
  }
  final(text: string): void {
    this.content.message({ kind: "content", input: { text }, turnComplete: false });
  }
  // Live Translate
  heard(piece: string): void {
    this.content.message({ kind: "content", input: { text: piece }, turnComplete: false });
  }
  said(piece: string): void {
    this.content.message({ kind: "content", output: piece, turnComplete: false });
  }
  turn(): void {
    this.content.message({ kind: "content", turnComplete: true });
  }
  /** The server closes the session. */
  drop(kind: LiveSessionError["kind"] = "transient", message = "1011: Internal error"): void {
    this.closed = true;
    this.cb.closed(new LiveSessionError(kind, message));
  }
}

class Harness {
  now = 1_000_000;
  readonly sessions: FakeSession[] = [];
  readonly behaviors: Behavior[] = [];
  readonly out: Recorded[] = [];
  readonly statuses: LiveRotatorStatus[] = [];
  fatal: LiveSessionError | null = null;
  readonly rotator: GeminiLiveRotator;
  readonly frameIds = new WeakMap<Int16Array, number>();
  frameCount = 0;
  readonly frameAt = new Map<number, number>();
  #intervals: (() => void)[] = [];
  #timeouts: { at: number; fn: () => void }[] = [];

  constructor(readonly opts: { mode?: GeminiLiveMode; rotateAfterMs?: number; startAttempts?: number } = {}) {
    const mode = opts.mode ?? "transcribe";
    const sinks = recordingSinks(this.out);
    this.rotator = new GeminiLiveRotator({
      mode,
      open: (cb, signal) => this.#open(cb, signal, mode),
      output: { source: sinks.source, translation: mode === "translate" ? sinks.translation : undefined },
      status: (s) => this.statuses.push(s),
      fatal: (e) => {
        this.fatal = e;
      },
      rotateAfterMs: opts.rotateAfterMs ?? 60_000,
      now: () => this.now,
      startAttempts: opts.startAttempts,
      timers: {
        setInterval: (fn) => {
          this.#intervals.push(fn);
          return this.#intervals.length - 1;
        },
        clearInterval: (h) => {
          this.#intervals[h as number] = () => {};
        },
        setTimeout: (fn, ms) => {
          this.#timeouts.push({ at: this.now + ms, fn });
          return null;
        },
        clearTimeout: () => {},
      },
    });
  }

  #open(cb: LiveSessionCallbacks, signal: AbortSignal, mode: GeminiLiveMode): Promise<LiveSessionHandle> {
    const s = new FakeSession(this.sessions.length + 1, cb, this.frameIds, mode, () => this.now);
    this.sessions.push(s);
    const behavior = this.behaviors.shift() ?? "ready";
    return new Promise<LiveSessionHandle>((resolve, reject) => {
      signal.addEventListener(
        "abort",
        () => {
          s.aborted = true;
          reject(new DOMException("Aborted", "AbortError"));
        },
        { once: true },
      );
      s.ready = () => resolve(s.handle);
      s.fail = reject;
      if (behavior === "ready") s.ready();
      else if (behavior instanceof LiveSessionError) reject(behavior);
    });
  }

  session(n: number): FakeSession {
    const s = this.sessions[n - 1];
    if (!s) throw new Error(`no session ${n}`);
    return s;
  }

  async start(): Promise<void> {
    const p = this.rotator.start();
    let done = false;
    p.then(
      () => (done = true),
      () => (done = true),
    );
    await settle();
    for (let i = 0; i < 400 && !done; i++) await this.advance(100, { audio: false });
    return p;
  }

  frame(speech: boolean): Int16Array {
    const f = new Int16Array(1600);
    if (speech) for (let i = 0; i < f.length; i++) f[i] = i % 2 ? 9000 : -9000;
    this.frameCount += 1;
    this.frameIds.set(f, this.frameCount);
    this.frameAt.set(this.frameCount, this.now);
    return f;
  }

  /** Moves the clock in 100 ms steps, pushing one audio frame per step (unless `audio: false`). */
  async advance(ms: number, o: { speech?: boolean; audio?: boolean } = {}): Promise<void> {
    const steps = Math.max(1, Math.round(ms / 100));
    for (let i = 0; i < steps; i++) {
      this.now += 100;
      if (o.audio !== false) this.rotator.pushAudio(this.frame(!!o.speech));
      const due = this.#timeouts.filter((t) => t.at <= this.now);
      this.#timeouts = this.#timeouts.filter((t) => t.at > this.now);
      for (const t of due) t.fn();
      for (const fn of this.#intervals) fn();
      await settle();
    }
  }

  /** Advances until `cond` holds (at most `maxMs`). */
  async until(cond: () => boolean, maxMs: number, o: { speech?: boolean; audio?: boolean } = {}): Promise<void> {
    for (let t = 0; t < maxMs && !cond(); t += 100) await this.advance(100, o);
    expect(cond()).toBe(true);
  }

  finals(channel: LiveChannel = "source"): string[] {
    return this.out.filter((e) => e.channel === channel && e.kind === "final").map((e) => e.text);
  }

  /** Every frame from `from` on reached at least one session (no audio gap). */
  expectNoAudioGap(from = 1): void {
    const heard = new Set<number>();
    for (const s of this.sessions) for (const f of s.frames) heard.add(f);
    const missing: number[] = [];
    for (let i = from; i <= this.frameCount; i++) if (!heard.has(i)) missing.push(i);
    expect(missing).toEqual([]);
  }

  /** A session's frames are one contiguous run (no frame skipped or repeated). */
  expectContiguous(s: FakeSession): void {
    for (let i = 1; i < s.frames.length; i++) expect(s.frames[i]).toBe(s.frames[i - 1] + 1);
  }
}

/** Starts a harness and runs session 1 to just before the next session is prepared. */
async function runToPrepare(h: Harness, rotate = 60_000): Promise<void> {
  await h.start();
  await h.advance(rotate - PREPARE_LEAD_MS - 1_000);
  expect(h.sessions).toHaveLength(1);
}

describe("GeminiLiveRotator", () => {
  it("rotates at a pause with an overlap and no audio gap", async () => {
    const h = new Harness();
    await h.start();
    expect(h.rotator.snapshot).toMatchObject({ primary: 1, stage: "running" });
    await h.advance(3_000, { speech: true });
    h.session(1).interim("hello");
    h.session(1).final("Hello world.");
    await h.advance(60_000 - PREPARE_LEAD_MS - 3_000 - 200);
    expect(h.sessions).toHaveLength(1);
    await h.advance(300);
    // The next session connects ahead of the rotation point and waits.
    expect(h.sessions).toHaveLength(2);
    expect(h.rotator.snapshot).toMatchObject({ primary: 1, next: 2, stage: "ready" });
    await h.advance(PREPARE_LEAD_MS - 300);
    expect(h.rotator.snapshot.stage).toBe("ready");
    expect(h.session(2).frames).toEqual([]);
    await h.advance(200);
    // Rotation point at a pause: both hear the audio (with a pre-roll).
    expect(h.rotator.snapshot.stage).toBe("overlap");
    expect(h.session(2).frames.length).toBeGreaterThan(1);
    expect(h.session(2).frames[0]).toBeLessThan(h.frameCount);
    await h.advance(OVERLAP_MS);
    expect(h.rotator.snapshot.stage).toBe("draining");
    expect(h.session(1).ended).toBe(1);
    const lastOld = h.session(1).frames.at(-1)!;
    await h.advance(DRAIN_MIN_MS + 100);
    expect(h.rotator.snapshot).toMatchObject({ primary: 2, next: null, stage: "running" });
    expect(h.session(1).closed).toBe(true);
    await h.advance(1_000);
    // Audio: session 2 heard everything after session 1 stopped hearing, from before that point.
    expect(h.session(2).frames[0]).toBeLessThan(lastOld);
    h.expectContiguous(h.session(1));
    h.expectContiguous(h.session(2));
    h.expectNoAudioGap(h.session(1).frames[0]);
    expect(h.finals()).toEqual(["Hello world."]);
    expect(h.statuses).toEqual([]);
  });

  it("publishes speech said during the overlap once", async () => {
    const h = new Harness();
    await runToPrepare(h);
    await h.until(() => h.rotator.snapshot.stage === "overlap", 12_000);
    await h.advance(1_500, { speech: true });
    h.session(1).interim("see you");
    h.session(2).interim("see you");
    await h.advance(400);
    // The old session's sentence is still open: no cut yet.
    expect(h.rotator.snapshot.stage).toBe("overlap");
    h.session(1).final("See you tomorrow.");
    await h.advance(300);
    expect(h.rotator.snapshot.stage).toBe("draining");
    h.session(2).final("see you tomorrow");
    await h.until(() => h.rotator.snapshot.primary === 2, 3_000);
    expect(h.finals()).toEqual(["See you tomorrow."]);
    // The next utterance is new speech.
    await h.advance(2_000);
    h.session(2).interim("great");
    h.session(2).final("Great, bye.");
    expect(h.finals()).toEqual(["See you tomorrow.", "Great, bye."]);
    h.expectNoAudioGap(h.session(1).frames[0]);
  });

  it("forces the switch during nonstop speech and dedupes the long seam", async () => {
    const h = new Harness();
    await h.start();
    await h.advance(60_000 - 6_000);
    // Nonstop speech from 54 s: no pause at the rotation point.
    await h.advance(3_000, { speech: true });
    h.session(1).final("This is part one of my story.");
    h.session(1).interim("and then");
    await h.until(() => h.rotator.snapshot.stage === "overlap", PAUSE_WAIT_MS + 4_000, { speech: true });
    // Forced at the latest PAUSE_WAIT_MS after the rotation point, with a full pre-roll.
    await h.advance(2_000, { speech: true });
    h.session(1).final("and then we went to the market to buy some apples");
    h.session(2).interim("to buy some apples");
    await h.advance(2_000, { speech: true });
    h.session(1).interim("and oranges");
    h.session(2).interim("to buy some apples and oranges");
    await h.advance(1_000, { speech: true });
    h.session(1).final("and oranges too.");
    await h.until(() => h.rotator.snapshot.stage === "draining", 1_000);
    h.session(2).final("to buy some apples and orange too");
    h.session(2).interim("and we");
    await h.until(() => h.rotator.snapshot.primary === 2, 4_000);
    h.session(2).final("and we went home.");
    expect(h.finals()).toEqual([
      "This is part one of my story.",
      "and then we went to the market to buy some apples",
      "and oranges too.",
      "and we went home.",
    ]);
    // Interims never showed the repeated words.
    const interimsAfterSwitch = h.out.filter((e) => e.kind === "interim").map((e) => e.text);
    expect(interimsAfterSwitch).not.toContain("to buy some apples");
    expect(interimsAfterSwitch).toContain("and we");
    h.expectNoAudioGap(h.session(1).frames[0]);
    h.expectContiguous(h.session(2));
  });

  it("drops a repeat the new session reports a little late, but not later new speech", async () => {
    const h = new Harness();
    await runToPrepare(h);
    await h.until(() => h.rotator.snapshot.stage === "overlap", 12_000);
    await h.advance(500, { speech: true });
    h.session(1).interim("okay");
    h.session(1).final("Okay.");
    await h.until(() => h.rotator.snapshot.primary === 2, 4_000);
    expect(h.rotator.snapshot.seams).toEqual(["source"]);
    // Session 2 reports the same word only now.
    h.session(2).interim("okay");
    h.session(2).final("okay");
    expect(h.finals()).toEqual(["Okay."]);
    expect(h.out.filter((e) => e.kind === "interim" && e.text === "okay").length).toBe(1);
    // After the grace, the same word is new speech.
    await h.advance(SEAM_GRACE_MS.transcribe + 200);
    h.session(2).interim("okay");
    h.session(2).final("Okay!");
    expect(h.finals()).toEqual(["Okay.", "Okay!"]);
    expect(h.rotator.snapshot.seams).toEqual([]);
  });

  it("replays the audio after a lost session and does not repeat what was published", async () => {
    const h = new Harness();
    await h.start();
    await h.advance(5_000, { speech: true });
    h.session(1).final("first sentence here");
    const finalFrame = h.frameCount;
    await h.advance(1_500, { speech: true });
    h.session(1).interim("second sent");
    const dropFrame = h.frameCount;
    h.session(1).drop();
    expect(h.statuses.at(-1)).toMatchObject({ kind: "reconnecting", attempt: 1 });
    // The first reconnect goes at once; speech continues meanwhile.
    await h.advance(300, { speech: true });
    expect(h.sessions).toHaveLength(2);
    expect(h.rotator.snapshot.primary).toBe(2);
    expect(h.statuses.at(-1)).toEqual({ kind: "connected" });
    const s2 = h.session(2);
    // Replayed from shortly before the last final, through the drop, up to now.
    expect(s2.frames[0]).toBeLessThanOrEqual(finalFrame - 10);
    expect(s2.frames).toContain(dropFrame + 1);
    h.expectContiguous(s2);
    h.expectNoAudioGap(1);
    s2.interim("here second");
    s2.final("here second sentence");
    expect(h.finals()).toEqual(["first sentence here", "second sentence"]);
    // The lost session's open caption was replaced by the new one, never published twice.
    expect(h.out.filter((e) => e.kind === "interim").map((e) => e.text)).toEqual(["second sent", "second"]);
  });

  it("backs off on repeated failures and gives up after too many", async () => {
    const h = new Harness();
    await h.start();
    await h.advance(1_000, { speech: true });
    const fail = new LiveSessionError("transient", "1006");
    for (let i = 0; i < MAX_FAILURES; i++) h.behaviors.push(fail);
    h.session(1).drop();
    await h.until(() => h.fatal !== null, 60_000, { speech: true });
    expect(h.sessions).toHaveLength(1 + MAX_FAILURES);
    expect(h.fatal?.kind).toBe("transient");
    const attempts = h.statuses.filter((s) => s.kind === "reconnecting").map((s) => (s.kind === "reconnecting" ? s.attempt : 0));
    expect(attempts).toEqual([1, 2, 3, 4, 5, 6]);
    // Every session is closed; nothing more happens.
    const n = h.sessions.length;
    await h.advance(20_000, { speech: true });
    expect(h.sessions).toHaveLength(n);
    expect(BACKOFF_MS.length).toBe(MAX_FAILURES);
  });

  it("stops at once when the key is refused", async () => {
    const h = new Harness();
    await h.start();
    h.session(1).drop("auth", "1008: API key not valid");
    expect(h.fatal?.kind).toBe("auth");
    await h.advance(2_000);
    expect(h.sessions).toHaveLength(1);
  });

  it("retries the first session and fails fast on a refused key", async () => {
    const ok = new Harness();
    ok.behaviors.push(new LiveSessionError("transient", "1006"), "ready");
    await ok.start();
    expect(ok.sessions).toHaveLength(2);
    expect(ok.statuses[0]).toMatchObject({ kind: "reconnecting", attempt: 1 });
    expect(ok.rotator.snapshot.primary).toBe(2);

    const bad = new Harness();
    bad.behaviors.push(new LiveSessionError("auth", "HTTP 401"));
    await expect(bad.start()).rejects.toMatchObject({ kind: "auth" });
    expect(bad.sessions).toHaveLength(1);

    const down = new Harness({ startAttempts: 2 });
    down.behaviors.push(new LiveSessionError("transient", "a"), new LiveSessionError("transient", "b"));
    await expect(down.start()).rejects.toMatchObject({ kind: "transient", message: "b" });
  });

  it("rotates at once when the server announces the end of the session", async () => {
    const h = new Harness({ rotateAfterMs: 570_000 });
    await h.start();
    await h.advance(20_000);
    h.session(1).cb.goAway(5_000);
    await settle();
    expect(h.sessions).toHaveLength(2);
    await h.until(() => h.rotator.snapshot.primary === 2, 6_000);
    expect(h.session(1).closed).toBe(true);
    h.expectNoAudioGap(1);
  });

  it("closes a muted session at its rotation point and reconnects when audio returns", async () => {
    const h = new Harness();
    await h.start();
    await h.advance(2_000, { speech: true });
    h.rotator.setMuted(true);
    expect(h.session(1).ended).toBe(1);
    // The capture hub withholds audio while muted.
    await h.advance(60_000, { audio: false });
    expect(h.sessions).toHaveLength(1);
    expect(h.session(1).closed).toBe(true);
    expect(h.rotator.snapshot).toMatchObject({ primary: null, dormant: true });
    h.rotator.setMuted(false);
    await h.advance(100, { speech: true });
    const first = h.frameCount;
    expect(h.statuses.slice(-2)).toEqual([{ kind: "connecting" }, { kind: "connected" }]);
    await h.advance(300, { speech: true });
    expect(h.rotator.snapshot.primary).toBe(2);
    expect(h.session(2).frames[0]).toBe(first);
    h.expectContiguous(h.session(2));
  });

  it("sends audioStreamEnd to every session hearing audio when muted", async () => {
    const h = new Harness();
    await runToPrepare(h);
    await h.until(() => h.rotator.snapshot.stage === "overlap", 12_000);
    h.rotator.setMuted(true);
    expect(h.session(1).ended).toBe(1);
    expect(h.session(2).ended).toBe(1);
  });

  it("keeps the running session when the next one cannot connect", async () => {
    const h = new Harness();
    await runToPrepare(h);
    h.behaviors.push(new LiveSessionError("transient", "1006"));
    await h.advance(1_100);
    expect(h.sessions).toHaveLength(2);
    expect(h.rotator.snapshot).toMatchObject({ primary: 1, next: null });
    const before = h.session(1).frames.length;
    await h.advance(1_000);
    expect(h.session(1).frames.length).toBe(before + 10);
    expect(h.sessions).toHaveLength(3);
    await h.until(() => h.rotator.snapshot.primary === 3, 15_000);
    h.expectNoAudioGap(1);
    expect(h.statuses).toEqual([]);
  });

  it("hands over at once when the old session drops during the overlap", async () => {
    const h = new Harness();
    await runToPrepare(h);
    await h.until(() => h.rotator.snapshot.stage === "overlap", 12_000);
    await h.advance(300, { speech: true });
    h.session(1).drop();
    expect(h.rotator.snapshot).toMatchObject({ primary: 2, reconnecting: false });
    await h.advance(500, { speech: true });
    expect(h.sessions).toHaveLength(2);
    h.expectNoAudioGap(1);
    expect(h.statuses).toEqual([]);
  });

  it("replays from the overlap when the new session drops while the old one drains", async () => {
    const h = new Harness();
    await runToPrepare(h);
    await h.until(() => h.rotator.snapshot.stage === "overlap", 12_000);
    const overlapFirst = h.session(2).frames[0];
    await h.advance(800, { speech: true });
    h.session(1).final("Good night.");
    await h.until(() => h.rotator.snapshot.stage === "draining", 2_000);
    h.session(2).drop();
    expect(h.rotator.snapshot.reconnecting).toBe(true);
    await h.advance(300);
    expect(h.sessions).toHaveLength(3);
    expect(h.rotator.snapshot.primary).toBe(3);
    expect(h.session(1).closed).toBe(true);
    expect(h.session(3).frames[0]).toBe(overlapFirst);
    h.session(3).final("good night");
    h.session(3).final("Sleep well.");
    expect(h.finals()).toEqual(["Good night.", "Sleep well."]);
    h.expectNoAudioGap(1);
  });

  it("reports a gap when a reconnect took longer than the audio buffer", async () => {
    const h = new Harness();
    await h.start();
    await h.advance(1_000, { speech: true });
    const quota = new LiveSessionError("quota", "429");
    h.behaviors.push(quota, quota, quota);
    h.session(1).drop();
    await h.until(() => h.rotator.snapshot.primary === 5, 60_000, { speech: true });
    expect(h.statuses.some((s) => s.kind === "waiting")).toBe(true);
    // The gap warning comes after "connected", so it stays on the status line.
    expect(h.statuses.slice(-2)).toEqual([{ kind: "connected" }, { kind: "gap" }]);
    // Only audio older than the buffer is missing.
    const s5 = h.session(5);
    expect(h.now - h.frameAt.get(s5.frames[0])!).toBeLessThanOrEqual(AUDIO_RING_MS + 100);
    h.expectContiguous(s5);
  });

  it("dedupes both channels of a Live Translate rotation", async () => {
    const h = new Harness({ mode: "translate" });
    await runToPrepare(h);
    await h.until(() => h.rotator.snapshot.stage === "overlap", 12_000);
    await h.advance(1_200, { speech: true });
    for (const s of [h.session(1), h.session(2)]) {
      s.heard("Hola");
      s.heard(" amigo");
    }
    h.session(1).said("Hello friend");
    h.session(1).turn();
    await h.until(() => h.rotator.snapshot.stage === "draining", 2_000);
    h.session(2).said("Hello, friend.");
    h.session(2).turn();
    await h.until(() => h.rotator.snapshot.primary === 2, 10_000);
    h.session(2).heard("¿Qué tal?");
    h.session(2).said("How are you?");
    h.session(2).turn();
    expect(h.finals("source")).toEqual(["Hola amigo", "¿Qué tal?"]);
    expect(h.finals("translation")).toEqual(["Hello friend", "How are you?"]);
  });

  it("publishes finished Live Translate lines on commit", async () => {
    const h = new Harness({ mode: "translate" });
    await h.start();
    h.session(1).heard("Buenos días");
    h.session(1).said("Good morning");
    h.rotator.commitPending();
    expect(h.finals("source")).toEqual(["Buenos días"]);
    expect(h.finals("translation")).toEqual(["Good morning"]);
  });

  it("sends audio held back by the send buffer on the tick, then audioStreamEnd", async () => {
    const h = new Harness();
    await h.start();
    await h.advance(1_000, { speech: true });
    const s1 = h.session(1);
    s1.sendOk = false;
    await h.advance(2_000, { speech: true });
    const sentBefore = s1.frames.length;
    expect(sentBefore).toBeLessThan(h.frameCount);
    // Muted while frames still wait: audioStreamEnd waits for them.
    h.rotator.setMuted(true);
    expect(s1.ended).toBe(0);
    await h.advance(300, { audio: false });
    expect(s1.frames.length).toBe(sentBefore);
    s1.sendOk = true;
    await h.advance(100, { audio: false });
    // Delivered on the tick without new audio, in order, and only then the end of the stream.
    expect(s1.frames.at(-1)).toBe(h.frameCount);
    h.expectContiguous(s1);
    expect(s1.ended).toBe(1);
    h.expectNoAudioGap(1);
    // Unmuted and muted again with nothing waiting: the end goes out at once.
    h.rotator.setMuted(false);
    await h.advance(200, { speech: true });
    h.rotator.setMuted(true);
    expect(s1.ended).toBe(2);
  });

  it("switches before the session's own end while a translation is still coming", async () => {
    const h = new Harness({ mode: "translate", rotateAfterMs: 585_000 });
    const t0 = h.now;
    await h.start();
    await h.advance(580_000, { speech: true });
    // Nonstop speech, and the old session's line never finishes.
    for (let i = 0; i < 300 && h.rotator.snapshot.primary !== 2; i++) {
      if (i % 10 === 0) h.session(1).heard(` word${i}`);
      await h.advance(100, { speech: true });
    }
    expect(h.rotator.snapshot.primary).toBe(2);
    expect(h.now).toBeLessThanOrEqual(t0 + GEMINI_LIVE_SESSION_LIMIT_MS - SESSION_SAFETY_MS);
    // The old session's open line was published when it was replaced.
    expect(h.finals("source").at(-1)).toMatch(/^word0 .*word\d+$/);
    h.expectNoAudioGap(1);
  });

  it("never gates the new session on a final about speech it did not hear (pause-timed rotation)", async () => {
    for (const [last, next, expected] of [
      // The new session's first word repeats the old session's last word.
      ["I said no", "No way, really?", "No way, really?"],
      // The new session's first utterance lies inside the old session's last sentence.
      ["Yeah, that sounds good to me.", "Yeah.", "Yeah."],
    ] as const) {
      const h = new Harness();
      await h.start();
      await h.advance(60_000 - 2_000, { speech: true });
      h.session(1).interim(last.split(" ")[0]);
      // Still speaking at the rotation point: the next session waits for a pause.
      await h.advance(2_000, { speech: true });
      expect(h.rotator.snapshot.stage).toBe("ready");
      const lastSpeech = h.frameCount;
      // The old session's final arrives 300 ms into the pause; the overlap starts on the next tick, inside the pause.
      await h.advance(300);
      h.session(1).final(last);
      await h.advance(100);
      expect(h.rotator.snapshot.stage).toBe("overlap");
      // The new session never heard the speech of that final.
      expect(h.session(2).frames[0]).toBeGreaterThan(lastSpeech);
      await h.until(() => h.rotator.snapshot.primary === 2, 5_000);
      // The user then speaks again.
      await h.advance(400, { speech: true });
      h.session(2).interim(next.toLowerCase().replace(/[^a-z ]/g, ""));
      h.session(2).final(next);
      expect(h.finals()).toEqual([last, expected]);
      h.expectNoAudioGap(1);
    }
  });

  it("never gates a reconnect's replay on a final about speech before it", async () => {
    const h = new Harness();
    await h.start();
    await h.advance(5_000, { speech: true });
    await h.advance(500);
    // The final arrives 500 ms after its speech ended; the session drops 1.7 s later (idle: a 2 s replay).
    h.session(1).interim("do you want");
    h.session(1).final("Do you want to go?");
    await h.advance(1_700);
    h.session(1).drop();
    await h.advance(300, { speech: true });
    expect(h.rotator.snapshot.primary).toBe(2);
    h.session(2).interim("go where");
    h.session(2).final("Go where?");
    expect(h.finals()).toEqual(["Do you want to go?", "Go where?"]);
  });

  it("replays all speech a stalled session never delivered and reports no gap", async () => {
    const h = new Harness();
    await h.start();
    await h.advance(3_000, { speech: true });
    h.session(1).interim("hello");
    h.session(1).final("Hello.");
    await h.advance(1_000);
    // The network stalls while the user speaks: the socket still takes 6 s of audio (lost with it), then its send
    // buffer is full and the audio waits in the ring until the stall is reported.
    const stallFrom = h.frameCount + 1;
    await h.advance(6_000, { speech: true });
    expect(h.session(1).frames.at(-1)).toBe(h.frameCount);
    h.session(1).sendOk = false;
    await h.advance(4_000, { speech: true });
    h.session(1).drop("transient", "the connection stalled");
    await h.advance(1_000, { speech: true });
    const s2 = h.session(2);
    expect(h.rotator.snapshot.primary).toBe(2);
    // From the start of the speech after the last transcript (with a pre-roll), not just the last 2 s.
    expect(s2.frames[0]).toBeLessThanOrEqual(stallFrom);
    expect(s2.frames[0]).toBeGreaterThan(stallFrom - 6);
    h.expectContiguous(s2);
    expect(s2.frames.at(-1)).toBe(h.frameCount);
    expect(h.statuses.map((s) => s.kind)).toEqual(["reconnecting", "connected"]);
    s2.interim("so as i");
    s2.final("So as I was saying.");
    expect(h.finals()).toEqual(["Hello.", "So as I was saying."]);
  });

  it("reconnects a muted session lost with audio it never sent, and ends that stream after the replay", async () => {
    const h = new Harness();
    await h.start();
    await h.advance(1_000, { speech: true });
    h.session(1).sendOk = false;
    await h.advance(1_500, { speech: true });
    h.rotator.setMuted(true);
    // Muted: no audio, and the session drops before it could take the waiting frames.
    await h.advance(6_000, { audio: false });
    h.session(1).drop();
    await h.advance(200, { audio: false });
    expect(h.rotator.snapshot).toMatchObject({ primary: 2, dormant: false });
    const s2 = h.session(2);
    h.expectNoAudioGap(1);
    expect(s2.endedAfter).toEqual([s2.frames.length]);
  });

  it("ends the stream of a replacement that connects while muted, after its replay", async () => {
    const h = new Harness();
    await h.start();
    await h.advance(2_000, { speech: true });
    h.session(1).interim("hello there");
    h.behaviors.push("pending");
    h.session(1).drop();
    await h.advance(200, { speech: true });
    // The user mutes while the replacement connects.
    h.rotator.setMuted(true);
    await h.advance(300, { audio: false });
    h.session(2).ready();
    await settle();
    await h.advance(100, { audio: false });
    const s2 = h.session(2);
    expect(s2.frames.length).toBeGreaterThan(10);
    expect(s2.endedAfter).toEqual([s2.frames.length]);
    // The sentence is finalized, and the session is closed at its rotation point instead of being replaced.
    s2.final("hello there");
    await h.advance(60_000, { audio: false });
    expect(h.rotator.snapshot).toMatchObject({ primary: null, dormant: true });
    expect(h.sessions).toHaveLength(2);
  });

  it("gives the new session the frames an old session held back by its send buffer has not got", async () => {
    const h = new Harness();
    await h.start();
    await h.advance(60_000 - 3_000);
    // A congested uplink: the old session takes nothing for 3 s around the rotation point.
    h.session(1).sendOk = false;
    const heldFrom = h.session(1).frames.at(-1)! + 1;
    await h.until(() => h.rotator.snapshot.stage === "overlap", 15_000);
    expect(h.session(2).frames[0]).toBe(heldFrom);
    await h.until(() => h.rotator.snapshot.stage === "draining", 5_000);
    h.session(1).sendOk = true;
    await h.until(() => h.rotator.snapshot.primary === 2, 5_000);
    await h.advance(1_000);
    h.expectNoAudioGap(1);
    h.expectContiguous(h.session(2));
    expect(h.statuses).toEqual([]);
  });

  it("backs off further each time the next session fails, without ending the running one", async () => {
    const h = new Harness();
    await runToPrepare(h);
    const fail = new LiveSessionError("transient", "1006");
    for (let i = 0; i < 100; i++) h.behaviors.push(fail);
    // 30 s from 10 s before the rotation point, past the running session's first minute.
    await h.advance(30_000, { speech: true });
    const attempts = h.sessions.length - 1;
    // 500 ms, 1 s, 2 s, 4 s, 8 s, 15 s ... between attempts, not a retry every 500 ms.
    expect(attempts).toBeGreaterThanOrEqual(5);
    expect(attempts).toBeLessThanOrEqual(7);
    expect(h.fatal).toBeNull();
    expect(h.rotator.snapshot.primary).toBe(1);
    expect(h.statuses).toEqual([]);
    // Once a rotation completes, the next one starts with the shortest back-off again.
    h.behaviors.length = 0;
    await h.until(() => h.rotator.snapshot.primary === h.sessions.length, 20_000);
    const promoted = h.sessions.length;
    h.behaviors.push(fail);
    await h.until(() => h.sessions.length === promoted + 1, 60_000);
    const failedAt = h.now;
    await h.until(() => h.sessions.length === promoted + 2, 2_000);
    expect(h.now - failedAt).toBeLessThanOrEqual(BACKOFF_MS[0] + 100);
  });

  it("aborts the first session's connection when stopped while it connects or backs off", async () => {
    const h = new Harness();
    h.behaviors.push("pending");
    let outcome: unknown = null;
    h.rotator.start().then(
      () => (outcome = "ready"),
      (e: unknown) => (outcome = e),
    );
    await settle();
    expect(h.sessions).toHaveLength(1);
    h.rotator.stop();
    await settle();
    expect(h.session(1).aborted).toBe(true);
    expect((outcome as Error | null)?.name).toBe("AbortError");

    const b = new Harness();
    b.behaviors.push(new LiveSessionError("transient", "1006"));
    let bOutcome: unknown = null;
    b.rotator.start().then(
      () => (bOutcome = "ready"),
      (e: unknown) => (bOutcome = e),
    );
    await settle();
    expect(b.statuses[0]).toMatchObject({ kind: "reconnecting", attempt: 1 });
    b.rotator.stop();
    await settle();
    expect((bOutcome as Error | null)?.name).toBe("AbortError");
    expect(b.sessions).toHaveLength(1);
  });

  it("stops cleanly", async () => {
    const h = new Harness();
    await runToPrepare(h);
    await h.advance(1_100);
    h.rotator.stop();
    expect(h.session(1).closed).toBe(true);
    expect(h.session(2).closed).toBe(true);
    h.session(1).final("late");
    await h.advance(1_000);
    expect(h.finals()).toEqual([]);
    expect(h.fatal).toBeNull();
  });
});

// ---------------------------------------------------------------------------------------------------------------
// WebSocket connection

class FakeSocket implements WebSocketLike {
  readyState = 0;
  bufferedAmount = 0;
  binaryType = "blob";
  onopen: ((ev: unknown) => void) | null = null;
  onmessage: ((ev: { data: unknown }) => void) | null = null;
  onerror: ((ev: unknown) => void) | null = null;
  onclose: ((ev: { code: number; reason: string; wasClean?: boolean }) => void) | null = null;
  readonly sent: string[] = [];
  closedWith: [number | undefined, string | undefined] | null = null;
  /** Answers the setup with setupComplete. */
  autoSetup = false;

  constructor(readonly url: string) {}

  send(data: string): void {
    this.sent.push(data);
    if (this.autoSetup && JSON.parse(data).setup) queueMicrotask(() => this.receive({ setupComplete: {} }));
  }
  close(code?: number, reason?: string): void {
    this.closedWith = [code, reason];
    this.readyState = 3;
  }
  open(): void {
    this.readyState = 1;
    this.onopen?.({});
  }
  receive(msg: unknown): void {
    this.onmessage?.({ data: new TextEncoder().encode(JSON.stringify(msg)).buffer });
  }
  serverClose(code: number, reason = ""): void {
    this.readyState = 3;
    this.onclose?.({ code, reason });
  }
  json(i: number): Record<string, unknown> {
    return JSON.parse(this.sent[i]) as Record<string, unknown>;
  }
}

function connection(over: Partial<ConstructorParameters<typeof GeminiLiveConnection>[0]> = {}) {
  const sockets: FakeSocket[] = [];
  const content: unknown[] = [];
  const closed: LiveSessionError[] = [];
  const goAways: (number | null)[] = [];
  const conn = new GeminiLiveConnection({
    url: "wss://example/ws?access_token=t",
    setup: { setup: { model: "models/m" } },
    createSocket: (url) => {
      const s = new FakeSocket(url);
      sockets.push(s);
      return s;
    },
    onContent: (m) => content.push(m),
    onGoAway: (ms) => goAways.push(ms),
    onClosed: (e) => closed.push(e),
    ...over,
  });
  return { conn, sockets, content, closed, goAways };
}

describe("GeminiLiveConnection", () => {
  afterEach(() => {
    vi.useRealTimers();
  });

  it("sends the setup, resolves on setupComplete and streams audio", async () => {
    const { conn, sockets, content, closed, goAways } = connection();
    const opening = conn.open(new AbortController().signal);
    const ws = sockets[0];
    expect(ws.binaryType).toBe("arraybuffer");
    ws.open();
    expect(ws.json(0)).toEqual({ setup: { model: "models/m" } });
    expect(conn.sendAudio(new Int16Array(4))).toBe(false);
    ws.receive({ setupComplete: {} });
    await opening;
    expect(conn.sendAudio(new Int16Array([1, -2]))).toBe(true);
    expect(ws.json(1)).toEqual({ realtimeInput: { audio: { data: "AQD+/w==", mimeType: "audio/pcm;rate=16000" } } });
    conn.endAudio();
    expect(ws.json(2)).toEqual({ realtimeInput: { audioStreamEnd: true } });
    ws.receive({ serverContent: { interimInputTranscription: { text: "hi" } } });
    ws.receive({ goAway: { timeLeft: "3s" } });
    expect(content).toEqual([{ kind: "content", interimInput: "hi", turnComplete: false }]);
    expect(goAways).toEqual([3000]);
    ws.serverClose(1011, "Internal error");
    expect(closed).toHaveLength(1);
    expect(closed[0]).toMatchObject({ kind: "transient", message: "1011: Internal error" });
  });

  it("decodes Blob frames in order", async () => {
    const { conn, sockets, content } = connection();
    const opening = conn.open(new AbortController().signal);
    const ws = sockets[0];
    ws.open();
    ws.onmessage?.({ data: new Blob([JSON.stringify({ setupComplete: {} })]) });
    ws.onmessage?.({ data: new Blob([JSON.stringify({ serverContent: { interimInputTranscription: { text: "a" } } })]) });
    ws.onmessage?.({ data: JSON.stringify({ serverContent: { interimInputTranscription: { text: "b" } } }) });
    await opening;
    await settle(30);
    expect(content.map((m) => (m as { interimInput: string }).interimInput)).toEqual(["a", "b"]);
  });

  it("rejects a refused setup with its classified reason", async () => {
    const { conn, sockets, closed } = connection();
    const opening = conn.open(new AbortController().signal);
    sockets[0].open();
    sockets[0].serverClose(1007, "Invalid JSON payload received. Unknown name \"customVocabulary\"");
    await expect(opening).rejects.toMatchObject({ kind: "config", info: { code: 1007, beforeSetup: true } });
    expect(closed).toEqual([]);
  });

  it("explains a failed handshake", async () => {
    const { conn, sockets } = connection();
    const opening = conn.open(new AbortController().signal);
    sockets[0].serverClose(1006, "");
    const err = await opening.catch((e: unknown) => e);
    expect(err).toBeInstanceOf(LiveSessionError);
    expect((err as LiveSessionError).kind).toBe("transient");
    expect((err as LiveSessionError).message).toBe(i18n.t("stt.gemini_detail_handshake", { code: 1006 }));
  });

  it("aborts and times out the setup", async () => {
    const a = connection();
    const ctrl = new AbortController();
    const opening = a.conn.open(ctrl.signal);
    ctrl.abort();
    await expect(opening).rejects.toMatchObject({ name: "AbortError" });
    expect(a.sockets[0].closedWith).toEqual([1000, "cancelled"]);

    vi.useFakeTimers();
    const b = connection({ setupTimeoutMs: 1_000 });
    const slow = b.conn.open(new AbortController().signal);
    b.sockets[0].open();
    const result = slow.catch((e: unknown) => e);
    await vi.advanceTimersByTimeAsync(1_100);
    expect(await result).toMatchObject({ kind: "transient", info: { beforeSetup: true } });
  });

  it("holds audio back while the send buffer is full and gives up a stalled connection", async () => {
    let now = 0;
    const { conn, sockets, closed } = connection({ maxBufferedBytes: 100, stallTimeoutMs: 1_000, now: () => now });
    const opening = conn.open(new AbortController().signal);
    sockets[0].open();
    sockets[0].receive({ setupComplete: {} });
    await opening;
    const ws = sockets[0];
    // Back-pressure: the frame waits, nothing is sent and the session goes on.
    ws.bufferedAmount = 500;
    expect(conn.sendAudio(new Int16Array(2))).toBe(false);
    expect(ws.sent).toHaveLength(1);
    // The buffer drains (slowly): no stall however long that takes.
    for (const amount of [400, 300, 200]) {
      now += 900;
      ws.bufferedAmount = amount;
      expect(conn.sendAudio(new Int16Array(2))).toBe(false);
    }
    expect(closed).toEqual([]);
    ws.bufferedAmount = 50;
    expect(conn.sendAudio(new Int16Array(2))).toBe(true);
    expect(ws.sent).toHaveLength(2);
    // A buffer that stays full without draining: the connection is given up once, with a close report.
    ws.bufferedAmount = 300;
    expect(conn.sendAudio(new Int16Array(2))).toBe(false);
    now += 600;
    expect(conn.sendAudio(new Int16Array(2))).toBe(false);
    expect(closed).toEqual([]);
    now += 500;
    expect(conn.sendAudio(new Int16Array(2))).toBe(false);
    expect(closed).toHaveLength(1);
    expect(closed[0].kind).toBe("transient");
    expect(ws.closedWith?.[0]).toBe(4000);
    expect(conn.sendAudio(new Int16Array(2))).toBe(false);
    ws.serverClose(1000);
    expect(closed).toHaveLength(1);

    const second = connection();
    const op2 = second.conn.open(new AbortController().signal);
    second.sockets[0].open();
    second.sockets[0].receive({ setupComplete: {} });
    await op2;
    second.conn.close();
    second.sockets[0].serverClose(1000);
    expect(second.closed).toEqual([]);
  });
});

// ---------------------------------------------------------------------------------------------------------------
// Ephemeral token

function response(status: number, body: string, headers: Record<string, string> = {}): LlmHttpResponse {
  return {
    status,
    headers,
    body: (async function* () {
      yield body;
    })(),
    cancel: () => {},
  };
}

describe("mintGeminiLiveToken", () => {
  it("sends the key in a header only and returns the token name", async () => {
    const requests: LlmHttpRequest[] = [];
    const transport: LlmHttpTransport = async (req) => {
      requests.push(req);
      return response(200, JSON.stringify({ name: "auth_tokens/xyz" }));
    };
    const token = await mintGeminiLiveToken({ apiKey: KEY, apiVersion: "v1beta", signal: new AbortController().signal, transports: [transport] });
    expect(token).toBe("auth_tokens/xyz");
    expect(requests).toHaveLength(1);
    expect(requests[0].url).toBe("https://generativelanguage.googleapis.com/v1beta/auth_tokens");
    expect(requests[0].url).not.toContain(KEY);
    expect(requests[0].method).toBe("POST");
    expect(requests[0].headers["x-goog-api-key"]).toBe(KEY);
    expect(JSON.parse(requests[0].body ?? "")).toEqual({ uses: 1 });
  });

  it("falls back to the next transport only when the first cannot reach the server", async () => {
    const unreachable: LlmHttpTransport = async () => {
      throw new LlmTransportError("Failed to fetch", true);
    };
    const ok: LlmHttpTransport = async () => response(200, JSON.stringify({ name: "auth_tokens/rust" }));
    expect(await mintGeminiLiveToken({ apiKey: KEY, apiVersion: "v1beta", signal: new AbortController().signal, transports: [unreachable, ok] })).toBe(
      "auth_tokens/rust",
    );
    await expect(
      mintGeminiLiveToken({ apiKey: KEY, apiVersion: "v1beta", signal: new AbortController().signal, transports: [unreachable] }),
    ).rejects.toMatchObject({ kind: "transient" });
    const refused: LlmHttpTransport = async () => response(400, JSON.stringify({ error: { message: `API key not valid: ${KEY}` } }));
    const err = await mintGeminiLiveToken({ apiKey: KEY, apiVersion: "v1beta", signal: new AbortController().signal, transports: [refused, ok] }).catch(
      (e: unknown) => e as LiveSessionError,
    );
    expect(err).toMatchObject({ kind: "auth" });
    expect((err as LiveSessionError).message).not.toContain(KEY);
  });

  it("tries the other API version when the token endpoint is not found", async () => {
    const urls: string[] = [];
    const movedTo = (version: string): LlmHttpTransport => async (req) => {
      urls.push(req.url);
      return req.url.includes(`/${version}/`)
        ? response(200, JSON.stringify({ name: `auth_tokens/${version}` }))
        : response(404, JSON.stringify({ error: { message: "Not Found" } }));
    };
    const signal = new AbortController().signal;
    expect(await mintGeminiLiveTokenAnyVersion({ apiKey: KEY, apiVersion: "v1beta", signal, transports: [movedTo("v1beta")] })).toEqual({
      token: "auth_tokens/v1beta",
      apiVersion: "v1beta",
    });
    expect(urls).toHaveLength(1);
    expect(await mintGeminiLiveTokenAnyVersion({ apiKey: KEY, apiVersion: "v1beta", signal, transports: [movedTo("v1alpha")] })).toEqual({
      token: "auth_tokens/v1alpha",
      apiVersion: "v1alpha",
    });
    expect(urls.slice(1)).toEqual([authTokenUrl("v1beta"), authTokenUrl("v1alpha")]);
    // Gone everywhere: the first refusal is reported; other refusals are not retried on another version.
    await expect(
      mintGeminiLiveTokenAnyVersion({ apiKey: KEY, apiVersion: "v1alpha", signal, transports: [movedTo("v2")] }),
    ).rejects.toMatchObject({ kind: "config", info: { code: 404 } });
    const refused: LlmHttpTransport = async (req) => {
      urls.push(req.url);
      return response(401, JSON.stringify({ error: { message: "API key not valid" } }));
    };
    const before = urls.length;
    await expect(mintGeminiLiveTokenAnyVersion({ apiKey: KEY, apiVersion: "v1beta", signal, transports: [refused] })).rejects.toMatchObject({
      kind: "auth",
    });
    expect(urls.length - before).toBe(1);
  });

  it("rejects a reply without a token and honours the abort", async () => {
    const empty: LlmHttpTransport = async () => response(200, "{}");
    await expect(
      mintGeminiLiveToken({ apiKey: KEY, apiVersion: "v1beta", signal: new AbortController().signal, transports: [empty] }),
    ).rejects.toMatchObject({ kind: "config" });
    const ctrl = new AbortController();
    ctrl.abort();
    await expect(mintGeminiLiveToken({ apiKey: KEY, apiVersion: "v1beta", signal: ctrl.signal, transports: [empty] })).rejects.toBeDefined();
  });
});

// ---------------------------------------------------------------------------------------------------------------
// Service driver

function params(geminiLive: Partial<STT_State["geminiLive"]> = {}): STT_State {
  const base = Service_STT_Schema.parse({});
  return { ...base, backend: "geminiLive" as STT_State["backend"], geminiLive: { ...base.geminiLive, ...geminiLive } };
}

describe("STT_GeminiLiveService", () => {
  const originalDeps = { ...geminiLiveDeps };
  let sockets: FakeSocket[];
  let publishText: ReturnType<typeof vi.fn>;
  let translationStatus: ServiceNetworkState;

  beforeEach(() => {
    sockets = [];
    publishText = vi.fn();
    translationStatus = ServiceNetworkState.disconnected;
    vi.mocked(invoke).mockClear();
    geminiLiveDeps.createSocket = (url) => {
      const s = new FakeSocket(url);
      s.autoSetup = true;
      sockets.push(s);
      queueMicrotask(() => s.open());
      return s;
    };
    let n = 0;
    geminiLiveDeps.transports = () => [async () => response(200, JSON.stringify({ name: `auth_tokens/t${++n}` }))];
    (globalThis as unknown as { window: unknown }).window = {
      ApiServer: {
        state: { services: { translation: { data: { llm: { geminiKey: KEY } } } } },
        translation: { serviceState: { get status() { return translationStatus; } } },
      },
      ApiShared: { pubsub: { publishText } },
    };
  });

  afterEach(() => {
    Object.assign(geminiLiveDeps, originalDeps);
    delete (globalThis as unknown as { window?: unknown }).window;
  });

  function receiver(muted = () => false) {
    const calls: string[] = [];
    const metas: (SttTextMeta | undefined)[] = [];
    const r: ISTTReceiver = {
      onStart: () => calls.push("start"),
      onStop: (e) => calls.push(`stop:${e ?? ""}`),
      onInterim: (v, m) => {
        calls.push(`interim:${v}`);
        metas.push(m);
      },
      onFinal: (v, m) => {
        calls.push(`final:${v}`);
        metas.push(m);
      },
      onStatus: () => {},
      isMuted: muted,
    };
    return { r, calls, metas };
  }

  const chunk = (value = 0.2, samples = 1600) => tauri.audioHandler?.({ payload: { samples: new Array(samples).fill(value), sample_rate: 16000, channels: 1 } });

  it("captures the microphone, streams it and publishes captions", async () => {
    const { r, calls, metas } = receiver();
    const svc = new STT_GeminiLiveService(r);
    svc.startAbortSignal = new AbortController().signal;
    await svc.start(params({ device: "USB Mic", language: "en-US" }));
    expect(calls).toEqual(["start"]);
    expect(vi.mocked(invoke)).toHaveBeenCalledWith("plugin:audio|start_audio_capture", { deviceName: "USB Mic", sampleRate: 16000 });
    const ws = sockets[0];
    expect(ws.url).toContain("access_token=auth_tokens/t1");
    expect(ws.url).not.toContain(KEY);
    expect(ws.json(0)).toMatchObject({ setup: { inputAudioTranscription: { languageCodes: ["en-US"] } } });

    chunk(0.2, 1000);
    chunk(0.2, 1000);
    const audio = ws.sent.slice(1).map((s) => JSON.parse(s) as { realtimeInput?: { audio?: { mimeType: string } } });
    expect(audio).toHaveLength(1);
    expect(audio[0].realtimeInput?.audio?.mimeType).toBe("audio/pcm;rate=16000");

    ws.receive({ serverContent: { interimInputTranscription: { text: "hello" } } });
    ws.receive({ serverContent: { inputTranscription: { text: "Hello there." } } });
    expect(calls.slice(1)).toEqual(["interim:hello", "final:Hello there."]);
    expect(metas[0]?.segmentId).toMatch(new RegExp(`^gemini-${PROCESS_NONCE}:\\d+:1$`));
    // Transcribe captions are translated by the Translation service as usual.
    expect(isLiveTranslatedStt(metas[0])).toBe(false);
    expect(metas.map((m) => m?.revision)).toEqual([1, 2]);
    expect(metas.map((m) => m?.lang)).toEqual(["en", "en"]);

    await svc.stop();
    expect(vi.mocked(invoke)).toHaveBeenCalledWith("plugin:audio|stop_audio_capture");
    expect(ws.closedWith?.[0]).toBe(1000);
    expect(calls.at(-1)).toBe("stop:");
    expect(calls.filter((c) => c.startsWith("stop"))).toHaveLength(1);
  });

  it("follows the STT mute with audioStreamEnd and withholds audio", async () => {
    let muted = false;
    const { r } = receiver(() => muted);
    const svc = new STT_GeminiLiveService(r);
    await svc.start(params());
    const ws = sockets[0];
    muted = true;
    chunk(0.2, 1600);
    expect(ws.sent.slice(1).map((s) => JSON.parse(s))).toEqual([{ realtimeInput: { audioStreamEnd: true } }]);
    muted = false;
    chunk(0.2, 1600);
    expect(JSON.parse(ws.sent.at(-1)!)).toHaveProperty("realtimeInput.audio");
    await svc.stop();
  });

  it("falls back to the setup without vocabulary and warns", async () => {
    const statuses: string[] = [];
    const { r, calls } = receiver();
    r.onStatus = (t) => statuses.push(t);
    const onStart = r.onStart;
    r.onStart = () => {
      onStart();
      statuses.push("");
    };
    let first = true;
    geminiLiveDeps.createSocket = (url) => {
      const s = new FakeSocket(url);
      sockets.push(s);
      queueMicrotask(() => {
        s.open();
        if (first) {
          first = false;
          s.serverClose(1007, "Invalid JSON payload received. Unknown name \"customVocabulary\"");
        }
        else s.receive({ setupComplete: {} });
      });
      return s;
    };
    const svc = new STT_GeminiLiveService(r);
    await svc.start(params({ vocabulary: "Sigil" }));
    expect(calls).toEqual(["start"]);
    expect(sockets).toHaveLength(2);
    expect(sockets[0].json(0)).toMatchObject({ setup: { inputAudioTranscription: { customVocabulary: ["Sigil"] } } });
    expect(sockets[1].json(0)).toEqual({
      setup: { model: "models/gemini-3.5-transcribe-live", generationConfig: { responseModalities: ["TEXT"] }, inputAudioTranscription: { languageCodes: [] } },
    });
    expect(statuses.at(-1)).toContain("customVocabulary");
    await svc.stop();
  });

  it("stops with a clear message when there is no key or the key is refused", async () => {
    (globalThis as unknown as { window: { ApiServer: { state: { services: { translation: { data: { llm: { geminiKey: string } } } } } } } }).window.ApiServer.state.services.translation.data.llm.geminiKey = "";
    const a = receiver();
    await new STT_GeminiLiveService(a.r).start(params());
    expect(a.calls).toEqual([`stop:${i18n.t("stt.gemini_error_no_key")}`]);
    expect(sockets).toHaveLength(0);

    geminiLiveDeps.transports = () => [async () => response(401, JSON.stringify({ error: { message: "API key not valid" } }))];
    const b = receiver();
    await new STT_GeminiLiveService(b.r).start(params({ apiKey: "bad-key-123456" }));
    expect(b.calls).toHaveLength(1);
    expect(b.calls[0]).toContain(i18n.t("stt.gemini_error_key", { detail: "HTTP 401: API key not valid" }));
    expect(vi.mocked(invoke)).not.toHaveBeenCalledWith("plugin:audio|start_audio_capture", expect.anything());
  });

  it("publishes captions and translations in Live Translate mode", async () => {
    translationStatus = ServiceNetworkState.connected;
    const statuses: string[] = [];
    const { r, calls, metas } = receiver();
    r.onStatus = (t) => statuses.push(t);
    // Service_STT clears the engine status when the session starts.
    const onStart = r.onStart;
    r.onStart = () => {
      onStart();
      statuses.push("");
    };
    const svc = new STT_GeminiLiveService(r);
    await svc.start(params({ mode: "translate", translateTo: "en" }));
    const ws = sockets[0];
    expect(ws.json(0)).toMatchObject({ setup: { generationConfig: { translationConfig: { targetLanguageCode: "en", echoTargetLanguage: true } } } });
    // The Translation service also runs, but it skips Live Translate's captions (P6-I0 wired the skip), so the status
    // line no longer warns that they are translated twice.
    expect(statuses).not.toContain(i18n.t("stt.gemini_status_translation_running"));
    ws.receive({ serverContent: { inputTranscription: { text: "Hola", languageCode: "es" } } });
    ws.receive({ serverContent: { outputTranscription: { text: "Hello" } } });
    ws.receive({ serverContent: { turnComplete: true } });
    expect(calls.slice(1)).toEqual(["interim:Hola", "final:Hola"]);
    // These captions already have their translation: the Translation service recognizes them.
    expect(metas[0]?.segmentId).toMatch(new RegExp(`^${LIVE_TRANSLATE_STT_STREAM}-${PROCESS_NONCE}:\\d+:1$`));
    expect(metas.every((m) => isLiveTranslatedStt(m))).toBe(true);
    expect(metas.map((m) => m?.lang)).toEqual(["es", "es"]);
    const events = publishText.mock.calls.map(([topic, ev]) => [topic, (ev as TranslationOutputEvent).type, (ev as TranslationOutputEvent).value]);
    expect(events).toEqual([
      [TextEventSource.translation, TextEventType.interim, "Hello"],
      [TextEventSource.translation, TextEventType.final, "Hello"],
    ]);
    // A line still being joined is published on commit (before the STT service closes the sentence); nothing after stop.
    ws.receive({ serverContent: { outputTranscription: { text: "Bye" } } });
    svc.commitPending();
    expect(publishText.mock.calls.at(-1)?.[1]).toMatchObject({ type: TextEventType.final, value: "Bye" });
    await svc.stop();
    const count = publishText.mock.calls.length;
    ws.receive({ serverContent: { outputTranscription: { text: "late" } } });
    expect(publishText.mock.calls.length).toBe(count);
  });
});
