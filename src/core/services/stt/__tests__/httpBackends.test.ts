import { beforeAll, describe, expect, it, vi } from "vitest";

vi.mock("@tauri-apps/api/core", () => ({ invoke: vi.fn(), isTauri: () => true }));
vi.mock("@tauri-apps/api/event", () => ({ listen: vi.fn() }));
vi.mock("@/utils/devLog", () => ({ devLog: () => {} }));
vi.mock("@/utils/rustAudioInput", () => ({ resolveRustInputDeviceForInvoke: () => null }));

import {
  baseLanguageTag,
  buildUtterancePipelineConfig,
  droppedStatusLine,
  HTTP_LATENCY_BUDGET_MS,
  HTTP_MAX_BACKLOG,
  normalizeEndpointUrl,
  rateLimitWaitMinutes,
  rateLimitWaitSeconds,
  speechFatalMessage,
  speechFinalMeta,
  speechStatusLine,
  type PipelineEngineSetup,
} from "../rustCapturePolling";
import { OpenAiAudioPreset, Service_STT_Schema, type STT_State } from "../schema";
import { MOONSHINE_DEFAULT_ENDPOINT, moonshineSetup } from "../services/moonshine";
import { openAiAudioSetup } from "../services/openaiAudio";
import { TRANSIENT_WARNING_MS } from "../statusChannel";
import { resolveOpenAiAudioConfig } from "../sttOptions";
import { initSttTestI18n, translated } from "./sttTestI18n";

beforeAll(async () => {
  await initSttTestI18n();
});

const openai = (o: Partial<STT_State["openai_audio"]>) =>
  openAiAudioSetup(resolveOpenAiAudioConfig({ ...Service_STT_Schema.parse({}).openai_audio, ...o }));

function setupOf(result: PipelineEngineSetup | string): PipelineEngineSetup {
  if (typeof result === "string") throw new Error(`expected a setup, got "${result}"`);
  return result;
}

describe("openai_audio → Rust HTTP engine", () => {
  it("Groq: the preset, the model and the key (only in options.apiKey); no URL is sent", () => {
    const s = setupOf(openai({ preset: OpenAiAudioPreset.groq, groqApiKey: " gsk_abc ", groqModel: "whisper-large-v3", language: "ja-JP", prompt: " VRChat " }));
    expect(s.engine).toEqual({ engine: "http", model: "", options: { preset: "groq", model: "whisper-large-v3", apiKey: "gsk_abc" } });
    expect(s.language).toBe("ja");
    expect(s.prompt).toBe("VRChat");
    // The custom server's key is never sent to Groq.
    const other = setupOf(openai({ preset: OpenAiAudioPreset.groq, groqApiKey: "gsk_1", apiKey: "local-key" }));
    expect(other.engine.options?.apiKey).toBe("gsk_1");
    expect(JSON.stringify(other)).not.toContain("local-key");
  });

  it("Groq without a key stops with the key message", () => {
    const r = openai({ preset: OpenAiAudioPreset.groq, groqApiKey: "  " });
    expect(typeof r).toBe("string");
    expect(r).toContain("console.groq.com");
  });

  it("custom servers: URL normalized like Rust, model required, key optional", () => {
    const s = setupOf(openai({ baseUrl: "localhost:8000/v1/", model: "Systran/faster-whisper-small", apiKey: "" }));
    expect(s.engine.options).toEqual({ preset: "openai", baseUrl: "http://localhost:8000/v1", model: "Systran/faster-whisper-small" });
    expect(s.language).toBe("auto");
    const lan = setupOf(openai({ baseUrl: "http://192.168.1.20:8000/v1", model: "m", apiKey: "k" }));
    expect(lan.engine.options).toMatchObject({ baseUrl: "http://192.168.1.20:8000/v1", apiKey: "k" });

    for (const [o, fragment] of [
      [{ baseUrl: "", model: "m" }, "base URL"],
      [{ baseUrl: "ftp://files.example", model: "m" }, "ftp://files.example"],
      [{ baseUrl: "http://127.0.0.1:8000/v1", model: " " }, "model id"],
    ] as const) {
      const r = openai(o);
      expect(typeof r, JSON.stringify(o)).toBe("string");
      expect(r).toContain(fragment);
    }
  });

  it("Groq's API typed as a custom server needs a key and is still recognized as Groq", () => {
    expect(openai({ baseUrl: "https://api.groq.com/openai/v1", model: "whisper-large-v3-turbo" })).toContain("console.groq.com");
    const s = setupOf(openai({ baseUrl: "https://api.groq.com/openai/v1", model: "whisper-large-v3-turbo", apiKey: "gsk" }));
    expect(s.engine.options).toMatchObject({ preset: "openai", baseUrl: "https://api.groq.com/openai/v1", apiKey: "gsk" });
  });
});

describe("moonshine → Rust HTTP engine (Moonshine preset)", () => {
  it("uses the endpoint (default when blank) and the language", () => {
    expect(setupOf(moonshineSetup({ endpoint: "", language: "en" }))).toEqual({
      engine: { engine: "http", model: "", options: { preset: "moonshine", baseUrl: MOONSHINE_DEFAULT_ENDPOINT } },
      language: "en",
    });
    expect(setupOf(moonshineSetup({ endpoint: "10.0.0.5:9000/", language: " " })).engine.options).toEqual({
      preset: "moonshine",
      baseUrl: "http://10.0.0.5:9000",
    });
    expect(setupOf(moonshineSetup({ endpoint: "http://x", language: "" })).language).toBe("auto");
    expect(moonshineSetup({ endpoint: "mailto:someone" })).toContain("mailto:someone");
  });
});

describe("pipeline config and helpers", () => {
  it("builds a finals-only config with the explicit engine", () => {
    const setup = setupOf(openai({ baseUrl: "http://127.0.0.1:8000/v1", model: "m" }));
    const c = buildUtterancePipelineConfig({ ...setup, hotwords: ["Sigil"] }, { threshold: "0.6" }, null);
    expect(c).toMatchObject({
      pipelineId: "mic",
      source: "",
      language: "auto",
      streaming: null,
      final: setup.engine,
      hotwords: ["Sigil"],
      prompt: "",
      hallucinationFilter: true,
      retainPcm: false,
      smartTurn: false,
    });
    expect(c.vad).toMatchObject({ threshold: 0.6 });
  });

  it("HTTP sessions wait up to Rust's longest latency budget, so speech said during a request limit wait is merged", () => {
    // Rust's default (20 s) would drop everything said during a 30 s Groq 429 wait, a slow LAN server's 22 s batch
    // or a Wi-Fi drop; the retired TS pipeline merged it. 120 s is Rust's MAX_LATENCY_BUDGET_MS (clamped above).
    expect(HTTP_LATENCY_BUDGET_MS).toBe(120_000);
    for (const setup of [
      setupOf(openai({ preset: OpenAiAudioPreset.groq, groqApiKey: "gsk" })),
      setupOf(openai({ baseUrl: "http://192.168.1.20:8000/v1", model: "m" })),
      setupOf(moonshineSetup({ endpoint: "", language: "en" })),
    ]) {
      const c = buildUtterancePipelineConfig(setup, undefined, "Microphone (USB)");
      expect(c.latencyBudgetMs, JSON.stringify(setup.engine.options?.preset)).toBe(HTTP_LATENCY_BUDGET_MS);
      expect(c.maxBacklog).toBe(HTTP_MAX_BACKLOG);
      expect(c.source).toBe("Microphone (USB)");
    }
  });

  it("normalizes endpoints like Rust's normalize_base_url", () => {
    expect(normalizeEndpointUrl("localhost:8880")).toBe("http://localhost:8880");
    expect(normalizeEndpointUrl(" https://host/tts/// ")).toBe("https://host/tts");
    expect(normalizeEndpointUrl("http://[::1]:8000/v1")).toBe("http://[::1]:8000/v1");
    expect(normalizeEndpointUrl("")).toBeNull();
    expect(normalizeEndpointUrl("ftp://x")).toBeNull();
    expect(normalizeEndpointUrl("http://")).toBeNull();
    expect(normalizeEndpointUrl("http://exa mple")).toBeNull();
  });

  it("base language tags", () => {
    expect(baseLanguageTag("en-US")).toBe("en");
    expect(baseLanguageTag("zh_TW")).toBe("zh");
    expect(baseLanguageTag("yue")).toBe("yue");
    expect(baseLanguageTag("auto")).toBeNull();
    expect(baseLanguageTag("")).toBeNull();
    expect(baseLanguageTag(null)).toBeNull();
  });

  it("final metadata", () => {
    expect(speechFinalMeta({ pipelineId: "mic", epoch: 4, segmentId: 12, revision: 1, lang: "JA", turnEnd: null })).toEqual({
      segmentId: "mic:4:12",
      revision: 1,
      lang: "ja",
    });
    expect(speechFinalMeta({ pipelineId: "mic", epoch: 4, segmentId: 13, revision: -1, lang: null, turnEnd: false })).toEqual({
      segmentId: "mic:4:13",
      turnEnd: false,
    });
  });

  it("status lines: translated, warnings transient, start lifecycle ignored, listening clears", () => {
    const line = (code: string, detail = "", level: "info" | "warning" | "error" = "warning") =>
      speechStatusLine({ code, detail, level }, "Groq");
    expect(line("listening", "", "info")).toBeNull();
    for (const code of ["starting", "loading_model", "loading_runtime", "downloading", "stopped"]) expect(line(code)).toBeUndefined();
    const wait = line("rate_limited", "waiting 7 s for the request limit")!;
    expect(wait[0]).toContain("7 s");
    expect(wait[1]).toMatchObject({ level: "warning" });
    expect(line("rate_limited", "rate limited by the server; retrying in 20 s")![0]).toContain("20 s");
    // A rate_limited warning that is not a wait follows a lost utterance: it says so, never "sent together".
    const skipped = "Some speech was skipped because the request limit was reached.";
    for (const detail of [
      "",
      "an utterance was dropped: it would have waited past the latency budget (rate_limited: the request limit allows the next request in 431.9 s)",
      "rate_limited: the request limit is used up for now (the next request would wait 11 min)",
      "rate_limited: Groq transcription: HTTP 429: Rate limit reached for model whisper-large-v3-turbo on audio seconds per hour (ASPH). Please try again in 7m12.5s.",
    ]) {
      const l = line("rate_limited", detail)!;
      expect(l[0], detail).toBe(skipped);
      expect(l[1]).toEqual({ level: "warning", expireMs: TRANSIENT_WARNING_MS });
    }
    for (const code of ["backlog_dropped", "capture_gap", "capture_error"]) {
      const l = line(code, "detail")!;
      expect(translated(l[0]), code).toBe(true);
      expect(l[1]).toEqual({ level: "warning", expireMs: TRANSIENT_WARNING_MS });
    }
    expect(line("engine_error", "Groq transcription: timed out; retrying (2/4)")![0]).toBe("Groq: Groq transcription: timed out; retrying (2/4)");
    expect(line("engine_error", "")).toBeUndefined();
    expect(line("something_new", "a note", "info")).toEqual(["Groq: a note"]);
  });

  it("reads only Rust's own wait formats as a wait", () => {
    expect(rateLimitWaitSeconds("waiting 12 s for the request limit")).toBe(12);
    expect(rateLimitWaitSeconds("rate limited by the server; retrying in 45 s")).toBe(45);
    expect(rateLimitWaitSeconds("Please try again in 7m12.5s")).toBeNull(); // not "5 s"
    expect(rateLimitWaitSeconds("the request limit allows the next request in 3.2 s")).toBeNull();
    expect(rateLimitWaitSeconds("the next request would wait 3 min")).toBeNull();
  });

  it("fatal rate-limit waits in minutes, from Rust's text or the server's", () => {
    expect(rateLimitWaitMinutes("the next request would wait 42 min")).toBe(42);
    expect(rateLimitWaitMinutes("Please try again in 7m12.5s.")).toBe(8);
    expect(rateLimitWaitMinutes("Please try again in 7m.")).toBe(7);
    expect(rateLimitWaitMinutes("try again in 1h2m3s")).toBe(63);
    expect(rateLimitWaitMinutes("try again in 45.2s")).toBeNull();
    expect(rateLimitWaitMinutes("retrying in 20 s")).toBeNull();
    expect(rateLimitWaitMinutes("in 5ms")).toBeNull();
    expect(
      speechFatalMessage({ code: "rate_limited", detail: "5 utterances failed in a row; last error: rate_limited: HTTP 429: Please try again in 7m12.5s." }, "Groq"),
    ).toContain("8 min");
  });

  it("fatal messages", () => {
    expect(speechFatalMessage({ code: "capture_error", detail: "audio device error: gone" }, "Groq")).toBe(
      "Microphone capture failed: audio device error: gone",
    );
    expect(speechFatalMessage({ code: "rate_limited", detail: "the next request would wait 61 min" }, "Groq")).toContain("61 min");
    expect(speechFatalMessage({ code: "engine_error", detail: "HTTP 401" }, "Groq")).toBe("Groq: HTTP 401");
    expect(speechFatalMessage({ code: "engine_error", detail: "" }, "Groq")).toBe("Groq: engine_error");
    const long = speechFatalMessage({ code: "engine_error", detail: "x".repeat(1_000) }, "Groq");
    expect(long.length).toBeLessThan(260);
  });

  it("dropped utterances: only real losses are reported", () => {
    for (const reason of ["backlog", "latency_budget", "rate_limited", "engine_error"]) {
      const l = droppedStatusLine(reason)!;
      expect(translated(l[0]), reason).toBe(true);
      expect(l[1]).toEqual({ level: "warning", expireMs: TRANSIENT_WARNING_MS });
    }
    for (const reason of ["merged", "no_speech", "muted", "unknown"]) expect(droppedStatusLine(reason)).toBeNull();
  });
});
