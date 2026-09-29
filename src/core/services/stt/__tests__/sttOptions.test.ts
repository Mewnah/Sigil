import { describe, expect, it } from "vitest";
import { OpenAiAudioPreset, Service_STT_Schema } from "../schema";
import {
  azureSegmentationSilenceMs,
  buildVadConfig,
  buildWhisperDecodeOptions,
  GROQ_BASE_URL,
  isGroqUrl,
  MAX_HOTWORDS,
  normalizeTranscriptionLanguage,
  parseHotwords,
  resolveOpenAiAudioConfig,
  stripWhisperNoiseMarkers,
  transcriptionsUrl,
} from "../sttOptions";

const defaults = Service_STT_Schema.parse({});

describe("schema defaults (all zSafe)", () => {
  it("has VAD, Whisper hardening and Groq preset defaults", () => {
    expect(defaults.vad).toEqual({ threshold: "0.5", minSilenceMs: "500", preRollMs: "400", minSpeechMs: "250", maxUtteranceMs: "18000" });
    expect(defaults.whisper).toMatchObject({ threads: "0", noSpeechThreshold: "0.6", hallucinationFilter: true, maxBacklog: "3", hotwords: "", initialPrompt: "" });
    expect(defaults.openai_audio).toMatchObject({ preset: "custom", groqModel: "whisper-large-v3-turbo", groqApiKey: "" });
    expect(defaults.deepgram.model).toBe("nova-3");
    expect(defaults.azure.segmentationSilenceMs).toBe("500");
  });

  it("recovers from garbage values instead of failing the whole slice", () => {
    const parsed = Service_STT_Schema.parse({
      vad: { threshold: "abc", minSilenceMs: null },
      whisper: { hallucinationFilter: "yes", maxBacklog: {} },
      openai_audio: { preset: "not-a-preset" },
    });
    expect(parsed.vad.threshold).toBe("0.5");
    expect(parsed.openai_audio.preset).toBe(OpenAiAudioPreset.custom);
    expect(parsed.whisper.maxBacklog).toBe("3");
  });
});

describe("buildVadConfig", () => {
  it("maps the defaults to the Rust contract (snake_case numbers)", () => {
    expect(buildVadConfig(defaults.vad)).toEqual({
      threshold: 0.5,
      neg_threshold: 0.35,
      min_silence_ms: 500,
      pre_roll_ms: 400,
      min_speech_ms: 250,
      max_utterance_ms: 18000,
    });
  });

  it("clamps out-of-range values and drops unusable ones", () => {
    expect(buildVadConfig({ threshold: "2", minSilenceMs: "10", preRollMs: "-5", minSpeechMs: "", maxUtteranceMs: "999999" })).toEqual({
      threshold: 0.95,
      neg_threshold: 0.8,
      min_silence_ms: 100,
      pre_roll_ms: 0,
      max_utterance_ms: 60000,
    });
    expect(buildVadConfig({ threshold: "0.1" })).toEqual({ threshold: 0.1, neg_threshold: 0.01 });
    expect(buildVadConfig(undefined)).toEqual({});
  });
});

describe("buildWhisperDecodeOptions", () => {
  it("leaves threads automatic by default and keeps the filter on", () => {
    expect(buildWhisperDecodeOptions(defaults.whisper)).toEqual({
      no_speech_threshold: 0.6,
      hallucination_filter: true,
      max_backlog: 3,
    });
  });

  it("passes prompt, hotwords and limits", () => {
    expect(
      buildWhisperDecodeOptions({
        threads: "6",
        initialPrompt: "  VRChat with friends. ",
        hotwords: "Sigil, VRChat\nMewnah;sigil",
        noSpeechThreshold: "5",
        hallucinationFilter: false,
        maxBacklog: "0",
      }),
    ).toEqual({
      n_threads: 6,
      initial_prompt: "VRChat with friends.",
      hotwords: ["Sigil", "VRChat", "Mewnah"],
      no_speech_threshold: 1,
      hallucination_filter: false,
      max_backlog: 1,
    });
  });

  it("parses hotword lists (CJK separators, dedupe, cap)", () => {
    expect(parseHotwords("東京、大阪，京都 ; ")).toEqual(["東京", "大阪", "京都"]);
    expect(parseHotwords(Array.from({ length: 80 }, (_, i) => `w${i}`).join(","))).toHaveLength(MAX_HOTWORDS);
    expect(parseHotwords(undefined)).toEqual([]);
  });
});

describe("OpenAI-compatible / Groq config", () => {
  it("the Groq preset uses Groq's endpoint, its own key and whisper-large-v3-turbo, and is rate limited", () => {
    const r = resolveOpenAiAudioConfig({
      ...defaults.openai_audio,
      preset: OpenAiAudioPreset.groq,
      apiKey: "local-key",
      groqApiKey: " gsk_test ",
      language: "en-US",
    });
    expect(r).toMatchObject({ baseUrl: GROQ_BASE_URL, apiKey: "gsk_test", model: "whisper-large-v3-turbo", language: "en", rateLimited: true });
  });

  it("a custom URL pointing at Groq is still rate limited", () => {
    const r = resolveOpenAiAudioConfig({ ...defaults.openai_audio, baseUrl: "https://api.groq.com/openai/v1/", model: "whisper-large-v3" });
    expect(r.rateLimited).toBe(true);
    expect(r.baseUrl).toBe("https://api.groq.com/openai/v1");
  });

  it("a local server is not rate limited and keeps its own key", () => {
    const r = resolveOpenAiAudioConfig({ ...defaults.openai_audio, apiKey: "k", model: "Systran/faster-whisper-small" });
    expect(r).toMatchObject({ preset: "custom", baseUrl: "http://127.0.0.1:8000/v1", apiKey: "k", rateLimited: false, language: "" });
    expect(transcriptionsUrl(r.baseUrl)).toBe("http://127.0.0.1:8000/v1/audio/transcriptions");
  });

  it("detects Groq hosts and normalizes languages", () => {
    expect(isGroqUrl("https://api.groq.com/openai/v1")).toBe(true);
    expect(isGroqUrl("http://localhost:8000/v1")).toBe(false);
    expect(isGroqUrl("not a url")).toBe(false);
    expect(normalizeTranscriptionLanguage("pt_BR")).toBe("pt");
    expect(normalizeTranscriptionLanguage("auto")).toBe("");
    expect(normalizeTranscriptionLanguage("English")).toBe("");
  });
});

describe("misc helpers", () => {
  it("strips Whisper non-speech markers", () => {
    expect(stripWhisperNoiseMarkers("[MUSIC] hello (laughs) there .")).toBe("hello there.");
    expect(stripWhisperNoiseMarkers("[BLANK_AUDIO]")).toBe("");
  });

  it("validates the Azure segmentation silence", () => {
    expect(azureSegmentationSilenceMs("500")).toBe(500);
    expect(azureSegmentationSilenceMs("0")).toBeNull();
    expect(azureSegmentationSilenceMs("99")).toBeNull();
    expect(azureSegmentationSilenceMs("5001")).toBeNull();
    expect(azureSegmentationSilenceMs("")).toBeNull();
  });
});
