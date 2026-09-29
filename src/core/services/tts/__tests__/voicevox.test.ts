import { describe, expect, it } from "vitest";
import { Service_TTS_Schema } from "../schema";
import {
  VOICEVOX_PRESET_HOSTS,
  VOICEVOX_RANGES,
  applyVoicevoxPreset,
  migrateVoicevoxSettings,
  normalizeVoicevoxHost,
  presetForHost,
  voicevoxHostProblem,
  voicevoxScales,
} from "../voicevoxPresets";

describe("VOICEVOX host presets", () => {
  it("uses each engine's default port", () => {
    expect(VOICEVOX_PRESET_HOSTS.voicevox).toBe("http://localhost:50021");
    expect(VOICEVOX_PRESET_HOSTS.aivisspeech).toBe("http://localhost:10101");
  });

  it("AivisSpeech resets pitch to 0 and sets its URL", () => {
    expect(applyVoicevoxPreset("aivisspeech")).toEqual({
      hostPreset: "aivisspeech",
      host: "http://localhost:10101",
      pitchScale: "0",
    });
  });

  it("VOICEVOX sets its URL and leaves pitch alone", () => {
    expect(applyVoicevoxPreset("voicevox")).toEqual({ hostPreset: "voicevox", host: "http://localhost:50021" });
  });

  it("custom keeps the current URL", () => {
    expect(applyVoicevoxPreset("custom")).toEqual({ hostPreset: "custom" });
  });

  it.each([
    ["http://localhost:50021", "voicevox"],
    ["http://127.0.0.1:50021/", "voicevox"],
    ["localhost:50021", "voicevox"],
    ["http://localhost:10101", "aivisspeech"],
    ["http://127.0.0.1:10101", "aivisspeech"],
    ["http://192.168.1.20:50021", "custom"],
    ["https://localhost:50021", "custom"],
    ["http://localhost:8080", "custom"],
    ["http://localhost:50021/engine", "custom"],
    ["not a url at all", "custom"],
    ["", "custom"],
  ] as const)("presetForHost(%s) = %s", (host, preset) => {
    expect(presetForHost(host)).toBe(preset);
  });
});

describe("normalizeVoicevoxHost", () => {
  it("trims, adds http:// and drops trailing slashes", () => {
    expect(normalizeVoicevoxHost(" http://localhost:50021/ ")).toBe("http://localhost:50021");
    expect(normalizeVoicevoxHost("localhost:10101")).toBe("http://localhost:10101");
    expect(normalizeVoicevoxHost("https://tts.example.com//")).toBe("https://tts.example.com");
    expect(normalizeVoicevoxHost("")).toBe("");
  });
});

describe("voicevoxHostProblem", () => {
  it("accepts local http and any https", () => {
    expect(voicevoxHostProblem("http://localhost:50021")).toBeNull();
    expect(voicevoxHostProblem("http://127.0.0.1:10101")).toBeNull();
    expect(voicevoxHostProblem("https://tts.example.com")).toBeNull();
  });

  it("explains hosts the webview cannot reach", () => {
    expect(voicevoxHostProblem("")).toBe("empty");
    expect(voicevoxHostProblem("http://192.168.1.20:50021")).toBe("remote_http");
    expect(voicevoxHostProblem("ftp://localhost:50021")).toBe("invalid_url");
    expect(voicevoxHostProblem("http://")).toBe("invalid_url");
  });
});

describe("voicevoxScales", () => {
  const base = { speedScale: "1.0", pitchScale: "0.0", intonationScale: "1.0", volumeScale: "1.0" };

  it("keeps 0 for intonation, volume and pitch", () => {
    expect(voicevoxScales({ ...base, intonationScale: "0", volumeScale: "0", pitchScale: "0" })).toEqual({
      speedScale: 1,
      pitchScale: 0,
      intonationScale: 0,
      volumeScale: 0,
    });
  });

  it("falls back on junk and clamps to the engine ranges", () => {
    expect(voicevoxScales({ speedScale: "0", pitchScale: "abc", intonationScale: "", volumeScale: "9" })).toEqual({
      speedScale: 1,
      pitchScale: 0,
      intonationScale: 1,
      volumeScale: 2,
    });
    expect(voicevoxScales({ ...base, pitchScale: "-1", speedScale: "5" })).toMatchObject({
      pitchScale: -0.15,
      speedScale: 2,
    });
  });

  it("pitch slider steps reach 0 exactly", () => {
    const { min, max, step } = VOICEVOX_RANGES.pitchScale;
    const stepsToZero = (0 - min) / step;
    expect(Math.abs(stepsToZero - Math.round(stepsToZero))).toBeLessThan(1e-9);
    expect(max).toBeGreaterThan(0);
  });
});

describe("voicevox settings migration", () => {
  it("infers the preset for settings saved before presets existed", () => {
    expect(migrateVoicevoxSettings({ host: "http://localhost:10101", speaker: "888753760" })).toEqual({
      host: "http://localhost:10101",
      speaker: "888753760",
      hostPreset: "aivisspeech",
    });
    expect(migrateVoicevoxSettings({ host: "http://192.168.1.5:50021" })).toMatchObject({ hostPreset: "custom" });
    expect(migrateVoicevoxSettings({})).toEqual({ hostPreset: "voicevox" });
  });

  it("keeps an existing valid preset", () => {
    const saved = { host: "http://localhost:50021", hostPreset: "custom" };
    expect(migrateVoicevoxSettings(saved)).toBe(saved);
  });

  it("parses old persisted TTS settings with the inferred preset", () => {
    const parsed = Service_TTS_Schema.parse({
      backend: "voicevox",
      voicevox: { host: "http://127.0.0.1:10101", speaker: "1", pitchScale: "0.05", intonationScale: "0" },
    });
    expect(parsed.voicevox.hostPreset).toBe("aivisspeech");
    expect(parsed.voicevox.host).toBe("http://127.0.0.1:10101");
    expect(parsed.voicevox.intonationScale).toBe("0");
    expect(parsed.voicevox.pitchScale).toBe("0.05");
  });

  it("gives fresh installs the VOICEVOX preset and survives corrupted slices", () => {
    expect(Service_TTS_Schema.parse({}).voicevox).toMatchObject({
      hostPreset: "voicevox",
      host: "http://localhost:50021",
    });
    expect(Service_TTS_Schema.parse({ voicevox: "garbage" }).voicevox.hostPreset).toBe("voicevox");
    expect(Service_TTS_Schema.parse({ voicevox: { hostPreset: "nope", host: "http://localhost:10101" } }).voicevox.hostPreset).toBe(
      "aivisspeech",
    );
  });
});

describe("other TTS schema fixes", () => {
  it("drops the trailing comma from persisted Azure voice ids", () => {
    const parsed = Service_TTS_Schema.parse({ azure: { voice: "zh-CN-XiaoyiNeural," } });
    expect(parsed.azure.voice).toBe("zh-CN-XiaoyiNeural");
  });

  it("adds optional Fish Speech key and model fields", () => {
    const parsed = Service_TTS_Schema.parse({ fishSpeech: { endpoint: "https://api.fish.audio" } });
    expect(parsed.fishSpeech).toMatchObject({ endpoint: "https://api.fish.audio", apiKey: "", model: "" });
  });
});
