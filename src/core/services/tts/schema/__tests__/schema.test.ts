import { describe, expect, it } from "vitest";
import { TextEventSource } from "@/types";
import { Service_TTS_Schema, TTS_Backends } from "..";
import { TTS_Route_Schema, TTS_Routes } from "../route";
import { ttsOutputMode, type ITTSService, type PcmChunk } from "../../types";

// P2-S0: the split schema keeps every legacy default and accepts the new backends; the ITTSService v2 output mode
// follows the capability order sink > pcm > self-playing.

describe("TTS schema split (P2-S0)", () => {
  it("keeps the legacy defaults of every slice", () => {
    const s = Service_TTS_Schema.parse({});
    expect(s.backend).toBe(TTS_Backends.native);
    expect(s.source).toBe(TextEventSource.stt);
    expect(s.inputField).toBe(true);
    expect(s.replaceWordsIgnoreCase).toBe(true);
    expect(s.windows.rate).toBe("1");
    expect(s.azure.language).toBe("English (United States)");
    expect(s.voicevox.hostPreset).toBe("voicevox");
    expect(s.kokoro.endpoint).toBe("http://localhost:8880");
    expect(s.melo.endpoint).toBe("http://localhost:8888");
    expect(s.chatterbox.exaggeration).toBe("0.5");
    expect(s.fishSpeech.endpoint).toBe("http://localhost:8080");
    expect(s.neuralOnnx.hubId).toBe("KittenML/kitten-tts-mini-0.8");
    expect(s.neuralSidecar.endpoint).toBe("http://127.0.0.1:8766");
  });

  it("accepts the new backends with their own slices", () => {
    expect(Service_TTS_Schema.parse({ backend: "localNeural" }).backend).toBe(TTS_Backends.localNeural);
    const s = Service_TTS_Schema.parse({ backend: "openaiSpeech" });
    expect(s.backend).toBe(TTS_Backends.openaiSpeech);
    expect(s.openaiSpeech.baseUrl).toBe("http://127.0.0.1:8880/v1");
    expect(Number(s.localNeural.speed)).toBe(1);
    expect(Service_TTS_Schema.parse({ backend: "nope" }).backend).toBe(TTS_Backends.native);
  });

  it("parses routes safely", () => {
    expect(TTS_Route_Schema.parse("both")).toBe(TTS_Routes.both);
    expect(TTS_Route_Schema.parse("elsewhere")).toBe(TTS_Routes.device);
  });
});

describe("ITTSService v2 output mode (P2-S0)", () => {
  const base: ITTSService = { start() {}, stop() {}, play() {}, dispose() {} };

  it("prefers sink, then pcm, then the legacy self-playing path", () => {
    async function* pcm(): AsyncIterable<PcmChunk> {
      yield { samples: new Float32Array(1), sampleRate: 24_000, channels: 1 };
    }
    expect(ttsOutputMode(base)).toBe("selfPlaying");
    expect(ttsOutputMode({ ...base, synthesizePcm: () => pcm() })).toBe("pcm");
    expect(ttsOutputMode({ ...base, synthesizePcm: () => pcm(), synthesizeToSink: async () => {} })).toBe("sink");
  });
});
