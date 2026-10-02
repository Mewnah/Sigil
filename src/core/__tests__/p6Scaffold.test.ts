import { describe, expect, it } from "vitest";
import { BackendSchema, PERSISTED_SERVICE_KEYS } from "../schema";
import { Element_TextStateSchemaN } from "@/client/elements/text/schema";
import { Services } from "@/services-registry";
import {
  CaptionEventSourceSchema,
  OthersEventSource,
  ReadingsEventSchema,
  ReadingsEventSource,
  TextEventSchema,
  TextEventSource,
} from "@/types";
import { STT_Backends, Service_STT_Schema } from "../services/stt/schema";
import { Service_TTS_Schema, TTS_Backends } from "../services/tts/schema";

/**
 * P6-S0 scaffold contracts: new topics, root keys, backend values and settings exist, and every P6 feature is off by
 * default (or is a new backend value nobody is switched to).
 */
describe("P6 scaffold", () => {
  it("defaults every P6 setting to off", () => {
    const s = BackendSchema.parse({});
    expect(s.lanAccess).toEqual({ mode: "lan", token: "", legacyNoToken: false });
    expect(s.captionWindow.enable).toBe(false);
    expect(s.captionWindow.sources).toEqual([TextEventSource.stt]);
    expect(s.readings.enable).toBe(false);
    expect(s.services.others.data.autoStart).toBe(false);
    expect(s.services.others.data.source).toBe("process:VRChat.exe");
    expect(s.services.others.data.translate).toBe(false);
    const vrc = s.services.vrc.data;
    expect(vrc.muteSync).toBe(false);
    expect(vrc.textbox.paginate).toBe(false);
    expect(vrc.textbox.commitOnTurnEnd).toBe(false);
    expect(vrc.tags.enable).toBe(false);
    expect(vrc.overlay.enable).toBe(false);
    expect(s.services.stt.data.local.musicGate).toBe(false);
    expect(s.services.stt.data.backend).toBe(STT_Backends.native);
    expect(s.services.tts.data.backend).toBe(TTS_Backends.native);
  });

  it("keeps saved settings from before P6 and repairs bad values", () => {
    const s = BackendSchema.parse({ lanAccess: { mode: "everywhere", token: 42 }, captionWindow: { sources: ["nope"] } });
    expect(s.lanAccess).toEqual({ mode: "lan", token: "", legacyNoToken: false });
    expect(s.captionWindow.sources).toEqual([TextEventSource.stt]);
    expect(PERSISTED_SERVICE_KEYS).toContain("others");
    expect(Services.others).toBe("others");
  });

  it("adds the new backend values without touching the old ones", () => {
    for (const b of ["qwen3", "edge", "gemini", "sbv2", "gptSovits", "pocket", "chatterboxLocal", "ggmlTts"]) {
      const tts = Service_TTS_Schema.parse({ backend: b });
      expect(tts.backend).toBe(b);
      expect(tts[b as keyof typeof tts]).toBeTypeOf("object");
    }
    expect(Service_TTS_Schema.parse({ backend: "openaiSpeech" }).backend).toBe(TTS_Backends.openaiSpeech);
    expect(Service_STT_Schema.parse({ backend: "geminiLive" }).backend).toBe(STT_Backends.geminiLive);
    expect(Service_STT_Schema.parse({}).geminiLive.model).toBe("gemini-3.5-transcribe-live");
  });

  it("keeps other players and readings outside text.*", () => {
    const topics: string[] = [...Object.values(OthersEventSource), ...Object.values(ReadingsEventSource)];
    for (const t of topics) {
      expect(t === TextEventSource.any || t.startsWith(`${TextEventSource.any}.`), t).toBe(false);
    }
    expect(CaptionEventSourceSchema.safeParse("others.stt").success).toBe(true);
    expect(CaptionEventSourceSchema.safeParse("readings.stt").success).toBe(false);
  });

  it("lets a text element show other players, and falls back for unknown sources", () => {
    expect(Element_TextStateSchemaN.parse({ sourceMain: "others.translation" }).sourceMain).toBe(OthersEventSource.translation);
    expect(Element_TextStateSchemaN.parse({ sourceMain: "readings.stt" }).sourceMain).toBe(TextEventSource.stt);
    expect(Element_TextStateSchemaN.parse({}).sourceMain).toBe(TextEventSource.stt);
  });

  it("validates text tags and readings payloads", () => {
    expect(TextEventSchema.parse({ type: 0, value: "hi", tags: ["emotion:happy"] }).tags).toEqual(["emotion:happy"]);
    expect(TextEventSchema.parse({ type: 0, value: "hi", tags: ["<b>"] }).tags).toBeUndefined();
    const r = ReadingsEventSchema.safeParse({ utteranceId: "mic-x:1:2", source: "text.stt", lang: "zh", value: "你好", pinyin: "nǐ hǎo" });
    expect(r.success).toBe(true);
  });
});
