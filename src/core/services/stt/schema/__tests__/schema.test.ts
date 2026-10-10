import { describe, expect, it } from "vitest";
import { BackendSchema } from "@/core/schema";
import { TextEventSchema, TextEventType } from "@/types";
import { OpenAiAudioPreset, Service_STT_Schema, STT_Backends } from "..";

// P2-S0: the split schema keeps every legacy default, accepts the new `local` backend, and the optional TextEvent
// revision fields are lenient (a malformed peer value is dropped, never the whole event).

describe("STT schema split (P2-S0)", () => {
  it("keeps the legacy defaults of every slice", () => {
    const s = Service_STT_Schema.parse({});
    expect(s.backend).toBe(STT_Backends.native);
    expect(s.native.language).toBe("en-US");
    expect(s.azure.segmentationSilenceMs).toBe("500");
    expect(s.vad.maxUtteranceMs).toBe("18000");
    expect(s.whisper.model).toBe("base.en");
    expect(s.deepgram.model).toBe("nova-3");
    expect(s.vosk.model).toBe("vosk-model-small-en-us-0.15");
    expect(s.moonshine.endpoint).toBe("http://localhost:8090");
    expect(s.openai_audio.preset).toBe(OpenAiAudioPreset.custom);
    expect(s.openai_audio.groqModel).toBe("whisper-large-v3-turbo");
  });

  it("accepts the local backend and keeps unknown backends safe", () => {
    const s = Service_STT_Schema.parse({ backend: "local" });
    expect(s.backend).toBe(STT_Backends.local);
    expect(s.local.language).toBe("auto");
    expect(s.local.tier).toBe("auto");
    expect(Service_STT_Schema.parse({ backend: "no-such-backend" }).backend).toBe(STT_Backends.native);
  });

  it("still normalizes the legacy Deepgram language typo", () => {
    expect(Service_STT_Schema.parse({ deepgram: { language: "zn-CN" } }).deepgram.language).toBe("zh-CN");
  });
});

// Interim results switch of the local backend: a new key in an already persisted slice (`services.stt.data.local`), so
// no migration step and no `PERSISTED_SERVICE_KEYS` change: zod fills the default when a saved file lacks the key.
describe("local.interim (interim results switch)", () => {
  it("defaults to on", () => {
    expect(Service_STT_Schema.parse({}).local.interim).toBe(true);
    expect(Service_STT_Schema.parse({ backend: "local" }).local.interim).toBe(true);
    expect(Service_STT_Schema.parse({ local: {} }).local.interim).toBe(true);
  });

  it("loads a profile saved before the key existed as on, keeping its other values", () => {
    const saved = { language: "ja", tier: "cpu", twoPass: false, smartTurn: true, hotwords: "Sigil", hallucinationFilter: false, musicGate: true };
    const s = Service_STT_Schema.parse({ backend: "local", local: saved });
    expect(s.local).toMatchObject({ ...saved, interim: true });
    // Through the whole persisted settings file (`Service_State.init` parses it with `BackendSchema`).
    const file = BackendSchema.parse({ services: { stt: { showActionButton: true, data: { backend: "local", local: saved } } } });
    expect(file.services.stt.data.local).toMatchObject({ ...saved, interim: true });
    expect(file.services.stt.showActionButton).toBe(true);
    expect(BackendSchema.parse({}).services.stt.data.local.interim).toBe(true);
  });

  it("keeps an explicit choice and falls back to on for a missing value", () => {
    expect(Service_STT_Schema.parse({ local: { interim: false } }).local.interim).toBe(false);
    expect(Service_STT_Schema.parse({ local: { interim: true } }).local.interim).toBe(true);
    expect(Service_STT_Schema.parse({ local: { interim: undefined } }).local.interim).toBe(true);
    // Round trip through JSON (how settings are stored).
    const off = Service_STT_Schema.parse({ local: { interim: false } });
    expect(Service_STT_Schema.parse(JSON.parse(JSON.stringify(off))).local.interim).toBe(false);
  });

  it("does not touch the other backends' interim switches", () => {
    const s = Service_STT_Schema.parse({ local: { interim: false } });
    expect(s.azure.interim).toBe(true);
    expect(s.deepgram.interim).toBe(true);
    expect(s.geminiLive.interim).toBe(true);
  });
});

describe("TextEvent revision fields (P2-S0)", () => {
  it("passes valid fields through", () => {
    const e = TextEventSchema.parse({
      type: TextEventType.final,
      value: "hi",
      segmentId: "mic:3:17",
      revision: 2,
      lang: "ja",
      turnEnd: true,
    });
    expect(e).toMatchObject({ segmentId: "mic:3:17", revision: 2, lang: "ja", turnEnd: true });
  });

  it("drops malformed fields without rejecting the event", () => {
    const e = TextEventSchema.safeParse({
      type: TextEventType.interim,
      value: "hi",
      segmentId: "<img src=x>",
      revision: -1,
      lang: "EN US",
      turnEnd: "yes",
    });
    expect(e.success).toBe(true);
    if (!e.success) return;
    expect(e.data.value).toBe("hi");
    expect(e.data.segmentId).toBeUndefined();
    expect(e.data.revision).toBeUndefined();
    expect(e.data.lang).toBeUndefined();
    expect(e.data.turnEnd).toBeUndefined();
  });

  it("leaves old events unchanged", () => {
    const e = TextEventSchema.parse({ type: TextEventType.final, value: "x" });
    expect(e.segmentId).toBeUndefined();
    expect(e.revision).toBeUndefined();
  });
});
