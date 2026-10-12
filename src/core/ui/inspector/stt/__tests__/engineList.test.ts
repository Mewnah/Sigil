import i18n from "i18next";
import { beforeAll, describe, expect, it } from "vitest";
import { OpenAiAudioPreset, STT_Backends } from "@/core/services/stt/schema";
import { initSttTestI18n } from "@/core/services/stt/__tests__/sttTestI18n";
import { PROVIDER_GROQ, sttEngineGroups, sttProviderValue, type SttEngineGroup } from "../engineList";

const t = (key: string) => i18n.t(key);
const values = (groups: SttEngineGroup[]) => groups.flatMap((g) => g.options.map((o) => o.value));
const groupValues = (groups: SttEngineGroup[]) => groups.map((g) => g.options.map((o) => o.value));

beforeAll(async () => {
  await initSttTestI18n();
});

describe("sttEngineGroups", () => {
  it("lists the groups in order, with the built-in engine first under Recommended", () => {
    const groups = sttEngineGroups(t, { desktop: true, current: STT_Backends.local });
    expect(groups.map((g) => g.label)).toEqual([
      "Recommended",
      "Online (your own account or key)",
      "Chrome / Edge microphone page",
      "Other engines",
    ]);
    expect(groups[0].options).toEqual([{ label: "Built-in (runs on your PC)", value: STT_Backends.local }]);
  });

  it("puts each engine in its group (desktop)", () => {
    expect(groupValues(sttEngineGroups(t, { desktop: true, current: STT_Backends.local }))).toEqual([
      [STT_Backends.local],
      [PROVIDER_GROQ, STT_Backends.deepgram, STT_Backends.azure, STT_Backends.geminiLive],
      [STT_Backends.chrome, STT_Backends.edge],
      [STT_Backends.whisper, STT_Backends.vosk, STT_Backends.openai_audio],
    ]);
  });

  it("offers Web Speech only where it can work (the web build)", () => {
    const desktop = sttEngineGroups(t, { desktop: true, current: STT_Backends.local });
    const web = sttEngineGroups(t, { desktop: false, current: STT_Backends.local });
    expect(values(desktop)).not.toContain(STT_Backends.native);
    expect(web[3].options.map((o) => o.value)).toEqual([
      STT_Backends.whisper,
      STT_Backends.vosk,
      STT_Backends.openai_audio,
      STT_Backends.native,
    ]);
    expect(web[3].options[3].label).toBe("Web Speech (browser only)");
  });

  it("never lists the retired Moonshine server", () => {
    for (const desktop of [true, false])
      for (const current of Object.values(STT_Backends).filter((b) => b !== STT_Backends.moonshine)) {
        const groups = sttEngineGroups(t, { desktop, current });
        expect(values(groups)).not.toContain(STT_Backends.moonshine);
        expect(groups.flatMap((g) => g.options.map((o) => o.label.toLowerCase())).join("|")).not.toContain("moonshine");
      }
  });

  it("lists every selectable engine exactly once", () => {
    const selectable = Object.values(STT_Backends).filter(
      (b) => b !== STT_Backends.moonshine && b !== STT_Backends.browser
    );
    for (const desktop of [true, false]) {
      const all = values(sttEngineGroups(t, { desktop, current: STT_Backends.local }));
      expect(new Set(all).size).toBe(all.length);
      // Web Speech is offered only in the web build. Groq is the openai_audio engine with the Groq preset: both rows exist.
      for (const b of selectable.filter((b) => desktop === false || b !== STT_Backends.native)) expect(all).toContain(b);
      expect(all).toContain(PROVIDER_GROQ);
    }
  });

  it("always contains the profile's current engine, so the selector never shows 'Select'", () => {
    for (const desktop of [true, false])
      for (const current of [...Object.values(STT_Backends), PROVIDER_GROQ]) {
        const all = values(sttEngineGroups(t, { desktop, current }));
        expect(all.filter((v) => v === current)).toHaveLength(1);
      }
  });

  it("names an unlisted current engine in the last group: Web Speech and the retired Moonshine by their labels, others by their id", () => {
    const native = sttEngineGroups(t, { desktop: true, current: STT_Backends.native });
    expect(native[3].options.at(-1)).toEqual({ label: "Web Speech (browser only)", value: STT_Backends.native });
    const browser = sttEngineGroups(t, { desktop: true, current: STT_Backends.browser });
    expect(browser[3].options.at(-1)).toEqual({ label: "Web Speech (browser only)", value: STT_Backends.browser });
    const moonshine = sttEngineGroups(t, { desktop: false, current: STT_Backends.moonshine });
    // A saved web profile on the retired Moonshine server shows its name, not the lower-case id.
    expect(moonshine[3].options.at(-1)).toEqual({ label: "Moonshine", value: STT_Backends.moonshine });
    expect(sttEngineGroups(t, { desktop: true, current: "some-future-engine" })[3].options.at(-1)).toEqual({
      label: "some-future-engine",
      value: "some-future-engine",
    });
    // A listed engine adds nothing.
    expect(groupValues(sttEngineGroups(t, { desktop: true, current: STT_Backends.vosk }))[3]).toEqual([
      STT_Backends.whisper,
      STT_Backends.vosk,
      STT_Backends.openai_audio,
    ]);
  });

  it("shows no raw translation key for any label", () => {
    for (const desktop of [true, false]) {
      const labels = sttEngineGroups(t, { desktop, current: STT_Backends.local }).flatMap((g) => [g.label, ...g.options.map((o) => o.label)]);
      for (const label of labels) expect(label).not.toMatch(/^[a-z_]+\.[a-z_]+$/);
    }
  });
});

describe("sttProviderValue", () => {
  it("shows Groq for the openai_audio engine with the Groq preset, every other pair by its backend", () => {
    expect(sttProviderValue(STT_Backends.openai_audio, OpenAiAudioPreset.groq)).toBe(PROVIDER_GROQ);
    expect(sttProviderValue(STT_Backends.openai_audio, OpenAiAudioPreset.custom)).toBe(STT_Backends.openai_audio);
    expect(sttProviderValue(STT_Backends.local, OpenAiAudioPreset.groq)).toBe(STT_Backends.local);
  });
});
