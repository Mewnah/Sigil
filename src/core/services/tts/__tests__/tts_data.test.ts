import { describe, expect, it } from "vitest";
import { azureVoiceLocale } from "../ssml";
import * as ttsData from "../tts_data";
import { azureVoices } from "../tts_data";

const entries = Object.entries(azureVoices).flatMap(([group, voices]) =>
  voices.map((voice) => ({ group, id: voice[0], label: voice[1], params: voice[2] })),
);

describe("azureVoices data", () => {
  it("has voices", () => {
    expect(entries.length).toBeGreaterThan(100);
  });

  it("every id is a valid Azure short name (no trailing commas)", () => {
    for (const { id } of entries) {
      expect(id, id).toMatch(/^[a-z]{2,3}-(?:[A-Z][a-z]{3}-)?[A-Z]{2}-[A-Za-z]+Neural$/);
      expect(azureVoiceLocale(id), id).toBeDefined();
    }
  });

  it("labels match the id's voice name", () => {
    for (const { id, label } of entries) {
      expect(id.endsWith(`-${label}`), `${id} / ${label}`).toBe(true);
    }
  });

  it("ids are unique", () => {
    const ids = entries.map((e) => e.id);
    expect(new Set(ids).size).toBe(ids.length);
  });

  it("styles and roles are lists of single tokens", () => {
    for (const { id, params } of entries) {
      for (const list of [params?.styles ?? [], params?.roles ?? []]) {
        for (const token of list) {
          expect(token, id).toMatch(/^[A-Za-z][A-Za-z-]*$/);
        }
      }
    }
  });

  it("YunyeNeural lists each role separately", () => {
    const yunye = entries.find((e) => e.id === "zh-CN-YunyeNeural");
    expect(yunye?.params?.roles).toEqual([
      "YoungAdultFemale",
      "YoungAdultMale",
      "OlderAdultFemale",
      "OlderAdultMale",
      "SeniorFemale",
      "SeniorMale",
      "Girl",
      "Boy",
    ]);
  });
});

describe("dead TikTok data", () => {
  it("tiktokVoices is gone", () => {
    expect("tiktokVoices" in ttsData).toBe(false);
  });
});
