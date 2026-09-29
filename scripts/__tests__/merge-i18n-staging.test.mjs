import { describe, expect, it } from "vitest";
import { flatten, mergeStaging, removeKey, validateStaging } from "../merge-i18n-staging.mjs";

const en = () => ({
  common: { ok: "OK", cancel: "Cancel" },
  tts: { tiktok_title: "TikTok", pitch: "Pitch" },
});

describe("merge-i18n-staging", () => {
  it("adds new keys without touching existing ones and keeps order", () => {
    const { en: out, added, conflicts } = mergeStaging(en(), [
      { name: "P0-B.json", doc: { voice_changer: { budget: "Latency budget" }, common: { ok: "OK" } } },
    ]);
    expect(conflicts).toEqual([]);
    expect(added).toEqual(["voice_changer.budget"]);
    expect(Object.keys(out)).toEqual(["common", "tts", "voice_changer"]);
    expect(out.common).toEqual({ ok: "OK", cancel: "Cancel" });
  });

  it("refuses to change English wording unless allowed", () => {
    const staging = [{ name: "P0-E.json", doc: { tts: { pitch: "Pitch (semitones)" } } }];
    expect(mergeStaging(en(), staging).conflicts).toHaveLength(1);
    const { en: out, changed } = mergeStaging(en(), staging, { allowEnChanges: true });
    expect(changed).toEqual(["tts.pitch"]);
    expect(out.tts.pitch).toBe("Pitch (semitones)");
  });

  it("flags two workstreams staging the same key differently, and section/string clashes", () => {
    const { conflicts } = mergeStaging(en(), [
      { name: "A.json", doc: { app: { x: "one" } } },
      { name: "B.json", doc: { app: { x: "two" }, common: { ok: { nested: "no" } } } },
    ]);
    expect(conflicts).toHaveLength(2);
    // Same value from two files is fine.
    expect(
      mergeStaging(en(), [
        { name: "A.json", doc: { app: { x: "one" } } },
        { name: "B.json", doc: { app: { x: "one" } } },
      ]).conflicts,
    ).toEqual([]);
  });

  it("validates shapes and reports $remove", () => {
    expect(() => validateStaging("x.json", { a: 1 })).toThrow();
    expect(() => validateStaging("x.json", { "a.b": "no dots" })).toThrow();
    expect(() => validateStaging("x.json", { $remove: "tts.x" })).toThrow();
    const { removals, conflicts } = mergeStaging(en(), [{ name: "P0-E.json", doc: { $remove: ["tts.tiktok_title"] } }]);
    expect(conflicts).toEqual([]);
    expect(removals).toEqual([{ key: "tts.tiktok_title", from: "P0-E.json" }]);
  });

  it("removeKey prunes empty sections and flatten skips $remove", () => {
    const doc = { a: { b: { c: "x" } }, d: "y" };
    expect(removeKey(doc, "a.b.c")).toBe(true);
    expect(doc).toEqual({ d: "y" });
    expect(removeKey(doc, "missing.key")).toBe(false);
    expect(flatten({ $remove: ["x"], s: { k: "v" } })).toEqual({ "s.k": "v" });
  });
});
