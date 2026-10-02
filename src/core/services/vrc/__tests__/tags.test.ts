import { describe, expect, it } from "vitest";
import {
  EMOTION_VALUES,
  EVENT_VALUES,
  RecentTags,
  TagParameterDriver,
  avatarParameterAddress,
  decorateWithTags,
  hasKnownTags,
  parseTags,
  sanitizeParameterName,
} from "../tags";
import { FakeClock } from "./fakeClock";

describe("tag values", () => {
  it("are a stable published contract", () => {
    expect(EMOTION_VALUES).toEqual({ neutral: 0, happy: 1, sad: 2, angry: 3, fearful: 4, disgusted: 5, surprised: 6 });
    expect(EVENT_VALUES).toEqual({ laughter: 1, applause: 2, cry: 3, cough: 4, sneeze: 5, breath: 6, bgm: 7 });
  });

  it("parse the speech pipeline's tags and ignore unknown kinds and values", () => {
    expect(parseTags(["lang:en", "emotion:happy", "event:laughter"])).toEqual({ emotion: "happy", event: "laughter" });
    expect(parseTags(["emotion:bored", "event:dog", "x:y", "nocolon"])).toEqual({});
    expect(parseTags(undefined)).toEqual({});
    expect(hasKnownTags(["lang:ja"])).toBe(false);
    expect(hasKnownTags(["emotion:neutral"])).toBe(true);
  });
});

describe("decorateWithTags", () => {
  it("puts the emotion and event emoji in front; neutral shows nothing", () => {
    expect(decorateWithTags("Ha!", ["emotion:happy", "event:laughter"])).toBe("😊😂 Ha!");
    expect(decorateWithTags("Hi", ["emotion:neutral"])).toBe("Hi");
    expect(decorateWithTags("Hi", undefined)).toBe("Hi");
  });
});

describe("sanitizeParameterName", () => {
  it("accepts VRChat names, nested ones included, and rejects OSC pattern characters", () => {
    expect(sanitizeParameterName(" SigilEmotion ")).toBe("SigilEmotion");
    expect(sanitizeParameterName("/VF/Emotion.Int/")).toBe("VF/Emotion.Int");
    for (const bad of ["", "  ", "a b", "a*b", "a?b", "a[1]", "a{b}", "a,b", "a#b", "a//b"]) {
      expect(sanitizeParameterName(bad)).toBeNull();
    }
    expect(avatarParameterAddress("SigilEvent")).toBe("/avatar/parameters/SigilEvent");
  });
});

function driver(opts: { emotion?: string; event?: string; resetMs?: number } = {}) {
  const clock = new FakeClock();
  const sent: [string, number][] = [];
  const d = new TagParameterDriver({
    send: (name, value) => sent.push([name, value]),
    emotionParameter: () => opts.emotion ?? "SigilEmotion",
    eventParameter: () => opts.event ?? "SigilEvent",
    resetMs: () => opts.resetMs ?? 5_000,
    clock,
  });
  return { d, clock, sent };
}

describe("TagParameterDriver", () => {
  it("sets both Int parameters on a tagged final and resets them later", () => {
    const h = driver();
    h.d.onFinal({ tags: ["emotion:surprised", "event:applause"] });
    expect(h.sent).toEqual([["SigilEmotion", 6], ["SigilEvent", 2]]);
    h.clock.advance(4_999);
    expect(h.sent).toHaveLength(2);
    h.clock.advance(1);
    expect(h.sent.slice(2)).toEqual([["SigilEmotion", 0], ["SigilEvent", 0]]);
  });

  it("ignores finals without known tags (engines without tags never touch the avatar)", () => {
    const h = driver();
    h.d.onFinal({ tags: ["lang:en"] });
    h.d.onFinal({});
    expect(h.sent).toEqual([]);
  });

  it("an emotion without an event clears the event; a new tagged final restarts the reset", () => {
    const h = driver();
    h.d.onFinal({ tags: ["event:laughter"] });
    h.clock.advance(4_000);
    h.d.onFinal({ tags: ["emotion:sad"] });
    expect(h.sent).toEqual([["SigilEmotion", 0], ["SigilEvent", 1], ["SigilEmotion", 2], ["SigilEvent", 0]]);
    h.clock.advance(4_000);
    expect(h.sent).toHaveLength(4);
    h.clock.advance(1_000);
    expect(h.sent.slice(4)).toEqual([["SigilEmotion", 0]]);
  });

  it("skips empty or invalid parameter names and never resets with resetMs 0", () => {
    const h = driver({ event: "", resetMs: 0 });
    h.d.onFinal({ tags: ["emotion:angry", "event:cough"] });
    expect(h.sent).toEqual([["SigilEmotion", 3]]);
    h.clock.advance(60_000);
    expect(h.sent).toHaveLength(1);
    h.d.reset();
    expect(h.sent.at(-1)).toEqual(["SigilEmotion", 0]);
    const bad = driver({ emotion: "has space", event: "has space" });
    bad.d.onFinal({ tags: ["emotion:angry"] });
    expect(bad.sent).toEqual([]);
  });
});

describe("RecentTags", () => {
  it("remembers tags by segment id, bounded", () => {
    const r = new RecentTags(2);
    r.remember({ segmentId: "a", tags: ["emotion:happy"] });
    r.remember({ segmentId: "b", tags: ["event:cry"] });
    r.remember({ segmentId: "c" });
    r.remember({ segmentId: "d", tags: ["emotion:sad"] });
    expect(r.get("a")).toBeUndefined();
    expect(r.get("b")).toEqual(["event:cry"]);
    expect(r.get("c")).toBeUndefined();
    expect(r.get("d")).toEqual(["emotion:sad"]);
    expect(r.get(undefined)).toBeUndefined();
  });
});
