import { describe, expect, it } from "vitest";
import { TextEventSource, TextEventType, type TextEvent } from "@/types";
import { CaptionModel, type CaptionModelOptions } from "../captionModel";

function ev(type: TextEventType, value: string, extra: Partial<TextEvent> = {}): TextEvent {
  return { type, value, emotes: {}, ...extra };
}
const interim = (value: string, extra: Partial<TextEvent> = {}) => ev(TextEventType.interim, value, extra);
const final = (value: string, extra: Partial<TextEvent> = {}) => ev(TextEventType.final, value, extra);

function setup(options: Partial<CaptionModelOptions> = {}) {
  let now = 0;
  const model = new CaptionModel({ interim: true, maxLines: 3, clearAfterMs: 6000, ...options }, () => now);
  return {
    model,
    advance: (ms: number) => {
      now += ms;
    },
    texts: () => model.lines.map((l) => `${l.kind}:${l.text}`),
  };
}

const STT = TextEventSource.stt;

describe("CaptionModel pairing", () => {
  it("revises one line per segment and keeps it after its final", () => {
    const { model, texts } = setup();
    expect(model.apply(STT, interim("hel", { segmentId: "mic-a:1:1", revision: 1 }))).toBe(true);
    expect(model.apply(STT, interim("hello wor", { segmentId: "mic-a:1:1", revision: 2 }))).toBe(true);
    expect(model.apply(STT, final("Hello world.", { segmentId: "mic-a:1:1", revision: 3 }))).toBe(true);
    expect(texts()).toEqual(["final:Hello world."]);
    // A late interim of the same segment is stale.
    expect(model.apply(STT, interim("hello", { segmentId: "mic-a:1:1", revision: 2 }))).toBe(false);
    expect(texts()).toEqual(["final:Hello world."]);
  });

  it("ignores lower revisions", () => {
    const { model, texts } = setup();
    model.apply(STT, interim("two", { segmentId: "s:1:1", revision: 5 }));
    expect(model.apply(STT, interim("one", { segmentId: "s:1:1", revision: 4 }))).toBe(false);
    expect(texts()).toEqual(["interim:two"]);
  });

  it("pairs sources without ids by the topic's open interim", () => {
    const { model, texts } = setup();
    model.apply(STT, interim("typing"));
    model.apply(STT, final("typed text"));
    model.apply(STT, interim("next"));
    expect(texts()).toEqual(["final:typed text", "interim:next"]);
  });

  it("keeps topics apart and pairs the transform's raw echo with its source line", () => {
    const { model, texts } = setup();
    model.apply(STT, final("hola", { segmentId: "m:1:1" }));
    model.apply(TextEventSource.translation, final("hello", { segmentId: "m:1:1" }));
    expect(texts()).toEqual(["final:hola", "final:hello"]);
    model.apply(TextEventSource.transform_source, interim("orig", { id: 7 }));
    model.apply(TextEventSource.transform_raw, final("rewritten", { id: 7 }));
    expect(texts()).toEqual(["final:hola", "final:hello", "final:rewritten"]);
  });

  it("drops the open interim of an earlier segment when the next one starts", () => {
    const { model, texts } = setup();
    model.apply(STT, interim("given up", { segmentId: "mic-a:1:1", revision: 1 }));
    model.apply(STT, interim("new", { segmentId: "mic-a:1:2", revision: 1 }));
    expect(texts()).toEqual(["interim:new"]);
    // Like the text element: a repeat of what left is stale, but its late final is still shown.
    expect(model.apply(STT, interim("given up", { segmentId: "mic-a:1:1", revision: 1 }))).toBe(false);
    expect(model.apply(STT, final("Given up, then finished.", { segmentId: "mic-a:1:1", revision: 3 }))).toBe(true);
    expect(texts()).toEqual(["interim:new", "final:Given up, then finished."]);
  });

  it("removes a line on an empty update (a cleared text field)", () => {
    const { model, texts } = setup();
    model.apply(TextEventSource.textfield, interim("abc"));
    expect(model.apply(TextEventSource.textfield, interim(""))).toBe(true);
    expect(texts()).toEqual([]);
    expect(model.apply(TextEventSource.textfield, interim(""))).toBe(false);
  });

  it("records language (lower case) and the committed prefix of interims", () => {
    const { model } = setup();
    model.apply(STT, interim("今日は", { segmentId: "x:1:1", lang: "ja", stableLength: 2 }));
    expect(model.lines[0]).toMatchObject({ lang: "ja", stableLength: 2, source: STT });
    model.apply(TextEventSource.translation, final("Hi", { segmentId: "x:1:1", lang: "zh-Hant" }));
    expect(model.lines[1].lang).toBe("zh-hant");
    // A final has no tentative tail.
    model.apply(STT, final("今日は晴れ", { segmentId: "x:1:1", stableLength: 3 }));
    expect(model.lines[0].stableLength).toBeUndefined();
  });
});

describe("CaptionModel limits", () => {
  it("keeps the newest maxLines lines", () => {
    const { model, texts } = setup({ maxLines: 2 });
    model.apply(STT, final("a"));
    model.apply(STT, final("b"));
    model.apply(STT, final("c"));
    expect(texts()).toEqual(["final:b", "final:c"]);
  });

  it("finals only when interims are off", () => {
    const { model, texts } = setup({ interim: false });
    expect(model.apply(STT, interim("x", { segmentId: "s:1:1" }))).toBe(false);
    model.apply(STT, final("done", { segmentId: "s:1:1" }));
    expect(texts()).toEqual(["final:done"]);
  });

  it("turning interims off or lowering maxLines updates the shown lines", () => {
    const { model, texts } = setup();
    model.apply(STT, final("a"));
    model.apply(STT, final("b"));
    model.apply(STT, interim("c"));
    expect(model.setOptions({ interim: false, maxLines: 3, clearAfterMs: 6000 })).toBe(true);
    expect(texts()).toEqual(["final:a", "final:b"]);
    expect(model.setOptions({ interim: false, maxLines: 1, clearAfterMs: 6000 })).toBe(true);
    expect(texts()).toEqual(["final:b"]);
    expect(model.setOptions({ interim: false, maxLines: 1, clearAfterMs: 6000 })).toBe(false);
  });

  it("expires lines clearAfterMs after their last update", () => {
    const { model, advance, texts } = setup({ clearAfterMs: 1000 });
    model.apply(STT, final("old"));
    advance(600);
    model.apply(STT, interim("live", { segmentId: "m:1:2" }));
    expect(model.nextExpiry()).toBe(1000);
    advance(400);
    expect(model.expire()).toBe(true);
    expect(texts()).toEqual(["interim:live"]);
    // An update restarts the line's timer.
    advance(500);
    model.apply(STT, interim("live more", { segmentId: "m:1:2", revision: 2 }));
    advance(900);
    expect(model.expire()).toBe(false);
    advance(100);
    expect(model.expire()).toBe(true);
    expect(texts()).toEqual([]);
    expect(model.nextExpiry()).toBeNull();
  });

  it("never expires with clearAfterMs 0", () => {
    const { model, advance } = setup({ clearAfterMs: 0 });
    model.apply(STT, final("stays"));
    advance(1e9);
    expect(model.expire()).toBe(false);
    expect(model.nextExpiry()).toBeNull();
    expect(model.lines).toHaveLength(1);
  });

  it("test lines are separate and removable", () => {
    const { model, texts } = setup();
    model.apply(STT, interim("speaking"));
    const key = model.push("Test caption", "test");
    expect(texts()).toEqual(["interim:speaking", "final:Test caption"]);
    expect(model.removeKey(key)).toBe(true);
    expect(model.removeKey(key)).toBe(false);
    expect(model.clear()).toBe(true);
    expect(model.clear()).toBe(false);
  });

  it("the clear timer leaves test lines alone", () => {
    const { model, advance, texts } = setup({ clearAfterMs: 1000 });
    model.push("Test caption", "test");
    expect(model.nextExpiry()).toBeNull();
    model.apply(STT, final("spoken"));
    expect(model.nextExpiry()).toBe(1000);
    advance(5000);
    expect(model.expire()).toBe(true);
    expect(texts()).toEqual(["final:Test caption"]);
    expect(model.nextExpiry()).toBeNull();
  });

  it("gives every line a unique key", () => {
    const { model } = setup({ maxLines: 10 });
    for (let i = 0; i < 5; i++) model.apply(STT, final(`l${i}`));
    expect(new Set(model.lines.map((l) => l.key)).size).toBe(5);
  });
});
