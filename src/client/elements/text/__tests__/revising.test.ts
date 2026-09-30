import { describe, expect, it } from "vitest";
import {
  CaptionDom,
  CaptionLine,
  CaptionUpdate,
  captionStyle,
  codePointBoundaries,
  commonPrefixLength,
  detachCaptionLines,
  findCaptionLine,
  floorBoundary,
  graphemeBoundaries,
  isStaleUpdate,
  keepLength,
  layoutAtoms,
  nextBoundary,
  profanityRanges,
  readCaptionMeta,
  RETIRED_LINE_TTL_MS,
  RetiredCaptionLines,
  segmentStream,
  supersededCaptionLines,
  type CaptionKeyed,
  type CaptionKind,
} from "../revising";

// --- a tiny fake DOM --------------------------------------------------------------------------------------------

type FakeNode = {
  kind: "element" | "text" | "span" | "img";
  data: string;
  className: string;
  url: string;
  parent: FakeNode | null;
  children: FakeNode[];
};

function makeDom() {
  const removed: FakeNode[] = [];
  const replaced: FakeNode[] = [];
  const mk = (kind: FakeNode["kind"], init: Partial<FakeNode> = {}): FakeNode => ({
    kind, data: "", className: "", url: "", parent: null, children: [], ...init,
  });
  const detach = (n: FakeNode) => {
    if (!n.parent) return;
    n.parent.children.splice(n.parent.children.indexOf(n), 1);
    n.parent = null;
  };
  const dom: CaptionDom<FakeNode> = {
    text: (data) => mk("text", { data }),
    span: (className, data) => mk("span", { className, data }),
    image: (className, url) => mk("img", { className, url }),
    setData: (n, data) => { n.data = data; },
    setClass: (n, className) => { n.className = className; },
    append: (parent, n) => { detach(n); n.parent = parent; parent.children.push(n); },
    remove: (n) => { removed.push(n); detach(n); },
    replace: (old, ...nodes) => {
      const parent = old.parent!;
      nodes.forEach(detach);
      parent.children.splice(parent.children.indexOf(old), 1, ...nodes);
      nodes.forEach((n) => { n.parent = parent; });
      old.parent = null;
      replaced.push(old);
    },
  };
  const root = mk("element");
  return { dom, root, removed, replaced };
}

/** Markup of a line: text as is, spans as {class:text}, images as <url.class>. */
function markup(el: FakeNode): string {
  return el.children.map((c) => {
    if (c.kind === "text") return c.data;
    if (c.kind === "span") return `{${c.className}:${c.data}}`;
    return `<${c.url}${c.className ? `.${c.className.replace(/ /g, ".")}` : ""}>`;
  }).join("");
}

/** What a viewer reads. */
function visible(el: FakeNode): string {
  return el.children.map((c) => (c.kind === "img" ? "[img]" : c.data)).join("");
}

const TOPIC = "text.translation";

function newLine(kind: CaptionKind = "interim", extra: Partial<CaptionKeyed> = {}) {
  const fake = makeDom();
  const line = new CaptionLine(fake.dom, fake.root, { topic: TOPIC, kind, ...extra });
  return { ...fake, line };
}

function upd(kind: CaptionKind, text: string, more: Partial<CaptionUpdate> = {}): CaptionUpdate {
  return { kind, text, mode: "positions", meta: { topic: TOPIC, kind }, ...more };
}

// --- text helpers -----------------------------------------------------------------------------------------------

describe("text helpers", () => {
  it("measures the common prefix in code units", () => {
    expect(commonPrefixLength("hello world", "hello there")).toBe(6);
    expect(commonPrefixLength("", "x")).toBe(0);
    expect(commonPrefixLength("same", "same")).toBe(4);
  });

  it("finds grapheme boundaries (accents and emoji sequences stay whole)", () => {
    expect([...graphemeBoundaries("éx")]).toEqual([1, 0, 1, 1]);
    const family = "\u{1F468}‍\u{1F469}‍\u{1F467}x";
    const b = graphemeBoundaries(family);
    expect(b[0]).toBe(1);
    expect(b[family.length - 1]).toBe(1);
    expect(b.slice(1, family.length - 1).every((v) => v === 0)).toBe(true);
  });

  it("falls back to code point boundaries", () => {
    expect([...codePointBoundaries("a\u{1F600}b")]).toEqual([1, 1, 0, 1, 1]);
  });

  it("floors and advances over boundaries", () => {
    const b = codePointBoundaries("a\u{1F600}b");
    expect(floorBoundary(b, 2)).toBe(1);
    expect(floorBoundary(b, 99)).toBe(4);
    expect(nextBoundary(b, 1, 4)).toBe(3);
    expect(nextBoundary(b, 1, 2)).toBe(2);
  });

  it("finds masked words", () => {
    expect(profanityRanges("what the f**k, man")).toEqual([[9, 13]]);
    expect(profanityRanges("no stars here")).toEqual([]);
  });
});

describe("layoutAtoms", () => {
  it("places position emotes on word starts, masks first", () => {
    expect(layoutAtoms("hi Kappa f**k you", { 3: "k.png", Kappa: "k.png", 9: "x.png" }, "positions")).toEqual([
      { kind: "emote", start: 3, end: 8, url: "k.png" },
      { kind: "mask", start: 9, end: 13 },
    ]);
  });

  it("ignores positions that are not word starts of this text", () => {
    expect(layoutAtoms("hi Kappa", { 4: "k.png", 2: "k.png", 50: "k.png" }, "positions")).toEqual([]);
  });

  it("replaces emote names in static mode, longest name first, never inside a mask", () => {
    expect(layoutAtoms("LULW and LUL", { LUL: "a", LULW: "b" }, "names")).toEqual([
      { kind: "emote", start: 0, end: 4, url: "b" },
      { kind: "emote", start: 9, end: 12, url: "a" },
    ]);
    expect(layoutAtoms("LUL f**k LUL", { LUL: "a" }, "names").map((a) => [a.kind, a.start, a.end])).toEqual([
      ["emote", 0, 3],
      ["mask", 4, 8],
      ["emote", 9, 12],
    ]);
  });

  it("survives junk emote maps", () => {
    expect(layoutAtoms("hi", null, "positions")).toEqual([]);
    expect(layoutAtoms("hi", "nope", "names")).toEqual([]);
    expect(layoutAtoms("hi there", { 3: 42, there: null }, "positions")).toEqual([]);
  });
});

describe("keepLength", () => {
  const plain = (text: string) => ({ text, atoms: layoutAtoms(text, {}, "positions"), bounds: graphemeBoundaries(text) });

  it("keeps the common prefix when only the tail changes", () => {
    expect(keepLength(plain("the quick brown fox"), 19, plain("the quick brown dog"))).toBe(16);
  });

  it("never keeps more than is shown", () => {
    expect(keepLength(plain("the quick brown fox"), 5, plain("the quick brown dog"))).toBe(5);
  });

  it("does not end inside a grapheme of the new text", () => {
    expect(keepLength(plain("cafe"), 4, plain("café au lait"))).toBe(3);
  });

  it("drops a mask that grew and an emote that appeared", () => {
    expect(keepLength(plain("oh f**k"), 7, plain("oh f**king hell"))).toBe(3);
    const next = { text: "Kappa", atoms: layoutAtoms("Kappa", { 0: "k.png" }, "positions"), bounds: graphemeBoundaries("Kappa") };
    expect(keepLength(plain("Kap"), 3, next)).toBe(0);
  });

  it("keeps an unchanged emote", () => {
    const at = (text: string) => ({ text, atoms: layoutAtoms(text, { 0: "k.png" }, "positions"), bounds: graphemeBoundaries(text) });
    expect(keepLength(at("Kappa hi"), 8, at("Kappa hello"))).toBe(7);
  });
});

describe("captionStyle", () => {
  const b = graphemeBoundaries("hello world");

  it("commits finals and keeps the legacy look for interims without stableLength", () => {
    expect(captionStyle("final", 3, b)).toEqual({ interim: false, tentativeFrom: Infinity });
    expect(captionStyle("interim", undefined, b)).toEqual({ interim: true, tentativeFrom: Infinity });
  });

  it("keeps the interim look for a committed prefix and marks the tail after stableLength", () => {
    // The final may still replace committed interim text once: it must not look final.
    expect(captionStyle("interim", 6, b)).toEqual({ interim: true, tentativeFrom: 6 });
    expect(captionStyle("interim", 11, b)).toEqual({ interim: true, tentativeFrom: 11 });
    expect(captionStyle("interim", 400, b)).toEqual({ interim: true, tentativeFrom: 11 });
    expect(captionStyle("interim", 1, graphemeBoundaries("é")).tentativeFrom).toBe(0);
  });

  it("treats malformed stableLength as absent", () => {
    for (const bad of [-1, 1.5, "3", null, NaN])
      expect(captionStyle("interim", bad, b)).toEqual({ interim: true, tentativeFrom: Infinity });
  });
});

// --- pairing ----------------------------------------------------------------------------------------------------

describe("pairing", () => {
  it("reads only well-formed pairing fields", () => {
    expect(readCaptionMeta("t", "final", { segmentId: "mic:1:4", id: 3, revision: 2 })).toEqual({
      topic: "t", kind: "final", segmentId: "mic:1:4", id: 3, revision: 2,
    });
    expect(readCaptionMeta("t", "interim", { segmentId: "has space", id: NaN, revision: -1 })).toEqual({ topic: "t", kind: "interim" });
    expect(readCaptionMeta("t", "interim", { segmentId: "x".repeat(129), revision: 1.5 })).toEqual({ topic: "t", kind: "interim" });
  });

  const lines: CaptionKeyed[] = [
    { topic: "text.stt", kind: "interim", segmentId: "mic:1:1", revision: 3 },
    { topic: "text.translation", kind: "interim", segmentId: "mic:1:1", revision: 1 },
    { topic: "text.transform_source", kind: "interim", segmentId: "mic:1:2", id: 7, revision: 5 },
    { topic: "text.textfield", kind: "final" },
    { topic: "text.textfield", kind: "interim" },
    { topic: "text.transform_source", kind: "interim", id: 8 },
  ];

  it("pairs by segmentId within the topic", () => {
    expect(findCaptionLine(lines, { topic: "text.translation", kind: "interim", segmentId: "mic:1:1" })).toBe(1);
    expect(findCaptionLine(lines, { topic: "text.stt", kind: "final", segmentId: "mic:1:1" })).toBe(0);
    expect(findCaptionLine(lines, { topic: "text.stt", kind: "final", segmentId: "mic:1:9" })).toBe(-1);
  });

  it("pairs the AI transform's id-only replies and its final", () => {
    expect(findCaptionLine(lines, { topic: "text.transform_source", kind: "interim", id: 7 })).toBe(2);
    // A final with a new segment id falls back to its id, but only onto a line without a segment id.
    expect(findCaptionLine(lines, { topic: "text.transform_source", kind: "final", segmentId: "mic:1:3", id: 8 })).toBe(5);
    expect(findCaptionLine(lines, { topic: "text.transform_source", kind: "final", segmentId: "mic:1:3", id: 7 })).toBe(-1);
  });

  it("pairs an id-only update with the newest line of that id", () => {
    // Segment A was given up without a final, so the transform's id did not advance: A and B share id 0.
    const shared: CaptionKeyed[] = [
      { topic: "text.transform_source", kind: "interim", segmentId: "mic:1:1", id: 0, revision: 2 },
      { topic: "text.transform_source", kind: "interim", segmentId: "mic:1:2", id: 0, revision: 9 },
      { topic: "text.stt", kind: "interim", id: 0 },
    ];
    // B's final went to the AI: its streamed reply belongs to B, not to the stale line of A.
    expect(findCaptionLine(shared, { topic: "text.transform_source", kind: "interim", id: 0 })).toBe(1);
    const unkeyed: CaptionKeyed[] = [
      { topic: "t", kind: "interim", id: 3 },
      { topic: "t", kind: "interim", id: 3 },
    ];
    expect(findCaptionLine(unkeyed, { topic: "t", kind: "final", segmentId: "x:1:1", id: 3 })).toBe(1);
  });

  it("pairs legacy updates with the topic's open unkeyed interim", () => {
    expect(findCaptionLine(lines, { topic: "text.textfield", kind: "final" })).toBe(4);
    expect(findCaptionLine(lines, { topic: "text.stt", kind: "interim" })).toBe(-1);
  });

  it("drops stale updates", () => {
    const line: CaptionKeyed = { topic: "t", kind: "interim", segmentId: "s", revision: 4 };
    expect(isStaleUpdate(line, { topic: "t", kind: "interim", segmentId: "s", revision: 3 })).toBe(true);
    expect(isStaleUpdate(line, { topic: "t", kind: "interim", segmentId: "s", revision: 4 })).toBe(false);
    expect(isStaleUpdate(line, { topic: "t", kind: "final", segmentId: "s", revision: 5 })).toBe(false);
    expect(isStaleUpdate(line, { topic: "t", kind: "interim", segmentId: "s" })).toBe(false);
    const done: CaptionKeyed = { ...line, kind: "final", revision: 5 };
    expect(isStaleUpdate(done, { topic: "t", kind: "interim", segmentId: "s", revision: 9 })).toBe(true);
    expect(isStaleUpdate(done, { topic: "t", kind: "final", segmentId: "s", revision: 6 })).toBe(false);
    expect(isStaleUpdate(done, { topic: "t", kind: "final", segmentId: "s", revision: 4 })).toBe(true);
  });

  it("remembers lines that left the screen", () => {
    const retired = new RetiredCaptionLines(2);
    retired.retire({ topic: "t", kind: "final", segmentId: "a", revision: 2 });
    retired.retire({ topic: "t", kind: "interim", segmentId: "b", revision: 4 });
    retired.retire({ topic: "t", kind: "interim" });
    expect(retired.size).toBe(2);
    expect(retired.blocks({ topic: "t", kind: "final", segmentId: "a", revision: 9 })).toBe(true);
    expect(retired.blocks({ topic: "u", kind: "final", segmentId: "a", revision: 9 })).toBe(false);
    expect(retired.blocks({ topic: "t", kind: "interim", segmentId: "b", revision: 3 })).toBe(true);
    expect(retired.blocks({ topic: "t", kind: "final", segmentId: "b", revision: 4 })).toBe(false);
    expect(retired.blocks({ topic: "t", kind: "final", segmentId: "b" })).toBe(false);
    retired.revive({ topic: "t", kind: "final", segmentId: "b" });
    expect(retired.blocks({ topic: "t", kind: "interim", segmentId: "b", revision: 0 })).toBe(false);
    retired.retire({ topic: "t", kind: "final", segmentId: "c" });
    retired.retire({ topic: "t", kind: "final", segmentId: "d" });
    expect(retired.size).toBe(2);
    expect(retired.blocks({ topic: "t", kind: "final", segmentId: "a" })).toBe(false);
  });

  it("blocks a repeat of the interim a line left with, but not its final at that revision", () => {
    const retired = new RetiredCaptionLines();
    // The AI transform shows the STT final (revision 7) as an interim while the AI works; the timer cleared it.
    retired.retire({ topic: "t", kind: "interim", segmentId: "s", revision: 7 });
    expect(retired.blocks({ topic: "t", kind: "interim", segmentId: "s", revision: 7 })).toBe(true);
    expect(retired.blocks({ topic: "t", kind: "interim", segmentId: "s", revision: 8 })).toBe(false);
    expect(retired.blocks({ topic: "t", kind: "final", segmentId: "s", revision: 7 })).toBe(false);
    expect(retired.blocks({ topic: "t", kind: "final", segmentId: "s", revision: 6 })).toBe(true);
  });

  it("forgets retired lines after the TTL, so a reused segment id is shown again", () => {
    // A restarted Sigil (or a linked host) publishes `mic:1:1` again: its interims and final must not be dropped
    // forever because an older `mic:1:1` once ended final on this page.
    let now = 1_000;
    const retired = new RetiredCaptionLines(256, RETIRED_LINE_TTL_MS, () => now);
    retired.retire({ topic: "text.stt", kind: "final", segmentId: "mic:1:1", revision: 12 });
    retired.retire({ topic: "text.translation", kind: "final", segmentId: "mic:1:1", revision: 3 });
    now += 5_000;
    expect(retired.blocks({ topic: "text.stt", kind: "interim", segmentId: "mic:1:1", revision: 1 })).toBe(true);
    now += RETIRED_LINE_TTL_MS;
    expect(retired.blocks({ topic: "text.stt", kind: "interim", segmentId: "mic:1:1", revision: 1 })).toBe(false);
    expect(retired.blocks({ topic: "text.stt", kind: "final", segmentId: "mic:1:1", revision: 4 })).toBe(false);
    expect(retired.blocks({ topic: "text.translation", kind: "final", segmentId: "mic:1:1", revision: 1 })).toBe(false);
    // Expired entries are pruned when the next line retires.
    retired.retire({ topic: "text.stt", kind: "final", segmentId: "mic:1:2" });
    expect(retired.size).toBe(1);
  });

  it("forgets everything on clear (the /client page reconnected to a possibly restarted host)", () => {
    const retired = new RetiredCaptionLines();
    retired.retire({ topic: "text.stt", kind: "final", segmentId: "mic:1:1", revision: 12 });
    retired.clear();
    expect(retired.size).toBe(0);
    expect(retired.blocks({ topic: "text.stt", kind: "final", segmentId: "mic:1:1", revision: 2 })).toBe(false);
  });

  it("names the stream of a segment id", () => {
    expect(segmentStream("mic:1:7")).toBe("mic");
    expect(segmentStream("mic:12:7.3")).toBe("mic");
    expect(segmentStream("tr.textfield.0:3")).toBe("tr.textfield.0");
    expect(segmentStream("plain")).toBeUndefined();
    expect(segmentStream(":1")).toBeUndefined();
  });

  it("gives up the stream's open interims of earlier segments when a new segment starts", () => {
    const open: CaptionKeyed[] = [
      { topic: "text.stt", kind: "final", segmentId: "mic:1:5", revision: 9 },
      { topic: "text.stt", kind: "interim", segmentId: "mic:1:6", revision: 3 },
      { topic: "text.translation", kind: "interim", segmentId: "mic:1:6", revision: 1 },
      { topic: "text.translation", kind: "interim", segmentId: "tr.textfield.0:2", revision: 1 },
      { topic: "text.stt", kind: "interim" },
      { topic: "text.stt", kind: "interim", segmentId: "mic:1:7", revision: 1 },
    ];
    // `mic:1:6` was dropped upstream (nothing published): the first interim of `mic:1:8` ends it and `mic:1:7`.
    expect(supersededCaptionLines(open, { topic: "text.stt", kind: "interim", segmentId: "mic:1:8", revision: 1 })).toEqual([1, 5]);
    // Per topic and per stream: typed-text translations are another stream.
    expect(supersededCaptionLines(open, { topic: "text.translation", kind: "interim", segmentId: "mic:1:8" })).toEqual([2]);
    expect(supersededCaptionLines(open, { topic: "text.translation", kind: "interim", segmentId: "tr.textfield.0:3" })).toEqual([3]);
    // A new STT session (epoch 2) ends the open interims of the old one too.
    expect(supersededCaptionLines(open, { topic: "text.stt", kind: "interim", segmentId: "mic:2:1" })).toEqual([1, 5]);
    // A final that opens a line (the late final of a given-up segment) gives up nothing; nor do unkeyed updates.
    expect(supersededCaptionLines(open, { topic: "text.stt", kind: "final", segmentId: "mic:1:4" })).toEqual([]);
    expect(supersededCaptionLines(open, { topic: "text.stt", kind: "final", segmentId: "mic:1:7" })).toEqual([1]);
    expect(supersededCaptionLines(open, { topic: "text.stt", kind: "interim" })).toEqual([]);
    expect(supersededCaptionLines(open, { topic: "text.stt", kind: "interim", segmentId: "nocolon" })).toEqual([]);
  });

  it("lets a segment whose first event is its final end the given-up interims before it", () => {
    // `mic:1:6` was dropped after an interim; the next segment ("Okay.") was too short for a partial.
    const open: CaptionKeyed[] = [
      { topic: "text.stt", kind: "interim", segmentId: "mic:1:6", revision: 3 },
      { topic: "text.translation", kind: "interim", segmentId: "mic:1:6", revision: 2 },
      { topic: "text.stt", kind: "interim", segmentId: "odd:x", revision: 1 },
    ];
    expect(supersededCaptionLines(open, { topic: "text.stt", kind: "final", segmentId: "mic:1:9", revision: 2 })).toEqual([0]);
    expect(supersededCaptionLines(open, { topic: "text.translation", kind: "final", segmentId: "mic:1:9", revision: 1 })).toEqual([1]);
    expect(supersededCaptionLines(open, { topic: "text.stt", kind: "final", segmentId: "mic:2:1" })).toEqual([0]);
    // The late final of a segment before the open one, and ids that do not order, end nothing.
    expect(supersededCaptionLines(open, { topic: "text.stt", kind: "final", segmentId: "mic:1:5" })).toEqual([]);
    expect(supersededCaptionLines(open, { topic: "text.translation", kind: "final", segmentId: "mic:1:5.4" })).toEqual([]);
    expect(supersededCaptionLines(open, { topic: "text.stt", kind: "final", segmentId: "odd:y" })).toEqual([]);
  });

  it("detaches the lines on screen when a /client page reconnects to a possibly restarted host", () => {
    const lines: CaptionKeyed[] = [
      { topic: "text.stt", kind: "final", segmentId: "mic:1:1", revision: 12 },
      { topic: "text.stt", kind: "interim", segmentId: "mic:1:2", revision: 3 },
      { topic: "text.transform_source", kind: "final", segmentId: "mic:1:1", id: 0, revision: 12 },
      { topic: "text.textfield", kind: "interim" },
    ];
    // The old connection's interims leave; the finals stay but no longer pair.
    expect(detachCaptionLines(lines)).toEqual([1, 3]);
    const kept = [lines[0], lines[2]];
    expect(kept.map((l) => [l.segmentId, l.id, l.revision])).toEqual([[undefined, undefined, undefined], [undefined, undefined, undefined]]);
    // The restarted host's `mic:1:1` (and the transform's id 0) open lines of their own instead of being dropped as
    // stale (an interim after the old final, or a lower revision) or merged into the old caption.
    expect(findCaptionLine(kept, { topic: "text.stt", kind: "interim", segmentId: "mic:1:1", revision: 1 })).toBe(-1);
    expect(findCaptionLine(kept, { topic: "text.stt", kind: "final", segmentId: "mic:1:1", revision: 2 })).toBe(-1);
    expect(findCaptionLine(kept, { topic: "text.transform_source", kind: "interim", id: 0 })).toBe(-1);
    expect(findCaptionLine(kept, { topic: "text.stt", kind: "final" })).toBe(-1);
    // A detached CaptionLine behaves the same.
    const { line } = newLine("final", { segmentId: "mic:1:4", revision: 5 });
    line.update(upd("final", "done", { meta: { topic: TOPIC, kind: "final", segmentId: "mic:1:4", id: 3, revision: 5 } }));
    expect(detachCaptionLines([line])).toEqual([]);
    expect(findCaptionLine([line], { topic: TOPIC, kind: "interim", segmentId: "mic:1:4", revision: 1 })).toBe(-1);
    expect(findCaptionLine([line], { topic: TOPIC, kind: "interim", id: 3 })).toBe(-1);
  });

  it("keeps an AI transform line that waits for its reply (another id) but gives up one without a final", () => {
    const lines: CaptionKeyed[] = [
      // Segment A: given up without a final, so the transform's id stayed 4.
      { topic: "text.transform_source", kind: "interim", segmentId: "mic:1:1", id: 4, revision: 2 },
      // Segment B: final (id 5), shown as an interim with the original text until the AI reply.
      { topic: "text.transform_source", kind: "interim", segmentId: "mic:1:2", id: 5, revision: 8 },
    ];
    expect(supersededCaptionLines(lines, { topic: "text.transform_source", kind: "interim", segmentId: "mic:1:3", id: 6 })).toEqual([]);
    expect(supersededCaptionLines(lines, { topic: "text.transform_source", kind: "interim", segmentId: "mic:1:3", id: 4 })).toEqual([0]);
  });

  it("never lets a late event of an earlier segment end the segment being spoken", () => {
    const lines: CaptionKeyed[] = [
      { topic: "text.transform_source", kind: "interim", segmentId: "mic:1:8", id: 4, revision: 3 },
      { topic: "text.translation", kind: "interim", segmentId: "mic:1:8", revision: 2 },
      { topic: "text.stt", kind: "interim", segmentId: "odd:x", revision: 1 },
    ];
    // Segment 7 was given up; its late final is shown by the AI transform as an interim (same id as 8's interims).
    expect(supersededCaptionLines(lines, { topic: "text.transform_source", kind: "interim", segmentId: "mic:1:7", id: 4 })).toEqual([]);
    // The translation of a given-up segment's late final is a line of its own (`mic:1:7.<n>`).
    expect(supersededCaptionLines(lines, { topic: "text.translation", kind: "interim", segmentId: "mic:1:7.12" })).toEqual([]);
    expect(supersededCaptionLines(lines, { topic: "text.translation", kind: "interim", segmentId: "mic:1:9" })).toEqual([1]);
    // Ids that do not order: the older line on screen is given up.
    expect(supersededCaptionLines(lines, { topic: "text.stt", kind: "interim", segmentId: "odd:y" })).toEqual([2]);
  });
});

// --- CaptionLine ------------------------------------------------------------------------------------------------

describe("CaptionLine", () => {
  it("re-renders only the changed tail, keeping the shown nodes", () => {
    const { line, root, removed, replaced } = newLine();
    line.update(upd("interim", "the quick brown fox"));
    line.flush("*");
    line.finish();
    const first = root.children[0];
    expect(visible(root)).toBe("the quick brown fox ");

    expect(line.update(upd("interim", "the quick brown dog"))).toBe(16);
    expect(root.children[0]).toBe(first);
    expect(first.data).toBe("the quick brown ");
    expect(line.cursor).toBe(16);
    line.step("*", "char");
    expect(visible(root)).toBe("the quick brown d");
    line.flush("*");
    line.finish();
    expect(visible(root)).toBe("the quick brown dog ");
    expect(root.children[0]).toBe(first);
    expect(replaced).toEqual([]);
    expect(removed.every((n) => n.data === " ")).toBe(true);
  });

  it("appends to a growing line without touching what is shown", () => {
    const { line, root, removed } = newLine();
    line.update(upd("interim", "Hello"));
    while (line.step("*", "char"));
    line.finish();
    const first = root.children[0];
    line.update(upd("interim", "Hello world"));
    expect(line.cursor).toBe(5);
    line.step("*", "word");
    expect(visible(root)).toBe("Hello ");
    line.step("*", "word");
    expect(visible(root)).toBe("Hello world");
    expect(root.children).toEqual([first]);
    expect(removed.filter((n) => n.data !== " ")).toEqual([]);
  });

  it("cuts back to the divergence and shortens", () => {
    const { line, root } = newLine("final");
    line.update(upd("final", "I want to go"));
    line.flush("*");
    expect(line.update(upd("final", "I want two goats"))).toBe(8);
    expect(visible(root)).toBe("I want t");
    line.flush("*");
    expect(visible(root)).toBe("I want two goats");
    line.update(upd("final", "I want"));
    expect(visible(root)).toBe("I want");
    expect(line.pending).toBe(false);
  });

  it("keeps the interim look on the committed prefix and marks the tentative tail", () => {
    const { line, root } = newLine();
    line.update(upd("interim", "Hola a to", { stableLength: 7 }));
    line.flush("*");
    // The committed prefix may still be replaced by the final: the line keeps the interim colour.
    expect(root.className).toBe("interim");
    expect(markup(root)).toBe("Hola a {interim tentative:to}");
    const first = root.children[0];

    line.update(upd("interim", "Hola a todos, amigos", { stableLength: 14 }));
    line.flush("*");
    expect(root.className).toBe("interim");
    expect(markup(root)).toBe("Hola a todos, {interim tentative:amigos}");
    expect(root.children[0]).toBe(first);

    line.update(upd("final", "Hola a todos, amigos."));
    line.flush("*");
    line.finish();
    expect(root.className).toBe("");
    expect(markup(root)).toBe("Hola a todos, amigos. ");
    expect(root.children[0]).toBe(first);
    expect(root.children.length).toBe(2);
  });

  it("never renders a fully committed interim like a final", () => {
    // P4-C defaults (LocalAgreement on, tentative tail hidden): stableLength === value.length.
    const { line, root } = newLine();
    line.update(upd("interim", "I think that the", { stableLength: 16 }));
    line.flush("*");
    expect(root.className).toBe("interim");
    expect(markup(root)).toBe("I think that the");
    line.update(upd("final", "The thing is, I think that the plan works."));
    line.flush("*");
    expect(root.className).toBe("");
  });

  it("moves the split both ways and between the legacy and the split look", () => {
    const { line, root } = newLine();
    line.update(upd("interim", "Hola a todos"));
    line.flush("*");
    expect(root.className).toBe("interim");
    expect(markup(root)).toBe("Hola a todos");

    line.update(upd("interim", "Hola a todos", { stableLength: 12 }));
    expect(root.className).toBe("interim");
    expect(markup(root)).toBe("Hola a todos");

    line.update(upd("interim", "Hola a todos", { stableLength: 5 }));
    expect(root.className).toBe("interim");
    expect(markup(root)).toBe("Hola {interim tentative:a todos}");

    line.update(upd("interim", "Hola a todos"));
    expect(root.className).toBe("interim");
    expect(markup(root)).toBe("Hola a todos");
    expect(root.children.length).toBe(1);
  });

  it("styles a committed mask of an interim line as an interim mask", () => {
    const { line, root } = newLine();
    line.update(upd("interim", "oh f**k it", { stableLength: 8 }));
    line.flush("[x]");
    expect(markup(root)).toBe("oh {interim profanity:[x]} {interim tentative:it}");
    line.update(upd("final", "oh f**k it"));
    expect(markup(root)).toBe("oh {profanity:[x]} it");
  });

  it("completes a half-revealed mask when the rest is rendered at once", () => {
    const { line, root } = newLine("final");
    line.update(upd("final", "oh f**k off"));
    while (line.cursor < 3) line.step("[x]", "char");
    line.step("[x]", "char");
    line.flush("[x]");
    expect(markup(root)).toBe("oh {profanity:[x]} off");
    expect(root.children.filter((c) => c.kind === "span").length).toBe(1);
  });

  it("keeps the legacy look for interims without stableLength", () => {
    const { line, root } = newLine();
    line.update(upd("interim", "oh f**k"));
    line.flush("[x]");
    expect(root.className).toBe("interim");
    expect(markup(root)).toBe("oh {interim profanity:[x]}");
    line.update(upd("final", "oh f**k"));
    expect(root.className).toBe("");
    expect(markup(root)).toBe("oh {profanity:[x]}");
  });

  it("reveals a mask character by character and keeps it through an update", () => {
    const { line, root } = newLine("final");
    line.update(upd("final", "oh f**k"));
    while (line.cursor < 3) line.step("[x]", "char");
    line.step("[x]", "char");
    expect(markup(root)).toBe("oh {profanity:[}");
    line.update(upd("final", "oh f**king"));
    expect(markup(root)).toBe("oh {profanity:[}");
    line.step("[x]", "char");
    line.step("[x]", "char");
    expect(markup(root)).toBe("oh {profanity:[x]}");
    expect(line.cursor).toBe(10);
    expect(line.pending).toBe(false);
  });

  it("drops a half-revealed mask when its word changes", () => {
    const { line, root } = newLine("final");
    line.update(upd("final", "oh f**k"));
    while (line.cursor < 3) line.step("[x]", "char");
    line.step("[x]", "char");
    line.update(upd("final", "oh fine"));
    expect(markup(root)).toBe("oh ");
    line.flush("[x]");
    expect(markup(root)).toBe("oh fine");
  });

  it("never lets a word step run over a mask", () => {
    const { line, root } = newLine("final");
    line.update(upd("final", "hello,f**k"));
    line.step("[x]", "word");
    expect(visible(root)).toBe("hello,");
    line.step("[x]", "word");
    expect(markup(root)).toBe("hello,{profanity:[x]}");
  });

  it("keeps an emote image across updates", () => {
    const { line, root } = newLine("final");
    line.update(upd("final", "hi Kappa there", { emotes: { 3: "k.png", Kappa: "k.png" } }));
    line.flush("*");
    expect(markup(root)).toBe("hi <k.png> there");
    const img = root.children[1];
    line.update(upd("final", "hi Kappa there friend", { emotes: { 3: "k.png", Kappa: "k.png" } }));
    line.flush("*");
    expect(markup(root)).toBe("hi <k.png> there friend");
    expect(root.children[1]).toBe(img);
  });

  it("renders emotes by name in static mode", () => {
    const { line, root } = newLine("final");
    line.update(upd("final", "gg LUL", { emotes: { LUL: "l.png" }, mode: "names" }));
    line.flush("*");
    expect(markup(root)).toBe("gg <l.png>");
  });

  it("steps whole graphemes", () => {
    const { line, root } = newLine("final");
    line.update(upd("final", "a\u{1F600}b"));
    line.step("*", "char");
    line.step("*", "char");
    expect(visible(root)).toBe("a\u{1F600}");
  });

  it("separates finished lines with exactly one space", () => {
    const { line, root } = newLine("final");
    line.update(upd("final", "Hi"));
    line.flush("*");
    line.finish();
    line.finish();
    expect(visible(root)).toBe("Hi ");
    line.update(upd("final", "Hi there"));
    line.flush("*");
    line.finish();
    expect(visible(root)).toBe("Hi there ");
  });

  it("adopts pairing fields and keeps the highest revision", () => {
    const { line } = newLine("interim");
    line.update(upd("interim", "a", { meta: { topic: TOPIC, kind: "interim", segmentId: "s1", revision: 2 } }));
    line.update(upd("interim", "ab", { meta: { topic: TOPIC, kind: "interim", id: 4 } }));
    line.update(upd("final", "abc", { meta: { topic: TOPIC, kind: "final", segmentId: "s1", id: 9, revision: 3 } }));
    expect([line.segmentId, line.id, line.revision, line.kind]).toEqual(["s1", 4, 3, "final"]);
  });
});
