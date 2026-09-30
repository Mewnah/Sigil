import { describe, expect, it } from "vitest";
import { TextEvent, TextEventType } from "@/types";
import { CANCEL_MARKER } from "../sequencer";
import { joinFragments, TrackerClock, UtteranceTracker } from "../tracker";

/** Manual clock: timers fire only on `advance`. */
function manualClock() {
  let now = 1_000;
  let seq = 0;
  const timers = new Map<number, { at: number; fn: () => void }>();
  const clock: TrackerClock = {
    now: () => now,
    setTimeout: (fn, ms) => {
      const id = ++seq;
      timers.set(id, { at: now + ms, fn });
      return id;
    },
    clearTimeout: (h) => {
      timers.delete(h as number);
    },
  };
  const advance = (ms: number) => {
    now += ms;
    for (const [id, t] of [...timers].sort((a, b) => a[1].at - b[1].at)) {
      if (t.at <= now && timers.has(id)) {
        timers.delete(id);
        t.fn();
      }
    }
  };
  return { clock, advance, pending: () => timers.size };
}

function harness(opts: { turnEndCommit?: boolean; holdMs?: number; maxHeldChars?: number } = {}) {
  const out: string[] = [];
  const ids: string[] = [];
  const { clock, advance, pending } = manualClock();
  const tracker = new UtteranceTracker({
    idPrefix: "tr.stt.1",
    turnEndCommit: () => opts.turnEndCommit ?? true,
    holdMs: () => opts.holdMs ?? 2_000,
    maxHeldChars: opts.maxHeldChars,
    clock,
    sink: {
      interim: (ev) => {
        out.push(`interim:${ev.value}`);
        ids.push(ev.line.id);
      },
      final: (ev) => {
        out.push(`final:${ev.value}`);
        ids.push(ev.line.id);
      },
      cancel: (ev) => {
        out.push(`cancel:${ev.value}`);
        ids.push(ev.line.id);
      },
      abandon: (line) => {
        out.push("abandon");
        ids.push(line.id);
      },
    },
  });
  const ev = (type: TextEventType, value: string, meta: Partial<TextEvent> = {}): TextEvent => ({ type, value, emotes: {}, ...meta });
  const interim = (value: string, meta: Partial<TextEvent> = {}) => tracker.push(ev(TextEventType.interim, value, meta));
  const final = (value: string, meta: Partial<TextEvent> = {}) => tracker.push(ev(TextEventType.final, value, meta));
  return { tracker, out, ids, interim, final, advance, pending };
}

const seg = (n: number, revision: number, extra: Partial<TextEvent> = {}) => ({ segmentId: `mic:1:${n}`, revision, ...extra });

describe("UtteranceTracker with segment ids", () => {
  it("forwards interims then the final, keyed by the segment id", () => {
    const h = harness();
    h.interim("hello", seg(1, 0));
    h.interim("hello wor", seg(1, 1));
    h.final("Hello world.", seg(1, 2));
    expect(h.out).toEqual(["interim:hello", "interim:hello wor", "final:Hello world."]);
    expect(new Set(h.ids)).toEqual(new Set(["mic:1:1"]));
  });

  it("drops interims after the segment's final and out-of-order revisions", () => {
    const h = harness();
    h.interim("one two", seg(1, 3));
    h.interim("one", seg(1, 2)); // older revision
    h.final("One two three.", seg(1, 4));
    h.interim("one two three", seg(1, 5)); // interim after its final
    expect(h.out).toEqual(["interim:one two", "final:One two three."]);
  });

  it("ignores a repeated final and translates a different late one as its own line", () => {
    const h = harness();
    h.final("Same text.", seg(1, 1));
    h.final("same text", seg(1, 2));
    expect(h.out).toEqual(["final:Same text."]);
    h.final("Something else.", seg(1, 3));
    expect(h.out).toEqual(["final:Same text.", "final:Something else."]);
    expect(h.ids[0]).not.toBe(h.ids[1]);
  });

  it("abandons a line whose segment was given up when the next segment starts", () => {
    const h = harness();
    h.interim("lost words", seg(1, 0));
    h.interim("new", seg(2, 0));
    h.final("New sentence.", seg(2, 1));
    expect(h.out).toEqual(["interim:lost words", "abandon", "interim:new", "final:New sentence."]);
    expect(h.ids).toEqual(["mic:1:1", "mic:1:1", "mic:1:2", "mic:1:2"]);
  });

  it("still translates the late final of a given-up segment (finals are never dropped)", () => {
    const h = harness();
    h.interim("first", seg(1, 0));
    h.interim("second", seg(2, 0));
    h.final("Second.", seg(2, 1));
    h.final("First sentence.", seg(1, 2)); // published late by the stall guard
    expect(h.out).toEqual(["interim:first", "abandon", "interim:second", "final:Second.", "final:First sentence."]);
  });

  it("commits a late final held ahead of a partial whose segment is then given up", () => {
    for (const turnEndCommit of [true, false]) {
      const h = harness({ turnEndCommit, holdMs: 1_000 });
      h.interim("first", seg(1, 0));
      h.interim("noise", seg(2, 0)); // segment 1 given up by the stall guard
      h.final("First sentence.", seg(1, 1)); // its late final joins the open line ahead of the partial
      expect(h.out.at(-1)).toBe("interim:First sentence. noise");
      expect(h.pending()).toBe(1); // the held final has a timer
      h.advance(60_000); // the VAD drops segment 2: nothing more comes
      expect(h.out).toContain("final:First sentence.");
      expect(h.out.filter((o) => o.startsWith("final:"))).toEqual(["final:First sentence."]);
    }
  });

  it("keeps holding a late final while the partial after it is still being spoken", () => {
    const h = harness({ holdMs: 1_000 });
    h.interim("first", seg(1, 0));
    h.interim("second", seg(2, 0));
    h.final("First.", seg(1, 1));
    h.advance(900);
    h.interim("second part", seg(2, 1)); // re-arms the hold
    h.advance(900);
    h.final("Second part.", seg(2, 2));
    expect(h.out.filter((o) => o.startsWith("final:"))).toEqual(["final:First. Second part."]);
    expect(h.pending()).toBe(0);
  });

  it("closes the open line on the cancel marker", () => {
    const h = harness();
    h.interim("half a", seg(1, 0));
    h.final(CANCEL_MARKER, seg(1, 1));
    h.interim("half a sentence", seg(1, 2)); // the engine keeps going: dropped
    expect(h.out).toEqual(["interim:half a", `cancel:${CANCEL_MARKER}`]);
  });
});

describe("UtteranceTracker without segment ids", () => {
  it("drops an interim that repeats the start of the final just closed", () => {
    const h = harness();
    h.interim("yes i think");
    h.final("Yes, I think so.");
    h.interim("yes i think");
    expect(h.out).toEqual(["interim:yes i think", "final:Yes, I think so."]);
    h.advance(2_000);
    h.interim("yes i think");
    expect(h.out.at(-1)).toBe("interim:yes i think");
  });

  it("generates distinct line ids", () => {
    const h = harness();
    h.final("one");
    h.final("two");
    expect(h.ids[0]).toMatch(/^tr\.stt\.1:\d+$/);
    expect(h.ids[0]).not.toBe(h.ids[1]);
  });
});

describe("Smart Turn commit (turnEnd)", () => {
  it("holds an unfinished turn and merges it with the next segment into one final", () => {
    const h = harness();
    h.interim("I think that", seg(1, 0));
    h.final("I think that", seg(1, 1, { turnEnd: false }));
    h.interim("we should", seg(2, 0));
    h.final("we should go.", seg(2, 1, { turnEnd: true }));
    expect(h.out).toEqual([
      "interim:I think that",
      "interim:I think that",
      "interim:I think that we should",
      "final:I think that we should go.",
    ]);
    expect(new Set(h.ids)).toEqual(new Set(["mic:1:1"]));
    expect(h.pending()).toBe(0);
  });

  it("commits a held line when the turn does not continue within holdMs", () => {
    const h = harness({ holdMs: 1_500 });
    h.final("So basically", seg(1, 1, { turnEnd: false }));
    h.advance(1_499);
    expect(h.out).toEqual(["interim:So basically"]);
    h.advance(1);
    expect(h.out).toEqual(["interim:So basically", "final:So basically"]);
  });

  it("restarts the hold while the speaker keeps talking and carries a running partial to a new line", () => {
    const h = harness({ holdMs: 1_000 });
    h.final("Well", seg(1, 1, { turnEnd: false }));
    h.advance(900);
    h.interim("you know", seg(2, 0));
    h.advance(900);
    expect(h.out.at(-1)).toBe("interim:Well you know");
    h.advance(200); // no event for 1 s: commit what is final, the partial continues on its own line
    expect(h.out.slice(-2)).toEqual(["final:Well", "interim:you know"]);
    h.final("you know it.", seg(2, 1));
    expect(h.out.at(-1)).toBe("final:you know it.");
    expect(h.ids.at(-1)).toBe("mic:1:2");
  });

  it("commits at once when the line grows past maxHeldChars", () => {
    const h = harness({ maxHeldChars: 20 });
    h.final("This is a long opening clause", seg(1, 1, { turnEnd: false }));
    expect(h.out).toEqual(["final:This is a long opening clause"]);
  });

  it("commits finals at once without a verdict or with the option off", () => {
    const h = harness();
    h.final("No verdict", seg(1, 1));
    expect(h.out).toEqual(["final:No verdict"]);
    const off = harness({ turnEndCommit: false });
    off.final("Not finished", seg(1, 1, { turnEnd: false }));
    expect(off.out).toEqual(["final:Not finished"]);
  });

  it("commits the held part when the cancel marker closes the unfinished tail", () => {
    const h = harness();
    h.final("Held part", seg(1, 1, { turnEnd: false }));
    h.interim("unfinished", seg(2, 0));
    h.final(CANCEL_MARKER, seg(2, 1));
    expect(h.out.at(-1)).toBe("final:Held part");
    expect(h.pending()).toBe(0);
  });

  it("keeps a held line when a later segment is abandoned", () => {
    const h = harness();
    h.final("Held", seg(1, 1, { turnEnd: false }));
    h.interim("dropped", seg(2, 0));
    h.interim("next", seg(3, 0));
    h.final("next one.", seg(3, 1, { turnEnd: true }));
    expect(h.out).toEqual(["interim:Held", "interim:Held dropped", "interim:Held next", "final:Held next one."]);
  });

  it("joins Japanese fragments without a space", () => {
    const h = harness();
    h.final("今日は", seg(1, 1, { turnEnd: false, lang: "ja" }));
    h.final("晴れです。", seg(2, 1, { turnEnd: true, lang: "ja" }));
    expect(h.out.at(-1)).toBe("final:今日は晴れです。");
  });

  it("reset drops a held line without output", () => {
    const h = harness();
    h.final("Held", seg(1, 1, { turnEnd: false }));
    h.tracker.reset();
    h.advance(10_000);
    expect(h.out).toEqual(["interim:Held"]);
  });
});

describe("joinFragments", () => {
  it("spaces words but not unspaced scripts", () => {
    expect(joinFragments("I think", "so")).toBe("I think so");
    expect(joinFragments("今日は", "晴れ")).toBe("今日は晴れ");
    expect(joinFragments("今日は", "晴れ", "ja")).toBe("今日は晴れ");
    expect(joinFragments("", "x")).toBe("x");
    expect(joinFragments("x ", "  ")).toBe("x");
  });
});
