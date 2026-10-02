import { describe, expect, it } from "vitest";
import { TextEvent, TextEventType } from "@/types";
import { TurnCommitGate } from "../turnCommit";
import { FakeClock } from "./fakeClock";

function harness(opts: { enabled?: boolean; holdMs?: number; maxChars?: number } = {}) {
  const clock = new FakeClock();
  const out: TextEvent[] = [];
  let enabled = opts.enabled ?? true;
  const gate = new TurnCommitGate({
    enabled: () => enabled,
    holdMs: () => opts.holdMs ?? 2_500,
    maxChars: () => opts.maxChars ?? 280,
    commit: (e) => out.push(e),
    clock,
  });
  return {
    gate,
    clock,
    out,
    values: () => out.map((e) => e.value),
    setEnabled: (v: boolean) => (enabled = v),
  };
}

const final = (value: string, extra: Partial<TextEvent> = {}): TextEvent => ({
  type: TextEventType.final,
  value,
  emotes: {},
  ...extra,
});
const interim = (value: string): TextEvent => ({ type: TextEventType.interim, value, emotes: {} });

describe("TurnCommitGate", () => {
  it("passes finals straight through when off or without a verdict", () => {
    const off = harness({ enabled: false });
    off.gate.final(final("Not finished", { turnEnd: false }));
    expect(off.values()).toEqual(["Not finished"]);

    const on = harness();
    on.gate.final(final("Cloud STT has no verdict."));
    on.gate.final(final("Done.", { turnEnd: true }));
    expect(on.values()).toEqual(["Cloud STT has no verdict.", "Done."]);
    expect(on.clock.pending).toBe(0);
  });

  it("holds an unfinished turn and joins it with the sentence that ends it", () => {
    const h = harness();
    h.gate.final(final("I think that", { turnEnd: false, segmentId: "mic:1:1", tags: ["emotion:happy"] }));
    expect(h.out).toHaveLength(0);
    expect(h.gate.holding).toBe(true);
    h.gate.final(final("we should go.", { turnEnd: true, segmentId: "mic:1:2", tags: ["event:laughter"] }));
    expect(h.values()).toEqual(["I think that we should go."]);
    expect(h.out[0].segmentId).toBe("mic:1:1");
    expect(h.out[0].tags).toEqual(["emotion:happy", "event:laughter"]);
    expect(h.out[0].turnEnd).toBe(true);
    expect(h.gate.holding).toBe(false);
  });

  it("commits after the hold time without speech; interims keep it waiting", () => {
    const h = harness({ holdMs: 2_000 });
    h.gate.final(final("So basically", { turnEnd: false }));
    h.clock.advance(1_500);
    h.gate.interim(interim("um"));
    h.clock.advance(1_500);
    expect(h.out).toHaveLength(0);
    h.clock.advance(600);
    expect(h.values()).toEqual(["So basically"]);
  });

  it("commits what is held before it would grow past the limit", () => {
    const h = harness({ maxChars: 30 });
    h.gate.final(final("This is a long opening", { turnEnd: false }));
    h.gate.final(final("clause that keeps going", { turnEnd: false }));
    expect(h.values()).toEqual(["This is a long opening"]);
    expect(h.gate.heldText).toBe("clause that keeps going");
  });

  it("joins Japanese without spaces", () => {
    const h = harness();
    h.gate.final(final("今日は", { turnEnd: false, lang: "ja" }));
    h.gate.final(final("晴れです。", { turnEnd: true, lang: "ja" }));
    expect(h.values()).toEqual(["今日は晴れです。"]);
  });

  it("never commits STT's cancel marker, but releases what is held", () => {
    const h = harness();
    h.gate.final(final("[...]"));
    expect(h.out).toHaveLength(0);
    h.gate.final(final("Held", { turnEnd: false }));
    h.gate.final(final("[...]"));
    expect(h.values()).toEqual(["Held"]);
  });

  it("sends typed text at once, after anything held", () => {
    const h = harness();
    h.gate.final(final("Spoken", { turnEnd: false }));
    h.gate.passThrough(final("typed message"));
    expect(h.values()).toEqual(["Spoken", "typed message"]);
  });

  it("switching off releases a held turn with the next final; reset drops it", () => {
    const h = harness();
    h.gate.final(final("Held", { turnEnd: false }));
    h.setEnabled(false);
    h.gate.final(final("next", { turnEnd: false }));
    expect(h.values()).toEqual(["Held", "next"]);

    const r = harness();
    r.gate.final(final("Dropped", { turnEnd: false }));
    r.gate.reset();
    r.clock.advance(10_000);
    expect(r.out).toHaveLength(0);
  });
});
