import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import { TextEvent, TextEventType } from "@/types";

const sendOsc = vi.fn();
vi.mock("../targets/utils", () => ({ sendOsc: (...args: unknown[]) => sendOsc(...args) }));
const pushSystemLog = vi.fn();
vi.mock("@/core/services/systemLog", () => ({ pushSystemLog: (...args: unknown[]) => pushSystemLog(...args) }));

const { default: VRC_TextboxTarget } = await import("../targets/textbox");

type TextboxState = {
  indicator: boolean;
  playNotificationSound: boolean;
  paginate: boolean;
  pageIntervalMs: string;
  commitOnTurnEnd: boolean;
  turnHoldMs: string;
};

let textbox: TextboxState;

function target(deps: { turnGate?: boolean; tags?: readonly string[] } = {}) {
  return new VRC_TextboxTarget({
    turnGate: () => deps.turnGate ?? true,
    tagsFor: () => deps.tags,
  });
}

const final = (value: string, extra: Partial<TextEvent> = {}): TextEvent => ({
  type: TextEventType.final,
  value,
  emotes: {},
  ...extra,
});

/** `/chatbox/input` texts sent so far. */
const chatbox = () => sendOsc.mock.calls.filter(([path]) => path === "/chatbox/input").map(([, args]) => args[0].String);
const typing = () => sendOsc.mock.calls.filter(([path]) => path === "/chatbox/typing").map(([, args]) => args[0].Bool);

describe("VRChat chatbox target", () => {
  beforeEach(() => {
    vi.useFakeTimers();
    sendOsc.mockReset();
    pushSystemLog.mockReset();
    textbox = {
      indicator: true,
      playNotificationSound: true,
      paginate: false,
      pageIntervalMs: "1500",
      commitOnTurnEnd: false,
      turnHoldMs: "2500",
    };
    vi.stubGlobal("window", { ApiServer: { state: { services: { vrc: { data: { textbox } } } } } });
  });

  afterEach(() => {
    vi.useRealTimers();
    vi.unstubAllGlobals();
  });

  it("sends a final at once, clearing the typing indicator first", () => {
    const t = target();
    t.pushFinal("Hello there.", final("Hello there."));
    expect(sendOsc.mock.calls).toEqual([
      ["/chatbox/typing", [{ Bool: false }]],
      ["/chatbox/input", [{ String: "Hello there." }, { Bool: true }]],
    ]);
  });

  it("never shows STT's cancel marker", () => {
    const t = target();
    t.pushFinal("[...]", final("[...]"));
    expect(chatbox()).toEqual([]);
  });

  it("turns the typing indicator off when the sentence is cancelled (mute or stop mid-sentence)", () => {
    const t = target();
    t.pushInterim("I was say", final("I was say", { type: TextEventType.interim }));
    vi.advanceTimersByTime(200);
    // A second interim inside the throttle window leaves a trailing "on" pending.
    t.pushInterim("I was saying", final("I was saying", { type: TextEventType.interim }));
    expect(typing()).toEqual([true]);
    t.pushFinal("[...]", final("[...]"));
    expect(typing().at(-1)).toBe(false);
    // The pending trailing "on" was dropped: the bubble stays off.
    vi.advanceTimersByTime(5_000);
    expect(typing()).toEqual([true, false]);
    expect(chatbox()).toEqual([]);
  });

  it("a cancelled held turn is sent and the indicator goes off", () => {
    textbox.commitOnTurnEnd = true;
    const t = target();
    t.pushFinal("I think that", final("I think that", { turnEnd: false }));
    expect(typing().at(-1)).toBe(true);
    t.pushFinal("[...]", final("[...]"));
    expect(chatbox()).toEqual(["I think that"]);
    expect(typing().at(-1)).toBe(false);
  });

  it("cancel turns off a typing indicator it turned on", () => {
    const t = target();
    t.cancel();
    expect(typing()).toEqual([]);
    t.pushInterim("Hel", final("Hel", { type: TextEventType.interim }));
    t.cancel();
    expect(typing()).toEqual([true, false]);
    vi.advanceTimersByTime(5_000);
    expect(typing()).toEqual([true, false]);
  });

  it("says once in the system log that long text was cut while pagination is off", () => {
    const t = target();
    const long = "word ".repeat(60).trim();
    t.pushFinal(long, final(long));
    vi.advanceTimersByTime(2_000);
    t.pushFinal(long, final(long));
    vi.advanceTimersByTime(2_000);
    expect(chatbox()).toHaveLength(2);
    expect(chatbox().every((m: string) => m.length <= 144 && m.endsWith("…"))).toBe(true);
    expect(pushSystemLog).toHaveBeenCalledTimes(1);
    expect(pushSystemLog.mock.calls[0][1]).toContain("Split long text into pages");
  });

  it("paginates long text without losing any of it, paced and silent after page one", () => {
    textbox.paginate = true;
    textbox.indicator = false;
    const long = Array.from({ length: 30 }, (_, i) => `Sentence number ${i} is here.`).join(" ");
    target().pushFinal(long, final(long));
    vi.advanceTimersByTime(60_000);
    const pages = chatbox();
    expect(pages.length).toBeGreaterThan(1);
    expect(pages.map((p: string) => p.replace(/ \(\d+\/\d+\)$/, "")).join(" ")).toBe(long);
    const inputs = sendOsc.mock.calls.filter(([path]) => path === "/chatbox/input").map(([, args]) => args);
    expect(inputs[0]).toHaveLength(2);
    expect(inputs.slice(1).every((a) => a[2]?.Bool === false)).toBe(true);
  });

  it("holds an unfinished turn (typing shown) and sends it joined when the turn ends", () => {
    textbox.commitOnTurnEnd = true;
    const t = target();
    t.pushFinal("I think that", final("I think that", { turnEnd: false }));
    expect(chatbox()).toEqual([]);
    expect(typing()).toContain(true);
    t.pushFinal("we should go.", final("we should go.", { turnEnd: true }));
    expect(chatbox()).toEqual(["I think that we should go."]);
    expect(typing().at(-1)).toBe(false);
  });

  it("does not hold a second time where the source is already turn-committed", () => {
    textbox.commitOnTurnEnd = true;
    const t = target({ turnGate: false });
    t.pushFinal("Translated line", final("Translated line", { turnEnd: false }));
    expect(chatbox()).toEqual(["Translated line"]);
  });

  it("commits a held turn after the hold time", () => {
    textbox.commitOnTurnEnd = true;
    textbox.turnHoldMs = "1000";
    const t = target();
    t.pushFinal("Trailing off", final("Trailing off", { turnEnd: false }));
    vi.advanceTimersByTime(999);
    expect(chatbox()).toEqual([]);
    vi.advanceTimersByTime(1);
    expect(chatbox()).toEqual(["Trailing off"]);
  });

  it("typed text is sent without waiting for the spoken turn", () => {
    textbox.commitOnTurnEnd = true;
    const t = target();
    t.pushFinal("Spoken", final("Spoken", { turnEnd: false }));
    t.pushFinal("typed", final("typed"), { typed: true });
    vi.advanceTimersByTime(1_500);
    expect(chatbox()).toEqual(["Spoken", "typed"]);
  });

  it("puts tag emoji in front and honours the notification sound setting", () => {
    textbox.playNotificationSound = false;
    target({ tags: ["emotion:happy"] }).pushFinal("Nice!", final("Nice!"));
    const input = sendOsc.mock.calls.find(([path]) => path === "/chatbox/input");
    expect(input?.[1]).toEqual([{ String: "😊 Nice!" }, { Bool: true }, { Bool: false }]);
  });

  it("cancel drops held and queued text", () => {
    textbox.commitOnTurnEnd = true;
    const t = target();
    t.pushFinal("Sent.", final("Sent."));
    t.pushFinal("Queued sentence that waits for the pacing interval to pass first.", final("Queued sentence that waits for the pacing interval to pass first."));
    t.pushFinal("Held", final("Held", { turnEnd: false }));
    t.cancel();
    vi.advanceTimersByTime(10_000);
    expect(chatbox()).toEqual(["Sent."]);
  });
});
