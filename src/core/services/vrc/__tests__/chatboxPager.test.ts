import { afterEach, describe, expect, it, vi } from "vitest";
import {
  CHATBOX_MAX_CHARS,
  CHATBOX_MIN_INTERVAL_MS,
  ChatboxPager,
  chatboxIntervalMs,
  graphemes,
  monotonicNow,
  normalizeChatboxText,
  paginateChatbox,
  truncateChatbox,
} from "../chatboxPager";
import { FakeClock } from "./fakeClock";

/** Joins pages back without their markers (and the spaces trimmed at the breaks). */
function unpage(pages: string[]): string {
  return pages.map((p) => p.replace(/ \(\d+\/\d+\)$/, "")).join(" ");
}

const LOREM =
  "The quick brown fox jumps over the lazy dog. Pack my box with five dozen liquor jugs! How vexingly quick daft " +
  "zebras jump? Sphinx of black quartz, judge my vow; the five boxing wizards jump quickly. Waltz, bad nymph, for " +
  "quick jigs vex. Glib jocks quiz nymph to vex dwarf.";

describe("paginateChatbox", () => {
  it("keeps short text as one page without a marker", () => {
    expect(paginateChatbox("Hello there!")).toEqual(["Hello there!"]);
    expect(paginateChatbox("   ")).toEqual([]);
    const exact = "x".repeat(CHATBOX_MAX_CHARS);
    expect(paginateChatbox(exact)).toEqual([exact]);
  });

  it("never exceeds 144 characters and never loses text (no truncation)", () => {
    for (const text of [LOREM, LOREM + " " + LOREM, LOREM.repeat(4)]) {
      const pages = paginateChatbox(text);
      expect(pages.length).toBeGreaterThan(1);
      for (const p of pages) expect(p.length).toBeLessThanOrEqual(CHATBOX_MAX_CHARS);
      expect(unpage(pages).replace(/\s+/g, " ")).toBe(normalizeChatboxText(text));
    }
  });

  it("numbers pages and breaks at sentence ends first", () => {
    const pages = paginateChatbox(LOREM);
    expect(pages.map((p) => p.match(/\((\d+)\/(\d+)\)$/)?.slice(1).join("/"))).toEqual(
      pages.map((_, i) => `${i + 1}/${pages.length}`),
    );
    // The first page ends after a sentence, not mid-word.
    expect(pages[0]).toMatch(/[.!?] \(1\/\d+\)$/);
  });

  it("falls back to clause punctuation, then spaces, then a hard cut", () => {
    const clauses = Array.from({ length: 12 }, (_, i) => `clause number ${i} goes here`).join(", ");
    const p1 = paginateChatbox(clauses);
    expect(p1[0]).toMatch(/, \(1\/\d+\)$|here \(1\/\d+\)$|here, \(1\/\d+\)$/);
    expect(p1[0].replace(/ \(\d+\/\d+\)$/, "")).toMatch(/,$/);

    const words = Array.from({ length: 60 }, (_, i) => `word${i}`).join(" ");
    for (const p of paginateChatbox(words)) expect(p.replace(/ \(\d+\/\d+\)$/, "")).toMatch(/^word\d+( word\d+)*$/);

    const solid = "a".repeat(400);
    const hard = paginateChatbox(solid);
    expect(unpage(hard).replace(/ /g, "")).toBe(solid);
    for (const p of hard) expect(p.length).toBeLessThanOrEqual(CHATBOX_MAX_CHARS);
  });

  it("splits Japanese after 。 and never inside a character", () => {
    const ja = "今日はとても良い天気ですね。".repeat(20);
    const pages = paginateChatbox(ja);
    for (const p of pages) {
      expect(p.length).toBeLessThanOrEqual(CHATBOX_MAX_CHARS);
      expect(p.replace(/ \(\d+\/\d+\)$/, "")).toMatch(/。$/);
    }
    expect(pages.map((p) => p.replace(/ \(\d+\/\d+\)$/, "")).join("")).toBe(ja);
  });

  it("keeps emoji and surrogate pairs whole", () => {
    const text = "👍🏽".repeat(100); // 4 UTF-16 units each, one grapheme
    const pages = paginateChatbox(text);
    for (const p of pages) {
      const body = p.replace(/ \(\d+\/\d+\)$/, "");
      expect(body.length).toBeLessThanOrEqual(CHATBOX_MAX_CHARS);
      expect(graphemes(body).every((g) => g === "👍🏽")).toBe(true);
    }
    expect(pages.map((p) => p.replace(/ \(\d+\/\d+\)$/, "")).join("")).toBe(text);
  });

  it("reserves room for two-digit page markers", () => {
    const pages = paginateChatbox(Array.from({ length: 400 }, (_, i) => `w${i}`).join(" "));
    expect(pages.length).toBeGreaterThanOrEqual(10);
    for (const p of pages) expect(p.length).toBeLessThanOrEqual(CHATBOX_MAX_CHARS);
    expect(pages.at(-1)).toMatch(new RegExp(`\\(${pages.length}/${pages.length}\\)$`));
  });
});

describe("truncateChatbox", () => {
  it("cuts at a word boundary with an ellipsis, within the limit", () => {
    const out = truncateChatbox(LOREM);
    expect(out.length).toBeLessThanOrEqual(CHATBOX_MAX_CHARS);
    expect(out.endsWith("…")).toBe(true);
    expect(LOREM.startsWith(out.slice(0, -1))).toBe(true);
    expect(out.slice(-2, -1)).not.toBe(" ");
    expect(truncateChatbox("short")).toBe("short");
  });

  it("never splits a character", () => {
    const out = truncateChatbox("😀".repeat(100));
    expect(out.length).toBeLessThanOrEqual(CHATBOX_MAX_CHARS);
    expect(graphemes(out.slice(0, -1)).every((g) => g === "😀")).toBe(true);
  });
});

describe("normalizeChatboxText / chatboxIntervalMs", () => {
  it("collapses spaces, keeps line breaks, drops control characters", () => {
    expect(normalizeChatboxText("  a \t b\r\n c\u0007d  ")).toBe("a b\nc d");
  });

  it("clamps the interval to VRChat's safe range", () => {
    expect(chatboxIntervalMs("1500")).toBe(1500);
    expect(chatboxIntervalMs("200")).toBe(CHATBOX_MIN_INTERVAL_MS);
    expect(chatboxIntervalMs("999999")).toBe(10_000);
    expect(chatboxIntervalMs("abc")).toBe(1500);
  });
});

function pager(opts: { paginate?: boolean; interval?: number; notify?: boolean; maxQueue?: number } = {}) {
  const clock = new FakeClock();
  const sent: { text: string; notify: boolean; at: number }[] = [];
  const drops: number[] = [];
  let truncations = 0;
  const p = new ChatboxPager({
    send: (text, notify) => sent.push({ text, notify, at: clock.now() }),
    paginate: () => opts.paginate ?? true,
    intervalMs: () => opts.interval ?? 1500,
    notify: () => opts.notify ?? true,
    clock,
    maxQueue: opts.maxQueue,
    onDrop: (n) => drops.push(n),
    onTruncate: () => truncations++,
  });
  return { p, clock, sent, drops, truncations: () => truncations };
}

describe("ChatboxPager", () => {
  it("sends at once when idle and paces the rest at the interval (VRChat drops faster messages)", () => {
    const h = pager();
    h.p.push("First.");
    expect(h.sent.map((s) => s.text)).toEqual(["First."]);
    h.clock.advance(500);
    h.p.push("Second, a much longer sentence that does not fit together with anything else waiting in the queue, really.");
    expect(h.sent).toHaveLength(1);
    h.clock.advance(999);
    expect(h.sent).toHaveLength(1);
    h.clock.advance(1);
    expect(h.sent).toHaveLength(2);
    expect(h.sent[1].at - h.sent[0].at).toBe(1500);
  });

  it("never sends closer than 1.3 s, whatever the setting says", () => {
    const h = pager({ interval: 100 });
    h.p.push(LOREM);
    h.clock.advance(20_000);
    const gaps = h.sent.slice(1).map((s, i) => s.at - h.sent[i].at);
    expect(gaps.every((g) => g >= CHATBOX_MIN_INTERVAL_MS)).toBe(true);
  });

  it("sends every page of a long message in order, sound only on the first", () => {
    const h = pager();
    h.p.push(LOREM);
    h.clock.advance(60_000);
    expect(h.sent.length).toBeGreaterThan(1);
    expect(unpage(h.sent.map((s) => s.text))).toBe(normalizeChatboxText(LOREM));
    expect(h.sent.map((s) => s.notify)).toEqual(h.sent.map((_, i) => i === 0));
  });

  it("merges short messages that are still waiting into one", () => {
    const h = pager();
    h.p.push("One.");
    h.p.push("Two.");
    h.p.push("Three.");
    expect(h.p.pending).toBe(1);
    h.clock.advance(1500);
    expect(h.sent.map((s) => s.text)).toEqual(["One.", "Two. Three."]);
  });

  it("does not merge into pages of a paginated message or past the limit", () => {
    const h = pager();
    h.p.push("Busy.");
    h.p.push(LOREM);
    const pages = h.p.pending;
    h.p.push("Tail.");
    expect(h.p.pending).toBe(pages + 1);
    // "Tail." + 140 characters is over 144: a message of its own.
    h.p.push("y".repeat(140));
    expect(h.p.pending).toBe(pages + 2);
  });

  it("cuts instead of paginating when pagination is off, and reports the cut", () => {
    const h = pager({ paginate: false });
    h.p.push("Short enough.");
    expect(h.truncations()).toBe(0);
    h.clock.advance(1500);
    h.p.push(LOREM);
    expect(h.sent).toHaveLength(2);
    expect(h.sent[1].text.length).toBeLessThanOrEqual(CHATBOX_MAX_CHARS);
    expect(h.sent[1].text.endsWith("…")).toBe(true);
    expect(h.truncations()).toBe(1);
    // Paginated text is never cut, so never reported.
    const paged = pager();
    paged.p.push(LOREM);
    expect(paged.truncations()).toBe(0);
  });

  it("respects the notification sound setting", () => {
    const h = pager({ notify: false });
    h.p.push("Quiet.");
    expect(h.sent[0].notify).toBe(false);
  });

  it("drops the oldest waiting pages beyond the backlog cap and reports it", () => {
    const h = pager({ maxQueue: 3 });
    h.p.push("Sent at once.");
    h.p.push(LOREM.repeat(2));
    expect(h.p.pending).toBe(3);
    expect(h.drops.reduce((a, b) => a + b, 0)).toBeGreaterThan(0);
  });

  it("cancel drops what waits but keeps pacing the next message", () => {
    const h = pager();
    h.p.push("A.");
    h.p.push(LOREM);
    h.p.cancel();
    expect(h.p.pending).toBe(0);
    expect(h.clock.pending).toBe(0);
    h.clock.advance(100);
    h.p.push("B.");
    expect(h.sent.map((s) => s.text)).toEqual(["A."]);
    h.clock.advance(1400);
    expect(h.sent.map((s) => s.text)).toEqual(["A.", "B."]);
  });
});

describe("ChatboxPager default clock", () => {
  afterEach(() => {
    vi.useRealTimers();
    vi.restoreAllMocks();
  });

  it("is monotonic (performance.now), not the wall clock", () => {
    vi.spyOn(Date, "now").mockReturnValue(0);
    vi.spyOn(performance, "now").mockReturnValue(1234.5);
    expect(monotonicNow()).toBe(1234.5);
  });

  it("keeps pacing when the system time steps back (Windows time sync)", () => {
    vi.useFakeTimers({ toFake: ["setTimeout", "clearTimeout", "Date", "performance"] });
    const sent: string[] = [];
    const p = new ChatboxPager({
      send: (text) => sent.push(text),
      paginate: () => true,
      intervalMs: () => 1500,
      notify: () => true,
    });
    p.push("Before the clock change.");
    expect(sent).toHaveLength(1);
    // The wall clock jumps ten minutes back; the next message must still go out one interval later.
    vi.setSystemTime(Date.now() - 600_000);
    p.push("After it, and long enough not to be merged into anything that is still waiting in the queue.");
    vi.advanceTimersByTime(1500);
    expect(sent).toHaveLength(2);
  });
});
