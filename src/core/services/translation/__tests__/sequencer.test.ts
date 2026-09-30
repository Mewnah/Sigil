import { afterEach, describe, expect, it, vi } from "vitest";
import {
  CANCEL_MARKER,
  FINAL_TIMEOUT_MAX_MS,
  FINAL_TIMEOUT_MS,
  finalDeadlineMs,
  isCancelMarker,
  normalizeForMatch,
  SequencerJob,
  SequencerTimeoutError,
  TranslationSequencer,
} from "../sequencer";

type Ev = { value: string };

interface Pending {
  job: SequencerJob<Ev>;
  resolve: (v: string) => void;
  reject: (e: unknown) => void;
}

type Deadline = number | ((e: Ev) => number);

/** A sequencer whose jobs settle only when the test says so. */
function harness(options: { staleInterimMs?: number; now?: () => number; timeoutMs?: { interim?: Deadline; final?: Deadline } } = {}) {
  const jobs: Pending[] = [];
  const published: string[] = [];
  const errors: { job: SequencerJob<Ev>; error: unknown }[] = [];
  const seq = new TranslationSequencer<Ev>({
    run: (job) =>
      new Promise<string>((resolve, reject) => {
        jobs.push({ job, resolve, reject });
      }),
    publish: (job, value) => {
      published.push(`${job.stream}:${job.kind}:${value}`);
    },
    onError: (job, error) => errors.push({ job, error }),
    ...options,
  });
  const find = (text: string, kind?: "interim" | "final") => {
    const p = jobs.find((j) => j.job.event.value === text && (!kind || j.job.kind === kind));
    if (!p) throw new Error(`no job for "${text}"`);
    return p;
  };
  const flush = () => new Promise<void>((r) => setTimeout(r, 0));
  return { seq, jobs, published, errors, find, flush };
}

afterEach(() => {
  vi.useRealTimers();
});

describe("TranslationSequencer", () => {
  it("publishes a final while an interim of the next utterance is in flight (acceptance)", async () => {
    const h = harness();
    h.seq.pushInterim("stt", { value: "hello" });
    h.find("hello").resolve("hola");
    await h.flush();
    h.seq.pushFinal("stt", { value: "hello world" });
    h.seq.pushInterim("stt", { value: "next" }); // next utterance starts while the final is in flight
    expect(h.jobs.filter((j) => j.job.kind === "final")).toHaveLength(1);

    h.find("hello world", "final").resolve("hola mundo");
    await h.flush();
    expect(h.published).toEqual(["stt:interim:hola", "stt:final:hola mundo"]);

    h.find("next").resolve("siguiente");
    await h.flush();
    expect(h.published).toEqual(["stt:interim:hola", "stt:final:hola mundo", "stt:interim:siguiente"]);
    expect(h.errors).toEqual([]);
  });

  it("publishes the final even when the next utterance's interim answers first, and keeps order", async () => {
    const h = harness();
    h.seq.pushFinal("stt", { value: "one" });
    h.seq.pushInterim("stt", { value: "tw" });
    h.find("tw").resolve("TW"); // held: utterance 0 is not settled yet
    await h.flush();
    expect(h.published).toEqual([]);
    h.find("one").resolve("ONE");
    await h.flush();
    expect(h.published).toEqual(["stt:final:ONE", "stt:interim:TW"]);
  });

  it("publishes finals in utterance order when they complete out of order", async () => {
    const h = harness();
    h.seq.pushFinal("stt", { value: "a" });
    h.seq.pushFinal("stt", { value: "b" });
    h.seq.pushFinal("stt", { value: "c" });
    h.find("c").resolve("C");
    h.find("b").resolve("B");
    await h.flush();
    expect(h.published).toEqual([]);
    h.find("a").resolve("A");
    await h.flush();
    expect(h.published).toEqual(["stt:final:A", "stt:final:B", "stt:final:C"]);
    expect(h.seq.pendingUtterances).toBe(0);
  });

  it("keeps one interim request in flight and replaces queued ones with the latest", async () => {
    const h = harness();
    h.seq.pushInterim("stt", { value: "h" });
    h.seq.pushInterim("stt", { value: "he" });
    h.seq.pushInterim("stt", { value: "hel" });
    expect(h.jobs.map((j) => j.job.event.value)).toEqual(["h"]);

    h.find("h").resolve("H");
    await h.flush();
    expect(h.jobs.map((j) => j.job.event.value)).toEqual(["h", "hel"]);
    expect(h.find("hel").job.revision).toBeGreaterThan(h.find("h").job.revision);

    h.find("hel").resolve("HEL");
    await h.flush();
    expect(h.published).toEqual(["stt:interim:H", "stt:interim:HEL"]);
  });

  it("aborts interim work when the final arrives and never publishes a stale interim after it", async () => {
    const h = harness();
    h.seq.pushInterim("stt", { value: "par" });
    const interim = h.find("par");
    h.seq.pushFinal("stt", { value: "partial done" });
    expect(interim.job.signal.aborted).toBe(true);

    interim.resolve("late"); // ignored: superseded
    h.find("partial done").resolve("FINAL");
    await h.flush();
    expect(h.published).toEqual(["stt:final:FINAL"]);
    expect(h.errors).toEqual([]); // superseded work is not an error
  });

  it("drops a queued interim when the final arrives", async () => {
    const h = harness();
    h.seq.pushInterim("stt", { value: "a" });
    h.seq.pushInterim("stt", { value: "ab" }); // queued behind "a"
    h.seq.pushFinal("stt", { value: "abc" });
    h.find("a").resolve("A");
    await h.flush();
    expect(h.jobs.map((j) => j.job.event.value)).toEqual(["a", "abc"]);
  });

  it("reports a failed final, closes its utterance and lets later ones through", async () => {
    const h = harness();
    h.seq.pushFinal("stt", { value: "bad" });
    h.seq.pushFinal("stt", { value: "good" });
    h.find("good").resolve("GOOD");
    h.find("bad").reject(new Error("boom"));
    await h.flush();
    expect(h.published).toEqual(["stt:final:GOOD"]);
    expect(h.errors).toHaveLength(1);
    expect(h.errors[0].job.kind).toBe("final");
  });

  it("times out a hung final so it cannot block later lines forever", async () => {
    vi.useFakeTimers();
    const h = harness({ timeoutMs: { final: 1_000 } });
    h.seq.pushFinal("stt", { value: "hung" });
    h.seq.pushFinal("stt", { value: "next" });
    h.find("next").resolve("NEXT");
    await vi.advanceTimersByTimeAsync(999);
    expect(h.published).toEqual([]);
    await vi.advanceTimersByTimeAsync(1);
    expect(h.published).toEqual(["stt:final:NEXT"]);
    expect(h.errors[0].error).toBeInstanceOf(SequencerTimeoutError);
    expect(h.find("hung").job.signal.aborted).toBe(true);
  });

  it("gives a long final more time, matching the backend's own per-character timeout", async () => {
    expect(finalDeadlineMs("short caption")).toBe(FINAL_TIMEOUT_MS + 130);
    // 20 s + 10 ms/char is LibreTranslate's own limit (translate.rs); the sequencer stays 10 s above it.
    const paragraph = "x".repeat(1_500);
    expect(finalDeadlineMs(paragraph)).toBe(45_000);
    expect(finalDeadlineMs(paragraph)).toBeGreaterThan(20_000 + 10 * 1_500);
    expect(finalDeadlineMs("x".repeat(100_000))).toBe(FINAL_TIMEOUT_MAX_MS);
    expect(FINAL_TIMEOUT_MAX_MS).toBeGreaterThan(60_000);

    vi.useFakeTimers();
    const h = harness();
    h.seq.pushFinal("textfield", { value: paragraph });
    await vi.advanceTimersByTimeAsync(35_000); // a slow CPU server needs ~35 s: the old fixed 30 s cut it off
    expect(h.errors).toEqual([]);
    h.find(paragraph).resolve("translated paragraph");
    await vi.advanceTimersByTimeAsync(0);
    expect(h.published).toEqual(["textfield:final:translated paragraph"]);

    h.seq.pushFinal("textfield", { value: paragraph });
    await vi.advanceTimersByTimeAsync(45_000);
    expect(h.errors[0].error).toBeInstanceOf(SequencerTimeoutError);
    expect((h.errors[0].error as SequencerTimeoutError).timeoutMs).toBe(45_000);
  });

  it("accepts a computed deadline per event", async () => {
    vi.useFakeTimers();
    const h = harness({ timeoutMs: { final: (e: Ev) => (e.value === "slow" ? 5_000 : 100) } });
    h.seq.pushFinal("stt", { value: "slow" });
    h.seq.pushFinal("textfield", { value: "fast" });
    await vi.advanceTimersByTimeAsync(100);
    expect(h.errors.map((e) => e.job.event.value)).toEqual(["fast"]);
    await vi.advanceTimersByTimeAsync(4_900);
    expect(h.errors.map((e) => e.job.event.value)).toEqual(["fast", "slow"]);
  });

  it("reports interim failures and continues with the queued interim", async () => {
    const h = harness();
    h.seq.pushInterim("stt", { value: "x" });
    h.seq.pushInterim("stt", { value: "xy" });
    h.find("x").reject(new Error("503"));
    await h.flush();
    expect(h.errors).toHaveLength(1);
    h.find("xy").resolve("XY");
    await h.flush();
    expect(h.published).toEqual(["stt:interim:XY"]);
  });

  it("moves on to the queued interim when the in-flight one times out", async () => {
    vi.useFakeTimers();
    const h = harness({ timeoutMs: { interim: 500 } });
    h.seq.pushInterim("stt", { value: "a" });
    h.seq.pushInterim("stt", { value: "ab" });
    await vi.advanceTimersByTimeAsync(500);
    expect(h.errors[0].error).toBeInstanceOf(SequencerTimeoutError);
    expect(h.jobs.map((j) => j.job.event.value)).toEqual(["a", "ab"]);
    h.find("a").resolve("A"); // too late: already timed out
    h.find("ab").resolve("AB");
    await vi.advanceTimersByTimeAsync(0);
    expect(h.published).toEqual(["stt:interim:AB"]);
  });

  it("holds only the latest interim of a later utterance", async () => {
    const h = harness();
    h.seq.pushFinal("stt", { value: "first" });
    h.seq.pushInterim("stt", { value: "s" });
    h.find("s").resolve("S");
    await h.flush();
    h.seq.pushInterim("stt", { value: "se" });
    h.find("se").resolve("SE");
    await h.flush();
    h.find("first").resolve("FIRST");
    await h.flush();
    expect(h.published).toEqual(["stt:final:FIRST", "stt:interim:SE"]);
  });

  it("keeps streams independent", async () => {
    const h = harness();
    h.seq.pushFinal("stt", { value: "spoken" });
    h.seq.pushFinal("textfield", { value: "typed" });
    h.find("typed").resolve("TYPED");
    await h.flush();
    expect(h.published).toEqual(["textfield:final:TYPED"]);
    h.find("spoken").resolve("SPOKEN");
    await h.flush();
    expect(h.published).toEqual(["textfield:final:TYPED", "stt:final:SPOKEN"]);
  });

  describe("cancel", () => {
    it("publishes the marker as the final when a partial was shown", async () => {
      const h = harness();
      h.seq.pushInterim("stt", { value: "i was say" });
      h.find("i was say").resolve("PARTIAL");
      await h.flush();
      h.seq.cancel("stt", { value: CANCEL_MARKER }, CANCEL_MARKER);
      expect(h.published).toEqual(["stt:interim:PARTIAL", `stt:final:${CANCEL_MARKER}`]);
      expect(h.jobs.some((j) => j.job.event.value === CANCEL_MARKER)).toBe(false); // never translated
    });

    it("closes silently when nothing was shown yet", async () => {
      const h = harness();
      h.seq.pushInterim("stt", { value: "unseen" });
      const interim = h.find("unseen");
      h.seq.cancel("stt", { value: CANCEL_MARKER }, CANCEL_MARKER);
      expect(interim.job.signal.aborted).toBe(true);
      interim.resolve("LATE");
      await h.flush();
      expect(h.published).toEqual([]);
      expect(h.seq.pendingUtterances).toBe(0);
    });

    it("does nothing without an open utterance", () => {
      const h = harness();
      h.seq.cancel("stt", { value: CANCEL_MARKER }, CANCEL_MARKER);
      expect(h.published).toEqual([]);
      expect(h.seq.pendingUtterances).toBe(0);
    });

    it("waits for earlier finals before publishing the marker", async () => {
      const h = harness();
      h.seq.pushInterim("stt", { value: "a" });
      h.find("a").resolve("A");
      await h.flush();
      h.seq.pushFinal("stt", { value: "a b" });
      h.seq.pushInterim("stt", { value: "c" });
      h.find("c").resolve("C"); // held behind the pending final: never shown
      await h.flush();
      h.seq.cancel("stt", { value: CANCEL_MARKER }, CANCEL_MARKER);
      h.find("a b").resolve("AB");
      await h.flush();
      expect(h.published).toEqual(["stt:interim:A", "stt:final:AB"]);
    });

    it("recognises the marker with surrounding whitespace only", () => {
      expect(isCancelMarker("[...]")).toBe(true);
      expect(isCancelMarker("  [...] ")).toBe(true);
      expect(isCancelMarker("[...] and more")).toBe(false);
    });
  });

  describe("late interims", () => {
    it("ignores an interim that repeats the start of the final just closed", () => {
      let now = 1_000;
      const h = harness({ now: () => now });
      h.seq.pushFinal("stt", { value: "Hello there friend" });
      now += 200;
      h.seq.pushInterim("stt", { value: "hello there" });
      expect(h.jobs.map((j) => j.job.kind)).toEqual(["final"]);
      expect(h.seq.pendingUtterances).toBe(1);
    });

    it("ignores a late interim whose final only adds punctuation and capitals", () => {
      let now = 1_000;
      const h = harness({ now: () => now });
      h.seq.pushFinal("stt", { value: "Yes, I think so." });
      now += 300;
      h.seq.pushInterim("stt", { value: "yes i think so" });
      h.seq.pushInterim("stt", { value: "yes i think" });
      expect(h.jobs.map((j) => j.job.kind)).toEqual(["final"]);
      expect(h.seq.pendingUtterances).toBe(1);

      const h2 = harness({ now: () => now });
      h2.seq.pushFinal("stt", { value: "Well... I'm not sure, honestly!" });
      h2.seq.pushInterim("stt", { value: "well im not sure" });
      expect(h2.jobs.map((j) => j.job.kind)).toEqual(["final"]);
    });

    it("normalizes case, punctuation, apostrophes and spacing for the comparison", () => {
      expect(normalizeForMatch("  Yes,   I think so. ")).toBe("yes i think so");
      expect(normalizeForMatch("I’m here—now")).toBe("im here now");
      expect(normalizeForMatch("¿Qué tal?")).toBe("qué tal");
      expect(normalizeForMatch("はい、そうです。")).toBe("はい そうです");
      expect(normalizeForMatch("...")).toBe("");
    });

    it("accepts a genuinely new interim, and a repeat once the window has passed", () => {
      let now = 1_000;
      const h = harness({ now: () => now, staleInterimMs: 500 });
      h.seq.pushFinal("stt", { value: "Hello there" });
      h.seq.pushInterim("stt", { value: "Goodbye" });
      expect(h.jobs).toHaveLength(2);

      const h2 = harness({ now: () => now, staleInterimMs: 500 });
      h2.seq.pushFinal("stt", { value: "Hello there" });
      now += 600;
      h2.seq.pushInterim("stt", { value: "Hello" });
      expect(h2.jobs).toHaveLength(2);
    });
  });

  it("reset aborts running jobs and ignores their late results", async () => {
    const h = harness();
    h.seq.pushInterim("stt", { value: "a" });
    h.seq.pushFinal("textfield", { value: "b" });
    expect(h.seq.runningJobs).toBe(2);
    h.seq.reset();
    expect(h.jobs.every((j) => j.job.signal.aborted)).toBe(true);
    expect(h.seq.runningJobs).toBe(0);
    h.find("a").resolve("A");
    h.find("b").resolve("B");
    await h.flush();
    expect(h.published).toEqual([]);
    expect(h.errors).toEqual([]);
    expect(h.seq.pendingUtterances).toBe(0);
  });

  it("treats a synchronous throw from run as a failure", async () => {
    const errors: unknown[] = [];
    const seq = new TranslationSequencer<Ev>({
      run: () => {
        throw new Error("sync");
      },
      publish: () => {},
      onError: (_job, e) => errors.push(e),
    });
    seq.pushFinal("stt", { value: "x" });
    await new Promise((r) => setTimeout(r, 0));
    expect(errors).toHaveLength(1);
    expect(seq.pendingUtterances).toBe(0);
    expect(seq.runningJobs).toBe(0);
  });
});

describe("TranslationSequencer (P4-C additions)", () => {
  function gapHarness(gapMs: number | (() => number)) {
    const jobs: Pending[] = [];
    const published: string[] = [];
    const seq = new TranslationSequencer<Ev>({
      run: (job) => new Promise<string>((resolve, reject) => jobs.push({ job, resolve, reject })),
      publish: (job, value) => {
        published.push(`${job.kind}:${value}`);
      },
      interimGapMs: gapMs,
    });
    return { seq, jobs, published };
  }

  it("throttles interim requests of one utterance, latest wins", async () => {
    vi.useFakeTimers();
    const h = gapHarness(300);
    h.seq.pushInterim("stt", { value: "a" });
    h.jobs[0].resolve("A");
    await vi.advanceTimersByTimeAsync(0);
    h.seq.pushInterim("stt", { value: "ab" });
    h.seq.pushInterim("stt", { value: "abc" });
    expect(h.jobs).toHaveLength(1);
    await vi.advanceTimersByTimeAsync(299);
    expect(h.jobs).toHaveLength(1);
    await vi.advanceTimersByTimeAsync(1);
    expect(h.jobs.map((j) => j.job.event.value)).toEqual(["a", "abc"]);
  });

  it("waits for the gap after a fast answer, and the final cancels a pending throttled interim", async () => {
    vi.useFakeTimers();
    const h = gapHarness(() => 200);
    h.seq.pushInterim("stt", { value: "a" });
    h.seq.pushInterim("stt", { value: "ab" }); // queued behind the in-flight one
    h.jobs[0].resolve("A");
    await vi.advanceTimersByTimeAsync(0);
    expect(h.jobs).toHaveLength(1); // the gap is not over yet
    h.seq.pushFinal("stt", { value: "abc" });
    await vi.advanceTimersByTimeAsync(500);
    expect(h.jobs.map((j) => `${j.job.kind}:${j.job.event.value}`)).toEqual(["interim:a", "final:abc"]);
  });

  it("abandon closes the open utterance silently and releases later ones", async () => {
    const h = harness();
    h.seq.pushInterim("stt", { value: "lost" });
    h.seq.abandon("stt");
    expect(h.find("lost").job.signal.aborted).toBe(true);
    h.seq.pushFinal("stt", { value: "next" });
    h.find("next").resolve("NEXT");
    await h.flush();
    expect(h.published).toEqual(["stt:final:NEXT"]);
    expect(h.errors).toEqual([]);
    h.seq.abandon("stt"); // nothing open: no-op
    h.seq.abandon("other");
  });

  it("tells whether an utterance is the head of its stream", () => {
    const h = harness();
    h.seq.pushFinal("stt", { value: "one" });
    h.seq.pushFinal("stt", { value: "two" });
    const [one, two] = h.jobs;
    expect(h.seq.isHead("stt", one.job.utterance)).toBe(true);
    expect(h.seq.isHead("stt", two.job.utterance)).toBe(false);
    expect(h.seq.isHead("nope", 0)).toBe(false);
  });

  it("skips the late-interim check for trusted interims", () => {
    const h = harness();
    h.seq.pushFinal("stt", { value: "yes i think" });
    h.seq.pushInterim("stt", { value: "yes" });
    expect(h.jobs).toHaveLength(1);
    h.seq.pushInterim("stt", { value: "yes" }, { trusted: true });
    expect(h.jobs).toHaveLength(2);
  });

  it("closes silently on cancel when publish showed nothing", async () => {
    const jobs: Pending[] = [];
    const published: string[] = [];
    const seq = new TranslationSequencer<Ev>({
      run: (job) => new Promise<string>((resolve, reject) => jobs.push({ job, resolve, reject })),
      publish: (job, value) => {
        if (job.kind === "interim") return false; // e.g. LocalAgreement committed nothing
        published.push(`${job.kind}:${value}`);
      },
    });
    seq.pushInterim("stt", { value: "half" });
    jobs[0].resolve("HALF");
    await new Promise((r) => setTimeout(r, 0));
    seq.cancel("stt", { value: CANCEL_MARKER }, CANCEL_MARKER);
    expect(published).toEqual([]);
  });
});
