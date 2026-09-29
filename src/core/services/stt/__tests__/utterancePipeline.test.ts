import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import { GroqRateLimiter } from "../rateLimiter";
import {
  httpTranscriptionError,
  pcmFromIpc,
  TranscriptionError,
  UtterancePipeline,
  UtteranceQueue,
  type PipelineFatal,
  type PipelineStatus,
} from "../utterancePipeline";

const SR = 16_000;

/** An utterance whose samples all carry `seq` (so merged batches can be decoded back to their seqs). */
function utterance(seq: number, seconds: number) {
  return { seq, samples: new Float32Array(Math.round(seconds * SR)).fill(seq) };
}

/** Seqs present in a merged batch (gaps are 0). */
function seqsIn(samples: Float32Array): number[] {
  const out: number[] = [];
  for (const v of samples) if (v !== 0 && out[out.length - 1] !== v) out.push(v);
  return out;
}

describe("UtteranceQueue", () => {
  const limits = { maxBatchSamples: 10 * SR, gapSamples: 0.2 * SR };

  it("merges from the head up to the batch limit, with silence between utterances", () => {
    const q = new UtteranceQueue();
    q.push(utterance(1, 3));
    q.push(utterance(2, 3));
    q.push(utterance(3, 5));
    expect(q.peekBatchSamples(limits)).toBe(6.2 * SR);
    const batch = q.takeBatch(limits)!;
    expect(batch.seqs).toEqual([1, 2]);
    expect(batch.samples.length).toBe(6.2 * SR);
    expect(batch.samples[3 * SR + 10]).toBe(0); // the gap
    expect(seqsIn(batch.samples)).toEqual([1, 2]);
    expect(q.length).toBe(1);
    expect(q.takeBatch(limits)!.seqs).toEqual([3]);
    expect(q.takeBatch(limits)).toBeNull();
  });

  it("always takes the first utterance, even when it alone exceeds the limit", () => {
    const q = new UtteranceQueue();
    q.push(utterance(1, 12));
    q.push(utterance(2, 1));
    expect(q.takeBatch(limits)!.seqs).toEqual([1]);
    expect(q.queuedSamples).toBe(SR);
  });
});

describe("pcmFromIpc", () => {
  it("reads little-endian f32 from the IPC payload shapes", () => {
    const pcm = new Float32Array([0.5, -0.25, 1]);
    expect(Array.from(pcmFromIpc(pcm.buffer))).toEqual([0.5, -0.25, 1]);
    expect(Array.from(pcmFromIpc(Array.from(new Uint8Array(pcm.buffer))))).toEqual([0.5, -0.25, 1]);
    // Unaligned view with a trailing partial sample.
    const bytes = new Uint8Array(1 + 12 + 2);
    bytes.set(new Uint8Array(pcm.buffer), 1);
    expect(Array.from(pcmFromIpc(bytes.subarray(1)))).toEqual([0.5, -0.25, 1]);
    expect(() => pcmFromIpc("nope")).toThrow();
  });
});

describe("httpTranscriptionError", () => {
  it("classifies statuses", () => {
    expect(httpTranscriptionError(429, "slow down", 3000)).toMatchObject({ kind: "rate_limited", retryAfterMs: 3000 });
    expect(httpTranscriptionError(503, "").kind).toBe("retryable");
    expect(httpTranscriptionError(408, "").kind).toBe("retryable");
    expect(httpTranscriptionError(401, "Invalid API Key").kind).toBe("fatal");
    expect(httpTranscriptionError(404, "model not found").message).toBe("HTTP 404: model not found");
  });
});

describe("UtterancePipeline", () => {
  let now = 1_000_000;
  beforeEach(() => {
    vi.useFakeTimers();
    now = 1_000_000;
  });
  afterEach(() => {
    vi.useRealTimers();
  });

  /** Advances the fake clock and timers together, flushing promise callbacks. */
  async function advance(ms: number) {
    const step = 50;
    for (let t = 0; t < ms; t += step) {
      now += step;
      await vi.advanceTimersByTimeAsync(step);
    }
  }

  function setup(opts: {
    transcribe?: (samples: Float32Array) => Promise<string>;
    limiter?: GroqRateLimiter | null;
    maxQueuedSeconds?: number;
    minBatchSeconds?: number;
  } = {}) {
    const texts: string[] = [];
    const statuses: (PipelineStatus | null)[] = [];
    const fatals: PipelineFatal[] = [];
    const sent: { at: number; seqs: number[] }[] = [];
    const pipeline = new UtterancePipeline({
      sampleRate: SR,
      now: () => now,
      limiter: opts.limiter ?? null,
      maxQueuedSeconds: opts.maxQueuedSeconds,
      minBatchSeconds: opts.minBatchSeconds,
      transcribe: async (samples) => {
        sent.push({ at: now, seqs: seqsIn(samples) });
        return opts.transcribe ? opts.transcribe(samples) : `text ${seqsIn(samples).join("+")}`;
      },
      onText: (t) => texts.push(t),
      onStatus: (s) => statuses.push(s),
      onFatal: (f) => fatals.push(f),
    });
    return { pipeline, texts, statuses, fatals, sent };
  }

  it("sends one request per utterance, in order", async () => {
    const { pipeline, texts, sent } = setup();
    pipeline.push(utterance(1, 2));
    await advance(100);
    pipeline.push(utterance(2, 2));
    await advance(100);
    pipeline.push(utterance(3, 2));
    await advance(100);
    expect(sent.map((s) => s.seqs)).toEqual([[1], [2], [3]]);
    expect(texts).toEqual(["text 1", "text 2", "text 3"]);
  });

  it("merges utterances that queued up while a request was in flight", async () => {
    let release!: () => void;
    const { pipeline, texts, sent } = setup({
      transcribe: (samples) =>
        seqsIn(samples)[0] === 1
          ? new Promise<string>((r) => (release = () => r("first")))
          : Promise.resolve(`merged ${seqsIn(samples).join("+")}`),
    });
    pipeline.push(utterance(1, 2));
    await advance(50);
    pipeline.push(utterance(2, 1));
    pipeline.push(utterance(3, 1));
    await advance(500);
    expect(sent).toHaveLength(1);
    release();
    await advance(100);
    expect(sent.map((s) => s.seqs)).toEqual([[1], [2, 3]]);
    expect(texts).toEqual(["first", "merged 2+3"]);
  });

  it("Groq: never more than 20 requests per minute and no final is ever dropped", async () => {
    const limiter = new GroqRateLimiter(now);
    const { pipeline, texts, sent, fatals } = setup({ limiter });
    // A heavy talker: a 1.5 s utterance every 2 s for 5 minutes (150 utterances).
    const total = 150;
    const pushedAt = new Map<number, number>();
    for (let i = 1; i <= total; i++) {
      pushedAt.set(i, now);
      pipeline.push(utterance(i, 1.5));
      await advance(2_000);
    }
    await advance(120_000);
    // Paced, not bursty: no utterance waits more than a few seconds for its request.
    const worstWait = Math.max(...sent.flatMap((s) => s.seqs.map((seq) => s.at - pushedAt.get(seq)!)));
    expect(worstWait).toBeLessThan(8_000);
    expect(fatals).toEqual([]);
    const times = sent.map((s) => s.at);
    for (let i = 0; i < times.length; i++) {
      const inMinute = times.filter((t) => t >= times[i] && t < times[i] + 60_000).length;
      expect(inMinute).toBeLessThanOrEqual(20);
    }
    // Every utterance was transcribed exactly once, in order (merged when the limit held requests back).
    expect(sent.flatMap((s) => s.seqs)).toEqual(Array.from({ length: total }, (_, i) => i + 1));
    expect(texts.length).toBe(sent.length);
    expect(sent.length).toBeLessThan(total);
    expect(pipeline.queued).toBe(0);
  });

  it("reports a noticeable rate-limit wait and clears it once sent", async () => {
    const limiter = new GroqRateLimiter(now, { requestsPerMinute: 1, requestsPerDay: 100, audioSecondsPerHour: 1e6, minBilledSeconds: 10 });
    const { pipeline, statuses, sent } = setup({ limiter });
    pipeline.push(utterance(1, 1));
    await advance(100);
    pipeline.push(utterance(2, 1));
    await advance(100);
    expect(statuses.at(-1)).toMatchObject({ kind: "rate_limited", queued: 1 });
    await advance(65_000);
    expect(sent.map((s) => s.seqs)).toEqual([[1], [2]]);
    expect(statuses.at(-1)).toBeNull();
  });

  it("retries a failed batch with backoff, keeping order", async () => {
    let failures = 2;
    const { pipeline, texts, statuses, sent } = setup({
      transcribe: async (samples) => {
        if (seqsIn(samples)[0] === 1 && failures-- > 0) throw new Error("network down");
        return `ok ${seqsIn(samples).join("+")}`;
      },
    });
    pipeline.push(utterance(1, 1));
    await advance(100);
    pipeline.push(utterance(2, 1));
    await advance(10_000);
    expect(texts).toEqual(["ok 1", "ok 2"]);
    expect(sent.map((s) => s.seqs)).toEqual([[1], [1], [1], [2]]);
    expect(statuses).toContainEqual(expect.objectContaining({ kind: "retrying", attempt: 1 }));
    expect(statuses.at(-1)).toBeNull();
  });

  it("stops after repeated failures", async () => {
    const { pipeline, fatals } = setup({ transcribe: async () => { throw new Error("ECONNREFUSED"); } });
    pipeline.push(utterance(1, 1));
    await advance(60_000);
    expect(fatals).toEqual([{ reason: "failed", attempts: 4, detail: "ECONNREFUSED" }]);
    expect(pipeline.disposed).toBe(true);
  });

  it("stops at once on a fatal error (bad key)", async () => {
    const { pipeline, fatals, sent } = setup({
      transcribe: async () => { throw httpTranscriptionError(401, "Invalid API Key"); },
    });
    pipeline.push(utterance(1, 1));
    await advance(5_000);
    expect(sent).toHaveLength(1);
    expect(fatals).toEqual([{ reason: "fatal", detail: "HTTP 401: Invalid API Key" }]);
  });

  it("honours Retry-After on 429 without counting it as a failure", async () => {
    let limited = 3;
    const { pipeline, texts, sent, fatals } = setup({
      transcribe: async () => {
        if (limited-- > 0) throw new TranscriptionError("HTTP 429", "rate_limited", 2_000);
        return "done";
      },
    });
    pipeline.push(utterance(1, 1));
    await advance(1_000);
    expect(sent).toHaveLength(1);
    await advance(10_000);
    expect(sent).toHaveLength(4);
    expect(sent[1].at - sent[0].at).toBeGreaterThanOrEqual(2_000);
    expect(texts).toEqual(["done"]);
    expect(fatals).toEqual([]);
  });

  it("gives up when the server asks to wait longer than the limit (daily cap)", async () => {
    const { pipeline, fatals } = setup({
      transcribe: async () => { throw new TranscriptionError("HTTP 429", "rate_limited", 3_600_000); },
    });
    pipeline.push(utterance(1, 1));
    await advance(1_000);
    expect(fatals).toEqual([{ reason: "wait_too_long", waitMs: 3_600_000 }]);
  });

  it("stops instead of growing without bound", async () => {
    const { pipeline, fatals } = setup({
      maxQueuedSeconds: 5,
      transcribe: () => new Promise<string>(() => {}),
    });
    pipeline.push(utterance(1, 2));
    pipeline.push(utterance(2, 2));
    pipeline.push(utterance(3, 2));
    pipeline.push(utterance(4, 2));
    expect(fatals).toEqual([expect.objectContaining({ reason: "backlog" })]);
  });

  it("dispose aborts the request in flight and ignores its late result", async () => {
    let signal!: AbortSignal;
    let resolve!: (t: string) => void;
    const texts: string[] = [];
    const pipeline = new UtterancePipeline({
      sampleRate: SR,
      now: () => now,
      transcribe: (_s, sig) => {
        signal = sig;
        return new Promise((r) => (resolve = r));
      },
      onText: (t) => texts.push(t),
      onStatus: () => {},
      onFatal: () => {},
    });
    pipeline.push(utterance(1, 1));
    await advance(50);
    pipeline.dispose();
    expect(signal.aborted).toBe(true);
    resolve("too late");
    await advance(50);
    expect(texts).toEqual([]);
    pipeline.push(utterance(2, 1));
    expect(pipeline.queued).toBe(0);
  });

  it("ignores empty transcripts", async () => {
    const { pipeline, texts } = setup({ transcribe: async () => "   " });
    pipeline.push(utterance(1, 1));
    await advance(100);
    expect(texts).toEqual([]);
  });

  describe("billing-minimum hold", () => {
    it("holds short utterances so they share one request, then sends after maxHoldMs", async () => {
      const { pipeline, sent, texts } = setup({ minBatchSeconds: 10 });
      pipeline.push(utterance(1, 2));
      await advance(1_000);
      pipeline.push(utterance(2, 2));
      await advance(2_000);
      expect(sent).toEqual([]);
      await advance(1_000); // 3.5 s after the first utterance arrived
      expect(sent.map((s) => s.seqs)).toEqual([[1, 2]]);
      expect(texts).toEqual(["text 1+2"]);
    });

    it("sends as soon as the batch reaches the minimum", async () => {
      const { pipeline, sent } = setup({ minBatchSeconds: 10 });
      pipeline.push(utterance(1, 6));
      await advance(500);
      expect(sent).toEqual([]);
      pipeline.push(utterance(2, 5));
      await advance(50);
      expect(sent.map((s) => s.seqs)).toEqual([[1, 2]]);
    });

    it("an utterance at or above the minimum goes out at once", async () => {
      const { pipeline, sent } = setup({ minBatchSeconds: 10 });
      pipeline.push(utterance(1, 12));
      await advance(50);
      expect(sent.map((s) => s.seqs)).toEqual([[1]]);
    });

    it("waits longer while the VAD reports the next utterance in progress, and sends when it arrives", async () => {
      const { pipeline, sent } = setup({ minBatchSeconds: 10 });
      pipeline.push(utterance(1, 2));
      await advance(1_000);
      pipeline.speechStarted(2);
      await advance(4_000); // past maxHoldMs (3.5 s), within maxHoldWhileSpeakingMs (8 s)
      expect(sent).toEqual([]);
      pipeline.push(utterance(2, 3));
      await advance(50);
      // Still below 10 s, but nobody is speaking and the oldest has waited longer than maxHoldMs: send now.
      expect(sent.map((s) => s.seqs)).toEqual([[1, 2]]);
    });

    it("never holds longer than maxHoldWhileSpeakingMs, even when the speech never ends", async () => {
      const { pipeline, sent } = setup({ minBatchSeconds: 10 });
      pipeline.push(utterance(1, 2));
      pipeline.speechStarted(2);
      await advance(7_900);
      expect(sent).toEqual([]);
      await advance(200);
      expect(sent.map((s) => s.seqs)).toEqual([[1]]);
    });

    it("speechEnded (a dropped or skipped utterance) releases the longer hold", async () => {
      const { pipeline, sent } = setup({ minBatchSeconds: 10 });
      pipeline.push(utterance(1, 2));
      pipeline.speechStarted(2);
      await advance(4_000);
      expect(sent).toEqual([]);
      pipeline.speechEnded(2);
      await advance(50);
      expect(sent.map((s) => s.seqs)).toEqual([[1]]);
    });

    it("a forced split keeps the speech in progress", async () => {
      const { pipeline, sent } = setup({ minBatchSeconds: 30 });
      pipeline.speechStarted(1);
      pipeline.push({ ...utterance(1, 2), continues: true });
      await advance(5_000);
      expect(sent).toEqual([]); // the rest of the same speech (seq 2) is still coming
      pipeline.push(utterance(2, 1));
      await advance(50);
      expect(sent.map((s) => s.seqs)).toEqual([[1, 2]]);
    });

    it("never holds a retry", async () => {
      let fail = true;
      const { pipeline, sent } = setup({
        minBatchSeconds: 10,
        transcribe: async (samples) => {
          if (fail) {
            fail = false;
            throw new Error("network down");
          }
          return `ok ${seqsIn(samples).join("+")}`;
        },
      });
      pipeline.push(utterance(1, 12));
      await advance(50);
      expect(sent).toHaveLength(1);
      await advance(1_100); // retry backoff 1 s; no extra hold
      expect(sent.map((s) => s.seqs)).toEqual([[1], [1]]);
    });

    it("Groq: no hold (no added latency) while requests stay within the daily pace", async () => {
      const limiter = new GroqRateLimiter(now);
      const { pipeline, sent } = setup({ limiter });
      pipeline.push(utterance(1, 2));
      await advance(50);
      expect(sent.map((s) => s.seqs)).toEqual([[1]]);
    });
  });

  /**
   * Hours of speech through the real pipeline and Groq limiter (fake clock). Utterances of 1-6 s; silences drawn
   * so speech fills `speechFraction` of the time; the VAD reports speech 250 ms into each utterance; requests take
   * 500 ms. Returns what Groq would see and bill. The clock is the faked `Date.now()`, so the loop can jump from
   * event to event while every timer still fires at its own time.
   */
  async function simulateGroq(opts: { speechFraction: number; hours: number; minBatchSeconds?: number; seed?: number }) {
    let seed = opts.seed ?? 7;
    const random = () => {
      seed = (seed + 0x6d2b79f5) | 0;
      let t = Math.imul(seed ^ (seed >>> 15), 1 | seed);
      t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t;
      return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
    };
    const clock = () => Date.now();
    const t0 = clock();
    const limiter = new GroqRateLimiter(t0);
    const requests: { at: number; seconds: number; seqs: number[] }[] = [];
    const endedAt = new Map<number, number>();
    const delivered: number[] = [];
    const delays: number[] = [];
    const fatals: PipelineFatal[] = [];
    const pipeline = new UtterancePipeline({
      sampleRate: SR,
      now: clock,
      limiter,
      minBatchSeconds: opts.minBatchSeconds,
      transcribe: (samples) => {
        requests.push({ at: clock(), seconds: samples.length / SR, seqs: seqsIn(samples) });
        return new Promise((resolve) => setTimeout(() => resolve("words"), 500));
      },
      onText: (_text, seqs) => {
        for (const seq of seqs) {
          delivered.push(seq);
          delays.push(clock() - endedAt.get(seq)!);
        }
      },
      onStatus: () => {},
      onFatal: (f) => fatals.push(f),
    });
    const meanSilence = (3.5 * (1 - opts.speechFraction)) / opts.speechFraction;
    const events: { at: number; run: () => void }[] = [];
    let speech = 0;
    let t = 0;
    let seq = 1;
    while (t < opts.hours * 3600) {
      const length = 1 + 5 * random();
      const silence = Math.max(0.3, -Math.log(1 - random()) * meanSilence);
      const s = seq++;
      speech += length;
      events.push({ at: t + 0.25, run: () => pipeline.speechStarted(s) });
      events.push({
        at: t + length,
        run: () => {
          endedAt.set(s, clock());
          // Values: the seq, so merged requests decode back to their utterances.
          pipeline.push({ seq: s, samples: new Float32Array(Math.round(length * SR)).fill(s) });
        },
      });
      t += length + silence;
    }
    for (const e of events) {
      const wait = t0 + Math.round(e.at * 1000) - clock();
      if (wait > 0) await vi.advanceTimersByTimeAsync(wait);
      e.run();
    }
    await vi.advanceTimersByTimeAsync(60_000);
    return { requests, delivered, delays: delays.sort((a, b) => a - b), fatals, utterances: seq - 1, speech, hours: (clock() - t0) / 3_600_000 };
  }

  const maxInWindow = (times: number[], windowMs: number) => {
    let best = 0;
    for (let lo = 0, hi = 0; hi < times.length; hi++) {
      while (times[hi] - times[lo] >= windowMs) lo++;
      best = Math.max(best, hi - lo + 1);
    }
    return best;
  };

  it("Groq long run, heavy talker (85% speech, 4 h): merges to the billing minimum within every Groq limit, drops nothing", async () => {
    const merged = await simulateGroq({ speechFraction: 0.85, hours: 4 });
    const baseline = await simulateGroq({ speechFraction: 0.85, hours: 4, minBatchSeconds: 0 });
    for (const run of [merged, baseline]) {
      // Every utterance transcribed exactly once, in order (the baseline only until the daily cap stopped it).
      expect(run.delivered).toEqual(Array.from({ length: run.delivered.length }, (_, i) => i + 1));
      const times = run.requests.map((r) => r.at);
      expect(maxInWindow(times, 60_000)).toBeLessThanOrEqual(20);
      // Billed audio (10 s minimum per request) in any rolling hour stays inside the 7,200 s budget.
      for (let i = 0; i < run.requests.length; i++) {
        const billed = run.requests
          .filter((r) => r.at >= run.requests[i].at && r.at < run.requests[i].at + 3_600_000)
          .reduce((sum, r) => sum + Math.max(10, r.seconds), 0);
        expect(billed).toBeLessThanOrEqual(7_200);
      }
    }
    // One request per short utterance (the baseline) spends ~630 requests/h and bills ~2.1x the real audio: the
    // 2,000 requests/day are gone after ~3 h and the session stops. Merging lasts the 4 h, near the billing minimum.
    expect(baseline.fatals).toEqual([expect.objectContaining({ reason: "wait_too_long" })]);
    expect(baseline.requests.length).toBe(2_000);
    expect(merged.fatals).toEqual([]);
    expect(merged.delivered.length).toBe(merged.utterances);
    const mergedPerHour = merged.requests.length / merged.hours;
    const baselineSpanMs = baseline.requests[baseline.requests.length - 1].at - baseline.requests[0].at;
    const baselinePerHour = (baseline.requests.length / baselineSpanMs) * 3_600_000;
    // Billed audio over audio actually sent.
    const billedRatio = (run: typeof merged) =>
      run.requests.reduce((sum, r) => sum + Math.max(10, r.seconds), 0) / run.requests.reduce((sum, r) => sum + r.seconds, 0);
    expect(mergedPerHour).toBeLessThan(0.7 * baselinePerHour);
    expect(mergedPerHour).toBeLessThan(430);
    expect(billedRatio(merged)).toBeLessThan(1.45);
    expect(billedRatio(baseline)).toBeGreaterThan(1.9);
    // The price is caption latency, bounded by the hold (8 s while speaking) plus the request.
    expect(merged.delays[merged.delays.length - 1]).toBeLessThanOrEqual(9_000);
  }, 60_000);

  it("Groq long run, light talker (20% speech, 8 h): one request per utterance, no added latency, lasts the day", async () => {
    const run = await simulateGroq({ speechFraction: 0.2, hours: 8 });
    expect(run.fatals).toEqual([]);
    expect(run.delivered).toEqual(Array.from({ length: run.utterances }, (_, i) => i + 1));
    expect(run.requests.length).toBe(run.utterances);
    expect(run.requests.length).toBeLessThan(2_000);
    expect(run.delays[Math.floor(run.delays.length / 2)]).toBeLessThanOrEqual(1_000);
  }, 60_000);
});
