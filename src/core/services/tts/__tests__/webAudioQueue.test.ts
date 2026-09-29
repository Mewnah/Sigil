import { describe, expect, it } from "vitest";
import { WebAudioClipQueue } from "../webAudioQueue";

/** Fake Web Audio: each decoded buffer lasts `bytes / 1000` seconds; time is set by the test. */
class FakeSource {
  buffer: { duration: number } | null = null;
  startedAt: number | null = null;
  stopped = false;
  onended: (() => void) | null = null;
  constructor(private log: FakeSource[]) {}
  connect() {}
  disconnect() {}
  start(when: number) {
    this.startedAt = when;
    this.log.push(this);
  }
  stop() {
    this.stopped = true;
    this.onended?.();
  }
}

class FakeContext {
  state: "running" | "suspended" | "closed" = "running";
  currentTime = 0;
  destination = {};
  started: FakeSource[] = [];
  decoded: number[] = [];
  closeCalls = 0;
  async decodeAudioData(data: ArrayBuffer) {
    this.decoded.push(data.byteLength);
    return { duration: data.byteLength / 1000 };
  }
  createBufferSource() {
    return new FakeSource(this.started);
  }
  async resume() {
    this.state = "running";
  }
  async close() {
    this.closeCalls++;
    this.state = "closed";
  }
}

function setup() {
  const contexts: FakeContext[] = [];
  const queue = new WebAudioClipQueue(() => {
    const ctx = new FakeContext();
    contexts.push(ctx);
    return ctx as unknown as AudioContext;
  });
  return { queue, contexts };
}

function deferred<T>() {
  let resolve!: (v: T) => void;
  let reject!: (e: unknown) => void;
  const promise = new Promise<T>((res, rej) => {
    resolve = res;
    reject = rej;
  });
  return { promise, resolve, reject };
}

describe("WebAudioClipQueue", () => {
  it("plays clips back to back instead of overlapping", async () => {
    const { queue, contexts } = setup();
    await Promise.all([
      queue.enqueue(async () => new Uint8Array(500)),
      queue.enqueue(async () => new Uint8Array(250)),
      queue.enqueue(async () => [1, 2, 3, 4]),
    ]);
    const starts = contexts[0].started.map((s) => s.startedAt);
    expect(starts).toEqual([0, 0.5, 0.75]);
  });

  it("keeps call order even when a later synthesis finishes first", async () => {
    const { queue, contexts } = setup();
    const slow = deferred<Uint8Array>();
    const first = queue.enqueue(() => slow.promise);
    let secondStarted = false;
    const second = queue.enqueue(async () => {
      secondStarted = true;
      return new Uint8Array(100);
    });
    await Promise.resolve();
    expect(secondStarted).toBe(false);
    slow.resolve(new Uint8Array(300));
    await Promise.all([first, second]);
    expect(contexts[0].decoded).toEqual([300, 100]);
  });

  it("drops clips still synthesizing when stopped, and silences playing ones", async () => {
    const { queue, contexts } = setup();
    await queue.enqueue(async () => new Uint8Array(1000));
    const playing = contexts[0].started[0];

    const inFlight = deferred<Uint8Array>();
    const pending = queue.enqueue(() => inFlight.promise);
    let queuedRan = false;
    const queued = queue.enqueue(async () => {
      queuedRan = true;
      return new Uint8Array(10);
    });
    await Promise.resolve();

    queue.stop();
    expect(playing.stopped).toBe(true);
    expect(contexts[0].closeCalls).toBe(1);

    inFlight.resolve(new Uint8Array(500));
    await Promise.all([pending, queued]);
    expect(queuedRan).toBe(false);
    expect(contexts[0].started).toHaveLength(1);
    expect(contexts).toHaveLength(1);
    expect(queue.activeClips).toBe(0);
  });

  it("reports failures and keeps going", async () => {
    const { queue, contexts } = setup();
    const failed = queue.enqueue(async () => {
      throw new Error("server down");
    });
    const next = queue.enqueue(async () => new Uint8Array(100));
    await expect(failed).rejects.toThrow("server down");
    await next;
    expect(contexts[0].started).toHaveLength(1);
  });

  it("skips empty results (audio routed elsewhere)", async () => {
    const { queue, contexts } = setup();
    await queue.enqueue(async () => null);
    await queue.enqueue(async () => new Uint8Array(0));
    await queue.enqueue(async () => undefined);
    expect(contexts).toHaveLength(0);
  });

  it("does not detach the caller's buffer and starts a new context after stop", async () => {
    const { queue, contexts } = setup();
    const bytes = new Uint8Array([1, 2, 3]);
    await queue.enqueue(async () => bytes);
    expect(bytes.byteLength).toBe(3);
    queue.stop();
    await queue.enqueue(async () => new Uint8Array(10));
    expect(contexts).toHaveLength(2);
    expect(contexts[1].started).toHaveLength(1);
  });

  it("resumes a suspended context before scheduling", async () => {
    const { queue, contexts } = setup();
    queue.warmUp();
    contexts[0].state = "suspended";
    await queue.enqueue(async () => new Uint8Array(10));
    expect(contexts[0].state).toBe("running");
  });
});
