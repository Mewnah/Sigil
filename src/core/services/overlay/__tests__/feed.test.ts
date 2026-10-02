import { describe, expect, it } from "vitest";
import { OthersEventSource, TextEventSource, TextEventType, type CaptionEventSource, type TextEvent } from "@/types";
import { CaptionFeed, TEST_CAPTION_MS, type CaptionBus, type FeedClock } from "../feed";

/** A pubsub that bubbles `a.b` to `a` like pubsub-js (subscribers get the original topic). */
class FakeBus implements CaptionBus {
  subs = new Map<string, { source: string; fn: (e: TextEvent | undefined, topic: string | undefined) => void }>();
  #next = 0;
  subscribe(source: CaptionEventSource, fn: (e: TextEvent | undefined, topic: string | undefined) => void): string {
    const token = `t${this.#next++}`;
    this.subs.set(token, { source, fn });
    return token;
  }
  unsubscribe(token: string): void {
    this.subs.delete(token);
  }
  publish(topic: string, event: TextEvent): void {
    for (const { source, fn } of [...this.subs.values()]) {
      if (topic === source || topic.startsWith(`${source}.`)) fn(event, topic);
    }
  }
}

class FakeClock implements FeedClock {
  t = 0;
  timers: { at: number; fn: () => void; id: number }[] = [];
  #id = 0;
  now = () => this.t;
  setTimeout = (fn: () => void, ms: number) => {
    const id = ++this.#id;
    this.timers.push({ at: this.t + ms, fn, id });
    return id;
  };
  clearTimeout = (h: unknown) => {
    this.timers = this.timers.filter((x) => x.id !== h);
  };
  advance(ms: number) {
    const end = this.t + ms;
    for (;;) {
      this.timers.sort((a, b) => a.at - b.at);
      const next = this.timers[0];
      if (!next || next.at > end) break;
      this.timers.shift();
      this.t = next.at;
      next.fn();
    }
    this.t = end;
  }
}

const final = (value: string, extra: Partial<TextEvent> = {}): TextEvent => ({ type: TextEventType.final, value, emotes: {}, ...extra });

function setup(sources: CaptionEventSource[], clearAfterMs = 1000) {
  const bus = new FakeBus();
  const clock = new FakeClock();
  let changes = 0;
  const feed = new CaptionFeed({ interim: true, maxLines: 3, clearAfterMs }, bus, () => changes++, clock);
  feed.configure(sources, { interim: true, maxLines: 3, clearAfterMs });
  return { bus, clock, feed, changes: () => changes, texts: () => feed.model.lines.map((l) => l.text) };
}

describe("CaptionFeed", () => {
  it("shows its sources only, never other players through 'any text'", () => {
    const { bus, texts } = setup([TextEventSource.any]);
    bus.publish(TextEventSource.stt, final("mine"));
    bus.publish(OthersEventSource.stt, final("theirs"));
    bus.publish(TextEventSource.transform_source, final("echo"));
    expect(texts()).toEqual(["mine"]);
  });

  it("does not double lines when 'any' and a specific topic are both selected", () => {
    const { bus, texts } = setup([TextEventSource.stt, TextEventSource.any]);
    expect(bus.subs.size).toBe(1);
    bus.publish(TextEventSource.stt, final("once"));
    expect(texts()).toEqual(["once"]);
  });

  it("other players' captions when asked for", () => {
    const { bus, texts } = setup([OthersEventSource.stt, OthersEventSource.translation]);
    bus.publish(OthersEventSource.stt, final("hola", { segmentId: "loopback-x:1:1" }));
    bus.publish(OthersEventSource.translation, final("hello", { segmentId: "loopback-x:1:1" }));
    bus.publish(TextEventSource.stt, final("mine"));
    expect(texts()).toEqual(["hola", "hello"]);
  });

  it("runs the clear timer", () => {
    const { bus, clock, texts, changes } = setup([TextEventSource.stt], 1000);
    bus.publish(TextEventSource.stt, final("bye soon"));
    const before = changes();
    clock.advance(999);
    expect(texts()).toEqual(["bye soon"]);
    clock.advance(10);
    expect(texts()).toEqual([]);
    expect(changes()).toBe(before + 1);
    expect(clock.timers).toHaveLength(0);
  });

  it("re-subscribes only when the sources change, and clears lines of dropped sources", () => {
    const { bus, feed, texts } = setup([TextEventSource.stt]);
    bus.publish(TextEventSource.stt, final("a"));
    feed.configure([TextEventSource.stt], { interim: true, maxLines: 3, clearAfterMs: 1000 });
    expect(texts()).toEqual(["a"]);
    feed.configure([TextEventSource.translation], { interim: true, maxLines: 3, clearAfterMs: 1000 });
    expect(texts()).toEqual([]);
    expect([...bus.subs.values()].map((s) => s.source)).toEqual([TextEventSource.translation]);
  });

  it("stop unsubscribes and empties", () => {
    const { bus, clock, feed, texts } = setup([TextEventSource.stt]);
    bus.publish(TextEventSource.stt, final("a"));
    feed.stop();
    expect(bus.subs.size).toBe(0);
    expect(texts()).toEqual([]);
    expect(clock.timers).toHaveLength(0);
  });

  it("a test caption stays for its own time", () => {
    const { clock, feed, texts } = setup([TextEventSource.stt], 0);
    feed.showTest("Test");
    expect(texts()).toEqual(["Test"]);
    clock.advance(TEST_CAPTION_MS);
    expect(texts()).toEqual([]);
  });

  it("a short clear timer does not cut a test caption short", () => {
    const { bus, clock, feed, texts } = setup([TextEventSource.stt], 1000);
    feed.showTest("Test");
    bus.publish(TextEventSource.stt, final("spoken"));
    clock.advance(1500);
    expect(texts()).toEqual(["Test"]);
    clock.advance(TEST_CAPTION_MS);
    expect(texts()).toEqual([]);
    expect(clock.timers).toHaveLength(0);
  });

  it("ignores malformed events", () => {
    const { bus, texts } = setup([TextEventSource.stt]);
    bus.publish(TextEventSource.stt, { type: 7 } as unknown as TextEvent);
    expect(texts()).toEqual([]);
  });
});
