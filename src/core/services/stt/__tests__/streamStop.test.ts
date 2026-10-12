import { proxy } from "valtio";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import { ServiceNetworkState } from "@/types";
import { STREAM_OFFLINE_CONFIRM_MS, StreamStop, type StreamStopDeps } from "../streamStop";

/** A service stand-in and the pubsub / Twitch live state the controller follows. */
function setup(options: { enabled?: boolean; live?: boolean | undefined } = {}) {
  const topics = new Map<string, () => void>();
  const liveListeners: (() => void)[] = [];
  const state = { enabled: options.enabled ?? true, active: true, live: options.live as boolean | undefined };
  const calls: string[] = [];
  const deps: StreamStopDeps = {
    subscribe: (topic, fn) => {
      topics.set(topic, fn);
      return () => topics.delete(topic);
    },
    onLive: (fn) => {
      liveListeners.push(fn);
      return () => liveListeners.splice(liveListeners.indexOf(fn), 1);
    },
    isLive: () => state.live,
    setTimeout: (fn, ms) => setTimeout(fn, ms),
    clearTimeout: (handle) => clearTimeout(handle as ReturnType<typeof setTimeout>),
  };
  const control: StreamStop = new StreamStop(
    {
      enabled: () => state.enabled,
      active: () => state.active,
      stop: () => {
        calls.push("stop");
        state.active = false;
      },
      // A service's own start() tells the controller, like the services do.
      start: () => {
        calls.push("start");
        control.noteStart();
        state.active = true;
      },
    },
    deps,
  );
  const dispose = control.attach();
  return {
    state,
    calls,
    control,
    dispose,
    ended: () => topics.get("stream.on_ended")?.(),
    started: () => topics.get("stream.on_started")?.(),
    live: () => liveListeners.forEach((fn) => fn()),
    subscribed: () => [...topics.keys()].sort(),
  };
}

describe("StreamStop", () => {
  beforeEach(() => {
    vi.useFakeTimers();
  });
  afterEach(() => {
    vi.useRealTimers();
  });

  it("stops only once the stream has stayed offline", async () => {
    const t = setup();
    t.ended();
    await vi.advanceTimersByTimeAsync(STREAM_OFFLINE_CONFIRM_MS - 1);
    expect(t.calls).toEqual([]);
    await vi.advanceTimersByTimeAsync(1);
    expect(t.calls).toEqual(["stop"]);
  });

  it("does not stop when the stream is live again at the check, even if no start event came", async () => {
    const t = setup({ live: true });
    t.ended();
    await vi.advanceTimersByTimeAsync(STREAM_OFFLINE_CONFIRM_MS * 2);
    expect(t.calls).toEqual([]);
  });

  it("stops when the live state is unknown (Twitch is not connected)", async () => {
    const t = setup({ live: undefined });
    t.ended();
    await vi.advanceTimersByTimeAsync(STREAM_OFFLINE_CONFIRM_MS);
    expect(t.calls).toEqual(["stop"]);
  });

  it("a stream that comes back before the stop cancels it, whichever way it is reported", async () => {
    for (const back of ["event", "live"] as const) {
      const t = setup({ live: false });
      t.ended();
      await vi.advanceTimersByTimeAsync(STREAM_OFFLINE_CONFIRM_MS / 2);
      if (back === "event") t.started();
      else t.live();
      await vi.advanceTimersByTimeAsync(STREAM_OFFLINE_CONFIRM_MS * 2);
      expect(t.calls, back).toEqual([]);
    }
  });

  it("a repeated end while one is pending does not postpone the stop", async () => {
    const t = setup();
    t.ended();
    await vi.advanceTimersByTimeAsync(STREAM_OFFLINE_CONFIRM_MS / 2);
    t.ended();
    await vi.advanceTimersByTimeAsync(STREAM_OFFLINE_CONFIRM_MS / 2);
    expect(t.calls).toEqual(["stop"]);
  });

  it("starts again what it stopped when the stream comes back, once", async () => {
    for (const back of ["event", "live"] as const) {
      const t = setup();
      t.ended();
      await vi.advanceTimersByTimeAsync(STREAM_OFFLINE_CONFIRM_MS);
      expect(t.calls).toEqual(["stop"]);
      if (back === "event") t.started();
      else t.live();
      expect(t.calls, back).toEqual(["stop", "start"]);
      // Its own start is not a start by hand: another stream end stops it again, and the next return starts it again.
      t.live();
      expect(t.calls, back).toEqual(["stop", "start"]);
      t.ended();
      await vi.advanceTimersByTimeAsync(STREAM_OFFLINE_CONFIRM_MS);
      t.started();
      expect(t.calls, back).toEqual(["stop", "start", "stop", "start"]);
    }
  });

  it("never starts what it did not stop, what the user started by hand, or anything once the option is off", async () => {
    const idle = setup();
    idle.started();
    idle.live();
    expect(idle.calls).toEqual([]);

    const byHand = setup();
    byHand.ended();
    await vi.advanceTimersByTimeAsync(STREAM_OFFLINE_CONFIRM_MS);
    byHand.control.noteStart(); // the user pressed Start (and stopped it again)
    byHand.state.active = false;
    byHand.started();
    expect(byHand.calls).toEqual(["stop"]);

    const off = setup();
    off.ended();
    await vi.advanceTimersByTimeAsync(STREAM_OFFLINE_CONFIRM_MS);
    off.state.enabled = false;
    off.started();
    expect(off.calls).toEqual(["stop"]);
  });

  it("does nothing while the option is off or the service is not running", async () => {
    const off = setup({ enabled: false });
    off.ended();
    await vi.advanceTimersByTimeAsync(STREAM_OFFLINE_CONFIRM_MS * 2);
    expect(off.calls).toEqual([]);

    const idle = setup();
    idle.state.active = false;
    idle.ended();
    await vi.advanceTimersByTimeAsync(STREAM_OFFLINE_CONFIRM_MS * 2);
    expect(idle.calls).toEqual([]);

    // Stopped by the user while the stop was pending.
    const late = setup();
    late.ended();
    late.state.active = false;
    await vi.advanceTimersByTimeAsync(STREAM_OFFLINE_CONFIRM_MS * 2);
    expect(late.calls).toEqual([]);
  });

  it("a Start by hand while the stop waits cancels it; the controller's own start does not forget what it stopped", async () => {
    const t = setup();
    t.ended();
    await vi.advanceTimersByTimeAsync(STREAM_OFFLINE_CONFIRM_MS / 3);
    t.control.noteStart(); // the user pressed Start while the stream looked offline
    await vi.advanceTimersByTimeAsync(2 * STREAM_OFFLINE_CONFIRM_MS);
    expect(t.calls).toEqual([]);

    // A stop that was made stays resumable across the controller's own start (it is not a start by hand).
    const own = setup();
    own.ended();
    await vi.advanceTimersByTimeAsync(STREAM_OFFLINE_CONFIRM_MS);
    own.started();
    expect(own.calls).toEqual(["stop", "start"]);
    own.ended();
    await vi.advanceTimersByTimeAsync(STREAM_OFFLINE_CONFIRM_MS);
    expect(own.calls).toEqual(["stop", "start", "stop"]);
  });

  it("disposing unsubscribes and drops a pending stop", async () => {
    const t = setup();
    expect(t.subscribed()).toEqual(["stream.on_ended", "stream.on_started"]);
    t.ended();
    t.dispose();
    expect(t.subscribed()).toEqual([]);
    await vi.advanceTimersByTimeAsync(STREAM_OFFLINE_CONFIRM_MS * 2);
    expect(t.calls).toEqual([]);
  });
});

/** The wiring the app uses: window.ApiShared.pubsub and the live state of the host's Twitch service, no fakes between. */
describe("StreamStop with the host's own wiring", () => {
  beforeEach(() => {
    vi.useFakeTimers();
  });
  afterEach(() => {
    vi.useRealTimers();
    vi.unstubAllGlobals();
  });

  function host(withTwitch = true) {
    const topics = new Map<string, () => void>();
    const twitch = { state: proxy({ liveStatus: ServiceNetworkState.connected }) };
    vi.stubGlobal("window", {
      ApiServer: withTwitch ? { twitch } : {},
      ApiShared: {
        pubsub: {
          subscribe: (topic: string, fn: () => void) => {
            topics.set(topic, fn);
            return topic;
          },
          unsubscribe: (id: string) => {
            topics.delete(id);
          },
        },
      },
    });
    const calls: string[] = [];
    const state = { active: true };
    const control: StreamStop = new StreamStop({
      enabled: () => true,
      active: () => state.active,
      stop: () => {
        calls.push("stop");
        state.active = false;
      },
      start: () => {
        calls.push("start");
        control.noteStart();
        state.active = true;
      },
    });
    return { topics, twitch, calls, state, control, dispose: control.attach() };
  }

  it("resumes when Twitch's liveStatus turns live again: the only signal, since nothing publishes stream.on_started", async () => {
    const h = host();
    expect([...h.topics.keys()].sort()).toEqual(["stream.on_ended", "stream.on_started"]);
    h.twitch.state.liveStatus = ServiceNetworkState.disconnected;
    h.topics.get("stream.on_ended")?.();
    await vi.advanceTimersByTimeAsync(STREAM_OFFLINE_CONFIRM_MS);
    expect(h.calls).toEqual(["stop"]);
    h.twitch.state.liveStatus = ServiceNetworkState.connected; // the next Helix poll sees the stream
    await vi.advanceTimersByTimeAsync(0); // valtio notifies on a microtask
    expect(h.calls).toEqual(["stop", "start"]);
    h.dispose();
    h.state.active = false;
    h.twitch.state.liveStatus = ServiceNetworkState.disconnected;
    h.twitch.state.liveStatus = ServiceNetworkState.connected;
    await vi.advanceTimersByTimeAsync(0);
    expect(h.calls).toEqual(["stop", "start"]);
  });

  it("a stream that is live again before the check keeps the service running", async () => {
    const h = host();
    h.twitch.state.liveStatus = ServiceNetworkState.disconnected;
    h.topics.get("stream.on_ended")?.();
    await vi.advanceTimersByTimeAsync(STREAM_OFFLINE_CONFIRM_MS / 2);
    h.twitch.state.liveStatus = ServiceNetworkState.connected;
    await vi.advanceTimersByTimeAsync(2 * STREAM_OFFLINE_CONFIRM_MS);
    expect(h.calls).toEqual([]);
  });

  it("still stops, and never resumes, when the host has no Twitch service", async () => {
    const h = host(false);
    h.topics.get("stream.on_ended")?.();
    await vi.advanceTimersByTimeAsync(STREAM_OFFLINE_CONFIRM_MS);
    expect(h.calls).toEqual(["stop"]);
  });
});
