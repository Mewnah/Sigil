import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";

/**
 * The /client mirror's connection (the OBS browser source): it is on stream for hours with nobody
 * to refresh it, so it must keep the loading view and retry when the first connection fails,
 * retry forever with a capped back-off, give a stalled attempt a deadline, and never tear down a
 * healthy data channel for a stale timer or a signaling-only error.
 */

vi.mock("peerjs", async () => ({ default: (await import("./fakePeerjs")).FakePeer, DataConnection: class {} }));
vi.mock("@/utils/devLog", () => ({ devLog: vi.fn() }));

import { Doc } from "yjs";
import { PEER_CLIENT_MIRROR_EVENT, PEER_INVALID_KEY, PeerjsProvider, peerErrorType, peerReconnectDelayMs, type PeerClientMirrorDetail } from "../provider";
import { FakePeer } from "./fakePeerjs";

const LINK = { id: "server", host: "localhost", port: "3030", token: "" };

let providers: PeerjsProvider[];

beforeEach(() => {
  vi.useFakeTimers();
  FakePeer.reset();
  providers = [];
  vi.spyOn(console, "error").mockImplementation(() => {});
  vi.spyOn(console, "warn").mockImplementation(() => {});
});

afterEach(() => {
  for (const provider of providers) provider.dispose();
  vi.useRealTimers();
});

function mirror() {
  const provider = new PeerjsProvider(new Doc());
  providers.push(provider);
  const phases: PeerClientMirrorDetail[] = [];
  provider.addEventListener(PEER_CLIENT_MIRROR_EVENT, (e) => phases.push((e as CustomEvent<PeerClientMirrorDetail>).detail));
  const state = { settled: "pending" as "pending" | "resolved" | "rejected", error: undefined as unknown };
  const connect = () => {
    const promise = provider.connectClient(LINK);
    promise.then(
      () => (state.settled = "resolved"),
      (error) => {
        state.settled = "rejected";
        state.error = error;
      },
    );
    return promise;
  };
  /** The Peer registers and its data channel to the host opens (the host answered). */
  const open = (peer = FakePeer.last!) => {
    peer.register();
    const conn = peer.conns[0]!;
    conn.accept();
    return conn;
  };
  return { provider, phases, state, connect, open };
}

describe("peerReconnectDelayMs", () => {
  it("backs off 1, 2, 4, 8, 16 s and then stays at 30 s", () => {
    expect([0, 1, 2, 3, 4, 5, 6, 50].map((n) => peerReconnectDelayMs(n))).toEqual([1_000, 2_000, 4_000, 8_000, 16_000, 30_000, 30_000, 30_000]);
    expect([0, 3, 4, 5, 50].map((n) => peerReconnectDelayMs(n, 10_000))).toEqual([1_000, 8_000, 10_000, 10_000, 10_000]);
  });
});

describe("PeerjsProvider mirror: first connection", () => {
  it("retries a failed first attempt in place and resolves once a later one opens", async () => {
    const { connect, open, phases, state } = mirror();
    const connected = connect();
    const first = FakePeer.last!;
    first.register();
    // The host has not registered 'server' yet (EXPIRE): the page must not give up.
    first.fail("peer-unavailable");
    await vi.advanceTimersByTimeAsync(0);
    expect(state.settled).toBe("pending");
    expect(first.destroy).toHaveBeenCalled();

    await vi.advanceTimersByTimeAsync(999);
    expect(FakePeer.all).toHaveLength(1);
    await vi.advanceTimersByTimeAsync(1);
    expect(FakePeer.all).toHaveLength(2);
    const second = FakePeer.last!;
    expect(second.id).not.toBe(first.id);
    expect(state.settled).toBe("pending");

    open(second);
    await connected;
    expect(state.settled).toBe("resolved");
    expect(phases).toEqual([{ phase: "error", message: "peer-unavailable" }, { phase: "synced" }]);
  });

  it("also retries when the signaling socket fails before anything opened", async () => {
    const { connect, open, state } = mirror();
    const connected = connect();
    // The Rust web server is not listening yet: the WebSocket fails, PeerJS aborts and destroys the Peer.
    FakePeer.last!.abort("socket-error");
    await vi.advanceTimersByTimeAsync(1_000);
    FakePeer.last!.abort("network");
    await vi.advanceTimersByTimeAsync(2_000);
    expect(FakePeer.all).toHaveLength(3);
    expect(state.settled).toBe("pending");
    open();
    await connected;
  });

  it("rejects only for a refused LAN token, and then retries nothing", async () => {
    const { connect, state, phases } = mirror();
    const connected = connect();
    connected.catch(() => {});
    FakePeer.last!.abort(PEER_INVALID_KEY);
    await vi.advanceTimersByTimeAsync(0);
    expect(state.settled).toBe("rejected");
    expect(peerErrorType(state.error)).toBe(PEER_INVALID_KEY);
    await vi.advanceTimersByTimeAsync(10 * 60_000);
    expect(FakePeer.all).toHaveLength(1);
    expect(phases).toEqual([]);
  });

  it("rejects when the browser has no WebRTC (a retry cannot change that)", async () => {
    const { connect, state } = mirror();
    connect().catch(() => {});
    FakePeer.last!.abort("browser-incompatible");
    await vi.advanceTimersByTimeAsync(60_000);
    expect(state.settled).toBe("rejected");
    expect(FakePeer.all).toHaveLength(1);
  });
});

describe("PeerjsProvider mirror: reconnecting", () => {
  it("never gives up: past the old 20-attempt cap it still retries every 30 s", async () => {
    const { connect, open, phases, state } = mirror();
    const connected = connect();
    open();
    await connected;

    FakePeer.last!.conns[0]!.close();
    for (let i = 0; i < 40; i++) {
      const wait = peerReconnectDelayMs(i);
      await vi.advanceTimersByTimeAsync(wait - 1);
      expect(FakePeer.all).toHaveLength(i + 1);
      await vi.advanceTimersByTimeAsync(1);
      expect(FakePeer.all).toHaveLength(i + 2);
      FakePeer.last!.abort("network");
    }
    expect(state.settled).toBe("resolved");
    expect(phases.some((p) => p.phase === "failed")).toBe(false);

    // The host is back at last (after about 20 minutes): the next try connects.
    await vi.advanceTimersByTimeAsync(30_000);
    const peer = FakePeer.last!;
    open(peer);
    expect(phases.at(-1)).toEqual({ phase: "synced" });
    await vi.advanceTimersByTimeAsync(10 * 60_000);
    expect(FakePeer.last).toBe(peer);
  });

  it("starts the back-off over once a channel opened", async () => {
    const { connect, open } = mirror();
    const connected = connect();
    open();
    await connected;
    for (let i = 0; i < 4; i++) {
      FakePeer.last!.conns[0]!.close();
      await vi.advanceTimersByTimeAsync(1_000);
      open();
    }
    // Four drops, each retried after 1 s: the delay did not grow.
    expect(FakePeer.all).toHaveLength(5);
  });

  it("abandons an attempt that neither opens nor fails, and tries again", async () => {
    const { connect, open, phases } = mirror();
    const connected = connect();
    const stalled = FakePeer.last!;
    stalled.register();
    // The data channel never opens (ICE stuck): PeerJS says nothing, and used to be waited on for ever.
    await vi.advanceTimersByTimeAsync(19_999);
    expect(stalled.destroy).not.toHaveBeenCalled();
    await vi.advanceTimersByTimeAsync(1);
    expect(stalled.destroy).toHaveBeenCalled();
    expect(phases).toEqual([{ phase: "error", message: "timeout" }]);
    await vi.advanceTimersByTimeAsync(1_000);
    expect(FakePeer.all).toHaveLength(2);
    open();
    await connected;
  });

  it("gives a signaling server that never answers the same deadline", async () => {
    const { connect } = mirror();
    connect();
    const silent = FakePeer.last!;
    await vi.advanceTimersByTimeAsync(20_000);
    expect(silent.destroy).toHaveBeenCalled();
    await vi.advanceTimersByTimeAsync(1_000);
    expect(FakePeer.all).toHaveLength(2);
  });

  it("does not wait for the deadline when ICE fails before the channel opens", async () => {
    const { connect } = mirror();
    connect();
    const peer = FakePeer.last!;
    peer.register();
    // PeerJS reports the failure on the connection only, then closes it without a 'close' event.
    peer.conns[0]!.emit("error", Object.assign(new Error("failed"), { type: "negotiation-failed" }));
    peer.conns[0]!.close();
    await vi.advanceTimersByTimeAsync(1_000);
    expect(FakePeer.all).toHaveLength(2);
  });

  it("keeps a healthy data channel through signaling errors and a stale timer", async () => {
    const { connect, open, phases } = mirror();
    const connected = connect();
    const conn = open();
    await connected;
    const peer = FakePeer.last!;

    // The signaling socket drops (a Wi-Fi roam, the web server rebinding) while the channel lives on.
    peer.fail("network");
    peer.fail("socket-closed");
    peer.disconnect();
    // The mirror registers again, and the channel never noticed.
    await vi.advanceTimersByTimeAsync(1_000);
    expect(peer.reconnect).toHaveBeenCalledTimes(1);
    peer.register();
    await vi.advanceTimersByTimeAsync(5 * 60_000);
    expect(FakePeer.all).toHaveLength(1);
    expect(peer.destroy).not.toHaveBeenCalled();
    expect(conn.close).not.toHaveBeenCalled();
    // Registered again over the same Peer: no second channel was made.
    expect(peer.connect).toHaveBeenCalledTimes(1);
    expect(phases).toEqual([{ phase: "synced" }]);

    // When the channel itself ends, exactly one retry follows (no timer storm).
    conn.close();
    expect(phases.at(-1)).toEqual({ phase: "reconnecting" });
    await vi.advanceTimersByTimeAsync(999);
    expect(FakePeer.all).toHaveLength(1);
    await vi.advanceTimersByTimeAsync(1);
    expect(FakePeer.all).toHaveLength(2);
    open();
    await vi.advanceTimersByTimeAsync(5 * 60_000);
    expect(FakePeer.all).toHaveLength(2);
  });

  it("drops an open channel whose ICE state stays disconnected, but not one that recovers", async () => {
    const { connect, open, phases } = mirror();
    const connected = connect();
    const conn = open();
    await connected;

    // A hiccup: disconnected, then back. The channel stays.
    conn.emit("iceStateChanged", "disconnected");
    await vi.advanceTimersByTimeAsync(9_000);
    conn.emit("iceStateChanged", "connected");
    await vi.advanceTimersByTimeAsync(60_000);
    expect(FakePeer.all).toHaveLength(1);
    expect(conn.close).not.toHaveBeenCalled();

    // The host stopped answering (its process died, the PC slept): ICE would only fail after ~30 s,
    // and the signaling error that used to give it away is ignored while the channel is open.
    conn.emit("iceStateChanged", "disconnected");
    await vi.advanceTimersByTimeAsync(9_999);
    expect(FakePeer.all).toHaveLength(1);
    await vi.advanceTimersByTimeAsync(1);
    expect(conn.close).toHaveBeenCalled();
    expect(phases.at(-1)).toEqual({ phase: "reconnecting" });
    await vi.advanceTimersByTimeAsync(1_000);
    expect(FakePeer.all).toHaveLength(2);
  });

  describe("signaling lost while the channel is open", () => {
    async function openMirror() {
      const m = mirror();
      const connected = m.connect();
      const conn = m.open();
      await connected;
      const peer = FakePeer.last!;
      /** The signaling socket dies the way PeerJS reports it: a network error, then disconnected. */
      const down = () => {
        peer.fail("network");
        peer.disconnect();
      };
      return { ...m, conn, peer, down };
    }

    it("tries to register again, and gives the channel up when the socket stays down (a killed or restarted host)", async () => {
      const { conn, peer, down, phases } = await openMirror();
      down();
      // Attempts after 1, 2, 4 and then every 5 s since the loss; each one fails like the first.
      let elapsed = 0;
      let attempts = 0;
      for (const at of [1_000, 3_000, 7_000, 12_000]) {
        await vi.advanceTimersByTimeAsync(at - elapsed);
        elapsed = at;
        expect(peer.reconnect).toHaveBeenCalledTimes(++attempts);
        down();
      }
      // Not before the deadline: a roam that takes a few seconds costs nothing.
      await vi.advanceTimersByTimeAsync(14_999 - elapsed);
      expect(conn.close).not.toHaveBeenCalled();
      expect(FakePeer.all).toHaveLength(1);

      await vi.advanceTimersByTimeAsync(1);
      expect(conn.close).toHaveBeenCalled();
      expect(peer.destroy).toHaveBeenCalled();
      expect(phases.at(-1)).toEqual({ phase: "reconnecting" });
      // The normal retry follows at once (the back-off starts over after an open channel).
      await vi.advanceTimersByTimeAsync(1_000);
      expect(FakePeer.all).toHaveLength(2);
      // The dead attempt does not keep trying in the background.
      const calls = peer.reconnect.mock.calls.length;
      await vi.advanceTimersByTimeAsync(60_000);
      expect(peer.reconnect.mock.calls.length).toBe(calls);
    });

    it("keeps the channel when the socket registers again in time, and counts again from zero", async () => {
      const { conn, peer, down, phases } = await openMirror();
      down();
      await vi.advanceTimersByTimeAsync(1_000);
      peer.register();
      await vi.advanceTimersByTimeAsync(60_000);
      expect(conn.close).not.toHaveBeenCalled();
      expect(peer.connect).toHaveBeenCalledTimes(1);
      expect(FakePeer.all).toHaveLength(1);

      // A later loss starts the 15 s over and the attempts at 1 s again.
      down();
      await vi.advanceTimersByTimeAsync(999);
      expect(peer.reconnect).toHaveBeenCalledTimes(1);
      await vi.advanceTimersByTimeAsync(1);
      expect(peer.reconnect).toHaveBeenCalledTimes(2);
      await vi.advanceTimersByTimeAsync(14_000);
      expect(conn.close).toHaveBeenCalled();
      expect(phases.at(-1)).toEqual({ phase: "reconnecting" });
    });

    it("lets go of the channel at once when the host refuses the mirror on its way back (LAN rules changed)", async () => {
      const { conn, peer, down, phases } = await openMirror();
      down();
      await vi.advanceTimersByTimeAsync(1_000);
      // The server closed this mirror's socket for a rules change; registering again is refused.
      peer.abort(PEER_INVALID_KEY);
      expect(conn.close).toHaveBeenCalled();
      expect(phases.at(-1)).toEqual({ phase: "failed", message: PEER_INVALID_KEY });
      // Slow retries only, as for any refused running mirror.
      await vi.advanceTimersByTimeAsync(59_999);
      expect(FakePeer.all).toHaveLength(1);
      await vi.advanceTimersByTimeAsync(1);
      expect(FakePeer.all).toHaveLength(2);
    });

    it("does nothing about a signaling loss after the channel already ended", async () => {
      const { conn, peer, down } = await openMirror();
      conn.close();
      await vi.advanceTimersByTimeAsync(1_000);
      const second = FakePeer.last!;
      expect(second).not.toBe(peer);
      // Late events of the replaced Peer change nothing.
      down();
      await vi.advanceTimersByTimeAsync(15_000);
      expect(peer.reconnect).not.toHaveBeenCalled();
      expect(FakePeer.all).toHaveLength(2);
    });

    it("does not start the signaling watch before the channel opened (the attempt just fails)", async () => {
      const { connect } = mirror();
      connect();
      const peer = FakePeer.last!;
      peer.register();
      peer.fail("network");
      peer.disconnect();
      await vi.advanceTimersByTimeAsync(1_000);
      expect(peer.reconnect).not.toHaveBeenCalled();
      expect(FakePeer.all).toHaveLength(2);
    });
  });

  it("ignores late events of an attempt that was replaced", async () => {
    const { connect, open, phases } = mirror();
    const connected = connect();
    const first = FakePeer.last!;
    first.register();
    first.fail("peer-unavailable");
    await vi.advanceTimersByTimeAsync(1_000);
    open();
    await connected;
    const healthy = FakePeer.last!;
    const before = phases.length;

    first.fail("network");
    first.conns[0]!.emit("close");
    first.conns[0]!.emit("error", { type: "negotiation-failed" });
    first.emit("open");
    await vi.advanceTimersByTimeAsync(120_000);
    expect(FakePeer.all).toHaveLength(2);
    expect(healthy.destroy).not.toHaveBeenCalled();
    expect(phases).toHaveLength(before);
  });

  it("keeps trying slowly when a running mirror is refused (token rotated), and says why", async () => {
    const { connect, open, phases } = mirror();
    const connected = connect();
    open();
    await connected;

    FakePeer.last!.conns[0]!.close();
    await vi.advanceTimersByTimeAsync(1_000);
    FakePeer.last!.abort(PEER_INVALID_KEY);
    expect(phases.at(-1)).toEqual({ phase: "failed", message: PEER_INVALID_KEY });
    await vi.advanceTimersByTimeAsync(59_999);
    expect(FakePeer.all).toHaveLength(2);
    await vi.advanceTimersByTimeAsync(1);
    expect(FakePeer.all).toHaveLength(3);
    // The host accepts it again (LAN switched back on).
    open();
    expect(phases.at(-1)).toEqual({ phase: "synced" });
  });

  it("stops everything on dispose", async () => {
    const { connect, provider } = mirror();
    connect();
    const peer = FakePeer.last!;
    peer.register();
    peer.fail("peer-unavailable");
    provider.dispose();
    await vi.advanceTimersByTimeAsync(10 * 60_000);
    expect(FakePeer.all).toHaveLength(1);
  });

  it("passes the LAN token as the PeerJS key and uses a new alphanumeric id for every attempt", async () => {
    const provider = new PeerjsProvider(new Doc());
    providers.push(provider);
    provider.connectClient({ ...LINK, host: "192.168.1.50", token: "lan-token" }).catch(() => {});
    for (let i = 0; i < 5; i++) {
      FakePeer.last!.abort("network");
      await vi.advanceTimersByTimeAsync(30_000);
    }
    const ids = FakePeer.all.map((p) => p.id);
    expect(new Set(ids).size).toBe(ids.length);
    for (const peer of FakePeer.all) {
      expect(peer.id).toMatch(/^[A-Za-z0-9]{32}$/);
      expect(peer.options).toMatchObject({ host: "192.168.1.50", port: 3030, key: "lan-token", path: "peer" });
    }
  });
});
