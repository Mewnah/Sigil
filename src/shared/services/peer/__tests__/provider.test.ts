import { beforeEach, describe, expect, it, vi } from "vitest";

/**
 * LAN revocation must reach the WebRTC data channel (P6-D): the document and every pubsub broadcast
 * go host → mirror over a PeerJS DataConnection, not over the `/peer` signaling socket the server
 * closes. When a mirror's signaling socket is gone (revoked by a new token, This PC only, legacy
 * off, or closed by the client), the server sends PeerJS `LEAVE` to the host
 * (src-tauri/src/services/web/peer.rs), and the host provider must drop that mirror's connections.
 */

type Handler = (...args: any[]) => void;

// Hoisted with the `peerjs` mock (the provider imports PeerJS before this file's body runs).
const { FakeEmitter, FakePeer } = vi.hoisted(() => {
  class FakeEmitter {
    #handlers = new Map<string, Handler[]>();
    on(event: string, fn: Handler) {
      this.#handlers.set(event, [...(this.#handlers.get(event) ?? []), fn]);
      return this;
    }
    emit(event: string, ...args: unknown[]) {
      for (const fn of this.#handlers.get(event) ?? []) fn(...args);
    }
  }
  class FakePeer extends FakeEmitter {
    static last?: FakePeer;
    socket = new FakeEmitter();
    destroy = vi.fn();
    constructor(readonly id: string, readonly options: unknown) {
      super();
      FakePeer.last = this;
    }
  }
  return { FakeEmitter, FakePeer };
});

/** Stand-in for PeerJS's DataConnection: `close()` emits `close` only when it was open (as PeerJS). */
class FakeConn extends FakeEmitter {
  open = false;
  sent: Uint8Array[] = [];
  close = vi.fn(() => {
    if (!this.open) return;
    this.open = false;
    this.emit("close");
  });
  constructor(readonly peer: string, readonly connectionId: string) {
    super();
  }
  send(data: Uint8Array) {
    if (!this.open) throw new Error("not open");
    this.sent.push(data);
  }
  accept() {
    this.open = true;
    this.emit("open");
  }
}

vi.mock("peerjs", () => ({ default: FakePeer, DataConnection: class {} }));
vi.mock("@/utils/devLog", () => ({ devLog: vi.fn() }));

import { Doc } from "yjs";
import { PEER_SERVER_LEAVE, PeerjsProvider } from "../provider";

function host() {
  const doc = new Doc();
  const provider = new PeerjsProvider(doc);
  provider.connectServer({ id: "server", host: "127.0.0.1", port: "3030" });
  const peer = FakePeer.last!;
  const connect = (peerId: string, connectionId: string, open = true) => {
    const conn = new FakeConn(peerId, connectionId);
    peer.emit("connection", conn);
    if (open) conn.accept();
    return conn;
  };
  const edit = (value: string) => doc.getMap("m").set("k", value);
  const leave = (src: unknown) => peer.socket.emit("message", { type: PEER_SERVER_LEAVE, src, dst: "server", payload: null });
  return { provider, peer, connect, edit, leave };
}

beforeEach(() => {
  FakePeer.last = undefined;
});

describe("PeerjsProvider host: mirrors that left signaling", () => {
  it("closes a revoked mirror's data channel and stops streaming to it", () => {
    const { provider, connect, edit, leave } = host();
    const lan = connect("lan-client", "dc_lan");
    const obsHere = connect("obs-here", "dc_obs");
    // Each got the initial document state on open.
    expect(lan.sent).toHaveLength(1);
    expect(obsHere.sent).toHaveLength(1);

    leave("lan-client");
    expect(lan.close).toHaveBeenCalled();
    expect(lan.open).toBe(false);

    edit("after revocation");
    provider.broadcastPubSub({ topic: "text.stt", data: { value: "secret caption" } } as any);
    expect(lan.sent).toHaveLength(1);
    // The mirror that did not leave keeps receiving.
    expect(obsHere.sent).toHaveLength(3);
    expect(obsHere.close).not.toHaveBeenCalled();
  });

  it("drops every connection of that peer, also one that never opened", () => {
    const { provider, connect, edit, leave } = host();
    const first = connect("lan-client", "dc_1");
    const pending = connect("lan-client", "dc_2", false);
    const other = connect("other", "dc_3");

    expect(provider.dropPeer("nobody")).toBe(0);
    leave("lan-client");
    expect(first.close).toHaveBeenCalled();
    expect(pending.close).toHaveBeenCalled();
    // The one that never opened left the broadcast set too (sending on it would throw).
    expect(() => edit("x")).not.toThrow();
    expect(other.sent).toHaveLength(2);
    expect(provider.dropPeer("lan-client")).toBe(0);

    // Should the dropped one still report `open`, it gets no document and is closed again.
    pending.accept();
    expect(pending.sent).toHaveLength(0);
    expect(pending.open).toBe(false);
    edit("y");
    expect(other.sent).toHaveLength(3);
  });

  it("ignores other signaling messages and malformed LEAVEs", () => {
    const { peer, connect, leave } = host();
    const conn = connect("lan-client", "dc_1");
    peer.socket.emit("message", { type: "HEARTBEAT" });
    peer.socket.emit("message", { type: "OFFER", src: "lan-client", payload: {} });
    peer.socket.emit("message", null);
    leave(undefined);
    leave(42);
    expect(conn.close).not.toHaveBeenCalled();
    expect(conn.open).toBe(true);
  });
});
