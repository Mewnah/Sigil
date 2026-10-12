import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";

/**
 * LAN revocation must reach the WebRTC data channel (P6-D): the document and every pubsub broadcast
 * go host → mirror over a PeerJS DataConnection, not over the `/peer` signaling socket the server
 * closes. When a mirror's signaling socket is gone (revoked by a new token, This PC only, legacy
 * off, or closed by the client), the server sends PeerJS `LEAVE` to the host
 * (src-tauri/src/services/web/peer.rs), and the host provider must drop that mirror's connections.
 */

vi.mock("peerjs", async () => ({ default: (await import("./fakePeerjs")).FakePeer, DataConnection: class {} }));
vi.mock("@/utils/devLog", () => ({ devLog: vi.fn() }));

import { Doc } from "yjs";
import { PEER_SERVER_LEAVE, PeerjsProvider, peerReconnectDelayMs, SERVER_RETRY_CEILING_MS } from "../provider";
import { FakeConn, FakePeer } from "./fakePeerjs";

const providers: PeerjsProvider[] = [];

function host() {
  const doc = new Doc();
  const provider = new PeerjsProvider(doc);
  providers.push(provider);
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
  FakePeer.reset();
});

afterEach(() => {
  for (const provider of providers.splice(0)) provider.dispose();
  vi.useRealTimers();
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

/**
 * The host's own registration (`server`) is what every /client mirror finds it by. It was made once
 * per page load and never again: one dropped signaling socket, one ID-TAKEN while the old socket
 * was still closing, or a first attempt before the web server listened, locked every OBS source
 * out until Sigil restarted.
 */
describe("PeerjsProvider host: registration recovery", () => {
  beforeEach(() => {
    vi.useFakeTimers();
  });

  it("re-registers the same Peer with a growing back-off, keeping its data channels", () => {
    const { peer, connect } = host();
    peer.register();
    const mirror = connect("obs-here", "dc_1");

    peer.disconnect();
    vi.advanceTimersByTime(999);
    expect(peer.reconnect).not.toHaveBeenCalled();
    vi.advanceTimersByTime(1);
    expect(peer.reconnect).toHaveBeenCalledTimes(1);

    // The server is still down: PeerJS gives up that attempt too ("disconnected" again).
    // 1, 2, 4, 8 s, then every 10 s (loopback attempts are cheap and every mirror depends on this one).
    for (const [attempt, wait] of [[2, 2_000], [3, 4_000], [4, 8_000], [5, 10_000], [6, 10_000], [7, 10_000]] as const) {
      peer.disconnect();
      vi.advanceTimersByTime(wait - 1);
      expect(peer.reconnect).toHaveBeenCalledTimes(attempt - 1);
      vi.advanceTimersByTime(1);
      expect(peer.reconnect).toHaveBeenCalledTimes(attempt);
    }

    // Registered again: the back-off starts over. Same id, same Peer, the mirror's channel untouched.
    peer.register();
    peer.disconnect();
    vi.advanceTimersByTime(1_000);
    expect(peer.reconnect).toHaveBeenCalledTimes(8);
    expect(FakePeer.all).toHaveLength(1);
    expect(peer.id).toBe("server");
    expect(peer.destroy).not.toHaveBeenCalled();
    expect(mirror.open).toBe(true);
    expect(mirror.close).not.toHaveBeenCalled();
  });

  it("keeps several attempts inside the 30 s the signaling server holds the host's departure back", () => {
    // HOST_LEAVE_GRACE_SECS in src-tauri/src/services/web/peer.rs: until it is over the mirrors are
    // not told the host is gone. A grace that ended after the third attempt (10 s) turned every
    // slow recovery into a LEAVE for every source.
    const GRACE_MS = 30_000;
    const landed: number[] = [];
    let at = 0;
    for (let attempt = 0; ; attempt++) {
      at += peerReconnectDelayMs(attempt, SERVER_RETRY_CEILING_MS);
      if (at >= GRACE_MS) break;
      landed.push(at);
    }
    expect(landed).toEqual([1_000, 3_000, 7_000, 15_000, 25_000]);
  });

  it("makes a new Peer with the same id when PeerJS destroyed the first one (ID-TAKEN, server not listening yet)", () => {
    const { edit } = host();
    const first = FakePeer.last!;
    // A first registration that fails is aborted by PeerJS: the error, then the Peer is destroyed.
    first.abort("unavailable-id");
    expect(FakePeer.all).toHaveLength(1);
    vi.advanceTimersByTime(999);
    expect(FakePeer.all).toHaveLength(1);
    vi.advanceTimersByTime(1);
    expect(FakePeer.all).toHaveLength(2);

    const second = FakePeer.last!;
    expect(second.id).toBe("server");
    expect(second.options).toMatchObject({ host: "127.0.0.1", port: 3030, key: "", path: "peer" });
    second.abort("socket-error");
    vi.advanceTimersByTime(1_999);
    expect(FakePeer.all).toHaveLength(2);
    vi.advanceTimersByTime(1);
    expect(FakePeer.all).toHaveLength(3);

    // The web server is up: the registration opens and works like the first one did.
    const third = FakePeer.last!;
    third.register();
    const conn = new FakeConn("obs-here", "dc_1");
    third.emit("connection", conn);
    conn.accept();
    expect(conn.sent).toHaveLength(1);
    edit("x");
    expect(conn.sent).toHaveLength(2);
    third.socket.emit("message", { type: PEER_SERVER_LEAVE, src: "obs-here", dst: "server", payload: null });
    expect(conn.close).toHaveBeenCalled();
    // The failed Peers are gone for good and nothing keeps retrying.
    vi.advanceTimersByTime(120_000);
    expect(FakePeer.all).toHaveLength(3);
  });

  it("retries ID-TAKEN of its own reconnect (the old socket still closing) without a second Peer", () => {
    const { peer } = host();
    peer.register();
    peer.disconnect();
    vi.advanceTimersByTime(1_000);
    expect(peer.reconnect).toHaveBeenCalledTimes(1);
    // The server still holds the previous socket: ID-TAKEN. The Peer opened before, so PeerJS only
    // disconnects it again; the next try comes later.
    peer.abort("unavailable-id");
    vi.advanceTimersByTime(1_999);
    expect(peer.reconnect).toHaveBeenCalledTimes(1);
    vi.advanceTimersByTime(1);
    expect(peer.reconnect).toHaveBeenCalledTimes(2);
    peer.register();
    expect(FakePeer.all).toHaveLength(1);
  });

  it("forces a registration that never opens or fails to retry", () => {
    const { peer } = host();
    vi.advanceTimersByTime(19_999);
    expect(peer.disconnect).not.toHaveBeenCalled();
    vi.advanceTimersByTime(1);
    expect(peer.disconnect).toHaveBeenCalledTimes(1);
    vi.advanceTimersByTime(1_000);
    expect(peer.reconnect).toHaveBeenCalledTimes(1);
    // An opened one is left alone.
    peer.register();
    vi.advanceTimersByTime(60_000);
    expect(peer.disconnect).toHaveBeenCalledTimes(1);
  });

  it("ignores late events of a Peer it replaced, and stops for good on dispose", () => {
    const { provider } = host();
    const first = FakePeer.last!;
    first.abort("unavailable-id");
    vi.advanceTimersByTime(1_000);
    const second = FakePeer.last!;
    expect(second).not.toBe(first);

    // The replaced Peer's stragglers start no second timer: one retry per failure.
    first.emit("disconnected");
    first.emit("close");
    second.register();
    vi.advanceTimersByTime(60_000);
    expect(FakePeer.all).toHaveLength(2);
    expect(second.reconnect).not.toHaveBeenCalled();

    second.disconnect();
    provider.dispose();
    expect(second.destroy).toHaveBeenCalled();
    vi.advanceTimersByTime(120_000);
    expect(second.reconnect).not.toHaveBeenCalled();
    expect(FakePeer.all).toHaveLength(2);
  });
});
