import { vi } from "vitest";

/**
 * Stand-ins for PeerJS's `Peer` and `DataConnection` (the tests mock `peerjs` with these):
 * `vi.mock("peerjs", async () => ({ default: (await import("./fakePeerjs")).FakePeer, DataConnection: class {} }))`.
 * They follow the PeerJS 1.5 behaviour the provider depends on: `close()` of a connection emits
 * `close` only when it was open, `destroy()` closes the connections and emits `close`, an aborted
 * Peer is destroyed when it never opened and only disconnected when it did.
 */

type Handler = (...args: any[]) => void;

export class FakeEmitter {
  #handlers = new Map<string, Handler[]>();
  on(event: string, fn: Handler) {
    this.#handlers.set(event, [...(this.#handlers.get(event) ?? []), fn]);
    return this;
  }
  emit(event: string, ...args: unknown[]) {
    for (const fn of this.#handlers.get(event) ?? []) fn(...args);
  }
}

export class FakeConn extends FakeEmitter {
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
  /** The channel opens. */
  accept() {
    this.open = true;
    this.emit("open");
  }
}

export class FakePeer extends FakeEmitter {
  /** The Peer constructed last. */
  static last?: FakePeer;
  /** Every Peer constructed since the last [`FakePeer.reset`], oldest first. */
  static all: FakePeer[] = [];
  static reset() {
    FakePeer.last = undefined;
    FakePeer.all = [];
  }

  socket = new FakeEmitter();
  open = false;
  disconnected = false;
  destroyed = false;
  /** The connections this Peer made with `connect()`. */
  conns: FakeConn[] = [];
  #everOpen = false;

  constructor(readonly id: string, readonly options: any) {
    super();
    FakePeer.last = this;
    FakePeer.all.push(this);
  }

  destroy = vi.fn(() => {
    if (this.destroyed) return;
    this.disconnect();
    for (const conn of this.conns) conn.close();
    this.destroyed = true;
    this.emit("close");
  });

  disconnect = vi.fn(() => {
    if (this.disconnected) return;
    this.disconnected = true;
    this.open = false;
    this.emit("disconnected", this.id);
  });

  reconnect = vi.fn(() => {
    if (this.destroyed || !this.disconnected) throw new Error("cannot reconnect");
    this.disconnected = false;
  });

  connect = vi.fn((peerId: string, _options?: unknown) => {
    const conn = new FakeConn(peerId, `dc_${FakePeer.all.indexOf(this)}_${this.conns.length}`);
    this.conns.push(conn);
    return conn;
  });

  /** The signaling server accepted the id. */
  register() {
    this.open = true;
    this.#everOpen = true;
    this.emit("open", this.id);
  }

  /** A PeerJS error event (`type` is the PeerJS error type). */
  fail(type: string) {
    this.emit("error", Object.assign(new Error(type), { type }));
  }

  /** PeerJS `_abort`: the error, then destroyed if it never opened, else just disconnected. */
  abort(type: string) {
    this.fail(type);
    if (this.#everOpen) this.disconnect();
    else this.destroy();
  }
}
