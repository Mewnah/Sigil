import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";

vi.mock("@tauri-apps/api/core", () => ({ invoke: vi.fn() }));
vi.mock("@tauri-apps/api/event", () => ({ listen: vi.fn(async () => () => {}) }));
vi.mock("react-toastify", () => ({
  toast: Object.assign(vi.fn(), { error: vi.fn(), warn: vi.fn(), success: vi.fn() }),
}));
vi.mock("i18next", () => ({ default: { t: (key: string) => key } }));

import { invoke } from "@tauri-apps/api/core";
import Service_PubSub from "../index";

/**
 * Two real Service_PubSub hosts on a fake network. Each host's Link socket connects to the other
 * host's /pubsub server, so an event reaches the other side by two paths:
 * - the sender's `pubsub_broadcast` → every WS peer on its server, i.e. the other host's Link socket
 *   (`source: "link"` there);
 * - the sender's own Link socket → the other host's server (`source: "local"` there).
 */
type Host = {
  name: string;
  ip: string;
  svc: Service_PubSub;
  delivered: { topic: string; data: any }[];
  /** This host's outbound Link socket (connected to the other host's server). */
  link?: FakeWebSocket;
};

type Delivery = { to: Host; wire: string; source: "local" | "link" };

let hosts: Host[];
let current: Host;
let queue: Delivery[];

class FakeWebSocket {
  static CONNECTING = 0;
  static OPEN = 1;
  static CLOSING = 2;
  static CLOSED = 3;
  static last?: FakeWebSocket;
  readonly OPEN = 1;
  readyState = 0;
  owner = current;
  onopen: (() => void) | null = null;
  onmessage: ((msg: { data: unknown }) => void) | null = null;
  onclose: (() => void) | null = null;
  onerror: (() => void) | null = null;
  constructor(readonly url: string) {
    FakeWebSocket.last = this;
  }
  send(wire: string) {
    queue.push({ to: other(this.owner), wire, source: "local" });
  }
  close() {
    this.readyState = FakeWebSocket.CLOSED;
  }
}

function other(h: Host) {
  return hosts.find((x) => x !== h)!;
}

function useHost(h: Host) {
  current = h;
  const w = window as any;
  w.Config.serverNetwork = { ip: h.ip, port: 3030 };
  w.ApiServer.state = { id: h.name, linkAddress: `${other(h).ip}:3030` };
}

function makeHost(name: string, ip: string): Host {
  const svc = new Service_PubSub();
  const host: Host = { name, ip, svc, delivered: [] };
  svc.publishLocally = ({ topic, data }) => void host.delivered.push({ topic, data });
  return host;
}

function linkTo(h: Host) {
  useHost(h);
  h.svc.linkConnect();
  const socket = FakeWebSocket.last!;
  expect(socket.owner).toBe(h);
  expect(socket.url.startsWith(`ws://${other(h).ip}:3030/pubsub?id=`)).toBe(true);
  socket.readyState = FakeWebSocket.OPEN;
  socket.onopen?.();
  h.link = socket;
}

/** Deliver queued frames until the network is quiet. */
function drain() {
  for (let hops = 0; queue.length; hops++) {
    expect(hops, "the event keeps bouncing").toBeLessThan(200);
    const { to, wire, source } = queue.shift()!;
    useHost(to);
    if (source === "link") to.link?.onmessage?.({ data: wire });
    else (to.svc as any).consumePubSubMessage(wire, "local");
  }
}

const values = (h: Host, topic: string) => h.delivered.filter((d) => d.topic === topic).map((d) => d.data.value);

beforeEach(() => {
  vi.useFakeTimers();
  queue = [];
  vi.stubGlobal("WebSocket", FakeWebSocket);
  vi.stubGlobal("window", {
    Config: { isServer: () => true, isApp: () => true, isClient: () => false, serverNetwork: {} },
    ApiShared: { peer: { broadcastPubSubParts: vi.fn() } },
    ApiServer: { state: {} },
  });
  vi.mocked(invoke).mockImplementation(async (cmd: string, args?: any) => {
    // A server broadcast reaches every WS peer on that server: here, the other host's Link socket.
    if (cmd === "plugin:web|pubsub_broadcast") queue.push({ to: other(current), wire: args.value, source: "link" });
    return undefined as any;
  });
  hosts = [makeHost("A", "10.0.0.1"), makeHost("B", "10.0.0.2")];
});

afterEach(() => {
  for (const h of hosts) {
    useHost(h);
    h.svc.linkDisconnect();
  }
  vi.useRealTimers();
  vi.unstubAllGlobals();
});

describe("Service_PubSub Link apps", () => {
  it("two hosts linked to each other deliver every event exactly once on each side", () => {
    const [A, B] = hosts;
    linkTo(A);
    linkTo(B);

    useHost(A);
    A.svc.publish("text.stt", { type: 0, value: "hello", emotes: {} });
    drain();
    expect(values(A, "text.stt")).toEqual(["hello"]);
    expect(values(B, "text.stt")).toEqual(["hello"]);

    useHost(B);
    B.svc.publishText("text.translation" as any, { type: 0, value: "hola" });
    drain();
    expect(values(A, "text.translation")).toEqual(["hola"]);
    expect(values(B, "text.translation")).toEqual(["hola"]);

    // The same text published again is a new event, not a duplicate.
    useHost(A);
    A.svc.publish("text.stt", { type: 0, value: "hello", emotes: {} });
    drain();
    expect(values(B, "text.stt")).toEqual(["hello", "hello"]);
  });

  it("a one-way link still delivers once", () => {
    const [A, B] = hosts;
    linkTo(A);
    useHost(A);
    A.svc.publish("text.stt", { type: 0, value: "one" });
    drain();
    expect(values(B, "text.stt")).toEqual(["one"]);
    useHost(B);
    B.svc.publish("text.stt", { type: 0, value: "two" });
    drain();
    expect(values(A, "text.stt")).toEqual(["one", "two"]);
    expect(values(B, "text.stt")).toEqual(["one", "two"]);
  });
});

describe("Service_PubSub inbound sanitization", () => {
  const attack = JSON.stringify({
    topic: "text.textfield",
    data: { type: 0, value: "hi <img src=x onerror=alert(1)>", emotes: { hi: 'x" onerror="alert(1)' } },
  });

  it("never publishes peer emote URLs or live markup", () => {
    const [A] = hosts;
    useHost(A);
    (A.svc as any).consumePubSubMessage(attack, "local");
    expect(A.delivered).toHaveLength(1);
    const data = A.delivered[0]!.data;
    expect(data.emotes).toEqual({});
    expect(data.value).toBe("hi <\u200Bimg src=x onerror=alert(1)>");
    // What goes on to /client, OBS and linked hosts is the sanitized event too.
    const broadcast = vi.mocked(invoke).mock.calls.find(([cmd]) => cmd === "plugin:web|pubsub_broadcast");
    const wire = JSON.parse((broadcast?.[1] as any).value);
    expect(wire.data).toEqual(data);
    expect(wire.mid).toMatch(/^[A-Za-z0-9_-]+$/);
  });

  it("lets the host's own emote scan fill in emotes for peer text", () => {
    const [A] = hosts;
    useHost(A);
    const seen: unknown[] = [];
    A.svc.setTextEmoteEnricher((d) => {
      seen.push(d.emotes);
      return d.emotes ? d : { ...d, emotes: { hi: "https://cdn.example/hi/1x" } };
    });
    (A.svc as any).consumePubSubMessage(attack, "local");
    expect(seen).toEqual([undefined]);
    expect(A.delivered[0]!.data.emotes).toEqual({ hi: "https://cdn.example/hi/1x" });
  });
});
