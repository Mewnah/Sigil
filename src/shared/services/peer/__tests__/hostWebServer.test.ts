import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";

vi.mock("@tauri-apps/api/core", () => ({ invoke: vi.fn() }));
const listeners = new Map<string, (e: { payload: unknown }) => void>();
vi.mock("@tauri-apps/api/event", () => ({
  listen: vi.fn(async (name: string, fn: (e: { payload: unknown }) => void) => {
    listeners.set(name, fn);
    return () => {};
  }),
}));
vi.mock("react-toastify", () => ({
  toast: Object.assign(vi.fn(), { error: vi.fn(), warn: vi.fn(), success: vi.fn() }),
}));
vi.mock("i18next", () => ({ default: { t: (key: string) => key } }));

import { invoke } from "@tauri-apps/api/core";
import { toast } from "react-toastify";
import { proxy } from "valtio";
import Service_PubSub, { type WebServerStatus } from "@/shared/services/pubsub";
import type { LanAccessSettings } from "../lanAccess";

/**
 * LAN security stage 2 on the host (P6-D): Service_PubSub keeps the Rust web server in sync with the
 * persisted `lanAccess` setting, makes the LAN token when there is none, and links to other Sigils
 * with their remote token.
 */

const TOKEN = "0123456789abcdefghjkmnpqrs";
const NEW_TOKEN = "zyxwvtsrqpnmkjhgfedcba9876";

type Settings = { lanOriginCheck: boolean; lanAccess: LanAccessSettings; linkAddress: string; id: string; uiScale: number };

function status(lan: Partial<WebServerStatus["lan"]> = {}): WebServerStatus {
  return {
    state: "listening",
    port: 3030,
    error: null,
    originCheck: true,
    originCheckSetting: true,
    originCheckEnv: null,
    lan: { mode: "lan", boundMode: "lan", boundAddress: "0.0.0.0:3030", tokenSet: true, legacyNoToken: false, ...lan },
  };
}

let settings: Settings;
let calls: { cmd: string; args?: any }[];
let regenerate: () => Promise<{ token: string; status: WebServerStatus }>;

class FakeWebSocket {
  static last?: FakeWebSocket;
  static CONNECTING = 0;
  static OPEN = 1;
  static CLOSING = 2;
  static CLOSED = 3;
  readonly OPEN = 1;
  readyState = 0;
  onopen: (() => void) | null = null;
  onmessage: ((msg: { data: unknown }) => void) | null = null;
  onclose: ((event?: { code: number }) => void) | null = null;
  onerror: (() => void) | null = null;
  sent: string[] = [];
  constructor(readonly url: string) {
    FakeWebSocket.last = this;
  }
  send(wire: string) {
    this.sent.push(wire);
  }
  close() {
    this.readyState = FakeWebSocket.CLOSED;
  }
}

/** Let valtio's batched subscriptions and pending promises run. */
async function flush() {
  for (let i = 0; i < 5; i++) await Promise.resolve();
  await vi.advanceTimersByTimeAsync(0);
}

const lanPushes = () => calls.filter((c) => c.cmd === "plugin:web|set_lan_access").map((c) => c.args.settings);

beforeEach(() => {
  vi.useFakeTimers();
  listeners.clear();
  calls = [];
  FakeWebSocket.last = undefined;
  settings = proxy<Settings>({
    lanOriginCheck: true,
    lanAccess: { mode: "lan", token: "", legacyNoToken: false },
    linkAddress: "",
    id: "host-a",
    uiScale: 1,
  });
  regenerate = async () => ({ token: NEW_TOKEN, status: status() });
  vi.stubGlobal("WebSocket", FakeWebSocket);
  vi.stubGlobal("navigator", { clipboard: { writeText: vi.fn() } });
  vi.stubGlobal("window", {
    Config: { isServer: () => true, isApp: () => true, isClient: () => false, serverNetwork: { ip: "10.0.0.1", port: "3030", host: "localhost" } },
    ApiShared: { peer: { broadcastPubSubParts: vi.fn() } },
    ApiServer: { state: settings },
  });
  vi.mocked(invoke).mockImplementation(async (cmd: string, args?: any) => {
    calls.push({ cmd, args });
    if (cmd === "plugin:web|regenerate_lan_token") return (await regenerate()) as any;
    if (cmd === "plugin:web|set_lan_access")
      return status({ mode: args.settings.mode, tokenSet: !!args.settings.token, legacyNoToken: args.settings.legacyNoToken }) as any;
    if (cmd === "plugin:web|set_origin_check") return status() as any;
    return undefined as any;
  });
});

afterEach(() => {
  vi.useRealTimers();
  vi.unstubAllGlobals();
});

describe("host LAN access sync", () => {
  it("makes a LAN token on first start and persists it", async () => {
    const svc = new Service_PubSub();
    await svc.attachHostWebServer(settings);
    await flush();

    expect(lanPushes()[0]).toEqual({ mode: "lan", token: "", legacyNoToken: false });
    expect(calls.filter((c) => c.cmd === "plugin:web|regenerate_lan_token")).toHaveLength(1);
    expect(settings.lanAccess.token).toBe(NEW_TOKEN);
    // Rust already has the new token: no extra push for it.
    expect(lanPushes()).toHaveLength(1);
    expect(svc.webServer.status?.lan.tokenSet).toBe(true);
  });

  it("pushes a persisted token without regenerating, in canonical form", async () => {
    settings.lanAccess.token = TOKEN.toUpperCase();
    const svc = new Service_PubSub();
    await svc.attachHostWebServer(settings);
    await flush();
    expect(lanPushes()[0]).toEqual({ mode: "lan", token: TOKEN, legacyNoToken: false });
    expect(calls.some((c) => c.cmd === "plugin:web|regenerate_lan_token")).toBe(false);
    expect(settings.lanAccess.token).toBe(TOKEN);
    expect(calls.some((c) => c.cmd === "plugin:web|set_origin_check")).toBe(true);
  });

  it("pushes mode and legacy changes, and nothing for unrelated settings", async () => {
    settings.lanAccess.token = TOKEN;
    const svc = new Service_PubSub();
    await svc.attachHostWebServer(settings);
    await flush();
    const before = lanPushes().length;

    settings.uiScale = 1.2;
    await flush();
    expect(lanPushes()).toHaveLength(before);

    settings.lanAccess.mode = "localhost";
    await flush();
    expect(lanPushes().at(-1)).toEqual({ mode: "localhost", token: TOKEN, legacyNoToken: false });

    settings.lanAccess.legacyNoToken = true;
    await flush();
    expect(lanPushes().at(-1)).toEqual({ mode: "localhost", token: TOKEN, legacyNoToken: true });

    // A whole new object (profile import) is noticed too.
    settings.lanAccess = { mode: "lan", token: TOKEN, legacyNoToken: false };
    await flush();
    expect(lanPushes().at(-1)).toEqual({ mode: "lan", token: TOKEN, legacyNoToken: false });
    expect(svc.webServer.status?.lan.mode).toBe("lan");
  });

  it("regenerates on request and when the token is cleared", async () => {
    settings.lanAccess.token = TOKEN;
    const svc = new Service_PubSub();
    await svc.attachHostWebServer(settings);
    await flush();

    await expect(svc.regenerateLanToken()).resolves.toBe(NEW_TOKEN);
    expect(settings.lanAccess.token).toBe(NEW_TOKEN);

    regenerate = async () => ({ token: TOKEN, status: status() });
    settings.lanAccess.token = "";
    await flush();
    expect(settings.lanAccess.token).toBe(TOKEN);
    expect(calls.filter((c) => c.cmd === "plugin:web|regenerate_lan_token")).toHaveLength(2);
  });

  it("reports a failed regeneration once and keeps the settings", async () => {
    settings.lanAccess.token = TOKEN;
    const svc = new Service_PubSub();
    await svc.attachHostWebServer(settings);
    await flush();
    regenerate = async () => {
      throw new Error("no RNG");
    };
    await expect(svc.regenerateLanToken()).resolves.toBeNull();
    expect(settings.lanAccess.token).toBe(TOKEN);
    expect(vi.mocked(toast.error)).toHaveBeenCalledWith("toasts.lan_token_regenerate_failed", expect.anything());
  });

  it("explains LAN refusals in their own toast", async () => {
    settings.lanAccess.token = TOKEN;
    const svc = new Service_PubSub();
    await svc.attachHostWebServer(settings);
    listeners.get("web:origin_rejected")!({ payload: { route: "pubsub", origin: "http://10.0.0.1:3030", remote: "10.0.0.9", reason: "invalid_token" } });
    expect(vi.mocked(toast.warn)).toHaveBeenLastCalledWith("toasts.web_lan_token_refused", expect.anything());
    await vi.advanceTimersByTimeAsync(30_000);
    listeners.get("web:origin_rejected")!({ payload: { route: "peer", origin: null, remote: "10.0.0.9", reason: "lan_disabled" } });
    expect(vi.mocked(toast.warn)).toHaveBeenLastCalledWith("toasts.web_lan_disabled_refused", expect.anything());
    await vi.advanceTimersByTimeAsync(30_000);
    listeners.get("web:origin_rejected")!({ payload: { route: "peer", origin: "https://evil.example", remote: "127.0.0.1", reason: "foreign_origin" } });
    expect(vi.mocked(toast.warn)).toHaveBeenLastCalledWith("toasts.web_origin_rejected", expect.anything());
  });
});

describe("Link apps with a remote token", () => {
  it("copies this host's address with its token", async () => {
    settings.lanAccess.token = TOKEN;
    const svc = new Service_PubSub();
    svc.copyLinkAddress();
    expect(vi.mocked(navigator.clipboard.writeText)).toHaveBeenLastCalledWith(`10.0.0.1:3030?token=${TOKEN}`);
    expect(vi.mocked(toast.success)).toHaveBeenCalled();
    svc.copyLinkAddress("192.168.5.5");
    expect(vi.mocked(navigator.clipboard.writeText)).toHaveBeenLastCalledWith(`192.168.5.5:3030?token=${TOKEN}`);
    // This PC only: still copied, with a warning.
    svc.webServer.status = status({ mode: "localhost", boundMode: "localhost" });
    svc.copyLinkAddress();
    expect(vi.mocked(toast.warn)).toHaveBeenLastCalledWith("toasts.lan_access_this_pc_only", expect.anything());
  });

  it("connects with the remote token and stops on a token refusal", () => {
    settings.linkAddress = `10.0.0.2:3030?token=${TOKEN}`;
    const svc = new Service_PubSub();
    svc.linkConnect();
    const socket = FakeWebSocket.last!;
    expect(socket.url).toMatch(new RegExp(`^ws://10\\.0\\.0\\.2:3030/pubsub\\?id=host-a-\\d+&token=${TOKEN}$`));
    socket.readyState = FakeWebSocket.OPEN;
    socket.onopen?.();
    expect(svc.serviceState.state).toBe("connected");

    // The other Sigil made a new token: 4401, no reconnect loop.
    socket.onclose?.({ code: 4401 });
    expect(vi.mocked(toast.error)).toHaveBeenLastCalledWith("toasts.link_token_refused", expect.anything());
    expect(svc.serviceState.state).toBe("disconnected");
    FakeWebSocket.last = undefined;
    vi.advanceTimersByTime(120_000);
    expect(FakeWebSocket.last).toBeUndefined();
  });

  it("stops when the other Sigil accepts only its own PC, but retries other drops", () => {
    settings.linkAddress = "10.0.0.2:3030";
    const svc = new Service_PubSub();
    svc.linkConnect();
    let socket = FakeWebSocket.last!;
    expect(socket.url).not.toContain("token=");
    socket.onclose?.({ code: 1006 });
    vi.advanceTimersByTime(1_000);
    expect(FakeWebSocket.last).not.toBe(socket);
    socket = FakeWebSocket.last!;
    socket.onclose?.({ code: 4403 });
    expect(vi.mocked(toast.error)).toHaveBeenLastCalledWith("toasts.link_lan_disabled", expect.anything());
    FakeWebSocket.last = undefined;
    vi.advanceTimersByTime(120_000);
    expect(FakeWebSocket.last).toBeUndefined();
  });

  it("refuses a malformed remote token and still refuses itself", () => {
    const svc = new Service_PubSub();
    settings.linkAddress = "10.0.0.2:3030?token=abc";
    svc.linkConnect();
    expect(FakeWebSocket.last).toBeUndefined();
    expect(vi.mocked(toast.error)).toHaveBeenLastCalledWith("settings.link_remote_token_invalid", expect.anything());

    settings.linkAddress = `10.0.0.1:3030?token=${TOKEN}`;
    svc.linkConnect();
    expect(FakeWebSocket.last).toBeUndefined();
    expect(vi.mocked(toast.error)).toHaveBeenLastCalledWith("toasts.cannot_connect_self");
  });
});
