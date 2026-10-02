import { beforeEach, describe, expect, it, vi } from "vitest";

const invokeMock = vi.fn();
const listenMock = vi.fn();
vi.mock("@tauri-apps/api/core", () => ({ invoke: (...args: unknown[]) => invokeMock(...args) }));
vi.mock("@tauri-apps/api/event", () => ({ listen: (...args: unknown[]) => listenMock(...args) }));

const { DEFAULT_LISTEN_PORT, OscReceiverClient, isLoopbackHost, listenOptionsFor, parseListenPort } = await import("../receiver");
const { OSC_EVENTS } = await import("../types");

const base = { enable: true, muteSync: true, oscQuery: true, oscListenPort: "9001", oscHost: "127.0.0.1" };

describe("listenOptionsFor", () => {
  it("runs the receiver only with the integration and mute sync on", () => {
    expect(listenOptionsFor(base)).toEqual({ host: "127.0.0.1", port: 9001, oscQuery: true, muteSync: true });
    expect(listenOptionsFor({ ...base, enable: false })).toBeNull();
    expect(listenOptionsFor({ ...base, muteSync: false })).toBeNull();
  });

  it("normalizes host and port", () => {
    expect(listenOptionsFor({ ...base, oscHost: "  ", oscListenPort: "abc", oscQuery: false })).toEqual({
      host: "127.0.0.1",
      port: DEFAULT_LISTEN_PORT,
      oscQuery: false,
      muteSync: true,
    });
    expect(parseListenPort("9002")).toBe(9002);
    // OSCQuery only finds a VRChat on this PC: never asked for with VRChat elsewhere (no warning the UI cannot clear).
    expect(listenOptionsFor({ ...base, oscHost: "192.168.1.20" })).toEqual({
      host: "192.168.1.20",
      port: 9001,
      oscQuery: false,
      muteSync: true,
    });
    expect(listenOptionsFor({ ...base, oscHost: "localhost" })?.oscQuery).toBe(true);
    expect(parseListenPort("0")).toBe(DEFAULT_LISTEN_PORT);
    expect(parseListenPort("70000")).toBe(DEFAULT_LISTEN_PORT);
  });
});

describe("isLoopbackHost", () => {
  it("knows this PC (OSCQuery) from another PC (fixed port)", () => {
    for (const h of ["127.0.0.1", " localhost ", "LOCALHOST", "::1", "[::1]", "127.0.1.1", "::ffff:127.0.0.1", ""]) {
      expect(isLoopbackHost(h), h).toBe(true);
    }
    for (const h of ["192.168.1.20", "vrchat-pc", "10.0.0.127", "1270.0.0.1"]) expect(isLoopbackHost(h), h).toBe(false);
  });
});

describe("OscReceiverClient", () => {
  beforeEach(() => {
    invokeMock.mockReset();
    listenMock.mockReset();
  });

  it("starts, skips identical options, restarts on change and stops, one call at a time", async () => {
    const order: string[] = [];
    invokeMock.mockImplementation(async (cmd: string) => {
      order.push(`${cmd}:start`);
      await new Promise((r) => setTimeout(r, 5));
      order.push(`${cmd}:end`);
      return cmd === "plugin:osc|listen_start" ? { running: true } : undefined;
    });
    const c = new OscReceiverClient();
    const opts = listenOptionsFor(base)!;
    const a = c.sync(opts);
    const b = c.sync({ ...opts });
    const d = c.sync({ ...opts, port: 9002 });
    const e = c.sync(null);
    expect(await a).toEqual({ running: true });
    expect(await b).toBeNull();
    expect(await d).toEqual({ running: true });
    expect(await e).toBeNull();
    expect(order).toEqual([
      "plugin:osc|listen_start:start",
      "plugin:osc|listen_start:end",
      "plugin:osc|listen_start:start",
      "plugin:osc|listen_start:end",
      "plugin:osc|listen_stop:start",
      "plugin:osc|listen_stop:end",
    ]);
    expect(invokeMock.mock.calls[1][1]).toEqual({ options: { ...opts, port: 9002 } });
  });

  it("stops a receiver left over from an earlier webview session on the first sync", async () => {
    invokeMock.mockResolvedValue(undefined);
    const c = new OscReceiverClient();
    await c.sync(null);
    await c.sync(null);
    expect(invokeMock.mock.calls.map((x) => x[0])).toEqual(["plugin:osc|listen_stop"]);
  });

  it("retries a failed start on the next sync", async () => {
    invokeMock.mockRejectedValueOnce("port busy").mockResolvedValue({ running: true });
    const c = new OscReceiverClient();
    const opts = listenOptionsFor(base)!;
    await expect(c.sync(opts)).rejects.toBe("port busy");
    await expect(c.sync(opts)).resolves.toEqual({ running: true });
  });

  it("routes the three events to the handlers and unsubscribes on dispose", async () => {
    const handlers: Record<string, (e: { payload: unknown }) => void> = {};
    const unlisten = vi.fn();
    listenMock.mockImplementation(async (name: string, cb: (e: { payload: unknown }) => void) => {
      handlers[name] = cb;
      return unlisten;
    });
    invokeMock.mockResolvedValue(undefined);
    const got: string[] = [];
    const c = new OscReceiverClient();
    await c.attach({
      onMuteSelf: (e) => got.push(`mute:${e.muted}`),
      onStatus: (s) => got.push(`status:${s.mode}`),
      onVrchat: (e) => got.push(`vrchat:${e.present}`),
    });
    handlers[OSC_EVENTS.muteSelf]({ payload: { muted: true, origin: "osc" } });
    handlers[OSC_EVENTS.receiver]({ payload: { mode: "oscQuery" } });
    handlers[OSC_EVENTS.vrchat]({ payload: { present: false } });
    expect(got).toEqual(["mute:true", "status:oscQuery", "vrchat:false"]);
    await c.dispose();
    expect(unlisten).toHaveBeenCalledTimes(3);
    expect(invokeMock).toHaveBeenCalledWith("plugin:osc|listen_stop");
  });
});
