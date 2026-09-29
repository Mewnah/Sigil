import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";

type Handler = (ev: { payload: unknown }) => void;
const handlers = new Map<string, Handler>();
const invoke = vi.fn();

vi.mock("@tauri-apps/api/event", () => ({
  listen: async (name: string, cb: Handler) => {
    handlers.set(name, cb);
    return () => handlers.delete(name);
  },
}));
vi.mock("@tauri-apps/api/core", () => ({ invoke: (...args: unknown[]) => invoke(...args) }));

const info = (name: string) => ({
  gpus: [{ index: 0, name, software: false }],
  primary_gpu: 0,
  nvidia_driver: "616.56",
  cuda_driver_version: "13.4",
  eps: {},
  cpu: { brand: "cpu", physical_cores: 8, logical_cores: 16, avx2: true, avx512: false, vnni: false },
  total_ram_mb: 32_768,
  probed_at_unix_ms: 1,
});

describe("Service_Sysinfo", () => {
  beforeEach(() => {
    handlers.clear();
    invoke.mockReset();
    vi.stubGlobal("window", { Config: { isApp: () => true } });
  });
  afterEach(() => {
    vi.unstubAllGlobals();
  });

  it("loads the cached probe in the background, once, and follows re-probe events", async () => {
    let release: (v: unknown) => void = () => {};
    invoke.mockImplementation(() => new Promise((r) => (release = r)));
    const { default: Service_Sysinfo, SYSINFO_EVENT } = await import("../index");
    const s = new Service_Sysinfo();

    await s.init(); // resolves without waiting for the probe
    expect(s.state.loading).toBe(true);
    expect(invoke).toHaveBeenCalledWith("plugin:sysinfo|get_system_info", { refresh: false });
    const again = s.load(); // shares the in-flight request
    expect(invoke).toHaveBeenCalledTimes(1);
    release(info("NVIDIA GeForce RTX 4090"));
    await again;
    expect(s.state.info?.gpus[0].name).toBe("NVIDIA GeForce RTX 4090");
    expect(s.state.loading).toBe(false);

    await s.load(); // already loaded: no new request
    expect(invoke).toHaveBeenCalledTimes(1);

    handlers.get(SYSINFO_EVENT)!({ payload: info("NVIDIA GeForce RTX 5090") });
    expect(s.state.info?.gpus[0].name).toBe("NVIDIA GeForce RTX 5090");
    s.dispose();
    expect(handlers.has(SYSINFO_EVENT)).toBe(false);
  });

  it("re-probes on refresh and reports errors without rejecting", async () => {
    const { default: Service_Sysinfo } = await import("../index");
    const s = new Service_Sysinfo();
    invoke.mockRejectedValueOnce("the graphics driver did not answer");
    await s.load();
    expect(s.state.error).toBe("the graphics driver did not answer");
    expect(s.state.info).toBeNull();

    invoke.mockResolvedValueOnce(info("NVIDIA GeForce RTX 4090"));
    await s.refresh();
    expect(invoke).toHaveBeenLastCalledWith("plugin:sysinfo|get_system_info", { refresh: true });
    expect(s.state.error).toBeNull();
    expect(s.state.info?.gpus[0].name).toBe("NVIDIA GeForce RTX 4090");
  });

  it("stays loading until every overlapping request finished", async () => {
    const releases: Array<(v: unknown) => void> = [];
    invoke.mockImplementation(() => new Promise((r) => releases.push(r)));
    const { default: Service_Sysinfo } = await import("../index");
    const s = new Service_Sysinfo();
    invoke.mockImplementationOnce(async () => info("first"));
    await s.load();
    const a = s.refresh();
    const b = s.refresh();
    expect(s.state.loading).toBe(true);
    releases[0](info("second"));
    await a;
    expect(s.state.loading).toBe(true); // b still running
    releases[1](info("third"));
    await b;
    expect(s.state.loading).toBe(false);
    expect(s.state.info?.gpus[0].name).toBe("third");
  });

  it("does nothing outside the desktop app", async () => {
    vi.stubGlobal("window", { Config: { isApp: () => false } });
    const { default: Service_Sysinfo } = await import("../index");
    const s = new Service_Sysinfo();
    await s.init();
    expect(invoke).not.toHaveBeenCalled();
    expect(handlers.size).toBe(0);
  });
});
