import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import { proxy } from "valtio";
import type { RustCatalogEntry, RustModelEntryStatus, RustModelProgress } from "@/types";

type Handler = (ev: { payload: unknown }) => void;
const listeners = new Map<string, Handler>();
const order: string[] = [];
const invoke = vi.fn();

vi.mock("@tauri-apps/api/event", () => ({
  listen: async (name: string, cb: Handler) => {
    order.push(`listen ${name}`);
    listeners.set(name, cb);
    return () => listeners.delete(name);
  },
}));
vi.mock("@tauri-apps/api/core", () => ({ invoke: (...args: unknown[]) => invoke(...args) }));

const CMD = (name: string) => `plugin:model-store|${name}`;

const catalog: RustCatalogEntry[] = [
  {
    id: "whisper/tiny",
    family: "whisper",
    kind: "model",
    version: "1",
    license: "MIT",
    status: "published",
    files: [{ path: "ggml-tiny.bin", size: 100, sha256: "a".repeat(64), sources: [] }],
  },
  {
    id: "vosk/en",
    family: "vosk",
    kind: "model",
    version: "0.15",
    license: "Apache-2.0",
    status: "published",
    files: [{ path: "vosk.zip", size: 50, sha256: "b".repeat(64), sources: [] }],
  },
];

const st = (id: string, state: RustModelEntryStatus["state"], extra: Partial<RustModelEntryStatus> = {}): RustModelEntryStatus => ({
  id,
  state,
  installed_version: null,
  installed_path: null,
  bytes_on_disk: 0,
  adopted: false,
  error: null,
  ...extra,
});

const disk = { models_root: "C:/models", total_bytes: 0, per_family: {}, free_bytes: 5e9 };

type Routes = Record<string, (args: Record<string, unknown> | undefined) => unknown>;
let routes: Routes;

function route(extra: Routes = {}) {
  routes = {
    [CMD("get_catalog")]: () => catalog,
    [CMD("get_entry_states")]: () => [st("whisper/tiny", "not_installed"), st("vosk/en", "installed", { bytes_on_disk: 50 })],
    [CMD("get_disk_usage")]: () => disk,
    [CMD("set_store_config")]: () => undefined,
    ...extra,
  };
  invoke.mockImplementation(async (cmd: string, args?: Record<string, unknown>) => {
    order.push(cmd);
    const h = routes[cmd];
    if (!h) throw new Error(`unexpected ${cmd}`);
    return h(args);
  });
}

function emit(name: string, payload: unknown) {
  listeners.get(name)!({ payload });
}

const flush = () => new Promise((r) => setTimeout(r, 0));

function stubWindow(settings: Partial<{ modelMirror: string; modelMirrorCustomUrl: string; modelMaxConcurrentDownloads: number }> = {}, app = true) {
  const state = proxy({ modelMirror: "auto", modelMirrorCustomUrl: "", modelMaxConcurrentDownloads: 1, ...settings });
  vi.stubGlobal("window", { Config: { isApp: () => app }, ApiServer: { state } });
  return state;
}

async function start(settings?: Parameters<typeof stubWindow>[0]) {
  const root = stubWindow(settings);
  const mod = await import("../index");
  const s = new mod.default();
  await s.init();
  await s.load();
  await flush();
  return { s, root, mod };
}

describe("Service_Models", () => {
  beforeEach(() => {
    listeners.clear();
    order.length = 0;
    invoke.mockReset();
    route();
  });
  afterEach(() => {
    vi.useRealTimers();
    vi.unstubAllGlobals();
  });

  it("listens before loading, then loads the catalog, states and disk usage and pushes the config", async () => {
    const { s, mod } = await start();
    expect(order.indexOf(`listen ${mod.MODELS_EVENTS.state}`)).toBeLessThan(order.indexOf(CMD("get_entry_states")));
    expect(order.indexOf(`listen ${mod.MODELS_EVENTS.progress}`)).toBeLessThan(order.indexOf(CMD("get_catalog")));
    expect(s.state.loaded).toBe(true);
    expect(s.state.catalog.map((e) => e.id)).toEqual(["whisper/tiny", "vosk/en"]);
    expect(s.state.states["vosk/en"].state).toBe("installed");
    expect(s.state.disk?.free_bytes).toBe(5e9);
    expect(invoke).toHaveBeenCalledWith(CMD("set_store_config"), { config: { mirror: "auto", custom_url: null, max_concurrent: 1 } });
    s.dispose();
    expect(listeners.size).toBe(0);
  });

  it("does nothing outside the desktop app and never rejects on IPC failures", async () => {
    stubWindow({}, false);
    const { default: Service_Models } = await import("../index");
    const web = new Service_Models();
    await web.init();
    expect(invoke).not.toHaveBeenCalled();

    stubWindow();
    route({ [CMD("get_catalog")]: () => Promise.reject("the store is unavailable") });
    const s = new Service_Models();
    await expect(s.init()).resolves.toBeUndefined();
    await s.load();
    expect(s.state.loaded).toBe(false);
    expect(s.state.error).toBe("the store is unavailable");
    s.dispose();
  });

  it("keeps a state event that arrived while a reload was in flight", async () => {
    const { s, mod } = await start();
    let release: (v: unknown) => void = () => {};
    routes[CMD("get_entry_states")] = () => new Promise((r) => (release = r));
    const reload = s.refresh();
    await flush();
    emit(mod.MODELS_EVENTS.state, st("whisper/tiny", "downloading"));
    release([st("whisper/tiny", "not_installed"), st("vosk/en", "installed")]);
    await reload;
    expect(s.state.states["whisper/tiny"].state).toBe("downloading");
    expect(s.state.states["vosk/en"].state).toBe("installed");
    s.dispose();
  });

  it("tracks progress while downloading and drops it when the install settles", async () => {
    const { s, mod } = await start();
    const p: RustModelProgress = { id: "whisper/tiny", file: "ggml-tiny.bin", bytes_done: 40, bytes_total: 100, bytes_per_sec: 10, mirror: "huggingface", attempt: 1 };
    emit(mod.MODELS_EVENTS.state, st("whisper/tiny", "queued"));
    emit(mod.MODELS_EVENTS.state, st("whisper/tiny", "downloading"));
    emit(mod.MODELS_EVENTS.progress, p);
    expect(s.state.progress["whisper/tiny"]).toMatchObject({ bytesDone: 40, fileBytes: { "ggml-tiny.bin": 40 } });
    emit(mod.MODELS_EVENTS.state, st("whisper/tiny", "installed", { bytes_on_disk: 100 }));
    expect(s.state.progress["whisper/tiny"]).toBeUndefined();
    emit(mod.MODELS_EVENTS.progress, { ...p, bytes_done: 100 }); // late event of the finished download
    expect(s.state.progress["whisper/tiny"]).toBeUndefined();
    s.dispose();
  });

  it("marks installs pending and reports rejected ones per entry", async () => {
    const { s } = await start();
    let release: () => void = () => {};
    routes[CMD("install_entries")] = () => new Promise<void>((r) => (release = r));
    const ok = s.install(["whisper/tiny", "whisper/tiny", " "]);
    expect(s.state.pending["whisper/tiny"]).toBe("install");
    release();
    expect(await ok).toBe(true);
    expect(invoke).toHaveBeenCalledWith(CMD("install_entries"), { ids: ["whisper/tiny"] });
    expect(s.state.pending["whisper/tiny"]).toBeUndefined();

    routes[CMD("install_entries")] = () => Promise.reject("unknown model id nope");
    expect(await s.install(["vosk/en"])).toBe(false);
    expect(s.state.opErrors["vosk/en"]).toBe("unknown model id nope");
    s.dismissError("vosk/en");
    expect(s.state.opErrors["vosk/en"]).toBeUndefined();
    expect(await s.install([])).toBe(true);
    s.dispose();
  });

  it("applies the status returned by remove / verify / import and reloads the disk usage", async () => {
    const { s } = await start();
    vi.useFakeTimers();
    routes[CMD("remove_entry")] = (a) => st(String(a?.id), "not_installed");
    routes[CMD("verify_entry")] = (a) => st(String(a?.id), "failed", { error: "ggml-tiny.bin is corrupt" });
    routes[CMD("import_local")] = (a) => st(String(a?.id), "installed", { bytes_on_disk: 100 });
    const diskCalls = () => invoke.mock.calls.filter(([c]) => c === CMD("get_disk_usage")).length;
    const before = diskCalls();

    expect(await s.remove("vosk/en")).toBe(true);
    expect(s.state.states["vosk/en"].state).toBe("not_installed");
    expect(await s.verify("whisper/tiny")).toBe(false);
    expect(s.state.states["whisper/tiny"].error).toBe("ggml-tiny.bin is corrupt");
    expect(await s.importLocal("whisper/tiny", "  C:/Downloads/ggml-tiny.bin ")).toBe(true);
    expect(invoke).toHaveBeenCalledWith(CMD("import_local"), { id: "whisper/tiny", path: "C:/Downloads/ggml-tiny.bin" });
    expect(await s.importLocal("whisper/tiny", "  ")).toBe(false);

    await vi.advanceTimersByTimeAsync(1000);
    expect(diskCalls()).toBe(before + 1); // one coalesced reload
    s.dispose();
  });

  it("does not let an operation's result overwrite a newer state event", async () => {
    const { s, mod } = await start();
    let release: (v: unknown) => void = () => {};
    routes[CMD("remove_entry")] = () => new Promise((r) => (release = r));
    const op = s.remove("vosk/en");
    emit(mod.MODELS_EVENTS.state, st("vosk/en", "queued")); // someone re-queued it meanwhile
    release(st("vosk/en", "not_installed"));
    await op;
    expect(s.state.states["vosk/en"].state).toBe("queued");
    s.dispose();
  });

  it("does not record a cancelled import as an error", async () => {
    const { s } = await start();
    routes[CMD("import_local")] = () => Promise.reject("cancelled");
    expect(await s.importLocal("whisper/tiny", "C:/Downloads/ggml-tiny.bin")).toBe(false);
    expect(s.state.opErrors["whisper/tiny"]).toBeUndefined();
    expect(s.state.pending["whisper/tiny"]).toBeUndefined();
    routes[CMD("import_local")] = () => Promise.reject("C:/Downloads/ggml-tiny.bin does not exist");
    expect(await s.importLocal("whisper/tiny", "C:/Downloads/ggml-tiny.bin")).toBe(false);
    expect(s.state.opErrors["whisper/tiny"]).toBe("C:/Downloads/ggml-tiny.bin does not exist");
    s.dispose();
  });

  it("re-fetches the states when an install queued nothing, clearing a stale update", async () => {
    // Loaded before startup maintenance re-recorded vosk/en at the catalog version 0.15.
    route({
      [CMD("get_entry_states")]: () => [st("whisper/tiny", "not_installed"), st("vosk/en", "installed", { installed_version: "0.14" })],
      [CMD("install_entries")]: () => undefined, // up to date: no job, no event
    });
    const { s, mod } = await start();
    expect(mod.hasUpdate(catalog[1], s.state.states["vosk/en"])).toBe(true);
    routes[CMD("get_entry_states")] = () => [st("whisper/tiny", "not_installed"), st("vosk/en", "installed", { installed_version: "0.15" })];
    const fetches = () => invoke.mock.calls.filter(([c]) => c === CMD("get_entry_states")).length;
    const before = fetches();

    vi.useFakeTimers();
    expect(await s.install(["vosk/en"])).toBe(true);
    await vi.advanceTimersByTimeAsync(mod.INSTALL_EVENT_GRACE_MS - 10);
    expect(fetches()).toBe(before);
    await vi.advanceTimersByTimeAsync(20);
    expect(fetches()).toBe(before + 1);
    expect(s.state.states["vosk/en"].installed_version).toBe("0.15");
    expect(mod.hasUpdate(catalog[1], s.state.states["vosk/en"])).toBe(false);

    // An install that produced state events needs no re-fetch.
    routes[CMD("install_entries")] = () => {
      emit(mod.MODELS_EVENTS.state, st("whisper/tiny", "queued"));
      return undefined;
    };
    expect(await s.install(["whisper/tiny"])).toBe(true);
    await vi.advanceTimersByTimeAsync(mod.INSTALL_EVENT_GRACE_MS + 10);
    expect(fetches()).toBe(before + 1);
    expect(s.state.states["whisper/tiny"].state).toBe("queued");
    s.dispose();
  });

  it("re-fetches states without overwriting a newer event", async () => {
    const { s, mod } = await start();
    const fetches = () => invoke.mock.calls.filter(([c]) => c === CMD("get_entry_states")).length;
    const before = fetches();
    let release: (v: unknown) => void = () => {};
    routes[CMD("get_entry_states")] = () => new Promise((r) => (release = r));
    const refresh = s.refreshStates();
    await flush();
    emit(mod.MODELS_EVENTS.state, st("whisper/tiny", "downloading"));
    release([st("whisper/tiny", "not_installed"), st("vosk/en", "not_installed")]);
    await refresh;
    expect(s.state.states["whisper/tiny"].state).toBe("downloading");
    expect(s.state.states["vosk/en"].state).toBe("not_installed");
    expect(fetches()).toBe(before + 1);
    expect(invoke.mock.calls.filter(([c]) => c === CMD("get_catalog"))).toHaveLength(1); // states only

    // Calls during a fetch share it and run one more pass.
    routes[CMD("get_entry_states")] = () => new Promise((r) => (release = r));
    const first = s.refreshStates();
    const second = s.refreshStates();
    await flush();
    routes[CMD("get_entry_states")] = () => [st("whisper/tiny", "installed"), st("vosk/en", "installed")];
    release([st("whisper/tiny", "downloading"), st("vosk/en", "not_installed")]);
    await Promise.all([first, second]);
    expect(fetches()).toBe(before + 3);
    expect(s.state.states["whisper/tiny"].state).toBe("installed");
    s.dispose();
  });

  it("cancels, keeping the install's pending flag", async () => {
    const { s } = await start();
    routes[CMD("cancel_install")] = () => undefined;
    expect(await s.cancel("whisper/tiny")).toBe(true);
    expect(invoke).toHaveBeenCalledWith(CMD("cancel_install"), { id: "whisper/tiny" });
    routes[CMD("cancel_install")] = () => Promise.reject("unknown model id x");
    expect(await s.cancel("x")).toBe(false);
    expect(s.state.opErrors.x).toBe("unknown model id x");
    s.dispose();
  });

  it("holds back an unusable custom mirror and pushes the fixed URL after typing stops", async () => {
    const { s, root } = await start({ modelMirror: "custom" });
    const pushes = () => invoke.mock.calls.filter(([c]) => c === CMD("set_store_config"));
    expect(s.state.configProblem).toBe("empty");
    expect(pushes()).toHaveLength(0);

    vi.useFakeTimers();
    root.modelMirrorCustomUrl = "https://mirror.exa";
    await vi.advanceTimersByTimeAsync(100);
    root.modelMirrorCustomUrl = "https://mirror.example.com/sigil";
    await vi.advanceTimersByTimeAsync(100);
    expect(pushes()).toHaveLength(0); // still typing
    await vi.advanceTimersByTimeAsync(600);
    expect(pushes()).toHaveLength(1);
    expect(pushes()[0][1]).toEqual({ config: { mirror: "custom", custom_url: "https://mirror.example.com/sigil", max_concurrent: 1 } });
    expect(s.state.configProblem).toBeNull();

    // Other settings apply at once; Rust errors are shown.
    routes[CMD("set_store_config")] = () => Promise.reject("the custom mirror URL must not contain credentials");
    root.modelMaxConcurrentDownloads = 2;
    await vi.advanceTimersByTimeAsync(0);
    expect(pushes()).toHaveLength(2);
    expect(pushes()[1][1]).toMatchObject({ config: { max_concurrent: 2 } });
    expect(s.state.configError).toBe("the custom mirror URL must not contain credentials");
    s.dispose();
  });
});
