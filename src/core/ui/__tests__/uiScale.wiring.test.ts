/**
 * Interface size on the real `ApiServer`: the settings file is read, and `--uiscale` ends up on the document root. Only the
 * edges are faked (the app-data folder, the Tauri window and monitor, the document, the other services' start-up).
 */
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";

const disk = vi.hoisted(() => ({ files: new Map<string, string>() }));
vi.mock("@tauri-apps/plugin-fs", () => ({
  exists: async (path: string) => disk.files.has(path),
  readFile: async (path: string) => {
    const text = disk.files.get(path);
    if (text === undefined) throw new Error(`ENOENT ${path}`);
    return new TextEncoder().encode(text);
  },
  writeFile: async (path: string, data: Uint8Array) => void disk.files.set(path, new TextDecoder().decode(data)),
  copyFile: async (from: string, to: string) => void disk.files.set(to, disk.files.get(from) ?? ""),
  rename: async (from: string, to: string) => {
    disk.files.set(to, disk.files.get(from) ?? "");
    disk.files.delete(from);
  },
  mkdir: async () => { },
}));

const tauri = vi.hoisted(() => ({
  currentMonitor: vi.fn(),
  onMoved: vi.fn(),
  onScaleChanged: vi.fn(),
  onResized: vi.fn(),
  isMaximized: vi.fn(),
  isMinimized: vi.fn(),
  isFullscreen: vi.fn(),
  innerSize: vi.fn(),
  outerPosition: vi.fn(),
  outerSize: vi.fn(),
  setPosition: vi.fn(),
  setSize: vi.fn(),
}));
vi.mock("@tauri-apps/api/core", async (importOriginal) => ({ ...(await importOriginal<object>()), isTauri: () => true }));
vi.mock("@tauri-apps/api/window", async (importOriginal) => ({
  ...(await importOriginal<object>()),
  currentMonitor: tauri.currentMonitor,
  getCurrentWindow: () => ({
    onMoved: tauri.onMoved,
    onScaleChanged: tauri.onScaleChanged,
    onResized: tauri.onResized,
    isMaximized: tauri.isMaximized,
    isMinimized: tauri.isMinimized,
    isFullscreen: tauri.isFullscreen,
    innerSize: tauri.innerSize,
    outerPosition: tauri.outerPosition,
    outerSize: tauri.outerSize,
    setPosition: tauri.setPosition,
    setSize: tauri.setSize,
  }),
}));
vi.mock("@/i18n", async (importOriginal) => ({
  ...(await importOriginal<object>()),
  initI18n: vi.fn(async () => { }),
  changeLanguage: vi.fn(async () => { }),
}));

import { PhysicalPosition, PhysicalSize } from "@tauri-apps/api/dpi";
import ApiServer from "@/core";
import { UI_SCALE_DEBOUNCE_MS } from "../uiScale";

const SETTINGS = "user/settings";
const MON_4K_100 = { size: { width: 3840, height: 2160 }, scaleFactor: 1, workArea: { position: { x: 0, y: 0 }, size: { width: 3840, height: 2112 } } };
const MON_4K_125 = { size: { width: 3840, height: 2160 }, scaleFactor: 1.25, workArea: { position: { x: 0, y: 0 }, size: { width: 3840, height: 2100 } } };
const MON_1080P = { size: { width: 1920, height: 1080 }, scaleFactor: 1, workArea: { position: { x: 0, y: 0 }, size: { width: 1920, height: 1032 } } };

let root: ReturnType<typeof fakeRoot>;
function fakeRoot() {
  const props = new Map<string, string>();
  const log: string[] = [];
  return {
    props,
    log,
    style: {
      setProperty: (name: string, value: string | null) => {
        props.set(name, String(value));
        log.push(`set ${name}=${value}`);
      },
      removeProperty: (name: string) => {
        props.delete(name);
        log.push(`remove ${name}`);
        return "";
      },
    },
    getBoundingClientRect: () => void log.push("layout"),
  };
}

/** What `window.ApiShared` is asked during `init`; `startServer` notes the Interface size it finds in the document. */
let sizeWhenServerStarted: string | undefined;
let client = false;
let pageListeners: Record<string, () => void>;

function api() {
  const server = new ApiServer();
  // The rest of the host's start-up is not under test here.
  vi.spyOn(server as unknown as { getHostServiceInitTasks(): unknown[] }, "getHostServiceInitTasks").mockReturnValue([]);
  vi.spyOn(server as unknown as { runSpeechDefaultOffers(): Promise<void> }, "runSpeechDefaultOffers").mockResolvedValue(undefined);
  return server;
}

beforeEach(() => {
  disk.files.clear();
  root = fakeRoot();
  client = false;
  pageListeners = {};
  sizeWhenServerStarted = undefined;
  tauri.currentMonitor.mockReset().mockResolvedValue(MON_4K_100);
  tauri.onMoved.mockReset().mockResolvedValue(() => { });
  tauri.onScaleChanged.mockReset().mockResolvedValue(() => { });
  tauri.onResized.mockReset().mockResolvedValue(() => { });
  // The window the default profile opens: 800 x 600, not maximized.
  tauri.isMaximized.mockReset().mockResolvedValue(false);
  tauri.isMinimized.mockReset().mockResolvedValue(false);
  tauri.isFullscreen.mockReset().mockResolvedValue(false);
  tauri.innerSize.mockReset().mockResolvedValue(new PhysicalSize(800, 600));
  tauri.outerPosition.mockReset().mockResolvedValue(new PhysicalPosition(100, 100));
  tauri.outerSize.mockReset().mockResolvedValue(new PhysicalSize(800, 600));
  tauri.setPosition.mockReset().mockResolvedValue(undefined);
  tauri.setSize.mockReset().mockResolvedValue(undefined);
  vi.stubGlobal("window", globalThis);
  vi.stubGlobal("document", { documentElement: root, body: { setAttribute: () => { } } });
  Object.assign(globalThis, {
    addEventListener: (type: string, listener: () => void) => void (pageListeners[type] = listener),
    Config: { isClient: () => client, isApp: () => true, isServer: () => !client },
    ApiShared: {
      peer: { startServer: vi.fn(async () => { sizeWhenServerStarted = root.props.get("--uiscale"); }) },
      pubsub: { setExternalSttHandler: vi.fn(), setExternalMicLifecycleHandler: vi.fn(), setTextEmoteEnricher: vi.fn() },
    },
  });
});

afterEach(() => {
  vi.useRealTimers();
  vi.unstubAllGlobals();
  delete (globalThis as Record<string, unknown>).addEventListener;
  delete (globalThis as Record<string, unknown>).Config;
  delete (globalThis as Record<string, unknown>).ApiShared;
});

describe("Interface size when the host starts", () => {
  it("a new install on a 4K monitor at 100% is drawn at 1.5 from the first paint: the size is in the document before anything else starts", async () => {
    const server = api();
    await server.init();
    expect(root.props.get("--uiscale")).toBe("1.5");
    expect(sizeWhenServerStarted).toBe("1.5");
    // The first write turns the root font-size transition off, so the wizard does not grow into its size.
    expect(root.log.slice(0, 4)).toEqual(["set transition=none", "set --uiscale=1.5", "layout", "remove transition"]);
    expect(server.state.uiScaleAuto).toBe(true);
    expect(server.state.uiScale).toBe(1); // the manual size is not overwritten by what Auto picked
  });

  it("is 1 on a 1080p monitor and at 150% on a 4K one (nothing changes for those users)", async () => {
    tauri.currentMonitor.mockResolvedValue(MON_1080P);
    await api().init();
    expect(root.props.get("--uiscale")).toBe("1");

    root = fakeRoot();
    vi.stubGlobal("document", { documentElement: root, body: { setAttribute: () => { } } });
    disk.files.clear();
    tauri.currentMonitor.mockResolvedValue({ ...MON_4K_100, scaleFactor: 1.5 });
    await api().init();
    expect(root.props.get("--uiscale")).toBe("1");
  });

  it("is 1 when the monitor cannot be read", async () => {
    tauri.currentMonitor.mockRejectedValue(new Error("no window"));
    vi.spyOn(console, "warn").mockImplementation(() => { });
    await api().init();
    expect(root.props.get("--uiscale")).toBe("1");
  });

  it("a profile that had picked L keeps L on any monitor", async () => {
    disk.files.set(SETTINGS, JSON.stringify({ id: "me", firstRunDecided: true, onboardingComplete: true, uiScale: 1.2 }));
    const server = api();
    await server.init();
    expect(root.props.get("--uiscale")).toBe("1.2");
    expect(server.state.uiScaleAuto).toBe(false);
  });

  it("a profile at the default M moves to Auto", async () => {
    disk.files.set(SETTINGS, JSON.stringify({ id: "me", firstRunDecided: true, onboardingComplete: true, uiScale: 1 }));
    await api().init();
    expect(root.props.get("--uiscale")).toBe("1.5");
  });

  it("the /client pages never read the monitor or set the size (init does nothing there)", async () => {
    client = true;
    await api().init();
    expect(tauri.currentMonitor).not.toHaveBeenCalled();
    expect(root.log).toEqual([]);
  });
});

describe("the window when Auto raises the Interface size", () => {
  const SAVED_M = { id: "me", firstRunDecided: true, onboardingComplete: true, uiScale: 1 };

  it("a profile at M with the default 800 x 600 window on a 4K monitor at 100% gets a window that holds 1.5: 1200 x 900", async () => {
    disk.files.set(SETTINGS, JSON.stringify(SAVED_M));
    await api().init();
    await vi.waitFor(() => expect(tauri.setSize).toHaveBeenCalledTimes(1));
    expect(tauri.setSize).toHaveBeenCalledWith(new PhysicalSize(1200, 900));
    expect(tauri.setPosition).toHaveBeenCalledWith(new PhysicalPosition(100, 100));
  });

  it("a window that is already big enough, a maximized one and a 1080p monitor are left alone", async () => {
    disk.files.set(SETTINGS, JSON.stringify(SAVED_M));
    tauri.innerSize.mockResolvedValue(new PhysicalSize(1500, 1100));
    tauri.outerSize.mockResolvedValue(new PhysicalSize(1500, 1100));
    await api().init();

    tauri.innerSize.mockResolvedValue(new PhysicalSize(800, 600));
    tauri.outerSize.mockResolvedValue(new PhysicalSize(800, 600));
    tauri.isMaximized.mockResolvedValue(true);
    await api().init();

    tauri.isMaximized.mockResolvedValue(false);
    tauri.currentMonitor.mockResolvedValue(MON_1080P);
    await api().init();

    await vi.waitFor(() => expect(tauri.innerSize).toHaveBeenCalledTimes(1)); // only the first run got as far as the size
    await new Promise(resolve => setTimeout(resolve, 20));
    expect(tauri.setSize).not.toHaveBeenCalled();
    expect(tauri.setPosition).not.toHaveBeenCalled();
  });

  it("a profile that picked L keeps its window", async () => {
    disk.files.set(SETTINGS, JSON.stringify({ ...SAVED_M, uiScale: 1.2 }));
    await api().init();
    await new Promise(resolve => setTimeout(resolve, 20));
    expect(tauri.isMaximized).not.toHaveBeenCalled();
    expect(tauri.setSize).not.toHaveBeenCalled();
  });
});

describe("Interface size while the host runs", () => {
  async function started() {
    tauri.currentMonitor.mockResolvedValue(MON_4K_125); // Auto is 1.2 here, apart from every manual chip
    const server = api();
    await server.init();
    return server;
  }

  it("picking S, M, L or XL turns Auto off and keeps that size; picking Auto turns it on and keeps the manual size", async () => {
    const server = await started();
    expect(root.props.get("--uiscale")).toBe("1.2");

    server.changeScale(1.4);
    expect(root.props.get("--uiscale")).toBe("1.4");
    expect(server.state.uiScaleAuto).toBe(false);
    expect(server.state.uiScale).toBe(1.4);

    server.changeScale(0.8);
    expect(root.props.get("--uiscale")).toBe("0.8");

    server.changeScaleAuto();
    expect(root.props.get("--uiscale")).toBe("1.2");
    expect(server.state.uiScaleAuto).toBe(true);
    expect(server.state.uiScale).toBe(0.8); // kept for when Auto is turned off again

    server.changeScale(1);
    expect(root.props.get("--uiscale")).toBe("1");
    expect(server.state.uiScaleAuto).toBe(false);
  });

  it("clamps a manual size", async () => {
    const server = await started();
    server.changeScale(9);
    expect(server.state.uiScale).toBe(1.5);
    expect(root.props.get("--uiscale")).toBe("1.5");
    server.changeScale(0.1);
    expect(server.state.uiScale).toBe(0.8);
    expect(root.props.get("--uiscale")).toBe("0.8");
  });

  it("shows the size Auto picked to the Settings chip, and keeps it up to date while a manual size is in use", async () => {
    const server = await started();
    expect(server.uiScale.detected.auto).toBe(1.2);
    server.changeScale(1);
    expect(server.uiScale.detected.auto).toBe(1.2);
  });

  it("follows the window to another monitor while Auto is on, but not when a manual size is picked", async () => {
    const server = await started();
    const moved = tauri.onMoved.mock.calls[0][0] as () => void;
    vi.useFakeTimers();

    tauri.currentMonitor.mockResolvedValue(MON_1080P);
    moved();
    await vi.advanceTimersByTimeAsync(UI_SCALE_DEBOUNCE_MS);
    expect(root.props.get("--uiscale")).toBe("1");

    server.changeScale(1.4);
    tauri.currentMonitor.mockResolvedValue(MON_4K_100);
    moved();
    await vi.advanceTimersByTimeAsync(UI_SCALE_DEBOUNCE_MS);
    expect(root.props.get("--uiscale")).toBe("1.4");
    expect(server.uiScale.detected.auto).toBe(1.5);

    server.changeScaleAuto();
    expect(root.props.get("--uiscale")).toBe("1.5");
  });

  it("stops following the window when the page is unloaded", async () => {
    const stopMoved = vi.fn();
    const stopScale = vi.fn();
    tauri.onMoved.mockResolvedValue(stopMoved);
    tauri.onScaleChanged.mockResolvedValue(stopScale);
    await started();
    const moved = tauri.onMoved.mock.calls[0][0] as () => void;
    expect(stopMoved).not.toHaveBeenCalled();
    pageListeners.pagehide();
    expect(stopMoved).toHaveBeenCalledTimes(1);
    expect(stopScale).toHaveBeenCalledTimes(1);

    vi.useFakeTimers();
    tauri.currentMonitor.mockResolvedValue(MON_1080P);
    tauri.currentMonitor.mockClear();
    moved();
    await vi.advanceTimersByTimeAsync(UI_SCALE_DEBOUNCE_MS * 2);
    expect(tauri.currentMonitor).not.toHaveBeenCalled();
    expect(root.props.get("--uiscale")).toBe("1.2");
  });

  it("follows a change of the desktop resolution of a window that stays where it is (a maximized window is resized, not moved)", async () => {
    tauri.currentMonitor.mockResolvedValue(MON_1080P);
    const server = api();
    await server.init();
    expect(root.props.get("--uiscale")).toBe("1");
    const resized = tauri.onResized.mock.calls[0][0] as () => void;
    vi.useFakeTimers();
    tauri.currentMonitor.mockResolvedValue(MON_4K_100);
    resized();
    await vi.advanceTimersByTimeAsync(UI_SCALE_DEBOUNCE_MS);
    expect(root.props.get("--uiscale")).toBe("1.5");
  });

  it("follows a change of the Windows display scale", async () => {
    await started();
    const scaleChanged = tauri.onScaleChanged.mock.calls[0][0] as () => void;
    vi.useFakeTimers();
    tauri.currentMonitor.mockResolvedValue({ ...MON_4K_125, scaleFactor: 1 });
    scaleChanged();
    await vi.advanceTimersByTimeAsync(UI_SCALE_DEBOUNCE_MS);
    expect(root.props.get("--uiscale")).toBe("1.5");
  });
});
