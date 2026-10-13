import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import { PhysicalPosition, PhysicalSize } from "@tauri-apps/api/dpi";
import fixture from "../uiScale.fixture.json";

const tauri = vi.hoisted(() => ({
  isTauri: vi.fn(() => true),
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
// `dpi` (the size classes) imports from `core`, so the rest of `core` stays real.
vi.mock("@tauri-apps/api/core", async (importOriginal) => ({ ...(await importOriginal<object>()), isTauri: tauri.isTauri }));
vi.mock("@tauri-apps/api/window", () => ({
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

import {
  AUTO_UI_SCALE_MAX,
  AUTO_UI_SCALE_MIN,
  UI_SCALE_AUTO_CHIP,
  UI_SCALE_DEBOUNCE_MS,
  UI_SCALE_MANUAL_CHIPS,
  UI_SCALE_MAX,
  UI_SCALE_MIN,
  UI_SCALE_START_TIMEOUT_MS,
  UI_WINDOW_MIN_SIZE,
  UiScaleController,
  applyUiScaleToDocument,
  autoUiScale,
  clampManualUiScale,
  effectiveUiScale,
  formatUiScalePercent,
  tauriWindowDeps,
  uiScaleAutoHint,
  uiScaleChipOptions,
  uiScaleChipPick,
  uiScaleChipValue,
  windowGrowth,
  type MonitorMetrics,
  type UiScaleDeps,
  type WindowMetrics,
} from "../uiScale";

const MON_1080P: MonitorMetrics = { width: 1920, height: 1080, scaleFactor: 1, workArea: { x: 0, y: 0, width: 1920, height: 1032 } };
const MON_4K_100: MonitorMetrics = { width: 3840, height: 2160, scaleFactor: 1, workArea: { x: 0, y: 0, width: 3840, height: 2112 } };
const MON_4K_150: MonitorMetrics = { width: 3840, height: 2160, scaleFactor: 1.5, workArea: { x: 0, y: 0, width: 3840, height: 2088 } };
const MON_4K_125: MonitorMetrics = { width: 3840, height: 2160, scaleFactor: 1.25, workArea: { x: 0, y: 0, width: 3840, height: 2100 } };

/** A window without a frame (the app has none): inner and outer are the same rectangle. */
const win = (x: number, y: number, width: number, height: number): WindowMetrics => ({
  innerWidth: width,
  innerHeight: height,
  outerX: x,
  outerY: y,
  outerWidth: width,
  outerHeight: height,
});

describe("autoUiScale: the shared fixture (the Rust side reads the same file)", () => {
  it("has the cases the rule is checked against", () => {
    expect(fixture.cases.length).toBeGreaterThanOrEqual(17);
    expect(new Set(fixture.cases.map(c => c.name)).size).toBe(fixture.cases.length);
  });

  for (const c of fixture.cases) {
    it(`${c.name}: ${c.width} x ${c.height} at ${c.scaleFactor} gives ${c.expected}`, () => {
      expect(autoUiScale({ width: c.width, height: c.height, scaleFactor: c.scaleFactor })).toBe(c.expected);
    });
  }
});

describe("autoUiScale: the rule", () => {
  it("never leaves [1, 2] and lands on a multiple of 0.05", () => {
    for (const width of [640, 1366, 1920, 2560, 3000, 3440, 3840, 4096, 5120, 7680]) {
      for (const height of [480, 768, 1080, 1440, 1600, 2160, 2880, 4320]) {
        for (const scaleFactor of [1, 1.25, 1.5, 1.75, 2, 2.5, 3]) {
          const s = autoUiScale({ width, height, scaleFactor });
          expect(s, `${width}x${height}@${scaleFactor}`).toBeGreaterThanOrEqual(AUTO_UI_SCALE_MIN);
          expect(s, `${width}x${height}@${scaleFactor}`).toBeLessThanOrEqual(AUTO_UI_SCALE_MAX);
          expect(Math.abs(s * 20 - Math.round(s * 20)), `${width}x${height}@${scaleFactor}`).toBeLessThan(1e-9);
        }
      }
    }
  });

  it("is the same for a landscape and a portrait monitor", () => {
    expect(autoUiScale({ width: 2160, height: 3840, scaleFactor: 1 })).toBe(autoUiScale(MON_4K_100));
  });

  it("is limited by the tighter side (an ultrawide is sized by its height)", () => {
    expect(autoUiScale({ width: 5120, height: 1440, scaleFactor: 1 })).toBe(1);
    expect(autoUiScale({ width: 3840, height: 1600, scaleFactor: 1 })).toBe(1.1);
  });

  it("gives 1 for an unknown monitor and for sizes or scales that are not positive finite numbers", () => {
    expect(autoUiScale(null)).toBe(1);
    expect(autoUiScale(undefined)).toBe(1);
    for (const bad of [0, -1, NaN, Infinity, "3840" as unknown as number]) {
      expect(autoUiScale({ ...MON_4K_100, width: bad })).toBe(1);
      expect(autoUiScale({ ...MON_4K_100, height: bad })).toBe(1);
      expect(autoUiScale({ ...MON_4K_100, scaleFactor: bad })).toBe(1);
    }
  });
});

describe("manual size and the size in effect", () => {
  it("keeps a manual size inside 0.8 to 1.5 and reads anything that is not a number as M", () => {
    expect(clampManualUiScale(0.8)).toBe(0.8);
    expect(clampManualUiScale(1.4)).toBe(1.4);
    expect(clampManualUiScale(0.1)).toBe(UI_SCALE_MIN);
    expect(clampManualUiScale(3)).toBe(UI_SCALE_MAX);
    expect(clampManualUiScale(NaN)).toBe(1);
    expect(clampManualUiScale("1.2" as unknown as number)).toBe(1);
  });

  it("is the monitor's Auto size while Auto is on, whatever the manual size is", () => {
    expect(effectiveUiScale({ auto: true, manual: 0.8, monitor: MON_4K_100 })).toBe(1.5);
    expect(effectiveUiScale({ auto: true, manual: 1.4, monitor: MON_4K_125 })).toBe(1.2);
    expect(effectiveUiScale({ auto: true, manual: 1.4, monitor: MON_4K_150 })).toBe(1);
  });

  it("is 1 in Auto when the monitor is unknown", () => {
    expect(effectiveUiScale({ auto: true, manual: 1.4, monitor: null })).toBe(1);
    expect(effectiveUiScale({ auto: true, manual: 1.4, monitor: undefined })).toBe(1);
  });

  it("is the clamped manual size while Auto is off, whatever the monitor is", () => {
    expect(effectiveUiScale({ auto: false, manual: 1.2, monitor: MON_4K_100 })).toBe(1.2);
    expect(effectiveUiScale({ auto: false, manual: 1, monitor: MON_4K_100 })).toBe(1);
    expect(effectiveUiScale({ auto: false, manual: 9, monitor: MON_4K_100 })).toBe(UI_SCALE_MAX);
    expect(effectiveUiScale({ auto: false, manual: 0.8, monitor: null })).toBe(0.8);
  });
});

describe("Interface size chips", () => {
  const t = (key: string, options?: Record<string, string>) => (options ? `${key}|${JSON.stringify(options)}` : key);

  it("are Auto, S, M, L, XL in that order, and Auto is only labelled Auto (the size it picked is the line below the row)", () => {
    const chips = uiScaleChipOptions(t);
    expect(chips.map(c => c.value)).toEqual([UI_SCALE_AUTO_CHIP, 0.8, 1, 1.2, 1.4]);
    expect(chips.map(c => c.label)).toEqual([
      "settings.ui_scale_auto",
      "settings.ui_scale_s",
      "settings.ui_scale_m",
      "settings.ui_scale_l",
      "settings.ui_scale_x",
    ]);
  });

  it("say what Auto picked for the monitor, or that the monitor could not be read", () => {
    expect(uiScaleAutoHint({ auto: 1.5, known: true }, t)).toBe(`settings.ui_scale_auto_hint|{"size":"150%"}`);
    expect(uiScaleAutoHint({ auto: 1, known: true }, t)).toBe(`settings.ui_scale_auto_hint|{"size":"100%"}`);
    // A monitor that was never read gives 1, which must not read like a real 1080p answer.
    expect(uiScaleAutoHint({ auto: 1, known: false }, t)).toBe("settings.ui_scale_auto_unknown");
  });

  it("keep the S, M, L and XL values the settings have always used", () => {
    expect(UI_SCALE_MANUAL_CHIPS.map(c => c.value)).toEqual([0.8, 1, 1.2, 1.4]);
  });

  it("label the size as a percentage", () => {
    expect(formatUiScalePercent(1)).toBe("100%");
    expect(formatUiScalePercent(1.1)).toBe("110%");
    expect(formatUiScalePercent(1.25)).toBe("125%");
    expect(formatUiScalePercent(2)).toBe("200%");
  });

  it("light Auto while it is on and the manual size while it is off", () => {
    expect(uiScaleChipValue({ auto: true, manual: 1.2 })).toBe(UI_SCALE_AUTO_CHIP);
    expect(uiScaleChipValue({ auto: false, manual: 1.2 })).toBe(1.2);
    expect(uiScaleChipValue({ auto: false, manual: 1 })).toBe(1);
  });

  it("picking Auto turns Auto on and leaves the manual size alone", () => {
    expect(uiScaleChipPick(UI_SCALE_AUTO_CHIP)).toEqual({ auto: true });
  });

  it("picking a size turns Auto off and sets that size, clamped", () => {
    expect(uiScaleChipPick(0.8)).toEqual({ auto: false, manual: 0.8 });
    expect(uiScaleChipPick(1)).toEqual({ auto: false, manual: 1 });
    expect(uiScaleChipPick(1.4)).toEqual({ auto: false, manual: 1.4 });
    expect(uiScaleChipPick("1.2")).toEqual({ auto: false, manual: 1.2 });
    expect(uiScaleChipPick(9)).toEqual({ auto: false, manual: UI_SCALE_MAX });
    expect(uiScaleChipPick("nonsense")).toEqual({ auto: false, manual: 1 });
  });
});

describe("applyUiScaleToDocument", () => {
  function fakeRoot() {
    const log: string[] = [];
    return {
      log,
      root: {
        style: {
          setProperty: (name: string, value: string | null) => void log.push(`set ${name}=${value}`),
          removeProperty: (name: string) => {
            log.push(`remove ${name}`);
            return "";
          },
        },
        getBoundingClientRect: () => void log.push("layout"),
      },
    };
  }

  it("writes --uiscale", () => {
    const { root, log } = fakeRoot();
    applyUiScaleToDocument(root, 1.25, false);
    expect(log).toEqual(["set --uiscale=1.25"]);
  });

  it("writes the first size with the font-size transition off and computed before it comes back, so the first paint does not animate", () => {
    const { root, log } = fakeRoot();
    applyUiScaleToDocument(root, 1.5, true);
    expect(log).toEqual(["set transition=none", "set --uiscale=1.5", "layout", "remove transition"]);
  });
});

describe("windowGrowth: the window the interface needs", () => {
  it("is 800 x 600 interface units, the size tauri.conf.json and the Rust first-launch sizing start from", () => {
    expect(UI_WINDOW_MIN_SIZE).toEqual({ width: 800, height: 600 });
  });

  it("grows the saved default 800 x 600 window to 800 x 600 units when Auto raises the size to 1.5", () => {
    expect(windowGrowth(1.5, win(100, 100, 800, 600), MON_4K_100)).toEqual({ x: 100, y: 100, width: 1200, height: 900 });
  });

  it("counts in logical px: 1.2 on a 4K monitor at 125% is 1200 x 900 physical", () => {
    expect(windowGrowth(1.2, win(100, 100, 800, 600), MON_4K_125)).toEqual({ x: 100, y: 100, width: 1200, height: 900 });
    expect(windowGrowth(2, win(0, 0, 800, 600), { ...MON_4K_100, width: 5120, height: 2880, workArea: { x: 0, y: 0, width: 5120, height: 2832 } })).toEqual({
      x: 0,
      y: 0,
      width: 1600,
      height: 1200,
    });
    // A half step rounds to a whole pixel.
    expect(windowGrowth(1.05, win(0, 0, 800, 600), MON_4K_125)).toEqual({ x: 0, y: 0, width: 1050, height: 788 });
  });

  it("leaves a window alone that is big enough, or bigger than the screen", () => {
    expect(windowGrowth(1.5, win(100, 100, 1200, 900), MON_4K_100)).toBeNull();
    expect(windowGrowth(1.5, win(100, 100, 1440, 1080), MON_4K_100)).toBeNull();
    expect(windowGrowth(1.5, win(0, 0, 5000, 3000), MON_4K_100)).toBeNull();
  });

  it("grows only the side that is too short", () => {
    expect(windowGrowth(1.5, win(100, 100, 1600, 700), MON_4K_100)).toEqual({ x: 100, y: 100, width: 1600, height: 900 });
    expect(windowGrowth(1.5, win(100, 100, 900, 1000), MON_4K_100)).toEqual({ x: 100, y: 100, width: 1200, height: 1000 });
  });

  it("does nothing at a size of 1 or less, so nothing changes for those users", () => {
    for (const scale of [1, 0.8, 0, -2, NaN, Infinity]) expect(windowGrowth(scale, win(0, 0, 400, 300), MON_4K_100)).toBeNull();
  });

  it("moves the window back inside the work area when it grows over an edge", () => {
    expect(windowGrowth(1.5, win(3000, 1500, 800, 600), MON_4K_100)).toEqual({ x: 2640, y: 1212, width: 1200, height: 900 });
  });

  it("keeps the frame: the outer size grows by what the inner size grows by", () => {
    const framed: WindowMetrics = { innerWidth: 800, innerHeight: 600, outerX: 3000, outerY: 1500, outerWidth: 816, outerHeight: 639 };
    expect(windowGrowth(1.5, framed, MON_4K_100)).toEqual({ x: 3840 - 1216, y: 2112 - 939, width: 1200, height: 900 });
  });

  it("works on a monitor whose work area starts left of or above the origin", () => {
    const left: MonitorMetrics = { ...MON_4K_100, workArea: { x: -3840, y: 0, width: 3840, height: 2112 } };
    expect(windowGrowth(1.5, win(-1000, 100, 800, 600), left)).toEqual({ x: -1200, y: 100, width: 1200, height: 900 });
    expect(windowGrowth(1.5, win(-4000, 100, 800, 600), left)).toEqual({ x: -3840, y: 100, width: 1200, height: 900 });
  });

  it("never asks for more than the work area", () => {
    const small: MonitorMetrics = { ...MON_4K_100, workArea: { x: 0, y: 0, width: 1000, height: 700 } };
    expect(windowGrowth(1.5, win(0, 0, 800, 600), small)).toEqual({ x: 0, y: 0, width: 1000, height: 700 });
  });

  it("does nothing without a work area, a monitor, or numbers it can use", () => {
    const { workArea: _unused, ...noWork } = MON_4K_100;
    expect(windowGrowth(1.5, win(0, 0, 800, 600), noWork)).toBeNull();
    expect(windowGrowth(1.5, win(0, 0, 800, 600), null)).toBeNull();
    expect(windowGrowth(1.5, win(0, 0, 800, 600), undefined)).toBeNull();
    expect(windowGrowth(1.5, win(0, 0, 800, 600), { ...MON_4K_100, scaleFactor: 0 })).toBeNull();
    expect(windowGrowth(1.5, win(0, 0, 800, 600), { ...MON_4K_100, workArea: { x: 0, y: 0, width: 0, height: 2000 } })).toBeNull();
    expect(windowGrowth(1.5, win(0, 0, 0, 0), MON_4K_100)).toBeNull();
    expect(windowGrowth(1.5, { ...win(0, 0, 800, 600), outerX: NaN }, MON_4K_100)).toBeNull();
  });
});

describe("UiScaleController", () => {
  type Harness = ReturnType<typeof harness>;
  /**
   * `window` gives the controller a window to look after: its metrics (`null` is a window that must not be touched:
   * maximized, minimized, fullscreen). Without it the controller has no window dependencies, as outside Tauri.
   */
  function harness(init: { monitor?: MonitorMetrics | null; auto?: boolean; manual?: number; window?: WindowMetrics | null; deps?: Partial<UiScaleDeps> } = {}) {
    const settings = { auto: init.auto ?? true, manual: init.manual ?? 1 };
    const h = {
      settings,
      monitor: init.monitor === undefined ? MON_4K_100 : init.monitor,
      window: init.window === undefined ? null : init.window,
      /** Whether the system gives the window the size it is asked for. */
      windowObeys: true,
      readError: null as Error | null,
      applied: [] as [number, boolean][],
      placed: [] as { x: number; y: number; width: number; height: number }[],
      onChange: () => { },
      unlisten: vi.fn(),
      readMonitor: vi.fn(),
      listen: vi.fn(),
      readWindow: vi.fn(),
      placeWindow: vi.fn(),
      controller: undefined as unknown as UiScaleController,
    };
    h.readMonitor.mockImplementation(async () => {
      if (h.readError) throw h.readError;
      return h.monitor;
    });
    h.listen.mockImplementation(async (cb: () => void) => {
      h.onChange = cb;
      return h.unlisten;
    });
    h.readWindow.mockImplementation(async () => h.window);
    h.placeWindow.mockImplementation(async (g: { x: number; y: number; width: number; height: number }) => {
      h.placed.push(g);
      if (h.windowObeys) h.window = win(g.x, g.y, g.width, g.height);
    });
    h.controller = new UiScaleController({
      getSettings: () => settings,
      readMonitor: h.readMonitor,
      listen: h.listen,
      apply: (value, first) => void h.applied.push([value, first]),
      ...(init.window !== undefined ? { readWindow: h.readWindow, placeWindow: h.placeWindow } : {}),
      ...init.deps,
    });
    return h;
  }
  /** Lets `void this.#subscribe()` and the promise chains behind it finish. */
  const flush = async () => {
    for (let i = 0; i < 5; i++) await Promise.resolve();
  };
  const moveTo = async (h: Harness, monitor: MonitorMetrics | null) => {
    h.monitor = monitor;
    h.onChange();
    await vi.advanceTimersByTimeAsync(UI_SCALE_DEBOUNCE_MS);
  };

  beforeEach(() => {
    vi.useFakeTimers();
  });
  afterEach(() => {
    vi.useRealTimers();
  });

  it("has applied the monitor's Auto size when start resolves (before anything is rendered)", async () => {
    const h = harness({ monitor: MON_4K_100 });
    await h.controller.start();
    expect(h.applied).toEqual([[1.5, true]]);
    expect(h.controller.effective).toBe(1.5);
  });

  it("applies 1 on a monitor that needs no larger interface (and still writes it once, as the first size)", async () => {
    const h = harness({ monitor: MON_1080P });
    await h.controller.start();
    expect(h.applied).toEqual([[1, true]]);
  });

  it("uses 1 in Auto when the monitor cannot be read (null, or an error)", async () => {
    const empty = harness({ monitor: null });
    await empty.controller.start();
    expect(empty.applied).toEqual([[1, true]]);

    const broken = harness();
    broken.readError = new Error("no window");
    const warn = vi.spyOn(console, "warn").mockImplementation(() => { });
    await broken.controller.start();
    expect(broken.applied).toEqual([[1, true]]);
    expect(warn).toHaveBeenCalled();
  });

  it("applies the clamped manual size while Auto is off, whatever the monitor is", async () => {
    const h = harness({ monitor: MON_4K_100, auto: false, manual: 1.2 });
    await h.controller.start();
    expect(h.applied).toEqual([[1.2, true]]);

    const wild = harness({ monitor: MON_4K_100, auto: false, manual: 7 });
    await wild.controller.start();
    expect(wild.applied).toEqual([[UI_SCALE_MAX, true]]);
  });

  it("does not wait for a monitor that never answers: after the start timeout it applies 1, and the answer is applied when it comes", async () => {
    let answer: (m: MonitorMetrics) => void = () => { };
    const h = harness({ deps: { readMonitor: () => new Promise<MonitorMetrics>(resolve => { answer = resolve; }) } });
    let started = false;
    void h.controller.start().then(() => { started = true; });
    await vi.advanceTimersByTimeAsync(UI_SCALE_START_TIMEOUT_MS - 1);
    expect(started).toBe(false);
    await vi.advanceTimersByTimeAsync(1);
    expect(started).toBe(true);
    expect(h.applied).toEqual([[1, true]]);

    answer(MON_4K_100);
    await flush();
    expect(h.applied).toEqual([[1, true], [1.5, false]]);
  });

  it("writes only when the size in effect changes", async () => {
    const h = harness({ monitor: MON_4K_100 });
    await h.controller.start();
    h.controller.refresh();
    h.controller.refresh();
    expect(h.applied).toEqual([[1.5, true]]);

    h.settings.manual = 1.2; // a manual size is not in effect while Auto is on
    h.controller.refresh();
    expect(h.applied).toEqual([[1.5, true]]);

    h.settings.auto = false;
    h.controller.refresh();
    expect(h.applied).toEqual([[1.5, true], [1.2, false]]);
    h.settings.auto = true;
    h.controller.refresh();
    expect(h.applied).toEqual([[1.5, true], [1.2, false], [1.5, false]]);
  });

  it("follows the window to another monitor, after the move has been quiet for the debounce time", async () => {
    const h = harness({ monitor: MON_4K_100 });
    await h.controller.start();
    await flush();
    expect(h.listen).toHaveBeenCalledTimes(1);

    h.monitor = MON_1080P;
    h.onChange();
    await vi.advanceTimersByTimeAsync(UI_SCALE_DEBOUNCE_MS - 1);
    expect(h.applied).toEqual([[1.5, true]]); // not yet
    await vi.advanceTimersByTimeAsync(1);
    expect(h.applied).toEqual([[1.5, true], [1, false]]);
  });

  it("follows a change of the Windows scale on the same monitor", async () => {
    const h = harness({ monitor: MON_4K_100 });
    await h.controller.start();
    await flush();
    await moveTo(h, MON_4K_150);
    expect(h.applied).toEqual([[1.5, true], [1, false]]);
    await moveTo(h, MON_4K_125);
    expect(h.applied).toEqual([[1.5, true], [1, false], [1.2, false]]);
  });

  it("reads the monitor once for a burst of move events (dragging the window)", async () => {
    const h = harness({ monitor: MON_4K_100 });
    await h.controller.start();
    await flush();
    expect(h.readMonitor).toHaveBeenCalledTimes(1);
    h.monitor = MON_1080P;
    for (let i = 0; i < 20; i++) {
      h.onChange();
      await vi.advanceTimersByTimeAsync(40); // each event comes inside the quiet time of the one before
    }
    expect(h.readMonitor).toHaveBeenCalledTimes(1);
    await vi.advanceTimersByTimeAsync(UI_SCALE_DEBOUNCE_MS);
    expect(h.readMonitor).toHaveBeenCalledTimes(2);
    expect(h.applied).toEqual([[1.5, true], [1, false]]);
  });

  it("does not write when the window moves within the same monitor", async () => {
    const h = harness({ monitor: MON_4K_100 });
    await h.controller.start();
    await flush();
    await moveTo(h, MON_4K_100);
    await moveTo(h, MON_4K_100);
    expect(h.applied).toEqual([[1.5, true]]);
  });

  it("keeps the last monitor when a later read fails or finds none (a hiccup never flips the interface size)", async () => {
    const h = harness({ monitor: MON_4K_100 });
    await h.controller.start();
    await flush();
    await moveTo(h, null);
    expect(h.applied).toEqual([[1.5, true]]);
    h.readError = new Error("window is gone");
    const warn = vi.spyOn(console, "warn").mockImplementation(() => { });
    await moveTo(h, MON_1080P);
    expect(h.applied).toEqual([[1.5, true]]);
    expect(warn).toHaveBeenCalled();
    h.readError = null;
    await moveTo(h, MON_1080P);
    expect(h.applied).toEqual([[1.5, true], [1, false]]);
  });

  it("only uses the newest read when two overlap", async () => {
    const answers: ((m: MonitorMetrics) => void)[] = [];
    const h = harness({ monitor: MON_4K_100 });
    await h.controller.start();
    await flush();
    h.readMonitor.mockImplementation(() => new Promise<MonitorMetrics>(resolve => answers.push(resolve)));
    h.onChange();
    await vi.advanceTimersByTimeAsync(UI_SCALE_DEBOUNCE_MS); // read 1 in flight
    h.onChange();
    await vi.advanceTimersByTimeAsync(UI_SCALE_DEBOUNCE_MS); // read 2 in flight
    expect(answers).toHaveLength(2);
    answers[1](MON_4K_125);
    await flush();
    answers[0](MON_1080P); // the older answer arrives last
    await flush();
    expect(h.applied).toEqual([[1.5, true], [1.2, false]]);
  });

  it("tracks the Auto size of the monitor for the Settings chip even while Auto is off, without applying it", async () => {
    const h = harness({ monitor: MON_4K_100, auto: false, manual: 1.2 });
    expect(h.controller.detected.auto).toBe(1);
    await h.controller.start();
    expect(h.controller.detected.auto).toBe(1.5);
    await flush();
    await moveTo(h, MON_4K_125);
    expect(h.controller.detected.auto).toBe(1.2);
    expect(h.applied).toEqual([[1.2, true]]);
  });

  it("stop removes the window listeners and drops a pending read", async () => {
    const h = harness({ monitor: MON_4K_100 });
    await h.controller.start();
    await flush();
    h.monitor = MON_1080P;
    h.onChange(); // a read is pending
    h.controller.stop();
    expect(h.unlisten).toHaveBeenCalledTimes(1);
    await vi.advanceTimersByTimeAsync(UI_SCALE_DEBOUNCE_MS * 4);
    expect(h.readMonitor).toHaveBeenCalledTimes(1); // only the one at start
    expect(h.applied).toEqual([[1.5, true]]);
    h.onChange(); // a late event after stop does nothing
    await vi.advanceTimersByTimeAsync(UI_SCALE_DEBOUNCE_MS * 4);
    expect(h.readMonitor).toHaveBeenCalledTimes(1);
  });

  it("stop before the listeners are attached still removes them once they are", async () => {
    let attach: (unlisten: () => void) => void = () => { };
    const unlisten = vi.fn();
    const h = harness({ deps: { listen: () => new Promise<() => void>(resolve => { attach = resolve; }) } });
    await h.controller.start();
    h.controller.stop();
    attach(unlisten);
    await flush();
    expect(unlisten).toHaveBeenCalledTimes(1);
  });

  it("keeps working when the window events cannot be listened to", async () => {
    const warn = vi.spyOn(console, "warn").mockImplementation(() => { });
    const h = harness({ monitor: MON_4K_100, deps: { listen: async () => { throw new Error("not allowed"); } } });
    await h.controller.start();
    await flush();
    expect(h.applied).toEqual([[1.5, true]]);
    expect(warn).toHaveBeenCalled();
    h.controller.stop(); // nothing to remove, and no throw
  });

  it("keeps the newest answer that came back when a read started after it fails or finds nothing", async () => {
    const answers: { resolve: (m: MonitorMetrics | null) => void; reject: (e: Error) => void }[] = [];
    const h = harness({ monitor: MON_4K_100 });
    await h.controller.start();
    await flush();
    h.readMonitor.mockImplementation(() => new Promise<MonitorMetrics | null>((resolve, reject) => answers.push({ resolve, reject })));
    const warn = vi.spyOn(console, "warn").mockImplementation(() => { });

    h.onChange();
    await vi.advanceTimersByTimeAsync(UI_SCALE_DEBOUNCE_MS); // read 1 in flight
    h.onChange();
    await vi.advanceTimersByTimeAsync(UI_SCALE_DEBOUNCE_MS); // read 2 in flight
    answers[0].resolve(MON_1080P);
    await flush();
    answers[1].resolve(null); // the newest read finds no monitor
    await flush();
    expect(h.applied).toEqual([[1.5, true], [1, false]]);

    h.onChange();
    await vi.advanceTimersByTimeAsync(UI_SCALE_DEBOUNCE_MS); // read 3
    h.onChange();
    await vi.advanceTimersByTimeAsync(UI_SCALE_DEBOUNCE_MS); // read 4
    answers[2].resolve(MON_4K_125);
    await flush();
    answers[3].reject(new Error("window is gone")); // the newest read fails
    await flush();
    expect(h.applied).toEqual([[1.5, true], [1, false], [1.2, false]]);
    expect(warn).toHaveBeenCalled();
  });

  it("says whether a monitor was read (Settings says so when it could not be), and keeps saying so after a failed read", async () => {
    const unread = harness({ monitor: null });
    expect(unread.controller.detected.known).toBe(false);
    await unread.controller.start();
    expect(unread.controller.detected).toEqual({ auto: 1, known: false });

    const h = harness({ monitor: MON_1080P });
    await h.controller.start();
    expect(h.controller.detected).toEqual({ auto: 1, known: true }); // a real 1080p answer
    await flush();
    await moveTo(h, null);
    expect(h.controller.detected.known).toBe(true);
    await moveTo(h, MON_4K_100);
    expect(h.controller.detected).toEqual({ auto: 1.5, known: true });
  });

  describe("the window", () => {
    it("grows a saved default window when Auto applies a larger size at start (a profile that moves from M to Auto)", async () => {
      const h = harness({ monitor: MON_4K_100, window: win(100, 100, 800, 600) });
      await h.controller.start();
      await flush();
      expect(h.applied).toEqual([[1.5, true]]);
      expect(h.placed).toEqual([{ x: 100, y: 100, width: 1200, height: 900 }]);
    });

    it("leaves a window that already holds the interface alone", async () => {
      const h = harness({ monitor: MON_4K_100, window: win(100, 100, 1440, 1080) });
      await h.controller.start();
      await flush();
      expect(h.readWindow).toHaveBeenCalledTimes(1); // once for the start, not once per refresh of it
      expect(h.placed).toEqual([]);
    });

    it("never touches the window at size 1 or when a manual size is picked, and does not even read it", async () => {
      const small = harness({ monitor: MON_1080P, window: win(0, 0, 400, 300) });
      await small.controller.start();
      await flush();
      const manual = harness({ monitor: MON_4K_100, auto: false, manual: 1.4, window: win(0, 0, 400, 300) });
      await manual.controller.start();
      await flush();
      const noWorkArea = harness({ monitor: { width: 3840, height: 2160, scaleFactor: 1 }, window: win(0, 0, 400, 300) });
      await noWorkArea.controller.start();
      await flush();
      for (const h of [small, manual, noWorkArea]) {
        expect(h.readWindow).not.toHaveBeenCalled();
        expect(h.placed).toEqual([]);
      }
    });

    it("leaves a window alone that must not be touched (maximized, minimized or fullscreen)", async () => {
      const h = harness({ monitor: MON_4K_100, window: null });
      await h.controller.start();
      await flush();
      await moveTo(h, MON_4K_100);
      expect(h.readWindow).toHaveBeenCalled();
      expect(h.placed).toEqual([]);
    });

    it("grows the window when it is moved to a bigger monitor (Auto goes from 1 to 1.5)", async () => {
      const h = harness({ monitor: MON_1080P, window: win(100, 100, 800, 600) });
      await h.controller.start();
      await flush();
      expect(h.placed).toEqual([]);
      await moveTo(h, MON_4K_100);
      expect(h.applied).toEqual([[1, true], [1.5, false]]);
      expect(h.placed).toEqual([{ x: 100, y: 100, width: 1200, height: 900 }]);
    });

    it("grows the window when the user turns Auto on", async () => {
      const h = harness({ monitor: MON_4K_100, auto: false, manual: 0.8, window: win(100, 100, 800, 600) });
      await h.controller.start();
      await flush();
      expect(h.placed).toEqual([]);
      h.settings.auto = true;
      h.controller.refresh();
      await flush();
      expect(h.placed).toEqual([{ x: 100, y: 100, width: 1200, height: 900 }]);
    });

    it("checks again when the window is resized, so a window restored from maximized or dragged smaller is grown", async () => {
      const h = harness({ monitor: MON_4K_100, window: win(100, 100, 1400, 1000) });
      await h.controller.start();
      await flush();
      expect(h.placed).toEqual([]);
      h.window = win(100, 100, 700, 500);
      await moveTo(h, MON_4K_100);
      expect(h.placed).toEqual([{ x: 100, y: 100, width: 1200, height: 900 }]);
    });

    it("asks only once for a size the system does not give, and again when the window changes size", async () => {
      const h = harness({ monitor: MON_4K_100, window: win(100, 100, 800, 600) });
      h.windowObeys = false;
      await h.controller.start();
      await flush();
      await moveTo(h, MON_4K_100);
      await moveTo(h, MON_4K_100);
      expect(h.placed).toHaveLength(1);
      h.window = win(100, 100, 900, 600);
      await moveTo(h, MON_4K_100);
      expect(h.placed).toHaveLength(2);
    });

    it("asks again when a window that was grown is shrunk back to the size it was", async () => {
      const h = harness({ monitor: MON_4K_100, window: win(100, 100, 800, 600) });
      await h.controller.start();
      await flush();
      expect(h.placed).toHaveLength(1);
      await moveTo(h, MON_4K_100); // the window fits now
      h.window = win(100, 100, 800, 600);
      await moveTo(h, MON_4K_100);
      expect(h.placed).toHaveLength(2);
    });

    it("keeps working when the window cannot be read or resized", async () => {
      const warn = vi.spyOn(console, "warn").mockImplementation(() => { });
      const h = harness({ monitor: MON_4K_100, window: win(100, 100, 800, 600) });
      h.readWindow.mockRejectedValue(new Error("no permission"));
      await h.controller.start();
      await flush();
      expect(h.applied).toEqual([[1.5, true]]);
      expect(warn).toHaveBeenCalledTimes(1);

      h.readWindow.mockResolvedValue(win(100, 100, 800, 600));
      h.placeWindow.mockRejectedValue(new Error("window is gone"));
      await moveTo(h, MON_4K_100);
      expect(h.applied).toEqual([[1.5, true]]);
      expect(warn).toHaveBeenCalledTimes(2);
    });

    it("does not resize after stop", async () => {
      const h = harness({ monitor: MON_4K_100, window: win(100, 100, 1400, 1000) });
      await h.controller.start();
      await flush();
      h.window = win(100, 100, 700, 500);
      h.controller.stop();
      h.controller.refresh();
      await flush();
      expect(h.placed).toEqual([]);
    });
  });

  it("start is idempotent", async () => {
    const h = harness({ monitor: MON_4K_100 });
    await h.controller.start();
    await h.controller.start();
    await flush();
    expect(h.readMonitor).toHaveBeenCalledTimes(1);
    expect(h.listen).toHaveBeenCalledTimes(1);
  });
});

describe("tauriWindowDeps (the Tauri window)", () => {
  beforeEach(() => {
    tauri.isTauri.mockReturnValue(true);
    for (const fn of [tauri.currentMonitor, tauri.onMoved, tauri.onScaleChanged, tauri.onResized, tauri.setPosition, tauri.setSize]) fn.mockReset();
    tauri.isMaximized.mockReset().mockResolvedValue(false);
    tauri.isMinimized.mockReset().mockResolvedValue(false);
    tauri.isFullscreen.mockReset().mockResolvedValue(false);
    tauri.innerSize.mockReset().mockResolvedValue(new PhysicalSize(800, 600));
    tauri.outerPosition.mockReset().mockResolvedValue(new PhysicalPosition(120, 80));
    tauri.outerSize.mockReset().mockResolvedValue(new PhysicalSize(816, 639));
  });

  it("reads the monitor's physical size, Windows scale and work area", async () => {
    tauri.currentMonitor.mockResolvedValue({
      name: "DELL",
      size: { width: 3840, height: 2160 },
      scaleFactor: 1.25,
      position: { x: 0, y: 0 },
      workArea: { position: { x: 0, y: 0 }, size: { width: 3840, height: 2100 } },
    });
    expect(await tauriWindowDeps().readMonitor()).toEqual({
      width: 3840,
      height: 2160,
      scaleFactor: 1.25,
      workArea: { x: 0, y: 0, width: 3840, height: 2100 },
    });
  });

  it("reads null when the window has no monitor", async () => {
    tauri.currentMonitor.mockResolvedValue(null);
    expect(await tauriWindowDeps().readMonitor()).toBeNull();
  });

  it("has no monitor or window outside Tauri (the browser build): it neither asks, listens nor resizes", async () => {
    tauri.isTauri.mockReturnValue(false);
    const deps = tauriWindowDeps();
    expect(await deps.readMonitor()).toBeNull();
    expect(await deps.readWindow?.()).toBeNull();
    await deps.placeWindow?.({ x: 0, y: 0, width: 1200, height: 900 });
    const unlisten = await deps.listen(() => { });
    expect(tauri.currentMonitor).not.toHaveBeenCalled();
    expect(tauri.onMoved).not.toHaveBeenCalled();
    expect(tauri.onResized).not.toHaveBeenCalled();
    expect(tauri.isMaximized).not.toHaveBeenCalled();
    expect(tauri.setSize).not.toHaveBeenCalled();
    expect(() => unlisten()).not.toThrow();
  });

  it("listens to moves, resizes and scale changes, and stops all three", async () => {
    const stopMoved = vi.fn();
    const stopScale = vi.fn();
    const stopResized = vi.fn();
    tauri.onMoved.mockResolvedValue(stopMoved);
    tauri.onScaleChanged.mockResolvedValue(stopScale);
    tauri.onResized.mockResolvedValue(stopResized);
    const onChange = vi.fn();
    const unlisten = await tauriWindowDeps().listen(onChange);
    expect(tauri.onMoved).toHaveBeenCalledWith(onChange);
    expect(tauri.onScaleChanged).toHaveBeenCalledWith(onChange);
    // A maximized window changes size when the desktop resolution changes, without being moved.
    expect(tauri.onResized).toHaveBeenCalledWith(onChange);
    expect(stopMoved).not.toHaveBeenCalled();
    unlisten();
    expect(stopMoved).toHaveBeenCalledTimes(1);
    expect(stopScale).toHaveBeenCalledTimes(1);
    expect(stopResized).toHaveBeenCalledTimes(1);
  });

  it("removes the listeners that did attach when another one fails", async () => {
    const stopMoved = vi.fn();
    const stopResized = vi.fn();
    tauri.onMoved.mockResolvedValue(stopMoved);
    tauri.onScaleChanged.mockRejectedValue(new Error("not allowed"));
    tauri.onResized.mockResolvedValue(stopResized);
    await expect(tauriWindowDeps().listen(() => { })).rejects.toThrow("not allowed");
    expect(stopMoved).toHaveBeenCalledTimes(1);
    expect(stopResized).toHaveBeenCalledTimes(1);

    tauri.onScaleChanged.mockResolvedValue(vi.fn());
    tauri.onResized.mockImplementation(() => { throw new Error("no such command"); });
    await expect(tauriWindowDeps().listen(() => { })).rejects.toThrow("no such command");
    expect(stopMoved).toHaveBeenCalledTimes(2);
  });

  it("reads the size and position of a normal window", async () => {
    expect(await tauriWindowDeps().readWindow?.()).toEqual({
      innerWidth: 800,
      innerHeight: 600,
      outerX: 120,
      outerY: 80,
      outerWidth: 816,
      outerHeight: 639,
    });
  });

  it("reads null for a window that is maximized, minimized or fullscreen, because resizing one would un-maximize or restore it", async () => {
    for (const state of [tauri.isMaximized, tauri.isMinimized, tauri.isFullscreen]) {
      state.mockResolvedValue(true);
      expect(await tauriWindowDeps().readWindow?.()).toBeNull();
      state.mockResolvedValue(false);
    }
  });

  it("moves the window and then sizes it, in physical px", async () => {
    await tauriWindowDeps().placeWindow?.({ x: 2640, y: 1212, width: 1200, height: 900 });
    expect(tauri.setPosition).toHaveBeenCalledWith(new PhysicalPosition(2640, 1212));
    expect(tauri.setSize).toHaveBeenCalledWith(new PhysicalSize(1200, 900));
    expect(tauri.setPosition.mock.invocationCallOrder[0]).toBeLessThan(tauri.setSize.mock.invocationCallOrder[0]);
  });
});
