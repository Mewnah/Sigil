/**
 * Interface size: the size every menu is drawn at on the host app (`html { font-size: calc(16px * var(--uiscale)) }` in
 * style.css, so every rem-based size follows).
 *
 * The choice is Auto, S, M, L or XL. S, M, L and XL are fixed multipliers (`uiScale`, 0.8 to 1.5 once clamped). Auto
 * (`uiScaleAuto`, the default) works the size out from the monitor the Sigil window is on, so a 4K monitor that Windows
 * runs at 100% or 125% (where the logical desktop is still 3072 x 1728 or more and every menu would look tiny) gets a
 * larger interface, while a monitor that is 1440p-sized or smaller in Windows' logical pixels is left at 1.
 *
 * - `autoUiScale` is the Auto rule. It is shared with the Rust side through `uiScale.fixture.json`: both languages read
 *   the same table of monitors, so a change to the rule is a change to that file and both tests.
 * - `effectiveUiScale` is the size in effect: Auto from the monitor, or the clamped manual size. `uiScale` in the
 *   settings is only ever the MANUAL size (what S/M/L/XL set, kept while Auto is on so that turning Auto off returns
 *   to it); nothing may read it as the size in effect.
 * - `UiScaleController` applies it on the host: once right after the settings load (before the first-run wizard or
 *   the editor paints, so a 4K-at-100% new install never shows a frame at 1), and again when the window moves to
 *   another monitor, is resized, or the Windows scale changes. While Auto is above 1 it also keeps the window from being
 *   smaller than the interface needs (`windowGrowth`). The /client pages, the caption window and every other secondary
 *   page never create one.
 */

import { isTauri } from "@tauri-apps/api/core";
import { PhysicalPosition, PhysicalSize } from "@tauri-apps/api/dpi";
import { currentMonitor, getCurrentWindow } from "@tauri-apps/api/window";
import { proxy } from "valtio";

/** Manual sizes are kept inside this range (the S and XL chips are 0.8 and 1.4). */
export const UI_SCALE_MIN = 0.8;
export const UI_SCALE_MAX = 1.5;

/** Auto never shrinks the interface below M and never goes past twice its size. */
export const AUTO_UI_SCALE_MIN = 1;
export const AUTO_UI_SCALE_MAX = 2;
/** The logical desktop size that is size 1: 2560 x 1440 (a 1440p monitor at 100%, a 4K one at 150%). */
const REFERENCE_SHORT_SIDE = 1440;
const REFERENCE_LONG_SIDE = 2560;
/** Auto is rounded to the nearest 5% so that two nearly equal monitors do not give two different sizes. */
const AUTO_STEPS_PER_UNIT = 20;

/** The window rectangle of a monitor without the taskbar, in physical pixels (virtual-screen coordinates). */
export interface WorkArea {
  x: number;
  y: number;
  width: number;
  height: number;
}

/**
 * A monitor as the window API reports it: `width` and `height` in physical pixels, `scaleFactor` is Windows' scale
 * (1.5 = 150%). The work area is only needed to keep the window on the screen when it grows (`windowGrowth`).
 */
export interface MonitorMetrics {
  width: number;
  height: number;
  scaleFactor: number;
  workArea?: WorkArea;
}

const isPositiveFinite = (n: unknown): n is number => typeof n === "number" && Number.isFinite(n) && n > 0;

/**
 * The Auto size for a monitor: `s = min(shortSide / 1440, longSide / 2560)` on the logical sides
 * (`physical / scaleFactor`), clamped to [1, 2] and rounded to the nearest 0.05. A monitor that is unknown, or has a
 * zero or invalid size or scale, gives 1. Matches the Rust side; `uiScale.fixture.json` is the shared table.
 */
export function autoUiScale(monitor: MonitorMetrics | null | undefined): number {
  if (!monitor || !isPositiveFinite(monitor.width) || !isPositiveFinite(monitor.height) || !isPositiveFinite(monitor.scaleFactor)) return 1;
  const shortSide = Math.min(monitor.width, monitor.height) / monitor.scaleFactor;
  const longSide = Math.max(monitor.width, monitor.height) / monitor.scaleFactor;
  const raw = Math.min(shortSide / REFERENCE_SHORT_SIDE, longSide / REFERENCE_LONG_SIDE);
  const clamped = Math.min(AUTO_UI_SCALE_MAX, Math.max(AUTO_UI_SCALE_MIN, raw));
  return Math.round(clamped * AUTO_STEPS_PER_UNIT) / AUTO_STEPS_PER_UNIT;
}

/** A manual size inside the allowed range; anything that is not a number is M. */
export function clampManualUiScale(value: number): number {
  if (typeof value !== "number" || Number.isNaN(value)) return 1;
  return Math.max(UI_SCALE_MIN, Math.min(UI_SCALE_MAX, value));
}

/** The size in effect: Auto from the monitor (1 when it is unknown), or the clamped manual size. */
export function effectiveUiScale(input: { auto: boolean; manual: number; monitor: MonitorMetrics | null | undefined }): number {
  return input.auto ? autoUiScale(input.monitor) : clampManualUiScale(input.manual);
}

// region ---SETTINGS CHIPS---

/** The value of the Auto chip; the other chips carry their manual size. */
export const UI_SCALE_AUTO_CHIP = "auto";
export type UiScaleChipValue = typeof UI_SCALE_AUTO_CHIP | number;
/** The manual chips, S, M, L and XL, in order. */
export const UI_SCALE_MANUAL_CHIPS = [
  { key: "s", value: 0.8 },
  { key: "m", value: 1 },
  { key: "l", value: 1.2 },
  { key: "x", value: 1.4 },
] as const;

/** "150%" for a size of 1.5. */
export function formatUiScalePercent(scale: number): string {
  return `${Math.round(scale * 100)}%`;
}

/** The chip that is lit: Auto while it is on, otherwise the manual size (which may be none of the chips). */
export function uiScaleChipValue(settings: { auto: boolean; manual: number }): UiScaleChipValue {
  return settings.auto ? UI_SCALE_AUTO_CHIP : settings.manual;
}

/** What picking a chip does to the settings: Auto turns Auto on, a manual size turns it off and sets the (clamped) size. */
export type UiScaleChipPick = { auto: true } | { auto: false; manual: number };
export function uiScaleChipPick(value: string | number): UiScaleChipPick {
  if (value === UI_SCALE_AUTO_CHIP) return { auto: true };
  return { auto: false, manual: clampManualUiScale(typeof value === "string" ? parseFloat(value) : value) };
}

type Translate = (key: string, options?: Record<string, string>) => string;

/**
 * The chips in order: Auto, S, M, L, XL. `t` is i18next's `t`; the S/M/L/X labels are the existing `settings.ui_scale_*`
 * keys. The Auto chip says only "Auto": the size it picked goes on the line below the row (`uiScaleAutoHint`), because
 * a chip with the size in it is too wide for the row.
 */
export function uiScaleChipOptions(t: Translate): { label: string; value: UiScaleChipValue }[] {
  return [
    { label: t("settings.ui_scale_auto"), value: UI_SCALE_AUTO_CHIP },
    ...UI_SCALE_MANUAL_CHIPS.map(({ key, value }) => ({ label: t(`settings.ui_scale_${key}`), value })),
  ];
}

/** What Auto picked ("Auto picks 150% for the monitor Sigil is on."), or that the monitor could not be read. */
export function uiScaleAutoHint(detected: { auto: number; known: boolean }, t: Translate): string {
  return detected.known
    ? t("settings.ui_scale_auto_hint", { size: formatUiScalePercent(detected.auto) })
    : t("settings.ui_scale_auto_unknown");
}

// endregion

// region ---APPLYING---

/** The part of `document.documentElement` that `applyUiScaleToDocument` touches. */
export interface UiScaleRoot {
  style: Pick<CSSStyleDeclaration, "setProperty" | "removeProperty">;
  getBoundingClientRect(): unknown;
}

/**
 * Writes `--uiscale`. The root font-size has a short transition (the Settings chips animate), which the FIRST write
 * must not play: it would show the wizard at 1 growing to its size over 100 ms. So the first write turns the
 * transition off, forces the style to be computed at the new size, and puts the transition back.
 */
export function applyUiScaleToDocument(root: UiScaleRoot, value: number, first: boolean): void {
  if (first) root.style.setProperty("transition", "none");
  root.style.setProperty("--uiscale", value.toString());
  if (first) {
    root.getBoundingClientRect();
    root.style.removeProperty("transition");
  }
}

// endregion

// region ---THE WINDOW---

/**
 * The window the interface is laid out for, in interface units (logical px divided by the interface size): the 800 x 600 of
 * `tauri.conf.json`, the same floor the Rust first-launch sizing uses (`window_size.rs` `MIN_SIZE`). A window that was
 * saved at 800 x 600 holds 533 x 400 units once Auto raises the size to 1.5, and the canvas column disappears.
 */
export const UI_WINDOW_MIN_SIZE = { width: 800, height: 600 } as const;

/** The window as the window API reports it, in physical pixels. */
export interface WindowMetrics {
  innerWidth: number;
  innerHeight: number;
  outerX: number;
  outerY: number;
  outerWidth: number;
  outerHeight: number;
}

/** Where the window goes: its new outer position and its new inner (client) size, in physical pixels. */
export interface WindowGrowth {
  x: number;
  y: number;
  width: number;
  height: number;
}

/**
 * What it takes for a window to hold the interface at `scale`: at least 800 x 600 units, so `800 x scale` by
 * `600 x scale` logical px, never more than the monitor's work area. It only ever GROWS the window (a window already
 * big enough is left alone, as is one that is too big for the screen), and moves it only as far as needed to keep it
 * inside the work area. `null` when nothing has to change, for a scale of 1 or less (nothing changes for those users),
 * and when the monitor has no work area or any number is not usable.
 */
export function windowGrowth(scale: number, win: WindowMetrics, monitor: MonitorMetrics | null | undefined): WindowGrowth | null {
  const work = monitor?.workArea;
  if (!monitor || !work || !isPositiveFinite(scale) || scale <= 1 || !isPositiveFinite(monitor.scaleFactor) || !isPositiveFinite(work.width) || !isPositiveFinite(work.height)) return null;
  const numbers = [win.innerWidth, win.innerHeight, win.outerX, win.outerY, win.outerWidth, win.outerHeight, work.x, work.y];
  if (!numbers.every(Number.isFinite) || win.innerWidth <= 0 || win.innerHeight <= 0) return null;

  const wantWidth = Math.min(Math.round(UI_WINDOW_MIN_SIZE.width * scale * monitor.scaleFactor), work.width);
  const wantHeight = Math.min(Math.round(UI_WINDOW_MIN_SIZE.height * scale * monitor.scaleFactor), work.height);
  const width = Math.max(win.innerWidth, wantWidth);
  const height = Math.max(win.innerHeight, wantHeight);
  if (width === win.innerWidth && height === win.innerHeight) return null;

  // The frame (title bar, borders, shadow) stays what it is: the outer size grows by what the inner size grows by.
  const outerWidth = win.outerWidth + (width - win.innerWidth);
  const outerHeight = win.outerHeight + (height - win.innerHeight);
  const inside = (position: number, size: number, start: number, extent: number) => Math.max(start, Math.min(position, start + extent - size));
  return {
    x: inside(win.outerX, outerWidth, work.x, work.width),
    y: inside(win.outerY, outerHeight, work.y, work.height),
    width,
    height,
  };
}

// endregion

// region ---FOLLOWING THE MONITOR---

/** What the controller needs from the outside (the Tauri window, the settings, the document). */
export interface UiScaleDeps {
  /** The Interface size settings: Auto on or off, and the manual size. Read at every refresh. */
  getSettings(): { auto: boolean; manual: number };
  /** The monitor the window is on in physical pixels. `null` (or a rejection) means it cannot be read. */
  readMonitor(): Promise<MonitorMetrics | null>;
  /** Calls `onChange` whenever the window moves, is resized, or the Windows scale changes. Resolves to the function that stops it. */
  listen(onChange: () => void): Promise<() => void>;
  /** Puts a size in effect. `first` is true for the first write of the run. */
  apply(value: number, first: boolean): void;
  /**
   * The window's size and position, or `null` when it should not be touched: maximized, minimized or fullscreen (a
   * maximized window is as big as the screen, and resizing one would un-maximize it). Without this and `placeWindow`
   * the window is never resized.
   */
  readWindow?(): Promise<WindowMetrics | null>;
  /** Moves and sizes the window. */
  placeWindow?(growth: WindowGrowth): Promise<void>;
  /** Quiet time after the last move, resize or scale event before the monitor is read again (default 250 ms). */
  debounceMs?: number;
  /** How long `start` waits for the first monitor read before it applies 1 and goes on (default 1500 ms). */
  startTimeoutMs?: number;
}

export const UI_SCALE_DEBOUNCE_MS = 250;
export const UI_SCALE_START_TIMEOUT_MS = 1500;

export class UiScaleController {
  /**
   * What Auto picks for the monitor the window is on, kept up to date even while Auto is off (the Settings row shows it),
   * and whether a monitor has been read at all (`known`; false means `auto` is the 1 of "unknown"). Not saved.
   */
  readonly detected = proxy({ auto: 1, known: false });

  #monitor: MonitorMetrics | null = null;
  /** The read that `#monitor` came from. */
  #monitorSeq = 0;
  #applied: number | null = null;
  #timer: ReturnType<typeof setTimeout> | undefined;
  #unlisten: (() => void) | null = null;
  #running = false;
  /** Bumped by every start and stop: a subscription that resolves after a stop must not stay attached. */
  #generation = 0;
  #readSeq = 0;
  /** The last window growth that was asked for: the same request twice in a row is not repeated. */
  #lastGrowth: string | undefined;

  constructor(private readonly deps: UiScaleDeps) { }

  /** The size in effect for the current settings and the last monitor that was read. */
  get effective(): number {
    const { auto, manual } = this.deps.getSettings();
    return effectiveUiScale({ auto, manual, monitor: this.#monitor });
  }

  /**
   * Reads the monitor, puts the size in effect, and starts following the window. Resolves once the first size is
   * applied (the window listeners attach after that without being waited for). Never rejects: a monitor that cannot
   * be read leaves the size at 1 in Auto.
   */
  async start(): Promise<void> {
    if (this.#running) return;
    this.#running = true;
    const generation = ++this.#generation;
    let answered = false;
    const read = this.#readMonitor().then(() => { answered = true; });
    // The app waits for this before it paints: a monitor query that never answers must not hold start-up. The answer,
    // when it comes, is still applied.
    let timer: ReturnType<typeof setTimeout> | undefined;
    const timeout = new Promise<void>(resolve => { timer = setTimeout(resolve, this.deps.startTimeoutMs ?? UI_SCALE_START_TIMEOUT_MS); });
    await Promise.race([read, timeout]);
    clearTimeout(timer);
    if (generation !== this.#generation) return;
    this.refresh();
    if (!answered) {
      void read.then(() => {
        if (generation === this.#generation) this.refresh();
      });
    }
    void this.#subscribe(generation);
  }

  /**
   * Puts the size for the current settings and monitor in effect. Writes only when the value differs from the last one
   * written. Then makes sure the window can hold it (`windowGrowth`), without waiting for that.
   */
  refresh(): void {
    const detected = autoUiScale(this.#monitor);
    if (this.detected.auto !== detected) this.detected.auto = detected;
    const known = this.#monitor !== null;
    if (this.detected.known !== known) this.detected.known = known;
    const value = this.effective;
    if (value !== this.#applied) {
      const first = this.#applied === null;
      this.#applied = value;
      this.deps.apply(value, first);
    }
    void this.#fitWindow();
  }

  /** Stops following the window: the pending read is dropped and the listeners are removed. */
  stop(): void {
    this.#running = false;
    this.#generation++;
    if (this.#timer !== undefined) clearTimeout(this.#timer);
    this.#timer = undefined;
    this.#unlisten?.();
    this.#unlisten = null;
  }

  async #subscribe(generation: number): Promise<void> {
    try {
      const unlisten = await this.deps.listen(this.#windowChanged);
      if (generation !== this.#generation) unlisten();
      else this.#unlisten = unlisten;
    } catch (error) {
      // No window events (no permission, an older shell): the size stays as it was read at start.
      console.warn("[uiScale] the window cannot be followed", error);
    }
  }

  /** A burst of move or resize events (dragging the window) becomes one read once it has been quiet for the debounce time. */
  #windowChanged = (): void => {
    if (!this.#running) return;
    if (this.#timer !== undefined) clearTimeout(this.#timer);
    const generation = this.#generation;
    this.#timer = setTimeout(() => {
      this.#timer = undefined;
      void this.#readMonitor().then(() => {
        if (generation === this.#generation) this.refresh();
      });
    }, this.deps.debounceMs ?? UI_SCALE_DEBOUNCE_MS);
  };

  /**
   * Keeps the newest answer that came back: a read is stored when it is newer than the one already stored, even if a
   * read started after it failed or found nothing. A failed or empty read never replaces what was known (1 in Auto when
   * nothing was), so a hiccup never flips the interface size.
   */
  async #readMonitor(): Promise<void> {
    const seq = ++this.#readSeq;
    let monitor: MonitorMetrics | null = null;
    try {
      monitor = await this.deps.readMonitor();
    } catch (error) {
      console.warn("[uiScale] the monitor cannot be read", error);
    }
    if (!this.#running || monitor === null || seq < this.#monitorSeq) return;
    this.#monitor = monitor;
    this.#monitorSeq = seq;
  }

  /**
   * Auto above 1 asks for a window of at least 800 x 600 interface units; a smaller one (a saved default window, or one
   * made for a smaller monitor) is grown. Runs after every refresh, so it also covers a window that was shrunk or
   * restored from maximized (resize events refresh). A failure only costs the growth.
   */
  async #fitWindow(): Promise<void> {
    const { readWindow, placeWindow } = this.deps;
    if (!readWindow || !placeWindow || !this.#running || !this.deps.getSettings().auto) return;
    const scale = this.effective;
    const monitor = this.#monitor;
    if (!(scale > 1) || !monitor?.workArea) return;
    const generation = this.#generation;
    try {
      const win = await readWindow();
      if (!win || generation !== this.#generation) return;
      const growth = windowGrowth(scale, win, monitor);
      if (!growth) {
        this.#lastGrowth = undefined; // the window fits: a later shrink to the same size is a new request
        return;
      }
      // A window the system will not grow (it gave less than asked) would be asked again by every resize event.
      const request = JSON.stringify([win.innerWidth, win.innerHeight, growth]);
      if (request === this.#lastGrowth) return;
      this.#lastGrowth = request;
      await placeWindow(growth);
    } catch (error) {
      console.warn("[uiScale] the window cannot be resized", error);
    }
  }
}

/** The monitor and window side of the controller on the Tauri window; outside Tauri (the browser build) there is no monitor. */
export function tauriWindowDeps(): Pick<UiScaleDeps, "readMonitor" | "listen" | "readWindow" | "placeWindow"> {
  return {
    async readMonitor() {
      if (!isTauri()) return null;
      const monitor = await currentMonitor();
      if (!monitor) return null;
      const { workArea } = monitor;
      return {
        width: monitor.size.width,
        height: monitor.size.height,
        scaleFactor: monitor.scaleFactor,
        workArea: workArea ? { x: workArea.position.x, y: workArea.position.y, width: workArea.size.width, height: workArea.size.height } : undefined,
      };
    },
    async listen(onChange) {
      if (!isTauri()) return () => { };
      const win = getCurrentWindow();
      // A maximized window changes size when the desktop resolution changes, without moving or changing scale.
      const attach = [() => win.onMoved(onChange), () => win.onScaleChanged(onChange), () => win.onResized(onChange)];
      const results = await Promise.allSettled(attach.map(async on => on()));
      const unlisteners = results.flatMap(r => (r.status === "fulfilled" ? [r.value] : []));
      const failed = results.find((r): r is PromiseRejectedResult => r.status === "rejected");
      if (failed) {
        unlisteners.forEach(fn => fn());
        throw failed.reason;
      }
      return () => unlisteners.forEach(fn => fn());
    },
    async readWindow() {
      if (!isTauri()) return null;
      const win = getCurrentWindow();
      const [maximized, minimized, fullscreen] = await Promise.all([win.isMaximized(), win.isMinimized(), win.isFullscreen()]);
      if (maximized || minimized || fullscreen) return null;
      const [inner, position, outer] = await Promise.all([win.innerSize(), win.outerPosition(), win.outerSize()]);
      return {
        innerWidth: inner.width,
        innerHeight: inner.height,
        outerX: position.x,
        outerY: position.y,
        outerWidth: outer.width,
        outerHeight: outer.height,
      };
    },
    async placeWindow(growth) {
      if (!isTauri()) return;
      const win = getCurrentWindow();
      // The order of the window-state plugin's own restore: position, then size.
      await win.setPosition(new PhysicalPosition(growth.x, growth.y));
      await win.setSize(new PhysicalSize(growth.width, growth.height));
    },
  };
}

// endregion
