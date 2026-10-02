import { invoke } from "@tauri-apps/api/core";
import { emitTo } from "@tauri-apps/api/event";
import i18n from "i18next";
import type { RustCaptionWindowOptions, RustVrOverlayConfig } from "@/types";
import { CAPTION_FRAME_EVENT, CAPTION_WINDOW_LABEL, type CaptionFrame } from "@/overlay/frame";
import { pushSystemLog } from "../systemLog";
import {
  captionWindowOptions,
  captionWindowStyle,
  feedOptions,
  toFrameLines,
  toVrCaption,
  vrOverlayConfig,
  VR_LIMITS,
  WINDOW_LIMITS,
  type CaptionWindowStatusView,
  type VrOverlaySettings,
  type VrOverlayStatusView,
} from "./config";
import { CaptionFeed, type CaptionBus } from "./feed";
import type { CaptionWindowState } from "./schema";

/**
 * The two caption displays of the overlay service (P6-B), each with its own caption feed. Kept apart from the
 * service (`index.ts`, which wires them to `window.ApiServer` and the pubsub) so they are unit-tested in node with the
 * Tauri APIs mocked.
 */

/** Re-send the latest frame this often while the caption window is open (a late or reloaded page catches up). */
export const FRAME_HEARTBEAT_MS = 1000;
/** Coalesce caption bursts into one frame. */
export const FRAME_COALESCE_MS = 30;
/** `vr_overlay_set_text` at most this often (Rust renders at most 10 Hz too). */
export const VR_TEXT_INTERVAL_MS = 100;

const LOG_LABEL = "Caption overlay";

/** UI mirror of both displays (`OverlaySection.tsx`). */
export interface OverlayUiState {
  window: CaptionWindowStatusView;
  vr: VrOverlayStatusView;
}

function errorText(e: unknown): string {
  return e instanceof Error ? e.message : String(e);
}

function sameWindowOptions(a: RustCaptionWindowOptions | null, b: RustCaptionWindowOptions): boolean {
  return !!a && a.alwaysOnTop === b.alwaysOnTop && a.clickThrough === b.clickThrough;
}

/** Runs async steps one after another (open / close / options never overtake each other). */
class Serial {
  #chain: Promise<void> = Promise.resolve();
  run(step: () => Promise<void>): void {
    this.#chain = this.#chain.then(step).catch((e) => console.error("[overlay]", e));
  }
  /** Resolves once every queued step has run (tests). */
  idle(): Promise<void> {
    return this.#chain;
  }
}

/**
 * The always-on-top caption window (`caption-window` plugin): opens / closes it with `captionWindow.enable`, pushes
 * always-on-top and click-through, and sends it frames (`src/overlay/frame.ts`) built from its caption feed.
 */
export class CaptionWindowDriver {
  readonly feed: CaptionFeed;
  readonly #ui: OverlayUiState;
  readonly #onClosedExternally: () => void;
  readonly #serial = new Serial();
  #settings: CaptionWindowState | null = null;
  #settingsKey = "";
  /** The user wants the window (enable is on). */
  #wanted = false;
  #applied: RustCaptionWindowOptions | null = null;
  #open = false;
  #seq = 0;
  #frame: CaptionFrame | null = null;
  #frameKey = "";
  #sendTimer: ReturnType<typeof setTimeout> | null = null;
  #heartbeat: ReturnType<typeof setInterval> | null = null;

  /**
   * @param onClosedExternally the window was closed by something other than the toggle (Alt+F4, the OS) while it was
   *   wanted: the service turns `captionWindow.enable` off so it does not come back at the next start.
   */
  constructor(ui: OverlayUiState, bus: CaptionBus, onClosedExternally: () => void) {
    this.#ui = ui;
    this.#onClosedExternally = onClosedExternally;
    this.feed = new CaptionFeed({ interim: true, maxLines: 3, clearAfterMs: 6000 }, bus, () => this.rebuild());
  }

  sync(settings: CaptionWindowState): void {
    const key = JSON.stringify(settings);
    if (key === this.#settingsKey) return;
    this.#settingsKey = key;
    const s = JSON.parse(key) as CaptionWindowState;
    this.#settings = s;
    if (s.enable) this.feed.configure(s.sources, feedOptions(s, WINDOW_LIMITS.maxLines));
    else this.feed.stop();
    this.rebuild();

    const options = captionWindowOptions(s);
    if (s.enable && !this.#wanted) {
      this.#wanted = true;
      this.#applied = options;
      this.#serial.run(() => this.#call("caption_window_open", { options }));
    } else if (!s.enable && this.#wanted) {
      this.#wanted = false;
      this.#applied = null;
      this.#serial.run(async () => {
        try {
          await invoke("plugin:caption-window|caption_window_close");
        } catch (e) {
          pushSystemLog(LOG_LABEL, `Closing the caption window failed: ${errorText(e)}`, "warning");
        }
      });
    } else if (s.enable && !sameWindowOptions(this.#applied, options)) {
      this.#applied = options;
      this.#serial.run(() => this.#call("caption_window_set_options", { options }));
    }
  }

  async #call(command: string, args: Record<string, unknown>): Promise<void> {
    try {
      const status = await invoke<CaptionWindowStatusView>(`plugin:caption-window|${command}`, args);
      this.onStatus(status);
      if (status.error) pushSystemLog(LOG_LABEL, status.error, "warning");
    } catch (e) {
      this.#ui.window = { ...this.#ui.window, error: errorText(e) };
      pushSystemLog(LOG_LABEL, `Caption window: ${errorText(e)}`, "warning");
    }
  }

  onStatus(status: CaptionWindowStatusView): void {
    this.#ui.window = { ...status, closedExternally: false };
    const wasOpen = this.#open;
    this.#open = status.open;
    if (status.open) {
      if (this.#heartbeat === null) this.#heartbeat = setInterval(() => this.#send(), FRAME_HEARTBEAT_MS);
      if (!wasOpen) this.#send();
      return;
    }
    if (this.#heartbeat !== null) clearInterval(this.#heartbeat);
    this.#heartbeat = null;
    // Closed by something other than the toggle (Rust tells them apart, so a quick off-on never counts).
    if (status.closedExternally && this.#wanted) this.#onClosedExternally();
  }

  /** Rebuild the frame (settings, lines or UI language changed) and send it when it differs. */
  rebuild(): void {
    const s = this.#settings;
    if (!s) return;
    const content = {
      style: captionWindowStyle(s),
      unlocked: !s.clickThrough,
      hint: i18n.t("overlay.window_unlocked_hint"),
      lines: toFrameLines(this.feed.model.lines),
    };
    const key = JSON.stringify(content);
    if (key === this.#frameKey) return;
    this.#frameKey = key;
    this.#frame = { seq: ++this.#seq, ...content };
    if (this.#sendTimer === null) {
      this.#sendTimer = setTimeout(() => {
        this.#sendTimer = null;
        this.#send();
      }, FRAME_COALESCE_MS);
    }
  }

  #send(): void {
    if (!this.#open || !this.#frame) return;
    emitTo(CAPTION_WINDOW_LABEL, CAPTION_FRAME_EVENT, this.#frame).catch(() => { });
  }

  resetPosition(): void {
    this.#serial.run(() => this.#call("caption_window_reset_position", {}));
  }

  /** Resolves once every queued open / close / options call has finished (tests). */
  idle(): Promise<void> {
    return this.#serial.idle();
  }
}

/**
 * The SteamVR overlay (`vr-overlay` plugin): starts / stops it with `services.vrc.data.overlay.enable`, pushes its
 * config, and sends the caption lines at most 10 times a second (only when they changed).
 */
export class VrOverlayDriver {
  readonly feed: CaptionFeed;
  readonly #ui: OverlayUiState;
  readonly #serial = new Serial();
  #settingsKey = "";
  #wanted = false;
  #configKey = "";
  #lastText = "";
  #textTimer: ReturnType<typeof setTimeout> | null = null;

  constructor(ui: OverlayUiState, bus: CaptionBus) {
    this.#ui = ui;
    this.feed = new CaptionFeed({ interim: true, maxLines: 3, clearAfterMs: 6000 }, bus, () => this.#scheduleText());
  }

  sync(settings: VrOverlaySettings): void {
    const key = JSON.stringify(settings);
    if (key === this.#settingsKey) return;
    this.#settingsKey = key;
    const s = JSON.parse(key) as VrOverlaySettings;
    if (s.enable) this.feed.configure(s.sources, feedOptions(s, VR_LIMITS.maxLines));
    else this.feed.stop();

    const config = vrOverlayConfig(s);
    const configKey = JSON.stringify(config);
    if (s.enable && !this.#wanted) {
      this.#wanted = true;
      this.#configKey = configKey;
      this.#serial.run(() => this.#start(config));
    } else if (!s.enable && this.#wanted) {
      this.#wanted = false;
      this.#configKey = "";
      this.#serial.run(async () => {
        try {
          await invoke("plugin:vr-overlay|vr_overlay_stop");
        } catch (e) {
          pushSystemLog(LOG_LABEL, `Stopping the SteamVR overlay failed: ${errorText(e)}`, "warning");
        }
      });
    } else if (s.enable && configKey !== this.#configKey) {
      this.#configKey = configKey;
      this.#serial.run(() => this.#status("vr_overlay_set_config", { config }));
    }
  }

  async #start(config: RustVrOverlayConfig): Promise<void> {
    await this.#status("vr_overlay_start", { config });
    // Whatever is on screen now goes to the new overlay.
    this.#lastText = "";
    this.#scheduleText();
  }

  async #status(command: string, args: Record<string, unknown>): Promise<void> {
    try {
      this.onStatus(await invoke<VrOverlayStatusView>(`plugin:vr-overlay|${command}`, args));
    } catch (e) {
      this.#ui.vr = { ...this.#ui.vr, state: "error", error: errorText(e) };
      pushSystemLog(LOG_LABEL, `SteamVR overlay: ${errorText(e)}`, "warning");
    }
  }

  onStatus(status: VrOverlayStatusView): void {
    this.#ui.vr = { ...status, reason: status.reason ?? null };
  }

  /** Ask Rust for the state (runtime installed?) without starting anything. */
  refresh(): void {
    this.#serial.run(() => this.#status("vr_overlay_status", {}));
  }

  recenter(): void {
    invoke("plugin:vr-overlay|vr_overlay_recenter").catch((e) => console.error("[overlay] recenter", e));
  }

  /** Resolves once every queued start / stop / config call has finished (tests). */
  idle(): Promise<void> {
    return this.#serial.idle();
  }

  #scheduleText(): void {
    if (!this.#wanted || this.#textTimer !== null) return;
    this.#textTimer = setTimeout(() => {
      this.#textTimer = null;
      this.#sendText();
    }, VR_TEXT_INTERVAL_MS);
  }

  #sendText(): void {
    if (!this.#wanted) return;
    const caption = toVrCaption(this.feed.model.lines);
    const key = JSON.stringify(caption);
    if (key === this.#lastText) return;
    this.#lastText = key;
    invoke("plugin:vr-overlay|vr_overlay_set_text", { caption }).catch((e) => console.error("[overlay] set_text", e));
  }
}
