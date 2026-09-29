import { listen } from "@tauri-apps/api/event";
import { invoke, isTauri } from "@tauri-apps/api/core";
import i18n from "i18next";
import { toast } from "react-toastify";
import { proxy, subscribe }                                  from "valtio";
import { IServiceInterface, TextEventSource, TextEventType } from "@/types";
import { BackendState }                                      from "../../schema";
import * as globalShortcut from "@tauri-apps/plugin-global-shortcut"
import { devLog } from "@/utils/devLog";
import { pushSystemLog } from "@/core/services/systemLog";
import { i18nReady } from "@/core/services/stt/statusMessages";
import { acceleratorKey, isValidAccelerator, recordKeyDown, ShortcutPressGate } from "./accelerator";

type ShortcutKeys = keyof BackendState["shortcuts"];
type InputCommands = "submit" | "delete" | "cancel";

/** A shortcut recording that gets no complete key combination in this time is cancelled (shortcuts rebound). */
const RECORD_TIMEOUT_MS = 15_000;

/**
 * pubsub
 * keyboard.key string
 * keyboard.cmd left | right | submit | delete
 */
class Service_Keyboard implements IServiceInterface {
  ui = proxy<{
    currentTarget: ShortcutKeys | "",
    showRecorder: boolean,
    /** Keys held so far while recording (e.g. "Control+Shift+"), for the recorder UI. */
    currentValue: string,
    /** Why the last key was not accepted while recording ("" = none; the UI shows `stt.shortcut_hint_<value>`). */
    recorderHint: "" | "needs_ctrl_alt" | "unsupported" | "in_use",
    backgroundInputActive: boolean
    backgroundInputValue: string
    backgroundTimer: number // in %
  }>({
    currentTarget: "",
    showRecorder: false,
    currentValue: "",
    recorderHint: "",
    backgroundInputActive: false,
    backgroundInputValue: "",
    backgroundTimer: 0,
  });

  readonly #pressGate = new ShortcutPressGate();
  /** Registrations run one after another: an overlapping unregisterAll/register pair fails with "already registered". */
  #bindChain: Promise<void> = Promise.resolve();
  #recordKeyHandler: ((e: KeyboardEvent) => void) | null = null;
  #recordTimer: ReturnType<typeof setTimeout> | null = null;

  async init() {
    // Global shortcuts (start/stop STT, mute STT, mute sound effects) work in every desktop build; only the
    // background text input needs the low-level keyboard hook (`background_input` feature).
    if (!isTauri())
      return;

    // Rebind whenever the stored shortcuts change (recorder, clear, settings import).
    subscribe(window.ApiServer.state.shortcuts, () => {
      if (!this.ui.showRecorder)
        void this.rebindShortcuts();
    });
    // Not awaited: host services init before the translations load, and a shortcut that cannot be registered is
    // reported with a translated toast.
    void i18nReady().then(() => this.rebindShortcuts());

    if (!window.Config.features.background_input)
      return;

    // send hotkeys
    listen<string>('keyboard', e => {
      const str = e.payload;
      str.startsWith("shortcut:") && this.processShortcut(str.slice(9) as ShortcutKeys)
      str.startsWith("key:") && this.processKey(str.slice(4));
      str.startsWith("cmd:") && this.processCommand(str.slice(4) as InputCommands);
    });
  }

  inputUpdateValue(value: string) {
    this.ui.backgroundInputValue  = value
    window.ApiShared.pubsub.publishText(TextEventSource.textfield, { type: TextEventType.interim, value });
  }

  private processKey(key: string) {
    this.ui.backgroundInputValue += key;
    this.triggerBackgroundTimer()
  }
  private processCommand(command: InputCommands) {
    if (command === "submit") {
      window.ApiShared.pubsub.publishText(TextEventSource.textfield, { type: TextEventType.final, value: this.ui.backgroundInputValue });
      this.stopBackgroundInput();
    }
    if (command === "cancel") {
      this.stopBackgroundInput();
    }

    if (command === "delete") {
      this.ui.backgroundInputValue = this.ui.backgroundInputValue.slice(0, -1);
      this.triggerBackgroundTimer()
    }
  }

  private processShortcut(shortcut: ShortcutKeys) {
    switch (shortcut) {
      case "bgInput":
        if (window.Config.features.background_input)
          this.startBackgroundInput();
        break;
      case "start":
        window.ApiServer.stt.toggleRunning();
        break;
      case "muteMic":
        window.ApiServer.stt.toggleMute();
        break;
      case "muteSound": {
        const muted = !window.ApiServer.state.muteSoundEffects;
        window.ApiServer.state.muteSoundEffects = muted;
        pushSystemLog("Sound effects", muted ? "Muted (shortcut)" : "Unmuted (shortcut)", "info");
        break;
      }
    }
  }

  // start/restart background timer
  backgroundTimer: NodeJS.Timeout | null = null;
  backgroundInterval: number | null = null;
  private triggerBackgroundTimer() {
    if (this.backgroundTimer !== null) {
      clearTimeout(this.backgroundTimer);
      clearInterval(this.backgroundInterval as number);
    }

    // min 1 second
    const time = Math.max(parseInt(window.ApiServer.state.backgroundInputTimer) || 5000, 1000);
    this.backgroundTimer = setTimeout(() => this.stopBackgroundInput(), time);
    this.ui.backgroundTimer = 100;
    this.backgroundInterval = setInterval(() => {
      this.ui.backgroundTimer -= 1;
    }, time/100) as unknown as number;
  }
  private startBackgroundInput() {
    if (!this.ui.backgroundInputActive) {
      invoke("plugin:keyboard|start_tracking");
      this.ui.backgroundInputActive = true;
    }
    this.triggerBackgroundTimer();
  }
  stopBackgroundInput() {
    // for immidiate cancel
    if (this.backgroundTimer !== null) {
      clearTimeout(this.backgroundTimer);
      clearInterval(this.backgroundInterval as number);
    }
    this.ui.backgroundInputActive = false;
    this.backgroundTimer = null;
    this.ui.backgroundInputValue = "";
    invoke("plugin:keyboard|stop_tracking");
  }

  /** Re-registers every stored shortcut (serialized; resolves when this rebind finished). */
  rebindShortcuts(): Promise<void> {
    this.#bindChain = this.#bindChain.then(() => this.#rebindShortcutsTauri(), () => this.#rebindShortcutsTauri());
    return this.#bindChain;
  }

  async #rebindShortcutsTauri(): Promise<void> {
    if (!isTauri())
      return;
    this.#pressGate.reset();
    try {
      await globalShortcut.unregisterAll();
    } catch (err) {
      devLog("[Keyboard] unregisterAll failed", err);
    }
    const shortcuts = window.ApiServer.state.shortcuts;
    for (const k of Object.keys(shortcuts) as ShortcutKeys[]) {
      const accelerator = shortcuts[k];
      if (!accelerator)
        continue;
      if (k === "bgInput" && !window.Config.features.background_input)
        continue;
      if (!isValidAccelerator(accelerator)) {
        // Can never register (e.g. saved by an older recorder): clear it once so it does not fail every launch.
        toast.error(i18n.t("toasts.invalid_shortkey", { key: accelerator }));
        shortcuts[k] = "";
        continue;
      }
      try {
        await globalShortcut.register(accelerator, (event) => {
          if (this.#pressGate.accept(accelerator, event.state, Date.now()))
            this.processShortcut(k);
        });
      } catch (err) {
        // Valid but taken (usually by another app): keep the setting, it may be free next launch.
        devLog("[Keyboard] register failed", accelerator, err);
        toast.warning(i18n.t("toasts.shortcut_unavailable", { key: accelerator }));
        pushSystemLog("Keyboard", `Shortcut ${accelerator} could not be registered: ${String(err)}`, "warning");
      }
    }
  }

  private rebindShortcutsNative() {
    const shortcuts =  Object
      .entries(window.ApiServer.state.shortcuts)
      .filter(sc => !!sc[1])
      .map(([key, value]) => ({name: key, keys: value.split("+")}));
    invoke("plugin:keyboard|apply_shortcuts", {shortcuts});
  }

  clearShortcut(shortcut: ShortcutKeys) {
    if (this.ui.showRecorder && this.ui.currentTarget === shortcut)
      this.cancelComboRecord();
    window.ApiServer.state.shortcuts[shortcut] = "";
  }

  /**
   * Records the next key combination for `target`: modifiers are collected, the first main key completes it (no
   * confirm step). Escape cancels; a letter/digit with no modifier, or with Shift only, asks for Control, Alt or
   * Super (it would block typing system-wide); a combination already used by another shortcut is refused. Global
   * shortcuts are paused while recording so pressing an existing one records it instead of firing it.
   */
  startShortcutRecord(target: ShortcutKeys) {
    this.#stopRecording();
    this.ui.currentTarget = target;
    this.ui.showRecorder = true;
    this.ui.currentValue = "";
    this.ui.recorderHint = "";
    this.#bindChain = this.#bindChain.then(() => globalShortcut.unregisterAll()).catch(() => {});

    const handler = (e: KeyboardEvent) => {
      e.preventDefault();
      e.stopPropagation();
      if (e.repeat)
        return;
      const step = recordKeyDown(e);
      switch (step.kind) {
        case "pending":
          this.ui.currentValue = step.preview;
          this.ui.recorderHint = "";
          return;
        case "cancel":
          this.cancelComboRecord();
          return;
        case "needs_modifier":
          this.ui.currentValue = step.preview;
          this.ui.recorderHint = "needs_ctrl_alt";
          return;
        case "unsupported":
          this.ui.recorderHint = "unsupported";
          return;
        case "shortcut":
          this.#finishRecording(step.value);
          return;
      }
    };
    this.#recordKeyHandler = handler;
    window.addEventListener("keydown", handler, true);
    this.#recordTimer = setTimeout(() => this.cancelComboRecord(), RECORD_TIMEOUT_MS);
  }

  #finishRecording(value: string) {
    const target = this.ui.currentTarget;
    if (!target) {
      this.cancelComboRecord();
      return;
    }
    const shortcuts = window.ApiServer.state.shortcuts;
    const key = acceleratorKey(value);
    const taken = (Object.keys(shortcuts) as ShortcutKeys[])
      .some(k => k !== target && shortcuts[k] !== "" && acceleratorKey(shortcuts[k]) === key);
    if (taken) {
      this.ui.currentValue = value;
      this.ui.recorderHint = "in_use";
      return;
    }
    this.#stopRecording();
    if (shortcuts[target] === value)
      void this.rebindShortcuts(); // unchanged: the store subscription will not fire, but recording unbound everything
    else
      shortcuts[target] = value; // the store subscription rebinds
  }

  /** Kept for callers of the old two-step recorder: recording now completes on the first full combination. */
  confirmShortcutRecord() {
    this.cancelComboRecord();
  }

  cancelComboRecord() {
    this.#stopRecording();
    void this.rebindShortcuts();
  }

  #stopRecording() {
    if (this.#recordKeyHandler) {
      window.removeEventListener("keydown", this.#recordKeyHandler, true);
      this.#recordKeyHandler = null;
    }
    if (this.#recordTimer !== null) {
      clearTimeout(this.#recordTimer);
      this.#recordTimer = null;
    }
    this.ui.showRecorder = false;
    this.ui.currentTarget = "";
    this.ui.currentValue = "";
    this.ui.recorderHint = "";
  }
}

export default Service_Keyboard;
