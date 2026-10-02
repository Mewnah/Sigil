import { isTauri } from "@tauri-apps/api/core";
import { listen } from "@tauri-apps/api/event";
import i18n from "i18next";
import { proxy, subscribe } from "valtio";
import { CAPTION_WINDOW, VR_OVERLAY_STATE_EVENT, type IServiceInterface } from "@/types";
import type { CaptionWindowStatusView, VrOverlayStatusView } from "./config";
import { CaptionWindowDriver, VrOverlayDriver, type OverlayUiState } from "./drivers";
import type { CaptionBus } from "./feed";

export type { OverlayUiState } from "./drivers";

function pubsubBus(): CaptionBus {
  return {
    subscribe: (source, fn) => window.ApiShared.pubsub.subscribeText(source, (event, topic) => fn(event, topic), true),
    unsubscribe: (token) => window.ApiShared.pubsub.unsubscribe(token),
  };
}

/**
 * Caption overlays (P6-B): the always-on-top, click-through caption window (root settings key `captionWindow`,
 * `caption-window` plugin, early route `/captions` in `src/overlay/**`) and the SteamVR overlay
 * (`services.vrc.data.overlay`, `vr-overlay` plugin). Each display keeps its own caption feed from the topics it
 * shows (`text.*` and `others.*`). Both default off; nothing here throws when VR is absent. Not a sidebar tab: the UI
 * is `src/core/ui/inspector/vrc/OverlaySection.tsx` (in the VRChat inspector).
 */
class Service_Overlay implements IServiceInterface {
  readonly state: OverlayUiState = proxy<OverlayUiState>({
    window: { open: false, options: { alwaysOnTop: true, clickThrough: true }, error: null, closedExternally: false },
    vr: { state: "off", runtimeInstalled: false, error: null, reason: null },
  });
  #window: CaptionWindowDriver | null = null;
  #vr: VrOverlayDriver | null = null;

  /** Start whatever the saved settings enable (both default off). Never throws when VR is absent. */
  async init() {
    if (this.#window || !isTauri()) return;
    const bus = pubsubBus();
    const win = new CaptionWindowDriver(this.state, bus, () => {
      // Closed by Alt+F4 or the OS: turn the setting off so the window does not come back at the next start.
      const settings = window.ApiServer?.state?.captionWindow;
      if (settings?.enable) settings.enable = false;
    });
    const vr = new VrOverlayDriver(this.state, bus);
    this.#window = win;
    this.#vr = vr;
    try {
      await listen<CaptionWindowStatusView>(CAPTION_WINDOW.stateEvent, (e) => win.onStatus(e.payload));
      await listen<VrOverlayStatusView>(VR_OVERLAY_STATE_EVENT, (e) => vr.onStatus(e.payload));
    } catch (e) {
      console.error("[overlay] cannot listen for overlay state", e);
    }
    const root = window.ApiServer.state;
    const syncAll = () => {
      win.sync(root.captionWindow);
      vr.sync(root.services.vrc.data.overlay);
    };
    // The whole settings object: a settings import may replace the slices themselves.
    subscribe(root, syncAll);
    i18n.on("languageChanged", () => win.rebuild());
    syncAll();
    if (!root.services.vrc.data.overlay.enable) vr.refresh();
  }

  /** Show a sample caption in both displays for a few seconds (setup check). */
  showTestCaption() {
    const text = i18n.t("overlay.test_caption");
    this.#window?.feed.showTest(text);
    this.#vr?.feed.showTest(text);
  }

  /** Move the caption window back to its default place and size. */
  resetWindowPosition() {
    this.#window?.resetPosition();
  }

  /** Place a world-attached SteamVR overlay in front of the user again. */
  recenterVr() {
    this.#vr?.recenter();
  }

  /** Re-read the SteamVR overlay state (after the runtime pack was installed). */
  refreshVrStatus() {
    this.#vr?.refresh();
  }
}

export default Service_Overlay;
