import OBSWebSocket from "obs-websocket-js";
import { localPageUrl } from "@/utils/localPages";

export class ObsBrowserSourceService {
  /** Canvas mirror for OBS browser source (same as upstream Curses `/client` link). */
  getObsBrowserSourceLink(): string {
    return window.ApiShared.peer.getClientLink();
  }

  /** Standalone captions page (pubsub); not exposed in the OBS inspector. */
  getCaptionsLink(options?: {
    mode?: string;
    size?: string;
    lines?: string;
  }): string {
    // Empty values are left out, as before.
    return localPageUrl("obs-captions.html", {
      params: {
        mode: options?.mode || undefined,
        size: options?.size || undefined,
        lines: options?.lines || undefined,
      },
    });
  }

  async setupObsScene({
    name,
    port,
    password,
  }: {
    name: string;
    port: string;
    password: string;
  }) {
    const obs = new OBSWebSocket();
    try {
      await obs.connect(`ws://127.0.0.1:${port}`, password);
      const activeScene = await obs.call("GetCurrentProgramScene");
      const canvas = window.ApiClient.document.fileBinder.get().canvas;
      await obs.call("CreateInput", {
        sceneName: activeScene.currentProgramSceneName,
        inputName: name,
        inputKind: "browser_source",
        inputSettings: {
          url: this.getObsBrowserSourceLink(),
          width: canvas.w,
          height: canvas.h,
        },
      });
      return "";
    } catch (error: unknown) {
      return error instanceof Error ? error.message : "Something went wrong";
    } finally {
      obs.disconnect();
    }
  }
}

