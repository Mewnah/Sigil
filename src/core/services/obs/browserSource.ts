import OBSWebSocket from "obs-websocket-js";
import { localPageUrl } from "@/utils/localPages";
import { normalizeLanToken } from "@/shared/services/peer/lanAccess";

/**
 * Where an OBS link points (P6-D): OBS on this PC (default; `localhost`, never needs the LAN token)
 * or OBS / a browser on another device (`lan`: this PC's LAN address plus the LAN token).
 */
export type ObsLinkTarget = { lan?: false } | { lan: true; /** One of this PC's addresses; default the primary one. */ ip?: string };

/** The host's LAN token, or "" while none is set. */
function hostLanToken(): string {
  return normalizeLanToken(window.ApiServer?.state?.lanAccess?.token) ?? "";
}

export class ObsBrowserSourceService {
  /** Canvas mirror for OBS browser source (same as upstream Curses `/client` link). */
  getObsBrowserSourceLink(target: ObsLinkTarget = {}): string {
    if (target.lan)
      return window.ApiShared.peer.getLanClientLink("/client", target.ip);
    return window.ApiShared.peer.getClientLink();
  }

  /** Standalone captions page (pubsub); not exposed in the OBS inspector. */
  getCaptionsLink(options?: {
    mode?: string;
    size?: string;
    lines?: string;
  }, target: ObsLinkTarget = {}): string {
    const lanHost = target.lan ? target.ip || window.Config.serverNetwork.ip : undefined;
    // Empty values are left out, as before.
    return localPageUrl("obs-captions.html", {
      host: lanHost,
      token: lanHost ? hostLanToken() : undefined,
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

