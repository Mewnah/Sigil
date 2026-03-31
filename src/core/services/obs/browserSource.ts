import OBSWebSocket from "obs-websocket-js";

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
    const n = window.Config.serverNetwork;
    const p = String(n.port);
    const q = new URLSearchParams({ port: p });
    if (options?.mode) q.set("mode", options.mode);
    if (options?.size) q.set("size", options.size);
    if (options?.lines) q.set("lines", options.lines);
    return `http://${n.host}:${p}/obs-captions.html?${q.toString()}`;
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

