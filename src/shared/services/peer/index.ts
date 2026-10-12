import { BaseEvent, IServiceInterface } from "@/types";
import i18n from "i18next";
import { toast } from "react-toastify";
import { proxy } from "valtio";
import {
  PeerjsProvider as PeerProvider,
  PEER_CLIENT_MIRROR_EVENT,
  PEER_INVALID_KEY,
  peerErrorType,
  type PeerClientMirrorDetail,
} from "./provider";
import { lanClientUrl, normalizeLanToken } from "./lanAccess";
import { invalidKeyMessage, MirrorRefusedError, MirrorStartError } from "./errors";
import AppConfiguration from "@/config";
import { getApiClient, getApiShared, getConfig } from "@/runtime/host";
import { sanitizeClientPubSubFrame } from "@/shared/services/pubsub/inbound";

export { MirrorRefusedError, MirrorStartError };

class Service_Peer implements IServiceInterface {
  #provider?: PeerProvider;

  /** Browser `/client` only: connection phase for OBS diagnostics. */
  clientMirrorState = proxy({
    phase: "idle" as "idle" | "connecting" | "synced" | "reconnecting" | "failed",
    lastError: "",
    /** Attempts that failed since the last time the mirror was in sync (the loading view's hint). */
    failures: 0,
  });

  broadcast(msg: BaseEvent) {
    this.#provider?.broadcastPubSub(msg);
  }

  /** When `data` is already JSON-serialized (same string `JSON.stringify` would produce). */
  broadcastPubSubParts(topic: string, dataJson: string) {
    this.#provider?.broadcastPubSubParts(topic, dataJson);
  }

  /**
   * Full URL for a browser or OBS on this PC (Curses-style: `http://localhost:PORT/client`, no
   * query required). Same-PC clients never need the LAN token, so it carries none.
   */
  getClientLink(pathname = "/client"): string {
    const n = getConfig().serverNetwork;
    const p = String(n.port);
    return `http://${n.host}:${p}${pathname}`;
  }

  copyClientLink(pathname = "/client") {
    navigator.clipboard.writeText(getApiShared().peer.getClientLink(pathname));
    toast.success(i18n.t("toasts.copied"));
  }

  /**
   * URL for a browser or OBS on another device (P6-D): this PC's LAN address (`ip`, default the
   * primary one) with the host's LAN token as `?token=`. Host only.
   */
  getLanClientLink(pathname = "/client", ip?: string): string {
    const n = getConfig().serverNetwork;
    const token = normalizeLanToken(window.ApiServer?.state?.lanAccess?.token) ?? "";
    return lanClientUrl(ip || n.ip, n.port, token, pathname);
  }

  startServer() {
    this.#initializePeer();
    this.#provider?.addEventListener("on_client_connected", e => {
      if (e instanceof CustomEvent) {
        this.#provider?.broadcastPubSubSingle(e.detail, { topic: "peers:init_data", data: getApiClient().getInitialConfig() });
      }
    });

    this.#provider?.connectServer({
      id: "server",
      // Broker WebSocket must use IPv4 loopback; page URL can still be http://localhost/…
      host: "127.0.0.1",
      port: getConfig().serverNetwork.port,
    });
  }
  /** Set until the first `peers:init_data` arrives (it answers `startClient`). */
  private onConfigReceived?: (data: any) => any;

  async startClient() {
    this.clientMirrorState.phase = "connecting";
    this.clientMirrorState.lastError = "";
    this.clientMirrorState.failures = 0;
    this.#initializePeer();
    const onMirror: EventListener = (e) => {
      if (!(e instanceof CustomEvent)) return;
      const d = e.detail as PeerClientMirrorDetail;
      const detail = (message?: string) => (message === PEER_INVALID_KEY ? invalidKeyMessage() : message ?? "");
      if (d.phase === "synced") {
        this.clientMirrorState.phase = "synced";
        this.clientMirrorState.lastError = "";
        this.clientMirrorState.failures = 0;
      } else if (d.phase === "reconnecting") {
        this.clientMirrorState.phase = "reconnecting";
      } else if (d.phase === "failed") {
        this.clientMirrorState.phase = "failed";
        this.clientMirrorState.lastError = detail(d.message);
        this.clientMirrorState.failures++;
      } else if (d.phase === "error") {
        this.clientMirrorState.lastError = detail(d.message);
        this.clientMirrorState.phase = "reconnecting";
        this.clientMirrorState.failures++;
      }
    };
    this.#provider?.addEventListener(PEER_CLIENT_MIRROR_EVENT, onMirror);
    // The runtime config the host sends first on every channel it opens. Ready before connecting,
    // so the first copy can never arrive ahead of its receiver.
    const runtimeConfig = new Promise<AppConfiguration["clientInitialState"]>((res) => this.onConfigReceived = res);
    this.#provider?.addEventListener("on_event_received", (e) => {
      if (e instanceof CustomEvent) try {
        if (e.detail.topic === "peers:init_data")
          this.#onHostConfig(e.detail.data);
        else {
          // Frames from the host are still validated: text.* must match TextEventSchema, emotes keep
          // only safe CDN URLs and markup in the text is neutralized; other topics pass through.
          const frame = sanitizeClientPubSubFrame(e.detail);
          if (frame)
            getApiShared().pubsub.publishLocally(frame);
        }
      } catch (error) {
        console.error(error)
      }
    });
    const network = getConfig().clientNetwork;
    try {
      // Resolves on the first open channel, however many attempts that takes; rejects only for a
      // refused LAN token (and the errors no retry can fix).
      await this.#provider?.connectClient({
        id: "server",
        host: network.host,
        port: network.port,
        token: network.token,
      });
    } catch (err) {
      this.clientMirrorState.phase = "failed";
      // Everything connectClient rejects with is final: the refused token, a browser without
      // WebRTC, a Peer that cannot be constructed. PeerJS's own invalid-key message quotes the key
      // (the token), and so can the browser's error for a bad URL: neither is shown as it is.
      const final = peerErrorType(err) === PEER_INVALID_KEY ? new MirrorRefusedError() : new MirrorStartError(err, network.token);
      this.clientMirrorState.lastError = final.message;
      throw final;
    }
    // wait for runtime config
    return runtimeConfig;
  }

  /**
   * `peers:init_data` from the host: the active scene and UI language. The host sends it on every
   * channel it opens, so a copy after the first means this mirror reconnected: whatever changed
   * while it was away (a scene switch, the language) is applied now. The Yjs document needs no
   * help, the host resends all of it.
   */
  #onHostConfig(data: AppConfiguration["clientInitialState"]) {
    const first = this.onConfigReceived;
    if (first) {
      this.onConfigReceived = undefined;
      first(data);
      return;
    }
    if (!data || typeof data !== "object") return;
    // A later reader (Service_Scenes.init, while the first start-up is still running) sees it too.
    getConfig().clientInitialState = data;
    // Not through `scenes:change`, which ignores scenes the mirror's document does not hold yet:
    // the host sends this before the document, and the elements draw once the scene arrives.
    if (typeof data.scene === "string" && data.scene)
      getApiClient().scenes.state.activeScene = data.scene;
    if (typeof data.uiLanguage === "string" && data.uiLanguage && data.uiLanguage !== i18n.language)
      void i18n.changeLanguage(data.uiLanguage);
  }

  async init() {
  }

  #initializePeer() {
    this.#provider?.dispose();
    this.#provider = new PeerProvider(getApiClient().document.file);
  }
}

export default Service_Peer;
