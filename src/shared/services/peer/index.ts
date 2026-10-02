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
import AppConfiguration from "@/config";
import { getApiClient, getApiShared, getConfig } from "@/runtime/host";
import { sanitizeClientPubSubFrame } from "@/shared/services/pubsub/inbound";

/** Shown on a `/client` mirror the host refused because of the LAN token (P6-D). */
function invalidKeyMessage() {
  return i18n.t("client.peer_invalid_key");
}

class Service_Peer implements IServiceInterface {
  #provider?: PeerProvider;

  /** Browser `/client` only: connection phase for OBS diagnostics. */
  clientMirrorState = proxy({
    phase: "idle" as "idle" | "connecting" | "synced" | "reconnecting" | "failed",
    lastError: "",
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
  private onConfigReceived?: (data: any) => any;

  async startClient() {
    this.clientMirrorState.phase = "connecting";
    this.clientMirrorState.lastError = "";
    this.#initializePeer();
    const onMirror: EventListener = (e) => {
      if (!(e instanceof CustomEvent)) return;
      const d = e.detail as PeerClientMirrorDetail;
      const detail = (message?: string) => (message === PEER_INVALID_KEY ? invalidKeyMessage() : message ?? "");
      if (d.phase === "synced") {
        this.clientMirrorState.phase = "synced";
        this.clientMirrorState.lastError = "";
      } else if (d.phase === "reconnecting") {
        this.clientMirrorState.phase = "reconnecting";
      } else if (d.phase === "failed") {
        this.clientMirrorState.phase = "failed";
        this.clientMirrorState.lastError = detail(d.message);
      } else if (d.phase === "error") {
        this.clientMirrorState.lastError = detail(d.message);
        this.clientMirrorState.phase = "reconnecting";
      }
    };
    this.#provider?.addEventListener(PEER_CLIENT_MIRROR_EVENT, onMirror);
    this.#provider?.addEventListener("on_event_received", (e) => {
      if (e instanceof CustomEvent) try {
        if (e.detail.topic === "peers:init_data")
          this.onConfigReceived?.(e.detail.data);
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
      await this.#provider?.connectClient({
        id: "server",
        host: network.host,
        port: network.port,
        token: network.token,
      });
    } catch (err) {
      this.clientMirrorState.phase = "failed";
      // PeerJS's own invalid-key message quotes the key (the token): replace it.
      const refused = peerErrorType(err) === PEER_INVALID_KEY;
      this.clientMirrorState.lastError = refused ? invalidKeyMessage() : err instanceof Error ? err.message : String(err);
      throw refused ? new Error(invalidKeyMessage()) : err;
    }
    // wait for runtime config
    return new Promise<AppConfiguration["clientInitialState"]>((res) => this.onConfigReceived = res);
  }

  async init() {
  }

  #initializePeer() {
    this.#provider?.dispose();
    this.#provider = new PeerProvider(getApiClient().document.file);
  }
}

export default Service_Peer;
