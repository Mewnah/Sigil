import { BaseEvent, IServiceInterface } from "@/types";
import i18n from "i18next";
import { toast } from "react-toastify";
import { proxy } from "valtio";
import {
  PeerjsProvider as PeerProvider,
  PEER_CLIENT_MIRROR_EVENT,
  type PeerClientMirrorDetail,
} from "./provider";
import AppConfiguration from "@/config";
import { getApiClient, getApiShared, getConfig } from "@/runtime/host";

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

  /** Full URL for browser / OBS (Curses-style: `http://localhost:PORT/client`, no query required). */
  getClientLink(pathname = "/client"): string {
    const n = getConfig().serverNetwork;
    const p = String(n.port);
    return `http://${n.host}:${p}${pathname}`;
  }

  copyClientLink(pathname = "/client") {
    navigator.clipboard.writeText(getApiShared().peer.getClientLink(pathname));
    toast.success(i18n.t("toasts.copied"));
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
      if (d.phase === "synced") {
        this.clientMirrorState.phase = "synced";
        this.clientMirrorState.lastError = "";
      } else if (d.phase === "reconnecting") {
        this.clientMirrorState.phase = "reconnecting";
      } else if (d.phase === "failed") {
        this.clientMirrorState.phase = "failed";
        this.clientMirrorState.lastError = d.message ?? "";
      } else if (d.phase === "error") {
        this.clientMirrorState.lastError = d.message ?? "";
        this.clientMirrorState.phase = "reconnecting";
      }
    };
    this.#provider?.addEventListener(PEER_CLIENT_MIRROR_EVENT, onMirror);
    this.#provider?.addEventListener("on_event_received", (e) => {
      if (e instanceof CustomEvent) try {
        if (e.detail.topic === "peers:init_data")
          this.onConfigReceived?.(e.detail.data);
        else
          getApiShared().pubsub.publishLocally(e.detail);
      } catch (error) {
        console.error(error)
      }
    });
    try {
      await this.#provider?.connectClient({
        id: "server",
        host: getConfig().clientNetwork.host,
        port: getConfig().clientNetwork.port,
      });
    } catch (err) {
      this.clientMirrorState.phase = "failed";
      this.clientMirrorState.lastError = err instanceof Error ? err.message : String(err);
      throw err;
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
