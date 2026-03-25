import { BaseEvent, IServiceInterface } from "@/types";
import { toast } from "react-toastify";
import { PeerjsProvider as PeerProvider } from "./provider";
import AppConfiguration from "@/config";
import { getApiClient, getApiShared, getConfig } from "@/runtime/host";

class Service_Peer implements IServiceInterface {
  #provider?: PeerProvider;

  broadcast(msg: BaseEvent) {
    this.#provider?.broadcastPubSub(msg);
  }

  getClientLink(): string {
    const n = getConfig().serverNetwork;
    return `http://${n.host}:${n.port}/client`;
  }

  copyClientLink() {
    navigator.clipboard.writeText(getApiShared().peer.getClientLink());
    toast.success("Copied!");
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
      host: getConfig().serverNetwork.host,
      port: getConfig().serverNetwork.port,
    });
  }
  private onConfigReceived?: (data: any) => any;

  async startClient() {
    this.#initializePeer();
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
    await this.#provider?.connectClient({
      id: "server",
      host: getConfig().clientNetwork.host,
      port: getConfig().clientNetwork.port,
    });
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
