import {
  ServiceNetworkState,
  TextEvent,
  TextEventType,
} from "@/types";
import OBSWebSocket, { EventSubscription, OBSWebSocketError } from "obs-websocket-js";
import { toast } from "react-toastify";
import { proxy } from "valtio";
import { serviceSubscibeToInput, serviceSubscibeToSource } from "../../../utils";
import { OBS_State } from "./schema";
import { devLog } from "@/utils/devLog";

export class ObsControlService {
  private wsInstance!: OBSWebSocket;
  private wsRequestCancelToken = false;
  private reconnectAttempts = 0;
  private reconnectTimeoutHandle: number = -1;
  private eventDisposers: (() => void)[] = [];

  wsState = proxy({
    status: ServiceNetworkState.disconnected,
  });

  constructor(private readonly getState: () => OBS_State) {}

  init() {
    this.wsInstance = new OBSWebSocket();
    this.wsInstance.on("Identified", () => this.wshandleConnected());
    this.wsInstance.on("CurrentProgramSceneChanged", (e) => this.trySwitchScene(e.sceneName));

    const state = this.getState();
    this.eventDisposers.push(
      serviceSubscibeToSource(state, "source", (e) => this.processTextEvent(e)),
    );
    this.eventDisposers.push(
      serviceSubscibeToInput(state, "inputField", (e) => this.processTextEvent(e)),
    );

    if (state.wsAutoStart) {
      void this.wsConnect();
    }
  }

  private trySwitchScene(obsSceneName?: string) {
    const state = this.getState();
    if (!state.scenesEnable) return;

    const trySwitch = (sceneName: string) => {
      const scene = Object.values(window.ApiClient.scenes.scenes).find((f) => f.name === sceneName);
      if (scene) window.ApiShared.pubsub.publish("scenes:change", scene.id);
    };

    if (obsSceneName && obsSceneName in state.scenesMap) {
      trySwitch(state.scenesMap[obsSceneName]);
    } else if (state.scenesFallback) {
      trySwitch(state.scenesFallback);
    }
  }

  private async wshandleConnected() {
    this.reconnectAttempts = 0;
    const state = this.getState();
    if (!state.scenesEnable) return;
    try {
      const currentScene = await this.wsInstance.call("GetCurrentProgramScene");
      this.trySwitchScene(currentScene.currentProgramSceneName);
    } catch (error) {
      console.error(error);
    }
  }

  private processTextEvent(data?: TextEvent) {
    const state = this.getState();
    if (
      !state.captionsEnable ||
      this.wsState.status !== ServiceNetworkState.connected ||
      !data?.value ||
      !(data.type === TextEventType.final || (data.type === TextEventType.interim && state.interim))
    ) {
      return;
    }
    this.wsInstance.call("SendStreamCaption", { captionText: data.value }).catch((e: OBSWebSocketError) => {
      if (e.code !== 501) this.toastError(e);
    });
  }

  private toastError(e: OBSWebSocketError) {
    const err = e.message ? "[OBS] " + e.message : "[OBS] Connection error";
    toast(err, { type: "error", autoClose: 2000 });
  }

  private wsHandleDisconnect(e: OBSWebSocketError) {
    const state = this.getState();
    if (e.code === 1006 || e.code === 1001) {
      if (this.wsRequestCancelToken) {
        this.wsRequestCancelToken = false;
        this.wsState.status = ServiceNetworkState.disconnected;
      } else if (state.wsAutoStart) {
        this.triggerReconnect();
      } else {
        this.wsState.status = ServiceNetworkState.disconnected;
        this.toastError(e);
      }
      return;
    }

    this.wsState.status = ServiceNetworkState.disconnected;
    if (e.code !== 1000) this.toastError(e);
  }

  private triggerReconnect() {
    const delay = Math.min(1000 * (2 ** this.reconnectAttempts), 30000);
    devLog(`[OBS] Reconnecting in ${delay}ms (Attempt ${this.reconnectAttempts + 1})`);

    this.reconnectTimeoutHandle = setTimeout(() => {
      this.reconnectAttempts++;
      void this.wsConnect();
    }, delay) as unknown as number;
  }

  async wsConnect() {
    const state = this.getState();
    if (state.wsPort === "") {
      toast("[OBS] Invalid connection port", { type: "error", autoClose: false });
      return;
    }
    this.wsState.status = ServiceNetworkState.connecting;

    try {
      this.wsInstance.disconnect();
      this.wsInstance.removeAllListeners("ConnectionClosed");
      await this.wsInstance.connect(`ws://127.0.0.1:${state.wsPort}`, state.wsPassword, {
        eventSubscriptions: EventSubscription.All | EventSubscription.Scenes,
      });
      this.wsState.status = ServiceNetworkState.connected;
      this.wsInstance.addListener("ConnectionClosed", (e) => this.wsHandleDisconnect(e));
    } catch (e: unknown) {
      if (e instanceof OBSWebSocketError) this.wsHandleDisconnect(e);
    }
  }

  wsDisconnect() {
    clearTimeout(this.reconnectTimeoutHandle);
    this.reconnectAttempts = 0;
    this.wsInstance.disconnect();
  }

  wsCancel() {
    if (this.wsState.status !== ServiceNetworkState.connecting) return;
    clearTimeout(this.reconnectTimeoutHandle);
    this.reconnectAttempts = 0;
    this.wsInstance.disconnect();
    this.wsRequestCancelToken = true;
  }

  dispose() {
    this.wsDisconnect();
    this.eventDisposers.forEach((d) => d());
    this.eventDisposers = [];
  }
}

