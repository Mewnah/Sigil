import { PUBSUB_STT_EXTERNAL_COMMAND } from "@/shared/services/pubsub";
import { STT_State } from "../schema";
import { ISTTReceiver, ISTTService } from "../types";

/**
 * Chrome / Edge: Web Speech runs in mic.html and sends results over the local pubsub WebSocket.
 * Lifecycle (connected/disconnected) is driven by mic.html via `stt.external.state`; host sends start/stop commands.
 */
export class STT_ExternalMicService implements ISTTService {
  constructor(private readonly bindings: ISTTReceiver) {}

  start(_params: STT_State): void {
    window.ApiShared.pubsub.broadcastLocalWebSocketOnly({
      topic: PUBSUB_STT_EXTERNAL_COMMAND,
      data: { cmd: "start" },
    });
  }

  stop(): void {
    window.ApiShared.pubsub.broadcastLocalWebSocketOnly({
      topic: PUBSUB_STT_EXTERNAL_COMMAND,
      data: { cmd: "stop" },
    });
    this.bindings.onStop();
  }

  dispose(): void {}
}
