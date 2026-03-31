import { STT_State } from "../schema";
import { ISTTReceiver, ISTTService } from "../types";

/**
 * Chrome / Edge: Web Speech runs in mic.html and sends results over the local pubsub WebSocket.
 * The host does not capture audio; it only marks STT as active and routes via processExternalMessage.
 */
export class STT_ExternalMicService implements ISTTService {
  constructor(private readonly bindings: ISTTReceiver) {}

  start(_params: STT_State): void {
    this.bindings.onStart();
  }

  stop(): void {
    this.bindings.onStop();
  }

  dispose(): void {}
}
