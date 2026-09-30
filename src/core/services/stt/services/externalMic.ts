import i18n from "i18next";
import { PUBSUB_STT_EXTERNAL_COMMAND } from "@/shared/services/pubsub";
import { STT_Backends, STT_State } from "../schema";
import { ISTTReceiver, ISTTService } from "../types";

/**
 * How long the "waiting for the microphone page" line stays up at most: `Service_STT`'s external-mic connect timeout
 * (`EXTERNAL_MIC_CONNECT_TIMEOUT_MS`, the same 6 s), after which a toast takes over. `Service_STT` clears the line as
 * soon as the page reports that Web Speech started or stopped; the expiry covers a page that never answers.
 */
export const EXTERNAL_MIC_WAITING_LINE_MS = 6_000;

/**
 * Chrome / Edge: Web Speech runs in mic.html and sends results over the local pubsub WebSocket.
 * Lifecycle (connected/disconnected) is driven by mic.html via `stt.external.state`; host sends start/stop commands.
 * While the page has not answered yet, the status line says what Sigil waits for.
 */
export class STT_ExternalMicService implements ISTTService {
  constructor(private readonly bindings: ISTTReceiver) {}

  start(params: STT_State): void {
    const browser = params.backend === STT_Backends.edge ? "Edge" : "Chrome";
    this.bindings.onStatus(i18n.t("stt.status_external_waiting", { browser }), { expireMs: EXTERNAL_MIC_WAITING_LINE_MS });
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
    this.bindings.onStatus("");
    this.bindings.onStop();
  }

  dispose(): void {}
}
