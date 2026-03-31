import { Doc, encodeStateAsUpdate }                                              from "yjs";
import Peer, {DataConnection}                                                               from "peerjs";
import {decoding, encoding}                                                                 from "lib0";
import {readSyncMessage, writeUpdate} from "y-protocols/sync";
import {nanoid}      from "nanoid";
import { BaseEvent } from "@/types";

/** PeerJS uses `ws://{host}:{port}/peer/…`. Normalize loopback names to IPv4 so signaling hits the Warp listener. */
function peerSignalingHost(host: string): string {
  const h = host.trim().toLowerCase();
  if (h === "localhost" || h === "127.0.0.1" || h === "::1") return "127.0.0.1";
  return host;
}

/** Dispatched on the browser `/client` PeerJS path for connection diagnostics (OBS, etc.). */
export const PEER_CLIENT_MIRROR_EVENT = "peer_client_mirror";

export type PeerClientMirrorDetail =
  | { phase: "synced" }
  | { phase: "reconnecting" }
  | { phase: "error"; message?: string }
  | { phase: "failed"; message?: string };

export class PeerjsProvider extends EventTarget {
  constructor(
    private document: Doc,
  ) {super();}

  #peer?: Peer;
  private peers: { [id: string]: DataConnection } = {};
  #clientLinkParams?: { id: string; host: string; port: string };
  #reconnectTimer: ReturnType<typeof setTimeout> | null = null;
  #reconnectAttempts = 0;
  private static readonly maxClientReconnects = 20;

  /** Stable reference so `dispose` can unregister the same listener. */
  #onDocumentUpdate = (update: Uint8Array) => {
    this.broadcastUpdate(this.serializeUpdate(update));
  };

  connectServer(params: {
    id: string,
    host: string,
    port: string
  }) {
    this.document.on("update", this.#onDocumentUpdate);
    this.#peer = new Peer(params.id, {
      host: peerSignalingHost(params.host),
      port:   parseInt(params.port),
      key:    '',
      path:   'peer',
      secure: false,
      debug: 0});
    this.#peer.on("open", () => {});
    this.#peer.on("connection", clientConn => {
      console.log("connected client", clientConn);
      this.peers[clientConn.connectionId] = clientConn;
      clientConn.on("open", () => {
        this.dispatchEvent(new CustomEvent("on_client_connected", {detail: clientConn.connectionId}))
        clientConn.send(this.serializeUpdate(encodeStateAsUpdate(this.document)));
      });
      clientConn.on("close", () => delete this.peers[clientConn.connectionId]);
    });
    this.#peer.on("disconnected", () => {});
  }

  #clearClientReconnectTimer() {
    if (this.#reconnectTimer) {
      clearTimeout(this.#reconnectTimer);
      this.#reconnectTimer = null;
    }
  }

  #destroyClientConnections() {
    for (const key of Object.keys(this.peers)) {
      try {
        this.peers[key]?.close();
      } catch {
        /* ignore */
      }
      delete this.peers[key];
    }
    this.#peer?.destroy();
    this.#peer = undefined;
  }

  #scheduleClientReconnect() {
    if (!this.#clientLinkParams) return;
    if (this.#reconnectTimer) return;
    if (this.#reconnectAttempts >= PeerjsProvider.maxClientReconnects) {
      console.error("[PeerJS] Too many reconnect failures. Reload the page to try again.");
      this.dispatchEvent(
        new CustomEvent<PeerClientMirrorDetail>(PEER_CLIENT_MIRROR_EVENT, {
          detail: {
            phase: "failed",
            message:
              "Too many reconnect failures. Reload this source or re-copy the client URL from Sigil.",
          },
        }),
      );
      return;
    }
    const exp = Math.min(this.#reconnectAttempts, 5);
    const delay = Math.min(30_000, 1000 * Math.pow(2, exp));
    this.#reconnectAttempts++;
    this.#reconnectTimer = setTimeout(() => {
      this.#reconnectTimer = null;
      void this.#runClientConnection(false);
    }, delay);
  }

  async connectClient(params: {
    id: string,
    host: string,
    port: string
  }) {
    this.#clientLinkParams = params;
    this.#clearClientReconnectTimer();
    this.#reconnectAttempts = 0;
    await this.#runClientConnection(true);
  }

  async #runClientConnection(isInitial: boolean): Promise<void> {
    const params = this.#clientLinkParams;
    if (!params) {
      if (isInitial) throw new Error("Missing client link parameters");
      return;
    }

    this.#destroyClientConnections();

    await new Promise<void>((resolve, reject) => {
      let settled = false;
      const finish = (ok: boolean, err?: unknown) => {
        if (settled) return;
        settled = true;
        if (ok) resolve();
        else if (isInitial) reject(err ?? new Error("PeerJS connection failed"));
        else resolve();
      };

      let peer: Peer;
      try {
        peer = new Peer(nanoid(64), {
          host: peerSignalingHost(params.host),
          port: parseInt(params.port, 10),
          key: "",
          path: "peer",
          secure: false,
          debug: 0,
        });
      } catch (e) {
        this.#scheduleClientReconnect();
        finish(false, e);
        return;
      }

      this.#peer = peer;

      peer.on("error", (err: unknown) => {
        console.error("[PeerJS]", err);
        const message =
          err && typeof err === "object" && "type" in err
            ? String((err as { type?: string }).type)
            : err instanceof Error
              ? err.message
              : String(err ?? "");
        this.dispatchEvent(
          new CustomEvent<PeerClientMirrorDetail>(PEER_CLIENT_MIRROR_EVENT, {
            detail: { phase: "error", message },
          }),
        );
        this.#scheduleClientReconnect();
        finish(false, err);
      });

      peer.on("open", () => {
        const hostConn = peer.connect(params.id, { serialization: "binary" });
        this.peers[hostConn.connectionId] = hostConn;
        hostConn.on("data", (handleData) =>
          this.readMessage(new Uint8Array(handleData as ArrayBuffer))
        );
        hostConn.on("close", () => {
          delete this.peers[hostConn.connectionId];
          this.dispatchEvent(
            new CustomEvent<PeerClientMirrorDetail>(PEER_CLIENT_MIRROR_EVENT, {
              detail: { phase: "reconnecting" },
            }),
          );
          this.#scheduleClientReconnect();
        });
        hostConn.on("open", () => {
          this.#reconnectAttempts = 0;
          this.dispatchEvent(
            new CustomEvent<PeerClientMirrorDetail>(PEER_CLIENT_MIRROR_EVENT, {
              detail: { phase: "synced" },
            }),
          );
          finish(true);
        });
      });
    });
  }

  dispose() {
    this.#clearClientReconnectTimer();
    this.#clientLinkParams = undefined;
    this.document.off("update", this.#onDocumentUpdate);
    this.#destroyClientConnections();
  }

  private readMessage(buffer: Uint8Array) {
    const decoder     = decoding.createDecoder(buffer);
    const encoder     = encoding.createEncoder();
    const messageType = decoding.readVarUint(decoder);

    if (messageType === 0) {
      // readSyncMessage already applies sync step updates to the document.
      readSyncMessage(decoder, encoder, this.document, 0);
    }
    else if (messageType === 1) {
      try {
        const topicStr = decoding.readVarString(decoder);
        const dataStr = decoding.readVarString(decoder);
        this.dispatchEvent(new CustomEvent("on_event_received", {
          detail: {topic: topicStr, data: JSON.parse(dataStr)}
        }));
      } catch (error) {console.error(error)}
    }
  }

  serializeUpdate(update: Uint8Array) {
    const encoder = encoding.createEncoder()
    encoding.writeVarUint(encoder, 0)
    writeUpdate(encoder, update);
    return encoding.toUint8Array(encoder);
  }

  serializeEvent(msg: BaseEvent) {
    try {
      const encoder = encoding.createEncoder();
      encoding.writeVarInt(encoder, 1);
      encoding.writeVarString(encoder, msg.topic);
      encoding.writeVarString(encoder, JSON.stringify(msg.data));
      return encoding.toUint8Array(encoder);
    } catch (error) {
      console.error(error);
    }
  }

  broadcastPubSubSingle(clientId: string, msg: BaseEvent) {
    if (!(clientId in this.peers))
      return;
      
    const serialized = this.serializeEvent(msg);
    this.peers[clientId].send(serialized);
  }

  broadcastPubSub(msg: BaseEvent) {
    const serialized = this.serializeEvent(msg);
    serialized && this.broadcastUpdate(serialized);
  }

  private broadcastUpdate(uint8Array: Uint8Array) {
    for (let peersKey in this.peers) {
      this.peers[peersKey].send(uint8Array);
    }
  }
}


