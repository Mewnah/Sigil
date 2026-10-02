import { Doc, encodeStateAsUpdate }                                              from "yjs";
import Peer, {DataConnection}                                                               from "peerjs";
import {decoding, encoding}                                                                 from "lib0";
import {readSyncMessage, writeUpdate} from "y-protocols/sync";
import {nanoid}      from "nanoid";
import { BaseEvent } from "@/types";
import { devLog } from "@/utils/devLog";

/** PeerJS uses `ws://{host}:{port}/peer/…`. Normalize loopback names to IPv4 so signaling hits the Warp listener. */
function peerSignalingHost(host: string): string {
  const h = host.trim().toLowerCase();
  if (h === "localhost" || h === "127.0.0.1" || h === "::1") return "127.0.0.1";
  return host;
}

/** Dispatched on the browser `/client` PeerJS path for connection diagnostics (OBS, etc.). */
export const PEER_CLIENT_MIRROR_EVENT = "peer_client_mirror";

/**
 * PeerJS error type when the server refuses the API key, which carries the LAN token (P6-D): the
 * mirror runs on another machine and its link has no token, an old one, or the host accepts only
 * its own PC.
 */
export const PEER_INVALID_KEY = "invalid-key";

/** PeerJS server message type: a peer's signaling socket is gone (its connections must close). */
export const PEER_SERVER_LEAVE = "LEAVE";

/** Error type of a PeerJS error (`peer.on("error")`), or "". */
export function peerErrorType(err: unknown): string {
  return err && typeof err === "object" && "type" in err ? String((err as { type?: unknown }).type ?? "") : "";
}

export type PeerClientMirrorDetail =
  | { phase: "synced" }
  | { phase: "reconnecting" }
  | { phase: "error"; message?: string }
  | { phase: "failed"; message?: string };

/** Where a `/client` mirror connects (`token`: the host's LAN token from the page URL, P6-D). */
export interface ClientLinkParams {
  id: string;
  host: string;
  port: string;
  token?: string;
}

export class PeerjsProvider extends EventTarget {
  constructor(
    private document: Doc,
  ) {super();}

  #peer?: Peer;
  private peers: { [id: string]: DataConnection } = {};
  #clientLinkParams?: ClientLinkParams;
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
    // The host signals over loopback, which never needs the LAN token.
    this.#peer = new Peer(params.id, {
      host: peerSignalingHost(params.host),
      port:   parseInt(params.port),
      key:    '',
      path:   'peer',
      secure: false,
      debug: 0});
    this.#peer.on("open", () => {});
    // The signaling server sends `LEAVE` (src = the mirror's id) when a mirror's signaling socket
    // is gone, in particular when a LAN rules change revoked it (src-tauri/src/services/web/peer.rs).
    // PeerJS closes its connections with that peer itself; drop them here too, so a revoked mirror
    // never keeps the data channel that carries the document and every pubsub broadcast, even if
    // PeerJS's handling changes. The same socket delivered that peer's OFFER first.
    this.#peer.socket?.on?.("message", (message: unknown) => {
      const m = message as { type?: unknown; src?: unknown } | null;
      if (m && m.type === PEER_SERVER_LEAVE && typeof m.src === "string") this.dropPeer(m.src);
    });
    this.#peer.on("connection", clientConn => {
      devLog("connected client", clientConn);
      this.peers[clientConn.connectionId] = clientConn;
      clientConn.on("open", () => {
        // Dropped (its peer left signaling) before the channel opened: never send it the document.
        if (this.peers[clientConn.connectionId] !== clientConn) {
          clientConn.close();
          return;
        }
        this.dispatchEvent(new CustomEvent("on_client_connected", {detail: clientConn.connectionId}))
        clientConn.send(this.serializeUpdate(encodeStateAsUpdate(this.document)));
      });
      clientConn.on("close", () => delete this.peers[clientConn.connectionId]);
    });
    this.#peer.on("disconnected", () => {});
  }

  /**
   * Host: close and forget every data connection with `peerId` (the PeerJS id the mirror signaled
   * with). Returns how many were dropped.
   */
  dropPeer(peerId: string): number {
    let dropped = 0;
    for (const [key, conn] of Object.entries(this.peers)) {
      if (conn.peer !== peerId) continue;
      delete this.peers[key];
      dropped++;
      try {
        conn.close();
      } catch {
        /* already closed */
      }
    }
    if (dropped) devLog("dropped data connections of a peer that left signaling", dropped);
    return dropped;
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

  async connectClient(params: ClientLinkParams) {
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
          // The LAN token (P6-D) rides in the PeerJS API key; "" on the host's own PC.
          key: params.token ?? "",
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
        const type = peerErrorType(err);
        // PeerJS's invalid-key message quotes the key, i.e. the LAN token: log the type only.
        if (type === PEER_INVALID_KEY) console.error("[PeerJS] the host refused this mirror's LAN token (invalid-key)");
        else console.error("[PeerJS]", err);
        const message = type || (err instanceof Error ? err.message : String(err ?? ""));
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

  /** Binary frame for pubsub; `dataJson` must be `JSON.stringify(data)` (avoids double-serializing `data`). */
  serializePubSubParts(topic: string, dataJson: string): Uint8Array | undefined {
    try {
      const encoder = encoding.createEncoder();
      encoding.writeVarInt(encoder, 1);
      encoding.writeVarString(encoder, topic);
      encoding.writeVarString(encoder, dataJson);
      return encoding.toUint8Array(encoder);
    } catch (error) {
      console.error(error);
    }
  }

  serializeEvent(msg: BaseEvent) {
    return this.serializePubSubParts(msg.topic, JSON.stringify(msg.data));
  }

  broadcastPubSubSingle(clientId: string, msg: BaseEvent) {
    if (!(clientId in this.peers))
      return;
      
    const serialized = this.serializeEvent(msg);
    this.peers[clientId].send(serialized);
  }

  broadcastPubSub(msg: BaseEvent) {
    this.broadcastPubSubParts(msg.topic, JSON.stringify(msg.data));
  }

  broadcastPubSubParts(topic: string, dataJson: string) {
    const serialized = this.serializePubSubParts(topic, dataJson);
    serialized && this.broadcastUpdate(serialized);
  }

  private broadcastUpdate(uint8Array: Uint8Array) {
    for (let peersKey in this.peers) {
      this.peers[peersKey].send(uint8Array);
    }
  }
}


