import { Doc, encodeStateAsUpdate }                                              from "yjs";
import Peer, {DataConnection}                                                               from "peerjs";
import {decoding, encoding}                                                                 from "lib0";
import {readSyncMessage, writeUpdate} from "y-protocols/sync";
import {customAlphabet}  from "nanoid";
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

/** PeerJS error type when the browser has no WebRTC: no retry can fix it. */
const PEER_BROWSER_INCOMPATIBLE = "browser-incompatible";

/** PeerJS server message type: a peer's signaling socket is gone (its connections must close). */
export const PEER_SERVER_LEAVE = "LEAVE";

/** Error type of a PeerJS error (`peer.on("error")`), or "". */
export function peerErrorType(err: unknown): string {
  return err && typeof err === "object" && "type" in err ? String((err as { type?: unknown }).type ?? "") : "";
}

/**
 * The id a `/client` mirror signals with. PeerJS's `validateId` accepts only alphanumeric words
 * joined by single `-`, `_` or spaces and answers anything else with `invalid-id`; nanoid's default
 * alphabet contains `_` and `-`, so about 1 in 9 random ids began, ended or doubled one of them and
 * the page failed to start. Alphanumeric only: always valid, still ~190 bits.
 */
export const newMirrorPeerId = customAlphabet("0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz", 32);

/** Delay before reconnect attempt number `attempt` (0-based): 1, 2, 4, 8, 16 s, then `ceilingMs` for good. */
export function peerReconnectDelayMs(attempt: number, ceilingMs = 30_000): number {
  return Math.min(ceilingMs, 1000 * 2 ** Math.min(Math.max(0, attempt), 5));
}

/**
 * The host retries its registration at least this often: attempts are loopback connects and every
 * mirror depends on this one, so it comes back quickly once the web server listens again. The
 * signaling server holds the departure of the host's old socket back for longer than the first
 * several of these attempts (`HOST_LEAVE_GRACE` in src-tauri/src/services/web/peer.rs).
 */
export const SERVER_RETRY_CEILING_MS = 10_000;

/** A mirror attempt that has neither opened its data channel nor failed after this long is abandoned. */
const CLIENT_OPEN_TIMEOUT_MS = 20_000;

/**
 * An open channel whose ICE state stays "disconnected" this long is dropped. ICE says "disconnected"
 * when the host stopped answering (its process or network is gone, the PC woke from sleep) and only
 * "failed", which closes the channel, after about 30 s. The signaling socket used to give that
 * away at once; a mirror now ignores signaling errors while its channel is open, for at most
 * {@link CLIENT_SIGNALING_DOWN_MS} (below).
 */
const CLIENT_ICE_DISCONNECTED_MS = 10_000;

/**
 * A mirror whose signaling socket is down while its channel is open tries to register again (that
 * is what makes a Wi-Fi roam cost nothing), and drops the channel when the socket is still down
 * after this long: a host that was killed or restarted takes the server with it, and waiting for
 * ICE alone left the source frozen for 15-20 s after Sigil was back.
 */
const CLIENT_SIGNALING_DOWN_MS = 15_000;

/** Re-registration attempts of a mirror whose signaling socket is down: 1, 2, 4 s, then every 5 s. */
const CLIENT_SIGNALING_RETRY_CEILING_MS = 5_000;

/** The host's registration that has not opened after this long is dropped and retried. */
const SERVER_OPEN_TIMEOUT_MS = 20_000;

/** A running mirror the host started refusing (token rotated, LAN switched off) is retried this slowly. */
const CLIENT_REFUSED_RETRY_MS = 60_000;

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

/** The host's own registration with the signaling server. */
interface ServerParams {
  id: string;
  host: string;
  port: string;
}

/** One try of a mirror to reach the host: its Peer, the data channel to the host, the open timeout. */
interface ClientAttempt {
  peer: Peer;
  conn?: DataConnection;
  watchdog?: ReturnType<typeof setTimeout>;
  /** Runs while the channel's ICE state is "disconnected". */
  iceTimer?: ReturnType<typeof setTimeout>;
  /** Runs while the signaling socket is down although the channel is open. */
  signalingDeadline?: ReturnType<typeof setTimeout>;
  /** The next `reconnect()` of the signaling socket. */
  signalingRetry?: ReturnType<typeof setTimeout>;
  signalingAttempts: number;
}

export class PeerjsProvider extends EventTarget {
  constructor(
    private document: Doc,
  ) {super();}

  // -- Host: the registration of `server`, which every mirror signals -----------------------------
  #serverParams?: ServerParams;
  #serverPeer?: Peer;
  #serverTimer: ReturnType<typeof setTimeout> | null = null;
  #serverWatchdog: ReturnType<typeof setTimeout> | null = null;
  #serverAttempts = 0;

  // -- Mirror: one attempt at a time, retried for as long as the page lives -----------------------
  private peers: { [id: string]: DataConnection } = {};
  #clientLinkParams?: ClientLinkParams;
  #attempt?: ClientAttempt;
  /** Settles `connectClient`: on the first open channel of any attempt, or an error no retry fixes. */
  #initial: { resolve: () => void; reject: (err: unknown) => void } | null = null;
  #reconnectTimer: ReturnType<typeof setTimeout> | null = null;
  #reconnectAttempts = 0;

  /** Stable reference so `dispose` can unregister the same listener. */
  #onDocumentUpdate = (update: Uint8Array) => {
    this.broadcastUpdate(this.serializeUpdate(update));
  };

  connectServer(params: ServerParams) {
    this.#serverParams = params;
    this.document.on("update", this.#onDocumentUpdate);
    this.#startServerPeer();
  }

  /** (Re)create the host's Peer with its fixed id. Every handler ignores a Peer that was replaced. */
  #startServerPeer() {
    const params = this.#serverParams;
    if (!params) return;
    this.#clearServerTimer();
    this.#destroyServerPeer();
    let peer: Peer;
    try {
      // The host signals over loopback, which never needs the LAN token.
      peer = new Peer(params.id, {
        host: peerSignalingHost(params.host),
        port:   parseInt(params.port),
        key:    '',
        path:   'peer',
        secure: false,
        debug: 0});
    } catch (error) {
      console.error("[PeerJS] could not start the host registration", error);
      this.#scheduleServerRecovery();
      return;
    }
    this.#serverPeer = peer;
    const current = () => this.#serverPeer === peer;
    this.#armServerWatchdog(peer);
    peer.on("open", () => {
      if (!current()) return;
      this.#serverAttempts = 0;
      this.#clearServerWatchdog();
    });
    // The signaling server sends `LEAVE` (src = the mirror's id) when a mirror's signaling socket
    // is gone, in particular when a LAN rules change revoked it (src-tauri/src/services/web/peer.rs).
    // PeerJS closes its connections with that peer itself; drop them here too, so a revoked mirror
    // never keeps the data channel that carries the document and every pubsub broadcast, even if
    // PeerJS's handling changes. The same socket delivered that peer's OFFER first.
    peer.socket?.on?.("message", (message: unknown) => {
      const m = message as { type?: unknown; src?: unknown } | null;
      if (m && m.type === PEER_SERVER_LEAVE && typeof m.src === "string") this.dropPeer(m.src);
    });
    peer.on("connection", clientConn => {
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
    // Without a registration no mirror can find the host, and none was ever re-registered before:
    // one dropped signaling socket (or ID-TAKEN while the old socket was still closing, or a web
    // server that was not listening yet) locked every OBS source out until Sigil restarted.
    // "disconnected" keeps the Peer and its data channels (reconnect() re-registers the same id);
    // "close" means PeerJS destroyed it (a first registration that failed): make a new one.
    peer.on("disconnected", () => {
      if (current()) this.#scheduleServerRecovery();
    });
    peer.on("close", () => {
      if (current()) this.#scheduleServerRecovery();
    });
    peer.on("error", (err: unknown) => {
      if (!current()) return;
      // Always followed by "disconnected" or "close" (PeerJS aborts on these), which retry.
      console.warn("[PeerJS] host registration:", peerErrorType(err) || err);
    });
  }

  #scheduleServerRecovery() {
    if (!this.#serverParams || this.#serverTimer) return;
    this.#clearServerWatchdog();
    const delay = peerReconnectDelayMs(this.#serverAttempts++, SERVER_RETRY_CEILING_MS);
    this.#serverTimer = setTimeout(() => {
      this.#serverTimer = null;
      this.#recoverServerPeer();
    }, delay);
  }

  #recoverServerPeer() {
    const peer = this.#serverPeer;
    if (peer && !peer.destroyed && peer.disconnected) {
      try {
        peer.reconnect();
        this.#armServerWatchdog(peer);
        return;
      } catch (error) {
        console.warn("[PeerJS] host re-registration failed; starting over", error);
      }
    }
    this.#startServerPeer();
  }

  /** A registration that neither opens nor fails is forced to fail, which queues the next retry. */
  #armServerWatchdog(peer: Peer) {
    this.#clearServerWatchdog();
    this.#serverWatchdog = setTimeout(() => {
      this.#serverWatchdog = null;
      if (this.#serverPeer !== peer || peer.open || peer.destroyed) return;
      console.warn("[PeerJS] the host registration did not open in time; retrying");
      try {
        peer.disconnect();
      } catch {
        /* already gone */
      }
      this.#scheduleServerRecovery();
    }, SERVER_OPEN_TIMEOUT_MS);
  }

  #clearServerWatchdog() {
    if (this.#serverWatchdog) {
      clearTimeout(this.#serverWatchdog);
      this.#serverWatchdog = null;
    }
  }

  #clearServerTimer() {
    if (this.#serverTimer) {
      clearTimeout(this.#serverTimer);
      this.#serverTimer = null;
    }
  }

  #destroyServerPeer() {
    const peer = this.#serverPeer;
    this.#serverPeer = undefined;
    this.#clearServerWatchdog();
    try {
      peer?.destroy();
    } catch {
      /* ignore */
    }
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

  /** Close every data connection and forget the current attempt; its late events are ignored. */
  #closeAllConnections() {
    for (const key of Object.keys(this.peers)) {
      try {
        this.peers[key]?.close();
      } catch {
        /* ignore */
      }
      delete this.peers[key];
    }
  }

  #destroyClientAttempt() {
    const attempt = this.#attempt;
    this.#attempt = undefined;
    if (attempt?.watchdog) clearTimeout(attempt.watchdog);
    if (attempt?.iceTimer) clearTimeout(attempt.iceTimer);
    if (attempt?.signalingDeadline) clearTimeout(attempt.signalingDeadline);
    if (attempt?.signalingRetry) clearTimeout(attempt.signalingRetry);
    this.#closeAllConnections();
    try {
      attempt?.peer.destroy();
    } catch {
      /* ignore */
    }
  }

  /**
   * Retry later, for as long as the page lives (1, 2, 4, 8, 16 s, then every 30 s): the host may
   * be restarting, asleep or not started yet, and a source that gave up stayed dead until someone
   * refreshed it in OBS.
   */
  #scheduleClientReconnect(delayMs?: number) {
    if (!this.#clientLinkParams) return;
    if (this.#reconnectTimer) return;
    const delay = delayMs ?? peerReconnectDelayMs(this.#reconnectAttempts);
    this.#reconnectAttempts++;
    this.#reconnectTimer = setTimeout(() => {
      this.#reconnectTimer = null;
      // A timer belongs to the attempt that failed. A healthy channel is never torn down for it.
      if (this.#attempt?.conn?.open) return;
      this.#runClientConnection();
    }, delay);
  }

  /**
   * Connect a mirror to the host. Resolves when the first attempt that succeeds has its data channel
   * open; attempts that fail are retried here (same back-off as a later reconnect) so the page keeps
   * its loading view. Rejects only for what no retry can fix: the host refusing the LAN token, a
   * browser without WebRTC, a Peer that cannot even be constructed.
   */
  connectClient(params: ClientLinkParams): Promise<void> {
    this.#clientLinkParams = params;
    this.#clearClientReconnectTimer();
    this.#reconnectAttempts = 0;
    return new Promise<void>((resolve, reject) => {
      this.#initial = { resolve, reject };
      this.#runClientConnection();
    });
  }

  #dispatchMirror(detail: PeerClientMirrorDetail) {
    this.dispatchEvent(new CustomEvent<PeerClientMirrorDetail>(PEER_CLIENT_MIRROR_EVENT, { detail }));
  }

  #runClientConnection() {
    const params = this.#clientLinkParams;
    if (!params) return;

    this.#clearClientReconnectTimer();
    this.#destroyClientAttempt();

    let peer: Peer;
    try {
      peer = new Peer(newMirrorPeerId(), {
        host: peerSignalingHost(params.host),
        port: parseInt(params.port, 10),
        // The LAN token (P6-D) rides in the PeerJS API key; "" on the host's own PC.
        key: params.token ?? "",
        path: "peer",
        secure: false,
        debug: 0,
      });
    } catch (error) {
      const initial = this.#initial;
      if (initial) {
        // At start-up this is a bad host/port or a blocked WebSocket: retrying cannot change it.
        this.#initial = null;
        this.#clientLinkParams = undefined;
        initial.reject(error);
        return;
      }
      console.error("[PeerJS]", error);
      this.#scheduleClientReconnect();
      return;
    }

    const attempt: ClientAttempt = { peer, signalingAttempts: 0 };
    this.#attempt = attempt;
    const current = () => this.#attempt === attempt;

    // PeerJS closes a connection that never opened without telling anyone ('close' fires only for
    // an open one), and a stalled ICE negotiation can sit for minutes: a deadline per attempt keeps
    // the retry loop alive.
    attempt.watchdog = setTimeout(() => {
      attempt.watchdog = undefined;
      if (!current() || attempt.conn?.open) return;
      this.#clientAttemptFailed(attempt, Object.assign(new Error("connection attempt timed out"), { type: "timeout" }));
    }, CLIENT_OPEN_TIMEOUT_MS);

    peer.on("error", (err: unknown) => {
      if (!current()) return;
      // Signaling is only needed to set a connection up: with the data channel open, a lost or
      // erroring signaling socket (a second-PC Wi-Fi roam, the web server rebinding) costs nothing
      // for a while (see `#signalingLost`). When the channel closes the next attempt makes a new
      // Peer anyway. The one exception is the host refusing this mirror (LAN token rotated, LAN
      // switched off) on a re-registration: it must not keep a channel the host no longer wants.
      if (attempt.conn?.open && peerErrorType(err) !== PEER_INVALID_KEY) {
        console.warn("[PeerJS] signaling error while the data channel is open (ignored):", peerErrorType(err) || err);
        return;
      }
      this.#clientAttemptFailed(attempt, err);
    });

    // The signaling socket went away (PeerJS then disconnects the Peer and never reconnects it by
    // itself). Before the channel opened the error above already failed the attempt.
    peer.on("disconnected", () => {
      if (!current() || !attempt.conn?.open) return;
      this.#signalingLost(attempt);
    });

    peer.on("open", () => {
      if (!current()) return;
      if (attempt.conn) {
        // Registered again after a lost signaling socket: the channel never closed, nothing to redo.
        this.#signalingRestored(attempt);
        return;
      }
      const hostConn = peer.connect(params.id, { serialization: "binary" });
      // PeerJS answers connect() on a disconnected Peer with an error event and no connection.
      if (!hostConn) return;
      attempt.conn = hostConn;
      this.peers[hostConn.connectionId] = hostConn;
      hostConn.on("data", (handleData) =>
        this.readMessage(new Uint8Array(handleData as ArrayBuffer))
      );
      // An ICE failure before the channel opened is reported here only (and PeerJS then closes it
      // without a 'close').
      hostConn.on("error", (err: unknown) => {
        if (!current() || attempt.conn !== hostConn || hostConn.open) return;
        this.#clientAttemptFailed(attempt, err);
      });
      hostConn.on("close", () => {
        if (!current() || attempt.conn !== hostConn) return;
        this.#clientAttemptFailed(attempt);
      });
      hostConn.on("iceStateChanged", (state) => {
        if (!current() || attempt.conn !== hostConn) return;
        if (attempt.iceTimer) clearTimeout(attempt.iceTimer);
        attempt.iceTimer = undefined;
        if (state !== "disconnected") return;
        attempt.iceTimer = setTimeout(() => {
          attempt.iceTimer = undefined;
          if (current() && attempt.conn === hostConn) this.#clientAttemptFailed(attempt);
        }, CLIENT_ICE_DISCONNECTED_MS);
      });
      hostConn.on("open", () => {
        if (!current() || attempt.conn !== hostConn) return;
        if (attempt.watchdog) clearTimeout(attempt.watchdog);
        attempt.watchdog = undefined;
        // Whatever retry is still queued is stale now: it would tear this healthy link down.
        this.#clearClientReconnectTimer();
        this.#reconnectAttempts = 0;
        this.#dispatchMirror({ phase: "synced" });
        const initial = this.#initial;
        this.#initial = null;
        initial?.resolve();
      });
    });
  }

  /**
   * The signaling socket of a mirror with an open channel is down. Keep the channel, try to
   * register again, and give up on the channel when that does not work out in time.
   */
  #signalingLost(attempt: ClientAttempt) {
    if (this.#attempt !== attempt) return;
    if (!attempt.signalingDeadline) {
      attempt.signalingDeadline = setTimeout(() => {
        attempt.signalingDeadline = undefined;
        if (this.#attempt !== attempt) return;
        console.warn("[PeerJS] the signaling connection stayed down; dropping the data channel to start over");
        this.#clientAttemptFailed(attempt);
      }, CLIENT_SIGNALING_DOWN_MS);
    }
    if (attempt.signalingRetry) return;
    const delay = peerReconnectDelayMs(attempt.signalingAttempts++, CLIENT_SIGNALING_RETRY_CEILING_MS);
    attempt.signalingRetry = setTimeout(() => {
      attempt.signalingRetry = undefined;
      if (this.#attempt !== attempt || attempt.peer.destroyed || !attempt.peer.disconnected) return;
      try {
        // A failure comes back as an error event and another "disconnected" (the next retry).
        attempt.peer.reconnect();
      } catch (error) {
        console.warn("[PeerJS] could not register the mirror again:", error);
        this.#signalingLost(attempt);
      }
    }, delay);
  }

  #signalingRestored(attempt: ClientAttempt) {
    if (attempt.signalingDeadline) clearTimeout(attempt.signalingDeadline);
    if (attempt.signalingRetry) clearTimeout(attempt.signalingRetry);
    attempt.signalingDeadline = undefined;
    attempt.signalingRetry = undefined;
    attempt.signalingAttempts = 0;
  }

  /**
   * The current attempt did not work (`err` set) or its open channel closed (no `err`). Tear it
   * down and queue the next one; events of an attempt that was already replaced do nothing.
   */
  #clientAttemptFailed(attempt: ClientAttempt, err?: unknown) {
    if (this.#attempt !== attempt) return;
    const type = peerErrorType(err);
    // PeerJS's invalid-key message quotes the key, i.e. the LAN token: log the type only.
    if (type === PEER_INVALID_KEY) console.error("[PeerJS] the host refused this mirror's LAN token (invalid-key)");
    else if (err !== undefined) console.error("[PeerJS]", err);
    this.#destroyClientAttempt();

    const initial = this.#initial;
    if (initial && (type === PEER_INVALID_KEY || type === PEER_BROWSER_INCOMPATIBLE)) {
      this.#initial = null;
      this.#clientLinkParams = undefined;
      initial.reject(err);
      return;
    }

    const refused = type === PEER_INVALID_KEY;
    if (err === undefined) this.#dispatchMirror({ phase: "reconnecting" });
    else if (refused) this.#dispatchMirror({ phase: "failed", message: type });
    else this.#dispatchMirror({ phase: "error", message: type || (err instanceof Error ? err.message : String(err ?? "")) });
    this.#scheduleClientReconnect(refused ? CLIENT_REFUSED_RETRY_MS : undefined);
  }

  dispose() {
    this.#clearClientReconnectTimer();
    this.#clientLinkParams = undefined;
    this.#initial = null;
    this.#serverParams = undefined;
    this.#clearServerTimer();
    this.document.off("update", this.#onDocumentUpdate);
    this.#destroyClientAttempt();
    this.#destroyServerPeer();
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

