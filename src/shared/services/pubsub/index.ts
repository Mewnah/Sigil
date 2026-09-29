import { BaseEvent, IServiceInterface, PartialWithRequired, ServiceNetworkState, TextEvent, TextEventSource, TextEventType } from "@/types";
import { listen } from '@tauri-apps/api/event';
import { invoke } from "@tauri-apps/api/core";
import { nanoid } from "nanoid";
import PubSub from "pubsub-js";
import i18n from "i18next";
import { toast } from "react-toastify";
import { proxy } from "valtio";
import { proxyMap, subscribeKey } from "valtio/utils";

import { z } from "zod";
import {
  encodeWire,
  LINK_PUBSUB_KEEPALIVE_TOPIC,
  newMessageId,
  nextVia,
  PUBSUB_STT_EXTERNAL_COMMAND,
  PUBSUB_STT_EXTERNAL_STATE,
  RecentMessageIds,
  routeInboundMessage,
  type InboundSource,
} from "./inbound";

/** Application WS ping interval for linked pubsub (see docs/RELEASE_AND_ENGINEERING.md#performance-and-connection-timeout-audit). */
export const LINK_PUBSUB_KEEPALIVE_MS = 25_000;

const LINK_PUBSUB_CONNECT_TIMEOUT_MS = 12_000;
const LINK_PUBSUB_MAX_RECONNECT_DELAY_MS = 30_000;
const LINK_PUBSUB_MAX_ATTEMPTS = 25;

const RegisteredEventSchema = z.object({
  label: z.string(),
  description: z.string().optional(),
  value: z.string(),
});

type RegisteredEvent = z.infer<typeof RegisteredEventSchema>;

export type TextEmoteEnricher = (
  data: PartialWithRequired<TextEvent, "type" | "value">
) => PartialWithRequired<TextEvent, "type" | "value">;

/** Host registers STT forwarding for external `text.stt` pubsub (keeps shared free of ApiServer). */
export type ExternalSttHandler = (event: TextEvent) => void;

/** Chrome/Edge mic.html → host: recognition actually started or stopped. */
export type ExternalMicLifecycleHandler = (active: boolean) => void;

export { PUBSUB_STT_EXTERNAL_COMMAND, PUBSUB_STT_EXTERNAL_STATE };

/** `plugin:web|server_status` / `web:server_status` payload (Rust `WebServerStatus`). */
export type WebServerStatus = {
  state: "starting" | "listening" | "failed" | "stopped";
  port: number;
  /** Bind / server error when `state` is `failed` or `stopped`. */
  error: string | null;
  /** Whether the Origin allow-list is enforced right now. */
  originCheck: boolean;
  /** The `lanOriginCheck` setting as Rust last saw it. */
  originCheckSetting: boolean;
  /** `SIGIL_LAN_ORIGIN_CHECK` override (wins over the setting), or null. */
  originCheckEnv: boolean | null;
};

/** `web:origin_rejected` payload (Rust `OriginRejection`), throttled per origin in Rust. */
export type WebOriginRejection = {
  route: string;
  origin: string | null;
  remote: string | null;
  reason: "foreign_origin" | "null_origin" | "malformed_origin" | "missing_origin_remote";
};

/** Minimum gap between "blocked a connection" toasts (Rust already throttles per origin). */
const ORIGIN_REJECTED_TOAST_GAP_MS = 20_000;

const TEXT_HISTORY_CAP = 120;

class Service_PubSub implements IServiceInterface {
  constructor() { }
  #historyBuf: { id: string; event: string; value: string }[] = [];
  #historyHead = 0;
  #historyCount = 0;

  #appendTextHistoryLine(entry: { id: string; event: string; value: string }) {
    const cap = TEXT_HISTORY_CAP;
    if (this.#historyBuf.length < cap) {
      this.#historyBuf.push(entry);
      this.#historyCount = this.#historyBuf.length;
    } else {
      this.#historyBuf[this.#historyHead] = entry;
      this.#historyHead = (this.#historyHead + 1) % cap;
      this.#historyCount = cap;
    }
    const n = this.#historyBuf.length;
    const start = n < cap ? 0 : this.#historyHead;
    const list: { id: string; event: string; value: string }[] = [];
    for (let i = 0; i < this.#historyCount; i++) {
      list.push(this.#historyBuf[(start + i) % n]!);
    }
    this.textHistory.list = list;
    this.textHistory.lastId = entry.id;
  }

  #socket?: WebSocket;
  #linkIntentionalClose = false;
  #linkFullAddress: string | null = null;
  #linkReconnectAttempts = 0;
  #linkReconnectTimer: ReturnType<typeof setTimeout> | null = null;
  #linkPingTimer: ReturnType<typeof setInterval> | null = null;
  #textEmoteEnricher?: TextEmoteEnricher;
  #externalSttHandler?: ExternalSttHandler;
  #externalMicLifecycleHandler?: ExternalMicLifecycleHandler;
  /** This host session's id in the `via` chain of relayed messages (see ./inbound.ts). */
  readonly #sessionId = nanoid(12);
  /** Ids (`mid`) of messages this host originated, relayed or accepted: repeats are dropped. */
  readonly #seenMessageIds = new RecentMessageIds();
  #hostWebServerAttached = false;
  #lastServerStatusToast = "";
  #lastOriginToastAt = 0;
  /** Local web server status (host app only; null until known). */
  webServer = proxy<{ status: WebServerStatus | null }>({ status: null });
  serviceState = proxy({
    state: ServiceNetworkState.disconnected,
  });
  textHistory = proxy<{
    lastId: string
    list: {
      id: string,
      event: string,
      value: string
    }[]
  }>({
    lastId: "",
    list: []
  })

  /**
   * Inbound message from the local WebSocket server (`source: "local"`) or from this host's Link
   * socket (`source: "link"`). Only allow-listed topics with valid payloads get through
   * (./inbound.ts): peer emotes are dropped (the host rescans), markup in peer text is neutralized,
   * and a message whose `mid` this host has already seen is dropped. Everything else is dropped
   * silently.
   */
  private consumePubSubMessage(stringEvent: unknown, source: InboundSource = "local") {
    if (!window.Config.isServer() || typeof stringEvent !== "string")
      return;
    try {
      const route = routeInboundMessage(stringEvent, source, this.#sessionId, this.#seenMessageIds);
      switch (route.kind) {
        case "ignore":
          return;
        case "mic_state":
          this.#externalMicLifecycleHandler?.(route.active);
          return;
        case "external_stt": {
          const event = this.applyEmotes(route.event);
          this.#externalSttHandler?.({ ...event, emotes: event.emotes ?? {} });
          return;
        }
        case "caption":
          this.#relay(route.topic, route.data, route.via, route.mid, source);
          return;
        case "text":
          this.#relay(route.topic, this.applyEmotes(route.data), route.via, route.mid, source);
          return;
      }
    } catch (error) {
      console.warn("[pubsub] inbound message handler failed", error);
    }
  }

  /**
   * Publish an accepted inbound event locally and pass it on, keeping its `mid` (or giving it one);
   * never back over the Link socket it came from.
   */
  #relay(topic: string, data: unknown, via: string[], mid: string | undefined, source: InboundSource) {
    this.publishLocally({ topic, data });
    this.#fanOutPubSubRemote(topic, data, nextVia(via, this.#sessionId), mid ?? this.#newMessageId(), source === "local");
  }

  /** A fresh `mid`, remembered so a copy that comes back without this host in `via` is dropped. */
  #newMessageId() {
    const mid = newMessageId();
    this.#seenMessageIds.add(mid);
    return mid;
  }

  public registeredEvents = proxyMap<string, RegisteredEvent>([]);

  registerEvent = (event: RegisteredEvent) => this.registeredEvents.set(event.value, event);
  unregisterEvent = (eventValue: string) => this.registeredEvents.delete(eventValue);

  /** Host registers Twitch (or other) emote scanning; shared layer stays free of ApiServer. */
  setTextEmoteEnricher(fn: TextEmoteEnricher | undefined) {
    this.#textEmoteEnricher = fn;
  }

  setExternalSttHandler(fn: ExternalSttHandler | undefined) {
    this.#externalSttHandler = fn;
  }

  setExternalMicLifecycleHandler(fn: ExternalMicLifecycleHandler | undefined) {
    this.#externalMicLifecycleHandler = fn;
  }

  /** Localhost pubsub WebSocket peers only (mic.html); does not fan out to PeerJS or link socket. */
  broadcastLocalWebSocketOnly(msg: BaseEvent) {
    window.Config.isApp() &&
      invoke("plugin:web|pubsub_broadcast", { value: JSON.stringify(msg) });
  }

  /**
   * Host app, after settings and i18n are loaded: keep the Rust Origin-check kill switch in sync
   * with `settings.lanOriginCheck` (a valtio proxy), and surface a failed web-server bind and
   * refused WebSocket connections as toasts. Idempotent.
   */
  async attachHostWebServer(settings: { lanOriginCheck: boolean }) {
    if (!window.Config.isServer() || !window.Config.isApp() || this.#hostWebServerAttached)
      return;
    this.#hostWebServerAttached = true;

    const pushOriginCheck = (enabled: boolean) =>
      invoke<WebServerStatus>("plugin:web|set_origin_check", { enabled })
        .then((status) => this.#onServerStatus(status))
        .catch((e) => console.warn("[pubsub] set_origin_check failed", e));
    subscribeKey(settings, "lanOriginCheck", (enabled) => void pushOriginCheck(enabled !== false));

    try {
      await listen<WebServerStatus>("web:server_status", (e) => this.#onServerStatus(e.payload));
      await listen<WebOriginRejection>("web:origin_rejected", (e) => this.#onOriginRejected(e.payload));
    } catch (e) {
      console.warn("[pubsub] web server event listeners failed", e);
    }
    // The reply carries the current status, so a bind failure from before the webview was
    // listening is surfaced too.
    await pushOriginCheck(settings.lanOriginCheck !== false);
  }

  #onServerStatus(status: WebServerStatus) {
    this.webServer.status = status;
    if (status.state !== "failed" && status.state !== "stopped") {
      this.#lastServerStatusToast = "";
      return;
    }
    const key = `${status.state}:${status.error ?? ""}`;
    if (key === this.#lastServerStatusToast)
      return;
    this.#lastServerStatusToast = key;
    console.error("[pubsub] local web server:", status.state, status.error);
    toast.error(
      i18n.t("toasts.web_server_failed", { port: status.port, error: status.error ?? status.state }),
      { toastId: "web-server-status", autoClose: 20_000 }
    );
  }

  #onOriginRejected(rejection: WebOriginRejection) {
    console.warn("[pubsub] local web server refused a WebSocket connection", rejection);
    const now = Date.now();
    if (now - this.#lastOriginToastAt < ORIGIN_REJECTED_TOAST_GAP_MS)
      return;
    this.#lastOriginToastAt = now;
    const who = rejection.origin ?? rejection.remote ?? "?";
    toast.warn(i18n.t("toasts.web_origin_rejected", { origin: who.slice(0, 120) }), {
      toastId: "web-origin-rejected",
      autoClose: 12_000,
    });
  }

  async init() {
    window.Config.isServer() && listen('pubsub', (event) => {
      this.consumePubSubMessage(event.payload, "local");
    });

    this.registerEvent({ label: "Speech-to-Text", value: TextEventSource.stt });
    this.registerEvent({ label: "Translation", value: TextEventSource.translation });
    this.registerEvent({ label: "Text field", value: TextEventSource.textfield });
    this.registerEvent({ label: "AI: Rewritten Text", value: TextEventSource.transform });
    this.registerEvent({ label: "AI: Original Text (Synced)", value: TextEventSource.transform_source });
    this.registerEvent({ label: "Any text source", value: TextEventSource.any });

    // Track final text on the host (desktop app): STT, TTS pipeline, typed input, etc. (PubSub bubbles text.* → "text".)
    !window.Config.isClient() &&
      this.subscribeText(TextEventSource.any, (event, eventName) => {
      if (event?.type === TextEventType.final) {
        const suffix = eventName?.replace("text.", "") || "text";
        // Transform echoes STT to transform_source/transform_raw for canvas sync; when AI is off, that duplicates the STT line in the transcript.
        const transformStatus = window.ApiServer?.transform?.serviceState?.status;
        if (
          (suffix === "transform_source" || suffix === "transform_raw") &&
          transformStatus !== ServiceNetworkState.connected
        ) {
          return;
        }
        const id = nanoid();
        this.#appendTextHistoryLine({ id, event: suffix, value: event.value });
      }
    });
  }

  private applyEmotes(data: PartialWithRequired<TextEvent, "type" | "value">) {
    if (this.#textEmoteEnricher)
      return this.#textEmoteEnricher(data);
    if (!data.emotes)
      return { ...data, emotes: {} };
    return data;
  }

  publishLocally({ topic, data }: BaseEvent) {
    PubSub.publishSync(topic, data);
  }

  /**
   * PeerJS + localhost WS (+ the Link socket when `toLink`); local PubSub already handled separately
   * when needed. `via` is the loop-prevention chain and `mid` the duplicate-suppression id
   * (./inbound.ts).
   */
  #fanOutPubSubRemote(topic: string, data: unknown, via: string[], mid: string, toLink: boolean) {
    const dataJson = JSON.stringify(data);
    window.ApiShared.peer.broadcastPubSubParts(topic, dataJson);
    const wire = encodeWire(topic, dataJson, via, mid);
    if (window.Config.isApp())
      void invoke("plugin:web|pubsub_broadcast", { value: wire });
    if (toLink && this.#socket && this.#socket.readyState === this.#socket.OPEN)
      this.#socket.send(wire);
  }

  publish(topic: string, data: any) {
    if (window.Config.isClient())
      return;
    const msg = { topic, data };
    this.publishLocally(msg);
    this.#fanOutPubSubRemote(topic, data, [this.#sessionId], this.#newMessageId(), true);
  }
  publishText(topic: TextEventSource, textData: PartialWithRequired<TextEvent, "type" | "value">) {
    let data = this.applyEmotes(textData);
    this.publish(topic, data);
  }

  public unsubscribe(key?: string) {
    key && PubSub.unsubscribe(key);
  }

  public subscribe(eventname: string, fn: (value: unknown) => void) {
    return PubSub.subscribe(eventname, (_, data) => fn(data));
  }

  public subscribeText(source: TextEventSource, fn: (value?: TextEvent, eventName?: string) => void, allowEmpty = false) {
    return PubSub.subscribe(source, (eventName, data: TextEvent) => {
      if (allowEmpty)
        fn(data, eventName);
      else if (data.value)
        fn(data, eventName);
    })
  }

  linkState = proxy({
    value: ServiceNetworkState.disconnected
  });

  copyLinkAddress() {
    const conf = window.Config.serverNetwork;
    navigator.clipboard.writeText(`${conf.ip}:${conf.port}`)
    toast.success(i18n.t("toasts.copied"));
  }

  #clearLinkSchedulers() {
    if (this.#linkReconnectTimer) {
      clearTimeout(this.#linkReconnectTimer);
      this.#linkReconnectTimer = null;
    }
    if (this.#linkPingTimer) {
      clearInterval(this.#linkPingTimer);
      this.#linkPingTimer = null;
    }
  }

  #stopLinkPing() {
    if (this.#linkPingTimer) {
      clearInterval(this.#linkPingTimer);
      this.#linkPingTimer = null;
    }
  }

  #startLinkPing() {
    this.#stopLinkPing();
    this.#linkPingTimer = setInterval(() => {
      if (this.#socket?.readyState === WebSocket.OPEN) {
        this.#socket.send(
          JSON.stringify({ topic: LINK_PUBSUB_KEEPALIVE_TOPIC, data: {} })
        );
      }
    }, LINK_PUBSUB_KEEPALIVE_MS);
  }

  #scheduleLinkReconnect() {
    if (this.#linkIntentionalClose || !this.#linkFullAddress || this.#linkReconnectTimer)
      return;
    if (this.#linkReconnectAttempts >= LINK_PUBSUB_MAX_ATTEMPTS) {
      toast.error(i18n.t("toasts.pubsub_failed"));
      this.linkDisconnect();
      return;
    }
    const exp = Math.min(this.#linkReconnectAttempts, 5);
    const delay = Math.min(LINK_PUBSUB_MAX_RECONNECT_DELAY_MS, 1000 * Math.pow(2, exp));
    this.#linkReconnectAttempts++;
    this.serviceState.state = ServiceNetworkState.connecting;
    this.#linkReconnectTimer = setTimeout(() => {
      this.#linkReconnectTimer = null;
      this.#openLinkSocket();
    }, delay);
  }

  #openLinkSocket() {
    const fullAddress = this.#linkFullAddress;
    if (!fullAddress || this.#linkIntentionalClose)
      return;

    try {
      this.#socket?.close();
    } catch {
      /* ignore */
    }
    this.#socket = undefined;
    this.#stopLinkPing();

    const wsUrl = `ws://${fullAddress}/pubsub?id=${window.ApiServer.state.id}-${Date.now()}`;
    const socket = new WebSocket(wsUrl);
    this.#socket = socket;

    const connectTimeout = setTimeout(() => {
      if (socket.readyState !== WebSocket.OPEN)
        socket.close();
    }, LINK_PUBSUB_CONNECT_TIMEOUT_MS);

    socket.onopen = () => {
      clearTimeout(connectTimeout);
      this.#linkReconnectAttempts = 0;
      this.serviceState.state = ServiceNetworkState.connected;
      socket.onmessage = (msg) => this.consumePubSubMessage(msg.data, "link");
      this.#startLinkPing();
    };

    socket.onclose = () => {
      clearTimeout(connectTimeout);
      this.#stopLinkPing();
      if (this.#socket === socket)
        this.#socket = undefined;
      if (this.#linkIntentionalClose) {
        this.serviceState.state = ServiceNetworkState.disconnected;
        return;
      }
      this.serviceState.state = ServiceNetworkState.disconnected;
      this.#scheduleLinkReconnect();
    };

    socket.onerror = () => {
      clearTimeout(connectTimeout);
    };
  }

  linkConnect() {
    const ipValidator = /^(([0-9]|[1-9][0-9]|1[0-9]{2}|2[0-4][0-9]|25[0-5])\.){3}([0-9]|[1-9][0-9]|1[0-9]{2}|2[0-4][0-9]|25[0-5]):[0-9]+$/;
    const fullAddress = window.ApiServer.state.linkAddress;
    if (!fullAddress.match(ipValidator))
      return;

    const conf = window.Config.serverNetwork;
    if (`${conf.ip}:${conf.port}` === fullAddress) {
      toast.error(i18n.t("toasts.cannot_connect_self"));
      return;
    }

    this.#clearLinkSchedulers();
    this.#linkIntentionalClose = false;
    this.#linkReconnectAttempts = 0;
    this.#linkFullAddress = fullAddress;
    this.serviceState.state = ServiceNetworkState.connecting;
    this.#openLinkSocket();
  }

  linkDisconnect() {
    this.#linkIntentionalClose = true;
    this.#linkFullAddress = null;
    this.#clearLinkSchedulers();
    try {
      this.#socket?.close();
    } catch {
      /* ignore */
    }
    this.#socket = undefined;
    this.serviceState.state = ServiceNetworkState.disconnected;
  }
}

export default Service_PubSub;
