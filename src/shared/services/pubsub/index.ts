import { BaseEvent, CaptionEventSource, IServiceInterface, OthersEventSource, PartialWithRequired, ReadingsEventSource, ServiceNetworkState, TextEvent, TextEventSource, TextEventType } from "@/types";
import { listen } from '@tauri-apps/api/event';
import { invoke } from "@tauri-apps/api/core";
import { nanoid } from "nanoid";
import PubSub from "pubsub-js";
import i18n from "i18next";
import { toast } from "react-toastify";
import { proxy, subscribe } from "valtio";
import { proxyMap, subscribeKey } from "valtio/utils";
import {
  isValidLanToken,
  lanLinkAddress,
  lanRefusalFromCloseCode,
  linkSocketUrl,
  normalizeLanToken,
  parseLinkAddress,
  type LanAccessMode,
  type LanAccessSettings,
  type LinkAddress,
} from "@/shared/services/peer/lanAccess";

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

/** LAN access part of {@link WebServerStatus} (Rust `LanStatus`; never contains the token). */
export type WebLanStatus = {
  /** The requested `lanAccess.mode`. */
  mode: LanAccessMode;
  /** The mode being served; differs from `mode` while a switch failed (see `error`), null while not listening. */
  boundMode: LanAccessMode | null;
  /** The listening address, e.g. `0.0.0.0:3030`. */
  boundAddress: string | null;
  /** Whether a valid LAN token is in force. */
  tokenSet: boolean;
  legacyNoToken: boolean;
};

/** `plugin:web|server_status` / `web:server_status` payload (Rust `WebServerStatus`). */
export type WebServerStatus = {
  state: "starting" | "listening" | "failed" | "stopped";
  port: number;
  /** Bind / server error when `state` is `failed` or `stopped`, or why a LAN mode switch did not happen while `listening`. */
  error: string | null;
  /** Whether the Origin allow-list is enforced right now. */
  originCheck: boolean;
  /** The `lanOriginCheck` setting as Rust last saw it. */
  originCheckSetting: boolean;
  /** `SIGIL_LAN_ORIGIN_CHECK` override (wins over the setting), or null. */
  originCheckEnv: boolean | null;
  /** LAN security stage 2 (P6-D). */
  lan: WebLanStatus;
};

/** Refusal reasons of the LAN token / mode (stage 2). */
export type WebLanRefusalReason = "missing_token" | "invalid_token" | "lan_disabled";

/**
 * `web:origin_rejected` payload (Rust `AccessRejection`), throttled per origin (or peer address) in
 * Rust: refused or revoked `/pubsub` and `/peer` connections, by the Origin allow-list (stage 1) or
 * the LAN token and mode (stage 2).
 */
export type WebOriginRejection = {
  route: string;
  origin: string | null;
  remote: string | null;
  reason: "foreign_origin" | "null_origin" | "malformed_origin" | "missing_origin_remote" | WebLanRefusalReason;
};

/** Reply of `plugin:web|regenerate_lan_token`. */
type LanTokenReply = { token: string; status: WebServerStatus };

/** The settings the host web server follows (a valtio proxy: the persisted root state). */
export type HostWebServerSettings = { lanOriginCheck: boolean; lanAccess: LanAccessSettings };

const LAN_REFUSAL_REASONS: ReadonlySet<string> = new Set<WebLanRefusalReason>(["missing_token", "invalid_token", "lan_disabled"]);

/** The `lanAccess` values Rust is told about, as one comparable string. */
function lanAccessKey(lan: LanAccessSettings): string {
  return JSON.stringify([lan.mode, lan.token, lan.legacyNoToken === true]);
}

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
  /** The other Sigil this host links to (address and remote LAN token), while linking. */
  #link: LinkAddress | null = null;
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
  /** The host's persisted settings, once {@link attachHostWebServer} ran. */
  #hostSettings?: HostWebServerSettings;
  /** {@link lanAccessKey} of the `lanAccess` values last handled (pushed or regenerated). */
  #lastLanKey = "";
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
   * Host app, after settings and i18n are loaded: keep the Rust web server in sync with the
   * persisted settings (a valtio proxy) — the Origin-check kill switch (`lanOriginCheck`) and LAN
   * access (`lanAccess`: mode, token, legacy; P6-D) — make sure a LAN token exists, and surface a
   * failed web-server bind and refused WebSocket connections as toasts. Idempotent.
   */
  async attachHostWebServer(settings: HostWebServerSettings) {
    if (!window.Config.isServer() || !window.Config.isApp() || this.#hostWebServerAttached)
      return;
    this.#hostWebServerAttached = true;
    this.#hostSettings = settings;

    const pushOriginCheck = (enabled: boolean) =>
      invoke<WebServerStatus>("plugin:web|set_origin_check", { enabled })
        .then((status) => this.#onServerStatus(status))
        .catch((e) => console.warn("[pubsub] set_origin_check failed", e));
    subscribeKey(settings, "lanOriginCheck", (enabled) => void pushOriginCheck(enabled !== false));

    // `lanAccess` may be patched field by field or replaced whole: watch the root, compare values.
    this.#lastLanKey = lanAccessKey(settings.lanAccess);
    subscribe(settings, () => {
      const key = lanAccessKey(settings.lanAccess);
      if (key === this.#lastLanKey)
        return;
      this.#lastLanKey = key;
      void this.#syncLanAccess();
    });

    try {
      await listen<WebServerStatus>("web:server_status", (e) => this.#onServerStatus(e.payload));
      await listen<WebOriginRejection>("web:origin_rejected", (e) => this.#onOriginRejected(e.payload));
    } catch (e) {
      console.warn("[pubsub] web server event listeners failed", e);
    }
    await this.#syncLanAccess();
    // The reply carries the current status, so a bind failure from before the webview was
    // listening is surfaced too.
    await pushOriginCheck(settings.lanOriginCheck !== false);
  }

  /**
   * Push `lanAccess` to Rust; when the persisted token is missing (first start) or malformed,
   * generate one (Rust) and store it, which pushes again through the settings subscription.
   */
  async #syncLanAccess() {
    const settings = this.#hostSettings;
    if (!settings)
      return;
    const lan = settings.lanAccess;
    const payload: LanAccessSettings = {
      mode: lan.mode === "localhost" ? "localhost" : "lan",
      token: normalizeLanToken(lan.token) ?? "",
      legacyNoToken: lan.legacyNoToken === true,
    };
    try {
      this.#onServerStatus(await invoke<WebServerStatus>("plugin:web|set_lan_access", { settings: payload }));
    } catch (e) {
      console.warn("[pubsub] set_lan_access failed", e);
    }
    if (!isValidLanToken(lan.token))
      await this.regenerateLanToken();
    else if (payload.token !== lan.token)
      settings.lanAccess.token = payload.token; // store the canonical form
  }

  /**
   * Host app: make a new LAN token (Rust's CSPRNG), put it in force at once (devices on other
   * machines still using the old one are disconnected) and persist it in `lanAccess.token`.
   * Returns the token, or null when it failed (a toast says why).
   */
  async regenerateLanToken(): Promise<string | null> {
    const settings = this.#hostSettings;
    if (!settings)
      return null;
    try {
      const reply = await invoke<LanTokenReply>("plugin:web|regenerate_lan_token");
      // Rust already has it: the settings subscription only needs to push it if something else changed.
      this.#lastLanKey = lanAccessKey({ ...settings.lanAccess, token: reply.token });
      settings.lanAccess.token = reply.token;
      this.#onServerStatus(reply.status);
      return reply.token;
    } catch (e) {
      console.error("[pubsub] regenerate_lan_token failed", e);
      toast.error(i18n.t("toasts.lan_token_regenerate_failed", { error: e instanceof Error ? e.message : String(e) }), {
        toastId: "lan-token-regenerate",
      });
      return null;
    }
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
    if (LAN_REFUSAL_REASONS.has(rejection.reason)) {
      const remote = (rejection.remote ?? "?").slice(0, 64);
      const key = rejection.reason === "lan_disabled" ? "toasts.web_lan_disabled_refused" : "toasts.web_lan_token_refused";
      toast.warn(i18n.t(key, { remote }), { toastId: "web-lan-refused", autoClose: 12_000 });
      return;
    }
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
    // P6-S0: other players (P6-A) and readings (P6-L); outside `text.*`, so "Any text source" never hears them.
    this.registerEvent({ label: "Other players: Speech-to-Text", value: OthersEventSource.stt });
    this.registerEvent({ label: "Other players: Translation", value: OthersEventSource.translation });
    this.registerEvent({ label: "Readings (romaji / pinyin)", value: ReadingsEventSource.stt });

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
  /** Publish a caption: own text (`text.*`) or other players' (`others.*`, P6-A). */
  publishText(topic: CaptionEventSource, textData: PartialWithRequired<TextEvent, "type" | "value">) {
    let data = this.applyEmotes(textData);
    this.publish(topic, data);
  }

  public unsubscribe(key?: string) {
    key && PubSub.unsubscribe(key);
  }

  public subscribe(eventname: string, fn: (value: unknown) => void) {
    return PubSub.subscribe(eventname, (_, data) => fn(data));
  }

  /** Subscribe to a caption topic (`text.*` or `others.*`); `text` hears every `text.*` but never `others.*`. */
  public subscribeText(source: CaptionEventSource, fn: (value?: TextEvent, eventName?: string) => void, allowEmpty = false) {
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

  /**
   * Copy this host's Link apps address for another Sigil: `<LAN IP>:<port>?token=<LAN token>`
   * (P6-D; pasting it there fills the address and the remote token). `ip` picks another of this
   * PC's addresses.
   */
  copyLinkAddress(ip?: string) {
    const conf = window.Config.serverNetwork;
    const token = normalizeLanToken(window.ApiServer?.state?.lanAccess?.token) ?? "";
    navigator.clipboard.writeText(lanLinkAddress(ip || conf.ip, conf.port, token));
    if (this.webServer.status?.lan?.mode === "localhost")
      toast.warn(i18n.t("toasts.lan_access_this_pc_only"), { toastId: "lan-this-pc-only" });
    else
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
    if (this.#linkIntentionalClose || !this.#link || this.#linkReconnectTimer)
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
    const link = this.#link;
    if (!link || this.#linkIntentionalClose)
      return;

    try {
      this.#socket?.close();
    } catch {
      /* ignore */
    }
    this.#socket = undefined;
    this.#stopLinkPing();

    // The other Sigil's LAN token rides along as `&token=` (P6-D).
    const wsUrl = linkSocketUrl(link, `${window.ApiServer.state.id}-${Date.now()}`);
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

    socket.onclose = (event?: CloseEvent) => {
      clearTimeout(connectTimeout);
      this.#stopLinkPing();
      const wasCurrent = this.#socket === socket;
      if (wasCurrent)
        this.#socket = undefined;
      if (this.#linkIntentionalClose) {
        this.serviceState.state = ServiceNetworkState.disconnected;
        return;
      }
      this.serviceState.state = ServiceNetworkState.disconnected;
      // The other Sigil refused this link (missing / old token, or it accepts only its own PC):
      // retrying cannot help until the user pastes its address again.
      const refusal = lanRefusalFromCloseCode(event?.code);
      if (refusal && wasCurrent) {
        toast.error(i18n.t(refusal === "token" ? "toasts.link_token_refused" : "toasts.link_lan_disabled"), {
          toastId: "link-refused",
          autoClose: 15_000,
        });
        this.linkDisconnect();
        return;
      }
      this.#scheduleLinkReconnect();
    };

    socket.onerror = () => {
      clearTimeout(connectTimeout);
    };
  }

  /**
   * Link to the Sigil at `linkAddress`: `<IPv4>:<port>`, optionally with `?token=<its LAN token>`
   * (P6-D; Settings stores the remote token that way). Needed when that Sigil is on another PC.
   */
  linkConnect() {
    const link = parseLinkAddress(window.ApiServer.state.linkAddress ?? "");
    if (!link)
      return;
    if (link.tokenInvalid) {
      toast.error(i18n.t("settings.link_remote_token_invalid"), { toastId: "link-token-invalid" });
      return;
    }

    const conf = window.Config.serverNetwork;
    if (`${conf.ip}:${conf.port}` === `${link.host}:${link.port}`) {
      toast.error(i18n.t("toasts.cannot_connect_self"));
      return;
    }

    this.#clearLinkSchedulers();
    this.#linkIntentionalClose = false;
    this.#linkReconnectAttempts = 0;
    this.#link = link;
    this.serviceState.state = ServiceNetworkState.connecting;
    this.#openLinkSocket();
  }

  linkDisconnect() {
    this.#linkIntentionalClose = true;
    this.#link = null;
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
