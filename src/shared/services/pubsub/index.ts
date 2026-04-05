import { BaseEvent, IServiceInterface, ObsCaptionEnvelopeSchema, PartialWithRequired, ServiceNetworkState, TextEvent, TextEventSchema, TextEventSource, TextEventType } from "@/types";
import { listen } from '@tauri-apps/api/event';
import { invoke } from "@tauri-apps/api/core";
import { nanoid } from "nanoid";
import PubSub from "pubsub-js";
import i18n from "i18next";
import { toast } from "react-toastify";
import { proxy } from "valtio";
import { proxyMap } from "valtio/utils";

import { z } from "zod";

/** Application WS ping interval for linked pubsub (see docs/PERFORMANCE_AUDIT.md). */
export const LINK_PUBSUB_KEEPALIVE_MS = 25_000;

const LINK_PUBSUB_CONNECT_TIMEOUT_MS = 12_000;
const LINK_PUBSUB_KEEPALIVE_TOPIC = "__sigil_keepalive";
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

export const PUBSUB_STT_EXTERNAL_STATE = "stt.external.state";
export const PUBSUB_STT_EXTERNAL_COMMAND = "stt.external.command";

const ExternalMicLifecycleStateSchema = z.object({ active: z.boolean() });

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

  private consumePubSubMessage(stringEvent?: string) {
    if (!window.Config.isServer())
      return;
    if (typeof stringEvent === "string") try {
      const raw = JSON.parse(stringEvent) as { topic?: string; data?: unknown };
      if (raw?.topic === LINK_PUBSUB_KEEPALIVE_TOPIC)
        return;
      const { topic, data }: BaseEvent = raw as BaseEvent;
      if (typeof data !== "object")
        return;

      if (topic === "obs.caption") {
        const caption = ObsCaptionEnvelopeSchema.safeParse(data);
        if (!caption.success) return;
        const msg = { topic, data: caption.data };
        this.publishLocally(msg);
        this.#fanOutPubSubRemote(topic, caption.data);
        return;
      }

      if (topic === PUBSUB_STT_EXTERNAL_STATE) {
        const parsed = ExternalMicLifecycleStateSchema.safeParse(data);
        if (!parsed.success) return;
        this.#externalMicLifecycleHandler?.(parsed.data.active);
        return;
      }

      const validated = TextEventSchema.safeParse(data);
      if (!validated.success)
        return;

      const textEvent = this.applyEmotes(validated.data);
      if (topic === "text.stt") {
        this.#externalSttHandler?.({
          ...textEvent,
          emotes: textEvent.emotes ?? {},
        });
        return;
      }

      const msg = { topic, data: textEvent };
      this.publishLocally(msg);
      this.#fanOutPubSubRemote(topic, textEvent);
    } catch (error) {
      // just ignore invalid messages
    }
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

  async init() {
    window.Config.isServer() && listen('pubsub', (event) => {
      this.consumePubSubMessage(event.payload as string);
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

  /** PeerJS + localhost WS + link socket; local PubSub already handled separately when needed. */
  #fanOutPubSubRemote(topic: string, data: unknown) {
    const dataJson = JSON.stringify(data);
    window.ApiShared.peer.broadcastPubSubParts(topic, dataJson);
    const wire = `{"topic":${JSON.stringify(topic)},"data":${dataJson}}`;
    if (window.Config.isApp())
      void invoke("plugin:web|pubsub_broadcast", { value: wire });
    if (this.#socket && this.#socket.readyState === this.#socket.OPEN)
      this.#socket.send(wire);
  }

  publish(topic: string, data: any) {
    if (window.Config.isClient())
      return;
    const msg = { topic, data };
    this.publishLocally(msg);
    this.#fanOutPubSubRemote(topic, data);
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
      socket.onmessage = (msg) => this.consumePubSubMessage(msg.data as string);
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
