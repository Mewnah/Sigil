import { BaseEvent, IServiceInterface, PartialWithRequired, ServiceNetworkState, TextEvent, TextEventSchema, TextEventSource, TextEventType } from "@/types";
import { listen } from '@tauri-apps/api/event';
import { invoke } from "@tauri-apps/api/core";
import { nanoid } from "nanoid";
import PubSub from "pubsub-js";
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

class Service_PubSub implements IServiceInterface {
  constructor() { }
  #socket?: WebSocket;
  #linkIntentionalClose = false;
  #linkFullAddress: string | null = null;
  #linkReconnectAttempts = 0;
  #linkReconnectTimer: ReturnType<typeof setTimeout> | null = null;
  #linkPingTimer: ReturnType<typeof setInterval> | null = null;
  #textEmoteEnricher?: TextEmoteEnricher;
  #externalSttHandler?: ExternalSttHandler;
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
      this.#publishPeers(msg);
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
        const maxEntries = 120;
        if (this.textHistory.list.length >= maxEntries)
          this.textHistory.list.shift();
        const id = nanoid();
        this.textHistory.list.push({ id, event: eventName?.replace("text.", "") || "text", value: event.value });
        this.textHistory.lastId = id;
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
  #publishPubSub(msg: BaseEvent) {
    window.Config.isApp() &&
      invoke("plugin:web|pubsub_broadcast", { value: JSON.stringify(msg) });
  }
  #publishLink(msg: BaseEvent) {
    if (this.#socket && this.#socket.readyState === this.#socket.OPEN)
      this.#socket.send(JSON.stringify(msg));
  }
  #publishPeers(msg: BaseEvent) {
    window.ApiShared.peer.broadcast(msg);
  }

  publish(topic: string, data: any) {
    if (window.Config.isClient())
      return;
    let msg = { topic, data };
    this.publishLocally(msg);
    this.#publishPeers(msg);
    this.#publishPubSub(msg);
    this.#publishLink(msg);
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
    toast.success("Copied!");
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
      toast.error("PubSub link failed repeatedly. Disconnected.");
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
      toast.error("Cannot connect to self");
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
