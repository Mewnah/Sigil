/**
 * Inbound pubsub routing (LAN security stage 1, P0-I): which messages the host accepts from the
 * local WebSocket server (`/pubsub`: mic.html, inputfield.html, local tools, other Sigil hosts that
 * linked to this one) and from its own Link-apps socket, and how each is handled.
 *
 * Pure and DOM-free so it is unit-tested (`__tests__/inbound.test.ts`); `Service_PubSub` executes
 * the returned route.
 *
 * Loop prevention: every message a Sigil host sends over a WebSocket carries `via`, the ids of the
 * host sessions it has passed through (oldest first). A host drops messages that already list its
 * own session id and appends its id when it relays, so Link apps can no longer bounce an event
 * between two hosts forever. Pages and tools send no `via`; older Sigil versions ignore it.
 *
 * Duplicate suppression: `via` stops loops but not two copies of one event arriving by different
 * paths (two hosts linked to each other, or any diamond). Every message a host originates or relays
 * also carries `mid`, a random id set once by the originating host and kept by every relay. A host
 * remembers recently seen ids ({@link RecentMessageIds}) and drops repeats.
 *
 * Peer text is untrusted: the canvas text element renders emote URLs and (in word animation) the
 * text itself as HTML in the host webview, which has full Tauri IPC. So the boundary drops
 * peer-supplied `emotes` (the host rescans with its own emote cache) and neutralizes tag-like `<`
 * sequences in the text ({@link neutralizeMarkup}).
 */
import {
  ObsCaptionEnvelopeSchema,
  OthersEventSource,
  ReadingsEventSchema,
  ReadingsEventSource,
  TextEventSchema,
  TextEventSource,
  type BaseEvent,
  type ObsCaptionEnvelope,
  type ReadingsEvent,
  type TextEvent,
} from "@/types";
import { nanoid } from "nanoid";
import { z } from "zod";

/** Where an inbound message came from. */
export type InboundSource =
  /** The host's own WebSocket server (Tauri `pubsub` event). */
  | "local"
  /** This host's outbound Link-apps socket to another Sigil. */
  | "link";

/** Application-level keepalive on Link-apps sockets; never routed. */
export const LINK_PUBSUB_KEEPALIVE_TOPIC = "__sigil_keepalive";

/** mic.html → host lifecycle topic (the host never relays it). */
export const PUBSUB_STT_EXTERNAL_STATE = "stt.external.state";
/** Host → mic.html command topic (never accepted inbound). */
export const PUBSUB_STT_EXTERNAL_COMMAND = "stt.external.command";

/** Longest accepted `via` chain; longer chains are treated as a loop and dropped. */
export const MAX_VIA_HOPS = 8;
const MAX_VIA_ID_LENGTH = 64;

/**
 * Text topics a peer may publish into this host. `text` itself (the bubbling parent every
 * subscriber hears) is deliberately absent.
 */
export const INBOUND_TEXT_TOPICS: ReadonlySet<string> = new Set<string>([
  TextEventSource.stt,
  TextEventSource.textfield,
  TextEventSource.translation,
  TextEventSource.transform,
  TextEventSource.transform_source,
  TextEventSource.transform_raw,
]);

const ViaSchema = z.array(z.string().min(1).max(MAX_VIA_ID_LENGTH)).max(MAX_VIA_HOPS);
const ExternalMicStateSchema = z.object({ active: z.boolean() });

/** Message ids (`mid`): nanoid alphabet, bounded length. */
const MessageIdSchema = z.string().regex(/^[A-Za-z0-9_-]{1,64}$/);

/** How many recent message ids a host remembers for duplicate suppression. */
export const RECENT_MESSAGE_IDS = 1024;

/** A fresh message id for an event this host originates (or relays without one). */
export function newMessageId(): string {
  return nanoid(16);
}

/** Bounded memory of recently seen message ids; the oldest id is forgotten first. */
export class RecentMessageIds {
  #ids = new Set<string>();
  constructor(readonly capacity = RECENT_MESSAGE_IDS) {}

  has(id: string): boolean {
    return this.#ids.has(id);
  }

  /** Remember `id`. Returns false when it was already known (a duplicate). */
  add(id: string): boolean {
    if (this.#ids.has(id))
      return false;
    this.#ids.add(id);
    if (this.#ids.size > this.capacity) {
      const oldest = this.#ids.values().next().value;
      if (oldest !== undefined)
        this.#ids.delete(oldest);
    }
    return true;
  }

  get size(): number {
    return this.#ids.size;
  }
}

/**
 * `<` followed by a letter, `/`, `!` or `?` starts markup in HTML; anything else after `<` is text.
 */
const TAG_OPEN = /<(?=[A-Za-z/!?])/g;

/**
 * Make untrusted text inert if some sink parses it as HTML: a zero-width space goes after every
 * `<` that would open a tag, so `<img …>` stays visible text. Ordinary text (`<3`, `a < b`) is
 * unchanged. Idempotent.
 */
export function neutralizeMarkup(value: string): string {
  return value.replace(TAG_OPEN, "<\u200B");
}

/** A text event from a peer after the boundary: no `emotes` (the host rescans), markup neutralized. */
export type PeerTextEvent = Omit<TextEvent, "emotes">;

/** Peer text as the host accepts it: markup neutralized, peer emotes dropped (the host rescans). */
function sanitizePeerTextEvent(event: TextEvent): PeerTextEvent {
  // Without `emotes` the host's emote enricher scans the text with its own emote cache instead of
  // trusting peer URLs and keys, which the text element writes into HTML unescaped.
  const { emotes: _peerEmotes, ...rest } = event;
  return { ...rest, value: neutralizeMarkup(rest.value) };
}

/** Longest emote URL a client keeps. */
const MAX_EMOTE_URL_LENGTH = 512;
/** Most emote entries (cursor and word keys) a client keeps per text event. */
const MAX_EMOTE_ENTRIES = 256;
/** Longest emote key (a cursor number or an emote word). */
const MAX_EMOTE_KEY_LENGTH = 100;
/**
 * Emote CDN URLs: `https://` or protocol-relative (7TV), host, then a path without quotes,
 * whitespace, `<`, `>`, `=`, `&`, `?`, `#`, backticks or backslashes, so a URL can never add an
 * attribute or tag even where it is written into HTML unquoted.
 */
const SAFE_EMOTE_URL = /^(?:https:)?\/\/[A-Za-z0-9.-]+(?::\d{1,5})?(?:\/[A-Za-z0-9._~%+:@/-]*)?$/;

/**
 * Keep only emote entries whose URL matches {@link SAFE_EMOTE_URL}, with bounded keys and count.
 * For text a client receives from its host (the host already dropped peer emotes and scanned its
 * own); returns a new object.
 */
export function sanitizeEmotes(emotes: unknown): Record<string, string> {
  const out: Record<string, string> = {};
  if (!emotes || typeof emotes !== "object" || Array.isArray(emotes))
    return out;
  let kept = 0;
  for (const [key, url] of Object.entries(emotes as Record<string, unknown>)) {
    if (kept >= MAX_EMOTE_ENTRIES)
      break;
    if (!key || key.length > MAX_EMOTE_KEY_LENGTH || key === "__proto__")
      continue;
    if (typeof url !== "string" || url.length > MAX_EMOTE_URL_LENGTH || !SAFE_EMOTE_URL.test(url))
      continue;
    out[key] = url;
    kept++;
  }
  return out;
}

/** `topic` is `root` or one of its children (`root.*`). */
function inHierarchy(topic: string, root: string): boolean {
  return topic === root || topic.startsWith(`${root}.`);
}

/** Readings as a client keeps them: every text field markup-neutralized (P6-S0). */
function sanitizeReadingsEvent(event: ReadingsEvent): ReadingsEvent {
  const n = (s: string | undefined) => (s === undefined ? undefined : neutralizeMarkup(s));
  return {
    ...event,
    value: neutralizeMarkup(event.value),
    hiragana: n(event.hiragana),
    romaji: n(event.romaji),
    pinyin: n(event.pinyin),
    segments: event.segments?.map((s) => ({ text: neutralizeMarkup(s.text), reading: n(s.reading) })),
  };
}

/**
 * For the /client PeerJS bridge (frames from the host it connected to): validates caption frames
 * (`text.*` and other players' `others.*`, P6-S0) with {@link TextEventSchema}, keeps only safe
 * emote URLs ({@link sanitizeEmotes}) and neutralizes markup in the text; `readings.*` frames
 * (P6-S0) must match {@link ReadingsEventSchema} and get every string neutralized. Other topics pass
 * through unchanged. Returns null for malformed frames.
 */
export function sanitizeClientPubSubFrame(msg: unknown): BaseEvent | null {
  if (!msg || typeof msg !== "object" || Array.isArray(msg))
    return null;
  const { topic, data } = msg as { topic?: unknown; data?: unknown };
  if (typeof topic !== "string" || !topic)
    return null;
  if (inHierarchy(topic, ReadingsEventSource.any)) {
    const readings = ReadingsEventSchema.safeParse(data);
    return readings.success ? { topic, data: sanitizeReadingsEvent(readings.data) } : null;
  }
  if (!inHierarchy(topic, TextEventSource.any) && !inHierarchy(topic, OthersEventSource.any))
    return { topic, data };
  const text = TextEventSchema.safeParse(data);
  if (!text.success)
    return null;
  return {
    topic,
    data: { ...text.data, value: neutralizeMarkup(text.data.value), emotes: sanitizeEmotes(text.data.emotes) },
  };
}

export type InboundRoute =
  | { kind: "ignore"; reason: InboundIgnoreReason }
  /** `obs.caption` envelope: publish locally and relay (keeping `mid` when the sender set one). */
  | { kind: "caption"; topic: "obs.caption"; data: ObsCaptionEnvelope; via: string[]; mid?: string }
  /** mic.html reports that Web Speech started or stopped. */
  | { kind: "mic_state"; active: boolean }
  /** Speech from mic.html or a local tool: goes through the host's STT pipeline (mute, filters). */
  | { kind: "external_stt"; event: PeerTextEvent }
  /** A text event: publish locally (after the host's emote scan) and relay, keeping `mid` if set. */
  | { kind: "text"; topic: TextEventSource; data: PeerTextEvent; via: string[]; mid?: string };

export type InboundIgnoreReason =
  | "not_json"
  | "keepalive"
  | "own_echo"
  | "bad_via"
  | "bad_mid"
  | "duplicate"
  | "topic_not_allowed"
  | "invalid_data";

/**
 * Decide what to do with one inbound message. `selfId` is this host's pubsub session id. `seen`,
 * when given, is this host's memory of recent message ids: a message whose `mid` is already in it
 * is ignored as a duplicate, and an accepted message's `mid` is added to it. Never throws.
 */
export function routeInboundMessage(raw: string, source: InboundSource, selfId: string, seen?: RecentMessageIds): InboundRoute {
  let parsed: unknown;
  try {
    parsed = JSON.parse(raw);
  } catch {
    return { kind: "ignore", reason: "not_json" };
  }
  if (!parsed || typeof parsed !== "object" || Array.isArray(parsed))
    return { kind: "ignore", reason: "not_json" };
  const { topic, data, via: rawVia, mid: rawMid } = parsed as { topic?: unknown; data?: unknown; via?: unknown; mid?: unknown };
  if (typeof topic !== "string")
    return { kind: "ignore", reason: "topic_not_allowed" };
  if (topic === LINK_PUBSUB_KEEPALIVE_TOPIC)
    return { kind: "ignore", reason: "keepalive" };

  let via: string[] = [];
  if (rawVia !== undefined) {
    const v = ViaSchema.safeParse(rawVia);
    if (!v.success)
      return { kind: "ignore", reason: "bad_via" };
    via = v.data;
  }
  if (via.includes(selfId))
    return { kind: "ignore", reason: "own_echo" };
  const fromSigilHost = via.length > 0;

  let mid: string | undefined;
  if (rawMid !== undefined) {
    const m = MessageIdSchema.safeParse(rawMid);
    if (!m.success)
      return { kind: "ignore", reason: "bad_mid" };
    mid = m.data;
  }
  if (mid !== undefined && seen?.has(mid))
    return { kind: "ignore", reason: "duplicate" };
  /** Accept: remember the id so a second copy arriving by another path is dropped. */
  const accept = <R extends InboundRoute>(route: R): R => {
    if (mid !== undefined)
      seen?.add(mid);
    return route;
  };

  if (!data || typeof data !== "object" || Array.isArray(data))
    return { kind: "ignore", reason: "invalid_data" };

  if (topic === "obs.caption") {
    const caption = ObsCaptionEnvelopeSchema.safeParse(data);
    return caption.success
      ? accept({ kind: "caption", topic, data: caption.data, via, ...(mid !== undefined && { mid }) })
      : { kind: "ignore", reason: "invalid_data" };
  }

  if (topic === PUBSUB_STT_EXTERNAL_STATE) {
    // Only a page connected to this host (mic.html) controls this host's external-mic state; a
    // linked host's mic page must not start or stop ours.
    if (source !== "local" || fromSigilHost)
      return { kind: "ignore", reason: "topic_not_allowed" };
    const state = ExternalMicStateSchema.safeParse(data);
    return state.success
      ? { kind: "mic_state", active: state.data.active }
      : { kind: "ignore", reason: "invalid_data" };
  }

  if (!INBOUND_TEXT_TOPICS.has(topic))
    return { kind: "ignore", reason: "topic_not_allowed" };

  const text = TextEventSchema.safeParse(data);
  if (!text.success)
    return { kind: "ignore", reason: "invalid_data" };
  const event = sanitizePeerTextEvent(text.data);

  // Speech from mic.html / a local tool runs through this host's STT pipeline. Another Sigil's STT
  // output (it carries `via`, or arrives on our Link socket) is already final text: re-running it
  // through our STT would publish it again as ours and bounce it back.
  if (topic === TextEventSource.stt && source === "local" && !fromSigilHost)
    return accept({ kind: "external_stt", event });

  return accept({ kind: "text", topic: topic as TextEventSource, data: event, via, ...(mid !== undefined && { mid }) });
}

/** `via` for a message this host relays or originates: the incoming chain plus `selfId`. */
export function nextVia(via: readonly string[], selfId: string): string[] {
  const chain = via.filter((id) => id !== selfId);
  chain.push(selfId);
  return chain.slice(-MAX_VIA_HOPS);
}

/**
 * Wire format sent to WebSocket peers and the Link socket (`data` already JSON-encoded). Pages
 * read `topic` and `data` and ignore the rest.
 */
export function encodeWire(topic: string, dataJson: string, via: readonly string[], mid?: string): string {
  const midPart = mid === undefined ? "" : `,"mid":${JSON.stringify(mid)}`;
  return `{"topic":${JSON.stringify(topic)},"data":${dataJson},"via":${JSON.stringify(via)}${midPart}}`;
}
