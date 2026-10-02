/**
 * Where other players' transcripts go (P6-A): `others.stt` (never `text.*`, so they never reach the chatbox, TTS,
 * history or any "any text" consumer) and the reverse translation pipeline, which publishes on `others.translation`.
 */
import type { SttTextMeta } from "@/core/services/stt/types";
import { OthersEventSource, TextEventSchema, TextEventType, type PartialWithRequired, type TextEvent } from "@/types";

/** Translation stream name of other players' speech (independent of the main pipeline's `stt` stream). */
export const OTHERS_TRANSLATION_STREAM = "others";

/** Revision metadata validated with the published `TextEvent` rules (malformed values are dropped). */
const META_SCHEMA = TextEventSchema.pick({ segmentId: true, revision: true, lang: true, turnEnd: true, tags: true });

/** `meta` as `TextEvent` fields, without the ones that are absent or malformed. */
export function othersTextMeta(meta: SttTextMeta | undefined): Pick<TextEvent, "segmentId" | "revision" | "lang" | "turnEnd" | "tags"> {
  const parsed = META_SCHEMA.safeParse(meta ?? {});
  const out: Pick<TextEvent, "segmentId" | "revision" | "lang" | "turnEnd" | "tags"> = {};
  if (!parsed.success) return out;
  const m = parsed.data;
  if (m.segmentId !== undefined) out.segmentId = m.segmentId;
  if (m.revision !== undefined) out.revision = m.revision;
  if (m.lang !== undefined) out.lang = m.lang;
  if (m.turnEnd !== undefined) out.turnEnd = m.turnEnd;
  if (m.tags !== undefined && m.tags.length > 0) out.tags = [...m.tags];
  return out;
}

export interface OthersPublisherOptions {
  /** Publish a caption event (production: `window.ApiShared.pubsub.publishText`). */
  publish(topic: OthersEventSource, event: PartialWithRequired<TextEvent, "type" | "value">): void;
  /** Feed the reverse translation pipeline (it ignores events while it is not running). */
  translate(stream: string, event: TextEvent): void;
  /** Interims are published (and translated); off = finals only. */
  interimEnabled(): boolean;
}

/** Turns the driver's transcripts into `others.stt` events and reverse-translation input. */
export class OthersPublisher {
  constructor(private readonly options: OthersPublisherOptions) {}

  interim(text: string, meta?: SttTextMeta): void {
    if (!this.options.interimEnabled()) return;
    this.#emit(TextEventType.interim, text, meta);
  }

  final(text: string, meta?: SttTextMeta): void {
    this.#emit(TextEventType.final, text, meta);
  }

  #emit(type: TextEventType, text: string, meta: SttTextMeta | undefined): void {
    const value = String(text ?? "").trim();
    if (!value) return;
    const fields = othersTextMeta(meta);
    // Separate objects: a subscriber (or the emote enricher) never changes what the translation sees.
    this.options.publish(OthersEventSource.stt, { type, value, ...fields, ...(fields.tags ? { tags: [...fields.tags] } : {}) });
    this.options.translate(OTHERS_TRANSLATION_STREAM, { type, value, emotes: {}, ...fields });
  }
}
