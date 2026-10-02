import {
  type CaptionKeyed,
  type CaptionKind,
  findCaptionLine,
  isStaleUpdate,
  readCaptionMeta,
  RETIRED_LINE_TTL_MS,
  RetiredCaptionLines,
  supersededCaptionLines,
} from "@/client/elements/text/revising";
import { TextEventSource, TextEventType, type TextEvent } from "@/types";

/**
 * The lines a caption display shows (P6-B: the caption window and the SteamVR overlay each keep one), fed with text
 * events from the caption topics. Pure and clock-injected, so it is unit-tested in node.
 *
 * Lines pair exactly like the text element's revising captions (`src/client/elements/text/revising.ts`): by
 * `segmentId`, then numeric `id`, then (sources with neither) the topic's open interim; stale revisions, interims after
 * a final and late updates of lines that already left are ignored; a new segment ends the open interims of earlier
 * segments of its stream. On top of that a display keeps at most `maxLines` lines (the newest) and lets each line go
 * `clearAfterMs` after its last update (0 = it stays until newer lines push it out). An empty text removes its line
 * (a cleared text field).
 */

export interface CaptionModelOptions {
  /** Show interim (still revising) text; off = finals only. */
  interim: boolean;
  /** Most lines kept (the newest). */
  maxLines: number;
  /** A line leaves this long after its last update (ms); 0 = never by time. */
  clearAfterMs: number;
}

export interface CaptionModelLine extends CaptionKeyed {
  /** Unique, stable key of the line (React key, frame key). */
  readonly key: string;
  text: string;
  /** Committed prefix of an interim (`TextEvent.stableLength`). */
  stableLength?: number;
  /** Lower-case BCP-47 language when the source knows it. */
  lang?: string;
  /** Topic the line was published on (`text.stt`, `others.translation`, ...). */
  readonly source: string;
  /** Model clock time of the last update. */
  updatedAt: number;
  /** A test line ({@link CaptionModel.push}): the clear timer leaves it alone (its owner removes it). */
  pinned?: boolean;
}

const LANG_RE = /^[a-z]{2,3}(-[a-z0-9]{1,8})*$/;

function monotonicNow(): number {
  return typeof performance !== "undefined" && typeof performance.now === "function" ? performance.now() : Date.now();
}

function sameOptions(a: CaptionModelOptions, b: CaptionModelOptions): boolean {
  return a.interim === b.interim && a.maxLines === b.maxLines && a.clearAfterMs === b.clearAfterMs;
}

export class CaptionModel {
  #options: CaptionModelOptions;
  readonly #now: () => number;
  #lines: CaptionModelLine[] = [];
  readonly #retired: RetiredCaptionLines;
  #nextKey = 1;

  constructor(options: CaptionModelOptions, now: () => number = monotonicNow) {
    this.#options = { ...options };
    this.#now = now;
    this.#retired = new RetiredCaptionLines(256, RETIRED_LINE_TTL_MS, now);
  }

  get lines(): readonly CaptionModelLine[] {
    return this.#lines;
  }

  get options(): Readonly<CaptionModelOptions> {
    return this.#options;
  }

  /** New options; true when the shown lines changed (fewer lines allowed, interims turned off). */
  setOptions(options: CaptionModelOptions): boolean {
    if (sameOptions(options, this.#options)) return false;
    this.#options = { ...options };
    let changed = false;
    if (!options.interim) {
      for (let i = this.#lines.length - 1; i >= 0; i--) {
        if (this.#lines[i].kind === "interim") {
          this.#remove(i);
          changed = true;
        }
      }
    }
    return this.#trim() || changed;
  }

  /** Apply an event published on `topic`; true when the shown lines changed. */
  apply(topic: string, event: TextEvent): boolean {
    const kind: CaptionKind = event.type === TextEventType.interim ? "interim" : "final";
    if (kind === "interim" && !this.#options.interim) return false;
    // The AI transform publishes each synced line on two topics with the same ids: they share one line.
    const pairTopic = topic === TextEventSource.transform_raw ? TextEventSource.transform_source : topic;
    const meta = readCaptionMeta(pairTopic, kind, event);
    const index = findCaptionLine(this.#lines, meta);
    const target = index === -1 ? undefined : this.#lines[index];
    if (target ? isStaleUpdate(target, meta) : this.#retired.blocks(meta)) return false;

    const value = typeof event.value === "string" ? event.value : "";
    if (!value.trim()) {
      if (!target) return false;
      this.#remove(index);
      return true;
    }
    const lang = typeof event.lang === "string" && LANG_RE.test(event.lang.toLowerCase()) ? event.lang.toLowerCase() : undefined;
    const stableLength =
      kind === "interim" && typeof event.stableLength === "number" && Number.isInteger(event.stableLength) && event.stableLength >= 0
        ? event.stableLength
        : undefined;
    const now = this.#now();

    if (target) {
      const changed =
        target.text !== value || target.kind !== kind || target.stableLength !== stableLength || (lang !== undefined && target.lang !== lang);
      target.text = value;
      target.kind = kind;
      target.stableLength = stableLength;
      if (lang !== undefined) target.lang = lang;
      if (meta.revision !== undefined) target.revision = meta.revision;
      if (meta.segmentId !== undefined) target.segmentId = meta.segmentId;
      if (meta.id !== undefined) target.id = meta.id;
      target.updatedAt = now;
      return changed;
    }

    // A new segment: open interims of earlier segments of its stream were given up upstream.
    const superseded = supersededCaptionLines(this.#lines, meta);
    for (let i = superseded.length - 1; i >= 0; i--) this.#remove(superseded[i]);
    this.#retired.revive(meta);
    this.#lines.push({ ...meta, key: `l${this.#nextKey++}`, text: value, stableLength, lang, source: topic, updatedAt: now });
    this.#trim();
    return true;
  }

  /** Show `text` as a final line of its own (the "test caption" button), untouched by the clear timer; returns its key. */
  push(text: string, source: string, lang?: string): string {
    const key = `l${this.#nextKey++}`;
    this.#lines.push({ topic: source, kind: "final", key, text, source, lang, updatedAt: this.#now(), pinned: true });
    this.#trim();
    return key;
  }

  /** Remove the line with `key`; true when it was shown. */
  removeKey(key: string): boolean {
    const i = this.#lines.findIndex((l) => l.key === key);
    if (i === -1) return false;
    this.#remove(i);
    return true;
  }

  /** Remove the lines whose clear timer ran out; true when any left. */
  expire(): boolean {
    const ttl = this.#options.clearAfterMs;
    if (!(ttl > 0)) return false;
    const now = this.#now();
    let changed = false;
    for (let i = this.#lines.length - 1; i >= 0; i--) {
      if (!this.#lines[i].pinned && now - this.#lines[i].updatedAt >= ttl) {
        this.#remove(i);
        changed = true;
      }
    }
    return changed;
  }

  /** Model-clock time at which the next line expires, or null. */
  nextExpiry(): number | null {
    const ttl = this.#options.clearAfterMs;
    const timed = this.#lines.filter((l) => !l.pinned);
    if (!(ttl > 0) || timed.length === 0) return null;
    return Math.min(...timed.map((l) => l.updatedAt)) + ttl;
  }

  /** Remove every line (the display was turned off or its sources changed); true when any were shown. */
  clear(): boolean {
    const had = this.#lines.length > 0;
    for (let i = this.#lines.length - 1; i >= 0; i--) this.#remove(i);
    return had;
  }

  #remove(index: number): void {
    const [line] = this.#lines.splice(index, 1);
    if (line) this.#retired.retire(line);
  }

  #trim(): boolean {
    const max = Math.max(1, Math.floor(this.#options.maxLines) || 1);
    let changed = false;
    while (this.#lines.length > max) {
      this.#remove(0);
      changed = true;
    }
    return changed;
  }
}
