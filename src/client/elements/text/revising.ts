/**
 * Revising captions for the text element (P4-G): pure rules plus a DOM-agnostic line view, so they are unit-tested
 * in node; `index.tsx` drives them with the real DOM, the animation timers and pubsub.
 *
 * A caption line is one sentence of one text topic. An update pairs with its line by `segmentId` (speech pipeline,
 * translation), else by numeric `id` (AI transform; the newest line with it), else - for sources that send neither
 * (typed text, chat, cloud STT) - with the topic's open interim line. Updates with a lower `revision` than the
 * line's, and interims after the line's final, are stale and ignored. A new segment ends the open interim lines of
 * earlier segments of its stream (they were given up upstream, which publishes nothing). Lines that left the screen
 * are remembered for a while, so a late update does not bring them back. An update never re-renders what is on
 * screen and unchanged: the line keeps the longest common prefix of what it shows and the new text (cut back to a
 * grapheme boundary and to whole masks and emotes), and only the rest is animated, or rendered at once when
 * animation is off.
 *
 * An interim line always has the interim look (the line element carries `.interim`): its text may still change, the
 * final may even replace a committed prefix once. `TextEvent.stableLength` additionally marks the uncommitted tail
 * (`.tentative`), so it can be styled apart. A final is fully committed.
 *
 * Remote-safe: this runs on /client and OBS pages too - no Tauri, no window globals, and text is never parsed as
 * HTML (text nodes, `textContent`, and emote URLs set as a property by the DOM adapter).
 */

// --- text helpers ------------------------------------------------------------------------------------------------

/** Length of the longest common prefix of `a` and `b`, in UTF-16 code units. */
export function commonPrefixLength(a: string, b: string): number {
  const n = Math.min(a.length, b.length);
  let i = 0;
  while (i < n && a.charCodeAt(i) === b.charCodeAt(i)) i++;
  return i;
}

interface GraphemeSegmenter {
  segment(input: string): Iterable<{ index: number }>;
}

let graphemes: GraphemeSegmenter | null | undefined;

function graphemeSegmenter(): GraphemeSegmenter | null {
  if (graphemes === undefined) {
    try {
      graphemes = typeof Intl !== "undefined" && typeof Intl.Segmenter === "function"
        ? new Intl.Segmenter(undefined, { granularity: "grapheme" })
        : null;
    } catch {
      graphemes = null;
    }
  }
  return graphemes;
}

function splitsSurrogatePair(text: string, i: number): boolean {
  if (i <= 0 || i >= text.length) return false;
  const hi = text.charCodeAt(i - 1);
  const lo = text.charCodeAt(i);
  return hi >= 0xd800 && hi <= 0xdbff && lo >= 0xdc00 && lo <= 0xdfff;
}

/** Code point boundaries (`b[i] === 1` unless `i` splits a surrogate pair): the fallback without `Intl.Segmenter`. */
export function codePointBoundaries(text: string): Uint8Array {
  const b = new Uint8Array(text.length + 1);
  for (let i = 0; i <= text.length; i++) b[i] = splitsSurrogatePair(text, i) ? 0 : 1;
  return b;
}

/**
 * `b[i] === 1` when `i` is a grapheme cluster boundary of `text` (0 and `text.length` always are), so an animation
 * step or a cut never shows half an emoji or a letter without its accent.
 */
export function graphemeBoundaries(text: string): Uint8Array {
  const segmenter = graphemeSegmenter();
  if (!segmenter) return codePointBoundaries(text);
  const b = new Uint8Array(text.length + 1);
  for (const s of segmenter.segment(text)) b[s.index] = 1;
  b[text.length] = 1;
  return b;
}

/** The last boundary at or before `index`. */
export function floorBoundary(bounds: Uint8Array, index: number): number {
  let i = Math.max(0, Math.min(Math.floor(index), bounds.length - 1));
  while (i > 0 && !bounds[i]) i--;
  return i;
}

/** The first boundary after `index`, but never past `limit` (> `index`). */
export function nextBoundary(bounds: Uint8Array, index: number, limit: number): number {
  let i = index + 1;
  while (i < limit && !bounds[i]) i++;
  return Math.min(i, limit);
}

/** End of the word at `from`: the next space at or after it, or the end of the text. */
export function wordEnd(text: string, from: number): number {
  const at = text.indexOf(" ", from);
  return at === -1 ? text.length : at;
}

// --- masks and emotes --------------------------------------------------------------------------------------------

/** A word with `*` in it was starred out by a profanity filter: the caption shows the element's mask instead. */
const PROFANITY_RE = /[^\s.,?!]*\*+[^\s.,?!]*/g;

/** `[start, end)` of every masked word of `text`. */
export function profanityRanges(text: string): [number, number][] {
  const out: [number, number][] = [];
  for (const m of text.matchAll(PROFANITY_RE)) {
    const at = m.index ?? 0;
    if (m[0].length) out.push([at, at + m[0].length]);
  }
  return out;
}

export type CaptionAtomKind = "mask" | "emote";

/** A piece of a line that is shown whole (never split by a cut or an animation step). */
export interface CaptionAtom {
  readonly kind: CaptionAtomKind;
  readonly start: number;
  readonly end: number;
  /** Emote image URL. */
  readonly url?: string;
}

/**
 * How emotes are placed. `positions` (animated captions): the numeric keys of `TextEvent.emotes`, word starts found
 * by the Twitch emote scanner, each emote covering its word. `names` (static captions): every occurrence of an emote
 * name, longest name first. These are the two behaviours the element always had.
 */
export type EmoteMode = "positions" | "names";

function escapeRegExp(s: string): string {
  return s.replace(/[.*+?^${}()|[\]\\]/g, "\\$&");
}

function emoteRecord(emotes: unknown): Record<string, unknown> {
  return emotes !== null && typeof emotes === "object" ? (emotes as Record<string, unknown>) : {};
}

/** Masks and emotes of `text`, sorted and never overlapping. Masks win over emotes. */
export function layoutAtoms(text: string, emotes: unknown, mode: EmoteMode): CaptionAtom[] {
  const masks = profanityRanges(text);
  const map = emoteRecord(emotes);
  const atoms: CaptionAtom[] = [];

  if (mode === "positions") {
    const starts = new Map<number, string>();
    for (const [key, url] of Object.entries(map)) {
      if (!/^\d+$/.test(key) || typeof url !== "string" || !url) continue;
      const at = Number(key);
      // Only a word start of this very text (a copied event may carry the positions of another text).
      if (at >= text.length || text[at] === " " || (at > 0 && text[at - 1] !== " ")) continue;
      starts.set(at, url);
    }
    const maskEnds = new Map<number, number>(masks);
    const candidates = [...new Set([...maskEnds.keys(), ...starts.keys()])].sort((a, b) => a - b);
    let pos = 0;
    for (const at of candidates) {
      if (at < pos) continue;
      const maskEnd = maskEnds.get(at);
      if (maskEnd !== undefined) {
        atoms.push({ kind: "mask", start: at, end: maskEnd });
        pos = maskEnd;
      } else {
        const end = wordEnd(text, at);
        atoms.push({ kind: "emote", start: at, end, url: starts.get(at) });
        pos = end;
      }
    }
    return atoms;
  }

  const names = Object.keys(map)
    .filter((k) => k.length > 0 && !/^\d+$/.test(k) && typeof map[k] === "string" && !!map[k])
    .sort((a, b) => b.length - a.length);
  const re = names.length ? new RegExp(names.map(escapeRegExp).join("|"), "g") : null;
  const emotesIn = (from: number, to: number) => {
    if (!re || to <= from) return;
    for (const m of text.slice(from, to).matchAll(re)) {
      if (!m[0]) continue;
      const start = from + (m.index ?? 0);
      atoms.push({ kind: "emote", start, end: start + m[0].length, url: map[m[0]] as string });
    }
  };
  let last = 0;
  for (const [start, end] of masks) {
    emotesIn(last, start);
    atoms.push({ kind: "mask", start, end });
    last = end;
  }
  emotesIn(last, text.length);
  return atoms;
}

function hasAtom(list: readonly CaptionAtom[], a: CaptionAtom): boolean {
  return list.some((b) => b.start === a.start && b.end === a.end && b.kind === a.kind && b.url === a.url);
}

/**
 * How much of a shown line survives an update: the longest common prefix of the first `shown` code units of the old
 * text and the new text, cut back so that it ends on a grapheme boundary of the new text and never splits or
 * silently changes a mask or emote of either version.
 */
export function keepLength(
  prev: { text: string; atoms: readonly CaptionAtom[] },
  shown: number,
  next: { text: string; atoms: readonly CaptionAtom[]; bounds: Uint8Array },
): number {
  let keep = floorBoundary(next.bounds, Math.min(shown, commonPrefixLength(prev.text, next.text)));
  for (let changed = true; changed && keep > 0;) {
    changed = false;
    for (const [mine, other] of [[next.atoms, prev.atoms], [prev.atoms, next.atoms]] as const) {
      for (const a of mine) {
        if (a.start >= keep) break;
        if (a.end > keep || !hasAtom(other, a)) {
          keep = floorBoundary(next.bounds, a.start);
          changed = true;
          break;
        }
      }
    }
  }
  return keep;
}

// --- styling -----------------------------------------------------------------------------------------------------

export type CaptionKind = "interim" | "final";

export interface CaptionStyle {
  /** The line is an interim: the line element carries `.interim` (the interim colour), committed prefix included. */
  interim: boolean;
  /** Text from here on is the uncommitted tail, marked `.tentative` (`Infinity`: none). */
  tentativeFrom: number;
}

/**
 * Finals are fully committed. An interim keeps the interim look as a whole (even its committed prefix may still be
 * replaced once, by the final); with a valid `stableLength` the text after it (floored to a grapheme boundary) is
 * also marked tentative. Without `stableLength` nothing is marked: the legacy look.
 */
export function captionStyle(kind: CaptionKind, stableLength: unknown, bounds: Uint8Array): CaptionStyle {
  if (kind === "final") return { interim: false, tentativeFrom: Infinity };
  if (typeof stableLength !== "number" || !Number.isInteger(stableLength) || stableLength < 0)
    return { interim: true, tentativeFrom: Infinity };
  return { interim: true, tentativeFrom: floorBoundary(bounds, Math.min(bounds.length - 1, stableLength)) };
}

const TENTATIVE_TEXT_CLASS = "interim tentative";

function maskClass(tentative: boolean, interim: boolean): string {
  return tentative ? "interim profanity tentative" : interim ? "interim profanity" : "profanity";
}

function emoteClass(tentative: boolean): string {
  return tentative ? TENTATIVE_TEXT_CLASS : "";
}

// --- pairing -----------------------------------------------------------------------------------------------------

/** Identity of a caption line (or of an update) as far as pairing is concerned. */
export interface CaptionKeyed {
  readonly topic: string;
  kind: CaptionKind;
  segmentId?: string;
  id?: number;
  revision?: number;
}

/** The pairing fields of one update (see {@link readCaptionMeta}). */
export type CaptionUpdateMeta = CaptionKeyed;

const SEGMENT_ID_RE = /^[A-Za-z0-9._:-]{1,128}$/;

/** The pairing fields of a text event; malformed values are ignored (peers and old sources may send anything). */
export function readCaptionMeta(
  topic: string,
  kind: CaptionKind,
  event: { segmentId?: unknown; id?: unknown; revision?: unknown },
): CaptionUpdateMeta {
  const meta: CaptionUpdateMeta = { topic, kind };
  if (typeof event.segmentId === "string" && SEGMENT_ID_RE.test(event.segmentId)) meta.segmentId = event.segmentId;
  if (typeof event.id === "number" && Number.isFinite(event.id)) meta.id = event.id;
  if (typeof event.revision === "number" && Number.isInteger(event.revision) && event.revision >= 0)
    meta.revision = event.revision;
  return meta;
}

/** Index of the last line matching `test`, or -1 (no `findLastIndex`: older OBS browser sources lack it). */
function lastIndex<T>(list: readonly T[], test: (item: T) => boolean): number {
  for (let i = list.length - 1; i >= 0; i--) if (test(list[i])) return i;
  return -1;
}

/**
 * Index of the line an update belongs to, or -1 for a new line. Only lines of the same topic match (an element on
 * "Any text source" gets the STT line and its translation under the same `segmentId`):
 * - by `segmentId`; failing that, an update that also has an `id` takes the newest line with that `id` and no
 *   `segmentId`;
 * - by `id` (the AI transform's streamed replies carry only the `id` of their line): the newest line with it. The
 *   transform's `id` only advances on STT finals, so a segment given up without a final shares its `id` with the
 *   next one, and the reply always belongs to the newer line;
 * - an update with neither takes the topic's first interim line that has neither (legacy sources).
 */
export function findCaptionLine(lines: readonly CaptionKeyed[], u: CaptionUpdateMeta): number {
  if (u.segmentId !== undefined) {
    const at = lines.findIndex((l) => l.topic === u.topic && l.segmentId === u.segmentId);
    if (at !== -1 || u.id === undefined) return at;
    return lastIndex(lines, (l) => l.topic === u.topic && l.segmentId === undefined && l.id === u.id);
  }
  if (u.id !== undefined) return lastIndex(lines, (l) => l.topic === u.topic && l.id === u.id);
  return lines.findIndex(
    (l) => l.topic === u.topic && l.kind === "interim" && l.segmentId === undefined && l.id === undefined,
  );
}

/**
 * The stream a segment id belongs to: the part before its first ":" (`mic-<nonce>` of the speech pipeline's
 * `mic-<nonce>:<epoch>:<n>`, `tr.textfield.<nonce>.0` of the translation's generated `tr.textfield.<nonce>.0:<n>`), or
 * undefined when the id has no such prefix.
 */
export function segmentStream(segmentId: string): string | undefined {
  const at = segmentId.indexOf(":");
  return at > 0 ? segmentId.slice(0, at) : undefined;
}

/**
 * Position of a segment within its stream: the numbers after the stream (`[1, 7]` of `mic:1:7`, `[1, 7, 2]` of the
 * translation's `mic:1:7.2`), or null when a part is not a number.
 */
function segmentOrder(segmentId: string): number[] | null {
  const out: number[] = [];
  for (const part of segmentId.slice(segmentId.indexOf(":") + 1).split(/[:.]/)) {
    if (!/^\d{1,15}$/.test(part)) return null;
    out.push(Number(part));
  }
  return out;
}

/** Whether segment `a` comes before `b` in their stream; null when their ids do not say. */
function segmentBefore(a: string, b: string): boolean | null {
  const x = segmentOrder(a);
  const y = segmentOrder(b);
  if (!x || !y) return null;
  for (let i = 0; i < Math.max(x.length, y.length); i++) {
    const p = x[i] ?? -1;
    const q = y[i] ?? -1;
    if (p !== q) return p < q;
  }
  return false;
}

/**
 * Indexes (ascending) of the lines given up by the new line an update opens. A text stream publishes its segments
 * one after another (interims, then at most one final), and a segment given up upstream (dropped or stalled speech, a
 * translation abandoned or failed) publishes nothing more, so a new segment means the open interim lines of earlier
 * segments of the same topic and stream are over; left alone, their last interim would stay on screen until the
 * clear timer. Only lines of earlier segments are given up: a late event of a given-up segment never ends the one
 * being spoken. An interim gives up the lines its ids put before it (or, when the ids do not order, the older lines
 * on screen). A final may be the late final of a given-up segment, so it gives up only lines its ids put strictly
 * before it (a short segment often has no interim at all: its final is the first sign of it). When the update
 * carries an `id` (the AI transform), only lines with the same `id` are given up: a line with another `id` had its
 * final and shows the original while the AI works.
 */
export function supersededCaptionLines(lines: readonly CaptionKeyed[], u: CaptionUpdateMeta): number[] {
  if (u.segmentId === undefined) return [];
  const segmentId = u.segmentId;
  const stream = segmentStream(segmentId);
  if (stream === undefined) return [];
  const out: number[] = [];
  lines.forEach((l, i) => {
    if (l.topic !== u.topic || l.kind !== "interim" || l.segmentId === undefined || l.segmentId === segmentId) return;
    if (segmentStream(l.segmentId) !== stream) return;
    if (u.id !== undefined && l.id !== u.id) return;
    const before = segmentBefore(l.segmentId, segmentId);
    if (u.kind === "final" ? before !== true : before === false) return;
    out.push(i);
  });
  return out;
}

/**
 * A /client page reconnected to its host, which may be a restarted Sigil whose segment ids and transform ids start
 * over (`mic:1:1`, id 0, ...). Returns the indexes (ascending) of the interim lines, which the old connection will
 * never update again, to be removed; the final lines stay on screen but stop pairing (their ids are dropped), so a
 * reused id opens a line of its own instead of being merged into an old one or dropped as stale. After a mere
 * network blip the removed interims simply come back with their next update.
 */
export function detachCaptionLines(lines: readonly CaptionKeyed[]): number[] {
  const out: number[] = [];
  lines.forEach((l, i) => {
    if (l.kind === "interim") {
      out.push(i);
      return;
    }
    delete l.segmentId;
    delete l.id;
    delete l.revision;
  });
  return out;
}

/**
 * A stale update of `line`: an interim after the line's final, or a lower revision than the line already shows.
 * Equal revisions are updates (the AI transform repeats its line's revision on the original and the reply).
 */
export function isStaleUpdate(line: CaptionKeyed, u: CaptionUpdateMeta): boolean {
  if (line.kind === "final" && u.kind === "interim") return true;
  return u.revision !== undefined && line.revision !== undefined && u.revision < line.revision;
}

function retiredKey(topic: string, segmentId: string): string {
  return `${topic}|${segmentId}`; // segment ids never contain "|"
}

/**
 * How long a line that left the screen is remembered. A late update is only plausible for a short while (every
 * producer sends at most one final per line, and the longest interim deadline, the translation's, is 60 s from a
 * request made before the line's final), while ids are not guaranteed unique for a page's lifetime: current hosts put a
 * per-process nonce in their ids, but an older Sigil starts over at `mic:1:1` / transform id 0 after a restart, and
 * linked older hosts publish the same ids on the same topics. Forgetting after this long keeps a reused id from being
 * blocked for good.
 */
export const RETIRED_LINE_TTL_MS = 60_000;

function monotonicNow(): number {
  return typeof performance !== "undefined" && typeof performance.now === "function" ? performance.now() : Date.now();
}

/**
 * Lines that already left the screen (cleared after inactivity, given up, or dropped), by topic and `segmentId`, so
 * a late update does not bring them back: a line that ended final does not return; one that left while interim
 * comes back only for an update that is newer (its final, when that took longer than the clear timer or came after
 * the segment was given up, is still shown). Entries are forgotten after `ttlMs` (see {@link RETIRED_LINE_TTL_MS})
 * and on {@link clear} (the host reconnected, so its ids may start over).
 */
export class RetiredCaptionLines {
  readonly #lines = new Map<string, { kind: CaptionKind; revision?: number; at: number }>();

  constructor(
    private readonly capacity = 256,
    private readonly ttlMs = RETIRED_LINE_TTL_MS,
    private readonly now: () => number = monotonicNow,
  ) { }

  retire(line: CaptionKeyed): void {
    if (line.segmentId === undefined) return;
    const key = retiredKey(line.topic, line.segmentId);
    const at = this.now();
    this.#lines.delete(key);
    this.#lines.set(key, { kind: line.kind, revision: line.revision, at });
    // Oldest first (a retired key is always re-inserted at the end).
    for (const [k, r] of this.#lines) {
      if (this.#lines.size <= this.capacity && at - r.at < this.ttlMs) break;
      this.#lines.delete(k);
    }
  }

  blocks(u: CaptionUpdateMeta): boolean {
    if (u.segmentId === undefined) return false;
    const key = retiredKey(u.topic, u.segmentId);
    const r = this.#lines.get(key);
    if (!r) return false;
    if (this.now() - r.at >= this.ttlMs) {
      this.#lines.delete(key);
      return false;
    }
    if (r.kind === "final") return true;
    if (u.revision === undefined || r.revision === undefined) return false;
    // A repeat of the interim the line left with is stale too; its final may repeat the revision (the AI transform
    // shows the STT final as an interim until the reply comes, with the same revision).
    return u.kind === "interim" ? u.revision <= r.revision : u.revision < r.revision;
  }

  /** The line is on screen again. */
  revive(u: CaptionUpdateMeta): void {
    if (u.segmentId !== undefined) this.#lines.delete(retiredKey(u.topic, u.segmentId));
  }

  /** Forgets every retired line (the /client page reconnected: the host may have restarted, reusing its ids). */
  clear(): void {
    this.#lines.clear();
  }

  get size(): number {
    return this.#lines.size;
  }
}

// --- line view ---------------------------------------------------------------------------------------------------

/**
 * The few DOM operations a caption line needs, so the line logic runs against a fake in tests. `N` is the node type.
 * Implementations must never parse text as HTML.
 */
export interface CaptionDom<N> {
  /** A text node. */
  text(data: string): N;
  /** A span with a class and plain-text content. */
  span(className: string, data: string): N;
  /** An emote image (the URL is set as a property). */
  image(className: string, url: string): N;
  /** Replaces the text of a text node or span. */
  setData(node: N, data: string): void;
  /** Sets the class of an element ("" removes it). */
  setClass(node: N, className: string): void;
  append(parent: N, node: N): void;
  remove(node: N): void;
  /** Puts `nodes` where `oldNode` is. */
  replace(oldNode: N, ...nodes: N[]): void;
}

interface Unit<N> {
  kind: "text" | "mask" | "emote";
  start: number;
  end: number;
  tentative: boolean;
  node: N;
  /** Text shown by a text unit (always `text.slice(start, end)`), or the mask text shown so far. */
  data: string;
  /** A mask still being revealed character by character. */
  partial?: boolean;
}

/** How much one step renders: a grapheme, a word (spaces one at a time), or everything up to the next mask/emote. */
export type CaptionStepMode = "char" | "word" | "all";

export interface CaptionUpdate {
  kind: CaptionKind;
  text: string;
  emotes?: unknown;
  stableLength?: unknown;
  mode: EmoteMode;
  meta: CaptionUpdateMeta;
}

/**
 * One caption line on screen. `update` applies a new version of the text (keeping what still matches), `step`
 * renders the next piece, `flush` renders the rest at once and `finish` adds the space that separates it from the
 * next line.
 */
export class CaptionLine<N> implements CaptionKeyed {
  readonly topic: string;
  kind: CaptionKind;
  segmentId?: string;
  id?: number;
  revision?: number;

  /** The line's current text. */
  text = "";
  /** Code units of `text` on screen. */
  cursor = 0;

  #atoms: CaptionAtom[] = [];
  #atomAt = new Map<number, CaptionAtom>();
  #bounds: Uint8Array = new Uint8Array([1]);
  #maskShown = 0;
  #units: Unit<N>[] = [];
  #trailing: N | null = null;
  #interim = false;
  #tentativeFrom = Infinity;

  constructor(private readonly dom: CaptionDom<N>, readonly element: N, meta: CaptionUpdateMeta) {
    this.topic = meta.topic;
    this.kind = meta.kind;
  }

  /** Text left to render. */
  get pending(): boolean {
    return this.cursor < this.text.length;
  }

  /** Applies a new version of the line; returns how many code units of what was on screen were kept. */
  update(u: CaptionUpdate): number {
    const text = u.text;
    const bounds = graphemeBoundaries(text);
    const atoms = layoutAtoms(text, u.emotes, u.mode);
    const atomAt = new Map(atoms.map((a) => [a.start, a] as const));
    const keep = keepLength({ text: this.text, atoms: this.#atoms }, this.cursor, { text, atoms, bounds });

    if (keep < this.cursor) {
      this.cursor = this.#truncate(keep);
      this.#maskShown = 0;
    } else if (this.#maskShown > 0) {
      const mask = atomAt.get(this.cursor);
      const last = this.#units[this.#units.length - 1];
      if (mask?.kind === "mask" && last?.partial && last.start === this.cursor) {
        last.end = mask.end;
      } else {
        this.cursor = this.#truncate(this.cursor);
        this.#maskShown = 0;
      }
    }

    this.text = text;
    this.#bounds = bounds;
    this.#atoms = atoms;
    this.#atomAt = atomAt;
    this.kind = u.kind;
    if (u.meta.segmentId !== undefined) this.segmentId = u.meta.segmentId;
    if (u.meta.id !== undefined && this.id === undefined) this.id = u.meta.id;
    if (u.meta.revision !== undefined) this.revision = Math.max(this.revision ?? 0, u.meta.revision);
    this.#restyle(captionStyle(u.kind, u.stableLength, bounds));
    if (this.pending) this.#dropTrailing();
    return keep;
  }

  /**
   * Renders the next piece of the line. `mask` is the element's profanity mask: shown whole, or one character per
   * step in `char` mode.
   */
  step(mask: string, mode: CaptionStepMode): boolean {
    if (!this.pending) return false;
    this.#dropTrailing();
    const at = this.cursor;
    const atom = this.#atomAt.get(at);
    if (atom?.kind === "mask") {
      const chars = Array.from(mask);
      if (mode !== "char" || chars.length === 0) {
        this.#putMask(atom, mask, false);
        this.cursor = atom.end;
        this.#maskShown = 0;
      } else {
        this.#maskShown++;
        const done = this.#maskShown >= chars.length;
        this.#putMask(atom, chars.slice(0, this.#maskShown).join(""), !done);
        if (done) {
          this.cursor = atom.end;
          this.#maskShown = 0;
        }
      }
    } else if (atom?.kind === "emote") {
      this.#putEmote(atom);
      this.cursor = atom.end;
    } else {
      const limit = this.#nextAtomStart(at);
      let end: number;
      if (mode === "all") end = limit;
      else if (mode === "word" && this.text[at] !== " ") end = Math.min(wordEnd(this.text, at), limit);
      else end = nextBoundary(this.#bounds, at, limit);
      this.#appendText(at, end);
      this.cursor = end;
    }
    return true;
  }

  /** Renders the rest of the line at once. */
  flush(mask: string): void {
    while (this.step(mask, "all"));
  }

  /** The line is fully shown: adds the space that separates it from the next line. */
  finish(): void {
    if (this.pending || this.#trailing !== null) return;
    this.#trailing = this.dom.text(" ");
    this.dom.append(this.element, this.#trailing);
  }

  #nextAtomStart(from: number): number {
    for (const a of this.#atoms) if (a.start >= from) return a.start;
    return this.text.length;
  }

  #dropTrailing(): void {
    if (this.#trailing === null) return;
    this.dom.remove(this.#trailing);
    this.#trailing = null;
  }

  /** Removes what is shown from `keep` on; returns the new shown length. */
  #truncate(keep: number): number {
    this.#dropTrailing();
    while (this.#units.length) {
      const u = this.#units[this.#units.length - 1];
      if (u.start < keep && u.end <= keep && !u.partial) return u.end;
      if (u.start < keep && u.kind === "text") {
        u.data = u.data.slice(0, keep - u.start);
        u.end = keep;
        this.dom.setData(u.node, u.data);
        return keep;
      }
      // At or after the cut, or a mask / emote across it (keepLength never cuts one; drop it whole to be safe).
      this.dom.remove(u.node);
      this.#units.pop();
    }
    return 0;
  }

  #textUnit(start: number, end: number, tentative: boolean, data: string): Unit<N> {
    const node = tentative ? this.dom.span(TENTATIVE_TEXT_CLASS, data) : this.dom.text(data);
    return { kind: "text", start, end, tentative, node, data };
  }

  #appendText(start: number, end: number): void {
    if (end <= start) return;
    const from = this.#tentativeFrom;
    if (start < from && end > from) {
      this.#appendText(start, from);
      this.#appendText(from, end);
      return;
    }
    const tentative = start >= from;
    const data = this.text.slice(start, end);
    const last = this.#units[this.#units.length - 1];
    if (last && last.kind === "text" && last.tentative === tentative && last.end === start) {
      last.data += data;
      last.end = end;
      this.dom.setData(last.node, last.data);
      return;
    }
    const unit = this.#textUnit(start, end, tentative, data);
    this.dom.append(this.element, unit.node);
    this.#units.push(unit);
  }

  #putMask(atom: CaptionAtom, shown: string, partial: boolean): void {
    const last = this.#units[this.#units.length - 1];
    if (last && last.kind === "mask" && last.partial && last.start === atom.start) {
      last.data = shown;
      last.end = atom.end;
      last.partial = partial;
      this.dom.setData(last.node, shown);
      return;
    }
    const tentative = atom.end > this.#tentativeFrom;
    const node = this.dom.span(maskClass(tentative, this.#interim), shown);
    this.dom.append(this.element, node);
    this.#units.push({ kind: "mask", start: atom.start, end: atom.end, tentative, node, data: shown, partial });
  }

  #putEmote(atom: CaptionAtom): void {
    const tentative = atom.end > this.#tentativeFrom;
    const node = this.dom.image(emoteClass(tentative), atom.url ?? "");
    this.dom.append(this.element, node);
    this.#units.push({ kind: "emote", start: atom.start, end: atom.end, tentative, node, data: "" });
  }

  /** Applies the committed / tentative split to what is shown; only the nodes whose look changes are touched. */
  #restyle(style: CaptionStyle): void {
    this.dom.setClass(this.element, style.interim ? "interim" : "");
    if (style.interim === this.#interim && style.tentativeFrom === this.#tentativeFrom) return;
    this.#interim = style.interim;
    const from = (this.#tentativeFrom = style.tentativeFrom);

    const units: Unit<N>[] = [];
    for (const u of this.#units) {
      if (u.kind !== "text") {
        u.tentative = u.end > from;
        this.dom.setClass(u.node, u.kind === "mask" ? maskClass(u.tentative, this.#interim) : emoteClass(u.tentative));
        units.push(u);
      } else if (u.start < from && u.end > from) {
        const committed = this.#textUnit(u.start, from, false, u.data.slice(0, from - u.start));
        const tentative = this.#textUnit(from, u.end, true, u.data.slice(from - u.start));
        this.dom.replace(u.node, committed.node, tentative.node);
        units.push(committed, tentative);
      } else if ((u.start >= from) !== u.tentative) {
        const moved = this.#textUnit(u.start, u.end, u.start >= from, u.data);
        this.dom.replace(u.node, moved.node);
        units.push(moved);
      } else {
        units.push(u);
      }
    }

    // Neighbouring text of the same look becomes one node again.
    this.#units = [];
    for (const u of units) {
      const prev = this.#units[this.#units.length - 1];
      if (prev && prev.kind === "text" && u.kind === "text" && prev.tentative === u.tentative && prev.end === u.start) {
        prev.data += u.data;
        prev.end = u.end;
        this.dom.setData(prev.node, prev.data);
        this.dom.remove(u.node);
      } else {
        this.#units.push(u);
      }
    }
  }
}
