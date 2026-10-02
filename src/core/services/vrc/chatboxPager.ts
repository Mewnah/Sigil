/**
 * VRChat chatbox pacing and pagination (P6-C).
 *
 * VRChat shows at most {@link CHATBOX_MAX_CHARS} characters per chatbox message and drops messages that arrive less
 * than about 1.3 s after the previous one. {@link ChatboxPager} queues every `/chatbox/input` send so none is dropped:
 * a message goes out at once when the chatbox is idle, later ones wait for the interval (never below
 * {@link CHATBOX_MIN_INTERVAL_MS}), and a short message still waiting is merged into the one queued before it (one
 * message read together instead of two flashing past).
 *
 * Long text is either split into numbered pages (`paginate`, {@link paginateChatbox}: breaks at sentence ends, then
 * clause punctuation, then spaces, never inside a character) or, as before P6-C, cut to one message
 * ({@link truncateChatbox}, now at a word boundary with an ellipsis). Lengths are UTF-16 code units, the unit of
 * VRChat's limit.
 *
 * Pure logic: no `window`, no Tauri. The textbox target (`targets/textbox.ts`) wires it to OSC.
 */
import { joinFragments } from "@/core/services/translation/tracker";

/** VRChat chatbox text limit (UTF-16 code units). */
export const CHATBOX_MAX_CHARS = 144;
/** VRChat drops chatbox messages sent closer together than about this. */
export const CHATBOX_MIN_INTERVAL_MS = 1_300;
/** Default time between chatbox messages / pages. */
export const CHATBOX_DEFAULT_INTERVAL_MS = 1_500;
/** Longest configurable time between chatbox messages. */
export const CHATBOX_MAX_INTERVAL_MS = 10_000;
/** Pages waiting at most (about a minute at the default interval); older ones are dropped beyond that. */
export const CHATBOX_MAX_QUEUE = 40;

/** Clamps a configured interval (string or number) to the safe range. */
export function chatboxIntervalMs(raw: unknown): number {
  const n = typeof raw === "number" ? raw : parseInt(String(raw ?? ""), 10);
  if (!Number.isFinite(n)) return CHATBOX_DEFAULT_INTERVAL_MS;
  return Math.min(CHATBOX_MAX_INTERVAL_MS, Math.max(CHATBOX_MIN_INTERVAL_MS, Math.round(n)));
}

let segmenter: Intl.Segmenter | null | undefined;

/** User-perceived characters of `text` (grapheme clusters; code points without `Intl.Segmenter`). */
export function graphemes(text: string): string[] {
  if (segmenter === undefined) {
    segmenter =
      typeof Intl !== "undefined" && typeof Intl.Segmenter === "function"
        ? new Intl.Segmenter(undefined, { granularity: "grapheme" })
        : null;
  }
  if (segmenter) return Array.from(segmenter.segment(text), (s) => s.segment);
  return Array.from(text);
}

/** Collapses spaces and tabs, drops control characters (keeps line breaks) and trims. */
export function normalizeChatboxText(text: string): string {
  return (
    text
      // eslint-disable-next-line no-control-regex
      .replace(/[\u0000-\u0009\u000B-\u001F\u007F]/g, " ")
      .replace(/\r\n?/g, "\n")
      .replace(/[ \t ]+/g, " ")
      .replace(/ ?\n ?/g, "\n")
      .trim()
  );
}

const SENTENCE_END = new Set([".", "!", "?", "…", "‼", "⁇", "⁈", "⁉"]);
const CJK_SENTENCE_END = new Set(["。", "！", "？", "｡", "︒"]);
const CLOSERS = new Set(["」", "』", "）", ")", "\"", "”", "’", "'", "】", "》", "〉"]);
const CLAUSE = new Set([",", ";", ":", "–", "—"]);
const CJK_CLAUSE = new Set(["、", "，", "；", "：", "､"]);

function isSpace(g: string | undefined): boolean {
  return g !== undefined && /^\s+$/u.test(g);
}

/** Whether a page may end after `gs[i - 1]` because a sentence ends there. */
function sentenceBreak(gs: string[], i: number): boolean {
  let j = i - 1;
  while (j > 0 && CLOSERS.has(gs[j])) j--;
  const end = gs[j];
  const next = gs[i];
  if (CJK_SENTENCE_END.has(end)) return true;
  return SENTENCE_END.has(end) && (next === undefined || isSpace(next));
}

/** Whether a page may end after `gs[i - 1]` at clause punctuation. */
function clauseBreak(gs: string[], i: number): boolean {
  const prev = gs[i - 1];
  if (CJK_CLAUSE.has(prev)) return true;
  return CLAUSE.has(prev) && (gs[i] === undefined || isSpace(gs[i]));
}

/** Whether a page may end before the space `gs[i]` (or after one). */
function spaceBreak(gs: string[], i: number): boolean {
  return isSpace(gs[i]) || isSpace(gs[i - 1]);
}

const BREAKS = [sentenceBreak, clauseBreak, spaceBreak];

/**
 * Where to end a page whose first `fit` graphemes fit: the last sentence end in the second half of that window, else
 * the last clause break, else the last space, else right at the limit (CJK text needs no space to break).
 */
function bestCut(gs: string[], fit: number): number {
  const minCut = Math.max(1, Math.ceil(fit / 2));
  for (const test of BREAKS) {
    for (let i = fit; i >= minCut; i--) {
      if (test(gs, i)) return i;
    }
  }
  return fit;
}

/** Splits normalized `text` into pages of at most `budget` UTF-16 units each. */
function splitPages(text: string, budget: number): string[] {
  const pages: string[] = [];
  let rest = text;
  while (rest.length > budget) {
    const gs = graphemes(rest);
    let units = 0;
    let fit = 0;
    while (fit < gs.length && units + gs[fit].length <= budget) {
      units += gs[fit].length;
      fit++;
    }
    // A single character longer than the budget (pathological emoji sequence): it gets a page of its own.
    const cut = fit === 0 ? 1 : bestCut(gs, fit);
    const page = gs.slice(0, cut).join("").trim();
    rest = gs.slice(cut).join("").trim();
    if (page) pages.push(page);
  }
  if (rest) pages.push(rest);
  return pages;
}

function pageMarker(page: number, count: number): string {
  return ` (${page}/${count})`;
}

/**
 * Text as chatbox pages of at most `max` UTF-16 units, each ending with a ` (n/total)` marker when there is more than
 * one. Empty text yields no page.
 */
export function paginateChatbox(text: string, max = CHATBOX_MAX_CHARS): string[] {
  const clean = normalizeChatboxText(text);
  if (!clean) return [];
  if (clean.length <= max) return [clean];
  let reserved = pageMarker(2, 2).length;
  // Markers only when they leave room for text (always, at VRChat's 144).
  if (max - reserved < 16) return splitPages(clean, max);
  for (;;) {
    const pages = splitPages(clean, max - reserved);
    const needed = pageMarker(pages.length, pages.length).length;
    if (needed <= reserved) return pages.map((p, i) => p + pageMarker(i + 1, pages.length));
    reserved = needed;
  }
}

/**
 * Text cut to one chatbox message (`paginate` off): at most `max` UTF-16 units, at a word boundary near the end when
 * there is one, with an ellipsis showing it was cut. Never splits a character.
 */
export function truncateChatbox(text: string, max = CHATBOX_MAX_CHARS): string {
  const clean = normalizeChatboxText(text);
  if (clean.length <= max) return clean;
  const gs = graphemes(clean);
  const budget = Math.max(1, max - 1);
  let units = 0;
  let fit = 0;
  while (fit < gs.length && units + gs[fit].length <= budget) {
    units += gs[fit].length;
    fit++;
  }
  let cut = fit;
  const minCut = Math.ceil(fit * 0.7);
  for (let i = fit; i >= minCut; i--) {
    if (isSpace(gs[i])) {
      cut = i;
      break;
    }
  }
  return `${gs.slice(0, Math.max(1, cut)).join("").trimEnd()}…`;
}

export interface ChatboxClock {
  /** Milliseconds on a monotonic clock (never steps back when the system time is corrected). */
  now(): number;
  setTimeout(fn: () => void, ms: number): unknown;
  clearTimeout(handle: unknown): void;
}

/**
 * Monotonic milliseconds (`performance.now()`; `Date.now()` only where it is missing). Pacing must not follow the wall
 * clock: when Windows time sync sets it back by ten minutes, every chatbox message would otherwise wait ten minutes.
 */
export function monotonicNow(): number {
  return typeof performance !== "undefined" && typeof performance.now === "function" ? performance.now() : Date.now();
}

const systemClock: ChatboxClock = {
  now: monotonicNow,
  setTimeout: (fn, ms) => setTimeout(fn, ms),
  clearTimeout: (h) => clearTimeout(h as ReturnType<typeof setTimeout>),
};

export interface ChatboxPagerOptions {
  /** Sends one chatbox message (`/chatbox/input`); `notify` = play VRChat's notification sound. */
  send(text: string, notify: boolean): void;
  /** Split long text into pages (else cut it to one message). */
  paginate(): boolean;
  /** Time between two chatbox messages (clamped with {@link chatboxIntervalMs}). */
  intervalMs(): number;
  /** Play VRChat's notification sound for new messages (later pages of a message are always silent). */
  notify(): boolean;
  clock?: ChatboxClock;
  /** Pages waiting at most (default {@link CHATBOX_MAX_QUEUE}). */
  maxQueue?: number;
  /** The backlog was over `maxQueue`: this many of the oldest waiting pages were dropped. */
  onDrop?(count: number): void;
  /** A message was cut to one chatbox message (`paginate` off and the text was too long). */
  onTruncate?(): void;
}

interface QueuedPage {
  text: string;
  notify: boolean;
  /** A whole message on one page: a later one-page message may be merged into it. */
  mergeable: boolean;
}

/** Paced chatbox sender: see the module comment. */
export class ChatboxPager {
  readonly #options: ChatboxPagerOptions;
  readonly #clock: ChatboxClock;
  #queue: QueuedPage[] = [];
  #lastSentAt = Number.NEGATIVE_INFINITY;
  #timer: unknown = null;

  constructor(options: ChatboxPagerOptions) {
    this.#options = options;
    this.#clock = options.clock ?? systemClock;
  }

  /** Pages waiting to be sent. */
  get pending(): number {
    return this.#queue.length;
  }

  /** Queues one message (split or cut as configured). */
  push(text: string): void {
    const pages = this.#options.paginate() ? paginateChatbox(text) : this.#truncated(text);
    if (pages.length === 0) return;
    const last = this.#queue.at(-1);
    if (pages.length === 1 && last?.mergeable) {
      const merged = joinFragments(last.text, pages[0]);
      if (merged.length <= CHATBOX_MAX_CHARS) {
        last.text = merged;
        this.#pump();
        return;
      }
    }
    const notify = this.#options.notify();
    pages.forEach((text, i) => this.#queue.push({ text, notify: notify && i === 0, mergeable: pages.length === 1 }));
    const max = Math.max(1, this.#options.maxQueue ?? CHATBOX_MAX_QUEUE);
    if (this.#queue.length > max) {
      const dropped = this.#queue.length - max;
      this.#queue.splice(0, dropped);
      this.#options.onDrop?.(dropped);
    }
    this.#pump();
  }

  /** Drops everything waiting (the pacing of the last message sent still applies). */
  cancel(): void {
    this.#queue = [];
    if (this.#timer !== null) {
      this.#clock.clearTimeout(this.#timer);
      this.#timer = null;
    }
  }

  #truncated(text: string): string[] {
    const cut = truncateChatbox(text);
    if (cut === "") return [];
    if (cut.length < normalizeChatboxText(text).length) this.#options.onTruncate?.();
    return [cut];
  }

  #pump(): void {
    if (this.#timer !== null || this.#queue.length === 0) return;
    const now = this.#clock.now();
    const interval = chatboxIntervalMs(this.#options.intervalMs());
    const wait = this.#lastSentAt + interval - now;
    if (wait > 0) {
      this.#timer = this.#clock.setTimeout(() => {
        this.#timer = null;
        this.#pump();
      }, wait);
      return;
    }
    const page = this.#queue.shift();
    if (!page) return;
    this.#lastSentAt = now;
    this.#options.send(page.text, page.notify);
    if (this.#queue.length > 0) {
      this.#timer = this.#clock.setTimeout(() => {
        this.#timer = null;
        this.#pump();
      }, interval);
    }
  }
}
