/**
 * Utterance tracker: turns one input stream's `TextEvent`s (STT or typed text) into translation LINES, the unit the
 * sequencer and LocalAgreement work on.
 *
 * - Events with a `segmentId` (the speech pipeline's `<pipelineId>-<nonce>:<epoch>:<segmentId>`): interims of a segment
 *   that already had its final, and interims whose `revision` is not above the last one seen, are dropped. A new
 *   segment starting while the open line's partial never got a final means that segment was given up (the
 *   speech pipeline abandons dropped segments without publishing anything): the line is abandoned. A late final
 *   of a given-up segment is still translated (finals are never dropped), as a line of its own.
 * - Events without ids (cloud STT, peers, typed text): an interim that repeats the start of the final just closed
 *   is a late duplicate and is dropped; anything else follows the usual interims-then-final order.
 * - Smart Turn (`turnEndCommit`): a final with `turnEnd: false` (the speaker has not finished) is held and merged
 *   with the following segment into one line, shown meanwhile as an interim; the line is committed by a final with
 *   `turnEnd` true or unknown, after `holdMs` without any new event, when it grows past `maxHeldChars`, or when
 *   STT's cancel marker closes the open sentence. Finals without a verdict (Smart Turn off) commit at once.
 *
 * Owned by P4-C.
 */
import { TextEvent, TextEventType } from "@/types";
import { isUnspaced } from "./languages";
import { isCancelMarker, normalizeForMatch } from "./sequencer";

export interface TrackerClock {
  now(): number;
  setTimeout(fn: () => void, ms: number): unknown;
  clearTimeout(handle: unknown): void;
}

const systemClock: TrackerClock = {
  now: () => Date.now(),
  setTimeout: (fn, ms) => setTimeout(fn, ms),
  clearTimeout: (h) => clearTimeout(h as ReturnType<typeof setTimeout>),
};

/** One translation line. */
export interface TrackedLine {
  /** Id published on the line's translation events (`TextEvent.segmentId`): its first source segment id, or generated. */
  readonly id: string;
  /** Source segment ids merged into this line, in order. */
  readonly segments: string[];
  /** Language of the latest source text of the line (`TextEvent.lang`), when known. */
  lang?: string;
  /** Smart Turn verdict of the line's final. */
  turnEnd?: boolean;
}

export interface LineEvent {
  /** Source text of the whole line so far. */
  value: string;
  line: TrackedLine;
}

export interface UtteranceTrackerSink {
  interim(ev: LineEvent): void;
  final(ev: LineEvent): void;
  /** STT's cancel marker closed the open line, and none of it was final (`value` is the marker). */
  cancel(ev: LineEvent): void;
  /** The open line ended without a final (its segment was given up): nothing more comes for it. */
  abandon(line: TrackedLine): void;
}

export interface UtteranceTrackerOptions {
  sink: UtteranceTrackerSink;
  /** Prefix of generated line ids (`[A-Za-z0-9._:-]`), e.g. `tr.stt.3`. */
  idPrefix: string;
  /** Hold finals of an unfinished turn (`turnEnd: false`) and merge them with the next segment. Default off. */
  turnEndCommit?: () => boolean;
  /** Longest wait for the turn to end after the last event of a held line (ms). */
  holdMs?: () => number;
  /** A held line longer than this (characters) is committed at once. */
  maxHeldChars?: number;
  /** Late-interim window for events without segment ids (ms; 0 = off). */
  staleInterimMs?: number;
  clock?: TrackerClock;
}

/** Default wait for the end of a turn Smart Turn said was not finished. */
export const TURN_HOLD_MS = 2_500;
/** Default length at which a held line is committed anyway (a long caption is hard to read and to speak). */
export const MAX_HELD_CHARS = 280;
/** Segment ids remembered to recognise late and repeated events. */
const MEMORY = 512;

/** Joins two sentence fragments of one line: no space between fragments of scripts written without spaces. */
export function joinFragments(a: string, b: string, lang = ""): string {
  const left = a.trimEnd();
  const right = b.trimStart();
  if (!left) return right;
  if (!right) return left;
  if (lang && isUnspaced(lang)) return left + right;
  const unspacedEdge = /[\p{Script=Han}\p{Script=Hiragana}\p{Script=Katakana}\p{Script=Thai}　-〿＀-￯]/u;
  const lastLeft = [...left].pop() ?? "";
  const firstRight = [...right][0] ?? "";
  if (unspacedEdge.test(lastLeft) && unspacedEdge.test(firstRight)) return left + right;
  return `${left} ${right}`;
}

interface OpenLine {
  line: TrackedLine;
  /** Final texts already merged into the line (held for the end of the turn). */
  held: { id: string | undefined; text: string }[];
  /** Partial of the segment in progress. */
  current: { id: string | undefined; text: string } | null;
  timer: unknown;
}

function remember<K, V>(map: Map<K, V>, key: K, value: V): void {
  map.delete(key);
  map.set(key, value);
  if (map.size > MEMORY) {
    const oldest = map.keys().next().value;
    if (oldest !== undefined) map.delete(oldest);
  }
}

export class UtteranceTracker {
  readonly #options: UtteranceTrackerOptions;
  readonly #clock: TrackerClock;
  #open: OpenLine | null = null;
  #counter = 0;
  /** Finalized segment id → its final text. */
  readonly #closed = new Map<string, string>();
  /** Highest revision seen per segment id. */
  readonly #revisions = new Map<string, number>();
  /** Segments given up without a final (their late final still gets translated). */
  readonly #abandoned = new Map<string, true>();
  readonly #usedIds = new Map<string, true>();
  #lastClosed: { text: string; at: number } | null = null;

  constructor(options: UtteranceTrackerOptions) {
    this.#options = options;
    this.#clock = options.clock ?? systemClock;
  }

  /** The line currently open (interims shown or finals held), if any. */
  get openLine(): TrackedLine | null {
    return this.#open?.line ?? null;
  }

  push(e: TextEvent): void {
    const text = (e.value ?? "").trim();
    if (!text) return;
    if (e.type === TextEventType.final) this.#final(e, text);
    else this.#interim(e, text);
  }

  /** Forgets everything (stop / restart); a held line is dropped without output. */
  reset(): void {
    if (this.#open) this.#clock.clearTimeout(this.#open.timer);
    this.#open = null;
    this.#closed.clear();
    this.#revisions.clear();
    this.#abandoned.clear();
    this.#lastClosed = null;
  }

  // --- events ---------------------------------------------------------------------------------------------------

  #interim(e: TextEvent, text: string): void {
    const sid = e.segmentId;
    if (sid) {
      if (this.#closed.has(sid) || this.#abandoned.has(sid)) return;
      const last = this.#revisions.get(sid);
      if (e.revision !== undefined) {
        if (last !== undefined && e.revision <= last) return;
        remember(this.#revisions, sid, e.revision);
      }
    } else if (!this.#open && this.#isLateInterim(text)) {
      return;
    }

    let open = this.#open;
    if (open?.current && sid && open.current.id && open.current.id !== sid) {
      this.#abandonCurrent(open);
      open = this.#open;
    }
    open ??= this.#openLine(sid);
    open.current = { id: sid, text };
    if (e.lang) open.line.lang = e.lang;
    if (open.held.length > 0) this.#armHold(open);
    this.#options.sink.interim({ value: this.#text(open), line: open.line });
  }

  #final(e: TextEvent, text: string): void {
    if (isCancelMarker(text)) {
      this.#cancel(e);
      return;
    }
    const sid = e.segmentId;
    if (sid) {
      const closed = this.#closed.get(sid);
      if (closed !== undefined) {
        if (normalizeForMatch(closed) === normalizeForMatch(text)) return; // a repeat
        const heldAt = this.#open?.held.findIndex((h) => h.id === sid) ?? -1;
        if (this.#open && heldAt >= 0) {
          // A revision of a final still held for the end of the turn: replace its text.
          this.#open.held[heldAt] = { id: sid, text };
          remember(this.#closed, sid, text);
          this.#armHold(this.#open);
          this.#options.sink.interim({ value: this.#text(this.#open), line: this.#open.line });
          return;
        }
        // A different final for a segment already translated (not expected from the speech pipeline): translate
        // it as a line of its own rather than lose it.
      }
      if (this.#abandoned.has(sid)) {
        this.#abandoned.delete(sid);
        const open = this.#open;
        if (open?.current && open.current.id !== sid) {
          // A late final of a given-up segment while a later one is being spoken: it joins the open line ahead of
          // the partial, so it is translated with it instead of cutting that sentence off.
          open.held.push({ id: sid, text });
          remember(this.#closed, sid, text);
          if (!open.line.segments.includes(sid)) open.line.segments.push(sid);
          // A line holding a final always has a hold timer: if the partial's segment is given up upstream (nothing
          // more comes for it), the final still goes out after `holdMs` instead of waiting for the next speech.
          this.#armHold(open);
          this.#options.sink.interim({ value: this.#text(open), line: open.line });
          return;
        }
        // It was spoken before anything held, so the held line goes out first and this final becomes its own line.
        if (open && open.held.length > 0) this.#commit(open);
      }
    }

    let open = this.#open;
    if (open?.current && sid && open.current.id && open.current.id !== sid) {
      this.#abandonCurrent(open);
      open = this.#open;
    }
    open ??= this.#openLine(sid);
    open.current = null;
    open.held.push({ id: sid, text });
    if (sid) {
      remember(this.#closed, sid, text);
      if (!open.line.segments.includes(sid)) open.line.segments.push(sid);
    }
    if (e.lang) open.line.lang = e.lang;

    const hold =
      e.turnEnd === false &&
      (this.#options.turnEndCommit?.() ?? false) &&
      this.#holdMs() > 0 &&
      this.#text(open).length < (this.#options.maxHeldChars ?? MAX_HELD_CHARS);
    if (hold) {
      open.line.turnEnd = false;
      this.#armHold(open);
      this.#options.sink.interim({ value: this.#text(open), line: open.line });
      return;
    }
    open.line.turnEnd = e.turnEnd;
    this.#commit(open);
  }

  #cancel(e: TextEvent): void {
    const open = this.#open;
    if (!open) return;
    if (e.segmentId) remember(this.#closed, e.segmentId, "");
    else if (open.current?.id) remember(this.#closed, open.current.id, "");
    open.current = null;
    if (open.held.length > 0) {
      // Only the unfinished tail was cancelled: what was final is committed.
      this.#commit(open);
      return;
    }
    this.#clock.clearTimeout(open.timer);
    this.#open = null;
    this.#lastClosed = null;
    this.#options.sink.cancel({ value: e.value.trim(), line: open.line });
  }

  // --- internals ------------------------------------------------------------------------------------------------

  #holdMs(): number {
    return this.#options.holdMs?.() ?? TURN_HOLD_MS;
  }

  #openLine(sid: string | undefined): OpenLine {
    let id = sid ?? `${this.#options.idPrefix}:${++this.#counter}`;
    if (this.#usedIds.has(id)) id = `${id}.${++this.#counter}`;
    remember(this.#usedIds, id, true);
    const open: OpenLine = { line: { id, segments: sid ? [sid] : [] }, held: [], current: null, timer: null };
    this.#open = open;
    return open;
  }

  #text(open: OpenLine): string {
    const lang = open.line.lang ?? "";
    let out = "";
    for (const h of open.held) out = joinFragments(out, h.text, lang);
    if (open.current) out = joinFragments(out, open.current.text, lang);
    return out;
  }

  /** (Re)starts the wait for the end of the turn. */
  #armHold(open: OpenLine): void {
    this.#clock.clearTimeout(open.timer);
    open.timer = this.#clock.setTimeout(() => {
      if (this.#open !== open) return;
      open.timer = null;
      this.#commit(open);
    }, this.#holdMs());
  }

  /**
   * Commits the held finals of `open` as the line's final. A partial of the next segment in progress (the hold
   * expired while it was being spoken) continues as a new line.
   */
  #commit(open: OpenLine): void {
    this.#clock.clearTimeout(open.timer);
    open.timer = null;
    if (this.#open === open) this.#open = null;
    const current = open.current;
    open.current = null;
    const value = this.#text(open);
    if (value) {
      this.#lastClosed = { text: value, at: this.#clock.now() };
      this.#options.sink.final({ value, line: open.line });
    }
    if (current) {
      const next = this.#openLine(current.id);
      next.current = current;
      if (open.line.lang) next.line.lang = open.line.lang;
      this.#options.sink.interim({ value: this.#text(next), line: next.line });
    }
  }

  /** The open line's partial segment was given up (a different segment is speaking now). */
  #abandonCurrent(open: OpenLine): void {
    const cur = open.current;
    open.current = null;
    if (cur?.id) remember(this.#abandoned, cur.id, true);
    if (open.held.length > 0) return; // the held finals still wait for the end of the turn
    this.#clock.clearTimeout(open.timer);
    if (this.#open === open) this.#open = null;
    this.#options.sink.abandon(open.line);
  }

  #isLateInterim(text: string): boolean {
    const last = this.#lastClosed;
    const window = this.#options.staleInterimMs ?? 1_500;
    if (!last || window <= 0 || this.#clock.now() - last.at > window) return false;
    const t = normalizeForMatch(text);
    return t.length > 0 && normalizeForMatch(last.text).startsWith(t);
  }
}
