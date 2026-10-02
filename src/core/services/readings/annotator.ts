import {
  ReadingsEventSchema,
  TextEventType,
  type CaptionEventSource,
  type ReadingsEvent,
  type RustReadingsRequest,
  type RustReadingsResult,
  type TextEvent,
} from "@/types";
import type { ReadingsState } from "./schema";

/**
 * Pure part of the readings service (P6-L): which caption lines get readings, the request sent to
 * `plugin:text-readings|text_readings_annotate`, the `readings.stt` event built from the result, and a serial queue
 * that coalesces revisions of the same line. No `window`, no Tauri: `./index.ts` injects both.
 */

/** Catalog id of the Japanese dictionary (Rust `text_readings::japanese::DICTIONARY_ID`). */
export const JAPANESE_DICTIONARY_ID = "text/jpreprocess-naist-jdic";
/** Catalog id of the music / singing gate model (Rust `speech::music_gate::MODEL_ID`). */
export const MUSIC_GATE_MODEL_ID = "vad/firered-aed";
/** Longest line that gets readings (UTF-16 units; the `ReadingsEvent.value` limit, Rust `MAX_TEXT_CHARS`). */
export const MAX_READINGS_TEXT = 2_000;
/** Lines waiting for readings at most; the oldest interims go first. */
export const MAX_PENDING = 16;
/** Lines whose last published revision is remembered (to drop late results). */
const MAX_REMEMBERED = 256;

export type ReadingsLanguage = "ja" | "zh";

const KANA = /[ぁ-ゖァ-ヺーㇰ-ㇿｦ-ﾝ]/;
const HANGUL = /[ᄀ-ᇿ㄰-㆏가-힯]/;

/**
 * The language a line is read as: its `lang` when that is Japanese or Chinese (`zh-*`, `cmn`, `yue` read as
 * Mandarin), none for any other `lang`; without `lang`, Japanese when the text has kana. Han-only text without a
 * `lang` is not guessed (Japanese and Chinese share it, and wrong readings are worse than none).
 */
export function readingsLanguage(lang: string | undefined, text: string): ReadingsLanguage | null {
  if (lang) {
    const base = lang.trim().toLowerCase().replace(/_/g, "-").split("-")[0];
    if (base === "ja" || base === "jpn") return "ja";
    if (base === "zh" || base === "cmn" || base === "yue" || base === "zho" || base === "chi") return "zh";
    return null;
  }
  if (KANA.test(text) && !HANGUL.test(text)) return "ja";
  return null;
}

/** The request for one line, or null when no reading of its language is switched on. */
export function readingsRequest(lang: ReadingsLanguage, text: string, settings: ReadingsState): RustReadingsRequest | null {
  if (lang === "ja") {
    if (!settings.hiragana && !settings.romaji) return null;
    return { text, lang, hiragana: settings.hiragana, romaji: settings.romaji, pinyin: false };
  }
  if (!settings.pinyin) return null;
  return { text, lang, hiragana: false, romaji: false, pinyin: true, pinyinStyle: settings.pinyinStyle };
}

const UTTERANCE_ID = /^[A-Za-z0-9._:-]{1,128}$/;

/** The id displays pair readings with: the line's `segmentId`, else its numeric `id`; null when it has neither. */
export function utteranceIdOf(event: Pick<TextEvent, "segmentId" | "id">): string | null {
  if (event.segmentId && UTTERANCE_ID.test(event.segmentId)) return event.segmentId;
  if (typeof event.id === "number" && Number.isFinite(event.id)) {
    const id = String(event.id);
    if (UTTERANCE_ID.test(id)) return id;
  }
  return null;
}

/** The `readings.stt` payload of one result (validated; null when it does not fit `ReadingsEventSchema`). */
export function toReadingsEvent(args: {
  utteranceId: string;
  revision?: number;
  source: CaptionEventSource;
  value: string;
  result: RustReadingsResult;
}): ReadingsEvent | null {
  const { result } = args;
  const event: ReadingsEvent = {
    utteranceId: args.utteranceId,
    source: args.source,
    lang: result.lang,
    value: args.value,
  };
  if (args.revision !== undefined) event.revision = args.revision;
  if (result.hiragana != null) event.hiragana = result.hiragana;
  if (result.romaji != null) event.romaji = result.romaji;
  if (result.pinyin != null) event.pinyin = result.pinyin;
  if (result.segments?.length) {
    event.segments = result.segments.map((s) => (s.reading != null ? { text: s.text, reading: s.reading } : { text: s.text }));
  }
  const parsed = ReadingsEventSchema.safeParse(event);
  return parsed.success ? parsed.data : null;
}

/** Whether `topic` is one of `sources` or below one of them (`others` selects `others.stt` and `others.translation`). */
export function sourceSelected(sources: readonly string[], topic: string): boolean {
  return sources.some((s) => topic === s || topic.startsWith(`${s}.`));
}

/** Whether an error from `text_readings_annotate` means the Japanese dictionary is missing or broken. */
export function isDictionaryError(message: string): boolean {
  return message.includes(JAPANESE_DICTIONARY_ID);
}

export interface ReadingsAnnotatorDeps {
  /** `plugin:text-readings|text_readings_annotate`. */
  annotate(request: RustReadingsRequest): Promise<RustReadingsResult>;
  /** Publish on `readings.stt`. */
  publish(event: ReadingsEvent): void;
  /** The `readings` settings, read per line. */
  settings(): ReadingsState;
  /** The Japanese dictionary is usable (false: Japanese lines are skipped until it is). */
  japaneseReady(): boolean;
  /** An annotate call failed because of the Japanese dictionary. */
  onDictionaryError?(message: string): void;
  /** System log line (deduplicated per message by the annotator). */
  log?(line: string, level: "info" | "warning" | "error"): void;
  /** Fallback id for a line without `segmentId` / `id`. */
  newId?(): string;
  now?(): number;
}

/** What the panel shows. */
export interface ReadingsAnnotatorState {
  /** A request is running. */
  busy: boolean;
  /** Lines published so far. */
  published: number;
  /** Last error message (cleared by the next success). */
  lastError: string | null;
  /** Duration of the last annotate call (ms). */
  lastLatencyMs: number | null;
  /** The last published event (preview in the panel). */
  last: ReadingsEvent | null;
}

interface Job {
  key: string;
  seq: number;
  utteranceId: string;
  revision?: number;
  source: CaptionEventSource;
  value: string;
  interim: boolean;
  request: RustReadingsRequest;
}

/**
 * Serial annotate queue: one request at a time; a newer revision of a waiting line replaces it; a result is published
 * only when no newer revision of its line has been published or is waiting.
 */
export class ReadingsAnnotator {
  readonly state: ReadingsAnnotatorState;
  readonly #deps: ReadingsAnnotatorDeps;
  readonly #pending = new Map<string, Job>();
  /** Last published arrival number per line key. */
  readonly #published = new Map<string, number>();
  #seq = 0;
  #running = false;
  #fallbackIds = 0;
  #lastLogged: string | null = null;
  /** Bumped by `clear()`: results of earlier requests are dropped. */
  #epoch = 0;

  constructor(deps: ReadingsAnnotatorDeps, state?: ReadingsAnnotatorState) {
    this.#deps = deps;
    this.state = state ?? { busy: false, published: 0, lastError: null, lastLatencyMs: null, last: null };
  }

  /** A caption line from `source`. Ignored unless readings are on, the source is selected and its language has a form switched on. */
  push(source: CaptionEventSource, event: TextEvent | undefined): void {
    if (!event) return;
    const settings = this.#deps.settings();
    if (!settings.enable || !sourceSelected(settings.sources, source)) return;
    const interim = event.type === TextEventType.interim;
    if (interim && !settings.interim) return;
    const value = typeof event.value === "string" ? event.value : "";
    if (!value.trim() || value.length > MAX_READINGS_TEXT) return;
    const lang = readingsLanguage(event.lang, value);
    if (!lang) return;
    if (lang === "ja" && !this.#deps.japaneseReady()) return;
    const request = readingsRequest(lang, value, settings);
    if (!request) return;
    const utteranceId = utteranceIdOf(event) ?? this.#newId();
    const key = `${source}|${utteranceId}`;
    const job: Job = { key, seq: ++this.#seq, utteranceId, source, value, interim, request };
    if (event.revision !== undefined) job.revision = event.revision;
    // Replace a waiting revision of the same line in place (Map keeps its position).
    this.#pending.set(key, job);
    this.#trim();
    void this.#pump();
  }

  /** Drop waiting lines and ignore results still in flight (readings switched off). */
  clear(): void {
    this.#pending.clear();
    this.#epoch++;
  }

  /** Lines waiting (tests, diagnostics). */
  get pendingCount(): number {
    return this.#pending.size;
  }

  #newId(): string {
    return this.#deps.newId?.() ?? `readings-${++this.#fallbackIds}`;
  }

  #trim(): void {
    while (this.#pending.size > MAX_PENDING) {
      let victim: string | undefined;
      for (const [key, job] of this.#pending) {
        if (job.interim) {
          victim = key;
          break;
        }
      }
      this.#pending.delete(victim ?? this.#pending.keys().next().value!);
    }
  }

  async #pump(): Promise<void> {
    if (this.#running) return;
    this.#running = true;
    this.state.busy = true;
    try {
      while (this.#pending.size > 0) {
        const [key, job] = this.#pending.entries().next().value!;
        this.#pending.delete(key);
        await this.#run(job);
      }
    } finally {
      this.#running = false;
      this.state.busy = false;
    }
  }

  async #run(job: Job): Promise<void> {
    const epoch = this.#epoch;
    const now = this.#deps.now ?? (() => performance.now());
    const started = now();
    let result: RustReadingsResult;
    try {
      result = await this.#deps.annotate(job.request);
    } catch (e) {
      if (epoch !== this.#epoch) return;
      const message = e instanceof Error ? e.message : String(e);
      this.state.lastError = message;
      if (job.request.lang === "ja" && isDictionaryError(message)) this.#deps.onDictionaryError?.(message);
      if (this.#lastLogged !== message) {
        this.#lastLogged = message;
        this.#deps.log?.(message, "warning");
      }
      return;
    }
    if (epoch !== this.#epoch) return;
    this.state.lastLatencyMs = Math.round(now() - started);
    this.state.lastError = null;
    this.#lastLogged = null;
    // A newer revision of this line is waiting, or already out: this result is stale.
    if (this.#pending.has(job.key)) return;
    if ((this.#published.get(job.key) ?? 0) > job.seq) return;
    const event = toReadingsEvent({ utteranceId: job.utteranceId, revision: job.revision, source: job.source, value: job.value, result });
    if (!event) {
      this.#deps.log?.("Readings did not fit the event limits and were not published", "warning");
      return;
    }
    this.#published.delete(job.key);
    this.#published.set(job.key, job.seq);
    while (this.#published.size > MAX_REMEMBERED) this.#published.delete(this.#published.keys().next().value!);
    this.#deps.publish(event);
    this.state.published++;
    this.state.last = event;
  }
}
