import { invoke, isTauri } from "@tauri-apps/api/core";
import { proxy, subscribe } from "valtio";
import { pushSystemLog } from "@/core/services/systemLog";
import {
  CaptionEventSourceSchema,
  ReadingsEventSource,
  type CaptionEventSource,
  type IServiceInterface,
  type ReadingsEvent,
  type RustReadingsRequest,
  type RustReadingsResult,
  type RustReadingsStatus,
} from "@/types";
import { JAPANESE_DICTIONARY_ID, ReadingsAnnotator, type ReadingsAnnotatorState } from "./annotator";
import type { ReadingsState } from "./schema";

export { JAPANESE_DICTIONARY_ID, MUSIC_GATE_MODEL_ID } from "./annotator";

/** System-log source of the readings service. */
const LOG_SOURCE = "Readings";
/** Free the dictionaries this long after readings are switched off (switching back on soon keeps them warm). */
const UNLOAD_AFTER_MS = 60_000;

/**
 * Readings (P6-L): annotates Japanese and Chinese caption lines of the topics in the root settings key `readings`
 * (`./schema.ts`) with `plugin:text-readings|text_readings_annotate` (hiragana / romaji from the NAIST-JDIC dictionary,
 * pinyin with tone sandhi) and publishes the result on `readings.stt` (`ReadingsEventSource.stt`, payload
 * `ReadingsEvent` keyed by the line's `utteranceId`: its `segmentId`, or its numeric `id`). Never `text.*`.
 * Not a sidebar tab: the settings UI is `./ReadingsSection.tsx`, mounted at the end of the STT inspector.
 * `ApiServer.readings` and its init task are frozen (P6-S0). Off by default.
 */
class Service_Readings implements IServiceInterface {
  /** Queue and last result (`./annotator.ts`), plus what the dictionary check found. */
  readonly state = proxy({
    busy: false,
    published: 0,
    lastError: null as string | null,
    lastLatencyMs: null as number | null,
    last: null as ReadingsEvent | null,
    /** `text_readings_status`: null until checked (Japanese lines are tried meanwhile). */
    japaneseInstalled: null as boolean | null,
  });

  readonly #annotator: ReadingsAnnotator;
  #subscriptions: string[] = [];
  #subscribedKey = "";
  /** Enabled with at least one valid source (lines are being read). */
  #active = false;
  #unloadTimer: ReturnType<typeof setTimeout> | null = null;
  #statusSeq = 0;
  #started = false;

  constructor() {
    this.#annotator = new ReadingsAnnotator(
      {
        annotate: (request: RustReadingsRequest) =>
          invoke<RustReadingsResult>("plugin:text-readings|text_readings_annotate", { request }),
        publish: (event) => window.ApiShared.pubsub.publish(ReadingsEventSource.stt, event),
        settings: () => this.data,
        japaneseReady: () => this.state.japaneseInstalled !== false,
        onDictionaryError: () => {
          this.state.japaneseInstalled = false;
          void this.refreshStatus();
        },
        log: (line, level) => pushSystemLog(LOG_SOURCE, line, level),
      },
      this.state as ReadingsAnnotatorState,
    );
  }

  get data(): ReadingsState {
    return window.ApiServer.state.readings;
  }

  /** Subscribe to the configured sources while `readings.enable` is on (off by default). */
  async init() {
    if (this.#started || !isTauri() || window.Config.isClient()) return;
    this.#started = true;
    // The root state: the `readings` object itself may be replaced (settings import / reset).
    subscribe(window.ApiServer.state, () => this.#apply());
    // The dictionary's install state changes from the Models tab or the requirement card.
    let dictionaryState = window.ApiServer.models.state.states[JAPANESE_DICTIONARY_ID]?.state;
    subscribe(window.ApiServer.models.state, () => {
      const next = window.ApiServer.models.state.states[JAPANESE_DICTIONARY_ID]?.state;
      if (next === dictionaryState) return;
      dictionaryState = next;
      if (this.data.enable) void this.refreshStatus();
    });
    this.#apply();
  }

  /** Ask Rust whether the Japanese dictionary is installed. */
  async refreshStatus(): Promise<RustReadingsStatus | null> {
    const seq = ++this.#statusSeq;
    try {
      const status = await invoke<RustReadingsStatus>("plugin:text-readings|text_readings_status");
      if (seq === this.#statusSeq) this.state.japaneseInstalled = status.japanese;
      return status;
    } catch (e) {
      pushSystemLog(LOG_SOURCE, `Readings status failed: ${e instanceof Error ? e.message : String(e)}`, "warning");
      return null;
    }
  }

  /**
   * (Re)subscribe when `enable` or `sources` change. Readings are active while enabled with at least one source; once
   * they stop (switched off, or every source unchecked) the queue is cleared and the dictionaries are freed a minute
   * later. `enable` is part of the key, so "on with no source" and "off" are different states.
   */
  #apply() {
    const s = this.data;
    const sources = s.enable ? this.#validSources(s.sources) : [];
    const key = JSON.stringify([s.enable, sources]);
    if (key === this.#subscribedKey) return;
    this.#subscribedKey = key;
    for (const token of this.#subscriptions) window.ApiShared.pubsub.unsubscribe(token);
    this.#subscriptions = [];
    const wasActive = this.#active;
    this.#active = sources.length > 0;
    if (!this.#active) {
      this.#annotator.clear();
      // Nothing was loaded when readings were never active since startup.
      if (wasActive) this.#scheduleUnload();
      // On with every source unchecked: keep the dictionary status current for the settings, load nothing.
      if (s.enable) void this.refreshStatus();
      return;
    }
    this.#cancelUnload();
    for (const source of sources) {
      this.#subscriptions.push(
        window.ApiShared.pubsub.subscribeText(source, (event, eventName) => {
          // A parent topic (`text`, `others`) hears its children: key the readings by the line's own topic.
          const topic = CaptionEventSourceSchema.safeParse(eventName);
          this.#annotator.push(topic.success ? topic.data : source, event);
        }),
      );
    }
    void this.refreshStatus().then((status) => {
      // Warm the Japanese dictionary up so the first line is not late (it loads in about a second).
      if (status?.japanese && this.#active && (this.data.hiragana || this.data.romaji)) {
        invoke("plugin:text-readings|text_readings_annotate", {
          request: { text: "あ", lang: "ja", hiragana: true, romaji: false, pinyin: false } satisfies RustReadingsRequest,
        }).catch(() => {});
      }
    });
  }

  /** Valid topics, without duplicates and without topics a selected parent already hears (each line once). */
  #validSources(sources: readonly string[]): CaptionEventSource[] {
    const valid: CaptionEventSource[] = [];
    for (const s of sources) {
      const parsed = CaptionEventSourceSchema.safeParse(s);
      if (parsed.success && !valid.includes(parsed.data)) valid.push(parsed.data);
    }
    return valid.filter((s) => !valid.some((parent) => parent !== s && s.startsWith(`${parent}.`)));
  }

  #scheduleUnload() {
    this.#cancelUnload();
    this.#unloadTimer = setTimeout(() => {
      this.#unloadTimer = null;
      if (this.#active) return;
      invoke("plugin:text-readings|text_readings_unload").catch(() => {});
    }, UNLOAD_AFTER_MS);
  }

  #cancelUnload() {
    if (this.#unloadTimer) clearTimeout(this.#unloadTimer);
    this.#unloadTimer = null;
  }

  /** Catalog id of the Japanese dictionary (for `<ModelRequirement>`). */
  get japaneseDictionaryId(): string {
    return JAPANESE_DICTIONARY_ID;
  }
}

export default Service_Readings;
