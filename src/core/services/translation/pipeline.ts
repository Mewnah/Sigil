/**
 * Translation pipeline host (P4-C): one instance turns an input text stream into translated `TextEvent`s.
 *
 *   TextEvent (stt / textfield) → UtteranceTracker (segments, revisions, Smart Turn merging)
 *     → TranslationSequencer (latest-wins interims with throttling, finals in order, deadlines)
 *     → engine (a backend, or the router's chain) with the language pair, rolling context and line glossary
 *     → LocalAgreement-2 for interims (committed prefix only grows) / the final as translated
 *     → publish (`segmentId` of the line, rising `revision`, target `lang`, `stableLength` on interims).
 *
 * Guarantees: interims never flicker when LocalAgreement is on (each published interim extends the previous one of
 * its line, or shows a tentative tail marked by `stableLength`); every final that an engine translates is published,
 * in line order (a final that fails is retried once, reported, and only then given up); streamed final text is
 * shown as interims only while it extends what the line already shows.
 *
 * `createTranslationPipeline` makes independent instances: the translation service runs one for speech and typed
 * text; P6's other-player translation runs a second one with a reversed pair and its own output topic.
 *
 * Owned by P4-C.
 */
import { ErrorThrottle } from "@/core/services/tts/errorThrottle";
import { PartialWithRequired, ServiceNetworkState, TextEvent, TextEventType } from "@/types";
import i18n from "i18next";
import { toast } from "react-toastify";
import { proxy } from "valtio";
import { pushSystemLog } from "../systemLog";
import { raceWithSignal } from "./abort";
import { filterGlossary } from "./glossary";
import { canonicalLanguage, guessLanguage, languageName, sameLanguage, TRANSLATION_LANGUAGES } from "./languages";
import { interimTranslationsOn } from "./interimPolicy";
import { completeTokens, LocalAgreement } from "./localAgreement";
import { withOneRetry } from "./retry";
import { engineLabel, routerLineBudgetMs } from "./router";
import {
  SHARED_PAIR_BACKENDS,
  TRANSLATION_ENGINE_IDS,
  Translation_Backends,
  Translation_State,
  TranslationEngineId,
} from "./schema";
import {
  CANCEL_MARKER,
  finalDeadlineMs,
  INTERIM_TIMEOUT_MS,
  isCancelMarker,
  SequencerJob,
  SequencerTimeoutError,
  TranslationSequencer,
} from "./sequencer";
import { LineEvent, TrackedLine, UtteranceTracker } from "./tracker";
import {
  ITranslationEngine,
  ITranslationEngineConstructor,
  TranslationContextPair,
  TranslationError,
  TranslationRequest,
  TranslationStatusUpdate,
} from "./types";
import { PROCESS_NONCE } from "@/utils/processNonce";

export type TranslationEngineMap = { [k in Translation_Backends]: ITranslationEngineConstructor };

/**
 * State of one pipeline (`Service_Translation.serviceState`; frozen by P4-S0 for the inspector, fields may be added).
 * `statusText` / `statusLevel` / `progress` / `missing` describe engine preparation (model download or load, server
 * start, a fallback in use) and are UI only, never captions; `activeEngine` is the engine that produced the last
 * published final (the router's pick), "" before the first one.
 */
export interface TranslationServiceState {
  status: ServiceNetworkState;
  error: string;
  statusText: string;
  statusLevel: "info" | "warning";
  /** 0–100 while downloading / loading; null otherwise. */
  progress: number | null;
  activeEngine: Translation_Backends | "";
  /** Catalog ids the engine reported missing (P4-C addition): the inspector offers them for install. */
  missing: string[];
}

/** What a pipeline publishes (`pubsub.publishText` shape). */
export type TranslationOutputEvent = PartialWithRequired<TextEvent, "type" | "value">;

export interface TranslationPipelineOptions {
  /** Id prefix of generated line ids and log label (`[A-Za-z0-9.-]`), e.g. `tr`. */
  name: string;
  /** Live settings, read per line (the glossary, context and partials apply without a restart). */
  settings: () => Translation_State;
  engines: TranslationEngineMap;
  publish: (event: TranslationOutputEvent) => void;
  /**
   * Language pair override (other players' speech is translated the other way). Sides left out come from the
   * settings. When set, every backend gets explicit `from` / `to`, the legacy ones included.
   */
  pair?: () => { from?: string; to?: string } | null;
  /** System-log source (default "Translation"). */
  logSource?: string;
  /** Show toasts for start failures and throttled line errors (default true). */
  toasts?: boolean;
  /**
   * The bundled llama-server translator runs on the processor (read per partial). With `interim: "auto"` that
   * turns interim translations off (`interimPolicy.ts`); default: never.
   */
  translatorOnCpu?: () => boolean;
}

export interface TranslationPipeline {
  readonly state: TranslationServiceState;
  start(): void;
  stop(): void;
  /** Feeds one input event; `stream` names an independent utterance stream (`stt`, `textfield`, ...). */
  push(stream: string, event: TextEvent): void;
  dispose(): void;
}

/** Rolling context entries kept per stream. */
const CONTEXT_KEEP = 10;
/** Context older than this is not sent (a new conversation). */
export const CONTEXT_MAX_AGE_MS = 5 * 60_000;
/** Longest final deadline the pipeline grants a chain of engines. */
const MAX_FINAL_DEADLINE_MS = 180_000;
const MAX_INTERIM_DEADLINE_MS = 60_000;

interface ContextEntry extends TranslationContextPair {
  from: string;
  to: string;
  at: number;
}

/** Earlier final pairs of each stream (Hy-MT2 background block, DeepL `context`). */
export class RollingContext {
  #byStream = new Map<string, ContextEntry[]>();

  constructor(private readonly now: () => number = () => Date.now()) {}

  add(stream: string, entry: Omit<ContextEntry, "at">): void {
    const list = this.#byStream.get(stream) ?? [];
    list.push({ ...entry, at: this.now() });
    while (list.length > CONTEXT_KEEP) list.shift();
    this.#byStream.set(stream, list);
  }

  /** Up to `turns` most recent pairs for the same target (and source, when both are known), oldest first. */
  recent(stream: string, from: string, to: string, turns: number): TranslationContextPair[] {
    if (turns <= 0) return [];
    const now = this.now();
    const list = (this.#byStream.get(stream) ?? []).filter(
      (e) => now - e.at <= CONTEXT_MAX_AGE_MS && e.to === to && (!from || !e.from || e.from === from),
    );
    return list.slice(-turns).map(({ source, target }) => ({ source, target }));
  }

  clear(): void {
    this.#byStream.clear();
  }
}

interface LineState {
  la: LocalAgreement | null;
  /** Text of the line's last published interim (committed part plus any tentative tail). */
  shownValue: string;
  /** Committed part of what the line shows (never shrinks). */
  shownCommitted: string;
  /** Publish revision counter of the line. */
  revision: number;
}

/** Message of `e`; "" when it has none (an engine threw `new Error(i18n.t(...))` before i18n was ready). */
function errorText(e: unknown): string {
  if (e instanceof SequencerTimeoutError) {
    return i18n.t("transl.error_timeout", { seconds: Math.round(e.timeoutMs / 1000) }) ?? "";
  }
  const text = e instanceof Error ? e.message : e === undefined || e === null ? "" : String(e);
  return text.trim();
}

/** Never empty: a failure the user cannot see explained is a failure they cannot fix. */
function startErrorText(e: unknown): string {
  return errorText(e) || i18n.t("transl.error_start_no_details") || "Translation could not start. Check the provider settings.";
}

function lineErrorText(e: unknown): string {
  return errorText(e) || i18n.t("common.error_no_details") || "no details";
}

/**
 * Whether a started engine can translate into `to` from some source: `supports("", to)` (unknown source), or from
 * any known language. An engine that cannot detect languages (Bergamot) answers false for an unknown source, yet
 * translates into `to` from the sources it has.
 */
function targetReachable(engine: ITranslationEngine, to: string): boolean {
  if (!engine.supports || engine.supports("", to) !== false) return true;
  return TRANSLATION_LANGUAGES.some((l) => !sameLanguage(l.code, to) && engine.supports?.(l.code, to) !== false);
}

/** Why a started engine cannot serve the pipeline's pair ("" = it can, or it cannot tell). */
function startPairProblem(engine: ITranslationEngine, from: string, to: string): "" | "target" | "pair" {
  if (!engine.supports) return "";
  if (!targetReachable(engine, to)) return "target";
  if (from && engine.supports(from, to) === false) return "pair";
  return "";
}

const LEGACY_TARGET: Partial<Record<Translation_Backends, (s: Translation_State) => string>> = {
  [Translation_Backends.azure]: (s) => s.azure.language,
  [Translation_Backends.libretranslate]: (s) => s.libretranslate.language,
  [Translation_Backends.onDevice]: (s) => s.onDevice.language,
};

class TranslationPipelineImpl implements TranslationPipeline {
  readonly state = proxy<TranslationServiceState>({
    status: ServiceNetworkState.disconnected,
    error: "",
    statusText: "",
    statusLevel: "info",
    progress: null,
    activeEngine: "",
    missing: [],
  });

  readonly #o: TranslationPipelineOptions;
  #engine: ITranslationEngine | null = null;
  #backend: Translation_Backends | null = null;
  /** Bumped on every start and stop; async work of an older session must not touch state. */
  #epoch = 0;
  #startController: AbortController | null = null;
  #errors = new ErrorThrottle();
  #trackers = new Map<string, UtteranceTracker>();
  #lines = new WeakMap<TrackedLine, LineState>();
  #used = new WeakMap<SequencerJob<LineEvent>, TranslationEngineId>();
  #context = new RollingContext();
  #sequencer: TranslationSequencer<LineEvent>;

  constructor(options: TranslationPipelineOptions) {
    this.#o = options;
    this.#sequencer = new TranslationSequencer<LineEvent>({
      run: (job) => this.#run(job),
      publish: (job, value) => this.#publishJob(job, value),
      onError: (job, error) => this.#reportError(job, error),
      timeoutMs: {
        interim: (event) => this.#interimDeadline(event.value),
        final: (event) => this.#finalDeadline(event.value),
      },
      // The tracker recognises late interims (with segment ids exactly, without them by text).
      staleInterimMs: 0,
      interimGapMs: () => this.#settings().partials.throttleMs,
    });
  }

  #settings(): Translation_State {
    return this.#o.settings();
  }

  // --- lifecycle ------------------------------------------------------------------------------------------------

  start(): void {
    this.#teardown();
    const epoch = ++this.#epoch;
    this.state.error = "";
    this.state.activeEngine = "";
    this.state.missing = [];
    this.#clearStatus();
    this.#errors.reset();

    const settings = this.#settings();
    const backend = settings.backend;
    const Engine = this.#o.engines[backend];
    if (!Engine) {
      this.state.status = ServiceNetworkState.disconnected;
      return;
    }
    const engine = new Engine();
    const controller = new AbortController();
    this.#engine = engine;
    this.#backend = backend;
    this.#startController = controller;
    this.state.status = ServiceNetworkState.connecting;

    const chainable = Object.fromEntries(TRANSLATION_ENGINE_IDS.map((id) => [id, this.#o.engines[id]])) as Record<
      TranslationEngineId,
      ITranslationEngineConstructor
    >;
    // With a pair override (P6's reverse pipeline) engines validate, warm up and route for the overridden pair.
    const engineSettings = this.#engineSettings(settings);
    let starting: Promise<void>;
    try {
      engine.attach?.({
        engines: chainable,
        status: (update) => {
          if (epoch === this.#epoch) this.#applyStatus(update);
        },
      });
      starting = Promise.resolve(engine.start(engineSettings, controller.signal));
    } catch (e) {
      starting = Promise.reject(e);
    }
    void (async () => {
      try {
        await raceWithSignal(starting, controller.signal);
      } catch (e) {
        // stop() or a newer start() owns the state now.
        if (epoch !== this.#epoch) return;
        this.#failStart(startErrorText(e), e instanceof TranslationError ? e.missing : []);
        return;
      }
      if (epoch !== this.#epoch) return;
      this.#startController = null;
      const { from, to } = this.#pair({ value: "", line: { id: "", segments: [] } });
      const problem = to ? startPairProblem(engine, from ?? "", to) : "";
      if (problem) {
        const engineName = engineLabel(backend);
        const message =
          problem === "target"
            ? i18n.t("transl.error_unsupported_target", { engine: engineName, language: languageName(to) }) ||
              `Unsupported target language: ${to}`
            : i18n.t("transl.error_unsupported_pair", { engine: engineName, pair: `${languageName(from)} → ${languageName(to)}` }) ||
              `Unsupported language pair: ${from} → ${to}`;
        this.#failStart(message, []);
        return;
      }
      this.state.status = ServiceNetworkState.connected;
    })();
  }

  /** The settings engines see: with a pair override, its languages replace the shared pair. */
  #engineSettings(settings: Translation_State): Translation_State {
    const override = this.#o.pair?.() ?? null;
    if (!override) return settings;
    // A shallow view: the engine slices stay the live settings objects.
    return {
      ...settings,
      sourceLanguage: override.from ?? settings.sourceLanguage,
      targetLanguage: override.to ?? settings.targetLanguage,
    };
  }

  stop(): void {
    this.#epoch++;
    this.#teardown();
    this.#clearStatus();
    this.state.missing = [];
    this.state.status = ServiceNetworkState.disconnected;
  }

  dispose(): void {
    this.stop();
  }

  push(stream: string, event: TextEvent): void {
    if (!this.#engine || this.state.status !== ServiceNetworkState.connected) return;
    if (!event?.value?.trim()) return;
    this.#tracker(stream).push(event);
  }

  /** Aborts a pending start, drops in-flight translations and stops the engine. */
  #teardown(): void {
    this.#startController?.abort();
    this.#startController = null;
    this.#sequencer.reset();
    for (const t of this.#trackers.values()) t.reset();
    this.#trackers.clear();
    this.#lines = new WeakMap();
    this.#context.clear();
    const engine = this.#engine;
    this.#engine = null;
    this.#backend = null;
    try {
      engine?.stop();
    } catch (e) {
      console.error("[Translation] engine stop failed:", e);
    }
  }

  #failStart(message: string, missing: string[]): void {
    // A new epoch: late status reports of the failed engine (or its router members) are ignored.
    this.#epoch++;
    this.#teardown();
    this.#clearStatus();
    this.state.status = ServiceNetworkState.disconnected;
    this.state.error = message;
    this.state.missing = [...missing];
    pushSystemLog(this.#o.logSource ?? "Translation", message, "error");
    if (this.#o.toasts !== false) toast(message, { type: "error", autoClose: false });
  }

  #clearStatus(): void {
    this.state.statusText = "";
    this.state.statusLevel = "info";
    this.state.progress = null;
  }

  #applyStatus(update: TranslationStatusUpdate): void {
    this.state.statusText = update.text;
    this.state.statusLevel = update.level ?? "info";
    this.state.progress = typeof update.progress === "number" && Number.isFinite(update.progress)
      ? Math.max(0, Math.min(100, update.progress))
      : null;
    if (update.missing) this.state.missing = [...update.missing];
    else if (!update.text) this.state.missing = [];
  }

  // --- input ----------------------------------------------------------------------------------------------------

  #tracker(stream: string): UtteranceTracker {
    let t = this.#trackers.get(stream);
    if (!t) {
      const safe = (s: string) => s.replace(/[^A-Za-z0-9.-]/g, "") || "x";
      t = new UtteranceTracker({
        // The process nonce keeps generated line ids unique across restarts and linked hosts (no ':' in the prefix).
        idPrefix: `${safe(this.#o.name)}.${safe(stream)}.${PROCESS_NONCE}.${this.#epoch}`,
        turnEndCommit: () => this.#settings().partials.turnEndCommit,
        holdMs: () => this.#settings().partials.turnHoldMs,
        sink: {
          interim: (ev) => {
            if (this.#interimEnabled()) this.#sequencer.pushInterim(stream, ev, { trusted: true });
          },
          final: (ev) => this.#sequencer.pushFinal(stream, ev),
          cancel: (ev) => this.#sequencer.cancel(stream, ev, CANCEL_MARKER),
          abandon: () => this.#sequencer.abandon(stream),
        },
      });
      this.#trackers.set(stream, t);
    }
    return t;
  }

  #interimEnabled(): boolean {
    const settings = this.#settings();
    const backend = this.#backend;
    // Only the automatic value looks at where the translator runs.
    const onCpu = backend !== null && settings[backend]?.interim === "auto" && (this.#o.translatorOnCpu?.() ?? false);
    return interimTranslationsOn(settings, backend, onCpu);
  }

  /** Canonical pair of a line; both undefined for the legacy backends (they use their own slice). */
  #pair(ev: LineEvent): { from?: string; to?: string } {
    const override = this.#o.pair?.() ?? null;
    const backend = this.#backend;
    if (override === null && (backend === null || !SHARED_PAIR_BACKENDS.includes(backend))) return {};
    const s = this.#settings();
    const to = canonicalLanguage(override?.to ?? s.targetLanguage);
    const from =
      canonicalLanguage(override?.from ?? s.sourceLanguage) || canonicalLanguage(ev.line.lang) || guessLanguage(ev.value);
    return { from, to };
  }

  /** Target language of published events (canonical; "" = unknown). */
  #targetLanguage(ev: LineEvent): string {
    const { to } = this.#pair(ev);
    if (to !== undefined) return to;
    const backend = this.#backend;
    const legacy = backend ? LEGACY_TARGET[backend] : undefined;
    return legacy ? canonicalLanguage(legacy(this.#settings())) : "";
  }

  // --- engine calls ---------------------------------------------------------------------------------------------

  #request(job: SequencerJob<LineEvent>): TranslationRequest {
    const s = this.#settings();
    const ev = job.event;
    const { from, to } = this.#pair(ev);
    const request: TranslationRequest = { text: ev.value, kind: job.kind, requestId: `${ev.line.id}#${job.revision}` };
    if (from !== undefined) request.from = from;
    if (to !== undefined) request.to = to;
    if (s.context.enabled) {
      const pairs = this.#context.recent(job.stream, from ?? "", to ?? this.#targetLanguage(ev), s.context.turns);
      if (pairs.length) request.context = pairs;
      const background = s.context.background.trim();
      if (background) request.background = background;
    }
    const glossary = filterGlossary(s.glossary, ev.value, from ?? "", to ?? this.#targetLanguage(ev));
    if (glossary.length) request.glossary = glossary;
    return request;
  }

  #run(job: SequencerJob<LineEvent>): Promise<string> {
    const engine = this.#engine;
    if (!engine) return Promise.reject(new DOMException("Aborted", "AbortError"));
    const request = this.#request(job);
    request.onEngine = (id) => this.#used.set(job, id);
    if (job.kind === "interim") return engine.translate(request, job.signal);
    if (this.#interimEnabled()) request.onPartial = (text) => this.#onStreamedFinal(job, text);
    return withOneRetry(() => engine.translate(request, job.signal), job.signal);
  }

  #interimDeadline(text: string): number {
    const s = this.#settings();
    if (this.#backend === Translation_Backends.auto) {
      return Math.min(MAX_INTERIM_DEADLINE_MS, Math.max(INTERIM_TIMEOUT_MS, routerLineBudgetMs(s.auto, "interim", text) + 1_000));
    }
    return INTERIM_TIMEOUT_MS;
  }

  /**
   * A final's deadline covers the whole chain (the router bounds each member's start wait and attempt itself; this
   * is the safety net) or the LLM's own timeout, and never less than the default.
   */
  #finalDeadline(text: string): number {
    const base = finalDeadlineMs(text);
    const s = this.#settings();
    if (this.#backend === Translation_Backends.auto) {
      return Math.max(base, Math.min(MAX_FINAL_DEADLINE_MS, routerLineBudgetMs(s.auto, "final", text) + 2_000));
    }
    if (this.#backend === Translation_Backends.llm) return Math.max(base, Math.min(MAX_FINAL_DEADLINE_MS, s.llm.timeoutMs + 5_000));
    return base;
  }

  // --- output ---------------------------------------------------------------------------------------------------

  #lineState(ev: LineEvent): LineState {
    let ls = this.#lines.get(ev.line);
    if (!ls) {
      ls = { la: null, shownValue: "", shownCommitted: "", revision: 0 };
      this.#lines.set(ev.line, ls);
    }
    return ls;
  }

  #emit(ev: LineEvent, ls: LineState, type: TextEventType, value: string, stableLength?: number): void {
    const out: TranslationOutputEvent = { type, value, segmentId: ev.line.id, revision: ++ls.revision };
    const lang = this.#targetLanguage(ev);
    if (lang) out.lang = lang;
    if (type === TextEventType.final && typeof ev.line.turnEnd === "boolean") out.turnEnd = ev.line.turnEnd;
    if (stableLength !== undefined) out.stableLength = Math.max(0, Math.min(value.length, stableLength));
    this.#o.publish(out);
  }

  #publishJob(job: SequencerJob<LineEvent>, raw: string): boolean {
    const ev = job.event;
    const ls = this.#lineState(ev);
    const value = raw.trim();

    if (job.kind === "final") {
      if (!value) return false;
      this.#emit(ev, ls, TextEventType.final, value);
      if (!isCancelMarker(value)) {
        const { from, to } = this.#pair(ev);
        this.#context.add(job.stream, {
          source: ev.value,
          target: value,
          from: from ?? "",
          to: to ?? this.#targetLanguage(ev),
        });
        this.state.activeEngine = this.#used.get(job) ?? this.#backend ?? "";
      }
      return true;
    }

    const partials = this.#settings().partials;
    if (!partials.localAgreement) {
      if (!value || value === ls.shownValue) return false;
      ls.shownValue = value;
      this.#emit(ev, ls, TextEventType.interim, value);
      return true;
    }

    ls.la ??= new LocalAgreement(this.#targetLanguage(ev));
    // LocalAgreement's committed prefix only grows, so every published committed part extends the previous one.
    const update = ls.la.update(value);
    const committed = update.committed.trimEnd();
    if (partials.showTentative) {
      const shown = (update.committed + update.tentative).trimEnd();
      if (!shown || shown === ls.shownValue) return false;
      ls.shownCommitted = committed;
      ls.shownValue = shown;
      this.#emit(ev, ls, TextEventType.interim, shown, committed.length);
      return true;
    }
    if (!committed || committed === ls.shownValue) return false;
    ls.shownCommitted = committed;
    ls.shownValue = committed;
    this.#emit(ev, ls, TextEventType.interim, committed, committed.length);
    return true;
  }

  /**
   * Streamed text of a final (LLM engines): shown as an interim while the line is the next to publish and the text
   * extends what the line already shows; its last token may be incomplete, so it is held back.
   */
  #onStreamedFinal(job: SequencerJob<LineEvent>, text: string): void {
    if (job.signal.aborted || !this.#sequencer.isHead(job.stream, job.utterance)) return;
    const ev = job.event;
    const ls = this.#lineState(ev);
    const lang = this.#targetLanguage(ev);
    const complete = completeTokens(text.trimStart(), lang).trimEnd();
    // With LocalAgreement a tentative tail may be replaced; without it the whole interim on screen counts as shown.
    const shown = this.#settings().partials.localAgreement ? ls.shownCommitted : ls.shownValue;
    if (!complete || complete.length <= shown.length || !complete.startsWith(shown)) return;
    if (complete === ls.shownValue) return;
    ls.shownCommitted = complete;
    ls.shownValue = complete;
    this.#emit(ev, ls, TextEventType.interim, complete, complete.length);
  }

  #reportError(job: SequencerJob<LineEvent>, error: unknown): void {
    const message = lineErrorText(error);
    const line = i18n.t("transl.error_line_failed", { error: message }) || message;
    console.warn(`[Translation] ${job.kind} translation failed:`, error);
    if (error instanceof TranslationError && error.missing.length) this.state.missing = [...error.missing];
    // A failed final loses a line, so it is always logged; interim failures (the next partial replaces
    // them) are logged only together with a toast. Toasts are throttled: a dead server fails every line.
    const show = this.#errors.shouldShow(message);
    if (job.kind === "final" || show) pushSystemLog(this.#o.logSource ?? "Translation", line, "warning");
    if (show && this.#o.toasts !== false) toast(line, { type: "warning", autoClose: 8_000 });
  }
}

/** Creates an independent translation pipeline (see the module docs). */
export function createTranslationPipeline(options: TranslationPipelineOptions): TranslationPipeline {
  return new TranslationPipelineImpl(options);
}
