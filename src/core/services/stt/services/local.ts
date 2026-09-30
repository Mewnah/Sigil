/**
 * The `local` STT backend (P2-E): Sigil's in-process speech pipeline (Rust `speech` plugin) with automatic model
 * routing by language and hardware ("Best for my PC").
 *
 * Start:
 * 1. `plugin:speech|preview_route` gives the plan for the settings on this PC (streaming / final / fallback /
 *    language-ID engines and the catalog ids they need).
 * 2. Missing models: nothing downloads without consent. An interactive start asks with a native dialog (the total
 *    size, and a smaller single-pass setup when that saves a lot); an automatic start (auto-start at launch) stops with
 *    a message instead. Downloads go through the model store (`window.ApiServer.models`), with progress on the status
 *    line; a stop cancels the downloads this start queued (the partial files are kept for resume).
 * 3. `plugin:speech|start_pipeline` loads the engines and opens the microphone through the capture hub; its progress
 *    arrives as `speech:status`.
 *
 * Running: the `speech:*` events of this session (same `pipelineId`, same epoch; the first epoch above the last one
 * seen while a start waits for its result) go through {@link SegmentPublisher}, which maps them to
 * `onInterim` / `onFinal` with `segmentId` / `revision` / `lang` / `turnEnd` metadata. `speech:status` feeds the
 * status line only (never published). Before Service_STT closes the open sentence (mute, stop), `commitPending`
 * publishes the finished sentences still held for their two-pass replacement. A pipeline that stops itself with a
 * retryable error (device lost, engine crash) is restarted automatically with back-off; after {@link MAX_RESTARTS}
 * failed restarts in a row (a session that ran {@link STABLE_RUN_MS} starts a new count), or on an error a restart
 * cannot fix (a model went missing, an unsupported or invalid request; see {@link restartCanFix}), the session stops
 * with a message.
 *
 * Pure parts (the publisher, config builders, status and consent texts) are exported for `__tests__/localDriver`.
 */
import { invoke } from "@tauri-apps/api/core";
import type { UnlistenFn } from "@tauri-apps/api/event";
import { confirm, message } from "@tauri-apps/plugin-dialog";
import i18n from "i18next";
import { proxy, subscribe } from "valtio";
import {
  SPEECH_EVENTS,
  type RustCatalogEntry,
  type RustModelEntryStatus,
  type RustSpeechEnvelope,
  type RustSpeechFinal,
  type RustSpeechPartial,
  type RustSpeechPipelineConfig,
  type RustSpeechPipelineStarted,
  type RustSpeechRoutePlan,
  type RustSpeechRouteRequest,
  type RustSpeechStatus,
  type RustSpeechTier,
  type RustSpeechTurnEnd,
  type RustSpeechUtterance,
} from "@/types";
import { formatBytes } from "@/core/services/diagnostics/format";
import {
  canDownloadEntry,
  entrySize,
  entryTitle,
  isBusyState,
  requirementFraction,
  summarizeRequirement,
  type EntryProgress,
  type PendingOp,
  type RequirementSummary,
} from "@/core/services/models/format";
import { pushSystemLog } from "@/core/services/systemLog";
import { PROCESS_NONCE } from "@/utils/processNonce";
import { resolveRustInputDeviceForInvoke } from "@/utils/rustAudioInput";
import { isAbortError, listenWithAbort, raceWithAbort } from "../abort";
import { SttStartLifecycle } from "../localEngineLifecycle";
import {
  isTwoPassPlan,
  normalizeLanguage,
  planHasEngine,
  planRoles,
  requiredModelIds,
  routeRequest,
  specOnGpu,
} from "../router";
import { noteSttPipelineStart } from "../rustCapturePolling";
import type { STT_State } from "../schema";
import { TRANSIENT_WARNING_MS } from "../statusChannel";
import { buildVadConfig, parseHotwords } from "../sttOptions";
import type { ISTTReceiver, ISTTService, SttStatusOptions, SttTextMeta } from "../types";

/** Pipeline key of speech-to-text (Rust `speech::engine_api::PIPELINE_MIC`). */
export const LOCAL_PIPELINE_ID = "mic";

// ---------------------------------------------------------------------------------------------------------------
// Two-pass timing. Mirrors Rust `speech::two_pass::TwoPassConfig::default()`: a replacement final must finish within
// `budget + perAudioSecond × audio seconds` (capped) of its segment's close, or Rust skips it and the first-pass final
// stands. The driver holds a first-pass final that long (plus IPC slack) waiting for its replacement.

/** Base budget of a two-pass replacement (ms from the segment's close). */
export const TWO_PASS_BUDGET_MS = 1_500;
/** Extra budget per second of utterance audio (ms). */
export const TWO_PASS_BUDGET_PER_AUDIO_SECOND_MS = 500;
/** Cap of a replacement's budget (ms). */
export const TWO_PASS_MAX_BUDGET_MS = 8_000;
/** Slack on top of Rust's budget for IPC and scheduling. */
export const HOLD_MARGIN_MS = 1_000;
/** Shortest hold after a first-pass final arrives. */
export const MIN_HOLD_MS = 300;
/**
 * A held first-pass final is published once the next sentence has had text waiting this long: live captions of the
 * new sentence win over the improvement of the old one.
 */
export const BLOCKED_HOLD_MS = 1_500;
/**
 * A segment still waiting for its first final is given up (its interim abandoned) once a later segment has had text
 * waiting this long. Its final is still published if it arrives later (after the segments ahead of it).
 */
export const STALL_MS = 8_000;
/** Holds in a row that ended without a replacement before holding is suspended (two-pass unavailable). */
export const MAX_HOLD_MISSES = 2;
/** Finished segment ids remembered to ignore late events. */
const FINISHED_MEMORY = 512;

// Auto-restart.

/** Waits before the automatic restarts after a pipeline error. */
export const RESTART_DELAYS_MS: readonly number[] = [1_000, 2_000, 5_000, 10_000, 20_000];
/** Failed restarts in a row after which the session stops. */
export const MAX_RESTARTS = RESTART_DELAYS_MS.length;
/**
 * A session that failed after running this long (since its last successful start) begins a new restart budget. A
 * restart that fails at start never ran, so it never resets the budget.
 */
export const STABLE_RUN_MS = 60_000;
/** `speech:status` codes of a fatal error that a restart can fix. */
const RETRYABLE_CODES: ReadonlySet<string> = new Set([
  "capture_error",
  "capture_gap",
  "engine_error",
  "engine_fallback",
  "backlog_dropped",
  "rate_limited",
  "",
]);
/**
 * Rust engine error kinds (`AsrErrorKind`) that lead a fatal detail or a start error as `<kind>: message`
 * (`worker.rs` fatals them at once, `is_fatal`): the same setup fails the same way on every start.
 */
const UNFIXABLE_KIND = /^\s*(invalid_input|unsupported|out_of_memory):/;

/**
 * Whether restarting the pipeline can fix a fatal error: its `speech:status` code (`""` for a failed start) is one a
 * restart can fix, and its detail does not start with an engine error kind that fails on every start
 * (`invalid_input`, `unsupported`). `out_of_memory` is worth a restart only when the next start can land on a lighter
 * setup: the plan has a CPU fallback, or the tier is automatic (the router plans again with the VRAM the resource
 * governor grants now). Any restart stays bounded by {@link MAX_RESTARTS}.
 */
export function restartCanFix(
  code: string,
  detail: string,
  plan: Pick<RustSpeechRoutePlan, "fallback"> | null,
  tier: RustSpeechTier | undefined,
): boolean {
  if (code === "not_installed" || !RETRYABLE_CODES.has(code)) return false;
  if (notInstalledId(detail)) return false;
  const kind = UNFIXABLE_KIND.exec(detail)?.[1];
  if (!kind) return true;
  if (kind === "out_of_memory") return !!plan?.fallback || (tier ?? "auto") === "auto";
  return false;
}

// Downloads.

/** Re-emit the download status (and re-check the store) this often while waiting on downloads. */
const DOWNLOAD_KEEPALIVE_MS = 15_000;
/** After this long, a failed / cancelled state of an entry this start queued is a report of the new attempt. */
const STALE_STATE_GRACE_MS = 2_000;
/** A queued download that never shows up as busy or installed within this long is reported as not started. */
const DOWNLOAD_START_TIMEOUT_MS = 30_000;
/** The smaller single-pass setup is offered when it downloads at most this share of the full one. */
const SMALLER_SETUP_RATIO = 0.8;
/** How long a stop waits for a `start_pipeline` call still in flight before giving up on it. */
const ORPHAN_START_WAIT_MS = 10_000;

// ---------------------------------------------------------------------------------------------------------------
// UI state

/** Live state of the local backend for the UI (the STT inspector's "Models in use"). */
export const localSpeechState = proxy({
  /** Plan of the running session (null when stopped or while it starts). */
  plan: null as RustSpeechRoutePlan | null,
  /** Automatic restarts in the current session (reset by each Start and Stop). */
  restarts: 0,
});

/** Latest epoch seen per pipeline key: events of older sessions are never accepted. */
const lastEpoch = new Map<string, number>();

function noteEpoch(pipelineId: string, epoch: number): void {
  if (Number.isFinite(epoch) && epoch > (lastEpoch.get(pipelineId) ?? 0)) lastEpoch.set(pipelineId, epoch);
}

// ---------------------------------------------------------------------------------------------------------------
// Settings → Rust

/** `stt.local` → the router request (normalized language; Rust fills anything missing). */
export function buildLocalRouteRequest(local: Partial<STT_State["local"]> | undefined): Required<RustSpeechRouteRequest> {
  return routeRequest({
    language: normalizeLanguage(local?.language) ?? "auto",
    tier: local?.tier,
    twoPass: local?.twoPass !== false,
  });
}

/** `stt.local` + `stt.vad` → `plugin:speech|start_pipeline { config }` for the router request `request`. */
export function buildLocalPipelineConfig(
  local: Partial<STT_State["local"]> | undefined,
  vad: Partial<STT_State["vad"]> | undefined,
  request: RustSpeechRouteRequest,
  source: string | null,
): RustSpeechPipelineConfig {
  const route = routeRequest(request);
  return {
    pipelineId: LOCAL_PIPELINE_ID,
    source: source ?? "",
    vad: buildVadConfig(vad),
    language: route.language,
    route,
    smartTurn: local?.smartTurn === true,
    hotwords: parseHotwords(local?.hotwords),
    hallucinationFilter: local?.hallucinationFilter !== false,
  };
}

// ---------------------------------------------------------------------------------------------------------------
// Status and error texts

export type StatusLine = [text: string, options?: SttStatusOptions];

const clampPercent = (p: number | null | undefined): number | null =>
  typeof p === "number" && Number.isFinite(p) && p >= 0 ? Math.min(100, p) : null;

/**
 * The status line for a `speech:status` event that is not fatal (`state` other than `error` / `stopped`); `null`
 * clears the line. `title` turns a catalog id into its display name.
 */
export function speechStatusLine(ev: RustSpeechStatus, title: (id: string) => string): StatusLine | null {
  const detail = String(ev.detail ?? "").trim();
  const model = ev.modelId ? title(ev.modelId) : String(ev.engine ?? "");
  const progress = clampPercent(ev.progress);
  const transient: SttStatusOptions = { level: "warning", expireMs: TRANSIENT_WARNING_MS };
  switch (ev.code) {
    case "listening":
    case "stopped":
      return null;
    case "starting":
      return [i18n.t("stt.local_status_starting")];
    case "loading_model":
      return [model ? i18n.t("stt.local_status_loading_model", { model }) : i18n.t("stt.local_status_starting"), { progress }];
    case "loading_runtime":
      return [i18n.t("stt.local_status_loading_runtime"), { progress }];
    case "downloading":
      return [i18n.t("stt.local_status_downloading", { title: model }), { progress }];
    case "backlog_dropped":
      return [i18n.t("stt.status_utterance_dropped"), transient];
    case "capture_gap":
      return [i18n.t("stt.local_status_capture_gap"), transient];
    case "capture_error":
      return [i18n.t("stt.error_capture", { detail }), transient];
    case "engine_fallback":
      return [i18n.t("stt.local_status_engine_fallback", { detail }), transient];
    case "not_installed":
      return [i18n.t("stt.local_status_not_installed", { model: model || detail }), transient];
    case "sherpa_language_auto":
      return [i18n.t("stt.local_status_language_auto"), transient];
    case "sherpa_hotwords_unavailable":
      return [i18n.t("stt.local_status_hotwords_unavailable"), transient];
    default:
      if (!detail) return null;
      return ev.level === "info" ? [i18n.t("stt.local_status_info", { detail })] : [i18n.t("stt.local_status_warning", { detail }), transient];
  }
}

/** Catalog id named by a Rust `not_installed: <id>[: why]` error, if any. */
export function notInstalledId(error: string): string | null {
  const m = /not_installed:\s*([A-Za-z0-9._-]+(?:\/[A-Za-z0-9._-]+)+)/.exec(error);
  return m ? m[1] : null;
}

/** A readable message for a failed `start_pipeline` (or a fatal `speech:status` detail). */
export function startErrorMessage(error: unknown, title: (id: string) => string): string {
  const text = errorText(error);
  const missing = notInstalledId(text);
  if (missing) return i18n.t("stt.local_error_not_installed", { model: title(missing) });
  if (/no speech engine for this request/i.test(text)) return i18n.t("stt.local_error_no_engine");
  return i18n.t("stt.local_error_start", { detail: text });
}

function errorText(error: unknown): string {
  if (typeof error === "string") return error;
  if (error instanceof Error) return error.message;
  try {
    return JSON.stringify(error);
  } catch {
    return String(error);
  }
}

// ---------------------------------------------------------------------------------------------------------------
// Consent

/** One download the consent dialog lists. */
export interface ConsentItem {
  title: string;
  bytes: number;
}

/** A download set: its entries and total size (`exact` false when a size is unknown). */
export interface ConsentSet {
  items: ConsentItem[];
  bytes: number;
  exact: boolean;
}

/** The single-pass alternative the consent dialog may offer. */
export interface SmallerConsentSet extends ConsentSet {
  /** It still shows live captions (a streaming model alone); false: captions after each sentence. */
  live: boolean;
}

/** What the consent dialog asks about. */
export interface ConsentRequest extends ConsentSet {
  /** The single-pass setup, when it downloads much less (null: not offered). */
  smaller: SmallerConsentSet | null;
}

export type ConsentChoice = "download" | "smaller" | "cancel";

const sizeLabel = (set: Pick<ConsentSet, "bytes" | "exact">) =>
  set.bytes <= 0 ? "" : set.exact ? formatBytes(set.bytes) : i18n.t("stt.local_size_at_least", { size: formatBytes(set.bytes) });

/** Texts of the consent dialog. */
export function consentTexts(req: ConsentRequest): { title: string; message: string; download: string; smaller: string | null; cancel: string } {
  const list = req.items
    .map((i) => (i.bytes > 0 ? i18n.t("stt.local_consent_item", { title: i.title, size: formatBytes(i.bytes) }) : `• ${i.title}`))
    .join("\n");
  let text = i18n.t("stt.local_consent_message", { size: sizeLabel(req), list });
  let smaller: string | null = null;
  if (req.smaller) {
    const size = sizeLabel(req.smaller);
    const hint = req.smaller.live
      ? size ? i18n.t("stt.local_consent_smaller_hint", { size }) : i18n.t("stt.local_consent_smaller_hint_free")
      : size ? i18n.t("stt.local_consent_smaller_hint_finals", { size }) : i18n.t("stt.local_consent_smaller_hint_finals_free");
    text += `\n\n${hint}`;
    smaller = size ? i18n.t("stt.local_consent_smaller", { size }) : i18n.t("stt.local_consent_smaller_free");
  }
  return {
    title: i18n.t("stt.local_consent_title"),
    message: text,
    download: sizeLabel(req) ? i18n.t("stt.local_consent_download", { size: sizeLabel(req) }) : i18n.t("stt.local_consent_download_unknown"),
    smaller,
    cancel: i18n.t("stt.local_consent_cancel"),
  };
}

/** The native consent dialog (two buttons, or three with the smaller setup). */
export async function askConsentDialog(req: ConsentRequest): Promise<ConsentChoice> {
  const t = consentTexts(req);
  if (t.smaller) {
    const r = await message(t.message, { title: t.title, kind: "info", buttons: { yes: t.download, no: t.smaller, cancel: t.cancel } });
    if (r === t.download || r === "Yes") return "download";
    if (r === t.smaller || r === "No") return "smaller";
    return "cancel";
  }
  const ok = await confirm(t.message, { title: t.title, kind: "info", okLabel: t.download, cancelLabel: t.cancel });
  return ok ? "download" : "cancel";
}

// ---------------------------------------------------------------------------------------------------------------
// Segment publisher

/** Where the publisher delivers transcripts (the STT receiver in production). */
export interface SegmentSink {
  interim(text: string, meta: SttTextMeta): void;
  final(text: string, meta: SttTextMeta): void;
  /** The segment's published interim was abandoned: no final follows. */
  abandon(meta: SttTextMeta): void;
}

export interface SegmentClock {
  now(): number;
  setTimeout(fn: () => void, ms: number): unknown;
  clearTimeout(handle: unknown): void;
}

const systemClock: SegmentClock = {
  now: () => Date.now(),
  setTimeout: (fn, ms) => setTimeout(fn, ms),
  clearTimeout: (h) => clearTimeout(h as ReturnType<typeof setTimeout>),
};

export interface SegmentPublisherOptions {
  pipelineId: string;
  epoch: number;
  /**
   * Stream part of the published segment ids (`<stream>:<epoch>:<n>`, [A-Za-z0-9._-], no ':'). Default
   * `<pipelineId>-<PROCESS_NONCE>`: epochs restart at 1 in every Sigil process, so the nonce keeps a restarted or
   * linked host's ids from matching lines this process already published.
   */
  stream?: string;
  /** The session re-decodes each segment (two-pass): hold first-pass finals for their replacement. */
  twoPass: boolean;
  sink: SegmentSink;
  clock?: SegmentClock;
  /** Holding was suspended (replacements stopped arriving) or resumed. */
  onHoldingChange?(holding: boolean): void;
}

interface Candidate {
  text: string;
  lang: string | null;
  revision: number;
}

interface FinalText extends Candidate {
  turnEnd: boolean | null;
}

interface Segment {
  id: number;
  /** Highest revision accepted. */
  lastRevision: number;
  /** Highest revision published (-1: nothing published). */
  shown: number;
  shownText: string;
  /** Newest partial not published yet. */
  partial: Candidate | null;
  first: FinalText | null;
  second: FinalText | null;
  dropped: boolean;
  turnEnd: boolean | null;
  /** Local time the utterance closed (VAD end). */
  closedAt: number | null;
  audioMs: number | null;
  /** Local time the first-pass final arrived. */
  firstAt: number | null;
  /** Local time this segment had text waiting behind an earlier one. */
  waitingSince: number | null;
}

/**
 * Turns one session's `speech:*` events into transcript calls, keeping the published stream well formed for
 * consumers without segment ids (captions, translation, chatbox, TTS): segments are published strictly one after
 * another, each as interims with rising revisions and then at most one final.
 *
 * - Partials become interims (the newest wins while a segment waits behind an earlier one).
 * - Finals: a two-pass session publishes the replacement (`pass: "second"`) as the segment's final; the first-pass
 *   final is shown as an interim meanwhile and becomes the final when no replacement arrives within Rust's two-pass
 *   budget (plus slack), or once the next sentence has had text waiting for {@link BLOCKED_HOLD_MS}. After
 *   {@link MAX_HOLD_MISSES} holds in a row without a replacement, first-pass finals are published at once until a
 *   replacement shows up again. Without two-pass, finals are published as they come.
 * - `dropped` abandons a published interim (no final). A segment still without a final when a later one has waited
 *   {@link STALL_MS} is given up the same way; its final is published if it arrives later.
 * - Smart Turn (`speech:turn_end`, or the final's own `turnEnd`) rides on the final as `turnEnd`.
 */
export class SegmentPublisher {
  readonly pipelineId: string;
  readonly epoch: number;
  readonly #stream: string;
  #twoPass: boolean;
  #holding: boolean;
  #misses = 0;
  readonly #sink: SegmentSink;
  readonly #clock: SegmentClock;
  readonly #onHoldingChange?: (holding: boolean) => void;
  readonly #segments = new Map<number, Segment>();
  /** Publication order (segment ids). */
  #queue: number[] = [];
  readonly #finished = new Set<number>();
  /** Given up by the stall guard: a late final is still published. */
  readonly #released = new Set<number>();
  #timer: unknown = null;
  #disposed = false;

  constructor(options: SegmentPublisherOptions) {
    this.pipelineId = options.pipelineId;
    this.epoch = options.epoch;
    this.#stream = options.stream ?? `${options.pipelineId}-${PROCESS_NONCE}`;
    this.#twoPass = options.twoPass;
    this.#holding = options.twoPass;
    this.#sink = options.sink;
    this.#clock = options.clock ?? systemClock;
    this.#onHoldingChange = options.onHoldingChange;
  }

  /** Whether first-pass finals are currently held for their replacement. */
  get holding(): boolean {
    return this.#holding;
  }

  /** The session's plan is known (or changed): two-pass or not. */
  setTwoPass(twoPass: boolean): void {
    if (this.#twoPass === twoPass) return;
    this.#twoPass = twoPass;
    this.#misses = 0;
    this.#setHolding(twoPass);
    this.#pump();
  }

  partial(ev: Pick<RustSpeechPartial, "segmentId" | "revision" | "text" | "lang">): void {
    if (this.#disposed || this.#finished.has(ev.segmentId) || this.#released.has(ev.segmentId)) return;
    const text = String(ev.text ?? "").trim();
    if (!text) return;
    const seg = this.#segment(ev.segmentId);
    if (seg.first || seg.second || seg.dropped || !(ev.revision > seg.lastRevision)) return;
    seg.lastRevision = ev.revision;
    seg.partial = { text, lang: ev.lang ?? null, revision: ev.revision };
    this.#pump();
  }

  final(
    ev: Pick<RustSpeechFinal, "segmentId" | "revision" | "text" | "lang" | "pass" | "latencyMs" | "startMs" | "endMs" | "turnEnd">,
  ): void {
    if (this.#disposed) return;
    const second = ev.pass === "second";
    if (second) this.#replacementSeen();
    const text = String(ev.text ?? "").trim();
    if (!text) return;
    const id = ev.segmentId;
    if (this.#finished.has(id)) return; // a replacement after its hold ended, or a duplicate
    const released = this.#released.delete(id);
    const seg = this.#segment(id, released);
    if (!(ev.revision > seg.lastRevision)) return;
    seg.lastRevision = ev.revision;
    const now = this.#clock.now();
    if (seg.closedAt === null) seg.closedAt = now - Math.max(0, Number(ev.latencyMs) || 0);
    if (seg.audioMs === null && Number.isFinite(ev.endMs - ev.startMs)) seg.audioMs = Math.max(0, ev.endMs - ev.startMs);
    if (typeof ev.turnEnd === "boolean" && seg.turnEnd !== true) seg.turnEnd = ev.turnEnd;
    const f: FinalText = { text, lang: ev.lang ?? null, revision: ev.revision, turnEnd: typeof ev.turnEnd === "boolean" ? ev.turnEnd : null };
    if (second) seg.second = f;
    else {
      seg.first = f;
      seg.firstAt = now;
    }
    seg.partial = null;
    this.#pump();
  }

  utterance(ev: RustSpeechUtterance): void {
    if (this.#disposed || this.#finished.has(ev.segmentId) || this.#released.has(ev.segmentId)) return;
    const seg = this.#segment(ev.segmentId);
    if (ev.kind === "utterance") {
      seg.closedAt = this.#clock.now();
      if (Number.isFinite(ev.endMs - ev.startMs)) seg.audioMs = Math.max(0, ev.endMs - ev.startMs);
    }
    else if (ev.kind === "dropped") {
      if (seg.first || seg.second) return; // Rust drops only segments without a final
      seg.dropped = true;
      seg.partial = null;
    }
    this.#pump();
  }

  turnEnd(ev: Pick<RustSpeechTurnEnd, "segmentId">): void {
    if (this.#disposed) return;
    const seg = this.#segments.get(ev.segmentId);
    if (seg) seg.turnEnd = true;
  }

  /**
   * The session ended (its epoch is over): publish every held final (first-pass text when the replacement never
   * came) and pending final in order, and abandon interims that will never get one.
   */
  flush(): void {
    if (this.#disposed) return;
    this.#clearTimer();
    for (const id of [...this.#queue]) {
      const seg = this.#segments.get(id);
      if (!seg) continue;
      const f = seg.second ?? seg.first;
      if (f) this.#commit(seg, f);
      else {
        if (seg.shown >= 0) this.#sink.abandon(this.#meta(seg, seg.shown, null));
        this.#finish(seg);
      }
    }
    this.dispose();
  }

  /**
   * Publish now the finals waiting at the head of the queue, in order (a first-pass final held for its replacement
   * goes out with its first-pass text; a replacement arriving later is ignored), and keep the session going. Stops at
   * the first segment still without a final: its sentence is really open. Nothing new is shown as an interim here,
   * so a caller about to close the open sentence (mute, stop) only closes one that is still being spoken. Returns
   * whether a final was published.
   */
  commitHeld(): boolean {
    if (this.#disposed) return false;
    let committed = false;
    while (this.#queue.length > 0) {
      const seg = this.#segments.get(this.#queue[0]);
      if (!seg) {
        this.#queue.shift();
        continue;
      }
      const f = seg.second ?? seg.first;
      if (!f) break;
      this.#commit(seg, f);
      committed = true;
    }
    return committed;
  }

  /** Stop without publishing anything more. */
  dispose(): void {
    this.#disposed = true;
    this.#clearTimer();
    this.#segments.clear();
    this.#queue = [];
  }

  // --- internals ------------------------------------------------------------------------------------------------

  #segment(id: number, append = false): Segment {
    let seg = this.#segments.get(id);
    if (seg) return seg;
    seg = {
      id,
      lastRevision: -1,
      shown: -1,
      shownText: "",
      partial: null,
      first: null,
      second: null,
      dropped: false,
      turnEnd: null,
      closedAt: null,
      audioMs: null,
      firstAt: null,
      waitingSince: null,
    };
    this.#segments.set(id, seg);
    // Segment order; a late final of a given-up segment goes to the end instead (after what is already shown).
    const at = append ? -1 : this.#queue.findIndex((q) => q > id);
    if (at < 0) this.#queue.push(id);
    else this.#queue.splice(at, 0, id);
    return seg;
  }

  #meta(seg: Segment, revision: number, lang: string | null, turnEnd?: boolean | null): SttTextMeta {
    const meta: SttTextMeta = { segmentId: `${this.#stream}:${this.epoch}:${seg.id}`, revision };
    const l = normalizeLanguage(lang);
    if (l) meta.lang = l;
    if (typeof turnEnd === "boolean") meta.turnEnd = turnEnd;
    return meta;
  }

  #show(seg: Segment, c: Candidate): void {
    if (c.revision <= seg.shown) return;
    if (c.text !== seg.shownText) this.#sink.interim(c.text, this.#meta(seg, c.revision, c.lang));
    seg.shown = c.revision;
    seg.shownText = c.text;
    if (seg.partial === c) seg.partial = null;
  }

  #commit(seg: Segment, f: FinalText): void {
    this.#sink.final(f.text, this.#meta(seg, f.revision, f.lang, seg.turnEnd ?? f.turnEnd));
    this.#finish(seg);
  }

  #finish(seg: Segment): void {
    this.#segments.delete(seg.id);
    this.#queue = this.#queue.filter((id) => id !== seg.id);
    this.#remember(this.#finished, seg.id);
  }

  #release(seg: Segment): void {
    if (seg.shown >= 0) this.#sink.abandon(this.#meta(seg, seg.shown, null));
    this.#segments.delete(seg.id);
    this.#queue = this.#queue.filter((id) => id !== seg.id);
    this.#remember(this.#released, seg.id);
  }

  #remember(set: Set<number>, id: number): void {
    set.add(id);
    if (set.size > FINISHED_MEMORY) {
      const oldest = set.values().next().value;
      if (oldest !== undefined) set.delete(oldest);
    }
  }

  #setHolding(holding: boolean): void {
    if (this.#holding === holding) return;
    this.#holding = holding;
    this.#onHoldingChange?.(holding);
  }

  #replacementSeen(): void {
    this.#misses = 0;
    if (this.#twoPass) this.#setHolding(true);
  }

  #holdMissed(): void {
    this.#misses += 1;
    if (this.#misses >= MAX_HOLD_MISSES) this.#setHolding(false);
  }

  /** When a held first-pass final of `seg` gives up on its replacement. */
  #holdDeadline(seg: Segment, now: number): number {
    const firstAt = seg.firstAt ?? now;
    const closedAt = seg.closedAt ?? firstAt;
    const audioMs = seg.audioMs ?? 0;
    const budget = Math.min(TWO_PASS_MAX_BUDGET_MS, TWO_PASS_BUDGET_MS + (audioMs * TWO_PASS_BUDGET_PER_AUDIO_SECOND_MS) / 1000);
    return Math.max(closedAt + budget + HOLD_MARGIN_MS, firstAt + MIN_HOLD_MS);
  }

  /** Earliest time a segment behind the head started waiting with text to show (null: none waits). */
  #blockedSince(now: number): number | null {
    let since: number | null = null;
    for (const id of this.#queue.slice(1)) {
      const s = this.#segments.get(id);
      if (!s) continue;
      if (s.partial || s.first || s.second) {
        if (s.waitingSince === null) s.waitingSince = now;
        since = since === null ? s.waitingSince : Math.min(since, s.waitingSince);
      }
    }
    return since;
  }

  #pump(): void {
    if (this.#disposed) return;
    this.#clearTimer();
    const now = this.#clock.now();
    let wakeAt = Infinity;
    while (this.#queue.length > 0) {
      const seg = this.#segments.get(this.#queue[0]);
      if (!seg) {
        this.#queue.shift();
        continue;
      }
      if (seg.second) {
        this.#commit(seg, seg.second);
        continue;
      }
      const blocked = this.#blockedSince(now);
      if (seg.first) {
        if (!this.#holding) {
          this.#commit(seg, seg.first);
          continue;
        }
        const deadline = this.#holdDeadline(seg, now);
        if (now >= deadline) {
          this.#holdMissed();
          this.#commit(seg, seg.first);
          continue;
        }
        if (blocked !== null && now - blocked >= BLOCKED_HOLD_MS) {
          this.#commit(seg, seg.first);
          continue;
        }
        this.#show(seg, seg.first);
        wakeAt = Math.min(deadline, blocked !== null ? blocked + BLOCKED_HOLD_MS : Infinity);
        break;
      }
      if (seg.dropped) {
        if (seg.shown >= 0) this.#sink.abandon(this.#meta(seg, seg.shown, null));
        this.#finish(seg);
        continue;
      }
      if (seg.partial) this.#show(seg, seg.partial);
      if (blocked !== null) {
        if (now - blocked >= STALL_MS) {
          this.#release(seg);
          continue;
        }
        wakeAt = blocked + STALL_MS;
      }
      break;
    }
    if (Number.isFinite(wakeAt)) this.#timer = this.#clock.setTimeout(() => {
      this.#timer = null;
      this.#pump();
    }, Math.max(0, wakeAt - now));
  }

  #clearTimer(): void {
    if (this.#timer !== null) {
      this.#clock.clearTimeout(this.#timer);
      this.#timer = null;
    }
  }
}

// ---------------------------------------------------------------------------------------------------------------
// Driver

/** The parts of `Service_Models` the driver uses (`window.ApiServer.models`). */
export interface LocalModelsApi {
  readonly state: {
    catalog: RustCatalogEntry[];
    states: Record<string, RustModelEntryStatus>;
    progress: Record<string, EntryProgress>;
    pending: Record<string, PendingOp>;
    opErrors: Record<string, string>;
    loaded: boolean;
    error: string | null;
    configProblem?: unknown;
  };
  load(): Promise<void>;
  install(ids: readonly string[]): Promise<boolean>;
  cancel(id: string): Promise<boolean>;
  refreshStates(): Promise<void>;
}

/** Everything the driver reaches outside itself (tests swap it). */
export interface LocalDriverDeps {
  /** Tauri is there (the desktop app). */
  available(): boolean;
  invoke<T>(cmd: string, args?: Record<string, unknown>): Promise<T>;
  /** Listen to a Tauri event until the returned function is called (rejects when `signal` aborts first). */
  listen<T>(event: string, handler: (payload: T) => void, signal: AbortSignal | null | undefined): Promise<UnlistenFn>;
  models(): LocalModelsApi | null;
  /** A custom model mirror is active (unpublished entries can download). */
  customMirror(): boolean;
  askConsent(req: ConsentRequest): Promise<ConsentChoice>;
  /** Persist a change of the `stt.local` settings. */
  patchLocal(fn: (local: STT_State["local"]) => void): void;
  /** Capture device for the hub (`null` = the default microphone). */
  resolveDevice(device: string): string | null;
  log(line: string, level: "info" | "warning"): void;
  clock: SegmentClock;
}

const defaultDeps: LocalDriverDeps = {
  available: () => typeof window !== "undefined" && !!window.Config?.isApp?.(),
  invoke: (cmd, args) => invoke(cmd, args),
  listen: (event, handler, signal) => listenWithAbort(event, (e) => handler(e.payload as never), signal),
  models: () => (typeof window !== "undefined" ? (window.ApiServer?.models as unknown as LocalModelsApi) ?? null : null),
  customMirror: () =>
    typeof window !== "undefined" &&
    window.ApiServer?.state?.modelMirror === "custom" &&
    !window.ApiServer?.models?.state?.configProblem,
  askConsent: askConsentDialog,
  patchLocal: (fn) => window.ApiServer.patchService("stt", (s) => fn(s.data.local)),
  resolveDevice: (device) => resolveRustInputDeviceForInvoke(device),
  log: (line, level) => pushSystemLog("STT", line, level),
  clock: systemClock,
};

class AbortedError extends DOMException {
  constructor() {
    super("Aborted", "AbortError");
  }
}

export class STT_LocalService implements ISTTService {
  startAbortSignal?: AbortSignal | null;
  startAutomatic?: boolean;

  readonly #deps: LocalDriverDeps;
  readonly #lifecycle: SttStartLifecycle;
  #unlisteners: UnlistenFn[] = [];
  #publisher: SegmentPublisher | null = null;
  /** Epoch of the running session; null while a (re)start waits for its result. */
  #epoch: number | null = null;
  /** Events of epochs at or below this are never accepted. */
  #floor = 0;
  #config: RustSpeechPipelineConfig | null = null;
  #twoPass = false;
  /** A session is live (started, not failed or stopped). */
  #running = false;
  #stopRequested = false;
  #restartTimer: unknown = null;
  /** Restarts in the current failing streak (see {@link STABLE_RUN_MS}). */
  #restartAttempts = 0;
  /** Local time of the last successful (re)start. */
  #listeningSince = 0;
  /** Plan of the last successful (re)start (kept after a failure: it decides whether a restart can help). */
  #plan: RustSpeechRoutePlan | null = null;
  /** A `start_pipeline` call in flight (a stop waits for it and stops what it started). */
  #pendingStart: Promise<RustSpeechPipelineStarted> | null = null;
  /**
   * A fatal `speech:status` of a session whose start result has not arrived yet (events and command results travel
   * separately): handled once the result names that epoch.
   */
  #earlyFatal: RustSpeechStatus | null = null;
  /** Downloads this session queued (a stop cancels those still running). */
  readonly #queuedDownloads = new Set<string>();

  constructor(private readonly receiver: ISTTReceiver, deps: Partial<LocalDriverDeps> = {}) {
    this.#deps = { ...defaultDeps, ...deps };
    this.#lifecycle = new SttStartLifecycle((error) => this.receiver.onStop(error));
  }

  // --- lifecycle --------------------------------------------------------------------------------------------------

  async start(params: STT_State): Promise<void> {
    this.#lifecycle.beginStart();
    this.#stopRequested = false;
    this.#restartAttempts = 0;
    this.#plan = null;
    localSpeechState.restarts = 0;
    const signal = this.startAbortSignal ?? null;
    try {
      if (!this.#deps.available()) throw new Error(i18n.t("stt.local_error_desktop_only"));
      this.#floor = lastEpoch.get(LOCAL_PIPELINE_ID) ?? 0;
      this.#epoch = null;
      await this.#attachListeners(signal);
      this.#throwIfAborted(signal);

      let started: RustSpeechPipelineStarted | null = null;
      // A second round only when the pipeline, planning again at start, wants a model the first round did not
      // install (the PC changed between the preview and the start, e.g. the voice changer started meanwhile).
      for (let round = 0; round < 2 && !started; round++) {
        const { plan, request } = await this.#prepare(params, signal);
        this.#throwIfAborted(signal);
        // Early events of the new session (before its result) are published with the planned pass structure.
        this.#twoPass = isTwoPassPlan(plan);
        const config = buildLocalPipelineConfig(params.local, params.vad, request, this.#deps.resolveDevice(params.local?.device ?? ""));
        this.#config = config;
        this.receiver.onStatus(i18n.t("stt.local_status_starting"));
        try {
          started = await this.#startPipeline(config, signal);
        } catch (e) {
          if (isAbortError(e) || this.#lifecycle.aborted) throw e;
          if (round === 0 && notInstalledId(errorText(e)) && !this.startAutomatic) {
            this.#floor = lastEpoch.get(LOCAL_PIPELINE_ID) ?? this.#floor;
            continue;
          }
          throw new Error(startErrorMessage(e, (id) => this.#title(id)));
        }
      }
      this.#throwIfAborted(signal);
      if (!started) throw new Error(i18n.t("stt.local_error_no_engine"));
      this.#onStarted(started);
      this.receiver.onStart();
      this.#takeEarlyFatal(started.epoch);
    } catch (error) {
      if (isAbortError(error) || this.#lifecycle.aborted) {
        this.#detach();
        return;
      }
      this.#detach();
      // Downloads live only as long as the start that asked for them (the Models tab is for background ones).
      this.#cancelDownloads();
      this.receiver.onStatus("");
      this.#lifecycle.notifyStopOnce(errorText(error));
    } finally {
      this.#lifecycle.finishStart();
    }
  }

  async stop(): Promise<void> {
    const hadWork = this.#lifecycle.starting || this.#running || this.#restartTimer !== null;
    this.#stopRequested = true;
    this.#lifecycle.abort();
    this.#clearRestart();
    this.#detach();
    this.#cancelDownloads();
    const pending = this.#pendingStart;
    await this.#stopPipeline();
    if (pending) {
      // A start the stop may have overtaken (the command reached Rust after `stop_pipeline`): stop what it started.
      // Service_STT starts the next session only after this resolves, so nothing else owns the pipeline meanwhile.
      let timer: ReturnType<typeof setTimeout> | undefined;
      const settled = await Promise.race([
        pending.then(() => true, () => false),
        new Promise<boolean>((resolve) => { timer = setTimeout(() => resolve(false), ORPHAN_START_WAIT_MS); }),
      ]);
      clearTimeout(timer);
      if (settled) await this.#stopPipeline();
    }
    this.#running = false;
    localSpeechState.plan = null;
    localSpeechState.restarts = 0;
    if (hadWork) this.#lifecycle.notifyStopOnce();
  }

  dispose(): void {
    void this.stop();
  }

  /**
   * Service_STT is about to close the open sentence (mute, stop, a new start): publish the finished sentences whose
   * first-pass final is still held for its two-pass replacement, so the "[...]" marker never replaces one.
   */
  commitPending(): void {
    if (this.#stopRequested) return;
    this.#publisher?.commitHeld();
  }

  // --- start steps ------------------------------------------------------------------------------------------------

  #throwIfAborted(signal: AbortSignal | null): void {
    if (this.#lifecycle.aborted || this.#stopRequested || signal?.aborted) throw new AbortedError();
  }

  async #attachListeners(signal: AbortSignal | null): Promise<void> {
    this.#detachListeners();
    const on = async <T extends RustSpeechEnvelope>(event: string, handle: (ev: T) => void) => {
      this.#unlisteners.push(await this.#deps.listen<T>(event, (ev) => {
        if (this.#accepts(ev)) handle(ev);
      }, signal));
    };
    await on<RustSpeechStatus>(SPEECH_EVENTS.status, (ev) => this.#onStatus(ev));
    await on<RustSpeechPartial>(SPEECH_EVENTS.partial, (ev) => this.#publisherFor(ev.epoch).partial(ev));
    await on<RustSpeechFinal>(SPEECH_EVENTS.final, (ev) => this.#publisherFor(ev.epoch).final(ev));
    await on<RustSpeechUtterance>(SPEECH_EVENTS.utterance, (ev) => this.#publisherFor(ev.epoch).utterance(ev));
    await on<RustSpeechTurnEnd>(SPEECH_EVENTS.turnEnd, (ev) => this.#publisherFor(ev.epoch).turnEnd(ev));
  }

  /** The plan to run and the models it needs, installed (with consent). */
  async #prepare(params: STT_State, signal: AbortSignal | null): Promise<{ plan: RustSpeechRoutePlan; request: Required<RustSpeechRouteRequest> }> {
    this.receiver.onStatus(i18n.t("stt.local_status_checking"));
    const request = buildLocalRouteRequest(params.local);
    const plan = await this.#preview(request, signal);
    if (!planHasEngine(plan)) throw new Error(i18n.t("stt.local_error_no_engine"));
    return this.#ensureModels(plan, request, params.local?.smartTurn === true, signal);
  }

  #preview(request: RustSpeechRouteRequest, signal: AbortSignal | null): Promise<RustSpeechRoutePlan> {
    return raceWithAbort(this.#deps.invoke<RustSpeechRoutePlan>("plugin:speech|preview_route", { request }), signal);
  }

  #summary(ids: readonly string[]): RequirementSummary {
    const s = this.#deps.models()!.state;
    return summarizeRequirement(s.catalog, s.states, ids, s.pending);
  }

  #title(id: string): string {
    const entry = this.#deps.models()?.state.catalog.find((e) => e.id === id);
    return entry ? entryTitle(entry) : id;
  }

  async #ensureModels(
    plan: RustSpeechRoutePlan,
    request: Required<RustSpeechRouteRequest>,
    smartTurn: boolean,
    signal: AbortSignal | null,
  ): Promise<{ plan: RustSpeechRoutePlan; request: Required<RustSpeechRouteRequest> }> {
    // Rust already knows what the plan needs; the store is asked only when something is missing (or for Smart Turn,
    // which is not part of the plan). An automatic start never asks: without Smart Turn the pipeline only warns.
    const withSmartTurn = smartTurn && !this.startAutomatic;
    if (plan.missing.length === 0 && !withSmartTurn) return { plan, request };
    const ids = requiredModelIds(plan, withSmartTurn);
    const models = this.#deps.models();
    if (!models) {
      if (plan.missing.length > 0) throw new Error(i18n.t("stt.local_error_models_missing", { size: "?" }));
      return { plan, request };
    }
    await raceWithAbort(models.load(), signal);
    this.#throwIfAborted(signal);
    if (!models.state.loaded) throw new Error(i18n.t("stt.local_error_catalog", { error: models.state.error ?? "" }));
    const summary = this.#summary(ids);
    if (summary.unknown.length > 0) throw new Error(i18n.t("stt.local_error_unknown_models", { ids: summary.unknown.join(", ") }));
    if (summary.ready) return { plan, request };
    this.#checkDownloadable(summary);

    let chosen = { plan, request, ids };
    const toFetch = summary.missing.filter((e) => !isBusyState(models.state.states[e.id]?.state));
    if (toFetch.length > 0) {
      const full = consentSet(toFetch);
      if (this.startAutomatic) throw new Error(i18n.t("stt.local_error_models_missing", { size: sizeLabel(full) || "?" }));
      const smaller = await this.#smallerSetup(plan, request, smartTurn, full, signal);
      this.#throwIfAborted(signal);
      this.receiver.onStatus(i18n.t("stt.local_status_consent"));
      const choice = await this.#deps.askConsent({ ...full, smaller: smaller?.set ?? null });
      this.#throwIfAborted(signal);
      if (choice === "cancel") throw new Error(i18n.t("stt.local_error_declined"));
      if (choice === "smaller" && smaller) {
        this.#deps.patchLocal((local) => { local.twoPass = false; });
        this.#deps.log(i18n.t("stt.local_log_smaller_setup"), "info");
        chosen = { plan: smaller.plan, request: smaller.request, ids: smaller.ids };
      }
    }
    await this.#installAndWait(chosen.ids, signal);
    return { plan: chosen.plan, request: chosen.request };
  }

  /** Entries only an import can install stop the start with a pointer to Models & Runtimes. */
  #checkDownloadable(summary: RequirementSummary): void {
    const importOnly = summary.missing.filter((e) => !canDownloadEntry(e, this.#deps.customMirror()));
    if (importOnly.length > 0)
      throw new Error(i18n.t("stt.local_error_import_needed", { list: importOnly.map((e) => entryTitle(e)).join(", ") }));
  }

  /** The single-pass plan when it downloads clearly less than the two-pass one. */
  async #smallerSetup(
    plan: RustSpeechRoutePlan,
    request: Required<RustSpeechRouteRequest>,
    smartTurn: boolean,
    full: ConsentSet,
    signal: AbortSignal | null,
  ): Promise<{ plan: RustSpeechRoutePlan; request: Required<RustSpeechRouteRequest>; ids: string[]; set: SmallerConsentSet } | null> {
    if (!request.twoPass || !isTwoPassPlan(plan) || full.bytes <= 0) return null;
    const altRequest = { ...request, twoPass: false };
    let altPlan: RustSpeechRoutePlan;
    try {
      altPlan = await this.#preview(altRequest, signal);
    } catch (e) {
      if (isAbortError(e)) throw e;
      return null;
    }
    if (!planHasEngine(altPlan)) return null;
    const ids = requiredModelIds(altPlan, smartTurn);
    const summary = this.#summary(ids);
    if (summary.unknown.length > 0) return null;
    const models = this.#deps.models()!;
    if (summary.missing.some((e) => !canDownloadEntry(e, this.#deps.customMirror()))) return null;
    const set = consentSet(summary.missing.filter((e) => !isBusyState(models.state.states[e.id]?.state)));
    if (!set.exact || set.bytes > full.bytes * SMALLER_SETUP_RATIO) return null;
    return { plan: altPlan, request: altRequest, ids, set: { ...set, live: !!altPlan.streaming } };
  }

  /** Queue the missing entries of `ids` (and wait for ones already on their way), reporting progress. */
  async #installAndWait(ids: readonly string[], signal: AbortSignal | null): Promise<void> {
    const models = this.#deps.models()!;
    const summary = this.#summary(ids);
    if (summary.ready) return;
    const before = new Map(summary.missing.map((e) => [e.id, models.state.states[e.id]]));
    const toFetch = summary.missing.filter((e) => !isBusyState(models.state.states[e.id]?.state)).map((e) => e.id);
    const tracked = summary.missing;
    const clock = this.#deps.clock;
    const installAt = clock.now();
    if (toFetch.length > 0) {
      this.receiver.onStatus(i18n.t("stt.local_status_downloading", { title: this.#title(toFetch[0]) }));
      const ok = await models.install(toFetch);
      toFetch.forEach((id) => this.#queuedDownloads.add(id));
      if (this.#lifecycle.aborted || this.#stopRequested || signal?.aborted) {
        // The stop ran while the downloads were being queued: do not leave them running.
        this.#cancelDownloads();
        throw new AbortedError();
      }
      if (!ok) {
        const id = toFetch.find((i) => models.state.opErrors[i]) ?? toFetch[0];
        throw new Error(i18n.t("stt.local_error_download_failed", { title: this.#title(id), error: models.state.opErrors[id] ?? "?" }));
      }
    }
    const sawBusy = new Set<string>();
    await new Promise<void>((resolve, reject) => {
      let done = false;
      let lastStatus: StatusLine | null = null;
      const finish = (fn: () => void) => {
        if (done) return;
        done = true;
        unsubscribe();
        clock.clearTimeout(keepAlive);
        signal?.removeEventListener("abort", onAbort);
        fn();
      };
      const check = () => {
        if (done) return;
        if (this.#lifecycle.aborted || this.#stopRequested) return finish(() => reject(new AbortedError()));
        const s = this.#summary(ids);
        if (s.ready) return finish(resolve);
        const states = models.state.states;
        for (const e of s.missing) {
          const st = states[e.id];
          if (isBusyState(st?.state)) sawBusy.add(e.id);
          // A failed / cancelled state counts once it is news: a new report after the entry was seen busy, not
          // queued by this start, or late enough that it cannot be the report from before `install()`.
          const changed = st !== undefined && st !== before.get(e.id);
          const news = sawBusy.has(e.id) || !toFetch.includes(e.id) || clock.now() - installAt > STALE_STATE_GRACE_MS;
          if (changed && news && (st.state === "failed" || st.state === "cancelled")) {
            const msg = st.state === "cancelled"
              ? i18n.t("stt.local_error_download_cancelled")
              : i18n.t("stt.local_error_download_failed", { title: entryTitle(e), error: st.error || "?" });
            return finish(() => reject(new Error(msg)));
          }
        }
        const stuck = s.missing.find((e) => !sawBusy.has(e.id) && !isBusyState(states[e.id]?.state));
        if (stuck && clock.now() - installAt > DOWNLOAD_START_TIMEOUT_MS) {
          const error = models.state.opErrors[stuck.id] ?? states[stuck.id]?.error ?? i18n.t("stt.local_error_download_not_started");
          return finish(() => reject(new Error(i18n.t("stt.local_error_download_failed", { title: entryTitle(stuck), error }))));
        }
        lastStatus = downloadStatusLine(tracked, s, states, models.state.progress);
        this.receiver.onStatus(...lastStatus);
      };
      const onAbort = () => finish(() => reject(new AbortedError()));
      const unsubscribe = subscribe(models.state, check);
      signal?.addEventListener("abort", onAbort, { once: true });
      // Downloads queued behind others (or a slow mirror) can go quiet for long: the store has its own stall and
      // failure handling, so re-state the line (keeps the connecting watchdog alive) and re-check the states.
      const tick = () => {
        if (done) return;
        if (lastStatus) this.receiver.onStatus(...lastStatus);
        void models.refreshStates().then(check, check);
        keepAlive = clock.setTimeout(tick, DOWNLOAD_KEEPALIVE_MS);
      };
      let keepAlive = clock.setTimeout(tick, DOWNLOAD_KEEPALIVE_MS);
      check();
    });
    this.#queuedDownloads.clear();
  }

  /** `start_pipeline`, remembered while in flight so a stop can clean up after it. */
  async #startPipeline(config: RustSpeechPipelineConfig, signal: AbortSignal | null): Promise<RustSpeechPipelineStarted> {
    // Same key as the HTTP drivers: their late cleanup stop must never cancel this newer start.
    noteSttPipelineStart();
    const call = this.#deps.invoke<RustSpeechPipelineStarted>("plugin:speech|start_pipeline", { config });
    this.#pendingStart = call;
    const clear = () => {
      if (this.#pendingStart === call) this.#pendingStart = null;
    };
    call.then(clear, clear);
    const started = await raceWithAbort(call, signal);
    this.#throwIfAborted(signal);
    return started;
  }

  /** The session that just started had already failed (its fatal status came before the start result). */
  #takeEarlyFatal(epoch: number): void {
    const fatal = this.#earlyFatal;
    this.#earlyFatal = null;
    if (fatal && fatal.epoch === epoch && this.#running && !this.#stopRequested) this.#sessionFailed(fatal);
  }

  #onStarted(started: RustSpeechPipelineStarted): void {
    noteEpoch(LOCAL_PIPELINE_ID, started.epoch);
    this.#epoch = started.epoch;
    this.#twoPass = isTwoPassPlan(started.plan);
    const pub = this.#publisherFor(started.epoch);
    pub.setTwoPass(this.#twoPass);
    this.#running = true;
    this.#listeningSince = this.#deps.clock.now();
    this.#plan = started.plan;
    localSpeechState.plan = started.plan;
    const roles = planRoles(started.plan)
      .map((r) => `${this.#title(r.spec.model ?? "")}${specOnGpu(r.spec) ? " (GPU)" : ""}`)
      .join(", ");
    this.#deps.log(i18n.t("stt.local_log_started", { models: roles || "-" }), "info");
  }

  async #stopPipeline(): Promise<void> {
    try {
      await this.#deps.invoke("plugin:speech|stop_pipeline", { pipelineId: LOCAL_PIPELINE_ID });
    } catch {
      /* not running */
    }
  }

  /** Cancel the downloads this session queued that have not finished (their partial files are kept for resume). */
  #cancelDownloads(): void {
    const models = this.#deps.models();
    if (!models) return;
    for (const id of this.#queuedDownloads) {
      const state = models.state.states[id]?.state;
      // A just-queued entry may not have reported "queued" yet: anything not settled is cancelled.
      if (state !== "installed" && state !== "failed" && state !== "cancelled") void models.cancel(id);
    }
    this.#queuedDownloads.clear();
  }

  #detachListeners(): void {
    this.#unlisteners.forEach((u) => {
      try {
        u();
      } catch {
        /* ignore */
      }
    });
    this.#unlisteners = [];
  }

  #detach(): void {
    this.#detachListeners();
    this.#publisher?.dispose();
    this.#publisher = null;
    this.#epoch = null;
  }

  // --- events -----------------------------------------------------------------------------------------------------

  #accepts(env: RustSpeechEnvelope | null | undefined): boolean {
    if (!env || env.pipelineId !== LOCAL_PIPELINE_ID || typeof env.epoch !== "number") return false;
    if (this.#stopRequested) return false;
    noteEpoch(env.pipelineId, env.epoch);
    return this.#epoch !== null ? env.epoch === this.#epoch : env.epoch > this.#floor;
  }

  #publisherFor(epoch: number): SegmentPublisher {
    if (this.#publisher && this.#publisher.epoch === epoch) return this.#publisher;
    this.#publisher?.flush();
    this.#publisher = new SegmentPublisher({
      pipelineId: LOCAL_PIPELINE_ID,
      epoch,
      twoPass: this.#twoPass,
      clock: this.#deps.clock,
      sink: {
        interim: (text, meta) => this.receiver.onInterim(text, meta),
        final: (text, meta) => this.receiver.onFinal(text, meta),
        abandon: (meta) => this.receiver.onInterim("", meta),
      },
      onHoldingChange: (holding) =>
        this.#deps.log(i18n.t(holding ? "stt.local_log_two_pass_resumed" : "stt.local_log_two_pass_suspended"), holding ? "info" : "warning"),
    });
    return this.#publisher;
  }

  #onStatus(ev: RustSpeechStatus): void {
    const current = this.#epoch !== null && ev.epoch === this.#epoch;
    if (ev.state === "error") {
      if (current && this.#running) this.#sessionFailed(ev);
      // While a start waits for its result, the session may already have failed: a rejected `start_pipeline`
      // reports that itself, a resolved one is checked against this (`#takeEarlyFatal`).
      else if (this.#epoch === null) this.#earlyFatal = ev;
      return;
    }
    if (ev.state === "stopped") {
      if (current && this.#running) {
        // Another start of the same pipeline key replaced this session. Its epoch is over: publish what its segments
        // hold (a held first-pass final is a finished sentence) before the stop closes the open one.
        this.#running = false;
        this.#publisher?.flush();
        this.#publisher = null;
        this.#detach();
        localSpeechState.plan = null;
        this.#lifecycle.notifyStopOnce(i18n.t("stt.local_error_stopped_elsewhere"));
      }
      return;
    }
    // Load progress that arrives after the start resolved is stale: only warnings and the "listening" clear count.
    const setup = ev.code === "starting" || ev.code === "loading_model" || ev.code === "loading_runtime" || ev.code === "downloading";
    if (this.#running && setup && ev.level === "info") return;
    const line = speechStatusLine(ev, (id) => this.#title(id));
    if (line) this.receiver.onStatus(...line);
    else if (ev.code === "listening" && this.#running) this.receiver.onStatus("");
  }

  // --- automatic restart ------------------------------------------------------------------------------------------

  #sessionFailed(ev: RustSpeechStatus): void {
    // A session that really ran for a while since its last successful start begins a new failing streak. Only here:
    // `#restart`'s failed starts never ran, so they keep counting against the budget.
    if (this.#deps.clock.now() - this.#listeningSince >= STABLE_RUN_MS) this.#restartAttempts = 0;
    this.#running = false;
    localSpeechState.plan = null;
    // The epoch is over: publish what its segments have (no replacement can come any more).
    this.#publisher?.flush();
    this.#publisher = null;
    this.#floor = Math.max(this.#floor, ev.epoch);
    this.#epoch = null;
    const title = (id: string) => this.#title(id);
    const rawDetail = String(ev.detail ?? "").trim();
    const detail = rawDetail || ev.code;
    if (!restartCanFix(ev.code, rawDetail, this.#plan, this.#config?.route?.tier)) {
      const text = ev.code === "not_installed" && ev.modelId
        ? i18n.t("stt.local_error_not_installed", { model: title(ev.modelId) })
        : i18n.t("stt.local_error_failed", { detail });
      this.#fail(text);
      return;
    }
    this.#scheduleRestart(detail);
  }

  #scheduleRestart(detail: string): void {
    const clock = this.#deps.clock;
    this.#restartAttempts += 1;
    if (this.#restartAttempts > MAX_RESTARTS) {
      this.#fail(i18n.t("stt.local_error_gave_up", { attempts: MAX_RESTARTS, detail }));
      return;
    }
    const delay = RESTART_DELAYS_MS[Math.min(this.#restartAttempts - 1, RESTART_DELAYS_MS.length - 1)];
    const line = i18n.t("stt.local_status_restarting", { detail, seconds: Math.ceil(delay / 1000) });
    this.receiver.onStatus(line, { level: "warning" });
    this.#deps.log(line, "warning");
    this.#clearRestart();
    this.#restartTimer = clock.setTimeout(() => {
      this.#restartTimer = null;
      void this.#restart();
    }, delay);
  }

  async #restart(): Promise<void> {
    if (this.#stopRequested || !this.#config) return;
    localSpeechState.restarts += 1;
    try {
      const started = await this.#startPipeline(this.#config, null);
      if (this.#stopRequested) return;
      this.#onStarted(started);
      this.receiver.onStatus("");
      this.#deps.log(i18n.t("stt.local_log_restarted"), "info");
      this.#takeEarlyFatal(started.epoch);
    } catch (e) {
      if (this.#stopRequested || isAbortError(e)) return;
      const text = errorText(e);
      // A missing model, or an engine error that fails the same way on every start: stop now with the reason.
      if (!restartCanFix("", text, this.#plan, this.#config?.route?.tier)) {
        this.#fail(startErrorMessage(e, (id) => this.#title(id)));
        return;
      }
      this.#floor = lastEpoch.get(LOCAL_PIPELINE_ID) ?? this.#floor;
      this.#scheduleRestart(text);
    }
  }

  #clearRestart(): void {
    if (this.#restartTimer !== null) {
      this.#deps.clock.clearTimeout(this.#restartTimer);
      this.#restartTimer = null;
    }
  }

  /** The session cannot go on: release everything and report it. */
  #fail(text: string): void {
    this.#running = false;
    this.#clearRestart();
    this.#detach();
    localSpeechState.plan = null;
    this.#deps.log(text, "warning");
    this.receiver.onStatus("");
    this.#lifecycle.notifyStopOnce(text);
  }
}

// ---------------------------------------------------------------------------------------------------------------
// Download helpers

/** Titles and sizes of `entries` for the consent dialog. */
export function consentSet(entries: readonly RustCatalogEntry[]): ConsentSet {
  let bytes = 0;
  let exact = true;
  const items = entries.map((e) => {
    const size = entrySize(e);
    bytes += size.bytes;
    exact &&= size.exact;
    return { title: entryTitle(e), bytes: size.bytes };
  });
  return { items, bytes, exact };
}

/** The status line while models download: overall percent and the entry in progress. */
export function downloadStatusLine(
  tracked: readonly RustCatalogEntry[],
  summary: RequirementSummary,
  states: Readonly<Record<string, RustModelEntryStatus | undefined>>,
  progress: Readonly<Record<string, EntryProgress | undefined>>,
): StatusLine {
  const fraction = requirementFraction(tracked, states, progress);
  const current = summary.busy.find((e) => states[e.id]?.state === "downloading") ?? summary.busy[0] ?? summary.missing[0];
  const title = current ? entryTitle(current) : "";
  if (current && states[current.id]?.state === "queued" && !summary.busy.some((e) => states[e.id]?.state !== "queued"))
    return [i18n.t("stt.local_status_waiting_download", { title }), { progress: fraction === null ? null : fraction * 100 }];
  if (fraction === null) return [i18n.t("stt.local_status_downloading", { title }), { progress: null }];
  const percent = Math.floor(fraction * 100);
  return [i18n.t("stt.local_status_downloading_percent", { percent, title }), { progress: fraction * 100 }];
}
