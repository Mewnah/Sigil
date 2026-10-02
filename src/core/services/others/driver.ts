/**
 * The other players' speech driver (P6-A): runs the Rust speech pipeline under the `loopback` key on the source the
 * Rust planner picks (`speech::session`), and turns its `speech:*` events into transcripts for {@link OthersSink}.
 *
 * Start:
 * 1. `plugin:speech|preview_route` for the Others language / tier / two-pass: models must already be installed (the
 *    Other players panel offers them with `<ModelRequirement>`; nothing downloads from here).
 * 2. `plugin:audio|list_loopback_processes { source, fallback }` returns the source plan: the program
 *    (`process:VRChat.exe`) when it runs, else the default output's loopback (fallback on) or nothing yet (fallback
 *    off: wait for the program); old Windows falls back to the default output, or fails with the fallback off.
 * 3. `plugin:speech|start_pipeline` with `pipelineId: "loopback"` on that source.
 *
 * Running: while the plan watches a program, the plan is asked again every {@link WATCH_INTERVAL_MS}; when the
 * program starts or exits the pipeline moves to the new source (a new epoch; the old session's held finals are
 * published first), or stops and waits for it (fallback off). A program that runs but whose capture cannot be opened
 * falls back to the default output (fallback on) and is tried again after {@link CAPTURE_RETRY_DELAYS_MS}.
 * Events of the current session go through the STT {@link SegmentPublisher} (segment ids
 * `loopback-<nonce>:<epoch>:<n>`). A pipeline that stops itself with a retryable error restarts with back-off (the
 * STT driver's rules: {@link restartCanFix}, {@link RESTART_DELAYS_MS}); a restart plans the source again.
 *
 * Never touches the user's own `mic` pipeline, its mute group or its start bookkeeping.
 */
import type { UnlistenFn } from "@tauri-apps/api/event";
import { invoke } from "@tauri-apps/api/core";
import i18n from "i18next";
import { proxy } from "valtio";
import {
  ServiceNetworkState,
  SPEECH_EVENTS,
  type RustSpeechEnvelope,
  type RustSpeechFinal,
  type RustSpeechPartial,
  type RustSpeechPipelineConfig,
  type RustSpeechPipelineStarted,
  type RustSpeechPipelineStatus,
  type RustSpeechRoutePlan,
  type RustSpeechRouteRequest,
  type RustSpeechStatus,
  type RustSpeechTurnEnd,
  type RustSpeechUtterance,
} from "@/types";
import { isAbortError, listenWithAbort, raceWithAbort } from "@/core/services/stt/abort";
import { isTwoPassPlan, normalizeLanguage, planHasEngine, routeRequest } from "@/core/services/stt/router";
import {
  MAX_RESTARTS,
  notInstalledId,
  RESTART_DELAYS_MS,
  restartCanFix,
  SegmentPublisher,
  speechStatusLine,
  STABLE_RUN_MS,
  startErrorMessage,
  type SegmentClock,
} from "@/core/services/stt/services/local";
import type { SttTextMeta } from "@/core/services/stt/types";
import type { Others_State } from "./schema";
import {
  isProcessSource,
  LIST_LOOPBACK_PROCESSES,
  LOOPBACK_DEFAULT_SOURCE,
  parsePlan,
  processExe,
  type OthersProcessList,
  type OthersSourcePlan,
} from "./source";

/** Pipeline key of other players (Rust `speech::engine_api::PIPELINE_LOOPBACK`). */
export const OTHERS_PIPELINE_ID = "loopback";
/** How often a watched program is checked for (start / exit). */
export const WATCH_INTERVAL_MS = 5_000;
/** Closed utterances waiting longer than this are dropped (stale captions of other people help nobody). */
export const OTHERS_LATENCY_BUDGET_MS = 12_000;
/** How long a stop waits for a `start_pipeline` call still in flight before giving up on it. */
const ORPHAN_START_WAIT_MS = 10_000;
/**
 * Back-off before a running program whose capture failed to open is tried again (fallback on; the default output is
 * captured meanwhile). Each failure in a row takes the next delay; the program exiting or a new start resets it.
 */
export const CAPTURE_RETRY_DELAYS_MS = [30_000, 60_000, 120_000, 300_000] as const;

/** What the driver is doing. */
export type OthersPhase = "idle" | "starting" | "running" | "waiting" | "restarting";

/** Live state for the UI (`Service_Others.serviceState`). */
export interface OthersDriverState {
  /** Service status: connecting while starting / restarting, connected while running or waiting for the program. */
  status: ServiceNetworkState;
  phase: OthersPhase;
  /** Why the last session stopped ("" = it did not fail). */
  error: string;
  /** Engine / capture status line (UI only, never a caption). */
  statusText: string;
  statusLevel: "info" | "warning";
  /** 0–100 while models load; null otherwise. */
  progress: number | null;
  /** Capture source in use (null while waiting or stopped). */
  source: string | null;
  /** The latest source plan. */
  plan: OthersSourcePlan | null;
  /** Engines of the running session (null while none runs). */
  routePlan: RustSpeechRoutePlan | null;
  /** Automatic restarts in this session. */
  restarts: number;
}

/** Where transcripts go (`OthersPublisher` in production). */
export interface OthersSink {
  interim(text: string, meta: SttTextMeta): void;
  final(text: string, meta: SttTextMeta): void;
}

/** Everything the driver reaches outside itself (tests swap it). */
export interface OthersDriverDeps {
  available(): boolean;
  invoke<T>(cmd: string, args?: Record<string, unknown>): Promise<T>;
  /** Listen to a Tauri event until the returned function is called (rejects when `signal` aborts first). */
  listen<T>(event: string, handler: (payload: T) => void, signal: AbortSignal | null | undefined): Promise<UnlistenFn>;
  /** Display name of a catalog id. */
  title(id: string): string;
  log(line: string, level: "info" | "warning"): void;
  clock: SegmentClock;
}

const systemClock: SegmentClock = {
  now: () => Date.now(),
  setTimeout: (fn, ms) => setTimeout(fn, ms),
  clearTimeout: (h) => clearTimeout(h as ReturnType<typeof setTimeout>),
};

const defaultDeps: OthersDriverDeps = {
  available: () => typeof window !== "undefined" && !!window.Config?.isApp?.(),
  invoke: (cmd, args) => invoke(cmd, args),
  listen: (event, handler, signal) => listenWithAbort(event, (e) => handler(e.payload as never), signal),
  title: (id) => id,
  log: () => {},
  clock: systemClock,
};

/** The router request of the Others settings. */
export function othersRouteRequest(s: Pick<Others_State, "language" | "tier" | "twoPass">): Required<RustSpeechRouteRequest> {
  return routeRequest({
    language: normalizeLanguage(s.language) ?? "auto",
    tier: s.tier,
    twoPass: s.twoPass !== false,
  });
}

/** `plugin:speech|start_pipeline { config }` of the other players' pipeline on `source`. */
export function buildOthersPipelineConfig(
  s: Pick<Others_State, "musicGate">,
  request: RustSpeechRouteRequest,
  source: string,
): RustSpeechPipelineConfig {
  const route = routeRequest(request);
  return {
    pipelineId: OTHERS_PIPELINE_ID,
    source,
    // Rust's voice detection defaults: the user's microphone tuning does not describe a game's mix.
    vad: {},
    language: route.language,
    route,
    smartTurn: false,
    hotwords: [],
    hallucinationFilter: true,
    latencyBudgetMs: OTHERS_LATENCY_BUDGET_MS,
    musicGate: s.musicGate === true,
  };
}

/** Status line text for a plan that captures something else than the requested program ("" = none). */
export function planNote(plan: OthersSourcePlan | null): { text: string; level: "info" | "warning" } {
  if (!plan?.fallback) return { text: "", level: "info" };
  const program = plan.watch ?? plan.requested.replace(/^process:/i, "");
  if (plan.fallback === "capture_failed") {
    return { text: i18n.t("others.status_fallback_capture_failed", { program }), level: "warning" };
  }
  if (plan.fallback === "unsupported") {
    return plan.source
      ? { text: i18n.t("others.status_fallback_unsupported"), level: "warning" }
      : { text: i18n.t("others.error_unsupported"), level: "warning" };
  }
  return plan.source
    ? { text: i18n.t("others.status_fallback_not_running", { program }), level: "info" }
    : { text: i18n.t("others.status_waiting", { program }), level: "info" };
}

/** `plan` with the default output captured instead of its program (whose capture failed to open). */
function captureFailedPlan(plan: OthersSourcePlan): OthersSourcePlan {
  return {
    ...plan,
    source: LOOPBACK_DEFAULT_SOURCE,
    fallback: "capture_failed",
    watch: plan.watch ?? (processExe(plan.source) || null),
  };
}

const sameSource = (a: string | null | undefined, b: string | null | undefined) =>
  (a ?? "").trim().toLowerCase() === (b ?? "").trim().toLowerCase();

/** A start failure of the models or engines (another source cannot fix it), not of the capture. */
const engineError = (text: string) => notInstalledId(text) !== null || /no speech engine for this request/i.test(text);

class AbortedError extends DOMException {
  constructor() {
    super("Aborted", "AbortError");
  }
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

export class OthersSpeechDriver {
  readonly state = proxy<OthersDriverState>({
    status: ServiceNetworkState.disconnected,
    phase: "idle",
    error: "",
    statusText: "",
    statusLevel: "info",
    progress: null,
    source: null,
    plan: null,
    routePlan: null,
    restarts: 0,
  });

  readonly #deps: OthersDriverDeps;
  readonly #sink: OthersSink;
  readonly #onStopped: (error?: string) => void;
  /** Bumped by every start and stop: async work of an older run must not touch anything. */
  #run = 0;
  #abort: AbortController | null = null;
  #settings: Others_State | null = null;
  #request: Required<RustSpeechRouteRequest> | null = null;
  #unlisteners: UnlistenFn[] = [];
  #publisher: SegmentPublisher | null = null;
  /** Epoch of the running session; null while a (re)start waits for its result. */
  #epoch: number | null = null;
  /** Events of epochs at or below this are never accepted. */
  #floor = 0;
  /** Latest epoch seen on the key. */
  #lastEpoch = 0;
  #twoPass = false;
  #pendingStart: Promise<RustSpeechPipelineStarted> | null = null;
  #earlyFatal: RustSpeechStatus | null = null;
  #watchTimer: unknown = null;
  #restartTimer: unknown = null;
  #restartAttempts = 0;
  #runningSince = 0;
  /** A plan change is being applied (watch ticks wait for it). */
  #applying = false;
  /** Epoch of a `loopback` session found running at start (an earlier webview's, or this driver's last run). */
  #leftoverEpoch: number | null = null;
  /** A running program whose capture failed to open: captured again from `retryAt` on (fallback on). */
  #captureFailure: { source: string; attempts: number; retryAt: number } | null = null;

  constructor(sink: OthersSink, onStopped: (error?: string) => void, deps: Partial<OthersDriverDeps> = {}) {
    this.#sink = sink;
    this.#onStopped = onStopped;
    this.#deps = { ...defaultDeps, ...deps };
  }

  get running(): boolean {
    return this.state.phase !== "idle";
  }

  // --- lifecycle --------------------------------------------------------------------------------------------------

  /** Start (or restart with new settings). Resolves once running, waiting for the program, or failed. */
  async start(settings: Others_State): Promise<void> {
    // A running session is replaced by the new start (`start_pipeline` replaces the key's session; its epoch is
    // refused from here on).
    const run = this.#newRun();
    this.#settings = { ...settings };
    this.#restartAttempts = 0;
    // A start by the user tries the program at once.
    this.#captureFailure = null;
    this.#leftoverEpoch = null;
    Object.assign(this.state, {
      status: ServiceNetworkState.connecting,
      phase: "starting" as OthersPhase,
      error: "",
      progress: null,
      source: null,
      plan: null,
      routePlan: null,
      restarts: 0,
    });
    this.#setStatus(i18n.t("others.status_starting"));
    const signal = this.#abort!.signal;
    try {
      if (!this.#deps.available()) throw new Error(i18n.t("others.error_desktop_only"));
      const request = othersRouteRequest(settings);
      const preview = await raceWithAbort(this.#deps.invoke<RustSpeechRoutePlan>("plugin:speech|preview_route", { request }), signal);
      this.#check(run);
      if (!planHasEngine(preview)) throw new Error(i18n.t("others.error_no_engine"));
      if (preview.missing.length > 0) {
        throw new Error(i18n.t("others.error_models_missing", { list: preview.missing.map((id) => this.#deps.title(id)).join(", ") }));
      }
      this.#request = request;
      this.#twoPass = isTwoPassPlan(preview);
      await this.#refuseRunningEpoch(signal);
      this.#check(run);
      await this.#attachListeners(signal);
      this.#check(run);
      const plan = await this.#planSource(signal);
      this.#check(run);
      await this.#apply(plan, run, signal);
      this.#check(run);
      this.#scheduleWatch(run);
    } catch (e) {
      if (isAbortError(e) || run !== this.#run) return;
      this.#fail(errorText(e));
    }
  }

  /** Stop everything; resolves once the pipeline is stopped. */
  async stop(): Promise<void> {
    const hadWork = this.running;
    const run = this.#newRun();
    this.#setIdle();
    const pending = this.#pendingStart;
    await this.#stopPipeline();
    if (pending) {
      // A start the stop may have overtaken: stop what it started.
      let timer: unknown;
      const settled = await Promise.race([
        pending.then(() => true, () => false),
        new Promise<boolean>((resolve) => {
          timer = this.#deps.clock.setTimeout(() => resolve(false), ORPHAN_START_WAIT_MS);
        }),
      ]);
      this.#deps.clock.clearTimeout(timer);
      // Unless a new start owns the key by now.
      if (settled && run === this.#run) await this.#stopPipeline();
    }
    if (hadWork && run === this.#run) this.#onStopped();
  }

  dispose(): void {
    void this.stop();
  }

  /** A new run: everything of the previous one is cancelled (timers, listeners, publisher, start waits). */
  #newRun(): number {
    const run = ++this.#run;
    this.#abort?.abort();
    this.#abort = new AbortController();
    this.#clearTimers();
    this.#detach(true);
    return run;
  }

  #check(run: number): void {
    if (run !== this.#run || this.#abort?.signal.aborted) throw new AbortedError();
  }

  #setIdle(): void {
    Object.assign(this.state, {
      status: ServiceNetworkState.disconnected,
      phase: "idle" as OthersPhase,
      statusText: "",
      statusLevel: "info" as const,
      progress: null,
      source: null,
      routePlan: null,
    });
  }

  #setStatus(text: string, level: "info" | "warning" = "info", progress: number | null = null): void {
    this.state.statusText = text;
    this.state.statusLevel = level;
    this.state.progress = progress;
  }

  /** The session cannot go on: release everything and report it. */
  #fail(text: string): void {
    this.#newRun();
    void this.#stopPipeline();
    this.#setIdle();
    this.state.error = text;
    this.#deps.log(text, "warning");
    this.#onStopped(text);
  }

  /**
   * A `loopback` session left running by an earlier webview (a reload), or by this driver's previous run, keeps
   * emitting until the new start replaces it: refuse its epoch from the start. A start replaces it through
   * `start_pipeline`; a first plan that waits for the program stops it (`#apply`).
   */
  async #refuseRunningEpoch(signal: AbortSignal | null): Promise<void> {
    try {
      const rows = await raceWithAbort(this.#deps.invoke<RustSpeechPipelineStatus[]>("plugin:speech|get_pipeline_status"), signal);
      const row = Array.isArray(rows) ? rows.find((r) => r?.pipelineId === OTHERS_PIPELINE_ID) : undefined;
      if (row && typeof row.epoch === "number") {
        this.#noteEpoch(row.epoch);
        this.#floor = Math.max(this.#floor, row.epoch);
        this.#leftoverEpoch = row.epoch;
      }
    } catch (e) {
      if (isAbortError(e)) throw e;
      /* status unavailable: the epoch rule still holds for this driver's own sessions */
    }
  }

  // --- source plan ------------------------------------------------------------------------------------------------

  async #planSource(signal: AbortSignal | null): Promise<OthersSourcePlan> {
    const s = this.#settings!;
    const list = await raceWithAbort(
      // A plan check only: no program list (window and audio-session enumeration) every few seconds.
      this.#deps.invoke<OthersProcessList>(LIST_LOOPBACK_PROCESSES, { source: s.source, fallback: s.fallback !== false, processes: false }),
      signal,
    );
    const plan = parsePlan(list?.plan);
    if (!plan) throw new Error(i18n.t("others.error_plan"));
    return this.#withCaptureBackoff(plan);
  }

  /** While a program whose capture failed waits for its retry, the plan captures the default output instead. */
  #withCaptureBackoff(plan: OthersSourcePlan): OthersSourcePlan {
    const failure = this.#captureFailure;
    if (!failure) return plan;
    if (!plan.source || !sameSource(plan.source, failure.source)) {
      // The program exited (or another source is planned): its next start is tried at once.
      this.#captureFailure = null;
      return plan;
    }
    if (this.#settings?.fallback === false || this.#deps.clock.now() >= failure.retryAt) return plan;
    return captureFailedPlan(plan);
  }

  /** The capture of a running program failed to open: back off before trying it again. */
  #noteCaptureFailure(source: string): number {
    const previous = this.#captureFailure;
    const attempts = previous && sameSource(previous.source, source) ? previous.attempts + 1 : 1;
    const delay = CAPTURE_RETRY_DELAYS_MS[Math.min(attempts - 1, CAPTURE_RETRY_DELAYS_MS.length - 1)];
    this.#captureFailure = { source, attempts, retryAt: this.#deps.clock.now() + delay };
    return delay;
  }

  /** Run on what `plan` says: start or move the pipeline, or wait for the program. */
  async #apply(plan: OthersSourcePlan, run: number, signal: AbortSignal | null): Promise<void> {
    this.state.plan = plan;
    if (!plan.source) {
      if (plan.fallback === "unsupported" || !plan.watch) throw new Error(i18n.t("others.error_unsupported"));
      // End the session before stopping it: the stop emits `stopped` for its epoch, which must not read as another
      // start replacing it.
      const running = this.state.source !== null;
      const epoch = this.#epoch;
      const leftover = this.#leftoverEpoch;
      this.#leftoverEpoch = null;
      this.#endSession();
      Object.assign(this.state, {
        status: ServiceNetworkState.connected,
        phase: "waiting" as OthersPhase,
        source: null,
        routePlan: null,
      });
      const note = planNote(plan);
      this.#setStatus(note.text, note.level);
      // Exactly the session being left (never a newer start of the key), or one left running before this start.
      if (running) await this.#stopPipeline(epoch);
      else if (leftover !== null) await this.#stopPipeline(leftover);
      this.#check(run);
      return;
    }
    if (plan.source === this.state.source && this.state.phase === "running") {
      const note = planNote(plan);
      if (note.text || this.state.statusText === "") this.#setStatus(note.text, note.level);
      return;
    }
    let started: RustSpeechPipelineStarted;
    try {
      started = await this.#startPipeline(plan.source, run, signal);
    } catch (e) {
      if (isAbortError(e) || run !== this.#run) throw e;
      if (isProcessSource(plan.source)) {
        // The program may have exited between the plan and the start, or it runs but its capture cannot be opened
        // (refused activation, the open timing out while the audio service is busy): with the fallback on, the
        // default output is captured until the program is tried again. Plan once more to tell which.
        const delay = this.#settings?.fallback !== false && !engineError(errorText(e)) ? this.#noteCaptureFailure(plan.source) : 0;
        const again = await this.#planSource(signal).catch(() => null);
        this.#check(run);
        if (again && again.source !== plan.source) {
          if (again.fallback === "capture_failed") {
            this.#deps.log(
              i18n.t("others.log_capture_failed", { program: processExe(plan.source), error: errorText(e), seconds: Math.round(delay / 1000) }),
              "warning",
            );
          }
          await this.#apply(again, run, signal);
          return;
        }
      }
      throw new Error(startErrorMessage(e, (id) => this.#deps.title(id)));
    }
    this.#check(run);
    this.#onStarted(started, plan);
  }

  #scheduleWatch(run: number): void {
    if (run !== this.#run || !this.state.plan?.watch) return;
    const clock = this.#deps.clock;
    if (this.#watchTimer !== null) clock.clearTimeout(this.#watchTimer);
    this.#watchTimer = clock.setTimeout(() => {
      this.#watchTimer = null;
      void this.#watchTick(run);
    }, WATCH_INTERVAL_MS);
  }

  async #watchTick(run: number): Promise<void> {
    if (run !== this.#run) return;
    // A restart plans on its own; a change being applied finishes first.
    if (this.#applying || this.state.phase === "restarting" || this.state.phase === "starting") {
      this.#scheduleWatch(run);
      return;
    }
    let plan: OthersSourcePlan;
    try {
      plan = await this.#planSource(this.#abort?.signal ?? null);
    } catch (e) {
      if (run === this.#run && !isAbortError(e)) this.#scheduleWatch(run);
      return;
    }
    if (run !== this.#run) return;
    if (plan.source !== this.state.source || (plan.source === null && this.state.phase !== "waiting")) {
      this.#applying = true;
      this.#deps.log(
        plan.source
          ? i18n.t("others.log_source_changed", { source: plan.source })
          : i18n.t("others.log_waiting", { program: plan.watch ?? "" }),
        "info",
      );
      try {
        await this.#apply(plan, run, this.#abort?.signal ?? null);
      } catch (e) {
        if (isAbortError(e) || run !== this.#run) return;
        this.#fail(errorText(e));
        return;
      } finally {
        this.#applying = false;
      }
    } else {
      // Same source for another reason (a failed capture's program exited, ...): the note follows the plan.
      const reasonChanged = plan.fallback !== this.state.plan?.fallback;
      this.state.plan = plan;
      if (reasonChanged && this.state.phase === "running") this.#restoreNote();
    }
    this.#scheduleWatch(run);
  }

  // --- pipeline ---------------------------------------------------------------------------------------------------

  async #startPipeline(source: string, run: number, signal: AbortSignal | null): Promise<RustSpeechPipelineStarted> {
    const config = buildOthersPipelineConfig(this.#settings!, this.#request!, source);
    // The session being replaced (a source move or a restart) is over: its held finals go out now, and its late
    // events (including its `stopped` status) are refused from here on.
    this.#endSession();
    this.#setStatus(i18n.t("others.status_starting"));
    if (this.state.phase !== "starting") {
      this.state.phase = "restarting";
      this.state.status = ServiceNetworkState.connecting;
    }
    const call = this.#deps.invoke<RustSpeechPipelineStarted>("plugin:speech|start_pipeline", { config });
    this.#pendingStart = call;
    const clear = () => {
      if (this.#pendingStart === call) this.#pendingStart = null;
    };
    call.then(clear, clear);
    let started: RustSpeechPipelineStarted;
    try {
      started = await raceWithAbort(call, signal);
    } catch (e) {
      this.#floor = Math.max(this.#floor, this.#lastEpoch);
      throw e;
    }
    if (run !== this.#run) {
      // Stopped meanwhile: end exactly this session (never a newer start of the key).
      void this.#deps.invoke("plugin:speech|stop_pipeline", { pipelineId: OTHERS_PIPELINE_ID, epoch: started.epoch }).catch(() => {});
      throw new AbortedError();
    }
    return started;
  }

  #onStarted(started: RustSpeechPipelineStarted, plan: OthersSourcePlan): void {
    this.#noteEpoch(started.epoch);
    this.#epoch = started.epoch;
    // The start replaced whatever ran on the key; a program captured at last needs no retry back-off any more.
    this.#leftoverEpoch = null;
    if (isProcessSource(plan.source)) this.#captureFailure = null;
    this.#twoPass = isTwoPassPlan(started.plan);
    this.#publisherFor(started.epoch).setTwoPass(this.#twoPass);
    this.#runningSince = this.#deps.clock.now();
    Object.assign(this.state, {
      status: ServiceNetworkState.connected,
      phase: "running" as OthersPhase,
      source: plan.source,
      routePlan: started.plan,
    });
    const note = planNote(plan);
    this.#setStatus(note.text, note.level);
    this.#deps.log(i18n.t("others.log_started", { source: plan.source ?? "" }), "info");
    const fatal = this.#earlyFatal;
    this.#earlyFatal = null;
    if (fatal && fatal.epoch === started.epoch) this.#sessionFailed(fatal);
  }

  /** Stop the key's session: exactly `epoch` when given (never a newer start), else whatever runs or is starting. */
  async #stopPipeline(epoch: number | null = null): Promise<void> {
    try {
      await this.#deps.invoke(
        "plugin:speech|stop_pipeline",
        epoch !== null ? { pipelineId: OTHERS_PIPELINE_ID, epoch } : { pipelineId: OTHERS_PIPELINE_ID },
      );
    } catch {
      /* not running */
    }
  }

  /** The current session is over: publish what its segments hold and refuse its later events. */
  #endSession(): void {
    if (this.#epoch !== null) this.#floor = Math.max(this.#floor, this.#epoch);
    this.#floor = Math.max(this.#floor, this.#lastEpoch);
    this.#epoch = null;
    this.#publisher?.flush();
    this.#publisher = null;
  }

  // --- events -----------------------------------------------------------------------------------------------------

  async #attachListeners(signal: AbortSignal): Promise<void> {
    const on = async <T extends RustSpeechEnvelope>(event: string, handle: (ev: T) => void) => {
      this.#unlisteners.push(
        await this.#deps.listen<T>(event, (ev) => {
          if (this.#accepts(ev)) handle(ev);
        }, signal),
      );
    };
    await on<RustSpeechStatus>(SPEECH_EVENTS.status, (ev) => this.#onStatus(ev));
    await on<RustSpeechPartial>(SPEECH_EVENTS.partial, (ev) => this.#publisherFor(ev.epoch).partial(ev));
    await on<RustSpeechFinal>(SPEECH_EVENTS.final, (ev) => this.#publisherFor(ev.epoch).final(ev));
    await on<RustSpeechUtterance>(SPEECH_EVENTS.utterance, (ev) => this.#publisherFor(ev.epoch).utterance(ev));
    await on<RustSpeechTurnEnd>(SPEECH_EVENTS.turnEnd, (ev) => this.#publisherFor(ev.epoch).turnEnd(ev));
  }

  #noteEpoch(epoch: number): void {
    if (Number.isFinite(epoch) && epoch > this.#lastEpoch) this.#lastEpoch = epoch;
  }

  #accepts(env: RustSpeechEnvelope | null | undefined): boolean {
    if (!env || env.pipelineId !== OTHERS_PIPELINE_ID || typeof env.epoch !== "number") return false;
    if (this.state.phase === "idle") return false;
    this.#noteEpoch(env.epoch);
    return this.#epoch !== null ? env.epoch === this.#epoch : env.epoch > this.#floor;
  }

  #publisherFor(epoch: number): SegmentPublisher {
    if (this.#publisher && this.#publisher.epoch === epoch) return this.#publisher;
    this.#publisher?.flush();
    this.#publisher = new SegmentPublisher({
      pipelineId: OTHERS_PIPELINE_ID,
      epoch,
      twoPass: this.#twoPass,
      clock: this.#deps.clock,
      sink: {
        interim: (text, meta) => this.#sink.interim(text, meta),
        final: (text, meta) => this.#sink.final(text, meta),
        // Like STT: an abandoned interim publishes nothing (the next line replaces it).
        abandon: () => {},
      },
    });
    return this.#publisher;
  }

  #onStatus(ev: RustSpeechStatus): void {
    const current = this.#epoch !== null && ev.epoch === this.#epoch;
    if (ev.state === "error") {
      if (current && this.state.phase === "running") this.#sessionFailed(ev);
      else if (this.#epoch === null) this.#earlyFatal = ev;
      return;
    }
    if (ev.state === "stopped") {
      if (current && this.state.phase === "running") {
        // Another start of the `loopback` key replaced this session (nothing else should use it).
        this.#fail(i18n.t("others.error_stopped_elsewhere"));
      }
      return;
    }
    const setup = ev.code === "starting" || ev.code === "loading_model" || ev.code === "loading_runtime" || ev.code === "downloading";
    if (this.state.phase === "running" && setup && ev.level === "info") return;
    const line = speechStatusLine(ev, (id) => this.#deps.title(id));
    if (line) {
      const [text, options] = line;
      this.#setStatus(text, options?.level ?? "info", options?.progress ?? null);
      const expire = options?.expireMs;
      if (expire) {
        this.#deps.clock.setTimeout(() => {
          if (this.state.statusText === text) this.#restoreNote();
        }, expire);
      }
    } else if (ev.code === "listening" && this.state.phase === "running") {
      this.#restoreNote();
    }
  }

  /** Back to the plan's note (or nothing) after a transient status. */
  #restoreNote(): void {
    const note = planNote(this.state.plan);
    this.#setStatus(note.text, note.level);
  }

  // --- automatic restart ------------------------------------------------------------------------------------------

  #sessionFailed(ev: RustSpeechStatus): void {
    if (this.#deps.clock.now() - this.#runningSince >= STABLE_RUN_MS) this.#restartAttempts = 0;
    this.#endSession();
    this.#floor = Math.max(this.#floor, ev.epoch);
    const rawDetail = String(ev.detail ?? "").trim();
    const detail = rawDetail || ev.code;
    if (!restartCanFix(ev.code, rawDetail, this.state.routePlan, this.#request?.tier)) {
      const text = ev.code === "not_installed" && ev.modelId
        ? i18n.t("others.error_not_installed", { model: this.#deps.title(ev.modelId) })
        : i18n.t("others.error_failed", { detail });
      this.#fail(text);
      return;
    }
    this.#scheduleRestart(detail);
  }

  #scheduleRestart(detail: string): void {
    this.#restartAttempts += 1;
    if (this.#restartAttempts > MAX_RESTARTS) {
      this.#fail(i18n.t("others.error_gave_up", { attempts: MAX_RESTARTS, detail }));
      return;
    }
    const delay = RESTART_DELAYS_MS[Math.min(this.#restartAttempts - 1, RESTART_DELAYS_MS.length - 1)];
    const line = i18n.t("others.status_restarting", { detail, seconds: Math.ceil(delay / 1000) });
    Object.assign(this.state, { status: ServiceNetworkState.connecting, phase: "restarting" as OthersPhase, routePlan: null });
    this.#setStatus(line, "warning");
    this.#deps.log(line, "warning");
    const run = this.#run;
    const clock = this.#deps.clock;
    if (this.#restartTimer !== null) clock.clearTimeout(this.#restartTimer);
    this.#restartTimer = clock.setTimeout(() => {
      this.#restartTimer = null;
      void this.#restart(run);
    }, delay);
  }

  async #restart(run: number): Promise<void> {
    if (run !== this.#run) return;
    this.state.restarts += 1;
    const signal = this.#abort?.signal ?? null;
    try {
      // The program may have started or exited meanwhile.
      const plan = await this.#planSource(signal);
      this.#check(run);
      this.state.source = null; // force a fresh start even on the same source
      await this.#apply(plan, run, signal);
      this.#check(run);
      this.#deps.log(i18n.t("others.log_restarted"), "info");
      this.#scheduleWatch(run);
    } catch (e) {
      if (isAbortError(e) || run !== this.#run) return;
      const text = errorText(e);
      if (!restartCanFix("", text, this.state.routePlan, this.#request?.tier)) {
        this.#fail(text);
        return;
      }
      this.#scheduleRestart(text);
    }
  }

  // --- cleanup ----------------------------------------------------------------------------------------------------

  #clearTimers(): void {
    const clock = this.#deps.clock;
    if (this.#watchTimer !== null) clock.clearTimeout(this.#watchTimer);
    if (this.#restartTimer !== null) clock.clearTimeout(this.#restartTimer);
    this.#watchTimer = null;
    this.#restartTimer = null;
  }

  /** Remove listeners; `flush` publishes what the current session's segments hold first. */
  #detach(flush: boolean): void {
    for (const u of this.#unlisteners) {
      try {
        u();
      } catch {
        /* ignore */
      }
    }
    this.#unlisteners = [];
    if (flush) this.#publisher?.flush();
    else this.#publisher?.dispose();
    this.#publisher = null;
    if (this.#epoch !== null) this.#floor = Math.max(this.#floor, this.#epoch);
    this.#epoch = null;
    this.#earlyFatal = null;
    this.#applying = false;
  }
}
