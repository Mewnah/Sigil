/**
 * Benchmark and tuning of the voice changer (P3-G): the per-PC autotuner
 * (`voice_changer_autotune`, P3-F), "Benchmark this PC" (a paced run of the selected voice with the
 * current settings through `plugin:bench|run_benchmark`), and the live recording of a real session
 * (`voice_changer_live_record_*`, the authoritative evidence for a friend's PC).
 *
 * Autotune and the benchmark measure what the PC can do, so both refuse to run while the voice
 * changer runs (Rust refuses too); live recording needs it running. A recording outlives an engine
 * restart (Rust ends it only after the voice changer has been stopped for 2 s), so a stop only ends
 * it here when no start follows within that grace.
 */
import { invoke } from "@tauri-apps/api/core";
import { listen, type UnlistenFn } from "@tauri-apps/api/event";
import { proxy } from "valtio";

import { devLog } from "@/utils/devLog";

import {
    BENCH_IPC,
    VC_COMMANDS,
    VC_EVENTS,
    type BenchProgress,
    type VcAutotuneProgress,
    type VcAutotuneRequest,
    type VcAutotuneResult,
    type VcLiveRecordSummary,
} from "./ipc";
import type { VoiceChangerState } from "./schema";
import { benchmarkSpec, normalizeAutotuneResult, normalizeLiveRecordSummary, summarizeBenchReport, type BenchSummary } from "./tuning";

/** What the bench controller needs from the voice-changer service. */
export interface VoiceBenchHost {
    settings(): VoiceChangerState;
    isRunning(): boolean;
    /** Apply the autotuner's params patch through the params sync (restarting when needed). */
    applyTuning(params: Readonly<Record<string, unknown>>): Promise<boolean>;
}

export interface VoiceBenchState {
    autotune: {
        running: boolean;
        progress: VcAutotuneProgress | null;
        /** The stored result for this PC (or the latest run). */
        result: VcAutotuneResult | null;
        error: string | null;
        /** The result was applied to the settings (this session). */
        applied: boolean;
    };
    benchmark: {
        running: boolean;
        progress: BenchProgress | null;
        summary: BenchSummary | null;
        error: string | null;
    };
    liveRecord: {
        recording: boolean;
        /** `Date.now()` when the recording started. */
        startedAt: number | null;
        maxSeconds: number;
        summary: VcLiveRecordSummary | null;
        error: string | null;
    };
}

/** Longest live recording (Rust `live_record::MAX_SECONDS`). */
export const LIVE_RECORD_MAX_SECONDS = 600;
/** Rust ends a recording 2 s after the voice changer stops (`live_record::STOPPED_GRACE`); collect it a little later. */
const LIVE_RECORD_COLLECT_DELAY_MS = 2600;
/**
 * A start this soon after a stop is a restart the recording runs through: Rust samples the engine
 * every 100 ms and ends the recording once it has seen it stopped for 2 s. Less a margin for the
 * event delivery; a slower start ends the recording here (it may already have ended in Rust).
 */
export const LIVE_RECORD_RESTART_GRACE_MS = 1900;

function errorText(e: unknown): string {
    if (typeof e === "string") return e;
    if (e instanceof Error) return e.message;
    try {
        return JSON.stringify(e);
    } catch {
        return String(e);
    }
}

export class VoiceBench {
    readonly state = proxy<VoiceBenchState>({
        autotune: { running: false, progress: null, result: null, error: null, applied: false },
        benchmark: { running: false, progress: null, summary: null, error: null },
        liveRecord: { recording: false, startedAt: null, maxSeconds: LIVE_RECORD_MAX_SECONDS, summary: null, error: null },
    });

    private unlisteners: UnlistenFn[] = [];
    /** Collects the recording once Rust ended it at `max_seconds`. */
    private deadlineTimer: ReturnType<typeof setTimeout> | null = null;
    /** Collects the recording after a stop that no start followed in time (Rust ended it). */
    private stoppedTimer: ReturnType<typeof setTimeout> | null = null;
    /** `Date.now()` of the earliest stop since the engine last ran (while a recording runs). */
    private stoppedAt: number | null = null;

    constructor(private readonly host: VoiceBenchHost) {}

    async init() {
        this.unlisteners.push(
            await listen<VcAutotuneProgress>(VC_EVENTS.autotuneProgress, (ev) => {
                if (this.state.autotune.running && ev.payload && typeof ev.payload === "object") {
                    this.state.autotune.progress = ev.payload;
                }
            }),
            await listen<BenchProgress>(BENCH_IPC.progressEvent, (ev) => {
                if (this.state.benchmark.running && ev.payload && typeof ev.payload === "object") {
                    this.state.benchmark.progress = ev.payload;
                }
            }),
        );
        await this.loadStoredAutotune();
    }

    dispose() {
        for (const u of this.unlisteners.splice(0)) u();
        this.clearLiveRecordTimers();
    }

    /** The result stored for this PC (GPU and driver), if any. */
    async loadStoredAutotune() {
        try {
            const r = normalizeAutotuneResult(await invoke<unknown>(VC_COMMANDS.getAutotune));
            if (r && !this.state.autotune.running) this.state.autotune.result = r;
        } catch (e) {
            devLog("[VoiceChanger] No stored autotune result:", e);
        }
    }

    // ---- autotune -----------------------------------------------------------------------------------

    /** Tune on the selected voice (`quick`: fewer candidates and blocks). Resolves the result or null. */
    async runAutotune(quick: boolean): Promise<VcAutotuneResult | null> {
        const a = this.state.autotune;
        if (a.running || this.host.isRunning()) return null;
        a.running = true;
        a.error = null;
        a.progress = null;
        try {
            const request: VcAutotuneRequest = { quick };
            const r = normalizeAutotuneResult(await invoke<unknown>(VC_COMMANDS.autotune, { request }));
            if (!r) throw new Error("the autotuner returned no result");
            a.result = r;
            a.applied = false;
            return r;
        } catch (e) {
            a.error = errorText(e);
            return null;
        } finally {
            a.running = false;
            a.progress = null;
        }
    }

    async cancelAutotune() {
        try {
            await invoke(VC_COMMANDS.cancelAutotune);
        } catch (e) {
            devLog("[VoiceChanger] Could not cancel the autotuner:", e);
        }
    }

    /** Apply the tuned settings. */
    async applyAutotune(): Promise<boolean> {
        const r = this.state.autotune.result;
        if (!r) return false;
        const ok = await this.host.applyTuning(r.params);
        this.state.autotune.applied = ok;
        return ok;
    }

    // ---- benchmark this PC ------------------------------------------------------------------------------

    /** A paced run of the selected voice with the current settings. */
    async runBenchmark(): Promise<BenchSummary | null> {
        const b = this.state.benchmark;
        if (b.running || this.host.isRunning()) return null;
        // No voice selected: nothing to measure (the panel disables the button then).
        const spec = benchmarkSpec(this.host.settings());
        if (!spec) return null;
        b.running = true;
        b.error = null;
        b.progress = null;
        try {
            const summary = summarizeBenchReport(await invoke<unknown>(BENCH_IPC.run, { spec }));
            if (!summary) throw new Error("the benchmark returned no report");
            b.summary = summary;
            if (summary.status === "error" || summary.status === "cancelled") b.error = summary.error ?? summary.status;
            return summary;
        } catch (e) {
            b.error = errorText(e);
            return null;
        } finally {
            b.running = false;
            b.progress = null;
        }
    }

    async cancelBenchmark() {
        try {
            await invoke(BENCH_IPC.cancel);
        } catch (e) {
            devLog("[VoiceChanger] Could not cancel the benchmark:", e);
        }
    }

    // ---- live record ------------------------------------------------------------------------------------

    async startLiveRecord(maxSeconds = LIVE_RECORD_MAX_SECONDS): Promise<boolean> {
        const l = this.state.liveRecord;
        if (l.recording || !this.host.isRunning()) return false;
        const seconds = Math.max(10, Math.min(LIVE_RECORD_MAX_SECONDS, Math.round(maxSeconds)));
        l.error = null;
        l.summary = null;
        try {
            await invoke(VC_COMMANDS.liveRecordStart, { request: { max_seconds: seconds } });
            l.recording = true;
            l.startedAt = Date.now();
            l.maxSeconds = seconds;
            this.clearLiveRecordTimers();
            // Rust stops by itself at the limit; collect the summary then.
            this.deadlineTimer = setTimeout(() => {
                this.deadlineTimer = null;
                void this.stopLiveRecord();
            }, seconds * 1000 + LIVE_RECORD_COLLECT_DELAY_MS);
            return true;
        } catch (e) {
            l.error = errorText(e);
            return false;
        }
    }

    /** Stop (or collect a recording that ended by itself) and keep its summary. */
    async stopLiveRecord(): Promise<VcLiveRecordSummary | null> {
        const l = this.state.liveRecord;
        if (!l.recording) return l.summary;
        this.clearLiveRecordTimers();
        try {
            const s = normalizeLiveRecordSummary(await invoke<unknown>(VC_COMMANDS.liveRecordStop));
            l.summary = s;
            if (!s) l.error = "the recorder returned no summary";
            return s;
        } catch (e) {
            l.error = errorText(e);
            return null;
        } finally {
            l.recording = false;
            l.startedAt = null;
        }
    }

    /**
     * The voice changer stopped (its `stopped` event, or a stop sent from here, whichever comes
     * first). Unless it starts again within the grace (`onEngineStarted`), Rust ends the recording:
     * collect it then.
     */
    onEngineStopped() {
        if (!this.state.liveRecord.recording || this.stoppedTimer) return;
        this.stoppedAt = Date.now();
        this.stoppedTimer = setTimeout(() => {
            this.stoppedTimer = null;
            this.stoppedAt = null;
            void this.stopLiveRecord();
        }, LIVE_RECORD_COLLECT_DELAY_MS);
    }

    /** The voice changer runs again: a restart within the grace keeps the recording going. */
    onEngineStarted() {
        if (!this.stoppedTimer || this.stoppedAt === null) return;
        if (Date.now() - this.stoppedAt >= LIVE_RECORD_RESTART_GRACE_MS) return; // collected as planned
        clearTimeout(this.stoppedTimer);
        this.stoppedTimer = null;
        this.stoppedAt = null;
    }

    private clearLiveRecordTimers() {
        if (this.deadlineTimer) clearTimeout(this.deadlineTimer);
        if (this.stoppedTimer) clearTimeout(this.stoppedTimer);
        this.deadlineTimer = null;
        this.stoppedTimer = null;
        this.stoppedAt = null;
    }
}
