/**
 * Voice-changer run metrics (`get_voice_changer_metrics`) and the execution provider (EP) in use.
 *
 * The P0 fields (`process_p99_ms`, `backlog_ms`, `dropped_ms`, `underruns`, `ep`) are optional on
 * the wire (params_contract.json: an older backend omits them), so they normalize to `null`.
 * Pure module: no Tauri, DOM or `window`.
 */
import type { RustEpOutcome } from "@/types";

/** Raw `get_voice_changer_metrics` reply. */
export interface VoiceChangerMetricsRust {
    buffer_ms: number;
    process_ms: number;
    chunk_ms: number;
    process_p99_ms?: number | null;
    backlog_ms?: number | null;
    dropped_ms?: number | null;
    underruns?: number | null;
    ep?: RustEpOutcome | null;
}

/** Normalized metrics: every number finite; P0 fields `null` when the backend does not report them. */
export interface VoiceChangerMetrics {
    /** Output queue (deque + ring) in ms. */
    buffer_ms: number;
    /** EMA of the per-block processing time. */
    process_ms: number;
    /** Block length in ms; 0 while the engine is still starting (devices not open yet). */
    chunk_ms: number;
    /** p99 of the per-block processing time over the recent window. */
    process_p99_ms: number | null;
    /** Captured audio waiting to be converted, in ms. */
    backlog_ms: number | null;
    /** Audio dropped by the latency budget since start, in ms. */
    dropped_ms: number | null;
    /** Output callbacks that ran dry since start. */
    underruns: number | null;
    /** EP of the running model; `null` for classic / WebSocket, while loading, or when stopped. */
    ep: RustEpOutcome | null;
}

function num(n: unknown): number {
    return typeof n === "number" && Number.isFinite(n) ? Math.max(0, n) : 0;
}

function optNum(n: unknown): number | null {
    return typeof n === "number" && Number.isFinite(n) ? Math.max(0, n) : null;
}

function normalizeEp(ep: unknown): RustEpOutcome | null {
    if (!ep || typeof ep !== "object") return null;
    const o = ep as Partial<RustEpOutcome>;
    if (typeof o.requested !== "string" || typeof o.active !== "string") return null;
    return {
        requested: o.requested.trim().toLowerCase(),
        active: o.active.trim().toLowerCase(),
        reason: typeof o.reason === "string" && o.reason.trim() ? o.reason.trim() : null,
    };
}

export function normalizeMetrics(raw: unknown): VoiceChangerMetrics | null {
    if (!raw || typeof raw !== "object") return null;
    const m = raw as Partial<VoiceChangerMetricsRust>;
    return {
        buffer_ms: num(m.buffer_ms),
        process_ms: num(m.process_ms),
        chunk_ms: num(m.chunk_ms),
        process_p99_ms: optNum(m.process_p99_ms),
        backlog_ms: optNum(m.backlog_ms),
        dropped_ms: optNum(m.dropped_ms),
        underruns: optNum(m.underruns),
        ep: normalizeEp(m.ep),
    };
}

/** Display name of an EP id (technical names; not translated). */
export function epDisplayName(ep: string | null | undefined): string {
    switch ((ep ?? "").trim().toLowerCase()) {
        case "":
        case "cpu":
            return "CPU";
        case "cuda":
        case "gpu":
            return "CUDA";
        case "dml":
        case "directml":
            return "DirectML";
        case "tensorrt":
        case "trt":
            return "TensorRT";
        case "nvrtx":
            return "TensorRT-RTX";
        default:
            return (ep ?? "").trim();
    }
}

export type EpStatus =
    | { kind: "granted"; active: string }
    | { kind: "fallback"; requested: string; active: string; reason: string | null };

/** Whether the model runs on the EP it asked for. `null` when there is no EP to report. */
export function epStatus(ep: RustEpOutcome | null | undefined): EpStatus | null {
    if (!ep) return null;
    if (ep.requested !== ep.active) {
        return { kind: "fallback", requested: ep.requested, active: ep.active, reason: ep.reason ?? null };
    }
    return { kind: "granted", active: ep.active };
}

/** EP-outcome components that belong to the voice changer (Rust diagnostics keys). */
export function isVoiceChangerEpComponent(component: string): boolean {
    return component === "vc" || component.startsWith("vc.");
}

/** Real-time health of the processing loop. */
export type RealtimeHealth = "ok" | "spikes" | "slow";

export interface MetricsAssessment {
    /**
     * Engine running but not converting yet: devices not open (no block length), or the model still
     * loading. Rust publishes the device rates (so `chunk_ms`) before the possibly slow model load,
     * so "no block processed yet this run" (zero processing time and nothing queued for output)
     * counts as starting too.
     */
    starting: boolean;
    /** `slow`: the average block takes longer than real time; `spikes`: only the slowest 1 % do. */
    realtime: RealtimeHealth;
    /** Backlog at or above 80 % of what the latency budget lets wait (older audio is about to be dropped). */
    backlogHigh: boolean;
    /** The budget dropped audio since the previous poll. */
    droppingNow: boolean;
    /** The output ran dry since the previous poll. */
    underrunNow: boolean;
}

/**
 * Rust zeroes the per-run counters at start and sets the processing time after each converted
 * block, so all-zero processing plus an empty output queue means the loop has not run yet.
 */
function noBlockProcessedYet(m: VoiceChangerMetrics): boolean {
    return m.process_ms === 0 && (m.process_p99_ms ?? 0) === 0 && m.buffer_ms === 0;
}

/**
 * Judge one metrics sample. `budgetMs` is the latency budget in effect (Rust lets
 * `max(budget, one block)` wait); `prev` is the previous sample of the same run, if any.
 */
export function assessMetrics(
    m: VoiceChangerMetrics,
    budgetMs: number,
    prev?: VoiceChangerMetrics | null,
): MetricsAssessment {
    const starting = !(m.chunk_ms > 0) || noBlockProcessedYet(m);
    let realtime: RealtimeHealth = "ok";
    if (!starting) {
        if (m.process_ms >= m.chunk_ms) realtime = "slow";
        else if (m.process_p99_ms != null && m.process_p99_ms >= m.chunk_ms) realtime = "spikes";
    }
    const allowed = Math.max(budgetMs, m.chunk_ms);
    const backlogHigh = !starting && m.backlog_ms != null && allowed > 0 && m.backlog_ms >= 0.8 * allowed;
    const droppingNow = m.dropped_ms != null && prev?.dropped_ms != null && m.dropped_ms > prev.dropped_ms;
    const underrunNow = m.underruns != null && prev?.underruns != null && m.underruns > prev.underruns;
    return { starting, realtime, backlogHigh, droppingNow, underrunNow };
}

/** `ms` for display: one decimal below 100 ms, whole ms above. */
export function formatMs(ms: number): string {
    if (!Number.isFinite(ms)) return "–";
    return ms >= 100 ? ms.toFixed(0) : ms.toFixed(1);
}
