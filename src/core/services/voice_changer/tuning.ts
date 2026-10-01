/**
 * Benchmark and autotune helpers of the voice-changer panel (P3-G):
 *
 * - The autotuner's result (`voice_changer_autotune` / `voice_changer_get_autotune`, Rust
 *   `bench::autotune::AutotuneResult`) and the params patch it recommends, applied only to the
 *   engine-shape params it may set ([`AUTOTUNE_PARAM_KEYS`]).
 * - "Benchmark this PC": a paced benchmark spec (`sigil-bench-spec/1`, `plugin:bench|run_benchmark`)
 *   of the selected voice with the current settings, and the few numbers of its report the panel
 *   shows (per-block compute p50 / p95 / p99, the real-time factor, the simulated mic-to-speaker
 *   latency, gaps, skipped audio, CPU and VRAM).
 * - Live recording (`voice_changer_live_record_*`): the summary it returns.
 *
 * Pure module: no Tauri, DOM or `window`.
 */
import { clampFinite, VC_LEVERS, VC_NUMBER_RANGES as R, type VcLever } from "./contract";
import type { VcAutotuneResult, VcLiveRecordSummary } from "./ipc";
import { LEVER_STATE_KEYS, toRustParams, type RustParamKey } from "./params";
import type { VoiceChangerState } from "./schema";

/** Params the autotuner may set: the engine shape of the voice, never devices, voices or consent. */
export const AUTOTUNE_PARAM_KEYS: readonly RustParamKey[] = [
    "rvc_preset",
    "rvc_execution_provider",
    "pitch_extractor",
    "extra_context_ms",
    "crossfade_ms",
    "block_ms",
    "chunk_frames",
    "precision",
    "worker_priority",
    ...VC_LEVERS,
];

function num(v: unknown): number | null {
    return typeof v === "number" && Number.isFinite(v) ? v : null;
}

function str(v: unknown): string | null {
    return typeof v === "string" && v.trim() !== "" ? v.trim() : null;
}

function strList(v: unknown): string[] {
    return Array.isArray(v) ? v.filter((x): x is string => typeof x === "string") : [];
}

function numMap(v: unknown): Record<string, number> {
    const out: Record<string, number> = {};
    if (!v || typeof v !== "object") return out;
    for (const [k, x] of Object.entries(v as Record<string, unknown>)) {
        if (typeof x === "number" && Number.isFinite(x)) out[k] = x;
    }
    return out;
}

/** An autotune result, or `null` when the reply is not one (`voice_changer_get_autotune` on a PC never tuned). */
export function normalizeAutotuneResult(raw: unknown): VcAutotuneResult | null {
    if (!raw || typeof raw !== "object") return null;
    const r = raw as Record<string, unknown>;
    const params = r.params && typeof r.params === "object" && !Array.isArray(r.params) ? (r.params as Record<string, unknown>) : null;
    if (!params) return null;
    return {
        gpu: str(r.gpu),
        driver: str(r.driver),
        created_unix_ms: num(r.created_unix_ms) ?? 0,
        params: { ...params },
        rtf_p99: num(r.rtf_p99) ?? 0,
        block_ms_p95: num(r.block_ms_p95) ?? 0,
        latency_ms_p50: num(r.latency_ms_p50),
        fidelity: num(r.fidelity),
        report_dir: str(r.report_dir),
        notes: strList(r.notes),
        met_limit: r.met_limit === true,
        slot: str(r.slot),
        format: str(r.format),
        execution_provider_active: str(r.execution_provider_active),
        latency_ms_p99: num(r.latency_ms_p99),
        fidelity_metrics: numMap(r.fidelity_metrics),
        seconds: num(r.seconds),
        conditions: r.conditions && typeof r.conditions === "object" && !Array.isArray(r.conditions) ? { ...(r.conditions as Record<string, unknown>) } : null,
        stale: strList(r.stale),
    };
}

/** The parts of an autotune `params` patch the panel shows (known keys only, in a stable order). */
export function autotuneParamRows(params: Readonly<Record<string, unknown>>): { key: string; value: string }[] {
    const rows: { key: string; value: string }[] = [];
    for (const key of AUTOTUNE_PARAM_KEYS) {
        const v = params[key];
        if (v == null) continue;
        if (typeof v === "number" && Number.isFinite(v)) rows.push({ key, value: String(Math.round(v * 100) / 100) });
        else if (typeof v === "string" || typeof v === "boolean") rows.push({ key, value: String(v) });
    }
    return rows;
}

/** A lever's state key (`partial_decode` -> `partialDecode`). */
export function leverStateKey(lever: VcLever): keyof VoiceChangerState {
    return LEVER_STATE_KEYS[lever];
}

// ---------------------------------------------------------------------------------------------
// "Benchmark this PC" (a paced run of the selected voice)
// ---------------------------------------------------------------------------------------------

/** Simulated capture time of the in-app benchmark. */
export const BENCHMARK_SECONDS = 12;

function clampNum(v: unknown, min: number, max: number, fallback: number): number {
    return clampFinite(v, min, max, fallback);
}

/**
 * A paced benchmark of the selected RVC voice with the current settings, on the voice changer's own
 * load path (`engine.kind = rvc`, the engine factory), fed by the synthetic voice (exact F0 known,
 * no files). `null` when no voice is selected.
 */
export function benchmarkSpec(vc: VoiceChangerState, seconds = BENCHMARK_SECONDS): Record<string, unknown> | null {
    const slotDir = vc.rvcSlotDir.trim();
    if (!slotDir) return null;
    const p = toRustParams(vc);
    const levers: Record<string, string> = {};
    for (const lever of VC_LEVERS) levers[lever] = String(p[lever] ?? "auto");
    const prefill = clampNum(p.output_prefill_ms, R.output_prefill_ms.min, R.output_prefill_ms.max, R.output_prefill_ms.default);
    return {
        schema: "sigil-bench-spec/1",
        name: "vc-this-pc",
        description: "In-app benchmark: the selected voice with the current voice-changer settings",
        input: { kind: "synthetic", signal: "voice", seconds: 8 },
        stream: {
            block_frames: p.chunk_frames,
            extra_context_ms: clampNum(p.extra_context_ms, 0, 2000, R.extra_context_ms.default),
            crossfade_ms: clampNum(p.crossfade_ms, 0, 500, R.crossfade_ms.default),
            output_prefill_ms: prefill,
            output_max_queue_ms: clampNum(p.output_max_queue_ms, prefill + 5, 600, Math.max(70, prefill + 5)),
            latency_budget_ms: clampNum(p.latency_budget_ms, R.latency_budget_ms.min, R.latency_budget_ms.max, R.latency_budget_ms.default),
            pitch_semitones: clampNum(p.pitch_semitones, -48, 48, 0),
            index_rate: clampNum(p.index_rate, 0, 1, 0),
            protect: clampNum(p.protect, 0, 0.5, R.protect.default),
            speaker_id: p.speaker_id,
            formant_shift: clampNum(p.formant_shift, -1, 1, 0),
            input_gain: clampNum(p.input_gain, 0, 16, 1),
            output_gain: clampNum(p.output_gain, 0, 16, 1),
            worker_priority: p.worker_priority || "auto",
            rms_mix: clampNum(p.rms_mix, 0, 1, R.rms_mix.default),
            formant_semitones: clampNum(p.formant_semitones, -12, 12, 0),
            noise_scale: clampNum(p.noise_scale, 0, 1, R.noise_scale.default),
            silence_db: clampNum(p.silence_db, -90, -10, R.silence_db.default),
        },
        engine: {
            kind: "rvc",
            slot_dir: slotDir,
            hubert: p.rvc_hubert_path,
            fcpe: p.rvc_fcpe_path,
            rmvpe: p.rvc_rmvpe_path,
            pitch_extractor: p.pitch_extractor,
            execution_provider: p.rvc_execution_provider,
            preset: p.rvc_preset,
            gpu_device_index: p.gpu_device_index,
            vram_cap_mb: p.vram_cap_mb,
            silence_skip: p.silence_skip,
            precision: p.precision,
            levers,
            f0_range: p.f0_range,
            block_ms: clampNum(p.block_ms, 0, 200, 0),
            lookahead_ms: clampNum(p.lookahead_ms, R.lookahead_ms.min, R.lookahead_ms.max, R.lookahead_ms.default),
        },
        runs: [{ mode: "paced", name: "paced", seconds }],
        output: { write_audio: false, write_timeline: true },
    };
}

/** What the panel shows of a benchmark report (`BenchReport` JSON). */
export interface BenchSummary {
    /** `passed`, `failed`, `error` or `cancelled` (the run's status). */
    status: string;
    error: string | null;
    blockP50: number | null;
    blockP95: number | null;
    blockP99: number | null;
    rtfP99: number | null;
    latencyP50: number | null;
    latencyP99: number | null;
    underruns: number | null;
    droppedMs: number | null;
    cpuCores: number | null;
    vramMb: number | null;
    /** The EP the model ran on, and why it differs from the request (when it does). */
    epActive: string | null;
    epRequested: string | null;
    epReason: string | null;
    warnings: string[];
}

/** Summary of the first run of a benchmark report; `null` when the reply is not a report. */
export function summarizeBenchReport(raw: unknown): BenchSummary | null {
    if (!raw || typeof raw !== "object") return null;
    const r = raw as Record<string, unknown>;
    const runs = Array.isArray(r.runs) ? r.runs : [];
    const run = (runs[0] ?? null) as Record<string, unknown> | null;
    if (!run || typeof run !== "object") return null;
    const m = numMap(run.metrics);
    const engine = (r.engine && typeof r.engine === "object" ? r.engine : {}) as Record<string, unknown>;
    const get = (k: string): number | null => (k in m ? m[k] : null);
    return {
        status: str(run.status) ?? "error",
        error: str(run.error),
        blockP50: get("block_ms_p50"),
        blockP95: get("block_ms_p95"),
        blockP99: get("block_ms_p99"),
        rtfP99: get("rtf_p99"),
        latencyP50: get("latency_ms_p50"),
        latencyP99: get("latency_ms_p99"),
        underruns: get("underruns"),
        droppedMs: get("dropped_ms"),
        cpuCores: get("cpu_cores"),
        vramMb: get("vram_engine_mb"),
        epActive: str(engine.execution_provider_active),
        epRequested: str(engine.execution_provider_requested) ?? str(engine.execution_provider),
        epReason: str(engine.execution_provider_reason),
        warnings: strList(engine.warnings),
    };
}

/** A live-record summary (`voice_changer_live_record_stop`), or `null`. */
export function normalizeLiveRecordSummary(raw: unknown): VcLiveRecordSummary | null {
    if (!raw || typeof raw !== "object") return null;
    const r = raw as Record<string, unknown>;
    const dir = str(r.report_dir);
    if (!dir) return null;
    return {
        seconds: num(r.seconds) ?? 0,
        ticks: num(r.ticks) ?? 0,
        report_dir: dir,
        metrics: numMap(r.metrics),
        notes: strList(r.notes),
    };
}
