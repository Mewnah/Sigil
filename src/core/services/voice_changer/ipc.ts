/**
 * Voice-changer IPC contract (P3-S0, frozen; change only through INTEGRATE): command names, event names
 * and payload types of the `voice-changer` plugin, mirroring the Rust commands
 * (`src-tauri/src/services/voice_changer/{commands,capabilities,cmd_import,cmd_bench}.rs`) and
 * `params_contract.json` (`commands_p3`, `events_p3`, `capabilities`, `import_error_codes`).
 * `__tests__/ipc.test.ts` checks every command against `src-tauri/plugin-commands/voice-changer.txt`.
 *
 * Payloads are snake_case (like the params and metrics); command arguments are camelCase (Tauri), e.g.
 * `invoke(VC_COMMANDS.renameSlot, { id, name })`, `invoke(VC_COMMANDS.importVoice, { request })`.
 *
 * Owners: capabilities (scaffold), import / slot store (P3-B), autotune / live record (P3-F), TensorRT
 * progress (P3-C2); the UI is P3-G's. Pure module: no Tauri, DOM or `window`.
 */
import {
    BASELINE_DENOISERS,
    BASELINE_PITCH_EXTRACTORS,
    BASELINE_PRECISIONS,
    BASELINE_RVC_PROFILES,
    EXECUTION_PROVIDERS,
    normalizeDenoiser,
    normalizeExecutionProvider,
    normalizePitchExtractor,
    normalizePrecision,
    normalizeRvcProfile,
    VC_LEVERS,
    type Denoiser,
    type PitchExtractorId,
    type Precision,
    type RvcExecutionProviderId,
    type RvcProfileId,
    type VcLever,
} from "./contract";

const P = "plugin:voice-changer|";

/** Every `voice-changer` plugin command (full IPC names). */
export const VC_COMMANDS = {
    // P0
    getParams: `${P}get_voice_changer_params`,
    setParams: `${P}set_voice_changer_params`,
    setEnabled: `${P}set_voice_changer_enabled`,
    start: `${P}start_voice_changer`,
    stop: `${P}stop_voice_changer`,
    isRunning: `${P}is_voice_changer_running`,
    getMetrics: `${P}get_voice_changer_metrics`,
    pushAuxPcm: `${P}voice_changer_push_aux_pcm`,
    pushAuxWav: `${P}voice_changer_push_aux_wav`,
    captureDebug: `${P}capture_voice_changer_debug`,
    listModelSlots: `${P}list_voice_changer_model_slots`,
    // P3 (P3-S0)
    getCapabilities: `${P}voice_changer_get_capabilities`,
    importVoice: `${P}voice_changer_import`,
    cancelImport: `${P}voice_changer_cancel_import`,
    listSlots: `${P}voice_changer_list_slots`,
    renameSlot: `${P}voice_changer_rename_slot`,
    deleteSlot: `${P}voice_changer_delete_slot`,
    duplicateSlot: `${P}voice_changer_duplicate_slot`,
    setActiveSlot: `${P}voice_changer_set_active_slot`,
    mergeSlots: `${P}voice_changer_merge_slots`,
    importDefaultVoice: `${P}voice_changer_import_default_voice`,
    autotune: `${P}voice_changer_autotune`,
    cancelAutotune: `${P}voice_changer_cancel_autotune`,
    getAutotune: `${P}voice_changer_get_autotune`,
    liveRecordStart: `${P}voice_changer_live_record_start`,
    liveRecordStop: `${P}voice_changer_live_record_stop`,
} as const;

/** Every `voice-changer` event. */
export const VC_EVENTS = {
    started: "voice_changer:started",
    stopped: "voice_changer:stopped",
    /** `{ code, message }` (see `VoiceChangerConversionErrorPayload`). */
    error: "voice_changer:error",
    /** Effective params after Rust changed what was sent. */
    params: "voice_changer:params",
    /** `VcImportProgress` (P3-B). */
    importProgress: "voice_changer:import_progress",
    /** No payload: re-list the slots (P3-B). */
    slotsChanged: "voice_changer:slots_changed",
    /** `VcAutotuneProgress` (P3-F). */
    autotuneProgress: "voice_changer:autotune_progress",
    /** `VcTrtProgress` (P3-C2). */
    trtProgress: "voice_changer:trt_progress",
} as const;

/**
 * The `bench` plugin (P1-F / P3-F), which runs "Benchmark this PC" (`run_benchmark { spec, outDir? }`
 * -> the report JSON). Not voice-changer commands: `src-tauri/plugin-commands/bench.txt`.
 */
export const BENCH_IPC = {
    run: "plugin:bench|run_benchmark",
    cancel: "plugin:bench|cancel_benchmark",
    /** `{ run, run_index, run_count, fraction, message }`, at most 4 per second. */
    progressEvent: "bench:progress",
    /** `{ ok, report, error }` when a benchmark ends. */
    doneEvent: "bench:done",
} as const;

/** Payload of `BENCH_IPC.progressEvent`. */
export interface BenchProgress {
    run: string;
    run_index: number;
    run_count: number;
    fraction: number;
    message: string;
}

// ---------------------------------------------------------------------------------------------
// Capabilities
// ---------------------------------------------------------------------------------------------

/** `voice_changer_get_capabilities`: the values this build runs. The UI offers only these. */
export interface VcCapabilities {
    pitch_extractors: PitchExtractorId[];
    rvc_execution_providers: RvcExecutionProviderId[];
    rvc_presets: RvcProfileId[];
    precisions: Precision[];
    denoisers: Denoiser[];
    /** Levers the engine can switch on (the rest always run their legacy path). */
    levers: VcLever[];
    /** `manifest.format` values the engine runs (`rvc`, `sigil-rvc-1`). */
    slot_formats: string[];
    /** `.pth` / `.zip` import and the slot store. */
    importer: boolean;
    autotune: boolean;
    live_record: boolean;
    max_hot_slots: number;
}

/** What every build runs (used before the reply arrives, or without a desktop backend). */
export const BASELINE_CAPABILITIES: VcCapabilities = {
    pitch_extractors: [...BASELINE_PITCH_EXTRACTORS],
    rvc_execution_providers: [...EXECUTION_PROVIDERS],
    rvc_presets: [...BASELINE_RVC_PROFILES],
    precisions: [...BASELINE_PRECISIONS],
    denoisers: [...BASELINE_DENOISERS],
    levers: [],
    slot_formats: ["rvc"],
    importer: false,
    autotune: false,
    live_record: false,
    max_hot_slots: 1,
};

function uniq<T>(values: T[]): T[] {
    return [...new Set(values)];
}

/** Keep only contract values (normalized exactly); the baseline is always included. */
export function normalizeCapabilities(raw: unknown): VcCapabilities {
    if (!raw || typeof raw !== "object") return { ...BASELINE_CAPABILITIES };
    const r = raw as Partial<Record<keyof VcCapabilities, unknown>>;
    const list = (v: unknown): string[] => (Array.isArray(v) ? v.filter((x): x is string => typeof x === "string") : []);
    const exact = <T extends string>(values: string[], norm: (s: string) => T | "", base: readonly T[]): T[] =>
        uniq([...base, ...values.map((v) => norm(v)).filter((v, i): v is T => v !== "" && v === values[i].trim().toLowerCase())]);
    const b = BASELINE_CAPABILITIES;
    return {
        pitch_extractors: exact(list(r.pitch_extractors), normalizePitchExtractor, b.pitch_extractors),
        rvc_execution_providers: exact(list(r.rvc_execution_providers), normalizeExecutionProvider, b.rvc_execution_providers),
        rvc_presets: exact(list(r.rvc_presets), normalizeRvcProfile, b.rvc_presets),
        precisions: exact(list(r.precisions), normalizePrecision, b.precisions),
        denoisers: exact(list(r.denoisers), normalizeDenoiser, b.denoisers),
        levers: uniq(list(r.levers).filter((l): l is VcLever => (VC_LEVERS as readonly string[]).includes(l))),
        slot_formats: uniq(["rvc", ...list(r.slot_formats)]),
        importer: r.importer === true,
        autotune: r.autotune === true,
        live_record: r.live_record === true,
        max_hot_slots:
            typeof r.max_hot_slots === "number" && Number.isFinite(r.max_hot_slots) && r.max_hot_slots >= 1
                ? Math.floor(r.max_hot_slots)
                : 1,
    };
}

// ---------------------------------------------------------------------------------------------
// Levers in use (metrics `levers`)
// ---------------------------------------------------------------------------------------------

/** Rust `engine::options::EffectiveLevers`: what the running engine really uses. */
export type VcEffectiveLevers = Record<VcLever, boolean> & {
    accelerator: "none" | "tensorrt_rtx" | "dml_hp";
    precision: Precision;
    /** Why a requested lever or accelerator is not used, one line each. */
    notes: string[];
};

export function normalizeEffectiveLevers(raw: unknown): VcEffectiveLevers | null {
    if (!raw || typeof raw !== "object") return null;
    const r = raw as Record<string, unknown>;
    const levers = Object.fromEntries(VC_LEVERS.map((l) => [l, r[l] === true])) as Record<VcLever, boolean>;
    const acc = r.accelerator === "tensorrt_rtx" || r.accelerator === "dml_hp" ? r.accelerator : "none";
    return {
        ...levers,
        accelerator: acc,
        precision: normalizePrecision(r.precision),
        notes: Array.isArray(r.notes) ? r.notes.filter((n): n is string => typeof n === "string") : [],
    };
}

// ---------------------------------------------------------------------------------------------
// Import and slot store (P3-B)
// ---------------------------------------------------------------------------------------------

/** One voice slot (Rust `vc::import::slot::SlotInfo`). */
export interface VcSlotInfo {
    /** Folder name under `models/` (stable id). */
    id: string;
    name: string;
    /** Absolute folder path (what `rvc_slot_dir` stores). */
    path: string;
    /** `rvc` (bring-your-own ONNX) or `sigil-rvc-1` (imported template voice). */
    format: string;
    template: string | null;
    version: string | null;
    sample_rate: number | null;
    f0: boolean | null;
    embedder: string | null;
    vocoder: string | null;
    has_index: boolean;
    speaker_count: number | null;
    fp16_ok: boolean | null;
    source_sha256: string | null;
    created_unix_ms: number | null;
    /** Why the slot cannot run in this build, if it cannot. */
    problem: string | null;
}

/** `voice_changer_import { request }`. */
export interface VcImportRequest {
    /** The `.pth` or voice `.zip` path. */
    source: string;
    name?: string | null;
    /** Re-import into this existing slot. */
    slot_id?: string | null;
    /** Make it the active voice when done. */
    activate?: boolean;
    smoke_fp16?: boolean;
    /**
     * Additive (P3-B): a retrieval `.index` file to adopt. Default: the archive's own, or the
     * `added_*.index` next to a `.pth` when that choice is unambiguous.
     */
    index?: string | null;
}

export interface VcImportResult {
    job_id: string;
    slot: VcSlotInfo;
    /** The same file was imported before; its slot was reused. */
    cached: boolean;
    warnings: string[];
    elapsed_ms: number;
}

export type VcImportStage = "queued" | "reading" | "converting" | "smoke" | "writing" | "done" | "failed";

/** Payload of `VC_EVENTS.importProgress`. */
export interface VcImportProgress {
    job_id: string;
    stage: VcImportStage;
    fraction: number;
    message: string;
}

/** `voice_changer_merge_slots { request }`: weighted merge of two same-template voices. */
export interface VcMergeRequest {
    a: string;
    b: string;
    /** Weight of `a` (0..1). */
    weight_a: number;
    name: string;
}

/** Import error codes (Rust `vc::import::ErrorCode`; the error string is `"<code>: <message>"`). */
export const IMPORT_ERROR_CODES = [
    "io",
    "job",
    "not_a_checkpoint",
    "legacy_format",
    "torchscript",
    "archive_rejected",
    "ambiguous_archive",
    "pickle_rejected",
    "tensor_invalid",
    "missing_tensor",
    "shape_mismatch",
    "non_finite",
    "unsupported_model",
    "no_template",
    "layout_invalid",
    "session",
    "time_limit",
    "precision_refused",
] as const;
export type ImportErrorCode = (typeof IMPORT_ERROR_CODES)[number];

/** The code of an import error string (`"<code>: <message>"`), or `null`. */
export function importErrorCode(error: unknown): ImportErrorCode | null {
    const text = typeof error === "string" ? error : error instanceof Error ? error.message : "";
    const code = text.split(":", 1)[0]?.trim() ?? "";
    return (IMPORT_ERROR_CODES as readonly string[]).includes(code) ? (code as ImportErrorCode) : null;
}

// ---------------------------------------------------------------------------------------------
// Autotune and live record (P3-F), TensorRT build progress (P3-C2)
// ---------------------------------------------------------------------------------------------

/** `voice_changer_autotune { request }`. */
export interface VcAutotuneRequest {
    /** Default: the active voice. */
    slot_dir?: string | null;
    /** EP ids to try; empty = every one available. */
    execution_providers?: RvcExecutionProviderId[];
    quick?: boolean;
    max_seconds?: number | null;
}

/** The chosen configuration (stored per GPU and driver). */
export interface VcAutotuneResult {
    gpu: string | null;
    driver: string | null;
    created_unix_ms: number;
    /** A snake_case `VoiceChangerParamsRust` patch: apply it through the params sync. */
    params: Record<string, unknown>;
    rtf_p99: number;
    block_ms_p95: number;
    latency_ms_p50: number | null;
    fidelity: number | null;
    report_dir: string | null;
    notes: string[];
    // Additive (P3-F): optional on the wire.
    /** The winner met the RTF limit (false: the fastest configuration found). */
    met_limit?: boolean;
    /** Slot folder tuned with, and its format (`rvc`, `sigil-rvc-1`). */
    slot?: string | null;
    format?: string | null;
    execution_provider_active?: string | null;
    latency_ms_p99?: number | null;
    /** The winner's fidelity metrics against its offline ceiling. */
    fidelity_metrics?: Record<string, number>;
    /** How long the sweep took. */
    seconds?: number | null;
    /**
     * The voice changer settings every candidate was measured with (Rust `TuneConditions`; Rust
     * fills them from the params, the UI never sends them).
     */
    conditions?: Record<string, unknown> | null;
    /**
     * `voice_changer_get_autotune` only: what changed since the tune (Rust `stale_reasons`); empty
     * while the stored tune still describes the current settings.
     */
    stale?: string[];
}

/** Payload of `VC_EVENTS.autotuneProgress`. */
export interface VcAutotuneProgress {
    stage: "preparing" | "sweeping" | "verifying" | "done" | "failed";
    fraction: number;
    candidate: string | null;
    message: string;
}

/** `voice_changer_live_record_start { request }`. */
export interface VcLiveRecordRequest {
    /** Default 600 s. */
    max_seconds?: number | null;
    note?: string | null;
}

export interface VcLiveRecordSummary {
    seconds: number;
    ticks: number;
    report_dir: string;
    metrics: Record<string, number>;
    /** Additive (P3-F). */
    notes?: string[];
}

/** Payload of `VC_EVENTS.trtProgress`. */
export interface VcTrtProgress {
    /** Session component (`vc.enc`, `vc.flowdec`, ...). */
    component: string;
    stage: "cache_hit" | "building" | "done" | "failed";
    fraction: number;
    message: string;
    elapsed_ms: number;
}
