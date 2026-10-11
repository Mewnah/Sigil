/**
 * The choices the voice-changer UI offers, from what this build runs
 * (`voice_changer_get_capabilities`, `ipc.ts` `VcCapabilities`), with no silent remapping (P3-G
 * acceptance): every value the build runs is offered, and a stored value it cannot run stays visible
 * and selected, marked with what runs instead (the same stand-in / base-rung rules Rust applies), so
 * the UI never shows one thing while the engine runs another.
 *
 * Pure module: no Tauri, DOM or `window`.
 */
import {
    DENOISERS,
    effectivePitchExtractor,
    effectiveRvcEp,
    GPU_ONLY_PITCH_EXTRACTORS,
    PITCH_EXTRACTORS,
    PRECISIONS,
    rvcBaseEp,
    RVC_EXECUTION_PROVIDERS,
    RVC_PROFILES,
    type Denoiser,
    type ExecutionProviderId,
    type PitchExtractorId,
    type Precision,
    type RvcExecutionProviderId,
    type RvcProfileId,
    type VcLever,
} from "./contract";
import type { VcCapabilities, VcEffectiveLevers } from "./ipc";

/** One choice of a select. */
export interface ChoiceOption<T extends string> {
    value: T;
    /** This build runs it. */
    available: boolean;
    /** What runs instead while it is selected (a stand-in or fallback), or `null` when it runs itself. */
    runsAs: string | null;
}

/** Contract order, the values the build runs plus the stored one (so it is never hidden). */
function offered<T extends string>(all: readonly T[], available: readonly string[], stored: string): T[] {
    return all.filter((v) => available.includes(v) || v === stored);
}

/**
 * F0 methods: every one the build runs, plus the stored one. `runsAs` is what runs on `ep` (a
 * stand-in while the build cannot run the method, fcpe for a GPU-only method on the CPU).
 */
export function pitchExtractorOptions(
    caps: VcCapabilities,
    stored: PitchExtractorId,
    ep: RvcExecutionProviderId,
): (ChoiceOption<PitchExtractorId> & { gpuOnly: boolean })[] {
    return offered(PITCH_EXTRACTORS, caps.pitch_extractors, stored).map((value) => {
        const run = effectivePitchExtractor(value, ep, caps.pitch_extractors);
        return {
            value,
            available: caps.pitch_extractors.includes(value),
            runsAs: run.fellBack ? run.extractor : null,
            gpuOnly: GPU_ONLY_PITCH_EXTRACTORS.includes(value),
        };
    });
}

/** RVC execution providers: an accelerator the build cannot run builds on its base rung. */
export function executionProviderOptions(
    caps: VcCapabilities,
    stored: RvcExecutionProviderId,
): ChoiceOption<RvcExecutionProviderId>[] {
    return offered(RVC_EXECUTION_PROVIDERS, caps.rvc_execution_providers, stored).map((value) => {
        const available = caps.rvc_execution_providers.includes(value);
        const base = rvcBaseEp(value);
        return { value, available, runsAs: available || base === value ? null : base };
    });
}

/** RVC presets: Studio-TRT without TensorRT-RTX runs its Quality shapes on CUDA. */
export function profileOptions(caps: VcCapabilities, stored: RvcProfileId): ChoiceOption<RvcProfileId>[] {
    return offered(RVC_PROFILES, caps.rvc_presets, stored).map((value) => {
        const available = caps.rvc_presets.includes(value);
        return { value, available, runsAs: available ? null : value === "studio_trt" ? "cuda" : null };
    });
}

/** Template-voice precision. */
export function precisionOptions(caps: VcCapabilities, stored: Precision): ChoiceOption<Precision>[] {
    return offered(PRECISIONS, caps.precisions, stored).map((value) => {
        const available = caps.precisions.includes(value);
        return { value, available, runsAs: available ? null : "fp32" };
    });
}

/**
 * The denoiser that runs for `kind` (Rust `denoise::effective_kind`): itself when the build runs it,
 * DPDFNet for DeepFilterNet3 (the same architecture, streaming), else none.
 */
export function effectiveDenoiser(kind: Denoiser, available: readonly string[]): Denoiser {
    if (kind === "off" || available.includes(kind)) return kind;
    if (kind === "dfn3" && available.includes("dpdfnet")) return "dpdfnet";
    return "off";
}

/** Input denoisers (the voice-changer path only; STT keeps the raw microphone). */
export function denoiserOptions(caps: VcCapabilities, stored: Denoiser): ChoiceOption<Denoiser>[] {
    return offered(DENOISERS, caps.denoisers, stored).map((value) => {
        const available = caps.denoisers.includes(value);
        const run = effectiveDenoiser(value, caps.denoisers);
        return { value, available, runsAs: run === value ? null : run };
    });
}

// ---------------------------------------------------------------------------------------------
// Levers
// ---------------------------------------------------------------------------------------------

/** Levers the legacy (bring-your-own ONNX) path can also use; the rest need an imported voice. */
export const LEGACY_PATH_LEVERS: readonly VcLever[] = ["anchored_noise", "partial_decode", "governor"];

/**
 * Where GPU priority runs (Rust `lever_applies`): the highest-priority CUDA stream on the CUDA rung
 * needs a hardware NVIDIA adapter (`cuda_stream::available`), the high-priority DirectML queue on the
 * DirectML rung its D3D12 entry points (`dml_queue::available`). `null`: not known.
 */
export interface GpuPrioritySupport {
    cuda: boolean | null;
    dml: boolean;
}

/** DXGI vendor id of NVIDIA (Rust `gpu_mem::VENDOR_NVIDIA`). */
const VENDOR_NVIDIA = 0x10de;

/**
 * Whether the system info (`plugin:sysinfo|get_system_info`) lists a hardware NVIDIA adapter, as
 * `cuda_stream::available` checks it (`gpu_mem::nvidia_adapter_indices`: NVIDIA vendor id, not the
 * software renderer); `null` before the probe, or when the graphics driver never answered it.
 */
export function nvidiaAdapterPresent(
    info: { gpus: readonly { vendor_id: number; software: boolean }[]; gpu_probe?: string } | null | undefined,
): boolean | null {
    if (!info || info.gpu_probe === "missing") return null;
    return info.gpus.some((g) => g.vendor_id === VENDOR_NVIDIA && !g.software);
}

/**
 * {@link GpuPrioritySupport} from the capabilities, exact where they tell:
 * - the DirectML queue runs exactly when the build offers `dml_hp` (`accelerator_available` of it is
 *   `cfg!(windows) && dml_queue::available()`);
 * - `gpu_priority` is offered when either queue runs (`lever_available`): not offered rules out both,
 *   offered without the DirectML queue means the CUDA stream; TensorRT-RTX also needs an NVIDIA
 *   adapter. Otherwise `nvidiaAdapter` decides: whether the system info lists a hardware NVIDIA
 *   adapter (the same DXGI list `cuda_stream` checks), `null` while unknown.
 */
export function gpuPrioritySupport(caps: VcCapabilities, nvidiaAdapter: boolean | null): GpuPrioritySupport {
    const eps = caps.rvc_execution_providers;
    const dml = eps.includes("dml_hp");
    let cuda: boolean | null = nvidiaAdapter;
    if (!caps.levers.includes("gpu_priority")) cuda = false;
    else if (!dml || eps.includes("tensorrt_rtx")) cuda = true;
    return { cuda, dml };
}

/**
 * Whether `lever` applies on the EP ladder rung `base` (Rust `capabilities::lever_applies`, for a
 * lever this build runs): GPU retrieval and IOBinding need a GPU, CUDA graphs CUDA, GPU priority the
 * rung's priority queue (`gpu`; without it, or while unknown, it is not reported as off).
 */
export function leverApplies(lever: VcLever, base: ExecutionProviderId, gpu?: GpuPrioritySupport): boolean {
    switch (lever) {
        case "gpu_knn":
        case "io_binding":
            return base !== "cpu";
        case "gpu_priority":
            if (base === "cuda") return gpu?.cuda ?? true;
            if (base === "dml") return gpu?.dml ?? true;
            return false;
        case "cuda_graphs":
            return base === "cuda";
        default:
            return true;
    }
}

export interface LeverView {
    lever: VcLever;
    /** This build can switch it on. */
    available: boolean;
    /** It applies on the selected EP's rung. */
    applies: boolean;
    /** Whether the running engine uses it (`null` while stopped or before the engine reports it). */
    inUse: boolean | null;
    /** Why the running engine does not use it although requested (Rust's note), if any. */
    note: string | null;
}

/**
 * Every lever's status for the selected EP, and what the running engine reports. `nvidiaAdapter`:
 * whether the PC has a hardware NVIDIA adapter (system info; `null` while unknown).
 */
export function leverViews(
    levers: readonly VcLever[],
    caps: VcCapabilities,
    ep: RvcExecutionProviderId,
    running: VcEffectiveLevers | null,
    nvidiaAdapter: boolean | null = null,
): LeverView[] {
    const base = rvcBaseEp(ep);
    const gpu = gpuPrioritySupport(caps, nvidiaAdapter);
    return levers.map((lever) => {
        const note = running?.notes.find((n) => n.startsWith(`${lever}:`)) ?? null;
        return {
            lever,
            available: caps.levers.includes(lever),
            applies: leverApplies(lever, base, gpu),
            inUse: running ? running[lever] : null,
            note: note ? note.slice(lever.length + 1).trim() : null,
        };
    });
}

/** Notes of the running engine that are not about one lever (accelerator, precision, F0). */
export function generalLeverNotes(running: VcEffectiveLevers | null, levers: readonly VcLever[]): string[] {
    if (!running) return [];
    return running.notes.filter((n) => !levers.some((l) => n.startsWith(`${l}:`)));
}

/**
 * Physical cores at or under which the Streamer preset on the processor falls behind while a game is
 * running. The headless benchmark of the final build (`docs/benchmarks/VC_BENCHMARKS.md` -> "The CPU
 * Streamer on 4, 5 and 6 physical cores") with the game stand-in (4 threads at 60 %): on 6 cores the voice
 * skipped 7-10 % of the audio, on 5 cores 14 %, on 4 cores 19-20 % (3 passes each); an 8-core PC skipped none.
 */
export const STREAMER_FEW_CORES_MAX = 6;

/**
 * The Block size (ms) from which the Streamer kept up under that stand-in on a 6-core slice: 160 ms dropped nothing
 * in one run, and so did 200 ms.
 */
export const STREAMER_KEEPS_UP_BLOCK_MS = 160;

/**
 * The same for 5 cores and fewer: 200 ms is the only longer block that was run on 4 cores (nothing dropped under the
 * stand-in, one run; its paced run missed the 98 % played requirement with 2 s of dry output a minute), and 5 cores
 * were never run on a longer block, so they take the 4-core figure. 160 ms is not measured on those PCs and
 * must not hide the note. 1 to 3 cores were not measured at all and are treated like 4.
 */
export const STREAMER_KEEPS_UP_BLOCK_MS_FEW = 200;

/** Cores from which {@link STREAMER_KEEPS_UP_BLOCK_MS} (160 ms) is the measured remedy; fewer take the 200 ms figure. */
export const STREAMER_CORES_FOR_160_MS = 6;

/** The Block size (ms) from which the note hides on a PC with `cores` physical cores. */
export function streamerKeepsUpBlockMs(cores: number): number {
    return cores >= STREAMER_CORES_FOR_160_MS ? STREAMER_KEEPS_UP_BLOCK_MS : STREAMER_KEEPS_UP_BLOCK_MS_FEW;
}

/**
 * Whether the Streamer preset runs on a processor with so few physical cores that it does not keep up while a
 * game is running: the Streamer preset on the CPU rung (an unset EP follows the preset), a known core count of
 * 1 to [`STREAMER_FEW_CORES_MAX`], and a Block size under [`streamerKeepsUpBlockMs`] for that core count (0 is
 * Auto, which the Streamer on the processor turns into 120 ms). Returns the core count and the Block size that
 * kept up, for the note, or `null` for no note.
 */
export function streamerFewCoresNote(input: {
    profile: RvcProfileId;
    ep: RvcExecutionProviderId | "";
    /** `null` / `undefined` / 0 while the system probe has not answered. */
    physicalCores: number | null | undefined;
    blockMs: number;
}): { cores: number; blockMs: number } | null {
    if (input.profile !== "streamer") return null;
    if (rvcBaseEp(effectiveRvcEp(input.ep, input.profile)) !== "cpu") return null;
    const cores = input.physicalCores;
    if (typeof cores !== "number" || !Number.isFinite(cores) || cores < 1 || cores > STREAMER_FEW_CORES_MAX) return null;
    const whole = Math.trunc(cores);
    const blockMs = streamerKeepsUpBlockMs(whole);
    if (input.blockMs >= blockMs) return null;
    return { cores: whole, blockMs };
}
