/**
 * The **Chunk size** control (the `block_ms` setting): what the choices are, what Auto resolves to on this PC,
 * and for which voices the setting reaches the engine at all.
 *
 * `block_ms` is how much new audio the voice engine converts per model call. 0 is Auto; a size above 0 is the
 * user's and is the block the engine runs (the plan rounds it up to whole 20 ms, at most 200 ms:
 * `engine/plan.rs` `block_frames`), on every execution provider and preset. It is one global setting: a preset
 * (`presets.ts`), another voice or a restart never changes it (Rust `state.rs`), and the output-buffer floor of a
 * CPU voice is sized for the block that runs (`factory.rs` `template_block`, `block_buffer.rs`).
 *
 * Which voices read it (Rust `factory.rs` / `families/*`):
 * - an imported (Sigil template) voice, or no listed voice (the default setup): the block;
 * - so-vits-svc voices: the block too (`engine_preset`, planned to 20 ms), but their Auto is the device-derived
 *   block (the Streamer's per-PC rule is the template engine's);
 * - DDSP-SVC voices: the block rounded up to 10 ms, 50 ms at least; Auto is the voice's preset (100 or 500 ms);
 * - Sigil-ZS and MeanVC2 voices run the block of their preset / tick option, and a sidecar voice only proposes a
 *   frame to its helper, which answers with its own: for these the control is shown disabled, with the reason;
 * - a bring-your-own ONNX (legacy) voice converts one audio device buffer per call (the Audio device buffer under
 *   Advanced), so the setting does not apply.
 *
 * Pure module: no Tauri, DOM or `window`. `__tests__/chunkSize.test.ts` pins it against the Rust sources.
 */
import { engineBlockMs } from "./chunkFrames";
import { rvcBaseEp, effectiveRvcEp, VC_NUMBER_RANGES, type RvcExecutionProviderId, type RvcProfileId } from "./contract";
import { familyOptionsForm, type VoiceKind } from "./families";
import type { VcFamilyId } from "./ipc";
import { STREAMER_KEEP_DERIVED_FROM_MS, streamerAutoBlockMs, streamerShortChunkNote, type StreamerShortChunkNote } from "./options";

/** The engine plans a block in whole multiples of this (ms; Rust `plan::block_frames`: even counts of 10 ms frames). */
export const CHUNK_SIZE_STEP_MS = 20;
/** The shortest block the engine plans. */
export const CHUNK_SIZE_MIN_MS = 20;
/** The longest block (the `block_ms` range, Rust `MAX_BLOCK_FRAMES` x 10 ms). */
export const CHUNK_SIZE_MAX_MS = VC_NUMBER_RANGES.block_ms.max;
/** Rust `MAX_DERIVED_BLOCK_MS`: the block derived from the device buffer is capped here. */
export const DERIVED_BLOCK_MAX_MS = 200;
/** The capture rate assumed while the engine is stopped (the real one is only known once it runs). */
export const ASSUMED_DEVICE_RATE = 48_000;

/** DDSP-SVC runs blocks of 50 ms at least, in steps of 10 ms (Rust `ddsp_svc::MIN_BLOCK_MS`, `tuning_for`). */
export const DDSP_CHUNK_SIZE_MIN_MS = 50;
export const DDSP_CHUNK_SIZE_MAX_MS = 500;
export const DDSP_CHUNK_SIZE_STEP_MS = 10;
/** The DDSP-SVC block of each preset (Rust `ddsp_svc::preset_shape`). */
export const DDSP_PRESET_BLOCK_MS = { sigil_singing: 100, sigil_singing_fast: 100, upstream_gui: 500 } as const;

/** Why the control does not apply to a voice (one line each in the inspector). */
export type ChunkSizeFixedReason = "legacy" | "sigil_zs" | "meanvc" | "sidecar_v2" | "family";

/** How each voice family treats `block_ms` (Rust `families/*`; `__tests__/chunkSize.test.ts` checks it against the sources). */
export const FAMILY_CHUNK_SIZE: Readonly<Record<VcFamilyId, "engine" | "ddsp" | Exclude<ChunkSizeFixedReason, "legacy" | "family">>> = {
    sovits: "engine",
    ddsp_svc: "ddsp",
    sigil_zs: "sigil_zs",
    meanvc: "meanvc",
    sidecar_v2: "sidecar_v2",
};

/** What the Chunk size does for the selected voice. */
export type ChunkSizeScope =
    /** The size is the engine block, planned to 20 ms. `streamerRule`: Auto on the Streamer / processor takes the block of the PC's cores (template voices). */
    | { kind: "engine"; streamerRule: boolean }
    /** DDSP-SVC: the size is the block (10 ms steps, 50 ms at least); Auto is the preset's block. */
    | { kind: "ddsp"; presetBlockMs: number }
    /** The setting does not reach this voice. */
    | { kind: "fixed"; reason: ChunkSizeFixedReason };

/** The DDSP-SVC block of the preset an options object names (the default preset for anything else). */
export function ddspPresetBlockMs(familyOptions: unknown): number {
    return DDSP_PRESET_BLOCK_MS[familyOptionsForm("ddsp_svc", familyOptions).preset];
}

/**
 * Which way the Chunk size reaches a voice. `kind`: what runs it (`families.ts` `voiceKind`; no listed voice counts as
 * an imported one, the default setup), `family` / `familyOptions`: a family voice's id and its `manifest.family`.
 */
export function chunkSizeScope(kind: VoiceKind, family: VcFamilyId | null, familyOptions?: unknown): ChunkSizeScope {
    if (kind === "legacy") return { kind: "fixed", reason: "legacy" };
    if (kind === "family") {
        if (!family) return { kind: "fixed", reason: "family" };
        const how = FAMILY_CHUNK_SIZE[family];
        if (how === "engine") return { kind: "engine", streamerRule: false };
        if (how === "ddsp") return { kind: "ddsp", presetBlockMs: ddspPresetBlockMs(familyOptions) };
        return { kind: "fixed", reason: how };
    }
    return { kind: "engine", streamerRule: true };
}

/** The sizes offered besides Auto (ms): every block the engine plans for template voices, 60 ms and up in 20 ms steps for DDSP-SVC. */
export function chunkSizeChoices(scope: ChunkSizeScope): number[] {
    if (scope.kind === "fixed") return [];
    const first = scope.kind === "ddsp" ? 60 : CHUNK_SIZE_MIN_MS;
    const sizes: number[] = [];
    for (let ms = first; ms <= CHUNK_SIZE_MAX_MS; ms += CHUNK_SIZE_STEP_MS) sizes.push(ms);
    return sizes;
}

/** The block (ms) a stored size runs as in `scope`: the engine plans to whole 20 ms, DDSP-SVC to 10 ms within its range. */
export function runningChunkSizeMs(scope: ChunkSizeScope, storedMs: number): number | null {
    if (scope.kind === "fixed" || !Number.isFinite(storedMs) || storedMs <= 0) return null;
    if (scope.kind === "ddsp") {
        const ms = Math.ceil(storedMs / DDSP_CHUNK_SIZE_STEP_MS - 1e-9) * DDSP_CHUNK_SIZE_STEP_MS;
        return Math.min(DDSP_CHUNK_SIZE_MAX_MS, Math.max(DDSP_CHUNK_SIZE_MIN_MS, ms));
    }
    return Math.max(CHUNK_SIZE_MIN_MS, engineBlockMs(storedMs));
}

export interface ChunkSizeInput {
    scope: ChunkSizeScope;
    profile: RvcProfileId;
    /** The stored RVC execution provider (`""`: the profile's default). */
    ep: RvcExecutionProviderId | "";
    /** Physical cores (system info); `null` / `undefined` while the probe has not answered. */
    physicalCores: number | null | undefined;
    /** The Audio device buffer setting (frames). */
    chunkFrames: number;
    /** The device block of the running engine (ms; the `chunk_ms` metric: it includes `device_block` and the real capture rate), else `null`. */
    runningDeviceMs?: number | null;
    /** The chunk the running engine reports converting per call (the `engine_block_ms` metric), else `null`. */
    runningBlockMs?: number | null;
    /** The stored Chunk size (`block_ms`; 0 = Auto). */
    storedMs: number;
}

/**
 * The block (ms) Auto runs for an imported voice (Rust `template_block` with a block of 0): derived from the device
 * block (capped at 200 ms, planned to whole 20 ms), and for the Streamer on the processor a derived block under
 * 60 ms is raised to the one its PC's cores keep up with (120 / 160 / 200 ms). `null` when that depends on the PC's
 * cores and they are not known yet.
 */
export function templateAutoBlockMs(
    i: Pick<ChunkSizeInput, "profile" | "ep" | "physicalCores" | "chunkFrames" | "runningDeviceMs"> & { streamerRule: boolean },
): number | null {
    const deviceMs = i.runningDeviceMs != null && i.runningDeviceMs > 0 ? i.runningDeviceMs : (i.chunkFrames / ASSUMED_DEVICE_RATE) * 1000;
    const derived = Math.min(DERIVED_BLOCK_MAX_MS, Number.isFinite(deviceMs) && deviceMs > 0 ? deviceMs : 0);
    const onProcessor = rvcBaseEp(effectiveRvcEp(i.ep, i.profile)) === "cpu";
    if (i.streamerRule && i.profile === "streamer" && onProcessor && derived < STREAMER_KEEP_DERIVED_FROM_MS) {
        const cores = i.physicalCores;
        return typeof cores === "number" && Number.isFinite(cores) && cores >= 0 ? streamerAutoBlockMs(Math.trunc(cores)) : null;
    }
    return Math.max(CHUNK_SIZE_MIN_MS, engineBlockMs(derived));
}

export interface ChunkSizeView {
    /** The setting reaches this voice. */
    applies: boolean;
    /** Why not, when it does not. */
    reason: ChunkSizeFixedReason | null;
    /** The sizes offered besides Auto (ms, ascending); the stored size is in the list even when the engine would plan another. */
    choices: number[];
    /**
     * What Auto runs on this PC (ms); `null` when it does not apply, or the PC's cores are not known yet. An estimate from
     * this PC's settings and cores unless {@link autoExact}.
     */
    autoMs: number | null;
    /**
     * `autoMs` is the chunk the running engine reports converting on Auto, not an estimate. The estimate mirrors Rust's
     * rule, which a developer switch (`SIGIL_VC_STREAMER_BLOCK`, `SIGIL_VC_PHYSICAL_CORES`), a device rate that is not 48 kHz
     * or a graphics-card provider that falls back to the processor at runtime can change.
     */
    autoExact: boolean;
    /** The stored size is not one the engine runs as it is: what it runs as (ms); `null` on Auto and when it runs as set. */
    runsAsMs: number | null;
    /** The warning for an explicit size shorter than the Streamer on the processor keeps up with on this PC (see `streamerShortChunkNote`). */
    shortChunk: StreamerShortChunkNote | null;
}

/** Everything the Chunk size control shows for the selected voice, settings and PC. */
export function chunkSizeView(i: ChunkSizeInput): ChunkSizeView {
    const { scope } = i;
    if (scope.kind === "fixed") {
        return { applies: false, reason: scope.reason, choices: [], autoMs: null, autoExact: false, runsAsMs: null, shortChunk: null };
    }
    const choices = chunkSizeChoices(scope);
    const stored = Number.isFinite(i.storedMs) && i.storedMs > 0 ? i.storedMs : 0;
    if (stored > 0 && !choices.includes(stored)) choices.push(stored);
    choices.sort((a, b) => a - b);
    const runs = runningChunkSizeMs(scope, stored);
    // On Auto the running engine says what it converts: that is Auto's chunk, whatever the estimate would say.
    const reported = i.runningBlockMs != null && Number.isFinite(i.runningBlockMs) && i.runningBlockMs > 0 ? i.runningBlockMs : null;
    const autoExact = stored === 0 && reported !== null;
    const estimate = scope.kind === "ddsp" ? scope.presetBlockMs : templateAutoBlockMs({ ...i, streamerRule: scope.streamerRule });
    const shortChunk =
        scope.kind === "engine" && scope.streamerRule
            ? streamerShortChunkNote({ profile: i.profile, ep: i.ep, physicalCores: i.physicalCores, blockMs: stored })
            : null;
    return {
        applies: true,
        reason: null,
        choices,
        autoMs: autoExact ? reported : estimate,
        autoExact,
        runsAsMs: runs !== null && runs !== stored ? runs : null,
        shortChunk,
    };
}
