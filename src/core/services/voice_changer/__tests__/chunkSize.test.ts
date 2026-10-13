/**
 * The Chunk size control (`block_ms`): the sizes it offers are the ones the engine runs (no size that is planned to
 * another), what Auto resolves to on this PC for each voice, preset and execution provider, which voices the
 * setting reaches (and why not for the others), and that the numbers agree with the Rust sources.
 */
import { describe, expect, it } from "vitest";

import { engineBlockMs } from "../chunkFrames";
import {
    CHUNK_SIZE_MAX_MS,
    CHUNK_SIZE_MIN_MS,
    CHUNK_SIZE_STEP_MS,
    DDSP_CHUNK_SIZE_MAX_MS,
    DDSP_CHUNK_SIZE_MIN_MS,
    DDSP_CHUNK_SIZE_STEP_MS,
    DDSP_PRESET_BLOCK_MS,
    DERIVED_BLOCK_MAX_MS,
    FAMILY_CHUNK_SIZE,
    chunkSizeChoices,
    chunkSizeScope,
    chunkSizeView,
    ddspPresetBlockMs,
    runningChunkSizeMs,
    templateAutoBlockMs,
    type ChunkSizeInput,
    type ChunkSizeScope,
} from "../chunkSize";
import { rvcProfilePatch } from "../presets";
import { VC_NUMBER_RANGES } from "../contract";
import { DDSP_SVC_PRESETS } from "../families";
import { VC_FAMILIES, type VcFamilyId } from "../ipc";
import { STREAMER_KEEP_DERIVED_FROM_MS, streamerAutoBlockMs } from "../options";
import { VoiceChangerStateSchema } from "../schema";

const ENGINE: ChunkSizeScope = { kind: "engine", streamerRule: true };
const SOVITS: ChunkSizeScope = { kind: "engine", streamerRule: false };

/** A Streamer / processor input on a PC with `cores` physical cores, the default 512-frame device buffer, Auto. */
const input = (over: Partial<ChunkSizeInput> = {}): ChunkSizeInput => ({
    scope: ENGINE,
    profile: "streamer",
    ep: "cpu",
    physicalCores: 8,
    chunkFrames: 512,
    runningDeviceMs: null,
    storedMs: 0,
    ...over,
});

describe("the sizes offered", () => {
    it("are every block the engine plans, in 20 ms steps up to the range's end, each run as itself", () => {
        const sizes = chunkSizeChoices(ENGINE);
        expect(sizes).toEqual([20, 40, 60, 80, 100, 120, 140, 160, 180, 200]);
        expect(sizes[0]).toBe(CHUNK_SIZE_MIN_MS);
        expect(sizes[sizes.length - 1]).toBe(CHUNK_SIZE_MAX_MS);
        expect(CHUNK_SIZE_MAX_MS).toBe(VC_NUMBER_RANGES.block_ms.max);
        for (const ms of sizes) {
            expect(ms % CHUNK_SIZE_STEP_MS, `${ms}`).toBe(0);
            // The engine plans the size as it is: nothing offered is rounded to another size.
            expect(engineBlockMs(ms), `${ms} ms`).toBe(ms);
            expect(runningChunkSizeMs(ENGINE, ms)).toBe(ms);
        }
        // The old list offered 30 and 50 ms, which the engine ran as 40 and 60.
        expect(sizes).not.toContain(30);
        expect(sizes).not.toContain(50);
        expect(chunkSizeChoices(SOVITS)).toEqual(sizes);
    });

    it("start at 60 ms for DDSP-SVC voices (they run 50 ms at least) and are none where the setting does not apply", () => {
        expect(chunkSizeChoices({ kind: "ddsp", presetBlockMs: 100 })).toEqual([60, 80, 100, 120, 140, 160, 180, 200]);
        expect(chunkSizeChoices({ kind: "fixed", reason: "legacy" })).toEqual([]);
        for (const ms of chunkSizeChoices({ kind: "ddsp", presetBlockMs: 100 })) {
            expect(runningChunkSizeMs({ kind: "ddsp", presetBlockMs: 100 }, ms)).toBe(ms);
        }
    });
});

describe("what Auto resolves to on this PC", () => {
    it("is the Streamer's block for the PC's cores on the processor: 120 / 160 / 200 ms", () => {
        for (const [cores, ms] of [[16, 120], [8, 120], [7, 120], [6, 160], [5, 200], [4, 200], [2, 200], [0, 200]] as const) {
            expect(templateAutoBlockMs({ ...input(), physicalCores: cores, streamerRule: true }), `${cores} cores`).toBe(ms);
            expect(chunkSizeView(input({ physicalCores: cores })).autoMs, `${cores} cores`).toBe(streamerAutoBlockMs(cores));
        }
        // The probe has not answered yet: no number rather than a wrong one.
        expect(chunkSizeView(input({ physicalCores: null })).autoMs).toBeNull();
        expect(chunkSizeView(input({ physicalCores: undefined })).autoMs).toBeNull();
        expect(chunkSizeView(input({ physicalCores: Number.NaN })).autoMs).toBeNull();
    });

    it("follows the Audio device buffer on a graphics card and for every other preset (20 ms at the default)", () => {
        for (const ep of ["cuda", "dml", "tensorrt_rtx", "dml_hp"] as const) {
            expect(chunkSizeView(input({ ep, physicalCores: 4 })).autoMs, ep).toBe(20);
        }
        for (const profile of ["low_latency", "quality", "studio_trt"] as const) {
            expect(chunkSizeView(input({ profile, ep: "cpu", physicalCores: 4 })).autoMs, profile).toBe(20);
        }
        // 1024 frames at 48 kHz are 21.3 ms: planned to 40 ms.
        expect(chunkSizeView(input({ profile: "quality", ep: "cuda", chunkFrames: 1024 })).autoMs).toBe(40);
        expect(chunkSizeView(input({ profile: "quality", ep: "cuda", chunkFrames: 160 })).autoMs).toBe(20);
        expect(chunkSizeView(input({ profile: "quality", ep: "cuda", chunkFrames: 2880 })).autoMs).toBe(60);
        // The derived block is capped at 200 ms.
        expect(chunkSizeView(input({ profile: "quality", ep: "cuda", chunkFrames: 48_000 })).autoMs).toBe(DERIVED_BLOCK_MAX_MS);
    });

    it("keeps a deliberately big device buffer on the Streamer (60 ms and more), raises a smaller one", () => {
        expect(STREAMER_KEEP_DERIVED_FROM_MS).toBe(60);
        // 2880 frames = 60 ms, 3840 = 80, 5760 = 120, 9600 = 200: kept whatever the PC.
        for (const [frames, ms] of [[2880, 60], [3840, 80], [5760, 120], [9600, 200]] as const) {
            expect(chunkSizeView(input({ chunkFrames: frames, physicalCores: 4 })).autoMs, `${frames} frames`).toBe(ms);
        }
        // Under 60 ms: the PC's block. 2816 frames are 58.7 ms.
        for (const frames of [160, 512, 1024, 2048, 2816]) {
            expect(chunkSizeView(input({ chunkFrames: frames, physicalCores: 4 })).autoMs, `${frames} frames`).toBe(200);
        }
    });

    it("reads the running engine's device buffer, and an unset provider follows the preset", () => {
        // 10 ms device buffers (device_block = model_divisor, or another capture rate) derive the same 20 ms block.
        expect(chunkSizeView(input({ profile: "quality", ep: "cuda", runningDeviceMs: 10 })).autoMs).toBe(20);
        expect(chunkSizeView(input({ profile: "quality", ep: "cuda", runningDeviceMs: 85.3 })).autoMs).toBe(100);
        // A running 120 ms buffer is kept on the Streamer; the stopped setting is ignored meanwhile.
        expect(chunkSizeView(input({ chunkFrames: 512, runningDeviceMs: 120, physicalCores: 4 })).autoMs).toBe(120);
        // No provider stored: the Streamer runs on the processor, the others on a graphics card.
        expect(chunkSizeView(input({ ep: "", physicalCores: 4 })).autoMs).toBe(200);
        expect(chunkSizeView(input({ ep: "", profile: "quality", physicalCores: 4 })).autoMs).toBe(20);
    });

    it("is the voice's preset for DDSP-SVC, the device-derived block for so-vits-svc, and none where the setting does not apply", () => {
        expect(chunkSizeView(input({ scope: { kind: "ddsp", presetBlockMs: 100 } })).autoMs).toBe(100);
        expect(chunkSizeView(input({ scope: { kind: "ddsp", presetBlockMs: 500 } })).autoMs).toBe(500);
        // so-vits-svc has no per-PC rule: the Streamer on 4 cores gets the device block, not 200 ms.
        expect(chunkSizeView(input({ scope: SOVITS, physicalCores: 4 })).autoMs).toBe(20);
        expect(chunkSizeView(input({ scope: { kind: "fixed", reason: "legacy" } })).autoMs).toBeNull();
    });

    it("names the DDSP-SVC preset's block from the voice's options", () => {
        expect(ddspPresetBlockMs(undefined)).toBe(100);
        expect(ddspPresetBlockMs({})).toBe(100);
        expect(ddspPresetBlockMs({ preset: "sigil_singing_fast" })).toBe(100);
        expect(ddspPresetBlockMs({ preset: "upstream_gui" })).toBe(500);
        expect(ddspPresetBlockMs({ preset: "nonsense" })).toBe(100);
        expect(chunkSizeScope("family", "ddsp_svc", { preset: "upstream_gui" })).toEqual({ kind: "ddsp", presetBlockMs: 500 });
    });
});

describe("which voices the setting reaches", () => {
    it("reaches imported voices and the default setup, with the Streamer's per-PC rule", () => {
        expect(chunkSizeScope("template", null)).toEqual(ENGINE);
        expect(chunkSizeScope("none", null)).toEqual(ENGINE);
        const view = chunkSizeView(input());
        expect(view).toMatchObject({ applies: true, reason: null });
    });

    it("is disabled for a bring-your-own ONNX voice, with its reason", () => {
        const scope = chunkSizeScope("legacy", null);
        expect(scope).toEqual({ kind: "fixed", reason: "legacy" });
        expect(chunkSizeView(input({ scope, storedMs: 160 }))).toEqual({
            applies: false,
            reason: "legacy",
            choices: [],
            autoMs: null,
            autoExact: false,
            runsAsMs: null,
            shortChunk: null,
        });
    });

    it("reaches so-vits-svc and DDSP-SVC voices and is disabled for the families that set their own chunk", () => {
        expect(chunkSizeScope("family", "sovits")).toEqual(SOVITS);
        expect(chunkSizeScope("family", "ddsp_svc")).toMatchObject({ kind: "ddsp" });
        expect(chunkSizeScope("family", "sigil_zs")).toEqual({ kind: "fixed", reason: "sigil_zs" });
        expect(chunkSizeScope("family", "meanvc")).toEqual({ kind: "fixed", reason: "meanvc" });
        expect(chunkSizeScope("family", "sidecar_v2")).toEqual({ kind: "fixed", reason: "sidecar_v2" });
        // A family this build does not know: the generic reason, never a control that does nothing.
        expect(chunkSizeScope("family", null)).toEqual({ kind: "fixed", reason: "family" });
        for (const family of VC_FAMILIES) {
            const scope = chunkSizeScope("family", family);
            expect(scope.kind === "fixed", family).toBe(FAMILY_CHUNK_SIZE[family] !== "engine" && FAMILY_CHUNK_SIZE[family] !== "ddsp");
        }
    });
});

describe("a stored size", () => {
    it("is always one of the choices, with what the engine runs it as when that differs", () => {
        const v = chunkSizeView(input({ storedMs: 160 }));
        expect(v.choices).toContain(160);
        expect(v.runsAsMs).toBeNull();
        // A size from the old list that the engine plans to another: kept visible, marked.
        for (const [stored, runs] of [[30, 40], [50, 60], [70, 80], [90, 100], [110, 120], [190, 200], [5, 20]] as const) {
            const view = chunkSizeView(input({ storedMs: stored }));
            expect(view.choices, `${stored}`).toContain(stored);
            expect([...view.choices].sort((a, b) => a - b), `${stored}`).toEqual(view.choices);
            expect(view.runsAsMs, `${stored}`).toBe(runs);
        }
        // Auto (0) and a nonsense value are Auto.
        for (const stored of [0, -5, Number.NaN]) {
            const view = chunkSizeView(input({ storedMs: stored }));
            expect(view.runsAsMs).toBeNull();
            expect(view.choices).toEqual(chunkSizeChoices(ENGINE));
        }
    });

    it("runs DDSP-SVC's size rounded up to 10 ms and not under 50 ms", () => {
        const ddsp: ChunkSizeScope = { kind: "ddsp", presetBlockMs: 100 };
        expect(runningChunkSizeMs(ddsp, 65)).toBe(70);
        expect(runningChunkSizeMs(ddsp, 40)).toBe(DDSP_CHUNK_SIZE_MIN_MS);
        expect(runningChunkSizeMs(ddsp, 20)).toBe(DDSP_CHUNK_SIZE_MIN_MS);
        expect(runningChunkSizeMs(ddsp, 120)).toBe(120);
        expect(runningChunkSizeMs(ddsp, 0)).toBeNull();
        expect(runningChunkSizeMs({ kind: "fixed", reason: "legacy" }, 120)).toBeNull();
        expect(chunkSizeView(input({ scope: ddsp, storedMs: 40 })).runsAsMs).toBe(50);
        expect(chunkSizeView(input({ scope: ddsp, storedMs: 40 })).choices).toContain(40);
    });
});

describe("the short-chunk warning", () => {
    it("names a size shorter than the Streamer keeps up with on this PC, and only that", () => {
        const note = (over: Partial<ChunkSizeInput>) => chunkSizeView(input(over)).shortChunk;
        expect(note({ physicalCores: 4, storedMs: 120 })).toEqual({ cores: 4, blockMs: 200, currentMs: 120, fewCores: true });
        expect(note({ physicalCores: 6, storedMs: 120 })).toEqual({ cores: 6, blockMs: 160, currentMs: 120, fewCores: true });
        expect(note({ physicalCores: 5, storedMs: 160 })).toEqual({ cores: 5, blockMs: 200, currentMs: 160, fewCores: true });
        // Auto, the PC's own block, or longer: no warning.
        expect(note({ physicalCores: 4, storedMs: 0 })).toBeNull();
        expect(note({ physicalCores: 4, storedMs: 200 })).toBeNull();
        expect(note({ physicalCores: 6, storedMs: 160 })).toBeNull();
        // Not the Streamer, not the processor, and not a voice the Streamer rule does not describe.
        expect(note({ physicalCores: 4, storedMs: 40, profile: "quality" })).toBeNull();
        expect(note({ physicalCores: 4, storedMs: 40, ep: "cuda" })).toBeNull();
        expect(note({ physicalCores: 4, storedMs: 40, scope: SOVITS })).toBeNull();
        expect(note({ physicalCores: 4, storedMs: 40, scope: { kind: "fixed", reason: "legacy" } })).toBeNull();
    });

    it("also names a short size on 7 or more cores (20 to 100 ms: the 6-thread Streamer falls behind there) and while the cores are unknown", () => {
        const note = (over: Partial<ChunkSizeInput>) => chunkSizeView(input(over)).shortChunk;
        // The sizes the new list offers at its short end, on an 8-core PC: warned until 120 ms.
        for (const storedMs of [20, 40, 60, 80, 100]) {
            expect(note({ physicalCores: 8, storedMs }), `${storedMs} ms`).toEqual({ cores: 8, blockMs: 120, currentMs: storedMs, fewCores: false });
        }
        for (const storedMs of [120, 140, 200]) expect(note({ physicalCores: 8, storedMs })).toBeNull();
        // The same size left over from another preset: the warning follows the Streamer on the processor when it is chosen.
        expect(note({ physicalCores: 8, storedMs: 40, profile: "quality", ep: "cuda" })).toBeNull();
        expect(note({ physicalCores: 8, storedMs: 40, profile: "streamer", ep: "cpu" })).not.toBeNull();
        // The probe has not answered: a conservative note for sizes no PC keeps up with, none for the others.
        for (const physicalCores of [null, undefined, 0]) {
            expect(note({ physicalCores, storedMs: 40 })).toEqual({ cores: null, blockMs: 120, currentMs: 40, fewCores: false });
            expect(note({ physicalCores, storedMs: 120 })).toBeNull();
            expect(note({ physicalCores, storedMs: 0 })).toBeNull();
        }
    });
});

describe("what Auto reads from the running engine", () => {
    it("is the chunk the engine reports on Auto, exact, whatever the estimate says", () => {
        // The estimate says 200 ms on 4 cores; the engine runs 20 ms (SIGIL_VC_STREAMER_BLOCK=0, say).
        const stopped = chunkSizeView(input({ physicalCores: 4 }));
        expect([stopped.autoMs, stopped.autoExact]).toEqual([200, false]);
        const running = chunkSizeView(input({ physicalCores: 4, runningBlockMs: 20 }));
        expect([running.autoMs, running.autoExact]).toEqual([20, true]);
        // Even while the cores are unknown: the engine knows.
        expect(chunkSizeView(input({ physicalCores: null, runningBlockMs: 120 }))).toMatchObject({ autoMs: 120, autoExact: true });
        // DDSP-SVC reports its preset's block.
        expect(chunkSizeView(input({ scope: { kind: "ddsp", presetBlockMs: 100 }, runningBlockMs: 100 }))).toMatchObject({ autoMs: 100, autoExact: true });
    });

    it("stays an estimate when a size is set (Auto is not what runs), when nothing is reported and where the setting does not apply", () => {
        expect(chunkSizeView(input({ physicalCores: 4, storedMs: 120, runningBlockMs: 120 }))).toMatchObject({ autoMs: 200, autoExact: false });
        for (const runningBlockMs of [null, undefined, 0, -1, Number.NaN]) {
            expect(chunkSizeView(input({ physicalCores: 4, runningBlockMs }))).toMatchObject({ autoMs: 200, autoExact: false });
        }
        expect(chunkSizeView(input({ scope: { kind: "fixed", reason: "sigil_zs" }, runningBlockMs: 100 }))).toMatchObject({ autoMs: null, autoExact: false });
    });
});

describe("presets and profiles never change the size", () => {
    it("keeps blockMs out of every preset's patch", () => {
        for (const profile of ["streamer", "low_latency", "quality", "studio_trt"] as const) {
            const patch = rvcProfilePatch(profile);
            expect("blockMs" in patch, profile).toBe(false);
        }
        // So a size set (or Auto) survives applying one.
        const state = VoiceChangerStateSchema.parse({ blockMs: 160 });
        const next = { ...state, ...rvcProfilePatch("quality") };
        expect(next.blockMs).toBe(160);
        expect(VoiceChangerStateSchema.parse({}).blockMs).toBe(0);
    });
});

// ---- Rust sources ------------------------------------------------------------------------------------

const rust = import.meta.glob(
    [
        "../../../../../src-tauri/src/services/neural_audio/vc/engine/presets.rs",
        "../../../../../src-tauri/src/services/neural_audio/vc/engine/plan.rs",
        "../../../../../src-tauri/src/services/voice_changer/factory.rs",
        "../../../../../src-tauri/src/services/neural_audio/vc/families/**/*.rs",
    ],
    { eager: true, query: "?raw", import: "default" },
) as Record<string, string>;

const rustFile = (suffix: string): string => {
    const hit = Object.entries(rust).find(([path]) => path.endsWith(suffix));
    if (!hit) throw new Error(`no ${suffix}`);
    return hit[1];
};

/** A family's production sources (`<x>.rs` plus `<x>/**`, test files left out). */
function familySource(family: VcFamilyId): string {
    return Object.entries(rust)
        .filter(([path]) => path.includes("/families/") && (path.endsWith(`/families/${family}.rs`) || path.includes(`/families/${family}/`)))
        .filter(([path]) => !/\/tests?(\.rs|\/)/.test(path))
        .map(([, src]) => src)
        .join("\n");
}

describe("agrees with the Rust sources", () => {
    it("the block range, the 20 ms plan and the derived block's cap", () => {
        const plan = rustFile("/engine/plan.rs");
        const presets = rustFile("/engine/presets.rs");
        const maxFrames = Number(/pub const MAX_BLOCK_FRAMES: usize = (\d+);/.exec(plan)?.[1]);
        expect(maxFrames * 10).toBe(CHUNK_SIZE_MAX_MS);
        expect(DERIVED_BLOCK_MAX_MS).toBe(CHUNK_SIZE_MAX_MS);
        expect(presets).toContain("pub const MAX_DERIVED_BLOCK_MS: f64 = (MAX_BLOCK_FRAMES * 10) as f64;");
        // The plan rounds up to an even number of 10 ms frames: whole 20 ms.
        expect(plan).toMatch(/let frames = frames\.div_ceil\(2\) \* 2;/);
        expect(CHUNK_SIZE_STEP_MS).toBe(20);
        expect(Number(/pub const CPU_STREAMER_KEEP_FROM_MS: f64 = ([0-9.]+);/.exec(presets)?.[1])).toBe(STREAMER_KEEP_DERIVED_FROM_MS);
    });

    it("DDSP-SVC's range, rounding and preset blocks", () => {
        const ddsp = familySource("ddsp_svc");
        expect(Number(/const MIN_BLOCK_MS: f64 = ([0-9.]+);/.exec(ddsp)?.[1])).toBe(DDSP_CHUNK_SIZE_MIN_MS);
        expect(Number(/const MAX_BLOCK_MS: f64 = ([0-9.]+);/.exec(ddsp)?.[1])).toBe(DDSP_CHUNK_SIZE_MAX_MS);
        expect(ddsp).toContain("let b = (block_ms / 10.0).ceil() * 10.0;");
        expect(DDSP_CHUNK_SIZE_STEP_MS).toBe(10);
        for (const preset of DDSP_SVC_PRESETS) {
            const row = new RegExp(`"${preset}" => PresetShape \\{ block_ms: ([0-9.]+),`).exec(ddsp)?.[1];
            // The default preset is the `_ =>` arm; the others are named.
            const want = row ?? /_ => PresetShape \{ block_ms: ([0-9.]+),/.exec(ddsp)?.[1];
            expect(Number(want), preset).toBe(DDSP_PRESET_BLOCK_MS[preset]);
        }
    });

    it("the engine factory gives a sidecar voice no Chunk size, the one family whose control is disabled but which still reads block_ms", () => {
        const factory = rustFile("/voice_changer/factory.rs");
        expect(factory).toMatch(/fn family_block_ms\(kind: FamilyKind, block_ms: f64\) -> f64 \{\s*if matches!\(kind, FamilyKind::SidecarV2\) \{\s*0\.0/);
        expect(factory).toContain("block_ms: family_block_ms(kind, request.block_ms),");
        // The families that read the size and have an enabled control get it as stored.
        expect(FAMILY_CHUNK_SIZE.sidecar_v2).toBe("sidecar_v2");
        expect(chunkSizeScope("family", "sidecar_v2")).toEqual({ kind: "fixed", reason: "sidecar_v2" });
    });

    it("which families read the Chunk size as their block", () => {
        const reads = (family: VcFamilyId) => /\b(?:req|request)\.block_ms\b/.test(familySource(family));
        // so-vits-svc and DDSP-SVC run the setting as their block; the sidecar only proposes it to its helper, which answers
        // with its own frame size; MeanVC2 and Sigil-ZS never read it.
        expect(VC_FAMILIES.filter(reads).sort()).toEqual(["ddsp_svc", "sidecar_v2", "sovits"]);
        for (const family of VC_FAMILIES) {
            const how = FAMILY_CHUNK_SIZE[family];
            if (how === "engine" || how === "ddsp") expect(reads(family), family).toBe(true);
            if (how === "meanvc" || how === "sigil_zs") expect(reads(family), family).toBe(false);
        }
        const sidecar = familySource("sidecar_v2");
        expect(sidecar).toContain("fn proposed_frame_ms");
        expect(sidecar).toContain("session.frame_ms()");
        expect(FAMILY_CHUNK_SIZE.sidecar_v2).toBe("sidecar_v2");
        // so-vits-svc plans the block like the template engine (`engine_preset` + `block_frames`), without the Streamer rule.
        const sovits = familySource("sovits");
        expect(sovits).toContain("engine_preset(req.preset, req.block_ms,");
        expect(sovits).toContain("block_frames(preset.block_ms)");
        expect(sovits).not.toContain("cpu_streamer");
    });
});
