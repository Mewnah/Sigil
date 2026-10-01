/**
 * Slot-store rows, import planning, merge partners, voice stepping and the content-encoder / pitch /
 * denoiser model ids against the Rust side (`vc/assets.rs` and the model-catalog fragments).
 */
import { describe, expect, it } from "vitest";

import assetsRs from "../../../../../src-tauri/src/services/neural_audio/vc/assets.rs?raw";
import vcBases from "../../../../../src-tauri/model-catalog/vc-bases.json";
import vcDenoise from "../../../../../src-tauri/model-catalog/vc-denoise.json";
import vcEmbedders from "../../../../../src-tauri/model-catalog/vc-embedders.json";
import { BASELINE_CAPABILITIES, type VcCapabilities, type VcSlotInfo } from "../ipc";
import { denoiserModelIds, DENOISER_MODEL_IDS, DEFAULT_VOICE_ID, PITCH_MODEL_IDS, starterModelIds, voiceModelIds } from "../requirements";
import {
    adjacentSlot,
    EMBEDDER_CATALOG_IDS,
    embedderDisplayName,
    embedderOf,
    fileNameOf,
    mergePartners,
    normalizeSlotInfo,
    normalizeSlotInfos,
    planImport,
    sampleRateLabel,
    slotInfoFromLegacy,
    slotProblem,
    vocoderDisplayName,
} from "../slots";

const slot = (over: Partial<VcSlotInfo> = {}): VcSlotInfo => ({
    id: "alto",
    name: "Alto",
    path: "C:\\models\\alto",
    format: "sigil-rvc-1",
    template: "v2-40k-f0-hifigan",
    version: "v2",
    sample_rate: 40000,
    f0: true,
    embedder: "contentvec",
    vocoder: "hifigan",
    has_index: true,
    speaker_count: 1,
    fp16_ok: true,
    source_sha256: null,
    created_unix_ms: null,
    problem: null,
    ...over,
});

const caps = (over: Partial<VcCapabilities> = {}): VcCapabilities => ({ ...BASELINE_CAPABILITIES, ...over });

type CatalogFragment = { entries?: { id: string }[] } | { id: string }[];
const catalogIds = (f: CatalogFragment): string[] => (Array.isArray(f) ? f : f.entries ?? []).map((e) => e.id);
const assetIds = [...assetsRs.matchAll(/pub const [A-Z0-9_]+: &str = "([^"]+)";/g)].map((m) => m[1]);

describe("slot-store rows", () => {
    it("keep only rows with an id and a path; optional fields become null", () => {
        expect(normalizeSlotInfo({ id: "a" })).toBeNull();
        expect(normalizeSlotInfo({ id: " ", path: "C:/x" })).toBeNull();
        const s = normalizeSlotInfo({ id: "a", path: "C:/m/a", sample_rate: "40k", f0: 1, has_index: true, speaker_count: 2 })!;
        expect(s).toMatchObject({ id: "a", name: "a", format: "rvc", sample_rate: null, f0: null, has_index: true, speaker_count: 2 });
        expect(normalizeSlotInfos([{ id: "a", path: "p" }, { id: "a", path: "q" }, null, { id: "b", path: "r", name: "B" }]).map((x) => x.name)).toEqual(["a", "B"]);
        expect(normalizeSlotInfos("nope")).toEqual([]);
        expect(slotInfoFromLegacy({ id: "x", path: "C:/m/x", has_manifest: false })).toMatchObject({ id: "x", format: "rvc", problem: null });
    });

    it("a slot the build cannot run says why", () => {
        expect(slotProblem(slot(), ["rvc", "sigil-rvc-1"])).toBeNull();
        expect(slotProblem(slot(), ["rvc"])).toContain("sigil-rvc-1");
        expect(slotProblem(slot({ problem: "template missing" }), ["rvc", "sigil-rvc-1"])).toBe("template missing");
    });

    it("merge partners share the template and content encoder", () => {
        const a = slot();
        const all = [
            a,
            slot({ id: "b", name: "B" }),
            slot({ id: "c", template: "v2-48k-f0-hifigan" }),
            slot({ id: "d", embedder: "spin-v2" }),
            slot({ id: "e", format: "rvc" }),
            slot({ id: "f", problem: "broken" }),
        ];
        expect(mergePartners(all, a).map((s) => s.id)).toEqual(["b"]);
        expect(mergePartners(all, all[4])).toEqual([]);
    });

    it("steps through runnable voices, wrapping", () => {
        const all = [slot({ id: "a", path: "C:/m/a" }), slot({ id: "b", path: "C:/m/b", problem: "x" }), slot({ id: "c", path: "C:/m/c" })];
        const formats = ["rvc", "sigil-rvc-1"];
        expect(adjacentSlot(all, "C:\\M\\A\\", 1, formats)?.id).toBe("c");
        expect(adjacentSlot(all, "C:/m/c", 1, formats)?.id).toBe("a");
        expect(adjacentSlot(all, "C:/m/a", -1, formats)?.id).toBe("c");
        expect(adjacentSlot(all, "", 1, formats)?.id).toBe("a");
        expect(adjacentSlot(all, "", -1, formats)?.id).toBe("c");
        expect(adjacentSlot([], "", 1, formats)).toBeUndefined();
    });
});

describe("import planning", () => {
    it("imports every voice once and pairs one .index with one .pth", () => {
        const plan = planImport(["C:\\v\\Kira.pth", "C:/v/added_IVF.index", " C:/v/kira.PTH ", "C:/v/notes.txt", "C:/v/pack.zip"]);
        expect(plan.items).toEqual([
            { source: "C:\\v\\Kira.pth", index: "C:/v/added_IVF.index" },
            { source: "C:/v/pack.zip", index: null },
        ]);
        expect(plan.ignored).toEqual(["C:/v/notes.txt"]);
        expect(plan.unpairedIndexes).toEqual([]);
    });

    it("leaves the index choice to Rust when it is ambiguous", () => {
        const plan = planImport(["a.pth", "b.pth", "x.index"]);
        expect(plan.items.every((i) => i.index === null)).toBe(true);
        expect(plan.unpairedIndexes).toEqual(["x.index"]);
        expect(fileNameOf("C:\\voices\\Kira v2.pth")).toBe("Kira v2.pth");
    });
});

describe("content encoder, vocoder and model ids (Rust vc::assets and the catalog)", () => {
    it("names a voice's content encoder like Rust EmbedderKind::from_manifest", () => {
        expect(embedderOf(null, "v2")).toBe("contentvec");
        expect(embedderOf("contentvec", "v1")).toBe("contentvec-256");
        expect(embedderOf("hubert_base", "v2")).toBe("contentvec");
        expect(embedderOf("Vec-256-Layer-9", "v2")).toBe("contentvec-256");
        expect(embedderOf("spin-v2", "v2")).toBe("spin-v2");
        expect(embedderOf("japanese-hubert-base", "v2")).toBe("hubert-ja");
        expect(embedderOf("korean_hubert_base", "v2")).toBe("hubert-ko");
        expect(embedderOf("chinese-hubert-base", "v2")).toBe("hubert-zh");
        expect(embedderOf("spin", "v2")).toBe("spin");
        expect(embedderOf("SPIN_v1", "v2")).toBe("spin");
        expect(embedderOf("wavlm-large", "v2")).toBeNull();
        expect(embedderDisplayName("whisper", "v2")).toBe("whisper");
        expect(vocoderDisplayName("RefineGAN")).toBe("RefineGAN");
        expect(vocoderDisplayName(null)).toBe("HiFi-GAN (NSF)");
        expect(sampleRateLabel(40000)).toBe("40 kHz");
        expect(sampleRateLabel(22050)).toBe("22.1 kHz");
        expect(sampleRateLabel(null)).toBeNull();
    });

    it("every id is a Rust asset id and a catalog entry", () => {
        const catalog = [...catalogIds(vcBases as CatalogFragment), ...catalogIds(vcEmbedders as CatalogFragment), ...catalogIds(vcDenoise as CatalogFragment)];
        const ids = [...Object.values(EMBEDDER_CATALOG_IDS), ...Object.values(PITCH_MODEL_IDS), ...Object.values(DENOISER_MODEL_IDS), DEFAULT_VOICE_ID];
        for (const id of ids) {
            expect(assetIds, `vc/assets.rs has ${id}`).toContain(id);
            expect(catalog, `catalog has ${id}`).toContain(id as string);
        }
    });

    it("lists the models a voice and its F0 method use", () => {
        const base = {
            mode: "rvc" as const,
            pitchExtractor: "rmvpe" as const,
            rvcExecutionProvider: "cuda" as const,
            fcpePath: "",
            rmvpePath: "",
            capabilities: caps(),
        };
        expect(voiceModelIds({ ...base, voice: slot() })).toEqual(["vc/embedder-contentvec", "vc/pitch-rmvpe"]);
        // A GPU-only method on the CPU runs fcpe; an explicit path replaces the store entry.
        expect(voiceModelIds({ ...base, rvcExecutionProvider: "cpu", voice: slot({ embedder: "spin-v2" }) })).toEqual([
            "vc/embedder-spin-v2",
            "vc/pitch-fcpe",
        ]);
        expect(voiceModelIds({ ...base, rmvpePath: "C:/base/rmvpe.onnx", voice: slot() })).toEqual(["vc/embedder-contentvec"]);
        // Bring-your-own ONNX voices bring their own content encoder; nof0 voices need no F0 model.
        expect(voiceModelIds({ ...base, voice: slot({ format: "rvc" }) })).toEqual(["vc/pitch-rmvpe"]);
        expect(voiceModelIds({ ...base, voice: slot({ f0: false }) })).toEqual(["vc/embedder-contentvec"]);
        // Fusion runs SwiftF0 and RMVPE once the build runs it; until then its stand-in (rmvpe).
        expect(voiceModelIds({ ...base, pitchExtractor: "fusion", voice: null })).toEqual(["vc/pitch-rmvpe"]);
        const withFusion = caps({ pitch_extractors: ["fcpe", "rmvpe", "dio", "swiftf0", "fusion"] });
        expect(voiceModelIds({ ...base, pitchExtractor: "fusion", voice: null, capabilities: withFusion })).toEqual([
            "vc/pitch-swiftf0",
            "vc/pitch-rmvpe",
        ]);
        // On the CPU rung fusion pairs SwiftF0 with FCPE (RMVPE is GPU-only); an explicit FCPE path
        // replaces its entry.
        const onCpu = { ...base, pitchExtractor: "fusion" as const, rvcExecutionProvider: "cpu" as const, voice: null, capabilities: withFusion };
        expect(voiceModelIds(onCpu)).toEqual(["vc/pitch-swiftf0", "vc/pitch-fcpe"]);
        expect(voiceModelIds({ ...onCpu, fcpePath: "C:/base/fcpe.onnx" })).toEqual(["vc/pitch-swiftf0"]);
        expect(voiceModelIds({ ...onCpu, rvcExecutionProvider: "dml_hp" })).toEqual(["vc/pitch-swiftf0", "vc/pitch-rmvpe"]);
        expect(voiceModelIds({ ...base, pitchExtractor: "dio", voice: null })).toEqual([]);
        expect(voiceModelIds({ ...base, mode: "classic", voice: slot() })).toEqual([]);
    });

    it("lists the denoiser model, following Rust's stand-in", () => {
        expect(denoiserModelIds("off", caps())).toEqual([]);
        expect(denoiserModelIds("gtcrn", caps())).toEqual([]);
        const c = caps({ denoisers: ["off", "gtcrn", "dpdfnet"] });
        expect(denoiserModelIds("gtcrn", c)).toEqual(["vc/denoise-gtcrn"]);
        expect(denoiserModelIds("dfn3", c)).toEqual(["vc/denoise-dpdfnet"]);
    });

    it("the starter set follows the current F0 method", () => {
        const input = { pitchExtractor: "fcpe" as const, rvcExecutionProvider: "cpu" as const, fcpePath: "", rmvpePath: "", capabilities: caps() };
        expect(starterModelIds(input)).toEqual(["vc/embedder-contentvec", "vc/pitch-fcpe", "vc/voice-default"]);
        expect(starterModelIds({ ...input, pitchExtractor: "rmvpe", rvcExecutionProvider: "cuda" })).toEqual([
            "vc/embedder-contentvec",
            "vc/pitch-rmvpe",
            "vc/voice-default",
        ]);
    });
});
