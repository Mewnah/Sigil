/**
 * Model-store entries the voice changer's current setup uses (P3-G), for the inspector's
 * `<ModelRequirement>` card: nothing downloads on its own; the card offers the download (or the
 * import of an entry that is not published) with its size.
 *
 * The engine finds each model the same way (`vc::assets`, `bases.rs`, `f0_fusion::resolve_pitch_model`):
 * an explicit path from the settings, then the model-store entry, then the legacy
 * `app_data/voice-changer/base/` files. Only entries the setup can use from the store are listed:
 * an explicit path skips its entry, and a bring-your-own ONNX voice's content encoder comes from its
 * own folder or `base/`, never the store.
 *
 * Pure module: no Tauri, DOM or `window`.
 */
import { effectivePitchExtractor, rvcBaseEp, type Denoiser, type PitchExtractorId, type RvcExecutionProviderId, type VcMode } from "./contract";
import type { VcCapabilities, VcSlotInfo } from "./ipc";
import { voiceFamilyCapability, voiceKind } from "./families";
import { effectiveDenoiser } from "./options";
import { EMBEDDER_CATALOG_IDS, embedderOf, isTemplateSlot } from "./slots";

/** Model-store ids of the pitch models (Rust `f0_fusion::pitch_model_id`). */
export const PITCH_MODEL_IDS: Readonly<Partial<Record<PitchExtractorId, string>>> = {
    fcpe: "vc/pitch-fcpe",
    rmvpe: "vc/pitch-rmvpe",
    swiftf0: "vc/pitch-swiftf0",
    crepe: "vc/pitch-crepe-tiny",
    pesto: "vc/pitch-pesto",
};

/**
 * Methods an F0 method runs on the EP ladder rung `baseEp` (Rust `f0_fusion::create_pitch_extractor_with`):
 * fusion takes its voicing from SwiftF0 and its pitch from RMVPE on a GPU, from FCPE on the CPU
 * (RMVPE is GPU-only in Sigil).
 */
function pitchParts(method: PitchExtractorId, baseEp: string): readonly PitchExtractorId[] {
    if (method === "fusion") return baseEp === "cpu" ? ["swiftf0", "fcpe"] : ["swiftf0", "rmvpe"];
    return [method];
}

/** Model-store ids of the denoisers. */
export const DENOISER_MODEL_IDS: Readonly<Record<Exclude<Denoiser, "off">, string>> = {
    gtcrn: "vc/denoise-gtcrn",
    dfn3: "vc/denoise-dfn3",
    dpdfnet: "vc/denoise-dpdfnet",
};

/** The MIT default voice (`voice_changer_import_default_voice` installs and imports it). */
export const DEFAULT_VOICE_ID = "vc/voice-default";

export interface VoiceRequirementInput {
    mode: VcMode;
    pitchExtractor: PitchExtractorId;
    /** The RVC EP asked for (the profile default when unset). */
    rvcExecutionProvider: RvcExecutionProviderId;
    /** Explicit base model paths (a set path replaces the store entry). */
    fcpePath: string;
    rmvpePath: string;
    /** The selected voice, when it is a listed slot. */
    voice: VcSlotInfo | null;
    capabilities: VcCapabilities;
}

/**
 * Store entries the RVC voice and its F0 method use (deduplicated, stable order; none outside RVC mode).
 *
 * A P5 family voice (`families.ts`) needs what its family lists (`VcFamilyCapability.requires`), its
 * content encoder only when the family runs one (`needs_content_encoder`: the voice's `embedder`,
 * ContentVec when unnamed, as the start preflight resolves it) and the pitch models only when it
 * tracks pitch (`needs_pitch`). A family this build does not run lists nothing (the voice shows why).
 */
export function voiceModelIds(input: VoiceRequirementInput): string[] {
    const ids: string[] = [];
    const add = (id: string | undefined) => {
        if (id && !ids.includes(id)) ids.push(id);
    };
    if (input.mode !== "rvc") return ids;
    const voice = input.voice;
    const addPitch = () => {
        const run = effectivePitchExtractor(input.pitchExtractor, input.rvcExecutionProvider, input.capabilities.pitch_extractors);
        for (const method of pitchParts(run.extractor, rvcBaseEp(input.rvcExecutionProvider))) {
            if (method === "fcpe" && input.fcpePath.trim()) continue;
            if (method === "rmvpe" && input.rmvpePath.trim()) continue;
            add(PITCH_MODEL_IDS[method]);
        }
    };
    if (voiceKind(voice) === "family") {
        const fam = voiceFamilyCapability(input.capabilities, voice);
        if (!fam || !fam.available) return ids;
        const encoders: readonly string[] = Object.values(EMBEDDER_CATALOG_IDS);
        let encoder: string | null = null;
        if (fam.needs_content_encoder) {
            const kind = embedderOf(voice?.embedder, null);
            encoder = kind ? EMBEDDER_CATALOG_IDS[kind] : null;
            add(encoder ?? undefined);
        }
        // The family's static list names its usual encoder; a voice that names another one needs only its own.
        for (const id of fam.requires) if (!(fam.needs_content_encoder && encoders.includes(id) && id !== encoder)) add(id);
        if (fam.needs_pitch) addPitch();
        return ids;
    }
    if (voice && isTemplateSlot(voice) && !voice.problem) {
        const kind = embedderOf(voice.embedder, voice.version);
        if (kind) add(EMBEDDER_CATALOG_IDS[kind]);
    }
    // A voice trained without pitch (nof0) runs no F0 extractor.
    if (voice?.f0 !== false) addPitch();
    return ids;
}

/** The store entry of the denoiser that runs for `denoiser` (none when off or not available). */
export function denoiserModelIds(denoiser: Denoiser, capabilities: VcCapabilities): string[] {
    const den = effectiveDenoiser(denoiser, capabilities.denoisers);
    return den === "off" ? [] : [DENOISER_MODEL_IDS[den]];
}

/**
 * First-run set: the content encoder of the default voice (and of most v2 voices), the pitch model
 * the current F0 method and EP run, and the default voice itself.
 */
export function starterModelIds(input: Omit<VoiceRequirementInput, "voice" | "mode">): string[] {
    const ids: string[] = [EMBEDDER_CATALOG_IDS.contentvec];
    for (const id of voiceModelIds({ ...input, mode: "rvc", voice: null })) if (!ids.includes(id)) ids.push(id);
    ids.push(DEFAULT_VOICE_ID);
    return ids;
}
