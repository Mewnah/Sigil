/**
 * RVC voice slots under `app_data/voice-changer/models/` (Rust `list_voice_changer_model_slots`).
 * Pure module: no Tauri, DOM or `window`.
 */

/** One slot folder, as Rust lists it. */
export interface VoiceChangerModelSlot {
    /** Folder name (shown as the voice name). */
    id: string;
    /** Absolute folder path (what `rvc_slot_dir` stores). */
    path: string;
    /** The folder has a `manifest.json` (otherwise Sigil infers it from the model). */
    has_manifest: boolean;
}

/** Keep only well-formed rows (a stale or older backend must not break the picker). */
export function normalizeSlots(raw: unknown): VoiceChangerModelSlot[] {
    if (!Array.isArray(raw)) return [];
    const out: VoiceChangerModelSlot[] = [];
    for (const r of raw) {
        if (!r || typeof r !== "object") continue;
        const s = r as Partial<VoiceChangerModelSlot>;
        if (typeof s.id !== "string" || typeof s.path !== "string" || !s.path.trim()) continue;
        out.push({ id: s.id, path: s.path, has_manifest: s.has_manifest === true });
    }
    return out;
}

/** Folder path key: separators unified, trailing separators dropped, case-insensitive (Windows paths). */
export function slotPathKey(path: string): string {
    return path.trim().replace(/\\/g, "/").replace(/\/+$/, "").toLowerCase();
}

/** Whether two folder strings name the same folder (separator, trailing-slash and case insensitive). */
export function sameSlotPath(a: string, b: string): boolean {
    const ka = slotPathKey(a);
    return ka !== "" && ka === slotPathKey(b);
}

/** The listed slot the selected folder is, if any. */
export function findSlot(slots: readonly VoiceChangerModelSlot[], slotDir: string): VoiceChangerModelSlot | undefined {
    return slots.find((s) => sameSlotPath(s.path, slotDir));
}
