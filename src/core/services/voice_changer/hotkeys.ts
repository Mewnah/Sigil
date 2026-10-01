/**
 * Voice-changer hotkeys (P3-G): global shortcuts that start / stop the voice changer, step through
 * the voices, and switch straight to up to four chosen voices. A voice switch while RVC runs is hot
 * (crossfaded over one block), so a hotkey can change the voice mid-sentence.
 *
 * The accelerators are app shortcuts like the others (`BackendState.shortcuts`, recorded with the
 * shared recorder, registered and conflict-checked by the keyboard service); which voice each
 * `vcVoiceN` hotkey selects is `BackendState.vcHotkeyVoices[N - 1]` (a slot id). Both keys are added
 * to the persisted schema by the P3 INTEGRATE step (hotspot `src/core/schema.ts`), and the keyboard
 * service dispatches `isVcShortcut` keys to `VoiceChangerService.runShortcut`; until then
 * `vcHotkeysSupported` is false and the UI hides the hotkeys.
 *
 * Pure module: no Tauri, DOM or `window`.
 */

/** Voice hotkeys that pick a chosen voice. */
export const VC_VOICE_HOTKEY_COUNT = 4;

export const VC_SHORTCUT_KEYS = [
    "vcToggle",
    "vcNextVoice",
    "vcPrevVoice",
    "vcVoice1",
    "vcVoice2",
    "vcVoice3",
    "vcVoice4",
] as const;
export type VcShortcutKey = (typeof VC_SHORTCUT_KEYS)[number];

export type VcShortcutAction =
    | { kind: "toggle" }
    | { kind: "step"; step: 1 | -1 }
    /** `index` is 0-based (hotkey `vcVoice1` is 0). */
    | { kind: "voice"; index: number };

export function isVcShortcut(key: unknown): key is VcShortcutKey {
    return typeof key === "string" && (VC_SHORTCUT_KEYS as readonly string[]).includes(key);
}

/** What a voice-changer shortcut does. */
export function vcShortcutAction(key: VcShortcutKey): VcShortcutAction {
    switch (key) {
        case "vcToggle":
            return { kind: "toggle" };
        case "vcNextVoice":
            return { kind: "step", step: 1 };
        case "vcPrevVoice":
            return { kind: "step", step: -1 };
        default:
            return { kind: "voice", index: Number(key.slice("vcVoice".length)) - 1 };
    }
}

/** The shortcut key of voice hotkey `index` (0-based). */
export function voiceHotkeyKey(index: number): VcShortcutKey {
    return `vcVoice${index + 1}` as VcShortcutKey;
}

/** Whether the app's shortcut store has the voice-changer keys (the P3 schema change is in). */
export function vcHotkeysSupported(shortcuts: unknown): boolean {
    return !!shortcuts && typeof shortcuts === "object" && VC_SHORTCUT_KEYS.every((k) => k in (shortcuts as object));
}

/** The slot ids of the voice hotkeys, always `VC_VOICE_HOTKEY_COUNT` long (`""` = none). */
export function normalizeHotkeyVoices(raw: unknown): string[] {
    const list = Array.isArray(raw) ? raw : [];
    return Array.from({ length: VC_VOICE_HOTKEY_COUNT }, (_, i) => (typeof list[i] === "string" ? list[i].trim() : ""));
}

/** `voices` with hotkey `index` set to `slotId` (`""` clears it); a voice keeps at most one hotkey. */
export function assignHotkeyVoice(voices: readonly string[], index: number, slotId: string): string[] {
    const id = slotId.trim();
    const out = normalizeHotkeyVoices(voices).map((v, i) => (i !== index && id !== "" && v === id ? "" : v));
    if (index >= 0 && index < VC_VOICE_HOTKEY_COUNT) out[index] = id;
    return out;
}
