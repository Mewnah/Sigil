// Voice-changer inspector (P5-G): hooks shared by the sections that depend on the selected voice and its family.
import type { VcCapabilities, VcFamilyCapability, VcSlotInfo } from "@/core/services/voice_changer/ipc";
import { familyVoiceControls, voiceFamilyCapability, voiceKind, type FamilyVoiceControls, type VoiceKind } from "@/core/services/voice_changer/families";
import { findSlot } from "@/core/services/voice_changer/slots";
import { ensureVoiceLibraryHost } from "@/core/services/voice_library";
import { isTauri } from "@tauri-apps/api/core";
import { useEffect } from "react";
import { useSnapshot } from "valtio";

/** The selected voice, when it is a listed slot. */
export function useSelectedVoice(): VcSlotInfo | undefined {
    const lib = useSnapshot(window.ApiServer.voiceChanger.library.state);
    const dir = useSnapshot(window.ApiServer.voiceChanger.state).voiceChanger.rvcSlotDir;
    return lib.slots ? (findSlot(lib.slots as VcSlotInfo[], dir) as VcSlotInfo | undefined) : undefined;
}

/** The capabilities snapshot, typed as the service holds it. */
export function useVcCapabilities(): VcCapabilities {
    return useSnapshot(window.ApiServer.voiceChanger.state).capabilities as VcCapabilities;
}

/** What runs the voice, its family's capability row (family voices only) and its controls. */
export function useVoiceFamily(voice: VcSlotInfo | undefined): {
    kind: VoiceKind;
    family: VcFamilyCapability | null;
    controls: FamilyVoiceControls | null;
} {
    const caps = useVcCapabilities();
    return {
        kind: voiceKind(voice),
        family: voiceFamilyCapability(caps, voice),
        controls: familyVoiceControls(caps, voice),
    };
}

/** Load the shared voice library once (desktop only) while `active`: pickers and reference names read it. */
export function useVoiceLibraryHost(active = true): void {
    useEffect(() => {
        if (!active || !isTauri()) return;
        ensureVoiceLibraryHost();
    }, [active]);
}
