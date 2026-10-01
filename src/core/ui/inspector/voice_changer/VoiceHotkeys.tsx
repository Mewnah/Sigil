// Voice-changer inspector (owned by P3-G): voice hotkeys. Global shortcuts that start / stop the
// voice changer, step through the voices, and switch straight to up to four chosen voices (a hot,
// crossfaded switch while RVC runs). The accelerators are app shortcuts (recorded and conflict-checked
// by the keyboard service); the section appears once the app settings carry the voice-changer keys
// (`hotkeys.ts` `vcHotkeysSupported`).
import type { BackendState } from "@/core/schema";
import {
    assignHotkeyVoice,
    normalizeHotkeyVoices,
    VC_VOICE_HOTKEY_COUNT,
    vcHotkeysSupported,
    voiceHotkeyKey,
    type VcShortcutKey,
} from "@/core/services/voice_changer/hotkeys";
import type { VcSlotInfo } from "@/core/services/voice_changer/ipc";
import { isTauri } from "@tauri-apps/api/core";
import { FC } from "react";
import { useTranslation } from "react-i18next";
import { useSnapshot } from "valtio";
import { InputShortcut } from "../components/input";
import { VcDetails, VcNote } from "./common";

type ShortcutKey = keyof BackendState["shortcuts"];

/**
 * A voice-changer shortcut key as the app's shortcut key type. The keys are part of the persisted
 * settings once `vcHotkeysSupported` is true (checked before this is used).
 */
function asShortcutKey(key: VcShortcutKey): ShortcutKey {
    return key as unknown as ShortcutKey;
}

export const VoiceHotkeys: FC = () => {
    const { t } = useTranslation();
    const app = useSnapshot(window.ApiServer.state) as unknown as Record<string, unknown>;
    const lib = useSnapshot(window.ApiServer.voiceChanger.library.state);
    if (!isTauri() || !vcHotkeysSupported(app.shortcuts)) return null;
    const voices = normalizeHotkeyVoices(app.vcHotkeyVoices);
    const slots = (lib.slots ?? []) as VcSlotInfo[];
    const setVoice = (index: number, slotId: string) => {
        const state = window.ApiServer.state as unknown as Record<string, unknown>;
        state.vcHotkeyVoices = assignHotkeyVoice(normalizeHotkeyVoices(state.vcHotkeyVoices), index, slotId);
    };
    return (
        <VcDetails titleKey="voice_changer.hotkeys_title" infoKey="voice_changer.hotkeys_info">
            <InputShortcut label="voice_changer.hotkey_toggle" shortcut={asShortcutKey("vcToggle")} />
            <InputShortcut label="voice_changer.hotkey_next_voice" shortcut={asShortcutKey("vcNextVoice")} />
            <InputShortcut label="voice_changer.hotkey_prev_voice" shortcut={asShortcutKey("vcPrevVoice")} />
            {Array.from({ length: VC_VOICE_HOTKEY_COUNT }, (_, i) => {
                const assigned = voices[i];
                const missing = assigned !== "" && !slots.some((s) => s.id === assigned);
                return (
                    <div key={i} className="flex flex-col gap-1 rounded-lg border border-base-content/10 p-2">
                        <select
                            className="select select-bordered select-sm w-full"
                            aria-label={t("voice_changer.hotkey_voice_n", { n: i + 1 })}
                            value={assigned}
                            onChange={(e) => setVoice(i, e.target.value)}
                        >
                            <option value="">{t("voice_changer.hotkey_voice_none", { n: i + 1 })}</option>
                            {missing ? <option value={assigned}>{t("voice_changer.hotkey_voice_missing", { id: assigned })}</option> : null}
                            {slots.map((s) => (
                                <option key={s.id} value={s.id}>
                                    {t("voice_changer.hotkey_voice_option", { n: i + 1, name: s.name })}
                                </option>
                            ))}
                        </select>
                        <InputShortcut label="voice_changer.hotkey_voice_key" shortcut={asShortcutKey(voiceHotkeyKey(i))} />
                    </div>
                );
            })}
            <VcNote>{t("voice_changer.hotkeys_note")}</VcNote>
        </VcDetails>
    );
};
