// Voice-changer inspector (P5-G): the zero-shot reference picker. Lists the shared voice library
// (`@/core/services/voice_library`, the same voices TTS cloning offers), previews a voice's clip, and opens the
// shared recorder (`<ReferenceVoiceRecorder>`, P5-F) to add a voice, which is then picked.
import { clipSecondsLabel, referenceState } from "@/core/services/voice_changer/families";
import { readClip, voiceLibraryUi, type VoiceMeta } from "@/core/services/voice_library";
import { isTauri } from "@tauri-apps/api/core";
import { FC, useCallback, useEffect, useRef, useState } from "react";
import { useTranslation } from "react-i18next";
import { RiAddLine, RiPlayFill, RiStopFill } from "react-icons/ri";
import { toast } from "react-toastify";
import { useSnapshot } from "valtio";
import Tooltip from "../../dropdown/Tooltip";
import { ReferenceVoiceRecorder } from "../components/ReferenceVoiceRecorder";
import { InspectorInfoButton, VcNote } from "./common";
import { useVoiceLibraryHost } from "./hooks";

/** Play one library clip at a time (WAV bytes from the voice library, as a blob URL). */
function useClipPreview() {
    const audio = useRef<HTMLAudioElement | null>(null);
    const url = useRef<string | null>(null);
    const [playing, setPlaying] = useState<string | null>(null);
    const [loading, setLoading] = useState<string | null>(null);
    const request = useRef(0);

    const stop = useCallback(() => {
        request.current++;
        audio.current?.pause();
        audio.current = null;
        if (url.current) URL.revokeObjectURL(url.current);
        url.current = null;
        setPlaying(null);
        setLoading(null);
    }, []);

    useEffect(() => stop, [stop]);

    const play = useCallback(
        async (id: string, onError: (e: unknown) => void) => {
            stop();
            const mine = ++request.current;
            setLoading(id);
            try {
                const bytes = await readClip(id);
                if (mine !== request.current) return;
                const u = URL.createObjectURL(new Blob([bytes], { type: "audio/wav" }));
                const a = new Audio(u);
                url.current = u;
                audio.current = a;
                a.onended = () => {
                    if (audio.current === a) stop();
                };
                setLoading(null);
                setPlaying(id);
                await a.play();
            } catch (e) {
                if (mine !== request.current) return;
                stop();
                onError(e);
            }
        },
        [stop],
    );

    return { playing, loading, play, stop };
}

function voiceLabel(v: VoiceMeta): string {
    const secs = clipSecondsLabel(v.durationMs);
    return secs ? `${v.name} (${secs})` : v.name;
}

/**
 * Pick a voice-library voice as a zero-shot reference. `value` is a voice id (`""` = none); `required`
 * hides the "none" choice once one is picked (a zero-shot voice needs one). The library loads when this mounts.
 */
export const ReferenceVoicePicker: FC<{
    value: string;
    onChange: (id: string) => void;
    labelKey?: string;
    infoKey?: string;
    required?: boolean;
    disabled?: boolean;
}> = ({ value, onChange, labelKey = "voice_changer.family_reference", infoKey = "voice_changer.family_reference_info", required, disabled }) => {
    const { t } = useTranslation();
    useVoiceLibraryHost();
    const lib = useSnapshot(voiceLibraryUi);
    const voices = lib.voices as readonly VoiceMeta[];
    const [adding, setAdding] = useState(false);
    const preview = useClipPreview();
    const state = referenceState(value, voices, lib.loaded);
    const current = voices.find((v) => v.id === value.trim());

    if (!isTauri()) return <VcNote>{t("voice_changer.voices_desktop_only")}</VcNote>;

    const options: { value: string; label: string; disabled?: boolean }[] = [];
    if (!required || value.trim() === "") {
        options.push({ value: "", label: required ? t("voice_changer.family_reference_choose") : t("voice_changer.family_reference_none"), disabled: !!required });
    }
    for (const v of voices) options.push({ value: v.id, label: voiceLabel(v) });
    if (state === "missing" || state === "unknown") {
        options.push({
            value: value.trim(),
            label: state === "missing" ? t("voice_changer.family_reference_missing_option", { id: value.trim() }) : value.trim(),
            disabled: true,
        });
    }

    return (
        <div className="flex w-full min-w-0 flex-col gap-1.5">
            <div className="flex min-w-0 items-center gap-1">
                <span className="min-w-0 flex-1 text-xs font-medium leading-snug text-base-content/80">{t(labelKey)}</span>
                {infoKey ? <InspectorInfoButton text={t(infoKey)} /> : null}
            </div>
            <div className="flex min-w-0 items-center gap-1">
                <select
                    className="select select-bordered select-sm min-w-0 flex-1"
                    aria-label={t(labelKey)}
                    value={value.trim()}
                    disabled={disabled}
                    onChange={(e) => {
                        preview.stop();
                        onChange(e.target.value);
                    }}
                >
                    {options.map((o) => (
                        <option key={o.value || "-"} value={o.value} disabled={o.disabled}>
                            {o.label}
                        </option>
                    ))}
                </select>
                <Tooltip content={preview.playing === value.trim() ? t("voice_changer.family_reference_stop") : t("voice_changer.family_reference_play")}>
                    <button
                        type="button"
                        className="btn btn-ghost btn-sm btn-square shrink-0"
                        aria-label={preview.playing === value.trim() ? t("voice_changer.family_reference_stop") : t("voice_changer.family_reference_play")}
                        disabled={state !== "ok"}
                        onClick={() => {
                            const id = value.trim();
                            if (preview.playing === id || preview.loading === id) preview.stop();
                            else void preview.play(id, (e) => toast.error(t("voice_changer.family_reference_play_failed", { message: String(e) })));
                        }}
                    >
                        {preview.loading === value.trim() ? (
                            <span className="loading loading-spinner loading-xs" />
                        ) : preview.playing === value.trim() ? (
                            <RiStopFill />
                        ) : (
                            <RiPlayFill />
                        )}
                    </button>
                </Tooltip>
            </div>
            {current?.transcript ? (
                <VcNote className="line-clamp-2 italic">{t("voice_changer.family_reference_transcript", { text: current.transcript })}</VcNote>
            ) : null}
            {state === "missing" ? <VcNote tone="warn">{t("voice_changer.family_reference_missing")}</VcNote> : null}
            {lib.loaded && voices.length === 0 && !adding ? <VcNote>{t("voice_changer.family_reference_empty")}</VcNote> : null}
            {lib.error && !lib.loaded ? <VcNote tone="warn">{t("voice_changer.family_reference_list_failed", { message: lib.error })}</VcNote> : null}
            {adding ? (
                <div className="flex flex-col gap-1.5 rounded-lg border border-base-content/10 p-2">
                    <ReferenceVoiceRecorder
                        compact
                        onSaved={(v) => {
                            setAdding(false);
                            onChange(v.id);
                        }}
                    />
                    <button type="button" className="btn btn-ghost btn-xs self-end" onClick={() => setAdding(false)}>
                        {t("voice_changer.voice_cancel")}
                    </button>
                </div>
            ) : (
                <button type="button" className="btn btn-ghost btn-xs gap-1 self-start" disabled={disabled} onClick={() => setAdding(true)}>
                    <RiAddLine aria-hidden /> {t("voice_changer.family_reference_add")}
                </button>
            )}
        </div>
    );
};
