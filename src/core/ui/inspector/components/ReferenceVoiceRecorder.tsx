/**
 * <ReferenceVoiceRecorder>: record (or import) a clean 3-10 s reference clip into the shared voice library, with a
 * live level meter, the quality report of the take (trim, SNR, clipping, loudness), preview, and save with a name
 * (transcribed by the local STT). Shared by the voice changer's zero-shot voices and TTS cloning, so both pickers
 * list the same voices (`@/core/services/voice_library`).
 *
 * Owned by P5-F; stub scaffolded by P5-S0. Frozen props (change only through INTEGRATE; props may be added):
 * `onSaved(voice)` after a take is saved, `compact` for use inside a picker (no title, tips in one line, the voice
 * changer's microphone, and no frame of its own: the embedding picker draws one). P5-F adds `device` (capture device;
 * default: the voice changer's input device), `replaceVoiceId` (re-record a library voice in place: the save
 * replaces its clip and keeps its id) and `onClose` (shows a Close button).
 *
 * Also exported (P5-F): `<ReferenceVoicePicker value onChange>`, the reference-voice picker both consumers mount
 * (the voice changer's zero-shot voices, `reference_voice_id`; TTS cloning, P6): the library's voices, preview,
 * a "New voice" recorder inline, and editing of the selected voice (name, transcript, language, re-record,
 * delete). And `useClipPreview()`, the WAV preview player both use.
 *
 * Desktop only: renders nothing in the web build.
 */
import {
    CLIP_SECONDS,
    RECORD_SECONDS,
    claimRecorder,
    deleteVoice,
    discardTake,
    ensureVoiceLibraryHost,
    importClip,
    readClip,
    readTake,
    saveTake,
    startRecording,
    stopRecording,
    cancelRecording,
    updateVoice,
    voiceLibraryUi,
    type VoiceMeta,
    type VoiceTake,
} from "@/core/services/voice_library";
import {
    IMPORT_FILTER,
    canSaveTake,
    findVoice,
    formatClock,
    formatSeconds,
    isMissingVoice,
    levelHint,
    meterFraction,
    recordDeviceArg,
    sortedIssues,
    speechProgress,
    voiceLabel,
} from "@/core/services/voice_library/format";
import { WHISPER_LANGUAGES } from "@/core/services/stt/whisperModels";
import { isTauri } from "@tauri-apps/api/core";
import { open } from "@tauri-apps/plugin-dialog";
import classNames from "classnames";
import { FC, ReactNode, useEffect, useId, useMemo, useRef, useState } from "react";
import { useTranslation } from "react-i18next";
import {
    RiAddLine,
    RiArrowDownSLine,
    RiArrowUpSLine,
    RiCloseLine,
    RiDeleteBin6Line,
    RiErrorWarningLine,
    RiAlertLine,
    RiFolderMusicLine,
    RiLoader4Line,
    RiPlayFill,
    RiRecordCircleFill,
    RiRefreshLine,
    RiStopFill,
    RiMicLine,
} from "react-icons/ri";
import { toast } from "react-toastify";
import { useSnapshot } from "valtio";
import { InputRustAudioInput, InputSelect } from "./input";
import { useLanguageName } from "../stt/common";

// -------------------------------------------------------------------------------------------------
// Preview player
// -------------------------------------------------------------------------------------------------

/** A playing preview clip (a webview `Audio`; a fake in tests). */
export interface PreviewAudio {
    play(): Promise<void>;
    pause(): void;
    onEnded(callback: (() => void) | null): void;
}

/** What the preview player needs from the webview. */
export interface ClipPreviewEnv {
    createUrl(bytes: ArrayBuffer): string;
    revokeUrl(url: string): void;
    createAudio(url: string): PreviewAudio;
}

const webviewPreviewEnv: ClipPreviewEnv = {
    createUrl: (bytes) => URL.createObjectURL(new Blob([bytes], { type: "audio/wav" })),
    revokeUrl: (url) => URL.revokeObjectURL(url),
    createAudio: (url) => {
        const a = new Audio(url);
        return {
            play: () => a.play(),
            pause: () => a.pause(),
            onEnded: (callback) => {
                a.onended = callback;
            },
        };
    },
};

/**
 * Plays WAV bytes (a take or a saved clip), one clip at a time. Every `play` and `stop` starts a new generation: a
 * clip whose bytes arrive after a newer `play`, a `stop` or an unmount (the hook stops it) is dropped, so two quick
 * clicks never play two clips over each other and nothing plays after its panel closed.
 */
export class ClipPreview {
    private audio: PreviewAudio | null = null;
    private url: string | null = null;
    private generation = 0;
    private readonly onPlaying: (key: string | null) => void;
    private readonly env: ClipPreviewEnv;

    constructor(onPlaying: (key: string | null) => void, env: ClipPreviewEnv = webviewPreviewEnv) {
        this.onPlaying = onPlaying;
        this.env = env;
    }

    stop = (): void => {
        this.generation++;
        if (this.audio) {
            this.audio.onEnded(null);
            this.audio.pause();
            this.audio = null;
        }
        if (this.url) {
            this.env.revokeUrl(this.url);
            this.url = null;
        }
        this.onPlaying(null);
    };

    /** Load and play `key`'s clip. Rejects with a load or playback error, unless the request was superseded. */
    play = async (key: string, load: () => Promise<ArrayBuffer>): Promise<void> => {
        this.stop();
        const mine = this.generation;
        let bytes: ArrayBuffer;
        try {
            bytes = await load();
        } catch (e) {
            if (mine === this.generation) throw e;
            return;
        }
        if (mine !== this.generation) return;
        const url = this.env.createUrl(bytes);
        const audio = this.env.createAudio(url);
        this.url = url;
        this.audio = audio;
        audio.onEnded(() => {
            if (mine === this.generation) this.stop();
        });
        this.onPlaying(key);
        try {
            await audio.play();
        } catch (e) {
            if (mine !== this.generation) return;
            this.stop();
            throw e;
        }
    };
}

/** The preview player of a panel (stopped on unmount). */
export function useClipPreview() {
    const [playing, setPlaying] = useState<string | null>(null);
    const [player] = useState(() => new ClipPreview(setPlaying));
    useEffect(() => () => player.stop(), [player]);
    return { playing, play: player.play, stop: player.stop };
}

function errorText(e: unknown): string {
    return typeof e === "string" ? e : e instanceof Error ? e.message : String(e);
}

/** Swallow a rejection whose message the library already put in `voiceLibraryUi.error`. */
const shown = () => undefined;

// -------------------------------------------------------------------------------------------------
// Pieces
// -------------------------------------------------------------------------------------------------

const Hint: FC<{ children: ReactNode; className?: string }> = ({ children, className }) => (
    <p className={classNames("text-xs leading-snug opacity-70", className)}>{children}</p>
);

const ErrorLine: FC<{ text: string; onDismiss: () => void }> = ({ text, onDismiss }) => {
    const { t } = useTranslation();
    return (
        <div role="alert" className="alert alert-error alert-soft flex items-start gap-1 px-2 py-1.5 text-xs leading-snug">
            <span className="min-w-0 flex-1 break-words">{text}</span>
            <button type="button" className="btn btn-ghost btn-xs btn-square -me-1" onClick={onDismiss} title={t("voice_library.close")} aria-label={t("voice_library.close")}>
                <RiCloseLine />
            </button>
        </div>
    );
};

const Bar: FC<{ fraction: number; tone: string; label: string; marker?: number }> = ({ fraction, tone, label, marker }) => (
    <div
        className="relative h-2 w-full overflow-hidden rounded-full bg-base-content/10"
        role="meter"
        aria-label={label}
        aria-valuemin={0}
        aria-valuemax={100}
        aria-valuenow={Math.round(fraction * 100)}
    >
        <div className={classNames("absolute inset-y-0 start-0 rounded-full transition-[width] duration-75", tone)} style={{ width: `${fraction * 100}%` }} />
        {marker !== undefined && marker > 0 && <div className="absolute inset-y-0 w-0.5 bg-base-content/60" style={{ insetInlineStart: `calc(${marker * 100}% - 1px)` }} />}
    </div>
);

/** The live meters while recording. */
const RecordingMeters: FC = () => {
    const { t } = useTranslation();
    const ui = useSnapshot(voiceLibraryUi);
    const p = ui.progress;
    const hint = levelHint(p);
    const speech = speechProgress(p?.voicedMs);
    const levelTone = hint === "hot" ? "bg-error" : hint === "good" ? "bg-success" : "bg-warning";
    const speechTone = speech.state === "more" ? "bg-info" : "bg-success";
    return (
        <div className="flex flex-col gap-1.5" aria-live="polite">
            <div className="flex items-center justify-between gap-2 text-xs">
                <span className="flex items-center gap-1 font-semibold text-error">
                    <RiRecordCircleFill className="animate-pulse" aria-hidden />
                    {t("voice_library.recording")}
                </span>
                <span className="tabular-nums opacity-80">
                    {t("voice_library.elapsed", { elapsed: formatClock(p?.elapsedMs), max: formatClock(RECORD_SECONDS.default * 1000) })}
                </span>
            </div>
            <Bar fraction={meterFraction(p?.levelDb)} marker={meterFraction(p?.peakDb)} tone={levelTone} label={t("voice_library.level")} />
            <Hint className={classNames(hint === "hot" && "text-error opacity-100", hint === "quiet" && "text-warning opacity-100")}>
                {t(`voice_library.level_${hint}`)}
            </Hint>
            <Bar fraction={speech.fraction} tone={speechTone} label={t("voice_library.speech_heard", { seconds: formatSeconds(p?.voicedMs) })} />
            <div className="flex items-center justify-between gap-2 text-xs">
                <span className="tabular-nums">{t("voice_library.speech_heard", { seconds: formatSeconds(p?.voicedMs) })}</span>
                <span className="opacity-70 text-end">{t(`voice_library.speech_${speech.state}`, { min: CLIP_SECONDS.min, max: CLIP_SECONDS.max })}</span>
            </div>
        </div>
    );
};

/** The quality report of a take. */
const TakeReport: FC<{ take: VoiceTake; stopReason?: string | null }> = ({ take, stopReason }) => {
    const { t } = useTranslation();
    const q = take.quality;
    const issues = sortedIssues(q);
    const facts = [
        t("voice_library.take_summary", { seconds: formatSeconds(q.durationMs), trimmed: formatSeconds(q.trimmedMs) }),
        q.snrDb !== null && q.snrDb !== undefined ? t("voice_library.snr", { snr: Math.round(q.snrDb) }) : t("voice_library.snr_unknown"),
    ];
    return (
        <div className="flex flex-col gap-1.5">
            <span className="text-xs font-semibold">
                {t(take.source === "imported" ? "voice_library.take_title_imported" : "voice_library.take_title_recorded")}
            </span>
            <Hint>{facts.join(" · ")}</Hint>
            {stopReason && <Hint>{t(`voice_library.stop_reason_${stopReason}`, { defaultValue: "" })}</Hint>}
            {issues.length > 0 && (
                <ul className="flex flex-col gap-1">
                    {issues.map(({ issue, severity }) => (
                        <li
                            key={issue}
                            className={classNames(
                                "flex items-start gap-1.5 rounded-md px-2 py-1 text-xs leading-snug",
                                severity === "error" ? "bg-error/10 text-error" : "bg-warning/10 text-warning",
                            )}
                        >
                            {severity === "error" ? <RiErrorWarningLine className="mt-0.5 shrink-0" aria-hidden /> : <RiAlertLine className="mt-0.5 shrink-0" aria-hidden />}
                            <span className="min-w-0">{t(`voice_library.issue_${issue}`, { min: CLIP_SECONDS.min, max: CLIP_SECONDS.max })}</span>
                        </li>
                    ))}
                </ul>
            )}
            {!q.ok && <Hint>{t("voice_library.issue_blocking_hint")}</Hint>}
        </div>
    );
};

/** The transcript language: "Detect" or a language. A stored language the list does not offer (one the speech-to-text
 *  detected, e.g. Cantonese) stays selectable, so the select shows what is stored. */
const LanguageSelect: FC<{ value: string; onChange: (v: string) => void }> = ({ value, onChange }) => {
    const { t } = useTranslation();
    const languageName = useLanguageName();
    const stored = value.trim();
    const options = useMemo(() => {
        const codes = [...WHISPER_LANGUAGES];
        if (stored && stored !== "auto" && !codes.includes(stored)) codes.push(stored);
        return [
            { label: t("voice_library.language_auto"), value: "auto" },
            ...codes.map((code) => ({ label: languageName(code), value: code })).sort((a, b) => a.label.localeCompare(b.label)),
        ];
    }, [t, languageName, stored]);
    const current = options.some((o) => o.value === stored) ? stored : "auto";
    return <InputSelect label="voice_library.language" value={current} onValueChange={onChange} options={options} />;
};

// -------------------------------------------------------------------------------------------------
// <ReferenceVoiceRecorder>
// -------------------------------------------------------------------------------------------------

export interface ReferenceVoiceRecorderProps {
    /** Called with the saved voice (the library list refreshes by itself). */
    onSaved?: (voice: VoiceMeta) => void;
    /** Compact layout inside a picker (no frame of its own; the picker draws one). */
    compact?: boolean;
    /** Capture device (`""` / absent: the voice changer's input device; the full layout offers a picker). */
    device?: string;
    /** Re-record this library voice: saving replaces its clip and keeps its id (and every use of it). */
    replaceVoiceId?: string;
    /** Shows a Close button. */
    onClose?: () => void;
}

const ReferenceVoiceRecorderInner: FC<ReferenceVoiceRecorderProps> = ({ onSaved, compact, device, replaceVoiceId, onClose }) => {
    const { t } = useTranslation();
    const ui = useSnapshot(voiceLibraryUi);
    const preview = useClipPreview();
    const vcDevice = window.ApiServer?.voiceChanger?.state?.voiceChanger?.inputDevice ?? "";
    const [pickedDevice, setPickedDevice] = useState<string>(device ?? vcDevice);
    const [name, setName] = useState("");
    // Re-recording keeps the voice's language (the effect below follows a change of voice).
    const [language, setLanguage] = useState(() => (replaceVoiceId ? findVoice(voiceLibraryUi.voices, replaceVoiceId)?.language : null) ?? "auto");
    const [transcribe, setTranscribe] = useState(true);
    const [stopReason, setStopReason] = useState<string | null>(null);
    const replacing = replaceVoiceId ? findVoice(ui.voices as VoiceMeta[], replaceVoiceId) : undefined;
    // One recorder in Rust: the panel that started a recording or import shows it and its take; the others stay idle
    // and blocked meanwhile (a recording there would discard the take waiting here).
    const me = useId();
    const mine = ui.owner === null || ui.owner === me;
    const take = mine ? (ui.take as VoiceTake | null) : null;
    const recording = ui.recording && mine;
    const takeElsewhere = !mine && ui.take !== null && !ui.recording && ui.busy === null;
    const elsewhere = !mine && (ui.recording || ui.busy !== null || ui.take !== null);

    useEffect(() => {
        ensureVoiceLibraryHost();
        return () => {
            if (voiceLibraryUi.owner === me) claimRecorder(null);
        };
    }, [me]);

    // A new take: suggest its name (an imported file's) unless one was typed; remember why recording stopped.
    const lastTake = useRef<string | null>(null);
    useEffect(() => {
        if (!take || take.takeId === lastTake.current) return;
        lastTake.current = take.takeId;
        if (take.suggestedName) setName((n) => n.trim() || take.suggestedName || "");
        setStopReason(take.source === "recorded" ? voiceLibraryUi.progress?.stopReason ?? null : null);
    }, [take]);

    useEffect(() => {
        if (replacing) setLanguage(replacing.language ?? "auto");
    }, [replacing?.id]); // eslint-disable-line react-hooks/exhaustive-deps

    const fail = (e: unknown) => toast.error(errorText(e));

    const record = () => {
        preview.stop();
        setStopReason(null);
        claimRecorder(me);
        const capture = compact ? device ?? vcDevice : pickedDevice;
        // `""` would mean "the voice changer's microphone" to Rust: the picker's "System default" is spelled out.
        void startRecording({ device: compact && !device ? "" : recordDeviceArg(capture), maxSeconds: RECORD_SECONDS.default }).catch(shown);
    };

    const pickFile = async () => {
        preview.stop();
        let path: string | string[] | null;
        try {
            path = await open({
                title: t("voice_library.import_title"),
                multiple: false,
                directory: false,
                filters: [{ name: t("voice_library.import_filter"), extensions: IMPORT_FILTER.extensions }],
            });
        } catch (e) {
            fail(e);
            return;
        }
        if (typeof path === "string") {
            claimRecorder(me);
            await importClip(path).catch(shown);
        }
    };

    const save = async () => {
        // One save at a time: a second Enter or click while the first one writes would save the take twice. The live
        // state, not this render's: a key repeat arrives before the re-render.
        if (!take || voiceLibraryUi.busy !== null) return;
        preview.stop();
        try {
            let voice: VoiceMeta;
            if (replacing) {
                voice = await updateVoice({ id: replacing.id, takeId: take.takeId, language, transcribe });
                toast.success(t("voice_library.replaced", { name: voiceLabel(voice) }));
            } else {
                voice = await saveTake({ takeId: take.takeId, name: name.trim(), transcribe, language: language === "auto" ? null : language });
                toast.success(t("voice_library.saved", { name: voiceLabel(voice) }));
            }
            setName("");
            onSaved?.(voice);
        } catch {
            // Saving and replacing are tracked operations: the error is in `voiceLibraryUi.error` (shown above).
        }
    };

    const discard = () => {
        preview.stop();
        if (take) void discardTake(take.takeId).catch(() => undefined);
    };

    const busy = mine ? ui.busy : null;
    const working = busy === "stopping" || busy === "importing" || busy === "saving";
    // The take's buttons also wait for a recording that is starting (it discards the take once it runs).
    const takeLocked = working || busy === "starting";

    // Compact: no frame of its own (the picker that embeds it draws one).
    return (
        <div className="flex flex-col gap-2">
            {(!compact || onClose) && (
                <div className="flex items-center justify-between gap-2">
                    {!compact ? (
                        <span className="flex items-center gap-1.5 font-header font-bold text-primary">
                            <RiMicLine aria-hidden />
                            {replacing ? t("voice_library.rerecord_title", { name: voiceLabel(replacing) }) : t("voice_library.title")}
                        </span>
                    ) : (
                        <span className="text-xs font-semibold">
                            {replacing ? t("voice_library.rerecord_title", { name: voiceLabel(replacing) }) : t("voice_library.new_voice")}
                        </span>
                    )}
                    {onClose && (
                        <button type="button" className="btn btn-ghost btn-xs btn-square" onClick={onClose} title={t("voice_library.close")} aria-label={t("voice_library.close")}>
                            <RiCloseLine />
                        </button>
                    )}
                </div>
            )}

            {!take && !recording && !working && (
                <>
                    {!compact && <Hint>{t("voice_library.description")}</Hint>}
                    <Hint>{t(compact ? "voice_library.tips_compact" : "voice_library.tips")}</Hint>
                    <div className="rounded-md bg-base-content/5 px-2 py-1.5 text-xs leading-snug">
                        <span className="opacity-70">{t("voice_library.prompt_label")}</span>{" "}
                        <span className="italic" dir="auto">
                            {t("voice_library.prompt_sentence")}
                        </span>
                    </div>
                    {!compact && <InputRustAudioInput label="voice_library.device" value={pickedDevice} onChange={setPickedDevice} />}
                    {elsewhere && <Hint className="text-warning opacity-100">{t(takeElsewhere ? "voice_library.take_elsewhere" : "voice_library.busy_elsewhere")}</Hint>}
                    <div className="flex flex-wrap items-center gap-2">
                        <button type="button" className="btn btn-sm btn-error gap-1" disabled={busy === "starting" || elsewhere} onClick={record}>
                            {busy === "starting" ? <RiLoader4Line className="animate-spin" aria-hidden /> : <RiRecordCircleFill aria-hidden />}
                            {t("voice_library.record")}
                        </button>
                        <button type="button" className="btn btn-sm btn-ghost gap-1" disabled={busy === "starting" || elsewhere} onClick={() => void pickFile()}>
                            <RiFolderMusicLine aria-hidden />
                            {t("voice_library.import")}
                        </button>
                    </div>
                </>
            )}

            {ui.error && mine && !recording && !working && <ErrorLine text={ui.error} onDismiss={() => (voiceLibraryUi.error = null)} />}

            {recording && busy !== "stopping" && (
                <>
                    <RecordingMeters />
                    <div className="flex flex-wrap items-center gap-2">
                        <button type="button" className="btn btn-sm btn-primary gap-1" disabled={working} onClick={() => void stopRecording().catch(shown)}>
                            <RiStopFill aria-hidden />
                            {t("voice_library.stop")}
                        </button>
                        <button type="button" className="btn btn-sm btn-ghost" disabled={working} onClick={() => void cancelRecording().catch(fail)}>
                            {t("voice_library.cancel")}
                        </button>
                    </div>
                </>
            )}

            {working && (
                <div className="flex items-center gap-2 text-xs opacity-80" role="status">
                    <RiLoader4Line className="animate-spin" aria-hidden />
                    {t(busy === "importing" ? "voice_library.importing" : busy === "saving" ? "voice_library.saving" : "voice_library.processing")}
                </div>
            )}

            {take && !recording && busy !== "stopping" && busy !== "importing" && (
                <>
                    <TakeReport take={take} stopReason={stopReason} />
                    <div className="flex flex-wrap items-center gap-2">
                        <button
                            type="button"
                            className="btn btn-sm btn-ghost gap-1"
                            onClick={() =>
                                preview.playing === take.takeId ? preview.stop() : void preview.play(take.takeId, () => readTake(take.takeId)).catch(fail)
                            }
                        >
                            {preview.playing === take.takeId ? <RiStopFill aria-hidden /> : <RiPlayFill aria-hidden />}
                            {preview.playing === take.takeId ? t("voice_library.stop_playback") : t("voice_library.play")}
                        </button>
                        <button type="button" className="btn btn-sm btn-ghost gap-1" disabled={takeLocked} onClick={record}>
                            {busy === "starting" ? <RiLoader4Line className="animate-spin" aria-hidden /> : <RiRefreshLine aria-hidden />}
                            {t("voice_library.record_again")}
                        </button>
                        <button type="button" className="btn btn-sm btn-ghost gap-1" disabled={takeLocked} onClick={discard}>
                            <RiDeleteBin6Line aria-hidden />
                            {t("voice_library.discard")}
                        </button>
                    </div>
                    {take.quality.ok && (
                        <div className="flex flex-col gap-2">
                            {!replacing && (
                                <label className="flex flex-col gap-1 text-xs">
                                    <span className="font-medium text-base-content/80">{t("voice_library.name")}</span>
                                    <input
                                        className="input input-bordered input-sm w-full"
                                        value={name}
                                        maxLength={64}
                                        dir="auto"
                                        placeholder={t("voice_library.name_placeholder")}
                                        onChange={(e) => setName(e.target.value)}
                                        onKeyDown={(e) => {
                                            if (e.key === "Enter" && !e.repeat && !takeLocked && canSaveTake(take, name)) void save();
                                        }}
                                    />
                                </label>
                            )}
                            <LanguageSelect value={language} onChange={setLanguage} />
                            <label className="flex cursor-pointer items-center justify-between gap-2 text-xs">
                                <span className="font-medium text-base-content/80">{t("voice_library.transcribe")}</span>
                                <input type="checkbox" className="toggle toggle-primary toggle-sm" checked={transcribe} onChange={(e) => setTranscribe(e.target.checked)} />
                            </label>
                            <div>
                                <button
                                    type="button"
                                    className="btn btn-sm btn-primary"
                                    disabled={takeLocked || !(replacing ? take.quality.ok : canSaveTake(take, name))}
                                    onClick={() => void save()}
                                >
                                    {replacing ? t("voice_library.replace") : t("voice_library.save")}
                                </button>
                            </div>
                        </div>
                    )}
                </>
            )}
        </div>
    );
};

export const ReferenceVoiceRecorder: FC<ReferenceVoiceRecorderProps> = (props) => {
    if (!isTauri()) return null;
    return <ReferenceVoiceRecorderInner {...props} />;
};

export default ReferenceVoiceRecorder;

// -------------------------------------------------------------------------------------------------
// <ReferenceVoicePicker>
// -------------------------------------------------------------------------------------------------

const NONE = "__none__";
/** The placeholder of a picker without "None" while no voice is chosen. */
const CHOOSE = "__choose__";

/** Edit one library voice: name, transcript, language, re-record, delete. */
const VoiceEditor: FC<{ voice: VoiceMeta; onDeleted: () => void; onRerecord: () => void }> = ({ voice, onDeleted, onRerecord }) => {
    const { t } = useTranslation();
    const [name, setName] = useState(voice.name);
    const [transcript, setTranscript] = useState(voice.transcript ?? "");
    const [confirmDelete, setConfirmDelete] = useState(false);
    const [saving, setSaving] = useState(false);

    // Follow changes made elsewhere (a transcript that arrived, another window's rename).
    useEffect(() => setName(voice.name), [voice.name]);
    useEffect(() => setTranscript(voice.transcript ?? ""), [voice.transcript]);

    const fail = (e: unknown) => toast.error(errorText(e));
    const dirty = name.trim() !== voice.name || transcript.trim() !== (voice.transcript ?? "");

    const saveChanges = async () => {
        setSaving(true);
        try {
            await updateVoice({
                id: voice.id,
                name: name.trim() !== voice.name ? name : undefined,
                transcript: transcript.trim() !== (voice.transcript ?? "") ? transcript : undefined,
            });
        } catch (e) {
            fail(e);
        } finally {
            setSaving(false);
        }
    };

    const transcriptNote = voice.transcribing
        ? t("voice_library.transcribing")
        : !voice.transcript && voice.transcriptError
          ? t(`voice_library.transcript_error_${voice.transcriptError}`, { defaultValue: "" })
          : null;

    return (
        <div className="flex flex-col gap-2 rounded-lg border border-base-content/10 p-2">
            <Hint>
                {t("voice_library.facts", {
                    seconds: formatSeconds(voice.durationMs),
                    rate: Math.round(voice.sampleRate / 100) / 10,
                    lufs: Math.round(voice.lufs),
                })}
                {" · "}
                {t(voice.source === "imported" ? "voice_library.source_imported" : "voice_library.source_recorded")}
            </Hint>
            <label className="flex flex-col gap-1 text-xs">
                <span className="font-medium text-base-content/80">{t("voice_library.name")}</span>
                <input className="input input-bordered input-sm w-full" value={name} maxLength={64} dir="auto" onChange={(e) => setName(e.target.value)} />
            </label>
            <label className="flex flex-col gap-1 text-xs">
                <span className="font-medium text-base-content/80">{t("voice_library.transcript")}</span>
                <textarea
                    className="w-full h-16 p-2 bg-base-100 border border-base-content/20 rounded text-xs"
                    dir="auto"
                    value={transcript}
                    maxLength={2000}
                    placeholder={t("voice_library.transcript_placeholder")}
                    onChange={(e) => setTranscript(e.target.value)}
                />
            </label>
            {transcriptNote && (
                <Hint className="flex items-center gap-1">
                    {voice.transcribing && <RiLoader4Line className="animate-spin" aria-hidden />}
                    {transcriptNote}
                </Hint>
            )}
            <LanguageSelect value={voice.language ?? "auto"} onChange={(v) => void updateVoice({ id: voice.id, language: v }).catch(fail)} />
            <div className="flex flex-wrap items-center gap-2">
                <button type="button" className="btn btn-xs btn-primary" disabled={!dirty || saving || !name.trim()} onClick={() => void saveChanges()}>
                    {t("voice_library.save_changes")}
                </button>
                <button
                    type="button"
                    className="btn btn-xs btn-ghost gap-1"
                    disabled={voice.transcribing}
                    onClick={() => void updateVoice({ id: voice.id, transcribe: true }).catch(fail)}
                >
                    <RiRefreshLine aria-hidden />
                    {t("voice_library.retranscribe")}
                </button>
                <button type="button" className="btn btn-xs btn-ghost gap-1" onClick={onRerecord}>
                    <RiRecordCircleFill aria-hidden />
                    {t("voice_library.rerecord")}
                </button>
                <button type="button" className="btn btn-xs btn-ghost gap-1 text-error" onClick={() => setConfirmDelete(true)}>
                    <RiDeleteBin6Line aria-hidden />
                    {t("voice_library.delete")}
                </button>
            </div>
            {confirmDelete && (
                <div role="alert" className="flex flex-col gap-2 rounded-md bg-error/10 p-2 text-xs">
                    <span>{t("voice_library.delete_confirm", { name: voiceLabel(voice) })}</span>
                    <div className="flex gap-2">
                        <button
                            type="button"
                            className="btn btn-xs btn-error"
                            onClick={() =>
                                void deleteVoice(voice.id)
                                    .then(() => {
                                        toast.success(t("voice_library.deleted", { name: voiceLabel(voice) }));
                                        onDeleted();
                                    })
                                    .catch(fail)
                            }
                        >
                            {t("voice_library.delete_yes")}
                        </button>
                        <button type="button" className="btn btn-xs btn-ghost" onClick={() => setConfirmDelete(false)}>
                            {t("voice_library.cancel")}
                        </button>
                    </div>
                </div>
            )}
        </div>
    );
};

export interface ReferenceVoicePickerProps {
    /** The selected voice id (`""` = none). */
    value: string;
    onChange: (id: string) => void;
    /** Label (i18n key); default `voice_library.picker_label`. */
    label?: string;
    /** Offer "None" (default true). Without it, an empty `value` shows a "Choose a voice…" placeholder (never a voice
     *  the consumer has not chosen). */
    allowNone?: boolean;
    disabled?: boolean;
    /** The consumer needs a transcript (TTS cloning): say so when the selected voice has none. */
    needsTranscript?: boolean;
}

type PickerPanel = "new" | "edit" | "rerecord" | null;

const ReferenceVoicePickerInner: FC<ReferenceVoicePickerProps> = ({ value, onChange, label, allowNone = true, disabled, needsTranscript }) => {
    const { t } = useTranslation();
    const ui = useSnapshot(voiceLibraryUi);
    const voices = ui.voices as VoiceMeta[];
    const preview = useClipPreview();
    const [panel, setPanel] = useState<PickerPanel>(null);
    const selected = findVoice(voices, value);
    const missing = isMissingVoice(voices, value, ui.loaded);

    useEffect(() => {
        ensureVoiceLibraryHost();
    }, []);

    const options = useMemo(() => {
        const list: { label: string; value: string }[] = [];
        if (allowNone) list.push({ label: t("voice_library.picker_none"), value: NONE });
        // No "None" and nothing chosen yet: say so, rather than showing a voice the consumer has not got.
        else if (!value) list.push({ label: t("voice_library.picker_choose"), value: CHOOSE });
        if (missing && value) list.push({ label: t("voice_library.picker_missing", { id: value }), value });
        for (const v of voices) list.push({ label: t("voice_library.voice_option", { name: voiceLabel(v), seconds: formatSeconds(v.durationMs) }), value: v.id });
        return list;
    }, [voices, allowNone, missing, value, t]);

    const current = value ? value : allowNone ? NONE : CHOOSE;
    const fail = (e: unknown) => toast.error(errorText(e));

    return (
        <div className={classNames("flex flex-col gap-1.5", disabled && "pointer-events-none opacity-50")}>
            <InputSelect
                label={label ?? "voice_library.picker_label"}
                value={current}
                onValueChange={(v) => {
                    if (v === CHOOSE) return;
                    preview.stop();
                    onChange(v === NONE ? "" : v);
                }}
                options={options}
            />
            {ui.loaded && voices.length === 0 && panel !== "new" && <Hint>{t("voice_library.picker_empty")}</Hint>}
            {missing && <Hint className="text-warning opacity-100">{t("voice_library.picker_missing_hint")}</Hint>}
            {needsTranscript && selected && !selected.transcript && (
                <Hint className={classNames(!selected.transcribing && "text-warning opacity-100")}>
                    {selected.transcribing ? t("voice_library.transcribing") : t("voice_library.transcript_needed")}
                </Hint>
            )}
            <div className="flex flex-wrap items-center gap-1">
                {selected && (
                    <button
                        type="button"
                        className="btn btn-xs btn-ghost gap-1"
                        onClick={() =>
                            preview.playing === selected.id ? preview.stop() : void preview.play(selected.id, () => readClip(selected.id)).catch(fail)
                        }
                    >
                        {preview.playing === selected.id ? <RiStopFill aria-hidden /> : <RiPlayFill aria-hidden />}
                        {preview.playing === selected.id ? t("voice_library.stop_playback") : t("voice_library.play_voice")}
                    </button>
                )}
                <button type="button" className="btn btn-xs btn-ghost gap-1" aria-expanded={panel === "new"} onClick={() => setPanel(panel === "new" ? null : "new")}>
                    <RiAddLine aria-hidden />
                    {t("voice_library.new_voice")}
                </button>
                {selected && (
                    <button
                        type="button"
                        className="btn btn-xs btn-ghost gap-1"
                        aria-expanded={panel === "edit" || panel === "rerecord"}
                        onClick={() => setPanel(panel === "edit" || panel === "rerecord" ? null : "edit")}
                    >
                        {panel === "edit" || panel === "rerecord" ? <RiArrowUpSLine aria-hidden /> : <RiArrowDownSLine aria-hidden />}
                        {t("voice_library.manage")}
                    </button>
                )}
            </div>
            {panel === "new" && (
                <div className="rounded-lg border border-base-content/10 p-2">
                    <ReferenceVoiceRecorder
                        compact
                        onClose={() => setPanel(null)}
                        onSaved={(voice) => {
                            onChange(voice.id);
                            setPanel(null);
                        }}
                    />
                </div>
            )}
            {panel === "edit" && selected && (
                // Keyed by voice: switching voices while editing starts from the new voice's name and transcript (a
                // draft typed for the previous voice is never saved into this one).
                <VoiceEditor
                    key={selected.id}
                    voice={selected}
                    onDeleted={() => {
                        onChange("");
                        setPanel(null);
                    }}
                    onRerecord={() => setPanel("rerecord")}
                />
            )}
            {panel === "rerecord" && selected && (
                <div className="rounded-lg border border-base-content/10 p-2">
                    <ReferenceVoiceRecorder key={selected.id} compact replaceVoiceId={selected.id} onClose={() => setPanel("edit")} onSaved={() => setPanel("edit")} />
                </div>
            )}
        </div>
    );
};

/** The library's reference-voice picker (see the module docs). Desktop only. */
export const ReferenceVoicePicker: FC<ReferenceVoicePickerProps> = (props) => {
    if (!isTauri()) return null;
    return <ReferenceVoicePickerInner {...props} />;
};
