/**
 * Pure presentation helpers of the voice library (no Tauri, DOM or `window`): issue severities, meter scaling,
 * recording guidance, voice labels and the save check. Used by `<ReferenceVoiceRecorder>` and the voice pickers.
 * Owned by P5-F.
 */
import {
    BLOCKING_ISSUES,
    CLIP_ISSUES,
    CLIP_SECONDS,
    IMPORT_EXTENSIONS,
    type ClipIssue,
    type ClipQuality,
    type RecordProgress,
    type VoiceMeta,
    type VoiceTake,
} from "./contract";

export type IssueSeverity = "error" | "warning";

/** Whether `issue` makes a take unusable. */
export function isBlockingIssue(issue: string): boolean {
    return (BLOCKING_ISSUES as readonly string[]).includes(issue);
}

/** Known issues of a quality report, blocking ones first (unknown codes from a newer build are dropped). */
export function sortedIssues(quality: Pick<ClipQuality, "issues"> | null | undefined): { issue: ClipIssue; severity: IssueSeverity }[] {
    const known = (quality?.issues ?? []).filter((i): i is ClipIssue => (CLIP_ISSUES as readonly string[]).includes(i));
    const unique = [...new Set(known)];
    return unique
        .map((issue) => ({ issue, severity: (isBlockingIssue(issue) ? "error" : "warning") as IssueSeverity }))
        .sort((a, b) => (a.severity === b.severity ? CLIP_ISSUES.indexOf(a.issue) - CLIP_ISSUES.indexOf(b.issue) : a.severity === "error" ? -1 : 1));
}

/** Meter floor (dBFS): levels at or below it draw an empty bar. */
export const METER_FLOOR_DB = -60;

/** A dBFS level as a 0..1 meter position (linear in dB between `METER_FLOOR_DB` and 0). */
export function meterFraction(db: number | null | undefined): number {
    if (typeof db !== "number" || !Number.isFinite(db)) return 0;
    return Math.min(1, Math.max(0, (db - METER_FLOOR_DB) / -METER_FLOOR_DB));
}

/** What the level says while recording: too quiet below -45 dBFS RMS, hot above -6 dBFS peak or when clipping. */
export type LevelHint = "silent" | "quiet" | "good" | "hot";

export function levelHint(progress: Pick<RecordProgress, "levelDb" | "peakDb" | "clipping"> | null | undefined): LevelHint {
    if (!progress) return "silent";
    if (progress.clipping || progress.peakDb > -1) return "hot";
    if (progress.levelDb <= -70) return "silent";
    if (progress.levelDb < -45) return "quiet";
    return "good";
}

/** `m:ss` of a duration in milliseconds. */
export function formatClock(ms: number | null | undefined): string {
    const total = Math.max(0, Math.floor((typeof ms === "number" && Number.isFinite(ms) ? ms : 0) / 1000));
    return `${Math.floor(total / 60)}:${String(total % 60).padStart(2, "0")}`;
}

/** Seconds with one decimal (`6.4`). */
export function formatSeconds(ms: number | null | undefined): string {
    const v = typeof ms === "number" && Number.isFinite(ms) ? ms / 1000 : 0;
    return (Math.round(v * 10) / 10).toFixed(1);
}

/** Speech heard while recording, against the 3-10 s a reference needs. */
export interface SpeechProgress {
    /** 0..1 of the 10 s maximum. */
    fraction: number;
    /** `more`: under 3 s; `enough`: 3-10 s; `full`: 10 s or more (the rest is cut). */
    state: "more" | "enough" | "full";
}

export function speechProgress(voicedMs: number | null | undefined): SpeechProgress {
    const ms = typeof voicedMs === "number" && Number.isFinite(voicedMs) ? Math.max(0, voicedMs) : 0;
    const max = CLIP_SECONDS.max * 1000;
    const state = ms >= max ? "full" : ms >= CLIP_SECONDS.min * 1000 ? "enough" : "more";
    return { fraction: Math.min(1, ms / max), state };
}

/** A take can be saved under `name`: no blocking issue and a non-blank name. */
export function canSaveTake(take: Pick<VoiceTake, "quality"> | null | undefined, name: string): boolean {
    return !!take && take.quality.ok && name.trim().length > 0;
}

/** The voice's label in a picker: its name, else its id. */
export function voiceLabel(voice: Pick<VoiceMeta, "name" | "id">): string {
    return voice.name.trim() || voice.id;
}

/** The voice with `id`, if listed. */
export function findVoice(voices: readonly VoiceMeta[], id: string | null | undefined): VoiceMeta | undefined {
    if (!id) return undefined;
    return voices.find((v) => v.id === id);
}

/** Whether a picker's value names a voice that is not (or no longer) in the library. */
export function isMissingVoice(voices: readonly VoiceMeta[], id: string | null | undefined, loaded: boolean): boolean {
    return loaded && !!id && !findVoice(voices, id);
}

/** The file picker filter of `importClip`. */
export const IMPORT_FILTER = { extensions: [...IMPORT_EXTENSIONS] };

/** The device sent to `startRecording`: the picked device, or the system default spelled out (`""` would mean the
 *  voice changer's input device to Rust). */
export function recordDeviceArg(picked: string | null | undefined): string {
    const d = (picked ?? "").trim();
    return d === "" ? "default" : d;
}
