/**
 * Voice-library IPC contract (P5-S0, frozen; change only through INTEGRATE; fields may be added): command
 * names, event names and payload types of the `voice-library` plugin (Rust `src-tauri/src/services/voice_library`,
 * commands in `src-tauri/plugin-commands/voice-library.txt`). Payloads and arguments are camelCase, e.g.
 * `invoke(VL_COMMANDS.get, { id })`, `invoke(VL_COMMANDS.saveTake, { request })`. Errors are strings.
 *
 * The library holds one clean 3-10 s reference clip per voice under `app_data/voices/<id>/` (`ref.wav`,
 * `meta.json`, `cache/<backend>.bin`), shared by the voice changer's zero-shot voices (`reference_voice_id`) and
 * TTS cloning. Owned by P5-F. Pure module: no Tauri, DOM or `window`.
 */

const P = "plugin:voice-library|";

/** Every `voice-library` plugin command (full IPC names). */
export const VL_COMMANDS = {
    /** `-> VoiceMeta[]` (sorted by name). */
    list: `${P}voice_library_list`,
    /** `{ id } -> VoiceMeta`. */
    get: `${P}voice_library_get`,
    /** `{ request: RecordRequest } -> null`; progress on `VL_EVENTS.recordProgress`. */
    recordStart: `${P}voice_library_record_start`,
    /** `-> VoiceTake` (trimmed, checked, normalised; not saved yet). */
    recordStop: `${P}voice_library_record_stop`,
    /** `-> null`. */
    recordCancel: `${P}voice_library_record_cancel`,
    /** `{ request: ImportClipRequest } -> VoiceTake`. */
    importClip: `${P}voice_library_import_clip`,
    /** `{ takeId } -> ArrayBuffer` (WAV bytes, binary IPC). */
    readTake: `${P}voice_library_read_take`,
    /** `{ id } -> ArrayBuffer` (WAV bytes, binary IPC). */
    readClip: `${P}voice_library_read_clip`,
    /** `{ takeId } -> null`. */
    discardTake: `${P}voice_library_discard_take`,
    /** `{ request: SaveTakeRequest } -> VoiceMeta`. */
    saveTake: `${P}voice_library_save_take`,
    /** `{ request: UpdateVoiceRequest } -> VoiceMeta`. */
    update: `${P}voice_library_update`,
    /** `{ id } -> null`. */
    delete: `${P}voice_library_delete`,
} as const;

/** Every `voice-library` event. */
export const VL_EVENTS = {
    /** No payload: re-list the voices. */
    changed: "voice_library:changed",
    /** `RecordProgress`, at most 20 per second while recording. */
    recordProgress: "voice_library:record_progress",
} as const;

/** Shortest and longest saved clip (seconds, after trimming). */
export const CLIP_SECONDS = { min: 3, max: 10 } as const;

/** `RecordRequest.maxSeconds`: Rust clamps it to `min..max` and uses `default` for 0 / absent. */
export const RECORD_SECONDS = { min: 3, max: 30, default: 12 } as const;

/** Loudness every saved clip is normalised to (LUFS), under a -1 dBFS sample peak. */
export const TARGET_LUFS = -20;

/** Below this speech-to-noise ratio a take is `noisy` (dB). */
export const NOISY_SNR_DB = 15;

/** Issue codes of `ClipQuality.issues`. */
export const CLIP_ISSUES = ["too_short", "too_long", "no_speech", "noisy", "clipped", "too_quiet"] as const;
export type ClipIssue = (typeof CLIP_ISSUES)[number];

/** Issues that make a take unusable (Rust refuses to save it); the others are warnings. */
export const BLOCKING_ISSUES: readonly ClipIssue[] = ["no_speech", "too_short"];

/** `RecordProgress.stopReason`: why the recorder stopped by itself. */
export const STOP_REASONS = ["max_length", "device_changed", "interrupted", "device_lost", "no_audio"] as const;
export type StopReason = (typeof STOP_REASONS)[number];

/** `VoiceMeta.transcriptError`: why there is no automatic transcript (`needs_language`: the clip's language is
 *  unknown and no installed model detects it, e.g. only English-only Whisper models; set it and transcribe again). */
export const TRANSCRIPT_ERRORS = ["no_stt_model", "failed", "empty", "needs_language"] as const;
export type TranscriptError = (typeof TRANSCRIPT_ERRORS)[number];

/** Audio files `importClip` reads (the file picker's filter). */
export const IMPORT_EXTENSIONS = ["wav", "wave", "flac", "ogg", "mp3"] as const;

/** One library voice (Rust `VoiceMeta`, `meta.json`). */
export interface VoiceMeta {
    /** Folder name under `voices/` (stable id). */
    id: string;
    name: string;
    createdUnixMs: number;
    durationMs: number;
    sampleRate: number;
    /** Loudness of the saved clip (LUFS). */
    lufs: number;
    snrDb: number | null;
    /** Sample peak of the saved clip (dBFS). */
    peakDbfs: number;
    /** Some samples were clipped in the original take. */
    clipped: boolean;
    transcript: string | null;
    /** BCP-47 language of the transcript. */
    language: string | null;
    source: "recorded" | "imported";
    /** sha256 of `ref.wav`. */
    sha256: string;
    /** Backends with a cached embedding (`cache/<backend>.bin`). */
    cachedBackends: string[];
    /** The local STT is writing the transcript (a `changed` event follows). */
    transcribing?: boolean;
    /** Why the last automatic transcript failed; cleared by a transcript. */
    transcriptError?: TranscriptError | null;
}

/** The checks of a take (Rust `ClipQuality`). */
export interface ClipQuality {
    durationMs: number;
    /** Audio trimmed off: silence at the ends, long pauses spliced down, anything past 10 s. */
    trimmedMs: number;
    snrDb: number | null;
    /** Sample peak of the kept audio before normalisation (dBFS). */
    peakDbfs: number;
    /** Fraction of clipped samples (0..1). */
    clippedRatio: number;
    lufsBefore: number;
    lufsAfter: number;
    /** Usable as a reference (no blocking issue). */
    ok: boolean;
    /** `BLOCKING_ISSUES` make `ok` false; the rest are warnings. */
    issues: ClipIssue[];
}

/** A recorded or imported clip waiting to be saved or discarded (Rust `VoiceTake`). */
export interface VoiceTake {
    takeId: string;
    quality: ClipQuality;
    source: "recorded" | "imported";
    suggestedName: string | null;
    /** Sample rate of the processed clip (Hz). */
    sampleRate?: number;
}

/** `voice_library_record_start { request }`. */
export interface RecordRequest {
    /** Capture device name (`""` = the voice changer's input device, else the system default; `"default"` = the
     *  system default). */
    device?: string;
    /** Stops by itself after this many seconds (default 12, at most 30). */
    maxSeconds?: number;
}

/** Payload of `VL_EVENTS.recordProgress`. */
export interface RecordProgress {
    /** Audio recorded so far. */
    elapsedMs: number;
    /** RMS level of the last ~50 ms (dBFS). */
    levelDb: number;
    /** Sample peak of the last ~50 ms (dBFS). */
    peakDb: number;
    /** A clipped sample was recorded in this take (sticky). */
    clipping: boolean;
    /** Speech heard so far (Silero). */
    voicedMs: number;
    /** The recorder stopped by itself (the take's last event): `stopRecording()` fetches the take. */
    stopped?: boolean;
    stopReason?: StopReason | null;
}

/** `voice_library_import_clip { request }`. */
export interface ImportClipRequest {
    path: string;
}

/** `voice_library_save_take { request }`. */
export interface SaveTakeRequest {
    takeId: string;
    name: string;
    /** Transcribe with the local STT (default true). */
    transcribe?: boolean;
    /** Language hint (BCP-47; absent = detect). */
    language?: string | null;
}

/** `voice_library_update { request }` (absent fields are kept). */
export interface UpdateVoiceRequest {
    id: string;
    name?: string | null;
    /** `""` clears it. */
    transcript?: string | null;
    /** `""` / `"auto"` clears it. */
    language?: string | null;
    /** Replace the clip with this take, keeping the id (cached embeddings are dropped; the transcript is redone
     *  unless `transcript` is given or `transcribe` is false). */
    takeId?: string | null;
    /** `true`: run the local STT again and replace the transcript. */
    transcribe?: boolean | null;
}

/** A voice id: 1-64 characters of `[a-z0-9-]`, not starting or ending with `-` (Rust `valid_voice_id`). */
export function validVoiceId(id: unknown): id is string {
    return typeof id === "string" && id.length > 0 && id.length <= 64 && /^[a-z0-9](?:[a-z0-9-]*[a-z0-9])?$/.test(id);
}
