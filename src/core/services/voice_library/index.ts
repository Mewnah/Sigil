/**
 * Host module of the shared reference-voice library (`plugin:voice-library`, Rust `services/voice_library`):
 * the voices list for every picker (the voice changer's zero-shot voices, TTS cloning), recording and importing a
 * reference clip, and editing the library. Not an `ApiServer` service: consumers import these functions.
 *
 * Owned by P5-F (with `src/core/ui/inspector/components/ReferenceVoiceRecorder.tsx`). Frozen by P5-S0: the
 * exports below and their signatures change only through INTEGRATE; exports may be added.
 *
 * - `voiceLibraryUi`: valtio state for pickers and the recorder (read-only for consumers): `voices`, `loaded`,
 *   `recording`, `progress` (the last `RecordProgress`), `error`; P5-F adds `take` (the take waiting to be saved
 *   or discarded), `busy` (the operation in flight) and `owner` (the recorder panel that started the recording or
 *   import, `claimRecorder`; `null` = any panel may show it, e.g. after its panel closed).
 * - `ensureVoiceLibraryHost()`: idempotent; subscribes to `voice_library:changed` / `record_progress` and loads
 *   the list once. Pickers call it on mount. A no-op outside the desktop app.
 * - `refreshVoices()`, `startRecording(request?)`, `stopRecording()`, `cancelRecording()`, `importClip(path)`,
 *   `readTake(takeId)` / `readClip(id)` (WAV bytes for preview), `discardTake(takeId)`, `saveTake(request)`,
 *   `updateVoice(request)`, `deleteVoice(id)`: the IPC commands of `contract.ts`; they reject with the Rust error
 *   string. Recording, importing, saving and replacing a voice's clip (`updateVoice` with `takeId`) are the UI's
 *   `busy` operation; a waiting take is discarded only once a new recording started or a new import arrived.
 * - When the recorder stops by itself (`RecordProgress.stopped`: its length limit, or the device dropped out),
 *   the host fetches the take at once (`stopRecording`), so `take` fills without a click.
 */
import { invoke, isTauri } from "@tauri-apps/api/core";
import { listen, type UnlistenFn } from "@tauri-apps/api/event";
import { proxy } from "valtio";

import {
    VL_COMMANDS,
    VL_EVENTS,
    type ImportClipRequest,
    type RecordProgress,
    type RecordRequest,
    type SaveTakeRequest,
    type UpdateVoiceRequest,
    type VoiceMeta,
    type VoiceTake,
} from "./contract";

export * from "./contract";

/** The library operation in flight (one at a time from the UI's point of view). */
export type VoiceLibraryBusy = "starting" | "stopping" | "importing" | "saving";

export interface VoiceLibraryUiState {
    voices: VoiceMeta[];
    /** The list was read at least once. */
    loaded: boolean;
    recording: boolean;
    progress: RecordProgress | null;
    error: string | null;
    /** The recorded or imported take waiting to be saved or discarded. */
    take: VoiceTake | null;
    busy: VoiceLibraryBusy | null;
    /** The recorder panel showing the recording / take (`null`: whichever panel is open). */
    owner: string | null;
}

export const voiceLibraryUi = proxy<VoiceLibraryUiState>({
    voices: [],
    loaded: false,
    recording: false,
    progress: null,
    error: null,
    take: null,
    busy: null,
    owner: null,
});

/** A recorder panel takes the recorder (before `startRecording` / `importClip`), or releases it (`null`, or on
 *  unmount: the take stays and the next panel shows it). */
export function claimRecorder(owner: string | null): void {
    voiceLibraryUi.owner = owner;
}

function message(e: unknown): string {
    return typeof e === "string" ? e : e instanceof Error ? e.message : String(e);
}

/** Run `op` as the UI's `busy` operation; its error lands in `voiceLibraryUi.error` and is rethrown. */
async function track<T>(busy: VoiceLibraryBusy, op: () => Promise<T>): Promise<T> {
    voiceLibraryUi.busy = busy;
    voiceLibraryUi.error = null;
    try {
        return await op();
    } catch (e) {
        voiceLibraryUi.error = message(e);
        throw e;
    } finally {
        if (voiceLibraryUi.busy === busy) voiceLibraryUi.busy = null;
    }
}

/** Re-read the voices list into `voiceLibraryUi`. */
export async function refreshVoices(): Promise<VoiceMeta[]> {
    try {
        const voices = await invoke<VoiceMeta[]>(VL_COMMANDS.list);
        voiceLibraryUi.voices = Array.isArray(voices) ? voices : [];
        voiceLibraryUi.loaded = true;
        return voiceLibraryUi.voices;
    } catch (e) {
        voiceLibraryUi.error = message(e);
        throw e;
    }
}

let host: Promise<UnlistenFn[]> | null = null;
/** The take fetch in flight (a click on Stop and an automatic stop share it). */
let stopping: Promise<VoiceTake> | null = null;
/** A start was asked for and not cancelled (progress may arrive before the start resolves). */
let recordingWanted = false;

/** Subscribe to the library's events and load the list once (idempotent; a no-op outside the desktop app). */
export function ensureVoiceLibraryHost(): void {
    if (host || !isTauri()) return;
    host = Promise.all([
        listen(VL_EVENTS.changed, () => {
            void refreshVoices().catch(() => undefined);
        }),
        listen<RecordProgress>(VL_EVENTS.recordProgress, (e) => {
            if (!recordingWanted) return;
            voiceLibraryUi.progress = e.payload;
            if (e.payload?.stopped) {
                void stopRecording().catch(() => undefined);
            }
        }),
    ]);
    host.catch((e) => {
        voiceLibraryUi.error = message(e);
        host = null;
    });
    void refreshVoices().catch(() => undefined);
}

/** Start recording a take (replacing a take that was waiting: it is discarded once the recording started, so a
 *  start that fails, e.g. an unplugged microphone, keeps it). */
export async function startRecording(request: RecordRequest = {}): Promise<void> {
    const previous = voiceLibraryUi.take;
    await track("starting", async () => {
        voiceLibraryUi.progress = null;
        recordingWanted = true;
        try {
            await invoke(VL_COMMANDS.recordStart, { request });
        } catch (e) {
            recordingWanted = false;
            throw e;
        }
        voiceLibraryUi.recording = recordingWanted;
        if (previous) {
            if (voiceLibraryUi.take?.takeId === previous.takeId) voiceLibraryUi.take = null;
            await invoke(VL_COMMANDS.discardTake, { takeId: previous.takeId }).catch(() => undefined);
        }
    });
}

/** Stop recording; resolves with the processed take (also in `voiceLibraryUi.take`). */
export async function stopRecording(): Promise<VoiceTake> {
    if (stopping) return stopping;
    stopping = track("stopping", async () => {
        try {
            const take = await invoke<VoiceTake>(VL_COMMANDS.recordStop);
            voiceLibraryUi.take = take;
            return take;
        } finally {
            recordingWanted = false;
            voiceLibraryUi.recording = false;
        }
    }).finally(() => {
        stopping = null;
    });
    return stopping;
}

export async function cancelRecording(): Promise<void> {
    recordingWanted = false;
    try {
        await invoke(VL_COMMANDS.recordCancel);
    } finally {
        voiceLibraryUi.recording = false;
        voiceLibraryUi.progress = null;
    }
}

/** Import an audio file as a take (also in `voiceLibraryUi.take`; a take that was waiting is discarded). */
export function importClip(path: string): Promise<VoiceTake> {
    const request: ImportClipRequest = { path };
    const previous = voiceLibraryUi.take;
    return track("importing", async () => {
        const take = await invoke<VoiceTake>(VL_COMMANDS.importClip, { request });
        if (previous && previous.takeId !== take.takeId) {
            await invoke(VL_COMMANDS.discardTake, { takeId: previous.takeId }).catch(() => undefined);
        }
        voiceLibraryUi.take = take;
        return take;
    });
}

/** The take's WAV bytes (preview playback). */
export function readTake(takeId: string): Promise<ArrayBuffer> {
    return invoke<ArrayBuffer>(VL_COMMANDS.readTake, { takeId });
}

/** A saved voice's WAV bytes (preview playback). */
export function readClip(id: string): Promise<ArrayBuffer> {
    return invoke<ArrayBuffer>(VL_COMMANDS.readClip, { id });
}

export async function discardTake(takeId: string): Promise<void> {
    if (voiceLibraryUi.take?.takeId === takeId) voiceLibraryUi.take = null;
    await invoke(VL_COMMANDS.discardTake, { takeId });
}

/** Insert or replace `voice` in the list (sorted like Rust: by name, case-insensitive, then id). */
function upsert(voice: VoiceMeta): void {
    const rest = voiceLibraryUi.voices.filter((v) => v.id !== voice.id);
    rest.push(voice);
    rest.sort((a, b) => a.name.toLowerCase().localeCompare(b.name.toLowerCase()) || a.id.localeCompare(b.id));
    voiceLibraryUi.voices = rest;
}

export function saveTake(request: SaveTakeRequest): Promise<VoiceMeta> {
    return track("saving", async () => {
        const voice = await invoke<VoiceMeta>(VL_COMMANDS.saveTake, { request });
        if (voiceLibraryUi.take?.takeId === request.takeId) voiceLibraryUi.take = null;
        upsert(voice);
        return voice;
    });
}

/** Edit a voice. Replacing its clip with the waiting take (`takeId`) is the recorder's save: it runs as the `saving`
 *  operation (its error lands in `voiceLibraryUi.error`); other edits are not tracked. */
export function updateVoice(request: UpdateVoiceRequest): Promise<VoiceMeta> {
    const run = async () => {
        const voice = await invoke<VoiceMeta>(VL_COMMANDS.update, { request });
        if (request.takeId && voiceLibraryUi.take?.takeId === request.takeId) voiceLibraryUi.take = null;
        upsert(voice);
        return voice;
    };
    return request.takeId ? track("saving", run) : run();
}

export async function deleteVoice(id: string): Promise<void> {
    await invoke(VL_COMMANDS.delete, { id });
    voiceLibraryUi.voices = voiceLibraryUi.voices.filter((v) => v.id !== id);
}

/** Tests only: forget the event subscriptions and in-flight state. */
export function __resetVoiceLibraryHostForTests(): void {
    host = null;
    stopping = null;
    recordingWanted = false;
    Object.assign(voiceLibraryUi, { voices: [], loaded: false, recording: false, progress: null, error: null, take: null, busy: null, owner: null });
}
