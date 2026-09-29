import { invoke } from "@tauri-apps/api/core";
import { emit, UnlistenFn } from "@tauri-apps/api/event";
import i18n from "i18next";
import { ISTTReceiver, ISpeechRecognitionService } from "../types";
import { STT_State } from "../schema";
import { devLog } from "@/utils/devLog";
import { pushSystemLog } from "@/core/services/systemLog";
import { resolveRustInputDeviceForInvoke } from "@/utils/rustAudioInput";
import { isAbortError, listenWithAbort, raceWithAbort } from "../abort";
import { SttDownloadProgressPayload } from "../downloadProgress";
import { SttStartLifecycle } from "../localEngineLifecycle";
import { TRANSIENT_WARNING_MS } from "../statusChannel";
import { downloadStatus } from "../statusMessages";
import { buildVadConfig, buildWhisperDecodeOptions, stripWhisperNoiseMarkers } from "../sttOptions";
import type { RustWhisperCaptureError, RustWhisperUtteranceDropped } from "@/types";

/** Rust `whisper:utterance_dropped`: the model ran slower than real time and skipped the oldest waiting utterance. */
type UtteranceDroppedPayload = RustWhisperUtteranceDropped;
/** Rust `whisper:capture_error`: the input device failed; the recording is over. */
type CaptureErrorPayload = RustWhisperCaptureError;

/** `whisper:load_status` lines Rust emits (English) → translated status keys. */
const LOAD_STATUS_KEYS: Record<string, string> = {
    "Copying local Whisper model...": "stt.status_whisper_copying",
    "Loading Whisper model into memory...": "stt.status_whisper_loading",
};

type ClearWhisperListenersOpts = {
    setup?: boolean;
    session?: boolean;
    dismissDownload?: boolean;
};

export class STT_WhisperService implements ISpeechRecognitionService {
    /** Model download / load listeners (start phase only). */
    private setupUnlisteners: UnlistenFn[] = [];
    /** Transcript, dropped-utterance and capture-error listeners (recording phase). */
    private sessionUnlisteners: UnlistenFn[] = [];
    private isRecording = false;
    private readonly lifecycle: SttStartLifecycle;
    startAbortSignal: AbortSignal | null = null;

    constructor(private readonly receiver: ISTTReceiver) {
        this.lifecycle = new SttStartLifecycle((error) => this.receiver.onStop(error));
    }

    #clearWhisperListeners(opts: ClearWhisperListenersOpts = {}): void {
        const { setup = true, session = true, dismissDownload = false } = opts;
        if (setup) {
            this.setupUnlisteners.forEach((u) => u());
            this.setupUnlisteners = [];
        }
        if (session) {
            this.sessionUnlisteners.forEach((u) => u());
            this.sessionUnlisteners = [];
        }
        if (dismissDownload) {
            void emit("whisper:download_dismiss", {});
        }
    }

    /** Tear down listeners and optionally notify stop; returns true when aborted. */
    #abortStart(notify = true): boolean {
        if (!this.lifecycle.checkAborted(notify)) return false;
        this.#clearWhisperListeners({ dismissDownload: true });
        return true;
    }

    async start(params: STT_State) {
        this.lifecycle.beginStart();
        try {
            devLog("[Whisper] Starting service...");

            // Download / local-model copy progress (percent, or < 0 when the size is unknown): UI status only.
            this.setupUnlisteners.push(await listenWithAbort<SttDownloadProgressPayload>(
                "whisper:download_progress",
                (event) => {
                    if (this.lifecycle.aborted) return;
                    this.receiver.onStatus(...downloadStatus(event.payload));
                },
                this.startAbortSignal,
            ));

            this.setupUnlisteners.push(await listenWithAbort<string>(
                "whisper:load_status",
                (event) => {
                    if (this.lifecycle.aborted) return;
                    const status = String(event.payload ?? "").trim();
                    const key = LOAD_STATUS_KEYS[status];
                    this.receiver.onStatus(key ? i18n.t(key) : status);
                },
                this.startAbortSignal,
            ));

            if (this.#abortStart()) return;

            const localPath =
                params.whisper.useLocalModel && params.whisper.localModelPath?.trim()
                    ? params.whisper.localModelPath.trim()
                    : null;

            devLog("[Whisper] Ensuring dependencies...", { model: params.whisper.model, language: params.whisper.language });
            await raceWithAbort(
                invoke("plugin:whisper|ensure_dependencies", {
                    model: params.whisper.model,
                    language: params.whisper.language,
                    localModelPath: localPath,
                }),
                this.startAbortSignal,
            );

            if (this.#abortStart()) return;

            // Download phase complete: the recording listeners replace the setup ones. Do not dismiss the download
            // row here: Rust already emitted `whisper:download_complete`, which the inspector shows briefly.
            this.#clearWhisperListeners({ session: false });
            this.receiver.onStatus("");

            devLog("[Whisper] Setting up segment transcript listener...");
            this.sessionUnlisteners.push(await listenWithAbort<string>(
                "whisper:segment_transcript",
                (event) => {
                    // Rust already filtered hallucinations; strip leftover markers like `[MUSIC]` / `(laughs)`.
                    const text = stripWhisperNoiseMarkers(String(event.payload ?? ""));
                    if (text) {
                        this.receiver.onFinal(text);
                    }
                },
                this.startAbortSignal,
            ));

            this.sessionUnlisteners.push(await listenWithAbort<UtteranceDroppedPayload>(
                "whisper:utterance_dropped",
                (event) => {
                    const seconds = Math.max(1, Math.round((event.payload?.audio_ms ?? 0) / 1000));
                    const line = i18n.t("stt.status_whisper_dropped", { seconds });
                    pushSystemLog("STT", line, "warning");
                    // Transient: clears itself once Whisper keeps up again instead of staying all session.
                    this.receiver.onStatus(line, { level: "warning", expireMs: TRANSIENT_WARNING_MS });
                },
                this.startAbortSignal,
            ));

            this.sessionUnlisteners.push(await listenWithAbort<CaptureErrorPayload>(
                "whisper:capture_error",
                (event) => {
                    if (!this.isRecording) return;
                    void this.#failRecording(i18n.t("stt.error_capture", { detail: event.payload?.message ?? "" }));
                },
                this.startAbortSignal,
            ));

            if (this.#abortStart()) return;

            devLog("[Whisper] Starting recording...");
            const deviceName = resolveRustInputDeviceForInvoke(params.whisper.device);
            await raceWithAbort(
                invoke("plugin:whisper|start_recording", {
                    deviceName,
                    captureLocal: true,
                    vad: buildVadConfig(params.vad),
                    decode: buildWhisperDecodeOptions(params.whisper),
                }),
                this.startAbortSignal,
            );

            if (this.lifecycle.aborted) {
                this.isRecording = false;
                try {
                    await invoke<string>("plugin:whisper|stop_recording");
                } catch {
                    /* ignore */
                }
                this.#clearWhisperListeners({ dismissDownload: true });
                this.lifecycle.notifyStopOnce();
                return;
            }

            this.isRecording = true;
            this.receiver.onStart();
            devLog("[Whisper] Recording started - real-time transcription active");
        } catch (error) {
            this.isRecording = false;
            this.#clearWhisperListeners({ dismissDownload: true });
            if (isAbortError(error) || this.lifecycle.aborted) {
                return;
            }
            console.error("[Whisper] Error starting:", error);
            this.receiver.onStatus("");
            this.lifecycle.notifyStopOnce(String(error));
        } finally {
            this.lifecycle.finishStart();
        }
    }

    /** The device failed mid-recording (Rust already ended the recording): release everything and report it. */
    async #failRecording(message: string): Promise<void> {
        this.isRecording = false;
        this.#clearWhisperListeners();
        try {
            await invoke<string>("plugin:whisper|stop_recording");
        } catch {
            /* ignore */
        }
        this.lifecycle.notifyStopOnce(message);
    }

    async stop() {
        this.lifecycle.abort();
        const hadWork = this.lifecycle.starting || this.isRecording;
        this.isRecording = false;

        this.#clearWhisperListeners({ dismissDownload: true });

        try {
            await invoke("plugin:whisper|cancel_whisper_download");
        } catch {
            /* ignore */
        }

        try {
            await invoke<string>("plugin:whisper|stop_recording");
        } catch {
            /* ignore — may not have been recording */
        }

        if (hadWork) {
            this.lifecycle.notifyStopOnce();
        }
        devLog("[Whisper] Stopped");
    }

    dispose() {
        devLog("[Whisper] Disposing service");
        void this.stop();
    }
}
