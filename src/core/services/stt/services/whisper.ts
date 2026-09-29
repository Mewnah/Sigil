import { invoke } from "@tauri-apps/api/core";
import { emit, UnlistenFn } from "@tauri-apps/api/event";
import { ISTTReceiver, ISpeechRecognitionService } from "../types";
import { STT_State } from "../schema";
import { devLog } from "@/utils/devLog";
import { resolveRustInputDeviceForInvoke } from "@/utils/rustAudioInput";
import { isAbortError, listenWithAbort, raceWithAbort } from "../abort";
import { formatDownloadProgress, SttDownloadProgressPayload } from "../downloadProgress";
import { SttStartLifecycle } from "../localEngineLifecycle";

/** Strip Whisper non-speech markers like `[CHIRPING]` / `(water splashing)` from segment text. */
const WHISPER_NOISE_TOKEN_RE = /(\[[^\]]+\]|\([^\)]+\))/g;

type ClearWhisperListenersOpts = {
    progress?: boolean;
    loadStatus?: boolean;
    partial?: boolean;
    dismissDownload?: boolean;
};

export class STT_WhisperService implements ISpeechRecognitionService {
    private unlistenProgress?: UnlistenFn;
    private unlistenLoadStatus?: UnlistenFn;
    private unlistenPartial?: UnlistenFn;
    private isRecording = false;
    private readonly lifecycle: SttStartLifecycle;
    startAbortSignal: AbortSignal | null = null;

    constructor(private readonly receiver: ISTTReceiver) {
        this.lifecycle = new SttStartLifecycle((error) => this.receiver.onStop(error));
    }

    #clearWhisperListeners(opts: ClearWhisperListenersOpts = {}): void {
        const {
            progress = true,
            loadStatus = true,
            partial = true,
            dismissDownload = false,
        } = opts;

        if (progress && this.unlistenProgress) {
            this.unlistenProgress();
            this.unlistenProgress = undefined;
        }
        if (loadStatus && this.unlistenLoadStatus) {
            this.unlistenLoadStatus();
            this.unlistenLoadStatus = undefined;
        }
        if (partial && this.unlistenPartial) {
            this.unlistenPartial();
            this.unlistenPartial = undefined;
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

            // Download / local-model copy progress (percent, or < 0 when the size is unknown).
            this.unlistenProgress = await listenWithAbort<SttDownloadProgressPayload>(
                "whisper:download_progress",
                (event) => {
                    const line = formatDownloadProgress(event.payload);
                    devLog(`[Whisper] ${line}`);
                    this.receiver.onInterim(line);
                },
                this.startAbortSignal,
            );

            this.unlistenLoadStatus = await listenWithAbort<string>(
                "whisper:load_status",
                (event) => {
                    const status = event.payload?.trim() ?? "";
                    if (status) {
                        this.receiver.onInterim(status);
                    }
                },
                this.startAbortSignal,
            );

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

            // Download phase complete — segment listener replaces progress/load listeners. Do not dismiss the
            // download row here: Rust already emitted `whisper:download_complete`, which the inspector shows briefly.
            this.#clearWhisperListeners();
            this.receiver.onInterim("");

            devLog("[Whisper] Setting up segment transcript listener...");
            this.unlistenPartial = await listenWithAbort<string>(
                "whisper:segment_transcript",
                (event) => {
                    const text = String(event.payload ?? "").replace(WHISPER_NOISE_TOKEN_RE, "").trim();

                    if (text) {
                        this.receiver.onFinal(text);
                    }
                },
                this.startAbortSignal,
            );

            if (this.#abortStart()) return;

            devLog("[Whisper] Starting recording...");
            const deviceName = resolveRustInputDeviceForInvoke(params.whisper.device);
            await raceWithAbort(
                invoke("plugin:whisper|start_recording", {
                    deviceName,
                    vadEnabled: params.whisper.vadEnabled,
                    silenceThresholdDb: parseFloat(params.whisper.silenceThresholdDb),
                    silenceDurationMs: parseInt(params.whisper.silenceDurationMs),
                    minChunkDurationMs: parseInt(params.whisper.minChunkDurationMs),
                    captureLocal: true,
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
            if (isAbortError(error) || this.lifecycle.aborted) {
                this.isRecording = false;
                this.#clearWhisperListeners({ dismissDownload: true });
                return;
            }
            console.error("[Whisper] Error starting:", error);
            this.isRecording = false;
            this.#clearWhisperListeners({ dismissDownload: true });
            this.lifecycle.notifyStopOnce(String(error));
        } finally {
            this.lifecycle.finishStart();
        }
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
