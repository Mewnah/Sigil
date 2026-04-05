import { invoke } from "@tauri-apps/api/core";
import { emit, listen, UnlistenFn } from "@tauri-apps/api/event";
import { ISTTReceiver, ISpeechRecognitionService } from "../types";
import { STT_State } from "../schema";
import { devLog } from "@/utils/devLog";

/** Strip Whisper non-speech markers like `[CHIRPING]` / `(water splashing)` from partials. */
const WHISPER_NOISE_TOKEN_RE = /(\[[^\]]+\]|\([^\)]+\))/g;

export class STT_WhisperService implements ISpeechRecognitionService {
    private unlistenProgress?: UnlistenFn;
    private unlistenPartial?: UnlistenFn;
    private isRecording = false;
    private accumulatedText = "";
    private starting = false;
    private startAborted = false;
    private stopNotified = false;

    constructor(private readonly receiver: ISTTReceiver) { }

    #resetStopNotified() {
        this.stopNotified = false;
    }

    #notifyStopOnce(error?: string) {
        if (this.stopNotified) return;
        this.stopNotified = true;
        this.receiver.onStop(error);
    }

    async start(params: STT_State) {
        this.startAborted = false;
        this.starting = true;
        this.#resetStopNotified();
        try {
            devLog("[Whisper] Starting service...");
            this.accumulatedText = "";

            this.unlistenProgress = await listen("whisper:download_progress", (event) => {
                const payload = event.payload as { file: string; progress: number };
                devLog(`[Whisper] Downloading ${payload.file}: ${payload.progress.toFixed(1)}%`);
                this.receiver.onInterim(`Downloading ${payload.file}: ${payload.progress.toFixed(0)}%...`);
            });

            if (this.startAborted) {
                if (this.unlistenProgress) {
                    this.unlistenProgress();
                    this.unlistenProgress = undefined;
                }
                void emit("whisper:download_dismiss", {});
                this.#notifyStopOnce();
                return;
            }

            const localPath =
                params.whisper.useLocalModel && params.whisper.localModelPath?.trim()
                    ? params.whisper.localModelPath.trim()
                    : null;

            devLog("[Whisper] Ensuring dependencies...", { model: params.whisper.model, language: params.whisper.language });
            await invoke("plugin:whisper|ensure_dependencies", {
                model: params.whisper.model,
                language: params.whisper.language,
                localModelPath: localPath,
            });

            if (this.startAborted) {
                if (this.unlistenProgress) {
                    this.unlistenProgress();
                    this.unlistenProgress = undefined;
                }
                void emit("whisper:download_dismiss", {});
                this.#notifyStopOnce();
                return;
            }

            if (this.unlistenProgress) {
                this.unlistenProgress();
                this.unlistenProgress = undefined;
            }
            this.receiver.onInterim("");

            devLog("[Whisper] Setting up partial result listener...");
            this.unlistenPartial = await listen("whisper:partial_result", (event) => {
                let text = event.payload as string;
                text = text.replace(WHISPER_NOISE_TOKEN_RE, "").trim();

                if (text) {
                    this.receiver.onFinal(text);
                }
            });

            if (this.startAborted) {
                if (this.unlistenPartial) {
                    this.unlistenPartial();
                    this.unlistenPartial = undefined;
                }
                void emit("whisper:download_dismiss", {});
                this.#notifyStopOnce();
                return;
            }

            devLog("[Whisper] Starting recording...");
            const deviceName = params.whisper.device || window.ApiServer.state.audioInputDevice || "";
            await invoke("plugin:whisper|start_recording", {
                deviceName: deviceName || null,
                vadEnabled: params.whisper.vadEnabled,
                silenceThresholdDb: parseFloat(params.whisper.silenceThresholdDb),
                silenceDurationMs: parseInt(params.whisper.silenceDurationMs),
                minChunkDurationMs: parseInt(params.whisper.minChunkDurationMs),
                captureLocal: true,
            });

            if (this.startAborted) {
                this.isRecording = false;
                try {
                    await invoke<string>("plugin:whisper|stop_recording");
                } catch {
                    /* ignore */
                }
                if (this.unlistenPartial) {
                    this.unlistenPartial();
                    this.unlistenPartial = undefined;
                }
                void emit("whisper:download_dismiss", {});
                this.#notifyStopOnce();
                return;
            }

            this.isRecording = true;
            this.receiver.onStart();
            devLog("[Whisper] Recording started - real-time transcription active");
        } catch (error) {
            console.error("[Whisper] Error starting:", error);
            this.isRecording = false;
            if (this.unlistenProgress) {
                this.unlistenProgress();
                this.unlistenProgress = undefined;
            }
            void emit("whisper:download_dismiss", {});
            if (this.unlistenPartial) {
                this.unlistenPartial();
                this.unlistenPartial = undefined;
            }
            if (!this.startAborted) {
                this.#notifyStopOnce(String(error));
            }
        } finally {
            this.starting = false;
        }
    }

    async stop() {
        this.startAborted = true;
        const hadWork = this.starting || this.isRecording;
        this.isRecording = false;

        if (this.unlistenProgress) {
            this.unlistenProgress();
            this.unlistenProgress = undefined;
        }
        void emit("whisper:download_dismiss", {});

        if (this.unlistenPartial) {
            this.unlistenPartial();
            this.unlistenPartial = undefined;
        }

        try {
            await invoke<string>("plugin:whisper|stop_recording");
        } catch {
            /* ignore — may not have been recording */
        }

        if (hadWork) {
            this.#notifyStopOnce();
        }
        devLog("[Whisper] Stopped");
    }

    dispose() {
        devLog("[Whisper] Disposing service");
        this.startAborted = true;
        this.isRecording = false;
        this.accumulatedText = "";

        if (this.unlistenProgress) {
            this.unlistenProgress();
            this.unlistenProgress = undefined;
            void emit("whisper:download_dismiss", {});
        }

        if (this.unlistenPartial) {
            this.unlistenPartial();
            this.unlistenPartial = undefined;
        }
    }
}
