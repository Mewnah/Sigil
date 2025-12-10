import { invoke } from "@tauri-apps/api/tauri";
import { listen, UnlistenFn } from "@tauri-apps/api/event";
import { ISTTReceiver, ISpeechRecognitionService } from "../types";
import { STT_State } from "../schema";

export class STT_WhisperService implements ISpeechRecognitionService {
    private unlistenProgress?: UnlistenFn;
    private unlistenPartial?: UnlistenFn;
    private isRecording = false;
    private accumulatedText = "";

    constructor(private readonly receiver: ISTTReceiver) { }

    async start(params: STT_State) {
        try {
            console.log("[Whisper] Starting service...");
            this.receiver.onStart();
            this.accumulatedText = ""; // Reset on start

            // Listen for download progress
            this.unlistenProgress = await listen("whisper:download_progress", (event) => {
                const payload = event.payload as { file: string; progress: number };
                console.log(`[Whisper] Downloading ${payload.file}: ${payload.progress.toFixed(1)}%`);
                this.receiver.onInterim(`Downloading ${payload.file}: ${payload.progress.toFixed(0)}%...`);
            });

            console.log("[Whisper] Ensuring dependencies...");
            await invoke("plugin:whisper|ensure_dependencies");

            if (this.unlistenProgress) {
                this.unlistenProgress();
                this.unlistenProgress = undefined;
            }

            // IMPORTANT: Set up partial result listener BEFORE starting recording
            // to ensure we don't miss any events from early chunks
            console.log("[Whisper] Setting up partial result listener...");
            this.unlistenPartial = await listen("whisper:partial_result", (event) => {
                let text = event.payload as string;
                // console.log("[Whisper] Partial result received:", text);

                // Filter out blank audio tokens and other non-speech sounds (e.g. [CHIRPING], (water splashing))
                text = text.replace(/(\[[^\]]+\]|\([^\)]+\))/g, "").trim();

                if (text) {
                    // console.log("[Whisper] Emitting final segment:", text);
                    // Emit as final immediately since Whisper processes distinct chunks
                    this.receiver.onFinal(text);
                } else {
                    // console.log("[Whisper] Skipping empty/blank segment");
                }
            });

            console.log("[Whisper] Starting recording...");
            await invoke("plugin:whisper|start_recording", {
                vadEnabled: params.whisper.vadEnabled,
                silenceThresholdDb: parseFloat(params.whisper.silenceThresholdDb),
                silenceDurationMs: parseInt(params.whisper.silenceDurationMs),
                minChunkDurationMs: parseInt(params.whisper.minChunkDurationMs),
                captureLocal: true,
            });
            this.isRecording = true;
            console.log("[Whisper] Recording started - real-time transcription active");
        } catch (error) {
            console.error("[Whisper] Error starting:", error);
            this.isRecording = false;
            if (this.unlistenPartial) {
                this.unlistenPartial();
                this.unlistenPartial = undefined;
            }
            this.receiver.onStop(String(error));
        }
    }

    async stop() {
        if (!this.isRecording) {
            console.warn("[Whisper] Stop called but not recording (ignoring duplicate call)");
            return; // Silently ignore duplicate calls
        }

        // Set to false immediately to prevent re-entrant calls
        this.isRecording = false;

        // Clean up partial result listener
        if (this.unlistenPartial) {
            this.unlistenPartial();
            this.unlistenPartial = undefined;
        }

        try {
            console.log("[Whisper] Stopping recording...");
            await invoke<string>("plugin:whisper|stop_recording");

            // No need to emit final accumulated text as we emit segments in real-time

        } catch (error) {
            console.error("[Whisper] Error stopping:", error);
        } finally {
            this.receiver.onStop();
        }
    }

    dispose() {
        console.log("[Whisper] Disposing service");
        this.isRecording = false;
        this.accumulatedText = "";

        if (this.unlistenProgress) {
            this.unlistenProgress();
            this.unlistenProgress = undefined;
        }

        if (this.unlistenPartial) {
            this.unlistenPartial();
            this.unlistenPartial = undefined;
        }
    }
}
