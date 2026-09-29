import { ISTTReceiver, ISpeechRecognitionService, RustAudioChunkPayload } from "../types";
import { STT_State } from "../schema";
import { convertFileSrc, invoke } from "@tauri-apps/api/core";
import { UnlistenFn } from "@tauri-apps/api/event";
import { appDataDir, join } from "@tauri-apps/api/path";
import { devLog } from "@/utils/devLog";
import { resolveRustInputDeviceForInvoke } from "@/utils/rustAudioInput";
import { isAbortError, listenWithAbort, raceWithAbort } from "../abort";
import { toMonoFloat32 } from "../audioConvert";
import { SttDownloadProgressPayload } from "../downloadProgress";
import { downloadStatus } from "../statusMessages";
import i18n from "i18next";
import { SttStartLifecycle } from "../localEngineLifecycle";
import { loadVoskModel, VoskLoadedModel, VoskRecognizer } from "../loadVoskModel";

/** Mirrors Rust `VoskModel` for `get_vosk_models`. */
export type VoskCatalogModel = {
    id: string;
    name: string;
    language: string;
    size: string;
    url: string;
};

/** Inspector fallback if `invoke` fails (e.g. non-Tauri dev). */
export const VOSK_MODELS_FALLBACK: VoskCatalogModel[] = [
    {
        id: "vosk-model-small-en-us-0.15",
        name: "English (US)",
        language: "en",
        size: "40 MB",
        url: "https://alphacephei.com/vosk/models/vosk-model-small-en-us-0.15.zip",
    },
];

export class STT_VoskService implements ISpeechRecognitionService {
    private model: VoskLoadedModel | null = null;
    private recognizer: VoskRecognizer | null = null;
    private isRunning = false;
    private unlistenAudio?: UnlistenFn;
    /** `vosk:download_progress` while the model download runs (start only). */
    private unlistenProgress?: UnlistenFn;
    private readonly lifecycle: SttStartLifecycle;
    startAbortSignal: AbortSignal | null = null;

    constructor(private readonly receiver: ISTTReceiver) {
        this.lifecycle = new SttStartLifecycle((error) => this.receiver.onStop(error));
    }

    #ensureRecognizer(sampleRate: number) {
        if (this.recognizer || !this.model) return;
        const Rec = this.model.KaldiRecognizer;
        this.recognizer = new Rec(sampleRate);

        this.recognizer.on("result", (msg: unknown) => {
            const message = msg as { result?: { text?: string } };
            const text = message.result?.text || "";
            if (text.trim()) {
                devLog("[Vosk] Final result:", text);
                this.receiver.onFinal(text);
            }
        });

        this.recognizer.on("partialresult", (msg: unknown) => {
            const message = msg as { result?: { partial?: string } };
            const partial = message.result?.partial || "";
            if (partial.trim()) {
                devLog("[Vosk] Partial result:", partial);
                this.receiver.onInterim(partial);
            }
        });
    }

    #teardownModelAndRecognizer() {
        if (this.recognizer) {
            this.recognizer.remove();
            this.recognizer = null;
        }
        if (this.model) {
            this.model.terminate();
            this.model = null;
        }
    }

    #clearProgressListener(): void {
        if (this.unlistenProgress) {
            this.unlistenProgress();
            this.unlistenProgress = undefined;
        }
    }

    /** Removes the audio and download-progress listeners (every start exit and stop). */
    #clearAudioListener(): void {
        this.#clearProgressListener();
        if (this.unlistenAudio) {
            this.unlistenAudio();
            this.unlistenAudio = undefined;
        }
    }

    #abortStart(notify = true): boolean {
        if (!this.lifecycle.checkAborted(notify)) return false;
        this.#clearAudioListener();
        this.#teardownModelAndRecognizer();
        return true;
    }

    async start(params: STT_State) {
        this.lifecycle.beginStart();
        try {
            devLog("[Vosk] Starting service...");
            this.receiver.onStatus(i18n.t("stt.status_vosk_loading"));

            const modelId = params.vosk.model || "vosk-model-small-en-us-0.15";
            const customUrl = params.vosk.modelUrl?.trim() || null;
            const localZip =
                params.vosk.useLocalZip && params.vosk.localZipPath?.trim()
                    ? params.vosk.localZipPath.trim()
                    : null;

            this.receiver.onStatus(i18n.t("stt.status_vosk_model", { model: modelId }));

            // Forward download progress as UI status: each update also re-arms the orchestrator's connecting
            // window, so a slow first download is not cut off while it is still advancing.
            this.unlistenProgress = await listenWithAbort<SttDownloadProgressPayload>(
                "vosk:download_progress",
                (event) => {
                    if (this.lifecycle.aborted) return;
                    this.receiver.onStatus(...downloadStatus(event.payload));
                },
                this.startAbortSignal,
            );

            if (this.#abortStart()) return;

            const relPath = await raceWithAbort(
                invoke<string>("plugin:vosk-stt|download_vosk_model", {
                    modelId,
                    urlOverride: customUrl,
                    localZipPath: localZip && localZip.length > 0 ? localZip : null,
                }),
                this.startAbortSignal,
            );
            this.#clearProgressListener();

            if (this.#abortStart()) return;

            const modelPath = await raceWithAbort(join(await appDataDir(), relPath), this.startAbortSignal);
            const modelUrl = convertFileSrc(modelPath);

            devLog(`[Vosk] Loading model from ${modelPath}`);
            this.receiver.onStatus(i18n.t("stt.status_vosk_loading_model", { model: modelId }));

            this.model = await loadVoskModel(modelUrl, this.startAbortSignal);

            if (this.#abortStart()) return;

            devLog("[Vosk] Model loaded successfully");

            const deviceName = resolveRustInputDeviceForInvoke(params.vosk.device);
            devLog("[Vosk] Setting up Rust audio capture...");

            this.unlistenAudio = await listenWithAbort<RustAudioChunkPayload>(
                "audio:chunk",
                (event) => {
                    if (!this.isRunning || !this.model) return;

                    // Device rate/channels: Vosk resamples internally, so only downmix here.
                    const { samples, sample_rate, channels } = event.payload;
                    this.#ensureRecognizer(sample_rate);
                    if (!this.recognizer) return;

                    const floats = toMonoFloat32(samples, channels);
                    this.recognizer.acceptWaveformFloat(floats, sample_rate);
                },
                this.startAbortSignal,
            );

            if (this.#abortStart()) return;

            devLog("[Vosk] Starting Rust audio capture with device:", deviceName ?? "default");

            this.isRunning = true;

            await raceWithAbort(
                invoke("plugin:audio|start_audio_capture", {
                    deviceName,
                    sampleRate: 16000,
                }),
                this.startAbortSignal,
            );

            if (this.lifecycle.aborted) {
                this.isRunning = false;
                try {
                    await invoke("plugin:audio|stop_audio_capture");
                } catch {
                    /* ignore */
                }
                this.#clearAudioListener();
                this.#teardownModelAndRecognizer();
                this.lifecycle.notifyStopOnce();
                return;
            }

            this.receiver.onStatus("");
            this.receiver.onStart();
            devLog("[Vosk] Recording started via Rust audio capture");
        } catch (error) {
            if (isAbortError(error) || this.lifecycle.aborted) {
                this.isRunning = false;
                this.#clearAudioListener();
                this.#teardownModelAndRecognizer();
                return;
            }
            console.error("[Vosk] Error starting:", error);
            this.receiver.onStatus("");
            this.isRunning = false;
            this.#clearAudioListener();
            this.#teardownModelAndRecognizer();
            this.lifecycle.notifyStopOnce(String(error));
        } finally {
            this.lifecycle.finishStart();
        }
    }

    async stop() {
        this.lifecycle.abort();
        const hadSession =
            this.isRunning || this.lifecycle.starting || !!this.unlistenAudio || !!this.model || !!this.recognizer;

        this.isRunning = false;

        try {
            await invoke("plugin:vosk-stt|cancel_vosk_model_download");
        } catch {
            /* ignore */
        }

        try {
            await invoke("plugin:audio|stop_audio_capture");
        } catch {
            /* ignore */
        }

        this.#clearAudioListener();
        this.#teardownModelAndRecognizer();

        if (hadSession) {
            this.lifecycle.notifyStopOnce();
        }
        devLog("[Vosk] Stopped");
    }

    dispose() {
        void this.stop();
    }
}
