import { ISTTReceiver, ISpeechRecognitionService, RustAudioChunkPayload } from "../types";
import { STT_State } from "../schema";
import { convertFileSrc, invoke } from "@tauri-apps/api/core";
import { UnlistenFn } from "@tauri-apps/api/event";
import { appDataDir, join } from "@tauri-apps/api/path";
import { devLog } from "@/utils/devLog";
import { resolveRustInputDeviceForInvoke } from "@/utils/rustAudioInput";
import { isAbortError, listenWithAbort, raceWithAbort } from "../abort";
import { resampleLinear, toMonoFloat32 } from "../audioConvert";
import { SttDownloadProgressPayload } from "../downloadProgress";
import { downloadStatus } from "../statusMessages";
import i18n from "i18next";
import { SttStartLifecycle } from "../localEngineLifecycle";
import { loadVoskModel, VoskLoadedModel, VoskRecognizer } from "../loadVoskModel";
import { errorText, shortDetail } from "../rustCapturePolling";

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

/**
 * Rate the recognizer runs at: `start_audio_capture({ sampleRate: 16000 })` makes the capture hub deliver 16 kHz mono
 * (rubato, band-limited) on the shared device stream, so Vosk sees the same raw microphone as every other engine.
 */
export const VOSK_SAMPLE_RATE = 16_000;

/**
 * Vosk recognizer events → transcript calls. Vosk reports the current partial after every audio block, so an
 * unchanged partial is published once instead of ~50 times a second (captions, interim translation and the VRChat
 * typing indicator all follow interims). A result without words abandons the open partial instead of leaving it
 * on screen. Pure.
 */
export class VoskTranscript {
    #partial = "";

    constructor(private readonly sink: { interim(text: string): void; final(text: string): void }) {}

    partial(text: unknown): void {
        const t = String(text ?? "").trim();
        if (!t || t === this.#partial) return;
        this.#partial = t;
        this.sink.interim(t);
    }

    result(text: unknown): void {
        const t = String(text ?? "").trim();
        const open = this.#partial !== "";
        this.#partial = "";
        if (t) this.sink.final(t);
        else if (open) this.sink.interim("");
    }
}

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

    #createRecognizer(model: VoskLoadedModel): VoskRecognizer {
        const recognizer = new model.KaldiRecognizer(VOSK_SAMPLE_RATE);
        const transcript = new VoskTranscript({
            interim: (text) => {
                if (text) devLog("[Vosk] Partial result:", text);
                this.receiver.onInterim(text);
            },
            final: (text) => {
                devLog("[Vosk] Final result:", text);
                this.receiver.onFinal(text);
            },
        });
        recognizer.on("result", (msg: unknown) => {
            transcript.result((msg as { result?: { text?: string } }).result?.text);
        });
        recognizer.on("partialresult", (msg: unknown) => {
            transcript.partial((msg as { result?: { partial?: string } }).result?.partial);
        });
        return recognizer;
    }

    /** One `audio:chunk`: downmix, and resample when the payload is not at the recognizer's rate. */
    #accept(payload: RustAudioChunkPayload): void {
        const recognizer = this.recognizer;
        if (!this.isRunning || !recognizer) return;
        const { samples, sample_rate, channels } = payload;
        if (!(sample_rate > 0) || !samples?.length) return;
        const mono = toMonoFloat32(samples, channels);
        recognizer.acceptWaveformFloat(resampleLinear(mono, sample_rate, VOSK_SAMPLE_RATE), VOSK_SAMPLE_RATE);
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
            this.recognizer = this.#createRecognizer(this.model);

            const deviceName = resolveRustInputDeviceForInvoke(params.vosk.device);
            devLog("[Vosk] Setting up Rust audio capture...");

            this.unlistenAudio = await listenWithAbort<RustAudioChunkPayload>(
                "audio:chunk",
                (event) => this.#accept(event.payload),
                this.startAbortSignal,
            );

            if (this.#abortStart()) return;

            devLog("[Vosk] Starting Rust audio capture with device:", deviceName ?? "default");

            this.isRunning = true;

            // The capture hub serves 16 kHz mono from the shared device stream (mute-gated like every STT consumer).
            await raceWithAbort(
                invoke("plugin:audio|start_audio_capture", {
                    deviceName,
                    sampleRate: VOSK_SAMPLE_RATE,
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
            this.lifecycle.notifyStopOnce(i18n.t("stt.error_engine_start", { engine: "Vosk", detail: shortDetail(errorText(error)) }));
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
