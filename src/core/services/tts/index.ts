import { IServiceInterface, ServiceNetworkState, TextEventType } from "@/types";
import { WordReplacementsCache, buildWordReplacementsCache, serviceSubscibeToInput, serviceSubscibeToSource } from "@/utils";
import i18n from "i18next";
import { toast } from "react-toastify";
import { proxy } from "valtio";
import { subscribeKey } from "valtio/utils";
import { pushSystemLog } from "../systemLog";
import { ErrorThrottle } from "./errorThrottle";
import { TTS_Backends } from "./schema";
import { TTS_AzureService } from "./services/azure";
import { TTS_NativeService } from "./services/native";
import { TTS_WindowsService } from "./services/windows";
import { TTS_UberduckService } from "./services/uberduck";
import { TTS_VoicevoxService } from "./services/voicevox";
import { TTS_KokoroService } from "./services/kokoro";
import { TTS_MeloService } from "./services/melo";
import { TTS_ChatterboxService } from "./services/chatterbox";
import { TTS_FishSpeechService } from "./services/fishspeech";
import { TTS_NeuralOnnxService } from "./services/neuralOnnx";
import { TTS_NeuralSidecarService } from "./services/neuralSidecar";
import {
  ITTSReceiver,
  ITTSService,
  ITTSServiceConstructor,
  TTS_OPTIONS_MISSING,
  errorMessage,
} from "./types";

export type TTSBackendMap = { [k in TTS_Backends]: ITTSServiceConstructor };

export const TTS_BACKENDS: TTSBackendMap = {
  [TTS_Backends.native]: TTS_NativeService,
  [TTS_Backends.windows]: TTS_WindowsService,
  [TTS_Backends.azure]: TTS_AzureService,
  [TTS_Backends.uberduck]: TTS_UberduckService,
  [TTS_Backends.voicevox]: TTS_VoicevoxService,
  [TTS_Backends.kokoro]: TTS_KokoroService,
  [TTS_Backends.melo]: TTS_MeloService,
  [TTS_Backends.chatterbox]: TTS_ChatterboxService,
  [TTS_Backends.fishSpeech]: TTS_FishSpeechService,
  [TTS_Backends.neuralOnnx]: TTS_NeuralOnnxService,
  [TTS_Backends.neuralSidecar]: TTS_NeuralSidecarService,
};

function isOptionsMissing(error: string): boolean {
  return error === TTS_OPTIONS_MISSING || error.includes(TTS_OPTIONS_MISSING);
}

/**
 * Text-to-speech host service: subscribes to the configured text source and speaks each final
 * through the selected backend.
 *
 * Lifecycle: `start()` creates one backend instance and sets the status to connecting; the backend
 * reports `onStart` (connected), `onStop(error?)` (disconnected) or `onError` (one utterance failed).
 * Each instance gets its own receiver, so callbacks from an instance that was already stopped or
 * replaced (a slow start finishing after Stop, for example) can no longer change the status or the
 * new instance.
 */
class Service_TTS implements IServiceInterface, ITTSReceiver {
  #serviceInstance?: ITTSService;
  /** Identity of the running instance's receiver; replaced on every start/stop. */
  #activeToken?: object;
  #errorThrottle = new ErrorThrottle();

  serviceState = proxy({
    status: ServiceNetworkState.disconnected,
    error: ""
  });

  private eventDisposers: (() => void)[] = [];

  #_wordReplacementsCache!: WordReplacementsCache;

  constructor(private readonly backends: TTSBackendMap = TTS_BACKENDS) {}

  get data() {
    return window.ApiServer.state.services.tts.data;
  }

  updateReplacementsCache() {
    this.#_wordReplacementsCache = buildWordReplacementsCache(this.data.replaceWords, this.data.replaceWordsIgnoreCase);
  }

  runReplacements(value: string) {
    if (this.#_wordReplacementsCache.isEmpty)
      return value;
    return value.replace(this.#_wordReplacementsCache.regexp, v => {
      const _v = this.data.replaceWordsIgnoreCase ? v.toLowerCase() : v;
      return this.#_wordReplacementsCache.map[_v];
    }).replace(/[<>]/gi, ""); // clear ssml tags
  }

  async init() {
    this.updateReplacementsCache();
    this.eventDisposers.push(subscribeKey(this.data, "replaceWords", () => this.updateReplacementsCache()));
    this.eventDisposers.push(subscribeKey(this.data, "replaceWordsIgnoreCase", () => this.updateReplacementsCache()));
    this.eventDisposers.push(serviceSubscibeToSource(this.data, "source", data => {
      if (data?.type === TextEventType.final)
        this.play(data.value);
    }));

    // Typed text arrives on text.textfield. When the source already covers it ("text" or
    // "text.textfield"), a second subscription would speak every typed line twice.
    this.eventDisposers.push(serviceSubscibeToInput(this.data, "inputField", data => {
      if (data?.type === TextEventType.final)
        this.play(data.value);
    }, "source"));

    if (this.data.autoStart)
      this.start();
    const unsubStream = window.ApiShared.pubsub.subscribe("stream.on_ended", () => {
      if (this.data.stopWithStream && this.serviceState.status === ServiceNetworkState.connected) {
        this.stop();
      }
    });
    this.eventDisposers.push(() => window.ApiShared.pubsub.unsubscribe(unsubStream));
  }

  /**
   * Stop the running backend, including one still connecting, and return to disconnected. The
   * backend is detached first, so its own `onStop()` (and anything a pending start reports later) is
   * ignored; the status is set here.
   */
  stop(): void {
    const instance = this.#serviceInstance;
    this.#detach();
    if (instance) {
      try {
        instance.stop();
      } catch (error) {
        console.error("[TTS] stop failed", error);
      }
      try {
        instance.dispose();
      } catch (error) {
        console.error("[TTS] dispose failed", error);
      }
    }
    this.#setStatus(ServiceNetworkState.disconnected);
  }

  #detach() {
    this.#activeToken = undefined;
    this.#serviceInstance = undefined;
  }

  #setStatus(value: ServiceNetworkState) {
    this.serviceState.status = value;
  }

  onStart(): void {
    this.#setStatus(ServiceNetworkState.connected);
  }

  onStop(error?: string | undefined): void {
    if (error) {
      const missingOpts = isOptionsMissing(error);
      const message = missingOpts
        ? i18n.t("toasts.tts_setup_required")
        : i18n.t("toasts.tts_stopped_error", { error });
      toast(message, {
        type: missingOpts ? "warning" : "error",
        autoClose: missingOpts ? 12_000 : 10_000,
      });
      this.serviceState.error = error;
    }
    const instance = this.#serviceInstance;
    this.#detach();
    // A backend that stops on its own (failed start, fatal error) is still disposed. Detached first,
    // so a dispose that calls stop() -> onStop() does not recurse.
    if (instance) {
      try {
        instance.dispose();
      } catch (e) {
        console.error("[TTS] dispose failed", e);
      }
    }
    this.#setStatus(ServiceNetworkState.disconnected);
  }

  /** A single utterance failed; the backend keeps running. */
  onError(error: string): void {
    if (!error) return;
    pushSystemLog("TTS", error, "error");
    if (this.#errorThrottle.shouldShow(error)) {
      toast(i18n.t("toasts.tts_play_failed", { error }), { type: "error", autoClose: 8_000 });
    }
  }

  play(value: string) {
    const patchedValue = this.runReplacements(value);
    if (!patchedValue?.trim())
      return;
    const instance = this.#serviceInstance;
    if (!instance || this.serviceState.status !== ServiceNetworkState.connected)
      return;
    try {
      const result = instance.play(patchedValue);
      if (result && typeof (result as Promise<void>).catch === "function")
        (result as Promise<void>).catch(e => this.#receiverCall(instance, r => r.onError(errorMessage(e))));
    } catch (e) {
      this.onError(errorMessage(e));
    }
  }

  start() {
    this.stop();
    this.serviceState.error = "";
    this.#errorThrottle.reset();

    const backend = this.data.backend;
    const Backend = this.backends[backend];
    if (!Backend)
      return;

    const token = {};
    const receiver: ITTSReceiver = {
      onStart: () => { if (this.#activeToken === token) this.onStart(); },
      onStop: (error) => { if (this.#activeToken === token) this.onStop(error); },
      onError: (error) => { if (this.#activeToken === token) this.onError(error); },
    };

    let instance: ITTSService;
    try {
      instance = new Backend(receiver);
    } catch (e) {
      this.onStop(errorMessage(e));
      return;
    }
    this.#activeToken = token;
    this.#serviceInstance = instance;
    this.#setStatus(ServiceNetworkState.connecting);
    try {
      const result = instance.start(this.data);
      if (result && typeof (result as Promise<void>).catch === "function")
        (result as Promise<void>).catch(e => receiver.onStop(errorMessage(e)));
    } catch (e) {
      receiver.onStop(errorMessage(e));
    }
  }

  /** Run `fn` against the service's receiver only while `instance` is still the active backend. */
  #receiverCall(instance: ITTSService, fn: (r: ITTSReceiver) => void) {
    if (this.#serviceInstance === instance) fn(this);
  }

  dispose() {
    this.stop();
    this.eventDisposers.forEach(d => d());
    this.eventDisposers = [];
  }
}

export default Service_TTS;
