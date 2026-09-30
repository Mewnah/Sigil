import { Services } from "@/services-registry";
import { IServiceInterface, ServiceNetworkState, TextEventType } from "@/types";
import { WordReplacementsCache, buildWordReplacementsCache, serviceSubscibeToInput, serviceSubscibeToSource } from "@/utils";
import { ask } from "@tauri-apps/plugin-dialog";
import i18n from "i18next";
import { toast } from "react-toastify";
import { proxy } from "valtio";
import { subscribeKey } from "valtio/utils";
import { i18nReady } from "../stt/statusMessages";
import { pushSystemLog } from "../systemLog";
import { planTtsDesktopDefault } from "./defaults";
import { ErrorThrottle } from "./errorThrottle";
import { TtsPipeline, ttsSinkTarget } from "./pipeline";
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
import { TTS_LocalNeuralService } from "./services/localNeural";
import { TTS_OpenAISpeechService } from "./services/openaiSpeech";
import {
  ITTSReceiver,
  ITTSService,
  ITTSServiceConstructor,
  TTS_OPTIONS_MISSING,
  errorMessage,
  ttsOutputMode,
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
  [TTS_Backends.localNeural]: TTS_LocalNeuralService,
  [TTS_Backends.openaiSpeech]: TTS_OpenAISpeechService,
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
 *
 * Output (P2-G): backends that produce PCM (ITTSService v2) speak through the sentence pipeline
 * (`pipeline.ts`): finals are split into sentences, synthesized a few ahead and played in order
 * through the unified Rust TTS sink with the top-level route (device / voice changer / both).
 * Self-playing backends (Web Speech, SAPI) and backends not migrated yet get whole finals via
 * `play()`. `skip()` / `stopSpeaking()` cancel sentences without stopping the service.
 */
class Service_TTS implements IServiceInterface, ITTSReceiver {
  #serviceInstance?: ITTSService;
  /** Sentence pipeline of the running instance (PCM backends only). */
  #pipeline?: TtsPipeline;
  /** Identity of the running instance's receiver; replaced on every start/stop. */
  #activeToken?: object;
  #errorThrottle = new ErrorThrottle();
  /** The one-time Kokoro offer is due in this launch (`runStartupOffer`). */
  #pendingLocalOffer = false;

  serviceState = proxy({
    status: ServiceNetworkState.disconnected,
    error: "",
    /** Sentences queued, synthesizing or playing in the pipeline (0 for self-playing backends). */
    pending: 0,
    /** A pipeline sentence is audible now. */
    speaking: false,
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
        this.play(data.value, data.lang);
    }));

    // Typed text arrives on text.textfield. When the source already covers it ("text" or
    // "text.textfield"), a second subscription would speak every typed line twice.
    this.eventDisposers.push(serviceSubscibeToInput(this.data, "inputField", data => {
      if (data?.type === TextEventType.final)
        this.play(data.value, data.lang);
    }, "source"));

    this.#applyDesktopDefault();

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
   * P2 desktop default (`defaults.ts`): a new desktop install starts on the in-process Kokoro voice
   * (`localNeural`); an existing profile on Web Speech / SAPI is offered it once by {@link runStartupOffer}.
   */
  #applyDesktopDefault(): void {
    let plan: ReturnType<typeof planTtsDesktopDefault>;
    try {
      plan = planTtsDesktopDefault({
        isDesktop: window.Config?.isApp?.() === true,
        alreadyMigrated: this.data.desktopDefaultMigrated,
        freshInstall: !window.ApiServer.state.onboardingComplete,
        backend: this.data.backend,
        localOfferShown: this.data.localOfferShown,
      });
    }
    catch (e) {
      console.error("[TTS] desktop default failed:", e);
      return;
    }
    if (plan.action === "none")
      return;
    window.ApiServer.patchService("tts", s => {
      if (plan.action === "local")
        s.data.backend = TTS_Backends.localNeural;
      s.data.desktopDefaultMigrated = true;
    });
    this.#pendingLocalOffer = plan.action === "offer";
  }

  /** Shows the one-time Kokoro offer when this launch planned it (called once after startup, after the STT offer). */
  async runStartupOffer(): Promise<void> {
    if (!this.#pendingLocalOffer)
      return;
    this.#pendingLocalOffer = false;
    await this.offerLocalTts();
  }

  /**
   * One-time offer to move an existing desktop profile to the in-process Kokoro voice (`localNeural`). Never a
   * silent switch: a native dialog asks, and `tts.localOfferShown` records that it was shown either way. A running
   * session is stopped; the voice downloads only after the user agrees in the Kokoro panel, which opens on yes.
   * Resolves true when the user switched.
   */
  async offerLocalTts(): Promise<boolean> {
    if (window.Config?.isApp?.() !== true || this.data.localOfferShown || this.data.backend === TTS_Backends.localNeural)
      return false;
    await i18nReady();
    window.ApiServer.patchService("tts", s => { s.data.localOfferShown = true; });
    let accepted = false;
    try {
      accepted = await ask(i18n.t("tts.local_offer_message"), {
        title: i18n.t("tts.local_offer_title"),
        kind: "info",
        okLabel: i18n.t("tts.local_offer_yes"),
        cancelLabel: i18n.t("tts.local_offer_no"),
      });
    }
    catch (e) {
      console.error("[TTS] local voice offer failed:", e);
      return false;
    }
    // Re-read after the dialog: the user may have picked a backend meanwhile.
    const backendNow: TTS_Backends = window.ApiServer.state.services.tts.data.backend;
    if (!accepted || backendNow === TTS_Backends.localNeural)
      return false;
    if (this.serviceState.status !== ServiceNetworkState.disconnected)
      this.stop();
    window.ApiServer.patchService("tts", s => { s.data.backend = TTS_Backends.localNeural; });
    pushSystemLog("TTS", i18n.t("tts.local_offer_switched"), "info");
    window.ApiServer.changeTab?.({ tab: Services.tts });
    return true;
  }

  /**
   * Stop the running backend, including one still connecting, and return to disconnected. The
   * backend is detached first, so its own `onStop()` (and anything a pending start reports later) is
   * ignored; the status is set here.
   */
  stop(): void {
    const instance = this.#serviceInstance;
    this.#disposePipeline();
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

  #disposePipeline() {
    const pipeline = this.#pipeline;
    this.#pipeline = undefined;
    if (pipeline) {
      try {
        pipeline.dispose();
      } catch (error) {
        console.error("[TTS] pipeline dispose failed", error);
      }
    }
    this.serviceState.pending = 0;
    this.serviceState.speaking = false;
  }

  /** Skip the sentence playing now (sentence pipeline only); the next one follows. */
  skip(): void {
    this.#pipeline?.skip();
  }

  /** Stop speaking everything queued and playing; the service keeps running. */
  stopSpeaking(): void {
    this.#pipeline?.stop();
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
    this.#disposePipeline();
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

  /** Speak `value`; `lang` is its language tag (the TextEvent's `lang`), passed to sink engines per sentence. */
  play(value: string, lang?: string | null) {
    const patchedValue = this.runReplacements(value);
    if (!patchedValue?.trim())
      return;
    const instance = this.#serviceInstance;
    if (!instance || this.serviceState.status !== ServiceNetworkState.connected)
      return;
    const pipeline = this.#pipeline;
    if (pipeline) {
      try {
        pipeline.speak(patchedValue, lang);
      } catch (e) {
        this.onError(errorMessage(e));
      }
      return;
    }
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
    if (ttsOutputMode(instance) !== "selfPlaying") {
      try {
        this.#pipeline = new TtsPipeline(instance, {
          sink: window.ApiServer.sound.sink,
          target: () => ttsSinkTarget(this.data, window.ApiServer.state.audioOutputDevice ?? ""),
          onError: (error) => receiver.onError(error),
          onChange: (state) => {
            if (this.#activeToken !== token) return;
            this.serviceState.pending = state.pending;
            this.serviceState.speaking = state.playing;
          },
        });
      } catch (e) {
        receiver.onStop(errorMessage(e));
        return;
      }
    }
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
