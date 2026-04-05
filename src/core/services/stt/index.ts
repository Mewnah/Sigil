import { IServiceInterface, ServiceNetworkState, TextEvent, TextEventSource, TextEventType } from "@/types";
import { WordReplacementsCache, buildWordReplacementsCache } from "@/utils";
import i18n from "i18next";
import { toast } from "react-toastify";
import { PUBSUB_STT_EXTERNAL_COMMAND } from "@/shared/services/pubsub";
import { proxy } from "valtio";
import { subscribeKey } from "valtio/utils";
import { STT_Backends } from "./schema";
import { STT_AzureService } from "./services/azure";
import { STT_DeepgramService } from "./services/deepgram";
import { STT_NativeService } from "./services/native";
import { STT_WhisperService } from "./services/whisper";
import { STT_VoskService } from "./services/vosk";
import { STT_MoonshineService } from "./services/moonshine";
import { STT_OpenAI_AudioService } from "./services/openaiAudio";
import { STT_ExternalMicService } from "./services/externalMic";
import { invoke } from "@tauri-apps/api/core";
import {
  ISTTReceiver,
  ISTTServiceConstructor,
  ISpeechRecognitionService,
  SttMuteState
} from "./types";

const backends: {
  [k in STT_Backends]?: ISTTServiceConstructor;
} = {
  [STT_Backends.native]: STT_NativeService,
  [STT_Backends.chrome]: STT_ExternalMicService,
  [STT_Backends.edge]: STT_ExternalMicService,
  [STT_Backends.browser]: undefined,
  [STT_Backends.azure]: STT_AzureService,
  [STT_Backends.deepgram]: STT_DeepgramService,
  [STT_Backends.whisper]: STT_WhisperService,
  [STT_Backends.vosk]: STT_VoskService,
  [STT_Backends.moonshine]: STT_MoonshineService,
  [STT_Backends.openai_audio]: STT_OpenAI_AudioService,
};

const EXTERNAL_MIC_CONNECT_TIMEOUT_MS = 6000;

class Service_STT implements IServiceInterface, ISTTReceiver {
  #serviceInstance?: ISpeechRecognitionService;
  #lastMessageState = {
    value: "",
    isInterim: false
  }

  #_wordReplacementsCache!: WordReplacementsCache;
  private eventDisposers: (() => void)[] = [];
  #externalMicConnectingTimer: ReturnType<typeof setTimeout> | null = null;

  updateLastMessage(value: string, isInterim: boolean) {
    this.#lastMessageState = { value, isInterim };
  }

  serviceState = proxy({
    status: ServiceNetworkState.disconnected,
    error: "",
    muted: SttMuteState.unmuted
  });

  isMuted() {
    return this.serviceState.muted === SttMuteState.muted || this.serviceState.muted === SttMuteState.pendingUnmute;
  }

  get data() {
    return window.ApiServer.state.services.stt.data;
  }

  #isChromeOrEdgeBackend(): boolean {
    return this.data.backend === STT_Backends.chrome || this.data.backend === STT_Backends.edge;
  }

  #clearExternalMicConnectingTimeout(): void {
    if (this.#externalMicConnectingTimer != null) {
      clearTimeout(this.#externalMicConnectingTimer);
      this.#externalMicConnectingTimer = null;
    }
  }

  #scheduleExternalMicConnectingTimeout(): void {
    this.#clearExternalMicConnectingTimeout();
    this.#externalMicConnectingTimer = setTimeout(() => {
      this.#externalMicConnectingTimer = null;
      if (!this.#isChromeOrEdgeBackend())
        return;
      if (this.serviceState.status !== ServiceNetworkState.connecting)
        return;
      toast.warning(i18n.t("toasts.stt_external_mic_timeout"));
      this.#setStatus(ServiceNetworkState.disconnected);
    }, EXTERNAL_MIC_CONNECT_TIMEOUT_MS);
  }

  /** mic.html → host: Web Speech actually started or stopped. */
  syncExternalMicLifecycle(active: boolean): void {
    if (!this.#isChromeOrEdgeBackend())
      return;
    this.#clearExternalMicConnectingTimeout();
    if (active) {
      this.#setStatus(ServiceNetworkState.connected);
    }
    else if (
      this.serviceState.status === ServiceNetworkState.connected ||
      this.serviceState.status === ServiceNetworkState.connecting
    ) {
      this.#setStatus(ServiceNetworkState.disconnected);
    }
  }

  stop(): void {
    this.#clearExternalMicConnectingTimeout();
    const ext = this.#isChromeOrEdgeBackend();
    const live =
      this.serviceState.status === ServiceNetworkState.connected ||
      this.serviceState.status === ServiceNetworkState.connecting;
    if (this.#serviceInstance)
      this.#serviceInstance.stop();
    else if (ext && live) {
      window.ApiShared.pubsub.broadcastLocalWebSocketOnly({
        topic: PUBSUB_STT_EXTERNAL_COMMAND,
        data: { cmd: "stop" },
      });
      this.onStop();
    }
    this.tryCancelSentence();
  }

  toggleMute() {
    if (this.serviceState.muted === SttMuteState.unmuted) {
      // cancel mid sentence and notify user
      this.tryCancelSentence();
      this.serviceState.muted = SttMuteState.muted;
    }
    else if (this.serviceState.muted === SttMuteState.muted) {
      // set pending if unmuting during interim results
      if (this.#lastMessageState.isInterim) {
        this.serviceState.muted = SttMuteState.pendingUnmute;
      }
      else {
        this.serviceState.muted = SttMuteState.unmuted;
      }
    }
    else
      this.serviceState.muted = SttMuteState.unmuted;
  }

  triggerPendingUnmute() {
    // apply unmute if pending
    if (this.serviceState.muted === SttMuteState.pendingUnmute) {
      this.serviceState.muted = SttMuteState.unmuted;
    }
  }

  updateReplacementsCache() {
    this.#_wordReplacementsCache = buildWordReplacementsCache(this.data.replaceWords, this.data.replaceWordsIgnoreCase);
  }

  runReplacements(value: string) {
    if (this.#_wordReplacementsCache.isEmpty)
      return value;
    return value.replace(this.#_wordReplacementsCache.regexp, v => {
      if (this.data.replaceWordsIgnoreCase) {
        let _v = v.toLowerCase();
        if (this.data.replaceWordsPreserveCase) {
          const isUppercase = v[0] === v[0].toUpperCase();
          let replacement = this.#_wordReplacementsCache.map[_v];

          const vCase = isUppercase ? replacement[0].toUpperCase() : replacement[0].toLowerCase();
          replacement = vCase + replacement.slice(1);
          return replacement;
        }
        else {
          return this.#_wordReplacementsCache.map[_v];
        }
      }

      let _v = this.data.replaceWordsIgnoreCase ? v.toLowerCase() : v;
      return this.#_wordReplacementsCache.map[_v];
    });
  }

  async init() {
    this.updateReplacementsCache();
    this.eventDisposers.push(subscribeKey(this.data, "replaceWords", () => this.updateReplacementsCache()));
    this.eventDisposers.push(subscribeKey(this.data, "replaceWordsIgnoreCase", () => this.updateReplacementsCache()));

    const unsubStream = window.ApiShared.pubsub.subscribe("stream.on_ended", () => {
      if (this.data.stopWithStream && this.serviceState.status === ServiceNetworkState.connected) {
        this.stop();
      }
    });
    this.eventDisposers.push(() => window.ApiShared.pubsub.unsubscribe(unsubStream));

    // native is bugged
    if (this.data.autoStart && this.data.backend !== STT_Backends.native)
      this.start();
  }

  tryCancelSentence() {
    if (this.#lastMessageState.isInterim) {
      this.#sendFinal("[...]");
      this.updateLastMessage("", false);
    }
  }

  processExternalMessage(event: Partial<TextEvent>) {
    if (!("type" in event) || !event.value)
      return;
    if (event.type === TextEventType.final)
      this.#sendFinal(event.value);
    else
      this.#sendInterim(event.value);
  }

  async #sendFinal(sentence: string) {
    let value = sentence;
    if (this.data.uwu) {
      value = await invoke<string>("plugin:uwu|translate", { value: sentence });
    }

    value = this.runReplacements(value);
    !this.isMuted() &&
      window.ApiShared.pubsub.publishText(TextEventSource.stt, {
        value,
        type: TextEventType.final,
      });

    this.updateLastMessage(value, false);

    // apply unmute if pending
    this.triggerPendingUnmute();
  }

  async #sendInterim(sentence: string) {
    if (sentence.trim() === "") {
      this.updateLastMessage("", false);
      return;
    }
    let value = sentence;
    if (this.data.uwu) {
      value = await invoke<string>("plugin:uwu|translate", { value: sentence });
    }
    value = this.runReplacements(value);
    !this.isMuted() &&
      window.ApiShared.pubsub.publishText(TextEventSource.stt, {
        value,
        type: TextEventType.interim,
      });
    this.updateLastMessage(value, true);
  }

  #setStatus(value: ServiceNetworkState) {
    this.serviceState.status = value;
  }

  onStart(): void {
    this.#setStatus(ServiceNetworkState.connected);
  }
  onStop(error?: string | undefined): void {
    if (error) {
      toast(error, { type: "error", autoClose: false });
      this.serviceState.error = error;
    }
    this.#setStatus(ServiceNetworkState.disconnected);
  }
  onInterim(value: string): void {
    this.#sendInterim(value);
  }
  onFinal(value: string): void {
    this.#sendFinal(value);
  }

  start() {
    this.stop();
    this.serviceState.error = "";

    let backend = this.data.backend;
    // Legacy persisted value: treat as Web Speech API (same as native).
    if (backend === STT_Backends.browser) {
      backend = STT_Backends.native;
    }
    if (!(backend in backends) || !backends[backend]) {
      return;
    }

    const serviceConstructor = backends[backend];
    if (!serviceConstructor)
      return;
    this.#serviceInstance = new serviceConstructor(this);
    this.#setStatus(ServiceNetworkState.connecting);
    this.#serviceInstance.start(this.data);
    if (this.#isChromeOrEdgeBackend())
      this.#scheduleExternalMicConnectingTimeout();
  }

  dispose() {
    this.stop();
    this.eventDisposers.forEach(d => d());
    this.eventDisposers = [];
  }
}

export default Service_STT;
