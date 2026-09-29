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
import { pushSystemLog } from "@/core/services/systemLog";
import { invoke } from "@tauri-apps/api/core";
import { migrateLegacyNativeInputDevice } from "@/utils/rustAudioInput";
import {
  ISTTReceiver,
  ISTTServiceConstructor,
  SttMuteState
} from "./types";
import { isAbortError, untilAborted } from "./abort";

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
/**
 * Inactivity window while connecting: fires when onStart never arrives (Web Speech quirks) or an async engine
 * hangs before ready. Re-armed whenever the engine reports progress (onInterim), so long first-time model
 * downloads are not cut off while they are still moving.
 */
const STT_CONNECTING_FALLBACK_MS = 45_000;
/** Local model engines: model download/load + WASM init + Rust audio can go quiet for longer between progress events. */
const STT_CONNECTING_FALLBACK_HEAVY_MS = 300_000;

function connectingFallbackMs(backend: STT_Backends): number {
  return backend === STT_Backends.whisper || backend === STT_Backends.vosk || backend === STT_Backends.moonshine
    ? STT_CONNECTING_FALLBACK_HEAVY_MS
    : STT_CONNECTING_FALLBACK_MS;
}

type SttEngineInstance = InstanceType<NonNullable<(typeof backends)[STT_Backends]>>;

class Service_STT implements IServiceInterface {
  #serviceInstance?: SttEngineInstance;
  /** Bumped on each user stop and on each successful start handoff so stale backend callbacks are ignored. */
  #sessionEpoch = 0;
  #lastMessageState = {
    value: "",
    isInterim: false
  }

  #_wordReplacementsCache!: WordReplacementsCache;
  private eventDisposers: (() => void)[] = [];
  #externalMicConnectingTimer: ReturnType<typeof setTimeout> | null = null;
  #connectingFallbackTimer: ReturnType<typeof setTimeout> | null = null;
  /** Serialize start() so overlapping async teardown/start cannot double-stop one engine. */
  #startChain: Promise<void> = Promise.resolve();
  /** Aborted synchronously on each `start()` / `stop()` so the chain is not blocked on a long in-flight `invoke`. */
  #startAbortController: AbortController | null = null;
  /** Every stop() chains backend teardown here so the next start awaits Rust/plugin cleanup (Whisper cpal, etc.). */
  #engineTeardown: Promise<void> = Promise.resolve();

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

  #clearConnectingFallbackTimeout(): void {
    if (this.#connectingFallbackTimer != null) {
      clearTimeout(this.#connectingFallbackTimer);
      this.#connectingFallbackTimer = null;
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

  /** (Re)arms the connecting inactivity timer for session `epoch`; on expiry the session is stopped. */
  #scheduleConnectingFallback(epoch: number, ms: number): void {
    this.#clearConnectingFallbackTimeout();
    this.#connectingFallbackTimer = setTimeout(() => {
      this.#connectingFallbackTimer = null;
      if (epoch !== this.#sessionEpoch)
        return;
      if (this.serviceState.status !== ServiceNetworkState.connecting)
        return;
      toast.warning(i18n.t("toasts.stt_connecting_timeout"));
      this.stop();
    }, ms);
  }

  /** `fallbackMs` null = no connecting fallback (Chrome/Edge use the external-mic timeout). */
  #wrapReceiver(epoch: number, fallbackMs: number | null): ISTTReceiver {
    return {
      onStart: () => this.#onStartScoped(epoch),
      onStop: (err) => this.#onStopScoped(epoch, err),
      onInterim: (v) => {
        if (epoch !== this.#sessionEpoch)
          return;
        // Progress while connecting (download %, load status) proves the engine is alive: restart the window.
        if (
          fallbackMs !== null &&
          this.#connectingFallbackTimer != null &&
          this.serviceState.status === ServiceNetworkState.connecting
        )
          this.#scheduleConnectingFallback(epoch, fallbackMs);
        void this.#sendInterim(v);
      },
      onFinal: (v) => {
        if (epoch !== this.#sessionEpoch)
          return;
        void this.#sendFinal(v);
      },
    };
  }

  #onStartScoped(epoch: number): void {
    if (epoch !== this.#sessionEpoch)
      return;
    this.#clearConnectingFallbackTimeout();
    this.#setStatus(ServiceNetworkState.connected);
  }

  #onStopScoped(epoch: number, error?: string): void {
    if (epoch !== this.#sessionEpoch)
      return;
    this.#clearConnectingFallbackTimeout();
    if (error) {
      toast(error, { type: "error", autoClose: false });
      this.serviceState.error = error;
    }
    this.#setStatus(ServiceNetworkState.disconnected);
  }

  async #finalizeStop(inst: SttEngineInstance): Promise<void> {
    try {
      await inst.stop();
    }
    catch {
      /* ignore */
    }
  }

  /**
   * Await previous engine teardown (e.g. Whisper async stop) before starting another. Leaves the timers alone:
   * the caller arms the connecting fallback before awaiting this, so a teardown that never settles still ends
   * the "connecting" status.
   */
  async #tearDownForNewStart(): Promise<void> {
    await this.#engineTeardown.catch(() => {});
    const inst = this.#serviceInstance;
    if (!inst)
      return;
    await this.#finalizeStop(inst);
    if (this.#serviceInstance === inst)
      this.#serviceInstance = undefined;
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
    this.#startAbortController?.abort();
    this.#startAbortController = null;
    this.#sessionEpoch++;
    this.#clearExternalMicConnectingTimeout();
    this.#clearConnectingFallbackTimeout();

    const ext = this.#isChromeOrEdgeBackend();
    const live =
      this.serviceState.status === ServiceNetworkState.connected ||
      this.serviceState.status === ServiceNetworkState.connecting;

    this.#setStatus(ServiceNetworkState.disconnected);

    const inst = this.#serviceInstance;
    this.#serviceInstance = undefined;
    if (inst) {
      this.#engineTeardown = this.#engineTeardown
        .then(() => this.#finalizeStop(inst))
        .catch(() => {});
    }
    else if (ext && live) {
      window.ApiShared.pubsub.broadcastLocalWebSocketOnly({
        topic: PUBSUB_STT_EXTERNAL_COMMAND,
        data: { cmd: "stop" },
      });
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

    // Starts queue behind the one-time legacy Settings input device migration (native cpal field).
    this.#startChain = migrateLegacyNativeInputDevice().catch((e: unknown) => {
      console.error("[STT] input device migration failed:", e);
    });

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
      try {
        value = await invoke<string>("plugin:uwu|translate", { value: sentence });
      } catch (e: unknown) {
        const msg = e instanceof Error ? e.message : String(e);
        pushSystemLog("STT", `UwU filter failed (final): ${msg}`, "warning");
      }
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
      try {
        value = await invoke<string>("plugin:uwu|translate", { value: sentence });
      } catch (e: unknown) {
        const msg = e instanceof Error ? e.message : String(e);
        pushSystemLog("STT", `UwU filter failed (interim): ${msg}`, "warning");
      }
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

  start() {
    this.#startAbortController?.abort();
    // Captured per call: by the time a queued start runs, stop() may have replaced/cleared the field.
    const controller = new AbortController();
    this.#startAbortController = controller;
    this.#startChain = this.#startChain
      .then(() => this.#runStartAsync(controller.signal))
      .catch((e: unknown) => {
        console.error("[STT] start chain failed:", e);
        toast.error(i18n.t("toasts.stt_start_failed"));
      });
  }

  async #runStartAsync(startSignal: AbortSignal) {
    // Superseded by stop() or a newer start() while queued behind the previous one.
    if (startSignal.aborted)
      return;
    this.#sessionEpoch++;
    const epoch = this.#sessionEpoch;
    this.#clearExternalMicConnectingTimeout();
    this.#setStatus(ServiceNetworkState.connecting);
    // (Re)armed before awaiting the previous engine's teardown: if its stop() never settles, the fallback still
    // fires (toast + stop()) instead of leaving the status on "connecting" indefinitely.
    this.#scheduleConnectingFallback(epoch, STT_CONNECTING_FALLBACK_MS);

    await this.#tearDownForNewStart();
    // stop() (or a newer start()) during teardown owns the status now; do not start a dead engine.
    if (startSignal.aborted || epoch !== this.#sessionEpoch)
      return;

    this.serviceState.error = "";

    const backend = this.data.backend === STT_Backends.browser ? STT_Backends.native : this.data.backend;
    const serviceConstructor = backends[backend];
    if (!serviceConstructor) {
      this.#clearConnectingFallbackTimeout();
      this.#setStatus(ServiceNetworkState.disconnected);
      return;
    }

    const external = backend === STT_Backends.chrome || backend === STT_Backends.edge;
    const fallbackMs = external ? null : connectingFallbackMs(backend);
    const inst = new serviceConstructor(this.#wrapReceiver(epoch, fallbackMs)) as SttEngineInstance;
    inst.startAbortSignal = startSignal;
    this.#serviceInstance = inst;

    if (fallbackMs === null) {
      // Chrome/Edge: the external-mic timeout replaces the teardown-phase fallback.
      this.#clearConnectingFallbackTimeout();
      this.#scheduleExternalMicConnectingTimeout();
    }
    else
      this.#scheduleConnectingFallback(epoch, fallbackMs);

    try {
      await Promise.race([Promise.resolve(inst.start(this.data)), untilAborted(startSignal)]);
    }
    catch (e: unknown) {
      if (!isAbortError(e))
        this.#onStopScoped(epoch, e instanceof Error ? e.message : String(e));
    }
  }

  dispose() {
    this.stop();
    this.eventDisposers.forEach(d => d());
    this.eventDisposers = [];
  }
}

export default Service_STT;
