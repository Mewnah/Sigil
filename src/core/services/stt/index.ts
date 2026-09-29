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
  SttMuteState,
  SttStatusLevel
} from "./types";
import { isAbortError, untilAborted } from "./abort";
import { createScopedReceiver, EMPTY_STT_STATUS, SttStatus } from "./statusChannel";
import { planDesktopDefaultMigration } from "./migrate";
import { findWhisperModel, formatModelSize } from "./whisperModels";
import { i18nReady } from "./statusMessages";
import { useAppUIStore } from "@/core/ui/store";
import { Services } from "@/services-registry";

export type SttBackendMap = {
  [k in STT_Backends]?: ISTTServiceConstructor;
};

const backends: SttBackendMap = {
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
 * hangs before ready. Re-armed whenever the engine reports status or progress (onStatus, or an early onInterim), so
 * long first-time model downloads are not cut off while they are still moving.
 */
const STT_CONNECTING_FALLBACK_MS = 45_000;
/** Local model engines: model download/load + WASM init + Rust audio can go quiet for longer between progress events. */
const STT_CONNECTING_FALLBACK_HEAVY_MS = 300_000;
/**
 * Model download / copy progress while connecting, for starts outside the STT inspector (Dashboard, control bar,
 * global shortcut): without it a multi-minute first download looks like a hung "connecting".
 */
const STT_PROGRESS_TOAST_ID = "stt-model-progress";

function connectingFallbackMs(backend: STT_Backends): number {
  return backend === STT_Backends.whisper || backend === STT_Backends.vosk || backend === STT_Backends.moonshine
    ? STT_CONNECTING_FALLBACK_HEAVY_MS
    : STT_CONNECTING_FALLBACK_MS;
}

type SttEngineInstance = InstanceType<NonNullable<SttBackendMap[STT_Backends]>>;

class Service_STT implements IServiceInterface {
  /** Engine classes per backend (injectable for tests). */
  constructor(private readonly engines: SttBackendMap = backends) {}

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
  /** Clears a transient engine status (`expireMs`) unless a newer status replaced it. */
  #statusExpiryTimer: ReturnType<typeof setTimeout> | null = null;
  /** Session epoch the download progress toast was shown for (`null` = not shown). */
  #progressToastEpoch: number | null = null;

  updateLastMessage(value: string, isInterim: boolean) {
    this.#lastMessageState = { value, isInterim };
  }

  serviceState = proxy({
    status: ServiceNetworkState.disconnected,
    error: "",
    muted: SttMuteState.unmuted,
    /**
     * Engine status for the UI (download / load progress, connecting, transient errors, rate-limit waits). Kept apart
     * from transcripts: never published as text, so it can never reach captions, translation, chatbox or TTS.
     */
    statusText: "",
    statusLevel: "info" as SttStatusLevel,
    /** 0-100 while a model download or copy reports progress, else null. */
    progress: null as number | null,
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

  /**
   * Receiver for one engine session (`epoch`). Transcripts go to #sendInterim/#sendFinal (published); status goes to
   * serviceState only (never published). `fallbackMs` null = no connecting fallback (Chrome/Edge use the
   * external-mic timeout).
   */
  #wrapReceiver(epoch: number, fallbackMs: number | null): ISTTReceiver {
    return createScopedReceiver({
      isCurrent: () => epoch === this.#sessionEpoch,
      start: () => this.#onStartScoped(epoch),
      stop: (err) => this.#onStopScoped(epoch, err),
      interim: (v) => void this.#sendInterim(v),
      final: (v) => void this.#sendFinal(v),
      status: (status) => this.#applyEngineStatus(status),
      // Progress while connecting (download %, load status, first words) proves the engine is alive: restart the
      // window so long first-time downloads are not cut off while they are still moving.
      alive: () => {
        if (
          fallbackMs !== null &&
          this.#connectingFallbackTimer != null &&
          this.serviceState.status === ServiceNetworkState.connecting
        )
          this.#scheduleConnectingFallback(epoch, fallbackMs);
      },
      // Engines skip work only on a hard mute: a pending unmute already counts as unmuted for them, so engines
      // without interims (HTTP / Groq) can never leave the unmute pending forever.
      muted: () => this.serviceState.muted === SttMuteState.muted,
    });
  }

  #applyEngineStatus(status: SttStatus): void {
    if (this.#statusExpiryTimer !== null) {
      clearTimeout(this.#statusExpiryTimer);
      this.#statusExpiryTimer = null;
    }
    if (this.serviceState.statusText !== status.text)
      this.serviceState.statusText = status.text;
    if (this.serviceState.progress !== status.progress)
      this.serviceState.progress = status.progress;
    if (this.serviceState.statusLevel !== status.level)
      this.serviceState.statusLevel = status.level;
    if (status.text && status.expireMs) {
      // Transient warnings (skipped speech) clear themselves; any newer status or clear cancels this timer.
      this.#statusExpiryTimer = setTimeout(() => {
        this.#statusExpiryTimer = null;
        this.#applyEngineStatus(EMPTY_STT_STATUS);
      }, status.expireMs);
    }
    this.#syncProgressToast();
  }

  /** The STT inspector already shows status and progress inline. */
  #sttInspectorOpen(): boolean {
    try {
      const { sidebar } = useAppUIStore.getState();
      return sidebar.show && sidebar.tab?.tab === Services.stt;
    }
    catch {
      return false;
    }
  }

  /**
   * Shows model download progress as a toast while connecting when the STT inspector is closed, updates it, and
   * dismisses it when the download ends or the session starts / stops. Closed by the user: stays closed for that
   * download.
   */
  #syncProgressToast(): void {
    const s = this.serviceState;
    const show =
      s.status === ServiceNetworkState.connecting && s.progress !== null && s.statusText !== "" && !this.#sttInspectorOpen();
    try {
      if (show) {
        // A controlled toast closes itself at 1; the end of the download dismisses it instead.
        const progress = Math.min(0.99, Math.max(0, (s.progress ?? 0) / 100));
        if (this.#progressToastEpoch !== this.#sessionEpoch) {
          this.#progressToastEpoch = this.#sessionEpoch;
          toast.info(s.statusText, { toastId: STT_PROGRESS_TOAST_ID, progress, autoClose: false, closeOnClick: true });
        }
        else
          toast.update(STT_PROGRESS_TOAST_ID, { render: s.statusText, progress });
      }
      else if (this.#progressToastEpoch !== null) {
        this.#progressToastEpoch = null;
        toast.dismiss(STT_PROGRESS_TOAST_ID);
      }
    }
    catch {
      /* toasts are best effort */
    }
  }

  #clearEngineStatus(): void {
    this.#applyEngineStatus(EMPTY_STT_STATUS);
  }

  #onStartScoped(epoch: number): void {
    if (epoch !== this.#sessionEpoch)
      return;
    this.#clearConnectingFallbackTimeout();
    // Setup status (download, loading) is over; engines report anything newer after this.
    this.#clearEngineStatus();
    this.#setStatus(ServiceNetworkState.connected);
  }

  #onStopScoped(epoch: number, error?: string): void {
    if (epoch !== this.#sessionEpoch)
      return;
    this.#clearConnectingFallbackTimeout();
    this.#clearEngineStatus();
    // The engine ended the session itself (error, dropped connection): close an open interim sentence and resolve a
    // pending unmute, exactly like a user stop, so neither leaks into the next session.
    this.#resetSentenceState();
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
    this.#clearEngineStatus();

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

  /**
   * Mute (true) or unmute (false) speech-to-text publishing; idempotent, so shortcuts and future external sources
   * (e.g. VRChat's mic mute) can set the state without toggling blindly. Muting mid-sentence closes the open
   * interim; unmuting mid-interim waits for that sentence to finish (pending unmute), so it is never half-published.
   * Engines also skip work while muted where they can (HTTP engines send no request).
   */
  setMuted(muted: boolean) {
    const current = this.serviceState.muted;
    if (muted) {
      if (current === SttMuteState.muted)
        return;
      if (current === SttMuteState.unmuted)
        this.tryCancelSentence();
      this.serviceState.muted = SttMuteState.muted;
      return;
    }
    if (current !== SttMuteState.muted)
      return; // already unmuted, or an unmute is pending
    this.serviceState.muted = this.#lastMessageState.isInterim ? SttMuteState.pendingUnmute : SttMuteState.unmuted;
  }

  /** Closes an open interim sentence ("[...]") and resolves a pending unmute: no sentence spans sessions. */
  #resetSentenceState(): void {
    this.tryCancelSentence();
    this.updateLastMessage("", false);
    this.triggerPendingUnmute();
  }

  toggleMute() {
    if (this.serviceState.muted === SttMuteState.pendingUnmute) {
      // A second press while the unmute waits for the sentence: unmute now.
      this.serviceState.muted = SttMuteState.unmuted;
      return;
    }
    this.setMuted(this.serviceState.muted === SttMuteState.unmuted);
  }

  /** Start when stopped; stop when running or starting (the `start` global shortcut). */
  toggleRunning() {
    if (this.serviceState.status === ServiceNetworkState.disconnected)
      this.start();
    else
      this.stop();
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

    // Starts queue behind the one-time legacy Settings input device migration (native cpal field) and behind
    // i18n (host services init before the translations load; engine status and errors are translated strings).
    this.#startChain = Promise.all([
      migrateLegacyNativeInputDevice().catch((e: unknown) => {
        console.error("[STT] input device migration failed:", e);
      }),
      i18nReady(),
    ]).then(() => {});

    const migrated = this.#migrateDesktopDefault();

    // Web Speech ("native") cannot run in the desktop webview. Right after the migration the model is not on disk
    // yet: the user starts once (one click downloads it) instead of a silent download at launch.
    const nativeBackend = this.data.backend === STT_Backends.native || this.data.backend === STT_Backends.browser;
    if (this.data.autoStart && !nativeBackend && !migrated)
      this.start();
  }

  /**
   * One-time desktop default migration: `native` (the old default, broken in the desktop webview) becomes local
   * Whisper behind the Silero VAD gate. Returns true when the backend was switched in this launch.
   */
  #migrateDesktopDefault(): boolean {
    let plan: ReturnType<typeof planDesktopDefaultMigration>;
    try {
      plan = planDesktopDefaultMigration({
        isDesktop: window.Config.isApp(),
        alreadyMigrated: this.data.desktopDefaultMigrated,
        backend: this.data.backend,
        nativeLanguage: this.data.native.language,
        whisperModel: this.data.whisper.model,
        whisperLanguage: this.data.whisper.language,
        logicalCores: typeof navigator !== "undefined" ? navigator.hardwareConcurrency : undefined,
      });
    }
    catch (e) {
      console.error("[STT] desktop default migration failed:", e);
      return false;
    }
    if (plan.action === "none")
      return false;
    if (plan.action === "mark") {
      window.ApiServer.patchService("stt", s => { s.data.desktopDefaultMigrated = true; });
      return false;
    }
    const { backend, whisperModel, whisperLanguage } = plan;
    window.ApiServer.patchService("stt", s => {
      s.data.backend = backend;
      if (whisperModel)
        s.data.whisper.model = whisperModel;
      if (whisperLanguage)
        s.data.whisper.language = whisperLanguage;
      s.data.desktopDefaultMigrated = true;
    });
    const model = this.data.whisper.model;
    const option = findWhisperModel(model);
    void i18nReady().then(() => {
      const message = i18n.t("stt.migrated_to_whisper", {
        model,
        size: option ? ` (${formatModelSize(option.sizeMb)})` : "",
      });
      pushSystemLog("STT", message, "info");
      // A fresh install goes through onboarding instead; only existing users need to hear about the switch.
      if (window.ApiServer.state.onboardingComplete)
        toast.info(message, { autoClose: 12_000 });
    });
    return true;
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
    // A pending unmute suppresses only the tail of the sentence that was open when the user unmuted. With no open
    // interim (engines that emit only finals) this final is a new sentence spoken after the unmute: publish it.
    const muted = this.serviceState.muted;
    const suppress =
      muted === SttMuteState.muted || (muted === SttMuteState.pendingUnmute && this.#lastMessageState.isInterim);
    !suppress &&
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
      // The engine abandoned the open sentence: nothing is left for a pending unmute to wait for.
      this.updateLastMessage("", false);
      this.triggerPendingUnmute();
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
    this.#syncProgressToast();
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
    // A new session starts with no open sentence (an engine that stopped mid-interim must not leave the next one
    // with a stale interim or a pending unmute that engines without interims can never resolve).
    this.#resetSentenceState();
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
    const serviceConstructor = this.engines[backend];
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
