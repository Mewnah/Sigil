import { invoke } from "@tauri-apps/api/core";
import { proxy, subscribe } from "valtio";
import { subscribeKey } from "valtio/utils";
import { IServiceInterface, TextEvent, TextEventSource, TextEventType } from "@/types";
import { audioDevicesApi } from "@/core/services/audioDevices";
import { pushSystemLog } from "@/core/services/systemLog";
import { SttMuteState } from "@/core/services/stt/types";
import { serviceSubscibeToInput, serviceSubscibeToSource } from "../../../utils";
import { MuteSyncController } from "./muteSync";
import { OscReceiverClient, listenOptionsFor } from "./receiver";
import { VRC_Backends } from "./schema";
import { RecentTags, TAG_RESET_MS, TagParameterDriver, avatarParameterAddress } from "./tags";
import VRC_KillFrenzyTarget from "./targets/killfrenzy";
import VRC_TextboxTarget from "./targets/textbox";
import { sendOsc as sendOscPacket, type OscArg } from "./targets/utils";
import { IVRCTarget, type RustOscReceiverStatus } from "./types";

/** Debounce of receiver restarts while settings are being typed (port, host). */
const RECEIVER_SYNC_DEBOUNCE_MS = 400;

/**
 * VRChat over OSC: the chatbox / KillFrenzy text targets (paced, paginated, optionally committed at the end of a
 * spoken turn), SenseVoice tags as avatar parameters and chatbox emoji, and mute sync with VRChat's microphone
 * (the Rust OSC receiver, P6-C).
 */
class Service_VRC implements IServiceInterface {
  get #state() {
    return window.ApiServer.state.services.vrc.data;
  }

  /** UI state (not persisted). */
  readonly serviceState = proxy({
    /** The OSC receiver (null until it reported once, or outside the desktop app). */
    receiver: null as RustOscReceiverStatus | null,
    /** VRChat's microphone mute as last reported (null = unknown). */
    vrchatMuted: null as boolean | null,
    /** Speech-to-text is muted because VRChat is. */
    mutedByVrchat: false,
  });

  targets: Record<VRC_Backends, IVRCTarget | undefined> = {
    [VRC_Backends.textbox]: undefined,
    [VRC_Backends.killfrenzy]: undefined
  }

  private eventDisposers: (() => void)[] = [];
  #receiver = new OscReceiverClient();
  #muteSync: MuteSyncController | null = null;
  #receiverTimer: ReturnType<typeof setTimeout> | null = null;
  #recentTags = new RecentTags();
  #tagDriver = new TagParameterDriver({
    send: (name, value) => sendOscPacket(avatarParameterAddress(name), [{ Int: value }]),
    emotionParameter: () => this.#state.tags.emotionParameter,
    eventParameter: () => this.#state.tags.eventParameter,
    resetMs: () => {
      const ms = parseInt(String(this.#state.tags.resetMs ?? ""), 10);
      return Number.isFinite(ms) ? Math.max(0, ms) : TAG_RESET_MS;
    },
  });

  get #activeTarget() {
    return this.targets[this.#state.target];
  }

  async init() {
    this.targets.textbox = new VRC_TextboxTarget({
      turnGate: () => this.#turnGateApplies(),
      tagsFor: (event) => this.#tagsFor(event),
    });
    this.targets.killfrenzy = new VRC_KillFrenzyTarget();
    // pubsub-js passes the topic a message was published on (also to `text` subscribers): typed text is told apart by it.
    this.eventDisposers.push(
      serviceSubscibeToSource(this.#state, "source", (data?: TextEvent, topic?: string) => this.#onSource(data, topic)),
    );
    // Typed text: not subscribed twice when the source already includes `text.textfield`.
    this.eventDisposers.push(serviceSubscibeToInput(this.#state, "inputField", (data) => this.#onTyped(data), "source"));
    // Tags of your own speech, whatever the chatbox source is (avatar parameters, emoji for translated lines).
    const ownSpeech = window.ApiShared.pubsub.subscribeText(TextEventSource.stt, (data) => this.#onOwnSpeech(data));
    this.eventDisposers.push(() => window.ApiShared.pubsub.unsubscribe(ownSpeech));
    // Nothing queued for a target that was switched away from, or for a disabled integration, goes out later.
    this.eventDisposers.push(subscribeKey(this.#state, "target", () => this.#cancelTargets(this.#state.target)));
    this.eventDisposers.push(subscribeKey(this.#state, "enable", (on) => {
      if (!on) {
        this.#cancelTargets();
        this.#tagDriver.reset();
      }
    }));
    this.eventDisposers.push(subscribeKey(this.#state.tags, "enable", (on) => {
      if (!on) this.#tagDriver.reset();
    }));

    void invoke<{ ok: boolean; error?: string }>("plugin:osc|bind_status")
      .then((s) => {
        if (!s.ok && s.error)
          pushSystemLog("VRChat OSC", s.error, "error");
      })
      .catch(() => {});

    await this.#initReceiver();
  }

  sendTest(value: string): void {
    this.#activeTarget?.pushFinal(value, undefined, { typed: true });
  }

  sendOsc(path: string, args: OscArg[]) {
    sendOscPacket(path, args);
  }

  /** Re-reads the receiver status (the inspector refreshes "last message" ages with it). */
  async refreshReceiverStatus(): Promise<void> {
    const status = await this.#receiver.status();
    if (status) this.serviceState.receiver = status;
  }

  // --- text -----------------------------------------------------------------------------------------------------

  /**
   * The chatbox source. With "any text" or "text field" as the source, typed text arrives here (the input-field
   * subscription below is skipped then) and is still a complete message: never held for the end of a spoken turn.
   */
  #onSource(data: TextEvent | undefined, topic?: string) {
    if (!this.#state.enable || !data) return;
    if (data.type === TextEventType.final)
      this.#activeTarget?.pushFinal(data.value, data, { typed: topic === TextEventSource.textfield });
    else this.#activeTarget?.pushInterim(data.value, data);
  }

  #onTyped(data: TextEvent | undefined) {
    if (!this.#state.enable || !data) return;
    if (data.type === TextEventType.final) this.#activeTarget?.pushFinal(data.value, data, { typed: true });
    else this.#activeTarget?.pushInterim(data.value, data);
  }

  #onOwnSpeech(data: TextEvent | undefined) {
    if (!data || data.type !== TextEventType.final) return;
    this.#recentTags.remember(data);
    if (this.#state.enable && this.#state.tags.enable) this.#tagDriver.onFinal(data);
  }

  /**
   * Whether the chatbox's Smart Turn gate applies: translated text is already committed at the end of the turn by the
   * translation's own setting (`partials.turnEndCommit`), so it is not held a second time.
   */
  #turnGateApplies(): boolean {
    if (this.#state.source !== TextEventSource.translation) return true;
    const translation = window.ApiServer.state.services.translation?.data;
    return !(translation?.partials?.turnEndCommit ?? true);
  }

  #tagsFor(event: TextEvent): readonly string[] | undefined {
    const tags = this.#state.tags;
    if (!tags.enable || !tags.chatbox) return undefined;
    return event.tags && event.tags.length > 0 ? event.tags : this.#recentTags.get(event.segmentId);
  }

  #cancelTargets(keep?: VRC_Backends) {
    for (const [key, target] of Object.entries(this.targets)) {
      if (key !== keep) target?.cancel();
    }
  }

  // --- mute sync (OSC receiver) ----------------------------------------------------------------------------------

  async #initReceiver(): Promise<void> {
    if (!window.Config?.isApp?.()) return;
    const stt = window.ApiServer.stt;
    const muteSync = new MuteSyncController({
      isMuted: () => stt.serviceState.muted === SttMuteState.muted,
      setMuted: (muted) => stt.setMuted(muted),
      subscribe: (cb) => subscribeKey(stt.serviceState, "muted", cb, true),
    });
    try {
      await this.#receiver.attach({
        onMuteSelf: (ev) => {
          muteSync.onVrchatMute(ev.muted);
          this.#syncMuteState();
          if (ev.muted) this.#reconcileCaptureGate();
        },
        onStatus: (status) => {
          this.serviceState.receiver = status;
          muteSync.adoptReceiverState(status.muteSelf);
          this.#syncMuteState();
        },
        onVrchat: (ev) => {
          if (!ev.present) muteSync.onVrchatGone();
          this.#syncMuteState();
        },
      });
    } catch (e) {
      muteSync.dispose();
      pushSystemLog("VRChat OSC", `OSC receiver unavailable: ${e instanceof Error ? e.message : String(e)}`, "warning");
      return;
    }
    this.#muteSync = muteSync;
    this.eventDisposers.push(subscribeKey(stt.serviceState, "muted", () => this.#syncMuteState()));
    this.eventDisposers.push(subscribe(this.#state, () => this.#scheduleReceiverSync()));
    this.#syncReceiver();
  }

  /**
   * Rust closes the capture gate the moment VRChat mutes (latency), even when speech-to-text then stays unmuted (the
   * user lifted VRChat's mute by hand earlier, or mute sync was just switched off). Speech-to-text's own mute state
   * is authoritative: the gate is set back to it right away.
   */
  #reconcileCaptureGate() {
    const muted = window.ApiServer.stt.serviceState.muted === SttMuteState.muted;
    if (muted) return; // closed either way
    audioDevicesApi.setCaptureMute("stt", false).catch((e: unknown) => {
      pushSystemLog("VRChat OSC", `Could not reopen the microphone gate: ${e instanceof Error ? e.message : String(e)}`, "warning");
    });
  }

  #scheduleReceiverSync() {
    if (this.#receiverTimer !== null) clearTimeout(this.#receiverTimer);
    this.#receiverTimer = setTimeout(() => {
      this.#receiverTimer = null;
      this.#syncReceiver();
    }, RECEIVER_SYNC_DEBOUNCE_MS);
  }

  /** Runs the receiver while the integration and mute sync are on; applies the controller state first. */
  #syncReceiver() {
    const options = listenOptionsFor(this.#state);
    this.#muteSync?.setEnabled(options !== null);
    this.#syncMuteState();
    this.#receiver
      .sync(options)
      .then((status) => {
        if (status) {
          this.serviceState.receiver = status;
          // After a webview reload Rust keeps the running receiver (identical options) with its deduplicated
          // MuteSelf, so no mute event comes again: a VRChat that is muted right now must still mute speech-to-text.
          this.#muteSync?.adoptReceiverState(status.muteSelf);
          this.#syncMuteState();
        } else if (options === null) this.serviceState.receiver = null;
      })
      .catch((e: unknown) => {
        pushSystemLog("VRChat OSC", `OSC receiver: ${e instanceof Error ? e.message : String(e)}`, "error");
      });
  }

  #syncMuteState() {
    const m = this.#muteSync;
    const vrchatMuted = m?.vrchatMuted ?? null;
    const mutedByVrchat = m?.mutedByVrchat ?? false;
    if (this.serviceState.vrchatMuted !== vrchatMuted) this.serviceState.vrchatMuted = vrchatMuted;
    if (this.serviceState.mutedByVrchat !== mutedByVrchat) this.serviceState.mutedByVrchat = mutedByVrchat;
  }

  dispose() {
    this.eventDisposers.forEach(d => d());
    this.eventDisposers = [];
    if (this.#receiverTimer !== null) clearTimeout(this.#receiverTimer);
    this.#receiverTimer = null;
    this.#cancelTargets();
    this.#tagDriver.dispose();
    this.#muteSync?.setEnabled(false);
    this.#muteSync?.dispose();
    this.#muteSync = null;
    void this.#receiver.dispose();
  }
}

export default Service_VRC;
