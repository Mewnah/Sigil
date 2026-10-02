import Service_Sound from "@/core/services/sound";
import { Services } from "@/services-registry";
import { InspectorTabPath } from "@/types";
import { toast } from "react-toastify";
import { BackendState } from "./schema";
import { useAppUIStore } from "./ui/store";
import Service_Discord from "./services/discord";
import Service_Keyboard from "./services/keyboard";
import Service_OBS from "./services/obs";
import Service_State from "./services/state";
import Service_STT from "./services/stt";
import Service_Translation from "./services/translation";
import Service_Transform from "./services/transform";
import Service_TTS from "./services/tts";
import Service_Twitch from "./services/twitch";
import Service_Kick from "./services/kick";
import Service_VRC from "./services/vrc";
import Service_History from "./services/history";
import { VoiceChangerService } from "./services/voice_changer";
import Service_Models from "./services/models";
import Service_Sysinfo from "./services/sysinfo";
import Service_Resources from "./services/resources";
import Service_AudioDevices from "./services/audioDevices";
import Service_Others from "./services/others";
import Service_Overlay from "./services/overlay";
import Service_Readings from "./services/readings";
import { ensureLlmServerHost } from "./services/llm_server";
import { initSystemLogListeners, pushSystemLog } from "./services/systemLog";
import i18n from "i18next";
import { changeLanguage, initI18n } from '@/i18n';

export { Services };

class ApiServer {
  constructor() { }

  private readonly _state = new Service_State();
  public readonly stt = new Service_STT();
  public readonly tts = new Service_TTS();
  public readonly translation = new Service_Translation();
  public readonly transform = new Service_Transform();
  public readonly twitch = new Service_Twitch();
  public readonly kick = new Service_Kick();
  public readonly discord = new Service_Discord();
  public readonly vrc = new Service_VRC();
  public readonly obs = new Service_OBS();
  public readonly keyboard = new Service_Keyboard();
  public readonly sound = new Service_Sound();
  public readonly history = new Service_History();
  public readonly voiceChanger = new VoiceChangerService();
  /** Model store mirror for the Models & Runtimes tab (P1-B; non-persisted, no patchService). */
  public readonly models = new Service_Models();
  /** System probe mirror: GPUs, drivers, EP eligibility (P1-D2). */
  public readonly sysinfo = new Service_Sysinfo();
  /** VRAM budget / claims / recommendations mirror (P1-D2). */
  public readonly resources = new Service_Resources();
  /** Capture-hub device events: picker refresh, STT restart after device changes (P1-C). */
  public readonly audioDevices = new Service_AudioDevices();
  /** Other players: loopback captions on `others.stt` and reverse translation on `others.translation` (P6-A). */
  public readonly others = new Service_Others();
  /** Caption window and SteamVR caption overlay (P6-B; settings `captionWindow` and `services.vrc.data.overlay`). */
  public readonly overlay = new Service_Overlay();
  /** Japanese / Chinese readings of captions on `readings.stt` (P6-L; settings `readings`). */
  public readonly readings = new Service_Readings();

  get state() {
    return this._state.state;
  }

  closeSidebar() {
    useAppUIStore.getState().closeSidebar();
  }
  changeTab(v?: InspectorTabPath) {
    useAppUIStore.getState().changeTab(v);
  }

  patchService<Key extends keyof BackendState["services"]>(
    service: Key,
    fn: (state: BackendState["services"][Key]) => void
  ) {
    fn(this.state.services[service]);
    // this.state.services[service] = produce(this.state.services[service], fn);
  }

  public changeTheme(value: string) {
    this.state.clientTheme = value;
    document.body.setAttribute("data-theme", value);
  }

  public changeScale(value: number) {
    this.state.uiScale = value;
    document.documentElement.style.setProperty("--uiscale", value.toString());
  }
  public changeLanguage(value: string) {
    this.state.uiLanguage = value;
    void changeLanguage(value);
  }

  public async init() {
    if (window.Config.isClient())
      return;
    await this._state.init();
    await window.ApiShared.peer.startServer();

    window.ApiShared.pubsub.setExternalSttHandler((event) => {
      window.ApiServer.stt.processExternalMessage(event);
    });
    window.ApiShared.pubsub.setExternalMicLifecycleHandler((active) => {
      window.ApiServer.stt.syncExternalMicLifecycle(active);
    });

    const initTasks = this.getHostServiceInitTasks();
    const results = await Promise.allSettled(initTasks.map((t) => t.run()));

    const failedLabels: string[] = [];
    results.forEach((result, index) => {
      const label = initTasks[index]?.label ?? `Service ${index}`;
      if (result.status === "rejected") {
        console.error(`[ApiServer] Service initialization failed (${label}):`, result.reason);
        failedLabels.push(label);
        const msg =
          result.reason instanceof Error ? result.reason.message : String(result.reason);
        pushSystemLog(label, `Initialization failed: ${msg}`, "error");
      }
    });
    initSystemLogListeners();

    await initI18n(this.state.uiLanguage);
    if (failedLabels.length > 0) {
      toast.warning(
        i18n.t("app.service_init_warning", { list: failedLabels.join(", ") }),
        { autoClose: 10_000 }
      );
    }
    this.changeTheme(this.state.clientTheme);
    this.changeScale(this.state.uiScale);
    void this.runSpeechDefaultOffers();

    window.ApiShared.pubsub.setTextEmoteEnricher((data) => {
      if (!data.emotes) {
        if (!window.ApiServer.twitch.emotes.shouldScanEmotesForPubsub(data.value)) {
          return { ...data, emotes: {} };
        }
        return { ...data, emotes: window.ApiServer.twitch.emotes.scanForEmotes(data.value) };
      }
      return data;
    });
  }

  /**
   * One-time offers of the P2 local speech defaults to existing desktop profiles (STT `local`, then the Kokoro
   * voice), one dialog after the other. Each service decides at init whether its offer is due.
   */
  private async runSpeechDefaultOffers() {
    try {
      await this.stt.runStartupOffer();
    }
    catch (e) {
      console.error("[ApiServer] STT default offer failed:", e);
    }
    try {
      await this.tts.runStartupOffer();
    }
    catch (e) {
      console.error("[ApiServer] TTS default offer failed:", e);
    }
  }

  /** Single source for parallel host init: label and runner stay aligned. */
  private getHostServiceInitTasks(): { label: string; run: () => Promise<void> }[] {
    const run = (fn: () => void | PromiseLike<unknown>) => async () => {
      await fn();
    };
    return [
      { label: "Twitch", run: run(() => this.twitch.init()) },
      { label: "Kick", run: run(() => this.kick.init()) },
      { label: "Discord", run: run(() => this.discord.init()) },
      { label: "Speech-to-Text", run: run(() => this.stt.init()) },
      { label: "Text-to-Speech", run: run(() => this.tts.init()) },
      { label: "Translation", run: run(() => this.translation.init()) },
      { label: "AI Transform", run: run(() => this.transform.init()) },
      { label: "VRChat", run: run(() => this.vrc.init()) },
      { label: "OBS", run: run(() => this.obs.init()) },
      { label: "Keyboard", run: run(() => this.keyboard.init()) },
      { label: "Voice changer", run: run(() => this.voiceChanger.init()) },
      { label: "Sound effects", run: run(() => this.sound.init()) },
      { label: "Models", run: run(() => this.models.init()) },
      { label: "System info", run: run(() => this.sysinfo.init()) },
      { label: "Resources", run: run(() => this.resources.init()) },
      // P4-B: state events + idle stop of a llama-server nobody leases (e.g. after a webview reload).
      { label: "Local model server", run: run(() => ensureLlmServerHost()) },
      { label: "Audio devices", run: run(() => this.audioDevices.init()) },
      // P6 streamer features (P6-S0): everything off by default.
      { label: "Other players", run: run(() => this.others.init()) },
      { label: "Caption overlay", run: run(() => this.overlay.init()) },
      { label: "Readings", run: run(() => this.readings.init()) },
    ];
  }
}

export default ApiServer;
