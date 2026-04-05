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
    ];
  }
}

export default ApiServer;
