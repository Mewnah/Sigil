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
    changeLanguage(value);
  }

  public async init() {
    if (window.Config.isClient())
      return;
    await this._state.init();
    await window.ApiShared.peer.startServer();

    // Initialize services in parallel
    const results = await Promise.allSettled([
      this.twitch.init(),
      this.kick.init(),
      this.discord.init(),
      this.stt.init(),
      this.tts.init(),
      this.translation.init(),
      this.transform.init(),
      this.vrc.init(),
      this.obs.init(),
      this.keyboard.init(),
      this.voiceChanger.init(),
      this.sound.init(),
    ]);

    const serviceInitLabels = [
      "Twitch",
      "Kick",
      "Discord",
      "Speech-to-Text",
      "Text-to-Speech",
      "Translation",
      "AI Transform",
      "VRChat",
      "OBS",
      "Keyboard",
      "Voice changer",
      "Sound effects",
    ] as const;
    const failedLabels: string[] = [];
    results.forEach((result, index) => {
      const label = serviceInitLabels[index] ?? `Service ${index}`;
      if (result.status === "rejected") {
        console.error(`[ApiServer] Service initialization failed (index ${index}):`, result.reason);
        failedLabels.push(label);
        const msg =
          result.reason instanceof Error ? result.reason.message : String(result.reason);
        pushSystemLog(label, `Initialization failed: ${msg}`, "error");
      }
    });
    if (failedLabels.length > 0) {
      toast.warning(
        `Some features failed to start: ${failedLabels.join(", ")}. Check the console for details.`,
        { autoClose: 10_000 }
      );
    }

    initSystemLogListeners();

    await initI18n(this.state.uiLanguage);
    this.changeTheme(this.state.clientTheme);
    this.changeScale(this.state.uiScale);

    window.ApiShared.pubsub.setTextEmoteEnricher((data) => {
      if (!data.emotes) {
        return { ...data, emotes: window.ApiServer.twitch.emotes.scanForEmotes(data.value) };
      }
      return data;
    });

    window.ApiShared.pubsub.setExternalSttHandler((event) => {
      window.ApiServer.stt.processExternalMessage(event);
    });
  }
}

export default ApiServer;
