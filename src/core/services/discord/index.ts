import { pushSystemLog } from "@/core/services/systemLog";
import { IServiceInterface, ServiceNetworkState, TextEventSource, TextEventType } from "@/types";
import { serviceSubscibeToInput, serviceSubscibeToSource } from "@/utils";
import { fetchWithTimeout } from "@/utils/fetchWithTimeout";

class Service_Discord implements IServiceInterface {
  get #state() {
    return window.ApiServer.state.services.discord;
  }

  private eventDisposers: (() => void)[] = [];

  get checkTwitch() {
    return (
      this.#state.data.postWithTwitchLive &&
      window.ApiServer.twitch.state.liveStatus !== ServiceNetworkState.connected
    );
  }

  #migratePostTextfieldSource() {
    const d = this.#state.data;
    if (d.postSource !== TextEventSource.textfield) return;
    d.postSource = TextEventSource.stt;
    d.postInput = d.postInput || true;
  }

  async init() {
    this.#migratePostTextfieldSource();

    this.eventDisposers.push(serviceSubscibeToSource(this.#state.data, "postSource", (data) => {
      if (this.checkTwitch) return;
      this.#state.data.postEnable &&
        data?.value &&
        data?.type === TextEventType.final &&
        this.say(data.value);
    }));
    this.eventDisposers.push(serviceSubscibeToInput(this.#state.data, "postInput", (data) => {
      if (this.checkTwitch) return;

      this.#state.data.postEnable &&
        data?.value &&
        data?.type === TextEventType.final &&
        this.say(data.value);
    }, "postSource"));
  }

  say(value: string) {
    const hook = this.#state.data.channelHook;
    if (!hook) return;
    void (async () => {
      try {
        const res = await fetchWithTimeout(hook, {
          method: "POST",
          timeoutMs: 25_000,
          headers: {
            "Content-type": "application/json",
          },
          body: JSON.stringify({
            content: value,
            embeds: null,
            username: this.#state.data.channelBotName || "Sigil",
            avatar_url: this.#state.data.channelAvatarUrl || "",
            attachments: [],
          }),
        });
        if (!res.ok)
          pushSystemLog("Discord", `Webhook rejected: HTTP ${res.status}`, "warning");
      } catch {
        pushSystemLog("Discord", "Webhook request failed (network or invalid URL)", "warning");
      }
    })();
  }

  dispose() {
    this.eventDisposers.forEach(d => d());
    this.eventDisposers = [];
  }
}

export default Service_Discord;
