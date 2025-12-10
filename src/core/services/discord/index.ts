import { IServiceInterface, ServiceNetworkState, TextEventType } from "@/types";
import { serviceSubscibeToInput, serviceSubscibeToSource } from "@/utils";

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

  async init() {
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
    }));
  }

  say(value: string) {
    this.#state.data.channelHook &&
      fetch(this.#state.data.channelHook, {
        method: "POST",
        headers: {
          "Content-type": "application/json",
        },
        body: JSON.stringify({
          content: value,
          embeds: null,
          username: this.#state.data.channelBotName || "Curses",
          avatar_url: this.#state.data.channelAvatarUrl || "",
          attachments: [],
        }),
      });
  }

  dispose() {
    this.eventDisposers.forEach(d => d());
    this.eventDisposers = [];
  }
}

export default Service_Discord;
