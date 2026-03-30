import { pushSystemLog } from "@/core/services/systemLog";
import { ServiceNetworkState, TextEventSource, TextEventType } from "@/types";
import { AuthProvider } from "@twurple/auth";
import { ChatClient } from "@twurple/chat";
import { proxy } from "valtio";

class TwitchChatApi {
  chatClient?: ChatClient;

  get #state() {
    return window.ApiServer.state.services.twitch;
  }

  state = proxy({
    username: "",
    connection: ServiceNetworkState.disconnected,
  });

  async connect(username: string, authProvider: AuthProvider) {
    const canStart =
      this.state.connection === ServiceNetworkState.disconnected ||
      this.state.connection === ServiceNetworkState.error;
    if (!canStart)
      return;
    this.state.connection = ServiceNetworkState.connecting;
    this.chatClient = new ChatClient({ authProvider, channels: [username] });
    this.chatClient.irc.onConnect(() => {
      this.state.username = username;
      this.state.connection = ServiceNetworkState.connected;
    });

    this.chatClient.irc.onDisconnect(() => {
      this.state.username = "";
      this.state.connection = ServiceNetworkState.disconnected;
    });

    this.chatClient.onMessage((_channel, _user, _message, msg) => {
      const selfLogin = window.ApiServer.twitch.state.user?.name;
      if (
        selfLogin &&
        msg.userInfo.userName.toLowerCase() === selfLogin.toLowerCase()
      ) {
        return;
      }

      if (this.#state.data.chatReceiveEnable) {
        window.ApiShared.pubsub.publishText(TextEventSource.textfield, {
          type: TextEventType.final,
          value: msg.text,
          textFieldType: "twitchChat",
        });
      }
    });

    try {
      await this.chatClient.connect();
    } catch (e) {
      const message = e instanceof Error ? e.message : String(e);
      this.state.username = "";
      this.state.connection = ServiceNetworkState.error;
      pushSystemLog("Twitch", `Chat failed to connect: ${message}`, "error");
      try {
        this.chatClient.quit();
      } catch {
        /* ignore */
      }
      this.chatClient = undefined;
    }
  }

  disconnect() {
    this.state.connection = ServiceNetworkState.disconnected;
    this.chatClient?.quit();
  }

  dispose() {
    this.disconnect(); 
  }

  post(message: string) {
    this.state.username &&
    this.state.connection === ServiceNetworkState.connected &&
    setTimeout(() => {
      this.chatClient?.say(this.state.username, message);
    }, parseFloat(this.#state.data.chatSendDelay) || 0);
  }
}

export default TwitchChatApi;
