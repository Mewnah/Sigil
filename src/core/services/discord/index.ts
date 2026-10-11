import { pushSystemLog } from "@/core/services/systemLog";
import { IServiceInterface, ServiceNetworkState, TextEventSource, TextEventType } from "@/types";
import { serviceSubscibeToInput, serviceSubscibeToSource } from "@/utils";
import { postableFinal } from "../postText";
import {
  buildDiscordExecuteUrl,
  buildDiscordWebhookJsonBody,
  chunkDiscordContent,
  normalizeDiscordWebhookUrl,
  postDiscordContentChunks,
} from "./webhookExecute";

async function logDiscordErrorResponse(res: Response, context: string) {
  let detail = "";
  try {
    const t = await res.text();
    if (t) detail = t.length > 400 ? `${t.slice(0, 400)}…` : t;
  } catch {
    /* ignore */
  }
  pushSystemLog(
    "Discord",
    detail
      ? `Webhook ${context}: HTTP ${res.status} — ${detail}`
      : `Webhook ${context}: HTTP ${res.status}`,
    "warning",
  );
}

class Service_Discord implements IServiceInterface {
  get #state() {
    return window.ApiServer.state.services.discord;
  }

  private eventDisposers: (() => void)[] = [];

  /** When enabled, block posts unless Twitch or Kick reports live. */
  get #blockBecauseNotLive() {
    if (!this.#state.data.postOnlyWhenLive) return false;
    const twitchLive =
      window.ApiServer.twitch.state.liveStatus === ServiceNetworkState.connected;
    const kickLive =
      window.ApiServer.kick.state.liveStatus === ServiceNetworkState.connected;
    return !twitchLive && !kickLive;
  }

  #migratePostTextfieldSource() {
    const d = this.#state.data;
    if (d.postSource !== TextEventSource.textfield) return;
    d.postSource = TextEventSource.stt;
    d.postInput = d.postInput || true;
  }

  async init() {
    this.#migratePostTextfieldSource();

    this.eventDisposers.push(
      serviceSubscibeToSource(this.#state.data, "postSource", (data) => {
        if (this.#blockBecauseNotLive) return;
        const text = postableFinal(data);
        this.#state.data.postEnable && text && this.say(text);
      }),
    );
    this.eventDisposers.push(
      serviceSubscibeToInput(
        this.#state.data,
        "postInput",
        (data) => {
          if (this.#blockBecauseNotLive) return;

          this.#state.data.postEnable &&
            data?.value &&
            data?.type === TextEventType.final &&
            this.say(data.value);
        },
        "postSource",
      ),
    );
  }

  sendTest(): void {
    this.say("Sigil — Discord webhook test. If you see this, the integration is working.");
  }

  say(value: string) {
    if (!value) return;
    const rawHook = this.#state.data.channelHook;
    const hook = normalizeDiscordWebhookUrl(rawHook);
    if (!hook) {
      if (rawHook.trim())
        pushSystemLog(
          "Discord",
          "Invalid webhook URL (use https://discord.com/api/webhooks/… from channel settings).",
          "warning",
        );
      return;
    }

    const d = this.#state.data;
    const executeUrl = buildDiscordExecuteUrl(hook, {
      threadId: d.webhookThreadId,
      wait: true,
    });
    const chunks = chunkDiscordContent(value);

    void (async () => {
      try {
        const last = await postDiscordContentChunks(executeUrl, chunks, (chunk) =>
          buildDiscordWebhookJsonBody(chunk, {
            channelBotName: d.channelBotName,
            channelAvatarUrl: d.channelAvatarUrl,
            webhookThreadName: d.webhookThreadName,
          }),
        );
        if (!last.ok) await logDiscordErrorResponse(last, "rejected");
      } catch {
        pushSystemLog("Discord", "Webhook request failed (network or timeout)", "warning");
      }
    })();
  }

  dispose() {
    this.eventDisposers.forEach((d) => d());
    this.eventDisposers = [];
  }
}

export default Service_Discord;
