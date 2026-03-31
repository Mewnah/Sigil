import { pushSystemLog } from "@/core/services/systemLog";
import { ServiceNetworkState } from "@/types";
import { fetchWithTimeout } from "@/utils/fetchWithTimeout";
import { toast } from "react-toastify";
import { proxy } from "valtio";

const KICK_CHAT_MAX_LEN = 500;

export default class KickChatApi {
  /** REST send path is ready (no IRC); mirrors Twitch “chat connected” for the inspector. */
  state = proxy({
    connection: ServiceNetworkState.disconnected,
  });

  get #kickData() {
    return window.ApiServer.state.services.kick;
  }

  connect(_channel: string) {
    const kick = this.#kickData;
    if (
      kick?.data?.token &&
      window.ApiServer.kick.state.user?.id
    ) {
      this.state.connection = ServiceNetworkState.connected;
    } else {
      this.state.connection = ServiceNetworkState.disconnected;
    }
  }

  disconnect() {
    this.state.connection = ServiceNetworkState.disconnected;
  }

  dispose() {
    this.disconnect();
  }

  post(message: string) {
    const trimmed = message.trim();
    if (!trimmed) return;

    const kick = this.#kickData;
    if (!kick?.data) return;

    const token = kick.data.token;
    const userIdStr = window.ApiServer.kick.state.user?.id;
    const broadcasterUserId = userIdStr ? parseInt(userIdStr, 10) : NaN;

    if (!token || !Number.isFinite(broadcasterUserId) || broadcasterUserId <= 0) return;

    const content = trimmed.slice(0, KICK_CHAT_MAX_LEN);
    const delayMs = parseFloat(kick.data.chatSendDelay) || 0;

    setTimeout(async () => {
      try {
        const response = await fetchWithTimeout("https://api.kick.com/public/v1/chat", {
          method: "POST",
          timeoutMs: 25_000,
          headers: {
            Authorization: `Bearer ${token}`,
            "Content-Type": "application/json",
          },
          body: JSON.stringify({
            content,
            type: "user",
            broadcaster_user_id: broadcasterUserId,
          }),
        });

        let parsed: { message?: string; data?: { is_sent?: boolean } } = {};
        try {
          parsed = (await response.json()) as typeof parsed;
        } catch {
          /* non-JSON body */
        }

        if (!response.ok) {
          const detail = parsed.message || response.statusText;
          if (response.status === 401) {
            pushSystemLog(
              "Kick",
              `Chat post failed (401): ${detail}. Try logging in again.`,
              "error",
            );
            toast.error("Kick: session expired or invalid. Try logging in again.");
          } else {
            pushSystemLog("Kick", `Chat post failed (${response.status}): ${detail}`, "error");
            toast.error(detail ? `Kick chat: ${detail}` : `Kick chat failed (${response.status})`);
          }
          return;
        }

        if (parsed.data?.is_sent === false) {
          pushSystemLog("Kick", `Chat post not sent: ${parsed.message || "unknown"}`, "warning");
        }
      } catch (e) {
        const msg = e instanceof Error ? e.message : String(e);
        pushSystemLog("Kick", `Chat post error: ${msg}`, "error");
        toast.error("Kick chat request failed");
      }
    }, delayMs);
  }
}
