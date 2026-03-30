import { invoke } from "@tauri-apps/api/core";
import { open } from "@tauri-apps/plugin-shell";
import { IServiceInterface, ServiceNetworkState, TextEventSource, TextEventType } from "@/types";
import { ApiClient, HelixUser } from "@twurple/api";
import { StaticAuthProvider } from "@twurple/auth";
import { proxy } from "valtio";
import { subscribeKey } from "valtio/utils";
import { toast } from "react-toastify";
import {
  serviceSubscibeToInput,
  serviceSubscibeToSource,
} from "../../../utils";
import { fetchWithTimeout } from "@/utils/fetchWithTimeout";
import {
  getTwitchClientId,
  getTwitchClientSecret,
  getTwitchRedirectImplicitUri,
  getTwitchRedirectUri,
} from "@/utils/integrationEnv";
import { generateOAuthCodeChallenge, generateOAuthRandomString } from "@/utils/oauthPkce";
import { isOAuthTauri, listenOAuthPayload } from "@/utils/oauthPopup";
import { pushSystemLog } from "@/core/services/systemLog";
import TwitchChatApi from "./chat";
import TwitchEmotesApi from "./emotes";

const scope = ["chat:read", "chat:edit", "channel:read:subscriptions"];

class Service_Twitch implements IServiceInterface {
  authProvider?: StaticAuthProvider;
  constructor() { }

  emotes!: TwitchEmotesApi;
  chat!: TwitchChatApi;

  liveCheckInterval?: ReturnType<typeof setInterval> | null = null;
  private eventDisposers: (() => void)[] = [];

  apiClient?: ApiClient;

  private codeVerifier = "";
  private oauthState = "";
  /** Redirect URI used for the last authorize request (must match token exchange). */
  private lastOAuthRedirectUri = "";

  state = proxy<{
    user: HelixUser | null;
    liveStatus: ServiceNetworkState;
  }>({
    liveStatus: ServiceNetworkState.disconnected,
    user: null,
  });

  get #state() {
    return window.ApiServer.state.services.twitch;
  }

  #migrateChatPostTextfieldSource() {
    const d = this.#state.data;
    if (d.chatPostSource !== TextEventSource.textfield) return;
    d.chatPostSource = TextEventSource.stt;
    d.chatPostInput = d.chatPostInput || true;
  }

  async init() {
    this.emotes = new TwitchEmotesApi();
    this.chat = new TwitchChatApi();

    this.#migrateChatPostTextfieldSource();

    this.connect();

    this.eventDisposers.push(subscribeKey(this.#state.data, "chatEnable", (value) => {
      if (value) {
        if (this.state.user && this.authProvider)
          void this.chat.connect(this.state.user.name, this.authProvider);
      } else this.chat.disconnect();
    }));

    this.eventDisposers.push(serviceSubscibeToSource(this.#state.data, "chatPostSource", (data) => {
      if (
        this.#state.data.chatPostLive &&
        this.state.liveStatus !== ServiceNetworkState.connected
      )
        return;
      this.#state.data.chatPostEnable &&
        data?.value &&
        data?.type === TextEventType.final &&
        this.chat.post(data.value);
    }));

    this.eventDisposers.push(serviceSubscibeToInput(this.#state.data, "chatPostInput", (data) => {
      if (
        this.#state.data.chatPostLive &&
        this.state.liveStatus !== ServiceNetworkState.connected
      )
        return;
      this.#state.data.chatPostEnable &&
        data?.textFieldType !== "twitchChat" &&
        data?.value &&
        data?.type === TextEventType.final &&
        this.chat.post(data.value);
    }, "chatPostSource"));
  }

  #stopLiveCheckInterval() {
    if (this.liveCheckInterval != null) {
      clearInterval(this.liveCheckInterval);
      this.liveCheckInterval = null;
    }
  }

  async #checkLive() {
    if (!this.state.user?.name) {
      this.state.liveStatus = ServiceNetworkState.disconnected;
      return;
    }
    try {
      const resp = await this.apiClient?.streams.getStreamByUserName(
        this.state.user.name
      );
      const prevStatus = this.state.liveStatus;
      this.state.liveStatus = !!resp
        ? ServiceNetworkState.connected
        : ServiceNetworkState.disconnected;
      if (prevStatus === ServiceNetworkState.connected && this.state.liveStatus == ServiceNetworkState.disconnected) {
        window.ApiShared.pubsub.publishLocally({ topic: "stream.on_ended" });
      }
    } catch {
      this.state.liveStatus = ServiceNetworkState.disconnected;
    }
  }

  #startLiveCheckInterval() {
    if (this.liveCheckInterval != null) return;
    this.liveCheckInterval = setInterval(() => void this.#checkLive(), 4000);
  }

  async #refreshAccessToken(): Promise<boolean> {
    const rt = this.#state.data.refreshToken;
    const clientId = getTwitchClientId();
    const clientSecret = getTwitchClientSecret();
    if (!rt || !clientId)
      return false;
    try {
      const params = new URLSearchParams({
        client_id: clientId,
        grant_type: "refresh_token",
        refresh_token: rt,
      });
      if (clientSecret)
        params.set("client_secret", clientSecret);
      const res = await fetchWithTimeout("https://id.twitch.tv/oauth2/token", {
        method: "POST",
        timeoutMs: 30_000,
        headers: { "Content-Type": "application/x-www-form-urlencoded" },
        body: params.toString(),
      });
      if (!res.ok)
        return false;
      const data: { access_token?: string; refresh_token?: string } = await res.json();
      if (typeof data.access_token !== "string")
        return false;
      this.#state.data.token = data.access_token;
      if (typeof data.refresh_token === "string")
        this.#state.data.refreshToken = data.refresh_token;
      return true;
    } catch {
      return false;
    }
  }

  async #exchangeCodeForToken(code: string) {
    const clientId = getTwitchClientId();
    const clientSecret = getTwitchClientSecret();
    const redirect = this.lastOAuthRedirectUri || getTwitchRedirectUri();
    try {
      const params = new URLSearchParams({
        client_id: clientId,
        code,
        grant_type: "authorization_code",
        redirect_uri: redirect,
        code_verifier: this.codeVerifier,
      });
      if (clientSecret)
        params.set("client_secret", clientSecret);
      const res = await fetchWithTimeout("https://id.twitch.tv/oauth2/token", {
        method: "POST",
        timeoutMs: 30_000,
        headers: { "Content-Type": "application/x-www-form-urlencoded" },
        body: params.toString(),
      });
      if (!res.ok) {
        const errText = await res.text().catch(() => res.statusText);
        pushSystemLog("Twitch", `Token exchange failed: ${res.status} ${errText}`, "error");
        const hint =
          /invalid client/i.test(errText)
            ? " Check Twitch client ID/secret in .env (SIGIL_* or legacy CURSES_*), or remove the secret for Public-app implicit sign-in."
            : "";
        toast.error(`Twitch login failed (token exchange).${hint} See System Logs.`);
        return;
      }
      const data: { access_token?: string; refresh_token?: string } = await res.json();
      if (typeof data.access_token !== "string") {
        pushSystemLog("Twitch", "Token exchange returned no access_token", "error");
        return;
      }
      this.#state.data.token = data.access_token;
      if (typeof data.refresh_token === "string")
        this.#state.data.refreshToken = data.refresh_token;
      await this.connect();
    } catch (e) {
      const msg = e instanceof Error ? e.message : String(e);
      pushSystemLog("Twitch", `Token exchange error: ${msg}`, "error");
    }
  }

  login() {
    const clientId = getTwitchClientId();
    if (!clientId) {
      toast.warning("Twitch: set SIGIL_TWITCH_CLIENT_ID in .env (CURSES_TWITCH_CLIENT_ID still works)");
      pushSystemLog(
        "Twitch",
        "Missing client id: set SIGIL_TWITCH_CLIENT_ID or CURSES_TWITCH_CLIENT_ID in .env",
        "warning"
      );
      return;
    }

    void (async () => {
      try {
        const clientSecret = getTwitchClientSecret();
        const useImplicit = !clientSecret;

        this.oauthState = generateOAuthRandomString(32);
        this.codeVerifier = useImplicit ? "" : generateOAuthRandomString(64);

        const link = new URL("https://id.twitch.tv/oauth2/authorize");
        link.searchParams.set("client_id", clientId);
        link.searchParams.set("scope", scope.join(" "));
        link.searchParams.set("state", this.oauthState);
        if (useImplicit) {
          link.searchParams.set("response_type", "token");
        } else {
          const codeChallenge = await generateOAuthCodeChallenge(this.codeVerifier);
          link.searchParams.set("response_type", "code");
          link.searchParams.set("code_challenge", codeChallenge);
          link.searchParams.set("code_challenge_method", "S256");
        }

        const thisRef = this;

        const finishWithError = (summary: string, userMessage?: string) => {
          pushSystemLog("Twitch", summary, "error");
          if (userMessage)
            toast.error(userMessage);
        };

        const handleImplicitToken = (access_token: string, state: string) => {
          if (state !== thisRef.oauthState) {
            finishWithError("OAuth state mismatch");
            return;
          }
          thisRef.#state.data.refreshToken = "";
          thisRef.#state.data.token = access_token;
          void thisRef.connect();
          pushSystemLog("Twitch", "Signed in (Public app / implicit). No refresh token.", "info");
          toast.info("Twitch: signed in. Re-login if chat stops when your token expires.");
        };

        const handleSuccess = async (code: string, state: string) => {
          if (state !== thisRef.oauthState) {
            finishWithError("OAuth state mismatch");
            return;
          }
          await thisRef.#exchangeCodeForToken(code);
        };

        let unlistenTauri: (() => void) | undefined;
        const cleanupTauri = () => {
          unlistenTauri?.();
          unlistenTauri = undefined;
        };

        if (isOAuthTauri()) {
          unlistenTauri = await listenOAuthPayload("twitch", async (p) => {
            if (p.error) {
              finishWithError(
                `OAuth error: ${p.error}${p.error_description ? ` — ${p.error_description}` : ""}`,
                "Twitch login was cancelled or denied."
              );
              cleanupTauri();
              return;
            }
            if (p.access_token && p.state) {
              handleImplicitToken(p.access_token, p.state);
              cleanupTauri();
              return;
            }
            if (p.code && p.state) {
              await handleSuccess(p.code, p.state);
              cleanupTauri();
            }
          });

          let redirectUri: string;
          try {
            redirectUri = await invoke<string>("oauth_loopback_start", {
              provider: "twitch",
              ...(useImplicit ? { mode: "implicit" } : {}),
            });
          } catch (err) {
            const msg = err instanceof Error ? err.message : String(err);
            finishWithError(`OAuth listener failed: ${msg}`, "Could not start sign-in (check port 17890).");
            cleanupTauri();
            return;
          }
          thisRef.lastOAuthRedirectUri = redirectUri;
          link.searchParams.set("redirect_uri", redirectUri);
          const authUrlTauri = link.toString();
          try {
            await open(authUrlTauri);
          } catch (err) {
            const msg = err instanceof Error ? err.message : String(err);
            finishWithError(`Open browser failed: ${msg}`, "Could not open your default browser.");
            cleanupTauri();
          }
          return;
        }

        this.lastOAuthRedirectUri = useImplicit
          ? getTwitchRedirectImplicitUri()
          : getTwitchRedirectUri();
        link.searchParams.set("redirect_uri", this.lastOAuthRedirectUri);
        const authUrl = link.toString();

        const handleMessage = async (msg: MessageEvent<unknown>) => {
          if (msg.data && typeof msg.data === "object" && msg.data !== null) {
            const d = msg.data as Record<string, unknown>;
            if (d.source === "sigil-oauth" && d.provider === "twitch" && d.flow === "implicit") {
              if (typeof d.error === "string" && d.error) {
                const desc = typeof d.error_description === "string" ? d.error_description : "";
                pushSystemLog("Twitch", `OAuth error: ${d.error}${desc ? ` — ${desc}` : ""}`, "error");
                toast.error("Twitch login was cancelled or denied.");
                window.removeEventListener("message", handleMessage, true);
                return;
              }
              if (typeof d.access_token === "string" && typeof d.state === "string") {
                handleImplicitToken(d.access_token, d.state);
                window.removeEventListener("message", handleMessage, true);
              }
              return;
            }
          }
          if (typeof msg.data !== "string")
            return;
          if (msg.data.startsWith("smplstt_tw_auth:")) {
            const rest = msg.data.slice("smplstt_tw_auth:".length);
            const idx = rest.indexOf(":");
            const code = idx === -1 ? rest : rest.slice(0, idx);
            const state = idx === -1 ? "" : rest.slice(idx + 1);
            if (state !== thisRef.oauthState) {
              pushSystemLog("Twitch", "OAuth state mismatch", "error");
              window.removeEventListener("message", handleMessage, true);
              return;
            }
            await thisRef.#exchangeCodeForToken(code);
            window.removeEventListener("message", handleMessage, true);
          } else if (msg.data.startsWith("smplstt_tw_auth_error:")) {
            const rest = msg.data.slice("smplstt_tw_auth_error:".length);
            const idx = rest.indexOf(":");
            const err = idx === -1 ? rest : rest.slice(0, idx);
            const desc = idx === -1 ? "" : rest.slice(idx + 1);
            pushSystemLog("Twitch", `OAuth error: ${err}${desc ? ` — ${desc}` : ""}`, "error");
            toast.error("Twitch login was cancelled or denied.");
            window.removeEventListener("message", handleMessage, true);
          }
        };

        const auth_window = window.open(authUrl, "sigil_oauth_twitch", "width=600,height=600");
        if (!auth_window) {
          toast.error("Pop-up was blocked. Allow pop-ups for this site or use the desktop app.");
          pushSystemLog("Twitch", "Pop-up blocked; allow pop-ups for this host or use the Sigil desktop build", "warning");
          return;
        }
        window.addEventListener("message", handleMessage, true);
        auth_window.onbeforeunload = () => {
          window.removeEventListener("message", handleMessage, true);
        };
      } catch (e) {
        const msg = e instanceof Error ? e.message : String(e);
        pushSystemLog("Twitch", `Login error: ${msg}`, "error");
      }
    })();
  }

  logout() {
    this.#stopLiveCheckInterval();
    this.lastOAuthRedirectUri = "";
    window.ApiServer.state.services.twitch.data.token = "";
    window.ApiServer.state.services.twitch.data.refreshToken = "";
    this.chat.dispose();
    delete this.apiClient;
    delete this.authProvider;
    this.emotes.dispose();
    this.state.user = null;
    this.state.liveStatus = ServiceNetworkState.disconnected;
  }

  dispose() {
    this.#stopLiveCheckInterval();
    this.eventDisposers.forEach(d => d());
    this.eventDisposers = [];
    this.chat.dispose();
    this.emotes.dispose();
  }

  async connect() {
    const clientId = getTwitchClientId();
    if (!clientId) {
      pushSystemLog(
        "Twitch",
        "Missing client id: set SIGIL_TWITCH_CLIENT_ID or CURSES_TWITCH_CLIENT_ID in .env",
        "warning"
      );
      return;
    }

    try {
      if (!this.#state.data.token) {
        if (this.#state.data.refreshToken) {
          const refreshed = await this.#refreshAccessToken();
          if (!refreshed)
            return this.logout();
        } else {
          return this.logout();
        }
      }

      let accessToken = this.#state.data.token;
      this.authProvider = new StaticAuthProvider(
        clientId,
        accessToken,
        scope
      );

      this.apiClient = new ApiClient({ authProvider: this.authProvider });

      let tokenInfo;
      try {
        tokenInfo = await this.apiClient.getTokenInfo();
      } catch {
        if (this.#state.data.refreshToken) {
          const refreshed = await this.#refreshAccessToken();
          if (!refreshed)
            return this.logout();
          accessToken = this.#state.data.token;
          this.authProvider = new StaticAuthProvider(clientId, accessToken, scope);
          this.apiClient = new ApiClient({ authProvider: this.authProvider });
          tokenInfo = await this.apiClient.getTokenInfo();
        } else {
          return this.logout();
        }
      }

      if (!tokenInfo.userId)
        return this.logout();

      const me = await this.apiClient.users.getUserById({
        id: tokenInfo.userId,
      });

      if (!me)
        return this.logout();

      this.state.user = me;

      this.#checkLive();
      this.#startLiveCheckInterval();

      void this.emotes.loadEmotes(me.id, this.apiClient, me.name);
      if (this.#state.data.chatEnable)
        void this.chat.connect(me.name, this.authProvider);
    } catch {
      this.logout();
    }
  }
}

export default Service_Twitch;
