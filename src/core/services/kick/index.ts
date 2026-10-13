import { invoke } from "@tauri-apps/api/core";
import { open } from "@tauri-apps/plugin-shell";
import { pushSystemLog } from "@/core/services/systemLog";
import { IServiceInterface, ServiceNetworkState, TextEventSource, TextEventType } from "@/types";
import { getKickClientId } from "@/utils/integrationEnv";
import { generateOAuthCodeChallenge, generateOAuthRandomString } from "@/utils/oauthPkce";
import { isOAuthTauri, listenOAuthPayload } from "@/utils/oauthPopup";
import {
    exchangeOAuthCode,
    oauthClientSecretConfigured,
    oauthErrorMessage,
    refreshOAuthToken,
} from "@/utils/oauthTokens";
import { fetchWithTimeout } from "@/utils/fetchWithTimeout";
import { postableFinal } from "../postText";
import i18n from "i18next";
import { toast } from "react-toastify";
import { proxy } from "valtio";
import { subscribeKey } from "valtio/utils";
import {
    serviceSubscibeToInput,
    serviceSubscibeToSource,
} from "../../../utils";
import KickChatApi from "./chat";
import KickEmotesApi from "./emotes";

const scope = ["user:read", "channel:read", "chat:write", "chat:read"];

class Service_Kick implements IServiceInterface {
    constructor() { }

    emotes!: KickEmotesApi;
    chat!: KickChatApi;

    liveCheckInterval?: any = null;
    private eventDisposers: (() => void)[] = [];

    // Store PKCE verifier for OAuth flow
    private codeVerifier: string = "";
    private oauthState: string = "";
    private lastOAuthRedirectUri = "";
    /** The refresh call in flight, so a burst of 401s makes one request. */
    private refreshing: Promise<boolean> | null = null;

    state = proxy<{
        user: { name: string, id: string, profilePictureUrl: string } | null;
        liveStatus: ServiceNetworkState;
    }>({
        liveStatus: ServiceNetworkState.disconnected,
        user: null,
    });

    get #state() {
        return window.ApiServer.state.services.kick;
    }

    #stopLiveCheckInterval() {
        if (this.liveCheckInterval != null) {
            clearInterval(this.liveCheckInterval);
            this.liveCheckInterval = undefined;
        }
    }

    async #checkLive() {
        if (!this.state.user?.name) {
            this.state.liveStatus = ServiceNetworkState.disconnected;
            return;
        }
        try {
            const response = await fetchWithTimeout(
                `https://api.kick.com/public/v1/channels?broadcaster_user_id=${this.state.user.id}`,
                {
                    timeoutMs: 25_000,
                    headers: {
                        "Authorization": `Bearer ${this.#state.data.token}`,
                    },
                }
            );

            if (response.ok) {
                const data = (await response.json()) as {
                    data?: Array<{ stream?: { is_live?: boolean } | null }>;
                };
                const stream = data.data?.[0]?.stream;
                const isLive =
                    stream != null &&
                    typeof stream === "object" &&
                    stream.is_live === true;
                this.state.liveStatus = isLive ? ServiceNetworkState.connected : ServiceNetworkState.disconnected;
            } else {
                this.state.liveStatus = ServiceNetworkState.disconnected;
                // 401: the access token expired. Get a new one; the next tick uses it.
                if (response.status === 401 && this.#state.data.refreshToken)
                    void this.#refreshAccessToken();
            }
        } catch (error) {
            this.state.liveStatus = ServiceNetworkState.disconnected;
        }
    }

    #startLiveCheckInterval() {
        if (this.liveCheckInterval != null) return;
        this.liveCheckInterval = setInterval(() => this.#checkLive(), 4000);
    }

    /** Legacy: “Text field” as sole source is replaced by STT + “Use keyboard input”. */
    #migrateChatPostTextfieldSource() {
        const d = this.#state.data;
        if (d.chatPostSource !== TextEventSource.textfield) return;
        d.chatPostSource = TextEventSource.stt;
        d.chatPostInput = d.chatPostInput || true;
    }

    async init() {
        this.emotes = new KickEmotesApi();
        this.chat = new KickChatApi();

        this.#migrateChatPostTextfieldSource();

        // login with token if available
        this.connect();

        this.eventDisposers.push(subscribeKey(this.#state.data, "chatEnable", (value) => {
            if (value) {
                if (this.state.user)
                    this.chat.connect(this.state.user.name);
            } else {
                this.chat.disconnect();
                window.ApiServer.patchService("kick", (s) => {
                    s.data.chatPostEnable = false;
                    s.data.chatPostLive = false;
                    s.data.chatReceiveEnable = false;
                });
            }
        }));

        this.eventDisposers.push(serviceSubscibeToSource(this.#state.data, "chatPostSource", (data) => {
            if (
                this.#state.data.chatPostLive &&
                this.state.liveStatus !== ServiceNetworkState.connected
            )
                return;
            const text = postableFinal(data);
            this.#state.data.chatEnable &&
                this.#state.data.chatPostEnable &&
                text &&
                this.chat.post(text);
        }));

        this.eventDisposers.push(serviceSubscibeToInput(this.#state.data, "chatPostInput", (data) => {
            if (
                this.#state.data.chatPostLive &&
                this.state.liveStatus !== ServiceNetworkState.connected
            )
                return;
            this.#state.data.chatEnable &&
                this.#state.data.chatPostEnable &&
                data?.textFieldType !== "kickChat" &&
                data?.value &&
                data?.type === TextEventType.final &&
                this.chat.post(data.value);
        }, "chatPostSource"));

        if (!this.#state.data.chatEnable) {
            window.ApiServer.patchService("kick", (s) => {
                s.data.chatPostEnable = false;
                s.data.chatPostLive = false;
                s.data.chatReceiveEnable = false;
            });
        }
    }

    async login() {
        try {
            const clientId = getKickClientId();
            if (!clientId) {
                toast.warning(i18n.t("integrations.kick.missing_client_id"));
                pushSystemLog(
                  "Kick",
                  "Missing client id: set SIGIL_KICK_CLIENT_ID or CURSES_KICK_CLIENT_ID in .env",
                  "warning"
                );
                return;
            }

            // Kick's token endpoint needs the app's client secret, which lives in the Rust side of the desktop
            // app (never in this bundle), so signing in needs the app and a build that has the secret.
            if (!isOAuthTauri()) {
                toast.warning(i18n.t("integrations.kick.needs_desktop_app"));
                pushSystemLog("Kick", "Sign-in needs the Sigil desktop app: the token exchange runs there", "warning");
                return;
            }
            if (!(await oauthClientSecretConfigured("kick"))) {
                toast.warning(i18n.t("integrations.kick.missing_client_secret"));
                pushSystemLog(
                  "Kick",
                  "This build has no Kick client secret compiled in: set it in .env (or the build environment) and rebuild",
                  "warning"
                );
                return;
            }

            this.codeVerifier = generateOAuthRandomString(64);
            this.oauthState = generateOAuthRandomString(32);
            const codeChallenge = await generateOAuthCodeChallenge(this.codeVerifier);

            const link = new URL("https://id.kick.com/oauth/authorize");
            link.searchParams.set("client_id", clientId);
            link.searchParams.set("response_type", "code");
            link.searchParams.set("scope", scope.join(" "));
            link.searchParams.set("state", this.oauthState);
            link.searchParams.set("code_challenge", codeChallenge);
            link.searchParams.set("code_challenge_method", "S256");

            const thisRef = this;

            const finishWithError = (summary: string, userMessage?: string) => {
                pushSystemLog("Kick", summary, "error");
                if (userMessage)
                    toast.error(userMessage);
            };

            const handleSuccess = async (code: string, state: string) => {
                if (state !== thisRef.oauthState) {
                    finishWithError("OAuth state mismatch");
                    return;
                }
                await thisRef.exchangeCodeForToken(code);
            };

            let unlistenTauri: (() => void) | undefined;
            const cleanupTauri = () => {
                unlistenTauri?.();
                unlistenTauri = undefined;
            };

            unlistenTauri = await listenOAuthPayload("kick", async (p) => {
                if (p.error) {
                    finishWithError(`OAuth error: ${p.error}`, i18n.t("integrations.kick.login_cancelled"));
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
                redirectUri = await invoke<string>("oauth_loopback_start", { provider: "kick" });
            } catch (err) {
                const msg = err instanceof Error ? err.message : String(err);
                finishWithError(`OAuth listener failed: ${msg}`, i18n.t("integrations.oauth_listen_failed"));
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
                finishWithError(`Open browser failed: ${msg}`, i18n.t("integrations.oauth_open_browser_failed"));
                cleanupTauri();
            }
        } catch (error) {
            const msg = error instanceof Error ? error.message : String(error);
            pushSystemLog("Kick", `Login error: ${msg}`, "error");
        }
    }

    /** Swap the authorization code for tokens. The request, with the client secret, is made by Rust. */
    private async exchangeCodeForToken(code: string) {
        try {
            const redirectUri = this.lastOAuthRedirectUri;
            if (!redirectUri)
                throw new Error("no redirect URI for this sign-in");

            const tokens = await exchangeOAuthCode("kick", {
                clientId: getKickClientId(),
                code,
                codeVerifier: this.codeVerifier,
                redirectUri,
            });
            this.#state.data.token = tokens.accessToken;
            this.#state.data.refreshToken = tokens.refreshToken ?? "";

            await this.connect();
        } catch (error) {
            pushSystemLog("Kick", `Token exchange failed: ${oauthErrorMessage(error)}`, "error");
            toast.error(i18n.t("integrations.kick.login_token_exchange_failed"));
        } finally {
            // A PKCE verifier is single use.
            this.codeVerifier = "";
        }
    }

    /**
     * Redeem the refresh token (in Rust, with the client secret). True when a new access token is stored.
     * Concurrent callers share one request; a result for a session that has since logged out is dropped.
     */
    #refreshAccessToken(): Promise<boolean> {
        if (this.refreshing)
            return this.refreshing;
        const refreshToken = this.#state.data.refreshToken;
        const clientId = getKickClientId();
        if (!refreshToken || !clientId || !isOAuthTauri())
            return Promise.resolve(false);
        const run = (async () => {
            try {
                const tokens = await refreshOAuthToken("kick", { clientId, refreshToken });
                if (this.#state.data.refreshToken !== refreshToken)
                    return false;
                this.#state.data.token = tokens.accessToken;
                if (tokens.refreshToken)
                    this.#state.data.refreshToken = tokens.refreshToken;
                return true;
            } catch (error) {
                pushSystemLog("Kick", `Token refresh failed: ${oauthErrorMessage(error)}`, "warning");
                return false;
            }
        })().finally(() => {
            this.refreshing = null;
        });
        this.refreshing = run;
        return run;
    }

    logout() {
        this.#stopLiveCheckInterval();
        this.lastOAuthRedirectUri = "";
        this.#state.data.token = "";
        this.#state.data.refreshToken = "";
        this.chat.dispose();
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
        try {
            if (!this.#state.data.token) {
                // Signed in earlier, access token gone: try the refresh token before giving up.
                if (!this.#state.data.refreshToken || !(await this.#refreshAccessToken()))
                    return this.logout();
            }

            // Fetch user info from Kick API
            const fetchUser = () => fetchWithTimeout("https://api.kick.com/public/v1/users", {
                timeoutMs: 30_000,
                headers: {
                    "Authorization": `Bearer ${this.#state.data.token}`,
                },
            });
            let response = await fetchUser();

            // 401: the access token expired. One refresh, one retry.
            if (response.status === 401 && this.#state.data.refreshToken && await this.#refreshAccessToken())
                response = await fetchUser();

            if (!response.ok) {
                console.error("[Kick] Failed to get user info:", response.status);
                return this.logout();
            }

            const data = await response.json();
            const userData = data.data?.[0];

            if (!userData) {
                return this.logout();
            }

            this.state.user = {
                name: userData.name || userData.username,
                id: userData.user_id?.toString() || userData.id?.toString() || "0",
                profilePictureUrl: userData.profile_picture || "",
            };

            // initial live check
            this.#checkLive();
            this.#startLiveCheckInterval();

            this.emotes.loadEmotes(this.state.user.id);
            if (this.#state.data.chatEnable)
                this.chat.connect(this.state.user.name);
        } catch (error) {
            console.error("[Kick] Connect error:", error);
            this.logout();
        }
    }
}

export default Service_Kick;
