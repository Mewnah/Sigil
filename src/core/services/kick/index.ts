import { invoke } from "@tauri-apps/api/core";
import { open } from "@tauri-apps/plugin-shell";
import { pushSystemLog } from "@/core/services/systemLog";
import { IServiceInterface, ServiceNetworkState, TextEventSource, TextEventType } from "@/types";
import { getKickClientId, getKickClientSecret, getKickRedirectUri } from "@/utils/integrationEnv";
import { generateOAuthCodeChallenge, generateOAuthRandomString } from "@/utils/oauthPkce";
import { isOAuthTauri, listenOAuthPayload } from "@/utils/oauthPopup";
import { fetchWithTimeout } from "@/utils/fetchWithTimeout";
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
                data?.textFieldType !== "kickChat" &&
                data?.value &&
                data?.type === TextEventType.final &&
                this.chat.post(data.value);
        }, "chatPostSource"));
    }

    async login() {
        try {
            const clientId = getKickClientId();
            if (!clientId) {
                toast.warning("Kick: set SIGIL_KICK_CLIENT_ID in .env (CURSES_KICK_CLIENT_ID still works)");
                pushSystemLog(
                  "Kick",
                  "Missing client id: set SIGIL_KICK_CLIENT_ID or CURSES_KICK_CLIENT_ID in .env",
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

            if (isOAuthTauri()) {
                unlistenTauri = await listenOAuthPayload("kick", async (p) => {
                    if (p.error) {
                        finishWithError(`OAuth error: ${p.error}`, "Kick login was cancelled or denied.");
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

            this.lastOAuthRedirectUri = getKickRedirectUri();
            link.searchParams.set("redirect_uri", this.lastOAuthRedirectUri);
            const authUrl = link.toString();

            const handleMessage = async (msg: MessageEvent<unknown>) => {
                if (typeof msg.data !== "string")
                    return;
                if (msg.data.startsWith("smplstt_kick_auth:")) {
                    const rest = msg.data.slice("smplstt_kick_auth:".length);
                    const idx = rest.indexOf(":");
                    const code = idx === -1 ? rest : rest.slice(0, idx);
                    const state = idx === -1 ? "" : rest.slice(idx + 1);
                    if (state !== thisRef.oauthState) {
                        finishWithError("OAuth state mismatch");
                        window.removeEventListener("message", handleMessage, true);
                        return;
                    }
                    await thisRef.exchangeCodeForToken(code);
                    window.removeEventListener("message", handleMessage, true);
                } else if (msg.data.startsWith("smplstt_kick_auth_error:")) {
                    const err = msg.data.slice("smplstt_kick_auth_error:".length).split(":")[0] ?? "error";
                    finishWithError(`OAuth error: ${err}`, "Kick login was cancelled or denied.");
                    window.removeEventListener("message", handleMessage, true);
                }
            };

            const auth_window = window.open(authUrl, "sigil_oauth_kick", "width=600,height=700");
            if (!auth_window) {
                toast.error("Pop-up was blocked. Allow pop-ups for this site or use the desktop app.");
                pushSystemLog("Kick", "Pop-up blocked; allow pop-ups for this host or use the Sigil desktop build", "warning");
                return;
            }
            window.addEventListener("message", handleMessage, true);
            auth_window.onbeforeunload = () => {
                window.removeEventListener("message", handleMessage, true);
            };
        } catch (error) {
            const msg = error instanceof Error ? error.message : String(error);
            pushSystemLog("Kick", `Login error: ${msg}`, "error");
        }
    }

    private async exchangeCodeForToken(code: string) {
        try {
            const clientId = getKickClientId();
            const clientSecret = getKickClientSecret();
            const redirect = this.lastOAuthRedirectUri || getKickRedirectUri();

            const response = await fetchWithTimeout("https://id.kick.com/oauth/token", {
                method: "POST",
                timeoutMs: 30_000,
                headers: {
                    "Content-Type": "application/x-www-form-urlencoded",
                },
                body: new URLSearchParams({
                    grant_type: "authorization_code",
                    client_id: clientId,
                    client_secret: clientSecret,
                    code: code,
                    redirect_uri: redirect,
                    code_verifier: this.codeVerifier,
                }).toString(),
            });

            if (!response.ok) {
                throw new Error(`Token exchange failed: ${response.status}`);
            }

            const data = await response.json();
            this.#state.data.token = data.access_token;
            this.#state.data.refreshToken = data.refresh_token;

            await this.connect();
        } catch (error) {
            console.error("[Kick] Token exchange error:", error);
        }
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
            if (!this.#state.data.token) return this.logout();

            // Fetch user info from Kick API
            const response = await fetchWithTimeout("https://api.kick.com/public/v1/users", {
                timeoutMs: 30_000,
                headers: {
                    "Authorization": `Bearer ${this.#state.data.token}`,
                },
            });

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
