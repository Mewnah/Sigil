import { IServiceInterface, ServiceNetworkState, TextEventType } from "@/types";
import { proxy } from "valtio";
import { subscribeKey } from "valtio/utils";
import {
    serviceSubscibeToInput,
    serviceSubscibeToSource,
} from "../../../utils";
import KickChatApi from "./chat";
import KickEmotesApi from "./emotes";

// OAuth scopes for Kick
const scope = ["user:read", "channel:read", "chat:write", "chat:read"];

// PKCE helper functions
function generateRandomString(length: number): string {
    const chars = 'ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789-._~';
    let result = '';
    const randomValues = crypto.getRandomValues(new Uint8Array(length));
    for (let i = 0; i < length; i++) {
        result += chars[randomValues[i] % chars.length];
    }
    return result;
}

async function generateCodeChallenge(verifier: string): Promise<string> {
    const encoder = new TextEncoder();
    const data = encoder.encode(verifier);
    const digest = await crypto.subtle.digest('SHA-256', data);
    return btoa(String.fromCharCode(...new Uint8Array(digest)))
        .replace(/\+/g, '-')
        .replace(/\//g, '_')
        .replace(/=+$/, '');
}

class Service_Kick implements IServiceInterface {
    constructor() { }

    emotes!: KickEmotesApi;
    chat!: KickChatApi;

    liveCheckInterval?: any = null;
    private eventDisposers: (() => void)[] = [];

    // Store PKCE verifier for OAuth flow
    private codeVerifier: string = "";
    private oauthState: string = "";

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

    async init() {
        this.emotes = new KickEmotesApi();
        this.chat = new KickChatApi();
        // check live status
        this.liveCheckInterval = setInterval(() => this.#checkLive(), 4000);

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
        }));
    }

    async login() {
        try {
            const clientId = import.meta.env.CURSES_KICK_CLIENT_ID;
            if (!clientId) {
                console.error("[Kick] No client ID configured. Please set CURSES_KICK_CLIENT_ID in .env");
                return;
            }

            const redirect =
                import.meta.env.MODE === "development"
                    ? "http://localhost:1420/oauth_kick.html"
                    : import.meta.env.CURSES_KICK_CLIENT_REDIRECT_LOCAL || "http://localhost:1420/oauth_kick.html";

            // Generate PKCE code verifier and challenge
            this.codeVerifier = generateRandomString(64);
            this.oauthState = generateRandomString(32);
            const codeChallenge = await generateCodeChallenge(this.codeVerifier);

            const link = new URL("https://id.kick.com/oauth/authorize");
            link.searchParams.set("client_id", clientId);
            link.searchParams.set("redirect_uri", redirect);
            link.searchParams.set("response_type", "code");
            link.searchParams.set("scope", scope.join(" "));
            link.searchParams.set("state", this.oauthState);
            link.searchParams.set("code_challenge", codeChallenge);
            link.searchParams.set("code_challenge_method", "S256");

            const auth_window = window.open(link.toString(), "", "width=600,height=700");
            const thisRef = this;

            const handleMessage = async (msg: MessageEvent<unknown>) => {
                if (typeof msg.data === "string") {
                    if (msg.data.startsWith("smplstt_kick_auth:")) {
                        const parts = msg.data.split(":");
                        const code = parts[1];
                        const state = parts[2];

                        if (state !== thisRef.oauthState) {
                            console.error("[Kick] OAuth state mismatch");
                            window.removeEventListener("message", handleMessage, true);
                            return;
                        }

                        await thisRef.exchangeCodeForToken(code);
                        window.removeEventListener("message", handleMessage, true);
                    } else if (msg.data.startsWith("smplstt_kick_auth_error:")) {
                        console.error("[Kick] OAuth error:", msg.data.split(":")[1]);
                        window.removeEventListener("message", handleMessage, true);
                    }
                }
            };

            if (auth_window) {
                window.addEventListener("message", handleMessage, true);
                auth_window.onbeforeunload = () => {
                    window?.removeEventListener("message", handleMessage, true);
                };
            }
        } catch (error) {
            console.error("[Kick] Login error:", error);
        }
    }

    private async exchangeCodeForToken(code: string) {
        try {
            const clientId = import.meta.env.CURSES_KICK_CLIENT_ID;
            const clientSecret = import.meta.env.CURSES_KICK_CLIENT_SECRET;
            const redirect =
                import.meta.env.MODE === "development"
                    ? "http://localhost:1420/oauth_kick.html"
                    : import.meta.env.CURSES_KICK_CLIENT_REDIRECT_LOCAL || "http://localhost:1420/oauth_kick.html";

            const response = await fetch("https://id.kick.com/oauth/token", {
                method: "POST",
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
        this.#state.data.token = "";
        this.#state.data.refreshToken = "";
        this.chat.dispose();
        this.emotes.dispose();
        this.state.user = null;
        this.state.liveStatus = ServiceNetworkState.disconnected;
    }

    dispose() {
        clearInterval(this.liveCheckInterval);
        this.eventDisposers.forEach(d => d());
        this.eventDisposers = [];
        this.chat.dispose();
        this.emotes.dispose();
    }

    async #checkLive() {
        if (!this.state.user?.name) {
            this.state.liveStatus = ServiceNetworkState.disconnected;
            return;
        }
        try {
            // Check if user is live via Kick API
            const response = await fetch(`https://api.kick.com/public/v1/channels?broadcaster_user_id=${this.state.user.id}`, {
                headers: {
                    "Authorization": `Bearer ${this.#state.data.token}`,
                },
            });

            if (response.ok) {
                const data = await response.json();
                const isLive = data.data?.[0]?.stream !== null;
                this.state.liveStatus = isLive ? ServiceNetworkState.connected : ServiceNetworkState.disconnected;
            }
        } catch (error) {
            // Don't log every failed check, just mark as disconnected
            this.state.liveStatus = ServiceNetworkState.disconnected;
        }
    }

    async connect() {
        try {
            if (!this.#state.data.token) return this.logout();

            // Fetch user info from Kick API
            const response = await fetch("https://api.kick.com/public/v1/users", {
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
