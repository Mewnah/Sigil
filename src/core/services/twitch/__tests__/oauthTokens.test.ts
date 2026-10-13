import { createHash } from "node:crypto";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";

/**
 * Twitch sign-in after any client secret moved to Rust. A Public app (no secret compiled in) keeps the
 * implicit grant; a Confidential app uses the code flow, with the exchange and the refresh made by the
 * `oauth-token` plugin. The frontend never sees a secret.
 */

const m = vi.hoisted(() => ({
  invoke: vi.fn(),
  open: vi.fn(),
  pushSystemLog: vi.fn(),
  toastWarning: vi.fn(),
  toastError: vi.fn(),
  toastInfo: vi.fn(),
  listenOAuthPayload: vi.fn(),
  exchangeOAuthCode: vi.fn(),
  refreshOAuthToken: vi.fn(),
  oauthClientSecretConfigured: vi.fn(),
  getTokenInfo: vi.fn(),
  clientId: "fakeTwitchClientId0001",
}));

vi.mock("@tauri-apps/api/core", () => ({ invoke: m.invoke, isTauri: () => true }));
vi.mock("@tauri-apps/plugin-shell", () => ({ open: m.open }));
vi.mock("@/core/services/systemLog", () => ({ pushSystemLog: m.pushSystemLog }));
vi.mock("react-toastify", () => ({ toast: { warning: m.toastWarning, error: m.toastError, info: m.toastInfo } }));
vi.mock("i18next", () => ({
  default: { t: (key: string, opts?: { hint?: string }) => (opts?.hint !== undefined ? `${key}|${opts.hint}` : key) },
}));
vi.mock("@/utils/integrationEnv", () => ({
  getTwitchClientId: () => m.clientId,
  getTwitchRedirectUri: () => "http://localhost:1420/oauth/twitch/callback",
  getTwitchRedirectImplicitUri: () => "http://localhost:1420/oauth/twitch/implicit",
}));
vi.mock("@/utils/oauthPopup", () => ({ isOAuthTauri: () => true, listenOAuthPayload: m.listenOAuthPayload }));
vi.mock("@/utils/oauthTokens", () => ({
  exchangeOAuthCode: m.exchangeOAuthCode,
  refreshOAuthToken: m.refreshOAuthToken,
  oauthClientSecretConfigured: m.oauthClientSecretConfigured,
  oauthErrorMessage: (e: unknown) => String(e),
}));
vi.mock("@/utils/fetchWithTimeout", () => ({ fetchWithTimeout: vi.fn() }));
vi.mock("@twurple/auth", () => ({ StaticAuthProvider: class {} }));
vi.mock("@twurple/api", () => ({
  ApiClient: class {
    getTokenInfo = m.getTokenInfo;
    users = { getUserById: async () => ({ id: "7", name: "faketwitchuser" }) };
    streams = { getStreamByUserName: async () => null };
  },
}));
vi.mock("../chat", () => ({ default: class {} }));
vi.mock("../emotes", () => ({ default: class {} }));

import Service_Twitch from "../index";

const LOOPBACK_CODE = "http://localhost:17890/oauth/twitch/callback";
const LOOPBACK_IMPLICIT = "http://localhost:17890/oauth/twitch/implicit";

type Data = { token: string; refreshToken: string; chatEnable: boolean };

let data: Data;
let service: Service_Twitch;
let payloadHandler: ((p: Record<string, string>) => Promise<void> | void) | undefined;

/** The Rust loopback server answers with the redirect URI it will serve (implicit or code). */
function loopback() {
  m.invoke.mockImplementation(async (cmd: string, args: { mode?: string }) => {
    if (cmd !== "oauth_loopback_start") return undefined;
    return args?.mode === "implicit" ? LOOPBACK_IMPLICIT : LOOPBACK_CODE;
  });
}

beforeEach(() => {
  vi.useFakeTimers();
  data = { token: "", refreshToken: "", chatEnable: false };
  vi.stubGlobal("window", {
    ApiServer: { state: { services: { twitch: { data } } }, patchService: vi.fn() },
    ApiShared: { pubsub: { publishLocally: vi.fn() } },
  });
  loopback();
  m.open.mockResolvedValue(undefined);
  m.getTokenInfo.mockResolvedValue({ userId: "7" });
  m.listenOAuthPayload.mockImplementation(async (_provider: string, handler: typeof payloadHandler) => {
    payloadHandler = handler;
    return vi.fn();
  });
  service = new Service_Twitch();
  (service as any).emotes = { loadEmotes: vi.fn(), dispose: vi.fn() };
  (service as any).chat = { connect: vi.fn(), disconnect: vi.fn(), dispose: vi.fn() };
});

afterEach(() => {
  service.dispose();
  vi.useRealTimers();
  vi.unstubAllGlobals();
  vi.clearAllMocks();
  payloadHandler = undefined;
});

/** `login()` runs its work in a floating async function: wait until it has opened the browser. */
async function loginAndWaitForBrowser(): Promise<URL> {
  service.login();
  await vi.waitFor(() => expect(m.open).toHaveBeenCalledTimes(1));
  return new URL(m.open.mock.calls[0]![0] as string);
}

describe("login", () => {
  it("a Public app (no secret compiled in) signs in with the implicit grant", async () => {
    m.oauthClientSecretConfigured.mockResolvedValue(false);
    const url = await loginAndWaitForBrowser();
    expect(m.oauthClientSecretConfigured).toHaveBeenCalledWith("twitch");
    expect(m.invoke).toHaveBeenCalledWith("oauth_loopback_start", { provider: "twitch", mode: "implicit" });
    expect(url.searchParams.get("response_type")).toBe("token");
    expect(url.searchParams.get("redirect_uri")).toBe(LOOPBACK_IMPLICIT);
    expect(url.searchParams.has("code_challenge")).toBe(false);

    await payloadHandler!({ provider: "twitch", access_token: "fake-implicit-token", state: url.searchParams.get("state")! });
    expect(data.token).toBe("fake-implicit-token");
    expect(data.refreshToken).toBe("");
    expect(m.exchangeOAuthCode).not.toHaveBeenCalled();
  });

  it("a Confidential app (secret compiled in) uses the code flow and has Rust exchange the code", async () => {
    m.oauthClientSecretConfigured.mockResolvedValue(true);
    m.exchangeOAuthCode.mockResolvedValue({ accessToken: "fake-access", refreshToken: "fake-refresh" });
    const url = await loginAndWaitForBrowser();
    expect(m.invoke).toHaveBeenCalledWith("oauth_loopback_start", { provider: "twitch" });
    expect(url.searchParams.get("response_type")).toBe("code");
    expect(url.searchParams.get("code_challenge_method")).toBe("S256");
    expect(url.searchParams.get("redirect_uri")).toBe(LOOPBACK_CODE);

    await payloadHandler!({ provider: "twitch", code: "fake-code", state: url.searchParams.get("state")! });
    await vi.waitFor(() => expect(data.token).toBe("fake-access"));

    const [provider, args] = m.exchangeOAuthCode.mock.calls[0]!;
    expect(provider).toBe("twitch");
    expect(args).toEqual({ clientId: m.clientId, code: "fake-code", codeVerifier: expect.any(String), redirectUri: LOOPBACK_CODE });
    expect(createHash("sha256").update(args.codeVerifier).digest("base64url")).toBe(url.searchParams.get("code_challenge"));
    expect(JSON.stringify(args)).not.toMatch(/secret/i);
    expect(data.refreshToken).toBe("fake-refresh");
    expect(service.state.user?.name).toBe("faketwitchuser");
  });

  it("a failed exchange is logged and toasted, with the 'invalid client' hint when Twitch says so", async () => {
    m.oauthClientSecretConfigured.mockResolvedValue(true);
    m.exchangeOAuthCode.mockRejectedValue("Twitch token request: HTTP 400 Bad Request: invalid client secret");
    const url = await loginAndWaitForBrowser();
    await payloadHandler!({ provider: "twitch", code: "fake-code", state: url.searchParams.get("state")! });
    await vi.waitFor(() => expect(m.toastError).toHaveBeenCalled());
    expect(m.pushSystemLog).toHaveBeenCalledWith("Twitch", expect.stringContaining("invalid client secret"), "error");
    expect(m.toastError).toHaveBeenCalledWith(
      "integrations.twitch.login_token_exchange_failed|integrations.twitch.token_exchange_hint_invalid_client",
    );
    expect(data.token).toBe("");

    m.toastError.mockClear();
    m.exchangeOAuthCode.mockRejectedValue("Twitch token request: HTTP 500");
    await payloadHandler!({ provider: "twitch", code: "fake-code-2", state: url.searchParams.get("state")! });
    await vi.waitFor(() => expect(m.toastError).toHaveBeenCalled());
    expect(m.toastError).toHaveBeenCalledWith("integrations.twitch.login_token_exchange_failed|");
  });
});

describe("refreshing the access token", () => {
  it("connect() redeems the refresh token through Rust when the access token is gone", async () => {
    data.refreshToken = "fake-refresh";
    m.refreshOAuthToken.mockResolvedValue({ accessToken: "fake-access-2", refreshToken: "fake-refresh-2" });
    await service.connect();
    expect(m.refreshOAuthToken).toHaveBeenCalledWith("twitch", { clientId: m.clientId, refreshToken: "fake-refresh" });
    expect(data.token).toBe("fake-access-2");
    expect(data.refreshToken).toBe("fake-refresh-2");
    expect(service.state.user?.name).toBe("faketwitchuser");
  });

  it("an expired access token (token check fails) is refreshed once and retried", async () => {
    data.token = "fake-expired";
    data.refreshToken = "fake-refresh";
    m.getTokenInfo.mockRejectedValueOnce(new Error("401")).mockResolvedValue({ userId: "7" });
    m.refreshOAuthToken.mockResolvedValue({ accessToken: "fake-access-3" });
    await service.connect();
    expect(m.refreshOAuthToken).toHaveBeenCalledTimes(1);
    expect(data.token).toBe("fake-access-3");
    expect(data.refreshToken).toBe("fake-refresh");
    expect(service.state.user?.name).toBe("faketwitchuser");
  });

  it("a failed refresh signs out", async () => {
    data.refreshToken = "fake-refresh";
    m.refreshOAuthToken.mockRejectedValue("Twitch token request: HTTP 400 Bad Request: Invalid refresh token");
    await service.connect();
    expect(data.token).toBe("");
    expect(data.refreshToken).toBe("");
    expect(service.state.user).toBeNull();
  });

  it("a Public app has no refresh token, so an expired session just signs out", async () => {
    data.token = "fake-expired";
    m.getTokenInfo.mockRejectedValue(new Error("401"));
    await service.connect();
    expect(m.refreshOAuthToken).not.toHaveBeenCalled();
    expect(data.token).toBe("");
  });
});
