import { createHash } from "node:crypto";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";

/**
 * Kick sign-in after the client secret moved to Rust: the service sends the code, the PKCE verifier and the
 * redirect URI to the `oauth-token` plugin (it never holds a secret), refreshes an expired access token
 * through it, and refuses to start a sign-in the build cannot finish.
 */

const m = vi.hoisted(() => ({
  invoke: vi.fn(),
  open: vi.fn(),
  pushSystemLog: vi.fn(),
  toastWarning: vi.fn(),
  toastError: vi.fn(),
  isOAuthTauri: vi.fn(() => true),
  listenOAuthPayload: vi.fn(),
  exchangeOAuthCode: vi.fn(),
  refreshOAuthToken: vi.fn(),
  oauthClientSecretConfigured: vi.fn(),
  fetchWithTimeout: vi.fn(),
  clientId: "01FAKEKICKCLIENTID",
}));

vi.mock("@tauri-apps/api/core", () => ({ invoke: m.invoke, isTauri: () => true }));
vi.mock("@tauri-apps/plugin-shell", () => ({ open: m.open }));
vi.mock("@/core/services/systemLog", () => ({ pushSystemLog: m.pushSystemLog }));
vi.mock("react-toastify", () => ({ toast: { warning: m.toastWarning, error: m.toastError, info: vi.fn() } }));
vi.mock("i18next", () => ({ default: { t: (key: string) => key } }));
vi.mock("@/utils/integrationEnv", () => ({ getKickClientId: () => m.clientId }));
vi.mock("@/utils/oauthPopup", () => ({
  isOAuthTauri: () => m.isOAuthTauri(),
  listenOAuthPayload: m.listenOAuthPayload,
}));
vi.mock("@/utils/oauthTokens", () => ({
  exchangeOAuthCode: m.exchangeOAuthCode,
  refreshOAuthToken: m.refreshOAuthToken,
  oauthClientSecretConfigured: m.oauthClientSecretConfigured,
  oauthErrorMessage: (e: unknown) => String(e),
}));
vi.mock("@/utils/fetchWithTimeout", () => ({ fetchWithTimeout: m.fetchWithTimeout }));
vi.mock("../chat", () => ({ default: class {} }));
vi.mock("../emotes", () => ({ default: class {} }));

import Service_Kick from "../index";

const LOOPBACK = "http://localhost:17890/oauth/kick/callback";

type Data = { token: string; refreshToken: string; chatEnable: boolean };

function response(status: number, body: unknown = {}) {
  return { ok: status >= 200 && status < 300, status, json: async () => body };
}

const USER_BODY = { data: [{ name: "fakestreamer", user_id: 42, profile_picture: "" }] };

let data: Data;
let service: Service_Kick;
let payloadHandler: ((p: Record<string, string>) => Promise<void>) | undefined;

beforeEach(() => {
  vi.useFakeTimers();
  data = { token: "", refreshToken: "", chatEnable: false };
  vi.stubGlobal("window", {
    ApiServer: { state: { services: { kick: { data } } }, patchService: vi.fn() },
  });
  m.isOAuthTauri.mockReturnValue(true);
  m.oauthClientSecretConfigured.mockResolvedValue(true);
  m.invoke.mockImplementation(async (cmd: string) => (cmd === "oauth_loopback_start" ? LOOPBACK : undefined));
  m.open.mockResolvedValue(undefined);
  m.listenOAuthPayload.mockImplementation(async (_provider: string, handler: typeof payloadHandler) => {
    payloadHandler = handler;
    return vi.fn();
  });
  // /users answers with the streamer; the live check (channels) answers "offline".
  m.fetchWithTimeout.mockImplementation(async (url: string) =>
    url.includes("/users") ? response(200, USER_BODY) : response(200, { data: [{ stream: null }] }),
  );
  service = new Service_Kick();
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

describe("login", () => {
  it("does not start without the desktop app (the secret lives there)", async () => {
    m.isOAuthTauri.mockReturnValue(false);
    await service.login();
    expect(m.toastWarning).toHaveBeenCalledWith("integrations.kick.needs_desktop_app");
    expect(m.oauthClientSecretConfigured).not.toHaveBeenCalled();
    expect(m.invoke).not.toHaveBeenCalled();
    expect(m.open).not.toHaveBeenCalled();
  });

  it("does not start when this build has no Kick client secret, and says so", async () => {
    m.oauthClientSecretConfigured.mockResolvedValue(false);
    await service.login();
    expect(m.oauthClientSecretConfigured).toHaveBeenCalledWith("kick");
    expect(m.toastWarning).toHaveBeenCalledWith("integrations.kick.missing_client_secret");
    expect(m.pushSystemLog).toHaveBeenCalledWith("Kick", expect.stringContaining("no Kick client secret"), "warning");
    expect(m.invoke).not.toHaveBeenCalled();
    expect(m.open).not.toHaveBeenCalled();
  });

  it("opens the authorize URL with a PKCE challenge, and hands Rust the matching verifier", async () => {
    m.exchangeOAuthCode.mockResolvedValue({ accessToken: "fake-access", refreshToken: "fake-refresh", expiresIn: 3600 });
    await service.login();

    expect(m.invoke).toHaveBeenCalledWith("oauth_loopback_start", { provider: "kick" });
    expect(m.open).toHaveBeenCalledTimes(1);
    const url = new URL(m.open.mock.calls[0]![0] as string);
    expect(url.origin + url.pathname).toBe("https://id.kick.com/oauth/authorize");
    expect(url.searchParams.get("client_id")).toBe(m.clientId);
    expect(url.searchParams.get("response_type")).toBe("code");
    expect(url.searchParams.get("redirect_uri")).toBe(LOOPBACK);
    expect(url.searchParams.get("code_challenge_method")).toBe("S256");
    expect(url.searchParams.has("client_secret")).toBe(false);
    const state = url.searchParams.get("state")!;
    const challenge = url.searchParams.get("code_challenge")!;

    // The browser comes back to the loopback server, which emits the payload.
    await payloadHandler!({ provider: "kick", code: "fake-code", state });

    expect(m.exchangeOAuthCode).toHaveBeenCalledTimes(1);
    const [provider, args] = m.exchangeOAuthCode.mock.calls[0]!;
    expect(provider).toBe("kick");
    expect(args).toEqual({ clientId: m.clientId, code: "fake-code", codeVerifier: expect.any(String), redirectUri: LOOPBACK });
    // The verifier Rust is given is the one whose S256 challenge went into the authorize URL.
    const expected = createHash("sha256").update(args.codeVerifier).digest("base64url");
    expect(expected).toBe(challenge);
    expect(args.codeVerifier.length).toBeGreaterThanOrEqual(43);
    // No secret anywhere in what the frontend sent.
    expect(JSON.stringify(args)).not.toMatch(/secret/i);

    expect(data.token).toBe("fake-access");
    expect(data.refreshToken).toBe("fake-refresh");
    expect((service as any).codeVerifier).toBe("");
  });

  it("ignores a callback with the wrong state", async () => {
    await service.login();
    await payloadHandler!({ provider: "kick", code: "fake-code", state: "not-the-state" });
    expect(m.exchangeOAuthCode).not.toHaveBeenCalled();
    expect(m.pushSystemLog).toHaveBeenCalledWith("Kick", "OAuth state mismatch", "error");
    expect(data.token).toBe("");
  });

  it("reports a failed exchange (log and toast) and keeps no token", async () => {
    m.exchangeOAuthCode.mockRejectedValue("Kick token request: HTTP 400 Bad Request: invalid_grant");
    await service.login();
    const state = new URL(m.open.mock.calls[0]![0] as string).searchParams.get("state")!;
    await payloadHandler!({ provider: "kick", code: "fake-code", state });
    expect(m.pushSystemLog).toHaveBeenCalledWith("Kick", expect.stringContaining("invalid_grant"), "error");
    expect(m.toastError).toHaveBeenCalledWith("integrations.kick.login_token_exchange_failed");
    expect(data.token).toBe("");
    expect((service as any).codeVerifier).toBe("");
  });

  it("a denied sign-in is reported as cancelled", async () => {
    await service.login();
    await payloadHandler!({ provider: "kick", error: "access_denied" });
    expect(m.toastError).toHaveBeenCalledWith("integrations.kick.login_cancelled");
    expect(m.exchangeOAuthCode).not.toHaveBeenCalled();
  });
});

describe("refreshing the access token", () => {
  it("connect() with no access token redeems the refresh token, stores the new pair and signs in", async () => {
    data.refreshToken = "fake-refresh";
    m.refreshOAuthToken.mockResolvedValue({ accessToken: "fake-access-2", refreshToken: "fake-refresh-2" });
    await service.connect();
    expect(m.refreshOAuthToken).toHaveBeenCalledWith("kick", { clientId: m.clientId, refreshToken: "fake-refresh" });
    expect(data.token).toBe("fake-access-2");
    expect(data.refreshToken).toBe("fake-refresh-2");
    expect(service.state.user?.name).toBe("fakestreamer");
    const usersCall = m.fetchWithTimeout.mock.calls.find((c) => String(c[0]).includes("/users"))!;
    expect(usersCall[1].headers.Authorization).toBe("Bearer fake-access-2");
  });

  it("keeps the old refresh token when the answer carries none", async () => {
    data.refreshToken = "fake-refresh";
    m.refreshOAuthToken.mockResolvedValue({ accessToken: "fake-access-2" });
    await service.connect();
    expect(data.refreshToken).toBe("fake-refresh");
  });

  it("a 401 from Kick means expired: one refresh, one retry", async () => {
    data.token = "fake-expired";
    data.refreshToken = "fake-refresh";
    m.refreshOAuthToken.mockResolvedValue({ accessToken: "fake-access-3", refreshToken: "fake-refresh-3" });
    let users = 0;
    m.fetchWithTimeout.mockImplementation(async (url: string, init: { headers: { Authorization: string } }) => {
      if (!url.includes("/users")) return response(200, { data: [{ stream: null }] });
      users++;
      return init.headers.Authorization === "Bearer fake-expired" ? response(401) : response(200, USER_BODY);
    });
    await service.connect();
    expect(users).toBe(2);
    expect(m.refreshOAuthToken).toHaveBeenCalledTimes(1);
    expect(data.token).toBe("fake-access-3");
    expect(service.state.user?.name).toBe("fakestreamer");
  });

  it("a second 401 (or a failed refresh) signs out instead of looping", async () => {
    data.token = "fake-expired";
    data.refreshToken = "fake-refresh";
    m.refreshOAuthToken.mockResolvedValue({ accessToken: "fake-still-bad" });
    m.fetchWithTimeout.mockImplementation(async () => response(401));
    await service.connect();
    expect(m.refreshOAuthToken).toHaveBeenCalledTimes(1);
    expect(data.token).toBe("");
    expect(data.refreshToken).toBe("");
    expect(service.state.user).toBeNull();

    // A refresh that fails: logged, signed out.
    data.token = "";
    data.refreshToken = "fake-refresh";
    m.refreshOAuthToken.mockRejectedValue("Kick token request: HTTP 400 Bad Request: invalid_grant");
    await service.connect();
    expect(m.pushSystemLog).toHaveBeenCalledWith("Kick", expect.stringContaining("Token refresh failed"), "warning");
    expect(data.refreshToken).toBe("");
  });

  it("without a refresh token an expired session just signs out", async () => {
    data.token = "fake-expired";
    m.fetchWithTimeout.mockImplementation(async () => response(401));
    await service.connect();
    expect(m.refreshOAuthToken).not.toHaveBeenCalled();
    expect(data.token).toBe("");
  });

  it("concurrent callers share one refresh request", async () => {
    data.refreshToken = "fake-refresh";
    let release!: (v: { accessToken: string }) => void;
    m.refreshOAuthToken.mockReturnValue(new Promise((resolve) => (release = resolve)));
    const a = service.connect();
    const b = service.connect();
    await vi.advanceTimersByTimeAsync(0);
    release({ accessToken: "fake-access-4" });
    await Promise.all([a, b]);
    expect(m.refreshOAuthToken).toHaveBeenCalledTimes(1);
    expect(data.token).toBe("fake-access-4");
  });

  it("drops a refresh result that arrives after the user signed out", async () => {
    data.refreshToken = "fake-refresh";
    let release!: (v: { accessToken: string }) => void;
    m.refreshOAuthToken.mockReturnValue(new Promise((resolve) => (release = resolve)));
    const pending = service.connect();
    await vi.advanceTimersByTimeAsync(0);
    service.logout();
    release({ accessToken: "fake-late-access" });
    await pending;
    expect(data.token).toBe("");
  });

  it("the live check refreshes on a 401 so the next tick has a working token", async () => {
    data.token = "fake-expired";
    data.refreshToken = "fake-refresh";
    m.refreshOAuthToken.mockResolvedValue({ accessToken: "fake-access-5" });
    // /users accepts the token; the channel (live) check answers 401, as an expired token would.
    m.fetchWithTimeout.mockImplementation(async (url: string) =>
      url.includes("/users") ? response(200, USER_BODY) : response(401),
    );
    await service.connect(); // starts the first live check without awaiting it
    await vi.advanceTimersByTimeAsync(0);
    expect(m.refreshOAuthToken).toHaveBeenCalledTimes(1);
    expect(data.token).toBe("fake-access-5");
  });
});
