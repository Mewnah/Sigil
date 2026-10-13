import * as fs from "node:fs";
import * as path from "node:path";
import { fileURLToPath } from "node:url";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";

const m = vi.hoisted(() => ({ invoke: vi.fn(), isTauri: vi.fn(() => true) }));
vi.mock("@tauri-apps/api/core", () => ({ invoke: m.invoke, isTauri: () => m.isTauri() }));

import { exchangeOAuthCode, oauthClientSecretConfigured, oauthErrorMessage, refreshOAuthToken } from "../oauthTokens";
import { fetchGoogleFontsList } from "../googleFonts";

const ROOT = path.join(path.dirname(fileURLToPath(import.meta.url)), "..", "..", "..");

beforeEach(() => {
  m.isTauri.mockReturnValue(true);
});
afterEach(() => {
  vi.clearAllMocks();
});

describe("exchangeOAuthCode / refreshOAuthToken", () => {
  it("send the browser flow's results and the public client id to the oauth-token plugin, nothing else", async () => {
    m.invoke.mockResolvedValue({ accessToken: "fake-access", refreshToken: "fake-refresh", expiresIn: 60 });
    const tokens = await exchangeOAuthCode("kick", {
      clientId: "fake-client",
      code: "fake-code",
      codeVerifier: "fake-verifier",
      redirectUri: "http://localhost:17890/oauth/kick/callback",
    });
    expect(tokens).toEqual({ accessToken: "fake-access", refreshToken: "fake-refresh", expiresIn: 60 });
    expect(m.invoke).toHaveBeenCalledWith("plugin:oauth-token|oauth_token_exchange", {
      provider: "kick",
      clientId: "fake-client",
      code: "fake-code",
      codeVerifier: "fake-verifier",
      redirectUri: "http://localhost:17890/oauth/kick/callback",
    });

    await refreshOAuthToken("twitch", { clientId: "fake-client", refreshToken: "fake-refresh" });
    expect(m.invoke).toHaveBeenLastCalledWith("plugin:oauth-token|oauth_token_refresh", {
      provider: "twitch",
      clientId: "fake-client",
      refreshToken: "fake-refresh",
    });
  });

  it("reject with what Rust says (a string), which oauthErrorMessage turns into text", async () => {
    m.invoke.mockRejectedValue("Kick token request: HTTP 400 Bad Request: invalid_grant");
    const err = await exchangeOAuthCode("kick", { clientId: "c", code: "x", codeVerifier: "v", redirectUri: "r" }).catch((e) => e);
    expect(oauthErrorMessage(err)).toBe("Kick token request: HTTP 400 Bad Request: invalid_grant");
    expect(oauthErrorMessage(new Error("boom"))).toBe("boom");
    expect(oauthErrorMessage(42)).toBe("42");
  });
});

describe("oauthClientSecretConfigured", () => {
  it("asks Rust and returns a strict boolean", async () => {
    m.invoke.mockResolvedValue(true);
    await expect(oauthClientSecretConfigured("kick")).resolves.toBe(true);
    expect(m.invoke).toHaveBeenCalledWith("plugin:oauth-token|oauth_client_secret_configured", { provider: "kick" });
    m.invoke.mockResolvedValue(false);
    await expect(oauthClientSecretConfigured("twitch")).resolves.toBe(false);
    m.invoke.mockResolvedValue("yes"); // anything that is not exactly true counts as not configured
    await expect(oauthClientSecretConfigured("kick")).resolves.toBe(false);
  });

  it("is false outside the Tauri host and when the command fails, without calling it", async () => {
    m.isTauri.mockReturnValue(false);
    await expect(oauthClientSecretConfigured("kick")).resolves.toBe(false);
    expect(m.invoke).not.toHaveBeenCalled();
    m.isTauri.mockReturnValue(true);
    m.invoke.mockRejectedValue("not allowed");
    await expect(oauthClientSecretConfigured("kick")).resolves.toBe(false);
  });
});

describe("fetchGoogleFontsList", () => {
  it("returns the items of the Rust answer and takes no key", async () => {
    m.invoke.mockResolvedValue({ items: [{ family: "Roboto" }] });
    await expect(fetchGoogleFontsList<{ family: string }>()).resolves.toEqual([{ family: "Roboto" }]);
    expect(m.invoke).toHaveBeenCalledWith("plugin:google-fonts|google_fonts_list");
  });

  it("rejects an answer without a list and passes Rust's message through", async () => {
    m.invoke.mockResolvedValue({});
    await expect(fetchGoogleFontsList()).rejects.toThrow(/empty/);
    m.invoke.mockRejectedValue("The Google Fonts list is not available in this build");
    await expect(fetchGoogleFontsList()).rejects.toBe("The Google Fonts list is not available in this build");
  });
});

describe("IPC contract with the Rust plugins", () => {
  /** Every `plugin:<name>|<command>` the two wrappers can call. */
  function calls(file: string): { plugin: string; command: string }[] {
    const text = fs.readFileSync(path.join(ROOT, file), "utf8");
    const plugins = [...text.matchAll(/plugin:([a-z-]+)/g)].map((x) => x[1]!);
    const out: { plugin: string; command: string }[] = [];
    for (const c of text.matchAll(/\|([a-z][a-z0-9_]*)[`"]/g)) {
      // The wrapper names its plugin once (a const or in the string itself).
      out.push({ plugin: plugins[0]!, command: c[1]! });
    }
    return out;
  }

  it.each(["src/utils/oauthTokens.ts", "src/utils/googleFonts.ts"])("%s only calls commands the plugin lists, allows and registers", (file) => {
    const found = calls(file);
    expect(found.length).toBeGreaterThan(0);
    const mainRs = fs.readFileSync(path.join(ROOT, "src-tauri/src/main.rs"), "utf8");
    for (const { plugin, command } of found) {
      const list = fs.readFileSync(path.join(ROOT, "src-tauri/plugin-commands", `${plugin}.txt`), "utf8");
      const listed = list.split(/\r?\n/).map((l) => l.split("#")[0]!.trim()).filter(Boolean);
      expect(listed, `${plugin}.txt lists ${command}`).toContain(command);
      const capability = JSON.parse(fs.readFileSync(path.join(ROOT, "src-tauri/capabilities", `${plugin}.json`), "utf8"));
      expect(capability.permissions).toContain(`${plugin}:default`);
      expect(capability.windows).toEqual(["main"]);
      const module = plugin.replace(/-/g, "_");
      expect(mainRs, `${plugin} is registered in main.rs`).toContain(`.plugin(services::${module}::init())`);
    }
  });

  it("the Rust commands and the plugin lists agree", () => {
    for (const [plugin, source] of [
      ["oauth-token", "oauth_token.rs"],
      ["google-fonts", "google_fonts.rs"],
    ] as const) {
      const rs = fs.readFileSync(path.join(ROOT, "src-tauri/src/services", source), "utf8");
      const handler = rs.match(/generate_handler!\[([^\]]*)\]/)![1]!.split(",").map((s) => s.trim()).filter(Boolean);
      const list = fs
        .readFileSync(path.join(ROOT, "src-tauri/plugin-commands", `${plugin}.txt`), "utf8")
        .split(/\r?\n/)
        .map((l) => l.split("#")[0]!.trim())
        .filter(Boolean);
      expect([...handler].sort()).toEqual([...list].sort());
    }
  });
});
