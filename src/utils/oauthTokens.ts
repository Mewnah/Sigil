import { invoke, isTauri } from "@tauri-apps/api/core";
import type { OAuthProvider } from "@/utils/oauthPopup";

/**
 * OAuth token calls (Kick, Twitch) made by the Rust `oauth-token` plugin (`src-tauri/src/services/oauth_token.rs`).
 *
 * Kick's token endpoint needs the app's client secret for both the authorization-code exchange and the refresh
 * grant. The secret is compiled into the binary and added in Rust; it never reaches this code, the bundle or the
 * web server that serves the bundle. This side sends only what the browser flow produced (the code, the PKCE
 * verifier, the redirect URI, or the refresh token) plus the public client id, and gets the tokens back.
 *
 * Needs the Tauri host: in a plain browser (`pnpm dev` without the app) these reject, and
 * `oauthClientSecretConfigured` is `false`.
 */

const PLUGIN = "plugin:oauth-token";

export type OAuthTokens = {
  accessToken: string;
  refreshToken?: string;
  /** Seconds until the access token expires, when the provider says. */
  expiresIn?: number;
};

export type ExchangeCodeArgs = {
  clientId: string;
  code: string;
  /** The PKCE verifier whose S256 challenge went into the authorize URL. */
  codeVerifier: string;
  /** Exactly the redirect URI of the authorize request (Rust only accepts Sigil's two localhost callbacks). */
  redirectUri: string;
};

/** A message from the Rust side (commands reject with a string) or from a thrown Error. */
export function oauthErrorMessage(e: unknown): string {
  if (typeof e === "string") return e;
  if (e instanceof Error) return e.message;
  return String(e);
}

/** `authorization_code` grant. Rejects with a readable message (no secret, code or token in it). */
export function exchangeOAuthCode(provider: OAuthProvider, args: ExchangeCodeArgs): Promise<OAuthTokens> {
  return invoke<OAuthTokens>(`${PLUGIN}|oauth_token_exchange`, { provider, ...args });
}

/** `refresh_token` grant. */
export function refreshOAuthToken(
  provider: OAuthProvider,
  args: { clientId: string; refreshToken: string },
): Promise<OAuthTokens> {
  return invoke<OAuthTokens>(`${PLUGIN}|oauth_token_refresh`, { provider, ...args });
}

/**
 * Whether this build holds the provider's client secret (a boolean: the value never leaves Rust). Always
 * `false` outside the Tauri host or when the command fails.
 */
export async function oauthClientSecretConfigured(provider: OAuthProvider): Promise<boolean> {
  if (!isTauri()) return false;
  try {
    return (await invoke<boolean>(`${PLUGIN}|oauth_client_secret_configured`, { provider })) === true;
  } catch {
    return false;
  }
}
