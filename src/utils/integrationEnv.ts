/**
 * Integration env vars (`SIGIL_*` preferred). `CURSES_*` is still read as a fallback for older `.env` files.
 *
 * Only PUBLIC values live here: OAuth client ids, redirect overrides and the debug flag, the names under
 * `public` in scripts/release/env-classification.json (vite.config.ts exposes exactly those). Client
 * secrets and API keys are never in the bundle; they are compiled into the Rust binary and used by the
 * `oauth-token` and `google-fonts` plugins (see src/utils/oauthTokens.ts). Do not add a secret getter here.
 */

const env = import.meta.env as Record<string, string | undefined>;

function firstEnv(...keys: string[]): string {
  for (const k of keys) {
    const v = env[k];
    if (v !== undefined && String(v).trim() !== "")
      return String(v).trim();
  }
  return "";
}

function firstEnvRaw(...keys: string[]): string | undefined {
  for (const k of keys) {
    const v = env[k];
    if (v !== undefined && String(v).trim() !== "")
      return String(v);
  }
  return undefined;
}

export function getTwitchClientId(): string {
  return firstEnv("SIGIL_TWITCH_CLIENT_ID", "CURSES_TWITCH_CLIENT_ID");
}

/** Browser host redirect for Twitch implicit (Public apps). */
export function getTwitchRedirectImplicitUri(): string {
  if (import.meta.env.MODE === "development")
    return "http://localhost:1420/oauth/twitch/implicit";
  const fromEnv = firstEnvRaw(
    "SIGIL_TWITCH_IMPLICIT_REDIRECT_LOCAL",
    "CURSES_TWITCH_IMPLICIT_REDIRECT_LOCAL"
  );
  if (fromEnv)
    return fromEnv;
  if (typeof window !== "undefined" && window.location?.origin)
    return `${window.location.origin}/oauth/twitch/implicit`;
  return "http://localhost:1420/oauth/twitch/implicit";
}

/** Browser host redirect for Twitch auth code (Confidential) or Kick. */
export function getTwitchRedirectUri(): string {
  if (import.meta.env.MODE === "development")
    return "http://localhost:1420/oauth/twitch/callback";
  const fromEnv = firstEnvRaw(
    "SIGIL_TWITCH_CLIENT_REDIRECT_LOCAL",
    "CURSES_TWITCH_CLIENT_REDIRECT_LOCAL"
  );
  if (fromEnv)
    return fromEnv;
  if (typeof window !== "undefined" && window.location?.origin)
    return `${window.location.origin}/oauth/twitch/callback`;
  return "http://localhost:1420/oauth/twitch/callback";
}

export function getKickClientId(): string {
  return firstEnv("SIGIL_KICK_CLIENT_ID", "CURSES_KICK_CLIENT_ID");
}
