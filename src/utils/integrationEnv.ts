/**
 * Integration env vars (`SIGIL_*` preferred). `CURSES_*` is still read as a fallback for older `.env` files.
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

/** Set for Confidential Twitch apps (auth code + refresh). Omit for Public apps (implicit grant). */
export function getTwitchClientSecret(): string {
  return firstEnv("SIGIL_TWITCH_CLIENT_SECRET", "CURSES_TWITCH_CLIENT_SECRET");
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

export function getKickClientSecret(): string {
  return firstEnv("SIGIL_KICK_CLIENT_SECRET", "CURSES_KICK_CLIENT_SECRET");
}

export function getKickRedirectUri(): string {
  if (import.meta.env.MODE === "development")
    return "http://localhost:1420/oauth/kick/callback";
  const fromEnv = firstEnvRaw(
    "SIGIL_KICK_CLIENT_REDIRECT_LOCAL",
    "CURSES_KICK_CLIENT_REDIRECT_LOCAL"
  );
  if (fromEnv)
    return fromEnv;
  if (typeof window !== "undefined" && window.location?.origin)
    return `${window.location.origin}/oauth/kick/callback`;
  return "http://localhost:1420/oauth/kick/callback";
}
