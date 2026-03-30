import { isTauri } from "@tauri-apps/api/core";
import { listen, type UnlistenFn } from "@tauri-apps/api/event";

export const SIGIL_OAUTH_EVENT = "sigil-oauth";

export type OAuthProvider = "twitch" | "kick";

export type SigilOAuthPayload = {
  provider: OAuthProvider;
  code?: string;
  access_token?: string;
  state?: string;
  error?: string;
  error_description?: string;
};

export async function listenOAuthPayload(
  provider: OAuthProvider,
  handler: (p: SigilOAuthPayload) => void
): Promise<UnlistenFn> {
  return listen<SigilOAuthPayload>(SIGIL_OAUTH_EVENT, (e) => {
    if (e.payload.provider === provider)
      handler(e.payload);
  });
}

export function isOAuthTauri(): boolean {
  return isTauri();
}
