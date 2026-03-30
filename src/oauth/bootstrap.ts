import type { OAuthProvider, SigilOAuthPayload } from "@/utils/oauthPopup";

/**
 * OAuth redirect targets (register these exact URLs in Twitch / Kick developer consoles).
 * Handled before full app init so callback windows/webviews stay lightweight.
 */

const TWITCH_IMPLICIT_TAURI_HINT =
  `<p style="font-family:system-ui,sans-serif;padding:24px;max-width:520px;color:#e4e4e7;">` +
  `Sign-in was sent to Sigil. For desktop, register ` +
  `<code style="background:#27272a;padding:2px 6px;border-radius:4px;">http://localhost:17890/oauth/twitch/implicit</code> ` +
  `for Public Twitch apps.</p>`;

async function tauriEmitOAuthAndClose(payload: SigilOAuthPayload, hintIfNoClose: string): Promise<void> {
  const { emit } = await import("@tauri-apps/api/event");
  await emit("sigil-oauth", payload);
  try {
    const { getCurrentWebviewWindow } = await import("@tauri-apps/api/webviewWindow");
    await getCurrentWebviewWindow().close();
    return;
  } catch {
    /* main window or close not allowed */
  }
  document.body.innerHTML = hintIfNoClose;
}

function parseTwitchImplicitParams(): {
  access_token: string;
  state: string;
  error: string;
  error_description: string;
} {
  const q = new URLSearchParams(location.search);
  if (q.get("error")) {
    return {
      error: q.get("error") ?? "",
      error_description: q.get("error_description") ?? "",
      state: q.get("state") ?? "",
      access_token: "",
    };
  }
  const h = new URLSearchParams(location.hash.replace(/^#/, ""));
  return {
    access_token: h.get("access_token") ?? "",
    state: h.get("state") ?? "",
    error: h.get("error") ?? "",
    error_description: h.get("error_description") ?? "",
  };
}

export async function runOAuthCallbackIfNeeded(): Promise<boolean> {
  const path = (location.pathname.replace(/\/+$/, "") || "/").toLowerCase();
  const { isTauri } = await import("@tauri-apps/api/core");

  if (path.endsWith("/oauth/twitch/implicit")) {
    const { access_token, state, error, error_description } = parseTwitchImplicitParams();

    if (isTauri()) {
      if (error)
        await tauriEmitOAuthAndClose(
          { provider: "twitch", error, error_description, state: state || undefined },
          TWITCH_IMPLICIT_TAURI_HINT
        );
      else if (access_token && state)
        await tauriEmitOAuthAndClose({ provider: "twitch", access_token, state }, TWITCH_IMPLICIT_TAURI_HINT);
      else
        await tauriEmitOAuthAndClose(
          {
            provider: "twitch",
            error: "invalid_callback",
            error_description: "Missing access token or state",
          },
          TWITCH_IMPLICIT_TAURI_HINT
        );
      return true;
    }

    if (window.opener && !window.opener.closed) {
      window.opener.postMessage(
        {
          source: "sigil-oauth",
          provider: "twitch",
          flow: "implicit",
          access_token: access_token || undefined,
          state: state || undefined,
          error: error || undefined,
          error_description: error_description || undefined,
        },
        "*"
      );
      window.close();
      return true;
    }

    document.body.innerHTML =
      `<p style="font-family:system-ui,sans-serif;padding:24px;max-width:480px;">` +
      `Sign-in finished. If Sigil did not update, close this tab and try again from the app.</p>`;
    return true;
  }

  let provider: OAuthProvider | null = null;
  if (path.endsWith("/oauth/twitch/callback")) provider = "twitch";
  else if (path.endsWith("/oauth/kick/callback")) provider = "kick";
  else return false;

  const params = new URLSearchParams(location.search);
  const code = params.get("code");
  const state = params.get("state");
  const error = params.get("error");
  const error_description = params.get("error_description") ?? "";

  const desktopHint =
    `<p style="font-family:system-ui,sans-serif;padding:24px;max-width:520px;color:#e4e4e7;">` +
    `Sign-in was sent to Sigil. If the app did not connect, use Twitch/Kick redirect ` +
    `<code style="background:#27272a;padding:2px 6px;border-radius:4px;">http://localhost:17890/oauth/${provider}/callback</code> ` +
    `for the desktop app (not the dev server URL).</p>`;

  if (isTauri()) {
    if (error)
      await tauriEmitOAuthAndClose(
        { provider, error, error_description, state: state ?? undefined },
        desktopHint
      );
    else if (code && state)
      await tauriEmitOAuthAndClose({ provider, code, state }, desktopHint);
    else
      await tauriEmitOAuthAndClose(
        { provider, error: "invalid_callback", error_description: "Missing code or state" },
        desktopHint
      );
    return true;
  }

  if (window.opener && !window.opener.closed) {
    if (provider === "twitch") {
      if (error)
        window.opener.postMessage(`smplstt_tw_auth_error:${error}:${error_description}`, "*");
      else if (code && state)
        window.opener.postMessage(`smplstt_tw_auth:${code}:${state}`, "*");
    } else {
      if (error)
        window.opener.postMessage(`smplstt_kick_auth_error:${error}`, "*");
      else if (code && state)
        window.opener.postMessage(`smplstt_kick_auth:${code}:${state}`, "*");
    }
    window.close();
    return true;
  }

  document.body.innerHTML =
    `<p style="font-family:system-ui,sans-serif;padding:24px;max-width:480px;">` +
    `Sign-in finished. If Sigil did not update, close this tab and try again from the app ` +
    `(desktop: use the in-app login button; browser host: allow pop-ups for this site).</p>`;
  return true;
}
