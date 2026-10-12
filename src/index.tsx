
import ReactDOM from "react-dom/client";
import "./style.css";
import type ApiServer from "./core";
import ApiClient from "./client";
import ClientView from "./client/ui/view";
import React, { ReactNode, Suspense } from "react";
import AppConfiguration from "@/config";
import ApiShared from "@/shared";
import ClientLoadingView from "./client/ui/view_loading";
import { ClientErrorBoundary, scheduleClientReload } from "./client/ui/clientRecovery";
import { handleStartupFailure, isObsBrowserSource } from "./client/ui/clientStartup";
import { changeLanguage, initI18n } from "@/i18n";
import { clientQueryWithoutForeignServer } from "@/utils/localPages";

declare global {
  interface Window {
    Config: AppConfiguration,
    ApiShared: ApiShared;
    ApiServer: ApiServer;
    ApiClient: ApiClient;
    reactRoot: ReactDOM.Root;
  }
}
window.global ||= window;

/**
 * Caption window (P6-B; Rust `caption_window::CAPTIONS_ROUTE`, `src/overlay/frame.ts` `CAPTION_WINDOW_PATH`): an early
 * route that only renders the captions the main window sends. It returns before AppConfiguration, ApiShared, ApiClient
 * or ApiServer are created, so the window never boots the host (no settings file, no PeerJS server, no services).
 */
const isCaptionWindow = window.location.pathname === "/captions" || window.location.pathname.startsWith("/captions/");

/** Canvas mirror (/client) is often opened in OBS’s embedded browser (older CEF than Chrome). Mark early so CSS fallbacks apply before paint. */
if (window.location.pathname.startsWith("/client")) {
  pinClientToServingHost();
  document.documentElement.setAttribute("data-sigil-client", "1");
  // OBS browser source can hold stale service workers across builds; disable SW control for mirror pages.
  if ("serviceWorker" in navigator) {
    navigator.serviceWorker.getRegistrations().then(regs => {
      regs.forEach(reg => reg.unregister());
    }).catch(() => {});
  }
}

// prevent rightclicks
window.addEventListener('contextmenu', e => {
  const ele = e.target as HTMLElement;
  if (ele.nodeName !== "INPUT" && ele.nodeName !== "TEXTAREA") {
    e.preventDefault();
    return false;
  }
}, false);

let root_ele = document.getElementById("root");
if (!root_ele)
  throw Error("Root not found");

// Singleton root pattern for HMR
if (!window.reactRoot) {
  window.reactRoot = ReactDOM.createRoot(root_ele);
}

// Set default theme
document.documentElement.setAttribute("data-theme", "sigil-dark");

function renderView(view: ReactNode) {
  window.reactRoot && window.reactRoot.render(view);
}

import SigilRoot from "./core/ui/sigil/SigilRoot";

(async function () {
  if (isCaptionWindow) {
    try {
      const { mountCaptionWindow } = await import("./overlay");
      mountCaptionWindow(window.reactRoot);
    } catch (error) {
      // Nothing to show: the window stays transparent; the reason is in this window's console.
      console.error("[captions] the caption window failed to start", error);
    }
    return;
  }
  try {
    const { runOAuthCallbackIfNeeded } = await import("./oauth/bootstrap");
    if (await runOAuthCallbackIfNeeded())
      return;

    window.Config = new AppConfiguration();
    window.ApiShared = new ApiShared();
    window.ApiClient = new ApiClient();

    await window.Config.init();
    await window.ApiShared.init();

    if (window.Config.isClient()) {
      await initI18n("en");
      renderView(<ClientLoadingView />);
    }

    // Initialize ApiClient (always needed)
    await window.ApiClient.init();

    // Determine Mode & Initialize Server API if needed
    if (window.Config.isClient()) {
      const hostLang = window.Config.clientInitialState?.uiLanguage ?? "en";
      await changeLanguage(hostLang);
      renderView(<ClientErrorBoundary><ClientView /></ClientErrorBoundary>);
    } else {
      // Assume Server/App Mode if not explicitly Client
      if (!window.ApiServer) {
        const serverApi = await import("./core");
        window.ApiServer = new serverApi.default();
        await window.ApiServer.init();
        // Settings and i18n are ready: sync the LAN Origin-check kill switch to Rust and surface
        // web-server failures / refused connections (no-op outside the desktop host).
        void window.ApiShared.pubsub.attachHostWebServer(window.ApiServer.state);
      }

      if (window.Config.isServer()) {
        document.documentElement.className = "host";
      }

      renderView(<SigilRoot />);
    }
  } catch (error: any) {
    console.error("App initialization failed:", error);
    // Rendered through the React root: a direct DOM write would be wiped by the pending commit of
    // the loading view, and JSX escapes the /client ?host=/?port= values (no HTML injection).
    // What shows, and whether the page reloads itself, depends on the failure (a connection that
    // merely fails is retried in place and never gets here): see clientStartup.tsx.
    handleStartupFailure(error, {
      isClient: window.Config?.isClient?.() === true,
      clientNetwork: window.Config?.clientNetwork,
      obs: isObsBrowserSource(),
      renderView,
      scheduleReload: scheduleClientReload,
    });
  }
})();

/**
 * LAN security (P0-I): a /client page only connects to the Sigil server that served it (or to
 * loopback). A foreign `?host=` (and its `?port=`) is removed from the URL before AppConfiguration
 * reads it, so a website cannot point the mirror, which renders what its server sends, at its own
 * PeerJS server. See `clientServerAddress` in src/utils/localPages.ts.
 */
function pinClientToServingHost() {
  const { pathname, search, hash, hostname, port } = window.location;
  const cleaned = clientQueryWithoutForeignServer(search, { hostname, port });
  if (cleaned === null)
    return;
  console.warn("[client] ignoring ?host=/?port= that do not point at the server that served this page");
  // Same-origin path, so this does not throw in practice; if it ever did, startup stops here
  // (fails closed) instead of connecting to the foreign host.
  window.history.replaceState(window.history.state, "", `${pathname}${cleaned}${hash}`);
}
