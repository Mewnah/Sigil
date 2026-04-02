
import ReactDOM from "react-dom/client";
import "./style.css";
import type ApiServer from "./core";
import ApiClient from "./client";
import ClientView from "./client/ui/view";
import React, { ReactNode, Suspense } from "react";
import AppConfiguration from "@/config";
import ApiShared from "@/shared";
import ClientLoadingView from "./client/ui/view_loading";
import { changeLanguage, initI18n } from "@/i18n";

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

/** Canvas mirror (/client) is often opened in OBS’s embedded browser (older CEF than Chrome). Mark early so CSS fallbacks apply before paint. */
if (window.location.pathname.startsWith("/client")) {
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
      renderView(<ClientView />);
    } else {
      // Assume Server/App Mode if not explicitly Client
      if (!window.ApiServer) {
        const serverApi = await import("./core");
        window.ApiServer = new serverApi.default();
        await window.ApiServer.init();
      }

      if (window.Config.isServer()) {
        document.documentElement.className = "host";
      }

      renderView(<SigilRoot />);
    }
  } catch (error: any) {
    console.error("App initialization failed:", error);
    if (root_ele) {
      const isClient = window.Config?.isClient?.() === true;
      const cn = window.Config?.clientNetwork;
      const clientHint = isClient && cn
        ? `<p style="color:#ccc;margin:12px 0;">The OBS browser source could not connect to Sigil (PeerJS). Keep the desktop app running on this PC, use <code>http://${cn.host}:${cn.port}/client</code> (same pattern as Curses), and check that nothing blocks WebSockets to that host and port.</p>
           <p style="color:#888;font-size:13px;">Trying Peer server: <code>${cn.host}:${cn.port}</code></p>`
        : "";
      root_ele.innerHTML = `
        <div style="padding: 24px; color: #f87171; background: #1a1a1a; height: 100vh; font-family: system-ui, sans-serif;">
          <h1 style="color:#fff;margin-top:0;">Application failed to start</h1>
          ${clientHint}
          <pre style="color:#fca5a5;white-space:pre-wrap;font-size:12px;">${String(error?.message || error)}\n${String(error?.stack || "")}</pre>
        </div>
      `;
    }
  }
})();
