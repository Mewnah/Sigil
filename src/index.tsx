
import ReactDOM from "react-dom/client";
import "./style.css";
import type ApiServer from "./core";
import ApiClient from "./client";
import ClientView from "./client/ui/view";
import React, { ReactNode, Suspense } from "react";
import AppConfiguration from "@/config";
import ApiShared from "@/shared";
import ClientLoadingView from "./client/ui/view_loading";

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
      // PWA: precaches production assets; update handler below avoids stale client bundle after host rebuilds.
      if ('serviceWorker' in navigator) {
        navigator.serviceWorker.register('/sw.js', { scope: '/', type: 'classic' }).then((sw) => {
          sw.addEventListener("updatefound", async _ => {
            console.log("found update");
            await sw.update();
            location.reload();
          })
        });
      }
    }

    // Initialize ApiClient (always needed)
    await window.ApiClient.init();

    // Determine Mode & Initialize Server API if needed
    if (window.Config.isClient()) {
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
        ? `<p style="color:#ccc;margin:12px 0;">Remote client could not connect to the Sigil host. Check that the desktop app is running on this machine or your LAN, then open the client link from the host again (includes <code>host</code>, <code>port</code>, and <code>id</code> in the URL).</p>
           <p style="color:#888;font-size:13px;">Trying: <code>${cn.host}:${cn.port}</code></p>`
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
