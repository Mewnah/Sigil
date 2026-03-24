
import ReactDOM from "react-dom/client";
import "./style.css";
import ApiServer from "./core";
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
    window.Config = new AppConfiguration();
    window.ApiShared = new ApiShared();
    window.ApiClient = new ApiClient();

    await window.Config.init();
    await window.ApiShared.init();

    if (window.Config.isClient()) {
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
      root_ele.innerHTML = `
        <div style="padding: 20px; color: red; background: #333; height: 100vh;">
          <h1>Application Failed to Start</h1>
          <pre>${error?.message || String(error)}\n${error?.stack || ''}</pre>
        </div>
      `;
    }
  }
})();
