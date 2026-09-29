
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
    const isClient = window.Config?.isClient?.() === true;
    const cn = window.Config?.clientNetwork;
    // Rendered through the React root: a direct DOM write would be wiped by the pending commit of
    // the loading view, and JSX escapes the /client ?host=/?port= values (no HTML injection).
    renderView(<StartupErrorView error={error} client={isClient && cn ? { host: cn.host, port: cn.port } : undefined} />);
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

/** Shown when initialization fails; every dynamic value is rendered as text. */
function StartupErrorView({ error, client }: { error: any; client?: { host: string; port: string } }) {
  const address = client ? `${client.host}:${client.port}` : "";
  return (
    <div style={{ padding: 24, color: "#f87171", background: "#1a1a1a", height: "100vh", fontFamily: "system-ui, sans-serif" }}>
      <h1 style={{ color: "#fff", marginTop: 0 }}>Application failed to start</h1>
      {client && (
        <>
          <p style={{ color: "#ccc", margin: "12px 0" }}>
            The OBS browser source could not connect to Sigil (PeerJS). Keep the desktop app running on this PC, use{" "}
            <code>{`http://${address}/client`}</code> (same pattern as Curses), and check that nothing blocks WebSockets
            to that host and port.
          </p>
          <p style={{ color: "#888", fontSize: 13 }}>
            Trying Peer server: <code>{address}</code>
          </p>
        </>
      )}
      <pre style={{ color: "#fca5a5", whiteSpace: "pre-wrap", fontSize: 12 }}>
        {[String(error?.message || error), String(error?.stack || "")].join("\n")}
      </pre>
    </div>
  );
}
