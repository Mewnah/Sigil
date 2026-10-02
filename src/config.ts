import { invoke, isTauri } from "@tauri-apps/api/core";
import { clientServerAddress } from "@/utils/localPages";
import { lanTokenFromQuery } from "@/shared/services/peer/lanAccess";

enum AppMode {
  server,
  client
}

enum AppPlatform {
  app,
  web
}

type NativeFeatures = {
  background_input: boolean
}

type InitialData = {
  scene: string
  /** Host UI language so the /client mirror can match Settings (omitted by older hosts). */
  uiLanguage?: string
}

type ClientNetwork = {
  serverId: string;
  host: string;
  port: string;
  /**
   * The host's LAN token from `?token=` (P6-D; normalized, "" when absent or malformed). Sent as the
   * PeerJS key; only needed when this mirror runs on another machine than the host.
   */
  token: string;
}
/** An IPv4 address of the host PC that other devices may reach it on (`plugin:web|config`). */
export type LanAddress = {
  ip: string;
  /** Network adapter name. */
  interface: string;
}
type ServerNetwork = {
  ip: string,
  host: string,
  port: string,
  /** Candidate addresses for LAN links, the primary (`ip`) first (P6-D). */
  lanAddresses: LanAddress[],
}

class AppConfiguration {
  mode: AppMode = AppMode.server;
  platform: AppPlatform = AppPlatform.app;
  clientInitialState?: InitialData;
  features: NativeFeatures = {
    background_input: false
  }
  serverNetwork!: ServerNetwork;
  clientNetwork!: ClientNetwork;

  public isApp = () => window.Config.platform === AppPlatform.app;
  public isWeb = () => window.Config.platform === AppPlatform.web;
  public isClient = () => window.Config.mode === AppMode.client;
  public isServer = () => window.Config.mode === AppMode.server;

  // region ---INITIALIZERS---
  private loadBase() {
    this.platform = isTauri() ? AppPlatform.app : AppPlatform.web;
    this.mode = window.location.pathname.startsWith('/client') ? AppMode.client : AppMode.server;
  }

  private async loadFeatures() {
    if (!this.isApp())
      return;
    this.features = await invoke<NativeFeatures>("get_native_features");
  }

  private async loadNetwork() {
    // client is always web
    // load network params from url query
    if (this.isClient()) {
      const q = new URLSearchParams(window.location.search.substring(1));
      // Only the server that served this page (or loopback); src/index.tsx already strips foreign ?host=/?port=.
      const { host, port } = clientServerAddress(window.location.search, window.location);
      this.clientNetwork = {
        serverId: q.get("id") ?? "",
        host,
        port,
        token: lanTokenFromQuery(q),
      };
    }
    // server is always app
    // load network params from rust
    else {
      const appConfig = await invoke<any>("plugin:web|config");
      this.serverNetwork = {
        ip: appConfig.local_ip,
        // Same as upstream Curses: advertise localhost so OBS/browser sources match `localhost:PORT/client`.
        host: "localhost",
        port: appConfig.port,
        lanAddresses: parseLanAddresses(appConfig.lan_addresses, appConfig.local_ip),
      };
    }
  }

  async init() {
    this.loadBase();
    await Promise.all([
      this.loadFeatures(),
      this.loadNetwork()
    ]);
  }
  //endregion

}

/** `lan_addresses` from `plugin:web|config`, validated; falls back to the primary address alone. */
function parseLanAddresses(raw: unknown, primary: unknown): LanAddress[] {
  const list = Array.isArray(raw)
    ? raw.filter((a): a is LanAddress => !!a && typeof a.ip === "string" && typeof a.interface === "string")
    : [];
  if (list.length === 0 && typeof primary === "string" && primary)
    return [{ ip: primary, interface: "" }];
  return list;
}

export default AppConfiguration;
