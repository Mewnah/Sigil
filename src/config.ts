import { invoke, isTauri } from "@tauri-apps/api/core";
import { clientServerAddress } from "@/utils/localPages";

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
}
type ServerNetwork = {
  ip: string,
  host: string,
  port: string
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

export default AppConfiguration;
