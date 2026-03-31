import { IServiceInterface, ObsCaptionEnvelope } from "@/types";
import { proxy } from "valtio";
import { ObsBrowserSourceService } from "./browserSource";
import { ObsCaptionFeedService } from "./captionFeed";
import { ObsControlService } from "./control";
import { OBS_State } from "./schema";

class Service_OBS implements IServiceInterface {
  private readonly control = new ObsControlService(() => this.#state);
  private readonly browserSource = new ObsBrowserSourceService();
  private readonly feed = new ObsCaptionFeedService(
    () => this.#state,
    (envelope) => this.publishCaptionEnvelope(envelope),
  );

  browserState = proxy({
    mode: "idle" as "idle" | "synced" | "failed",
    lastError: "",
  });

  wsState = this.control.wsState;

  get #state() {
    return window.ApiServer.state.services.obs.data;
  }

  private publishCaptionEnvelope(envelope: ObsCaptionEnvelope) {
    window.ApiShared.pubsub.publish("obs.caption", envelope);
  }

  private migrateLegacyBrowserFields(state: OBS_State) {
    // One-time in-memory compatibility: old native caption fields become browser captions defaults.
    if (!state.browserSource && state.source) {
      state.browserSource = state.source;
    }
    if (!state.browserInputField && state.inputField) {
      state.browserInputField = state.inputField;
    }
    if (!state.browserInterim && state.interim) {
      state.browserInterim = state.interim;
    }
    if (!state.browserCaptionsEnable && state.captionsEnable) {
      state.browserCaptionsEnable = true;
    }
    if ("browserMode" in state) {
      delete (state as Record<string, unknown>).browserMode;
    }
  }

  async init() {
    this.migrateLegacyBrowserFields(this.#state);
    this.control.init();
    this.feed.init();
  }

  getObsBrowserSourceLink() {
    return this.browserSource.getObsBrowserSourceLink();
  }

  async wsConnect() {
    await this.control.wsConnect();
  }

  wsDisconnect() {
    this.control.wsDisconnect();
  }

  wsCancel() {
    this.control.wsCancel();
  }

  async setupObsScene({
    name,
    port,
    password,
  }: {
    name: string;
    port: string;
    password: string;
  }) {
    return this.browserSource.setupObsScene({
      name,
      port,
      password,
    });
  }

  dispose() {
    this.control.dispose();
    this.feed.dispose();
  }
}

export default Service_OBS;
