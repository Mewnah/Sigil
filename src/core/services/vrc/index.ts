import { invoke } from "@tauri-apps/api/core";
import { IServiceInterface, TextEventType } from "@/types";
import { pushSystemLog } from "@/core/services/systemLog";
import { serviceSubscibeToInput, serviceSubscibeToSource } from "../../../utils";
import { VRC_Backends } from "./schema";
import VRC_KillFrenzyTarget from "./targets/killfrenzy";
import VRC_TextboxTarget from "./targets/textbox";
import { sendOsc as sendOscPacket, type OscArg } from "./targets/utils";
import { IVRCTarget } from "./types";

class Service_VRC implements IServiceInterface {
  get #state() {
    return window.ApiServer.state.services.vrc.data;
  }

  targets: Record<VRC_Backends, IVRCTarget | undefined> = {
    [VRC_Backends.textbox]: undefined,
    [VRC_Backends.killfrenzy]: undefined
  }

  private eventDisposers: (() => void)[] = [];

  get #activeTarget() {
    return this.targets[this.#state.target];
  }

  async init() {
    this.targets.textbox = new VRC_TextboxTarget();
    this.targets.killfrenzy = new VRC_KillFrenzyTarget();
    this.eventDisposers.push(serviceSubscibeToSource(this.#state, "source", (data) => {
      if (!this.#state.enable || !data) return;
      if (data.type === TextEventType.final) {
        this.#activeTarget?.pushFinal(data.value);
      } else {
        this.#activeTarget?.pushInterim(data.value);
      }
    }));
    this.eventDisposers.push(serviceSubscibeToInput(this.#state, "inputField", (data) => {
      if (!this.#state.enable || !data) return;
      if (data.type === TextEventType.final) {
        this.#activeTarget?.pushFinal(data.value);
      } else {
        this.#activeTarget?.pushInterim(data.value);
      }
    }));

    void invoke<{ ok: boolean; error?: string }>("plugin:osc|bind_status")
      .then((s) => {
        if (!s.ok && s.error)
          pushSystemLog("VRChat OSC", s.error, "error");
      })
      .catch(() => {});
  }
  sendTest(value: string): void {
    this.#activeTarget?.pushFinal(value);
  }

  sendOsc(path: string, args: OscArg[]) {
    sendOscPacket(path, args);
  }

  dispose() {
    this.eventDisposers.forEach(d => d());
    this.eventDisposers = [];
  }
}

export default Service_VRC;
