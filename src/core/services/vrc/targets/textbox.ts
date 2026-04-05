import { IVRCTarget } from "../types";
import { sendOsc, type OscArg }    from "./utils";
import throttle       from "lodash/throttle";

/** VRChat chatbox text limit (see VRChat OSC docs). */
const VRCHAT_CHATBOX_MAX_CHARS = 144;

class VRC_TextboxTarget implements IVRCTarget {
  get state() {
    return window.ApiServer.state.services.vrc.data.textbox;
  }

  #sendIndicator(value: boolean) {
    sendOsc("/chatbox/typing", [{ Bool: value }]);
  }

  #sendIndicatorThrottled = throttle(this.#sendIndicator, 1400);

  pushFinal(value: string): void {
    if (this.state.indicator) {
      this.#sendIndicatorThrottled.cancel();
      sendOsc("/chatbox/typing", [{ Bool: false }]);
    }
    const text = value.slice(0, VRCHAT_CHATBOX_MAX_CHARS);
    const args: OscArg[] = [{ String: text }, { Bool: true }];
    if (!this.state.playNotificationSound) args.push({ Bool: false });
    sendOsc("/chatbox/input", args);
  }

  pushInterim(_value: string): void {
    if (this.state.indicator)
      this.#sendIndicatorThrottled(true);
  }
  cancel(): void {}
}

export default VRC_TextboxTarget;
