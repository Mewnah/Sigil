import { ServiceNetworkState } from "@/types";

/** What a Start / Stop control of the UI does with speech-to-text. */
export type SttControlAction = "start" | "stop" | "wait";

/** The part of `Service_STT.serviceState` the controls read. */
export interface SttControlState {
  status: ServiceNetworkState;
  /** STT is waiting between attempts of an automatic retry (a first start tried again, a reconnect, a restart). */
  retrying: boolean;
}

/**
 * What a Start / Stop control offers for `state`.
 *
 * - `disconnected`: Start.
 * - `connected`: Stop.
 * - `connecting` while STT retries by itself: Stop. The retry has no end of its own (it goes on until the user stops
 *   it), and the settings are locked meanwhile, so without a Stop here a streamer could neither end it nor fix the
 *   microphone or the key.
 * - any other `connecting` (a start in progress: models loading, a first connection): wait. Those end by themselves,
 *   and the connecting watchdog stops one that never does.
 */
export function sttControlAction(state: SttControlState): SttControlAction {
  if (state.status === ServiceNetworkState.disconnected) return "start";
  if (state.status === ServiceNetworkState.connected) return "stop";
  if (state.status === ServiceNetworkState.connecting && state.retrying) return "stop";
  return "wait";
}

/**
 * The status a Start / Stop button should render for `state`: a retry wait reads as running (Stop), the way a
 * session that is up does, while the status line next to it says STT is reconnecting.
 */
export function sttButtonStatus(state: SttControlState): ServiceNetworkState {
  return sttControlAction(state) === "stop" ? ServiceNetworkState.connected : state.status;
}
