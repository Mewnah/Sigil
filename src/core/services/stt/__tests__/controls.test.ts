import { describe, expect, it } from "vitest";
import { ServiceNetworkState } from "@/types";
import { sttButtonStatus, sttControlAction } from "../controls";

describe("what the Start / Stop controls offer for speech-to-text", () => {
  it("offers Start when stopped and Stop when running", () => {
    expect(sttControlAction({ status: ServiceNetworkState.disconnected, retrying: false })).toBe("start");
    expect(sttControlAction({ status: ServiceNetworkState.connected, retrying: false })).toBe("stop");
    expect(sttControlAction({ status: ServiceNetworkState.connected, retrying: true })).toBe("stop");
  });

  it("offers Stop while STT retries by itself, however long the status keeps reading 'connecting'", () => {
    // A first start tried again, a reconnect after an outage: there is no end, and the settings are locked meanwhile.
    expect(sttControlAction({ status: ServiceNetworkState.connecting, retrying: true })).toBe("stop");
  });

  it("leaves an ordinary start in progress to finish (the connecting watchdog bounds it)", () => {
    expect(sttControlAction({ status: ServiceNetworkState.connecting, retrying: false })).toBe("wait");
    expect(sttControlAction({ status: ServiceNetworkState.error, retrying: false })).toBe("wait");
  });

  it("renders a retry wait with the Stop button of a running session, everything else as it is", () => {
    expect(sttButtonStatus({ status: ServiceNetworkState.connecting, retrying: true })).toBe(ServiceNetworkState.connected);
    expect(sttButtonStatus({ status: ServiceNetworkState.connecting, retrying: false })).toBe(ServiceNetworkState.connecting);
    expect(sttButtonStatus({ status: ServiceNetworkState.disconnected, retrying: false })).toBe(ServiceNetworkState.disconnected);
    expect(sttButtonStatus({ status: ServiceNetworkState.connected, retrying: false })).toBe(ServiceNetworkState.connected);
  });
});
