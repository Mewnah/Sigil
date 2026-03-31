import { ObsCaptionEnvelope, ServiceNetworkState, TextEvent, TextEventSource } from "@/types";

export enum ObsCaptionOutputMode {
  plain = "plain",
  styled = "styled",
}

export type ObsClientState = "idle" | "connecting" | "synced" | "reconnecting" | "failed";

export type ObsCaptionInputSettings = {
  source: TextEventSource;
  includeInput: boolean;
  includeInterim: boolean;
};

export interface IObsCaptionFeedTarget {
  publish(event: ObsCaptionEnvelope): void;
}

export interface IObsControlStateTarget {
  status: ServiceNetworkState;
}

export interface IObsTextEventTarget {
  process(data?: TextEvent): void;
}