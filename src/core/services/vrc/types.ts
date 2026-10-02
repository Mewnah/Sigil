import type { TextEvent } from "@/types";

export interface PushFinalOptions {
  /** Typed text (keyboard input field): a complete message, never held for the end of a spoken turn. */
  typed?: boolean;
}

export interface IVRCTarget {
  /** A final line. `event` is the full `TextEvent` when known (Smart Turn verdict, language, tags). */
  pushFinal(value: string, event?: TextEvent, options?: PushFinalOptions): void;
  /** An interim line (speech in progress). */
  pushInterim(value: string, event?: TextEvent): void;
  /** Drop anything queued or held (target switched off or disposed). */
  cancel(): void;
}

// --- Rust `osc` plugin receiver (P6-C; osc.rs). Wire format is camelCase. ------------------------------------------

/** Tauri events of the `osc` plugin. */
export const OSC_EVENTS = {
  /** VRChat's MuteSelf changed ({@link RustOscMuteSelf}); deduplicated by Rust. */
  muteSelf: "osc:mute_self",
  /** Receiver state changed ({@link RustOscReceiverStatus}). */
  receiver: "osc:receiver",
  /** VRChat on this PC appeared or quit (OSCQuery browse; {@link RustOscVrchatPresence}). Not sent on restarts. */
  vrchat: "osc:vrchat",
} as const;

/** Argument of `plugin:osc|listen_start` (Rust `OscListenOptions`). */
export interface RustOscListenOptions {
  /** Where VRChat runs (the send host): loopback → OSCQuery or 127.0.0.1; another PC → 0.0.0.0, that PC only. */
  host: string;
  /** UDP port without OSCQuery (VRChat's default send port is 9001). */
  port: number;
  /** Advertise a free port with OSCQuery (VRChat on this PC only). */
  oscQuery: boolean;
  /** Rust closes the capture hub's STT gate at once when VRChat reports a mute. */
  muteSync: boolean;
}

export type RustOscReceiverMode = "off" | "oscQuery" | "port";

/** Rust `OscReceiverStatus`. */
export interface RustOscReceiverStatus {
  running: boolean;
  mode: RustOscReceiverMode;
  address: string | null;
  port: number | null;
  oscQueryPort: number | null;
  advertised: boolean;
  vrchatFound: boolean;
  muteSelf: boolean | null;
  packets: number;
  lastPacketAgoMs: number | null;
  error: string | null;
  warning: string | null;
}

/** Payload of {@link OSC_EVENTS.muteSelf} (Rust `MuteSelfEvent`). */
export interface RustOscMuteSelf {
  muted: boolean;
  /** `osc`: a message from VRChat; `oscQuery`: read from VRChat's OSCQuery server when Sigil found it. */
  origin: "osc" | "oscQuery";
}

/** Payload of {@link OSC_EVENTS.vrchat} (Rust `VrchatPresenceEvent`). */
export interface RustOscVrchatPresence {
  present: boolean;
}
