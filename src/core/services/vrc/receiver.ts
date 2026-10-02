/**
 * TS side of the Rust OSC receiver (P6-C; `osc` plugin, osc.rs): which options to run it with, serialized
 * start / stop calls, and its events. The receiver runs only while the VRChat integration and mute sync are on.
 */
import { invoke } from "@tauri-apps/api/core";
import { listen, type UnlistenFn } from "@tauri-apps/api/event";
import type { VRC_State } from "./schema";
import {
  OSC_EVENTS,
  type RustOscListenOptions,
  type RustOscMuteSelf,
  type RustOscReceiverStatus,
  type RustOscVrchatPresence,
} from "./types";

/** VRChat's default OSC send port (Rust `osc_query::FALLBACK_LISTEN_PORT`). */
export const DEFAULT_LISTEN_PORT = 9001;

/** A listen port setting as a port number (anything invalid → {@link DEFAULT_LISTEN_PORT}). */
export function parseListenPort(raw: unknown): number {
  const n = parseInt(String(raw ?? ""), 10);
  return Number.isInteger(n) && n >= 1 && n <= 65535 ? n : DEFAULT_LISTEN_PORT;
}

/** Whether the VRChat host setting means this PC (as Rust's `is_loopback_host`; empty = the default 127.0.0.1). */
export function isLoopbackHost(raw: unknown): boolean {
  const host = String(raw ?? "").trim().toLowerCase().replace(/^\[(.*)\]$/, "$1");
  if (host === "" || host === "localhost" || host === "::1") return true;
  const v4 = host.replace(/^::ffff:/, "");
  return /^127\.\d{1,3}\.\d{1,3}\.\d{1,3}$/.test(v4);
}

/**
 * Receiver options for the VRChat settings, or null when the receiver should not run. OSCQuery only finds a VRChat on
 * this PC, so it is never asked for with VRChat on another PC (the settings hide the checkbox there).
 */
export function listenOptionsFor(
  state: Pick<VRC_State, "enable" | "muteSync" | "oscQuery" | "oscListenPort" | "oscHost">,
): RustOscListenOptions | null {
  if (!state.enable || !state.muteSync) return null;
  const host = String(state.oscHost ?? "").trim() || "127.0.0.1";
  return {
    host,
    port: parseListenPort(state.oscListenPort),
    oscQuery: !!state.oscQuery && isLoopbackHost(host),
    muteSync: true,
  };
}

function sameOptions(a: RustOscListenOptions | null, b: RustOscListenOptions | null): boolean {
  if (a === null || b === null) return a === b;
  return a.host === b.host && a.port === b.port && a.oscQuery === b.oscQuery && a.muteSync === b.muteSync;
}

export interface OscReceiverHandlers {
  onMuteSelf(event: RustOscMuteSelf): void;
  onStatus(status: RustOscReceiverStatus): void;
  onVrchat(event: RustOscVrchatPresence): void;
}

export class OscReceiverClient {
  #chain: Promise<unknown> = Promise.resolve();
  /** Options last sent to Rust (`undefined`: nothing sent yet). */
  #applied: RustOscListenOptions | null | undefined = undefined;
  #unlisten: UnlistenFn[] = [];
  #disposed = false;

  /** Subscribes to the receiver's events. Rejects when events are unavailable (no Tauri). */
  async attach(handlers: OscReceiverHandlers): Promise<void> {
    const unlisten = await Promise.all([
      listen<RustOscMuteSelf>(OSC_EVENTS.muteSelf, (e) => handlers.onMuteSelf(e.payload)),
      listen<RustOscReceiverStatus>(OSC_EVENTS.receiver, (e) => handlers.onStatus(e.payload)),
      listen<RustOscVrchatPresence>(OSC_EVENTS.vrchat, (e) => handlers.onVrchat(e.payload)),
    ]);
    if (this.#disposed) {
      unlisten.forEach((u) => u());
      return;
    }
    this.#unlisten.push(...unlisten);
  }

  /**
   * Starts, restarts or stops the receiver for `options` (null = stopped). Calls run one after another; identical
   * options are not sent again. Resolves with the receiver's status after a start, null otherwise.
   */
  sync(options: RustOscListenOptions | null): Promise<RustOscReceiverStatus | null> {
    const run = async (): Promise<RustOscReceiverStatus | null> => {
      if (this.#disposed && options !== null) return null;
      if (this.#applied !== undefined && sameOptions(this.#applied, options)) return null;
      if (options === null) {
        // Also on the first sync: a receiver left running by an earlier webview session (reload) is stopped.
        await invoke<void>("plugin:osc|listen_stop");
        this.#applied = null;
        return null;
      }
      try {
        const status = await invoke<RustOscReceiverStatus>("plugin:osc|listen_start", { options });
        this.#applied = options;
        return status;
      } catch (e) {
        // Retried by the next sync.
        this.#applied = undefined;
        throw e;
      }
    };
    const next = this.#chain.then(run, run);
    this.#chain = next.catch(() => {});
    return next;
  }

  /** Current receiver status (null when unavailable). */
  async status(): Promise<RustOscReceiverStatus | null> {
    try {
      return await invoke<RustOscReceiverStatus>("plugin:osc|listen_status");
    } catch {
      return null;
    }
  }

  /** Stops the receiver and the event subscriptions. */
  dispose(): Promise<unknown> {
    this.#disposed = true;
    this.#unlisten.forEach((u) => u());
    this.#unlisten = [];
    return this.sync(null).catch(() => null);
  }
}
