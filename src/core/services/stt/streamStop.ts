/**
 * "Stop with Twitch stream" for the services that have the option (speech-to-text, text-to-speech, Rewrite).
 *
 * The Twitch service publishes `stream.on_ended` when a poll finds no stream right after one that did. One empty
 * reply is not an ended stream (a gap between broadcasts, an OBS restart that starts a new broadcast, a Helix hiccup),
 * and a service stopped by it used to stay stopped for the rest of the session. So:
 *
 * - the stop waits {@link STREAM_OFFLINE_CONFIRM_MS}; if the stream is live again by then (`stream.on_started`, or
 *   Twitch's live state), the service keeps running;
 * - a service this option stopped is started again when the stream comes back online during the same session, unless
 *   the option was switched off meanwhile or the service was started by hand.
 *
 * The services pass their own `enabled` / `active` / `stop` / `start`; this module holds no state of theirs.
 */
import { ServiceNetworkState } from "@/types";
import { subscribeKey } from "valtio/utils";

/** How long the stream must stay offline before the services are stopped. */
export const STREAM_OFFLINE_CONFIRM_MS = 60_000;

export interface StreamStopTarget {
  /** The service's "Stop with Twitch stream" option is on. */
  enabled(): boolean;
  /** The service is running, starting or reconnecting (anything but stopped). */
  active(): boolean;
  stop(): void;
  /** Start again after a stop this option made (an automatic start: it must not ask questions). */
  start(): void;
}

export interface StreamStopDeps {
  /** Calls `fn` on every local publish of `topic`; returns the unsubscribe. */
  subscribe(topic: string, fn: () => void): () => void;
  /** Calls `fn` when Twitch's live state turns live; returns the unsubscribe. */
  onLive(fn: () => void): () => void;
  /** Twitch's live state now (`undefined`: unknown). */
  isLive(): boolean | undefined;
  setTimeout(fn: () => void, ms: number): unknown;
  clearTimeout(handle: unknown): void;
}

function defaultDeps(): StreamStopDeps {
  const twitchState = () => window.ApiServer?.twitch?.state;
  return {
    subscribe: (topic, fn) => {
      const id = window.ApiShared.pubsub.subscribe(topic, fn);
      return () => window.ApiShared.pubsub.unsubscribe(id);
    },
    onLive: (fn) => {
      const state = twitchState();
      if (!state) return () => {};
      return subscribeKey(state, "liveStatus", (live) => {
        if (live === ServiceNetworkState.connected) fn();
      });
    },
    isLive: () => {
      const state = twitchState();
      return state ? state.liveStatus === ServiceNetworkState.connected : undefined;
    },
    setTimeout: (fn, ms) => setTimeout(fn, ms),
    clearTimeout: (handle) => clearTimeout(handle as ReturnType<typeof setTimeout>),
  };
}

export class StreamStop {
  #pending: unknown = null;
  /** This option stopped the service, and the stream has not come back yet. */
  #stoppedByStream = false;
  /** The start in progress is this option's own (it must not forget that it stopped the service). */
  #resuming = false;

  constructor(
    private readonly target: StreamStopTarget,
    private readonly deps: StreamStopDeps = defaultDeps(),
  ) {}

  /** Follow the stream; returns the dispose function. */
  attach(): () => void {
    const offs = [
      this.deps.subscribe("stream.on_ended", () => this.#ended()),
      this.deps.subscribe("stream.on_started", () => this.#live()),
      this.deps.onLive(() => this.#live()),
    ];
    return () => {
      offs.forEach((off) => off());
      this.#clearPending();
      this.#stoppedByStream = false;
    };
  }

  /**
   * The service was started (by the user, a shortcut, a link): a stop this option made before is forgotten, and so is
   * a stop that waits for its confirmation. The user chose to run it now; the stop only judges the stream's state at
   * the time it was armed, and would otherwise end the session they just started when its timer runs out.
   */
  noteStart(): void {
    if (this.#resuming) return;
    this.#stoppedByStream = false;
    this.#clearPending();
  }

  #ended(): void {
    if (!this.target.enabled() || !this.target.active() || this.#pending !== null) return;
    this.#pending = this.deps.setTimeout(() => this.#confirm(), STREAM_OFFLINE_CONFIRM_MS);
  }

  #confirm(): void {
    this.#pending = null;
    if (!this.target.enabled() || !this.target.active()) return;
    // The stream is back (or never left): keep running.
    if (this.deps.isLive() === true) return;
    this.#stoppedByStream = true;
    this.target.stop();
  }

  #live(): void {
    // A stream that came back before the stop is not stopped at all.
    this.#clearPending();
    if (!this.#stoppedByStream) return;
    this.#stoppedByStream = false;
    if (!this.target.enabled() || this.target.active()) return;
    this.#resuming = true;
    try {
      this.target.start();
    } finally {
      this.#resuming = false;
    }
  }

  #clearPending(): void {
    if (this.#pending === null) return;
    this.deps.clearTimeout(this.#pending);
    this.#pending = null;
  }
}
