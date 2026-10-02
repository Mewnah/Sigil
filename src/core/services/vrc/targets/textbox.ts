import type { TextEvent } from "@/types";
import { TextEventType } from "@/types";
import { pushSystemLog } from "@/core/services/systemLog";
import throttle from "lodash/throttle";
import { CHATBOX_MAX_CHARS, ChatboxPager, chatboxIntervalMs } from "../chatboxPager";
import { decorateWithTags } from "../tags";
import { TurnCommitGate } from "../turnCommit";
import { IVRCTarget, type PushFinalOptions } from "../types";
import { sendOsc, type OscArg } from "./utils";

/** What the chatbox target needs from the VRChat service. */
export interface TextboxTargetDeps {
  /** Smart Turn gating applies to the current source (setting on, and the source is not already turn-committed). */
  turnGate(): boolean;
  /** Tags to show as emoji for a committed line (undefined = none / emoji off). */
  tagsFor(event: TextEvent): readonly string[] | undefined;
}

/**
 * VRChat chatbox (`/chatbox/input`). Finals go through the Smart Turn gate (`textbox.commitOnTurnEnd`) and the paced
 * pager (every message at least `pageIntervalMs` apart, long text paginated or cut); interims drive the typing
 * indicator, which also stays on while a sentence waits for the end of the turn.
 */
class VRC_TextboxTarget implements IVRCTarget {
  readonly #deps: TextboxTargetDeps;
  readonly #pager: ChatboxPager;
  readonly #gate: TurnCommitGate;
  /** Lines committed so far (tells `pushFinal` whether its final was sent or dropped). */
  #commits = 0;
  /** The typing indicator was last sent as on. */
  #typingShown = false;
  /** The one-time hint that long text is cut (pagination off) was logged. */
  #truncationHinted = false;

  constructor(deps: TextboxTargetDeps) {
    this.#deps = deps;
    this.#pager = new ChatboxPager({
      send: (text, notify) => {
        const args: OscArg[] = [{ String: text }, { Bool: true }];
        if (!notify) args.push({ Bool: false });
        sendOsc("/chatbox/input", args);
      },
      paginate: () => this.state.paginate,
      intervalMs: () => chatboxIntervalMs(this.state.pageIntervalMs),
      notify: () => this.state.playNotificationSound,
      onDrop: (count) =>
        pushSystemLog("VRChat OSC", `Chatbox backlog too long: ${count} waiting page(s) dropped.`, "warning"),
      onTruncate: () => {
        if (this.#truncationHinted) return;
        this.#truncationHinted = true;
        pushSystemLog(
          "VRChat OSC",
          `Chatbox text longer than ${CHATBOX_MAX_CHARS} characters was cut. Turn on "Split long text into pages" in the VRChat settings to send all of it.`,
          "info",
        );
      },
    });
    this.#gate = new TurnCommitGate({
      enabled: () => this.state.commitOnTurnEnd && this.#deps.turnGate(),
      holdMs: () => {
        const ms = parseInt(String(this.state.turnHoldMs ?? ""), 10);
        return Number.isFinite(ms) ? Math.min(10_000, Math.max(500, ms)) : 2_500;
      },
      // One message, or two pages when paginating: a long monologue still goes out while it is spoken.
      maxChars: () => (this.state.paginate ? 2 * CHATBOX_MAX_CHARS : CHATBOX_MAX_CHARS),
      commit: (event) => this.#commit(event),
    });
  }

  get state() {
    return window.ApiServer.state.services.vrc.data.textbox;
  }

  #sendIndicator(value: boolean) {
    this.#typingShown = value;
    sendOsc("/chatbox/typing", [{ Bool: value }]);
  }

  #sendIndicatorThrottled = throttle((value: boolean) => this.#sendIndicator(value), 1400);

  /** Turns the typing indicator off now; a throttled "on" still pending from the last interims is dropped. */
  #clearIndicator(): void {
    this.#sendIndicatorThrottled.cancel();
    if (this.state.indicator || this.#typingShown) this.#sendIndicator(false);
  }

  #commit(event: TextEvent): void {
    this.#commits++;
    this.#clearIndicator();
    this.#pager.push(decorateWithTags(event.value, this.#deps.tagsFor(event)));
  }

  pushFinal(value: string, event?: TextEvent, options?: PushFinalOptions): void {
    const commits = this.#commits;
    const ev: TextEvent = event ? { ...event, value } : { type: TextEventType.final, value, emotes: {} };
    if (options?.typed) this.#gate.passThrough(ev);
    else this.#gate.final(ev);
    if (this.#gate.holding) {
      // Still waiting for the end of the turn: keep showing that the user is talking.
      if (this.state.indicator) this.#sendIndicatorThrottled(true);
    } else if (this.#commits === commits) {
      // The sentence ended with nothing to send (STT's cancel marker after a mute or a stop): the typing bubble the
      // interims turned on goes off too, instead of staying over the avatar until the next spoken sentence.
      this.#clearIndicator();
    }
  }

  pushInterim(value: string, event?: TextEvent): void {
    this.#gate.interim(event ? { ...event, value } : { type: TextEventType.interim, value, emotes: {} });
    if (this.state.indicator)
      this.#sendIndicatorThrottled(true);
  }

  cancel(): void {
    this.#gate.reset();
    this.#pager.cancel();
    this.#sendIndicatorThrottled.cancel();
    // Nothing more is coming: a typing bubble this target turned on must not stay up.
    if (this.#typingShown) this.#sendIndicator(false);
  }
}

export default VRC_TextboxTarget;
