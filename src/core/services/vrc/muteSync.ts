/**
 * Mute sync with VRChat's microphone (P6-C): muting yourself in VRChat mutes Sigil's speech-to-text, unmuting
 * unmutes it again, without fighting the user's own mute.
 *
 * Ownership rule: Sigil unmutes only a mute VRChat caused. When speech-to-text was already muted by the user as
 * VRChat muted, VRChat's unmute leaves it muted; when the user unmutes speech-to-text by hand while VRChat is muted
 * (to talk to Sigil only), VRChat's later unmute changes nothing either. Turning mute sync off, or VRChat quitting
 * (it never sends an unmute then), releases a mute VRChat caused.
 *
 * The capture hub's gate closes in Rust the moment the mute arrives (osc.rs), well within the 200 ms target; this
 * controller then sets the service state (`stt.setMuted`), which keeps the gate and the UI in step.
 *
 * Pure logic: speech-to-text is reached through {@link MuteSyncTarget}.
 */
export interface MuteSyncTarget {
  /** Speech-to-text is hard-muted (a pending unmute counts as unmuted). */
  isMuted(): boolean;
  setMuted(muted: boolean): void;
  /** Calls `cb` synchronously on every change of the mute state; returns the unsubscribe function. */
  subscribe(cb: () => void): () => void;
}

export class MuteSyncController {
  readonly #target: MuteSyncTarget;
  readonly #unsubscribe: () => void;
  #enabled = false;
  /** The current speech-to-text mute was caused by VRChat. */
  #owned = false;
  /** Set while this controller changes the mute itself (subscription callbacks are synchronous). */
  #applying = false;
  /** VRChat's last reported mute (`null` = unknown). */
  #vrchat: boolean | null = null;

  constructor(target: MuteSyncTarget) {
    this.#target = target;
    this.#unsubscribe = target.subscribe(() => {
      // A change made by someone else (the user, a shortcut): a VRChat mute the user lifted is no longer ours.
      if (!this.#applying && this.#owned && !this.#target.isMuted()) this.#owned = false;
    });
  }

  get enabled(): boolean {
    return this.#enabled;
  }

  /** Speech-to-text is muted because VRChat is. */
  get mutedByVrchat(): boolean {
    return this.#owned && this.#target.isMuted();
  }

  /** VRChat's last reported mute (`null` = unknown). */
  get vrchatMuted(): boolean | null {
    return this.#vrchat;
  }

  /** Turns mute sync on or off. Off releases a mute VRChat caused; on waits for VRChat's next report. */
  setEnabled(on: boolean): void {
    if (on === this.#enabled) return;
    this.#enabled = on;
    this.#vrchat = null;
    if (!on) this.release();
  }

  /** VRChat reported its microphone mute (deduplicated here too: a receiver restart repeats the last value). */
  onVrchatMute(muted: boolean): void {
    if (!this.#enabled || this.#vrchat === muted) return;
    this.#vrchat = muted;
    if (muted) {
      if (!this.#target.isMuted()) {
        this.#apply(true);
        this.#owned = true;
      }
      return;
    }
    this.release();
  }

  /**
   * VRChat's mute as the Rust receiver knows it (`listen_start` result, receiver status), taken only while this
   * controller does not know it yet. After a webview reload Rust keeps the running receiver and its deduplicated
   * MuteSelf, so no mute event comes again; without this a muted VRChat would leave speech-to-text unmuted until its
   * next change.
   */
  adoptReceiverState(muteSelf: boolean | null | undefined): void {
    if (typeof muteSelf !== "boolean" || this.#vrchat !== null) return;
    this.onVrchatMute(muteSelf);
  }

  /** VRChat went away (quit): its mute no longer applies. */
  onVrchatGone(): void {
    this.#vrchat = null;
    this.release();
  }

  /** Unmutes speech-to-text if VRChat muted it (and nobody changed it since). */
  release(): void {
    const owned = this.#owned;
    this.#owned = false;
    if (owned && this.#target.isMuted()) this.#apply(false);
  }

  dispose(): void {
    this.#unsubscribe();
  }

  #apply(muted: boolean): void {
    this.#applying = true;
    try {
      this.#target.setMuted(muted);
    } finally {
      this.#applying = false;
    }
  }
}
