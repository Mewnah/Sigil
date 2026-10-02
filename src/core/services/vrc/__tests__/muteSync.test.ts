import { describe, expect, it } from "vitest";
import { MuteSyncController } from "../muteSync";

/** Speech-to-text stand-in with synchronous change notifications (like `subscribeKey(..., true)`). */
function stt(initial = false) {
  let muted = initial;
  const listeners = new Set<() => void>();
  const calls: boolean[] = [];
  return {
    calls,
    get muted() {
      return muted;
    },
    /** The user (or a shortcut) changes the mute. */
    user(v: boolean) {
      muted = v;
      listeners.forEach((l) => l());
    },
    target: {
      isMuted: () => muted,
      setMuted: (v: boolean) => {
        calls.push(v);
        muted = v;
        listeners.forEach((l) => l());
      },
      subscribe: (cb: () => void) => {
        listeners.add(cb);
        return () => listeners.delete(cb);
      },
    },
  };
}

function controller(initial = false) {
  const s = stt(initial);
  const c = new MuteSyncController(s.target);
  c.setEnabled(true);
  return { s, c };
}

describe("MuteSyncController", () => {
  it("mutes with VRChat and unmutes with VRChat", () => {
    const { s, c } = controller();
    c.onVrchatMute(true);
    expect(s.muted).toBe(true);
    expect(c.mutedByVrchat).toBe(true);
    expect(c.vrchatMuted).toBe(true);
    c.onVrchatMute(false);
    expect(s.muted).toBe(false);
    expect(s.calls).toEqual([true, false]);
  });

  it("does nothing while off, and ignores repeated reports", () => {
    const s = stt();
    const c = new MuteSyncController(s.target);
    c.onVrchatMute(true);
    expect(s.calls).toEqual([]);
    c.setEnabled(true);
    c.onVrchatMute(true);
    c.onVrchatMute(true);
    expect(s.calls).toEqual([true]);
  });

  it("keeps the user's own mute when VRChat unmutes", () => {
    const { s, c } = controller(true);
    c.onVrchatMute(true);
    expect(c.mutedByVrchat).toBe(false);
    c.onVrchatMute(false);
    expect(s.muted).toBe(true);
    expect(s.calls).toEqual([]);
  });

  it("a manual unmute while VRChat is muted is not undone, nor redone by VRChat's unmute", () => {
    const { s, c } = controller();
    c.onVrchatMute(true);
    s.user(false); // talk to Sigil only
    expect(c.mutedByVrchat).toBe(false);
    s.user(true); // and mute again by hand
    c.onVrchatMute(false);
    expect(s.muted).toBe(true);
    expect(s.calls).toEqual([true]);
  });

  it("turning mute sync off or VRChat quitting releases VRChat's mute only", () => {
    const a = controller();
    a.c.onVrchatMute(true);
    a.c.setEnabled(false);
    expect(a.s.muted).toBe(false);

    const b = controller();
    b.c.onVrchatMute(true);
    b.c.onVrchatGone();
    expect(b.s.muted).toBe(false);
    expect(b.c.vrchatMuted).toBeNull();
    // VRChat back and still muted: muted again (not swallowed as a repeat).
    b.c.onVrchatMute(true);
    expect(b.s.muted).toBe(true);

    const c = controller(true);
    c.c.onVrchatMute(true);
    c.c.setEnabled(false);
    expect(c.s.muted).toBe(true);
  });

  it("adopts the receiver's known MuteSelf after a webview reload (no new mute event comes)", () => {
    // Rust kept the receiver running and VRChat is muted: listen_start returns muteSelf true.
    const { s, c } = controller();
    c.adoptReceiverState(true);
    expect(s.muted).toBe(true);
    expect(c.mutedByVrchat).toBe(true);
    expect(c.vrchatMuted).toBe(true);
    // VRChat's next unmute still releases it.
    c.onVrchatMute(false);
    expect(s.muted).toBe(false);
  });

  it("adopts the receiver state only while VRChat's mute is unknown here", () => {
    const { s, c } = controller();
    c.adoptReceiverState(null);
    c.adoptReceiverState(undefined);
    expect(c.vrchatMuted).toBeNull();
    c.onVrchatMute(true);
    s.user(false); // the user lifted VRChat's mute by hand
    // A later status repeating VRChat's mute must not mute again.
    c.adoptReceiverState(true);
    expect(s.muted).toBe(false);
    expect(s.calls).toEqual([true]);
    // Off: nothing is adopted.
    const off = stt();
    const d = new MuteSyncController(off.target);
    d.adoptReceiverState(true);
    expect(off.calls).toEqual([]);
  });

  it("re-enabling waits for VRChat's next report", () => {
    const { s, c } = controller();
    c.onVrchatMute(true);
    c.setEnabled(false);
    c.setEnabled(true);
    expect(c.vrchatMuted).toBeNull();
    c.onVrchatMute(true);
    expect(s.muted).toBe(true);
    c.dispose();
  });
});
