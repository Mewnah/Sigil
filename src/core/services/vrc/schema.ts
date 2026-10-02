import { CaptionEventSourceSchema, TextEventSource, TextEventSourceSchema } from "@/types";
import { zSafe, zStringNumber } from "@/utils";
import { z } from "zod";

export enum VRC_Backends {
  textbox = "textbox",
  killfrenzy = "killfrenzy",
}
const zodVRC_Backends = z.nativeEnum(VRC_Backends);

/** Where the SteamVR caption overlay floats (Rust `vr_overlay::VrOverlayAttach`, P6-B). */
export const VRC_OVERLAY_ATTACH = ["head", "leftHand", "rightHand", "world"] as const;

/**
 * SteamVR caption overlay settings (P6-B; `vr-overlay` plugin, UI `inspector/vrc/OverlaySection.tsx`). Off by
 * default; Sigil never launches SteamVR. Numbers are strings like the rest of this slice (`zStringNumber`).
 */
export const VRC_Overlay_Schema = z.object({
  enable: zSafe(z.coerce.boolean(), false),
  /** Caption sources shown, in order (own captions, translation, other players). */
  sources: zSafe(z.array(CaptionEventSourceSchema).max(4), [TextEventSource.stt]),
  /** Show interim (still revising) text. */
  interim: zSafe(z.coerce.boolean(), true),
  attach: zSafe(z.enum(VRC_OVERLAY_ATTACH), "head"),
  widthM: zSafe(zStringNumber(), "0.6"),
  distanceM: zSafe(zStringNumber(), "1.2"),
  offsetYM: zSafe(zStringNumber(), "-0.25"),
  fontSizePx: zSafe(zStringNumber(), "36"),
  opacity: zSafe(zStringNumber(), "0.9"),
  maxLines: zSafe(zStringNumber(), "3"),
  textColor: zSafe(z.coerce.string(), "rgba(255,255,255,1)"),
  backgroundColor: zSafe(z.coerce.string(), "rgba(0,0,0,0.6)"),
  /** A final leaves the overlay after this long (ms; 0 = stays until replaced). */
  clearAfterMs: zSafe(zStringNumber(), "6000"),
}).default({});

export const Service_VRC_Schema = z.object({
  enable: zSafe(z.coerce.boolean(), false),
  oscHost: zSafe(z.coerce.string(), "127.0.0.1"),
  oscPort: zSafe(zStringNumber(), "9000"),
  source: zSafe(TextEventSourceSchema, TextEventSource.stt),
  target: zSafe(zodVRC_Backends, VRC_Backends.textbox),
  inputField: zSafe(z.coerce.boolean(), true),
  /**
   * P6-C: VRChat's own mic mute (`/avatar/parameters/MuteSelf`, received over OSC) mutes Sigil's speech-to-text too
   * (`stt.setMuted` and the Rust capture gate). Off by default; turning it on starts the OSC receiver.
   */
  muteSync: zSafe(z.coerce.boolean(), false),
  /** P6-C OSC receiver: advertise it with OSCQuery (mDNS) so VRChat finds it on any free port. */
  oscQuery: zSafe(z.coerce.boolean(), true),
  /** P6-C OSC receiver: UDP port when OSCQuery is off or unavailable (VRChat sends to 9001 by default). */
  oscListenPort: zSafe(zStringNumber(), "9001"),
  textbox: z.object({
    indicator: zSafe(z.coerce.boolean(), true),
    /** When false, OSC `/chatbox/input` third arg suppresses VRChat notification SFX. */
    playNotificationSound: zSafe(z.coerce.boolean(), true),
    /**
     * P6-C chatbox pagination: text longer than the chatbox's 144 characters is sent as numbered pages instead of
     * being cut to one message (`vrc/chatboxPager.ts`). A new feature, so off by default (plan global rule); with it
     * off, long text is cut at a word boundary with an ellipsis and the system log says once how to send all of it.
     * "No chatbox truncation" (P6-C acceptance) therefore holds with this setting on.
     */
    paginate: zSafe(z.coerce.boolean(), false),
    /**
     * P6-C chatbox pacing: time between chatbox messages and pages (ms, clamped to 1300-10000). Every message is
     * queued this far apart, because VRChat drops chatbox messages sent faster than about 1.3 s.
     */
    pageIntervalMs: zSafe(zStringNumber(), "1500"),
    /** P6-C: send a sentence only when Smart Turn says the speaker finished (`TextEvent.turnEnd`). */
    commitOnTurnEnd: zSafe(z.coerce.boolean(), false),
    /** P6-C: longest wait for the end of an unfinished turn after the last speech (ms). */
    turnHoldMs: zSafe(zStringNumber(), "2500"),
  }).default({}),
  /**
   * P6-C: SenseVoice emotion and audio-event tags (`TextEvent.tags`) to avatar parameters and the chatbox. Parameter
   * names "" are not sent.
   */
  tags: z.object({
    enable: zSafe(z.coerce.boolean(), false),
    /** Int avatar parameter set to the emotion's index (0 = neutral). */
    emotionParameter: zSafe(z.coerce.string(), "SigilEmotion"),
    /** Int avatar parameter set to the audio event's index (0 = none). */
    eventParameter: zSafe(z.coerce.string(), "SigilEvent"),
    /** Show tags as emoji in the chatbox text. */
    chatbox: zSafe(z.coerce.boolean(), false),
    /** Both parameters go back to 0 this long after the last tagged sentence (ms; 0 = they stay). */
    resetMs: zSafe(zStringNumber(), "5000"),
  }).default({}),
  /** P6-B: captions in a SteamVR overlay. */
  overlay: VRC_Overlay_Schema,
  killfrenzy: z.object({
    syncDelay: zSafe(zStringNumber(), "250"),
    visibleTimer: zSafe(zStringNumber(), "5000"),
    syncPoints: zSafe(zStringNumber(), "4"),
    indicator: zSafe(z.coerce.boolean(), false),
    KAT_Visible: zSafe(z.coerce.string(), "KAT_Visible"),
    KAT_Pointer: zSafe(z.coerce.string(), "KAT_Pointer"),
    KAT_CharSync: zSafe(z.coerce.string(), "KAT_CharSync"),
    splitSentences: zSafe(z.coerce.boolean(), true)
  }).default({})
}).default({});

export type VRC_State = z.infer<typeof Service_VRC_Schema>;
