import {
  type CaptionEventSource,
  OthersEventSource,
  TextEventSource,
  type RustCaptionWindowOptions,
  type RustCaptionWindowStatus,
  type RustVrOverlayAttach,
  type RustVrOverlayCaption,
  type RustVrOverlayConfig,
  type RustVrOverlayStatus,
} from "@/types";
import type { VRC_State } from "../vrc/schema";
import { CAPTION_WINDOW_MAX_LINES, captionFontFamily, type CaptionFrameLine, type CaptionFrameStyle } from "@/overlay/frame";
import type { CaptionModelLine, CaptionModelOptions } from "./captionModel";
import type { CaptionWindowState } from "./schema";

/**
 * Pure conversions for the overlay service (P6-B): settings (strings, as stored) to the Rust option structs and the
 * caption window's frame style, caption lines to wire lines, and caption-source handling. Every number is clamped the
 * way the Rust side clamps it, so what the UI shows is what renders.
 */

export type VrOverlaySettings = VRC_State["overlay"];

/** Why the SteamVR overlay waits or failed (Rust `vr_overlay::VrOverlayReason`, a P6-B status field). */
export const VR_OVERLAY_REASONS = [
  "runtimeMissing",
  "steamvrNotRunning",
  "steamvrNotInstalled",
  "steamvrTooOld",
  "steamvrQuit",
  "connectFailed",
  "overlayError",
] as const;
export type VrOverlayReason = (typeof VR_OVERLAY_REASONS)[number];

/** `RustCaptionWindowStatus` with the P6-B `closedExternally` flag (set on the close event of a window closed by Alt+F4 or the OS). */
export interface CaptionWindowStatusView extends RustCaptionWindowStatus {
  closedExternally?: boolean;
}

/** `RustVrOverlayStatus` with the P6-B `reason` field. */
export interface VrOverlayStatusView extends RustVrOverlayStatus {
  reason?: VrOverlayReason | null;
}

/** Ranges shared with Rust (`vr_overlay::settings`) and the inspector. */
export const VR_LIMITS = {
  widthM: [0.1, 3],
  distanceM: [0.3, 5],
  offsetYM: [-2, 2],
  fontSizePx: [12, 96],
  opacity: [0.05, 1],
  maxLines: [1, 8],
} as const;

/** Caption window ranges; the line cap is the page's own (`CAPTION_WINDOW_MAX_LINES`), so no setting is cut short. */
export const WINDOW_LIMITS = {
  fontSize: [8, 200],
  maxLines: [1, CAPTION_WINDOW_MAX_LINES],
} as const;

/** Longest clear timer (ms). */
export const MAX_CLEAR_AFTER_MS = 600_000;

/** Rust `VrOverlayConfig::default()` (checked by `rustContract.test.ts`). */
export const VR_DEFAULTS: RustVrOverlayConfig = {
  attach: "head",
  widthM: 0.6,
  distanceM: 1.2,
  offsetYM: -0.25,
  fontSizePx: 36,
  opacity: 0.9,
  maxLines: 3,
  textColor: "rgba(255,255,255,1)",
  backgroundColor: "rgba(0,0,0,0.6)",
};

const ATTACH: readonly RustVrOverlayAttach[] = ["head", "leftHand", "rightHand", "world"];

/** `value` as a number clamped to `[lo, hi]`, or `fallback` when it is not a number. */
export function clampNumber(value: unknown, [lo, hi]: readonly [number, number], fallback: number): number {
  const n = typeof value === "number" ? value : typeof value === "string" && value.trim() !== "" ? Number(value) : NaN;
  return Number.isFinite(n) ? Math.min(hi, Math.max(lo, n)) : fallback;
}

/** The caption lines a display keeps and their clear timer. */
export function feedOptions(s: { interim: boolean; maxLines: string; clearAfterMs: string }, maxLines: readonly [number, number]): CaptionModelOptions {
  return {
    interim: !!s.interim,
    maxLines: Math.round(clampNumber(s.maxLines, maxLines, 3)),
    clearAfterMs: Math.round(clampNumber(s.clearAfterMs, [0, MAX_CLEAR_AFTER_MS], 6000)),
  };
}

/** Always-on-top and click-through for Rust. */
export function captionWindowOptions(s: CaptionWindowState): RustCaptionWindowOptions {
  return { alwaysOnTop: !!s.alwaysOnTop, clickThrough: !!s.clickThrough };
}

/** The caption window's look. */
export function captionWindowStyle(s: CaptionWindowState): CaptionFrameStyle {
  return {
    fontFamily: captionFontFamily(s.fontFamily),
    fontSize: Math.round(clampNumber(s.fontSize, WINDOW_LIMITS.fontSize, 28)),
    textColor: s.textColor || "rgba(255,255,255,1)",
    backgroundColor: s.backgroundColor || "rgba(0,0,0,0.6)",
    maxLines: Math.round(clampNumber(s.maxLines, WINDOW_LIMITS.maxLines, 3)),
  };
}

/** The SteamVR overlay's config for Rust. */
export function vrOverlayConfig(s: VrOverlaySettings): RustVrOverlayConfig {
  return {
    attach: ATTACH.includes(s.attach as RustVrOverlayAttach) ? (s.attach as RustVrOverlayAttach) : VR_DEFAULTS.attach,
    widthM: clampNumber(s.widthM, VR_LIMITS.widthM, VR_DEFAULTS.widthM),
    distanceM: clampNumber(s.distanceM, VR_LIMITS.distanceM, VR_DEFAULTS.distanceM),
    offsetYM: clampNumber(s.offsetYM, VR_LIMITS.offsetYM, VR_DEFAULTS.offsetYM),
    fontSizePx: Math.round(clampNumber(s.fontSizePx, VR_LIMITS.fontSizePx, VR_DEFAULTS.fontSizePx)),
    opacity: clampNumber(s.opacity, VR_LIMITS.opacity, VR_DEFAULTS.opacity),
    maxLines: Math.round(clampNumber(s.maxLines, VR_LIMITS.maxLines, VR_DEFAULTS.maxLines)),
    textColor: s.textColor || VR_DEFAULTS.textColor,
    backgroundColor: s.backgroundColor || VR_DEFAULTS.backgroundColor,
  };
}

/**
 * The topics to subscribe to, in order and without overlap: "Any text source" (`text`) already hears every `text.*`
 * topic, and `others` every `others.*` (pubsub bubbles), so listing both would show each line twice.
 */
export function normalizeSources(sources: readonly CaptionEventSource[]): CaptionEventSource[] {
  const unique = [...new Set(sources)];
  const anyText = unique.includes(TextEventSource.any);
  const anyOthers = unique.includes(OthersEventSource.any);
  return unique.filter((s) => {
    if (anyText && s !== TextEventSource.any && s.startsWith(`${TextEventSource.any}.`)) return false;
    if (anyOthers && s !== OthersEventSource.any && s.startsWith(`${OthersEventSource.any}.`)) return false;
    return true;
  });
}

/**
 * Whether a display subscribed to `subscribed` shows an event published on `topic`. Through "Any text source" the AI
 * transform's echoes of other topics (`text.transform_source`, `text.transform_raw`) are skipped: they repeat the
 * speech line the display already shows.
 */
export function acceptsTopic(subscribed: string, topic: string): boolean {
  if (subscribed !== TextEventSource.any) return true;
  return topic !== TextEventSource.transform_source && topic !== TextEventSource.transform_raw;
}

/** Lines for the caption window's frame. */
export function toFrameLines(lines: readonly CaptionModelLine[]): CaptionFrameLine[] {
  return lines.map((l) => ({
    key: l.key,
    text: l.text,
    interim: l.kind === "interim",
    stable: l.kind === "interim" && l.stableLength !== undefined ? l.stableLength : null,
    lang: l.lang ?? null,
    source: l.source,
  }));
}

/** Lines for `vr_overlay_set_text`. */
export function toVrCaption(lines: readonly CaptionModelLine[]): RustVrOverlayCaption {
  return {
    lines: lines.map((l) => ({ text: l.text, kind: l.kind, source: l.source, lang: l.lang ?? null })),
  };
}
