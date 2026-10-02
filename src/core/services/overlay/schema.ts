import { CaptionEventSourceSchema, TextEventSource } from "@/types";
import { zSafe, zStringNumber } from "@/utils";
import { z } from "zod";

/**
 * Root key `captionWindow` (P6-S0; owned by P6-B): the always-on-top, click-through caption window (`caption-window`
 * plugin, window label `captions`, early route `/captions` in `src/overlay/**`). Off by default. The window's
 * position and size are not stored here: `tauri-plugin-window-state` keeps them per window label.
 *
 * - `sources`: caption topics shown, in order (own captions, translation, other players).
 * - `alwaysOnTop` / `clickThrough`: pushed to Rust (`RustCaptionWindowOptions`); click-through off lets the user move
 *   and resize the window.
 * - Look: `fontFamily`, `fontSize` (px), colours as CSS colours, `maxLines`, `clearAfterMs` (0 = a final stays until
 *   replaced).
 */
export const CaptionWindow_Schema = z.object({
  enable: zSafe(z.coerce.boolean(), false),
  sources: zSafe(z.array(CaptionEventSourceSchema).max(4), [TextEventSource.stt]),
  interim: zSafe(z.coerce.boolean(), true),
  alwaysOnTop: zSafe(z.coerce.boolean(), true),
  clickThrough: zSafe(z.coerce.boolean(), true),
  fontFamily: zSafe(z.coerce.string(), "Outfit"),
  fontSize: zSafe(zStringNumber(), "28"),
  textColor: zSafe(z.coerce.string(), "rgba(255,255,255,1)"),
  backgroundColor: zSafe(z.coerce.string(), "rgba(0,0,0,0.6)"),
  maxLines: zSafe(zStringNumber(), "3"),
  clearAfterMs: zSafe(zStringNumber(), "6000"),
}).default({});

export type CaptionWindowState = z.infer<typeof CaptionWindow_Schema>;
