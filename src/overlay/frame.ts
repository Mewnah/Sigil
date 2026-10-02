/**
 * The caption window's frame (P6-B): what the main window's overlay service (`src/core/services/overlay`) sends to the
 * caption window (label `captions`, early route `/captions`) as the Tauri event {@link CAPTION_FRAME_EVENT}. The page
 * only renders frames: it never runs `ApiServer`, never reads settings and has no IPC rights besides listening to
 * events and dragging itself (`src-tauri/capabilities/captions.json`).
 *
 * Kept free of app imports: the early route loads this module before anything else.
 *
 * A frame is complete (style + every line): the service re-sends the latest one about once a second while the window
 * is open, so a page that loaded (or reloaded) after a change catches up; `seq` lets the page skip repeats.
 */

/** Tauri event carrying a {@link CaptionFrame} to the caption window. */
export const CAPTION_FRAME_EVENT = "caption_window:frame";
/** Label of the caption window (Rust `caption_window::CAPTIONS_WINDOW_LABEL`). */
export const CAPTION_WINDOW_LABEL = "captions";
/** Path of the early route (Rust `caption_window::CAPTIONS_ROUTE`). */
export const CAPTION_WINDOW_PATH = "/captions";

export interface CaptionFrameStyle {
  /** CSS font-family list (the user's font first, then CJK-capable fallbacks). */
  fontFamily: string;
  /** Pixels. */
  fontSize: number;
  /** CSS colours. */
  textColor: string;
  backgroundColor: string;
  /** Most caption rows visible (older wrapped rows scroll out at the top). */
  maxLines: number;
}

export interface CaptionFrameLine {
  /** Stable key of the line (React key). */
  key: string;
  text: string;
  /** Still revising: drawn dimmer. */
  interim: boolean;
  /** Committed prefix of an interim (UTF-16 units); the rest is a tentative tail. Null: unknown or final. */
  stable: number | null;
  /** Lower-case BCP-47 language (sets `lang` for CJK glyph shapes). */
  lang: string | null;
  /** Topic of the line (`text.stt`, `others.translation`, ...). */
  source: string;
}

export interface CaptionFrame {
  /** Increases with every new content; repeats of the same frame keep it. */
  seq: number;
  style: CaptionFrameStyle;
  /** Click-through is off: show the frame that is dragged to move the window. */
  unlocked: boolean;
  /** Localized hint shown while unlocked (the page has no i18n of its own). */
  hint: string;
  lines: CaptionFrameLine[];
}

/**
 * Most caption rows the window shows: the settings range (`WINDOW_LIMITS.maxLines` in `core/services/overlay/config`)
 * ends here, and the page keeps this many lines of a frame, so every line the user asked for reaches the box.
 */
export const CAPTION_WINDOW_MAX_LINES = 20;
/** Longest caption line the page keeps (UTF-16 units); a longer one keeps its newest text, like the VR overlay. */
export const CAPTION_WINDOW_MAX_TEXT = 2_000;
const LANG_RE = /^[a-z]{2,3}(-[a-z0-9]{1,8})*$/;
const FALLBACK_FONTS = `"Segoe UI", "Yu Gothic UI", "Microsoft YaHei UI", "Microsoft JhengHei UI", "Malgun Gothic", sans-serif`;

/** A CSS font-family list for `family` with CJK-capable fallbacks (quotes and separators that could break out stripped). */
export function captionFontFamily(family: string): string {
  const name = family.replace(/["';{}<>\\]/g, "").trim().slice(0, 64);
  return name ? `"${name}", ${FALLBACK_FONTS}` : FALLBACK_FONTS;
}

function isRecord(v: unknown): v is Record<string, unknown> {
  return typeof v === "object" && v !== null;
}

function str(v: unknown, max: number): string {
  return typeof v === "string" ? v.slice(0, max) : "";
}

/**
 * The newest `max` UTF-16 units of `text`: a caption grows at its end and the box is anchored at the bottom. Never
 * starts inside a surrogate pair (a leading lone low surrogate is dropped too).
 */
function tail(text: string, max: number): string {
  if (text.length <= max) return text;
  const out = text.slice(-max);
  const first = out.charCodeAt(0);
  return first >= 0xdc00 && first <= 0xdfff ? out.slice(1) : out;
}

function num(v: unknown, lo: number, hi: number, fallback: number): number {
  return typeof v === "number" && Number.isFinite(v) ? Math.min(hi, Math.max(lo, v)) : fallback;
}

/** A CSS colour value that cannot carry anything but a colour (no `;`, `url(`, braces). */
function cssColor(v: unknown, fallback: string): string {
  const s = str(v, 64).trim();
  return /^(#[0-9a-fA-F]{3,8}|rgba?\([0-9.,%\s/]+\)|[a-zA-Z]{3,20})$/.test(s) ? s : fallback;
}

/**
 * The frame the page renders, rebuilt from an event payload with every field checked (events come from Sigil's own
 * main window, but the page never trusts a shape it did not check). Returns null for anything that is not a frame.
 */
export function readCaptionFrame(payload: unknown): CaptionFrame | null {
  if (!isRecord(payload) || !isRecord(payload.style) || !Array.isArray(payload.lines)) return null;
  const s = payload.style;
  const lines: CaptionFrameLine[] = [];
  for (const raw of payload.lines.slice(-CAPTION_WINDOW_MAX_LINES)) {
    if (!isRecord(raw)) continue;
    const full = typeof raw.text === "string" ? raw.text : "";
    const text = tail(full, CAPTION_WINDOW_MAX_TEXT);
    if (!text.trim()) continue;
    // Units cut from the head: the committed prefix shrinks by as many (all tentative when it was cut whole).
    const dropped = full.length - text.length;
    const lang = typeof raw.lang === "string" && LANG_RE.test(raw.lang) ? raw.lang : null;
    const stable =
      typeof raw.stable === "number" && Number.isInteger(raw.stable) && raw.stable >= 0
        ? Math.min(Math.max(0, raw.stable - dropped), text.length)
        : null;
    lines.push({
      key: str(raw.key, 200) || String(lines.length),
      text,
      interim: raw.interim === true,
      stable,
      lang,
      source: str(raw.source, 64),
    });
  }
  return {
    seq: num(payload.seq, 0, Number.MAX_SAFE_INTEGER, 0),
    style: {
      fontFamily: typeof s.fontFamily === "string" && s.fontFamily ? s.fontFamily.replace(/[;{}<>\\]/g, "").slice(0, 300) : FALLBACK_FONTS,
      fontSize: num(s.fontSize, 8, 200, 28),
      textColor: cssColor(s.textColor, "rgba(255,255,255,1)"),
      backgroundColor: cssColor(s.backgroundColor, "rgba(0,0,0,0.6)"),
      maxLines: Math.round(num(s.maxLines, 1, CAPTION_WINDOW_MAX_LINES, 3)),
    },
    unlocked: payload.unlocked === true,
    hint: str(payload.hint, 300),
    lines,
  };
}
