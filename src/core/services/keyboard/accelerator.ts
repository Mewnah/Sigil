/**
 * Global shortcut strings for `@tauri-apps/plugin-global-shortcut` (parsed by the `global-hotkey` 0.7 crate:
 * `Mod+Mod+Key`, modifiers first, exactly one main key, `+` as separator). Pure (unit-tested).
 */

/** `KeyboardEvent.code` of modifier keys → accelerator modifier name. */
const MODIFIER_BY_CODE: Record<string, Modifier> = {
  ControlLeft: "Control",
  ControlRight: "Control",
  AltLeft: "Alt",
  AltRight: "Alt",
  ShiftLeft: "Shift",
  ShiftRight: "Shift",
  MetaLeft: "Super",
  MetaRight: "Super",
  OSLeft: "Super",
  OSRight: "Super",
};

export type Modifier = "Control" | "Alt" | "Shift" | "Super";
/** Canonical modifier order in stored shortcuts. */
const MODIFIER_ORDER: readonly Modifier[] = ["Control", "Alt", "Shift", "Super"];

/** Accelerator tokens `global-hotkey` accepts as modifiers (upper-cased). */
const MODIFIER_TOKENS: Record<string, Modifier> = {
  CONTROL: "Control",
  CTRL: "Control",
  COMMANDORCONTROL: "Control",
  COMMANDORCTRL: "Control",
  CMDORCTRL: "Control",
  CMDORCONTROL: "Control",
  ALT: "Alt",
  OPTION: "Alt",
  SHIFT: "Shift",
  SUPER: "Super",
  COMMAND: "Super",
  CMD: "Super",
};

const letters = Object.fromEntries(
  Array.from({ length: 26 }, (_, i) => String.fromCharCode(65 + i)).map((c) => [`Key${c}`, c]),
);
const digits = Object.fromEntries(Array.from({ length: 10 }, (_, i) => [`Digit${i}`, String(i)]));
const fKeys = Object.fromEntries(Array.from({ length: 24 }, (_, i) => [`F${i + 1}`, `F${i + 1}`]));
const numpadDigits = Object.fromEntries(Array.from({ length: 10 }, (_, i) => [`Numpad${i}`, `Numpad${i}`]));

/** `KeyboardEvent.code` of a main key → the accelerator token `global-hotkey` parses. */
export const KEY_TOKEN_BY_CODE: Readonly<Record<string, string>> = {
  ...letters,
  ...digits,
  ...fKeys,
  ...numpadDigits,
  NumpadAdd: "NumpadAdd",
  NumpadSubtract: "NumpadSubtract",
  NumpadMultiply: "NumpadMultiply",
  NumpadDivide: "NumpadDivide",
  NumpadDecimal: "NumpadDecimal",
  NumpadEnter: "NumpadEnter",
  NumpadEqual: "NumpadEqual",
  Backquote: "`",
  Backslash: "\\",
  BracketLeft: "[",
  BracketRight: "]",
  Comma: ",",
  Period: ".",
  Slash: "/",
  Semicolon: ";",
  Quote: "'",
  Minus: "-",
  Equal: "=",
  Space: "Space",
  Enter: "Enter",
  Tab: "Tab",
  Backspace: "Backspace",
  Delete: "Delete",
  Insert: "Insert",
  Home: "Home",
  End: "End",
  PageUp: "PageUp",
  PageDown: "PageDown",
  ArrowUp: "Up",
  ArrowDown: "Down",
  ArrowLeft: "Left",
  ArrowRight: "Right",
  Pause: "Pause",
  ScrollLock: "ScrollLock",
  PrintScreen: "PrintScreen",
  CapsLock: "CapsLock",
  NumLock: "NumLock",
  AudioVolumeMute: "AudioVolumeMute",
  AudioVolumeDown: "AudioVolumeDown",
  AudioVolumeUp: "AudioVolumeUp",
  MediaPlayPause: "MediaPlayPause",
  MediaStop: "MediaStop",
  MediaTrackNext: "MediaTrackNext",
  MediaTrackPrevious: "MediaTrackPrevious",
};

/** Every main-key token `global-hotkey` 0.7 accepts (upper-cased), including its aliases. */
const ACCEPTED_KEY_TOKENS: ReadonlySet<string> = new Set([
  ...Object.values(KEY_TOKEN_BY_CODE).map((t) => t.toUpperCase()),
  ...Object.keys(KEY_TOKEN_BY_CODE).map((c) => c.toUpperCase()),
  "ESCAPE", "ESC", "PAUSEBREAK", "NUM0", "NUM1", "NUM2", "NUM3", "NUM4", "NUM5", "NUM6", "NUM7", "NUM8", "NUM9",
  "NUMADD", "NUMPADPLUS", "NUMPLUS", "NUMDECIMAL", "NUMDIVIDE", "NUMENTER", "NUMEQUAL", "NUMMULTIPLY", "NUMSUBTRACT",
  "VOLUMEDOWN", "VOLUMEUP", "VOLUMEMUTE", "MEDIAPLAY", "MEDIAPAUSE", "MEDIATRACKPREV",
]);

/**
 * Keys that may be a shortcut on their own: a bare letter, digit or navigation key would be swallowed system-wide
 * (typing it in any app would toggle the action), so those need Control, Alt or Super. Shift alone is not enough:
 * Shift+M is how a capital M is typed.
 */
export function isStandaloneSafeKey(token: string): boolean {
  const t = token.toUpperCase();
  return /^F([1-9]|1\d|2[0-4])$/.test(t) || t === "PAUSE" || t === "SCROLLLOCK" || /^MEDIA/.test(t) || /^AUDIOVOLUME/.test(t);
}

export interface KeyEventLike {
  /** Physical key (US-QWERTY position). */
  code: string;
  /** Character the active layout produces (`KeyboardEvent.key`); used for letters, see {@link mainKeyToken}. */
  key?: string;
  ctrlKey: boolean;
  altKey: boolean;
  shiftKey: boolean;
  metaKey: boolean;
}

export type RecordStep =
  /** Only modifiers so far: keep listening. */
  | { kind: "pending"; preview: string }
  /** Escape without modifiers. */
  | { kind: "cancel" }
  | { kind: "shortcut"; value: string }
  /** A key that would block normal typing without Control, Alt or Super (Shift alone is not enough). */
  | { kind: "needs_modifier"; preview: string }
  /** A key global shortcuts cannot use. */
  | { kind: "unsupported"; code: string };

function modifiersOf(e: KeyEventLike): Modifier[] {
  const mods: Modifier[] = [];
  if (e.ctrlKey) mods.push("Control");
  if (e.altKey) mods.push("Alt");
  if (e.shiftKey) mods.push("Shift");
  if (e.metaKey) mods.push("Super");
  return mods;
}

/**
 * The accelerator token for the main key of `e`. `global-hotkey` registers letters as Windows virtual keys (VK_A to
 * VK_Z), which follow the active keyboard layout, while `KeyboardEvent.code` is the US-QWERTY position: on QWERTZ
 * the key labelled Z has code `KeyY`, on AZERTY the key labelled A has code `KeyQ` and M has code `Semicolon`. So a
 * letter comes from `e.key` (the layout's character) whenever that is a plain ASCII letter; everything else, and
 * non-Latin layouts (whose VK letters do follow the QWERTY positions), keeps the physical-code mapping.
 */
export function mainKeyToken(e: KeyEventLike): string | undefined {
  const key = e.key;
  if (typeof key === "string" && /^[a-z]$/i.test(key) && !e.code.startsWith("Numpad")) return key.toUpperCase();
  return KEY_TOKEN_BY_CODE[e.code];
}

/** One keydown while recording a shortcut → what to do next. */
export function recordKeyDown(e: KeyEventLike): RecordStep {
  const mods = modifiersOf(e);
  const modifierKey = MODIFIER_BY_CODE[e.code];
  if (modifierKey) {
    const all = MODIFIER_ORDER.filter((m) => m === modifierKey || mods.includes(m));
    return { kind: "pending", preview: [...all, ""].join("+") };
  }
  if (e.code === "Escape" && mods.length === 0) return { kind: "cancel" };
  const token = mainKeyToken(e);
  if (!token) return { kind: "unsupported", code: e.code };
  const value = [...mods, token].join("+");
  // RegisterHotKey swallows the combination system-wide: Shift+letter would eat every capital letter typed.
  const typingSafe = mods.some((m) => m !== "Shift");
  if (!typingSafe && !isStandaloneSafeKey(token)) return { kind: "needs_modifier", preview: value };
  return { kind: "shortcut", value };
}

/**
 * Whether `global-hotkey` would parse `value` (modifiers first, exactly one known main key). Used to tell a stored
 * shortcut that can never register (clear it) from one another app holds right now (keep it).
 */
export function isValidAccelerator(value: string): boolean {
  const tokens = value.split("+").map((t) => t.trim());
  if (tokens.length === 0 || tokens.some((t) => t === "")) return false;
  const key = tokens[tokens.length - 1].toUpperCase();
  if (!ACCEPTED_KEY_TOKENS.has(key)) return false;
  return tokens.slice(0, -1).every((t) => t.toUpperCase() in MODIFIER_TOKENS);
}

/** Same shortcut regardless of modifier order, aliases and case (for duplicate checks). */
export function acceleratorKey(value: string): string {
  const tokens = value.split("+").map((t) => t.trim()).filter(Boolean);
  if (tokens.length === 0) return "";
  const main = tokens[tokens.length - 1].toUpperCase();
  const mods = new Set(tokens.slice(0, -1).map((t) => MODIFIER_TOKENS[t.toUpperCase()] ?? t.toUpperCase()));
  return [...MODIFIER_ORDER.filter((m) => mods.has(m)), main].join("+");
}

/**
 * Turns global-shortcut events into single actions. The plugin reports both `Pressed` and `Released` (a handler
 * that ignores `state` fires twice, which cancels out a toggle), and a held key may repeat `Pressed`.
 */
export class ShortcutPressGate {
  readonly #held = new Map<string, number>();

  constructor(private readonly staleMs = 3_000) {}

  /** True when this event should trigger the shortcut's action. */
  accept(shortcut: string, state: "Pressed" | "Released", now: number): boolean {
    if (state === "Released") {
      this.#held.delete(shortcut);
      return false;
    }
    const since = this.#held.get(shortcut);
    // A missed `Released` must not disable the shortcut forever.
    if (since !== undefined && now - since < this.staleMs) return false;
    this.#held.set(shortcut, now);
    return true;
  }

  reset(): void {
    this.#held.clear();
  }
}
