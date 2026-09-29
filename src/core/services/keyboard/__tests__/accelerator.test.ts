import { describe, expect, it } from "vitest";
import {
  acceleratorKey,
  isStandaloneSafeKey,
  isValidAccelerator,
  KEY_TOKEN_BY_CODE,
  mainKeyToken,
  recordKeyDown,
  ShortcutPressGate,
} from "../accelerator";

type Mods = Partial<{ ctrlKey: boolean; altKey: boolean; shiftKey: boolean; metaKey: boolean }>;
const key = (code: string, mods: Mods = {}) => ({
  code,
  ctrlKey: false,
  altKey: false,
  shiftKey: false,
  metaKey: false,
  ...mods,
});
/** A keydown with the character the active layout produced (`KeyboardEvent.key`). */
const layoutKey = (code: string, produced: string, mods: Mods = {}) => ({ ...key(code, mods), key: produced });

describe("shortcut recording", () => {
  it("collects modifiers and completes on the first main key, modifiers first", () => {
    expect(recordKeyDown(key("ControlLeft", { ctrlKey: true }))).toEqual({ kind: "pending", preview: "Control+" });
    expect(recordKeyDown(key("ShiftRight", { ctrlKey: true, shiftKey: true }))).toEqual({ kind: "pending", preview: "Control+Shift+" });
    expect(recordKeyDown(key("KeyM", { ctrlKey: true, shiftKey: true }))).toEqual({ kind: "shortcut", value: "Control+Shift+M" });
    expect(recordKeyDown(key("Digit1", { altKey: true }))).toEqual({ kind: "shortcut", value: "Alt+1" });
    expect(recordKeyDown(key("NumpadMultiply", { ctrlKey: true }))).toEqual({ kind: "shortcut", value: "Control+NumpadMultiply" });
    expect(recordKeyDown(key("MetaLeft", { metaKey: true }))).toEqual({ kind: "pending", preview: "Super+" });
  });

  it("accepts function keys alone but asks for a modifier on typing keys", () => {
    expect(recordKeyDown(key("F13"))).toEqual({ kind: "shortcut", value: "F13" });
    expect(recordKeyDown(key("Pause"))).toEqual({ kind: "shortcut", value: "Pause" });
    expect(recordKeyDown(key("KeyM"))).toEqual({ kind: "needs_modifier", preview: "M" });
    expect(recordKeyDown(key("Space"))).toEqual({ kind: "needs_modifier", preview: "Space" });
  });

  it("does not accept Shift alone on typing keys (Shift+M is a capital M in every app)", () => {
    expect(recordKeyDown(key("KeyM", { shiftKey: true }))).toEqual({ kind: "needs_modifier", preview: "Shift+M" });
    expect(recordKeyDown(key("Digit1", { shiftKey: true }))).toEqual({ kind: "needs_modifier", preview: "Shift+1" });
    expect(recordKeyDown(key("Space", { shiftKey: true }))).toEqual({ kind: "needs_modifier", preview: "Shift+Space" });
    expect(recordKeyDown(key("Enter", { shiftKey: true }))).toEqual({ kind: "needs_modifier", preview: "Shift+Enter" });
    expect(recordKeyDown(key("Slash", { shiftKey: true }))).toEqual({ kind: "needs_modifier", preview: "Shift+/" });
    // Shift on top of Control / Alt / Super, or with a key that is safe alone, is fine.
    expect(recordKeyDown(key("KeyM", { shiftKey: true, ctrlKey: true }))).toEqual({ kind: "shortcut", value: "Control+Shift+M" });
    expect(recordKeyDown(key("KeyM", { shiftKey: true, altKey: true }))).toEqual({ kind: "shortcut", value: "Alt+Shift+M" });
    expect(recordKeyDown(key("KeyM", { metaKey: true }))).toEqual({ kind: "shortcut", value: "Super+M" });
    expect(recordKeyDown(key("F5", { shiftKey: true }))).toEqual({ kind: "shortcut", value: "Shift+F5" });
  });

  it("records letters as the active layout labels them (global-hotkey registers layout-dependent virtual keys)", () => {
    // German QWERTZ: the key labelled Z sits where US-QWERTY has Y.
    expect(recordKeyDown(layoutKey("KeyY", "z", { ctrlKey: true }))).toEqual({ kind: "shortcut", value: "Control+Z" });
    expect(recordKeyDown(layoutKey("KeyZ", "y", { ctrlKey: true }))).toEqual({ kind: "shortcut", value: "Control+Y" });
    // French AZERTY: A/Q and Z/W are swapped, and M sits on the US semicolon key.
    expect(recordKeyDown(layoutKey("KeyQ", "a", { ctrlKey: true }))).toEqual({ kind: "shortcut", value: "Control+A" });
    expect(recordKeyDown(layoutKey("KeyW", "z", { altKey: true }))).toEqual({ kind: "shortcut", value: "Alt+Z" });
    expect(recordKeyDown(layoutKey("Semicolon", "m", { ctrlKey: true }))).toEqual({ kind: "shortcut", value: "Control+M" });
    expect(recordKeyDown(layoutKey("KeyZ", "Z", { ctrlKey: true, shiftKey: true }))).toEqual({ kind: "shortcut", value: "Control+Shift+Z" });
    // Not an ASCII letter (Cyrillic layout, AltGr symbol, dead key, IME): the physical key, as global-hotkey maps it.
    expect(recordKeyDown(layoutKey("KeyZ", "я", { ctrlKey: true }))).toEqual({ kind: "shortcut", value: "Control+Z" });
    expect(recordKeyDown(layoutKey("KeyQ", "@", { ctrlKey: true, altKey: true }))).toEqual({ kind: "shortcut", value: "Control+Alt+Q" });
    expect(recordKeyDown(layoutKey("BracketLeft", "Dead", { ctrlKey: true }))).toEqual({ kind: "shortcut", value: "Control+[" });
    expect(mainKeyToken(layoutKey("KeyA", "Process"))).toBe("A");
    // Layout-independent keys keep their code mapping.
    expect(mainKeyToken(layoutKey("F13", "F13"))).toBe("F13");
    expect(mainKeyToken(layoutKey("Numpad1", "1"))).toBe("Numpad1");
    expect(mainKeyToken(layoutKey("Digit1", "&"))).toBe("1");
    // A layout letter still needs Control, Alt or Super.
    expect(recordKeyDown(layoutKey("KeyY", "z"))).toEqual({ kind: "needs_modifier", preview: "Z" });
  });

  it("cancels on a bare Escape and rejects unknown keys", () => {
    expect(recordKeyDown(key("Escape"))).toEqual({ kind: "cancel" });
    expect(recordKeyDown(key("IntlRo", { ctrlKey: true }))).toEqual({ kind: "unsupported", code: "IntlRo" });
  });

  it("every recorded shortcut is one global-hotkey parses", () => {
    for (const code of Object.keys(KEY_TOKEN_BY_CODE)) {
      const step = recordKeyDown(key(code, { ctrlKey: true, altKey: true }));
      expect(step.kind).toBe("shortcut");
      if (step.kind === "shortcut") expect(isValidAccelerator(step.value), step.value).toBe(true);
    }
  });
});

describe("accelerator validation", () => {
  it("accepts what global-hotkey accepts", () => {
    for (const v of ["Control+Shift+M", "CommandOrControl+Alt+F5", "F24", "Ctrl+`", "Shift+Numpad0", "Alt+Up", "Control+BackSpace", "super+keyk"])
      expect(isValidAccelerator(v), v).toBe(true);
  });

  it("rejects values the old recorder could store but global-hotkey cannot parse", () => {
    for (const v of ["", "Control+", "M+Control", "Control+Multiply", "LeftWindows+A", "Control+A+B", "Control+Print"])
      expect(isValidAccelerator(v), v).toBe(false);
  });

  it("compares shortcuts regardless of modifier order and aliases", () => {
    expect(acceleratorKey("Shift+Control+m")).toBe(acceleratorKey("Ctrl+Shift+M"));
    expect(acceleratorKey("Alt+M")).not.toBe(acceleratorKey("Control+M"));
    expect(isStandaloneSafeKey("F12")).toBe(true);
    expect(isStandaloneSafeKey("F25")).toBe(false);
    expect(isStandaloneSafeKey("A")).toBe(false);
  });
});

describe("ShortcutPressGate", () => {
  it("fires once per press: ignores Released and key-repeat Pressed", () => {
    const gate = new ShortcutPressGate();
    expect(gate.accept("F13", "Pressed", 0)).toBe(true);
    expect(gate.accept("F13", "Pressed", 30)).toBe(false);
    expect(gate.accept("F13", "Released", 100)).toBe(false);
    expect(gate.accept("F13", "Pressed", 200)).toBe(true);
  });

  it("recovers when a Released event was missed", () => {
    const gate = new ShortcutPressGate(3_000);
    expect(gate.accept("F13", "Pressed", 0)).toBe(true);
    expect(gate.accept("F13", "Pressed", 3_500)).toBe(true);
  });

  it("tracks shortcuts independently", () => {
    const gate = new ShortcutPressGate();
    expect(gate.accept("F13", "Pressed", 0)).toBe(true);
    expect(gate.accept("F14", "Pressed", 10)).toBe(true);
  });
});
