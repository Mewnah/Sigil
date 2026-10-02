import { describe, expect, it } from "vitest";
import { CAPTION_WINDOW, VR_OVERLAY_STATE_EVENT } from "@/types";
import { WINDOW_LIMITS } from "@/core/services/overlay/config";
import {
  CAPTION_FRAME_EVENT,
  CAPTION_WINDOW_LABEL,
  CAPTION_WINDOW_MAX_LINES,
  CAPTION_WINDOW_MAX_TEXT,
  CAPTION_WINDOW_PATH,
  captionFontFamily,
  readCaptionFrame,
} from "../frame";

const good = {
  seq: 3,
  style: { fontFamily: '"Outfit", sans-serif', fontSize: 30, textColor: "rgba(255,255,255,1)", backgroundColor: "#000a", maxLines: 4 },
  unlocked: true,
  hint: "Drag to move",
  lines: [
    { key: "l1", text: "Hello", interim: false, stable: null, lang: null, source: "text.stt" },
    { key: "l2", text: "こんにちは", interim: true, stable: 2, lang: "ja", source: "others.stt" },
  ],
};

describe("caption window names", () => {
  it("match Rust and the shared types", () => {
    expect(CAPTION_WINDOW_LABEL).toBe(CAPTION_WINDOW.label);
    expect(CAPTION_WINDOW_PATH).toBe(`/${CAPTION_WINDOW.route}`);
    expect(CAPTION_WINDOW_PATH).toBe("/captions");
    // Tauri event names allow alphanumerics, '-', '/', ':' and '_'.
    for (const name of [CAPTION_FRAME_EVENT, CAPTION_WINDOW.stateEvent, VR_OVERLAY_STATE_EVENT]) expect(name).toMatch(/^[A-Za-z0-9\-/:_]+$/);
  });
});

describe("readCaptionFrame", () => {
  it("accepts a well-formed frame unchanged", () => {
    expect(readCaptionFrame(good)).toEqual(good);
  });

  it("rejects what is not a frame", () => {
    for (const bad of [null, 1, "x", {}, { style: {} }, { lines: [] }]) expect(readCaptionFrame(bad)).toBeNull();
  });

  it("neutralizes unsafe or broken style values", () => {
    const f = readCaptionFrame({
      ...good,
      style: { fontFamily: "x; } body { display:none", fontSize: 9999, textColor: "red; background:url(x)", backgroundColor: 5, maxLines: -1 },
    })!;
    expect(f.style.fontFamily).not.toMatch(/[;{}]/);
    expect(f.style.fontSize).toBe(200);
    expect(f.style.textColor).toBe("rgba(255,255,255,1)");
    expect(f.style.backgroundColor).toBe("rgba(0,0,0,0.6)");
    expect(f.style.maxLines).toBe(1);
  });

  it("checks every line", () => {
    const f = readCaptionFrame({
      ...good,
      lines: [
        { key: "a", text: "  ", interim: false },
        "nope",
        { key: "b", text: "ok", interim: "yes", stable: 99, lang: "JA; x", source: 5 },
        ...Array.from({ length: 30 }, (_, i) => ({ key: `k${i}`, text: `t${i}` })),
      ],
    })!;
    expect(f.lines).toHaveLength(CAPTION_WINDOW_MAX_LINES);
    expect(f.lines[0].text).toBe(`t${30 - CAPTION_WINDOW_MAX_LINES}`);
    const g = readCaptionFrame({ ...good, lines: [{ key: "b", text: "ok", interim: "yes", stable: 99, lang: "JA; x", source: 5 }] })!;
    expect(g.lines[0]).toEqual({ key: "b", text: "ok", interim: false, stable: 2, lang: null, source: "" });
  });

  it("shows every line the settings allow", () => {
    // One cap for the settings range and the page: a frame at the largest setting reaches the box whole.
    expect(WINDOW_LIMITS.maxLines[1]).toBe(CAPTION_WINDOW_MAX_LINES);
    const max = WINDOW_LIMITS.maxLines[1];
    const lines = Array.from({ length: max }, (_, i) => ({ key: `k${i}`, text: `t${i}` }));
    const f = readCaptionFrame({ ...good, style: { ...good.style, maxLines: max }, lines })!;
    expect(f.style.maxLines).toBe(max);
    expect(f.lines.map((l) => l.text)).toEqual(lines.map((l) => l.text));
    expect(readCaptionFrame({ ...good, style: { ...good.style, maxLines: max + 5 } })!.style.maxLines).toBe(max);
  });

  it("keeps the newest text of an over-long line and moves its committed prefix with it", () => {
    const over = 503;
    const long = "a".repeat(CAPTION_WINDOW_MAX_TEXT + over - 3) + "END";
    const line = (text: string, stable: number | null) => ({ ...good, lines: [{ key: "x", text, interim: true, stable, lang: null, source: "text.stt" }] });

    const f = readCaptionFrame(line(long, 2_400))!.lines[0];
    expect(f.text).toHaveLength(CAPTION_WINDOW_MAX_TEXT);
    expect(f.text.endsWith("END")).toBe(true);
    expect(f.text).toBe(long.slice(over));
    expect(f.stable).toBe(2_400 - over);

    // The committed prefix was cut whole: everything shown is tentative.
    expect(readCaptionFrame(line(long, 100))!.lines[0].stable).toBe(0);
    // A final keeps no prefix.
    expect(readCaptionFrame(line(long, null))!.lines[0].stable).toBeNull();
    // A committed prefix covering everything still ends at the text's end.
    expect(readCaptionFrame(line(long, long.length))!.lines[0].stable).toBe(CAPTION_WINDOW_MAX_TEXT);
  });

  it("never starts an over-long line inside a surrogate pair", () => {
    // 1000 emoji (2000 units) + "b": the plain cut would start on the first emoji's low surrogate.
    const long = "😀".repeat(1_000) + "b";
    expect(long.length).toBe(CAPTION_WINDOW_MAX_TEXT + 1);
    const f = readCaptionFrame({ ...good, lines: [{ key: "x", text: long, interim: true, stable: 10, lang: null, source: "text.stt" }] })!.lines[0];
    expect(f.text).toBe("😀".repeat(999) + "b");
    expect(f.text.codePointAt(0)).toBe(0x1f600);
    expect(f.stable).toBe(8);
  });
});

describe("captionFontFamily", () => {
  it("quotes the user's font and adds CJK fallbacks", () => {
    expect(captionFontFamily("Outfit")).toMatch(/^"Outfit", "Segoe UI", "Yu Gothic UI"/);
    expect(captionFontFamily('Evil"; }')).toMatch(/^"Evil", /);
    expect(captionFontFamily("   ")).toMatch(/^"Segoe UI"/);
  });
});
