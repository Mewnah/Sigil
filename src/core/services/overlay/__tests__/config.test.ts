import { describe, expect, it } from "vitest";
import { OthersEventSource, TextEventSource } from "@/types";
import { Service_VRC_Schema } from "../../vrc/schema";
import { CaptionWindow_Schema } from "../schema";
import {
  acceptsTopic,
  captionWindowOptions,
  captionWindowStyle,
  clampNumber,
  feedOptions,
  normalizeSources,
  toFrameLines,
  toVrCaption,
  vrOverlayConfig,
  VR_LIMITS,
} from "../config";
import type { CaptionModelLine } from "../captionModel";

describe("defaults", () => {
  it("both displays are off by default", () => {
    expect(CaptionWindow_Schema.parse(undefined).enable).toBe(false);
    expect(Service_VRC_Schema.parse(undefined).overlay.enable).toBe(false);
  });

  it("the default VR config matches the Rust defaults", () => {
    expect(vrOverlayConfig(Service_VRC_Schema.parse(undefined).overlay)).toEqual({
      attach: "head",
      widthM: 0.6,
      distanceM: 1.2,
      offsetYM: -0.25,
      fontSizePx: 36,
      opacity: 0.9,
      maxLines: 3,
      textColor: "rgba(255,255,255,1)",
      backgroundColor: "rgba(0,0,0,0.6)",
    });
  });

  it("the default window", () => {
    const s = CaptionWindow_Schema.parse(undefined);
    expect(captionWindowOptions(s)).toEqual({ alwaysOnTop: true, clickThrough: true });
    expect(captionWindowStyle(s)).toMatchObject({ fontSize: 28, maxLines: 3 });
    expect(captionWindowStyle(s).fontFamily.startsWith('"Outfit", ')).toBe(true);
    expect(feedOptions(s, [1, 20])).toEqual({ interim: true, maxLines: 3, clearAfterMs: 6000 });
  });
});

describe("clamping", () => {
  it("numbers from strings, clamped, with fallbacks", () => {
    expect(clampNumber("2.5", [0, 1], 0.5)).toBe(1);
    expect(clampNumber("", [0, 1], 0.5)).toBe(0.5);
    expect(clampNumber("abc", [0, 1], 0.5)).toBe(0.5);
    expect(clampNumber(-3, [0, 1], 0.5)).toBe(0);
  });

  it("VR config is clamped like Rust", () => {
    const base = Service_VRC_Schema.parse(undefined).overlay;
    const c = vrOverlayConfig({ ...base, attach: "world", widthM: "100", distanceM: "0", offsetYM: "-9", fontSizePx: "4", opacity: "0", maxLines: "50" });
    expect(c).toMatchObject({
      attach: "world",
      widthM: VR_LIMITS.widthM[1],
      distanceM: VR_LIMITS.distanceM[0],
      offsetYM: VR_LIMITS.offsetYM[0],
      fontSizePx: VR_LIMITS.fontSizePx[0],
      opacity: VR_LIMITS.opacity[0],
      maxLines: VR_LIMITS.maxLines[1],
    });
  });

  it("clear timer is bounded", () => {
    expect(feedOptions({ interim: false, maxLines: "0", clearAfterMs: "-5" }, [1, 8])).toEqual({ interim: false, maxLines: 1, clearAfterMs: 0 });
    expect(feedOptions({ interim: true, maxLines: "99", clearAfterMs: "99999999" }, [1, 8]).clearAfterMs).toBe(600_000);
  });
});

describe("sources", () => {
  it("drop duplicates and topics an 'any' source already covers", () => {
    expect(normalizeSources([TextEventSource.stt, TextEventSource.stt, OthersEventSource.stt])).toEqual([TextEventSource.stt, OthersEventSource.stt]);
    expect(normalizeSources([TextEventSource.stt, TextEventSource.any, TextEventSource.translation, OthersEventSource.translation])).toEqual([
      TextEventSource.any,
      OthersEventSource.translation,
    ]);
    expect(normalizeSources([OthersEventSource.stt, OthersEventSource.any])).toEqual([OthersEventSource.any]);
  });

  it("'any text' never shows the AI transform's echoes; other players never come through 'text'", () => {
    expect(acceptsTopic(TextEventSource.any, TextEventSource.stt)).toBe(true);
    expect(acceptsTopic(TextEventSource.any, TextEventSource.transform_source)).toBe(false);
    expect(acceptsTopic(TextEventSource.any, TextEventSource.transform_raw)).toBe(false);
    expect(acceptsTopic(TextEventSource.transform, TextEventSource.transform)).toBe(true);
    expect(acceptsTopic(OthersEventSource.stt, OthersEventSource.stt)).toBe(true);
  });
});

describe("wire lines", () => {
  const lines: CaptionModelLine[] = [
    { topic: "text.stt", kind: "final", key: "l1", text: "Hi", source: "text.stt", updatedAt: 0 },
    { topic: "others.stt", kind: "interim", key: "l2", text: "こんにちは", stableLength: 2, lang: "ja", source: "others.stt", updatedAt: 0 },
  ];

  it("frame lines carry the tentative tail of interims only", () => {
    expect(toFrameLines(lines)).toEqual([
      { key: "l1", text: "Hi", interim: false, stable: null, lang: null, source: "text.stt" },
      { key: "l2", text: "こんにちは", interim: true, stable: 2, lang: "ja", source: "others.stt" },
    ]);
  });

  it("VR lines", () => {
    expect(toVrCaption(lines)).toEqual({
      lines: [
        { text: "Hi", kind: "final", source: "text.stt", lang: null },
        { text: "こんにちは", kind: "interim", source: "others.stt", lang: "ja" },
      ],
    });
  });
});
