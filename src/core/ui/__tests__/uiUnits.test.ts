/**
 * The px helpers for code that has to compute in pixels (floating-ui offsets, the info popover, the drawer drag): they
 * read the root font-size, 16px x Interface size, so a length written as "px at size 1" follows the interface.
 */
import { afterEach, describe, expect, it, vi } from "vitest";
import { BASE_REM_PX, cssLength, remPx, screenPxToRem, uiPx } from "../uiUnits";

/** A document whose root has the given computed font-size, as the browser reports it ("24px"). */
function rootFontSize(fontSize: string) {
  vi.stubGlobal("document", { documentElement: {} });
  vi.stubGlobal("getComputedStyle", () => ({ fontSize }));
}

afterEach(() => {
  vi.unstubAllGlobals();
});

describe("remPx", () => {
  it("is 16 without a document (tests, server rendering)", () => {
    expect(remPx()).toBe(BASE_REM_PX);
    expect(BASE_REM_PX).toBe(16);
  });

  it("is the root font-size: 16 x Interface size", () => {
    rootFontSize("16px");
    expect(remPx()).toBe(16);
    rootFontSize("24px");
    expect(remPx()).toBe(24);
    rootFontSize("32px");
    expect(remPx()).toBe(32);
  });

  it("follows the font-size transition: a fractional mid-way value is used as it is", () => {
    rootFontSize("16.4509px");
    expect(remPx()).toBeCloseTo(16.4509, 4);
  });

  it("falls back to 16 for a value that is not a usable size", () => {
    for (const bad of ["", "normal", "NaNpx", "0px", "-4px"]) {
      rootFontSize(bad);
      expect(remPx()).toBe(16);
    }
  });
});

describe("uiPx", () => {
  it("is the same number at Interface size 1", () => {
    rootFontSize("16px");
    expect([8, 20, 100, 320].map(uiPx)).toEqual([8, 20, 100, 320]);
  });

  it("scales a px written for size 1: 8 is 12 at 150% and 16 at 200%", () => {
    rootFontSize("24px");
    expect(uiPx(8)).toBe(12);
    expect(uiPx(320)).toBe(480);
    rootFontSize("32px");
    expect(uiPx(8)).toBe(16);
  });

  it("keeps a manual S size small: 80% is 6.4 for an 8", () => {
    rootFontSize("12.8px");
    expect(uiPx(8)).toBeCloseTo(6.4, 10);
  });
});

describe("screenPxToRem", () => {
  it("turns a mouse distance into the rem a width prop takes, whatever the size", () => {
    rootFontSize("16px");
    expect(screenPxToRem(350)).toBe(21.875);
    rootFontSize("24px");
    expect(screenPxToRem(525)).toBe(21.875);
  });

  it("is the inverse of rem to screen px at the current size", () => {
    rootFontSize("30.4px");
    expect(screenPxToRem(304) * remPx()).toBeCloseTo(304, 8);
  });
});

describe("cssLength", () => {
  it("reads a number as rem, which is what Modal.Body and Drawer take", () => {
    expect(cssLength(31.25)).toBe("31.25rem");
    expect(cssLength(26.25)).toBe("26.25rem");
    expect(cssLength(0)).toBe("0rem");
  });

  it("passes a string through (auto, a calc, a min())", () => {
    expect(cssLength("auto")).toBe("auto");
    expect(cssLength("min(90vw, 32rem)")).toBe("min(90vw, 32rem)");
  });

  it("keeps the 500 px of the old default modal: 31.25rem is 500px at 16px per rem", () => {
    expect(31.25 * BASE_REM_PX).toBe(500);
    expect(26.25 * BASE_REM_PX).toBe(420);
    expect(23.75 * BASE_REM_PX).toBe(380);
    expect(17.5 * BASE_REM_PX).toBe(280);
  });
});
