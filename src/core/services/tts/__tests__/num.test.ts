import { describe, expect, it } from "vitest";
import { clampedOr, finiteOr, positiveOr } from "../num";

describe("finiteOr", () => {
  it("parses numeric strings, including 0", () => {
    expect(finiteOr("1.25", 1)).toBe(1.25);
    expect(finiteOr(" -0.15 ", 0)).toBe(-0.15);
    // The old `parseFloat(x) || 1` turned a valid 0 into the fallback.
    expect(finiteOr("0", 1)).toBe(0);
    expect(finiteOr("0.0", 1)).toBe(0);
    expect(finiteOr(0, 1)).toBe(0);
  });

  it("falls back on blanks and junk instead of returning NaN", () => {
    // The old `parseFloat(x) ?? 1` returned NaN here (NaN is not nullish), which JSON sends as null.
    for (const junk of ["", "   ", "abc", "NaN", "1.0abc", "Infinity", "-Infinity"]) {
      expect(finiteOr(junk, 1)).toBe(1);
    }
    expect(finiteOr(Number.NaN, 2)).toBe(2);
    expect(finiteOr(Number.POSITIVE_INFINITY, 2)).toBe(2);
    expect(finiteOr(undefined, 3)).toBe(3);
    expect(finiteOr(null, 3)).toBe(3);
    expect(finiteOr({}, 3)).toBe(3);
  });

  it("never returns a non-finite number", () => {
    for (const v of ["", "x", "NaN", Number.NaN, Number.POSITIVE_INFINITY, undefined]) {
      expect(Number.isFinite(finiteOr(v, 1))).toBe(true);
    }
  });
});

describe("clampedOr", () => {
  it("clamps into range", () => {
    expect(clampedOr("5", 1, 0, 2)).toBe(2);
    expect(clampedOr("-1", 1, 0, 2)).toBe(0);
    expect(clampedOr("0.5", 1, 0, 2)).toBe(0.5);
  });

  it("uses the (clamped) fallback for junk", () => {
    expect(clampedOr("abc", 1, 0, 2)).toBe(1);
    expect(clampedOr("abc", 9, 0, 2)).toBe(2);
  });

  it("keeps 0 when 0 is inside the range", () => {
    expect(clampedOr("0", 1, 0, 1)).toBe(0);
  });
});

describe("positiveOr", () => {
  it("accepts only numbers above 0", () => {
    expect(positiveOr("1.5", 1)).toBe(1.5);
    expect(positiveOr("0", 1)).toBe(1);
    expect(positiveOr("-2", 1)).toBe(1);
    expect(positiveOr("", 1)).toBe(1);
  });
});
