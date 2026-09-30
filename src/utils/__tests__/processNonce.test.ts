import { describe, expect, it } from "vitest";
import { makeNonce, PROCESS_ID_BASE, PROCESS_NONCE } from "../processNonce";

describe("processNonce", () => {
  it("makes a base36 tag that fits segment-id streams", () => {
    expect(PROCESS_NONCE).toMatch(/^[0-9a-z]{6}$/);
    expect(makeNonce(12)).toMatch(/^[0-9a-z]{12}$/);
    // Not a constant: 20 fresh tags of 12 characters are all different.
    expect(new Set(Array.from({ length: 20 }, () => makeNonce(12))).size).toBe(20);
  });

  it("makes a safe integer id base with room to count up", () => {
    expect(Number.isSafeInteger(PROCESS_ID_BASE)).toBe(true);
    expect(PROCESS_ID_BASE).toBeGreaterThanOrEqual(0);
    expect(PROCESS_ID_BASE).toBeLessThan(2 ** 40);
    expect(Number.isSafeInteger(PROCESS_ID_BASE + 1e9)).toBe(true);
  });
});
