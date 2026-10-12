import { describe, expect, it } from "vitest";
import { nanoid } from "nanoid";
import { util } from "peerjs";

/**
 * Runs the real PeerJS (not mocked here): the id a /client mirror signals with must pass PeerJS's
 * own `validateId`, or `new Peer(id)` aborts with `invalid-id` and the OBS source fails to start.
 * `nanoid(64)` failed that rule for about 11 % of its ids (a leading, trailing or doubled `_` / `-`).
 */

import { newMirrorPeerId } from "../provider";

const SAMPLES = 20_000;

describe("newMirrorPeerId", () => {
  it("passes PeerJS's validateId for every one of 20,000 ids", () => {
    const rejected: string[] = [];
    for (let i = 0; i < SAMPLES; i++) {
      const id = newMirrorPeerId();
      if (!util.validateId(id)) rejected.push(id);
    }
    expect(rejected).toEqual([]);
  });

  it("is long, alphanumeric and different every time", () => {
    const ids = new Set<string>();
    for (let i = 0; i < 2_000; i++) ids.add(newMirrorPeerId());
    expect(ids.size).toBe(2_000);
    for (const id of ids) expect(id).toMatch(/^[A-Za-z0-9]{32}$/);
  });

  it("is checked against the strict rule: the old nanoid(64) ids failed it about one time in nine", () => {
    // The check above would be worthless if this validator let everything through.
    expect(util.validateId("_abc")).toBe(false);
    expect(util.validateId("abc-")).toBe(false);
    expect(util.validateId("a--b")).toBe(false);
    expect(util.validateId("a-b_c")).toBe(true);
    let rejected = 0;
    for (let i = 0; i < 5_000; i++) if (!util.validateId(nanoid(64))) rejected++;
    expect(rejected).toBeGreaterThan(300);
    expect(rejected).toBeLessThan(900);
  });
});
