/**
 * `rustTreatsAsThisPc` mirrors Rust's `is_loopback_host` (osc.rs). Both are checked against the same table,
 * `src-tauri/src/services/osc_host_cases.json`, so the Windows Firewall notes of the VRChat settings and the plugin agree about
 * which VRChat addresses are this PC.
 */
import fs from "node:fs";
import path from "node:path";
import { describe, expect, it } from "vitest";
import { effectiveOscHost, rustTreatsAsThisPc } from "../rustHost";

const TABLE = JSON.parse(fs.readFileSync(path.resolve(__dirname, "../../../../../../src-tauri/src/services/osc_host_cases.json"), "utf8")) as {
  thisPc: string[];
  elsewhere: string[];
};

describe("rustTreatsAsThisPc", () => {
  it("has a table worth checking", () => {
    expect(TABLE.thisPc.length).toBeGreaterThan(10);
    expect(TABLE.elsewhere.length).toBeGreaterThan(10);
  });

  it("agrees with Rust on every address that is this PC", () => {
    for (const host of TABLE.thisPc) expect(rustTreatsAsThisPc(host), JSON.stringify(host)).toBe(true);
  });

  it("agrees with Rust on every address that is not", () => {
    for (const host of TABLE.elsewhere) expect(rustTreatsAsThisPc(host), JSON.stringify(host)).toBe(false);
  });

  it("is false for anything that is not text", () => {
    for (const host of [undefined, null, 0, {}]) expect(rustTreatsAsThisPc(host)).toBe(false);
  });
});

describe("effectiveOscHost", () => {
  it("trims, and an empty or missing setting means the default 127.0.0.1", () => {
    expect(effectiveOscHost(" 192.168.1.5 ")).toBe("192.168.1.5");
    for (const raw of ["", "   ", undefined, null]) expect(effectiveOscHost(raw)).toBe("127.0.0.1");
    expect(rustTreatsAsThisPc(effectiveOscHost(""))).toBe(true);
  });
});
