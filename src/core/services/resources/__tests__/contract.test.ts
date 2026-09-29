/**
 * Drift guard between the Rust sysinfo / resources code and its TS mirror: every catalog id the
 * Rust policy can return has a label, and every reason / eligibility code, subsystem and topic has
 * an English UI string (in public/i18n/en or, before INTEGRATE merges it, in the P1-D2 staging file).
 * Reads source files only (no network, no Rust build).
 */
import fs from "node:fs";
import path from "node:path";
import { describe, expect, it } from "vitest";
import { modelLabel, RECOMMENDATION_TOPICS, SUBSYSTEMS } from "../format";
import { EP_KEYS } from "../../sysinfo/format";

const root = path.resolve(__dirname, "../../../../..");
const read = (p: string) => fs.readFileSync(path.join(root, p), "utf8");

function flatten(obj: Record<string, unknown>, prefix = "", out = new Map<string, string>()) {
  for (const [k, v] of Object.entries(obj)) {
    const key = prefix ? `${prefix}.${k}` : k;
    if (v && typeof v === "object") flatten(v as Record<string, unknown>, key, out);
    else out.set(key, String(v));
  }
  return out;
}

function englishKeys(): Map<string, string> {
  const keys = flatten(JSON.parse(read("public/i18n/en/translation.json")));
  const staging = path.join(root, "scripts/i18n-staging/P1-D2.json");
  if (fs.existsSync(staging)) flatten(JSON.parse(fs.readFileSync(staging, "utf8")), "", keys);
  return keys;
}

const policy = read("src-tauri/src/services/resources/policy.rs");
const eps = read("src-tauri/src/services/sysinfo/eps.rs");

describe("Rust ↔ TS contract for sysinfo / resources", () => {
  it("labels every catalog id the Rust policy can recommend", () => {
    const idsBlock = policy.slice(policy.indexOf("pub mod ids"), policy.indexOf("pub const RESERVE_PERCENT"));
    const ids = [...idsBlock.matchAll(/pub const \w+: &str = "([^"]+)";/g)].map((m) => m[1]);
    expect(ids.length).toBeGreaterThanOrEqual(10);
    for (const id of ids) expect(modelLabel(id), id).not.toBe(id);
  });

  it("has an English string for every reason code, eligibility code, subsystem and topic", () => {
    const keys = englishKeys();
    const reasonBlock = policy.slice(policy.indexOf("pub fn as_str(self)"), policy.indexOf("/// A candidate model"));
    const reasons = [...reasonBlock.matchAll(/=> "([a-z_]+)"/g)].map((m) => m[1]);
    expect(reasons).toContain("vc_running");
    for (const r of reasons) expect(keys.has(`models.gpu.rec_reason_${r}`), r).toBe(true);

    const codes = [...eps.matchAll(/pub const [A-Z_]+: &str = "([a-z_]+)";/g)].map((m) => m[1]);
    expect(codes).toContain("gpu_too_old");
    for (const c of codes.filter((c) => c !== "ok")) expect(keys.has(`models.gpu.ep_code_${c}`), c).toBe(true);

    for (const s of SUBSYSTEMS) expect(keys.has(`models.gpu.subsystem_${s}`), s).toBe(true);
    for (const t of RECOMMENDATION_TOPICS) expect(keys.has(`models.gpu.rec_topic_${t}`), t).toBe(true);
    for (const status of ["ok", "unverified", "no"]) expect(keys.has(`models.gpu.ep_${status}`), status).toBe(true);
    expect(EP_KEYS).toEqual(["cuda", "tensorrt", "tensorrt_rtx", "directml"]);
  });
});
