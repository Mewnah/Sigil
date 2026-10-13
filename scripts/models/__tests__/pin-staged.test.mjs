import fs from "node:fs";
import path from "node:path";
import { afterEach, describe, expect, it } from "vitest";
import { loadCatalog } from "../../release/check-catalog-pins.mjs";
import { collectAssets } from "../assets-v1.mjs";
import { applyPinsToFragment, main, manifestOf, parseArgs, planPins, sumsText } from "../pin-staged.mjs";
import { BYTES, fragmentBText, makeWorld, sha256 } from "./fixtures.mjs";

const worlds = [];
afterEach(() => {
  while (worlds.length) worlds.pop().cleanup();
});
const world = (opts) => {
  const w = makeWorld(opts);
  worlds.push(w);
  return w;
};
const io = () => {
  const out = [];
  const err = [];
  return { out, err, io: { out: (m) => out.push(m), err: (m) => err.push(m) } };
};

const hashesOf = (names) => new Map(names.map((n) => [n, { size: BYTES[n].length, sha256: sha256(BYTES[n]) }]));

describe("planPins", () => {
  it("pins the unpinned, accepts the equal, flags the different", () => {
    const w = world();
    const rows = collectAssets(loadCatalog(w.catalogDir));
    const hashes = hashesOf(["pack-1.zip", "t-solo-model.onnx", "t-multi-a.bin"]);
    hashes.set("t-multi-a.bin", { size: 7, sha256: "f".repeat(64) });
    const acts = Object.fromEntries(planPins(rows, hashes).map((a) => [a.id, a.action]));
    expect(acts).toEqual({ "t/pack": "pin", "t/solo": "ok", "t/multi": "mismatch" });
    const forced = planPins(rows, hashes, { force: true }).find((a) => a.id === "t/multi");
    expect(forced.action).toBe("pin");
    expect(forced.was).toMatch(/sha256/);
    expect(planPins(rows, hashes, { only: ["t/solo"] }).map((a) => a.id)).toEqual(["t/solo"]);
  });
});

describe("applyPinsToFragment", () => {
  it("writes the pin into the hand-written layout and nothing else", () => {
    const text = fragmentBText();
    const acts = [{ id: "t/pack", path: "pack-1.zip", fact: { size: BYTES["pack-1.zip"].length, sha256: sha256(BYTES["pack-1.zip"]) } }];
    const { text: out, doc } = applyPinsToFragment(text, acts);
    expect(doc.entries[0].files[0]).toMatchObject({ size: BYTES["pack-1.zip"].length, sha256: sha256(BYTES["pack-1.zip"]) });
    const before = new Set(text.split("\n"));
    expect(out.split("\n").filter((l) => !before.has(l))).toHaveLength(2);
    expect(out).toContain('"archive": { "format": "zip", "include": ["a.dll", "sigil-pack.json"] }');
  });
  it("refuses a file the fragment does not have", () => {
    expect(() => applyPinsToFragment(fragmentBText(), [{ id: "t/pack", path: "nope.zip", fact: { size: 1, sha256: "a".repeat(64) } }])).toThrow(/not in the fragment/);
  });
});

describe("sidecars", () => {
  it("SHA256SUMS in sha256sum format and a MANIFEST naming the entries", () => {
    const w = world();
    const rows = collectAssets(loadCatalog(w.catalogDir));
    const hashes = hashesOf(["t-solo-model.onnx", "pack-1.zip"]);
    expect(sumsText(hashes)).toBe(`${sha256(BYTES["pack-1.zip"])}  pack-1.zip\n${sha256(BYTES["t-solo-model.onnx"])}  t-solo-model.onnx\n`);
    const m = manifestOf(rows, hashes, new Date("2026-10-13T00:00:00Z"));
    expect(m.schema).toBe("sigil-assets-staging/1");
    expect(m.files).toEqual([
      { asset: "pack-1.zip", size: BYTES["pack-1.zip"].length, sha256: sha256(BYTES["pack-1.zip"]), entries: ["t/pack"] },
      { asset: "t-solo-model.onnx", size: BYTES["t-solo-model.onnx"].length, sha256: sha256(BYTES["t-solo-model.onnx"]), entries: ["t/solo"] },
    ]);
  });
});

describe("main", () => {
  it("dry run prints, --write pins the zip and writes the sidecars", async () => {
    const w = world();
    const dry = io();
    expect(await main(["--staging", w.staging, "--catalog-dir", w.catalogDir], dry.io)).toBe(0);
    expect(dry.out.join("\n")).toMatch(/pin\s+t\/pack pack-1\.zip/);
    expect(fs.readFileSync(path.join(w.catalogDir, "b.json"), "utf8")).toBe(fragmentBText());
    expect(fs.existsSync(path.join(w.staging, "MANIFEST.json"))).toBe(false);

    const wr = io();
    expect(await main(["--staging", w.staging, "--catalog-dir", w.catalogDir, "--write"], wr.io)).toBe(0);
    const b = JSON.parse(fs.readFileSync(path.join(w.catalogDir, "b.json"), "utf8"));
    expect(b.entries[0].files[0]).toMatchObject({ size: BYTES["pack-1.zip"].length, sha256: sha256(BYTES["pack-1.zip"]) });
    expect(fs.readFileSync(path.join(w.staging, "SHA256SUMS.txt"), "utf8")).toContain("  pack-1.zip");
    // A second run has nothing left to pin.
    const again = io();
    expect(await main(["--staging", w.staging, "--catalog-dir", w.catalogDir], again.io)).toBe(0);
    expect(again.err.join("\n")).toMatch(/0 to pin/);
  });

  it("a staged file that differs from its pin is a MISMATCH (exit 1) until --force", async () => {
    const w = world();
    fs.writeFileSync(path.join(w.staging, "t-multi-a.bin"), Buffer.from("rebuilt"));
    const r = io();
    expect(await main(["--staging", w.staging, "--catalog-dir", w.catalogDir, "--write"], r.io)).toBe(1);
    expect(r.out.join("\n")).toMatch(/MISMATCH t\/multi a\.bin/);
    const a = JSON.parse(fs.readFileSync(path.join(w.catalogDir, "a.json"), "utf8"));
    expect(a.entries[1].files[0].sha256).toBe(sha256(BYTES["t-multi-a.bin"]));
    const f = io();
    expect(await main(["--staging", w.staging, "--catalog-dir", w.catalogDir, "--force", "--write"], f.io)).toBe(0);
    const a2 = JSON.parse(fs.readFileSync(path.join(w.catalogDir, "a.json"), "utf8"));
    expect(a2.entries[1].files[0]).toMatchObject({ size: 7, sha256: sha256(Buffer.from("rebuilt")) });
  });

  it("usage errors", async () => {
    const r = io();
    expect(await main([], r.io)).toBe(2);
    expect(await main(["--staging", "/definitely/not/here"], r.io)).toBe(2);
    expect(() => parseArgs(["--staging"])).toThrow(/needs a value/);
    expect(parseArgs(["--help"]).help).toBe(true);
  });
});
