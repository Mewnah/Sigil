import fs from "node:fs";
import path from "node:path";
import { afterEach, describe, expect, it } from "vitest";
import { loadCatalog } from "../../release/check-catalog-pins.mjs";
import {
  RELEASE_BASE_URL,
  assetOfUrl,
  buildPlan,
  collectAssets,
  comparePin,
  formatBytes,
  hashFile,
  isPinnedFile,
  parsePlanArgs,
  planStaging,
  planToJson,
  scanStaging,
  selectIds,
} from "../assets-v1.mjs";
import { BASE, BYTES, gitOid, makeWorld, sha256 } from "./fixtures.mjs";

const worlds = [];
afterEach(() => {
  while (worlds.length) worlds.pop().cleanup();
});
const world = (opts) => {
  const w = makeWorld(opts);
  worlds.push(w);
  return w;
};

describe("assetOfUrl", () => {
  it("accepts flat assets of the release and nothing else", () => {
    expect(RELEASE_BASE_URL).toBe(BASE);
    expect(assetOfUrl(`${BASE}pack-1.zip`)).toEqual({ name: "pack-1.zip" });
    expect(assetOfUrl(`${BASE}a%20b.bin`)).toEqual({ name: "a b.bin" });
    expect(assetOfUrl("https://huggingface.co/o/r/resolve/main/x")).toBeNull();
    expect(assetOfUrl("https://github.com/Mewnah/Sigil-assets/releases/download/other-tag/x.zip")).toBeNull();
  });
  it("reports a nested or empty name as an error (a release has no folders)", () => {
    expect(assetOfUrl(`${BASE}folder/n.txt`)).toEqual({ error: expect.stringContaining("not a flat asset name") });
    expect(assetOfUrl(BASE)).toEqual({ error: expect.stringContaining("no asset name") });
    expect(assetOfUrl(`${BASE}%E0%A4%A`)).toEqual({ error: expect.stringContaining("percent-encoding") });
  });
});

describe("pins", () => {
  it("isPinnedFile needs a hash and a positive size", () => {
    expect(isPinnedFile({ size: 1, sha256: "a".repeat(64) })).toBe(true);
    expect(isPinnedFile({ size: 1, git_oid: "a".repeat(40) })).toBe(true);
    expect(isPinnedFile({ size: 0, sha256: "a".repeat(64) })).toBe(false);
    expect(isPinnedFile({ size: 5 })).toBe(false);
  });
  it("comparePin explains the first difference", () => {
    const row = { size: 3, sha256: "a".repeat(64) };
    expect(comparePin(row, { size: 4, sha256: "a".repeat(64) })).toMatch(/size 4, catalog 3/);
    expect(comparePin(row, { size: 3, sha256: "b".repeat(64) })).toMatch(/sha256 b+, catalog a+/);
    expect(comparePin(row, { size: 3, sha256: "a".repeat(64) })).toBeNull();
    expect(comparePin({ size: 3, git_oid: "c".repeat(40) }, { size: 3, sha256: "x", git_oid: "d".repeat(40) })).toMatch(/git blob id/);
    expect(comparePin({ size: 3 }, { size: 3, sha256: "x" })).toMatch(/no pin/);
  });
  it("formatBytes", () => {
    expect(formatBytes(5)).toBe("5 B");
    expect(formatBytes(2048)).toBe("2.0 KiB");
    expect(formatBytes(3 * 1024 * 1024)).toBe("3.0 MiB");
    expect(formatBytes(5 * 1024 ** 3)).toBe("5.00 GiB");
  });
  it("hashFile gives sha256 and the git blob id", async () => {
    const w = world();
    const f = path.join(w.staging, "t-multi-a.bin");
    expect(await hashFile(f, { gitOid: true })).toEqual({ size: BYTES["t-multi-a.bin"].length, sha256: sha256(BYTES["t-multi-a.bin"]), git_oid: gitOid(BYTES["t-multi-a.bin"]) });
  });
});

describe("collectAssets and planStaging", () => {
  it("lists every assets-v1 file, marks upstream-backed ones and invalid URLs", () => {
    const w = world();
    const rows = collectAssets(loadCatalog(w.catalogDir));
    const by = Object.fromEntries(rows.map((r) => [`${r.id}:${r.path}`, r]));
    expect(rows.length).toBe(7); // solo 1, multi 2, dep 1, pack 1, up 1, nested 1 (t/pub has no assets-v1 source)
    expect(by["t/solo:model.onnx"]).toMatchObject({ asset: "t-solo-model.onnx", pinned: true, upstream: false, status: "unpublished", archive: false });
    expect(by["t/pack:pack-1.zip"]).toMatchObject({ asset: "pack-1.zip", pinned: false, archive: true });
    expect(by["t/up:model.gguf"]).toMatchObject({ asset: "up-model.gguf", upstream: true, pinned: true });
    expect(by["t/nested:n.txt"]).toMatchObject({ asset: null, error: expect.stringContaining("flat") });
  });

  it("classifies assets and entries", () => {
    const w = world({ stage: ["t-solo-model.onnx", "t-multi-a.bin", "pack-1.zip", "stray.bin"] });
    const rows = collectAssets(loadCatalog(w.catalogDir));
    // t-multi-a.bin has the wrong size in the staging folder.
    fs.writeFileSync(path.join(w.staging, "t-multi-a.bin"), Buffer.from("different length bytes"));
    const plan = planStaging(rows, scanStaging(w.staging));
    const state = Object.fromEntries(plan.files.map((f) => [f.asset, f.state]));
    expect(state).toMatchObject({
      "t-solo-model.onnx": "ok",
      "t-multi-a.bin": "size-mismatch",
      "t-multi-b.json": "missing",
      "pack-1.zip": "unpinned",
      "t-dep-model.onnx": "missing",
      "up-model.gguf": "missing",
    });
    expect(plan.files.find((f) => f.asset === "up-model.gguf").optional).toBe(true);
    expect(plan.extras).toEqual(["stray.bin"]);
    expect(plan.problems.map((p) => p.message).join("\n")).toMatch(/t\/nested n\.txt: .*flat/);
    const e = Object.fromEntries([...plan.entries].map(([id, v]) => [id, v]));
    expect(e["t/solo"]).toMatchObject({ ready: true, staged: true });
    expect(e["t/multi"].ready).toBe(false);
    expect(e["t/multi"].reasons.join()).toMatch(/a\.bin: size-mismatch/);
    expect(e["t/pack"]).toMatchObject({ ready: false });
    // An upstream-backed entry is ready with nothing staged, but not "staged".
    expect(e["t/up"]).toMatchObject({ ready: true, staged: false });
  });

  it("two entries naming one asset with different pins conflict", () => {
    const rows = [
      { id: "a/x", path: "m", asset: "same.bin", size: 3, sha256: "a".repeat(64), pinned: true, upstream: false, status: "unpublished" },
      { id: "b/y", path: "m", asset: "same.bin", size: 4, sha256: "b".repeat(64), pinned: true, upstream: false, status: "unpublished" },
    ];
    const plan = planStaging(rows, new Map([["same.bin", { file: "f", size: 3 }]]));
    expect(plan.files[0].state).toBe("conflict");
    expect(plan.entries.get("a/x").ready).toBe(false);
  });

  it("scanStaging ignores folders, dotfiles, partials and the sidecars", () => {
    const w = world({ stage: ["t-solo-model.onnx"] });
    fs.mkdirSync(path.join(w.staging, "sub"));
    fs.writeFileSync(path.join(w.staging, ".hidden"), "x");
    fs.writeFileSync(path.join(w.staging, "big.zip.partial"), "x");
    fs.writeFileSync(path.join(w.staging, "SHA256SUMS.txt"), "x");
    fs.writeFileSync(path.join(w.staging, "MANIFEST.json"), "{}");
    expect([...scanStaging(w.staging).keys()]).toEqual(["t-solo-model.onnx"]);
  });
});

describe("selectIds", () => {
  it("all-staged = unpublished, ready and at least one staged file", () => {
    const w = world({ stage: ["t-solo-model.onnx", "t-multi-a.bin", "t-multi-b.json", "up-model.gguf"] });
    const catalog = loadCatalog(w.catalogDir);
    const plan = planStaging(collectAssets(catalog), scanStaging(w.staging));
    // t/pack is unstaged and unpinned; t/dep has no file; t/up has its optional mirror staged.
    expect(selectIds("all-staged", catalog, plan).ids.sort()).toEqual(["t/multi", "t/solo", "t/up"]);
    const none = planStaging(collectAssets(catalog), new Map());
    expect(selectIds("all-staged", catalog, none).ids).toEqual([]);
  });
  it("explicit ids are checked against the catalog; all-staged needs a staging plan", () => {
    const w = world();
    const catalog = loadCatalog(w.catalogDir);
    expect(selectIds("t/solo, t/multi t/solo", catalog, null)).toEqual({ ids: ["t/solo", "t/multi"], errors: [] });
    expect(selectIds("t/solo,nope/x", catalog, null).errors).toEqual(["nope/x is not a catalog id"]);
    expect(selectIds("all-staged", catalog, null).errors[0]).toMatch(/needs --staging/);
    expect(selectIds("", catalog, null).errors[0]).toMatch(/--ids needs/);
    expect(selectIds("all-staged,t/solo", catalog, planStaging([], new Map())).errors[0]).toMatch(/cannot be combined/);
  });
});

describe("buildPlan and the PowerShell uploader's JSON", () => {
  it("uploads only files that equal their pin; --hash catches wrong bytes of the right size", async () => {
    const w = world({ stage: ["t-solo-model.onnx", "t-multi-a.bin", "t-multi-b.json"] });
    // Same length as the pinned bytes, other content.
    fs.writeFileSync(path.join(w.staging, "t-multi-b.json"), Buffer.from('{"b":2}'));
    const quick = await buildPlan({ staging: w.staging, catalogDir: w.catalogDir });
    expect(planToJson(quick, w.staging).upload.map((f) => f.asset)).toEqual(["t-multi-a.bin", "t-multi-b.json", "t-solo-model.onnx"]);
    const hashed = await buildPlan({ staging: w.staging, catalogDir: w.catalogDir, hash: true });
    const json = planToJson(hashed, w.staging);
    expect(json.upload.map((f) => f.asset)).toEqual(["t-multi-a.bin", "t-solo-model.onnx"]);
    expect(json.blocked).toEqual([expect.objectContaining({ asset: "t-multi-b.json", state: "hash-mismatch" })]);
    expect(hashed.entries.get("t/multi").ready).toBe(false);
    expect(json.totals).toEqual({ files: 2, bytes: BYTES["t-multi-a.bin"].length + BYTES["t-solo-model.onnx"].length });
    expect(json.notStaged.map((f) => f.asset)).toContain("pack-1.zip");
    expect(json.optionalMirrors.map((f) => f.asset)).toEqual(["up-model.gguf"]);
  });
});

describe("parsePlanArgs", () => {
  it("needs the plan command and --staging", () => {
    expect(parsePlanArgs(["plan", "--staging", "d", "--json", "--hash"])).toMatchObject({ staging: "d", json: true, hash: true });
    expect(() => parsePlanArgs(["plan"])).toThrow(/--staging/);
    expect(() => parsePlanArgs(["upload"])).toThrow(/unknown command/);
    expect(() => parsePlanArgs(["plan", "--staging", "d", "--nope"])).toThrow(/unknown argument/);
    expect(parsePlanArgs(["--help"]).help).toBe(true);
  });
});
