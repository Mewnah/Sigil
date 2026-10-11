import * as fs from "node:fs";
import * as os from "node:os";
import * as path from "node:path";
import { afterEach, describe, expect, it } from "vitest";
import {
  checkCatalog,
  checkFile,
  classifySource,
  compareInstalled,
  expandReleaseSet,
  idsForGroup,
  loadCatalog,
  loadReleaseSet,
  main,
  findRedistNode,
  parseArgs,
  parseRawGithubUrl,
  readInstallRecords,
  requiresClosure,
  routingCpuIds,
  verifyFileOnline,
  verifyOnline,
} from "../check-catalog-pins.mjs";

const COMMIT = "a0c709d9fac510f2c807aa3af52872340dc37a4a";
const SHA = "ab".repeat(32);
const SHA_B = "cd".repeat(32);
const OID = "5359861c739e955e79d9a303bcbc70fb988958b1";

const tmpRoots = [];
function tmp() {
  const d = fs.mkdtempSync(path.join(os.tmpdir(), "sigil-pins-"));
  tmpRoots.push(d);
  return d;
}
afterEach(() => {
  while (tmpRoots.length) fs.rmSync(tmpRoots.pop(), { recursive: true, force: true });
});

const hfSrc = (rev = COMMIT, file = "m.gguf") => ({ kind: "huggingface", url: `https://huggingface.co/org/repo/resolve/${rev}/${file}`, revision: rev });
const ghSrc = (tag = "v1", asset = "pack.zip") => ({ kind: "github-release", url: `https://github.com/o/r/releases/download/${tag}/${asset}`, revision: tag });

function entry(id, over = {}) {
  return {
    id,
    family: id.split("/")[0],
    kind: "model",
    version: "1",
    license: "MIT",
    status: "published",
    files: [{ path: "m.gguf", size: 10, sha256: SHA, sources: [hfSrc()] }],
    ...over,
  };
}

/** Write fragments into a temp catalog dir and load them. */
function catalogOf(fragments) {
  const dir = tmp();
  for (const [name, doc] of Object.entries(fragments)) fs.writeFileSync(path.join(dir, name), JSON.stringify(doc));
  return { dir, catalog: loadCatalog(dir) };
}

const emptySet = { defaults: {}, light: {} };

describe("classifySource", () => {
  it("accepts a Hugging Face commit and rejects branches, tags and a differing revision field", () => {
    expect(classifySource(hfSrc())).toMatchObject({ host: "huggingface", immutable: true });
    expect(classifySource({ kind: "huggingface", url: "https://huggingface.co/org/repo/resolve/main/m.gguf" })).toMatchObject({ immutable: false });
    expect(classifySource({ kind: "huggingface", url: "https://huggingface.co/org/repo/resolve/v1.0/m.gguf" }).reason).toMatch(/branch or tag/);
    expect(classifySource({ ...hfSrc(), revision: "b".repeat(40) }).reason).toMatch(/differs/);
  });

  it("accepts release tags, refuses moving ones and recognises rolling releases", () => {
    expect(classifySource(ghSrc("b11146"))).toMatchObject({ host: "github-release", immutable: true, rolling: false });
    expect(classifySource(ghSrc("latest"))).toMatchObject({ immutable: false });
    expect(classifySource(ghSrc("Nightly")).reason).toMatch(/moving/);
    const set = { rollingTags: [{ tag: "O/R@asr-models", why: "x" }] };
    expect(classifySource(ghSrc("asr-models"), set)).toMatchObject({ immutable: true, rolling: true, key: "o/r@asr-models" });
    expect(classifySource(ghSrc("v2"), set).rolling).toBe(false);
  });

  it("wants a commit in raw.githubusercontent.com URLs", () => {
    const raw = (ref) => ({ kind: "direct", url: `https://raw.githubusercontent.com/o/r/${ref}/dir/m.onnx` });
    expect(classifySource(raw(OID))).toMatchObject({ host: "github-raw", immutable: true });
    expect(classifySource(raw("main"))).toMatchObject({ immutable: false });
  });

  it("wants a version in NVIDIA redistributable paths, trusts Mozilla attachments and flags hash-only hosts", () => {
    expect(classifySource({ kind: "nvidia-redist", url: "https://developer.download.nvidia.com/compute/cuda/redist/x/cuda_cudart-windows-x86_64-13.2.51-archive.zip" }).immutable).toBe(true);
    expect(classifySource({ kind: "nvidia-redist", url: "https://developer.download.nvidia.com/compute/latest/x.zip" }).immutable).toBe(false);
    expect(classifySource({ kind: "direct", url: "https://firefox-settings-attachments.cdn.mozilla.net/main-workspace/translations-models/abc.bin" }).immutable).toBe(true);
    expect(classifySource({ kind: "direct", url: "https://alphacephei.com/vosk/models/vosk-model-small-en-us-0.15.zip" })).toMatchObject({ immutable: false, hashOnly: true });
  });

  it("refuses plain http and garbage", () => {
    expect(classifySource({ kind: "direct", url: "http://example.com/x" })).toMatchObject({ immutable: false, reason: "not https" });
    expect(classifySource({ kind: "direct", url: "not a url" })).toMatchObject({ immutable: false });
  });
});

describe("checkFile", () => {
  const rules = (e, f, o) => checkFile(e, f, o).map((p) => `${p.level}:${p.rule}`);

  it("passes a pinned published file", () => {
    const e = entry("mt/x");
    expect(checkFile(e, e.files[0])).toEqual([]);
  });

  it("refuses a published file without a hash, with a malformed hash, with both, or with no size", () => {
    const e = entry("mt/x");
    expect(rules(e, { path: "a", size: 1, sources: [hfSrc()] })).toContain("error:pin.missing");
    expect(rules(e, { path: "a", size: 1, sha256: "XYZ", sources: [hfSrc()] })).toContain("error:pin.shape");
    expect(rules(e, { path: "a", size: 1, git_oid: SHA, sources: [hfSrc()] })).toContain("error:pin.shape");
    expect(rules(e, { path: "a", size: 1, sha256: SHA, git_oid: OID, sources: [hfSrc()] })).toContain("error:pin.both");
    expect(rules(e, { path: "a", size: 0, sha256: SHA, sources: [hfSrc()] })).toContain("error:pin.size");
    expect(rules(e, { path: "a", size: 5, git_oid: OID, sources: [hfSrc()] })).toEqual([]);
  });

  it("lets an unpublished import-only entry stay unpinned", () => {
    const e = entry("runtime/p", { status: "unpublished" });
    expect(checkFile(e, { path: "p.zip", size: 0, sources: [ghSrc("assets-v1", "p.zip")] })).toEqual([]);
  });

  it("refuses a published file whose source can change", () => {
    const e = entry("mt/x");
    expect(rules(e, { path: "a", size: 1, sha256: SHA, sources: [hfSrc("main")] })).toContain("error:src.mutable");
    expect(rules(e, { path: "a", size: 1, sha256: SHA, sources: [ghSrc("latest")] })).toContain("error:src.mutable");
  });

  it("accepts a hash-only host in the lenient pass and, strictly, only when the release set lists it", () => {
    const e = entry("vosk/small-en", { files: [{ path: "v.zip", size: 9, sha256: SHA, sources: [{ kind: "direct", url: "https://alphacephei.com/vosk/models/v.zip" }] }] });
    expect(checkFile(e, e.files[0])).toEqual([]);
    expect(rules(e, e.files[0], { strict: true, set: {} })).toEqual(["error:src.hash-only"]);
    expect(rules(e, e.files[0], { strict: true, set: { tofu: [{ prefix: "vosk/", why: "md5 only. More." }] } })).toEqual(["note:src.hash-only"]);
  });
});

describe("release set", () => {
  const { catalog } = catalogOf({
    "a.json": { entries: [entry("stt/a"), entry("stt/b", { requires: ["runtime/r"] }), entry("runtime/r", { kind: "runtime" }), entry("tts/piper-1"), entry("tts/piper-2")] },
  });

  it("expands ids and prefixes, keeping the default tier when both tiers select an id", () => {
    const set = { defaults: { g: { ids: ["stt/a"] } }, light: { l: { ids: ["stt/a", "stt/b"], prefixes: ["tts/piper-"] } } };
    const { members, problems } = expandReleaseSet(set, catalog);
    expect(problems).toEqual([]);
    expect(members.get("stt/a")).toMatchObject({ tier: "defaults", group: "g" });
    expect(members.get("stt/b").tier).toBe("light");
    expect([...members.keys()].sort()).toEqual(["stt/a", "stt/b", "tts/piper-1", "tts/piper-2"]);
    expect(idsForGroup(members, "defaults")).toEqual(["stt/a"]);
    expect(idsForGroup(members, "all")).toHaveLength(4);
  });

  it("reports ids that are not in the catalog and prefixes that match nothing", () => {
    const { problems } = expandReleaseSet({ defaults: { g: { ids: ["stt/nope"], prefixes: ["zzz/"] } }, light: {} }, catalog);
    expect(problems.map((p) => p.rule).sort()).toEqual(["set.missing", "set.prefix"]);
  });

  it("follows requires transitively and ignores ids the catalog lacks", () => {
    expect([...requiresClosure(["stt/b"], catalog)].sort()).toEqual(["runtime/r", "stt/b"]);
    expect([...requiresClosure(["x/unknown"], catalog)]).toEqual(["x/unknown"]);
  });
});

describe("checkCatalog", () => {
  const set = (over = {}) => ({ defaults: { g: { ids: ["stt/a"] } }, light: {}, ...over });
  const errors = (r) => r.problems.filter((p) => p.level === "error").map((p) => p.rule);

  it("passes a pinned release set and counts it", () => {
    const { catalog } = catalogOf({ "a.json": { entries: [entry("stt/a"), entry("stt/spare", { status: "unpublished", files: [{ path: "x", size: 0, sources: [ghSrc()] }] })] } });
    const r = checkCatalog(catalog, set());
    expect(errors(r)).toEqual([]);
    expect(r.summary).toMatchObject({ entries: 2, published: 1, unpublished: 1, defaults: 1, light: 0, closure: 1, errors: 0 });
  });

  it("refuses an unpublished member and an unpublished requirement unless allowUnpublished names it", () => {
    const { catalog } = catalogOf({
      "a.json": {
        entries: [entry("stt/a", { requires: ["runtime/pack"] }), entry("runtime/pack", { kind: "runtime", status: "unpublished", files: [{ path: "p.zip", size: 0, sources: [ghSrc("assets-v1", "p.zip")] }] })],
      },
    });
    expect(errors(checkCatalog(catalog, set()))).toEqual(["set.unpublished"]);
    const allowed = checkCatalog(catalog, set({ allowUnpublished: [{ id: "runtime/pack", why: "built locally" }] }));
    expect(errors(allowed)).toEqual([]);
    expect(allowed.problems.map((p) => p.rule)).toContain("req.unpublished");
  });

  it("warns about an allowUnpublished entry nothing needs and errors on one the catalog lacks", () => {
    const { catalog } = catalogOf({ "a.json": { entries: [entry("stt/a"), entry("runtime/pack", { status: "unpublished", files: [{ path: "p.zip", size: 0, sources: [ghSrc()] }] })] } });
    const r = checkCatalog(catalog, set({ allowUnpublished: [{ id: "runtime/pack", why: "x" }, { id: "runtime/ghost", why: "y" }] }));
    expect(r.problems.find((p) => p.rule === "set.allow-unused")?.level).toBe("warn");
    expect(errors(r)).toEqual(["set.allow"]);
  });

  it("applies the build.rs rule to every published entry, in or out of the release set", () => {
    const { catalog } = catalogOf({ "a.json": { entries: [entry("stt/a"), entry("tts/big", { files: [{ path: "b", size: 5, sources: [hfSrc()] }] })] } });
    expect(errors(checkCatalog(catalog, set()))).toEqual(["pin.missing"]);
  });

  it("reports a requirement the catalog does not have", () => {
    const { catalog } = catalogOf({ "a.json": { entries: [entry("stt/a", { requires: ["runtime/ghost"] })] } });
    expect(errors(checkCatalog(catalog, set()))).toContain("req.missing");
  });

  it("needs every CPU model, runtime and the language ID of the routing table in the defaults", () => {
    const routing = {
      languageId: "stt/lid",
      models: { "stt/a": { runtimes: ["runtime/r"] }, "stt/light": {}, "stt/gpu": { gpu: true, runtimes: ["runtime/gpu"] } },
    };
    expect([...routingCpuIds(routing)].sort()).toEqual(["runtime/r", "stt/a", "stt/lid", "stt/light"]);
    const { catalog } = catalogOf({
      "a.json": { entries: [entry("stt/a"), entry("stt/light"), entry("stt/lid"), entry("runtime/r", { kind: "runtime" }), entry("stt/gpu")] },
      "stt-routing.json": { entries: [], routing },
    });
    const r = checkCatalog(catalog, { defaults: { g: { ids: ["stt/a"] } }, light: { l: { ids: ["stt/light"] } } });
    const byRule = Object.fromEntries(r.problems.filter((p) => p.level === "error").map((p) => [p.id, p.rule]));
    expect(byRule).toEqual({ "stt/lid": "routing.not-in-set", "runtime/r": "routing.not-in-set", "stt/light": "routing.not-default" });
  });

  it("warns about a rolling tag no source uses", () => {
    const { catalog } = catalogOf({ "a.json": { entries: [entry("stt/a")] } });
    const r = checkCatalog(catalog, set({ rollingTags: [{ tag: "o/r@asr-models", why: "x" }] }));
    expect(r.problems.map((p) => p.rule)).toContain("set.rolling-unused");
  });

  it("reports duplicate ids across fragments and unreadable fragments", () => {
    const dir = tmp();
    fs.writeFileSync(path.join(dir, "a.json"), JSON.stringify({ entries: [entry("stt/a")] }));
    fs.writeFileSync(path.join(dir, "b.json"), JSON.stringify({ entries: [entry("stt/a")] }));
    fs.writeFileSync(path.join(dir, "c.json"), "{ nope");
    fs.writeFileSync(path.join(dir, "_schema.json"), "{}");
    const catalog = loadCatalog(dir);
    expect(catalog.problems.map((p) => p.rule).sort()).toEqual(["catalog.duplicate", "catalog.parse"]);
    expect(catalog.entries.size).toBe(1);
  });
});

describe("the committed catalog", () => {
  const catalog = loadCatalog();
  const set = loadReleaseSet();

  it("passes the static release check", () => {
    const r = checkCatalog(catalog, set);
    const errs = r.problems.filter((p) => p.level === "error").map((p) => p.message);
    expect(errs).toEqual([]);
    expect(r.summary.defaults).toBeGreaterThanOrEqual(15);
  });

  it("holds a published, hash-pinned entry for every id the release set and its requirements name", () => {
    const { members } = expandReleaseSet(set, catalog);
    for (const id of requiresClosure([...members.keys()], catalog)) {
      const e = catalog.entries.get(id);
      expect(e, id).toBeDefined();
      if (!set.allowUnpublished.some((a) => a.id === id)) expect(e.entry.status ?? "published", id).toBe("published");
    }
  });

  it("keeps the CPU speech path of the routing table inside the defaults", () => {
    const { members } = expandReleaseSet(set, catalog);
    for (const id of routingCpuIds(catalog.routing)) expect(members.get(id)?.tier, id).toBe("defaults");
  });
});

/** A fake pin-catalog client: answers from a map, counts requests. */
function fakeNet(answers) {
  const calls = [];
  return {
    calls,
    requests: 0,
    async getJson(url) {
      calls.push(url);
      this.requests += 1;
      if (!(url in answers)) throw new Error(`unexpected request ${url}`);
      const a = answers[url];
      if (a instanceof Error) throw a;
      return a;
    },
  };
}

const TREE_URL = `https://huggingface.co/api/models/org/repo/tree/${COMMIT}`;
const REL_URL = "https://api.github.com/repos/o/r/releases/tags/v1";

describe("online verification", () => {
  it("verifies an LFS file against the Hub tree and reuses the listing for its siblings", async () => {
    const net = fakeNet({
      [TREE_URL]: [
        { type: "file", path: "m.gguf", size: 10, oid: "x".repeat(40), lfs: { oid: SHA, size: 10 } },
        { type: "file", path: "tok.json", size: 7, oid: OID },
      ],
    });
    const cache = { trees: new Map(), releases: new Map() };
    expect(await verifyFileOnline({ path: "m.gguf", size: 10, sha256: SHA, sources: [hfSrc()] }, net, cache)).toMatchObject({ status: "verified", via: "huggingface" });
    expect(await verifyFileOnline({ path: "tok.json", size: 7, git_oid: OID, sources: [hfSrc(COMMIT, "tok.json")] }, net, cache)).toMatchObject({ status: "verified", detail: "git blob id" });
    expect(net.calls).toEqual([TREE_URL]);
  });

  it("reports a changed digest, a changed size and a file that left the repo as mismatches", async () => {
    const net = fakeNet({ [TREE_URL]: [{ type: "file", path: "m.gguf", size: 10, oid: "x".repeat(40), lfs: { oid: SHA_B, size: 10 } }] });
    const cache = { trees: new Map(), releases: new Map() };
    expect((await verifyFileOnline({ path: "m.gguf", size: 10, sha256: SHA, sources: [hfSrc()] }, net, cache)).detail).toMatch(/sha256/);
    expect((await verifyFileOnline({ path: "m.gguf", size: 11, sha256: SHA_B, sources: [hfSrc()] }, net, cache)).detail).toMatch(/size 11 in the catalog, 10/);
    expect(await verifyFileOnline({ path: "gone.gguf", size: 1, sha256: SHA, sources: [hfSrc(COMMIT, "gone.gguf")] }, net, cache)).toMatchObject({ status: "mismatch" });
    expect(await verifyFileOnline({ path: "m", size: 1, sha256: SHA, sources: [hfSrc("main")] }, net, cache)).toMatchObject({ status: "mismatch" });
  });

  it("verifies a release asset by its digest, and says so when the asset has none", async () => {
    const net = fakeNet({
      [REL_URL]: { assets: [{ name: "pack.zip", size: 99, digest: `sha256:${SHA}` }, { name: "old.zip", size: 5, digest: null }] },
    });
    const cache = { trees: new Map(), releases: new Map() };
    expect(await verifyFileOnline({ path: "pack.zip", size: 99, sha256: SHA, sources: [ghSrc()] }, net, cache)).toMatchObject({ status: "verified", via: "github-release" });
    expect((await verifyFileOnline({ path: "pack.zip", size: 99, sha256: SHA_B, sources: [ghSrc()] }, net, cache)).status).toBe("mismatch");
    expect(await verifyFileOnline({ path: "old.zip", size: 5, sha256: SHA, sources: [ghSrc("v1", "old.zip")] }, net, cache)).toMatchObject({ status: "unverifiable" });
    expect((await verifyFileOnline({ path: "old.zip", size: 6, sha256: SHA, sources: [ghSrc("v1", "old.zip")] }, net, cache)).status).toBe("mismatch");
    expect((await verifyFileOnline({ path: "x.zip", size: 1, sha256: SHA, sources: [ghSrc("v1", "x.zip")] }, net, cache)).detail).toMatch(/not an asset/);
    expect(net.calls).toEqual([REL_URL]);
  });

  it("re-hashes a digest-less file by downloading it when asked, within the size limit", async () => {
    const file = { path: "v.zip", size: 100, sha256: SHA, sources: [{ kind: "direct", url: "https://alphacephei.com/v.zip" }] };
    const same = { urls: [], hashByDownload: async (url) => (same.urls.push(url), { size: 100, sha256: SHA }) };
    expect(await verifyFileOnline(file, same, undefined, { downloadLimit: 1000 })).toMatchObject({ status: "verified", via: "download" });
    expect(same.urls).toEqual(["https://alphacephei.com/v.zip"]);
    const changed = { urls: [], hashByDownload: async () => ({ size: 100, sha256: SHA_B }) };
    expect((await verifyFileOnline(file, changed, undefined, { downloadLimit: 1000 })).detail).toMatch(/downloaded/);
    const shorter = { urls: [], hashByDownload: async () => ({ size: 99, sha256: SHA }) };
    expect((await verifyFileOnline(file, shorter, undefined, { downloadLimit: 1000 })).status).toBe("mismatch");
    // Too big, no limit, or a git_oid pin: nothing is downloaded.
    const never = { hashByDownload: async () => { throw new Error("must not download"); } };
    expect((await verifyFileOnline(file, never, undefined, { downloadLimit: 50 })).status).toBe("unverifiable");
    expect((await verifyFileOnline(file, never)).status).toBe("unverifiable");
    expect((await verifyFileOnline({ ...file, sha256: undefined, git_oid: OID }, never, undefined, { downloadLimit: 1000 })).status).toBe("unverifiable");
  });

  it("re-hashes a release asset that publishes no digest by downloading it", async () => {
    const net = fakeNet({ [REL_URL]: { assets: [{ name: "old.zip", size: 5, digest: null }] } });
    net.hashByDownload = async () => ({ size: 5, sha256: SHA });
    const file = { path: "old.zip", size: 5, sha256: SHA, sources: [ghSrc("v1", "old.zip")] };
    expect(await verifyFileOnline(file, net, undefined, { downloadLimit: 100 })).toMatchObject({ status: "verified", via: "download" });
    net.hashByDownload = async () => ({ size: 5, sha256: SHA_B });
    expect((await verifyFileOnline(file, net, undefined, { downloadLimit: 100 })).status).toBe("mismatch");
  });

  it("verifies NVIDIA redistributables against the redistrib manifest, nested or not", async () => {
    const base = "https://developer.download.nvidia.com/compute/cuda/redist/";
    const cudart = { kind: "nvidia-redist", url: `${base}cuda_cudart/windows-x86_64/cuda_cudart-windows-x86_64-13.2.51-archive.zip`, revision: "13.2.0" };
    const net = fakeNet({
      [`${base}redistrib_13.2.0.json`]: {
        release_date: "2026-01-01",
        cuda_cudart: { name: "CUDA Runtime", "windows-x86_64": { relative_path: "cuda_cudart/windows-x86_64/cuda_cudart-windows-x86_64-13.2.51-archive.zip", sha256: SHA, md5: "x", size: "3092970" } },
        cudnn: { "windows-x86_64": { cuda13: { relative_path: "cudnn/windows-x86_64/cudnn-13.zip", sha256: SHA_B, size: "7" } } },
      },
    });
    const cache = { trees: new Map(), releases: new Map() };
    expect(await verifyFileOnline({ path: "a.zip", size: 3092970, sha256: SHA, sources: [cudart] }, net, cache)).toMatchObject({ status: "verified", via: "nvidia-redist" });
    expect((await verifyFileOnline({ path: "a.zip", size: 3092970, sha256: SHA_B, sources: [cudart] }, net, cache)).status).toBe("mismatch");
    const nested = { kind: "nvidia-redist", url: `${base}cudnn/windows-x86_64/cudnn-13.zip`, revision: "13.2.0" };
    expect((await verifyFileOnline({ path: "c.zip", size: 7, sha256: SHA_B, sources: [nested] }, net, cache)).status).toBe("verified");
    const gone = { kind: "nvidia-redist", url: `${base}cuda_x/windows-x86_64/x.zip`, revision: "13.2.0" };
    expect((await verifyFileOnline({ path: "x.zip", size: 1, sha256: SHA, sources: [gone] }, net, cache)).detail).toMatch(/not in redistrib_13.2.0.json/);
    expect(net.calls).toEqual([`${base}redistrib_13.2.0.json`]);
    expect(findRedistNode({ a: { b: { relative_path: "p" } } }, "p")).toEqual({ relative_path: "p" });
    expect(findRedistNode({ a: 1 }, "p")).toBeNull();
  });

  it("verifies a raw.githubusercontent.com file pinned by git blob id through the contents API", async () => {
    const raw = { kind: "direct", url: `https://raw.githubusercontent.com/o/r/${COMMIT}/bin/win64/lib%20x.dll`, revision: COMMIT };
    expect(parseRawGithubUrl(raw.url)).toEqual({ owner: "o", repo: "r", commit: COMMIT, path: "bin/win64/lib x.dll" });
    expect(parseRawGithubUrl(`https://raw.githubusercontent.com/o/r/main/x`)).toBeNull();
    const api = `https://api.github.com/repos/o/r/contents/bin/win64/lib%20x.dll?ref=${COMMIT}`;
    const ok = fakeNet({ [api]: { sha: OID, size: 837272 } });
    expect(await verifyFileOnline({ path: "x.dll", size: 837272, git_oid: OID, sources: [raw] }, ok)).toMatchObject({ status: "verified", via: "github-raw" });
    expect((await verifyFileOnline({ path: "x.dll", size: 837272, git_oid: "f".repeat(40), sources: [raw] }, ok)).status).toBe("mismatch");
    // A sha256 pin cannot be read from the blob id: nothing is requested.
    const none = fakeNet({});
    expect((await verifyFileOnline({ path: "x.dll", size: 1, sha256: SHA, sources: [raw] }, none)).status).toBe("unverifiable");
    expect(none.calls).toEqual([]);
  });

  it("calls a host without published digests unverifiable", async () => {
    const r = await verifyFileOnline({ path: "v.zip", size: 1, sha256: SHA, sources: [{ kind: "direct", url: "https://alphacephei.com/v.zip" }] }, fakeNet({}));
    expect(r).toMatchObject({ status: "unverifiable", via: "none" });
  });

  it("walks entries, skips unpublished ones and turns a failed lookup into unverifiable", async () => {
    const { catalog } = catalogOf({
      "a.json": { entries: [entry("mt/ok"), entry("mt/down", { files: [{ path: "m.gguf", size: 10, sha256: SHA, sources: [{ kind: "huggingface", url: `https://huggingface.co/other/repo/resolve/${COMMIT}/m.gguf`, revision: COMMIT }] }] }), entry("rt/local", { status: "unpublished" })] },
    });
    const net = fakeNet({
      [TREE_URL]: [{ type: "file", path: "m.gguf", size: 10, oid: "x".repeat(40), lfs: { oid: SHA, size: 10 } }],
      [`https://huggingface.co/api/models/other/repo/tree/${COMMIT}`]: new Error("HTTP 503"),
    });
    const seen = [];
    const rows = await verifyOnline(catalog, ["mt/ok", "mt/down", "rt/local", "mt/ghost"], net, { onFile: (r) => seen.push(r.id) });
    expect(rows.map((r) => [r.id, r.status])).toEqual([["mt/ok", "verified"], ["mt/down", "unverifiable"]]);
    expect(rows[1].detail).toMatch(/lookup failed: HTTP 503/);
    expect(seen).toEqual(["mt/ok", "mt/down"]);
  });
});

describe("install records", () => {
  function store(records) {
    const root = tmp();
    for (const [rel, rec] of Object.entries(records)) {
      const dir = path.join(root, "models", ...rel.split("/"));
      fs.mkdirSync(dir, { recursive: true });
      fs.writeFileSync(path.join(dir, "installed.json"), typeof rec === "string" ? rec : JSON.stringify(rec));
    }
    return root;
  }
  const { catalog } = catalogOf({
    "a.json": {
      entries: [
        entry("mt/ok"),
        entry("mt/bad"),
        entry("mt/old", { version: "2" }),
        entry("mt/oid", { files: [{ path: "c.json", size: 7, git_oid: OID, sources: [hfSrc(COMMIT, "c.json")] }] }),
        entry("mt/none"),
      ],
    },
  });

  it("reads records at any depth and ignores damaged ones", () => {
    const root = store({
      "mt/ok": { id: "mt/ok", version: "1", files: [] },
      "mt-llm/mt/deep": { id: "mt/deep", version: "1", files: [] },
      "mt/broken": "{ nope",
    });
    expect([...readInstallRecords(root).keys()].sort()).toEqual(["mt/deep", "mt/ok"]);
    expect(readInstallRecords(path.join(root, "missing")).size).toBe(0);
  });

  it("matches a store that verified the pinned bytes and flags every kind of difference", () => {
    const root = store({
      "mt/ok": { id: "mt/ok", version: "1", origin: "download", files: [{ path: "m.gguf", size: 10, sha256: SHA }] },
      "mt/bad": { id: "mt/bad", version: "1", files: [{ path: "m.gguf", size: 10, sha256: SHA_B }] },
      "mt/old": { id: "mt/old", version: "1", files: [{ path: "m.gguf", size: 10, sha256: SHA }] },
      "mt/oid": { id: "mt/oid", version: "1", files: [{ path: "c.json", size: 7, sha256: SHA }] },
    });
    const rows = compareInstalled(catalog, readInstallRecords(root), ["mt/ok", "mt/bad", "mt/old", "mt/oid", "mt/none", "mt/ghost"]);
    expect(Object.fromEntries(rows.map((r) => [r.id, r.status]))).toEqual({
      "mt/ok": "match",
      "mt/bad": "mismatch",
      "mt/old": "other-version",
      "mt/oid": "match",
      "mt/none": "absent",
      "mt/ghost": "absent",
    });
    expect(rows[0].origin).toBe("download");
  });
});

describe("command line", () => {
  it("enforces the request spacing the dev network needs and validates the group", () => {
    expect(parseArgs([]).delayMs).toBe(2500);
    expect(() => parseArgs(["--delay-ms", "500"])).toThrow(/at least 2000/);
    expect(() => parseArgs(["--group", "some"])).toThrow(/defaults, light or all/);
    expect(parseArgs(["--online", "--group", "all", "--installed", "d"])).toMatchObject({ online: true, group: "all", installed: "d" });
    expect(() => parseArgs(["--bogus"])).toThrow(/unknown argument/);
    expect(parseArgs(["--hash-by-download"])).toMatchObject({ hashByDownload: true, maxDownloadMb: 50 });
    expect(parseArgs(["--hash-by-download", "--max-download-mb", "5"]).maxDownloadMb).toBe(5);
    expect(() => parseArgs(["--max-download-mb", "0"])).toThrow(/positive/);
    expect(() => parseArgs(["--report"])).toThrow(/needs a value/);
  });

  function run(argv, deps) {
    const lines = [];
    const io = { out: (l) => lines.push(l), err: (l) => lines.push(`ERR ${l}`) };
    return main(argv, io, deps).then((code) => ({ code, text: lines.join("\n") }));
  }

  const fixture = () => {
    const { dir } = catalogOf({ "a.json": { entries: [entry("stt/a")] } });
    const setFile = path.join(tmp(), "set.json");
    fs.writeFileSync(setFile, JSON.stringify({ schema: "sigil-release-set/1", defaults: { g: { ids: ["stt/a"] } }, light: {} }));
    return { dir, setFile };
  };

  it("exits 0 on a clean catalog and writes the report", async () => {
    const { dir, setFile } = fixture();
    const report = path.join(tmp(), "r.json");
    const { code, text } = await run(["--catalog-dir", dir, "--release-set", setFile, "--report", report]);
    expect(code).toBe(0);
    expect(text).toMatch(/1 entries \(1 published, 0 unpublished\)/);
    expect(text.trim().endsWith("OK")).toBe(true);
    expect(JSON.parse(fs.readFileSync(report, "utf8"))).toMatchObject({ schema: "sigil-pin-check/1", summary: { errors: 0 } });
  });

  it("exits 1 with the rule and the entry when a pin is missing", async () => {
    const { dir, setFile } = fixture();
    fs.writeFileSync(path.join(dir, "b.json"), JSON.stringify({ entries: [entry("tts/x", { files: [{ path: "x", size: 3, sources: [hfSrc()] }] })] }));
    const { code, text } = await run(["--catalog-dir", dir, "--release-set", setFile]);
    expect(code).toBe(1);
    expect(text).toMatch(/ERROR pin\.missing: tts\/x x/);
    expect(text.trim().endsWith("FAILED")).toBe(true);
  });

  it("exits 2 on a usage error", async () => {
    expect((await run(["--group", "nope"])).code).toBe(2);
  });

  it("fails on an online mismatch and passes on a verified release set", async () => {
    const { dir, setFile } = fixture();
    const good = fakeNet({ [TREE_URL]: [{ type: "file", path: "m.gguf", size: 10, oid: "x".repeat(40), lfs: { oid: SHA, size: 10 } }] });
    const ok = await run(["--catalog-dir", dir, "--release-set", setFile, "--online"], { net: good });
    expect(ok.code).toBe(0);
    expect(ok.text).toMatch(/online: 1 files: \{"verified":1\}/);
    const bad = fakeNet({ [TREE_URL]: [{ type: "file", path: "m.gguf", size: 10, oid: "x".repeat(40), lfs: { oid: SHA_B, size: 10 } }] });
    const failed = await run(["--catalog-dir", dir, "--release-set", setFile, "--online"], { net: bad });
    expect(failed.code).toBe(1);
    expect(failed.text).toMatch(/ERROR online\.mismatch: stt\/a m\.gguf/);
  });

  it("fails on an install record that disagrees with the catalog", async () => {
    const { dir, setFile } = fixture();
    const root = tmp();
    fs.mkdirSync(path.join(root, "models", "stt", "a"), { recursive: true });
    fs.writeFileSync(path.join(root, "models", "stt", "a", "installed.json"), JSON.stringify({ id: "stt/a", version: "1", files: [{ path: "m.gguf", size: 10, sha256: SHA_B }] }));
    const { code, text } = await run(["--catalog-dir", dir, "--release-set", setFile, "--installed", root]);
    expect(code).toBe(1);
    expect(text).toMatch(/ERROR installed\.mismatch: stt\/a/);
  });
});
