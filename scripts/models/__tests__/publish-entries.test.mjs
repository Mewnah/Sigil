import fs from "node:fs";
import os from "node:os";
import path from "node:path";
import { afterEach, describe, expect, it } from "vitest";
import { Throttle } from "../pin-catalog.mjs";
import { RELEASE_BASE_URL } from "../assets-v1.mjs";
import { main, normalizeBase, parseArgs, preflight, run, settleRequires, urlUnder } from "../publish-entries.mjs";
import { loadCatalog } from "../../release/check-catalog-pins.mjs";
import { BASE, BYTES, COMMIT, makeWorld, sha256 } from "./fixtures.mjs";

const worlds = [];
afterEach(() => {
  while (worlds.length) worlds.pop().cleanup();
});
const world = (opts) => {
  const w = makeWorld(opts);
  worlds.push(w);
  return w;
};

/** A fetch that serves `served` (url -> Buffer | number status) and records the virtual time of each request. */
function fakeServer(served, clock) {
  const log = [];
  const fetchImpl = async (url) => {
    log.push({ url, at: clock.now });
    const hit = served.get(url);
    if (hit === undefined) return new Response("not found", { status: 404 });
    if (typeof hit === "number") return new Response("err", { status: hit });
    return new Response(hit, { status: 200, headers: { "content-type": "application/octet-stream" } });
  };
  return { fetchImpl, log };
}

function virtualClock() {
  const clock = { now: 0 };
  const throttle = new Throttle(2500, () => clock.now, async (ms) => {
    clock.now += ms;
  });
  return { clock, throttle };
}

const tmpDirs = [];
afterEach(() => {
  while (tmpDirs.length) fs.rmSync(tmpDirs.pop(), { recursive: true, force: true });
});

function servedFromBytes(names) {
  return new Map(names.map((n) => [BASE + n, BYTES[n]]));
}

/** A GitHub-API-free metadata client for the upstream (Hugging Face) check. */
function fakeNet(files) {
  return {
    requests: 0,
    async getJson(url) {
      this.requests += 1;
      if (url === `https://huggingface.co/api/models/org/repo/tree/${COMMIT}`) {
        return [{ type: "file", path: "model.gguf", size: BYTES["up-model.gguf"].length, oid: "a".repeat(40), lfs: { oid: files.hf ?? sha256(BYTES["up-model.gguf"]), size: BYTES["up-model.gguf"].length } }];
      }
      if (url.startsWith("https://api.github.com/repos/Mewnah/Sigil-assets/releases/tags/assets-v1")) return { assets: files.assets ?? [] };
      throw new Error(`unexpected metadata request ${url}`);
    },
  };
}

async function runWith(w, argv, { served, net = fakeNet({}) } = {}) {
  const { clock, throttle } = virtualClock();
  const server = fakeServer(served, clock);
  const out = [];
  const err = [];
  const temp = fs.mkdtempSync(path.join(os.tmpdir(), "sigil-pub-test-"));
  tmpDirs.push(temp);
  const opts = parseArgs(["--catalog-dir", w.catalogDir, ...argv]);
  const code = await run(opts, { out: (m) => out.push(m), err: (m) => err.push(m) }, { fetch: server.fetchImpl, throttle, net, tempDir: temp, skipPostCheck: true });
  return { code, out: out.join("\n"), err: err.join("\n"), log: server.log, temp, clock };
}

const statusOf = (w, file, id) => JSON.parse(fs.readFileSync(path.join(w.catalogDir, file), "utf8")).entries.find((e) => e.id === id).status;

describe("arguments", () => {
  it("parses the options and rejects unsafe ones", () => {
    const o = parseArgs(["--ids", "a/b,c/d", "--staging", "s", "--write", "--except", "c/d", "--delay-ms", "3000"]);
    expect(o).toMatchObject({ ids: "a/b,c/d", staging: "s", write: true, except: ["c/d"], delayMs: 3000, digestOnly: false, assetsBase: RELEASE_BASE_URL });
    expect(() => parseArgs([])).toThrow(/--ids/);
    expect(() => parseArgs(["--ids", "x", "--delay-ms", "1000"])).toThrow(/at least 2000/);
    expect(() => parseArgs(["--ids", "x", "--nope"])).toThrow(/unknown argument/);
    expect(() => parseArgs(["--ids", "x", "--digest-only", "--assets-base", "http://127.0.0.1:1/"])).toThrow(/cannot be combined/);
    expect(parseArgs(["--help"]).help).toBe(true);
  });
  it("--assets-base is https, or http on this PC only", () => {
    expect(normalizeBase("http://127.0.0.1:8123")).toBe("http://127.0.0.1:8123/");
    expect(normalizeBase("https://example.org/a/b")).toBe("https://example.org/a/b/");
    expect(() => normalizeBase("http://example.org/")).toThrow(/https/);
    expect(() => normalizeBase("https://u:p@example.org/")).toThrow(/credentials/);
    expect(() => normalizeBase("nope")).toThrow(/not a URL/);
  });
  it("urlUnder percent-encodes the asset name", () => {
    expect(urlUnder("http://127.0.0.1:1/", "a b+c.bin")).toBe("http://127.0.0.1:1/a%20b%2Bc.bin");
  });
});

describe("preflight and settleRequires", () => {
  it("flags unpinned files, nested URLs and unpublished requirements outside the run", () => {
    const w = world();
    const catalog = loadCatalog(w.catalogDir);
    const get = (id) => catalog.entries.get(id).entry;
    expect(preflight(get("t/pack"), catalog, new Set(["t/pack"])).errors.join("\n")).toMatch(/pack-1\.zip: not pinned/);
    expect(preflight(get("t/nested"), catalog, new Set()).errors.join("\n")).toMatch(/not a flat asset name/);
    expect(preflight(get("t/dep"), catalog, new Set(["t/dep"])).errors.join("\n")).toMatch(/requires t\/solo, which is unpublished and not part of this run/);
    expect(preflight(get("t/dep"), catalog, new Set(["t/dep", "t/solo"])).errors).toEqual([]);
    expect(preflight(get("t/pub"), catalog, new Set()).note).toBe("already published");
  });
  it("an entry stays unpublished when what it requires failed", () => {
    const w = world();
    const catalog = loadCatalog(w.catalogDir);
    const ok = new Map([["t/solo", false], ["t/dep", true]]);
    expect(settleRequires(ok, catalog).get("t/dep")).toBe(false);
    expect(settleRequires(new Map([["t/solo", true], ["t/dep", true]]), catalog).get("t/dep")).toBe(true);
  });
});

describe("run", () => {
  it("downloads, hashes and deletes the temp file; a dry run changes nothing", async () => {
    const w = world();
    const served = servedFromBytes(["t-solo-model.onnx", "t-multi-a.bin", "t-multi-b.json", "t-dep-model.onnx"]);
    const r = await runWith(w, ["--ids", "t/solo,t/multi,t/dep"], { served });
    expect(r.code).toBe(0);
    expect(r.out).toMatch(/verified t\/solo/);
    expect(r.out).toMatch(/dry run: 3 entries would become published: t\/solo, t\/multi, t\/dep/);
    expect(statusOf(w, "a.json", "t/solo")).toBe("unpublished");
    expect(fs.readdirSync(r.temp)).toEqual([]);
    expect(r.log.map((l) => l.url)).toEqual([BASE + "t-solo-model.onnx", BASE + "t-multi-a.bin", BASE + "t-multi-b.json", BASE + "t-dep-model.onnx"]);
  });

  it("requests are at least 2 s apart, one at a time", async () => {
    const w = world();
    const served = servedFromBytes(["t-solo-model.onnx", "t-multi-a.bin", "t-multi-b.json", "t-dep-model.onnx"]);
    const r = await runWith(w, ["--ids", "t/solo,t/multi,t/dep"], { served });
    const times = r.log.map((l) => l.at);
    for (let i = 1; i < times.length; i += 1) expect(times[i] - times[i - 1]).toBeGreaterThanOrEqual(2000);
  });

  it("--write flips only the status lines, in both fragment layouts", async () => {
    const w = world();
    const beforeA = fs.readFileSync(path.join(w.catalogDir, "a.json"), "utf8");
    const served = servedFromBytes(["t-solo-model.onnx", "t-multi-a.bin", "t-multi-b.json"]);
    const r = await runWith(w, ["--ids", "t/solo,t/multi", "--write"], { served });
    expect(r.code).toBe(0);
    expect(statusOf(w, "a.json", "t/solo")).toBe("published");
    expect(statusOf(w, "a.json", "t/multi")).toBe("published");
    expect(statusOf(w, "a.json", "t/dep")).toBe("unpublished");
    const afterA = fs.readFileSync(path.join(w.catalogDir, "a.json"), "utf8");
    const was = beforeA.split("\n");
    expect(afterA.split("\n").filter((l, i) => l !== was[i])).toHaveLength(2);
  });

  it("wrong bytes, a 404 and a server error keep those entries unpublished and exit 1", async () => {
    const w = world();
    const served = servedFromBytes(["t-solo-model.onnx", "t-multi-a.bin"]);
    served.set(BASE + "t-multi-a.bin", Buffer.from("multi X")); // same length, other bytes
    // t-multi-b.json is not uploaded (404); t-dep-model.onnx answers 503 (retried, then gives up).
    served.set(BASE + "t-dep-model.onnx", 503);
    const r = await runWith(w, ["--ids", "t/solo,t/multi,t/dep", "--write"], { served });
    expect(r.code).toBe(1);
    expect(r.out).toMatch(/FAILED .*a\.bin.*sha256/);
    expect(r.out).toMatch(/b\.json.*HTTP 404/);
    expect(r.out).toMatch(/HTTP 503/);
    // t/solo verified, so it flips; the two that failed stay unpublished.
    expect(statusOf(w, "a.json", "t/solo")).toBe("published");
    expect(statusOf(w, "a.json", "t/multi")).toBe("unpublished");
    expect(statusOf(w, "a.json", "t/dep")).toBe("unpublished");
    expect(fs.readdirSync(r.temp)).toEqual([]);
  });

  it("flips the entries that verified even when another failed, and holds one whose requirement failed", async () => {
    const w = world();
    const served = servedFromBytes(["t-multi-a.bin", "t-multi-b.json", "t-dep-model.onnx"]);
    // t/solo (required by t/dep) is not uploaded.
    const r = await runWith(w, ["--ids", "t/solo,t/multi,t/dep", "--write"], { served });
    expect(r.code).toBe(1);
    expect(statusOf(w, "a.json", "t/multi")).toBe("published");
    expect(statusOf(w, "a.json", "t/solo")).toBe("unpublished");
    expect(statusOf(w, "a.json", "t/dep")).toBe("unpublished");
    expect(r.out).toMatch(/HELD\s+t\/dep/);
  });

  it("a size larger than the pin stops the download early", async () => {
    const w = world();
    const served = servedFromBytes(["t-multi-b.json", "t-multi-a.bin"]);
    served.set(BASE + "t-multi-a.bin", Buffer.alloc(5000, 1));
    const r = await runWith(w, ["--ids", "t/multi"], { served });
    expect(r.code).toBe(1);
    expect(r.out).toMatch(/more than the pinned 7 bytes/);
    expect(fs.readdirSync(r.temp)).toEqual([]);
  });

  it("an HTML page instead of the file fails", async () => {
    const w = world();
    const { clock, throttle } = virtualClock();
    const fetchImpl = async () => new Response("<html></html>", { status: 200, headers: { "content-type": "text/html; charset=utf-8" } });
    const out = [];
    const temp = fs.mkdtempSync(path.join(os.tmpdir(), "sigil-pub-test-"));
    tmpDirs.push(temp);
    const code = await run(parseArgs(["--catalog-dir", w.catalogDir, "--ids", "t/solo"]), { out: (m) => out.push(m), err: () => {} }, { fetch: fetchImpl, throttle, net: fakeNet({}), tempDir: temp, skipPostCheck: true });
    void clock;
    expect(code).toBe(1);
    expect(out.join("\n")).toMatch(/HTML page/);
  });

  it("refuses unpinned files and ids that are not in the catalog; skips published entries", async () => {
    const w = world();
    const r1 = await runWith(w, ["--ids", "t/pack"], { served: new Map() });
    expect(r1.code).toBe(1);
    expect(r1.out).toMatch(/pack-1\.zip: not pinned/);
    expect(r1.log).toEqual([]);
    const r2 = await runWith(w, ["--ids", "nope/x"], { served: new Map() });
    expect(r2.code).toBe(2);
    expect(r2.err).toMatch(/not a catalog id/);
    const r3 = await runWith(w, ["--ids", "t/pub"], { served: new Map() });
    expect(r3.code).toBe(0);
    expect(r3.out).toMatch(/skip\s+t\/pub: already published/);
  });

  it("all-staged needs --staging and selects only fully staged entries", async () => {
    const w = world({ stage: ["t-solo-model.onnx", "t-multi-a.bin", "t-multi-b.json"] });
    const none = await runWith(w, ["--ids", "all-staged"], { served: new Map() });
    expect(none.code).toBe(2);
    expect(none.err).toMatch(/needs --staging/);
    const served = servedFromBytes(["t-solo-model.onnx", "t-multi-a.bin", "t-multi-b.json"]);
    const r = await runWith(w, ["--ids", "all-staged", "--staging", w.staging], { served });
    expect(r.code).toBe(0);
    expect(r.out).toMatch(/would become published: t\/multi, t\/solo|would become published: t\/solo, t\/multi/);
    expect(r.out).not.toMatch(/t\/dep|t\/pack|t\/nested/);
    const ex = await runWith(w, ["--ids", "all-staged", "--staging", w.staging, "--except", "t/multi"], { served });
    expect(ex.out).toMatch(/1 entry/);
  });

  it("an upstream-backed file is checked against its host's metadata; its release mirror only when staged", async () => {
    const w = world({ stage: [] });
    const r = await runWith(w, ["--ids", "t/up"], { served: new Map() });
    expect(r.code).toBe(0);
    expect(r.out).toMatch(/verified t\/up/);
    expect(r.log).toEqual([]);
    // A wrong LFS digest upstream fails the entry.
    const bad = await runWith(w, ["--ids", "t/up"], { served: new Map(), net: fakeNet({ hf: "9".repeat(64) }) });
    expect(bad.code).toBe(1);
    expect(bad.out).toMatch(/upstream \(huggingface\): sha256/);
    // With the mirror staged it is downloaded and checked too.
    const w2 = world({ stage: ["up-model.gguf"] });
    const ok = await runWith(w2, ["--ids", "t/up", "--staging", w2.staging], { served: servedFromBytes(["up-model.gguf"]) });
    expect(ok.code).toBe(0);
    expect(ok.log.map((l) => l.url)).toEqual([BASE + "up-model.gguf"]);
    const missing = await runWith(w2, ["--ids", "t/up", "--staging", w2.staging], { served: new Map() });
    expect(missing.code).toBe(1);
    expect(missing.out).toMatch(/HTTP 404/);
  });

  it("--digest-only compares GitHub's digests and never downloads", async () => {
    const w = world();
    const assets = [
      { name: "t-solo-model.onnx", size: BYTES["t-solo-model.onnx"].length, digest: `sha256:${sha256(BYTES["t-solo-model.onnx"])}` },
      { name: "t-multi-a.bin", size: BYTES["t-multi-a.bin"].length, digest: `sha256:${"0".repeat(64)}` },
      { name: "t-multi-b.json", size: BYTES["t-multi-b.json"].length },
    ];
    const r = await runWith(w, ["--ids", "t/solo,t/multi", "--digest-only"], { served: new Map(), net: fakeNet({ assets }) });
    expect(r.code).toBe(1);
    expect(r.log).toEqual([]);
    expect(r.out).toMatch(/verified t\/solo/);
    expect(r.out).toMatch(/t\/multi a\.bin: release API \(github-release\): sha256/);
    expect(r.out).toMatch(/b\.json: .*publishes no digest.*without --digest-only/);
  });
});

describe("main", () => {
  it("usage errors exit 2 and --help exits 0", async () => {
    const out = [];
    const err = [];
    const io = { out: (m) => out.push(m), err: (m) => err.push(m) };
    expect(await main([], io)).toBe(2);
    expect(err.join("\n")).toMatch(/--ids/);
    expect(await main(["--help"], io)).toBe(0);
    expect(out.join("\n")).toMatch(/Flip catalog entries/);
  });
});
