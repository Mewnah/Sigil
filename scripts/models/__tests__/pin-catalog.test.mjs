import { describe, expect, it } from "vitest";
import {
  applyFact,
  buildHfUrl,
  hfTreeFacts,
  MIN_DELAY_MS,
  nextLink,
  parseArgs,
  parseGithubReleaseUrl,
  parseHfUrl,
  pinFragments,
  pinHfRevision,
  serializeFragment,
  Throttle,
} from "../pin-catalog.mjs";

const SHA = "5359861c739e955e79d9a303bcbc70fb988958b1";
const LFS = "60ed5bc3dd14eea856493d334349b405782ddcaf0028d4b5df4088345fba2efe";

describe("URL parsing", () => {
  it("parses and rebuilds Hub file URLs (models, datasets, hf-mirror)", () => {
    const p = parseHfUrl(`https://huggingface.co/ggerganov/whisper.cpp/resolve/main/ggml-base.bin`);
    expect(p).toEqual({ host: "huggingface.co", repoType: "models", repo: "ggerganov/whisper.cpp", revision: "main", path: "ggml-base.bin" });
    expect(buildHfUrl({ ...p, revision: SHA })).toBe(`https://huggingface.co/ggerganov/whisper.cpp/resolve/${SHA}/ggml-base.bin`);
    const d = parseHfUrl("https://hf-mirror.com/datasets/org/set/resolve/v1/dir/a%20b.onnx");
    expect(d).toMatchObject({ host: "hf-mirror.com", repoType: "datasets", repo: "org/set", revision: "v1", path: "dir/a b.onnx" });
    expect(buildHfUrl(d)).toBe("https://hf-mirror.com/datasets/org/set/resolve/v1/dir/a%20b.onnx");
    expect(parseHfUrl("https://huggingface.co/org/repo/blob/main/x.bin")).toBeNull();
    expect(parseHfUrl("http://huggingface.co/org/repo/resolve/main/x.bin")).toBeNull();
    expect(parseHfUrl("https://example.com/org/repo/resolve/main/x.bin")).toBeNull();
  });

  it("parses GitHub release asset URLs", () => {
    expect(parseGithubReleaseUrl("https://github.com/Mewnah/Sigil/releases/download/assets-v1/pack.zip")).toEqual({
      owner: "Mewnah",
      repo: "Sigil",
      tag: "assets-v1",
      asset: "pack.zip",
    });
    expect(parseGithubReleaseUrl("https://github.com/Mewnah/Sigil/archive/main.zip")).toBeNull();
  });

  it("follows Link rel=next", () => {
    expect(nextLink('<https://huggingface.co/api/x?cursor=abc>; rel="next"')).toBe("https://huggingface.co/api/x?cursor=abc");
    expect(nextLink('<https://a>; rel="prev"')).toBeNull();
    expect(nextLink(null)).toBeNull();
  });
});

describe("applying facts", () => {
  it("maps tree items to sha256 (LFS) or git oid (plain files)", () => {
    const facts = hfTreeFacts([
      { type: "file", path: "README.md", size: 3196, oid: "bb0749e3c17cdf1d040a7e95fbf45610a212ec0a" },
      { type: "file", path: "ggml-base.bin", size: 147951465, oid: "a".repeat(40), lfs: { oid: LFS, size: 147951465 } },
      { type: "directory", path: "models", oid: "b".repeat(40) },
    ]);
    expect(facts.get("README.md")).toEqual({ size: 3196, git_oid: "bb0749e3c17cdf1d040a7e95fbf45610a212ec0a" });
    expect(facts.get("ggml-base.bin")).toEqual({ size: 147951465, sha256: LFS });
    expect(facts.has("models")).toBe(false);
  });

  it("rewrites size and hash in a stable key order and drops the other hash", () => {
    const file = { path: "m.bin", size: 0, git_oid: "c".repeat(40), sources: [], archive: { format: "zip" } };
    const changes = applyFact(file, { size: 10, sha256: LFS });
    expect(Object.keys(file)).toEqual(["path", "size", "sha256", "sources", "archive"]);
    expect(file.sha256).toBe(LFS);
    expect(file.git_oid).toBeUndefined();
    expect(changes.join(" ")).toContain("git_oid removed");
    expect(applyFact(file, { size: 10, sha256: LFS })).toEqual([]);
  });

  it("pins Hub sources from a branch to the commit", () => {
    const file = {
      path: "m.bin",
      sources: [
        { kind: "huggingface", url: "https://huggingface.co/o/r/resolve/main/m.bin" },
        { kind: "hf-mirror", url: "https://hf-mirror.com/o/r/resolve/main/m.bin" },
        { kind: "direct", url: "https://example.com/m.bin" },
      ],
    };
    expect(pinHfRevision(file, "models", "o/r", "main", SHA)).toBe(2);
    expect(file.sources[0]).toEqual({ kind: "huggingface", url: `https://huggingface.co/o/r/resolve/${SHA}/m.bin`, revision: SHA });
    expect(file.sources[1].url).toBe(`https://hf-mirror.com/o/r/resolve/${SHA}/m.bin`);
    expect(file.sources[2]).toEqual({ kind: "direct", url: "https://example.com/m.bin" });
    expect(pinHfRevision(file, "models", "o/r", SHA, SHA)).toBe(0);
  });
});

describe("pinFragments", () => {
  it("resolves a branch once, lists each directory once, and reports what stays unpinned", async () => {
    const doc = {
      entries: [
        {
          id: "whisper/base",
          files: [{ path: "ggml-base.bin", size: 0, sources: [{ kind: "huggingface", url: "https://huggingface.co/ggerganov/whisper.cpp/resolve/main/ggml-base.bin" }] }],
        },
        {
          id: "base/card",
          files: [{ path: "README.md", size: 0, sources: [{ kind: "huggingface", url: "https://huggingface.co/ggerganov/whisper.cpp/resolve/main/README.md" }] }],
        },
        {
          id: "vosk/x",
          files: [{ path: "x.zip", size: 0, sources: [{ kind: "direct", url: "https://alphacephei.com/vosk/models/x.zip" }] }],
        },
      ],
    };
    const calls = [];
    const net = {
      async getJson(url, opts) {
        calls.push([url, Boolean(opts?.immutable)]);
        if (url.endsWith("/revision/main")) return { sha: SHA };
        if (url.includes(`/tree/${SHA}`)) {
          return [
            { type: "file", path: "ggml-base.bin", size: 147951465, oid: "a".repeat(40), lfs: { oid: LFS, size: 147951465 } },
            { type: "file", path: "README.md", size: 3196, oid: "bb0749e3c17cdf1d040a7e95fbf45610a212ec0a" },
          ];
        }
        throw new Error(`unexpected ${url}`);
      },
      async hashByDownload() {
        throw new Error("must not download without --hash-by-download");
      },
    };
    const { changes, unpinned } = await pinFragments([{ name: "f.json", doc }], net, { log: () => {} });
    expect(calls).toEqual([
      ["https://huggingface.co/api/models/ggerganov/whisper.cpp/revision/main", false],
      [`https://huggingface.co/api/models/ggerganov/whisper.cpp/tree/${SHA}`, true],
    ]);
    const [base, card] = doc.entries.map((e) => e.files[0]);
    expect(base).toMatchObject({ size: 147951465, sha256: LFS });
    expect(base.sources[0].url).toContain(`/resolve/${SHA}/`);
    expect(card).toMatchObject({ size: 3196, git_oid: "bb0749e3c17cdf1d040a7e95fbf45610a212ec0a" });
    expect(unpinned).toEqual(["f.json vosk/x x.zip"]);
    expect(changes.length).toBeGreaterThan(0);
    expect(serializeFragment(doc).endsWith("}\n")).toBe(true);

    // A second run over the pinned document makes no request.
    calls.length = 0;
    const again = await pinFragments([{ name: "f.json", doc }], net, { log: () => {}, only: ["whisper/", "base/"] });
    expect(calls).toEqual([]);
    expect(again.unpinned).toEqual([]);
  });

  it("hashes by download only when asked", async () => {
    const doc = { entries: [{ id: "vosk/x", files: [{ path: "x.zip", size: 0, sources: [{ kind: "direct", url: "https://a.example/x.zip" }] }] }] };
    const net = { getJson: async () => { throw new Error("no metadata source"); }, hashByDownload: async () => ({ size: 5, sha256: LFS }) };
    const { unpinned } = await pinFragments([{ name: "f", doc }], net, { log: () => {}, hashByDownload: true });
    expect(unpinned).toEqual([]);
    expect(doc.entries[0].files[0]).toMatchObject({ size: 5, sha256: LFS });
  });
});

describe("gentleness", () => {
  it("spaces request starts by at least the delay", async () => {
    let now = 0;
    const slept = [];
    const t = new Throttle(2500, () => now, async (ms) => {
      slept.push(ms);
      now += ms;
    });
    await t.wait();
    now += 100;
    await t.wait();
    await t.wait();
    expect(slept).toEqual([2400, 2500]);
  });

  it("refuses a delay under two seconds", () => {
    expect(() => parseArgs(["--delay-ms", "500"])).toThrow(/at least/);
    expect(parseArgs(["--write", "--only", "whisper/", "--no-cache"])).toMatchObject({ write: true, only: ["whisper/"], cacheDir: null });
    expect(MIN_DELAY_MS).toBe(2000);
    expect(() => parseArgs(["--bogus"])).toThrow(/unknown argument/);
  });
});
