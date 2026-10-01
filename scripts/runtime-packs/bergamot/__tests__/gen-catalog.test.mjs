import fs from "node:fs";
import path from "node:path";
import { describe, expect, it } from "vitest";
import {
  ATTACHMENTS_BASE,
  buildFragment,
  canonicalCode,
  compareVersions,
  DEFAULT_OUT,
  directionId,
  filterApplies,
  fragmentText,
  RUNTIME_ID,
  selectDirections,
} from "../gen-catalog.mjs";

const hash = (n) => n.toString(16).padStart(64, "0");
let seq = 1;

/** One registry record like Remote Settings returns it. */
function record(from, to, version, fileType, { filter = "", name, size = 1000 } = {}) {
  const pair = `${from.replace("-Hans", "").replace("-Hant", "")}${to.replace("-Hans", "").replace("-Hant", "")}`.replace(/-/g, "");
  const names = {
    model: `model.${pair}.intgemm.alphas.bin`,
    lex: `lex.50.50.${pair}.s2t.bin`,
    vocab: `vocab.${pair}.spm`,
    srcvocab: `srcvocab.${pair}.spm`,
    trgvocab: `trgvocab.${pair}.spm`,
  };
  const n = seq++;
  return {
    name: name ?? names[fileType],
    fromLang: from,
    toLang: to,
    version,
    fileType,
    filter_expression: filter,
    attachment: { hash: hash(n), size, filename: names[fileType], location: `main-workspace/translations-models/${n}.bin` },
  };
}

const set = (from, to, version, opts = {}, types = ["model", "lex", "vocab"]) => types.map((t) => record(from, to, version, t, opts));

describe("filter expressions and versions", () => {
  it("applies Windows release rules and refuses unknown expressions", () => {
    expect(filterApplies("")).toBe(true);
    expect(filterApplies(undefined)).toBe(true);
    expect(filterApplies("env.appinfo.OS != 'Android' || env.channel != 'release'")).toBe(true);
    expect(filterApplies("env.appinfo.OS == 'Android'")).toBe(false);
    expect(filterApplies("env.channel == 'default' || env.channel == 'nightly'")).toBe(false);
    expect(filterApplies("env.channel == 'default' || env.channel == 'nightly'", "nightly")).toBe(true);
    expect(filterApplies("env.version|versionCompare('150.0') >= 0")).toBeNull();
  });

  it("orders versions with pre-releases before their release", () => {
    const sorted = ["2.0", "1.0a1", "2.0a1", "1.1", "10.0", "2.1"].sort(compareVersions);
    expect(sorted).toEqual(["1.0a1", "1.1", "2.0a1", "2.0", "2.1", "10.0"]);
  });

  it("maps registry codes to canonical ids", () => {
    expect(canonicalCode("zh-Hans")).toBe("zh");
    expect(canonicalCode("zh-Hant")).toBe("zh-Hant");
    expect(directionId("zh-Hant", "en")).toBe("mt-bergamot/zh-hant_en");
    expect(directionId("en", "zh-Hans")).toBe("mt-bergamot/en_zh");
  });
});

describe("direction selection", () => {
  it("takes the highest applicable release version with a complete file set", () => {
    const records = [
      ...set("en", "ja", "2.1", { filter: "env.appinfo.OS != 'Android' || env.channel != 'release'" }, ["model", "lex", "srcvocab", "trgvocab"]),
      ...set("en", "ja", "2.2", { filter: "env.appinfo.OS == 'Android'" }, ["model", "lex", "srcvocab", "trgvocab"]),
      ...set("en", "ja", "3.0a1", {}, ["model", "lex", "srcvocab", "trgvocab"]),
      // 2.3 lacks its shortlist: skipped with a warning, 2.1 wins.
      ...set("en", "ja", "2.3", {}, ["model", "srcvocab", "trgvocab"]),
      ...set("ja", "en", "1.0"),
      ...set("ja", "en", "2.0"),
      ...set("mt", "en", "1.0", { filter: "env.channel == 'default' || env.channel == 'nightly'" }),
      ...set("de", "en", "1.0", { filter: "env.version|versionCompare('999.0') >= 0" }),
    ];
    const { directions, warnings } = selectDirections(records);
    expect(directions.map((d) => `${d.from}-${d.to}@${d.version}`)).toEqual(["en-ja@2.1", "ja-en@2.0"]);
    expect(directions[0].files.map((f) => f.fileType)).toEqual(["model", "lex", "srcvocab", "trgvocab"]);
    expect(warnings.some((w) => w.includes("2.3") && w.includes("shortlist"))).toBe(true);
    expect(warnings.some((w) => w.includes("unknown filter expression"))).toBe(true);
    const nightly = selectDirections(records, { channel: "nightly" }).directions.map((d) => `${d.from}-${d.to}@${d.version}`);
    expect(nightly).toContain("mt-en@1.0");
    expect(nightly, "pre-releases count on nightly").toContain("en-ja@3.0a1");
  });

  it("refuses incomplete, duplicated or unsafe file sets", () => {
    const bad = [
      ...set("en", "de", "1.0", {}, ["model", "lex"]), // no vocabulary
      ...set("en", "fr", "1.0", {}, ["model", "lex", "vocab", "srcvocab", "trgvocab"]), // both vocabulary kinds
      ...set("en", "es", "1.0", { name: "../evil.bin" }, ["model", "lex", "vocab"]),
      ...set("en", "it", "1.0", {}, ["model", "model", "lex", "vocab"]),
    ];
    expect(selectDirections(bad).directions).toEqual([]);
  });
});

describe("fragment", () => {
  it("builds pinned, runtime-requiring entries in a stable order", () => {
    const records = [...set("ja", "en", "2.0"), ...set("en", "zh-Hant", "2.0", {}, ["model", "lex", "srcvocab", "trgvocab"]), ...set("en", "ja", "2.0")];
    const { fragment } = buildFragment(records);
    expect(fragment.entries.map((e) => e.id)).toEqual(["mt-bergamot/en_ja", "mt-bergamot/en_zh-hant", "mt-bergamot/ja_en"]);
    const e = fragment.entries[1];
    expect(e).toMatchObject({ family: "mt-bergamot", kind: "model", version: "2.0", status: "published", requires: [RUNTIME_ID] });
    expect(e.title).toBe("English → Chinese (Traditional) (Firefox Translations)");
    for (const f of e.files) {
      expect(f.sha256).toMatch(/^[0-9a-f]{64}$/);
      expect(f.sources[0].kind).toBe("direct");
      expect(f.sources[0].url.startsWith(ATTACHMENTS_BASE)).toBe(true);
    }
    // Deterministic text for identical input.
    expect(fragmentText(buildFragment(records).fragment)).toBe(fragmentText(fragment));
  });

  it("the committed fragment is well formed", () => {
    const doc = JSON.parse(fs.readFileSync(DEFAULT_OUT, "utf8"));
    expect(path.basename(DEFAULT_OUT)).toBe("mt-bergamot.json");
    expect(doc.entries.length).toBeGreaterThanOrEqual(100);
    const ids = new Set();
    for (const e of doc.entries) {
      expect(e.id).toMatch(/^mt-bergamot\/[a-z-]+_[a-z-]+$/);
      expect(ids.has(e.id)).toBe(false);
      ids.add(e.id);
      expect(e.requires).toEqual([RUNTIME_ID]);
      const names = e.files.map((f) => f.path);
      expect(names.some((n) => /^model\..+\.intgemm(\.alphas|8)\.bin$/.test(n))).toBe(true);
      expect(names.some((n) => n.startsWith("lex."))).toBe(true);
      expect(names.some((n) => n.startsWith("vocab.")) || names.some((n) => n.startsWith("srcvocab."))).toBe(true);
      for (const f of e.files) {
        expect(f.size).toBeGreaterThan(0);
        expect(f.sha256).toMatch(/^[0-9a-f]{64}$/);
        expect(f.sources[0].url.startsWith(ATTACHMENTS_BASE)).toBe(true);
      }
    }
    for (const id of ["mt-bergamot/en_ja", "mt-bergamot/ja_en", "mt-bergamot/en_zh", "mt-bergamot/zh_en", "mt-bergamot/en_ko"]) {
      expect(ids.has(id)).toBe(true);
    }
  });
});
