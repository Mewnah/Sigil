import { describe, expect, it } from "vitest";
import { editEntry, findEntry, findObjectEnd } from "../catalog-edit.mjs";
import { fragmentA, fragmentBText } from "./fixtures.mjs";

const SHA = "c".repeat(64);
const OID = "d".repeat(40);

describe("findObjectEnd", () => {
  it("skips braces inside strings and escaped quotes", () => {
    const t = '{"a":"}{\\"}","b":{"c":1}} tail';
    expect(t[findObjectEnd(t, 0)]).toBe("}");
    expect(findObjectEnd(t, 0)).toBe(t.indexOf("} tail"));
  });
  it("rejects unbalanced text", () => {
    expect(() => findObjectEnd('{"a":1', 0)).toThrow(/unbalanced/);
  });
});

describe("editEntry setStatus", () => {
  it("changes only the status of the named entry (JSON.stringify layout)", () => {
    const text = `${JSON.stringify(fragmentA(), null, 2)}\n`;
    const out = editEntry(text, "t/multi", { setStatus: "published" });
    const doc = JSON.parse(out);
    expect(doc.entries.find((e) => e.id === "t/multi").status).toBe("published");
    expect(doc.entries.find((e) => e.id === "t/solo").status).toBe("unpublished");
    expect(out.split("\n").length).toBe(text.split("\n").length);
    const diff = out.split("\n").filter((l, i) => l !== text.split("\n")[i]);
    expect(diff).toEqual(['      "status": "published",']);
  });

  it("keeps the compact layout of a hand-written fragment", () => {
    const text = fragmentBText();
    const out = editEntry(text, "t/pack", { setStatus: "published" });
    expect(out.replace('"status": "published"', '"status": "unpublished"')).toBe(text);
    expect(out).toContain('"requires": ["t/pub"]');
    expect(out).toContain('"archive": { "format": "zip", "include": ["a.dll", "sigil-pack.json"] }');
  });

  it("refuses an unknown id and an invalid status", () => {
    const text = fragmentBText();
    expect(() => editEntry(text, "t/none", { setStatus: "published" })).toThrow(/found 0 times/);
    expect(() => editEntry(text, "t/pack", { setStatus: "maybe" })).toThrow(/invalid status/);
  });

  it("does not confuse an id that appears only in requires with an entry", () => {
    const text = fragmentBText();
    expect(() => findEntry(text, "t/pub")).toThrow(/found 0 times/);
  });
});

describe("editEntry pinFile", () => {
  it("pins an unpinned file in the compact layout, in pin-catalog's key order", () => {
    const text = fragmentBText();
    const out = editEntry(text, "t/pack", { pinFile: { path: "pack-1.zip", size: 1234, sha256: SHA } });
    const file = JSON.parse(out).entries[0].files[0];
    expect(Object.keys(file)).toEqual(["path", "size", "sha256", "sources", "archive"]);
    expect(file.size).toBe(1234);
    expect(file.sha256).toBe(SHA);
    // Nothing else moved.
    const lines = text.split("\n");
    const changed = out.split("\n").filter((l) => !lines.includes(l));
    expect(changed).toEqual(['          "size": 1234,', `          "sha256": "${SHA}",`]);
  });

  it("pins on one line when the file object is inline", () => {
    const text = '{"entries":[{"id":"x/y","status":"unpublished","files":[{"path":"a.zip","size":0,"sources":[]}]}]}';
    const out = editEntry(text, "x/y", { pinFile: { path: "a.zip", size: 9, sha256: SHA } });
    expect(JSON.parse(out).entries[0].files[0]).toEqual({ path: "a.zip", size: 9, sha256: SHA, sources: [] });
  });

  it("replaces an existing pin and switches git_oid to sha256", () => {
    const text = `${JSON.stringify(fragmentA(), null, 2)}\n`;
    const out = editEntry(text, "t/solo", { pinFile: { path: "model.onnx", size: 77, sha256: SHA } });
    expect(JSON.parse(out).entries[0].files[0]).toMatchObject({ size: 77, sha256: SHA });
    const withOid = '{"entries":[{"id":"x/y","files":[{"path":"a.bin","size":5,"git_oid":"' + OID + '","sources":[]}]}]}';
    const swapped = JSON.parse(editEntry(withOid, "x/y", { pinFile: { path: "a.bin", size: 6, sha256: SHA } })).entries[0].files[0];
    expect(swapped).toEqual({ path: "a.bin", size: 6, sha256: SHA, sources: [] });
    const toOid = JSON.parse(editEntry(`${JSON.stringify(JSON.parse(withOid), null, 2)}\n`, "x/y", { pinFile: { path: "a.bin", size: 6, git_oid: "e".repeat(40) } })).entries[0].files[0];
    expect(toOid).toEqual({ path: "a.bin", size: 6, git_oid: "e".repeat(40), sources: [] });
  });

  it("rejects a bad pin and a file that is not there", () => {
    const text = fragmentBText();
    expect(() => editEntry(text, "t/pack", { pinFile: { path: "pack-1.zip", size: 0, sha256: SHA } })).toThrow(/invalid size/);
    expect(() => editEntry(text, "t/pack", { pinFile: { path: "pack-1.zip", size: 5, sha256: "xyz" } })).toThrow(/64 lower-case/);
    expect(() => editEntry(text, "t/pack", { pinFile: { path: "pack-1.zip", size: 5 } })).toThrow(/exactly one/);
    expect(() => editEntry(text, "t/pack", { pinFile: { path: "other.zip", size: 5, sha256: SHA } })).toThrow(/found 0 times/);
  });
});
