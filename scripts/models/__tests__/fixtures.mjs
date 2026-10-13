// Shared fixtures for the assets-v1 tooling tests: a tiny catalog in two fragment layouts (the
// JSON.stringify layout and the compact hand-written one) and the bytes its files stand for.
import crypto from "node:crypto";
import fs from "node:fs";
import os from "node:os";
import path from "node:path";

export const BASE = "https://github.com/Mewnah/Sigil-assets/releases/download/assets-v1/";
export const COMMIT = "0123456789abcdef0123456789abcdef01234567";

export const sha256 = (buf) => crypto.createHash("sha256").update(buf).digest("hex");
export const gitOid = (buf) => crypto.createHash("sha1").update(`blob ${buf.length}\0`).update(buf).digest("hex");

/** The bytes behind each asset name. */
export const BYTES = {
  "t-solo-model.onnx": Buffer.from("solo model bytes"),
  "t-multi-a.bin": Buffer.from("multi a"),
  "t-multi-b.json": Buffer.from('{"b":1}'),
  "pack-1.zip": Buffer.from("PK pack bytes"),
  "t-dep-model.onnx": Buffer.from("dependent model bytes"),
  "up-model.gguf": Buffer.from("upstream model bytes"),
};

const src = (name) => ({ kind: "github-release", url: BASE + name, revision: "assets-v1" });
const pin = (name) => ({ size: BYTES[name].length, sha256: sha256(BYTES[name]) });

/** Fragment a.json in JSON.stringify layout. */
export function fragmentA() {
  return {
    $schema: "./_schema.json",
    entries: [
      {
        id: "t/solo",
        family: "t",
        kind: "model",
        version: "1",
        license: "MIT",
        status: "unpublished",
        files: [{ path: "model.onnx", ...pin("t-solo-model.onnx"), sources: [src("t-solo-model.onnx")] }],
      },
      {
        id: "t/multi",
        family: "t",
        kind: "model",
        version: "1",
        license: "MIT",
        status: "unpublished",
        files: [
          { path: "a.bin", ...pin("t-multi-a.bin"), sources: [src("t-multi-a.bin")] },
          { path: "b.json", ...pin("t-multi-b.json"), sources: [src("t-multi-b.json")] },
        ],
      },
      {
        id: "t/dep",
        family: "t",
        kind: "model",
        version: "1",
        license: "MIT",
        status: "unpublished",
        requires: ["t/solo"],
        files: [{ path: "model.onnx", ...pin("t-dep-model.onnx"), sources: [src("t-dep-model.onnx")] }],
      },
      {
        id: "t/pub",
        family: "t",
        kind: "model",
        version: "1",
        license: "MIT",
        status: "published",
        files: [{ path: "x.bin", size: 5, sha256: "a".repeat(64), sources: [{ kind: "direct", url: `https://raw.githubusercontent.com/o/r/${COMMIT}/x.bin` }] }],
      },
    ],
  };
}

/** Fragment b.json written by hand: compact arrays and an inline archive, an unpinned zip and an upstream-backed file. */
export function fragmentBText() {
  const up = BYTES["up-model.gguf"];
  return `{
  "$schema": "./_schema.json",
  "entries": [
    {
      "id": "t/pack",
      "family": "runtime",
      "kind": "runtime",
      "version": "1",
      "license": "MIT",
      "status": "unpublished",
      "requires": ["t/pub"],
      "variant": "cpu",
      "files": [
        {
          "path": "pack-1.zip",
          "size": 0,
          "sources": [{ "kind": "github-release", "url": "${BASE}pack-1.zip", "revision": "assets-v1" }],
          "archive": { "format": "zip", "include": ["a.dll", "sigil-pack.json"] }
        }
      ]
    },
    {
      "id": "t/up",
      "family": "t",
      "kind": "model",
      "version": "1",
      "license": "MIT",
      "status": "unpublished",
      "files": [
        {
          "path": "model.gguf",
          "size": ${up.length},
          "sha256": "${sha256(up)}",
          "sources": [
            { "kind": "huggingface", "url": "https://huggingface.co/org/repo/resolve/${COMMIT}/model.gguf", "revision": "${COMMIT}" },
            { "kind": "github-release", "url": "${BASE}up-model.gguf", "revision": "assets-v1" }
          ]
        }
      ]
    },
    {
      "id": "t/nested",
      "family": "t",
      "kind": "model",
      "version": "1",
      "license": "MIT",
      "status": "unpublished",
      "files": [
        { "path": "n.txt", "size": 3, "sha256": "${"b".repeat(64)}", "sources": [{ "kind": "github-release", "url": "${BASE}folder/n.txt" }] }
      ]
    }
  ]
}
`;
}

/** Create a scratch folder (removed by the caller). */
export function tempDir(prefix = "sigil-assets-test-") {
  return fs.mkdtempSync(path.join(os.tmpdir(), prefix));
}

/** A catalog folder with both fragments, a staging folder with the assets, returns paths. */
export function makeWorld({ stage = Object.keys(BYTES) } = {}) {
  const root = tempDir();
  const catalogDir = path.join(root, "catalog");
  const staging = path.join(root, "staging");
  fs.mkdirSync(catalogDir);
  fs.mkdirSync(staging);
  fs.writeFileSync(path.join(catalogDir, "a.json"), `${JSON.stringify(fragmentA(), null, 2)}\n`);
  fs.writeFileSync(path.join(catalogDir, "b.json"), fragmentBText());
  for (const name of stage) fs.writeFileSync(path.join(staging, name), BYTES[name] ?? Buffer.from(`stray ${name}`));
  return { root, catalogDir, staging, cleanup: () => fs.rmSync(root, { recursive: true, force: true }) };
}
