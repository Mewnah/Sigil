/**
 * Pin the catalog from local files: the offline counterpart of pin-catalog.mjs for assets that are
 * not on the network yet.
 *
 * pin-catalog.mjs reads a host's own metadata (Hugging Face tree API, GitHub release digests), so it
 * cannot pin a file of the `assets-v1` release before that file is uploaded, and
 * `--hash-by-download` against the not-yet-existing release is not an option. This script hashes the
 * staged copy instead (a folder holding the files under their release asset names, see
 * assets-v1.mjs) and writes the same thing pin-catalog writes: `size` and `sha256` of the file as the
 * store downloads it (for an archive entry, the zip / tar itself), in the key order path, size, hash,
 * sources, archive. Edits are made in place in the fragment's own layout (catalog-edit.mjs).
 *
 * - A catalog file without a pin gets one.
 * - A catalog file that has a pin is only checked: a staged file with other bytes is a MISMATCH
 *   (exit 1), never silently re-pinned. `--force` re-pins it (a rebuilt pack).
 * - Staged files that no catalog file names, and catalog files with nothing staged, are left alone
 *   (listed with `--verbose`).
 * - With `--write` the staging folder also gets SHA256SUMS.txt and MANIFEST.json (what each file is,
 *   which entries use it) next to the files; the upload script ignores both.
 *
 * Usage (repo root):
 *   node scripts/models/pin-staged.mjs --staging <dir>                 # dry run: print what would change
 *   node scripts/models/pin-staged.mjs --staging <dir> --write
 *   node scripts/models/pin-staged.mjs --staging <dir> --only runtime/sigil-transcribe- --force --write
 * Options:
 *   --staging <dir>      the folder of staged assets (required)
 *   --only <prefix>      only entries whose id starts with <prefix> (repeatable)
 *   --force              re-pin files that already have a pin and differ from the staged copy
 *   --write              write the fragments and the staging sidecars
 *   --catalog-dir <dir>  fragments to edit (default src-tauri/model-catalog; for tests)
 *   --verbose            also list files that are already pinned and equal
 * Hashing is sequential and reads each staged file once; run big passes through scripts/bench/gpu-lock.ps1.
 */
import fs from "node:fs";
import path from "node:path";
import { isDeepStrictEqual } from "node:util";
import { fileURLToPath, pathToFileURL } from "node:url";
import { loadCatalog } from "../release/check-catalog-pins.mjs";
import { collectAssets, comparePin, formatBytes, hashFile, scanStaging, RELEASE, RELEASE_BASE_URL } from "./assets-v1.mjs";
import { editEntry } from "./catalog-edit.mjs";
import { CATALOG_DIR, applyFact } from "./pin-catalog.mjs";

// ---------------------------------------------------------------------------------------------
// Pure
// ---------------------------------------------------------------------------------------------

/**
 * Decide, per catalog row with a staged copy, what to do. `hashes` maps asset name -> `{ size, sha256, git_oid? }`.
 * Actions: `ok` (pinned and equal), `pin` (no pin yet, or `force`), `mismatch` (pinned and different).
 */
export function planPins(rows, hashes, { only = [], force = false } = {}) {
  const actions = [];
  for (const r of rows) {
    if (!r.asset || !hashes.has(r.asset)) continue;
    if (only.length > 0 && !only.some((p) => r.id.startsWith(p))) continue;
    const h = hashes.get(r.asset);
    const base = { id: r.id, path: r.path, asset: r.asset, fragment: r.fragment, fact: { size: h.size, sha256: h.sha256 } };
    if (!r.pinned) actions.push({ ...base, action: "pin", was: r.size > 0 || r.sha256 || r.git_oid ? "incomplete pin" : "no pin" });
    else {
      const why = comparePin(r, h);
      if (!why) actions.push({ ...base, action: "ok" });
      else if (force) actions.push({ ...base, action: "pin", was: why });
      else actions.push({ ...base, action: "mismatch", was: why });
    }
  }
  return actions;
}

/**
 * Apply the `pin` actions of one fragment to its text. Returns `{ text, doc }`; throws when the edited
 * text does not parse to exactly the document `applyFact` produces.
 */
export function applyPinsToFragment(text, actions) {
  const expected = JSON.parse(text);
  let out = text;
  for (const a of actions) {
    const entry = (expected.entries ?? []).find((e) => e.id === a.id);
    const file = entry?.files?.find((f) => f.path === a.path);
    if (!file) throw new Error(`${a.id} ${a.path}: not in the fragment`);
    applyFact(file, a.fact);
    out = editEntry(out, a.id, { pinFile: { path: a.path, ...a.fact } });
  }
  const got = JSON.parse(out);
  if (!isDeepStrictEqual(got, expected)) throw new Error("the edited fragment does not equal the pinned document (layout not understood); nothing written");
  return { text: out, doc: got };
}

/** `sha256sum`-format listing of the staged assets. */
export function sumsText(hashes) {
  return `${[...hashes.entries()]
    .sort(([a], [b]) => a.localeCompare(b))
    .map(([asset, h]) => `${h.sha256}  ${asset}`)
    .join("\n")}\n`;
}

/** The staging folder's MANIFEST.json content. */
export function manifestOf(rows, hashes, now = new Date()) {
  const files = new Map();
  for (const r of rows) {
    if (!r.asset || !hashes.has(r.asset)) continue;
    const f = files.get(r.asset) ?? { asset: r.asset, size: hashes.get(r.asset).size, sha256: hashes.get(r.asset).sha256, entries: [] };
    if (!f.entries.includes(r.id)) f.entries.push(r.id);
    files.set(r.asset, f);
  }
  return {
    schema: "sigil-assets-staging/1",
    generated_at: now.toISOString(),
    release: { ...RELEASE, baseUrl: RELEASE_BASE_URL },
    note: "Upload with: pwsh scripts/release/publish-assets.ps1 -Staging <this folder>. Hashes were taken from the files in this folder when the catalog was pinned.",
    files: [...files.values()].sort((a, b) => a.asset.localeCompare(b.asset)),
  };
}

export function parseArgs(argv) {
  const opts = { staging: null, only: [], force: false, write: false, catalogDir: CATALOG_DIR, verbose: false, help: false };
  for (let i = 0; i < argv.length; i += 1) {
    const a = argv[i];
    const val = () => {
      if (i + 1 >= argv.length) throw new Error(`${a} needs a value`);
      return argv[++i];
    };
    if (a === "--staging") opts.staging = val();
    else if (a === "--only") opts.only.push(val());
    else if (a === "--force") opts.force = true;
    else if (a === "--write") opts.write = true;
    else if (a === "--catalog-dir") opts.catalogDir = val();
    else if (a === "--verbose") opts.verbose = true;
    else if (a === "--help" || a === "-h") opts.help = true;
    else throw new Error(`unknown argument ${a}`);
  }
  if (!opts.help && !opts.staging) throw new Error("--staging <dir> is required");
  return opts;
}

// ---------------------------------------------------------------------------------------------
// CLI
// ---------------------------------------------------------------------------------------------

export async function main(argv = process.argv.slice(2), io = { out: console.log, err: console.error }) {
  let opts;
  try {
    opts = parseArgs(argv);
  } catch (e) {
    io.err(`pin-staged: ${e.message}`);
    return 2;
  }
  if (opts.help) {
    io.out(fs.readFileSync(fileURLToPath(import.meta.url), "utf8").split("*/")[0] + "*/");
    return 0;
  }
  if (!fs.existsSync(opts.staging) || !fs.statSync(opts.staging).isDirectory()) {
    io.err(`pin-staged: ${opts.staging} is not a folder`);
    return 2;
  }
  const catalog = loadCatalog(opts.catalogDir);
  const rows = collectAssets(catalog);
  const staged = scanStaging(opts.staging);

  // Hash each staged asset the catalog names (once), in the catalog's order.
  const hashes = new Map();
  for (const r of rows) {
    if (!r.asset || hashes.has(r.asset) || !staged.has(r.asset)) continue;
    if (opts.only.length > 0 && !opts.only.some((p) => r.id.startsWith(p))) continue;
    io.err(`hash ${r.asset} (${formatBytes(staged.get(r.asset).size)})`);
    hashes.set(r.asset, await hashFile(staged.get(r.asset).file, { gitOid: Boolean(r.git_oid) }));
  }

  const actions = planPins(rows, hashes, opts);
  let mismatches = 0;
  for (const a of actions) {
    if (a.action === "mismatch") {
      mismatches += 1;
      io.out(`MISMATCH ${a.id} ${a.path}: staged ${a.asset} has ${a.was} (use --force to re-pin a rebuilt file)`);
    } else if (a.action === "pin") {
      io.out(`pin      ${a.id} ${a.path}: size ${a.fact.size}, sha256 ${a.fact.sha256.slice(0, 16)}... (${a.was})`);
    } else if (opts.verbose) io.out(`ok       ${a.id} ${a.path}`);
  }
  const pins = actions.filter((a) => a.action === "pin");
  io.err(`${actions.length} staged file(s) matched to the catalog: ${actions.filter((a) => a.action === "ok").length} already pinned and equal, ${pins.length} to pin, ${mismatches} mismatch(es).`);

  if (opts.write) {
    const byFragment = new Map();
    for (const a of pins) {
      if (!byFragment.has(a.fragment)) byFragment.set(a.fragment, []);
      byFragment.get(a.fragment).push(a);
    }
    for (const [name, list] of byFragment) {
      const file = path.join(opts.catalogDir, name);
      const { text } = applyPinsToFragment(fs.readFileSync(file, "utf8"), list);
      fs.writeFileSync(file, text);
      io.err(`wrote ${name} (${list.length} file pin(s))`);
    }
    if (hashes.size > 0 && opts.only.length === 0) {
      fs.writeFileSync(path.join(opts.staging, "SHA256SUMS.txt"), sumsText(hashes));
      fs.writeFileSync(path.join(opts.staging, "MANIFEST.json"), `${JSON.stringify(manifestOf(rows, hashes), null, 2)}\n`);
      io.err("wrote SHA256SUMS.txt and MANIFEST.json in the staging folder");
    }
  }
  return mismatches > 0 ? 1 : 0;
}

if (process.argv[1] && import.meta.url === pathToFileURL(path.resolve(process.argv[1])).href) {
  main().then(
    (code) => {
      process.exitCode = code;
    },
    (e) => {
      console.error(`pin-staged: ${e.message}`);
      process.exitCode = 1;
    },
  );
}
