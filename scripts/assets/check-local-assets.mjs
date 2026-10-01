/**
 * Check a SIGIL_LOCAL_ASSETS folder against the model catalog: every file of the selected entries
 * must be at `<dir>/<entry id>/<catalog path>` with the catalog's size and sha256. Use it after the
 * voice-changer export scripts (scripts/vc-templates/export_*.py) staged their files, before pointing
 * Sigil at the folder:
 *
 *   node scripts/assets/check-local-assets.mjs                       # unpublished vc/ and bench/ entries
 *   node scripts/assets/check-local-assets.mjs --all --prefix vc/    # published ones too (staged copies)
 *   node scripts/assets/check-local-assets.mjs --dir D:\sigil-assets --prefix vc/embedder-
 *
 * Then start Sigil (or `Sigil.exe --headless fetch <id>`) with SIGIL_LOCAL_ASSETS=<dir>; the model
 * store copies the files from there and verifies them again. Nothing here touches the network.
 *
 * Options:
 *   --dir <folder>      the staged assets (default %LOCALAPPDATA%\sigil-dev-cache\vc-assets)
 *   --prefix <id>       only entries whose id starts with this (repeatable; default vc/ and bench/)
 *   --all               include published entries (default: unpublished only, the ones that can
 *                       only come from a local folder)
 *   --fragment <file>   catalog fragment(s) to read (repeatable; default: every non-`_` fragment)
 */
import crypto from "node:crypto";
import fs from "node:fs";
import os from "node:os";
import path from "node:path";
import { fileURLToPath } from "node:url";

const here = path.dirname(fileURLToPath(import.meta.url));
const repo = path.resolve(here, "..", "..");
const catalogDir = path.join(repo, "src-tauri", "model-catalog");

function parseArgs(argv) {
  const opts = { dir: null, prefixes: [], all: false, fragments: [] };
  for (let i = 0; i < argv.length; i++) {
    const a = argv[i];
    const next = () => {
      if (i + 1 >= argv.length) throw new Error(`${a} needs a value`);
      return argv[++i];
    };
    if (a === "--dir") opts.dir = next();
    else if (a === "--prefix") opts.prefixes.push(next());
    else if (a === "--all") opts.all = true;
    else if (a === "--fragment") opts.fragments.push(next());
    else if (a === "--help" || a === "-h") {
      console.log(fs.readFileSync(fileURLToPath(import.meta.url), "utf8").split("*/")[0]);
      process.exit(0);
    } else throw new Error(`unknown option ${a}`);
  }
  const localAppData = process.env.LOCALAPPDATA || path.join(os.homedir(), "AppData", "Local");
  opts.dir ??= path.join(localAppData, "sigil-dev-cache", "vc-assets");
  if (opts.prefixes.length === 0) opts.prefixes = ["vc/", "bench/"];
  if (opts.fragments.length === 0) {
    opts.fragments = fs
      .readdirSync(catalogDir)
      .filter((f) => f.endsWith(".json") && !f.startsWith("_"))
      .map((f) => path.join(catalogDir, f));
  }
  return opts;
}

function sha256(file) {
  const h = crypto.createHash("sha256");
  const fd = fs.openSync(file, "r");
  const buf = Buffer.allocUnsafe(1 << 20);
  try {
    for (;;) {
      const n = fs.readSync(fd, buf, 0, buf.length, null);
      if (n === 0) break;
      h.update(buf.subarray(0, n));
    }
  } finally {
    fs.closeSync(fd);
  }
  return h.digest("hex");
}

function main() {
  const opts = parseArgs(process.argv.slice(2));
  let checked = 0;
  const problems = [];
  for (const fragment of opts.fragments) {
    const data = JSON.parse(fs.readFileSync(fragment, "utf8"));
    for (const entry of data.entries ?? []) {
      if (!opts.prefixes.some((p) => entry.id.startsWith(p))) continue;
      if (!opts.all && entry.status !== "unpublished") continue;
      for (const file of entry.files) {
        const local = path.join(opts.dir, ...entry.id.split("/"), ...file.path.split("/"));
        if (!fs.existsSync(local)) {
          if (!file.sha256 && !file.git_oid) {
            // import-only (unpinned) files are built by a maintainer step, not by the export scripts
            console.log(`note: ${entry.id}: ${file.path} is import-only and not staged`);
            continue;
          }
          checked++;
          problems.push(`${entry.id}: ${file.path} is missing (${local})`);
          continue;
        }
        checked++;
        const size = fs.statSync(local).size;
        if (file.size && size !== file.size) {
          problems.push(`${entry.id}: ${file.path} is ${size} bytes, the catalog says ${file.size}`);
          continue;
        }
        if (file.sha256) {
          const got = sha256(local);
          if (got !== file.sha256) problems.push(`${entry.id}: ${file.path} sha256 ${got} != ${file.sha256}`);
        } else {
          console.log(`note: ${entry.id}: ${file.path} is not pinned (accepted as is by an import)`);
        }
      }
    }
  }
  for (const p of problems) console.error(`FAIL ${p}`);
  console.log(`${checked - problems.length}/${checked} files match the catalog in ${opts.dir}`);
  if (problems.length === 0) console.log(`Set SIGIL_LOCAL_ASSETS=${opts.dir} to install them from there.`);
  process.exit(problems.length ? 1 : 0);
}

main();
