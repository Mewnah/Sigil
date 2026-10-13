/**
 * The `assets-v1` release of this repository: which catalog files it must carry, and what a local
 * staging folder holds of them. Shared by pin-staged.mjs, publish-entries.mjs and (through the
 * `plan` command below) scripts/release/publish-assets.ps1.
 *
 * Background: Sigil-built packs, exports and templates are `unpublished` catalog entries whose files
 * list `https://github.com/Mewnah/Sigil-assets/releases/download/assets-v1/<asset name>` as their source.
 * The Models tab offers a download only for `published` entries, so the flow is: stage the files
 * under their asset names, pin them (size + sha256) in the catalog, upload them to the release, then
 * flip the entries to `published` after checking the uploaded bytes (publish-entries.mjs).
 *
 * A GitHub release is flat: an asset name has no `/`. A catalog source URL under the release base
 * with a `/` after it is reported as a problem (it can never resolve).
 *
 * Commands (repo root):
 *   node scripts/models/assets-v1.mjs plan --staging <dir> [--hash] [--json] [--catalog-dir <dir>]
 *       Compare a staging folder with the catalog: what the upload would send, what is missing, what
 *       no entry names. `--hash` also hashes every candidate against its pin (slow; sequential).
 */
import crypto from "node:crypto";
import fs from "node:fs";
import path from "node:path";
import { fileURLToPath, pathToFileURL } from "node:url";
import { loadCatalog } from "../release/check-catalog-pins.mjs";
import { CATALOG_DIR } from "./pin-catalog.mjs";

export const RELEASE = { owner: "Mewnah", repo: "Sigil-assets", tag: "assets-v1" };
export const RELEASE_BASE_URL = `https://github.com/${RELEASE.owner}/${RELEASE.repo}/releases/download/${RELEASE.tag}/`;
/** Files in a staging folder that are not release assets. */
export const STAGING_SIDECARS = new Set(["SHA256SUMS.txt", "MANIFEST.json"]);

// ---------------------------------------------------------------------------------------------
// Catalog side (pure)
// ---------------------------------------------------------------------------------------------

/**
 * The asset a source URL names: `{ name }` for a flat asset of the release at `base`, `{ error }` for a
 * URL under the base that cannot be an asset (nested, empty), `null` for any other URL.
 */
export function assetOfUrl(url, base = RELEASE_BASE_URL) {
  if (typeof url !== "string" || !url.startsWith(base)) return null;
  let rest;
  try {
    rest = decodeURIComponent(url.slice(base.length));
  } catch {
    return { error: "the asset name is not valid percent-encoding" };
  }
  if (rest === "") return { error: "no asset name after the release base" };
  if (/[/\\?#]/.test(rest)) return { error: `"${rest}" is not a flat asset name (a release has no folders)` };
  return { name: rest };
}

/** True when the file carries a hash and a positive size (what a published entry needs). */
export function isPinnedFile(file) {
  return Boolean(file.sha256 || file.git_oid) && Number.isSafeInteger(file.size) && file.size > 0;
}

/**
 * One row per catalog file that lists an assets-v1 source.
 * `upstream` is true when the file also has a source on another host (Hugging Face, ...): then
 * the release copy is only a mirror and the file does not depend on it.
 */
export function collectAssets(catalog, base = RELEASE_BASE_URL) {
  const rows = [];
  for (const [id, { entry, fragment }] of catalog.entries) {
    for (const file of entry.files ?? []) {
      const sources = file.sources ?? [];
      const ours = sources.map((s) => ({ s, a: assetOfUrl(s.url, base) })).filter((x) => x.a);
      if (ours.length === 0) continue;
      const good = ours.find((x) => x.a.name);
      rows.push({
        id,
        fragment,
        status: entry.status ?? "published",
        path: file.path,
        size: file.size,
        sha256: file.sha256,
        git_oid: file.git_oid,
        pinned: isPinnedFile(file),
        asset: good ? good.a.name : null,
        error: good ? null : ours[0].a.error,
        upstream: sources.some((s) => !assetOfUrl(s.url, base)),
        archive: Boolean(file.archive),
      });
    }
  }
  return rows;
}

// ---------------------------------------------------------------------------------------------
// Staging side
// ---------------------------------------------------------------------------------------------

/** The release assets in a staging folder: Map(name -> { file, size }). Sub-folders, sidecars and dotfiles are ignored. */
export function scanStaging(dir) {
  const out = new Map();
  for (const d of fs.readdirSync(dir, { withFileTypes: true })) {
    if (!d.isFile() || d.name.startsWith(".") || STAGING_SIDECARS.has(d.name) || d.name.endsWith(".partial")) continue;
    const file = path.join(dir, d.name);
    out.set(d.name, { file, size: fs.statSync(file).size });
  }
  return out;
}

/**
 * Compare catalog rows with a staging folder.
 *
 * Per asset `state`: `ok` (present, pinned, size equal), `unpinned` (present, the catalog has no pin
 * yet: run pin-staged.mjs), `size-mismatch`, `missing`, `conflict` (two entries pin one asset name
 * differently), `invalid` (the URL cannot be a release asset).
 *
 * Per entry `state`: `ready` (every file is a staged `ok` asset or backed by another host) or
 * `incomplete`; `staged` says that at least one file comes from the staging folder.
 */
export function planStaging(rows, staged) {
  const assets = new Map();
  const problems = [];
  for (const r of rows) {
    if (!r.asset) {
      problems.push({ level: "error", id: r.id, message: `${r.id} ${r.path}: ${r.error}` });
      continue;
    }
    const prev = assets.get(r.asset);
    if (!prev) assets.set(r.asset, { asset: r.asset, rows: [r], size: r.size, sha256: r.sha256, git_oid: r.git_oid, pinned: r.pinned, entries: [r.id] });
    else {
      prev.rows.push(r);
      if (!prev.entries.includes(r.id)) prev.entries.push(r.id);
      if (r.size !== prev.size || r.sha256 !== prev.sha256 || r.git_oid !== prev.git_oid) prev.conflict = true;
    }
  }
  const files = [];
  for (const a of assets.values()) {
    const have = staged.get(a.asset);
    let state;
    if (a.conflict) state = "conflict";
    else if (!have) state = "missing";
    else if (!a.pinned) state = "unpinned";
    else if (have.size !== a.size) state = "size-mismatch";
    else state = "ok";
    // An asset whose every file has another host is an optional mirror: a missing copy is fine.
    const optional = a.rows.every((r) => r.upstream);
    files.push({
      asset: a.asset,
      state,
      optional,
      expectedSize: a.pinned ? a.size : null,
      stagedSize: have ? have.size : null,
      file: have ? have.file : null,
      entries: a.entries,
    });
  }
  files.sort((x, y) => x.asset.localeCompare(y.asset));

  const byAsset = new Map(files.map((f) => [f.asset, f]));
  const entries = new Map();
  for (const r of rows) {
    if (!entries.has(r.id)) entries.set(r.id, { id: r.id, status: r.status, files: [], staged: false, ready: true, reasons: [] });
    const e = entries.get(r.id);
    const f = r.asset ? byAsset.get(r.asset) : null;
    if (f && f.state === "ok") {
      e.staged = true;
      e.files.push({ path: r.path, asset: r.asset, via: "staging" });
    } else if (r.upstream && r.pinned) {
      e.files.push({ path: r.path, asset: r.asset, via: "upstream" });
    } else {
      e.ready = false;
      e.reasons.push(`${r.path}: ${f ? f.state : "invalid"}`);
      e.files.push({ path: r.path, asset: r.asset, via: null });
    }
  }
  const extras = [...staged.keys()].filter((n) => !assets.has(n)).sort();
  return { files, entries, extras, problems };
}

/**
 * Entries of `spec`: a comma/space separated id list, or `all-staged` = every unpublished entry
 * that is `ready` and takes at least one file from the staging folder. Returns `{ ids, errors }`.
 */
export function selectIds(spec, catalog, plan) {
  const errors = [];
  const tokens = String(spec ?? "")
    .split(/[,\s]+/)
    .filter(Boolean);
  if (tokens.length === 0) return { ids: [], errors: ["--ids needs a list of catalog ids or all-staged"] };
  if (tokens.includes("all-staged")) {
    if (tokens.length > 1) errors.push("all-staged cannot be combined with other ids");
    if (!plan) return { ids: [], errors: ["all-staged needs --staging <dir> (the folder the files were staged in)"] };
    const ids = [...plan.entries.values()].filter((e) => e.status === "unpublished" && e.ready && e.staged).map((e) => e.id);
    return { ids, errors };
  }
  const ids = [];
  for (const t of tokens) {
    if (!catalog.entries.has(t)) errors.push(`${t} is not a catalog id`);
    else if (!ids.includes(t)) ids.push(t);
  }
  return { ids, errors };
}

// ---------------------------------------------------------------------------------------------
// Hashing
// ---------------------------------------------------------------------------------------------

/** Stream a file through sha256 (and sha1 of the git blob form when asked). Returns `{ size, sha256, git_oid? }`. */
export async function hashFile(file, { gitOid = false } = {}) {
  const size = fs.statSync(file).size;
  const sha = crypto.createHash("sha256");
  const git = gitOid ? crypto.createHash("sha1").update(`blob ${size}\0`) : null;
  for await (const chunk of fs.createReadStream(file, { highWaterMark: 1 << 20 })) {
    sha.update(chunk);
    git?.update(chunk);
  }
  return { size, sha256: sha.digest("hex"), ...(git ? { git_oid: git.digest("hex") } : {}) };
}

/** Does a hashed file equal the catalog pin of `row`? Returns null when equal, else a reason. */
export function comparePin(row, hashed) {
  if (hashed.size !== row.size) return `size ${hashed.size}, catalog ${row.size}`;
  if (row.sha256) return hashed.sha256 === row.sha256 ? null : `sha256 ${hashed.sha256}, catalog ${row.sha256}`;
  if (row.git_oid) return hashed.git_oid === row.git_oid ? null : `git blob id ${hashed.git_oid}, catalog ${row.git_oid}`;
  return "the catalog has no pin";
}

export function formatBytes(n) {
  if (n >= 1 << 30) return `${(n / (1 << 30)).toFixed(2)} GiB`;
  if (n >= 1 << 20) return `${(n / (1 << 20)).toFixed(1)} MiB`;
  if (n >= 1 << 10) return `${(n / (1 << 10)).toFixed(1)} KiB`;
  return `${n} B`;
}

// ---------------------------------------------------------------------------------------------
// CLI: plan
// ---------------------------------------------------------------------------------------------

export function parsePlanArgs(argv) {
  const opts = { staging: null, hash: false, json: false, catalogDir: CATALOG_DIR, help: false };
  const args = [...argv];
  const cmd = args.shift();
  if (cmd === "--help" || cmd === "-h") opts.help = true;
  else if (cmd !== "plan") throw new Error(`unknown command ${cmd ?? "(none)"}; use: plan --staging <dir>`);
  for (let i = 0; i < args.length; i += 1) {
    const a = args[i];
    const val = () => {
      if (i + 1 >= args.length) throw new Error(`${a} needs a value`);
      return args[++i];
    };
    if (a === "--staging") opts.staging = val();
    else if (a === "--hash") opts.hash = true;
    else if (a === "--json") opts.json = true;
    else if (a === "--catalog-dir") opts.catalogDir = val();
    else if (a === "--help" || a === "-h") opts.help = true;
    else throw new Error(`unknown argument ${a}`);
  }
  if (!opts.help && !opts.staging) throw new Error("plan needs --staging <dir>");
  return opts;
}

/** Build the plan object for a staging folder (hashing the candidates when asked). */
export async function buildPlan({ staging, catalogDir = CATALOG_DIR, hash = false, onHash = () => {} }) {
  const catalog = loadCatalog(catalogDir);
  const rows = collectAssets(catalog);
  const plan = planStaging(rows, scanStaging(staging));
  const problems = [...plan.problems];
  if (hash) {
    const byAsset = new Map(plan.files.map((f) => [f.asset, f]));
    for (const r of rows) {
      const f = r.asset ? byAsset.get(r.asset) : null;
      if (!f || f.state !== "ok" || f.hashed) continue;
      onHash(f.asset);
      const h = await hashFile(f.file, { gitOid: Boolean(r.git_oid) });
      const why = comparePin(r, h);
      f.hashed = true;
      if (why) {
        f.state = "hash-mismatch";
        f.detail = why;
        problems.push({ level: "error", id: r.id, message: `${f.asset}: ${why}` });
      }
    }
    // An entry whose staged file failed its hash is not ready.
    for (const e of plan.entries.values()) {
      for (const ef of e.files) {
        if (ef.via === "staging" && byAsset.get(ef.asset)?.state === "hash-mismatch") {
          e.ready = false;
          e.reasons.push(`${ef.path}: hash-mismatch`);
        }
      }
    }
  }
  return { catalog, rows, ...plan, problems };
}

/** The JSON the PowerShell uploader reads. */
export function planToJson(plan, staging) {
  const upload = plan.files.filter((f) => f.state === "ok");
  return {
    schema: "sigil-assets-plan/1",
    release: { ...RELEASE, baseUrl: RELEASE_BASE_URL },
    staging: path.resolve(staging),
    upload: upload.map((f) => ({ asset: f.asset, size: f.stagedSize, file: f.file, entries: f.entries })),
    notStaged: plan.files.filter((f) => f.state === "missing" && !f.optional).map((f) => ({ asset: f.asset, entries: f.entries })),
    optionalMirrors: plan.files.filter((f) => f.state === "missing" && f.optional).map((f) => ({ asset: f.asset, entries: f.entries })),
    blocked: plan.files.filter((f) => !["ok", "missing"].includes(f.state)).map((f) => ({ asset: f.asset, state: f.state, detail: f.detail ?? null, entries: f.entries })),
    extras: plan.extras,
    problems: plan.problems,
    totals: { files: upload.length, bytes: upload.reduce((a, f) => a + f.stagedSize, 0) },
  };
}

export function printPlan(plan, staging, out = console.log) {
  const json = planToJson(plan, staging);
  out(`staging: ${json.staging}`);
  out(`release: ${RELEASE.owner}/${RELEASE.repo} tag ${RELEASE.tag}`);
  out(`would upload ${json.totals.files} file(s), ${formatBytes(json.totals.bytes)}`);
  for (const f of json.upload) out(`  ${f.asset.padEnd(64)} ${formatBytes(f.size).padStart(10)}  ${f.entries.join(", ")}`);
  if (json.blocked.length > 0) {
    out(`not uploadable (${json.blocked.length}):`);
    for (const f of json.blocked) out(`  ${f.asset}: ${f.state}${f.detail ? ` (${f.detail})` : ""}`);
  }
  if (json.notStaged.length > 0) {
    out(`in the catalog, not staged (${json.notStaged.length}):`);
    for (const f of json.notStaged) out(`  ${f.asset}  (${f.entries.join(", ")})`);
  }
  if (json.optionalMirrors.length > 0) {
    out(`optional mirrors not staged (${json.optionalMirrors.length}; the file also has an upstream host):`);
    for (const f of json.optionalMirrors) out(`  ${f.asset}  (${f.entries.join(", ")})`);
  }
  if (json.extras.length > 0) out(`in the folder but named by no entry (ignored): ${json.extras.join(", ")}`);
  for (const p of json.problems) out(`  ${p.level.toUpperCase()}: ${p.message}`);
}

async function main(argv) {
  const opts = parsePlanArgs(argv);
  if (opts.help) {
    const text = fs.readFileSync(fileURLToPath(import.meta.url), "utf8");
    console.log(text.slice(0, text.indexOf("*/") + 2));
    return 0;
  }
  if (!fs.existsSync(opts.staging) || !fs.statSync(opts.staging).isDirectory()) throw new Error(`${opts.staging} is not a folder`);
  const plan = await buildPlan({ staging: opts.staging, catalogDir: opts.catalogDir, hash: opts.hash, onHash: (a) => !opts.json && console.error(`hash ${a}`) });
  if (opts.json) console.log(JSON.stringify(planToJson(plan, opts.staging)));
  else printPlan(plan, opts.staging);
  return plan.problems.some((p) => p.level === "error") ? 1 : 0;
}

if (process.argv[1] && import.meta.url === pathToFileURL(path.resolve(process.argv[1])).href) {
  main(process.argv.slice(2)).then(
    (code) => {
      process.exitCode = code;
    },
    (e) => {
      console.error(`assets-v1: ${e.message}`);
      process.exitCode = 2;
    },
  );
}
