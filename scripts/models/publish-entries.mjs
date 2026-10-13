/**
 * Flip catalog entries from `unpublished` to `published` once their files are really on the
 * `assets-v1` release (or at an upstream host) with the bytes the catalog pins.
 *
 * The Models tab offers a download button only for published entries. Sigil-built packs, exports and
 * templates start `unpublished` with a pinned `github-release` source on
 * https://github.com/Mewnah/Sigil-assets/releases/download/assets-v1/<asset>; after the staged files are
 * uploaded (scripts/release/publish-assets.ps1) this script proves the uploads and flips the status.
 * It never uploads anything and never uses a credential.
 *
 * For each selected entry, one file at a time and at least 2 s between request starts:
 *   - every `assets-v1` source URL must answer and its bytes must hash to the catalog pin (size and
 *     sha256 / git blob id). The file is downloaded to a temp file that is deleted right after;
 *     `--digest-only` asks the GitHub release API for the digest GitHub computed instead (one
 *     request for the whole release; a file whose asset publishes no digest then fails).
 *   - a file that also has an upstream source (Hugging Face, ...) is checked against that host's own
 *     metadata (no download); its assets-v1 URL is an optional mirror, checked only when the asset
 *     is in the staging folder.
 *   - an entry is flipped only when every file passed, every file is pinned with a size, and every
 *     entry it `requires` is published (or flipped in the same run).
 *
 * Dry run by default: it prints what it would do and writes nothing. `--write` edits the fragments in
 * place (only the status line changes; layout and generated fragments stay as they are).
 *
 * Usage (repo root):
 *   node scripts/models/publish-entries.mjs --ids all-staged --staging <dir>            # dry run
 *   node scripts/models/publish-entries.mjs --ids all-staged --staging <dir> --write
 *   node scripts/models/publish-entries.mjs --ids bench/gpu-load,vc/pitch-fcpe --write
 *   node scripts/models/publish-entries.mjs --ids all-staged --staging <dir> --digest-only --write
 * Options:
 *   --ids <list|all-staged>  comma-separated catalog ids, or every unpublished entry that is fully
 *                            staged (needs --staging; an entry served only by an upstream host has
 *                            nothing staged, so name it explicitly)
 *   --except <id>            drop an id from the selection (repeatable)
 *   --staging <dir>          the staging folder (see assets-v1.mjs)
 *   --write                  flip the verified entries in the fragments
 *   --digest-only            compare against GitHub's release asset digests instead of downloading
 *   --assets-base <url>      serve the assets from here instead of the release (tests: an https URL
 *                            or http://127.0.0.1:<port>/ with the staged files laid out flat)
 *   --catalog-dir <dir>      fragments to read and edit (default src-tauri/model-catalog)
 *   --temp-dir <dir>         where downloads are written (default the system temp folder)
 *   --delay-ms <n>           spacing between request starts (default 2500, minimum 2000)
 * Exit code: 0 every selected entry verified, 1 any failure, 2 usage.
 *
 * Credentials: the downloads from the release carry none. The metadata lookups (Hugging Face tree API,
 * GitHub release API) go through pin-catalog.mjs's client, which sends HF_TOKEN / GITHUB_TOKEN as
 * headers only when you have set them yourself; this script never reads, asks for or stores one.
 */
import fs from "node:fs";
import os from "node:os";
import path from "node:path";
import crypto from "node:crypto";
import { Readable, Transform } from "node:stream";
import { pipeline } from "node:stream/promises";
import { fileURLToPath, pathToFileURL } from "node:url";
import { FreshReleaseNet, checkCatalog, classifySource, loadCatalog, loadReleaseSet, verifyFileOnline } from "../release/check-catalog-pins.mjs";
import { CATALOG_DIR, MIN_DELAY_MS, Throttle } from "./pin-catalog.mjs";
import { RELEASE_BASE_URL, assetOfUrl, collectAssets, comparePin, formatBytes, isPinnedFile, planStaging, scanStaging, selectIds } from "./assets-v1.mjs";
import { editEntry } from "./catalog-edit.mjs";

const USER_AGENT = "Sigil-publish-entries (https://github.com/Mewnah/Sigil)";
const IDLE_MS = 60_000;

// ---------------------------------------------------------------------------------------------
// Pure
// ---------------------------------------------------------------------------------------------

export function parseArgs(argv) {
  const opts = {
    ids: null,
    except: [],
    staging: null,
    write: false,
    digestOnly: false,
    assetsBase: RELEASE_BASE_URL,
    catalogDir: CATALOG_DIR,
    tempDir: null,
    delayMs: 2500,
    help: false,
  };
  for (let i = 0; i < argv.length; i += 1) {
    const a = argv[i];
    const val = () => {
      if (i + 1 >= argv.length) throw new Error(`${a} needs a value`);
      return argv[++i];
    };
    if (a === "--ids") opts.ids = val();
    else if (a === "--except") opts.except.push(val());
    else if (a === "--staging") opts.staging = val();
    else if (a === "--write") opts.write = true;
    else if (a === "--digest-only") opts.digestOnly = true;
    else if (a === "--assets-base") opts.assetsBase = val();
    else if (a === "--catalog-dir") opts.catalogDir = val();
    else if (a === "--temp-dir") opts.tempDir = val();
    else if (a === "--delay-ms") opts.delayMs = Number(val());
    else if (a === "--help" || a === "-h") opts.help = true;
    else throw new Error(`unknown argument ${a}`);
  }
  if (opts.help) return opts;
  if (!opts.ids) throw new Error("--ids <list|all-staged> is required");
  if (!Number.isFinite(opts.delayMs) || opts.delayMs < MIN_DELAY_MS) throw new Error(`--delay-ms must be at least ${MIN_DELAY_MS} (the dev network blocks request bursts)`);
  opts.assetsBase = normalizeBase(opts.assetsBase);
  if (opts.digestOnly && opts.assetsBase !== RELEASE_BASE_URL) throw new Error("--digest-only reads the real release's API digests; it cannot be combined with --assets-base");
  return opts;
}

/** A base URL must be https, or http on this PC (a local test server); it ends with `/`. */
export function normalizeBase(raw) {
  let u;
  try {
    u = new URL(raw);
  } catch {
    throw new Error(`--assets-base ${raw} is not a URL`);
  }
  const loopback = ["127.0.0.1", "localhost", "[::1]"].includes(u.hostname);
  if (u.protocol !== "https:" && !(u.protocol === "http:" && loopback)) throw new Error("--assets-base must be https:// (or http://127.0.0.1 for a local test)");
  if (u.username || u.password || u.search || u.hash) throw new Error("--assets-base must not carry credentials, a query or a fragment");
  return u.href.endsWith("/") ? u.href : `${u.href}/`;
}

/** The URL of an assets-v1 source under `base` (the catalog's own base is replaced by `base`). */
export function urlUnder(base, assetName) {
  return base + encodeURIComponent(assetName).replace(/%2F/gi, "/");
}

/**
 * Static checks before any request. Returns `{ errors, note }`: `errors` block the entry, `note` is a
 * reason to skip it silently (already published).
 */
export function preflight(entry, catalog, selected) {
  const errors = [];
  if ((entry.status ?? "published") === "published") return { errors, note: "already published" };
  for (const file of entry.files ?? []) {
    if (!isPinnedFile(file)) errors.push(`${file.path}: not pinned (size and sha256 needed; pin it with scripts/models/pin-staged.mjs)`);
    const usable = (file.sources ?? []).some((s) => assetOfUrl(s.url, RELEASE_BASE_URL)?.name || classifySource(s).immutable);
    if (!usable) errors.push(`${file.path}: no release asset source and no immutable upstream source`);
    for (const s of file.sources ?? []) {
      const a = assetOfUrl(s.url, RELEASE_BASE_URL);
      if (a?.error) errors.push(`${file.path}: ${s.url}: ${a.error}`);
    }
  }
  for (const r of entry.requires ?? []) {
    const dep = catalog.entries.get(r);
    if (!dep) errors.push(`requires ${r}, which is not in the catalog`);
    else if ((dep.entry.status ?? "published") !== "published" && !selected.has(r)) {
      errors.push(`requires ${r}, which is unpublished and not part of this run (publish it first or add it to --ids), so its download button would not work`);
    }
  }
  return { errors };
}

/**
 * Decide which verified entries may flip: an entry whose required entry fails (or is skipped) stays
 * unpublished. `ok` is Map(id -> boolean) of the entries that passed their own checks.
 */
export function settleRequires(ok, catalog) {
  const result = new Map(ok);
  let changed = true;
  while (changed) {
    changed = false;
    for (const [id, good] of result) {
      if (!good) continue;
      for (const r of catalog.entries.get(id)?.entry.requires ?? []) {
        const dep = catalog.entries.get(r);
        const depPublished = dep && (dep.entry.status ?? "published") === "published";
        if (!depPublished && result.get(r) !== true) {
          result.set(id, false);
          changed = true;
        }
      }
    }
  }
  return result;
}

// ---------------------------------------------------------------------------------------------
// Network
// ---------------------------------------------------------------------------------------------

/**
 * GET `url`, stream it into a temp file under `tempDir` while hashing, compare with the pin of `file`,
 * and delete the temp file. Returns `{ status: "verified" | "failed", detail }`.
 */
export async function verifyByDownload(url, file, ctx) {
  const { throttle, fetchImpl, tempDir, sleep, log } = ctx;
  let res = null;
  let ac = null;
  for (let attempt = 1; ; attempt += 1) {
    await throttle.wait();
    ac = new AbortController();
    try {
      log(`GET ${url}`);
      res = await fetchImpl(url, { headers: { "User-Agent": USER_AGENT, Accept: "application/octet-stream" }, redirect: "follow", signal: ac.signal });
    } catch (e) {
      if (attempt >= 3) return { status: "failed", detail: `${url}: ${e.message}` };
      log(`  network error (${e.message}); retrying`);
      continue;
    }
    if (res.ok) break;
    if (res.status === 404) return { status: "failed", detail: `${url}: HTTP 404 (not uploaded to the release yet, or the release is a draft)` };
    const retryable = res.status === 429 || res.status >= 500;
    if (!retryable || attempt >= 3) return { status: "failed", detail: `${url}: HTTP ${res.status}` };
    const after = Number(res.headers.get("retry-after"));
    const wait = Number.isFinite(after) && after > 0 ? Math.min(after, 120) * 1000 : 10_000 * attempt;
    log(`  HTTP ${res.status}; waiting ${Math.round(wait / 1000)} s`);
    await sleep(wait);
  }
  const type = res.headers.get("content-type") ?? "";
  if (/text\/html/i.test(type)) {
    ac.abort();
    return { status: "failed", detail: `${url}: the server answered with an HTML page, not the file` };
  }
  fs.mkdirSync(tempDir, { recursive: true });
  const tmp = path.join(tempDir, `${crypto.randomBytes(6).toString("hex")}.part`);
  const sha = crypto.createHash("sha256");
  const git = file.git_oid ? crypto.createHash("sha1").update(`blob ${file.size}\0`) : null;
  let size = 0;
  let idle = setTimeout(() => ac.abort(), IDLE_MS);
  const tap = new Transform({
    transform(chunk, _enc, cb) {
      size += chunk.length;
      clearTimeout(idle);
      idle = setTimeout(() => ac.abort(), IDLE_MS);
      if (size > file.size) return cb(new Error(`more than the pinned ${file.size} bytes arrived`));
      sha.update(chunk);
      git?.update(chunk);
      cb(null, chunk);
    },
  });
  try {
    await pipeline(Readable.fromWeb(res.body), tap, fs.createWriteStream(tmp));
  } catch (e) {
    return { status: "failed", detail: `${url}: ${e.message}` };
  } finally {
    clearTimeout(idle);
    fs.rmSync(tmp, { force: true, maxRetries: 5, retryDelay: 100 });
  }
  const why = comparePin(file, { size, sha256: sha.digest("hex"), git_oid: git?.digest("hex") });
  return why ? { status: "failed", detail: `${url}: ${why}` } : { status: "verified", detail: `${formatBytes(size)} downloaded and hashed` };
}

/**
 * Check every file of one entry. `ctx` carries the shared throttle, fetch, temp dir, log, the GitHub
 * metadata client (`net`), its cache, `staged` (Map of staged assets or null) and options.
 * Returns the rows `{ path, status: "verified" | "failed" | "skipped", detail }`.
 */
export async function verifyEntry(entry, ctx) {
  const rows = [];
  for (const file of entry.files ?? []) {
    const ours = [];
    const others = [];
    for (const s of file.sources ?? []) {
      const a = assetOfUrl(s.url, RELEASE_BASE_URL);
      if (a?.name) ours.push({ s, name: a.name });
      else if (!a) others.push(s);
    }
    const hasUpstream = others.some((s) => classifySource(s).immutable);
    let checked = 0;
    if (hasUpstream) {
      let row;
      try {
        row = await verifyFileOnline({ ...file, sources: others }, ctx.net, ctx.cache);
      } catch (e) {
        row = { status: "unverifiable", detail: `lookup failed: ${e.message}`, via: "error" };
      }
      checked += 1;
      rows.push({ path: file.path, status: row.status === "verified" ? "verified" : "failed", detail: `upstream (${row.via}): ${row.detail}` });
    }
    for (const o of ours) {
      // An upstream-backed file needs no release copy; check it only when someone staged it.
      if (hasUpstream && !(ctx.staged && ctx.staged.has(o.name))) {
        rows.push({ path: file.path, status: "skipped", detail: `optional mirror ${o.name} is not staged` });
        continue;
      }
      checked += 1;
      if (ctx.digestOnly) {
        let row;
        try {
          row = await verifyFileOnline({ ...file, sources: [o.s] }, ctx.net, ctx.cache);
        } catch (e) {
          row = { status: "unverifiable", detail: `lookup failed: ${e.message}`, via: "error" };
        }
        rows.push({
          path: file.path,
          status: row.status === "verified" ? "verified" : "failed",
          detail: row.status === "unverifiable" ? `${row.detail}; run without --digest-only to download and hash it` : `release API (${row.via}): ${row.detail}`,
        });
      } else {
        const r = await verifyByDownload(urlUnder(ctx.assetsBase, o.name), file, ctx);
        rows.push({ path: file.path, ...r });
      }
    }
    if (checked === 0 && !rows.some((r) => r.path === file.path && r.status !== "skipped")) {
      rows.push({ path: file.path, status: "failed", detail: "no source could be checked" });
    }
  }
  return rows;
}

// ---------------------------------------------------------------------------------------------
// Run
// ---------------------------------------------------------------------------------------------

/** Everything the CLI does, with the I/O injected. Returns the exit code. */
export async function run(opts, io = { out: console.log, err: console.error }, deps = {}) {
  const catalog = loadCatalog(opts.catalogDir);
  const hardErrors = catalog.problems.filter((p) => p.level === "error");
  if (hardErrors.length > 0) {
    for (const p of hardErrors) io.err(`catalog: ${p.message}`);
    return 1;
  }
  let staged = null;
  let plan = null;
  if (opts.staging) {
    if (!fs.existsSync(opts.staging) || !fs.statSync(opts.staging).isDirectory()) {
      io.err(`publish-entries: ${opts.staging} is not a folder`);
      return 2;
    }
    staged = scanStaging(opts.staging);
    plan = planStaging(collectAssets(catalog), staged);
  }
  const sel = selectIds(opts.ids, catalog, plan);
  if (sel.errors.length > 0) {
    for (const e of sel.errors) io.err(`publish-entries: ${e}`);
    return 2;
  }
  const except = new Set(opts.except);
  for (const x of except) if (!catalog.entries.has(x)) io.err(`publish-entries: --except ${x} is not a catalog id (ignored)`);
  const ids = sel.ids.filter((id) => !except.has(id));
  if (ids.length === 0) {
    io.err(`publish-entries: nothing selected${opts.ids === "all-staged" ? " (no unpublished entry has all its files staged and pinned; run: node scripts/models/assets-v1.mjs plan --staging <dir>)" : ""}`);
    return 1;
  }
  const selected = new Set(ids);

  const throttle = deps.throttle ?? new Throttle(opts.delayMs);
  const fetchImpl = deps.fetch ?? globalThis.fetch;
  const sleep = deps.sleep ?? throttle.sleep;
  const tempRoot = opts.tempDir ?? deps.tempDir ?? fs.mkdtempSync(path.join(os.tmpdir(), "sigil-publish-"));
  const ownsTemp = !opts.tempDir && !deps.tempDir;
  const net = deps.net ?? new FreshReleaseNet({ throttle, cacheDir: null, fetchImpl, log: (m) => io.err(m) });
  const ctx = {
    throttle,
    fetchImpl,
    sleep,
    tempDir: tempRoot,
    log: (m) => io.err(m),
    net,
    cache: { trees: new Map(), releases: new Map(), manifests: new Map() },
    staged,
    assetsBase: opts.assetsBase,
    digestOnly: opts.digestOnly,
  };

  io.out(`publish-entries: ${ids.length} entr${ids.length === 1 ? "y" : "ies"} (${opts.digestOnly ? "release API digests" : "download and hash"}, one request every ${throttle.delayMs / 1000} s)${opts.write ? "" : " [dry run]"}`);
  const ok = new Map();
  const failures = [];
  try {
    for (const id of ids) {
      const { entry } = catalog.entries.get(id);
      const pre = preflight(entry, catalog, selected);
      if (pre.note) {
        io.out(`  skip     ${id}: ${pre.note}`);
        ok.set(id, false);
        continue;
      }
      if (pre.errors.length > 0) {
        for (const e of pre.errors) io.out(`  FAILED   ${id}: ${e}`);
        ok.set(id, false);
        failures.push(id);
        continue;
      }
      const rows = await verifyEntry(entry, ctx);
      const bad = rows.filter((r) => r.status === "failed");
      for (const r of rows) {
        if (r.status !== "verified") io.out(`  ${r.status === "failed" ? "FAILED  " : "note    "} ${id} ${r.path}: ${r.detail}`);
      }
      if (bad.length > 0) {
        ok.set(id, false);
        failures.push(id);
        io.out(`  FAILED   ${id}: ${bad.length} file(s) did not verify`);
      } else {
        ok.set(id, true);
        io.out(`  verified ${id} (${entry.files.length} file(s))`);
      }
    }
  } finally {
    if (ownsTemp) fs.rmSync(tempRoot, { recursive: true, force: true });
  }

  const settled = settleRequires(ok, catalog);
  for (const [id, good] of settled) {
    if (ok.get(id) && !good) {
      io.out(`  HELD     ${id}: a required entry did not verify`);
      failures.push(id);
    }
  }
  const flip = [...settled].filter(([, good]) => good).map(([id]) => id);

  if (flip.length > 0 && opts.write) {
    const byFragment = new Map();
    for (const id of flip) {
      const name = catalog.entries.get(id).fragment;
      if (!byFragment.has(name)) byFragment.set(name, []);
      byFragment.get(name).push(id);
    }
    for (const [name, list] of byFragment) {
      const file = path.join(opts.catalogDir, name);
      let text = fs.readFileSync(file, "utf8");
      for (const id of list) text = editEntry(text, id, { setStatus: "published" });
      JSON.parse(text);
      fs.writeFileSync(file, text);
      io.out(`  wrote ${name}: ${list.join(", ")} -> published`);
    }
    if (!deps.skipPostCheck) {
      const after = checkCatalog(loadCatalog(opts.catalogDir), loadReleaseSet());
      io.out(`catalog check after the flip: ${after.summary.errors} error(s), ${after.summary.warnings} warning(s) (node scripts/release/check-catalog-pins.mjs)`);
      for (const p of after.problems.filter((x) => x.level === "error")) io.out(`  ERROR ${p.rule}: ${p.message}`);
      if (after.summary.errors > 0) return 1;
    }
  } else if (flip.length > 0) {
    io.out(`dry run: ${flip.length} entr${flip.length === 1 ? "y" : "ies"} would become published: ${flip.join(", ")}. Re-run with --write.`);
  }
  if (failures.length > 0) io.out(`not published (${failures.length}): ${[...new Set(failures)].join(", ")}`);
  return failures.length > 0 ? 1 : 0;
}

export async function main(argv = process.argv.slice(2), io = { out: console.log, err: console.error }, deps = {}) {
  let opts;
  try {
    opts = parseArgs(argv);
  } catch (e) {
    io.err(`publish-entries: ${e.message}`);
    return 2;
  }
  if (opts.help) {
    io.out(fs.readFileSync(fileURLToPath(import.meta.url), "utf8").split("*/")[0] + "*/");
    return 0;
  }
  return run(opts, io, deps);
}

if (process.argv[1] && import.meta.url === pathToFileURL(path.resolve(process.argv[1])).href) {
  main().then(
    (code) => {
      // exitCode, not process.exit(): exiting while fetch's keep-alive sockets close trips a libuv assertion on Windows.
      process.exitCode = code;
    },
    (e) => {
      console.error(`publish-entries: ${e.stack ?? e.message}`);
      process.exitCode = 1;
    },
  );
}
