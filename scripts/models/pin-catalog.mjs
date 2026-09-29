/**
 * Pin the model catalog: fill in `size` and `sha256` (or `git_oid` for small non-LFS files) of every
 * file in src-tauri/model-catalog/*.json and rewrite Hugging Face URLs from a branch (`main`) to the
 * immutable commit they resolve to. Release builds refuse `published` entries with an unpinned file
 * (build.rs), so run this after adding or updating entries.
 *
 * Where the facts come from (metadata only; nothing is downloaded unless you ask for it):
 * - huggingface / hf-mirror URLs (`https://huggingface.co/<repo>/resolve/<rev>/<path>`, datasets
 *   too): the Hub tree API (`/api/models/<repo>/tree/<commit>/<dir>`). LFS files get the LFS
 *   sha256, plain git files their git blob id. One request per repo directory, plus one per branch
 *   to resolve it to a commit.
 * - github-release URLs (`https://github.com/<o>/<r>/releases/download/<tag>/<name>`): the GitHub
 *   release API (`assets[].digest`, published for assets uploaded since mid 2025).
 * - anything else (direct, nvidia-redist, modelscope, custom): only with `--hash-by-download`, which
 *   streams the file once through sha256 without writing it to disk.
 *
 * Network gentleness (the dev PC's router blocks request bursts): every request is sequential with
 * at least 2 s between request starts (`--delay-ms`, minimum 2000), 429/5xx back off, and API
 * responses for immutable revisions are cached on disk (`--cache-dir`), so re-runs cost nothing.
 *
 * Usage (repo root):
 *   node scripts/models/pin-catalog.mjs                      # dry run over every fragment
 *   node scripts/models/pin-catalog.mjs --write              # update the fragments in place
 *   node scripts/models/pin-catalog.mjs --fragment src-tauri/model-catalog/stt-legacy.json --write
 *   node scripts/models/pin-catalog.mjs --only whisper/ --force --write
 *   node scripts/models/pin-catalog.mjs --hash-by-download --only vosk/ --write
 * Options:
 *   --fragment <file>     a fragment to process (repeatable; default: every non-`_` *.json)
 *   --write               write changes (default: print them and exit 1 if anything is unpinned)
 *   --only <prefix>       only entries whose id starts with <prefix> (repeatable)
 *   --force               re-pin files that already have a hash
 *   --hash-by-download    hash files that have no metadata source by streaming them (sequential)
 *   --delay-ms <n>        spacing between request starts (default 2500, minimum 2000)
 *   --cache-dir <dir>     API response cache (default %LOCALAPPDATA%/sigil-dev-cache/pin-catalog)
 *   --no-cache            do not read or write the cache
 * Environment: HF_TOKEN (gated Hub repos) and GITHUB_TOKEN are sent as headers, never in URLs.
 */
import crypto from "node:crypto";
import fs from "node:fs";
import os from "node:os";
import path from "node:path";
import { fileURLToPath, pathToFileURL } from "node:url";

const __dirname = path.dirname(fileURLToPath(import.meta.url));
export const ROOT = path.join(__dirname, "..", "..");
export const CATALOG_DIR = path.join(ROOT, "src-tauri", "model-catalog");
export const MIN_DELAY_MS = 2000;
const USER_AGENT = "Sigil-pin-catalog (https://github.com/Mewnah/Sigil)";

const SHA1_RE = /^[0-9a-f]{40}$/;
const SHA256_RE = /^[0-9a-f]{64}$/;

// ---------------------------------------------------------------------------------------------
// URL parsing (pure)
// ---------------------------------------------------------------------------------------------

/**
 * Parse a Hub file URL (huggingface.co or hf-mirror.com).
 * Returns `{ host, repoType: "models"|"datasets"|"spaces", repo, revision, path }` or null.
 */
export function parseHfUrl(url) {
  let u;
  try {
    u = new URL(url);
  } catch {
    return null;
  }
  if (u.protocol !== "https:" || !["huggingface.co", "hf-mirror.com"].includes(u.hostname)) return null;
  const parts = u.pathname.split("/").filter(Boolean).map(decodeURIComponent);
  let repoType = "models";
  if (parts[0] === "datasets" || parts[0] === "spaces") repoType = parts.shift();
  // <org>/<name>/resolve/<revision>/<path...>
  if (parts.length < 5 || parts[2] !== "resolve") return null;
  const [org, name, , revision, ...rest] = parts;
  if (!org || !name || !revision || rest.length === 0) return null;
  return { host: u.hostname, repoType: repoType === "models" ? "models" : repoType, repo: `${org}/${name}`, revision, path: rest.join("/") };
}

/** Build a Hub file URL for `host` from parsed parts (path segments are percent-encoded). */
export function buildHfUrl({ host, repoType, repo, revision, path: filePath }) {
  const prefix = repoType === "models" ? "" : `${repoType}/`;
  const enc = (p) => p.split("/").map(encodeURIComponent).join("/");
  return `https://${host}/${prefix}${enc(repo)}/resolve/${encodeURIComponent(revision)}/${enc(filePath)}`;
}

/** Parse `https://github.com/<owner>/<repo>/releases/download/<tag>/<asset>`. */
export function parseGithubReleaseUrl(url) {
  let u;
  try {
    u = new URL(url);
  } catch {
    return null;
  }
  if (u.protocol !== "https:" || u.hostname !== "github.com") return null;
  const parts = u.pathname.split("/").filter(Boolean).map(decodeURIComponent);
  if (parts.length !== 6 || parts[2] !== "releases" || parts[3] !== "download") return null;
  return { owner: parts[0], repo: parts[1], tag: parts[4], asset: parts[5] };
}

// ---------------------------------------------------------------------------------------------
// Applying facts to a fragment (pure)
// ---------------------------------------------------------------------------------------------

/** True when the file already carries a hash and a plausible size. */
export function isPinned(file) {
  return Boolean(file.sha256 || file.git_oid);
}

/**
 * Hub tree items (`[{type, path, size, oid, lfs?: {oid, size}}]`) → Map(path → fact) where a fact is
 * `{ size, sha256 }` (LFS) or `{ size, git_oid }` (plain git file).
 */
export function hfTreeFacts(items) {
  const facts = new Map();
  for (const it of items) {
    if (!it || it.type !== "file" || typeof it.path !== "string") continue;
    if (it.lfs && typeof it.lfs.oid === "string" && SHA256_RE.test(it.lfs.oid)) {
      facts.set(it.path, { size: Number(it.lfs.size ?? it.size), sha256: it.lfs.oid });
    } else if (typeof it.oid === "string" && SHA1_RE.test(it.oid)) {
      facts.set(it.path, { size: Number(it.size), git_oid: it.oid });
    }
  }
  return facts;
}

/**
 * Apply one fact to a catalog file (mutates it; keeps key order stable: path, size, hash, sources,
 * archive). Returns a list of human-readable changes.
 */
export function applyFact(file, fact) {
  const changes = [];
  if (!Number.isSafeInteger(fact.size) || fact.size < 0) throw new Error(`invalid size for ${file.path}`);
  if (file.size !== fact.size) changes.push(`size ${file.size} -> ${fact.size}`);
  const hashKey = fact.sha256 ? "sha256" : "git_oid";
  const hash = fact.sha256 ?? fact.git_oid;
  if (file[hashKey] !== hash) changes.push(`${hashKey} ${file[hashKey] ?? "(none)"} -> ${hash}`);
  const other = hashKey === "sha256" ? "git_oid" : "sha256";
  if (other in file) changes.push(`${other} removed`);
  // Rebuild so the order stays path, size, hash, then everything else as it was.
  const rebuilt = { path: file.path, size: fact.size, [hashKey]: hash };
  for (const [k, v] of Object.entries(file)) {
    if (!(k in rebuilt) && k !== other) rebuilt[k] = v;
  }
  for (const k of Object.keys(file)) delete file[k];
  Object.assign(file, rebuilt);
  return changes;
}

/**
 * Rewrite Hub sources of `file` that point at `repo`@`branch` to `commit`, and set `revision`.
 * Returns the number of rewritten sources.
 */
export function pinHfRevision(file, repoType, repo, branch, commit) {
  let n = 0;
  for (const src of file.sources ?? []) {
    const p = parseHfUrl(src.url);
    if (!p || p.repoType !== repoType || p.repo !== repo) continue;
    if (p.revision !== branch && p.revision !== commit) continue;
    const url = buildHfUrl({ ...p, revision: commit });
    if (src.url !== url || src.revision !== commit) {
      src.url = url;
      src.revision = commit;
      n += 1;
    }
  }
  return n;
}

/** Serialize a fragment the way the repo stores it (2-space indent, trailing newline). */
export function serializeFragment(doc) {
  return `${JSON.stringify(doc, null, 2)}\n`;
}

/** Entries of `doc` selected by `only` prefixes (all when empty). */
export function selectEntries(doc, only) {
  const entries = Array.isArray(doc.entries) ? doc.entries : [];
  return only.length === 0 ? entries : entries.filter((e) => only.some((p) => String(e.id).startsWith(p)));
}

// ---------------------------------------------------------------------------------------------
// Throttled, cached network access
// ---------------------------------------------------------------------------------------------

export class Throttle {
  constructor(delayMs, now = () => Date.now(), sleep = (ms) => new Promise((r) => setTimeout(r, ms))) {
    this.delayMs = Math.max(MIN_DELAY_MS, delayMs);
    this.now = now;
    this.sleep = sleep;
    this.last = -Infinity;
  }
  /** Wait until at least `delayMs` passed since the previous request start. */
  async wait() {
    const due = this.last + this.delayMs;
    const t = this.now();
    if (t < due) await this.sleep(due - t);
    this.last = this.now();
  }
}

function defaultCacheDir() {
  const base = process.env.LOCALAPPDATA || path.join(os.homedir(), ".cache");
  return path.join(base, "sigil-dev-cache", "pin-catalog");
}

function cacheKey(url) {
  return crypto.createHash("sha256").update(url).digest("hex").slice(0, 40);
}

function authHeaders(url) {
  const h = { "User-Agent": USER_AGENT, Accept: "application/json" };
  const host = new URL(url).hostname;
  if (host === "huggingface.co" && process.env.HF_TOKEN) h.Authorization = `Bearer ${process.env.HF_TOKEN}`;
  if (host === "api.github.com" && process.env.GITHUB_TOKEN) h.Authorization = `Bearer ${process.env.GITHUB_TOKEN}`;
  return h;
}

export class Net {
  constructor({ throttle, cacheDir, fetchImpl = globalThis.fetch, log = console.error }) {
    this.throttle = throttle;
    this.cacheDir = cacheDir;
    this.fetch = fetchImpl;
    this.log = log;
    this.requests = 0;
  }

  /** GET JSON. `immutable` responses (pinned revisions) are cached. Follows `Link: rel="next"`. */
  async getJson(url, { immutable = false } = {}) {
    const cachePath = this.cacheDir ? path.join(this.cacheDir, `${cacheKey(url)}.json`) : null;
    if (immutable && cachePath && fs.existsSync(cachePath)) {
      return JSON.parse(fs.readFileSync(cachePath, "utf8")).body;
    }
    let body = null;
    let next = url;
    while (next) {
      const res = await this.#request(next);
      const page = await res.json();
      body = body === null ? page : Array.isArray(body) && Array.isArray(page) ? body.concat(page) : page;
      next = nextLink(res.headers.get("link"));
    }
    if (immutable && cachePath) {
      fs.mkdirSync(this.cacheDir, { recursive: true });
      fs.writeFileSync(cachePath, JSON.stringify({ url, body }));
    }
    return body;
  }

  /** Stream `url` through sha256 (nothing written to disk). */
  async hashByDownload(url) {
    const res = await this.#request(url, { accept: "application/octet-stream" });
    const hash = crypto.createHash("sha256");
    let size = 0;
    for await (const chunk of res.body) {
      hash.update(chunk);
      size += chunk.length;
    }
    const declared = Number(res.headers.get("content-length"));
    if (Number.isFinite(declared) && declared > 0 && declared !== size) {
      throw new Error(`${url}: body ended at ${size} of ${declared} bytes`);
    }
    return { size, sha256: hash.digest("hex") };
  }

  async #request(url, { accept } = {}) {
    for (let attempt = 1; ; attempt += 1) {
      await this.throttle.wait();
      this.requests += 1;
      const headers = authHeaders(url);
      if (accept) headers.Accept = accept;
      let res;
      try {
        res = await this.fetch(url, { headers, redirect: "follow" });
      } catch (e) {
        if (attempt >= 3) throw new Error(`${url}: ${e.message}`);
        this.log(`  network error (${e.message}); retrying`);
        continue;
      }
      if (res.ok) return res;
      const retryable = res.status === 429 || res.status >= 500;
      if (!retryable || attempt >= 3) {
        throw new Error(`${url}: HTTP ${res.status}`);
      }
      const after = Number(res.headers.get("retry-after"));
      const wait = Number.isFinite(after) && after > 0 ? Math.min(after, 120) * 1000 : 10_000 * attempt;
      this.log(`  HTTP ${res.status}; waiting ${Math.round(wait / 1000)} s`);
      await this.throttle.sleep(wait);
    }
  }
}

/** `<https://...>; rel="next"` → URL, else null. */
export function nextLink(link) {
  if (!link) return null;
  for (const part of link.split(",")) {
    const m = part.match(/<([^>]+)>\s*;\s*rel="?next"?/);
    if (m) return m[1];
  }
  return null;
}

// ---------------------------------------------------------------------------------------------
// Pinning
// ---------------------------------------------------------------------------------------------

/**
 * Pin every selected file of `docs` (array of { name, doc }). Returns { changes, unpinned } where
 * `changes` is a list of strings and `unpinned` the files that are still without a hash.
 */
export async function pinFragments(docs, net, opts) {
  const { only = [], force = false, hashByDownload = false, log = console.error } = opts;
  const changes = [];
  const pending = [];
  for (const { name, doc } of docs) {
    for (const entry of selectEntries(doc, only)) {
      for (const file of entry.files ?? []) {
        if (!force && isPinned(file) && file.size > 0 && !hasBranchHfSource(file)) continue;
        pending.push({ name, entry, file });
      }
    }
  }

  // 1. Hugging Face: resolve branches, then one tree listing per (repo, commit, dir).
  const branchCache = new Map();
  const treeCache = new Map();
  for (const item of pending) {
    const src = (item.file.sources ?? []).map((s) => parseHfUrl(s.url)).find(Boolean);
    if (!src) continue;
    const repoKey = `${src.repoType}:${src.repo}`;
    let commit = src.revision;
    if (!SHA1_RE.test(commit)) {
      const bk = `${repoKey}@${commit}`;
      if (!branchCache.has(bk)) {
        log(`resolve ${src.repo}@${commit}`);
        const info = await net.getJson(`https://huggingface.co/api/${src.repoType}/${src.repo}/revision/${encodeURIComponent(commit)}`);
        if (!info || !SHA1_RE.test(info.sha)) throw new Error(`${src.repo}@${commit}: no commit sha in the Hub reply`);
        branchCache.set(bk, info.sha);
      }
      const sha = branchCache.get(bk);
      if (pinHfRevision(item.file, src.repoType, src.repo, commit, sha) > 0) {
        changes.push(`${item.name} ${item.entry.id} ${item.file.path}: revision ${commit} -> ${sha}`);
      }
      commit = sha;
    } else {
      pinHfRevision(item.file, src.repoType, src.repo, commit, commit);
    }
    const dir = src.path.includes("/") ? src.path.slice(0, src.path.lastIndexOf("/")) : "";
    const tk = `${repoKey}@${commit}/${dir}`;
    if (!treeCache.has(tk)) {
      log(`tree ${src.repo}@${commit.slice(0, 12)}/${dir}`);
      const url = `https://huggingface.co/api/${src.repoType}/${src.repo}/tree/${commit}${dir ? `/${dir.split("/").map(encodeURIComponent).join("/")}` : ""}`;
      treeCache.set(tk, hfTreeFacts(await net.getJson(url, { immutable: true })));
    }
    const fact = treeCache.get(tk).get(src.path);
    if (!fact) {
      log(`  ! ${item.entry.id} ${item.file.path}: ${src.path} is not in ${src.repo}@${commit.slice(0, 12)}`);
      continue;
    }
    for (const c of applyFact(item.file, fact)) changes.push(`${item.name} ${item.entry.id} ${item.file.path}: ${c}`);
    item.done = true;
  }

  // 2. GitHub release assets with a published digest.
  const releaseCache = new Map();
  for (const item of pending) {
    if (item.done) continue;
    const gh = (item.file.sources ?? []).map((s) => parseGithubReleaseUrl(s.url)).find(Boolean);
    if (!gh) continue;
    const rk = `${gh.owner}/${gh.repo}@${gh.tag}`;
    if (!releaseCache.has(rk)) {
      log(`release ${rk}`);
      releaseCache.set(
        rk,
        await net.getJson(`https://api.github.com/repos/${gh.owner}/${gh.repo}/releases/tags/${encodeURIComponent(gh.tag)}`, { immutable: true }),
      );
    }
    const asset = (releaseCache.get(rk)?.assets ?? []).find((a) => a.name === gh.asset);
    const digest = typeof asset?.digest === "string" && asset.digest.startsWith("sha256:") ? asset.digest.slice(7) : null;
    if (asset && digest && SHA256_RE.test(digest)) {
      for (const c of applyFact(item.file, { size: Number(asset.size), sha256: digest })) {
        changes.push(`${item.name} ${item.entry.id} ${item.file.path}: ${c}`);
      }
      item.done = true;
    }
  }

  // 3. Everything else: stream once through sha256 (opt-in).
  for (const item of pending) {
    if (item.done) continue;
    if (!hashByDownload) continue;
    const url = item.file.sources?.[0]?.url;
    if (!url) continue;
    log(`hash-by-download ${item.entry.id} ${item.file.path} (${url})`);
    const fact = await net.hashByDownload(url);
    for (const c of applyFact(item.file, fact)) changes.push(`${item.name} ${item.entry.id} ${item.file.path}: ${c}`);
    item.done = true;
  }

  const unpinned = [];
  for (const { name, doc } of docs) {
    for (const entry of selectEntries(doc, only)) {
      for (const file of entry.files ?? []) {
        if (!isPinned(file)) unpinned.push(`${name} ${entry.id} ${file.path}`);
      }
    }
  }
  return { changes, unpinned };
}

function hasBranchHfSource(file) {
  return (file.sources ?? []).some((s) => {
    const p = parseHfUrl(s.url);
    return p && !SHA1_RE.test(p.revision);
  });
}

// ---------------------------------------------------------------------------------------------
// CLI
// ---------------------------------------------------------------------------------------------

export function parseArgs(argv) {
  const opts = { fragments: [], write: false, only: [], force: false, hashByDownload: false, delayMs: 2500, cacheDir: defaultCacheDir(), help: false };
  for (let i = 0; i < argv.length; i += 1) {
    const a = argv[i];
    const val = () => {
      if (i + 1 >= argv.length) throw new Error(`${a} needs a value`);
      return argv[++i];
    };
    if (a === "--fragment") opts.fragments.push(val());
    else if (a === "--write") opts.write = true;
    else if (a === "--only") opts.only.push(val());
    else if (a === "--force") opts.force = true;
    else if (a === "--hash-by-download") opts.hashByDownload = true;
    else if (a === "--delay-ms") opts.delayMs = Number(val());
    else if (a === "--cache-dir") opts.cacheDir = val();
    else if (a === "--no-cache") opts.cacheDir = null;
    else if (a === "--help" || a === "-h") opts.help = true;
    else throw new Error(`unknown argument ${a}`);
  }
  if (!Number.isFinite(opts.delayMs) || opts.delayMs < MIN_DELAY_MS) {
    throw new Error(`--delay-ms must be at least ${MIN_DELAY_MS} (the dev network blocks request bursts)`);
  }
  return opts;
}

function defaultFragments() {
  return fs
    .readdirSync(CATALOG_DIR)
    .filter((n) => n.endsWith(".json") && !n.startsWith("_"))
    .sort()
    .map((n) => path.join(CATALOG_DIR, n));
}

async function main() {
  const opts = parseArgs(process.argv.slice(2));
  if (opts.help) {
    const text = fs.readFileSync(fileURLToPath(import.meta.url), "utf8");
    console.log(text.slice(0, text.indexOf("*/") + 2));
    return 0;
  }
  const files = opts.fragments.length > 0 ? opts.fragments.map((f) => path.resolve(f)) : defaultFragments();
  const docs = files.map((f) => ({ name: path.relative(ROOT, f), file: f, doc: JSON.parse(fs.readFileSync(f, "utf8")) }));
  const net = new Net({ throttle: new Throttle(opts.delayMs), cacheDir: opts.cacheDir });
  const { changes, unpinned } = await pinFragments(docs, net, opts);
  for (const c of changes) console.log(c);
  console.error(`${changes.length} change(s), ${net.requests} request(s).`);
  if (opts.write) {
    for (const d of docs) {
      const text = serializeFragment(d.doc);
      if (fs.readFileSync(d.file, "utf8") !== text) {
        fs.writeFileSync(d.file, text);
        console.error(`wrote ${d.name}`);
      }
    }
  }
  if (unpinned.length > 0) {
    console.error(`still unpinned (release builds refuse published entries with these):\n  ${unpinned.join("\n  ")}`);
    return opts.write ? 0 : 1;
  }
  return 0;
}

if (process.argv[1] && import.meta.url === pathToFileURL(process.argv[1]).href) {
  // exitCode, not process.exit(): exiting while fetch's keep-alive sockets close trips a libuv
  // assertion in Node on Windows.
  main().then(
    (code) => {
      process.exitCode = code;
    },
    (e) => {
      console.error(`pin-catalog: ${e.message}`);
      process.exitCode = 1;
    },
  );
}
