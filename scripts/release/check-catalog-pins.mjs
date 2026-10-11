/**
 * Release catalog pin check (P7-D): proves that what a new Sigil install downloads is pinned.
 *
 * Three layers, cheapest first:
 *
 *  1. Static (offline, no dependencies, runs in CI before the compile). Every `published` entry of
 *     src-tauri/model-catalog/*.json has a size and a sha256 / git_oid on every file (the rule
 *     build.rs applies to release builds), and every source URL is immutable: a Hugging Face commit,
 *     a release tag, a commit of raw.githubusercontent.com, a versioned redistributable. The release
 *     set (scripts/release/release-set.json: the defaults and the light options) is held to more: its
 *     members and everything they require are published, the CPU rows of the STT routing table
 *     are all in it, and a member may require an unpublished entry only when the file says so.
 *  2. `--online` (throttled, sequential, at least 2 s between requests): re-derives size and digest of
 *     every release-set file from the host's own metadata (the Hugging Face tree API, GitHub release
 *     asset digests, the same sources scripts/models/pin-catalog.mjs pins from) and compares them
 *     with the catalog. A mismatch fails the check. Files whose host publishes no digest are listed as
 *     unverifiable (the Vosk archives: alphacephei publishes only md5); `--hash-by-download` hashes
 *     those again by downloading them (one at a time, up to `--max-download-mb`, default 50).
 *  3. `--installed <app data dir>`: compares the install records of a model store that actually
 *     downloaded the entries (`models/**\/installed.json`, written after the store verified every
 *     byte against the catalog's pin) with the catalog today. A match proves the pinned hash is the
 *     hash of the bytes the source served.
 *
 * Usage (repo root):
 *   node scripts/release/check-catalog-pins.mjs                       # static
 *   node scripts/release/check-catalog-pins.mjs --online              # + the defaults, from upstream metadata
 *   node scripts/release/check-catalog-pins.mjs --online --group all  # + the light options
 *   node scripts/release/check-catalog-pins.mjs --online --hash-by-download  # + re-hash files without a published digest
 *   node scripts/release/check-catalog-pins.mjs --installed "%LOCALAPPDATA%\sigil-dev-cache\appdata-p7"
 *   node scripts/release/check-catalog-pins.mjs --report out.json     # machine-readable result
 * Exit code: 0 clean (warnings allowed), 1 errors, 2 usage.
 *
 * Environment: HF_TOKEN (gated repos) and GITHUB_TOKEN are sent as headers by pin-catalog's client,
 * never in URLs.
 */
import fs from "node:fs";
import os from "node:os";
import path from "node:path";
import { fileURLToPath, pathToFileURL } from "node:url";
import { CATALOG_DIR, MIN_DELAY_MS, Net, Throttle, hfTreeFacts, parseGithubReleaseUrl, parseHfUrl } from "../models/pin-catalog.mjs";

const HERE = path.dirname(fileURLToPath(import.meta.url));
export const RELEASE_SET_PATH = path.join(HERE, "release-set.json");

const SHA1_RE = /^[0-9a-f]{40}$/;
const SHA256_RE = /^[0-9a-f]{64}$/;
/** Release tags that name a moving target. */
const MOVING_TAGS = new Set(["latest", "nightly", "main", "master", "dev", "develop", "head", "edge", "canary"]);
/** Hosts whose URL carries a commit (content-addressed). */
const RAW_GITHUB_HOST = "raw.githubusercontent.com";

// ---------------------------------------------------------------------------------------------
// Catalog and release set
// ---------------------------------------------------------------------------------------------

/**
 * Read every fragment of `dir`. Returns `{ fragments, entries, routing, problems }`; `entries` maps
 * id -> `{ entry, fragment }`.
 */
export function loadCatalog(dir = CATALOG_DIR) {
  const fragments = [];
  const entries = new Map();
  const problems = [];
  let routing = null;
  const names = fs
    .readdirSync(dir)
    .filter((n) => n.endsWith(".json") && !n.startsWith("_"))
    .sort();
  for (const name of names) {
    let doc;
    try {
      doc = JSON.parse(fs.readFileSync(path.join(dir, name), "utf8"));
    } catch (e) {
      problems.push({ level: "error", rule: "catalog.parse", id: null, message: `${name}: ${e.message}` });
      continue;
    }
    fragments.push({ name, doc });
    if (doc.routing && typeof doc.routing === "object") routing = { name, ...doc.routing };
    for (const entry of Array.isArray(doc.entries) ? doc.entries : []) {
      if (entries.has(entry.id)) {
        problems.push({ level: "error", rule: "catalog.duplicate", id: entry.id, message: `${name}: duplicate id (also in ${entries.get(entry.id).fragment})` });
        continue;
      }
      entries.set(entry.id, { entry, fragment: name });
    }
  }
  return { fragments, entries, routing, problems };
}

export function loadReleaseSet(file = RELEASE_SET_PATH) {
  const set = JSON.parse(fs.readFileSync(file, "utf8"));
  if (set.schema !== "sigil-release-set/1") throw new Error(`${file}: unknown schema ${set.schema}`);
  return set;
}

/**
 * Expand the release set against the catalog. Returns `{ members, problems }` where `members` maps
 * id -> `{ tier: "defaults" | "light", group }` (a default stays a default when a light group also
 * selects it).
 */
export function expandReleaseSet(set, catalog) {
  const members = new Map();
  const problems = [];
  const add = (tier, group, id) => {
    const had = members.get(id);
    if (!had || (had.tier === "light" && tier === "defaults")) members.set(id, { tier, group });
  };
  for (const tier of ["defaults", "light"]) {
    for (const [group, sel] of Object.entries(set[tier] ?? {})) {
      for (const id of sel.ids ?? []) {
        if (!catalog.entries.has(id)) problems.push({ level: "error", rule: "set.missing", id, message: `${tier}/${group}: ${id} is not in the catalog` });
        else add(tier, group, id);
      }
      for (const prefix of sel.prefixes ?? []) {
        const hits = [...catalog.entries.keys()].filter((id) => id.startsWith(prefix));
        if (hits.length === 0) problems.push({ level: "error", rule: "set.prefix", id: prefix, message: `${tier}/${group}: no catalog id starts with "${prefix}"` });
        for (const id of hits) add(tier, group, id);
      }
    }
  }
  return { members, problems };
}

/** `requires` closure of `ids` (the ids themselves included), following only ids the catalog has. */
export function requiresClosure(ids, catalog) {
  const seen = new Set();
  const stack = [...ids];
  while (stack.length > 0) {
    const id = stack.pop();
    if (seen.has(id)) continue;
    seen.add(id);
    for (const r of catalog.entries.get(id)?.entry.requires ?? []) stack.push(r);
  }
  return seen;
}

/** Ids the CPU speech path of the routing table can plan (models that do not need a GPU, their runtimes, language ID). */
export function routingCpuIds(routing) {
  const ids = new Set();
  if (!routing) return ids;
  if (routing.languageId) ids.add(routing.languageId);
  for (const [id, m] of Object.entries(routing.models ?? {})) {
    if (m.gpu) continue;
    ids.add(id);
    for (const r of m.runtimes ?? []) ids.add(r);
  }
  return ids;
}

// ---------------------------------------------------------------------------------------------
// Static checks
// ---------------------------------------------------------------------------------------------

/** What a source URL pins to. `immutable` is false when the URL can start serving other bytes. */
export function classifySource(src, set = { rollingTags: [] }) {
  let u;
  try {
    u = new URL(src.url);
  } catch {
    return { host: "invalid", immutable: false, reason: "not a URL" };
  }
  if (u.protocol !== "https:") return { host: u.hostname, immutable: false, reason: "not https" };
  const hf = parseHfUrl(src.url);
  if (hf) {
    if (!SHA1_RE.test(hf.revision)) return { host: "huggingface", immutable: false, reason: `revision "${hf.revision}" is a branch or tag, not a commit` };
    if (src.revision && src.revision !== hf.revision) return { host: "huggingface", immutable: false, reason: `revision ${src.revision} differs from the URL's ${hf.revision}` };
    return { host: "huggingface", immutable: true, hf };
  }
  const gh = parseGithubReleaseUrl(src.url);
  if (gh) {
    if (MOVING_TAGS.has(gh.tag.toLowerCase())) return { host: "github-release", immutable: false, reason: `tag "${gh.tag}" is a moving target` };
    const key = `${gh.owner}/${gh.repo}@${gh.tag}`;
    const rolling = (set.rollingTags ?? []).some((r) => r.tag.toLowerCase() === key.toLowerCase());
    return { host: "github-release", immutable: true, gh, rolling, key };
  }
  if (u.hostname === RAW_GITHUB_HOST) {
    const parts = u.pathname.split("/").filter(Boolean);
    if (parts.length < 4 || !SHA1_RE.test(parts[2])) return { host: "github-raw", immutable: false, reason: "not pinned to a commit" };
    return { host: "github-raw", immutable: true };
  }
  if (src.kind === "nvidia-redist") {
    // The redistributable's path names its version (cuda_cudart-windows-x86_64-13.2.51-archive.zip).
    return /\d+\.\d+\.\d+/.test(u.pathname) ? { host: "nvidia-redist", immutable: true } : { host: "nvidia-redist", immutable: false, reason: "no version in the path" };
  }
  if (u.hostname === "firefox-settings-attachments.cdn.mozilla.net") return { host: "mozilla-attachment", immutable: true };
  // Any other host (alphacephei.com, ...) can replace the file under the same URL: only the hash protects.
  return { host: u.hostname, immutable: false, reason: "host does not pin content (the hash is the only protection)", hashOnly: true };
}

/** Problems of one file of an entry. `strict` = release-set rules. */
export function checkFile(entry, file, { strict = false, set = {} } = {}) {
  const out = [];
  const at = `${entry.id} ${file.path}`;
  const published = (entry.status ?? "published") === "published";
  const hasSha = file.sha256 !== undefined;
  const hasOid = file.git_oid !== undefined;
  if (hasSha && !SHA256_RE.test(file.sha256)) out.push({ level: "error", rule: "pin.shape", id: entry.id, message: `${at}: sha256 is not 64 lower-case hex characters` });
  if (hasOid && !SHA1_RE.test(file.git_oid)) out.push({ level: "error", rule: "pin.shape", id: entry.id, message: `${at}: git_oid is not 40 lower-case hex characters` });
  if (hasSha && hasOid) out.push({ level: "error", rule: "pin.both", id: entry.id, message: `${at}: has both sha256 and git_oid` });
  const pinned = (hasSha && SHA256_RE.test(file.sha256)) || (hasOid && SHA1_RE.test(file.git_oid));
  if (published && !pinned) out.push({ level: "error", rule: "pin.missing", id: entry.id, message: `${at}: published entry without a sha256 / git_oid (release builds refuse it)` });
  if (pinned && !(Number.isSafeInteger(file.size) && file.size > 0)) out.push({ level: "error", rule: "pin.size", id: entry.id, message: `${at}: pinned file without a positive size` });
  if (published || pinned) {
    for (const src of file.sources ?? []) {
      const c = classifySource(src, set);
      if (!c.immutable && !c.hashOnly) out.push({ level: "error", rule: "src.mutable", id: entry.id, message: `${at}: ${src.url} is not immutable (${c.reason})` });
      else if (c.hashOnly) {
        const accepted = (set.tofu ?? []).find((t) => entry.id.startsWith(t.prefix));
        if (strict && !accepted) out.push({ level: "error", rule: "src.hash-only", id: entry.id, message: `${at}: ${src.url}: ${c.reason}; list it under "tofu" in release-set.json with the reason, or pin an immutable source` });
        else if (strict) out.push({ level: "note", rule: "src.hash-only", id: entry.id, message: `${at}: hash-only source (accepted: ${accepted.why.split(".")[0]})` });
      }
      if (c.rolling && strict) out.push({ level: "note", rule: "src.rolling", id: entry.id, message: `${at}: ${c.key} is a rolling release (pinned by hash)` });
    }
  }
  return out;
}

/**
 * Every static problem of the catalog and the release set.
 * Returns `{ problems, members, closure, summary }`.
 */
export function checkCatalog(catalog, set) {
  const problems = [...catalog.problems];
  const { members, problems: setProblems } = expandReleaseSet(set, catalog);
  problems.push(...setProblems);

  // 1. Whole catalog: the rules build.rs applies to release builds, plus immutable sources.
  let published = 0;
  for (const [id, { entry }] of catalog.entries) {
    const isPublished = (entry.status ?? "published") === "published";
    if (isPublished) published += 1;
    for (const file of entry.files ?? []) problems.push(...checkFile(entry, file, { strict: false, set }));
    for (const r of entry.requires ?? []) {
      const dep = catalog.entries.get(r);
      if (!dep) problems.push({ level: "error", rule: "req.missing", id, message: `${id} requires ${r}, which is not in the catalog` });
      else if (isPublished && (dep.entry.status ?? "published") !== "published") {
        problems.push({ level: "note", rule: "req.unpublished", id, message: `${id} (published) requires ${r}, which is unpublished (import-only)` });
      }
    }
  }

  // 2. Release set: published, pinned, immutable, the whole closure.
  const allowed = new Map((set.allowUnpublished ?? []).map((a) => [a.id, a.why]));
  const closure = requiresClosure([...members.keys()], catalog);
  const used = new Set();
  for (const id of closure) {
    const hit = catalog.entries.get(id);
    if (!hit) continue; // reported as req.missing above
    const { entry } = hit;
    const member = members.get(id);
    const unpublished = (entry.status ?? "published") !== "published";
    if (unpublished) {
      if (allowed.has(id)) used.add(id);
      else problems.push({ level: "error", rule: "set.unpublished", id, message: `${id} is unpublished but ${member ? "is in the release set" : "is required by a release-set entry"}; publish it or list it under allowUnpublished` });
    }
    for (const file of entry.files ?? []) {
      for (const p of checkFile(entry, file, { strict: true, set })) {
        // The whole-catalog pass above already reported the lenient rules; the strict pass adds only what is new.
        if (!problems.some((q) => q.rule === p.rule && q.message === p.message)) problems.push(p);
      }
    }
  }
  for (const id of allowed.keys()) {
    if (!catalog.entries.has(id)) problems.push({ level: "error", rule: "set.allow", id, message: `allowUnpublished names ${id}, which is not in the catalog` });
    else if (!used.has(id)) problems.push({ level: "warn", rule: "set.allow-unused", id, message: `allowUnpublished lists ${id}, but nothing in the release set needs it any more` });
  }

  // 3. The CPU rows of the STT routing table are all in the defaults.
  for (const id of routingCpuIds(catalog.routing)) {
    const m = members.get(id);
    if (!catalog.entries.has(id)) problems.push({ level: "error", rule: "routing.missing", id, message: `stt-routing.json names ${id}, which is not in the catalog` });
    else if (!m) problems.push({ level: "error", rule: "routing.not-in-set", id, message: `stt-routing.json can plan ${id} on the CPU but it is not in release-set.json` });
    else if (m.tier !== "defaults") problems.push({ level: "error", rule: "routing.not-default", id, message: `${id} is in a CPU row of stt-routing.json but only in the light options of release-set.json` });
  }

  // 4. rollingTags names tags the catalog uses.
  for (const r of set.rollingTags ?? []) {
    const uses = [...catalog.entries.values()].some(({ entry }) => (entry.files ?? []).some((f) => (f.sources ?? []).some((s) => classifySource(s, set).key?.toLowerCase() === r.tag.toLowerCase())));
    if (!uses) problems.push({ level: "warn", rule: "set.rolling-unused", id: null, message: `rollingTags lists ${r.tag}, which no catalog source uses` });
  }

  const count = (level) => problems.filter((p) => p.level === level).length;
  const summary = {
    entries: catalog.entries.size,
    published,
    unpublished: catalog.entries.size - published,
    defaults: [...members.values()].filter((m) => m.tier === "defaults").length,
    light: [...members.values()].filter((m) => m.tier === "light").length,
    closure: closure.size,
    errors: count("error"),
    warnings: count("warn"),
    notes: count("note"),
  };
  return { problems, members, closure, summary };
}

// ---------------------------------------------------------------------------------------------
// Online verification
// ---------------------------------------------------------------------------------------------

/** Pin-catalog's client, but GitHub release lookups are never served from the cache (rolling tags move). */
export class FreshReleaseNet extends Net {
  getJson(url, opts = {}) {
    const fresh = url.startsWith("https://api.github.com/") ? { ...opts, immutable: false } : opts;
    return super.getJson(url, fresh);
  }
}

function hfTreeUrl(hf, commit) {
  const dir = hf.path.includes("/") ? hf.path.slice(0, hf.path.lastIndexOf("/")) : "";
  const tail = dir ? `/${dir.split("/").map(encodeURIComponent).join("/")}` : "";
  return { dir, url: `https://huggingface.co/api/${hf.repoType}/${hf.repo}/tree/${commit}${tail}` };
}

/** `https://raw.githubusercontent.com/<owner>/<repo>/<commit>/<path>` -> its parts, or null. */
export function parseRawGithubUrl(url) {
  let u;
  try {
    u = new URL(url);
  } catch {
    return null;
  }
  if (u.protocol !== "https:" || u.hostname !== RAW_GITHUB_HOST) return null;
  const parts = u.pathname.split("/").filter(Boolean).map(decodeURIComponent);
  if (parts.length < 4 || !SHA1_RE.test(parts[2])) return null;
  return { owner: parts[0], repo: parts[1], commit: parts[2], path: parts.slice(3).join("/") };
}

/** The node of an NVIDIA redistrib manifest whose `relative_path` is `rel` (the manifests nest by component, platform and CUDA variant). */
export function findRedistNode(node, rel) {
  if (!node || typeof node !== "object") return null;
  if (node.relative_path === rel) return node;
  for (const child of Object.values(node)) {
    const hit = findRedistNode(child, rel);
    if (hit) return hit;
  }
  return null;
}

/**
 * Re-derive one file's size and digest from its host. Returns `{ status, detail, via }` with status
 * `verified`, `mismatch` or `unverifiable`.
 */
export async function verifyFileOnline(file, net, cache = { trees: new Map(), releases: new Map(), manifests: new Map() }, { downloadLimit = 0 } = {}) {
  cache.manifests ??= new Map();
  for (const src of file.sources ?? []) {
    const hf = parseHfUrl(src.url);
    if (hf) {
      if (!SHA1_RE.test(hf.revision)) return { status: "mismatch", detail: `revision ${hf.revision} is not a commit`, via: "huggingface" };
      const { dir, url } = hfTreeUrl(hf, hf.revision);
      const key = `${hf.repoType}:${hf.repo}@${hf.revision}/${dir}`;
      if (!cache.trees.has(key)) cache.trees.set(key, hfTreeFacts(await net.getJson(url, { immutable: true })));
      const fact = cache.trees.get(key).get(hf.path);
      if (!fact) return { status: "mismatch", detail: `${hf.path} is not in ${hf.repo}@${hf.revision.slice(0, 12)}`, via: "huggingface" };
      return compareFact(file, fact, "huggingface");
    }
    // NVIDIA publishes a manifest with the sha256 of every redistributable archive (redistrib_<revision>.json).
    const nv = src.kind === "nvidia-redist" && src.revision ? new URL(src.url) : null;
    if (nv && nv.pathname.includes("/redist/")) {
      const at = nv.pathname.indexOf("/redist/") + "/redist/".length;
      const manifestUrl = `${nv.origin}${nv.pathname.slice(0, at)}redistrib_${src.revision}.json`;
      if (!cache.manifests.has(manifestUrl)) cache.manifests.set(manifestUrl, await net.getJson(manifestUrl, { immutable: true }));
      const node = findRedistNode(cache.manifests.get(manifestUrl), decodeURIComponent(nv.pathname.slice(at)));
      if (!node) return { status: "mismatch", detail: `${nv.pathname.slice(at)} is not in redistrib_${src.revision}.json`, via: "nvidia-redist" };
      return compareFact(file, { size: Number(node.size), sha256: node.sha256 }, "nvidia-redist");
    }
    // A file pinned by git blob id at a commit of raw.githubusercontent.com: the contents API reports the blob id.
    const raw = file.git_oid ? parseRawGithubUrl(src.url) : null;
    if (raw) {
      const api = `https://api.github.com/repos/${raw.owner}/${raw.repo}/contents/${raw.path.split("/").map(encodeURIComponent).join("/")}?ref=${raw.commit}`;
      const meta = await net.getJson(api, { immutable: true });
      if (!meta || typeof meta.sha !== "string") return { status: "mismatch", detail: `${raw.path} is not in ${raw.owner}/${raw.repo}@${raw.commit.slice(0, 12)}`, via: "github-raw" };
      return compareFact(file, { size: Number(meta.size), git_oid: meta.sha }, "github-raw");
    }
    const gh = parseGithubReleaseUrl(src.url);
    if (gh) {
      const key = `${gh.owner}/${gh.repo}@${gh.tag}`;
      if (!cache.releases.has(key)) {
        cache.releases.set(key, await net.getJson(`https://api.github.com/repos/${gh.owner}/${gh.repo}/releases/tags/${encodeURIComponent(gh.tag)}`));
      }
      const asset = (cache.releases.get(key)?.assets ?? []).find((a) => a.name === gh.asset);
      if (!asset) return { status: "mismatch", detail: `${gh.asset} is not an asset of ${key} any more`, via: "github-release" };
      const digest = typeof asset.digest === "string" && asset.digest.startsWith("sha256:") ? asset.digest.slice(7) : null;
      if (!digest || !SHA256_RE.test(digest)) {
        if (Number(asset.size) !== file.size) return { status: "mismatch", detail: `size ${file.size} in the catalog, ${asset.size} on the release`, via: "github-release" };
        return (await verifyByDownload(file, net, downloadLimit)) ?? { status: "unverifiable", detail: "the release asset publishes no digest (size matches)", via: "github-release" };
      }
      return compareFact(file, { size: Number(asset.size), sha256: digest }, "github-release");
    }
  }
  return (await verifyByDownload(file, net, downloadLimit)) ?? { status: "unverifiable", detail: "the host publishes no digest this script can read", via: "none" };
}

/**
 * Hash the file again by downloading it (opt-in, sequential, only up to `limit` bytes) when its host publishes no
 * digest. This catches a file the host replaced after the pin; it cannot catch a pin that was wrong from the start
 * (that needs a second, independent source). Returns null when not applicable.
 */
async function verifyByDownload(file, net, limit) {
  const src = (file.sources ?? [])[0];
  if (!(limit > 0) || !file.sha256 || !src || file.size > limit || typeof net.hashByDownload !== "function") return null;
  const got = await net.hashByDownload(src.url);
  if (got.size !== file.size) return { status: "mismatch", detail: `size ${file.size} in the catalog, ${got.size} downloaded`, via: "download" };
  if (got.sha256 !== file.sha256) return { status: "mismatch", detail: `sha256 ${file.sha256} in the catalog, ${got.sha256} downloaded`, via: "download" };
  return { status: "verified", detail: "sha256 re-derived by downloading the file again (its host publishes no digest)", via: "download" };
}

function compareFact(file, fact, via) {
  if (fact.size !== file.size) return { status: "mismatch", detail: `size ${file.size} in the catalog, ${fact.size} at the source`, via };
  if (fact.sha256) {
    if (file.sha256 !== fact.sha256) return { status: "mismatch", detail: `sha256 ${file.sha256 ?? "(none)"} in the catalog, ${fact.sha256} at the source`, via };
  } else if (file.git_oid !== fact.git_oid) {
    return { status: "mismatch", detail: `git_oid ${file.git_oid ?? "(none)"} in the catalog, ${fact.git_oid} at the source`, via };
  }
  return { status: "verified", detail: fact.sha256 ? "sha256" : "git blob id", via };
}

/**
 * Verify every file of `ids` against upstream metadata. `onFile` is called after each file.
 * Returns the rows `{ id, path, status, detail, via }`.
 */
export async function verifyOnline(catalog, ids, net, { onFile = () => {}, downloadLimit = 0, noDownload = () => false } = {}) {
  const cache = { trees: new Map(), releases: new Map(), manifests: new Map() };
  const rows = [];
  for (const id of ids) {
    const hit = catalog.entries.get(id);
    if (!hit || (hit.entry.status ?? "published") !== "published") continue;
    for (const file of hit.entry.files ?? []) {
      let row;
      try {
        row = await verifyFileOnline(file, net, cache, { downloadLimit: noDownload(id) ? 0 : downloadLimit });
      } catch (e) {
        row = { status: "unverifiable", detail: `lookup failed: ${e.message}`, via: "error" };
      }
      const full = { id, path: file.path, ...row };
      rows.push(full);
      onFile(full);
    }
  }
  return rows;
}

// ---------------------------------------------------------------------------------------------
// Install-record evidence
// ---------------------------------------------------------------------------------------------

/** Every `installed.json` under `<appData>/models`, keyed by entry id. */
export function readInstallRecords(appData) {
  const records = new Map();
  const root = path.join(appData, "models");
  if (!fs.existsSync(root)) return records;
  const walk = (dir, depth) => {
    for (const d of fs.readdirSync(dir, { withFileTypes: true })) {
      const p = path.join(dir, d.name);
      if (d.isDirectory()) {
        if (depth < 6) walk(p, depth + 1);
      } else if (d.name === "installed.json") {
        try {
          const rec = JSON.parse(fs.readFileSync(p, "utf8"));
          if (typeof rec.id === "string") records.set(rec.id, rec);
        } catch {
          /* a damaged record is the store's business */
        }
      }
    }
  };
  walk(root, 0);
  return records;
}

/**
 * Compare install records with the catalog. Status per id: `match` (every pinned file equals what
 * the store verified), `mismatch`, `other-version` (the record is of another catalog version) or
 * `absent`.
 */
export function compareInstalled(catalog, records, ids) {
  const rows = [];
  for (const id of ids) {
    const hit = catalog.entries.get(id);
    const rec = records.get(id);
    if (!hit || !rec) {
      rows.push({ id, status: "absent", detail: "" });
      continue;
    }
    if (String(rec.version) !== String(hit.entry.version)) {
      rows.push({ id, status: "other-version", detail: `record ${rec.version}, catalog ${hit.entry.version}`, origin: rec.origin });
      continue;
    }
    const byPath = new Map((rec.files ?? []).map((f) => [f.path, f]));
    const bad = [];
    let compared = 0;
    for (const f of hit.entry.files ?? []) {
      const r = byPath.get(f.path);
      if (!r) {
        bad.push(`${f.path}: not in the record`);
      } else if (f.sha256) {
        compared += 1;
        if (r.sha256 !== f.sha256 || r.size !== f.size) bad.push(`${f.path}: record ${String(r.sha256).slice(0, 12)}/${r.size}, catalog ${f.sha256.slice(0, 12)}/${f.size}`);
      } else if (f.git_oid) {
        compared += 1;
        if (r.size !== f.size || (r.git_oid !== undefined && r.git_oid !== f.git_oid)) bad.push(`${f.path}: record ${r.size} bytes, catalog ${f.size}`);
      }
    }
    rows.push(bad.length > 0 ? { id, status: "mismatch", detail: bad.join("; "), origin: rec.origin } : { id, status: "match", detail: `${compared} file(s)`, origin: rec.origin });
  }
  return rows;
}

// ---------------------------------------------------------------------------------------------
// CLI
// ---------------------------------------------------------------------------------------------

export function parseArgs(argv) {
  const opts = { online: false, group: "defaults", installed: null, report: null, catalogDir: CATALOG_DIR, set: RELEASE_SET_PATH, delayMs: 2500, cacheDir: null, quiet: false, strictOnline: false, hashByDownload: false, maxDownloadMb: 50, help: false };
  for (let i = 0; i < argv.length; i += 1) {
    const a = argv[i];
    const val = () => {
      if (i + 1 >= argv.length) throw new Error(`${a} needs a value`);
      return argv[++i];
    };
    if (a === "--online") opts.online = true;
    else if (a === "--group") opts.group = val();
    else if (a === "--installed") opts.installed = val();
    else if (a === "--report") opts.report = val();
    else if (a === "--catalog-dir") opts.catalogDir = val();
    else if (a === "--release-set") opts.set = val();
    else if (a === "--delay-ms") opts.delayMs = Number(val());
    else if (a === "--cache-dir") opts.cacheDir = val();
    else if (a === "--strict-online") opts.strictOnline = true;
    else if (a === "--hash-by-download") opts.hashByDownload = true;
    else if (a === "--max-download-mb") opts.maxDownloadMb = Number(val());
    else if (a === "--quiet") opts.quiet = true;
    else if (a === "--help" || a === "-h") opts.help = true;
    else throw new Error(`unknown argument ${a}`);
  }
  if (!["defaults", "light", "all"].includes(opts.group)) throw new Error(`--group must be defaults, light or all, got ${opts.group}`);
  if (!Number.isFinite(opts.maxDownloadMb) || opts.maxDownloadMb <= 0) throw new Error("--max-download-mb must be a positive number");
  if (!Number.isFinite(opts.delayMs) || opts.delayMs < MIN_DELAY_MS) throw new Error(`--delay-ms must be at least ${MIN_DELAY_MS} (the dev network blocks request bursts)`);
  return opts;
}

function defaultCacheDir() {
  const base = process.env.LOCALAPPDATA || path.join(os.homedir(), ".cache");
  return path.join(base, "sigil-dev-cache", "pin-catalog");
}

/** Release-set ids of `group` ("defaults", "light" or "all"). */
export function idsForGroup(members, group) {
  return [...members.entries()].filter(([, m]) => group === "all" || m.tier === group).map(([id]) => id);
}

export async function main(argv = process.argv.slice(2), io = { out: console.log, err: console.error }, deps = {}) {
  let opts;
  try {
    opts = parseArgs(argv);
  } catch (e) {
    io.err(`check-catalog-pins: ${e.message}`);
    return 2;
  }
  if (opts.help) {
    const text = fs.readFileSync(fileURLToPath(import.meta.url), "utf8");
    io.out(text.slice(0, text.indexOf("*/") + 2));
    return 0;
  }
  const catalog = loadCatalog(opts.catalogDir);
  const set = loadReleaseSet(opts.set);
  const result = checkCatalog(catalog, set);
  const report = { schema: "sigil-pin-check/1", generated_at: new Date().toISOString(), summary: result.summary, problems: result.problems, online: null, installed: null };

  const s = result.summary;
  io.out(`catalog: ${s.entries} entries (${s.published} published, ${s.unpublished} unpublished); release set: ${s.defaults} defaults + ${s.light} light options, ${s.closure} entries with what they require`);
  for (const p of result.problems.filter((x) => x.level === "error")) io.out(`  ERROR ${p.rule}: ${p.message}`);
  for (const p of result.problems.filter((x) => x.level === "warn")) io.out(`  warn  ${p.rule}: ${p.message}`);
  if (!opts.quiet) {
    const grouped = new Map();
    for (const p of result.problems.filter((x) => x.level === "note")) {
      const k = p.rule;
      if (!grouped.has(k)) grouped.set(k, []);
      grouped.get(k).push(p);
    }
    for (const [rule, list] of grouped) io.out(`  note  ${rule}: ${list.length} (${list.slice(0, 3).map((p) => p.id).join(", ")}${list.length > 3 ? ", ..." : ""})`);
  }
  let failed = s.errors > 0;

  if (opts.installed) {
    const records = readInstallRecords(opts.installed);
    const rows = compareInstalled(catalog, records, [...result.closure]);
    report.installed = { dir: opts.installed, records: records.size, rows };
    const tally = rows.reduce((a, r) => ((a[r.status] = (a[r.status] ?? 0) + 1), a), {});
    io.out(`install records in ${opts.installed}: ${records.size} found; release set and its requirements: ${JSON.stringify(tally)}`);
    for (const r of rows.filter((x) => x.status === "mismatch")) io.out(`  ERROR installed.mismatch: ${r.id}: ${r.detail}`);
    for (const r of rows.filter((x) => x.status === "other-version")) io.out(`  warn  installed.other-version: ${r.id}: ${r.detail}`);
    if (tally.mismatch) failed = true;
  }

  if (opts.online) {
    const ids = [...requiresClosure(idsForGroup(result.members, opts.group), catalog)];
    const net = deps.net ?? new FreshReleaseNet({ throttle: new Throttle(opts.delayMs), cacheDir: opts.cacheDir ?? defaultCacheDir(), log: (m) => io.err(m) });
    io.out(`online: verifying ${ids.length} entries (${opts.group}) against their hosts, one request every ${opts.delayMs / 1000} s ...`);
    const downloadLimit = opts.hashByDownload ? opts.maxDownloadMb * 1e6 : 0;
    const rows = await verifyOnline(catalog, ids, net, {
      downloadLimit,
      // Entries attested elsewhere (the Firefox Translations models: 322 files) are never downloaded here.
      noDownload: (id) => (set.attested ?? []).some((t) => id.startsWith(t.prefix)),
      onFile: (r) => r.status === "mismatch" && io.out(`  ERROR online.mismatch: ${r.id} ${r.path}: ${r.detail}`),
    });
    const tally = rows.reduce((a, r) => ((a[r.status] = (a[r.status] ?? 0) + 1), a), {});
    const covered = (r) => [...(set.tofu ?? []), ...(set.attested ?? [])].find((t) => r.id.startsWith(t.prefix));
    const unverifiable = rows.filter((r) => r.status === "unverifiable");
    const unlisted = unverifiable.filter((r) => !covered(r));
    report.online = { group: opts.group, requests: net.requests, tally, rows };
    io.out(`online: ${rows.length} files: ${JSON.stringify(tally)}, ${net.requests} request(s)`);
    // One line per kind of unverifiable file, not one per file (the Firefox Translations models alone are 322).
    const kinds = new Map();
    for (const r of unverifiable) {
      const c = covered(r);
      const key = `${c ? "listed" : "UNLISTED"} | ${r.detail}`;
      if (!kinds.has(key)) kinds.set(key, { n: 0, example: r, by: c ? (c.why ?? c.by) : "" });
      kinds.get(key).n += 1;
    }
    for (const [key, k] of kinds) io.out(`  ${key.startsWith("listed") ? "note " : "warn "} online.unverifiable: ${k.n} file(s), e.g. ${k.example.id} ${k.example.path}: ${k.example.detail}${k.by ? ` [${k.by.split(". ")[0]}]` : ""}`);
    if (tally.mismatch) failed = true;
    if (unlisted.length > 0 && opts.strictOnline) failed = true;
  }

  if (opts.report) fs.writeFileSync(opts.report, `${JSON.stringify(report, null, 2)}\n`);
  io.out(failed ? "FAILED" : "OK");
  return failed ? 1 : 0;
}

if (process.argv[1] && import.meta.url === pathToFileURL(path.resolve(process.argv[1])).href) {
  main().then(
    (code) => {
      // exitCode, not process.exit(): exiting while fetch's keep-alive sockets close trips a libuv assertion on Windows.
      process.exitCode = code;
    },
    (e) => {
      console.error(`check-catalog-pins: ${e.stack ?? e.message}`);
      process.exitCode = 1;
    },
  );
}
