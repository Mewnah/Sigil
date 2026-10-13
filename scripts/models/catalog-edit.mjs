/**
 * Format-preserving edits of a catalog fragment's text.
 *
 * Several fragments are written by hand or by a generator in their own layout (arrays on one line,
 * `"archive": { "format": "zip" }` inline), so re-serializing them with JSON.stringify (what
 * pin-catalog.mjs does) would rewrite lines nobody touched. The edits below change only the bytes
 * they mean to:
 *
 * - `setStatus`:  an entry's `"status"` value (`unpublished` -> `published`).
 * - `pinFile`:    a file's `size` and `sha256` / `git_oid`, in the key order pin-catalog writes
 *                 (path, size, hash, sources, archive), whatever whitespace the fragment uses.
 *
 * `editEntry(text, id, edit)` returns the new text. Callers compare `JSON.parse` of the result with
 * the object they meant to produce (see `pin-staged.mjs`), so a layout this module does not
 * understand fails loudly instead of corrupting a fragment.
 */

const SHA256_RE = /^[0-9a-f]{64}$/;
const SHA1_RE = /^[0-9a-f]{40}$/;

function escapeRe(s) {
  return s.replace(/[.*+?^${}()|[\]\\]/g, "\\$&");
}

/** Index of the `}` that closes the object opening at `text[start]` (string- and escape-aware). */
export function findObjectEnd(text, start) {
  if (text[start] !== "{") throw new Error("findObjectEnd: not at an opening brace");
  let depth = 0;
  let inString = false;
  for (let i = start; i < text.length; i += 1) {
    const c = text[i];
    if (inString) {
      if (c === "\\") i += 1;
      else if (c === '"') inString = false;
    } else if (c === '"') inString = true;
    else if (c === "{") depth += 1;
    else if (c === "}") {
      depth -= 1;
      if (depth === 0) return i;
    }
  }
  throw new Error("findObjectEnd: unbalanced braces");
}

/** The nearest `{` before `index` that opens an object containing it (the key is the object's first). */
function openingBrace(text, index) {
  const at = text.lastIndexOf("{", index);
  if (at < 0) throw new Error("no enclosing object");
  return at;
}

/** `{ start, end }` (end exclusive) of the entry object whose `"id"` is `id`. Exactly one must match. */
export function findEntry(text, id) {
  const re = new RegExp(`"id"\\s*:\\s*"${escapeRe(id)}"`, "g");
  const hits = [...text.matchAll(re)];
  if (hits.length !== 1) throw new Error(`entry ${id}: found ${hits.length} times in the fragment (expected 1)`);
  const start = openingBrace(text, hits[0].index);
  const end = findObjectEnd(text, start) + 1;
  const obj = JSON.parse(text.slice(start, end));
  if (obj.id !== id) throw new Error(`entry ${id}: the object around its id is ${obj.id} (the id must be the entry's first key)`);
  return { start, end };
}

/** `{ start, end }` of the file object with `"path": path` inside `text[from, to)`. */
function findFile(text, from, to, path) {
  const slice = text.slice(from, to);
  const re = new RegExp(`"path"\\s*:\\s*"${escapeRe(path)}"`, "g");
  const hits = [...slice.matchAll(re)];
  if (hits.length !== 1) throw new Error(`file ${path}: found ${hits.length} times in the entry (expected 1)`);
  const start = openingBrace(text, from + hits[0].index);
  const end = findObjectEnd(text, start) + 1;
  if (start < from || end > to) throw new Error(`file ${path}: not inside the entry`);
  const obj = JSON.parse(text.slice(start, end));
  if (obj.path !== path) throw new Error(`file ${path}: the object around its path is ${obj.path} (the path must be the file's first key)`);
  return { start, end };
}

/** Rewrite the size and hash of one file object's text. */
function pinFileText(fileText, { size, sha256, git_oid }) {
  if (!Number.isSafeInteger(size) || size <= 0) throw new Error(`invalid size ${size}`);
  if ((sha256 === undefined) === (git_oid === undefined)) throw new Error("pin exactly one of sha256 and git_oid");
  if (sha256 !== undefined && !SHA256_RE.test(sha256)) throw new Error("sha256 must be 64 lower-case hex digits");
  if (git_oid !== undefined && !SHA1_RE.test(git_oid)) throw new Error("git_oid must be 40 lower-case hex digits");
  const key = sha256 !== undefined ? "sha256" : "git_oid";
  const value = sha256 ?? git_oid;
  const other = key === "sha256" ? "git_oid" : "sha256";
  let out = fileText;
  // 1. size
  const sizeRe = /("size"\s*:\s*)\d+/;
  if (!sizeRe.test(out)) throw new Error("the file has no size");
  out = out.replace(sizeRe, (_m, head) => `${head}${size}`);
  // 2. the other hash kind goes away
  out = out.replace(new RegExp(`"${other}"\\s*:\\s*"[0-9a-f]+"\\s*,\\s*`), "");
  // 3. the hash: replace in place, or insert right after the size (same separator as after the size)
  const hashRe = new RegExp(`("${key}"\\s*:\\s*)"[0-9a-f]*"`);
  if (hashRe.test(out)) out = out.replace(hashRe, (_m, head) => `${head}"${value}"`);
  else {
    const afterSize = /("size"\s*:\s*\d+)(\s*,)(\s*)/;
    if (!afterSize.test(out)) throw new Error("cannot place the hash after the size");
    out = out.replace(afterSize, (_m, sizeText, comma, gap) => `${sizeText}${comma}${gap}"${key}": "${value}",${gap}`);
  }
  return out;
}

/**
 * Apply one edit to entry `id` of the fragment `text`:
 *   { setStatus: "published" | "unpublished" }
 *   { pinFile: { path, size, sha256 | git_oid } }
 */
export function editEntry(text, id, edit) {
  const { start, end } = findEntry(text, id);
  if (edit.setStatus !== undefined) {
    if (!["published", "unpublished"].includes(edit.setStatus)) throw new Error(`invalid status ${edit.setStatus}`);
    const slice = text.slice(start, end);
    const re = /("status"\s*:\s*)"(?:un)?published"/g;
    const hits = [...slice.matchAll(re)];
    if (hits.length !== 1) throw new Error(`entry ${id}: found ${hits.length} status fields (expected 1)`);
    return text.slice(0, start) + slice.replace(re, (_m, head) => `${head}"${edit.setStatus}"`) + text.slice(end);
  }
  if (edit.pinFile) {
    const { path, ...fact } = edit.pinFile;
    const file = findFile(text, start, end, path);
    return text.slice(0, file.start) + pinFileText(text.slice(file.start, file.end), fact) + text.slice(file.end);
  }
  throw new Error("editEntry: no edit given");
}
