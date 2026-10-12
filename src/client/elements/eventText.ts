/**
 * Event text: the small template language that turns an event's value into a line of text.
 *
 * It replaces lodash `template`, whose `<% %>` and `${}` delimiters compile their contents to JavaScript. A scene
 * document that carries such a text could run code in the window that opens it, and the host window has Tauri
 * IPC. Nothing in this file evaluates, compiles or looks anything up by name outside the event value itself.
 *
 * Remote-safe: pure TypeScript with no `window`, no Tauri and no lodash, so it runs the same on a /client page.
 *
 * Syntax (documented for users in docs/manual/02-studio.md, "Event text"):
 *
 * - `${event}`, `${event.a.b}`, `<%= event %>`, `<%= event.a.b %>` insert the value, as is.
 * - `<%- event.a %>` inserts the value with `& < > " '` HTML-escaped (what lodash's `<%-` did).
 * - A path is `event` followed by `.name`, `?.name` (the same as `.name`), `[0]`, `["key"]` or `['key']`.
 *   Nothing else is a path: no calls, operators, other variables, template literals or computed keys.
 * - A path reads own data properties of objects and arrays only. It never walks a prototype chain, never runs a
 *   getter, and refuses the names `__proto__`, `constructor` and `prototype` (so the template text itself can
 *   never reach `Function`, `Object.prototype` or `window`).
 * - A missing path, `null` and `undefined` insert nothing (lodash did the same for `null` and `undefined`).
 * - Everything the reader does not understand is left as typed: `<% code %>` blocks (they cannot run, and
 *   stripping them would silently change what `<% if (x) { %>A<% } %>` shows), a path that is not allowed, an
 *   unterminated `${` or `<%`. Showing the text unchanged tells the author at once, and a hostile document can
 *   only print its own words, which it could have written as plain text anyway.
 *
 * Reading is one linear pass: every opener looks for its closer with `indexOf`, and once a closer is known not to
 * exist further on, no later opener of that kind searches again, so text full of unterminated `<%=` or `${`
 * cannot cost quadratic time.
 */

/** One piece of a parsed event text: plain text, or a value taken from the event. */
export type EventTextPart =
  | string
  | { readonly path: readonly string[]; readonly escape: boolean };

/** What `compileEventText` returns: render the text for one event value. Never throws for plain data. */
export type EventTextRenderer = (event: unknown) => string;

/** Property names a path may never use, whichever syntax names them. */
const REFUSED_KEYS: ReadonlySet<string> = new Set(["__proto__", "constructor", "prototype"]);

/** Arrays inside arrays are printed this deep; deeper (or cyclic) levels print nothing. */
const MAX_ARRAY_DEPTH = 8;

const OPENER = /<%([-=]?)|\$\{/g;
const IDENT = /[\p{L}_$][\p{L}\p{N}_$]*/uy;
const DIGITS = /[0-9]+/y;
const SPACE = /\s/;

const HTML_ESCAPES: Readonly<Record<string, string>> = {
  "&": "&amp;",
  "<": "&lt;",
  ">": "&gt;",
  '"': "&quot;",
  "'": "&#39;",
};

/** HTML-escape `& < > " '`, the five characters lodash's `_.escape` handled. */
export function escapeEventHtml(value: string): string {
  return value.replace(/[&<>"']/g, (c) => HTML_ESCAPES[c]);
}

function skipSpace(src: string, i: number): number {
  while (i < src.length && SPACE.test(src[i])) i++;
  return i;
}

/**
 * The keys of a path expression (`event.a[0]["b"]` gives `["a", "0", "b"]`), or `null` when the text is anything
 * but a plain path on `event`. Hand-written and linear: no backtracking, no escapes inside quoted keys.
 */
function parsePath(src: string): string[] | null {
  let end = src.length;
  while (end > 0 && SPACE.test(src[end - 1])) end--;
  let i = skipSpace(src, 0);
  if (i >= end || !src.startsWith("event", i)) return null;
  i += "event".length;

  const keys: string[] = [];
  const push = (key: string): boolean => {
    if (REFUSED_KEYS.has(key)) return false;
    keys.push(key);
    return true;
  };

  while (i < end) {
    const c = src[i];
    let bracket = c === "[";
    if (c === "?" && src[i + 1] === ".") {
      i += 2;
      bracket = src[i] === "[";
    } else if (c === ".") {
      i += 1;
    } else if (!bracket) {
      return null;
    }

    if (!bracket) {
      IDENT.lastIndex = i;
      const m = IDENT.exec(src);
      if (!m || i + m[0].length > end || !push(m[0])) return null;
      i += m[0].length;
      continue;
    }

    // [ 0 ]  [ "key" ]  [ 'key' ]
    i = skipSpace(src, i + 1);
    let key: string;
    const q = src[i];
    if (q === '"' || q === "'") {
      const close = src.indexOf(q, i + 1);
      if (close < 0 || close >= end) return null;
      key = src.slice(i + 1, close);
      if (key.includes("\\")) return null;
      i = close + 1;
    } else {
      DIGITS.lastIndex = i;
      const m = DIGITS.exec(src);
      if (!m) return null;
      key = m[0];
      i += key.length;
    }
    i = skipSpace(src, i);
    if (src[i] !== "]" || i >= end) return null;
    i += 1;
    if (!push(key)) return null;
  }
  return keys;
}

/**
 * Split an event text into plain text and values. Delimiters pair up the way lodash's did (`<%-`, `<%=`, `<%`
 * up to the first `%>`, `${` up to the first `}`); a pair whose inside is not an allowed path stays plain text.
 */
export function parseEventText(text: string): EventTextPart[] {
  const parts: EventTextPart[] = [];
  const opener = new RegExp(OPENER); // own lastIndex: never shared with another call
  let copyFrom = 0;
  let noPercentCloser = false;
  let noBraceCloser = false;

  for (;;) {
    const m = opener.exec(text);
    if (!m) break;
    const brace = m[0] === "${";
    if (brace ? noBraceCloser : noPercentCloser) continue;

    const bodyStart = m.index + m[0].length;
    const close = text.indexOf(brace ? "}" : "%>", bodyStart);
    if (close < 0) {
      if (brace) noBraceCloser = true;
      else noPercentCloser = true;
      continue;
    }
    // The pair is one unit, as in lodash: its inside is never read again for other delimiters.
    opener.lastIndex = close + (brace ? 1 : 2);

    const kind = brace ? "=" : m[1]; // "=" interpolate, "-" escape, "" code block
    if (kind === "") continue;
    const path = parsePath(text.slice(bodyStart, close));
    if (!path) continue;

    if (m.index > copyFrom) parts.push(text.slice(copyFrom, m.index));
    parts.push({ path, escape: kind === "-" });
    copyFrom = opener.lastIndex;
  }
  if (copyFrom < text.length) parts.push(text.slice(copyFrom));
  return parts;
}

/** Read an own data property chain. Anything else (primitive, function, inherited, accessor) gives `undefined`. */
function readPath(root: unknown, path: readonly string[]): unknown {
  let cur = root;
  for (const key of path) {
    if (cur === null || typeof cur !== "object") return undefined;
    const d = Object.getOwnPropertyDescriptor(cur, key);
    if (!d || !("value" in d)) return undefined;
    cur = d.value;
  }
  return cur;
}

/**
 * The text of a value the way `'' + value` made it in lodash, without calling anything the value carries:
 * `null` and `undefined` give nothing, an array its items joined by commas, any other object `[object Object]`.
 */
function stringifyValue(value: unknown, depth = 0): string {
  switch (typeof value) {
    case "string":
      return value;
    case "number":
    case "boolean":
    case "bigint":
      return String(value);
    case "object": {
      if (value === null) return "";
      if (!Array.isArray(value)) return "[object Object]";
      if (depth >= MAX_ARRAY_DEPTH) return "";
      let out = "";
      for (let i = 0; i < value.length; i++) {
        if (i > 0) out += ",";
        out += stringifyValue(value[i], depth + 1);
      }
      return out;
    }
    default: // undefined, symbol, function
      return "";
  }
}

/** Parse once, render many times: the returned function renders the text for one event value. */
export function compileEventText(text: string): EventTextRenderer {
  const parts = parseEventText(text);
  if (parts.every((p) => typeof p === "string")) {
    const literal = parts.join("");
    return () => literal;
  }
  return (event) => {
    let out = "";
    for (const part of parts) {
      if (typeof part === "string") {
        out += part;
        continue;
      }
      const s = stringifyValue(readPath(event, part.path));
      out += part.escape ? escapeEventHtml(s) : s;
    }
    return out;
  };
}

/** Render an event text for one event value (parses every call; use `compileEventText` for a hot path). */
export function renderEventText(text: string, event: unknown): string {
  return compileEventText(text)(event);
}
