/**
 * LAN security stage 2 (P6-D): the LAN token as the host, the pages, `/client` and Link apps see it.
 * Pure helpers (no window, Tauri or DOM), shared by the host UI, the pubsub Link client and the
 * `/client` mirror. The server side is src-tauri/src/services/web/auth.rs.
 *
 * - Clients on other machines need the host's persistent LAN token; anything on the host's own PC
 *   (OBS there, `mic.html`, local tools) never does, so localhost links carry no token.
 * - The token travels as `?token=` on `/pubsub` and in every page URL, and as the PeerJS `key` on
 *   `/peer` (PeerJS sends its own unrelated `token` parameter there).
 * - Refusals after the WebSocket upgrade use close codes {@link WS_CLOSE_LAN_UNAUTHORIZED} (missing
 *   or wrong token) and {@link WS_CLOSE_LAN_DISABLED} (the host accepts only its own PC).
 */

/** Values of `lanAccess.mode` (mirrors `LAN_ACCESS_MODES` in src/core/schema.ts and Rust `BindMode`). */
export type LanAccessMode = "lan" | "localhost";

/** The root setting `lanAccess` (wire format of `plugin:web|set_lan_access`). */
export interface LanAccessSettings {
  mode: LanAccessMode;
  /** "" until the host generated one. */
  token: string;
  /** Accept other machines without the token (stage 1 behaviour). */
  legacyNoToken: boolean;
}

/** Symbols in a LAN token (Crockford base32, 130 bits). */
export const LAN_TOKEN_LENGTH = 26;

/** Lower-case Crockford base32: no i, l, o, u. */
const LAN_TOKEN_ALPHABET = "0123456789abcdefghjkmnpqrstvwxyz";

/** Longest input looked at (the Rust side uses the same bound). */
const MAX_TOKEN_INPUT = 128;

/** Query parameter that carries the token on `/pubsub` and in page URLs. */
export const LAN_TOKEN_PARAM = "token";

/** WebSocket close code: the LAN token is missing or wrong (Rust `CLOSE_LAN_UNAUTHORIZED`). */
export const WS_CLOSE_LAN_UNAUTHORIZED = 4401;

/** WebSocket close code: the host accepts only clients on its own PC (Rust `CLOSE_LAN_DISABLED`). */
export const WS_CLOSE_LAN_DISABLED = 4403;

/**
 * The canonical token, or null if `raw` is not one. Same rules as Rust `normalize_token`: ASCII case
 * is ignored, `-`, spaces and tabs are skipped, and the look-alikes `o` → `0`, `i` / `l` → `1` are
 * mapped (the shared vectors are in both test suites).
 */
export function normalizeLanToken(raw: string | null | undefined): string | null {
  if (typeof raw !== "string" || raw.length > MAX_TOKEN_INPUT) return null;
  let out = "";
  for (const ch of raw) {
    let c = ch.length === 1 && ch.charCodeAt(0) < 128 ? ch.toLowerCase() : ch;
    if (c === "-" || c === " " || c === "\t") continue;
    if (c === "o") c = "0";
    else if (c === "i" || c === "l") c = "1";
    if (c.length !== 1 || !LAN_TOKEN_ALPHABET.includes(c)) return null;
    out += c;
  }
  return out.length === LAN_TOKEN_LENGTH ? out : null;
}

export function isValidLanToken(raw: string | null | undefined): boolean {
  return normalizeLanToken(raw) !== null;
}

/** `abcde-fghij-…` groups of five, for reading a token aloud or typing it on another PC. */
export function groupLanToken(token: string): string {
  return token.replace(/(.{5})(?=.)/g, "$1-");
}

/** `abcd…wxyz` for showing that a token exists without revealing it. */
export function maskLanToken(token: string): string {
  if (token.length <= 8) return "•".repeat(token.length);
  return `${token.slice(0, 4)}${"•".repeat(token.length - 8)}${token.slice(-4)}`;
}

/** `host` as it must appear in a URL authority: bare IPv6 literals get brackets. */
function urlHost(host: string): string {
  const h = host.trim();
  return h.includes(":") && !h.startsWith("[") ? `[${h}]` : h;
}

/**
 * Link to the `/client` canvas mirror for a device on the LAN: `http://<ip>:<port>/client?token=…`.
 * The page connects to the server that served it, so no `?host=` is needed.
 */
export function lanClientUrl(ip: string, port: string | number, token: string, pathname = "/client"): string {
  const path = pathname.startsWith("/") ? pathname : `/${pathname}`;
  const valid = normalizeLanToken(token);
  const query = valid ? `?${new URLSearchParams({ [LAN_TOKEN_PARAM]: valid }).toString()}` : "";
  return `http://${urlHost(ip)}:${String(port).trim()}${path}${query}`;
}

/** What a Link apps address names: another Sigil's server and its LAN token. */
export interface LinkAddress {
  host: string;
  port: string;
  /** Normalized token, or "" when none was given. */
  token: string;
  /** A token was given but is not a valid LAN token (the link still connects without it). */
  tokenInvalid: boolean;
}

const LINK_HOST_PORT = /^(?:(?:https?|wss?):\/\/)?((?:(?:25[0-5]|2[0-4]\d|1\d\d|[1-9]?\d)\.){3}(?:25[0-5]|2[0-4]\d|1\d\d|[1-9]?\d)):(\d{1,5})(?=$|[/?#])/i;

/**
 * Parse a Link apps address: `192.168.1.50:3030`, optionally with `?token=…` (or `#token=…`) and an
 * optional `http(s)://` / `ws(s)://` scheme and path, as copied by "Copy my address" or from a LAN
 * link. The host must be an IPv4 address with an explicit port (as before stage 2). Null if not.
 */
export function parseLinkAddress(raw: string): LinkAddress | null {
  const text = raw.trim();
  const m = LINK_HOST_PORT.exec(text);
  if (!m) return null;
  const host = m[1]!;
  const portNumber = Number(m[2]);
  if (!(portNumber >= 1 && portNumber <= 65535)) return null;
  const rest = text.slice(m[0].length);
  const queryStart = rest.indexOf("?");
  const hashStart = rest.indexOf("#");
  const query = queryStart >= 0 ? rest.slice(queryStart + 1, hashStart > queryStart ? hashStart : undefined) : "";
  const hash = hashStart >= 0 ? rest.slice(hashStart + 1) : "";
  const given = new URLSearchParams(query).get(LAN_TOKEN_PARAM) ?? new URLSearchParams(hash).get(LAN_TOKEN_PARAM);
  const token = normalizeLanToken(given);
  return {
    host,
    port: String(portNumber),
    token: token ?? "",
    tokenInvalid: given !== null && given.trim() !== "" && token === null,
  };
}

/**
 * Split a stored Link address into the address part (what the user typed before any `?` / `#`) and
 * the token, for the two Settings fields.
 */
export function splitLinkAddress(raw: string): { address: string; token: string } {
  const text = raw.trim();
  const cut = text.search(/[?#]/);
  const address = cut >= 0 ? text.slice(0, cut) : text;
  if (cut < 0) return { address, token: "" };
  const params = new URLSearchParams(text.slice(cut + 1).replace(/#/g, "&"));
  const given = params.get(LAN_TOKEN_PARAM) ?? "";
  return { address, token: normalizeLanToken(given) ?? given.trim() };
}

/** The stored form of a Link address: `address` plus `?token=<token>` when a token is set. */
export function formatLinkAddress(address: string, token: string): string {
  const base = splitLinkAddress(address).address;
  const t = token.trim();
  if (!t) return base;
  const normalized = normalizeLanToken(t) ?? t;
  return `${base}?${new URLSearchParams({ [LAN_TOKEN_PARAM]: normalized }).toString()}`;
}

/** The address another Sigil pastes to link to this host: `<ip>:<port>?token=…`. */
export function lanLinkAddress(ip: string, port: string | number, token: string): string {
  return formatLinkAddress(`${ip}:${String(port).trim()}`, normalizeLanToken(token) ?? "");
}

/** `ws://host:port/pubsub?id=…[&token=…]` for a Link apps connection. */
export function linkSocketUrl(link: Pick<LinkAddress, "host" | "port" | "token">, id: string): string {
  const query = new URLSearchParams({ id });
  if (link.token) query.set(LAN_TOKEN_PARAM, link.token);
  return `ws://${urlHost(link.host)}:${link.port}/pubsub?${query.toString()}`;
}

/** The LAN token in a page's query (`?token=`), normalized, or "" when absent or malformed. */
export function lanTokenFromQuery(search: string | URLSearchParams): string {
  const q = typeof search === "string" ? new URLSearchParams(search) : search;
  return normalizeLanToken(q.get(LAN_TOKEN_PARAM)) ?? "";
}

/** Why a WebSocket close code means the host refused this client, or null. */
export function lanRefusalFromCloseCode(code: number | undefined): "token" | "lan_disabled" | null {
  if (code === WS_CLOSE_LAN_UNAUTHORIZED) return "token";
  if (code === WS_CLOSE_LAN_DISABLED) return "lan_disabled";
  return null;
}
