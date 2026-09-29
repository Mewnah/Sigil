/**
 * URLs of the static pages the local web server serves next to the app (`public/*.html`).
 *
 * Owned by P0-I; scaffolded by P0-S0 so other workstreams can switch their hand-built URLs to it.
 * FROZEN SIGNATURE (P0-S0): `localPageUrl(page, options?) => string`, `LocalPage`,
 * `LocalPageUrlOptions`. P0-I may extend the `LocalPage` union and the options (and, in LAN auth
 * stage 2, append a token) without breaking callers.
 *
 * Behaviour (identical to the URLs built by hand before):
 * - `mic.html` opens in an external Chrome/Edge, so it defaults to `127.0.0.1` (a secure context
 *   for getUserMedia); the other pages default to `window.Config.serverNetwork.host` ("localhost").
 * - Every URL carries `?port=<port>` so the page reaches the pubsub WebSocket on the right port.
 * - The page's own origin (`http://<host>:<port>`) is what the server's Origin allow-list accepts
 *   (LAN security stage 1), so these URLs must keep pointing at this machine's web server.
 *
 * The reverse direction lives here too: {@link clientServerAddress} decides which server a served
 * `/client` page may connect to (only the one that served it, or loopback).
 */

/** Static pages served by the host's web server. */
export type LocalPage = "mic.html" | "obs-captions.html" | "inputfield.html";

export type LocalPageParamValue = string | number | boolean | null | undefined;

export interface LocalPageUrlOptions {
  /** Host in the URL. Default: "127.0.0.1" for mic.html, the server network host otherwise. */
  host?: string;
  /** Port of the local web server. Default: `window.Config.serverNetwork.port`. */
  port?: number | string;
  /** Extra query parameters; `null` / `undefined` are skipped, `port` is always set from `port`. */
  params?: Record<string, LocalPageParamValue>;
}

const LOOPBACK_PAGES: ReadonlySet<LocalPage> = new Set<LocalPage>(["mic.html"]);

/** `host` as it must appear in a URL authority: bare IPv6 literals get brackets. */
function urlHost(host: string): string {
  const h = host.trim();
  return h.includes(":") && !h.startsWith("[") ? `[${h}]` : h;
}

/** Absolute `http://host:port/<page>?port=<port>&...` URL of a local page. */
export function localPageUrl(page: LocalPage, options: LocalPageUrlOptions = {}): string {
  const network = typeof window !== "undefined" ? window.Config?.serverNetwork : undefined;
  const port = String(options.port ?? network?.port ?? "").trim();
  const host = urlHost(options.host ?? (LOOPBACK_PAGES.has(page) ? "127.0.0.1" : network?.host ?? "localhost"));
  const query = new URLSearchParams({ port });
  for (const [key, value] of Object.entries(options.params ?? {})) {
    if (key === "port" || value === null || value === undefined) continue;
    query.set(key, String(value));
  }
  return `http://${host}:${port}/${page}?${query.toString()}`;
}

/**
 * Whether `host` (as written in a URL or a `?host=` value) is a loopback name or address:
 * `localhost`, `*.localhost`, `127.0.0.0/8` or `::1`. DNS names that merely resolve to loopback
 * do not count.
 */
export function isLoopbackHost(host: string): boolean {
  let h = host.trim().toLowerCase();
  if (h.startsWith("[") && h.endsWith("]"))
    h = h.slice(1, -1);
  if (h.endsWith("."))
    h = h.slice(0, -1);
  if (h === "localhost" || /^(?:[a-z0-9](?:[a-z0-9-]*[a-z0-9])?\.)+localhost$/.test(h))
    return true;
  const v4 = /^127\.(\d{1,3})\.(\d{1,3})\.(\d{1,3})$/.exec(h);
  if (v4)
    return v4.slice(1).every((octet) => Number(octet) <= 255);
  return h === "::1" || h === "0:0:0:0:0:0:0:1" || h === "::ffff:127.0.0.1";
}

/** The Sigil web server a `/client` page connects PeerJS to. */
export interface ClientServerAddress {
  host: string;
  port: string;
}

/** The page location fields {@link clientServerAddress} needs (`window.location`). */
export interface ClientPageLocation {
  hostname: string;
  port: string;
}

/**
 * Which Sigil server a `/client` page (canvas mirror, OBS browser source) connects to: the server
 * that served the page. `?host=` is honoured only when it names the page's own host or a loopback
 * address (old Curses-style links such as `?host=localhost&port=3030`, or the Vite dev server
 * pointing at 127.0.0.1). Any other `?host=` is ignored together with its `?port=`, so a website
 * cannot open `/client?host=evil.example` and feed the mirror (and its HTML sinks) from its own
 * PeerJS server. `?port=` must be a port number.
 */
export function clientServerAddress(search: string | URLSearchParams, page: ClientPageLocation): ClientServerAddress & {
  /** `?host=` / `?port=` values that were present but not used. */
  ignored: ("host" | "port")[];
} {
  const q = typeof search === "string" ? new URLSearchParams(search) : search;
  const ignored: ("host" | "port")[] = [];
  const pageHost = page.hostname;
  let host = pageHost;
  let port = page.port;

  const rawHost = q.get("host");
  const hostAllowed =
    rawHost === null || rawHost.trim().toLowerCase() === pageHost.toLowerCase() || isLoopbackHost(rawHost);
  if (rawHost !== null) {
    if (hostAllowed)
      host = rawHost.trim();
    else
      ignored.push("host");
  }

  const rawPort = q.get("port");
  if (rawPort !== null) {
    const valid = /^\d{1,5}$/.test(rawPort) && Number(rawPort) >= 1 && Number(rawPort) <= 65535;
    if (hostAllowed && valid)
      port = rawPort;
    else
      ignored.push("port");
  }
  return { host, port, ignored };
}

/**
 * `search` without the `?host=` / `?port=` values {@link clientServerAddress} would ignore, so code
 * that reads the query directly (AppConfiguration) sees only the allowed server. Returns null when
 * nothing needs to change.
 */
export function clientQueryWithoutForeignServer(search: string, page: ClientPageLocation): string | null {
  const q = new URLSearchParams(search);
  const { ignored } = clientServerAddress(q, page);
  if (!ignored.length)
    return null;
  for (const key of ignored)
    q.delete(key);
  const rest = q.toString();
  return rest ? `?${rest}` : "";
}
