import i18n from "i18next";
import { isTauri } from "@tauri-apps/api/core";
import {
  LLM_DEFAULT_URLS,
  LLM_PROVIDERS,
  LlmConnectionSettings,
  LlmEndpoint,
  LlmError,
  LlmProviderId,
  LlmTransport,
} from "./types";

/**
 * Resolves user connection settings to an endpoint: provider default URLs (`LLM_DEFAULT_URLS`), the key header
 * (`x-goog-api-key` for Gemini, Bearer otherwise), OpenRouter attribution headers, the `bundled` server's URL and
 * key, and the transport. Throws `LlmError("not_configured")` with a user-facing message when a required value is
 * missing or unusable.
 *
 * Transport for `auto` (inside Tauri; outside it everything is `webview`):
 * - cloud providers (`gemini`, `openrouter`, `openai`): `webview` (they allow browser CORS; CSP allows `https:`);
 * - `bundled` and `custom` on `http(s)://localhost` / `http(s)://127.0.0.1`: `webview`, with `rust` as the
 *   automatic fallback (`fallbackTransport`) when the browser cannot reach the server;
 * - any other `custom` host (LAN, remote): `rust` (CSP blocks plain-http LAN hosts, and CORS is unknown).
 */
export function resolveLlmEndpoint(settings: LlmConnectionSettings, options: { inTauri?: boolean } = {}): LlmEndpoint {
  const inTauri = options.inTauri ?? runningInTauri();
  // Unknown stored values (a newer settings file) behave like a custom server.
  const provider: LlmProviderId = (LLM_PROVIDERS as readonly string[]).includes(settings.provider) ? settings.provider : "custom";
  const name = llmProviderName(provider);
  const headers: Record<string, string> = {};
  let baseUrl: string;
  let apiKey: string;
  let model: string;

  switch (provider) {
    case "bundled": {
      const server = settings.bundled;
      if (!server || !text(server.baseUrl)) throw notConfigured("llm.error_bundled_not_running", "The local model server is not running.");
      baseUrl = normalizeBaseUrl(server.baseUrl);
      apiKey = text(server.apiKey);
      model = text(server.model) || "default";
      break;
    }
    case "gemini":
    case "openrouter":
    case "openai": {
      baseUrl = normalizeBaseUrl(text(settings.baseUrl) || LLM_DEFAULT_URLS[provider]);
      apiKey = text(settings.apiKey);
      model = text(settings.model);
      if (!apiKey) throw notConfigured("llm.error_key_missing", `Enter an API key for ${name} first.`, { provider: name });
      if (provider === "openrouter") {
        // OpenRouter app attribution (optional headers).
        headers["HTTP-Referer"] = "https://github.com/Mewnah/Sigil";
        headers["X-Title"] = "Sigil";
      }
      break;
    }
    default: {
      const url = text(settings.baseUrl);
      if (!url) throw notConfigured("llm.error_url_missing", "Enter the API URL of your OpenAI-compatible server first.");
      baseUrl = normalizeBaseUrl(url);
      apiKey = text(settings.apiKey);
      model = text(settings.model);
      break;
    }
  }

  if (!model) throw notConfigured("llm.error_model_missing", "Enter a model name first.");
  if (!isHttpUrl(baseUrl)) throw notConfigured("llm.error_url_invalid", `"${baseUrl}" is not an http:// or https:// URL.`, { url: baseUrl });
  // Header values must be printable ASCII; a pasted key with a line break or a non-ASCII character would make every
  // request fail with an opaque "invalid header" error.
  if (apiKey && !/^[\x21-\x7e]+$/.test(apiKey)) throw notConfigured("llm.error_key_invalid", `The API key for ${name} contains spaces or characters that are not allowed.`, { provider: name });

  const { transport, fallbackTransport } = chooseTransport(provider === "custom" || provider === "bundled" ? provider : "cloud", settings.transport, baseUrl, inTauri);
  return {
    provider,
    baseUrl,
    apiKey,
    keyHeader: provider === "gemini" ? "x-goog-api-key" : "bearer",
    model,
    headers,
    transport,
    ...(fallbackTransport ? { fallbackTransport } : {}),
  };
}

/** Display name of a provider (brand names are not translated). */
export function llmProviderName(provider: LlmProviderId): string {
  switch (provider) {
    case "bundled":
      return i18n.t("llm.provider_bundled", { defaultValue: "Local model server" }) || "Local model server";
    case "gemini":
      return "Gemini";
    case "openrouter":
      return "OpenRouter";
    case "openai":
      return "OpenAI";
    default:
      return i18n.t("llm.provider_custom", { defaultValue: "Custom server" }) || "Custom server";
  }
}

/** `host[:port]` at the start of a URL typed without a scheme; groups: host, port, what follows. */
const SCHEMELESS_HOST = /^(\[[0-9a-f:.]+\]|[a-z0-9-]+(?:\.[a-z0-9-]+)*)(:\d{1,5})?(\/|$)/i;

/**
 * `https://host/v1/` → `https://host/v1`. A pasted full endpoint (`.../v1/chat/completions`) is cut back to its
 * base, since the client appends the path itself.
 *
 * A URL typed without a scheme gets one (`new URL` would read "localhost:" as the scheme): plain `http://` only for a
 * host on this PC or the local network (`localhost:11434/v1`, `192.168.1.20:1234`, `my-pc.local`, `gpu-box:8080`),
 * `https://` for every other host, so leaving the scheme out never sends a key in cleartext over the internet.
 */
export function normalizeBaseUrl(url: string): string {
  let out = url.trim().replace(/[?#].*$/, "");
  if (!/^[a-z][a-z0-9+.-]*:\/\//i.test(out)) {
    const m = SCHEMELESS_HOST.exec(out);
    if (m) {
      const host = m[1].toLowerCase();
      // A bare word ("llama") is not taken for a single-label host unless a port or path follows.
      const hostLike = host.includes(".") || host.startsWith("[") || host === "localhost" || m[2] !== undefined || m[3] === "/";
      if (hostLike) out = `${isLocalNetworkHost(host) ? "http" : "https"}://${out}`;
    }
  }
  out = out.replace(/\/+$/, "");
  out = out.replace(/\/(?:chat\/completions|completions|completion|models)$/i, "");
  return out.replace(/\/+$/, "");
}

/** Suffixes of names that only resolve on the local network (mDNS, router DNS, RFC 8375 / 6761). */
const LOCAL_NAME_SUFFIXES = [".localhost", ".local", ".lan", ".home", ".home.arpa", ".internal", ".localdomain"];

/**
 * Whether `host` (a hostname, an IPv4 literal or a bracketed IPv6 literal) is on this PC or the local network:
 * `localhost`, single-label names, local-only suffixes (`.local`, `.lan`, `.home.arpa`, …), loopback, private
 * (RFC 1918), link-local and shared (100.64/10, e.g. Tailscale) IPv4, and loopback, unique-local and link-local IPv6.
 */
export function isLocalNetworkHost(host: string): boolean {
  const h = host.trim().toLowerCase().replace(/\.$/, "");
  if (!h) return false;
  if (h.startsWith("[")) {
    const v6 = h.replace(/^\[|\]$/g, "");
    return v6 === "::1" || /^f[cd][0-9a-f]{2}:/.test(v6) || /^fe[89ab][0-9a-f]:/.test(v6);
  }
  const v4 = /^(\d{1,3})\.(\d{1,3})\.(\d{1,3})\.(\d{1,3})$/.exec(h);
  if (v4) {
    const [a, b] = [Number(v4[1]), Number(v4[2])];
    return (
      a === 127 ||
      a === 10 ||
      (a === 172 && b >= 16 && b <= 31) ||
      (a === 192 && b === 168) ||
      (a === 169 && b === 254) ||
      (a === 100 && b >= 64 && b <= 127) ||
      h === "0.0.0.0"
    );
  }
  if (h === "localhost" || !h.includes(".")) return true;
  return LOCAL_NAME_SUFFIXES.some((suffix) => h.endsWith(suffix));
}

export function isHttpUrl(url: string): boolean {
  try {
    const u = new URL(url);
    return (u.protocol === "http:" || u.protocol === "https:") && u.hostname !== "";
  } catch {
    return false;
  }
}

/**
 * Whether the webview may fetch `url` directly: the CSP's `connect-src` allows `https:` everywhere but plain http
 * only on `localhost` / `127.0.0.1`.
 */
export function isLoopbackUrl(url: string): boolean {
  try {
    const host = new URL(url).hostname.toLowerCase();
    return host === "localhost" || host === "127.0.0.1";
  } catch {
    return false;
  }
}

/** Origin (`scheme://host:port`) of a URL, or "" when it does not parse. */
export function urlOrigin(url: string): string {
  try {
    return new URL(url).origin;
  } catch {
    return "";
  }
}

function chooseTransport(
  kind: "bundled" | "custom" | "cloud",
  choice: LlmConnectionSettings["transport"],
  baseUrl: string,
  inTauri: boolean,
): { transport: LlmTransport; fallbackTransport?: LlmTransport } {
  if (!inTauri) return { transport: "webview" };
  if (choice === "webview" || choice === "rust") return { transport: choice };
  if (kind === "cloud") return { transport: "webview" };
  if (isLoopbackUrl(baseUrl)) return { transport: "webview", fallbackTransport: "rust" };
  return { transport: "rust" };
}

/** A setting as trimmed text (persisted values can be missing or of another type in hand-edited files). */
function text(value: unknown): string {
  return typeof value === "string" ? value.trim() : "";
}

function runningInTauri(): boolean {
  try {
    return isTauri();
  } catch {
    return false;
  }
}

function notConfigured(key: string, fallback: string, values: Record<string, string> = {}): LlmError {
  const text = i18n.t(key, { ...values, defaultValue: fallback });
  return new LlmError("not_configured", text && text !== key ? text : fallback, { retryable: false });
}
