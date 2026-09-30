import i18n from "i18next";
import { beforeAll, describe, expect, it, vi } from "vitest";

vi.mock("@tauri-apps/api/core", () => ({ invoke: vi.fn(), isTauri: () => true, Channel: class {} }));

import { isLocalNetworkHost, isLoopbackUrl, normalizeBaseUrl, resolveLlmEndpoint } from "../providers";
import { LlmConnectionSettings, LlmError } from "../types";

function settings(patch: Partial<LlmConnectionSettings> = {}): LlmConnectionSettings {
  return { provider: "custom", model: "m", apiKey: "", baseUrl: "", transport: "auto", ...patch };
}

const inTauri = { inTauri: true };

function notConfigured(fn: () => unknown): LlmError {
  try {
    fn();
  } catch (e) {
    expect(e).toBeInstanceOf(LlmError);
    expect((e as LlmError).kind).toBe("not_configured");
    expect((e as LlmError).retryable).toBe(false);
    return e as LlmError;
  }
  throw new Error("expected a not_configured error");
}

beforeAll(async () => {
  await i18n.init({ lng: "en", resources: { en: { translation: {} } } });
});

describe("resolveLlmEndpoint", () => {
  it("sends the Gemini key in x-goog-api-key to the OpenAI-compatible endpoint", () => {
    const e = resolveLlmEndpoint(settings({ provider: "gemini", apiKey: " AIzaKey ", model: "gemini-3.5-flash-lite" }), inTauri);
    expect(e).toMatchObject({
      provider: "gemini",
      baseUrl: "https://generativelanguage.googleapis.com/v1beta/openai",
      apiKey: "AIzaKey",
      keyHeader: "x-goog-api-key",
      model: "gemini-3.5-flash-lite",
      transport: "webview",
    });
    expect(e.fallbackTransport).toBeUndefined();
  });

  it("fills provider defaults and OpenRouter attribution, never putting the key in headers", () => {
    const or = resolveLlmEndpoint(settings({ provider: "openrouter", apiKey: "sk-or-1", model: "google/gemini-2.5-flash-lite" }), inTauri);
    expect(or.baseUrl).toBe("https://openrouter.ai/api/v1");
    expect(or.keyHeader).toBe("bearer");
    expect(or.headers).toEqual({ "HTTP-Referer": "https://github.com/Mewnah/Sigil", "X-Title": "Sigil" });
    expect(JSON.stringify(or.headers)).not.toContain("sk-or-1");
    const oa = resolveLlmEndpoint(settings({ provider: "openai", apiKey: "sk-1", baseUrl: "https://proxy.example/v1/", model: "gpt-5-mini" }), inTauri);
    expect(oa.baseUrl).toBe("https://proxy.example/v1");
  });

  it("requires what each provider needs", () => {
    expect(notConfigured(() => resolveLlmEndpoint(settings({ provider: "gemini", model: "g" }), inTauri)).message).toBe("Enter an API key for Gemini first.");
    notConfigured(() => resolveLlmEndpoint(settings({ provider: "openai", model: "g" }), inTauri));
    expect(notConfigured(() => resolveLlmEndpoint(settings({ provider: "custom" }), inTauri)).message).toContain("API URL");
    notConfigured(() => resolveLlmEndpoint(settings({ provider: "custom", baseUrl: "http://127.0.0.1:8080/v1", model: " " }), inTauri));
    notConfigured(() => resolveLlmEndpoint(settings({ provider: "custom", baseUrl: "ftp://host/v1" }), inTauri));
    notConfigured(() => resolveLlmEndpoint(settings({ provider: "custom", baseUrl: "not a url" }), inTauri));
    notConfigured(() => resolveLlmEndpoint(settings({ provider: "bundled" }), inTauri));
    // A key with a line break or non-ASCII characters cannot be sent in a header.
    notConfigured(() => resolveLlmEndpoint(settings({ provider: "openai", apiKey: "sk-a\nb", model: "g" }), inTauri));
    notConfigured(() => resolveLlmEndpoint(settings({ provider: "openai", apiKey: "sk-ä", model: "g" }), inTauri));
  });

  it("lets a custom server run without a key (no auth header is sent)", () => {
    const e = resolveLlmEndpoint(settings({ baseUrl: " http://127.0.0.1:8080/v1 ", model: "qwen" }), inTauri);
    expect(e).toMatchObject({ provider: "custom", apiKey: "", baseUrl: "http://127.0.0.1:8080/v1", model: "qwen" });
  });

  it("uses the running bundled server's URL, key and alias", () => {
    const e = resolveLlmEndpoint(
      settings({ provider: "bundled", model: "", bundled: { baseUrl: "http://127.0.0.1:5123/v1/", apiKey: "k3y-123456", model: "hy-mt2" } }),
      inTauri,
    );
    expect(e).toMatchObject({ provider: "bundled", baseUrl: "http://127.0.0.1:5123/v1", apiKey: "k3y-123456", keyHeader: "bearer", model: "hy-mt2" });
    expect(e.transport).toBe("webview");
    expect(e.fallbackTransport).toBe("rust");
  });

  it("routes by host in auto: LAN and remote custom hosts through Rust, loopback direct with a Rust fallback", () => {
    const lan = resolveLlmEndpoint(settings({ baseUrl: "http://192.168.1.20:11434/v1" }), inTauri);
    expect(lan.transport).toBe("rust");
    expect(lan.fallbackTransport).toBeUndefined();
    expect(resolveLlmEndpoint(settings({ baseUrl: "https://api.groq.com/openai/v1" }), inTauri).transport).toBe("rust");
    const local = resolveLlmEndpoint(settings({ baseUrl: "http://localhost:11434/v1" }), inTauri);
    expect(local).toMatchObject({ transport: "webview", fallbackTransport: "rust" });
    // Explicit choices win and get no fallback.
    expect(resolveLlmEndpoint(settings({ baseUrl: "http://192.168.1.20/v1", transport: "webview" }), inTauri)).toMatchObject({ transport: "webview" });
    const forced = resolveLlmEndpoint(settings({ baseUrl: "http://localhost:1234/v1", transport: "rust" }), inTauri);
    expect(forced.transport).toBe("rust");
    expect(forced.fallbackTransport).toBeUndefined();
    expect(resolveLlmEndpoint(settings({ provider: "openai", apiKey: "k-123", transport: "rust" }), inTauri).transport).toBe("rust");
  });

  it("sends a public host typed without a scheme over https, even through the proxy", () => {
    const e = resolveLlmEndpoint(settings({ baseUrl: "api.groq.com/openai/v1", apiKey: "gsk_test123", model: "llama-3.1-8b-instant" }), inTauri);
    expect(e).toMatchObject({ baseUrl: "https://api.groq.com/openai/v1", transport: "rust" });
    expect(resolveLlmEndpoint(settings({ baseUrl: "api.groq.com/openai/v1" }), { inTauri: false }).baseUrl).toBe("https://api.groq.com/openai/v1");
    expect(resolveLlmEndpoint(settings({ baseUrl: "localhost:11434/v1" }), inTauri)).toMatchObject({ baseUrl: "http://localhost:11434/v1", transport: "webview", fallbackTransport: "rust" });
  });

  it("uses the webview for everything outside Tauri", () => {
    const e = resolveLlmEndpoint(settings({ baseUrl: "http://192.168.1.20:11434/v1", transport: "rust" }), { inTauri: false });
    expect(e.transport).toBe("webview");
    expect(e.fallbackTransport).toBeUndefined();
  });

  it("detects Tauri by itself when not told", () => {
    expect(resolveLlmEndpoint(settings({ baseUrl: "http://10.0.0.2:8080/v1" })).transport).toBe("rust");
  });

  it("treats an unknown stored provider as a custom server", () => {
    const e = resolveLlmEndpoint(settings({ provider: "future" as never, baseUrl: "http://localhost:8080/v1" }), inTauri);
    expect(e.provider).toBe("custom");
  });
});

describe("URL helpers", () => {
  it("normalizes pasted base URLs", () => {
    expect(normalizeBaseUrl("http://localhost:11434/v1/")).toBe("http://localhost:11434/v1");
    expect(normalizeBaseUrl("http://localhost:11434/v1/chat/completions")).toBe("http://localhost:11434/v1");
    expect(normalizeBaseUrl("https://x.example/v1/models?x=1")).toBe("https://x.example/v1");
    expect(normalizeBaseUrl(" https://x.example/api/v1// ")).toBe("https://x.example/api/v1");
    // A URL typed without a scheme means plain http on this PC and the local network...
    expect(normalizeBaseUrl("localhost:11434/v1")).toBe("http://localhost:11434/v1");
    expect(normalizeBaseUrl("localhost")).toBe("http://localhost");
    expect(normalizeBaseUrl("127.0.0.1:8080/v1")).toBe("http://127.0.0.1:8080/v1");
    expect(normalizeBaseUrl("192.168.1.20:1234/v1")).toBe("http://192.168.1.20:1234/v1");
    expect(normalizeBaseUrl("10.0.0.5:8000/v1")).toBe("http://10.0.0.5:8000/v1");
    expect(normalizeBaseUrl("172.20.1.1:8080")).toBe("http://172.20.1.1:8080");
    expect(normalizeBaseUrl("100.101.102.103:11434/v1")).toBe("http://100.101.102.103:11434/v1");
    expect(normalizeBaseUrl("my-pc.local:8080")).toBe("http://my-pc.local:8080");
    expect(normalizeBaseUrl("gpu-box:8080/v1")).toBe("http://gpu-box:8080/v1");
    expect(normalizeBaseUrl("gpu-box/v1")).toBe("http://gpu-box/v1");
    expect(normalizeBaseUrl("[::1]:11434/v1")).toBe("http://[::1]:11434/v1");
    // ...and https everywhere else, so a key never goes out in cleartext because the scheme was left out.
    expect(normalizeBaseUrl("api.groq.com/openai/v1")).toBe("https://api.groq.com/openai/v1");
    expect(normalizeBaseUrl("api.openai.com/v1/chat/completions")).toBe("https://api.openai.com/v1");
    expect(normalizeBaseUrl("8.8.8.8:8080/v1")).toBe("https://8.8.8.8:8080/v1");
    expect(normalizeBaseUrl("172.32.0.1:8080")).toBe("https://172.32.0.1:8080");
    expect(normalizeBaseUrl("[2001:db8::1]:8080")).toBe("https://[2001:db8::1]:8080");
    // An explicit scheme is kept as typed.
    expect(normalizeBaseUrl("http://my-server.example/v1")).toBe("http://my-server.example/v1");
    // Not a URL: left for the validity check to reject.
    expect(normalizeBaseUrl("not a url")).toBe("not a url");
    expect(normalizeBaseUrl("llama")).toBe("llama");
  });

  it("tells local-network hosts from internet hosts", () => {
    for (const host of ["localhost", "LOCALHOST", "app.localhost", "gpu-box", "nas.local", "router.lan", "pc.home.arpa", "box.internal", "127.0.0.1", "10.1.2.3", "172.16.0.1", "172.31.255.255", "192.168.0.2", "169.254.1.1", "100.64.0.1", "[::1]", "[fd12:3456::1]", "[fe80::1]"]) {
      expect(isLocalNetworkHost(host), host).toBe(true);
    }
    for (const host of ["api.groq.com", "example.com", "8.8.8.8", "172.15.0.1", "172.32.0.1", "100.128.0.1", "192.169.0.1", "[2001:db8::1]", "[fd:1::1]", ""]) {
      expect(isLocalNetworkHost(host), host).toBe(false);
    }
  });

  it("knows which hosts the CSP lets the webview reach over plain http", () => {
    expect(isLoopbackUrl("http://localhost:8080/v1")).toBe(true);
    expect(isLoopbackUrl("http://127.0.0.1:8080/v1")).toBe(true);
    expect(isLoopbackUrl("http://[::1]:8080/v1")).toBe(false);
    expect(isLoopbackUrl("http://192.168.0.2/v1")).toBe(false);
    expect(isLoopbackUrl("not a url")).toBe(false);
  });
});
