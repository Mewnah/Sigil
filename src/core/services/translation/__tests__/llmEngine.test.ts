import i18n from "i18next";
import { afterEach, beforeAll, beforeEach, describe, expect, it, vi } from "vitest";
import type { LlmCallOptions, LlmChatRequest, LlmCompletionRequest, LlmConnectionSettings, LlmEndpoint, LlmResult } from "@/core/services/llm/types";
import type { RustLlmModelInfo, RustLlmServerEndpoint } from "@/types";

vi.mock("@tauri-apps/api/core", () => ({ invoke: vi.fn(), isTauri: () => false }));

// P4-A client and P4-B server host, mocked at their frozen contracts.
const { client, endpoints, server } = vi.hoisted(() => ({
  client: {
    chat: vi.fn<(req: LlmChatRequest, opts: LlmCallOptions) => Promise<LlmResult>>(),
    complete: vi.fn<(req: LlmCompletionRequest, opts: LlmCallOptions) => Promise<LlmResult>>(),
    listModels: vi.fn<(signal: AbortSignal) => Promise<string[]>>(),
  },
  endpoints: [] as LlmEndpoint[],
  server: {
    acquireLlmServer: vi.fn(),
    getLlmServerStatus: vi.fn(),
    listLlmModels: vi.fn<() => Promise<RustLlmModelInfo[]>>(),
  },
}));
vi.mock("@/core/services/llm/client", () => ({
  createLlmClient: (endpoint: LlmEndpoint) => {
    endpoints.push(endpoint);
    return { endpoint, ...client };
  },
}));
vi.mock("@/core/services/llm/providers", async () => {
  const { LlmError } = await import("@/core/services/llm/types");
  return {
    resolveLlmEndpoint: (s: LlmConnectionSettings): LlmEndpoint => {
      if (s.provider === "bundled") {
        if (!s.bundled) throw new LlmError("not_configured", "llm.error_bundled_not_running");
        return { provider: "bundled", baseUrl: s.bundled.baseUrl, apiKey: s.bundled.apiKey, keyHeader: "bearer", model: s.bundled.model, headers: {}, transport: "webview" };
      }
      if (s.provider !== "custom" && !s.apiKey) throw new LlmError("not_configured", "llm.error_key_missing");
      return {
        provider: s.provider,
        baseUrl: s.baseUrl || "https://example.invalid/v1",
        apiKey: s.apiKey,
        keyHeader: s.provider === "gemini" ? "x-goog-api-key" : "bearer",
        model: s.model,
        headers: {},
        transport: "webview",
      };
    },
  };
});
const lease = (model = "hy-mt2", port = 5000) => {
  const l = {
    holder: "translation",
    endpoint: { baseUrl: `http://127.0.0.1:${port}/v1`, apiKey: "k", model } as RustLlmServerEndpoint,
    refresh: vi.fn<() => Promise<RustLlmServerEndpoint>>(),
    release: vi.fn(),
  };
  l.refresh.mockResolvedValue(l.endpoint);
  return l;
};
vi.mock("@/core/services/llm_server", () => server);

import { LlmError } from "@/core/services/llm/types";
import { Service_Translation_Schema, Translation_State } from "../schema";
import { LLM_HOLDER, LLM_INTERIM_TIMEOUT_MS, LLM_SEED, Translation_LlmService } from "../services/llm";
import { TranslationError, TranslationRequest } from "../types";

const model = (patch: Partial<RustLlmModelInfo> = {}): RustLlmModelInfo => ({
  id: "mt/hy-mt2-1.8b-q4_k_m",
  title: "Hy-MT2 1.8B",
  promptFormat: "hy-mt2",
  sizeBytes: 1,
  vramMb: 1,
  installed: true,
  specialist: false,
  languages: [],
  recommended: true,
  ...patch,
});

const result = (text: string): LlmResult => ({ text, raw: text, finishReason: "stop", model: "m", totalMs: 1 });
const live = () => new AbortController().signal;

function state(patch: (s: Translation_State) => void = () => {}): Translation_State {
  const s = Service_Translation_Schema.parse({ targetLanguage: "ja", sourceLanguage: "en" });
  patch(s);
  return s;
}

async function startedEngine(patch?: (s: Translation_State) => void) {
  const e = new Translation_LlmService();
  const statuses: string[] = [];
  e.attach({ engines: {} as never, status: (u) => statuses.push(u.text) });
  await e.start(state(patch), live());
  return { e, statuses };
}

beforeAll(async () => {
  await i18n.init({ lng: "en", resources: { en: { translation: {} } } });
});

beforeEach(() => {
  endpoints.length = 0;
  client.chat.mockReset().mockResolvedValue(result("おはよう"));
  client.complete.mockReset().mockResolvedValue(result("おはよう"));
  client.listModels.mockReset().mockResolvedValue(["m"]);
  server.acquireLlmServer.mockReset().mockImplementation(async () => lease());
  server.getLlmServerStatus.mockReset().mockResolvedValue({ model: "mt/hy-mt2-1.8b-q4_k_m" });
  server.listLlmModels.mockReset().mockResolvedValue([model()]);
});

afterEach(() => {
  vi.useRealTimers();
});

describe("bundled llama-server", () => {
  it("acquires the server for the configured model and translates with the catalog's prompt family", async () => {
    const { e, statuses } = await startedEngine();
    expect(server.acquireLlmServer).toHaveBeenCalledWith(
      LLM_HOLDER,
      { model: "mt/hy-mt2-1.8b-q4_k_m", variant: "auto", ctxSize: 4096, gpuLayers: null, flashAttn: "auto" },
      expect.any(AbortSignal),
    );
    expect(statuses).toEqual(["transl.llm_status_starting", ""]);
    await expect(e.translate({ text: "Good morning", kind: "final", from: "en", to: "ja" }, live())).resolves.toBe("おはよう");
    const [req, opts] = client.chat.mock.calls[0];
    expect(req.messages).toEqual([
      {
        role: "user",
        content:
          "Translate the following text into Japanese. Note that you should only output the translated result without any additional explanation:\n\nGood morning",
      },
    ]);
    expect(req).toMatchObject({ temperature: 0.2, topK: 20, topP: 0.6, repetitionPenalty: 1.05, seed: LLM_SEED, maxTokens: 512, stream: false, timeoutMs: 15_000 });
    expect(req.reasoningEffort).toBe("default");
    expect(opts.onDelta).toBeUndefined();
  });

  it("streams cleaned partials of a final", async () => {
    const { e } = await startedEngine();
    client.chat.mockImplementation(async (_req, opts) => {
      opts.onDelta?.({ text: "Japanese: おは", delta: "" });
      opts.onDelta?.({ text: "Japanese: おはよう", delta: "" });
      return result("Japanese: おはようございます");
    });
    const partials: string[] = [];
    const text = await e.translate({ text: "Good morning", kind: "final", onPartial: (p) => partials.push(p) }, live());
    expect(text).toBe("おはようございます");
    expect(partials).toEqual(["おは", "おはよう"]);
    expect(client.chat.mock.calls[0][0].stream).toBe(true);
  });

  it("gives interims a shorter deadline and never streams them", async () => {
    const { e } = await startedEngine();
    await e.translate({ text: "Good", kind: "interim", onPartial: () => {} }, live());
    expect(client.chat.mock.calls[0][0]).toMatchObject({ stream: false, timeoutMs: LLM_INTERIM_TIMEOUT_MS });
  });

  it("sends raw completions for raw families", async () => {
    server.listLlmModels.mockResolvedValue([model({ id: "mt/milmmt-46-4b-q4_k_m", promptFormat: "milmmt" })]);
    const { e } = await startedEngine((s) => (s.llm.serverModel = "mt/milmmt-46-4b-q4_k_m"));
    await e.translate({ text: "Good morning", kind: "final" }, live());
    expect(client.chat).not.toHaveBeenCalled();
    expect(client.complete.mock.calls[0][0].prompt).toBe("Translate this from English to Japanese:\nEnglish: Good morning\nJapanese:");
  });

  it("guesses the family from the model name when the catalog does not answer", async () => {
    server.listLlmModels.mockRejectedValue(new Error("no catalog"));
    const { e } = await startedEngine((s) => (s.llm.serverModel = "C:/models/Seed-X-PPO-7B.Q4_K_M.gguf"));
    await e.translate({ text: "Hi", kind: "final" }, live());
    expect(client.complete.mock.calls[0][0].prompt.endsWith("Hi <ja>")).toBe(true);
  });

  it("fails the start with the ids to install when the model is missing", async () => {
    server.acquireLlmServer.mockRejectedValue({ kind: "not_installed", message: "missing", missing: ["mt/hy-mt2-1.8b-q4_k_m", "runtime/llama-server-vulkan"] });
    const err = await new Translation_LlmService().start(state(), live()).catch((x) => x);
    expect(err).toBeInstanceOf(TranslationError);
    expect(err.message).toBe("transl.llm_error_not_installed");
    expect(err.missing).toEqual(["mt/hy-mt2-1.8b-q4_k_m", "runtime/llama-server-vulkan"]);
    const refused = await new Translation_LlmService().start(state(), live()).catch((x) => x);
    expect(refused.missing).toHaveLength(2);
    server.acquireLlmServer.mockRejectedValue({ kind: "refused", message: "VC running on 8 GB", missing: [] });
    await expect(new Translation_LlmService().start(state(), live())).rejects.toThrow("transl.llm_error_refused");
  });

  it("releases a lease that arrives after the start was aborted", async () => {
    const l = lease();
    let give!: () => void;
    server.acquireLlmServer.mockImplementation(() => new Promise((r) => (give = () => r(l))));
    const c = new AbortController();
    const p = new Translation_LlmService().start(state(), c.signal);
    c.abort(new Error("stopped"));
    give();
    await expect(p).rejects.toThrow("stopped");
    expect(l.release).toHaveBeenCalled();
  });

  it("retries a final on the refreshed endpoint after a connection failure", async () => {
    const l = lease();
    server.acquireLlmServer.mockResolvedValue(l);
    const { e } = await startedEngine();
    l.refresh.mockResolvedValue({ ...l.endpoint, baseUrl: "http://127.0.0.1:6000/v1" });
    client.chat.mockRejectedValueOnce(new LlmError("network", "connection refused")).mockResolvedValueOnce(result("もう一度"));
    await expect(e.translate({ text: "Again", kind: "final" }, live())).resolves.toBe("もう一度");
    expect(endpoints.at(-1)?.baseUrl).toBe("http://127.0.0.1:6000/v1");
    expect(server.acquireLlmServer).toHaveBeenCalledTimes(1);
  });

  it("re-acquires when the server now serves another model, but does not restart a stopped server", async () => {
    const l = lease("hy-mt2");
    server.acquireLlmServer.mockResolvedValueOnce(l).mockResolvedValueOnce(lease("hy-mt2", 7000));
    const { e } = await startedEngine();
    l.refresh.mockResolvedValue({ baseUrl: "http://127.0.0.1:5000/v1", apiKey: "k", model: "other-model" });
    client.chat.mockRejectedValueOnce(new LlmError("network", "reset")).mockResolvedValueOnce(result("戻った"));
    await expect(e.translate({ text: "Back", kind: "final" }, live())).resolves.toBe("戻った");
    expect(server.acquireLlmServer).toHaveBeenCalledTimes(2);
    expect(l.release).toHaveBeenCalled();

    const stopped = lease();
    server.acquireLlmServer.mockResolvedValueOnce(stopped);
    const second = await startedEngine();
    stopped.refresh.mockRejectedValue(new Error("not running"));
    client.chat.mockRejectedValueOnce(new LlmError("network", "refused"));
    await expect(second.e.translate({ text: "x", kind: "final" }, live())).rejects.toThrow("transl.llm_error_server_gone");
    expect(server.acquireLlmServer).toHaveBeenCalledTimes(3);
  });

  it("does not retry interims after a connection failure", async () => {
    const { e } = await startedEngine();
    client.chat.mockRejectedValueOnce(new LlmError("network", "refused"));
    const err = await e.translate({ text: "x", kind: "interim" }, live()).catch((x) => x);
    expect(err).toBeInstanceOf(TranslationError);
    expect(err.message).toBe("transl.llm_error_unreachable");
    expect(client.chat).toHaveBeenCalledTimes(1);
  });

  it("answers supports() from the catalog's languages", async () => {
    server.listLlmModels.mockResolvedValue([model({ languages: ["ja", "en"] })]);
    const { e } = await startedEngine();
    expect(e.supports("en", "ja")).toBe(true);
    expect(e.supports("", "ja")).toBe(true);
    expect(e.supports("en", "ko")).toBe(false);
  });

  it("releases the lease on stop", async () => {
    const l = lease();
    server.acquireLlmServer.mockResolvedValue(l);
    const { e } = await startedEngine();
    e.stop();
    expect(l.release).toHaveBeenCalledTimes(1);
    await expect(e.translate({ text: "x", kind: "final" }, live())).rejects.toThrow("transl.error_not_running");
  });
});

describe("cloud and custom providers", () => {
  it("uses the generic fenced prompt for cloud models, without a seed or raw completions", async () => {
    const { e } = await startedEngine((s) => {
      s.llm.provider = "gemini";
      s.llm.geminiKey = "g";
    });
    expect(server.acquireLlmServer).not.toHaveBeenCalled();
    expect(endpoints[0]).toMatchObject({ provider: "gemini", keyHeader: "x-goog-api-key", model: "gemini-3.5-flash-lite" });
    await e.translate({ text: "Hello", kind: "final", glossary: [{ source: "Hello", target: "やあ" }] }, live());
    const req = client.chat.mock.calls[0][0];
    expect(req.seed).toBeUndefined();
    expect(req.topK).toBeUndefined();
    expect(req.messages[0].role).toBe("system");
    expect(req.messages[1].content).toContain("<source>\nHello\n</source>");
    expect(req.messages[1].content).toContain("Hello → やあ");
  });

  it("fails the start without a key", async () => {
    await expect(new Translation_LlmService().start(state((s) => (s.llm.provider = "openai")), live())).rejects.toThrow("llm.error_key_missing");
  });

  it("maps a rejected key to a non-retryable error", async () => {
    const { e } = await startedEngine((s) => {
      s.llm.provider = "openrouter";
      s.llm.openRouterKey = "bad";
    });
    client.chat.mockRejectedValue(new LlmError("unauthorized", "401 invalid key", { status: 401 }));
    const err = await e.translate({ text: "x", kind: "final" }, live()).catch((x) => x);
    expect(err).toMatchObject({ message: "transl.llm_error_auth", retryable: false, status: 401 });
  });

  it("checks that a custom server is reachable, tolerating servers without /models", async () => {
    const custom = (s: Translation_State) => {
      s.llm.provider = "custom";
      s.llm.customUrl = "http://192.168.1.5:8080/v1";
      s.llm.customModel = "local";
    };
    client.listModels.mockRejectedValueOnce(new LlmError("bad_request", "404"));
    await expect(startedEngine(custom)).resolves.toBeDefined();
    client.listModels.mockRejectedValueOnce(new LlmError("network", "connection refused"));
    await expect(new Translation_LlmService().start(state(custom), live())).rejects.toThrow("transl.llm_error_unreachable");
  });

  it("passes a line already in the target language through", async () => {
    const { e } = await startedEngine();
    const request: TranslationRequest = { text: "おはよう", kind: "final", from: "ja", to: "ja" };
    await expect(e.translate(request, live())).resolves.toBe("おはよう");
    expect(client.chat).not.toHaveBeenCalled();
  });

  it("rejects an empty reply", async () => {
    const { e } = await startedEngine();
    client.chat.mockResolvedValue(result("  \n"));
    await expect(e.translate({ text: "x", kind: "final" }, live())).rejects.toThrow("transl.llm_error_empty");
  });

  it("propagates the caller's abort reason", async () => {
    const { e } = await startedEngine();
    client.chat.mockImplementation((_req, opts) => new Promise((_r, reject) => opts.signal.addEventListener("abort", () => reject(opts.signal.reason))));
    const c = new AbortController();
    const p = e.translate({ text: "x", kind: "interim" }, c.signal);
    const reason = new Error("superseded");
    c.abort(reason);
    await expect(p).rejects.toBe(reason);
  });
});
