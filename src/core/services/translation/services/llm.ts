import { createLlmClient } from "@/core/services/llm/client";
import { resolveLlmEndpoint } from "@/core/services/llm/providers";
import {
  ILlmClient,
  LlmConnectionSettings,
  LlmDelta,
  LlmError,
  LlmErrorKind,
  LlmResult,
  LlmSampling,
} from "@/core/services/llm/types";
import { acquireLlmServer, getLlmServerStatus, listLlmModels, LlmServerLease } from "@/core/services/llm_server";
import type { RustLlmModelInfo, RustLlmServerConfig, RustLlmServerEndpoint, RustLlmServerError } from "@/types";
import i18n from "i18next";
import { raceWithSignal } from "../abort";
import { canonicalLanguage } from "../languages";
import {
  buildTranslationPrompt,
  cleanTranslationOutput,
  PROMPT_FORMAT_SAMPLING,
  resolvePromptFormat,
  ResolvedPromptFormat,
} from "../prompt";
import { Translation_State } from "../schema";
import { ITranslationEngine, TranslationEngineHost, TranslationError, TranslationRequest } from "../types";

/**
 * LLM translation engine (`llm` backend): the shared OpenAI-compatible client (`@/core/services/llm`, P4-A) against
 * the bundled llama-server (a lease from `@/core/services/llm_server`, P4-B), Gemini, OpenRouter, OpenAI or a custom
 * server; prompts from `../prompt.ts` (per model family, source fenced as data, rolling context and the line's
 * glossary). Finals stream through `request.onPartial` when `llm.stream` is on.
 *
 * - Bundled server: `start` acquires the shared server for the configured model (it downloads nothing; a missing
 *   model or runtime fails the start with `TranslationError.missing` = the catalog ids to install). The prompt
 *   family comes from the catalog (`list_llm_models` `promptFormat`) unless set explicitly.
 * - When the server moved (helper restart: new port) a final is retried once on the refreshed endpoint; when it
 *   serves another model now (another consumer restarted it) the lease is re-acquired once per 10 s; when it is
 *   gone (the user stopped it) the line fails, so the router falls through instead of restarting it.
 * - Interims use a shorter deadline; llama.cpp and custom servers get a fixed seed, so the same partial translates
 *   the same way (steadier LocalAgreement).
 *
 * Owned by P4-C.
 */

type LlmSlice = Translation_State["llm"];

/** Lease holder name shown in the server panel. */
export const LLM_HOLDER = "translation";
/** Fixed sampling seed for local servers (deterministic partials). */
export const LLM_SEED = 1234;
/** Deadline cap of one interim request (ms). */
export const LLM_INTERIM_TIMEOUT_MS = 8_000;
/** Reachability probe of a custom server at start (ms). */
export const LLM_CUSTOM_PROBE_MS = 5_000;
/** Minimum gap between two re-acquisitions of the bundled server (ms). */
export const LLM_REACQUIRE_GAP_MS = 10_000;

/** Connection settings of the slice's provider (`bundled` needs the running server's endpoint). */
export function llmConnection(cfg: LlmSlice, bundled?: RustLlmServerEndpoint | null): LlmConnectionSettings {
  switch (cfg.provider) {
    case "bundled":
      return { provider: "bundled", model: "", apiKey: "", baseUrl: "", transport: cfg.transport, bundled: bundled ?? null };
    case "gemini":
      return { provider: "gemini", model: cfg.geminiModel.trim(), apiKey: cfg.geminiKey.trim(), baseUrl: "", transport: cfg.transport };
    case "openrouter":
      return { provider: "openrouter", model: cfg.openRouterModel.trim(), apiKey: cfg.openRouterKey.trim(), baseUrl: "", transport: cfg.transport };
    case "openai":
      return { provider: "openai", model: cfg.openaiModel.trim(), apiKey: cfg.openaiKey.trim(), baseUrl: "", transport: cfg.transport };
    case "custom":
    default:
      return { provider: "custom", model: cfg.customModel.trim(), apiKey: cfg.customKey.trim(), baseUrl: cfg.customUrl.trim(), transport: cfg.transport };
  }
}

/** Server config for the slice (`RustLlmServerConfig`). */
export function llmServerConfig(cfg: LlmSlice): RustLlmServerConfig {
  return {
    model: cfg.serverModel.trim() || "auto",
    variant: cfg.serverVariant,
    ctxSize: cfg.serverContext,
    gpuLayers: cfg.serverGpuLayers < 0 ? null : cfg.serverGpuLayers,
    flashAttn: cfg.serverFlashAttn,
  };
}

export function isRustLlmServerError(e: unknown): e is RustLlmServerError {
  return (
    typeof e === "object" &&
    e !== null &&
    typeof (e as { kind?: unknown }).kind === "string" &&
    typeof (e as { message?: unknown }).message === "string"
  );
}

/** A failed LLM translation that keeps the client's error kind (the engine retries connection failures once). */
export class LlmTranslationError extends TranslationError {
  constructor(
    message: string,
    readonly kind: LlmErrorKind,
    options: { status?: number; retryable?: boolean } = {},
  ) {
    super(message, options);
    this.name = "LlmTranslationError";
  }
}

function messageOf(e: unknown): string {
  if (e instanceof Error) return e.message;
  if (isRustLlmServerError(e)) return e.message;
  return e === undefined || e === null ? "" : String(e);
}

/** Maps a client failure to a line error with a user-facing message. */
export function llmTranslationError(e: unknown): TranslationError {
  if (e instanceof TranslationError) return e;
  if (e instanceof LlmError) {
    const error = e.message;
    switch (e.kind) {
      case "unauthorized":
        return new LlmTranslationError(i18n.t("transl.llm_error_auth", { error }), e.kind, { status: e.status, retryable: false });
      case "rate_limited":
        return new LlmTranslationError(i18n.t("transl.llm_error_rate_limited", { error }), e.kind, { status: e.status, retryable: true });
      case "not_configured":
        return new LlmTranslationError(error, e.kind, { retryable: false });
      case "network":
        return new LlmTranslationError(i18n.t("transl.llm_error_unreachable", { error }), e.kind, { status: e.status, retryable: true });
      default:
        return new LlmTranslationError(i18n.t("transl.llm_error_request", { error }), e.kind, { status: e.status, retryable: e.retryable });
    }
  }
  return new TranslationError(i18n.t("transl.llm_error_request", { error: messageOf(e) }), { retryable: true });
}

/** Start failure of the bundled server, worded for the user (`missing` ids for the install UI). */
export function llmServerStartError(e: unknown): TranslationError {
  if (isRustLlmServerError(e)) {
    const missing = Array.isArray(e.missing) ? e.missing.filter((id) => typeof id === "string") : [];
    switch (e.kind) {
      case "not_installed":
        return new TranslationError(i18n.t("transl.llm_error_not_installed", { ids: missing.join(", ") || e.message }), {
          retryable: false,
          missing,
        });
      case "refused":
        return new TranslationError(i18n.t("transl.llm_error_refused", { reason: e.message }), { retryable: false, missing });
      default:
        return new TranslationError(i18n.t("transl.llm_error_server", { error: e.message }), { retryable: false, missing });
    }
  }
  return new TranslationError(i18n.t("transl.llm_error_server", { error: messageOf(e) }), { retryable: false });
}

/** Catalog row of the served model: the configured id, or for `auto` the governor's pick / the running model. */
export async function servedModelInfo(serverModel: string): Promise<RustLlmModelInfo | undefined> {
  const models = await listLlmModels();
  let id = serverModel.trim();
  if (!id || id === "auto") {
    id = models.find((m) => m.recommended)?.id ?? "";
    if (!id) id = (await getLlmServerStatus()).model ?? "";
  }
  return models.find((m) => m.id === id);
}

/** Providers whose endpoint accepts a raw completion (`LlmCompletionRequest`). */
function rawAvailable(provider: LlmSlice["provider"]): boolean {
  return provider === "bundled" || provider === "custom" || provider === "openrouter";
}

function linkedTimeout(signal: AbortSignal, ms: number): { signal: AbortSignal; dispose(): void } {
  const c = new AbortController();
  const onAbort = () => c.abort(signal.reason);
  if (signal.aborted) c.abort(signal.reason);
  else signal.addEventListener("abort", onAbort, { once: true });
  const timer = setTimeout(() => c.abort(new DOMException("Timed out", "TimeoutError")), ms);
  return {
    signal: c.signal,
    dispose: () => {
      clearTimeout(timer);
      signal.removeEventListener("abort", onAbort);
    },
  };
}

export class Translation_LlmService implements ITranslationEngine {
  #cfg: LlmSlice | null = null;
  #pair = { from: "", to: "" };
  #client: ILlmClient | null = null;
  #lease: LlmServerLease | null = null;
  #format: ResolvedPromptFormat = "generic";
  /** Canonical languages of the served model (catalog); [] = broadly multilingual. */
  #languages: string[] = [];
  #host: TranslationEngineHost | null = null;
  #reacquiredAt = Number.NEGATIVE_INFINITY;
  #recovering: Promise<ILlmClient | null> | null = null;

  attach(host: TranslationEngineHost): void {
    this.#host = host;
  }

  async start(state: Translation_State, signal: AbortSignal): Promise<void> {
    const cfg = { ...state.llm };
    const to = canonicalLanguage(state.targetLanguage);
    if (!to) throw new TranslationError(i18n.t("transl.error_target_missing"), { retryable: false });
    this.#pair = { from: canonicalLanguage(state.sourceLanguage), to };

    let modelName = "";
    let catalogFormat: string | null = null;
    if (cfg.provider === "bundled") {
      this.#host?.status({ text: i18n.t("transl.llm_status_starting") || "" });
      let lease: LlmServerLease;
      try {
        lease = await acquireLlmServer(LLM_HOLDER, llmServerConfig(cfg), signal);
      } catch (e) {
        this.#host?.status({ text: "" });
        if (signal.aborted) throw signal.reason ?? e;
        throw llmServerStartError(e);
      }
      if (signal.aborted) {
        lease.release();
        throw signal.reason ?? new DOMException("Aborted", "AbortError");
      }
      this.#lease = lease;
      try {
        const info = await servedModelInfo(cfg.serverModel);
        if (info) {
          modelName = info.id;
          catalogFormat = info.promptFormat;
          this.#languages = info.languages.map((l) => canonicalLanguage(l)).filter((l) => l !== "");
        }
      } catch {
        // No catalog answer: the prompt family is guessed from the model name.
      }
      if (!modelName) modelName = cfg.serverModel.trim() === "auto" ? lease.endpoint.model : cfg.serverModel;
    }

    let client: ILlmClient;
    try {
      const endpoint = resolveLlmEndpoint(llmConnection(cfg, this.#lease?.endpoint));
      client = createLlmClient(endpoint);
      if (!modelName) modelName = endpoint.model;
    } catch (e) {
      this.#releaseLease();
      this.#host?.status({ text: "" });
      throw e instanceof LlmError ? new TranslationError(e.message || i18n.t("transl.llm_error_not_configured"), { retryable: false }) : llmTranslationError(e);
    }

    if (cfg.provider === "custom") await this.#probe(client, signal);
    if (signal.aborted) {
      this.#releaseLease();
      throw signal.reason ?? new DOMException("Aborted", "AbortError");
    }
    this.#format = resolvePromptFormat(cfg.promptFormat, { model: modelName, catalogFormat });
    this.#client = client;
    this.#cfg = cfg;
    this.#host?.status({ text: "" });
  }

  async translate(request: TranslationRequest, signal: AbortSignal): Promise<string> {
    const cfg = this.#cfg;
    const client = this.#client;
    if (!cfg || !client) throw new TranslationError(i18n.t("transl.error_not_running"), { retryable: false });
    const to = canonicalLanguage(request.to ?? this.#pair.to);
    const from = canonicalLanguage(request.from ?? this.#pair.from);
    if (!to) throw new TranslationError(i18n.t("transl.error_target_missing"), { retryable: false });
    // A line already in the target language is its own translation.
    if (from && from === to) return request.text;

    try {
      return await this.#call(client, cfg, request, from, to, signal);
    } catch (e) {
      if (signal.aborted) throw signal.reason ?? e;
      const connection = e instanceof LlmTranslationError && (e.kind === "network" || e.kind === "server");
      if (cfg.provider === "bundled" && request.kind === "final" && connection) {
        const recovered = await this.#recover(cfg, signal);
        if (recovered) return this.#call(recovered, cfg, request, from, to, signal);
        throw new TranslationError(i18n.t("transl.llm_error_server_gone"), { retryable: false });
      }
      throw e;
    }
  }

  stop(): void {
    this.#cfg = null;
    this.#client = null;
    this.#recovering = null;
    this.#releaseLease();
  }

  supports(from: string, to: string): boolean {
    if (this.#languages.length === 0) return true;
    const langs = this.#languages;
    const has = (code: string) => langs.includes(canonicalLanguage(code));
    return has(to) && (!from || has(from));
  }

  // --- internals ------------------------------------------------------------------------------------------------

  async #call(
    client: ILlmClient,
    cfg: LlmSlice,
    request: TranslationRequest,
    from: string,
    to: string,
    signal: AbortSignal,
  ): Promise<string> {
    const format = this.#format;
    const bundled = cfg.provider === "bundled";
    const built = buildTranslationPrompt(
      format,
      {
        text: request.text,
        from,
        to,
        context: request.context,
        glossary: request.glossary,
        background: request.background,
        customPrompt: cfg.customPrompt,
      },
      { rawAvailable: rawAvailable(cfg.provider), bundled },
    );
    const sampling: LlmSampling = { temperature: cfg.temperature };
    if (bundled) Object.assign(sampling, PROMPT_FORMAT_SAMPLING[format] ?? {});
    if (bundled || cfg.provider === "custom") sampling.seed = LLM_SEED;
    if (built.stop.length) sampling.stop = built.stop;

    const stream = cfg.stream && request.kind === "final" && typeof request.onPartial === "function";
    const timeoutMs = request.kind === "interim" ? Math.min(cfg.timeoutMs, LLM_INTERIM_TIMEOUT_MS) : cfg.timeoutMs;
    const cleanup = { source: request.text, to, format };
    const onDelta = stream
      ? (d: LlmDelta) => {
          const partial = cleanTranslationOutput(d.text, cleanup);
          if (partial) request.onPartial?.(partial);
        }
      : undefined;

    let result: LlmResult;
    try {
      result =
        built.mode === "chat"
          ? await client.chat(
              { ...sampling, messages: built.messages, maxTokens: cfg.maxTokens, reasoningEffort: cfg.reasoningEffort, stream, timeoutMs },
              { signal, onDelta },
            )
          : await client.complete({ ...sampling, prompt: built.prompt, maxTokens: cfg.maxTokens, stream, timeoutMs }, { signal, onDelta });
    } catch (e) {
      if (signal.aborted) throw signal.reason ?? e;
      throw llmTranslationError(e);
    }
    const text = cleanTranslationOutput(result.text, cleanup);
    if (!text) throw new LlmTranslationError(i18n.t("transl.llm_error_empty"), "empty", { retryable: false });
    return text;
  }

  /** A custom server must answer `/models` (or at least be reachable) before the start succeeds. */
  async #probe(client: ILlmClient, signal: AbortSignal): Promise<void> {
    const probe = linkedTimeout(signal, LLM_CUSTOM_PROBE_MS);
    try {
      await client.listModels(probe.signal);
    } catch (e) {
      if (signal.aborted) throw signal.reason ?? e;
      const kind = e instanceof LlmError ? e.kind : probe.signal.aborted ? "timeout" : "network";
      // Servers without `/models` still translate; only an unreachable server or a rejected key fails the start.
      if (kind === "network" || kind === "timeout" || kind === "unauthorized") {
        throw new TranslationError(i18n.t("transl.llm_error_unreachable", { error: messageOf(e) || kind }), { retryable: false });
      }
    } finally {
      probe.dispose();
    }
  }

  /**
   * After a connection failure: the refreshed endpoint when the server still runs our model (restart on a new port),
   * a re-acquired lease when it now serves another model, null when the server is gone.
   */
  #recover(cfg: LlmSlice, signal: AbortSignal): Promise<ILlmClient | null> {
    this.#recovering ??= this.#doRecover(cfg, signal).finally(() => {
      this.#recovering = null;
    });
    return raceWithSignal(this.#recovering, signal);
  }

  async #doRecover(cfg: LlmSlice, signal: AbortSignal): Promise<ILlmClient | null> {
    const lease = this.#lease;
    if (!lease) return null;
    let endpoint: RustLlmServerEndpoint;
    try {
      endpoint = await lease.refresh();
    } catch {
      return null; // stopped (by the user or the last other holder): do not start it again from here
    }
    if (this.#lease !== lease || this.#cfg !== cfg) return null;
    if (endpoint.model === lease.endpoint.model) return this.#useEndpoint(cfg, endpoint);

    if (Date.now() - this.#reacquiredAt < LLM_REACQUIRE_GAP_MS) return null;
    this.#reacquiredAt = Date.now();
    let fresh: LlmServerLease;
    try {
      fresh = await acquireLlmServer(LLM_HOLDER, llmServerConfig(cfg), signal);
    } catch {
      return null;
    }
    if (this.#lease !== lease || this.#cfg !== cfg) {
      fresh.release();
      return null;
    }
    this.#lease = fresh;
    lease.release();
    return this.#useEndpoint(cfg, fresh.endpoint);
  }

  #useEndpoint(cfg: LlmSlice, endpoint: RustLlmServerEndpoint): ILlmClient | null {
    try {
      const client = createLlmClient(resolveLlmEndpoint(llmConnection(cfg, endpoint)));
      this.#client = client;
      return client;
    } catch {
      return null;
    }
  }

  #releaseLease(): void {
    const lease = this.#lease;
    this.#lease = null;
    try {
      lease?.release();
    } catch (e) {
      console.error("[Translation] llm-server lease release failed:", e);
    }
  }
}
