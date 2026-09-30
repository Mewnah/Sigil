import { createLlmClient, LlmClientDeps } from "@/core/services/llm/client";
import { ILlmClient, LlmEndpoint } from "@/core/services/llm/types";
import { devLog } from "@/utils/devLog";
import i18n from "i18next";
import { Transform_State } from "../schema";
import { ITransformEngine, TransformRequest, TransformRequestError } from "../types";
import { buildChatRequest, resolveTransformEndpoint, toTransformError } from "./llmRequest";

/**
 * OpenAI-compatible chat backend (OpenAI, OpenRouter, Gemini, or any custom server) on the shared LLM client.
 * Settings are read on every request, so key, URL, model, prompt, token limit, reasoning, streaming and transport
 * edits apply to the next line without a restart; the client is rebuilt only when the connection changes.
 *
 * Streaming: with `stream` on, `request.onPartial` receives the sanitized reply as it arrives. A provider that refuses
 * streamed requests (OpenAI for some models of unverified organizations) gets the line again without streaming, and
 * plain requests from then on (the shared client remembers it until a restart or a connection change). Aborting
 * `signal` (timeout, stop) cancels the HTTP request itself; the promise then rejects with the signal's reason.
 */
export class Transform_LLMService implements ITransformEngine {
    #getData: (() => Transform_State) | null = null;
    #client: ILlmClient | null = null;
    #clientKey = "";

    constructor(
        private readonly clientDeps: LlmClientDeps = {},
        private readonly resolveOptions: { inTauri?: boolean } = {},
    ) {}

    async start(getData: () => Transform_State): Promise<void> {
        const endpoint = resolveTransformEndpoint(getData(), this.resolveOptions);
        devLog(`[LLM] transform starting (${endpoint.provider}, ${endpoint.transport}), api key configured: ${endpoint.apiKey !== ""}`);
        this.#getData = getData;
    }

    stop(): void {
        this.#getData = null;
        this.#client = null;
        this.#clientKey = "";
    }

    async transform(request: TransformRequest, signal: AbortSignal): Promise<string> {
        const getData = this.#getData;
        if (!getData) throw new TransformRequestError("request", i18n.t("transform.error_not_running"));
        const data = getData();
        const endpoint = resolveTransformEndpoint(data, this.resolveOptions);
        const chat = buildChatRequest(data, request);
        const onPartial = request.onPartial;
        try {
            const result = await this.#clientFor(endpoint).chat(chat, {
                signal,
                onDelta:
                    chat.stream && onPartial
                        ? (delta) => {
                              if (!signal.aborted && delta.text) onPartial(delta.text);
                          }
                        : undefined,
            });
            return result.text;
        } catch (error) {
            if (signal.aborted) throw signal.reason ?? error;
            throw toTransformError(error);
        }
    }

    #clientFor(endpoint: LlmEndpoint): ILlmClient {
        const key = JSON.stringify(endpoint);
        if (!this.#client || key !== this.#clientKey) {
            this.#client = createLlmClient(endpoint, this.clientDeps);
            this.#clientKey = key;
        }
        return this.#client;
    }
}
