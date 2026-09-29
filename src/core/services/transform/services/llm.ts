import OpenAI from "openai";
import type { ChatCompletionCreateParamsNonStreaming } from "openai/resources/chat/completions";
import { devLog } from "@/utils/devLog";
import i18n from "i18next";
import { Transform_State } from "../schema";
import { ITransformEngine, TransformRequest, TransformRequestError } from "../types";
import {
    buildChatRequest,
    classifyLlmError,
    cleanLlmOutput,
    contentText,
    ResolvedProvider,
    resolveProvider,
} from "./llmRequest";

/**
 * OpenAI-compatible chat backend (OpenAI, OpenRouter, or any custom server). Settings are read on
 * every request, so key, URL, model, prompt, token limit and reasoning edits apply to the next line
 * without a restart; the SDK client is rebuilt only when the connection settings change.
 */
export class Transform_LLMService implements ITransformEngine {
    #getData: (() => Transform_State) | null = null;
    #client: OpenAI | null = null;
    #clientKey = "";

    async start(getData: () => Transform_State): Promise<void> {
        const provider = resolveProvider(getData());
        devLog(`[LLM] starting (${provider.provider}), api key configured: ${provider.apiKey !== ""}`);
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
        const provider = resolveProvider(data);
        const body = buildChatRequest(data, provider, request) as unknown as ChatCompletionCreateParamsNonStreaming;

        let completion: OpenAI.Chat.Completions.ChatCompletion;
        try {
            completion = await this.#clientFor(provider).chat.completions.create(body, { signal });
        } catch (error) {
            if (signal.aborted) throw signal.reason ?? error;
            throw classifyLlmError(error);
        }

        const choice = completion?.choices?.[0];
        const text = cleanLlmOutput(contentText(choice?.message?.content));
        if (!text) {
            if (choice?.finish_reason === "length") {
                throw new TransformRequestError("truncated", `finish_reason=length (max tokens ${body.max_tokens ?? body.max_completion_tokens})`);
            }
            throw new TransformRequestError("empty", `no text returned by ${provider.model}`);
        }
        return text;
    }

    #clientFor(p: ResolvedProvider): OpenAI {
        const key = JSON.stringify([p.baseURL, p.apiKey, p.headers]);
        if (!this.#client || key !== this.#clientKey) {
            this.#client = new OpenAI({
                apiKey: p.apiKey,
                baseURL: p.baseURL,
                dangerouslyAllowBrowser: true,
                defaultHeaders: p.headers,
                // Retries are the service's call: a caption line is useless after its deadline.
                maxRetries: 0,
            });
            this.#clientKey = key;
        }
        return this.#client;
    }
}
