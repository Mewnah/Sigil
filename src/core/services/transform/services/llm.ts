import OpenAI from "openai";
import { Service_Transform_Schema } from "../schema";
import { ITransformReceiver, ITransformService } from "../types";
import { TextEvent } from "@/types";
import { z } from "zod";

type TransformData = z.infer<typeof Service_Transform_Schema>;

export class Transform_LLMService implements ITransformService {
    private openai: OpenAI | null = null;
    private systemPrompt: string = "";
    private model: string = "gpt-3.5-turbo";

    constructor(private receiver: ITransformReceiver) { }

    start(data: TransformData): void {
        let apiKey = "";
        let baseURL = "";
        let model = "";

        // Resolve Config based on Provider
        switch (data.provider) {
            case "openrouter":
                apiKey = data.openRouterKey || data.openaiKey; // fallback for legacy
                baseURL = data.openRouterUrl;
                model = data.openRouterModel;
                break;
            case "custom":
                apiKey = data.customKey;
                baseURL = data.customUrl;
                model = data.customModel;
                break;
            case "openai":
            default:
                apiKey = data.openaiKey;
                baseURL = data.openaiUrl;
                model = data.openaiModel;
                break;
        }

        if (!apiKey) {
            // Allow dummy key if custom (local LLM might not need it, but SDK requires string)
            if (data.provider !== "custom") {
                this.receiver.onStop(`API Key missing for ${data.provider}`);
                return;
            }
            apiKey = "dummy";
        }

        console.log(`[LLM] STARTING. Key Present: ${apiKey !== "dummy"} (${apiKey.substring(0, 8)}...)`);

        const defaultHeaders: Record<string, string> = {
            "Authorization": `Bearer ${apiKey}`, // Ensure Auth header is explicit
        };

        if (data.provider === "openrouter") {
            defaultHeaders["HTTP-Referer"] = "https://github.com/mmpneo/curses"; // Using repo URL as referer
            defaultHeaders["X-Title"] = "Curses+";
        }

        try {
            this.openai = new OpenAI({
                apiKey: apiKey,
                baseURL: baseURL || undefined,
                dangerouslyAllowBrowser: true,
                defaultHeaders: defaultHeaders,
                maxRetries: 0 // Disable auto-retry to handle rate limits manually
            });
            this.systemPrompt = data.systemPrompt;
            this.model = model;

            this.receiver.onStart();
        } catch (error: any) {
            this.receiver.onStop(`Failed to initialize LLM: ${error.message}`);
        }
    }

    stop(): void {
        this.openai = null;
        this.receiver.onStop();
    }

    async transform(id: number, e: TextEvent): Promise<void> {
        if (!this.openai || !e.value.trim()) return;

        try {
            const response = await this.openai.chat.completions.create({
                model: this.model,
                messages: [
                    { role: "system", content: this.systemPrompt },
                    { role: "user", content: e.value },
                ],
                max_tokens: 100,
            });

            if (!response || !response.choices || response.choices.length === 0) {
                throw new Error(`Invalid response from ${this.model}: No choices returned.`);
            }

            let result = response.choices[0]?.message?.content?.trim();
            if (result) {
                // Clean common raw tokens that might leak
                result = result.replace(/^<s>/, "").replace(/<\/s>$/, "").trim();
                this.receiver.onTransform(id, e, result);
            }
        } catch (error: any) {
            const message = error?.message || String(error);
            const status = error?.status || "Unknown";

            console.error(`[LLM] Transform Error (${status}): ${message}`);

            if (error?.status === 429) {
                this.receiver.onTransform(id, e, "[AI Rate Limited]");
            }
        }
    }
}
