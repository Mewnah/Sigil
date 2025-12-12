import OpenAI from "openai";
import { Service_Transform_Schema } from "../schema";
import { ITransformReceiver, ITransformService } from "../types";
import { TextEvent } from "@/types";
import { z } from "zod";

type TransformData = z.infer<typeof Service_Transform_Schema>;

export class Transform_OpenAIService implements ITransformService {
    private openai: OpenAI | null = null;
    private systemPrompt: string = "";
    private model: string = "";

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
            this.receiver.onStop(`API Key missing for ${data.provider}`);
            return;
        }

        try {
            this.openai = new OpenAI({
                apiKey: apiKey,
                baseURL: baseURL || undefined, // undefined uses default
                dangerouslyAllowBrowser: true
            });
            this.systemPrompt = data.systemPrompt;
            this.model = model;
            this.receiver.onStart();
        } catch (error: any) {
            this.receiver.onStop(`Failed to initialize: ${error.message}`);
        }
    }

    stop(): void {
        this.openai = null;
        this.receiver.onStop();
    }

    async transform(id: number, e: TextEvent, history: any[]): Promise<void> {
        if (!this.openai || !e.value.trim()) return;

        try {
            const response = await this.openai.chat.completions.create({
                model: this.model || "gpt-3.5-turbo",
                messages: [
                    { role: "system", content: this.systemPrompt },
                    ...history,
                    { role: "user", content: e.value },
                ],
                max_tokens: 100,
            });

            const result = response.choices[0]?.message?.content?.trim();
            if (result) {
                this.receiver.onTransform(id, e, result);
            }
        } catch (error: any) {
            console.error("OpenAI Transform Error:", error);
        }
    }
}
