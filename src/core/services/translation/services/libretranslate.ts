import { invoke } from "@tauri-apps/api/core";
import { TextEvent } from "@/types";
import {
    ITranslationReceiver,
    ITranslationService,
} from "../types";
import { Translation_State } from "../schema";

interface LibreTranslateLanguage {
    code: string;
    name: string;
}

export class Translation_LibreTranslateService implements ITranslationService {
    #receiver: ITranslationReceiver;
    #endpoint: string = "http://localhost:5000";
    #sourceLanguage: string = "en";
    #targetLanguage: string = "es";
    #autoDetect: boolean = false;
    #isRunning: boolean = false;

    constructor(receiver: ITranslationReceiver) {
        this.#receiver = receiver;
    }

    async start(data: Translation_State): Promise<void> {
        const config = data.libretranslate;
        this.#endpoint = config.endpoint;
        this.#sourceLanguage = config.languageFrom;
        this.#targetLanguage = config.language;
        this.#autoDetect = config.autoDetect;

        try {
            // Set the endpoint in the backend
            await invoke("plugin:translate|set_endpoint", {
                endpoint: this.#endpoint,
            });

            // Check if LibreTranslate is available
            const isAvailable = await invoke<boolean>("plugin:translate|check_availability");
            if (!isAvailable) {
                this.#receiver.onStop(
                    `LibreTranslate not available at ${this.#endpoint}. Please ensure the service is running.`
                );
                return;
            }

            this.#isRunning = true;
            this.#receiver.onStart();
        } catch (error) {
            this.#receiver.onStop(`Failed to connect to LibreTranslate: ${error}`);
        }
    }

    stop(): void {
        this.#isRunning = false;
        this.#receiver.onStop("");
    }

    dispose(): void {
        this.stop();
    }

    async translate(id: number, event: TextEvent): Promise<void> {
        if (!this.#isRunning || !event.value) return;

        try {
            let sourceLanguage = this.#sourceLanguage;

            // Auto-detect source language if enabled
            if (this.#autoDetect) {
                try {
                    sourceLanguage = await invoke<string>("plugin:translate|detect_language", {
                        text: event.value,
                    });
                } catch {
                    // Fall back to configured source language on detection failure
                    sourceLanguage = this.#sourceLanguage;
                }
            }

            const translated = await invoke<string>("plugin:translate|translate", {
                text: event.value,
                source: sourceLanguage,
                target: this.#targetLanguage,
            });

            this.#receiver.onTranslation(id, event, translated);
        } catch (error) {
            console.error("LibreTranslate translation error:", error);
            // Don't stop the service on translation errors, just log them
        }
    }

    // Static helper to fetch available languages
    static async getLanguages(endpoint: string = "http://localhost:5000"): Promise<LibreTranslateLanguage[]> {
        try {
            await invoke("plugin:translate|set_endpoint", { endpoint });
            return await invoke<LibreTranslateLanguage[]>("plugin:translate|get_languages");
        } catch {
            return [];
        }
    }
}
