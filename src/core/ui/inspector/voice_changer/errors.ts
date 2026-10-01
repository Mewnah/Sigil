// Voice-changer inspector: translated slot-store / importer errors (moved out of VoiceLibrary.tsx by P5-G so the
// family sections can use it without the voice library's imports).
import { importErrorCode } from "@/core/services/voice_changer/ipc";
import type { TFunction } from "i18next";

/** A translated explanation of an importer / slot-store error (`"<code>: <message>"`), plus the message. */
export function importErrorText(t: TFunction, error: unknown): string {
    const text = typeof error === "string" ? error : error instanceof Error ? error.message : String(error);
    const code = importErrorCode(text);
    if (!code) return text;
    const message = text.slice(code.length + 1).trim();
    const explanation = t(`voice_changer.import_error_${code}`);
    return message ? `${explanation} (${message})` : explanation;
}
