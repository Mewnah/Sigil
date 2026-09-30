import i18n from "i18next";
import { isEmptyValue } from "@/utils";
import { isTauri } from "@tauri-apps/api/core";
import { STT_State } from "../schema";
import { baseLanguageTag } from "../rustCapturePolling";
import {
  ISTTReceiver,
  ISTTService,
  SttTextMeta
} from "../types";

/**
 * Stop message for a Web Speech `error` event (`SpeechRecognitionErrorEvent.error`); `null` for errors that do not
 * end the session (`no-speech`: a silent stretch). Pure.
 */
export function nativeErrorMessage(code: string | undefined, language: string, desktop: boolean): string | null {
  switch (code) {
    case "no-speech":
      return null;
    case "network":
      // WebView2 has no speech service: the desktop app always ends up here.
      return desktop ? i18n.t("stt.error_native_desktop") : i18n.t("stt.error_native_network");
    case "not-allowed":
    case "service-not-allowed":
      return i18n.t("stt.error_native_not_allowed");
    case "audio-capture":
      return i18n.t("stt.error_native_mic");
    case "language-not-supported":
      return i18n.t("stt.error_native_language", { language });
    default:
      return i18n.t("stt.error_native_generic", { detail: code || "error" });
  }
}

export class STT_NativeService implements ISTTService {
  constructor(private bindings: ISTTReceiver) {}

  #instance?: SpeechRecognition;
  /** WebView2 often omits `onstart`; Tauri fallback clears STT connecting timeout. */
  #nativeStartFallback?: number;
  #nativeConnectSent = false;
  /** When false, `onend` must not call `start()` again (user stopped or is switching backend). */
  #shouldRestartAfterEnd = true;
  #meta: SttTextMeta | undefined;
  #onUnload?: () => void;

  dispose(): void {}

  #clearNativeStartFallback(): void {
    if (this.#nativeStartFallback) {
      clearTimeout(this.#nativeStartFallback);
      this.#nativeStartFallback = undefined;
    }
  }

  #clearUnload(): void {
    if (this.#onUnload) {
      window.removeEventListener("beforeunload", this.#onUnload);
      this.#onUnload = undefined;
    }
  }

  /** One `onFinal` / `onInterim` per `onresult` event to avoid duplicate growing finals from the same batch. */
  #processResults = (event: any) => {
    let interim_transcript = "";
    let final_transcript = "";
    for (let i = event.resultIndex; i < event.results.length; ++i) {
      const row = event.results[i];
      if (row.isFinal) {
        final_transcript += row[0].transcript;
      } else {
        interim_transcript += row[0].transcript;
      }
    }
    if (final_transcript.trim()) {
      this.bindings.onFinal(final_transcript, this.#meta);
    }
    if (interim_transcript.trim()) {
      this.bindings.onInterim(interim_transcript, this.#meta);
    }
  };

  start(state: STT_State): void {
    this.#clearNativeStartFallback();
    this.#nativeConnectSent = false;
    this.#shouldRestartAfterEnd = true;

    if (Object.values(state.native).some(isEmptyValue)) {
      this.bindings.onStop(i18n.t("stt.error_native_options"));
      return;
    }

    const SpCtor =
      window.webkitSpeechRecognition ||
      (window as unknown as { SpeechRecognition?: typeof window.webkitSpeechRecognition }).SpeechRecognition;
    if (!SpCtor) {
      this.bindings.onStop(i18n.t("stt.error_native_unavailable"));
      return;
    }

    const language = state.native.language;
    const lang = baseLanguageTag(language);
    this.#meta = lang ? { lang } : undefined;
    this.bindings.onStatus(i18n.t("stt.status_native_starting"));

    try {
      this.#instance = new SpCtor();
      this.#instance.lang = language;
      this.#instance.continuous = true;
      this.#instance.interimResults = true;

      const notifyStart = () => {
        if (this.#nativeConnectSent)
          return;
        this.#nativeConnectSent = true;
        this.#clearNativeStartFallback();
        this.bindings.onStart();
      };
      this.#instance.onstart = () => notifyStart();
      this.#instance.onresult = (event: unknown) => this.#processResults(event);

      this.#instance.addEventListener("error", (error: unknown) => {
        const message = nativeErrorMessage((error as { error?: string }).error, language, isTauri());
        if (message === null) return;
        this.stop(message);
      });
      this.#instance.onend = () => {
        if (!this.#shouldRestartAfterEnd || !this.#instance) return;
        try {
          this.#instance.start();
        } catch {
          /* ignore — recognition may be tearing down */
        }
      };
      this.#instance.start();
      if (isTauri()) {
        this.#nativeStartFallback = window.setTimeout(() => notifyStart(), 800);
      }
    } catch (e) {
      this.#clearNativeStartFallback();
      this.#instance = undefined;
      this.bindings.onStatus("");
      this.bindings.onStop(i18n.t("stt.error_native_generic", { detail: e instanceof Error ? e.message : String(e) }));
      return;
    }

    this.#clearUnload();
    const instance = this.#instance;
    this.#onUnload = () => instance?.stop();
    window.addEventListener("beforeunload", this.#onUnload);
  }

  stop(error?: string): void {
    this.#clearNativeStartFallback();
    this.#clearUnload();
    this.#shouldRestartAfterEnd = false;
    const instance = this.#instance;
    this.#instance = undefined;
    if (!instance)
      return;
    instance.onend = null;
    instance.stop();
    if (error) this.bindings.onStatus("");
    this.bindings.onStop(error);
  }
}
