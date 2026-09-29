import { isEmptyValue } from "@/utils";
import { isTauri } from "@tauri-apps/api/core";
import { STT_State } from "../schema";
import {
  ISTTReceiver,
  ISTTService
} from "../types";

export class STT_NativeService implements ISTTService {
  constructor(private bindings: ISTTReceiver) {}
  
  #instance?: SpeechRecognition;
  /** WebView2 often omits `onstart`; Tauri fallback clears STT connecting timeout. */
  #nativeStartFallback?: number;
  #nativeConnectSent = false;
  /** When false, `onend` must not call `start()` again (user stopped or is switching backend). */
  #shouldRestartAfterEnd = true;

  dispose(): void {}

  #clearNativeStartFallback(): void {
    if (this.#nativeStartFallback) {
      clearTimeout(this.#nativeStartFallback);
      this.#nativeStartFallback = undefined;
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
      this.bindings.onFinal(final_transcript);
    }
    if (interim_transcript.trim()) {
      this.bindings.onInterim(interim_transcript);
    }
  };
  
  start(state: STT_State): void {
    this.#clearNativeStartFallback();
    this.#nativeConnectSent = false;
    this.#shouldRestartAfterEnd = true;

    if (Object.values(state.native).some(isEmptyValue)) {
      this.bindings.onStop("Options missing");
      return;
    }

    const SpCtor =
      window.webkitSpeechRecognition ||
      (window as unknown as { SpeechRecognition?: typeof window.webkitSpeechRecognition }).SpeechRecognition;
    if (!SpCtor) {
      this.bindings.onStop(
        "Web Speech API is not available in this app (try Whisper or Vosk in the desktop build).",
      );
      return;
    }

    try {
      this.#instance = new SpCtor();
      this.#instance.lang = state.native.language;
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
        const err = error as { error?: string };
        if (err.error === "no-speech") return;
        if (err.error === "network" && isTauri()) {
          this.stop(
            "Web Speech is not supported in the desktop app. Use Whisper, Vosk, or the Chrome / Edge external mic instead.",
          );
          return;
        }
        this.stop(err.error);
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
      this.bindings.onStop(e instanceof Error ? e.message : String(e));
      return;
    }

    window.onbeforeunload = () => this.#instance?.stop();
  }

  stop(error?: string): void {
    this.#clearNativeStartFallback();
    this.#shouldRestartAfterEnd = false;
    if (!this.#instance)
      return;
    this.#instance.onend = null;
    this.#instance.stop();
    this.bindings.onStop(error);
  }
}
