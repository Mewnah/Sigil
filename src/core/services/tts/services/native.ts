import i18n from "i18next";
import { clampedOr } from "../num";
import { TTS_State } from "../schema";
import { ITTSReceiver, ITTSService, TTS_OPTIONS_MISSING, errorMessage } from "../types";

/** How long to wait for the webview to list its voices before giving up. */
const VOICES_TIMEOUT_MS = 5_000;

/** Voices once the webview has loaded them (they arrive asynchronously on first use), or `[]` on timeout. */
export function waitForVoices(synth: SpeechSynthesis, timeoutMs = VOICES_TIMEOUT_MS): Promise<SpeechSynthesisVoice[]> {
  const now = synth.getVoices();
  if (now.length) return Promise.resolve(now);
  return new Promise((resolve) => {
    const done = () => {
      clearTimeout(timer);
      synth.removeEventListener("voiceschanged", done);
      resolve(synth.getVoices());
    };
    const timer = setTimeout(done, timeoutMs);
    // addEventListener, not `onvoiceschanged =`, so the inspector's own handler is not replaced.
    synth.addEventListener("voiceschanged", done);
  });
}

/** Web Speech API (`speechSynthesis`) on the webview's default output. */
export class TTS_NativeService implements ITTSService {
  #voice?: SpeechSynthesisVoice;
  #running = false;
  #stopped = false;
  /**
   * Utterances being spoken. Chromium can garbage-collect an unreferenced utterance mid-speech, and
   * then its end event never fires.
   */
  #pending = new Set<SpeechSynthesisUtterance>();

  constructor(private bindings: ITTSReceiver) {}

  dispose(): void {
    this.#halt();
  }

  get state() {
    return window.ApiServer.state.services.tts.data.native;
  }

  async start(_state: TTS_State) {
    const synth = typeof window !== "undefined" ? window.speechSynthesis : undefined;
    if (!synth || typeof SpeechSynthesisUtterance === "undefined")
      return this.bindings.onStop(i18n.t("tts.native_unavailable"));
    if (!this.state.voice)
      return this.bindings.onStop(TTS_OPTIONS_MISSING);

    try {
      const voices = await waitForVoices(synth);
      if (this.#stopped) return;
      const voice = voices.find(v => v.voiceURI === this.state.voice);
      if (!voice)
        return this.bindings.onStop(i18n.t("tts.native_voice_missing"));
      this.#voice = voice;
      this.#running = true;
      this.bindings.onStart();
    } catch (error) {
      if (!this.#stopped) this.bindings.onStop(errorMessage(error));
    }
  }

  /** Each line gets its own utterance, so a line queued behind another keeps its own text and settings. */
  play(value: string): void {
    const voice = this.#voice;
    if (!this.#running || !voice || !value.trim()) return;

    const utterance = new SpeechSynthesisUtterance(value);
    utterance.voice = voice;
    utterance.lang = voice.lang;
    utterance.pitch = clampedOr(this.state.pitch, 1, 0, 2);
    utterance.rate = clampedOr(this.state.rate, 1, 0.1, 10);
    utterance.volume = clampedOr(this.state.volume, 1, 0, 1);

    const release = () => this.#pending.delete(utterance);
    utterance.onend = release;
    utterance.onerror = (event) => {
      release();
      // cancel() on Stop reports "interrupted"/"canceled"; that is not a failure.
      if (!this.#running || event.error === "interrupted" || event.error === "canceled") return;
      this.bindings.onError(`Web Speech: ${event.error}`);
    };
    this.#pending.add(utterance);
    window.speechSynthesis.speak(utterance);
  }

  stop(): void {
    this.#halt();
    this.bindings.onStop();
  }

  #halt() {
    this.#stopped = true;
    this.#running = false;
    if (this.#pending.size) {
      this.#pending.clear();
      window.speechSynthesis?.cancel();
    }
  }
}
