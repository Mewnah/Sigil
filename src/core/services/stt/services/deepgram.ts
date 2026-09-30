import {
  ISTTReceiver,
  ISTTService,
  SttTextMeta
} from "../types";

import i18n from "i18next";
import { isEmptyValue } from "../../../../utils";
import { STT_State } from "../schema";
import { normalizeDeepgramLanguage } from "../stt_data";
import { devLog } from "@/utils/devLog";
import { baseLanguageTag } from "../rustCapturePolling";

/** How often the recorder follows the STT mute. */
export const DEEPGRAM_MUTE_POLL_MS = 250;
/** MediaRecorder timeslice: one WebM chunk is sent this often. */
export const DEEPGRAM_TIMESLICE_MS = 250;
/**
 * KeepAlive spacing while the recorder is paused: Deepgram closes a stream that gets neither audio nor a KeepAlive
 * for about 10 s (NET-0001); its docs ask for one every 3-5 s.
 */
export const DEEPGRAM_KEEPALIVE_MS = 4_000;
const KEEPALIVE_MESSAGE = JSON.stringify({ type: "KeepAlive" });

/**
 * getUserMedia constraints for a stored Web Audio `deviceId`. `exact` is used only for an id that currently
 * exists; unknown ids (e.g. legacy cpal names from the old picker, unplugged devices) fall back to the default
 * microphone instead of failing with OverconstrainedError.
 */
async function resolveAudioConstraints(deviceId: string): Promise<MediaTrackConstraints | true> {
  if (!deviceId || deviceId === "default")
    return true;
  let inputs: MediaDeviceInfo[] = [];
  try {
    inputs = (await navigator.mediaDevices.enumerateDevices()).filter((d) => d.kind === "audioinput");
  } catch {
    /* treat as not found */
  }
  if (inputs.some((d) => d.deviceId === deviceId))
    return { deviceId: { exact: deviceId } };
  // Before mic permission is granted, ids are hidden (""): prefer the device without requiring it.
  if (inputs.length > 0 && inputs.every((d) => !d.deviceId))
    return { deviceId };
  console.warn(`[Deepgram] Input device "${deviceId}" not found; using the default microphone.`);
  return true;
}

/** Stop message for a failed `getUserMedia`. Pure. */
export function deepgramMicrophoneError(error: unknown): string {
  const name = error instanceof Error || error instanceof DOMException ? error.name : "";
  const msg = error instanceof Error || error instanceof DOMException ? error.message : String(error);
  const lower = `${name} ${msg}`.toLowerCase();
  if (name === "NotAllowedError" || name === "SecurityError" || lower.includes("permission") || lower.includes("denied"))
    return i18n.t("stt.error_mic_denied");
  if (name === "NotFoundError" || name === "OverconstrainedError")
    return i18n.t("stt.error_mic_missing");
  return i18n.t("stt.error_mic_generic", { detail: msg || name || "error" });
}

/**
 * Stop message for a socket that closed on its own: `null` for a clean close. Before the socket opened, any failure
 * means the handshake was refused (browsers hide the HTTP status of a rejected key behind code 1006). Pure.
 */
export function deepgramCloseMessage(opened: boolean, code: number, reason: string, wasClean: boolean): string | null {
  if (!opened) return i18n.t("stt.error_deepgram_connect");
  if (wasClean && (code === 1000 || code === 1005)) return null;
  const detail = reason.trim() ? `${code}: ${reason.trim()}` : String(code);
  return i18n.t("stt.error_deepgram_lost", { detail });
}

export class STT_DeepgramService implements ISTTService {
  constructor(private bindings: ISTTReceiver) { }

  dispose(): void { }

  private socket?: WebSocket;
  private recorder?: MediaRecorder;
  private stream?: MediaStream;
  /** Prevents double `onStop` when both `onerror` and `onclose` fire. */
  private sessionEnded = false;
  /** Bumped by `start()` and `stop()`; a microphone that opens under an older token is released unused. */
  #startToken = 0;
  /** The socket opened (a failure before that is a refused connection). */
  #opened = false;
  /** A partial is on screen (an empty final then abandons it). */
  #interimOpen = false;
  #muteTimer?: ReturnType<typeof setInterval>;
  #lastKeepAlive = 0;
  /** The recorder was started (it starts only once STT is not muted; a stopped recorder is never restarted). */
  #recorderStarted = false;
  #meta: SttTextMeta | undefined;

  start(state: STT_State): void {
    const token = ++this.#startToken;

    // Only the key and the language are required: `device` ("" = default microphone), the legacy `tier` (empty
    // since the Nova model picker replaced it) and `model` (falls back to nova-3) may be empty.
    if (isEmptyValue(state.deepgram.key) || isEmptyValue(state.deepgram.language))
      return this.bindings.onStop(i18n.t("stt.error_deepgram_options"));

    this.sessionEnded = false;
    this.#opened = false;
    this.#interimOpen = false;
    this.#recorderStarted = false;
    const lang = baseLanguageTag(normalizeDeepgramLanguage(state.deepgram.language));
    this.#meta = lang ? { lang } : undefined;

    // Service device, else the Settings Web Audio default.
    const deviceId = (state.deepgram.device || window.ApiServer.state.audioInputDevice || "").trim();
    this.bindings.onStatus(i18n.t("stt.status_mic_opening"));
    void this.#openMicrophoneAndSession(token, deviceId, state);
  }

  async #openMicrophoneAndSession(token: number, deviceId: string, state: STT_State): Promise<void> {
    let stream: MediaStream;
    try {
      const audio = await resolveAudioConstraints(deviceId);
      // stop() (or a newer start()) ran while devices were enumerated: never open the microphone.
      if (token !== this.#startToken)
        return;
      stream = await navigator.mediaDevices.getUserMedia({ audio });
    } catch (err: unknown) {
      if (token !== this.#startToken)
        return;
      this.bindings.onStatus("");
      this.bindings.onStop(deepgramMicrophoneError(err));
      return;
    }

    if (token !== this.#startToken) {
      // stop() ran while the microphone was opening: release it and do not open a (billed) socket.
      stream.getTracks().forEach((track) => track.stop());
      return;
    }

    this.stream = stream;
    this.bindings.onStatus(i18n.t("stt.status_connecting_engine", { engine: "Deepgram" }));
    try {
      this.#openSession(state, stream);
    } catch (err) {
      // Not a microphone problem. Do not echo `err.message`: the WebSocket constructor
      // quotes the rejected subprotocol, i.e. the API key.
      this.sessionEnded = true;
      this.socket = undefined;
      this.stopRecorder();
      this.recorder = undefined;
      this.bindings.onStatus("");
      this.bindings.onStop(i18n.t("stt.error_deepgram_start", { detail: err instanceof Error ? err.name : "error" }));
    }
  }

  #openSession(state: STT_State, stream: MediaStream): void {
      this.recorder = new MediaRecorder(stream, { mimeType: "audio/webm" });
      this.socket = new WebSocket(
        deepgramListenUrl(state),
        ["token", state.deepgram.key]
      );

      this.socket.onerror = () => {
        if (this.sessionEnded) return;
        this.sessionEnded = true;
        this.stopRecorder();
        try {
          this.socket?.close();
        } catch {
          /* ignore */
        }
        this.socket = undefined;
        this.bindings.onStatus("");
        this.bindings.onStop(this.#opened ? i18n.t("stt.error_deepgram_lost", { detail: "error" }) : i18n.t("stt.error_deepgram_connect"));
      };

      this.socket.onopen = () => {
        this.#opened = true;
        this.bindings.onStart();
        this.recorder?.addEventListener("dataavailable", (event) => {
          if (!this.socket || this.socket.readyState === this.socket.CLOSING)
            return;
          if (
            this.socket.readyState === this.socket.OPEN &&
            this.recorder?.state === "recording"
          )
            this.socket.send(event.data);
        });
        // Starts the recorder now, or (while STT is muted) only once it is unmuted: muted audio is never recorded.
        this.#followMute();
      };
      this.socket.onmessage = (e) => {
        try {
          const rec = JSON.parse(e.data);
          if (rec?.type && rec.type !== "Results")
            return; // Metadata, SpeechStarted, UtteranceEnd
          const transcript = String(rec?.channel?.alternatives?.[0]?.transcript ?? "");
          const final = !!(rec?.speech_final || rec?.is_final);
          if (!transcript.trim()) {
            // A final without words closes a partial that turned out not to be speech.
            if (final && this.#interimOpen) {
              this.#interimOpen = false;
              this.bindings.onInterim("");
            }
            return;
          }
          if (final) {
            this.#interimOpen = false;
            this.bindings.onFinal(transcript, this.#meta);
          }
          else {
            this.#interimOpen = true;
            this.bindings.onInterim(transcript, this.#meta);
          }
        } catch (err) {
          devLog("[Deepgram] Failed to parse message:", err);
        }
      };
      this.socket.onclose = (ev) => {
        this.stopRecorder();
        if (this.sessionEnded)
          return;
        this.sessionEnded = true;
        this.socket = undefined;
        this.bindings.onStatus("");
        const message = deepgramCloseMessage(this.#opened, ev.code, ev.reason ?? "", ev.wasClean);
        return this.bindings.onStop(message ?? undefined);
      };
  }

  /**
   * While STT is hard-muted the recorder is paused, so muted speech never leaves the PC (and is not billed); the
   * socket is kept open with KeepAlive messages. Pausing keeps the WebM stream continuous, unlike dropping chunks.
   * The first check runs synchronously when the socket opens: a session that opens while muted does not start the
   * recorder at all until it is unmuted, so not even the first timeslice of muted audio is recorded or sent.
   */
  #followMute(): void {
    this.#clearMuteTimer();
    this.#lastKeepAlive = Date.now();
    this.#syncMute();
    this.#muteTimer = setInterval(() => this.#syncMute(), DEEPGRAM_MUTE_POLL_MS);
  }

  /** Recorder follows the mute (start / pause / resume); KeepAlive while no audio flows. */
  #syncMute(): void {
    const recorder = this.recorder;
    const socket = this.socket;
    if (!recorder || !socket || socket.readyState !== socket.OPEN) return;
    const muted = this.bindings.isMuted();
    try {
      if (muted) {
        if (recorder.state === "recording") {
          recorder.pause();
          devLog("[Deepgram] Paused (muted)");
        }
      }
      else if (recorder.state === "paused") {
        recorder.resume();
        devLog("[Deepgram] Resumed");
      }
      else if (recorder.state === "inactive" && !this.#recorderStarted) {
        this.#recorderStarted = true;
        recorder.start(DEEPGRAM_TIMESLICE_MS);
      }
    } catch (err) {
      devLog("[Deepgram] Recorder start/pause/resume failed:", err);
    }
    // Not started yet (muted since the socket opened) counts as paused; a recorder that ended on its own does not
    // (the socket then closes on Deepgram's no-audio timeout and the session reports it).
    const silent = recorder.state === "paused" || (recorder.state === "inactive" && !this.#recorderStarted);
    if (silent && Date.now() - this.#lastKeepAlive >= DEEPGRAM_KEEPALIVE_MS) {
      this.#lastKeepAlive = Date.now();
      socket.send(KEEPALIVE_MESSAGE);
    }
    else if (recorder.state === "recording") {
      this.#lastKeepAlive = Date.now();
    }
  }

  #clearMuteTimer(): void {
    if (this.#muteTimer !== undefined) {
      clearInterval(this.#muteTimer);
      this.#muteTimer = undefined;
    }
  }

  stopSocket() {
    this.socket?.close();
  }
  stopRecorder() {
    this.#clearMuteTimer();
    // Also forgets a recorder that never started (muted for the whole session).
    const recorder = this.recorder;
    this.recorder = undefined;
    if (recorder && recorder.state !== "inactive")
      recorder.stop();
    // Release the microphone; MediaRecorder.stop() does not stop the stream tracks.
    this.stream?.getTracks().forEach((track) => track.stop());
    this.stream = undefined;
  }

  stop(): void {
    this.#startToken++;
    this.stopSocket();
    this.stopRecorder();
  }
}

/**
 * Builds the streaming URL. Without `model=` Deepgram falls back to its legacy default model (research §6), so the
 * model is always sent (nova-3 unless the user picked another). The key travels in the WebSocket subprotocol.
 */
export function deepgramListenUrl(state: Pick<STT_State, "deepgram">): string {
  const dg = state.deepgram;
  const params = new URLSearchParams({
    model: dg.model?.trim() || "nova-3",
    language: normalizeDeepgramLanguage(dg.language),
    punctuate: String(!!dg.punctuate),
    interim_results: String(!!dg.interim),
    profanity_filter: String(!!dg.profanity),
  });
  return `wss://api.deepgram.com/v1/listen?${params.toString()}`;
}
