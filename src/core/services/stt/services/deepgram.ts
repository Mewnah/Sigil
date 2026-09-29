import {
  ISTTReceiver,
  ISTTService
} from "../types";

import { isEmptyValue } from "../../../../utils";
import { STT_State } from "../schema";
import { devLog } from "@/utils/devLog";

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

  start(state: STT_State): void {
    const token = ++this.#startToken;

    // `device` is optional ("" / "default" = default microphone).
    const required = Object.entries(state.deepgram)
      .filter(([k]) => k !== "device")
      .map(([, v]) => v);
    if (required.some(isEmptyValue))
      return this.bindings.onStop("Options missing");

    this.sessionEnded = false;

    // Service device, else the Settings Web Audio default.
    const deviceId = (state.deepgram.device || window.ApiServer.state.audioInputDevice || "").trim();
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
      const msg = err instanceof Error ? err.message : String(err);
      this.bindings.onStop(
        msg.toLowerCase().includes("permission") || msg.toLowerCase().includes("denied")
          ? "Microphone permission denied or unavailable. Allow the mic for Sigil or pick another input device."
          : `Microphone: ${msg}`
      );
      return;
    }

    if (token !== this.#startToken) {
      // stop() ran while the microphone was opening: release it and do not open a (billed) socket.
      stream.getTracks().forEach((track) => track.stop());
      return;
    }

    this.stream = stream;
    try {
      this.#openSession(state, stream);
    } catch (err) {
      // Not a microphone problem. Do not echo `err.message`: the WebSocket constructor
      // quotes the rejected subprotocol, i.e. the API key.
      this.sessionEnded = true;
      this.socket = undefined;
      this.recorder = undefined;
      this.stopRecorder();
      this.bindings.onStop(
        `[Deepgram] Failed to start (${err instanceof Error ? err.name : "error"}). Check the API key and audio settings.`
      );
    }
  }

  #openSession(state: STT_State, stream: MediaStream): void {
      this.recorder = new MediaRecorder(stream, { mimeType: "audio/webm" });
      this.socket = new WebSocket(
        `wss://api.deepgram.com/v1/listen?punctuate=${!!state.deepgram.punctuate}&interim_results=${!!state.deepgram.interim}&language=${state.deepgram.language}&tier=${state.deepgram.tier}`,
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
        this.bindings.onStop("[Deepgram] WebSocket error");
      };

      this.socket.onopen = () => {
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
        this.recorder?.start(250);
      };
      this.socket.onmessage = (e) => {
        try {
          const rec = JSON.parse(e.data);
          const transcript = rec?.channel?.alternatives?.[0]?.transcript;

          if (!transcript)
            return;
          if (rec.speech_final || rec.is_final)
            this.bindings.onFinal(transcript);
          else
            this.bindings.onInterim(transcript);
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
        if (ev.wasClean)
          return this.bindings.onStop();
        else
          return this.bindings.onStop(ev.code === 1006 ? "" : "[Deepgram] Unexpected error");
      };
  }

  stopSocket() {
    this.socket?.close();
  }
  stopRecorder() {
    if (this.recorder && this.recorder?.state !== "inactive") {
      this.recorder?.stop();
      this.recorder = undefined;
    }
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
