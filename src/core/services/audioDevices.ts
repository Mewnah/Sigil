/**
 * Audio devices service (host / desktop only): the TS side of the Rust capture hub.
 *
 * - Keeps the device lists and defaults (`audio:devices_changed`) in a valtio proxy, so every picker refreshes
 *   when a device is plugged, unplugged or the Windows default changes. Loopback sources are the outputs.
 * - Surfaces capture outages: `audio:stream_error` (the hub is reconnecting a stream, same epoch) and
 *   `audio:stream_recovered`, as toasts and system-log lines. `audio:stream_closed` ends an outage whose source
 *   lost its last consumer first (STT stopped or switched device while the device was gone).
 * - Restarts speech-to-text where the hub cannot follow on its own: browser-captured backends (Azure, Deepgram,
 *   Web Speech) after the default microphone changed, or when their explicitly chosen microphone itself went away
 *   or came back (never for an unrelated device), and hub backends whose start failed because their device was
 *   missing, once that device appears ({@link planSttRestart}).
 * - Drives the Rust mute gate: while STT is hard-muted the capture hub withholds microphone audio from every STT
 *   consumer (`plugin:audio|set_capture_mute`, group `stt`), so muted audio never reaches an engine.
 *
 * Owned by P1-C; scaffolded by P1-S0b. Event names and payloads (`AUDIO_DEVICE_EVENTS`, `RustAudioDevicesChanged`,
 * `RustAudioStreamEvent`, `RustAudioDevice` in src/types.ts) are frozen by P1-S0b; the Rust side lives in the
 * `audio` plugin (audio/hub.rs, audio/watchdog.rs, audio/device_watch.rs).
 */
import { invoke } from "@tauri-apps/api/core";
import { listen, type UnlistenFn } from "@tauri-apps/api/event";
import i18n from "i18next";
import { toast } from "react-toastify";
import { proxy } from "valtio";
import { subscribeKey } from "valtio/utils";
import {
  AUDIO_DEVICE_EVENTS,
  ServiceNetworkState,
  type RustAudioDevice,
  type RustAudioDevicesChanged,
  type RustAudioStreamEvent,
  type RustCaptureMuteGroup,
  type RustCaptureStatus,
} from "@/types";
import { pushSystemLog } from "@/core/services/systemLog";
import { STT_Backends } from "@/core/services/stt/schema";
import { SttMuteState } from "@/core/services/stt/types";
import {
  isLoopbackSourceId,
  isRustSourceAvailable,
  loopbackOutputName,
  resolveRustInputDeviceForInvoke,
  type RustDeviceView,
} from "@/utils/rustAudioInput";

export { AUDIO_DEVICE_EVENTS };

/**
 * Rust `hub::STREAM_CLOSED_EVENT`: a source in an outage lost its last consumer before it recovered (payload
 * {@link RustAudioStreamEvent}, `message` null). Alias of `AUDIO_DEVICE_EVENTS.streamClosed`.
 */
export const AUDIO_STREAM_CLOSED_EVENT = AUDIO_DEVICE_EVENTS.streamClosed;

/** Mute groups of the capture hub's gate (Rust `hub::MuteGroup`). */
export type CaptureMuteGroup = RustCaptureMuteGroup;

export type { RustCaptureStatus };

export const audioDevicesApi = {
  listInputDevices: () => invoke<RustAudioDevice[]>("plugin:audio|list_input_devices"),
  listOutputDevices: () => invoke<RustAudioDevice[]>("plugin:audio|list_output_devices"),
  /** Inputs, outputs and defaults in one call (reason `snapshot`). */
  getDeviceSnapshot: () => invoke<RustAudioDevicesChanged>("plugin:audio|get_device_snapshot"),
  setCaptureMute: (group: CaptureMuteGroup, muted: boolean) =>
    invoke<void>("plugin:audio|set_capture_mute", { group, muted }),
  getCaptureStatus: () => invoke<RustCaptureStatus>("plugin:audio|get_capture_status"),
};

/** How an STT backend captures audio. */
export type SttCaptureClass = "hub" | "webview" | "external" | "none";

export function sttCaptureClass(backend: string): SttCaptureClass {
  switch (backend) {
    case STT_Backends.whisper:
    case STT_Backends.vosk:
    case STT_Backends.moonshine:
    case STT_Backends.openai_audio:
      return "hub";
    case STT_Backends.azure:
    case STT_Backends.deepgram:
    case STT_Backends.native:
    case STT_Backends.browser:
      return "webview";
    case STT_Backends.chrome:
    case STT_Backends.edge:
      return "external";
    default:
      return "none";
  }
}

/** Whether any name of `before` is missing from `after` (identical names counted separately). */
export function lostAny(before: readonly string[], after: readonly string[]): boolean {
  const left = new Map<string, number>();
  for (const n of after) left.set(n, (left.get(n) ?? 0) + 1);
  for (const n of before) {
    const count = left.get(n) ?? 0;
    if (count === 0) return true;
    left.set(n, count - 1);
  }
  return false;
}

export interface SttRestartInput {
  backend: string;
  status: ServiceNetworkState;
  /** Error that ended the last STT session ("" if none). */
  error: string;
  /** Hub backends: the Rust source id STT captures from (`null` = system default input). */
  source: string | null;
  /** Webview backends: the backend follows the system default microphone (no explicit device picked). */
  followsDefaultMic: boolean;
  /** The last STT failure happened while its capture source was missing. */
  failedWhileSourceMissing: boolean;
  /**
   * Webview backends with an explicit device: whether its Web Audio `deviceId` was enumerated before and after
   * the change (`null` / absent: unknown). Cpal names and Web Audio ids cannot be matched reliably, so the
   * webview's own device list decides whether the change concerns this device at all.
   */
  webDeviceBefore?: boolean | null;
  webDeviceAfter?: boolean | null;
  before: RustDeviceView;
  after: RustDeviceView;
}

/**
 * Whether a device change should restart speech-to-text. The capture hub reconnects and follows devices for the
 * Rust-captured backends by itself; the webview's own capture (getUserMedia / Azure SDK) does neither.
 */
export function planSttRestart(i: SttRestartInput): "restart" | "none" {
  switch (sttCaptureClass(i.backend)) {
    case "webview": {
      if (i.followsDefaultMic) {
        if (i.status !== ServiceNetworkState.connected) return "none";
        return i.before.defaultInput !== i.after.defaultInput && i.after.defaultInput !== null ? "restart" : "none";
      }
      // An explicit device: only a change of that device matters, never an unrelated device coming or going.
      const was = i.webDeviceBefore ?? null;
      const is = i.webDeviceAfter ?? null;
      if (was === null || is === null || was === is) return "none";
      if (i.status === ServiceNetworkState.connected) {
        // Gone: reopen (Deepgram falls back to the default microphone; Azure reports the missing device instead
        // of listening to nothing). Back: move a session running on the fallback back to the chosen device.
        return is || i.after.inputs.length > 0 ? "restart" : "none";
      }
      // A start that failed because the device was missing is retried once it is back.
      if (is && i.status === ServiceNetworkState.disconnected && i.error !== "" && i.failedWhileSourceMissing)
        return "restart";
      return "none";
    }
    case "hub":
      // A start that failed because its device was missing is retried once the device is there. Running sessions
      // are never touched: the hub keeps them alive across unplug / replug and default changes.
      if (
        i.status === ServiceNetworkState.disconnected &&
        i.error !== "" &&
        i.failedWhileSourceMissing &&
        !isRustSourceAvailable(i.source, i.before) &&
        isRustSourceAvailable(i.source, i.after)
      )
        return "restart";
      return "none";
    default:
      return "none";
  }
}

/**
 * Whether the webview currently lists the audio input `deviceId` (`null`: unknown, i.e. no media-devices API,
 * ids hidden before microphone permission, or an error). Never rejects.
 */
export async function webAudioInputPresent(deviceId: string): Promise<boolean | null> {
  const media = typeof navigator === "undefined" ? undefined : navigator.mediaDevices;
  if (!media?.enumerateDevices) return null;
  try {
    const inputs = (await media.enumerateDevices()).filter((d) => d.kind === "audioinput");
    if (inputs.length > 0 && inputs.every((d) => !d.deviceId)) return null;
    return inputs.some((d) => d.deviceId === deviceId);
  } catch {
    return null;
  }
}

/** Debounce of device events before STT restart decisions (one unplug raises several). */
const RESTART_DEBOUNCE_MS = 1_000;
/** Minimum spacing of automatic STT restarts. */
const RESTART_MIN_INTERVAL_MS = 5_000;

function viewOf(ev: Pick<RustAudioDevicesChanged, "inputs" | "outputs" | "default_input" | "default_output">): RustDeviceView {
  return {
    inputs: ev.inputs.map((d) => d.name),
    outputs: ev.outputs.map((d) => d.name),
    defaultInput: ev.default_input,
    defaultOutput: ev.default_output,
  };
}

/** Display name of a capture source key from `RustAudioStreamEvent.source`. */
export function captureSourceLabel(source: string): string {
  if (source === "" || source === "default") return i18n.t("audio_devices.source_default_input");
  if (isLoopbackSourceId(source)) {
    const output = loopbackOutputName(source) ?? "";
    return output === ""
      ? i18n.t("audio_devices.loopback_default")
      : i18n.t("audio_devices.loopback_label", { name: output });
  }
  return source;
}

export default class Service_AudioDevices {
  readonly state = proxy({
    inputs: [] as RustAudioDevice[],
    outputs: [] as RustAudioDevice[],
    defaultInput: null as string | null,
    defaultOutput: null as string | null,
    /** Bumped on every device change: pickers of other device kinds (Web Audio, SAPI) re-read their lists. */
    revision: 0,
    /** Whether a device list was ever loaded. */
    loaded: false,
    /** Capture sources in an outage (the hub is reconnecting them), by source key. */
    failing: {} as Record<string, { epoch: number; message: string }>,
  });

  #disposers: (() => void)[] = [];
  #loading: Promise<void> | null = null;
  #sttGateMuted: boolean | null = null;
  #restartTimer: ReturnType<typeof setTimeout> | null = null;
  #restartBaseline: RustDeviceView | null = null;
  #lastRestartAt = 0;
  #sttFailure: { failedWhileSourceMissing: boolean } | null = null;
  /** Last known presence of the webview STT backend's explicitly chosen Web Audio device. */
  #webDevice: { deviceId: string; present: boolean | null } | null = null;

  async init(): Promise<void> {
    if (!window.Config.isApp()) return;
    try {
      const unlistens: UnlistenFn[] = await Promise.all([
        listen<RustAudioDevicesChanged>(AUDIO_DEVICE_EVENTS.devicesChanged, (e) => this.#onDevicesChanged(e.payload)),
        listen<RustAudioStreamEvent>(AUDIO_DEVICE_EVENTS.streamError, (e) => this.#onStreamError(e.payload)),
        listen<RustAudioStreamEvent>(AUDIO_DEVICE_EVENTS.streamRecovered, (e) => this.#onStreamRecovered(e.payload)),
        listen<RustAudioStreamEvent>(AUDIO_STREAM_CLOSED_EVENT, (e) => this.#onStreamClosed(e.payload)),
      ]);
      this.#disposers.push(...unlistens);
    } catch (e) {
      console.warn("[AudioDevices] Failed to listen for device events:", e);
    }
    await this.refresh();
    await this.#probeWebDevice();
    this.#watchStt();
  }

  /** Load the lists once (pickers call it; cheap after the first load). */
  ensureLoaded(): Promise<void> {
    if (this.state.loaded) return Promise.resolve();
    return this.refresh();
  }

  /** Re-read devices and defaults from Rust. Never rejects. */
  refresh(): Promise<void> {
    if (!window.Config.isApp()) return Promise.resolve();
    if (this.#loading) return this.#loading;
    this.#loading = (async () => {
      try {
        const snapshot = await audioDevicesApi.getDeviceSnapshot();
        this.#apply(snapshot);
      } catch (e) {
        console.warn("[AudioDevices] Failed to list audio devices:", e);
      } finally {
        this.#loading = null;
      }
    })();
    return this.#loading;
  }

  /** Current lists as names. */
  view(): RustDeviceView {
    return {
      inputs: this.state.inputs.map((d) => d.name),
      outputs: this.state.outputs.map((d) => d.name),
      defaultInput: this.state.defaultInput,
      defaultOutput: this.state.defaultOutput,
    };
  }

  #apply(ev: RustAudioDevicesChanged): void {
    this.state.inputs = ev.inputs;
    this.state.outputs = ev.outputs;
    this.state.defaultInput = ev.default_input;
    this.state.defaultOutput = ev.default_output;
    this.state.loaded = true;
    this.state.revision++;
  }

  #onDevicesChanged(ev: RustAudioDevicesChanged): void {
    const before = this.view();
    this.#apply(ev);
    pushSystemLog(
      "Audio",
      i18n.t("audio_devices.log_devices_changed", {
        reason: ev.reason,
        inputs: ev.inputs.length,
        outputs: ev.outputs.length,
        input: ev.default_input ?? "-",
      }),
    );
    // The first event of a burst is the baseline the restart decision compares against.
    if (!this.#restartBaseline) this.#restartBaseline = before;
    if (this.#restartTimer) clearTimeout(this.#restartTimer);
    this.#restartTimer = setTimeout(() => void this.#maybeRestartStt(), RESTART_DEBOUNCE_MS);
  }

  #sttSource(): string | null {
    const data = window.ApiServer.state.services.stt.data;
    const backend = data.backend;
    const device =
      backend === STT_Backends.whisper ? data.whisper.device
      : backend === STT_Backends.vosk ? data.vosk.device
      : backend === STT_Backends.moonshine ? data.moonshine.device
      : backend === STT_Backends.openai_audio ? data.openai_audio.device
      : undefined;
    return resolveRustInputDeviceForInvoke(device);
  }

  /** Web Audio `deviceId` the webview STT backend is pinned to (`""`: follows the default microphone). */
  #sttWebDevice(): string {
    const data = window.ApiServer.state.services.stt.data;
    const device =
      data.backend === STT_Backends.azure ? data.azure.device
      : data.backend === STT_Backends.deepgram ? data.deepgram.device || window.ApiServer.state.audioInputDevice
      : "";
    const d = (device ?? "").trim();
    return d === "default" ? "" : d;
  }

  #sttFollowsDefaultMic(): boolean {
    return this.#sttWebDevice() === "";
  }

  /** The pinned webview device (when the STT backend is a webview one with an explicit device), else `""`. */
  #pinnedWebDevice(): string {
    const backend = window.ApiServer?.state?.services?.stt?.data?.backend;
    return backend !== undefined && sttCaptureClass(backend) === "webview" ? this.#sttWebDevice() : "";
  }

  /**
   * Re-read whether the pinned webview device is listed; returns its previous and current presence (`null`:
   * unknown, or nothing pinned).
   */
  async #probeWebDevice(): Promise<{ before: boolean | null; after: boolean | null }> {
    const deviceId = this.#pinnedWebDevice();
    if (deviceId === "") {
      this.#webDevice = null;
      return { before: null, after: null };
    }
    const before = this.#webDevice?.deviceId === deviceId ? this.#webDevice.present : null;
    const after = await webAudioInputPresent(deviceId);
    // The setting may have changed while the list was read: only the current device's answer is kept.
    if (this.#pinnedWebDevice() !== deviceId) return { before: null, after: null };
    this.#webDevice = { deviceId, present: after };
    return { before, after };
  }

  async #maybeRestartStt(): Promise<void> {
    this.#restartTimer = null;
    const before = this.#restartBaseline;
    this.#restartBaseline = null;
    if (!before || !window.ApiServer?.stt) return;
    const web = await this.#probeWebDevice();
    const stt = window.ApiServer?.stt;
    if (!stt) return;
    const data = window.ApiServer.state.services.stt.data;
    const decision = planSttRestart({
      backend: data.backend,
      status: stt.serviceState.status,
      error: stt.serviceState.error,
      source: this.#sttSource(),
      followsDefaultMic: this.#sttFollowsDefaultMic(),
      failedWhileSourceMissing: this.#sttFailure?.failedWhileSourceMissing ?? false,
      webDeviceBefore: web.before,
      webDeviceAfter: web.after,
      before,
      after: this.view(),
    });
    if (decision !== "restart") return;
    const now = Date.now();
    if (now - this.#lastRestartAt < RESTART_MIN_INTERVAL_MS) return;
    this.#lastRestartAt = now;
    this.#sttFailure = null;
    const line = i18n.t("audio_devices.stt_restarted");
    pushSystemLog("Audio", line);
    toast.info(line, { toastId: "audio-stt-restart", autoClose: 4_000 });
    stt.start();
  }

  #onStreamError(ev: RustAudioStreamEvent): void {
    this.state.failing[ev.source] = { epoch: ev.epoch, message: ev.message ?? "" };
    const line = i18n.t("audio_devices.stream_error", {
      source: captureSourceLabel(ev.source),
      message: ev.message ?? "",
    });
    pushSystemLog("Audio", line, "warning");
    try {
      toast.warning(line, { toastId: `audio-stream-${ev.source}`, autoClose: false });
    } catch {
      /* toasts are best effort */
    }
  }

  /** The outage's source lost its last consumer before it recovered: no recovery will follow for that epoch. */
  #onStreamClosed(ev: RustAudioStreamEvent): void {
    const failing = this.state.failing[ev.source];
    // A newer outage of the same source (a later session) keeps its toast.
    if (!failing || failing.epoch !== ev.epoch) return;
    delete this.state.failing[ev.source];
    pushSystemLog("Audio", i18n.t("audio_devices.stream_closed", { source: captureSourceLabel(ev.source) }));
    try {
      toast.dismiss(`audio-stream-${ev.source}`);
    } catch {
      /* toasts are best effort */
    }
  }

  #onStreamRecovered(ev: RustAudioStreamEvent): void {
    delete this.state.failing[ev.source];
    const line = i18n.t("audio_devices.stream_recovered", { source: captureSourceLabel(ev.source) });
    pushSystemLog("Audio", line);
    try {
      toast.dismiss(`audio-stream-${ev.source}`);
      toast.success(line, { toastId: `audio-stream-ok-${ev.source}`, autoClose: 3_000 });
    } catch {
      /* toasts are best effort */
    }
  }

  /** Whether the current STT backend's capture device is known to be missing right now. */
  #sttSourceMissing(): boolean {
    const backend = window.ApiServer.state.services.stt.data.backend;
    switch (sttCaptureClass(backend)) {
      case "hub":
        return this.state.loaded && !isRustSourceAvailable(this.#sttSource(), this.view());
      case "webview": {
        const deviceId = this.#pinnedWebDevice();
        return deviceId !== "" && this.#webDevice?.deviceId === deviceId && this.#webDevice.present === false;
      }
      default:
        return false;
    }
  }

  /** STT hard mute → Rust mute gate; STT failures → remember whether the device was missing. */
  #watchStt(): void {
    const stt = window.ApiServer?.stt;
    if (!stt) return;
    const syncMute = () => {
      const muted = stt.serviceState.muted === SttMuteState.muted;
      if (muted === this.#sttGateMuted) return;
      this.#sttGateMuted = muted;
      audioDevicesApi.setCaptureMute("stt", muted).catch((e) => {
        // Retry on the next change.
        this.#sttGateMuted = null;
        console.warn("[AudioDevices] Failed to set the capture mute gate:", e);
      });
    };
    syncMute();
    this.#disposers.push(subscribeKey(stt.serviceState, "muted", syncMute));
    this.#disposers.push(
      subscribeKey(stt.serviceState, "status", (status) => {
        if (status === ServiceNetworkState.disconnected && stt.serviceState.error !== "") {
          this.#sttFailure = { failedWhileSourceMissing: this.#sttSourceMissing() };
        } else if (status !== ServiceNetworkState.disconnected) {
          this.#sttFailure = null;
          // Baseline for the pinned webview device (the setting may have changed since the last check).
          if (status === ServiceNetworkState.connected) void this.#probeWebDevice();
        }
      }),
    );
  }

  dispose(): void {
    if (this.#restartTimer) clearTimeout(this.#restartTimer);
    this.#disposers.forEach((d) => d());
    this.#disposers = [];
  }
}
