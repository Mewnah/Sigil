import { invoke, isTauri } from "@tauri-apps/api/core";

function isCpalDefaultSentinel(s: string): boolean {
  const t = s.trim().toLowerCase();
  return t === "default" || t === "communications";
}

/**
 * Capture-hub source ids (Rust `audio::loopback`): `""`/`"default"` = system default input, `"<input name>"`,
 * and `"loopback:<output name>"` = what that output device plays (WASAPI loopback; `"loopback:default"` follows the
 * default output). Loopback sources are only offered where a consumer asks for them (other players' audio must never
 * become the user's own speech-to-text by accident).
 */
export const LOOPBACK_SOURCE_PREFIX = "loopback:";
/** Loopback of the system default output device. */
export const LOOPBACK_DEFAULT_SOURCE = "loopback:default";

/** Source id capturing what output device `outputName` plays. */
export function loopbackSourceId(outputName: string): string {
  return `${LOOPBACK_SOURCE_PREFIX}${outputName.trim()}`;
}

export function isLoopbackSourceId(id: string | null | undefined): boolean {
  return (id ?? "").trim().toLowerCase().startsWith(LOOPBACK_SOURCE_PREFIX);
}

/** Output device name of a loopback id; `""` for the default output. `null` when `id` is not a loopback id. */
export function loopbackOutputName(id: string): string | null {
  if (!isLoopbackSourceId(id)) return null;
  const name = id.trim().slice(LOOPBACK_SOURCE_PREFIX.length).trim();
  return name.toLowerCase() === "default" ? "" : name;
}

/** Device lists as the capture hub reports them (`audio:devices_changed`), names only. */
export interface RustDeviceView {
  inputs: string[];
  outputs: string[];
  defaultInput: string | null;
  defaultOutput: string | null;
}

/**
 * Whether the capture source `source` (as passed to Rust: `null` = system default input) can be opened with the
 * devices in `view`.
 */
export function isRustSourceAvailable(source: string | null, view: RustDeviceView): boolean {
  if (source === null || isCpalDefaultSentinel(source) || source.trim() === "") return view.defaultInput !== null;
  const output = loopbackOutputName(source);
  if (output !== null) return output === "" ? view.defaultOutput !== null : view.outputs.includes(output);
  return view.inputs.includes(source.trim());
}

export interface RustSourceOption {
  label: string;
  value: string;
}

export interface RustSourceOptionLabels {
  systemDefault: string;
  loopbackDefault: string;
  /** Label of one output's loopback source. */
  loopback: (outputName: string) => string;
  /** Label of a saved device that is not connected right now. */
  unavailable: (name: string) => string;
}

/**
 * Picker options for a Rust-captured source: `inputs` holds the system default and the microphones, `loopback`
 * (only with `includeLoopback`) every output as a loopback source. The saved `value` is kept as an option when its
 * device is not connected (so the picker never silently shows another device) or when it is a loopback source in a
 * picker that does not offer loopback.
 */
export function buildRustSourceOptions(
  devices: {
    inputs: readonly { readonly id: string; readonly name: string }[];
    outputs: readonly { readonly id: string; readonly name: string }[];
  },
  value: string,
  includeLoopback: boolean,
  labels: RustSourceOptionLabels,
): { inputs: RustSourceOption[]; loopback: RustSourceOption[] } {
  const inputs: RustSourceOption[] = [{ label: labels.systemDefault, value: "" }];
  for (const d of devices.inputs) inputs.push({ label: d.name, value: d.id });
  const loopback: RustSourceOption[] = [];
  if (includeLoopback) {
    loopback.push({ label: labels.loopbackDefault, value: LOOPBACK_DEFAULT_SOURCE });
    for (const d of devices.outputs) loopback.push({ label: labels.loopback(d.name), value: loopbackSourceId(d.name) });
  }
  const current = value.trim();
  const listed = [...inputs, ...loopback].some((o) => o.value === current);
  if (current === "" || isCpalDefaultSentinel(current) || listed) return { inputs, loopback };
  const output = loopbackOutputName(current);
  if (output === null) {
    inputs.push({ label: labels.unavailable(current), value: current });
  } else {
    const label =
      output === "" ? labels.loopbackDefault
      : devices.outputs.some((d) => d.name === output) ? labels.loopback(output)
      : labels.unavailable(labels.loopback(output));
    (includeLoopback ? loopback : inputs).push({ label, value: current });
  }
  return { inputs, loopback };
}

/**
 * Resolves which capture source to pass to Rust (`start_audio_capture` / `start_recording` / utterance capture):
 * per-service device → Settings native default (`audioInputDeviceNative`) → `null` (system default).
 * `audioInputDevice` is a Web Audio `deviceId` and is never passed to cpal. A per-service value may be a loopback id;
 * the app-wide default never is (it is the user's microphone).
 */
export function resolveRustInputDeviceForInvoke(perService: string | undefined): string | null {
  const p = (perService ?? "").trim();
  if (p.length > 0) {
    if (isCpalDefaultSentinel(p)) return null;
    return p;
  }
  const g = (window.ApiServer.state.audioInputDeviceNative ?? "").trim();
  if (g.length === 0) return null;
  if (isCpalDefaultSentinel(g) || isLoopbackSourceId(g)) return null;
  return g;
}

/**
 * One-time migration: older builds stored the Settings → Audio cpal device name in `audioInputDevice`
 * (the Web Audio field). When the native field is still empty and the legacy value exactly matches a
 * current `plugin:audio|list_input_devices` id, move it to `audioInputDeviceNative` and clear the legacy field.
 *
 * `audioInputDeviceNativeMigrated` makes it run once: it is set after the first conclusive check (nothing to
 * migrate, or the device list was read), and whenever the user picks a native device in Settings/Onboarding,
 * so a later launch never re-interprets `audioInputDevice` or overrides a deliberate "system default" choice.
 * A failed device enumeration leaves the flag unset so the next launch retries.
 */
export async function migrateLegacyNativeInputDevice(): Promise<void> {
  if (!isTauri()) return;
  const state = window.ApiServer.state;
  if (state.audioInputDeviceNativeMigrated) return;
  const legacy = state.audioInputDevice;
  if (state.audioInputDeviceNative !== "" || legacy === "") {
    state.audioInputDeviceNativeMigrated = true;
    return;
  }

  let devices: { id: string; name: string }[];
  try {
    devices = await invoke<{ id: string; name: string }[]>("plugin:audio|list_input_devices");
  } catch {
    return;
  }
  // A native-device pick made in Settings/Onboarding while devices were enumerated already set the flag (and may
  // have chosen "system default"); leave that choice alone.
  if (state.audioInputDeviceNativeMigrated) return;
  state.audioInputDeviceNativeMigrated = true;
  // Settings may have changed while devices were enumerated.
  if (state.audioInputDeviceNative !== "" || state.audioInputDevice !== legacy) return;
  if (!devices.some((d) => d.id === legacy)) return;

  state.audioInputDeviceNative = legacy;
  state.audioInputDevice = "";
}
