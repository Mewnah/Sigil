import { invoke, isTauri } from "@tauri-apps/api/core";

function isCpalDefaultSentinel(s: string): boolean {
  const t = s.trim().toLowerCase();
  return t === "default" || t === "communications";
}

/**
 * Resolves which cpal device name to pass to Rust (`start_audio_capture` / `start_recording`):
 * per-service device → Settings native default (`audioInputDeviceNative`) → `null` (system default).
 * `audioInputDevice` is a Web Audio `deviceId` and is never passed to cpal.
 */
export function resolveRustInputDeviceForInvoke(perService: string | undefined): string | null {
  const p = (perService ?? "").trim();
  if (p.length > 0) {
    if (isCpalDefaultSentinel(p)) return null;
    return p;
  }
  const g = (window.ApiServer.state.audioInputDeviceNative ?? "").trim();
  if (g.length === 0) return null;
  if (isCpalDefaultSentinel(g)) return null;
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
