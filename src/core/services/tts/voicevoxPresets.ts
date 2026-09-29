/**
 * Engine presets for the VOICEVOX backend. AivisSpeech Engine exposes the same HTTP API
 * (`/version`, `/speakers`, `/audio_query`, `/synthesis`) on port 10101, so it runs through the same
 * backend.
 *
 * Kept free of `window`/Tauri so the schema and unit tests can import it.
 */
import { clampedOr, positiveOr } from "./num";

export const VOICEVOX_HOST_PRESETS = ["voicevox", "aivisspeech", "custom"] as const;
export type VoicevoxHostPreset = (typeof VOICEVOX_HOST_PRESETS)[number];
export type VoicevoxEnginePreset = Exclude<VoicevoxHostPreset, "custom">;

/** Default engine URL of each preset (each engine's out-of-the-box port on this PC). */
export const VOICEVOX_PRESET_HOSTS: Record<VoicevoxEnginePreset, string> = {
  voicevox: "http://localhost:50021",
  aivisspeech: "http://localhost:10101",
};

export const VOICEVOX_DEFAULT_HOST = VOICEVOX_PRESET_HOSTS.voicevox;

/** Engine download pages shown in the inspector. */
export const VOICEVOX_PRESET_WEBSITES: Record<VoicevoxEnginePreset, string> = {
  voicevox: "https://voicevox.hiroshiba.jp/",
  aivisspeech: "https://aivis-project.com/",
};

const PRESET_PORTS: Record<string, VoicevoxEnginePreset> = {
  "50021": "voicevox",
  "10101": "aivisspeech",
};

/** Hosts the webview may reach over plain http (the CSP allows `http://localhost:*` and `http://127.0.0.1:*`). */
const LOCAL_HTTP_HOSTS = new Set(["localhost", "127.0.0.1"]);

/**
 * Canonical engine URL: trimmed, `http://` added when no scheme was typed, and no trailing slash,
 * so `${host}/version` never becomes `//version`.
 */
export function normalizeVoicevoxHost(host: string): string {
  let h = (host ?? "").trim();
  if (!h) return "";
  if (!/^[a-z][a-z0-9+.-]*:\/\//i.test(h)) h = `http://${h}`;
  return h.replace(/\/+$/, "");
}

/** The preset a stored host corresponds to (used to migrate settings saved before presets existed). */
export function presetForHost(host: string): VoicevoxHostPreset {
  let url: URL;
  try {
    url = new URL(normalizeVoicevoxHost(host));
  } catch {
    return "custom";
  }
  if (url.protocol !== "http:" || !LOCAL_HTTP_HOSTS.has(url.hostname)) return "custom";
  if (url.pathname !== "/" && url.pathname !== "") return "custom";
  return PRESET_PORTS[url.port] ?? "custom";
}

export type VoicevoxHostProblem = "empty" | "invalid_url" | "remote_http";

/**
 * Why the webview cannot reach `host`, or `null` when it can. The app's CSP allows plain http only
 * to localhost and 127.0.0.1; an engine on another PC must be served over https.
 */
export function voicevoxHostProblem(host: string): VoicevoxHostProblem | null {
  const normalized = normalizeVoicevoxHost(host);
  if (!normalized) return "empty";
  let url: URL;
  try {
    url = new URL(normalized);
  } catch {
    return "invalid_url";
  }
  if (url.protocol === "https:") return null;
  if (url.protocol !== "http:") return "invalid_url";
  return LOCAL_HTTP_HOSTS.has(url.hostname) ? null : "remote_http";
}

export type VoicevoxPresetPatch = {
  hostPreset: VoicevoxHostPreset;
  host?: string;
  pitchScale?: string;
};

/**
 * Settings to apply when the user picks a preset. Engine presets set their default URL; "custom" keeps
 * the current URL for editing. AivisSpeech also resets pitch to 0, because its voices degrade with
 * any non-zero pitchScale.
 */
export function applyVoicevoxPreset(preset: VoicevoxHostPreset): VoicevoxPresetPatch {
  if (preset === "custom") return { hostPreset: preset };
  const patch: VoicevoxPresetPatch = { hostPreset: preset, host: VOICEVOX_PRESET_HOSTS[preset] };
  if (preset === "aivisspeech") patch.pitchScale = "0";
  return patch;
}

/**
 * Slider ranges (the bounds VOICEVOX's own editor uses) and engine defaults. Every step lands exactly
 * on the default: pitch needs 0.01, because with step 0.1 from -0.15 the slider skipped 0.
 */
export const VOICEVOX_RANGES = {
  speedScale: { min: 0.5, max: 2, step: 0.1, fallback: 1 },
  pitchScale: { min: -0.15, max: 0.15, step: 0.01, fallback: 0 },
  intonationScale: { min: 0, max: 2, step: 0.1, fallback: 1 },
  volumeScale: { min: 0, max: 2, step: 0.1, fallback: 1 },
} as const;

export type VoicevoxScales = {
  speedScale: number;
  pitchScale: number;
  intonationScale: number;
  volumeScale: number;
};

/**
 * The audio_query overrides from stored settings. 0 stays 0 for pitch, intonation and volume;
 * speed must be positive. Junk values fall back to the engine defaults.
 */
export function voicevoxScales(settings: {
  speedScale: string;
  pitchScale: string;
  intonationScale: string;
  volumeScale: string;
}): VoicevoxScales {
  const r = VOICEVOX_RANGES;
  return {
    speedScale: Math.min(r.speedScale.max, Math.max(r.speedScale.min, positiveOr(settings.speedScale, r.speedScale.fallback))),
    pitchScale: clampedOr(settings.pitchScale, r.pitchScale.fallback, r.pitchScale.min, r.pitchScale.max),
    intonationScale: clampedOr(settings.intonationScale, r.intonationScale.fallback, r.intonationScale.min, r.intonationScale.max),
    volumeScale: clampedOr(settings.volumeScale, r.volumeScale.fallback, r.volumeScale.min, r.volumeScale.max),
  };
}

/**
 * Fill in `hostPreset` for settings saved before presets existed: infer it from the stored host,
 * so a user already pointing at AivisSpeech (or a LAN engine) is not relabelled "VOICEVOX".
 * Non-object values become `{}` so the slice falls back to defaults instead of failing to parse.
 */
export function migrateVoicevoxSettings(raw: unknown): unknown {
  if (raw === undefined) return raw;
  if (!raw || typeof raw !== "object" || Array.isArray(raw)) return {};
  const obj = raw as Record<string, unknown>;
  if ((VOICEVOX_HOST_PRESETS as readonly unknown[]).includes(obj.hostPreset)) return raw;
  const host = typeof obj.host === "string" && obj.host.trim() ? obj.host : VOICEVOX_DEFAULT_HOST;
  return { ...obj, hostPreset: presetForHost(host) };
}
