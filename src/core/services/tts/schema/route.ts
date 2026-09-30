import { zSafe } from "@/utils";
import { z } from "zod";

/**
 * Where synthesized speech plays: the top-level `route` of `Service_TTS_Schema` (mirror of Rust
 * `tts_sink::SinkRoute` / `RustTtsSinkRoute`). Owned by P2-G.
 *
 * - `device`: the backend's output device.
 * - `voiceChanger`: into the running voice changer (speech comes out in the converted voice); while the voice
 *   changer is not running it plays on the device, like the old per-backend `routeThroughVoiceChanger` flag.
 * - `both`: the device and the voice changer.
 *
 * Applies to every backend that plays through the TTS sink; Web Speech and Windows SAPI play by themselves.
 */
export enum TTS_Routes {
  device = "device",
  voiceChanger = "voiceChanger",
  both = "both",
}

export const zodTTS_Routes = z.nativeEnum(TTS_Routes);

export const TTS_Route_Schema = zSafe(zodTTS_Routes, TTS_Routes.device);

/** Backends that had their own `routeThroughVoiceChanger` flag before the top-level route existed. */
const LEGACY_ROUTE_FLAG_BACKENDS = ["neuralOnnx", "neuralSidecar"];

/**
 * One-time migration of saved TTS settings without a top-level `route`: the active backend's old
 * `routeThroughVoiceChanger` flag becomes `route: "voiceChanger"`, anything else `route: "device"` (what every other
 * backend did). Settings that already have a `route` are returned unchanged, so it runs once. The old flags stay in
 * their slices (persisted keys are never removed) but no longer decide the route.
 */
export function migrateTtsRoute(raw: unknown): unknown {
  if (!raw || typeof raw !== "object" || Array.isArray(raw)) return raw;
  const data = raw as Record<string, unknown>;
  if ("route" in data) return raw;
  const backend = typeof data.backend === "string" ? data.backend : "";
  const slice = LEGACY_ROUTE_FLAG_BACKENDS.includes(backend) ? data[backend] : undefined;
  const flagged =
    !!slice && typeof slice === "object" && (slice as Record<string, unknown>).routeThroughVoiceChanger === true;
  return { ...data, route: flagged ? TTS_Routes.voiceChanger : TTS_Routes.device };
}
