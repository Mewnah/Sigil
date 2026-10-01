/**
 * RVC presets (Streamer / Low latency / Quality / Studio-TRT): the settings each one applies
 * (P3-G). The preset itself (`rvc_preset`) also picks the engine's benchmark-decided lever defaults
 * (`auto` levers, Rust `engine::presets::auto_lever`) and the EP an unset `rvc_execution_provider`
 * asks for (`params_contract.json` `rvc_preset.default_ep`, mirrored by `profileDefaultEp`), so the EP
 * each preset sets here is that same default.
 *
 * Extra context follows the minimal / small / higher split of the plan; with partial decode and the
 * F0 cache (template voices) a larger context costs little, while bring-your-own ONNX voices
 * re-encode all of it every block.
 *
 * Lookahead (template voices; P3-I0): Quality and Studio-TRT decode 20 ms of future context into
 * every returned sample, at 20 ms more delay. Measured on the MIT default voice (CUDA, RMVPE, 40 ms
 * blocks, the shipping auto levers, against the offline ceiling): speaker cosine 0.940 -> 0.975,
 * V/UV error 7.4 % -> 3.5 %, MCD 1.66 -> 1.40 dB, DNSMOS 3.12 -> 3.29, per-call p95 11.8 -> 12.5 ms
 * (P3-D's run with the plain RMVPE session: 0.947 -> 0.969). RMVPE with lookahead also beat F0
 * fusion with lookahead on that voice (cosine 0.961, V/UV 2.1 %), so the presets keep RMVPE. The
 * engine factory skips lookahead while the content encoder runs on DirectML (35-49 ms per call
 * there with it), with a lever note.
 *
 * Pure module: no Tauri, DOM or `window`.
 */
import { profileDefaultEp, type RvcProfileId } from "./contract";
import type { VoiceChangerState } from "./schema";

type ProfileSettings = Pick<
    VoiceChangerState,
    "rvcExecutionProvider" | "pitchExtractor" | "indexRate" | "extraContextMs" | "lookaheadMs"
> &
    Partial<Pick<VoiceChangerState, "vramCapMb">>;

export const RVC_PROFILE_SETTINGS: Readonly<Record<RvcProfileId, ProfileSettings>> = {
    // CPU with headroom for the game: the light F0 method, no retrieval, minimal context.
    streamer: {
        rvcExecutionProvider: profileDefaultEp("streamer"),
        pitchExtractor: "fcpe",
        indexRate: 0,
        extraContextMs: 64,
        lookaheadMs: 0,
    },
    low_latency: {
        rvcExecutionProvider: profileDefaultEp("low_latency"),
        pitchExtractor: "rmvpe",
        indexRate: 0,
        extraContextMs: 96,
        lookaheadMs: 0,
        vramCapMb: 768,
    },
    quality: {
        rvcExecutionProvider: profileDefaultEp("quality"),
        pitchExtractor: "rmvpe",
        indexRate: 0.5,
        extraContextMs: 192,
        lookaheadMs: 20,
        vramCapMb: 1536,
    },
    // Quality's shapes on TensorRT-RTX (engine cache; the first start builds the engines).
    studio_trt: {
        rvcExecutionProvider: profileDefaultEp("studio_trt"),
        pitchExtractor: "rmvpe",
        indexRate: 0.5,
        extraContextMs: 192,
        lookaheadMs: 20,
        vramCapMb: 1536,
    },
};

/** The settings patch of applying `profile`. */
export function rvcProfilePatch(profile: RvcProfileId): Partial<VoiceChangerState> {
    return { rvcProfile: profile, ...RVC_PROFILE_SETTINGS[profile] };
}
