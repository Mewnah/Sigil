/**
 * What a params change does to a running engine (`restart.ts`), against the frozen contract
 * (`params_contract.json`: the P3 hot fields and the per-voice `slot.json` fields) and the worker's
 * behaviour (voice / F0 method / F0 range switch without a restart in RVC mode).
 */
import { describe, expect, it } from "vitest";

import contract from "../../../../../src-tauri/src/services/voice_changer/params_contract.json";
import { VC_LEVERS, VC_MODES } from "../contract";
import { RUST_PARAM_KEYS, toRustParams, type RustParamKey } from "../params";
import { changedParamKeys, paramApply, restartNeeded } from "../restart";
import { VoiceChangerStateSchema } from "../schema";

const defaults = () => VoiceChangerStateSchema.parse({});

describe("param apply rules", () => {
    it("every param has a rule in every mode", () => {
        for (const key of RUST_PARAM_KEYS) {
            for (const mode of VC_MODES) {
                expect(["hot", "switch", "restart", "idle"], `${key} in ${mode}`).toContain(paramApply(key, mode));
            }
        }
    });

    it("the contract's P3 hot fields never restart the RVC engine", () => {
        for (const key of contract.params_p3_hot_fields as RustParamKey[]) {
            expect(paramApply(key, "rvc"), key).toBe("hot");
            for (const mode of VC_MODES) expect(restartNeeded([key], mode), `${key} in ${mode}`).toBe(false);
        }
    });

    it("per-voice fields apply live or switch the voice in RVC mode (never a restart)", () => {
        for (const key of [...contract.slot_json_p3_fields, "pitch_semitones", "formant_shift", "index_rate", "protect", "pitch_extractor", "speaker_id"] as RustParamKey[]) {
            expect(["hot", "switch"], key).toContain(paramApply(key, "rvc"));
        }
    });

    it("the voice selection switches without a restart in RVC mode and is idle in the others", () => {
        for (const key of ["rvc_slot_dir", "slot_id", "pitch_extractor", "f0_range"] as RustParamKey[]) {
            expect(paramApply(key, "rvc"), key).toBe("switch");
            expect(restartNeeded([key], "rvc"), key).toBe(false);
            for (const mode of ["classic", "onnx", "sidecar"] as const) expect(paramApply(key, mode), `${key} ${mode}`).toBe("idle");
        }
    });

    it("the RVC engine shape restarts RVC only", () => {
        const shape: RustParamKey[] = [
            "rvc_execution_provider",
            "rvc_preset",
            "precision",
            "block_ms",
            "extra_context_ms",
            "crossfade_ms",
            "hot_slots",
            "vram_budget_mb",
            "gpu_device_index",
            "vram_cap_mb",
            "silence_skip",
            ...VC_LEVERS,
        ];
        for (const key of shape) {
            expect(restartNeeded([key], "rvc"), key).toBe(true);
            expect(restartNeeded([key], "classic"), key).toBe(false);
        }
    });

    it("devices, the mode, the chunk size and the denoiser restart every mode", () => {
        for (const key of ["mode", "input_device", "output_device", "monitor_output_device", "chunk_frames", "denoiser"] as RustParamKey[]) {
            for (const mode of VC_MODES) expect(restartNeeded([key], mode), `${key} ${mode}`).toBe(true);
        }
    });

    it("each mode's own model or bridge restarts only that mode", () => {
        expect(restartNeeded(["sidecar_ws_url"], "sidecar")).toBe(true);
        expect(restartNeeded(["sidecar_ws_url"], "rvc")).toBe(false);
        expect(restartNeeded(["onnx_model_path"], "onnx")).toBe(true);
        expect(restartNeeded(["onnx_execution_provider"], "classic")).toBe(false);
    });

    it("budget, gains, priority and the mic gate are hot everywhere", () => {
        for (const key of ["latency_budget_ms", "input_gain", "output_gain", "monitor_gain", "worker_priority", "aux_mic_gate"] as RustParamKey[]) {
            for (const mode of VC_MODES) expect(paramApply(key, mode), `${key} ${mode}`).toBe("hot");
        }
    });
});

describe("changed params", () => {
    it("ignores f32 round-trip noise and `enabled`", () => {
        const a = toRustParams({ ...defaults(), indexRate: 1 / 3, enabled: false });
        const b = { ...a, index_rate: Math.fround(1 / 3), enabled: true };
        expect(changedParamKeys(a, b)).toEqual([]);
        expect(changedParamKeys(a, { ...b, protect: 0.1, precision: "fp16" })).toEqual(["protect", "precision"]);
    });
});
