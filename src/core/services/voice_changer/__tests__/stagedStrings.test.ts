/**
 * The chunk strings merged from `scripts/i18n-staging/fix-W1-vc-chunk-size.json`: a reworded English string keeps every
 * `{{placeholder}}` of the one it replaced (the other locales keep their translation, which still interpolates the same
 * names), and the new ones carry the placeholders the inspector passes. The chunk wording: the engine's unit is the
 * Chunk size, the sound card's is the Audio device buffer, and "block" names neither.
 */
import { describe, expect, it } from "vitest";

import en from "../../../../../public/i18n/en/translation.json";

const english = (en as { voice_changer: Record<string, string> }).voice_changer;
const placeholders = (text: string) => [...new Set([...text.matchAll(/\{\{\s*([a-z_]+)\s*\}\}/g)].map((m) => m[1]))].sort();

/** The strings that were reworded, with the placeholders the English string had before (and the code still passes). */
const REWORDED: Record<string, string[]> = {
    hot_swap_note: [],
    toast_error_sidecar_length: ["message"],
    tips_body: [],
    metrics_spikes_warn: [],
    latency_budget_info: [],
    lever_info_anchored_noise: [],
    lever_info_nsf_phase0: [],
    lever_info_io_binding: [],
    lever_info_cuda_graphs: [],
    bench_block_pct: [],
    bench_block_p95: [],
    bench_block_p99: [],
    bench_verdict_slow: [],
    lookahead_ms_info: [],
    family_info_ddsp_svc: [],
    family_info_sigil_zs: [],
    family_info_sidecar_v2: [],
    family_ddsp_preset_info: [],
    family_ddsp_preset_sigil_singing: [],
    family_ddsp_preset_sigil_singing_fast: [],
    family_ddsp_steps_info: [],
    family_zs_preset_info: [],
    family_zs_preset_sigil_low: [],
    family_zs_preset_sigil_lean: [],
    family_zs_preset_upstream_readme: [],
    family_zs_steps_info: [],
    family_zs_prompt_s_info: [],
    family_sidecar_timeout: [],
    family_sidecar_timeout_info: [],
    family_block_note: [],
    lookahead_ms_info_v2: [],
    output_cushion_info: [],
    output_buffer_raised: ["prefill", "queue", "stored_prefill", "stored_queue"],
    sola_search_ms_info: [],
    content_context_ms_info: [],
    device_buffer_info: [],
};
const NEW_KEYS = ["chunk_size_auto_about_ms", "chunk_size_fixed_legacy", "chunk_size_short_note", "latency_budget_raised", "metrics_call_label"];

describe("the chunk strings", () => {
    it("keep the placeholders of the English string they reworded", () => {
        expect(Object.keys(REWORDED).length).toBeGreaterThan(20);
        for (const [key, names] of Object.entries(REWORDED)) {
            expect(english[key], `${key} exists`).toBeTypeOf("string");
            expect(placeholders(english[key]), key).toEqual(names);
        }
    });

    it("carry the placeholders the inspector passes to the new ones", () => {
        for (const key of NEW_KEYS) expect(english[key], `${key} exists`).toBeTypeOf("string");
        expect(placeholders(english.chunk_size_auto_about_ms)).toEqual(["ms"]);
        expect(placeholders(english.chunk_size_fixed_legacy)).toEqual(["ms"]);
        expect(placeholders(english.chunk_size_short_note)).toEqual(["block", "current"]);
        expect(placeholders(english.latency_budget_raised)).toEqual(["budget", "chunk"]);
        expect(placeholders(english.metrics_call_label)).toEqual([]);
        // The existing note for a PC with few cores is passed the same names as before.
        expect(placeholders(english.chunk_size_few_cores_note)).toEqual(["block", "cores", "current"]);
    });

    it("never call the engine's unit a block, and name the WebSocket converter's fix", () => {
        for (const key of [...Object.keys(REWORDED), ...NEW_KEYS]) {
            // (A placeholder may be called {{block}}: the words are what the user reads.)
            expect(english[key].replace(/\{\{[^}]*\}\}/g, ""), key).not.toMatch(/\bblocks?\b/i);
        }
        // The WebSocket toast and the Audio device buffer's help point at the one setting that fixes a wrong length.
        expect(english.toast_error_sidecar_length).toContain("Audio device buffer");
        expect(english.toast_error_sidecar_length).not.toMatch(/chunk size/i);
        expect(english.device_buffer_info).toContain("WebSocket");
        expect(english.bench_verdict_slow).toContain("Chunk size");
    });
});
