//! Checks `params_contract.json` (the frozen P0 Rust <-> TS voice-changer contract) against the Rust
//! side: every enum normalizes exactly as listed (no silent remapping), every default and range
//! matches, and every P0 field travels under its contract name. P0-B's TS test reads the same file.

use super::budget::{clamp_latency_budget_ms, DEFAULT_LATENCY_BUDGET_MS, MAX_LATENCY_BUDGET_MS, MIN_LATENCY_BUDGET_MS};
use super::params::{
    normalize_execution_provider, normalize_latency_preset, normalize_pitch_extractor, normalize_rvc_preset,
    AuxMicGate, VoiceChangerHotParams, VoiceChangerParams, DEFAULT_VOCODER_OVERSAMPLE,
};
use super::state::VoiceChangerState;
use crate::services::neural_audio::chunk_processor::ALLOWED_CHUNK_FRAMES;
use crate::services::neural_audio::DEFAULT_CHUNK_FRAMES;
use crate::services::rt_thread::WorkerPriority;
use serde_json::Value;

fn contract() -> Value {
    serde_json::from_str(include_str!("params_contract.json")).expect("params_contract.json parses")
}

fn f(v: &Value) -> f64 {
    v.as_f64().expect("number")
}

fn strs(v: &Value) -> Vec<String> {
    v.as_array().unwrap().iter().map(|s| s.as_str().unwrap().to_string()).collect()
}

#[test]
fn chunk_frames_match_allowed_list() {
    let c = contract();
    let listed: Vec<u32> = c["chunk_frames"]["values"]
        .as_array()
        .unwrap()
        .iter()
        .map(|v| v.as_u64().unwrap() as u32)
        .collect();
    assert_eq!(listed, ALLOWED_CHUNK_FRAMES);
    assert_eq!(c["chunk_frames"]["default"].as_u64().unwrap() as usize, DEFAULT_CHUNK_FRAMES);
}

#[test]
fn rvc_execution_providers_normalize_as_listed() {
    let c = contract();
    let ep = &c["enums"]["rvc_execution_provider"];
    for v in ep["values"].as_array().unwrap() {
        let s = v.as_str().unwrap();
        assert_eq!(normalize_execution_provider(s), s);
        assert_eq!(normalize_execution_provider(&s.to_uppercase()), s);
    }
    for (alias, target) in ep["aliases"].as_object().unwrap() {
        assert_eq!(normalize_execution_provider(alias), target.as_str().unwrap());
    }
    assert_eq!(normalize_execution_provider("bogus"), ep["fallback"].as_str().unwrap());
}

/// Every string enum of the contract: listed values map to themselves (case-insensitively),
/// legacy aliases to their target, anything else to the fallback.
#[test]
fn string_enums_normalize_as_listed() {
    let c = contract();
    let e = &c["enums"];
    let check = |name: &str, norm: &dyn Fn(&str) -> String| {
        let spec = &e[name];
        for v in strs(&spec["values"]) {
            assert_eq!(norm(&v), v, "{name}: {v}");
            assert_eq!(norm(&v.to_uppercase()), v, "{name}: {v} (upper case)");
        }
        if let Some(aliases) = spec["legacy_aliases"].as_object() {
            for (alias, target) in aliases {
                assert_eq!(norm(alias), target.as_str().unwrap(), "{name}: alias {alias}");
            }
        }
        assert_eq!(norm("definitely-not-a-value"), spec["fallback"].as_str().unwrap(), "{name}: fallback");
    };
    check("latency_preset", &|s| normalize_latency_preset(s).to_string());
    check("rvc_preset", &|s| normalize_rvc_preset(s).to_string());
    check("pitch_extractor", &|s| normalize_pitch_extractor(s).to_string());
    check("aux_mic_gate", &|s| AuxMicGate::from_str_lossy(s).as_str().to_string());
    check("worker_priority", &|s| WorkerPriority::from_str_lossy(s).as_str().to_string());
    // The Rust enums list exactly the contract values, in order.
    assert_eq!(strs(&e["aux_mic_gate"]["values"]), AuxMicGate::ALL.map(|g| g.as_str().to_string()));
    assert_eq!(strs(&e["worker_priority"]["values"]), WorkerPriority::ALL.map(|p| p.as_str().to_string()));
}

#[test]
fn latency_budget_matches_contract() {
    let c = contract();
    let b = &c["numbers"]["latency_budget_ms"];
    assert_eq!(MIN_LATENCY_BUDGET_MS, f(&b["min"]));
    assert_eq!(MAX_LATENCY_BUDGET_MS, f(&b["max"]));
    assert_eq!(DEFAULT_LATENCY_BUDGET_MS, f(&b["default"]));
    assert_eq!(clamp_latency_budget_ms(0.0), f(&b["min"]));
    assert_eq!(clamp_latency_budget_ms(1e9), f(&b["max"]));
}

#[test]
fn serde_defaults_match_contract() {
    let c = contract();
    // Only the fields without a serde default; everything else must take the contract default.
    let p: VoiceChangerParams = serde_json::from_str(
        r#"{"enabled":false,"pitch_semitones":0,"formant_shift":0,"output_device":"default"}"#,
    )
    .unwrap();
    let n = &c["numbers"];
    let e = &c["enums"];
    assert_eq!(f64::from(p.vocoder_window_ms), f(&n["vocoder_window_ms"]["default"]));
    assert_eq!(u64::from(p.vocoder_oversample), e["vocoder_oversample"]["default"].as_u64().unwrap());
    assert_eq!(p.vocoder_oversample, DEFAULT_VOCODER_OVERSAMPLE);
    assert_eq!(p.chunk_frames as usize, DEFAULT_CHUNK_FRAMES);
    assert_eq!(f64::from(p.input_gain), f(&n["input_gain"]["default"]));
    assert_eq!(f64::from(p.output_gain), f(&n["output_gain"]["default"]));
    assert_eq!(f64::from(p.monitor_gain), f(&n["monitor_gain"]["default"]));
    assert_eq!(p.output_prefill_ms, f(&n["output_prefill_ms"]["default"]));
    assert_eq!(p.output_max_queue_ms, f(&n["output_max_queue_ms"]["default"]));
    assert!((f64::from(p.buffer_truncate_rms) - f(&n["buffer_truncate_rms"]["default"])).abs() < 1e-9);
    assert_eq!(f64::from(p.buffer_truncate_chunks), f(&n["buffer_truncate_chunks"]["default"]));
    assert_eq!(p.extra_context_ms, f(&n["extra_context_ms"]["default"]));
    assert_eq!(p.crossfade_ms, f(&n["crossfade_ms"]["default"]));
    assert_eq!(f64::from(p.index_rate), f(&n["index_rate"]["default"]));
    assert!((f64::from(p.protect) - f(&n["protect"]["default"])).abs() < 1e-6);
    assert_eq!(p.speaker_id as f64, f(&n["speaker_id"]["default"]));
    assert_eq!(f64::from(p.gpu_device_index), f(&n["gpu_device_index"]["default"]));
    assert_eq!(f64::from(p.vram_cap_mb), f(&n["vram_cap_mb"]["default"]));
    assert_eq!(p.latency_preset, e["latency_preset"]["default"].as_str().unwrap());
    assert_eq!(p.rvc_preset, e["rvc_preset"]["default"].as_str().unwrap());
    assert_eq!(p.pitch_extractor, e["pitch_extractor"]["default"].as_str().unwrap());
    assert_eq!(p.rvc_execution_provider, e["rvc_execution_provider"]["default"].as_str().unwrap());
    assert_eq!(p.onnx_execution_provider, e["onnx_execution_provider"]["default"].as_str().unwrap());
    assert_eq!(
        serde_json::to_value(&p.mode).unwrap(),
        e["mode"]["default"],
        "mode default"
    );
    // The hot-params defaults (used before the first params sync) agree with the serde defaults.
    let h = VoiceChangerHotParams::default();
    assert_eq!(h.output_prefill_ms, p.output_prefill_ms);
    assert_eq!(h.output_max_queue_ms, p.output_max_queue_ms);
    assert_eq!(h.chunk_frames, p.chunk_frames);
}

/// A fresh voice changer reports every P0 params field under its contract name with the contract
/// default (and `get` never omits them).
#[test]
fn p0_params_fields_travel_with_contract_defaults() {
    let c = contract();
    let got = serde_json::to_value(VoiceChangerState::new().snapshot_params()).unwrap();
    for key in c["params_p0_new_fields"].as_object().unwrap().keys() {
        assert!(!got[key].is_null(), "params field {key} missing from get_voice_changer_params");
    }
    let e = &c["enums"];
    assert_eq!(got["latency_budget_ms"].as_f64(), c["numbers"]["latency_budget_ms"]["default"].as_f64());
    assert_eq!(got["aux_mic_gate"], e["aux_mic_gate"]["default"]);
    assert_eq!(got["worker_priority"], e["worker_priority"]["default"]);
    assert_eq!(got["input_device"], "");
    assert_eq!(got["rvc_consent_acknowledged"], false);
}
