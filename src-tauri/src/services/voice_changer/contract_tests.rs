//! Checks `params_contract.json` (the frozen P0 Rust <-> TS voice-changer contract) against the Rust
//! side: every enum normalizes exactly as listed (no silent remapping), every default and range
//! matches, and every P0 field travels under its contract name. P0-B's TS test reads the same file.

use super::budget::{clamp_latency_budget_ms, DEFAULT_LATENCY_BUDGET_MS, MAX_LATENCY_BUDGET_MS, MIN_LATENCY_BUDGET_MS};
use super::capabilities;
use super::params::{
    base_ep, normalize_denoiser, normalize_execution_provider, normalize_f0_range, normalize_latency_preset,
    normalize_lever_mode, normalize_pitch_extractor, normalize_precision, normalize_rvc_preset, pitch_stand_in,
    preset_default_ep, AuxMicGate, VoiceChangerHotParams, VoiceChangerParams, BASELINE_PITCH_EXTRACTORS,
    BLOCK_MS_RANGE, DEFAULT_VOCODER_OVERSAMPLE, FORMANT_SEMITONES_RANGE, GPU_ONLY_PITCH_EXTRACTORS, HOT_SLOTS_RANGE,
    LOOKAHEAD_MS_RANGE, NOISE_SCALE_RANGE, PITCH_EXTRACTORS, RMS_MIX_RANGE, RVC_EXECUTION_PROVIDERS, RVC_PRESETS, SILENCE_DB_RANGE,
    VRAM_BUDGET_MB_RANGE,
};
use super::state::VoiceChangerState;
use crate::services::neural_audio::chunk_processor::ALLOWED_CHUNK_FRAMES;
use crate::services::neural_audio::vc::engine::options::{DenoiserKind, F0Range, Lever, LeverMode, PrecisionMode};
use crate::services::neural_audio::{RvcPresetProfile, DEFAULT_CHUNK_FRAMES};
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
    check("precision", &|s| normalize_precision(s).to_string());
    check("lever_mode", &|s| normalize_lever_mode(s).to_string());
    check("denoiser", &|s| normalize_denoiser(s).to_string());
    check("f0_range", &|s| normalize_f0_range(s).to_string());
    // The Rust enums list exactly the contract values, in order.
    assert_eq!(strs(&e["aux_mic_gate"]["values"]), AuxMicGate::ALL.map(|g| g.as_str().to_string()));
    assert_eq!(strs(&e["worker_priority"]["values"]), WorkerPriority::ALL.map(|p| p.as_str().to_string()));
    assert_eq!(strs(&e["precision"]["values"]), PrecisionMode::ALL.map(|p| p.as_str().to_string()));
    assert_eq!(strs(&e["lever_mode"]["values"]), LeverMode::ALL.map(|m| m.as_str().to_string()));
    assert_eq!(strs(&e["denoiser"]["values"]), DenoiserKind::ALL.map(|d| d.as_str().to_string()));
    assert_eq!(strs(&e["f0_range"]["values"]), F0Range::ALL.map(|r| r.as_str().to_string()));
    assert_eq!(strs(&e["pitch_extractor"]["values"]), PITCH_EXTRACTORS.map(str::to_string));
    assert_eq!(strs(&e["pitch_extractor"]["baseline"]), BASELINE_PITCH_EXTRACTORS.map(str::to_string));
    assert_eq!(strs(&e["pitch_extractor"]["gpu_only"]), GPU_ONLY_PITCH_EXTRACTORS.map(str::to_string));
    assert_eq!(strs(&e["rvc_execution_provider"]["values"]), RVC_EXECUTION_PROVIDERS.map(str::to_string));
    assert_eq!(strs(&e["rvc_preset"]["values"]), RVC_PRESETS.map(str::to_string));
}

/// P3: every lever travels under its contract name; stand-ins, base rungs and preset EPs are the
/// contract's; the baseline values are always offered by the capabilities.
#[test]
fn p3_levers_stand_ins_and_bases_match_contract() {
    let c = contract();
    let e = &c["enums"];
    assert_eq!(strs(&c["levers"]), Lever::ALL.map(|l| l.as_str().to_string()));
    // P3 levers, plus the P3B ones (P3B-S0) in their own section.
    let lever_keys: Vec<String> = ["params_p3_new_fields", "params_p3b_new_fields"]
        .iter()
        .flat_map(|section| c[*section].as_object().unwrap().iter())
        .filter(|(_, v)| v.as_str().unwrap().contains("enums.lever_mode"))
        .map(|(k, _)| k.clone())
        .collect();
    assert_eq!(lever_keys.len(), Lever::COUNT);
    for l in Lever::ALL {
        assert!(lever_keys.contains(&l.as_str().to_string()), "{}", l.as_str());
    }
    for (p, stand_in) in e["pitch_extractor"]["stand_in"].as_object().unwrap() {
        assert_eq!(pitch_stand_in(p), stand_in.as_str().unwrap(), "{p}");
    }
    for (ep, base) in e["rvc_execution_provider"]["base"].as_object().unwrap() {
        assert_eq!(base_ep(ep), base.as_str().unwrap(), "{ep}");
    }
    for (preset, ep) in e["rvc_preset"]["default_ep"].as_object().unwrap() {
        assert_eq!(preset_default_ep(RvcPresetProfile::from_str(preset)), ep.as_str().unwrap(), "{preset}");
    }
    let caps = capabilities::capabilities();
    let has = |list: &[&str], v: &str| list.contains(&v);
    for v in strs(&e["pitch_extractor"]["baseline"]) {
        assert!(has(&caps.pitch_extractors, &v), "{v}");
    }
    for v in strs(&e["rvc_execution_provider"]["baseline"]) {
        assert!(has(&caps.rvc_execution_providers, &v), "{v}");
    }
    for v in strs(&e["rvc_preset"]["baseline"]) {
        assert!(has(&caps.rvc_presets, &v), "{v}");
    }
    for v in strs(&e["precision"]["baseline"]) {
        assert!(has(&caps.precisions, &v), "{v}");
    }
    for v in strs(&e["denoiser"]["baseline"]) {
        assert!(has(&caps.denoisers, &v), "{v}");
    }
    let caps_json = serde_json::to_value(&caps).unwrap();
    for key in strs(&c["capabilities"]["fields"]) {
        assert!(caps_json.get(&key).is_some(), "capabilities field {key}");
    }
}

/// P3 numeric ranges and defaults equal the Rust constants.
#[test]
fn p3_numbers_match_contract() {
    let c = contract();
    let n = &c["numbers"];
    let range = |key: &str, (lo, hi): (f64, f64)| {
        assert_eq!(f(&n[key]["min"]), lo, "{key}.min");
        assert_eq!(f(&n[key]["max"]), hi, "{key}.max");
    };
    range("block_ms", BLOCK_MS_RANGE);
    range("lookahead_ms", LOOKAHEAD_MS_RANGE);
    range("rms_mix", (f64::from(RMS_MIX_RANGE.0), f64::from(RMS_MIX_RANGE.1)));
    range("formant_semitones", (f64::from(FORMANT_SEMITONES_RANGE.0), f64::from(FORMANT_SEMITONES_RANGE.1)));
    range("noise_scale", (f64::from(NOISE_SCALE_RANGE.0), f64::from(NOISE_SCALE_RANGE.1)));
    range("silence_db", (f64::from(SILENCE_DB_RANGE.0), f64::from(SILENCE_DB_RANGE.1)));
    range("hot_slots", (f64::from(HOT_SLOTS_RANGE.0), f64::from(HOT_SLOTS_RANGE.1)));
    range("vram_budget_mb", (f64::from(VRAM_BUDGET_MB_RANGE.0), f64::from(VRAM_BUDGET_MB_RANGE.1)));
    let got = serde_json::to_value(VoiceChangerState::new().snapshot_params()).unwrap();
    for key in ["block_ms", "lookahead_ms", "rms_mix", "formant_semitones", "noise_scale", "silence_db", "hot_slots", "vram_budget_mb"] {
        let want = f(&n[key]["default"]);
        let have = got[key].as_f64().unwrap_or_else(|| panic!("{key} missing"));
        assert!((have - want).abs() < 1e-5, "{key}: {have} vs {want}");
    }
}

/// A fresh voice changer reports every P3 params field under its contract name with the contract
/// default (and `get` never omits them); the hot and per-voice lists name real fields.
#[test]
fn p3_params_fields_travel_with_contract_defaults() {
    let c = contract();
    let e = &c["enums"];
    let got = serde_json::to_value(VoiceChangerState::new().snapshot_params()).unwrap();
    for key in c["params_p3_new_fields"].as_object().unwrap().keys() {
        assert!(!got[key].is_null(), "params field {key} missing from get_voice_changer_params");
    }
    assert_eq!(got["slot_id"], "");
    assert_eq!(got["precision"], e["precision"]["default"]);
    assert_eq!(got["f0_range"], e["f0_range"]["default"]);
    assert_eq!(got["denoiser"], e["denoiser"]["default"]);
    for l in Lever::ALL {
        assert_eq!(got[l.as_str()], e["lever_mode"]["default"], "{}", l.as_str());
    }
    for key in strs(&c["params_p3_hot_fields"]) {
        assert!(got.get(&key).is_some(), "{key}");
    }
    let slot = serde_json::to_value(VoiceChangerState::new().slot_settings()).unwrap();
    for key in strs(&c["slot_json_p3_fields"]) {
        assert!(!slot[&key].is_null(), "slot.json field {key}");
    }
}

/// Every P3 command of the contract is in the plugin's command list (the ACL build input).
#[test]
fn p3_commands_are_listed_for_the_acl() {
    let c = contract();
    let listed: Vec<&str> = include_str!("../../../plugin-commands/voice-changer.txt")
        .lines()
        .map(str::trim)
        .filter(|l| !l.is_empty() && !l.starts_with('#'))
        .collect();
    for key in c["commands_p3"].as_object().unwrap().keys().filter(|k| !k.starts_with('$')) {
        assert!(listed.contains(&key.as_str()), "{key} missing from plugin-commands/voice-changer.txt");
    }
}

/// P3B (P3B-S0): the new params travel under their contract names with the contract defaults
/// (automatic as `-1`), the enums normalize as listed, the ranges equal the Rust constants, and the
/// new levers are offered only when their module says it runs.
#[test]
fn p3b_params_match_contract() {
    use super::params::{
        auto_or_clamped, DeviceBlockMode, OutputCushion, AUTO_MS, CONTENT_CONTEXT_MS_RANGE, SOLA_SEARCH_MS_RANGE,
    };
    let c = contract();
    let e = &c["enums"];
    let n = &c["numbers"];
    let got = serde_json::to_value(VoiceChangerState::new().snapshot_params()).unwrap();
    for key in c["params_p3b_new_fields"].as_object().unwrap().keys() {
        assert!(!got[key].is_null(), "params field {key} missing from get_voice_changer_params");
    }
    for (name, values) in [
        ("output_cushion", OutputCushion::ALL.map(|v| v.as_str().to_string()).to_vec()),
        ("device_block", DeviceBlockMode::ALL.map(|v| v.as_str().to_string()).to_vec()),
    ] {
        assert_eq!(strs(&e[name]["values"]), values, "{name}");
        assert_eq!(got[name], e[name]["default"], "{name} default");
    }
    for v in strs(&e["output_cushion"]["values"]) {
        assert_eq!(OutputCushion::from_str_lossy(&v.to_uppercase()).as_str(), v);
    }
    assert_eq!(OutputCushion::from_str_lossy("bogus").as_str(), e["output_cushion"]["fallback"].as_str().unwrap());
    for v in strs(&e["device_block"]["values"]) {
        assert_eq!(DeviceBlockMode::from_str_lossy(&v.to_uppercase()).as_str(), v);
    }
    assert_eq!(DeviceBlockMode::from_str_lossy("bogus").as_str(), e["device_block"]["fallback"].as_str().unwrap());
    for (key, range) in [("sola_search_ms", SOLA_SEARCH_MS_RANGE), ("content_context_ms", CONTENT_CONTEXT_MS_RANGE)] {
        assert_eq!((f(&n[key]["min"]), f(&n[key]["max"])), range, "{key} range");
        assert_eq!(f(&n[key]["auto"]), AUTO_MS, "{key} auto");
        assert_eq!(f(&n[key]["default"]), AUTO_MS, "{key} default");
        assert_eq!(got[key].as_f64(), Some(AUTO_MS), "{key} on the wire");
        assert_eq!(auto_or_clamped(AUTO_MS, range), None);
        assert_eq!(auto_or_clamped(range.1 * 2.0, range), Some(range.1));
    }
    for (lever, available) in [
        (Lever::ContentNorm, crate::services::neural_audio::vc::engine::content::running_norm_available()),
        (Lever::F0Decode, crate::services::neural_audio::vc::f0_decode::available()),
    ] {
        assert_eq!(capabilities::lever_available(lever), available, "{}", lever.as_str());
        assert_eq!(got[lever.as_str()], e["lever_mode"]["default"], "{}", lever.as_str());
    }
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

/// The importer's error codes (the UI maps them to messages) are the contract's; a new
/// `ErrorCode` variant fails to compile here until it is listed (the match has no wildcard).
#[test]
fn import_error_codes_match_contract() {
    use crate::services::neural_audio::vc::import::ErrorCode::{self, *};
    let all: [ErrorCode; 18] = [
        Io,
        Job,
        NotACheckpoint,
        LegacyFormat,
        TorchScript,
        ArchiveRejected,
        AmbiguousArchive,
        PickleRejected,
        TensorInvalid,
        MissingTensor,
        ShapeMismatch,
        NonFinite,
        UnsupportedModel,
        NoTemplate,
        LayoutInvalid,
        Session,
        TimeLimit,
        PrecisionRefused,
    ];
    for code in all {
        match code {
            Io | Job | NotACheckpoint | LegacyFormat | TorchScript | ArchiveRejected | AmbiguousArchive | PickleRejected
            | TensorInvalid | MissingTensor | ShapeMismatch | NonFinite | UnsupportedModel | NoTemplate | LayoutInvalid
            | Session | TimeLimit | PrecisionRefused => {}
        }
    }
    assert_eq!(strs(&contract()["import_error_codes"]), all.map(|c| c.as_str().to_string()));
}

/// P5 (P5-S0): the reference voice travels as a per-voice param (stored only as a valid library id),
/// the families' ids, formats, info fields, creation kinds and controls are the contract's, and the
/// P5 commands are listed for the ACL.
#[test]
fn p5_families_and_reference_param_match_contract() {
    use crate::services::neural_audio::vc::families::{FamilyKind, CONTROL_IDS};
    let c = contract();
    let got = serde_json::to_value(VoiceChangerState::new().snapshot_params()).unwrap();
    for key in c["params_p5_new_fields"].as_object().unwrap().keys() {
        assert!(!got[key].is_null(), "params field {key} missing from get_voice_changer_params");
    }
    assert_eq!(got["reference_voice_id"], "");
    let slot = serde_json::to_value(VoiceChangerState::new().slot_settings()).unwrap();
    for key in strs(&c["slot_json_p5_fields"]) {
        assert!(!slot[&key].is_null(), "slot.json field {key}");
    }
    let st = VoiceChangerState::new();
    for (sent, stored) in [(" alice ", "alice"), ("../evil", ""), ("Bob", ""), ("", "")] {
        let p = VoiceChangerParams { reference_voice_id: Some(sent.into()), ..st.snapshot_params() };
        let (effective, _) = st.update_params(p);
        assert_eq!(effective.reference_voice_id.as_deref(), Some(stored), "{sent:?}");
    }

    let fam = &c["families"];
    assert_eq!(strs(&fam["ids"]), FamilyKind::ALL.map(|k| k.as_str().to_string()));
    for k in FamilyKind::ALL {
        assert_eq!(fam["formats"][k.as_str()], k.format(), "{}", k.as_str());
    }
    assert_eq!(strs(&fam["controls"]), CONTROL_IDS.map(str::to_string));
    let caps = serde_json::to_value(capabilities::capabilities()).unwrap();
    let creation = strs(&fam["creation"]);
    for entry in caps["families"].as_array().unwrap() {
        let mut keys: Vec<String> = entry.as_object().unwrap().keys().cloned().collect();
        keys.sort();
        let mut want = strs(&fam["info_fields"]);
        want.sort();
        assert_eq!(keys, want);
        assert!(creation.contains(&entry["creation"].as_str().unwrap().to_string()));
    }
    let listed: Vec<&str> = include_str!("../../../plugin-commands/voice-changer.txt")
        .lines()
        .map(str::trim)
        .filter(|l| !l.is_empty() && !l.starts_with('#'))
        .collect();
    for key in c["commands_p5"].as_object().unwrap().keys().filter(|k| !k.starts_with('$')) {
        assert!(listed.contains(&key.as_str()), "{key} missing from plugin-commands/voice-changer.txt");
    }
    // SlotInfo carries the P5 fields.
    let info = serde_json::to_value(crate::services::neural_audio::vc::import::slot::SlotInfo::default()).unwrap();
    for key in strs(&fam["slot_info_p5_fields"]) {
        assert!(info.get(&key).is_some(), "SlotInfo field {key}");
    }
}
