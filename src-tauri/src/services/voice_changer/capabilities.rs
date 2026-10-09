//! What this build of the voice changer can run, and the lever resolution (P3-S0, frozen).
//!
//! Scaffold-owned aggregator: it only reads the `available()` functions of the modules that
//! implement each feature, so a workstream turns its feature on by making its own `available()`
//! return true (P3-B importer, P3-C1 engine, P3-C2 GPU layer, P3-D pitch bank / retrieval, P3-E
//! denoisers / governor, P3-F autotune / live record). Nothing here needs editing when a feature
//! lands.
//!
//! - IPC `voice_changer_get_capabilities` → [`VcCapabilities`] (TS mirror `VcCapabilities` in
//!   `src/core/services/voice_changer/ipc.ts`): the values the UI may offer. Baseline values
//!   (contract `baseline` lists) are always present; P3 values appear once they run.
//! - [`resolve_levers`]: request + availability + the preset's benchmark-decided `auto` default
//!   (`engine::presets::auto_lever`) → `EffectiveLevers`. A lever that is off, unavailable or not
//!   applicable to the EP rung falls back to the legacy path; a requested (`on`) one that cannot
//!   run gets a note.
//! - P5-S0: `families` lists every P5 voice family (`vc::families`) with its facts and whether this
//!   build runs it (`FamilyKind::available`); `slot_formats` gains the formats of the families that
//!   run.
//! - P3B-S0: the `content_norm` and `f0_decode` levers read `content::running_norm_available` and
//!   `vc::f0_decode::available`.

use serde::Serialize;
use tauri::command;

use super::params::{
    accelerator_of, base_ep, pitch_extractor_available, HOT_SLOTS_RANGE, PITCH_EXTRACTORS, RVC_EXECUTION_PROVIDERS,
    RVC_PRESETS,
};
use crate::services::neural_audio::denoise;
use crate::services::neural_audio::gpu;
use crate::services::neural_audio::vc::engine::{
    self, presets, Accelerator, DenoiserKind, EffectiveLevers, Lever, LeverMode, LeverRequest, LeverSet, PrecisionMode,
};
use crate::services::neural_audio::vc::families::{FamilyInfo, FamilyKind};
use crate::services::neural_audio::vc::import;
use crate::services::neural_audio::vc::manifest::{LEGACY_RVC_FORMAT, SIGIL_RVC_FORMAT};
use crate::services::neural_audio::RvcPresetProfile;

/// The values this build runs (snake_case; contract `capabilities`).
#[derive(Clone, Debug, Serialize)]
pub struct VcCapabilities {
    pub pitch_extractors: Vec<&'static str>,
    pub rvc_execution_providers: Vec<&'static str>,
    pub rvc_presets: Vec<&'static str>,
    pub precisions: Vec<&'static str>,
    pub denoisers: Vec<&'static str>,
    /// Levers the engine can switch on (the others always run their legacy path).
    pub levers: Vec<&'static str>,
    /// `manifest.format` values the engine factory runs.
    pub slot_formats: Vec<&'static str>,
    /// `.pth` / `.zip` import and the slot store (P3-B).
    pub importer: bool,
    /// Per-PC autotuner (P3-F).
    pub autotune: bool,
    /// In-app live-record benchmark (P3-F).
    pub live_record: bool,
    pub max_hot_slots: u32,
    /// P5 voice families (P5-S0): every family, with whether this build runs it.
    pub families: Vec<FamilyCapability>,
}

/// One P5 voice family as the UI sees it (P5-S0, frozen; TS `VcFamilyCapability`): its static facts
/// (`FamilyInfo`, flattened) and whether this build runs it.
#[derive(Clone, Debug, Serialize)]
pub struct FamilyCapability {
    #[serde(flatten)]
    pub info: FamilyInfo,
    pub available: bool,
}

/// Every P5 family with its availability.
pub(crate) fn family_capabilities() -> Vec<FamilyCapability> {
    FamilyKind::ALL
        .into_iter()
        .map(|k| FamilyCapability { info: k.info(), available: k.available() })
        .collect()
}

/// Whether this build can switch `lever` on at all (EP applicability is [`lever_applies`]).
pub(crate) fn lever_available(lever: Lever) -> bool {
    match lever {
        Lever::PartialDecode | Lever::AnchoredNoise | Lever::NsfPhase0 | Lever::F0Cache => engine::available(),
        Lever::GpuKnn => engine::retrieval::gpu_knn_available(),
        Lever::IoBinding => gpu::io_binding::available(),
        Lever::CudaGraphs => gpu::io_binding::available() && gpu::cuda_graph::available(),
        Lever::GpuPriority => gpu::cuda_stream::available() || gpu::dml_queue::available(),
        Lever::Governor => engine::governor::available(),
        // P3B-S0: offered once their workstream lands (P3B-D, P3B-F).
        Lever::ContentNorm => engine::content::running_norm_available(),
        Lever::F0Decode => crate::services::neural_audio::vc::f0_decode::available(),
    }
}

/// Whether `lever` applies on the EP ladder rung `base` (`cpu` / `cuda` / `dml`).
pub(crate) fn lever_applies(lever: Lever, base: &str) -> bool {
    match lever {
        Lever::GpuKnn | Lever::IoBinding => base != "cpu",
        Lever::CudaGraphs => base == "cuda",
        Lever::GpuPriority => match base {
            "cuda" => gpu::cuda_stream::available(),
            "dml" => gpu::dml_queue::available(),
            _ => false,
        },
        _ => true,
    }
}

/// Whether an accelerator runs in this build.
pub(crate) fn accelerator_available(accelerator: Accelerator) -> bool {
    match accelerator {
        Accelerator::None => true,
        Accelerator::TensorRtRtx => gpu::trt::available(),
        Accelerator::DmlHighPriority => cfg!(windows) && gpu::dml_queue::available(),
    }
}

/// The values the UI may offer.
pub(crate) fn capabilities() -> VcCapabilities {
    let engine_ok = engine::available();
    VcCapabilities {
        pitch_extractors: PITCH_EXTRACTORS.into_iter().filter(|p| pitch_extractor_available(p)).collect(),
        rvc_execution_providers: RVC_EXECUTION_PROVIDERS
            .into_iter()
            .filter(|ep| accelerator_available(accelerator_of(ep)))
            .collect(),
        rvc_presets: RVC_PRESETS
            .into_iter()
            .filter(|p| *p != "studio_trt" || accelerator_available(Accelerator::TensorRtRtx))
            .collect(),
        precisions: PrecisionMode::ALL
            .into_iter()
            .filter(|p| *p != PrecisionMode::Fp16 || engine_ok)
            .map(PrecisionMode::as_str)
            .collect(),
        denoisers: DenoiserKind::ALL
            .into_iter()
            .filter(|&d| denoise::available(d))
            .map(DenoiserKind::as_str)
            .collect(),
        levers: Lever::ALL.into_iter().filter(|&l| lever_available(l)).map(Lever::as_str).collect(),
        slot_formats: {
            let mut formats = if engine_ok {
                vec![LEGACY_RVC_FORMAT, SIGIL_RVC_FORMAT]
            } else {
                vec![LEGACY_RVC_FORMAT]
            };
            formats.extend(FamilyKind::ALL.into_iter().filter(|k| k.available()).map(FamilyKind::format));
            formats
        },
        importer: import::slot::available() && import::worker::available(),
        autotune: crate::services::bench::autotune::available(),
        live_record: crate::services::bench::live_record::available(),
        max_hot_slots: if engine_ok { HOT_SLOTS_RANGE.1 } else { HOT_SLOTS_RANGE.0 },
        families: family_capabilities(),
    }
}

/// The levers, accelerator and precision a run uses: `request` (per lever), the preset's `auto`
/// defaults, what this build runs and what applies on the rung of `requested_ep` (an RVC EP id,
/// possibly an accelerator). Requested levers or accelerators that cannot run get a note.
pub(crate) fn resolve_levers(
    request: &LeverRequest,
    preset: RvcPresetProfile,
    requested_ep: &str,
    precision: PrecisionMode,
) -> EffectiveLevers {
    let base = base_ep(requested_ep);
    let mut levers = LeverSet::default();
    let mut notes = Vec::new();
    for lever in Lever::ALL {
        let mode = request.get(lever);
        let runnable = lever_available(lever) && lever_applies(lever, base);
        let on = match mode {
            LeverMode::Off => false,
            LeverMode::On => runnable,
            LeverMode::Auto => runnable && presets::auto_lever(preset, lever, base),
        };
        if mode == LeverMode::On && !on {
            let why = if lever_available(lever) {
                format!("does not apply on the {base} execution provider")
            } else if lever == Lever::ContentNorm {
                // Offered only when the content encoder has an installed stream variant (P3B-D).
                "needs the content encoder's stream variant, which is not installed".to_string()
            } else {
                "not available in this build".to_string()
            };
            notes.push(format!("{}: {why}", lever.as_str()));
        }
        levers.set(lever, on);
    }
    let wanted = accelerator_of(requested_ep);
    let accelerator = if accelerator_available(wanted) {
        wanted
    } else {
        notes.push(format!("{}: not available in this build; running on {base}", wanted.as_str()));
        Accelerator::None
    };
    EffectiveLevers { levers, accelerator, precision, notes }
}

/// `voice_changer_get_capabilities`.
#[command]
pub(super) fn voice_changer_get_capabilities() -> VcCapabilities {
    capabilities()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::services::voice_changer::params::{
        normalize_denoiser, normalize_execution_provider, normalize_pitch_extractor, normalize_precision, normalize_rvc_preset,
        BASELINE_PITCH_EXTRACTORS,
    };

    /// Every offered value is a normalized contract value, and the baseline is always offered.
    #[test]
    fn capabilities_offer_normalized_values_and_the_baseline() {
        let c = capabilities();
        for p in &c.pitch_extractors {
            assert_eq!(normalize_pitch_extractor(p), *p);
        }
        for p in &BASELINE_PITCH_EXTRACTORS {
            assert!(c.pitch_extractors.contains(p), "{p}");
        }
        for ep in &c.rvc_execution_providers {
            assert_eq!(normalize_execution_provider(ep), *ep);
        }
        for ep in ["cpu", "cuda", "dml"] {
            assert!(c.rvc_execution_providers.contains(&ep), "{ep}");
        }
        for p in &c.rvc_presets {
            assert_eq!(normalize_rvc_preset(p), *p);
        }
        for p in ["streamer", "low_latency", "quality"] {
            assert!(c.rvc_presets.contains(&p));
        }
        for p in &c.precisions {
            assert_eq!(normalize_precision(p), *p);
        }
        assert!(c.precisions.contains(&"auto") && c.precisions.contains(&"fp32"));
        for d in &c.denoisers {
            assert_eq!(normalize_denoiser(d), *d);
        }
        assert!(c.denoisers.contains(&"off"));
        for l in &c.levers {
            assert!(Lever::from_name(l).is_some());
        }
        assert!(c.slot_formats.contains(&LEGACY_RVC_FORMAT));
        assert!(c.max_hot_slots >= 1);
        let v = serde_json::to_value(&c).unwrap();
        for key in ["pitch_extractors", "rvc_execution_providers", "rvc_presets", "precisions", "denoisers", "levers", "slot_formats", "importer", "autotune", "live_record", "max_hot_slots", "families"] {
            assert!(v.get(key).is_some(), "{key}");
        }
        // Every family is listed once, flattened, with its availability; only running ones add a
        // slot format.
        let fams = v["families"].as_array().unwrap();
        assert_eq!(fams.len(), FamilyKind::ALL.len());
        for (f, k) in fams.iter().zip(FamilyKind::ALL) {
            assert_eq!(f["family"], k.as_str());
            assert_eq!(f["format"], k.format());
            assert_eq!(f["available"], k.available());
            assert!(f["execution_providers"].is_array() && f["controls"].is_array());
            assert_eq!(c.slot_formats.contains(&k.format()), k.available());
        }
    }

    /// Off is always off; On runs only what this build runs on that rung (else a note); Auto follows
    /// the preset table. Written against `lever_available`, so it holds as features land.
    #[test]
    fn levers_resolve_to_the_legacy_path_unless_runnable() {
        let mut on = LeverRequest::default();
        let mut off = LeverRequest::default();
        for lever in Lever::ALL {
            on.set(lever, LeverMode::On);
            off.set(lever, LeverMode::Off);
        }
        for ep in ["cpu", "cuda", "dml", "tensorrt_rtx", "dml_hp"] {
            let base = base_ep(ep);
            let e = resolve_levers(&off, RvcPresetProfile::Quality, ep, PrecisionMode::Auto);
            assert_eq!(e.levers.enabled().count(), 0, "{ep}");
            let e = resolve_levers(&on, RvcPresetProfile::Quality, ep, PrecisionMode::Fp32);
            assert_eq!(e.precision, PrecisionMode::Fp32);
            for lever in Lever::ALL {
                let runnable = lever_available(lever) && lever_applies(lever, base);
                assert_eq!(e.levers.get(lever), runnable, "{ep} {}", lever.as_str());
                let noted = e.notes.iter().any(|n| n.starts_with(&format!("{}:", lever.as_str())));
                assert_eq!(noted, !runnable, "{ep} {}", lever.as_str());
            }
            let auto = resolve_levers(&LeverRequest::default(), RvcPresetProfile::Quality, ep, PrecisionMode::Auto);
            for lever in Lever::ALL {
                let expect = lever_available(lever)
                    && lever_applies(lever, base)
                    && presets::auto_lever(RvcPresetProfile::Quality, lever, base);
                assert_eq!(auto.levers.get(lever), expect);
            }
            assert!(auto.notes.iter().all(|n| !n.contains(": not available in this build;")) || accelerator_of(ep) != Accelerator::None);
        }
        // An accelerator this build cannot run falls back to its base rung with a note.
        let e = resolve_levers(&off, RvcPresetProfile::StudioTrt, "tensorrt_rtx", PrecisionMode::Auto);
        if gpu::trt::available() {
            assert_eq!(e.accelerator, Accelerator::TensorRtRtx);
        } else {
            assert_eq!(e.accelerator, Accelerator::None);
            assert!(e.notes.iter().any(|n| n.starts_with("tensorrt_rtx: not available")), "{:?}", e.notes);
        }
    }
}
