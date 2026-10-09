//! Request vocabulary of the Sigil Voice Engine (SVE): the switchable levers, their request
//! modes, and the other option enums the IPC params carry (plan "VC best engine").
//!
//! # Frozen by P3-S0 (change only through INTEGRATE)
//! Scaffold-owned: implementers read these types, they do not edit them. The persisted / IPC
//! strings are `params_contract.json`'s (`enums.lever_mode`, `levers`, `enums.precision`,
//! `enums.f0_range`, `enums.denoiser`, the accelerators of `enums.rvc_execution_provider`), checked
//! by `voice_changer::contract_tests` and the TS `params.test.ts`.
//!
//! Every lever is independently switchable and falls back to the proven legacy path:
//! - `off`, or `on` for a lever this build (or this PC, slot or EP) cannot run, means the legacy
//!   behaviour, and the resolved [`EffectiveLevers::notes`] say why a requested lever is off;
//! - `auto` means the benchmark-decided default of the preset (`engine::presets::auto_lever`,
//!   P3-C1 / P3-F), which starts as "off" everywhere until a measurement flips it.
//!
//! Resolution (request + availability + preset) is `voice_changer::capabilities::resolve_levers`.
//!
//! # Added by P3B-S0 (frozen; persisted strings never renamed)
//! [`Lever::ContentNorm`] (`content_norm`, P3B-D: window-independent content normalisation) and
//! [`Lever::F0Decode`] (`f0_decode`, P3B-F: cross-tick F0 decoding), appended to the contract order.
//! Their `auto` value is off (`presets::auto_lever`: both are opt-in, P3B-I0), and they run only
//! where `content::running_norm_available()` / `vc::f0_decode::available()` say so (both `true`
//! now). `content_norm` also needs the encoder's `stream` variant (an opt-in export,
//! `export_embedders.py --stream`) and turns itself off at load, with a note, when none is
//! installed.

use serde::ser::SerializeMap;
use serde::{Serialize, Serializer};

/// How a lever is requested (params `partial_decode`, `f0_cache`, ...).
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum LeverMode {
    /// The preset's benchmark-decided default.
    #[default]
    Auto,
    On,
    Off,
}

impl LeverMode {
    pub const ALL: [LeverMode; 3] = [LeverMode::Auto, LeverMode::On, LeverMode::Off];

    pub fn as_str(self) -> &'static str {
        match self {
            LeverMode::Auto => "auto",
            LeverMode::On => "on",
            LeverMode::Off => "off",
        }
    }

    /// Case-insensitive; anything unknown is [`LeverMode::Auto`].
    pub fn from_str_lossy(s: &str) -> Self {
        match s.trim().to_ascii_lowercase().as_str() {
            "on" => LeverMode::On,
            "off" => LeverMode::Off,
            _ => LeverMode::Auto,
        }
    }
}

/// The switchable engine features (plan "VC best engine": the quality levers competitors lack).
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub enum Lever {
    /// flowdec over the return window only (P3-C1). Legacy: decode the whole context.
    PartialDecode,
    /// Counter-based, time-anchored Gaussian noise (P3-C1). Legacy: fresh noise every tick.
    AnchoredNoise,
    /// Phase-continuous NSF excitation through `phase0` (P3-C1). Legacy: phase 0 every tick.
    NsfPhase0,
    /// Incremental F0 over the new block plus a margin, with a rolling cache (P3-C1). Legacy: F0
    /// over the whole context every tick.
    F0Cache,
    /// Exact kNN retrieval on the GPU with the index resident (P3-D). Legacy: CPU IVF.
    GpuKnn,
    /// Device-resident IOBinding at fixed shapes (P3-C2). Legacy: host tensors every call.
    IoBinding,
    /// CUDA graphs captured per fixed shape (P3-C2; needs IOBinding). Legacy: eager launches.
    CudaGraphs,
    /// Highest-priority CUDA stream / high-priority DirectML queue (P3-C2). Legacy: default
    /// stream / queue.
    GpuPriority,
    /// Adaptive quality governor (P3-E). Legacy: fixed quality, the latency budget drops audio.
    Governor,
    /// Window-independent content normalisation with running statistics and real left context
    /// (P3B-D). Legacy: every window normalises its own statistics.
    ContentNorm,
    /// Cross-tick F0 decoding over the cached salience (P3B-F; not for singing). Legacy: each
    /// call's own decode.
    F0Decode,
}

impl Lever {
    pub const COUNT: usize = 11;
    /// Every lever, in contract (`levers`) order.
    pub const ALL: [Lever; Lever::COUNT] = [
        Lever::PartialDecode,
        Lever::AnchoredNoise,
        Lever::NsfPhase0,
        Lever::F0Cache,
        Lever::GpuKnn,
        Lever::IoBinding,
        Lever::CudaGraphs,
        Lever::GpuPriority,
        Lever::Governor,
        Lever::ContentNorm,
        Lever::F0Decode,
    ];

    /// The IPC / persisted param name of the lever.
    pub fn as_str(self) -> &'static str {
        match self {
            Lever::PartialDecode => "partial_decode",
            Lever::AnchoredNoise => "anchored_noise",
            Lever::NsfPhase0 => "nsf_phase0",
            Lever::F0Cache => "f0_cache",
            Lever::GpuKnn => "gpu_knn",
            Lever::IoBinding => "io_binding",
            Lever::CudaGraphs => "cuda_graphs",
            Lever::GpuPriority => "gpu_priority",
            Lever::Governor => "governor",
            Lever::ContentNorm => "content_norm",
            Lever::F0Decode => "f0_decode",
        }
    }

    pub fn from_name(name: &str) -> Option<Lever> {
        Lever::ALL.into_iter().find(|l| l.as_str() == name)
    }

    fn index(self) -> usize {
        Lever::ALL.iter().position(|&l| l == self).unwrap_or(0)
    }
}

/// The requested mode of every lever (all [`LeverMode::Auto`] by default).
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct LeverRequest {
    modes: [LeverMode; Lever::COUNT],
}

impl LeverRequest {
    pub fn get(&self, lever: Lever) -> LeverMode {
        self.modes[lever.index()]
    }

    pub fn set(&mut self, lever: Lever, mode: LeverMode) {
        self.modes[lever.index()] = mode;
    }
}

/// Which levers are on. Serializes as `{ "<lever>": bool, ... }` in contract order.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct LeverSet {
    on: [bool; Lever::COUNT],
}

impl LeverSet {
    pub fn get(&self, lever: Lever) -> bool {
        self.on[lever.index()]
    }

    pub fn set(&mut self, lever: Lever, on: bool) {
        self.on[lever.index()] = on;
    }

    /// The levers that are on.
    pub fn enabled(&self) -> impl Iterator<Item = Lever> + '_ {
        Lever::ALL.into_iter().filter(|&l| self.get(l))
    }
}

impl Serialize for LeverSet {
    fn serialize<S: Serializer>(&self, serializer: S) -> Result<S::Ok, S::Error> {
        let mut map = serializer.serialize_map(Some(Lever::COUNT))?;
        for lever in Lever::ALL {
            map.serialize_entry(lever.as_str(), &self.get(lever))?;
        }
        map.end()
    }
}

/// GPU accelerator on top of an EP ladder rung (`rvc_execution_provider` `tensorrt_rtx` /
/// `dml_hp`). Sessions are always built on the base rung (`cuda` / `dml`); the accelerator is
/// applied by `neural_audio::gpu` only when it is available, else the base rung runs as before.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum Accelerator {
    #[default]
    None,
    /// TensorRT-RTX (or classic TensorRT) with an engine cache, on the CUDA rung (P3-C2).
    #[serde(rename = "tensorrt_rtx")]
    TensorRtRtx,
    /// DirectML on a high-priority D3D12 queue through `OrtDmlApi` (P3-C2).
    #[serde(rename = "dml_hp")]
    DmlHighPriority,
}

impl Accelerator {
    pub fn as_str(self) -> &'static str {
        match self {
            Accelerator::None => "none",
            Accelerator::TensorRtRtx => "tensorrt_rtx",
            Accelerator::DmlHighPriority => "dml_hp",
        }
    }
}

/// Graph precision request (params `precision`). `auto` = the template's `layout.json`
/// recommendation per graph and device class (ADR 0004: `enc` fp32; `flowdec` / `knn` fp16 on
/// GPUs, fp32 on the CPU), overridable per PC by the autotuner among variants that pass parity.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum PrecisionMode {
    #[default]
    Auto,
    Fp32,
    Fp16,
}

impl PrecisionMode {
    pub const ALL: [PrecisionMode; 3] = [PrecisionMode::Auto, PrecisionMode::Fp32, PrecisionMode::Fp16];

    pub fn as_str(self) -> &'static str {
        match self {
            PrecisionMode::Auto => "auto",
            PrecisionMode::Fp32 => "fp32",
            PrecisionMode::Fp16 => "fp16",
        }
    }

    /// Case-insensitive; anything unknown is [`PrecisionMode::Auto`].
    pub fn from_str_lossy(s: &str) -> Self {
        match s.trim().to_ascii_lowercase().as_str() {
            "fp32" => PrecisionMode::Fp32,
            "fp16" => PrecisionMode::Fp16,
            _ => PrecisionMode::Auto,
        }
    }
}

/// F0 search range (params `f0_range`, per voice). `speech`: the legacy 50-1100 Hz; `singing`: up
/// to 2.5 kHz with the octave gate tuned for song (P3-D).
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum F0Range {
    #[default]
    Speech,
    Singing,
}

impl F0Range {
    pub const ALL: [F0Range; 2] = [F0Range::Speech, F0Range::Singing];

    pub fn as_str(self) -> &'static str {
        match self {
            F0Range::Speech => "speech",
            F0Range::Singing => "singing",
        }
    }

    /// Case-insensitive; anything unknown is [`F0Range::Speech`].
    pub fn from_str_lossy(s: &str) -> Self {
        match s.trim().to_ascii_lowercase().as_str() {
            "singing" => F0Range::Singing,
            _ => F0Range::Speech,
        }
    }
}

/// Input denoiser on the voice-changer path only (params `denoiser`; STT keeps the raw feed).
/// GTCRN runs at 16 kHz inside the harness; DeepFilterNet3 / DPDFNet at 48 kHz before the aux mix
/// (P3-E, `neural_audio::denoise`).
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum DenoiserKind {
    #[default]
    Off,
    Gtcrn,
    Dfn3,
    Dpdfnet,
}

impl DenoiserKind {
    pub const ALL: [DenoiserKind; 4] = [DenoiserKind::Off, DenoiserKind::Gtcrn, DenoiserKind::Dfn3, DenoiserKind::Dpdfnet];

    pub fn as_str(self) -> &'static str {
        match self {
            DenoiserKind::Off => "off",
            DenoiserKind::Gtcrn => "gtcrn",
            DenoiserKind::Dfn3 => "dfn3",
            DenoiserKind::Dpdfnet => "dpdfnet",
        }
    }

    /// Case-insensitive; anything unknown is [`DenoiserKind::Off`].
    pub fn from_str_lossy(s: &str) -> Self {
        match s.trim().to_ascii_lowercase().as_str() {
            "gtcrn" => DenoiserKind::Gtcrn,
            "dfn3" => DenoiserKind::Dfn3,
            "dpdfnet" => DenoiserKind::Dpdfnet,
            _ => DenoiserKind::Off,
        }
    }
}

/// What a run uses: the levers that are on after request, availability and preset defaults,
/// the accelerator and precision request, and why a requested lever is off. Reported in the
/// voice-changer metrics (`levers`) and benchmark reports.
#[derive(Clone, Debug, Default, PartialEq, Serialize)]
pub struct EffectiveLevers {
    #[serde(flatten)]
    pub levers: LeverSet,
    pub accelerator: Accelerator,
    pub precision: PrecisionMode,
    /// One line per requested (`on`) lever or accelerator that is not used, e.g.
    /// `cuda_graphs: not available in this build`.
    pub notes: Vec<String>,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn names_round_trip_and_unknown_values_fall_back() {
        for lever in Lever::ALL {
            assert_eq!(Lever::from_name(lever.as_str()), Some(lever));
        }
        assert_eq!(Lever::from_name("bogus"), None);
        for m in LeverMode::ALL {
            assert_eq!(LeverMode::from_str_lossy(&m.as_str().to_uppercase()), m);
            assert_eq!(serde_json::to_value(m).unwrap(), m.as_str());
        }
        assert_eq!(LeverMode::from_str_lossy("maybe"), LeverMode::Auto);
        for p in PrecisionMode::ALL {
            assert_eq!(PrecisionMode::from_str_lossy(p.as_str()), p);
        }
        for r in F0Range::ALL {
            assert_eq!(F0Range::from_str_lossy(r.as_str()), r);
        }
        for d in DenoiserKind::ALL {
            assert_eq!(DenoiserKind::from_str_lossy(d.as_str()), d);
            assert_eq!(serde_json::to_value(d).unwrap(), d.as_str());
        }
        for a in [Accelerator::None, Accelerator::TensorRtRtx, Accelerator::DmlHighPriority] {
            assert_eq!(serde_json::to_value(a).unwrap(), a.as_str());
        }
    }

    #[test]
    fn effective_levers_serialize_flat() {
        let mut e = EffectiveLevers::default();
        e.levers.set(Lever::CudaGraphs, true);
        let v = serde_json::to_value(&e).unwrap();
        assert_eq!(v["cuda_graphs"], true);
        assert_eq!(v["partial_decode"], false);
        assert_eq!(v["accelerator"], "none");
        assert_eq!(v["precision"], "auto");
        let mut r = LeverRequest::default();
        r.set(Lever::Governor, LeverMode::On);
        assert_eq!(r.get(Lever::Governor), LeverMode::On);
        assert_eq!(r.get(Lever::F0Cache), LeverMode::Auto);
        assert_eq!(e.levers.enabled().collect::<Vec<_>>(), vec![Lever::CudaGraphs]);
    }
}
