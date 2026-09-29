//! Runtime-pack build variants and what this PC can run.
//!
//! Every Sigil-built engine DLL exists in up to five variants (the CMake presets in
//! `native/cmake/SigilVariant.cmake`): `cuda13`, `vulkan`, `cpu-avx512`, `cpu-avx2` and `cpu`
//! (SSE4.2 baseline). The GPU variants compile their CPU fallback kernels with the `cpu-avx2`
//! baseline (one DLL per variant, ggml linked statically with `GGML_BACKEND_DL=OFF`), so they also
//! need AVX2. Selection walks [`RuntimeVariant::PREFERENCE`] and takes the first variant that is
//! installed and supported; a variant that then fails to load or to create its backend falls
//! through to the next one (`pack::open_engine`).
//!
//! Instruction sets are a hard requirement everywhere, even for an explicitly requested variant or
//! folder: a DLL built for AVX2 / AVX-512 loads fine on a CPU without them and then dies with
//! `STATUS_ILLEGAL_INSTRUCTION`, taking Sigil with it ([`VariantSupport::cpu_supports`]). Only a
//! missing GPU is "try anyway", because the engine reports it as an error.

use serde::{Deserialize, Serialize};

use crate::services::diagnostics::gpu_mem;

/// Build variant of a runtime pack.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, PartialOrd, Ord, Serialize, Deserialize)]
#[serde(rename_all = "kebab-case")]
pub enum RuntimeVariant {
    Cuda13,
    Vulkan,
    CpuAvx512,
    CpuAvx2,
    Cpu,
}

impl RuntimeVariant {
    /// Selection order: the first available and supported variant wins.
    pub const PREFERENCE: [RuntimeVariant; 5] = [
        RuntimeVariant::Cuda13,
        RuntimeVariant::Vulkan,
        RuntimeVariant::CpuAvx512,
        RuntimeVariant::CpuAvx2,
        RuntimeVariant::Cpu,
    ];

    pub fn as_str(self) -> &'static str {
        match self {
            RuntimeVariant::Cuda13 => "cuda13",
            RuntimeVariant::Vulkan => "vulkan",
            RuntimeVariant::CpuAvx512 => "cpu-avx512",
            RuntimeVariant::CpuAvx2 => "cpu-avx2",
            RuntimeVariant::Cpu => "cpu",
        }
    }

    pub fn parse(s: &str) -> Option<RuntimeVariant> {
        Self::PREFERENCE.into_iter().find(|v| v.as_str() == s)
    }

    pub fn is_gpu(self) -> bool {
        matches!(self, RuntimeVariant::Cuda13 | RuntimeVariant::Vulkan)
    }

    /// Human name for messages.
    pub fn label(self) -> &'static str {
        match self {
            RuntimeVariant::Cuda13 => "CUDA 13",
            RuntimeVariant::Vulkan => "Vulkan",
            RuntimeVariant::CpuAvx512 => "CPU (AVX-512)",
            RuntimeVariant::CpuAvx2 => "CPU (AVX2)",
            RuntimeVariant::Cpu => "CPU",
        }
    }
}

impl std::fmt::Display for RuntimeVariant {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(self.as_str())
    }
}

/// What this PC can run.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct VariantSupport {
    /// NVIDIA Turing or newer with driver R580+ (sysinfo EP eligibility) and the CUDA 13 runtime.
    pub cuda13: bool,
    /// A Vulkan 1.2+ capable GPU.
    pub vulkan: bool,
    pub avx512: bool,
    pub avx2: bool,
}

/// CPU features of the `cpu-avx2` variant (MSVC `/arch:AVX2` plus ggml's FMA / F16C / BMI2).
#[cfg(target_arch = "x86_64")]
fn cpu_avx2() -> bool {
    std::arch::is_x86_feature_detected!("avx2")
        && std::arch::is_x86_feature_detected!("fma")
        && std::arch::is_x86_feature_detected!("f16c")
        && std::arch::is_x86_feature_detected!("bmi2")
}

/// CPU features of the `cpu-avx512` variant (MSVC `/arch:AVX512`: F, CD, BW, DQ and VL; the OS
/// must also save the AVX-512 state, which `is_x86_feature_detected!` checks).
#[cfg(target_arch = "x86_64")]
fn cpu_avx512() -> bool {
    cpu_avx2()
        && std::arch::is_x86_feature_detected!("avx512f")
        && std::arch::is_x86_feature_detected!("avx512cd")
        && std::arch::is_x86_feature_detected!("avx512bw")
        && std::arch::is_x86_feature_detected!("avx512dq")
        && std::arch::is_x86_feature_detected!("avx512vl")
}

/// Whether the Vulkan loader is installed (`System32\vulkan-1.dll`, shipped by every current GPU
/// driver). Only the file is checked: loading it would pull the GPU drivers into the process.
fn vulkan_loader_present() -> bool {
    #[cfg(windows)]
    {
        let root = std::env::var_os("SystemRoot")
            .or_else(|| std::env::var_os("windir"))
            .map(std::path::PathBuf::from)
            .unwrap_or_else(|| std::path::PathBuf::from(r"C:\Windows"));
        root.join("System32").join("vulkan-1.dll").is_file()
    }
    #[cfg(not(windows))]
    {
        false
    }
}

/// Vulkan support from the adapter list: a hardware GPU from a vendor that ships Vulkan 1.2+
/// drivers, plus the loader. A driver that turns out not to support the engine's Vulkan level makes
/// the engine's create fail with `SIGIL_E_BACKEND_UNAVAILABLE`, and the next variant is used.
pub(crate) fn vulkan_from_adapters(adapters: &[gpu_mem::GpuAdapter], loader_present: bool) -> bool {
    loader_present
        && adapters.iter().any(|a| {
            !a.software && matches!(a.vendor_id, gpu_mem::VENDOR_NVIDIA | gpu_mem::VENDOR_AMD | gpu_mem::VENDOR_INTEL)
        })
}

impl VariantSupport {
    /// CPU features only (instant; no GPU driver is touched).
    pub fn detect_cpu() -> VariantSupport {
        #[cfg(target_arch = "x86_64")]
        {
            VariantSupport {
                cuda13: false,
                vulkan: false,
                avx512: cpu_avx512(),
                avx2: cpu_avx2(),
            }
        }
        #[cfg(not(target_arch = "x86_64"))]
        {
            VariantSupport::default()
        }
    }

    /// Detect support: CPU features, CUDA 13 eligibility from the (cached) system probe (NVIDIA
    /// Turing or newer, driver R580+), Vulkan from the DXGI adapters and the Vulkan loader. The
    /// first call may wait for the system probe (up to its GPU timeout); later calls are cheap.
    pub fn detect() -> VariantSupport {
        let cpu = Self::detect_cpu();
        let info = crate::services::sysinfo::cached();
        let adapters: Vec<gpu_mem::GpuAdapter> = info.gpus.iter().map(|g| g.adapter.clone()).collect();
        VariantSupport {
            cuda13: info.eps.cuda.eligible,
            vulkan: vulkan_from_adapters(&adapters, vulkan_loader_present()),
            ..cpu
        }
    }

    /// Whether this CPU can execute the code of a Sigil-built `variant` DLL (instruction sets
    /// only; GPU availability is not considered). GPU variants carry `cpu-avx2` CPU kernels.
    ///
    /// Unlike a missing GPU, which makes the engine's create fail with an error, a missing
    /// instruction set is not detected by the DLL: it loads, and the first vector loop kills the
    /// process with `STATUS_ILLEGAL_INSTRUCTION`. A variant failing this check is never loaded.
    pub fn cpu_supports(&self, variant: RuntimeVariant) -> bool {
        match variant {
            RuntimeVariant::Cuda13 | RuntimeVariant::Vulkan | RuntimeVariant::CpuAvx2 => self.avx2,
            RuntimeVariant::CpuAvx512 => self.avx512 && self.avx2,
            RuntimeVariant::Cpu => true,
        }
    }

    /// Why this CPU cannot execute `variant`'s code (`None` when it can); see [`Self::cpu_supports`].
    pub fn cpu_unsupported_reason(&self, variant: RuntimeVariant) -> Option<String> {
        if self.cpu_supports(variant) {
            return None;
        }
        Some(match variant {
            RuntimeVariant::CpuAvx512 if self.avx2 => "this CPU does not support AVX-512".into(),
            RuntimeVariant::CpuAvx512 => "this CPU does not support AVX-512 (or AVX2)".into(),
            v if v.is_gpu() => format!("this CPU does not support AVX2, which the CPU code of the {} build needs", v.label()),
            _ => "this CPU does not support AVX2".into(),
        })
    }

    /// Refuse a build whose instruction sets this CPU lacks, by its variant string (a pack
    /// manifest's `variant`, or the DLL's own `sigil_engine_get_info` variant). A string that is not
    /// a known variant is not judged (development builds may use their own labels).
    pub fn check_build(&self, variant: &str) -> Result<(), String> {
        match RuntimeVariant::parse(variant).and_then(|v| self.cpu_unsupported_reason(v).map(|why| (v, why))) {
            None => Ok(()),
            Some((v, why)) => Err(format!(
                "it is the {} build, which cannot run on this PC: {why} (install the {} variant instead)",
                v.as_str(),
                if self.avx2 { RuntimeVariant::CpuAvx2.as_str() } else { RuntimeVariant::Cpu.as_str() }
            )),
        }
    }

    /// Whether `variant` can run here: its instruction sets ([`Self::cpu_supports`]) and, for the
    /// GPU variants, the GPU.
    pub fn supports(&self, variant: RuntimeVariant) -> bool {
        self.cpu_supports(variant)
            && match variant {
                RuntimeVariant::Cuda13 => self.cuda13,
                RuntimeVariant::Vulkan => self.vulkan,
                _ => true,
            }
    }

    /// Why `variant` cannot run here (`None` when it can).
    pub fn unsupported_reason(&self, variant: RuntimeVariant) -> Option<String> {
        if self.supports(variant) {
            return None;
        }
        match variant {
            RuntimeVariant::Cuda13 if !self.cuda13 => Some(CUDA13_UNSUPPORTED.into()),
            RuntimeVariant::Vulkan if !self.vulkan => Some(VULKAN_UNSUPPORTED.into()),
            _ => self.cpu_unsupported_reason(variant),
        }
    }

    /// Why a runtime pack that is NOT a Sigil-built engine cannot be used here (`None` when it
    /// can): the NVIDIA runtime, the ONNX Runtime CUDA provider and third-party servers. Their GPU
    /// variants need only the GPU (their CPU code is not built with Sigil's `cpu-avx2` baseline:
    /// ONNX Runtime's CUDA EP runs on any x64 CPU); their CPU variants need the instruction set
    /// they are named after.
    pub fn runtime_unsupported_reason(&self, variant: RuntimeVariant) -> Option<String> {
        match variant {
            RuntimeVariant::Cuda13 => (!self.cuda13).then(|| CUDA13_UNSUPPORTED.into()),
            RuntimeVariant::Vulkan => (!self.vulkan).then(|| VULKAN_UNSUPPORTED.into()),
            _ => self.cpu_unsupported_reason(variant),
        }
    }
}

const CUDA13_UNSUPPORTED: &str =
    "CUDA 13 needs an NVIDIA GPU of the Turing generation (GTX 16 / RTX 20) or newer with driver R580 or newer";
const VULKAN_UNSUPPORTED: &str = "no Vulkan-capable GPU driver was found";

/// The preferred variant among `available` that this PC supports (deterministic).
#[allow(dead_code)] // Frozen API (P1-S0b); `candidates` is the fallback-aware form.
pub fn best_variant(available: &[RuntimeVariant], support: &VariantSupport) -> Option<RuntimeVariant> {
    RuntimeVariant::PREFERENCE
        .into_iter()
        .find(|v| available.contains(v) && support.supports(*v))
}

/// Variants to try, in order:
/// - `forced`: that variant alone, when it is available and this CPU can execute its code
///   ([`VariantSupport::cpu_supports`]). A missing GPU does not drop it (the caller asked for it and
///   gets the real backend error from create), but a missing instruction set does: such a DLL
///   would crash the process instead of failing.
/// - otherwise every available and supported variant in [`RuntimeVariant::PREFERENCE`] order, so a
///   variant that fails to load falls through to the next.
pub fn candidates(available: &[RuntimeVariant], support: &VariantSupport, forced: Option<RuntimeVariant>) -> Vec<RuntimeVariant> {
    match forced {
        Some(v) => available
            .iter()
            .copied()
            .filter(|a| *a == v && support.cpu_supports(v))
            .take(1)
            .collect(),
        None => RuntimeVariant::PREFERENCE
            .into_iter()
            .filter(|v| available.contains(v) && support.supports(*v))
            .collect(),
    }
}
