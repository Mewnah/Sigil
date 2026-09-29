//! Execution-provider eligibility: can this PC's GPU, driver and OS run an EP at all?
//!
//! Rules (Sigil links ONNX Runtime 1.28, whose Windows GPU providers are CUDA 13 builds):
//! - **CUDA**: an NVIDIA GPU of compute capability 7.5+ (Turing or newer; CUDA 13 dropped Maxwell,
//!   Pascal and Volta) and a driver that supports CUDA 13 (R580 or newer).
//! - **TensorRT** (classic TensorRT 10 for CUDA 13): the same floor as CUDA.
//! - **TensorRT-RTX**: an NVIDIA GPU of compute capability 8.0+ (Ampere or newer) and R580+.
//! - **DirectML**: Windows and a hardware adapter with Direct3D 12 at feature level 11_0+.
//!
//! Runtime files (the NVIDIA runtime pack, provider DLLs, TensorRT libraries) are a separate,
//! installable concern reported by the diagnostics probe and the runtime packs: `eligible` here
//! means "installing the runtime would make this EP work on this PC".
//!
//! Pure: [`assess`] only looks at its inputs (unit-tested as a table below). Unknown facts never
//! make an EP ineligible; they produce `eligible: true` with code [`UNVERIFIED`], because every GPU
//! session still falls back to the next EP on failure. The one exception is a probe that got no GPU
//! facts at all (the graphics driver did not answer): [`unknown`] reports every GPU EP as not
//! eligible with code [`GPU_UNKNOWN`], so nothing picks a GPU runtime blind, and the UI says the
//! check failed instead of "no graphics card".

use super::arch::{self, NvidiaGen};
use super::cuda::{parse_cuda_version, parse_driver_version, MIN_CUDA_DRIVER_API, MIN_DRIVER_FOR_CUDA13};
use super::dxgi::DIRECTML_MIN_FEATURE_LEVEL;
use super::{Eligibility, EpEligibility, GpuInfo};
use crate::services::diagnostics::gpu_mem::VENDOR_NVIDIA;

// Machine-readable codes (`Eligibility::code`); the UI translates by code.
pub const OK: &str = "ok";
pub const UNVERIFIED: &str = "unverified";
pub const NO_GPU: &str = "no_gpu";
pub const SOFTWARE_ONLY: &str = "software_only";
pub const NO_NVIDIA_GPU: &str = "no_nvidia_gpu";
pub const GPU_TOO_OLD: &str = "gpu_too_old";
pub const DRIVER_TOO_OLD: &str = "driver_too_old";
pub const NO_D3D12: &str = "no_d3d12";
pub const UNSUPPORTED_OS: &str = "unsupported_os";
/// The GPU probe got no answer from the graphics driver: nothing is known about the GPUs.
pub const GPU_UNKNOWN: &str = "gpu_unknown";

/// Oldest NVIDIA generation the CUDA 13 EP (and classic TensorRT) supports.
pub const CUDA_MIN_GEN: NvidiaGen = NvidiaGen::Turing;
/// Oldest NVIDIA generation TensorRT-RTX supports.
pub const TENSORRT_RTX_MIN_GEN: NvidiaGen = NvidiaGen::Ampere;

fn yes(code: &str, reason: Option<String>) -> Eligibility {
    Eligibility {
        eligible: true,
        reason,
        code: code.to_string(),
    }
}

fn no(code: &str, reason: String) -> Eligibility {
    Eligibility {
        eligible: false,
        reason: Some(reason),
        code: code.to_string(),
    }
}

/// What is known about one NVIDIA GPU's generation.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum GenFact {
    /// Exact compute capability (from NVML).
    Cc(u32, u32),
    /// Generation from the PCI id table.
    Gen(NvidiaGen),
    Unknown,
}

fn gen_fact(g: &GpuInfo) -> GenFact {
    if let Some((a, b)) = g
        .compute_capability
        .as_deref()
        .and_then(arch::parse_compute_capability)
    {
        return GenFact::Cc(a, b);
    }
    match g.arch.as_deref().and_then(NvidiaGen::parse) {
        Some(gen) => GenFact::Gen(gen),
        None => GenFact::Unknown,
    }
}

/// `Some(true)` meets the floor, `Some(false)` is older, `None` unknown.
fn meets(fact: GenFact, floor: NvidiaGen) -> Option<bool> {
    match fact {
        GenFact::Cc(a, b) => Some((a, b) >= floor.min_compute_capability()),
        GenFact::Gen(g) => Some(g >= floor),
        GenFact::Unknown => None,
    }
}

/// Whether this one GPU can run the CUDA 13 EP as far as its hardware goes (the driver is checked
/// PC-wide by [`assess`]): `Some(false)` for software renderers, other vendors and NVIDIA cards
/// older than Turing, `None` for an NVIDIA card of unknown generation (treated as capable, like
/// [`UNVERIFIED`]).
pub fn meets_cuda_floor(g: &GpuInfo) -> Option<bool> {
    if g.adapter.software || g.adapter.vendor_id != VENDOR_NVIDIA {
        return Some(false);
    }
    meets(gen_fact(g), CUDA_MIN_GEN)
}

fn describe_gen(g: &GpuInfo) -> String {
    let arch = g
        .arch
        .as_deref()
        .map(capitalize)
        .unwrap_or_else(|| "unknown generation".into());
    match &g.compute_capability {
        Some(cc) => format!("{} ({arch}, compute capability {cc})", g.adapter.name),
        None => format!("{} ({arch})", g.adapter.name),
    }
}

fn capitalize(s: &str) -> String {
    let mut c = s.chars();
    c.next()
        .map(|f| f.to_uppercase().chain(c).collect())
        .unwrap_or_default()
}

/// Driver check shared by the NVIDIA EPs: `Ok(None)` fine, `Ok(Some(note))` unverified,
/// `Err(reason)` too old.
fn driver_check(ep: &str, nvidia_driver: Option<&str>, cuda_driver_version: Option<&str>) -> Result<Option<String>, String> {
    let driver_label = nvidia_driver.unwrap_or("of unknown version");
    if let Some(cuda) = cuda_driver_version.and_then(parse_cuda_version) {
        if cuda < MIN_CUDA_DRIVER_API {
            return Err(format!(
                "NVIDIA driver {driver_label} supports CUDA {}.{}; {ep} needs CUDA {}.{} (driver {} or newer)",
                cuda.0, cuda.1, MIN_CUDA_DRIVER_API.0, MIN_CUDA_DRIVER_API.1, MIN_DRIVER_FOR_CUDA13.0
            ));
        }
        return Ok(None);
    }
    match nvidia_driver.and_then(parse_driver_version) {
        Some(v) if v < MIN_DRIVER_FOR_CUDA13 => Err(format!(
            "NVIDIA driver {driver_label} is too old for {ep}: it needs driver {} or newer (CUDA {}.{})",
            MIN_DRIVER_FOR_CUDA13.0, MIN_CUDA_DRIVER_API.0, MIN_CUDA_DRIVER_API.1
        )),
        Some(_) => Ok(None),
        None => Ok(Some("the NVIDIA driver version could not be read".into())),
    }
}

/// One NVIDIA EP with a GPU-generation floor.
fn nvidia_ep(ep: &str, floor: NvidiaGen, floor_label: &str, gpus: &[GpuInfo], nvidia_driver: Option<&str>, cuda: Option<&str>) -> Eligibility {
    let hardware: Vec<&GpuInfo> = gpus.iter().filter(|g| !g.adapter.software).collect();
    if hardware.is_empty() {
        return no(NO_GPU, format!("{ep} needs an NVIDIA GPU; no graphics card was found"));
    }
    let nvidia: Vec<&GpuInfo> = hardware
        .iter()
        .copied()
        .filter(|g| g.adapter.vendor_id == VENDOR_NVIDIA)
        .collect();
    if nvidia.is_empty() {
        return no(NO_NVIDIA_GPU, format!("{ep} needs an NVIDIA GPU; this PC has none"));
    }
    let verdicts: Vec<(&GpuInfo, Option<bool>)> = nvidia
        .iter()
        .map(|g| (*g, meets(gen_fact(g), floor)))
        .collect();
    let verified = verdicts.iter().any(|(_, v)| *v == Some(true));
    let unknown = verdicts.iter().any(|(_, v)| v.is_none());
    if !verified && !unknown {
        // Every NVIDIA GPU is known to be too old: explain with the newest one.
        let newest = nvidia
            .iter()
            .max_by_key(|g| match gen_fact(g) {
                GenFact::Cc(a, b) => (a, b),
                GenFact::Gen(gen) => gen.min_compute_capability(),
                GenFact::Unknown => (0, 0),
            })
            .copied()
            .unwrap_or(nvidia[0]);
        return no(
            GPU_TOO_OLD,
            format!("{} is too old for {ep}: it needs {floor_label} or newer", describe_gen(newest)),
        );
    }
    let mut notes: Vec<String> = Vec::new();
    match driver_check(ep, nvidia_driver, cuda) {
        Err(reason) => return no(DRIVER_TOO_OLD, reason),
        Ok(Some(note)) => notes.push(note),
        Ok(None) => {}
    }
    if !verified {
        notes.push("the GPU generation could not be verified".into());
    }
    if notes.is_empty() {
        yes(OK, None)
    } else {
        yes(UNVERIFIED, Some(capitalize(&notes.join("; "))))
    }
}

fn directml(gpus: &[GpuInfo], windows: bool) -> Eligibility {
    if !windows {
        return no(UNSUPPORTED_OS, "DirectML is only available on Windows".into());
    }
    let hardware: Vec<&GpuInfo> = gpus.iter().filter(|g| !g.adapter.software).collect();
    if hardware.is_empty() {
        return if gpus.is_empty() {
            no(NO_GPU, "DirectML needs a graphics card; none was found".into())
        } else {
            no(
                SOFTWARE_ONLY,
                "only the Microsoft Basic Render Driver is available; install the graphics driver".into(),
            )
        };
    }
    if hardware.iter().any(|g| g.d3d12 == Some(true)) {
        return yes(OK, None);
    }
    if hardware.iter().any(|g| g.d3d12.is_none()) {
        return yes(UNVERIFIED, Some("Direct3D 12 support could not be tested".into()));
    }
    let names: Vec<&str> = hardware.iter().map(|g| g.adapter.name.as_str()).collect();
    no(
        NO_D3D12,
        format!(
            "{} does not support Direct3D 12 at feature level {DIRECTML_MIN_FEATURE_LEVEL}, which DirectML needs",
            names.join(", ")
        ),
    )
}

/// Every GPU EP when the probe got no GPU facts (the graphics driver did not answer).
pub fn unknown() -> EpEligibility {
    let e = || {
        no(
            GPU_UNKNOWN,
            "the graphics driver did not answer, so GPU support could not be checked".into(),
        )
    };
    EpEligibility {
        cuda: e(),
        tensorrt: e(),
        tensorrt_rtx: e(),
        directml: e(),
    }
}

/// Eligibility of every GPU EP for this PC.
pub fn assess(gpus: &[GpuInfo], nvidia_driver: Option<&str>, cuda_driver_version: Option<&str>, windows: bool) -> EpEligibility {
    EpEligibility {
        cuda: nvidia_ep(
            "CUDA",
            CUDA_MIN_GEN,
            "Turing (GeForce RTX 20 / GTX 16 series)",
            gpus,
            nvidia_driver,
            cuda_driver_version,
        ),
        tensorrt: nvidia_ep(
            "TensorRT",
            CUDA_MIN_GEN,
            "Turing (GeForce RTX 20 / GTX 16 series)",
            gpus,
            nvidia_driver,
            cuda_driver_version,
        ),
        tensorrt_rtx: nvidia_ep(
            "TensorRT-RTX",
            TENSORRT_RTX_MIN_GEN,
            "Ampere (GeForce RTX 30 series)",
            gpus,
            nvidia_driver,
            cuda_driver_version,
        ),
        directml: directml(gpus, windows),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::services::diagnostics::gpu_mem::{VENDOR_AMD, VENDOR_INTEL, VENDOR_MICROSOFT};

    fn gpu(index: u32, vendor_id: u32, name: &str, cc: Option<&str>, arch: Option<&str>) -> GpuInfo {
        crate::services::sysinfo::test_support::gpu(index, vendor_id, name, 8192, cc, arch)
    }

    fn warp() -> GpuInfo {
        gpu(9, VENDOR_MICROSOFT, "Microsoft Basic Render Driver", None, None)
    }

    struct Row {
        name: &'static str,
        gpus: Vec<GpuInfo>,
        driver: Option<&'static str>,
        cuda: Option<&'static str>,
        /// (cuda, tensorrt, tensorrt_rtx, directml) codes.
        want: [&'static str; 4],
    }

    #[test]
    fn eligibility_table() {
        let rtx4090 = || gpu(0, VENDOR_NVIDIA, "NVIDIA GeForce RTX 4090", Some("8.9"), Some("ada"));
        let rtx2060 = || gpu(0, VENDOR_NVIDIA, "NVIDIA GeForce RTX 2060", Some("7.5"), Some("turing"));
        let gtx1080 = || gpu(0, VENDOR_NVIDIA, "NVIDIA GeForce GTX 1080", Some("6.1"), Some("pascal"));
        let rows = vec![
            Row {
                name: "4090, R616",
                gpus: vec![rtx4090(), warp()],
                driver: Some("616.56"),
                cuda: Some("13.2"),
                want: [OK, OK, OK, OK],
            },
            Row {
                name: "2060, R580: TRT-RTX needs Ampere",
                gpus: vec![rtx2060()],
                driver: Some("580.97"),
                cuda: None,
                want: [OK, OK, GPU_TOO_OLD, OK],
            },
            Row {
                name: "GTX 1080: CUDA 13 dropped Pascal",
                gpus: vec![gtx1080()],
                driver: Some("581.57"),
                cuda: Some("13.0"),
                want: [GPU_TOO_OLD, GPU_TOO_OLD, GPU_TOO_OLD, OK],
            },
            Row {
                name: "4090 on R566 (CUDA 12.7)",
                gpus: vec![rtx4090()],
                driver: Some("566.36"),
                cuda: Some("12.7"),
                want: [DRIVER_TOO_OLD, DRIVER_TOO_OLD, DRIVER_TOO_OLD, OK],
            },
            Row {
                name: "4090, only the DXGI driver number",
                gpus: vec![rtx4090()],
                driver: Some("572.16"),
                cuda: None,
                want: [DRIVER_TOO_OLD, DRIVER_TOO_OLD, DRIVER_TOO_OLD, OK],
            },
            Row {
                name: "4090, driver unknown",
                gpus: vec![rtx4090()],
                driver: None,
                cuda: None,
                want: [UNVERIFIED, UNVERIFIED, UNVERIFIED, OK],
            },
            Row {
                name: "old card + new card: the new one qualifies",
                gpus: vec![gtx1080(), gpu(1, VENDOR_NVIDIA, "NVIDIA GeForce RTX 3060", Some("8.6"), Some("ampere"))],
                driver: Some("616.56"),
                cuda: Some("13.2"),
                want: [OK, OK, OK, OK],
            },
            Row {
                name: "generation from the id table only",
                gpus: vec![gpu(0, VENDOR_NVIDIA, "NVIDIA GeForce RTX 3060", None, Some("ampere"))],
                driver: Some("616.56"),
                cuda: None,
                want: [OK, OK, OK, OK],
            },
            Row {
                name: "unknown NVIDIA generation",
                gpus: vec![gpu(0, VENDOR_NVIDIA, "NVIDIA Future GPU", None, None)],
                driver: Some("700.10"),
                cuda: Some("14.0"),
                want: [UNVERIFIED, UNVERIFIED, UNVERIFIED, OK],
            },
            Row {
                name: "compute capability newer than the table",
                gpus: vec![gpu(0, VENDOR_NVIDIA, "NVIDIA Future GPU", Some("14.0"), None)],
                driver: Some("700.10"),
                cuda: Some("14.0"),
                want: [OK, OK, OK, OK],
            },
            Row {
                name: "AMD only",
                gpus: vec![gpu(0, VENDOR_AMD, "AMD Radeon RX 7900 XTX", None, Some("rdna3"))],
                driver: None,
                cuda: None,
                want: [NO_NVIDIA_GPU, NO_NVIDIA_GPU, NO_NVIDIA_GPU, OK],
            },
            Row {
                name: "Intel iGPU + NVIDIA laptop",
                gpus: vec![
                    gpu(0, VENDOR_INTEL, "Intel(R) UHD Graphics", None, Some("xe-lp")),
                    gpu(1, VENDOR_NVIDIA, "NVIDIA GeForce RTX 4060 Laptop GPU", Some("8.9"), Some("ada")),
                ],
                driver: Some("616.56"),
                cuda: Some("13.2"),
                want: [OK, OK, OK, OK],
            },
            Row {
                name: "software renderer only",
                gpus: vec![warp()],
                driver: None,
                cuda: None,
                want: [NO_GPU, NO_GPU, NO_GPU, SOFTWARE_ONLY],
            },
            Row {
                name: "nothing at all",
                gpus: vec![],
                driver: None,
                cuda: None,
                want: [NO_GPU, NO_GPU, NO_GPU, NO_GPU],
            },
        ];
        for r in rows {
            let e = assess(&r.gpus, r.driver, r.cuda, true);
            let got = [
                e.cuda.code.as_str(),
                e.tensorrt.code.as_str(),
                e.tensorrt_rtx.code.as_str(),
                e.directml.code.as_str(),
            ];
            assert_eq!(got, r.want, "{}", r.name);
            for el in [&e.cuda, &e.tensorrt, &e.tensorrt_rtx, &e.directml] {
                // Ineligible always explains itself; a clean OK has no reason.
                assert_eq!(el.eligible, el.code == OK || el.code == UNVERIFIED, "{}: {el:?}", r.name);
                assert_eq!(el.reason.is_none(), el.code == OK, "{}: {el:?}", r.name);
            }
        }
    }

    #[test]
    fn reasons_name_the_gpu_and_the_floor() {
        let e = assess(
            &[gpu(0, VENDOR_NVIDIA, "NVIDIA GeForce GTX 1080", Some("6.1"), Some("pascal"))],
            Some("581.57"),
            None,
            true,
        );
        let reason = e.cuda.reason.unwrap();
        assert!(
            reason.contains("GTX 1080") && reason.contains("Pascal") && reason.contains("6.1") && reason.contains("Turing"),
            "{reason}"
        );
        let e = assess(
            &[gpu(0, VENDOR_NVIDIA, "NVIDIA GeForce RTX 4090", Some("8.9"), Some("ada"))],
            Some("566.36"),
            Some("12.7"),
            true,
        );
        let reason = e.cuda.reason.unwrap();
        assert!(reason.contains("566.36") && reason.contains("12.7") && reason.contains("580"), "{reason}");
    }

    #[test]
    fn directml_needs_windows_and_d3d12() {
        let mut g = gpu(0, VENDOR_AMD, "AMD Radeon HD 6450", None, None);
        g.d3d12 = Some(false);
        g.d3d12_feature_level = None;
        assert_eq!(assess(&[g.clone()], None, None, true).directml.code, NO_D3D12);
        g.d3d12 = None;
        assert_eq!(assess(&[g.clone()], None, None, true).directml.code, UNVERIFIED);
        assert_eq!(assess(&[g], None, None, false).directml.code, UNSUPPORTED_OS);
    }

    #[test]
    fn cuda_floor_per_gpu() {
        let rtx3070 = gpu(0, VENDOR_NVIDIA, "NVIDIA GeForce RTX 3070", Some("8.6"), Some("ampere"));
        let gtx1080ti = gpu(1, VENDOR_NVIDIA, "NVIDIA GeForce GTX 1080 Ti", Some("6.1"), Some("pascal"));
        let by_table = gpu(2, VENDOR_NVIDIA, "NVIDIA GeForce GTX 1660", None, Some("turing"));
        let future = gpu(3, VENDOR_NVIDIA, "NVIDIA Future GPU", None, None);
        let amd = gpu(4, VENDOR_AMD, "AMD Radeon RX 7900 XTX", None, Some("rdna3"));
        assert_eq!(meets_cuda_floor(&rtx3070), Some(true));
        assert_eq!(meets_cuda_floor(&gtx1080ti), Some(false));
        assert_eq!(meets_cuda_floor(&by_table), Some(true));
        assert_eq!(meets_cuda_floor(&future), None);
        assert_eq!(meets_cuda_floor(&amd), Some(false));
        assert_eq!(meets_cuda_floor(&warp()), Some(false));
    }

    #[test]
    fn an_unanswered_probe_is_not_no_gpu() {
        let e = unknown();
        for el in [&e.cuda, &e.tensorrt, &e.tensorrt_rtx, &e.directml] {
            assert!(!el.eligible);
            assert_eq!(el.code, GPU_UNKNOWN);
            assert!(el
                .reason
                .as_deref()
                .is_some_and(|r| r.contains("did not answer")));
        }
    }

    #[test]
    fn assessment_is_deterministic() {
        let gpus = vec![gpu(0, VENDOR_NVIDIA, "NVIDIA GeForce RTX 4090", Some("8.9"), Some("ada")), warp()];
        let a = assess(&gpus, Some("616.56"), Some("13.2"), true);
        let b = assess(&gpus, Some("616.56"), Some("13.2"), true);
        assert_eq!(a, b);
    }
}
