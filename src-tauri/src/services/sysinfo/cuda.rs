//! NVIDIA driver facts from `nvcuda.dll`, loaded dynamically from System32 only.
//!
//! [`driver_cuda_version`] calls `cuDriverGetVersion`, which needs neither `cuInit` nor a context
//! (no VRAM, no driver threads); the DLL is unloaded again right after. Also holds the pure
//! version helpers and the CUDA 13 floors the eligibility rules use.

/// NVIDIA driver branch CUDA 13 needs (R580).
pub const MIN_DRIVER_FOR_CUDA13: (u32, u32) = (580, 0);
/// CUDA driver API version Sigil's ONNX Runtime (1.28, CUDA 13 build) needs.
pub const MIN_CUDA_DRIVER_API: (u32, u32) = (13, 0);

/// NVIDIA driver number `"616.56"` → `(616, 56)`. Also accepts Linux-style `"580.95.05"` (first
/// two components). `None` for anything else.
pub fn parse_driver_version(s: &str) -> Option<(u32, u32)> {
    let mut parts = s.trim().split('.');
    let major = parts.next()?.trim().parse().ok()?;
    let minor = parts
        .next()
        .map(|p| p.trim().parse().ok())
        .unwrap_or(Some(0))?;
    Some((major, minor))
}

/// CUDA version `"13.2"` → `(13, 2)`.
pub fn parse_cuda_version(s: &str) -> Option<(u32, u32)> {
    let (a, b) = s.trim().split_once('.')?;
    Some((a.parse().ok()?, b.parse().ok()?))
}

/// `cuDriverGetVersion` / `nvmlSystemGetCudaDriverVersion` integer (`13020`) → `"13.2"`.
pub fn cuda_version_string(v: i32) -> Option<String> {
    (v > 0).then(|| format!("{}.{}", v / 1000, (v % 1000) / 10))
}

/// Highest CUDA version the installed NVIDIA driver supports (e.g. `"13.2"`), or `None` without an
/// NVIDIA driver. Loads System32's `nvcuda.dll` for the call only.
pub fn driver_cuda_version() -> Option<String> {
    imp::driver_cuda_version().and_then(cuda_version_string)
}

#[cfg(windows)]
mod imp {
    use libloading::os::windows::{Library, LOAD_LIBRARY_SEARCH_SYSTEM32};

    type CuDriverGetVersion = unsafe extern "system" fn(*mut i32) -> i32;

    pub fn driver_cuda_version() -> Option<i32> {
        // SAFETY: System32-only load of the NVIDIA driver's CUDA DLL (no planting from the app or
        // working directory). Its DllMain does not initialize CUDA; nothing here calls cuInit.
        let lib = unsafe { Library::load_with_flags("nvcuda.dll", LOAD_LIBRARY_SEARCH_SYSTEM32) }.ok()?;
        // SAFETY: documented signature `CUresult cuDriverGetVersion(int *driverVersion)`.
        let f: CuDriverGetVersion = *unsafe { lib.get::<CuDriverGetVersion>(b"cuDriverGetVersion\0") }.ok()?;
        let mut v = 0i32;
        // SAFETY: valid out pointer; the call needs no prior cuInit.
        let rc = unsafe { f(&mut v) };
        drop(lib);
        (rc == 0 && v > 0).then_some(v)
    }
}

#[cfg(not(windows))]
mod imp {
    pub fn driver_cuda_version() -> Option<i32> {
        None
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parses_driver_and_cuda_versions() {
        assert_eq!(parse_driver_version("616.56"), Some((616, 56)));
        assert_eq!(parse_driver_version("580.97"), Some((580, 97)));
        assert_eq!(parse_driver_version("580.95.05"), Some((580, 95)));
        assert_eq!(parse_driver_version("566"), Some((566, 0)));
        assert_eq!(parse_driver_version(""), None);
        assert_eq!(parse_driver_version("abc"), None);
        assert!(parse_driver_version("579.99").unwrap() < MIN_DRIVER_FOR_CUDA13);
        assert!(parse_driver_version("580.88").unwrap() >= MIN_DRIVER_FOR_CUDA13);

        assert_eq!(parse_cuda_version("13.2"), Some((13, 2)));
        assert_eq!(parse_cuda_version("12.9"), Some((12, 9)));
        assert_eq!(parse_cuda_version("13"), None);
        assert!(parse_cuda_version("12.9").unwrap() < MIN_CUDA_DRIVER_API);

        assert_eq!(cuda_version_string(13020).as_deref(), Some("13.2"));
        assert_eq!(cuda_version_string(12090).as_deref(), Some("12.9"));
        assert_eq!(cuda_version_string(0), None);
    }

    #[test]
    fn driver_query_never_panics() {
        // No NVIDIA driver (CI): None. Dev rig: a version at least as new as CUDA 12.
        if let Some(v) = driver_cuda_version() {
            assert!(parse_cuda_version(&v).is_some(), "unexpected CUDA version {v}");
        }
    }
}
