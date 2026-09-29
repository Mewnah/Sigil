//! NVML (NVIDIA Management Library), loaded dynamically: driver version, CUDA driver version,
//! per-GPU compute capability, PCI ids and device-wide memory use (all processes, e.g. VRChat and
//! OBS too), which DXGI cannot report.
//!
//! - `nvml.dll` is loaded from System32 only (the driver installs it there), then from
//!   `%ProgramW6432%\NVIDIA Corporation\NVSMI` (older drivers), always by full path, never from the
//!   application or working directory. Nothing links it at load time.
//! - NVML never creates a CUDA context (no VRAM). One handle is shared for the process lifetime
//!   ([`shared`]); initialization runs without holding the lock, so a slow driver never blocks
//!   callers that only ask whether NVML is ready ([`loaded`]).
//! - NVML is thread-safe; device handles are re-fetched per call and never stored.

use std::sync::{Arc, Mutex, PoisonError};

/// Device-wide memory of one GPU (MiB), as NVML reports it (all processes plus driver reserve).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct NvmlMemory {
    pub total_mb: u64,
    pub used_mb: u64,
    pub free_mb: u64,
}

/// One GPU as NVML sees it (NVML index = PCI bus order).
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct NvmlDevice {
    pub index: u32,
    pub name: String,
    /// 16-bit PCI device id (NVML's combined id is `device << 16 | vendor`).
    pub pci_device_id: u32,
    pub pci_vendor_id: u32,
    /// `00000000:01:00.0`.
    pub pci_bus_id: Option<String>,
    pub compute_capability: Option<(u32, u32)>,
    pub memory: Option<NvmlMemory>,
}

/// System-level NVML facts.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct NvmlSystem {
    /// `616.56`.
    pub driver_version: Option<String>,
    /// `cuDriverGetVersion`-style integer (`13020`).
    pub cuda_driver_version: Option<i32>,
    pub nvml_version: Option<String>,
    pub devices: Vec<NvmlDevice>,
}

/// Map each DXGI adapter (by position in `adapters`) to an NVML device index, or `None`.
///
/// NVML has no LUID, so NVIDIA adapters are paired by PCI device id in enumeration order, then by
/// name, then by order among the remaining NVIDIA devices. Two identical cards may swap, which is
/// harmless: they share every reported property except live memory use.
pub fn match_devices(adapters: &[AdapterKey<'_>], devices: &[NvmlDevice]) -> Vec<Option<u32>> {
    use crate::services::diagnostics::gpu_mem::VENDOR_NVIDIA;

    fn take(devices: &[NvmlDevice], used: &mut [bool], pick: impl Fn(&NvmlDevice) -> bool) -> Option<u32> {
        let j = (0..devices.len()).find(|&j| !used[j] && pick(&devices[j]))?;
        used[j] = true;
        Some(devices[j].index)
    }

    let mut used = vec![false; devices.len()];
    let mut out = vec![None; adapters.len()];
    let nvidia: Vec<usize> = (0..adapters.len())
        .filter(|&i| adapters[i].vendor_id == VENDOR_NVIDIA && !adapters[i].software)
        .collect();
    for pass in 0..3 {
        for &i in &nvidia {
            if out[i].is_some() {
                continue;
            }
            let a = &adapters[i];
            out[i] = match pass {
                0 => take(devices, &mut used, |d| d.pci_device_id == a.device_id),
                1 => take(devices, &mut used, |d| !d.name.is_empty() && d.name.eq_ignore_ascii_case(a.name.trim())),
                _ => take(devices, &mut used, |_| true),
            };
        }
    }
    out
}

/// What [`match_devices`] needs from a DXGI adapter.
#[derive(Clone, Copy, Debug)]
pub struct AdapterKey<'a> {
    pub vendor_id: u32,
    pub device_id: u32,
    pub name: &'a str,
    pub software: bool,
}

enum Slot {
    Empty,
    Loading,
    Ready(Arc<imp::Nvml>),
    Failed(String),
}

static SLOT: Mutex<Slot> = Mutex::new(Slot::Empty);

fn slot() -> std::sync::MutexGuard<'static, Slot> {
    SLOT.lock().unwrap_or_else(PoisonError::into_inner)
}

/// The process-wide NVML handle, loading and initializing NVML on first use. A failure is
/// remembered; `retry_failed` tries again (e.g. after the user installed a driver).
pub fn shared(retry_failed: bool) -> Result<Arc<imp::Nvml>, String> {
    {
        let mut s = slot();
        match &*s {
            Slot::Ready(h) => return Ok(h.clone()),
            Slot::Loading => return Err("NVML is still starting".into()),
            Slot::Failed(e) if !retry_failed => return Err(e.clone()),
            _ => {}
        }
        *s = Slot::Loading;
    }
    // Initialize without the lock: a slow or hung driver never blocks `loaded()`.
    let result = imp::Nvml::load().map(Arc::new);
    let mut s = slot();
    match result {
        Ok(h) => {
            *s = Slot::Ready(h.clone());
            Ok(h)
        }
        Err(e) => {
            *s = Slot::Failed(e.clone());
            Err(e)
        }
    }
}

/// The handle if NVML is already initialized (never loads it).
pub fn loaded() -> Option<Arc<imp::Nvml>> {
    match &*slot() {
        Slot::Ready(h) => Some(h.clone()),
        _ => None,
    }
}

/// Device-wide memory of NVML device `index`, only if NVML is already initialized (the resource
/// snapshot must never be the thing that loads a GPU library).
pub fn memory_if_loaded(index: u32) -> Option<NvmlMemory> {
    loaded()?.memory(index)
}

#[cfg(windows)]
pub mod imp {
    use super::{NvmlDevice, NvmlMemory, NvmlSystem};
    use libloading::os::windows::{Library, LOAD_LIBRARY_SEARCH_SYSTEM32, LOAD_WITH_ALTERED_SEARCH_PATH};
    use std::ffi::{c_char, c_void, CStr};

    const NVML_SUCCESS: i32 = 0;
    const DRIVER_VERSION_BUF: usize = 80;
    const NAME_BUF: usize = 96;
    const MIB: u64 = 1024 * 1024;

    #[repr(C)]
    struct PciInfo {
        bus_id_legacy: [c_char; 16],
        domain: u32,
        bus: u32,
        device: u32,
        /// `device << 16 | vendor`.
        pci_device_id: u32,
        pci_sub_system_id: u32,
        bus_id: [c_char; 32],
    }

    #[repr(C)]
    struct Memory {
        total: u64,
        free: u64,
        used: u64,
    }

    type Device = *mut c_void;
    type FnVoid = unsafe extern "C" fn() -> i32;
    type FnStr = unsafe extern "C" fn(*mut c_char, u32) -> i32;
    type FnInt = unsafe extern "C" fn(*mut i32) -> i32;
    type FnCount = unsafe extern "C" fn(*mut u32) -> i32;
    type FnHandle = unsafe extern "C" fn(u32, *mut Device) -> i32;
    type FnDevStr = unsafe extern "C" fn(Device, *mut c_char, u32) -> i32;
    type FnPci = unsafe extern "C" fn(Device, *mut PciInfo) -> i32;
    type FnCc = unsafe extern "C" fn(Device, *mut i32, *mut i32) -> i32;
    type FnMem = unsafe extern "C" fn(Device, *mut Memory) -> i32;
    type FnErr = unsafe extern "C" fn(i32) -> *const c_char;

    /// A loaded and initialized NVML (shut down and unloaded on drop).
    pub struct Nvml {
        shutdown: FnVoid,
        error_string: Option<FnErr>,
        driver_version: Option<FnStr>,
        nvml_version: Option<FnStr>,
        cuda_version: Option<FnInt>,
        count: FnCount,
        handle: FnHandle,
        name: Option<FnDevStr>,
        pci: Option<FnPci>,
        cc: Option<FnCc>,
        memory: Option<FnMem>,
        // Last field: dropped after `Drop::drop` ran nvmlShutdown.
        _lib: Library,
    }

    fn load_library() -> Result<Library, String> {
        // SAFETY: System32-only search: the NVIDIA driver installs nvml.dll there.
        if let Ok(lib) = unsafe { Library::load_with_flags("nvml.dll", LOAD_LIBRARY_SEARCH_SYSTEM32) } {
            return Ok(lib);
        }
        let base = std::env::var_os("ProgramW6432")
            .or_else(|| std::env::var_os("ProgramFiles"))
            .map(std::path::PathBuf::from)
            .unwrap_or_else(|| std::path::PathBuf::from(r"C:\Program Files"));
        let path = base
            .join("NVIDIA Corporation")
            .join("NVSMI")
            .join("nvml.dll");
        if !path.is_file() {
            return Err("nvml.dll was not found (no NVIDIA driver installed)".into());
        }
        // SAFETY: absolute path under Program Files (admin-writable only); altered search path so
        // its own imports resolve from its folder.
        unsafe { Library::load_with_flags(&path, LOAD_WITH_ALTERED_SEARCH_PATH) }.map_err(|e| format!("cannot load {}: {e}", path.display()))
    }

    /// Copy a symbol out of `lib` (the library outlives every copy: both live in `Nvml`).
    fn sym<T: Copy>(lib: &Library, name: &[u8]) -> Option<T> {
        // SAFETY: every call site names a documented NVML export with its exact C signature.
        unsafe { lib.get::<T>(name) }.ok().map(|s| *s)
    }

    fn c_buf_to_string(buf: &[c_char]) -> Option<String> {
        let bytes: Vec<u8> = buf
            .iter()
            .take_while(|&&c| c != 0)
            .map(|&c| c as u8)
            .collect();
        let s = String::from_utf8_lossy(&bytes).trim().to_string();
        (!s.is_empty()).then_some(s)
    }

    impl Nvml {
        pub fn load() -> Result<Nvml, String> {
            let lib = load_library()?;
            let init: FnVoid = sym(&lib, b"nvmlInit_v2\0").ok_or("nvml.dll has no nvmlInit_v2")?;
            let shutdown: FnVoid = sym(&lib, b"nvmlShutdown\0").ok_or("nvml.dll has no nvmlShutdown")?;
            let count: FnCount = sym(&lib, b"nvmlDeviceGetCount_v2\0").ok_or("nvml.dll has no nvmlDeviceGetCount_v2")?;
            let handle: FnHandle = sym(&lib, b"nvmlDeviceGetHandleByIndex_v2\0").ok_or("nvml.dll has no nvmlDeviceGetHandleByIndex_v2")?;
            let error_string: Option<FnErr> = sym(&lib, b"nvmlErrorString\0");
            // SAFETY: `nvmlReturn_t nvmlInit_v2(void)`; reference counted by NVML.
            let rc = unsafe { init() };
            if rc != NVML_SUCCESS {
                let why = error_string
                    // SAFETY: nvmlErrorString returns a static NUL-terminated string.
                    .map(|f| {
                        unsafe { CStr::from_ptr(f(rc)) }
                            .to_string_lossy()
                            .into_owned()
                    })
                    .unwrap_or_default();
                return Err(format!("nvmlInit failed (code {rc}{}{why})", if why.is_empty() { "" } else { ": " }));
            }
            Ok(Nvml {
                shutdown,
                error_string,
                driver_version: sym(&lib, b"nvmlSystemGetDriverVersion\0"),
                nvml_version: sym(&lib, b"nvmlSystemGetNVMLVersion\0"),
                cuda_version: sym(&lib, b"nvmlSystemGetCudaDriverVersion_v2\0").or_else(|| sym(&lib, b"nvmlSystemGetCudaDriverVersion\0")),
                count,
                handle,
                name: sym(&lib, b"nvmlDeviceGetName\0"),
                pci: sym(&lib, b"nvmlDeviceGetPciInfo_v3\0").or_else(|| sym(&lib, b"nvmlDeviceGetPciInfo_v2\0")),
                cc: sym(&lib, b"nvmlDeviceGetCudaComputeCapability\0"),
                memory: sym(&lib, b"nvmlDeviceGetMemoryInfo\0"),
                _lib: lib,
            })
        }

        fn string(&self, f: Option<FnStr>) -> Option<String> {
            let f = f?;
            let mut buf = [0 as c_char; DRIVER_VERSION_BUF];
            // SAFETY: buffer and its length match.
            (unsafe { f(buf.as_mut_ptr(), buf.len() as u32) } == NVML_SUCCESS).then(|| c_buf_to_string(&buf))?
        }

        fn device(&self, index: u32) -> Option<Device> {
            let mut d: Device = std::ptr::null_mut();
            // SAFETY: valid out pointer; an out-of-range index returns an error.
            (unsafe { (self.handle)(index, &mut d) } == NVML_SUCCESS && !d.is_null()).then_some(d)
        }

        /// Device-wide memory of device `index` (all processes).
        pub fn memory(&self, index: u32) -> Option<NvmlMemory> {
            let f = self.memory?;
            let d = self.device(index)?;
            let mut m = Memory { total: 0, free: 0, used: 0 };
            // SAFETY: valid device handle and out pointer (nvmlMemory_t v1 layout).
            let ok = unsafe { f(d, &mut m) } == NVML_SUCCESS;
            (ok && m.total > 0).then_some(NvmlMemory {
                total_mb: m.total / MIB,
                used_mb: m.used / MIB,
                free_mb: m.free / MIB,
            })
        }

        fn describe(&self, index: u32) -> Option<NvmlDevice> {
            let d = self.device(index)?;
            let name = self.name.and_then(|f| {
                let mut buf = [0 as c_char; NAME_BUF];
                // SAFETY: valid device handle; buffer and its length match.
                (unsafe { f(d, buf.as_mut_ptr(), buf.len() as u32) } == NVML_SUCCESS).then(|| c_buf_to_string(&buf))?
            });
            let pci = self.pci.and_then(|f| {
                // SAFETY: plain-old-data struct; zeroed is a valid initial state.
                let mut p: PciInfo = unsafe { std::mem::zeroed() };
                // SAFETY: valid device handle and out pointer (nvmlPciInfo_t v3 layout).
                (unsafe { f(d, &mut p) } == NVML_SUCCESS).then_some(p)
            });
            let cc = self.cc.and_then(|f| {
                let (mut major, mut minor) = (0i32, 0i32);
                // SAFETY: valid device handle and out pointers.
                (unsafe { f(d, &mut major, &mut minor) } == NVML_SUCCESS && major > 0).then_some((major as u32, minor.max(0) as u32))
            });
            Some(NvmlDevice {
                index,
                name: name.unwrap_or_default(),
                pci_device_id: pci.as_ref().map(|p| p.pci_device_id >> 16).unwrap_or(0),
                pci_vendor_id: pci.as_ref().map(|p| p.pci_device_id & 0xFFFF).unwrap_or(0),
                pci_bus_id: pci.as_ref().and_then(|p| c_buf_to_string(&p.bus_id)),
                compute_capability: cc,
                memory: self.memory(index),
            })
        }

        /// Driver, CUDA and NVML versions plus every device.
        pub fn system(&self) -> NvmlSystem {
            let mut n = 0u32;
            // SAFETY: valid out pointer.
            let count = if unsafe { (self.count)(&mut n) } == NVML_SUCCESS { n.min(64) } else { 0 };
            let cuda_driver_version = self.cuda_version.and_then(|f| {
                let mut v = 0i32;
                // SAFETY: valid out pointer.
                (unsafe { f(&mut v) } == NVML_SUCCESS && v > 0).then_some(v)
            });
            NvmlSystem {
                driver_version: self.string(self.driver_version),
                cuda_driver_version,
                nvml_version: self.string(self.nvml_version),
                devices: (0..count).filter_map(|i| self.describe(i)).collect(),
            }
        }

        #[allow(dead_code)] // Diagnostics helper for error codes returned by optional calls.
        pub fn error_text(&self, rc: i32) -> String {
            self.error_string
                // SAFETY: nvmlErrorString returns a static NUL-terminated string.
                .map(|f| {
                    unsafe { CStr::from_ptr(f(rc)) }
                        .to_string_lossy()
                        .into_owned()
                })
                .unwrap_or_else(|| format!("NVML error {rc}"))
        }
    }

    impl Drop for Nvml {
        fn drop(&mut self) {
            // SAFETY: balances the successful nvmlInit_v2 in `load`.
            unsafe { (self.shutdown)() };
        }
    }
}

#[cfg(not(windows))]
pub mod imp {
    use super::{NvmlMemory, NvmlSystem};

    pub struct Nvml;

    impl Nvml {
        pub fn load() -> Result<Nvml, String> {
            Err("NVML is only probed on Windows".into())
        }
        pub fn memory(&self, _index: u32) -> Option<NvmlMemory> {
            None
        }
        pub fn system(&self) -> NvmlSystem {
            NvmlSystem::default()
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::services::diagnostics::gpu_mem::{VENDOR_INTEL, VENDOR_MICROSOFT, VENDOR_NVIDIA};

    fn dev(index: u32, device: u32, name: &str) -> NvmlDevice {
        NvmlDevice {
            index,
            name: name.into(),
            pci_device_id: device,
            pci_vendor_id: VENDOR_NVIDIA,
            pci_bus_id: None,
            compute_capability: Some((8, 9)),
            memory: None,
        }
    }

    fn key(vendor_id: u32, device_id: u32, name: &str, software: bool) -> AdapterKey<'_> {
        AdapterKey {
            vendor_id,
            device_id,
            name,
            software,
        }
    }

    #[test]
    fn pairs_dxgi_adapters_with_nvml_devices() {
        // DXGI order: iGPU, 3060, 4090, WARP. NVML (PCI order): 4090, 3060.
        let adapters = [
            key(VENDOR_INTEL, 0xA780, "Intel(R) UHD Graphics 770", false),
            key(VENDOR_NVIDIA, 0x2503, "NVIDIA GeForce RTX 3060", false),
            key(VENDOR_NVIDIA, 0x2684, "NVIDIA GeForce RTX 4090", false),
            key(VENDOR_MICROSOFT, 0x8C, "Microsoft Basic Render Driver", true),
        ];
        let devices = [dev(0, 0x2684, "NVIDIA GeForce RTX 4090"), dev(1, 0x2503, "NVIDIA GeForce RTX 3060")];
        assert_eq!(match_devices(&adapters, &devices), vec![None, Some(1), Some(0), None]);

        // Two identical cards pair in order; an unknown PCI id falls back to the name, then order.
        let adapters = [
            key(VENDOR_NVIDIA, 0x2684, "NVIDIA GeForce RTX 4090", false),
            key(VENDOR_NVIDIA, 0x2684, "NVIDIA GeForce RTX 4090", false),
        ];
        let devices = [dev(0, 0x2684, "NVIDIA GeForce RTX 4090"), dev(1, 0x2684, "NVIDIA GeForce RTX 4090")];
        assert_eq!(match_devices(&adapters, &devices), vec![Some(0), Some(1)]);

        let adapters = [key(VENDOR_NVIDIA, 0x9999, "NVIDIA Future GPU", false)];
        let devices = [dev(3, 0, "NVIDIA Future GPU")];
        assert_eq!(match_devices(&adapters, &devices), vec![Some(3)]);

        let devices = [dev(5, 0, "")];
        assert_eq!(match_devices(&adapters, &devices), vec![Some(5)]);
        assert_eq!(match_devices(&adapters, &[]), vec![None]);
    }

    #[test]
    fn loading_is_best_effort() {
        // CI has no NVIDIA driver (Err); the dev rig reports the driver and its GPUs. Never panics,
        // and a second call reuses the first handle.
        match shared(false) {
            Ok(h) => {
                let sys = h.system();
                assert!(sys.driver_version.is_some(), "NVML loaded but reported no driver version");
                for d in &sys.devices {
                    assert!(d.memory.map(|m| m.total_mb > 0).unwrap_or(true));
                }
                assert!(loaded().is_some());
                assert!(Arc::ptr_eq(&h, &shared(false).unwrap()));
            }
            Err(e) => assert!(!e.is_empty()),
        }
    }
}
