//! DXGI adapters (reused from `diagnostics::gpu_mem`: names, vendor / device ids, LUIDs, dedicated
//! VRAM, driver versions, per-process VRAM budget) plus a Direct3D 12 capability test per adapter,
//! which is what the DirectML execution provider needs (a D3D12 device at feature level 11_0+).
//!
//! The D3D12 test calls `D3D12CreateDevice` with a null output (capability test only: no device
//! is created). `d3d12.dll` is loaded from System32 at run time, so Sigil gains no load-time import,
//! and stays loaded afterwards (DirectML sessions load it anyway).

use std::collections::BTreeMap;

use crate::services::diagnostics::gpu_mem::{self, GpuAdapter};

/// DirectML's minimum Direct3D feature level (`D3D_FEATURE_LEVEL_11_0`).
pub const DIRECTML_MIN_FEATURE_LEVEL: &str = "11_0";

/// Feature levels tested, highest first (`D3D_FEATURE_LEVEL` value, label).
pub(crate) const FEATURE_LEVELS: [(i32, &str); 5] = [(0xc200, "12_2"), (0xc100, "12_1"), (0xc000, "12_0"), (0xb100, "11_1"), (0xb000, "11_0")];

/// Result of the Direct3D 12 test for one adapter.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum D3d12Support {
    /// Highest supported feature level (`"12_2"` … `"11_0"`).
    Level(&'static str),
    /// Tested: no feature level DirectML can use.
    Unsupported,
}

/// Every DXGI adapter (`EnumAdapters1` order = DirectML device id).
pub fn adapters() -> Vec<GpuAdapter> {
    gpu_mem::adapters()
}

/// Direct3D 12 support per hardware adapter, keyed by LUID (`high:low` hex, as in
/// [`GpuAdapter::luid`]). Software adapters are skipped. An empty map means the test could not
/// run (no `d3d12.dll`), which callers treat as "not verified".
pub fn d3d12_support() -> BTreeMap<String, D3d12Support> {
    imp::d3d12_support()
}

#[cfg(windows)]
mod imp {
    use super::{D3d12Support, FEATURE_LEVELS};
    use crate::services::diagnostics::gpu_mem::is_software_adapter;
    use libloading::os::windows::{Library, LOAD_LIBRARY_SEARCH_SYSTEM32};
    use std::collections::BTreeMap;
    use std::ffi::c_void;
    use windows::core::{Interface, GUID};
    use windows::Win32::Graphics::Dxgi::{CreateDXGIFactory1, IDXGIFactory1, DXGI_ADAPTER_FLAG_SOFTWARE};

    /// `IID_ID3D12Device`.
    const IID_ID3D12_DEVICE: GUID = GUID::from_u128(0x189819f1_1db6_4b57_be54_1821339b85f7);

    type D3d12CreateDevice = unsafe extern "system" fn(*mut c_void, i32, *const GUID, *mut *mut c_void) -> i32;

    pub fn d3d12_support() -> BTreeMap<String, D3d12Support> {
        let mut out = BTreeMap::new();
        // SAFETY: System32-only load of the OS Direct3D 12 runtime.
        let Ok(lib) = (unsafe { Library::load_with_flags("d3d12.dll", LOAD_LIBRARY_SEARCH_SYSTEM32) }) else {
            return out;
        };
        // SAFETY: documented export and signature of d3d12.dll.
        let Ok(create) = (unsafe { lib.get::<D3d12CreateDevice>(b"D3D12CreateDevice\0") }).map(|s| *s) else {
            return out;
        };
        // Keep d3d12.dll loaded for the process lifetime: the runtime is not designed to be
        // unloaded after use, and DirectML sessions need it anyway.
        let _ = lib.into_raw();

        // SAFETY: plain factory creation; no COM apartment is required for DXGI.
        let Ok(factory) = (unsafe { CreateDXGIFactory1::<IDXGIFactory1>() }) else {
            return out;
        };
        for index in 0u32..64 {
            // SAFETY: enumeration stops at DXGI_ERROR_NOT_FOUND (an Err here).
            let Ok(adapter) = (unsafe { factory.EnumAdapters1(index) }) else {
                break;
            };
            // SAFETY: valid adapter interface.
            let Ok(desc) = (unsafe { adapter.GetDesc1() }) else {
                continue;
            };
            // The same rule as `gpu_mem::adapters`: the flag alone misses the Basic Render Driver's unflagged twins.
            let name_len = desc.Description.iter().position(|&c| c == 0).unwrap_or(desc.Description.len());
            let name = String::from_utf16_lossy(&desc.Description[..name_len]);
            let flagged = (desc.Flags & DXGI_ADAPTER_FLAG_SOFTWARE.0 as u32) != 0;
            if is_software_adapter(flagged, desc.VendorId, desc.DeviceId, name.trim()) {
                continue;
            }
            let luid = format!("{:08x}:{:08x}", desc.AdapterLuid.HighPart as u32, desc.AdapterLuid.LowPart);
            let support = FEATURE_LEVELS
                .iter()
                .find(|(level, _)| {
                    // SAFETY: a live IDXGIAdapter1 pointer, a valid IID and a null device pointer
                    // (capability test: returns S_FALSE on success and creates nothing).
                    let hr = unsafe { create(adapter.as_raw(), *level, &IID_ID3D12_DEVICE, std::ptr::null_mut()) };
                    hr >= 0
                })
                .map(|(_, label)| D3d12Support::Level(label))
                .unwrap_or(D3d12Support::Unsupported);
            out.insert(luid, support);
        }
        out
    }
}

#[cfg(not(windows))]
mod imp {
    use super::D3d12Support;
    use std::collections::BTreeMap;

    pub fn d3d12_support() -> BTreeMap<String, D3d12Support> {
        BTreeMap::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn feature_levels_are_ordered_and_include_the_directml_floor() {
        assert!(FEATURE_LEVELS.windows(2).all(|w| w[0].0 > w[1].0));
        assert_eq!(FEATURE_LEVELS.last().map(|l| l.1), Some(DIRECTML_MIN_FEATURE_LEVEL));
    }

    #[test]
    fn d3d12_test_covers_every_hardware_adapter() {
        let list = adapters();
        let support = d3d12_support();
        if support.is_empty() {
            return; // no d3d12.dll (not Windows 10+): nothing to check
        }
        for a in list.iter().filter(|a| !a.software) {
            assert!(support.contains_key(&a.luid), "no D3D12 result for {}", a.name);
        }
        // Any Windows 10+ machine with a driver-backed GPU supports at least feature level 11_0.
        for a in list
            .iter()
            .filter(|a| !a.software && a.dedicated_vram_mb >= 1024)
        {
            assert!(matches!(support.get(&a.luid), Some(D3d12Support::Level(_))), "{} lacks D3D12", a.name);
        }
    }
}
