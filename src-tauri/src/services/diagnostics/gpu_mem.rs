//! GPU adapters and per-process VRAM through DXGI (no vendor DLL is loaded).
//!
//! - [`adapters`]: every DXGI adapter in `EnumAdapters1` order, which is also the device index the
//!   DirectML execution provider uses. Includes the user-mode driver version and, for NVIDIA, the
//!   familiar driver number (e.g. `581.57`), derived without loading `nvcuda.dll` or NVML.
//! - [`process_vram_usage`]: this process's local (dedicated) VRAM use and the OS budget, from
//!   `IDXGIAdapter3::QueryVideoMemoryInfo` — the numbers the VRAM ledger and benchmark use.
//!
//! Everything is best effort: on failure the lists are empty / `None`, never an error.

use serde::{Deserialize, Serialize};

pub const VENDOR_NVIDIA: u32 = 0x10DE;
pub const VENDOR_AMD: u32 = 0x1002;
pub const VENDOR_INTEL: u32 = 0x8086;
pub const VENDOR_MICROSOFT: u32 = 0x1414;

/// This process's VRAM use on one adapter (local segment group), in MiB.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct VramUsage {
    /// What the OS currently lets this process use before it starts demoting allocations.
    pub budget_mb: u64,
    /// This process's current usage.
    pub process_usage_mb: u64,
    pub available_for_reservation_mb: u64,
}

/// One DXGI adapter.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct GpuAdapter {
    /// `EnumAdapters1` index = DirectML `device_id`.
    pub index: u32,
    pub name: String,
    /// `nvidia` / `amd` / `intel` / `microsoft` / `other`.
    pub vendor: String,
    pub vendor_id: u32,
    pub device_id: u32,
    /// Adapter LUID as `high:low` hex (stable for the boot session; matches CUDA's device LUID).
    pub luid: String,
    pub dedicated_vram_mb: u64,
    pub shared_memory_mb: u64,
    /// Microsoft Basic Render Driver / WARP.
    pub software: bool,
    /// User-mode driver version, e.g. `32.0.15.8157`.
    pub driver_version: Option<String>,
    /// NVIDIA's driver number, e.g. `581.57` (NVIDIA adapters only).
    pub nvidia_driver: Option<String>,
    /// This process's VRAM use on the adapter at probe time.
    pub vram: Option<VramUsage>,
}

pub fn vendor_name(vendor_id: u32) -> &'static str {
    match vendor_id {
        VENDOR_NVIDIA => "nvidia",
        VENDOR_AMD => "amd",
        VENDOR_INTEL => "intel",
        VENDOR_MICROSOFT => "microsoft",
        _ => "other",
    }
}

/// Split the 64-bit UMD version from `CheckInterfaceSupport` into `a.b.c.d`.
pub fn umd_version_parts(umd: i64) -> [u16; 4] {
    let v = umd as u64;
    [(v >> 48) as u16, (v >> 32) as u16, (v >> 16) as u16, v as u16]
}

/// NVIDIA's public driver number from the UMD version: the last five digits of `c.d`
/// (`32.0.15.8157` → `581.57`).
pub fn nvidia_driver_from_umd(umd: i64) -> Option<String> {
    let [_, _, c, d] = umd_version_parts(umd);
    if d > 9999 {
        return None;
    }
    let n = u32::from(c % 10) * 10_000 + u32::from(d);
    Some(format!("{}.{:02}", n / 100, n % 100))
}

/// The DXGI indices of NVIDIA adapters, in enumeration order.
pub fn nvidia_adapter_indices(list: &[GpuAdapter]) -> Vec<u32> {
    list.iter().filter(|a| a.vendor_id == VENDOR_NVIDIA && !a.software).map(|a| a.index).collect()
}

/// Every DXGI adapter with its driver version and this process's VRAM use.
pub fn adapters() -> Vec<GpuAdapter> {
    imp::adapters()
}

/// This process's local VRAM use on DXGI adapter `index`.
#[allow(dead_code)] // For the voice-changer VRAM metric / P3 VRAM ledger; adapters() covers the UI.
pub fn process_vram_usage(index: u32) -> Option<VramUsage> {
    imp::process_vram_usage(index)
}

#[cfg(windows)]
mod imp {
    use super::{nvidia_driver_from_umd, umd_version_parts, vendor_name, GpuAdapter, VramUsage, VENDOR_NVIDIA};
    use windows::core::Interface;
    use windows::Win32::Graphics::Dxgi::{
        CreateDXGIFactory1, IDXGIAdapter1, IDXGIAdapter3, IDXGIDevice, IDXGIFactory1, DXGI_ADAPTER_FLAG_SOFTWARE,
        DXGI_MEMORY_SEGMENT_GROUP_LOCAL, DXGI_QUERY_VIDEO_MEMORY_INFO,
    };

    const MIB: u64 = 1024 * 1024;

    fn factory() -> Option<IDXGIFactory1> {
        // SAFETY: plain factory creation; no COM apartment is required for DXGI.
        unsafe { CreateDXGIFactory1::<IDXGIFactory1>() }.ok()
    }

    fn vram_of(adapter: &IDXGIAdapter1) -> Option<VramUsage> {
        let a3: IDXGIAdapter3 = adapter.cast().ok()?;
        let mut info = DXGI_QUERY_VIDEO_MEMORY_INFO::default();
        // SAFETY: node 0 exists on every adapter; `info` is a valid out pointer.
        unsafe { a3.QueryVideoMemoryInfo(0, DXGI_MEMORY_SEGMENT_GROUP_LOCAL, &mut info) }.ok()?;
        Some(VramUsage {
            budget_mb: info.Budget / MIB,
            process_usage_mb: info.CurrentUsage / MIB,
            available_for_reservation_mb: info.AvailableForReservation / MIB,
        })
    }

    pub fn adapters() -> Vec<GpuAdapter> {
        let Some(factory) = factory() else {
            return Vec::new();
        };
        let mut out = Vec::new();
        for index in 0u32..64 {
            // SAFETY: enumeration stops at DXGI_ERROR_NOT_FOUND (an Err here).
            let Ok(adapter) = (unsafe { factory.EnumAdapters1(index) }) else {
                break;
            };
            // SAFETY: valid adapter interface.
            let Ok(desc) = (unsafe { adapter.GetDesc1() }) else {
                continue;
            };
            let name_len = desc.Description.iter().position(|&c| c == 0).unwrap_or(desc.Description.len());
            let name = String::from_utf16_lossy(&desc.Description[..name_len]).trim().to_string();
            // SAFETY: IDXGIDevice's IID is the documented key for the UMD driver version.
            let umd = unsafe { adapter.CheckInterfaceSupport(&IDXGIDevice::IID) }.ok();
            let software = (desc.Flags & DXGI_ADAPTER_FLAG_SOFTWARE.0 as u32) != 0;
            out.push(GpuAdapter {
                index,
                name,
                vendor: vendor_name(desc.VendorId).to_string(),
                vendor_id: desc.VendorId,
                device_id: desc.DeviceId,
                luid: format!("{:08x}:{:08x}", desc.AdapterLuid.HighPart as u32, desc.AdapterLuid.LowPart),
                dedicated_vram_mb: desc.DedicatedVideoMemory as u64 / MIB,
                shared_memory_mb: desc.SharedSystemMemory as u64 / MIB,
                software,
                driver_version: umd.map(|v| {
                    let [a, b, c, d] = umd_version_parts(v);
                    format!("{a}.{b}.{c}.{d}")
                }),
                nvidia_driver: if desc.VendorId == VENDOR_NVIDIA { umd.and_then(nvidia_driver_from_umd) } else { None },
                vram: vram_of(&adapter),
            });
        }
        out
    }

    #[allow(dead_code)]
    pub fn process_vram_usage(index: u32) -> Option<VramUsage> {
        let factory = factory()?;
        // SAFETY: an out-of-range index returns DXGI_ERROR_NOT_FOUND.
        let adapter = unsafe { factory.EnumAdapters1(index) }.ok()?;
        vram_of(&adapter)
    }
}

#[cfg(not(windows))]
mod imp {
    pub fn adapters() -> Vec<super::GpuAdapter> {
        Vec::new()
    }
    #[allow(dead_code)]
    pub fn process_vram_usage(_index: u32) -> Option<super::VramUsage> {
        None
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn umd(a: u16, b: u16, c: u16, d: u16) -> i64 {
        ((u64::from(a) << 48) | (u64::from(b) << 32) | (u64::from(c) << 16) | u64::from(d)) as i64
    }

    #[test]
    fn nvidia_driver_number_from_umd_version() {
        assert_eq!(nvidia_driver_from_umd(umd(32, 0, 15, 8157)).as_deref(), Some("581.57"));
        assert_eq!(nvidia_driver_from_umd(umd(31, 0, 15, 5222)).as_deref(), Some("552.22"));
        assert_eq!(nvidia_driver_from_umd(umd(32, 0, 16, 105)).as_deref(), Some("601.05"));
        assert_eq!(umd_version_parts(umd(32, 0, 15, 8157)), [32, 0, 15, 8157]);
    }

    #[test]
    fn vendor_names_and_nvidia_indices() {
        assert_eq!(vendor_name(0x10DE), "nvidia");
        assert_eq!(vendor_name(0x1234), "other");
        let mk = |index, vendor_id, software| GpuAdapter {
            index,
            name: String::new(),
            vendor: vendor_name(vendor_id).into(),
            vendor_id,
            device_id: 0,
            luid: String::new(),
            dedicated_vram_mb: 0,
            shared_memory_mb: 0,
            software,
            driver_version: None,
            nvidia_driver: None,
            vram: None,
        };
        let list = vec![mk(0, VENDOR_INTEL, false), mk(1, VENDOR_NVIDIA, false), mk(2, VENDOR_MICROSOFT, true)];
        assert_eq!(nvidia_adapter_indices(&list), vec![1]);
    }

    #[test]
    fn enumeration_never_panics() {
        // CI runners have only the Basic Render Driver; the dev rig has a real GPU. Either way the
        // call must succeed and report adapters in enumeration order.
        let list = adapters();
        assert!(list.windows(2).all(|w| w[0].index < w[1].index));
    }
}
