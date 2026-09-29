//! Shared cpal device lookup by name (id === name from `list_*_devices`).
//! Empty / `"default"` / missing → host default device.

use cpal::traits::{DeviceTrait, HostTrait};
use cpal::{Device, Host};

fn is_default_sentinel(name: &str) -> bool {
    let t = name.trim();
    t.is_empty() || t.eq_ignore_ascii_case("default")
}

/// Resolve an input device by cpal name, or the host default.
pub fn resolve_input_device(host: &Host, device_name: Option<&str>) -> Result<Device, String> {
    match device_name {
        Some(name) if !is_default_sentinel(name) => host
            .input_devices()
            .map_err(|e| e.to_string())?
            .find(|d| d.name().unwrap_or_default() == name)
            .ok_or_else(|| format!("Input device '{}' not found", name)),
        _ => host
            .default_input_device()
            .ok_or_else(|| "No default input device".to_string()),
    }
}

/// Resolve an output device by cpal name, or the host default.
pub fn resolve_output_device(host: &Host, device_name: &str) -> Result<Device, String> {
    if is_default_sentinel(device_name) {
        host.default_output_device()
            .ok_or_else(|| "No default output device".to_string())
    } else {
        host.output_devices()
            .map_err(|e| e.to_string())?
            .find(|d| d.name().unwrap_or_default() == device_name)
            .ok_or_else(|| format!("Output device '{}' not found", device_name))
    }
}
