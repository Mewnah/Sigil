//! Shared cpal device lookup by name (id === name from `list_*_devices`).
//! Empty / `"default"` / missing → host default device. Names are compared trimmed.
//!
//! A microphone picked by name is found by its exact name, else by the one device of the same
//! family ([`match_saved_device`]): Windows renumbers a USB endpoint that moves to another port
//! ("Microphone (USB Audio)" becomes "Microphone (2- USB Audio)"), and a saved name that only
//! matches exactly would leave the capture retrying a name that never returns. It never falls
//! back to the system default: that can be a virtual cable carrying other audio.
//!
//! That family rule is for the start of a capture. While the hub reopens a microphone it lost, the
//! stricter [`match_runtime_device`] applies: only a family member that was not listed when the
//! loss began can stand in, so a second identical microphone that was there all along (two USB
//! headsets, a webcam next to a headset) never takes over for the lost one.

use cpal::traits::{DeviceTrait, HostTrait};
use cpal::{Device, Host};

fn is_default_sentinel(name: &str) -> bool {
    let t = name.trim();
    t.is_empty() || t.eq_ignore_ascii_case("default")
}

fn same_name(device: &Device, wanted: &str) -> bool {
    device.name().map(|n| n.trim() == wanted.trim()).unwrap_or(false)
}

/// `name` lower-cased and trimmed, without Windows' endpoint numbering (`N- ` at the start of the
/// name or right after an opening parenthesis).
pub(crate) fn device_family(name: &str) -> String {
    let lower = name.trim().to_lowercase();
    let mut out = String::with_capacity(lower.len());
    let mut rest = lower.as_str();
    let mut at_start = true;
    while let Some(c) = rest.chars().next() {
        if at_start {
            let digits = rest.bytes().take_while(u8::is_ascii_digit).count();
            if digits > 0 && rest[digits..].starts_with("- ") {
                rest = &rest[digits + 2..];
                at_start = false;
                continue;
            }
        }
        out.push(c);
        rest = &rest[c.len_utf8()..];
        at_start = c == '(';
    }
    out
}

/// Index of the device `saved` names among `names`: the exact name, else the only device of the
/// same family (`renamed = true`). `None` when it is absent or several devices share the family.
pub(crate) fn match_saved_device(saved: &str, names: &[String]) -> Option<(usize, bool)> {
    let saved = saved.trim();
    if let Some(i) = names.iter().position(|n| n.trim() == saved) {
        return Some((i, false));
    }
    let family = device_family(saved);
    let mut same = names
        .iter()
        .enumerate()
        .filter(|(_, n)| device_family(n) == family);
    match (same.next(), same.next()) {
        (Some((i, _)), None) => Some((i, true)),
        _ => None,
    }
}

/// Index of the device `saved` names while a capture follows it after it was lost: the exact name,
/// else the one device of its family that was **not** present when the loss started
/// (`present_at_loss`: the family's names then). `None` for `present_at_loss`: no snapshot was
/// taken, so only the exact name counts. A device that was there all along never stands in for the
/// lost one: with two identical USB microphones, unplugging one must not move the capture to the
/// other. `names` are the devices listed now.
pub(crate) fn match_runtime_device(
    saved: &str,
    names: &[String],
    present_at_loss: Option<&[String]>,
) -> Option<(usize, bool)> {
    let saved = saved.trim();
    if let Some(i) = names.iter().position(|n| n.trim() == saved) {
        return Some((i, false));
    }
    let present = present_at_loss?;
    let family = device_family(saved);
    let mut same = names
        .iter()
        .enumerate()
        .filter(|(_, n)| device_family(n) == family && !present.iter().any(|p| p.trim() == n.trim()));
    match (same.next(), same.next()) {
        (Some((i, _)), None) => Some((i, true)),
        _ => None,
    }
}

/// The names among `names` of `saved`'s family (the loss snapshot of [`match_runtime_device`]).
pub(crate) fn family_members(saved: &str, names: &[String]) -> Vec<String> {
    let family = device_family(saved);
    names
        .iter()
        .filter(|n| device_family(n) == family)
        .map(|n| n.trim().to_string())
        .collect()
}

/// The names of the input devices `host` lists now (`None`: it cannot list them).
pub(crate) fn input_device_names(host: &Host) -> Option<Vec<String>> {
    let devices = host.input_devices().ok()?;
    Some(devices.map(|d| d.name().unwrap_or_default()).collect())
}

/// Resolve an input device by cpal name, or the host default. A named device that is not listed
/// under that exact name is taken from the one device of its family (a renumbered USB endpoint).
pub fn resolve_input_device(host: &Host, device_name: Option<&str>) -> Result<Device, String> {
    match device_name {
        Some(name) if !is_default_sentinel(name) => {
            let devices: Vec<Device> = host.input_devices().map_err(|e| e.to_string())?.collect();
            let names: Vec<String> = devices.iter().map(|d| d.name().unwrap_or_default()).collect();
            let (i, renamed) = match_saved_device(name, &names)
                .ok_or_else(|| format!("Input device '{}' not found", name.trim()))?;
            if renamed {
                tracing::info!(
                    target: "sigil::audio::hub",
                    "input device '{}' is now listed as '{}' (renumbered endpoint)",
                    name.trim(),
                    names[i].trim()
                );
            }
            Ok(devices.into_iter().nth(i).expect("index from the same list"))
        }
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
            .find(|d| same_name(d, device_name))
            .ok_or_else(|| format!("Output device '{}' not found", device_name.trim()))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn names(list: &[&str]) -> Vec<String> {
        list.iter().map(|n| n.to_string()).collect()
    }

    #[test]
    fn family_drops_windows_endpoint_numbering() {
        assert_eq!(device_family("Microphone (2- USB Audio)"), "microphone (usb audio)");
        assert_eq!(device_family(" Microphone (13- USB Audio) "), "microphone (usb audio)");
        // Parentheses and digits that are not endpoint numbering stay.
        assert_eq!(device_family("Line 1 (Virtual Audio Cable)"), "line 1 (virtual audio cable)");
        assert_eq!(device_family("Mic (2-USB)"), "mic (2-usb)");
    }

    /// The renamed endpoint of a USB mic is found under its new name; a missing or ambiguous one
    /// is not guessed, and nothing resolves to another family.
    #[test]
    fn a_renumbered_usb_microphone_is_found_by_family() {
        let listed = names(&["Microphone (Realtek(R) Audio)", "Microphone (2- USB Audio)"]);
        // The exact name wins, even when another device shares the family.
        assert_eq!(match_saved_device("Microphone (Realtek(R) Audio)", &listed), Some((0, false)));
        assert_eq!(match_saved_device("Microphone (2- USB Audio)", &listed), Some((1, false)));
        // Saved as "(USB Audio)" (or "(3- USB Audio)"), re-plugged as "(2- USB Audio)".
        assert_eq!(match_saved_device("Microphone (USB Audio)", &listed), Some((1, true)));
        assert_eq!(match_saved_device("Microphone (3- USB Audio)", &listed), Some((1, true)));
        // Unplugged for good: never another family, never a default.
        assert_eq!(match_saved_device("Microphone (Yeti Stereo)", &listed), None);
        // Two devices of the family and neither is the saved name: do not guess.
        let both = names(&["Microphone (2- USB Audio)", "Microphone (3- USB Audio)"]);
        assert_eq!(match_saved_device("Microphone (USB Audio)", &both), None);
        assert_eq!(match_saved_device("Microphone (USB Audio)", &[]), None);
    }

    /// A capture that follows a lost microphone takes only a device that was not listed when it was
    /// lost: the other identical microphone, there all along, never stands in.
    #[test]
    fn a_lost_microphone_is_replaced_only_by_a_device_that_was_not_there_at_loss() {
        let headset = "Microphone (USB Audio Device)";
        // The webcam's microphone (same family) was listed all along; the headset is gone.
        let only_webcam = names(&["Microphone (2- USB Audio Device)"]);
        let at_loss = family_members(headset, &only_webcam);
        assert_eq!(at_loss, names(&["Microphone (2- USB Audio Device)"]));
        assert_eq!(match_runtime_device(headset, &only_webcam, Some(&at_loss)), None);
        // Only the exact name counts when no snapshot was taken.
        assert_eq!(match_runtime_device(headset, &only_webcam, None), None);
        // Re-plugged into another port it is listed under a number the snapshot did not hold.
        let moved = names(&["Microphone (2- USB Audio Device)", "Microphone (3- USB Audio Device)"]);
        assert_eq!(match_runtime_device(headset, &moved, Some(&at_loss)), Some((1, true)));
        // Back under its own name: exact, whatever the snapshot says.
        let back = names(&["Microphone (2- USB Audio Device)", headset]);
        assert_eq!(match_runtime_device(headset, &back, Some(&at_loss)), Some((1, false)));
        assert_eq!(match_runtime_device(headset, &back, None), Some((1, false)));
        // Two newcomers: no guess.
        let two = names(&["Microphone (3- USB Audio Device)", "Microphone (4- USB Audio Device)"]);
        assert_eq!(match_runtime_device(headset, &two, Some(&[])), None);
    }
}
