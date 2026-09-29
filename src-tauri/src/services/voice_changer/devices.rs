//! Opening the saved input / output / monitor devices of a run.
//!
//! Devices are persisted by name (cpal's name is the id `list_*_devices` returns). A saved name
//! can go stale between sessions: the device is unplugged, or Windows renumbers the endpoint when
//! it moves to another USB port ("Microphone (2- USB Audio)" becomes "Microphone (3- USB Audio)").
//! Lookup is therefore: the exact name, else the one device of the same family (the name without
//! Windows' `N- ` endpoint numbering), else a clear "not connected" error. It never guesses
//! between several devices of one family and never falls back to the system default, which could
//! be a virtual cable carrying the voice changer's own output (a feedback loop) or open speakers.

use cpal::traits::{DeviceTrait, HostTrait};

/// Which side of the run a device is for.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(super) enum DeviceRole {
    Input,
    Output,
    Monitor,
}

impl DeviceRole {
    fn label(self) -> &'static str {
        match self {
            DeviceRole::Input => "input",
            DeviceRole::Output => "output",
            DeviceRole::Monitor => "monitor",
        }
    }
}

/// Empty / `"default"` (any case) = the system default device.
pub(super) fn is_default_device(name: &str) -> bool {
    let t = name.trim();
    t.is_empty() || t.eq_ignore_ascii_case("default")
}

/// `name` lower-cased and trimmed, without Windows' endpoint numbering (`N- ` at the start of the
/// name or right after an opening parenthesis).
pub(super) fn device_family(name: &str) -> String {
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
pub(super) fn match_saved_device(saved: &str, names: &[String]) -> Option<(usize, bool)> {
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

/// A device opened for a run, and the name it was found under.
pub(super) struct OpenedDevice {
    pub(super) device: cpal::Device,
    pub(super) name: String,
    /// The saved name, when the device was found under a renumbered name instead.
    pub(super) renamed_from: Option<String>,
}

/// Find the device `saved` names for `role` (see the module docs). The error is a user-facing
/// sentence.
pub(super) fn open_saved_device(host: &cpal::Host, role: DeviceRole, saved: &str) -> Result<OpenedDevice, String> {
    let input = role == DeviceRole::Input;
    if is_default_device(saved) {
        let device = if input {
            host.default_input_device()
        } else {
            host.default_output_device()
        };
        let device = device.ok_or_else(|| {
            format!(
                "There is no default {} device. Connect one or choose a device in the voice changer settings.",
                if input { "input" } else { "output" }
            )
        })?;
        let name = device.name().unwrap_or_else(|_| "default".to_string());
        return Ok(OpenedDevice {
            device,
            name,
            renamed_from: None,
        });
    }
    let listed: Vec<cpal::Device> = if input {
        host.input_devices()
            .map_err(|e| format!("Cannot list input devices: {e}"))?
            .collect()
    } else {
        host.output_devices()
            .map_err(|e| format!("Cannot list output devices: {e}"))?
            .collect()
    };
    let names: Vec<String> = listed
        .iter()
        .map(|d| d.name().unwrap_or_default())
        .collect();
    let Some((i, renamed)) = match_saved_device(saved, &names) else {
        return Err(format!(
            "The {} device '{}' is not connected. Connect it or choose another {} device in the voice changer settings.",
            role.label(),
            saved.trim(),
            role.label()
        ));
    };
    let name = names[i].clone();
    let device = listed.into_iter().nth(i).expect("index from the same list");
    Ok(OpenedDevice {
        device,
        name,
        renamed_from: renamed.then(|| saved.trim().to_string()),
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn device_family_drops_windows_endpoint_numbering() {
        assert_eq!(device_family("Microphone (2- USB Audio)"), "microphone (usb audio)");
        assert_eq!(device_family(" Microphone (13- USB Audio) "), "microphone (usb audio)");
        assert_eq!(device_family("2- USB Audio"), "usb audio");
        // Parentheses and digits that are not endpoint numbering stay.
        assert_eq!(device_family("Speakers (Realtek(R) Audio)"), "speakers (realtek(r) audio)");
        assert_eq!(device_family("Line 1 (Virtual Audio Cable)"), "line 1 (virtual audio cable)");
        assert_eq!(device_family("Mic (2-USB)"), "mic (2-usb)");
        assert_eq!(device_family("Headset (Ünïcode 3- X)"), "headset (ünïcode 3- x)");
    }

    /// The exact name wins; a renumbered device is found when it is the only one of its family;
    /// an absent or ambiguous device is not guessed.
    #[test]
    fn saved_devices_match_exactly_or_by_unique_family() {
        let names: Vec<String> = [
            "Microphone (3- USB Audio)",
            "Microphone (Realtek(R) Audio)",
            "Headset Microphone (2- Arctis 7)",
            "Headset Microphone (4- Arctis 7)",
            "CABLE Output (VB-Audio Virtual Cable)",
        ]
        .map(String::from)
        .to_vec();
        assert_eq!(match_saved_device("Microphone (Realtek(R) Audio)", &names), Some((1, false)));
        assert_eq!(match_saved_device(" Headset Microphone (4- Arctis 7) ", &names), Some((3, false)));
        // Windows moved the USB mic from endpoint 2 to 3.
        assert_eq!(match_saved_device("Microphone (2- USB Audio)", &names), Some((0, true)));
        // Saved before Windows started numbering it.
        assert_eq!(match_saved_device("Microphone (USB Audio)", &names), Some((0, true)));
        // Two identical headsets, neither the saved endpoint: do not guess.
        assert_eq!(match_saved_device("Headset Microphone (7- Arctis 7)", &names), None);
        // Unplugged: never another device.
        assert_eq!(match_saved_device("Microphone (Yeti Stereo)", &names), None);
        assert!(is_default_device(" Default "));
        assert!(is_default_device(""));
        assert!(!is_default_device("Microphone (Yeti Stereo)"));
    }
}
