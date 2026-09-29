//! Capture source specs, including output devices opened for loopback capture (phase 1 plumbing).
//!
//! A capture source is named by the same string the device pickers persist:
//! - `""`, `"default"` or `"communications"`: the system default input (follows default changes);
//! - `"<input device name>"`: that input device;
//! - `"loopback:default"` (or `"loopback:"`): what the system default output plays (follows changes);
//! - `"loopback:<output device name>"`: what that output device plays.
//!
//! Loopback uses WASAPI shared-mode loopback (cpal 0.15 opens an input stream on a render endpoint
//! with `AUDCLNT_STREAMFLAGS_LOOPBACK`) at the output's mix format. Two properties differ from a
//! microphone and every consumer must expect them:
//! - WASAPI delivers **no packets at all** while nothing plays on the endpoint (not even silence),
//!   so a loopback source is never treated as stalled, and time-based consumers (VAD) must fill
//!   idle periods themselves;
//! - the format is the output mix format (often 48 kHz stereo, sometimes 5.1/7.1).
//!
//! Phase 2 (per-process loopback through the Windows 11 process-loopback API) is P6.

use cpal::traits::DeviceTrait;
use cpal::{Device, Host, SupportedStreamConfig};

use super::device_resolve::{resolve_input_device, resolve_output_device};

/// Prefix of loopback source ids (`"loopback:<output device name>"`).
pub const LOOPBACK_PREFIX: &str = "loopback:";
/// Key of the system default input source.
pub const DEFAULT_INPUT_KEY: &str = "default";
/// Key of the system default output's loopback source.
pub const DEFAULT_LOOPBACK_KEY: &str = "loopback:default";

/// What a consumer asked to capture. Default specs follow the system default device.
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub enum SourceSpec {
    /// The system default input device (follows default changes).
    DefaultInput,
    /// An input device by cpal name.
    Input(String),
    /// Loopback of the system default output device (follows default changes).
    DefaultLoopback,
    /// Loopback of an output device by cpal name.
    Loopback(String),
}

fn is_default_word(s: &str) -> bool {
    s.is_empty() || s.eq_ignore_ascii_case("default") || s.eq_ignore_ascii_case("communications")
}

/// `s` without an ASCII `prefix` matched case-insensitively. Compares bytes, so a device name whose
/// first characters are multi-byte (Cyrillic, Arabic, Greek ...) never makes it slice inside a
/// character: a match means those bytes are ASCII, so the end of the prefix is a char boundary.
fn strip_prefix_ascii_ci<'a>(s: &'a str, prefix: &str) -> Option<&'a str> {
    let n = prefix.len();
    let head = s.as_bytes().get(..n)?;
    if head.eq_ignore_ascii_case(prefix.as_bytes()) {
        s.get(n..)
    } else {
        None
    }
}

impl SourceSpec {
    /// Parse a persisted / IPC device id (see the module docs). `None` is the default input.
    pub fn parse(device: Option<&str>) -> Self {
        let id = device.unwrap_or("").trim();
        if let Some(rest) = strip_prefix_ascii_ci(id, LOOPBACK_PREFIX) {
            let name = rest.trim();
            return if is_default_word(name) {
                SourceSpec::DefaultLoopback
            } else {
                SourceSpec::Loopback(name.to_string())
            };
        }
        if is_default_word(id) {
            SourceSpec::DefaultInput
        } else {
            SourceSpec::Input(id.to_string())
        }
    }

    /// Stable source key used in events (`RustAudioStreamEvent.source`) and status.
    pub fn key(&self) -> String {
        match self {
            SourceSpec::DefaultInput => DEFAULT_INPUT_KEY.to_string(),
            SourceSpec::Input(name) => name.clone(),
            SourceSpec::DefaultLoopback => DEFAULT_LOOPBACK_KEY.to_string(),
            SourceSpec::Loopback(name) => format!("{LOOPBACK_PREFIX}{name}"),
        }
    }

    pub fn is_loopback(&self) -> bool {
        matches!(self, SourceSpec::DefaultLoopback | SourceSpec::Loopback(_))
    }

    /// Whether the source moves to a new device when the system default changes.
    pub fn follows_default(&self) -> bool {
        matches!(self, SourceSpec::DefaultInput | SourceSpec::DefaultLoopback)
    }

    /// The explicit spec of a resolved physical device (used to reopen it after a failure).
    pub fn physical(name: &str, loopback: bool) -> Self {
        if loopback {
            SourceSpec::Loopback(name.to_string())
        } else {
            SourceSpec::Input(name.to_string())
        }
    }
}

/// Resolve `spec` to a cpal device: `(device, is_loopback)`.
pub fn resolve_cpal_device(host: &Host, spec: &SourceSpec) -> Result<(Device, bool), String> {
    match spec {
        SourceSpec::DefaultInput => resolve_input_device(host, None)
            .map(|d| (d, false))
            .map_err(|_| "No default input device (is a microphone connected and enabled?)".to_string()),
        SourceSpec::Input(name) => resolve_input_device(host, Some(name)).map(|d| (d, false)),
        SourceSpec::DefaultLoopback => {
            ensure_loopback_supported()?;
            resolve_output_device(host, "default")
                .map(|d| (d, true))
                .map_err(|_| "No default output device to capture (loopback)".to_string())
        }
        SourceSpec::Loopback(name) => {
            ensure_loopback_supported()?;
            resolve_output_device(host, name)
                .map(|d| (d, true))
                .map_err(|_| format!("Output device '{name}' not found (loopback)"))
        }
    }
}

fn ensure_loopback_supported() -> Result<(), String> {
    if cfg!(windows) {
        Ok(())
    } else {
        Err("Loopback capture of output devices is only supported on Windows".to_string())
    }
}

/// Stream config for capturing `device`: the input default for microphones, the output mix format
/// for loopback (WASAPI loopback must use the endpoint's shared-mode mix format).
pub fn capture_config(device: &Device, loopback: bool) -> Result<SupportedStreamConfig, String> {
    let config = if loopback {
        device.default_output_config()
    } else {
        device.default_input_config()
    };
    config.map_err(|e| {
        let what = if loopback { "Output device format (loopback)" } else { "Input device format" };
        format!("{what}: {e}")
    })
}

#[cfg(test)]
mod tests {
    use super::SourceSpec;

    #[test]
    fn source_ids_parse_to_specs_and_back() {
        for (id, spec, key) in [
            (None, SourceSpec::DefaultInput, "default"),
            (Some(""), SourceSpec::DefaultInput, "default"),
            (Some(" Default "), SourceSpec::DefaultInput, "default"),
            (Some("communications"), SourceSpec::DefaultInput, "default"),
            (Some("Mic (USB)"), SourceSpec::Input("Mic (USB)".into()), "Mic (USB)"),
            (Some("loopback:"), SourceSpec::DefaultLoopback, "loopback:default"),
            (Some("LOOPBACK:default"), SourceSpec::DefaultLoopback, "loopback:default"),
            (Some("loopback: Speakers (Realtek)"), SourceSpec::Loopback("Speakers (Realtek)".into()), "loopback:Speakers (Realtek)"),
            // Non-ASCII names: byte 9 falls inside a multi-byte character (must not panic).
            (Some("Микрофон (Realtek(R) Audio)"), SourceSpec::Input("Микрофон (Realtek(R) Audio)".into()), "Микрофон (Realtek(R) Audio)"),
            (Some("ميكروفون (USB)"), SourceSpec::Input("ميكروفون (USB)".into()), "ميكروفون (USB)"),
            (Some("Μικρόφωνο"), SourceSpec::Input("Μικρόφωνο".into()), "Μικρόφωνο"),
            (Some("マイク"), SourceSpec::Input("マイク".into()), "マイク"),
            (Some("loopbackµ"), SourceSpec::Input("loopbackµ".into()), "loopbackµ"),
            (Some("loopback:Динамики (USB)"), SourceSpec::Loopback("Динамики (USB)".into()), "loopback:Динамики (USB)"),
            (Some("Loopback:ﺳﻤﺎﻋﺎﺕ"), SourceSpec::Loopback("ﺳﻤﺎﻋﺎﺕ".into()), "loopback:ﺳﻤﺎﻋﺎﺕ"),
        ] {
            let parsed = SourceSpec::parse(id);
            assert_eq!(parsed, spec, "{id:?}");
            assert_eq!(parsed.key(), key);
            // Keys round-trip (what the pickers persist is what the events report).
            assert_eq!(SourceSpec::parse(Some(&parsed.key())), parsed);
        }
        assert!(SourceSpec::DefaultLoopback.is_loopback() && SourceSpec::DefaultLoopback.follows_default());
        assert!(!SourceSpec::Input("x".into()).follows_default());
        assert_eq!(SourceSpec::physical("Speakers", true), SourceSpec::Loopback("Speakers".into()));
        assert_eq!(SourceSpec::physical("Mic", false), SourceSpec::Input("Mic".into()));
    }

    #[test]
    fn parsing_never_slices_inside_a_character() {
        // Every multi-byte character at every byte offset around the prefix length.
        for c in ['о', 'م', 'ώ', 'µ', 'マ', '🎤'] {
            for ascii in 0..=10 {
                let id: String = "loopback:x".chars().take(ascii).chain(std::iter::repeat_n(c, 6)).collect();
                let parsed = SourceSpec::parse(Some(&id));
                assert_eq!(SourceSpec::parse(Some(&parsed.key())), parsed, "{id:?}");
                if ascii >= 9 {
                    assert!(parsed.is_loopback(), "{id:?}");
                } else {
                    assert_eq!(parsed, SourceSpec::Input(id.clone()));
                }
            }
        }
    }
}
