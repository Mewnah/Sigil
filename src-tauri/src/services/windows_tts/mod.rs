//! `windows-tts` plugin: Windows SAPI 5 voices (`SpVoice`) speaking straight to a SAPI audio output.
//!
//! The COM object is created in plugin setup on the main (STA) thread, and the commands are sync so
//! they run on that same thread. `Speak` uses `SVSFlagsAsync`, so a call returns immediately.

use serde::{Deserialize, Deserializer, Serialize};
use tauri::{
    command,
    plugin::{Builder, TauriPlugin},
    Manager, Runtime, State,
};
use windows::{
    core::BSTR,
    Win32::{
        Media::Speech::{ISpeechObjectToken, ISpeechObjectTokens, ISpeechVoice, SVSFDefault, SVSFlagsAsync, SpVoice, SpeechVoiceSpeakFlags},
        System::Com::{CoCreateInstance, CoInitialize, CLSCTX_ALL},
    },
};

use self::intf::Intf;
mod intf;

/// SAPI `SetRate` range.
const SAPI_RATE_MIN: i32 = -10;
const SAPI_RATE_MAX: i32 = 10;
/// Lowest rate multiplier the UI offers; maps to SAPI -10.
const RATE_MULTIPLIER_MIN: f32 = 0.1;
/// Highest rate multiplier the UI offers; maps to SAPI +10.
const RATE_MULTIPLIER_MAX: f32 = 5.0;

#[derive(Default)]
pub struct WindowsTTSPlugin {
    intf: Option<Intf<ISpeechVoice>>,
}

#[derive(Serialize, Deserialize, Debug)]
struct SpeechObject {
    pub id: String,
    pub label: String,
}

#[derive(Debug)]
struct ISpeechToken {
    id: String,
    pub t: Intf<ISpeechObjectToken>,
}

impl ISpeechToken {
    fn get_desc(&self) -> Option<SpeechObject> {
        unsafe {
            let id = self.t.0.Id().ok()?;
            let label = self.t.0.GetDescription(0).ok()?;
            Some(SpeechObject {
                id: id.to_string(),
                label: label.to_string(),
            })
        }
    }
}

impl WindowsTTSPlugin {
    fn new() -> Self {
        let hr = unsafe { CoInitialize(None) };
        if hr.is_err() {
            return Self::default();
        }
        let Ok(instance) = (unsafe { CoCreateInstance(&SpVoice, None, CLSCTX_ALL) }) else {
            return Self::default();
        };

        Self { intf: Some(Intf(instance)) }
    }

    fn list_devices(&self) -> Option<Vec<ISpeechToken>> {
        self.intf
            .as_ref()
            .and_then(|sp| unsafe { sp.GetAudioOutputs(&BSTR::new(), &BSTR::new()) }.ok())
            .and_then(into_speech_tokens)
    }
    fn list_voices(&self) -> Option<Vec<ISpeechToken>> {
        self.intf
            .as_ref()
            .and_then(|sp| unsafe { sp.GetVoices(&BSTR::new(), &BSTR::new()) }.ok())
            .and_then(into_speech_tokens)
    }
}

fn into_speech_tokens(tokens: ISpeechObjectTokens) -> Option<Vec<ISpeechToken>> {
    let count = unsafe { tokens.Count() }.ok()?;
    let list = (0..count)
        .filter_map(|i| {
            let token = unsafe { tokens.Item(i) }.ok()?;
            let id = unsafe { token.Id() }.ok()?;
            Some(ISpeechToken {
                id: id.to_string(),
                t: Intf(token),
            })
        })
        .collect();
    Some(list)
}

#[derive(Serialize, Deserialize, Debug)]
pub struct RpcWindowsTTSConfig {
    devices: Vec<SpeechObject>,
    voices: Vec<SpeechObject>,
}

#[command]
fn get_voices(state: State<'_, WindowsTTSPlugin>) -> Result<RpcWindowsTTSConfig, &'static str> {
    let Some(devices): Option<Vec<SpeechObject>> = state
        .list_devices()
        .map(|list| list.iter().filter_map(ISpeechToken::get_desc).collect())
    else {
        return Err("Failed to get device list");
    };
    let Some(voices): Option<Vec<SpeechObject>> = state
        .list_voices()
        .map(|list| list.iter().filter_map(ISpeechToken::get_desc).collect())
    else {
        return Err("Failed to get voice list");
    };

    Ok(RpcWindowsTTSConfig { voices, devices })
}

/// A number that may arrive as a JSON number, a numeric string, or `null` (the webview serializes
/// `NaN` from a cleared input as `null`, which used to fail the whole command).
fn lenient_f32<'de, D: Deserializer<'de>>(d: D) -> Result<Option<f32>, D::Error> {
    let v = serde_json::Value::deserialize(d)?;
    Ok(match v {
        serde_json::Value::Number(n) => n.as_f64().map(|x| x as f32),
        serde_json::Value::String(s) => s.trim().parse::<f32>().ok(),
        _ => None,
    }
    .filter(|x| x.is_finite()))
}

#[derive(Serialize, Deserialize, Debug)]
pub struct RpcWindowsTTSSpeak {
    device: String,
    voice: String,
    value: String,
    /// 0 – 1 (missing / invalid = 1).
    #[serde(default, deserialize_with = "lenient_f32")]
    volume: Option<f32>,
    /// Multiplier: 0.1 – 1 – 5 (missing / invalid = 1).
    #[serde(default, deserialize_with = "lenient_f32")]
    rate: Option<f32>,
}

/// Sigil's rate multiplier → SAPI `SetRate` (-10..=10).
///
/// `[1, 5]` maps linearly onto `[0, 10]` (unchanged from before, truncating toward zero), and
/// `[0.1, 1)` now maps linearly onto `[-10, 0)`. The old `rate < 1` branch produced -10..-100,
/// outside SAPI's range, so `SetRate` failed and every rate below 1.0 made speech fail.
pub(crate) fn sapi_rate(multiplier: f32) -> i32 {
    if !multiplier.is_finite() {
        return 0;
    }
    let steps = if multiplier >= 1.0 {
        (multiplier - 1.0) / (RATE_MULTIPLIER_MAX - 1.0) * SAPI_RATE_MAX as f32
    } else {
        (multiplier - 1.0) / (1.0 - RATE_MULTIPLIER_MIN) * -(SAPI_RATE_MIN as f32)
    };
    // `as` saturates on overflow; the clamp keeps out-of-range multipliers valid.
    (steps as i32).clamp(SAPI_RATE_MIN, SAPI_RATE_MAX)
}

/// Volume 0–1 → SAPI `SetVolume` (0..=100).
pub(crate) fn sapi_volume(volume: f32) -> i32 {
    if !volume.is_finite() {
        return 100;
    }
    ((volume * 100.0).round() as i32).clamp(0, 100)
}

#[command]
fn speak(data: RpcWindowsTTSSpeak, state: State<'_, WindowsTTSPlugin>) -> Result<(), &'static str> {
    if data.value.trim().is_empty() {
        return Ok(());
    }
    let Some(sp_voice) = &state.intf else {
        return Err("Plugin is not initialized");
    };

    if unsafe { sp_voice.0.SetVolume(sapi_volume(data.volume.unwrap_or(1.0))) }.is_err() {
        return Err("Unable to update volume");
    }
    if unsafe { sp_voice.0.SetRate(sapi_rate(data.rate.unwrap_or(1.0))) }.is_err() {
        return Err("Unable to update rate");
    }

    // An empty device keeps SAPI's current (default) output.
    if !data.device.is_empty() {
        let Some(_apply_res_device) = state
            .list_devices()
            .as_deref()
            .and_then(|list| list.iter().find(|t| t.id == data.device))
            .and_then(|token| unsafe { sp_voice.0.putref_AudioOutput(&token.t.0).ok() })
        else {
            return Err("Failed to apply device");
        };
    }
    let Some(_apply_res_voice) = state
        .list_voices()
        .as_deref()
        .and_then(|list| list.iter().find(|t| t.id == data.voice))
        .and_then(|token| unsafe { sp_voice.0.putref_Voice(&token.t.0).ok() })
    else {
        return Err("Failed to apply voice");
    };

    if unsafe { sp_voice.Speak(&data.value.into(), SpeechVoiceSpeakFlags(SVSFDefault.0 | SVSFlagsAsync.0)) }.is_err() {
        Err("Unable to process text")
    } else {
        Ok(())
    }
}

pub fn init<R: Runtime>() -> TauriPlugin<R> {
    Builder::new("windows-tts")
        .invoke_handler(tauri::generate_handler![speak, get_voices])
        .setup(|app, _api| {
            app.manage(WindowsTTSPlugin::new());
            Ok(())
        })
        .build()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn sapi_rate_stays_in_range_for_every_ui_value() {
        let mut r = 0.0f32;
        while r <= 6.0 {
            let s = sapi_rate(r);
            assert!((SAPI_RATE_MIN..=SAPI_RATE_MAX).contains(&s), "rate {r} -> {s}");
            r += 0.05;
        }
        for x in [f32::NAN, f32::INFINITY, f32::NEG_INFINITY, -3.0, 1e9] {
            let s = sapi_rate(x);
            assert!((SAPI_RATE_MIN..=SAPI_RATE_MAX).contains(&s), "rate {x} -> {s}");
        }
    }

    #[test]
    fn sapi_rate_mapping_points() {
        assert_eq!(sapi_rate(1.0), 0);
        assert_eq!(sapi_rate(5.0), 10);
        assert_eq!(sapi_rate(3.0), 5);
        // Unchanged from the previous mapping for >= 1 (truncation toward zero).
        assert_eq!(sapi_rate(1.2), 0);
        assert_eq!(sapi_rate(2.0), 2);
        // Previously -10..-100 (rejected by SAPI); now a proportional slow-down.
        assert_eq!(sapi_rate(0.1), -10);
        assert_eq!(sapi_rate(0.55), -5);
        assert_eq!(sapi_rate(0.95), 0);
        assert_eq!(sapi_rate(0.0), -10);
        assert_eq!(sapi_rate(f32::NAN), 0);
    }

    #[test]
    fn sapi_rate_is_monotonic() {
        let mut prev = i32::MIN;
        let mut r = 0.1f32;
        while r <= 5.0 {
            let s = sapi_rate(r);
            assert!(s >= prev, "rate {r}: {s} < {prev}");
            prev = s;
            r += 0.05;
        }
    }

    #[test]
    fn sapi_volume_is_clamped() {
        assert_eq!(sapi_volume(1.0), 100);
        assert_eq!(sapi_volume(0.29), 29);
        assert_eq!(sapi_volume(0.0), 0);
        assert_eq!(sapi_volume(-1.0), 0);
        assert_eq!(sapi_volume(3.0), 100);
        assert_eq!(sapi_volume(f32::NAN), 100);
    }

    #[test]
    fn speak_payload_tolerates_null_and_strings() {
        let v: RpcWindowsTTSSpeak = serde_json::from_value(serde_json::json!({
            "device": "d", "voice": "v", "value": "hello", "volume": null, "rate": "0.5"
        }))
        .unwrap();
        assert_eq!(v.volume, None);
        assert_eq!(v.rate, Some(0.5));
        let v: RpcWindowsTTSSpeak = serde_json::from_value(serde_json::json!({
            "device": "d", "voice": "v", "value": "hello", "volume": 0.8, "rate": 1.5
        }))
        .unwrap();
        assert_eq!(v.volume, Some(0.8));
        assert_eq!(v.rate, Some(1.5));
        let v: RpcWindowsTTSSpeak =
            serde_json::from_value(serde_json::json!({"device": "", "voice": "v", "value": "x", "rate": "fast"})).unwrap();
        assert_eq!((v.volume, v.rate), (None, None));
    }
}
