//! Real-time voice changer: capture → **phase vocoder** pitch (constant-time) → device rate match
//! ([`rubato`]) → optional formant tilt → playback.
//!
//! The output queue is **capped** (~[`MAX_QUEUE_MS`]) as a safety valve; with balanced
//! sample rates through the chain, underruns should be rare. Hold/crossfade still softens gaps.

use cpal::traits::{DeviceTrait, HostTrait, StreamTrait};
use pitch_shift::PitchShifter;
use rubato::{
    Resampler, SincFixedIn, SincInterpolationParameters, SincInterpolationType, WindowFunction,
};
use serde::{Deserialize, Serialize};
use std::collections::VecDeque;
use std::sync::mpsc::{channel, Sender};
use std::sync::{Arc, Mutex};
use std::thread;
use tauri::{
    command,
    plugin::{Builder, TauriPlugin},
    AppHandle, Emitter, Manager, Runtime, State,
};

const DEFAULT_PITCH_SEMITONES: f32 = 0.0;
const DEFAULT_FORMANT_SHIFT: f32 = 0.0;
const DEFAULT_VOCODER_WINDOW_MS: u32 = 45;
const DEFAULT_VOCODER_OVERSAMPLE: u32 = 8;

/// Input frames per processing block (per channel).
const CHUNK_FRAMES: usize = 512;

/// Output prefill before starting playback (latency cushion).
const PREFILL_MS: f64 = 55.0;

/// After an underrun, blend this many **frames** (per channel) from hold → new audio (stereo = 2×).
const RECOVER_FRAMES: usize = 16;

/// Cap queue depth so deeper pitch (ratio > 1) cannot build multi-second latency.
/// Drops oldest samples when exceeded (keeps playback near real-time).
const MAX_QUEUE_MS: f64 = 100.0;

/// Treat pitch as zero below this (avoid ratio drift).
const PITCH_EPSILON: f32 = 0.01;

#[derive(Clone, Serialize, Deserialize, Debug)]
pub struct VoiceChangerPreset {
    pub id: String,
    pub name: String,
    pub pitch: f32,
    pub formant: f32,
}

fn get_presets() -> Vec<VoiceChangerPreset> {
    vec![
        VoiceChangerPreset {
            id: "default".to_string(),
            name: "Default".to_string(),
            pitch: 0.0,
            formant: 0.0,
        },
        VoiceChangerPreset {
            id: "deeper".to_string(),
            name: "Deeper".to_string(),
            pitch: -4.0,
            formant: -0.2,
        },
        VoiceChangerPreset {
            id: "higher".to_string(),
            name: "Higher".to_string(),
            pitch: 5.0,
            formant: 0.2,
        },
        VoiceChangerPreset {
            id: "chipmunk".to_string(),
            name: "Chipmunk".to_string(),
            pitch: 10.0,
            formant: 0.4,
        },
        VoiceChangerPreset {
            id: "deep_voice".to_string(),
            name: "Deep Voice".to_string(),
            pitch: -8.0,
            formant: -0.3,
        },
        VoiceChangerPreset {
            id: "subtle_lower".to_string(),
            name: "Subtle Lower".to_string(),
            pitch: -2.0,
            formant: 0.0,
        },
        VoiceChangerPreset {
            id: "subtle_higher".to_string(),
            name: "Subtle Higher".to_string(),
            pitch: 2.0,
            formant: 0.0,
        },
    ]
}

#[derive(Clone)]
pub struct VoiceChangerState {
    enabled: Arc<Mutex<bool>>,
    pitch_semitones: Arc<Mutex<f32>>,
    formant_shift: Arc<Mutex<f32>>,
    vocoder_window_ms: Arc<Mutex<u32>>,
    vocoder_oversample: Arc<Mutex<u32>>,
    output_device: Arc<Mutex<String>>,
    stop_sender: Arc<Mutex<Option<Sender<()>>>>,
    is_running: Arc<Mutex<bool>>,
}

impl VoiceChangerState {
    fn new() -> Self {
        Self {
            enabled: Arc::new(Mutex::new(false)),
            pitch_semitones: Arc::new(Mutex::new(DEFAULT_PITCH_SEMITONES)),
            formant_shift: Arc::new(Mutex::new(DEFAULT_FORMANT_SHIFT)),
            vocoder_window_ms: Arc::new(Mutex::new(DEFAULT_VOCODER_WINDOW_MS)),
            vocoder_oversample: Arc::new(Mutex::new(DEFAULT_VOCODER_OVERSAMPLE)),
            output_device: Arc::new(Mutex::new("default".to_string())),
            stop_sender: Arc::new(Mutex::new(None)),
            is_running: Arc::new(Mutex::new(false)),
        }
    }
}

fn default_param_vocoder_window_ms() -> u32 {
    DEFAULT_VOCODER_WINDOW_MS
}

fn default_param_vocoder_oversample() -> u32 {
    DEFAULT_VOCODER_OVERSAMPLE
}

#[derive(Serialize, Deserialize, Debug)]
pub struct VoiceChangerParams {
    pub enabled: bool,
    pub pitch_semitones: f32,
    pub formant_shift: f32,
    #[serde(default = "default_param_vocoder_window_ms")]
    pub vocoder_window_ms: u32,
    #[serde(default = "default_param_vocoder_oversample")]
    pub vocoder_oversample: u32,
    pub output_device: String,
}

fn clamp_vocoder_window_ms(v: u32) -> usize {
    v.clamp(30, 60) as usize
}

fn clamp_vocoder_oversample(v: u32) -> usize {
    match v {
        4 | 8 | 16 | 32 => v as usize,
        _ => DEFAULT_VOCODER_OVERSAMPLE as usize,
    }
}

struct OutputPlayback {
    deque: VecDeque<f32>,
    prefill_target: usize,
    max_queue_samples: usize,
    ch_out: usize,
    prefilled: bool,
    /// Last output per channel (for underrun hold).
    hold: Vec<f32>,
    /// True if the previous sample in this callback was filled from hold (deque empty).
    starving: bool,
    /// Interleaved samples left to crossfade after leaving starvation.
    recover_left: usize,
}

impl OutputPlayback {
    fn new(prefill_target: usize, max_queue_samples: usize, ch_out: usize) -> Self {
        Self {
            deque: VecDeque::with_capacity(max_queue_samples.max(prefill_target) + 8192),
            prefill_target,
            max_queue_samples: max_queue_samples.max(prefill_target + 1),
            ch_out,
            prefilled: false,
            hold: vec![0.0f32; ch_out.max(1)],
            starving: false,
            recover_left: 0,
        }
    }

    fn push_interleaved(&mut self, samples: &[f32]) {
        self.deque.extend(samples.iter().copied());
        while self.deque.len() > self.max_queue_samples {
            self.deque.pop_front();
        }
    }

    fn fill_callback(&mut self, data: &mut [f32]) {
        let ch_out = self.ch_out.max(1);
        debug_assert_eq!(data.len() % ch_out, 0);

        if !self.prefilled {
            if self.deque.len() >= self.prefill_target {
                self.prefilled = true;
            } else {
                data.fill(0.0);
                return;
            }
        }

        let recover_span = (RECOVER_FRAMES * ch_out).max(1);

        for i in 0..data.len() {
            let ch = i % ch_out;
            if let Some(v) = self.deque.pop_front() {
                if self.starving {
                    self.recover_left = recover_span;
                    self.starving = false;
                }

                let out_v = if self.recover_left > 0 {
                    let done = recover_span - self.recover_left;
                    let t = (done as f32 + 1.0) / recover_span as f32;
                    self.recover_left -= 1;
                    self.hold[ch].mul_add(1.0 - t, v * t)
                } else {
                    v
                };

                data[i] = out_v;
                self.hold[ch] = out_v;
            } else {
                self.starving = true;
                data[i] = self.hold[ch];
            }
        }
    }
}

/// One-pole lowpass state per channel (simple formant / brightness tilt).
#[derive(Clone, Default)]
struct FormantLpState {
    lp: f32,
}

/// Device rate only: map timeline of input device to output device (no pitch).
fn device_resample_ratio(r_in: u32, r_out: u32) -> f64 {
    r_out as f64 / r_in as f64
}

fn apply_phase_vocoder_channels(
    shifters: &mut [PitchShifter],
    oversample: usize,
    pitch_semitones: f32,
    planes_in: &[Vec<f32>],
    planes_out: &mut [Vec<f32>],
) {
    let frames = planes_in[0].len();
    // Always call `shift_pitch` so per-channel FIFO / phase state tracks the stream (0 semitones => unity shift).
    let shift = if pitch_semitones.abs() < PITCH_EPSILON {
        0.0f32
    } else {
        pitch_semitones
    };
    for c in 0..planes_in.len() {
        shifters[c].shift_pitch(
            oversample,
            shift,
            &planes_in[c][..frames],
            &mut planes_out[c][..frames],
        );
    }
}

fn sinc_params() -> SincInterpolationParameters {
    SincInterpolationParameters {
        sinc_len: 256,
        f_cutoff: 0.95,
        oversampling_factor: 256,
        interpolation: SincInterpolationType::Cubic,
        window: WindowFunction::BlackmanHarris2,
    }
}

fn make_device_resampler(
    ratio: f64,
    channels: usize,
) -> Result<SincFixedIn<f32>, rubato::ResamplerConstructionError> {
    const MAX_REL_RATIO: f64 = 4.0;
    SincFixedIn::new(
        ratio,
        MAX_REL_RATIO,
        sinc_params(),
        CHUNK_FRAMES,
        channels,
    )
}

fn interleaved_to_planar(interleaved: &[f32], ch_in: usize, ch_work: usize) -> Vec<Vec<f32>> {
    let frames = interleaved.len() / ch_in;
    let mut planes: Vec<Vec<f32>> = (0..ch_work).map(|_| Vec::with_capacity(frames)).collect();
    for f in 0..frames {
        for c in 0..ch_work {
            let v = if c < ch_in {
                interleaved[f * ch_in + c]
            } else {
                interleaved[f * ch_in + ch_in - 1]
            };
            planes[c].push(v);
        }
    }
    planes
}

fn planar_to_interleaved(planes: &[Vec<f32>], ch_work: usize, ch_out: usize) -> Vec<f32> {
    let frames = planes.first().map(|p| p.len()).unwrap_or(0);
    let mut out = Vec::with_capacity(frames * ch_out);
    for f in 0..frames {
        for c in 0..ch_out {
            let v = if c < ch_work {
                planes[c].get(f).copied().unwrap_or(0.0)
            } else {
                planes[ch_work - 1].get(f).copied().unwrap_or(0.0)
            };
            out.push(v);
        }
    }
    out
}

/// Spectral tilt: negative blends toward lowpassed (darker); positive emphasizes highs.
fn apply_formant_interleaved(
    samples: &mut [f32],
    ch: usize,
    formant: f32,
    states: &mut [FormantLpState],
    sample_rate: f64,
) {
    if formant.abs() < 1e-4 || ch == 0 {
        return;
    }
    let fc = 2000.0;
    let alpha = (1.0 - (-2.0 * std::f64::consts::PI * fc / sample_rate).exp()) as f32;
    let frames = samples.len() / ch;
    for f in 0..frames {
        for c in 0..ch {
            let i = f * ch + c;
            let x = samples[i];
            let st = &mut states[c];
            st.lp += alpha * (x - st.lp);
            samples[i] = if formant >= 0.0 {
                x + formant * (x - st.lp)
            } else {
                let w = -formant;
                (1.0 - w) * x + w * st.lp
            };
        }
    }
}

/// Get available presets
#[command]
fn get_voice_changer_presets() -> Vec<VoiceChangerPreset> {
    get_presets()
}

/// Get current voice changer parameters
#[command]
fn get_voice_changer_params(state: State<'_, VoiceChangerState>) -> Result<VoiceChangerParams, String> {
    Ok(VoiceChangerParams {
        enabled: *state.enabled.lock().map_err(|_| "Lock failed")?,
        pitch_semitones: *state.pitch_semitones.lock().map_err(|_| "Lock failed")?,
        formant_shift: *state.formant_shift.lock().map_err(|_| "Lock failed")?,
        vocoder_window_ms: *state.vocoder_window_ms.lock().map_err(|_| "Lock failed")?,
        vocoder_oversample: *state.vocoder_oversample.lock().map_err(|_| "Lock failed")?,
        output_device: state
            .output_device
            .lock()
            .map_err(|_| "Lock failed")?
            .clone(),
    })
}

/// Set voice changer parameters
#[command]
fn set_voice_changer_params(state: State<'_, VoiceChangerState>, params: VoiceChangerParams) -> Result<(), String> {
    *state.enabled.lock().map_err(|_| "Lock failed")? = params.enabled;
    *state.pitch_semitones.lock().map_err(|_| "Lock failed")? = params.pitch_semitones.clamp(-12.0, 12.0);
    *state.formant_shift.lock().map_err(|_| "Lock failed")? = params.formant_shift.clamp(-1.0, 1.0);
    *state.vocoder_window_ms.lock().map_err(|_| "Lock failed")? = params.vocoder_window_ms.clamp(30, 60);
    *state.vocoder_oversample.lock().map_err(|_| "Lock failed")? = match params.vocoder_oversample {
        4 | 8 | 16 | 32 => params.vocoder_oversample,
        _ => DEFAULT_VOCODER_OVERSAMPLE,
    };
    *state.output_device.lock().map_err(|_| "Lock failed")? = params.output_device;
    Ok(())
}

/// Set enabled state
#[command]
fn set_voice_changer_enabled(state: State<'_, VoiceChangerState>, enabled: bool) -> Result<(), String> {
    *state.enabled.lock().map_err(|_| "Lock failed")? = enabled;
    Ok(())
}

/// Apply a preset
#[command]
fn apply_voice_changer_preset(state: State<'_, VoiceChangerState>, preset_id: String) -> Result<(), String> {
    let presets = get_presets();
    let preset = presets
        .iter()
        .find(|p| p.id == preset_id)
        .ok_or_else(|| format!("Preset '{}' not found", preset_id))?;

    *state.pitch_semitones.lock().map_err(|_| "Lock failed")? = preset.pitch;
    *state.formant_shift.lock().map_err(|_| "Lock failed")? = preset.formant;
    Ok(())
}

/// Start the voice changer (capture -> process -> playback loop)
#[command]
async fn start_voice_changer<R: Runtime>(app: AppHandle<R>, state: State<'_, VoiceChangerState>, input_device: Option<String>) -> Result<(), String> {
    if *state.is_running.lock().map_err(|_| "Lock failed")? {
        return Err("Voice changer already running".to_string());
    }

    if let Some(sender) = state.stop_sender.lock().unwrap().take() {
        let _ = sender.send(());
    }

    let (tx, rx) = channel();
    *state.stop_sender.lock().unwrap() = Some(tx);
    *state.is_running.lock().unwrap() = true;
    *state.enabled.lock().unwrap() = true;

    let pitch_semitones = state.pitch_semitones.clone();
    let formant_shift = state.formant_shift.clone();
    let vocoder_window_ms = state.vocoder_window_ms.clone();
    let vocoder_oversample = state.vocoder_oversample.clone();
    let is_running = state.is_running.clone();
    let output_device_name = state
        .output_device
        .lock()
        .map_err(|_| "Lock failed")?
        .clone();
    let input_device_name = input_device.clone();
    let app_clone = app.clone();

    thread::spawn(move || {
        let result = run_voice_changer_loop(
            rx,
            pitch_semitones,
            formant_shift,
            vocoder_window_ms,
            vocoder_oversample,
            is_running.clone(),
            input_device_name,
            output_device_name,
        );

        if let Err(e) = result {
            eprintln!("[VoiceChanger] Error: {}", e);
        }

        *is_running.lock().unwrap() = false;
        let _ = app_clone.emit("voice_changer:stopped", ());
    });

    let _ = app.emit("voice_changer:started", ());
    Ok(())
}

fn run_voice_changer_loop(
    rx: std::sync::mpsc::Receiver<()>,
    pitch_semitones: Arc<Mutex<f32>>,
    formant_shift: Arc<Mutex<f32>>,
    vocoder_window_ms: Arc<Mutex<u32>>,
    vocoder_oversample: Arc<Mutex<u32>>,
    is_running: Arc<Mutex<bool>>,
    input_device_name: Option<String>,
    output_device_name: String,
) -> Result<(), String> {
    let host = cpal::default_host();

    let input = if let Some(ref name) = input_device_name {
        if !name.is_empty() {
            host.input_devices()
                .map_err(|e| e.to_string())?
                .find(|d| d.name().unwrap_or_default() == *name)
                .ok_or_else(|| format!("Input device '{}' not found", name))?
        } else {
            host.default_input_device().ok_or("No default input device")?
        }
    } else {
        host.default_input_device().ok_or("No default input device")?
    };

    let output = if output_device_name.is_empty() || output_device_name == "default" {
        host.default_output_device().ok_or("No default output device")?
    } else {
        host.output_devices()
            .map_err(|e| e.to_string())?
            .find(|d| d.name().unwrap_or_default() == output_device_name)
            .ok_or_else(|| format!("Output device '{}' not found", output_device_name))?
    };

    let input_config = input.default_input_config().map_err(|e| e.to_string())?;
    let output_config = output.default_output_config().map_err(|e| e.to_string())?;

    let r_in = input_config.sample_rate().0;
    let r_out = output_config.sample_rate().0;
    let ch_in = input_config.channels() as usize;
    let ch_out = output_config.channels() as usize;
    let ch_work = ch_in.max(ch_out).max(1);

    #[cfg(debug_assertions)]
    println!(
        "[VoiceChanger] Input:  {} Hz, {} ch | Output: {} Hz, {} ch | work {} ch",
        r_in, ch_in, r_out, ch_out, ch_work
    );

    if !matches!(output_config.sample_format(), cpal::SampleFormat::F32) {
        return Err(format!(
            "Output sample format {:?} not supported (need F32)",
            output_config.sample_format()
        ));
    }

    let buffer: Arc<Mutex<Vec<f32>>> = Arc::new(Mutex::new(Vec::new()));
    let buffer_clone = buffer.clone();
    let is_running_input = is_running.clone();

    let prefill_samples = ((r_out as f64) * PREFILL_MS / 1000.0).ceil() as usize * ch_out;
    let max_queue_samples = ((r_out as f64) * MAX_QUEUE_MS / 1000.0).ceil() as usize * ch_out;
    let playback: Arc<Mutex<OutputPlayback>> = Arc::new(Mutex::new(OutputPlayback::new(
        prefill_samples,
        max_queue_samples,
        ch_out,
    )));
    let playback_clone = playback.clone();

    let input_stream = match input_config.sample_format() {
        cpal::SampleFormat::F32 => input.build_input_stream(
            &input_config.into(),
            move |data: &[f32], _: &cpal::InputCallbackInfo| {
                if !*is_running_input.lock().unwrap() {
                    return;
                }
                let mut buf = buffer_clone.lock().unwrap();
                buf.extend_from_slice(data);
            },
            |e| eprintln!("[VoiceChanger] Input error: {}", e),
            None,
        ),
        cpal::SampleFormat::I16 => {
            let is_running_i16 = is_running.clone();
            let buffer_i16 = buffer.clone();
            input.build_input_stream(
                &input_config.into(),
                move |data: &[i16], _: &cpal::InputCallbackInfo| {
                    if !*is_running_i16.lock().unwrap() {
                        return;
                    }
                    let samples: Vec<f32> = data.iter().map(|&s| s as f32 / 32768.0).collect();
                    let mut buf = buffer_i16.lock().unwrap();
                    buf.extend_from_slice(&samples);
                },
                |e| eprintln!("[VoiceChanger] Input error: {}", e),
                None,
            )
        }
        _ => return Err("Unsupported input sample format".to_string()),
    }
    .map_err(|e| e.to_string())?;

    let output_stream = output
        .build_output_stream(
            &output_config.into(),
            move |data: &mut [f32], _: &cpal::OutputCallbackInfo| {
                let mut pb = playback_clone.lock().unwrap();
                pb.fill_callback(data);
            },
            |e| eprintln!("[VoiceChanger] Output error: {}", e),
            None,
        )
        .map_err(|e| e.to_string())?;

    input_stream.play().map_err(|e| e.to_string())?;
    output_stream.play().map_err(|e| e.to_string())?;

    #[cfg(debug_assertions)]
    println!(
        "[VoiceChanger] Phase vocoder pitch + device resample {:.4} (prefill ~{} samples, max queue ~{})",
        device_resample_ratio(r_in, r_out),
        prefill_samples,
        max_queue_samples
    );

    let initial_window = clamp_vocoder_window_ms(*vocoder_window_ms.lock().unwrap());
    let mut last_window_ms = initial_window;
    let mut phase_shifters: Vec<PitchShifter> = (0..ch_work)
        .map(|_| PitchShifter::new(initial_window, r_in as usize))
        .collect();

    let ratio_device = device_resample_ratio(r_in, r_out);
    let mut resampler =
        make_device_resampler(ratio_device, ch_work).map_err(|e| format!("Voice changer resampler: {}", e))?;

    let mut pitched_planes: Vec<Vec<f32>> = (0..ch_work)
        .map(|_| vec![0.0f32; CHUNK_FRAMES])
        .collect();

    let mut formant_states: Vec<FormantLpState> = vec![FormantLpState::default(); ch_work];

    loop {
        if rx.try_recv().is_ok() || !*is_running.lock().unwrap() {
            break;
        }

        let pitch = *pitch_semitones.lock().unwrap();
        let formant = *formant_shift.lock().unwrap();
        let window_ms = clamp_vocoder_window_ms(*vocoder_window_ms.lock().unwrap());
        let oversample = clamp_vocoder_oversample(*vocoder_oversample.lock().unwrap());

        if window_ms != last_window_ms {
            last_window_ms = window_ms;
            phase_shifters = (0..ch_work)
                .map(|_| PitchShifter::new(window_ms, r_in as usize))
                .collect();
        }

        let need = CHUNK_FRAMES * ch_in;
        let chunk: Vec<f32> = {
            let mut buf = buffer.lock().unwrap();
            if buf.len() < need {
                drop(buf);
                thread::sleep(std::time::Duration::from_millis(2));
                continue;
            }
            buf.drain(..need).collect()
        };

        let planes = interleaved_to_planar(&chunk, ch_in, ch_work);
        apply_phase_vocoder_channels(
            &mut phase_shifters,
            oversample,
            pitch,
            &planes,
            &mut pitched_planes,
        );

        let mut processed = match resampler.process(&pitched_planes, None) {
            Ok(out_planes) => planar_to_interleaved(&out_planes, ch_work, ch_out),
            Err(e) => {
                eprintln!("[VoiceChanger] Resample error: {}", e);
                planar_to_interleaved(&pitched_planes, ch_work, ch_out)
            }
        };

        apply_formant_interleaved(
            &mut processed,
            ch_out,
            formant,
            &mut formant_states,
            r_out as f64,
        );

        {
            let mut pb = playback.lock().unwrap();
            pb.push_interleaved(&processed);
        }
    }

    #[cfg(debug_assertions)]
    println!("[VoiceChanger] Stopped audio streams");
    Ok(())
}

/// Stop the voice changer
#[command]
fn stop_voice_changer<R: Runtime>(app: AppHandle<R>, state: State<'_, VoiceChangerState>) -> Result<(), String> {
    *state.enabled.lock().unwrap() = false;
    *state.is_running.lock().unwrap() = false;
    if let Some(sender) = state.stop_sender.lock().unwrap().take() {
        let _ = sender.send(());
    }
    let _ = app.emit("voice_changer:stopped", ());
    Ok(())
}

/// Check if voice changer is running
#[command]
fn is_voice_changer_running(state: State<'_, VoiceChangerState>) -> bool {
    *state.is_running.lock().unwrap()
}

pub fn init<R: Runtime>() -> TauriPlugin<R> {
    Builder::new("voice-changer")
        .setup(|app, _api| {
            app.manage(VoiceChangerState::new());
            Ok(())
        })
        .invoke_handler(tauri::generate_handler![
            get_voice_changer_presets,
            get_voice_changer_params,
            set_voice_changer_params,
            set_voice_changer_enabled,
            apply_voice_changer_preset,
            start_voice_changer,
            stop_voice_changer,
            is_voice_changer_running
        ])
        .build()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn device_resample_ratio_standard() {
        let r = device_resample_ratio(44100, 48000);
        assert!((r - 48000.0 / 44100.0).abs() < 1e-6);
    }

    #[test]
    fn phase_vocoder_preserves_buffer_length() {
        let mut shifter = PitchShifter::new(DEFAULT_VOCODER_WINDOW_MS as usize, 48000);
        let len = CHUNK_FRAMES;
        let in_b: Vec<f32> = (0..len)
            .map(|i| (i as f32 * 0.01).sin() * 0.1)
            .collect();
        let mut out_b = vec![0.0f32; len];
        shifter.shift_pitch(DEFAULT_VOCODER_OVERSAMPLE as usize, 5.0f32, &in_b, &mut out_b);
        assert_eq!(in_b.len(), out_b.len());
    }

    #[test]
    fn apply_phase_vocoder_stereo_smoke() {
        let mut shifters: Vec<PitchShifter> =
            (0..2).map(|_| PitchShifter::new(DEFAULT_VOCODER_WINDOW_MS as usize, 48000)).collect();
        let planes_in = vec![vec![0.5f32; 64], vec![-0.25f32; 64]];
        let mut planes_out = vec![vec![0.0f32; 64], vec![0.0f32; 64]];
        apply_phase_vocoder_channels(
            &mut shifters,
            DEFAULT_VOCODER_OVERSAMPLE as usize,
            0.0,
            &planes_in,
            &mut planes_out,
        );
        assert_eq!(planes_out[0].len(), planes_in[0].len());
        assert_eq!(planes_out[1].len(), planes_in[1].len());
    }
}
