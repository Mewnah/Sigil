//! Output devices of the TTS sink: one persistent cpal stream per device, fed through a lock-free
//! ring of slot-tagged samples.
//!
//! - The sink's scheduler thread writes each utterance's audio into the ring tagged with the
//!   utterance's *slot* (0..64). Cancelling an utterance (stop / skip) sets its slot bit in
//!   [`PlayoutShared::cancelled`]; the output callback then drops that utterance's frames still in
//!   the ring, so stop and skip are immediate even though the ring runs ~200 ms ahead, and a later
//!   utterance already written behind it plays on untouched.
//! - Continuity breaks (a discard, an underrun, the first audio after silence) are click-free: the
//!   callback decays the last output value to silence while starved and ramps from it into the new
//!   audio over [`RECOVER_MS`].
//! - The callback never allocates, locks, logs or does IPC; it registers MMCSS "Pro Audio" once per
//!   callback thread (`rt_thread`). Streams run on their own owner thread (cpal streams are not
//!   `Send`); dropping the [`OpenOutput::handle`] closes the stream.
//! - Any cpal sample format with a float conversion is supported (WASAPI shared mode is f32).

use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::sync::mpsc;
use std::sync::{Arc, Mutex};
use std::thread;
use std::time::Duration;

use cpal::traits::{DeviceTrait, HostTrait, StreamTrait};
use cpal::{FromSample, SampleFormat, SizedSample};
use rtrb::RingBuffer;

use crate::services::rt_thread;

/// Ring capacity (seconds of device audio).
pub(crate) const RING_SECONDS: f32 = 1.0;
/// Ramp from the held value into new audio after a continuity break.
pub(crate) const RECOVER_MS: f32 = 2.0;
/// Time constant of the decay to silence while starved.
const DECAY_MS: f32 = 1.5;
/// How long opening a device may take.
const OPEN_TIMEOUT: Duration = Duration::from_secs(5);
/// Scratch for non-f32 devices (frames); larger callbacks are processed in pieces.
const SCRATCH_FRAMES: usize = 4_096;
/// Slots (utterances that can have audio in one output's ring at the same time).
pub(crate) const SLOTS: u32 = 64;

/// One ring entry: a sample and the slot of the utterance it belongs to.
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub(crate) struct Tagged {
    pub v: f32,
    pub slot: u8,
}

/// State shared between the scheduler and one output's callback.
#[derive(Debug, Default)]
pub(crate) struct PlayoutShared {
    /// Frames the callback took from the ring (played or discarded).
    pub read_frames: AtomicU64,
    /// Bit `n` set: drop slot `n`'s frames instead of playing them.
    pub cancelled: AtomicU64,
    /// The stream reported an error (device lost); set by the error callback.
    pub failed: AtomicBool,
    /// Message of that error (written by the error callback, never the data callback).
    pub error: Mutex<Option<String>>,
    /// Callbacks that ran dry while audio was playing.
    pub underruns: AtomicU64,
}

impl PlayoutShared {
    pub(crate) fn fail(&self, message: String) {
        if let Ok(mut e) = self.error.lock() {
            e.get_or_insert(message);
        }
        self.failed.store(true, Ordering::Release);
    }

    pub(crate) fn error_message(&self) -> String {
        self.error
            .lock()
            .ok()
            .and_then(|e| e.clone())
            .unwrap_or_else(|| "the output device stopped".to_string())
    }
}

/// Callback side of one output: pops tagged frames, drops cancelled slots, keeps joins click-free.
pub(crate) struct Playout {
    consumer: rtrb::Consumer<Tagged>,
    ch: usize,
    shared: Arc<PlayoutShared>,
    /// Last output value per channel.
    hold: Vec<f32>,
    /// Value per channel a recovery crossfade starts from (the output when the break happened).
    ramp_from: Vec<f32>,
    recover_frames: usize,
    recover_left: usize,
    decay: f32,
    /// Continuity broken since the last played frame: the next one starts a ramp.
    broken: bool,
    /// Audio played in the previous callback (an empty callback after audio is an underrun).
    was_playing: bool,
}

impl Playout {
    pub(crate) fn new(consumer: rtrb::Consumer<Tagged>, ch: usize, rate: u32, shared: Arc<PlayoutShared>) -> Self {
        let rate = rate.max(1) as f32;
        Self {
            consumer,
            ch: ch.max(1),
            shared,
            hold: vec![0.0; ch.max(1)],
            ramp_from: vec![0.0; ch.max(1)],
            recover_frames: ((RECOVER_MS / 1000.0 * rate) as usize).max(1),
            recover_left: 0,
            decay: (-1.0 / (DECAY_MS / 1000.0 * rate)).exp(),
            broken: true,
            was_playing: false,
        }
    }

    /// Fill one device buffer (interleaved, whole frames).
    pub(crate) fn fill(&mut self, data: &mut [f32]) {
        let ch = self.ch;
        let frames = data.len() / ch;
        let mask = self.shared.cancelled.load(Ordering::Acquire);
        let avail = self.consumer.slots() / ch;
        let mut out_f = 0usize;
        let mut consumed = 0usize;
        if avail > 0 {
            if let Ok(chunk) = self.consumer.read_chunk(avail * ch) {
                let (a, b) = chunk.as_slices();
                let at = |k: usize| if k < a.len() { a[k] } else { b[k - a.len()] };
                while consumed < avail && out_f < frames {
                    let base = consumed * ch;
                    consumed += 1;
                    let slot = at(base).slot;
                    if mask & (1u64 << (slot as u32 % SLOTS)) != 0 {
                        self.broken = true;
                        continue;
                    }
                    if self.broken {
                        // Crossfade linearly from what was last output into the new audio.
                        self.broken = false;
                        self.recover_left = self.recover_frames;
                        self.ramp_from.copy_from_slice(&self.hold);
                    }
                    let t = if self.recover_left > 0 {
                        let done = self.recover_frames - self.recover_left;
                        self.recover_left -= 1;
                        (done + 1) as f32 / self.recover_frames as f32
                    } else {
                        1.0
                    };
                    for c in 0..ch {
                        let v = at(base + c).v;
                        let out = if t < 1.0 { self.ramp_from[c] + (v - self.ramp_from[c]) * t } else { v };
                        data[out_f * ch + c] = out;
                        self.hold[c] = out;
                    }
                    out_f += 1;
                }
                chunk.commit(consumed * ch);
            }
        }
        if consumed > 0 {
            self.shared.read_frames.fetch_add(consumed as u64, Ordering::Release);
        }
        if out_f < frames {
            if self.was_playing && out_f > 0 {
                self.shared.underruns.fetch_add(1, Ordering::Relaxed);
            }
            self.broken = true;
            for f in out_f..frames {
                for c in 0..ch {
                    self.hold[c] *= self.decay;
                    if self.hold[c].abs() < 1e-6 {
                        self.hold[c] = 0.0;
                    }
                    data[f * ch + c] = self.hold[c];
                }
            }
        }
        self.was_playing = out_f == frames;
    }
}

/// An opened output: the scheduler's ends of its ring.
pub(crate) struct OpenOutput {
    /// Device name it resolved to.
    pub name: String,
    pub rate: u32,
    pub channels: usize,
    pub producer: rtrb::Producer<Tagged>,
    pub shared: Arc<PlayoutShared>,
    /// Keeps the stream open; dropping it closes the stream.
    pub handle: Box<dyn Send>,
}

/// Opens output devices (cpal in the app; a fake in tests).
pub(crate) trait OutputBackend: Send + Sync {
    /// Resolve `device` (`""` = the system default) and open a stream on it. Blocking (bounded).
    fn open(&self, device: &str) -> Result<OpenOutput, String>;
    /// Name the system default output resolves to now (to follow a changed default).
    fn default_name(&self) -> Option<String>;
}

/// A new ring for `rate` / `channels` (capacity: whole frames, [`RING_SECONDS`]).
pub(crate) fn ring(rate: u32, channels: usize) -> (rtrb::Producer<Tagged>, rtrb::Consumer<Tagged>) {
    let frames = ((rate.max(1) as f32 * RING_SECONDS) as usize).max(1_024);
    RingBuffer::new(frames * channels.max(1))
}

/// Empty / `"default"` (any case) = the system default output.
pub(crate) fn is_default_device(name: &str) -> bool {
    let t = name.trim();
    t.is_empty() || t.eq_ignore_ascii_case("default")
}

/// `name` lower-cased and trimmed, without Windows' endpoint numbering (`N- ` at the start of the
/// name or right after an opening parenthesis). Same rule as the voice changer's device matching.
fn device_family(name: &str) -> String {
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

/// Index of the device a saved name means among `names`: the exact (trimmed) name, else the same
/// name ignoring case, else the one device of the same family (Windows' "2- " renumbering), else
/// the one device the saved name is a prefix of. The last rule adopts names saved from the old
/// SAPI / WinMM device list, which truncates names to 31 characters
/// ("CABLE Input (VB-Audio Virtual C" for "CABLE Input (VB-Audio Virtual Cable)").
pub(crate) fn match_output_device(saved: &str, names: &[String]) -> Option<usize> {
    let saved = saved.trim();
    if saved.is_empty() {
        return None;
    }
    if let Some(i) = names.iter().position(|n| n.trim() == saved) {
        return Some(i);
    }
    let lower = saved.to_lowercase();
    let unique = |pred: &dyn Fn(&str) -> bool| {
        let mut hits = names.iter().enumerate().filter(|(_, n)| pred(n));
        match (hits.next(), hits.next()) {
            (Some((i, _)), None) => Some(i),
            _ => None,
        }
    };
    if let Some(i) = unique(&|n: &str| n.trim().to_lowercase() == lower) {
        return Some(i);
    }
    let family = device_family(saved);
    if let Some(i) = unique(&|n: &str| device_family(n) == family) {
        return Some(i);
    }
    if saved.chars().count() >= 12 {
        if let Some(i) = unique(&|n: &str| n.trim().to_lowercase().starts_with(&lower)) {
            return Some(i);
        }
        return unique(&|n: &str| device_family(n).starts_with(&family));
    }
    None
}

/// The app's output backend (cpal default host: WASAPI shared mode on Windows).
pub(crate) struct CpalBackend;

fn resolve(host: &cpal::Host, wanted: &str) -> Result<(cpal::Device, String), String> {
    if is_default_device(wanted) {
        let device = host
            .default_output_device()
            .ok_or_else(|| "There is no default output device.".to_string())?;
        let name = device.name().unwrap_or_else(|_| "default".to_string());
        return Ok((device, name.trim().to_string()));
    }
    let devices: Vec<cpal::Device> = host.output_devices().map_err(|e| format!("Listing output devices failed: {e}"))?.collect();
    let names: Vec<String> = devices.iter().map(|d| d.name().unwrap_or_default()).collect();
    match match_output_device(wanted, &names) {
        Some(i) => {
            let name = names[i].trim().to_string();
            Ok((devices.into_iter().nth(i).expect("index"), name))
        }
        None => Err(format!("Output device '{}' not found", wanted.trim())),
    }
}

fn build_stream(device: &cpal::Device, config: &cpal::SupportedStreamConfig, mut playout: Playout, shared: Arc<PlayoutShared>) -> Result<cpal::Stream, String> {
    let sc = config.config();
    let on_error = move |e: cpal::StreamError| shared.fail(e.to_string());
    let built = match config.sample_format() {
        SampleFormat::F32 => device.build_output_stream(
            &sc,
            move |data: &mut [f32], _: &cpal::OutputCallbackInfo| {
                rt_thread::prepare_audio_callback_thread_once();
                playout.fill(data);
            },
            on_error,
            None,
        ),
        SampleFormat::F64 => build_converting::<f64>(device, &sc, playout, on_error),
        SampleFormat::I16 => build_converting::<i16>(device, &sc, playout, on_error),
        SampleFormat::I32 => build_converting::<i32>(device, &sc, playout, on_error),
        SampleFormat::U16 => build_converting::<u16>(device, &sc, playout, on_error),
        SampleFormat::I8 => build_converting::<i8>(device, &sc, playout, on_error),
        SampleFormat::U8 => build_converting::<u8>(device, &sc, playout, on_error),
        other => return Err(format!("Unsupported output sample format: {other:?}")),
    };
    built.map_err(|e| format!("Failed to open the output stream: {e}"))
}

fn build_converting<T>(
    device: &cpal::Device,
    sc: &cpal::StreamConfig,
    mut playout: Playout,
    on_error: impl FnMut(cpal::StreamError) + Send + 'static,
) -> Result<cpal::Stream, cpal::BuildStreamError>
where
    T: SizedSample + FromSample<f32> + 'static,
{
    let ch = usize::from(sc.channels.max(1));
    let mut scratch = vec![0.0f32; SCRATCH_FRAMES * ch];
    device.build_output_stream(
        sc,
        move |data: &mut [T], _: &cpal::OutputCallbackInfo| {
            rt_thread::prepare_audio_callback_thread_once();
            for piece in data.chunks_mut(scratch.len()) {
                let s = &mut scratch[..piece.len()];
                playout.fill(s);
                for (d, &v) in piece.iter_mut().zip(s.iter()) {
                    *d = <T as FromSample<f32>>::from_sample_(v);
                }
            }
        },
        on_error,
        None,
    )
}

/// Closes its stream's owner thread when dropped.
struct StreamHandle(#[allow(dead_code)] mpsc::Sender<()>);

impl OutputBackend for CpalBackend {
    fn open(&self, wanted: &str) -> Result<OpenOutput, String> {
        // Resolve avrt.dll here, off the audio thread.
        rt_thread::preload();
        type Opened = (String, u32, usize, rtrb::Producer<Tagged>, Arc<PlayoutShared>);
        let (ready_tx, ready_rx) = mpsc::channel::<Result<Opened, String>>();
        let (stop_tx, stop_rx) = mpsc::channel::<()>();
        let wanted = wanted.to_string();
        thread::Builder::new()
            .name("sigil-tts-output".into())
            .spawn(move || {
                let opened = (|| {
                    let host = cpal::default_host();
                    let (device, name) = resolve(&host, &wanted)?;
                    let config = device
                        .default_output_config()
                        .map_err(|e| format!("The output device '{name}' cannot be opened: {e}"))?;
                    let rate = config.sample_rate().0;
                    let ch = usize::from(config.channels().max(1));
                    let (producer, consumer) = ring(rate, ch);
                    let shared = Arc::new(PlayoutShared::default());
                    let playout = Playout::new(consumer, ch, rate, shared.clone());
                    let stream = build_stream(&device, &config, playout, shared.clone())?;
                    stream.play().map_err(|e| format!("Failed to start the output device '{name}': {e}"))?;
                    Ok((stream, (name, rate, ch, producer, shared)))
                })();
                match opened {
                    Ok((stream, out)) => {
                        if ready_tx.send(Ok(out)).is_ok() {
                            // Until the handle is dropped (or the open timed out).
                            let _ = stop_rx.recv();
                        }
                        drop(stream);
                    }
                    Err(e) => {
                        let _ = ready_tx.send(Err(e));
                    }
                }
            })
            .map_err(|e| format!("Failed to start the audio output thread: {e}"))?;
        match ready_rx.recv_timeout(OPEN_TIMEOUT) {
            Ok(Ok((name, rate, channels, producer, shared))) => {
                tracing::info!(target: "sigil::tts_sink", "output '{name}' opened: {rate} Hz, {channels} ch");
                Ok(OpenOutput {
                    name,
                    rate,
                    channels,
                    producer,
                    shared,
                    handle: Box::new(StreamHandle(stop_tx)),
                })
            }
            Ok(Err(e)) => Err(e),
            // Dropping `stop_tx` makes a late-starting stream close itself at once.
            Err(_) => Err("Timed out opening the output device".to_string()),
        }
    }

    fn default_name(&self) -> Option<String> {
        cpal::default_host()
            .default_output_device()
            .and_then(|d| d.name().ok())
            .map(|n| n.trim().to_string())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn push(p: &mut rtrb::Producer<Tagged>, frames: &[(f32, u8)], ch: usize) {
        for &(v, slot) in frames {
            for _ in 0..ch {
                p.push(Tagged { v, slot }).unwrap();
            }
        }
    }

    fn playout(ch: usize) -> (rtrb::Producer<Tagged>, Playout, Arc<PlayoutShared>) {
        let (p, c) = ring(48_000, ch);
        let shared = Arc::new(PlayoutShared::default());
        (p, Playout::new(c, ch, 48_000, shared.clone()), shared)
    }

    /// Audio plays in order; cancelled slots are dropped (counted as read) while the next
    /// utterance's frames written behind them still play; every join ramps without a click.
    #[test]
    fn cancelled_slots_are_skipped_and_joins_are_click_free() {
        let (mut p, mut out, shared) = playout(2);
        let a: Vec<(f32, u8)> = (0..500).map(|_| (0.5, 1)).collect();
        let b: Vec<(f32, u8)> = (0..500).map(|_| (-0.5, 2)).collect();
        push(&mut p, &a, 2);
        push(&mut p, &b, 2);
        let mut buf = vec![0.0f32; 200 * 2];
        out.fill(&mut buf);
        assert_eq!(shared.read_frames.load(Ordering::Acquire), 200);
        // Starts from silence with a short ramp, then the utterance level.
        assert!(buf[0].abs() < 0.05, "{}", buf[0]);
        assert_eq!(buf[398], 0.5);
        // Skip slot 1 mid-utterance: its remaining 300 frames are dropped, slot 2 follows.
        shared.cancelled.fetch_or(1 << 1, Ordering::Release);
        out.fill(&mut buf);
        assert_eq!(shared.read_frames.load(Ordering::Acquire), 200 + 300 + 200);
        let left: Vec<f32> = buf.iter().step_by(2).copied().collect();
        let max_step = left.windows(2).map(|w| (w[1] - w[0]).abs()).fold((0.5f32 - left[0]).abs(), f32::max);
        assert!(max_step < 0.02, "no click at the skip: step {max_step}");
        assert_eq!(*left.last().unwrap(), -0.5);
    }

    /// Starved: the last value decays to silence (no DC hold, no click), and the underrun is
    /// counted only when audio was playing.
    #[test]
    fn underrun_decays_to_silence() {
        let (mut p, mut out, shared) = playout(1);
        let mut buf = vec![0.0f32; 480];
        out.fill(&mut buf);
        assert!(buf.iter().all(|&s| s == 0.0), "idle output is silent");
        assert_eq!(shared.underruns.load(Ordering::Relaxed), 0);
        push(&mut p, &vec![(0.8, 0); 480 + 100], 1);
        out.fill(&mut buf);
        out.fill(&mut buf);
        assert_eq!(shared.underruns.load(Ordering::Relaxed), 1);
        assert!((buf[99] - 0.8).abs() < 1e-6);
        let tail = &buf[100..];
        assert!(tail.windows(2).all(|w| w[1] <= w[0]) && tail[0] < 0.8 && *tail.last().unwrap() < 0.01);
    }

    #[test]
    fn saved_device_names_are_matched_tolerantly() {
        let names: Vec<String> = [
            "Speakers (Realtek(R) Audio)",
            "CABLE Input (VB-Audio Virtual Cable)",
            "2- Headphones (USB Audio)",
            "Headset Earphone (HyperX Cloud)",
            "Headset Earphone (Other)",
        ]
        .iter()
        .map(|s| s.to_string())
        .collect();
        assert_eq!(match_output_device(" Speakers (Realtek(R) Audio) ", &names), Some(0));
        assert_eq!(match_output_device("speakers (realtek(r) audio)", &names), Some(0));
        // Renumbered by Windows.
        assert_eq!(match_output_device("Headphones (USB Audio)", &names), Some(2));
        // An old SAPI / WinMM name, truncated to 31 characters.
        assert_eq!(match_output_device("CABLE Input (VB-Audio Virtual C", &names), Some(1));
        // Ambiguous prefixes and short fragments never guess.
        assert_eq!(match_output_device("Headset Earphone (", &names), None);
        assert_eq!(match_output_device("CABLE", &names), None);
        assert_eq!(match_output_device("Missing device", &names), None);
        assert_eq!(match_output_device("", &names), None);
    }
}
