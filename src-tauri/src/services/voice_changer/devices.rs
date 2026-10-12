//! The run's devices: the capture (a consumer of the capture hub) and the output / monitor ports,
//! each following its device.
//!
//! # Finding a saved device
//! Devices are persisted by name (cpal's name is the id `list_*_devices` returns). A saved name
//! can go stale between sessions: the device is unplugged, or Windows renumbers the endpoint when
//! it moves to another USB port ("Microphone (2- USB Audio)" becomes "Microphone (3- USB Audio)").
//! Lookup is therefore: the exact name, else the one device of the same family (the name without
//! Windows' `N- ` endpoint numbering), else a clear "not connected" error. It never guesses
//! between several devices of one family and never falls back to the system default, which could
//! be a virtual cable carrying the voice changer's own output (a feedback loop) or open speakers.
//!
//! While a run follows a device that disappeared, the family rule is stricter
//! ([`match_runtime_device`]): only a device of the family that was **not** present when the loss
//! started can stand in for it (the same device, renumbered on another port). Otherwise unplugging a
//! USB headset next to a generic USB dongle of the same family would move the capture (or the
//! converted output) to the dongle. These lookups run on a helper thread ([`DeviceLookup`]):
//! enumerating endpoints is COM / RPC work the worker that produces the audio must never wait for.
//!
//! # Capture: the hub's conditioned copy ([`HubInput`], P3-E)
//! The voice changer no longer opens its own input stream: it attaches a device-rate consumer to
//! the capture hub (`audio::hub`, `ConsumerOptions::device_rate("vc")` in the `voice_changer` mute
//! group), so one physical microphone feeds STT (its own raw 16 kHz consumer) and the voice changer
//! from one stream. Everything the voice changer does to its audio (gain, denoiser, TTS mix) happens
//! on its own copy after the read: STT always gets the raw microphone.
//! - The hub reconnects a failed or unplugged device under the same source epoch and follows the
//!   system default; a restart or any other discontinuity (the hub's own drop-oldest, a mute edge)
//!   reaches the worker as a gap ([`CaptureSink::gap`]): the latency budget fades out into it and the
//!   next audio fades in, and the worker resets the model's stream state.
//! - While the capture delivers nothing (the device is gone and the hub is reconnecting, the
//!   `voice_changer` mute gate is closed, or nothing arrived for [`IDLE_FILL_AFTER`]), silence is
//!   synthesized on the QPC clock, so the output keeps playing (TTS routed into the voice changer
//!   included) instead of starving.
//! - A device that comes back in another format (another default device) is converted to the run's
//!   format (channel adapter plus the band-limited `PolyphaseResampler`), so everything downstream
//!   keeps its rate.
//! - A named microphone that comes back renumbered (Windows renames a USB endpoint moved to another
//!   port; the hub reconnects by exact name) is followed by [`InputFollow`]: while the hub has been
//!   reconnecting it for a while, the saved name is looked up again by the rules above and the
//!   capture moves to the device's new name ([`HubInput::reattach`]).
//!
//! # Output and monitor ([`OutputPort`], P3-E)
//! Each port owns its cpal stream, lock-free ring, `OutputPlayback` queue and, when the device's
//! format differs from the run's, a channel adapter and resampler. A port whose device disconnects
//! (a cpal `DeviceNotAvailable`, or no callback for [`OUTPUT_STALL_AFTER`]) keeps the run going and
//! retries every [`REOPEN_EVERY`] until the device is back (plan T7: resumes within 2 s); a port on
//! the system default follows it to a new default device.

use std::sync::atomic::{AtomicBool, AtomicU64, AtomicUsize, Ordering};
use std::sync::mpsc::{self, Receiver, TryRecvError};
use std::sync::Arc;
use std::time::{Duration, Instant};

use cpal::traits::{DeviceTrait, HostTrait, StreamTrait};
use cpal::{SampleFormat, StreamError};

use super::params::OutputCushion;
use super::playback::{adapt_channels_interleaved_into, OutputPlayback, RingPlayout, OUT_RING_MAX_SAMPLES, OUT_RING_MIN_SAMPLES};
use super::resample::PolyphaseResampler;
use super::state::{MmcssFlag, MmcssStatus};
use crate::services::audio::hub::{CaptureConsumer, CaptureHub, ConsumerOptions, MuteGroup, SourceSpec, StreamFormat};
use crate::services::rt_thread;

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

// Both family rules are shared with the capture hub (`audio::device_resolve`), which reopens a lost
// microphone by the runtime rule too, so a second identical microphone that was there all along
// never stands in for it. `match_runtime_device` gives the index of the device `saved` names while a
// run follows it after it disappeared: the exact name, else the one device of its family that was
// not present when it was lost (`present_at_loss`: the family's names then; `None`: this lookup
// takes that snapshot, so only the exact name counts).
pub(super) use crate::services::audio::device_resolve::{family_members, match_runtime_device, match_saved_device};

/// How a saved name is matched (see the module docs).
#[derive(Clone, Copy, Debug)]
pub(super) enum MatchRule<'a> {
    /// The start of a run: [`match_saved_device`].
    Start,
    /// A device the run follows after it disappeared: [`match_runtime_device`] with this snapshot.
    Runtime(Option<&'a [String]>),
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
    open_device(host, role, saved, MatchRule::Start).0
}

/// [`open_saved_device`] by `rule`, with the names of `saved`'s family that are present (empty for
/// the system default).
pub(super) fn open_device(host: &cpal::Host, role: DeviceRole, saved: &str, rule: MatchRule<'_>) -> (Result<OpenedDevice, String>, Vec<String>) {
    let input = role == DeviceRole::Input;
    if is_default_device(saved) {
        let device = if input {
            host.default_input_device()
        } else {
            host.default_output_device()
        };
        let Some(device) = device else {
            return (
                Err(format!(
                    "There is no default {} device. Connect one or choose a device in the voice changer settings.",
                    if input { "input" } else { "output" }
                )),
                Vec::new(),
            );
        };
        let name = device.name().unwrap_or_else(|_| "default".to_string());
        return (
            Ok(OpenedDevice {
                device,
                name,
                renamed_from: None,
            }),
            Vec::new(),
        );
    }
    let listed: Result<Vec<cpal::Device>, String> = if input {
        host.input_devices()
            .map(|d| d.collect())
            .map_err(|e| format!("Cannot list input devices: {e}"))
    } else {
        host.output_devices()
            .map(|d| d.collect())
            .map_err(|e| format!("Cannot list output devices: {e}"))
    };
    let listed = match listed {
        Ok(l) => l,
        Err(e) => return (Err(e), Vec::new()),
    };
    let names: Vec<String> = listed.iter().map(|d| d.name().unwrap_or_default()).collect();
    let family = family_members(saved, &names);
    let found = match rule {
        MatchRule::Start => match_saved_device(saved, &names),
        MatchRule::Runtime(present) => match_runtime_device(saved, &names, present),
    };
    let Some((i, renamed)) = found else {
        return (
            Err(format!(
                "The {} device '{}' is not connected. Connect it or choose another {} device in the voice changer settings.",
                role.label(),
                saved.trim(),
                role.label()
            )),
            family,
        );
    };
    let name = names[i].clone();
    let device = listed.into_iter().nth(i).expect("index from the same list");
    (
        Ok(OpenedDevice {
            device,
            name,
            renamed_from: renamed.then(|| saved.trim().to_string()),
        }),
        family,
    )
}

/// What a lookup of a device the run follows found.
pub(super) struct Lookup<T> {
    /// The device to move to, or why there is none (a user-facing sentence).
    pub(super) found: Result<T, String>,
    /// The names of the saved device's family that are present (the loss snapshot).
    pub(super) family: Vec<String>,
}

/// One lookup (on the helper thread): the saved name and the loss snapshot (`None`: take it).
pub(super) type LookupFn<T> = Arc<dyn Fn(&str, Option<&[String]>) -> Lookup<T> + Send + Sync>;

/// Request to the lookup helper: (generation, loss snapshot).
type LookupRequest = (u64, Option<Vec<String>>);
/// The worker's ends of the lookup helper: requests out, (generation, answer) back.
type LookupHelper<T> = (mpsc::Sender<LookupRequest>, Receiver<(u64, Lookup<T>)>);

/// The lookups of one device the run follows, on a helper thread (one at a time; the worker only
/// starts them and polls the answer), and its loss snapshot ([`match_runtime_device`]).
pub(super) struct DeviceLookup<T: Send + 'static> {
    saved: String,
    resolve: LookupFn<T>,
    helper: Option<LookupHelper<T>>,
    /// Answers of an earlier generation (before a [`Self::reset`]) are dropped.
    generation: u64,
    in_flight: bool,
    /// The family's names when the loss started (`None` until the first lookup of this loss).
    present_at_loss: Option<Vec<String>>,
}

impl<T: Send + 'static> DeviceLookup<T> {
    pub(super) fn new(saved: &str, resolve: LookupFn<T>) -> DeviceLookup<T> {
        DeviceLookup {
            saved: saved.trim().to_string(),
            resolve,
            helper: None,
            generation: 0,
            in_flight: false,
            present_at_loss: None,
        }
    }

    /// A lookup is running.
    pub(super) fn busy(&self) -> bool {
        self.in_flight
    }

    /// Start a lookup (nothing while one runs). The first one of a loss takes its snapshot.
    pub(super) fn start(&mut self) {
        if self.in_flight {
            return;
        }
        for _ in 0..2 {
            if self.helper.is_none() {
                let (req_tx, req_rx) = mpsc::channel::<LookupRequest>();
                let (res_tx, res_rx) = mpsc::channel();
                let (resolve, saved) = (self.resolve.clone(), self.saved.clone());
                let spawned = std::thread::Builder::new()
                    .name("sigil-vc-device-lookup".into())
                    .spawn(move || {
                        while let Ok((generation, present)) = req_rx.recv() {
                            let lookup = resolve(&saved, present.as_deref());
                            if res_tx.send((generation, lookup)).is_err() {
                                break;
                            }
                        }
                    });
                if spawned.is_err() {
                    return;
                }
                self.helper = Some((req_tx, res_rx));
            }
            let request = (self.generation, self.present_at_loss.clone());
            match self.helper.as_ref().map(|(tx, _)| tx.send(request)) {
                Some(Ok(())) => {
                    self.in_flight = true;
                    return;
                }
                // The helper died (a panic in the lookup): start a new one.
                _ => self.helper = None,
            }
        }
    }

    /// The answer of the lookup that ran, once it is there (the device, or why there is none).
    pub(super) fn take(&mut self) -> Option<Result<T, String>> {
        loop {
            let (_, rx) = self.helper.as_ref()?;
            match rx.try_recv() {
                Ok((generation, lookup)) => {
                    if generation != self.generation {
                        continue;
                    }
                    self.in_flight = false;
                    if self.present_at_loss.is_none() {
                        self.present_at_loss = Some(lookup.family);
                    }
                    return Some(lookup.found);
                }
                Err(TryRecvError::Empty) => return None,
                Err(TryRecvError::Disconnected) => {
                    self.helper = None;
                    self.in_flight = false;
                    return None;
                }
            }
        }
    }

    /// A loss ends (or a new one starts): the snapshot is forgotten and the answer of a lookup still
    /// running is dropped.
    pub(super) fn reset(&mut self) {
        self.generation += 1;
        self.in_flight = false;
        self.present_at_loss = None;
    }
}

// ---------------------------------------------------------------------------------------------
// Capture: the hub consumer
// ---------------------------------------------------------------------------------------------

/// Nothing captured for this long (while the device is not known to be gone or muted): fill silence.
pub(super) const IDLE_FILL_AFTER: Duration = Duration::from_millis(250);
/// How far behind the hub lets the voice changer fall before it drops the oldest audio itself. Above
/// the largest latency budget (2 s), so the budget, with its crossfaded splice, always acts first.
pub(super) const HUB_MAX_BACKLOG_MS: u32 = 3_000;

/// What the voice changer captures: the hub source and the device format at start (the run format).
#[derive(Clone, Debug, PartialEq)]
pub(super) struct InputTarget {
    pub(super) spec: SourceSpec,
    /// The device's current name (found by the saved-name rules above).
    pub(super) name: String,
    pub(super) renamed_from: Option<String>,
    pub(super) rate: u32,
    pub(super) ch: usize,
}

/// Resolve the saved input device to a hub source and its format. The system default stays a
/// default source (the hub follows default changes); a named device is pinned by its current name.
pub(super) fn resolve_input(host: &cpal::Host, saved: &str) -> Result<InputTarget, String> {
    input_target(open_saved_device(host, DeviceRole::Input, saved)?, saved)
}

/// [`resolve_input`] for a device the run follows ([`MatchRule::Runtime`]).
pub(super) fn resolve_input_runtime(host: &cpal::Host, saved: &str, present_at_loss: Option<&[String]>) -> Lookup<InputTarget> {
    let (opened, family) = open_device(host, DeviceRole::Input, saved, MatchRule::Runtime(present_at_loss));
    Lookup {
        found: opened.and_then(|o| input_target(o, saved)),
        family,
    }
}

fn input_target(opened: OpenedDevice, saved: &str) -> Result<InputTarget, String> {
    let config = opened
        .device
        .default_input_config()
        .map_err(|e| format!("The input device '{}' cannot be opened: {e}", opened.name))?;
    let spec = if is_default_device(saved) {
        SourceSpec::DefaultInput
    } else {
        SourceSpec::Input(opened.name.trim().to_string())
    };
    Ok(InputTarget {
        spec,
        name: opened.name,
        renamed_from: opened.renamed_from,
        rate: config.sample_rate().0,
        ch: usize::from(config.channels()).max(1),
    })
}

/// Where the capture puts what it reads (the worker's latency-budget backlog).
pub(super) trait CaptureSink {
    /// Whole frames in the run format.
    fn audio(&mut self, frames: &[f32]);
    /// The audio is discontinuous here; `lost_frames` were lost at the gap (0 = unknown).
    fn gap(&mut self, lost_frames: u64);
}

/// What one [`HubInput::pump`] delivered.
#[derive(Clone, Debug, Default, PartialEq)]
pub(super) struct Intake {
    /// Captured frames (run format).
    pub(super) frames: usize,
    /// Synthesized silent frames (idle fill).
    pub(super) filled: usize,
    /// Discontinuities reported to the sink.
    pub(super) gaps: u32,
    /// A discontinuity was a stream restart or a move to another device.
    pub(super) restarted: bool,
    /// Frames the hub dropped because the voice changer fell too far behind.
    pub(super) hub_dropped: u64,
    /// The capture ended for good (unsupervised legacy capture only): why.
    pub(super) ended: Option<String>,
}

/// Device format → run format.
struct FormatAdapter {
    from: StreamFormat,
    chans: Vec<f32>,
    resampler: Option<PolyphaseResampler>,
}

impl FormatAdapter {
    fn new(from: StreamFormat, rate: u32, ch: usize) -> FormatAdapter {
        FormatAdapter {
            from,
            chans: Vec::new(),
            resampler: (from.sample_rate != rate).then(|| PolyphaseResampler::new(from.sample_rate, rate, ch)),
        }
    }

    fn convert(&mut self, input: &[f32], ch: usize, out: &mut Vec<f32>) {
        adapt_channels_interleaved_into(input, usize::from(self.from.channels.max(1)), ch, &mut self.chans);
        out.clear();
        match self.resampler.as_mut() {
            Some(r) => r.process(&self.chans, out),
            None => out.extend_from_slice(&self.chans),
        }
    }
}

/// Idle fill in progress: silence synthesized since `origin_ns` (QPC).
struct Fill {
    origin_ns: i64,
    frames: u64,
}

/// The voice changer's capture: a device-rate consumer of the capture hub (see the module docs).
pub(super) struct HubInput {
    consumer: CaptureConsumer,
    muted: Box<dyn Fn() -> bool + Send>,
    rate: u32,
    ch: usize,
    read_buf: Vec<f32>,
    adapter: Option<FormatAdapter>,
    adapted: Vec<f32>,
    zeros: Vec<f32>,
    /// Local QPC time of the last read that delivered audio.
    last_audio_ns: i64,
    fill: Option<Fill>,
    /// QPC capture time just past the newest captured frame (the hub's block stamps).
    newest_capture_ns: Option<i64>,
    /// Interleaved samples of the latest read (the output ring must bridge one capture period).
    burst: Arc<AtomicUsize>,
}

impl HubInput {
    /// Attach to `hub` for `target` (blocking up to the hub's attach timeout: call it from the
    /// worker thread). `muted`: whether the `voice_changer` mute gate is closed.
    pub(super) fn attach(hub: &CaptureHub, target: &InputTarget, muted: Box<dyn Fn() -> bool + Send>) -> Result<HubInput, String> {
        let opts = ConsumerOptions::device_rate("vc")
            .with_mute_group(Some(MuteGroup::VoiceChanger))
            .with_max_backlog_ms(HUB_MAX_BACKLOG_MS);
        let consumer = hub.attach(target.spec.clone(), opts)?;
        let block = (target.rate as usize / 10).max(1) * target.ch; // 100 ms
        Ok(HubInput {
            consumer,
            muted,
            rate: target.rate.max(1),
            ch: target.ch.max(1),
            read_buf: Vec::with_capacity(block * 5),
            adapter: None,
            adapted: Vec::with_capacity(block * 5),
            zeros: vec![0.0; block],
            last_audio_ns: crate::services::audio::hub::now_qpc_ns(),
            fill: None,
            newest_capture_ns: None,
            burst: Arc::new(AtomicUsize::new(0)),
        })
    }

    /// Hub source key (`default`, a device name).
    pub(super) fn source(&self) -> &str {
        self.consumer.source()
    }

    /// The capture callback length the output rings bridge (interleaved samples, run format).
    pub(super) fn burst(&self) -> Arc<AtomicUsize> {
        self.burst.clone()
    }

    /// QPC capture time just past the newest real frame read so far.
    pub(super) fn newest_capture_ns(&self) -> Option<i64> {
        self.newest_capture_ns
    }

    /// Whether silence is being synthesized (no real capture).
    pub(super) fn filling(&self) -> bool {
        self.fill.is_some()
    }

    /// The hub is reconnecting the device.
    pub(super) fn recovering(&self) -> bool {
        self.consumer.is_recovering()
    }

    /// Park until the hub delivers (or `timeout`).
    pub(super) fn wait(&self, timeout: Duration) {
        self.consumer.wait(timeout);
    }

    /// Move the capture to another hub source (the same microphone under a renumbered name,
    /// [`InputFollow`]), keeping the run format, the mute gate and the output rings' burst. The old
    /// consumer is detached. Silence keeps being synthesized until the new source delivers, and its
    /// first audio is spliced in like any capture restart. Blocks up to the hub's attach timeout.
    pub(super) fn reattach(&mut self, hub: &CaptureHub, target: &InputTarget) -> Result<(), String> {
        let opts = ConsumerOptions::device_rate("vc")
            .with_mute_group(Some(MuteGroup::VoiceChanger))
            .with_max_backlog_ms(HUB_MAX_BACKLOG_MS);
        self.consumer = hub.attach(target.spec.clone(), opts)?;
        // A new stream: its resampler history does not continue the old one.
        self.adapter = None;
        Ok(())
    }

    /// Move everything the hub has into `sink` (splitting at discontinuities), and fill silence
    /// while the capture is idle. `now_ns`: QPC now; `block_frames`: the worker's block (the idle
    /// fill starts at most one block back).
    pub(super) fn pump(&mut self, sink: &mut dyn CaptureSink, now_ns: i64, block_frames: usize) -> Intake {
        let mut intake = Intake::default();
        loop {
            self.read_buf.clear();
            let info = self.consumer.read(&mut self.read_buf);
            intake.hub_dropped += info.dropped_frames;
            if info.samples_out > 0 {
                if self.fill.take().is_some() {
                    // Real audio after synthesized silence: splice.
                    sink.gap(0);
                    intake.gaps += 1;
                }
                let format = info.format.unwrap_or(StreamFormat {
                    sample_rate: self.rate,
                    channels: self.ch as u16,
                });
                let same = format.sample_rate == self.rate && usize::from(format.channels.max(1)) == self.ch;
                let frames = if same {
                    self.adapter = None;
                    sink.audio(&self.read_buf);
                    self.read_buf.len() / self.ch
                } else {
                    if self.adapter.as_ref().is_none_or(|a| a.from != format) {
                        self.adapter = Some(FormatAdapter::new(format, self.rate, self.ch));
                    }
                    let adapter = self.adapter.as_mut().expect("set above");
                    adapter.convert(&self.read_buf, self.ch, &mut self.adapted);
                    sink.audio(&self.adapted);
                    self.adapted.len() / self.ch
                };
                intake.frames += frames;
                // One device block per read in steady state (the hub wakes the worker per block);
                // a catch-up read after a stall is not a capture period and must not grow the
                // output rings' top-up target.
                if frames <= (self.rate as usize) / 20 {
                    self.burst.store(frames * self.ch, Ordering::Relaxed);
                }
                self.last_audio_ns = now_ns;
                if let Some(end) = info.end_qpc_ns {
                    self.newest_capture_ns = Some(end);
                }
            }
            if info.gap_after {
                sink.gap(info.dropped_frames);
                intake.gaps += 1;
                if info.restarted {
                    intake.restarted = true;
                    // A new stream: its resampler history does not continue the old one.
                    self.adapter = None;
                }
                continue;
            }
            break;
        }
        if intake.frames == 0 {
            if let Some(why) = self.consumer.ended() {
                intake.ended = Some(why);
                return intake;
            }
            let idle = Duration::from_nanos(now_ns.saturating_sub(self.last_audio_ns).max(0) as u64);
            let muted = (self.muted)();
            if self.fill.is_some() || muted || self.consumer.is_recovering() || idle >= IDLE_FILL_AFTER {
                intake.filled = self.fill_silence(sink, now_ns, block_frames, &mut intake);
            }
        }
        intake
    }

    fn fill_silence(&mut self, sink: &mut dyn CaptureSink, now_ns: i64, block_frames: usize, intake: &mut Intake) -> usize {
        let block_ns = (block_frames as i64).saturating_mul(1_000_000_000) / i64::from(self.rate);
        if self.fill.is_none() {
            // The capture stopped: fade out whatever was captured before the silence (unless this
            // pump already reported the discontinuity that stopped it).
            if intake.gaps == 0 {
                sink.gap(0);
                intake.gaps += 1;
            }
            self.fill = Some(Fill {
                origin_ns: self.last_audio_ns.max(now_ns - block_ns),
                frames: 0,
            });
        }
        let fill = self.fill.as_mut().expect("set above");
        let elapsed = now_ns.saturating_sub(fill.origin_ns).max(0) as u128;
        let due = (elapsed * u128::from(self.rate) / 1_000_000_000) as u64;
        let mut todo = due.saturating_sub(fill.frames) as usize;
        let total = todo;
        let per = self.zeros.len() / self.ch;
        while todo > 0 {
            let n = todo.min(per);
            sink.audio(&self.zeros[..n * self.ch]);
            todo -= n;
        }
        fill.frames += total as u64;
        total
    }
}

// ---------------------------------------------------------------------------------------------
// Input follow: a named microphone that comes back renumbered
// ---------------------------------------------------------------------------------------------

/// How long the hub must have been reconnecting a named input before the voice changer looks for it
/// under a renumbered name (a replug into the same port comes back under its own name first).
pub(super) const RENUMBER_AFTER: Duration = Duration::from_millis(500);
/// How often it looks while the device stays away.
pub(super) const RENUMBER_CHECK_EVERY: Duration = Duration::from_secs(1);

/// Looks a saved input name up for [`InputFollow`] ([`resolve_input_runtime`] on a fresh cpal host;
/// tests replace it).
pub(super) type InputResolver = LookupFn<InputTarget>;

/// Follows a saved, named input device to its new name while the hub reconnects it: Windows renames
/// a USB endpoint moved to another port ("Microphone (2- USB Audio)" becomes "Microphone (3- USB
/// Audio)") and the hub reconnects by exact name, so such a device would never come back. When the
/// hub starts reconnecting, a lookup takes the loss snapshot (the family's devices present then);
/// after [`RENUMBER_AFTER`] the saved name is looked up again at most every
/// [`RENUMBER_CHECK_EVERY`] with the runtime rule ([`match_runtime_device`]: the exact name, else the
/// one device of its family that was not there at the loss), on a helper thread ([`DeviceLookup`]);
/// a device found under another name is where the capture moves. The system default needs none of
/// this (the hub follows it).
pub(super) struct InputFollow {
    /// `None` for the system default.
    lookup: Option<DeviceLookup<InputTarget>>,
    recovering_since: Option<Instant>,
    next_check: Option<Instant>,
}

impl InputFollow {
    pub(super) fn new(saved: &str) -> InputFollow {
        Self::with_resolver(
            saved,
            Arc::new(|saved: &str, present: Option<&[String]>| resolve_input_runtime(&cpal::default_host(), saved, present)),
        )
    }

    pub(super) fn with_resolver(saved: &str, resolver: InputResolver) -> InputFollow {
        InputFollow {
            lookup: (!is_default_device(saved)).then(|| DeviceLookup::new(saved, resolver)),
            recovering_since: None,
            next_check: None,
        }
    }

    /// Call about once per worker iteration with whether the hub is reconnecting the capture.
    /// `Some(target)` (the device under its new name, in the run format of `current`) when the
    /// capture should move to it.
    pub(super) fn poll(&mut self, recovering: bool, current: &InputTarget, now: Instant) -> Option<InputTarget> {
        let lookup = self.lookup.as_mut()?;
        if !recovering {
            if self.recovering_since.take().is_some() {
                lookup.reset();
            }
            self.next_check = None;
            return None;
        }
        let since = match self.recovering_since {
            Some(t) => t,
            None => {
                // The loss starts: snapshot the family's devices now (a stand-in must be new).
                lookup.reset();
                lookup.start();
                self.recovering_since = Some(now);
                now
            }
        };
        if let Some(Ok(found)) = lookup.take() {
            if found.name.trim() != current.name.trim() {
                return Some(InputTarget {
                    renamed_from: Some(current.name.clone()),
                    rate: current.rate,
                    ch: current.ch,
                    ..found
                });
            }
        }
        if now.duration_since(since) < RENUMBER_AFTER || self.next_check.is_some_and(|t| now < t) || lookup.busy() {
            return None;
        }
        self.next_check = Some(now + RENUMBER_CHECK_EVERY);
        lookup.start();
        None
    }
}

// ---------------------------------------------------------------------------------------------
// Output ports
// ---------------------------------------------------------------------------------------------

/// A playing port with no device callback for this long is treated as disconnected.
pub(super) const OUTPUT_STALL_AFTER: Duration = Duration::from_millis(1_500);
/// Retry period while a port's device is gone.
pub(super) const REOPEN_EVERY: Duration = Duration::from_millis(250);
/// How often a port on the system default checks whether the default moved.
const DEFAULT_CHECK_EVERY: Duration = Duration::from_secs(1);

/// Device-lost flags set by cpal error callbacks (real-time safe), read by the worker.
#[derive(Default)]
pub(super) struct StreamErrors {
    pub(super) device_lost: AtomicBool,
    pub(super) backend_errors: AtomicU64,
}

impl StreamErrors {
    fn note(&self, e: StreamError) {
        match e {
            StreamError::DeviceNotAvailable => self.device_lost.store(true, Ordering::Relaxed),
            _ => {
                self.backend_errors.fetch_add(1, Ordering::Relaxed);
            }
        }
    }
}

/// Open an f32 output stream playing `playout`, registering its callback thread with MMCSS.
fn build_output_stream(
    device: &cpal::Device,
    config: &cpal::SupportedStreamConfig,
    mut playout: RingPlayout,
    mmcss: Arc<MmcssStatus>,
    which: fn(&MmcssStatus) -> &MmcssFlag,
    errors: Arc<StreamErrors>,
) -> Result<cpal::Stream, String> {
    device
        .build_output_stream(
            &config.config(),
            move |data: &mut [f32], _: &cpal::OutputCallbackInfo| {
                which(&mmcss).record(rt_thread::register_mmcss_pro_audio_once());
                playout.fill(data);
            },
            move |e| errors.note(e),
            None,
        )
        .map_err(|e| format!("Failed to open the output stream: {e}"))
}

fn ms_to_samples(ms: f64, rate: u32, ch: usize) -> usize {
    ((f64::from(rate)) * ms / 1000.0).ceil() as usize * ch
}

fn ring_capacity(prefill: usize, max_queue: usize) -> usize {
    max_queue
        .saturating_mul(2)
        .clamp(OUT_RING_MIN_SAMPLES, OUT_RING_MAX_SAMPLES)
        .max(prefill.saturating_add(1))
}

/// An output device found for a port: the device, its name and its config.
pub(super) type OutputFound = (cpal::Device, String, cpal::SupportedStreamConfig);

/// Find the output device of a port and its config (f32 only).
pub(super) fn resolve_output(host: &cpal::Host, role: DeviceRole, saved: &str) -> Result<OutputFound, String> {
    let opened = open_saved_device(host, role, saved)?;
    if let Some(from) = &opened.renamed_from {
        tracing::info!("voice changer: {role:?} device '{from}' was renumbered by Windows; using '{}'", opened.name);
    }
    output_found(opened, role)
}

/// [`resolve_output`] for a device a port follows ([`MatchRule::Runtime`]).
pub(super) fn resolve_output_runtime(host: &cpal::Host, role: DeviceRole, saved: &str, present_at_loss: Option<&[String]>) -> Lookup<OutputFound> {
    let (opened, family) = open_device(host, role, saved, MatchRule::Runtime(present_at_loss));
    Lookup {
        found: opened.and_then(|o| output_found(o, role)),
        family,
    }
}

fn output_found(opened: OpenedDevice, role: DeviceRole) -> Result<OutputFound, String> {
    let config = opened
        .device
        .default_output_config()
        .map_err(|e| format!("The {} device '{}' cannot be opened: {e}", role.label(), opened.name))?;
    if !matches!(config.sample_format(), SampleFormat::F32) {
        return Err(format!(
            "The {} device '{}' uses {:?} samples (32-bit float is needed)",
            role.label(),
            opened.name,
            config.sample_format()
        ));
    }
    Ok((opened.device, opened.name, config))
}

/// A port's device changed state (the worker tells the user).
#[derive(Clone, Debug, PartialEq, Eq)]
pub(super) enum PortEvent {
    /// The device disconnected (message); the port retries.
    Lost(String),
    /// Playing again on the device (its name; `renamed_from`: the name it had when it was lost, when
    /// it came back under another one).
    Restored { name: String, renamed_from: Option<String> },
    /// Moved to the new system default device (its name).
    Moved(String),
}

/// Queue sizing of a port (the hot params; ms).
#[derive(Clone, Copy, Debug, PartialEq)]
pub(super) struct PortLimits {
    pub(super) prefill_ms: f64,
    pub(super) max_queue_ms: f64,
    /// The smallest cushion the active voice allows (`min_output_buffer_ms`; 0 = none): the adaptive
    /// cushion's floor. Fixed ports already hold it in `prefill_ms`.
    pub(super) min_prefill_ms: f64,
}

/// A running output device (see the module docs).
pub(super) struct OutputPort {
    role: DeviceRole,
    saved: String,
    name: String,
    /// Format of the audio the worker pushes.
    run_rate: u32,
    run_ch: usize,
    dev_rate: u32,
    dev_ch: usize,
    limits: PortLimits,
    stream: Option<cpal::Stream>,
    errors: Arc<StreamErrors>,
    callbacks: Arc<AtomicU64>,
    seen_callbacks: u64,
    last_progress: Instant,
    producer: rtrb::Producer<f32>,
    ring_cap: usize,
    playback: OutputPlayback,
    underruns: Option<Arc<AtomicU64>>,
    mmcss: Arc<MmcssStatus>,
    which: fn(&MmcssStatus) -> &MmcssFlag,
    /// Capture burst (interleaved samples at the capture format) for the ring target.
    burst: Arc<AtomicUsize>,
    burst_ch: usize,
    burst_rate: u32,
    /// How the queue's cushion is sized (kept for a reopened device).
    cushion: OutputCushion,
    resampler: Option<PolyphaseResampler>,
    chans: Vec<f32>,
    resampled: Vec<f32>,
    /// While the device is gone: when to look for it again.
    lost: Option<Instant>,
    /// Looks the saved device up off the worker thread (while it is gone, and for a port on the
    /// system default, whether the default moved).
    lookup: DeviceLookup<OutputFound>,
    default_check: Instant,
}

/// What a port needs to start (besides its device).
pub(super) struct PortSetup {
    pub(super) role: DeviceRole,
    pub(super) saved: String,
    pub(super) run_rate: u32,
    pub(super) run_ch: usize,
    pub(super) limits: PortLimits,
    pub(super) underruns: Option<Arc<AtomicU64>>,
    pub(super) mmcss: Arc<MmcssStatus>,
    pub(super) which: fn(&MmcssStatus) -> &MmcssFlag,
    pub(super) burst: Arc<AtomicUsize>,
    pub(super) burst_ch: usize,
    pub(super) burst_rate: u32,
    /// How the queue's cushion is sized (the main output of an RVC run; the monitor stays fixed).
    pub(super) cushion: OutputCushion,
}

impl OutputPort {
    /// Start `device` (already resolved with [`resolve_output`]); the stream plays at once.
    pub(super) fn start(setup: PortSetup, device: cpal::Device, name: String, config: cpal::SupportedStreamConfig) -> Result<OutputPort, String> {
        let dev_rate = config.sample_rate().0;
        let dev_ch = usize::from(config.channels()).max(1);
        let (producer, ring_cap, playback, callbacks, errors, stream) = Self::open_stream(&setup, &device, &config, dev_rate, dev_ch)?;
        let role = setup.role;
        let lookup = DeviceLookup::new(
            &setup.saved,
            Arc::new(move |saved: &str, present: Option<&[String]>| resolve_output_runtime(&cpal::default_host(), role, saved, present)),
        );
        Ok(OutputPort {
            role: setup.role,
            saved: setup.saved,
            name,
            run_rate: setup.run_rate.max(1),
            run_ch: setup.run_ch.max(1),
            dev_rate,
            dev_ch,
            limits: setup.limits,
            stream: Some(stream),
            errors,
            callbacks,
            seen_callbacks: 0,
            last_progress: Instant::now(),
            producer,
            ring_cap,
            playback,
            underruns: setup.underruns,
            mmcss: setup.mmcss,
            which: setup.which,
            burst: setup.burst,
            burst_ch: setup.burst_ch.max(1),
            burst_rate: setup.burst_rate.max(1),
            cushion: setup.cushion,
            resampler: (dev_rate != setup.run_rate).then(|| PolyphaseResampler::new(setup.run_rate, dev_rate, dev_ch)),
            chans: Vec::new(),
            resampled: Vec::new(),
            lost: None,
            lookup,
            default_check: Instant::now(),
        })
    }

    #[allow(clippy::type_complexity)]
    fn open_stream(
        setup: &PortSetup,
        device: &cpal::Device,
        config: &cpal::SupportedStreamConfig,
        dev_rate: u32,
        dev_ch: usize,
    ) -> Result<(rtrb::Producer<f32>, usize, OutputPlayback, Arc<AtomicU64>, Arc<StreamErrors>, cpal::Stream), String> {
        let prefill = ms_to_samples(setup.limits.prefill_ms, dev_rate, dev_ch);
        let max_queue = ms_to_samples(setup.limits.max_queue_ms, dev_rate, dev_ch);
        let ring_cap = ring_capacity(prefill, max_queue);
        let (producer, consumer) = rtrb::RingBuffer::<f32>::new(ring_cap);
        let callback_len = Arc::new(AtomicUsize::new(0));
        // The main output counts into the metrics' counter; the monitor into its own.
        let underruns = setup.underruns.clone().unwrap_or_else(|| Arc::new(AtomicU64::new(0)));
        let mut playback = OutputPlayback::new(prefill, max_queue, dev_ch, callback_len.clone())
            .with_capture_burst(setup.burst.clone(), setup.burst_ch, setup.burst_rate, dev_rate)
            .with_cushion(setup.cushion);
        playback.set_cushion_floor(ms_to_samples(setup.limits.min_prefill_ms, dev_rate, dev_ch));
        let callbacks = Arc::new(AtomicU64::new(0));
        let playout = RingPlayout::new(consumer, dev_ch, callback_len)
            .with_callback_counter(callbacks.clone())
            .with_underrun_counter(underruns);
        let errors = Arc::new(StreamErrors::default());
        let stream = build_output_stream(device, config, playout, setup.mmcss.clone(), setup.which, errors.clone())?;
        stream
            .play()
            .map_err(|e| format!("The {} device could not be started: {e}", setup.role.label()))?;
        Ok((producer, ring_cap, playback, callbacks, errors, stream))
    }

    pub(super) fn name(&self) -> &str {
        &self.name
    }

    /// The device is gone (the port discards audio and retries).
    pub(super) fn is_lost(&self) -> bool {
        self.lost.is_some()
    }

    pub(super) fn backend_errors(&self) -> u64 {
        self.errors.backend_errors.load(Ordering::Relaxed)
    }

    fn ring_used(&self) -> usize {
        self.ring_cap.saturating_sub(self.producer.slots())
    }

    /// Queued frames (deque + ring) at the device rate, and that rate.
    pub(super) fn queued_frames(&self) -> (usize, u32) {
        ((self.playback.queue_len() + self.ring_used()) / self.dev_ch, self.dev_rate)
    }

    /// Queued interleaved samples (deque + ring) at the device format.
    pub(super) fn queued_samples(&self) -> usize {
        self.playback.queue_len() + self.ring_used()
    }

    /// Frames trimmed by the latency control so far, and its recoveries (see `OutputPlayback`).
    pub(super) fn trims(&self) -> (u64, u64) {
        (self.playback.trimmed_frames(), self.playback.recoveries())
    }

    /// The cushion the queue holds now (ms): the adaptive target, else the prefill (P3B-H).
    pub(super) fn cushion_ms(&self) -> f64 {
        self.playback.cushion_samples() as f64 / self.dev_ch as f64 * 1000.0 / f64::from(self.dev_rate.max(1))
    }

    /// New queue limits from the hot params.
    pub(super) fn set_limits(&mut self, limits: PortLimits) {
        self.limits = limits;
        self.playback
            .set_cushion_floor(ms_to_samples(limits.min_prefill_ms, self.dev_rate, self.dev_ch));
        let used = self.ring_used();
        self.playback.update_limits(
            ms_to_samples(limits.prefill_ms, self.dev_rate, self.dev_ch),
            ms_to_samples(limits.max_queue_ms, self.dev_rate, self.dev_ch),
            used,
        );
    }

    /// The input skipped audio: trim the catch-up (see `OutputPlayback::note_discontinuity`).
    pub(super) fn note_discontinuity(&mut self) {
        self.playback.note_discontinuity();
    }

    /// Queue one processed block (run format). Discarded while the device is gone.
    pub(super) fn push(&mut self, block: &[f32], truncate: (bool, f32, u32)) {
        if self.stream.is_none() {
            return;
        }
        let data: &[f32] = if self.run_ch == self.dev_ch && self.resampler.is_none() {
            block
        } else {
            adapt_channels_interleaved_into(block, self.run_ch, self.dev_ch, &mut self.chans);
            match self.resampler.as_mut() {
                Some(r) => {
                    self.resampled.clear();
                    r.process(&self.chans, &mut self.resampled);
                    &self.resampled
                }
                None => &self.chans,
            }
        };
        let used = self.ring_cap.saturating_sub(self.producer.slots());
        self.playback.push_interleaved(data, truncate.0, truncate.1, truncate.2, used);
        let used = self.ring_cap.saturating_sub(self.producer.slots());
        self.playback.flush_to_ring(&mut self.producer, used);
    }

    /// Watch the device: loss, reconnection, default moves. Call about once per block. Never waits
    /// for a device lookup (they run on the port's helper thread); a device that was found is opened
    /// here.
    pub(super) fn maintain(&mut self) -> Option<PortEvent> {
        let now = Instant::now();
        if let Some(next_try) = self.lost {
            if let Some(Ok(found)) = self.lookup.take() {
                let lost_name = self.name.clone();
                if self.open_found(found).is_ok() {
                    self.lost = None;
                    self.lookup.reset();
                    let renamed_from = (!is_default_device(&self.saved) && self.name.trim() != lost_name.trim()).then_some(lost_name);
                    return Some(PortEvent::Restored {
                        name: self.name.clone(),
                        renamed_from,
                    });
                }
            }
            if now >= next_try && !self.lookup.busy() {
                self.lookup.start();
                self.lost = Some(now + REOPEN_EVERY);
            }
            return None;
        }
        let calls = self.callbacks.load(Ordering::Relaxed);
        if calls != self.seen_callbacks {
            self.seen_callbacks = calls;
            self.last_progress = now;
        }
        let stalled = now.duration_since(self.last_progress) >= OUTPUT_STALL_AFTER;
        if self.errors.device_lost.load(Ordering::Relaxed) || stalled {
            self.stream = None;
            // The loss starts: the first lookup snapshots the family's devices (a stand-in must be new).
            self.lookup.reset();
            self.lookup.start();
            self.lost = Some(now + REOPEN_EVERY);
            let why = if stalled { "stopped playing" } else { "was disconnected" };
            return Some(PortEvent::Lost(format!("The {} device '{}' {why}", self.role.label(), self.name)));
        }
        if is_default_device(&self.saved) {
            if let Some(Ok(found)) = self.lookup.take() {
                if found.1.trim() != self.name.trim() {
                    let to = found.1.clone();
                    match self.open_found(found) {
                        Ok(()) => return Some(PortEvent::Moved(self.name.clone())),
                        Err(e) => {
                            // The old stream is closed already: the port is lost until a device opens.
                            self.lookup.reset();
                            self.lookup.start();
                            self.lost = Some(now + REOPEN_EVERY);
                            return Some(PortEvent::Lost(format!(
                                "The {} device could not move to the new default device '{to}' ({e})",
                                self.role.label()
                            )));
                        }
                    }
                }
            }
            if now.duration_since(self.default_check) >= DEFAULT_CHECK_EVERY && !self.lookup.busy() {
                self.default_check = now;
                self.lookup.start();
            }
        }
        None
    }

    /// Open a device found for this port (possibly in another format); the queue restarts empty.
    fn open_found(&mut self, (device, name, config): OutputFound) -> Result<(), String> {
        let dev_rate = config.sample_rate().0;
        let dev_ch = usize::from(config.channels()).max(1);
        let setup = PortSetup {
            role: self.role,
            saved: self.saved.clone(),
            run_rate: self.run_rate,
            run_ch: self.run_ch,
            limits: self.limits,
            underruns: self.underruns.clone(),
            mmcss: self.mmcss.clone(),
            which: self.which,
            burst: self.burst.clone(),
            burst_ch: self.burst_ch,
            burst_rate: self.burst_rate,
            cushion: self.cushion,
        };
        // Release the old stream first (an exclusive driver may refuse a second one).
        self.stream = None;
        let (producer, ring_cap, playback, callbacks, errors, stream) = Self::open_stream(&setup, &device, &config, dev_rate, dev_ch)?;
        self.producer = producer;
        self.ring_cap = ring_cap;
        self.playback = playback;
        self.callbacks = callbacks;
        self.seen_callbacks = 0;
        self.last_progress = Instant::now();
        self.errors = errors;
        self.stream = Some(stream);
        if dev_rate != self.dev_rate || dev_ch != self.dev_ch || self.name != name {
            self.resampler = (dev_rate != self.run_rate).then(|| PolyphaseResampler::new(self.run_rate, dev_rate, dev_ch));
        }
        self.dev_rate = dev_rate;
        self.dev_ch = dev_ch;
        self.name = name;
        Ok(())
    }
}

#[cfg(test)]
pub(super) mod tests {
    use super::*;
    use crate::services::audio::device_resolve::device_family;

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

    // ---- the hub consumer, over a scripted capture backend ----

    use crate::services::audio::hub::{
        CaptureBackend, CaptureStream, ErrorSink, HubConfig, ResolvedDevice, StreamSink,
    };
    use crate::services::speech_core::resample::StreamResampler;
    use std::sync::Mutex;

    /// One scripted device (48 kHz stereo unless changed) whose callback the test plays.
    #[derive(Clone)]
    pub(crate) struct ScriptedMic(Arc<ScriptedInner>);

    pub(crate) struct ScriptedInner {
        name: &'static str,
        format: Mutex<StreamFormat>,
        present: AtomicBool,
        sink: Mutex<Option<StreamSink>>,
        errors: Mutex<Option<ErrorSink>>,
    }

    struct ScriptedStream(Arc<ScriptedInner>);

    impl CaptureStream for ScriptedStream {
        fn stop(self: Box<Self>, released: Option<std::sync::mpsc::Sender<()>>) {
            drop(self.0.sink.lock().unwrap().take());
            if let Some(r) = released {
                let _ = r.send(());
            }
        }
    }

    impl CaptureBackend for ScriptedMic {
        fn resolve(&self, _spec: &SourceSpec) -> Result<ResolvedDevice, String> {
            if !self.0.present.load(Ordering::SeqCst) {
                return Err("not present".into());
            }
            Ok(ResolvedDevice {
                name: self.0.name.to_string(),
                loopback: false,
                format: *self.0.format.lock().unwrap(),
                handle: None,
            })
        }

        fn open(&self, device: ResolvedDevice, sink: StreamSink, errors: ErrorSink) -> Result<Box<dyn CaptureStream>, String> {
            if !self.0.present.load(Ordering::SeqCst) || device.format != *self.0.format.lock().unwrap() {
                return Err("not present".into());
            }
            *self.0.sink.lock().unwrap() = Some(sink);
            *self.0.errors.lock().unwrap() = Some(errors);
            Ok(Box::new(ScriptedStream(self.0.clone())))
        }
    }

    impl ScriptedMic {
        pub(crate) fn new(format: StreamFormat) -> ScriptedMic {
            ScriptedMic(Arc::new(ScriptedInner {
                name: "Mic A",
                format: Mutex::new(format),
                present: AtomicBool::new(true),
                sink: Mutex::new(None),
                errors: Mutex::new(None),
            }))
        }

        /// One device callback. Returns whether a stream took it.
        pub(crate) fn push(&self, data: &[f32]) -> bool {
            match self.0.sink.lock().unwrap().as_mut() {
                Some(s) => {
                    s.push(data, crate::services::audio::hub::now_qpc_ns());
                    true
                }
                None => false,
            }
        }

        /// Unplug (the stream dies) or plug back in, possibly in another format.
        pub(crate) fn unplug(&self) {
            self.0.present.store(false, Ordering::SeqCst);
            drop(self.0.sink.lock().unwrap().take());
            if let Some(mut e) = self.0.errors.lock().unwrap().take() {
                e.report("unplugged".into());
            }
        }

        pub(crate) fn plug(&self, format: StreamFormat) {
            *self.0.format.lock().unwrap() = format;
            self.0.present.store(true, Ordering::SeqCst);
        }

        pub(crate) fn live(&self) -> bool {
            self.0.sink.lock().unwrap().is_some()
        }
    }

    pub(crate) fn test_hub(mic: &ScriptedMic) -> CaptureHub {
        CaptureHub::new(
            Arc::new(mic.clone()),
            HubConfig {
                tick: Duration::from_millis(10),
                stall_timeout: Duration::ZERO,
                first_data_timeout: Duration::ZERO,
                retry_min: Duration::from_millis(20),
                retry_max: Duration::from_millis(100),
                share_streams: true,
                supervise: true,
                attach_timeout: Duration::from_secs(10),
            },
        )
        .unwrap()
    }

    pub(crate) const STEREO_48K: StreamFormat = StreamFormat {
        sample_rate: 48_000,
        channels: 2,
    };

    pub(crate) fn target(format: StreamFormat) -> InputTarget {
        InputTarget {
            spec: SourceSpec::DefaultInput,
            name: "Mic A".into(),
            renamed_from: None,
            rate: format.sample_rate,
            ch: usize::from(format.channels),
        }
    }

    /// Records what the capture delivered.
    #[derive(Default)]
    pub(crate) struct Recorded {
        pub(crate) audio: Vec<f32>,
        /// (sample index where the gap sits, lost frames)
        pub(crate) gaps: Vec<(usize, u64)>,
    }

    impl CaptureSink for Recorded {
        fn audio(&mut self, frames: &[f32]) {
            self.audio.extend_from_slice(frames);
        }
        fn gap(&mut self, lost_frames: u64) {
            self.gaps.push((self.audio.len(), lost_frames));
        }
    }

    fn now() -> i64 {
        crate::services::audio::hub::now_qpc_ns()
    }

    fn wait_until(what: &str, mut cond: impl FnMut() -> bool) {
        let deadline = Instant::now() + Duration::from_secs(10);
        while !cond() {
            assert!(Instant::now() < deadline, "timed out waiting for: {what}");
            std::thread::sleep(Duration::from_millis(2));
        }
    }

    fn tone(frames: usize, ch: usize, from: usize) -> Vec<f32> {
        (0..frames)
            .flat_map(|i| {
                let v = (((from + i) as f32) * 0.013).sin() * 0.5;
                std::iter::repeat_n(v, ch)
            })
            .collect()
    }

    /// The acceptance check "STT PCM equals device input": one stream feeds both. The voice
    /// changer's consumer receives the device's PCM exactly; whatever it then does to its own copy
    /// (gain, denoiser) never reaches STT, whose 16 kHz feed equals the device input resampled on
    /// its own.
    #[test]
    fn stt_gets_the_raw_device_input_while_the_voice_changer_conditions_its_copy() {
        let mic = ScriptedMic::new(STEREO_48K);
        let hub = test_hub(&mic);
        let mut stt = hub.attach(SourceSpec::DefaultInput, ConsumerOptions::speech("stt")).unwrap();
        let mut vc = HubInput::attach(&hub, &target(STEREO_48K), Box::new(|| false)).unwrap();
        wait_until("stream live", || mic.live());
        let input = tone(480 * 50, 2, 0);
        let mut vc_got = Recorded::default();
        for block in input.chunks(480 * 2) {
            assert!(mic.push(block));
            let intake = vc.pump(&mut vc_got, now(), 480);
            assert_eq!(intake.filled, 0);
            // The voice changer conditions its own copy in place (gain 4, a "denoiser" that zeroes).
            for s in vc_got.audio.iter_mut() {
                *s = 0.0;
            }
        }
        // STT: exactly what a standalone 16 kHz consumer of the raw device input produces.
        let mut stt_pcm = Vec::new();
        loop {
            let before = stt_pcm.len();
            stt.read(&mut stt_pcm);
            if stt_pcm.len() == before {
                break;
            }
        }
        let mut reference = Vec::new();
        let mut r = StreamResampler::new(48_000, 2).unwrap();
        r.push_interleaved(&input, &mut reference).unwrap();
        assert!(!stt_pcm.is_empty());
        assert_eq!(stt_pcm.len(), reference.len());
        for (i, (a, b)) in stt_pcm.iter().zip(reference.iter()).enumerate() {
            assert!((a - b).abs() < 1e-6, "STT sample {i} differs: {a} vs {b}");
        }
        // And the voice changer read the device PCM exactly, whole frames, no gaps.
        let mut exact = Recorded::default();
        let mut vc2 = HubInput::attach(&hub, &target(STEREO_48K), Box::new(|| false)).unwrap();
        let more = tone(480 * 4, 2, 7);
        for block in more.chunks(480 * 2) {
            assert!(mic.push(block));
        }
        vc2.pump(&mut exact, now(), 480);
        assert_eq!(exact.audio, more);
        assert!(exact.gaps.is_empty());
        drop(vc);
    }

    /// Unplug and replug in another format: the gap is reported (the budget splices it), silence is
    /// synthesized while the hub reconnects, and the new format is converted to the run's.
    #[test]
    fn device_restarts_are_gaps_and_formats_are_converted() {
        let mic = ScriptedMic::new(STEREO_48K);
        let hub = test_hub(&mic);
        let mut vc = HubInput::attach(&hub, &target(STEREO_48K), Box::new(|| false)).unwrap();
        wait_until("stream live", || mic.live());
        let mut got = Recorded::default();
        assert!(mic.push(&tone(480, 2, 0)));
        vc.pump(&mut got, now(), 480);
        assert_eq!(got.audio.len(), 960);
        mic.unplug();
        wait_until("recovering", || vc.recovering());
        // While the device is gone: silence on the clock, after a gap that fades out the audio.
        std::thread::sleep(Duration::from_millis(30));
        let intake = vc.pump(&mut got, now(), 480);
        assert!(intake.filled > 0 && vc.filling(), "{intake:?}");
        assert_eq!(got.gaps.len(), 1);
        // Back at 44.1 kHz mono: converted to 48 kHz stereo, after a gap.
        let mono44 = StreamFormat {
            sample_rate: 44_100,
            channels: 1,
        };
        mic.plug(mono44);
        hub.devices_changed();
        wait_until("reopened", || mic.live());
        let before = got.audio.len();
        for i in 0..20 {
            assert!(mic.push(&tone(441, 1, i * 441)));
        }
        let intake = vc.pump(&mut got, now(), 480);
        assert!(intake.gaps >= 1, "{intake:?}");
        assert!(!vc.filling());
        let frames = (got.audio.len() - before) / 2;
        // 20 x 10 ms at 44.1 kHz = 9600 frames at 48 kHz, less the resampler's look-ahead.
        assert!((9_500..=9_600).contains(&frames), "{frames}");
        for f in got.audio[before..].chunks(2) {
            assert_eq!(f[0], f[1], "mono copied to both channels");
        }
    }

    /// At the start of a run the one device of a family stands in for a renumbered one; while a run
    /// follows a device that disappeared, only a device that was not there when it was lost does (the
    /// same device on another port), never one that was there all along.
    #[test]
    fn at_runtime_only_a_device_new_since_the_loss_stands_in() {
        let headset = "Microphone (2- USB Audio Device)";
        let dongle = "Microphone (USB Audio Device)";
        let at_loss = vec![dongle.to_string()];
        let only_dongle = vec![dongle.to_string()];
        // The start rule takes the dongle (the one device of the family) ...
        assert_eq!(match_saved_device(headset, &only_dongle), Some((0, true)));
        // ... the runtime rule never does.
        assert_eq!(match_runtime_device(headset, &only_dongle, Some(&at_loss)), None);
        assert_eq!(match_runtime_device(headset, &only_dongle, None), None, "the snapshot lookup takes only the exact name");
        // The headset renumbered on another port is new since the loss: followed.
        let moved = vec![dongle.to_string(), "Microphone (3- USB Audio Device)".to_string()];
        assert_eq!(match_runtime_device(headset, &moved, Some(&at_loss)), Some((1, true)));
        // Back under its own name: the exact name, whatever the snapshot.
        let back = vec![dongle.to_string(), headset.to_string()];
        assert_eq!(match_runtime_device(headset, &back, Some(&at_loss)), Some((1, false)));
        assert_eq!(match_runtime_device(headset, &back, None), Some((1, false)));
        // Two new devices of the family: not guessed.
        let two = vec!["Microphone (3- USB Audio Device)".to_string(), "Microphone (4- USB Audio Device)".to_string()];
        assert_eq!(match_runtime_device(headset, &two, Some(&[])), None);
        assert_eq!(
            family_members(headset, &[dongle.to_string(), "Speakers (Realtek(R) Audio)".to_string()]),
            vec![dongle.to_string()]
        );
    }

    /// Device lookups never block the thread that asks (the worker): they run one at a time on a
    /// helper thread; the first one of a loss takes the snapshot the later ones use; the answer of a
    /// lookup that was running when the loss ended (a reset) is dropped.
    #[test]
    fn device_lookups_run_off_the_calling_thread() {
        let running = Arc::new(AtomicUsize::new(0));
        let most = Arc::new(AtomicUsize::new(0));
        let resolve: LookupFn<usize> = {
            let (running, most) = (running.clone(), most.clone());
            Arc::new(move |saved: &str, present: Option<&[String]>| {
                assert_eq!(saved, "Mic");
                let n = running.fetch_add(1, Ordering::SeqCst) + 1;
                most.fetch_max(n, Ordering::SeqCst);
                std::thread::sleep(Duration::from_millis(60));
                running.fetch_sub(1, Ordering::SeqCst);
                Lookup {
                    // The answer is the size of the snapshot it was given (`usize::MAX`: none).
                    found: Ok(present.map_or(usize::MAX, <[String]>::len)),
                    family: vec!["a".into(), "b".into()],
                }
            })
        };
        let wait = |l: &mut DeviceLookup<usize>| {
            let deadline = Instant::now() + Duration::from_secs(10);
            loop {
                if let Some(r) = l.take() {
                    return r;
                }
                assert!(Instant::now() < deadline, "the lookup never answered");
                std::thread::sleep(Duration::from_millis(2));
            }
        };
        let mut l = DeviceLookup::new(" Mic ", resolve);
        let t0 = Instant::now();
        l.start();
        assert!(l.busy());
        assert!(l.take().is_none());
        l.start();
        assert!(t0.elapsed() < Duration::from_millis(40), "the caller never waits: {:?}", t0.elapsed());
        assert_eq!(wait(&mut l), Ok(usize::MAX), "the first lookup takes the snapshot");
        assert!(!l.busy());
        l.start();
        assert_eq!(wait(&mut l), Ok(2), "later lookups use it");
        // A reset while a lookup runs: its answer is dropped, the next lookup snapshots again.
        l.start();
        l.reset();
        assert!(!l.busy());
        l.start();
        assert_eq!(wait(&mut l), Ok(usize::MAX));
        assert_eq!(most.load(Ordering::SeqCst), 1, "one lookup at a time");
    }

    /// A saved microphone that comes back renumbered is followed: nothing happens while the hub
    /// captures; when the hub starts reconnecting, the loss snapshot is taken at once (off the worker
    /// thread); after [`RENUMBER_AFTER`] the saved name is looked up again at most every
    /// [`RENUMBER_CHECK_EVERY`]. A device of the family that was there at the loss (a USB dongle next
    /// to the unplugged headset) is never followed; the headset under a new name is, in the run's
    /// format. The same name (the hub reconnects it itself) and the system default change nothing.
    #[test]
    fn a_renumbered_microphone_is_followed_while_the_hub_reconnects() {
        let headset = "Microphone (2- USB Audio)";
        let dongle = "Microphone (USB Audio)";
        let calls = Arc::new(AtomicUsize::new(0));
        // What Windows lists right now: the headset is unplugged, the dongle stays.
        let listed: Arc<Mutex<Vec<String>>> = Arc::new(Mutex::new(vec![dongle.to_string()]));
        let resolver: InputResolver = {
            let (calls, listed) = (calls.clone(), listed.clone());
            Arc::new(move |saved: &str, at_loss: Option<&[String]>| {
                assert_eq!(saved, headset);
                calls.fetch_add(1, Ordering::SeqCst);
                let names = listed.lock().unwrap().clone();
                let found = match_runtime_device(saved, &names, at_loss)
                    .map(|(i, _)| InputTarget {
                        spec: SourceSpec::Input(names[i].clone()),
                        name: names[i].clone(),
                        renamed_from: None,
                        rate: 44_100,
                        ch: 1,
                    })
                    .ok_or_else(|| "not connected".to_string());
                Lookup {
                    found,
                    family: family_members(saved, &names),
                }
            })
        };
        let current = InputTarget {
            spec: SourceSpec::Input(headset.into()),
            name: headset.into(),
            renamed_from: None,
            rate: 48_000,
            ch: 2,
        };
        let calls_now = || calls.load(Ordering::SeqCst);
        let settle = || std::thread::sleep(Duration::from_millis(20));
        let mut follow = InputFollow::with_resolver(&format!(" {headset} "), resolver.clone());
        let t0 = Instant::now();
        // Capturing: nothing to do.
        assert_eq!(follow.poll(false, &current, t0), None);
        settle();
        assert_eq!(calls_now(), 0);
        // The hub starts reconnecting: the loss snapshot (only the dongle is there).
        assert_eq!(follow.poll(true, &current, t0), None);
        wait_until("snapshot lookup", || calls_now() == 1);
        settle();
        assert_eq!(follow.poll(true, &current, t0 + RENUMBER_AFTER / 2), None);
        settle();
        assert_eq!(calls_now(), 1, "no lookup before RENUMBER_AFTER");
        // Long enough: a lookup runs; the dongle was there at the loss, so it is not followed.
        let mut now = t0 + RENUMBER_AFTER;
        assert_eq!(follow.poll(true, &current, now), None);
        wait_until("second lookup", || calls_now() == 2);
        settle();
        assert_eq!(follow.poll(true, &current, now), None, "a device present at the loss never stands in");
        // Not before the next check is due.
        now += RENUMBER_CHECK_EVERY / 2;
        assert_eq!(follow.poll(true, &current, now), None);
        settle();
        assert_eq!(calls_now(), 2);
        // It came back under its old name: the hub reconnects it itself.
        listed.lock().unwrap().push(headset.to_string());
        now += RENUMBER_CHECK_EVERY;
        assert_eq!(follow.poll(true, &current, now), None);
        wait_until("third lookup", || calls_now() == 3);
        settle();
        assert_eq!(follow.poll(true, &current, now), None, "same name: nothing to follow");
        // Renumbered on another port (and another format): follow it, in the run's format.
        *listed.lock().unwrap() = vec![dongle.to_string(), "Microphone (3- USB Audio)".to_string()];
        now += RENUMBER_CHECK_EVERY;
        assert_eq!(follow.poll(true, &current, now), None);
        wait_until("fourth lookup", || calls_now() == 4);
        let found = loop {
            if let Some(t) = follow.poll(true, &current, now) {
                break t;
            }
            std::thread::sleep(Duration::from_millis(2));
        };
        assert_eq!(found.spec, SourceSpec::Input("Microphone (3- USB Audio)".into()));
        assert_eq!((found.rate, found.ch), (48_000, 2), "the run's format");
        assert_eq!(found.renamed_from.as_deref(), Some(headset));
        // Capturing again ends the loss; a new one takes a new snapshot at once.
        assert_eq!(follow.poll(false, &current, now), None);
        assert_eq!(follow.poll(true, &current, now + RENUMBER_CHECK_EVERY), None);
        wait_until("a new snapshot", || calls_now() == 5);
        // The system default is the hub's to follow.
        let mut default = InputFollow::with_resolver("", resolver);
        assert_eq!(default.poll(true, &current, t0 + Duration::from_secs(60)), None);
        settle();
        assert_eq!(calls_now(), 5);
    }

    /// Moving the capture to another source keeps the run going: silence while it opens, then the
    /// new source's audio, spliced in after a gap.
    #[test]
    fn a_reattached_capture_splices_the_new_source_in() {
        let mic = ScriptedMic::new(STEREO_48K);
        let hub = test_hub(&mic);
        let mut vc = HubInput::attach(&hub, &target(STEREO_48K), Box::new(|| false)).unwrap();
        wait_until("stream live", || mic.live());
        let mut got = Recorded::default();
        assert!(mic.push(&tone(480, 2, 0)));
        vc.pump(&mut got, now(), 480);
        // The capture stalls: silence is synthesized, after a gap.
        let t = now() + 300_000_000;
        let intake = vc.pump(&mut got, t, 480);
        assert!(intake.filled > 0 && vc.filling(), "{intake:?}");
        let renamed = InputTarget {
            spec: SourceSpec::Input("Mic A".into()),
            ..target(STEREO_48K)
        };
        vc.reattach(&hub, &renamed).unwrap();
        wait_until("new stream live", || mic.live());
        let gaps = got.gaps.len();
        assert!(mic.push(&tone(480, 2, 480)));
        let before = got.audio.len();
        let intake = vc.pump(&mut got, now(), 480);
        assert_eq!(intake.frames, 480, "{intake:?}");
        assert!(!vc.filling());
        assert_eq!(&got.audio[got.audio.len() - 960..], &tone(480, 2, 480)[..]);
        assert!(got.audio.len() >= before + 960);
        assert!(got.gaps.len() > gaps, "the new source is spliced in");
    }

    /// The voice-changer mute gate: nothing leaves the hub, the capture fills silence at once (so
    /// TTS routed into the voice changer keeps playing), and unmuting splices the audio back in.
    #[test]
    fn a_muted_capture_fills_silence() {
        let mic = ScriptedMic::new(STEREO_48K);
        let hub = Arc::new(test_hub(&mic));
        let h2 = hub.clone();
        let mut vc = HubInput::attach(&hub, &target(STEREO_48K), Box::new(move || h2.is_muted(MuteGroup::VoiceChanger))).unwrap();
        wait_until("stream live", || mic.live());
        let mut got = Recorded::default();
        mic.push(&tone(480, 2, 0));
        vc.pump(&mut got, now(), 480);
        hub.set_muted(MuteGroup::VoiceChanger, true);
        mic.push(&tone(480, 2, 480));
        let t0 = now();
        vc.pump(&mut got, t0, 480);
        std::thread::sleep(Duration::from_millis(40));
        let intake = vc.pump(&mut got, now(), 480);
        assert!(intake.filled >= 480, "{intake:?}");
        assert!(got.audio[960..].iter().all(|&s| s == 0.0), "muted audio never arrives");
        hub.set_muted(MuteGroup::VoiceChanger, false);
        mic.push(&tone(480, 2, 960));
        let before = got.audio.len();
        let intake = vc.pump(&mut got, now(), 480);
        assert_eq!(intake.frames, 480);
        assert_eq!(&got.audio[before..], &tone(480, 2, 960)[..]);
        assert!(got.gaps.iter().any(|&(at, _)| at == before), "{:?}", got.gaps);
    }
}
