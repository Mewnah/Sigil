//! The reference recorder: one take at a time, captured from the capture hub at the device rate
//! (a consumer of its own: the voice changer's and speech-to-text's feeds are untouched).
//!
//! A take runs on its own thread ("sigil-voice-record"): it waits on the hub consumer (park /
//! unpark, no sleep-polling), downmixes to mono, runs Silero over a streaming 16 kHz copy for the
//! live "speech heard" count, and reports [`RecordProgress`] at most every 50 ms. It stops by itself
//! at the requested length, when the device changes format, when the audio is interrupted (a
//! stream restart or a backlog drop would put a seam into the clip), when the source ends, or when
//! no audio arrives at all; the last progress event then says `stopped` and why.
//!
//! Epoch discipline: every start and stop bumps the recorder's epoch under its lock; the thread
//! emits progress only while holding that lock with its own epoch current, so after `stop` /
//! `cancel` returns no event of that take is emitted.

use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Mutex, MutexGuard, PoisonError};
use std::thread::JoinHandle;
use std::time::{Duration, Instant};

use super::dsp;
use super::process::Vad;
use super::RecordProgress;

/// Progress events at most this often.
pub const PROGRESS_INTERVAL: Duration = Duration::from_millis(50);
/// Longest take (seconds).
pub const MAX_SECONDS: f32 = 30.0;
/// Shortest `max_seconds` accepted.
pub const MIN_SECONDS: f32 = 3.0;
/// Default `max_seconds`.
pub const DEFAULT_SECONDS: f32 = 12.0;
/// Without any audio for this long after the start, the take ends (`no_audio`).
const NO_AUDIO_TIMEOUT: Duration = Duration::from_secs(4);
/// The wall clock may run this far past `max_seconds` (a source delivering slower than real time).
const WALL_CLOCK_GRACE: Duration = Duration::from_secs(3);

/// Where progress reports go (the Tauri event in the app).
pub type ProgressSink = Arc<dyn Fn(&RecordProgress) + Send + Sync>;
/// Makes the live speech detector on the recording thread (`None`: no live speech count).
pub type VadFactory = Box<dyn FnOnce() -> Option<Box<dyn Vad + Send>> + Send>;

/// Why a take stopped by itself (`RecordProgress::stop_reason`).
pub const STOP_MAX_LENGTH: &str = "max_length";
pub const STOP_DEVICE_CHANGED: &str = "device_changed";
pub const STOP_INTERRUPTED: &str = "interrupted";
pub const STOP_DEVICE_LOST: &str = "device_lost";
pub const STOP_NO_AUDIO: &str = "no_audio";

/// `max_seconds` as used: finite, within [`MIN_SECONDS`]..=[`MAX_SECONDS`], default
/// [`DEFAULT_SECONDS`] for zero / negative / non-finite values.
pub fn sanitize_max_seconds(v: f32) -> f32 {
    if !v.is_finite() || v <= 0.0 {
        DEFAULT_SECONDS
    } else {
        v.clamp(MIN_SECONDS, MAX_SECONDS)
    }
}

/// What a source delivered in one [`CaptureSource::next`].
#[derive(Clone, Debug, Default, PartialEq)]
pub struct SourceRead {
    /// `(sample rate, channels)` of the samples appended (`None` when nothing was appended).
    pub format: Option<(u32, u16)>,
    /// The audio is discontinuous after the appended samples.
    pub gap: bool,
    /// The source is over (the error text).
    pub ended: Option<String>,
}

/// Where a take's audio comes from (the hub in the app; a synthetic source in tests).
pub trait CaptureSource: Send {
    /// Wait up to `timeout` for audio, then append everything available (interleaved) to `out`.
    fn next(&mut self, out: &mut Vec<f32>, timeout: Duration) -> SourceRead;
}

/// The hub consumer of a take: the device's own audio at its own rate and channels (the voice
/// changer's denoiser and gain never touch it), no mute group (the user asked for this recording),
/// woken every 20 ms. A stall of more than 2 s drops audio, which ends the take as interrupted.
pub fn consumer_options() -> crate::services::audio::hub::ConsumerOptions {
    crate::services::audio::hub::ConsumerOptions {
        format: crate::services::audio::hub::ConsumerFormat::DeviceRate,
        max_backlog_ms: 2_000,
        wake_ms: 20,
        mute_group: None,
        label: "voice-library",
    }
}

/// The capture hub as a [`CaptureSource`].
pub struct HubSource(pub crate::services::audio::hub::CaptureConsumer);

impl CaptureSource for HubSource {
    fn next(&mut self, out: &mut Vec<f32>, timeout: Duration) -> SourceRead {
        self.0.wait(timeout);
        let info = self.0.read(out);
        let ended = if info.samples_out == 0 { self.0.ended() } else { None };
        SourceRead {
            format: info.format.filter(|_| info.samples_out > 0).map(|f| (f.sample_rate, f.channels)),
            gap: info.gap_after,
            ended,
        }
    }
}

/// A finished take's audio.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct Captured {
    pub mono: Vec<f32>,
    pub rate: u32,
    /// Why it stopped by itself (`None`: stopped by the user).
    pub stop_reason: Option<&'static str>,
    /// The source's error when it ended.
    pub error: Option<String>,
}

/// Capture until stopped (see the module docs). `emit` gets every progress report; `vad` (Silero
/// in the app) counts the speech heard.
pub fn capture(
    source: &mut dyn CaptureSource,
    max_seconds: f32,
    stop: &AtomicBool,
    vad: Option<&mut dyn Vad>,
    emit: &dyn Fn(&RecordProgress),
) -> Captured {
    let started = Instant::now();
    let wall_limit = Duration::from_secs_f32(max_seconds) + WALL_CLOCK_GRACE;
    let mut buf: Vec<f32> = Vec::with_capacity(16_384);
    let mut mono: Vec<f32> = Vec::new();
    let mut format: Option<(u32, u16)> = None;
    let mut max_frames = usize::MAX;
    let mut resampler: Option<crate::services::speech_core::resample::StreamResampler> = None;
    let mut vad = vad;
    let mut pending16k: Vec<f32> = Vec::new();
    let mut voiced_frames: u64 = 0;
    let mut clipping = false;
    let mut window_sq = 0.0f64;
    let mut window_n = 0usize;
    let mut window_peak = 0.0f32;
    let mut last_emit = Instant::now() - PROGRESS_INTERVAL;
    let mut stop_reason: Option<&'static str> = None;
    let mut error: Option<String> = None;
    const VAD_N: usize = crate::services::speech_core::VAD_FRAME_SAMPLES;

    let progress = |mono_len: usize, rate: u32, voiced: u64, sq: f64, n: usize, peak: f32, clip: bool, reason: Option<&'static str>| RecordProgress {
        elapsed_ms: if rate > 0 { mono_len as u64 * 1000 / u64::from(rate) } else { 0 },
        level_db: dsp::power_to_db(if n > 0 { sq / n as f64 } else { 0.0 }),
        peak_db: dsp::amp_to_db(f64::from(peak)),
        clipping: clip,
        voiced_ms: (voiced as f64 * super::trim::FRAME_SECONDS * 1000.0) as u64,
        stopped: reason.is_some(),
        stop_reason: reason.map(str::to_string),
    };

    loop {
        if stop.load(Ordering::Acquire) {
            break;
        }
        buf.clear();
        let read = source.next(&mut buf, Duration::from_millis(25));
        if let Some(f) = read.format {
            match format {
                None => {
                    format = Some(f);
                    max_frames = (f64::from(max_seconds) * f64::from(f.0)) as usize;
                    mono.reserve(max_frames.min(1 << 24));
                    resampler = crate::services::speech_core::resample::StreamResampler::new(f.0, 1).ok();
                }
                Some(old) if old != f => {
                    stop_reason = Some(STOP_DEVICE_CHANGED);
                    break;
                }
                _ => {}
            }
        }
        if let (Some((_, ch)), false) = (format, buf.is_empty()) {
            let ch = usize::from(ch.max(1));
            if !clipping {
                clipping = buf.iter().any(|v| v.abs() >= dsp::CLIP_THRESHOLD);
            }
            let start = mono.len();
            dsp::downmix_into(&buf, ch, &mut mono);
            if mono.len() > max_frames {
                mono.truncate(max_frames);
            }
            let fresh = &mono[start..];
            for &v in fresh {
                window_sq += f64::from(v) * f64::from(v);
                window_peak = window_peak.max(v.abs());
            }
            window_n += fresh.len();
            if let (Some(r), Some(vad)) = (resampler.as_mut(), vad.as_deref_mut()) {
                if r.push_interleaved(fresh, &mut pending16k).is_ok() {
                    let whole = pending16k.len() / VAD_N * VAD_N;
                    for frame in pending16k[..whole].as_chunks::<VAD_N>().0 {
                        if vad.prob(frame).unwrap_or(0.0) >= 0.5 {
                            voiced_frames += 1;
                        }
                    }
                    pending16k.drain(..whole);
                }
            }
            if mono.len() >= max_frames {
                stop_reason = Some(STOP_MAX_LENGTH);
            }
        }
        if stop_reason.is_none() && read.gap && !mono.is_empty() {
            stop_reason = Some(STOP_INTERRUPTED);
        }
        if stop_reason.is_none() {
            if let Some(e) = read.ended {
                error = Some(e);
                stop_reason = Some(STOP_DEVICE_LOST);
            }
        }
        if stop_reason.is_none() {
            let waited = started.elapsed();
            if mono.is_empty() && waited >= NO_AUDIO_TIMEOUT {
                stop_reason = Some(STOP_NO_AUDIO);
            } else if waited >= wall_limit {
                stop_reason = Some(STOP_MAX_LENGTH);
            }
        }
        if stop_reason.is_some() {
            break;
        }
        if last_emit.elapsed() >= PROGRESS_INTERVAL {
            let rate = format.map_or(0, |f| f.0);
            emit(&progress(mono.len(), rate, voiced_frames, window_sq, window_n, window_peak, clipping, None));
            last_emit = Instant::now();
            window_sq = 0.0;
            window_n = 0;
            window_peak = 0.0;
        }
    }
    let rate = format.map_or(0, |f| f.0);
    if stop_reason.is_some() {
        emit(&progress(mono.len(), rate, voiced_frames, window_sq, window_n, window_peak, clipping, stop_reason));
    }
    Captured { mono, rate, stop_reason, error }
}

// ---------------------------------------------------------------------------------------------
// The recorder (one take at a time)
// ---------------------------------------------------------------------------------------------

/// A running take.
pub struct Session {
    stop: Arc<AtomicBool>,
    handle: Option<JoinHandle<Captured>>,
}

impl Session {
    /// Stop the take and wait for its audio (blocking; the thread ends within one wait period).
    pub fn finish(mut self) -> Captured {
        self.stop.store(true, Ordering::Release);
        self.handle
            .take()
            .and_then(|h| h.join().ok())
            .unwrap_or_default()
    }
}

impl Drop for Session {
    fn drop(&mut self) {
        // A session dropped without `finish` still stops its thread (and with it the consumer).
        self.stop.store(true, Ordering::Release);
    }
}

#[derive(Default)]
struct State {
    epoch: u64,
    session: Option<Session>,
}

/// The recorder: epoch and the running session.
pub struct Recorder {
    state: Mutex<State>,
}

/// The plugin's recorder.
pub static RECORDER: Recorder = Recorder::new();

impl Default for Recorder {
    fn default() -> Self {
        Recorder::new()
    }
}

impl Recorder {
    pub const fn new() -> Recorder {
        Recorder { state: Mutex::new(State { epoch: 0, session: None }) }
    }

    fn lock(&self) -> MutexGuard<'_, State> {
        self.state.lock().unwrap_or_else(PoisonError::into_inner)
    }

    /// A new epoch for a take about to start; returns it with the session it replaces (stopped:
    /// its progress is silenced at once; `finish` it off the async runtime).
    pub fn begin(&self) -> (u64, Option<Session>) {
        let mut st = self.lock();
        st.epoch += 1;
        let old = st.session.take();
        if let Some(s) = &old {
            s.stop.store(true, Ordering::Release);
        }
        (st.epoch, old)
    }

    /// End the current take (stop or cancel): bumps the epoch (no more progress) and hands back
    /// the session to `finish`. `None` when nothing is recording.
    pub fn end(&self) -> Option<Session> {
        let mut st = self.lock();
        st.epoch += 1;
        let s = st.session.take();
        if let Some(s) = &s {
            s.stop.store(true, Ordering::Release);
        }
        s
    }

    /// Whether a take is recording (or stopped by itself and waiting for `end`).
    #[cfg(test)]
    pub fn active(&self) -> bool {
        self.lock().session.is_some()
    }

    /// Start the take of `epoch` on its own thread, unless another start, stop or cancel came
    /// since [`Recorder::begin`] (then the source is dropped and this errors).
    pub fn start(
        &'static self,
        epoch: u64,
        mut source: Box<dyn CaptureSource>,
        max_seconds: f32,
        sink: ProgressSink,
        make_vad: VadFactory,
    ) -> Result<(), String> {
        let mut st = self.lock();
        if st.epoch != epoch {
            return Err("the recording was cancelled".into());
        }
        let stop = Arc::new(AtomicBool::new(false));
        let thread_stop = Arc::clone(&stop);
        let handle = std::thread::Builder::new()
            .name("sigil-voice-record".into())
            .spawn(move || {
                let mut vad = make_vad();
                let emit = |p: &RecordProgress| {
                    let st = self.lock();
                    if st.epoch == epoch {
                        sink(p);
                    }
                };
                let vad_ref: Option<&mut dyn Vad> = vad.as_mut().map(|v| v.as_mut() as &mut dyn Vad);
                capture(source.as_mut(), max_seconds, &thread_stop, vad_ref, &emit)
            })
            .map_err(|e| format!("cannot start the recorder: {e}"))?;
        st.session = Some(Session { stop, handle: Some(handle) });
        Ok(())
    }
}

#[cfg(test)]
pub(crate) mod tests {
    use super::*;
    use crate::services::audio::hub::{
        now_qpc_ns, CaptureBackend, CaptureHub, CaptureStream, ErrorSink, HubConfig, ResolvedDevice, SourceSpec, StreamFormat, StreamSink,
    };
    use std::sync::atomic::{AtomicU64, AtomicUsize};

    // ---- A capture hub over one fake microphone (the test plays its audio callback) ------------

    /// The fake microphone's format: 48 kHz stereo, like most USB headsets.
    const MIC: StreamFormat = StreamFormat { sample_rate: 48_000, channels: 2 };

    /// The open stream: its generation, the hub's sink (the audio callback's end) and error sink.
    type Live = Arc<Mutex<(u64, Option<StreamSink>, Option<ErrorSink>)>>;

    /// One microphone behind the real [`CaptureHub`].
    #[derive(Clone, Default)]
    pub(crate) struct OneMic {
        live: Live,
        opens: Arc<AtomicUsize>,
    }

    struct MicStream {
        live: Live,
        generation: u64,
    }

    impl CaptureStream for MicStream {
        fn stop(self: Box<Self>, released: Option<std::sync::mpsc::Sender<()>>) {
            let mut live = self.live.lock().unwrap();
            if live.0 == self.generation {
                live.1 = None;
                live.2 = None;
            }
            if let Some(r) = released {
                let _ = r.send(());
            }
        }
    }

    impl CaptureBackend for OneMic {
        fn resolve(&self, spec: &SourceSpec) -> Result<ResolvedDevice, String> {
            if spec.is_loopback() {
                return Err("no loopback here".into());
            }
            Ok(ResolvedDevice { name: "Mic".into(), loopback: false, format: MIC, handle: None })
        }

        fn open(&self, _device: ResolvedDevice, sink: StreamSink, errors: ErrorSink) -> Result<Box<dyn CaptureStream>, String> {
            let mut live = self.live.lock().unwrap();
            live.0 += 1;
            live.1 = Some(sink);
            live.2 = Some(errors);
            self.opens.fetch_add(1, Ordering::SeqCst);
            Ok(Box::new(MicStream { live: Arc::clone(&self.live), generation: live.0 }))
        }
    }

    impl OneMic {
        /// A hub over this microphone (production layout, fast supervision).
        pub(crate) fn hub(&self) -> CaptureHub {
            let config = HubConfig {
                tick: Duration::from_millis(10),
                stall_timeout: Duration::ZERO,
                first_data_timeout: Duration::ZERO,
                retry_min: Duration::from_millis(20),
                retry_max: Duration::from_millis(200),
                attach_timeout: Duration::from_secs(10),
                ..HubConfig::default()
            };
            CaptureHub::new(Arc::new(self.clone()), config).unwrap()
        }

        /// One device buffer (interleaved stereo). Whether a stream took it.
        pub(crate) fn push(&self, data: &[f32]) -> bool {
            match self.live.lock().unwrap().1.as_mut() {
                Some(sink) => {
                    sink.push(data, now_qpc_ns());
                    true
                }
                None => false,
            }
        }

        /// Play `mono` as the device would (10 ms stereo buffers, both channels equal), faster than
        /// real time but never more than half a second ahead of what the recorder reported
        /// (`recorded_ms`, from its progress events), so a test machine under load never makes the
        /// consumer fall 2 s behind (which would end the take as interrupted).
        pub(crate) fn play(&self, mono: &[f32], recorded_ms: &AtomicU64) {
            const AHEAD_MS: u64 = 500;
            let deadline = Instant::now() + Duration::from_secs(60);
            for (i, chunk) in mono.chunks(480).enumerate() {
                while (i as u64) * 10 > recorded_ms.load(Ordering::Acquire) + AHEAD_MS {
                    assert!(Instant::now() < deadline, "the recorder stopped reading");
                    std::thread::sleep(Duration::from_millis(2));
                }
                let stereo: Vec<f32> = chunk.iter().flat_map(|&v| [v, v]).collect();
                assert!(self.push(&stereo), "no live stream");
                if i % 5 == 4 {
                    std::thread::sleep(Duration::from_millis(5));
                }
            }
        }

        /// The driver kills the stream (cpal reports the error).
        pub(crate) fn fail(&self, message: &str) {
            let errors = {
                let mut live = self.live.lock().unwrap();
                live.1 = None;
                live.2.take()
            };
            if let Some(mut e) = errors {
                e.report(message.into());
            }
        }

        pub(crate) fn opens(&self) -> usize {
            self.opens.load(Ordering::SeqCst)
        }
    }

    /// Through the real hub with the app's consumer options: every sample of a take arrives in
    /// order at the device rate, and the recorder stops exactly at its length.
    #[test]
    fn a_take_through_the_capture_hub_is_complete_and_exact() {
        let mic = OneMic::default();
        let hub = mic.hub();
        let consumer = hub.attach(SourceSpec::DefaultInput, consumer_options()).unwrap();
        let audio: Vec<f32> = (0..96_000).map(|i| ((i % 4_800) as f32 / 4_800.0 - 0.5) * 0.4).collect();
        let recorded_ms = Arc::new(AtomicU64::new(0));
        let player = {
            let (mic, audio, recorded_ms) = (mic.clone(), audio.clone(), Arc::clone(&recorded_ms));
            std::thread::spawn(move || mic.play(&audio, &recorded_ms))
        };
        let stop = AtomicBool::new(false);
        let events = Mutex::new(Vec::new());
        let got = capture(&mut HubSource(consumer), 2.0, &stop, None, &|p| {
            recorded_ms.store(p.elapsed_ms, Ordering::Release);
            events.lock().unwrap().push(p.clone());
        });
        player.join().unwrap();
        assert_eq!((got.rate, got.stop_reason, got.error.as_deref()), (48_000, Some(STOP_MAX_LENGTH), None));
        assert_eq!(got.mono, audio, "the downmixed take is the played audio, sample for sample");
        let last = events.into_inner().unwrap().pop().unwrap();
        assert!(last.stopped && last.elapsed_ms == 2_000, "{last:?}");
    }

    /// The real default microphone through a cpal-backed hub, with the real Silero VAD: three
    /// seconds arrive at the device rate and go through the take pipeline (nothing is stored; with
    /// nobody speaking the take is `no_speech`). `cargo test real_microphone -- --ignored --nocapture`.
    #[test]
    #[ignore = "opens the real default microphone for 3 s"]
    fn real_microphone_take_through_the_hub() {
        let hub = CaptureHub::new(Arc::new(crate::services::audio::stream::CpalBackend), HubConfig::default()).unwrap();
        let consumer = hub.attach(SourceSpec::DefaultInput, consumer_options()).expect("a default microphone");
        let stop = AtomicBool::new(false);
        let mut vad = super::super::process::silero().unwrap();
        let last = Mutex::new(None);
        let got = capture(&mut HubSource(consumer), 3.0, &stop, Some(&mut vad), &|p| *last.lock().unwrap() = Some(p.clone()));
        eprintln!("rate {} Hz, {} samples, stop {:?}, last {:?}", got.rate, got.mono.len(), got.stop_reason, last.lock().unwrap());
        assert_eq!(got.stop_reason, Some(STOP_MAX_LENGTH), "{:?}", got.error);
        assert!(got.rate >= 8_000 && got.mono.len() == (3.0 * got.rate as f64) as usize);
        match super::super::process::process_take(got.mono, got.rate, &mut vad) {
            Ok(p) => eprintln!("{:?}", p.quality),
            // A hands-free Bluetooth microphone (8 kHz) is refused by name.
            Err(e) => eprintln!("refused: {e}"),
        }
    }

    /// A stream restart (driver reset, device moved) puts a seam into the audio: the take ends as
    /// interrupted with everything captured before it.
    #[test]
    fn a_stream_restart_ends_the_take_as_interrupted() {
        let mic = OneMic::default();
        let hub = mic.hub();
        let consumer = hub.attach(SourceSpec::DefaultInput, consumer_options()).unwrap();
        // Speech-to-text listens to the same microphone, so the hub reopens the stream even after
        // the recorder has let go of it.
        let _stt = hub
            .attach(SourceSpec::DefaultInput, crate::services::audio::hub::ConsumerOptions::device_rate("stt"))
            .unwrap();
        let player = {
            let mic = mic.clone();
            std::thread::spawn(move || {
                // Half a second: never more than the pacing allows ahead.
                mic.play(&vec![0.25; 24_000], &AtomicU64::new(0));
                mic.fail("The device was reset (AUDCLNT_E_DEVICE_INVALIDATED)");
                let deadline = Instant::now() + Duration::from_secs(10);
                while mic.opens() < 2 {
                    assert!(Instant::now() < deadline, "the hub did not reopen the stream");
                    std::thread::sleep(Duration::from_millis(2));
                }
                // Audio after the restart (never part of the take).
                for _ in 0..50 {
                    if mic.push(&[0.5; 960]) {
                        std::thread::sleep(Duration::from_millis(5));
                    }
                }
            })
        };
        let stop = AtomicBool::new(false);
        let got = capture(&mut HubSource(consumer), 10.0, &stop, None, &|_| {});
        player.join().unwrap();
        assert_eq!(got.stop_reason, Some(STOP_INTERRUPTED));
        assert_eq!(got.mono.len(), 24_000, "the audio before the seam is kept whole");
        assert!(got.mono.iter().all(|&v| v == 0.25), "nothing from after the restart");
    }

    /// A synthetic source: `chunks` of interleaved audio, then optional events.
    struct Fake {
        chunks: Vec<(Vec<f32>, (u32, u16), bool)>,
        ended: Option<String>,
        at: usize,
        /// Delay per read (emulates real time).
        delay: Duration,
    }

    impl CaptureSource for Fake {
        fn next(&mut self, out: &mut Vec<f32>, timeout: Duration) -> SourceRead {
            if self.at >= self.chunks.len() {
                std::thread::sleep(timeout.min(Duration::from_millis(5)));
                return SourceRead { ended: self.ended.clone(), ..SourceRead::default() };
            }
            std::thread::sleep(self.delay);
            let (data, format, gap) = self.chunks[self.at].clone();
            self.at += 1;
            out.extend_from_slice(&data);
            SourceRead { format: Some(format), gap, ended: None }
        }
    }

    fn tone(frames: usize, ch: usize, amp: f32) -> Vec<f32> {
        (0..frames).flat_map(|i| std::iter::repeat_n(amp * (i as f32 * 0.06).sin(), ch)).collect()
    }

    struct Always(f32);
    impl Vad for Always {
        fn prob(&mut self, _: &[f32]) -> Result<f32, String> {
            Ok(self.0)
        }
        fn reset(&mut self) {}
    }

    #[test]
    fn a_take_stops_at_its_length_with_progress() {
        // 48 kHz stereo in 10 ms blocks, 5 s offered, 3 s requested.
        let chunks = (0..500).map(|_| (tone(480, 2, 0.5), (48_000, 2), false)).collect();
        let mut src = Fake { chunks, ended: None, at: 0, delay: Duration::ZERO };
        let events = Mutex::new(Vec::new());
        let stop = AtomicBool::new(false);
        let mut vad = Always(0.9);
        let got = capture(&mut src, 3.0, &stop, Some(&mut vad), &|p| events.lock().unwrap().push(p.clone()));
        assert_eq!((got.rate, got.mono.len(), got.stop_reason), (48_000, 144_000, Some(STOP_MAX_LENGTH)));
        let events = events.into_inner().unwrap();
        let last = events.last().unwrap();
        assert!(last.stopped && last.stop_reason.as_deref() == Some(STOP_MAX_LENGTH));
        assert_eq!(last.elapsed_ms, 3_000);
        assert!(last.voiced_ms > 2_800, "{}", last.voiced_ms);
        assert!(!last.clipping);
        assert!(events.iter().rev().skip(1).all(|p| !p.stopped));
    }

    #[test]
    fn format_changes_gaps_ends_and_silence_stop_a_take() {
        let stop = AtomicBool::new(false);
        let none = |_: &RecordProgress| {};
        let mut src = Fake {
            chunks: vec![(tone(480, 1, 0.1), (48_000, 1), false), (tone(441, 1, 0.1), (44_100, 1), false)],
            ended: None,
            at: 0,
            delay: Duration::ZERO,
        };
        let got = capture(&mut src, 10.0, &stop, None, &none);
        assert_eq!((got.mono.len(), got.stop_reason), (480, Some(STOP_DEVICE_CHANGED)));

        let mut src = Fake {
            chunks: vec![(tone(480, 1, 0.1), (48_000, 1), false), (tone(480, 1, 0.1), (48_000, 1), true)],
            ended: None,
            at: 0,
            delay: Duration::ZERO,
        };
        let got = capture(&mut src, 10.0, &stop, None, &none);
        assert_eq!((got.mono.len(), got.stop_reason), (960, Some(STOP_INTERRUPTED)));

        let mut src = Fake { chunks: vec![(tone(480, 1, 0.1), (48_000, 1), false)], ended: Some("unplugged".into()), at: 0, delay: Duration::ZERO };
        let got = capture(&mut src, 10.0, &stop, None, &none);
        assert_eq!((got.stop_reason, got.error.as_deref()), (Some(STOP_DEVICE_LOST), Some("unplugged")));

        let mut src = Fake { chunks: vec![], ended: None, at: 0, delay: Duration::ZERO };
        let t = Instant::now();
        let got = capture(&mut src, 10.0, &stop, None, &none);
        assert_eq!(got.stop_reason, Some(STOP_NO_AUDIO));
        assert!(t.elapsed() >= NO_AUDIO_TIMEOUT && t.elapsed() < NO_AUDIO_TIMEOUT + Duration::from_secs(2));
    }

    #[test]
    fn clipping_is_sticky_and_levels_are_reported() {
        let mut chunks: Vec<(Vec<f32>, (u32, u16), bool)> = vec![(vec![0.0, 1.0, 0.0, 0.2], (16_000, 2), false)];
        chunks.extend((0..60).map(|_| (tone(160, 2, 0.1), (16_000, 2), false)));
        let mut src = Fake { chunks, ended: None, at: 0, delay: Duration::from_millis(2) };
        let events = Mutex::new(Vec::new());
        let stop = AtomicBool::new(false);
        let got = capture(&mut src, 0.5, &stop, None, &|p| events.lock().unwrap().push(p.clone()));
        assert_eq!((got.stop_reason, got.mono.len()), (Some(STOP_MAX_LENGTH), 8_000));
        let events = events.into_inner().unwrap();
        assert!(!events.is_empty());
        assert!(events.iter().all(|p| p.clipping), "a clipped sample in one channel is reported for the rest of the take");
        let mid = &events[events.len() / 2];
        assert!(mid.level_db > -30.0 && mid.level_db < -15.0, "{}", mid.level_db);
        assert!(mid.peak_db > -21.0 && mid.peak_db < -19.0, "{}", mid.peak_db);
    }

    #[test]
    fn stop_and_cancel_silence_the_take_and_hand_back_its_audio() {
        static R: Recorder = Recorder::new();
        let sent = Arc::new(AtomicUsize::new(0));
        let s = Arc::clone(&sent);
        let (epoch, old) = R.begin();
        assert!(old.is_none());
        let chunks = (0..2_000).map(|_| (tone(160, 1, 0.1), (16_000, 1), false)).collect();
        let src = Fake { chunks, ended: None, at: 0, delay: Duration::from_millis(5) };
        let sink: ProgressSink = Arc::new(move |_: &RecordProgress| {
            s.fetch_add(1, Ordering::SeqCst);
        });
        let no_vad: VadFactory = Box::new(|| -> Option<Box<dyn Vad + Send>> { None });
        R.start(epoch, Box::new(src), 30.0, sink, no_vad).unwrap();
        assert!(R.active());
        std::thread::sleep(Duration::from_millis(300));
        let session = R.end().unwrap();
        let after_end = sent.load(Ordering::SeqCst);
        assert!(after_end > 0, "progress was reported while recording");
        let got = session.finish();
        assert!(!got.mono.is_empty() && got.stop_reason.is_none(), "stopped by the user");
        std::thread::sleep(Duration::from_millis(100));
        assert!(sent.load(Ordering::SeqCst) <= after_end + 1, "no progress after end");
        assert!(!R.active() && R.end().is_none());

        // A start that lost the race against a cancel is refused.
        let (epoch, _) = R.begin();
        let _ = R.end();
        let src = Fake { chunks: vec![], ended: None, at: 0, delay: Duration::ZERO };
        let sink: ProgressSink = Arc::new(|_: &RecordProgress| {});
        let no_vad: VadFactory = Box::new(|| -> Option<Box<dyn Vad + Send>> { None });
        assert!(R.start(epoch, Box::new(src), 5.0, sink, no_vad).is_err());
        assert!(!R.active());
    }

    #[test]
    fn max_seconds_is_sanitized() {
        assert_eq!(sanitize_max_seconds(0.0), DEFAULT_SECONDS);
        assert_eq!(sanitize_max_seconds(f32::NAN), DEFAULT_SECONDS);
        assert_eq!(sanitize_max_seconds(1.0), MIN_SECONDS);
        assert_eq!(sanitize_max_seconds(500.0), MAX_SECONDS);
        assert_eq!(sanitize_max_seconds(8.0), 8.0);
    }
}
