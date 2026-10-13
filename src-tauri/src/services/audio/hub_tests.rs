//! Capture hub tests over a scripted fake backend: the test thread plays the audio callback
//! (`StreamSink::push`) and the device driver (errors, unplug / replug, default changes).

use std::collections::HashSet;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::{mpsc, Arc, Mutex};
use std::time::{Duration, Instant};

use super::hub::{
    now_qpc_ns, CaptureBackend, CaptureConsumer, CaptureHub, CaptureStream, ConsumerOptions, ErrorSink, HubConfig,
    HubEventSink, MuteGroup, ResolvedDevice, SourceSpec, StreamEvent, StreamFormat, StreamSink,
    MAX_CONSUMERS_PER_STREAM,
};
use super::consumer::ReadInfo;
use super::device_resolve::match_saved_device;
use crate::services::speech_core::resample::StreamResampler;

const STEREO_48K: StreamFormat = StreamFormat {
    sample_rate: 48_000,
    channels: 2,
};
const MONO_44K: StreamFormat = StreamFormat {
    sample_rate: 44_100,
    channels: 1,
};
/// A 10 ms WASAPI period at 48 kHz.
const BLOCK: usize = 480;

// ---------------------------------------------------------------------------------------------
// Fake backend
// ---------------------------------------------------------------------------------------------

struct FakeDevice {
    name: &'static str,
    loopback: bool,
    format: StreamFormat,
    present: bool,
}

struct FakeEntry {
    key: String,
    sink: Option<StreamSink>,
    errors: Option<ErrorSink>,
}

struct FakeInner {
    devices: Mutex<Vec<FakeDevice>>,
    default_input: Mutex<Option<&'static str>>,
    default_output: Mutex<Option<&'static str>>,
    entries: Mutex<Vec<FakeEntry>>,
    opens: AtomicUsize,
    stops: AtomicUsize,
    refuse_open: Mutex<HashSet<String>>,
    /// How long `open` blocks (a slow IAudioClient Initialize/Start, e.g. Bluetooth hands-free).
    open_delay: Mutex<Duration>,
}

#[derive(Clone)]
struct FakeBackend(Arc<FakeInner>);

impl FakeBackend {
    fn new() -> Self {
        Self(Arc::new(FakeInner {
            devices: Mutex::new(vec![
                FakeDevice {
                    name: "Mic A",
                    loopback: false,
                    format: STEREO_48K,
                    present: true,
                },
                FakeDevice {
                    name: "Mic B",
                    loopback: false,
                    format: MONO_44K,
                    present: true,
                },
                FakeDevice {
                    name: "Speakers",
                    loopback: true,
                    format: STEREO_48K,
                    present: true,
                },
            ]),
            default_input: Mutex::new(Some("Mic A")),
            default_output: Mutex::new(Some("Speakers")),
            entries: Mutex::new(Vec::new()),
            opens: AtomicUsize::new(0),
            stops: AtomicUsize::new(0),
            refuse_open: Mutex::new(HashSet::new()),
            open_delay: Mutex::new(Duration::ZERO),
        }))
    }

    fn opens(&self) -> usize {
        self.0.opens.load(Ordering::SeqCst)
    }

    fn stops(&self) -> usize {
        self.0.stops.load(Ordering::SeqCst)
    }

    /// Play the audio callback of the live stream `key`. Returns whether a stream took it.
    fn push(&self, key: &str, data: &[f32]) -> bool {
        self.push_at(key, data, now_qpc_ns())
    }

    /// Like [`Self::push`] with the driver's capture stamp `capture_qpc_ns`.
    fn push_at(&self, key: &str, data: &[f32], capture_qpc_ns: i64) -> bool {
        let mut entries = self.0.entries.lock().unwrap();
        match entries.iter_mut().rev().find(|e| e.key == key && e.sink.is_some()) {
            Some(entry) => {
                entry.sink.as_mut().unwrap().push(data, capture_qpc_ns);
                true
            }
            None => false,
        }
    }

    /// Push one 10 ms block to `key` every 10 ms for `duration`.
    fn feed_for(&self, key: &str, duration: Duration) {
        let until = Instant::now() + duration;
        while Instant::now() < until {
            assert!(self.push(key, &vec![0.1; BLOCK * 2]), "no live stream {key}");
            std::thread::sleep(Duration::from_millis(10));
        }
    }

    /// The driver kills stream `key` (cpal reports the error, then its audio thread exits).
    fn fail(&self, key: &str, message: &str) {
        let mut entries = self.0.entries.lock().unwrap();
        if let Some(entry) = entries.iter_mut().rev().find(|e| e.key == key && e.sink.is_some()) {
            entry.sink = None;
            if let Some(mut errors) = entry.errors.take() {
                errors.report(message.to_string());
            }
        }
    }

    /// A new input device appears (a USB mic re-plugged into another port is listed under a new name).
    fn add_input(&self, name: &'static str) {
        self.0.devices.lock().unwrap().push(FakeDevice {
            name,
            loopback: false,
            format: STEREO_48K,
            present: true,
        });
    }

    fn set_present(&self, name: &str, present: bool) {
        for d in self.0.devices.lock().unwrap().iter_mut().filter(|d| d.name == name) {
            d.present = present;
        }
    }

    fn unplug(&self, name: &str) {
        self.set_present(name, false);
        self.fail(&format!("in:{name}"), "The device was unplugged (AUDCLNT_E_DEVICE_INVALIDATED)");
    }

    fn set_default_input(&self, name: Option<&'static str>) {
        *self.0.default_input.lock().unwrap() = name;
    }

    fn live_streams(&self) -> usize {
        self.0.entries.lock().unwrap().iter().filter(|e| e.sink.is_some()).count()
    }
}

struct FakeHandle {
    inner: Arc<FakeInner>,
    index: usize,
}

impl CaptureStream for FakeHandle {
    fn stop(self: Box<Self>, released: Option<mpsc::Sender<()>>) {
        let sink = self.inner.entries.lock().unwrap()[self.index].sink.take();
        drop(sink);
        self.inner.stops.fetch_add(1, Ordering::SeqCst);
        if let Some(released) = released {
            let _ = released.send(());
        }
    }
}

impl CaptureBackend for FakeBackend {
    fn resolve(&self, spec: &SourceSpec) -> Result<ResolvedDevice, String> {
        let (name, loopback) = match spec {
            SourceSpec::DefaultInput => (self.0.default_input.lock().unwrap().map(str::to_string), false),
            // Like the cpal backend: the exact name, else the one renumbered device of its family.
            SourceSpec::Input(n) => {
                let devices = self.0.devices.lock().unwrap();
                let names: Vec<String> = devices
                    .iter()
                    .filter(|d| !d.loopback && d.present)
                    .map(|d| d.name.to_string())
                    .collect();
                (Some(match_saved_device(n, &names).map_or_else(|| n.clone(), |(i, _)| names[i].clone())), false)
            }
            SourceSpec::DefaultLoopback => (self.0.default_output.lock().unwrap().map(str::to_string), true),
            SourceSpec::Loopback(n) => (Some(n.clone()), true),
            SourceSpec::Process(exe) => (Some(format!("process:{exe}")), true),
        };
        let name = name.ok_or("No default device")?;
        let devices = self.0.devices.lock().unwrap();
        let device = devices
            .iter()
            .find(|d| d.name == name && d.loopback == loopback && d.present)
            .ok_or_else(|| format!("Device '{name}' not found"))?;
        Ok(ResolvedDevice {
            name: device.name.to_string(),
            loopback,
            format: device.format,
            handle: None,
        })
    }

    fn input_device_names(&self) -> Option<Vec<String>> {
        let devices = self.0.devices.lock().unwrap();
        Some(devices.iter().filter(|d| !d.loopback && d.present).map(|d| d.name.to_string()).collect())
    }

    fn open(&self, device: ResolvedDevice, sink: StreamSink, errors: ErrorSink) -> Result<Box<dyn CaptureStream>, String> {
        let delay = *self.0.open_delay.lock().unwrap();
        if !delay.is_zero() {
            std::thread::sleep(delay);
        }
        let key = device.physical_key();
        if self.0.refuse_open.lock().unwrap().contains(&key) {
            return Err("exclusive mode".to_string());
        }
        let present = self
            .0
            .devices
            .lock()
            .unwrap()
            .iter()
            .any(|d| d.name == device.name && d.loopback == device.loopback && d.present);
        if !present {
            return Err(format!("Device '{}' not found", device.name));
        }
        self.0.opens.fetch_add(1, Ordering::SeqCst);
        let mut entries = self.0.entries.lock().unwrap();
        entries.push(FakeEntry {
            key,
            sink: Some(sink),
            errors: Some(errors),
        });
        Ok(Box::new(FakeHandle {
            inner: Arc::clone(&self.0),
            index: entries.len() - 1,
        }))
    }
}

#[derive(Default)]
struct Recorder(Mutex<Vec<(&'static str, StreamEvent)>>);

impl Recorder {
    fn events(&self) -> Vec<(&'static str, StreamEvent)> {
        self.0.lock().unwrap().clone()
    }

    fn count(&self, kind: &str) -> usize {
        self.events().iter().filter(|(k, _)| *k == kind).count()
    }
}

impl HubEventSink for Recorder {
    fn stream_error(&self, event: &StreamEvent) {
        self.0.lock().unwrap().push(("error", event.clone()));
    }

    fn stream_recovered(&self, event: &StreamEvent) {
        self.0.lock().unwrap().push(("recovered", event.clone()));
    }

    fn stream_closed(&self, event: &StreamEvent) {
        self.0.lock().unwrap().push(("closed", event.clone()));
    }
}

fn fast_config() -> HubConfig {
    HubConfig {
        tick: Duration::from_millis(10),
        stall_timeout: Duration::ZERO,
        first_data_timeout: Duration::ZERO,
        retry_min: Duration::from_millis(20),
        retry_max: Duration::from_millis(200),
        share_streams: true,
        supervise: true,
        attach_timeout: Duration::from_secs(10),
    }
}

fn hub_with(config: HubConfig) -> (CaptureHub, FakeBackend, Arc<Recorder>) {
    let fake = FakeBackend::new();
    let hub = CaptureHub::new(Arc::new(fake.clone()), config).unwrap();
    let recorder = Arc::new(Recorder::default());
    hub.set_event_sink(recorder.clone());
    (hub, fake, recorder)
}

fn wait_until(what: &str, mut cond: impl FnMut() -> bool) {
    let deadline = Instant::now() + Duration::from_secs(10);
    while !cond() {
        assert!(Instant::now() < deadline, "timed out waiting for: {what}");
        std::thread::sleep(Duration::from_millis(2));
    }
}

/// Read until nothing more arrives; returns the samples and every non-empty read's info.
fn read_all(c: &mut CaptureConsumer) -> (Vec<f32>, Vec<ReadInfo>) {
    let mut out = Vec::new();
    let mut infos = Vec::new();
    loop {
        let before = out.len();
        let info = c.read(&mut out);
        if out.len() == before && !info.gap_after && info.dropped_frames == 0 {
            break;
        }
        infos.push(info);
    }
    (out, infos)
}

/// One stereo block whose samples encode `(block, frame, channel)`.
fn tagged_block(block: usize) -> Vec<f32> {
    (0..BLOCK)
        .flat_map(|f| [block as f32 + f as f32 * 1e-4, -(block as f32) - f as f32 * 1e-4])
        .collect()
}

fn block_of(sample: f32) -> usize {
    sample.floor() as usize
}

// ---------------------------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------------------------

#[test]
fn fan_out_is_identical_and_one_stream_serves_one_physical_device() {
    let (hub, fake, _) = hub_with(fast_config());
    let mut by_default = hub.attach(SourceSpec::DefaultInput, ConsumerOptions::device_rate("a")).unwrap();
    let mut by_name = hub
        .attach(SourceSpec::Input("Mic A".into()), ConsumerOptions::device_rate("b"))
        .unwrap();
    assert_eq!(fake.opens(), 1, "\"default\" and \"Mic A\" are the same physical device");
    assert_ne!(by_default.epoch(), by_name.epoch(), "two sources, two epochs");
    assert_eq!((by_default.source(), by_name.source()), ("default", "Mic A"));

    let mut expected = Vec::new();
    for b in 0..20 {
        let block = tagged_block(b);
        assert!(fake.push("in:Mic A", &block));
        expected.extend_from_slice(&block);
    }
    let (a, infos) = read_all(&mut by_default);
    let (b, _) = read_all(&mut by_name);
    assert_eq!(a, expected);
    assert_eq!(b, expected, "every consumer gets the identical samples");
    assert_eq!(infos[0].format, Some(STEREO_48K));
    assert!(infos.iter().all(|i| !i.gap_after));

    let status = hub.status().unwrap();
    assert_eq!(status.streams.len(), 1);
    assert_eq!(status.streams[0].consumers, 2);
    assert_eq!(status.sources.len(), 2);
    assert!(status.sources.iter().all(|s| s.stream == "in:Mic A"));
}

#[test]
fn raw_16k_mono_output_is_exact_and_matches_the_speech_resampler() {
    let (hub, fake, _) = hub_with(fast_config());
    let mut one = hub.attach(SourceSpec::DefaultInput, ConsumerOptions::speech("stt-1")).unwrap();
    let mut two = hub.attach(SourceSpec::DefaultInput, ConsumerOptions::speech("stt-2")).unwrap();
    // One second of a 440 Hz tone at 48 kHz stereo, in 10 ms device blocks.
    let tone: Vec<f32> = (0..48_000)
        .flat_map(|i| {
            let s = 0.5 * (2.0 * std::f32::consts::PI * 440.0 * i as f32 / 48_000.0).sin();
            [s, s]
        })
        .collect();
    for block in tone.chunks(BLOCK * 2) {
        assert!(fake.push("in:Mic A", block));
    }
    let (mut out, infos) = read_all(&mut one);
    one.flush(&mut out);
    assert_eq!(out.len(), 16_000, "exactly 1 s of 16 kHz audio");
    assert!(infos.iter().all(|i| i.format == Some(StreamFormat::STT)));
    let (mut out2, _) = read_all(&mut two);
    two.flush(&mut out2);
    assert_eq!(out, out2, "identical 16 kHz fan-out");

    // Same result as the speech front end's resampler on the whole signal.
    let mut reference = Vec::new();
    let mut r = StreamResampler::new(48_000, 2).unwrap();
    r.push_interleaved(&tone, &mut reference).unwrap();
    r.flush(&mut reference).unwrap();
    assert_eq!(reference.len(), out.len());
    let max_diff = out.iter().zip(&reference).map(|(a, b)| (a - b).abs()).fold(0.0f32, f32::max);
    assert!(max_diff < 1e-5, "max diff {max_diff}");
    // Still a 440 Hz tone at 16 kHz (880 zero crossings per second), full amplitude.
    let crossings = out[1_000..15_000].windows(2).filter(|w| (w[0] < 0.0) != (w[1] < 0.0)).count();
    assert!((860..=900).contains(&(crossings * 16_000 / 14_000)), "{crossings} crossings");
    let peak = out[1_000..15_000].iter().fold(0.0f32, |m, s| m.max(s.abs()));
    assert!((0.45..0.55).contains(&peak), "peak {peak}");

    // A 44.1 kHz mono device: round(0.5 s * 16000) samples.
    let mut b = hub
        .attach(SourceSpec::Input("Mic B".into()), ConsumerOptions::speech("stt-b"))
        .unwrap();
    for block in vec![0.25f32; 22_050].chunks(441) {
        assert!(fake.push("in:Mic B", block));
    }
    let (mut out_b, _) = read_all(&mut b);
    b.flush(&mut out_b);
    assert_eq!(out_b.len(), 8_000);
}

#[test]
fn a_lagging_consumer_drops_its_own_oldest_audio_only() {
    let (hub, fake, _) = hub_with(fast_config());
    let mut lagging = hub
        .attach(SourceSpec::DefaultInput, ConsumerOptions::device_rate("lagging").with_max_backlog_ms(100))
        .unwrap();
    let mut steady = hub
        .attach(SourceSpec::DefaultInput, ConsumerOptions::device_rate("steady").with_max_backlog_ms(2_000))
        .unwrap();
    let mut all_steady = Vec::new();
    let mut block = 0;
    for _round in 0..3 {
        // 250 ms arrive between two reads of the lagging consumer (budget 100 ms).
        for _ in 0..25 {
            assert!(fake.push("in:Mic A", &tagged_block(block)));
            block += 1;
        }
        let (got, infos) = read_all(&mut lagging);
        let dropped: u64 = infos.iter().map(|i| i.dropped_frames).sum();
        assert_eq!(dropped, 15 * BLOCK as u64, "15 oldest blocks dropped");
        // What is left is the newest 100 ms, in order.
        let blocks: Vec<usize> = got.chunks(BLOCK * 2).map(|c| block_of(c[0])).collect();
        assert_eq!(blocks, ((block - 10)..block).collect::<Vec<_>>());
        let (s, _) = read_all(&mut steady);
        all_steady.extend(s);
    }
    // The steady consumer got every block, untouched by the other consumer's lag.
    assert_eq!(all_steady.len(), block * BLOCK * 2);
    assert_eq!(steady.dropped_frames() + steady.producer_dropped_frames(), 0);
    assert_eq!(lagging.producer_dropped_frames(), 0, "the producer never had to drop the newest audio");
    assert_eq!(lagging.dropped_frames(), 3 * 15 * BLOCK as u64);
}

#[test]
fn a_consumer_that_stops_reading_never_blocks_the_others() {
    let (hub, fake, _) = hub_with(fast_config());
    let stalled = hub
        .attach(SourceSpec::DefaultInput, ConsumerOptions::device_rate("stalled").with_max_backlog_ms(100))
        .unwrap();
    let mut steady = hub.attach(SourceSpec::DefaultInput, ConsumerOptions::device_rate("steady")).unwrap();
    let mut received = 0;
    for b in 0..200 {
        assert!(fake.push("in:Mic A", &tagged_block(b)));
        received += read_all(&mut steady).0.len();
    }
    assert_eq!(received, 200 * BLOCK * 2);
    assert!(stalled.producer_dropped_frames() > 0, "the full ring dropped (and counted) audio");
}

#[test]
fn stream_errors_restart_under_the_same_epoch() {
    let (hub, fake, events) = hub_with(fast_config());
    let mut c = hub.attach(SourceSpec::DefaultInput, ConsumerOptions::device_rate("x")).unwrap();
    let epoch = c.epoch();
    for b in 0..5 {
        fake.push("in:Mic A", &tagged_block(b));
    }
    assert_eq!(read_all(&mut c).0.len(), 5 * BLOCK * 2);

    fake.fail("in:Mic A", "driver reset");
    wait_until("stream_error", || events.count("error") == 1);
    wait_until("reopen", || fake.opens() == 2);
    assert_eq!(events.count("recovered"), 0, "recovered means audio flows, not just reopened");
    for b in 5..10 {
        assert!(fake.push("in:Mic A", &tagged_block(b)));
    }
    wait_until("stream_recovered", || events.count("recovered") == 1);
    let evs = events.events();
    assert_eq!(
        evs[0].1,
        StreamEvent {
            source: "default".into(),
            epoch,
            message: Some("driver reset".into())
        }
    );
    assert_eq!(
        evs[1].1,
        StreamEvent {
            source: "default".into(),
            epoch,
            message: None
        }
    );
    assert_eq!(c.epoch(), epoch);
    assert!(!c.is_recovering());

    let (after, infos) = read_all(&mut c);
    assert!(infos[0].gap_after && infos[0].restarted, "the restart is reported before new audio");
    assert_eq!(after.len(), 5 * BLOCK * 2);
    assert_eq!(block_of(after[0]), 5);
    let status = hub.status().unwrap();
    assert_eq!((status.streams[0].generation, status.streams[0].failures), (2, 1));
    assert_eq!(status.sources[0].epoch, epoch);
}

#[test]
fn unplug_then_replug_recovers_quickly_after_a_device_change() {
    let (hub, fake, events) = hub_with(HubConfig {
        // Slow backoff: recovery must come from the device-change nudge, not from polling.
        retry_min: Duration::from_millis(50),
        retry_max: Duration::from_secs(30),
        ..fast_config()
    });
    let mut c = hub
        .attach(SourceSpec::Input("Mic A".into()), ConsumerOptions::speech("stt"))
        .unwrap();
    fake.unplug("Mic A");
    wait_until("stream_error", || events.count("error") == 1);
    assert!(c.is_recovering());
    // Let the backoff grow while the device is gone (attempts at 50, 100, 200, 400 ms ...).
    std::thread::sleep(Duration::from_millis(500));
    assert_eq!(fake.opens(), 1, "nothing to open while unplugged");

    fake.set_present("Mic A", true);
    let replugged = Instant::now();
    hub.devices_changed();
    wait_until("reopened", || fake.opens() == 2);
    // 50 ms of audio (the 16 kHz resampler needs a full 1024-frame chunk before its first output).
    for _ in 0..5 {
        assert!(fake.push("in:Mic A", &vec![0.1; BLOCK * 2]));
    }
    wait_until("stream_recovered", || events.count("recovered") == 1);
    assert!(replugged.elapsed() < Duration::from_secs(2), "recovered in {:?}", replugged.elapsed());
    assert_eq!(events.events()[1].1.epoch, c.epoch());
    let (out, _) = read_all(&mut c);
    assert!(!out.is_empty());
}

#[test]
fn a_microphone_replugged_under_a_renumbered_name_is_followed_under_the_same_epoch() {
    // The mic moved to another USB port: Windows lists it as "(2- USB Audio)", not "(USB Audio)".
    let (hub, fake, events) = hub_with(fast_config());
    fake.add_input("Mic (USB Audio)");
    let mut c = hub
        .attach(SourceSpec::Input("Mic (USB Audio)".into()), ConsumerOptions::speech("stt"))
        .unwrap();
    let epoch = c.epoch();
    fake.unplug("Mic (USB Audio)");
    wait_until("stream_error", || events.count("error") == 1);
    // Other microphones never stand in for it, not even after a device change.
    hub.devices_changed();
    std::thread::sleep(Duration::from_millis(200));
    assert_eq!(fake.opens(), 1, "Mic A and Mic B are other families: nothing to open");
    assert!(c.is_recovering());

    fake.add_input("Mic (2- USB Audio)");
    hub.devices_changed();
    wait_until("reopened on the renamed endpoint", || fake.opens() == 2);
    for _ in 0..5 {
        assert!(fake.push("in:Mic (2- USB Audio)", &vec![0.1; BLOCK * 2]));
    }
    wait_until("stream_recovered", || events.count("recovered") == 1);
    assert_eq!(events.events()[1].1.epoch, epoch, "same source epoch: STT never restarts");
    assert_eq!(c.epoch(), epoch);
    assert!(!c.is_recovering());
    let (out, _) = read_all(&mut c);
    assert!(!out.is_empty());
}

#[test]
fn a_start_with_a_saved_name_that_was_renumbered_opens_the_one_device_of_its_family() {
    // The saved name is from before the mic moved to another USB port; nothing else may stand in for it.
    let (hub, fake, events) = hub_with(fast_config());
    fake.add_input("Mic (2- USB Audio)");
    let mut c = hub
        .attach(SourceSpec::Input("Mic (USB Audio)".into()), ConsumerOptions::speech("stt"))
        .unwrap();
    assert_eq!(c.source(), "Mic (USB Audio)", "events and status keep the saved name");
    assert_eq!(fake.opens(), 1);
    for _ in 0..5 {
        assert!(fake.push("in:Mic (2- USB Audio)", &vec![0.1; BLOCK * 2]));
    }
    assert!(!read_all(&mut c).0.is_empty());
    assert_eq!(events.count("error"), 0);

    // A name of no listed family still fails the start (never the system default, which is "Mic A" here).
    let err = hub
        .attach(SourceSpec::Input("Yeti Stereo Microphone".into()), ConsumerOptions::speech("other"))
        .err()
        .expect("an unknown microphone does not open");
    assert!(err.contains("not found"), "{err}");
    assert_eq!(fake.opens(), 1);
}

#[test]
fn two_candidates_of_the_family_are_not_guessed_between() {
    let (hub, fake, events) = hub_with(fast_config());
    fake.add_input("Mic (USB Audio)");
    let c = hub
        .attach(SourceSpec::Input("Mic (USB Audio)".into()), ConsumerOptions::speech("stt"))
        .unwrap();
    fake.unplug("Mic (USB Audio)");
    wait_until("stream_error", || events.count("error") == 1);
    fake.add_input("Mic (2- USB Audio)");
    fake.add_input("Mic (3- USB Audio)");
    hub.devices_changed();
    std::thread::sleep(Duration::from_millis(300));
    assert_eq!(fake.opens(), 1, "ambiguous: neither takes over");
    assert!(c.is_recovering());
    // The exact name returning wins again.
    fake.set_present("Mic (USB Audio)", true);
    hub.devices_changed();
    wait_until("reopened under the saved name", || fake.opens() == 2);
    assert!(fake.push("in:Mic (USB Audio)", &vec![0.1; BLOCK * 2]));
}

#[test]
fn a_second_identical_microphone_that_was_there_all_along_never_stands_in() {
    // Two USB devices list as the same family (a headset and a webcam). STT chose the headset.
    let (hub, fake, events) = hub_with(fast_config());
    fake.add_input("Mic (USB Audio)");
    fake.add_input("Mic (2- USB Audio)");
    let mut c = hub
        .attach(SourceSpec::Input("Mic (USB Audio)".into()), ConsumerOptions::speech("stt"))
        .unwrap();
    let epoch = c.epoch();
    assert_eq!(fake.opens(), 1);
    fake.unplug("Mic (USB Audio)");
    wait_until("stream_error", || events.count("error") == 1);
    // The webcam's microphone is the only family member, and it is not the lost device: the hub
    // keeps asking for the headset, device changes included.
    hub.devices_changed();
    std::thread::sleep(Duration::from_millis(300));
    assert_eq!(fake.opens(), 1, "the other microphone must not open");
    assert!(c.is_recovering(), "no stand-in: the consumer still waits for its own device");
    assert_eq!(events.count("recovered"), 0);
    assert!(!fake.push("in:Mic (2- USB Audio)", &vec![0.1; BLOCK * 2]), "the sibling has no stream");

    // The headset returns under its own name: it reopens, same epoch.
    fake.set_present("Mic (USB Audio)", true);
    hub.devices_changed();
    wait_until("reopened under the saved name", || fake.opens() == 2);
    for _ in 0..5 {
        assert!(fake.push("in:Mic (USB Audio)", &vec![0.1; BLOCK * 2]));
    }
    wait_until("stream_recovered", || events.count("recovered") == 1);
    assert_eq!(c.epoch(), epoch);
    assert!(!read_all(&mut c).0.is_empty());
}

#[test]
fn a_renumbered_endpoint_that_appears_after_the_loss_takes_over_even_with_a_sibling_listed() {
    // The headset and the webcam microphone are both "USB Audio"; the headset is re-plugged into
    // another port and comes back as "(3- USB Audio)": the one family member that was not listed
    // when it was lost. The webcam's microphone stays out of it.
    let (hub, fake, events) = hub_with(fast_config());
    fake.add_input("Mic (USB Audio)");
    fake.add_input("Mic (2- USB Audio)");
    let c = hub
        .attach(SourceSpec::Input("Mic (USB Audio)".into()), ConsumerOptions::speech("stt"))
        .unwrap();
    let epoch = c.epoch();
    fake.unplug("Mic (USB Audio)");
    wait_until("stream_error", || events.count("error") == 1);
    fake.add_input("Mic (3- USB Audio)");
    hub.devices_changed();
    wait_until("reopened on the new endpoint", || fake.opens() == 2);
    for _ in 0..5 {
        assert!(fake.push("in:Mic (3- USB Audio)", &vec![0.1; BLOCK * 2]));
    }
    assert!(!fake.push("in:Mic (2- USB Audio)", &vec![0.1; BLOCK * 2]), "the sibling stays closed");
    wait_until("stream_recovered", || events.count("recovered") == 1);
    assert_eq!(events.events()[1].1.epoch, epoch);
}

#[test]
fn a_busy_microphone_is_retried_by_name_and_never_swapped_for_its_sibling() {
    // The headset is still listed, but another app holds it (exclusive mode): the reopen fails by
    // name. The webcam's microphone (same family) must not be opened instead.
    let (hub, fake, events) = hub_with(fast_config());
    fake.add_input("Mic (USB Audio)");
    fake.add_input("Mic (2- USB Audio)");
    let c = hub
        .attach(SourceSpec::Input("Mic (USB Audio)".into()), ConsumerOptions::speech("stt"))
        .unwrap();
    fake.0.refuse_open.lock().unwrap().insert("in:Mic (USB Audio)".to_string());
    fake.fail("in:Mic (USB Audio)", "driver reset");
    wait_until("stream_error", || events.count("error") == 1);
    hub.devices_changed();
    std::thread::sleep(Duration::from_millis(300));
    assert_eq!(fake.opens(), 1, "no stream opened: the headset is busy, the sibling is not a stand-in");
    assert!(c.is_recovering());
    let status = hub.status().unwrap();
    assert_eq!(status.streams.len(), 1, "no stream was created for the sibling");

    fake.0.refuse_open.lock().unwrap().clear();
    wait_until("reopened once the headset is free", || fake.opens() == 2);
    assert!(fake.push("in:Mic (USB Audio)", &vec![0.1; BLOCK * 2]));
}

#[test]
fn the_loss_snapshot_is_taken_again_for_every_outage() {
    // First outage: the sibling "(2- ...)" was there all along. After the headset recovered, the
    // sibling is unplugged and the headset is lost again: now a returning "(2- ...)" is a newcomer.
    let (hub, fake, events) = hub_with(fast_config());
    fake.add_input("Mic (USB Audio)");
    fake.add_input("Mic (2- USB Audio)");
    let c = hub
        .attach(SourceSpec::Input("Mic (USB Audio)".into()), ConsumerOptions::speech("stt"))
        .unwrap();
    let epoch = c.epoch();
    fake.unplug("Mic (USB Audio)");
    wait_until("first stream_error", || events.count("error") == 1);
    fake.set_present("Mic (USB Audio)", true);
    hub.devices_changed();
    wait_until("reopened", || fake.opens() == 2);
    for _ in 0..5 {
        assert!(fake.push("in:Mic (USB Audio)", &vec![0.1; BLOCK * 2]));
    }
    wait_until("first recovery", || events.count("recovered") == 1);

    // Second outage: both are gone, then only the sibling's name comes back.
    fake.set_present("Mic (2- USB Audio)", false);
    fake.unplug("Mic (USB Audio)");
    wait_until("second stream_error", || events.count("error") == 2);
    fake.set_present("Mic (2- USB Audio)", true);
    hub.devices_changed();
    wait_until("reopened on the family member that was not listed at this loss", || fake.opens() == 3);
    for _ in 0..5 {
        assert!(fake.push("in:Mic (2- USB Audio)", &vec![0.1; BLOCK * 2]));
    }
    wait_until("second recovery", || events.count("recovered") == 2);
    assert_eq!(c.epoch(), epoch);
}

#[test]
fn default_sources_follow_the_default_device_and_explicit_ones_stay() {
    let (hub, fake, events) = hub_with(fast_config());
    let mut follows = hub.attach(SourceSpec::DefaultInput, ConsumerOptions::device_rate("follows")).unwrap();
    let mut pinned = hub
        .attach(SourceSpec::Input("Mic A".into()), ConsumerOptions::device_rate("pinned"))
        .unwrap();
    let epoch = follows.epoch();
    assert!(fake.push("in:Mic A", &tagged_block(0)));
    assert_eq!(read_all(&mut follows).0.len(), BLOCK * 2);

    fake.set_default_input(Some("Mic B"));
    hub.devices_changed();
    wait_until("Mic B opened", || fake.opens() == 2);
    // Mic B is 44.1 kHz mono: the device-rate consumer's format follows.
    wait_until("move delivered", || {
        fake.push("in:Mic B", &[0.5f32; 441]);
        let (_, infos) = read_all(&mut follows);
        infos.iter().any(|i| i.format == Some(MONO_44K))
    });
    assert_eq!(follows.epoch(), epoch, "a move keeps the epoch");
    assert_eq!(events.count("error"), 0, "a healthy move is not an outage");
    // The pinned consumer stays on Mic A, which stays open for it alone.
    assert!(fake.push("in:Mic A", &tagged_block(7)));
    let (pinned_audio, _) = read_all(&mut pinned);
    let pinned_blocks: Vec<usize> = pinned_audio.chunks(BLOCK * 2).map(|c| block_of(c[0])).collect();
    assert_eq!(pinned_blocks, vec![0, 7]);
    let (nothing, _) = read_all(&mut follows);
    assert!(nothing.is_empty(), "the default source left Mic A");
    assert_eq!(hub.status().unwrap().streams.len(), 2);

    // Unplugging the default makes Windows pick another default: the source follows it back.
    fake.set_default_input(Some("Mic A"));
    fake.set_present("Mic B", false);
    fake.fail("in:Mic B", "unplugged");
    hub.devices_changed();
    wait_until("recovered on Mic A", || events.count("recovered") >= 1 || {
        let status = hub.status().unwrap();
        status.sources.iter().any(|s| s.source == "default" && s.stream == "in:Mic A" && !s.recovering)
    });
    assert!(fake.push("in:Mic A", &tagged_block(9)));
    let (back, _) = read_all(&mut follows);
    assert_eq!(block_of(back[0]), 9);
    assert_eq!(fake.opens(), 2, "Mic A was still open for the pinned consumer: no reopen");
}

/// The silent-microphone status names the device ("Sigil hears nothing from <mic>"): for the Windows
/// default that is the device behind it now, which moves with the default.
#[test]
fn a_consumer_names_the_device_it_reads_also_after_the_default_moved() {
    let (hub, fake, _events) = hub_with(fast_config());
    let follows = hub.attach(SourceSpec::DefaultInput, ConsumerOptions::speech("follows")).unwrap();
    let pinned = hub
        .attach(SourceSpec::Input("Mic B".into()), ConsumerOptions::speech("pinned"))
        .unwrap();
    assert_eq!(follows.device_name().as_deref(), Some("Mic A"));
    assert_eq!(pinned.device_name().as_deref(), Some("Mic B"));

    fake.set_default_input(Some("Mic B"));
    hub.devices_changed();
    wait_until("the default source names Mic B", || follows.device_name().as_deref() == Some("Mic B"));
    assert_eq!(pinned.device_name().as_deref(), Some("Mic B"));
    hub.shutdown();
    assert_eq!(follows.device_name(), None, "a stopped hub answers nothing");
}

#[test]
fn the_mute_gate_withholds_audio_from_gated_consumers_only() {
    let (hub, fake, _) = hub_with(fast_config());
    let mut stt = hub
        .attach(
            SourceSpec::DefaultInput,
            ConsumerOptions::device_rate("stt").with_mute_group(Some(MuteGroup::Stt)),
        )
        .unwrap();
    let mut vc = hub
        .attach(
            SourceSpec::DefaultInput,
            ConsumerOptions::device_rate("vc").with_mute_group(Some(MuteGroup::VoiceChanger)),
        )
        .unwrap();
    for b in 0..3 {
        fake.push("in:Mic A", &tagged_block(b));
    }
    assert_eq!(read_all(&mut stt).0.len(), 3 * BLOCK * 2);
    assert_eq!(read_all(&mut vc).0.len(), 3 * BLOCK * 2);

    hub.set_muted(MuteGroup::Stt, true);
    assert!(hub.is_muted(MuteGroup::Stt));
    for b in 3..6 {
        fake.push("in:Mic A", &tagged_block(b));
    }
    let (muted, infos) = read_all(&mut stt);
    assert!(muted.is_empty(), "muted audio never reaches the consumer");
    assert!(infos.iter().any(|i| i.gap_after), "the mute edge closes open state");
    assert_eq!(read_all(&mut vc).0.len(), 3 * BLOCK * 2, "other groups keep flowing");

    hub.set_muted(MuteGroup::Stt, false);
    for b in 6..8 {
        fake.push("in:Mic A", &tagged_block(b));
    }
    let (after, _) = read_all(&mut stt);
    let blocks: Vec<usize> = after.chunks(BLOCK * 2).map(|c| block_of(c[0])).collect();
    assert_eq!(blocks, vec![6, 7]);
    let status = hub.status().unwrap();
    assert!(status.muted.is_empty());
}

#[test]
fn silent_inputs_are_restarted_but_silent_loopback_is_not() {
    let (hub, fake, events) = hub_with(HubConfig {
        stall_timeout: Duration::from_millis(150),
        ..fast_config()
    });
    let mic = hub.attach(SourceSpec::DefaultInput, ConsumerOptions::speech("mic")).unwrap();
    let loopback = hub
        .attach(SourceSpec::DefaultLoopback, ConsumerOptions::speech("others"))
        .unwrap();
    assert!(loopback.is_loopback() && !mic.is_loopback());
    wait_until("stall detected", || events.count("error") >= 1);
    let evs = events.events();
    assert_eq!(evs[0].1.source, "default");
    assert!(evs[0].1.message.as_deref().unwrap().contains("stalled"));
    wait_until("restarted", || fake.opens() >= 3);

    // A device that keeps delivering nothing stays one outage: no error / recovered toggling.
    std::thread::sleep(Duration::from_millis(700));
    assert_eq!(events.count("error"), 1);
    assert_eq!(events.count("recovered"), 0);
    let status = hub.status().unwrap();
    let mic_stream = status.streams.iter().find(|s| s.key == "in:Mic A").unwrap();
    assert!(mic_stream.failures >= 2, "{mic_stream:?}");

    // Audio again: recovered once, and a mic that delivers is left alone.
    wait_until("recovered", || {
        fake.push("in:Mic A", &vec![0.0; BLOCK * 2]);
        std::thread::sleep(Duration::from_millis(5));
        events.count("recovered") == 1
    });
    for _ in 0..80 {
        fake.push("in:Mic A", &vec![0.0; BLOCK * 2]);
        std::thread::sleep(Duration::from_millis(5));
    }
    assert_eq!(events.count("error"), 1);
    assert!(events.events().iter().all(|(_, e)| e.source != "loopback:default"));
}

#[test]
fn a_slow_device_open_is_not_a_stall() {
    // The open takes longer than the stall timeout (Bluetooth hands-free, a profile switch): the
    // stall clock starts when the open completed, not before it.
    let (hub, fake, events) = hub_with(HubConfig {
        stall_timeout: Duration::from_millis(300),
        ..fast_config()
    });
    *fake.0.open_delay.lock().unwrap() = Duration::from_millis(700);
    let mut c = hub.attach(SourceSpec::DefaultInput, ConsumerOptions::device_rate("bt")).unwrap();
    fake.feed_for("in:Mic A", Duration::from_millis(600));
    assert_eq!(events.count("error"), 0, "{:?}", events.events());
    assert_eq!(fake.opens(), 1, "no restart");
    let (_, infos) = read_all(&mut c);
    assert!(infos.iter().all(|i| !i.restarted));
}

#[test]
fn a_new_stream_gets_a_grace_period_for_its_first_audio() {
    let (hub, fake, events) = hub_with(HubConfig {
        stall_timeout: Duration::from_millis(250),
        first_data_timeout: Duration::from_millis(1_500),
        ..fast_config()
    });
    let _c = hub.attach(SourceSpec::DefaultInput, ConsumerOptions::speech("slow start")).unwrap();
    // Longer than the steady-state stall timeout, shorter than the first-data grace.
    std::thread::sleep(Duration::from_millis(600));
    assert_eq!(events.count("error"), 0, "silent start inside the grace period");
    fake.feed_for("in:Mic A", Duration::from_millis(300));
    assert_eq!(events.count("error"), 0);
    // Once audio flowed, the steady-state timeout applies.
    wait_until("stall after the audio stopped", || events.count("error") == 1);
    assert!(events.events()[0].1.message.as_deref().unwrap().contains("stalled"));
}

#[test]
fn liveness_ignores_bogus_driver_capture_timestamps() {
    // A driver that reports capture position 0 (or a stale QPC): the blocks carry an ancient
    // capture stamp while audio flows normally. That is not a stall.
    let (hub, fake, events) = hub_with(HubConfig {
        stall_timeout: Duration::from_millis(200),
        ..fast_config()
    });
    let mut c = hub
        .attach(SourceSpec::DefaultInput, ConsumerOptions::device_rate("virtual").with_max_backlog_ms(2_000))
        .unwrap();
    let until = Instant::now() + Duration::from_millis(700);
    let mut pushed = 0;
    while Instant::now() < until {
        assert!(fake.push_at("in:Mic A", &tagged_block(pushed), 0));
        pushed += 1;
        std::thread::sleep(Duration::from_millis(10));
    }
    assert_eq!(events.count("error"), 0, "{:?}", events.events());
    assert_eq!(fake.opens(), 1);
    let (out, infos) = read_all(&mut c);
    assert_eq!(out.len(), pushed * BLOCK * 2, "every block delivered, no restart cut");
    assert!(infos.iter().all(|i| !i.gap_after));
}

#[test]
fn a_source_closed_during_an_outage_ends_the_outage() {
    let (hub, fake, events) = hub_with(HubConfig {
        retry_max: Duration::from_secs(30),
        ..fast_config()
    });
    // A healthy source closes silently.
    let healthy = hub.attach(SourceSpec::DefaultInput, ConsumerOptions::speech("healthy")).unwrap();
    assert!(healthy.close_and_wait());
    assert!(events.events().is_empty());

    let pinned = hub
        .attach(SourceSpec::Input("Mic B".into()), ConsumerOptions::speech("pinned"))
        .unwrap();
    let epoch = pinned.epoch();
    fake.unplug("Mic B");
    wait_until("stream_error", || events.count("error") == 1);
    // The user stops STT (or picks another device) while the mic is still unplugged.
    assert!(pinned.close_and_wait());
    wait_until("stream_closed", || events.count("closed") == 1);
    let evs = events.events();
    assert_eq!(
        evs.last().unwrap(),
        &(
            "closed",
            StreamEvent {
                source: "Mic B".into(),
                epoch,
                message: None
            }
        )
    );
    assert_eq!(events.count("recovered"), 0);
    let status = hub.status().unwrap();
    assert!(status.sources.is_empty() && status.streams.is_empty());
}

#[test]
fn the_legacy_capture_flag_bypasses_supervision() {
    // `SIGIL_CAPTURE_SHARING=0`: one stream per consumer and no watchdog at all.
    let (hub, fake, events) = hub_with(HubConfig {
        stall_timeout: Duration::from_millis(150),
        share_streams: false,
        supervise: false,
        ..fast_config()
    });
    let other = hub.attach(SourceSpec::DefaultInput, ConsumerOptions::device_rate("other")).unwrap();
    // Opened last, so the fake's pushes and failures on "in:Mic A" hit this consumer's own stream.
    let mut c = hub.attach(SourceSpec::DefaultInput, ConsumerOptions::device_rate("legacy")).unwrap();
    assert_eq!(fake.opens(), 2);
    // Silence is never treated as a stall.
    std::thread::sleep(Duration::from_millis(400));
    assert_eq!(events.count("error"), 0);
    for b in 0..3 {
        assert!(fake.push("in:Mic A", &tagged_block(b)));
    }

    // A device error ends the capture instead of restarting it.
    assert!(c.ended().is_none());
    fake.fail("in:Mic A", "driver reset");
    wait_until("stream_error", || events.count("error") == 1);
    wait_until("consumer ended", || c.ended().is_some());
    assert_eq!(c.ended().as_deref(), Some("driver reset"));
    let (out, _) = read_all(&mut c);
    assert_eq!(out.len(), 3 * BLOCK * 2, "audio before the error is still delivered");
    hub.devices_changed();
    std::thread::sleep(Duration::from_millis(300));
    assert_eq!(fake.opens(), 2, "no reopen: not after the backoff, not on a device change");
    assert!(other.ended().is_none(), "the other consumer's own stream is unaffected");

    // Stopping the engine closes the outage.
    assert!(c.close_and_wait());
    wait_until("stream_closed", || events.count("closed") == 1);
    assert_eq!(events.count("recovered"), 0);
}

#[test]
fn unsupervised_a_new_consumer_reopens_a_failed_stream_once() {
    let (hub, fake, events) = hub_with(HubConfig {
        supervise: false,
        ..fast_config()
    });
    let a = hub
        .attach(SourceSpec::Input("Mic A".into()), ConsumerOptions::speech("a"))
        .unwrap();
    fake.unplug("Mic A");
    wait_until("a ended", || a.ended().is_some());
    // Joining the dead stream is a fresh start: one reopen, which fails while unplugged...
    let b = hub
        .attach(SourceSpec::Input("Mic A".into()), ConsumerOptions::speech("b"))
        .unwrap();
    wait_until("b ended", || b.ended().is_some());
    assert!(b.ended().unwrap().contains("not found"), "{:?}", b.ended());
    assert_eq!(fake.opens(), 1);
    // ...and succeeds once the device is back.
    fake.set_present("Mic A", true);
    let mut c = hub
        .attach(SourceSpec::Input("Mic A".into()), ConsumerOptions::device_rate("c"))
        .unwrap();
    wait_until("reopened", || fake.opens() == 2);
    wait_until("audio flows", || {
        fake.push("in:Mic A", &tagged_block(1));
        !read_all(&mut c).0.is_empty()
    });
    assert!(c.ended().is_none());
    wait_until("recovered", || events.count("recovered") == 1);
    assert_eq!(events.count("error"), 1);
}

#[test]
fn the_last_consumer_releases_the_device() {
    let (hub, fake, _) = hub_with(fast_config());
    let a = hub.attach(SourceSpec::DefaultInput, ConsumerOptions::speech("a")).unwrap();
    let b = hub.attach(SourceSpec::DefaultInput, ConsumerOptions::speech("b")).unwrap();
    assert!(a.close_and_wait());
    assert_eq!(fake.stops(), 0, "b still uses the device");
    assert!(b.close_and_wait());
    assert_eq!(fake.stops(), 1);
    assert_eq!(fake.live_streams(), 0);
    let status = hub.status().unwrap();
    assert!(status.streams.is_empty() && status.sources.is_empty());
    // Dropping a consumer detaches it too.
    let c = hub.attach(SourceSpec::DefaultInput, ConsumerOptions::speech("c")).unwrap();
    drop(c);
    wait_until("dropped consumer detached", || fake.stops() == 2);
}

#[test]
fn attach_fails_fast_and_leaves_nothing_behind() {
    let (hub, fake, _) = hub_with(fast_config());
    let err = hub
        .attach(SourceSpec::Input("Nope".into()), ConsumerOptions::speech("x"))
        .err()
        .unwrap();
    assert!(err.contains("Nope"), "{err}");
    fake.0.refuse_open.lock().unwrap().insert("in:Mic B".into());
    let err = hub
        .attach(SourceSpec::Input("Mic B".into()), ConsumerOptions::speech("x"))
        .err()
        .unwrap();
    assert_eq!(err, "exclusive mode");
    let status = hub.status().unwrap();
    assert!(status.sources.is_empty() && status.streams.is_empty());

    let mut keep = Vec::new();
    for i in 0..MAX_CONSUMERS_PER_STREAM {
        keep.push(hub.attach(SourceSpec::DefaultInput, ConsumerOptions::speech("many")).unwrap_or_else(|e| panic!("{i}: {e}")));
    }
    assert!(hub.attach(SourceSpec::DefaultInput, ConsumerOptions::speech("one too many")).is_err());
}

#[test]
fn a_full_command_ring_rebuilds_a_silent_stream_without_losing_consumers() {
    let (hub, fake, _) = hub_with(fast_config());
    // A loopback stream with nothing playing never runs its callback, so commands pile up.
    let mut keeper = hub
        .attach(SourceSpec::DefaultLoopback, ConsumerOptions::device_rate("keeper"))
        .unwrap();
    for _ in 0..40 {
        drop(hub.attach(SourceSpec::DefaultLoopback, ConsumerOptions::device_rate("churn")).unwrap());
    }
    wait_until("rebuilt", || fake.opens() >= 2);
    let _ = hub.status().unwrap(); // every queued detach processed
    assert!(fake.push("loopback:Speakers", &tagged_block(3)));
    let (out, _) = read_all(&mut keeper);
    assert_eq!(block_of(out[0]), 3);
}

#[test]
fn without_sharing_every_consumer_gets_its_own_stream() {
    let (hub, fake, _) = hub_with(HubConfig {
        share_streams: false,
        ..fast_config()
    });
    let a = hub.attach(SourceSpec::DefaultInput, ConsumerOptions::speech("a")).unwrap();
    let b = hub.attach(SourceSpec::DefaultInput, ConsumerOptions::speech("b")).unwrap();
    assert_eq!(fake.opens(), 2);
    assert_eq!(a.source(), b.source());
    assert_ne!(a.epoch(), b.epoch());
    assert!(!hub.status().unwrap().sharing);
}

#[test]
fn the_callback_wakes_a_waiting_consumer_without_polling() {
    let (hub, fake, _) = hub_with(fast_config());
    let mut c = hub.attach(SourceSpec::DefaultInput, ConsumerOptions::device_rate("waiter")).unwrap();
    let (tx, rx) = mpsc::channel();
    let reader = std::thread::spawn(move || {
        // Register this thread as the waker, then wait for audio with a long timeout.
        c.wait(Duration::from_millis(1));
        let _ = tx.send(());
        let started = Instant::now();
        let mut out = Vec::new();
        while out.is_empty() {
            c.wait(Duration::from_secs(5));
            c.read(&mut out);
        }
        started.elapsed()
    });
    rx.recv().unwrap();
    std::thread::sleep(Duration::from_millis(50));
    assert!(fake.push("in:Mic A", &tagged_block(1)));
    let waited = reader.join().unwrap();
    assert!(waited < Duration::from_millis(1_000), "woken after {waited:?}");
}

/// Real WASAPI devices (manual: `cargo test real_devices -- --ignored --nocapture`). Needs a
/// microphone; opens it for about a second, plus a loopback stream on the default output.
#[test]
#[ignore = "opens real audio devices"]
fn real_devices_share_one_stream_and_open_loopback() {
    use super::device_watch::snapshot_cpal;
    use super::stream::CpalBackend;

    let snapshot = snapshot_cpal().expect("device list");
    println!("inputs: {:?}", snapshot.inputs.iter().map(|d| &d.name).collect::<Vec<_>>());
    println!("outputs: {:?}", snapshot.outputs.iter().map(|d| &d.name).collect::<Vec<_>>());
    println!("defaults: {:?} / {:?}", snapshot.default_input, snapshot.default_output);
    let hub = CaptureHub::new(Arc::new(CpalBackend), HubConfig::default()).unwrap();
    let recorder = Arc::new(Recorder::default());
    hub.set_event_sink(recorder.clone());

    let mut stt = hub.attach(SourceSpec::DefaultInput, ConsumerOptions::speech("stt")).expect("default input");
    let named = snapshot.default_input.clone().expect("default input name");
    let mut raw = hub
        .attach(SourceSpec::Input(named), ConsumerOptions::device_rate("raw"))
        .expect("same input by name");
    let status = hub.status().unwrap();
    println!("{}", serde_json::to_string_pretty(&status).unwrap());
    assert_eq!(status.streams.len(), 1, "one stream for the default input and its name");

    let started = Instant::now();
    let (mut out16, mut out_raw) = (Vec::new(), Vec::new());
    let mut last_qpc = None;
    while started.elapsed() < Duration::from_millis(1_200) {
        stt.wait(Duration::from_millis(50));
        let info = stt.read(&mut out16);
        if info.end_qpc_ns.is_some() {
            last_qpc = info.end_qpc_ns;
        }
        raw.read(&mut out_raw);
    }
    let age_ms = last_qpc.map(|q| (now_qpc_ns() - q) as f64 / 1e6);
    println!(
        "16 kHz samples: {}, raw samples: {}, newest audio age: {age_ms:?} ms",
        out16.len(),
        out_raw.len()
    );
    assert!(out16.len() > 14_000, "about a second of 16 kHz audio");
    assert!(age_ms.is_some_and(|a| (-5.0..200.0).contains(&a)), "QPC capture stamps are current");

    if snapshot.default_output.is_some() {
        let loopback = hub
            .attach(SourceSpec::DefaultLoopback, ConsumerOptions::speech("others"))
            .expect("loopback of the default output");
        let status = hub.status().unwrap();
        assert!(status.streams.iter().any(|s| s.loopback && s.state == "running"), "{status:?}");
        drop(loopback);
    }
    assert!(stt.close_and_wait());
    assert!(raw.close_and_wait());
    assert_eq!(recorder.count("error"), 0);
}
