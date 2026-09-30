//! Sink tests: wire formats, IPC parsing, and the engine driven step by step against fake outputs
//! (a simulated device clock) and a fake voice changer.

use std::collections::VecDeque;
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};

use super::engine::{SinkConfig, SinkCore, SinkEvents};
use super::output::{match_output_device, ring, OpenOutput, OutputBackend, Playout, PlayoutShared};
use super::vc_route::{NoVc, VcRoute};
use super::*;

// ---------------------------------------------------------------------------------------------
// Fakes
// ---------------------------------------------------------------------------------------------

struct FakeInner {
    devices: Vec<(String, u32, usize)>,
    open: Vec<(String, Playout, Arc<PlayoutShared>)>,
    opens: usize,
}

#[derive(Clone)]
struct FakeOutputs(Arc<Mutex<FakeInner>>);

impl FakeOutputs {
    fn new(devices: &[(&str, u32, usize)]) -> Self {
        Self(Arc::new(Mutex::new(FakeInner {
            devices: devices.iter().map(|&(n, r, c)| (n.to_string(), r, c)).collect(),
            open: Vec::new(),
            opens: 0,
        })))
    }

    fn opens(&self) -> usize {
        self.0.lock().unwrap().opens
    }

    /// Run the device callback of the latest stream opened on `name` for `frames` frames.
    fn play(&self, name: &str, frames: usize) -> Vec<f32> {
        let mut inner = self.0.lock().unwrap();
        let ch = inner.devices.iter().find(|d| d.0 == name).map_or(1, |d| d.2);
        let Some(entry) = inner.open.iter_mut().rev().find(|o| o.0 == name) else {
            return vec![0.0; frames * ch];
        };
        let mut buf = vec![0.0f32; frames * ch];
        entry.1.fill(&mut buf);
        buf
    }

    fn shared(&self, name: &str) -> Arc<PlayoutShared> {
        let inner = self.0.lock().unwrap();
        inner.open.iter().rev().find(|o| o.0 == name).expect("open").2.clone()
    }
}

impl OutputBackend for FakeOutputs {
    fn open(&self, device: &str) -> Result<OpenOutput, String> {
        let mut inner = self.0.lock().unwrap();
        let names: Vec<String> = inner.devices.iter().map(|d| d.0.clone()).collect();
        let idx = if output::is_default_device(device) {
            0
        } else {
            match_output_device(device, &names).ok_or_else(|| format!("Output device '{}' not found", device.trim()))?
        };
        let (name, rate, ch) = inner.devices[idx].clone();
        let (producer, consumer) = ring(rate, ch);
        let shared = Arc::new(PlayoutShared::default());
        inner.open.push((name.clone(), Playout::new(consumer, ch, rate, shared.clone()), shared.clone()));
        inner.opens += 1;
        Ok(OpenOutput {
            name,
            rate,
            channels: ch,
            producer,
            shared,
            handle: Box::new(()),
        })
    }

    fn default_name(&self) -> Option<String> {
        self.0.lock().unwrap().devices.first().map(|d| d.0.clone())
    }
}

#[derive(Default)]
struct Recorded {
    utterances: Mutex<Vec<UtteranceEvent>>,
    statuses: Mutex<Vec<(Instant, SinkStatus)>>,
}

struct Rec(Arc<Recorded>);

impl SinkEvents for Rec {
    fn utterance(&self, event: &UtteranceEvent) {
        self.0.utterances.lock().unwrap().push(event.clone());
    }

    fn status(&self, status: &SinkStatus) {
        self.0.statuses.lock().unwrap().push((Instant::now(), status.clone()));
    }
}

impl Recorded {
    fn states(&self, id: &str) -> Vec<UtteranceState> {
        self.utterances.lock().unwrap().iter().filter(|e| e.id == id).map(|e| e.state).collect()
    }

    fn last(&self, id: &str) -> Option<UtteranceEvent> {
        self.utterances.lock().unwrap().iter().rev().find(|e| e.id == id).cloned()
    }
}

#[derive(Default)]
struct FakeVcInner {
    format: Option<(u32, usize)>,
    queue: VecDeque<(u64, Vec<f32>)>,
    cap: usize,
    cancelled: Vec<u64>,
}

#[derive(Clone, Default)]
struct FakeVc(Arc<Mutex<FakeVcInner>>);

impl FakeVc {
    fn running(rate: u32, ch: usize, cap: usize) -> Self {
        let vc = Self::default();
        {
            let mut i = vc.0.lock().unwrap();
            i.format = Some((rate, ch));
            i.cap = cap;
        }
        vc
    }

    fn queued(&self) -> usize {
        self.0.lock().unwrap().queue.iter().map(|(_, b)| b.len()).sum()
    }

    /// Mix `n` samples out of the queue (the voice changer's worker); returns them.
    fn consume(&self, n: usize) -> Vec<(u64, f32)> {
        let mut i = self.0.lock().unwrap();
        let mut out = Vec::new();
        while out.len() < n {
            let Some((tag, block)) = i.queue.front_mut() else { break };
            let take = (n - out.len()).min(block.len());
            let tag = *tag;
            out.extend(block.drain(..take).map(|s| (tag, s)));
            if block.is_empty() {
                i.queue.pop_front();
            }
        }
        out
    }
}

impl VcRoute for FakeVc {
    fn format(&self) -> Option<(u32, usize)> {
        self.0.lock().unwrap().format
    }

    fn try_push(&self, tag: u64, block: Vec<f32>, format: (u32, usize)) -> Result<(), Option<Vec<f32>>> {
        let mut i = self.0.lock().unwrap();
        if i.format != Some(format) {
            return Err(None);
        }
        let queued: usize = i.queue.iter().map(|(_, b)| b.len()).sum();
        if queued >= i.cap {
            return Err(Some(block));
        }
        i.queue.push_back((tag, block));
        Ok(())
    }

    fn pending(&self, tag: u64) -> usize {
        self.0.lock().unwrap().queue.iter().filter(|(t, _)| *t == tag).map(|(_, b)| b.len()).sum()
    }

    fn cancel(&self, tag: u64) {
        let mut i = self.0.lock().unwrap();
        i.queue.retain(|(t, _)| *t != tag);
        i.cancelled.push(tag);
    }
}

struct Harness {
    core: Arc<SinkCore>,
    outputs: FakeOutputs,
    events: Arc<Recorded>,
    now: Instant,
}

const DEV: &str = "Speakers (Test)";

impl Harness {
    fn new(config: SinkConfig, vc: Box<dyn VcRoute>) -> Self {
        let outputs = FakeOutputs::new(&[(DEV, 48_000, 2), ("Cable (Test)", 16_000, 1)]);
        let events = Arc::new(Recorded::default());
        let core = SinkCore::new(config, Box::new(outputs.clone()), vc, Box::new(Rec(events.clone())));
        Self {
            core,
            outputs,
            events,
            now: Instant::now(),
        }
    }

    fn device() -> Self {
        Self::new(SinkConfig::default(), Box::new(NoVc))
    }

    fn begin(&self, id: &str, target: SinkTarget) {
        self.core
            .begin(UtteranceSpec {
                id: id.into(),
                target,
            })
            .unwrap();
    }

    /// Run the scheduler and `ms` of the device clock on `name`, in 10 ms device callbacks.
    fn run(&mut self, name: &str, ms: u64) -> Vec<f32> {
        let mut out = Vec::new();
        let rate = if name == DEV { 48_000 } else { 16_000 };
        for _ in 0..ms / 10 {
            self.core.step(self.now);
            out.extend(self.outputs.play(name, rate as usize / 100));
            self.now += Duration::from_millis(10);
        }
        self.core.step(self.now);
        out
    }
}

fn target(route: SinkRoute, device: &str) -> SinkTarget {
    SinkTarget {
        route,
        device: device.into(),
        volume: 1.0,
    }
}

fn stereo(frames: usize, v: f32) -> Vec<f32> {
    vec![v; frames * 2]
}

/// Frames of `out` (stereo) whose left channel is within `eps` of `v`.
fn count_level(out: &[f32], v: f32, eps: f32) -> usize {
    out.iter().step_by(2).filter(|&&s| (s - v).abs() < eps).count()
}

// ---------------------------------------------------------------------------------------------
// Wire formats and IPC
// ---------------------------------------------------------------------------------------------

#[test]
fn wire_formats_are_frozen() {
    let spec: UtteranceSpec =
        serde_json::from_str(r#"{"id":"tts-1-0","target":{"route":"voiceChanger","device":"Speakers"}}"#).unwrap();
    assert_eq!(spec.target.route, SinkRoute::VoiceChanger);
    assert_eq!(spec.target.volume, 1.0);
    let spec: UtteranceSpec = serde_json::from_str(r#"{"id":"x"}"#).unwrap();
    assert_eq!(spec.target, SinkTarget::default());
    assert_eq!(serde_json::to_string(&SinkRoute::Both).unwrap(), r#""both""#);
    let ev = UtteranceEvent {
        id: "a".into(),
        state: UtteranceState::Cancelled,
        error: None,
        played_ms: 120,
    };
    assert_eq!(
        serde_json::to_string(&ev).unwrap(),
        r#"{"id":"a","state":"cancelled","error":null,"playedMs":120}"#
    );
    let st = SinkStatus {
        playing: Some("a".into()),
        queued: vec!["b".into()],
        queued_ms: 800,
        devices: vec![],
    };
    assert_eq!(
        serde_json::to_string(&st).unwrap(),
        r#"{"playing":"a","queued":["b"],"queuedMs":800,"devices":[]}"#
    );
    let r = SpeakReport {
        sample_rate: 24_000,
        frames: 48_000,
        first_audio_ms: 180,
        total_ms: 900,
    };
    assert_eq!(
        serde_json::to_string(&r).unwrap(),
        r#"{"sampleRate":24000,"frames":48000,"firstAudioMs":180,"totalMs":900}"#
    );
    assert_eq!(SinkError::Cancelled.to_string(), "cancelled", "TS matches this string");
}

fn headers(id: &str, rate: &str, ch: &str) -> tauri::http::HeaderMap {
    let mut h = tauri::http::HeaderMap::new();
    h.insert(HEADER_UTTERANCE, id.parse().unwrap());
    h.insert(HEADER_SAMPLE_RATE, rate.parse().unwrap());
    h.insert(HEADER_CHANNELS, ch.parse().unwrap());
    h
}

#[test]
fn push_pcm_parses_a_binary_body_and_headers() {
    let samples = [0.5f32, -0.25, 1.0, 0.0];
    let bytes: Vec<u8> = samples.iter().flat_map(|s| s.to_le_bytes()).collect();
    let got = parse_push(&headers("tts-3-1", "24000", "2"), &InvokeBody::Raw(bytes.clone())).unwrap();
    assert_eq!(
        got,
        PcmPush {
            id: "tts-3-1".into(),
            sample_rate: 24_000,
            channels: 2,
            samples: samples.to_vec(),
        }
    );
    // The postMessage fallback's JSON byte array is accepted too.
    let json = serde_json::Value::Array(bytes.iter().map(|&b| serde_json::json!(b)).collect());
    assert_eq!(parse_push(&headers("a", "24000", "2"), &InvokeBody::Json(json)).unwrap().samples, samples.to_vec());
    // Errors.
    let bad = |h: tauri::http::HeaderMap, body: InvokeBody| parse_push(&h, &body).unwrap_err();
    assert!(matches!(bad(headers("a", "x", "1"), InvokeBody::Raw(vec![0; 4])), SinkError::InvalidFormat(_)));
    assert!(matches!(bad(headers("a", "24000", "1"), InvokeBody::Raw(vec![0; 5])), SinkError::InvalidFormat(_)));
    assert!(matches!(
        bad(headers("a", "24000", "1"), InvokeBody::Json(serde_json::json!({"samples": [1]}))),
        SinkError::InvalidFormat(_)
    ));
    let mut h = headers("a", "24000", "1");
    h.remove(HEADER_CHANNELS);
    assert!(bad(h, InvokeBody::Raw(vec![])).to_string().contains(HEADER_CHANNELS));
}

#[test]
fn api_without_setup_is_unavailable() {
    // The global sink only exists inside the app (plugin setup).
    if SINK.get().is_none() {
        assert!(matches!(end("x"), Err(SinkError::Unavailable(_))));
        assert!(matches!(writer("x"), Err(SinkError::Unavailable(_))));
        assert_eq!(status(), SinkStatus::default());
        cancel(None);
        flush();
    }
}

// ---------------------------------------------------------------------------------------------
// Engine
// ---------------------------------------------------------------------------------------------

/// First audio after sentence one: the first utterance plays as soon as its audio arrived, while
/// later ones are still being produced; utterances play in begin order (not push order), back to
/// back without a gap, and report queued -> playing -> done.
#[test]
fn utterances_play_in_begin_order_without_gaps() {
    let mut h = Harness::device();
    h.begin("a", target(SinkRoute::Device, ""));
    h.begin("b", target(SinkRoute::Device, ""));
    // b's producer finishes first; it still waits for a.
    h.core.push("b", &stereo(4_800, -0.25), 48_000, 2).unwrap();
    h.core.end("b").unwrap();
    let silent = h.run(DEV, 50);
    assert!(silent.iter().all(|&s| s == 0.0), "nothing plays before a");
    h.core.push("a", &stereo(9_600, 0.5), 48_000, 2).unwrap();
    h.core.end("a").unwrap();
    let out = h.run(DEV, 400);
    let left: Vec<f32> = out.iter().step_by(2).copied().collect();
    let first_a = left.iter().position(|&s| s != 0.0).expect("a plays");
    assert!(first_a < 480, "a starts within one callback: {first_a}");
    // 200 ms of a, then 100 ms of b, with no silent frame between them.
    let a_frames = count_level(&out, 0.5, 1e-6);
    let b_frames = count_level(&out, -0.25, 1e-6);
    assert!(a_frames >= 9_600 - 100 && b_frames >= 4_800 - 100, "{a_frames} {b_frames}");
    let last_b = left.iter().rposition(|&s| (s + 0.25).abs() < 1e-6).unwrap();
    assert!(left[first_a..=last_b].iter().all(|&s| s != 0.0), "gapless join");
    assert_eq!(
        h.events.states("a"),
        vec![UtteranceState::Queued, UtteranceState::Playing, UtteranceState::Done]
    );
    assert_eq!(
        h.events.states("b"),
        vec![UtteranceState::Queued, UtteranceState::Playing, UtteranceState::Done]
    );
    let order: Vec<(String, UtteranceState)> =
        h.events.utterances.lock().unwrap().iter().map(|e| (e.id.clone(), e.state)).collect();
    let pos = |id: &str, s: UtteranceState| order.iter().position(|e| e.0 == id && e.1 == s).unwrap();
    assert!(pos("a", UtteranceState::Done) < pos("b", UtteranceState::Done));
    assert!((195..=205).contains(&h.events.last("a").unwrap().played_ms));
    assert_eq!(h.core.status(), SinkStatus { devices: vec![DEV.into()], ..Default::default() });
    assert_eq!(h.outputs.opens(), 1, "one persistent stream");
}

/// Skip drops the playing utterance's audio already in the ring at once and the next one follows;
/// flush (stop) silences everything; the output stays open.
#[test]
fn skip_and_stop_cut_audio_already_written_ahead() {
    let mut h = Harness::device();
    h.begin("a", target(SinkRoute::Device, ""));
    h.begin("b", target(SinkRoute::Device, ""));
    h.core.push("a", &stereo(96_000, 0.5), 48_000, 2).unwrap();
    h.core.end("a").unwrap();
    h.core.push("b", &stereo(96_000, -0.25), 48_000, 2).unwrap();
    h.core.end("b").unwrap();
    h.run(DEV, 100);
    assert_eq!(h.core.status().playing.as_deref(), Some("a"));
    h.core.cancel(Some("a"));
    let out = h.run(DEV, 30);
    let left: Vec<f32> = out.iter().step_by(2).copied().collect();
    // Within one callback (plus the 2 ms ramp) the output is b, not a's 200 ms in the ring.
    assert!(left[480 + 200..].iter().all(|&s| (s + 0.25).abs() < 1e-6), "b follows the skip at once");
    let max_step = left.windows(2).map(|w| (w[1] - w[0]).abs()).fold(0.0f32, f32::max);
    assert!(max_step < 0.02, "no click: {max_step}");
    assert_eq!(h.events.last("a").unwrap().state, UtteranceState::Cancelled);
    assert!(h.events.last("a").unwrap().played_ms >= 90);
    assert!(matches!(h.core.push("a", &stereo(1, 0.1), 48_000, 2), Err(SinkError::Cancelled)));
    assert!(matches!(h.core.end("a"), Err(SinkError::Cancelled)));
    h.core.flush();
    let out = h.run(DEV, 50);
    assert!(out[480 * 2 * 2..].iter().all(|&s| s.abs() < 1e-3), "stop silences at once");
    assert_eq!(h.events.last("b").unwrap().state, UtteranceState::Cancelled);
    assert_eq!(h.core.status().devices, vec![DEV.to_string()], "the device stays open");
    // Slots of cancelled utterances are freed once the callback passed their audio.
    h.begin("c", target(SinkRoute::Device, ""));
    h.core.push("c", &stereo(480, 0.3), 48_000, 2).unwrap();
    h.core.end("c").unwrap();
    let out = h.run(DEV, 50);
    assert!(count_level(&out, 0.3, 1e-6) > 300);
    assert_eq!(h.events.last("c").unwrap().state, UtteranceState::Done);
}

/// Back-pressure: the head is admitted up to its cap and later utterances up to the total cap; a
/// blocked producer resumes once playback made room, and cancellation ends its wait.
#[test]
fn producers_block_until_playback_makes_room() {
    let cfg = SinkConfig {
        head_cap: Duration::from_millis(500),
        total_cap: Duration::from_millis(800),
        ..SinkConfig::default()
    };
    let mut h = Harness::new(cfg, Box::new(NoVc));
    h.begin("a", target(SinkRoute::Device, ""));
    h.begin("b", target(SinkRoute::Device, ""));
    // Admitted whole even though it exceeds the head cap (below the cap when pushed).
    h.core.push("a", &stereo(48_000, 0.5), 48_000, 2).unwrap();
    let core = h.core.clone();
    let blocked = std::thread::spawn(move || core.push("a", &stereo(4_800, 0.5), 48_000, 2));
    let core = h.core.clone();
    let later = std::thread::spawn(move || core.push("b", &stereo(4_800, 0.1), 48_000, 2));
    std::thread::sleep(Duration::from_millis(100));
    assert!(!blocked.is_finished() && !later.is_finished(), "both wait for room");
    // Play until a's pending audio dropped under the head cap.
    for _ in 0..80 {
        h.run(DEV, 20);
        if blocked.is_finished() {
            break;
        }
    }
    assert_eq!(blocked.join().unwrap(), Ok(()));
    // b waits for the total cap; cancel ends its wait.
    if !later.is_finished() {
        h.core.cancel(Some("b"));
    }
    let r = later.join().unwrap();
    assert!(r.is_ok() || r == Err(SinkError::Cancelled), "{r:?}");
}

/// A streamed utterance waits for its prebuffer (or the prebuffer timeout) before it starts, so
/// the first audio does not stutter; an ended one starts at once.
#[test]
fn streamed_audio_waits_for_the_prebuffer() {
    let mut h = Harness::device();
    h.begin("a", target(SinkRoute::Device, ""));
    h.core.push("a", &stereo(2_400, 0.5), 48_000, 2).unwrap(); // 50 ms
    let out = h.run(DEV, 100);
    assert!(out.iter().all(|&s| s == 0.0), "under the prebuffer");
    h.core.push("a", &stereo(9_600, 0.5), 48_000, 2).unwrap(); // 250 ms in total
    let out = h.run(DEV, 50);
    assert!(count_level(&out, 0.5, 1e-6) > 1_000, "started");

    h.begin("b", target(SinkRoute::Device, ""));
    h.run(DEV, 300); // a plays out (not ended yet, keeps b waiting)
    h.core.end("a").unwrap();
    h.core.push("b", &stereo(480, 0.25), 48_000, 2).unwrap();
    let out = h.run(DEV, 100);
    assert_eq!(count_level(&out, 0.25, 1e-6), 0, "b is under its prebuffer");
    h.now += Duration::from_millis(1_600); // the prebuffer timeout
    let out = h.run(DEV, 50);
    assert!(count_level(&out, 0.25, 1e-6) > 300, "b starts after the prebuffer timeout");
}

/// Resampled audio keeps its exact length (24 kHz mono -> 48 kHz stereo) and the gain applies.
#[test]
fn resampled_utterances_keep_their_length() {
    let mut h = Harness::device();
    h.begin(
        "a",
        SinkTarget {
            route: SinkRoute::Device,
            device: "".into(),
            volume: 0.5,
        },
    );
    let tone: Vec<f32> = (0..12_000).map(|i| 0.8 * (i as f32 * 0.05).sin()).collect();
    h.core.push("a", &tone, 24_000, 1).unwrap();
    h.core.end("a").unwrap();
    let out = h.run(DEV, 700);
    let peak = out.iter().fold(0.0f32, |m, &s| m.max(s.abs()));
    assert!((0.38..=0.41).contains(&peak), "gain 0.5 of 0.8: {peak}");
    let done = h.events.last("a").unwrap();
    assert_eq!(done.state, UtteranceState::Done);
    assert_eq!(done.played_ms, 500, "exactly 24 000 source frames");
}

/// Route `voiceChanger`: tagged blocks in the capture format go to the voice changer (no device is
/// opened); done once it mixed them. Cancel removes the queued audio. Without a running voice
/// changer the route plays on the device; `both` plays on both.
#[test]
fn voice_changer_route_feeds_the_aux_queue_and_falls_back() {
    let vc = FakeVc::running(16_000, 1, 16_000 * 10);
    let mut h = Harness::new(SinkConfig::default(), Box::new(vc.clone()));
    h.begin("a", target(SinkRoute::VoiceChanger, ""));
    h.core.push("a", &vec![0.5; 24_000], 24_000, 1).unwrap(); // 1 s
    h.core.end("a").unwrap();
    for _ in 0..5 {
        h.core.step(h.now);
    }
    assert_eq!(h.outputs.opens(), 0, "no device for the voice-changer route");
    assert_eq!(vc.queued(), 16_000, "1 s at the capture rate");
    // Queued in the voice changer is not audible yet: "playing" once it mixes some.
    assert_eq!(h.events.states("a"), vec![UtteranceState::Queued]);
    let mut mixed = vc.consume(1_600);
    h.core.step(h.now);
    assert_eq!(h.events.states("a"), vec![UtteranceState::Queued, UtteranceState::Playing]);
    mixed.extend(vc.consume(16_000));
    assert_eq!(mixed.len(), 16_000);
    assert!(mixed.iter().all(|&(tag, _)| tag == mixed[0].0));
    h.core.step(h.now);
    assert_eq!(h.events.last("a").unwrap().state, UtteranceState::Done);
    assert_eq!(h.events.last("a").unwrap().played_ms, 1_000);

    // Cancel while queued in the voice changer.
    h.begin("b", target(SinkRoute::VoiceChanger, ""));
    h.core.push("b", &vec![0.5; 16_000], 16_000, 1).unwrap();
    h.core.end("b").unwrap();
    h.core.step(h.now);
    assert!(vc.queued() > 0);
    h.core.cancel(Some("b"));
    assert_eq!(vc.queued(), 0);
    assert_eq!(vc.0.lock().unwrap().cancelled.len(), 1);

    // `both`: device and voice changer.
    h.begin("c", target(SinkRoute::Both, ""));
    h.core.push("c", &stereo(4_800, 0.5), 48_000, 2).unwrap();
    h.core.end("c").unwrap();
    let out = h.run(DEV, 150);
    assert!(count_level(&out, 0.5, 1e-6) > 4_000);
    assert_eq!(vc.queued(), 1_600, "100 ms at 16 kHz mono");
    vc.consume(1_600);
    h.run(DEV, 20);
    assert_eq!(h.events.last("c").unwrap().state, UtteranceState::Done);

    // Voice changer stopped: the voice-changer route falls back to the device.
    vc.0.lock().unwrap().format = None;
    h.begin("d", target(SinkRoute::VoiceChanger, ""));
    h.core.push("d", &stereo(4_800, 0.25), 48_000, 2).unwrap();
    h.core.end("d").unwrap();
    let out = h.run(DEV, 150);
    assert!(count_level(&out, 0.25, 1e-6) > 4_000);
    assert_eq!(h.events.last("d").unwrap().state, UtteranceState::Done);
}

/// The voice changer's full aux queue is back-pressure: blocks wait and follow in order.
#[test]
fn voice_changer_back_pressure_keeps_order() {
    let vc = FakeVc::running(16_000, 1, 3_200);
    let mut h = Harness::new(SinkConfig::default(), Box::new(vc.clone()));
    h.begin("a", target(SinkRoute::VoiceChanger, ""));
    let ramp: Vec<f32> = (0..16_000).map(|i| i as f32 / 16_000.0).collect();
    h.core.push("a", &ramp, 16_000, 1).unwrap();
    h.core.end("a").unwrap();
    let mut mixed = Vec::new();
    for _ in 0..40 {
        h.core.step(h.now);
        assert!(vc.queued() <= 3_200 + 1_600, "never above the cap plus one block");
        mixed.extend(vc.consume(800).into_iter().map(|(_, s)| s));
        h.now += Duration::from_millis(50);
    }
    assert_eq!(mixed, ramp, "every sample, in order");
    h.core.step(h.now);
    assert_eq!(h.events.last("a").unwrap().state, UtteranceState::Done);
}

/// Device names are honoured (tolerant matching of old SAPI names); a missing device fails the
/// utterance with a readable error and the next utterance still plays.
#[test]
fn the_device_field_is_honoured() {
    let mut h = Harness::device();
    h.begin("a", target(SinkRoute::Device, "cable (test)"));
    h.core.push("a", &vec![0.5; 1_600], 16_000, 1).unwrap();
    h.core.end("a").unwrap();
    let out = h.run("Cable (Test)", 200);
    assert!(out.iter().filter(|&&s| (s - 0.5).abs() < 1e-6).count() > 1_400);
    assert_eq!(h.events.last("a").unwrap().state, UtteranceState::Done);
    assert_eq!(h.core.status().devices, vec!["Cable (Test)".to_string()]);

    h.begin("b", target(SinkRoute::Device, "Missing"));
    h.begin("c", target(SinkRoute::Device, ""));
    h.core.push("b", &stereo(480, 0.5), 48_000, 2).unwrap();
    h.core.end("b").unwrap();
    h.core.push("c", &stereo(4_800, 0.25), 48_000, 2).unwrap();
    h.core.end("c").unwrap();
    let out = h.run(DEV, 200);
    let b = h.events.last("b").unwrap();
    assert_eq!(b.state, UtteranceState::Error);
    assert_eq!(b.error.as_deref(), Some("audio output: Output device 'Missing' not found"));
    assert!(count_level(&out, 0.25, 1e-6) > 4_000, "the next utterance plays");
}

/// A lost device fails the utterance playing on it; the next one opens the device again.
#[test]
fn a_failed_output_is_reopened_for_the_next_utterance() {
    let mut h = Harness::device();
    h.begin("a", target(SinkRoute::Device, ""));
    h.core.push("a", &stereo(48_000, 0.5), 48_000, 2).unwrap();
    h.core.end("a").unwrap();
    h.run(DEV, 50);
    h.outputs.shared(DEV).fail("device unplugged".into());
    h.run(DEV, 20);
    let a = h.events.last("a").unwrap();
    assert_eq!(a.state, UtteranceState::Error);
    assert!(a.error.unwrap().contains("device unplugged"));
    h.begin("b", target(SinkRoute::Device, ""));
    h.core.push("b", &stereo(4_800, 0.25), 48_000, 2).unwrap();
    h.core.end("b").unwrap();
    let out = h.run(DEV, 150);
    assert!(count_level(&out, 0.25, 1e-6) > 4_000);
    assert_eq!(h.outputs.opens(), 2);
}

/// A head whose producer went silent fails after the stall timeout (later speech is not blocked
/// forever); an utterance ended without audio is done without opening a device.
#[test]
fn stalled_and_empty_utterances() {
    let cfg = SinkConfig {
        stall_timeout: Duration::from_secs(1),
        first_audio_timeout: Duration::from_secs(5),
        ..SinkConfig::default()
    };
    let mut h = Harness::new(cfg, Box::new(NoVc));
    h.begin("empty", target(SinkRoute::Device, ""));
    h.core.end("empty").unwrap();
    h.core.step(h.now);
    assert_eq!(h.events.states("empty"), vec![UtteranceState::Queued, UtteranceState::Done]);
    assert_eq!(h.outputs.opens(), 0);

    // Audio started, then the producer died mid-sentence: the stall timeout.
    h.begin("died", target(SinkRoute::Device, ""));
    h.core.push("died", &stereo(480, 0.5), 48_000, 2).unwrap();
    h.begin("next", target(SinkRoute::Device, ""));
    h.core.push("next", &stereo(4_800, 0.25), 48_000, 2).unwrap();
    h.core.end("next").unwrap();
    h.run(DEV, 50);
    h.now += Duration::from_millis(1_100);
    let out = h.run(DEV, 150);
    let died = h.events.last("died").unwrap();
    assert_eq!(died.state, UtteranceState::Error);
    assert_eq!(died.error.as_deref(), Some("timed out waiting for audio"));
    assert!(count_level(&out, 0.25, 1e-6) > 4_000);
}

/// Review finding: the pipeline begins an utterance before its sentence is synthesized, so a
/// head without any audio yet is still synthesizing (a self-hosted server on a weak CPU may take
/// minutes). It is not dropped by the stall timeout; only the much longer first-audio bound ends
/// it, and late audio then plays normally.
#[test]
fn slow_first_audio_plays_late_instead_of_failing() {
    let cfg = SinkConfig {
        stall_timeout: Duration::from_secs(1),
        first_audio_timeout: Duration::from_secs(5),
        ..SinkConfig::default()
    };
    let mut h = Harness::new(cfg, Box::new(NoVc));
    h.begin("slow", target(SinkRoute::Device, ""));
    h.core.step(h.now);
    // Three stall timeouts pass while the sentence synthesizes.
    h.now += Duration::from_secs(3);
    h.core.step(h.now);
    assert_eq!(h.events.states("slow"), vec![UtteranceState::Queued]);
    h.core.push("slow", &stereo(4_800, 0.25), 48_000, 2).unwrap();
    h.core.end("slow").unwrap();
    let out = h.run(DEV, 150);
    assert!(count_level(&out, 0.25, 1e-6) > 4_000);
    h.run(DEV, 100);
    assert_eq!(h.events.last("slow").unwrap().state, UtteranceState::Done);

    // A producer that never delivers anything still fails eventually, and later speech plays.
    h.begin("never", target(SinkRoute::Device, ""));
    h.begin("after", target(SinkRoute::Device, ""));
    h.core.push("after", &stereo(4_800, -0.25), 48_000, 2).unwrap();
    h.core.end("after").unwrap();
    h.core.step(h.now);
    h.now += Duration::from_secs(4);
    h.core.step(h.now);
    assert_eq!(h.events.states("never"), vec![UtteranceState::Queued]);
    h.now += Duration::from_millis(1_100);
    let out = h.run(DEV, 150);
    assert_eq!(h.events.last("never").unwrap().error.as_deref(), Some("timed out waiting for audio"));
    assert!(count_level(&out, -0.25, 1e-6) > 4_000);
}

/// The default first-audio bound covers every backend's synthesis budget (self-hosted servers
/// get up to `LOCAL_TTS_TIMEOUT_MAX`; VOICEVOX 120 s + 120 s), and the stall timeout stays short.
#[test]
fn default_first_audio_bound_covers_synthesis_budgets() {
    let cfg = SinkConfig::default();
    assert!(cfg.first_audio_timeout > crate::services::http_client::LOCAL_TTS_TIMEOUT_MAX);
    assert!(cfg.first_audio_timeout > Duration::from_secs(240));
    assert!(cfg.stall_timeout < cfg.first_audio_timeout);
}

#[test]
fn producer_errors_are_specific() {
    let h = Harness::device();
    h.begin("a", target(SinkRoute::Device, ""));
    let bad = |r: Result<(), SinkError>| matches!(r, Err(SinkError::InvalidFormat(_)));
    assert!(bad(h.core.push("a", &[0.0; 4], 1_000, 1)), "rate");
    assert!(bad(h.core.push("a", &[0.0; 4], 48_000, 0)), "channels");
    assert!(bad(h.core.push("a", &[0.0; 3], 48_000, 2)), "partial frame");
    assert!(bad(h.core.begin(UtteranceSpec {
        id: "a".into(),
        target: SinkTarget::default()
    })));
    assert!(bad(h.core.begin(UtteranceSpec {
        id: " ".into(),
        target: SinkTarget::default()
    })));
    assert!(matches!(h.core.push("zz", &[0.0; 2], 48_000, 1), Err(SinkError::NotFound(_))));
    assert!(matches!(h.core.end("zz"), Err(SinkError::NotFound(_))));
    h.core.end("a").unwrap();
    assert!(matches!(h.core.push("a", &[0.0; 2], 48_000, 1), Err(SinkError::NotFound(_))), "after end");
    // A writer sees cancellation.
    h.begin("w", target(SinkRoute::Device, ""));
    let mut w = UtteranceWriter {
        id: "w".into(),
        sink: h.core.clone(),
        cancelled: h.core.cancel_flag("w").unwrap(),
    };
    w.push(&[0.1; 480], 48_000, 1).unwrap();
    assert!(!w.is_cancelled());
    h.core.cancel(Some("w"));
    assert!(w.is_cancelled());
    assert_eq!(w.push(&[0.1; 480], 48_000, 1), Err(SinkError::Cancelled));
    assert!(matches!(h.core.cancel_flag("w"), Err(SinkError::Cancelled)));
    // An id may be reused after it was cancelled.
    h.begin("w", target(SinkRoute::Device, ""));
}

/// Status events are coalesced to at most one per interval and report the queue.
#[test]
fn status_events_are_throttled() {
    let mut h = Harness::device();
    for i in 0..5 {
        h.begin(&format!("u{i}"), target(SinkRoute::Device, ""));
        h.core.push(&format!("u{i}"), &stereo(2_400, 0.5), 48_000, 2).unwrap();
        h.core.end(&format!("u{i}")).unwrap();
    }
    let status = h.core.status();
    assert_eq!(status.queued.len(), 5);
    assert!(status.queued_ms >= 250, "{}", status.queued_ms);
    h.run(DEV, 400);
    let statuses = h.events.statuses.lock().unwrap().clone();
    assert!(!statuses.is_empty());
    // Driven with a simulated clock: 400 ms of scheduling may emit at most ~2-3 updates.
    assert!(statuses.len() <= 3, "{} status events", statuses.len());
}

/// A route change between sentences keeps the order: a voice-changer sentence waits until the
/// device sentence before it finished playing (and the other way round).
#[test]
fn route_changes_wait_for_earlier_audio() {
    let vc = FakeVc::running(16_000, 1, 16_000 * 10);
    let mut h = Harness::new(SinkConfig::default(), Box::new(vc.clone()));
    h.begin("dev", target(SinkRoute::Device, ""));
    h.begin("vc", target(SinkRoute::VoiceChanger, ""));
    h.core.push("dev", &stereo(14_400, 0.5), 48_000, 2).unwrap(); // 300 ms
    h.core.end("dev").unwrap();
    h.core.push("vc", &vec![0.25; 1_600], 16_000, 1).unwrap();
    h.core.end("vc").unwrap();
    h.run(DEV, 100);
    assert_eq!(vc.queued(), 0, "the voice-changer sentence waits for the device one");
    h.run(DEV, 300);
    assert_eq!(h.events.last("dev").unwrap().state, UtteranceState::Done);
    assert_eq!(vc.queued(), 1_600);
    // And back: a device sentence waits while the voice changer still holds audio.
    h.begin("dev2", target(SinkRoute::Device, ""));
    h.core.push("dev2", &stereo(4_800, -0.5), 48_000, 2).unwrap();
    h.core.end("dev2").unwrap();
    let out = h.run(DEV, 50);
    assert_eq!(count_level(&out, -0.5, 1e-6), 0);
    vc.consume(1_600);
    let out = h.run(DEV, 150);
    assert!(count_level(&out, -0.5, 1e-6) > 4_000);
}

/// IPC bodies are handed over without another copy; non-finite samples become silence.
#[test]
fn owned_pushes_are_sanitized() {
    let mut h = Harness::device();
    h.begin("a", target(SinkRoute::Device, ""));
    h.core
        .push_owned("a", vec![0.5, 0.5, f32::NAN, f32::INFINITY], 48_000, 2)
        .unwrap();
    assert!(matches!(h.core.push_owned("a", vec![0.5; 3], 48_000, 2), Err(SinkError::InvalidFormat(_))));
    h.core.end("a").unwrap();
    let out = h.run(DEV, 30);
    assert!(out.iter().all(|s| s.is_finite()));
    assert_eq!(h.events.last("a").unwrap().state, UtteranceState::Done);
}
