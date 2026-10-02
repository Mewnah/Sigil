//! SAPI into PCM: one native `ISpVoice` renders each sentence into a write-only stream
//! ([`super::stream`]) whose writes go straight into a TTS sink utterance ([`PcmOut`]).
//!
//! Runs only on the SAPI STA thread ([`super::sta`]): the voice, its tokens and the engines it
//! loads live in that apartment, and the wait loop pumps its messages (calls from SAPI's worker
//! thread into an apartment-threaded engine arrive as window messages).
//!
//! # One sentence
//! 1. Voice: the request's token id (`""` = the Windows default voice); `SetVoice` only on change.
//! 2. Format: the engine's own output format (`ISpTTSEngine::GetOutputFormat`, cached per voice),
//!    as 16-bit PCM at the engine's rate and channels, so SAPI never resamples (the sink does it,
//!    band-limited). The Windows 11 desktop voices render 16 kHz mono. Unknown: 22.05 kHz mono.
//! 3. `SpStream` over the PCM stream, `SetOutput(stream, FALSE)`, rate, volume 100 (the sink applies
//!    the user's volume), then an asynchronous `Speak` (plain text, never SAPI XML).
//! 4. SAPI writes from its own thread, much faster than real time (about 140x on the desktop
//!    voices); each write is decoded and pushed into the utterance in about 50 ms pieces, so the
//!    first audio arrives within milliseconds and the sink's back-pressure paces long sentences.
//! 5. Ends when SAPI is done; then the rest is flushed and the stream released.
//!
//! # Cancel and failure
//! - A cancelled utterance (stop / skip, or the job's own cancel flag) makes the next write fail
//!   (`E_ABORT`), which ends the speech at once; the loop also issues an asynchronous purge. A voice
//!   that does not finish within [`PURGE_GRACE`] after that is wedged and is replaced.
//! - A voice that writes nothing for [`FIRST_AUDIO_TIMEOUT`] (before its first audio) or
//!   [`STALL_TIMEOUT`] (after it), while not waiting for room in the sink, is stalled: purged and
//!   reported.

use std::collections::HashMap;
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::sync::{Arc, Mutex, MutexGuard, PoisonError};
use std::time::{Duration, Instant};

use windows::core::{IUnknown, Interface, GUID, HRESULT, PCWSTR, PWSTR};
use windows::Win32::Foundation::{E_ABORT, S_FALSE, S_OK, WAIT_FAILED};
use windows::Win32::Media::Audio::WAVEFORMATEX;
use windows::Win32::Media::Speech::{
    ISpObjectToken, ISpStream, ISpTTSEngine, ISpVoice, SpObjectToken, SpStream, SpVoice, SPF_ASYNC, SPF_IS_NOT_XML,
    SPF_PURGEBEFORESPEAK, SPVOICESTATUS,
};
use windows::Win32::System::Com::{CoCreateInstance, CoTaskMemFree, CLSCTX_ALL};
use windows::Win32::UI::WindowsAndMessaging::{MsgWaitForMultipleObjects, QS_ALLINPUT};

use super::pcm::{PcmAssembler, PcmFormat, PcmOut, BYTES_PER_SAMPLE, FALLBACK_FORMAT, WAVE_FORMAT_PCM};
use super::stream::{new_stream, StreamSink};
use crate::services::tts_sink::{SinkError, SpeakReport};

/// `SPDFID_WaveFormatEx`: the SAPI stream format id of a `WAVEFORMATEX` stream (not in the
/// `windows` crate's metadata; checked against a SAPI memory stream by a test).
pub(crate) const SPDFID_WAVEFORMATEX: GUID = GUID::from_u128(0xc31adbae_527f_4ff5_a230_f62bb61ff70c);

/// Audio collected before a push into the sink.
const PUSH_MS: u32 = 50;
/// Wait slice of the render loop (messages are pumped in between).
const WAIT_SLICE_MS: u32 = 10;
/// A voice that produced no audio for this long after `Speak` is stalled (loading a large voice
/// takes a few seconds at most).
pub(crate) const FIRST_AUDIO_TIMEOUT: Duration = Duration::from_secs(30);
/// A voice that stopped writing for this long (while the sink had room) is stalled.
pub(crate) const STALL_TIMEOUT: Duration = Duration::from_secs(15);
/// How long a purged voice may take to finish before it counts as wedged.
pub(crate) const PURGE_GRACE: Duration = Duration::from_secs(3);

/// The target format offered to an engine that asks for one (`GetOutputFormat`'s target; the
/// desktop voices ignore it and report their own).
const PROBE_TARGET: PcmFormat = FALLBACK_FORMAT;

/// One sentence to render.
#[derive(Clone, Debug, Default, PartialEq)]
pub(crate) struct RenderRequest {
    /// SAPI voice token id (`""` = the Windows default voice).
    pub voice: String,
    pub text: String,
    /// SAPI rate -10..=10.
    pub rate: i32,
}

/// Why a render did not complete.
#[derive(Clone, Debug, PartialEq, Eq)]
pub(crate) enum RenderError {
    /// Stop / skip: not a failure.
    Cancelled,
    /// This sentence failed; the voice is fine.
    Failed(String),
    /// The voice did not finish after a purge: it must be replaced before the next sentence.
    Wedged(String),
}

impl RenderError {
    /// The IPC error text (`"cancelled"` is what TS treats as a stop / skip).
    pub(crate) fn message(&self) -> String {
        match self {
            RenderError::Cancelled => SinkError::Cancelled.to_string(),
            RenderError::Failed(m) | RenderError::Wedged(m) => m.clone(),
        }
    }
}

fn lock<T>(m: &Mutex<T>) -> MutexGuard<'_, T> {
    m.lock().unwrap_or_else(PoisonError::into_inner)
}

fn ms(d: Duration) -> u32 {
    d.as_millis().min(u128::from(u32::MAX)) as u32
}

/// A NUL-terminated UTF-16 copy of `s` (interior NULs would end the string early: dropped).
fn wide(s: &str) -> Vec<u16> {
    s.encode_utf16().filter(|&c| c != 0).chain(Some(0)).collect()
}

fn hr_text(hr: HRESULT) -> String {
    format!("0x{:08X}", hr.0 as u32)
}

// ---------------------------------------------------------------------------------------------
// The render target: SAPI's writes into the sink
// ---------------------------------------------------------------------------------------------

struct TargetInner {
    assembler: PcmAssembler,
    /// Taken at the end (the sink utterance is not held after the sentence).
    out: Option<Box<dyn PcmOut>>,
    /// Why writes stopped (the sink refused).
    error: Option<SinkError>,
    first_audio: Option<Duration>,
    last_audio: Option<Duration>,
}

/// Bridges SAPI's writes (its thread) into the sink and tells the render loop how it goes.
pub(crate) struct RenderTarget {
    inner: Mutex<TargetInner>,
    start: Instant,
    /// Milliseconds after `start` of the latest write (`u64::MAX`: none yet).
    last_write_ms: AtomicU64,
    /// A push is waiting for room in the sink (back-pressure is not a stall).
    in_push: AtomicBool,
    /// Writes are refused from now on (cancel).
    stopped: AtomicBool,
}

impl RenderTarget {
    pub(crate) fn new(format: PcmFormat, out: Box<dyn PcmOut>, start: Instant) -> Self {
        Self {
            inner: Mutex::new(TargetInner {
                assembler: PcmAssembler::new(format, PUSH_MS),
                out: Some(out),
                error: None,
                first_audio: None,
                last_audio: None,
            }),
            start,
            last_write_ms: AtomicU64::new(u64::MAX),
            in_push: AtomicBool::new(false),
            stopped: AtomicBool::new(false),
        }
    }

    /// Refuse further writes (the speech aborts at its next write).
    pub(crate) fn stop(&self) {
        self.stopped.store(true, Ordering::Release);
    }

    /// Whether SAPI may count as stalled at `now`: no write for the applicable timeout while the
    /// sink was not applying back-pressure.
    pub(crate) fn stalled(&self, now: Instant) -> bool {
        if self.in_push.load(Ordering::Acquire) {
            return false;
        }
        let since_start = now.saturating_duration_since(self.start);
        match self.last_write_ms.load(Ordering::Acquire) {
            u64::MAX => since_start >= FIRST_AUDIO_TIMEOUT,
            last => since_start.saturating_sub(Duration::from_millis(last)) >= STALL_TIMEOUT,
        }
    }

    /// The sink error that stopped the writes, if any.
    pub(crate) fn error(&self) -> Option<SinkError> {
        lock(&self.inner).error.clone()
    }

    /// Flush what is pending and release the utterance. Returns the report, or the sink error.
    pub(crate) fn finish(&self) -> Result<SpeakReport, SinkError> {
        let mut inner = lock(&self.inner);
        if let Some(e) = inner.error.clone() {
            inner.out = None;
            return Err(e);
        }
        let mut out = inner.out.take();
        let flushed = match out.as_deref_mut() {
            Some(o) => inner.assembler.finish(o),
            None => Ok(0),
        };
        flushed?;
        if inner.first_audio.is_none() && inner.assembler.frames() > 0 {
            let now = self.start.elapsed();
            inner.first_audio = Some(now);
            inner.last_audio = Some(now);
        }
        let total = inner.last_audio.unwrap_or_else(|| self.start.elapsed());
        Ok(SpeakReport {
            sample_rate: inner.assembler.format().rate,
            frames: inner.assembler.frames(),
            first_audio_ms: ms(inner.first_audio.unwrap_or(total)),
            total_ms: ms(total),
        })
    }
}

impl StreamSink for RenderTarget {
    fn write(&self, bytes: &[u8]) -> bool {
        if self.stopped.load(Ordering::Acquire) {
            return false;
        }
        let mut inner = lock(&self.inner);
        if inner.error.is_some() {
            return false;
        }
        self.last_write_ms.store(ms(self.start.elapsed()).into(), Ordering::Release);
        let TargetInner {
            assembler,
            out,
            error,
            first_audio,
            last_audio,
        } = &mut *inner;
        let Some(out) = out.as_deref_mut() else {
            return false;
        };
        self.in_push.store(true, Ordering::Release);
        let pushed = assembler.write(bytes, out);
        self.in_push.store(false, Ordering::Release);
        // Leaving the push counts as progress (a long back-pressure wait is not a stall).
        self.last_write_ms.store(ms(self.start.elapsed()).into(), Ordering::Release);
        match pushed {
            Ok(0) => true,
            Ok(_) => {
                let now = self.start.elapsed();
                first_audio.get_or_insert(now);
                *last_audio = Some(now);
                true
            }
            Err(e) => {
                *error = Some(e);
                false
            }
        }
    }
}

// ---------------------------------------------------------------------------------------------
// The renderer
// ---------------------------------------------------------------------------------------------

/// The STA thread's SAPI voice for rendering into the sink.
pub(crate) struct Renderer {
    voice: ISpVoice,
    /// Token id of the voice set now (`""`: the default voice).
    current: Option<String>,
    /// Render format per voice token id.
    formats: HashMap<String, PcmFormat>,
}

impl Renderer {
    /// A new SAPI voice (COM must be initialized on this thread).
    pub(crate) fn new() -> Result<Self, String> {
        // SAFETY: COM is initialized on the calling (STA) thread.
        let voice: ISpVoice = unsafe { CoCreateInstance(&SpVoice, None, CLSCTX_ALL) }
            .map_err(|e| format!("Windows speech is not available: {}", e.message()))?;
        Ok(Self {
            voice,
            current: None,
            formats: HashMap::new(),
        })
    }

    /// Make `voice` (token id, `""` = default) the current voice; returns its token.
    fn select_voice(&mut self, voice: &str) -> Result<ISpObjectToken, RenderError> {
        let voice = voice.trim();
        if self.current.as_deref() != Some(voice) {
            if voice.is_empty() {
                // SAFETY: a live voice on its own thread; `None` selects the default voice.
                unsafe { self.voice.SetVoice(None::<&ISpObjectToken>) }
            } else {
                let token = token_by_id(voice)?;
                // SAFETY: as above.
                unsafe { self.voice.SetVoice(&token) }
            }
            .map_err(|e| RenderError::Failed(format!("Could not select the Windows voice: {}", e.message())))?;
            self.current = Some(voice.to_string());
        }
        // SAFETY: a live voice; returns its current token.
        unsafe { self.voice.GetVoice() }
            .map_err(|e| RenderError::Failed(format!("Could not read the Windows voice: {}", e.message())))
    }

    /// The render format of `token` (probed once per voice).
    fn format_of(&mut self, token: &ISpObjectToken) -> PcmFormat {
        let id = token_id(token).unwrap_or_default();
        if let Some(f) = self.formats.get(&id) {
            return *f;
        }
        let format = probe_format(token).unwrap_or_else(|e| {
            tracing::info!(target: "sigil::tts", "Windows speech: the voice did not report its format ({e}); using {} Hz", FALLBACK_FORMAT.rate);
            FALLBACK_FORMAT
        });
        tracing::debug!(target: "sigil::tts", "Windows speech: voice {id:?} renders {} Hz, {} channel(s)", format.rate, format.channels);
        self.formats.insert(id, format);
        format
    }

    /// Render `request` into `out`. `cancelled` is polled between waits; `received` is when the
    /// request arrived (the report's times count from it).
    pub(crate) fn render(
        &mut self,
        request: &RenderRequest,
        out: Box<dyn PcmOut>,
        cancelled: &dyn Fn() -> bool,
        received: Instant,
    ) -> Result<SpeakReport, RenderError> {
        if cancelled() {
            return Err(RenderError::Cancelled);
        }
        let text = request.text.trim();
        let token = self.select_voice(&request.voice)?;
        let format = self.format_of(&token);
        let target = Arc::new(RenderTarget::new(format, out, received));
        if text.is_empty() {
            return target.finish().map_err(sink_failure);
        }
        let spoken = self.speak_into(&target, format, text, request.rate, cancelled);
        // Release the stream (and with it the utterance) whatever happened; never fails in
        // practice, and a failure only keeps the stream alive until the next sentence. A wedged
        // voice is left alone: it is still busy, and calling into it could block this thread (the
        // caller abandons it; its stream refuses every write from now on).
        if !matches!(spoken, Err(RenderError::Wedged(_))) {
            // SAFETY: a live voice on its own thread.
            let _ = unsafe { self.voice.SetOutput(None::<&IUnknown>, false) };
        }
        spoken?;
        target.finish().map_err(sink_failure)
    }

    fn speak_into(
        &mut self,
        target: &Arc<RenderTarget>,
        format: PcmFormat,
        text: &str,
        rate: i32,
        cancelled: &dyn Fn() -> bool,
    ) -> Result<(), RenderError> {
        let failed = |what: &str, e: windows::core::Error| RenderError::Failed(format!("Windows speech: {what} failed ({})", e.message()));
        let stream = new_stream(target.clone());
        let wave = wave_format(format);
        // SAFETY: plain COM calls on this STA thread; `wave` outlives `SetBaseStream`, which copies it.
        unsafe {
            let sp: ISpStream = CoCreateInstance(&SpStream, None, CLSCTX_ALL).map_err(|e| failed("creating the stream", e))?;
            sp.SetBaseStream(&stream, &SPDFID_WAVEFORMATEX, &wave)
                .map_err(|e| failed("setting the stream format", e))?;
            self.voice.SetOutput(&sp, false).map_err(|e| failed("setting the output", e))?;
            self.voice.SetRate(rate.clamp(-10, 10)).map_err(|e| failed("setting the rate", e))?;
            self.voice.SetVolume(100).map_err(|e| failed("setting the volume", e))?;
            let wtext = wide(text);
            self.voice
                .Speak(PCWSTR(wtext.as_ptr()), (SPF_ASYNC.0 | SPF_IS_NOT_XML.0) as u32, None)
                .map_err(|e| failed("speaking", e))?;
        }
        self.wait(target, cancelled)
    }

    /// Wait until SAPI is done, pumping messages; purge on cancel or stall. Polls the voice every
    /// [`WAIT_SLICE_MS`] (its completion event stays signalled between sentences, so it cannot
    /// pace this loop).
    fn wait(&mut self, target: &RenderTarget, cancelled: &dyn Fn() -> bool) -> Result<(), RenderError> {
        let mut purge: Option<(Instant, RenderError)> = None;
        loop {
            // SAFETY: no handles; returns after the slice or when a message arrives.
            let woke = unsafe { MsgWaitForMultipleObjects(None, false, WAIT_SLICE_MS, QS_ALLINPUT) };
            if woke == WAIT_FAILED {
                std::thread::sleep(Duration::from_millis(u64::from(WAIT_SLICE_MS)));
            }
            super::sta::pump_messages();
            // SAFETY: a live voice; a zero timeout only polls (S_OK: done, S_FALSE: still speaking).
            let hr = unsafe { (Interface::vtable(&self.voice).WaitUntilDone)(Interface::as_raw(&self.voice), 0) };
            if hr == S_OK {
                break;
            }
            if hr != S_FALSE {
                self.purge();
                return Err(RenderError::Failed(format!("Windows speech failed while speaking ({})", hr_text(hr))));
            }
            let now = Instant::now();
            match &purge {
                None => {
                    let why = if cancelled() {
                        Some(RenderError::Cancelled)
                    } else if target.stalled(now) {
                        Some(RenderError::Failed("Windows speech stopped responding".into()))
                    } else {
                        None
                    };
                    if let Some(why) = why {
                        target.stop();
                        self.purge();
                        purge = Some((now, why));
                    }
                }
                Some((at, why)) if now.saturating_duration_since(*at) >= PURGE_GRACE => {
                    let reason = match why {
                        RenderError::Cancelled => "did not stop when cancelled".to_string(),
                        other => other.message(),
                    };
                    return Err(RenderError::Wedged(format!("The Windows voice {reason}; it will be reloaded")));
                }
                Some(_) => {}
            }
        }
        if let Some((_, why)) = purge {
            return Err(why);
        }
        if let Some(e) = target.error() {
            return Err(sink_failure(e));
        }
        match self.last_result() {
            status if status == E_ABORT && cancelled() => Err(RenderError::Cancelled),
            // Only Sigil's stream aborts (cancel / sink error, handled above): anything else cut
            // the sentence short.
            status if status == E_ABORT => Err(RenderError::Failed("Windows speech was interrupted".into())),
            status if status.is_err() => Err(RenderError::Failed(format!("Windows speech failed ({})", hr_text(status)))),
            _ => Ok(()),
        }
    }

    /// Drop everything the voice is speaking (asynchronous: never blocks on a busy engine).
    fn purge(&self) {
        // SAFETY: a live voice; a NULL text with the purge flag only purges.
        if let Err(e) = unsafe { self.voice.Speak(PCWSTR::null(), (SPF_PURGEBEFORESPEAK.0 | SPF_ASYNC.0) as u32, None) } {
            tracing::warn!(target: "sigil::tts", "Windows speech: purge failed: {}", e.message());
        }
    }

    /// `hrLastResult` of the voice (S_OK when unknown).
    fn last_result(&self) -> HRESULT {
        let mut status = SPVOICESTATUS::default();
        let mut bookmark = PWSTR::null();
        // SAFETY: valid out pointers; the bookmark string (if any) is freed below.
        let ok = unsafe { self.voice.GetStatus(&mut status, &mut bookmark) };
        if !bookmark.is_null() {
            // SAFETY: SAPI allocated it with CoTaskMemAlloc.
            unsafe { CoTaskMemFree(Some(bookmark.0 as *const _)) };
        }
        if ok.is_ok() {
            status.hrLastResult
        } else {
            S_OK
        }
    }
}

fn sink_failure(e: SinkError) -> RenderError {
    match e {
        SinkError::Cancelled => RenderError::Cancelled,
        other => RenderError::Failed(other.to_string()),
    }
}

/// The 16-bit PCM `WAVEFORMATEX` of `format`.
fn wave_format(format: PcmFormat) -> WAVEFORMATEX {
    WAVEFORMATEX {
        wFormatTag: WAVE_FORMAT_PCM,
        nChannels: format.channels,
        nSamplesPerSec: format.rate,
        nAvgBytesPerSec: format.avg_bytes_per_sec(),
        nBlockAlign: format.block_align(),
        wBitsPerSample: BYTES_PER_SAMPLE * 8,
        cbSize: 0,
    }
}

/// The SAPI token with id `id` (a voice that is not installed fails with a readable error).
fn token_by_id(id: &str) -> Result<ISpObjectToken, RenderError> {
    let missing = |_| RenderError::Failed("The selected Windows voice is not installed".into());
    // SAFETY: plain COM calls on this STA thread; `wid` outlives `SetId`.
    unsafe {
        let token: ISpObjectToken = CoCreateInstance(&SpObjectToken, None, CLSCTX_ALL)
            .map_err(|e| RenderError::Failed(format!("Windows speech is not available: {}", e.message())))?;
        let wid = wide(id);
        token.SetId(PCWSTR::null(), PCWSTR(wid.as_ptr()), false).map_err(missing)?;
        Ok(token)
    }
}

/// The token's id.
fn token_id(token: &ISpObjectToken) -> Option<String> {
    // SAFETY: SAPI returns a CoTaskMemAlloc'd string, freed here.
    unsafe {
        let id = token.GetId().ok()?;
        let s = id.to_string().ok();
        CoTaskMemFree(Some(id.0 as *const _));
        s
    }
}

/// The voice engine's own output format (its engine is created once for the question).
fn probe_format(token: &ISpObjectToken) -> Result<PcmFormat, String> {
    // SAFETY: plain COM calls on this STA thread; the returned WAVEFORMATEX is freed here.
    unsafe {
        let mut raw = std::ptr::null_mut();
        token
            .CreateInstance(None::<&IUnknown>, CLSCTX_ALL.0, &ISpTTSEngine::IID, &mut raw)
            .map_err(|e| format!("creating the engine: {}", e.message()))?;
        if raw.is_null() {
            return Err("the engine was not created".into());
        }
        let engine = ISpTTSEngine::from_raw(raw);
        let target = wave_format(PROBE_TARGET);
        let mut format_id = GUID::zeroed();
        let mut wave: *mut WAVEFORMATEX = std::ptr::null_mut();
        let asked = engine.GetOutputFormat(&SPDFID_WAVEFORMATEX, &target, &mut format_id, &mut wave);
        let reported = (!wave.is_null()).then(|| *wave);
        if !wave.is_null() {
            CoTaskMemFree(Some(wave as *const _));
        }
        asked.map_err(|e| format!("GetOutputFormat: {}", e.message()))?;
        let w = reported.ok_or("no format")?;
        if format_id != SPDFID_WAVEFORMATEX {
            return Err(format!("format id {format_id:?}"));
        }
        PcmFormat::from_engine(w.wFormatTag, w.nChannels, w.nSamplesPerSec, w.wBitsPerSample).ok_or_else(|| {
            format!(
                "unsupported format (tag {}, {} ch, {} Hz, {} bit)",
                { w.wFormatTag },
                { w.nChannels },
                { w.nSamplesPerSec },
                { w.wBitsPerSample }
            )
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use windows::Win32::Media::Speech::{ISpStreamFormat, SpMemoryStream};

    /// Collects pushed audio; can refuse (cancel) after some pushes.
    #[derive(Clone, Default)]
    struct Collect(Arc<Mutex<CollectInner>>);

    #[derive(Default)]
    struct CollectInner {
        samples: Vec<f32>,
        rate: u32,
        channels: u16,
        pushes: usize,
        cancel_after: Option<usize>,
    }

    impl PcmOut for Collect {
        fn push(&mut self, samples: &[f32], sample_rate: u32, channels: u16) -> Result<(), SinkError> {
            let mut i = self.0.lock().unwrap();
            if i.cancel_after.is_some_and(|n| i.pushes >= n) {
                return Err(SinkError::Cancelled);
            }
            i.pushes += 1;
            i.samples.extend_from_slice(samples);
            i.rate = sample_rate;
            i.channels = channels;
            Ok(())
        }
    }

    #[test]
    fn errors_map_to_the_ipc_texts() {
        assert_eq!(RenderError::Cancelled.message(), "cancelled");
        assert_eq!(sink_failure(SinkError::Cancelled), RenderError::Cancelled);
        assert!(matches!(sink_failure(SinkError::Device("x".into())), RenderError::Failed(m) if m.contains("x")));
        assert_eq!(wide("a\0b"), vec![u16::from(b'a'), u16::from(b'b'), 0]);
        assert_eq!(hr_text(E_ABORT), "0x80004004");
        let w = wave_format(PcmFormat { rate: 16_000, channels: 1 });
        assert_eq!(({ w.nBlockAlign }, { w.nAvgBytesPerSec }, { w.wBitsPerSample }), (2, 32_000, 16));
    }

    #[test]
    fn target_reports_progress_errors_and_stalls() {
        let start = Instant::now();
        let out = Collect::default();
        let t = RenderTarget::new(PcmFormat { rate: 16_000, channels: 1 }, Box::new(out.clone()), start);
        assert!(!t.stalled(start));
        assert!(t.stalled(start + FIRST_AUDIO_TIMEOUT), "no audio at all");
        let bytes: Vec<u8> = (0..1_600i16).flat_map(|s| s.to_le_bytes()).collect(); // 100 ms
        assert!(t.write(&bytes));
        // After audio the bound is the (shorter) stall timeout, counted from the last write.
        assert!(!t.stalled(start + STALL_TIMEOUT - Duration::from_secs(1)));
        assert!(t.stalled(Instant::now() + STALL_TIMEOUT));
        // Waiting for room in the sink is never a stall.
        t.in_push.store(true, Ordering::Release);
        assert!(!t.stalled(Instant::now() + FIRST_AUDIO_TIMEOUT * 10));
        t.in_push.store(false, Ordering::Release);
        let report = t.finish().unwrap();
        assert_eq!((report.sample_rate, report.frames), (16_000, 1_600));
        assert_eq!(out.0.lock().unwrap().samples.len(), 1_600);
        // After finish the utterance is released: writes are refused.
        assert!(!t.write(&bytes));

        // A cancelled sink stops the writes and is the result.
        let out = Collect::default();
        out.0.lock().unwrap().cancel_after = Some(0);
        let t = RenderTarget::new(PcmFormat { rate: 16_000, channels: 1 }, Box::new(out), start);
        assert!(!t.write(&bytes));
        assert_eq!(t.error(), Some(SinkError::Cancelled));
        assert_eq!(t.finish().unwrap_err(), SinkError::Cancelled);

        // stop() refuses writes at once.
        let t = RenderTarget::new(PcmFormat { rate: 16_000, channels: 1 }, Box::new(Collect::default()), start);
        t.stop();
        assert!(!t.write(&bytes));
        assert_eq!(t.finish().unwrap().frames, 0);
    }

    /// COM STA for a test thread (balanced on drop).
    struct Sta;
    impl Sta {
        fn init() -> Option<Sta> {
            use windows::Win32::System::Com::{CoInitializeEx, COINIT_APARTMENTTHREADED, COINIT_DISABLE_OLE1DDE};
            // SAFETY: COM initialization of the test thread.
            unsafe { CoInitializeEx(None, COINIT_APARTMENTTHREADED | COINIT_DISABLE_OLE1DDE) }
                .is_ok()
                .then_some(Sta)
        }
    }
    impl Drop for Sta {
        fn drop(&mut self) {
            // SAFETY: balances the successful CoInitializeEx above.
            unsafe { windows::Win32::System::Com::CoUninitialize() };
        }
    }

    #[test]
    fn the_wave_format_id_matches_sapi() {
        let Some(_sta) = Sta::init() else { return };
        // SAFETY: COM calls on this STA thread; the format is freed here.
        unsafe {
            let Ok(mem) = CoCreateInstance::<_, ISpStreamFormat>(&SpMemoryStream, None, CLSCTX_ALL) else {
                eprintln!("SAPI is not installed: skipped");
                return;
            };
            let mut id = GUID::zeroed();
            let wave = mem.GetFormat(&mut id as *mut GUID as *const GUID).unwrap();
            if !wave.is_null() {
                CoTaskMemFree(Some(wave as *const _));
            }
            assert_eq!(id, SPDFID_WAVEFORMATEX);
        }
    }

    /// The first installed SAPI voice, if any.
    fn installed_voice(renderer: &mut Renderer) -> Option<String> {
        let token = renderer.select_voice("").ok()?;
        token_id(&token)
    }

    #[test]
    fn real_sapi_renders_streams_and_cancels() {
        let Some(_sta) = Sta::init() else { return };
        let Ok(mut r) = Renderer::new() else {
            eprintln!("SAPI is not installed: skipped");
            return;
        };
        let Some(voice) = installed_voice(&mut r) else {
            eprintln!("no SAPI voice installed: skipped");
            return;
        };
        let req = RenderRequest {
            voice: voice.clone(),
            text: "Hello from Sigil. This sentence is rendered into memory.".into(),
            rate: 0,
        };
        let out = Collect::default();
        let report = r.render(&req, Box::new(out.clone()), &|| false, Instant::now()).unwrap();
        let got = out.0.lock().unwrap();
        assert!(got.rate >= 8_000 && (1..=2).contains(&got.channels), "{} Hz {} ch", got.rate, got.channels);
        let secs = got.samples.len() as f32 / got.channels as f32 / got.rate as f32;
        assert!((1.0..20.0).contains(&secs), "{secs} s of speech");
        let peak = got.samples.iter().fold(0f32, |m, s| m.max(s.abs()));
        assert!(peak > 0.05, "audible ({peak})");
        assert!(got.pushes > 1, "streamed in pieces ({})", got.pushes);
        assert_eq!(report.frames as usize, got.samples.len() / got.channels as usize);
        assert_eq!(report.sample_rate, got.rate);
        assert!(report.first_audio_ms <= report.total_ms);
        drop(got);

        // The same voice again: no voice switch, same format, and a faster rate is shorter.
        let fast = Collect::default();
        r.render(&RenderRequest { rate: 10, ..req.clone() }, Box::new(fast.clone()), &|| false, Instant::now())
            .unwrap();
        assert!(fast.0.lock().unwrap().samples.len() < out.0.lock().unwrap().samples.len());

        // The sink cancels after the first push: the speech ends at once with `Cancelled`.
        let cancel = Collect::default();
        cancel.0.lock().unwrap().cancel_after = Some(1);
        let long = RenderRequest {
            text: "One two three four five six seven eight nine ten. ".repeat(20),
            ..req.clone()
        };
        let t = Instant::now();
        let err = r.render(&long, Box::new(cancel.clone()), &|| false, Instant::now()).unwrap_err();
        assert_eq!(err, RenderError::Cancelled);
        assert!(t.elapsed() < Duration::from_secs(5), "{:?}", t.elapsed());
        assert_eq!(cancel.0.lock().unwrap().pushes, 1);

        // The job's cancel flag stops a render too, and the voice is usable afterwards.
        let flag = AtomicBool::new(false);
        let polled = Collect::default();
        let err = r
            .render(
                &long,
                Box::new(polled),
                &|| {
                    let was = flag.load(Ordering::Relaxed);
                    flag.store(true, Ordering::Relaxed);
                    was
                },
                Instant::now(),
            )
            .unwrap_err();
        assert_eq!(err, RenderError::Cancelled);
        let after = Collect::default();
        r.render(&req, Box::new(after.clone()), &|| false, Instant::now()).unwrap();
        assert!(!after.0.lock().unwrap().samples.is_empty());

        // Empty text: nothing rendered, not an error.
        let empty = r.render(&RenderRequest { text: "  ".into(), ..req.clone() }, Box::new(Collect::default()), &|| false, Instant::now());
        assert_eq!(empty.unwrap().frames, 0);

        // A voice that is not installed.
        let missing = RenderRequest {
            voice: r"HKEY_LOCAL_MACHINE\SOFTWARE\Microsoft\Speech\Voices\Tokens\SIGIL_NO_SUCH_VOICE".into(),
            ..req
        };
        let err = r.render(&missing, Box::new(Collect::default()), &|| false, Instant::now()).unwrap_err();
        assert!(matches!(err, RenderError::Failed(ref m) if m.contains("not installed")), "{err:?}");
    }
}
