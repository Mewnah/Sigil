//! Transcripts from the local speech-to-text, in the background after a save (zero-shot TTS
//! needs what is said in its reference; the voice changer does not).
//!
//! Engine choice: the speech router's plan for the clip's language with **installed models only**
//! (a library save never downloads), its utterance model first, then its streaming model; with no
//! routed model installed, the best installed Whisper model (English-only `.en` models only for a
//! clip known to be English). Language identification runs first when the language is unknown and
//! the chosen engine cannot detect it, or when no installed model detects it at all; without an
//! installed language-ID model the voice then gets `needs_language` (never a guessed language).
//! Engines are loaded for the job and dropped after it; jobs run one at a time, and a job whose
//! voice changed while it waited is skipped.
//!
//! A job's result is applied only while the voice still has the clip it was made from (`sha256`)
//! and nobody edited the transcript since the job was queued (an edit generation per voice), so a
//! slow transcription never overwrites the user's own text.

use std::collections::BTreeMap;
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Mutex, PoisonError};

use crate::services::speech::engine_api::{AsrEngine, DecodeOptions, EngineContext, EngineKind, EngineSpec, RouteRequest, Tier};
use crate::services::speech::{engines, lang, router};

const LOG: &str = "sigil::voice_library";

/// `VoiceMeta::transcript_error` codes.
pub const ERROR_NO_STT_MODEL: &str = "no_stt_model";
pub const ERROR_FAILED: &str = "failed";
pub const ERROR_EMPTY: &str = "empty";
/// The clip's language is unknown and no installed model detects it (e.g. only English-only
/// Whisper models): the user sets the language, then transcribes again.
pub const ERROR_NEEDS_LANGUAGE: &str = "needs_language";

/// Installed Whisper models tried when the router has nothing installed, best first (`.en` models
/// only for a clip known to be English).
const WHISPER_FALLBACK: &[&str] = &[
    "whisper/large-v3-turbo-q5_0",
    "whisper/large-v3-turbo-q8_0",
    "whisper/large-v3-turbo",
    "whisper/medium-q5_0",
    "whisper/medium-q8_0",
    "whisper/medium",
    "whisper/small-q5_1",
    "whisper/small-q8_0",
    "whisper/small",
    "whisper/medium.en-q5_0",
    "whisper/medium.en-q8_0",
    "whisper/medium.en",
    "whisper/small.en-q5_1",
    "whisper/small.en-q8_0",
    "whisper/small.en",
    "whisper/base-q5_1",
    "whisper/base-q8_0",
    "whisper/base",
    "whisper/base.en-q5_1",
    "whisper/base.en-q8_0",
    "whisper/base.en",
    "whisper/large-v3-q5_0",
    "whisper/large-v3",
    "whisper/tiny-q5_1",
    "whisper/tiny-q8_0",
    "whisper/tiny",
    "whisper/tiny.en-q5_1",
    "whisper/tiny.en-q8_0",
    "whisper/tiny.en",
];

/// What transcribes a 16 kHz clip: [`transcribe`] in the app, a fake in tests.
pub type TranscribeFn = dyn Fn(&[f32], Option<&str>, &AtomicBool) -> Result<Transcript, TranscribeError> + Send + Sync;
/// A shared [`TranscribeFn`].
pub type Transcriber = Arc<TranscribeFn>;

/// A transcript and the language it is in.
#[derive(Clone, Debug, PartialEq)]
pub struct Transcript {
    pub text: String,
    pub language: Option<String>,
    /// The model that produced it (logs).
    pub model: String,
}

/// Why there is no transcript.
#[derive(Clone, Debug, PartialEq)]
pub enum TranscribeError {
    /// No local speech-to-text model is installed.
    NoModel,
    /// Every candidate engine failed (the last error).
    Failed(String),
    /// The engine heard nothing it could write down.
    Empty,
    /// The language is unknown and no installed model detects it.
    NeedsLanguage,
    Cancelled,
}

impl TranscribeError {
    pub fn code(&self) -> Option<&'static str> {
        match self {
            TranscribeError::NoModel => Some(ERROR_NO_STT_MODEL),
            TranscribeError::Failed(_) => Some(ERROR_FAILED),
            TranscribeError::Empty => Some(ERROR_EMPTY),
            TranscribeError::NeedsLanguage => Some(ERROR_NEEDS_LANGUAGE),
            TranscribeError::Cancelled => None,
        }
    }
}

/// The engines to try for `language` (normalized or `None` = detect), and the language-ID engine.
fn candidates(language: Option<&str>) -> (Vec<EngineSpec>, Option<EngineSpec>) {
    let mut ctx = router::context();
    // Installed models only: everything else is "unavailable" to the router.
    if let Ok(table) = router::table() {
        for id in table.referenced_ids() {
            if !ctx.installed.contains(&id) {
                ctx.unavailable.insert(id);
            }
        }
    }
    // Two-pass planning puts the utterance (final) model in the plan next to the streaming one
    // where the PC may run both; the final model is tried first (better text and punctuation).
    let request = RouteRequest {
        language: language.unwrap_or(lang::AUTO).to_string(),
        tier: Tier::Auto,
        two_pass: true,
    };
    let plan = router::plan_with(&request, &ctx);
    let mut out: Vec<EngineSpec> = Vec::new();
    for spec in [plan.final_engine.clone(), plan.streaming.clone(), plan.fallback.clone()].into_iter().flatten() {
        if spec.engine != EngineKind::Http && !out.iter().any(|s| s.model == spec.model) {
            out.push(spec);
        }
    }
    let fallback = whisper_fallback(language, |id| {
        if out.iter().any(|s| s.model == id) {
            return false;
        }
        let needs = engines::requirements(&EngineSpec::new(EngineKind::Whisper, id));
        !needs.is_empty() && needs.iter().all(|(_, installed)| *installed)
    });
    if let Some(id) = fallback {
        out.push(EngineSpec::new(EngineKind::Whisper, id));
    }
    out.retain(|s| engines::requirements(s).iter().all(|(_, installed)| *installed));
    (out, plan.language_id)
}

/// Whether a Whisper model only knows English (`.en`): it cannot detect a language, and it writes
/// any speech as English words.
fn english_only(id: &str) -> bool {
    id.contains(".en")
}

/// The best [`WHISPER_FALLBACK`] model for `language` that `usable` accepts. English-only models
/// only for a clip known to be English (asked for, or identified): with the language unknown they
/// would write a Japanese clip as English words.
fn whisper_fallback(language: Option<&str>, usable: impl Fn(&str) -> bool) -> Option<&'static str> {
    let english = language == Some("en");
    WHISPER_FALLBACK
        .iter()
        .copied()
        .find(|id| (english || !english_only(id)) && usable(id))
}

/// The routing table's language-ID engine, when it is installed (it runs on its own here: the
/// router plans it only next to a final engine for "detect", and there is none when the installed
/// models must be told the language).
fn installed_language_id() -> Option<EngineSpec> {
    let table = router::table().ok()?;
    let id = table.language_id.as_deref()?;
    let spec = EngineSpec::new(table.models.get(id)?.engine, id);
    let needs = engines::requirements(&spec);
    (!needs.is_empty() && needs.iter().all(|(_, installed)| *installed)).then_some(spec)
}

/// The spoken language of `audio16k` by the language-ID engine `lid` (normalized).
fn identify_language(lid: &EngineSpec, audio16k: &[f32], ctx: &EngineContext) -> Option<String> {
    engines::create(lid, ctx)
        .ok()
        .and_then(|mut l| l.transcribe(audio16k, &DecodeOptions::default()).ok())
        .and_then(|r| r.lang)
        .and_then(|l| lang::normalize(&l))
}

/// A clip of unknown language that no installed model transcribes without being told the
/// language (only English-only Whisper models, say): the engines are planned for the language
/// that language ID `identified` (stored with the transcript, as it was detected). Without one,
/// the user is asked for the language when some installed model knows English, else there is no
/// model.
fn unknown_language(identified: Option<String>, english_installed: bool) -> Result<String, TranscribeError> {
    match identified {
        Some(code) => Ok(code),
        None if english_installed => Err(TranscribeError::NeedsLanguage),
        None => Err(TranscribeError::NoModel),
    }
}

/// Decode `audio` with `engine`: the cleaned text and the language the engine reports (normalized;
/// the requested one when it reports none).
fn decode(engine: &mut dyn AsrEngine, audio: &[f32], options: &DecodeOptions) -> Result<(String, Option<String>), String> {
    let caps = engine.info().caps.clone();
    let result = if caps.utterance {
        engine.transcribe(audio, options)
    } else if caps.streaming {
        engine.begin_stream(options).and_then(|_| {
            engine.push_stream(audio)?;
            engine.end_stream()
        })
    } else {
        return Err("the engine neither decodes utterances nor streams".into());
    };
    let r = result.map_err(|e| e.to_string())?;
    let language = r.lang.as_deref().and_then(lang::normalize).or_else(|| options.language.clone());
    let text = crate::services::speech_core::hallucination::clean_transcript(&r.text, language.as_deref(), r.no_speech_prob.unwrap_or(0.0))
        .unwrap_or_default();
    Ok((text, language))
}

/// Transcribe a 16 kHz mono clip with the local speech-to-text (blocking; seconds).
pub fn transcribe(audio16k: &[f32], language: Option<&str>, cancel: &AtomicBool) -> Result<Transcript, TranscribeError> {
    let mut language = language.and_then(lang::normalize);
    let (mut specs, lid) = candidates(language.as_deref());
    let ctx = EngineContext::headless();
    if specs.is_empty() && language.is_none() {
        // Nothing installed detects the language (only English-only Whisper models, or models
        // that must be told it): language ID names it first, and only models for that language
        // run, so an English-only model never writes a Japanese clip as English words.
        let identified = installed_language_id().and_then(|lid| identify_language(&lid, audio16k, &ctx));
        let english_installed = identified.is_none() && !candidates(Some("en")).0.is_empty();
        let code = unknown_language(identified, english_installed)?;
        specs = candidates(Some(&code)).0;
        language = Some(code);
    }
    if specs.is_empty() {
        return Err(TranscribeError::NoModel);
    }
    let mut last_error = String::new();
    for spec in specs {
        if cancel.load(Ordering::Relaxed) {
            return Err(TranscribeError::Cancelled);
        }
        let mut engine = match engines::create(&spec, &ctx) {
            Ok(e) => e,
            Err(e) => {
                tracing::info!(target: LOG, "transcript: {} could not load: {e}", spec.model);
                last_error = e.to_string();
                continue;
            }
        };
        let caps = engine.info().caps.clone();
        let mut code = language.clone();
        if code.is_none() && !caps.detects_language {
            if let Some(lid) = &lid {
                code = identify_language(lid, audio16k, &ctx);
            }
        }
        // An engine that lists its languages gets only one of them (else it detects).
        let code = code.filter(|c| caps.languages.is_empty() || caps.languages.iter().any(|l| l == c));
        let options = DecodeOptions { language: code.clone(), ..DecodeOptions::default() };
        match decode(engine.as_mut(), audio16k, &options) {
            Ok((text, _)) if text.trim().is_empty() => return Err(TranscribeError::Empty),
            Ok((text, detected)) => {
                tracing::info!(target: LOG, "transcript by {}: {} characters", spec.model, text.chars().count());
                // The language asked for wins; else what the engine (or language ID) heard.
                let language = code.or(detected);
                return Ok(Transcript { text: text.trim().to_string(), language, model: spec.model });
            }
            Err(e) => {
                tracing::info!(target: LOG, "transcript: {} failed: {e}", spec.model);
                last_error = e;
            }
        }
    }
    Err(TranscribeError::Failed(last_error))
}

// ---------------------------------------------------------------------------------------------
// Background jobs
// ---------------------------------------------------------------------------------------------

#[derive(Default)]
struct Jobs {
    /// Queued or running jobs per voice (a voice can have several: a save, then a re-record or
    /// "Transcribe again" while the first job still runs). A voice is pending while its count is
    /// above zero.
    pending: BTreeMap<String, usize>,
    /// Transcript edit generation per voice (bumped by every user edit).
    edits: BTreeMap<String, u64>,
}

static JOBS: Mutex<Jobs> = Mutex::new(Jobs { pending: BTreeMap::new(), edits: BTreeMap::new() });
/// One transcription at a time (engines are large).
static RUNNER: Mutex<()> = Mutex::new(());

fn jobs() -> std::sync::MutexGuard<'static, Jobs> {
    JOBS.lock().unwrap_or_else(PoisonError::into_inner)
}

/// Whether voice `id` has a queued or running transcription.
pub fn is_pending(id: &str) -> bool {
    jobs().pending.get(id).is_some_and(|n| *n > 0)
}

/// One queued or running job of a voice: counted while it lives, so the count drops when the job
/// ends however it ends (done, a panic in an engine, a thread that could not start).
struct PendingJob(String);

impl PendingJob {
    fn new(id: &str) -> PendingJob {
        *jobs().pending.entry(id.to_string()).or_insert(0) += 1;
        PendingJob(id.to_string())
    }
}

impl Drop for PendingJob {
    fn drop(&mut self) {
        let mut j = jobs();
        if let Some(n) = j.pending.get_mut(&self.0) {
            *n = n.saturating_sub(1);
            if *n == 0 {
                j.pending.remove(&self.0);
            }
        }
    }
}

/// The user edited voice `id`'s transcript: a running job must not overwrite it.
pub fn note_user_edit(id: &str) {
    *jobs().edits.entry(id.to_string()).or_insert(0) += 1;
}

fn edit_generation(id: &str) -> u64 {
    jobs().edits.get(id).copied().unwrap_or(0)
}

/// What a job needs.
pub struct Job {
    pub app_data: PathBuf,
    pub voice_id: String,
    /// The clip the transcript is for.
    pub sha256: String,
    pub audio16k: Vec<f32>,
    pub language: Option<String>,
    generation: u64,
}

impl Job {
    pub fn new(app_data: &Path, voice_id: &str, sha256: &str, audio16k: Vec<f32>, language: Option<String>) -> Job {
        Job {
            app_data: app_data.to_path_buf(),
            voice_id: voice_id.to_string(),
            sha256: sha256.to_string(),
            audio16k,
            language,
            generation: edit_generation(voice_id),
        }
    }
}

/// Queue `job` on a background thread; `transcriber` is [`transcribe`] in the app. `done` runs
/// after the result was applied (or dropped). Marks the voice pending at once, until its last job
/// ends.
pub fn spawn(job: Job, transcriber: Transcriber, done: Arc<dyn Fn() + Send + Sync>) {
    let pending = PendingJob::new(&job.voice_id);
    let id = job.voice_id.clone();
    let spawned = std::thread::Builder::new().name("sigil-voice-transcript".into()).spawn({
        let done = Arc::clone(&done);
        move || {
            run(job, transcriber.as_ref());
            // No longer pending before `done` announces the change.
            drop(pending);
            done();
        }
    });
    if let Err(e) = spawned {
        // The closure (and its `PendingJob`) was dropped with the thread that never started.
        tracing::warn!(target: LOG, "cannot start a transcript job for voice {id}: {e}");
        done();
    }
}

/// Whether `job` still matters: its voice still has the clip it was made from and nobody edited
/// the transcript since it was queued.
fn current(job: &Job, meta: &super::VoiceMeta) -> bool {
    meta.sha256 == job.sha256 && edit_generation(&job.voice_id) == job.generation
}

/// Run one job (blocking): transcribe, then apply the result if it is still wanted. A job that
/// is already superseded when its turn comes (the voice was re-recorded, edited or deleted while
/// it waited) skips the speech-to-text.
pub fn run(job: Job, transcriber: &TranscribeFn) {
    let _one_at_a_time = RUNNER.lock().unwrap_or_else(PoisonError::into_inner);
    if !super::store::get(&job.app_data, &job.voice_id).is_ok_and(|meta| current(&job, &meta)) {
        tracing::debug!(target: LOG, "transcript of voice {} skipped: superseded", job.voice_id);
        return;
    }
    let cancel = AtomicBool::new(false);
    let outcome = transcriber(&job.audio16k, job.language.as_deref(), &cancel);
    let applied = super::store::update(&job.app_data, &job.voice_id, |meta| {
        if !current(&job, meta) {
            return Err("stale".into());
        }
        match &outcome {
            Ok(t) => {
                meta.transcript = Some(t.text.clone());
                if meta.language.is_none() {
                    meta.language = t.language.clone();
                }
                meta.transcript_error = None;
            }
            Err(e) => {
                if let Some(code) = e.code() {
                    meta.transcript_error = Some(code.to_string());
                }
            }
        }
        Ok(())
    });
    if let Err(e) = applied {
        tracing::debug!(target: LOG, "transcript of voice {} not applied: {e}", job.voice_id);
    }
    if let Err(TranscribeError::Failed(e)) = &outcome {
        tracing::warn!(target: LOG, "transcript of voice {} failed: {e}", job.voice_id);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::services::voice_library::store::tests::{wav, Scratch};
    use crate::services::voice_library::VoiceMeta;

    fn voice(s: &Scratch) -> VoiceMeta {
        crate::services::voice_library::store::create(&s.0, &wav(3.5), VoiceMeta { name: "t".into(), source: "recorded".into(), ..Default::default() })
            .unwrap()
    }

    #[test]
    fn a_job_writes_the_transcript_and_detected_language() {
        let s = Scratch::new("tx-ok");
        let v = voice(&s);
        let job = Job::new(&s.0, &v.id, &v.sha256, vec![0.0; 16], None);
        run(job, &|_: &[f32], _: Option<&str>, _: &AtomicBool| {
            Ok(Transcript { text: "hello there".into(), language: Some("en".into()), model: "m".into() })
        });
        let got = super::super::store::get(&s.0, &v.id).unwrap();
        assert_eq!((got.transcript.as_deref(), got.language.as_deref()), (Some("hello there"), Some("en")));
        assert!(!is_pending(&v.id) && !got.transcribing);
    }

    #[test]
    fn a_user_edit_or_a_new_clip_wins_over_a_running_job() {
        let s = Scratch::new("tx-stale");
        let v = voice(&s);
        let job = Job::new(&s.0, &v.id, &v.sha256, vec![], None);
        let id = v.id.clone();
        run(job, &move |_: &[f32], _: Option<&str>, _: &AtomicBool| {
            // The user types a transcript while the engine works.
            note_user_edit(&id);
            Ok(Transcript { text: "machine".into(), language: None, model: "m".into() })
        });
        assert_eq!(super::super::store::get(&s.0, &v.id).unwrap().transcript, None);
        // Jobs superseded before their turn never load an engine: one for an older clip, one
        // queued before an edit, one for a deleted voice.
        let skipped = |_: &[f32], _: Option<&str>, _: &AtomicBool| -> Result<Transcript, TranscribeError> { panic!("a superseded job ran the speech-to-text") };
        run(Job::new(&s.0, &v.id, "0".repeat(64).as_str(), vec![], None), &skipped);
        let queued = Job::new(&s.0, &v.id, &v.sha256, vec![], None);
        note_user_edit(&v.id);
        run(queued, &skipped);
        assert_eq!(super::super::store::get(&s.0, &v.id).unwrap().transcript, None);
        run(Job::new(&s.0, "gone-voice", &v.sha256, vec![], None), &skipped);
        assert!(!is_pending("gone-voice"));
    }

    /// A voice with several jobs (a save, then a re-record or "Transcribe again" while the first
    /// still runs) stays `transcribing` until the last one ends.
    #[test]
    fn a_voice_stays_pending_until_its_last_job_ends() {
        use std::sync::mpsc::channel;
        let s = Scratch::new("tx-two");
        let v = voice(&s);
        let (started_tx, started) = channel::<()>();
        let (done_tx, done) = channel::<&'static str>();
        let gate_a = Arc::new(Mutex::new(()));
        let gate_b = Arc::new(Mutex::new(()));
        let hold_a = gate_a.lock().unwrap();
        let hold_b = gate_b.lock().unwrap();
        let blocked = |gate: &Arc<Mutex<()>>, started: Option<std::sync::mpsc::Sender<()>>| -> Transcriber {
            let gate = Arc::clone(gate);
            let started = Mutex::new(started);
            Arc::new(move |_: &[f32], _: Option<&str>, _: &AtomicBool| {
                if let Some(tx) = started.lock().unwrap().take() {
                    let _ = tx.send(());
                }
                let _wait = gate.lock().unwrap_or_else(PoisonError::into_inner);
                Ok(Transcript { text: "words".into(), language: None, model: "m".into() })
            })
        };
        let signal = |name: &'static str| -> Arc<dyn Fn() + Send + Sync> {
            let tx = Mutex::new(done_tx.clone());
            Arc::new(move || {
                let _ = tx.lock().unwrap().send(name);
            })
        };
        // Job A runs (and holds the one-at-a-time runner); job B queues behind it.
        spawn(Job::new(&s.0, &v.id, &v.sha256, vec![], None), blocked(&gate_a, Some(started_tx)), signal("a"));
        started.recv_timeout(std::time::Duration::from_secs(20)).expect("job A runs");
        spawn(Job::new(&s.0, &v.id, &v.sha256, vec![], None), blocked(&gate_b, None), signal("b"));
        drop(hold_a);
        assert_eq!(done.recv_timeout(std::time::Duration::from_secs(20)), Ok("a"));
        assert!(is_pending(&v.id), "job B is still queued or running");
        assert!(super::super::store::get(&s.0, &v.id).unwrap().transcribing);
        drop(hold_b);
        assert_eq!(done.recv_timeout(std::time::Duration::from_secs(20)), Ok("b"));
        assert!(!is_pending(&v.id));
        assert!(!super::super::store::get(&s.0, &v.id).unwrap().transcribing);
    }

    /// English-only Whisper models (`.en`) only for a clip known to be English: with "Detect" they
    /// would write a Japanese clip as English words and store it as English.
    #[test]
    fn english_only_models_are_never_guessed() {
        let only = |ids: &'static [&'static str]| move |id: &str| ids.contains(&id);
        assert_eq!(whisper_fallback(None, only(&["whisper/base.en"])), None);
        assert_eq!(whisper_fallback(Some("ja"), only(&["whisper/base.en"])), None);
        assert_eq!(whisper_fallback(Some("en"), only(&["whisper/base.en"])), Some("whisper/base.en"));
        assert_eq!(whisper_fallback(None, only(&["whisper/small.en", "whisper/tiny"])), Some("whisper/tiny"));
        assert_eq!(whisper_fallback(Some("en"), only(&["whisper/small.en", "whisper/tiny"])), Some("whisper/small.en"));
        assert_eq!(whisper_fallback(Some("de"), only(&[])), None);
        // A clip of unknown language that only such models could transcribe: the language that
        // language ID heard is planned for (and stored, as it was detected); without language ID
        // the voice asks for the language instead of a guess.
        assert_eq!(unknown_language(Some("ja".into()), false), Ok("ja".into()));
        assert_eq!(unknown_language(None, true), Err(TranscribeError::NeedsLanguage));
        assert_eq!(unknown_language(None, false), Err(TranscribeError::NoModel));
    }

    #[test]
    fn failures_are_recorded_by_code() {
        let s = Scratch::new("tx-err");
        let v = voice(&s);
        for (err, code) in [
            (TranscribeError::NoModel, Some(ERROR_NO_STT_MODEL)),
            (TranscribeError::Failed("x".into()), Some(ERROR_FAILED)),
            (TranscribeError::Empty, Some(ERROR_EMPTY)),
        ] {
            let job = Job::new(&s.0, &v.id, &v.sha256, vec![], None);
            let e = err.clone();
            run(job, &move |_: &[f32], _: Option<&str>, _: &AtomicBool| Err(e.clone()));
            assert_eq!(super::super::store::get(&s.0, &v.id).unwrap().transcript_error.as_deref(), code);
        }
    }

    #[test]
    fn spawned_jobs_are_pending_until_done() {
        let s = Scratch::new("tx-spawn");
        let v = voice(&s);
        let (tx, rx) = std::sync::mpsc::channel();
        let gate = Arc::new(Mutex::new(()));
        let held = gate.lock().unwrap();
        let g = Arc::clone(&gate);
        spawn(
            Job::new(&s.0, &v.id, &v.sha256, vec![], Some("ja".into())),
            Arc::new(move |_: &[f32], l: Option<&str>, _: &AtomicBool| {
                let _wait = g.lock().unwrap();
                Ok(Transcript { text: "こんにちは".into(), language: l.map(str::to_string), model: "m".into() })
            }),
            Arc::new(move || {
                let _ = tx.send(());
            }),
        );
        assert!(is_pending(&v.id));
        assert!(super::super::store::get(&s.0, &v.id).unwrap().transcribing);
        drop(held);
        rx.recv_timeout(std::time::Duration::from_secs(10)).unwrap();
        let got = super::super::store::get(&s.0, &v.id).unwrap();
        assert!(!got.transcribing);
        assert_eq!(got.transcript.as_deref(), Some("こんにちは"));
    }

    /// The real local STT on the speech fixture ("Hello there, the weather is lovely today.").
    /// Run with an app-data folder holding an installed STT model, e.g. the P2 dev cache:
    /// `SIGIL_APP_DATA_DIR=%LOCALAPPDATA%/sigil-dev-cache/appdata-p2b cargo test real_stt -- --ignored`.
    #[test]
    #[ignore = "needs SIGIL_APP_DATA_DIR with an installed local STT model"]
    fn real_stt_transcribes_the_fixture() {
        let bytes: &[u8] = include_bytes!("../speech_core/testdata/speech_16k.wav");
        let (audio, rate) = crate::services::voice_library::process::decode_wav_bytes(bytes).unwrap();
        assert_eq!(rate, 16_000);
        for language in [None, Some("en")] {
            let t = transcribe(&audio, language, &AtomicBool::new(false)).expect("a transcript");
            eprintln!("{language:?}: {t:?}");
            assert!(t.text.to_lowercase().contains("weather"), "{t:?}");
            assert_eq!(t.language.as_deref(), Some("en"), "the language is detected when not given: {t:?}");
        }
    }

    #[test]
    fn error_codes() {
        assert_eq!(TranscribeError::NoModel.code(), Some(ERROR_NO_STT_MODEL));
        assert_eq!(TranscribeError::NeedsLanguage.code(), Some(ERROR_NEEDS_LANGUAGE));
        assert_eq!(TranscribeError::Cancelled.code(), None);
    }
}
