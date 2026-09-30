//! sherpa-onnx streaming engine (`OnlineRecognizer`: Nemotron 3.5 ASR streaming, and any sherpa
//! streaming transducer / CTC package brought by the user).
//!
//! Owned by P2-B.
//!
//! # Frozen signature (P2-S0)
//! `create(&EngineSpec, &EngineContext) -> Result<Box<dyn AsrEngine>, AsrError>` (called by
//! `engines::create`; blocking).
//!
//! # Behaviour
//! - One stream per utterance: `begin_stream` opens a sherpa stream, `push_stream` feeds 16 kHz
//!   audio and returns the partial when the hypothesis changed (a decode step runs only once a
//!   whole model chunk, e.g. 560 ms, is buffered), `end_stream` pads the tail with one decode
//!   window of silence (sherpa decodes NeMo streams only in whole windows, so less padding loses the
//!   last word), finishes the input and returns the final. NeMo streams also start with 0.3 s of
//!   silence (else the first word is often lost). See `online::StreamPadding`. sherpa's
//!   input-finished state is terminal, so the next utterance always gets a new stream (never a
//!   reset).
//! - Language: Nemotron 3.5 is prompt-conditioned; the language goes to the stream's `language`
//!   option before any audio. Its auto mode is worse on every language, so the router always passes
//!   one ([`EngineCaps::needs_language`]); `options.language` of the spec is the fallback, and
//!   without either (or with one the model's `prompt_dictionary` lacks) the stream runs on auto and
//!   a `sherpa_language_auto` warning is reported once per engine.
//! - Hotwords: only k2 transducers decoded with modified beam search take them (`decoding: "beam"`
//!   option); NeMo streaming transducers decode greedily, so hotwords are ignored there
//!   (`caps.hotwords` false).
//! - Threads: 1 by default (a decode runs every chunk and ONNX Runtime's spinning cannot be turned
//!   off through the C API; ADR 0002). Measured on a Ryzen 7 7800X3D, one thread: Nemotron 560 ms
//!   RTF 0.30-0.35, 160 ms RTF 0.80-0.87 (engine-probe matrix, 2026-09-29).
//! - Cancel: the canceller stops between decode steps; the call returns `Cancelled` and the stream
//!   stays open (the pipeline aborts it).

use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;
use std::time::Instant;

use super::super::engine_api::{
    AsrEngine, AsrError, AsrResult, Canceller, DecodeOptions, EngineCaps, EngineContext, EngineInfo, EngineKind, EngineSpec, Hypothesis, StatusLevel,
    SAMPLE_RATE,
};
use super::super::events::codes;
use super::sherpa_offline::{
    engine_threads, only_language, report, requested_language, resolve, runtime, runtime_label, status_codes, validate_byo, Options,
};
use crate::services::speech_rt::sherpa::catalog::{CatalogModel, Task};
use crate::services::speech_rt::sherpa::detect::SherpaModel;
use crate::services::speech_rt::sherpa::hotwords;
use crate::services::speech_rt::sherpa::languages::PromptDictionary;
use crate::services::speech_rt::sherpa::lib::STREAMING_DEFAULT_THREADS;
use crate::services::speech_rt::sherpa::online::{Decoding, OnlineModel, OnlineRecognizer, OnlineRecognizerConfig, OnlineStream, StreamPadding};
use crate::services::speech_rt::sherpa::onnx_info;

/// Create the engine (blocking: loads the model).
pub fn create(spec: &EngineSpec, ctx: &EngineContext) -> Result<Box<dyn AsrEngine>, AsrError> {
    let opts = Options::parse(&spec.options);
    let resolved = resolve(spec, Task::Online, &opts)?;
    let SherpaModel::Online(mut config) = resolved.config.clone() else {
        return Err(AsrError::unsupported(format!("{} is not a streaming model", resolved.label)));
    };
    let rt = runtime(ctx)?;
    config.num_threads = Some(engine_threads(spec, STREAMING_DEFAULT_THREADS));
    if let Some(score) = opts.hotwords_score {
        config.hotwords_score = score;
    }
    // Encoder facts: the language prompts of a prompt-conditioned NeMo encoder (Nemotron 3.5) and
    // the decode window that sets the tail padding.
    let summary = match &config.model {
        OnlineModel::Transducer { encoder, .. } => onnx_info::summary(encoder).ok(),
        _ => None,
    };
    let prompts = summary.as_ref().and_then(prompts_of);
    let padding = StreamPadding::for_encoder(summary.as_ref());
    // Beam search (for hotwords) only where sherpa supports it: k2 transducers. `validate` refuses
    // it on NeMo streaming transducers (sherpa would exit), so try and fall back to greedy.
    // `decoding: "greedy"` keeps greedy search even with hotwords (as in `sherpa_offline`).
    if !opts.greedy && (opts.beam_search || !opts.hotwords.is_empty()) {
        let mut beam = config.clone();
        beam.decoding = Decoding::ModifiedBeamSearch {
            max_active_paths: opts.max_active_paths.unwrap_or(4),
        };
        match beam.validate() {
            Ok(()) => config = beam,
            Err(e) => tracing::info!(target: "sigil::sherpa", "{}: greedy search only ({e})", resolved.label),
        }
    }
    if resolved.byo {
        validate_byo(&SherpaModel::Online(config.clone()), &resolved.label, ctx)?;
    }
    report(
        ctx,
        codes::LOADING_MODEL,
        format!("Loading {}", resolved.label),
        resolved.catalog.map(|c| c.id),
        StatusLevel::Info,
    );
    let recognizer = OnlineRecognizer::new(&rt, &config).map_err(|e| AsrError::runtime(format!("{}: {e}", resolved.label)))?;
    let languages: Vec<String> = match (resolved.catalog, &prompts) {
        (Some(c), _) => c.languages.iter().map(|l| l.to_string()).collect(),
        (None, Some(p)) => p.base_codes(),
        (None, None) => Vec::new(),
    };
    let info = EngineInfo {
        kind: EngineKind::SherpaOnline,
        model: resolved.label.clone(),
        runtime: runtime_label(),
        device: "cpu".into(),
        caps: EngineCaps {
            streaming: true,
            utterance: true,
            languages,
            needs_language: prompts.is_some() || resolved.catalog.is_some_and(|c| c.family.needs_language()),
            detects_language: false,
            hotwords: config.supports_hotwords(),
            prompt: false,
            tags: false,
        },
        vram_mb: 0,
        fallback_reason: None,
    };
    Ok(Box::new(SherpaOnlineEngine {
        info,
        model_id: resolved.catalog.map(|c| c.id),
        catalog: resolved.catalog,
        recognizer,
        prompts,
        default_language: opts.language.clone(),
        default_hotwords: opts.hotwords.clone(),
        tail_padding: SAMPLE_RATE as usize * opts.tail_padding_ms.unwrap_or(padding.tail_ms) as usize / 1000,
        head_padding: SAMPLE_RATE as usize * padding.head_ms as usize / 1000,
        stream: None,
        cancel: Arc::new(AtomicBool::new(false)),
        warned_auto: false,
        ctx: ctx.clone(),
    }))
}

/// The encoder's language prompts, when it is prompt-conditioned (`prompt_index` input).
fn prompts_of(summary: &onnx_info::OnnxSummary) -> Option<PromptDictionary> {
    if !summary.has_input("prompt_index") {
        return None;
    }
    let dict = summary.meta("prompt_dictionary")?;
    PromptDictionary::parse(dict, summary.meta("auto_prompt_id"))
}

#[cfg(test)]
fn read_prompts(encoder: &std::path::Path) -> Option<PromptDictionary> {
    prompts_of(&onnx_info::summary(encoder).ok()?)
}

/// The open stream of an utterance.
struct ActiveStream {
    stream: OnlineStream,
    /// Last hypothesis returned as a partial.
    last_text: String,
    /// Language the stream was told (`None`: auto / single-language model default).
    lang: Option<String>,
}

/// The sherpa-onnx `OnlineRecognizer` engine.
pub struct SherpaOnlineEngine {
    info: EngineInfo,
    model_id: Option<&'static str>,
    catalog: Option<&'static CatalogModel>,
    recognizer: OnlineRecognizer,
    prompts: Option<PromptDictionary>,
    default_language: Option<String>,
    default_hotwords: Vec<String>,
    /// Samples of silence appended by `end_stream`.
    tail_padding: usize,
    /// Samples of silence fed first by `begin_stream` (NeMo cache-aware models).
    head_padding: usize,
    stream: Option<ActiveStream>,
    cancel: Arc<AtomicBool>,
    /// The auto-language warning was reported.
    warned_auto: bool,
    ctx: EngineContext,
}

impl SherpaOnlineEngine {
    fn config(&self) -> &OnlineRecognizerConfig {
        self.recognizer.config()
    }

    /// The language to set on a new stream (`None`: leave the model on auto / its only language).
    fn stream_language(&mut self, options: &DecodeOptions) -> Option<String> {
        let wanted = requested_language(options, &self.default_language);
        let Some(prompts) = &self.prompts else {
            // Not prompt-conditioned: the model has one language (or detects it implicitly).
            return None;
        };
        match wanted {
            Some(l) if prompts.supports(&l) => Some(l),
            other => {
                if !self.warned_auto {
                    self.warned_auto = true;
                    let detail = match &other {
                        Some(l) => format!(
                            "{} has no {l} prompt; it runs on automatic language detection, which is less accurate",
                            self.info.model
                        ),
                        None => format!(
                            "{} was not told the spoken language; it runs on automatic language detection, which is less accurate",
                            self.info.model
                        ),
                    };
                    tracing::warn!(target: "sigil::sherpa", "{detail}");
                    report(&self.ctx, status_codes::LANGUAGE_AUTO, detail, self.model_id, StatusLevel::Warning);
                }
                None
            }
        }
    }

    fn open(&mut self, options: &DecodeOptions) -> Result<(), AsrError> {
        self.stream = None;
        let lang = self.stream_language(options);
        let phrases = if options.hotwords.is_empty() {
            &self.default_hotwords
        } else {
            &options.hotwords
        };
        let words = self
            .config()
            .supports_hotwords()
            .then(|| hotwords::transducer_hotwords(phrases))
            .flatten();
        let mut stream = match words {
            Some(w) => self.recognizer.create_stream_with_hotwords(&w),
            None => self.recognizer.create_stream(),
        }
        .map_err(|e| AsrError::runtime(format!("{}: {e}", self.info.model)))?;
        if let Some(l) = &lang {
            stream
                .set_option("language", l)
                .map_err(|e| AsrError::runtime(format!("{}: {e}", self.info.model)))?;
        }
        if self.head_padding > 0 {
            stream
                .accept_waveform(SAMPLE_RATE, &vec![0.0f32; self.head_padding])
                .map_err(|e| AsrError::runtime(format!("{}: {e}", self.info.model)))?;
        }
        self.stream = Some(ActiveStream {
            stream,
            last_text: String::new(),
            lang: lang.or_else(|| only_language(self.catalog)),
        });
        Ok(())
    }

    fn cancelled(&self) -> bool {
        self.cancel.load(Ordering::SeqCst)
    }

    /// Feed audio and decode what is ready; `Some(text)` when the hypothesis changed.
    fn feed(&mut self, audio: &[f32]) -> Result<Option<Hypothesis>, AsrError> {
        if self.cancelled() {
            return Err(AsrError::cancelled());
        }
        let model = self.info.model.clone();
        let cancel = self.cancel.clone();
        let active = self
            .stream
            .as_mut()
            .ok_or_else(|| AsrError::invalid_input(format!("{model}: no stream is open (begin_stream first)")))?;
        active
            .stream
            .accept_waveform(SAMPLE_RATE, audio)
            .map_err(|e| AsrError::invalid_input(format!("{model}: {e}")))?;
        let Some(steps) = active.stream.decode_cancellable(Some(&cancel)) else {
            return Err(AsrError::cancelled());
        };
        if steps == 0 {
            return Ok(None);
        }
        let text = active
            .stream
            .result()
            .map_err(|e| AsrError::runtime(format!("{model}: {e}")))?
            .text;
        if text.is_empty() || text == active.last_text {
            return Ok(None);
        }
        active.last_text.clone_from(&text);
        Ok(Some(Hypothesis {
            text,
            lang: active.lang.clone(),
        }))
    }

    /// Finish the open stream and return its final (the stream is consumed either way).
    fn finish(&mut self) -> Result<AsrResult, AsrError> {
        let model = self.info.model.clone();
        let mut active = self
            .stream
            .take()
            .ok_or_else(|| AsrError::invalid_input(format!("{model}: no stream is open (begin_stream first)")))?;
        if self.cancelled() {
            return Err(AsrError::cancelled());
        }
        let t0 = Instant::now();
        if self.tail_padding > 0 {
            let pad = vec![0.0f32; self.tail_padding];
            active
                .stream
                .accept_waveform(SAMPLE_RATE, &pad)
                .map_err(|e| AsrError::runtime(format!("{model}: {e}")))?;
        }
        active.stream.input_finished();
        if active
            .stream
            .decode_cancellable(Some(&self.cancel))
            .is_none()
        {
            return Err(AsrError::cancelled());
        }
        let r = active
            .stream
            .result()
            .map_err(|e| AsrError::runtime(format!("{model}: {e}")))?;
        let flush = t0.elapsed();
        Ok(AsrResult {
            text: r.text,
            lang: active.lang,
            tags: Vec::new(),
            no_speech_prob: None,
            confidence: None,
            // The final's own latency: the tail flush (the stream's earlier decodes produced partials).
            decode_ms: flush.as_millis().min(u128::from(u32::MAX)) as u32,
        })
    }
}

impl AsrEngine for SherpaOnlineEngine {
    fn info(&self) -> &EngineInfo {
        &self.info
    }

    /// Decode a closed utterance by streaming it through a fresh stream in one go.
    fn transcribe(&mut self, audio: &[f32], options: &DecodeOptions) -> Result<AsrResult, AsrError> {
        let t0 = Instant::now();
        self.begin_stream(options)?;
        let fed = self.feed(audio);
        let out = match fed {
            Ok(_) => self.finish(),
            Err(e) => {
                self.stream = None;
                Err(e)
            }
        }?;
        Ok(AsrResult {
            decode_ms: t0.elapsed().as_millis().min(u128::from(u32::MAX)) as u32,
            ..out
        })
    }

    fn begin_stream(&mut self, options: &DecodeOptions) -> Result<(), AsrError> {
        self.cancel.store(false, Ordering::SeqCst);
        self.open(options)
    }

    fn push_stream(&mut self, audio: &[f32]) -> Result<Option<Hypothesis>, AsrError> {
        if audio.is_empty() {
            return Ok(None);
        }
        self.feed(audio)
    }

    fn end_stream(&mut self) -> Result<AsrResult, AsrError> {
        self.finish()
    }

    fn abort_stream(&mut self) {
        self.stream = None;
    }

    fn canceller(&self) -> Option<Canceller> {
        Some(Canceller::from_flag(self.cancel.clone()))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::services::speech_rt::sherpa::catalog::ids;

    #[test]
    fn prompts_are_read_from_encoder_metadata() {
        use crate::services::neural_audio::test_onnx::{Dim, Graph, FLOAT, INT64};
        let dir = std::env::temp_dir().join(format!("sigil-sherpa-online-test-{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        let mut g = Graph::new();
        g.input("audio_signal", FLOAT, &[Dim::Named("N"), Dim::Fixed(128), Dim::Named("T")])
            .input("prompt_index", INT64, &[Dim::Named("N")])
            .output("outputs", FLOAT, &[Dim::Named("N")]);
        let mut bytes = g.model_bytes(17);
        // metadata_props (field 14): {"prompt_dictionary": ..., "auto_prompt_id": "101"}
        for (k, v) in [
            ("prompt_dictionary", r#"{"auto": 101, "en-US": 0, "ja-JP": 12}"#),
            ("auto_prompt_id", "101"),
        ] {
            let mut entry = vec![0x0a, k.len() as u8];
            entry.extend_from_slice(k.as_bytes());
            entry.extend_from_slice(&[0x12, v.len() as u8]);
            entry.extend_from_slice(v.as_bytes());
            bytes.push(0x72);
            bytes.push(entry.len() as u8);
            bytes.extend_from_slice(&entry);
        }
        let enc = dir.join("encoder.onnx");
        std::fs::write(&enc, &bytes).unwrap();
        let p = read_prompts(&enc).expect("a prompt-conditioned encoder");
        assert!(p.supports("ja") && p.supports("en") && !p.supports("ko"));
        assert_eq!(p.base_codes(), vec!["en", "ja"]);

        // No `prompt_index` input: not prompt-conditioned, whatever the metadata says.
        let mut g = Graph::new();
        g.input("audio_signal", FLOAT, &[Dim::Named("N")])
            .output("outputs", FLOAT, &[Dim::Named("N")]);
        let plain = dir.join("plain.onnx");
        std::fs::write(&plain, g.model_bytes(17)).unwrap();
        assert!(read_prompts(&plain).is_none());
        assert!(read_prompts(&dir.join("missing.onnx")).is_none());
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn utterance_models_are_refused_and_missing_models_name_their_id() {
        let spec = EngineSpec::new(EngineKind::SherpaOnline, ids::SENSE_VOICE);
        let e = create(&spec, &EngineContext::headless()).err().unwrap();
        // Installed on a developer PC: refused as an utterance model; otherwise not installed.
        assert!(
            e.message.contains("sherpa-offline") || e.kind == crate::services::speech::engine_api::AsrErrorKind::NotInstalled,
            "{e}"
        );
        for id in [ids::NEMOTRON_560, ids::NEMOTRON_160] {
            if crate::services::model_store::installed_path(id).is_some() {
                continue;
            }
            let e = create(&EngineSpec::new(EngineKind::SherpaOnline, id), &EngineContext::headless())
                .err()
                .unwrap();
            assert_eq!(e.kind, crate::services::speech::engine_api::AsrErrorKind::NotInstalled, "{e}");
            assert!(e.message.contains(id));
        }
    }

    // -----------------------------------------------------------------------------------------
    // Real models (ignored; see `speech_rt::sherpa::testkit` for the environment)
    // -----------------------------------------------------------------------------------------

    use std::sync::Mutex;
    use std::time::Duration;

    use crate::services::speech::engine_api::{AsrErrorKind, EngineStatus, StatusSink};
    use crate::services::speech_rt::sherpa::testkit::{clip_16k, dll_lock, installed, runtime_installed, squash};

    /// Pipeline capture block (the hub's 16 kHz consumer reads about 32 ms at a time).
    const BLOCK: usize = 512;

    fn real_engine(id: &str, options: serde_json::Value, statuses: Option<Arc<Mutex<Vec<EngineStatus>>>>) -> Option<Box<dyn AsrEngine>> {
        if !runtime_installed() || installed(id).is_none() {
            return None;
        }
        let mut spec = EngineSpec::new(EngineKind::SherpaOnline, id);
        spec.options = options;
        let ctx = match statuses {
            Some(s) => EngineContext {
                status: StatusSink::new(move |st| s.lock().unwrap().push(st)),
            },
            None => EngineContext::headless(),
        };
        Some(super::super::create(&spec, &ctx).unwrap_or_else(|e| panic!("{id}: {e}")))
    }

    fn with_lang(lang: &str) -> DecodeOptions {
        DecodeOptions {
            language: Some(lang.into()),
            ..DecodeOptions::default()
        }
    }

    /// Stream `audio` the way the pipeline does (32 ms blocks) and return the partials and final.
    fn stream(e: &mut dyn AsrEngine, audio: &[f32], options: &DecodeOptions) -> (Vec<Hypothesis>, AsrResult, Duration) {
        e.begin_stream(options).unwrap();
        let mut partials = Vec::new();
        let t0 = Instant::now();
        for block in audio.chunks(BLOCK) {
            if let Some(h) = e.push_stream(block).unwrap() {
                partials.push(h);
            }
        }
        let fin = e.end_stream().unwrap();
        (partials, fin, t0.elapsed())
    }

    #[test]
    #[ignore = "needs SIGIL_APP_DATA_DIR with runtime/sherpa-onnx + stt/nemotron-3.5-streaming-*-int8, and the test clips"]
    fn real_nemotron_streams_partials_and_finals_in_the_given_language() {
        let _g = dll_lock();
        for id in [ids::NEMOTRON_560, ids::NEMOTRON_160] {
            let Some(mut e) = real_engine(id, serde_json::Value::Null, None) else {
                continue;
            };
            let caps = e.info().caps.clone();
            assert!(caps.streaming && caps.utterance && caps.needs_language && !caps.hotwords && !caps.detects_language);
            assert!(caps.languages.contains(&"ja".to_string()) && caps.languages.contains(&"en".to_string()));
            for (clip, lang, needle) in [
                ("ja-nemotron.wav", "ja", "国のために何ができるか"),
                ("en.wav", "en", "presentedhimwithfiftypiecesofgold"),
                ("de.wav", "de", "wursthatzwei"),
                // Neither the last word (tail padding) nor the first (head padding) is lost.
                ("es.wav", "es", "nopreguntesquépuedehacer"),
                ("es.wav", "es", "portupaís"),
                ("uk.wav", "uk", "своєїкраїни"),
            ] {
                let Some(audio) = clip_16k(clip) else { return };
                let (partials, fin, wall) = stream(e.as_mut(), &audio, &with_lang(lang));
                let rtf = wall.as_secs_f64() / (audio.len() as f64 / f64::from(SAMPLE_RATE));
                eprintln!(
                    "{id} {clip}: {} partials, final {:?} ({} ms flush), RTF {rtf:.3}",
                    partials.len(),
                    fin.text,
                    fin.decode_ms
                );
                assert!(!partials.is_empty(), "{id} {clip}: partials while speaking");
                assert!(partials
                    .iter()
                    .all(|p| p.lang.as_deref() == Some(lang) && !p.text.is_empty()));
                assert!(partials.windows(2).all(|w| w[0].text != w[1].text), "a partial is sent only on change");
                // Transcript checks on the default 560 ms model; the 160 ms model trades accuracy
                // for latency (measured: "Una pregunta es ..." for "No preguntes ...").
                if id == ids::NEMOTRON_560 {
                    assert!(squash(&fin.text).contains(&squash(needle)), "{id} {clip}: {}", fin.text);
                } else {
                    assert!(!fin.text.is_empty(), "{id} {clip}");
                }
                assert_eq!(fin.lang.as_deref(), Some(lang));
                assert!(fin.tags.is_empty());
                // Pushing the whole clip at once gives the same final as 32 ms blocks: chunking
                // is the model's, not the caller's.
                let whole = e.transcribe(&audio, &with_lang(lang)).unwrap();
                assert_eq!(whole.text, fin.text, "{id} {clip}");
            }
        }
    }

    #[test]
    #[ignore = "needs SIGIL_APP_DATA_DIR with runtime/sherpa-onnx + stt/nemotron-3.5-streaming-560ms-int8, and the test clips"]
    fn real_nemotron_without_a_language_warns_once_and_runs_on_auto() {
        let _g = dll_lock();
        let statuses = Arc::new(Mutex::new(Vec::new()));
        let Some(mut e) = real_engine(ids::NEMOTRON_560, serde_json::Value::Null, Some(statuses.clone())) else {
            return;
        };
        let Some(audio) = clip_16k("en.wav") else { return };
        for _ in 0..2 {
            let (_, fin, _) = stream(e.as_mut(), &audio, &DecodeOptions::default());
            assert!(squash(&fin.text).contains("presentedhim"), "{}", fin.text);
            assert_eq!(fin.lang, None);
        }
        let _ = stream(e.as_mut(), &audio, &with_lang("xx"));
        let warnings = statuses
            .lock()
            .unwrap()
            .iter()
            .filter(|s| s.code == super::super::sherpa_offline::status_codes::LANGUAGE_AUTO)
            .count();
        assert_eq!(warnings, 1, "reported once per engine");
        // The spec's default language applies when a call brings none.
        let Some(mut en) = real_engine(ids::NEMOTRON_560, serde_json::json!({ "language": "en-US" }), None) else {
            return;
        };
        let (_, fin, _) = stream(en.as_mut(), &audio, &DecodeOptions::default());
        assert_eq!(fin.lang.as_deref(), Some("en"));
    }

    #[test]
    #[ignore = "needs SIGIL_APP_DATA_DIR with runtime/sherpa-onnx + stt/nemotron-3.5-streaming-560ms-int8, and the test clips"]
    fn real_nemotron_stream_rules_cancel_and_abort() {
        let _g = dll_lock();
        let Some(mut e) = real_engine(ids::NEMOTRON_560, serde_json::Value::Null, None) else {
            return;
        };
        let Some(audio) = clip_16k("en.wav") else { return };
        // No stream open.
        assert_eq!(e.push_stream(&audio[..BLOCK]).unwrap_err().kind, AsrErrorKind::InvalidInput);
        assert_eq!(e.end_stream().unwrap_err().kind, AsrErrorKind::InvalidInput);
        // Cancel mid-stream: the running and later pushes return Cancelled until the next stream.
        e.begin_stream(&with_lang("en")).unwrap();
        e.push_stream(&audio[..audio.len() / 2]).unwrap();
        e.canceller().unwrap().cancel();
        assert!(e
            .push_stream(&audio[audio.len() / 2..])
            .unwrap_err()
            .is_cancelled());
        assert!(e.end_stream().unwrap_err().is_cancelled());
        // A new stream after a cancel works, and a second begin replaces an open stream.
        e.begin_stream(&with_lang("en")).unwrap();
        e.push_stream(&audio[..BLOCK * 20]).unwrap();
        let (_, fin, _) = stream(e.as_mut(), &audio, &with_lang("en"));
        assert!(squash(&fin.text).contains("tribalchief"), "{}", fin.text);
        // Abort drops the stream without a result.
        e.begin_stream(&with_lang("en")).unwrap();
        e.push_stream(&audio[..BLOCK * 10]).unwrap();
        e.abort_stream();
        assert_eq!(e.end_stream().unwrap_err().kind, AsrErrorKind::InvalidInput);
    }

    // -----------------------------------------------------------------------------------------
    // Through the real pipeline (ignored): capture source → Silero segmenter → engine workers →
    // events, with the sherpa engines from `engines::create`.
    // -----------------------------------------------------------------------------------------

    mod through_the_pipeline {
        use std::collections::VecDeque;
        use std::sync::atomic::{AtomicU64, Ordering as AtomicOrdering};
        use std::sync::Mutex;
        use std::time::Duration;

        use crate::services::speech::engine_api::{EngineKind, EngineSpec, PipelineConfig, SAMPLE_RATE};
        use crate::services::speech::events::{SpeechFinal, SpeechPartial, SpeechStatus, SpeechTurnEnd, SpeechUtterance};
        use crate::services::speech::pipeline::{start_with, stop_pipeline_key, AudioSource, EventSink, SessionDeps, SourceRead};
        use crate::services::speech_rt::sherpa::catalog::ids;
        use crate::services::speech_rt::sherpa::testkit::{clip_16k, dll_lock, installed, runtime_installed, squash};

        /// Plays 16 kHz audio in real time (a capture device would), then silence.
        struct WavSource {
            audio: VecDeque<f32>,
            started: std::time::Instant,
            delivered: usize,
        }

        impl AudioSource for WavSource {
            fn read(&mut self, out: &mut Vec<f32>) -> SourceRead {
                let due = (self.started.elapsed().as_secs_f64() * f64::from(SAMPLE_RATE)) as usize;
                while self.delivered < due {
                    out.push(self.audio.pop_front().unwrap_or(0.0));
                    self.delivered += 1;
                }
                SourceRead::default()
            }
            fn wait(&self, timeout: Duration) {
                std::thread::park_timeout(timeout.min(Duration::from_millis(10)));
            }
            fn muted(&self) -> bool {
                false
            }
            fn ended(&self) -> Option<String> {
                None
            }
            fn is_loopback(&self) -> bool {
                false
            }
            fn key(&self) -> String {
                "test-wav".into()
            }
            fn dropped_frames(&self) -> u64 {
                0
            }
            fn close(self: Box<Self>) {}
        }

        #[derive(Default)]
        struct Collected {
            partials: Vec<SpeechPartial>,
            finals: Vec<SpeechFinal>,
            statuses: Vec<SpeechStatus>,
        }

        struct Sink(std::sync::Arc<Mutex<Collected>>);

        impl EventSink for Sink {
            fn partial(&self, event: &SpeechPartial) {
                self.0.lock().unwrap().partials.push(event.clone());
            }
            fn final_transcript(&self, event: &SpeechFinal) {
                self.0.lock().unwrap().finals.push(event.clone());
            }
            fn status(&self, event: &SpeechStatus) {
                self.0.lock().unwrap().statuses.push(event.clone());
            }
            fn utterance(&self, _event: &SpeechUtterance, _audio_ms: u64) {}
            fn turn_end(&self, _event: &SpeechTurnEnd) {}
        }

        /// Run one pipeline session on `clip` (with 0.5 s of silence before and 1.5 s after) and
        /// return what it emitted once a final arrived (or after a timeout).
        async fn run(pipeline_id: &str, clip: &str, config: PipelineConfig) -> Option<Collected> {
            let audio = clip_16k(clip)?;
            let lead = vec![0.0f32; SAMPLE_RATE as usize / 2];
            let tail = vec![0.0f32; SAMPLE_RATE as usize * 3 / 2];
            let samples: VecDeque<f32> = lead.into_iter().chain(audio).chain(tail).collect();
            let total_s = samples.len() as f64 / f64::from(SAMPLE_RATE);
            let collected = std::sync::Arc::new(Mutex::new(Collected::default()));
            let mut deps = SessionDeps::production(std::sync::Arc::new(Sink(collected.clone())));
            deps.source = Box::new(move |_source: &str| {
                Ok(Box::new(WavSource {
                    audio: samples,
                    started: std::time::Instant::now(),
                    delivered: 0,
                }) as Box<dyn AudioSource>)
            });
            let config = PipelineConfig {
                pipeline_id: pipeline_id.into(),
                ..config
            };
            let started = start_with(config, deps)
                .await
                .unwrap_or_else(|e| panic!("{pipeline_id}: {e}"));
            assert_eq!(started.pipeline_id, pipeline_id);
            // Real time plus model loading and decoding.
            let deadline = std::time::Instant::now() + Duration::from_secs_f64(total_s + 30.0);
            loop {
                tokio::time::sleep(Duration::from_millis(100)).await;
                if !collected.lock().unwrap().finals.is_empty() || std::time::Instant::now() > deadline {
                    break;
                }
            }
            assert!(stop_pipeline_key(pipeline_id).await, "{pipeline_id} was running");
            let out = std::mem::take(&mut *collected.lock().unwrap());
            for s in &out.statuses {
                eprintln!("  {pipeline_id} status {:?} {} {}", s.state, s.code, s.detail);
            }
            Some(out)
        }

        fn spec(kind: EngineKind, id: &str) -> EngineSpec {
            EngineSpec::new(kind, id)
        }

        /// Every sherpa engine decodes a real utterance through the pipeline; SenseVoice's tags and
        /// the languages ride on the finals; Nemotron streams partials under the final's segment.
        #[test]
        #[ignore = "needs SIGIL_APP_DATA_DIR with runtime/sherpa-onnx and the sherpa STT models, and the test clips"]
        fn real_sherpa_engines_decode_through_the_pipeline() {
            // The DLL lock serializes native loads with the other sherpa tests; it is held outside
            // the async runtime.
            let _g = dll_lock();
            if !runtime_installed() {
                return;
            }
            let rt = tokio::runtime::Builder::new_multi_thread()
                .worker_threads(2)
                .enable_all()
                .build()
                .expect("tokio runtime");
            rt.block_on(decode_every_engine());
        }

        async fn decode_every_engine() {
            static N: AtomicU64 = AtomicU64::new(0);
            let key = |name: &str| format!("p2b-{name}-{}", N.fetch_add(1, AtomicOrdering::Relaxed));

            // Streaming: Nemotron 3.5 (560 ms) partials and its stream final, in English.
            if installed(ids::NEMOTRON_560).is_some() {
                let mut s = spec(EngineKind::SherpaOnline, ids::NEMOTRON_560);
                s.options = serde_json::json!({ "language": "en" });
                let config = PipelineConfig {
                    language: "en".into(),
                    streaming: Some(s),
                    ..PipelineConfig::default()
                };
                if let Some(out) = run(&key("nemotron"), "en.wav", config).await {
                    let f = out.finals.first().expect("a Nemotron final");
                    eprintln!(
                        "nemotron: {} partials, final {:?} rev {} lang {:?} {} ms after VAD end",
                        out.partials.len(),
                        f.text,
                        f.revision,
                        f.lang,
                        f.latency_ms
                    );
                    assert!(squash(&f.text).contains("presentedhim"), "{}", f.text);
                    assert_eq!(f.engine, "sherpa-online");
                    assert_eq!(f.lang.as_deref(), Some("en"));
                    let same: Vec<_> = out
                        .partials
                        .iter()
                        .filter(|p| p.segment_id == f.segment_id)
                        .collect();
                    assert!(!same.is_empty(), "partials of the final's segment");
                    assert!(same.iter().all(|p| p.revision < f.revision), "the final has the newest revision");
                }
            }

            // Utterance finals: SenseVoice (tags), Qwen3-ASR (ja), Parakeet Ultra (de), Moonshine (en).
            for (id, clip, lang, needle, tags) in [
                (ids::SENSE_VOICE, "zh.wav", "zh", "", true),
                (ids::QWEN3_ASR_0_6B, "ja-qwen.wav", "ja", "運動神経", false),
                (ids::PARAKEET_ULTRA, "de.wav", "de", "wursthatzwei", false),
                (ids::MOONSHINE_BASE_EN, "en.wav", "en", "tribalchieftain", false),
            ] {
                if installed(id).is_none() {
                    continue;
                }
                let config = PipelineConfig {
                    language: lang.into(),
                    final_engine: Some(spec(EngineKind::SherpaOffline, id)),
                    ..PipelineConfig::default()
                };
                let Some(out) = run(&key("final"), clip, config).await else { continue };
                let f = out
                    .finals
                    .first()
                    .unwrap_or_else(|| panic!("{id}: a final"));
                eprintln!("{id}: {:?} lang {:?} tags {:?} {} ms after VAD end", f.text, f.lang, f.tags, f.latency_ms);
                assert!(squash(&f.text).contains(needle), "{id}: {}", f.text);
                assert_eq!(f.engine, "sherpa-offline");
                assert_eq!(f.model, id);
                assert_eq!(f.lang.as_deref(), Some(lang), "{id}");
                if tags {
                    assert!(
                        f.tags.iter().any(|t| t.starts_with("emotion:")),
                        "{id}: tags ride on finals: {:?}",
                        f.tags
                    );
                    assert!(f.tags.iter().any(|t| t == "lang:zh"), "{:?}", f.tags);
                }
            }
        }
    }
}
