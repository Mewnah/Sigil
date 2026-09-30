//! Sigil's sherpa-onnx STT models (`model-catalog/stt-sherpa.json`): what each catalog id is, which
//! files of its installed folder make up the sherpa config, and what the model can do.
//!
//! The catalog fragment says where the files come from (pinned archives and Hub files); this table
//! says how to run them. Every id here has an entry in `stt-sherpa.json` and every entry there is
//! here (unit-tested against the embedded catalog), and each [`Layout`] file is either a catalog
//! file or covered by its archive's `include` list.
//!
//! The router (P2-C) and the Models UI read [`all`] for the language sets and capabilities; the
//! engines (`speech::engines::sherpa_*`) build configs with [`CatalogModel::config`] from
//! `model_store::installed_path(id)`, and never download.

use std::path::{Path, PathBuf};

use super::detect::SherpaModel;
use super::languages::{ENGLISH, NEMOTRON_LANGUAGES, NEMOTRON_TRANSCRIPTION_READY, PARAKEET_LANGUAGES, QWEN3_LANGUAGES, SENSE_VOICE_LANGUAGES};
use super::lid::LanguageIdConfig;
use super::offline::{OfflineModel, OfflineRecognizerConfig};
use super::online::{OnlineModel, OnlineRecognizerConfig};

/// Catalog ids of the sherpa STT models.
pub mod ids {
    /// Nemotron 3.5 ASR streaming 0.6B, 560 ms chunks, int8: the default CPU streaming model
    /// (partials and stream finals; measured RTF 0.30-0.35 on one thread of a 7800X3D vs 0.80-0.87 for
    /// 160 ms chunks, and fewer word errors).
    pub const NEMOTRON_560: &str = "stt/nemotron-3.5-streaming-560ms-int8";
    /// Nemotron 3.5 ASR streaming 0.6B, 160 ms chunks, int8: lowest partial latency, ~2.7× the CPU.
    pub const NEMOTRON_160: &str = "stt/nemotron-3.5-streaming-160ms-int8";
    /// Parakeet Ultra (Moondream's retrain of Parakeet TDT 0.6B v3), int8: European finals.
    pub const PARAKEET_ULTRA: &str = "stt/parakeet-ultra-int8";
    /// NVIDIA Parakeet TDT 0.6B v3, int8 (official sherpa package): European finals fallback.
    pub const PARAKEET_V3: &str = "stt/parakeet-tdt-0.6b-v3-int8";
    /// Qwen3-ASR 0.6B, int8: ja / ko / ru / zh and 26 more languages. It transcribes any of them
    /// without being told the language, but does not report which one it heard (see
    /// [`Family::detects_language`](super::Family::detects_language)).
    pub const QWEN3_ASR_0_6B: &str = crate::services::resources::policy::ids::STT_QWEN3_ASR_0_6B;
    /// SenseVoiceSmall (2024-07-17 package), int8: zh / yue / en / ja / ko with emotion and event tags.
    pub const SENSE_VOICE: &str = "stt/sense-voice-small-int8";
    /// Moonshine base (English), quantized: a small, fast English utterance model.
    pub const MOONSHINE_BASE_EN: &str = "stt/moonshine-base-en-int8";
    /// Whisper base (multilingual) encoder / decoder for spoken language identification.
    pub const LID_WHISPER_BASE: &str = "stt/lid-whisper-base-int8";
}

/// What a model does.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Task {
    /// Streaming recognizer (`OnlineRecognizer`): partials and stream finals.
    Online,
    /// Utterance recognizer (`OfflineRecognizer`): finals of closed utterances.
    Offline,
    /// Spoken language identification.
    LanguageId,
}

/// Model family (decides options, capabilities and the config layout).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Family {
    /// Prompt-conditioned NeMo cache-aware streaming transducer (per-stream `language`).
    NemotronStreaming,
    /// NeMo TDT offline transducer (Parakeet v3 / Ultra).
    ParakeetTdt,
    Qwen3Asr,
    SenseVoice,
    Moonshine,
    WhisperLid,
}

/// How a family takes hotwords.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum HotwordSupport {
    None,
    /// Context graph on a transducer decoded with modified beam search (needs a `bpe.vocab`).
    TransducerBeamSearch,
    /// Qwen3-ASR system-prompt context (per-stream `hotwords` option).
    Qwen3Context,
}

impl Family {
    pub fn task(self) -> Task {
        match self {
            Family::NemotronStreaming => Task::Online,
            Family::WhisperLid => Task::LanguageId,
            _ => Task::Offline,
        }
    }

    /// Must be told the language (auto-detection is worse on every language: Nemotron).
    pub fn needs_language(self) -> bool {
        matches!(self, Family::NemotronStreaming)
    }

    /// Reports the spoken language in its result (`AsrResult::lang` without a forced language).
    ///
    /// Not Qwen3-ASR: the model recognizes the language on its own, but sherpa-onnx 1.13.8 strips
    /// its `language <Name>` prefix and returns neither the language nor those tokens (measured:
    /// `lang` empty, tokens start at the transcript). The router therefore plans the language-ID
    /// engine for a Qwen3-ASR final in auto mode, which is what gives a streaming model that needs
    /// a language (Nemotron) its language and every final its `lang`.
    pub fn detects_language(self) -> bool {
        matches!(self, Family::SenseVoice | Family::WhisperLid)
    }

    /// Emits emotion / audio-event tags.
    pub fn tags(self) -> bool {
        matches!(self, Family::SenseVoice)
    }

    pub fn hotwords(self) -> HotwordSupport {
        match self {
            Family::ParakeetTdt => HotwordSupport::TransducerBeamSearch,
            Family::Qwen3Asr => HotwordSupport::Qwen3Context,
            _ => HotwordSupport::None,
        }
    }

    /// Uses a free-form decoder prompt (Qwen3-ASR context).
    pub fn prompt(self) -> bool {
        matches!(self, Family::Qwen3Asr)
    }

    /// Default ONNX Runtime threads (streaming: 1, see `lib::threads_or`).
    pub fn default_threads(self) -> u32 {
        match self.task() {
            Task::Online => super::lib::STREAMING_DEFAULT_THREADS,
            _ => super::lib::DEFAULT_THREADS,
        }
    }
}

/// Files of an installed model folder, relative to it.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Layout {
    Transducer {
        encoder: &'static str,
        decoder: &'static str,
        joiner: &'static str,
        tokens: &'static str,
    },
    Qwen3 {
        conv_frontend: &'static str,
        encoder: &'static str,
        decoder: &'static str,
        /// Folder with `vocab.json` / `merges.txt` / `tokenizer_config.json`.
        tokenizer: &'static str,
    },
    SenseVoice {
        model: &'static str,
        tokens: &'static str,
    },
    /// Moonshine v2 layout (encoder + merged decoder, no preprocessor).
    Moonshine {
        encoder: &'static str,
        merged_decoder: &'static str,
        tokens: &'static str,
    },
    WhisperLid {
        encoder: &'static str,
        decoder: &'static str,
    },
}

impl Layout {
    /// Every file (and folder) the config references, relative to the install folder.
    pub fn paths(&self) -> Vec<&'static str> {
        match *self {
            Layout::Transducer {
                encoder,
                decoder,
                joiner,
                tokens,
            } => vec![encoder, decoder, joiner, tokens],
            Layout::Qwen3 {
                conv_frontend,
                encoder,
                decoder,
                tokenizer,
            } => vec![conv_frontend, encoder, decoder, tokenizer],
            Layout::SenseVoice { model, tokens } => vec![model, tokens],
            Layout::Moonshine {
                encoder,
                merged_decoder,
                tokens,
            } => vec![encoder, merged_decoder, tokens],
            Layout::WhisperLid { encoder, decoder } => vec![encoder, decoder],
        }
    }
}

/// One sherpa catalog model.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct CatalogModel {
    pub id: &'static str,
    pub family: Family,
    pub layout: Layout,
    /// Languages the model transcribes (normalized codes); empty for language ID (any).
    pub languages: &'static [&'static str],
    /// Streaming chunk of an online model (ms), 0 otherwise.
    pub chunk_ms: u32,
}

const NEMOTRON_LAYOUT: Layout = Layout::Transducer {
    encoder: "encoder.int8.onnx",
    decoder: "decoder.int8.onnx",
    joiner: "joiner.int8.onnx",
    tokens: "tokens.txt",
};
const PARAKEET_LAYOUT: Layout = NEMOTRON_LAYOUT;

static QWEN3_CODES: [&str; 30] = {
    let mut out = [""; 30];
    let mut i = 0;
    while i < QWEN3_LANGUAGES.len() {
        out[i] = QWEN3_LANGUAGES[i].0;
        i += 1;
    }
    out
};

static MODELS: &[CatalogModel] = &[
    CatalogModel {
        id: ids::NEMOTRON_560,
        family: Family::NemotronStreaming,
        layout: NEMOTRON_LAYOUT,
        languages: NEMOTRON_LANGUAGES,
        chunk_ms: 560,
    },
    CatalogModel {
        id: ids::NEMOTRON_160,
        family: Family::NemotronStreaming,
        layout: NEMOTRON_LAYOUT,
        languages: NEMOTRON_LANGUAGES,
        chunk_ms: 160,
    },
    CatalogModel {
        id: ids::PARAKEET_ULTRA,
        family: Family::ParakeetTdt,
        layout: PARAKEET_LAYOUT,
        languages: PARAKEET_LANGUAGES,
        chunk_ms: 0,
    },
    CatalogModel {
        id: ids::PARAKEET_V3,
        family: Family::ParakeetTdt,
        layout: PARAKEET_LAYOUT,
        languages: PARAKEET_LANGUAGES,
        chunk_ms: 0,
    },
    CatalogModel {
        id: ids::QWEN3_ASR_0_6B,
        family: Family::Qwen3Asr,
        layout: Layout::Qwen3 {
            conv_frontend: "conv_frontend.onnx",
            encoder: "encoder.int8.onnx",
            decoder: "decoder.int8.onnx",
            tokenizer: "tokenizer",
        },
        languages: &QWEN3_CODES,
        chunk_ms: 0,
    },
    CatalogModel {
        id: ids::SENSE_VOICE,
        family: Family::SenseVoice,
        layout: Layout::SenseVoice {
            model: "model.int8.onnx",
            tokens: "tokens.txt",
        },
        languages: SENSE_VOICE_LANGUAGES,
        chunk_ms: 0,
    },
    CatalogModel {
        id: ids::MOONSHINE_BASE_EN,
        family: Family::Moonshine,
        layout: Layout::Moonshine {
            encoder: "encoder_model.ort",
            merged_decoder: "decoder_model_merged.ort",
            tokens: "tokens.txt",
        },
        languages: ENGLISH,
        chunk_ms: 0,
    },
    CatalogModel {
        id: ids::LID_WHISPER_BASE,
        family: Family::WhisperLid,
        layout: Layout::WhisperLid {
            encoder: "base-encoder.int8.onnx",
            decoder: "base-decoder.int8.onnx",
        },
        languages: &[],
        chunk_ms: 0,
    },
];

/// Every sherpa catalog STT model.
pub fn all() -> &'static [CatalogModel] {
    MODELS
}

/// The model with catalog id `id`.
pub fn find(id: &str) -> Option<&'static CatalogModel> {
    MODELS.iter().find(|m| m.id == id)
}

/// Feature dimension of NeMo models (128 mel bins; also in their encoder metadata).
const NEMO_FEATURE_DIM: u32 = 128;
/// Longest audio one Qwen3-ASR call decodes (s): the pipeline force-closes a segment at
/// `max_utterance_ms` (clamped to 60 s by `speech_core::VadConfig::sanitized`, default 18 s) and
/// adds at most 2 s of pre-roll.
pub const QWEN3_MAX_AUDIO_SECONDS: u32 = 62;
/// Qwen3-ASR audio tokens per second of audio (measured through sherpa-onnx 1.13.8: 662 for
/// 51 s, 780 for 60 s).
pub const QWEN3_AUDIO_TOKENS_PER_SECOND: u32 = 13;
/// Prompt tokens besides the audio: the chat scaffold (18 measured) and the custom-terms / prompt
/// context, counted at one token per character of its longest form
/// ([`hotwords::MAX_QWEN3_CONTEXT_CHARS`](super::hotwords::MAX_QWEN3_CONTEXT_CHARS); CJK terms
/// are about one token per character).
pub const QWEN3_PROMPT_TOKENS: u32 = 32 + super::hotwords::MAX_QWEN3_CONTEXT_CHARS as u32;
/// Qwen3-ASR: most tokens one utterance may produce, sized for the longest utterance
/// ([`QWEN3_MAX_AUDIO_SECONDS`]) rather than a typical one: measured 331 tokens for 60 s of fast
/// conversational English and 182 for 51 s of Mandarin, so 768 leaves room for denser speech (a
/// fixed 256 silently cut long captions). The rare repetition loops (research §1.1) are cut by
/// sherpa-onnx 1.13.8 itself ("decode collapsed into a repetition loop"); this cap only bounds
/// what that guard misses.
pub const QWEN3_MAX_NEW_TOKENS: i32 = 768;
/// sherpa-onnx's Qwen3-ASR `max_total_len` for [`QWEN3_MAX_NEW_TOKENS`].
pub const QWEN3_MAX_TOTAL_LEN: i32 = qwen3_max_total_len(QWEN3_MAX_NEW_TOKENS);

/// Qwen3-ASR `max_total_len` (the decoder's KV context: prompt + audio + generated tokens) that
/// fits the longest utterance, the longest context and `max_new_tokens` (0: the default cap).
/// sherpa-onnx's default (512) is far too small: past ~37 s of audio it drops audio tokens to fit
/// and leaves no room to generate (measured: 51 s and 60 s utterances decoded to a single token;
/// at 1024 a 60 s one stopped after 227 of its 331 tokens). Larger values cost no memory or time
/// (measured: same private memory and decode time at 1024, 2048 and 4096).
pub const fn qwen3_max_total_len(max_new_tokens: i32) -> i32 {
    let context = QWEN3_MAX_AUDIO_SECONDS * QWEN3_AUDIO_TOKENS_PER_SECOND + QWEN3_PROMPT_TOKENS;
    context as i32 + if max_new_tokens > 0 { max_new_tokens } else { QWEN3_MAX_NEW_TOKENS }
}

impl CatalogModel {
    pub fn task(&self) -> Task {
        self.family.task()
    }

    /// Languages whose finals from this model are caption quality: every language of an utterance
    /// model; for Nemotron streaming only its "transcription-ready" locales (the router takes an
    /// utterance model's final for the rest).
    pub fn final_languages(&self) -> &'static [&'static str] {
        match self.family {
            Family::NemotronStreaming => NEMOTRON_TRANSCRIPTION_READY,
            _ => self.languages,
        }
    }

    /// Whether `code` (normalized) is one of the model's languages (language ID: any).
    pub fn supports_language(&self, code: &str) -> bool {
        self.languages.is_empty() || self.languages.contains(&code)
    }

    /// The sherpa config of the model installed in `dir` (default options; the engine layers its
    /// settings on top). Files are checked when the recognizer is created.
    pub fn config(&self, dir: &Path) -> SherpaModel {
        let p = |rel: &str| -> PathBuf { dir.join(rel) };
        match (self.family, self.layout) {
            (
                Family::NemotronStreaming,
                Layout::Transducer {
                    encoder,
                    decoder,
                    joiner,
                    tokens,
                },
            ) => {
                let mut c = OnlineRecognizerConfig::new(
                    OnlineModel::Transducer {
                        encoder: p(encoder),
                        decoder: p(decoder),
                        joiner: p(joiner),
                    },
                    p(tokens),
                );
                c.feature_dim = NEMO_FEATURE_DIM;
                SherpaModel::Online(c)
            }
            (
                _,
                Layout::Transducer {
                    encoder,
                    decoder,
                    joiner,
                    tokens,
                },
            ) => {
                let mut c = OfflineRecognizerConfig::new(
                    OfflineModel::Transducer {
                        encoder: p(encoder),
                        decoder: p(decoder),
                        joiner: p(joiner),
                    },
                    Some(p(tokens)),
                );
                c.model_type = Some("nemo_transducer".into());
                c.feature_dim = NEMO_FEATURE_DIM;
                SherpaModel::Offline(c)
            }
            (
                _,
                Layout::Qwen3 {
                    conv_frontend,
                    encoder,
                    decoder,
                    tokenizer,
                },
            ) => SherpaModel::Offline(OfflineRecognizerConfig::new(
                OfflineModel::Qwen3Asr {
                    conv_frontend: p(conv_frontend),
                    encoder: p(encoder),
                    decoder: p(decoder),
                    tokenizer: p(tokenizer),
                    max_total_len: Some(QWEN3_MAX_TOTAL_LEN),
                    max_new_tokens: Some(QWEN3_MAX_NEW_TOKENS),
                    hotwords: None,
                },
                None,
            )),
            (_, Layout::SenseVoice { model, tokens }) => SherpaModel::Offline(OfflineRecognizerConfig::new(
                OfflineModel::SenseVoice {
                    model: p(model),
                    language: None,
                    use_itn: true,
                },
                Some(p(tokens)),
            )),
            (
                _,
                Layout::Moonshine {
                    encoder,
                    merged_decoder,
                    tokens,
                },
            ) => SherpaModel::Offline(OfflineRecognizerConfig::new(
                OfflineModel::Moonshine {
                    preprocessor: None,
                    encoder: p(encoder),
                    uncached_decoder: None,
                    cached_decoder: None,
                    merged_decoder: Some(p(merged_decoder)),
                },
                Some(p(tokens)),
            )),
            (_, Layout::WhisperLid { encoder, decoder }) => SherpaModel::LanguageId(LanguageIdConfig {
                encoder: p(encoder),
                decoder: p(decoder),
                tail_paddings: None,
                num_threads: None,
            }),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::services::model_store::{Catalog, EntryKind};

    #[test]
    fn ids_are_unique_and_tasks_match_layouts() {
        for (i, m) in MODELS.iter().enumerate() {
            assert!(MODELS[i + 1..].iter().all(|o| o.id != m.id), "duplicate {}", m.id);
            assert!(m.id.starts_with("stt/"), "{}", m.id);
            let cfg = m.config(Path::new("C:/models/x"));
            match m.task() {
                Task::Online => assert!(matches!(cfg, SherpaModel::Online(_)), "{}", m.id),
                Task::Offline => assert!(matches!(cfg, SherpaModel::Offline(_)), "{}", m.id),
                Task::LanguageId => assert!(matches!(cfg, SherpaModel::LanguageId(_)), "{}", m.id),
            }
            assert_eq!(m.chunk_ms > 0, m.task() == Task::Online, "{}", m.id);
            assert_eq!(m.languages.is_empty(), m.task() == Task::LanguageId, "{}", m.id);
            assert!(find(m.id).is_some());
        }
        assert!(find("stt/nope").is_none());
        assert_eq!(QWEN3_CODES.len(), QWEN3_LANGUAGES.len());
        assert!(QWEN3_CODES.iter().all(|c| !c.is_empty()));
    }

    #[test]
    fn configs_use_the_catalog_files_and_family_options() {
        let dir = Path::new("C:/m");
        let SherpaModel::Online(n) = find(ids::NEMOTRON_560).unwrap().config(dir) else {
            panic!()
        };
        assert_eq!(n.tokens, dir.join("tokens.txt"));
        assert_eq!(n.feature_dim, 128);
        assert!(n.num_threads.is_none() && !n.decoding.is_beam_search());
        let SherpaModel::Offline(pk) = find(ids::PARAKEET_ULTRA).unwrap().config(dir) else {
            panic!()
        };
        assert_eq!(pk.model_type.as_deref(), Some("nemo_transducer"));
        assert_eq!(pk.feature_dim, 128);
        let SherpaModel::Offline(q) = find(ids::QWEN3_ASR_0_6B).unwrap().config(dir) else {
            panic!()
        };
        assert!(
            matches!(q.model, OfflineModel::Qwen3Asr { max_new_tokens: Some(QWEN3_MAX_NEW_TOKENS), max_total_len: Some(QWEN3_MAX_TOTAL_LEN), ref tokenizer, .. } if tokenizer == &dir.join("tokenizer"))
        );
        assert!(q.tokens.is_none());
        let SherpaModel::Offline(m) = find(ids::MOONSHINE_BASE_EN).unwrap().config(dir) else {
            panic!()
        };
        assert!(matches!(
            m.model,
            OfflineModel::Moonshine {
                preprocessor: None,
                merged_decoder: Some(_),
                ..
            }
        ));
        assert_eq!(find(ids::QWEN3_ASR_0_6B).unwrap().family.hotwords(), HotwordSupport::Qwen3Context);
        assert!(find(ids::SENSE_VOICE).unwrap().family.tags());
        assert!(find(ids::NEMOTRON_160).unwrap().family.needs_language());
        assert_eq!(find(ids::NEMOTRON_160).unwrap().family.default_threads(), 1);
        assert_eq!(find(ids::SENSE_VOICE).unwrap().family.default_threads(), 2);
        assert!(find(ids::LID_WHISPER_BASE).unwrap().supports_language("xx"));
        let nemotron = find(ids::NEMOTRON_560).unwrap();
        assert!(nemotron.final_languages().contains(&"ja") && !nemotron.final_languages().contains(&"zh"));
        assert!(nemotron.supports_language("zh"));
        assert_eq!(find(ids::SENSE_VOICE).unwrap().final_languages(), SENSE_VOICE_LANGUAGES);
        assert!(!find(ids::MOONSHINE_BASE_EN)
            .unwrap()
            .supports_language("ja"));
    }

    /// Only families whose results carry the detected language claim it: the router plans the
    /// language-ID engine for the others in auto mode.
    #[test]
    fn only_families_that_report_the_language_claim_detection() {
        assert!(Family::SenseVoice.detects_language() && Family::WhisperLid.detects_language());
        for f in [Family::Qwen3Asr, Family::ParakeetTdt, Family::Moonshine, Family::NemotronStreaming] {
            assert!(!f.detects_language(), "{f:?}");
        }
        assert!(!find(ids::QWEN3_ASR_0_6B).unwrap().family.detects_language());
    }

    /// The Qwen3-ASR limits cover the longest utterance the pipeline can close, with the longest
    /// context and every token the cap allows (a too-small `max_total_len` drops audio and
    /// generates nothing).
    #[test]
    fn qwen3_limits_cover_the_longest_utterance() {
        use crate::services::speech_core::VadConfig;
        let widest = VadConfig {
            max_utterance_ms: u32::MAX,
            pre_roll_ms: u32::MAX,
            ..VadConfig::default()
        }
        .sanitized();
        assert!(widest.max_utterance_ms + widest.pre_roll_ms <= QWEN3_MAX_AUDIO_SECONDS * 1000, "{widest:?}");
        let audio = (QWEN3_MAX_AUDIO_SECONDS * QWEN3_AUDIO_TOKENS_PER_SECOND) as i32;
        // Measured: 60 s of audio took 780 audio tokens and produced 331 text tokens; sherpa's
        // default `max_total_len` (512) truncates long utterances.
        const {
            assert!(QWEN3_MAX_AUDIO_SECONDS * QWEN3_AUDIO_TOKENS_PER_SECOND >= 780 && QWEN3_MAX_NEW_TOKENS >= 2 * 331);
            assert!(QWEN3_PROMPT_TOKENS as usize > super::super::hotwords::MAX_QWEN3_CONTEXT_CHARS);
            assert!(QWEN3_MAX_TOTAL_LEN > 512);
        }
        assert_eq!(QWEN3_MAX_TOTAL_LEN, audio + QWEN3_PROMPT_TOKENS as i32 + QWEN3_MAX_NEW_TOKENS);
        // A larger `maxNewTokens` option grows the context with it; 0 means the default.
        assert_eq!(qwen3_max_total_len(1024) - qwen3_max_total_len(768), 256);
        assert_eq!(qwen3_max_total_len(0), QWEN3_MAX_TOTAL_LEN);
    }

    /// Auto mode on the shipped routing table: the Qwen3-ASR-0.6B final does not report the
    /// language, so the plan runs the language-ID engine (which gives Nemotron its language and
    /// each final its `lang`), in two-pass and in single-pass mode.
    #[test]
    fn auto_mode_plans_language_id_for_a_qwen3_final() {
        use crate::services::speech::engine_api::{EngineSpec, RouteRequest, Tier};
        use crate::services::speech::router::{self, RouteContext};
        let table = router::table().expect("the embedded routing table loads");
        let ctx = RouteContext {
            physical_cores: 8,
            installed: table.referenced_ids(),
            gpu_reason: Some("no_gpu".into()),
            ..RouteContext::default()
        };
        let model = |s: &Option<EngineSpec>| s.as_ref().map(|s| s.model.clone());
        for two_pass in [true, false] {
            let request = RouteRequest {
                language: "auto".into(),
                tier: Tier::Cpu,
                two_pass,
            };
            let plan = router::plan_with(&request, &ctx);
            assert_eq!(model(&plan.final_engine).as_deref(), Some(ids::QWEN3_ASR_0_6B), "{plan:?}");
            assert_eq!(model(&plan.language_id).as_deref(), Some(ids::LID_WHISPER_BASE), "{plan:?}");
            assert!(plan.models.iter().any(|m| m == ids::LID_WHISPER_BASE), "{plan:?}");
            if two_pass {
                // Nemotron needs a language: it gets the detected one instead of waiting forever.
                assert_eq!(model(&plan.streaming).as_deref(), Some(ids::NEMOTRON_560), "{plan:?}");
            }
        }
    }

    /// The registry and `model-catalog/stt-sherpa.json` describe the same models, and every file
    /// a config references is installed by its catalog entry.
    #[test]
    fn registry_matches_the_embedded_catalog() {
        let catalog = Catalog::embedded().unwrap();
        let fragment: serde_json::Value = serde_json::from_str(include_str!("../../../../model-catalog/stt-sherpa.json")).unwrap();
        let fragment_ids: Vec<&str> = fragment["entries"]
            .as_array()
            .unwrap()
            .iter()
            .map(|e| e["id"].as_str().unwrap())
            .collect();
        for id in &fragment_ids {
            assert!(find(id).is_some(), "{id} is in stt-sherpa.json but not in the registry");
        }
        for m in MODELS {
            assert!(fragment_ids.contains(&m.id), "{} is missing from stt-sherpa.json", m.id);
            let e = catalog
                .get(m.id)
                .unwrap_or_else(|| panic!("{} is not embedded", m.id));
            assert_eq!(e.kind, EntryKind::Model);
            assert_eq!(e.family, "stt");
            assert!(
                e.requires
                    .iter()
                    .any(|r| r == super::super::lib::RUNTIME_ID),
                "{} requires the runtime",
                m.id
            );
            for rel in m.layout.paths() {
                let covered = e.files.iter().any(|f| match &f.archive {
                    None => f.path == rel || f.path.starts_with(&format!("{rel}/")),
                    Some(a) => {
                        !a.include.is_empty()
                            && a.include
                                .iter()
                                .any(|g| g == rel || g.starts_with(&format!("{rel}/")))
                    }
                });
                assert!(covered, "{}: {rel} is not installed by its catalog entry", m.id);
            }
        }
    }
}
