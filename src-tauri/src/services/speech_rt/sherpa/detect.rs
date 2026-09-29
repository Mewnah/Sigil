//! Recognize an official sherpa-onnx model package folder (as extracted from the k2-fsa
//! `asr-models` / `tts-models` releases) and build the matching config.
//!
//! Used by the engine probe and, in P2, by "import a local model folder". Catalog models do not
//! need it (their entries name the family and files). Detection only looks at file and folder
//! names; `hints.kind` overrides it whenever a package is named unusually. When a role has fp32
//! and int8 files, int8 wins (the CPU tier) unless `hints.prefer_fp32`.

use std::path::{Path, PathBuf};

use serde::{Deserialize, Serialize};

use super::lid::LanguageIdConfig;
use super::offline::{OfflineModel, OfflineRecognizerConfig};
use super::online::{OnlineModel, OnlineRecognizerConfig};
use super::speaker::SpeakerEmbeddingConfig;
use super::tts::{TtsConfig, TtsModel};

/// What a model package is, with a ready-to-use config. Serializable both ways so the exact config
/// can be handed to [`super::probe::validate_in_child`].
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(tag = "task", content = "config", rename_all = "snake_case")]
pub enum SherpaModel {
    /// Streaming ASR.
    Online(OnlineRecognizerConfig),
    /// Non-streaming ASR.
    Offline(OfflineRecognizerConfig),
    Tts(TtsConfig),
    LanguageId(LanguageIdConfig),
    SpeakerEmbedding(SpeakerEmbeddingConfig),
}

impl SherpaModel {
    /// `online/<family>`, `offline/<family>`, `tts/<family>`, `lid/whisper`, `speaker`.
    pub fn label(&self) -> String {
        match self {
            SherpaModel::Online(c) => format!("online/{}", c.model.family()),
            SherpaModel::Offline(c) => format!("offline/{}", c.model.family()),
            SherpaModel::Tts(c) => format!("tts/{}", c.model.family()),
            SherpaModel::LanguageId(_) => "lid/whisper".into(),
            SherpaModel::SpeakerEmbedding(_) => "speaker".into(),
        }
    }

    /// The config's own checks (files and option combinations), as the matching `new` runs them.
    pub fn validate(&self) -> Result<(), String> {
        match self {
            SherpaModel::Online(c) => c.validate(),
            SherpaModel::Offline(c) => c.validate(),
            SherpaModel::Tts(c) => c.validate(),
            SherpaModel::LanguageId(c) => c.validate(),
            SherpaModel::SpeakerEmbedding(c) => c.validate(),
        }
    }
}

/// Overrides for packages whose names do not tell the family.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct DetectHints {
    /// Family or task: `sense_voice`, `transducer`, `paraformer`, `nemo_ctc`, `whisper`,
    /// `moonshine`, `qwen3_asr`, `canary`, `cohere_transcribe`, `fire_red_asr`, `dolphin`,
    /// `zipformer_ctc`, `wenet_ctc`, `omnilingual`, `kokoro`, `kitten`, `vits`, `piper`,
    /// `matcha`, `lid`, `speaker`.
    pub kind: Option<String>,
    /// `online` or `offline` for families that exist in both forms.
    pub mode: Option<String>,
    /// Prefer fp32 files over int8 ones.
    pub prefer_fp32: bool,
    /// Matcha's separately downloaded vocoder.
    pub vocoder: Option<PathBuf>,
}

/// Build the config of the model package at `path` (a folder, or one `.onnx` file for speaker
/// embeddings).
pub fn detect(path: &Path, hints: &DetectHints) -> Result<SherpaModel, String> {
    let kind = hints.kind.as_deref().map(normalize_kind);
    if path.is_file() {
        return match kind.as_deref() {
            None | Some("speaker") => Ok(SherpaModel::SpeakerEmbedding(SpeakerEmbeddingConfig {
                model: path.to_path_buf(),
                num_threads: None,
            })),
            Some(k) => Err(format!("{} is a file; pass the {k} model's folder", path.display())),
        };
    }
    if !path.is_dir() {
        return Err(format!("{} does not exist", path.display()));
    }
    let pkg = Package::scan(path, hints.prefer_fp32)?;
    let name = path
        .file_name()
        .map(|n| n.to_string_lossy().to_lowercase())
        .unwrap_or_default();
    let has = |s: &str| name.contains(s);
    let online = match hints.mode.as_deref() {
        Some("online") | Some("streaming") => true,
        Some("offline") => false,
        _ => has("streaming") || has("nemotron") || has("online"),
    };

    let family = match kind {
        Some(k) => k,
        None => guess_family(&pkg, &name).ok_or_else(|| {
            format!(
                "cannot tell which sherpa-onnx model family {} holds; pass `-- kind=<family>` \
                 (sense_voice, transducer, whisper, moonshine, qwen3_asr, kokoro, kitten, vits, ...)",
                path.display()
            )
        })?,
    };
    build(&family, &pkg, &name, online, hints)
}

fn normalize_kind(k: &str) -> String {
    let k = k.trim().to_lowercase().replace('-', "_");
    match k.as_str() {
        "sensevoice" => "sense_voice".into(),
        "piper" => "vits".into(),
        "qwen3" | "qwen3asr" | "qwen" => "qwen3_asr".into(),
        "parakeet" | "nemo_transducer" | "nemotron" | "zipformer" => "transducer".into(),
        "language_id" | "slid" => "lid".into(),
        "speaker_embedding" | "campplus" => "speaker".into(),
        "cohere" => "cohere_transcribe".into(),
        _ => k,
    }
}

fn guess_family(pkg: &Package, name: &str) -> Option<String> {
    let has = |s: &str| name.contains(s);
    let f = if pkg.file("voices.bin").is_some() {
        if has("kitten") {
            "kitten"
        } else {
            "kokoro"
        }
    } else if pkg.role(&["conv", "frontend"]).is_some() {
        "qwen3_asr"
    } else if pkg.role(&["encoder"]).is_some() && pkg.role(&["decoder"]).is_some() && pkg.role(&["joiner"]).is_some() {
        "transducer"
    } else if has("whisper") {
        "whisper"
    } else if has("moonshine") {
        "moonshine"
    } else if has("sense-voice") || has("sense_voice") || has("sensevoice") {
        "sense_voice"
    } else if has("paraformer") {
        "paraformer"
    } else if has("dolphin") {
        "dolphin"
    } else if has("fire-red") || has("fire_red") || has("firered") {
        "fire_red_asr"
    } else if has("canary") {
        "canary"
    } else if has("cohere") {
        "cohere_transcribe"
    } else if has("omnilingual") {
        "omnilingual"
    } else if has("wenet") {
        "wenet_ctc"
    } else if has("zipformer") && has("ctc") {
        "zipformer_ctc"
    } else if has("nemo") && has("ctc") {
        "nemo_ctc"
    } else if has("t-one") || has("t_one") {
        "tone_ctc"
    } else if has("matcha") {
        "matcha"
    } else if pkg.file("tokens.txt").is_some() && (pkg.dir("espeak-ng-data").is_some() || pkg.file("lexicon.txt").is_some()) {
        "vits"
    } else if pkg.file("tokens.txt").is_none() && pkg.models.len() == 1 {
        "speaker"
    } else {
        return None;
    };
    Some(f.to_string())
}

fn build(family: &str, pkg: &Package, name: &str, online: bool, hints: &DetectHints) -> Result<SherpaModel, String> {
    let has = |s: &str| name.contains(s);
    let tokens = || {
        pkg.tokens()
            .ok_or_else(|| format!("{}: tokens.txt is missing", pkg.root.display()))
    };
    let need = |what: &str, v: Option<PathBuf>| v.ok_or_else(|| format!("{}: no {what} model file", pkg.root.display()));
    Ok(match family {
        "transducer" => {
            let (encoder, decoder, joiner) = (
                need("encoder", pkg.role(&["encoder"]))?,
                need("decoder", pkg.role(&["decoder"]))?,
                need("joiner", pkg.role(&["joiner"]))?,
            );
            if online {
                SherpaModel::Online(OnlineRecognizerConfig::new(
                    OnlineModel::Transducer { encoder, decoder, joiner },
                    tokens()?,
                ))
            } else {
                let mut c = OfflineRecognizerConfig::new(OfflineModel::Transducer { encoder, decoder, joiner }, Some(tokens()?));
                if has("nemo") || has("parakeet") {
                    c.model_type = Some("nemo_transducer".into());
                }
                SherpaModel::Offline(c)
            }
        }
        "paraformer" => match (pkg.role(&["encoder"]), pkg.role(&["decoder"])) {
            (Some(encoder), Some(decoder)) => {
                SherpaModel::Online(OnlineRecognizerConfig::new(OnlineModel::Paraformer { encoder, decoder }, tokens()?))
            }
            _ => SherpaModel::Offline(OfflineRecognizerConfig::new(
                OfflineModel::Paraformer {
                    model: need("paraformer", pkg.any_model())?,
                },
                Some(tokens()?),
            )),
        },
        "sense_voice" => SherpaModel::Offline(OfflineRecognizerConfig::new(
            OfflineModel::SenseVoice {
                model: need("SenseVoice", pkg.role(&["model"]).or_else(|| pkg.any_model()))?,
                language: None,
                use_itn: true,
            },
            Some(tokens()?),
        )),
        "whisper" | "lid" => {
            let encoder = need("encoder", pkg.role(&["encoder"]))?;
            let decoder = need("decoder", pkg.role(&["decoder"]))?;
            if family == "lid" {
                SherpaModel::LanguageId(LanguageIdConfig {
                    encoder,
                    decoder,
                    tail_paddings: None,
                    num_threads: None,
                })
            } else {
                SherpaModel::Offline(OfflineRecognizerConfig::new(
                    OfflineModel::Whisper {
                        encoder,
                        decoder,
                        language: None,
                        task: None,
                        tail_paddings: None,
                    },
                    Some(tokens()?),
                ))
            }
        }
        "moonshine" => {
            let uncached = pkg.role(&["uncached"]);
            let cached = pkg.role_where(|t| t.iter().any(|x| x == "cached") && !t.iter().any(|x| x == "uncached"));
            let merged = pkg.role(&["merged"]);
            let encoder = need(
                "encoder",
                pkg.role(&["encoder"])
                    .or_else(|| pkg.role(&["encode"]))
                    .or_else(|| pkg.role(&["encoder", "model"])),
            )?;
            SherpaModel::Offline(OfflineRecognizerConfig::new(
                OfflineModel::Moonshine {
                    preprocessor: pkg
                        .role(&["preprocess"])
                        .or_else(|| pkg.role(&["preprocessor"])),
                    encoder,
                    uncached_decoder: uncached,
                    cached_decoder: cached,
                    merged_decoder: merged,
                },
                Some(tokens()?),
            ))
        }
        "qwen3_asr" => {
            let tokenizer = pkg
                .dir("tokenizer")
                .or_else(|| pkg.file("vocab.json").map(|_| pkg.root.clone()))
                .ok_or_else(|| format!("{}: no tokenizer folder (vocab.json)", pkg.root.display()))?;
            SherpaModel::Offline(OfflineRecognizerConfig::new(
                OfflineModel::Qwen3Asr {
                    conv_frontend: need("conv frontend", pkg.role(&["conv", "frontend"]))?,
                    encoder: need("encoder", pkg.role(&["encoder"]))?,
                    decoder: need("decoder", pkg.role(&["decoder"]))?,
                    tokenizer,
                    max_total_len: None,
                    max_new_tokens: None,
                    hotwords: None,
                },
                pkg.tokens(),
            ))
        }
        "canary" | "cohere_transcribe" | "fire_red_asr" => {
            let encoder = need("encoder", pkg.role(&["encoder"]))?;
            let decoder = need("decoder", pkg.role(&["decoder"]))?;
            let model = match family {
                "canary" => OfflineModel::Canary {
                    encoder,
                    decoder,
                    src_lang: None,
                    tgt_lang: None,
                    use_pnc: true,
                },
                "cohere_transcribe" => OfflineModel::CohereTranscribe {
                    encoder,
                    decoder,
                    language: None,
                    use_punct: true,
                    use_itn: true,
                },
                _ => OfflineModel::FireRedAsr { encoder, decoder },
            };
            SherpaModel::Offline(OfflineRecognizerConfig::new(model, Some(tokens()?)))
        }
        "nemo_ctc" | "zipformer_ctc" | "zipformer2_ctc" | "tone_ctc" | "dolphin" | "wenet_ctc" | "omnilingual" | "fire_red_asr_ctc" => {
            let model = need("CTC", pkg.role(&["model"]).or_else(|| pkg.any_model()))?;
            match (family, online) {
                ("tone_ctc", _) => SherpaModel::Online(OnlineRecognizerConfig::new(OnlineModel::ToneCtc { model }, tokens()?)),
                ("nemo_ctc", true) => SherpaModel::Online(OnlineRecognizerConfig::new(OnlineModel::NemoCtc { model }, tokens()?)),
                ("zipformer_ctc" | "zipformer2_ctc", true) => {
                    SherpaModel::Online(OnlineRecognizerConfig::new(OnlineModel::Zipformer2Ctc { model }, tokens()?))
                }
                _ => {
                    let m = match family {
                        "nemo_ctc" => OfflineModel::NemoCtc { model },
                        "dolphin" => OfflineModel::Dolphin { model },
                        "wenet_ctc" => OfflineModel::WenetCtc { model },
                        "omnilingual" => OfflineModel::Omnilingual { model },
                        "fire_red_asr_ctc" => OfflineModel::FireRedAsrCtc { model },
                        _ => OfflineModel::ZipformerCtc { model },
                    };
                    SherpaModel::Offline(OfflineRecognizerConfig::new(m, Some(tokens()?)))
                }
            }
        }
        "kokoro" | "kitten" => {
            let model = need("TTS", pkg.role(&["model"]).or_else(|| pkg.any_model()))?;
            let voices = pkg
                .file("voices.bin")
                .ok_or_else(|| format!("{}: voices.bin is missing", pkg.root.display()))?;
            let data_dir = pkg.dir("espeak-ng-data");
            let tts = if family == "kitten" {
                TtsModel::Kitten {
                    model,
                    voices,
                    tokens: tokens()?,
                    data_dir,
                }
            } else {
                TtsModel::Kokoro {
                    model,
                    voices,
                    tokens: tokens()?,
                    data_dir,
                    lexicons: pkg.kokoro_lexicons(),
                    lang: None,
                }
            };
            let mut c = TtsConfig::new(tts);
            if family == "kokoro" {
                c.rule_fsts = pkg.rule_fsts();
            }
            SherpaModel::Tts(c)
        }
        "vits" => {
            let model = need("VITS", pkg.any_model())?;
            let mut c = TtsConfig::new(TtsModel::Vits {
                model,
                tokens: tokens()?,
                lexicon: pkg.file("lexicon.txt"),
                data_dir: pkg.dir("espeak-ng-data"),
                noise_scale: None,
                noise_scale_w: None,
            });
            c.rule_fsts = pkg.rule_fsts();
            SherpaModel::Tts(c)
        }
        "matcha" => {
            let vocoder = hints
                .vocoder
                .clone()
                .or_else(|| pkg.role(&["vocos"]).or_else(|| pkg.role(&["hifigan"])))
                .ok_or_else(|| format!("{}: Matcha needs a vocoder (`-- vocoder=<file>`)", pkg.root.display()))?;
            let acoustic = pkg
                .models
                .iter()
                .find(|m| **m != vocoder)
                .cloned()
                .ok_or_else(|| format!("{}: no Matcha acoustic model", pkg.root.display()))?;
            let mut c = TtsConfig::new(TtsModel::Matcha {
                acoustic_model: acoustic,
                vocoder,
                tokens: tokens()?,
                lexicon: pkg.file("lexicon.txt"),
                data_dir: pkg.dir("espeak-ng-data"),
                noise_scale: None,
            });
            c.rule_fsts = pkg.rule_fsts();
            SherpaModel::Tts(c)
        }
        "speaker" => SherpaModel::SpeakerEmbedding(SpeakerEmbeddingConfig {
            model: need("speaker embedding", pkg.any_model())?,
            num_threads: None,
        }),
        other => return Err(format!("unknown sherpa-onnx model family {other:?}")),
    })
}

/// Top-level contents of a package folder.
struct Package {
    root: PathBuf,
    /// `.onnx` / `.ort` files, best precision first.
    models: Vec<PathBuf>,
    files: Vec<PathBuf>,
    dirs: Vec<PathBuf>,
}

fn lower_name(p: &Path) -> String {
    p.file_name()
        .map(|n| n.to_string_lossy().to_lowercase())
        .unwrap_or_default()
}

/// Name tokens of a model file: `encoder-epoch-99-avg-1.int8.onnx` → `encoder, epoch, 99, avg, 1, int8`.
fn tokens_of(p: &Path) -> Vec<String> {
    let name = lower_name(p);
    let stem = name.trim_end_matches(".onnx").trim_end_matches(".ort");
    stem.split(['-', '_', '.'])
        .filter(|s| !s.is_empty())
        .map(str::to_string)
        .collect()
}

impl Package {
    fn scan(root: &Path, prefer_fp32: bool) -> Result<Package, String> {
        let rd = std::fs::read_dir(root).map_err(|e| format!("cannot read {}: {e}", root.display()))?;
        let (mut models, mut files, mut dirs) = (Vec::new(), Vec::new(), Vec::new());
        for entry in rd.flatten() {
            let p = entry.path();
            if p.is_dir() {
                dirs.push(p);
            } else if p.is_file() {
                let n = lower_name(&p);
                if n.ends_with(".onnx") || n.ends_with(".ort") {
                    models.push(p.clone());
                }
                files.push(p);
            }
        }
        // int8 first on the CPU tier (or last with prefer_fp32), then fp32, then fp16; shorter
        // names first; then by name, so detection is deterministic.
        let rank = |p: &PathBuf| {
            let t = tokens_of(p);
            let q = if t
                .iter()
                .any(|x| x == "int8" || x == "quantized" || x == "q8")
            {
                if prefer_fp32 {
                    2
                } else {
                    0
                }
            } else if t.iter().any(|x| x == "fp16") {
                3
            } else {
                1
            };
            (q, lower_name(p).len(), lower_name(p))
        };
        models.sort_by_key(rank);
        files.sort();
        dirs.sort();
        Ok(Package {
            root: root.to_path_buf(),
            models,
            files,
            dirs,
        })
    }

    fn file(&self, name: &str) -> Option<PathBuf> {
        self.files.iter().find(|p| lower_name(p) == name).cloned()
    }

    fn dir(&self, name: &str) -> Option<PathBuf> {
        self.dirs.iter().find(|p| lower_name(p) == name).cloned()
    }

    /// `tokens.txt`, else Whisper's `<size>-tokens.txt`.
    fn tokens(&self) -> Option<PathBuf> {
        self.file("tokens.txt").or_else(|| {
            self.files
                .iter()
                .find(|p| lower_name(p).ends_with("tokens.txt"))
                .cloned()
        })
    }

    /// Best model file whose name tokens include all of `words`.
    fn role(&self, words: &[&str]) -> Option<PathBuf> {
        self.role_where(|t| words.iter().all(|w| t.iter().any(|x| x == w)))
    }

    fn role_where(&self, pred: impl Fn(&[String]) -> bool) -> Option<PathBuf> {
        self.models.iter().find(|p| pred(&tokens_of(p))).cloned()
    }

    fn any_model(&self) -> Option<PathBuf> {
        self.models.first().cloned()
    }

    /// Kokoro lexicons: every `lexicon*.txt`, with US English preferred over British when both
    /// exist (sherpa's multi-lang example).
    fn kokoro_lexicons(&self) -> Vec<PathBuf> {
        let lex: Vec<PathBuf> = self
            .files
            .iter()
            .filter(|p| {
                let n = lower_name(p);
                n.starts_with("lexicon") && n.ends_with(".txt")
            })
            .cloned()
            .collect();
        let has_us = lex.iter().any(|p| lower_name(p).contains("us-en"));
        let mut lex: Vec<PathBuf> = lex
            .into_iter()
            .filter(|p| !(has_us && lower_name(p).contains("gb-en")))
            .collect();
        // English first: sherpa consults lexicons in order.
        lex.sort_by_key(|p| (!lower_name(p).contains("en"), lower_name(p)));
        lex
    }

    /// Text-normalization FSTs in sherpa's documented order (phone, date, number, then the rest).
    fn rule_fsts(&self) -> Vec<PathBuf> {
        let mut fsts: Vec<PathBuf> = self
            .files
            .iter()
            .filter(|p| lower_name(p).ends_with(".fst"))
            .cloned()
            .collect();
        let order = |p: &PathBuf| {
            let n = lower_name(p);
            let k = if n.starts_with("phone") {
                0
            } else if n.starts_with("date") {
                1
            } else if n.starts_with("number") {
                2
            } else {
                3
            };
            (k, n)
        };
        fsts.sort_by_key(order);
        fsts
    }
}
