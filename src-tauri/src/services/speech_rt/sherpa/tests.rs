//! Tests for the sherpa-onnx wrapper.
//!
//! Always run (no DLL, no model, no network): struct layout against the header, missing / foreign
//! DLLs are errors (never panics), config → C struct mapping, file validation, non-ASCII paths,
//! option combinations sherpa would `exit(-1)` on (refused before any native call), the C API's
//! "0 means unset" floats, the streaming input state and the short-utterance guard, ONNX graph
//! output counting, model package detection, probe argument parsing and WAV reading.
//!
//! Ignored (real runtime + models): need `SIGIL_TEST_MODELS_DIR` laid out like the dev cache
//! `%LOCALAPPDATA%\sigil-dev-cache`:
//! - `sherpa-onnx/pack-1.13.8/` (onnxruntime.dll + sherpa-onnx-c-api.dll of the pinned
//!   `sherpa-onnx-v1.13.8-win-x64-shared-MT-Release-lib.tar.bz2`), or `SIGIL_SHERPA_RUNTIME_DIR`;
//! - `sherpa-onnx/models/sherpa-onnx-sense-voice-zh-en-ja-ko-yue-int8-2024-07-17/`;
//! - `sherpa-onnx/models/kokoro-int8-multi-lang-v1_0/`.
//!
//! The coexistence test additionally uses `SIGIL_TEST_CUDA_BIN` / `SIGIL_TEST_CUDNN_BIN` (see
//! `ort_session::preload_test_cuda_runtime`) to run Sigil's own CUDA sessions at the same time.
//! Run: `cargo test --manifest-path src-tauri/Cargo.toml speech_rt::sherpa -- --include-ignored --test-threads=1`.

use std::ffi::{c_char, CStr};
use std::mem::{align_of, offset_of, size_of};
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};
use std::sync::{Arc, Mutex, MutexGuard};

use super::bindings::*;
use super::lib::{c_float, ffi_path, load_uncached, threads, CStrArena, C_API_DLL, C_FLOAT_ZERO, DEFAULT_THREADS, MAX_THREADS, ORT_DLL};
use super::offline::{long_enough, normalize_tag, should_decode, speech_text, MIN_UTTERANCE_MS};
use super::online::InputState;
use super::onnx_info::graph_output_count;
use super::probe::{read_model_config, read_wav, write_wav, Extra};
use super::tts::{noise_scale, silence_scale, DEFAULT_SILENCE_SCALE, MAX_SILENCE_SCALE, MIN_SILENCE_SCALE};
use super::*;
use crate::services::neural_audio::test_onnx::{Dim, Graph, FLOAT, INT64};
use crate::services::speech_rt::EngineProbeArgs;

// ---------------------------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------------------------

/// Tests that load DLLs (fake or real) run one at a time: a fake `onnxruntime.dll` loaded by one
/// test must never be visible to another test's real load.
pub(crate) fn dll_lock() -> MutexGuard<'static, ()> {
    static LOCK: Mutex<()> = Mutex::new(());
    LOCK.lock().unwrap_or_else(|e| e.into_inner())
}

fn tmp_dir(name: &str) -> PathBuf {
    static N: AtomicUsize = AtomicUsize::new(0);
    let dir = std::env::temp_dir()
        .join("sigil-sherpa-tests")
        .join(format!("{name}-{}-{}", std::process::id(), N.fetch_add(1, Ordering::Relaxed)));
    let _ = std::fs::remove_dir_all(&dir);
    std::fs::create_dir_all(&dir).unwrap();
    dir
}

/// Create `root/rel` (a file, or a folder when `rel` ends with `/`).
fn touch(root: &Path, rel: &str) -> PathBuf {
    let p = root.join(rel.trim_end_matches('/'));
    if rel.ends_with('/') {
        std::fs::create_dir_all(&p).unwrap();
    } else {
        std::fs::create_dir_all(p.parent().unwrap()).unwrap();
        std::fs::write(&p, b"x").unwrap();
    }
    p
}

fn package(name: &str, files: &[&str]) -> PathBuf {
    let root = tmp_dir("pkg").join(name);
    std::fs::create_dir_all(&root).unwrap();
    for f in files {
        touch(&root, f);
    }
    root
}

/// A stand-in transducer decoder with `outputs` graph outputs: one for an icefall / k2 decoder
/// (`decoder_out`), more for a NeMo one (`outputs`, `prednet_lengths`, `states`, ...). Only the
/// graph's outputs matter to the wrapper; the initializer makes the parser seek past a large field.
fn transducer_decoder(path: &Path, outputs: usize) -> PathBuf {
    let mut g = Graph::new();
    g.input("y", INT64, &[Dim::Named("N"), Dim::Fixed(2)])
        .init_f32("embedding", &[500, 8], &vec![0.25; 4000])
        .node("Identity", &["y"], &["decoder_out"], &[]);
    for i in 0..outputs {
        let name = if i == 0 { "decoder_out".to_string() } else { format!("state_{i}") };
        g.output(&name, FLOAT, &[Dim::Named("N"), Dim::Fixed(512)]);
    }
    std::fs::write(path, g.model_bytes(17)).unwrap();
    path.to_path_buf()
}

fn streaming_transducer(name: &str, decoder_outputs: usize) -> OnlineRecognizerConfig {
    let root = package(name, &["encoder.int8.onnx", "joiner.int8.onnx", "tokens.txt", "hotwords.txt"]);
    let decoder = transducer_decoder(&root.join("decoder.onnx"), decoder_outputs);
    OnlineRecognizerConfig::new(
        OnlineModel::Transducer {
            encoder: root.join("encoder.int8.onnx"),
            decoder,
            joiner: root.join("joiner.int8.onnx"),
        },
        root.join("tokens.txt"),
    )
}

fn sense_voice_config(root: &Path) -> OfflineRecognizerConfig {
    OfflineRecognizerConfig::new(
        OfflineModel::SenseVoice {
            model: touch(root, "model.int8.onnx"),
            language: None,
            use_itn: true,
        },
        Some(touch(root, "tokens.txt")),
    )
}

/// # Safety
/// `p` must be null or a valid C string.
unsafe fn cs<'a>(p: *const c_char) -> Option<&'a str> {
    if p.is_null() {
        None
    } else {
        Some(CStr::from_ptr(p).to_str().unwrap())
    }
}

fn name_of(p: *const c_char) -> String {
    // SAFETY: pointers under test come from a live CStrArena or a static C string.
    let s = unsafe { cs(p) }.expect("non-null");
    Path::new(s)
        .file_name()
        .unwrap()
        .to_string_lossy()
        .into_owned()
}

// ---------------------------------------------------------------------------------------------
// Layout and loading
// ---------------------------------------------------------------------------------------------

/// The vendored bindings already assert every size / alignment / offset at compile time; this
/// restates the ones the wrapper depends on, as computed from sherpa-onnx 1.13.8 `c-api.h` for
/// x86_64-pc-windows-msvc, so a regenerated header that moves them fails loudly here too.
#[test]
fn struct_layout_matches_header() {
    let sizes: &[(&str, usize, usize)] = &[
        ("SherpaOnnxFeatureConfig", size_of::<SherpaOnnxFeatureConfig>(), 8),
        ("SherpaOnnxOnlineModelConfig", size_of::<SherpaOnnxOnlineModelConfig>(), 136),
        ("SherpaOnnxOnlineRecognizerConfig", size_of::<SherpaOnnxOnlineRecognizerConfig>(), 272),
        ("SherpaOnnxOnlineRecognizerResult", size_of::<SherpaOnnxOnlineRecognizerResult>(), 48),
        ("SherpaOnnxOfflineModelConfig", size_of::<SherpaOnnxOfflineModelConfig>(), 504),
        ("SherpaOnnxOfflineRecognizerConfig", size_of::<SherpaOnnxOfflineRecognizerConfig>(), 608),
        ("SherpaOnnxOfflineRecognizerResult", size_of::<SherpaOnnxOfflineRecognizerResult>(), 128),
        (
            "SherpaOnnxOfflineSenseVoiceModelConfig",
            size_of::<SherpaOnnxOfflineSenseVoiceModelConfig>(),
            24,
        ),
        (
            "SherpaOnnxOfflineQwen3ASRModelConfig",
            size_of::<SherpaOnnxOfflineQwen3ASRModelConfig>(),
            64,
        ),
        (
            "SherpaOnnxOfflineWhisperModelConfig",
            size_of::<SherpaOnnxOfflineWhisperModelConfig>(),
            48,
        ),
        (
            "SherpaOnnxOfflineMoonshineModelConfig",
            size_of::<SherpaOnnxOfflineMoonshineModelConfig>(),
            40,
        ),
        (
            "SherpaOnnxOfflineTtsKokoroModelConfig",
            size_of::<SherpaOnnxOfflineTtsKokoroModelConfig>(),
            64,
        ),
        ("SherpaOnnxOfflineTtsModelConfig", size_of::<SherpaOnnxOfflineTtsModelConfig>(), 416),
        ("SherpaOnnxOfflineTtsConfig", size_of::<SherpaOnnxOfflineTtsConfig>(), 448),
        ("SherpaOnnxGenerationConfig", size_of::<SherpaOnnxGenerationConfig>(), 56),
        ("SherpaOnnxGeneratedAudio", size_of::<SherpaOnnxGeneratedAudio>(), 16),
        (
            "SherpaOnnxSpokenLanguageIdentificationConfig",
            size_of::<SherpaOnnxSpokenLanguageIdentificationConfig>(),
            40,
        ),
        (
            "SherpaOnnxSpeakerEmbeddingExtractorConfig",
            size_of::<SherpaOnnxSpeakerEmbeddingExtractorConfig>(),
            24,
        ),
    ];
    for (name, actual, expected) in sizes {
        assert_eq!(actual, expected, "size of {name}");
    }
    assert_eq!(align_of::<SherpaOnnxOfflineRecognizerConfig>(), 8);
    assert_eq!(align_of::<SherpaOnnxOfflineTtsConfig>(), 8);
    let offsets: &[(&str, usize, usize)] = &[
        (
            "online.decoding_method",
            offset_of!(SherpaOnnxOnlineRecognizerConfig, decoding_method),
            144,
        ),
        ("online.hotwords_buf", offset_of!(SherpaOnnxOnlineRecognizerConfig, hotwords_buf), 232),
        ("online.hr", offset_of!(SherpaOnnxOnlineRecognizerConfig, hr), 248),
        ("offline_model.tokens", offset_of!(SherpaOnnxOfflineModelConfig, tokens), 96),
        ("offline_model.provider", offset_of!(SherpaOnnxOfflineModelConfig, provider), 112),
        ("offline_model.sense_voice", offset_of!(SherpaOnnxOfflineModelConfig, sense_voice), 152),
        ("offline_model.qwen3_asr", offset_of!(SherpaOnnxOfflineModelConfig, qwen3_asr), 408),
        (
            "offline_model.cohere_transcribe",
            offset_of!(SherpaOnnxOfflineModelConfig, cohere_transcribe),
            472,
        ),
        ("offline.lm_config", offset_of!(SherpaOnnxOfflineRecognizerConfig, lm_config), 512),
        (
            "offline.decoding_method",
            offset_of!(SherpaOnnxOfflineRecognizerConfig, decoding_method),
            528,
        ),
        ("offline.hr", offset_of!(SherpaOnnxOfflineRecognizerConfig, hr), 584),
        ("offline_result.lang", offset_of!(SherpaOnnxOfflineRecognizerResult, lang), 48),
        (
            "offline_result.segment_count",
            offset_of!(SherpaOnnxOfflineRecognizerResult, segment_count),
            120,
        ),
        ("tts_model.kokoro", offset_of!(SherpaOnnxOfflineTtsModelConfig, kokoro), 128),
        ("tts_model.kitten", offset_of!(SherpaOnnxOfflineTtsModelConfig, kitten), 192),
        ("tts_model.supertonic", offset_of!(SherpaOnnxOfflineTtsModelConfig, supertonic), 360),
        ("tts.rule_fars", offset_of!(SherpaOnnxOfflineTtsConfig, rule_fars), 432),
        ("tts.silence_scale", offset_of!(SherpaOnnxOfflineTtsConfig, silence_scale), 440),
        ("generation.reference_audio", offset_of!(SherpaOnnxGenerationConfig, reference_audio), 16),
        ("generation.extra", offset_of!(SherpaOnnxGenerationConfig, extra), 48),
    ];
    for (name, actual, expected) in offsets {
        assert_eq!(actual, expected, "offset of {name}");
    }
    // Zero-initialised configs are the C API's documented starting point.
    let c = SherpaOnnxOfflineRecognizerConfig::default();
    assert!(c.model_config.tokens.is_null() && c.decoding_method.is_null() && c.feat_config.sample_rate == 0);
}

#[test]
fn missing_runtime_files_are_errors_not_panics() {
    let _g = dll_lock();
    let empty = tmp_dir("no-pack");
    let e = load_uncached(&empty.join("nope")).unwrap_err();
    assert!(e.contains(ORT_DLL), "{e}");

    // Only the ONNX Runtime half of the pack.
    let half = tmp_dir("half-pack");
    std::fs::write(half.join(ORT_DLL), b"not a dll").unwrap();
    let e = load_uncached(&half).unwrap_err();
    assert!(e.contains(C_API_DLL) && e.contains("missing"), "{e}");
}

#[test]
fn files_that_are_not_dlls_are_refused() {
    let _g = dll_lock();
    let dir = tmp_dir("garbage-pack");
    std::fs::write(dir.join(ORT_DLL), vec![0x4d_u8; 4096]).unwrap();
    std::fs::write(dir.join(C_API_DLL), vec![0x5a_u8; 4096]).unwrap();
    let e = load_uncached(&dir).unwrap_err();
    assert!(e.contains("cannot load") || e.contains("already loaded"), "{e}");
}

/// A real DLL that is not ONNX Runtime (a copy of System32's version.dll) is refused by the ORT
/// API check before sherpa's C API is loaded.
#[cfg(windows)]
#[test]
fn a_dll_that_is_not_onnx_runtime_is_refused() {
    let _g = dll_lock();
    let system = std::env::var_os("SystemRoot")
        .map(PathBuf::from)
        .unwrap_or_else(|| PathBuf::from(r"C:\Windows"));
    let version_dll = system.join("System32").join("version.dll");
    if !version_dll.is_file() {
        eprintln!("no version.dll; skipped");
        return;
    }
    let dir = tmp_dir("foreign-dll");
    std::fs::copy(&version_dll, dir.join(ORT_DLL)).unwrap();
    std::fs::copy(&version_dll, dir.join(C_API_DLL)).unwrap();
    let e = load_uncached(&dir).unwrap_err();
    assert!(e.contains("is not ONNX Runtime") || e.contains("already loaded"), "{e}");
    // The fake was unloaded again: nothing named onnxruntime.dll stays in the process.
    if SherpaRuntime::loaded().is_none() {
        assert!(super::lib::loaded_module_path(ORT_DLL).is_none());
    }
}

#[test]
fn probe_without_a_pack_is_an_error() {
    let _g = dll_lock();
    let args = EngineProbeArgs {
        engine: "sherpa".into(),
        variant: Some("cuda13".into()),
        ..EngineProbeArgs::default()
    };
    let e = headless_probe(&args).unwrap_err();
    assert!(e.contains("CPU-only"), "{e}");
    if SherpaRuntime::loaded().is_some() {
        return; // A real runtime from an ignored test is cached for the process.
    }
    let args = EngineProbeArgs {
        engine: "sherpa".into(),
        runtime_dir: Some(tmp_dir("probe-no-pack")),
        ..EngineProbeArgs::default()
    };
    let e = headless_probe(&args).unwrap_err();
    assert!(e.contains(ORT_DLL), "{e}");
    // The frozen dispatcher routes both spellings here.
    let e = crate::services::speech_rt::headless_engine_probe(&EngineProbeArgs {
        engine: "sherpa-onnx".into(),
        runtime_dir: Some(tmp_dir("probe-no-pack")),
        ..EngineProbeArgs::default()
    })
    .unwrap_err();
    assert!(e.contains(ORT_DLL), "{e}");
}

// ---------------------------------------------------------------------------------------------
// FFI helpers
// ---------------------------------------------------------------------------------------------

#[test]
fn string_arena_and_lists() {
    let mut a = CStrArena::default();
    assert!(a.str("a\0b").is_err());
    assert!(a.opt(None).unwrap().is_null());
    assert!(a.opt(Some("")).unwrap().is_null());
    let p = a.str("greedy_search").unwrap();
    // SAFETY: `p` lives in `a`.
    assert_eq!(unsafe { cs(p) }, Some("greedy_search"));
    assert!(a.path_list(&[]).unwrap().is_null());
    let list = a
        .path_list(&[PathBuf::from("C:/m/a.txt"), PathBuf::from("C:/m/b.txt")])
        .unwrap();
    // SAFETY: as above.
    assert_eq!(unsafe { cs(list) }, Some("C:/m/a.txt,C:/m/b.txt"));
    assert!(a.path_list(&[PathBuf::from("C:/m/a,b.txt")]).is_err());
    // Earlier pointers stay valid while the arena grows.
    for i in 0..100 {
        a.str(&format!("filler {i}")).unwrap();
    }
    // SAFETY: as above.
    assert_eq!(unsafe { cs(p) }, Some("greedy_search"));
}

#[test]
fn thread_counts_are_clamped() {
    assert_eq!(threads(None), DEFAULT_THREADS as i32);
    assert_eq!(threads(Some(0)), 1);
    assert_eq!(threads(Some(3)), 3);
    assert_eq!(threads(Some(64)), MAX_THREADS as i32);
}

#[test]
fn ascii_paths_pass_through() {
    let p = Path::new(r"C:\Users\sigil\AppData\Roaming\com.sigil.app\runtimes\sherpa-onnx\1.13.8\cpu\model.onnx");
    assert_eq!(ffi_path(p).unwrap().to_str().unwrap(), p.to_str().unwrap());
}

/// Non-ASCII folders (user names such as `José` or `山田`) are passed as 8.3 short paths, which
/// sherpa-onnx and espeak-ng can open through the narrow C runtime.
#[cfg(windows)]
#[test]
fn non_ascii_paths_use_the_short_form_when_available() {
    let dir = tmp_dir("unicode").join("José-山田");
    std::fs::create_dir_all(&dir).unwrap();
    let file = dir.join("tokens.txt");
    std::fs::write(&file, b"x").unwrap();
    let c = ffi_path(&file).unwrap();
    let s = c.to_str().unwrap();
    if s.is_ascii() {
        assert!(Path::new(s).is_file(), "short path {s} must name the same file");
    } else {
        // 8.3 names disabled on this volume: the UTF-8 path is passed as a best effort.
        assert_eq!(s, file.to_str().unwrap());
    }
}

#[test]
fn tags_and_similarity() {
    assert_eq!(normalize_tag(Some("<|en|>".into())).as_deref(), Some("en"));
    assert_eq!(normalize_tag(Some("<|NEUTRAL|>".into())).as_deref(), Some("neutral"));
    assert_eq!(normalize_tag(Some("<|nospeech|>".into())), None);
    assert_eq!(normalize_tag(Some("  ".into())), None);
    assert_eq!(normalize_tag(None), None);
    assert_eq!(normalize_tag(Some("ja".into())).as_deref(), Some("ja"));
    assert_eq!(speech_text(" 。 ".into()), "");
    assert_eq!(speech_text("...!?".into()), "");
    assert_eq!(speech_text(" 50円。 ".into()), "50円。");
    assert_eq!(speech_text("はい。".into()), "はい。");
    assert!((cosine_similarity(&[1.0, 0.0], &[1.0, 0.0]).unwrap() - 1.0).abs() < 1e-6);
    assert!(cosine_similarity(&[1.0, 0.0], &[0.0, 1.0]).unwrap().abs() < 1e-6);
    assert_eq!(cosine_similarity(&[1.0], &[1.0, 2.0]), None);
    assert_eq!(cosine_similarity(&[0.0, 0.0], &[1.0, 2.0]), None);
}

// ---------------------------------------------------------------------------------------------
// Config → C struct
// ---------------------------------------------------------------------------------------------

#[test]
fn sense_voice_config_maps_to_the_c_struct() {
    let root = package("sv", &["model.int8.onnx", "tokens.txt"]);
    let cfg = OfflineRecognizerConfig::new(
        OfflineModel::SenseVoice {
            model: root.join("model.int8.onnx"),
            language: None,
            use_itn: true,
        },
        Some(root.join("tokens.txt")),
    );
    cfg.validate().unwrap();
    let mut arena = CStrArena::default();
    let c = cfg.to_raw(&mut arena).unwrap();
    let m = &c.model_config;
    assert_eq!(name_of(m.sense_voice.model), "model.int8.onnx");
    // SAFETY: pointers into `arena` / static strings.
    unsafe {
        assert_eq!(cs(m.sense_voice.language), Some("auto"));
        assert_eq!(cs(m.provider), Some("cpu"), "sherpa is always CPU-only");
        assert_eq!(cs(c.decoding_method), Some("greedy_search"));
        assert!(m.whisper.encoder.is_null() && m.transducer.encoder.is_null() && m.qwen3_asr.encoder.is_null());
        assert!(c.hotwords_file.is_null() && c.rule_fsts.is_null());
    }
    assert_eq!(m.sense_voice.use_itn, 1);
    assert_eq!(name_of(m.tokens), "tokens.txt");
    assert_eq!(m.num_threads, DEFAULT_THREADS as i32);
    assert_eq!((c.feat_config.sample_rate, c.feat_config.feature_dim), (16_000, 80));
}

#[test]
fn qwen3_asr_config_uses_its_tokenizer_folder() {
    let root = package("q3", &["conv_frontend.onnx", "encoder.int8.onnx", "decoder.int8.onnx", "tokenizer/"]);
    let cfg = OfflineRecognizerConfig::new(
        OfflineModel::Qwen3Asr {
            conv_frontend: root.join("conv_frontend.onnx"),
            encoder: root.join("encoder.int8.onnx"),
            decoder: root.join("decoder.int8.onnx"),
            tokenizer: root.join("tokenizer"),
            max_total_len: None,
            max_new_tokens: Some(256),
            hotwords: Some("Sigil,VRChat".into()),
        },
        None,
    );
    cfg.validate().expect("Qwen3-ASR needs no tokens.txt");
    let mut arena = CStrArena::default();
    let c = cfg.to_raw(&mut arena).unwrap();
    let q = &c.model_config.qwen3_asr;
    assert_eq!(name_of(q.tokenizer), "tokenizer");
    assert_eq!(q.max_new_tokens, 256);
    assert_eq!(q.max_total_len, 0, "0 = sherpa's default");
    assert!(q.temperature <= 1e-6, "captions decode greedily");
    // SAFETY: pointers into `arena`.
    unsafe {
        assert_eq!(cs(q.hotwords), Some("Sigil,VRChat"));
        assert!(cs(c.model_config.tokens).is_none());
    }
}

#[test]
fn validation_names_the_missing_file() {
    let root = package("missing", &["tokens.txt"]);
    let cfg = OfflineRecognizerConfig::new(
        OfflineModel::SenseVoice {
            model: root.join("model.int8.onnx"),
            language: Some("ja".into()),
            use_itn: false,
        },
        Some(root.join("tokens.txt")),
    );
    let e = cfg.validate().unwrap_err();
    assert!(e.contains("model.int8.onnx") && e.contains("does not exist"), "{e}");

    let mut no_tokens = cfg.clone();
    touch(&root, "model.int8.onnx");
    no_tokens.tokens = None;
    assert!(no_tokens
        .validate()
        .unwrap_err()
        .contains("tokens.txt is required"));

    let tts = TtsConfig::new(TtsModel::Kitten {
        model: touch(&root, "kitten.onnx"),
        voices: touch(&root, "voices.bin"),
        tokens: root.join("tokens.txt"),
        data_dir: Some(root.join("espeak-ng-data")),
    });
    let e = tts.validate().unwrap_err();
    assert!(e.contains("espeak-ng-data"), "{e}");
}

#[test]
fn streaming_config_maps_endpoint_rules_and_beam_search() {
    let root = package("streaming", &["encoder.int8.onnx", "joiner.int8.onnx", "tokens.txt", "hotwords.txt"]);
    transducer_decoder(&root.join("decoder.onnx"), 1);
    let mut cfg = OnlineRecognizerConfig::new(
        OnlineModel::Transducer {
            encoder: root.join("encoder.int8.onnx"),
            decoder: root.join("decoder.onnx"),
            joiner: root.join("joiner.int8.onnx"),
        },
        root.join("tokens.txt"),
    );
    {
        let mut arena = CStrArena::default();
        assert_eq!(
            cfg.to_raw(&mut arena).unwrap().model_config.num_threads,
            1,
            "streaming decodes every chunk: one thread, no spinning worker pool"
        );
    }
    cfg.num_threads = Some(3);
    cfg.endpoint = Some(EndpointRules::default());
    cfg.decoding = Decoding::ModifiedBeamSearch { max_active_paths: 8 };
    cfg.hotwords_file = Some(root.join("hotwords.txt"));
    cfg.validate().unwrap();
    let mut arena = CStrArena::default();
    let c = cfg.to_raw(&mut arena).unwrap();
    assert_eq!(c.enable_endpoint, 1);
    assert!((c.rule2_min_trailing_silence - 1.2).abs() < 1e-6);
    assert_eq!(c.max_active_paths, 8);
    assert_eq!(c.model_config.num_threads, 3);
    assert_eq!(name_of(c.model_config.transducer.joiner), "joiner.int8.onnx");
    assert_eq!(name_of(c.hotwords_file), "hotwords.txt");
    // SAFETY: pointers into `arena` / static strings.
    unsafe {
        assert_eq!(cs(c.decoding_method), Some("modified_beam_search"));
        assert_eq!(cs(c.model_config.provider), Some("cpu"));
        assert!(c.model_config.paraformer.encoder.is_null());
    }
    // Endpointing stays off unless asked for (Sigil's Silero segmenter decides).
    cfg.endpoint = None;
    assert_eq!(cfg.to_raw(&mut arena).unwrap().enable_endpoint, 0);
}

// ---------------------------------------------------------------------------------------------
// Option combinations sherpa-onnx answers with exit(-1), refused before any native call
// ---------------------------------------------------------------------------------------------

/// Audio after `InputFinished` makes kaldi-native-fbank abort the process, and sherpa's reset keeps
/// that state: a finished stream refuses audio and resets for good.
#[test]
fn streaming_input_finished_is_terminal() {
    let mut input = InputState::default();
    input.accept().unwrap();
    input
        .reset()
        .expect("a reset after an endpoint starts the next utterance");
    input.accept().unwrap();
    assert!(input.finish(), "the first input_finished makes the native call");
    assert!(!input.finish(), "and only the first");
    assert!(input.is_finished());
    let e = input.accept().unwrap_err();
    assert!(e.contains("create a new stream"), "{e}");
    assert!(input.reset().is_err(), "reset never reopens a finished stream");
    assert!(input.accept().is_err(), "audio after input_finished + reset is still refused");
}

#[test]
fn beam_search_and_hotwords_need_a_transducer_offline() {
    let root = tmp_dir("offline-beam");
    // SenseVoice: greedy only; its constructor exits on modified beam search.
    let mut sv = sense_voice_config(&root);
    sv.validate().unwrap();
    assert!(!sv.supports_hotwords());
    let e = sv.require_hotwords_support().unwrap_err();
    assert!(e.contains("does not take hotwords"), "{e}");
    sv.decoding = Decoding::ModifiedBeamSearch { max_active_paths: 4 };
    let e = sv.validate().unwrap_err();
    assert!(e.contains("need a transducer model") && e.contains("sense_voice"), "{e}");
    sv.decoding = Decoding::GreedySearch;
    sv.hotwords_file = Some(touch(&root, "hotwords.txt"));
    assert!(sv
        .validate()
        .unwrap_err()
        .contains("needs modified beam search"));

    // Qwen3-ASR: hotwords are prompt context (model config or the stream's `hotwords` option),
    // never a hotwords stream, file or beam search.
    let q3 = OfflineRecognizerConfig::new(
        OfflineModel::Qwen3Asr {
            conv_frontend: touch(&root, "conv_frontend.onnx"),
            encoder: touch(&root, "encoder.int8.onnx"),
            decoder: touch(&root, "decoder.int8.onnx"),
            tokenizer: touch(&root, "tokenizer/"),
            max_total_len: None,
            max_new_tokens: None,
            hotwords: Some("Sigil".into()),
        },
        None,
    );
    q3.validate().unwrap();
    assert_eq!(q3.hotwords_mode(), super::offline::HotwordsMode::PromptContext);
    q3.require_hotwords_support().unwrap();
    assert_eq!(sv.hotwords_mode(), super::offline::HotwordsMode::None);
    let mut q3_file = q3.clone();
    q3_file.hotwords_file = Some(touch(&root, "q3-hotwords.txt"));
    assert!(q3_file.validate().unwrap_err().contains("prompt context"));
    let mut q3_beam = q3.clone();
    q3_beam.decoding = Decoding::ModifiedBeamSearch { max_active_paths: 4 };
    assert!(q3_beam.validate().unwrap_err().contains("Qwen3-ASR"));

    // Transducers (k2 and NeMo Parakeet): beam search and hotwords are supported.
    let files = ["encoder.int8.onnx", "decoder.int8.onnx", "joiner.int8.onnx", "tokens.txt"];
    let pk = package("sherpa-onnx-nemo-parakeet-tdt-0.6b-v3-int8", &files);
    let SherpaModel::Offline(mut tdt) = detect(&pk, &DetectHints::default()).unwrap() else {
        panic!()
    };
    assert!(tdt
        .require_hotwords_support()
        .unwrap_err()
        .contains("modified beam search"));
    tdt.decoding = Decoding::ModifiedBeamSearch { max_active_paths: 4 };
    tdt.validate().unwrap();
    assert!(tdt.supports_hotwords());
    tdt.require_hotwords_support().unwrap();
    tdt.modeling_unit = Some("bpe".into());
    assert!(tdt.validate().unwrap_err().contains("bpe_vocab"));
    tdt.bpe_vocab = Some(touch(&pk, "bpe.vocab"));
    tdt.validate().unwrap();
}

#[test]
fn beam_search_and_hotwords_need_a_k2_streaming_transducer() {
    // Greedy search never reads the decoder, whatever it is.
    let nemo = streaming_transducer("sherpa-onnx-nemotron-3.5-asr-streaming-0.6b-160ms-int8", 3);
    nemo.validate().unwrap();
    assert!(!nemo.supports_hotwords());

    // Nemotron / NeMo cache-aware transducers (decoder with several outputs): greedy only.
    let mut beam = nemo.clone();
    beam.decoding = Decoding::ModifiedBeamSearch { max_active_paths: 4 };
    let e = beam.validate().unwrap_err();
    assert!(e.contains("NeMo transducer"), "{e}");
    let mut hot = nemo.clone();
    hot.hotwords_file = Some(nemo.tokens.with_file_name("hotwords.txt"));
    assert!(hot
        .validate()
        .unwrap_err()
        .contains("needs modified beam search"));
    hot.decoding = Decoding::ModifiedBeamSearch { max_active_paths: 4 };
    assert!(hot.validate().unwrap_err().contains("NeMo transducer"));

    // icefall / k2 streaming zipformer (one decoder output): beam search and hotwords work.
    let mut k2 = streaming_transducer("sherpa-onnx-streaming-zipformer-en", 1);
    k2.decoding = Decoding::ModifiedBeamSearch { max_active_paths: 4 };
    k2.hotwords_file = Some(k2.tokens.with_file_name("hotwords.txt"));
    k2.validate().unwrap();
    assert!(k2.supports_hotwords());

    // A decoder that is not an ONNX graph is an error, never a guess.
    let mut broken = k2.clone();
    if let OnlineModel::Transducer { decoder, .. } = &broken.model {
        std::fs::write(decoder, b"not onnx").unwrap();
    }
    broken.hotwords_file = None;
    assert!(broken
        .validate()
        .unwrap_err()
        .contains("not a readable ONNX model"));

    // Other streaming families decode greedily only.
    let root = package("paraformer", &["encoder.onnx", "decoder.onnx", "tokens.txt"]);
    let mut para = OnlineRecognizerConfig::new(
        OnlineModel::Paraformer {
            encoder: root.join("encoder.onnx"),
            decoder: root.join("decoder.onnx"),
        },
        root.join("tokens.txt"),
    );
    para.validate().unwrap();
    para.decoding = Decoding::ModifiedBeamSearch { max_active_paths: 4 };
    assert!(para
        .validate()
        .unwrap_err()
        .contains("need a transducer model"));
}

#[test]
fn endpoint_rules_must_be_positive() {
    let mut cfg = streaming_transducer("endpoint", 1);
    cfg.endpoint = Some(EndpointRules::default());
    cfg.validate().unwrap();
    for bad in [0.0, -1.0, f32::NAN, f32::INFINITY] {
        cfg.endpoint = Some(EndpointRules {
            rule3_min_utterance_length: bad,
            ..EndpointRules::default()
        });
        let e = cfg.validate().unwrap_err();
        assert!(e.contains("endpoint rules must be positive"), "{bad}: {e}");
    }
}

#[test]
fn onnx_graph_outputs_are_counted_without_reading_initializers() {
    let dir = tmp_dir("onnx-info");
    for n in [1, 2, 4] {
        let p = transducer_decoder(&dir.join(format!("decoder-{n}.onnx")), n);
        assert_eq!(graph_output_count(&p).unwrap(), n);
    }
    // Truncated inside the graph, no graph at all, garbage, missing: errors.
    let full = std::fs::read(dir.join("decoder-2.onnx")).unwrap();
    std::fs::write(dir.join("cut.onnx"), &full[..full.len() / 2]).unwrap();
    assert!(graph_output_count(&dir.join("cut.onnx")).is_err());
    std::fs::write(dir.join("empty-model.onnx"), [0x08u8, 0x07]).unwrap();
    assert!(graph_output_count(&dir.join("empty-model.onnx"))
        .unwrap_err()
        .contains("no graph"));
    std::fs::write(dir.join("garbage.onnx"), vec![0xffu8; 64]).unwrap();
    assert!(graph_output_count(&dir.join("garbage.onnx")).is_err());
    assert!(graph_output_count(&dir.join("missing.onnx")).is_err());
}

/// The C API reads an exact 0 as "unset" (`SHERPA_ONNX_OR`) and substitutes its default.
#[test]
fn zero_float_options_reach_the_c_api_as_zero_not_as_the_default() {
    assert_eq!(c_float(0.0, 1.5), C_FLOAT_ZERO);
    assert_eq!(c_float(f32::NAN, 1.5), 1.5);
    assert_eq!(c_float(-0.5, 1.5), -0.5);
    // Silence: an exact 0 would become the 0.2 default, and sherpa skips scaling outside
    // [0.01, 10] (full-length pauses), so "no pause" is the smallest scale it applies.
    assert_eq!(silence_scale(None), DEFAULT_SILENCE_SCALE);
    assert_eq!(silence_scale(Some(0.0)), MIN_SILENCE_SCALE, "no pause between sentences");
    assert_eq!(silence_scale(Some(C_FLOAT_ZERO)), MIN_SILENCE_SCALE);
    assert_eq!(silence_scale(Some(-3.0)), MIN_SILENCE_SCALE);
    assert_eq!(silence_scale(Some(f32::NAN)), DEFAULT_SILENCE_SCALE);
    assert_eq!(silence_scale(Some(99.0)), MAX_SILENCE_SCALE);
    assert_eq!(silence_scale(Some(0.5)), 0.5);
    assert_eq!((MIN_SILENCE_SCALE, MAX_SILENCE_SCALE), (0.01, 10.0), "sherpa's ScaleSilence range");
    assert_eq!(noise_scale(None, 0.667), 0.667);
    assert_eq!(noise_scale(Some(0.0), 0.667), C_FLOAT_ZERO);

    let root = package("piper-zero", &["voice.onnx", "tokens.txt"]);
    let mut tts = TtsConfig::new(TtsModel::Vits {
        model: root.join("voice.onnx"),
        tokens: root.join("tokens.txt"),
        lexicon: None,
        data_dir: None,
        noise_scale: Some(0.0),
        noise_scale_w: Some(0.0),
    });
    tts.silence_scale = Some(0.0);
    let mut arena = CStrArena::default();
    let c = tts.to_raw(&mut arena).unwrap();
    assert_eq!(c.silence_scale, MIN_SILENCE_SCALE);
    assert_eq!((c.model.vits.noise_scale, c.model.vits.noise_scale_w), (C_FLOAT_ZERO, C_FLOAT_ZERO));

    let mut asr = sense_voice_config(&tmp_dir("hotword-score"));
    asr.hotwords_score = 0.0;
    asr.blank_penalty = f32::NAN;
    let c = asr.to_raw(&mut arena).unwrap();
    assert_eq!(c.hotwords_score, C_FLOAT_ZERO);
    assert_eq!(c.blank_penalty, 0.0);
}

#[test]
fn offline_streams_skip_the_model_on_too_little_audio() {
    assert_eq!(MIN_UTTERANCE_MS, 100);
    assert!(!should_decode(None), "decode without accepted audio");
    assert!(!should_decode(Some((0, 16_000))));
    assert!(!should_decode(Some((400, 16_000))), "25 ms");
    assert!(!should_decode(Some((1_599, 16_000))));
    assert!(should_decode(Some((1_600, 16_000))), "100 ms");
    assert!(should_decode(Some((4_800, 48_000))));
    assert!(!long_enough(10_000, 0));
    assert!(long_enough(usize::MAX, 384_000), "no overflow");
}

#[test]
fn kokoro_config_joins_lexicons_and_rule_fsts() {
    let root = package(
        "kokoro-int8-multi-lang-v1_0",
        &[
            "model.int8.onnx",
            "voices.bin",
            "tokens.txt",
            "espeak-ng-data/",
            "lexicon-us-en.txt",
            "lexicon-zh.txt",
            "phone-zh.fst",
            "date-zh.fst",
        ],
    );
    let mut cfg = TtsConfig::new(TtsModel::Kokoro {
        model: root.join("model.int8.onnx"),
        voices: root.join("voices.bin"),
        tokens: root.join("tokens.txt"),
        data_dir: Some(root.join("espeak-ng-data")),
        lexicons: vec![root.join("lexicon-us-en.txt"), root.join("lexicon-zh.txt")],
        lang: Some("en-us".into()),
    });
    cfg.rule_fsts = vec![root.join("phone-zh.fst"), root.join("date-zh.fst")];
    cfg.validate().unwrap();
    let mut arena = CStrArena::default();
    let c = cfg.to_raw(&mut arena).unwrap();
    let k = &c.model.kokoro;
    // SAFETY: pointers into `arena` / static strings.
    unsafe {
        let lex = cs(k.lexicon).unwrap();
        assert_eq!(lex.split(',').count(), 2);
        assert!(lex.ends_with("lexicon-zh.txt"));
        assert_eq!(cs(k.lang), Some("en-us"));
        assert_eq!(cs(c.rule_fsts).unwrap().split(',').count(), 2);
        assert_eq!(cs(c.model.provider), Some("cpu"));
        assert!(c.model.vits.model.is_null() && c.model.kitten.model.is_null());
    }
    assert_eq!(k.length_scale, 1.0);
    assert_eq!(c.max_num_sentences, 1, "first sentence first");
    assert!((c.silence_scale - 0.2).abs() < 1e-6);
}

// ---------------------------------------------------------------------------------------------
// Package detection
// ---------------------------------------------------------------------------------------------

fn file_name(p: &Path) -> String {
    p.file_name().unwrap().to_string_lossy().into_owned()
}

#[test]
fn detects_sense_voice_parakeet_and_nemotron_packages() {
    let sv = package(
        "sherpa-onnx-sense-voice-zh-en-ja-ko-yue-int8-2025-09-09",
        &["model.int8.onnx", "tokens.txt", "test_wavs/"],
    );
    match detect(&sv, &DetectHints::default()).unwrap() {
        SherpaModel::Offline(c) => {
            assert!(matches!(&c.model, OfflineModel::SenseVoice { model, .. } if file_name(model) == "model.int8.onnx"));
        }
        other => panic!("{other:?}"),
    }

    let files = ["encoder.int8.onnx", "decoder.int8.onnx", "joiner.int8.onnx", "tokens.txt"];
    let parakeet = package("sherpa-onnx-nemo-parakeet-tdt-0.6b-v3-int8", &files);
    match detect(&parakeet, &DetectHints::default()).unwrap() {
        SherpaModel::Offline(c) => {
            assert_eq!(c.model_type.as_deref(), Some("nemo_transducer"));
            assert!(matches!(c.model, OfflineModel::Transducer { .. }));
        }
        other => panic!("{other:?}"),
    }

    let nemotron = package("sherpa-onnx-nemotron-3.5-asr-streaming-0.6b-160ms-int8-2026-06-11", &files);
    let m = detect(&nemotron, &DetectHints::default()).unwrap();
    assert_eq!(m.label(), "online/transducer");
    // `mode=offline` overrides the name.
    let hints = DetectHints {
        mode: Some("offline".into()),
        ..DetectHints::default()
    };
    assert_eq!(detect(&nemotron, &hints).unwrap().label(), "offline/transducer");
}

#[test]
fn int8_files_win_unless_fp32_is_preferred() {
    let root = package(
        "sherpa-onnx-streaming-zipformer-en-2023-06-26",
        &[
            "encoder-epoch-99-avg-1-chunk-16-left-128.onnx",
            "encoder-epoch-99-avg-1-chunk-16-left-128.int8.onnx",
            "decoder-epoch-99-avg-1-chunk-16-left-128.onnx",
            "joiner-epoch-99-avg-1-chunk-16-left-128.onnx",
            "joiner-epoch-99-avg-1-chunk-16-left-128.int8.onnx",
            "tokens.txt",
        ],
    );
    let SherpaModel::Online(c) = detect(&root, &DetectHints::default()).unwrap() else {
        panic!("streaming package");
    };
    let OnlineModel::Transducer { encoder, decoder, joiner } = &c.model else {
        panic!("transducer");
    };
    assert!(file_name(encoder).ends_with(".int8.onnx"));
    assert!(file_name(joiner).ends_with(".int8.onnx"));
    assert_eq!(file_name(decoder), "decoder-epoch-99-avg-1-chunk-16-left-128.onnx");
    let fp32 = DetectHints {
        prefer_fp32: true,
        ..DetectHints::default()
    };
    let SherpaModel::Online(c) = detect(&root, &fp32).unwrap() else {
        panic!()
    };
    let OnlineModel::Transducer { encoder, .. } = &c.model else { panic!() };
    assert_eq!(file_name(encoder), "encoder-epoch-99-avg-1-chunk-16-left-128.onnx");
}

#[test]
fn detects_qwen3_whisper_moonshine_and_language_id() {
    let q3 = package(
        "sherpa-onnx-qwen3-asr-0.6B-int8-2026-03-25",
        &["conv_frontend.onnx", "encoder.int8.onnx", "decoder.int8.onnx", "tokenizer/"],
    );
    let SherpaModel::Offline(c) = detect(&q3, &DetectHints::default()).unwrap() else {
        panic!()
    };
    assert!(matches!(&c.model, OfflineModel::Qwen3Asr { tokenizer, .. } if file_name(tokenizer) == "tokenizer"));
    assert!(matches!(
        c.model,
        OfflineModel::Qwen3Asr {
            max_total_len: Some(super::catalog::QWEN3_MAX_TOTAL_LEN),
            max_new_tokens: Some(super::catalog::QWEN3_MAX_NEW_TOKENS),
            ..
        }
    ));
    assert!(c.tokens.is_none());
    c.validate().unwrap();

    let whisper = package(
        "sherpa-onnx-whisper-tiny",
        &[
            "tiny-encoder.onnx",
            "tiny-decoder.onnx",
            "tiny-encoder.int8.onnx",
            "tiny-decoder.int8.onnx",
            "tiny-tokens.txt",
        ],
    );
    let SherpaModel::Offline(c) = detect(&whisper, &DetectHints::default()).unwrap() else {
        panic!()
    };
    assert!(matches!(&c.model, OfflineModel::Whisper { encoder, .. } if file_name(encoder) == "tiny-encoder.int8.onnx"));
    assert_eq!(file_name(c.tokens.as_ref().unwrap()), "tiny-tokens.txt");
    let lid = DetectHints {
        kind: Some("lid".into()),
        ..DetectHints::default()
    };
    assert!(matches!(detect(&whisper, &lid).unwrap(), SherpaModel::LanguageId(_)));

    let moonshine = package(
        "sherpa-onnx-moonshine-tiny-en-int8",
        &[
            "preprocess.onnx",
            "encode.int8.onnx",
            "uncached_decode.int8.onnx",
            "cached_decode.int8.onnx",
            "tokens.txt",
        ],
    );
    let SherpaModel::Offline(c) = detect(&moonshine, &DetectHints::default()).unwrap() else {
        panic!()
    };
    let OfflineModel::Moonshine {
        preprocessor,
        encoder,
        uncached_decoder,
        cached_decoder,
        merged_decoder,
    } = &c.model
    else {
        panic!()
    };
    assert_eq!(file_name(preprocessor.as_ref().unwrap()), "preprocess.onnx");
    assert_eq!(file_name(encoder), "encode.int8.onnx");
    assert_eq!(file_name(uncached_decoder.as_ref().unwrap()), "uncached_decode.int8.onnx");
    assert_eq!(file_name(cached_decoder.as_ref().unwrap()), "cached_decode.int8.onnx");
    assert!(merged_decoder.is_none());
}

#[test]
fn detects_tts_packages() {
    let kokoro = package(
        "kokoro-int8-multi-lang-v1_0",
        &[
            "model.int8.onnx",
            "voices.bin",
            "tokens.txt",
            "espeak-ng-data/",
            "dict/",
            "lexicon-gb-en.txt",
            "lexicon-us-en.txt",
            "lexicon-zh.txt",
            "date-zh.fst",
            "number-zh.fst",
            "phone-zh.fst",
        ],
    );
    let SherpaModel::Tts(c) = detect(&kokoro, &DetectHints::default()).unwrap() else {
        panic!()
    };
    let TtsModel::Kokoro { lexicons, data_dir, .. } = &c.model else {
        panic!()
    };
    let lex: Vec<String> = lexicons.iter().map(|p| file_name(p)).collect();
    assert_eq!(lex, ["lexicon-us-en.txt", "lexicon-zh.txt"], "US English replaces British, English first");
    assert!(data_dir.is_some());
    let fsts: Vec<String> = c.rule_fsts.iter().map(|p| file_name(p)).collect();
    assert_eq!(fsts, ["phone-zh.fst", "date-zh.fst", "number-zh.fst"]);
    c.validate().unwrap();

    let kitten = package(
        "kitten-nano-en-v0_8-int8",
        &["model.int8.onnx", "voices.bin", "tokens.txt", "espeak-ng-data/"],
    );
    assert_eq!(detect(&kitten, &DetectHints::default()).unwrap().label(), "tts/kitten");

    let piper = package("vits-piper-en_US-amy-low", &["en_US-amy-low.onnx", "tokens.txt", "espeak-ng-data/"]);
    let SherpaModel::Tts(c) = detect(&piper, &DetectHints::default()).unwrap() else {
        panic!()
    };
    assert!(matches!(&c.model, TtsModel::Vits { model, data_dir: Some(_), .. } if file_name(model) == "en_US-amy-low.onnx"));
}

#[test]
fn detects_speaker_models_and_asks_for_a_kind_otherwise() {
    let root = package("speaker", &["3dspeaker_speech_campplus_sv_zh-cn_16k-common.onnx"]);
    let file = root.join("3dspeaker_speech_campplus_sv_zh-cn_16k-common.onnx");
    assert_eq!(detect(&file, &DetectHints::default()).unwrap().label(), "speaker");
    assert_eq!(detect(&root, &DetectHints::default()).unwrap().label(), "speaker");

    let unknown = package("my-model", &["model.onnx", "tokens.txt"]);
    let e = detect(&unknown, &DetectHints::default()).unwrap_err();
    assert!(e.contains("kind="), "{e}");
    let sv = DetectHints {
        kind: Some("SenseVoice".into()),
        ..DetectHints::default()
    };
    assert_eq!(detect(&unknown, &sv).unwrap().label(), "offline/sense_voice");
    assert!(detect(&unknown.join("nope"), &DetectHints::default()).is_err());
}

/// `validate_in_child` hands the child the exact config as JSON (`config=<file>`).
#[test]
fn detected_models_round_trip_through_json_for_the_child_check() {
    let files = ["encoder.int8.onnx", "decoder.int8.onnx", "joiner.int8.onnx", "tokens.txt"];
    let pk = package("sherpa-onnx-nemo-parakeet-tdt-0.6b-v3-int8", &files);
    let hints = DetectHints {
        prefer_fp32: true,
        ..DetectHints::default()
    };
    let mut model = detect(&pk, &hints).unwrap();
    if let SherpaModel::Offline(c) = &mut model {
        c.num_threads = Some(3);
        c.hotwords_score = 2.5;
    }
    let dir = tmp_dir("child-config");
    let file = dir.join("config.json");
    std::fs::write(&file, serde_json::to_string(&model).unwrap()).unwrap();
    assert_eq!(read_model_config(&file).unwrap(), model);
    model.validate().unwrap();

    let kokoro = package(
        "kokoro-int8-multi-lang-v1_0",
        &[
            "model.int8.onnx",
            "voices.bin",
            "tokens.txt",
            "espeak-ng-data/",
            "lexicon-us-en.txt",
            "date-zh.fst",
        ],
    );
    let tts = detect(&kokoro, &DetectHints::default()).unwrap();
    let back: SherpaModel = serde_json::from_value(serde_json::to_value(&tts).unwrap()).unwrap();
    assert_eq!(back, tts);

    std::fs::write(&file, b"{\"task\":\"offline\"}").unwrap();
    assert!(read_model_config(&file)
        .unwrap_err()
        .contains("not a sherpa model config"));
    assert!(read_model_config(&dir.join("missing.json")).is_err());
    // A config the in-process load would refuse is refused before any child starts.
    let mut bad = model.clone();
    if let SherpaModel::Offline(c) = &mut bad {
        c.decoding = Decoding::ModifiedBeamSearch { max_active_paths: 4 };
        c.model = OfflineModel::SenseVoice {
            model: pk.join("encoder.int8.onnx"),
            language: None,
            use_itn: true,
        };
    }
    assert!(super::probe::validate_in_child(&bad, std::time::Duration::from_secs(1))
        .unwrap_err()
        .contains("need a transducer model"));
}

#[test]
fn configs_round_trip_through_json() {
    // P2 stores these in catalog metadata / settings; the tags are part of that format.
    let json = serde_json::json!({
        "model": { "family": "sense_voice", "model": "C:/m/model.int8.onnx", "language": "ja" },
        "tokens": "C:/m/tokens.txt",
        "num_threads": 1
    });
    let c: OfflineRecognizerConfig = serde_json::from_value(json).unwrap();
    assert!(matches!(&c.model, OfflineModel::SenseVoice { use_itn: true, language: Some(l), .. } if l == "ja"));
    assert_eq!(c.decoding, Decoding::GreedySearch);
    let back: OfflineRecognizerConfig = serde_json::from_value(serde_json::to_value(&c).unwrap()).unwrap();
    assert_eq!(back, c);
    let t: TtsConfig = serde_json::from_value(serde_json::json!({
        "model": { "family": "kitten", "model": "a.onnx", "voices": "v.bin", "tokens": "t.txt" }
    }))
    .unwrap();
    assert_eq!(t.model.family(), "kitten");
}

// ---------------------------------------------------------------------------------------------
// Probe helpers
// ---------------------------------------------------------------------------------------------

#[test]
fn probe_extras_parse_key_values() {
    let e = Extra::parse(&["kind=sense_voice".into(), "repeat=5".into(), "load_only=1".into(), "lang=".into()]).unwrap();
    assert_eq!(e.str("kind"), Some("sense_voice"));
    assert_eq!(e.num::<usize>("repeat").unwrap(), Some(5));
    assert!(e.flag("load_only"));
    assert_eq!(e.str("lang"), None, "empty values are unset");
    assert!(Extra::parse(&["oops".into()]).is_err());
    let c = Extra::parse(&["config=C:\\Temp\\a=b.json".into()]).unwrap();
    assert_eq!(c.str("config"), Some("C:\\Temp\\a=b.json"), "only the first `=` splits");
    assert!(Extra::parse(&["repeat=x".into()])
        .unwrap()
        .num::<usize>("repeat")
        .is_err());
}

#[test]
fn wav_reader_downmixes_and_scales() {
    let dir = tmp_dir("wav");
    let path = dir.join("stereo16.wav");
    let spec = hound::WavSpec {
        channels: 2,
        sample_rate: 22_050,
        bits_per_sample: 16,
        sample_format: hound::SampleFormat::Int,
    };
    let mut w = hound::WavWriter::create(&path, spec).unwrap();
    for (l, r) in [(16384i16, 0i16), (-32768, -32768), (0, 8192)] {
        w.write_sample(l).unwrap();
        w.write_sample(r).unwrap();
    }
    w.finalize().unwrap();
    let (mono, rate) = read_wav(&path).unwrap();
    assert_eq!(rate, 22_050);
    assert_eq!(mono.len(), 3);
    assert!((mono[0] - 0.25).abs() < 1e-4 && (mono[1] + 1.0).abs() < 1e-4 && (mono[2] - 0.125).abs() < 1e-4);

    let float_path = dir.join("mono32f.wav");
    write_wav(&float_path, &[0.5, -0.5], 24_000).unwrap();
    let (mono, rate) = read_wav(&float_path).unwrap();
    assert_eq!((mono, rate), (vec![0.5, -0.5], 24_000));
    assert!(read_wav(&dir.join("missing.wav")).is_err());
}

// ---------------------------------------------------------------------------------------------
// ONNX summary and stream padding (P2-B)
// ---------------------------------------------------------------------------------------------

/// `graph` serialized with `metadata_props` appended (field 14, as exporters write it).
fn model_with_metadata(graph: &Graph, metadata: &[(&str, &str)]) -> Vec<u8> {
    fn varint(out: &mut Vec<u8>, mut v: u64) {
        while v >= 0x80 {
            out.push((v as u8) | 0x80);
            v >>= 7;
        }
        out.push(v as u8);
    }
    let mut bytes = graph.model_bytes(17);
    for (k, v) in metadata {
        let mut entry = vec![0x0a];
        varint(&mut entry, k.len() as u64);
        entry.extend_from_slice(k.as_bytes());
        entry.push(0x12);
        varint(&mut entry, v.len() as u64);
        entry.extend_from_slice(v.as_bytes());
        bytes.push(0x72);
        varint(&mut bytes, entry.len() as u64);
        bytes.extend_from_slice(&entry);
    }
    bytes
}

#[test]
fn onnx_summary_reads_inputs_outputs_and_metadata() {
    use super::onnx_info::summary;
    let dir = tmp_dir("onnx-summary");
    let mut g = Graph::new();
    g.input("audio_signal", FLOAT, &[Dim::Named("N"), Dim::Fixed(128), Dim::Named("T")])
        .input("length", INT64, &[Dim::Named("N")])
        .input("prompt_index", INT64, &[Dim::Named("N")])
        .output("outputs", FLOAT, &[Dim::Named("N")])
        .output("encoded_lengths", INT64, &[Dim::Named("N")]);
    let long = "x".repeat(300_000);
    let p = dir.join("encoder.onnx");
    std::fs::write(
        &p,
        model_with_metadata(&g, &[("window_size", "65"), ("prompt_dictionary", r#"{"en": 0}"#), ("big", &long)]),
    )
    .unwrap();
    let s = summary(&p).unwrap();
    assert_eq!(s.inputs, vec!["audio_signal", "length", "prompt_index"]);
    assert_eq!(s.outputs, 2);
    assert!(s.has_input("prompt_index") && !s.has_input("x"));
    assert_eq!(s.meta("window_size"), Some("65"));
    assert_eq!(s.meta("prompt_dictionary"), Some(r#"{"en": 0}"#));
    assert_eq!(s.meta("big").map(str::len), Some(300_000), "values up to 1 MiB are read");
    assert_eq!(s.meta("missing"), None);
    // A graph without metadata; a truncated file is an error, not a panic.
    let plain = dir.join("plain.onnx");
    std::fs::write(&plain, g.model_bytes(17)).unwrap();
    assert!(summary(&plain).unwrap().metadata.is_empty());
    let full = std::fs::read(&p).unwrap();
    std::fs::write(dir.join("cut.onnx"), &full[..full.len() - 10]).unwrap();
    assert!(summary(&dir.join("cut.onnx")).is_err());
}

#[test]
fn stream_padding_follows_the_nemo_decode_window() {
    use super::online::{window_padding_ms, StreamPadding, DEFAULT_TAIL_PADDING_MS, NEMO_HEAD_PADDING_MS};
    use super::onnx_info::OnnxSummary;
    // Nemotron 3.5: 160 ms chunks decode 25-frame windows, 560 ms chunks 65-frame windows.
    assert_eq!(window_padding_ms(25), 280);
    assert_eq!(window_padding_ms(65), 680);
    let nemo = OnnxSummary {
        metadata: vec![("window_size".into(), "65".into())],
        ..OnnxSummary::default()
    };
    let p = StreamPadding::for_encoder(Some(&nemo));
    assert_eq!(p.head_ms, NEMO_HEAD_PADDING_MS);
    assert!(p.tail_ms > 560, "a whole chunk plus its lookahead");
    for other in [None, Some(&OnnxSummary::default())] {
        let p = StreamPadding::for_encoder(other);
        assert_eq!((p.head_ms, p.tail_ms), (0, DEFAULT_TAIL_PADDING_MS));
    }
    let bogus = OnnxSummary {
        metadata: vec![("window_size".into(), "0".into())],
        ..OnnxSummary::default()
    };
    assert_eq!(StreamPadding::for_encoder(Some(&bogus)).tail_ms, DEFAULT_TAIL_PADDING_MS);
}

// ---------------------------------------------------------------------------------------------
// Real runtime and models (ignored; see the module docs)
// ---------------------------------------------------------------------------------------------

fn models_root() -> Option<PathBuf> {
    std::env::var_os("SIGIL_TEST_MODELS_DIR").map(PathBuf::from)
}

fn real_runtime() -> Option<Arc<SherpaRuntime>> {
    let dir = std::env::var_os(super::lib::RUNTIME_DIR_ENV)
        .map(PathBuf::from)
        .or_else(|| models_root().map(|r| r.join("sherpa-onnx").join("pack-1.13.8")));
    let Some(dir) = dir.filter(|d| d.join(C_API_DLL).is_file()) else {
        eprintln!("no sherpa runtime pack (SIGIL_TEST_MODELS_DIR/sherpa-onnx/pack-1.13.8); skipped");
        return None;
    };
    Some(SherpaRuntime::load(&dir).expect("the pinned sherpa-onnx pack loads"))
}

fn real_model(name: &str) -> Option<PathBuf> {
    let p = models_root()?.join("sherpa-onnx").join("models").join(name);
    if p.is_dir() {
        Some(p)
    } else {
        eprintln!("{} is missing; skipped", p.display());
        None
    }
}

// The original FunAudioLLM SenseVoiceSmall (the 2025-09-09 package is the WSYue Cantonese fine-tune).
const SENSE_VOICE: &str = "sherpa-onnx-sense-voice-zh-en-ja-ko-yue-int8-2024-07-17";
const KOKORO: &str = "kokoro-int8-multi-lang-v1_0";

fn sense_voice(rt: &Arc<SherpaRuntime>) -> Option<(OfflineRecognizer, PathBuf)> {
    let dir = real_model(SENSE_VOICE)?;
    let SherpaModel::Offline(cfg) = detect(&dir, &DetectHints::default()).unwrap() else {
        panic!("SenseVoice is an offline model")
    };
    Some((OfflineRecognizer::new(rt, &cfg).expect("SenseVoice loads"), dir))
}

#[test]
#[ignore = "needs SIGIL_TEST_MODELS_DIR with the sherpa-onnx 1.13.8 pack"]
fn real_runtime_loads_once_and_binds_its_own_onnxruntime() {
    let _g = dll_lock();
    let Some(rt) = real_runtime() else { return };
    let info = rt.info();
    assert_eq!(info.version, EXPECTED_VERSION);
    assert!(info.onnxruntime_version.starts_with("1.28."), "{}", info.onnxruntime_version);
    let module = info.onnxruntime_module.as_deref().expect("module path");
    assert!(
        Path::new(module).starts_with(&info.dir) || module.to_lowercase().starts_with(&info.dir.to_lowercase()),
        "{module} must come from {}",
        info.dir
    );
    // Sigil's statically linked runtime is a different, independent copy.
    assert!(crate::services::neural_audio::ort_session::runtime_version().starts_with("1.28."));
    // Cached for the process: loading again (from anywhere) returns the same runtime.
    let again = SherpaRuntime::load(Path::new(&info.dir)).unwrap();
    assert!(Arc::ptr_eq(&rt, &again));
    let elsewhere = SherpaRuntime::load(&tmp_dir("other-pack")).unwrap();
    assert!(Arc::ptr_eq(&rt, &elsewhere));
}

#[test]
#[ignore = "needs SIGIL_TEST_MODELS_DIR with the sherpa-onnx pack and the SenseVoice int8 package"]
fn real_sense_voice_transcribes_english_and_japanese() {
    let _g = dll_lock();
    let Some(rt) = real_runtime() else { return };
    let Some((rec, dir)) = sense_voice(&rt) else { return };
    let (samples, rate) = read_wav(&dir.join("test_wavs").join("en.wav")).unwrap();
    let r = rec
        .transcribe(rate, &samples, &DecodeOptions::default())
        .unwrap();
    eprintln!("en: {r:?}");
    assert_eq!(r.lang.as_deref(), Some("en"), "{r:?}");
    let words = r.text.to_lowercase();
    assert!(words.contains("chieftain") || words.contains("gold"), "{}", r.text);
    // Same audio, same answer (greedy decoding, a new stream each call).
    assert_eq!(
        rec.transcribe(rate, &samples, &DecodeOptions::default())
            .unwrap()
            .text,
        r.text
    );
    // Degenerate input never reaches a model that could throw inside ONNX Runtime.
    let tiny = rec
        .transcribe(16_000, &[0.0; 800], &DecodeOptions::default())
        .unwrap();
    assert!(tiny.text.is_empty() && tiny.lang.is_none());
    let silence = rec
        .transcribe(16_000, &[0.0; 4_800], &DecodeOptions::default())
        .unwrap();
    assert_eq!(silence.text, "", "SenseVoice's lone `。` on silence is dropped");
    // Per-stream language option (informational: which families read it is recorded in ADR 0002).
    let hint = DecodeOptions {
        language: Some("zh".into()),
        hotwords: None,
    };
    let hinted = rec.transcribe(rate, &samples, &hint).unwrap();
    eprintln!("en.wav with a per-stream language=zh hint: {} / {:?}", hinted.text, hinted.lang);
    // Hotwords on SenseVoice: sherpa would exit(-1) ("Only transducer models support contextual
    // biasing"); the wrapper refuses first.
    let hot = DecodeOptions {
        language: None,
        hotwords: Some("Sigil".into()),
    };
    assert!(rec.transcribe(rate, &samples, &hot).is_err());
    assert!(rec.create_stream_with_hotwords("Sigil").is_err());
    // Streams used directly: no audio, or 25 ms, never reaches the model.
    let mut empty = rec.create_stream().unwrap();
    assert!(!empty.decode());
    assert_eq!(empty.result().unwrap(), OfflineResult::default());
    let mut short = rec.create_stream().unwrap();
    short.accept_waveform(16_000, &[0.0; 400]).unwrap();
    assert!(!short.decode());
    assert!(short.result().unwrap().text.is_empty());
    let mut full = rec.create_stream().unwrap();
    full.accept_waveform(rate, &samples).unwrap();
    assert!(full.decode());
    assert_eq!(full.result().unwrap().text, r.text);

    let (samples, rate) = read_wav(&dir.join("test_wavs").join("ja.wav")).unwrap();
    let r = rec
        .transcribe(rate, &samples, &DecodeOptions::default())
        .unwrap();
    eprintln!("ja: {r:?}");
    assert_eq!(r.lang.as_deref(), Some("ja"), "{r:?}");
    assert!(!r.text.is_empty());

    // Streams can be moved to other threads (Send) and the recognizer shared (Sync).
    let shared = Arc::new(rec);
    let handles: Vec<_> = (0..2)
        .map(|_| {
            let (rec, samples) = (shared.clone(), samples.clone());
            std::thread::spawn(move || {
                rec.transcribe(rate, &samples, &DecodeOptions::default())
                    .unwrap()
                    .text
            })
        })
        .collect();
    for h in handles {
        assert_eq!(h.join().unwrap(), r.text);
    }
}

#[test]
#[ignore = "needs SIGIL_TEST_MODELS_DIR with the sherpa-onnx pack and the Kokoro int8 multi-lang v1.0 package"]
fn real_kokoro_speaks_streams_chunks_and_cancels() {
    let _g = dll_lock();
    let Some(rt) = real_runtime() else { return };
    let Some(dir) = real_model(KOKORO) else { return };
    let SherpaModel::Tts(cfg) = detect(&dir, &DetectHints::default()).unwrap() else {
        panic!()
    };
    let tts = OfflineTts::new(&rt, &cfg).expect("Kokoro loads");
    assert_eq!(tts.sample_rate(), 24_000);
    assert!(tts.num_speakers() >= 50, "{}", tts.num_speakers());

    let text = "Hello from Sigil. This sentence checks the second chunk. And a third one for good measure.";
    let mut chunks = 0usize;
    let mut cb = |c: &[f32]| {
        assert!(!c.is_empty());
        chunks += 1;
        true
    };
    let full = tts
        .generate(text, &GenerateOptions::default(), None, Some(&mut cb))
        .unwrap();
    assert!(!full.cancelled);
    assert!(chunks >= 2, "one callback per sentence batch, got {chunks}");
    let secs = full.samples.len() as f32 / full.sample_rate as f32;
    let rms = (full.samples.iter().map(|s| s * s).sum::<f32>() / full.samples.len() as f32).sqrt();
    assert!(secs > 2.0 && rms > 0.01, "{secs} s, rms {rms}");

    // Cancel after the first sentence: generation stops early and keeps what was made.
    let cancel = AtomicBool::new(false);
    let mut first = |_: &[f32]| {
        cancel.store(true, Ordering::Relaxed);
        true
    };
    let part = tts
        .generate(text, &GenerateOptions::default(), Some(&cancel), Some(&mut first))
        .unwrap();
    assert!(part.cancelled);
    assert!(
        part.samples.len() < full.samples.len(),
        "{} vs {}",
        part.samples.len(),
        full.samples.len()
    );

    // A flag already set (a skipped utterance) returns at once, without synthesizing a batch.
    let pre = AtomicBool::new(true);
    let t0 = std::time::Instant::now();
    let skipped = tts
        .generate(text, &GenerateOptions::default(), Some(&pre), None)
        .unwrap();
    assert!(skipped.cancelled && skipped.samples.is_empty());
    assert!(t0.elapsed() < std::time::Duration::from_millis(200), "{:?}", t0.elapsed());
    // "No pause between sentences" (the 0.01 floor) is shorter than the default 0.2 pause; a
    // scale sherpa does not apply (an exact 0 or a tiny epsilon) would come out longer.
    let tight = tts
        .generate(
            text,
            &GenerateOptions {
                silence_scale: Some(0.0),
                ..GenerateOptions::default()
            },
            None,
            None,
        )
        .unwrap();
    assert!(
        tight.samples.len() < full.samples.len(),
        "{} vs {}",
        tight.samples.len(),
        full.samples.len()
    );

    // A panicking callback is reported, never unwound into C.
    let mut boom = |_: &[f32]| -> bool { panic!("callback bug") };
    assert!(tts
        .generate("Short.", &GenerateOptions::default(), None, Some(&mut boom))
        .is_err());
    assert!(tts
        .generate(
            "Short.",
            &GenerateOptions {
                sid: 9999,
                ..GenerateOptions::default()
            },
            None,
            None
        )
        .is_err());
}

/// Acceptance: no ORT conflict with Sigil's CUDA sessions. Sigil's statically linked ONNX
/// Runtime keeps running its EP canary (CUDA when `SIGIL_TEST_CUDA_BIN` is set, else CPU) in a
/// loop on another thread while sherpa's own ORT decodes; both must keep producing correct
/// results, and the transcript must equal the one decoded without the load.
#[test]
#[ignore = "needs SIGIL_TEST_MODELS_DIR with the sherpa-onnx pack and SenseVoice; SIGIL_TEST_CUDA_BIN for CUDA"]
fn real_runtime_coexists_with_sigil_cuda_sessions() {
    use crate::services::diagnostics::canary;
    let _g = dll_lock();
    let ep = if crate::services::neural_audio::ort_session::preload_test_cuda_runtime() {
        "cuda"
    } else {
        "cpu"
    };
    if let Err(e) = canary::run(ep, 0) {
        panic!("Sigil's {ep} canary fails before sherpa is involved: {e}");
    }
    let Some(rt) = real_runtime() else { return };
    let Some((rec, dir)) = sense_voice(&rt) else { return };
    let (samples, rate) = read_wav(&dir.join("test_wavs").join("en.wav")).unwrap();
    let reference = rec
        .transcribe(rate, &samples, &DecodeOptions::default())
        .unwrap()
        .text;

    let stop = Arc::new(AtomicBool::new(false));
    let runs = Arc::new(AtomicUsize::new(0));
    let (stop2, runs2) = (stop.clone(), runs.clone());
    let sigil = std::thread::spawn(move || -> Result<(), String> {
        while !stop2.load(Ordering::Relaxed) {
            canary::run(ep, 0)?;
            runs2.fetch_add(1, Ordering::Relaxed);
        }
        Ok(())
    });
    for _ in 0..3 {
        let text = rec
            .transcribe(rate, &samples, &DecodeOptions::default())
            .unwrap()
            .text;
        assert_eq!(text, reference, "sherpa output changed while Sigil's {ep} sessions ran");
    }
    stop.store(true, Ordering::Relaxed);
    sigil
        .join()
        .unwrap()
        .unwrap_or_else(|e| panic!("Sigil's {ep} canary failed next to sherpa: {e}"));
    assert!(runs.load(Ordering::Relaxed) > 0);
    canary::run(ep, 0).expect("Sigil's canary still passes after sherpa decoded");
    canary::run("cpu", 0).expect("Sigil's CPU canary still passes");
    eprintln!("sigil {ep} canary runs during sherpa decoding: {}", runs.load(Ordering::Relaxed));
}
