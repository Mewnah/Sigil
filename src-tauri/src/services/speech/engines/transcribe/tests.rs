//! Tests of the transcribe.cpp engine adapter (P2-D).
//!
//! Always run (no DLL, no model, no network): option parsing, the family rules (Cohere never for
//! ja / zh / ko / ru), language decisions, long-utterance splitting and joining, Whisper prompts,
//! capabilities, the wrapper options the governor's admission produces, error mapping and the
//! worker hand-over.
//!
//! Ignored (real runtime + models): `cargo test --manifest-path src-tauri/Cargo.toml
//! speech::engines::transcribe -- --ignored --test-threads=1` with `SIGIL_APP_DATA_DIR` = a model
//! store that has installed (`Sigil.exe --headless fetch`, e.g. from local assets):
//! - `runtime/sigil-transcribe-cuda13` + `runtime/nvidia-cuda13` (GPU tests) and a CPU pack
//!   (`runtime/sigil-transcribe-cpu-avx2` / `-cpu-avx512`);
//! - `stt/qwen3-asr-1.7b-q8_0` and `bench/transcribe-whisper-tiny-q8_0`;
//! - optionally `stt/voxtral-mini-4b-realtime-q4_k_m`, `stt/cohere-transcribe-q4_k`,
//!   `stt/granite-speech-5.0-turboctc-q8_0`, `stt/qwen3-asr-1.7b-bf16` (their tests skip with a
//!   message when missing);
//!
//! and `SIGIL_TEST_MODELS_DIR` with transcribe.cpp's samples in `models/transcribe/samples/`
//! (`jfk.wav`, `ja-short.wav`, `ja-long.wav`). The worker test also needs a debug `sigil.exe` built
//! from this tree (`SIGIL_STT_WORKER_EXE`, default `target/debug/sigil.exe`). A missing required
//! input FAILS the test unless `SIGIL_TEST_ALLOW_SKIP=1`.

use super::*;
use crate::services::resources::{self, Subsystem};
use crate::services::speech::engine_api::{EngineKind, StatusSink};
use crate::services::speech::two_pass::{FinalJob, TwoPass, TwoPassConfig, TwoPassOutcome};
use crate::services::speech_rt::transcribe::{Capabilities, DeviceInfo, Features};
use std::path::PathBuf;
use std::sync::Mutex;
use std::time::Duration;

// ---------------------------------------------------------------------------------------------
// Options
// ---------------------------------------------------------------------------------------------

#[test]
fn options_parse_leniently_and_clamp() {
    let o = Options::parse(&Value::Null);
    assert_eq!(o.language, None);
    assert!(o.cpu_fallback && o.warmup && !o.append_only);
    assert_eq!(o.max_segment_seconds, GPU_WARMUP_SECONDS);
    assert_eq!((o.punctuation, o.itn), (Toggle::Default, Toggle::Default));
    assert!(o.admission.is_none());

    let o = Options::parse(&serde_json::json!({
        "language": "JA-jp",
        "device": 1,
        "deviceId": " 0000:01:00.0 ",
        "cpuFallback": false,
        "minFreeVramMb": 100000,
        "nCtx": 100,
        "warmup": false,
        "warmupSeconds": 99,
        "maxSegmentSeconds": 2,
        "pnc": "off",
        "itn": "ON",
        "noSpeechThreshold": 3,
        "commit": "stable_prefix",
        "agreement": 40,
        "appendOnly": true,
        "isolation": "worker",
        "workerMemoryLimitMb": 10,
    }));
    assert_eq!(o.language.as_deref(), Some("ja"));
    assert_eq!((o.device, o.device_id.as_deref()), (Some(1), Some("0000:01:00.0")));
    assert!(!o.cpu_fallback && !o.warmup && o.append_only);
    assert_eq!(o.min_free_vram_mb, Some(16_384));
    assert_eq!(o.n_ctx, None, "below the 256-token floor: the model's own context");
    assert_eq!(o.warmup_seconds, Some(30.0));
    assert_eq!(o.max_segment_seconds, 5.0);
    assert_eq!((o.punctuation, o.itn), (Toggle::Off, Toggle::On));
    assert_eq!(o.no_speech_threshold, Some(1.0));
    assert_eq!((o.commit.as_deref(), o.agreement), (Some("stable_prefix"), Some(16)));
    assert_eq!(o.isolation, Isolation::Worker);
    assert_eq!(o.worker_memory_limit_mb, Some(1024));

    // Wrong types and unknown values are ignored, never an error.
    let o = Options::parse(&serde_json::json!({
        "language": 5, "device": -1, "cpuFallback": "no", "commit": "whenever", "isolation": "sandbox",
        "maxSegmentSeconds": "long", "nCtx": 8192.7,
    }));
    assert_eq!(o.language, None);
    assert_eq!(o.device, None);
    assert!(o.cpu_fallback);
    assert_eq!(o.commit, None);
    assert_eq!(o.max_segment_seconds, GPU_WARMUP_SECONDS);
    assert_eq!(o.n_ctx, Some(8192));
    assert_eq!(Isolation::parse("in-process"), Some(Isolation::InProcess));
    assert_eq!(Isolation::parse("Worker_Process"), Some(Isolation::Worker));
    assert_eq!(Isolation::parse("sandbox"), None);
}

#[test]
fn the_parents_admission_travels_to_the_worker() {
    let a = Admission {
        gpu: Some(RuntimeVariant::Cuda13),
        code: governor::codes::FITS,
        reason: None,
        need_mb: 3500,
        available_mb: 5700,
    };
    let fwd = ForwardedAdmission::from_admission(&a);
    let o = Options::parse(&serde_json::json!({ ADMISSION_KEY: fwd }));
    let back = o.admission.unwrap().to_admission();
    assert_eq!((back.gpu, back.code), (Some(RuntimeVariant::Cuda13), governor::codes::FITS));

    let denied = Admission::forced_cpu(governor::codes::WORKER_CRASHED, "Qwen3 runs on the CPU: crashes");
    let back = ForwardedAdmission::from_admission(&denied).to_admission();
    assert_eq!((back.gpu, back.code), (None, governor::codes::WORKER_CRASHED));
    assert_eq!(back.reason.as_deref(), Some("Qwen3 runs on the CPU: crashes"));
    // Every governor code survives; anything else (or a CPU variant as "GPU") is a CPU request.
    for code in [
        governor::codes::NO_GPU,
        governor::codes::GPU_UNKNOWN,
        governor::codes::INTEGRATED_GPU,
        governor::codes::SMALL_GPU,
        governor::codes::NO_GPU_RUNTIME,
        governor::codes::VC_RUNNING,
        governor::codes::GPU_FULL,
    ] {
        let f = ForwardedAdmission {
            gpu: None,
            code: code.into(),
            reason: None,
        };
        assert_eq!(f.to_admission().code, code);
    }
    let odd = ForwardedAdmission {
        gpu: Some("cpu-avx2".into()),
        code: "made-up".into(),
        reason: None,
    };
    let back = odd.to_admission();
    assert_eq!((back.gpu, back.code), (None, governor::codes::CPU_REQUESTED));
}

// ---------------------------------------------------------------------------------------------
// Family rules and languages
// ---------------------------------------------------------------------------------------------

fn langs(v: &[&str]) -> Vec<String> {
    v.iter().map(|s| s.to_string()).collect()
}

#[test]
fn cohere_is_never_used_for_cjk_or_russian_and_always_gets_a_language() {
    let cohere = langs(&["en", "fr", "de", "es", "it", "pt", "nl", "pl", "el", "ar", "vi"]);
    for l in ["ja", "zh", "ko", "ru", "yue", "zh-TW", "ja-JP"] {
        let e = decode_language(Family::Cohere, "Cohere Transcribe", Some(l), None, &cohere, false).unwrap_err();
        assert_eq!(e.kind, AsrErrorKind::Unsupported, "{l}");
        assert!(e.message.contains("never used"), "{e}");
        // Also as the engine's default language.
        assert!(decode_language(Family::Cohere, "Cohere Transcribe", None, Some(l), &cohere, false).is_err());
    }
    // Even when a future model build lists Japanese.
    let with_ja = langs(&["en", "ja"]);
    assert!(decode_language(Family::Cohere, "c", Some("ja"), None, &with_ja, false).is_err());
    // No language, and no language ID: refused with a hint to choose one.
    let e = decode_language(Family::Cohere, "Cohere Transcribe", None, None, &cohere, false).unwrap_err();
    assert_eq!(e.kind, AsrErrorKind::InvalidInput);
    assert!(e.message.contains("choose the language"), "{e}");
    // A European language passes; the request wins over the engine default.
    assert_eq!(
        decode_language(Family::Cohere, "c", Some("de-AT"), Some("fr"), &cohere, false).unwrap(),
        (Some("de".into()), false)
    );
    assert_eq!(
        decode_language(Family::Cohere, "c", None, Some("fr"), &cohere, false).unwrap(),
        (Some("fr".into()), false)
    );
    // A language Cohere does not know is refused, not guessed.
    assert_eq!(
        decode_language(Family::Cohere, "c", Some("fi"), None, &cohere, false)
            .unwrap_err()
            .kind,
        AsrErrorKind::Unsupported
    );
}

#[test]
fn detecting_models_drop_unknown_hints_and_single_language_models_need_none() {
    let qwen = langs(&["zh", "en", "yue", "ja", "ko", "ru"]);
    assert_eq!(
        decode_language(Family::Qwen3Asr, "q", Some("ja"), None, &qwen, true).unwrap(),
        (Some("ja".into()), false)
    );
    assert_eq!(decode_language(Family::Qwen3Asr, "q", None, None, &qwen, true).unwrap(), (None, false));
    assert_eq!(
        decode_language(Family::Qwen3Asr, "q", Some("auto"), None, &qwen, true).unwrap(),
        (None, false),
        "auto = detect"
    );
    // A hint the model does not list: it detects instead (and says so once).
    assert_eq!(decode_language(Family::Qwen3Asr, "q", Some("sw"), None, &qwen, true).unwrap(), (None, true));
    // English-only Granite: no hint needed, anything else refused.
    let en = langs(&["en"]);
    assert_eq!(
        decode_language(Family::Granite, "g", None, None, &en, false).unwrap(),
        (Some("en".into()), false)
    );
    assert_eq!(
        decode_language(Family::Granite, "g", Some("ja"), None, &en, false)
            .unwrap_err()
            .kind,
        AsrErrorKind::Unsupported
    );
    // Unknown language list: any hint is passed through.
    assert_eq!(
        decode_language(Family::Other, "m", Some("tr"), None, &[], false).unwrap(),
        (Some("tr".into()), false)
    );
    // A non-detecting multilingual model needs a language.
    assert_eq!(
        decode_language(Family::Other, "m", None, None, &langs(&["en", "de"]), false)
            .unwrap_err()
            .kind,
        AsrErrorKind::InvalidInput
    );
}

fn model_info(arch: &str, languages: &[&str], detect: bool, streaming: bool, prompt: bool) -> ModelInfo {
    ModelInfo {
        arch: Some(arch.into()),
        backend: "CUDA0".into(),
        device: Some(DeviceInfo {
            kind: Some("cuda".into()),
            ..DeviceInfo::default()
        }),
        capabilities: Capabilities {
            languages: languages.iter().map(|l| Some(l.to_string())).collect(),
            language_detect: detect,
            streaming,
            features: Features {
                initial_prompt: prompt,
                ..Features::default()
            },
            ..Capabilities::default()
        },
        ..ModelInfo::default()
    }
}

#[test]
fn capabilities_follow_the_family_rules() {
    // transcribe.cpp lists ja / zh / ko for Cohere; Sigil never offers them.
    let cohere = model_info("cohere_asr", &["en", "fr", "de", "ja", "zh", "ko", "vi"], false, false, false);
    let caps = caps_of(&cohere, Family::Cohere);
    assert_eq!(caps.languages, langs(&["en", "fr", "de", "vi"]));
    assert!(caps.needs_language && !caps.detects_language && !caps.streaming && caps.utterance);
    assert!(!caps.hotwords && !caps.prompt && !caps.tags);

    let qwen = model_info("qwen3_asr", &["zh", "en", "yue", "ja", "fil"], true, false, false);
    let caps = caps_of(&qwen, Family::Qwen3Asr);
    assert!(caps.detects_language && caps.tags && !caps.needs_language);
    assert!(caps.languages.contains(&"fil".to_string()) && caps.languages.contains(&"yue".to_string()));
    assert!(!caps.hotwords, "no context biasing in transcribe.cpp 0.2.4");

    let voxtral = model_info("voxtral_realtime", &["en", "ja"], true, true, false);
    assert!(caps_of(&voxtral, Family::VoxtralRealtime).streaming);

    let whisper = model_info("whisper", &["en", "ja"], true, false, true);
    let caps = caps_of(&whisper, Family::Whisper);
    assert!(caps.prompt && caps.hotwords);

    assert_eq!(device_of(&qwen), "cuda");
    let mut cpu = qwen.clone();
    cpu.device = Some(DeviceInfo {
        kind: Some("cpu".into()),
        ..DeviceInfo::default()
    });
    assert_eq!(device_of(&cpu), "cpu");
}

// ---------------------------------------------------------------------------------------------
// Long utterances, joining, prompts
// ---------------------------------------------------------------------------------------------

#[test]
fn long_utterances_split_at_their_quietest_point_and_cover_everything() {
    let sr = SAMPLE_RATE as usize;
    // 28 s of "speech" with a silent gap at 16.0..16.2 s: the 20 s piece is cut in the gap.
    let mut audio: Vec<f32> = (0..28 * sr).map(|i| ((i as f32) * 0.05).sin() * 0.3).collect();
    for s in &mut audio[16 * sr..16 * sr + sr / 5] {
        *s = 0.0;
    }
    let pieces = split_points(&audio, 20 * sr);
    assert_eq!(pieces.len(), 2, "{pieces:?}");
    let cut = pieces[0].end;
    assert!((16 * sr..16 * sr + sr / 5).contains(&cut), "cut at {} s", cut as f32 / sr as f32);
    assert_eq!(pieces[0].start, 0);
    assert_eq!(pieces[1].start, cut);
    assert_eq!(pieces[1].end, audio.len());

    // Everything covered, nothing longer than the cap, for many lengths.
    for secs in [0usize, 1, 19, 20, 21, 45, 61] {
        let a = vec![0.1f32; secs * sr + 7];
        let p = split_points(&a, 20 * sr);
        assert_eq!(p.first().unwrap().start, 0);
        assert_eq!(p.last().unwrap().end, a.len());
        for w in p.windows(2) {
            assert_eq!(w[0].end, w[1].start);
        }
        assert!(p.iter().all(|r| r.len() <= 20 * sr), "{secs}: {p:?}");
    }
    // A tiny cap is raised to 0.2 s pieces (cut in their last 30 %), never to empty ones.
    let tiny = split_points(&vec![0.0; sr], 10);
    assert!(tiny.len() > 1);
    assert!(tiny.iter().all(|r| r.len() <= sr / 5 && (r.len() >= sr / 5 * 7 / 10 || r.end == sr)), "{tiny:?}");
}

/// Review P2-D: a forced split never leaves a sliver: 20.2 s of continuous speech whose quietest
/// point is at 19.99 s used to leave a 0.2 s last piece that models refuse or misdecode.
#[test]
fn a_split_never_leaves_a_tiny_last_piece() {
    let sr = SAMPLE_RATE as usize;
    let len = 20 * sr + sr / 5;
    let mut audio: Vec<f32> = (0..len).map(|i| ((i as f32) * 0.05).sin() * 0.3).collect();
    for s in &mut audio[20 * sr - sr / 50 - 10..20 * sr - 10] {
        *s = 0.0;
    }
    let pieces = split_points(&audio, 20 * sr);
    assert_eq!(pieces.len(), 2, "{pieces:?}");
    assert!(pieces[1].len() >= MIN_TAIL_SAMPLES, "last piece {} samples", pieces[1].len());
    assert!(pieces[0].len() <= 20 * sr);
    // Every length just above the cap, quietest point anywhere.
    for extra in [1usize, 160, 3200, 8000, 15_999, 16_000, 40_000] {
        let a = vec![0.1f32; 20 * sr + extra];
        let p = split_points(&a, 20 * sr);
        assert!(p.last().unwrap().len() >= MIN_TAIL_SAMPLES, "{extra}: {p:?}");
        assert!(p.iter().all(|r| r.len() <= 20 * sr && !r.is_empty()), "{extra}: {p:?}");
        assert_eq!(p.last().unwrap().end, a.len());
    }
}

#[test]
fn transcripts_join_with_spaces_only_between_words() {
    let t = |v: &[&str]| join_texts(&v.iter().map(|s| s.to_string()).collect::<Vec<_>>());
    assert_eq!(t(&["Hello there.", " How are you? "]), "Hello there. How are you?");
    assert_eq!(t(&["今日は", "いい天気ですね。"]), "今日はいい天気ですね。");
    assert_eq!(t(&["안녕하세요", "반갑습니다"]), "안녕하세요반갑습니다", "Hangul pieces of one utterance join too");
    assert_eq!(t(&["VRChat", "で遊ぶ"]), "VRChatで遊ぶ");
    assert_eq!(t(&["", "  ", "one"]), "one");
    assert_eq!(t(&[]), "");
}

#[test]
fn whisper_prompts_combine_prompt_and_hotwords_within_bounds() {
    let o = DecodeOptions {
        prompt: Some("  Stream chat. ".into()),
        hotwords: vec!["Sigil".into(), " ".into(), "VRChat".into()],
        ..DecodeOptions::default()
    };
    assert_eq!(whisper_prompt(&o).as_deref(), Some("Stream chat. Sigil, VRChat."));
    assert_eq!(whisper_prompt(&DecodeOptions::default()), None);
    // Long multibyte prompts are cut on a character boundary.
    let long = DecodeOptions {
        prompt: Some("語".repeat(1000)),
        ..DecodeOptions::default()
    };
    let p = whisper_prompt(&long).unwrap();
    assert!(p.len() <= MAX_PROMPT_BYTES && p.chars().all(|c| c == '語'));
}

// ---------------------------------------------------------------------------------------------
// Governor → wrapper options
// ---------------------------------------------------------------------------------------------

fn resolved(id: &str) -> ResolvedModel {
    ResolvedModel {
        label: id.into(),
        catalog_id: Some(id.into()),
        title: "Qwen3-ASR 1.7B".into(),
        path: PathBuf::from(r"C:\models\q.gguf"),
        file_bytes: 2_185_030_624,
        gpu_mb: 3100,
        family: Family::Qwen3Asr,
    }
}

fn admission(gpu: Option<RuntimeVariant>, code: &'static str) -> Admission {
    Admission {
        gpu,
        code,
        reason: gpu.is_none().then(|| "Qwen3-ASR runs on the CPU: test".to_string()),
        need_mb: 3500,
        available_mb: 0,
    }
}

#[test]
fn the_gpu_is_used_only_through_the_governors_admission() {
    let model = resolved("stt/qwen3-asr-1.7b-q8_0");
    let mut spec = EngineSpec::new(EngineKind::Transcribe, "stt/qwen3-asr-1.7b-q8_0");
    spec.threads = Some(500);
    let opts = Options::default();

    // Admitted: the GPU build plus the CPU ladder behind it; never the unmeasured vulkan build.
    let a = admission(Some(RuntimeVariant::Cuda13), governor::codes::FITS);
    let mo = model_options(&model, &spec, &opts, Want::Gpu(RuntimeVariant::Cuda13), &a, Some("0000:01:00.0".into()));
    assert_eq!(mo.backend, BackendChoice::Auto);
    let allowed = mo.allowed_variants.clone().unwrap();
    assert_eq!(allowed[0], RuntimeVariant::Cuda13);
    assert!(!allowed.contains(&RuntimeVariant::Vulkan));
    assert!(allowed[1..].iter().all(|v| !v.is_gpu()));
    assert_eq!(mo.threads, Some(64));
    assert_eq!(mo.device_id.as_deref(), Some("0000:01:00.0"), "the GPU the governor budgeted");
    assert!(mo.cpu_fallback && mo.warmup);
    assert_eq!(mo.min_free_vram_mb, DEFAULT_MIN_FREE_VRAM_MB);

    // Denied (any reason): the CPU backend only.
    for code in [governor::codes::GPU_FULL, governor::codes::VC_RUNNING, governor::codes::NO_GPU_RUNTIME] {
        let a = admission(None, code);
        let mo = model_options(&model, &spec, &opts, Want::Auto, &a, None);
        assert_eq!(mo.backend, BackendChoice::Cpu, "{code}");
        assert_eq!(mo.variant, None);
        assert_eq!(mo.allowed_variants, None);
    }
    // A named CPU build is forced.
    let a = admission(None, governor::codes::CPU_REQUESTED);
    let mo = model_options(&model, &spec, &opts, Want::Cpu(Some(RuntimeVariant::CpuAvx2)), &a, None);
    assert_eq!((mo.backend, mo.variant), (BackendChoice::Cpu, Some(RuntimeVariant::CpuAvx2)));

    // An explicit device wins over the governor's primary GPU.
    let explicit = Options::parse(&serde_json::json!({ "device": 1 }));
    let a = admission(Some(RuntimeVariant::Cuda13), governor::codes::FITS);
    let mo = model_options(&model, &spec, &explicit, Want::Auto, &a, Some("0000:01:00.0".into()));
    assert_eq!((mo.device, mo.device_id.as_deref()), (Some(1), None));
    let by_id = Options::parse(&serde_json::json!({ "deviceId": "NVIDIA GeForce RTX 4060", "minFreeVramMb": 512 }));
    let mo = model_options(&model, &spec, &by_id, Want::Auto, &a, Some("0000:01:00.0".into()));
    assert_eq!(mo.device_id.as_deref(), Some("NVIDIA GeForce RTX 4060"));
    assert_eq!(mo.min_free_vram_mb, 512);
}

#[test]
fn a_denied_gpu_without_cpu_fallback_says_why() {
    let full = Admission {
        reason: Some("Qwen3-ASR runs on the CPU: it needs 3.4 GB of video memory and 1.8 GB is free for it".into()),
        ..admission(None, governor::codes::GPU_FULL)
    };
    let e = denial_error(&full);
    assert_eq!(e.kind, AsrErrorKind::OutOfMemory);
    assert!(e.message.contains("1.8 GB is free") && e.message.contains("CPU fallback is off"), "{e}");
    assert_eq!(denial_error(&admission(None, governor::codes::VC_RUNNING)).kind, AsrErrorKind::OutOfMemory);
    for code in [governor::codes::NO_GPU, governor::codes::NO_GPU_RUNTIME, governor::codes::SMALL_GPU] {
        assert_eq!(denial_error(&admission(None, code)).kind, AsrErrorKind::Unsupported, "{code}");
    }
}

#[test]
fn wrapper_errors_map_to_engine_errors() {
    let asr_error = |e| asr_error_for(e, Want::Auto);
    let e = asr_error(TranscribeError::new(
        ErrorKind::InvalidInput,
        "another stream is active on this model (between its first push and finish)",
    ));
    assert_eq!(e.kind, AsrErrorKind::Busy);
    assert_eq!(
        asr_error(TranscribeError::new(ErrorKind::InvalidInput, "bad language")).kind,
        AsrErrorKind::InvalidInput
    );
    for (k, want) in [
        (ErrorKind::OutOfMemory, AsrErrorKind::OutOfMemory),
        (ErrorKind::Unsupported, AsrErrorKind::Unsupported),
        (ErrorKind::Cancelled, AsrErrorKind::Cancelled),
        (ErrorKind::Load, AsrErrorKind::Runtime),
        (ErrorKind::Backend, AsrErrorKind::Runtime),
        (ErrorKind::Runtime, AsrErrorKind::Runtime),
    ] {
        assert_eq!(asr_error(TranscribeError::new(k, "x")).kind, want, "{k:?}");
    }
    // A model that is not installed keeps its id first (the pipeline reads it from there) ...
    let e = asr_error(TranscribeError::new(ErrorKind::NotInstalled, "stt/qwen3-asr-1.7b-q8_0: the model is not installed"));
    assert_eq!(e.kind, AsrErrorKind::NotInstalled);
    assert!(e.message.starts_with("stt/qwen3-asr-1.7b-q8_0: "), "{e}");
    // ... and a missing runtime is prefixed with the pack this PC should install.
    let e = asr_error(TranscribeError::new(ErrorKind::NotInstalled, "the transcribe.cpp runtime is not installed"));
    assert!(e.message.starts_with("runtime/sigil-transcribe-"), "{e}");
}

/// Review P2-D: a missing runtime is named for what the load runs on, never the GPU pack for a
/// CPU run (a forced CPU build, or a GPU the governor denied).
#[test]
fn missing_runtimes_are_named_for_what_the_load_runs_on() {
    let missing = || TranscribeError::new(ErrorKind::NotInstalled, "no installed transcribe.cpp variant fits (CPU requested)");
    let e = asr_error_for(missing(), Want::Cpu(Some(RuntimeVariant::CpuAvx2)));
    assert!(e.message.starts_with("runtime/sigil-transcribe-cpu-avx2: "), "{e}");
    let e = asr_error_for(missing(), Want::Cpu(None));
    assert!(e.message.starts_with("runtime/sigil-transcribe-cpu"), "{e}");
    assert!(!e.message.contains("cuda13"), "{e}");

    // The effective want: the admitted GPU build, else a CPU build (the named one if any).
    let admitted = admission(Some(RuntimeVariant::Cuda13), governor::codes::FITS);
    let denied = admission(None, governor::codes::GPU_FULL);
    assert_eq!(effective_want(Want::Auto, &admitted), Want::Gpu(RuntimeVariant::Cuda13));
    assert_eq!(effective_want(Want::Gpu(RuntimeVariant::Cuda13), &denied), Want::Cpu(None));
    assert_eq!(effective_want(Want::Auto, &denied), Want::Cpu(None));
    let cpu = admission(None, governor::codes::CPU_REQUESTED);
    assert_eq!(
        effective_want(Want::Cpu(Some(RuntimeVariant::CpuAvx2)), &cpu),
        Want::Cpu(Some(RuntimeVariant::CpuAvx2))
    );
}

/// Review P2-D: a spec naming a GPU build this PC can use needs exactly that build (plus the
/// NVIDIA runtime for cuda13); a CPU pack never stands in for it.
#[test]
fn a_named_gpu_build_needs_that_build_not_any_installed_one() {
    let all = |_: RuntimeVariant| true;
    let cuda_only = |v: RuntimeVariant| v == RuntimeVariant::Cuda13;
    let none = |_: RuntimeVariant| false;
    let cuda = Want::Gpu(RuntimeVariant::Cuda13);
    // An RTX PC with only a CPU pack: the CPU pack does not satisfy the GPU tier.
    assert!(!variant_satisfies(cuda, RuntimeVariant::CpuAvx2, &all, &cuda_only));
    assert!(variant_satisfies(cuda, RuntimeVariant::Cuda13, &all, &cuda_only));
    assert_eq!(wanted_variant(cuda, &cuda_only, RuntimeVariant::CpuAvx512), RuntimeVariant::Cuda13);
    // A PC that cannot use CUDA for speech (AMD, Pascal, a small card): the governor runs it on
    // the CPU, so a CPU pack is what it needs.
    assert!(variant_satisfies(cuda, RuntimeVariant::CpuAvx2, &all, &none));
    assert_eq!(wanted_variant(cuda, &none, RuntimeVariant::CpuAvx2), RuntimeVariant::CpuAvx2);
    // Auto: any build that runs here; the GPU build is recommended when usable.
    assert!(variant_satisfies(Want::Auto, RuntimeVariant::CpuAvx2, &all, &cuda_only));
    assert_eq!(wanted_variant(Want::Auto, &cuda_only, RuntimeVariant::CpuAvx2), RuntimeVariant::Cuda13);
    assert_eq!(wanted_variant(Want::Auto, &none, RuntimeVariant::CpuAvx2), RuntimeVariant::CpuAvx2);
    // CPU requests: the named build exactly; any CPU build otherwise, never a GPU build.
    let avx2 = Want::Cpu(Some(RuntimeVariant::CpuAvx2));
    assert!(!variant_satisfies(avx2, RuntimeVariant::CpuAvx512, &all, &cuda_only));
    assert_eq!(wanted_variant(avx2, &cuda_only, RuntimeVariant::CpuAvx512), RuntimeVariant::CpuAvx2);
    assert!(!variant_satisfies(Want::Cpu(None), RuntimeVariant::Cuda13, &all, &cuda_only));
    assert!(variant_satisfies(Want::Cpu(None), RuntimeVariant::Cpu, &all, &cuda_only));
    assert_eq!(wanted_variant(Want::Cpu(None), &cuda_only, RuntimeVariant::CpuAvx2), RuntimeVariant::CpuAvx2);

    // Live: on a PC that can use CUDA 13 for speech, the GPU tier asks for its pack and the
    // NVIDIA runtime whatever else is installed.
    let reqs = requirements(&gpu_spec("stt/qwen3-asr-1.7b-q8_0"));
    let support = VariantSupport::detect();
    if gpu_usable(RuntimeVariant::Cuda13, &support) {
        assert_eq!(reqs[1].0, "runtime/sigil-transcribe-cuda13", "{reqs:?}");
        assert_eq!(reqs.get(2).map(|r| r.0.as_str()), Some(NVIDIA_RUNTIME_ID));
    } else {
        assert!(!reqs[1].0.ends_with("cuda13"), "{reqs:?}");
    }
}

/// Review P2-D: a model admitted to the GPU that still loads on the CPU says why, even when no
/// GPU attempt failed (the GPU build is not installed or was skipped).
#[test]
fn an_admitted_gpu_that_was_not_used_is_never_silent() {
    let why = gpu_not_used_reason("Qwen3-ASR 1.7B", RuntimeVariant::Cuda13, &[]);
    assert!(
        why.starts_with("Qwen3-ASR 1.7B runs on the CPU: ") && why.contains("runtime/sigil-transcribe-cuda13") && why.contains("not installed"),
        "{why}"
    );
    let skipped = Attempt {
        variant: Some(RuntimeVariant::Cuda13),
        pack_id: Some("runtime/sigil-transcribe-cuda13".into()),
        dir: None,
        error: "the NVIDIA CUDA 13 runtime is not installed".into(),
    };
    let other = Attempt {
        variant: Some(RuntimeVariant::Vulkan),
        error: "not allowed by the resource governor for this model".into(),
        ..skipped.clone()
    };
    let why = gpu_not_used_reason("q", RuntimeVariant::Cuda13, &[other, skipped]);
    assert!(why.contains("was not used (the NVIDIA CUDA 13 runtime is not installed)"), "{why}");
    // A CPU model holds VRAM only in a worker whose GPU runtime was initialized (the parent keeps
    // claiming the context); in-process the context has its own pinned claim.
    assert_eq!(cpu_vram_mb(true, true), models::GPU_CONTEXT_MB);
    assert_eq!(cpu_vram_mb(true, false), 0);
    assert_eq!(cpu_vram_mb(false, true), 0);
}

/// Review P2-D: the worker's crash CPU spec exists only with the CPU fallback; restarts are
/// admitted again (GPU with new claims, the CPU for now, or refused without the fallback).
#[test]
fn worker_restarts_follow_the_governor_and_the_cpu_fallback() {
    let spec = {
        let mut s = gpu_spec("stt/qwen3-asr-1.7b-q8_0");
        s.options = serde_json::json!({ "isolation": "worker", "language": "ja" });
        s
    };
    let gpu = admission(Some(RuntimeVariant::Cuda13), governor::codes::FITS);
    let with = Options::parse(&spec.options);
    let without = Options::parse(&serde_json::json!({ "isolation": "worker", "cpuFallback": false }));
    let cpu = crash_cpu_spec(&spec, &with, &gpu, "Qwen3-ASR 1.7B").expect("a CPU spec with the fallback");
    let fwd: ForwardedAdmission = serde_json::from_value(cpu.options[ADMISSION_KEY].clone()).unwrap();
    assert_eq!((fwd.gpu, fwd.code.as_str()), (None, governor::codes::WORKER_CRASHED));
    assert_eq!(cpu.options["isolation"], "inProcess");
    assert_eq!(cpu.options["language"], "ja");
    assert!(crash_cpu_spec(&spec, &without, &gpu, "q").is_none(), "no CPU spec without the fallback");
    assert!(crash_cpu_spec(&spec, &with, &admission(None, governor::codes::NO_GPU), "q").is_none());

    // The child spec carries the admission and, on the GPU, the budgeted device.
    let child = child_spec(&spec, Some("0000:01:00.0"), &gpu);
    assert_eq!(child.options["deviceId"], "0000:01:00.0");
    let fwd: ForwardedAdmission = serde_json::from_value(child.options[ADMISSION_KEY].clone()).unwrap();
    assert_eq!(fwd.gpu.as_deref(), Some("cuda13"));
    assert!(child_spec(&spec, Some("0000:01:00.0"), &admission(None, governor::codes::GPU_FULL))
        .options
        .get("deviceId")
        .is_none());

    match readmission(&gpu, GpuClaims::none(), &spec, None, true, "q") {
        Readmission::Gpu { spec, .. } => assert_eq!(spec.options[ADMISSION_KEY]["gpu"], "cuda13"),
        _ => panic!("admitted: the GPU"),
    }
    let full = Admission {
        reason: Some("Qwen3-ASR runs on the CPU: it needs 3.4 GB of video memory and 1.8 GB is free for it".into()),
        ..admission(None, governor::codes::GPU_FULL)
    };
    match readmission(&full, GpuClaims::none(), &spec, Some("0000:01:00.0"), true, "q") {
        Readmission::Cpu { spec, reason } => {
            assert!(reason.contains("1.8 GB is free"), "{reason}");
            assert!(spec.options[ADMISSION_KEY]["gpu"].is_null());
            assert!(spec.options.get("deviceId").is_none());
        }
        _ => panic!("denied with the fallback: the CPU for now"),
    }
    match readmission(&full, GpuClaims::none(), &spec, None, false, "q") {
        Readmission::Refused(e) => assert_eq!(e.kind, AsrErrorKind::OutOfMemory),
        _ => panic!("denied without the fallback: refused"),
    }
}

#[test]
fn workers_get_a_generous_memory_cap() {
    assert_eq!(worker_memory_limit_mb(0), 8192);
    // Qwen3-ASR-1.7B Q8_0 (2084 MiB) and BF16 (3894 MiB).
    assert_eq!(worker_memory_limit_mb(2_185_030_624), 2083 * 3 + 4096);
    assert_eq!(worker_memory_limit_mb(4_083_087_904), 3893 * 3 + 4096);
}

#[test]
fn requirements_name_the_model_and_a_runtime_for_this_pc() {
    let reqs = requirements(&EngineSpec::new(EngineKind::Transcribe, "stt/qwen3-asr-1.7b-q8_0"));
    assert_eq!(reqs[0].0, "stt/qwen3-asr-1.7b-q8_0");
    assert!(reqs[1].0.starts_with("runtime/sigil-transcribe-"), "{reqs:?}");
    if reqs[1].0 == "runtime/sigil-transcribe-cuda13" {
        assert_eq!(reqs.get(2).map(|r| r.0.as_str()), Some(NVIDIA_RUNTIME_ID));
    }
    // A forced CPU build never asks for the NVIDIA runtime; a file needs only the runtime.
    let mut cpu = EngineSpec::new(EngineKind::Transcribe, r"C:\models\my.gguf");
    cpu.variant = Some("cpu-avx2".into());
    let reqs = requirements(&cpu);
    assert_eq!(reqs.len(), 1, "{reqs:?}");
    assert_eq!(reqs[0].0, "runtime/sigil-transcribe-cpu-avx2");
}

/// The error of a create that must fail (engines are not `Debug`).
fn create_err(spec: &EngineSpec, ctx: &EngineContext) -> AsrError {
    match create(spec, ctx) {
        Ok(e) => panic!("expected an error, got an engine on {}", e.info().device),
        Err(e) => e,
    }
}

#[test]
fn bad_models_fail_fast_without_loading_anything() {
    let ctx = EngineContext::headless();
    let e = create_err(&EngineSpec::new(EngineKind::Transcribe, "stt/not-a-transcribe-model"), &ctx);
    assert_eq!(e.kind, AsrErrorKind::InvalidInput);
    let missing = std::env::temp_dir().join("sigil-p2d-no-such-model.gguf");
    let e = create_err(&EngineSpec::new(EngineKind::Transcribe, missing.to_string_lossy()), &ctx);
    assert_eq!(e.kind, AsrErrorKind::InvalidInput);
    assert!(e.message.contains("does not exist"), "{e}");
    assert_eq!(create_err(&EngineSpec::new(EngineKind::Transcribe, " "), &ctx).kind, AsrErrorKind::InvalidInput);
}

// ---------------------------------------------------------------------------------------------
// Real runtime + models (ignored)
// ---------------------------------------------------------------------------------------------

/// Env changes and GPU loads in the ignored tests are serialized.
static REAL: Mutex<()> = Mutex::new(());

fn require<T>(value: Option<T>, what: &str) -> Option<T> {
    if value.is_some() {
        return value;
    }
    if std::env::var_os("SIGIL_TEST_ALLOW_SKIP").is_some_and(|v| !v.is_empty()) {
        eprintln!("{what} missing; skipped (SIGIL_TEST_ALLOW_SKIP)");
        return None;
    }
    panic!(
        "{what} missing (SIGIL_APP_DATA_DIR = {:?}, SIGIL_TEST_MODELS_DIR = {:?}): see the top of speech/engines/transcribe/tests.rs, \
         or set SIGIL_TEST_ALLOW_SKIP=1",
        std::env::var_os("SIGIL_APP_DATA_DIR"),
        std::env::var_os("SIGIL_TEST_MODELS_DIR")
    );
}

/// An optional model: skipped with a message when not installed.
fn optional(id: &str) -> bool {
    let here = crate::services::model_store::installed_path(id).is_some();
    if !here {
        eprintln!("{id} is not installed in the test model store; skipped");
    }
    here
}

fn installed(id: &str) -> Option<()> {
    require(crate::services::model_store::installed_path(id).map(|_| ()), id)
}

fn sample(name: &str) -> Option<Vec<f32>> {
    let path = std::env::var_os("SIGIL_TEST_MODELS_DIR")
        .map(PathBuf::from)?
        .join("models")
        .join("transcribe")
        .join("samples")
        .join(name);
    let mut reader = hound::WavReader::open(&path).ok()?;
    let spec = reader.spec();
    let samples: Vec<f32> = match spec.sample_format {
        hound::SampleFormat::Float => reader.samples::<f32>().filter_map(Result::ok).collect(),
        hound::SampleFormat::Int => {
            let scale = (1i64 << (spec.bits_per_sample - 1)) as f32;
            reader
                .samples::<i32>()
                .filter_map(Result::ok)
                .map(|s| s as f32 / scale)
                .collect()
        }
    };
    crate::services::speech_rt::transcribe::to_16k_mono(&samples, spec.sample_rate, spec.channels).ok()
}

fn cjk(text: &str) -> bool {
    text.chars()
        .any(|c| ('\u{3040}'..='\u{30ff}').contains(&c) || ('\u{4e00}'..='\u{9fff}').contains(&c))
}

fn claims_labelled(label: &str) -> Vec<(Subsystem, u64, bool)> {
    resources::snapshot()
        .claims
        .into_iter()
        .filter(|c| c.label == label)
        .map(|c| (c.subsystem, c.mb, c.external))
        .collect()
}

fn process_vram_mb() -> Option<u64> {
    governor::primary_adapter()
        .and_then(governor::sample)
        .map(|s| s.process_mb)
}

fn statuses() -> (EngineContext, std::sync::Arc<Mutex<Vec<EngineStatus>>>) {
    let seen: std::sync::Arc<Mutex<Vec<EngineStatus>>> = Default::default();
    let sink = std::sync::Arc::clone(&seen);
    (
        EngineContext {
            status: StatusSink::new(move |s| sink.lock().unwrap().push(s)),
        },
        seen,
    )
}

fn gpu_spec(model: &str) -> EngineSpec {
    let mut spec = EngineSpec::new(EngineKind::Transcribe, model);
    spec.variant = Some("cuda13".into());
    spec
}

/// The acceptance rule on real hardware: with room, Qwen3-ASR-1.7B loads on CUDA through the
/// governor, holds a claim sized to the measured VRAM, detects Japanese, splits a 28 s utterance at
/// the warm-up length and releases its claims when dropped.
#[test]
#[ignore = "needs SIGIL_APP_DATA_DIR with runtime/sigil-transcribe-cuda13 (+ runtime/nvidia-cuda13) and stt/qwen3-asr-1.7b-q8_0, SIGIL_TEST_MODELS_DIR samples"]
fn real_qwen3_final_on_the_gpu_through_the_governor() {
    let _one = REAL.lock().unwrap_or_else(|e| e.into_inner());
    let Some(()) = installed("stt/qwen3-asr-1.7b-q8_0") else { return };
    let Some(()) = installed("runtime/sigil-transcribe-cuda13") else { return };
    let Some(ja) = require(sample("ja-short.wav"), "samples/ja-short.wav") else { return };
    let Some(ja_long) = require(sample("ja-long.wav"), "samples/ja-long.wav") else { return };

    let before = process_vram_mb();
    let (ctx, seen) = statuses();
    let t0 = std::time::Instant::now();
    let mut e = create(&gpu_spec("stt/qwen3-asr-1.7b-q8_0"), &ctx).expect("load");
    let load_ms = t0.elapsed().as_millis();
    let info = e.info().clone();
    eprintln!("loaded in {load_ms} ms: {info:?}");
    assert_eq!(info.device, "cuda", "{info:?}");
    assert!(info.fallback_reason.is_none(), "{info:?}");
    assert!(info.runtime.contains("cuda13"), "{}", info.runtime);
    assert!((2800..=4300).contains(&info.vram_mb), "claimed {} MB", info.vram_mb);
    assert!(seen.lock().unwrap().iter().any(|s| s.code == codes::LOADING_MODEL));
    // The model claim (resized to the measured footprint) and the shared context claim.
    let model_claims = claims_labelled("speech: stt/qwen3-asr-1.7b-q8_0");
    assert_eq!(model_claims.len(), 1, "{model_claims:?}");
    assert_eq!(model_claims[0].0, Subsystem::SttFinal);
    assert!(!model_claims[0].2);
    let context = claims_labelled("speech: CUDA 13 runtime context (transcribe.cpp)");
    eprintln!(
        "VRAM: process {before:?} -> {:?} MiB; claims model {:?}, context {:?}",
        process_vram_mb(),
        model_claims,
        context
    );

    let r = e.transcribe(&ja, &DecodeOptions::default()).unwrap();
    eprintln!("ja-short ({} ms): {} {:?} {:?}", r.decode_ms, r.text, r.lang, r.tags);
    assert!(cjk(&r.text), "{}", r.text);
    assert_eq!(r.lang.as_deref(), Some("ja"));
    assert!(r.tags.contains(&"lang:ja".to_string()), "{:?}", r.tags);
    // With the hint there is nothing to detect (no tag).
    let r = e
        .transcribe(
            &ja,
            &DecodeOptions {
                language: Some("ja".into()),
                ..DecodeOptions::default()
            },
        )
        .unwrap();
    assert!(r.tags.is_empty() && cjk(&r.text));

    // 28.2 s: split at a quiet point below the 20 s warm-up length, joined without spaces.
    let r = e.transcribe(&ja_long, &DecodeOptions::default()).unwrap();
    eprintln!("ja-long ({} ms): {}", r.decode_ms, r.text);
    assert!(cjk(&r.text) && r.text.chars().count() > 40, "{}", r.text);
    let after_runs = process_vram_mb();
    eprintln!("VRAM after runs: {after_runs:?} MiB");

    drop(e);
    assert!(claims_labelled("speech: stt/qwen3-asr-1.7b-q8_0").is_empty(), "claims released with the engine");
    // The CUDA context stays with the pinned DLL (about 380 MiB), and so does its claim.
    assert_eq!(
        claims_labelled("speech: CUDA 13 runtime context (transcribe.cpp)"),
        vec![(Subsystem::SttFinal, models::GPU_CONTEXT_MB, false)]
    );
    eprintln!("VRAM after drop: {:?} MiB", process_vram_mb());
}

/// The GPU tier is not used when the governor says no: a full card (another subsystem's claim)
/// keeps the model on the CPU with a reason; without the CPU fallback the load is refused.
#[test]
#[ignore = "needs SIGIL_APP_DATA_DIR with runtime/sigil-transcribe-cuda13 (+ runtime/nvidia-cuda13) and bench/transcribe-whisper-tiny-q8_0, SIGIL_TEST_MODELS_DIR samples"]
fn real_governor_denial_keeps_the_model_off_the_gpu() {
    let _one = REAL.lock().unwrap_or_else(|e| e.into_inner());
    let Some(()) = installed("bench/transcribe-whisper-tiny-q8_0") else { return };
    let Some(()) = installed("runtime/sigil-transcribe-cuda13") else { return };
    let Some(jfk) = require(sample("jfk.wav"), "samples/jfk.wav") else { return };
    let whisper = crate::services::model_store::installed_path("bench/transcribe-whisper-tiny-q8_0")
        .unwrap()
        .join("whisper-tiny-Q8_0.gguf");
    let spec = gpu_spec(&whisper.to_string_lossy());

    // Room: on the GPU.
    let e = create(&spec, &EngineContext::headless()).expect("load on the GPU");
    assert_eq!(e.info().device, "cuda", "{:?}", e.info());
    drop(e);

    // A 1 TB claim of another subsystem fills the card as far as the governor knows.
    let hog = resources::claim_vram(Subsystem::Llm, "test: p2d governor hog", 1_000_000);
    let (ctx, seen) = statuses();
    let mut e = create(&spec, &ctx).expect("load on the CPU");
    let info = e.info().clone();
    eprintln!("{info:?}");
    assert_eq!(info.device, "cpu");
    assert_eq!(info.vram_mb, 0);
    let why = info.fallback_reason.clone().unwrap_or_default();
    assert!(why.contains("runs on the CPU") && why.contains("video memory"), "{why}");
    assert!(claims_labelled(&format!("speech: {}", info.model)).is_empty(), "a CPU engine claims nothing");
    let r = e.transcribe(&jfk, &DecodeOptions::default()).unwrap();
    assert!(r.text.to_lowercase().contains("country"), "{}", r.text);
    assert!(seen.lock().unwrap().iter().any(|s| s.code == codes::LOADING_MODEL && s.detail.contains("processor")));
    drop(e);

    let mut strict = spec.clone();
    strict.options = serde_json::json!({ "cpuFallback": false });
    let err = create_err(&strict, &EngineContext::headless());
    assert_eq!(err.kind, AsrErrorKind::OutOfMemory, "{err}");
    drop(hog);
}

/// Dev check "two-pass ja with VRAM logged": the GPU final replaces the first pass through the
/// real two-pass queue well inside the budget (P2 exit: ≤ 800 ms after VAD end for 8 s), with a
/// CPU fallback engine standing by.
#[test]
#[ignore = "needs SIGIL_APP_DATA_DIR with the cuda13 + a CPU pack, stt/qwen3-asr-1.7b-q8_0 and bench/transcribe-whisper-tiny-q8_0, SIGIL_TEST_MODELS_DIR samples"]
fn real_two_pass_ja_with_vram_logged() {
    let _one = REAL.lock().unwrap_or_else(|e| e.into_inner());
    let Some(()) = installed("stt/qwen3-asr-1.7b-q8_0") else { return };
    let Some(()) = installed("bench/transcribe-whisper-tiny-q8_0") else { return };
    let Some(ja) = require(sample("ja-short.wav"), "samples/ja-short.wav") else { return };
    let vram0 = process_vram_mb();
    let primary = create(&gpu_spec("stt/qwen3-asr-1.7b-q8_0"), &EngineContext::headless()).expect("GPU final");
    assert_eq!(primary.info().device, "cuda");
    let vram_loaded = process_vram_mb();
    let whisper = crate::services::model_store::installed_path("bench/transcribe-whisper-tiny-q8_0")
        .unwrap()
        .join("whisper-tiny-Q8_0.gguf");
    let mut cpu = EngineSpec::new(EngineKind::Transcribe, whisper.to_string_lossy());
    cpu.variant = Some("cpu-any".into());
    let fallback = create(&cpu, &EngineContext::headless()).expect("CPU fallback");
    assert_eq!(fallback.info().device, "cpu");

    let (tx, rx) = std::sync::mpsc::channel();
    let two_pass = TwoPass::spawn(
        primary,
        Some(fallback),
        TwoPassConfig::default(),
        Box::new(move |o| {
            let _ = tx.send((o, std::time::Instant::now()));
        }),
    )
    .unwrap();
    // Two utterances: the second measures the steady state (8 s of the clip, as in the exit check).
    let eight: std::sync::Arc<[f32]> = ja[..(8 * SAMPLE_RATE as usize).min(ja.len())].into();
    for segment_id in 1..=2u64 {
        let closed_at = std::time::Instant::now();
        assert!(two_pass.submit(FinalJob {
            segment_id,
            revision: 2,
            audio: std::sync::Arc::clone(&eight),
            options: DecodeOptions {
                language: Some("ja".into()),
                ..DecodeOptions::default()
            },
            first_pass: Some(AsrResult {
                text: "first pass".into(),
                ..AsrResult::default()
            }),
            closed_at,
        }));
        let (outcome, at) = rx.recv_timeout(Duration::from_secs(30)).expect("an outcome");
        let latency = at.duration_since(closed_at).as_millis();
        match outcome {
            TwoPassOutcome::Final {
                segment_id: id,
                result,
                engine,
                model,
                promoted_fallback,
                ..
            } => {
                assert_eq!(id, segment_id);
                assert!(!promoted_fallback);
                assert_eq!((engine.as_str(), model.as_str()), ("transcribe", "stt/qwen3-asr-1.7b-q8_0"));
                assert!(cjk(&result.text), "{}", result.text);
                eprintln!(
                    "two-pass ja segment {segment_id}: final {latency} ms after VAD end: {} | VRAM MiB before {vram0:?}, loaded {vram_loaded:?}, now {:?}",
                    result.text,
                    process_vram_mb()
                );
                assert!(latency <= 800, "{latency} ms");
            }
            TwoPassOutcome::Skipped { reason, .. } => panic!("skipped: {reason}"),
        }
    }
    two_pass.shutdown();
}

/// Worker isolation end to end: the model loads in a `--headless stt-worker` child, the parent
/// holds its external claim, and transcripts come back over the pipes.
#[test]
#[ignore = "needs a debug sigil.exe (SIGIL_STT_WORKER_EXE), SIGIL_APP_DATA_DIR with the cuda13 pack and bench/transcribe-whisper-tiny-q8_0, SIGIL_TEST_MODELS_DIR samples"]
fn real_worker_isolation_round_trip() {
    let _one = REAL.lock().unwrap_or_else(|e| e.into_inner());
    let Some(()) = installed("bench/transcribe-whisper-tiny-q8_0") else { return };
    let Some(jfk) = require(sample("jfk.wav"), "samples/jfk.wav") else { return };
    if std::env::var_os(worker_proc::WORKER_EXE_ENV).is_none() {
        let exe = std::env::current_exe().unwrap();
        let sigil = exe.parent().and_then(Path::parent).map(|d| d.join("sigil.exe"));
        let Some(p) = require(sigil.filter(|p| p.is_file()), "target/debug/sigil.exe (cargo build)") else { return };
        std::env::set_var(worker_proc::WORKER_EXE_ENV, p);
    }
    let whisper = crate::services::model_store::installed_path("bench/transcribe-whisper-tiny-q8_0")
        .unwrap()
        .join("whisper-tiny-Q8_0.gguf");
    let mut spec = gpu_spec(&whisper.to_string_lossy());
    spec.options = serde_json::json!({ "isolation": "worker" });
    let (ctx, seen) = statuses();
    let mut e = create(&spec, &ctx).expect("load in a worker");
    let info = e.info().clone();
    eprintln!("{info:?}");
    assert!(info.runtime.ends_with("(worker process)"), "{}", info.runtime);
    assert!(seen.lock().unwrap().iter().any(|s| s.code == codes::LOADING_MODEL), "status crosses the pipe");
    let label = "speech: whisper-tiny-Q8_0.gguf (worker process)";
    if info.device == "cuda" {
        let claims = claims_labelled(label);
        assert_eq!(claims.len(), 1, "{claims:?}");
        assert!(claims[0].2, "external claim");
        assert_eq!(claims[0].1, info.vram_mb, "resized to the child's measured VRAM");
        assert!(info.vram_mb >= 300, "{info:?}");
    }
    for _ in 0..2 {
        let r = e.transcribe(&jfk, &DecodeOptions::default()).unwrap();
        assert!(r.text.to_lowercase().contains("country"), "{}", r.text);
    }
    drop(e);
    assert!(claims_labelled(label).is_empty());
}

/// Voxtral Mini 4B Realtime streams partials (and flicker-free committed text in append-only
/// mode). Skipped when the model is not installed.
#[test]
#[ignore = "needs SIGIL_APP_DATA_DIR with the cuda13 pack and stt/voxtral-mini-4b-realtime-q4_k_m, SIGIL_TEST_MODELS_DIR samples"]
fn real_voxtral_realtime_streams_partials() {
    let _one = REAL.lock().unwrap_or_else(|e| e.into_inner());
    if !optional("stt/voxtral-mini-4b-realtime-q4_k_m") {
        return;
    }
    let Some(jfk) = require(sample("jfk.wav"), "samples/jfk.wav") else { return };
    for append_only in [false, true] {
        let mut spec = gpu_spec("stt/voxtral-mini-4b-realtime-q4_k_m");
        spec.options = serde_json::json!({ "appendOnly": append_only });
        let mut e = create(&spec, &EngineContext::headless()).expect("load Voxtral");
        eprintln!("{:?}", e.info());
        assert!(e.info().caps.streaming);
        // The streaming path detects the language itself; a hint must not break it.
        let hint = DecodeOptions {
            language: append_only.then(|| "en".to_string()),
            ..DecodeOptions::default()
        };
        e.begin_stream(&hint).unwrap();
        let mut partials: Vec<String> = Vec::new();
        for chunk in jfk.chunks(SAMPLE_RATE as usize / 5) {
            if let Some(h) = e.push_stream(chunk).unwrap() {
                partials.push(h.text);
            }
        }
        let f = e.end_stream().unwrap();
        eprintln!("append_only {append_only}: {} partials, final {:?}", partials.len(), f.text);
        assert!(!partials.is_empty());
        assert!(f.text.to_lowercase().contains("country"), "{}", f.text);
        if append_only {
            for w in partials.windows(2) {
                assert!(w[1].starts_with(w[0].as_str()), "append-only partials never retract: {w:?}");
            }
        }
        // The engine's stream is mid-utterance only between the first push and the end.
        assert!(!e.transcribe(&jfk[..SAMPLE_RATE as usize], &DecodeOptions::default()).is_err_and(|x| x.kind == AsrErrorKind::Busy));
    }

    // Review P2-D: a reused stream re-checks the VRAM headroom per utterance. With room it stays on
    // the GPU; once the headroom is gone the next utterance moves Voxtral to the CPU (reported) and
    // the stream reopens there.
    let (ctx, seen) = statuses();
    let mut e = TranscribeEngine::create(&gpu_spec("stt/voxtral-mini-4b-realtime-q4_k_m"), &ctx, Options::default()).expect("load Voxtral");
    assert_eq!(e.info().device, "cuda");
    let second = &jfk[..2 * SAMPLE_RATE as usize];
    for _ in 0..2 {
        e.begin_stream(&DecodeOptions::default()).unwrap();
        for chunk in second.chunks(SAMPLE_RATE as usize / 5) {
            e.push_stream(chunk).unwrap();
        }
        e.end_stream().unwrap();
        assert_eq!(e.info().device, "cuda", "room: the reused stream stays on the GPU");
    }
    e.model.set_min_free_vram_mb(4_000_000);
    e.begin_stream(&DecodeOptions::default()).unwrap();
    assert_eq!((e.info().device.as_str(), e.info().vram_mb), ("cpu", 0));
    assert!(e.info().fallback_reason.as_deref().is_some_and(|w| w.contains("VRAM")), "{:?}", e.info());
    assert!(seen.lock().unwrap().iter().any(|s| s.code == codes::ENGINE_FALLBACK));
    for chunk in second.chunks(SAMPLE_RATE as usize / 5) {
        e.push_stream(chunk).unwrap();
    }
    let f = e.end_stream().unwrap();
    eprintln!("Voxtral after the move to the CPU: {:?}", f.text);
    assert!(!f.text.is_empty());
}

/// Cohere Transcribe, Granite TurboCTC and Qwen3 BF16 decode English on the GPU tier; Cohere
/// refuses Japanese. Each model is skipped when not installed.
#[test]
#[ignore = "needs SIGIL_APP_DATA_DIR with the cuda13 pack and any of stt/cohere-transcribe-q4_k, stt/granite-speech-5.0-turboctc-q8_0, stt/qwen3-asr-1.7b-bf16, SIGIL_TEST_MODELS_DIR samples"]
fn real_english_finals_of_the_other_gpu_models() {
    let _one = REAL.lock().unwrap_or_else(|e| e.into_inner());
    let Some(jfk) = require(sample("jfk.wav"), "samples/jfk.wav") else { return };
    for id in ["stt/cohere-transcribe-q4_k", "stt/granite-speech-5.0-turboctc-q8_0", "stt/qwen3-asr-1.7b-bf16"] {
        if !optional(id) {
            continue;
        }
        let before = process_vram_mb();
        let t0 = std::time::Instant::now();
        let mut e = create(&gpu_spec(id), &EngineContext::headless()).expect(id);
        let load_ms = t0.elapsed().as_millis();
        let loaded = process_vram_mb();
        let en = DecodeOptions {
            language: Some("en".into()),
            ..DecodeOptions::default()
        };
        let r = e.transcribe(&jfk, &en).unwrap();
        eprintln!(
            "{id}: {} on {} in {load_ms} ms load, {} ms decode, claimed {} MB, process VRAM {before:?} -> {loaded:?} MiB: {}",
            e.info().runtime,
            e.info().device,
            r.decode_ms,
            e.info().vram_mb,
            r.text
        );
        assert!(r.text.to_lowercase().contains("country"), "{id}: {}", r.text);
        if id.starts_with("stt/cohere-") {
            let ja = DecodeOptions {
                language: Some("ja".into()),
                ..DecodeOptions::default()
            };
            assert_eq!(e.transcribe(&jfk, &ja).unwrap_err().kind, AsrErrorKind::Unsupported);
            assert!(!e.info().caps.languages.iter().any(|l| l == "ja" || l == "zh" || l == "ko"));
            assert_eq!(
                e.transcribe(&jfk, &DecodeOptions::default()).unwrap_err().kind,
                AsrErrorKind::InvalidInput,
                "Cohere needs the language"
            );
        }
    }
}
