//! Tests for the transcribe.cpp wrapper.
//!
//! Always run (no DLL, no model, no network): option / config JSON, result parsing against the
//! engine's real output shapes, variant planning, error mapping, resampling, probe arguments and the
//! embedded catalog entries.
//!
//! Ignored (real runtime + models): `cargo test --manifest-path src-tauri/Cargo.toml
//! speech_rt::transcribe -- --ignored --test-threads=1` with
//! - `SIGIL_TEST_MODELS_DIR` laid out like the dev cache: `models/transcribe/whisper-tiny-Q8_0.gguf`,
//!   `models/transcribe/Qwen3-ASR-1.7B-Q8_0.gguf`, `models/transcribe/ggml-tiny.en.bin`
//!   (whisper-rs coexistence; also looked up at the root and in Sigil's whisper folder) and
//!   `models/transcribe/samples/{jfk,ja-short}.wav` (transcribe.cpp's samples);
//! - packs from `native/engines/transcribe/build.ps1` in `SIGIL_TEST_TRANSCRIBE_PACKS` (default
//!   `%LOCALAPPDATA%\sigil-native\packs`, folders `transcribe-<variant>`: `cpu-avx2` or `cpu`, and
//!   `cuda13` for the GPU tests);
//! - for `cuda13`: `SIGIL_TEST_CUDA_BIN` = a CUDA 13 `bin` folder when the NVIDIA pack is not installed.
//!
//! A missing input FAILS the ignored test (so `-- --ignored` never passes without testing anything);
//! set `SIGIL_TEST_ALLOW_SKIP=1` to skip such tests with a message instead.

use super::engine::{backend_for, combined_kind, falls_through, kind_of_status, plan_variants, stream_closed_error, takeover_reason};
use super::probe;
use super::types::{
    create_config, estimated_model_vram_mb, parse_pci_bus_id, same_pci_bus_id, select_device, vram_shortfall, warmup_seconds, DeviceRef,
};
use super::*;
use crate::services::native_runtime::{abi, RuntimeVariant, VariantSupport};
use crate::services::speech_rt::EngineProbeArgs;
use std::path::{Path, PathBuf};

// ---------------------------------------------------------------------------------------------
// JSON contract
// ---------------------------------------------------------------------------------------------

#[test]
fn create_config_escapes_paths_and_sets_only_given_fields() {
    let mut o = ModelOptions::new(r#"C:\models\qwen "3".gguf"#);
    let v: serde_json::Value = serde_json::from_str(&create_config(&o, None, 2, None)).unwrap();
    assert_eq!(v["model"], r#"C:\models\qwen "3".gguf"#);
    assert!(v.get("backend").is_none(), "variant default backend");
    assert!(v.get("device").is_none());
    assert!(v.get("n_ctx").is_none());
    assert_eq!(v["threads"], 2);
    assert_eq!(v["kv_type"], "auto");

    o.device = Some(1);
    o.n_ctx = Some(4096);
    o.kv_type = KvType::F16;
    let v: serde_json::Value = serde_json::from_str(&create_config(&o, Some("cuda"), 500, None)).unwrap();
    assert_eq!(v["backend"], "cuda");
    assert_eq!(v["device"], 1);
    assert_eq!(v["n_ctx"], 4096);
    assert_eq!(v["kv_type"], "f16");
    assert_eq!(v["threads"], 64, "clamped to the engine's limit");
    // A device index only makes sense with an explicit backend.
    let v: serde_json::Value = serde_json::from_str(&create_config(&o, Some("auto"), 2, None)).unwrap();
    assert!(v.get("device").is_none());
}

#[test]
fn create_config_sends_the_device_only_to_gpu_backends() {
    // The shim counts devices per backend kind and the CPU backend has one: a GPU index on the CPU
    // fallback made the fallback itself fail ("no cpu device matches").
    let mut o = ModelOptions::new("m.gguf");
    o.device = Some(1);
    let resolved = DeviceRef::Id("0000:01:00.0".into());
    for cpu in ["cpu", "cpu_accel", "auto"] {
        let v: serde_json::Value = serde_json::from_str(&create_config(&o, Some(cpu), 2, None)).unwrap();
        assert!(v.get("device").is_none(), "{cpu}: {v}");
        let v: serde_json::Value = serde_json::from_str(&create_config(&o, Some(cpu), 2, Some(&resolved))).unwrap();
        assert!(v.get("device").is_none(), "{cpu} with a resolved id: {v}");
    }
    for gpu in ["cuda", "vulkan"] {
        let v: serde_json::Value = serde_json::from_str(&create_config(&o, Some(gpu), 2, None)).unwrap();
        assert_eq!(v["device"], 1, "{gpu}");
    }
    // A resolved stable id wins over the index (a later GPU backend orders devices differently);
    // an explicit device_id wins over the index too.
    let v: serde_json::Value = serde_json::from_str(&create_config(&o, Some("vulkan"), 2, Some(&resolved))).unwrap();
    assert_eq!(v["device"], "0000:01:00.0");
    o.device_id = Some("  NVIDIA GeForce RTX 3060 ".into());
    assert_eq!(o.device_ref(), Some(DeviceRef::Id("NVIDIA GeForce RTX 3060".into())));
    let v: serde_json::Value = serde_json::from_str(&create_config(&o, Some("cuda"), 2, None)).unwrap();
    assert_eq!(v["device"], "NVIDIA GeForce RTX 3060");
    o.device_id = Some("   ".into());
    assert_eq!(o.device_ref(), Some(DeviceRef::Index(1)), "a blank id falls back to the index");
}

fn device(kind: &str, name: &str, description: &str, id: Option<&str>, free_mb: u64) -> DeviceInfo {
    DeviceInfo {
        name: Some(name.into()),
        description: Some(description.into()),
        kind: Some(kind.into()),
        device_id: id.map(str::to_string),
        device_type: Some("gpu".into()),
        memory_total: 8192 * 1024 * 1024,
        memory_free: free_mb * 1024 * 1024,
    }
}

#[test]
fn devices_are_selected_per_backend_and_resolved_to_a_stable_id() {
    // Vulkan lists the iGPU first; CUDA only sees the NVIDIA cards.
    let all = vec![
        device("cpu", "CPU", "Ryzen", None, 0),
        device("vulkan", "Vulkan0", "AMD Radeon(TM) Graphics", Some("0000:0e:00.0"), 400),
        device("vulkan", "Vulkan1", "NVIDIA GeForce RTX 3060", Some("0000:01:00.0"), 6000),
        device("vulkan", "Vulkan2", "NVIDIA GeForce RTX 4060", Some("0000:0A:00.0"), 7000),
        device("cuda", "CUDA0", "NVIDIA GeForce RTX 3060", Some("0000:01:00.0"), 6000),
        device("cuda", "CUDA1", "NVIDIA GeForce RTX 4060", Some("0000:0a:00.0"), 7000),
    ];
    let second_cuda = select_device(&all, "cuda", Some(&DeviceRef::Index(1))).unwrap();
    assert_eq!(second_cuda.name.as_deref(), Some("CUDA1"));
    // The index is only meaningful on its own backend: the same GPU on Vulkan is found by id
    // (PCI ids compare case-insensitively), never by the Vulkan index 1.
    let stable = DeviceRef::stable_for(&second_cuda).unwrap();
    assert_eq!(stable, DeviceRef::Id("0000:0a:00.0".into()));
    assert_eq!(select_device(&all, "vulkan", Some(&stable)).unwrap().name.as_deref(), Some("Vulkan2"));
    assert_eq!(select_device(&all, "vulkan", Some(&DeviceRef::Index(1))).unwrap().name.as_deref(), Some("Vulkan1"));
    // No preference: the backend's first device; unknown ids / indexes: none (the create fails
    // over instead of picking another GPU).
    assert_eq!(select_device(&all, "cuda", None).unwrap().name.as_deref(), Some("CUDA0"));
    assert!(select_device(&all, "cuda", Some(&DeviceRef::Index(2))).is_none());
    assert!(select_device(&all, "cuda", Some(&DeviceRef::Id("0000:99:00.0".into()))).is_none());
    assert!(select_device(&all, "cuda", Some(&DeviceRef::Id("NVIDIA GeForce RTX 4060".into()))).is_some(), "by description");
    // Without a PCI id the description is the stable id; the backend-specific name never is.
    let no_pci = device("cuda", "CUDA0", "NVIDIA GeForce RTX 3060", None, 1);
    assert_eq!(DeviceRef::stable_for(&no_pci), Some(DeviceRef::Id("NVIDIA GeForce RTX 3060".into())));
    let nothing = DeviceInfo {
        name: Some("CUDA0".into()),
        ..DeviceInfo::default()
    };
    assert_eq!(DeviceRef::stable_for(&nothing), None);
    // The shim's devices answer parses (null entries are devices it could not describe).
    let list: DeviceList = serde_json::from_str(r#"{"devices":[null,{"kind":"cuda","name":"CUDA0","memory_free":5}]}"#).unwrap();
    assert_eq!(list.devices.len(), 2);
    assert!(list.devices[0].is_none());
}

#[test]
fn pci_bus_ids_match_in_every_spelling() {
    // NVML / DXGI write an 8-digit domain, CUDA a 4-digit one, Vulkan may omit it.
    assert_eq!(parse_pci_bus_id("00000000:01:00.0"), Some((0, 1, 0, 0)));
    assert_eq!(parse_pci_bus_id(" 0000:0A:1f.7 "), Some((0, 0x0a, 0x1f, 7)));
    assert_eq!(parse_pci_bus_id("01:00.0"), Some((0, 1, 0, 0)));
    for bad in ["", "NVIDIA GeForce RTX 4090", "0000:01:00", "0000:01:00.10", "0000:001:00.0", "x:01:00.0", "1:2:3:4.0"] {
        assert_eq!(parse_pci_bus_id(bad), None, "{bad:?}");
    }
    assert!(same_pci_bus_id("00000000:01:00.0", "0000:01:00.0"));
    assert!(same_pci_bus_id("0000:0a:00.0", "0A:00.0"));
    assert!(!same_pci_bus_id("0000:01:00.0", "0000:02:00.0"));
    assert!(!same_pci_bus_id("NVIDIA GeForce RTX 4090", "NVIDIA GeForce RTX 4090"), "descriptions are not bus ids");
    // select_device finds a CUDA device from NVML's spelling of its bus id.
    let cuda = vec![device("cuda", "CUDA0", "NVIDIA GeForce RTX 4090", Some("0000:01:00.0"), 20_000)];
    let found = select_device(&cuda, "cuda", Some(&DeviceRef::Id("00000000:01:00.0".into())));
    assert_eq!(found.and_then(|d| d.name).as_deref(), Some("CUDA0"));
}

#[test]
fn vram_checks_keep_gpu_runs_away_from_ggml_cuda_aborts() {
    // Qwen3-ASR-1.7B Q8_0 (2185030624 bytes) measured about 2970 MiB after load + warm-up.
    let qwen = estimated_model_vram_mb(2_185_030_624);
    assert!((2900..=3300).contains(&qwen), "{qwen}");
    assert_eq!(estimated_model_vram_mb(0), 0);
    let rtx3060 = device("cuda", "CUDA0", "NVIDIA GeForce RTX 3060", Some("0000:01:00.0"), 3000);
    let why = vram_shortfall(&rtx3060, qwen + u64::from(DEFAULT_MIN_FREE_VRAM_MB)).unwrap();
    assert!(why.contains("3000 MB of 8192 MB") && why.contains("RTX 3060"), "{why}");
    assert!(vram_shortfall(&rtx3060, 2999).is_none());
    assert!(vram_shortfall(&rtx3060, 0).is_none(), "0 disables the check");
    let unknown = DeviceInfo {
        memory_free: 0,
        ..rtx3060.clone()
    };
    assert!(vram_shortfall(&unknown, 1_000_000).is_none(), "unknown free memory is not a shortfall");
    const { assert!(DEFAULT_MIN_FREE_VRAM_MB >= 256) };
}

#[test]
fn gpu_warmups_cover_the_longest_utterance() {
    // GPU: the VAD cap (the pool is sized at load time), bounded by the model and the shim's range.
    assert_eq!(warmup_seconds(None, true, 0), f64::from(GPU_WARMUP_SECONDS));
    const { assert!(GPU_WARMUP_SECONDS >= 15.0, "at least the VAD cap") };
    assert_eq!(warmup_seconds(None, true, 8_000), 8.0, "clamped to the model's input limit");
    assert_eq!(warmup_seconds(None, false, 0), f64::from(CPU_WARMUP_SECONDS));
    assert_eq!(warmup_seconds(Some(45.0), true, 0), 30.0, "the shim accepts 0.1..=30 s");
    assert!((warmup_seconds(Some(0.01), false, 0) - 0.1).abs() < 1e-9);
    assert_eq!(warmup_seconds(Some(f32::NAN), true, 0), f64::from(GPU_WARMUP_SECONDS));
    assert_eq!(warmup_seconds(Some(-1.0), false, 0), f64::from(CPU_WARMUP_SECONDS));
}

#[test]
fn transcribe_options_json_carries_only_meaningful_fields() {
    let v: serde_json::Value = serde_json::from_str(&TranscribeOptions::default().to_json()).unwrap();
    assert_eq!(v, serde_json::json!({ "timestamps": "segment" }));

    let o = TranscribeOptions {
        language: Some("  ".into()),
        prompt: Some("   ".into()),
        no_speech_threshold: Some(f32::NAN),
        ..Default::default()
    };
    let v: serde_json::Value = serde_json::from_str(&o.to_json()).unwrap();
    assert!(v.get("language").is_none() && v.get("prompt").is_none() && v.get("no_speech_threshold").is_none());

    let o = TranscribeOptions {
        language: Some("ja".into()),
        translate: true,
        target_language: Some("en".into()),
        timestamps: Timestamps::Word,
        words: true,
        punctuation: Toggle::Off,
        itn: Toggle::On,
        prompt: Some(" VRChat, Sigil ".into()),
        temperature_fallback: Some(false),
        no_speech_threshold: Some(1.7),
        commit: Some("stable_prefix".into()),
        agreement: Some(20),
    };
    let v: serde_json::Value = serde_json::from_str(&o.to_json()).unwrap();
    assert_eq!(v["language"], "ja");
    assert_eq!(v["task"], "translate");
    assert_eq!(v["target_language"], "en");
    assert_eq!(v["timestamps"], "word");
    assert_eq!(v["words"], true);
    assert_eq!(v["pnc"], "off");
    assert_eq!(v["itn"], "on");
    assert_eq!(v["prompt"], "VRChat, Sigil");
    assert_eq!(v["temperature_fallback"], false);
    assert_eq!(v["no_speech_threshold"], 1.0);
    assert_eq!(v["commit"], "stable_prefix");
    assert_eq!(v["agreement"], 16, "clamped to the shim's maximum");
    let v: serde_json::Value = serde_json::from_str(
        &TranscribeOptions {
            agreement: Some(0),
            ..Default::default()
        }
        .to_json(),
    )
    .unwrap();
    assert!(v.get("agreement").is_none(), "0 = the engine's default");
}

/// Real outputs of sigil_transcribe.dll 0.2.4-s1 (cpu-avx2, whisper-tiny.en).
const INFO_JSON: &str = r#"{"name":"transcribe","version":"0.2.4-s1","variant":"cpu-avx2","kind":"asr","backend":"CPU","requested_backend":"cpu","upstream_version":"0.2.4","upstream_commit":"unknown","arch":"whisper","model_variant":"whisper-tiny.en","model_name":"","device":{"name":"CPU","description":"AMD Ryzen 7 7800X3D 8-Core Processor","kind":"cpu","device_id":null,"type":"cpu","memory_total":33377124352,"memory_free":14853464064},"capabilities":{"sample_rate":16000,"languages":["en"],"language_detect":false,"translate":false,"streaming":false,"spec_decode":false,"max_timestamps":"segment","max_audio_ms":0,"features":{"initial_prompt":true,"temperature_fallback":true,"long_form":true,"cancellation":true,"pnc":false,"itn":false,"diarization":false}},"limits":{"n_ctx":0,"max_audio_ms":0,"max_kv_bytes":0},"threads":0,"load_ms":77.65399932861328}"#;
const RESULT_JSON: &str = r#"{"text":"And so my fellow Americans ask not what your country can do for you ask what you can do for your country.","language":"","language_source":"none","timestamps":"segment","segments":[{"start_ms":0,"end_ms":8000,"text":"And so my fellow Americans ask not what your country can do for you"},{"start_ms":8000,"end_ms":11000,"text":"ask what you can do for your country."}],"truncated":false,"audio_ms":11000,"wall_ms":279.674,"rtf":0.02542490909090909,"timings":{"load_ms":77.65399932861328,"mel_ms":12.593000411987305,"encode_ms":173.1199951171875,"decode_ms":92.93900299072266,"wall_ms":279.674},"backend":"CPU","some_future_field":{"x":1}}"#;

#[test]
fn engine_json_parses_into_the_public_types() {
    let info: ModelInfo = serde_json::from_str(INFO_JSON).unwrap();
    assert_eq!(info.variant, "cpu-avx2");
    assert_eq!(info.arch.as_deref(), Some("whisper"));
    assert!(!info.on_gpu());
    assert_eq!(info.capabilities.language_codes(), vec!["en"]);
    assert!(info.capabilities.features.initial_prompt);
    assert_eq!(info.max_audio_ms(), 0);
    assert_eq!(info.device.as_ref().unwrap().device_type.as_deref(), Some("cpu"));

    let t: Transcript = serde_json::from_str(RESULT_JSON).unwrap();
    assert!(t.text.starts_with("And so my fellow Americans"));
    assert_eq!(t.segments.len(), 2);
    assert_eq!((t.segments[1].start_ms, t.segments[1].end_ms), (8000, 11000));
    assert!(t.words.is_empty(), "missing words -> empty");
    assert!(!t.truncated);
    assert_eq!(t.backend, "CPU");
    assert!(t.rtf > 0.0 && t.timings.encode_ms > 0.0);

    // Older / partial answers still parse.
    let t: Transcript = serde_json::from_str(r#"{"text":"x"}"#).unwrap();
    assert_eq!(t.text, "x");
    let gpu: ModelInfo = serde_json::from_str(
        r#"{"device":{"kind":"cuda","memory_total":25757220864,"memory_free":1},"limits":{"max_audio_ms":5000},"capabilities":{"max_audio_ms":9000}}"#,
    )
    .unwrap();
    assert!(gpu.on_gpu());
    assert_eq!(gpu.max_audio_ms(), 5000, "the session limit wins over the model limit");
    let u: StreamUpdate = serde_json::from_str(r#"{"partial":"hel","committed":"he","tentative":"l","revision":3}"#).unwrap();
    assert_eq!((u.partial.as_str(), u.revision), ("hel", 3));
}

// ---------------------------------------------------------------------------------------------
// Selection and errors
// ---------------------------------------------------------------------------------------------

#[test]
fn variant_plans_follow_the_backend_choice() {
    use RuntimeVariant::*;
    let all = [Cuda13, Vulkan, CpuAvx512, CpuAvx2, Cpu];
    let full = VariantSupport {
        cuda13: true,
        vulkan: true,
        avx512: true,
        avx2: true,
    };
    assert_eq!(plan_variants(BackendChoice::Auto, None, &all, &full, None), all.to_vec());
    assert_eq!(plan_variants(BackendChoice::Gpu, None, &all, &full, None), vec![Cuda13, Vulkan]);
    assert_eq!(plan_variants(BackendChoice::Cpu, None, &all, &full, None), vec![CpuAvx512, CpuAvx2, Cpu]);

    let no_gpu = VariantSupport {
        avx2: true,
        ..VariantSupport::default()
    };
    assert_eq!(plan_variants(BackendChoice::Auto, None, &all, &no_gpu, None), vec![CpuAvx2, Cpu]);
    assert!(plan_variants(BackendChoice::Gpu, None, &all, &no_gpu, None).is_empty());
    // GPU variants carry AVX2 host kernels: no AVX2, no GPU variant.
    let old_cpu = VariantSupport {
        cuda13: true,
        ..VariantSupport::default()
    };
    assert_eq!(plan_variants(BackendChoice::Auto, None, &all, &old_cpu, None), vec![Cpu]);
    // A forced variant is tried alone (the caller gets the real error), whatever the choice.
    assert_eq!(plan_variants(BackendChoice::Cpu, Some(Cuda13), &all, &no_gpu, None), vec![Cuda13]);
    assert!(plan_variants(BackendChoice::Auto, Some(Vulkan), &[Cpu], &full, None).is_empty());

    // The resource governor's admission removes every other variant (never the unmeasured vulkan
    // build behind its back), including a forced one it did not allow.
    let admitted = [Cuda13, CpuAvx512, CpuAvx2, Cpu];
    assert_eq!(
        plan_variants(BackendChoice::Auto, None, &all, &full, Some(&admitted)),
        vec![Cuda13, CpuAvx512, CpuAvx2, Cpu]
    );
    assert!(plan_variants(BackendChoice::Auto, Some(Vulkan), &all, &full, Some(&admitted)).is_empty());
    let cpu_only = [CpuAvx512, CpuAvx2, Cpu];
    assert_eq!(plan_variants(BackendChoice::Auto, None, &all, &full, Some(&cpu_only)), cpu_only.to_vec());
    // The CPU choice on a PC with only a GPU pack runs that build's CPU backend (GPU builds carry
    // AVX2 CPU kernels), unless the governor forbids the build or the CPU lacks AVX2.
    assert_eq!(plan_variants(BackendChoice::Cpu, None, &[Cuda13], &full, None), vec![Cuda13]);
    assert_eq!(plan_variants(BackendChoice::Cpu, None, &[Cuda13, Vulkan], &no_gpu, None), vec![Cuda13, Vulkan]);
    assert!(plan_variants(BackendChoice::Cpu, None, &[Cuda13], &full, Some(&cpu_only)).is_empty());
    assert!(plan_variants(BackendChoice::Cpu, None, &[Cuda13], &old_cpu, None).is_empty());
    // An installed CPU build always wins over a GPU build's CPU backend.
    assert_eq!(plan_variants(BackendChoice::Cpu, None, &[Cuda13, Cpu], &full, None), vec![Cpu]);
}

#[test]
fn backends_match_variants() {
    use RuntimeVariant::*;
    assert_eq!(backend_for(BackendChoice::Auto, Some(Cuda13)), Some("cuda"));
    assert_eq!(backend_for(BackendChoice::Gpu, Some(Vulkan)), Some("vulkan"));
    assert_eq!(backend_for(BackendChoice::Auto, Some(CpuAvx2)), Some("cpu"));
    assert_eq!(backend_for(BackendChoice::Cpu, Some(Cuda13)), Some("cpu"), "CPU backend of a GPU build");
    assert_eq!(backend_for(BackendChoice::Auto, None), None, "unknown folder: the DLL's own default");
}

#[test]
fn model_level_failures_stop_the_variant_walk() {
    // Backend failures fall through everywhere; out-of-memory and runtime failures only on a GPU
    // (VRAM, a driver problem); a bad / unsupported model or bad options stop at once.
    assert!(falls_through(abi::SIGIL_E_BACKEND_UNAVAILABLE, false));
    assert!(falls_through(abi::SIGIL_E_BACKEND_UNAVAILABLE, true));
    assert!(falls_through(abi::SIGIL_E_OUT_OF_MEMORY, true));
    assert!(!falls_through(abi::SIGIL_E_OUT_OF_MEMORY, false));
    assert!(falls_through(abi::SIGIL_E_RUNTIME, true));
    assert!(!falls_through(abi::SIGIL_E_RUNTIME, false));
    for status in [
        abi::SIGIL_E_LOAD_FAILED,
        abi::SIGIL_E_INVALID_ARGUMENT,
        abi::SIGIL_E_UNSUPPORTED,
        abi::SIGIL_E_CANCELLED,
    ] {
        assert!(!falls_through(status, true), "{status}");
        assert!(!falls_through(status, false), "{status}");
    }

    use ErrorKind::*;
    assert_eq!(combined_kind(&[OutOfMemory, OutOfMemory]), OutOfMemory);
    assert_eq!(combined_kind(&[Backend, OutOfMemory]), OutOfMemory);
    assert_eq!(combined_kind(&[OutOfMemory, Load]), OutOfMemory, "a smaller model may fit the GPU");
    assert_eq!(combined_kind(&[Load, Backend]), Backend);
    assert_eq!(combined_kind(&[Load]), Load);
    assert_eq!(combined_kind(&[]), NotInstalled);
}

#[test]
fn models_and_streams_move_between_threads() {
    // P2 moves models into spawn_blocking / worker threads, and a stream may live on another thread
    // than its model (the fallback closes it through a shared slot).
    fn send<T: Send>() {}
    send::<TranscribeModel>();
    send::<TranscribeStream>();
    fn send_sync<T: Send + Sync + Clone>() {}
    send_sync::<TranscribeCanceller>();
}

#[test]
fn a_cpu_runtime_taking_over_from_a_failed_gpu_is_reported() {
    use RuntimeVariant::*;
    let why = takeover_reason(Some("the cuda13 runtime could not load the model (out of memory)"), false, CpuAvx2).unwrap();
    assert!(why.contains("cuda13") && why.contains("cpu-avx2"), "{why}");
    assert_eq!(takeover_reason(None, false, CpuAvx2), None, "no GPU failed (none installed / supported)");
    assert_eq!(takeover_reason(Some("cuda13 failed"), true, Vulkan), None, "still on a GPU");
    let e = stream_closed_error();
    assert_eq!(e.kind, ErrorKind::Backend);
    assert!(e.message.contains("open a new stream"));
}

#[test]
fn engine_statuses_map_to_error_kinds() {
    assert_eq!(kind_of_status(abi::SIGIL_E_CANCELLED), ErrorKind::Cancelled);
    assert_eq!(kind_of_status(abi::SIGIL_E_BACKEND_UNAVAILABLE), ErrorKind::Backend);
    assert_eq!(kind_of_status(abi::SIGIL_E_OUT_OF_MEMORY), ErrorKind::OutOfMemory);
    assert_eq!(kind_of_status(abi::SIGIL_E_INVALID_ARGUMENT), ErrorKind::InvalidInput);
    assert_eq!(kind_of_status(abi::SIGIL_E_LOAD_FAILED), ErrorKind::Load);
    assert_eq!(kind_of_status(abi::SIGIL_E_UNSUPPORTED), ErrorKind::Unsupported);
    assert_eq!(kind_of_status(abi::SIGIL_E_RUNTIME), ErrorKind::Runtime);
    assert_eq!(kind_of_status(-99), ErrorKind::Runtime);
    let e = TranscribeError::new(ErrorKind::Backend, "CUDA0 lost");
    assert_eq!(e.to_string(), "CUDA0 lost");
    assert_eq!(String::from(e), "CUDA0 lost");
}

#[test]
fn missing_model_or_runtime_is_an_error_not_a_panic() {
    let err = TranscribeModel::open(&ModelOptions::new(r"C:\definitely\missing.gguf")).unwrap_err();
    assert_eq!(err.kind, ErrorKind::Load);

    let dir = std::env::temp_dir().join(format!("sigil-transcribe-test-{}", std::process::id()));
    std::fs::create_dir_all(&dir).unwrap();
    let model = dir.join("m.gguf");
    std::fs::write(&model, b"GGUF").unwrap();
    let mut opts = ModelOptions::new(&model);
    opts.runtime_dir = Some(dir.join("no-such-runtime"));
    assert_eq!(TranscribeModel::open(&opts).unwrap_err().kind, ErrorKind::NotInstalled);
    // A folder without the DLL.
    opts.runtime_dir = Some(dir.clone());
    let err = TranscribeModel::open(&opts).unwrap_err();
    assert_eq!(err.kind, ErrorKind::Load, "{err}");
    // Installed packs: a GPU request limited to a CPU build is refused like a CPU runtime folder
    // (never a silent CPU run), before any pack is looked at.
    if std::env::var_os(RUNTIME_DIR_ENV).is_none() {
        let mut gpu_on_cpu_build = ModelOptions::new(&model);
        gpu_on_cpu_build.backend = BackendChoice::Gpu;
        gpu_on_cpu_build.variant = Some(RuntimeVariant::CpuAvx2);
        let err = TranscribeModel::open(&gpu_on_cpu_build).unwrap_err();
        assert_eq!(err.kind, ErrorKind::Backend, "{err}");
    }
    let _ = std::fs::remove_dir_all(&dir);
}

// ---------------------------------------------------------------------------------------------
// Audio and probe
// ---------------------------------------------------------------------------------------------

#[test]
fn audio_is_resampled_to_16k_mono_with_exact_length() {
    let passthrough = to_16k_mono(&[0.1, -0.2, 0.3], 16_000, 1).unwrap();
    assert_eq!(passthrough, vec![0.1, -0.2, 0.3]);

    // 1 s of a 440 Hz tone, 48 kHz stereo -> 16000 mono samples with the same tone.
    let frames = 48_000;
    let mut stereo = Vec::with_capacity(frames * 2);
    for i in 0..frames {
        let s = (2.0 * std::f32::consts::PI * 440.0 * i as f32 / 48_000.0).sin() * 0.5;
        stereo.push(s);
        stereo.push(s);
    }
    let mono = to_16k_mono(&stereo, 48_000, 2).unwrap();
    assert_eq!(mono.len(), 16_000);
    let rms = (mono[2000..14000].iter().map(|x| x * x).sum::<f32>() / 12_000.0).sqrt();
    assert!((rms - 0.5 / 2f32.sqrt()).abs() < 0.02, "rms {rms}");

    assert_eq!(to_16k_mono(&[0.0], 0, 1).unwrap_err().kind, ErrorKind::InvalidInput);
    assert_eq!(to_16k_mono(&[0.0], 16_000, 0).unwrap_err().kind, ErrorKind::InvalidInput);
}

fn probe_args(extra: &[&str]) -> EngineProbeArgs {
    EngineProbeArgs {
        engine: "transcribe".into(),
        model: Some(PathBuf::from("m.gguf")),
        wav: None,
        variant: Some("cuda13".into()),
        runtime_dir: Some(PathBuf::from("rt")),
        extra: extra.iter().map(|s| s.to_string()).collect(),
    }
}

#[test]
fn probe_arguments_parse() {
    let p = probe::parse(&probe_args(&[
        "--repeat",
        "5",
        "--backend",
        "gpu",
        "--threads",
        "3",
        "--n-ctx",
        "4096",
        "--kv",
        "f16",
        "--language",
        "ja",
        "--timestamps",
        "none",
        "--no-temperature-fallback",
        "--stream",
        "--whisper-rs",
        "w.bin",
        "--no-warmup",
        "--no-cpu-fallback",
        "--cancel-after-ms",
        "40",
        "--ort-canary",
        "directml",
        "--device-id",
        "0000:01:00.0",
        "--warmup-seconds",
        "15",
        "--min-free-vram-mb",
        "0",
    ]))
    .unwrap();
    assert_eq!(p.repeat, 5);
    assert_eq!(p.model.backend, BackendChoice::Gpu);
    assert_eq!(p.model.variant, Some(RuntimeVariant::Cuda13));
    assert_eq!(p.model.runtime_dir.as_deref(), Some(Path::new("rt")));
    assert_eq!((p.model.threads, p.model.n_ctx, p.model.kv_type), (Some(3), Some(4096), KvType::F16));
    assert!(!p.model.warmup && !p.model.cpu_fallback);
    assert_eq!(p.transcribe.language.as_deref(), Some("ja"));
    assert_eq!(p.transcribe.timestamps, Timestamps::None);
    assert_eq!(p.transcribe.temperature_fallback, Some(false));
    assert!(p.stream);
    assert_eq!(p.whisper_rs.as_deref(), Some(Path::new("w.bin")));
    assert_eq!(p.cancel_after_ms, Some(40));
    assert_eq!(p.ort_canary, Some("dml"));
    assert_eq!(p.model.device_id.as_deref(), Some("0000:01:00.0"));
    assert_eq!(p.model.warmup_seconds, Some(15.0));
    assert_eq!(p.model.min_free_vram_mb, 0);
    assert_eq!(probe::parse(&probe_args(&[])).unwrap().model.min_free_vram_mb, DEFAULT_MIN_FREE_VRAM_MB);
    assert!(probe::parse(&probe_args(&["--ort-canary", "tpu"])).is_err());

    assert!(probe::parse(&probe_args(&["--repeat", "0"])).is_err());
    assert!(probe::parse(&probe_args(&["--repeat"])).is_err());
    assert!(probe::parse(&probe_args(&["--backend", "tpu"])).is_err());
    assert!(probe::parse(&probe_args(&["--bogus"])).is_err());
    let mut no_model = probe_args(&[]);
    no_model.model = None;
    assert!(probe::parse(&no_model).is_err());
    let mut bad_variant = probe_args(&[]);
    bad_variant.variant = Some("cuda12".into());
    assert!(probe::parse(&bad_variant).is_err());
}

#[test]
fn probe_reports_errors_as_err() {
    let mut args = probe_args(&[]);
    args.model = Some(PathBuf::from(r"C:\definitely\missing.gguf"));
    assert!(headless_probe(&args).is_err());
    args.wav = Some(PathBuf::from(r"C:\definitely\missing.wav"));
    assert!(headless_probe(&args).unwrap_err().contains("missing.wav"));
}

#[test]
fn wav_reader_handles_int_and_float() {
    let dir = std::env::temp_dir().join(format!("sigil-transcribe-wav-{}", std::process::id()));
    std::fs::create_dir_all(&dir).unwrap();
    let int_path = dir.join("i16.wav");
    let spec = hound::WavSpec {
        channels: 2,
        sample_rate: 44_100,
        bits_per_sample: 16,
        sample_format: hound::SampleFormat::Int,
    };
    let mut w = hound::WavWriter::create(&int_path, spec).unwrap();
    for s in [16384i16, -16384, 32767, -32768] {
        w.write_sample(s).unwrap();
    }
    w.finalize().unwrap();
    let (samples, rate, channels) = probe::read_wav(&int_path).unwrap();
    assert_eq!((rate, channels), (44_100, 2));
    assert_eq!(samples.len(), 4);
    assert!((samples[0] - 0.5).abs() < 1e-6 && (samples[3] + 1.0).abs() < 1e-6);

    let f_path = dir.join("f32.wav");
    let spec = hound::WavSpec {
        channels: 1,
        sample_rate: 16_000,
        bits_per_sample: 32,
        sample_format: hound::SampleFormat::Float,
    };
    let mut w = hound::WavWriter::create(&f_path, spec).unwrap();
    w.write_sample(0.25f32).unwrap();
    w.finalize().unwrap();
    assert_eq!(probe::read_wav(&f_path).unwrap().0, vec![0.25]);
    let _ = std::fs::remove_dir_all(&dir);
}

// ---------------------------------------------------------------------------------------------
// Catalog
// ---------------------------------------------------------------------------------------------

#[test]
fn catalog_lists_every_transcribe_variant_as_a_local_pack() {
    use crate::services::model_store::{Catalog, EntryKind, PublishStatus};
    let catalog = Catalog::from_json(crate::services::model_store::EMBEDDED_CATALOG_JSON).unwrap();
    let packs = crate::services::native_runtime::engine_pack_entries(&catalog, Some(ENGINE));
    let variants: Vec<RuntimeVariant> = packs.iter().map(|p| p.variant).collect();
    for v in RuntimeVariant::PREFERENCE {
        assert!(variants.contains(&v), "runtime/sigil-transcribe-{} missing", v.as_str());
    }
    for p in &packs {
        assert!(!p.published, "{} must stay unpublished (local build)", p.id);
        let entry = catalog.get(&p.id).unwrap();
        assert_eq!(entry.kind, EntryKind::Runtime);
        assert_eq!(entry.status, PublishStatus::Unpublished);
        assert_eq!(entry.files.len(), 1);
        let archive = entry.files[0].archive.as_ref().expect("zip pack");
        assert!(archive.include.iter().any(|g| g == "sigil-pack.json"));
        assert!(entry.files[0]
            .path
            .ends_with(&format!("-{}.zip", p.variant.as_str())));
    }
    // Bench models of the spike are pinned.
    for id in ["bench/transcribe-whisper-tiny-q8_0", "bench/transcribe-qwen3-asr-1.7b-q8_0"] {
        let e = catalog.get(id).unwrap_or_else(|| panic!("{id} missing"));
        assert!(e.files.iter().all(|f| f.sha256.is_some() && f.size > 0), "{id} must be hash-pinned");
    }
    // Measured (ADR 0003b): about 3.5 GB of process VRAM for Qwen3-ASR-1.7B Q8_0 while loaded and
    // running, plus about 0.4 GB of CUDA context.
    let qwen = catalog.get("bench/transcribe-qwen3-asr-1.7b-q8_0").unwrap();
    assert!(qwen.min_vram_mb.is_some_and(|mb| mb >= 3500 + 378), "{:?}", qwen.min_vram_mb);
}

#[test]
fn catalog_versions_follow_the_engine_pin() {
    // engine.json = <UPSTREAM.json version>-s<shim_revision> = every pack entry's version and zip.
    let engine: serde_json::Value = serde_json::from_str(include_str!("../../../../../native/engines/transcribe/engine.json")).unwrap();
    let upstream: serde_json::Value = serde_json::from_str(include_str!("../../../../../native/engines/transcribe/UPSTREAM.json")).unwrap();
    let version = engine["version"].as_str().unwrap();
    assert_eq!(
        version,
        format!("{}-s{}", upstream["version"].as_str().unwrap(), upstream["shim_revision"].as_u64().unwrap())
    );
    let catalog = crate::services::model_store::Catalog::from_json(crate::services::model_store::EMBEDDED_CATALOG_JSON).unwrap();
    for p in crate::services::native_runtime::engine_pack_entries(&catalog, Some(ENGINE)) {
        let entry = catalog.get(&p.id).unwrap();
        assert_eq!(entry.version, version, "{}", p.id);
        assert_eq!(entry.files[0].path, format!("sigil-transcribe-{version}-{}.zip", p.variant.as_str()));
    }
}

// ---------------------------------------------------------------------------------------------
// Real runtime + models (ignored)
// ---------------------------------------------------------------------------------------------

fn models_dir() -> Option<PathBuf> {
    std::env::var_os("SIGIL_TEST_MODELS_DIR")
        .filter(|v| !v.is_empty())
        .map(PathBuf::from)
}

/// A missing test input: a panic, so `-- --ignored` never reports a pass without testing
/// anything, unless `SIGIL_TEST_ALLOW_SKIP` is set (then the test skips with a message).
fn require<T>(value: Option<T>, what: &str) -> Option<T> {
    if value.is_some() {
        return value;
    }
    if std::env::var_os("SIGIL_TEST_ALLOW_SKIP").is_some_and(|v| !v.is_empty()) {
        eprintln!("{what} missing; skipped (SIGIL_TEST_ALLOW_SKIP)");
        return None;
    }
    panic!(
        "{what} missing (SIGIL_TEST_MODELS_DIR = {:?}, packs in {}): see the top of speech_rt/transcribe/tests.rs, \
         or set SIGIL_TEST_ALLOW_SKIP=1 to skip",
        models_dir(),
        packs_dir().display()
    );
}

fn packs_dir() -> PathBuf {
    std::env::var_os("SIGIL_TEST_TRANSCRIBE_PACKS")
        .map(PathBuf::from)
        .unwrap_or_else(|| {
            PathBuf::from(std::env::var_os("LOCALAPPDATA").unwrap_or_default())
                .join("sigil-native")
                .join("packs")
        })
}

fn pack(variant: &str) -> Option<PathBuf> {
    let dir = packs_dir().join(format!("transcribe-{variant}"));
    dir.join("sigil_transcribe.dll").is_file().then_some(dir)
}

fn test_file(rel: &str) -> Option<PathBuf> {
    let p = models_dir()?.join("models").join("transcribe").join(rel);
    p.is_file().then_some(p)
}

fn wav_16k(rel: &str) -> Option<Vec<f32>> {
    let (s, r, c) = probe::read_wav(&test_file(rel)?).ok()?;
    to_16k_mono(&s, r, c).ok()
}

fn whisper_bin() -> Option<PathBuf> {
    let mut candidates = vec![];
    if let Some(p) = test_file("ggml-tiny.en.bin") {
        candidates.push(p);
    }
    if let Some(d) = models_dir() {
        candidates.push(d.join("ggml-tiny.en.bin"));
    }
    if let Some(a) = std::env::var_os("APPDATA") {
        candidates.push(
            PathBuf::from(a)
                .join("com.sigil.app")
                .join("whisper")
                .join("ggml-tiny.en.bin"),
        );
    }
    candidates.into_iter().find(|p| p.is_file())
}

fn cpu_pack() -> Option<PathBuf> {
    require(pack("cpu-avx2").or_else(|| pack("cpu")), "a cpu-avx2 / cpu pack")
}

fn open_cpu(model: &Path) -> Option<TranscribeModel> {
    let mut o = ModelOptions::new(model);
    o.runtime_dir = Some(cpu_pack()?);
    Some(TranscribeModel::open(&o).expect("open on the CPU pack"))
}

#[test]
#[ignore = "needs SIGIL_TEST_MODELS_DIR (models/transcribe) and a cpu / cpu-avx2 pack"]
fn real_whisper_tiny_on_cpu() {
    let Some(model) = require(test_file("whisper-tiny-Q8_0.gguf"), "models/transcribe/whisper-tiny-Q8_0.gguf") else {
        return;
    };
    let Some(jfk) = require(wav_16k("samples/jfk.wav"), "models/transcribe/samples/jfk.wav") else {
        return;
    };
    let Some(mut m) = open_cpu(&model) else {
        return;
    };
    assert_eq!(m.info().arch.as_deref(), Some("whisper"));
    assert!(!m.info().on_gpu());
    assert!(m.fallback_reason().is_none());
    // A CPU build refuses an explicit GPU request instead of silently running on the CPU.
    let mut gpu = ModelOptions::new(&model);
    gpu.runtime_dir = m.runtime().dir.clone().into();
    gpu.backend = BackendChoice::Gpu;
    assert_eq!(TranscribeModel::open(&gpu).unwrap_err().kind, ErrorKind::Backend);
    // A GPU index never reaches the CPU backend (it has one device).
    let mut indexed = ModelOptions::new(&model);
    indexed.runtime_dir = m.runtime().dir.clone().into();
    indexed.device = Some(3);
    indexed.warmup = false;
    assert!(TranscribeModel::open(&indexed).is_ok(), "device index on a CPU build");
    let t = m
        .transcribe(
            &jfk,
            &TranscribeOptions {
                language: Some("en".into()),
                ..Default::default()
            },
        )
        .unwrap();
    let text = t.text.to_lowercase();
    assert!(text.contains("ask not what your country can do for you"), "{text}");
    assert_eq!(t.language_source, "hint");
    assert!(!t.segments.is_empty() && t.segments.last().unwrap().end_ms > 9000);
    assert!(t.rtf < 1.0, "whisper-tiny slower than real time on this CPU: {}", t.rtf);
    eprintln!("whisper-tiny cpu: rtf {:.4} ({:.0} ms for {:.0} ms)", t.rtf, t.wall_ms, t.audio_ms);

    // Auto language detection on the multilingual model.
    let t = m.transcribe(&jfk, &TranscribeOptions::default()).unwrap();
    assert_eq!((t.language.as_str(), t.language_source.as_str()), ("en", "detected"));
    // Offline models refuse streaming with a typed error; bad input is InvalidInput.
    assert_eq!(
        m.open_stream(&TranscribeOptions::default())
            .unwrap_err()
            .kind,
        ErrorKind::Unsupported
    );
    assert_eq!(
        m.transcribe(&[], &TranscribeOptions::default())
            .unwrap_err()
            .kind,
        ErrorKind::InvalidInput
    );
    let bad = TranscribeOptions {
        language: Some("xx-not-a-language".into()),
        ..Default::default()
    };
    assert_eq!(m.transcribe(&jfk, &bad).unwrap_err().kind, ErrorKind::InvalidInput);
    // The same model still works after errors.
    assert!(m
        .transcribe(&jfk, &TranscribeOptions::default())
        .unwrap()
        .text
        .to_lowercase()
        .contains("country"));

    // A model the runtime cannot load is a model-level failure: a typed Load error, no fallback.
    let dir = std::env::temp_dir().join(format!("sigil-transcribe-bad-{}", std::process::id()));
    std::fs::create_dir_all(&dir).unwrap();
    let bogus = dir.join("not-a-model.gguf");
    std::fs::write(&bogus, b"GGUF but not really a model").unwrap();
    let mut bad_model = ModelOptions::new(&bogus);
    bad_model.runtime_dir = m.runtime().dir.clone().into();
    let err = TranscribeModel::open(&bad_model).unwrap_err();
    assert_eq!(err.kind, ErrorKind::Load, "{err}");
    let _ = std::fs::remove_dir_all(&dir);
}

#[test]
#[ignore = "needs SIGIL_TEST_MODELS_DIR (whisper-tiny GGUF, ggml-tiny.en.bin, jfk.wav) and a CPU pack"]
fn coexists_with_whisper_rs_in_one_process() {
    use whisper_rs::{FullParams, SamplingStrategy, WhisperContext, WhisperContextParameters};
    let Some(model) = require(test_file("whisper-tiny-Q8_0.gguf"), "models/transcribe/whisper-tiny-Q8_0.gguf") else {
        return;
    };
    let Some(bin) = require(whisper_bin(), "ggml-tiny.en.bin (whisper-rs)") else {
        return;
    };
    let Some(jfk) = require(wav_16k("samples/jfk.wav"), "models/transcribe/samples/jfk.wav") else {
        return;
    };
    let run_whisper_rs = |ctx: &WhisperContext| -> String {
        let mut state = ctx.create_state().unwrap();
        let mut p = FullParams::new(SamplingStrategy::Greedy { best_of: 1 });
        p.set_n_threads(4);
        p.set_language(Some("en"));
        p.set_print_progress(false);
        p.set_print_realtime(false);
        state.full(p, &jfk).unwrap();
        state
            .as_iter()
            .filter_map(|s| s.to_str_lossy().ok().map(|t| t.into_owned()))
            .collect::<String>()
    };
    // whisper-rs (static ggml inside this test binary) first, then the engine DLL's own static ggml.
    let ctx = WhisperContext::new_with_params(&bin, WhisperContextParameters::default()).unwrap();
    let before = run_whisper_rs(&ctx);
    let Some(mut m) = open_cpu(&model) else {
        return;
    };
    let engine_text = m
        .transcribe(&jfk, &TranscribeOptions::default())
        .unwrap()
        .text;
    // Interleave: both runtimes keep working with their own ggml state.
    let after = run_whisper_rs(&ctx);
    let engine_again = m
        .transcribe(&jfk, &TranscribeOptions::default())
        .unwrap()
        .text;
    for t in [&before, &after, &engine_text, &engine_again] {
        assert!(t.to_lowercase().contains("ask not what your country"), "{t}");
    }
    assert_eq!(before, after, "whisper-rs output changed after the engine DLL loaded");
    assert_eq!(engine_text, engine_again);
    // No ggml DLL was loaded by name: both copies are static.
    #[cfg(windows)]
    for name in ["ggml.dll", "ggml-base.dll", "ggml-cpu.dll"] {
        use windows::core::PCWSTR;
        use windows::Win32::System::LibraryLoader::GetModuleHandleW;
        let wide: Vec<u16> = name.encode_utf16().chain(std::iter::once(0)).collect();
        // SAFETY: NUL-terminated wide string.
        assert!(unsafe { GetModuleHandleW(PCWSTR(wide.as_ptr())) }.is_err(), "{name} is loaded");
    }
    drop(m);
    // whisper-rs still works after the engine model is gone.
    assert_eq!(run_whisper_rs(&ctx), before);
}

#[test]
#[ignore = "needs SIGIL_TEST_MODELS_DIR (whisper-tiny GGUF, samples/ja-short.wav) and a CPU pack"]
fn cancel_interrupts_a_running_decode() {
    let Some(model) = require(test_file("whisper-tiny-Q8_0.gguf"), "models/transcribe/whisper-tiny-Q8_0.gguf") else {
        return;
    };
    let Some(ja) = require(wav_16k("samples/ja-short.wav"), "models/transcribe/samples/ja-short.wav") else {
        return;
    };
    let Some(dir) = cpu_pack() else {
        return;
    };
    // One thread and four repetitions of the clip (44 s, two 30 s windows; tiny on Japanese also
    // re-decodes at higher temperatures): the decode takes seconds, far longer than the cancel delay.
    let mut o = ModelOptions::new(&model);
    o.runtime_dir = Some(dir);
    o.threads = Some(1);
    let mut m = TranscribeModel::open(&o).expect("open on the CPU pack");
    let long: Vec<f32> = ja.iter().copied().cycle().take(ja.len() * 4).collect();
    let started = std::time::Instant::now();
    let full = m.transcribe(&long, &TranscribeOptions::default()).unwrap();
    let full_ms = started.elapsed().as_secs_f64() * 1000.0;
    const CANCEL_AFTER_MS: u64 = 300;
    assert!(
        full_ms > 2.0 * CANCEL_AFTER_MS as f64,
        "the uncancelled decode took only {full_ms:.0} ms; it must be well over {CANCEL_AFTER_MS} ms to prove anything"
    );
    assert!(!full.text.is_empty());

    let canceller = m.canceller();
    let timer = std::thread::spawn(move || {
        std::thread::sleep(std::time::Duration::from_millis(CANCEL_AFTER_MS));
        canceller.cancel();
    });
    let started = std::time::Instant::now();
    let r = m.transcribe(&long, &TranscribeOptions::default());
    let cancelled_ms = started.elapsed().as_secs_f64() * 1000.0;
    timer.join().unwrap();
    match r {
        Err(e) => assert_eq!(e.kind, ErrorKind::Cancelled, "{e}"),
        Ok(_) => panic!("the decode ignored the cancel at {CANCEL_AFTER_MS} ms and ran to the end ({cancelled_ms:.0} ms)"),
    }
    assert!(
        cancelled_ms < full_ms * 0.8,
        "the cancel took effect only after {cancelled_ms:.0} ms of a {full_ms:.0} ms decode"
    );
    eprintln!("cancelled after {cancelled_ms:.0} ms (uncancelled: {full_ms:.0} ms)");
    // A cancel does not poison the next call.
    let t = m.transcribe(&ja, &TranscribeOptions::default()).unwrap();
    assert!(!t.text.is_empty());
}

fn cuda_pack() -> Option<PathBuf> {
    require(pack("cuda13"), "a cuda13 pack")
}

#[test]
#[ignore = "needs SIGIL_TEST_MODELS_DIR (whisper-tiny GGUF, samples/jfk.wav), a cuda13 pack and SIGIL_TEST_CUDA_BIN or the NVIDIA pack"]
fn gpu_failures_and_low_vram_fall_back_to_the_cpu() {
    let Some(model) = require(test_file("whisper-tiny-Q8_0.gguf"), "models/transcribe/whisper-tiny-Q8_0.gguf") else {
        return;
    };
    let Some(jfk) = require(wav_16k("samples/jfk.wav"), "models/transcribe/samples/jfk.wav") else {
        return;
    };
    let Some(dir) = cuda_pack() else {
        return;
    };
    let base = || {
        let mut o = ModelOptions::new(&model);
        o.runtime_dir = Some(dir.clone());
        o
    };
    let check_cpu_transcript = |m: &mut TranscribeModel| {
        assert!(!m.info().on_gpu(), "{:?}", m.info().device);
        let t = m.transcribe(&jfk, &TranscribeOptions::default()).unwrap();
        assert!(t.text.to_lowercase().contains("country"), "{}", t.text);
    };

    // 1. A GPU index this PC does not have: the CUDA load fails over to the CPU backend, which
    //    must not receive the GPU index (it made the CPU fallback fail as well).
    let mut o = base();
    o.device = Some(7);
    let mut m = TranscribeModel::open(&o).expect("CPU fallback after a bad GPU index");
    let why = m.fallback_reason().expect("a fallback reason").to_string();
    assert!(why.contains("device"), "{why}");
    assert!(m.runtime().gpu_backend_initialized, "the CUDA backend was initialized: its context stays");
    check_cpu_transcript(&mut m);
    drop(m);

    // 2. Not enough free VRAM for the model plus the headroom: refused before anything loads (the
    //    device query already initialized CUDA, so the engine keeps claiming its context).
    let mut o = base();
    o.min_free_vram_mb = 4_000_000;
    let mut m = TranscribeModel::open(&o).expect("CPU fallback for a GPU without room");
    let why = m.fallback_reason().expect("a fallback reason").to_string();
    assert!(why.contains("VRAM"), "{why}");
    assert!(m.runtime().gpu_backend_initialized);
    check_cpu_transcript(&mut m);
    assert!(m.check_headroom().is_ok(), "no check on the CPU");
    drop(m);
    // The CPU backend of the same DLL never touches the GPU.
    let mut cpu = base();
    cpu.backend = BackendChoice::Cpu;
    let mut m = TranscribeModel::open(&cpu).expect("CPU backend of the cuda13 DLL");
    assert!(!m.runtime().gpu_backend_initialized);
    check_cpu_transcript(&mut m);
    drop(m);
    o.cpu_fallback = false;
    assert_eq!(TranscribeModel::open(&o).unwrap_err().kind, ErrorKind::OutOfMemory);

    // 3. On CUDA with room; the headroom disappears later: the next run moves to the CPU instead of
    //    starting a GPU run that could abort the process. Without cpu_fallback the run is refused.
    let mut m = TranscribeModel::open(&base()).expect("open on CUDA");
    assert!(m.info().on_gpu(), "{:?}", m.info().device);
    assert!(m.fallback_reason().is_none());
    let canceller = m.canceller();
    m.set_min_free_vram_mb(4_000_000);
    let t = m.transcribe(&jfk, &TranscribeOptions::default()).unwrap();
    assert!(t.text.to_lowercase().contains("country"), "{}", t.text);
    assert_eq!(t.backend, "CPU", "the run itself went to the CPU copy");
    check_cpu_transcript(&mut m);
    let why = m.fallback_reason().expect("a fallback reason").to_string();
    assert!(why.contains("VRAM") && why.contains("CPU"), "{why}");
    canceller.cancel(); // follows the model to its CPU instance (no call running: a no-op)
    assert!(!m
        .transcribe(&jfk, &TranscribeOptions::default())
        .unwrap()
        .text
        .is_empty());
    drop(m);
    let mut strict = base();
    strict.cpu_fallback = false;
    let mut m = TranscribeModel::open(&strict).expect("open on CUDA");
    assert!(m.check_headroom().is_ok(), "room: the per-utterance check passes");
    m.set_min_free_vram_mb(4_000_000);
    assert_eq!(
        m.transcribe(&jfk, &TranscribeOptions::default())
            .unwrap_err()
            .kind,
        ErrorKind::OutOfMemory
    );
    assert_eq!(m.check_headroom().unwrap_err().kind, ErrorKind::OutOfMemory);
    assert!(m.info().on_gpu(), "no silent move without cpu_fallback");
    drop(m);
    // With the CPU fallback, the per-utterance check (what a reused stream runs) moves the model.
    let mut m = TranscribeModel::open(&base()).expect("open on CUDA");
    m.set_min_free_vram_mb(4_000_000);
    m.check_headroom().expect("moved to the CPU");
    assert!(!m.info().on_gpu());
    assert!(m.fallback_reason().is_some_and(|w| w.contains("VRAM")));
}

#[test]
#[ignore = "needs SIGIL_TEST_MODELS_DIR (Qwen3-ASR-1.7B-Q8_0.gguf, samples/ja-short.wav), a cuda13 pack and SIGIL_TEST_CUDA_BIN or the NVIDIA pack"]
fn real_qwen3_asr_on_cuda() {
    let Some(model) = require(test_file("Qwen3-ASR-1.7B-Q8_0.gguf"), "models/transcribe/Qwen3-ASR-1.7B-Q8_0.gguf") else {
        return;
    };
    let Some(ja) = require(wav_16k("samples/ja-short.wav"), "models/transcribe/samples/ja-short.wav") else {
        return;
    };
    let Some(dir) = cuda_pack() else {
        return;
    };
    let mut o = ModelOptions::new(&model);
    o.runtime_dir = Some(dir);
    o.backend = BackendChoice::Gpu;
    o.cpu_fallback = false;
    o.n_ctx = Some(4096);
    let mut m = TranscribeModel::open(&o).expect("open on CUDA");
    assert!(m.info().on_gpu(), "{:?}", m.info().device);
    assert!(m.info().backend.to_lowercase().contains("cuda"), "{}", m.info().backend);
    let mut rtfs = Vec::new();
    let mut text = String::new();
    for _ in 0..3 {
        let t = m.transcribe(&ja, &TranscribeOptions::default()).unwrap();
        rtfs.push(t.rtf);
        text = t.text;
    }
    assert!(text.chars().count() > 10, "{text}");
    assert!(
        text.chars()
            .any(|c| ('\u{3040}'..='\u{30ff}').contains(&c) || ('\u{4e00}'..='\u{9fff}').contains(&c)),
        "{text}"
    );
    assert!(
        rtfs.iter().all(|r| *r < 0.5),
        "Qwen3-ASR-1.7B on CUDA should be well under real time: {rtfs:?}"
    );
    eprintln!("qwen3-asr-1.7b q8_0 cuda: rtf {rtfs:?}, text {text}");

    // Unloading returns the model's VRAM (switching STT models must not leak ~2 GB each time).
    let nvidia = crate::services::diagnostics::gpu_mem::adapters()
        .into_iter()
        .find(|a| a.vendor_id == crate::services::diagnostics::gpu_mem::VENDOR_NVIDIA && !a.software)
        .map(|a| a.index);
    let usage = || {
        nvidia
            .and_then(crate::services::diagnostics::gpu_mem::process_vram_usage)
            .map(|u| u.process_usage_mb)
    };
    let loaded = usage();
    drop(m);
    let unloaded = usage();
    let mut again = TranscribeModel::open(&o).expect("reopen on CUDA");
    assert!(!again
        .transcribe(&ja, &TranscribeOptions::default())
        .unwrap()
        .text
        .is_empty());
    let reloaded = usage();
    drop(again);
    eprintln!(
        "process VRAM MiB: loaded {loaded:?}, after drop {unloaded:?}, reloaded {reloaded:?}, after second drop {:?}",
        usage()
    );
    if let (Some(l), Some(u), Some(r)) = (loaded, unloaded, reloaded) {
        assert!(u + 1500 < l, "dropping the model freed too little VRAM ({l} -> {u} MiB)");
        assert!(r <= l + 256, "reloading grew VRAM ({l} -> {r} MiB)");
    }
}
