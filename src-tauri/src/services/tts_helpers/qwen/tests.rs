//! Tests of the Qwen3-TTS helper engine: the plan table, the catalog fragments, resolution against
//! installed files, the helper spec, the load-log parser and the clone cache format (pure), plus
//! lifecycle and cloning tests with a real child process: the "sigil-tts-server" is this test binary
//! re-run as a fake (an HTTP server with the real server's routes, key checks, info, latents and
//! load log lines), started through the real `server::start_launch` → `process_host` path. Nothing
//! touches the network.

use super::clone::{self, CacheEntry, CACHE_FORMAT, CACHE_VERSION, MAX_CACHE_ENTRIES};
use super::plan::{self, Knobs, Mode, QwenConfig, Size, Variant};
use super::server::{self, build_spec, resolve_with, test_hooks, Launch};
use super::*;
use crate::services::diagnostics::gpu_mem::{VENDOR_AMD, VENDOR_NVIDIA};
use crate::services::model_store::Catalog;
use crate::services::process_host::{self, HelperLogLine, HelperPriority, HelperState, PortRequest, RestartPolicy};
use crate::services::resources::{self, Recommendation, RecommendationTopic, ResourceSnapshot};
use crate::services::sysinfo::test_support::{gpu, system};
use crate::services::sysinfo::SystemInfo;
use crate::services::tts_helpers::TtsHelperState;
use base64::Engine as _;
use std::collections::{HashMap, HashSet};
use std::io::{Read, Write};
use std::net::TcpListener;
use std::path::{Path, PathBuf};
use std::time::{Duration, Instant};

// ---------------------------------------------------------------------------------------------
// Fixtures
// ---------------------------------------------------------------------------------------------

fn nvidia(vram_mb: u64) -> SystemInfo {
    system(vec![gpu(0, VENDOR_NVIDIA, "NVIDIA GeForce RTX 4060", vram_mb, Some("8.9"), Some("ada"))])
}

fn amd(vram_mb: u64) -> SystemInfo {
    system(vec![gpu(0, VENDOR_AMD, "AMD Radeon RX 7800 XT", vram_mb, None, None)])
}

fn no_gpu() -> SystemInfo {
    system(Vec::new())
}

fn rec(choice: &str, alternatives: &[&str], device: &str) -> Recommendation {
    Recommendation {
        topic: RecommendationTopic::Qwen3Tts,
        choice: choice.into(),
        reason: "test".into(),
        alternatives: alternatives.iter().map(|s| s.to_string()).collect(),
        reason_code: "fits".into(),
        device: device.into(),
        vram_mb: 0,
        available_mb: 0,
    }
}

fn knobs(size: Option<Size>, variant: Option<Variant>, mode: Mode) -> Knobs {
    Knobs {
        size,
        variant,
        mode,
        language: "auto",
    }
}

fn run_plan(k: &Knobs, sys: &SystemInfo, r: &Recommendation, installed: &[&str]) -> plan::Plan {
    let set: HashSet<String> = installed.iter().map(|s| s.to_string()).collect();
    let is = |id: &str| set.contains(id);
    let footprint = |id: &str| match id {
        MODEL_1_7B | plan::MODEL_1_7B_BASE | plan::MODEL_1_7B_DESIGN => Some(3200),
        MODEL_0_6B | plan::MODEL_0_6B_BASE => Some(1500),
        _ => None,
    };
    plan::plan(
        k,
        &plan::Env {
            sys,
            recommendation: r,
            installed: &is,
            footprint_mb: &footprint,
        },
    )
}

const ALL_MODELS: [&str; 6] = [
    MODEL_1_7B,
    plan::MODEL_1_7B_BASE,
    plan::MODEL_1_7B_DESIGN,
    MODEL_0_6B,
    plan::MODEL_0_6B_BASE,
    plan::TOKENIZER_ID,
];

// ---------------------------------------------------------------------------------------------
// Plan
// ---------------------------------------------------------------------------------------------

#[test]
fn normalize_checks_every_knob() {
    let k = plan::normalize(&QwenConfig::default()).unwrap();
    assert_eq!(k, knobs(None, None, Mode::Preset));
    let k = plan::normalize(&QwenConfig {
        model: " 0.6B ".into(),
        variant: "CUDA13".into(),
        mode: "clone".into(),
        language: "ja-JP".into(),
    })
    .unwrap();
    assert_eq!((k.size, k.variant, k.mode, k.language), (Some(Size::Small), Some(Variant::Cuda13), Mode::Clone, "japanese"));
    assert_eq!(plan::normalize(&QwenConfig { model: "1.7b".into(), ..Default::default() }).unwrap().size, Some(Size::Large));
    assert_eq!(plan::normalize(&QwenConfig { mode: "design".into(), ..Default::default() }).unwrap().mode, Mode::Design);
    for bad in [
        QwenConfig { model: "7b".into(), ..Default::default() },
        QwenConfig { variant: "rocm".into(), ..Default::default() },
        QwenConfig { mode: "sing".into(), ..Default::default() },
    ] {
        assert!(plan::normalize(&bad).is_err(), "{bad:?}");
    }
    // A language the model lacks speaks with auto.
    assert_eq!(plan::normalize(&QwenConfig { language: "ar".into(), ..Default::default() }).unwrap().language, "auto");
}

#[test]
fn language_names_follow_qwentts() {
    for (tag, name) in [
        ("en", "english"),
        ("EN-us", "english"),
        ("ja", "japanese"),
        ("zh-Hant", "chinese"),
        ("yue", "chinese"),
        ("ko", "korean"),
        ("pt_BR", "portuguese"),
        ("es-419", "spanish"),
        ("Japanese", "japanese"),
        ("auto", "auto"),
        ("", "auto"),
    ] {
        assert_eq!(plan::language_name(tag), Some(name), "{tag}");
    }
    assert_eq!(plan::language_name("ar"), None);
    assert_eq!(plan::language_name("vi"), None);
}

#[test]
fn talker_ids_cover_every_size_and_mode() {
    for id in plan::TALKER_IDS {
        let (s, m) = plan::parse_talker_id(id).unwrap_or_else(|| panic!("{id}"));
        assert_eq!(plan::talker_id(s, m), Some(id));
    }
    assert_eq!(plan::talker_id(Size::Small, Mode::Design), None, "there is no 0.6B VoiceDesign");
    assert_eq!(plan::parse_talker_id(plan::TOKENIZER_ID), None);
    // The governor's size ids are the CustomVoice talkers.
    assert_eq!(plan::talker_id(Size::Large, Mode::Preset), Some(resources::policy::ids::TTS_QWEN3_1_7B));
    assert_eq!(plan::talker_id(Size::Small, Mode::Preset), Some(resources::policy::ids::TTS_QWEN3_0_6B));
    assert_eq!(Size::from_policy_id(MODEL_0_6B), Some(Size::Small));
}

#[test]
fn auto_size_follows_the_governor_and_prefers_installed_alternatives() {
    let sys = nvidia(8188);
    let all = [&ALL_MODELS[..], &[plan::RUNTIME_VULKAN]].concat();
    // The governor's choice when it is installed.
    let p = run_plan(&knobs(None, None, Mode::Preset), &sys, &rec(MODEL_1_7B, &[MODEL_0_6B], "gpu"), &all);
    assert_eq!((p.size, p.talker, p.variant), (Size::Large, MODEL_1_7B, Variant::Vulkan));
    assert!(p.missing.is_empty() && p.refused.is_none());
    assert_eq!(p.vram_mb, 3200);
    // The voice changer runs on this 8 GB card: 0.6B (cloning talker in clone mode).
    let p = run_plan(&knobs(None, None, Mode::Clone), &sys, &rec(MODEL_0_6B, &[], "gpu"), &all);
    assert_eq!((p.size, p.talker, p.vram_mb), (Size::Small, plan::MODEL_0_6B_BASE, 1500));
    // 1.7B recommended but only 0.6B installed (and accepted by the governor): 0.6B, no download.
    let small_only = [MODEL_0_6B, plan::TOKENIZER_ID, plan::RUNTIME_VULKAN];
    let p = run_plan(&knobs(None, None, Mode::Preset), &sys, &rec(MODEL_1_7B, &[MODEL_0_6B], "gpu"), &small_only);
    assert_eq!((p.size, p.talker), (Size::Small, MODEL_0_6B));
    assert!(p.missing.is_empty());
    // ... but not when the governor does not list it.
    let p = run_plan(&knobs(None, None, Mode::Preset), &sys, &rec(MODEL_1_7B, &[], "gpu"), &small_only);
    assert_eq!(p.talker, MODEL_1_7B);
    assert_eq!(p.missing, vec![MODEL_1_7B.to_string()]);
    // Nothing installed: the recommended talker, the tokenizer and the default pack, in that order.
    let p = run_plan(&knobs(None, None, Mode::Clone), &sys, &rec(MODEL_1_7B, &[MODEL_0_6B], "gpu"), &[]);
    assert_eq!(p.missing, vec![plan::MODEL_1_7B_BASE.to_string(), plan::TOKENIZER_ID.into(), plan::RUNTIME_VULKAN.into()]);
}

#[test]
fn explicit_size_and_design_mode() {
    let sys = nvidia(8188);
    let all = [&ALL_MODELS[..], &[plan::RUNTIME_VULKAN]].concat();
    // The user's size wins over the governor.
    let p = run_plan(&knobs(Some(Size::Large), None, Mode::Preset), &sys, &rec(MODEL_0_6B, &[], "gpu"), &all);
    assert_eq!((p.size, p.talker), (Size::Large, MODEL_1_7B));
    let p = run_plan(&knobs(Some(Size::Small), None, Mode::Clone), &sys, &rec(MODEL_1_7B, &[], "gpu"), &all);
    assert_eq!(p.talker, plan::MODEL_0_6B_BASE);
    // Voice design is 1.7B only, whatever the size setting says.
    for size in [None, Some(Size::Small), Some(Size::Large)] {
        let p = run_plan(&knobs(size, None, Mode::Design), &sys, &rec(MODEL_0_6B, &[], "gpu"), &all);
        assert_eq!((p.size, p.talker), (Size::Large, plan::MODEL_1_7B_DESIGN), "{size:?}");
    }
}

#[test]
fn variant_auto_explicit_and_refused() {
    let models = &ALL_MODELS[..];
    let with = |extra: &[&'static str]| [models, extra].concat();
    let r = rec(MODEL_1_7B, &[], "gpu");
    let k = knobs(None, None, Mode::Preset);
    // NVIDIA with the cuda13 pack and the NVIDIA runtime: CUDA 13.
    let p = run_plan(&k, &nvidia(12_282), &r, &with(&[plan::RUNTIME_CUDA13, plan::NVIDIA_RUNTIME_ID, plan::RUNTIME_VULKAN]));
    assert_eq!(p.variant, Variant::Cuda13);
    assert!(p.missing.is_empty());
    // The NVIDIA runtime is missing: Vulkan when it is installed ...
    let p = run_plan(&k, &nvidia(12_282), &r, &with(&[plan::RUNTIME_CUDA13, plan::RUNTIME_VULKAN]));
    assert_eq!(p.variant, Variant::Vulkan);
    // ... else CUDA 13 with the runtime to install.
    let p = run_plan(&k, &nvidia(12_282), &r, &with(&[plan::RUNTIME_CUDA13]));
    assert_eq!((p.variant, p.missing.clone()), (Variant::Cuda13, vec![plan::NVIDIA_RUNTIME_ID.to_string()]));
    // The NVIDIA runtime is there (the voice changer uses it) but no server pack: the CUDA one.
    let p = run_plan(&k, &nvidia(12_282), &r, &with(&[plan::NVIDIA_RUNTIME_ID]));
    assert_eq!((p.variant, p.missing.clone()), (Variant::Cuda13, vec![plan::RUNTIME_CUDA13.to_string()]));
    // Nothing of either: the Vulkan pack (every vendor).
    let p = run_plan(&k, &nvidia(12_282), &r, &with(&[]));
    assert_eq!((p.variant, p.missing.clone()), (Variant::Vulkan, vec![plan::RUNTIME_VULKAN.to_string()]));
    // AMD: never CUDA, even when the cuda13 pack happens to be installed.
    let p = run_plan(&k, &amd(16_368), &r, &with(&[plan::RUNTIME_CUDA13, plan::NVIDIA_RUNTIME_ID]));
    assert_eq!((p.variant, p.missing.clone()), (Variant::Vulkan, vec![plan::RUNTIME_VULKAN.to_string()]));
    // Explicit cuda13 on AMD can never work: refused, nothing to download.
    let p = run_plan(&knobs(None, Some(Variant::Cuda13), Mode::Preset), &amd(16_368), &r, &with(&[]));
    assert!(p.refused.as_deref().is_some_and(|w| w.contains("NVIDIA")), "{:?}", p.refused);
    assert!(p.missing.is_empty());
    // No usable GPU (the governor says CPU): the CPU pack, no VRAM claim.
    let p = run_plan(&k, &no_gpu(), &rec(MODEL_0_6B, &[], "cpu"), &with(&[plan::RUNTIME_CPU]));
    assert_eq!((p.variant, p.size, p.vram_mb), (Variant::Cpu, Size::Small, 0));
    assert!(p.missing.is_empty());
    // Explicit CPU on a GPU PC.
    let p = run_plan(&knobs(None, Some(Variant::Cpu), Mode::Preset), &nvidia(12_282), &r, &with(&[]));
    assert_eq!((p.variant, p.vram_mb), (Variant::Cpu, 0));
    assert_eq!(p.missing, vec![plan::RUNTIME_CPU.to_string()]);
    // Explicit vulkan with only CUDA installed.
    let p = run_plan(&knobs(None, Some(Variant::Vulkan), Mode::Preset), &nvidia(12_282), &r, &with(&[plan::RUNTIME_CUDA13, plan::NVIDIA_RUNTIME_ID]));
    assert_eq!((p.variant, p.missing.clone()), (Variant::Vulkan, vec![plan::RUNTIME_VULKAN.to_string()]));
}

#[test]
fn engine_facts_match_the_scaffold() {
    assert_eq!(server::HELPER_NAME, ENGINE.helper_name());
    let i = info();
    assert_eq!(i.engine, ENGINE);
    assert!(i.gpu && i.cloning && !i.experimental);
    assert!(i.requires.iter().any(|r| r == MODEL_1_7B) && i.requires.iter().any(|r| r == plan::TOKENIZER_ID));
    assert_eq!(CACHE_BACKEND, "qwen3-tts");
    // The voice library accepts the cache backend name (its own rules decide).
    let app_data = temp_dir("backend-name");
    make_voice(&app_data, "name-check-000001", 200.0, None);
    crate::services::voice_library::write_cache(&app_data, "name-check-000001", CACHE_BACKEND, b"{}").expect("a valid cache backend name");
    assert_eq!(
        crate::services::voice_library::read_cache(&app_data, "name-check-000001", CACHE_BACKEND).as_deref(),
        Some(&b"{}"[..])
    );
    // The engine config is camelCase and tolerant of the rest of the TS settings slice.
    let c: QwenConfig = serde_json::from_value(serde_json::json!({ "model": "0.6b", "mode": "clone", "voice": "x", "device": "" })).unwrap();
    assert_eq!((c.model.as_str(), c.mode.as_str(), c.variant.as_str()), ("0.6b", "clone", ""));
    let wire = serde_json::to_value(QwenCloneVoice { voice_id: "a-1".into(), voice: "sigil-a-1".into(), cached: true }).unwrap();
    assert_eq!(wire, serde_json::json!({ "voiceId": "a-1", "voice": "sigil-a-1", "cached": true }));
}

#[test]
fn load_timeout_scales_with_size() {
    assert_eq!(plan::load_timeout_ms(0), 60_000);
    assert_eq!(plan::load_timeout_ms(2_333_984_928), 60_000 + 3 * 30_000);
    assert_eq!(plan::load_timeout_ms(u64::MAX), 600_000);
}

// ---------------------------------------------------------------------------------------------
// Catalog fragments
// ---------------------------------------------------------------------------------------------

#[test]
fn catalog_has_every_talker_the_tokenizer_and_the_packs() {
    let catalog = Catalog::embedded().unwrap();
    let tok = catalog.get(plan::TOKENIZER_ID).expect("tokenizer entry");
    assert_eq!(tok.family, "tts");
    assert!(tok.files.len() == 1 && tok.files[0].path.ends_with(".gguf"));
    for id in plan::TALKER_IDS {
        let e = catalog.get(id).unwrap_or_else(|| panic!("{id} is catalogued"));
        assert_eq!(e.family, "tts", "{id}");
        assert_eq!(e.requires, vec![plan::TOKENIZER_ID.to_string()], "{id}");
        assert_eq!(e.files.len(), 1, "{id}");
        let f = &e.files[0];
        assert!(f.path.starts_with("qwen-talker-") && f.path.ends_with("-Q8_0.gguf"), "{id}: {}", f.path);
        assert!(f.size > 500_000_000, "{id}");
        assert!(f.sha256.as_deref().is_some_and(|s| s.len() == 64), "{id} is pinned");
        let src = &f.sources[0];
        assert!(src.url.contains("/resolve/b7ee2e8c7459c3bea99da23e3d178125a7d1713c/"), "{id} pinned to a commit: {}", src.url);
        // The checkpoint's mode is in its file name.
        let (_, mode) = plan::parse_talker_id(id).unwrap();
        let marker = match mode {
            Mode::Preset => "-customvoice-",
            Mode::Clone => "-base-",
            Mode::Design => "-voicedesign-",
        };
        assert!(f.path.contains(marker), "{id}: {}", f.path);
        assert!(e.min_vram_mb.is_some(), "{id} has a footprint");
    }
    for (id, variant) in [(plan::RUNTIME_VULKAN, "vulkan"), (plan::RUNTIME_CUDA13, "cuda13"), (plan::RUNTIME_CPU, "cpu")] {
        let e = catalog.get(id).unwrap_or_else(|| panic!("{id} is catalogued"));
        assert_eq!(e.variant.as_deref(), Some(variant));
        let archive = e.files[0].archive.as_ref().expect("a zip");
        assert!(archive.include.iter().any(|p| p == plan::SERVER_EXE) && archive.include.iter().any(|p| p == "sigil-pack.json"));
        assert!(e.files[0].path.contains(&format!("-{}-{variant}.zip", e.version)), "{}", e.files[0].path);
    }
    assert!(catalog
        .get(plan::RUNTIME_CUDA13)
        .unwrap()
        .requires
        .iter()
        .any(|r| r == plan::NVIDIA_RUNTIME_ID));
}

#[test]
fn catalog_footprints_equal_the_governors_estimates() {
    let catalog = Catalog::embedded().unwrap();
    let footprint = |id: &str| catalog.get(id).and_then(|e| e.min_vram_mb).unwrap();
    // A 24 GB card with nothing else loaded: 1.7B.
    let big = resources::recommend(RecommendationTopic::Qwen3Tts, &nvidia(24_564), &ResourceSnapshot::default());
    assert_eq!(big.choice, MODEL_1_7B);
    assert_eq!(big.vram_mb, footprint(MODEL_1_7B));
    // An 8 GB card with the voice changer on the GPU: 0.6B.
    let vc = ResourceSnapshot {
        vc_running: true,
        vc_gpu: Some(true),
        ..Default::default()
    };
    let small = resources::recommend(RecommendationTopic::Qwen3Tts, &nvidia(8188), &vc);
    assert_eq!(small.choice, MODEL_0_6B);
    assert_eq!(small.vram_mb, footprint(MODEL_0_6B));
    // Every talker of a size claims the same.
    for id in plan::TALKER_IDS {
        let (size, _) = plan::parse_talker_id(id).unwrap();
        assert_eq!(footprint(id), footprint(size.policy_id()), "{id}");
    }
}

// ---------------------------------------------------------------------------------------------
// Resolution and the helper spec
// ---------------------------------------------------------------------------------------------

/// Parent of every temp folder these tests make (`<temp>/sigil-qwen-tests/<pid>-<tag>`). On first
/// use it removes the folders of earlier runs that are older than an hour (a run going on at the
/// same time is never touched), so repeated test runs do not pile up model and voice fixtures.
fn temp_root() -> PathBuf {
    static SWEPT: std::sync::Once = std::sync::Once::new();
    let root = std::env::temp_dir().join("sigil-qwen-tests");
    SWEPT.call_once(|| {
        let mine = format!("{}-", std::process::id());
        let stale = |e: &std::fs::DirEntry| {
            e.metadata()
                .and_then(|m| m.modified())
                .ok()
                .and_then(|t| t.elapsed().ok())
                .is_some_and(|age| age > Duration::from_secs(3600))
        };
        for e in std::fs::read_dir(&root).into_iter().flatten().flatten() {
            if !e.file_name().to_string_lossy().starts_with(&mine) && stale(&e) {
                let _ = std::fs::remove_dir_all(e.path());
            }
        }
        // The flat folders of earlier versions of these tests.
        for e in std::fs::read_dir(std::env::temp_dir()).into_iter().flatten().flatten() {
            if e.file_name().to_string_lossy().starts_with("sigil-qwen-test-") && stale(&e) {
                let _ = std::fs::remove_dir_all(e.path());
            }
        }
    });
    root
}

fn temp_dir(tag: &str) -> PathBuf {
    let d = temp_root().join(format!("{}-{tag}", std::process::id()));
    let _ = std::fs::remove_dir_all(&d);
    std::fs::create_dir_all(&d).unwrap();
    d
}

/// Installed folders for `ids` under `root` (the catalog's files present, a valid pack manifest).
fn install(root: &Path, catalog: &Catalog, ids: &[&str], pack_variant: &str) -> HashMap<String, PathBuf> {
    let mut map = HashMap::new();
    for id in ids {
        let dir = root.join(id.replace('/', "_"));
        std::fs::create_dir_all(&dir).unwrap();
        if id.starts_with("runtime/qwentts-server-") {
            let exe = b"MZ fake sigil-tts-server";
            std::fs::write(dir.join(plan::SERVER_EXE), exe).unwrap();
            let sha = {
                use sha2::{Digest, Sha256};
                hex::encode(Sha256::digest(exe))
            };
            let manifest = serde_json::json!({
                "schema": 1, "engine": "qwentts-server", "version": "6fae929-s1", "variant": pack_variant,
                "program": plan::SERVER_EXE,
                "files": [{ "path": plan::SERVER_EXE, "size": exe.len(), "sha256": sha }],
            });
            std::fs::write(dir.join("sigil-pack.json"), serde_json::to_vec(&manifest).unwrap()).unwrap();
        } else if let Some(e) = catalog.get(id) {
            for f in &e.files {
                std::fs::write(dir.join(&f.path), b"GGUF").unwrap();
            }
        }
        map.insert(id.to_string(), dir);
    }
    map
}

#[test]
fn resolve_builds_the_launch_from_installed_files() {
    let catalog = Catalog::embedded().unwrap();
    let root = temp_dir("resolve");
    let ids = [plan::MODEL_1_7B_BASE, plan::TOKENIZER_ID, plan::RUNTIME_VULKAN];
    let map = install(&root, &catalog, &ids, "vulkan");
    let installed = |id: &str| map.get(id).cloned();
    let config = QwenConfig {
        mode: "clone".into(),
        language: "ja".into(),
        ..Default::default()
    };
    let (l, p) = resolve_with(&config, &catalog, &nvidia(12_282), &rec(MODEL_1_7B, &[], "gpu"), &installed, &Vec::new).unwrap();
    assert_eq!(p.talker, plan::MODEL_1_7B_BASE);
    assert_eq!(l.model, plan::MODEL_1_7B_BASE);
    assert_eq!((l.mode, l.size, l.variant), (Mode::Clone, Size::Large, Variant::Vulkan));
    assert_eq!(l.language, "japanese");
    // The configured language never decides the server: it always runs `--lang auto` (each request
    // names its language), and a launch that differs only in its language is the same server.
    let args = build_spec(&l).args.join(" ");
    assert!(args.contains("--lang auto") && !args.contains("japanese"), "{args}");
    let (english, _) = resolve_with(
        &QwenConfig { language: "en".into(), ..config.clone() },
        &catalog,
        &nvidia(12_282),
        &rec(MODEL_1_7B, &[], "gpu"),
        &installed,
        &Vec::new,
    )
    .unwrap();
    assert_eq!(english.language, "english");
    assert!(english.same_server(&l) && l.same_server(&english));
    assert_eq!(build_spec(&english).args, build_spec(&l).args);
    assert_eq!(l.talker_file(), "qwen-talker-1.7b-base-Q8_0.gguf");
    assert_eq!(l.codec_file(), "qwen-tokenizer-12hz-Q8_0.gguf");
    assert_eq!(l.exe, map[plan::RUNTIME_VULKAN].join(plan::SERVER_EXE));
    assert_eq!(l.vram_mb, 3200);
    assert!(l.load_timeout_ms >= 120_000, "about 2.4 GB to load: {}", l.load_timeout_ms);
    assert!(l.dll_dirs.is_empty());

    // Another mode needs another talker: named in the error.
    let e = resolve_with(&QwenConfig::default(), &catalog, &nvidia(12_282), &rec(MODEL_1_7B, &[], "gpu"), &installed, &Vec::new).unwrap_err();
    assert!(e.contains(MODEL_1_7B) && e.contains("downloads"), "{e}");

    // A pack imported into the wrong slot is refused by name.
    let wrong = install(&temp_dir("resolve-wrong"), &catalog, &ids, "cpu");
    let wrong_installed = |id: &str| wrong.get(id).cloned();
    let e = resolve_with(&config, &catalog, &nvidia(12_282), &rec(MODEL_1_7B, &[], "gpu"), &wrong_installed, &Vec::new).unwrap_err();
    assert!(e.contains("cpu build"), "{e}");

    // A damaged server program is refused.
    std::fs::write(map[plan::RUNTIME_VULKAN].join(plan::SERVER_EXE), b"MZ changed!!!!!!!!!!!!!!").unwrap();
    let e = resolve_with(&config, &catalog, &nvidia(12_282), &rec(MODEL_1_7B, &[], "gpu"), &installed, &Vec::new).unwrap_err();
    assert!(e.contains("does not match"), "{e}");
}

#[test]
fn resolve_cuda13_needs_the_nvidia_runtime_folders() {
    let catalog = Catalog::embedded().unwrap();
    let root = temp_dir("resolve-cuda");
    let ids = [MODEL_1_7B, plan::TOKENIZER_ID, plan::RUNTIME_CUDA13, plan::NVIDIA_RUNTIME_ID];
    let map = install(&root, &catalog, &ids, "cuda13");
    let installed = |id: &str| map.get(id).cloned();
    let config = QwenConfig::default();
    let e = resolve_with(&config, &catalog, &nvidia(12_282), &rec(MODEL_1_7B, &[], "gpu"), &installed, &Vec::new).unwrap_err();
    assert!(e.contains("NVIDIA GPU runtime"), "{e}");
    let cuda_dir = root.join("cuda-bin");
    let dirs = || vec![cuda_dir.clone()];
    let (l, _) = resolve_with(&config, &catalog, &nvidia(12_282), &rec(MODEL_1_7B, &[], "gpu"), &installed, &dirs).unwrap();
    assert_eq!(l.variant, Variant::Cuda13);
    assert_eq!(l.dll_dirs, vec![cuda_dir.clone()]);
    let spec = build_spec(&l);
    let path = spec.env.iter().find(|(k, _)| k == "PATH").map(|(_, v)| v.clone()).unwrap();
    assert!(path.starts_with(&cuda_dir.display().to_string()), "{path}");
}

fn test_launch(tag: &str, mode: Mode, variant: Variant) -> Launch {
    let dir = temp_dir(&format!("model-{tag}"));
    let talker = dir.join(format!("qwen-talker-1.7b-{}-Q8_0.gguf", mode.as_str()));
    let codec = dir.join("qwen-tokenizer-12hz-Q8_0.gguf");
    std::fs::write(&talker, b"GGUF").unwrap();
    std::fs::write(&codec, b"GGUF").unwrap();
    Launch {
        helper_name: format!("t-qwen-{tag}"),
        exe: test_exe(),
        talker_path: talker,
        codec_path: codec,
        model: plan::talker_id(Size::Large, mode).unwrap().into(),
        size: Size::Large,
        mode,
        variant,
        language: "auto".into(),
        vram_mb: 3200,
        load_timeout_ms: 20_000,
        dll_dirs: Vec::new(),
    }
}

#[test]
fn spec_binds_loopback_and_keeps_the_key_off_the_command_line() {
    let l = test_launch("spec", Mode::Preset, Variant::Vulkan);
    let spec = build_spec(&l);
    assert_eq!(spec.name, l.helper_name);
    let joined = spec.args.join(" ");
    assert!(joined.contains("--host 127.0.0.1 --port {port}"), "{joined}");
    assert!(joined.contains(&format!("--alias {}", MODEL_1_7B)) && joined.contains("--lang auto"), "{joined}");
    assert!(joined.contains("--model ") && joined.contains("--codec "), "{joined}");
    assert!(!joined.to_ascii_lowercase().contains("key"), "the key never goes on the command line: {joined}");
    assert_eq!(spec.env, vec![(server::API_KEY_ENV.to_string(), "{api_key}".to_string())]);
    assert!(spec.env_remove.iter().any(|v| v == server::API_KEY_ENV));
    assert!(spec.api_key && spec.port == PortRequest::Auto && spec.high_qos);
    assert_eq!(spec.health.as_ref().unwrap().path, "/health");
    assert_eq!(spec.priority, HelperPriority::Normal);
    assert!(matches!(spec.restart, RestartPolicy::OnFailure { max_restarts: 2, .. }));
    process_host::spawn::validate(&spec).unwrap();
    // The CPU pack yields the processor to the game.
    assert_eq!(build_spec(&test_launch("spec-cpu", Mode::Preset, Variant::Cpu)).priority, HelperPriority::BelowNormal);
}

// ---------------------------------------------------------------------------------------------
// Load log
// ---------------------------------------------------------------------------------------------

fn lines(text: &[&str]) -> Vec<HelperLogLine> {
    text.iter()
        .enumerate()
        .map(|(i, t)| HelperLogLine {
            seq: i as u64,
            at_unix_ms: 0,
            stream: crate::services::process_host::logs::LogStream::Stderr,
            text: t.to_string(),
        })
        .collect()
}

#[test]
fn load_log_is_measured() {
    // Lines of upstream's tests/clone-CUDA0-Q8_0.log.
    let log = lines(&[
        "[Load] Talker backend: CUDA0 (CPU threads: 16)",
        "[GGUF] ../models/qwen-talker-1.7b-base-Q8_0.gguf: 480 tensors, data at offset 5954496",
        "[WeightCtx] Loaded 316 tensors, 1765.0 MB into backend",
        "[WeightCtx] Loaded 88 tensors, 177.5 MB into backend",
        "[KVCache] Allocated: 28 layers, 8 KV heads, head_dim 128, max_seq_len 4096, 1 sets -> 896 MB",
        "[KVCache] Allocated: 5 layers, 8 KV heads, head_dim 128, max_seq_len 16, 1 sets -> 0 MB",
        "[DAC] Loaded: 4 blocks (strides 8/5/4/3), 24 kHz mono out, weights 142.9 MB",
        "[WeightCtx] Loaded 118 tensors, 142.9 MB into backend",
    ]);
    let r = server::parse_load_report(&log);
    assert_eq!(r.backend.as_deref(), Some("CUDA0"));
    assert!((r.buffers_mb - (1765.0 + 177.5 + 896.0 + 142.9)).abs() < 1e-6, "{}", r.buffers_mb);
    assert!(r.on_gpu());
    let mb = server::measured_vram_mb(&r, Variant::Cuda13).unwrap();
    assert_eq!(mb, 2982 + server::CUDA_BASE_MB + server::COMPUTE_ALLOWANCE_MB);
    // A restarted server logs its load again: only the newest instance counts.
    let mut twice = log.clone();
    twice.extend(lines(&[
        "[Server] sigil-tts-server 6fae929-s1 (qwentts.cpp 6fae929, cuda13)",
        "[Load] Talker backend: CUDA0 (CPU threads: 16)",
        "[WeightCtx] Loaded 316 tensors, 900.0 MB into backend",
        "[KVCache] Allocated: 28 layers, 8 KV heads, head_dim 128, max_seq_len 4096, 1 sets -> 448 MB",
    ]));
    let again = server::parse_load_report(&twice);
    assert!((again.buffers_mb - 1348.0).abs() < 1e-6, "{}", again.buffers_mb);
    // On the CPU (or before the backend line) nothing is claimed.
    let cpu = server::parse_load_report(&lines(&["[Load] Talker backend: CPU (CPU threads: 8)", "[WeightCtx] Loaded 1 tensors, 10.0 MB into backend"]));
    assert!(!cpu.on_gpu());
    assert_eq!(server::measured_vram_mb(&cpu, Variant::Vulkan), None);
    assert_eq!(server::measured_vram_mb(&server::LoadReport::default(), Variant::Vulkan), None);
}

// ---------------------------------------------------------------------------------------------
// Clone cache format
// ---------------------------------------------------------------------------------------------

fn b64(bytes: &[u8]) -> String {
    base64::engine::general_purpose::STANDARD.encode(bytes)
}

fn entry(talker: &str, codec: &str, seed: u8) -> CacheEntry {
    CacheEntry {
        talker: talker.into(),
        codec: codec.into(),
        spk_b64: b64(&[seed; 16]),
        rvq_b64: b64(&[seed; 22]),
        ref_t: 8,
        spk_dim: 4,
    }
}

#[test]
fn cache_round_trip_merge_and_validation() {
    let sha = "ab".repeat(32);
    let f = clone::merge_cache(None, &sha, entry("t1.gguf", "c.gguf", 1));
    assert_eq!((f.format.as_str(), f.version, f.entries.len()), (CACHE_FORMAT, CACHE_VERSION, 1));
    let bytes = serde_json::to_vec(&f).unwrap();
    let back = clone::decode_cache(&bytes, &sha.to_uppercase()).expect("same clip");
    assert_eq!(back, f);
    assert!(clone::find_entry(&back, "T1.GGUF", "c.gguf").is_some());
    assert!(clone::find_entry(&back, "t2.gguf", "c.gguf").is_none());
    // Another clip, format or version: ignored.
    assert!(clone::decode_cache(&bytes, &"cd".repeat(32)).is_none());
    let mut other = f.clone();
    other.version = 2;
    assert!(clone::decode_cache(&serde_json::to_vec(&other).unwrap(), &sha).is_none());
    assert!(clone::decode_cache(b"not json", &sha).is_none());
    // Damaged latents are dropped.
    let mut bad = f.clone();
    bad.entries.push(CacheEntry { spk_b64: "!!".into(), ..entry("t3.gguf", "c.gguf", 3) });
    bad.entries.push(CacheEntry { spk_b64: b64(&[1; 3]), ..entry("t4.gguf", "c.gguf", 4) });
    assert_eq!(clone::decode_cache(&serde_json::to_vec(&bad).unwrap(), &sha).unwrap().entries.len(), 1);
    // Merge: newest first, the same pair replaced, bounded, a stale clip's entries dropped.
    let mut m = f.clone();
    for i in 2..=6u8 {
        m = clone::merge_cache(Some(m), &sha, entry(&format!("t{i}.gguf"), "c.gguf", i));
    }
    assert_eq!(m.entries.len(), MAX_CACHE_ENTRIES);
    assert_eq!(m.entries[0].talker, "t6.gguf");
    let replaced = clone::merge_cache(Some(m.clone()), &sha, entry("t5.gguf", "c.gguf", 9));
    assert_eq!(replaced.entries.iter().filter(|e| e.talker == "t5.gguf").count(), 1);
    assert_eq!(replaced.entries[0].spk_b64, b64(&[9; 16]));
    let fresh = clone::merge_cache(Some(m), &"ef".repeat(32), entry("t1.gguf", "c.gguf", 1));
    assert_eq!(fresh.entries.len(), 1);
    // Request bodies.
    assert_eq!(clone::server_voice_name("alice-1a2b3c"), "sigil-alice-1a2b3c");
    let body = clone::wav_body("sigil-x", "hi", b"RIFF");
    assert_eq!(body["wav_b64"], b64(b"RIFF"));
    let body = clone::latents_body("sigil-x", "hi", &entry("t", "c", 1));
    assert!(body.get("wav_b64").is_none() && body["spk_b64"].is_string() && body["ref_text"] == "hi");
}

// ---------------------------------------------------------------------------------------------
// Lifecycle with a fake server (child process)
// ---------------------------------------------------------------------------------------------

const ROLE_ENV: &str = "SIGIL_QWEN_FAKE";
const ENTRY: &str = "services::tts_helpers::qwen::tests::fake_qwentts_server_entry";

fn test_exe() -> PathBuf {
    std::env::current_exe().expect("test binary path")
}

/// The lifecycle tests share the one server state: run them one at a time.
async fn serial() -> tokio::sync::MutexGuard<'static, ()> {
    static SERIAL: std::sync::OnceLock<tokio::sync::Mutex<()>> = std::sync::OnceLock::new();
    SERIAL.get_or_init(Default::default).lock().await
}

/// Route every spawn of the Qwen3-TTS helper to the fake (`backend` is its load-log device).
fn install_fake(model_type: &'static str, backend: &'static str) {
    test_hooks::set(move |mut spec| {
        let args = spec.args.join("\t");
        spec.program = test_exe();
        spec.args = ["--exact", ENTRY, "--ignored", "--nocapture", "--test-threads=1", "-q"]
            .iter()
            .map(|s| s.to_string())
            .collect();
        spec.env.push((ROLE_ENV.into(), "ok".into()));
        spec.env.push(("SIGIL_QWEN_FAKE_PORT".into(), "{port}".into()));
        spec.env.push(("SIGIL_QWEN_FAKE_TYPE".into(), model_type.into()));
        spec.env.push(("SIGIL_QWEN_FAKE_BACKEND".into(), backend.into()));
        spec.env.push(("SIGIL_QWEN_FAKE_ARGS".into(), args));
        if let Some(h) = spec.health.as_mut() {
            h.interval_ms = 50;
            h.timeout_ms = 20_000;
        }
        spec
    });
}

fn claim_mb(model: &str) -> Option<u64> {
    resources::snapshot()
        .claims
        .into_iter()
        .find(|c| c.label == format!("qwen3-tts: {model}"))
        .map(|c| c.mb)
}

async fn wait_until(timeout: Duration, mut f: impl FnMut() -> bool) -> bool {
    let deadline = Instant::now() + timeout;
    while Instant::now() < deadline {
        if f() {
            return true;
        }
        tokio::time::sleep(Duration::from_millis(25)).await;
    }
    f()
}

fn helper_log(name: &str, needle: &str) -> Vec<String> {
    process_host::logs(name)
        .into_iter()
        .map(|l| l.text)
        .filter(|t| t.contains(needle))
        .collect()
}

async fn get_status(url: &str, key: Option<&str>) -> u16 {
    let mut rb = reqwest::Client::builder()
        .no_proxy()
        .build()
        .unwrap()
        .get(url)
        .timeout(Duration::from_secs(5));
    if let Some(k) = key {
        rb = rb.bearer_auth(k);
    }
    rb.send().await.map(|r| r.status().as_u16()).unwrap_or(0)
}

#[tokio::test(flavor = "multi_thread")]
async fn start_warm_up_measure_reuse_and_stop() {
    let _s = serial().await;
    install_fake("custom_voice", "Vulkan0");
    let l = test_launch("life", Mode::Preset, Variant::Vulkan);
    let endpoint = server::start_launch(l.clone(), server::stop_epoch()).await.expect("fake server starts");
    assert_eq!(endpoint.engine, ENGINE);
    assert!(endpoint.base_url.starts_with("http://127.0.0.1:") && endpoint.base_url.ends_with("/v1"), "{}", endpoint.base_url);
    assert_eq!(endpoint.api_key.len(), 64);
    assert_eq!((endpoint.model.as_str(), endpoint.pcm_sample_rate), (MODEL_1_7B, 24_000));
    let st = server::status();
    assert_eq!((st.state, st.model.as_deref()), (TtsHelperState::Ready, Some(MODEL_1_7B)));
    assert!(!serde_json::to_string(&st).unwrap().contains(&endpoint.api_key), "the status never carries the key");

    // The key is required (the fake enforces it like sigil-tts-server).
    let root = endpoint.base_url.trim_end_matches("/v1").to_string();
    assert_eq!(get_status(&format!("{root}/v1/models"), None).await, 401);
    assert_eq!(get_status(&format!("{root}/v1/models"), Some("wrong")).await, 401);
    assert_eq!(get_status(&format!("{root}/v1/models"), Some(&endpoint.api_key)).await, 200);
    assert_eq!(get_status(&format!("{root}/health"), None).await, 200);

    // The command line reached the process without the key; the warm-up spoke with the first speaker.
    let args = helper_log(&l.helper_name, "fake-qwen args:").join("");
    for flag in ["--host\t127.0.0.1", "--alias\ttts/qwen3-tts-1.7b", "--lang\tauto", "--model\t", "--codec\t"] {
        assert!(args.contains(flag), "{flag} missing: {args}");
    }
    assert!(process_host::logs(&l.helper_name).iter().all(|x| !x.text.contains(&endpoint.api_key)));
    let warm = helper_log(&l.helper_name, "fake-qwen speech:");
    assert_eq!(warm.len(), 1, "{warm:?}");
    assert!(warm[0].contains("\"voice\":\"serena\"") && warm[0].contains("max_new_tokens"), "{}", warm[0]);
    assert!(warm[0].contains("\"language\":\"auto\""), "the warm-up names the configured language: {}", warm[0]);

    // The claim starts at the plan and is resized to the logged buffers + Vulkan base + compute.
    assert!(claim_mb(&l.model).is_some());
    let want = 1765 + 896 + server::VULKAN_BASE_MB + server::COMPUTE_ALLOWANCE_MB;
    assert!(
        wait_until(Duration::from_secs(10), || claim_mb(&l.model) == Some(want)).await,
        "claim {:?}, want {want}",
        claim_mb(&l.model)
    );

    // Same launch: the running server is reused (no second process, no second warm-up).
    let pid = process_host::get(&l.helper_name).unwrap().pid;
    let again = server::start_launch(l.clone(), server::stop_epoch()).await.unwrap();
    assert_eq!(again, endpoint);
    assert_eq!(process_host::get(&l.helper_name).unwrap().pid, pid);
    assert_eq!(helper_log(&l.helper_name, "fake-qwen speech:").len(), 1);
    // Another language is the same server too: no reload (requests carry their own language).
    let japanese = Launch { language: "japanese".into(), ..l.clone() };
    let again = server::start_launch(japanese, server::stop_epoch()).await.unwrap();
    assert_eq!(again, endpoint);
    assert_eq!(process_host::get(&l.helper_name).unwrap().pid, pid);
    assert_eq!(helper_log(&l.helper_name, "fake-qwen speech:").len(), 1);
    assert_eq!(server::status().state, TtsHelperState::Ready);

    server::stop().await.unwrap();
    assert_eq!(server::status().state, TtsHelperState::Stopped);
    assert_eq!(claim_mb(&l.model), None, "the claim is released");
    assert!(server::current_target().is_err());
    let pid = pid.unwrap();
    assert!(process_host::job::wait_pid_exit(pid, Duration::from_secs(5)), "the helper process ended");
}

#[tokio::test(flavor = "multi_thread")]
async fn a_talker_of_another_mode_is_refused_and_stopped() {
    let _s = serial().await;
    install_fake("custom_voice", "Vulkan0");
    let l = test_launch("wrong-type", Mode::Clone, Variant::Vulkan);
    let e = server::start_launch(l.clone(), server::stop_epoch()).await.unwrap_err();
    assert!(e.contains("custom_voice") && e.contains("base"), "{e}");
    let st = server::status();
    assert_eq!(st.state, TtsHelperState::Failed);
    assert!(st.error.as_deref().is_some_and(|x| x.contains("custom_voice")));
    assert_eq!(claim_mb(&l.model), None);
    assert!(wait_until(Duration::from_secs(5), || process_host::get(&l.helper_name).is_none_or(|i| i.state != HelperState::Running)).await);
    server::stop().await.unwrap();
}

#[tokio::test(flavor = "multi_thread")]
async fn a_gpu_pack_on_the_cpu_releases_its_claim() {
    let _s = serial().await;
    install_fake("custom_voice", "CPU");
    let l = test_launch("cpu-backend", Mode::Preset, Variant::Vulkan);
    server::start_launch(l.clone(), server::stop_epoch()).await.unwrap();
    assert!(
        wait_until(Duration::from_secs(10), || claim_mb(&l.model) == Some(0)).await,
        "claim {:?}",
        claim_mb(&l.model)
    );
    server::stop().await.unwrap();
}

/// A library voice `id` under `app_data` with a clip of `freq` Hz and `transcript`.
fn make_voice(app_data: &Path, id: &str, freq: f32, transcript: Option<&str>) {
    let dir = app_data.join("voices").join(id);
    std::fs::create_dir_all(&dir).unwrap();
    let spec = hound::WavSpec {
        channels: 1,
        sample_rate: 24_000,
        bits_per_sample: 16,
        sample_format: hound::SampleFormat::Int,
    };
    let mut w = hound::WavWriter::create(dir.join("ref.wav"), spec).unwrap();
    for i in 0..24_000 * 3 {
        let s = (i as f32 * freq * std::f32::consts::TAU / 24_000.0).sin() * 0.3;
        w.write_sample((s * 32767.0) as i16).unwrap();
    }
    w.finalize().unwrap();
    let meta = serde_json::json!({ "id": id, "name": id, "transcript": transcript, "language": "en", "source": "recorded" });
    std::fs::write(dir.join("meta.json"), serde_json::to_vec(&meta).unwrap()).unwrap();
}

fn cache_path(app_data: &Path, id: &str) -> PathBuf {
    app_data.join("voices").join(id).join("cache").join(format!("{CACHE_BACKEND}.bin"))
}

#[tokio::test(flavor = "multi_thread")]
async fn clone_registers_caches_and_reuses_library_voices() {
    let _s = serial().await;
    install_fake("base", "CUDA0");
    let app_data = temp_dir("clone-appdata");
    let id = "alice-1a2b3c";
    make_voice(&app_data, id, 220.0, Some("Hello there, this is Alice."));

    // Not running yet.
    assert!(clone::prepare_in(&app_data, id).await.unwrap_err().contains("not running"));

    let l = test_launch("clone", Mode::Clone, Variant::Cuda13);
    let endpoint = server::start_launch(l.clone(), server::stop_epoch()).await.unwrap();
    let wav_regs = || helper_log(&l.helper_name, "fake-qwen register wav").len();
    let latent_regs = || helper_log(&l.helper_name, "fake-qwen register latents").len();
    // Base talkers warm up without a voice and without an instruction.
    let warm = helper_log(&l.helper_name, "fake-qwen speech:");
    assert!(warm.len() == 1 && !warm[0].contains("\"voice\"") && !warm[0].contains("instructions"), "{warm:?}");

    // First use: extracted from the clip, with the transcript; the latents are cached.
    let v = clone::prepare_in(&app_data, id).await.unwrap();
    assert_eq!(v, QwenCloneVoice { voice_id: id.into(), voice: format!("sigil-{id}"), cached: false });
    assert_eq!((wav_regs(), latent_regs()), (1, 0));
    assert!(helper_log(&l.helper_name, "fake-qwen register wav").join("").contains("ref_text=Hello there, this is Alice."));
    let clip = crate::services::voice_library::reference_clip(&app_data, id).unwrap();
    let cache = clone::decode_cache(&std::fs::read(cache_path(&app_data, id)).unwrap(), &clip.sha256).expect("cache written");
    assert!(clone::find_entry(&cache, &l.talker_file(), &l.codec_file()).is_some());

    // Registered with this instance: nothing is sent again.
    assert!(clone::prepare_in(&app_data, id).await.unwrap().cached);
    assert_eq!((wav_regs(), latent_regs()), (1, 0));

    // The registry is forgotten (a restart): registered again from the cached speaker data.
    clone::forget_all();
    assert!(clone::prepare_in(&app_data, id).await.unwrap().cached);
    assert_eq!((wav_regs(), latent_regs()), (1, 1));

    // An edited transcript registers again (still from the cache).
    make_voice(&app_data, id, 220.0, Some("Hello again."));
    assert!(clone::prepare_in(&app_data, id).await.unwrap().cached);
    assert_eq!((wav_regs(), latent_regs()), (1, 2));
    assert!(helper_log(&l.helper_name, "fake-qwen register latents").join("").contains("ref_text=Hello again."));

    // A re-recorded clip (another sha256) is extracted again; a refused cache falls back to the clip.
    make_voice(&app_data, id, 330.0, Some("Hello again."));
    assert!(!clone::prepare_in(&app_data, id).await.unwrap().cached);
    assert_eq!(wav_regs(), 2);
    let clip = crate::services::voice_library::reference_clip(&app_data, id).unwrap();
    let mut refused = clone::decode_cache(&std::fs::read(cache_path(&app_data, id)).unwrap(), &clip.sha256).unwrap();
    refused.entries[0].spk_b64 = b64(b"REFUSE!!");
    std::fs::write(cache_path(&app_data, id), serde_json::to_vec(&refused).unwrap()).unwrap();
    clone::forget_all();
    assert!(!clone::prepare_in(&app_data, id).await.unwrap().cached);
    assert_eq!(wav_regs(), 3);

    // A voice without a transcript still clones (timbre only).
    make_voice(&app_data, "bob-4d5e6f", 180.0, None);
    assert!(!clone::prepare_in(&app_data, "bob-4d5e6f").await.unwrap().cached);
    let last = helper_log(&l.helper_name, "fake-qwen register wav").pop().unwrap();
    assert!(last.contains("sigil-bob-4d5e6f") && last.trim_end().ends_with("ref_text="), "{last}");

    // Bad ids and missing voices.
    assert!(clone::prepare("../evil").await.is_err());
    assert!(clone::prepare_in(&app_data, "gone-000001").await.unwrap_err().contains("gone-000001"));

    // The cloned voice speaks (the request shape TypeScript sends through openai_speech).
    let t = server::timed_speech(
        &endpoint,
        serde_json::json!({ "input": "Hi.", "voice": format!("sigil-{id}"), "response_format": "pcm", "language": "english" }),
        Duration::from_secs(10),
    )
    .await
    .unwrap();
    assert!(t.audio_ms > 0.0 && t.rtf().is_some());

    server::stop().await.unwrap();
    assert!(clone::prepare_in(&app_data, id).await.is_err(), "nothing to register with after a stop");
}

#[tokio::test(flavor = "multi_thread")]
async fn a_crashed_server_is_restarted_found_again_and_its_voices_registered_again() {
    let _s = serial().await;
    install_fake("base", "Vulkan0");
    let app_data = temp_dir("crash-appdata");
    let id = "dave-123456";
    make_voice(&app_data, id, 200.0, Some("Hi there."));
    let l = test_launch("crash", Mode::Clone, Variant::Vulkan);
    let first = server::start_launch(l.clone(), server::stop_epoch()).await.unwrap();
    let pid = process_host::get(&l.helper_name).unwrap().pid;
    assert!(!clone::prepare_in(&app_data, id).await.unwrap().cached);

    // It crashes; the process host restarts it.
    let root = first.base_url.trim_end_matches("/v1").to_string();
    let _ = reqwest::Client::builder()
        .no_proxy()
        .build()
        .unwrap()
        .post(format!("{root}/sigil/test/crash"))
        .bearer_auth(&first.api_key)
        .timeout(Duration::from_secs(5))
        .send()
        .await;
    assert!(
        wait_until(Duration::from_secs(10), || process_host::get(&l.helper_name).is_some_and(|i| i.restarts == 1)).await,
        "{:?}",
        process_host::get(&l.helper_name)
    );
    // The watcher shows the restart as "starting" (the event that makes TypeScript start again); the
    // process host waits 1 s before the restart, far longer than a watcher round.
    assert!(
        wait_until(Duration::from_secs(5), || server::status().state == TtsHelperState::Starting).await,
        "{:?}",
        server::status()
    );

    // A start with the same settings (TypeScript after the "starting" event) waits for that restart
    // instead of spawning another server, and gets its endpoint. It is ready the moment the start
    // returns, not only after the watcher's next look.
    let again = server::start_launch(l.clone(), server::stop_epoch()).await.expect("the restarted server is found");
    assert_eq!(server::status().state, TtsHelperState::Ready, "ready when the start returns");

    // TypeScript's next step: the new process has none of the old one's voices, so the voice is
    // registered again from the cached speaker data, right away (no wait for the watcher).
    assert!(clone::prepare_in(&app_data, id).await.expect("prepare right after the start").cached);
    assert_eq!(helper_log(&l.helper_name, "fake-qwen register wav").len(), 1);
    assert_eq!(helper_log(&l.helper_name, "fake-qwen register latents").len(), 1);

    let info = process_host::get(&l.helper_name).unwrap();
    assert_eq!((info.state, info.restarts), (HelperState::Running, 1), "{info:?}");
    assert_ne!(info.pid, pid, "a new process");
    assert_eq!(get_status(&format!("{}/models", again.base_url), Some(&again.api_key)).await, 200);

    // The watcher's later rounds keep that registration (the restart was taken into account once).
    tokio::time::sleep(Duration::from_millis(1_000)).await;
    assert_eq!(server::status().state, TtsHelperState::Ready);
    assert!(clone::prepare_in(&app_data, id).await.unwrap().cached);
    assert_eq!(helper_log(&l.helper_name, "fake-qwen register latents").len(), 1, "registered once");
    server::stop().await.unwrap();
}

#[tokio::test(flavor = "multi_thread")]
async fn clone_is_refused_by_a_preset_server() {
    let _s = serial().await;
    install_fake("custom_voice", "Vulkan0");
    let app_data = temp_dir("clone-preset");
    make_voice(&app_data, "carol-777777", 200.0, Some("Hi."));
    let l = test_launch("clone-preset", Mode::Preset, Variant::Vulkan);
    server::start_launch(l, server::stop_epoch()).await.unwrap();
    let e = clone::prepare_in(&app_data, "carol-777777").await.unwrap_err();
    assert!(e.contains("cannot clone"), "{e}");
    server::stop().await.unwrap();
}

// ---------------------------------------------------------------------------------------------
// The real server (ignored; the P6-E manual verification through Sigil's own code path)
// ---------------------------------------------------------------------------------------------

/// The first file in `dir` whose name starts with `prefix` and ends with `.gguf`.
fn find_gguf(dir: &Path, prefix: &str) -> Option<PathBuf> {
    let mut hits: Vec<PathBuf> = std::fs::read_dir(dir)
        .ok()?
        .filter_map(|e| e.ok().map(|e| e.path()))
        .filter(|p| {
            p.file_name()
                .and_then(|n| n.to_str())
                .is_some_and(|n| n.starts_with(prefix) && n.ends_with(".gguf"))
        })
        .collect();
    hits.sort();
    hits.into_iter().next()
}

/// Time to first audio, real-time factor and cloning from a library clip with the real server:
///
/// ```text
/// $env:SIGIL_QWENTTS_EXE = "$env:LOCALAPPDATA\sigil-native\packs\qwentts-server-cuda13\sigil-tts-server.exe"
/// $env:SIGIL_TEST_MODELS_DIR = "<folder with qwen-talker-<size>-base-Q8_0.gguf and qwen-tokenizer-12hz-Q8_0.gguf>"
/// # optional: SIGIL_TEST_REF_WAV / SIGIL_TEST_REF_TEXT (default: freeman.wav / freeman.txt in the models folder),
/// #           SIGIL_QWENTTS_DLL_DIRS (cuda13: folders with cudart64_13 / cublas64_13; default: the NVIDIA pack or CUDA_PATH),
/// #           SIGIL_GPU_LOCK (a GPU timing lock file to wait for, hold and release)
/// cargo test --manifest-path src-tauri/Cargo.toml real_model_clone_end_to_end -- --ignored --nocapture
/// ```
#[tokio::test(flavor = "multi_thread")]
#[ignore = "needs a built sigil-tts-server.exe (SIGIL_QWENTTS_EXE) and Qwen3-TTS GGUFs (SIGIL_TEST_MODELS_DIR)"]
async fn real_model_clone_end_to_end() {
    let _s = serial().await;
    let (Some(exe), Some(dir)) = (std::env::var_os("SIGIL_QWENTTS_EXE"), std::env::var_os("SIGIL_TEST_MODELS_DIR")) else {
        eprintln!("SIGIL_QWENTTS_EXE / SIGIL_TEST_MODELS_DIR not set: skipped");
        return;
    };
    let exe = PathBuf::from(exe);
    let dir = PathBuf::from(dir);
    let talker = find_gguf(&dir, "qwen-talker-0.6b-base")
        .or_else(|| find_gguf(&dir, "qwen-talker-1.7b-base"))
        .expect("a qwen-talker-*-base-*.gguf in SIGIL_TEST_MODELS_DIR");
    let codec = find_gguf(&dir, "qwen-tokenizer-12hz").expect("a qwen-tokenizer-12hz-*.gguf in SIGIL_TEST_MODELS_DIR");
    let manifest: serde_json::Value = std::fs::read(exe.with_file_name("sigil-pack.json"))
        .ok()
        .and_then(|b| serde_json::from_slice(&b).ok())
        .unwrap_or_default();
    let variant = match manifest["variant"].as_str().unwrap_or("cuda13") {
        "cpu" => Variant::Cpu,
        "vulkan" => Variant::Vulkan,
        _ => Variant::Cuda13,
    };
    let dll_dirs: Vec<PathBuf> = match std::env::var("SIGIL_QWENTTS_DLL_DIRS") {
        Ok(v) => v.split(';').filter(|s| !s.is_empty()).map(PathBuf::from).collect(),
        Err(_) if variant == Variant::Cuda13 => {
            let dirs = server::nvidia_dll_dirs();
            if dirs.is_empty() {
                std::env::var_os("CUDA_PATH")
                    .map(PathBuf::from)
                    .map(|c| vec![c.join("bin").join("x64"), c.join("bin")])
                    .unwrap_or_default()
            } else {
                dirs
            }
        }
        Err(_) => Vec::new(),
    };
    let size = if talker.to_string_lossy().contains("0.6b") { Size::Small } else { Size::Large };
    let lock = std::env::var_os("SIGIL_GPU_LOCK").map(PathBuf::from);
    if let Some(l) = &lock {
        while l.exists() {
            eprintln!("waiting for the GPU timing lock {}", l.display());
            tokio::time::sleep(Duration::from_secs(15)).await;
        }
        std::fs::write(l, format!("P6-E real_model_clone_end_to_end (pid {})", std::process::id())).unwrap();
    }
    test_hooks::set(|spec| spec);
    let launch = Launch {
        helper_name: "t-qwen-real".into(),
        exe,
        talker_path: talker.clone(),
        codec_path: codec,
        model: plan::talker_id(size, Mode::Clone).unwrap().into(),
        size,
        mode: Mode::Clone,
        variant,
        language: "auto".into(),
        vram_mb: if size == Size::Small { 1500 } else { 3200 },
        load_timeout_ms: 300_000,
        dll_dirs,
    };
    let run = async {
        let t0 = Instant::now();
        let endpoint = server::start_launch(launch.clone(), server::stop_epoch()).await.expect("the real server starts");
        eprintln!("started {} on {} in {} ms (load, info check and warm-up)", launch.model, variant.as_str(), t0.elapsed().as_millis());
        for l in process_host::logs(&launch.helper_name).iter().filter(|l| l.text.contains("warm-up") || l.text.contains("[Load]")) {
            eprintln!("  {}", l.text);
        }

        // A library voice from the reference clip.
        let wav = std::env::var_os("SIGIL_TEST_REF_WAV").map(PathBuf::from).unwrap_or_else(|| dir.join("freeman.wav"));
        let text = std::env::var("SIGIL_TEST_REF_TEXT")
            .ok()
            .or_else(|| std::fs::read_to_string(dir.join("freeman.txt")).ok())
            .unwrap_or_default();
        let app_data = temp_dir("real-appdata");
        let id = "reference-000001";
        let vdir = app_data.join("voices").join(id);
        std::fs::create_dir_all(&vdir).unwrap();
        std::fs::copy(&wav, vdir.join("ref.wav")).expect("SIGIL_TEST_REF_WAV (or freeman.wav in the models folder)");
        let meta = serde_json::json!({ "id": id, "name": "Reference", "transcript": text.trim(), "source": "imported" });
        std::fs::write(vdir.join("meta.json"), serde_json::to_vec(&meta).unwrap()).unwrap();

        let t = Instant::now();
        let v = clone::prepare_in(&app_data, id).await.expect("the clip registers");
        assert!(!v.cached);
        eprintln!("cloned from the clip in {} ms (transcript: {})", t.elapsed().as_millis(), !text.trim().is_empty());
        clone::forget_all();
        let t = Instant::now();
        assert!(clone::prepare_in(&app_data, id).await.unwrap().cached, "registered from the cached speaker data");
        eprintln!("re-registered from the cache in {} ms", t.elapsed().as_millis());

        for (i, sentence) in [
            "Hello! This is Sigil speaking with a cloned voice, streamed as it is made.",
            "The quick brown fox jumps over the lazy dog, then takes a well deserved nap in the warm afternoon sun.",
        ]
        .iter()
        .enumerate()
        {
            let body = serde_json::json!({ "input": sentence, "voice": v.voice, "language": "english", "response_format": "pcm" });
            let m = server::timed_speech(&endpoint, body, Duration::from_secs(300)).await.expect("speech");
            eprintln!(
                "sentence {}: first audio {:.0} ms, {:.2} s of audio in {:.0} ms, RTF {:.3}",
                i + 1,
                m.first_audio_ms,
                m.audio_ms / 1000.0,
                m.total_ms,
                m.rtf().unwrap_or(f64::NAN)
            );
            assert!(m.audio_ms > 500.0, "the clone spoke");
        }
        eprintln!("VRAM claim: {:?} MiB", claim_mb(&launch.model));
        server::stop().await.unwrap();
    };
    let result = tokio::time::timeout(Duration::from_secs(900), run).await;
    if let Some(l) = &lock {
        let _ = std::fs::remove_file(l);
    }
    result.expect("finished within 15 minutes");
}

// ---------------------------------------------------------------------------------------------
// The fake sigil-tts-server (runs in a child process)
// ---------------------------------------------------------------------------------------------

fn fake_spk() -> Vec<u8> {
    (0..16u32).flat_map(|i| (i as f32 * 0.25).to_le_bytes()).collect()
}

#[test]
#[ignore = "child-process entry point of the qwen tests; does nothing without SIGIL_QWEN_FAKE"]
fn fake_qwentts_server_entry() {
    if std::env::var(ROLE_ENV).is_err() {
        return;
    }
    let port: u16 = std::env::var("SIGIL_QWEN_FAKE_PORT").unwrap().parse().unwrap();
    let key = std::env::var(server::API_KEY_ENV).unwrap_or_default();
    let model_type = std::env::var("SIGIL_QWEN_FAKE_TYPE").unwrap_or_else(|_| "base".into());
    let backend = std::env::var("SIGIL_QWEN_FAKE_BACKEND").unwrap_or_else(|_| "Vulkan0".into());
    eprintln!("fake-qwen args: {}", std::env::var("SIGIL_QWEN_FAKE_ARGS").unwrap_or_default());
    eprintln!("[Load] Talker backend: {backend} (CPU threads: 8)");
    eprintln!("[WeightCtx] Loaded 316 tensors, 1765.0 MB into backend");
    eprintln!("[KVCache] Allocated: 28 layers, 8 KV heads, head_dim 128, max_seq_len 4096, 1 sets -> 896 MB");
    let listener = TcpListener::bind(("127.0.0.1", port)).expect("bind");
    let speakers = if model_type == "custom_voice" { vec!["serena", "vivian", "ryan"] } else { Vec::new() };
    let b64e = base64::engine::general_purpose::STANDARD;
    let mut voices: HashSet<String> = HashSet::new();
    for stream in listener.incoming() {
        let Ok(mut sock) = stream else { continue };
        let _ = sock.set_read_timeout(Some(Duration::from_secs(5)));
        let mut buf = Vec::new();
        let mut tmp = [0u8; 65536];
        let head_end = loop {
            if let Some(p) = buf.windows(4).position(|w| w == b"\r\n\r\n") {
                break Some(p + 4);
            }
            match sock.read(&mut tmp) {
                Ok(0) | Err(_) => break None,
                Ok(n) => buf.extend_from_slice(&tmp[..n]),
            }
        };
        let Some(head_end) = head_end else { continue };
        let head = String::from_utf8_lossy(&buf[..head_end]).to_string();
        let len: usize = head
            .lines()
            .find_map(|l| l.to_ascii_lowercase().strip_prefix("content-length:").map(|v| v.trim().parse().unwrap_or(0)))
            .unwrap_or(0);
        while buf.len() < head_end + len {
            match sock.read(&mut tmp) {
                Ok(0) | Err(_) => break,
                Ok(n) => buf.extend_from_slice(&tmp[..n]),
            }
        }
        let body = String::from_utf8_lossy(&buf[head_end..(head_end + len).min(buf.len())]).to_string();
        let mut words = head.split_whitespace();
        let method = words.next().unwrap_or("").to_string();
        let path = words.next().unwrap_or("/").to_string();
        let authed = head.lines().any(|l| {
            l.to_ascii_lowercase().starts_with("authorization:") && l.split_once(':').map(|(_, v)| v.trim()) == Some(format!("Bearer {key}").as_str())
        });
        let json = |status: &str, value: serde_json::Value| (status.to_string(), "application/json".to_string(), serde_json::to_vec(&value).unwrap());
        let (status, ctype, out) = match (method.as_str(), path.as_str()) {
            ("GET", "/health") => json("200 OK", serde_json::json!({ "status": "ok" })),
            _ if !authed => json("401 Unauthorized", serde_json::json!({ "error": { "message": "missing or wrong API key" } })),
            ("GET", "/v1/models") => json("200 OK", serde_json::json!({ "object": "list", "data": [] })),
            // A crash (the process host restarts the helper).
            ("POST", "/sigil/test/crash") => {
                eprintln!("fake-qwen crashing");
                std::process::exit(3);
            }
            ("GET", "/sigil/info") => json(
                "200 OK",
                serde_json::json!({
                    "server": "sigil-tts-server", "pack_version": "test", "variant": "test", "upstream_commit": "fake",
                    "model": "fake", "model_type": model_type, "sample_rate": 24000, "num_codebooks": 16,
                    "speakers": speakers, "languages": ["chinese", "english", "japanese"],
                }),
            ),
            ("POST", "/v1/audio/speech") => {
                eprintln!("fake-qwen speech: {body}");
                let req: serde_json::Value = serde_json::from_str(&body).unwrap_or_default();
                let voice = req["voice"].as_str().unwrap_or("");
                if !voice.is_empty() && !voices.contains(voice) && !speakers.contains(&voice) {
                    json("400 Bad Request", serde_json::json!({ "error": { "message": format!("unknown voice '{voice}'") } }))
                } else {
                    ("200 OK".into(), "audio/pcm".into(), vec![0u8; 4800])
                }
            }
            ("POST", "/v1/audio/voices") => {
                let req: serde_json::Value = serde_json::from_str(&body).unwrap_or_default();
                let name = req["name"].as_str().unwrap_or("").to_ascii_lowercase();
                let ref_text = req["ref_text"].as_str().unwrap_or("");
                if model_type != "base" {
                    json("400 Bad Request", serde_json::json!({ "error": { "message": format!("voice registration is only valid for base models (loaded: {model_type})") } }))
                } else if let Some(wav) = req["wav_b64"].as_str() {
                    let wav = b64e.decode(wav).unwrap_or_default();
                    eprintln!("fake-qwen register wav {name} ({} bytes) ref_text={ref_text}", wav.len());
                    voices.insert(name.clone());
                    let mut rvq = wav.len().to_le_bytes().to_vec();
                    rvq.extend_from_slice(&[7u8; 14]);
                    json(
                        "200 OK",
                        serde_json::json!({
                            "name": name, "status": "registered", "extracted": true, "ref_T": 37, "spk_dim": 16, "num_codebooks": 16,
                            "spk_b64": b64e.encode(fake_spk()), "rvq_b64": b64e.encode(rvq),
                        }),
                    )
                } else {
                    let spk = b64e.decode(req["spk_b64"].as_str().unwrap_or("")).unwrap_or_default();
                    if spk != fake_spk() {
                        json("400 Bad Request", serde_json::json!({ "error": { "message": "these latents do not belong to this model" } }))
                    } else {
                        eprintln!("fake-qwen register latents {name} ref_text={ref_text}");
                        voices.insert(name.clone());
                        json("200 OK", serde_json::json!({ "name": name, "status": "registered", "extracted": false, "ref_T": 37, "spk_dim": 16 }))
                    }
                }
            }
            _ => json("404 Not Found", serde_json::json!({ "error": { "message": "not found" } })),
        };
        let _ = sock.write_all(format!("HTTP/1.1 {status}\r\nContent-Type: {ctype}\r\nContent-Length: {}\r\nConnection: close\r\n\r\n", out.len()).as_bytes());
        let _ = sock.write_all(&out);
        let _ = sock.flush();
    }
}
