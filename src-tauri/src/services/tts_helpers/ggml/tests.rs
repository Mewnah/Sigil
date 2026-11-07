//! Tests of the ggml helper runner, the request adapter and the four engine plans (P6-F).
//!
//! The lifecycle tests run the real `process_host` with this test binary re-run as a fake server
//! (`fake_ggml_server_entry`: the routes, key checks and answer shapes of the four servers); the
//! adapter tests run a fake upstream in process. Nothing touches the network.

use std::collections::HashMap;
use std::io::{Read, Write};
use std::net::TcpListener;
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicU64, AtomicUsize, Ordering};
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};

use serde_json::{json, Map, Value};

use super::adapter::{self, Upstream};
use super::launch::{self, build_spec, find_program, pick_pack, safe_pack_arg, verify_sigil_pack, PackChoice, PlanEnv, Readiness};
use super::runner::{rate_of_content_type, Slot};
use super::*;
use crate::services::model_store::{ArchiveFormat, Catalog};
use crate::services::process_host::{self, HelperPriority, PortRequest};
use crate::services::tts_helpers::{breeze, magpie, omnivoice, s2, TtsHelperEngine, TtsHelperState};

// ---------------------------------------------------------------------------------------------
// Fixtures
// ---------------------------------------------------------------------------------------------

fn temp_dir(tag: &str) -> PathBuf {
    static N: AtomicU64 = AtomicU64::new(0);
    let nanos = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|d| d.as_nanos())
        .unwrap_or_default();
    let dir = std::env::temp_dir()
        .join("sigil-p6f-tests")
        .join(format!("{tag}-{}-{nanos}-{}", std::process::id(), N.fetch_add(1, Ordering::Relaxed)));
    std::fs::create_dir_all(&dir).unwrap();
    dir
}

fn catalog() -> Catalog {
    Catalog::embedded().expect("embedded catalog")
}

/// Fake installs of catalog entries under `root`: model files with a few bytes, runtime packs as
/// their program (in the layout of the real zip) plus `sigil-pack.json` for Sigil-built packs.
fn install(root: &Path, catalog: &Catalog, ids: &[&str]) -> HashMap<String, PathBuf> {
    let mut out = HashMap::new();
    for id in ids {
        let dir = root.join(id.replace('/', "_"));
        std::fs::create_dir_all(&dir).unwrap();
        let entry = catalog.get(id).unwrap_or_else(|| panic!("{id} is in the catalog"));
        if id.starts_with("runtime/") {
            let program_bytes = format!("MZ fake {id}");
            match *id {
                i if i.starts_with("runtime/nemo-speech-") => {
                    // The v0.2.0 zips keep a top-level folder named after the archive
                    // (scripts\windows\package-release.ps1 includes the base directory).
                    let variant = entry.variant.clone().unwrap_or_default();
                    let bin = dir.join(format!("nemo-speech-0.2.0-windows-x86_64-{variant}")).join("bin");
                    std::fs::create_dir_all(&bin).unwrap();
                    std::fs::write(bin.join("nemo-speech.exe"), &program_bytes).unwrap();
                    std::fs::write(bin.join("ggml.dll"), b"MZ").unwrap();
                }
                i if i.starts_with("runtime/breeze-server-") => {
                    std::fs::write(dir.join("breeze-server.exe"), &program_bytes).unwrap();
                }
                i if i.starts_with("runtime/crispasr-") => {
                    let top = dir.join("crispasr-windows-x86_64");
                    std::fs::create_dir_all(&top).unwrap();
                    std::fs::write(top.join("CrispASR.exe"), &program_bytes).unwrap();
                }
                i if i.starts_with("runtime/s2-server-") => {
                    let variant = entry.variant.clone().unwrap_or_default();
                    write_s2_pack(&dir, &variant, &["--cuda", "0"], &["--gpu-layers", "18", "--codec-cpu"]);
                }
                _ => {}
            }
        } else {
            for f in &entry.files {
                let p = dir.join(&f.path);
                std::fs::create_dir_all(p.parent().unwrap()).unwrap();
                std::fs::write(&p, format!("fake {}", f.path)).unwrap();
            }
        }
        out.insert(id.to_string(), dir);
    }
    out
}

fn write_s2_pack(dir: &Path, variant: &str, server_args: &[&str], low_vram_args: &[&str]) {
    let exe = dir.join("s2.exe");
    let bytes = format!("MZ fake s2 {variant}");
    std::fs::write(&exe, &bytes).unwrap();
    let sha = launch::sha256_cached(&exe).unwrap();
    let manifest = json!({
        "engine": "s2-server", "variant": variant, "version": "2c33261-s1", "program": "s2.exe",
        "files": [{ "path": "s2.exe", "size": bytes.len(), "sha256": sha }],
        "server_args": server_args, "low_vram_args": low_vram_args,
    });
    std::fs::write(dir.join("sigil-pack.json"), serde_json::to_vec_pretty(&manifest).unwrap()).unwrap();
}

const NVIDIA_GPU: GpuFacts = GpuFacts { any_gpu: true, cuda: true };
const AMD_GPU: GpuFacts = GpuFacts { any_gpu: true, cuda: false };
const NO_GPU: GpuFacts = GpuFacts { any_gpu: false, cuda: false };

/// Run `f` with a planning environment over `installed` (and NVIDIA runtime folders `nvidia`).
fn with_env<T>(
    catalog: &Catalog,
    installed: &HashMap<String, PathBuf>,
    gpu: GpuFacts,
    nvidia: Vec<PathBuf>,
    app_data: Option<PathBuf>,
    f: impl FnOnce(&PlanEnv<'_>) -> T,
) -> T {
    let lookup = |id: &str| installed.get(id).cloned();
    let dirs = move || nvidia.clone();
    let env = PlanEnv {
        catalog,
        installed: &lookup,
        gpu,
        nvidia_dirs: &dirs,
        app_data,
    };
    f(&env)
}

fn config(v: Value) -> GgmlConfig {
    serde_json::from_value(v).unwrap()
}

/// A library voice `id` under `app_data` (a 3 s, 24 kHz sine clip) with `transcript`.
fn make_voice(app_data: &Path, id: &str, transcript: Option<&str>) {
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
        let s = (i as f32 * 220.0 * std::f32::consts::TAU / 24_000.0).sin() * 0.3;
        w.write_sample((s * 32767.0) as i16).unwrap();
    }
    w.finalize().unwrap();
    let meta = json!({ "id": id, "name": id, "transcript": transcript, "language": "en", "source": "recorded" });
    std::fs::write(dir.join("meta.json"), serde_json::to_vec(&meta).unwrap()).unwrap();
}

/// One part of a `multipart/form-data` body: `(name, filename, bytes)`.
type Field = (String, Option<String>, Vec<u8>);

/// The fields of a `multipart/form-data` body.
fn multipart_fields(body: &[u8], content_type: &str) -> Vec<Field> {
    let boundary = content_type.split("boundary=").nth(1).expect("boundary").trim().trim_matches('"');
    let delim = format!("--{boundary}");
    let mut out = Vec::new();
    let mut rest = body;
    let find = |hay: &[u8], needle: &[u8]| hay.windows(needle.len()).position(|w| w == needle);
    while let Some(start) = find(rest, delim.as_bytes()) {
        rest = &rest[start + delim.len()..];
        if rest.starts_with(b"--") {
            break;
        }
        rest = rest.strip_prefix(b"\r\n").unwrap_or(rest);
        let head_end = find(rest, b"\r\n\r\n").expect("part head");
        let head = String::from_utf8_lossy(&rest[..head_end]).to_string();
        let data_start = head_end + 4;
        let end = find(&rest[data_start..], format!("\r\n{delim}").as_bytes()).expect("part end");
        let data = rest[data_start..data_start + end].to_vec();
        let disposition = head
            .lines()
            .find(|l| l.to_ascii_lowercase().starts_with("content-disposition:"))
            .unwrap_or("")
            .to_string();
        let attr = |key: &str| {
            disposition
                .split(';')
                .map(str::trim)
                .find_map(|p| p.strip_prefix(&format!("{key}=\"")).map(|v| v.trim_end_matches('"').to_string()))
        };
        out.push((attr("name").unwrap_or_default(), attr("filename"), data));
        rest = &rest[data_start + end + 2..];
    }
    out
}

fn field<'a>(fields: &'a [Field], name: &str) -> Option<&'a Field> {
    fields.iter().find(|f| f.0 == name)
}

fn text_field(fields: &[Field], name: &str) -> Option<String> {
    field(fields, name).map(|f| String::from_utf8_lossy(&f.2).to_string())
}

// ---------------------------------------------------------------------------------------------
// Shared pieces
// ---------------------------------------------------------------------------------------------

#[test]
fn variant_parse_round_trips() {
    for v in Variant::ALL {
        assert_eq!(Variant::parse(v.as_str()).unwrap(), Some(v));
        assert_eq!(v.on_gpu(), v != Variant::Cpu);
    }
    assert_eq!(Variant::parse("").unwrap(), None);
    assert_eq!(Variant::parse(" Auto ").unwrap(), None);
    assert!(Variant::parse("metal").is_err());
    assert!(Variant::Cuda.nvidia() && Variant::Cuda13.nvidia() && !Variant::Vulkan.nvidia());
}

#[test]
fn config_is_camel_case_and_ignores_unknown_members() {
    let c = config(json!({
        "model": "q4_k", "variant": "vulkan", "language": "ja", "voice": "Sofia", "referenceVoiceId": "alice-1",
        "quality": "fast", "lowVram": true, "seed": 5, "somethingNew": [1, 2],
    }));
    assert_eq!(c.model, "q4_k");
    assert_eq!(c.reference_voice_id, "alice-1");
    assert!(c.low_vram);
    assert_eq!(config(json!({})), GgmlConfig::default());
}

#[test]
fn language_base_and_timeouts() {
    assert_eq!(language_base("pt-BR").as_deref(), Some("pt"));
    assert_eq!(language_base(" zh_Hant ").as_deref(), Some("zh"));
    assert_eq!(language_base("auto"), None);
    assert_eq!(language_base(""), None);
    assert_eq!(language_base("x1"), None);
    assert_eq!(load_timeout_ms(0), 60_000);
    assert_eq!(load_timeout_ms(1 << 30), 90_000);
    assert_eq!(load_timeout_ms(6 << 30), 240_000);
    assert_eq!(load_timeout_ms(u64::MAX), 600_000);
}

#[test]
fn packs_are_picked_like_the_ts_mirror() {
    let catalog = catalog();
    let packs = [
        PackChoice { variant: Variant::Cuda13, runtime: "runtime/crispasr-cuda13" },
        PackChoice { variant: Variant::Vulkan, runtime: "runtime/crispasr-vulkan" },
        PackChoice { variant: Variant::Cpu, runtime: "runtime/crispasr-cpu" },
    ];
    let root = temp_dir("packs");
    let none = HashMap::new();
    // Nothing installed: CUDA 13 on a CUDA GPU would need both halves, so Vulkan is suggested.
    with_env(&catalog, &none, NVIDIA_GPU, vec![], None, |env| {
        assert_eq!(pick_pack("X", &packs, None, env).unwrap().variant, Variant::Vulkan);
        assert_eq!(pick_pack("X", &packs, Some(Variant::Cuda13), env).unwrap().variant, Variant::Cuda13);
        assert!(pick_pack("X", &packs, Some(Variant::Cuda), env).unwrap_err().contains("no cuda build"));
    });
    with_env(&catalog, &none, NO_GPU, vec![], None, |env| {
        assert_eq!(pick_pack("X", &packs, None, env).unwrap().variant, Variant::Cpu);
        assert!(pick_pack("X", &packs, Some(Variant::Vulkan), env).unwrap_err().contains("no graphics card"));
    });
    with_env(&catalog, &none, AMD_GPU, vec![], None, |env| {
        assert!(pick_pack("X", &packs, Some(Variant::Cuda13), env).unwrap_err().contains("NVIDIA"));
        assert_eq!(pick_pack("X", &packs, None, env).unwrap().variant, Variant::Vulkan);
    });
    // The NVIDIA runtime alone (the voice changer's) makes CUDA 13 the half-installed choice.
    let nv = install(&root, &catalog, &[launch::NVIDIA_RUNTIME_ID]);
    with_env(&catalog, &nv, NVIDIA_GPU, vec![], None, |env| {
        assert_eq!(pick_pack("X", &packs, None, env).unwrap().variant, Variant::Cuda13);
    });
    // An installed Vulkan pack wins over a half-installed CUDA one; a complete CUDA one wins over both.
    let mut both = install(&root, &catalog, &["runtime/crispasr-vulkan", launch::NVIDIA_RUNTIME_ID]);
    with_env(&catalog, &both, NVIDIA_GPU, vec![], None, |env| {
        assert_eq!(pick_pack("X", &packs, None, env).unwrap().variant, Variant::Vulkan);
    });
    both.extend(install(&root, &catalog, &["runtime/crispasr-cuda13"]));
    with_env(&catalog, &both, NVIDIA_GPU, vec![], None, |env| {
        assert_eq!(pick_pack("X", &packs, None, env).unwrap().variant, Variant::Cuda13);
    });
    // An engine without a CPU pack refuses a PC without a GPU.
    let gpu_only = [PackChoice { variant: Variant::Vulkan, runtime: "runtime/crispasr-vulkan" }];
    with_env(&catalog, &none, NO_GPU, vec![], None, |env| {
        assert!(pick_pack("X", &gpu_only, None, env).unwrap_err().contains("needs a graphics card"));
    });
}

#[test]
fn programs_are_found_in_the_pack_layouts() {
    let root = temp_dir("find");
    assert!(find_program(&root, "a.exe").is_none());
    std::fs::create_dir_all(root.join("zz").join("bin")).unwrap();
    std::fs::create_dir_all(root.join("aa").join("tools")).unwrap();
    std::fs::write(root.join("zz").join("bin").join("Server.EXE"), b"MZ").unwrap();
    std::fs::write(root.join("aa").join("tools").join("server.exe"), b"MZ").unwrap();
    // Case-insensitive, and bin/ folders are searched first at every level.
    assert_eq!(find_program(&root, "server.exe").unwrap(), root.join("zz").join("bin").join("Server.EXE"));
    std::fs::write(root.join("server.exe"), b"MZ").unwrap();
    assert_eq!(find_program(&root, "server.exe").unwrap(), root.join("server.exe"));
    // Not deeper than four levels.
    let deep = root.join("1").join("2").join("3").join("4").join("5");
    std::fs::create_dir_all(&deep).unwrap();
    std::fs::write(deep.join("deep.exe"), b"MZ").unwrap();
    assert!(find_program(&root, "deep.exe").is_none());
}

#[test]
fn sigil_packs_are_checked() {
    let dir = temp_dir("sigil-pack");
    write_s2_pack(&dir, "vulkan", &["--vulkan", "0"], &[]);
    let (exe, m) = verify_sigil_pack(&dir, "s2-server", Variant::Vulkan, "s2.exe").unwrap();
    assert_eq!(exe, dir.join("s2.exe"));
    assert_eq!(m.server_args, vec!["--vulkan", "0"]);
    assert!(verify_sigil_pack(&dir, "qwentts-server", Variant::Vulkan, "s2.exe").unwrap_err().contains("not a qwentts-server pack"));
    assert!(verify_sigil_pack(&dir, "s2-server", Variant::Cuda13, "s2.exe").unwrap_err().contains("is the vulkan build"));
    // A replaced program is refused.
    std::fs::write(dir.join("s2.exe"), b"MZ something else entirely").unwrap();
    assert!(verify_sigil_pack(&dir, "s2-server", Variant::Vulkan, "s2.exe").unwrap_err().contains("does not match"));
    // Arguments that would move the server off loopback or point it elsewhere are refused.
    write_s2_pack(&dir, "vulkan", &["--host", "0.0.0.0"], &[]);
    assert!(verify_sigil_pack(&dir, "s2-server", Variant::Vulkan, "s2.exe").unwrap_err().contains("--host"));
    assert!(safe_pack_arg("--gpu-layers") && safe_pack_arg("18") && safe_pack_arg("--codec-cpu"));
    for bad in ["--port=1", "--model", "C:\\x.gguf", "a b", "--voice-dir", "", "--server"] {
        assert!(!safe_pack_arg(bad), "{bad}");
    }
}

fn sample_launch() -> Launch {
    Launch {
        engine: TtsHelperEngine::Magpie,
        helper_name: "tts-magpie".into(),
        program: PathBuf::from(r"C:\packs\nemo\bin\nemo-speech.exe"),
        args: vec!["serve".into(), "--port".into(), launch::port_arg()],
        env: vec![(magpie::API_KEY_ENV.into(), launch::key_env())],
        env_remove: vec!["NEMO_SPEECH_*".into()],
        server_key: true,
        readiness: Readiness::Health("/health".into()),
        adapter: None,
        model: "tts/magpie-tts-357m-v2602".into(),
        variant: Variant::Cuda,
        pcm_rate: 22_050,
        vram_mb: 1500,
        load_timeout_ms: 90_000,
        dll_dirs: vec![PathBuf::from(r"C:\nv\bin")],
        warmup: None,
        summary: String::new(),
    }
}

#[test]
fn spec_binds_loopback_and_keeps_the_key_off_the_command_line() {
    let l = sample_launch();
    let spec = build_spec(&l);
    assert_eq!(spec.port, PortRequest::Auto);
    assert!(spec.api_key);
    assert!(spec.args.iter().any(|a| a == process_host::spawn::PORT_PLACEHOLDER));
    assert!(!spec.args.iter().any(|a| a.contains(process_host::spawn::API_KEY_PLACEHOLDER)));
    assert!(spec.env.iter().any(|(k, v)| k == magpie::API_KEY_ENV && v == process_host::spawn::API_KEY_PLACEHOLDER));
    let path = spec.env.iter().find(|(k, _)| k == "PATH").map(|(_, v)| v.clone()).unwrap();
    assert!(path.starts_with(r"C:\nv\bin"));
    assert_eq!(spec.health.as_ref().unwrap().path, "/health");
    assert_eq!(spec.health.as_ref().unwrap().timeout_ms, 90_000);
    assert_eq!(spec.priority, HelperPriority::Normal);
    assert_eq!(spec.cwd.as_deref(), Some(Path::new(r"C:\packs\nemo\bin")));
    process_host::spawn::validate_name(&spec.name).unwrap();

    let mut cpu = sample_launch();
    cpu.variant = Variant::Cpu;
    cpu.server_key = false;
    cpu.readiness = Readiness::AnyHttp;
    cpu.dll_dirs.clear();
    let spec = build_spec(&cpu);
    assert_eq!(spec.priority, HelperPriority::BelowNormal);
    assert!(!spec.api_key);
    assert!(spec.health.is_none());
    assert!(!spec.env.iter().any(|(k, _)| k == "PATH"));

    // Bookkeeping does not make another server; arguments do.
    let mut other = sample_launch();
    other.model = "x".into();
    other.vram_mb = 1;
    other.warmup = Some(json!({}));
    assert!(other.same_server(&sample_launch()));
    other.args.push("--tts.voice-name".into());
    assert!(!other.same_server(&sample_launch()));
}

// ---------------------------------------------------------------------------------------------
// Catalog
// ---------------------------------------------------------------------------------------------

fn engine_ids() -> Vec<&'static str> {
    let mut ids: Vec<&'static str> = vec![
        magpie::RUNTIME_CUDA,
        magpie::RUNTIME_VULKAN,
        magpie::RUNTIME_CPU,
        magpie::CODEC_ID,
        breeze::RUNTIME_VULKAN,
        omnivoice::RUNTIME_CUDA13,
        omnivoice::RUNTIME_VULKAN,
        omnivoice::RUNTIME_CPU,
        omnivoice::CODEC_ID,
        s2::RUNTIME_CUDA13,
        s2::RUNTIME_VULKAN,
        s2::RUNTIME_CPU,
        s2::TOKENIZER_ID,
    ];
    for m in magpie::MODELS {
        ids.push(m.model);
        ids.push(m.tokenizer);
    }
    ids.extend(breeze::MODELS.iter().map(|m| m.1));
    ids.extend(omnivoice::MODELS.iter().map(|m| m.1));
    ids.extend(s2::MODELS.iter().map(|m| m.1));
    ids
}

#[test]
fn catalog_has_every_pack_and_model_the_engines_name() {
    let catalog = catalog();
    for id in engine_ids() {
        let e = catalog.get(id).unwrap_or_else(|| panic!("{id} is missing from tts-ggml-experimental.json"));
        if e.status == crate::services::model_store::PublishStatus::Published {
            for f in &e.files {
                assert!(f.sha256.is_some() && f.size > 0, "{id}: {} is published but not pinned", f.path);
            }
        }
        if id.starts_with("tts/") && !id.contains("tokenizer") && !id.contains("codec") {
            assert!(e.min_vram_mb.is_some_and(|mb| mb >= 1000), "{id} needs a VRAM footprint");
        }
    }
    // Runtime packs carry the variant their engine plans with.
    for (id, variant) in [
        (magpie::RUNTIME_CUDA, "cuda"),
        (magpie::RUNTIME_VULKAN, "vulkan"),
        (magpie::RUNTIME_CPU, "cpu"),
        (breeze::RUNTIME_VULKAN, "vulkan"),
        (omnivoice::RUNTIME_CUDA13, "cuda13"),
        (s2::RUNTIME_CUDA13, "cuda13"),
    ] {
        assert_eq!(catalog.get(id).unwrap().variant.as_deref(), Some(variant), "{id}");
    }
    // CUDA 13 packs install the NVIDIA runtime with them.
    for id in [omnivoice::RUNTIME_CUDA13, s2::RUNTIME_CUDA13] {
        assert!(catalog.get(id).unwrap().requires.iter().any(|r| r == launch::NVIDIA_RUNTIME_ID), "{id}");
    }
    // The Magpie v2602 tokenizer is the pinned .nemo member set, model_config.yaml included.
    let tok = catalog.get(magpie::MODELS[0].tokenizer).unwrap();
    assert_eq!(tok.files.len(), 7);
    assert!(tok.files.iter().any(|f| f.path == "model_config.yaml" && f.sha256.is_some()));
}

/// The Magpie packs and models are the facts of NVIDIA's NeMo-Speech.cpp v0.2.0 release: the three
/// Windows zips (GitHub's asset digests) and the v2607 GGUF and NanoCodec (its models/index.json,
/// whose revisions and hashes are the Hugging Face commits' LFS values).
#[test]
fn magpie_catalog_matches_nemo_speech_v020() {
    use crate::services::model_store::PublishStatus;
    let catalog = catalog();
    for (id, variant, size, sha256) in [
        (magpie::RUNTIME_CUDA, "cuda", 102_183_465u64, "9ffe5e842c39a7d3211f3b2c3b3aa1bca8c0b2455d96d961ed5f22ec5108f98d"),
        (magpie::RUNTIME_VULKAN, "vulkan", 19_302_464, "edf15a04ba98740aa0ab75ae4681ac7ad0a488b22b35ce2cbebd10590bb1c1e1"),
        (magpie::RUNTIME_CPU, "cpu", 5_528_418, "82c80451086e86194aba9af19f800da77cd66d1c28d993d0efceffd6cdb49bb8"),
    ] {
        let e = catalog.get(id).unwrap();
        assert_eq!((e.version.as_str(), e.variant.as_deref()), ("0.2.0", Some(variant)), "{id}");
        assert_eq!(e.status, PublishStatus::Published, "{id}");
        let name = format!("nemo-speech-0.2.0-windows-x86_64-{variant}.zip");
        assert_eq!(e.files.len(), 1, "{id}");
        let f = &e.files[0];
        assert_eq!((f.path.as_str(), f.size, f.sha256.as_deref()), (name.as_str(), size, Some(sha256)), "{id}");
        assert_eq!(f.sources.len(), 1, "{id}");
        assert_eq!(f.sources[0].url, format!("https://github.com/NVIDIA/NeMo-Speech.cpp/releases/download/v0.2.0/{name}"), "{id}");
        assert_eq!(f.sources[0].revision.as_deref(), Some("v0.2.0"), "{id}");
        assert_eq!(f.archive.as_ref().map(|a| (a.format, a.strip)), Some((ArchiveFormat::Zip, 0)), "{id}");
    }
    // No model asks for a newer server than the packs the catalog ships.
    let needed = magpie::MODELS.iter().map(|m| m.min_pack).max().unwrap();
    for runtime in [magpie::RUNTIME_CUDA, magpie::RUNTIME_VULKAN, magpie::RUNTIME_CPU] {
        let version = &catalog.get(runtime).unwrap().version;
        let parts: Vec<u32> = version.split('.').map(|p| p.parse().unwrap()).collect();
        assert!((parts[0], parts[1], parts[2]) >= needed, "{runtime} {version} is older than a model needs: {needed:?}");
    }
    assert_eq!(magpie::MODELS[1].min_pack, (0, 2, 0));

    // v2607 and NanoCodec: file name, size, sha256 and Hugging Face commit as in v0.2.0's index.
    let hf = |repo: &str, rev: &str, file: &str| format!("https://huggingface.co/{repo}/resolve/{rev}/{file}");
    for (id, file, size, sha256, repo, rev) in [
        (
            "tts/magpie-tts-357m-v2607",
            "magpie_tts_multilingual_357m.v2607.f16.gguf",
            568_663_328u64,
            "30d27551fc4a050e8095139f0185796024493373e83d49443c8ddba33c514e68",
            "nvidia/magpie_tts_multilingual_357m",
            "19806879b16d3f2ccf28fb112b1bcd16a3c7923e",
        ),
        (
            magpie::CODEC_ID,
            "nemo_nano_codec_22khz_1.89kbps_21.5fps.decoder.f16.gguf",
            78_823_104,
            "cc86d36d821a27cdc1d4ef600a3e2b0dabe76e88fcc2a8652d9543134c07ef2d",
            "nvidia/nemo-nano-codec-22khz-1.89kbps-21.5fps",
            "fc00890b604aa2de298d2641ffc6c5f6caf8c4d7",
        ),
    ] {
        let f = &catalog.get(id).unwrap().files[0];
        assert_eq!((f.path.as_str(), f.size, f.sha256.as_deref()), (file, size, Some(sha256)), "{id}");
        assert_eq!(f.sources[0].url, hf(repo, rev, file), "{id}");
        assert_eq!(f.sources[0].revision.as_deref(), Some(rev), "{id}");
    }
    // The v2607 tokenizer is one zip (members at its root), and each checkpoint requires its own
    // tokenizer: the server refuses a GGUF and a tokenizer of different checkpoints.
    let tok = catalog.get(magpie::MODELS[1].tokenizer).unwrap();
    assert_eq!(tok.files.len(), 1);
    assert_eq!(tok.files[0].archive.as_ref().map(|a| (a.format, a.strip)), Some((ArchiveFormat::Zip, 0)));
    for m in magpie::MODELS {
        let e = catalog.get(m.model).unwrap();
        assert!(e.requires.iter().any(|r| r == m.tokenizer) && e.requires.iter().any(|r| r == magpie::CODEC_ID), "{}", m.model);
        let other = magpie::MODELS.iter().find(|o| o.key != m.key).unwrap();
        assert!(!e.requires.iter().any(|r| r == other.tokenizer), "{} requires the other checkpoint's tokenizer", m.model);
    }
}

// ---------------------------------------------------------------------------------------------
// Engine plans
// ---------------------------------------------------------------------------------------------

#[test]
fn magpie_language_codes_and_speakers() {
    assert_eq!(magpie::language_code("en"), Some("en-US"));
    assert_eq!(magpie::language_code("pt-PT"), Some("pt-BR"));
    assert_eq!(magpie::language_code("ar"), Some("ar-MSA"));
    assert_eq!(magpie::language_code("ar_SA"), Some("ar-SA"));
    assert_eq!(magpie::language_code("ja-JP"), Some("ja-JP"));
    assert_eq!(magpie::language_code("ru"), None);
    assert_eq!(magpie::language_code("auto"), None);
    assert_eq!(magpie::speaker_for("").unwrap(), "John");
    assert_eq!(magpie::speaker_for("sofia").unwrap(), "Sofia");
    assert!(magpie::speaker_for("Alloy").is_err());
    assert_eq!(magpie::model_for("").unwrap().key, "v2602");
    assert_eq!(magpie::model_for("V2607").unwrap().key, "v2607");
    assert!(magpie::model_for("v9").is_err());
}

#[test]
fn magpie_plan_builds_the_server_command() {
    let catalog = catalog();
    let root = temp_dir("magpie");
    let m = magpie::MODELS[0];
    let none = HashMap::new();
    let err = with_env(&catalog, &none, NVIDIA_GPU, vec![], None, |env| magpie::plan(&GgmlConfig::default(), env).unwrap_err());
    assert!(err.contains("needs downloads first"), "{err}");
    assert!(err.contains(m.model) && err.contains(m.tokenizer) && err.contains(magpie::CODEC_ID), "{err}");

    let installed = install(&root, &catalog, &[m.model, m.tokenizer, magpie::CODEC_ID, magpie::RUNTIME_VULKAN, magpie::RUNTIME_CUDA]);
    let nv = vec![PathBuf::from(r"C:\nv")];
    let l = with_env(&catalog, &installed, NVIDIA_GPU, nv.clone(), None, |env| {
        magpie::plan(&config(json!({ "language": "de", "voice": "aria" })), env).unwrap()
    });
    assert_eq!(l.variant, Variant::Cuda);
    assert!(l.program.ends_with(r"bin\nemo-speech.exe") || l.program.ends_with("bin/nemo-speech.exe"));
    let args = l.args.join(" ");
    assert!(args.starts_with("serve --tts-model "), "{args}");
    assert!(args.ends_with("--host 127.0.0.1 --port {port} --no-ui --device cuda:0"), "{args}");
    // Voice and language go with every request, never on the command line.
    assert!(!args.contains("--tts.language-code") && !args.contains("--tts.voice-name"), "{args}");
    assert!(l.args.iter().any(|a| a.ends_with("magpie_tts_multilingual_357m.v2602.f16.gguf")));
    assert_eq!(l.env, vec![(magpie::API_KEY_ENV.to_string(), "{api_key}".to_string())]);
    assert_eq!(l.env_remove, vec!["NEMO_SPEECH_*".to_string()]);
    assert!(l.server_key && l.adapter.is_none());
    assert_eq!(l.pcm_rate, 22_050);
    assert_eq!(l.vram_mb, 1500);
    assert_eq!(l.dll_dirs, nv);
    let warmup = l.warmup.as_ref().unwrap();
    assert_eq!((&warmup["language"], &warmup["voice"]), (&json!("de-DE"), &json!("Aria")));
    // Another voice or language is the same server (TS reloads for neither either).
    let other = with_env(&catalog, &installed, NVIDIA_GPU, nv.clone(), None, |env| {
        magpie::plan(&config(json!({ "language": "fr", "voice": "Leo" })), env).unwrap()
    });
    assert!(other.same_server(&l));

    // Vulkan on AMD. Korean needs v2607, and the official packs speak no Japanese or Mandarin:
    // the warm-up then speaks English.
    for language in ["ko", "ja", "zh"] {
        let l = with_env(&catalog, &installed, AMD_GPU, vec![], None, |env| magpie::plan(&config(json!({ "language": language })), env).unwrap());
        assert_eq!(l.variant, Variant::Vulkan);
        assert!(l.args.join(" ").ends_with("--device vulkan"));
        assert_eq!(l.warmup.as_ref().unwrap()["language"], "en-US", "{language}");
        assert!(l.dll_dirs.is_empty());
    }
    // The checkpoints' languages are the TS mirror's (`magpieLanguages`).
    assert_eq!(magpie::MODELS[0].languages, ["en-US", "es-ES", "de-DE", "fr-FR", "it-IT", "vi-VN", "hi-IN"]);
    assert_eq!(&magpie::MODELS[1].languages[7..], ["ko-KR", "ar-MSA", "pt-BR"]);

    // A tokenizer folder without model_config.yaml is refused.
    std::fs::remove_file(installed[m.tokenizer].join("model_config.yaml")).unwrap();
    let err = with_env(&catalog, &installed, AMD_GPU, vec![], None, |env| magpie::plan(&GgmlConfig::default(), env).unwrap_err());
    assert!(err.contains("model_config.yaml"), "{err}");
}

#[test]
fn magpie_v2607_plan_takes_its_own_model_and_tokenizer() {
    let catalog = catalog();
    let root = temp_dir("magpie-v2607");
    let (old, new) = (magpie::MODELS[0], magpie::MODELS[1]);
    assert_eq!((old.key, new.key), ("v2602", "v2607"));
    let installed = install(
        &root,
        &catalog,
        &[old.model, old.tokenizer, new.model, new.tokenizer, magpie::CODEC_ID, magpie::RUNTIME_VULKAN],
    );
    // The model store extracts the v2607 tokenizer zip into its entry folder; the plan looks for the
    // checkpoint's model_config.yaml there (each checkpoint has its own).
    std::fs::write(installed[new.tokenizer].join("model_config.yaml"), "nemo_version: 2.8.0rc0\n").unwrap();
    let plan = |cfg: Value| with_env(&catalog, &installed, AMD_GPU, vec![], None, |env| magpie::plan(&config(cfg), env));

    let l = plan(json!({ "model": "v2607", "language": "ko", "voice": "leo" })).unwrap();
    assert_eq!(l.model, new.model);
    assert!(l.args.iter().any(|a| a.ends_with("magpie_tts_multilingual_357m.v2607.f16.gguf")), "{:?}", l.args);
    let tokenizer_arg = l.args.iter().skip_while(|a| *a != "--tokenizer-dir").nth(1).unwrap();
    assert_eq!(Path::new(tokenizer_arg), installed[new.tokenizer]);
    let warmup = l.warmup.as_ref().unwrap();
    assert_eq!((&warmup["language"], &warmup["voice"]), (&json!("ko-KR"), &json!("Leo")));
    // The warm-up sentence is written in the scripts an English one is not (Korean, Arabic).
    assert_eq!(warmup["input"], "안녕하세요.");
    let ar = plan(json!({ "model": "v2607", "language": "ar" })).unwrap();
    assert_eq!((&ar.warmup.as_ref().unwrap()["language"], &ar.warmup.as_ref().unwrap()["input"]), (&json!("ar-MSA"), &json!("مرحبا.")));
    let pt = plan(json!({ "model": "v2607", "language": "pt-BR" })).unwrap();
    assert_eq!(pt.warmup.as_ref().unwrap()["input"], "Hello there.");
    // The model's name in the profile may be the catalog id as well.
    assert_eq!(plan(json!({ "model": new.model })).unwrap().model, new.model);

    // v2602 keeps the default (a saved profile that names no model), its own files, and Korean falls back
    // to the English warm-up there.
    let d = plan(json!({ "language": "ko" })).unwrap();
    assert_eq!(d.model, old.model);
    assert!(d.args.iter().any(|a| a.ends_with("magpie_tts_multilingual_357m.v2602.f16.gguf")));
    assert_eq!(d.warmup.as_ref().unwrap()["language"], "en-US");
    // A model change restarts the server (TS reloads on it).
    assert!(!d.same_server(&l));

    // Without the v2607 files the plan names them (the requirement card lists the same ids).
    let missing = install(&root, &catalog, &[old.model, old.tokenizer, magpie::CODEC_ID, magpie::RUNTIME_VULKAN]);
    let err = with_env(&catalog, &missing, AMD_GPU, vec![], None, |env| magpie::plan(&config(json!({ "model": "v2607" })), env).unwrap_err());
    assert!(err.contains(new.model) && err.contains(new.tokenizer), "{err}");
}

#[test]
fn magpie_pack_version_follows_the_model_store_layout() {
    let at = |parts: &[&str]| parts.iter().fold(PathBuf::from("appdata"), |p, c| p.join(c));
    let v = |runtime: &str, parts: &[&str]| magpie::pack_version(runtime, &at(parts));
    assert_eq!(v(magpie::RUNTIME_CPU, &["runtimes", "nemo-speech-cpu", "0.1.0", "cpu"]), Some((0, 1, 0)));
    assert_eq!(v(magpie::RUNTIME_VULKAN, &["runtimes", "nemo-speech-vulkan", "0.2.0", "vulkan"]), Some((0, 2, 0)));
    // The version folder itself, and a folder inside the pack.
    assert_eq!(v(magpie::RUNTIME_CUDA, &["runtimes", "nemo-speech-cuda", "0.10.3"]), Some((0, 10, 3)));
    assert_eq!(v(magpie::RUNTIME_CUDA, &["runtimes", "nemo-speech-cuda", "0.2.0", "cuda", "bin"]), Some((0, 2, 0)));
    // Other spellings of a version.
    assert_eq!(v(magpie::RUNTIME_CPU, &["runtimes", "NeMo-Speech-CPU", "v1.2", "cpu"]), Some((1, 2, 0)));
    assert_eq!(v(magpie::RUNTIME_CPU, &["runtimes", "nemo-speech-cpu", "0.3.1-rc1", "cpu"]), Some((0, 3, 1)));
    // Not a model-store folder, another pack's folder, or a folder name that is no version: unknown.
    assert_eq!(v(magpie::RUNTIME_CPU, &["packs", "nemo"]), None);
    assert_eq!(v(magpie::RUNTIME_CPU, &["runtimes", "nemo-speech-vulkan", "0.1.0", "vulkan"]), None);
    assert_eq!(v(magpie::RUNTIME_CPU, &["runtimes", "nemo-speech-cpu", "latest", "cpu"]), None);
    assert_eq!(v(magpie::RUNTIME_CPU, &["runtimes", "nemo-speech-cpu"]), None);
}

#[test]
fn magpie_v2607_refuses_an_old_pack_and_v2602_runs_on_any() {
    let catalog = catalog();
    let root = temp_dir("magpie-old-pack");
    let (old, new) = (magpie::MODELS[0], magpie::MODELS[1]);
    let mut installed = install(
        &root,
        &catalog,
        &[old.model, old.tokenizer, new.model, new.tokenizer, magpie::CODEC_ID, magpie::RUNTIME_CPU],
    );
    std::fs::write(installed[new.tokenizer].join("model_config.yaml"), "nemo_version: 2.8.0rc0\n").unwrap();
    // A pack as the model store lays it out: <app data>/runtimes/nemo-speech-cpu/<version>/cpu.
    let pack = |version: &str| {
        let dir = root.join("runtimes").join("nemo-speech-cpu").join(version).join("cpu");
        std::fs::create_dir_all(dir.join("bin")).unwrap();
        std::fs::write(dir.join("bin").join("nemo-speech.exe"), "MZ fake").unwrap();
        dir
    };
    let plan = |installed: &HashMap<String, PathBuf>, model: &str| {
        with_env(&catalog, installed, NO_GPU, vec![], None, |env| magpie::plan(&config(json!({ "model": model })), env))
    };

    // The 0.1.0 pack a user installed before the catalog moved on: still installed, still fine for v2602.
    installed.insert(magpie::RUNTIME_CPU.into(), pack("0.1.0"));
    assert_eq!(plan(&installed, "v2602").unwrap().model, old.model);
    // v2607 on it would exit with "unsupported frame_stacking_factor=2": the plan says what to update.
    let err = plan(&installed, "v2607").unwrap_err();
    assert!(err.contains("v2607") && err.contains("0.2.0") && err.contains("0.1.0"), "{err}");
    assert!(err.contains(magpie::RUNTIME_CPU) && err.contains("Models & Runtimes"), "{err}");

    // The updated pack, and a pack folder outside the model store's layout (development), run both.
    installed.insert(magpie::RUNTIME_CPU.into(), pack("0.2.0"));
    assert_eq!(plan(&installed, "v2607").unwrap().model, new.model);
    assert_eq!(plan(&installed, "v2602").unwrap().model, old.model);
    let dev = root.join("dev-pack");
    std::fs::create_dir_all(dev.join("bin")).unwrap();
    std::fs::write(dev.join("bin").join("nemo-speech.exe"), "MZ fake").unwrap();
    installed.insert(magpie::RUNTIME_CPU.into(), dev);
    assert_eq!(plan(&installed, "v2607").unwrap().model, new.model);
}

#[test]
fn breeze_plan_runs_behind_the_adapter() {
    let catalog = catalog();
    let root = temp_dir("breeze");
    let app_data = temp_dir("breeze-appdata");
    let installed = install(&root, &catalog, &[breeze::MODELS[1].1, breeze::RUNTIME_VULKAN]);
    let l = with_env(&catalog, &installed, AMD_GPU, vec![], Some(app_data.clone()), |env| {
        breeze::plan(&config(json!({ "model": "q4_k" })), env).unwrap()
    });
    assert_eq!(l.variant, Variant::Vulkan);
    assert_eq!(l.adapter, Some(AdapterKind::Breeze));
    assert!(!l.server_key);
    let args = l.args.join(" ");
    assert!(args.contains("--host 127.0.0.1 --port {port} --ws-port -1 --voices-dir"), "{args}");
    assert!(!args.contains("--webui") && !args.contains("--cors") && !args.contains("--cpu"));
    assert!(app_data.join("tts-helpers").join("breeze").join("voices").is_dir());
    assert_eq!(l.vram_mb, 3200);
    // The processor runs the same pack with --cpu and claims no VRAM.
    let l = with_env(&catalog, &installed, NO_GPU, vec![], Some(app_data.clone()), |env| {
        breeze::plan(&config(json!({ "model": "q4_k" })), env).unwrap()
    });
    assert_eq!(l.variant, Variant::Cpu);
    assert!(l.args.iter().any(|a| a == "--cpu"));
    assert_eq!(l.vram_mb, 0);
    // The default model is not installed; CUDA has no Breeze build.
    with_env(&catalog, &installed, AMD_GPU, vec![], Some(app_data.clone()), |env| {
        assert!(breeze::plan(&GgmlConfig::default(), env).unwrap_err().contains(breeze::MODELS[0].1));
        assert!(breeze::plan(&config(json!({ "model": "q4_k", "variant": "cuda13" })), env).unwrap_err().contains("no cuda13 build"));
    });
}

#[test]
fn omnivoice_plan_copies_the_library_voice_and_uses_the_nvidia_runtime() {
    let catalog = catalog();
    let root = temp_dir("omni");
    let app_data = temp_dir("omni-appdata");
    let installed = install(
        &root,
        &catalog,
        &[omnivoice::MODELS[0].1, omnivoice::CODEC_ID, omnivoice::RUNTIME_CUDA13, launch::NVIDIA_RUNTIME_ID],
    );
    // CUDA 13 without the runtime folders is refused with a reason.
    let err = with_env(&catalog, &installed, NVIDIA_GPU, vec![], Some(app_data.clone()), |env| omnivoice::plan(&GgmlConfig::default(), env).unwrap_err());
    assert!(err.contains("NVIDIA GPU runtime"), "{err}");

    make_voice(&app_data, "alice-000001", Some("Hello there, this is Alice."));
    make_voice(&app_data, "bob-000002", None);
    let nv = vec![PathBuf::from(r"C:\nv")];
    let l = with_env(&catalog, &installed, NVIDIA_GPU, nv.clone(), Some(app_data.clone()), |env| {
        omnivoice::plan(&config(json!({ "referenceVoiceId": "alice-000001", "quality": "fast", "language": "ja" })), env).unwrap()
    });
    assert_eq!(l.variant, Variant::Cuda13);
    assert!(l.program.to_string_lossy().to_ascii_lowercase().ends_with("crispasr.exe"));
    let args = l.args.join(" ");
    assert!(args.starts_with("--server --backend omnivoice -m "), "{args}");
    assert!(args.contains("--host 127.0.0.1 --port {port} --voice-dir"), "{args}");
    assert!(args.ends_with("--tts-steps 16"), "{args}");
    assert!(!args.contains(" -l "), "the language is per request: {args}");
    assert_eq!(l.env, vec![(omnivoice::API_KEY_ENV.to_string(), "{api_key}".to_string())]);
    assert_eq!(l.dll_dirs, nv);
    assert_eq!(l.warmup.as_ref().unwrap()["language"], "ja");
    let voices = app_data.join("tts-helpers").join("omnivoice").join("voices");
    assert_eq!(std::fs::read_to_string(voices.join("alice-000001.txt")).unwrap(), "Hello there, this is Alice.");
    assert_eq!(
        std::fs::read(voices.join("alice-000001.wav")).unwrap(),
        std::fs::read(app_data.join("voices").join("alice-000001").join("ref.wav")).unwrap()
    );

    // Another voice or language is the same server; quality is not.
    let l2 = with_env(&catalog, &installed, NVIDIA_GPU, nv.clone(), Some(app_data.clone()), |env| {
        omnivoice::plan(&config(json!({ "quality": "fast", "language": "en" })), env).unwrap()
    });
    assert!(l2.same_server(&l));
    let l3 = with_env(&catalog, &installed, NVIDIA_GPU, nv.clone(), Some(app_data.clone()), |env| omnivoice::plan(&GgmlConfig::default(), env).unwrap());
    assert!(!l3.same_server(&l));

    // A voice without a transcript, or an id that is not a file name, is refused.
    with_env(&catalog, &installed, NVIDIA_GPU, nv.clone(), Some(app_data.clone()), |env| {
        assert!(omnivoice::plan(&config(json!({ "referenceVoiceId": "bob-000002" })), env).unwrap_err().contains("no transcript"));
        assert!(omnivoice::plan(&config(json!({ "referenceVoiceId": "../x" })), env).is_err());
        assert!(omnivoice::plan(&config(json!({ "quality": "ultra" })), env).is_err());
    });
    assert_eq!(omnivoice::voice_file_stem("a_b-1"), Some("a_b-1"));
    assert_eq!(omnivoice::voice_file_stem("a.b"), None);
}

#[test]
fn s2_plan_uses_the_pack_manifest() {
    let catalog = catalog();
    let root = temp_dir("s2");
    let installed = install(&root, &catalog, &[s2::MODELS[0].1, s2::TOKENIZER_ID, s2::RUNTIME_CUDA13, launch::NVIDIA_RUNTIME_ID]);
    let nv = vec![PathBuf::from(r"C:\nv")];
    let l = with_env(&catalog, &installed, NVIDIA_GPU, nv.clone(), None, |env| s2::plan(&GgmlConfig::default(), env).unwrap());
    assert_eq!(l.variant, Variant::Cuda13);
    assert_eq!(l.readiness, Readiness::AnyHttp);
    assert_eq!(l.adapter, Some(AdapterKind::S2));
    let args = l.args.join(" ");
    assert!(args.contains("--server --host 127.0.0.1 --port {port} --cuda 0"), "{args}");
    assert!(args.contains("tokenizer.json"), "{args}");
    assert!(!args.contains("--gpu-layers"));
    assert_eq!(l.vram_mb, 7_400);
    let low = with_env(&catalog, &installed, NVIDIA_GPU, nv.clone(), None, |env| s2::plan(&config(json!({ "lowVram": true })), env).unwrap());
    assert!(low.args.join(" ").ends_with("--cuda 0 --gpu-layers 18 --codec-cpu"));
    assert!(!low.same_server(&l));
    assert!(low.vram_mb < l.vram_mb);
    // A damaged pack is refused before anything starts.
    let pack_dir = &installed[s2::RUNTIME_CUDA13];
    std::fs::write(pack_dir.join("s2.exe"), b"MZ tampered").unwrap();
    let err = with_env(&catalog, &installed, NVIDIA_GPU, nv, None, |env| s2::plan(&GgmlConfig::default(), env).unwrap_err());
    assert!(err.contains("does not match"), "{err}");
}

#[test]
fn engine_facts_match_the_scaffold() {
    for (engine, info) in [
        (TtsHelperEngine::Magpie, magpie::info()),
        (TtsHelperEngine::Breeze, breeze::info()),
        (TtsHelperEngine::Omnivoice, omnivoice::info()),
        (TtsHelperEngine::S2, s2::info()),
    ] {
        assert_eq!(info.engine, engine);
        assert!(info.experimental && info.gpu);
        let catalog = catalog();
        for id in &info.requires {
            assert!(catalog.get(id).is_some(), "{id}");
        }
    }
    assert!(!magpie::info().cloning && breeze::info().cloning && omnivoice::info().cloning && s2::info().cloning);
}

// ---------------------------------------------------------------------------------------------
// Adapter: pure pieces
// ---------------------------------------------------------------------------------------------

#[test]
fn adapter_requests_are_checked() {
    use warp::http::header::{HeaderMap, HeaderValue, AUTHORIZATION, HOST, ORIGIN};
    let key = "a".repeat(64);
    let ok = || {
        let mut h = HeaderMap::new();
        h.insert(HOST, HeaderValue::from_static("127.0.0.1:5123"));
        h.insert(AUTHORIZATION, HeaderValue::from_str(&format!("Bearer {key}")).unwrap());
        h
    };
    assert!(adapter::check_request(&ok(), &key).is_none());
    let mut h = ok();
    h.insert(HOST, HeaderValue::from_static("localhost:5123"));
    assert!(adapter::check_request(&h, &key).is_none());
    h.insert(HOST, HeaderValue::from_static("[::1]:5123"));
    assert!(adapter::check_request(&h, &key).is_none());
    h.insert(HOST, HeaderValue::from_static("evil.example:5123"));
    assert_eq!(adapter::check_request(&h, &key).unwrap().0, 403);
    let mut h = ok();
    h.insert(ORIGIN, HeaderValue::from_static("http://127.0.0.1:5123"));
    assert_eq!(adapter::check_request(&h, &key).unwrap().0, 403);
    let mut h = ok();
    h.insert(AUTHORIZATION, HeaderValue::from_static("Bearer nope"));
    assert_eq!(adapter::check_request(&h, &key).unwrap().0, 401);
    h.remove(AUTHORIZATION);
    assert_eq!(adapter::check_request(&h, &key).unwrap().0, 401);
    assert!(adapter::same_secret(b"abc", b"abc") && !adapter::same_secret(b"abc", b"abd") && !adapter::same_secret(b"ab", b"abc"));
}

fn body_of(v: Value) -> Map<String, Value> {
    v.as_object().unwrap().clone()
}

#[test]
fn breeze_requests_become_its_native_form() {
    let reference = adapter::Reference {
        wav: b"RIFF....WAVEfmt fake".to_vec(),
        transcript: "Hello there.".into(),
    };
    let body = body_of(json!({
        "input": "Hi", "instructions": "A calm voice.", "voice": "saved_1", "cfg_scale": 2.5, "seed": 42.4,
        "temperature": "0.7", "top_k": 50, "speed": 1.5, "stream": true, "max_new_tokens": f64::NAN,
    }));
    let (path, ct, bytes) = adapter::build_upstream(AdapterKind::Breeze, &body, "Hi", None);
    assert_eq!(path, "/v1/audio/speech");
    let f = multipart_fields(&bytes, &ct);
    assert_eq!(text_field(&f, "text").as_deref(), Some("Hi"));
    assert_eq!(text_field(&f, "instruction").as_deref(), Some("A calm voice."));
    assert_eq!(text_field(&f, "voice_id").as_deref(), Some("saved_1"));
    assert_eq!(text_field(&f, "cfg_scale").as_deref(), Some("2.5"));
    assert_eq!(text_field(&f, "seed").as_deref(), Some("42"));
    assert_eq!(text_field(&f, "temperature").as_deref(), Some("0.7"));
    assert_eq!(text_field(&f, "top_k").as_deref(), Some("50"));
    assert!(field(&f, "speed").is_none() && field(&f, "max_new_tokens").is_none() && field(&f, "ref_audio").is_none());

    // A cloned voice: the clip and its transcript, never a voice id or a path.
    let body = body_of(json!({ "input": "Hi", "voice": "../../x" }));
    let (_, ct, bytes) = adapter::build_upstream(AdapterKind::Breeze, &body, "Hi", Some(&reference));
    let f = multipart_fields(&bytes, &ct);
    let audio = field(&f, "ref_audio").unwrap();
    assert_eq!(audio.1.as_deref(), Some("reference.wav"));
    assert_eq!(audio.2, reference.wav);
    assert_eq!(text_field(&f, "ref_text").as_deref(), Some("Hello there."));
    assert!(field(&f, "voice_id").is_none());
    let (_, ct, bytes) = adapter::build_upstream(AdapterKind::Breeze, &body, "Hi", None);
    assert!(field(&multipart_fields(&bytes, &ct), "voice_id").is_none(), "paths are never voice ids");
}

#[test]
fn s2_requests_become_generate_with_params() {
    let reference = adapter::Reference {
        wav: vec![1, 2, 3, 4],
        transcript: "Ref text.".into(),
    };
    let body = body_of(json!({
        "input": "Hello [laugh]", "voice": "C:/secret.s2voice", "voice_dir": "C:/", "temperature": 0.6, "top_k": 25.2,
        "max_new_tokens": 96, "segment_sentences": true,
    }));
    let (path, ct, bytes) = adapter::build_upstream(AdapterKind::S2, &body, "Hello [laugh]", Some(&reference));
    assert_eq!(path, "/generate");
    let f = multipart_fields(&bytes, &ct);
    assert_eq!(text_field(&f, "text").as_deref(), Some("Hello [laugh]"));
    let params: Value = serde_json::from_str(&text_field(&f, "params").unwrap()).unwrap();
    assert_eq!(params["output_format"], "pcm_s16le");
    assert_eq!(params["chunked"], false);
    assert_eq!(params["temperature"], 0.6);
    assert_eq!(params["top_k"], 25);
    assert_eq!(params["max_new_tokens"], 96);
    assert_eq!(params["segment_sentences"], true);
    assert_eq!(field(&f, "reference").unwrap().2, vec![1, 2, 3, 4]);
    assert_eq!(text_field(&f, "reference_text").as_deref(), Some("Ref text."));
    for forbidden in ["voice", "voice_dir", "voice_id", "voice_profile"] {
        assert!(field(&f, forbidden).is_none(), "{forbidden} is never forwarded");
    }
    let (_, ct, bytes) = adapter::build_upstream(AdapterKind::S2, &body_of(json!({ "stream": true })), "x", None);
    let params: Value = serde_json::from_str(&text_field(&multipart_fields(&bytes, &ct), "params").unwrap()).unwrap();
    assert_eq!(params["chunked"], true);
}

#[test]
fn answers_and_errors_are_mapped() {
    use reqwest::header::{HeaderMap, HeaderValue, CONTENT_TYPE};
    let mut h = HeaderMap::new();
    assert_eq!(adapter::answer_rate(&h, AdapterKind::Breeze), 24_000);
    assert_eq!(adapter::answer_rate(&h, AdapterKind::S2), 44_100);
    h.insert("x-audio-sample-rate", HeaderValue::from_static("22050"));
    assert_eq!(adapter::answer_rate(&h, AdapterKind::S2), 22_050);
    h.insert("x-sample-rate", HeaderValue::from_static("48000"));
    assert_eq!(adapter::answer_rate(&h, AdapterKind::S2), 48_000);
    let mut h = HeaderMap::new();
    h.insert(CONTENT_TYPE, HeaderValue::from_static("audio/pcm; rate=16000"));
    assert_eq!(adapter::answer_rate(&h, AdapterKind::Breeze), 16_000);
    h.insert("x-sample-rate", HeaderValue::from_static("7"));
    assert_eq!(adapter::answer_rate(&h, AdapterKind::Breeze), 24_000, "out-of-range rates are ignored");
    assert_eq!(rate_of_content_type("audio/pcm;rate=\"24000\""), Some(24_000));
    assert_eq!(rate_of_content_type("audio/pcm"), None);
    assert_eq!(adapter::map_upstream_status(409), 503);
    assert_eq!(adapter::map_upstream_status(503), 503);
    assert_eq!(adapter::map_upstream_status(404), 404);
    assert_eq!(adapter::map_upstream_status(500), 502);
    assert_eq!(adapter::upstream_message(br#"{"error":{"message":"bad voice"}}"#), "bad voice");
    assert_eq!(adapter::upstream_message(br#"{"detail":"nope"}"#), "nope");
    assert_eq!(adapter::upstream_message(b"Server busy processing another synthesis request"), "Server busy processing another synthesis request");
}

// ---------------------------------------------------------------------------------------------
// Adapter: in-process fake upstream
// ---------------------------------------------------------------------------------------------

/// One request a fake upstream received: (start, end of its answer, fields).
type Received = (Instant, Option<Instant>, Vec<Field>);

#[derive(Default)]
struct Upstreams {
    log: Mutex<Vec<Received>>,
    active: AtomicUsize,
    max_active: AtomicUsize,
}

/// A fake Breeze / s2.cpp server: answers after `header_delay` (a whole-clip server answers only
/// once the sentence is done), then streams `chunks` pieces of PCM, `delay` apart; `409` when the
/// text is "busy". A request counts as active from its arrival to the end of its answer.
async fn fake_upstream(
    kind: AdapterKind,
    header_delay: Duration,
    chunks: usize,
    delay: Duration,
) -> (String, Arc<Upstreams>, tokio::sync::oneshot::Sender<()>) {
    use warp::Filter;
    let state = Arc::new(Upstreams::default());
    let s = state.clone();
    let path = if kind == AdapterKind::Breeze { "v1" } else { "generate" };
    let route = warp::post()
        .and(warp::path::full())
        .and(warp::header::<String>("content-type"))
        .and(warp::body::bytes())
        .and_then(move |full: warp::path::FullPath, ct: String, body: warp::hyper::body::Bytes| {
            let s = s.clone();
            async move {
                assert!(full.as_str().contains(path), "{}", full.as_str());
                let fields = multipart_fields(&body, &ct);
                let busy = text_field(&fields, "text").as_deref() == Some("busy");
                let index = {
                    let mut log = s.log.lock().unwrap();
                    log.push((Instant::now(), None, fields));
                    log.len() - 1
                };
                if busy {
                    s.log.lock().unwrap()[index].1 = Some(Instant::now());
                    let mut r = warp::http::Response::new(warp::hyper::Body::from("Another generation is running"));
                    *r.status_mut() = warp::http::StatusCode::CONFLICT;
                    return Ok::<_, std::convert::Infallible>(r);
                }
                let now = s.active.fetch_add(1, Ordering::SeqCst) + 1;
                s.max_active.fetch_max(now, Ordering::SeqCst);
                tokio::time::sleep(header_delay).await;
                let st = s.clone();
                let stream = futures::stream::unfold(0usize, move |i| {
                    let st = st.clone();
                    async move {
                        if i == chunks {
                            st.active.fetch_sub(1, Ordering::SeqCst);
                            st.log.lock().unwrap()[index].1 = Some(Instant::now());
                            return None;
                        }
                        if i > 0 {
                            tokio::time::sleep(delay).await;
                        }
                        Some((Ok::<_, std::io::Error>(vec![0u8; 960]), i + 1))
                    }
                });
                let mut r = warp::http::Response::new(warp::hyper::Body::wrap_stream(stream));
                let headers = r.headers_mut();
                if kind == AdapterKind::Breeze {
                    headers.insert("content-type", "audio/pcm".parse().unwrap());
                    headers.insert("x-sample-rate", "24000".parse().unwrap());
                } else {
                    headers.insert("content-type", "audio/L16".parse().unwrap());
                    headers.insert("x-audio-sample-rate", "44100".parse().unwrap());
                }
                Ok(r)
            }
        });
    let (tx, rx) = tokio::sync::oneshot::channel::<()>();
    let (addr, server) = warp::serve(route).bind_with_graceful_shutdown(([127, 0, 0, 1], 0), async {
        let _ = rx.await;
    });
    tokio::spawn(server);
    (format!("http://127.0.0.1:{}", addr.port()), state, tx)
}

fn client() -> reqwest::Client {
    reqwest::Client::builder().no_proxy().build().unwrap()
}

#[tokio::test(flavor = "multi_thread")]
async fn adapter_forwards_streams_and_clones_from_the_library() {
    let (base, up, _stop) = fake_upstream(AdapterKind::Breeze, Duration::ZERO, 4, Duration::from_millis(20)).await;
    let app_data = temp_dir("adapter-appdata");
    make_voice(&app_data, "alice-000001", Some("Hello there, this is Alice."));
    make_voice(&app_data, "bob-000002", None);
    let a = adapter::start_with(AdapterKind::Breeze, Upstream::Fixed(base), Some(app_data.clone()), adapter::RECONNECT_FOR)
        .await
        .unwrap();
    let url = format!("{}/audio/speech", a.base_url());

    // Key, Origin and body checks.
    let r = client().post(&url).json(&json!({ "input": "Hi" })).send().await.unwrap();
    assert_eq!(r.status(), 401);
    let r = client()
        .post(&url)
        .bearer_auth(a.key())
        .header("origin", "http://evil.example")
        .json(&json!({ "input": "Hi" }))
        .send()
        .await
        .unwrap();
    assert_eq!(r.status(), 403);
    let r = client().post(&url).bearer_auth(a.key()).json(&json!({ "input": "  " })).send().await.unwrap();
    assert_eq!(r.status(), 400);
    let r = client().get(format!("http://127.0.0.1:{}/health", a.port())).send().await.unwrap();
    assert_eq!(r.status(), 200);

    // A cloned sentence streams back as raw PCM with the server's rate.
    let r = client()
        .post(&url)
        .bearer_auth(a.key())
        .json(&json!({ "input": "Hi there", "instructions": "Cheerful.", "seed": 42, "sigil_reference_voice": "alice-000001" }))
        .send()
        .await
        .unwrap();
    assert_eq!(r.status(), 200);
    assert_eq!(r.headers()["content-type"], "audio/pcm; rate=24000");
    assert_eq!(r.bytes().await.unwrap().len(), 4 * 960);
    {
        let log = up.log.lock().unwrap();
        let fields = &log.last().unwrap().2;
        assert_eq!(text_field(fields, "text").as_deref(), Some("Hi there"));
        assert_eq!(text_field(fields, "ref_text").as_deref(), Some("Hello there, this is Alice."));
        let wav = std::fs::read(app_data.join("voices").join("alice-000001").join("ref.wav")).unwrap();
        assert_eq!(field(fields, "ref_audio").unwrap().2, wav);
    }

    // A library voice without a transcript, an unknown voice, a busy server.
    for (body, status) in [
        (json!({ "input": "Hi", "sigil_reference_voice": "bob-000002" }), 400),
        (json!({ "input": "Hi", "sigil_reference_voice": "nobody-000009" }), 400),
        (json!({ "input": "busy" }), 503),
    ] {
        let r = client().post(&url).bearer_auth(a.key()).json(&body).send().await.unwrap();
        assert_eq!(r.status(), status, "{body}");
        let v: Value = r.json().await.unwrap();
        assert!(v["error"]["message"].as_str().is_some_and(|m| !m.is_empty()), "{v}");
    }
}

/// Every request the fake upstream received started after the answer before it ended.
fn assert_one_after_another(up: &Upstreams) {
    let log = up.log.lock().unwrap();
    for pair in log.windows(2) {
        let (prev_end, next_start) = (pair[0].1.expect("finished"), pair[1].0);
        assert!(next_start >= prev_end, "a request started before the previous answer ended");
    }
}

#[tokio::test(flavor = "multi_thread")]
async fn an_abandoned_answer_is_drained_before_the_next_request() {
    let (base, up, _stop) = fake_upstream(AdapterKind::S2, Duration::ZERO, 10, Duration::from_millis(60)).await;
    let a = adapter::start_with(AdapterKind::S2, Upstream::Fixed(base), None, adapter::RECONNECT_FOR).await.unwrap();
    let url = format!("{}/audio/speech", a.base_url());
    // The first request is abandoned after its first chunk (a skip).
    let mut first = client().post(&url).bearer_auth(a.key()).json(&json!({ "input": "one" })).send().await.unwrap();
    assert_eq!(first.headers()["content-type"], "audio/pcm; rate=44100");
    assert!(first.chunk().await.unwrap().is_some());
    drop(first);
    // Two more arrive at once: each waits for the one before (and the drain).
    let c = client();
    let (r2, r3) = tokio::join!(
        c.post(&url).bearer_auth(a.key()).json(&json!({ "input": "two" })).send(),
        async {
            tokio::time::sleep(Duration::from_millis(5)).await;
            c.post(&url).bearer_auth(a.key()).json(&json!({ "input": "three" })).send().await
        }
    );
    assert_eq!(r2.unwrap().bytes().await.unwrap().len(), 10 * 960);
    assert_eq!(r3.unwrap().bytes().await.unwrap().len(), 10 * 960);
    assert_eq!(up.max_active.load(Ordering::SeqCst), 1, "one request at a time");
    assert_one_after_another(&up);
    let log = up.log.lock().unwrap();
    assert_eq!(log.len(), 3);
    assert_eq!(text_field(&log[0].2, "text").as_deref(), Some("one"));
}

/// A whole-clip server (s2.cpp with `chunked: false`) sends nothing until the sentence is done. A
/// client that leaves before that (a skip) must neither free the one-request lock nor cancel the
/// engine's work: the next sentence waits until the abandoned answer was read to its end, instead
/// of reaching a server that is still busy (and answers 409 / 503).
#[tokio::test(flavor = "multi_thread")]
async fn a_request_abandoned_before_its_answer_begins_keeps_the_engine_to_itself() {
    let (base, up, _stop) = fake_upstream(AdapterKind::S2, Duration::from_millis(600), 3, Duration::from_millis(20)).await;
    let a = adapter::start_with(AdapterKind::S2, Upstream::Fixed(base), None, adapter::RECONNECT_FOR).await.unwrap();
    let url = format!("{}/audio/speech", a.base_url());
    // The first sentence is skipped while the engine still works on it (no answer headers yet).
    let skipped = client()
        .post(&url)
        .bearer_auth(a.key())
        .json(&json!({ "input": "one" }))
        .timeout(Duration::from_millis(150))
        .send()
        .await;
    assert!(skipped.is_err_and(|e| e.is_timeout()), "the client gave up before the answer began");
    assert_eq!(up.active.load(Ordering::SeqCst), 1, "the engine still works on the skipped sentence");
    // The next sentence comes at once; it reaches the engine only after the skipped answer ended.
    let r = client().post(&url).bearer_auth(a.key()).json(&json!({ "input": "two" })).send().await.unwrap();
    assert_eq!(r.status(), 200);
    assert_eq!(r.bytes().await.unwrap().len(), 3 * 960);
    assert_eq!(up.max_active.load(Ordering::SeqCst), 1, "one request at a time");
    assert_one_after_another(&up);
    let log = up.log.lock().unwrap();
    assert_eq!(log.len(), 2);
    assert_eq!(text_field(&log[1].2, "text").as_deref(), Some("two"));
}

/// A refused connection is described as "cannot connect" (the TS backend's cue to find the helper
/// again and send the sentence once more); other failures are not.
#[tokio::test(flavor = "multi_thread")]
async fn connection_failures_say_cannot_connect() {
    // A port nothing listens on (bound, then released).
    let port = TcpListener::bind(("127.0.0.1", 0)).unwrap().local_addr().unwrap().port();
    let e = runner::loopback_client().post(format!("http://127.0.0.1:{port}/generate")).send().await.unwrap_err();
    assert!(e.is_connect(), "{e}");
    let m = adapter::describe_send_error(&e, Duration::from_secs(90));
    assert!(m.contains("cannot connect") && m.contains("90 s"), "{m}");
}

// ---------------------------------------------------------------------------------------------
// Lifecycle with a fake server (child process)
// ---------------------------------------------------------------------------------------------

const ROLE_ENV: &str = "SIGIL_GGML_FAKE";
const ENTRY: &str = "services::tts_helpers::ggml::tests::fake_ggml_server_entry";
const FAKE_KEY_ENV: &str = "SIGIL_GGML_FAKE_KEY";

/// The lifecycle tests share the process host: run them one at a time.
async fn serial() -> tokio::sync::MutexGuard<'static, ()> {
    static SERIAL: std::sync::OnceLock<tokio::sync::Mutex<()>> = std::sync::OnceLock::new();
    SERIAL.get_or_init(Default::default).lock().await
}

/// A launch that runs this test binary as a fake server of `mode` (`openai` with a key and a health
/// route, `s2` without either) under `helper_name`.
fn fake_launch(helper_name: &str, mode: &str, extra_arg: &str) -> Launch {
    let s2 = mode == "s2";
    let mut env = vec![
        (ROLE_ENV.to_string(), "1".to_string()),
        ("SIGIL_GGML_FAKE_PORT".to_string(), launch::port_arg()),
        ("SIGIL_GGML_FAKE_MODE".to_string(), mode.to_string()),
        ("SIGIL_GGML_FAKE_EXTRA".to_string(), extra_arg.to_string()),
    ];
    if !s2 {
        env.push((FAKE_KEY_ENV.to_string(), launch::key_env()));
    }
    Launch {
        engine: if s2 { TtsHelperEngine::S2 } else { TtsHelperEngine::Magpie },
        helper_name: helper_name.to_string(),
        program: std::env::current_exe().unwrap(),
        args: ["--exact", ENTRY, "--ignored", "--nocapture", "--test-threads=1", "-q"].iter().map(|s| s.to_string()).collect(),
        env,
        env_remove: Vec::new(),
        server_key: !s2,
        readiness: if s2 { Readiness::AnyHttp } else { Readiness::Health("/health".into()) },
        adapter: s2.then_some(AdapterKind::S2),
        model: format!("test/{mode}-{extra_arg}"),
        variant: Variant::Vulkan,
        pcm_rate: if s2 { 44_100 } else { 22_050 },
        vram_mb: 321,
        load_timeout_ms: 20_000,
        dll_dirs: Vec::new(),
        warmup: Some(json!({ "input": "warm up", "response_format": "pcm" })),
        summary: "fake".into(),
    }
}

fn helper_log(name: &str, needle: &str) -> Vec<String> {
    process_host::logs(name)
        .into_iter()
        .map(|l| l.text)
        .filter(|t| t.contains(needle))
        .collect()
}

fn claim_mb(label: &str) -> Option<u64> {
    crate::services::resources::snapshot()
        .claims
        .into_iter()
        .find(|c| c.label == label)
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

static SLOT_OPENAI: Slot = Slot::new(TtsHelperEngine::Magpie, "fake openai");
static SLOT_S2: Slot = Slot::new(TtsHelperEngine::S2, "fake s2");

#[tokio::test(flavor = "multi_thread")]
async fn start_reuse_replace_restart_and_stop() {
    let _serial = serial().await;
    let l = fake_launch("tts-test-openai", "openai", "a");
    let endpoint = SLOT_OPENAI.start_launch(l.clone(), SLOT_OPENAI.stop_epoch()).await.expect("the fake starts");
    assert!(endpoint.base_url.starts_with("http://127.0.0.1:") && endpoint.base_url.ends_with("/v1"));
    assert_eq!(endpoint.api_key.len(), 64);
    assert_eq!(endpoint.pcm_sample_rate, 22_050);
    assert_eq!(SLOT_OPENAI.status().state, TtsHelperState::Ready);
    assert_eq!(claim_mb("magpie: test/openai-a"), Some(321));
    // The warm-up ran with the key; a request without it is refused by the server.
    assert_eq!(helper_log(&l.helper_name, "fake-ggml speech: warm up").len(), 1);
    let t = runner::timed_speech(&endpoint, &json!({ "input": "Hello" }), Duration::from_secs(10)).await.unwrap();
    assert!(t.audio_ms > 0.0);
    let anonymous = client().post(format!("{}/audio/speech", endpoint.base_url)).json(&json!({ "input": "x" })).send().await.unwrap();
    assert_eq!(anonymous.status(), 401);

    // The same launch reuses the running server (no new warm-up).
    let again = SLOT_OPENAI.start_launch(l.clone(), SLOT_OPENAI.stop_epoch()).await.unwrap();
    assert_eq!(again, endpoint);
    assert_eq!(helper_log(&l.helper_name, "fake-ggml speech: warm up").len(), 1);

    // A crash: the process host restarts it, and a start finds it again (same key).
    let _ = client().post(format!("{}/crash", endpoint.base_url.trim_end_matches("/v1"))).send().await;
    assert!(wait_until(Duration::from_secs(20), || helper_log(&l.helper_name, "fake-ggml listening").len() >= 2).await, "restarted");
    let after = SLOT_OPENAI.start_launch(l.clone(), SLOT_OPENAI.stop_epoch()).await.unwrap();
    assert_eq!(after.api_key, endpoint.api_key);
    assert!(runner::timed_speech(&after, &json!({ "input": "Back" }), Duration::from_secs(10)).await.is_ok());

    // Another launch replaces it (the claim follows).
    let other = fake_launch("tts-test-openai", "openai", "b");
    let e2 = SLOT_OPENAI.start_launch(other.clone(), SLOT_OPENAI.stop_epoch()).await.unwrap();
    assert_eq!(e2.model, "test/openai-b");
    assert!(wait_until(Duration::from_secs(5), || claim_mb("magpie: test/openai-a").is_none()).await);
    assert_eq!(claim_mb("magpie: test/openai-b"), Some(321));

    // Stop: the process ends, the claim goes, the status says stopped.
    SLOT_OPENAI.stop("tts-test-openai").await.unwrap();
    assert_eq!(SLOT_OPENAI.status().state, TtsHelperState::Stopped);
    assert!(claim_mb("magpie: test/openai-b").is_none());
    assert!(process_host::get("tts-test-openai").is_none_or(|i| i.state != process_host::HelperState::Running));
}

#[tokio::test(flavor = "multi_thread")]
async fn a_server_without_health_runs_behind_the_adapter() {
    let _serial = serial().await;
    let l = fake_launch("tts-test-s2", "s2", "a");
    let endpoint = SLOT_S2.start_launch(l.clone(), SLOT_S2.stop_epoch()).await.expect("the fake starts");
    // The endpoint is the adapter's, not the helper's.
    let helper_port = process_host::handle("tts-test-s2").unwrap().port.unwrap();
    assert!(!endpoint.base_url.contains(&format!(":{helper_port}/")));
    assert_eq!(endpoint.api_key.len(), 64);
    assert_eq!(helper_log(&l.helper_name, "fake-ggml generate: text=warm up").len(), 1);
    let t = runner::timed_speech(&endpoint, &json!({ "input": "Hello", "temperature": 0.5 }), Duration::from_secs(10)).await.unwrap();
    // 8820 bytes at the server's 44.1 kHz.
    assert!((t.audio_ms - 100.0).abs() < 1.0, "{t:?}");
    assert!(helper_log(&l.helper_name, "\"temperature\":0.5").len() == 1);

    // A start cancelled by a stop never leaves a server behind.
    SLOT_S2.stop("tts-test-s2").await.unwrap();
    let stopper = async {
        tokio::time::sleep(Duration::from_millis(30)).await;
        SLOT_S2.stop("tts-test-s2").await
    };
    let starter = SLOT_S2.start(|| Ok(fake_launch("tts-test-s2", "s2", "slow")));
    let (started, stopped) = tokio::join!(starter, stopper);
    stopped.unwrap();
    if let Ok(e) = started {
        panic!("a start overtaken by a stop returned {e:?}");
    }
    assert!(
        wait_until(Duration::from_secs(10), || process_host::get("tts-test-s2").is_none_or(|i| i.state != process_host::HelperState::Running)).await,
        "no server left running"
    );
    assert_eq!(SLOT_S2.status().state, TtsHelperState::Stopped);
}

static SLOT_S2_RESTART: Slot = Slot::new(TtsHelperEngine::S2, "fake s2 restart");

/// s2.cpp has no health route and listens only once its model is loaded, while the process host
/// calls a restarted helper running at once. A start that finds it restarted (the TS backend after
/// a crash) must return only once it answers again, not while the model still loads.
#[tokio::test(flavor = "multi_thread")]
async fn a_restarted_server_without_health_is_found_again_once_it_answers() {
    let _serial = serial().await;
    let name = "tts-test-s2-restart";
    let l = fake_launch(name, "s2", "slow");
    let endpoint = SLOT_S2_RESTART.start_launch(l.clone(), SLOT_S2_RESTART.stop_epoch()).await.expect("the fake starts");
    let base = process_host::handle(name).unwrap().base_url.unwrap();
    let _ = client().post(format!("{base}/crash")).send().await;
    assert!(
        wait_until(Duration::from_secs(20), || process_host::get(name).is_some_and(|i| i.restarts >= 1)).await,
        "the process host restarts it"
    );
    let again = SLOT_S2_RESTART.start_launch(l.clone(), SLOT_S2_RESTART.stop_epoch()).await.expect("found again");
    assert_eq!(again, endpoint, "the same adapter");
    // It answers now (the fake loads for 1.5 s before it listens).
    let base = process_host::handle(name).unwrap().base_url.unwrap();
    let probe = client().get(format!("{base}/")).timeout(Duration::from_secs(1)).send().await;
    assert!(probe.is_ok(), "start returned before the restarted server answered: {probe:?}");
    assert!(wait_until(Duration::from_secs(5), || helper_log(name, "fake-ggml listening").len() == 2).await);
    let t = runner::timed_speech(&again, &json!({ "input": "Back again" }), Duration::from_secs(10)).await.unwrap();
    assert!(t.audio_ms > 0.0);
    SLOT_S2_RESTART.stop(name).await.unwrap();
}

// ---------------------------------------------------------------------------------------------
// Manual speak test with a real pack and real models (P6-F verification)
// ---------------------------------------------------------------------------------------------

/// Every file under `dir` by lower-case file name (first hit wins).
fn index_files(dir: &Path, out: &mut HashMap<String, PathBuf>, depth: usize) {
    let Ok(rd) = std::fs::read_dir(dir) else { return };
    for e in rd.flatten() {
        let p = e.path();
        if p.is_dir() && depth > 0 {
            index_files(&p, out, depth - 1);
        } else if p.is_file() {
            out.entry(e.file_name().to_string_lossy().to_ascii_lowercase()).or_insert(p);
        }
    }
}

/// Unpack, next to itself, the zip of every `tts/` archive entry found in the models folder (the
/// model store extracts such an entry into its folder with the members at the root, as the Magpie
/// v2607 tokenizer zip is built); members that already exist are kept. Returns the files written.
fn unpack_staged_zips(catalog: &Catalog, files: &HashMap<String, PathBuf>) -> usize {
    let mut written = 0;
    for e in catalog.entries.iter().filter(|e| e.id.starts_with("tts/")) {
        for f in e.files.iter().filter(|f| f.archive.as_ref().is_some_and(|a| a.format == ArchiveFormat::Zip)) {
            let name = f.path.rsplit('/').next().unwrap_or("").to_ascii_lowercase();
            let Some(zip_path) = files.get(&name) else { continue };
            let dir = zip_path.parent().unwrap();
            let file = std::fs::File::open(zip_path).unwrap();
            let mut zip = zip::ZipArchive::new(std::io::BufReader::new(file)).unwrap_or_else(|e| panic!("{} is not a zip: {e}", zip_path.display()));
            for i in 0..zip.len() {
                let mut member = zip.by_index(i).unwrap();
                if member.is_dir() {
                    continue;
                }
                let Some(leaf) = member.enclosed_name().and_then(|p| p.file_name().map(|n| n.to_owned())) else { continue };
                let target = dir.join(leaf);
                if target.exists() {
                    continue;
                }
                let mut out = std::fs::File::create(&target).unwrap();
                std::io::copy(&mut member, &mut out).unwrap();
                written += 1;
            }
        }
    }
    written
}

/// Speak through the endpoint and keep the audio: (first audio ms, total ms, PCM bytes, rate).
async fn speak_to_bytes(endpoint: &crate::services::tts_helpers::TtsHelperEndpoint, body: &Value) -> Result<(f64, f64, Vec<u8>, u32), String> {
    let started = Instant::now();
    let mut rb = client().post(format!("{}/audio/speech", endpoint.base_url)).json(body).timeout(Duration::from_secs(600));
    if !endpoint.api_key.is_empty() {
        rb = rb.bearer_auth(&endpoint.api_key);
    }
    let mut resp = rb.send().await.map_err(|e| e.to_string())?;
    if !resp.status().is_success() {
        let status = resp.status();
        return Err(format!("{status}: {}", resp.text().await.unwrap_or_default()));
    }
    let rate = resp
        .headers()
        .get("content-type")
        .and_then(|v| v.to_str().ok())
        .and_then(rate_of_content_type)
        .unwrap_or(endpoint.pcm_sample_rate);
    let mut first = None;
    let mut pcm = Vec::new();
    while let Some(chunk) = resp.chunk().await.map_err(|e| e.to_string())? {
        if first.is_none() && !chunk.is_empty() {
            first = Some(started.elapsed());
        }
        pcm.extend_from_slice(&chunk);
    }
    let total = started.elapsed();
    Ok((first.unwrap_or(total).as_secs_f64() * 1000.0, total.as_secs_f64() * 1000.0, pcm, rate))
}

/// Run one engine for real and speak a few sentences through Sigil's own path (plan, process host,
/// adapter, warm-up), logging time to first audio and the real-time factor and writing WAVs:
///
/// ```powershell
/// $env:SIGIL_GGML_ENGINE = "magpie"            # magpie | breeze | omnivoice | s2
/// $env:SIGIL_GGML_PACK_DIR = "<the extracted runtime pack zip>"
/// $env:SIGIL_TEST_MODELS_DIR = "<a folder with the engine's model files>"
/// # optional: SIGIL_GGML_VARIANT (cuda13|cuda|vulkan|cpu), SIGIL_GGML_MODEL (q4_k, v2607, ...),
/// #           SIGIL_GGML_VOICE (a voice-library id to clone), SIGIL_GGML_TEXT (one sentence per line),
/// #           SIGIL_GGML_LANGUAGE (Magpie: en, ko, ar, pt-BR, ...; put sentences of that language in SIGIL_GGML_TEXT),
/// #           SIGIL_GPU_LOCK (hold the shared GPU timing lock), SIGIL_GGML_OUT (WAV folder)
/// cargo test --manifest-path src-tauri/Cargo.toml real_ggml_engine_speaks -- --ignored --nocapture
/// ```
///
/// Magpie, one folder per checkpoint (the two tokenizers share the name `model_config.yaml`):
/// v2602 = `magpie_tts_multilingual_357m.v2602.f16.gguf`, the codec GGUF and the 7 tokenizer files;
/// v2607 = `magpie_tts_multilingual_357m.v2607.f16.gguf`, the codec GGUF and
/// `magpie-tts-357m-v2607-tokenizer.zip` (this test unpacks a zip of an archive entry next to
/// itself, as the model store does when it installs one). Set `SIGIL_GGML_MODEL` to `v2602` or `v2607`.
#[tokio::test(flavor = "multi_thread")]
#[ignore = "needs a real runtime pack and models (SIGIL_GGML_ENGINE, SIGIL_GGML_PACK_DIR, SIGIL_TEST_MODELS_DIR)"]
async fn real_ggml_engine_speaks() {
    let engine = std::env::var("SIGIL_GGML_ENGINE").expect("SIGIL_GGML_ENGINE");
    let pack_dir = PathBuf::from(std::env::var("SIGIL_GGML_PACK_DIR").expect("SIGIL_GGML_PACK_DIR"));
    let models_dir = PathBuf::from(std::env::var("SIGIL_TEST_MODELS_DIR").expect("SIGIL_TEST_MODELS_DIR"));
    let voice = std::env::var("SIGIL_GGML_VOICE").unwrap_or_default();
    let catalog = catalog();
    let mut files = HashMap::new();
    index_files(&models_dir, &mut files, 4);
    if unpack_staged_zips(&catalog, &files) > 0 {
        files.clear();
        index_files(&models_dir, &mut files, 4);
    }
    let (runtimes, slot): (Vec<&str>, &'static Slot) = match engine.as_str() {
        "magpie" => (vec![magpie::RUNTIME_CUDA, magpie::RUNTIME_VULKAN, magpie::RUNTIME_CPU], &magpie::SLOT),
        "breeze" => (vec![breeze::RUNTIME_VULKAN], &breeze::SLOT),
        "omnivoice" => (vec![omnivoice::RUNTIME_CUDA13, omnivoice::RUNTIME_VULKAN, omnivoice::RUNTIME_CPU], &omnivoice::SLOT),
        "s2" => (vec![s2::RUNTIME_CUDA13, s2::RUNTIME_VULKAN, s2::RUNTIME_CPU], &s2::SLOT),
        other => panic!("unknown engine {other}"),
    };
    // Installed: the pack for every variant of the engine, and each catalog entry whose files are
    // all in the models folder (by file name; an entry's files must share one folder).
    let mut installed: HashMap<String, PathBuf> = runtimes.iter().map(|id| (id.to_string(), pack_dir.clone())).collect();
    for e in &catalog.entries {
        if e.id.starts_with("runtime/") {
            continue;
        }
        let hits: Vec<&PathBuf> = e
            .files
            .iter()
            .filter_map(|f| files.get(&f.path.rsplit('/').next().unwrap_or("").to_ascii_lowercase()))
            .collect();
        if hits.len() == e.files.len() && !hits.is_empty() {
            installed.insert(e.id.clone(), hits[0].parent().unwrap().to_path_buf());
        }
    }
    if let Some(p) = crate::services::model_store::installed_path(launch::NVIDIA_RUNTIME_ID) {
        installed.insert(launch::NVIDIA_RUNTIME_ID.into(), p);
    }
    let gpu = GpuFacts::of(&crate::services::sysinfo::cached());
    // Magpie speaks the language of SIGIL_GGML_LANGUAGE (a BCP-47 tag: ko, ar, pt-BR for v2607); the others
    // keep English.
    let language = std::env::var("SIGIL_GGML_LANGUAGE").ok().filter(|l| !l.trim().is_empty()).unwrap_or_else(|| "en".into());
    let cfg = config(json!({
        "model": std::env::var("SIGIL_GGML_MODEL").unwrap_or_default(),
        "variant": std::env::var("SIGIL_GGML_VARIANT").unwrap_or_default(),
        "language": language,
        "referenceVoiceId": if engine == "omnivoice" { voice.clone() } else { String::new() },
    }));
    let app_data = crate::services::diagnostics::app_data_dir().unwrap_or_else(|| temp_dir("real-appdata"));
    let launch = with_env(&catalog, &installed, gpu, launch::nvidia_dll_dirs(), Some(app_data), |env| match engine.as_str() {
        "magpie" => magpie::plan(&cfg, env),
        "breeze" => breeze::plan(&cfg, env),
        "omnivoice" => omnivoice::plan(&cfg, env),
        _ => s2::plan(&cfg, env),
    })
    .expect("plan");
    println!("plan: {} ({})", launch.summary, launch.program.display());

    let lock = std::env::var("SIGIL_GPU_LOCK").ok().filter(|_| launch.variant.on_gpu()).map(PathBuf::from);
    if let Some(lock) = &lock {
        while lock.exists() {
            println!("waiting for the GPU timing lock {}", lock.display());
            tokio::time::sleep(Duration::from_secs(15)).await;
        }
        std::fs::write(lock, format!("P6-F real_ggml_engine_speaks {engine} since {:?}", std::time::SystemTime::now())).unwrap();
    }
    let out_dir = std::env::var("SIGIL_GGML_OUT").map(PathBuf::from).unwrap_or_else(|_| temp_dir(&format!("real-{engine}")));
    std::fs::create_dir_all(&out_dir).unwrap();
    let texts: Vec<String> = std::env::var("SIGIL_GGML_TEXT")
        .map(|t| t.lines().map(str::to_string).filter(|l| !l.trim().is_empty()).collect())
        .unwrap_or_else(|_| {
            vec![
                "Hello! This is Sigil speaking through an experimental voice engine.".into(),
                "The quick brown fox jumps over the lazy dog, and then takes a well deserved nap in the warm afternoon sun.".into(),
            ]
        });
    let result = async {
        let started = Instant::now();
        let endpoint = slot.start_launch(launch.clone(), slot.stop_epoch()).await?;
        println!("ready (with warm-up) in {} ms: {}", started.elapsed().as_millis(), endpoint.base_url);
        for (i, text) in texts.iter().enumerate() {
            let mut body = json!({ "input": text, "response_format": "pcm" });
            match engine.as_str() {
                "magpie" => {
                    body["voice"] = json!("Sofia");
                    body["language"] = json!(magpie::language_code(&language).unwrap_or("en-US"));
                }
                "breeze" => {
                    body["instructions"] = json!("A warm young woman, friendly and upbeat, speaking at a relaxed pace.");
                    body["seed"] = json!(42);
                    body["stream"] = json!(true);
                }
                "omnivoice" => {
                    body["language"] = json!("en");
                    body["seed"] = json!(7);
                    body["stream"] = json!(true);
                }
                _ => {}
            }
            if !voice.is_empty() {
                if engine == "omnivoice" {
                    body["voice"] = json!(voice);
                    body["consent_attestation"] = json!("P6-F manual test with the developer's own voice");
                } else if engine != "magpie" {
                    body["sigil_reference_voice"] = json!(voice);
                }
            }
            let (first_ms, total_ms, pcm, rate) = speak_to_bytes(&endpoint, &body).await?;
            let audio_ms = (pcm.len() / 2) as f64 * 1000.0 / f64::from(rate.max(1));
            println!(
                "sentence {}: first audio {first_ms:.0} ms, {audio_ms:.0} ms of audio at {rate} Hz in {total_ms:.0} ms (RTF {:.3})",
                i + 1,
                if audio_ms > 0.0 { total_ms / audio_ms } else { f64::NAN }
            );
            let spec = hound::WavSpec {
                channels: 1,
                sample_rate: rate,
                bits_per_sample: 16,
                sample_format: hound::SampleFormat::Int,
            };
            let path = out_dir.join(format!("{engine}-{}.wav", i + 1));
            let mut w = hound::WavWriter::create(&path, spec).map_err(|e| e.to_string())?;
            for pair in pcm.as_chunks::<2>().0 {
                w.write_sample(i16::from_le_bytes(*pair)).map_err(|e| e.to_string())?;
            }
            w.finalize().map_err(|e| e.to_string())?;
            println!("  -> {}", path.display());
            if audio_ms <= 0.0 {
                return Err(format!("sentence {} produced no audio", i + 1));
            }
        }
        Ok::<(), String>(())
    }
    .await;
    let _ = slot.stop(&launch.helper_name).await;
    if let Some(lock) = &lock {
        let _ = std::fs::remove_file(lock);
    }
    for l in process_host::logs(&launch.helper_name).iter().rev().take(20).rev() {
        println!("helper: {}", l.text);
    }
    result.expect("the engine spoke every sentence");
}

// ---------------------------------------------------------------------------------------------
// The fake server (runs in a child process)
// ---------------------------------------------------------------------------------------------

#[test]
#[ignore = "child-process entry point of the ggml helper tests; does nothing without SIGIL_GGML_FAKE"]
fn fake_ggml_server_entry() {
    if std::env::var(ROLE_ENV).is_err() {
        return;
    }
    let port: u16 = std::env::var("SIGIL_GGML_FAKE_PORT").unwrap().parse().unwrap();
    let mode = std::env::var("SIGIL_GGML_FAKE_MODE").unwrap_or_default();
    let extra = std::env::var("SIGIL_GGML_FAKE_EXTRA").unwrap_or_default();
    let key = std::env::var(FAKE_KEY_ENV).unwrap_or_default();
    if extra == "slow" {
        // A model that takes a while to load (s2.cpp listens once loaded).
        std::thread::sleep(Duration::from_millis(1500));
    }
    let listener = TcpListener::bind(("127.0.0.1", port)).expect("bind");
    eprintln!("fake-ggml listening ({mode})");
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
        let header = |name: &str| {
            head.lines()
                .find_map(|l| l.split_once(':').filter(|(k, _)| k.trim().eq_ignore_ascii_case(name)).map(|(_, v)| v.trim().to_string()))
        };
        let len: usize = header("content-length").and_then(|v| v.parse().ok()).unwrap_or(0);
        while buf.len() < head_end + len {
            match sock.read(&mut tmp) {
                Ok(0) | Err(_) => break,
                Ok(n) => buf.extend_from_slice(&tmp[..n]),
            }
        }
        let body = buf[head_end..(head_end + len).min(buf.len())].to_vec();
        let mut words = head.split_whitespace();
        let method = words.next().unwrap_or("").to_string();
        let path = words.next().unwrap_or("/").to_string();
        let authed = key.is_empty() || header("authorization").as_deref() == Some(format!("Bearer {key}").as_str());
        let json = |status: &str, v: Value| (status.to_string(), vec![("content-type".to_string(), "application/json".to_string())], serde_json::to_vec(&v).unwrap());
        let (status, headers, out) = match (method.as_str(), path.as_str(), mode.as_str()) {
            ("GET", "/health", "openai") => json("200 OK", json!({ "status": "ok" })),
            (_, "/crash", _) => {
                eprintln!("fake-ggml crashing");
                std::process::exit(3);
            }
            (_, _, "openai") if !authed => json("401 Unauthorized", json!({ "error": { "message": "missing or wrong API key" } })),
            ("POST", "/v1/audio/speech", "openai") => {
                let req: Value = serde_json::from_slice(&body).unwrap_or_default();
                eprintln!("fake-ggml speech: {}", req["input"].as_str().unwrap_or(""));
                ("200 OK".into(), vec![("content-type".into(), "audio/pcm".into())], vec![0u8; 4410])
            }
            ("POST", "/generate", "s2") => {
                let ct = header("content-type").unwrap_or_default();
                let fields = multipart_fields(&body, &ct);
                eprintln!(
                    "fake-ggml generate: text={} params={}",
                    text_field(&fields, "text").unwrap_or_default(),
                    text_field(&fields, "params").unwrap_or_default()
                );
                (
                    "200 OK".into(),
                    vec![("content-type".into(), "audio/L16".into()), ("x-audio-sample-rate".into(), "44100".into())],
                    vec![0u8; 8820],
                )
            }
            _ => json("404 Not Found", json!({ "error": "not found" })),
        };
        let mut response = format!("HTTP/1.1 {status}\r\nContent-Length: {}\r\nConnection: close\r\n", out.len());
        for (k, v) in headers {
            response.push_str(&format!("{k}: {v}\r\n"));
        }
        response.push_str("\r\n");
        let _ = sock.write_all(response.as_bytes());
        let _ = sock.write_all(&out);
        let _ = sock.flush();
    }
}
