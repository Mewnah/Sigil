//! Tests of the native runtime: ABI mirror, variants, DLL loading failures, the loader against an
//! in-process fake engine (function table implemented in Rust), pack manifests and hashes, catalog
//! discovery and planning, the NVIDIA / provider packs, the IPC rows and the probe helpers.
//!
//! `real_echo_engine_end_to_end` exercises a real `sigil_echo.dll` built by
//! `pwsh scripts/native/build-runtime-packs.ps1 -Engine echo` (ignored unless `SIGIL_TEST_ECHO_DIR`
//! names `native/out/dev/echo/<variant>`):
//! `SIGIL_TEST_ECHO_DIR=native/out/dev/echo/cpu cargo test -- --ignored real_echo_engine_end_to_end`.

use std::path::{Path, PathBuf};

use super::abi::*;
use super::loader::{EngineLibrary, Vtable};
use super::pack::{self, PackFile, PackManifest};
use super::variant::{best_variant, candidates, vulkan_from_adapters};
use super::*;
use crate::services::model_store::Catalog;

fn temp_dir(tag: &str) -> PathBuf {
    let d = std::env::temp_dir().join(format!("sigil-native-runtime-{tag}-{}", std::process::id()));
    let _ = std::fs::remove_dir_all(&d);
    std::fs::create_dir_all(&d).unwrap();
    d
}

fn sha256_hex(bytes: &[u8]) -> String {
    use sha2::{Digest, Sha256};
    hex::encode(Sha256::digest(bytes))
}

// ---------------------------------------------------------------------------------------------
// ABI mirror
// ---------------------------------------------------------------------------------------------

#[test]
fn abi_layout_matches_header() {
    use std::mem::{align_of, size_of};
    // x86_64: 2 x u32 + 5 pointers + u64.
    assert_eq!(size_of::<SigilEngineInfo>(), 56);
    assert_eq!(SIGIL_ENGINE_INFO_V1_SIZE as usize, size_of::<SigilEngineInfo>());
    assert_eq!(align_of::<SigilEngineInfo>(), 8);
    assert_eq!(size_of::<SigilAudioView>(), 24);
    assert_eq!(size_of::<SigilAudioBuffer>(), 24);
    assert_eq!(SIGIL_ENGINE_ABI_VERSION, 1);
    assert_eq!(SYMBOLS.len(), 16);
    assert_eq!(status_name(SIGIL_E_CANCELLED), "SIGIL_E_CANCELLED");
    assert_eq!(status_name(-99), "SIGIL_E_UNKNOWN");
    assert_eq!(Vtable::NAMES, SYMBOLS, "the loader resolves exactly the header's exports, in order");
}

#[test]
fn header_and_mirror_agree() {
    // The C header is the source of truth; check the constants the mirror copies.
    let header = include_str!("../../../../native/common/sigil_engine.h");
    assert!(header.contains("#define SIGIL_ENGINE_ABI_VERSION 1u"));
    for (name, value) in [
        ("SIGIL_E_INVALID_ARGUMENT", SIGIL_E_INVALID_ARGUMENT),
        ("SIGIL_E_UNSUPPORTED", SIGIL_E_UNSUPPORTED),
        ("SIGIL_E_LOAD_FAILED", SIGIL_E_LOAD_FAILED),
        ("SIGIL_E_RUNTIME", SIGIL_E_RUNTIME),
        ("SIGIL_E_OUT_OF_MEMORY", SIGIL_E_OUT_OF_MEMORY),
        ("SIGIL_E_CANCELLED", SIGIL_E_CANCELLED),
        ("SIGIL_E_BACKEND_UNAVAILABLE", SIGIL_E_BACKEND_UNAVAILABLE),
    ] {
        assert!(header.contains(&format!("#define {name} ({value})")), "{name}");
    }
    for (name, bit) in [("PROCESS", 0), ("STREAM", 1), ("RENDER", 2), ("CANCEL", 3), ("GPU", 4)] {
        assert!(header.contains(&format!("#define SIGIL_CAP_{name} (1ull << {bit})")), "{name}");
    }
    for s in SYMBOLS {
        assert!(header.contains(&format!("SIGIL_ENGINE_CALL {s}(")), "{s} is declared in the header");
    }
}

// ---------------------------------------------------------------------------------------------
// Variants
// ---------------------------------------------------------------------------------------------

#[test]
fn variants_round_trip_and_selection_is_deterministic() {
    for v in RuntimeVariant::PREFERENCE {
        assert_eq!(RuntimeVariant::parse(v.as_str()), Some(v));
        assert_eq!(serde_json::to_value(v).unwrap(), serde_json::json!(v.as_str()));
        assert_eq!(v.to_string(), v.as_str());
    }
    assert_eq!(RuntimeVariant::parse("avx2"), None);
    let all = RuntimeVariant::PREFERENCE;
    let cpu_only = VariantSupport {
        avx2: true,
        ..VariantSupport::default()
    };
    assert_eq!(best_variant(&all, &cpu_only), Some(RuntimeVariant::CpuAvx2));
    let gpu = VariantSupport {
        cuda13: true,
        vulkan: true,
        avx512: true,
        avx2: true,
    };
    assert_eq!(best_variant(&all, &gpu), Some(RuntimeVariant::Cuda13));
    assert_eq!(
        best_variant(&[RuntimeVariant::Vulkan, RuntimeVariant::Cpu], &cpu_only),
        Some(RuntimeVariant::Cpu)
    );
    assert_eq!(best_variant(&[], &gpu), None);
}

#[test]
fn gpu_variants_need_avx2_and_candidates_fall_through_in_order() {
    // A GPU without AVX2: the GPU variants carry cpu-avx2 kernels, so only plain cpu runs.
    let old_cpu = VariantSupport {
        cuda13: true,
        vulkan: true,
        avx512: false,
        avx2: false,
    };
    assert!(!old_cpu.supports(RuntimeVariant::Cuda13) && !old_cpu.supports(RuntimeVariant::Vulkan));
    assert!(old_cpu.unsupported_reason(RuntimeVariant::Vulkan).unwrap().contains("AVX2"));
    assert!(old_cpu.supports(RuntimeVariant::Cpu));
    let pc = VariantSupport {
        cuda13: false,
        vulkan: true,
        avx512: false,
        avx2: true,
    };
    let installed = [RuntimeVariant::Cpu, RuntimeVariant::Cuda13, RuntimeVariant::CpuAvx2, RuntimeVariant::Vulkan];
    assert_eq!(
        candidates(&installed, &pc, None),
        vec![RuntimeVariant::Vulkan, RuntimeVariant::CpuAvx2, RuntimeVariant::Cpu],
        "preference order, unsupported cuda13 skipped"
    );
    assert!(pc.unsupported_reason(RuntimeVariant::Cuda13).unwrap().contains("R580"));
    // A forced variant is tried without its GPU (create reports the real backend error)...
    assert_eq!(candidates(&installed, &pc, Some(RuntimeVariant::Cuda13)), vec![RuntimeVariant::Cuda13]);
    // ...but only when it is installed...
    assert!(candidates(&installed, &pc, Some(RuntimeVariant::CpuAvx512)).is_empty());
    // ...and never without its instruction sets: such a DLL does not fail, it crashes the process.
    let with_avx512 = [RuntimeVariant::CpuAvx512, RuntimeVariant::CpuAvx2, RuntimeVariant::Cpu];
    assert!(candidates(&with_avx512, &pc, Some(RuntimeVariant::CpuAvx512)).is_empty(), "AVX2-only CPU");
    assert!(candidates(&installed, &old_cpu, Some(RuntimeVariant::Cuda13)).is_empty(), "GPU builds carry AVX2 code");
    assert!(candidates(&installed, &old_cpu, Some(RuntimeVariant::Vulkan)).is_empty());
    assert!(candidates(&installed, &old_cpu, Some(RuntimeVariant::CpuAvx2)).is_empty());
    assert_eq!(candidates(&installed, &old_cpu, Some(RuntimeVariant::Cpu)), vec![RuntimeVariant::Cpu]);
    let avx512_pc = VariantSupport {
        avx512: true,
        ..pc
    };
    assert_eq!(
        candidates(&with_avx512, &avx512_pc, Some(RuntimeVariant::CpuAvx512)),
        vec![RuntimeVariant::CpuAvx512]
    );
    let cpu = VariantSupport::detect_cpu();
    assert!(!cpu.cuda13 && !cpu.vulkan, "CPU detection never claims a GPU");
    assert!(!cpu.avx512 || cpu.avx2, "AVX-512 implies AVX2");
}

#[test]
fn instruction_sets_are_a_hard_requirement() {
    use RuntimeVariant::*;
    let old_cpu = VariantSupport {
        cuda13: true,
        vulkan: true,
        avx512: false,
        avx2: false,
    };
    let avx2_only = VariantSupport {
        avx2: true,
        ..VariantSupport::default()
    };
    // Instruction sets only: a missing GPU does not matter here.
    for v in [Cuda13, Vulkan, CpuAvx2] {
        assert!(avx2_only.cpu_supports(v) && !old_cpu.cpu_supports(v), "{v}");
        assert!(old_cpu.cpu_unsupported_reason(v).unwrap().contains("AVX2"), "{v}");
    }
    assert!(!avx2_only.cpu_supports(CpuAvx512));
    assert_eq!(avx2_only.cpu_unsupported_reason(CpuAvx512).as_deref(), Some("this CPU does not support AVX-512"));
    assert!(old_cpu.cpu_supports(Cpu) && old_cpu.cpu_unsupported_reason(Cpu).is_none());
    assert!(old_cpu.cpu_unsupported_reason(Cuda13).unwrap().contains("CUDA 13 build"));

    // By variant string (a manifest or the DLL's own info); unknown labels are not judged.
    let err = avx2_only.check_build("cpu-avx512").unwrap_err();
    assert!(err.contains("cpu-avx512 build") && err.contains("AVX-512") && err.contains("cpu-avx2 variant"), "{err}");
    assert!(old_cpu.check_build("cuda13").unwrap_err().contains("install the cpu variant"));
    assert!(avx2_only.check_build("cpu-avx2").is_ok() && avx2_only.check_build("vulkan").is_ok());
    assert!(old_cpu.check_build("cpu").is_ok() && old_cpu.check_build("dev-build").is_ok());

    // The NVIDIA runtime / CUDA provider / third-party runtimes: GPU variants need only the GPU.
    assert_eq!(old_cpu.runtime_unsupported_reason(Cuda13), None, "ONNX Runtime's CUDA EP runs without AVX2");
    assert_eq!(old_cpu.runtime_unsupported_reason(Vulkan), None);
    assert!(old_cpu.unsupported_reason(Cuda13).unwrap().contains("AVX2"), "engine packs still need AVX2");
    assert!(avx2_only.runtime_unsupported_reason(Cuda13).unwrap().contains("R580"));
    assert!(avx2_only.runtime_unsupported_reason(Vulkan).unwrap().contains("Vulkan"));
    assert!(old_cpu.runtime_unsupported_reason(CpuAvx2).unwrap().contains("AVX2"));
    assert_eq!(old_cpu.runtime_unsupported_reason(Cpu), None);
}

#[test]
fn vulkan_needs_a_hardware_gpu_and_the_loader() {
    use crate::services::diagnostics::gpu_mem::{GpuAdapter, VENDOR_MICROSOFT, VENDOR_NVIDIA};
    let adapter = |vendor_id: u32, software: bool| GpuAdapter {
        index: 0,
        name: "gpu".into(),
        vendor: "x".into(),
        vendor_id,
        device_id: 1,
        luid: "0:0".into(),
        dedicated_vram_mb: 8192,
        shared_memory_mb: 0,
        software,
        driver_version: None,
        nvidia_driver: None,
        vram: None,
    };
    assert!(vulkan_from_adapters(&[adapter(VENDOR_NVIDIA, false)], true));
    assert!(!vulkan_from_adapters(&[adapter(VENDOR_NVIDIA, false)], false), "no vulkan-1.dll");
    assert!(!vulkan_from_adapters(&[adapter(VENDOR_MICROSOFT, true)], true), "WARP is not a Vulkan GPU");
    assert!(!vulkan_from_adapters(&[], true));
}

// ---------------------------------------------------------------------------------------------
// DLL loading failures (never panics)
// ---------------------------------------------------------------------------------------------

#[test]
fn missing_dll_is_an_error_not_a_panic() {
    let dir = std::env::temp_dir().join("sigil-native-runtime-missing");
    assert!(load_library_in_dir(&dir, "sigil_nope.dll").is_err());
    assert!(EngineLibrary::load(&dir, "sigil_nope.dll").unwrap_err().contains("does not exist"));
    for bad in ["", "..\\x.dll", "a/b.dll", "sub\\x.dll"] {
        assert!(load_library_in_dir(&dir, bad).unwrap_err().contains("invalid DLL name"), "{bad:?}");
    }
}

#[test]
fn a_file_that_is_not_a_dll_is_refused() {
    let dir = temp_dir("garbage");
    std::fs::write(dir.join("sigil_garbage.dll"), b"this is not a PE file").unwrap();
    let err = load_library_in_dir(&dir, "sigil_garbage.dll").unwrap_err();
    assert!(err.contains("sigil_garbage.dll"), "{err}");
    #[cfg(windows)]
    assert!(err.contains("not a 64-bit Windows DLL"), "{err}");
    assert!(EngineLibrary::load(&dir, "sigil_garbage.dll").is_err());
    let _ = std::fs::remove_dir_all(&dir);
}

/// A real DLL that is not a Sigil engine (a renamed copy of a small system DLL) is refused before
/// anything but `sigil_engine_abi_version` is looked up.
#[cfg(windows)]
#[test]
fn a_dll_without_the_abi_exports_is_refused() {
    let system = std::env::var_os("SystemRoot").map(PathBuf::from).unwrap_or_else(|| PathBuf::from(r"C:\Windows"));
    let src = system.join("System32").join("version.dll");
    if !src.is_file() {
        eprintln!("{} is missing; skipped", src.display());
        return;
    }
    let dir = temp_dir("noexports");
    std::fs::copy(&src, dir.join("sigil_noexports.dll")).unwrap();
    let err = EngineLibrary::load(&dir, "sigil_noexports.dll").unwrap_err();
    assert!(err.contains("not a Sigil engine") && err.contains("sigil_engine_abi_version"), "{err}");
    // The plain loader loads it fine (it is a valid DLL).
    assert!(load_library_in_dir(&dir, "sigil_noexports.dll").is_ok());
    let _ = std::fs::remove_dir_all(&dir);
}

// ---------------------------------------------------------------------------------------------
// Fake engine: the ABI implemented in Rust, loaded through `EngineLibrary::from_parts`
// ---------------------------------------------------------------------------------------------

mod fake {
    use super::super::abi::*;
    use super::super::loader::Vtable;
    use std::cell::{Cell, RefCell};
    use std::ffi::{c_char, c_void, CStr, CString};
    use std::sync::atomic::{AtomicBool, Ordering};
    use std::sync::Mutex;
    use std::time::{Duration, Instant};

    thread_local! {
        static LAST_ERROR: RefCell<CString> = RefCell::new(CString::default());
        /// Strings and audio buffers handed out on this thread and not freed yet.
        pub static LIVE: Cell<i64> = const { Cell::new(0) };
    }

    static EVENTS: Mutex<Vec<(u64, &'static str)>> = Mutex::new(Vec::new());
    static LOG: Mutex<Option<(usize, usize)>> = Mutex::new(None);

    pub fn events(id: u64) -> Vec<&'static str> {
        EVENTS.lock().unwrap().iter().filter(|(i, _)| *i == id).map(|(_, e)| *e).collect()
    }

    pub fn live() -> i64 {
        LIVE.with(Cell::get)
    }

    fn event(id: u64, what: &'static str) {
        EVENTS.lock().unwrap().push((id, what));
    }

    fn set_error(msg: &str) {
        LAST_ERROR.with(|e| *e.borrow_mut() = CString::new(msg).unwrap());
    }

    unsafe fn json(p: *const c_char) -> serde_json::Value {
        if p.is_null() {
            return serde_json::json!({});
        }
        serde_json::from_str(unsafe { CStr::from_ptr(p) }.to_str().unwrap()).unwrap_or(serde_json::Value::Null)
    }

    unsafe fn out_string(out: *mut *mut c_char, s: String) {
        unsafe { *out = CString::new(s).unwrap().into_raw() };
        LIVE.with(|l| l.set(l.get() + 1));
    }

    pub struct Engine {
        id: u64,
        gain: f32,
        cancel: AtomicBool,
    }

    pub struct Stream {
        engine_id: u64,
        frames: u64,
    }

    struct SyncInfo(SigilEngineInfo);
    // SAFETY: the pointers are to static C string literals.
    unsafe impl Sync for SyncInfo {}

    const fn info(struct_size: u32, abi_version: u32, name: &'static CStr) -> SyncInfo {
        SyncInfo(SigilEngineInfo {
            struct_size,
            abi_version,
            name: name.as_ptr(),
            version: c"1.2.3".as_ptr(),
            variant: c"cpu".as_ptr(),
            kind: c"test".as_ptr(),
            build_info: c"rust fake".as_ptr(),
            capabilities: SIGIL_CAP_PROCESS | SIGIL_CAP_STREAM | SIGIL_CAP_RENDER | SIGIL_CAP_CANCEL,
        })
    }
    static INFO: SyncInfo = info(56, 1, c"fake");
    static INFO_SMALL: SyncInfo = info(40, 1, c"fake");
    static INFO_ABI2: SyncInfo = info(56, 2, c"fake");
    static INFO_BAD_NAME: SyncInfo = info(56, 1, c"Bad Name");

    pub unsafe extern "C" fn abi_v1() -> u32 {
        1
    }
    pub unsafe extern "C" fn abi_v2() -> u32 {
        2
    }
    pub unsafe extern "C" fn get_info() -> *const SigilEngineInfo {
        &INFO.0
    }
    pub unsafe extern "C" fn get_info_null() -> *const SigilEngineInfo {
        std::ptr::null()
    }
    pub unsafe extern "C" fn get_info_small() -> *const SigilEngineInfo {
        &INFO_SMALL.0
    }
    pub unsafe extern "C" fn get_info_abi2() -> *const SigilEngineInfo {
        &INFO_ABI2.0
    }
    pub unsafe extern "C" fn get_info_bad_name() -> *const SigilEngineInfo {
        &INFO_BAD_NAME.0
    }
    unsafe extern "C" fn last_error() -> *const c_char {
        LAST_ERROR.with(|e| e.borrow().as_ptr())
    }
    unsafe extern "C" fn set_log(f: SigilLogFn, user: *mut c_void) {
        *LOG.lock().unwrap() = f.map(|f| (f as usize, user as usize));
    }
    unsafe extern "C" fn create(cfg: *const c_char, out: *mut *mut SigilEngine) -> SigilStatus {
        set_error("");
        let cfg = unsafe { json(cfg) };
        match cfg["fail"].as_str() {
            Some("load") => {
                set_error("model file missing");
                return SIGIL_E_LOAD_FAILED;
            }
            Some("backend") => {
                set_error("no GPU backend");
                return SIGIL_E_BACKEND_UNAVAILABLE;
            }
            _ => {}
        }
        // Exercise the host's log bridge (must never crash, NULL message included).
        if let Some((f, user)) = *LOG.lock().unwrap() {
            // SAFETY: stored from a valid `sigil_log_fn`.
            let f: unsafe extern "C" fn(*mut c_void, i32, *const c_char) = unsafe { std::mem::transmute(f) };
            unsafe {
                f(user as *mut c_void, SIGIL_LOG_DEBUG, c"fake: created".as_ptr());
                f(user as *mut c_void, 42, std::ptr::null());
            }
        }
        let id = cfg["id"].as_u64().unwrap_or(0);
        event(id, "create");
        let e = Box::new(Engine {
            id,
            gain: cfg["gain"].as_f64().unwrap_or(1.0) as f32,
            cancel: AtomicBool::new(false),
        });
        unsafe { *out = Box::into_raw(e).cast() };
        SIGIL_OK
    }
    unsafe extern "C" fn destroy(e: *mut SigilEngine) {
        let e = unsafe { Box::from_raw(e.cast::<Engine>()) };
        event(e.id, "destroy");
    }
    unsafe extern "C" fn cancel(e: *mut SigilEngine) {
        unsafe { &*e.cast::<Engine>() }.cancel.store(true, Ordering::Release);
    }
    unsafe extern "C" fn query(e: *mut SigilEngine, req: *const c_char, out: *mut *mut c_char) -> SigilStatus {
        set_error("");
        let e = unsafe { &*e.cast::<Engine>() };
        e.cancel.store(false, Ordering::Release);
        let req = unsafe { json(req) };
        match req["op"].as_str().unwrap_or("info") {
            "info" => unsafe { out_string(out, r#"{"name":"fake","backend":"cpu"}"#.into()) },
            "sleep" => {
                let until = Instant::now() + Duration::from_millis(req["ms"].as_u64().unwrap_or(0));
                while Instant::now() < until {
                    if e.cancel.load(Ordering::Acquire) {
                        set_error("cancelled by the host");
                        return SIGIL_E_CANCELLED;
                    }
                    std::thread::sleep(Duration::from_millis(1));
                }
                unsafe { out_string(out, "{}".into()) };
            }
            "fail" => {
                set_error("query failed on purpose");
                return SIGIL_E_RUNTIME;
            }
            "null" => {}
            _ => {
                set_error("unknown op");
                return SIGIL_E_INVALID_ARGUMENT;
            }
        }
        SIGIL_OK
    }
    unsafe extern "C" fn process(
        _e: *mut SigilEngine,
        view: *const SigilAudioView,
        _opts: *const c_char,
        out: *mut *mut c_char,
    ) -> SigilStatus {
        let v = unsafe { &*view };
        unsafe { out_string(out, format!(r#"{{"frames":{},"channels":{}}}"#, v.frames, v.channels)) };
        SIGIL_OK
    }
    unsafe extern "C" fn render(
        e: *mut SigilEngine,
        view: *const SigilAudioView,
        opts: *const c_char,
        out: *mut SigilAudioBuffer,
    ) -> SigilStatus {
        let e = unsafe { &*e.cast::<Engine>() };
        let opts = unsafe { json(opts) };
        let out = unsafe { &mut *out };
        if opts["bad_buffer"].as_bool() == Some(true) {
            *out = SigilAudioBuffer {
                samples: std::ptr::null_mut(),
                frames: 10,
                sample_rate: 16_000,
                channels: 1,
            };
            return SIGIL_OK;
        }
        if view.is_null() {
            *out = SigilAudioBuffer {
                samples: std::ptr::null_mut(),
                frames: 0,
                sample_rate: 16_000,
                channels: 1,
            };
            return SIGIL_OK;
        }
        let v = unsafe { &*view };
        let n = (v.frames * u64::from(v.channels)) as usize;
        let input = if n == 0 { &[][..] } else { unsafe { std::slice::from_raw_parts(v.samples, n) } };
        let data: Box<[f32]> = input.iter().map(|s| s * e.gain).collect();
        *out = SigilAudioBuffer {
            samples: Box::into_raw(data).cast(),
            frames: v.frames,
            sample_rate: v.sample_rate,
            channels: v.channels,
        };
        LIVE.with(|l| l.set(l.get() + 1));
        SIGIL_OK
    }
    unsafe extern "C" fn stream_open(e: *mut SigilEngine, _opts: *const c_char, out: *mut *mut SigilStream) -> SigilStatus {
        let e = unsafe { &*e.cast::<Engine>() };
        event(e.id, "stream_open");
        let s = Box::new(Stream {
            engine_id: e.id,
            frames: 0,
        });
        unsafe { *out = Box::into_raw(s).cast() };
        SIGIL_OK
    }
    unsafe extern "C" fn stream_push(s: *mut SigilStream, view: *const SigilAudioView, out: *mut *mut c_char) -> SigilStatus {
        let s = unsafe { &mut *s.cast::<Stream>() };
        let v = unsafe { &*view };
        if v.frames == 0 {
            return SIGIL_OK; // no partial
        }
        s.frames += v.frames;
        unsafe { out_string(out, format!(r#"{{"partial":{}}}"#, s.frames)) };
        SIGIL_OK
    }
    unsafe extern "C" fn stream_finish(s: *mut SigilStream, out: *mut *mut c_char) -> SigilStatus {
        let s = unsafe { &mut *s.cast::<Stream>() };
        unsafe { out_string(out, format!(r#"{{"final":{}}}"#, s.frames)) };
        s.frames = 0;
        SIGIL_OK
    }
    unsafe extern "C" fn stream_close(s: *mut SigilStream) {
        let s = unsafe { Box::from_raw(s.cast::<Stream>()) };
        event(s.engine_id, "stream_close");
    }
    unsafe extern "C" fn free_string(p: *mut c_char) {
        if !p.is_null() {
            drop(unsafe { CString::from_raw(p) });
            LIVE.with(|l| l.set(l.get() - 1));
        }
    }
    unsafe extern "C" fn free_audio(b: *mut SigilAudioBuffer) {
        let b = unsafe { &mut *b };
        if !b.samples.is_null() {
            let n = (b.frames * u64::from(b.channels)) as usize;
            drop(unsafe { Box::from_raw(std::ptr::slice_from_raw_parts_mut(b.samples, n)) });
            LIVE.with(|l| l.set(l.get() - 1));
        }
        *b = SigilAudioBuffer::default();
    }

    pub fn vtable() -> Vtable {
        Vtable {
            abi_version: abi_v1,
            get_info,
            last_error,
            set_log_callback: set_log,
            create,
            destroy,
            cancel,
            query,
            process,
            render,
            stream_open,
            stream_push,
            stream_finish,
            stream_close,
            free_string,
            free_audio,
        }
    }
}

fn fake_library() -> EngineLibrary {
    EngineLibrary::from_parts(fake::vtable(), PathBuf::from("fake/sigil_fake.dll"), None).expect("fake engine loads")
}

#[test]
fn abi_mismatches_and_bad_info_are_refused() {
    let path = PathBuf::from("fake/sigil_fake.dll");
    let mut vt = fake::vtable();
    vt.abi_version = fake::abi_v2;
    let err = EngineLibrary::from_parts(vt, path.clone(), None).unwrap_err();
    assert!(err.contains("ABI v2") && err.contains("supports v1") && err.contains("newer Sigil"), "{err}");

    for (get_info, needle) in [
        (fake::get_info_null as abi::GetInfoFn, "returned NULL"),
        (fake::get_info_small, "needs at least 56"),
        (fake::get_info_abi2, "reports ABI v2"),
        (fake::get_info_bad_name, "invalid engine name"),
    ] {
        let mut vt = fake::vtable();
        vt.get_info = get_info;
        let err = EngineLibrary::from_parts(vt, path.clone(), None).unwrap_err();
        assert!(err.contains(needle) && err.contains("sigil_fake.dll"), "{needle}: {err}");
    }
    assert!(loader::abi_mismatch_message("x.dll", 0).contains("older Sigil"));
}

#[test]
fn fake_engine_calls_copy_and_free_everything() {
    let lib = fake_library();
    let info = lib.info();
    assert_eq!((info.name.as_str(), info.version.as_str(), info.variant.as_str()), ("fake", "1.2.3", "cpu"));
    assert_eq!(info.capability_names(), ["process", "stream", "render", "cancel"]);
    assert!(info.has(SIGIL_CAP_RENDER) && !info.has(SIGIL_CAP_GPU));
    let before = fake::live();

    let mut e = lib.create(r#"{"id":101,"gain":0.5}"#).expect("create");
    assert_eq!(e.query(r#"{"op":"info"}"#).unwrap(), r#"{"name":"fake","backend":"cpu"}"#);
    assert_eq!(e.query("").unwrap(), r#"{"name":"fake","backend":"cpu"}"#, "blank JSON means {{}}");
    assert_eq!(e.query(r#"{"op":"null"}"#).unwrap(), "{}", "a NULL result reads as {{}}");

    let audio = [0.1f32, -0.2, 0.3, -0.4];
    let view = AudioView {
        samples: &audio,
        sample_rate: 16_000,
        channels: 2,
    };
    assert_eq!(e.process(view, "{}").unwrap(), r#"{"frames":2,"channels":2}"#);
    let out = e.render(Some(view), "{}").unwrap();
    assert_eq!(out.samples, vec![0.05, -0.1, 0.15, -0.2]);
    assert_eq!((out.sample_rate, out.channels, out.frames()), (16_000, 2, 2));
    let silent = e.render(None, "{}").unwrap();
    assert!(silent.samples.is_empty() && silent.sample_rate == 16_000);
    let empty = e.process(AudioView { samples: &[], sample_rate: 16_000, channels: 1 }, "{}").unwrap();
    assert_eq!(empty, r#"{"frames":0,"channels":1}"#);

    // Invalid audio never reaches the engine.
    for (samples, sr, ch, needle) in [
        (&audio[..3], 16_000, 2u16, "whole number"),
        (&audio[..], 0, 1, "sample rate of 0"),
        (&audio[..], 16_000, 0, "0 channels"),
    ] {
        let err = e
            .process_audio(AudioView { samples, sample_rate: sr, channels: ch }, "{}")
            .unwrap_err();
        assert_eq!(err.status, SIGIL_E_INVALID_ARGUMENT);
        assert!(err.message.contains(needle), "{}", err.message);
    }
    assert!(e.query("{\"op\":\"x\0\"}").unwrap_err().contains("NUL byte"));

    // A malformed buffer from the engine is an error, not a crash.
    let err = e.render_audio(Some(view), r#"{"bad_buffer":true}"#).unwrap_err();
    assert!(err.message.contains("without samples"), "{}", err.message);

    // Engine errors carry the status and the engine's last error.
    let err = e.query_engine(r#"{"op":"fail"}"#).unwrap_err();
    assert_eq!(err.status, SIGIL_E_RUNTIME);
    assert_eq!(err.message, "fake query failed (SIGIL_E_RUNTIME): query failed on purpose");
    drop(e);
    assert_eq!(fake::live(), before, "every string and buffer was released by the engine's own free");
    assert_eq!(fake::events(101), ["create", "destroy"]);
}

#[test]
fn create_failures_report_status_and_fallback_class() {
    let lib = fake_library();
    let err = lib.create_engine(r#"{"fail":"load"}"#).unwrap_err();
    assert_eq!(err.status, SIGIL_E_LOAD_FAILED);
    assert_eq!(err.message, "fake create failed (SIGIL_E_LOAD_FAILED): model file missing");
    assert!(!err.is_backend_failure(false), "a CPU variant does not fall through on a load failure");
    assert!(err.is_backend_failure(true), "a GPU variant does");
    let err = lib.create_engine(r#"{"fail":"backend"}"#).unwrap_err();
    assert_eq!(err.status, SIGIL_E_BACKEND_UNAVAILABLE);
    assert!(err.is_backend_failure(false));
    assert!(lib.create("{\"a\":\"\0\"}").unwrap_err().contains("NUL byte"));
}

#[test]
fn streams_keep_their_engine_alive_and_close_before_destroy() {
    let lib = fake_library();
    let mut e = lib.create(r#"{"id":202}"#).unwrap();
    let mut s = e.open_stream("{}").unwrap();
    drop(e); // the stream still owns the engine
    let chunk = [0.0f32; 160];
    let view = AudioView {
        samples: &chunk,
        sample_rate: 16_000,
        channels: 1,
    };
    assert_eq!(s.push(view).unwrap().as_deref(), Some(r#"{"partial":160}"#));
    assert_eq!(s.push(view).unwrap().as_deref(), Some(r#"{"partial":320}"#));
    assert_eq!(s.push(AudioView { samples: &[], sample_rate: 16_000, channels: 1 }).unwrap(), None);
    assert_eq!(s.finish().unwrap(), r#"{"final":320}"#);
    assert_eq!(s.push(view).unwrap().as_deref(), Some(r#"{"partial":160}"#), "pushable after finish");
    assert_eq!(fake::events(202), ["create", "stream_open"], "not destroyed while the stream lives");
    // Streams move to other threads (Send).
    let s = std::thread::spawn(move || {
        let _ = s.finish();
        s
    })
    .join()
    .unwrap();
    drop(s);
    assert_eq!(fake::events(202), ["create", "stream_open", "stream_close", "destroy"]);
}

#[test]
fn cancel_from_another_thread_interrupts_a_running_call() {
    let lib = fake_library();
    let mut e = lib.create(r#"{"id":303}"#).unwrap();
    let canceller = e.canceller();
    assert!(canceller.is_alive());
    let c2 = canceller.clone();
    let fire = std::thread::spawn(move || {
        std::thread::sleep(std::time::Duration::from_millis(30));
        c2.cancel();
    });
    let started = std::time::Instant::now();
    let err = e.query_engine(r#"{"op":"sleep","ms":10000}"#).unwrap_err();
    fire.join().unwrap();
    assert!(err.is_cancelled(), "{err:?}");
    assert!(started.elapsed() < std::time::Duration::from_secs(5));
    // A later call runs normally (the engine resets its flag per call).
    assert!(e.query(r#"{"op":"sleep","ms":1}"#).is_ok());
    drop(e);
    assert!(!canceller.is_alive());
    canceller.cancel(); // no-op after the instance is gone
    assert_eq!(fake::events(303), ["create", "destroy"]);
}

#[test]
fn engine_handles_are_send() {
    fn send<T: Send>() {}
    fn send_sync<T: Send + Sync>() {}
    send::<EngineInstance>();
    send::<EngineStream>();
    send_sync::<EngineCanceller>();
    send_sync::<EngineLibrary>();
}

// ---------------------------------------------------------------------------------------------
// Pack manifests and hashes
// ---------------------------------------------------------------------------------------------

fn write_pack(dir: &Path, dll_bytes: &[u8]) -> PackManifest {
    std::fs::create_dir_all(dir).unwrap();
    std::fs::write(dir.join("sigil_echo.dll"), dll_bytes).unwrap();
    let manifest = PackManifest {
        schema: 1,
        engine: "echo".into(),
        version: "0.1.0".into(),
        variant: "cpu".into(),
        abi_version: 1,
        library: "sigil_echo.dll".into(),
        kind: "test".into(),
        capabilities: 15,
        build_info: "test".into(),
        files: vec![PackFile {
            path: "sigil_echo.dll".into(),
            size: dll_bytes.len() as u64,
            sha256: sha256_hex(dll_bytes),
        }],
    };
    std::fs::write(dir.join(PACK_MANIFEST), serde_json::to_vec_pretty(&manifest).unwrap()).unwrap();
    manifest
}

#[test]
fn pack_manifests_verify_sizes_and_hashes() {
    let dir = temp_dir("manifest");
    assert_eq!(PackManifest::read(&dir).unwrap(), None, "no manifest");
    assert_eq!(pack::verify_library(&dir, "sigil_echo.dll").unwrap(), None);
    let m = write_pack(&dir, b"MZ fake dll bytes");
    assert_eq!(PackManifest::read(&dir).unwrap().as_ref(), Some(&m));
    assert_eq!(pack::verify_library(&dir, "sigil_echo.dll").unwrap(), Some(m.clone()));
    assert_eq!(m.info().name, "echo");
    assert!(pack::verify_library(&dir, "sigil_other.dll").unwrap_err().contains("not listed"));

    // Same size, different bytes: the hash catches it (after the cached hash is invalidated by the
    // new modification time).
    std::thread::sleep(std::time::Duration::from_millis(20));
    std::fs::write(dir.join("sigil_echo.dll"), b"MZ fake dll bytez").unwrap();
    let err = pack::verify_library(&dir, "sigil_echo.dll").unwrap_err();
    assert!(err.contains("does not match its pack manifest") && err.contains("reinstall"), "{err}");
    std::fs::write(dir.join("sigil_echo.dll"), b"MZ").unwrap();
    assert!(pack::verify_library(&dir, "sigil_echo.dll").unwrap_err().contains("bytes but the pack manifest says"));
    std::fs::remove_file(dir.join("sigil_echo.dll")).unwrap();
    assert!(pack::verify_library(&dir, "sigil_echo.dll").unwrap_err().contains("missing from the pack"));

    // Manifests from another ABI or schema, unsafe paths and missing libraries are refused.
    let bad = |edit: &dyn Fn(&mut PackManifest)| {
        let mut m = m.clone();
        edit(&mut m);
        m.validate()
    };
    assert!(bad(&|m| m.schema = 2).unwrap_err().contains("schema 2"));
    assert!(bad(&|m| m.files[0].path = "../x.dll".into()).unwrap_err().contains("unsafe"));
    assert!(bad(&|m| m.files[0].path = "C:/x.dll".into()).unwrap_err().contains("unsafe"));
    assert!(bad(&|m| m.files[0].sha256 = "ABC".into()).unwrap_err().contains("64 lower-case"));
    assert!(bad(&|m| m.library = "sigil_x.dll".into()).unwrap_err().contains("not listed"));
    assert!(bad(&|m| m.variant = "gpu".into()).unwrap_err().contains("unknown variant"));
    assert!(bad(&|m| m.engine = "Echo".into()).unwrap_err().contains("invalid engine name"));
    assert!(bad(&|m| m.files.push(m.files[0].clone())).unwrap_err().contains("twice"));

    let dir2 = temp_dir("manifest-abi");
    let mut m2 = write_pack(&dir2, b"MZ");
    m2.abi_version = 2;
    std::fs::write(dir2.join(PACK_MANIFEST), serde_json::to_vec(&m2).unwrap()).unwrap();
    let err = pack::verify_library(&dir2, "sigil_echo.dll").unwrap_err();
    assert!(err.contains("ABI v2"), "an ABI mismatch is refused before the DLL is loaded: {err}");
    std::fs::write(dir2.join(PACK_MANIFEST), b"{ not json").unwrap();
    assert!(PackManifest::read(&dir2).unwrap_err().contains("not a valid pack manifest"));

    let info = EngineInfo {
        abi_version: 1,
        name: "echo".into(),
        version: "0.2.0".into(),
        variant: "cpu".into(),
        ..EngineInfo::default()
    };
    let err = m.check_matches(&info).unwrap_err();
    assert!(err.contains("version \"0.1.0\" vs \"0.2.0\""), "{err}");
    let _ = std::fs::remove_dir_all(&dir);
    let _ = std::fs::remove_dir_all(&dir2);
}

#[test]
fn safe_relative_paths() {
    for ok in ["a.dll", "bin/x64/cudnn64_9.dll", "sigil-pack.json", "a+b_c-d.e"] {
        assert!(pack::safe_rel_path(ok), "{ok}");
    }
    for bad in ["", "/a", "a\\b", "C:x", "a/../b", "./a", "a//b", "a.", "a /b", "a b"] {
        assert!(!pack::safe_rel_path(bad), "{bad}");
    }
}

// ---------------------------------------------------------------------------------------------
// Catalog discovery and planning
// ---------------------------------------------------------------------------------------------

fn test_catalog() -> Catalog {
    let entry = |id: &str, variant: &str, status: &str| {
        serde_json::json!({
            "id": id, "family": "runtime", "kind": "runtime", "version": "0.1.0", "license": "MIT",
            "status": status, "variant": variant,
            "files": [{ "path": "p.zip", "size": 0, "sources": [{ "kind": "direct", "url": "https://example.invalid/p.zip" }] }]
        })
    };
    let mut cuda = entry("runtime/sigil-echo-cuda13", "cuda13", "published");
    cuda["requires"] = serde_json::json!(["runtime/nvidia-cuda13"]);
    let doc = serde_json::json!({ "entries": [
        entry("runtime/sigil-echo-cpu", "cpu", "unpublished"),
        entry("runtime/sigil-echo-cpu-avx2", "cpu-avx2", "unpublished"),
        cuda,
        entry("runtime/sigil-long-name-x-vulkan", "vulkan", "published"),
        entry("runtime/nvidia-cuda13", "cuda13", "published"),
        entry("runtime/llama-server-cpu", "cpu", "published"),
        entry("runtime/sigil-bad-cpu", "vulkan", "published"),
    ]});
    Catalog::from_json(&doc.to_string()).unwrap()
}

#[test]
fn engine_pack_ids_follow_the_convention() {
    assert_eq!(
        pack::parse_engine_pack_id("runtime/sigil-echo-cpu-avx2", Some("cpu-avx2")),
        Some(("echo".to_string(), RuntimeVariant::CpuAvx2))
    );
    assert_eq!(
        pack::parse_engine_pack_id("runtime/sigil-long-name-x-vulkan", Some("vulkan")),
        Some(("long-name-x".to_string(), RuntimeVariant::Vulkan))
    );
    assert_eq!(pack::parse_engine_pack_id("runtime/sigil-bad-cpu", Some("vulkan")), None, "id and variant disagree");
    assert_eq!(pack::parse_engine_pack_id("runtime/llama-server-cpu", Some("cpu")), None);
    assert_eq!(pack::parse_engine_pack_id("runtime/sigil-echo-cpu", None), None);
    assert_eq!(engine_pack_id("echo", RuntimeVariant::Cuda13), "runtime/sigil-echo-cuda13");
    assert_eq!(engine_dll_name("echo"), "sigil_echo.dll");

    let c = test_catalog();
    let echo: Vec<_> = engine_pack_entries(&c, Some("echo")).into_iter().map(|e| (e.id, e.variant, e.published)).collect();
    assert_eq!(
        echo,
        vec![
            ("runtime/sigil-echo-cpu".to_string(), RuntimeVariant::Cpu, false),
            ("runtime/sigil-echo-cpu-avx2".to_string(), RuntimeVariant::CpuAvx2, false),
            ("runtime/sigil-echo-cuda13".to_string(), RuntimeVariant::Cuda13, true),
        ]
    );
    assert_eq!(engine_pack_entries(&c, None).len(), 4);
}

#[test]
fn planning_orders_installed_variants_and_explains_skips() {
    let c = test_catalog();
    let root = PathBuf::from("Z:/packs");
    let installed_all = |id: &str| Some(root.join(id.replace('/', "_")));
    let avx2_pc = VariantSupport {
        avx2: true,
        ..VariantSupport::default()
    };
    let (plan, skipped) = pack::plan("echo", &OpenRequest::default(), &c, &installed_all, &avx2_pc).unwrap();
    let order: Vec<_> = plan.iter().map(|p| p.variant.unwrap()).collect();
    assert_eq!(order, [RuntimeVariant::CpuAvx2, RuntimeVariant::Cpu]);
    assert_eq!(plan[0].pack_id.as_deref(), Some("runtime/sigil-echo-cpu-avx2"));
    assert_eq!(skipped.len(), 1);
    assert_eq!(skipped[0].variant, Some(RuntimeVariant::Cuda13));
    assert!(skipped[0].error.contains("CUDA 13"), "{}", skipped[0].error);

    // Nothing installed: the ids to install are named.
    let none = |_: &str| None;
    let err = pack::plan("echo", &OpenRequest::default(), &c, &none, &avx2_pc).unwrap_err();
    assert!(err.contains("not installed") && err.contains("runtime/sigil-echo-cpu"), "{err}");
    // Only unsupported variants installed.
    let only_cuda = |id: &str| (id == "runtime/sigil-echo-cuda13").then(|| root.clone());
    let err = pack::plan("echo", &OpenRequest::default(), &c, &only_cuda, &avx2_pc).unwrap_err();
    assert!(err.contains("none of the installed echo variants"), "{err}");
    // A forced variant that is not installed.
    let only_cpu = |id: &str| (id == "runtime/sigil-echo-cpu").then(|| root.clone());
    let req = OpenRequest {
        variant: Some(RuntimeVariant::CpuAvx2),
        ..OpenRequest::default()
    };
    assert!(pack::plan("echo", &req, &c, &only_cpu, &avx2_pc).unwrap_err().contains("cpu-avx2 variant of echo is not installed"));
    // Unknown engine / invalid names.
    assert!(pack::plan("nope", &OpenRequest::default(), &c, &installed_all, &avx2_pc).unwrap_err().contains("no runtime pack"));
    assert!(pack::plan("../x", &OpenRequest::default(), &c, &installed_all, &avx2_pc).unwrap_err().contains("invalid engine name"));
    // A runtime folder overrides the catalog.
    let dir = temp_dir("plan-dir");
    let req = OpenRequest {
        runtime_dir: Some(dir.clone()),
        ..OpenRequest::default()
    };
    let (plan, _) = pack::plan("echo", &req, &Catalog::default(), &none, &avx2_pc).unwrap();
    assert_eq!((plan.len(), plan[0].variant, plan[0].pack_id.clone()), (1, None, None));
    let req = OpenRequest {
        runtime_dir: Some(dir.join("missing")),
        ..OpenRequest::default()
    };
    assert!(pack::plan("echo", &req, &Catalog::default(), &none, &avx2_pc).unwrap_err().contains("is not a folder"));
    let _ = std::fs::remove_dir_all(&dir);
}

/// Write a pack whose manifest claims `variant` (the bytes are not a real DLL: every check here
/// must refuse it before anything is loaded).
fn write_variant_pack(dir: &Path, variant: &str) -> PackManifest {
    let mut m = write_pack(dir, format!("MZ {variant} build").as_bytes());
    m.variant = variant.into();
    std::fs::write(dir.join(PACK_MANIFEST), serde_json::to_vec_pretty(&m).unwrap()).unwrap();
    m
}

#[test]
fn unsupported_instruction_sets_are_never_planned_or_loaded() {
    let c = test_catalog();
    let root = temp_dir("plan-cpu");
    let avx2_pc = VariantSupport {
        avx2: true,
        ..VariantSupport::default()
    };
    let old_cpu = VariantSupport {
        cuda13: true,
        ..VariantSupport::default()
    };

    // A forced variant this CPU cannot execute: refused with the reason, not "tried anyway".
    let installed_all = |id: &str| Some(root.join(id.replace('/', "_")));
    let forced = |v| OpenRequest {
        variant: Some(v),
        ..OpenRequest::default()
    };
    let err = pack::plan("echo", &forced(RuntimeVariant::CpuAvx2), &c, &installed_all, &old_cpu).unwrap_err();
    assert!(err.contains("cpu-avx2 variant of echo cannot run on this PC") && err.contains("AVX2"), "{err}");
    let err = pack::plan("echo", &forced(RuntimeVariant::Cuda13), &c, &installed_all, &old_cpu).unwrap_err();
    assert!(err.contains("cuda13 variant") && err.contains("AVX2"), "{err}");
    // Without its GPU a forced variant is still tried (create reports the backend error).
    let (plan, skipped) = pack::plan("echo", &forced(RuntimeVariant::Cuda13), &c, &installed_all, &avx2_pc).unwrap();
    assert_eq!(plan.iter().map(|p| p.variant).collect::<Vec<_>>(), [Some(RuntimeVariant::Cuda13)]);
    assert!(skipped.iter().all(|a| a.error == "only cuda13 was requested"), "{skipped:?}");

    // A development folder whose manifest names a build this CPU cannot run.
    let dir = root.join("dev-avx512");
    write_variant_pack(&dir, "cpu-avx512");
    let none = |_: &str| None;
    let in_dir = |variant| OpenRequest {
        variant,
        runtime_dir: Some(dir.clone()),
        ..OpenRequest::default()
    };
    let err = pack::plan("echo", &in_dir(None), &Catalog::default(), &none, &avx2_pc).unwrap_err();
    assert!(err.contains("cpu-avx512 build of echo") && err.contains("AVX-512"), "{err}");
    let err = pack::plan("echo", &in_dir(Some(RuntimeVariant::Cpu)), &Catalog::default(), &none, &avx2_pc).unwrap_err();
    assert!(err.contains("holds the cpu-avx512 build of echo, not cpu"), "{err}");
    let avx512_pc = VariantSupport {
        avx512: true,
        ..avx2_pc
    };
    let (plan, _) = pack::plan("echo", &in_dir(None), &Catalog::default(), &none, &avx512_pc).unwrap();
    assert_eq!(plan[0].variant, Some(RuntimeVariant::CpuAvx512), "the manifest names the folder's variant");
    // A forced variant for a folder without a manifest is checked too.
    let bare = root.join("dev-bare");
    std::fs::create_dir_all(&bare).unwrap();
    let req = OpenRequest {
        variant: Some(RuntimeVariant::CpuAvx512),
        runtime_dir: Some(bare.clone()),
        ..OpenRequest::default()
    };
    assert!(pack::plan("echo", &req, &Catalog::default(), &none, &avx2_pc).unwrap_err().contains("AVX-512"));

    // The loader refuses such a pack by its manifest before loading (the bytes are not even a DLL;
    // on a CPU with AVX-512 the same call gets as far as the load and fails there).
    let err = EngineLibrary::load_checked(&dir, "sigil_echo.dll", &avx2_pc, false).unwrap_err();
    assert!(err.contains("cpu-avx512 build") && err.contains("AVX-512"), "{err}");
    let err = EngineLibrary::load_checked(&dir, "sigil_echo.dll", &avx512_pc, false).unwrap_err();
    assert!(!err.contains("AVX-512"), "loaded past the CPU check: {err}");
    let _ = std::fs::remove_dir_all(&root);
}

#[test]
fn installed_packs_must_carry_a_matching_manifest() {
    let root = temp_dir("preflight");
    let avx2_pc = VariantSupport {
        avx2: true,
        ..VariantSupport::default()
    };
    let candidate = |dir: &Path, variant: Option<RuntimeVariant>, pack_id: Option<&str>| pack::Candidate {
        variant,
        pack_id: pack_id.map(str::to_string),
        dir: dir.to_path_buf(),
    };

    // No manifest: an installed pack is refused (never loaded unverified), a development folder is
    // not (it loads logged as not hash-verified).
    let bare = root.join("bare");
    std::fs::create_dir_all(&bare).unwrap();
    std::fs::write(bare.join("sigil_echo.dll"), b"MZ").unwrap();
    let installed_bare = candidate(&bare, Some(RuntimeVariant::Cpu), Some("runtime/sigil-echo-cpu"));
    let err = pack::preflight("echo", &installed_bare, &avx2_pc).unwrap_err();
    assert!(err.contains("has no sigil-pack.json") && err.contains("reinstall runtime/sigil-echo-cpu"), "{err}");
    let err = EngineLibrary::load_checked(&bare, "sigil_echo.dll", &avx2_pc, true).unwrap_err();
    assert!(err.contains("cannot be verified"), "{err}");
    assert_eq!(pack::preflight("echo", &candidate(&bare, None, None), &avx2_pc).unwrap(), "sigil_echo.dll");

    // A mislabeled pack: an AVX-512 build installed as the cpu variant is refused before loading,
    // whatever this CPU supports (the manifest and the DLL agree with each other, not with the
    // catalog entry).
    let mislabeled = root.join("mislabeled");
    write_variant_pack(&mislabeled, "cpu-avx512");
    let avx512_pc = VariantSupport {
        avx512: true,
        ..avx2_pc
    };
    for pc in [avx2_pc, avx512_pc] {
        let err = pack::preflight("echo", &candidate(&mislabeled, Some(RuntimeVariant::Cpu), Some("runtime/sigil-echo-cpu")), &pc)
            .unwrap_err();
        assert!(err.contains("holds the cpu-avx512 build but is installed as cpu") && err.contains("reinstall"), "{err}");
    }
    // The installed variant itself must run here.
    let err = pack::preflight("echo", &candidate(&mislabeled, Some(RuntimeVariant::CpuAvx512), Some("x")), &avx2_pc).unwrap_err();
    assert!(err.contains("AVX-512"), "{err}");
    // A development folder without a forced variant: the manifest's build is checked.
    let err = pack::preflight("echo", &candidate(&mislabeled, None, None), &avx2_pc).unwrap_err();
    assert!(err.contains("AVX-512"), "{err}");
    assert_eq!(pack::preflight("echo", &candidate(&mislabeled, None, None), &avx512_pc).unwrap(), "sigil_echo.dll");

    // Another engine's pack.
    let other = root.join("other");
    let mut m = write_pack(&other, b"MZ other");
    m.engine = "other".into();
    std::fs::write(other.join(PACK_MANIFEST), serde_json::to_vec(&m).unwrap()).unwrap();
    let err = pack::preflight("echo", &candidate(&other, Some(RuntimeVariant::Cpu), Some("runtime/sigil-echo-cpu")), &avx2_pc)
        .unwrap_err();
    assert!(err.contains("holds the 'other' engine"), "{err}");

    // A correct pack passes and names its library.
    let good = root.join("good");
    write_pack(&good, b"MZ good");
    assert_eq!(
        pack::preflight("echo", &candidate(&good, Some(RuntimeVariant::Cpu), Some("runtime/sigil-echo-cpu")), &avx2_pc).unwrap(),
        "sigil_echo.dll"
    );
    let _ = std::fs::remove_dir_all(&root);
}

#[test]
fn opening_from_an_empty_or_missing_folder_is_an_error() {
    let dir = temp_dir("open-empty");
    let req = OpenRequest {
        runtime_dir: Some(dir.clone()),
        config_json: Some("{}".into()),
        ..OpenRequest::default()
    };
    let err = open_engine("echo", &req).unwrap_err();
    assert!(err.contains("could not be loaded") && err.contains("sigil_echo.dll"), "{err}");
    let _ = std::fs::remove_dir_all(&dir);
}

// ---------------------------------------------------------------------------------------------
// Embedded catalog: the NVIDIA runtime and provider packs
// ---------------------------------------------------------------------------------------------

/// `bin/**/<name pattern>` include globs, as the model store matches them.
fn include_matches(glob: &str, file: &str) -> bool {
    let Some(pattern) = glob.strip_prefix("bin/**/") else {
        return false;
    };
    match pattern.split_once('*') {
        None => pattern == file,
        Some((pre, post)) => file.len() >= pre.len() + post.len() && file.starts_with(pre) && file.ends_with(post),
    }
}

#[test]
fn nvidia_pack_entry_matches_the_interim_pins_and_the_preload_order() {
    use crate::services::diagnostics::nvidia_pack;
    use crate::services::model_store::{EntryKind, PublishStatus, SourceKind};
    let catalog = Catalog::embedded().unwrap();
    let e = catalog.get(nvidia::NVIDIA_PACK_ID).expect("runtime/nvidia-cuda13 is in the catalog");
    assert_eq!((e.kind, e.status, e.variant.as_deref()), (EntryKind::Runtime, PublishStatus::Published, Some("cuda13")));
    assert_eq!(e.version, nvidia_pack::LABEL, "same pinned release as the interim installer");
    assert_eq!(e.files.len(), nvidia_pack::ARCHIVES.len());
    for (f, a) in e.files.iter().zip(nvidia_pack::ARCHIVES.iter()) {
        assert_eq!(f.sha256.as_deref(), Some(a.sha256), "{}", f.path);
        assert_eq!(f.size, a.size, "{}", f.path);
        assert_eq!(f.sources.len(), 1);
        assert_eq!(f.sources[0].kind, SourceKind::NvidiaRedist);
        assert_eq!(f.sources[0].url, nvidia_pack::archive_url(a));
        assert!(f.sources[0].url.ends_with(&f.path), "{}", f.path);
        let spec = f.archive.as_ref().expect("extracted");
        assert_eq!(spec.strip, 1);
        for dll in a.dlls {
            assert!(spec.include.iter().any(|g| include_matches(g, dll)), "{dll} is extracted from {}", f.path);
        }
    }
    // Every DLL the interim installer preloads is preloaded from the store pack too, with the same
    // requirement (cuDNN sub-libraries beyond PRELOAD_ORDER are picked up by `preload_list`).
    for (name, required) in nvidia_pack::PRELOAD_ORDER.iter() {
        let listed = nvidia::PRELOAD_ORDER.iter().any(|(n, r)| n == name && r == required);
        let globbed = !required && name.starts_with("cudnn") && name.ends_with("64_9.dll");
        assert!(listed || globbed, "{name}");
    }
    for (name, required) in nvidia::PRELOAD_ORDER.iter().filter(|(_, r)| *r) {
        assert!(nvidia_pack::PRELOAD_ORDER.iter().any(|(n, r)| n == name && r == required), "{name}");
    }
    let size: u64 = e.files.iter().map(|f| f.size).sum();
    assert_eq!(size, nvidia_pack::download_bytes());
}

#[test]
fn provider_and_echo_entries_are_well_formed() {
    use crate::services::model_store::PublishStatus;
    let catalog = Catalog::embedded().unwrap();
    let p = catalog.get(nvidia::CUDA_PROVIDER_PACK_ID).expect("provider pack entry");
    assert_eq!(p.status, PublishStatus::Unpublished, "local-only until the maintainer publishes it");
    assert_eq!(p.version, crate::services::neural_audio::ort_session::runtime_version(), "provider matches the linked ORT");
    assert_eq!(p.requires, [nvidia::NVIDIA_PACK_ID]);
    let f = &p.files[0];
    assert_eq!(f.path, nvidia::CUDA_PROVIDER_FILE);
    let pin = f.sha256.as_deref().expect("pinned");
    // The provider ort-sys placed next to the build output (when present) is the pinned file.
    if let Some(deps) = crate::services::neural_audio::ort_session::provider_dir() {
        for dir in [deps.clone(), deps.parent().map(Path::to_path_buf).unwrap_or(deps)] {
            let dll = dir.join(nvidia::CUDA_PROVIDER_FILE);
            if dll.is_file() {
                assert_eq!(pack::sha256_cached(&dll).unwrap(), pin, "{}", dll.display());
                assert_eq!(std::fs::metadata(&dll).unwrap().len(), f.size);
            }
        }
    }
    let echo = engine_pack_entries(&catalog, Some("echo"));
    let variants: Vec<_> = echo.iter().map(|e| e.variant).collect();
    assert_eq!(variants, [RuntimeVariant::Cpu, RuntimeVariant::CpuAvx2]);
    for e in &echo {
        let entry = catalog.get(&e.id).unwrap();
        assert_eq!(entry.files[0].path, format!("sigil-echo-{}-{}.zip", entry.version, e.variant.as_str()));
        assert!(entry.files[0].archive.is_some());
    }
}

// ---------------------------------------------------------------------------------------------
// DLL discovery, ordered preloads, provider verification, leftovers
// ---------------------------------------------------------------------------------------------

#[test]
fn dll_discovery_prefers_shallow_paths_and_ignores_case() {
    let root = temp_dir("find");
    std::fs::create_dir_all(root.join("bin/x64")).unwrap();
    std::fs::create_dir_all(root.join("deep/a/b/c/d/e")).unwrap();
    std::fs::write(root.join("bin/x64/CUDNN64_9.dll"), b"x").unwrap();
    std::fs::write(root.join("bin/cudnn64_9.dll"), b"y").unwrap();
    std::fs::write(root.join("bin/x64/cublas64_13.dll"), b"z").unwrap();
    std::fs::write(root.join("bin/readme.txt"), b"t").unwrap();
    std::fs::write(root.join("deep/a/b/c/d/e/too_deep.dll"), b"d").unwrap();
    let found = dll_dirs::find_dlls(&root, 4);
    assert_eq!(found.get("cudnn64_9.dll").unwrap(), &root.join("bin/cudnn64_9.dll"), "shallower wins");
    assert!(found.contains_key("cublas64_13.dll"));
    assert!(!found.contains_key("readme.txt") && !found.contains_key("too_deep.dll"));

    // A missing required DLL fails; a missing optional one is skipped; garbage fails to load.
    let err = dll_dirs::preload_ordered(&[("missing64.dll", true)], &found).unwrap_err();
    assert!(err.contains("missing64.dll is missing"), "{err}");
    assert_eq!(dll_dirs::preload_ordered(&[("missing64.dll", false)], &found).unwrap(), Vec::<String>::new());
    let err = dll_dirs::preload_ordered(&[("cublas64_13.dll", true)], &found).unwrap_err();
    assert!(err.contains("cannot load"), "{err}");
    assert!(add_dll_directory(&root.join("nope")).is_err());
    assert!(add_dll_directory(&root).is_ok() && add_dll_directory(&root).is_ok(), "idempotent");
    let _ = std::fs::remove_dir_all(&root);
}

#[test]
fn every_cudnn_sub_library_of_the_pack_is_preloaded() {
    let root = PathBuf::from("Z:/pack/bin/x64");
    let index: std::collections::BTreeMap<String, PathBuf> = [
        "cudart64_13.dll",
        "cudnn64_9.dll",
        "cudnn_graph64_9.dll",
        "cudnn_engines_tensor_ir64_9.dll", // new in cuDNN 9.2x
        "cudnn_ext64_9.dll",
        "cufft64_12.dll",
        "zlibwapi.dll",
    ]
    .into_iter()
    .map(|n| (n.to_string(), root.join(n)))
    .collect();
    let list = nvidia::preload_list(&index);
    assert_eq!(&list[..nvidia::PRELOAD_ORDER.len()], nvidia::PRELOAD_ORDER.map(|(n, r)| (n.to_string(), r)).as_slice());
    let extra: Vec<_> = list[nvidia::PRELOAD_ORDER.len()..].to_vec();
    assert_eq!(
        extra,
        [("cudnn_engines_tensor_ir64_9.dll".to_string(), false), ("cudnn_ext64_9.dll".to_string(), false)],
        "unknown cuDNN sub-libraries are loaded too (optional); other DLLs are not"
    );
}

#[test]
fn provider_pack_files_are_checked_against_the_pin() {
    let dir = temp_dir("provider");
    assert!(nvidia::verify_provider(&dir, None).unwrap_err().contains("missing"));
    std::fs::write(dir.join(nvidia::CUDA_PROVIDER_FILE), b"provider bytes").unwrap();
    assert_eq!(nvidia::verify_provider(&dir, None).unwrap(), dir.join(nvidia::CUDA_PROVIDER_FILE));
    let pin = sha256_hex(b"provider bytes");
    assert!(nvidia::verify_provider(&dir, Some(&pin)).is_ok());
    assert!(nvidia::verify_provider(&dir, Some(&pin.to_uppercase())).is_ok());
    let err = nvidia::verify_provider(&dir, Some(&"0".repeat(64))).unwrap_err();
    assert!(err.contains("does not match the pinned CUDA provider"), "{err}");
    let _ = std::fs::remove_dir_all(&dir);
}

#[test]
fn leftover_runtimes_are_listed_and_removed() {
    let app = temp_dir("legacy");
    assert!(nvidia::legacy_runtimes(&app, true).is_empty());
    let cu12 = app.join("runtimes/nvidia-cu12/cuda-12.9.1_cudnn-9.10.2/bin");
    std::fs::create_dir_all(&cu12).unwrap();
    std::fs::write(cu12.join("cudnn64_9.dll"), vec![0u8; 2048]).unwrap();
    let rows = nvidia::legacy_runtimes(&app, false);
    assert_eq!(rows.len(), 1);
    assert_eq!(rows[0].id, nvidia::LEGACY_CU12_ID);
    assert_eq!(rows[0].bytes, 2048);
    assert!(rows[0].note.contains("CUDA 13"), "{}", rows[0].note);
    let listed = legacy_rows(&app, false);
    assert!(listed[0].legacy && !listed[0].loadable && listed[0].error.is_some());
    nvidia::remove_legacy(&app, nvidia::LEGACY_CU12_ID, false).unwrap();
    assert!(!app.join("runtimes/nvidia-cu12").exists());
    assert!(nvidia::remove_legacy(&app, nvidia::LEGACY_CU12_ID, false).is_ok(), "already gone: ok");
    assert!(nvidia::remove_legacy(&app, "runtime/nvidia-cuda13", true).is_err(), "only leftovers");
    std::fs::create_dir_all(app.join("runtimes/.trash-nvidia-cu12-1/bin")).unwrap();
    nvidia::sweep_legacy_trash(&app);
    assert!(!app.join("runtimes/.trash-nvidia-cu12-1").exists());
    let _ = std::fs::remove_dir_all(&app);
}

/// A complete interim (P0 installer) NVIDIA pack under `app/runtimes/nvidia-cu13/<label>`, as
/// `diagnostics::nvidia_pack::verify_installed` checks it (one-byte DLLs).
fn write_interim_pack(app: &Path) -> PathBuf {
    use crate::services::diagnostics::nvidia_pack;
    let dir = nvidia_pack::install_dir(app);
    std::fs::create_dir_all(dir.join("bin")).unwrap();
    let files: Vec<nvidia_pack::InstalledFile> = nvidia_pack::required_dlls()
        .map(|name| {
            std::fs::write(dir.join("bin").join(name), b"x").unwrap();
            nvidia_pack::InstalledFile {
                name: name.to_string(),
                size: 1,
                sha256: sha256_hex(b"x"),
            }
        })
        .collect();
    let manifest = nvidia_pack::InstalledManifest {
        schema: 1,
        family: nvidia_pack::FAMILY.into(),
        label: nvidia_pack::LABEL.into(),
        installed_at_unix_ms: 0,
        files,
        sources: Vec::new(),
    };
    std::fs::write(dir.join("installed.json"), serde_json::to_vec(&manifest).unwrap()).unwrap();
    assert!(nvidia_pack::status(Some(app)).installed, "the fake interim pack verifies");
    dir
}

#[test]
fn the_interim_pack_is_a_leftover_only_while_the_store_pack_is_installed() {
    let app = temp_dir("legacy-interim");
    let interim = write_interim_pack(&app);
    assert!(nvidia::legacy_runtimes(&app, false).is_empty(), "the only NVIDIA runtime is not a leftover");
    let rows = nvidia::legacy_runtimes(&app, true);
    assert_eq!(rows.iter().map(|r| r.id).collect::<Vec<_>>(), [nvidia::LEGACY_INTERIM_ID]);

    // A stale "previous installer" row after the store pack was removed: refused, nothing deleted.
    let err = nvidia::remove_legacy(&app, nvidia::LEGACY_INTERIM_ID, false).unwrap_err();
    assert!(err.contains("not a leftover") && err.contains("runtime/nvidia-cuda13"), "{err}");
    assert!(interim.join("installed.json").is_file());

    // Marked for removal (deferred because it was loaded): no longer offered, Remove is a no-op.
    std::fs::write(interim.join(".pending-removal"), b"remove at next start\n").unwrap();
    assert!(nvidia::legacy_runtimes(&app, true).is_empty(), "removal already pending");
    assert!(nvidia::remove_legacy(&app, nvidia::LEGACY_INTERIM_ID, true).is_ok());
    assert!(interim.join("installed.json").is_file(), "left for the next start's sweep");
    std::fs::remove_file(interim.join(".pending-removal")).unwrap();

    // With the store pack installed it is removed.
    nvidia::remove_legacy(&app, nvidia::LEGACY_INTERIM_ID, true).unwrap();
    assert!(!interim.join("installed.json").exists());
    assert!(nvidia::legacy_runtimes(&app, true).is_empty());
    assert!(nvidia::remove_legacy(&app, nvidia::LEGACY_INTERIM_ID, true).is_ok(), "already gone: ok");
    let _ = std::fs::remove_dir_all(&app);
}

// ---------------------------------------------------------------------------------------------
// list_runtime_packs rows
// ---------------------------------------------------------------------------------------------

#[test]
fn rows_explain_why_a_pack_cannot_be_used() {
    let c = test_catalog();
    let root = temp_dir("rows");
    let echo_cpu = root.join("echo-cpu");
    write_pack(&echo_cpu, b"MZ echo");
    let echo_avx2 = root.join("echo-avx2"); // installed without a manifest
    std::fs::create_dir_all(&echo_avx2).unwrap();
    let installed = |id: &str| match id {
        "runtime/sigil-echo-cpu" => Some(echo_cpu.clone()),
        "runtime/sigil-echo-cpu-avx2" => Some(echo_avx2.clone()),
        "runtime/sigil-echo-cuda13" => Some(root.clone()),
        "runtime/llama-server-cpu" => Some(root.clone()),
        _ => None,
    };
    let check_provider = |_: &Path| Ok(());
    let inputs = RowInputs {
        installed: &installed,
        support: VariantSupport {
            avx2: false,
            ..VariantSupport::default()
        },
        check_provider: &check_provider,
        nvidia_error: None,
    };
    let rows = pack_rows(&c, &inputs);
    let row = |id: &str| rows.iter().find(|r| r.id == id).unwrap();
    assert_eq!(rows.len(), 7, "every runtime entry of the catalog");

    let cpu = row("runtime/sigil-echo-cpu");
    assert!(cpu.loadable, "{cpu:?}");
    assert_eq!(cpu.variant, Some(RuntimeVariant::Cpu));
    assert_eq!(cpu.info.as_ref().unwrap().name, "echo");
    assert_eq!(cpu.dir.as_deref(), Some(echo_cpu.display().to_string().as_str()));

    let avx2 = row("runtime/sigil-echo-cpu-avx2");
    assert!(!avx2.loadable);
    let why = avx2.error.as_deref().unwrap();
    assert!(why.contains("AVX2") && why.contains("sigil-pack.json"), "{why}");

    let cuda = row("runtime/sigil-echo-cuda13");
    assert!(cuda.error.as_deref().unwrap().contains("needs runtime/nvidia-cuda13"), "{cuda:?}");

    let not_installed = row("runtime/sigil-long-name-x-vulkan");
    assert_eq!((not_installed.loadable, not_installed.error.as_deref()), (false, Some("not installed")));
    assert!(row("runtime/llama-server-cpu").loadable, "plain runtime packs only need a supported variant");
    assert!(!rows.iter().any(|r| r.legacy));

    // Provider pack problems and NVIDIA preload failures are reported on their rows.
    let doc = serde_json::json!({ "entries": [
        { "id": "runtime/ort-cuda13-provider", "family": "runtime", "kind": "runtime", "version": "1.28.0",
          "license": "MIT", "status": "unpublished", "variant": "cuda13",
          "files": [{ "path": "onnxruntime_providers_cuda.dll", "size": 1, "sources": [{ "kind": "direct", "url": "https://example.invalid/x" }] }] },
        { "id": "runtime/nvidia-cuda13", "family": "runtime", "kind": "runtime", "version": "1",
          "license": "x", "status": "published", "variant": "cuda13",
          "files": [{ "path": "a.zip", "size": 1, "sources": [{ "kind": "nvidia-redist", "url": "https://example.invalid/a.zip" }] }] }
    ]});
    let c2 = Catalog::from_json(&doc.to_string()).unwrap();
    let everything = |_: &str| Some(root.clone());
    let bad_provider = |_: &Path| Err("provider mismatch".to_string());
    let gpu_pc = VariantSupport {
        cuda13: true,
        vulkan: true,
        avx512: false,
        avx2: true,
    };
    let rows = pack_rows(
        &c2,
        &RowInputs {
            installed: &everything,
            support: gpu_pc,
            check_provider: &bad_provider,
            nvidia_error: Some("cannot load cudnn64_9.dll".into()),
        },
    );
    assert_eq!(rows[0].error.as_deref(), Some("provider mismatch"));
    assert_eq!(rows[1].error.as_deref(), Some("cannot load cudnn64_9.dll"));

    // The NVIDIA runtime and the CUDA provider need the GPU, not AVX2 (ONNX Runtime's CUDA EP runs
    // on any x64 CPU); a Sigil-built cuda13 engine on the same PC does need AVX2.
    let good_provider = |_: &Path| Ok(());
    let gpu_without_avx2 = VariantSupport {
        cuda13: true,
        ..VariantSupport::default()
    };
    let rows = pack_rows(
        &c2,
        &RowInputs {
            installed: &everything,
            support: gpu_without_avx2,
            check_provider: &good_provider,
            nvidia_error: None,
        },
    );
    assert!(rows.iter().all(|r| r.loadable && r.error.is_none()), "{rows:?}");
    let rows = pack_rows(
        &c2,
        &RowInputs {
            installed: &everything,
            support: VariantSupport {
                avx2: true,
                ..VariantSupport::default()
            },
            check_provider: &good_provider,
            nvidia_error: None,
        },
    );
    assert!(rows.iter().all(|r| !r.loadable && r.error.as_deref().unwrap().contains("R580")), "{rows:?}");
    let with_nvidia = |id: &str| match id {
        "runtime/sigil-echo-cuda13" | "runtime/nvidia-cuda13" => Some(root.clone()),
        _ => None,
    };
    let rows = pack_rows(
        &c,
        &RowInputs {
            installed: &with_nvidia,
            support: gpu_without_avx2,
            check_provider: &good_provider,
            nvidia_error: None,
        },
    );
    let engine = rows.iter().find(|r| r.id == "runtime/sigil-echo-cuda13").unwrap();
    assert!(engine.error.as_deref().unwrap().contains("AVX2"), "{engine:?}");
    assert!(rows.iter().find(|r| r.id == "runtime/nvidia-cuda13").unwrap().loadable);
    let _ = std::fs::remove_dir_all(&root);
}

/// The IPC body on this PC (read-only: the embedded catalog and whatever app data exists; no pack
/// is loaded): one row per runtime entry, in catalog order, and leftovers flagged as legacy.
#[test]
fn live_rows_cover_every_runtime_entry() {
    let rows = runtime_pack_statuses().expect("rows");
    let catalog = Catalog::embedded().unwrap();
    let runtime_ids: Vec<&str> = catalog
        .entries
        .iter()
        .filter(|e| e.kind == crate::services::model_store::EntryKind::Runtime)
        .map(|e| e.id.as_str())
        .collect();
    let row_ids: Vec<&str> = rows.iter().filter(|r| !r.legacy).map(|r| r.id.as_str()).collect();
    assert_eq!(row_ids, runtime_ids);
    for r in &rows {
        assert_eq!(r.loadable, r.error.is_none(), "{r:?}");
        assert!(!r.legacy || r.id.starts_with("legacy/"), "{r:?}");
        if r.dir.is_none() && !r.legacy {
            assert!(r.error.as_deref().unwrap().starts_with("not installed"), "{r:?}");
        }
    }
}

#[test]
fn row_json_matches_the_ts_contract() {
    let row = RuntimePackStatus {
        id: "runtime/sigil-echo-cpu".into(),
        variant: Some(RuntimeVariant::CpuAvx2),
        dir: None,
        loadable: false,
        info: None,
        error: Some("not installed".into()),
        title: None,
        legacy: false,
    };
    let v = serde_json::to_value(&row).unwrap();
    for key in ["id", "variant", "dir", "loadable", "info", "error", "title", "legacy"] {
        assert!(v.get(key).is_some(), "{key}");
    }
    assert_eq!(v["variant"], "cpu-avx2");
    // Rows from before `title` / `legacy` still parse.
    let old: RuntimePackStatus =
        serde_json::from_str(r#"{"id":"x","variant":null,"dir":null,"loadable":true,"info":null,"error":null}"#).unwrap();
    assert!(!old.legacy && old.title.is_none());
}

// ---------------------------------------------------------------------------------------------
// Probe helpers
// ---------------------------------------------------------------------------------------------

#[test]
fn probe_config_and_audio_helpers() {
    let cfg = probe::build_config(Some(Path::new("C:/m/model.bin")), &[
        "gain=0.5".into(),
        "--backend=cpu".into(),
        "--config".into(),
        r#"{"threads":2,"gain":0.25}"#.into(),
        "name=\"quoted\"".into(),
    ])
    .unwrap();
    assert_eq!(cfg["model"], "C:/m/model.bin");
    assert_eq!(cfg["gain"], 0.25, "later arguments win");
    assert_eq!(cfg["backend"], "cpu", "a non-JSON value is a string");
    assert_eq!(cfg["threads"], 2);
    assert_eq!(cfg["name"], "quoted");
    assert!(probe::build_config(None, &["--config".into()]).is_err());
    assert!(probe::build_config(None, &["--config".into(), "[1]".into()]).unwrap_err().contains("object"));
    assert!(probe::build_config(None, &["oops".into()]).unwrap_err().contains("unknown extra argument"));
    assert!(probe::build_config(None, &["=1".into()]).is_err());

    let (tone, sr, ch) = probe::test_tone();
    assert_eq!((tone.len(), sr, ch), (16_000, 16_000, 1));
    assert!(tone.iter().all(|s| s.abs() <= 0.5 + 1e-6));
    assert_eq!(probe::max_abs_diff(&[1.0, 2.0], &[1.0, 2.5]), Some(0.5));
    assert_eq!(probe::max_abs_diff(&[1.0], &[]), None);

    let dir = temp_dir("wav");
    let wav = dir.join("a.wav");
    let spec = hound::WavSpec {
        channels: 2,
        sample_rate: 22_050,
        bits_per_sample: 16,
        sample_format: hound::SampleFormat::Int,
    };
    let mut w = hound::WavWriter::create(&wav, spec).unwrap();
    for s in [16384i16, -16384, 0, 32767] {
        w.write_sample(s).unwrap();
    }
    w.finalize().unwrap();
    let (samples, sr, ch) = probe::read_wav(&wav).unwrap();
    assert_eq!((sr, ch), (22_050, 2));
    assert_eq!(samples.len(), 4);
    assert!((samples[0] - 0.5).abs() < 1e-6 && (samples[1] + 0.5).abs() < 1e-6);
    assert!(probe::read_wav(&dir.join("missing.wav")).is_err());
    let _ = std::fs::remove_dir_all(&dir);
}

#[test]
fn headless_probe_errors_never_panic() {
    use crate::services::speech_rt::EngineProbeArgs;
    let args = |engine: &str, variant: Option<&str>, dir: Option<PathBuf>| EngineProbeArgs {
        engine: engine.into(),
        variant: variant.map(str::to_string),
        runtime_dir: dir,
        ..EngineProbeArgs::default()
    };
    assert!(headless_probe(&args("echo", Some("gpu"), None)).unwrap_err().contains("unknown variant"));
    let missing = std::env::temp_dir().join("sigil-native-runtime-no-such-dir");
    assert!(headless_probe(&args("echo", None, Some(missing))).unwrap_err().contains("is not a folder"));
    assert!(headless_probe(&args("Bad Name", None, Some(std::env::temp_dir()))).unwrap_err().contains("invalid engine name"));
}

// ---------------------------------------------------------------------------------------------
// A real engine DLL (manual: build it first)
// ---------------------------------------------------------------------------------------------

/// End to end against a real `sigil_echo.dll`: hash-verified load, ABI and info, every call,
/// streams, cross-thread cancel, create failures with their status, a tampered pack refused, the
/// variant fallback when a backend is unavailable, and the headless probe report.
#[test]
#[ignore = "needs a built echo pack: SIGIL_TEST_ECHO_DIR=native/out/dev/echo/<variant>"]
fn real_echo_engine_end_to_end() {
    let Some(dir) = std::env::var_os("SIGIL_TEST_ECHO_DIR").map(PathBuf::from) else {
        eprintln!("SIGIL_TEST_ECHO_DIR is not set; skipped");
        return;
    };
    let dir = std::fs::canonicalize(&dir).expect("SIGIL_TEST_ECHO_DIR exists");
    let lib = EngineLibrary::load(&dir, "sigil_echo.dll").expect("the echo pack loads");
    let info = lib.info().clone();
    eprintln!("{info:?}");
    assert_eq!((info.name.as_str(), info.kind.as_str(), info.abi_version), ("echo", "test", 1));
    assert_eq!(info.capability_names(), ["process", "stream", "render", "cancel"]);
    let again = EngineLibrary::load(&dir, "sigil_echo.dll").unwrap();
    assert!(std::ptr::eq(lib.info(), again.info()), "one module per path");

    let mut e = lib.create(r#"{"gain":0.5}"#).unwrap();
    let q: serde_json::Value = serde_json::from_str(&e.query(r#"{"op":"info"}"#).unwrap()).unwrap();
    assert_eq!((q["name"].as_str(), q["gain"].as_f64()), (Some("echo"), Some(0.5)));
    let audio: Vec<f32> = (0..960).map(|i| ((i as f32) * 0.01).sin() * 0.5).collect();
    let view = AudioView {
        samples: &audio,
        sample_rate: 48_000,
        channels: 2,
    };
    let stats: serde_json::Value = serde_json::from_str(&e.process(view, "{}").unwrap()).unwrap();
    assert_eq!(stats["frames"], 480);
    let out = e.render(Some(view), "{}").unwrap();
    assert_eq!((out.sample_rate, out.channels, out.samples.len()), (48_000, 2, 960));
    assert!(out.samples.iter().zip(&audio).all(|(o, i)| (o - i * 0.5).abs() < 1e-7));
    let tone = e.render(None, r#"{"freq":1000,"ms":10,"sample_rate":16000}"#).unwrap();
    assert_eq!(tone.frames(), 160);
    let err = e.query_engine(r#"{"op":"nope"}"#).unwrap_err();
    assert_eq!(err.status, SIGIL_E_INVALID_ARGUMENT);
    assert!(err.message.contains("unknown query op"), "{}", err.message);
    let err = e.render_audio(Some(view), r#"{"gain":"loud"}"#).unwrap_err();
    assert_eq!(err.status, SIGIL_E_INVALID_ARGUMENT, "{}", err.message);

    let mut s = e.open_stream("{}").unwrap();
    for chunk in audio.chunks(320) {
        assert!(s.push(AudioView { samples: chunk, sample_rate: 48_000, channels: 2 }).unwrap().is_some());
    }
    let fin: serde_json::Value = serde_json::from_str(&s.finish().unwrap()).unwrap();
    assert_eq!(fin["frames"], 480);

    let canceller = e.canceller();
    let t = std::thread::spawn(move || {
        std::thread::sleep(std::time::Duration::from_millis(50));
        canceller.cancel();
    });
    let err = e.query_engine(r#"{"op":"sleep","ms":10000}"#).unwrap_err();
    t.join().unwrap();
    assert!(err.is_cancelled(), "{err:?}");
    drop(s);
    drop(e);

    let err = lib.create_engine(r#"{"fail":"backend"}"#).unwrap_err();
    assert_eq!(err.status, SIGIL_E_BACKEND_UNAVAILABLE);
    let err = lib.create_engine(r#"{"gain":100}"#).unwrap_err();
    assert_eq!(err.status, SIGIL_E_INVALID_ARGUMENT);
    assert!(lib.create_engine("[1,2]").unwrap_err().message.contains("JSON object"));

    // A tampered copy of the pack is refused before loading.
    let tampered = temp_dir("echo-tampered");
    std::fs::copy(dir.join("sigil_echo.dll"), tampered.join("sigil_echo.dll")).unwrap();
    let mut m = PackManifest::read(&dir).unwrap().expect("the build writes sigil-pack.json");
    m.files[0].sha256 = "0".repeat(64);
    std::fs::write(tampered.join(PACK_MANIFEST), serde_json::to_vec(&m).unwrap()).unwrap();
    assert!(EngineLibrary::load(&tampered, "sigil_echo.dll").unwrap_err().contains("does not match its pack manifest"));
    // A manifest claiming another ABI is refused too.
    m.files[0].sha256 = pack::sha256_cached(&dir.join("sigil_echo.dll")).unwrap();
    m.abi_version = 2;
    std::fs::write(tampered.join(PACK_MANIFEST), serde_json::to_vec(&m).unwrap()).unwrap();
    assert!(EngineLibrary::load(&tampered, "sigil_echo.dll").unwrap_err().contains("ABI v2"));
    let _ = std::fs::remove_dir_all(&tampered);

    // open_engine: a backend failure at create is reported (no other variant to fall back to).
    let req = OpenRequest {
        runtime_dir: Some(dir.clone()),
        config_json: Some(r#"{"fail":"backend"}"#.into()),
        ..OpenRequest::default()
    };
    let err = open_engine("echo", &req).unwrap_err();
    assert!(err.contains("SIGIL_E_BACKEND_UNAVAILABLE"), "{err}");
    let req = OpenRequest {
        runtime_dir: Some(dir.clone()),
        config_json: Some("{}".into()),
        ..OpenRequest::default()
    };
    let opened = open_engine("echo", &req).unwrap();
    assert!(opened.instance.is_some() && opened.attempts.is_empty());

    // The headless probe runs every stage and reports success.
    let report = headless_probe(&crate::services::speech_rt::EngineProbeArgs {
        engine: "echo".into(),
        runtime_dir: Some(dir.clone()),
        extra: vec!["gain=0.5".into()],
        ..Default::default()
    })
    .expect("probe passes");
    let r: serde_json::Value = serde_json::from_str(&report).unwrap();
    eprintln!("{report}");
    assert_eq!(r["ok"], true);
    assert_eq!(r["render"]["echo_check"]["pass"], true);
    assert_eq!(r["cancel"]["pass"], true);
    assert!(r["stream"]["partials"].as_u64().unwrap() > 0);
}
