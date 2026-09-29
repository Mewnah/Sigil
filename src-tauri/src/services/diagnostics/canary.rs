//! GPU execution-provider canary: a 1-D `Conv` graph run once per EP and device before the first
//! real session (and by the deep probe), so a GPU EP that registers but cannot actually execute
//! kernels (missing cuDNN sub-library, too-old driver, broken DirectML device) is caught up front
//! and the session falls back with a precise reason instead of failing mid-stream.
//!
//! The ONNX model is encoded here by hand (a few hundred bytes of protobuf), so there is no file
//! to ship. `Y = Conv(X, W)` with `X = 0..16` and `W = [0.25, 0.5, 0.25]` must give `Y = 1..15`
//! (length 14); the output is checked, so an EP that "runs" but computes garbage also fails.
//!
//! # Caching
//! Results are cached per (EP, device) so the several sessions of one voice-changer start pay for
//! the canary once:
//! - a pass, and a failure that cannot heal while Sigil runs (a DLL is missing, no such device, the
//!   driver is too old), are kept until [`invalidate`] (runtime pack installed / removed) or
//!   [`forget_failures`] (an explicit user retry: "Test GPU acceleration", a voice-changer start);
//! - any other failure (out of memory while a game holds the VRAM, a DirectML device removed by a
//!   driver reset / TDR, a busy GPU, anything unrecognised) is reused for only
//!   [`TRANSIENT_FAILURE_TTL`], then the canary runs again.

use std::collections::HashMap;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Mutex, OnceLock, PoisonError};
use std::time::{Duration, Instant};

use ort::value::Tensor;

use crate::services::neural_audio::ort_session;

/// Input length of the canary signal.
pub const CANARY_IN: usize = 16;
/// Output length (valid 3-tap convolution).
pub const CANARY_OUT: usize = CANARY_IN - 2;
const KERNEL: [f32; 3] = [0.25, 0.5, 0.25];

// ---- minimal protobuf writer ----------------------------------------------------------------

fn varint(out: &mut Vec<u8>, mut v: u64) {
    while v >= 0x80 {
        out.push((v as u8) | 0x80);
        v >>= 7;
    }
    out.push(v as u8);
}

fn key(out: &mut Vec<u8>, field: u32, wire: u8) {
    varint(out, (u64::from(field) << 3) | u64::from(wire));
}

fn field_varint(out: &mut Vec<u8>, field: u32, v: u64) {
    key(out, field, 0);
    varint(out, v);
}

fn field_bytes(out: &mut Vec<u8>, field: u32, bytes: &[u8]) {
    key(out, field, 2);
    varint(out, bytes.len() as u64);
    out.extend_from_slice(bytes);
}

/// `TypeProto { tensor_type { elem_type: FLOAT, shape { dim { dim_value } ... } } }`.
fn float_tensor_type(dims: &[u64]) -> Vec<u8> {
    let mut shape = Vec::new();
    for &d in dims {
        let mut dim = Vec::new();
        field_varint(&mut dim, 1, d); // Dimension.dim_value
        field_bytes(&mut shape, 1, &dim); // TensorShapeProto.dim
    }
    let mut tensor = Vec::new();
    field_varint(&mut tensor, 1, 1); // elem_type = FLOAT
    field_bytes(&mut tensor, 2, &shape); // shape
    let mut ty = Vec::new();
    field_bytes(&mut ty, 1, &tensor); // TypeProto.tensor_type
    ty
}

fn value_info(name: &str, dims: &[u64]) -> Vec<u8> {
    let mut v = Vec::new();
    field_bytes(&mut v, 1, name.as_bytes()); // name
    field_bytes(&mut v, 2, &float_tensor_type(dims)); // type
    v
}

fn build_model() -> Vec<u8> {
    let mut node = Vec::new();
    field_bytes(&mut node, 1, b"X"); // input
    field_bytes(&mut node, 1, b"W"); // input
    field_bytes(&mut node, 2, b"Y"); // output
    field_bytes(&mut node, 3, b"canary_conv"); // name
    field_bytes(&mut node, 4, b"Conv"); // op_type

    let mut weight = Vec::new();
    let mut dims = Vec::new();
    for d in [1u64, 1, KERNEL.len() as u64] {
        varint(&mut dims, d);
    }
    field_bytes(&mut weight, 1, &dims); // dims (packed)
    field_varint(&mut weight, 2, 1); // data_type = FLOAT
    field_bytes(&mut weight, 8, b"W"); // name
    let raw: Vec<u8> = KERNEL.iter().flat_map(|k| k.to_le_bytes()).collect();
    field_bytes(&mut weight, 9, &raw); // raw_data

    let mut graph = Vec::new();
    field_bytes(&mut graph, 1, &node); // node
    field_bytes(&mut graph, 2, b"sigil_ep_canary"); // name
    field_bytes(&mut graph, 5, &weight); // initializer
    field_bytes(&mut graph, 11, &value_info("X", &[1, 1, CANARY_IN as u64])); // input
    field_bytes(&mut graph, 12, &value_info("Y", &[1, 1, CANARY_OUT as u64])); // output

    let mut opset = Vec::new();
    field_varint(&mut opset, 2, 13); // default domain, opset 13

    let mut model = Vec::new();
    field_varint(&mut model, 1, 7); // ir_version
    field_bytes(&mut model, 2, b"sigil"); // producer_name
    field_bytes(&mut model, 7, &graph); // graph
    field_bytes(&mut model, 8, &opset); // opset_import
    model
}

/// The serialized canary model.
pub fn model_bytes() -> &'static [u8] {
    static MODEL: OnceLock<Vec<u8>> = OnceLock::new();
    MODEL.get_or_init(build_model)
}

/// Expected output: `Y[i] = 0.25 i + 0.5 (i + 1) + 0.25 (i + 2) = i + 1`.
pub fn expected_output() -> [f32; CANARY_OUT] {
    std::array::from_fn(|i| i as f32 + 1.0)
}

/// Build, run and check the canary on `ep` / `device_id` (uncached). Returns the wall time in ms.
pub fn run(ep: &str, device_id: i32) -> Result<u64, String> {
    let started = Instant::now();
    // One intra/inter thread: the canary proves the EP, it must not spin up a full thread pool.
    let opts = ort_session::VcSessionOptions {
        streamer_profile: true,
        ..ort_session::VcSessionOptions::default()
    };
    let mut builder = ort_session::builder_for_ep(ep, device_id, &opts)?;
    let mut session = builder
        .commit_from_memory(model_bytes())
        .map_err(|e| ort_session::explain_ep_error(ep, &e.to_string()))?;
    let input: Vec<f32> = (0..CANARY_IN).map(|i| i as f32).collect();
    let x = Tensor::from_array(([1usize, 1, CANARY_IN], input)).map_err(|e| format!("canary input: {e}"))?;
    let outputs = session
        .run(ort::inputs!["X" => &x])
        .map_err(|e| ort_session::explain_ep_error(ep, &format!("test inference failed: {e}")))?;
    let (_, y) = outputs["Y"]
        .try_extract_tensor::<f32>()
        .map_err(|e| format!("canary output: {e}"))?;
    let expected = expected_output();
    if y.len() != CANARY_OUT || y.iter().zip(expected.iter()).any(|(a, b)| !a.is_finite() || (a - b).abs() > 1e-3) {
        return Err(format!(
            "{} returned wrong results for a test convolution (got {:?})",
            ort_session::ep_label(ep),
            &y[..y.len().min(4)]
        ));
    }
    Ok(started.elapsed().as_millis() as u64)
}

static GENERATION: AtomicU64 = AtomicU64::new(0);

/// How long a transient canary failure is reused before the canary runs again (long enough to
/// cover the few sessions one voice-changer start builds, short enough that closing the game that
/// held the VRAM and restarting the voice changer retries the GPU).
pub const TRANSIENT_FAILURE_TTL: Duration = Duration::from_secs(30);

/// Failures that can clear up while Sigil keeps running (checked before [`PERMANENT_MARKERS`]).
const TRANSIENT_MARKERS: &[&str] = &[
    "out of memory",
    "cuda failure 2:",
    "cudaerrormemoryallocation",
    "8007000e", // E_OUTOFMEMORY
    "887a0005", // DXGI_ERROR_DEVICE_REMOVED
    "887a0006", // DXGI_ERROR_DEVICE_HUNG
    "887a0007", // DXGI_ERROR_DEVICE_RESET
    "887a0020", // DXGI_ERROR_DRIVER_INTERNAL_ERROR
    "device removed",
    "device_removed",
    "device hung",
    "device reset",
    "busy",
    "cuda failure 46:", // cudaErrorDevicesUnavailable (exclusive-process mode, another app)
    "timeout",
    "timed out",
];

/// Failures that stay until something is installed, updated or restarted.
const PERMANENT_MARKERS: &[&str] = &[
    "error 126", // module not found
    "error 127", // procedure not found
    "which is missing",
    "was not found",
    "were not found",
    "could not be found",
    "cannot load",
    "is not installed",
    "driver is too old",
    "driver version is insufficient",
    "cuda failure 35:",  // cudaErrorInsufficientDriver
    "cuda failure 100:", // cudaErrorNoDevice
    "cuda failure 101:", // cudaErrorInvalidDevice
    "no cuda-capable device",
    "invalid device ordinal",
    "887a0002", // DXGI_ERROR_NOT_FOUND: no adapter with that index
    "887a0004", // DXGI_ERROR_UNSUPPORTED
    "returned wrong results",
];

/// Whether a canary failure may clear up while Sigil runs (and so must not stick in the cache).
/// Unrecognised failures count as transient: re-running a canary costs well under a second, a
/// wrongly sticky failure costs the GPU until Sigil restarts.
pub fn is_transient_failure(error: &str) -> bool {
    let lower = error.to_ascii_lowercase();
    TRANSIENT_MARKERS.iter().any(|m| lower.contains(m)) || !PERMANENT_MARKERS.iter().any(|m| lower.contains(m))
}

type CacheKey = (String, i32, u64);

#[derive(Clone, Debug)]
struct CachedRun {
    result: Result<u64, String>,
    at: Instant,
    transient: bool,
}

impl CachedRun {
    fn new(result: Result<u64, String>, at: Instant) -> Self {
        let transient = result.as_ref().err().is_some_and(|e| is_transient_failure(e));
        Self { result, at, transient }
    }

    /// Whether this result may still be returned at `now` instead of running the canary again.
    fn is_fresh(&self, now: Instant) -> bool {
        !self.transient || now.saturating_duration_since(self.at) < TRANSIENT_FAILURE_TTL
    }
}

fn cache() -> &'static Mutex<HashMap<CacheKey, CachedRun>> {
    static CACHE: OnceLock<Mutex<HashMap<CacheKey, CachedRun>>> = OnceLock::new();
    CACHE.get_or_init(|| Mutex::new(HashMap::new()))
}

/// [`run`], cached per (EP, device) as described in the module docs. Concurrent callers for the
/// same key may both run it the first time (harmless; the canary is cheap next to a real model load).
pub fn run_cached(ep: &str, device_id: i32) -> Result<u64, String> {
    let key = (ep.to_string(), device_id, GENERATION.load(Ordering::Acquire));
    {
        let cache = cache().lock().unwrap_or_else(PoisonError::into_inner);
        if let Some(hit) = cache.get(&key).filter(|c| c.is_fresh(Instant::now())) {
            return hit.result.clone();
        }
    }
    let result = run(ep, device_id);
    let entry = CachedRun::new(result.clone(), Instant::now());
    match &result {
        Ok(ms) => tracing::info!(target: "sigil::ort", "{} canary passed on device {device_id} in {ms} ms", ort_session::ep_label(ep)),
        Err(e) => tracing::warn!(
            target: "sigil::ort",
            "{} canary failed on device {device_id} ({}): {e}",
            ort_session::ep_label(ep),
            if entry.transient {
                format!("tried again after {} s", TRANSIENT_FAILURE_TTL.as_secs())
            } else {
                "kept until a GPU test or a runtime change".to_string()
            }
        ),
    }
    cache().lock().unwrap_or_else(PoisonError::into_inner).insert(key, entry);
    result
}

/// Forget cached canary results (a runtime pack was installed or removed).
pub fn invalidate() {
    GENERATION.fetch_add(1, Ordering::AcqRel);
    cache().lock().unwrap_or_else(PoisonError::into_inner).clear();
}

/// Forget cached failures but keep passes: the user explicitly asked to try the GPU again ("Test
/// GPU acceleration", a voice-changer start), so the next session re-runs every failed canary.
pub fn forget_failures() {
    cache()
        .lock()
        .unwrap_or_else(PoisonError::into_inner)
        .retain(|_, c| c.result.is_ok());
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn varint_encoding() {
        let mut v = Vec::new();
        varint(&mut v, 1);
        varint(&mut v, 300);
        assert_eq!(v, vec![0x01, 0xAC, 0x02]);
    }

    #[test]
    fn canary_model_loads_and_computes_on_cpu() {
        let ms = run("cpu", 0).expect("the canary model must load and compute on the CPU EP");
        assert!(ms < 60_000);
        // Deterministic bytes: the model is built once and never changes.
        assert_eq!(model_bytes(), build_model().as_slice());
    }

    #[test]
    fn cached_results_follow_invalidation() {
        assert!(run_cached("cpu", 0).is_ok());
        invalidate();
        assert!(run_cached("cpu", 0).is_ok());
    }

    #[test]
    fn failures_are_classified_transient_or_permanent() {
        // Real messages (after `explain_ep_error`): these can heal while Sigil runs.
        for transient in [
            "the GPU is out of memory (CUDA failure 2: out of memory ; GPU=0 ; hostname=H2 ; expr=cudaMalloc)",
            "test inference failed: DirectML: 887A0005 The GPU device instance has been suspended",
            "the GPU driver reset or the GPU was removed (DXGI_ERROR_DEVICE_REMOVED)",
            "CUDA failure 46: all CUDA-capable devices are busy or unavailable",
            "something nobody has seen before",
        ] {
            assert!(is_transient_failure(transient), "{transient}");
        }
        // These need an install, a driver update or another device index.
        for permanent in [
            "cudnn64_9.dll (CUDA 13 / cuDNN 9 runtime) was not found; install the NVIDIA GPU runtime pack in Settings (Error loading ...)",
            "the NVIDIA driver is too old for CUDA 13 (it needs driver R580 or newer); update the GPU driver (CUDA failure 35: CUDA driver version is insufficient)",
            "no CUDA-capable NVIDIA GPU was found (CUDA failure 100: no CUDA-capable device is detected)",
            "dml_provider_factory.cc(123): Exception(1) 887A0002 The object was not found.",
            "DirectML returned wrong results for a test convolution (got [0.0])",
        ] {
            assert!(!is_transient_failure(permanent), "{permanent}");
        }
        // Out of memory wins over words that would otherwise make a failure sticky.
        assert!(is_transient_failure("cannot load the model: out of memory"));
    }

    #[test]
    fn only_passes_and_permanent_failures_stay_cached() {
        let t0 = Instant::now();
        let later = t0 + TRANSIENT_FAILURE_TTL + Duration::from_secs(1);
        let pass = CachedRun::new(Ok(12), t0);
        assert!(pass.is_fresh(t0) && pass.is_fresh(later));
        let missing_dll = CachedRun::new(Err("cudnn64_9.dll (CUDA 13 / cuDNN 9 runtime) was not found".into()), t0);
        assert!(!missing_dll.transient && missing_dll.is_fresh(later));
        let oom = CachedRun::new(Err("the GPU is out of memory (CUDA failure 2: out of memory)".into()), t0);
        assert!(oom.transient);
        assert!(oom.is_fresh(t0 + Duration::from_secs(1)), "one voice-changer start reuses the failure");
        assert!(!oom.is_fresh(later), "a later start runs the canary again");
    }

    #[test]
    fn forgetting_failures_keeps_passes() {
        let generation = GENERATION.load(Ordering::Acquire);
        let fail_key = ("test-ep-forget".to_string(), 9_001, generation);
        let pass_key = ("test-ep-forget".to_string(), 9_002, generation);
        {
            let mut c = cache().lock().unwrap_or_else(PoisonError::into_inner);
            c.insert(fail_key.clone(), CachedRun::new(Err("x.dll was not found".into()), Instant::now()));
            c.insert(pass_key.clone(), CachedRun::new(Ok(1), Instant::now()));
        }
        forget_failures();
        let c = cache().lock().unwrap_or_else(PoisonError::into_inner);
        assert!(!c.contains_key(&fail_key), "a user retry re-runs a failed canary");
        // (A concurrent `invalidate()` from another test may have dropped the pass as well.)
        assert!(c.get(&pass_key).is_none_or(|p| p.result.is_ok()));
    }
}
