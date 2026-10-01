//! Shared content-encoder pool (P3-C1): ContentVec 768 / layer 12, ContentVec 256 / layer 9 +
//! final_proj, SPIN-v2, SPIN (v1; P3-I0) and zh / ja / ko HuBERT, chosen per slot from its manifest / checkpoint
//! metadata; fp16 on GPUs, int8 on VNNI CPUs (`int8_gate`), shared across hot slots (one session
//! per embedder + EP + precision, plan T3: "+<= 150 MB per extra hot voice").
//!
//! # Frozen by P3-S0 (signatures; P3-C1 implements)
//! - [`EmbedderKind`] with [`EmbedderKind::from_manifest`], [`EmbedderKind::as_str`] (the manifest
//!   / catalog name) and [`EmbedderKind::frontend`] (receptive field and hop at 16 kHz: 400 / 320
//!   for every supported encoder; P3-A verifies the CJK exports).
//! - [`ContentEncoderPool::global`], [`ContentEncoderPool::acquire`] → [`SharedEncoder`];
//!   [`ContentEncoder::encode`] writes `frames × channels` features into a caller-owned buffer
//!   (no per-tick allocation) and returns the frame count.
//!
//! # Resolution (P3-C1)
//! A slot's encoder file is found by [`resolve_encoder_file`] ([`acquire_for_slot`] loads it):
//! 1. an explicit path (the UI's HuBERT path) for the ContentVec kinds, the one encoder a user can
//!    pick by hand today, when it produces the slot's feature width (a 768-channel HuBERT set for
//!    v2 voices does not serve a v1 voice: [`acquire_for_slot`] falls back to the steps below with
//!    a warning); a SPIN / CJK voice always uses its own catalog entry;
//! 2. the model-store entry of the kind (`vc/embedder-<name>`, `vc::assets`), trying the precision
//!    variants in [`variant_order`]: `model.fp16.onnx` first on GPUs, `model.int8.onnx` first on
//!    CPUs with VNNI (`int8_gate`), then `model.fp32.onnx` and `model.onnx`;
//! 3. for ContentVec, the legacy `voice-changer/base/` files (`hubert_base.onnx`, ...) next to the
//!    slot folder, so existing setups keep working with no download.
//!
//! A GPU-precision variant whose session falls back to the CPU is rebuilt from the CPU variant.
//! Encoders run through `ort_session::build_session` (EP ladder, outcome `vc.content`) with host
//! tensors; the pool shares one session per (file, EP, device, threads) between hot slots.

use std::path::{Path, PathBuf};
use std::sync::{Arc, Mutex, OnceLock, PoisonError};

use half::f16;
use ort::session::{Session, SessionInputValue};
use ort::value::{TensorElementType, TensorRef};

use crate::services::diagnostics::EpOutcome;
use crate::services::neural_audio::ort_session::{build_session, normalize_ep, ModelSource, VcSessionOptions};
use crate::services::neural_audio::vc::assets;
use crate::services::neural_audio::vc::int8_gate;

use super::options::PrecisionMode;

/// A content encoder family (the `embedder` of a voice).
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum EmbedderKind {
    /// ContentVec / HuBERT base, 768 channels from layer 12 (RVC v2).
    ContentVec768,
    /// ContentVec / HuBERT base, layer 9 through `final_proj` to 256 channels (RVC v1).
    ContentVec256,
    /// SPIN-v2 (Applio).
    SpinV2,
    /// Applio's first SPIN checkpoint (another network than SPIN-v2; `vc/embedder-spin`, P3-I0).
    Spin,
    HubertJa,
    HubertKo,
    HubertZh,
}

impl EmbedderKind {
    pub const ALL: [EmbedderKind; 7] = [
        EmbedderKind::ContentVec768,
        EmbedderKind::ContentVec256,
        EmbedderKind::SpinV2,
        EmbedderKind::Spin,
        EmbedderKind::HubertJa,
        EmbedderKind::HubertKo,
        EmbedderKind::HubertZh,
    ];

    /// The manifest / catalog name.
    pub fn as_str(self) -> &'static str {
        match self {
            EmbedderKind::ContentVec768 => "contentvec",
            EmbedderKind::ContentVec256 => "contentvec-256",
            EmbedderKind::SpinV2 => "spin-v2",
            EmbedderKind::Spin => "spin",
            EmbedderKind::HubertJa => "hubert-ja",
            EmbedderKind::HubertKo => "hubert-ko",
            EmbedderKind::HubertZh => "hubert-zh",
        }
    }

    /// Feature channels the encoder produces.
    pub fn channels(self) -> usize {
        match self {
            EmbedderKind::ContentVec256 => 256,
            _ => 768,
        }
    }

    /// Model-store id of the encoder (`vc::assets`).
    pub fn catalog_id(self) -> &'static str {
        assets::embedder_id(self.as_str()).unwrap_or(assets::EMBEDDER_CONTENTVEC)
    }

    /// The embedder a slot needs: its manifest `embedder` name, else ContentVec by channel count.
    ///
    /// Accepts the manifest names ([`as_str`](Self::as_str)) and the names RVC / Applio
    /// checkpoints record (`embedder_model`: `contentvec`, `hubert_base`, `spin`,
    /// `japanese-hubert-base`, `chinese-hubert-base`, `korean-hubert-base`, ...). The weights'
    /// channel count is authoritative for ContentVec (a v1 voice named `contentvec` uses the
    /// 256-channel encoder); the other encoders exist only with 768 channels.
    pub fn from_manifest(embedder: Option<&str>, emb_channels: u32) -> Result<EmbedderKind, String> {
        let by_channels = || match emb_channels {
            768 => Ok(EmbedderKind::ContentVec768),
            256 => Ok(EmbedderKind::ContentVec256),
            c => Err(format!("no content encoder produces {c}-channel features (RVC v1 uses 256, v2 768)")),
        };
        let Some(raw) = embedder.map(str::trim).filter(|s| !s.is_empty()) else {
            return by_channels();
        };
        let name = raw.to_ascii_lowercase().replace(['_', ' '], "-");
        let kind = match name.as_str() {
            "contentvec" | "content-vec" | "hubert" | "hubert-base" | "contentvec-768" | "vec-768-layer-12" | "contentvec-base" => {
                return by_channels();
            }
            "contentvec-256" | "vec-256-layer-9" => EmbedderKind::ContentVec256,
            "spin-v2" | "spinv2" => EmbedderKind::SpinV2,
            // Applio's first SPIN is another checkpoint than SPIN-v2 (its own catalog entry).
            "spin" | "spin-v1" | "spinv1" => EmbedderKind::Spin,
            "hubert-ja" | "japanese-hubert-base" | "rinna-hubert-base" | "japanese-hubert" => EmbedderKind::HubertJa,
            "hubert-ko" | "korean-hubert-base" | "korean-hubert" => EmbedderKind::HubertKo,
            "hubert-zh" | "chinese-hubert-base" | "chinese-hubert" => EmbedderKind::HubertZh,
            other => return Err(format!("the voice was trained with the content encoder '{other}', which Sigil does not have")),
        };
        if kind.channels() as u32 != emb_channels {
            return Err(format!(
                "the voice's {emb_channels}-channel features do not match its content encoder {} ({} channels)",
                kind.as_str(),
                kind.channels()
            ));
        }
        Ok(kind)
    }

    /// `(receptive field, hop)` of the encoder's conv front end at 16 kHz.
    pub fn frontend(self) -> (usize, usize) {
        (400, 320)
    }
}

/// Model files to try on a device class, most preferred first: a variant name (`fp16` / `fp32` /
/// `int8`, file `model.<variant>.onnx`) or `None` (`model.onnx`).
pub fn variant_order(gpu: bool, precision: PrecisionMode, int8_ok: bool) -> Vec<Option<&'static str>> {
    match (gpu, precision) {
        (true, PrecisionMode::Fp32) => vec![Some("fp32"), None],
        (true, _) => vec![Some("fp16"), Some("fp32"), None],
        // CPU fp16 is ONNX Runtime's slow emulation: an explicit fp16 request runs fp32 there.
        (false, PrecisionMode::Auto) if int8_ok => vec![Some("int8"), Some("fp32"), None],
        (false, _) => vec![Some("fp32"), None],
    }
}

/// Legacy `voice-changer/base/` file names of the ContentVec encoders.
const LEGACY_768: &[&str] = &["hubert_base.onnx", "contentvec.onnx", "vec-768-layer-12.onnx", "hubert.onnx"];
const LEGACY_256: &[&str] = &["vec-256-layer-9.onnx", "hubert_base_256.onnx"];

/// Where an encoder file came from.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ResolvedEncoder {
    pub path: PathBuf,
    /// `fp16` / `fp32` / `int8` / `model` (store entries), `explicit` or `legacy`.
    pub variant: &'static str,
    /// Store variants of the same entry still usable on the CPU (after a GPU fallback).
    pub cpu_fallback: Option<PathBuf>,
}

fn store_variant(kind: EmbedderKind, variant: Option<&'static str>) -> Option<(PathBuf, &'static str)> {
    assets::model_file(kind.catalog_id(), variant).map(|p| (p, variant.unwrap_or("model")))
}

/// Find the encoder file of `kind` (see the module docs for the order). `slot_dir` enables the
/// legacy `voice-changer/base/` fallback (`<app_data>/voice-changer/models/<slot>` → `../../base`).
pub fn resolve_encoder_file(
    kind: EmbedderKind,
    explicit: Option<&Path>,
    slot_dir: Option<&Path>,
    gpu: bool,
    precision: PrecisionMode,
) -> Result<ResolvedEncoder, String> {
    let contentvec = matches!(kind, EmbedderKind::ContentVec768 | EmbedderKind::ContentVec256);
    if let Some(p) = explicit.filter(|p| !p.as_os_str().is_empty()) {
        if contentvec {
            if !p.is_file() {
                return Err(format!("content encoder not found: {}", p.display()));
            }
            return Ok(ResolvedEncoder {
                path: p.to_path_buf(),
                variant: "explicit",
                cpu_fallback: None,
            });
        }
    }
    let int8_ok = int8_gate::int8_content_encoder_enabled();
    for v in variant_order(gpu, precision, int8_ok) {
        if let Some((path, variant)) = store_variant(kind, v) {
            let cpu_fallback = if gpu {
                variant_order(false, precision, int8_ok)
                    .into_iter()
                    .find_map(|cv| store_variant(kind, cv))
                    .map(|(p, _)| p)
                    .filter(|p| p != &path)
            } else {
                None
            };
            return Ok(ResolvedEncoder { path, variant, cpu_fallback });
        }
    }
    if contentvec {
        let names = if kind == EmbedderKind::ContentVec768 { LEGACY_768 } else { LEGACY_256 };
        let mut dirs: Vec<PathBuf> = Vec::new();
        if let Some(slot) = slot_dir {
            if let Some(vc_root) = slot.parent().and_then(Path::parent) {
                dirs.push(vc_root.join("base"));
            }
            dirs.push(slot.to_path_buf());
        }
        for dir in dirs {
            for name in names {
                let p = dir.join(name);
                if p.is_file() {
                    return Ok(ResolvedEncoder {
                        path: p,
                        variant: "legacy",
                        cpu_fallback: None,
                    });
                }
            }
        }
    }
    Err(format!(
        "the {} content encoder this voice needs is not installed: add '{}' in Models & Runtimes",
        kind.as_str(),
        kind.catalog_id()
    ))
}

/// The pooled encoder of a slot: [`resolve_encoder_file`], then
/// [`ContentEncoderPool::acquire_resolved`]. An explicit file is checked with one encode of
/// `probe_16k` samples of silence (the plan's context length, so the probe is also a warm-up at the
/// live shape): when it does not produce `kind`'s feature width, the store / legacy resolution runs
/// instead and `warnings` says so; if nothing else resolves either, the explicit file's error is
/// returned. When the explicit file wins although a faster store variant of the kind is installed
/// (fp16 on a GPU, int8 on a VNNI CPU), `warnings` says how to use it.
#[allow(clippy::too_many_arguments)]
pub fn acquire_for_slot(
    kind: EmbedderKind,
    explicit: Option<&Path>,
    slot_dir: Option<&Path>,
    ep: &str,
    opts: &VcSessionOptions,
    precision: PrecisionMode,
    probe_16k: usize,
    warnings: &mut Vec<String>,
) -> Result<SharedEncoder, String> {
    let pool = ContentEncoderPool::global();
    let gpu = normalize_ep(ep) != "cpu";
    let resolved = resolve_encoder_file(kind, explicit, slot_dir, gpu, precision)?;
    let encoder = pool.acquire_resolved(kind, &resolved, ep, opts)?;
    if resolved.variant != "explicit" {
        return Ok(encoder);
    }
    match probe_width(&encoder, probe_16k) {
        Ok(()) => {
            let int8_ok = int8_gate::int8_content_encoder_enabled();
            let faster = variant_order(gpu, precision, int8_ok)
                .into_iter()
                .next()
                .flatten()
                .filter(|v| matches!(*v, "fp16" | "int8"))
                .and_then(|v| store_variant(kind, Some(v)));
            if let Some((_, v)) = faster {
                warnings.push(format!(
                    "content encoder: the HuBERT path from the settings ({}) runs instead of the installed {v} {} encoder ({}); clear the path to use the faster one",
                    resolved.path.display(),
                    kind.as_str(),
                    kind.catalog_id()
                ));
            }
            Ok(encoder)
        }
        Err(e) => {
            drop(encoder);
            pool.release_unused();
            let fallback = resolve_encoder_file(kind, None, slot_dir, gpu, precision).and_then(|r| {
                let enc = pool.acquire_resolved(kind, &r, ep, opts)?;
                probe_width(&enc, probe_16k)?;
                Ok((enc, r))
            });
            match fallback {
                Ok((enc, r)) => {
                    warnings.push(format!(
                        "content encoder: the HuBERT path from the settings ({}) does not fit this voice ({e}); using {} instead",
                        resolved.path.display(),
                        r.path.display()
                    ));
                    Ok(enc)
                }
                Err(_) => {
                    pool.release_unused();
                    Err(e)
                }
            }
        }
    }
}

/// One encode of `len` samples of silence (at least the front end's 720): fails when the encoder
/// does not produce its kind's feature width.
fn probe_width(encoder: &SharedEncoder, len: usize) -> Result<(), String> {
    let probe = vec![0.0f32; len.max(720)];
    let mut feats = Vec::new();
    encoder
        .lock()
        .unwrap_or_else(PoisonError::into_inner)
        .encode(&probe, &mut feats)
        .map(|_| ())
}

/// One loaded encoder session (behind the pool's lock; shared by every hot slot using it).
pub struct ContentEncoder {
    kind: EmbedderKind,
    channels: usize,
    session: Session,
    input_name: String,
    /// `[N]` (1) or `[1, N]` (2).
    input_rank: usize,
    /// The graph takes / returns fp16 (converted here; fp32 on both sides is the normal case).
    half_input: bool,
    half_in: Vec<f16>,
    outcome: EpOutcome,
    path: PathBuf,
    variant: &'static str,
}

impl ContentEncoder {
    /// Build the encoder session for `path` on the EP ladder of `ep` (outcome `vc.content`).
    pub fn load(kind: EmbedderKind, path: &Path, variant: &'static str, ep: &str, opts: &VcSessionOptions) -> Result<ContentEncoder, String> {
        let (session, outcome) =
            build_session(Some("vc.content"), ep, opts, ModelSource::File(path)).map_err(|e| format!("content encoder {}: {e}", path.display()))?;
        let input = session
            .inputs()
            .first()
            .ok_or_else(|| format!("content encoder {} declares no inputs", path.display()))?;
        let input_name = input.name().to_string();
        let half_input = match input.dtype().tensor_type() {
            Some(TensorElementType::Float32) => false,
            Some(TensorElementType::Float16) => true,
            other => return Err(format!("content encoder {} takes {other:?} audio (expected float)", path.display())),
        };
        let input_rank = input.dtype().tensor_shape().map(|s| s.len()).unwrap_or(2);
        if !(1..=2).contains(&input_rank) {
            return Err(format!(
                "content encoder {} takes a rank-{input_rank} input (expected [N] or [1, N])",
                path.display()
            ));
        }
        Ok(ContentEncoder {
            kind,
            channels: kind.channels(),
            session,
            input_name,
            input_rank,
            half_input,
            half_in: Vec::new(),
            outcome,
            path: path.to_path_buf(),
            variant,
        })
    }

    pub fn kind(&self) -> EmbedderKind {
        self.kind
    }

    /// Feature channels per frame (768 or 256).
    pub fn channels(&self) -> usize {
        self.channels
    }

    /// Where the session runs.
    pub fn outcome(&self) -> &EpOutcome {
        &self.outcome
    }

    /// The model file and its variant (`fp16`, `int8`, `explicit`, ...).
    pub fn source(&self) -> (&Path, &'static str) {
        (&self.path, self.variant)
    }

    /// Encode `mono_16k` into `out` (`frames × channels`, resized by the encoder, capacity kept
    /// across calls); returns the frame count.
    pub fn encode(&mut self, mono_16k: &[f32], out: &mut Vec<f32>) -> Result<usize, String> {
        let n = mono_16k.len();
        let err = |e: ort::Error| format!("content encoder input: {e}");
        if self.half_input {
            self.half_in.clear();
            self.half_in
                .extend(mono_16k.iter().map(|&x| f16::from_f32(x)));
        }
        let input: SessionInputValue<'_> = match (self.half_input, self.input_rank) {
            (true, 1) => TensorRef::from_array_view(([n], self.half_in.as_slice()))
                .map_err(err)?
                .into(),
            (true, _) => TensorRef::from_array_view(([1, n], self.half_in.as_slice()))
                .map_err(err)?
                .into(),
            (false, 1) => TensorRef::from_array_view(([n], mono_16k))
                .map_err(err)?
                .into(),
            (false, _) => TensorRef::from_array_view(([1, n], mono_16k))
                .map_err(err)?
                .into(),
        };
        let outputs = self
            .session
            .run(ort::inputs![self.input_name.as_str() => input])
            .map_err(|e| format!("content encoder run: {e}"))?;
        let value = &outputs[0];
        let c = self.channels;
        let dims: Vec<usize>;
        let flat: Vec<f32>;
        let data: &[f32] = match value.try_extract_tensor::<f32>() {
            Ok((shape, data)) => {
                dims = shape.iter().map(|d| (*d).max(0) as usize).collect();
                data
            }
            Err(_) => {
                let (shape, data) = value
                    .try_extract_tensor::<f16>()
                    .map_err(|e| format!("content encoder output: {e}"))?;
                dims = shape.iter().map(|d| (*d).max(0) as usize).collect();
                flat = data.iter().map(|h| h.to_f32()).collect();
                &flat
            }
        };
        // Time-major [T * C]. Accepted: [1, T, C], [T, C], [1, C, T].
        match dims.as_slice() {
            [1, ch, t] if *ch == c && *t != c => {
                out.clear();
                out.resize(t * c, 0.0);
                for ti in 0..*t {
                    for ci in 0..c {
                        out[ti * c + ci] = data[ci * t + ti];
                    }
                }
                Ok(*t)
            }
            [.., ch] if *ch == c => {
                out.clear();
                out.extend_from_slice(data);
                Ok(data.len() / c)
            }
            _ => Err(format!(
                "content encoder {} returned {:?}, not {c}-channel features for a {} voice (the slot needs the '{}' encoder)",
                self.path.display(),
                dims,
                if c == 256 { "v1" } else { "v2" },
                self.kind.as_str()
            )),
        }
    }
}

/// A pooled encoder.
pub type SharedEncoder = Arc<Mutex<ContentEncoder>>;

#[derive(Clone, Debug, PartialEq, Eq)]
struct PoolKey {
    kind: EmbedderKind,
    path: PathBuf,
    ep: &'static str,
    device: i32,
    streamer: bool,
    threads: Option<usize>,
    prefix: Option<&'static str>,
}

/// Encoders shared across hot slots, keyed by kind + EP + precision.
pub struct ContentEncoderPool {
    entries: Mutex<Vec<(PoolKey, SharedEncoder)>>,
}

impl ContentEncoderPool {
    pub fn global() -> &'static ContentEncoderPool {
        static POOL: OnceLock<ContentEncoderPool> = OnceLock::new();
        POOL.get_or_init(|| ContentEncoderPool {
            entries: Mutex::new(Vec::new()),
        })
    }

    /// The pooled encoder for `kind` on `ep` (loading it on first use). `model` overrides the
    /// store-resolved file (the UI's explicit HuBERT path).
    pub fn acquire(
        &self,
        kind: EmbedderKind,
        model: Option<&Path>,
        ep: &str,
        opts: &VcSessionOptions,
        precision: PrecisionMode,
    ) -> Result<SharedEncoder, String> {
        let gpu = normalize_ep(ep) != "cpu";
        let resolved = match model {
            Some(p) => ResolvedEncoder {
                path: p.to_path_buf(),
                variant: "explicit",
                cpu_fallback: None,
            },
            None => resolve_encoder_file(kind, None, None, gpu, precision)?,
        };
        self.acquire_resolved(kind, &resolved, ep, opts)
    }

    /// [`acquire`](Self::acquire) for an already resolved file ([`resolve_encoder_file`]).
    pub fn acquire_resolved(
        &self,
        kind: EmbedderKind,
        resolved: &ResolvedEncoder,
        ep: &str,
        opts: &VcSessionOptions,
    ) -> Result<SharedEncoder, String> {
        let key = PoolKey {
            kind,
            path: resolved.path.clone(),
            ep: normalize_ep(ep),
            device: opts.cuda_device_index,
            streamer: opts.streamer_profile,
            threads: opts.intra_threads,
            prefix: opts.component_prefix,
        };
        let mut entries = self.entries.lock().unwrap_or_else(PoisonError::into_inner);
        if let Some((_, enc)) = entries.iter().find(|(k, _)| *k == key) {
            return Ok(enc.clone());
        }
        let mut encoder = ContentEncoder::load(kind, &resolved.path, resolved.variant, ep, opts)?;
        if encoder.outcome.active == "cpu" && key.ep != "cpu" && resolved.variant == "fp16" {
            // The GPU variant landed on the CPU (the EP ladder fell back): ONNX Runtime's CPU fp16
            // path is slow, so run the CPU variant of the same entry.
            if let Some(cpu) = &resolved.cpu_fallback {
                let variant = if cpu.to_string_lossy().contains(".int8.") { "int8" } else { "fp32" };
                let mut rebuilt = ContentEncoder::load(kind, cpu, variant, "cpu", opts)?;
                // Keep the original request and its fallback reason (the rebuild recorded a plain
                // CPU session).
                rebuilt.outcome = encoder.outcome.clone();
                crate::services::diagnostics::record_ep_outcome(&super::generator::component_key(opts, "vc.content"), rebuilt.outcome.clone());
                encoder = rebuilt;
            }
        }
        let shared = Arc::new(Mutex::new(encoder));
        entries.push((key, shared.clone()));
        Ok(shared)
    }

    /// Drop encoders no hot slot uses any more (frees their VRAM).
    pub fn release_unused(&self) {
        self.entries
            .lock()
            .unwrap_or_else(PoisonError::into_inner)
            .retain(|(_, enc)| Arc::strong_count(enc) > 1);
    }

    /// Encoders currently pooled.
    pub fn len(&self) -> usize {
        self.entries
            .lock()
            .unwrap_or_else(PoisonError::into_inner)
            .len()
    }

    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::services::neural_audio::vc::test_models;

    #[test]
    fn embedder_names_map_to_kinds_and_check_channels() {
        assert_eq!(EmbedderKind::from_manifest(None, 768).unwrap(), EmbedderKind::ContentVec768);
        assert_eq!(EmbedderKind::from_manifest(None, 256).unwrap(), EmbedderKind::ContentVec256);
        assert!(EmbedderKind::from_manifest(None, 512).is_err());
        // RVC / Applio names; a v1 voice named contentvec uses the 256-channel encoder.
        assert_eq!(EmbedderKind::from_manifest(Some("contentvec"), 256).unwrap(), EmbedderKind::ContentVec256);
        assert_eq!(EmbedderKind::from_manifest(Some("ContentVec"), 768).unwrap(), EmbedderKind::ContentVec768);
        assert_eq!(
            EmbedderKind::from_manifest(Some("hubert_base"), 768).unwrap(),
            EmbedderKind::ContentVec768
        );
        assert_eq!(EmbedderKind::from_manifest(Some("spin_v2"), 768).unwrap(), EmbedderKind::SpinV2);
        assert_eq!(EmbedderKind::from_manifest(Some("spin"), 768).unwrap(), EmbedderKind::Spin);
        assert_eq!(EmbedderKind::from_manifest(Some("SPIN_v1"), 768).unwrap(), EmbedderKind::Spin);
        assert!(EmbedderKind::from_manifest(Some("spin"), 256).is_err());
        assert_eq!(EmbedderKind::Spin.catalog_id(), crate::services::neural_audio::vc::assets::EMBEDDER_SPIN);
        assert_eq!(
            EmbedderKind::from_manifest(Some("japanese-hubert-base"), 768).unwrap(),
            EmbedderKind::HubertJa
        );
        assert_eq!(
            EmbedderKind::from_manifest(Some("chinese-hubert-base"), 768).unwrap(),
            EmbedderKind::HubertZh
        );
        assert_eq!(
            EmbedderKind::from_manifest(Some("korean_hubert_base"), 768).unwrap(),
            EmbedderKind::HubertKo
        );
        // Manifest names round-trip.
        for kind in EmbedderKind::ALL {
            assert_eq!(EmbedderKind::from_manifest(Some(kind.as_str()), kind.channels() as u32).unwrap(), kind);
            assert!(assets::embedder_id(kind.as_str()).is_some());
            assert_eq!(kind.frontend(), (400, 320));
        }
        // A SPIN voice with 256 channels, an unknown encoder.
        assert!(EmbedderKind::from_manifest(Some("spin-v2"), 256).is_err());
        assert!(EmbedderKind::from_manifest(Some("wavlm-large"), 768)
            .unwrap_err()
            .contains("wavlm-large"));
    }

    #[test]
    fn variants_prefer_fp16_on_gpus_and_int8_on_vnni_cpus() {
        assert_eq!(variant_order(true, PrecisionMode::Auto, true), vec![Some("fp16"), Some("fp32"), None]);
        assert_eq!(variant_order(true, PrecisionMode::Fp32, true), vec![Some("fp32"), None]);
        assert_eq!(variant_order(false, PrecisionMode::Auto, true), vec![Some("int8"), Some("fp32"), None]);
        assert_eq!(variant_order(false, PrecisionMode::Auto, false), vec![Some("fp32"), None]);
        assert_eq!(variant_order(false, PrecisionMode::Fp16, true), vec![Some("fp32"), None]);
    }

    /// A stand-in encoder (a 400/320 conv) with `channels` output channels.
    fn standin(channels: usize) -> Vec<u8> {
        use crate::services::neural_audio::test_onnx::{Attr, Dim, Graph, FLOAT};
        let w: Vec<f32> = (0..channels * 400).map(|i| ((i % 5) as f32 - 2.0) * 1e-3).collect();
        let mut g = Graph::new();
        g.input("source", FLOAT, &[Dim::Fixed(1), Dim::Named("n")])
            .output("feats", FLOAT, &[Dim::Fixed(1), Dim::Named("t"), Dim::Fixed(channels as u64)])
            .init_i64("axes1", &[1], &[1])
            .init_f32("conv_w", &[channels as u64, 1, 400], &w)
            .node("Unsqueeze", &["source", "axes1"], &["x3"], &[])
            .node(
                "Conv",
                &["x3", "conv_w"],
                &["c"],
                &[Attr::Ints("strides", vec![320]), Attr::Ints("kernel_shape", vec![400]), Attr::Int("group", 1)],
            )
            .node("Transpose", &["c"], &["feats"], &[Attr::Ints("perm", vec![0, 2, 1])]);
        g.model_bytes(13)
    }

    /// The UI's HuBERT path (a 768-channel file kept for v2 voices) does not block a v1 voice: the
    /// width probe fails, the slot falls back to its 256-channel encoder with a warning. A file of
    /// the right width is used as before, and with nothing else to fall back to the explicit
    /// file's error is the load error.
    #[test]
    fn explicit_encoder_of_the_wrong_width_falls_back() {
        let root = std::env::temp_dir().join(format!("sigil-content-width-{}", std::process::id()));
        let slot = root.join("voice-changer").join("models").join("v1");
        let base = root.join("voice-changer").join("base");
        std::fs::create_dir_all(&slot).unwrap();
        std::fs::create_dir_all(&base).unwrap();
        let explicit = root.join("hubert_768.onnx");
        std::fs::write(&explicit, standin(768)).unwrap();
        let opts = VcSessionOptions::default();
        // Nothing to fall back to: the width error.
        let mut warnings = Vec::new();
        let err = acquire_for_slot(EmbedderKind::ContentVec256, Some(&explicit), Some(&slot), "cpu", &opts, PrecisionMode::Auto, 1_600, &mut warnings)
            .err()
            .unwrap();
        assert!(err.contains("contentvec-256"), "{err}");
        // The legacy 256-channel file next to the slot.
        let v1 = base.join("vec-256-layer-9.onnx");
        std::fs::write(&v1, standin(256)).unwrap();
        let enc = acquire_for_slot(EmbedderKind::ContentVec256, Some(&explicit), Some(&slot), "cpu", &opts, PrecisionMode::Auto, 1_600, &mut warnings).unwrap();
        assert_eq!(enc.lock().unwrap().source(), (v1.as_path(), "legacy"));
        assert_eq!(enc.lock().unwrap().channels(), 256);
        assert!(warnings.iter().any(|w| w.contains("does not fit this voice")), "{warnings:?}");
        // The same explicit file still serves a v2 voice, with no warning.
        let mut warnings = Vec::new();
        let enc2 = acquire_for_slot(EmbedderKind::ContentVec768, Some(&explicit), Some(&slot), "cpu", &opts, PrecisionMode::Auto, 1_600, &mut warnings).unwrap();
        assert_eq!(enc2.lock().unwrap().source(), (explicit.as_path(), "explicit"));
        assert!(warnings.is_empty(), "{warnings:?}");
        drop((enc, enc2));
        ContentEncoderPool::global().release_unused();
        let _ = std::fs::remove_dir_all(&root);
    }

    /// Explicit files win for ContentVec, the legacy base folder next to the slot is found, and a
    /// SPIN voice never runs on the UI's ContentVec file.
    #[test]
    fn encoder_files_resolve_explicit_then_store_then_legacy() {
        let root = std::env::temp_dir().join(format!("sigil-content-resolve-{}", std::process::id()));
        let slot = root.join("voice-changer").join("models").join("v");
        let base = root.join("voice-changer").join("base");
        std::fs::create_dir_all(&slot).unwrap();
        std::fs::create_dir_all(&base).unwrap();
        let explicit = root.join("mine.onnx");
        std::fs::write(&explicit, b"x").unwrap();
        let r = resolve_encoder_file(EmbedderKind::ContentVec768, Some(&explicit), Some(&slot), true, PrecisionMode::Auto).unwrap();
        assert_eq!((r.path.as_path(), r.variant), (explicit.as_path(), "explicit"));
        assert!(resolve_encoder_file(
            EmbedderKind::ContentVec768,
            Some(&root.join("missing.onnx")),
            None,
            true,
            PrecisionMode::Auto
        )
        .is_err());
        // No store entry in tests: the legacy base file.
        std::fs::write(base.join("hubert_base.onnx"), b"x").unwrap();
        let r = resolve_encoder_file(EmbedderKind::ContentVec768, None, Some(&slot), false, PrecisionMode::Auto).unwrap();
        assert_eq!((r.path, r.variant), (base.join("hubert_base.onnx"), "legacy"));
        // SPIN ignores the explicit ContentVec file and has no legacy fallback.
        let err = resolve_encoder_file(EmbedderKind::SpinV2, Some(&explicit), Some(&slot), true, PrecisionMode::Auto).unwrap_err();
        assert!(err.contains("vc/embedder-spin-v2"), "{err}");
        let _ = std::fs::remove_dir_all(&root);
    }

    /// The stand-in encoder (a 400/320 conv to 768 channels) runs through the pool: features are
    /// time-major with the conv's frame count, the session is shared, and a mismatched channel
    /// count is refused with the encoder the slot needs.
    #[test]
    fn pooled_encoder_encodes_and_is_shared() {
        let dir = std::env::temp_dir().join(format!("sigil-content-pool-{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        let path = dir.join("enc.onnx");
        std::fs::write(&path, test_models::content_encoder_bytes()).unwrap();
        let pool = ContentEncoderPool {
            entries: Mutex::new(Vec::new()),
        };
        let opts = VcSessionOptions::default();
        let a = pool
            .acquire(EmbedderKind::ContentVec768, Some(&path), "cpu", &opts, PrecisionMode::Auto)
            .unwrap();
        let b = pool
            .acquire(EmbedderKind::ContentVec768, Some(&path), "cpu", &opts, PrecisionMode::Auto)
            .unwrap();
        assert!(Arc::ptr_eq(&a, &b));
        assert_eq!(pool.len(), 1);
        let audio: Vec<f32> = (0..16_000).map(|i| (i as f32 * 0.05).sin() * 0.3).collect();
        let mut feats = Vec::with_capacity(8);
        let frames = a.lock().unwrap().encode(&audio, &mut feats).unwrap();
        assert_eq!(frames, (16_000 - 400) / 320 + 1);
        assert_eq!(feats.len(), frames * 768);
        assert_eq!(a.lock().unwrap().outcome().active, "cpu");
        // The v1 encoder kind on a 768-channel file: refused, naming the encoder the slot needs.
        let v1 = pool
            .acquire(EmbedderKind::ContentVec256, Some(&path), "cpu", &opts, PrecisionMode::Auto)
            .unwrap();
        let err = v1.lock().unwrap().encode(&audio, &mut feats).unwrap_err();
        assert!(err.contains("contentvec-256"), "{err}");
        drop((a, b, v1));
        pool.release_unused();
        assert!(pool.is_empty());
        let _ = std::fs::remove_dir_all(&dir);
    }
}
