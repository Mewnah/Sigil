//! The MeanVC2 speaker embedding: what a voice converts toward.
//!
//! MeanVC2 conditions on a 256-value speaker vector from WavLM-Large plus a fine-tuned ECAPA-TDNN
//! (upstream `runtime/src/speaker.py`, the raw, unnormalised output of `extract_embedding`). It is
//! computed once per reference clip, off the real-time path, and comes from (first match wins):
//!
//! 1. **`speaker.npy` in the voice folder**: a pre-extracted embedding (upstream
//!    `preprocess/extract_spk_emb.py` writes exactly this; upstream's `*_npy.exe` builds take the
//!    same file). Backend [`NPY_BACKEND`]; the reference voice then only names the voice.
//! 2. **The voice library's cache** for the reference clip (`voices/<id>/cache/<backend>.bin`),
//!    through `families::reference::cached_embedding`, keyed by the encoder that computed it: the
//!    backend is [`ENCODER_BACKEND_PREFIX`] plus a fingerprint of the encoder file
//!    ([`encoder_backend`]), so a re-imported encoder (a fixed export) or a voice folder's own
//!    `speaker.onnx` never reuses another encoder's embedding.
//! 3. **The speaker encoder graph** over the clip: `speaker.onnx` in the voice folder, else the
//!    model store's [`super::SPEAKER_ID`] (`model.onnx`). Its contract: one float input, the 16 kHz
//!    mono clip `[1, N]` (or `[N]`) in −1..1, and a first output of 256 values (the raw embedding).
//!    The fine-tuned weights are a manual Google Drive download (ADR 0006), so the graph is a local
//!    export and optional: a voice with (1) never needs it.
//! 4. **No encoder installed** (it was removed after use): the newest embedding an earlier encoder
//!    cached for this clip ([`latest_cached`]; the voice warns), else the load names the encoder.
//!
//! Every embedding is a `ReferenceEmbedding` with one tensor, [`SPEAKER_TENSOR`] `[1, 256]`; a
//! reference from another family (another backend) is never used.

use std::io::{Read, Seek, SeekFrom};
use std::path::{Path, PathBuf};

use ort::session::Session;
use ort::value::{TensorElementType, TensorRef};
use sha2::{Digest, Sha256};

use super::graphs::SPK_DIM;
use super::npy;
use crate::services::neural_audio::ort_session::{build_session, ModelSource, VcSessionOptions};
use crate::services::neural_audio::vc::families::reference::{sha256_file, valid_backend, ReferenceClip, ReferenceEmbedding, ReferenceTensor};

/// Cache backends of embeddings computed by a speaker encoder graph start with this; the rest is
/// the encoder's fingerprint ([`encoder_backend`]).
pub const ENCODER_BACKEND_PREFIX: &str = "meanvc2-ecapa-";
/// Bytes read from each end of the encoder file for its fingerprint.
const FINGERPRINT_SPAN: u64 = 64 * 1024;
/// Hex digits of the fingerprint in the backend name.
const FINGERPRINT_HEX: usize = 12;
/// Backend of an embedding read from a voice folder's `speaker.npy`.
pub const NPY_BACKEND: &str = "meanvc2-npy-1";
/// Backends of this family start with this.
const BACKEND_PREFIX: &str = "meanvc2-";
/// The embedding's tensor.
pub const SPEAKER_TENSOR: &str = "spks";
/// A voice folder's pre-extracted embedding.
pub const SPEAKER_NPY: &str = "speaker.npy";
/// A voice folder's own speaker encoder graph.
pub const SPEAKER_ONNX: &str = "speaker.onnx";
/// Shortest reference the encoder is run on (seconds).
pub const MIN_CLIP_SECONDS: f64 = 1.0;
/// Longest stretch of a reference the encoder sees (seconds; the library keeps 3-10 s clips).
pub const MAX_CLIP_SECONDS: f64 = 30.0;

/// An embedding of this family holding `spks`.
pub fn embedding(backend: &str, voice_id: &str, source_sha256: &str, spks: Vec<f32>) -> ReferenceEmbedding {
    ReferenceEmbedding {
        voice_id: voice_id.to_string(),
        backend: backend.to_string(),
        clip_sha256: source_sha256.to_string(),
        tensors: vec![ReferenceTensor::new(SPEAKER_TENSOR, vec![1, SPK_DIM], spks)],
    }
}

/// The speaker vector of an embedding this family made (refused: another family's, a wrong size,
/// non-finite or all-zero values).
pub fn spks_of(e: &ReferenceEmbedding) -> Result<Vec<f32>, String> {
    if !e.backend.starts_with(BACKEND_PREFIX) {
        return Err(format!("the reference embedding is from {:?}, not a MeanVC2 speaker encoder", e.backend));
    }
    let t = e
        .tensor(SPEAKER_TENSOR)
        .ok_or_else(|| format!("the MeanVC2 reference embedding has no `{SPEAKER_TENSOR}` tensor"))?;
    if t.data.len() != SPK_DIM || t.element_count() != SPK_DIM {
        return Err(format!("the MeanVC2 speaker embedding has shape {:?}, expected [1, {SPK_DIM}]", t.shape));
    }
    check_vector(&t.data)?;
    Ok(t.data.clone())
}

fn check_vector(v: &[f32]) -> Result<(), String> {
    if v.iter().any(|x| !x.is_finite()) {
        return Err("the speaker embedding holds NaN or infinite values".into());
    }
    if v.iter().all(|&x| x == 0.0) {
        return Err("the speaker embedding is all zeros".into());
    }
    Ok(())
}

/// A voice folder's `speaker.npy`, when there is one.
pub fn slot_npy(slot_dir: &Path) -> Option<PathBuf> {
    let p = slot_dir.join(SPEAKER_NPY);
    p.is_file().then_some(p)
}

/// The embedding in a `speaker.npy` file (its sha256 is the source key).
pub fn from_npy(path: &Path, voice_id: &str) -> Result<ReferenceEmbedding, String> {
    let spks = npy::read_vector(path, SPK_DIM)?;
    check_vector(&spks).map_err(|e| format!("{}: {e}", path.display()))?;
    Ok(embedding(NPY_BACKEND, voice_id, &sha256_file(path)?, spks))
}

/// The speaker encoder graph to use: the voice folder's own, else the model store's.
pub fn encoder_path(slot_dir: &Path) -> Option<PathBuf> {
    let own = slot_dir.join(SPEAKER_ONNX);
    if own.is_file() {
        return Some(own);
    }
    crate::services::neural_audio::vc::assets::model_file(super::SPEAKER_ID, None)
}

/// The cache backend of embeddings computed by the encoder at `encoder`:
/// [`ENCODER_BACKEND_PREFIX`] plus 12 hex digits of sha256 over the file's size, modification time
/// and first and last 64 KiB. Cheap (the encoder is about 1.2 GB, read in full only by ONNX
/// Runtime), and it changes whenever the file is replaced: re-importing a fixed export, or a voice
/// folder's own `speaker.onnx`, gets its own cache entry instead of another encoder's embedding.
pub fn encoder_backend(encoder: &Path) -> Result<String, String> {
    let fail = |e: std::io::Error| format!("MeanVC2 speaker encoder {}: {e}", encoder.display());
    let mut f = std::fs::File::open(encoder).map_err(fail)?;
    let meta = f.metadata().map_err(fail)?;
    let len = meta.len();
    let modified = meta
        .modified()
        .ok()
        .and_then(|t| t.duration_since(std::time::UNIX_EPOCH).ok())
        .map_or(0, |d| d.as_nanos());
    let mut hasher = Sha256::new();
    hasher.update(len.to_le_bytes());
    hasher.update(modified.to_le_bytes());
    let mut buf = Vec::with_capacity(FINGERPRINT_SPAN as usize);
    (&mut f).take(FINGERPRINT_SPAN).read_to_end(&mut buf).map_err(fail)?;
    hasher.update(&buf);
    if len > FINGERPRINT_SPAN {
        f.seek(SeekFrom::Start(FINGERPRINT_SPAN.max(len - FINGERPRINT_SPAN)))
            .map_err(fail)?;
        buf.clear();
        f.take(FINGERPRINT_SPAN).read_to_end(&mut buf).map_err(fail)?;
        hasher.update(&buf);
    }
    let hex: String = hasher
        .finalize()
        .iter()
        .take(FINGERPRINT_HEX / 2)
        .map(|b| format!("{b:02x}"))
        .collect();
    Ok(format!("{ENCODER_BACKEND_PREFIX}{hex}"))
}

/// With no encoder installed: the newest embedding any encoder cached for `clip` in the voice
/// library (`cache/meanvc2-ecapa-*.bin` of its voice, newest file first, the first that is for this
/// very clip and a usable speaker vector). `None` for an ad-hoc clip or an empty cache.
pub fn latest_cached(app_data: &Path, clip: &ReferenceClip) -> Option<ReferenceEmbedding> {
    if clip.voice_id.is_empty() {
        return None;
    }
    let dir = crate::services::voice_library::voice_dir(app_data, &clip.voice_id)
        .ok()?
        .join(crate::services::voice_library::CACHE_DIR);
    let mut found: Vec<(std::time::SystemTime, String)> = std::fs::read_dir(&dir)
        .ok()?
        .filter_map(Result::ok)
        .filter_map(|entry| {
            let name = entry.file_name().into_string().ok()?;
            let backend = name.strip_suffix(".bin")?;
            if !backend.starts_with(ENCODER_BACKEND_PREFIX) || !valid_backend(backend) {
                return None;
            }
            let modified = entry
                .metadata()
                .and_then(|m| m.modified())
                .unwrap_or(std::time::UNIX_EPOCH);
            Some((modified, backend.to_string()))
        })
        .collect();
    found.sort_by(|a, b| b.cmp(a));
    found.into_iter().find_map(|(_, backend)| {
        let bytes = crate::services::voice_library::read_cache(app_data, &clip.voice_id, &backend)?;
        let e = ReferenceEmbedding::from_bytes(&bytes).ok()?;
        let ours = e.backend == backend && e.voice_id == clip.voice_id && e.clip_sha256 == clip.sha256;
        (ours && spks_of(&e).is_ok()).then_some(e)
    })
}

/// The encoder's input name and rank (1 or 2), checked against the contract.
fn encoder_io(session: &Session) -> Result<(String, usize, String), String> {
    let inputs = session.inputs();
    if inputs.len() != 1 {
        return Err(format!(
            "the MeanVC2 speaker encoder must take one input (the 16 kHz clip), it takes {}",
            inputs.len()
        ));
    }
    let input = &inputs[0];
    if input.dtype().tensor_type() != Some(TensorElementType::Float32) {
        return Err("the MeanVC2 speaker encoder's input is not float32".into());
    }
    let rank = input.dtype().tensor_shape().map(|s| s.len()).unwrap_or(0);
    if !(1..=2).contains(&rank) {
        return Err(format!("the MeanVC2 speaker encoder's input has rank {rank}, expected [1, N] or [N]"));
    }
    let output = session
        .outputs()
        .first()
        .map(|o| o.name().to_string())
        .ok_or("the MeanVC2 speaker encoder has no output")?;
    Ok((input.name().to_string(), rank, output))
}

/// Run the speaker encoder at `encoder` over `wav` (16 kHz mono).
pub fn encode(encoder: &Path, opts: &VcSessionOptions, wav: &[f32]) -> Result<Vec<f32>, String> {
    let (mut session, _) =
        build_session(None, "cpu", opts, ModelSource::File(encoder)).map_err(|e| format!("MeanVC2 speaker encoder {}: {e}", encoder.display()))?;
    encode_with(&mut session, wav)
}

fn encode_with(session: &mut Session, wav: &[f32]) -> Result<Vec<f32>, String> {
    let (input, rank, output) = encoder_io(session)?;
    let tensor = if rank == 1 {
        TensorRef::from_array_view(([wav.len()], wav))
    } else {
        TensorRef::from_array_view(([1usize, wav.len()], wav))
    }
    .map_err(|e| format!("MeanVC2 speaker encoder input: {e}"))?;
    let outputs = session
        .run(ort::inputs![input.as_str() => tensor])
        .map_err(|e| format!("MeanVC2 speaker encoder: {e}"))?;
    let (_, data) = outputs[output.as_str()]
        .try_extract_tensor::<f32>()
        .map_err(|e| format!("MeanVC2 speaker encoder output: {e}"))?;
    if data.len() != SPK_DIM {
        return Err(format!("the MeanVC2 speaker encoder returned {} values, expected {SPK_DIM}", data.len()));
    }
    let spks = data.to_vec();
    check_vector(&spks)?;
    Ok(spks)
}

/// The clip as the encoder takes it: 16 kHz mono, at most [`MAX_CLIP_SECONDS`].
pub fn clip_audio(clip: &ReferenceClip) -> Result<Vec<f32>, String> {
    let mut wav = clip.load_mono(16_000)?;
    if (wav.len() as f64) < MIN_CLIP_SECONDS * 16_000.0 {
        return Err(format!(
            "the reference clip {} is {:.1} s; MeanVC2 needs at least {MIN_CLIP_SECONDS} s of speech",
            clip.name,
            wav.len() as f64 / 16_000.0
        ));
    }
    wav.truncate((MAX_CLIP_SECONDS * 16_000.0) as usize);
    Ok(wav)
}

/// The embedding of `clip` computed by the encoder graph, under `backend` (its
/// [`encoder_backend`]; the caller caches it).
pub fn from_clip(clip: &ReferenceClip, encoder: &Path, backend: &str, opts: &VcSessionOptions) -> Result<ReferenceEmbedding, String> {
    let wav = clip_audio(clip)?;
    let spks = encode(encoder, opts, &wav)?;
    Ok(embedding(backend, &clip.voice_id, &clip.sha256, spks))
}

#[cfg(test)]
pub(crate) mod tests {
    use super::*;
    use crate::services::neural_audio::test_onnx::{Attr, Dim, Graph, FLOAT};

    /// A fresh temporary folder no other test uses (tests run in parallel).
    pub(crate) fn test_dir(tag: &str) -> std::path::PathBuf {
        static NEXT: std::sync::atomic::AtomicUsize = std::sync::atomic::AtomicUsize::new(0);
        let n = NEXT.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
        let dir = std::env::temp_dir().join(format!("sigil-meanvc-{tag}-{}-{n}", std::process::id()));
        let _ = std::fs::remove_dir_all(&dir);
        std::fs::create_dir_all(&dir).unwrap();
        dir
    }

    /// A stand-in speaker encoder: `emb[i] = mean(wav) * (i + 1) + 0.01 (i + 1)`, `[1, 256]`.
    pub(crate) fn stand_in_encoder() -> Vec<u8> {
        stand_in_encoder_with_bias(0.01)
    }

    /// [`stand_in_encoder`] with `bias (i + 1)` in place of `0.01 (i + 1)` (another encoder: same
    /// graph and size, other weights, like a re-export).
    pub(crate) fn stand_in_encoder_with_bias(bias: f32) -> Vec<u8> {
        let scale: Vec<f32> = (0..SPK_DIM).map(|i| (i + 1) as f32).collect();
        let bias: Vec<f32> = (0..SPK_DIM).map(|i| bias * (i + 1) as f32).collect();
        let mut g = Graph::new();
        g.input("wav16k", FLOAT, &[Dim::Fixed(1), Dim::Named("N")])
            .output("emb", FLOAT, &[Dim::Fixed(1), Dim::Fixed(SPK_DIM as u64)])
            .init_f32("scale", &[1, SPK_DIM as u64], &scale)
            .init_f32("bias", &[1, SPK_DIM as u64], &bias)
            .node(
                "ReduceMean",
                &["wav16k"],
                &["m"],
                &[Attr::Ints("axes", vec![1]), Attr::Int("keepdims", 1)],
            )
            .node("Mul", &["m", "scale"], &["s"], &[])
            .node("Add", &["s", "bias"], &["emb"], &[]);
        g.model_bytes(13)
    }

    #[test]
    fn embeddings_round_trip_and_other_families_are_refused() {
        let spks: Vec<f32> = (0..SPK_DIM).map(|i| i as f32 * 0.01 - 1.0).collect();
        let e = embedding("meanvc2-ecapa-0123456789ab", "alice", "ab", spks.clone());
        assert_eq!(spks_of(&e).unwrap(), spks);
        let back = ReferenceEmbedding::from_bytes(&e.to_bytes().unwrap()).unwrap();
        assert_eq!(spks_of(&back).unwrap(), spks);
        let mut other = e.clone();
        other.backend = "sigil-zs-1".into();
        assert!(spks_of(&other).unwrap_err().contains("not a MeanVC2"));
        let mut short = e.clone();
        short.tensors[0] = ReferenceTensor::new(SPEAKER_TENSOR, vec![1, 192], vec![0.5; 192]);
        assert!(spks_of(&short).is_err());
        let mut zero = e.clone();
        zero.tensors[0].data.fill(0.0);
        assert!(spks_of(&zero).unwrap_err().contains("zeros"));
        let mut nan = e;
        nan.tensors[0].data[7] = f32::NAN;
        assert!(spks_of(&nan).is_err());
    }

    #[test]
    fn a_voice_folders_npy_is_read_with_its_hash() {
        let dir = test_dir("speaker-npy");
        std::fs::create_dir_all(&dir).unwrap();
        assert!(slot_npy(&dir).is_none());
        let spks: Vec<f32> = (0..SPK_DIM).map(|i| (i as f32).sin()).collect();
        std::fs::write(dir.join(SPEAKER_NPY), npy::encode(&spks, "(256,)", "<f4", 1)).unwrap();
        let path = slot_npy(&dir).unwrap();
        let e = from_npy(&path, "v1").unwrap();
        assert_eq!((e.backend.as_str(), e.voice_id.as_str(), e.clip_sha256.len()), (NPY_BACKEND, "v1", 64));
        assert_eq!(spks_of(&e).unwrap(), spks);
        std::fs::write(dir.join(SPEAKER_NPY), npy::encode(&[0.0; SPK_DIM], "(256,)", "<f4", 1)).unwrap();
        assert!(from_npy(&path, "v1").unwrap_err().contains("zeros"));
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn the_encoder_contract_runs_on_onnx_runtime() {
        let bytes = stand_in_encoder();
        let (mut session, _) = build_session(None, "cpu", &VcSessionOptions::default(), ModelSource::Memory(&bytes)).unwrap();
        let wav = vec![0.5f32; 16_000];
        let spks = encode_with(&mut session, &wav).unwrap();
        assert_eq!(spks.len(), SPK_DIM);
        assert!((spks[0] - 0.51).abs() < 1e-5 && (spks[255] - 256.0 * 0.51).abs() < 1e-3);
        // A file with another shape is refused by name.
        let mut g = Graph::new();
        g.input("wav16k", FLOAT, &[Dim::Fixed(1), Dim::Named("N")])
            .output("emb", FLOAT, &[Dim::Fixed(1), Dim::Fixed(1)])
            .node(
                "ReduceMean",
                &["wav16k"],
                &["emb"],
                &[Attr::Ints("axes", vec![1]), Attr::Int("keepdims", 1)],
            );
        let (mut wrong, _) = build_session(None, "cpu", &VcSessionOptions::default(), ModelSource::Memory(&g.model_bytes(13))).unwrap();
        assert!(encode_with(&mut wrong, &wav).unwrap_err().contains("256"));
    }

    /// The cache key names the encoder file: stable for the same file, new for a replaced one (even
    /// at the same size), always a valid cache file stem of this family.
    #[test]
    fn the_encoder_backend_fingerprints_the_encoder_file() {
        let dir = test_dir("speaker-fp");
        let a = dir.join("a.onnx");
        std::fs::write(&a, stand_in_encoder()).unwrap();
        let fa = encoder_backend(&a).unwrap();
        assert_eq!(encoder_backend(&a).unwrap(), fa, "stable");
        assert!(fa.starts_with(ENCODER_BACKEND_PREFIX) && fa.len() == ENCODER_BACKEND_PREFIX.len() + FINGERPRINT_HEX, "{fa}");
        assert!(valid_backend(&fa), "{fa}");
        // Another export of the same size (other weights) in another file.
        let b = dir.join("b.onnx");
        std::fs::write(&b, stand_in_encoder_with_bias(0.02)).unwrap();
        assert_eq!(std::fs::metadata(&a).unwrap().len(), std::fs::metadata(&b).unwrap().len());
        assert_ne!(encoder_backend(&b).unwrap(), fa);
        // Replaced in place (a re-import), even with the old modification time restored.
        let old = std::fs::metadata(&a).unwrap().modified().unwrap();
        std::fs::write(&a, stand_in_encoder_with_bias(0.03)).unwrap();
        std::fs::File::options()
            .write(true)
            .open(&a)
            .unwrap()
            .set_modified(old)
            .unwrap();
        assert_ne!(encoder_backend(&a).unwrap(), fa, "the content sample sees the new weights");
        // A large file: the head and the tail are both read.
        let big = dir.join("big.onnx");
        let mut bytes = vec![7u8; 3 * FINGERPRINT_SPAN as usize];
        std::fs::write(&big, &bytes).unwrap();
        let t = std::fs::metadata(&big).unwrap().modified().unwrap();
        let f0 = encoder_backend(&big).unwrap();
        *bytes.last_mut().unwrap() = 8;
        std::fs::write(&big, &bytes).unwrap();
        std::fs::File::options().write(true).open(&big).unwrap().set_modified(t).unwrap();
        assert_ne!(encoder_backend(&big).unwrap(), f0, "the tail is part of the fingerprint");
        assert!(encoder_backend(&dir.join("missing.onnx")).unwrap_err().contains("speaker encoder"));
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn short_clips_are_refused_and_long_ones_trimmed() {
        let dir = test_dir("speaker-clip");
        std::fs::create_dir_all(&dir).unwrap();
        let short = dir.join("short.wav");
        crate::services::bench::signals::write_wav(&short, &vec![0.1f32; 8_000], 16_000, 1).unwrap();
        let clip = ReferenceClip::from_wav(&short).unwrap();
        assert!(clip_audio(&clip).unwrap_err().contains("at least"));
        let long = dir.join("long.wav");
        crate::services::bench::signals::write_wav(&long, &vec![0.1f32; 16_000 * 35], 16_000, 1).unwrap();
        let clip = ReferenceClip::from_wav(&long).unwrap();
        assert_eq!(clip_audio(&clip).unwrap().len(), 16_000 * 30);
        let _ = std::fs::remove_dir_all(&dir);
    }
}
