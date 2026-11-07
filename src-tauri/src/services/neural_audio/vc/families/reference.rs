//! Zero-shot reference voices as the voice-conversion families see them (P5-S0, scaffold-owned,
//! frozen: change only through INTEGRATE; structs may gain fields).
//!
//! - [`ReferenceClip`]: a clean 3–10 s clip of the target voice. The shared voice library
//!   (`services::voice_library::reference_clip`, P5-F) returns one for a library voice
//!   (`voices/<id>/ref.wav`); tests and benchmarks make one from any WAV with
//!   [`ReferenceClip::from_wav`]. [`ReferenceClip::load_mono`] decodes it at the rate a family
//!   needs (band-limited resampling, the harness's sinc parameters).
//! - [`ReferenceEmbedding`]: what a zero-shot family computes once per clip, as named f32
//!   tensors (Sigil-ZS: the CAM++ style `[1,192]`, the prompt mel `[1,80,L]` and the prompt
//!   condition `[1,L,384]`; MeanVC2: the speaker vector and its GTM memories). Its binary codec
//!   ([`ReferenceEmbedding::to_bytes`] / [`ReferenceEmbedding::from_bytes`]) is the format of the
//!   library's per-backend cache file `voices/<id>/cache/<backend>.bin`.
//! - [`cached_embedding`]: the cached embedding when it was computed from the same clip
//!   (`sha256`) by the same backend string, else `compute` it and store it in the library.
//!
//! The engine factory installs the embedding a family returns (`FamilyEngine::reference`) into
//! the voice's stream harness, so every tick's `ConversionCtx::reference` carries it (warm-ups
//! included). A family detects a changed reference with [`ReferenceEmbedding::same_source`].

// P5-S0 frozen API: the zero-shot families (P5-B, P5-C) and the voice library (P5-F) use it as
// they land; until then only the tests do.
#![allow(dead_code)]

use std::io::Read;
use std::path::{Path, PathBuf};

use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};

/// Magic of the cache file format (8 bytes), followed by a little-endian `u32` header length, the
/// JSON header ([`CacheHeader`]) and the tensors' f32 data (little-endian, in header order).
pub const CACHE_MAGIC: &[u8; 8] = b"SGREF1\0\0";
/// Largest JSON header accepted (bytes).
const MAX_HEADER_BYTES: usize = 1 << 20;
/// Most f32 values an embedding may hold (a 30 s prompt condition is about 1.2 M).
pub const MAX_EMBEDDING_VALUES: usize = 64 << 20;
/// Clips longer than this are refused by [`ReferenceClip::load_mono`] (the library keeps 3–10 s;
/// the bound only stops a wrong file from allocating gigabytes).
pub const MAX_CLIP_SECONDS: f64 = 120.0;

/// A reference clip (see the module docs).
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct ReferenceClip {
    /// Voice-library id (`voices/<id>/`); empty for an ad-hoc clip ([`ReferenceClip::from_wav`]),
    /// which is never cached.
    pub voice_id: String,
    /// Display name.
    pub name: String,
    /// The clip (a WAV file).
    pub path: PathBuf,
    /// sha256 (lower-case hex) of the clip file: the embedding cache key.
    pub sha256: String,
    /// The file's sample rate (Hz).
    pub sample_rate: u32,
    pub duration_ms: u64,
    /// What is said in the clip (local STT), when known (zero-shot TTS needs it; VC does not).
    #[serde(default)]
    pub transcript: Option<String>,
    /// BCP-47 language of the transcript, when known.
    #[serde(default)]
    pub language: Option<String>,
}

impl ReferenceClip {
    /// An ad-hoc clip from a WAV file (tests, benchmarks): reads the header and hashes the file.
    pub fn from_wav(path: &Path) -> Result<ReferenceClip, String> {
        let reader = hound::WavReader::open(path).map_err(|e| format!("cannot read {}: {e}", path.display()))?;
        let spec = reader.spec();
        let frames = u64::from(reader.duration());
        drop(reader);
        let rate = spec.sample_rate.max(1);
        Ok(ReferenceClip {
            voice_id: String::new(),
            name: path.file_stem().map(|s| s.to_string_lossy().into_owned()).unwrap_or_default(),
            path: path.to_path_buf(),
            sha256: sha256_file(path)?,
            sample_rate: rate,
            duration_ms: frames * 1000 / u64::from(rate),
            transcript: None,
            language: None,
        })
    }

    /// The clip as mono f32 at `sample_rate` (channels averaged; band-limited resampling).
    pub fn load_mono(&self, sample_rate: u32) -> Result<Vec<f32>, String> {
        if self.duration_ms as f64 > MAX_CLIP_SECONDS * 1000.0 {
            return Err(format!("{} is longer than {MAX_CLIP_SECONDS} s", self.path.display()));
        }
        let (data, rate, ch) = crate::services::bench::signals::read_wav(&self.path)?;
        if rate == 0 || ch == 0 {
            return Err(format!("{} has no audio", self.path.display()));
        }
        if data.len() as f64 > MAX_CLIP_SECONDS * f64::from(rate) * ch as f64 {
            return Err(format!("{} is longer than {MAX_CLIP_SECONDS} s", self.path.display()));
        }
        let mono: Vec<f32> = if ch == 1 {
            data
        } else {
            data.chunks_exact(ch).map(|f| f.iter().sum::<f32>() / ch as f32).collect()
        };
        crate::services::bench::signals::resample(&mono, rate, sample_rate.max(1))
    }
}

/// sha256 (lower-case hex) of a file.
pub fn sha256_file(path: &Path) -> Result<String, String> {
    let mut f = std::fs::File::open(path).map_err(|e| format!("cannot read {}: {e}", path.display()))?;
    let mut hasher = Sha256::new();
    let mut buf = vec![0u8; 1 << 16];
    loop {
        let n = f.read(&mut buf).map_err(|e| format!("cannot read {}: {e}", path.display()))?;
        if n == 0 {
            break;
        }
        hasher.update(&buf[..n]);
    }
    Ok(hasher.finalize().iter().map(|b| format!("{b:02x}")).collect())
}

/// One named tensor of an embedding (row-major f32).
#[derive(Clone, Debug, Default, PartialEq)]
pub struct ReferenceTensor {
    pub name: String,
    pub shape: Vec<usize>,
    pub data: Vec<f32>,
}

impl ReferenceTensor {
    pub fn new(name: impl Into<String>, shape: Vec<usize>, data: Vec<f32>) -> ReferenceTensor {
        ReferenceTensor { name: name.into(), shape, data }
    }

    /// Product of the shape (what `data.len()` must be).
    pub fn element_count(&self) -> usize {
        self.shape.iter().try_fold(1usize, |acc, &d| acc.checked_mul(d)).unwrap_or(usize::MAX)
    }
}

/// A zero-shot family's per-clip reference (see the module docs).
#[derive(Clone, Debug, Default, PartialEq)]
pub struct ReferenceEmbedding {
    /// Voice-library id of the clip (empty for an ad-hoc clip).
    pub voice_id: String,
    /// Who computed it, with a version (a safe file stem, `[a-z0-9._-]`, e.g. `sigil-zs-1`): the
    /// cache file is `cache/<backend>.bin`; a family that changes what it stores bumps it.
    pub backend: String,
    /// [`ReferenceClip::sha256`] of the clip it was computed from.
    pub clip_sha256: String,
    pub tensors: Vec<ReferenceTensor>,
}

#[derive(Serialize, Deserialize)]
struct CacheTensor {
    name: String,
    shape: Vec<usize>,
}

#[derive(Serialize, Deserialize)]
struct CacheHeader {
    voice_id: String,
    backend: String,
    clip_sha256: String,
    tensors: Vec<CacheTensor>,
}

/// Whether `backend` is a safe cache file stem.
pub fn valid_backend(backend: &str) -> bool {
    !backend.is_empty()
        && backend.len() <= 64
        && !backend.starts_with('.')
        && backend.chars().all(|c| c.is_ascii_lowercase() || c.is_ascii_digit() || matches!(c, '.' | '_' | '-'))
}

impl ReferenceEmbedding {
    pub fn tensor(&self, name: &str) -> Option<&ReferenceTensor> {
        self.tensors.iter().find(|t| t.name == name)
    }

    /// Same clip, same backend: the family's prepared state for `other` is valid for `self`.
    pub fn same_source(&self, other: &ReferenceEmbedding) -> bool {
        self.voice_id == other.voice_id && self.backend == other.backend && self.clip_sha256 == other.clip_sha256
    }

    /// Every tensor's data matches its shape, and the total stays within [`MAX_EMBEDDING_VALUES`].
    pub fn validate(&self) -> Result<(), String> {
        let mut total = 0usize;
        for t in &self.tensors {
            let n = t.element_count();
            if n != t.data.len() {
                return Err(format!("reference tensor {} has {} values for shape {:?}", t.name, t.data.len(), t.shape));
            }
            total = total.saturating_add(n);
        }
        if total > MAX_EMBEDDING_VALUES {
            return Err(format!("the reference embedding holds {total} values (at most {MAX_EMBEDDING_VALUES})"));
        }
        Ok(())
    }

    /// The cache file bytes (see [`CACHE_MAGIC`]).
    pub fn to_bytes(&self) -> Result<Vec<u8>, String> {
        self.validate()?;
        let header = CacheHeader {
            voice_id: self.voice_id.clone(),
            backend: self.backend.clone(),
            clip_sha256: self.clip_sha256.clone(),
            tensors: self.tensors.iter().map(|t| CacheTensor { name: t.name.clone(), shape: t.shape.clone() }).collect(),
        };
        let json = serde_json::to_vec(&header).map_err(|e| e.to_string())?;
        if json.len() > MAX_HEADER_BYTES {
            return Err("the reference embedding header is too large".into());
        }
        let values: usize = self.tensors.iter().map(|t| t.data.len()).sum();
        let mut out = Vec::with_capacity(CACHE_MAGIC.len() + 4 + json.len() + values * 4);
        out.extend_from_slice(CACHE_MAGIC);
        out.extend_from_slice(&(json.len() as u32).to_le_bytes());
        out.extend_from_slice(&json);
        for t in &self.tensors {
            for v in &t.data {
                out.extend_from_slice(&v.to_le_bytes());
            }
        }
        Ok(out)
    }

    /// Parse cache file bytes (bounds-checked; never trusts the header's sizes).
    pub fn from_bytes(bytes: &[u8]) -> Result<ReferenceEmbedding, String> {
        let bad = |why: &str| format!("not a reference embedding cache: {why}");
        if bytes.len() < CACHE_MAGIC.len() + 4 || &bytes[..CACHE_MAGIC.len()] != CACHE_MAGIC {
            return Err(bad("wrong magic"));
        }
        let at = CACHE_MAGIC.len();
        let len = u32::from_le_bytes([bytes[at], bytes[at + 1], bytes[at + 2], bytes[at + 3]]) as usize;
        if len > MAX_HEADER_BYTES || bytes.len() < at + 4 + len {
            return Err(bad("truncated header"));
        }
        let header: CacheHeader = serde_json::from_slice(&bytes[at + 4..at + 4 + len]).map_err(|e| bad(&e.to_string()))?;
        let mut pos = at + 4 + len;
        let mut tensors = Vec::with_capacity(header.tensors.len());
        let mut total = 0usize;
        for t in header.tensors {
            let n = t.shape.iter().try_fold(1usize, |acc, &d| acc.checked_mul(d)).ok_or_else(|| bad("shape overflows"))?;
            total = total.checked_add(n).filter(|&v| v <= MAX_EMBEDDING_VALUES).ok_or_else(|| bad("too many values"))?;
            let end = n.checked_mul(4).and_then(|b| pos.checked_add(b)).filter(|&e| e <= bytes.len()).ok_or_else(|| bad("truncated data"))?;
            let data = bytes[pos..end].as_chunks::<4>().0.iter().map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]])).collect();
            pos = end;
            tensors.push(ReferenceTensor { name: t.name, shape: t.shape, data });
        }
        if pos != bytes.len() {
            return Err(bad("trailing bytes"));
        }
        Ok(ReferenceEmbedding {
            voice_id: header.voice_id,
            backend: header.backend,
            clip_sha256: header.clip_sha256,
            tensors,
        })
    }
}

/// The embedding of `clip` for `backend`: the voice library's cache when it holds one computed from
/// this clip by this backend, else `compute(clip)` (its `voice_id`, `backend` and `clip_sha256` are
/// set here) stored in the cache. Ad-hoc clips (no `voice_id`) and a missing `app_data` skip the
/// cache. A cache that cannot be read or written only costs the computation (logged at debug).
pub fn cached_embedding(
    app_data: Option<&Path>,
    clip: &ReferenceClip,
    backend: &str,
    compute: impl FnOnce(&ReferenceClip) -> Result<ReferenceEmbedding, String>,
) -> Result<ReferenceEmbedding, String> {
    if !valid_backend(backend) {
        return Err(format!("invalid reference cache backend {backend:?}"));
    }
    let cache = app_data.filter(|_| !clip.voice_id.is_empty());
    if let Some(app_data) = cache {
        if let Some(bytes) = crate::services::voice_library::read_cache(app_data, &clip.voice_id, backend) {
            match ReferenceEmbedding::from_bytes(&bytes) {
                Ok(e) if e.backend == backend && e.clip_sha256 == clip.sha256 && e.voice_id == clip.voice_id => return Ok(e),
                Ok(_) => tracing::debug!("reference cache {}/{backend}: stale, recomputing", clip.voice_id),
                Err(e) => tracing::debug!("reference cache {}/{backend}: {e}", clip.voice_id),
            }
        }
    }
    let mut embedding = compute(clip)?;
    embedding.voice_id = clip.voice_id.clone();
    embedding.backend = backend.to_string();
    embedding.clip_sha256 = clip.sha256.clone();
    embedding.validate()?;
    if let Some(app_data) = cache {
        match embedding.to_bytes() {
            Ok(bytes) => {
                if let Err(e) = crate::services::voice_library::write_cache(app_data, &clip.voice_id, backend, &bytes) {
                    tracing::debug!("reference cache {}/{backend}: not stored: {e}", clip.voice_id);
                }
            }
            Err(e) => tracing::debug!("reference cache {}/{backend}: not stored: {e}", clip.voice_id),
        }
    }
    Ok(embedding)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn sample() -> ReferenceEmbedding {
        ReferenceEmbedding {
            voice_id: "alice".into(),
            backend: "sigil-zs-1".into(),
            clip_sha256: "ab".repeat(32),
            tensors: vec![
                ReferenceTensor::new("style", vec![1, 3], vec![0.5, -1.0, 2.25]),
                ReferenceTensor::new("prompt_mel", vec![1, 2, 2], vec![1.0, 2.0, 3.0, f32::MIN_POSITIVE]),
                ReferenceTensor::new("empty", vec![0], vec![]),
            ],
        }
    }

    #[test]
    fn the_cache_codec_round_trips_bit_exactly() {
        let e = sample();
        let bytes = e.to_bytes().unwrap();
        assert_eq!(&bytes[..8], CACHE_MAGIC);
        let back = ReferenceEmbedding::from_bytes(&bytes).unwrap();
        assert_eq!(back, e);
        assert!(back.same_source(&e));
        assert_eq!(back.tensor("style").unwrap().data, vec![0.5, -1.0, 2.25]);
        assert!(back.tensor("nope").is_none());
    }

    #[test]
    fn damaged_cache_files_are_refused() {
        let bytes = sample().to_bytes().unwrap();
        assert!(ReferenceEmbedding::from_bytes(&bytes[..bytes.len() - 1]).is_err(), "truncated data");
        let mut extra = bytes.clone();
        extra.push(0);
        assert!(ReferenceEmbedding::from_bytes(&extra).is_err(), "trailing bytes");
        let mut magic = bytes.clone();
        magic[0] = b'X';
        assert!(ReferenceEmbedding::from_bytes(&magic).is_err());
        assert!(ReferenceEmbedding::from_bytes(&bytes[..10]).is_err());
        // A header that claims a huge tensor never allocates it.
        let header = br#"{"voice_id":"a","backend":"b","clip_sha256":"c","tensors":[{"name":"x","shape":[1000000000,1000000000]}]}"#;
        let mut hostile = CACHE_MAGIC.to_vec();
        hostile.extend_from_slice(&(header.len() as u32).to_le_bytes());
        hostile.extend_from_slice(header);
        assert!(ReferenceEmbedding::from_bytes(&hostile).is_err());
    }

    #[test]
    fn shapes_must_match_their_data() {
        let mut e = sample();
        e.tensors[0].data.pop();
        assert!(e.validate().is_err() && e.to_bytes().is_err());
    }

    #[test]
    fn backends_are_safe_file_stems() {
        for ok in ["sigil-zs-1", "meanvc2.v1", "a_b"] {
            assert!(valid_backend(ok), "{ok}");
        }
        for bad in ["", "../x", "A", "a/b", ".hidden", "a b"] {
            assert!(!valid_backend(bad), "{bad}");
        }
    }

    #[test]
    fn ad_hoc_clips_load_resampled_and_are_never_cached() {
        let dir = std::env::temp_dir().join(format!("sigil-ref-clip-{}-{}", std::process::id(), crate::services::neural_audio::vc::import::unix_ms()));
        std::fs::create_dir_all(&dir).unwrap();
        let wav = dir.join("ref.wav");
        // 0.5 s of a 220 Hz tone, stereo, 22.05 kHz.
        let tone: Vec<f32> = (0..11_025).flat_map(|i| {
            let v = 0.25 * (std::f32::consts::TAU * 220.0 * i as f32 / 22_050.0).sin();
            [v, v]
        }).collect();
        crate::services::bench::signals::write_wav(&wav, &tone, 22_050, 2).unwrap();
        let clip = ReferenceClip::from_wav(&wav).unwrap();
        assert_eq!((clip.sample_rate, clip.duration_ms, clip.sha256.len()), (22_050, 500, 64));
        assert!(clip.voice_id.is_empty());
        let mono = clip.load_mono(16_000).unwrap();
        assert_eq!(mono.len(), 8_000);
        assert!(mono[2_000..6_000].iter().any(|v| v.abs() > 0.2));
        let mut calls = 0;
        for _ in 0..2 {
            let e = cached_embedding(Some(&dir), &clip, "test-1", |c| {
                calls += 1;
                Ok(ReferenceEmbedding { tensors: vec![ReferenceTensor::new("n", vec![1], vec![c.duration_ms as f32])], ..Default::default() })
            })
            .unwrap();
            assert_eq!((e.backend.as_str(), e.clip_sha256.as_str()), ("test-1", clip.sha256.as_str()));
        }
        assert_eq!(calls, 2, "an ad-hoc clip is computed every time");
        assert!(cached_embedding(None, &clip, "../bad", |_| Ok(ReferenceEmbedding::default())).is_err());
        let _ = std::fs::remove_dir_all(&dir);
    }
}
