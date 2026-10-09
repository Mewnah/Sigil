//! Model-based quality scorers of the benchmark (P3-F; plan T5 / T6): CAM++ speaker similarity,
//! DNSMOS P.835 and the character error rate of the converted speech through Sigil's own STT.
//!
//! The models are the store entries of kind `bench` (`vc/assets.rs`: `bench/campplus`,
//! `bench/dnsmos-p835`), never needed by normal users; a spec can also name the files. Contracts
//! (`scripts/vc-templates/export_bench.py`, reference values in its `reports/bench.json`):
//! - **CAM++**: `x` f32 `[1, T, 80]` = Kaldi fbank of 16 kHz audio (`dsp::kaldi_fbank`) minus its
//!   mean over time → `embedding` `[1, 192]`; speaker similarity is the cosine of two embeddings.
//! - **DNSMOS P.835** (`model.onnx`): `input_1` f32 `[1, 144160]` (9.01 s at 16 kHz) → `[1, 3]` raw
//!   SIG, BAK, OVRL, mapped by DNS-Challenge's non-personalised polynomials. Clips are scored as in
//!   `dnsmos_local.py`: shorter clips are tiled to 9.01 s, longer ones in 9.01 s windows hopped by
//!   1 s, and the window scores are averaged.
//! - **CER**: any `speech` engine (`speech::engines::create`, e.g. Whisper with a local ggml
//!   model) transcribes the clip; [`quality::cer`](super::quality::cer) scores it.
//!
//! Sessions are background sessions (2 threads, recorded as `bench.quality.*`), built through
//! `ort_session` on the analysis EP of the run (CPU by default).
//!
//! # Scorer v2 (P3B-B, ADR 0009)
//! - The v2 speaker scorer is `bench/campplus-exact` ([`SpeakerEmbedder::resolve_v2`]): the same
//!   3D-Speaker checkpoint as `bench/campplus`, re-exported with exact segment pooling (>= 100 dB SNR
//!   against upstream PyTorch; the legacy file differs from upstream on every length that is not a
//!   whole number of 100-frame segments). `speaker_model: legacy` selects the old file.
//! - Embeddings are of speech frames only ([`SpeakerEmbedder::embed_masked`] with a
//!   [`speech_frame_mask`]): the P3 scorer embedded the whole region, near-silence included, and the
//!   ceiling's re-rendered silence (6 frames at -78 dBFS, the speech bit-identical) moved the cosine
//!   of identical stream audio by 0.006-0.008 between two builds. A pair is embedded on the same
//!   frames (the reference's mask), so the same WAV pair always gives a bit-identical cosine.
//! - A target-speaker set ([`centroid`]): the identity metric compares a rendition with the
//!   normalised mean embedding of reference clips of the target voice.
//! - Every model is stamped with its file's sha256 ([`file_sha256`]).

use std::path::{Path, PathBuf};

use ort::session::Session;
use ort::value::Tensor;
use serde::{Deserialize, Serialize};

use super::dsp::kaldi_fbank;
use crate::services::diagnostics::EpOutcome;
use crate::services::neural_audio::ort_session::{build_session, ModelSource, VcSessionOptions};
use crate::services::neural_audio::vc::assets;

/// Store file of `id` (`model.onnx` or a named file), when installed.
fn store_file(id: &str, file: &str) -> Option<PathBuf> {
    let p = crate::services::model_store::installed_path(id)?.join(file);
    p.is_file().then_some(p)
}

/// `explicit` when given (must exist), else the store entry's file, else the dev-cache layout
/// under `SIGIL_TEST_MODELS_DIR` (`vc-assets/<id>/<file>`, used by the dev-rig tests).
pub fn resolve_bench_model(explicit: &str, id: &str, file: &str) -> Result<Option<PathBuf>, String> {
    let t = explicit.trim();
    if !t.is_empty() {
        let p = PathBuf::from(t);
        return if p.is_file() { Ok(Some(p)) } else { Err(format!("{id}: {} not found", p.display())) };
    }
    if let Some(p) = store_file(id, file) {
        return Ok(Some(p));
    }
    if let Some(root) = std::env::var_os("SIGIL_TEST_MODELS_DIR").map(PathBuf::from) {
        let p = root.join("vc-assets").join(id).join(file);
        if p.is_file() {
            return Ok(Some(p));
        }
    }
    Ok(None)
}

fn analysis_opts() -> VcSessionOptions {
    VcSessionOptions {
        intra_threads: Some(2),
        ..VcSessionOptions::default()
    }
}

fn load(component: &str, path: &Path, ep: &str) -> Result<(Session, EpOutcome), String> {
    build_session(Some(component), ep, &analysis_opts(), ModelSource::File(path)).map_err(|e| format!("{}: {e}", path.display()))
}

// ---------------------------------------------------------------------------------------------
// CAM++
// ---------------------------------------------------------------------------------------------

/// CAM++ speaker embedder.
pub struct SpeakerEmbedder {
    session: Session,
    pub outcome: EpOutcome,
    pub path: PathBuf,
    /// The store id it was resolved as (`bench/campplus-exact`, `bench/campplus`), or `explicit`.
    pub id: String,
    /// sha256 of the model file (the scorer's version stamp).
    pub sha256: String,
}

/// Shortest clip CAM++ embeds (0.5 s: 48 fbank frames).
const MIN_EMBED_SAMPLES: usize = 8_000;
/// Fewest frames an embedding is computed from (0.5 s of 10 ms fbank frames).
pub const MIN_EMBED_FRAMES: usize = 48;
/// Frames within this many dB of a clip's loudest frame are speech for the speaker embedding (the
/// MCD's speech range).
pub const SPEAKER_SPEECH_RANGE_DB: f64 = 40.0;

/// sha256 (hex) of a file.
pub fn file_sha256(path: &Path) -> Result<String, String> {
    use sha2::{Digest, Sha256};
    use std::io::Read;
    let mut f = std::fs::File::open(path).map_err(|e| format!("{}: {e}", path.display()))?;
    let mut h = Sha256::new();
    let mut buf = vec![0u8; 1 << 20];
    loop {
        let n = f.read(&mut buf).map_err(|e| format!("{}: {e}", path.display()))?;
        if n == 0 {
            break;
        }
        h.update(&buf[..n]);
    }
    Ok(hex::encode(h.finalize()))
}

/// Which Kaldi fbank frames (`dsp::kaldi_fbank`'s grid: 400 samples every 160) of a 16 kHz clip
/// are speech: frame energy within `range_db` of the clip's loudest frame.
pub fn speech_frame_mask(audio_16k: &[f32], range_db: f64) -> Vec<bool> {
    let energy: Vec<f64> = super::dsp::frames(audio_16k.len(), 400, 160)
        .map(|s| {
            let e = audio_16k[s..s + 400].iter().map(|&v| f64::from(v) * f64::from(v)).sum::<f64>() / 400.0;
            10.0 * (e + 1e-12).log10()
        })
        .collect();
    let peak = energy.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
    energy.into_iter().map(|e| e >= peak - range_db).collect()
}

/// The L2-normalised mean of L2-normalised embeddings (a target-speaker set's centroid).
pub fn centroid(embeddings: &[Vec<f32>]) -> Option<Vec<f32>> {
    let dim = embeddings.first()?.len();
    let mut acc = vec![0.0f64; dim];
    for e in embeddings.iter().filter(|e| e.len() == dim) {
        for (a, &v) in acc.iter_mut().zip(e) {
            *a += f64::from(v);
        }
    }
    let norm = acc.iter().map(|v| v * v).sum::<f64>().sqrt();
    (norm > 0.0 && norm.is_finite()).then(|| acc.iter().map(|v| (v / norm) as f32).collect())
}

impl SpeakerEmbedder {
    fn load_as(path: &Path, ep: &str, id: &str) -> Result<SpeakerEmbedder, String> {
        let (session, outcome) = load("bench.quality.campplus", path, ep)?;
        Ok(SpeakerEmbedder {
            session,
            outcome,
            path: path.to_path_buf(),
            id: id.to_string(),
            sha256: file_sha256(path)?,
        })
    }

    /// The store's `bench/campplus` (or `explicit`); `None` when not installed (the v1 scorer).
    pub fn resolve(explicit: &str, ep: &str) -> Result<Option<SpeakerEmbedder>, String> {
        let id = if explicit.trim().is_empty() { assets::BENCH_CAMPPLUS } else { "explicit" };
        match resolve_bench_model(explicit, assets::BENCH_CAMPPLUS, "model.onnx")? {
            Some(p) => SpeakerEmbedder::load_as(&p, ep, id).map(Some),
            None => Ok(None),
        }
    }

    /// The v2 speaker scorer: `""` = the store's `bench/campplus-exact`, `legacy` = `bench/campplus`,
    /// anything else a file; `None` when the store entry is not installed.
    pub fn resolve_v2(spec: &str, ep: &str) -> Result<Option<SpeakerEmbedder>, String> {
        let t = spec.trim();
        if t.eq_ignore_ascii_case("legacy") {
            return Self::resolve("", ep);
        }
        let id = if t.is_empty() { assets::BENCH_CAMPPLUS_EXACT } else { "explicit" };
        match resolve_bench_model(t, assets::BENCH_CAMPPLUS_EXACT, "model.onnx")? {
            Some(p) => SpeakerEmbedder::load_as(&p, ep, id).map(Some),
            None => Ok(None),
        }
    }

    /// `{id, path, sha256, ep}` for reports.
    pub fn describe(&self) -> serde_json::Value {
        serde_json::json!({
            "id": self.id,
            "path": self.path.display().to_string(),
            "sha256": self.sha256,
            "ep": self.outcome.active,
        })
    }

    /// L2-normalized 192-d embedding of a 16 kHz clip (at least 0.5 s; every frame).
    pub fn embed(&mut self, audio_16k: &[f32]) -> Result<Vec<f32>, String> {
        if audio_16k.len() < MIN_EMBED_SAMPLES {
            return Err(format!("CAM++: {} samples is shorter than 0.5 s", audio_16k.len()));
        }
        let (fb, frames) = kaldi_fbank(audio_16k, 80);
        self.embed_fbank(fb, frames)
    }

    /// L2-normalized embedding of the fbank frames of `audio_16k` that `mask` keeps (the frame grid
    /// of [`speech_frame_mask`]; frames beyond the mask are dropped), mean-normalized over those
    /// frames. At least [`MIN_EMBED_FRAMES`] must remain.
    pub fn embed_masked(&mut self, audio_16k: &[f32], mask: &[bool]) -> Result<Vec<f32>, String> {
        let (fb, frames) = kaldi_fbank(audio_16k, 80);
        let mut kept = Vec::with_capacity(fb.len());
        let mut n = 0usize;
        for f in 0..frames {
            if mask.get(f).copied().unwrap_or(false) {
                kept.extend_from_slice(&fb[f * 80..(f + 1) * 80]);
                n += 1;
            }
        }
        if n < MIN_EMBED_FRAMES {
            return Err(format!("CAM++: {n} speech frames (at least {MIN_EMBED_FRAMES} needed)"));
        }
        self.embed_fbank(kept, n)
    }

    /// Cosine of two clips' speakers on the same frames (`mask`, usually the reference's speech).
    pub fn similarity_masked(&mut self, a_16k: &[f32], b_16k: &[f32], mask: &[bool]) -> Result<f64, String> {
        let a = self.embed_masked(a_16k, mask)?;
        let b = self.embed_masked(b_16k, mask)?;
        Ok(cosine(&a, &b))
    }

    fn embed_fbank(&mut self, mut fb: Vec<f32>, frames: usize) -> Result<Vec<f32>, String> {
        if frames == 0 {
            return Err("CAM++: no fbank frames".into());
        }
        // Mean normalization over time (per bin).
        for b in 0..80 {
            let mean = (0..frames).map(|f| f64::from(fb[f * 80 + b])).sum::<f64>() / frames as f64;
            for f in 0..frames {
                fb[f * 80 + b] -= mean as f32;
            }
        }
        let x = Tensor::from_array(([1usize, frames, 80], fb)).map_err(|e| format!("CAM++ input: {e}"))?;
        let out = self
            .session
            .run(ort::inputs!["x" => &x])
            .map_err(|e| format!("CAM++: {e}"))?;
        let (_, e) = out["embedding"]
            .try_extract_tensor::<f32>()
            .map_err(|e| format!("CAM++ output: {e}"))?;
        let norm = e.iter().map(|v| f64::from(*v) * f64::from(*v)).sum::<f64>().sqrt();
        if !(norm > 0.0 && norm.is_finite()) {
            return Err("CAM++ returned a degenerate embedding".into());
        }
        Ok(e.iter().map(|v| (f64::from(*v) / norm) as f32).collect())
    }

    /// Cosine similarity of two clips' speakers.
    pub fn similarity(&mut self, a_16k: &[f32], b_16k: &[f32]) -> Result<f64, String> {
        let a = self.embed(a_16k)?;
        let b = self.embed(b_16k)?;
        Ok(cosine(&a, &b))
    }
}

/// Cosine of two vectors (0 when either is zero).
pub fn cosine(a: &[f32], b: &[f32]) -> f64 {
    let (mut ab, mut aa, mut bb) = (0.0f64, 0.0f64, 0.0f64);
    for (&x, &y) in a.iter().zip(b) {
        ab += f64::from(x) * f64::from(y);
        aa += f64::from(x) * f64::from(x);
        bb += f64::from(y) * f64::from(y);
    }
    if aa <= 0.0 || bb <= 0.0 {
        0.0
    } else {
        ab / (aa * bb).sqrt()
    }
}

// ---------------------------------------------------------------------------------------------
// DNSMOS P.835
// ---------------------------------------------------------------------------------------------

/// DNSMOS input window: 9.01 s at 16 kHz.
pub const DNSMOS_LEN: usize = 144_160;
/// `np.poly1d` coefficients (highest power first) of DNS-Challenge `dnsmos_local.py`
/// `get_polyfit_val`, `is_personalized_MOS = False`.
const POLY_OVRL: [f64; 3] = [-0.067_662_83, 1.115_464_68, 0.046_025_35];
const POLY_SIG: [f64; 3] = [-0.083_972_78, 1.220_839_53, 0.005_243_9];
const POLY_BAK: [f64; 3] = [-0.131_668_88, 1.609_155_14, -0.396_045_46];

fn poly(c: &[f64; 3], x: f64) -> f64 {
    c[0] * x * x + c[1] * x + c[2]
}

/// DNSMOS P.835 scores of a clip.
#[derive(Debug, Clone, Copy, Default, PartialEq, Serialize, Deserialize)]
pub struct DnsmosScore {
    pub sig: f64,
    pub bak: f64,
    pub ovrl: f64,
    /// 9.01 s windows averaged.
    pub windows: usize,
}

/// DNSMOS P.835 scorer.
pub struct Dnsmos {
    session: Session,
    pub outcome: EpOutcome,
    pub path: PathBuf,
}

/// Window starts of `dnsmos_local.py` for a (tiled) clip of `len` samples: one per second while
/// a whole 9.01 s window fits, `trunc(floor(len / 16000) − 9.01) + 1` of them.
fn dnsmos_windows(len: usize) -> Vec<usize> {
    let secs = (len / 16_000) as f64;
    let hops = ((secs - 9.01).trunc() as i64 + 1).max(0) as usize;
    (0..hops)
        .map(|i| i * 16_000)
        .filter(|&s| s + DNSMOS_LEN <= len)
        .collect()
}

impl Dnsmos {
    pub fn load(path: &Path, ep: &str) -> Result<Dnsmos, String> {
        let (session, outcome) = load("bench.quality.dnsmos", path, ep)?;
        Ok(Dnsmos {
            session,
            outcome,
            path: path.to_path_buf(),
        })
    }

    /// The store's `bench/dnsmos-p835` (or `explicit`); `None` when not installed.
    pub fn resolve(explicit: &str, ep: &str) -> Result<Option<Dnsmos>, String> {
        match resolve_bench_model(explicit, assets::BENCH_DNSMOS, "model.onnx")? {
            Some(p) => Dnsmos::load(&p, ep).map(Some),
            None => Ok(None),
        }
    }

    /// Raw `[SIG, BAK, OVRL]` of one 9.01 s window.
    fn raw(&mut self, window: &[f32]) -> Result<[f64; 3], String> {
        let x = Tensor::from_array(([1usize, DNSMOS_LEN], window.to_vec())).map_err(|e| format!("DNSMOS input: {e}"))?;
        let out = self
            .session
            .run(ort::inputs!["input_1" => &x])
            .map_err(|e| format!("DNSMOS: {e}"))?;
        let (_, v) = out[0].try_extract_tensor::<f32>().map_err(|e| format!("DNSMOS output: {e}"))?;
        if v.len() < 3 {
            return Err(format!("DNSMOS returned {} values", v.len()));
        }
        Ok([f64::from(v[0]), f64::from(v[1]), f64::from(v[2])])
    }

    /// Scores of a 16 kHz clip (`dnsmos_local.py`'s windowing; at least 1 s of audio).
    pub fn score(&mut self, audio_16k: &[f32]) -> Result<DnsmosScore, String> {
        if audio_16k.len() < 16_000 {
            return Err("DNSMOS: the clip is shorter than 1 s".into());
        }
        let mut audio = audio_16k.to_vec();
        while audio.len() < DNSMOS_LEN {
            audio.extend_from_within(..audio.len());
        }
        let starts = dnsmos_windows(audio.len());
        let mut acc = [0.0f64; 3];
        for &s in &starts {
            let [sig, bak, ovr] = self.raw(&audio[s..s + DNSMOS_LEN])?;
            acc[0] += poly(&POLY_SIG, sig);
            acc[1] += poly(&POLY_BAK, bak);
            acc[2] += poly(&POLY_OVRL, ovr);
        }
        let n = starts.len().max(1) as f64;
        Ok(DnsmosScore {
            sig: acc[0] / n,
            bak: acc[1] / n,
            ovrl: acc[2] / n,
            windows: starts.len(),
        })
    }
}

// ---------------------------------------------------------------------------------------------
// STT (CER)
// ---------------------------------------------------------------------------------------------

/// Which Sigil STT engine transcribes converted speech for CER (a `speech::EngineSpec`, e.g.
/// `{"engine": "whisper", "model": "C:/models/ggml-small.bin", "language": "en"}`).
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct SttSpec {
    /// `whisper`, `sherpa-offline`, `transcribe`, ... (`speech::EngineKind` wire names).
    pub engine: String,
    /// Catalog id or absolute path (as the engine takes it).
    #[serde(default)]
    pub model: String,
    #[serde(default)]
    pub language: Option<String>,
    #[serde(default)]
    pub threads: Option<u32>,
    /// Engine-specific options (never logged).
    #[serde(default)]
    pub options: serde_json::Value,
    /// What the speaker said; empty = the STT transcript of the unconverted input is the
    /// reference.
    #[serde(default)]
    pub reference_text: String,
}

/// A loaded STT engine.
pub struct Transcriber {
    engine: Box<dyn crate::services::speech::engine_api::AsrEngine>,
    language: Option<String>,
    pub description: String,
}

impl Transcriber {
    pub fn create(spec: &SttSpec) -> Result<Transcriber, String> {
        use crate::services::speech::engine_api::{EngineContext, EngineKind, EngineSpec};
        let kind: EngineKind = serde_json::from_value(serde_json::Value::String(spec.engine.trim().to_string()))
            .map_err(|_| format!("stt.engine '{}' is not a Sigil STT engine", spec.engine))?;
        let es = EngineSpec {
            engine: kind,
            model: spec.model.clone(),
            threads: spec.threads,
            variant: None,
            options: spec.options.clone(),
        };
        let engine = crate::services::speech::engines::create(&es, &EngineContext::headless()).map_err(|e| format!("STT engine: {e}"))?;
        Ok(Transcriber {
            engine,
            language: spec.language.clone().filter(|l| !l.trim().is_empty()),
            description: format!("{} {}", spec.engine, spec.model),
        })
    }

    /// Transcript of a 16 kHz clip (cut into pieces of at most 25 s, as utterances).
    pub fn transcribe(&mut self, audio_16k: &[f32]) -> Result<String, String> {
        use crate::services::speech::engine_api::DecodeOptions;
        let options = DecodeOptions {
            language: self.language.clone(),
            hotwords: Vec::new(),
            prompt: None,
        };
        let mut text = String::new();
        for piece in audio_16k.chunks(16_000 * 25) {
            let r = self.engine.transcribe(piece, &options).map_err(|e| format!("STT: {e}"))?;
            if !r.text.trim().is_empty() {
                if !text.is_empty() {
                    text.push(' ');
                }
                text.push_str(r.text.trim());
            }
        }
        Ok(text)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn dnsmos_windows_follow_dnsmos_local() {
        // Exactly 9.01 s (after tiling): one window.
        assert_eq!(dnsmos_windows(DNSMOS_LEN), vec![0]);
        // 10 s: floor(10) - 9.01 = 0.99 -> 1 window; 12.3 s: 2.99 -> 3 windows.
        assert_eq!(dnsmos_windows(160_000), vec![0]);
        assert_eq!(dnsmos_windows(196_800), vec![0, 16_000, 32_000]);
        assert!(dnsmos_windows(1_000).is_empty());
        // The polynomials map the raw model range onto 1..5 monotonically.
        assert!(poly(&POLY_OVRL, 3.0) > poly(&POLY_OVRL, 2.0));
        assert!((poly(&POLY_OVRL, 2.5) - 2.412).abs() < 0.01, "{}", poly(&POLY_OVRL, 2.5));
    }

    /// Speech frames are on the fbank grid, relative to the clip's loudest frame; the centroid of a
    /// set is the normalised mean; a file's sha256.
    #[test]
    fn speech_masks_centroids_and_stamps() {
        let mut x = vec![0.0f32; 8_000];
        x.extend((0..16_000).map(|n| (0.3 * (n as f64 * 0.05).sin()) as f32));
        x.extend(vec![1e-6f32; 8_000]);
        let m = speech_frame_mask(&x, SPEAKER_SPEECH_RANGE_DB);
        assert_eq!(m.len(), super::super::dsp::frames(x.len(), 400, 160).count());
        // Frames inside the tone are speech, the silence on both sides is not.
        assert!(!m[10] && m[60] && m[140] && !m[m.len() - 5], "{m:?}");
        let c = centroid(&[vec![1.0, 0.0], vec![0.0, 1.0]]).unwrap();
        assert!((c[0] - std::f32::consts::FRAC_1_SQRT_2).abs() < 1e-6 && (c[1] - c[0]).abs() < 1e-7);
        assert!(centroid(&[]).is_none());
        assert!(centroid(&[vec![0.0, 0.0]]).is_none());
        let p = std::env::temp_dir().join(format!("sigil-sha-{}", std::process::id()));
        std::fs::write(&p, b"abc").unwrap();
        assert_eq!(file_sha256(&p).unwrap(), "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad");
        let _ = std::fs::remove_file(&p);
        assert!(file_sha256(Path::new("Z:/no/such/file")).is_err());
    }

    /// The v2 speaker scorer on the dev rig: the same pair scored twice (two loads of the model)
    /// gives a bit-identical cosine, and two different 3.8 s utterances of one speaker score well
    /// above the different-speaker level. `SIGIL_TEST_MODELS_DIR` = the dev cache
    /// (`vc-assets/bench/campplus-exact`, `models/transcribe/samples`).
    #[test]
    #[ignore = "needs SIGIL_TEST_MODELS_DIR with vc-assets/bench/campplus-exact and models/transcribe/samples"]
    fn campplus_v2_is_deterministic() {
        let Some(root) = std::env::var_os("SIGIL_TEST_MODELS_DIR").map(PathBuf::from) else {
            return;
        };
        let clip = |name: &str| -> Vec<f32> {
            let (x, sr, ch) = super::super::signals::read_wav(&root.join("models").join("transcribe").join("samples").join(name)).unwrap();
            super::super::signals::resample(&super::super::signals::to_mono(&x, ch), sr, 16_000).unwrap()
        };
        let jfk = clip("jfk.wav");
        let (a, b) = (&jfk[4_800..4_800 + 60_800], &jfk[96_000..96_000 + 60_800]);
        let score = || {
            let mut sp = SpeakerEmbedder::resolve_v2("", "cpu").unwrap().expect("bench/campplus-exact in the dev cache");
            let mask = speech_frame_mask(a, SPEAKER_SPEECH_RANGE_DB);
            (sp.similarity_masked(a, b, &mask).unwrap(), sp.sha256.clone())
        };
        let (c1, sha) = score();
        let (c2, _) = score();
        assert_eq!(c1.to_bits(), c2.to_bits(), "{c1} vs {c2}");
        let ja = clip("ja-short.wav");
        let mut sp = SpeakerEmbedder::resolve_v2("", "cpu").unwrap().unwrap();
        let other = sp.similarity_masked(a, &ja[..60_800], &speech_frame_mask(a, SPEAKER_SPEECH_RANGE_DB)).unwrap();
        eprintln!("CAM++ v2 ({}): same speaker, different words {c1:.4}; another speaker {other:.4}", &sha[..16]);
        assert!(c1 - other > 0.3, "{c1} {other}");
    }

    #[test]
    fn cosine_and_model_resolution() {
        assert!((cosine(&[1.0, 0.0], &[1.0, 0.0]) - 1.0).abs() < 1e-12);
        assert!(cosine(&[1.0, 0.0], &[0.0, 2.0]).abs() < 1e-12);
        assert_eq!(cosine(&[0.0, 0.0], &[1.0, 1.0]), 0.0);
        assert!(resolve_bench_model("Z:/no/such/campplus.onnx", assets::BENCH_CAMPPLUS, "model.onnx").is_err());
    }

    /// The real models on the dev rig: DNSMOS ranks clean speech over the same speech at 0 dB SNR,
    /// CAM++ gives two halves of one speaker a higher cosine than two different speakers (the same
    /// checks as `export_bench.py`). `SIGIL_TEST_MODELS_DIR` = the dev cache (`vc-assets/bench/...`
    /// and `sherpa-onnx/clips/*.wav`).
    #[test]
    #[ignore = "needs SIGIL_TEST_MODELS_DIR with vc-assets/bench/{campplus,dnsmos-p835} and sherpa-onnx/clips"]
    fn real_scorers_rank_like_the_export_checks() {
        let Some(root) = std::env::var_os("SIGIL_TEST_MODELS_DIR").map(PathBuf::from) else {
            return;
        };
        let clip = |name: &str| -> Vec<f32> {
            let (x, sr, ch) = super::super::signals::read_wav(&root.join("sherpa-onnx").join("clips").join(name)).unwrap();
            let mono = super::super::signals::to_mono(&x, ch);
            super::super::signals::resample(&mono, sr, 16_000).unwrap()
        };
        let jfk = clip("en-jfk.wav");
        let mut d = Dnsmos::resolve("", "cpu").unwrap().expect("dnsmos installed");
        let clean = d.score(&jfk).unwrap();
        let mut rng = super::super::signals::SplitMix64::new(4);
        let p = jfk.iter().map(|v| f64::from(*v) * f64::from(*v)).sum::<f64>() / jfk.len() as f64;
        let noisy: Vec<f32> = jfk.iter().map(|v| v + ((3.0 * p).sqrt() * rng.range(-1.0, 1.0)) as f32).collect();
        let bad = d.score(&noisy).unwrap();
        eprintln!("DNSMOS clean {clean:?} noisy {bad:?}");
        assert!(clean.ovrl - bad.ovrl >= 0.5, "{clean:?} {bad:?}");
        assert!((1.0..=5.0).contains(&clean.ovrl));
        let mut c = SpeakerEmbedder::resolve("", "cpu").unwrap().expect("campplus installed");
        let half = jfk.len() / 2;
        let same = c.similarity(&jfk[..half], &jfk[half..]).unwrap();
        let ja = clip("ja.wav");
        let other = c.similarity(&jfk[..half], &ja).unwrap();
        eprintln!("CAM++ same {same:.3} other {other:.3}");
        assert!(same - other >= 0.2, "{same} {other}");
    }
}
