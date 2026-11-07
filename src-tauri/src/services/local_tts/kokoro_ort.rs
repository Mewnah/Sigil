//! Kokoro on Sigil's own ONNX Runtime, with sherpa-onnx's text frontend.
//!
//! sherpa-onnx has the best Kokoro text frontend Sigil can load (misaki's US / British lexicons,
//! the Mandarin lexicon and number / date / phone rules, C espeak-ng for everything else, its
//! sentence splitting), but its bundled ONNX Runtime busy-spins and cannot be tuned through the C
//! API. Its C API has no text-to-tokens call, so Sigil gets the tokens through a *tap*:
//!
//! 1. [`tap_model`] builds a tiny stand-in "Kokoro" graph with the real model's inputs (`tokens`,
//!    `style`, `speed`), metadata and output, whose "audio" is its input token ids as floats plus
//!    [`TAP_OFFSET`]. sherpa loads it instead of the real graph (a few KB, cached per model in
//!    `<temp>/sigil-local-tts/`), runs its whole frontend and hands the tap's output to the chunk
//!    callback. The offset keeps every value far from sherpa's silence threshold, so nothing sherpa
//!    does to "audio" can change a token; [`decode_tap`] turns the values back into token sequences.
//! 2. [`KokoroModel`] runs the real graph through `neural_audio::ort_session` (CPU EP, no spinning,
//!    2-4 intra-op threads) with the voice's style row from `voices.bin`, exactly as sherpa does
//!    ([`StyleTable::row`]), then shortens long pauses like sherpa's `silence_scale`
//!    ([`scale_silence`]).
//!
//! Measured against plain sherpa-onnx Kokoro on the dev CPU (7800X3D, fp32 v1.0, 4 threads, same
//! background load, `real_kokoro_*` tests): the same speech (correlation 0.996-0.997 with sherpa's
//! output, identical length, pauses within 0.1%) for 15-20% less CPU time. Wall time: 0.23-0.24 RTF
//! in a session with ONNX Runtime's CPU memory arena (vs 0.26-0.32 in sherpa-onnx), but today's
//! ort rc.13's `ep::CPU::default()` disables that arena (`use_arena: false`), which made every
//! sentence length the session had not seen before cost 20-35% more (0.29-0.40); this session opts
//! into it (`VcSessionOptions::cpu_arena`, P2-I0). Measured at P2 integration (7800X3D, 4 threads,
//! interleaved): median RTF 0.294 here vs 0.319 inside sherpa-onnx, new and repeated sentence
//! lengths alike, with 22% less CPU, so [`SIGIL_ORT_BY_DEFAULT`] is `true`
//! (`SIGIL_LOCAL_TTS_KOKORO=sherpa` goes back to plain sherpa-onnx). If anything in this path fails
//! (the tap does not load, unexpected tap values), the loader falls back to plain sherpa-onnx, so the
//! voice still speaks.

use std::path::{Path, PathBuf};

use ort::session::Session;
use ort::value::Tensor;

use crate::services::neural_audio::ort_session::{build_session, ModelSource, VcSessionOptions, EP_CPU};

/// Added to every token id in the tap's output (token ids are < 512, far from sherpa's 0.01
/// silence threshold and exact in f32).
pub const TAP_OFFSET: f32 = 1000.0;
/// Kokoro's padding token (`$`), at both ends of every model input.
pub const PAD: i64 = 0;
/// Largest token id accepted from the tap (Kokoro's vocabulary has 178 symbols).
const MAX_TOKEN: i64 = 511;
/// Style vector width of Kokoro voices.
pub const STYLE_DIM: usize = 256;
/// Diagnostics component the session's EP outcome is recorded under (shown as "Local TTS").
pub const EP_COMPONENT: &str = "tts.kitten";
/// Environment switch: `sigil` runs Kokoro's graph on Sigil's ONNX Runtime (this module), `sherpa`
/// entirely inside sherpa-onnx; unset = [`SIGIL_ORT_BY_DEFAULT`].
pub const ENV_KOKORO_RUNTIME: &str = "SIGIL_LOCAL_TTS_KOKORO";
/// Whether Kokoro runs on Sigil's ONNX Runtime by default: yes since the session uses ONNX
/// Runtime's CPU memory arena (P2-I0), where `real_kokoro_rtf_on_four_threads_and_first_audio`
/// measured it faster than sherpa-onnx on new sentence lengths.
pub const SIGIL_ORT_BY_DEFAULT: bool = true;
/// Samples at or below this magnitude count as silence ([`scale_silence`], as in sherpa-onnx).
pub const SILENCE_LEVEL: f32 = 0.01;
/// Shortest silence [`scale_silence`] shortens (seconds, as in sherpa-onnx).
pub const MIN_PAUSE_S: f32 = 0.2;

/// Whether Kokoro's graph runs on Sigil's ONNX Runtime: [`ENV_KOKORO_RUNTIME`], else
/// [`SIGIL_ORT_BY_DEFAULT`].
pub fn use_sigil_runtime() -> bool {
    runtime_choice(std::env::var(ENV_KOKORO_RUNTIME).ok().as_deref())
}

/// [`use_sigil_runtime`] for a given value of the environment variable.
pub fn runtime_choice(value: Option<&str>) -> bool {
    match value.map(|v| v.trim().to_ascii_lowercase()).as_deref() {
        Some("sigil" | "ort" | "sigil-ort") => true,
        Some("sherpa" | "sherpa-onnx") => false,
        _ => SIGIL_ORT_BY_DEFAULT,
    }
}

// -------------------------------------------------------------------------------------------------
// Tap graph (hand-encoded ONNX protobuf; field numbers follow onnx.proto)
// -------------------------------------------------------------------------------------------------

const FLOAT: u64 = 1;
const INT64: u64 = 7;
/// `AttributeProto.AttributeType.INT`.
const ATTR_INT: u64 = 2;

fn varint(out: &mut Vec<u8>, mut v: u64) {
    while v >= 0x80 {
        out.push((v as u8) | 0x80);
        v >>= 7;
    }
    out.push(v as u8);
}

fn field_varint(out: &mut Vec<u8>, field: u32, v: u64) {
    varint(out, u64::from(field) << 3);
    varint(out, v);
}

fn field_bytes(out: &mut Vec<u8>, field: u32, bytes: &[u8]) {
    varint(out, (u64::from(field) << 3) | 2);
    varint(out, bytes.len() as u64);
    out.extend_from_slice(bytes);
}

/// `ValueInfoProto` of a tensor with named (symbolic) dimensions.
fn value_info(name: &str, elem_type: u64, dims: &[&str]) -> Vec<u8> {
    let mut shape = Vec::new();
    for d in dims {
        let mut dim = Vec::new();
        field_bytes(&mut dim, 2, d.as_bytes()); // dim_param
        field_bytes(&mut shape, 1, &dim);
    }
    let mut tensor = Vec::new();
    field_varint(&mut tensor, 1, elem_type);
    field_bytes(&mut tensor, 2, &shape);
    let mut ty = Vec::new();
    field_bytes(&mut ty, 1, &tensor); // tensor_type
    let mut v = Vec::new();
    field_bytes(&mut v, 1, name.as_bytes());
    field_bytes(&mut v, 2, &ty);
    v
}

fn initializer(name: &str, data_type: u64, dims: &[u64], raw: &[u8]) -> Vec<u8> {
    let mut t = Vec::new();
    if !dims.is_empty() {
        let mut packed = Vec::new();
        for &d in dims {
            varint(&mut packed, d);
        }
        field_bytes(&mut t, 1, &packed);
    }
    field_varint(&mut t, 2, data_type);
    field_bytes(&mut t, 8, name.as_bytes());
    field_bytes(&mut t, 9, raw);
    t
}

fn node(op: &str, inputs: &[&str], outputs: &[&str], int_attrs: &[(&str, i64)]) -> Vec<u8> {
    let mut n = Vec::new();
    for i in inputs {
        field_bytes(&mut n, 1, i.as_bytes());
    }
    for o in outputs {
        field_bytes(&mut n, 2, o.as_bytes());
    }
    field_bytes(&mut n, 3, format!("tap_{op}_{}", outputs.first().copied().unwrap_or("")).as_bytes());
    field_bytes(&mut n, 4, op.as_bytes());
    for (name, v) in int_attrs {
        let mut a = Vec::new();
        field_bytes(&mut a, 1, name.as_bytes());
        field_varint(&mut a, 3, *v as u64);
        field_varint(&mut a, 20, ATTR_INT);
        field_bytes(&mut n, 5, &a);
    }
    n
}

/// The tap graph (see the module docs): inputs `tokens` (int64 `[1, n]`), `style` (float
/// `[1, 256]`), `speed` (float `[1]`) as the real Kokoro graph has them, and one float output
/// `audio` (`[n]`) = `tokens + TAP_OFFSET + 0 * (sum(style) + speed)` (every input is used, so ONNX
/// Runtime keeps them all). `metadata` is copied from the real model (sherpa-onnx reads the sample
/// rate, speaker count, voice, style shape, ... from it); `audio_name` is the real graph's output.
pub fn tap_model(metadata: &[(String, String)], audio_name: &str) -> Vec<u8> {
    let mut graph = Vec::new();
    let nodes = [
        node("Reshape", &["tokens", "tap_flat_shape"], &["tap_flat"], &[]),
        node("Cast", &["tap_flat"], &["tap_ids"], &[("to", FLOAT as i64)]),
        node("ReduceSum", &["style"], &["tap_style_sum"], &[("keepdims", 0)]),
        node("Add", &["tap_style_sum", "speed"], &["tap_inputs"], &[]),
        node("Mul", &["tap_inputs", "tap_zero"], &["tap_nothing"], &[]),
        node("Add", &["tap_ids", "tap_offset"], &["tap_shifted"], &[]),
        node("Add", &["tap_shifted", "tap_nothing"], &[audio_name], &[]),
    ];
    for n in &nodes {
        field_bytes(&mut graph, 1, n);
    }
    field_bytes(&mut graph, 2, b"sigil_kokoro_token_tap");
    let flat: Vec<u8> = (-1i64).to_le_bytes().to_vec();
    field_bytes(&mut graph, 5, &initializer("tap_flat_shape", INT64, &[1], &flat));
    field_bytes(&mut graph, 5, &initializer("tap_offset", FLOAT, &[], &TAP_OFFSET.to_le_bytes()));
    field_bytes(&mut graph, 5, &initializer("tap_zero", FLOAT, &[], &0f32.to_le_bytes()));
    field_bytes(&mut graph, 11, &value_info("tokens", INT64, &["batch", "num_tokens"]));
    field_bytes(&mut graph, 11, &value_info("style", FLOAT, &["batch", "style_dim"]));
    field_bytes(&mut graph, 11, &value_info("speed", FLOAT, &["one"]));
    field_bytes(&mut graph, 12, &value_info(audio_name, FLOAT, &["num_samples"]));

    let mut model = Vec::new();
    field_varint(&mut model, 1, 8); // ir_version
    field_bytes(&mut model, 2, b"sigil-local-tts"); // producer_name
    field_bytes(&mut model, 7, &graph);
    let mut opset = Vec::new();
    field_varint(&mut opset, 2, 13);
    field_bytes(&mut model, 8, &opset);
    for (k, v) in metadata {
        let mut entry = Vec::new();
        field_bytes(&mut entry, 1, k.as_bytes());
        field_bytes(&mut entry, 2, v.as_bytes());
        field_bytes(&mut model, 14, &entry);
    }
    model
}

/// FNV-1a, for the tap file name (stable across runs, unlike `DefaultHasher`).
fn fnv1a(bytes: &[u8]) -> u64 {
    let mut h: u64 = 0xcbf2_9ce4_8422_2325;
    for b in bytes {
        h ^= u64::from(*b);
        h = h.wrapping_mul(0x0000_0100_0000_01b3);
    }
    h
}

/// Write the tap for `metadata` into `dir` (once; the name is a hash of its bytes) and return its
/// path.
pub fn write_tap(dir: &Path, metadata: &[(String, String)], audio_name: &str) -> Result<PathBuf, String> {
    let bytes = tap_model(metadata, audio_name);
    let path = dir.join(format!("kokoro-tap-{:016x}.onnx", fnv1a(&bytes)));
    if std::fs::read(&path).is_ok_and(|b| b == bytes) {
        return Ok(path);
    }
    std::fs::create_dir_all(dir).map_err(|e| format!("cannot create {}: {e}", dir.display()))?;
    // Write under a unique name, then rename: another process may be writing the same tap.
    let tmp = dir.join(format!("kokoro-tap-{}-{:x}.tmp", std::process::id(), fnv1a(path.as_os_str().as_encoded_bytes())));
    std::fs::write(&tmp, &bytes).map_err(|e| format!("cannot write {}: {e}", tmp.display()))?;
    if let Err(e) = std::fs::rename(&tmp, &path) {
        let _ = std::fs::remove_file(&tmp);
        // A concurrent writer may have put identical bytes there first.
        if !std::fs::read(&path).is_ok_and(|b| b == bytes) {
            return Err(format!("cannot write {}: {e}", path.display()));
        }
    }
    Ok(path)
}

/// Default folder of the tap files.
pub fn tap_dir() -> PathBuf {
    std::env::temp_dir().join("sigil-local-tts")
}

/// Token sequences from the tap's output: values below `TAP_OFFSET` (silence sherpa inserted) are
/// skipped; the rest are token ids, split at the padding token into `[PAD, tokens.., PAD]`
/// sequences. `Err` for a value that is not a token (the tap is not what sherpa ran).
pub fn decode_tap(samples: &[f32]) -> Result<Vec<Vec<i64>>, String> {
    let mut ids = Vec::with_capacity(samples.len());
    for &x in samples {
        if !x.is_finite() {
            return Err("the token tap returned a non-finite value".into());
        }
        if x < TAP_OFFSET - 0.5 {
            continue;
        }
        let id = (x - TAP_OFFSET).round();
        if (x - TAP_OFFSET - id).abs() > 1e-3 || id < 0.0 || id > MAX_TOKEN as f32 {
            return Err(format!("the token tap returned {x}, which is not a token id"));
        }
        ids.push(id as i64);
    }
    Ok(ids
        .split(|t| *t == PAD)
        .filter(|s| !s.is_empty())
        .map(|s| {
            let mut v = Vec::with_capacity(s.len() + 2);
            v.push(PAD);
            v.extend_from_slice(s);
            v.push(PAD);
            v
        })
        .collect())
}

// -------------------------------------------------------------------------------------------------
// Style table (voices.bin)
// -------------------------------------------------------------------------------------------------

/// A Kokoro `voices.bin`: `[speakers][rows][256]` little-endian f32 (row = style for an input of
/// that many tokens).
#[derive(Clone, Debug, PartialEq)]
pub struct StyleTable {
    data: Vec<f32>,
    speakers: usize,
    rows: usize,
}

impl StyleTable {
    /// Parse `bytes` for `speakers` voices.
    pub fn parse(bytes: &[u8], speakers: usize) -> Result<Self, String> {
        if speakers == 0 {
            return Err("the Kokoro model lists no voices".into());
        }
        if !bytes.len().is_multiple_of(4) {
            return Err(format!("voices.bin is not whole f32 values ({} bytes)", bytes.len()));
        }
        let data: Vec<f32> = bytes
            .as_chunks::<4>().0.iter()
            .map(|b| f32::from_le_bytes([b[0], b[1], b[2], b[3]]))
            .collect();
        let per_speaker = data.len() / speakers;
        if !data.len().is_multiple_of(speakers) || per_speaker == 0 || !per_speaker.is_multiple_of(STYLE_DIM) {
            return Err(format!(
                "voices.bin holds {} values, not {speakers} voices of N x {STYLE_DIM}",
                data.len()
            ));
        }
        Ok(Self {
            rows: per_speaker / STYLE_DIM,
            data,
            speakers,
        })
    }

    pub fn load(path: &Path, speakers: usize) -> Result<Self, String> {
        let bytes = std::fs::read(path).map_err(|e| format!("cannot read {}: {e}", path.display()))?;
        Self::parse(&bytes, speakers)
    }

    #[cfg(test)]
    pub fn speakers(&self) -> usize {
        self.speakers
    }

    #[cfg(test)]
    pub fn rows(&self) -> usize {
        self.rows
    }

    /// The style of speaker `sid` for a model input of `input_len` tokens (padding included), as
    /// sherpa-onnx picks it: row `input_len - 2` (the tokens without padding), capped at the last
    /// row.
    pub fn row(&self, sid: usize, input_len: usize) -> Result<&[f32], String> {
        if sid >= self.speakers {
            return Err(format!("voice {sid} does not exist (this model has {} voices)", self.speakers));
        }
        let r = input_len.saturating_sub(2).min(self.rows - 1);
        let at = (sid * self.rows + r) * STYLE_DIM;
        Ok(&self.data[at..at + STYLE_DIM])
    }
}

// -------------------------------------------------------------------------------------------------
// Silence scaling (sherpa-onnx `GeneratedAudio::ScaleSilence`)
// -------------------------------------------------------------------------------------------------

/// Shorten every pause (a run of samples within ±[`SILENCE_LEVEL`] lasting at least
/// [`MIN_PAUSE_S`], including a leading or trailing one) to `scale` of its length, keeping its
/// start, as sherpa-onnx does with its `silence_scale`. `scale` outside `(0, 1)` returns the audio
/// unchanged (Sigil only shortens).
pub fn scale_silence(samples: &[f32], sample_rate: u32, scale: f32) -> Vec<f32> {
    if !(scale.is_finite() && scale > 0.0 && scale < 1.0) || samples.is_empty() {
        return samples.to_vec();
    }
    let min_len = ((sample_rate as f32 * MIN_PAUSE_S) as usize).max(1);
    let mut out = Vec::with_capacity(samples.len());
    let mut i = 0;
    while i < samples.len() {
        if samples[i].abs() > SILENCE_LEVEL {
            out.push(samples[i]);
            i += 1;
            continue;
        }
        let start = i;
        while i < samples.len() && samples[i].abs() <= SILENCE_LEVEL {
            i += 1;
        }
        let run = &samples[start..i];
        if run.len() >= min_len {
            let keep = ((run.len() as f32 * scale) as usize).max(1);
            out.extend_from_slice(&run[..keep]);
        } else {
            out.extend_from_slice(run);
        }
    }
    out
}

// -------------------------------------------------------------------------------------------------
// The Kokoro graph on Sigil's ONNX Runtime
// -------------------------------------------------------------------------------------------------

/// The real Kokoro graph on Sigil's ONNX Runtime (CPU).
pub struct KokoroModel {
    session: Session,
    styles: StyleTable,
}

impl KokoroModel {
    /// Load `model` (the real Kokoro graph) with the styles of `voices` for `speakers` voices, on
    /// `threads` intra-op threads.
    pub fn load(model: &Path, voices: &Path, speakers: usize, threads: u32) -> Result<Self, String> {
        let styles = StyleTable::load(voices, speakers)?;
        let opts = VcSessionOptions {
            intra_threads: Some(threads.clamp(1, 8) as usize),
            // Every sentence length is a new input shape: without the arena each costs 20-35% more.
            cpu_arena: true,
            ..VcSessionOptions::default()
        };
        let (session, _outcome) = build_session(Some(EP_COMPONENT), EP_CPU, &opts, ModelSource::File(model))
            .map_err(|e| format!("cannot load the Kokoro model: {e}"))?;
        let names: Vec<&str> = session.inputs().iter().map(|i| i.name()).collect();
        if names != ["tokens", "style", "speed"] {
            return Err(format!("unexpected Kokoro model inputs {names:?} (expected tokens, style, speed)"));
        }
        Ok(Self { session, styles })
    }

    /// Synthesize one token sequence (`[PAD, .., PAD]`) with speaker `sid` at `speed`.
    pub fn run(&mut self, tokens: &[i64], sid: u32, speed: f32) -> Result<Vec<f32>, String> {
        if tokens.len() < 3 {
            return Ok(Vec::new());
        }
        let style = self.styles.row(sid as usize, tokens.len())?.to_vec();
        self.run_with_style(tokens, style, speed)
    }

    /// Synthesize one token sequence with an explicit style vector ([`STYLE_DIM`] values).
    pub fn run_with_style(&mut self, tokens: &[i64], style: Vec<f32>, speed: f32) -> Result<Vec<f32>, String> {
        if style.len() != STYLE_DIM {
            return Err(format!("a Kokoro style has {STYLE_DIM} values, not {}", style.len()));
        }
        let t_tokens = Tensor::<i64>::from_array(([1usize, tokens.len()], tokens.to_vec())).map_err(|e| e.to_string())?;
        let t_style = Tensor::<f32>::from_array(([1usize, STYLE_DIM], style)).map_err(|e| e.to_string())?;
        let t_speed = Tensor::<f32>::from_array(([1usize], vec![speed])).map_err(|e| e.to_string())?;
        let outputs = self
            .session
            .run(ort::inputs![t_tokens, t_style, t_speed])
            .map_err(|e| format!("Kokoro inference failed: {e}"))?;
        let (_shape, audio) = outputs[0]
            .try_extract_tensor::<f32>()
            .map_err(|e| format!("Kokoro returned no audio: {e}"))?;
        Ok(audio.to_vec())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_tap_returns_its_tokens_on_onnx_runtime() {
        let meta = vec![
            ("model_type".to_string(), "kokoro".to_string()),
            ("sample_rate".to_string(), "24000".to_string()),
        ];
        let bytes = tap_model(&meta, "audio");
        let (mut session, _) = build_session(None, EP_CPU, &VcSessionOptions::default(), ModelSource::Memory(&bytes))
            .expect("the tap loads");
        let names: Vec<&str> = session.inputs().iter().map(|i| i.name()).collect();
        assert_eq!(names, ["tokens", "style", "speed"], "same inputs as Kokoro");
        let tokens = vec![0i64, 50, 83, 177, 0];
        let out = session
            .run(ort::inputs![
                Tensor::<i64>::from_array(([1usize, tokens.len()], tokens.clone())).unwrap(),
                Tensor::<f32>::from_array(([1usize, STYLE_DIM], vec![0.25f32; STYLE_DIM])).unwrap(),
                Tensor::<f32>::from_array(([1usize], vec![1.3f32])).unwrap()
            ])
            .unwrap();
        let (_, audio) = out[0].try_extract_tensor::<f32>().unwrap();
        assert_eq!(audio.to_vec(), [1000.0, 1050.0, 1083.0, 1177.0, 1000.0]);
        assert_eq!(decode_tap(audio).unwrap(), [tokens]);
    }

    #[test]
    fn tap_files_are_written_once_per_metadata() {
        let dir = std::env::temp_dir().join(format!("sigil-kokoro-tap-test-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&dir);
        let a = vec![("n_speakers".to_string(), "54".to_string())];
        let b = vec![("n_speakers".to_string(), "103".to_string())];
        let pa = write_tap(&dir, &a, "audio").unwrap();
        assert_eq!(write_tap(&dir, &a, "audio").unwrap(), pa, "same metadata, same file");
        let pb = write_tap(&dir, &b, "audio").unwrap();
        assert_ne!(pa, pb);
        assert_eq!(std::fs::read(&pa).unwrap(), tap_model(&a, "audio"));
        // A damaged file is replaced under the same name.
        std::fs::write(&pa, b"junk").unwrap();
        assert!(write_tap(&dir, &a, "audio").is_ok());
        assert_eq!(std::fs::read(&pa).unwrap(), tap_model(&a, "audio"));
        let leftovers = std::fs::read_dir(&dir)
            .unwrap()
            .filter(|e| e.as_ref().unwrap().path().extension().is_some_and(|x| x == "tmp"))
            .count();
        assert_eq!(leftovers, 0);
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn tap_output_decodes_into_padded_sequences() {
        let t = |ids: &[i64]| ids.iter().map(|i| *i as f32 + TAP_OFFSET).collect::<Vec<f32>>();
        // Two sentences in one batch, with silence sherpa may put between them.
        let mut samples = t(&[0, 5, 6, 0]);
        samples.extend([0.0, 0.0, 0.004]);
        samples.extend(t(&[0, 7, 0]));
        assert_eq!(decode_tap(&samples).unwrap(), [vec![0, 5, 6, 0], vec![0, 7, 0]]);
        // Missing padding is added; an empty sentence disappears.
        assert_eq!(decode_tap(&t(&[9, 10, 0, 0, 0])).unwrap(), [vec![0, 9, 10, 0]]);
        assert!(decode_tap(&[]).unwrap().is_empty());
        // Anything that is not a token id means the tap is not what ran.
        assert!(decode_tap(&[1000.4]).is_err());
        assert!(decode_tap(&[f32::NAN]).is_err());
        assert!(decode_tap(&[TAP_OFFSET + 600.0]).is_err());
    }

    #[test]
    fn style_rows_follow_the_token_count() {
        // 2 voices x 3 rows x 256: value = voice * 10 + row.
        let mut bytes = Vec::new();
        for v in 0..2 {
            for r in 0..3 {
                for _ in 0..STYLE_DIM {
                    bytes.extend_from_slice(&((v * 10 + r) as f32).to_le_bytes());
                }
            }
        }
        let t = StyleTable::parse(&bytes, 2).unwrap();
        assert_eq!((t.speakers(), t.rows()), (2, 3));
        assert_eq!(t.row(0, 3).unwrap()[0], 1.0, "one token plus padding → row 1");
        assert_eq!(t.row(1, 4).unwrap()[0], 12.0);
        assert_eq!(t.row(1, 400).unwrap()[0], 12.0, "capped at the last row");
        assert_eq!(t.row(0, 0).unwrap()[0], 0.0);
        assert!(t.row(2, 3).is_err(), "no third voice");
        assert!(StyleTable::parse(&bytes, 0).is_err());
        assert!(StyleTable::parse(&bytes[..bytes.len() - 4], 2).is_err());
        assert!(StyleTable::parse(&bytes[..6], 1).is_err());
    }

    #[test]
    fn the_runtime_switch_is_parsed() {
        assert!(runtime_choice(Some("sigil")));
        assert!(runtime_choice(Some(" ORT ")));
        assert!(!runtime_choice(Some("sherpa")));
        assert!(!runtime_choice(Some("Sherpa-ONNX")));
        assert_eq!(runtime_choice(None), SIGIL_ORT_BY_DEFAULT);
        assert_eq!(runtime_choice(Some("something else")), SIGIL_ORT_BY_DEFAULT);
    }

    #[test]
    fn long_pauses_are_shortened_like_sherpa() {
        let rate = 1000; // 0.2 s = 200 samples
        let voice = |n: usize| vec![0.5f32; n];
        let quiet = |n: usize| vec![0.005f32; n];
        // Leading pause (300), speech, short pause (100, kept), speech, long pause (500), speech,
        // trailing pause (250).
        let mut a = quiet(300);
        a.extend(voice(50));
        a.extend(quiet(100));
        a.extend(voice(50));
        a.extend(quiet(500));
        a.extend(voice(50));
        a.extend(quiet(250));
        let out = scale_silence(&a, rate, 0.2);
        assert_eq!(out.len(), 60 + 50 + 100 + 50 + 100 + 50 + 50);
        assert_eq!(&out[..60], &quiet(60)[..]);
        assert_eq!(out[60], 0.5);
        // Unchanged for scales that do not shorten, and for empty input.
        assert_eq!(scale_silence(&a, rate, 1.0), a);
        assert_eq!(scale_silence(&a, rate, f32::NAN), a);
        assert!(scale_silence(&[], rate, 0.2).is_empty());
        // Exactly the threshold counts as silence.
        assert_eq!(scale_silence(&[0.01; 400], rate, 0.5).len(), 200);
    }
}
