//! Test-only stand-in for a KittenTTS speech package, so the whole Kitten path (text
//! preprocessing, espeak-ng, tokenizer, NPZ voices, the ORT session on Sigil's runtime, chunking
//! and tail trim) runs offline in `cargo test` without downloading a real model.
//!
//! The ONNX graph is built with `neural_audio::test_onnx`: it takes Kitten's three
//! inputs (`input_ids` int64 `[1, seq]`, `style` f32 `[1, STYLE_DIM]`, `speed` f32 `[1]`) and returns
//! `waveform = ConstantOfShape([1, OUT_SAMPLES], LEVEL) * speed`, so a test can check that the speed
//! (times the voice's speed prior) reached the model and that every chunk was trimmed.

use std::io::Write;
use std::path::{Path, PathBuf};

use crate::services::neural_audio::test_onnx::{Attr, Dim, Graph, FLOAT, INT64};

/// Samples the fake model returns per chunk (before Kitten's 2 000-sample tail trim).
pub const OUT_SAMPLES: usize = 4800;
/// Style vector width of the fake voices.
pub const STYLE_DIM: usize = 8;
/// Sample value before the speed multiplication.
pub const LEVEL: f32 = 0.1;

/// The fake Kitten ONNX model.
pub fn model_bytes() -> Vec<u8> {
    let mut g = Graph::new();
    g.input("input_ids", INT64, &[Dim::Fixed(1), Dim::Named("seq")])
        .input("style", FLOAT, &[Dim::Fixed(1), Dim::Fixed(STYLE_DIM as u64)])
        .input("speed", FLOAT, &[Dim::Fixed(1)])
        .output("waveform", FLOAT, &[Dim::Fixed(1), Dim::Fixed(OUT_SAMPLES as u64)])
        .init_i64("out_shape", &[2], &[1, OUT_SAMPLES as i64])
        .node("ConstantOfShape", &["out_shape"], &["base"], &[Attr::FloatTensor("value", vec![1], vec![LEVEL])])
        .node("Mul", &["base", "speed"], &["waveform"], &[]);
    g.model_bytes(13)
}

/// NPY v1.0 bytes of a little-endian f32 C-order matrix.
fn npy(rows: usize, cols: usize, values: &[f32]) -> Vec<u8> {
    let mut header = format!("{{'descr': '<f4', 'fortran_order': False, 'shape': ({rows}, {cols}), }}");
    while (10 + header.len() + 1) % 64 != 0 {
        header.push(' ');
    }
    header.push('\n');
    let mut out = b"\x93NUMPY\x01\x00".to_vec();
    out.extend_from_slice(&(header.len() as u16).to_le_bytes());
    out.extend_from_slice(header.as_bytes());
    for v in values {
        out.extend_from_slice(&v.to_le_bytes());
    }
    out
}

/// A voices NPZ with one voice per name (4 style rows each).
pub fn voices_npz(names: &[&str]) -> Vec<u8> {
    let rows = 4;
    let mut zip = zip::ZipWriter::new(std::io::Cursor::new(Vec::new()));
    let options = zip::write::SimpleFileOptions::default().compression_method(zip::CompressionMethod::Stored);
    for (i, name) in names.iter().enumerate() {
        let values: Vec<f32> = (0..rows * STYLE_DIM).map(|k| (i * 100 + k) as f32 * 0.01).collect();
        zip.start_file(format!("{name}.npy"), options).expect("npz entry");
        zip.write_all(&npy(rows, STYLE_DIM, &values)).expect("npz write");
    }
    zip.finish().expect("npz finish").into_inner()
}

/// Write a complete fake speech package (`config.json`, `model.onnx`, `voices.npz`) into a fresh
/// temp folder named after `tag`. `Jasper` has speed prior 0.5, `Luna` none.
pub fn write_package(tag: &str) -> PathBuf {
    let dir = std::env::temp_dir().join(format!("sigil-fake-kitten-{tag}-{}", std::process::id()));
    let _ = std::fs::remove_dir_all(&dir);
    std::fs::create_dir_all(&dir).expect("package dir");
    write_file(&dir, "model.onnx", &model_bytes());
    write_file(&dir, "voices.npz", &voices_npz(&["Jasper", "Luna"]));
    write_file(
        &dir,
        "config.json",
        br#"{"type":"ONNX1","model_file":"model.onnx","voices":"voices.npz","speed_priors":{"Jasper":0.5},"voice_aliases":{"Jazz":"Jasper"}}"#,
    );
    dir
}

fn write_file(dir: &Path, name: &str, bytes: &[u8]) {
    std::fs::write(dir.join(name), bytes).expect("write package file");
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::services::local_tts::kitten::hub;

    #[test]
    fn fake_package_speaks_through_the_full_kitten_pipeline() {
        let dir = write_package("pipeline");
        let tts = hub::load_from_dir(&dir).expect("load the fake package");
        let mut voices = tts.available_voices.clone();
        voices.sort();
        assert_eq!(voices, ["Jasper", "Luna"]);

        // Without cleaning, two sentences are two chunks, each OUT_SAMPLES minus the 2 000-sample
        // tail trim. Upstream's cleaning strips punctuation first, so the same text is one chunk.
        let per_chunk = OUT_SAMPLES - 2000;
        let raw = tts.generate("Hello world. This is Sigil!", "Jasper", 1.0, false).expect("synthesis");
        assert_eq!(raw.len(), 2 * per_chunk);
        let audio = tts.generate("Hello world. This is Sigil on 3 CPUs!", "Jasper", 1.0, true).expect("synthesis");
        assert_eq!(audio.len(), per_chunk);
        // speed 1.0 × Jasper's prior 0.5 reached the model's `speed` input.
        assert!(audio.iter().all(|s| (s - LEVEL * 0.5).abs() < 1e-6), "{:?}", &audio[..4]);

        // Alias → Jasper; Luna has no prior.
        let alias = tts.generate("Hi.", "Jazz", 2.0, true).unwrap();
        assert!(alias.iter().all(|s| (s - LEVEL).abs() < 1e-6));
        let luna = tts.generate("Hi.", "Luna", 1.5, true).unwrap();
        assert!(luna.iter().all(|s| (s - LEVEL * 1.5).abs() < 1e-6));
        assert!(tts.generate("Hi.", "Nobody", 1.0, true).is_err());
        let _ = std::fs::remove_dir_all(&dir);
    }
}
