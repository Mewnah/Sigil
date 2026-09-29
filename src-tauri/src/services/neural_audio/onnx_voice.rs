//! ONNX voice-chunk inference: float32 tensor in → float32 tensor out.
//! Input/output ranks `[1, N]` or `[N]` are supported. Shapes are read from the model;
//! when `N` differs from the live mic chunk (`chunk_frames * input_channels`), samples are
//! adapted (mono/stereo and linear resampling).

use crate::services::neural_audio::ort_session::session_builder_with_ep;

use ndarray::{Array1, Array2};
use ort::session::Session;
use ort::tensor::TensorElementType;
use ort::value::{Outlet, TensorRef, ValueType};

const MAX_MODEL_CHANNELS: usize = 8;
// Stereo models must stay supported (compile-time check).
const _: () = assert!(MAX_MODEL_CHANNELS >= 2);

/// Element count for VC I/O from ONNX dimensions; dynamic axes fall back to `runtime_need`.
fn element_count_for_io(dims: &[i64], runtime_need: usize) -> usize {
    if !dims.is_empty() && dims.iter().all(|&d| d >= 0) {
        return dims.iter().map(|&d| d as usize).product();
    }
    match dims {
        // Dynamic batch, static length.
        [_, b] if *b >= 0 => *b as usize,
        _ => runtime_need,
    }
}

/// Concrete ORT input shape for this run: dynamic batch → `1`, dynamic last dim → `n_in`.
fn concrete_input_shape(original: &[i64], n_in: usize) -> Result<Vec<i64>, String> {
    let n = n_in as i64;
    let out: Vec<i64> = match original {
        [a, b] => {
            let row = if *a < 0 { 1 } else { *a };
            let col = if *b < 0 { n } else { *b };
            vec![row, col]
        }
        [a] => {
            let d = if *a < 0 { n } else { *a };
            vec![d]
        }
        dims => {
            let mut o = Vec::with_capacity(dims.len());
            let mut neg = 0i32;
            for &d in dims {
                if d < 0 {
                    o.push(n);
                    neg += 1;
                } else {
                    o.push(d);
                }
            }
            if neg > 1 {
                return Err(
                    "ONNX VC input has multiple dynamic dimensions; only one is supported.".into(),
                );
            }
            o
        }
    };
    let product: usize = out
        .iter()
        .map(|&d| d.max(0) as usize)
        .try_fold(1usize, |acc, x| acc.checked_mul(x))
        .ok_or_else(|| "ONNX VC input shape overflow".to_string())?;
    if product != n_in {
        return Err(format!(
            "ONNX VC input shape {:?} does not match adapted length {} (product {})",
            out, n_in, product
        ));
    }
    Ok(out)
}

fn require_f32_tensor<'a>(outlet: &'a Outlet, label: &str) -> Result<&'a [i64], String> {
    match outlet.dtype() {
        ValueType::Tensor { ty, shape, .. } => {
            if *ty != TensorElementType::Float32 {
                return Err(format!(
                    "ONNX VC {} must be float32, got {:?}",
                    label, ty
                ));
            }
            Ok(shape.as_ref())
        }
        other => Err(format!(
            "ONNX VC {} must be a tensor, got {:?}",
            label, other
        )),
    }
}

fn deinterleave(interleaved: &[f32], ch: usize, frames: usize) -> Option<Vec<Vec<f32>>> {
    if ch == 0 || interleaved.len() != ch.checked_mul(frames)? {
        return None;
    }
    let mut out: Vec<Vec<f32>> = (0..ch).map(|_| vec![0.0; frames]).collect();
    for f in 0..frames {
        for c in 0..ch {
            out[c][f] = interleaved[f * ch + c];
        }
    }
    Some(out)
}

fn interleave(chans: &[Vec<f32>], ch: usize, frames: usize) -> Option<Vec<f32>> {
    if chans.len() != ch || chans.iter().any(|v| v.len() != frames) {
        return None;
    }
    let mut out = vec![0.0; ch.checked_mul(frames)?];
    for f in 0..frames {
        for c in 0..ch {
            out[f * ch + c] = chans[c][f];
        }
    }
    Some(out)
}

fn linear_resample_1d(src: &[f32], dst_len: usize) -> Vec<f32> {
    if dst_len == 0 {
        return Vec::new();
    }
    if src.is_empty() {
        return vec![0.0; dst_len];
    }
    if src.len() == dst_len {
        return src.to_vec();
    }
    let mut out = vec![0.0; dst_len];
    if dst_len == 1 {
        out[0] = src[src.len() / 2];
        return out;
    }
    for (i, o) in out.iter_mut().enumerate() {
        let t = i as f32 * (src.len() - 1) as f32 / (dst_len - 1) as f32;
        let j0 = t.floor() as usize;
        let j1 = (j0 + 1).min(src.len() - 1);
        let f = t - j0 as f32;
        *o = src[j0] * (1.0 - f) + src[j1] * f;
    }
    out
}

fn convert_channels(chans: Vec<Vec<f32>>, from_ch: usize, to_ch: usize) -> Option<Vec<Vec<f32>>> {
    if from_ch != chans.len() {
        return None;
    }
    let frames = chans.first()?.len();
    if chans.iter().any(|v| v.len() != frames) {
        return None;
    }
    if from_ch == to_ch {
        return Some(chans);
    }
    if from_ch == 2 && to_ch == 1 {
        let l = &chans[0];
        let r = &chans[1];
        let mono: Vec<f32> = l
            .iter()
            .zip(r.iter())
            .map(|(&a, &b)| 0.5 * (a + b))
            .collect();
        return Some(vec![mono]);
    }
    if from_ch == 1 && to_ch == 2 {
        let m = chans[0].clone();
        return Some(vec![m.clone(), m]);
    }
    None
}

fn adapt_runtime_to_model(
    interleaved: &[f32],
    runtime_need: usize,
    ch_in: usize,
    n_in: usize,
    frames_rt: usize,
) -> Vec<f32> {
    debug_assert_eq!(interleaved.len(), runtime_need);
    if n_in == runtime_need {
        return interleaved.to_vec();
    }
    if frames_rt == 0 || runtime_need != ch_in.saturating_mul(frames_rt) {
        return linear_resample_1d(interleaved, n_in);
    }
    if n_in % frames_rt != 0 {
        return linear_resample_1d(interleaved, n_in);
    }
    let ch_m = n_in / frames_rt;
    if ch_m == 0 || ch_m > MAX_MODEL_CHANNELS {
        return linear_resample_1d(interleaved, n_in);
    }
    let Some(chans) = deinterleave(interleaved, ch_in, frames_rt) else {
        return linear_resample_1d(interleaved, n_in);
    };
    let chans = if let Some(c) = convert_channels(chans, ch_in, ch_m) {
        c
    } else {
        return linear_resample_1d(interleaved, n_in);
    };
    interleave(&chans, ch_m, frames_rt).unwrap_or_else(|| linear_resample_1d(interleaved, n_in))
}

fn adapt_model_to_runtime(
    model: &[f32],
    n_out: usize,
    ch_in: usize,
    runtime_need: usize,
    frames_rt: usize,
) -> Vec<f32> {
    if n_out == runtime_need {
        return model.to_vec();
    }
    if frames_rt == 0 || runtime_need != ch_in.saturating_mul(frames_rt) {
        return linear_resample_1d(model, runtime_need);
    }
    if n_out % frames_rt != 0 {
        return linear_resample_1d(model, runtime_need);
    }
    let ch_m = n_out / frames_rt;
    if ch_m == 0 || ch_m > MAX_MODEL_CHANNELS {
        return linear_resample_1d(model, runtime_need);
    }
    let Some(chans) = deinterleave(model, ch_m, frames_rt) else {
        return linear_resample_1d(model, runtime_need);
    };
    if let Some(converted) = convert_channels(chans, ch_m, ch_in) {
        interleave(&converted, ch_in, frames_rt).unwrap_or_else(|| linear_resample_1d(model, runtime_need))
    } else {
        linear_resample_1d(model, runtime_need)
    }
}

pub struct OnnxVoiceBridge {
    session: Session,
    runtime_need: usize,
    ch_in: usize,
    n_in: usize,
    input_shape_concrete: Vec<i64>,
    chunk_frames: usize,
}

impl OnnxVoiceBridge {
    pub fn new(
        model_path: &str,
        runtime_need: usize,
        ch_in: usize,
        execution_provider_hint: &str,
    ) -> Result<Self, String> {
        let path = model_path.trim();
        if path.is_empty() {
            return Err("ONNX model path is empty.".to_string());
        }
        if ch_in == 0 {
            return Err("ONNX VC: input channel count is zero.".to_string());
        }
        let chunk_frames = runtime_need / ch_in;
        if chunk_frames * ch_in != runtime_need {
            return Err(format!(
                "ONNX VC: runtime_need {} must be divisible by channel count {}",
                runtime_need, ch_in
            ));
        }

        let session = session_builder_with_ep(execution_provider_hint)?
            .commit_from_file(path)
            .map_err(|e| format!("Failed to load ONNX model {:?}: {}", path, e))?;

        let inputs = session.inputs();
        let outputs = session.outputs();
        if inputs.is_empty() {
            return Err("ONNX VC model has no inputs.".to_string());
        }
        if outputs.is_empty() {
            return Err("ONNX VC model has no outputs.".to_string());
        }

        let in_dims = require_f32_tensor(&inputs[0], "input")?;
        let out_dims = require_f32_tensor(&outputs[0], "output")?;

        let n_in = element_count_for_io(in_dims, runtime_need);
        let n_out_meta = element_count_for_io(out_dims, runtime_need);
        if n_in == 0 {
            return Err("ONNX VC input element count resolved to zero.".to_string());
        }

        let input_shape_concrete = concrete_input_shape(in_dims, n_in)?;

        let ch_m_in = n_in / chunk_frames;
        let uses_aligned =
            n_in % chunk_frames == 0 && ch_m_in > 0 && ch_m_in <= MAX_MODEL_CHANNELS;
        let adapting_in = n_in != runtime_need;
        let hint_out = if n_out_meta != runtime_need {
            "Output size will be taken from each run and adapted."
        } else {
            "Output matches mic chunk."
        };
        if adapting_in && uses_aligned && ch_m_in != ch_in {
            eprintln!(
                "[VoiceChanger] ONNX: model expects {} samples ({} ch × {} frames); mic is {} ch ({} samples). Downmix/upmix for inference. {}",
                n_in,
                ch_m_in,
                chunk_frames,
                ch_in,
                runtime_need,
                hint_out
            );
        } else if adapting_in {
            eprintln!(
                "[VoiceChanger] ONNX: model input {} samples vs mic {}; linear resampling. {}",
                n_in, runtime_need, hint_out
            );
        } else {
            eprintln!(
                "[VoiceChanger] ONNX: input/output element size {} ({} ch). {}",
                runtime_need, ch_in, hint_out
            );
        }

        Ok(Self {
            session,
            runtime_need,
            ch_in,
            n_in,
            input_shape_concrete,
            chunk_frames,
        })
    }

    /// `interleaved` must be `runtime_need` samples (interleaved per frame).
    pub fn infer(&mut self, interleaved: &[f32]) -> Result<Vec<f32>, String> {
        if interleaved.len() != self.runtime_need {
            return Err(format!(
                "ONNX VC expected {} samples per chunk, got {}",
                self.runtime_need,
                interleaved.len()
            ));
        }
        let model_in = adapt_runtime_to_model(
            interleaved,
            self.runtime_need,
            self.ch_in,
            self.n_in,
            self.chunk_frames,
        );
        if model_in.len() != self.n_in {
            return Err(format!(
                "ONNX VC internal: adapted input length {} != model {}",
                model_in.len(),
                self.n_in
            ));
        }

        let outputs = match self.input_shape_concrete.len() {
            1 => {
                let arr = Array1::from_shape_vec((self.n_in,), model_in).map_err(|e| e.to_string())?;
                let tensor_ref = TensorRef::from_array_view(&arr).map_err(|e| e.to_string())?;
                self.session
                    .run(ort::inputs![tensor_ref])
                    .map_err(|e| format!("ONNX inference: {}", e))?
            }
            2 => {
                let d0 = self.input_shape_concrete[0] as usize;
                let d1 = self.input_shape_concrete[1] as usize;
                let arr = Array2::from_shape_vec((d0, d1), model_in).map_err(|e| e.to_string())?;
                let tensor_ref = TensorRef::from_array_view(&arr).map_err(|e| e.to_string())?;
                self.session
                    .run(ort::inputs![tensor_ref])
                    .map_err(|e| format!("ONNX inference: {}", e))?
            }
            _ => {
                return Err(format!(
                    "ONNX VC unsupported input rank {}",
                    self.input_shape_concrete.len()
                ));
            }
        };

        if outputs.len() < 1 {
            return Err("ONNX model returned no outputs".to_string());
        }
        let (_shape, data) = outputs[0]
            .try_extract_tensor::<f32>()
            .map_err(|e| format!("ONNX output tensor: {}", e))?;
        let n_out = data.len();
        Ok(adapt_model_to_runtime(
            data,
            n_out,
            self.ch_in,
            self.runtime_need,
            self.chunk_frames,
        ))
    }
}

#[cfg(test)]
mod tests {
    use super::{
        adapt_model_to_runtime, adapt_runtime_to_model, convert_channels, deinterleave, interleave,
        linear_resample_1d, OnnxVoiceBridge,
    };

    const T_FRAMES: usize = 512;

    #[test]
    fn new_rejects_missing_file() {
        let r = OnnxVoiceBridge::new("/nonexistent/vc_model.onnx", 1024, 2, "cpu");
        assert!(r.is_err());
    }

    #[test]
    fn deinterleave_interleave_roundtrip() {
        let frames = T_FRAMES;
        let ch = 2usize;
        let mut v = vec![0.0; ch * frames];
        for f in 0..frames {
            v[f * ch] = f as f32;
            v[f * ch + 1] = -(f as f32);
        }
        let planes = deinterleave(&v, ch, frames).unwrap();
        let back = interleave(&planes, ch, frames).unwrap();
        assert_eq!(v, back);
    }

    #[test]
    fn stereo_to_mono_adapt() {
        let frames = T_FRAMES;
        let ch = 2usize;
        let mut v = vec![0.0; ch * frames];
        for f in 0..frames {
            v[f * ch] = 1.0;
            v[f * ch + 1] = 3.0;
        }
        let n_in = frames; // mono model
        let out = adapt_runtime_to_model(&v, v.len(), ch, n_in, frames);
        assert_eq!(out.len(), n_in);
        for &x in &out {
            assert!((x - 2.0).abs() < 1e-5);
        }
    }

    #[test]
    fn mono_to_stereo_adapt_back() {
        let frames = T_FRAMES;
        let mono: Vec<f32> = (0..frames).map(|f| f as f32 * 0.1).collect();
        let need = frames * 2;
        let back = adapt_model_to_runtime(&mono, mono.len(), 2, need, frames);
        assert_eq!(back.len(), need);
        let planes = deinterleave(&back, 2, frames).unwrap();
        assert_eq!(planes[0], mono);
        assert_eq!(planes[1], mono);
    }

    #[test]
    fn linear_resample_shortens() {
        let src: Vec<f32> = (0..10).map(|i| i as f32).collect();
        let dst = linear_resample_1d(&src, 5);
        assert_eq!(dst.len(), 5);
        assert!((dst[0] - 0.0).abs() < 1e-5);
        assert!((dst[4] - 9.0).abs() < 1e-5);
    }

    #[test]
    fn convert_channels_rejects_bad_len() {
        assert!(convert_channels(vec![vec![0.0]], 2, 1).is_none());
    }
}
