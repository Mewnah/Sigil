//! A Sigil-ZS voice's reference (ADR 0006 "Once per voice"; `sampler.py` `prepare_reference`):
//!
//! ```text
//! ref22k      = the clip at 22.05 kHz: the preset's prompt length of its speech (select_prompt)
//! ref16k      = torchaudio resample 22050 -> 16000            (dsp::resample_torchaudio)
//! style       = campplus(Kaldi fbank(ref16k) - its mean)       [1, 192]
//! prompt_mel  = mel(ref22k)                                    [1, 80, P]
//! prompt_cond = lr(xlsr(ref16k), P)                            [1, P, 384]
//! ```
//!
//! It runs on the CPU at the clip's own shapes (about 1 s for a 3 s prompt, XLS-R included), so the
//! GPU holds only the fixed-shape tick plans and a reference change never competes with the running
//! voice for the GPU. The result is cached per voice-library voice by `families::reference`
//! ([`backend`] names the cache file: the format version and the prompt length).
//!
//! **The prompt is speech.** Upstream cuts the reference's first `max_prompt` seconds, so a clip
//! that starts with a pause gives a prompt of room tone: measured on the dev rig (exact CAM++
//! cosine to the reference, `ja-short.wav` with 1.7 s of room tone first), a 1.5 s prompt converted
//! toward nobody (-0.02) and a 3 s one (1.3 s of speech) reached 0.43 against upstream offline's
//! 0.51, while prompts holding 3 s or more of speech reached 0.55-0.56. [`select_prompt`] therefore
//! drops the quiet ends and takes the prompt-length window with the most speech.

use std::sync::Arc;

use crate::services::neural_audio::gpu::{GpuLevers, TensorSpec};
use crate::services::neural_audio::ort_session::VcSessionOptions;
use crate::services::neural_audio::vc::families::reference::{ReferenceClip, ReferenceEmbedding, ReferenceTensor};

use super::dsp::{fbank_centered, resample_torchaudio};
use super::graphs::{GraphName, GraphSet};
use super::plans::{build_plan, GraphIo, Prepared};
use super::preset::{mel_frames, xlsr_frames, COND_DIM, CONTENT_DIM, MEL_BINS, MIN_PROMPT_S, SR_OUT, STYLE_DIM};
use super::sampler::Stage;

/// Cache format version (bump when what is stored changes; 2: the prompt is [`select_prompt`]'s).
pub const BACKEND_VERSION: &str = "sigil-zs-2";
/// Tensor names in the embedding.
pub const STYLE: &str = "style";
pub const PROMPT_MEL: &str = "prompt_mel";
pub const PROMPT_COND: &str = "prompt_cond";

/// The cache backend of a prompt length: `sigil-zs-2.p3000` for 3 s.
pub fn backend(prompt_s: f32) -> String {
    format!("{BACKEND_VERSION}.p{}", (prompt_s * 1000.0).round() as u32)
}

/// Analysis frame of [`select_prompt`] (20 ms at 22.05 kHz).
const SEL_FRAME: usize = 441;
/// Audio kept around the speech when the quiet ends are dropped (frames: 100 ms).
const SEL_MARGIN: usize = 5;
/// A frame is speech when it is this far above the clip's floor (its 10th-percentile frame)...
const SEL_ABOVE_FLOOR_DB: f64 = 10.0;
/// ...and no more than this below its loud frames (the 95th percentile).
const SEL_BELOW_PEAK_DB: f64 = 40.0;
/// A clip whose loud and quiet frames are closer than this has no pauses to drop.
const SEL_MIN_RANGE_DB: f64 = 15.0;

/// The prompt of a 22.05 kHz clip as `(first sample, samples)`: the clip without its quiet ends
/// (a [`SEL_MARGIN`] of audio kept on each side), and of that the `prompt_s` window holding the
/// most speech frames (the earliest such window; the whole of it when shorter). Deterministic, so
/// a library voice's cache stays valid. Speech is an energy decision (see the constants): enough
/// for a recorded or imported clip, whose noise floor sits well under its speech.
pub fn select_prompt(wav: &[f32], prompt_s: f32) -> (usize, usize) {
    let want = ((SR_OUT as f32 * prompt_s.max(0.0)) as usize).min(wav.len());
    let frames = wav.len() / SEL_FRAME;
    if frames < 2 {
        return (0, want);
    }
    let db: Vec<f64> = wav
        .chunks_exact(SEL_FRAME)
        .map(|f| {
            let ms = f.iter().map(|&v| f64::from(v) * f64::from(v)).sum::<f64>() / SEL_FRAME as f64;
            10.0 * (ms + 1e-12).log10()
        })
        .collect();
    let mut sorted = db.clone();
    sorted.sort_by(f64::total_cmp);
    let pct = |p: f64| sorted[((sorted.len() - 1) as f64 * p).round() as usize];
    let (floor, loud) = (pct(0.10), pct(0.95));
    if loud - floor < SEL_MIN_RANGE_DB {
        // No pauses to tell apart (all speech, or all noise): upstream's cut.
        return (0, want);
    }
    let threshold = (floor + SEL_ABOVE_FLOOR_DB).max(loud - SEL_BELOW_PEAK_DB);
    let speech: Vec<bool> = db.iter().map(|&d| d >= threshold).collect();
    let (Some(first), Some(last)) = (speech.iter().position(|&s| s), speech.iter().rposition(|&s| s)) else {
        return (0, want);
    };
    let lo = first.saturating_sub(SEL_MARGIN);
    let hi = (last + 1 + SEL_MARGIN).min(frames);
    // Speech up to the last whole frame keeps the clip's last partial frame too.
    let end = if hi == frames { wav.len() } else { hi * SEL_FRAME };
    let span = end - lo * SEL_FRAME;
    if span <= want {
        return (lo * SEL_FRAME, span);
    }
    // The prompt-length window over [lo, hi) with the most speech frames (a sliding count).
    let win = want.div_ceil(SEL_FRAME).min(hi - lo);
    let mut count = speech[lo..lo + win].iter().filter(|&&s| s).count();
    let (mut best, mut best_at) = (count, lo);
    for start in lo + 1..=hi - win {
        count = count + usize::from(speech[start + win - 1]) - usize::from(speech[start - 1]);
        if count > best {
            (best, best_at) = (count, start);
        }
    }
    let start = (best_at * SEL_FRAME).min(wav.len() - want);
    (start, want)
}

/// The prompt (22.05 kHz) of a clip: [`select_prompt`]'s `prompt_s` seconds of it.
pub fn prompt_audio(clip: &ReferenceClip, prompt_s: f32) -> Result<Vec<f32>, String> {
    let wav = clip.load_mono(SR_OUT)?;
    if !wav.iter().all(|v| v.is_finite()) {
        return Err(format!("the reference clip {} holds invalid samples", clip.name));
    }
    let (start, len) = select_prompt(&wav, prompt_s);
    let min = (SR_OUT as f32 * MIN_PROMPT_S) as usize;
    if len < min {
        return Err(format!(
            "the reference clip {} has {:.1} s of speech; zero-shot voices need at least {MIN_PROMPT_S} s",
            clip.name,
            len as f32 / SR_OUT as f32
        ));
    }
    Ok(wav[start..start + len].to_vec())
}

fn run_once(
    graphs: &GraphSet,
    graph: GraphName,
    opts: &VcSessionOptions,
    io: GraphIo<'_>,
    feed: impl FnOnce(&mut dyn Stage) -> Result<(), String>,
    read: &str,
) -> Result<Vec<f32>, String> {
    let (path, _) = graphs.path(graph, false);
    let component = "vc.zs.reference";
    let mut plan = build_plan(component, "cpu", opts, &path, io, GpuLevers::default(), true)?;
    feed(&mut plan)?;
    Stage::run(&mut plan)?;
    let out = Stage::output(&plan, read)?.to_vec();
    if !out.iter().all(|v| v.is_finite()) {
        return Err(format!("Sigil-ZS: the reference's {} output is not finite", graph.stem()));
    }
    Ok(out)
}

/// The reference tensors of a prompt (22.05 kHz audio, already cut), computed on the CPU.
pub fn compute(graphs: &GraphSet, ref22k: &[f32], opts: &VcSessionOptions) -> Result<Vec<ReferenceTensor>, String> {
    let opts = VcSessionOptions {
        intra_threads: Some(2),
        streamer_profile: false,
        vram_cap_mb: None,
        ..opts.clone()
    };
    let ref16k = resample_torchaudio(ref22k, SR_OUT, 16_000);
    let t = xlsr_frames(ref16k.len());
    let p = mel_frames(ref22k.len());
    if t < 2 || p < 2 {
        return Err("Sigil-ZS: the reference prompt is too short".into());
    }
    // Style (CAM++ over the mean-removed fbank).
    let (fb, frames) = fbank_centered(&ref16k);
    let style = run_once(
        graphs,
        GraphName::Campplus,
        &opts,
        GraphIo {
            inputs: &[TensorSpec::f32("fbank", &[1, frames, 80])],
            outputs: &[TensorSpec::f32("style", &[1, STYLE_DIM])],
            dims: &[("F", frames as i64)],
        },
        |s| {
            s.input_f32("fbank")?.copy_from_slice(&fb);
            Ok(())
        },
        "style",
    )?;
    // Prompt mel.
    let prompt_mel = run_once(
        graphs,
        GraphName::Mel,
        &opts,
        GraphIo {
            inputs: &[TensorSpec::f32("wav22k", &[1, ref22k.len()])],
            outputs: &[TensorSpec::f32("mel", &[1, MEL_BINS, p])],
            dims: &[("S", ref22k.len() as i64)],
        },
        |s| {
            s.input_f32("wav22k")?.copy_from_slice(ref22k);
            Ok(())
        },
        "mel",
    )?;
    // Content of the prompt, stretched to its mel length.
    let feats = run_once(
        graphs,
        GraphName::Xlsr,
        &opts,
        GraphIo {
            inputs: &[TensorSpec::f32("wav16k", &[1, ref16k.len()])],
            outputs: &[TensorSpec::f32("feats", &[1, t, CONTENT_DIM])],
            dims: &[("N", ref16k.len() as i64)],
        },
        |s| {
            s.input_f32("wav16k")?.copy_from_slice(&ref16k);
            Ok(())
        },
        "feats",
    )?;
    let prompt_cond = run_once(
        graphs,
        GraphName::Lr,
        &opts,
        GraphIo {
            inputs: &[TensorSpec::f32("feats", &[1, t, CONTENT_DIM]), TensorSpec::i64("n_frames", &[1])],
            outputs: &[TensorSpec::f32("cond", &[1, p, COND_DIM])],
            dims: &[("T", t as i64)],
        },
        |s| {
            s.input_f32("feats")?.copy_from_slice(&feats);
            s.input_i64("n_frames")?[0] = p as i64;
            Ok(())
        },
        "cond",
    )?;
    Ok(vec![
        ReferenceTensor::new(STYLE, vec![1, STYLE_DIM], style),
        ReferenceTensor::new(PROMPT_MEL, vec![1, MEL_BINS, p], prompt_mel),
        ReferenceTensor::new(PROMPT_COND, vec![1, p, COND_DIM], prompt_cond),
    ])
}

/// The embedding of `clip` for a `prompt_s` prompt: the voice library's cache when it holds one from
/// this clip, else computed (and cached for a library voice).
pub fn embedding(graphs: &GraphSet, clip: &ReferenceClip, prompt_s: f32, app_data: Option<&std::path::Path>, opts: &VcSessionOptions) -> Result<Arc<ReferenceEmbedding>, String> {
    let backend = backend(prompt_s);
    let compute_now = |c: &ReferenceClip| -> Result<ReferenceEmbedding, String> {
        let wav = prompt_audio(c, prompt_s)?;
        Ok(ReferenceEmbedding { tensors: compute(graphs, &wav, opts)?, ..Default::default() })
    };
    let e = crate::services::neural_audio::vc::families::reference::cached_embedding(app_data, clip, &backend, compute_now)?;
    match prepare(&e) {
        Ok(_) => Ok(Arc::new(e)),
        Err(why) => {
            // A cache file that parses but does not hold this backend's tensors: compute it afresh
            // and replace the cache, so the next load reads it again.
            tracing::warn!(target: "sigil::vc", "Sigil-ZS: the cached reference of {} is unusable ({why}); recomputing", clip.name);
            let mut fresh = compute_now(clip)?;
            fresh.voice_id = clip.voice_id.clone();
            fresh.backend = backend;
            fresh.clip_sha256 = clip.sha256.clone();
            prepare(&fresh)?;
            if let Some(app_data) = app_data.filter(|_| !clip.voice_id.is_empty()) {
                let stored = fresh
                    .to_bytes()
                    .and_then(|b| crate::services::voice_library::write_cache(app_data, &clip.voice_id, &fresh.backend, &b));
                if let Err(e) = stored {
                    tracing::debug!(target: "sigil::vc", "Sigil-ZS: reference cache of {} not replaced: {e}", clip.voice_id);
                }
            }
            Ok(Arc::new(fresh))
        }
    }
}

/// The DiT's inputs from an embedding (validated: names, shapes, finite values).
pub fn prepare(e: &ReferenceEmbedding) -> Result<Prepared, String> {
    let get = |name: &str| e.tensor(name).ok_or_else(|| format!("the reference has no {name}"));
    let style = get(STYLE)?;
    let mel = get(PROMPT_MEL)?;
    let cond = get(PROMPT_COND)?;
    let p = match mel.shape.as_slice() {
        [1, MEL_BINS, p] => *p,
        s => return Err(format!("the reference prompt mel has shape {s:?}")),
    };
    if style.shape != [1, STYLE_DIM] || cond.shape != [1, p, COND_DIM] || p < 2 {
        return Err(format!("the reference shapes {:?} / {:?} / {:?} do not fit", style.shape, mel.shape, cond.shape));
    }
    e.validate()?;
    for t in [style, mel, cond] {
        if !t.data.iter().all(|v| v.is_finite()) {
            return Err(format!("the reference's {} is not finite", t.name));
        }
    }
    Ok(Prepared { style: style.data.clone(), prompt_mel: mel.data.clone(), prompt_cond: cond.data.clone(), frames: p })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn backends_name_the_prompt_length() {
        assert_eq!(backend(3.0), "sigil-zs-2.p3000");
        assert_eq!(backend(1.5), "sigil-zs-2.p1500");
        assert!(crate::services::neural_audio::vc::families::reference::valid_backend(&backend(2.25)));
    }

    /// `segments` of (seconds, level): speech-like (a voiced tone with a syllable envelope) at a
    /// level > 0, room noise at -38 dBFS for `NOISE`, digital silence for `SILENT`.
    const NOISE: f32 = -1.0;
    const SILENT: f32 = 0.0;
    fn clip(segments: &[(f32, f32)]) -> Vec<f32> {
        let sr = SR_OUT as f32;
        let mut out = Vec::new();
        let mut s = 0x2545_F491u32;
        for &(secs, level) in segments {
            for i in 0..(secs * sr) as usize {
                s ^= s << 13;
                s ^= s >> 17;
                s ^= s << 5;
                let n = (s as f32 / u32::MAX as f32 - 0.5) * 2.0;
                let t = i as f32 / sr;
                out.push(if level == NOISE {
                    0.0126 * 1.73 * n
                } else if level == SILENT {
                    0.0
                } else {
                    let syllable = 0.6 + 0.4 * (std::f32::consts::TAU * 4.0 * t).sin();
                    level * syllable * (std::f32::consts::TAU * 150.0 * t).sin() + 0.002 * n
                });
            }
        }
        out
    }

    fn secs(samples: usize) -> f32 {
        samples as f32 / SR_OUT as f32
    }

    #[test]
    fn the_prompt_skips_a_quiet_start_and_takes_the_most_speech() {
        // 1.7 s of room tone, then 5 s of speech, then room tone (ja-short.wav's shape).
        let wav = clip(&[(1.7, NOISE), (5.0, 0.2), (2.0, NOISE)]);
        let (start, len) = select_prompt(&wav, 3.0);
        assert_eq!(len, (SR_OUT as f32 * 3.0) as usize);
        assert!(secs(start) >= 1.55 && secs(start + len) <= 6.75, "{} .. {}", secs(start), secs(start + len));
        // Speech, a long pause, more speech: the 3 s window lies in the longer stretch of speech.
        let wav = clip(&[(1.0, 0.2), (2.5, NOISE), (4.0, 0.15), (0.5, NOISE)]);
        let (start, len) = select_prompt(&wav, 3.0);
        assert!(secs(start) >= 3.4 && secs(start + len) <= 7.6, "{} .. {}", secs(start), secs(start + len));
        // Digital silence around the speech is dropped the same way.
        let wav = clip(&[(2.0, SILENT), (4.0, 0.2), (1.0, SILENT)]);
        let (start, _) = select_prompt(&wav, 3.0);
        assert!(secs(start) >= 1.85, "{}", secs(start));
    }

    #[test]
    fn short_speech_is_kept_whole_and_level_clips_are_cut_as_upstream() {
        // Less speech than the prompt: the speech with its margins, nothing more.
        let wav = clip(&[(2.0, NOISE), (1.0, 0.2), (2.0, NOISE)]);
        let (start, len) = select_prompt(&wav, 3.0);
        assert!((secs(start) - 1.9).abs() < 0.03 && (secs(len) - 1.2).abs() < 0.05, "{} + {}", secs(start), secs(len));
        // No pauses (all speech, or all noise): the first seconds, as upstream.
        assert_eq!(select_prompt(&clip(&[(6.0, 0.2)]), 3.0), (0, 66_150));
        assert_eq!(select_prompt(&clip(&[(6.0, NOISE)]), 3.0), (0, 66_150));
        // A clip shorter than the prompt, and one too short to analyse.
        assert_eq!(select_prompt(&clip(&[(2.0, 0.2)]), 3.0), (0, 44_100));
        assert_eq!(select_prompt(&[0.1; 500], 3.0), (0, 500));
    }

    #[test]
    fn prepare_checks_names_shapes_and_values() {
        let p = 5;
        let good = ReferenceEmbedding {
            tensors: vec![
                ReferenceTensor::new(STYLE, vec![1, STYLE_DIM], vec![0.1; STYLE_DIM]),
                ReferenceTensor::new(PROMPT_MEL, vec![1, MEL_BINS, p], vec![-4.0; MEL_BINS * p]),
                ReferenceTensor::new(PROMPT_COND, vec![1, p, COND_DIM], vec![0.5; p * COND_DIM]),
            ],
            ..Default::default()
        };
        let r = prepare(&good).unwrap();
        assert_eq!((r.frames, r.style.len(), r.prompt_mel.len(), r.prompt_cond.len()), (p, STYLE_DIM, MEL_BINS * p, p * COND_DIM));
        let mut bad = good.clone();
        bad.tensors[2] = ReferenceTensor::new(PROMPT_COND, vec![1, p + 1, COND_DIM], vec![0.5; (p + 1) * COND_DIM]);
        assert!(prepare(&bad).is_err(), "prompt lengths disagree");
        let mut nan = good.clone();
        nan.tensors[0].data[3] = f32::NAN;
        assert!(prepare(&nan).is_err());
        let mut missing = good;
        missing.tensors.pop();
        assert!(prepare(&missing).unwrap_err().contains(PROMPT_COND));
    }
}
