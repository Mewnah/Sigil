//! The input contract of a so-vits-svc ONNX export (P5-D), resolved once at load from the graph's
//! declared inputs (by name, or by position for the canonical order).
//!
//! so-vits-svc 4.0 / 4.1's `onnx_export.py` (the MoeVS / MoeSS format):
//!
//! | input | type | shape | what Sigil feeds |
//! |---|---|---|---|
//! | `c` | f32 / f16 | `[1, S, C]` | content features (ContentVec), time-major, at the encoder's 50 Hz |
//! | `f0` | f32 / f16 | `[1, T]` | F0 in Hz on the model's frame grid (`hop / sr`), unvoiced frames interpolated |
//! | `mel2ph` | i64 | `[1, T]` | the content frame (1-based; 0 = a zero frame) of each model frame |
//! | `uv` | f32 or i64 | `[1, T]` | 1 = voiced |
//! | `noise` | f32 / f16 | `[1, 192, T]` | the flow prior's noise × noise scale (so-vits default 0.4) |
//! | `sid` | i64 `[1]`, or f32 `[T, N]` | | the speaker id, or (a `CharaMix` export) per-frame speaker weights |
//! | `vol` | f32 / f16 | `[1, T]` | optional (4.1 `vol_embedding`): the input's RMS per frame |
//!
//! Also accepted: exports without `mel2ph` (so-vits-svc 4.0 graphs exported elsewhere, w-okada
//! style), whose content is `[1, C, T]` at the model's frame rate (Sigil upsamples it) with an
//! optional `c_lengths` / `lengths` input (`T`) and the speaker as `sid` or `g`; a graph without a
//! `noise` input draws its own noise. A graph of another kind (an RVC generator, so-vits-svc 3.0, a
//! content encoder, ...) is refused by what it is.

use std::path::Path;

use ort::session::Session;
use ort::value::TensorElementType;

use crate::services::neural_audio::vc::rvc::{graph_inputs, graph_outputs, identify_graph, wrong_graph_message, GraphKind, GraphValue};

/// Latent (flow noise) channels of so-vits-svc's prior.
pub const DEFAULT_NOISE_CHANNELS: usize = 192;

/// Role of one generator input.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Role {
    Content,
    F0,
    Mel2ph,
    Uv,
    Noise,
    Speaker,
    Volume,
    Lengths,
}

fn role_for_name(name: &str) -> Option<Role> {
    match name.to_ascii_lowercase().as_str() {
        "c" | "hidden_unit" | "units" | "content" => Some(Role::Content),
        "f0" | "pitchf" | "nsff0" => Some(Role::F0),
        "mel2ph" => Some(Role::Mel2ph),
        "uv" => Some(Role::Uv),
        "noise" | "z" => Some(Role::Noise),
        "sid" | "g" | "spk" | "speaker" | "spk_mix" => Some(Role::Speaker),
        "vol" | "volume" => Some(Role::Volume),
        "c_lengths" | "lengths" | "x_lengths" => Some(Role::Lengths),
        _ => None,
    }
}

/// so-vits-svc 4.x `onnx_export.py` input order.
const CANONICAL: [Role; 7] = [Role::Content, Role::F0, Role::Mel2ph, Role::Uv, Role::Noise, Role::Speaker, Role::Volume];

/// How the speaker is fed.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum SpeakerInput {
    /// An int64 id of this rank (`[1]` or `[1, 1]`).
    Id { rank: usize },
    /// Per-frame speaker weights `[T, N]` (MoeVS `CharaMix`); `N` when declared.
    Mix { speakers: Option<usize> },
}

/// The resolved contract of one export.
#[derive(Clone, Debug, PartialEq)]
pub struct SovitsGraph {
    pub names: Vec<String>,
    pub roles: Vec<Role>,
    /// Per input: the export takes fp16 (float roles).
    pub half: Vec<bool>,
    /// Content width when declared.
    pub content_width: Option<usize>,
    /// Content layout: `[1, S, C]` (true) or `[1, C, T]`.
    pub content_time_major: bool,
    /// `uv` is int64 (else float).
    pub uv_int: bool,
    pub noise_channels: usize,
    pub speaker: Option<SpeakerInput>,
    /// The output is fp16.
    pub output_half: bool,
}

impl SovitsGraph {
    /// Resolve `session`'s inputs (`path` names the file in messages).
    pub fn resolve(session: &Session, path: &Path) -> Result<SovitsGraph, String> {
        Self::from_values(&graph_inputs(session), &graph_outputs(session), path)
    }

    /// [`resolve`](Self::resolve) from declared values.
    pub fn from_values(inputs: &[GraphValue], outputs: &[GraphValue], path: &Path) -> Result<SovitsGraph, String> {
        let file = path.file_name().map(|n| n.to_string_lossy().into_owned()).unwrap_or_default();
        match identify_graph(inputs, outputs) {
            GraphKind::Sovits4 | GraphKind::Unknown => {}
            GraphKind::RvcGenerator => {
                return Err(format!("{file} is an RVC generator, not a so-vits-svc export; add it as an RVC voice (bring-your-own ONNX)"));
            }
            _ => {
                return Err(wrong_graph_message(path, "a so-vits-svc export", inputs, outputs)
                    .unwrap_or_else(|| format!("{file} is not a so-vits-svc export")))
            }
        }
        let names: Vec<String> = inputs.iter().map(|i| i.name.clone()).collect();
        let roles: Vec<Role> = match names.iter().map(|n| role_for_name(n)).collect::<Option<Vec<_>>>() {
            Some(r) => r,
            None if (6..=7).contains(&names.len()) => CANONICAL[..names.len()].to_vec(),
            None => {
                return Err(format!(
                    "{file} has inputs {names:?}; a so-vits-svc 4.x export takes c, f0, mel2ph, uv, noise and sid (plus vol)"
                ))
            }
        };
        for (i, role) in roles.iter().enumerate() {
            if roles[..i].contains(role) {
                return Err(format!("{file} maps two inputs to {role:?} ({names:?})"));
            }
        }
        let has = |r: Role| roles.contains(&r);
        for (role, what) in [(Role::Content, "content (c)"), (Role::F0, "f0"), (Role::Uv, "uv")] {
            if !has(role) {
                return Err(format!("{file} has no {what} input ({names:?}); it is not a so-vits-svc 4.x export"));
            }
        }
        let mut g = SovitsGraph {
            names: names.clone(),
            roles: roles.clone(),
            half: Vec::with_capacity(roles.len()),
            content_width: None,
            content_time_major: has(Role::Mel2ph),
            uv_int: false,
            noise_channels: DEFAULT_NOISE_CHANNELS,
            speaker: None,
            output_half: outputs.first().is_some_and(|o| o.dtype == Some(TensorElementType::Float16)),
        };
        let float_ok = |t: Option<TensorElementType>| matches!(t, Some(TensorElementType::Float32) | Some(TensorElementType::Float16));
        for (input, role) in inputs.iter().zip(roles.iter()) {
            let t = input.dtype;
            let dims = input.dims.clone().unwrap_or_default();
            let fixed = |i: usize| dims.get(i).and_then(|&d| usize::try_from(d).ok()).filter(|&d| d > 0);
            let bad = |want: &str| format!("{file} input '{}' ({role:?}) is {t:?} {dims:?}; expected {want}", input.name);
            let half = t == Some(TensorElementType::Float16);
            match role {
                Role::Content => {
                    if !float_ok(t) || dims.len() != 3 {
                        return Err(bad("float [1, frames, channels]"));
                    }
                    // The channel axis is the one with a static content width.
                    match (fixed(1), fixed(2)) {
                        (_, Some(c)) if c > 4 => {
                            g.content_width = Some(c);
                            g.content_time_major = true;
                        }
                        (Some(c), _) if c > 4 => {
                            g.content_width = Some(c);
                            g.content_time_major = false;
                        }
                        _ => {}
                    }
                }
                Role::F0 | Role::Volume => {
                    if !float_ok(t) || !(1..=2).contains(&dims.len()) {
                        return Err(bad("float [1, frames]"));
                    }
                }
                Role::Mel2ph => {
                    if t != Some(TensorElementType::Int64) || dims.len() != 2 {
                        return Err(bad("int64 [1, frames]"));
                    }
                }
                Role::Uv => {
                    g.uv_int = match t {
                        Some(TensorElementType::Int64) => true,
                        _ if float_ok(t) => false,
                        _ => return Err(bad("float or int64 [1, frames]")),
                    };
                    if !(1..=2).contains(&dims.len()) {
                        return Err(bad("[1, frames]"));
                    }
                }
                Role::Noise => {
                    if !float_ok(t) || dims.len() != 3 {
                        return Err(bad("float [1, 192, frames]"));
                    }
                    if let Some(c) = fixed(1) {
                        g.noise_channels = c;
                    }
                }
                Role::Speaker => {
                    g.speaker = Some(match t {
                        Some(TensorElementType::Int64) if (1..=2).contains(&dims.len()) => SpeakerInput::Id { rank: dims.len() },
                        _ if float_ok(t) && dims.len() == 2 => SpeakerInput::Mix { speakers: fixed(1) },
                        _ => return Err(bad("an int64 speaker id [1] or float speaker weights [frames, speakers]")),
                    });
                }
                Role::Lengths => {
                    if t != Some(TensorElementType::Int64) || dims.len() != 1 {
                        return Err(bad("int64 [1]"));
                    }
                }
            }
            g.half.push(half && matches!(role, Role::Content | Role::F0 | Role::Volume | Role::Noise | Role::Uv | Role::Speaker));
        }
        let output = outputs.first().ok_or_else(|| format!("{file} declares no output"))?;
        if !float_ok(output.dtype) {
            return Err(format!("{file} outputs {:?} audio; expected float", output.dtype));
        }
        Ok(g)
    }

    pub fn has(&self, role: Role) -> bool {
        self.roles.contains(&role)
    }

    /// Whether the export maps content frames itself (`mel2ph`): content at the encoder's rate.
    pub fn maps_content(&self) -> bool {
        self.has(Role::Mel2ph)
    }

    /// Whether `role`'s input is fp16.
    pub fn half(&self, role: Role) -> bool {
        self.roles.iter().zip(self.half.iter()).any(|(&r, &h)| r == role && h)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use TensorElementType::{Float16, Float32, Int64};

    fn v(name: &str, t: TensorElementType, dims: &[i64]) -> GraphValue {
        GraphValue::new(name, t, dims)
    }

    fn audio() -> Vec<GraphValue> {
        vec![v("audio", Float32, &[1, 1, -1])]
    }

    fn sovits41(sid_mix: bool, vol: bool) -> Vec<GraphValue> {
        let mut i = vec![
            v("c", Float32, &[-1, -1, 768]),
            v("f0", Float32, &[1, -1]),
            v("mel2ph", Int64, &[1, -1]),
            v("uv", Float32, &[1, -1]),
            v("noise", Float32, &[1, 192, -1]),
            if sid_mix { v("sid", Float32, &[-1, 3]) } else { v("sid", Int64, &[1]) },
        ];
        if vol {
            i.push(v("vol", Float32, &[1, -1]));
        }
        i
    }

    #[test]
    fn moevs_exports_resolve() {
        let g = SovitsGraph::from_values(&sovits41(false, true), &audio(), Path::new("nyaru_SoVits.onnx")).unwrap();
        assert_eq!(g.content_width, Some(768));
        assert!(g.content_time_major && g.maps_content() && !g.uv_int);
        assert_eq!(g.speaker, Some(SpeakerInput::Id { rank: 1 }));
        assert!(g.has(Role::Volume) && g.has(Role::Noise));
        assert_eq!(g.noise_channels, 192);
        let g = SovitsGraph::from_values(&sovits41(true, false), &audio(), Path::new("mix.onnx")).unwrap();
        assert_eq!(g.speaker, Some(SpeakerInput::Mix { speakers: Some(3) }));
        assert!(!g.has(Role::Volume));
    }

    #[test]
    fn positional_and_half_exports_resolve() {
        let mut i = sovits41(false, false);
        for (k, x) in i.iter_mut().enumerate() {
            x.name = format!("input.{k}");
        }
        i[0].dtype = Some(Float16);
        i[4].dtype = Some(Float16);
        let g = SovitsGraph::from_values(&i, &[v("o", Float16, &[1, 1, -1])], Path::new("p.onnx")).unwrap();
        assert_eq!(g.roles, CANONICAL[..6].to_vec());
        assert!(g.half(Role::Content) && g.half(Role::Noise) && !g.half(Role::F0));
        assert!(g.output_half);
    }

    /// A so-vits-svc 4.0 graph without mel2ph: channel-major content at the frame rate, `g`, lengths.
    #[test]
    fn exports_without_mel2ph_resolve() {
        let i = vec![
            v("c", Float32, &[1, 256, -1]),
            v("f0", Float32, &[1, -1]),
            v("uv", Float32, &[1, -1]),
            v("g", Int64, &[1, 1]),
            v("c_lengths", Int64, &[1]),
        ];
        let g = SovitsGraph::from_values(&i, &audio(), Path::new("w.onnx")).unwrap();
        assert_eq!(g.content_width, Some(256));
        assert!(!g.content_time_major && !g.maps_content());
        assert_eq!(g.speaker, Some(SpeakerInput::Id { rank: 2 }));
        assert!(g.has(Role::Lengths) && !g.has(Role::Noise));
    }

    #[test]
    fn other_graphs_are_refused_by_what_they_are() {
        let rvc = vec![
            v("phone", Float32, &[1, -1, 768]),
            v("phone_lengths", Int64, &[1]),
            v("pitch", Int64, &[1, -1]),
            v("pitchf", Float32, &[1, -1]),
            v("ds", Int64, &[1]),
            v("rnd", Float32, &[1, 192, -1]),
        ];
        let err = SovitsGraph::from_values(&rvc, &audio(), Path::new("g.onnx")).unwrap_err();
        assert!(err.contains("RVC generator"), "{err}");
        let v3 = vec![
            v("hidden_unit", Float32, &[1, -1, 256]),
            v("lengths", Int64, &[1]),
            v("pitch", Int64, &[1, -1]),
            v("sid", Int64, &[1]),
        ];
        let err = SovitsGraph::from_values(&v3, &audio(), Path::new("old.onnx")).unwrap_err();
        assert!(err.contains("3.0"), "{err}");
        let encoder = vec![v("audio", Float16, &[1, -1])];
        let outs = vec![v("units9", Float16, &[1, -1, 256]), v("unit12", Float16, &[1, -1, 768])];
        let err = SovitsGraph::from_values(&encoder, &outs, Path::new("contentvec-h.onnx")).unwrap_err();
        assert!(err.contains("content encoder"), "{err}");
        // Wrong types are named.
        let mut bad = sovits41(false, false);
        bad[2].dtype = Some(Float32);
        let err = SovitsGraph::from_values(&bad, &audio(), Path::new("b.onnx")).unwrap_err();
        assert!(err.contains("mel2ph") && err.contains("int64"), "{err}");
        let mut no_uv = sovits41(false, false);
        no_uv.remove(3);
        assert!(SovitsGraph::from_values(&no_uv, &audio(), Path::new("n.onnx")).is_err());
    }
}
