//! Research engines over sidecar protocol v2, inside the harness (P5-E).
//!
//! A sidecar voice is a slot whose options ([`SidecarV2Options`], `manifest.family`) name either a
//! **recipe** Sigil starts itself, or a **custom** protocol v2 server the user runs:
//!
//! | recipe | engine | script (`scripts/neural-sidecar/`) |
//! |---|---|---|
//! | `xvc` | X-VC (zero-shot, best speaker similarity; 12 GB+ GPUs) | `xvc_sidecar.py` |
//! | `streamvoiceanon` | StreamVoiceAnon (privacy mode: anonymized toward a reference) | `streamvoiceanon_sidecar.py` |
//! | `seedvc` | Seed-VC real-time (the zero-shot fallback while Sigil-ZS runs in-engine) | `seedvc_sidecar.py` |
//! | `custom` | any protocol v2 (or v1) server at `url` | yours |
//!
//! Recipes run as `process_host` helpers ([`recipes`]): the user's Python interpreter
//! (`python`), the script, `--host 127.0.0.1 --port {port}`, the voice changer's device, the model
//! folder (`model_dir`) and extra `args`; the key goes through the environment and the WebSocket's
//! `Authorization` header. One helper serves every voice with the same settings (StreamVoiceAnon,
//! whose model holds one stream, converts for one voice at a time: the voice that starts or is
//! switched to takes it over, and the one it replaces plays silence for that moment). A recipe
//! with a `url` attaches to a server the user started by hand instead. A load the voice changer
//! no longer wants ([`create_cancellable`]) stops waiting for the helper's start or the handshake
//! at once.
//!
//! The voice is [`model::SidecarModel`]: the client of `neural_audio::sidecar_bridge::v2` (async
//! pipelined frames, rate negotiation, reference upload, pitch per frame) behind a jitter buffer
//! whose frame deadline is the latency budget (`timeout_ms` overrides it). A stalled sidecar never
//! blocks the voice changer's loop: its frames fail at the budget (the worker mutes them), the
//! sidecar is told to drop what is stale, a late reply is discarded, and a lost connection is
//! re-established in the background. A recipe that converts toward a reference uploads the voice's
//! library clip (`FamilyRequest::reference`).
//!
//! # Frozen by P5-S0 (signatures; see `families` for the contract)
//! [`available`], [`info`], [`create`], [`check_slot`], [`normalize_options`], [`create_slot`],
//! [`run_import_job`], [`SidecarV2Options`] (the `manifest.family` object; TS
//! `VcSidecarV2Options`), [`RECIPES`].

pub mod model;
pub mod recipes;

use std::path::Path;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;
use std::time::Duration;

use serde::{Deserialize, Serialize};
use serde_json::{json, Map};

use super::reference::ReferenceClip;
use super::{CreatedSlot, FamilyEngine, FamilyInfo, FamilyKind, FamilyRequest, FamilySlotContext, FamilySlotRequest, CREATION_SIDECAR};
use crate::services::diagnostics::EpOutcome;
use crate::services::neural_audio::sidecar_bridge::v2::{
    ClientConfig, Endpoint, EndpointFn, ReferenceMeta, ReferenceSource, Session, SidecarClient, CONNECT_CANCELLED,
};
use crate::services::neural_audio::vc::engine::options::EffectiveLevers;
use crate::services::neural_audio::vc::import::ImportJob;
use crate::services::neural_audio::vc::manifest::RvcManifest;
use crate::services::process_host;
use crate::services::resources::{self, Subsystem};
use crate::services::voice_changer::factory::LOAD_CANCELLED;

/// Sidecar recipes ([`SidecarV2Options::recipe`]): `xvc` (X-VC), `streamvoiceanon`
/// (StreamVoiceAnon, the privacy mode), `seedvc` (seed-vc-realtime, the Sigil-ZS fallback), `custom`
/// (a protocol v2 server the user runs, at `url`).
pub const RECIPES: [&str; 4] = ["custom", "xvc", "streamvoiceanon", "seedvc"];

/// Per-frame timeout bounds (`timeout_ms`, ms).
const MIN_TIMEOUT_MS: u32 = 20;
const MAX_TIMEOUT_MS: u32 = 5_000;
/// The budget when the request carries none (the voice changer's default).
const DEFAULT_BUDGET_MS: f64 = 300.0;
/// Frame proposed to a custom server when the request names no block.
const DEFAULT_FRAME_MS: f64 = 100.0;
/// Input rates offered to the server (16 kHz first: the harness's denoiser and the research
/// models' content encoders run at 16 kHz).
const INPUT_RATES: [u32; 6] = [16_000, 22_050, 24_000, 32_000, 44_100, 48_000];
/// GPU memory claimed for a custom server that reports none but says it runs on a GPU.
const CUSTOM_VRAM_HINT_MB: u64 = 1_000;

/// The family's options (`manifest.family`, snake_case).
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(default)]
pub struct SidecarV2Options {
    /// One of [`RECIPES`] (default `custom`).
    pub recipe: String,
    /// `custom`: the server's WebSocket URL (loopback only: `ws://127.0.0.1:<port>/...`,
    /// `localhost` or `[::1]`). A recipe with a URL attaches to a server the user started instead
    /// of starting one.
    pub url: String,
    /// Recipes: the Python interpreter that runs the sidecar (`""` = `python` on PATH; else a full
    /// path, e.g. a virtual environment's `python.exe`).
    pub python: String,
    /// Recipes: the model / checkpoint folder (`""` = the recipe's default; else a full path).
    pub model_dir: String,
    /// Recipes: extra command-line arguments (never `--host`, `--port` or the key).
    pub args: Vec<String>,
    /// Per-frame timeout in ms (`None` = the latency budget).
    pub timeout_ms: Option<u32>,
}

impl Default for SidecarV2Options {
    fn default() -> Self {
        SidecarV2Options {
            recipe: RECIPES[0].to_string(),
            url: String::new(),
            python: String::new(),
            model_dir: String::new(),
            args: Vec::new(),
            timeout_ms: None,
        }
    }
}

/// Whether `url` is a WebSocket URL on this PC (sidecars never leave loopback).
pub fn loopback_ws_url(url: &str) -> bool {
    let u = url.trim().to_ascii_lowercase();
    let Some(rest) = u.strip_prefix("ws://") else {
        return false;
    };
    let host_port = rest.split(['/', '?', '#']).next().unwrap_or_default();
    if host_port.contains('@') {
        return false;
    }
    let host = if let Some(h) = host_port.strip_prefix('[') {
        h.split(']').next().map(|h| format!("[{h}]")).unwrap_or_default()
    } else {
        host_port.split(':').next().unwrap_or_default().to_string()
    };
    matches!(host.as_str(), "127.0.0.1" | "localhost" | "[::1]")
}

/// Sidecar v2 voices run in this build.
pub fn available() -> bool {
    true
}

pub fn info() -> FamilyInfo {
    FamilyInfo {
        family: "sidecar_v2",
        format: super::SIDECAR_V2_FORMAT,
        experimental: true,
        // A recipe may convert toward the voice's reference (X-VC, seed-vc-realtime).
        zero_shot: true,
        singing: false,
        // The sidecar runs its own runtime on its own device; any rung of the ladder can drive it.
        execution_providers: &["cuda", "dml", "cpu"],
        needs_pitch: false,
        needs_content_encoder: false,
        creation: CREATION_SIDECAR,
        import_extensions: &[],
        controls: &["pitch"],
        requires: &[],
    }
}

/// The normalized options of a slot.
fn options_of(manifest: &RvcManifest) -> Result<SidecarV2Options, String> {
    let v = normalize_options(&super::options_of(manifest))?;
    serde_json::from_value(v).map_err(|e| format!("invalid sidecar options: {e}"))
}

/// The reference voice as the protocol uploads it (decoded at the rate the server asks for).
fn reference_source(clip: &ReferenceClip) -> ReferenceSource {
    let c = clip.clone();
    ReferenceSource {
        meta: ReferenceMeta {
            voice_id: clip.voice_id.clone(),
            name: clip.name.clone(),
            sha256: clip.sha256.clone(),
            duration_ms: clip.duration_ms,
            transcript: clip.transcript.clone(),
            language: clip.language.clone(),
        },
        load: Arc::new(move |rate| c.load_mono(rate)),
    }
}

/// The EP outcome of the sidecar's device (`vc.sidecar`): what it says it runs on, else the rung
/// it was asked for.
fn sidecar_outcome(requested: &str, device: Option<&str>) -> EpOutcome {
    let active = match device.map(str::trim) {
        Some(d) if d.starts_with("cuda") => "cuda".to_string(),
        Some("dml" | "directml") => "dml".to_string(),
        Some("cpu") => "cpu".to_string(),
        Some(d) if !d.is_empty() => d.chars().take(32).collect(),
        _ => return EpOutcome::new(requested, requested, Some("the sidecar does not say which device it runs on".into())),
    };
    if active == requested {
        EpOutcome::granted(active)
    } else {
        EpOutcome::new(requested, active, Some("the sidecar process chose its own device".into()))
    }
}

/// Frames a voice keeps in flight: as many as the budget holds, plus the frame being captured and
/// one of margin (sized from the frame the server negotiated; the hello offers it for the
/// proposed one).
pub(crate) fn in_flight_window(budget_ms: f64, frame_ms: f64) -> usize {
    let frames = (budget_ms.max(0.0) / frame_ms.max(1.0)).ceil();
    (if frames.is_finite() { frames as usize } else { 0 } + 2).clamp(2, 1024)
}

/// The frame proposed in the hello: the recipe's, else the requested engine block, else the
/// device block.
fn proposed_frame_ms(recipe: Option<&recipes::Recipe>, request: &FamilyRequest) -> f64 {
    if let Some(r) = recipe {
        return r.frame_ms;
    }
    let ms = if request.block_ms > 0.0 {
        request.block_ms
    } else if request.chunk_frames > 0 && request.device_rate > 0 {
        f64::from(request.chunk_frames) * 1000.0 / f64::from(request.device_rate)
    } else {
        DEFAULT_FRAME_MS
    };
    if ms.is_finite() {
        ms.clamp(20.0, 500.0)
    } else {
        DEFAULT_FRAME_MS
    }
}

/// What a voice's load reports about the session (shown as warnings; the voice still loads).
fn load_warnings(label: &str, session: &Session, recipe: bool, has_reference: bool, budget_ms: f64) -> Vec<String> {
    let mut warnings = Vec::new();
    if session.protocol < 2 {
        warnings.push(format!(
            "the {label} sidecar speaks protocol v1: the pitch shift and the reference voice are not sent to it"
        ));
    } else {
        if has_reference && !session.reference_used {
            warnings.push(format!("the {label} sidecar does not use a reference voice; the voice's reference is ignored"));
        }
        // The zero-shot recipes follow the reference voice's register by design (X-VC, Seed-VC);
        // only a server the user runs is reported for ignoring the pitch shift.
        if !recipe && !session.pitch() {
            warnings.push(format!("the {label} sidecar does not apply the pitch shift"));
        }
    }
    if let Some(compute) = session.latency_ms {
        let frame = session.frame_ms();
        if compute > frame {
            warnings.push(format!(
                "the {label} sidecar needs about {compute:.0} ms per {frame:.0} ms frame: it is slower than real time on this PC, so part of the voice will be muted (try a lighter setting or a faster device)"
            ));
        }
        if compute >= budget_ms {
            warnings.push(format!(
                "the {label} sidecar needs about {compute:.0} ms per frame, more than the {budget_ms:.0} ms latency budget: its frames will be muted; raise the latency budget or the voice's timeout"
            ));
        }
    }
    warnings
}

/// Connect (or start the recipe's helper) and wrap it as a model; stops waiting once
/// `request.cancel` is set ([`create_cancellable`]).
pub fn create(request: FamilyRequest) -> Result<FamilyEngine, String> {
    let cancel = request.cancel.clone();
    create_cancellable(request, cancel)
}

/// [`create`] that gives up soon after `cancel` is set (the voice changer no longer wants the
/// voice: another was selected, or it stopped), with the factory's `LOAD_CANCELLED`: a helper's
/// start (minutes on a first run) and the handshake wait in short slices, and a start nobody needs
/// is called off.
pub fn create_cancellable(request: FamilyRequest, cancel: Option<Arc<AtomicBool>>) -> Result<FamilyEngine, String> {
    if cancel.as_ref().is_some_and(|c| c.load(Ordering::Relaxed)) {
        return Err(LOAD_CANCELLED.to_string());
    }
    let options = options_of(&request.manifest)?;
    let budget_ms = match options.timeout_ms {
        Some(t) => f64::from(t),
        None if request.latency_budget_ms.is_finite() && request.latency_budget_ms > 0.0 => request.latency_budget_ms,
        None => DEFAULT_BUDGET_MS,
    }
    .clamp(f64::from(MIN_TIMEOUT_MS), f64::from(MAX_TIMEOUT_MS));
    let recipe = recipes::recipe(&options.recipe);
    if recipe.is_some_and(|r| r.needs_reference) && request.reference.is_none() {
        return Err(format!(
            "choose a reference voice from the voice library for this {} voice",
            recipe.map_or("sidecar", |r| r.label)
        ));
    }
    let label = recipe.map_or("custom", |r| r.label).to_string();
    let app_data = request.app_data.clone().or_else(crate::services::diagnostics::app_data_dir);
    let (endpoint, lease): (EndpointFn, _) = match recipe {
        Some(r) if options.url.is_empty() => {
            let script = recipes::find_script(r, app_data.as_deref())?;
            let spec = recipes::helper_spec(r, &options, &script, &request.ep);
            let lease = recipes::lease(spec, r, cancel.as_deref())?;
            let name = lease.name().to_string();
            // Resolved on every (re)connect: a restarted helper may listen on another port.
            let endpoint: EndpointFn = Arc::new(move || {
                let h = process_host::handle(&name).ok_or("the sidecar helper is not running")?;
                let port = h.port.ok_or("the sidecar helper has no port")?;
                Ok(Endpoint {
                    url: recipes::helper_url(port),
                    api_key: h.api_key,
                })
            });
            (endpoint, Some(lease))
        }
        _ => (ClientConfig::fixed(options.url.clone(), None), None),
    };
    let frame_ms = proposed_frame_ms(recipe, &request);
    let mut config = ClientConfig::new(options.recipe.clone(), endpoint);
    config.input_rates = INPUT_RATES.to_vec();
    config.frame_ms = frame_ms;
    config.frame_timeout = Duration::from_secs_f64(budget_ms / 1000.0);
    // What the hello offers (the server may pick another frame: the model sizes its own window
    // from the negotiated one, `SidecarModel::new`).
    config.max_in_flight = in_flight_window(budget_ms, frame_ms);
    config.cancel = cancel.clone();
    config.params = Map::from_iter([
        ("formant_semitones".to_string(), json!(0.0)),
        ("speaker_id".to_string(), json!(0)),
    ]);
    config.reference = request.reference.as_ref().map(reference_source);
    // A v1 server could never receive the reference a recipe converts toward.
    config.allow_v1 = !recipe.is_some_and(|r| r.needs_reference);
    // A custom server may load its model when the first session opens.
    config.handshake_timeout = Duration::from_secs(if lease.is_some() { 30 } else { 120 });
    let client = SidecarClient::connect(config).map_err(|e| if e == CONNECT_CANCELLED { LOAD_CANCELLED.to_string() } else { e })?;
    let session = client.session().clone();

    let warnings = load_warnings(&label, &session, recipe.is_some(), request.reference.is_some(), budget_ms);
    // GPU memory the sidecar holds (one claim per helper; a custom server's for as long as the
    // voice is loaded).
    let on_gpu = session.device.as_deref().is_some_and(|d| d != "cpu");
    let vram_mb = session
        .vram_mb
        .map(|m| m.round() as u64)
        .or_else(|| on_gpu.then(|| recipe.map_or(CUSTOM_VRAM_HINT_MB, |r| r.vram_hint_mb)));
    let custom_claim = match (&lease, vram_mb) {
        (Some(l), Some(mb)) => {
            l.claim_vram(mb);
            None
        }
        (None, Some(mb)) => Some(resources::claim_external_vram(Subsystem::Vc, &format!("voice changer sidecar ({label})"), mb)),
        _ => None,
    };
    let device_block_ms = if request.device_rate > 0 {
        f64::from(request.chunk_frames) * 1000.0 / f64::from(request.device_rate)
    } else {
        0.0
    };
    let outcome = sidecar_outcome(&request.ep, session.device.as_deref());
    let m = model::SidecarModel::new(client, label.clone(), budget_ms, device_block_ms, lease, custom_claim);
    let model_delay_16k = m.start_delay_16k();
    tracing::info!(
        "voice changer: {label} sidecar voice ready: v{} {} -> {} Hz, {:.0} ms frames, starts {:.0} ms behind, budget {budget_ms:.0} ms",
        session.protocol,
        session.input_rate,
        session.output_rate,
        session.frame_ms(),
        model_delay_16k as f64 / 16.0
    );
    let levers = EffectiveLevers {
        precision: request.precision,
        notes: vec!["sidecar: the helper process runs its own runtime; Sigil's GPU levers do not apply to it".into()],
        ..EffectiveLevers::default()
    };
    Ok(FamilyEngine {
        model: Box::new(m),
        levers,
        outcomes: vec![("vc.sidecar".into(), outcome)],
        warnings,
        model_delay_16k,
        reference: None,
        min_output_buffer_ms: None,
    })
}

/// Cheap checks of one slot: valid options, and for a recipe Sigil starts, its script, the
/// interpreter (when given as a path) and the model folder.
pub fn check_slot(_slot_dir: &Path, manifest: &RvcManifest) -> Result<(), String> {
    let options = options_of(manifest)?;
    let Some(recipe) = recipes::recipe(&options.recipe) else {
        return Ok(());
    };
    if !options.url.is_empty() {
        return Ok(());
    }
    recipes::find_script(recipe, crate::services::diagnostics::app_data_dir().as_deref())?;
    let python = options.python.trim();
    if Path::new(python).is_absolute() && !Path::new(python).is_file() {
        return Err(format!("the Python interpreter {python} was not found"));
    }
    let dir = options.model_dir.trim();
    if !dir.is_empty() && !Path::new(dir).is_dir() {
        return Err(format!("the {} model folder {dir} was not found", recipe.label));
    }
    Ok(())
}

/// Validate and normalize [`SidecarV2Options`].
pub fn normalize_options(options: &serde_json::Value) -> Result<serde_json::Value, String> {
    let mut o: SidecarV2Options = serde_json::from_value(options.clone()).map_err(|e| format!("invalid sidecar options: {e}"))?;
    o.recipe = o.recipe.trim().to_ascii_lowercase();
    if !RECIPES.contains(&o.recipe.as_str()) {
        return Err(format!("unknown sidecar recipe {:?} (one of {})", o.recipe, RECIPES.join(", ")));
    }
    o.url = o.url.trim().to_string();
    if !o.url.is_empty() && !loopback_ws_url(&o.url) {
        return Err("a sidecar URL must be a ws:// address on this PC (127.0.0.1, localhost or [::1])".into());
    }
    if o.recipe == "custom" && o.url.is_empty() {
        return Err("a custom sidecar needs its ws://127.0.0.1:<port> URL".into());
    }
    o.timeout_ms = o.timeout_ms.map(|t| t.clamp(MIN_TIMEOUT_MS, MAX_TIMEOUT_MS));
    o.python = o.python.trim().to_string();
    o.model_dir = o.model_dir.trim().to_string();
    recipes::validate_python(&o.python)?;
    recipes::validate_model_dir(&o.model_dir)?;
    recipes::validate_args(&o.args)?;
    serde_json::to_value(o).map_err(|e| e.to_string())
}

/// A sidecar slot holds only its manifest (the recipe or URL is in its options); a recipe whose
/// script is not installed yet is created with a warning.
pub fn create_slot(_request: &FamilySlotRequest, ctx: &FamilySlotContext<'_>) -> Result<CreatedSlot, String> {
    let mut created = super::default_create_slot(FamilyKind::SidecarV2, ctx)?;
    if let Ok(o) = serde_json::from_value::<SidecarV2Options>(ctx.options.clone()) {
        if let Some(r) = recipes::recipe(&o.recipe).filter(|_| o.url.is_empty()) {
            if let Err(e) = recipes::find_script(r, ctx.app_data) {
                created.warnings.push(e);
            }
        }
    }
    Ok(created)
}

/// Sidecar voices are never imported from files.
pub fn run_import_job(_job: &ImportJob) -> Result<String, String> {
    Err("job: sidecar voices are configured, not imported".into())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::services::neural_audio::ort_session::VcSessionOptions;
    use crate::services::neural_audio::sidecar_bridge::test_server::{Behavior, TestServer};
    use crate::services::neural_audio::vc::engine::options::{F0Range, PrecisionMode};
    use crate::services::neural_audio::vc::traits::ConversionCtx;
    use crate::services::neural_audio::RvcPresetProfile;

    #[test]
    fn only_loopback_websockets_are_accepted() {
        for ok in ["ws://127.0.0.1:18888/vc", "ws://localhost:9", "WS://[::1]:80/x", "ws://127.0.0.1"] {
            assert!(loopback_ws_url(ok), "{ok}");
        }
        for bad in ["wss://127.0.0.1:1", "ws://192.168.1.2:1", "ws://127.0.0.1.evil.com:1", "http://127.0.0.1:1", "ws://user@127.0.0.1:1", ""] {
            assert!(!loopback_ws_url(bad), "{bad}");
        }
    }

    #[test]
    fn options_are_normalized_or_refused() {
        let o = normalize_options(&serde_json::json!({"recipe": " XVC ", "timeout_ms": 1})).unwrap();
        assert_eq!(o["recipe"], "xvc");
        assert_eq!(o["timeout_ms"], 20);
        assert!(normalize_options(&serde_json::json!({})).is_err(), "custom needs a URL");
        assert!(normalize_options(&serde_json::json!({"url": "ws://10.0.0.1:1"})).is_err());
        assert!(normalize_options(&serde_json::json!({"recipe": "rm -rf"})).is_err());
        assert!(normalize_options(&serde_json::json!({"url": "ws://127.0.0.1:1/vc"})).is_ok());
        // Recipe settings: the bind and the key stay Sigil's, paths are full paths.
        assert!(normalize_options(&serde_json::json!({"recipe": "seedvc", "args": ["--host", "0.0.0.0"]})).is_err());
        assert!(normalize_options(&serde_json::json!({"recipe": "seedvc", "python": "venv/python"})).is_err());
        assert!(normalize_options(&serde_json::json!({"recipe": "seedvc", "model_dir": "models"})).is_err());
        let o = normalize_options(&serde_json::json!({"recipe": "seedvc", "python": " C:/v/python.exe ", "model_dir": "D:/m",
            "args": ["--steps", "6"], "timeout_ms": 9000, "url": "ws://127.0.0.1:9/vc"}))
        .unwrap();
        assert_eq!((o["python"].as_str(), o["timeout_ms"].as_u64()), (Some("C:/v/python.exe"), Some(5000)));
        assert_eq!(o["url"], "ws://127.0.0.1:9/vc", "a recipe may attach to a running server");
    }

    #[test]
    fn the_family_is_available_and_answers_pitch() {
        assert!(available() && FamilyKind::SidecarV2.available());
        let i = info();
        assert_eq!((i.family, i.creation, i.controls), ("sidecar_v2", CREATION_SIDECAR, &["pitch"][..]));
        assert!(i.experimental && !i.needs_pitch && !i.needs_content_encoder);
    }

    #[test]
    fn the_sidecar_device_is_reported_as_an_ep_outcome() {
        assert_eq!(sidecar_outcome("cuda", Some("cuda:0")), EpOutcome::granted("cuda"));
        let o = sidecar_outcome("cuda", Some("cpu"));
        assert_eq!((o.active.as_str(), o.is_gpu()), ("cpu", false));
        assert_eq!(sidecar_outcome("dml", Some("directml")).active, "dml");
        let o = sidecar_outcome("cpu", None);
        assert_eq!(o.active, "cpu");
        assert!(o.reason.is_some());
        assert!(sidecar_outcome("cpu", Some("mps")).is_gpu());
    }

    fn request(manifest: RvcManifest, budget_ms: f64) -> FamilyRequest {
        FamilyRequest {
            kind: FamilyKind::SidecarV2,
            slot_dir: std::env::temp_dir(),
            manifest,
            app_data: None,
            ep: "cpu".into(),
            session: VcSessionOptions::default(),
            levers: EffectiveLevers::default(),
            precision: PrecisionMode::Auto,
            preset: RvcPresetProfile::Streamer,
            block_ms: 0.0,
            chunk_frames: 480,
            device_rate: 48_000,
            extra_context_ms: 0.0,
            lookahead_ms: 0.0,
            f0_range: F0Range::Speech,
            silence_skip: false,
            noise_seed: 0,
            pitch: None,
            pitch_method: String::new(),
            content_encoder: None,
            reference: None,
            latency_budget_ms: budget_ms,
            cancel: None,
        }
    }

    /// A custom sidecar voice through the family registry: options from the manifest, the budget
    /// from the request, the session's device as the EP outcome, and a working model.
    #[test]
    fn a_custom_sidecar_voice_loads_through_the_registry() {
        let mut b = Behavior::delayed(Duration::from_millis(30));
        b.welcome["device"] = json!("cuda");
        b.welcome["vram_mb"] = json!(512.0);
        b.welcome["params"] = json!(["speaker_id"]);
        let server = TestServer::spawn(b);
        let manifest = super::super::family_manifest(FamilyKind::SidecarV2, "Mine", json!({"url": server.url}));
        let mut engine = super::super::create(request(manifest, 500.0)).unwrap();
        assert_eq!(engine.outcomes[0].0, "vc.sidecar");
        assert_eq!(engine.outcomes[0].1.active, "cuda");
        assert!(engine.warnings.is_empty(), "{:?}", engine.warnings);
        // The 100 ms frame (the device block here is 10 ms) plays in 20 ms slots.
        assert_eq!(engine.model.fixed_block(), Some(320));
        assert!(engine.model_delay_16k >= 1600, "at least one 100 ms frame: {}", engine.model_delay_16k);
        let audio = vec![0.25f32; 1600];
        let mut loud = 0;
        let start = std::time::Instant::now();
        for k in 0..40u64 {
            let due = start + Duration::from_millis(20 * k);
            if let Some(d) = due.checked_duration_since(std::time::Instant::now()) {
                std::thread::sleep(d);
            }
            let ctx = ConversionCtx {
                mono_16k: &audio,
                new_frames: 320,
                pitch_shift_semitones: -2.0,
                index_rate: 0.0,
                protect: 0.33,
                speaker_id: 3,
                stream_pos_16k: k * 320,
                reference: None,
                singing: Default::default(),
            };
            let out = engine.model.convert(&ctx).unwrap_or_else(|e| panic!("tick {k}: {e}"));
            assert_eq!(out.len(), 320);
            if out.iter().all(|&x| (x - 0.25).abs() < 1e-6) {
                loud += 1;
            }
        }
        assert!(loud >= 20, "the echo plays after about one frame plus the round trip: {loud} of 40 ticks");
        let log = server.log();
        assert!(log.params.iter().any(|p| p["speaker_id"] == 3), "{:?}", log.params);
        assert!(!log.frames.is_empty() && log.frames.iter().all(|f| f.2 == -2.0 && f.3 == 1600));
        assert_eq!(log.hello.as_ref().unwrap()["frame_timeout_ms"], 500);
    }

    /// A server slower than the budget, or slower than real time, is loaded with a warning that
    /// names the numbers (its frames would be muted).
    #[test]
    fn a_server_slower_than_the_budget_loads_with_a_warning() {
        let mut b = Behavior::delayed(Duration::from_millis(10));
        b.welcome["latency_ms"] = json!(450.0);
        let server = TestServer::spawn(b);
        let manifest = super::super::family_manifest(FamilyKind::SidecarV2, "Slow", json!({"url": server.url}));
        let engine = super::super::create(request(manifest, 300.0)).unwrap();
        assert!(engine.warnings.iter().any(|w| w.contains("450 ms") && w.contains("300 ms")), "{:?}", engine.warnings);
        assert!(engine.warnings.iter().any(|w| w.contains("slower than real time")), "{:?}", engine.warnings);
    }

    #[test]
    fn recipes_need_a_reference_and_invalid_options_fail_the_load() {
        let manifest = super::super::family_manifest(FamilyKind::SidecarV2, "X", json!({"recipe": "xvc"}));
        let err = create(request(manifest, 300.0)).err().unwrap();
        assert!(err.contains("reference voice"), "{err}");
        let manifest = super::super::family_manifest(FamilyKind::SidecarV2, "X", json!({"recipe": "custom"}));
        assert!(check_slot(Path::new("."), &manifest).is_err());
        let manifest = super::super::family_manifest(FamilyKind::SidecarV2, "X", json!({"url": "ws://127.0.0.1:1/vc"}));
        assert!(check_slot(Path::new("."), &manifest).is_ok());
        let manifest =
            super::super::family_manifest(FamilyKind::SidecarV2, "X", json!({"recipe": "seedvc", "model_dir": "C:/definitely/not/here"}));
        let problem = check_slot(Path::new("."), &manifest).unwrap_err();
        assert!(problem.contains("model folder") || problem.contains("not installed"), "{problem}");
    }

    #[test]
    fn a_custom_server_that_is_not_running_fails_promptly() {
        let port = std::net::TcpListener::bind("127.0.0.1:0").unwrap().local_addr().unwrap().port();
        let manifest = super::super::family_manifest(FamilyKind::SidecarV2, "X", json!({"url": format!("ws://127.0.0.1:{port}/vc")}));
        let t0 = std::time::Instant::now();
        let err = create(request(manifest, 300.0)).err().unwrap();
        assert!(err.contains("sidecar"), "{err}");
        assert!(t0.elapsed() < Duration::from_secs(10));
    }

    /// Manual end-to-end check of a recipe helper (the real Python script under `process_host`):
    /// `SIGIL_TEST_SIDECAR_PYTHON=<python with websockets and numpy> cargo test --manifest-path
    /// src-tauri/Cargo.toml sidecar_recipe_live -- --ignored --nocapture`. Runs `seedvc_sidecar.py
    /// --echo --echo-delay-ms 120` (the protocol and the helper without a model).
    #[test]
    #[ignore = "starts a Python helper; set SIGIL_TEST_SIDECAR_PYTHON"]
    fn sidecar_recipe_live() {
        let Ok(python) = std::env::var("SIGIL_TEST_SIDECAR_PYTHON") else {
            eprintln!("SIGIL_TEST_SIDECAR_PYTHON is not set; skipped");
            return;
        };
        let wav = std::env::temp_dir().join(format!("sigil-sidecar-ref-{}.wav", std::process::id()));
        {
            let spec = hound::WavSpec { channels: 1, sample_rate: 22_050, bits_per_sample: 16, sample_format: hound::SampleFormat::Int };
            let mut w = hound::WavWriter::create(&wav, spec).unwrap();
            for i in 0..22_050 * 4 {
                w.write_sample(((i as f32 * 0.05).sin() * 8000.0) as i16).unwrap();
            }
            w.finalize().unwrap();
        }
        let manifest = super::super::family_manifest(
            FamilyKind::SidecarV2,
            "Live",
            json!({"recipe": "seedvc", "python": python, "args": ["--echo", "--echo-delay-ms", "120"]}),
        );
        let mut req = request(manifest, 600.0);
        req.reference = Some(ReferenceClip::from_wav(&wav).unwrap());
        let t0 = std::time::Instant::now();
        let engine = create(req).unwrap_or_else(|e| panic!("{e}"));
        eprintln!(
            "helper + session ready in {:?}; start delay {:.0} ms; warnings {:?}",
            t0.elapsed(),
            engine.model_delay_16k as f64 / 16.0,
            engine.warnings
        );
        // The production path: the stream harness, 20 ms device blocks at 48 kHz in real time.
        let mut h = crate::services::neural_audio::vc::stream::VcStreamHarness::new(engine.model, 48_000, 48_000, 1, 1, 0, 320, 960).unwrap();
        let (mut first, mut errors, mut loud, mut longest) = (None, 0, 0, Duration::ZERO);
        let start = std::time::Instant::now();
        let blocks = 200;
        for k in 0..blocks {
            let due = start + Duration::from_millis(20 * k as u64);
            if let Some(d) = due.checked_duration_since(std::time::Instant::now()) {
                std::thread::sleep(d);
            }
            let block: Vec<f32> = (0..960).map(|i| 0.3 * (2.0 * std::f32::consts::PI * 220.0 * (k * 960 + i) as f32 / 48_000.0).sin()).collect();
            let c0 = std::time::Instant::now();
            let r = h.push_and_convert(&block);
            longest = longest.max(c0.elapsed());
            match r {
                Ok(out) if out.iter().map(|x| x * x).sum::<f32>() / out.len().max(1) as f32 > 0.01 => {
                    loud += 1;
                    first.get_or_insert(k);
                }
                Ok(_) => {}
                Err(e) => {
                    errors += 1;
                    eprintln!("block {k}: {e}");
                }
            }
        }
        let first = first.expect("audio came back");
        eprintln!(
            "first audio after {} ms; {loud} loud of {} blocks after it; {errors} muted; longest call {longest:?}",
            first * 20,
            blocks - first
        );
        assert!(longest < Duration::from_millis(50), "no call waited for the helper: {longest:?}");
        assert_eq!(errors, 0);
        assert!(loud + 3 >= blocks - first, "continuous after the start");
        drop(h);
        let _ = std::fs::remove_file(&wav);
    }

    /// Manual end-to-end check of a single-stream helper (StreamVoiceAnon's script in echo mode,
    /// `--echo-single-stream`) under `process_host`: voice B loads on the helper voice A plays on
    /// (shared, not refused), B's stream takes the model over, A then plays silence without
    /// errors, and B plays. `SIGIL_TEST_SIDECAR_PYTHON=<python with websockets and numpy> cargo
    /// test --manifest-path src-tauri/Cargo.toml sidecar_single_stream_live -- --ignored --nocapture`.
    #[test]
    #[ignore = "starts a Python helper; set SIGIL_TEST_SIDECAR_PYTHON"]
    fn sidecar_single_stream_live() {
        let Ok(python) = std::env::var("SIGIL_TEST_SIDECAR_PYTHON") else {
            eprintln!("SIGIL_TEST_SIDECAR_PYTHON is not set; skipped");
            return;
        };
        let wav = std::env::temp_dir().join(format!("sigil-sidecar-ss-ref-{}.wav", std::process::id()));
        {
            let spec = hound::WavSpec { channels: 1, sample_rate: 16_000, bits_per_sample: 16, sample_format: hound::SampleFormat::Int };
            let mut w = hound::WavWriter::create(&wav, spec).unwrap();
            for i in 0..16_000 * 3 {
                w.write_sample(((i as f32 * 0.05).sin() * 8000.0) as i16).unwrap();
            }
            w.finalize().unwrap();
        }
        let voice = || {
            let manifest = super::super::family_manifest(
                FamilyKind::SidecarV2,
                "Privacy",
                json!({"recipe": "streamvoiceanon", "python": python, "args": ["--echo", "--echo-single-stream"]}),
            );
            let mut req = request(manifest, 600.0);
            req.reference = Some(ReferenceClip::from_wav(&wav).unwrap());
            create(req).unwrap_or_else(|e| panic!("{e}"))
        };
        let mut a = voice().model;
        let tone = |k: u64| -> Vec<f32> { (0..1600).map(|i| 0.3 * (((k * 320 + i) as f32) * 0.08).sin()).collect() };
        let tick = |m: &mut Box<dyn crate::services::neural_audio::vc::traits::VoiceConversionModel>, k: u64| {
            let audio = tone(k);
            let slot = m.fixed_block().unwrap();
            let ctx = ConversionCtx {
                mono_16k: &audio,
                new_frames: slot,
                pitch_shift_semitones: 0.0,
                index_rate: 0.0,
                protect: 0.33,
                speaker_id: 0,
                stream_pos_16k: k * slot as u64,
                reference: None,
                singing: Default::default(),
            };
            m.convert(&ctx).map(|o| (o.iter().map(|x| x * x).sum::<f32>() / o.len().max(1) as f32).sqrt())
        };
        // Real time: one slot of input per call.
        let slot_s = a.fixed_block().unwrap() as f64 / f64::from(a.input_sample_rate());
        let start = std::time::Instant::now();
        let pace = |k: u64| {
            if let Some(d) = (start + Duration::from_secs_f64(slot_s * k as f64)).checked_duration_since(std::time::Instant::now()) {
                std::thread::sleep(d);
            }
        };
        let mut k = 0u64;
        let mut a_loud = 0;
        while k < 60 {
            pace(k);
            if tick(&mut a, k).unwrap_or_else(|e| panic!("A tick {k}: {e}")) > 0.05 {
                a_loud += 1;
            }
            k += 1;
        }
        assert!(a_loud > 20, "voice A plays: {a_loud}");
        // Voice B loads on the same helper while A plays (refused before the fix).
        let mut b = voice().model;
        b.reset_stream();
        let (mut a_errors, mut a_loud_late, mut b_loud) = (0, 0, 0);
        while k < 160 {
            pace(k);
            let ra = tick(&mut a, k);
            let rb = tick(&mut b, k).unwrap_or_else(|e| panic!("B tick {k}: {e}"));
            match ra {
                Err(_) => a_errors += 1,
                Ok(r) if r > 0.05 && k > 100 => a_loud_late += 1,
                Ok(_) => {}
            }
            if rb > 0.05 {
                b_loud += 1;
            }
            k += 1;
        }
        eprintln!("A loud {a_loud}, A errors {a_errors}, A loud after the switch {a_loud_late}, B loud {b_loud}");
        assert_eq!(a_errors, 0, "the replaced voice is silent, not failing");
        assert_eq!(a_loud_late, 0, "B holds the model");
        assert!(b_loud > 40, "voice B plays: {b_loud}");
        drop((a, b));
        let _ = std::fs::remove_file(&wav);
    }

    /// Mono samples and rate of a WAV file (tests).
    fn read_wav(path: &Path) -> (Vec<f32>, u32) {
        let mut r = hound::WavReader::open(path).unwrap_or_else(|e| panic!("{}: {e}", path.display()));
        let spec = r.spec();
        let ch = usize::from(spec.channels.max(1));
        let all: Vec<f32> = match spec.sample_format {
            hound::SampleFormat::Float => r.samples::<f32>().map(Result::unwrap).collect(),
            hound::SampleFormat::Int => {
                let scale = (1u64 << (spec.bits_per_sample - 1)) as f32;
                r.samples::<i32>().map(|s| s.unwrap() as f32 / scale).collect()
            }
        };
        (all.chunks(ch).map(|c| c.iter().sum::<f32>() / ch as f32).collect(), spec.sample_rate)
    }

    /// Manual measurement of the Seed-VC recipe with the real model (Sigil's ONNX export), the
    /// helper under `process_host`, and the voice in the stream harness in real time: the delay a
    /// listener hears (onset of the converted speech after the onset of the input), muted blocks,
    /// and that no call waits. Set `SIGIL_TEST_SIDECAR_PYTHON` (onnxruntime-gpu or -directml,
    /// numpy, websockets), `SIGIL_TEST_SEEDVC_GRAPHS` (the export folder) and
    /// `SIGIL_TEST_SIDECAR_SPEECH` (a speech WAV, also the reference unless
    /// `SIGIL_TEST_SIDECAR_REFERENCE` names another); optional `SIGIL_TEST_SIDECAR_EP`
    /// (`cuda` / `dml` / `cpu`), `SIGIL_TEST_SIDECAR_DLL_DIR` (CUDA 13 + cuDNN 9 folder),
    /// `SIGIL_TEST_SEEDVC_PRESET` and `SIGIL_TEST_SIDECAR_BUDGET_MS` (default 600). Run with
    /// `cargo test --manifest-path src-tauri/Cargo.toml sidecar_seedvc_model_live -- --ignored --nocapture`.
    #[test]
    #[ignore = "starts the Seed-VC sidecar on a real model; see the doc comment"]
    fn sidecar_seedvc_model_live() {
        let env = |k: &str| std::env::var(k).ok().filter(|v| !v.trim().is_empty());
        let (Some(python), Some(graphs), Some(speech)) =
            (env("SIGIL_TEST_SIDECAR_PYTHON"), env("SIGIL_TEST_SEEDVC_GRAPHS"), env("SIGIL_TEST_SIDECAR_SPEECH"))
        else {
            eprintln!("SIGIL_TEST_SIDECAR_PYTHON / SIGIL_TEST_SEEDVC_GRAPHS / SIGIL_TEST_SIDECAR_SPEECH not set; skipped");
            return;
        };
        // Show Sigil's log lines (the voice's playout summary when it unloads).
        let _ = tracing_subscriber::fmt().with_test_writer().with_max_level(tracing::Level::INFO).try_init();
        let ep = env("SIGIL_TEST_SIDECAR_EP").unwrap_or_else(|| "cuda".into());
        let preset = env("SIGIL_TEST_SEEDVC_PRESET").unwrap_or_else(|| "upstream_readme".into());
        let budget: f64 = env("SIGIL_TEST_SIDECAR_BUDGET_MS").and_then(|b| b.parse().ok()).unwrap_or(600.0);
        let mut args = vec!["--preset".to_string(), preset.clone()];
        if let Some(d) = env("SIGIL_TEST_SIDECAR_DLL_DIR") {
            args.extend(["--dll-dir".to_string(), d]);
        }
        let manifest = super::super::family_manifest(
            FamilyKind::SidecarV2,
            "Seed-VC live",
            json!({"recipe": "seedvc", "python": python, "model_dir": graphs, "args": args}),
        );
        let mut req = request(manifest, budget);
        req.ep = ep.clone();
        let reference = env("SIGIL_TEST_SIDECAR_REFERENCE").unwrap_or_else(|| speech.clone());
        req.reference = Some(ReferenceClip::from_wav(Path::new(&reference)).unwrap());
        let t0 = std::time::Instant::now();
        let engine = create(req).unwrap_or_else(|e| panic!("{e}"));
        eprintln!(
            "[{ep} {preset}] helper + session ready in {:.1} s; outcome {:?}; start delay {:.0} ms; warnings {:?}",
            t0.elapsed().as_secs_f64(),
            engine.outcomes,
            engine.model_delay_16k as f64 / 16.0,
            engine.warnings
        );
        let (voice, rate) = read_wav(Path::new(&speech));
        // 1 s of silence, the speech, 1 s of silence; 20 ms device blocks in real time.
        let block = (rate / 50) as usize;
        let mut input = vec![0.0f32; rate as usize];
        input.extend_from_slice(&voice);
        input.resize(input.len() + rate as usize, 0.0);
        let mut h = crate::services::neural_audio::vc::stream::VcStreamHarness::new(engine.model, rate, 48_000, 1, 1, 0, 320, block).unwrap();
        let db = |x: &[f32]| 10.0 * ((x.iter().map(|v| v * v).sum::<f32>() / x.len().max(1) as f32) + 1e-12).log10();
        // Loudness envelopes (dB per 20 ms block) of what went in and what came out.
        let (mut env_in, mut env_out) = (Vec::new(), Vec::new());
        let (mut errors, mut longest, mut calls) = (0, Duration::ZERO, Vec::new());
        let start = std::time::Instant::now();
        for (k, chunk) in input.chunks_exact(block).enumerate() {
            let due = start + Duration::from_millis(20 * k as u64);
            if let Some(d) = due.checked_duration_since(std::time::Instant::now()) {
                std::thread::sleep(d);
            }
            env_in.push(db(chunk).max(-80.0));
            let c0 = std::time::Instant::now();
            let r = h.push_and_convert(chunk);
            let took = c0.elapsed();
            longest = longest.max(took);
            calls.push(took.as_secs_f64() * 1000.0);
            match r {
                Ok(out) => env_out.push(db(&out).max(-80.0)),
                Err(e) => {
                    env_out.push(-80.0);
                    errors += 1;
                    if errors <= 5 {
                        eprintln!("block {k}: {e}");
                    }
                }
            }
        }
        calls.sort_by(f64::total_cmp);
        let p99 = calls[(calls.len() * 99 / 100).min(calls.len() - 1)];
        // The delay a listener hears: the shift (in blocks) that best lines the output's syllable
        // envelope up with the input's (converted speech keeps the timing of the words).
        let corr = |lag: usize| {
            let n = env_in.len() - lag;
            let (a, b) = (&env_in[..n], &env_out[lag..lag + n]);
            let (ma, mb) = (a.iter().sum::<f32>() / n as f32, b.iter().sum::<f32>() / n as f32);
            let cov: f32 = a.iter().zip(b).map(|(x, y)| (x - ma) * (y - mb)).sum();
            let (va, vb): (f32, f32) = (a.iter().map(|x| (x - ma).powi(2)).sum(), b.iter().map(|y| (y - mb).powi(2)).sum());
            cov / (va * vb).sqrt().max(1e-9)
        };
        let (best, r) = (0..60usize).map(|l| (l, corr(l))).max_by(|a, b| a.1.total_cmp(&b.1)).unwrap();
        eprintln!(
            "[{ep} {preset}] heard delay {} ms (envelope cross-correlation {r:.2}); {errors} muted blocks of {}; call p99 {p99:.2} ms, longest {longest:?}",
            best * 20,
            calls.len()
        );
        // The playout summary (lag, late slots) is logged when the voice unloads.
        drop(h);
        assert!(longest < Duration::from_millis(50), "no call waited for the sidecar: {longest:?}");
    }
}
