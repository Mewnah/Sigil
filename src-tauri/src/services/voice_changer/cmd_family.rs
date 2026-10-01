//! P5 voice-family slots IPC (P5-S0, scaffold-owned): create a family voice (a zero-shot voice from
//! a library voice, an imported DDSP-SVC / so-vits-svc voice, a sidecar voice) and edit its
//! options. The families fill the slot themselves (`vc::families::create_slot`); this module adds
//! what every slot operation shares with the importer (`cmd_import`): one slot operation at a time,
//! a job id with `voice_changer:import_progress` events and `voice_changer_cancel_import`, a staging
//! folder renamed into place (a failed or cancelled creation leaves nothing listed), the
//! `voice_changer:slots_changed` event, and optional activation.
//!
//! # Frozen by P5-S0 (command names, arguments and payloads; TS `ipc.ts` `VC_COMMANDS`)
//! - `voice_changer_create_family_slot { request: FamilySlotRequest } -> VcImportResult` (`cached`
//!   is always false). The request's `reference_voice_id` is stored in the new voice's `slot.json`
//!   (the per-voice `reference_voice_id` param); a DDSP-SVC (singing-only) voice starts with
//!   `f0_range: "singing"` and `pitch_extractor: "crepe"` there (P5-I0). Refused (`job: ...`) for an
//!   unknown family, a family this build does not run, invalid options, an import family without
//!   `source`, or an invalid reference id.
//! - `voice_changer_set_family_options { id, options } -> SlotInfo`: normalize `options` with the
//!   slot's family and store them as `manifest.family` (every other manifest field kept as it is).
//!   A loaded voice keeps the options it was loaded with until it loads again.

use std::path::{Path, PathBuf};
use std::sync::atomic::Ordering;
use std::time::Instant;

use tauri::{command, AppHandle, Manager, Runtime, State};

use super::cmd_import::{
    activate, blocking, emit_progress, emit_slots_changed, io_err, job_err, models_dir, slot_lock, templates_dir, JobTicket,
    VcImportResult,
};
use super::params::normalize_reference_voice_id;
use super::persist::{slot_settings_path, to_json_bytes, write_atomic, SlotSettingsFile, SETTINGS_VERSION};
use super::state::VoiceChangerState;
use crate::services::neural_audio::vc::engine::options::F0Range;
use crate::services::neural_audio::vc::families::{self, FamilyKind, FamilySlotContext, FamilySlotRequest, CREATION_IMPORT};
use crate::services::neural_audio::vc::import::slot::{self, SlotInfo};
use crate::services::neural_audio::vc::import::{self, clean_name};
use crate::services::neural_audio::vc::manifest::load_slot_manifest;

/// Default display name of a new voice of `kind` (data, not UI text: the user renames it).
fn family_title(kind: FamilyKind) -> &'static str {
    match kind {
        FamilyKind::DdspSvc => "DDSP-SVC voice",
        FamilyKind::SigilZs => "Zero-shot voice",
        FamilyKind::MeanVc => "MeanVC2 voice",
        FamilyKind::Sovits => "so-vits-svc voice",
        FamilyKind::SidecarV2 => "Sidecar voice",
    }
}

/// The new voice's display name: the request's, else the reference voice's (zero-shot), else the
/// imported file's stem, else the family's.
fn slot_name(request: &FamilySlotRequest, kind: FamilyKind, reference: Option<&str>, app_data: Option<&Path>) -> String {
    if let Some(n) = request.name.as_deref().and_then(clean_name) {
        return n;
    }
    if let (Some(id), Some(app_data)) = (reference, app_data) {
        if let Some(n) = crate::services::voice_library::reference_clip(app_data, id).ok().and_then(|c| clean_name(&c.name)) {
            return n;
        }
    }
    if let Some(n) = request
        .source
        .as_deref()
        .map(str::trim)
        .filter(|s| !s.is_empty())
        .and_then(|s| Path::new(s).file_stem().map(|n| n.to_string_lossy().into_owned()))
        .and_then(|n| clean_name(&n))
    {
        return n;
    }
    family_title(kind).to_string()
}

/// F0 method a new singing voice starts with (P5-I0, from P5-A's sung E2→D#6 sweep at +0 / +12
/// semitones: CREPE-tiny, SwiftF0 and fusion keep gross pitch errors at 0.00 % up to D#7, while
/// RMVPE reads notes above about 1.15 kHz an octave low (2.36 %) and FCPE up to two semitones flat).
/// CREPE-tiny is a 2 MB published download and costs less VRAM than fusion's RMVPE.
const SINGING_PITCH_EXTRACTOR: &str = "crepe";

/// The new voice's `slot.json` (the voice changer's per-voice file): its reference voice
/// (zero-shot), and for a singing-only family (DDSP-SVC) the singing F0 range with an F0 method that
/// covers it ([`SINGING_PITCH_EXTRACTOR`]). so-vits-svc voices are speech voices as often as
/// singing ones: they keep the user's F0 settings (the inspector suggests singing mode). `None` when
/// the voice needs none of it.
fn new_slot_settings(kind: FamilyKind, reference: &str) -> Option<SlotSettingsFile> {
    let singing = kind == FamilyKind::DdspSvc;
    if reference.is_empty() && !singing {
        return None;
    }
    Some(SlotSettingsFile {
        version: SETTINGS_VERSION,
        reference_voice_id: (!reference.is_empty()).then(|| reference.to_string()),
        f0_range: singing.then(|| F0Range::Singing.as_str().to_string()),
        pitch_extractor: singing.then(|| SINGING_PITCH_EXTRACTOR.to_string()),
        ..SlotSettingsFile::default()
    })
}

/// Write a new voice's `slot.json`.
fn write_slot_settings(slot_dir: &Path, file: &SlotSettingsFile) -> Result<(), String> {
    let bytes = to_json_bytes(file).ok_or_else(|| io_err("cannot encode slot.json"))?;
    write_atomic(&slot_settings_path(slot_dir), &bytes).map_err(|e| io_err(format!("{}: {e}", slot_dir.display())))
}

/// Create the voice in a staging folder and rename it into place (see the module docs).
async fn create_family_slot<R: Runtime>(app: &AppHandle<R>, request: FamilySlotRequest, ticket: &JobTicket) -> Result<VcImportResult, String> {
    let started = Instant::now();
    let job_id = ticket.id.clone();
    let kind = FamilyKind::from_name(&request.family).ok_or_else(|| job_err(format!("unknown voice family {:?}", request.family)))?;
    if !kind.available() {
        return Err(job_err(format!("the {} voice family is not available in this build yet", kind.as_str())));
    }
    let info = kind.info();
    let options = families::normalize_options(kind, request.options.as_ref()).map_err(job_err)?;
    let raw_reference = request.reference_voice_id.as_deref().map(str::trim).unwrap_or_default();
    let reference = normalize_reference_voice_id(raw_reference);
    if !raw_reference.is_empty() && reference.is_empty() {
        return Err(job_err(format!("invalid reference voice id {raw_reference:?}")));
    }
    if info.creation == CREATION_IMPORT && request.source.as_deref().map(str::trim).unwrap_or_default().is_empty() {
        return Err(job_err("choose a file to import"));
    }
    emit_progress(app, &job_id, "queued", 0.0, "Waiting for another voice operation to finish");
    let _guard = slot_lock().lock().await;
    ticket.check()?;
    let models = models_dir(app)?;
    let templates = templates_dir(app);
    let app_data = app.path().app_data_dir().ok();
    let exe = std::env::current_exe().ok();
    // The reference voice's name reads its clip's metadata (file I/O): off the async runtime.
    let name = {
        let (request, reference, app_data) = (request.clone(), reference.clone(), app_data.clone());
        blocking(move || Ok(slot_name(&request, kind, Some(reference.as_str()).filter(|r| !r.is_empty()), app_data.as_deref()))).await?
    };
    let settings = new_slot_settings(kind, &reference);
    let id = slot::new_slot_id(&models, &name);
    let final_dir = slot::slot_dir(&models, &id)?;
    let staging = models.join(format!(".{id}.sigil-new-family"));
    emit_progress(app, &job_id, "reading", 0.05, "Preparing the voice");
    let cancel = ticket.cancel.clone();
    let (app2, job2) = (app.clone(), job_id.clone());
    let models2 = models.clone();
    let created = blocking(move || {
        let _ = std::fs::remove_dir_all(&staging);
        std::fs::create_dir_all(&staging).map_err(|e| io_err(format!("{}: {e}", staging.display())))?;
        let progress = move |stage: &str, fraction: f64, message: &str| emit_progress(&app2, &job2, stage, fraction, message);
        let ctx = FamilySlotContext {
            models_dir: &models2,
            slot_dir: staging.clone(),
            name,
            options,
            app_data: app_data.as_deref(),
            templates_dir: templates.as_deref(),
            exe: exe.as_deref(),
            cancel: &cancel,
            progress: &progress,
        };
        let result = families::create_slot(kind, &request, &ctx).and_then(|created| {
            if cancel.load(Ordering::Acquire) {
                return Err(job_err("the import was cancelled"));
            }
            if let Some(settings) = &settings {
                write_slot_settings(&staging, settings)?;
            }
            if !staging.join(import::MANIFEST_FILE).is_file() {
                return Err(job_err(format!("the {} family wrote no manifest.json", kind.as_str())));
            }
            std::fs::rename(&staging, &final_dir).map_err(|e| io_err(format!("{}: {e}", final_dir.display())))?;
            Ok(created)
        });
        if result.is_err() {
            let _ = std::fs::remove_dir_all(&staging);
        }
        result
    })
    .await?;
    emit_progress(app, &job_id, "writing", 0.95, "Adding the voice");
    let info = {
        let (m, t) = (models.clone(), templates_dir(app));
        blocking(move || slot::slot_info_with(&m, &id, t.as_deref())).await?
    };
    Ok(VcImportResult {
        job_id,
        slot: info,
        cached: false,
        warnings: created.warnings,
        elapsed_ms: started.elapsed().as_millis() as u64,
    })
}

/// Create a P5 family voice (see the module docs).
#[command]
pub(super) async fn voice_changer_create_family_slot<R: Runtime>(
    app: AppHandle<R>,
    state: State<'_, VoiceChangerState>,
    request: FamilySlotRequest,
) -> Result<VcImportResult, String> {
    let ticket = JobTicket::new();
    let activate_after = request.activate;
    let result = create_family_slot(&app, request, &ticket).await;
    match &result {
        Ok(r) => {
            if activate_after {
                activate(&app, &state, &r.slot.id, Path::new(&r.slot.path));
            }
            emit_slots_changed(&app);
            emit_progress(&app, &ticket.id, "done", 1.0, "Added");
        }
        Err(e) => {
            tracing::warn!("voice family slot not created: {e}");
            emit_progress(&app, &ticket.id, "failed", 1.0, e.clone());
        }
    }
    result
}

/// Store a family voice's options (see the module docs).
#[command]
pub(super) async fn voice_changer_set_family_options<R: Runtime>(
    app: AppHandle<R>,
    id: String,
    options: serde_json::Value,
) -> Result<SlotInfo, String> {
    let _guard = slot_lock().lock().await;
    let models = models_dir(&app)?;
    let templates = templates_dir(&app);
    let info = blocking(move || {
        let dir: PathBuf = slot::slot_dir(&models, id.trim())?;
        let manifest_path = dir.join(import::MANIFEST_FILE);
        if !manifest_path.is_file() {
            return Err(job_err(format!("there is no voice slot {id:?}")));
        }
        let m = load_slot_manifest(&dir).map_err(job_err)?;
        let kind = m.slot_format().family().ok_or_else(|| job_err("only voice-family voices have family options"))?;
        let options = families::normalize_options(kind, Some(&options)).map_err(job_err)?;
        // Patch the JSON, so every other field (known or not) stays as it was.
        let bytes = std::fs::read(&manifest_path).map_err(|e| io_err(format!("{}: {e}", manifest_path.display())))?;
        let mut v: serde_json::Value = serde_json::from_slice(&bytes).map_err(|e| io_err(format!("{}: {e}", manifest_path.display())))?;
        let obj = v.as_object_mut().ok_or_else(|| io_err(format!("{}: not a JSON object", manifest_path.display())))?;
        obj.insert("family".into(), options);
        let out = serde_json::to_vec_pretty(&v).map_err(|e| io_err(e.to_string()))?;
        import::write_atomic(&manifest_path, &out).map_err(|e| e.to_string())?;
        slot::slot_info_with(&models, id.trim(), templates.as_deref())
    })
    .await?;
    emit_slots_changed(&app);
    Ok(info)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn new_voices_are_named_from_the_request_the_file_or_the_family() {
        let mut r = FamilySlotRequest { family: "sigil_zs".into(), ..Default::default() };
        assert_eq!(slot_name(&r, FamilyKind::SigilZs, None, None), "Zero-shot voice");
        r.source = Some("C:/voices/My Singer.pt".into());
        assert_eq!(slot_name(&r, FamilyKind::DdspSvc, None, None), "My Singer");
        r.name = Some("  Alto  ".into());
        assert_eq!(slot_name(&r, FamilyKind::DdspSvc, Some("alice"), None), "Alto");
        for k in FamilyKind::ALL {
            assert!(!family_title(k).is_empty());
        }
    }

    #[test]
    fn the_reference_is_written_to_the_new_voice_slot_json() {
        let dir = std::env::temp_dir().join(format!("sigil-vc-family-ref-{}-{}", std::process::id(), import::unix_ms()));
        std::fs::create_dir_all(&dir).unwrap();
        let file = new_slot_settings(FamilyKind::SigilZs, "alice").expect("a zero-shot voice has settings");
        write_slot_settings(&dir, &file).unwrap();
        let v: serde_json::Value = serde_json::from_slice(&std::fs::read(slot_settings_path(&dir)).unwrap()).unwrap();
        assert_eq!(v["reference_voice_id"], "alice");
        assert_eq!(v["version"], SETTINGS_VERSION);
        assert!(v["f0_range"].is_null() && v["pitch_extractor"].is_null(), "a speech voice keeps the user's F0 settings");
        let _ = std::fs::remove_dir_all(&dir);
    }

    /// A new DDSP-SVC voice starts in the singing F0 range with an F0 method that reaches D#7
    /// (P5-I0, P5-A's measurement); other new voices without a reference write no `slot.json`.
    #[test]
    fn singing_voices_start_with_a_singing_pitch_setup() {
        let ddsp = FamilyKind::DdspSvc.info();
        assert!(ddsp.singing && ddsp.needs_pitch);
        let file = new_slot_settings(FamilyKind::DdspSvc, "").expect("a singing voice has settings");
        assert_eq!(file.f0_range.as_deref(), Some("singing"));
        assert_eq!(file.pitch_extractor.as_deref(), Some(SINGING_PITCH_EXTRACTOR));
        assert_eq!(file.reference_voice_id, None);
        assert_eq!(super::super::params::normalize_pitch_extractor(SINGING_PITCH_EXTRACTOR), SINGING_PITCH_EXTRACTOR);
        assert!(new_slot_settings(FamilyKind::Sovits, "").is_none(), "so-vits voices keep the user's F0 settings");
    }
}
