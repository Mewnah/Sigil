//! Voice import and slot-store IPC: `.pth` / `.zip` import through the isolated worker
//! (`vc::import::worker`), the slot store (`vc::import::slot`), set-active, and auto-import of the
//! MIT default voice. Payloads are snake_case (like the rest of the voice-changer plugin);
//! arguments camelCase (Tauri). Errors are `"<code>: <message>"` with the importer's stable codes
//! (`vc::import::ErrorCode`), so the UI can map them to messages: slot-store requests that cannot
//! be served (an unknown slot, a bad name, a cancelled import, the voice in use) are `job`, file
//! system failures `io`.
//!
//! One import, merge, duplicate, rename or delete runs at a time (they share one lock): imports
//! are memory-hungry (the worker may use up to 3 GB), and slot ids and folders are never raced.
//! Before each import (and when the slot list is read while nothing runs) interrupted slot swaps
//! of killed imports are restored (`slot::recover_all`).
//!
//! The import cache: importing a file whose sha256 matches an imported slot (same importer version,
//! same template layout, files intact) returns that slot (`cached: true`) without running the
//! worker, renamed when the request names it. A request that adds something the slot lacks (an
//! explicit `index`, the `.pth`'s sibling index for a slot without one, or `smoke_fp16` for a slot
//! that never ran the fp16 check) imports the file again into that slot, keeping its name; while the
//! voice changer holds that voice the slot is returned with a warning that the addition was not
//! applied. `slot_id` forces a re-import into that slot.
//!
//! Activating a voice (`activate`, `voice_changer_set_active_slot`) selects its folder in RVC mode
//! and applies its `slot.json` exactly like `set_voice_changer_params` does, then emits
//! `voice_changer:params` with the effective params; a running voice changer keeps its current
//! voice until the UI restarts it.
//!
//! # Frozen by P3-S0 (command names, arguments and payloads)
//! Commands (also in `plugin-commands/voice-changer.txt` and `ipc.ts` `VC_COMMANDS`):
//! - `voice_changer_import { request: VcImportRequest } -> VcImportResult`
//! - `voice_changer_cancel_import { jobId } -> ()`
//! - `voice_changer_list_slots -> SlotInfo[]`
//! - `voice_changer_rename_slot { id, name } -> SlotInfo`
//! - `voice_changer_delete_slot { id } -> ()`
//! - `voice_changer_duplicate_slot { id, name? } -> SlotInfo`
//! - `voice_changer_set_active_slot { id } -> VoiceChangerParams` (the effective params, as
//!   `set_voice_changer_params` returns them; also emits `voice_changer:params`)
//! - `voice_changer_merge_slots { request: VcMergeRequest } -> SlotInfo`
//! - `voice_changer_import_default_voice -> VcImportResult`
//!
//! Events: [`IMPORT_PROGRESS_EVENT`] ([`VcImportProgress`]), [`SLOTS_CHANGED_EVENT`] (no payload).
//! Added (additive): `VcImportRequest.index`, an explicit retrieval index file to adopt.

use std::collections::HashMap;
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::sync::{Arc, Mutex, OnceLock};
use std::time::Instant;

use serde::{Deserialize, Serialize};
use tauri::{command, AppHandle, Emitter, Manager, Runtime, State};

use super::commands::PARAMS_EVENT;
use super::params::{VoiceChangerMode, VoiceChangerParams};
use super::state::VoiceChangerState;
use crate::services::neural_audio::vc::assets::VOICE_DEFAULT;
use crate::services::neural_audio::vc::import::slot::{self, MergeSpec, SlotInfo};
use crate::services::neural_audio::vc::import::worker::{self, WorkerCommand, WorkerJob, WorkerLimits};
use crate::services::neural_audio::vc::import::{self, clean_name, ErrorCode, ImportError};

/// Event: progress of an import job.
pub(crate) const IMPORT_PROGRESS_EVENT: &str = "voice_changer:import_progress";
/// Event: the slot list changed (import, rename, delete, duplicate, merge).
pub(crate) const SLOTS_CHANGED_EVENT: &str = "voice_changer:slots_changed";

/// Slot id and display name of the auto-imported MIT default voice.
const DEFAULT_VOICE_NAME: &str = "Default voice";
/// File name of the voice inside its model-store entry (`vc::assets::VOICE_DEFAULT`).
const DEFAULT_VOICE_FILE: &str = "default.pth";

/// `voice_changer_import` request.
#[derive(Clone, Debug, Deserialize)]
pub struct VcImportRequest {
    /// The `.pth` or voice `.zip` to import.
    pub source: String,
    /// Display name (default: from the checkpoint metadata / the file name).
    #[serde(default)]
    pub name: Option<String>,
    /// Re-import into this existing slot instead of a new one (skips the import cache).
    #[serde(default)]
    pub slot_id: Option<String>,
    /// Make it the active voice when done.
    #[serde(default)]
    pub activate: bool,
    /// Also smoke-decode with the fp16 graphs (slow on the CPU; default off).
    #[serde(default)]
    pub smoke_fp16: bool,
    /// A retrieval `.index` file to adopt. Default: the archive's own `added_*.index`, or for a
    /// `.pth` the `added_*.index` next to it when that choice is unambiguous
    /// (`slot::sibling_index`).
    #[serde(default)]
    pub index: Option<String>,
}

/// `voice_changer_import` / `voice_changer_import_default_voice` result.
#[derive(Clone, Debug, Serialize)]
pub struct VcImportResult {
    pub job_id: String,
    pub slot: SlotInfo,
    /// The same file was imported before; its slot was reused (sha256 cache).
    pub cached: bool,
    pub warnings: Vec<String>,
    pub elapsed_ms: u64,
}

/// Payload of [`IMPORT_PROGRESS_EVENT`].
#[derive(Clone, Debug, Serialize)]
pub struct VcImportProgress {
    pub job_id: String,
    /// `queued`, `reading`, `converting`, `smoke`, `writing`, `done` or `failed`.
    pub stage: String,
    /// 0..1.
    pub fraction: f64,
    pub message: String,
}

/// `voice_changer_merge_slots` request (weighted merge of two same-template voices).
#[derive(Clone, Debug, Deserialize)]
pub struct VcMergeRequest {
    pub a: String,
    pub b: String,
    /// Weight of `a` (0..1).
    pub weight_a: f32,
    pub name: String,
}

// ---- shared state ---------------------------------------------------------------------------------

/// One slot-changing operation at a time (see the module docs).
pub(super) fn slot_lock() -> &'static tokio::sync::Mutex<()> {
    static LOCK: OnceLock<tokio::sync::Mutex<()>> = OnceLock::new();
    LOCK.get_or_init(|| tokio::sync::Mutex::new(()))
}

/// Cancellation flags of queued and running imports.
fn jobs() -> &'static Mutex<HashMap<String, Arc<AtomicBool>>> {
    static JOBS: OnceLock<Mutex<HashMap<String, Arc<AtomicBool>>>> = OnceLock::new();
    JOBS.get_or_init(|| Mutex::new(HashMap::new()))
}

/// A registered job; unregistered when dropped.
pub(super) struct JobTicket {
    pub(super) id: String,
    pub(super) cancel: Arc<AtomicBool>,
}

impl JobTicket {
    pub(super) fn new() -> JobTicket {
        static COUNTER: AtomicU64 = AtomicU64::new(0);
        let id = format!("vcimp-{:x}-{:x}", import::unix_ms(), COUNTER.fetch_add(1, Ordering::Relaxed));
        let cancel = Arc::new(AtomicBool::new(false));
        jobs().lock().unwrap_or_else(|e| e.into_inner()).insert(id.clone(), cancel.clone());
        JobTicket { id, cancel }
    }

    pub(super) fn check(&self) -> Result<(), String> {
        if self.cancel.load(Ordering::Acquire) {
            Err(ImportError::new(ErrorCode::Job, "the import was cancelled").to_string())
        } else {
            Ok(())
        }
    }
}

impl Drop for JobTicket {
    fn drop(&mut self) {
        jobs().lock().unwrap_or_else(|e| e.into_inner()).remove(&self.id);
    }
}

pub(super) fn job_err(msg: impl Into<String>) -> String {
    ImportError::new(ErrorCode::Job, msg).to_string()
}

pub(super) fn io_err(msg: impl Into<String>) -> String {
    ImportError::new(ErrorCode::Io, msg).to_string()
}

/// `app_data/voice-changer/models` (created).
pub(super) fn models_dir<R: Runtime>(app: &AppHandle<R>) -> Result<PathBuf, String> {
    let app_data = app.path().app_data_dir().map_err(|e| io_err(format!("no app data folder: {e}")))?;
    let dir = crate::services::neural_audio::vc::bases::models_dir(&app_data);
    std::fs::create_dir_all(&dir).map_err(|e| io_err(format!("{}: {e}", dir.display())))?;
    Ok(dir)
}

/// The bundled templates: Tauri's resource folder (registered for the engine factory and the slot
/// store), else the importer's own probe.
pub(super) fn templates_dir<R: Runtime>(app: &AppHandle<R>) -> Option<PathBuf> {
    if let Ok(res) = app.path().resource_dir() {
        let dir = res.join("resources").join("vc-templates");
        if dir.is_dir() {
            import::set_templates_dir(dir.clone());
            return Some(dir);
        }
    }
    import::templates_dir()
}

pub(super) fn emit_progress<R: Runtime>(app: &AppHandle<R>, job_id: &str, stage: &str, fraction: f64, message: impl Into<String>) {
    let _ = app.emit(
        IMPORT_PROGRESS_EVENT,
        VcImportProgress { job_id: job_id.to_string(), stage: stage.to_string(), fraction, message: message.into() },
    );
}

pub(super) fn emit_slots_changed<R: Runtime>(app: &AppHandle<R>) {
    let _ = app.emit(SLOTS_CHANGED_EVENT, ());
}

/// Run blocking slot-store work off the async runtime.
pub(super) async fn blocking<T: Send + 'static>(f: impl FnOnce() -> Result<T, String> + Send + 'static) -> Result<T, String> {
    tokio::task::spawn_blocking(f).await.map_err(|e| io_err(format!("the slot store task failed: {e}")))?
}

fn same_dir(a: &str, b: &Path) -> bool {
    let a = a.trim();
    !a.is_empty() && (Path::new(a) == b || std::fs::canonicalize(a).ok().zip(std::fs::canonicalize(b).ok()).is_some_and(|(x, y)| x == y))
}

/// The slot folders the running voice changer holds loaded: the voice that plays, one it is loading
/// or about to switch to, and the idle hot voices (`VoiceSet`; `hot_slots` above 1 keeps voices
/// loaded after a switch, and an idle one comes back with the weights it was loaded with). The
/// worker publishes them (`VoiceChangerState::held_slots`, from `VoiceSet::slot_dirs` on every
/// change of the set; cleared when the run ends). `None` while it has not published them (not
/// running, or the first voice still loading); then the selected voice stands for them.
fn held_slot_dirs(state: &VoiceChangerState) -> Option<Vec<PathBuf>> {
    super::state::lock(&state.held_slots).clone()
}

/// Whether a running voice changer holds the voice in `dir`: any voice it has loaded when `held`
/// is known ([`held_slot_dirs`]), else the selected voice (`selected`).
fn slot_in_use(running: bool, selected: &str, held: Option<&[PathBuf]>, dir: &Path) -> bool {
    running
        && match held {
            Some(held) => held.iter().any(|h| same_dir(&h.to_string_lossy(), dir)),
            None => same_dir(selected, dir),
        }
}

/// Whether the voice changer holds the voice in `dir` (its files must not be replaced or deleted:
/// a held voice would keep, or come back with, the weights it was loaded with).
fn in_use(state: &VoiceChangerState, dir: &Path) -> bool {
    let held = held_slot_dirs(state);
    slot_in_use(state.is_running.load(Ordering::Acquire), &state.current_slot_dir(), held.as_deref(), dir)
}

/// Select the voice in `dir` (RVC mode), apply its settings, and tell the UI.
pub(super) fn activate<R: Runtime>(app: &AppHandle<R>, state: &VoiceChangerState, id: &str, dir: &Path) -> VoiceChangerParams {
    let mut p = state.snapshot_params();
    p.rvc_slot_dir = dir.to_string_lossy().into_owned();
    p.slot_id = Some(id.to_string());
    p.mode = VoiceChangerMode::Rvc;
    let (effective, _) = state.update_params(p);
    let _ = app.emit(PARAMS_EVENT, &effective);
    effective
}

// ---- import -----------------------------------------------------------------------------------------

/// Validate the source path of a request.
fn check_source(source: &str) -> Result<PathBuf, String> {
    let s = source.trim();
    if s.is_empty() {
        return Err(job_err("choose a .pth or .zip voice file to import"));
    }
    let path = PathBuf::from(s);
    let meta = std::fs::metadata(&path).map_err(|e| io_err(format!("{}: {e}", path.display())))?;
    if !meta.is_file() {
        return Err(job_err(format!("{} is not a file", path.display())));
    }
    Ok(path)
}

/// Whether `path` names a voice archive (else a `.pth`, whose sibling index may be adopted).
fn is_zip_name(path: &Path) -> bool {
    path.extension().and_then(|e| e.to_str()).is_some_and(|e| e.eq_ignore_ascii_case("zip"))
}

/// The display name a new slot's id is derived from.
fn id_name(request: &VcImportRequest, source: &Path) -> String {
    request
        .name
        .as_deref()
        .and_then(clean_name)
        .or_else(|| source.file_stem().and_then(|s| clean_name(&s.to_string_lossy())))
        .unwrap_or_else(|| "voice".into())
}

/// What an import whose file is already in an intact slot (the sha256 cache) does with it.
#[derive(Debug, PartialEq, Eq)]
enum CacheUse {
    /// Return the slot as it is.
    Reuse,
    /// Import the file again into that slot, for what the request adds (named for the warning when
    /// that cannot happen).
    Reimport(&'static str),
}

/// [`CacheUse`] of a request: an explicit index is always applied; a `.pth`'s own sibling index
/// (`sibling`: one was found, and the cached slot has none) and an fp16 check the slot never ran
/// are added by importing again.
fn cache_use(explicit_index: bool, sibling: bool, want_fp16: bool, fp16_checked: bool) -> CacheUse {
    if explicit_index {
        CacheUse::Reimport("the retrieval index")
    } else if sibling {
        CacheUse::Reimport("the retrieval index next to the file")
    } else if want_fp16 && !fp16_checked {
        CacheUse::Reimport("the fp16 check")
    } else {
        CacheUse::Reuse
    }
}

/// `rename_cached`: a name given with the request renames a voice served from the cache (a user
/// import; the default voice keeps whatever name the user gave it).
async fn run_import<R: Runtime>(
    app: &AppHandle<R>,
    state: &VoiceChangerState,
    request: VcImportRequest,
    ticket: &JobTicket,
    rename_cached: bool,
) -> Result<VcImportResult, String> {
    let started = Instant::now();
    let job_id = ticket.id.clone();
    let source = check_source(&request.source)?;
    emit_progress(app, &job_id, "queued", 0.0, "Waiting for another voice operation to finish");
    let _guard = slot_lock().lock().await;
    ticket.check()?;
    let models = models_dir(app)?;
    let templates = templates_dir(app)
        .ok_or_else(|| ImportError::new(ErrorCode::NoTemplate, "the bundled voice templates were not found").to_string())?;
    {
        let models = models.clone();
        blocking(move || {
            slot::recover_all(&models);
            Ok(())
        })
        .await?;
    }

    emit_progress(app, &job_id, "reading", 0.05, "Checking the file");
    let sha = {
        let source = source.clone();
        blocking(move || import::sha256_file(&source).map(|(sha, _)| sha).map_err(|e| e.to_string())).await?
    };
    ticket.check()?;
    let explicit_index = match request.index.as_deref().map(str::trim).filter(|s| !s.is_empty()) {
        Some(ix) => {
            let p = PathBuf::from(ix);
            if !p.is_file() {
                return Err(job_err(format!("the index {} was not found", p.display())));
            }
            Some(p)
        }
        None => None,
    };

    // The import cache: the same file in an intact slot is reused, unless the request adds
    // something that slot lacks; then the file is imported again into that slot.
    let mut target = request.slot_id.as_deref().map(str::trim).filter(|s| !s.is_empty()).map(str::to_string);
    let mut keep_name = None;
    if target.is_none() {
        let (m, s, src, explicit) = (models.clone(), sha.clone(), source.clone(), explicit_index.is_some());
        let cached = blocking(move || {
            Ok(slot::find_cached(&m, &s).map(|info| {
                let dir = PathBuf::from(&info.path);
                let sibling = !explicit && !info.has_index && !is_zip_name(&src) && slot::sibling_index(&src).is_some();
                let fp16_checked = slot::fp16_checked(&dir);
                (info, sibling, fp16_checked)
            }))
        })
        .await?;
        if let Some((mut info, sibling, fp16_checked)) = cached {
            let dir = PathBuf::from(&info.path);
            let mut warnings = Vec::new();
            match cache_use(explicit_index.is_some(), sibling, request.smoke_fp16, fp16_checked) {
                CacheUse::Reimport(_) if !in_use(state, &dir) => {
                    target = Some(info.id.clone());
                    keep_name = Some(info.name.clone());
                }
                use_ => {
                    if let CacheUse::Reimport(what) = use_ {
                        warnings.push(format!(
                            "{what} was not applied: the running voice changer has this voice loaded; stop it and import the file again"
                        ));
                    }
                    // A name given with the request renames the voice it already is.
                    let wanted = request.name.as_deref().and_then(clean_name).filter(|n| rename_cached && *n != info.name);
                    if let Some(name) = wanted {
                        let (m, id) = (models.clone(), info.id.clone());
                        info = blocking(move || slot::rename_slot(&m, &id, &name)).await?;
                        emit_slots_changed(app);
                    }
                    if request.activate {
                        activate(app, state, &info.id, &dir);
                    }
                    return Ok(VcImportResult { job_id, slot: info, cached: true, warnings, elapsed_ms: started.elapsed().as_millis() as u64 });
                }
            }
        }
    }

    // The slot the voice goes into.
    let id = match target.as_deref() {
        Some(id) => {
            let dir = slot::slot_dir(&models, id)?;
            if dir.is_dir() {
                // Only an imported voice (or an empty folder) is replaced; a bring-your-own ONNX voice
                // or another engine's slot is never overwritten.
                let info = slot::slot_info_with(&models, id, Some(&templates))?;
                let has_voice = dir.join(import::MANIFEST_FILE).is_file()
                    || crate::services::neural_audio::vc::bases::find_generator_onnx(&dir).is_some();
                if has_voice && info.format != crate::services::neural_audio::vc::manifest::SIGIL_RVC_FORMAT {
                    return Err(job_err(format!("{id} is not an imported voice ({}); import into a new slot instead", info.format)));
                }
                if in_use(state, &dir) {
                    return Err(job_err("stop the voice changer before re-importing a voice it has loaded"));
                }
            }
            id.to_string()
        }
        None => slot::new_slot_id(&models, &id_name(&request, &source)),
    };
    let out_dir = slot::slot_dir(&models, &id)?;
    let index = match explicit_index {
        Some(p) => Some(p),
        None if !is_zip_name(&source) => slot::sibling_index(&source),
        None => None,
    };
    let job = WorkerJob {
        smoke_fp16: request.smoke_fp16,
        // A cache re-import keeps the voice's current name unless the request names it.
        name: request.name.as_deref().and_then(clean_name).or(keep_name),
        index,
        ..WorkerJob::new(source.clone(), out_dir.clone(), Some(templates.clone()))
    };
    emit_progress(app, &job_id, "converting", 0.15, "Converting the voice");
    let exe = std::env::current_exe().map_err(|e| io_err(format!("cannot find Sigil's executable: {e}")))?;
    let cancel = ticket.cancel.clone();
    let record = blocking(move || {
        worker::run_isolated_with(&WorkerCommand::app(&exe), &job, WorkerLimits::default(), &cancel, None).map_err(|e| e.to_string())
    })
    .await?;

    emit_progress(app, &job_id, "writing", 0.95, "Adding the voice");
    let info = {
        let (m, t, id) = (models.clone(), templates.clone(), id.clone());
        blocking(move || slot::slot_info_with(&m, &id, Some(&t))).await?
    };
    let warnings: Vec<String> = record["warnings"]
        .as_array()
        .map(|a| a.iter().filter_map(|w| w.as_str().map(str::to_string)).collect())
        .unwrap_or_default();
    if request.activate {
        activate(app, state, &id, &out_dir);
    }
    emit_slots_changed(app);
    Ok(VcImportResult { job_id, slot: info, cached: false, warnings, elapsed_ms: started.elapsed().as_millis() as u64 })
}

/// Run an import with its progress events (`done` / `failed` at the end).
async fn import_with_events<R: Runtime>(
    app: &AppHandle<R>,
    state: &VoiceChangerState,
    request: VcImportRequest,
    ticket: JobTicket,
    rename_cached: bool,
) -> Result<VcImportResult, String> {
    let result = run_import(app, state, request, &ticket, rename_cached).await;
    match &result {
        Ok(r) => emit_progress(app, &ticket.id, "done", 1.0, if r.cached { "Already imported" } else { "Imported" }),
        Err(e) => emit_progress(app, &ticket.id, "failed", 1.0, e.clone()),
    }
    if let Err(e) = &result {
        tracing::warn!("voice import failed: {e}");
    }
    result
}

#[command]
pub(super) async fn voice_changer_import<R: Runtime>(
    app: AppHandle<R>,
    state: State<'_, VoiceChangerState>,
    request: VcImportRequest,
) -> Result<VcImportResult, String> {
    import_with_events(&app, &state, request, JobTicket::new(), true).await
}

/// Cancel a queued or running import (unknown or finished jobs are ignored).
#[command]
pub(super) fn voice_changer_cancel_import(job_id: String) -> Result<(), String> {
    if let Some(flag) = jobs().lock().unwrap_or_else(|e| e.into_inner()).get(job_id.trim()) {
        flag.store(true, Ordering::Release);
    }
    Ok(())
}

/// Every voice slot (imported and bring-your-own).
#[command]
pub(super) async fn voice_changer_list_slots<R: Runtime>(app: AppHandle<R>) -> Result<Vec<SlotInfo>, String> {
    let models = models_dir(&app)?;
    let _ = templates_dir(&app);
    // Tidy interrupted swaps only while no import could be writing.
    let idle = slot_lock().try_lock().ok();
    blocking(move || {
        if idle.is_some() {
            slot::recover_all(&models);
        }
        slot::list_slots(&models)
    })
    .await
}

#[command]
pub(super) async fn voice_changer_rename_slot<R: Runtime>(app: AppHandle<R>, id: String, name: String) -> Result<SlotInfo, String> {
    let models = models_dir(&app)?;
    let _ = templates_dir(&app);
    let _guard = slot_lock().lock().await;
    let info = blocking(move || slot::rename_slot(&models, id.trim(), &name)).await?;
    emit_slots_changed(&app);
    Ok(info)
}

/// Delete a voice. A voice the running voice changer holds ([`in_use`]) cannot be deleted; deleting
/// the selected voice clears the selection.
#[command]
pub(super) async fn voice_changer_delete_slot<R: Runtime>(
    app: AppHandle<R>,
    state: State<'_, VoiceChangerState>,
    id: String,
) -> Result<(), String> {
    let models = models_dir(&app)?;
    let id = id.trim().to_string();
    let dir = slot::slot_dir(&models, &id)?;
    let _guard = slot_lock().lock().await;
    if in_use(&state, &dir) {
        return Err(job_err("stop the voice changer before deleting a voice it has loaded"));
    }
    let was_selected = same_dir(&state.current_slot_dir(), &dir);
    {
        let (m, id) = (models.clone(), id.clone());
        blocking(move || slot::delete_slot(&m, &id)).await?;
    }
    if was_selected {
        let mut p = state.snapshot_params();
        p.rvc_slot_dir = String::new();
        p.slot_id = Some(String::new());
        let (effective, _) = state.update_params(p);
        let _ = app.emit(PARAMS_EVENT, &effective);
    }
    emit_slots_changed(&app);
    Ok(())
}

#[command]
pub(super) async fn voice_changer_duplicate_slot<R: Runtime>(app: AppHandle<R>, id: String, name: Option<String>) -> Result<SlotInfo, String> {
    let models = models_dir(&app)?;
    let _ = templates_dir(&app);
    let _guard = slot_lock().lock().await;
    let info = blocking(move || slot::duplicate_slot(&models, id.trim(), name.as_deref())).await?;
    emit_slots_changed(&app);
    Ok(info)
}

/// Make `id` the active voice (RVC mode); returns the effective params.
#[command]
pub(super) fn voice_changer_set_active_slot<R: Runtime>(
    app: AppHandle<R>,
    state: State<'_, VoiceChangerState>,
    id: String,
) -> Result<VoiceChangerParams, String> {
    let models = models_dir(&app)?;
    let id = id.trim();
    let dir = slot::slot_dir(&models, id)?;
    if !dir.is_dir() {
        return Err(job_err(format!("there is no voice slot {id:?}")));
    }
    Ok(activate(&app, &state, id, &dir))
}

#[command]
pub(super) async fn voice_changer_merge_slots<R: Runtime>(app: AppHandle<R>, request: VcMergeRequest) -> Result<SlotInfo, String> {
    let models = models_dir(&app)?;
    let _ = templates_dir(&app);
    let _guard = slot_lock().lock().await;
    let spec = MergeSpec { a: request.a.trim().to_string(), b: request.b.trim().to_string(), weight_a: request.weight_a, name: request.name };
    let info = blocking(move || slot::merge_slots(&models, &spec)).await?;
    emit_slots_changed(&app);
    Ok(info)
}

/// Install the MIT default voice through the model store (one download, `vc/voice-default`) and
/// import it. It becomes the active voice when none is selected. Calling it again returns the
/// imported slot (the import cache).
#[command]
pub(super) async fn voice_changer_import_default_voice<R: Runtime>(
    app: AppHandle<R>,
    state: State<'_, VoiceChangerState>,
) -> Result<VcImportResult, String> {
    let ticket = JobTicket::new();
    emit_progress(&app, &ticket.id, "reading", 0.0, "Installing the default voice");
    let installed = crate::services::model_store::ensure(&[VOICE_DEFAULT]).await;
    let source = match installed.map(|dirs| dirs.into_iter().next()) {
        Ok(Some(dir)) => default_voice_file(&dir),
        Ok(None) => Err(job_err("the model store returned no folder for the default voice")),
        Err(e) => Err(job_err(format!("the default voice could not be installed: {e}"))),
    };
    let source = match source {
        Ok(s) => s,
        Err(e) => {
            emit_progress(&app, &ticket.id, "failed", 1.0, e.clone());
            return Err(e);
        }
    };
    let selected = state.current_slot_dir();
    let activate = selected.is_empty() || !Path::new(&selected).is_dir();
    let request = VcImportRequest {
        source: source.to_string_lossy().into_owned(),
        name: Some(DEFAULT_VOICE_NAME.into()),
        slot_id: None,
        activate,
        smoke_fp16: false,
        index: None,
    };
    import_with_events(&app, &state, request, ticket, false).await
}

/// The voice file of the default-voice entry: `default.pth`, else its only `.pth`.
fn default_voice_file(dir: &Path) -> Result<PathBuf, String> {
    let direct = dir.join(DEFAULT_VOICE_FILE);
    if direct.is_file() {
        return Ok(direct);
    }
    let pths: Vec<PathBuf> = std::fs::read_dir(dir)
        .map_err(|e| io_err(format!("{}: {e}", dir.display())))?
        .flatten()
        .map(|e| e.path())
        .filter(|p| p.is_file() && p.extension().and_then(|e| e.to_str()).is_some_and(|e| e.eq_ignore_ascii_case("pth")))
        .collect();
    match pths.as_slice() {
        [one] => Ok(one.clone()),
        _ => Err(job_err(format!("the default voice entry in {} holds no single .pth voice", dir.display()))),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn requests_parse_with_defaults_and_the_additive_index() {
        let r: VcImportRequest = serde_json::from_value(serde_json::json!({ "source": "C:/v.zip" })).unwrap();
        assert!(r.name.is_none() && r.slot_id.is_none() && !r.activate && !r.smoke_fp16 && r.index.is_none());
        let r: VcImportRequest = serde_json::from_value(serde_json::json!({
            "source": "C:/v.pth", "name": "V", "slot_id": null, "activate": true, "smoke_fp16": true, "index": "C:/added.index"
        }))
        .unwrap();
        assert_eq!((r.name.as_deref(), r.activate, r.index.as_deref()), (Some("V"), true, Some("C:/added.index")));
        let m: VcMergeRequest = serde_json::from_value(serde_json::json!({ "a": "x", "b": "y", "weight_a": 0.25, "name": "Z" })).unwrap();
        assert_eq!((m.a.as_str(), m.weight_a), ("x", 0.25));
    }

    #[test]
    fn payloads_are_snake_case_with_the_contract_fields() {
        let p = serde_json::to_value(VcImportProgress { job_id: "j".into(), stage: "done".into(), fraction: 1.0, message: String::new() }).unwrap();
        for k in ["job_id", "stage", "fraction", "message"] {
            assert!(p.get(k).is_some(), "{k}");
        }
        let r = serde_json::to_value(VcImportResult { job_id: "j".into(), slot: SlotInfo::default(), cached: true, warnings: vec![], elapsed_ms: 3 }).unwrap();
        for k in ["job_id", "slot", "cached", "warnings", "elapsed_ms"] {
            assert!(r.get(k).is_some(), "{k}");
        }
        let s = serde_json::to_value(SlotInfo::default()).unwrap();
        for k in [
            "id", "name", "path", "format", "template", "version", "sample_rate", "f0", "embedder", "vocoder", "has_index", "speaker_count", "fp16_ok",
            "source_sha256", "created_unix_ms", "problem",
        ] {
            assert!(s.get(k).is_some(), "SlotInfo.{k}");
        }
    }

    #[test]
    fn held_voices_are_in_use_while_running() {
        let d = std::env::temp_dir().join(format!("sigil-vc-held-{}-{}", std::process::id(), import::unix_ms()));
        let (a, b, c) = (d.join("a"), d.join("b"), d.join("c"));
        for dir in [&a, &b, &c] {
            std::fs::create_dir_all(dir).unwrap();
        }
        let sel_b = b.to_string_lossy().into_owned();
        // Stopped: nothing is in use.
        assert!(!slot_in_use(false, &sel_b, None, &b));
        assert!(!slot_in_use(false, &sel_b, Some(&[a.clone(), b.clone()]), &a));
        // Running, the held set unknown: the selected voice.
        assert!(slot_in_use(true, &sel_b, None, &b) && !slot_in_use(true, &sel_b, None, &a));
        // Running with the held set: A still plays (or is kept hot) while B is selected; C, the
        // selected voice whose load failed, is not held.
        let held = [a.clone(), b.clone()];
        assert!(slot_in_use(true, &sel_b, Some(&held), &a), "an idle or still-playing voice");
        assert!(slot_in_use(true, &sel_b, Some(&held), &b));
        let sel_c = c.to_string_lossy().into_owned();
        assert!(!slot_in_use(true, &sel_c, Some(&held), &c), "selected but not loaded");
        // The same folder spelled another way.
        let spelled = d.join(".").join("a");
        assert!(slot_in_use(true, "", Some(&[spelled]), &a));
        let _ = std::fs::remove_dir_all(&d);
    }

    #[test]
    fn cached_imports_reimport_for_what_the_request_adds() {
        assert_eq!(cache_use(false, false, false, false), CacheUse::Reuse);
        assert_eq!(cache_use(false, false, false, true), CacheUse::Reuse);
        // An explicit index is always applied (it may replace the slot's index).
        assert_eq!(cache_use(true, false, false, true), CacheUse::Reimport("the retrieval index"));
        // A sibling index only counts when the cached slot has none (the caller checks).
        assert!(matches!(cache_use(false, true, false, true), CacheUse::Reimport(w) if w.contains("next to the file")));
        // The fp16 check only when it never ran for that slot.
        assert_eq!(cache_use(false, false, true, true), CacheUse::Reuse);
        assert_eq!(cache_use(false, false, true, false), CacheUse::Reimport("the fp16 check"));
    }

    #[test]
    fn job_tickets_register_cancel_and_unregister() {
        let t = JobTicket::new();
        assert!(t.check().is_ok());
        voice_changer_cancel_import(t.id.clone()).unwrap();
        let err = t.check().unwrap_err();
        assert!(err.starts_with("job:") && err.contains("cancelled"), "{err}");
        let id = t.id.clone();
        drop(t);
        assert!(!jobs().lock().unwrap().contains_key(&id));
        voice_changer_cancel_import("no-such-job".into()).unwrap();
    }

    #[test]
    fn sources_names_and_default_voice_files() {
        let d = std::env::temp_dir().join(format!("sigil-vc-cmdimport-{}-{}", std::process::id(), import::unix_ms()));
        std::fs::create_dir_all(&d).unwrap();
        assert!(check_source("  ").unwrap_err().starts_with("job:"));
        assert!(check_source(&d.to_string_lossy()).unwrap_err().starts_with("job:"), "a folder");
        assert!(check_source(&d.join("missing.pth").to_string_lossy()).unwrap_err().starts_with("io:"));
        std::fs::write(d.join("My Voice.pth"), b"x").unwrap();
        let src = check_source(&format!(" {} ", d.join("My Voice.pth").display())).unwrap();
        let req = |name: Option<&str>| VcImportRequest { source: String::new(), name: name.map(str::to_string), slot_id: None, activate: false, smoke_fp16: false, index: None };
        assert_eq!(id_name(&req(None), &src), "My Voice");
        assert_eq!(id_name(&req(Some(" Kira ")), &src), "Kira");
        assert!(is_zip_name(Path::new("a/B.ZIP")) && !is_zip_name(&src));

        assert!(default_voice_file(&d).unwrap().ends_with("My Voice.pth"), "the only .pth");
        std::fs::write(d.join(DEFAULT_VOICE_FILE), b"x").unwrap();
        assert!(default_voice_file(&d).unwrap().ends_with(DEFAULT_VOICE_FILE));
        let _ = std::fs::remove_dir_all(&d);
    }
}
