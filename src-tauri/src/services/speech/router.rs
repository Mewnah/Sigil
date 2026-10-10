//! Language router: request + PC state → engines (P2-C; research §1.1 "zero-setup automatic model
//! pick by language" and "two-pass captioning").
//!
//! The router maps a [`RouteRequest`] (language or `"auto"`, tier, two-pass) and a [`RouteContext`]
//! (GPU, VRAM the resource governor grants, voice-changer state, cores, installed / installable
//! catalog ids) to a [`RoutePlan`]: a streaming model for partials, a final (utterance) model, a
//! CPU fallback for a GPU final, a spoken-language-ID engine for `"auto"` when the final cannot
//! detect the language itself, every catalog id the plan needs and which of them are missing.
//!
//! # Table
//! The choices are data: the `routing` object of `model-catalog/stt-routing.json` (read with
//! `include_str!`; its `$comment` documents the rows and where the numbers come from). Rows are
//! matched by normalized language (`lang::normalize`), then `"auto"`, then `"*"`. Per row:
//! - streaming = the first usable `streaming` model (CPU; partials and first-pass finals);
//! - CPU final = the first usable `cpuFinal` model;
//! - GPU final = the first usable `gpuFinal` model that fits: `vramMb` ≤ the VRAM the governor
//!   grants the STT final (`resources::policy::room`), and the card has `minTotalVramMb`;
//! - "usable" = in the table, not [`RouteContext::unavailable`] (not in the catalog in effect, or an
//!   unpublished Sigil-built pack that is neither installed nor in the
//!   [`model_store::LOCAL_ASSETS_ENV`] folder the model store installs it from), its runtimes
//!   usable, the PC has its `minCores`, and the language rules below allow it. A custom mirror is
//!   not consulted: an unpublished pack served only by one counts as unavailable until imported.
//!
//! # Tier
//! - `cpu`: CPU models only.
//! - `gpu` / `auto`: a GPU final when [`RouteContext::gpu`] (a usable GPU with a transcribe.cpp
//!   build for it), its runtime pack is usable, and a `gpuFinal` of the row fits. `auto` also keeps
//!   the GPU for the voice changer: while it runs on the GPU of a card below `vcShareGpuMinTotalMb`
//!   (16 GB class) the finals stay on the CPU (`gpu_denied_vc_running`); `gpu` then runs anyway
//!   (`vc_shares_gpu`). Anything else falls back to the CPU route with a note saying why.
//!
//! # Passes
//! Two passes (streaming partials, then the final re-decodes the closed utterance under the same
//! `segmentId`, `revision + 1`) when `twoPass` is asked for, the row has both, and the PC allows it:
//! always with a GPU final; on the CPU only when the CPU may run two engines: `cpuTwoPassMinCores`
//! physical cores (0 = unknown counts as enough) and no voice changer that runs, or may run, on the
//! CPU (a running voice changer whose device is not known yet counts as on the CPU here, just as it
//! counts as on the GPU for the GPU rule below). Otherwise one pass: the streaming model alone when
//! the row says its own finals are caption quality (`streamFinal`) and the route is on the CPU,
//! else the final alone (finals only, no partials).
//!
//! `interim: false` ([`RouteRequest::interim`], "no live partial captions") turns the two-pass
//! plan into the final alone, so the streaming model is not planned, loaded or run (its CPU is
//! saved); the one-pass choices above are unchanged, because a stream that is the plan's only
//! engine produces the finals too, and swapping it for another model would change downloads and
//! quality. Everything else (tier, language ID, CPU fallback of a GPU final) is planned for the
//! final-only plan exactly as for any other finals-only plan.
//!
//! # CPU fallback of a GPU final
//! A GPU final gets the row's CPU final as `fallback` (promoted for the rest of the session when
//! the GPU engine fails). Next to a streaming model that fallback would be a second CPU engine, so
//! it is planned only where the CPU may run two engines (the CPU two-pass rule above); elsewhere the
//! first-pass finals stand after a GPU failure (`no_cpu_fallback`). A finals-only GPU plan always
//! keeps it (the fallback then replaces the GPU engine, one CPU engine).
//!
//! # Rules enforced in code (whatever the table says)
//! - Cohere Transcribe is never used for ja / zh / ko / ru / yue (it silently drops Japanese and
//!   has no Russian), nor in `"auto"` mode (no language ID); nor is any model with
//!   `requiresLanguage`, or for a language outside its `languages` / inside its `neverLanguages`.
//! - A model that needs a language (Nemotron 3.5 streaming) always gets one: with an explicit
//!   language its spec carries `options.language`; in `"auto"` mode it runs only next to a final
//!   that detects the language (or a language-ID engine), which the pipeline waits for before
//!   opening the stream. Otherwise it is dropped (`nemotron_needs_language`).
//! - `"auto"` never streams where an explicit choice of the detected language would not: the auto
//!   streaming spec carries `options.streamLanguages` ([`STREAM_LANGUAGES_OPTION`]), the detected
//!   languages the stream may be opened with: the model's languages whose own row streams with it
//!   (so not zh / yue, whose rows have no partials, nor languages the model lacks). The pipeline
//!   opens the stream only for such a language ([`stream_languages`]) and sends every other
//!   utterance to the final alone. No such language at all drops the stream.
//! - `"auto"` uses spoken language ID: the final detects the language itself (Qwen3-ASR,
//!   SenseVoice, Whisper), or the table's `languageId` engine runs first on each utterance.
//! - GPU specs only on the GPU tier; fallbacks are always CPU models.
//!
//! The shipped table's own rules (zh / yue → SenseVoice, ja / ko / ru → Qwen3-ASR, European
//! languages → a Parakeet Ultra final on the CPU) are checked by the unit tests below, together
//! with the shared fixture.
//!
//! # Notes
//! [`RoutePlan::notes`] carries machine codes ([`notes`]) that say why the plan looks the way it
//! does (`gpu_denied_vc_running`, `few_cores_single_pass`, ...); the UI maps known codes to text
//! (`stt.route_note_<code>`, governor reasons `models.gpu.rec_reason_<code>`) and shows unknown
//! ones as-is.
//!
//! # Frozen signatures (P2-S0)
//! - [`plan_with`]`(&RouteRequest, &RouteContext) -> RoutePlan`: pure and deterministic (tests, the
//!   shared fixture).
//! - [`plan`]`(&RouteRequest) -> RoutePlan`: [`plan_with`] on the live [`context`].
//! - [`context`]`() -> RouteContext`: `sysinfo::cached()`, `resources::snapshot()` and the
//!   governor's room for the STT final, `model_store::installed_path`, the catalog in effect.
//! - IPC `plugin:speech|preview_route { request }` → [`preview_route`] (installs nothing).
//! - [`stream_languages`]`(&EngineSpec, engine languages) -> Vec<String>`: the languages the
//!   pipeline may open a plan's stream with (added by the P2-C review fix).
//! - Shared fixture `src-tauri/test-fixtures/routing.json` (Rust test here, TS mirror test in
//!   `src/core/services/stt/__tests__/router.test.ts`, P2-E): `{ version: 1, cases: [{ name,
//!   request: RouteRequest, context: RouteContext, expect }] }` where every `expect` key is
//!   optional: `language`, `tier`, `streamingModel` / `finalModel` / `fallbackModel` /
//!   `languageIdModel` (catalog id or `null` for "none"), `streamingLanguage` (the streaming
//!   spec's `options.language`, `null` = none), `neverModels` (ids that must not appear anywhere in
//!   the plan), `notesInclude` (codes that must be in `notes`) and `missingInclude` (ids that must
//!   be in `missing`). [`RouteContext`] may gain fields (P2-C owns it and the fixture).

use std::collections::{BTreeMap, BTreeSet};
use std::path::{Path, PathBuf};
use std::sync::OnceLock;

use serde::{Deserialize, Serialize};
use tauri::command;

use super::engine_api::{EngineKind, EngineSpec, RoutePlan, RouteRequest, Tier};
use super::lang;
use crate::services::model_store::{self, Catalog, CatalogEntry, PublishStatus};
use crate::services::resources::policy::{self, GpuClass};
use crate::services::resources::{self, RecommendationTopic, ResourceSnapshot};
use crate::services::speech_rt::sherpa::catalog as sherpa_catalog;
use crate::services::sysinfo::{self, eps, SystemInfo};

/// Log target of the router.
const LOG: &str = "sigil::speech::router";

/// The embedded routing table document.
const TABLE_JSON: &str = include_str!("../../../model-catalog/stt-routing.json");

/// Never-route rules that hold whatever the table says: model id prefix → languages it must never
/// transcribe (research §1.1: Cohere silently drops Japanese sentences and has no Russian).
const NEVER: &[(&str, &[&str])] = &[("stt/cohere-", &["ja", "zh", "ko", "ru", "yue"])];
/// Model id prefixes that never run in `"auto"` mode (no language ID of their own).
const NEVER_AUTO: &[&str] = &["stt/cohere-"];
/// Model id prefixes that must always be given a language (their auto mode is worse on every
/// language: Nemotron 3.5 streaming).
const NEEDS_LANGUAGE: &[&str] = &["stt/nemotron-"];

/// Option of an `"auto"` plan's streaming spec: the detected languages (normalized codes, sorted)
/// the pipeline may open the stream with; every other utterance goes to the final alone. Engines
/// ignore it (their option parsers skip unknown keys).
pub const STREAM_LANGUAGES_OPTION: &str = "streamLanguages";

/// Machine codes of [`RoutePlan::notes`] (new codes may be added; the UI shows unknown ones as-is).
///
/// UI text: `stt.route_note_<code>` (public/i18n/*/translation.json). A plan for a PC
/// without a GPU for the final also carries the governor's reason code (`no_gpu`, `gpu_unknown`,
/// `integrated_gpu`, `small_gpu`, `no_gpu_runtime`), whose text is `models.gpu.rec_reason_<code>`.
pub mod notes {
    /// `"auto"`: the language is detected per utterance.
    pub const AUTO_LANGUAGE: &str = "auto_language";
    /// `"auto"`: a separate spoken-language-ID engine runs before the final.
    pub const LANGUAGE_ID: &str = "language_id";
    /// No model is dedicated to the language: a general multilingual model transcribes it.
    pub const GENERIC_LANGUAGE_MODEL: &str = "generic_language_model";
    /// No streaming model for the language (or the pass structure): finals only, no partials.
    pub const FINALS_ONLY: &str = "finals_only";
    /// One pass because two-pass was not asked for.
    pub const SINGLE_PASS: &str = "single_pass";
    /// One pass because the PC has few cores for two CPU engines.
    pub const FEW_CORES_SINGLE_PASS: &str = "few_cores_single_pass";
    /// One pass because the voice changer runs on the CPU.
    pub const VC_CPU_SINGLE_PASS: &str = "vc_running_single_pass";
    /// One pass because the running voice changer has not reported its device yet (it may use
    /// the CPU); a restart after it started plans again.
    pub const VC_UNKNOWN_SINGLE_PASS: &str = "vc_unknown_single_pass";
    /// A GPU final next to a CPU stream has no CPU fallback here (few cores, or a voice changer
    /// that may use the CPU): after a GPU failure the first-pass finals stand.
    pub const NO_CPU_FALLBACK: &str = "no_cpu_fallback";
    /// A GPU tier was asked for but the PC has no usable GPU for the STT final (plus the governor's
    /// reason code: `no_gpu`, `gpu_unknown`, `integrated_gpu`, `small_gpu`, `no_gpu_runtime`).
    pub const GPU_UNAVAILABLE: &str = "gpu_unavailable";
    /// The transcribe.cpp GPU runtime pack cannot be installed here (not in the catalog, or a
    /// Sigil-built pack that was not imported).
    pub const GPU_RUNTIME_UNAVAILABLE: &str = "gpu_runtime_unavailable";
    /// `auto` tier: the voice changer runs on this (smaller) GPU; finals stay on the CPU.
    pub const GPU_DENIED_VC_RUNNING: &str = "gpu_denied_vc_running";
    /// `gpu` tier while the voice changer runs on the GPU: both share it.
    pub const VC_SHARES_GPU: &str = "vc_shares_gpu";
    /// No GPU final of the language fits the VRAM the governor grants now.
    pub const GPU_FULL: &str = "gpu_full";
    /// No GPU model beats the CPU final for this language (or none is available).
    pub const NO_GPU_MODEL_FOR_LANGUAGE: &str = "no_gpu_model_for_language";
    /// A model that needs a language was dropped because nothing in the plan detects it.
    pub const NEMOTRON_NEEDS_LANGUAGE: &str = "nemotron_needs_language";
    /// A table entry broke a never-route rule and was skipped (`never_route` plus the model id in
    /// the log).
    pub const NEVER_ROUTE: &str = "never_route";
    /// No engine is available for the request (nothing installable).
    pub const NO_ENGINE: &str = "no_engine";
    /// The embedded routing table failed to load (a build bug; see the log).
    pub const ROUTING_TABLE_INVALID: &str = "routing_table_invalid";

    /// Every code above (the UI-text test checks each has a string).
    #[cfg(test)]
    pub const ALL: &[&str] = &[
        AUTO_LANGUAGE,
        LANGUAGE_ID,
        GENERIC_LANGUAGE_MODEL,
        FINALS_ONLY,
        SINGLE_PASS,
        FEW_CORES_SINGLE_PASS,
        VC_CPU_SINGLE_PASS,
        VC_UNKNOWN_SINGLE_PASS,
        NO_CPU_FALLBACK,
        GPU_UNAVAILABLE,
        GPU_RUNTIME_UNAVAILABLE,
        GPU_DENIED_VC_RUNNING,
        VC_SHARES_GPU,
        GPU_FULL,
        NO_GPU_MODEL_FOR_LANGUAGE,
        NEMOTRON_NEEDS_LANGUAGE,
        NEVER_ROUTE,
        NO_ENGINE,
        ROUTING_TABLE_INVALID,
    ];
    /// The resource governor's GPU reason codes a plan may carry (their UI text is
    /// `models.gpu.rec_reason_<code>`).
    #[cfg(test)]
    pub const GOVERNOR_REASONS: &[&str] = &["no_gpu", "gpu_unknown", "integrated_gpu", "small_gpu", "no_gpu_runtime"];
}

// ---------------------------------------------------------------------------------------------
// Context
// ---------------------------------------------------------------------------------------------

/// What the router knows about this PC. Wire: camelCase (the shared fixture serializes it).
#[derive(Clone, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase", default)]
pub struct RouteContext {
    /// A GPU-tier engine may run here (a supported GPU and an allowed transcribe GPU variant).
    pub gpu: bool,
    /// Best allowed GPU runtime variant (`cuda13`, `vulkan`), if any.
    pub gpu_variant: Option<String>,
    /// VRAM the resource governor would grant to `stt_final` now (MB).
    pub vram_available_mb: u64,
    /// The voice changer is running (it has VRAM / CPU priority).
    pub vc_running: bool,
    pub physical_cores: u32,
    /// Installed catalog ids (models and runtime packs).
    pub installed: BTreeSet<String>,
    /// Dedicated VRAM of the primary GPU (MB; 0 = unknown). BF16 finals need a 16 GB-class card
    /// (`minTotalVramMb`), and the voice changer keeps smaller cards to itself in `auto`.
    pub vram_total_mb: u64,
    /// Whether the running voice changer uses a GPU EP (`None`: not running, or not known yet).
    pub vc_gpu: Option<bool>,
    /// Catalog ids that cannot be installed here: missing from the catalog in effect, or
    /// unpublished (Sigil-built) and neither installed nor in the local-assets folder. The router
    /// never plans them.
    pub unavailable: BTreeSet<String>,
    /// Why `gpu` is false: the governor's reason code (`no_gpu`, `gpu_unknown`, `integrated_gpu`,
    /// `small_gpu`, `no_gpu_runtime`).
    pub gpu_reason: Option<String>,
}

/// What the router needs of the catalog in effect.
struct CatalogFacts {
    /// Every id → publish status.
    status: BTreeMap<String, PublishStatus>,
    /// The unpublished entries (their files are looked up in the local-assets folder).
    unpublished: BTreeMap<String, CatalogEntry>,
}

/// The catalog in effect, cached (`None`, logged once, when it cannot be loaded: then only
/// installed models are routed).
fn catalog_facts() -> Option<&'static CatalogFacts> {
    static CATALOG: OnceLock<Option<CatalogFacts>> = OnceLock::new();
    CATALOG
        .get_or_init(|| match Catalog::load() {
            Ok(c) => Some(CatalogFacts {
                status: c.entries.iter().map(|e| (e.id.clone(), e.status)).collect(),
                unpublished: c
                    .entries
                    .into_iter()
                    .filter(|e| e.status == PublishStatus::Unpublished)
                    .map(|e| (e.id.clone(), e))
                    .collect(),
            }),
            Err(e) => {
                tracing::warn!(target: LOG, "model catalog unavailable ({e}); routing only to installed models");
                None
            }
        })
        .as_ref()
}

/// The [`model_store::LOCAL_ASSETS_ENV`] folder, read the way the model store reads it.
fn local_assets_root() -> Option<PathBuf> {
    std::env::var_os(model_store::LOCAL_ASSETS_ENV)
        .filter(|v| !v.is_empty())
        .map(PathBuf::from)
}

/// `rel` (a `/`-separated catalog path) under `base`; `None` for anything that could leave `base`
/// (the model store's catalog path rule: `[A-Za-z0-9._+-]` segments, no `.` / `..`).
fn join_catalog_path(base: &Path, rel: &str) -> Option<PathBuf> {
    let mut path = base.to_path_buf();
    for part in rel.split('/') {
        let ok = !part.is_empty()
            && part != "."
            && part != ".."
            && part
                .chars()
                .all(|c| c.is_ascii_alphanumeric() || matches!(c, '.' | '_' | '+' | '-'));
        if !ok {
            return None;
        }
        path.push(part);
    }
    Some(path)
}

/// Every file of `entry` is in the local-assets folder `root` where the model store's install
/// looks for it (`model_store::install::find_local_asset`): `<root>/<id>/<path>`, `<root>/<path>`,
/// or `<root>/<file name of the first source URL>`. The install still verifies each file's hash.
fn has_local_assets(root: &Path, entry: &CatalogEntry) -> bool {
    !entry.files.is_empty()
        && entry.files.iter().all(|file| {
            let by_id = join_catalog_path(root, &entry.id).and_then(|dir| join_catalog_path(&dir, &file.path));
            let by_path = join_catalog_path(root, &file.path);
            let by_name = file
                .sources
                .first()
                .and_then(|s| s.url.rsplit('/').next())
                .and_then(|name| join_catalog_path(root, name));
            [by_id, by_path, by_name]
                .into_iter()
                .flatten()
                .any(|p| p.is_file())
        })
}

/// The GPU variant of transcribe.cpp the primary GPU can use, if any.
fn gpu_variant_for(sys: &SystemInfo) -> Option<String> {
    let cuda = sys.eps.cuda.eligible
        && sys
            .primary()
            .is_some_and(|g| eps::meets_cuda_floor(g) != Some(false));
    if cuda {
        return Some("cuda13".into());
    }
    match policy::gpu_runtime(RecommendationTopic::SttFinal) {
        policy::GpuRuntime::VulkanOrCuda => Some("vulkan".into()),
        policy::GpuRuntime::Cuda13Only => None,
    }
}

/// Split `ids` into (installed, unavailable): an id is unavailable when the catalog in effect does
/// not have it, or it is unpublished (a Sigil-built pack: local import or the local-assets folder
/// only) and neither installed nor `has_local_source`. Without a catalog only installed ids are
/// available.
fn availability(
    ids: &BTreeSet<String>,
    catalog: Option<&BTreeMap<String, PublishStatus>>,
    is_installed: impl Fn(&str) -> bool,
    has_local_source: impl Fn(&str) -> bool,
) -> (BTreeSet<String>, BTreeSet<String>) {
    let installed: BTreeSet<String> = ids.iter().filter(|id| is_installed(id)).cloned().collect();
    let unavailable = ids
        .iter()
        .filter(|id| {
            let here = installed.contains(id.as_str());
            match catalog.map(|c| c.get(id.as_str())) {
                None => !here,
                Some(None) => true,
                Some(Some(PublishStatus::Unpublished)) => !here && !has_local_source(id),
                Some(Some(PublishStatus::Published)) => false,
            }
        })
        .cloned()
        .collect();
    (installed, unavailable)
}

/// The context for a system probe and a resource snapshot (pure; [`context`] feeds it live data).
fn context_from(sys: &SystemInfo, snap: &ResourceSnapshot, installed: BTreeSet<String>, unavailable: BTreeSet<String>) -> RouteContext {
    let room = policy::room(RecommendationTopic::SttFinal, sys, snap);
    let runtime_ok = policy::gpu_runtime_available(RecommendationTopic::SttFinal, sys);
    let gpu_variant = (room.gpu == GpuClass::Usable && runtime_ok)
        .then(|| gpu_variant_for(sys))
        .flatten();
    let gpu = gpu_variant.is_some();
    let gpu_reason = (!gpu).then(|| {
        match room.gpu {
            GpuClass::NoGpu => "no_gpu",
            GpuClass::Unknown => "gpu_unknown",
            GpuClass::Integrated => "integrated_gpu",
            GpuClass::TooSmall => "small_gpu",
            GpuClass::Usable => "no_gpu_runtime",
        }
        .to_string()
    });
    RouteContext {
        gpu,
        gpu_variant,
        vram_available_mb: room.available_mb,
        vc_running: snap.vc_running,
        physical_cores: sys.cpu.physical_cores,
        installed,
        vram_total_mb: room.total_mb,
        vc_gpu: if snap.vc_running { snap.vc_gpu } else { None },
        unavailable,
        gpu_reason,
    }
}

/// The live context (cached probes and a few `stat` calls; call it off the async runtime).
pub fn context() -> RouteContext {
    let sys = sysinfo::cached();
    let snap = resources::snapshot();
    let ids = table()
        .map(RoutingTable::referenced_ids)
        .unwrap_or_default();
    let facts = catalog_facts();
    let local_root = local_assets_root();
    let (installed, unavailable) = availability(
        &ids,
        facts.map(|f| &f.status),
        |id| model_store::installed_path(id).is_some(),
        |id| {
            local_root
                .as_deref()
                .zip(facts.and_then(|f| f.unpublished.get(id)))
                .is_some_and(|(root, entry)| has_local_assets(root, entry))
        },
    );
    context_from(&sys, &snap, installed, unavailable)
}

// ---------------------------------------------------------------------------------------------
// Table
// ---------------------------------------------------------------------------------------------

/// One model the table may route to.
#[derive(Clone, Debug, PartialEq, Eq, Deserialize)]
#[serde(rename_all = "camelCase", deny_unknown_fields)]
pub(crate) struct ModelRow {
    pub engine: EngineKind,
    /// Runs on the GPU (transcribe.cpp GPU variant; its runtimes come from `gpuRuntimes`).
    #[serde(default)]
    pub gpu: bool,
    /// Runtime packs a CPU model needs.
    #[serde(default)]
    pub runtimes: Vec<String>,
    /// Languages it may be routed for (normalized codes); empty = any.
    #[serde(default)]
    pub languages: Vec<String>,
    #[serde(default)]
    pub needs_language: bool,
    #[serde(default)]
    pub detects_language: bool,
    /// Never in `"auto"` mode.
    #[serde(default)]
    pub requires_language: bool,
    #[serde(default)]
    pub never_languages: Vec<String>,
    /// Pass an explicit language as `options.language` (sherpa engines).
    #[serde(default)]
    pub language_option: bool,
    #[serde(default)]
    pub vram_mb: u64,
    #[serde(default)]
    pub min_total_vram_mb: u64,
    #[serde(default)]
    pub min_cores: u32,
}

/// One language row.
#[derive(Clone, Debug, Default, PartialEq, Eq, Deserialize)]
#[serde(rename_all = "camelCase", deny_unknown_fields)]
pub(crate) struct RouteRow {
    pub name: String,
    /// Normalized codes, or `auto` / `*`.
    pub languages: Vec<String>,
    #[serde(default)]
    pub streaming: Vec<String>,
    /// The streaming model's own finals are caption quality (a CPU single pass uses it alone).
    #[serde(default)]
    pub stream_final: bool,
    #[serde(default)]
    pub cpu_final: Vec<String>,
    #[serde(default)]
    pub gpu_final: Vec<String>,
}

/// The `routing` object of `stt-routing.json`.
#[derive(Clone, Debug, PartialEq, Eq, Deserialize)]
#[serde(rename_all = "camelCase", deny_unknown_fields)]
pub(crate) struct RoutingTable {
    pub version: u32,
    /// Spoken-language-ID engine for `"auto"` with a final that cannot detect the language.
    #[serde(default)]
    pub language_id: Option<String>,
    pub cpu_two_pass_min_cores: u32,
    pub vc_share_gpu_min_total_mb: u64,
    /// GPU variant → runtime packs a GPU model needs on it.
    pub gpu_runtimes: BTreeMap<String, Vec<String>>,
    pub models: BTreeMap<String, ModelRow>,
    pub routes: Vec<RouteRow>,
}

impl RoutingTable {
    /// Parse a whole `stt-routing.json` document.
    pub(crate) fn from_document(text: &str) -> Result<Self, String> {
        let doc: serde_json::Value = serde_json::from_str(text).map_err(|e| format!("stt-routing.json: {e}"))?;
        let routing = doc
            .get("routing")
            .cloned()
            .ok_or_else(|| "stt-routing.json has no 'routing' object".to_string())?;
        let table: RoutingTable = serde_json::from_value(routing).map_err(|e| format!("stt-routing.json routing: {e}"))?;
        table.validate()?;
        Ok(table)
    }

    /// Structural checks: known version, every referenced model defined with a kind that fits its
    /// role, rows for `auto` and `*`, GPU models only in GPU lists.
    fn validate(&self) -> Result<(), String> {
        if self.version != 1 {
            return Err(format!("stt-routing.json: unsupported routing version {}", self.version));
        }
        let model = |id: &str, row: &str| {
            self.models
                .get(id)
                .ok_or_else(|| format!("stt-routing.json: row '{row}' names unknown model {id}"))
        };
        let mut has_auto = false;
        let mut has_other = false;
        for row in &self.routes {
            has_auto |= row.languages.iter().any(|l| l == lang::AUTO);
            has_other |= row.languages.iter().any(|l| l == "*");
            for l in &row.languages {
                if l != lang::AUTO && l != "*" && lang::normalize(l).as_deref() != Some(l.as_str()) {
                    return Err(format!("stt-routing.json: row '{}' has a non-normalized language '{l}'", row.name));
                }
            }
            for id in &row.streaming {
                let m = model(id, &row.name)?;
                if m.engine != EngineKind::SherpaOnline || m.gpu {
                    return Err(format!("stt-routing.json: {id} in '{}' is not a CPU streaming model", row.name));
                }
            }
            for id in &row.cpu_final {
                if model(id, &row.name)?.gpu {
                    return Err(format!("stt-routing.json: GPU model {id} in the CPU finals of '{}'", row.name));
                }
            }
            for id in &row.gpu_final {
                if !model(id, &row.name)?.gpu {
                    return Err(format!("stt-routing.json: CPU model {id} in the GPU finals of '{}'", row.name));
                }
            }
        }
        if !has_auto || !has_other {
            return Err("stt-routing.json: the 'auto' and '*' rows are required".into());
        }
        if let Some(id) = &self.language_id {
            let m = self
                .models
                .get(id)
                .ok_or_else(|| format!("stt-routing.json: unknown languageId model {id}"))?;
            if !m.detects_language || m.gpu {
                return Err(format!("stt-routing.json: languageId {id} must be a CPU model that detects languages"));
            }
        }
        Ok(())
    }

    /// The row for a normalized language (`None` = auto).
    fn row_for(&self, language: Option<&str>) -> Option<&RouteRow> {
        let key = language.unwrap_or(lang::AUTO);
        self.routes
            .iter()
            .find(|r| r.languages.iter().any(|l| l == key))
            .or_else(|| {
                language
                    .is_some()
                    .then(|| {
                        self.routes
                            .iter()
                            .find(|r| r.languages.iter().any(|l| l == "*"))
                    })
                    .flatten()
            })
    }

    /// Every catalog id the table can put in a plan (models and runtime packs).
    pub(crate) fn referenced_ids(&self) -> BTreeSet<String> {
        let mut ids: BTreeSet<String> = self.models.keys().cloned().collect();
        for m in self.models.values() {
            ids.extend(m.runtimes.iter().cloned());
        }
        for packs in self.gpu_runtimes.values() {
            ids.extend(packs.iter().cloned());
        }
        ids.extend(self.language_id.iter().cloned());
        ids
    }
}

/// The embedded table (parsed once).
pub(crate) fn table() -> Result<&'static RoutingTable, &'static str> {
    static TABLE: OnceLock<Result<RoutingTable, String>> = OnceLock::new();
    TABLE
        .get_or_init(|| RoutingTable::from_document(TABLE_JSON).inspect_err(|e| tracing::error!(target: LOG, "{e}")))
        .as_ref()
        .map_err(String::as_str)
}

// ---------------------------------------------------------------------------------------------
// Planning
// ---------------------------------------------------------------------------------------------

/// Table-level permission: the model's `languages`, `neverLanguages` and `requiresLanguage`.
fn table_permits(m: &ModelRow, language: Option<&str>) -> bool {
    match language {
        None => !m.requires_language,
        Some(l) => (m.languages.is_empty() || m.languages.iter().any(|x| x == l)) && !m.never_languages.iter().any(|x| x == l),
    }
}

/// The never-route rules of this module, whatever the table says: [`NEVER`], [`NEVER_AUTO`], and a
/// sherpa model only for the languages its own catalog entry (`speech_rt::sherpa::catalog`) lists.
fn code_permits(id: &str, language: Option<&str>) -> bool {
    match language {
        None => !NEVER_AUTO.iter().any(|p| id.starts_with(p)),
        Some(l) => {
            !NEVER
                .iter()
                .any(|(prefix, langs)| id.starts_with(prefix) && langs.contains(&l))
                && sherpa_catalog::find(id).is_none_or(|m| m.supports_language(l))
        }
    }
}

fn needs_language(id: &str, m: &ModelRow) -> bool {
    m.needs_language || NEEDS_LANGUAGE.iter().any(|p| id.starts_with(p))
}

/// A chosen model.
#[derive(Clone, Copy, Debug)]
struct Pick<'a> {
    id: &'a str,
    row: &'a ModelRow,
}

impl Pick<'_> {
    fn needs_language(&self) -> bool {
        needs_language(self.id, self.row)
    }
}

struct Planner<'a> {
    table: &'a RoutingTable,
    ctx: &'a RouteContext,
    /// Normalized language; `None` = auto.
    language: Option<&'a str>,
    notes: Vec<String>,
}

impl<'a> Planner<'a> {
    fn note(&mut self, code: &str) {
        if !self.notes.iter().any(|n| n == code) {
            self.notes.push(code.to_string());
        }
    }

    fn available(&self, id: &str) -> bool {
        !self.ctx.unavailable.contains(id)
    }

    /// In the table, installable, its CPU runtimes installable, enough cores, and allowed for the
    /// language by the table and by the code rules (a table entry against a code rule is logged).
    fn usable(&mut self, id: &'a str) -> Option<Pick<'a>> {
        let row = self.table.models.get(id)?;
        if !table_permits(row, self.language) {
            return None;
        }
        if !code_permits(id, self.language) {
            tracing::warn!(
                target: LOG,
                "routing table lists {id} for {}, which a never-route rule forbids; skipped",
                self.language.unwrap_or(lang::AUTO)
            );
            self.note(notes::NEVER_ROUTE);
            return None;
        }
        if !self.available(id) || !row.runtimes.iter().all(|r| self.available(r)) {
            return None;
        }
        if row.min_cores > 0 && self.ctx.physical_cores > 0 && self.ctx.physical_cores < row.min_cores {
            return None;
        }
        Some(Pick { id, row })
    }

    fn first_usable(&mut self, ids: &'a [String], gpu: bool) -> Option<Pick<'a>> {
        ids.iter()
            .find_map(|id| self.usable(id).filter(|p| p.row.gpu == gpu))
    }

    /// The voice changer runs and may use the GPU (unknown counts as yes).
    fn vc_on_gpu(&self) -> bool {
        self.ctx.vc_running && self.ctx.vc_gpu != Some(false)
    }

    /// The voice changer runs and may use the CPU (unknown counts as yes, as in [`Self::vc_on_gpu`]:
    /// while its model loads the device is not known, and the plan lasts the whole session).
    fn vc_on_cpu(&self) -> bool {
        self.ctx.vc_running && self.ctx.vc_gpu != Some(true)
    }

    /// The CPU may run two speech engines at once: enough physical cores (0 = unknown counts as
    /// enough) and no voice changer that may use the CPU.
    fn cpu_runs_two_engines(&self) -> bool {
        let cores = self.ctx.physical_cores;
        (cores == 0 || cores >= self.table.cpu_two_pass_min_cores) && !self.vc_on_cpu()
    }

    /// The GPU variant the GPU finals may use for this request, with notes for why not.
    fn gpu_variant(&mut self, tier: Tier) -> Option<&'a str> {
        if tier == Tier::Cpu {
            return None;
        }
        let ctx = self.ctx;
        let Some(variant) = ctx.gpu_variant.as_deref().filter(|_| ctx.gpu) else {
            if tier == Tier::Gpu {
                self.note(notes::GPU_UNAVAILABLE);
            }
            if let Some(reason) = ctx.gpu_reason.as_deref() {
                self.note(reason);
            }
            return None;
        };
        let packs = self.table.gpu_runtimes.get(variant);
        if !packs.is_some_and(|p| p.iter().all(|id| self.available(id))) {
            self.note(notes::GPU_RUNTIME_UNAVAILABLE);
            return None;
        }
        if tier == Tier::Auto && self.vc_on_gpu() && ctx.vram_total_mb < self.table.vc_share_gpu_min_total_mb {
            self.note(notes::GPU_DENIED_VC_RUNNING);
            return None;
        }
        Some(variant)
    }

    /// The first GPU final of `ids` that fits, with notes when none does.
    fn gpu_final(&mut self, ids: &'a [String]) -> Option<Pick<'a>> {
        let mut candidates = Vec::new();
        for id in ids {
            if let Some(p) = self.usable(id).filter(|p| p.row.gpu) {
                candidates.push(p);
            }
        }
        if candidates.is_empty() {
            self.note(notes::NO_GPU_MODEL_FOR_LANGUAGE);
            return None;
        }
        let ctx = self.ctx;
        let fit = candidates
            .into_iter()
            .find(|p| p.row.vram_mb <= ctx.vram_available_mb && (p.row.min_total_vram_mb == 0 || ctx.vram_total_mb >= p.row.min_total_vram_mb));
        if fit.is_none() {
            self.note(notes::GPU_FULL);
        }
        fit
    }

    fn spec(&self, pick: Pick<'_>, variant: Option<&str>) -> EngineSpec {
        let mut spec = EngineSpec::new(pick.row.engine, pick.id);
        if pick.row.gpu {
            spec.variant = variant.map(str::to_string);
        }
        if let Some(l) = self.language {
            if pick.row.language_option || pick.needs_language() {
                spec.options = serde_json::json!({ "language": l });
            }
        }
        spec
    }

    /// Catalog ids a pick needs: the model, then its runtime packs.
    fn ids_of(&self, pick: Pick<'_>, variant: Option<&str>, out: &mut Vec<String>) {
        let runtimes: &[String] = if pick.row.gpu {
            variant
                .and_then(|v| self.table.gpu_runtimes.get(v))
                .map_or(&[], Vec::as_slice)
        } else {
            &pick.row.runtimes
        };
        for id in std::iter::once(pick.id).chain(runtimes.iter().map(String::as_str)) {
            if !out.iter().any(|x| x == id) {
                out.push(id.to_string());
            }
        }
    }
}

/// `"auto"` mode: the detected languages the streaming model `id` may be opened with, sorted. A
/// language qualifies when the model may run it (the table's `languages` / `neverLanguages` and the
/// code rules, including its sherpa catalog languages) and that language's own row streams with
/// this model: auto mode never streams where an explicit choice of the detected language would not
/// (zh / yue have no partials, Nemotron has no Thai). The candidates are the model's table
/// languages, else its sherpa catalog languages, else the languages of every explicit row.
/// `None` (no limit) only for a model without any language list that the `*` row streams with.
fn auto_stream_languages(table: &RoutingTable, id: &str, m: &ModelRow) -> Option<Vec<String>> {
    let streams = |row: &RouteRow| row.streaming.iter().any(|x| x == id);
    let catalog: Vec<String> = sherpa_catalog::find(id)
        .map(|c| c.languages.iter().map(|l| (*l).to_string()).collect())
        .unwrap_or_default();
    let candidates: Vec<String> = if !m.languages.is_empty() {
        m.languages.clone()
    } else if !catalog.is_empty() {
        catalog
    } else if table
        .routes
        .iter()
        .any(|r| r.languages.iter().any(|l| l == "*") && streams(r))
    {
        return None;
    } else {
        table
            .routes
            .iter()
            .flat_map(|r| r.languages.iter())
            .filter(|l| l.as_str() != lang::AUTO && l.as_str() != "*")
            .cloned()
            .collect()
    };
    let allowed: BTreeSet<String> = candidates
        .into_iter()
        .filter(|l| table_permits(m, Some(l)) && code_permits(id, Some(l)) && table.row_for(Some(l)).is_some_and(streams))
        .collect();
    Some(allowed.into_iter().collect())
}

/// The languages the pipeline may open `spec`'s stream with (`pipeline::StreamFeed`, where an
/// empty list means "any"): an `"auto"` plan's [`STREAM_LANGUAGES_OPTION`] narrowed to
/// `engine_languages` (the engine's own list, empty = any), else `engine_languages`. A plan limit
/// is never widened to "any": when it does not overlap the engine's list (a table / catalog
/// mismatch the tests rule out) the plan's list is returned as is.
pub(crate) fn stream_languages(spec: &EngineSpec, engine_languages: &[String]) -> Vec<String> {
    let Some(limit) = spec
        .options
        .get(STREAM_LANGUAGES_OPTION)
        .and_then(|v| v.as_array())
    else {
        return engine_languages.to_vec();
    };
    let limit: BTreeSet<String> = limit
        .iter()
        .filter_map(|v| v.as_str())
        .filter_map(lang::normalize)
        .collect();
    if limit.is_empty() {
        // Written by the router only when non-empty; a hand-made empty list still limits.
        return vec![lang::AUTO.to_string()];
    }
    let both: Vec<String> = limit
        .iter()
        .filter(|l| engine_languages.is_empty() || engine_languages.contains(l))
        .cloned()
        .collect();
    if both.is_empty() {
        limit.into_iter().collect()
    } else {
        both
    }
}

/// Plan against an explicit table (tests; [`plan_with`] uses the embedded one).
pub(crate) fn plan_table(table: &RoutingTable, request: &RouteRequest, ctx: &RouteContext) -> RoutePlan {
    let normalized = lang::normalize(&request.language);
    let mut p = Planner {
        table,
        ctx,
        language: normalized.as_deref(),
        notes: Vec::new(),
    };
    let language = normalized.clone().unwrap_or_else(|| lang::AUTO.to_string());
    let auto = p.language.is_none();
    if auto {
        p.note(notes::AUTO_LANGUAGE);
    }
    let Some(row) = table.row_for(p.language) else {
        p.note(notes::NO_ENGINE);
        return RoutePlan {
            language,
            tier: Tier::Cpu,
            notes: p.notes,
            ..RoutePlan::default()
        };
    };
    if row.languages.iter().any(|l| l == "*") {
        p.note(notes::GENERIC_LANGUAGE_MODEL);
    }

    // 1. Candidates. In auto mode the stream runs only for detected languages whose own row
    //    streams with it; a model that no such language allows is no candidate at all.
    let mut streaming = p.first_usable(&row.streaming, false);
    let mut auto_languages = None;
    if auto {
        if let Some(s) = streaming {
            match auto_stream_languages(table, s.id, s.row) {
                Some(list) if list.is_empty() => {
                    p.note(notes::NEMOTRON_NEEDS_LANGUAGE);
                    streaming = None;
                }
                other => auto_languages = other,
            }
        }
    }
    let cpu_final = p.first_usable(&row.cpu_final, false);
    let variant = p.gpu_variant(request.tier);
    let gpu_final = match variant {
        Some(_) => p.gpu_final(&row.gpu_final),
        None => None,
    };
    let (tier, variant) = match gpu_final {
        Some(_) => (Tier::Gpu, variant),
        None => (Tier::Cpu, None),
    };
    if tier == Tier::Gpu && p.vc_on_gpu() {
        p.note(notes::VC_SHARES_GPU);
    }
    let best_final = gpu_final.or(cpu_final);

    // 2. Passes.
    let cores_ok = ctx.physical_cores == 0 || ctx.physical_cores >= table.cpu_two_pass_min_cores;
    let cpu_two_engines = p.cpu_runs_two_engines();
    let (mut streaming, final_pick) = match (streaming, best_final) {
        (Some(s), Some(f)) => {
            if request.two_pass && (tier == Tier::Gpu || cpu_two_engines) {
                if request.interim {
                    (Some(s), Some(f))
                } else {
                    // No live partials wanted: the stream would only feed first-pass text that the
                    // final replaces, so it is not planned (never loaded or run) and the final
                    // engine decodes each closed utterance alone (`finals_only`). Only here: a
                    // stream that is the plan's sole engine stays, its finals are the captions.
                    (None, Some(f))
                }
            } else {
                p.note(if !request.two_pass {
                    notes::SINGLE_PASS
                } else if !cores_ok {
                    notes::FEW_CORES_SINGLE_PASS
                } else if ctx.vc_gpu == Some(false) {
                    notes::VC_CPU_SINGLE_PASS
                } else {
                    notes::VC_UNKNOWN_SINGLE_PASS
                });
                // The stream alone only where its own finals are caption quality, and never a
                // stream that would have to guess its language.
                let stream_alone = tier == Tier::Cpu && row.stream_final && !(auto && s.needs_language());
                if stream_alone {
                    (Some(s), None)
                } else {
                    (None, Some(f))
                }
            }
        }
        other => other,
    };

    // 3. Language: auto mode uses spoken language ID (a final that detects the language, else the
    //    table's language-ID engine run first on each utterance), and a model that needs a
    //    language always gets one (the pipeline holds its stream until the first detection).
    let mut language_id = None;
    if auto {
        if final_pick.is_some_and(|f| !f.row.detects_language) {
            language_id = table.language_id.as_deref().and_then(|id| p.usable(id));
            if language_id.is_some() {
                p.note(notes::LANGUAGE_ID);
            }
        }
        if streaming.is_some_and(|s| s.needs_language()) {
            let detected = final_pick.is_some_and(|f| f.row.detects_language) || language_id.is_some();
            if final_pick.is_none() || !detected {
                p.note(notes::NEMOTRON_NEEDS_LANGUAGE);
                streaming = None;
            }
        }
    }
    match (streaming, final_pick) {
        (None, Some(_)) => p.note(notes::FINALS_ONLY),
        (None, None) => p.note(notes::NO_ENGINE),
        _ => {}
    }

    // 4. The CPU final behind a GPU final. Next to a CPU stream it would be a second CPU engine
    //    after a GPU failure, so only where the CPU may run two (else the first-pass finals stand);
    //    a finals-only GPU plan always keeps it (it replaces the GPU engine).
    let fallback = match (final_pick, cpu_final) {
        (Some(f), Some(c)) if f.row.gpu && c.id != f.id => {
            if streaming.is_some() && !cpu_two_engines {
                p.note(notes::NO_CPU_FALLBACK);
                None
            } else {
                Some(c)
            }
        }
        _ => None,
    };

    let mut models = Vec::new();
    for (pick, v) in [(streaming, None), (final_pick, variant), (fallback, None), (language_id, None)] {
        if let Some(pick) = pick {
            p.ids_of(pick, v, &mut models);
        }
    }
    let missing = models
        .iter()
        .filter(|id| !ctx.installed.contains(id.as_str()))
        .cloned()
        .collect();
    let vram_mb = final_pick
        .filter(|f| f.row.gpu)
        .map_or(0, |f| f.row.vram_mb);
    let streaming = streaming.map(|s| {
        let mut spec = p.spec(s, None);
        if let Some(list) = auto_languages {
            // Auto mode: `spec` has no `language` (nothing else in its options).
            let mut options = spec.options.as_object().cloned().unwrap_or_default();
            options.insert(STREAM_LANGUAGES_OPTION.to_string(), serde_json::json!(list));
            spec.options = serde_json::Value::Object(options);
        }
        spec
    });
    RoutePlan {
        language,
        tier,
        streaming,
        final_engine: final_pick.map(|f| p.spec(f, variant)),
        fallback: fallback.map(|f| p.spec(f, None)),
        language_id: language_id.map(|l| p.spec(l, None)),
        models,
        missing,
        vram_mb,
        notes: p.notes,
    }
}

/// Route on the live context.
pub fn plan(request: &RouteRequest) -> RoutePlan {
    plan_with(request, &context())
}

/// Pure routing decision.
pub fn plan_with(request: &RouteRequest, ctx: &RouteContext) -> RoutePlan {
    match table() {
        Ok(t) => plan_table(t, request, ctx),
        Err(_) => RoutePlan {
            language: lang::normalize(&request.language).unwrap_or_else(|| lang::AUTO.to_string()),
            tier: Tier::Cpu,
            notes: vec![notes::ROUTING_TABLE_INVALID.to_string()],
            ..RoutePlan::default()
        },
    }
}

/// `plugin:speech|preview_route`: the plan the pipeline would run for `request` on this PC now.
/// Pure: installs and loads nothing.
#[command]
pub async fn preview_route(request: RouteRequest) -> Result<RoutePlan, String> {
    tauri::async_runtime::spawn_blocking(move || plan(&request))
        .await
        .map_err(|e| format!("preview_route: {e}"))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::services::diagnostics::gpu_mem::{VENDOR_AMD, VENDOR_MICROSOFT, VENDOR_NVIDIA};
    use crate::services::sysinfo::test_support::{gpu, system};

    /// The shared fixture (also read by the TS router mirror test).
    const FIXTURE: &str = include_str!("../../../test-fixtures/routing.json");

    const NEMOTRON: &str = "stt/nemotron-3.5-streaming-560ms-int8";
    const PARAKEET_ULTRA: &str = "stt/parakeet-ultra-int8";
    const QWEN3_0_6B: &str = "stt/qwen3-asr-0.6b-int8";
    const SENSE_VOICE: &str = "stt/sense-voice-small-int8";
    const QWEN3_Q8: &str = "stt/qwen3-asr-1.7b-q8_0";
    const QWEN3_BF16: &str = "stt/qwen3-asr-1.7b-bf16";
    const COHERE: &str = "stt/cohere-transcribe-q4_k";
    const LID: &str = "stt/lid-whisper-base-int8";

    fn plan_models(plan: &RoutePlan) -> Vec<&str> {
        [&plan.streaming, &plan.final_engine, &plan.fallback, &plan.language_id]
            .into_iter()
            .flatten()
            .map(|s| s.model.as_str())
            .chain(plan.models.iter().map(String::as_str))
            .collect()
    }

    fn t() -> &'static RoutingTable {
        table().expect("the embedded routing table loads")
    }

    /// A PC: `vram_total` 0 = CPU only; everything installed.
    fn pc(vram_total: u64, vram_available: u64, cores: u32) -> RouteContext {
        RouteContext {
            gpu: vram_total > 0,
            gpu_variant: (vram_total > 0).then(|| "cuda13".to_string()),
            vram_available_mb: vram_available,
            vram_total_mb: vram_total,
            physical_cores: cores,
            installed: t().referenced_ids(),
            gpu_reason: (vram_total == 0).then(|| "no_gpu".to_string()),
            ..RouteContext::default()
        }
    }

    fn req(language: &str, tier: Tier, two_pass: bool) -> RouteRequest {
        RouteRequest {
            language: language.into(),
            tier,
            two_pass,
            interim: true,
        }
    }

    /// [`req`] with interim results (live partial captions) off.
    fn req_quiet(language: &str, tier: Tier, two_pass: bool) -> RouteRequest {
        RouteRequest {
            interim: false,
            ..req(language, tier, two_pass)
        }
    }

    fn model(spec: &Option<EngineSpec>) -> Option<&str> {
        spec.as_ref().map(|s| s.model.as_str())
    }

    fn check_optional(name: &str, key: &str, want: &serde_json::Value, got: Option<&str>) {
        match want {
            serde_json::Value::Null => assert_eq!(got, None, "{name}: {key}"),
            serde_json::Value::String(id) => assert_eq!(got, Some(id.as_str()), "{name}: {key}"),
            other => panic!("{name}: {key} must be a string or null, not {other}"),
        }
    }

    /// Every case parses into the request / context types, and every expectation holds.
    #[test]
    fn shared_fixture_cases_hold() {
        let doc: serde_json::Value = serde_json::from_str(FIXTURE).unwrap();
        assert_eq!(doc["version"], 1);
        let cases = doc["cases"].as_array().unwrap();
        assert!(cases.len() >= 30, "the fixture covers every rule");
        let mut names = BTreeSet::new();
        for case in cases {
            let name = case["name"].as_str().unwrap();
            assert!(names.insert(name), "duplicate case name {name}");
            let request: RouteRequest = serde_json::from_value(case["request"].clone()).unwrap();
            let ctx: RouteContext = serde_json::from_value(case["context"].clone()).unwrap();
            let plan = plan_with(&request, &ctx);
            let models = plan_models(&plan);
            let expect = case["expect"]
                .as_object()
                .unwrap_or_else(|| panic!("{name}: expect"));
            for (key, want) in expect {
                match key.as_str() {
                    "language" => assert_eq!(plan.language, want.as_str().unwrap(), "{name}: language"),
                    "tier" => assert_eq!(serde_json::to_value(plan.tier).unwrap(), *want, "{name}: tier {plan:?}"),
                    "streamingModel" => check_optional(name, key, want, model(&plan.streaming)),
                    "finalModel" => check_optional(name, key, want, model(&plan.final_engine)),
                    "fallbackModel" => check_optional(name, key, want, model(&plan.fallback)),
                    "languageIdModel" => check_optional(name, key, want, model(&plan.language_id)),
                    "streamingLanguage" => {
                        let got = plan
                            .streaming
                            .as_ref()
                            .and_then(|s| s.options.get("language"))
                            .and_then(|l| l.as_str());
                        check_optional(name, key, want, got);
                    }
                    "neverModels" => {
                        for never in want.as_array().unwrap() {
                            let never = never.as_str().unwrap();
                            assert!(!models.contains(&never), "{name}: plan uses {never}: {plan:?}");
                        }
                    }
                    "notesInclude" => {
                        for n in want.as_array().unwrap() {
                            let n = n.as_str().unwrap();
                            assert!(plan.notes.iter().any(|x| x == n), "{name}: note {n} missing from {:?}", plan.notes);
                        }
                    }
                    "missingInclude" => {
                        for id in want.as_array().unwrap() {
                            let id = id.as_str().unwrap();
                            assert!(plan.missing.iter().any(|x| x == id), "{name}: {id} not in missing {:?}", plan.missing);
                        }
                    }
                    other => panic!("{name}: unknown expect key {other}"),
                }
            }
        }
    }

    #[test]
    fn embedded_table_is_valid_and_rows_do_not_overlap() {
        let table = t();
        let mut seen = BTreeMap::new();
        for row in &table.routes {
            for l in &row.languages {
                if let Some(other) = seen.insert(l.clone(), row.name.clone()) {
                    panic!("language {l} is in rows {other} and {}", row.name);
                }
            }
        }
        // A broken document is refused with a message instead of a panic.
        assert!(RoutingTable::from_document("{}").is_err());
        assert!(RoutingTable::from_document(r#"{"routing":{"version":2}}"#).is_err());
        assert!(RoutingTable::from_document("not json").is_err());
    }

    /// The plan's rules as data: zh / yue → SenseVoice, ja / ko / ru → Qwen3-ASR, European
    /// languages → a Parakeet Ultra final on the CPU, Cohere never for ja / zh / ko / ru, the
    /// Nemotron language rule.
    #[test]
    fn shipped_table_encodes_the_plan_rules() {
        let table = t();
        let row = |l: &str| table.row_for(Some(l)).unwrap();
        for l in ["zh", "yue"] {
            assert_eq!(row(l).cpu_final.first().map(String::as_str), Some(SENSE_VOICE), "{l}");
            assert!(row(l).streaming.is_empty(), "{l}: no Nemotron partials (Mandarin CER ~19)");
        }
        for l in ["ja", "ko", "ru"] {
            assert_eq!(row(l).cpu_final.first().map(String::as_str), Some(QWEN3_0_6B), "{l}");
            assert_eq!(row(l).gpu_final.last().map(String::as_str), Some(QWEN3_Q8), "{l}");
        }
        for l in table.models[PARAKEET_ULTRA]
            .languages
            .iter()
            .filter(|l| l.as_str() != "ru")
        {
            assert_eq!(row(l).cpu_final.first().map(String::as_str), Some(PARAKEET_ULTRA), "{l}");
        }
        let cohere = &table.models[COHERE];
        for l in ["ja", "zh", "ko", "ru", "yue"] {
            assert!(!cohere.languages.iter().any(|x| x == l) && cohere.never_languages.iter().any(|x| x == l));
            assert!(!row(l).gpu_final.iter().any(|x| x == COHERE), "{l}");
        }
        assert!(cohere.requires_language);
        assert!(!table
            .row_for(None)
            .unwrap()
            .gpu_final
            .iter()
            .any(|x| x == COHERE));
        assert!(table.models[NEMOTRON].needs_language);
        assert_eq!(table.language_id.as_deref(), Some(LID));
    }

    /// The table's sherpa entries agree with `speech_rt::sherpa::catalog` (languages and
    /// capabilities), and every id is in the embedded catalog. GPU (transcribe.cpp) models that
    /// the catalog does not list yet are reported: until P2-D adds them the live context marks
    /// them unavailable and the router never plans them.
    #[test]
    fn table_models_match_the_catalogs() {
        let table = t();
        let catalog = Catalog::embedded().unwrap();
        for (id, m) in &table.models {
            match m.engine {
                EngineKind::SherpaOnline | EngineKind::SherpaOffline => {
                    let c = sherpa_catalog::find(id).unwrap_or_else(|| panic!("{id} is not a sherpa catalog model"));
                    for l in &m.languages {
                        assert!(c.supports_language(l), "{id} does not support {l}");
                    }
                    assert_eq!(m.languages.is_empty(), c.languages.is_empty(), "{id}");
                    assert_eq!(m.needs_language, c.family.needs_language(), "{id}");
                    assert_eq!(m.detects_language, c.family.detects_language(), "{id}");
                    assert_eq!(m.engine == EngineKind::SherpaOnline, c.task() == sherpa_catalog::Task::Online, "{id}");
                    assert!(!m.gpu, "{id}");
                }
                EngineKind::Whisper => assert!(id.starts_with("whisper/") && m.languages.is_empty(), "{id}"),
                EngineKind::Transcribe => assert!(m.gpu && m.vram_mb > 0, "{id}"),
                EngineKind::Http => panic!("{id}: HTTP engines are not routed"),
            }
            if catalog.get(id).is_none() {
                assert_eq!(m.engine, EngineKind::Transcribe, "{id} is missing from the catalog");
                eprintln!("{id} is not in the catalog yet (P2-D's stt-gguf.json); the router skips it until then");
            }
        }
        for id in table
            .gpu_runtimes
            .values()
            .flatten()
            .chain(table.models.values().flat_map(|m| m.runtimes.iter()))
        {
            assert!(catalog.get(id).is_some(), "{id}");
        }
    }

    /// Every language × tier × two-pass × PC keeps the invariants.
    #[test]
    fn every_request_on_every_pc_keeps_the_rules() {
        let table = t();
        let mut languages: Vec<String> = table
            .routes
            .iter()
            .flat_map(|r| r.languages.iter().cloned())
            .filter(|l| l != "*")
            .collect();
        languages.extend(["sw", "he", "zh-HK", "pt-BR", "zh-yue", "", "xx-invalid-1"].map(String::from));
        let mut cpu_vc = pc(0, 0, 8);
        cpu_vc.vc_running = true;
        cpu_vc.vc_gpu = Some(false);
        let mut gpu_vc = pc(12_288, 5_400, 8);
        gpu_vc.vc_running = true;
        gpu_vc.vc_gpu = Some(true);
        let mut no_pack = pc(24_576, 15_000, 16);
        no_pack
            .unavailable
            .insert("runtime/sigil-transcribe-cuda13".into());
        let mut nothing_installed = pc(24_576, 15_000, 16);
        nothing_installed.installed.clear();
        let mut cpu_vc_unknown = pc(0, 0, 8);
        cpu_vc_unknown.vc_running = true;
        let mut big_vc_unknown = pc(24_576, 15_000, 16);
        big_vc_unknown.vc_running = true;
        let pcs = [
            ("cpu 8 cores", pc(0, 0, 8)),
            ("cpu 4 cores", pc(0, 0, 4)),
            ("cpu unknown cores", pc(0, 0, 0)),
            ("8 GB", pc(8_188, 3_600, 6)),
            ("12 GB", pc(12_288, 6_600, 8)),
            ("16 GB", pc(16_376, 11_400, 8)),
            ("24 GB", pc(24_576, 15_000, 16)),
            ("small free VRAM", pc(12_288, 1_000, 8)),
            ("vc on cpu", cpu_vc),
            ("vc on gpu", gpu_vc),
            ("no transcribe pack", no_pack),
            ("nothing installed", nothing_installed),
            ("cpu, vc device unknown", cpu_vc_unknown),
            ("24 GB, vc device unknown", big_vc_unknown),
        ];
        let is_gpu = |s: &EngineSpec| table.models.get(&s.model).is_some_and(|m| m.gpu);
        for (pc_name, ctx) in &pcs {
            for l in &languages {
                for tier in [Tier::Auto, Tier::Cpu, Tier::Gpu] {
                    for two_pass in [true, false] {
                        let r = req(l, tier, two_pass);
                        let plan = plan_with(&r, ctx);
                        let what = format!("{pc_name} / {l:?} / {tier:?} / two-pass {two_pass}: {plan:?}");
                        assert_eq!(plan, plan_with(&r, ctx), "deterministic: {what}");
                        let norm = lang::normalize(l);
                        assert_eq!(plan.language, norm.clone().unwrap_or_else(|| "auto".into()), "{what}");
                        assert!(plan.streaming.is_some() || plan.final_engine.is_some(), "an engine: {what}");
                        assert_ne!(plan.tier, Tier::Auto, "{what}");
                        let models = plan_models(&plan);
                        // Cohere: never ja / zh / ko / ru / yue, never auto.
                        if norm
                            .as_deref()
                            .is_none_or(|c| ["ja", "zh", "ko", "ru", "yue"].contains(&c))
                        {
                            assert!(!models.iter().any(|m| m.starts_with("stt/cohere-")), "{what}");
                        }
                        // Nemotron always gets a language.
                        if let Some(s) = plan
                            .streaming
                            .as_ref()
                            .filter(|s| s.model.starts_with("stt/nemotron-"))
                        {
                            match &norm {
                                Some(code) => assert_eq!(s.options["language"], code.as_str(), "{what}"),
                                None => {
                                    let f = plan
                                        .final_engine
                                        .as_ref()
                                        .expect("a final next to Nemotron in auto mode");
                                    assert!(table.models[&f.model].detects_language || plan.language_id.is_some(), "{what}");
                                }
                            }
                        }
                        // Auto mode streams only for languages whose own row streams with the
                        // model (never zh / yue); explicit languages carry no limit.
                        let limit = plan
                            .streaming
                            .as_ref()
                            .and_then(|s| s.options.get(STREAM_LANGUAGES_OPTION));
                        match (&norm, &plan.streaming) {
                            (None, Some(s)) => {
                                let list = limit
                                    .and_then(|v| v.as_array())
                                    .expect("an auto stream limit");
                                assert!(!list.is_empty(), "{what}");
                                for l in list {
                                    let l = l.as_str().unwrap();
                                    assert!(!["zh", "yue"].contains(&l), "{what}");
                                    let row = table.row_for(Some(l)).unwrap();
                                    assert!(row.streaming.contains(&s.model), "{l}: {what}");
                                }
                            }
                            _ => assert!(limit.is_none(), "{what}"),
                        }
                        // A second CPU engine (CPU two-pass, or the CPU fallback of a GPU final next
                        // to the stream) only where the CPU may run two.
                        let cpu_two = (ctx.physical_cores == 0 || ctx.physical_cores >= table.cpu_two_pass_min_cores)
                            && !(ctx.vc_running && ctx.vc_gpu != Some(true));
                        if plan.streaming.is_some() && (plan.fallback.is_some() || (plan.tier == Tier::Cpu && plan.final_engine.is_some())) {
                            assert!(cpu_two, "{what}");
                        }
                        // GPU only on the GPU tier; streaming, fallbacks and language ID on the CPU.
                        for s in [&plan.streaming, &plan.fallback, &plan.language_id]
                            .into_iter()
                            .flatten()
                        {
                            assert!(!is_gpu(s) && s.variant.is_none(), "{what}");
                        }
                        if plan.tier == Tier::Gpu {
                            let f = plan.final_engine.as_ref().unwrap();
                            assert!(is_gpu(f) && f.variant.as_deref() == Some("cuda13") && plan.vram_mb > 0, "{what}");
                            assert!(plan.vram_mb <= ctx.vram_available_mb, "{what}");
                            assert!(
                                plan.models
                                    .iter()
                                    .any(|m| m == "runtime/sigil-transcribe-cuda13"),
                                "{what}"
                            );
                            assert_ne!(tier, Tier::Cpu, "{what}");
                        } else {
                            assert!(plan.final_engine.as_ref().is_none_or(|f| !is_gpu(f)), "{what}");
                            assert_eq!(plan.vram_mb, 0, "{what}");
                        }
                        // Bookkeeping: every spec model is listed, missing ⊆ models, nothing
                        // unavailable is planned.
                        for s in [&plan.streaming, &plan.final_engine, &plan.fallback, &plan.language_id]
                            .into_iter()
                            .flatten()
                        {
                            assert!(plan.models.contains(&s.model), "{what}");
                        }
                        for m in &plan.missing {
                            assert!(plan.models.contains(m) && !ctx.installed.contains(m), "{what}");
                        }
                        for m in &plan.models {
                            assert!(!ctx.unavailable.contains(m), "{what}");
                        }
                        if ctx.installed.is_empty() {
                            assert_eq!(plan.missing, plan.models, "{what}");
                        }
                        // The plan's CPU rules.
                        if plan.tier == Tier::Cpu {
                            let fin = model(&plan.final_engine);
                            match norm.as_deref() {
                                Some("zh" | "yue") => assert_eq!(fin, Some(SENSE_VOICE), "{what}"),
                                Some("ja" | "ko" | "ru") => assert_eq!(fin, Some(QWEN3_0_6B), "{what}"),
                                Some(code)
                                    if code != "en"
                                        && table.models[PARAKEET_ULTRA]
                                            .languages
                                            .iter()
                                            .any(|x| x == code) =>
                                {
                                    assert_eq!(fin, Some(PARAKEET_ULTRA), "{what}")
                                }
                                _ => {}
                            }
                        }
                        if norm.is_none() {
                            assert!(plan.notes.iter().any(|n| n == notes::AUTO_LANGUAGE), "{what}");
                        }
                        // Every note is a known code (the UI has text for it).
                        for n in &plan.notes {
                            assert!(
                                notes::ALL.contains(&n.as_str()) || notes::GOVERNOR_REASONS.contains(&n.as_str()),
                                "unknown note {n}: {what}"
                            );
                        }
                    }
                }
            }
        }
    }

    /// A table entry that breaks a never-route rule is skipped by the code, not trusted.
    #[test]
    fn code_rules_override_a_bad_table() {
        let bad = r#"{"routing":{
            "version": 1, "cpuTwoPassMinCores": 6, "vcShareGpuMinTotalMb": 15000,
            "gpuRuntimes": {"cuda13": ["runtime/sigil-transcribe-cuda13"]},
            "models": {
                "stt/cohere-transcribe-q4_k": {"engine": "transcribe", "gpu": true, "languages": [], "vramMb": 1000},
                "stt/nemotron-3.5-streaming-560ms-int8": {"engine": "sherpa-online", "languages": []},
                "stt/qwen3-asr-0.6b-int8": {"engine": "sherpa-offline", "languages": []}
            },
            "routes": [
                {"name": "ja", "languages": ["ja"], "streaming": [], "cpuFinal": ["stt/qwen3-asr-0.6b-int8"], "gpuFinal": ["stt/cohere-transcribe-q4_k"]},
                {"name": "auto", "languages": ["auto"], "streaming": ["stt/nemotron-3.5-streaming-560ms-int8"], "streamFinal": true, "cpuFinal": [], "gpuFinal": ["stt/cohere-transcribe-q4_k"]},
                {"name": "other", "languages": ["*"], "streaming": ["stt/nemotron-3.5-streaming-560ms-int8"], "streamFinal": true}
            ]}}"#;
        let table = RoutingTable::from_document(bad).unwrap();
        let ctx = pc(24_576, 15_000, 8);
        let ja = plan_table(&table, &req("ja", Tier::Gpu, true), &ctx);
        assert_eq!(model(&ja.final_engine), Some(QWEN3_0_6B), "{ja:?}");
        assert_eq!(ja.tier, Tier::Cpu);
        assert!(ja.notes.iter().any(|n| n == notes::NEVER_ROUTE), "{ja:?}");
        // Auto: Cohere is never used, and Nemotron alone (no language source) is dropped.
        let auto = plan_table(&table, &req("auto", Tier::Gpu, true), &ctx);
        assert!(auto.streaming.is_none() && auto.final_engine.is_none(), "{auto:?}");
        assert!(
            auto.notes
                .iter()
                .any(|n| n == notes::NEMOTRON_NEEDS_LANGUAGE),
            "{auto:?}"
        );
        assert!(auto.notes.iter().any(|n| n == notes::NO_ENGINE), "{auto:?}");
        // An explicit language reaches Nemotron even when the table forgot `needsLanguage`.
        let de = plan_table(&table, &req("de", Tier::Cpu, true), &ctx);
        assert_eq!(de.streaming.as_ref().unwrap().options["language"], "de");
        // A sherpa model never runs a language its catalog entry lacks, whatever the table says.
        let sw = plan_table(&table, &req("sw", Tier::Cpu, true), &ctx);
        assert!(sw.streaming.is_none() && sw.final_engine.is_none(), "{sw:?}");
        assert!(sw.notes.iter().any(|n| n == notes::NEVER_ROUTE), "{sw:?}");
    }

    #[test]
    fn auto_mode_adds_language_id_for_finals_that_cannot_detect() {
        let table = RoutingTable::from_document(
            r#"{"routing":{
            "version": 1, "cpuTwoPassMinCores": 6, "vcShareGpuMinTotalMb": 15000, "gpuRuntimes": {},
            "languageId": "stt/lid-whisper-base-int8",
            "models": {
                "stt/nemotron-3.5-streaming-560ms-int8": {"engine": "sherpa-online", "languages": [], "needsLanguage": true},
                "stt/parakeet-ultra-int8": {"engine": "sherpa-offline", "languages": []},
                "stt/lid-whisper-base-int8": {"engine": "sherpa-offline", "languages": [], "detectsLanguage": true}
            },
            "routes": [
                {"name": "auto", "languages": ["auto"], "streaming": ["stt/nemotron-3.5-streaming-560ms-int8"], "cpuFinal": ["stt/parakeet-ultra-int8"]},
                {"name": "other", "languages": ["*"], "streaming": ["stt/nemotron-3.5-streaming-560ms-int8"], "cpuFinal": ["stt/parakeet-ultra-int8"]}
            ]}}"#,
        )
        .unwrap();
        let ctx = pc(0, 0, 8);
        let plan = plan_table(&table, &req("auto", Tier::Auto, true), &ctx);
        assert_eq!(model(&plan.language_id), Some(LID), "{plan:?}");
        assert_eq!(model(&plan.streaming), Some(NEMOTRON));
        let options = &plan.streaming.as_ref().unwrap().options;
        assert!(options.get("language").is_none(), "no fixed language in auto mode");
        // The table lists no languages for Nemotron: its sherpa catalog languages are the candidates,
        // and the `*` row streams with it.
        let mut catalog: Vec<&str> = sherpa_catalog::find(NEMOTRON).unwrap().languages.to_vec();
        catalog.sort_unstable();
        assert_eq!(options[STREAM_LANGUAGES_OPTION], serde_json::json!(catalog));
        assert!(plan.notes.iter().any(|n| n == notes::LANGUAGE_ID));
        // Without the language-ID engine Nemotron cannot learn the language: finals only.
        let mut no_lid = ctx.clone();
        no_lid.unavailable.insert(LID.into());
        let plan = plan_table(&table, &req("auto", Tier::Auto, true), &no_lid);
        assert!(plan.streaming.is_none() && plan.language_id.is_none(), "{plan:?}");
        assert_eq!(model(&plan.final_engine), Some(PARAKEET_ULTRA));
        assert!(plan
            .notes
            .iter()
            .any(|n| n == notes::NEMOTRON_NEEDS_LANGUAGE));
    }

    fn stream_limit(plan: &RoutePlan) -> Option<Vec<String>> {
        plan.streaming
            .as_ref()
            .and_then(|s| s.options.get(STREAM_LANGUAGES_OPTION))
            .map(|v| serde_json::from_value(v.clone()).unwrap())
    }

    /// Auto mode streams only for detected languages whose own row streams with the model: never
    /// zh / yue (SenseVoice rows, no partials) nor languages Nemotron lacks (th, id, el, ...); and
    /// the pipeline's gate (`worker::supported_language` over [`stream_languages`]) keeps a stream
    /// closed for them, so those utterances get the final alone.
    #[test]
    fn auto_streams_only_for_languages_whose_route_streams() {
        let table = t();
        for ctx in [pc(0, 0, 8), pc(12_288, 6_600, 8), pc(0, 0, 0)] {
            let plan = plan_with(&req("auto", Tier::Auto, true), &ctx);
            let spec = plan.streaming.as_ref().expect("auto streams on this PC");
            assert_eq!(spec.model, NEMOTRON);
            assert!(spec.options.get("language").is_none(), "{plan:?}");
            let limit = stream_limit(&plan).expect("an auto stream carries its languages");
            let expected: Vec<String> = table.models[NEMOTRON]
                .languages
                .iter()
                .filter(|l| !["zh", "yue"].contains(&l.as_str()))
                .cloned()
                .collect::<BTreeSet<_>>()
                .into_iter()
                .collect();
            assert_eq!(limit, expected, "{plan:?}");
            for l in ["en", "ja", "ko", "ru", "de", "ar", "vi", "nb", "uk"] {
                assert!(limit.iter().any(|x| x == l), "{l}");
            }
            // The pipeline's gate, as `StreamFeed::begin` applies it to the sticky language.
            let engine_languages: Vec<String> = sherpa_catalog::find(NEMOTRON)
                .unwrap()
                .languages
                .iter()
                .map(|l| (*l).to_string())
                .collect();
            let gate = stream_languages(spec, &engine_languages);
            assert_eq!(gate, limit);
            for l in ["zh", "yue", "th", "id", "fa", "el", "sw"] {
                assert!(!limit.iter().any(|x| x == l), "{l}: {limit:?}");
                let detected = Some(l.to_string());
                assert_eq!(crate::services::speech::worker::supported_language(&gate, detected), None, "{l}");
            }
            assert_eq!(
                crate::services::speech::worker::supported_language(&gate, Some("ja".into())).as_deref(),
                Some("ja")
            );
            // Without the plan's limit the engine alone would take the sticky zh.
            assert_eq!(
                crate::services::speech::worker::supported_language(&engine_languages, Some("zh".into())).as_deref(),
                Some("zh")
            );
            // Every listed language streams with Nemotron when chosen explicitly (where the PC runs
            // two passes, as it does here).
            for l in &limit {
                let explicit = plan_with(&req(l, Tier::Auto, true), &ctx);
                assert_eq!(model(&explicit.streaming), Some(NEMOTRON), "{l}: {explicit:?}");
            }
        }
        // Explicit languages carry `language`, never a limit.
        let ja = plan_with(&req("ja", Tier::Auto, true), &pc(0, 0, 8));
        assert_eq!(stream_limit(&ja), None);
    }

    #[test]
    fn stream_languages_narrow_the_engine_list() {
        let engine: Vec<String> = ["en", "ja", "zh"].map(String::from).to_vec();
        let mut spec = EngineSpec::new(EngineKind::SherpaOnline, NEMOTRON);
        // No limit: the engine's own list.
        assert_eq!(stream_languages(&spec, &engine), engine);
        spec.options = serde_json::json!({ "language": "ja" });
        assert_eq!(stream_languages(&spec, &engine), engine);
        // A limit narrows it (normalized, sorted).
        spec.options = serde_json::json!({ STREAM_LANGUAGES_OPTION: ["ja", "en-US", "ko"] });
        assert_eq!(stream_languages(&spec, &engine), ["en", "ja"]);
        // An engine without a list takes the limit.
        assert_eq!(stream_languages(&spec, &[]), ["en", "ja", "ko"]);
        // A limit is never widened to "any" (an empty list would mean any language).
        spec.options = serde_json::json!({ STREAM_LANGUAGES_OPTION: ["ko"] });
        assert_eq!(stream_languages(&spec, &engine), ["ko"]);
        spec.options = serde_json::json!({ STREAM_LANGUAGES_OPTION: [] });
        let none = stream_languages(&spec, &engine);
        assert!(!none.is_empty());
        for l in ["en", "ja", "zh"] {
            assert_eq!(crate::services::speech::worker::supported_language(&none, Some(l.into())), None);
        }
    }

    /// A table whose explicit rows never stream with the auto stream model drops the stream in
    /// auto mode; a model without any language list that the `*` row streams with runs unlimited.
    #[test]
    fn auto_stream_limit_follows_the_table() {
        let doc = |other_streams: bool| {
            format!(
                r#"{{"routing":{{
                "version": 1, "cpuTwoPassMinCores": 6, "vcShareGpuMinTotalMb": 15000, "gpuRuntimes": {{}},
                "models": {{
                    "stt/stream-x": {{"engine": "sherpa-online", "languages": [], "needsLanguage": true}},
                    "stt/qwen3-asr-0.6b-int8": {{"engine": "sherpa-offline", "languages": [], "detectsLanguage": true}}
                }},
                "routes": [
                    {{"name": "ja", "languages": ["ja"], "streaming": [], "cpuFinal": ["stt/qwen3-asr-0.6b-int8"]}},
                    {{"name": "auto", "languages": ["auto"], "streaming": ["stt/stream-x"], "cpuFinal": ["stt/qwen3-asr-0.6b-int8"]}},
                    {{"name": "other", "languages": ["*"], "streaming": [{}], "cpuFinal": ["stt/qwen3-asr-0.6b-int8"]}}
                ]}}}}"#,
                if other_streams { r#""stt/stream-x""# } else { "" }
            )
        };
        let ctx = pc(0, 0, 8);
        let closed = RoutingTable::from_document(&doc(false)).unwrap();
        let plan = plan_table(&closed, &req("auto", Tier::Auto, true), &ctx);
        assert!(plan.streaming.is_none(), "{plan:?}");
        assert_eq!(model(&plan.final_engine), Some(QWEN3_0_6B));
        assert!(plan
            .notes
            .iter()
            .any(|n| n == notes::NEMOTRON_NEEDS_LANGUAGE));
        let open = RoutingTable::from_document(&doc(true)).unwrap();
        let plan = plan_table(&open, &req("auto", Tier::Auto, true), &ctx);
        assert_eq!(model(&plan.streaming), Some("stt/stream-x"), "{plan:?}");
        assert!(plan.streaming.as_ref().unwrap().options.is_null(), "no limit: {plan:?}");
    }

    /// A GPU final next to a CPU stream gets its CPU fallback only where the CPU may run two
    /// engines (the CPU two-pass rule); a finals-only GPU plan always keeps it.
    #[test]
    fn gpu_two_pass_has_a_cpu_fallback_only_where_the_cpu_runs_two_engines() {
        let fallback_of = |ctx: &RouteContext, l: &str| {
            let plan = plan_with(&req(l, Tier::Auto, true), ctx);
            assert_eq!(plan.tier, Tier::Gpu, "{plan:?}");
            plan
        };
        // 8 cores, no voice changer: two-pass with the fallback.
        let plan = fallback_of(&pc(12_288, 6_600, 8), "ja");
        assert_eq!((model(&plan.streaming), model(&plan.fallback)), (Some(NEMOTRON), Some(QWEN3_0_6B)));
        assert!(!plan.notes.iter().any(|n| n == notes::NO_CPU_FALLBACK));
        // 4 cores, a voice changer on the CPU, or one whose device is not known yet: no fallback.
        let mut vc_cpu = pc(12_288, 6_600, 8);
        vc_cpu.vc_running = true;
        vc_cpu.vc_gpu = Some(false);
        let mut vc_unknown = pc(24_576, 15_000, 16);
        vc_unknown.vc_running = true;
        for ctx in [pc(12_288, 6_600, 4), vc_cpu, vc_unknown] {
            let plan = fallback_of(&ctx, "ja");
            assert_eq!(model(&plan.streaming), Some(NEMOTRON), "{plan:?}");
            assert_eq!(model(&plan.fallback), None, "{plan:?}");
            assert!(!plan.models.iter().any(|m| m == QWEN3_0_6B), "not downloaded: {plan:?}");
            assert!(plan.notes.iter().any(|n| n == notes::NO_CPU_FALLBACK), "{plan:?}");
            // Finals-only GPU plans keep it (zh has no stream; single pass has none).
            let zh = fallback_of(&ctx, "zh");
            assert_eq!((model(&zh.streaming), model(&zh.fallback)), (None, Some(SENSE_VOICE)), "{zh:?}");
            let single = plan_with(&req("ja", Tier::Auto, false), &ctx);
            assert_eq!(model(&single.fallback), Some(QWEN3_0_6B), "{single:?}");
        }
        // A voice changer on the GPU does not load the CPU.
        let mut vc_gpu = pc(24_576, 15_000, 16);
        vc_gpu.vc_running = true;
        vc_gpu.vc_gpu = Some(true);
        assert_eq!(model(&fallback_of(&vc_gpu, "ja").fallback), Some(QWEN3_0_6B));
    }

    /// A running voice changer whose device is not known yet counts as on the CPU for the CPU
    /// two-pass rule (and as on the GPU for the GPU rule): CPU routes run one pass.
    #[test]
    fn an_unknown_voice_changer_device_keeps_cpu_routes_single_pass() {
        let mut ctx = pc(12_288, 6_600, 8);
        ctx.vc_running = true;
        let ja = plan_with(&req("ja", Tier::Auto, true), &ctx);
        assert_eq!(ja.tier, Tier::Cpu, "{ja:?}");
        assert_eq!((model(&ja.streaming), model(&ja.final_engine)), (None, Some(QWEN3_0_6B)), "{ja:?}");
        for n in [notes::GPU_DENIED_VC_RUNNING, notes::VC_UNKNOWN_SINGLE_PASS, notes::FINALS_ONLY] {
            assert!(ja.notes.iter().any(|x| x == n), "{n}: {ja:?}");
        }
        // English keeps Nemotron's own finals (caption quality) as the one pass.
        let en = plan_with(&req("en", Tier::Auto, true), &ctx);
        assert_eq!((model(&en.streaming), model(&en.final_engine)), (Some(NEMOTRON), None), "{en:?}");
        // Once it reports the GPU (which it keeps on this card) the CPU runs two passes.
        ctx.vc_gpu = Some(true);
        let ja = plan_with(&req("ja", Tier::Auto, true), &ctx);
        assert_eq!(ja.tier, Tier::Cpu, "{ja:?}");
        assert_eq!(
            (model(&ja.streaming), model(&ja.final_engine)),
            (Some(NEMOTRON), Some(QWEN3_0_6B)),
            "{ja:?}"
        );
        // On a CPU-only PC the note says which case it is.
        let mut cpu = pc(0, 0, 8);
        cpu.vc_running = true;
        for (device, note, other) in [
            (Some(false), notes::VC_CPU_SINGLE_PASS, notes::VC_UNKNOWN_SINGLE_PASS),
            (None, notes::VC_UNKNOWN_SINGLE_PASS, notes::VC_CPU_SINGLE_PASS),
        ] {
            cpu.vc_gpu = device;
            let ja = plan_with(&req("ja", Tier::Auto, true), &cpu);
            assert_eq!(model(&ja.streaming), None, "{ja:?}");
            assert!(ja.notes.iter().any(|x| x == note) && !ja.notes.iter().any(|x| x == other), "{ja:?}");
        }
    }

    /// `interim: false` (no live partial captions): a two-pass plan becomes its final engine alone
    /// (the streaming model is not planned, so it is neither a requirement nor loaded or run);
    /// every other plan, including a stream that is the only engine, is exactly the plan
    /// `interim: true` gives.
    #[test]
    fn interim_off_drops_only_the_stream_that_runs_next_to_a_final() {
        // English on an 8-core CPU: Nemotron + Parakeet Ultra becomes Parakeet Ultra alone.
        let cpu = pc(0, 0, 8);
        let on = plan_with(&req("en", Tier::Auto, true), &cpu);
        assert_eq!((model(&on.streaming), model(&on.final_engine)), (Some(NEMOTRON), Some(PARAKEET_ULTRA)), "{on:?}");
        let off = plan_with(&req_quiet("en", Tier::Auto, true), &cpu);
        assert_eq!((model(&off.streaming), model(&off.final_engine)), (None, Some(PARAKEET_ULTRA)), "{off:?}");
        assert!(off.notes.iter().any(|n| n == notes::FINALS_ONLY), "{off:?}");
        assert!(!plan_models(&off).contains(&NEMOTRON), "the stream is not a requirement: {off:?}");
        assert!(!off.missing.iter().any(|m| m == NEMOTRON), "{off:?}");
        let mut nothing_installed = pc(0, 0, 8);
        nothing_installed.installed.clear();
        let fresh = plan_with(&req_quiet("en", Tier::Auto, true), &nothing_installed);
        assert!(!fresh.models.iter().any(|m| m == NEMOTRON) && !fresh.missing.iter().any(|m| m == NEMOTRON), "nothing to download for it: {fresh:?}");
        assert!(plan_with(&req("en", Tier::Auto, true), &nothing_installed).missing.iter().any(|m| m == NEMOTRON));

        // Auto language keeps its language ID / detecting final; only the stream goes.
        let auto_on = plan_with(&req("auto", Tier::Auto, true), &cpu);
        let auto_off = plan_with(&req_quiet("auto", Tier::Auto, true), &cpu);
        assert_eq!(model(&auto_on.streaming), Some(NEMOTRON), "{auto_on:?}");
        assert_eq!(model(&auto_off.streaming), None, "{auto_off:?}");
        assert_eq!(auto_off.final_engine, auto_on.final_engine);
        assert_eq!(auto_off.language_id, auto_on.language_id);

        // A GPU final stays on the GPU and keeps its CPU fallback (a finals-only GPU plan always has
        // one), even where the CPU could not have run the stream next to it.
        let gpu = pc(12_288, 6_600, 4);
        let on = plan_with(&req("ja", Tier::Auto, true), &gpu);
        assert_eq!((on.tier, model(&on.streaming), model(&on.fallback)), (Tier::Gpu, Some(NEMOTRON), None), "{on:?}");
        let off = plan_with(&req_quiet("ja", Tier::Auto, true), &gpu);
        assert_eq!((off.tier, model(&off.streaming), model(&off.fallback)), (Tier::Gpu, None, Some(QWEN3_0_6B)), "{off:?}");
        assert_eq!(off.final_engine, on.final_engine);
        assert_eq!(off.vram_mb, on.vram_mb);

        // A tier that offers only a streaming engine keeps it: its finals are the captions.
        for l in ["nb", "ar", "hi", "tr"] {
            let cpu_only = plan_with(&req_quiet(l, Tier::Cpu, true), &cpu);
            assert_eq!((model(&cpu_only.streaming), model(&cpu_only.final_engine)), (Some(NEMOTRON), None), "{l}: {cpu_only:?}");
            assert_eq!(cpu_only, plan_with(&req(l, Tier::Cpu, true), &cpu), "{l}: identical to interim on");
        }
        // ...while the same language with a GPU final available drops it.
        let ar = plan_with(&req_quiet("ar", Tier::Auto, true), &pc(12_288, 6_600, 8));
        assert_eq!((ar.tier, model(&ar.streaming)), (Tier::Gpu, None), "{ar:?}");

        // One-pass choices are untouched: a 4-core PC streams English alone with or without interim.
        let few = pc(0, 0, 4);
        let few_off = plan_with(&req_quiet("en", Tier::Auto, true), &few);
        assert_eq!((model(&few_off.streaming), model(&few_off.final_engine)), (Some(NEMOTRON), None), "{few_off:?}");
        assert_eq!(few_off, plan_with(&req("en", Tier::Auto, true), &few));
    }

    /// The same rule over every language × tier × pass setting × PC (the PCs of
    /// `every_request_on_every_pc_keeps_the_rules`, trimmed): a plan with interim off is never
    /// two-pass; it is the interim-on plan minus the stream when that was two-pass, else equal.
    #[test]
    fn interim_off_matches_interim_on_everywhere_but_the_dropped_stream() {
        let table = t();
        let mut languages: Vec<String> = table
            .routes
            .iter()
            .flat_map(|r| r.languages.iter().cloned())
            .filter(|l| l != "*")
            .collect();
        languages.extend(["sw", "he", "zh-HK", "pt-BR", "", "xx-invalid-1"].map(String::from));
        let mut cpu_vc = pc(0, 0, 8);
        cpu_vc.vc_running = true;
        cpu_vc.vc_gpu = Some(false);
        let mut gpu_vc = pc(12_288, 5_400, 8);
        gpu_vc.vc_running = true;
        gpu_vc.vc_gpu = Some(true);
        let mut vc_unknown = pc(24_576, 15_000, 16);
        vc_unknown.vc_running = true;
        let mut nothing_installed = pc(24_576, 15_000, 16);
        nothing_installed.installed.clear();
        let pcs = [
            ("cpu 8 cores", pc(0, 0, 8)),
            ("cpu 4 cores", pc(0, 0, 4)),
            ("cpu unknown cores", pc(0, 0, 0)),
            ("12 GB", pc(12_288, 6_600, 8)),
            ("16 GB", pc(16_376, 11_400, 8)),
            ("24 GB", pc(24_576, 15_000, 16)),
            ("vc on cpu", cpu_vc),
            ("vc on gpu", gpu_vc),
            ("vc device unknown", vc_unknown),
            ("nothing installed", nothing_installed),
        ];
        let mut dropped = 0;
        for (pc_name, ctx) in &pcs {
            for l in &languages {
                for tier in [Tier::Auto, Tier::Cpu, Tier::Gpu] {
                    for two_pass in [true, false] {
                        let what = format!("{pc_name} / {l:?} / {tier:?} / two-pass {two_pass}");
                        let on = plan_with(&req(l, tier, two_pass), ctx);
                        let off = plan_with(&req_quiet(l, tier, two_pass), ctx);
                        assert!(off.streaming.is_some() || off.final_engine.is_some(), "an engine: {what}: {off:?}");
                        assert!(off.streaming.is_none() || off.final_engine.is_none(), "never two-pass: {what}: {off:?}");
                        if let (Some(stream), Some(_)) = (&on.streaming, &on.final_engine) {
                            dropped += 1;
                            assert!(off.streaming.is_none(), "{what}: {off:?}");
                            assert_eq!(off.final_engine, on.final_engine, "{what}");
                            assert_eq!((off.tier, off.vram_mb, &off.language_id), (on.tier, on.vram_mb, &on.language_id), "{what}");
                            assert!(on.fallback.is_none() || off.fallback == on.fallback, "{what}: {off:?}");
                            assert!(off.notes.iter().any(|n| n == notes::FINALS_ONLY), "{what}: {:?}", off.notes);
                            assert!(!off.models.contains(&stream.model), "{what}: {off:?}");
                            assert!(!off.missing.contains(&stream.model), "{what}: {off:?}");
                            // Nothing new to download, except the CPU fallback a finals-only GPU plan always keeps.
                            if on.fallback.is_some() || off.fallback.is_none() {
                                assert!(off.models.iter().all(|m| on.models.contains(m)), "nothing new to download: {what}");
                            }
                        } else {
                            assert_eq!(off, on, "{what}");
                        }
                    }
                }
            }
        }
        assert!(dropped > 100, "the sweep covers many two-pass plans ({dropped})");
    }

    #[test]
    fn plans_list_runtimes_and_missing_ids_in_order() {
        let mut ctx = pc(12_288, 6_600, 8);
        ctx.installed = [NEMOTRON, "runtime/sherpa-onnx"].map(String::from).into();
        let plan = plan_with(&req("ja", Tier::Gpu, true), &ctx);
        assert_eq!(
            plan.models,
            [
                NEMOTRON,
                "runtime/sherpa-onnx",
                QWEN3_Q8,
                "runtime/sigil-transcribe-cuda13",
                "runtime/nvidia-cuda13",
                QWEN3_0_6B
            ]
        );
        assert_eq!(
            plan.missing,
            [QWEN3_Q8, "runtime/sigil-transcribe-cuda13", "runtime/nvidia-cuda13", QWEN3_0_6B]
        );
        assert_eq!(plan.vram_mb, 3500);
        let f = plan.final_engine.as_ref().unwrap();
        assert_eq!((f.engine, f.variant.as_deref()), (EngineKind::Transcribe, Some("cuda13")));
        assert!(f.options.is_null(), "transcribe specs get the language through DecodeOptions");
        let fb = plan.fallback.as_ref().unwrap();
        assert_eq!((fb.engine, fb.model.as_str()), (EngineKind::SherpaOffline, QWEN3_0_6B));
        assert_eq!(fb.options["language"], "ja");
        // The same request on 16 GB takes BF16.
        let big = plan_with(&req("ja", Tier::Gpu, true), &pc(16_376, 11_400, 8));
        assert_eq!(model(&big.final_engine), Some(QWEN3_BF16));
        // A GPU model that the catalog does not have yet is skipped for the next one.
        let mut no_cohere = pc(12_288, 6_600, 8);
        no_cohere.unavailable.insert(COHERE.into());
        let en = plan_with(&req("en", Tier::Auto, true), &no_cohere);
        assert_eq!(model(&en.final_engine), Some(QWEN3_Q8));
    }

    #[test]
    fn availability_follows_the_catalog_and_installs() {
        let ids: BTreeSet<String> = ["a/published", "a/unpublished", "a/imported", "a/unknown"]
            .map(String::from)
            .into();
        let catalog: BTreeMap<String, PublishStatus> = [
            ("a/published", PublishStatus::Published),
            ("a/unpublished", PublishStatus::Unpublished),
            ("a/imported", PublishStatus::Unpublished),
        ]
        .into_iter()
        .map(|(k, v)| (k.to_string(), v))
        .collect();
        let (installed, unavailable) = availability(&ids, Some(&catalog), |id| id == "a/imported", |_| false);
        assert_eq!(installed, ["a/imported"].map(String::from).into());
        assert_eq!(unavailable, ["a/unknown", "a/unpublished"].map(String::from).into());
        // An unpublished pack in the local-assets folder can be installed from there; a local file
        // never makes an id the catalog lacks available.
        let local = |id: &str| id == "a/unpublished" || id == "a/unknown";
        let (installed, unavailable) = availability(&ids, Some(&catalog), |id| id == "a/imported", local);
        assert_eq!(installed, ["a/imported"].map(String::from).into());
        assert_eq!(unavailable, ["a/unknown"].map(String::from).into());
        // Without a catalog only installed ids can be used.
        let (_, unavailable) = availability(&ids, None, |id| id == "a/imported", local);
        assert_eq!(unavailable.len(), 3);
    }

    /// The router looks for an unpublished pack where the model store's install does
    /// (`<root>/<id>/<path>`, `<root>/<path>`, `<root>/<source file name>`), and every file must
    /// be there.
    #[test]
    fn local_assets_are_found_where_the_model_store_looks() {
        let entry: CatalogEntry = serde_json::from_value(serde_json::json!({
            "id": "runtime/sigil-test-pack",
            "family": "runtime",
            "kind": "runtime",
            "version": "1",
            "license": "MIT",
            "status": "unpublished",
            "files": [
                {"path": "pack-1-cuda13.zip", "size": 0, "sources": [
                    {"kind": "github-release", "url": "https://example.invalid/releases/download/a/released-name.zip"}
                ]},
                {"path": "bin/extra.dll", "size": 0, "sources": []}
            ]
        }))
        .unwrap();
        let root = std::env::temp_dir().join(format!("sigil-router-local-assets-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&root);
        let put = |rel: &str| {
            let p = root.join(rel);
            std::fs::create_dir_all(p.parent().unwrap()).unwrap();
            std::fs::write(&p, b"x").unwrap();
        };
        std::fs::create_dir_all(&root).unwrap();
        assert!(!has_local_assets(&root, &entry), "an empty folder");
        put("runtime/sigil-test-pack/pack-1-cuda13.zip");
        assert!(!has_local_assets(&root, &entry), "every file is needed");
        put("bin/extra.dll");
        assert!(has_local_assets(&root, &entry), "<root>/<id>/<path> and <root>/<path>");
        std::fs::remove_file(root.join("runtime/sigil-test-pack/pack-1-cuda13.zip")).unwrap();
        put("released-name.zip");
        assert!(has_local_assets(&root, &entry), "the source file name");
        let _ = std::fs::remove_dir_all(&root);
        // Paths that could leave the folder are never looked up.
        let base = Path::new("C:/assets");
        assert_eq!(join_catalog_path(base, "a/b.zip"), Some(base.join("a").join("b.zip")));
        for bad in ["../x.zip", "a/../../x", "a//b", "", "a/./b", "C:/x", r"a\b", "a b"] {
            assert_eq!(join_catalog_path(base, bad), None, "{bad:?}");
        }
    }

    #[test]
    fn live_context_follows_the_system_probe_and_the_governor() {
        let nvidia = system(vec![
            gpu(0, VENDOR_NVIDIA, "NVIDIA GeForce RTX 4060", 8_188, Some("8.9"), Some("ada")),
            gpu(1, VENDOR_MICROSOFT, "Microsoft Basic Render Driver", 0, None, None),
        ]);
        let snap = ResourceSnapshot::default();
        let ctx = context_from(&nvidia, &snap, BTreeSet::new(), BTreeSet::new());
        assert!(
            ctx.gpu && ctx.gpu_variant.as_deref() == Some("cuda13") && ctx.gpu_reason.is_none(),
            "{ctx:?}"
        );
        assert_eq!(ctx.vram_total_mb, 8_188);
        assert_eq!(ctx.vram_available_mb, 8_188 - policy::reserve_mb(8_188));
        assert_eq!(ctx.physical_cores, 8);
        // The voice changer on the GPU takes its estimate from the room.
        let vc = ResourceSnapshot {
            vc_running: true,
            vc_gpu: Some(true),
            ..ResourceSnapshot::default()
        };
        let busy = context_from(&nvidia, &vc, BTreeSet::new(), BTreeSet::new());
        assert!(busy.vc_running && busy.vc_gpu == Some(true));
        assert_eq!(busy.vram_available_mb, ctx.vram_available_mb - policy::VC_ESTIMATE_MB);
        // AMD: no transcribe.cpp GPU build yet (vulkan unbuilt) → CPU, with the governor's reason.
        let amd = system(vec![gpu(0, VENDOR_AMD, "AMD Radeon RX 7800 XT", 16_368, None, Some("rdna3"))]);
        let ctx = context_from(&amd, &snap, BTreeSet::new(), BTreeSet::new());
        assert!(!ctx.gpu && ctx.gpu_variant.is_none());
        assert_eq!(ctx.gpu_reason.as_deref(), Some("no_gpu_runtime"));
        let cpu = system(vec![gpu(0, VENDOR_MICROSOFT, "Microsoft Basic Render Driver", 0, None, None)]);
        assert_eq!(
            context_from(&cpu, &snap, BTreeSet::new(), BTreeSet::new())
                .gpu_reason
                .as_deref(),
            Some("no_gpu")
        );
        // A voice changer that is not running reports no device.
        let stale = ResourceSnapshot {
            vc_running: false,
            vc_gpu: Some(true),
            ..ResourceSnapshot::default()
        };
        assert_eq!(context_from(&nvidia, &stale, BTreeSet::new(), BTreeSet::new()).vc_gpu, None);
    }

    /// Every note code has English UI text: `stt.route_note_<code>` in the English locale (a new
    /// code may sit in `scripts/i18n-staging/<WS>.json` until its phase merges); governor reasons
    /// use the existing `models.gpu.rec_reason_<code>`.
    #[test]
    fn every_note_code_has_ui_text() {
        let root = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("..");
        let read = |rel: &str| -> Option<serde_json::Value> {
            let text = std::fs::read_to_string(root.join(rel)).ok()?;
            Some(serde_json::from_str(&text).unwrap_or_else(|e| panic!("{rel}: {e}")))
        };
        let en = read("public/i18n/en/translation.json").expect("the English locale");
        let staging_dir = root.join("scripts/i18n-staging");
        let staged: Vec<serde_json::Value> = std::fs::read_dir(&staging_dir)
            .map(|dir| {
                dir.flatten()
                    .map(|e| e.path())
                    .filter(|p| p.extension().is_some_and(|x| x == "json"))
                    .filter_map(|p| read(&p.strip_prefix(&root).unwrap_or(&p).to_string_lossy()))
                    .collect()
            })
            .unwrap_or_default();
        let has = |doc: &serde_json::Value, path: &[&str]| {
            path.iter()
                .try_fold(doc, |v, k| v.get(*k))
                .and_then(|v| v.as_str())
                .is_some_and(|s| !s.trim().is_empty())
        };
        for code in notes::ALL {
            let key = format!("route_note_{code}");
            let path = ["stt", key.as_str()];
            assert!(
                has(&en, &path) || staged.iter().any(|s| has(s, &path)),
                "no UI text for the route note {code} (stt.{key})"
            );
        }
        for code in notes::GOVERNOR_REASONS {
            let key = format!("rec_reason_{code}");
            assert!(has(&en, &["models", "gpu", key.as_str()]), "no UI text for models.gpu.{key}");
        }
    }

    #[test]
    fn context_wire_format_is_camel_case_and_lenient() {
        let ctx: RouteContext = serde_json::from_str(
            r#"{"gpu":true,"gpuVariant":"cuda13","vramAvailableMb":5000,"vramTotalMb":12288,"vcGpu":true,"unavailable":["x/y"]}"#,
        )
        .unwrap();
        assert!(ctx.gpu && ctx.vc_gpu == Some(true) && ctx.unavailable.contains("x/y") && ctx.installed.is_empty());
        let v = serde_json::to_value(RouteContext::default()).unwrap();
        for key in [
            "gpu",
            "gpuVariant",
            "vramAvailableMb",
            "vcRunning",
            "physicalCores",
            "installed",
            "vramTotalMb",
            "vcGpu",
            "unavailable",
            "gpuReason",
        ] {
            assert!(v.get(key).is_some(), "{key}");
        }
    }
}
