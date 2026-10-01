//! Adding a so-vits-svc voice (P5-D): `voice_changer_create_family_slot` with `family: "sovits"` and
//! `source` = the user's export.
//!
//! Accepted sources:
//! - an `.onnx` export (so-vits-svc 4.0 / 4.1 `onnx_export.py`, MoeVS format), with the files that
//!   travel with it picked up from its folder: the training `config.json`, the MoeVS JSON
//!   (`checkpoints/<name>/<name>.json` next to `<name>_SoVits.onnx`, as so-vits-svc 4.1's
//!   `onnx_export.py` writes it, or `<name>.json` one level up, MoeVS's `Mods/` layout), and a FAISS
//!   feature index (`*.index`) for `cluster_ratio`. The folder may hold other voices' files (a
//!   Downloads folder), so an index is adopted only when its name or the folder's ties it to the
//!   export, and the import says which index and which generic `config.json` it took;
//! - a folder holding the same files (every index and config in it is the voice's);
//! - a `.zip` of them (read with the importer's zip guard: entry count, sizes, ratio, names).
//!
//! Operating-system metadata (macOS `__MACOSX/` and `._name` AppleDouble twins, hidden files,
//! `Thumbs.db`) is never taken for a voice file.
//!
//! The graph is opened on the CPU and resolved (`graph::SovitsGraph`), then run once on a short
//! probe: that measures its hop (output samples per frame; the config's hop must agree, and without
//! a config the rate follows the hop, `config::rate_for_hop`) and proves it runs and stays finite.
//! The slot gets `sovits.onnx`, the training `config.json` (kept for reference), the feature index
//! (`feature.index`), the record `sovits.json` and the manifest. Training checkpoints (`G_*.pth`),
//! kmeans cluster models (`kmeans_*.pt`), so-vits-svc's pickled feature index
//! (`feature_and_index.pkl`) and so-vits-svc 3.0 exports are refused or skipped by name. An ONNX
//! file is not a pickle, so nothing here needs the isolated import worker.

use std::fs;
use std::io::Read;
use std::path::{Path, PathBuf};
use std::sync::atomic::Ordering;

use ort::session::{Session, SessionInputValue};
use ort::value::TensorRef;

use crate::services::neural_audio::ort_session::{build_session, ModelSource, VcSessionOptions};
use crate::services::neural_audio::vc::engine::content::EmbedderKind;
use crate::services::neural_audio::vc::faiss::FaissIndex;
use crate::services::neural_audio::vc::import::zipguard::{looks_like_zip, GuardedZip, ZipLimits};
use crate::services::neural_audio::vc::import::{is_metadata_entry, sha256_file, write_atomic, ErrorCode, ImportError, MAX_INDEX_BYTES};
use crate::services::neural_audio::vc::manifest::RvcVersion;
use crate::services::neural_audio::vc::rvc::non_onnx_file_problem;

use super::super::{family_manifest, write_family_manifest, CreatedSlot, FamilyKind, FamilySlotContext, FamilySlotRequest};
use super::config::{encoder_for, rate_for_hop, SovitsConfig, SovitsRecord, MAX_CONFIG_BYTES, RECORD_FILE, RECORD_FORMAT, RECORD_VERSION};
use super::graph::{Role, SovitsGraph, SpeakerInput};
use super::model::{float_input, to_half};

/// The generator's file name in the slot.
pub const GENERATOR_FILE: &str = "sovits.onnx";
/// The feature index's file name in the slot.
pub const INDEX_FILE: &str = "feature.index";
/// The training config kept in the slot.
pub const CONFIG_FILE: &str = "config.json";
/// Largest so-vits-svc export taken (4.1 44.1 kHz exports are ~200 MB; fp32 with extras < 1 GB).
const MAX_GENERATOR_BYTES: u64 = 2 << 30;
/// Model frames of the import probe.
const PROBE_FRAMES: usize = 32;

fn err(code: ErrorCode, msg: impl Into<String>) -> String {
    ImportError::new(code, msg).to_string()
}

fn lower_ext(name: &str) -> String {
    Path::new(name).extension().map(|e| e.to_string_lossy().to_ascii_lowercase()).unwrap_or_default()
}

fn base_name(name: &str) -> &str {
    name.rsplit(['/', '\\']).next().unwrap_or(name)
}

/// Where a file of the voice comes from.
enum Source {
    File(PathBuf),
    /// An entry of the archive.
    Entry(String),
}

/// The voice's files, from a file, a folder or an archive.
struct Bundle {
    generator: Source,
    generator_name: String,
    /// `(label, bytes)` of every candidate config (JSON) file.
    configs: Vec<(String, Vec<u8>)>,
    index: Option<Source>,
    /// Files seen and not taken (warnings).
    skipped: Vec<String>,
    zip: Option<GuardedZip<fs::File>>,
}

/// The so-vits-svc export among `onnx` names: the only one, else the one named like an export
/// (`*_SoVits.onnx`, `*sovits*`), else ambiguous.
fn choose_generator(onnx: &[String]) -> Result<String, String> {
    match onnx {
        [] => Err(err(ErrorCode::NotACheckpoint, "no .onnx file found; export the voice with so-vits-svc's onnx_export.py first")),
        [one] => Ok(one.clone()),
        many => {
            let named: Vec<&String> = many.iter().filter(|n| base_name(n).to_ascii_lowercase().contains("sovits")).collect();
            match named.as_slice() {
                [one] => Ok((*one).clone()),
                _ => Err(err(
                    ErrorCode::AmbiguousArchive,
                    format!("several .onnx files ({}); add the so-vits-svc export itself", many.iter().map(|n| base_name(n)).collect::<Vec<_>>().join(", ")),
                )),
            }
        }
    }
}

/// The `.index` to adopt among `names` (feature retrieval): one, else one named `feature*` / `added*`.
fn choose_index(names: &[String]) -> Option<String> {
    match names {
        [] => None,
        [one] => Some(one.clone()),
        many => many
            .iter()
            .find(|n| {
                let b = base_name(n).to_ascii_lowercase();
                b.starts_with("feature") || b.starts_with("added")
            })
            .cloned(),
    }
}

/// Why a checkpoint-like file is not taken (training checkpoints, kmeans cluster models, so-vits-svc
/// 4.1's pickled feature index).
fn torch_file_note(name: &str) -> String {
    let b = base_name(name).to_ascii_lowercase();
    if b.starts_with("kmeans") || b.contains("cluster") {
        format!("{} is a kmeans cluster model (a PyTorch pickle), which Sigil does not read; feature retrieval uses a FAISS .index instead", base_name(name))
    } else if b.starts_with("feature_and_index") || b.ends_with(".pkl") {
        format!(
            "{} is so-vits-svc's feature-retrieval index saved as a Python pickle, which Sigil does not read; feature retrieval uses a FAISS .index (RVC's train_index writes one) instead",
            base_name(name)
        )
    } else {
        format!("{} is a PyTorch file and was skipped", base_name(name))
    }
}

/// The speakers a voice has: the config's `spk` names (MoeVS `Characters`) when it has them, else
/// its `n_speakers`. so-vits-svc 4.0 configs keep the template's `n_speakers: 200` (the speaker
/// table's size) for a voice of one speaker, so the names are what the voice really has.
fn speaker_count(config: &SovitsConfig) -> usize {
    if config.speakers.is_empty() {
        config.n_speakers.unwrap_or(0) as usize
    } else {
        config.speakers.len()
    }
}

/// An ONNX Runtime load error with what usually causes it on the import's CPU check: an fp16 export
/// (ONNX Runtime's CPU has no fp16 kernels for most of a generator's operators).
fn load_error(file: &str, e: &str) -> String {
    let lower = e.to_ascii_lowercase();
    if lower.contains("could not find an implementation") || (lower.contains("float16") && lower.contains("not implemented")) {
        err(
            ErrorCode::Session,
            format!("{file} does not load on the CPU ({e}); it is probably an fp16 export, which this check cannot run: export the voice in fp32 (so-vits-svc's onnx_export.py does)"),
        )
    } else {
        err(ErrorCode::Session, format!("{file} does not load: {e}"))
    }
}

fn read_config_file(path: &Path) -> Option<Vec<u8>> {
    let meta = fs::metadata(path).ok()?;
    if !meta.is_file() || meta.len() > MAX_CONFIG_BYTES {
        return None;
    }
    fs::read(path).ok()
}

/// Whether a file on disk is operating-system metadata, never a voice file (macOS `._name`
/// AppleDouble twins and other hidden files, `Thumbs.db`, `desktop.ini`; the importer's rule).
fn is_metadata_file(path: &Path) -> bool {
    path.file_name().is_some_and(|n| is_metadata_entry(&n.to_string_lossy()))
}

/// The files of `dir` (not recursive), without operating-system metadata, sorted.
fn voice_files(dir: &Path) -> std::io::Result<Vec<PathBuf>> {
    let mut files: Vec<PathBuf> = fs::read_dir(dir)?
        .flatten()
        .map(|e| e.path())
        .filter(|p| p.is_file() && !is_metadata_file(p))
        .collect();
    files.sort();
    Ok(files)
}

/// Exporter names that say nothing about the voice (so-vits-svc 4.0's `onnx_export.py` writes
/// `model.onnx`).
const GENERIC_STEMS: [&str; 6] = ["model", "sovits", "export", "onnx", "generator", "g"];

/// `s` (lower case) trimmed of separators, when it can name a voice: at least 3 characters and not
/// a generic exporter name.
fn usable_key(s: &str) -> Option<String> {
    let s = s.trim_matches(['_', '-', ' ', '.']).to_ascii_lowercase();
    (s.chars().count() >= 3 && !GENERIC_STEMS.contains(&s.as_str())).then_some(s)
}

/// The export's own name (`nyaru` for `nyaru_SoVits.onnx`), lower case; `None` for a generic name.
fn name_key(stem: &str) -> Option<String> {
    let lower = stem.to_ascii_lowercase();
    usable_key(lower.strip_suffix("sovits").unwrap_or(&lower))
}

/// The lower-case name of `dir` (the export's folder).
fn folder_name(dir: &Path) -> String {
    dir.file_name().map(|n| n.to_string_lossy().to_ascii_lowercase()).unwrap_or_default()
}

/// Whether `dir` is the voice's own folder by name: it and the export's own name contain one
/// another (`checkpoints/nyaru/nyaru_SoVits.onnx`, `Nyaru voice/nyaru_SoVits.onnx`), never for a
/// generic export name.
fn own_folder(stem: &str, dir: &Path) -> bool {
    let folder = folder_name(dir);
    name_key(stem).is_some_and(|k| folder.contains(&k) || (usable_key(&folder).is_some() && k.contains(folder.as_str())))
}

/// The feature index to adopt among `indexes` (paths) found next to a single `.onnx` export of
/// `stem` in `dir` (P5-D review: the folder may be a Downloads folder holding another voice's
/// index). An index whose name contains the export's name (or, for a generic export name such as
/// 4.0's `model.onnx`, its folder's) is the voice's; otherwise, in the voice's own folder
/// ([`own_folder`]), the folder's index ([`choose_index`]). `notes` says which index was added, or
/// which were left out and how to include them.
fn export_index(indexes: &[String], stem: &str, dir: &Path, notes: &mut Vec<String>) -> Option<String> {
    if indexes.is_empty() {
        return None;
    }
    let key = name_key(stem).or_else(|| usable_key(&folder_name(dir)));
    let named: Vec<String> = indexes
        .iter()
        .filter(|p| key.as_deref().is_some_and(|k| base_name(p).to_ascii_lowercase().contains(k)))
        .cloned()
        .collect();
    let chosen = if !named.is_empty() {
        choose_index(&named)
    } else if own_folder(stem, dir) {
        choose_index(indexes)
    } else {
        None
    };
    match &chosen {
        Some(p) => notes.push(format!("added the feature index {} found next to the export", base_name(p))),
        None => notes.push(format!(
            "{} next to the export {} not added: {} not named after the voice; add the voice's folder (or a zip of its files) to include its feature index",
            indexes.iter().map(|p| base_name(p)).collect::<Vec<_>>().join(", "),
            if indexes.len() == 1 { "was" } else { "were" },
            if indexes.len() == 1 { "it is" } else { "they are" },
        )),
    }
    chosen
}

/// The voice's files next to an `.onnx` export (see the module docs). `single_file`: the user chose
/// the export alone, so its folder's index is adopted only when tied to the export
/// ([`export_index`]) and a generic `config.json` from a folder that is not the voice's own is
/// noted; otherwise (a folder import) every file in the folder is the voice's.
fn bundle_from_onnx(path: &Path, single_file: bool) -> Result<Bundle, String> {
    let dir = path.parent().unwrap_or(Path::new("."));
    let stem = path.file_stem().map(|s| s.to_string_lossy().into_owned()).unwrap_or_default();
    let mut candidates: Vec<PathBuf> = vec![dir.join(CONFIG_FILE), dir.join(format!("{stem}.json"))];
    if let Some(short) = stem.strip_suffix("_SoVits").or_else(|| stem.strip_suffix("_sovits")) {
        candidates.push(dir.join(format!("{short}.json")));
    }
    if let (Some(parent), Some(folder)) = (dir.parent(), dir.file_name()) {
        candidates.push(parent.join(format!("{}.json", folder.to_string_lossy())));
    }
    candidates.dedup();
    let configs: Vec<(String, Vec<u8>)> = candidates
        .into_iter()
        .filter(|p| !is_metadata_file(p))
        .filter_map(|p| read_config_file(&p).map(|b| (p.file_name().map(|n| n.to_string_lossy().into_owned()).unwrap_or_default(), b)))
        .collect();
    let indexes: Vec<String> = voice_files(dir)
        .unwrap_or_default()
        .into_iter()
        .filter(|p| lower_ext(&p.to_string_lossy()) == "index")
        .map(|p| p.to_string_lossy().into_owned())
        .collect();
    let mut notes = Vec::new();
    let index = if single_file {
        if !own_folder(&stem, dir) && configs.iter().any(|(label, _)| label == CONFIG_FILE) {
            notes.push(format!(
                "used the config.json found next to the export (in {}); if it belongs to another voice, this voice's speakers and sample rate will be wrong: add the voice's own folder (or a zip of its files) instead",
                dir.display()
            ));
        }
        export_index(&indexes, &stem, dir, &mut notes)
    } else {
        choose_index(&indexes)
    };
    Ok(Bundle {
        generator: Source::File(path.to_path_buf()),
        generator_name: path.file_name().map(|n| n.to_string_lossy().into_owned()).unwrap_or_default(),
        configs,
        index: index.map(|p| Source::File(PathBuf::from(p))),
        skipped: notes,
        zip: None,
    })
}

/// The voice's files in a folder (not recursive).
fn bundle_from_folder(dir: &Path) -> Result<Bundle, String> {
    let files = voice_files(dir).map_err(|e| err(ErrorCode::Io, format!("{}: {e}", dir.display())))?;
    let names = |ext: &str| -> Vec<String> {
        files.iter().filter(|p| lower_ext(&p.to_string_lossy()) == ext).map(|p| p.to_string_lossy().into_owned()).collect()
    };
    let generator = PathBuf::from(choose_generator(&names("onnx"))?);
    let mut bundle = bundle_from_onnx(&generator, false)?;
    // Every other JSON in the folder may be the MoeVS config (named after the export's folder).
    for p in names("json").into_iter().map(PathBuf::from) {
        let label = p.file_name().map(|n| n.to_string_lossy().into_owned()).unwrap_or_default();
        if bundle.configs.iter().all(|(l, _)| *l != label) {
            if let Some(b) = read_config_file(&p) {
                bundle.configs.push((label, b));
            }
        }
    }
    for ext in ["pt", "pth", "ckpt", "pkl"] {
        bundle.skipped.extend(names(ext).iter().map(|n| torch_file_note(n)));
    }
    Ok(bundle)
}

/// The voice's files in a zip (guarded).
fn bundle_from_zip(path: &Path) -> Result<Bundle, String> {
    let file = fs::File::open(path).map_err(|e| err(ErrorCode::Io, format!("{}: {e}", path.display())))?;
    let mut zip = GuardedZip::open(file, ZipLimits::VOICE_ARCHIVE).map_err(|e| e.to_string())?;
    // macOS Finder's `__MACOSX/.../._name` twins carry the same extensions (P5-D review).
    let entries: Vec<(String, u64)> = zip
        .entries()
        .iter()
        .filter(|e| !is_metadata_entry(&e.name))
        .map(|e| (e.name.clone(), e.size))
        .collect();
    let by_ext = |ext: &str| -> Vec<String> { entries.iter().filter(|(n, _)| lower_ext(n) == ext).map(|(n, _)| n.clone()).collect() };
    let generator = choose_generator(&by_ext("onnx"))?;
    let mut configs = Vec::new();
    for (name, size) in entries.iter().filter(|(n, _)| lower_ext(n) == "json") {
        if *size <= MAX_CONFIG_BYTES {
            configs.push((base_name(name).to_string(), zip.read(name, MAX_CONFIG_BYTES).map_err(|e| e.to_string())?));
        }
    }
    let mut skipped = Vec::new();
    for ext in ["pt", "pth", "ckpt", "pkl"] {
        skipped.extend(by_ext(ext).iter().map(|n| torch_file_note(n)));
    }
    Ok(Bundle {
        generator_name: base_name(&generator).to_string(),
        generator: Source::Entry(generator),
        configs,
        index: choose_index(&by_ext("index")).map(Source::Entry),
        skipped,
        zip: Some(zip),
    })
}

/// Write `source` to `dest` (a file copy, or the archive entry), at most `max` bytes.
fn materialize(bundle: &mut Bundle, source: &Source, dest: &Path, max: u64) -> Result<(), String> {
    match source {
        Source::File(p) => {
            let len = fs::metadata(p).map_err(|e| err(ErrorCode::Io, format!("{}: {e}", p.display())))?.len();
            if len > max {
                return Err(err(ErrorCode::UnsupportedModel, format!("{} is {len} bytes (at most {max})", p.display())));
            }
            fs::copy(p, dest).map_err(|e| err(ErrorCode::Io, format!("{}: {e}", dest.display())))?;
        }
        Source::Entry(name) => {
            let zip = bundle.zip.as_mut().ok_or_else(|| err(ErrorCode::Io, "the archive is closed"))?;
            let bytes = zip.read(name, max).map_err(|e| e.to_string())?;
            write_atomic(dest, &bytes).map_err(|e| e.to_string())?;
        }
    }
    Ok(())
}

/// What a bundle's config files say.
struct BundleConfig {
    /// Every config merged, the training config first (see `config::SovitsConfig::merged`).
    merged: SovitsConfig,
    /// The training config's bytes (kept in the slot).
    training_raw: Option<Vec<u8>>,
    /// The speech encoder the training config names (so-vits-svc 4.0 configs name none).
    training_encoder: Option<String>,
}

/// The configs of a bundle: the first so-vits-svc training config (by its keys) wins, MoeVS / MoeSS
/// JSONs add what it does not say.
fn bundle_config(configs: &[(String, Vec<u8>)], warnings: &mut Vec<String>) -> BundleConfig {
    let mut training: Option<(SovitsConfig, Vec<u8>)> = None;
    let mut others: Vec<SovitsConfig> = Vec::new();
    for (label, bytes) in configs {
        match SovitsConfig::parse(bytes, label) {
            Ok(Some(c)) if c.is_training_config() && training.is_none() => training = Some((c, bytes.clone())),
            Ok(Some(c)) => others.push(c),
            Ok(None) => {}
            Err(e) => warnings.push(e),
        }
    }
    let training_raw = training.as_ref().map(|(_, b)| b.clone());
    let training_encoder = training.as_ref().and_then(|(c, _)| c.speech_encoder.clone());
    let mut merged = training.map(|(c, _)| c).unwrap_or_default();
    for o in others {
        merged = merged.merged(o);
    }
    BundleConfig {
        merged,
        training_raw,
        training_encoder,
    }
}

/// What the import probe measured.
#[derive(Debug, PartialEq)]
struct Probe {
    hop: usize,
}

/// Run the export once on `PROBE_FRAMES` frames of a steady voiced input; measures its hop.
fn probe(session: &mut Session, g: &SovitsGraph, width: usize, mix_width: usize) -> Result<Probe, String> {
    let t = PROBE_FRAMES;
    let s = if g.maps_content() { t / 2 + 1 } else { t };
    let content = vec![0.01f32; s * width];
    let f0 = vec![220.0f32; t];
    let ones = vec![1.0f32; t];
    let uv_i = vec![1i64; t];
    let mel2ph: Vec<i64> = (0..t).map(|j| (j / 2).min(s - 1) as i64 + 1).collect();
    let noise = vec![0.0f32; g.noise_channels * t];
    let vol = vec![0.05f32; t];
    let mut mix = vec![0.0f32; t * mix_width.max(1)];
    for j in 0..t {
        mix[j * mix_width.max(1)] = 1.0;
    }
    let ids = [0i64];
    let lengths = [s as i64];
    let halves: Vec<Vec<half::f16>> = [&content, &f0, &ones, &noise, &vol, &mix]
        .iter()
        .map(|v| {
            let mut h = Vec::new();
            to_half(v, &mut h);
            h
        })
        .collect();
    let float = float_input;
    let ranks: Vec<usize> = session.inputs().iter().map(|i| i.dtype().tensor_shape().map(|s| s.len()).unwrap_or(2)).collect();
    let row = |rank: usize| if rank == 1 { vec![t] } else { vec![1, t] };
    let mut feed: Vec<(String, SessionInputValue<'_>)> = Vec::new();
    for (i, (name, role)) in g.names.iter().zip(g.roles.iter()).enumerate() {
        let half = g.half(*role);
        let v: SessionInputValue<'_> = match role {
            Role::Content => {
                let shape = if g.content_time_major { vec![1, s, width] } else { vec![1, width, s] };
                float(shape, &content, &halves[0], half, "c")?
            }
            Role::F0 => float(row(ranks[i]), &f0, &halves[1], half, "f0")?,
            Role::Uv if g.uv_int => TensorRef::from_array_view((row(ranks[i]), &uv_i[..])).map_err(|e| e.to_string())?.into(),
            Role::Uv => float(row(ranks[i]), &ones, &halves[2], half, "uv")?,
            Role::Mel2ph => TensorRef::from_array_view((vec![1, t], &mel2ph[..])).map_err(|e| e.to_string())?.into(),
            Role::Noise => float(vec![1, g.noise_channels, t], &noise, &halves[3], half, "noise")?,
            Role::Volume => float(row(ranks[i]), &vol, &halves[4], half, "vol")?,
            Role::Speaker => match g.speaker {
                Some(SpeakerInput::Mix { .. }) => float(vec![t, mix_width.max(1)], &mix, &halves[5], half, "sid")?,
                Some(SpeakerInput::Id { rank: 2 }) => TensorRef::from_array_view((vec![1usize, 1], &ids[..])).map_err(|e| e.to_string())?.into(),
                _ => TensorRef::from_array_view((vec![1usize], &ids[..])).map_err(|e| e.to_string())?.into(),
            },
            Role::Lengths => TensorRef::from_array_view((vec![1usize], &lengths[..])).map_err(|e| e.to_string())?.into(),
        };
        feed.push((name.clone(), v));
    }
    let outputs = session.run(feed).map_err(|e| err(ErrorCode::Session, format!("the export does not run: {e}")))?;
    let audio: Vec<f32> = match outputs[0].try_extract_tensor::<f32>() {
        Ok((_, d)) => d.to_vec(),
        Err(_) => outputs[0]
            .try_extract_tensor::<half::f16>()
            .map_err(|e| err(ErrorCode::Session, format!("the export's output is not audio: {e}")))?
            .1
            .iter()
            .map(|h| h.to_f32())
            .collect(),
    };
    if audio.is_empty() || !audio.len().is_multiple_of(t) {
        return Err(err(
            ErrorCode::UnsupportedModel,
            format!("the export returned {} samples for {t} frames (not a whole number per frame)", audio.len()),
        ));
    }
    if !audio.iter().all(|x| x.is_finite()) {
        return Err(err(ErrorCode::NonFinite, "the export's output is not finite on a test input"));
    }
    Ok(Probe { hop: audio.len() / t })
}

/// Fill `ctx.slot_dir` with the voice in `request.source` (see the module docs).
pub fn create_slot(request: &FamilySlotRequest, ctx: &FamilySlotContext<'_>) -> Result<CreatedSlot, String> {
    let cancelled = || ctx.cancel.load(Ordering::Acquire);
    let check = || if cancelled() { Err(err(ErrorCode::Job, "the import was cancelled")) } else { Ok(()) };
    let raw = request.source.as_deref().map(str::trim).filter(|s| !s.is_empty()).ok_or_else(|| err(ErrorCode::Job, "choose a file to import"))?;
    let source = PathBuf::from(raw);
    if !source.exists() {
        return Err(err(ErrorCode::Io, format!("{} does not exist", source.display())));
    }
    (ctx.progress)("reading", 0.1, "Reading the voice");
    let mut bundle = if source.is_dir() {
        bundle_from_folder(&source)?
    } else {
        let mut head = [0u8; 8];
        let n = fs::File::open(&source).and_then(|mut f| f.read(&mut head)).map_err(|e| err(ErrorCode::Io, format!("{}: {e}", source.display())))?;
        let ext = lower_ext(raw);
        if looks_like_zip(&head[..n]) && ext == "zip" {
            bundle_from_zip(&source)?
        } else if matches!(ext.as_str(), "pth" | "pt" | "ckpt") || looks_like_zip(&head[..n]) || (n >= 2 && head[0] == 0x80) {
            return Err(err(
                ErrorCode::UnsupportedModel,
                format!(
                    "{} is a PyTorch checkpoint; so-vits-svc voices are added as their ONNX export (so-vits-svc's onnx_export.py writes <name>_SoVits.onnx and <name>.json), together with the training config.json",
                    base_name(raw)
                ),
            ));
        } else {
            if let Some(problem) = non_onnx_file_problem(&source) {
                return Err(err(ErrorCode::UnsupportedModel, problem));
            }
            bundle_from_onnx(&source, true)?
        }
    };
    check()?;
    let mut warnings: Vec<String> = std::mem::take(&mut bundle.skipped);

    // The generator into the slot, then its graph.
    let gen_dest = ctx.slot_dir.join(GENERATOR_FILE);
    let generator = match &bundle.generator {
        Source::File(p) => Source::File(p.clone()),
        Source::Entry(n) => Source::Entry(n.clone()),
    };
    materialize(&mut bundle, &generator, &gen_dest, MAX_GENERATOR_BYTES)?;
    if let Some(problem) = non_onnx_file_problem(&gen_dest) {
        return Err(err(ErrorCode::UnsupportedModel, problem));
    }
    check()?;
    (ctx.progress)("converting", 0.35, "Checking the export");
    let opts = VcSessionOptions {
        intra_threads: Some(2),
        ..VcSessionOptions::default()
    };
    let (mut session, _) =
        build_session(None, "cpu", &opts, ModelSource::File(&gen_dest)).map_err(|e| load_error(&bundle.generator_name, &e.to_string()))?;
    let graph = SovitsGraph::resolve(&session, Path::new(&bundle.generator_name)).map_err(|e| err(ErrorCode::UnsupportedModel, e))?;
    let BundleConfig {
        merged: config,
        training_raw,
        training_encoder,
    } = bundle_config(&bundle.configs, &mut warnings);
    if config.sovits3 {
        return Err(err(
            ErrorCode::UnsupportedModel,
            "the voice is a so-vits-svc 3.0 export (hubert-soft), which Sigil does not run; export it with so-vits-svc 4.0 or 4.1",
        ));
    }
    let width = graph
        .content_width
        .or(config.ssl_dim.map(|d| d as usize))
        .ok_or_else(|| err(ErrorCode::UnsupportedModel, "the export does not declare its content width; add its config.json"))?;
    let config_read = !config.sources.is_empty();
    let (kind, encoder_note) = encoder_for(config.speech_encoder.as_deref(), width, config_read).map_err(|e| err(ErrorCode::UnsupportedModel, e))?;
    warnings.extend(encoder_note);
    let speakers = speaker_count(&config);
    let mix_width = match graph.speaker {
        Some(SpeakerInput::Mix { speakers: Some(n) }) => n,
        Some(SpeakerInput::Mix { speakers: None }) => {
            if speakers == 0 {
                return Err(err(
                    ErrorCode::UnsupportedModel,
                    "the export takes speaker weights but no config says how many speakers it has; add the MoeVS JSON onnx_export.py wrote next to it",
                ));
            }
            speakers
        }
        _ => 0,
    };
    check()?;
    (ctx.progress)("smoke", 0.6, "Testing the voice");
    let measured = probe(&mut session, &graph, width, mix_width)?;
    drop(session);
    let hop = measured.hop;
    if let Some(h) = config.hop.filter(|&h| h as usize != hop) {
        return Err(err(
            ErrorCode::ShapeMismatch,
            format!("the voice's config says {h} samples per frame but the export produces {hop}; the config belongs to another voice"),
        ));
    }
    let sample_rate = match config.sample_rate {
        Some(sr) => sr,
        None => {
            let sr = rate_for_hop(hop as u32).ok_or_else(|| {
                err(ErrorCode::UnsupportedModel, format!("the export produces {hop} samples per frame and no config says its sample rate; add its config.json"))
            })?;
            warnings.push(format!("no config.json: the sample rate is taken as {sr} Hz from the export's {hop}-sample hop"));
            sr
        }
    };
    if !(8_000..=96_000).contains(&sample_rate) {
        return Err(err(ErrorCode::UnsupportedModel, format!("a sample rate of {sample_rate} Hz is not supported")));
    }
    check()?;

    (ctx.progress)("writing", 0.8, "Writing the voice");
    let mut index = None;
    if let Some(ix) = bundle.index.take() {
        let dest = ctx.slot_dir.join(INDEX_FILE);
        match materialize(&mut bundle, &ix, &dest, MAX_INDEX_BYTES).and_then(|_| FaissIndex::load(&dest)) {
            Ok(i) if i.dim() == width => index = Some(INDEX_FILE.to_string()),
            Ok(i) => {
                warnings.push(format!("the feature index has {} dimensions but the voice uses {width}; it was skipped", i.dim()));
                let _ = fs::remove_file(&dest);
            }
            Err(e) => {
                warnings.push(format!("the feature index could not be read ({e}); it was skipped"));
                let _ = fs::remove_file(&dest);
            }
        }
    }
    if let Some(raw_cfg) = &training_raw {
        write_atomic(&ctx.slot_dir.join(CONFIG_FILE), raw_cfg).map_err(|e| e.to_string())?;
    }
    let (sha, _) = sha256_file(&gen_dest).map_err(|e| e.to_string())?;
    let record = SovitsRecord {
        format: RECORD_FORMAT.into(),
        version: RECORD_VERSION,
        hop: hop as u32,
        speech_encoder: training_encoder,
        speakers: config.speakers.clone(),
        character_mix: mix_width > 0,
        volume: graph.has(Role::Volume),
        source: bundle.generator_name.clone(),
        configs: config.sources.clone(),
    };
    let bytes = serde_json::to_vec_pretty(&record).map_err(|e| err(ErrorCode::Io, e.to_string()))?;
    write_atomic(&ctx.slot_dir.join(RECORD_FILE), &bytes).map_err(|e| e.to_string())?;
    let mut manifest = family_manifest(FamilyKind::Sovits, &ctx.name, ctx.options.clone());
    manifest.generator = GENERATOR_FILE.to_string();
    manifest.sample_rate = sample_rate;
    manifest.emb_channels = width as u32;
    let v1 = kind == EmbedderKind::ContentVec256;
    manifest.version = if v1 { RvcVersion::V1 } else { RvcVersion::V2 };
    manifest.emb_layer = if v1 { 9 } else { 12 };
    manifest.use_final_proj = v1;
    manifest.embedder = Some(kind.as_str().to_string());
    manifest.speaker_count = Some(if mix_width > 0 { mix_width } else { speakers.max(1) } as u32);
    manifest.source_sha256 = Some(sha);
    manifest.import_record = Some(RECORD_FILE.to_string());
    manifest.index = index;
    write_family_manifest(&ctx.slot_dir, &manifest).map_err(|e| err(ErrorCode::Io, e))?;
    Ok(CreatedSlot { warnings })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn generators_and_indexes_are_chosen_by_name() {
        assert!(choose_generator(&[]).unwrap_err().starts_with("not_a_checkpoint"));
        assert_eq!(choose_generator(&["a/x.onnx".into()]).unwrap(), "a/x.onnx");
        assert_eq!(choose_generator(&["a/contentvec.onnx".into(), "a/nyaru_SoVits.onnx".into()]).unwrap(), "a/nyaru_SoVits.onnx");
        assert!(choose_generator(&["a.onnx".into(), "b.onnx".into()]).unwrap_err().starts_with("ambiguous_archive"));
        assert_eq!(choose_index(&["x/feature_and_index.index".into(), "x/other.index".into()]).as_deref(), Some("x/feature_and_index.index"));
        assert_eq!(choose_index(&["a.index".into(), "b.index".into()]), None);
        assert!(torch_file_note("logs/kmeans_10000.pt").contains("kmeans"));
        assert!(torch_file_note("G_8000.pth").contains("skipped"));
        let pkl = torch_file_note("logs/44k/feature_and_index.pkl");
        assert!(pkl.contains("pickle") && pkl.contains("FAISS .index"), "{pkl}");
    }

    /// The `spk` names win over `n_speakers` (so-vits-svc 4.0 configs keep the template's 200 for a
    /// voice of one speaker); without names, `n_speakers` (or MoeVS `Characters`) counts.
    #[test]
    fn speakers_come_from_the_names_first() {
        let one = SovitsConfig::parse(br#"{"data": {}, "model": {"n_speakers": 200}, "spk": {"nyaru": 0}}"#, "config.json").unwrap().unwrap();
        assert_eq!(speaker_count(&one), 1);
        let unnamed = SovitsConfig::parse(br#"{"data": {}, "model": {"n_speakers": 3}}"#, "config.json").unwrap().unwrap();
        assert_eq!(speaker_count(&unnamed), 3);
        let moevs = SovitsConfig::parse(br#"{"Rate": 44100, "Characters": ["a", "b"]}"#, "a.json").unwrap().unwrap();
        assert_eq!(speaker_count(&moevs), 2);
        assert_eq!(speaker_count(&SovitsConfig::default()), 0);
    }

    #[test]
    fn cpu_load_errors_name_fp16_exports() {
        let msg = load_error("v.onnx", "Could not find an implementation for Conv(11) node with name '/dec/conv_pre/Conv'");
        assert!(msg.starts_with("session") && msg.contains("fp16") && msg.contains("fp32"), "{msg}");
        let other = load_error("v.onnx", "Protobuf parsing failed");
        assert!(other.contains("does not load: Protobuf") && !other.contains("fp16"), "{other}");
    }

    #[test]
    fn configs_merge_with_the_training_config_first() {
        let moevs = br#"{"Rate": 32000, "Hop": 320, "Hubert": "vec-256-layer-9", "Characters": ["a"], "CharaMix": false}"#.to_vec();
        let training = br#"{"data": {"sampling_rate": 44100, "hop_length": 512}, "model": {"speech_encoder": "vec768l12", "ssl_dim": 768}, "spk": {"x": 0}}"#.to_vec();
        let mut w = Vec::new();
        let b = bundle_config(&[("nyaru.json".into(), moevs), ("config.json".into(), training.clone()), ("bad.json".into(), b"{".to_vec())], &mut w);
        let c = &b.merged;
        assert_eq!((c.sample_rate, c.hop), (Some(44_100), Some(512)));
        assert_eq!(c.speech_encoder.as_deref(), Some("vec768l12"));
        assert_eq!(c.speakers, vec!["x".to_string()]);
        assert_eq!(b.training_raw, Some(training));
        assert_eq!(b.training_encoder.as_deref(), Some("vec768l12"));
        assert_eq!(w.len(), 1, "the broken JSON is a warning: {w:?}");
    }

    /// A so-vits-svc 4.0 training config (no `speech_encoder`) is still the training config: it is
    /// kept and wins over a MoeSS JSON, whose `hubert4.0` only fills the encoder (P5-D review).
    #[test]
    fn a_4_0_training_config_is_the_training_config() {
        let v40 = br#"{"data": {"sampling_rate": 44100, "hop_length": 512}, "model": {"ssl_dim": 256, "n_speakers": 200}, "spk": {"nyaru": 0}}"#.to_vec();
        let moess = br#"{"Rate": 32000, "Hop": 320, "Hubert": "hubert4.0", "SoVits4": true, "Characters": ["a", "b"]}"#.to_vec();
        let mut w = Vec::new();
        // The MoeSS JSON first: the order of files never makes it the training config.
        let b = bundle_config(&[("nyaru.json".into(), moess), ("config.json".into(), v40.clone())], &mut w);
        assert!(w.is_empty(), "{w:?}");
        assert_eq!(b.training_raw, Some(v40), "the 4.0 config is kept in the slot");
        assert_eq!(b.training_encoder, None);
        assert_eq!((b.merged.sample_rate, b.merged.hop), (Some(44_100), Some(512)), "the training config wins");
        assert_eq!(b.merged.speakers, vec!["nyaru".to_string()]);
        assert_eq!(b.merged.speech_encoder.as_deref(), Some("hubert4.0"));
    }

    fn scratch(tag: &str) -> PathBuf {
        let dir = std::env::temp_dir().join(format!("sigil-sovits-slot-{tag}-{}", std::process::id()));
        let _ = fs::remove_dir_all(&dir);
        fs::create_dir_all(&dir).unwrap();
        dir
    }

    fn names_of(b: &Bundle) -> Option<String> {
        b.index.as_ref().map(|s| match s {
            Source::File(p) => p.file_name().unwrap().to_string_lossy().into_owned(),
            Source::Entry(n) => base_name(n).to_string(),
        })
    }

    /// A single export in a folder of other files (Downloads) adopts only an index named after it,
    /// and says what it added or left out; in the voice's own folder its index is adopted as before
    /// (P5-D review).
    #[test]
    fn a_single_export_adopts_only_its_own_index() {
        let root = scratch("index");
        // Downloads: another RVC voice's index next to the export.
        let downloads = root.join("Downloads");
        fs::create_dir_all(&downloads).unwrap();
        let onnx = downloads.join("nyaru_SoVits.onnx");
        fs::write(&onnx, b"onnx").unwrap();
        fs::write(downloads.join("added_IVF256_Flat_nprobe_1_other_v2.index"), b"i").unwrap();
        fs::write(downloads.join("config.json"), b"{}").unwrap();
        let b = bundle_from_onnx(&onnx, true).unwrap();
        assert_eq!(names_of(&b), None);
        assert!(b.skipped.iter().any(|n| n.contains("added_IVF256_Flat_nprobe_1_other_v2.index") && n.contains("not added")), "{:?}", b.skipped);
        assert!(b.skipped.iter().any(|n| n.contains("config.json") && n.contains("another voice")), "{:?}", b.skipped);
        // An index named after the voice is the voice's, wherever it is.
        fs::write(downloads.join("added_IVF256_Flat_nprobe_1_nyaru_v2.index"), b"i").unwrap();
        let b = bundle_from_onnx(&onnx, true).unwrap();
        assert_eq!(names_of(&b).as_deref(), Some("added_IVF256_Flat_nprobe_1_nyaru_v2.index"));
        assert!(b.skipped.iter().any(|n| n.contains("added the feature index added_IVF256_Flat_nprobe_1_nyaru_v2.index")), "{:?}", b.skipped);
        // The exporter's own folder (`checkpoints/nyaru/`): its lone index, no config note.
        let own = root.join("checkpoints").join("nyaru");
        fs::create_dir_all(&own).unwrap();
        let onnx = own.join("nyaru_SoVits.onnx");
        fs::write(&onnx, b"onnx").unwrap();
        fs::write(own.join("feature.index"), b"i").unwrap();
        fs::write(own.join("._feature.index"), b"\x00\x05\x16\x07").unwrap();
        fs::write(own.join("config.json"), b"{}").unwrap();
        let b = bundle_from_onnx(&onnx, true).unwrap();
        assert_eq!(names_of(&b).as_deref(), Some("feature.index"), "the AppleDouble twin is not a second index");
        assert!(!b.skipped.iter().any(|n| n.contains("another voice")), "{:?}", b.skipped);
        // so-vits-svc 4.0's generic `model.onnx`: an index named after its folder.
        let v40 = root.join("alto");
        fs::create_dir_all(&v40).unwrap();
        let onnx = v40.join("model.onnx");
        fs::write(&onnx, b"onnx").unwrap();
        fs::write(v40.join("added_IVF100_Flat_nprobe_1_alto_v2.index"), b"i").unwrap();
        assert_eq!(names_of(&bundle_from_onnx(&onnx, true).unwrap()).as_deref(), Some("added_IVF100_Flat_nprobe_1_alto_v2.index"));
        // A folder import takes the folder's index as before (the user chose the folder), silently.
        let b = bundle_from_onnx(&downloads.join("nyaru_SoVits.onnx"), false).unwrap();
        assert!(b.index.is_some() && b.skipped.is_empty(), "{:?}", b.skipped);
        let _ = fs::remove_dir_all(&root);
    }

    #[test]
    fn export_names_tie_files_to_the_voice() {
        assert_eq!(name_key("nyaru_SoVits").as_deref(), Some("nyaru"));
        assert_eq!(name_key("Nyaru-sovits").as_deref(), Some("nyaru"));
        assert_eq!(name_key("model"), None);
        assert_eq!(name_key("ab"), None);
        assert!(own_folder("nyaru_SoVits", Path::new("checkpoints/nyaru")));
        assert!(own_folder("nyaru_v2_SoVits", Path::new("x/Nyaru")));
        assert!(own_folder("nyaru_SoVits", Path::new("x/Nyaru voice")));
        assert!(!own_folder("nyaru_SoVits", Path::new("C:/Users/me/Downloads")));
        assert!(!own_folder("model", Path::new("checkpoints/model")));
    }

    /// A macOS Finder zip's `__MACOSX/` AppleDouble twins and a folder's `._name` files are never
    /// voice files (P5-D review: they made the export ambiguous and the config a JSON warning).
    #[test]
    fn os_metadata_is_not_a_voice_file() {
        use crate::services::neural_audio::vc::import::zipguard::tests::build_zip;
        let root = scratch("macos");
        let apple_double = b"\x00\x05\x16\x07\x00\x02\x00\x00Mac OS X        ".as_slice();
        let zip_bytes = build_zip(&[
            ("nyaru/nyaru_SoVits.onnx", b"onnx".as_slice(), zip::CompressionMethod::Stored),
            ("nyaru/config.json", br#"{"data": {}, "model": {}}"#.as_slice(), zip::CompressionMethod::Stored),
            ("nyaru/my.index", b"i".as_slice(), zip::CompressionMethod::Stored),
            ("__MACOSX/nyaru/._nyaru_SoVits.onnx", apple_double, zip::CompressionMethod::Stored),
            ("__MACOSX/nyaru/._config.json", apple_double, zip::CompressionMethod::Stored),
            ("__MACOSX/nyaru/._my.index", apple_double, zip::CompressionMethod::Stored),
            ("nyaru/.DS_Store", b"x".as_slice(), zip::CompressionMethod::Stored),
        ]);
        let zip_path = root.join("nyaru.zip");
        fs::write(&zip_path, zip_bytes).unwrap();
        let b = bundle_from_zip(&zip_path).unwrap();
        assert_eq!(b.generator_name, "nyaru_SoVits.onnx");
        assert_eq!(b.configs.iter().map(|(l, _)| l.as_str()).collect::<Vec<_>>(), vec!["config.json"]);
        assert_eq!(names_of(&b).as_deref(), Some("my.index"), "the lone index, not ambiguous with its twin");
        // A folder written by macOS on a FAT / exFAT drive.
        let folder = root.join("voice");
        fs::create_dir_all(&folder).unwrap();
        fs::write(folder.join("model.onnx"), b"onnx").unwrap();
        fs::write(folder.join("._model.onnx"), apple_double).unwrap();
        fs::write(folder.join("._config.json"), apple_double).unwrap();
        fs::write(folder.join("Thumbs.db"), b"x").unwrap();
        let b = bundle_from_folder(&folder).unwrap();
        assert_eq!(b.generator_name, "model.onnx");
        assert!(b.configs.is_empty(), "{:?}", b.configs.iter().map(|(l, _)| l).collect::<Vec<_>>());
        let _ = fs::remove_dir_all(&root);
    }
}
