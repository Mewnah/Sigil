//! Tauri build script: ACL manifests for the app commands and every inlined plugin.
//!
//! Plugin command lists are data-driven (P0-S0): each inlined plugin has one file
//! `plugin-commands/<plugin-name>.txt` listing its IPC commands, one per line (`#` starts a
//! comment). The file stem is the plugin name used in `Builder::new("<plugin-name>")` and in the
//! capability id `"<plugin-name>:default"`. Every plugin gets `DefaultPermissionRule::AllowAllCommands`.
//!
//! To add a command: add the `#[command]`, list it in the plugin's `generate_handler!`, and add a
//! line to `plugin-commands/<plugin-name>.txt`. A command missing here compiles fine but fails at
//! runtime with a "not allowed" ACL error. A new plugin also needs its own
//! `capabilities/<plugin-name>.json` granting `"<plugin-name>:default"` to the `main` window.
//!
//! The AppManifest list (commands registered directly in main.rs) stays inline below.
//!
//! Model catalog (P1-S0b): every `model-catalog/<area>.json` fragment (files starting with `_`,
//! such as `_schema.json`, are skipped) is validated against the rules of `_schema.json` and the
//! merged entries are written to `$OUT_DIR/model_catalog.json` as `{"entries":[...]}` (fragments in
//! file-name order, entries in file order). `model_store::EMBEDDED_CATALOG_JSON` embeds that file.
//! Ids must be unique across fragments, every `requires` id must exist, and no entry's install
//! folder may equal or sit inside another entry's. Release builds
//! (`PROFILE=release`) also refuse `published` entries with a file that has neither `sha256` nor
//! `git_oid`: release catalogs are fully pinned; trust-on-first-use is for development only.
//! Adding or editing a fragment is a data-only change (no Rust edit, no hotspot).
//! A fragment may also carry a top-level `routing` object (P2-S0: the STT routing table in
//! `stt-routing.json`, read by `speech::router`); it is checked to be an object and not embedded.

use std::collections::{BTreeMap, BTreeSet};
use std::fs;
use std::path::Path;

use serde_json::{Map, Value};

const PLUGIN_COMMANDS_DIR: &str = "plugin-commands";
const MODEL_CATALOG_DIR: &str = "model-catalog";
const MODEL_CATALOG_OUT: &str = "model_catalog.json";

/// Parse one `plugin-commands/<plugin>.txt` file: trimmed non-empty lines, `#` comments stripped.
fn parse_command_list(plugin: &str, text: &str) -> Vec<String> {
    let mut seen = BTreeSet::new();
    let mut commands = Vec::new();
    for (line_no, raw) in text.lines().enumerate() {
        let line = raw.split('#').next().unwrap_or("").trim();
        if line.is_empty() {
            continue;
        }
        let valid = line
            .chars()
            .all(|c| c.is_ascii_lowercase() || c.is_ascii_digit() || c == '_')
            && !line.starts_with(|c: char| c.is_ascii_digit());
        if !valid {
            panic!(
                "{PLUGIN_COMMANDS_DIR}/{plugin}.txt:{}: invalid command name {line:?} (expected snake_case)",
                line_no + 1
            );
        }
        if !seen.insert(line.to_string()) {
            panic!("{PLUGIN_COMMANDS_DIR}/{plugin}.txt:{}: duplicate command {line:?}", line_no + 1);
        }
        commands.push(line.to_string());
    }
    if commands.is_empty() {
        // tauri-build emits no default permission for a plugin without commands, so its
        // "<plugin>:default" capability entry would fail to resolve.
        panic!("{PLUGIN_COMMANDS_DIR}/{plugin}.txt lists no commands");
    }
    commands
}

/// Read every `plugin-commands/*.txt` (sorted by plugin name) as `(plugin, commands)`.
/// Strings are leaked because tauri-build takes `&'static` slices; this runs once per build.
fn load_inlined_plugins() -> Vec<(&'static str, tauri_build::InlinedPlugin)> {
    let dir = Path::new(PLUGIN_COMMANDS_DIR);
    println!("cargo:rerun-if-changed={PLUGIN_COMMANDS_DIR}");
    let mut files: Vec<_> = fs::read_dir(dir)
        .unwrap_or_else(|e| panic!("cannot read {PLUGIN_COMMANDS_DIR}/: {e}"))
        .map(|entry| entry.expect("plugin-commands entry").path())
        .filter(|p| p.extension().and_then(|e| e.to_str()) == Some("txt"))
        .collect();
    files.sort();

    let mut plugins = Vec::with_capacity(files.len());
    for path in files {
        println!("cargo:rerun-if-changed={}", path.display());
        let plugin = path
            .file_stem()
            .and_then(|s| s.to_str())
            .unwrap_or_else(|| panic!("non-UTF-8 plugin command file name: {}", path.display()))
            .to_string();
        if plugin.is_empty()
            || !plugin
                .chars()
                .all(|c| c.is_ascii_lowercase() || c.is_ascii_digit() || c == '-')
        {
            panic!("{}: plugin names are lower-case kebab-case", path.display());
        }
        let text = fs::read_to_string(&path).unwrap_or_else(|e| panic!("cannot read {}: {e}", path.display()));
        let commands: Vec<&'static str> = parse_command_list(&plugin, &text)
            .into_iter()
            .map(|c| -> &'static str { Box::leak(c.into_boxed_str()) })
            .collect();
        let commands: &'static [&'static str] = Vec::leak(commands);
        let name: &'static str = Box::leak(plugin.into_boxed_str());
        plugins.push((
            name,
            tauri_build::InlinedPlugin::new()
                .commands(commands)
                .default_permission(tauri_build::DefaultPermissionRule::AllowAllCommands),
        ));
    }
    plugins
}

// ---------------------------------------------------------------------------------------------
// Model catalog: validate the fragments and embed the merged catalog
// ---------------------------------------------------------------------------------------------

const ENTRY_KEYS: &[&str] = &[
    "id",
    "family",
    "kind",
    "version",
    "title",
    "description",
    "license",
    "status",
    "requires",
    "min_vram_mb",
    "variant",
    "files",
];
const FILE_KEYS: &[&str] = &["path", "size", "sha256", "git_oid", "sources", "archive"];
const SOURCE_KEYS: &[&str] = &["kind", "url", "revision"];
const ARCHIVE_KEYS: &[&str] = &["format", "strip", "include"];
const ENTRY_KINDS: &[&str] = &["model", "runtime", "template", "bench"];
const ENTRY_STATUSES: &[&str] = &["published", "unpublished"];
const SOURCE_KINDS: &[&str] = &[
    "huggingface",
    "hf-mirror",
    "modelscope",
    "github-release",
    "nvidia-redist",
    "direct",
    "custom",
];
const ARCHIVE_FORMATS: &[&str] = &["zip", "tar.bz2", "tar.gz"];

/// `stt/silero-vad-v6.2`: `/`-separated segments of `[a-z0-9][a-z0-9._-]*`.
fn valid_catalog_id(id: &str) -> bool {
    !id.is_empty()
        && id.split('/').all(|seg| {
            seg.starts_with(|c: char| c.is_ascii_lowercase() || c.is_ascii_digit())
                && seg
                    .chars()
                    .all(|c| c.is_ascii_lowercase() || c.is_ascii_digit() || matches!(c, '.' | '_' | '-'))
        })
}

fn valid_family(family: &str) -> bool {
    family.starts_with(|c: char| c.is_ascii_lowercase() || c.is_ascii_digit())
        && family
            .chars()
            .all(|c| c.is_ascii_lowercase() || c.is_ascii_digit() || matches!(c, '_' | '-'))
}

/// Relative install path: no leading `/`, no `..`, only `[A-Za-z0-9._/+-]`.
fn valid_install_path(path: &str) -> bool {
    !path.is_empty()
        && !path.starts_with('/')
        && !path.contains("..")
        && path
            .chars()
            .all(|c| c.is_ascii_alphanumeric() || matches!(c, '.' | '_' | '/' | '+' | '-'))
        && path.split('/').all(|seg| !seg.is_empty())
}

fn is_lower_hex(s: &str, len: usize) -> bool {
    s.len() == len && s.chars().all(|c| c.is_ascii_digit() || ('a'..='f').contains(&c))
}

fn check_keys(ctx: &str, obj: &Map<String, Value>, allowed: &[&str], required: &[&str]) {
    for key in obj.keys() {
        if !allowed.contains(&key.as_str()) {
            panic!("{ctx}: unknown key {key:?} (allowed: {})", allowed.join(", "));
        }
    }
    for key in required {
        if !obj.contains_key(*key) {
            panic!("{ctx}: missing required key {key:?}");
        }
    }
}

fn str_field<'a>(ctx: &str, obj: &'a Map<String, Value>, key: &str) -> &'a str {
    obj.get(key)
        .and_then(Value::as_str)
        .unwrap_or_else(|| panic!("{ctx}: {key:?} must be a string"))
}

fn check_enum(ctx: &str, key: &str, value: &str, allowed: &[&str]) {
    if !allowed.contains(&value) {
        panic!("{ctx}: {key} {value:?} is not one of {}", allowed.join(", "));
    }
}

fn validate_catalog_source(ctx: &str, source: &Value) {
    let obj = source
        .as_object()
        .unwrap_or_else(|| panic!("{ctx}: a source must be an object"));
    check_keys(ctx, obj, SOURCE_KEYS, &["kind", "url"]);
    check_enum(ctx, "source kind", str_field(ctx, obj, "kind"), SOURCE_KINDS);
    let url = str_field(ctx, obj, "url");
    if !url.starts_with("https://") || url.len() <= "https://".len() {
        panic!("{ctx}: source url {url:?} must be an https:// URL pinned to an immutable revision");
    }
    if let Some(rev) = obj.get("revision") {
        if !rev.is_string() {
            panic!("{ctx}: revision must be a string");
        }
    }
}

/// Returns whether the file is hash-pinned (`sha256` or `git_oid`).
fn validate_catalog_file(ctx: &str, file: &Value) -> bool {
    let obj = file
        .as_object()
        .unwrap_or_else(|| panic!("{ctx}: a file must be an object"));
    check_keys(ctx, obj, FILE_KEYS, &["path", "size", "sources"]);
    let path = str_field(ctx, obj, "path");
    if !valid_install_path(path) {
        panic!("{ctx}: file path {path:?} must be relative, without '..', using [A-Za-z0-9._/+-]");
    }
    let ctx = format!("{ctx} file {path:?}");
    if obj.get("size").and_then(Value::as_u64).is_none() {
        panic!("{ctx}: size must be a non-negative integer (bytes)");
    }
    let sha256 = obj.get("sha256");
    let git_oid = obj.get("git_oid");
    if let Some(v) = sha256 {
        if !v.as_str().is_some_and(|s| is_lower_hex(s, 64)) {
            panic!("{ctx}: sha256 must be 64 lower-case hex characters");
        }
    }
    if let Some(v) = git_oid {
        if !v.as_str().is_some_and(|s| is_lower_hex(s, 40)) {
            panic!("{ctx}: git_oid must be 40 lower-case hex characters");
        }
    }
    if sha256.is_some() && git_oid.is_some() {
        panic!("{ctx}: give either sha256 or git_oid, not both");
    }
    let sources = obj
        .get("sources")
        .and_then(Value::as_array)
        .unwrap_or_else(|| panic!("{ctx}: sources must be an array"));
    if sources.is_empty() {
        panic!("{ctx}: sources must list at least one mirror");
    }
    for (i, source) in sources.iter().enumerate() {
        validate_catalog_source(&format!("{ctx} source #{i}"), source);
    }
    if let Some(archive) = obj.get("archive") {
        let a = archive
            .as_object()
            .unwrap_or_else(|| panic!("{ctx}: archive must be an object"));
        check_keys(&ctx, a, ARCHIVE_KEYS, &["format"]);
        check_enum(&ctx, "archive format", str_field(&ctx, a, "format"), ARCHIVE_FORMATS);
        if let Some(strip) = a.get("strip") {
            if strip.as_u64().is_none() {
                panic!("{ctx}: archive strip must be a non-negative integer");
            }
        }
        if let Some(include) = a.get("include") {
            if !include.as_array().is_some_and(|l| l.iter().all(Value::is_string)) {
                panic!("{ctx}: archive include must be an array of glob strings");
            }
        }
    }
    sha256.is_some() || git_oid.is_some()
}

/// Validate one entry; returns its id.
fn validate_catalog_entry(ctx: &str, entry: &Value, release: bool) -> String {
    let obj = entry
        .as_object()
        .unwrap_or_else(|| panic!("{ctx}: an entry must be an object"));
    check_keys(
        ctx,
        obj,
        ENTRY_KEYS,
        &["id", "family", "kind", "version", "license", "files", "status"],
    );
    let id = str_field(ctx, obj, "id");
    if !valid_catalog_id(id) {
        panic!("{ctx}: id {id:?} must be '/'-separated segments of [a-z0-9][a-z0-9._-]*");
    }
    let ctx = format!("{ctx} ({id})");
    let family = str_field(&ctx, obj, "family");
    if !valid_family(family) {
        panic!("{ctx}: family {family:?} must match [a-z0-9][a-z0-9_-]*");
    }
    check_enum(&ctx, "kind", str_field(&ctx, obj, "kind"), ENTRY_KINDS);
    let status = str_field(&ctx, obj, "status");
    check_enum(&ctx, "status", status, ENTRY_STATUSES);
    if str_field(&ctx, obj, "version").is_empty() {
        panic!("{ctx}: version must not be empty");
    }
    str_field(&ctx, obj, "license");
    for key in ["title", "description", "variant"] {
        if obj.get(key).is_some_and(|v| !v.is_string()) {
            panic!("{ctx}: {key} must be a string");
        }
    }
    if obj.get("min_vram_mb").is_some_and(|v| v.as_u64().is_none()) {
        panic!("{ctx}: min_vram_mb must be a non-negative integer");
    }
    if let Some(requires) = obj.get("requires") {
        let ok = requires
            .as_array()
            .is_some_and(|l| l.iter().all(|r| r.as_str().is_some_and(valid_catalog_id)));
        if !ok {
            panic!("{ctx}: requires must be an array of catalog ids");
        }
    }
    let files = obj
        .get("files")
        .and_then(Value::as_array)
        .unwrap_or_else(|| panic!("{ctx}: files must be an array"));
    if files.is_empty() {
        panic!("{ctx}: files must list at least one file");
    }
    let mut paths = BTreeSet::new();
    for (i, file) in files.iter().enumerate() {
        let pinned = validate_catalog_file(&format!("{ctx} file #{i}"), file);
        let path = file["path"].as_str().unwrap_or_default().to_string();
        if !paths.insert(path.clone()) {
            panic!("{ctx}: duplicate file path {path:?}");
        }
        if release && status == "published" && !pinned {
            panic!(
                "{ctx}: file {path:?} of a published entry has no sha256 / git_oid; release catalogs must be fully pinned"
            );
        }
    }
    id.to_string()
}

/// Families whose files live in a shared legacy folder (mirrors `model_store::legacy::LEGACY_FOLDERS`).
const LEGACY_FAMILIES: &[&str] = &["whisper", "vosk"];

/// Folder-name form of an id segment (mirrors `model_store::layout::sanitize_dir_name`).
fn sanitize_dir_name(s: &str) -> String {
    let mut out: String = s
        .chars()
        .map(|c| if c.is_ascii_alphanumeric() || matches!(c, '.' | '_' | '+' | '-') { c } else { '_' })
        .collect();
    while out.ends_with('.') {
        out.pop();
    }
    if out.is_empty() {
        out = "_".into();
    }
    out
}

/// The folders an entry owns, lower-case and `/`-separated (mirrors `Layout::owned_dir_keys`):
/// `models/<family>/<id rest>` and, for runtime packs, `runtimes/<id rest>`.
fn owned_dir_keys(entry: &Value) -> Vec<String> {
    let id = entry["id"].as_str().unwrap_or_default();
    let family = entry["family"].as_str().unwrap_or_default();
    let segs: Vec<&str> = id.split('/').collect();
    let rest = if segs.len() > 1 && segs[0] == family { &segs[1..] } else { &segs[..] };
    let rest = rest.iter().map(|s| sanitize_dir_name(s)).collect::<Vec<_>>().join("/");
    let mut keys = vec![format!("models/{}/{rest}", sanitize_dir_name(family)).to_lowercase()];
    if entry["kind"].as_str() == Some("runtime") && !LEGACY_FAMILIES.contains(&family) {
        keys.push(format!("runtimes/{rest}").to_lowercase());
    }
    keys
}

/// No two entries may share an install folder, and none may sit inside another's (the same rule
/// as `model_store::catalog::validate`, so a bad fragment fails the build).
fn check_entry_folders(entries: &[Value]) {
    let mut dirs: BTreeMap<String, String> = BTreeMap::new();
    for entry in entries {
        let id = entry["id"].as_str().unwrap_or_default();
        for key in owned_dir_keys(entry) {
            if let Some(other) = dirs.insert(key, id.to_string()) {
                panic!("{MODEL_CATALOG_DIR}: entry {id:?} installs into the same folder as {other:?}");
            }
        }
    }
    for (key, id) in &dirs {
        let mut ancestor = key.as_str();
        while let Some((parent, _)) = ancestor.rsplit_once('/') {
            if let Some(other) = dirs.get(parent).filter(|o| *o != id) {
                panic!("{MODEL_CATALOG_DIR}: entry {id:?} installs inside the folder of {other:?} ({parent})");
            }
            ancestor = parent;
        }
    }
}

/// Validate every `model-catalog/*.json` fragment and write the merged catalog to `$OUT_DIR`.
fn build_model_catalog() {
    let dir = Path::new(MODEL_CATALOG_DIR);
    println!("cargo:rerun-if-changed={MODEL_CATALOG_DIR}");
    let release = std::env::var("PROFILE").as_deref() == Ok("release");
    let mut files: Vec<_> = fs::read_dir(dir)
        .unwrap_or_else(|e| panic!("cannot read {MODEL_CATALOG_DIR}/: {e}"))
        .map(|entry| entry.expect("model-catalog entry").path())
        .filter(|p| p.extension().and_then(|e| e.to_str()) == Some("json"))
        .filter(|p| {
            !p.file_name()
                .and_then(|n| n.to_str())
                .is_some_and(|n| n.starts_with('_'))
        })
        .collect();
    files.sort();

    let mut merged = Vec::new();
    let mut ids = BTreeSet::new();
    for path in &files {
        println!("cargo:rerun-if-changed={}", path.display());
        let ctx = path.display().to_string();
        let text = fs::read_to_string(path).unwrap_or_else(|e| panic!("cannot read {ctx}: {e}"));
        let doc: Value = serde_json::from_str(&text).unwrap_or_else(|e| panic!("{ctx}: invalid JSON: {e}"));
        let obj = doc
            .as_object()
            .unwrap_or_else(|| panic!("{ctx}: a fragment must be a JSON object"));
        check_keys(&ctx, obj, &["$schema", "$comment", "entries", "routing"], &["entries"]);
        // `routing` (P2-S0): a data table read by `speech::router` with `include_str!` (it is not
        // part of the embedded catalog). Only its shape is checked here.
        if let Some(routing) = obj.get("routing") {
            if !routing.is_object() {
                panic!("{ctx}: routing must be a JSON object");
            }
        }
        let entries = obj["entries"]
            .as_array()
            .unwrap_or_else(|| panic!("{ctx}: entries must be an array"));
        for (i, entry) in entries.iter().enumerate() {
            let id = validate_catalog_entry(&format!("{ctx} entry #{i}"), entry, release);
            if !ids.insert(id.clone()) {
                panic!("{ctx}: duplicate catalog id {id:?} (ids are unique across all fragments)");
            }
            merged.push(entry.clone());
        }
    }
    for entry in &merged {
        for req in entry.get("requires").and_then(Value::as_array).into_iter().flatten() {
            let req = req.as_str().unwrap_or_default();
            if !ids.contains(req) {
                panic!(
                    "{MODEL_CATALOG_DIR}: entry {:?} requires unknown id {req:?}",
                    entry["id"].as_str().unwrap_or_default()
                );
            }
        }
    }
    check_entry_folders(&merged);

    let out_dir = std::env::var("OUT_DIR").expect("OUT_DIR is set by cargo");
    let out = Path::new(&out_dir).join(MODEL_CATALOG_OUT);
    let json = serde_json::to_string(&serde_json::json!({ "entries": merged }))
        .expect("serialize the merged model catalog");
    // Only rewrite on change, so an unchanged catalog does not force a recompile.
    if fs::read_to_string(&out).ok().as_deref() != Some(json.as_str()) {
        fs::write(&out, json).unwrap_or_else(|e| panic!("cannot write {}: {e}", out.display()));
    }
}

fn main() {
    build_model_catalog();
    tauri_build::try_build(
        tauri_build::Attributes::new()
            .app_manifest(tauri_build::AppManifest::new().commands(&[
                "get_native_features",
                "get_port",
                "app_close",
                "oauth_loopback_start",
            ]))
            .plugins(load_inlined_plugins()),
    )
    .expect("failed to run tauri-build");
}
