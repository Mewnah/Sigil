//! Catalog loading and runtime validation.
//!
//! The embedded catalog is validated by build.rs. A development override
//! ([`super::CATALOG_OVERRIDE_ENV`]: one fragment file, or a folder of fragments) gets the same
//! checks here at load time (unknown keys, id / family / path / hash formats, https sources, unique
//! ids, known `requires`), plus checks build.rs cannot do: no dependency cycles, no two entries
//! sharing an install folder and no entry folder nested inside another's. An override may use `http://` sources on loopback hosts, so a local
//! fixture or LAN test server can stand in for a mirror.

use super::layout::{valid_rel_path, Layout};
use super::{Catalog, CatalogEntry, CATALOG_OVERRIDE_ENV};
use serde_json::{Map, Value};
use std::collections::{BTreeMap, HashMap, HashSet};
use std::fs;
use std::path::Path;

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

/// Validation switches.
#[derive(Clone, Copy, Debug, Default)]
pub(crate) struct ValidateOpts {
    /// Accept `http://` sources whose host is loopback (dev overrides and tests).
    pub allow_http_loopback: bool,
}

/// `stt/silero-vad-v6.2`: `/`-separated segments of `[a-z0-9][a-z0-9._-]*`.
pub(crate) fn valid_id(id: &str) -> bool {
    !id.is_empty()
        && id.split('/').all(|seg| {
            seg.starts_with(|c: char| c.is_ascii_lowercase() || c.is_ascii_digit())
                && seg
                    .chars()
                    .all(|c| c.is_ascii_lowercase() || c.is_ascii_digit() || matches!(c, '.' | '_' | '-'))
        })
}

fn valid_family(f: &str) -> bool {
    f.starts_with(|c: char| c.is_ascii_lowercase() || c.is_ascii_digit())
        && f.chars().all(|c| c.is_ascii_lowercase() || c.is_ascii_digit() || matches!(c, '_' | '-'))
}

fn is_lower_hex(s: &str, len: usize) -> bool {
    s.len() == len && s.chars().all(|c| c.is_ascii_digit() || ('a'..='f').contains(&c))
}

/// `http://` on 127.0.0.0/8, `[::1]` or `localhost`.
pub(crate) fn is_loopback_http(url: &reqwest::Url) -> bool {
    if url.scheme() != "http" {
        return false;
    }
    let Some(host) = url.host_str() else { return false };
    let bare = host.trim_start_matches('[').trim_end_matches(']');
    match bare.parse::<std::net::IpAddr>() {
        Ok(ip) => ip.is_loopback(),
        Err(_) => bare.eq_ignore_ascii_case("localhost"),
    }
}

/// Whether a source URL is acceptable under `opts`.
pub(crate) fn valid_source_url(url: &str, opts: ValidateOpts) -> bool {
    match reqwest::Url::parse(url) {
        Ok(u) if u.scheme() == "https" => u.host_str().is_some_and(|h| !h.is_empty()),
        Ok(u) => opts.allow_http_loopback && is_loopback_http(&u),
        Err(_) => false,
    }
}

fn check_keys(ctx: &str, obj: &Map<String, Value>, allowed: &[&str]) -> Result<(), String> {
    match obj.keys().find(|k| !allowed.contains(&k.as_str())) {
        Some(k) => Err(format!("{ctx}: unknown key {k:?}")),
        None => Ok(()),
    }
}

/// Parse one fragment document (`{"entries":[...]}`, optional `$schema` / `$comment`), rejecting
/// unknown keys at every level.
pub(crate) fn parse_fragment(ctx: &str, text: &str) -> Result<Vec<CatalogEntry>, String> {
    let doc: Value = serde_json::from_str(text).map_err(|e| format!("{ctx}: invalid JSON: {e}"))?;
    let obj = doc.as_object().ok_or_else(|| format!("{ctx}: a fragment must be a JSON object"))?;
    check_keys(ctx, obj, &["$schema", "$comment", "entries"])?;
    let entries = obj
        .get("entries")
        .and_then(Value::as_array)
        .ok_or_else(|| format!("{ctx}: \"entries\" must be an array"))?;
    let mut out = Vec::with_capacity(entries.len());
    for (i, e) in entries.iter().enumerate() {
        let ectx = format!("{ctx} entry #{i}");
        let eo = e.as_object().ok_or_else(|| format!("{ectx}: an entry must be an object"))?;
        check_keys(&ectx, eo, ENTRY_KEYS)?;
        for (j, f) in eo.get("files").and_then(Value::as_array).into_iter().flatten().enumerate() {
            let fctx = format!("{ectx} file #{j}");
            let fo = f.as_object().ok_or_else(|| format!("{fctx}: a file must be an object"))?;
            check_keys(&fctx, fo, FILE_KEYS)?;
            for s in fo.get("sources").and_then(Value::as_array).into_iter().flatten() {
                check_keys(&fctx, s.as_object().ok_or_else(|| format!("{fctx}: a source must be an object"))?, SOURCE_KEYS)?;
            }
            if let Some(a) = fo.get("archive") {
                check_keys(&fctx, a.as_object().ok_or_else(|| format!("{fctx}: archive must be an object"))?, ARCHIVE_KEYS)?;
            }
        }
        out.push(serde_json::from_value::<CatalogEntry>(e.clone()).map_err(|err| format!("{ectx}: {err}"))?);
    }
    Ok(out)
}

/// Load an override: a fragment file, or every `*.json` (not starting with `_`) in a folder.
pub(crate) fn load_override(path: &Path) -> Result<Catalog, String> {
    let ctx = format!("{CATALOG_OVERRIDE_ENV}={}", path.display());
    let mut files = Vec::new();
    if path.is_dir() {
        let rd = fs::read_dir(path).map_err(|e| format!("{ctx}: {e}"))?;
        for e in rd.flatten() {
            let p = e.path();
            let name = e.file_name().to_string_lossy().into_owned();
            if p.is_file() && name.ends_with(".json") && !name.starts_with('_') {
                files.push(p);
            }
        }
        files.sort();
    } else if path.is_file() {
        files.push(path.to_path_buf());
    } else {
        return Err(format!("{ctx}: no such file or folder"));
    }
    let mut entries = Vec::new();
    for f in &files {
        let text = fs::read_to_string(f).map_err(|e| format!("{}: {e}", f.display()))?;
        entries.extend(parse_fragment(&f.display().to_string(), &text)?);
    }
    let catalog = Catalog { entries };
    validate(&catalog, ValidateOpts { allow_http_loopback: true }).map_err(|e| format!("{ctx}: {e}"))?;
    Ok(catalog)
}

/// The catalog in effect: the override when [`CATALOG_OVERRIDE_ENV`] is set, else the embedded one.
pub(crate) fn load_effective() -> Result<Catalog, String> {
    match std::env::var_os(CATALOG_OVERRIDE_ENV).filter(|v| !v.is_empty()) {
        Some(p) => load_override(Path::new(&p)),
        None => Catalog::embedded(),
    }
}

/// Semantic checks on a whole catalog.
pub(crate) fn validate(catalog: &Catalog, opts: ValidateOpts) -> Result<(), String> {
    let mut ids = HashSet::new();
    for e in &catalog.entries {
        let ctx = format!("entry {:?}", e.id);
        if !valid_id(&e.id) {
            return Err(format!("{ctx}: id must be '/'-separated segments of [a-z0-9][a-z0-9._-]*"));
        }
        if !ids.insert(e.id.as_str()) {
            return Err(format!("{ctx}: duplicate id"));
        }
        if !valid_family(&e.family) {
            return Err(format!("{ctx}: invalid family {:?}", e.family));
        }
        if e.version.trim().is_empty() {
            return Err(format!("{ctx}: empty version"));
        }
        if e.files.is_empty() {
            return Err(format!("{ctx}: no files"));
        }
        let mut paths = HashSet::new();
        for f in &e.files {
            if !valid_rel_path(&f.path) {
                return Err(format!("{ctx}: unsafe file path {:?}", f.path));
            }
            if !paths.insert(f.path.to_lowercase()) {
                return Err(format!("{ctx}: duplicate file path {:?}", f.path));
            }
            if f.sha256.is_some() && f.git_oid.is_some() {
                return Err(format!("{ctx}: {:?} has both sha256 and git_oid", f.path));
            }
            if f.sha256.as_deref().is_some_and(|h| !is_lower_hex(h, 64)) {
                return Err(format!("{ctx}: {:?} sha256 must be 64 lower-case hex digits", f.path));
            }
            if f.git_oid.as_deref().is_some_and(|h| !is_lower_hex(h, 40)) {
                return Err(format!("{ctx}: {:?} git_oid must be 40 lower-case hex digits", f.path));
            }
            if f.sources.is_empty() {
                return Err(format!("{ctx}: {:?} has no sources", f.path));
            }
            if let Some(s) = f.sources.iter().find(|s| !valid_source_url(&s.url, opts)) {
                return Err(format!("{ctx}: {:?} source {:?} must be an https:// URL", f.path, s.url));
            }
        }
    }
    // Every folder an entry owns (`models/<family>/<id rest>`, `runtimes/<id rest>` for packs):
    // no two entries may share one, and none may sit inside another's (the startup sweep and
    // installs replace or delete unreferenced folders under an entry's own folders).
    let mut dirs: HashMap<String, &str> = HashMap::new();
    for e in &catalog.entries {
        for key in Layout::owned_dir_keys(e) {
            if let Some(other) = dirs.insert(key, &e.id) {
                return Err(format!("entry {:?}: installs into the same folder as {other:?}", e.id));
            }
        }
    }
    for (key, id) in &dirs {
        let mut ancestor = key.as_str();
        while let Some((parent, _)) = ancestor.rsplit_once('/') {
            if let Some(other) = dirs.get(parent).filter(|o| **o != *id) {
                return Err(format!("entry {id:?}: installs inside the folder of {other:?} ({parent})"));
            }
            ancestor = parent;
        }
    }
    for e in &catalog.entries {
        if let Some(r) = e.requires.iter().find(|r| !ids.contains(r.as_str())) {
            return Err(format!("entry {:?} requires unknown id {r:?}", e.id));
        }
    }
    let all: Vec<String> = catalog.entries.iter().map(|e| e.id.clone()).collect();
    install_order(catalog, &all).map(|_| ())
}

/// `ids` plus everything they require, dependencies first, each once. Unknown ids and cycles are
/// errors (with a "did you mean" hint).
pub(crate) fn install_order(catalog: &Catalog, ids: &[String]) -> Result<Vec<String>, String> {
    #[derive(Clone, Copy, PartialEq)]
    enum Mark {
        Visiting,
        Done,
    }
    fn visit(
        catalog: &Catalog,
        id: &str,
        marks: &mut HashMap<String, Mark>,
        out: &mut Vec<String>,
        path: &mut Vec<String>,
    ) -> Result<(), String> {
        match marks.get(id) {
            Some(Mark::Done) => return Ok(()),
            Some(Mark::Visiting) => {
                path.push(id.to_string());
                return Err(format!("the catalog has a dependency cycle: {}", path.join(" -> ")));
            }
            None => {}
        }
        let entry = catalog.get(id).ok_or_else(|| unknown_id(catalog, id))?;
        marks.insert(id.to_string(), Mark::Visiting);
        path.push(id.to_string());
        for r in &entry.requires {
            visit(catalog, r, marks, out, path)?;
        }
        path.pop();
        marks.insert(id.to_string(), Mark::Done);
        out.push(id.to_string());
        Ok(())
    }
    let mut marks = HashMap::new();
    let mut out = Vec::new();
    for id in ids {
        visit(catalog, id.trim(), &mut marks, &mut out, &mut Vec::new())?;
    }
    Ok(out)
}

/// "unknown catalog id" with the closest id as a hint.
pub(crate) fn unknown_id(catalog: &Catalog, id: &str) -> String {
    match suggest(catalog, id) {
        Some(s) => format!("unknown catalog id {id:?} (did you mean {s:?}?)"),
        None => format!("unknown catalog id {id:?}"),
    }
}

fn punctuation_insensitive(s: &str) -> String {
    s.to_ascii_lowercase()
        .chars()
        .map(|c| if c.is_ascii_alphanumeric() { c } else { '/' })
        .collect()
}

/// The single catalog id equal to `id` up to case and punctuation (`base.test-small` →
/// `base/test-small`); `None` when there is no such id or more than one.
pub(crate) fn punctuation_match(catalog: &Catalog, id: &str) -> Option<String> {
    let want = punctuation_insensitive(id.trim());
    let mut hits = catalog.entries.iter().filter(|e| punctuation_insensitive(&e.id) == want);
    let first = hits.next()?;
    hits.next().is_none().then(|| first.id.clone())
}

/// The catalog id closest to `id` (punctuation-insensitive first, then edit distance ≤ 3).
pub(crate) fn suggest(catalog: &Catalog, id: &str) -> Option<String> {
    if let Some(m) = punctuation_match(catalog, id) {
        return Some(m);
    }
    catalog
        .entries
        .iter()
        .map(|e| (edit_distance(&e.id, &id.to_ascii_lowercase()), &e.id))
        .filter(|(d, _)| *d <= 3)
        .min_by_key(|(d, _)| *d)
        .map(|(_, id)| id.clone())
}

fn edit_distance(a: &str, b: &str) -> usize {
    let b: Vec<char> = b.chars().collect();
    let mut prev: Vec<usize> = (0..=b.len()).collect();
    for (i, ca) in a.chars().enumerate() {
        let mut cur = vec![i + 1; b.len() + 1];
        for (j, cb) in b.iter().enumerate() {
            cur[j + 1] = (prev[j] + usize::from(ca != *cb)).min(prev[j + 1] + 1).min(cur[j] + 1);
        }
        prev = cur;
    }
    prev[b.len()]
}

/// Entries grouped by family (for listings).
pub(crate) fn by_family(catalog: &Catalog) -> BTreeMap<&str, Vec<&CatalogEntry>> {
    let mut m: BTreeMap<&str, Vec<&CatalogEntry>> = BTreeMap::new();
    for e in &catalog.entries {
        m.entry(e.family.as_str()).or_default().push(e);
    }
    m
}

#[cfg(test)]
mod tests {
    use super::*;

    fn entry_json(id: &str, requires: &[&str], url: &str) -> String {
        let req = if requires.is_empty() {
            String::new()
        } else {
            format!(r#","requires":[{}]"#, requires.iter().map(|r| format!("{r:?}")).collect::<Vec<_>>().join(","))
        };
        format!(
            r#"{{"id":"{id}","family":"t","kind":"model","version":"1","license":"MIT","status":"published"{req},
            "files":[{{"path":"f.bin","size":3,"sources":[{{"kind":"direct","url":"{url}"}}]}}]}}"#
        )
    }

    fn catalog(entries: &[String]) -> Catalog {
        Catalog::from_json(&format!(r#"{{"entries":[{}]}}"#, entries.join(","))).unwrap()
    }

    #[test]
    fn embedded_catalog_passes_runtime_validation() {
        let c = Catalog::embedded().unwrap();
        validate(&c, ValidateOpts::default()).unwrap();
        assert!(c.get("base/test-small").is_some(), "the self-test entry ships in base.json");
    }

    #[test]
    fn install_order_puts_dependencies_first_and_detects_cycles() {
        let c = catalog(&[
            entry_json("t/app", &["t/runtime", "t/weights"], "https://x.invalid/a"),
            entry_json("t/runtime", &[], "https://x.invalid/r"),
            entry_json("t/weights", &["t/runtime"], "https://x.invalid/w"),
        ]);
        let order = install_order(&c, &["t/app".into(), "t/runtime".into()]).unwrap();
        assert_eq!(order, ["t/runtime", "t/weights", "t/app"]);
        let err = install_order(&c, &["t/nope".into()]).unwrap_err();
        assert!(err.contains("unknown catalog id"), "{err}");

        let cyc = catalog(&[
            entry_json("t/a", &["t/b"], "https://x.invalid/a"),
            entry_json("t/b", &["t/a"], "https://x.invalid/b"),
        ]);
        let err = validate(&cyc, ValidateOpts::default()).unwrap_err();
        assert!(err.contains("cycle"), "{err}");
    }

    #[test]
    fn validation_rejects_bad_urls_paths_and_shared_folders() {
        let http = catalog(&[entry_json("t/a", &[], "http://127.0.0.1:9/a")]);
        assert!(validate(&http, ValidateOpts::default()).is_err());
        validate(&http, ValidateOpts { allow_http_loopback: true }).unwrap();
        let lan = catalog(&[entry_json("t/a", &[], "http://192.168.1.5/a")]);
        assert!(validate(&lan, ValidateOpts { allow_http_loopback: true }).is_err());
        // `t/x` (family t) and `x` (family t) would share models/t/x.
        let clash = catalog(&[
            entry_json("t/x", &[], "https://x.invalid/a"),
            entry_json("x", &[], "https://x.invalid/b"),
        ]);
        let err = validate(&clash, ValidateOpts::default()).unwrap_err();
        assert!(err.contains("same folder"), "{err}");
    }

    #[test]
    fn validation_rejects_entry_folders_inside_other_entries() {
        // models/t/sherpa/zh would sit inside models/t/sherpa (the sweep would delete it).
        let nested = catalog(&[
            entry_json("t/sherpa", &[], "https://x.invalid/a"),
            entry_json("t/sherpa/zh", &[], "https://x.invalid/b"),
        ]);
        let err = validate(&nested, ValidateOpts::default()).unwrap_err();
        assert!(err.contains("inside the folder of"), "{err}");
        // Siblings that only share a name prefix are fine.
        let siblings = catalog(&[
            entry_json("t/sherpa", &[], "https://x.invalid/a"),
            entry_json("t/sherpa-zh", &[], "https://x.invalid/b"),
            entry_json("t/sherpa.zh", &[], "https://x.invalid/c"),
        ]);
        validate(&siblings, ValidateOpts::default()).unwrap();
        // Runtime packs of two families would share runtimes/x/.
        let pack = |id: &str, family: &str| {
            entry_json(id, &[], "https://x.invalid/p")
                .replace(r#""family":"t""#, &format!(r#""family":"{family}""#))
                .replace(r#""kind":"model""#, r#""kind":"runtime""#)
        };
        let shared = catalog(&[pack("runtime/x", "runtime"), pack("rt/x", "rt")]);
        let err = validate(&shared, ValidateOpts::default()).unwrap_err();
        assert!(err.contains("same folder"), "{err}");
        let nested_pack = catalog(&[pack("runtime/x", "runtime"), pack("rt/x/cuda", "rt")]);
        let err = validate(&nested_pack, ValidateOpts::default()).unwrap_err();
        assert!(err.contains("inside the folder of"), "{err}");
    }

    #[test]
    fn fragments_reject_unknown_keys() {
        let ok = format!(r#"{{"$schema":"./_schema.json","$comment":["x"],"entries":[{}]}}"#, entry_json("t/a", &[], "https://x.invalid/a"));
        assert_eq!(parse_fragment("ok", &ok).unwrap().len(), 1);
        let typo = ok.replace("\"license\"", "\"licence\"");
        let err = parse_fragment("typo", &typo).unwrap_err();
        assert!(err.contains("licence"), "{err}");
        let src = ok.replace("\"kind\":\"direct\"", "\"kind\":\"direct\",\"mirror\":1");
        assert!(parse_fragment("src", &src).is_err());
    }

    #[test]
    fn override_loads_a_file_or_a_folder() {
        let d = std::env::temp_dir().join(format!("sigil-ms-override-{}-{}", std::process::id(), super::super::layout::nonce()));
        fs::create_dir_all(&d).unwrap();
        fs::write(d.join("a.json"), format!(r#"{{"entries":[{}]}}"#, entry_json("t/a", &[], "http://127.0.0.1:1/a"))).unwrap();
        fs::write(d.join("b.json"), format!(r#"{{"entries":[{}]}}"#, entry_json("t/b", &["t/a"], "https://x.invalid/b"))).unwrap();
        fs::write(d.join("_schema.json"), "not a fragment").unwrap();
        let c = load_override(&d).unwrap();
        assert_eq!(c.entries.len(), 2);
        let single = load_override(&d.join("a.json")).unwrap();
        assert_eq!(single.entries.len(), 1);
        fs::write(d.join("c.json"), format!(r#"{{"entries":[{}]}}"#, entry_json("t/a", &[], "https://x.invalid/dup"))).unwrap();
        assert!(load_override(&d).unwrap_err().contains("duplicate"));
        assert!(load_override(&d.join("missing.json")).is_err());
        let _ = fs::remove_dir_all(&d);
    }

    #[test]
    fn suggestions_catch_punctuation_and_typos() {
        let c = catalog(&[entry_json("base/test-small", &[], "https://x.invalid/a"), entry_json("t/whisper-base", &[], "https://x.invalid/b")]);
        assert_eq!(suggest(&c, "base.test-small").as_deref(), Some("base/test-small"));
        assert_eq!(punctuation_match(&c, "BASE.test_small").as_deref(), Some("base/test-small"));
        assert_eq!(punctuation_match(&c, "base/test-smal"), None);
        assert_eq!(suggest(&c, "t/whisper-bse").as_deref(), Some("t/whisper-base"));
        assert_eq!(suggest(&c, "completely-different"), None);
    }
}
