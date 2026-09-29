//! Download sources per file, in the order they are tried.
//!
//! - Catalog order first (the catalog lists Hugging Face, hf-mirror, ModelScope, Sigil GitHub
//!   release, ...). For every `huggingface.co` source without an explicit hf-mirror sibling, the
//!   equivalent `hf-mirror.com` URL is appended: same path, same pinned revision, and the hash check
//!   still applies, so friends where the Hub is slow or blocked get a working mirror for free.
//! - [`MirrorPreference`] other than `auto` moves that family to the front (stable order otherwise);
//!   `custom` puts `<custom_url>/<entry id>/<file path>` first, with the catalog sources behind it.
//! - `unpublished` entries never go to the catalog URLs (nothing is released there yet): a local
//!   import or local assets satisfy them, and the custom mirror only for files the catalog pins by
//!   hash. Unpinned files of unpublished entries (today's Sigil-built runtime packs, which carry
//!   DLLs) get no network source at all, in any build: trust on first use never covers executables,
//!   and a user-picked import is the only trusted route for them.

use super::catalog::{valid_source_url, ValidateOpts};
use super::{CatalogEntry, CatalogFile, MirrorPreference, PublishStatus, SourceKind, StoreConfig};
use crate::services::http_client::{is_local_endpoint, normalize_base_url};
use reqwest::Url;

pub(crate) const HF_HOST: &str = "huggingface.co";
pub(crate) const HF_MIRROR_HOST: &str = "hf-mirror.com";

/// One concrete URL to try.
#[derive(Clone, Debug, PartialEq, Eq)]
pub(crate) struct ResolvedSource {
    pub kind: SourceKind,
    pub url: Url,
    /// Not listed in the catalog (hf-mirror twin or the custom mirror).
    pub derived: bool,
}

/// The hf-mirror twin of a Hub URL.
pub(crate) fn hf_mirror_url(url: &Url) -> Option<Url> {
    if url.scheme() != "https" || url.host_str() != Some(HF_HOST) {
        return None;
    }
    let mut u = url.clone();
    u.set_host(Some(HF_MIRROR_HOST)).ok()?;
    Some(u)
}

/// `<base>/<entry id>/<file path>` with every segment percent-encoded.
pub(crate) fn custom_source_url(base: &str, entry_id: &str, file_path: &str) -> Option<Url> {
    let mut u = Url::parse(base).ok()?;
    {
        let mut segs = u.path_segments_mut().ok()?;
        segs.pop_if_empty();
        for s in entry_id.split('/').chain(file_path.split('/')) {
            segs.push(s);
        }
    }
    Some(u)
}

fn preferred_kind(pref: MirrorPreference) -> Option<SourceKind> {
    match pref {
        MirrorPreference::Auto => None,
        MirrorPreference::Huggingface => Some(SourceKind::Huggingface),
        MirrorPreference::HfMirror => Some(SourceKind::HfMirror),
        MirrorPreference::Modelscope => Some(SourceKind::Modelscope),
        MirrorPreference::GithubRelease => Some(SourceKind::GithubRelease),
        MirrorPreference::Custom => Some(SourceKind::Custom),
    }
}

/// The ordered sources for `file` of `entry` under `config`.
pub(crate) fn sources_for(
    entry: &CatalogEntry,
    file: &CatalogFile,
    config: &StoreConfig,
    opts: ValidateOpts,
) -> Vec<ResolvedSource> {
    let mut list: Vec<ResolvedSource> = Vec::new();
    if entry.status == PublishStatus::Published {
        for s in &file.sources {
            if !valid_source_url(&s.url, opts) {
                continue;
            }
            if let Ok(url) = Url::parse(&s.url) {
                list.push(ResolvedSource {
                    kind: s.kind,
                    url,
                    derived: false,
                });
            }
        }
        if !list.iter().any(|s| s.kind == SourceKind::HfMirror) {
            let twins: Vec<ResolvedSource> = list
                .iter()
                .filter(|s| s.kind == SourceKind::Huggingface)
                .filter_map(|s| hf_mirror_url(&s.url))
                .map(|url| ResolvedSource {
                    kind: SourceKind::HfMirror,
                    url,
                    derived: true,
                })
                .collect();
            list.extend(twins);
        }
    }
    let pinned = file.sha256.is_some() || file.git_oid.is_some();
    if config.mirror == MirrorPreference::Custom && (entry.status == PublishStatus::Published || pinned) {
        if let Some(url) = config
            .custom_url
            .as_deref()
            .and_then(|base| custom_source_url(base, &entry.id, &file.path))
        {
            list.insert(
                0,
                ResolvedSource {
                    kind: SourceKind::Custom,
                    url,
                    derived: true,
                },
            );
        }
    } else if let Some(kind) = preferred_kind(config.mirror) {
        // Stable: preferred family first, everything else keeps catalog order.
        list.sort_by_key(|s| s.kind != kind);
    }
    let mut seen = std::collections::HashSet::new();
    list.retain(|s| seen.insert(s.url.as_str().to_string()));
    list
}

/// `host:port` of a URL (one download per host at a time).
pub(crate) fn host_key(url: &Url) -> String {
    format!(
        "{}:{}",
        url.host_str().unwrap_or("").to_ascii_lowercase(),
        url.port_or_known_default().unwrap_or(0)
    )
}

/// Parse a `--mirror` / `modelMirror` value.
pub(crate) fn parse_mirror(raw: &str) -> Result<MirrorPreference, String> {
    serde_json::from_value(serde_json::Value::String(raw.trim().to_ascii_lowercase())).map_err(|_| {
        format!("unknown mirror {raw:?} (auto, huggingface, hf-mirror, modelscope, github-release, custom)")
    })
}

/// Validate a custom mirror base URL: https anywhere, or http on this PC or the local network
/// (every file is still hash-checked). Returns it without trailing slashes.
pub(crate) fn validate_custom_url(raw: &str) -> Result<String, String> {
    let trimmed = raw.trim();
    if !trimmed.contains("://") {
        return Err(format!("custom mirror {trimmed:?} must start with https:// (or http:// on your own network)"));
    }
    let base = normalize_base_url(trimmed)?;
    let url = Url::parse(&base).map_err(|e| format!("custom mirror {trimmed:?}: {e}"))?;
    if url.query().is_some() || url.fragment().is_some() || !url.username().is_empty() || url.password().is_some() {
        return Err("the custom mirror URL must not contain credentials, a query or a fragment".into());
    }
    if url.scheme() == "http" && !is_local_endpoint(&url) {
        return Err("a custom mirror on the internet must use https://".into());
    }
    Ok(base)
}

/// Validate and normalize a [`StoreConfig`] pushed by the UI or the CLI.
pub(crate) fn normalize_config(mut cfg: StoreConfig) -> Result<StoreConfig, String> {
    cfg.max_concurrent = cfg.max_concurrent.clamp(1, 2);
    let raw = cfg.custom_url.take().filter(|u| !u.trim().is_empty());
    cfg.custom_url = match (cfg.mirror, raw) {
        (MirrorPreference::Custom, None) => {
            return Err("choose a custom mirror URL, or pick another mirror".into());
        }
        (MirrorPreference::Custom, Some(u)) => Some(validate_custom_url(&u)?),
        // Kept (when valid) so switching back to "custom" needs no retyping.
        (_, Some(u)) => validate_custom_url(&u).ok(),
        (_, None) => None,
    };
    Ok(cfg)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::services::model_store::Catalog;

    fn entry(status: &str) -> CatalogEntry {
        let json = format!(
            r#"{{"entries":[{{"id":"stt/demo","family":"stt","kind":"model","version":"1","license":"MIT","status":"{status}",
            "files":[{{"path":"sub/m.onnx","size":1,"sources":[
              {{"kind":"huggingface","url":"https://huggingface.co/org/repo/resolve/0123456789abcdef0123456789abcdef01234567/sub/m.onnx"}},
              {{"kind":"modelscope","url":"https://www.modelscope.cn/models/org/repo/resolve/master/sub/m.onnx"}},
              {{"kind":"github-release","url":"https://github.com/Mewnah/Sigil/releases/download/assets-v1/m.onnx"}}]}}]}}]}}"#
        );
        Catalog::from_json(&json).unwrap().entries.remove(0)
    }

    fn kinds(list: &[ResolvedSource]) -> Vec<SourceKind> {
        list.iter().map(|s| s.kind).collect()
    }

    #[test]
    fn auto_keeps_catalog_order_and_appends_hf_mirror() {
        let e = entry("published");
        let list = sources_for(&e, &e.files[0], &StoreConfig::default(), ValidateOpts::default());
        assert_eq!(
            kinds(&list),
            [SourceKind::Huggingface, SourceKind::Modelscope, SourceKind::GithubRelease, SourceKind::HfMirror]
        );
        assert_eq!(
            list[3].url.as_str(),
            "https://hf-mirror.com/org/repo/resolve/0123456789abcdef0123456789abcdef01234567/sub/m.onnx"
        );
        assert!(list[3].derived);
    }

    #[test]
    fn preference_moves_a_family_first_and_custom_goes_in_front() {
        let e = entry("published");
        let cfg = StoreConfig {
            mirror: MirrorPreference::HfMirror,
            ..StoreConfig::default()
        };
        let list = sources_for(&e, &e.files[0], &cfg, ValidateOpts::default());
        assert_eq!(list[0].kind, SourceKind::HfMirror);
        assert_eq!(list[1].kind, SourceKind::Huggingface);

        let cfg = normalize_config(StoreConfig {
            mirror: MirrorPreference::Custom,
            custom_url: Some(" http://192.168.1.10:8080/sigil/ ".into()),
            max_concurrent: 9,
        })
        .unwrap();
        assert_eq!(cfg.max_concurrent, 2);
        assert_eq!(cfg.custom_url.as_deref(), Some("http://192.168.1.10:8080/sigil"));
        let list = sources_for(&e, &e.files[0], &cfg, ValidateOpts::default());
        assert_eq!(list[0].kind, SourceKind::Custom);
        assert_eq!(list[0].url.as_str(), "http://192.168.1.10:8080/sigil/stt/demo/sub/m.onnx");
        assert_eq!(list.len(), 5);
    }

    #[test]
    fn unpublished_entries_only_use_the_custom_mirror_for_pinned_files() {
        let mut e = entry("unpublished");
        assert!(sources_for(&e, &e.files[0], &StoreConfig::default(), ValidateOpts::default()).is_empty());
        let cfg = StoreConfig {
            mirror: MirrorPreference::Custom,
            custom_url: Some("https://mirror.example/assets".into()),
            max_concurrent: 1,
        };
        // Unpinned (a Sigil-built pack): import or local assets only, even with a custom mirror.
        assert!(sources_for(&e, &e.files[0], &cfg, ValidateOpts::default()).is_empty());
        e.files[0].sha256 = Some("a".repeat(64));
        let list = sources_for(&e, &e.files[0], &cfg, ValidateOpts::default());
        assert_eq!(kinds(&list), [SourceKind::Custom]);
        // Published entries keep the custom mirror for unpinned (development) files.
        let p = entry("published");
        assert_eq!(sources_for(&p, &p.files[0], &cfg, ValidateOpts::default())[0].kind, SourceKind::Custom);
    }

    #[test]
    fn custom_urls_are_validated() {
        assert!(validate_custom_url("https://mirror.example/x/").is_ok());
        assert!(validate_custom_url("http://127.0.0.1:9000").is_ok());
        assert!(validate_custom_url("http://nas.local/models").is_ok());
        for bad in ["http://mirror.example.com", "ftp://x", "mirror.example", "https://u:p@m.example", "https://m.example/?k=1"] {
            assert!(validate_custom_url(bad).is_err(), "{bad}");
        }
        assert!(normalize_config(StoreConfig {
            mirror: MirrorPreference::Custom,
            custom_url: None,
            max_concurrent: 1
        })
        .is_err());
        // A bad URL is dropped (not an error) while another mirror is selected.
        let cfg = normalize_config(StoreConfig {
            mirror: MirrorPreference::Auto,
            custom_url: Some("http://public.example.com".into()),
            max_concurrent: 0,
        })
        .unwrap();
        assert_eq!(cfg.custom_url, None);
        assert_eq!(cfg.max_concurrent, 1);
        assert_eq!(parse_mirror("HF-Mirror").unwrap(), MirrorPreference::HfMirror);
        assert!(parse_mirror("nope").is_err());
    }

    #[test]
    fn host_keys_include_the_port() {
        let a = Url::parse("https://huggingface.co/a").unwrap();
        let b = Url::parse("http://127.0.0.1:8080/b").unwrap();
        assert_eq!(host_key(&a), "huggingface.co:443");
        assert_eq!(host_key(&b), "127.0.0.1:8080");
    }
}
