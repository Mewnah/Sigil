//! Static assets for `/client`, OBS and the helper pages (the app's embedded dist), with an
//! `index.html` fallback for SPA routes such as `/client`.
//!
//! Framing: display pages (`/client`, `obs-captions.html`, the SPA) may be framed by any page so
//! overlay tools can embed them; they only show what the host sends. Pages with controls that act
//! on the host (`mic.html` starts speech input, `inputfield.html` sends text) may only be framed by
//! their own origin, so another site cannot overlay them and trick clicks (P0-I).
//!
//! Assets are public (no LAN token): they hold no data, and the WebSocket routes authenticate.
//! Page URLs on other machines carry the LAN token (`?token=`, P6-D), so every response sets
//! `Referrer-Policy: same-origin`: requests to other sites (emote CDNs, fonts) never see the URL.
//!
//! `/sw.js` is the one path that never falls back to `index.html`. Curses and early Sigil builds
//! registered a caching service worker there, and an OBS browser source keeps it between runs. The
//! browser re-checks that URL by itself; an `index.html` answer (`text/html`) fails the check and
//! leaves the old worker, and the old page, in place. So it is served as JavaScript, uncached: the
//! kill-switch worker from `public/sw.js`, which removes the old one. Nothing in Sigil registers it.

use std::sync::Arc;

use tauri::{AssetResolver, Runtime};
use warp::http::header::*;
use warp::http::{HeaderValue, Response, StatusCode};
use warp::{filters::BoxedFilter, path::FullPath, Reply};
use warp::{Filter, Rejection};

pub fn path<R: Runtime>(resolver: Arc<AssetResolver<R>>) -> BoxedFilter<(impl Reply,)> {
    warp::path::full()
        .and_then(move |path: FullPath| file_response(path, resolver.clone()))
        .boxed()
}

/// Asset path for a request path (aliases for the captions page).
fn asset_path(path: &str) -> &str {
    match path {
        "/obs-captions" | "/obs-captions/" | "/obs-captions.html" | "/obs-captions.html/" => "/obs-captions.html",
        p => p,
    }
}

/// `Referrer-Policy` of every served asset (page URLs may contain the LAN token).
const REFERRER_POLICY_VALUE: &str = "same-origin";

/// The URL an old Curses / early Sigil service worker was registered at (`register('/sw.js')`).
const KILL_SWITCH_PATH: &str = "/sw.js";
/// A worker script needs a JavaScript type; the resolver gives `text/javascript` for a `.js` asset.
const KILL_SWITCH_CONTENT_TYPE: &str = "text/javascript; charset=utf-8";
/// The browser must see a changed script on every update check.
const KILL_SWITCH_CACHE_CONTROL: &str = "no-cache";

/// Is this request path the kill-switch worker? Exact match only: the old worker lived at this URL.
fn is_kill_switch(path: &str) -> bool {
    path == KILL_SWITCH_PATH
}

/// Does the resolver's `mime_type` say JavaScript? When `/sw.js` is missing from the bundle the
/// resolver answers with the SPA fallback page (`text/html`), which must not be served as the worker.
fn is_javascript_mime(mime_type: &str) -> bool {
    let essence = mime_type.split(';').next().unwrap_or("").trim().to_ascii_lowercase();
    matches!(essence.as_str(), "text/javascript" | "application/javascript" | "application/x-javascript")
}

/// Does the body start like a page (`<!doctype html>`, `<html`) rather than a script? The resolver
/// types an asset by its content first, so this is the second guard against the SPA fallback.
fn looks_like_markup(body: &[u8]) -> bool {
    let body = body.strip_prefix(b"\xEF\xBB\xBF").unwrap_or(body);
    body.iter().find(|b| !b.is_ascii_whitespace()) == Some(&b'<')
}

/// `(Content-Type, Cache-Control)` of the response for the requested path. `None` means a 404: it
/// is the kill switch but the resolver has no script for it.
fn delivery(requested: &str, resolver_mime: &str, body: &[u8]) -> Option<(String, Option<&'static str>)> {
    if !is_kill_switch(requested) {
        return Some((resolver_mime.to_string(), None));
    }
    (is_javascript_mime(resolver_mime) && !looks_like_markup(body))
        .then(|| (KILL_SWITCH_CONTENT_TYPE.to_string(), Some(KILL_SWITCH_CACHE_CONTROL)))
}

/// `(Content-Security-Policy, X-Frame-Options)` for a served asset path.
fn frame_headers(asset: &str) -> (&'static str, Option<&'static str>) {
    match asset.to_ascii_lowercase().as_str() {
        "/mic.html" | "/inputfield.html" => ("frame-ancestors 'self'", Some("SAMEORIGIN")),
        _ => ("frame-ancestors *", None),
    }
}

/// The asset chosen for a request path.
struct Selected<'a> {
    /// The path whose frame headers apply (`/index.html` when the SPA fallback answered).
    served_path: &'a str,
    content_type: String,
    cache_control: Option<&'static str>,
    bytes: Vec<u8>,
}

const INDEX_PATH: &str = "/index.html";

/// Pick what answers `path`. `lookup` is the asset resolver: `path -> (mime type, bytes)`.
///
/// `None` is a 404. Tauri's resolver already falls back to `index.html` for a path that is not in
/// the bundle (the SPA routes), so the `None` arms below are only reached by a resolver that does
/// not; what keeps `/sw.js` from being answered with the page in production is `delivery`, which
/// refuses an answer that is not JavaScript. Tested against a lookup that behaves like Tauri's.
fn select_asset<'a>(path: &'a str, lookup: impl Fn(&str) -> Option<(String, Vec<u8>)>) -> Option<Selected<'a>> {
    let (served_path, (mime_type, bytes)) = match lookup(path) {
        Some(asset) => (path, asset),
        // `/sw.js` is never answered with the SPA page (see the module docs).
        None if is_kill_switch(path) => return None,
        None => (INDEX_PATH, lookup(INDEX_PATH)?),
    };
    let (content_type, cache_control) = delivery(path, &mime_type, &bytes)?;
    Some(Selected { served_path, content_type, cache_control, bytes })
}

async fn file_response<R: Runtime>(path: FullPath, resolver: Arc<AssetResolver<R>>) -> Result<impl Reply, Rejection> {
    let path = asset_path(path.as_str());
    let selected = select_asset(path, |p| resolver.get(p.to_string()).map(|asset| (asset.mime_type, asset.bytes)))
        .ok_or_else(warp::reject::not_found)?;
    Ok(build_response(selected.served_path, selected.content_type, selected.cache_control, selected.bytes))
}

/// The response for an asset: the same headers for every asset, plus `Cache-Control` when one is given.
fn build_response(
    served_path: &str,
    content_type: String,
    cache_control: Option<&'static str>,
    bytes: Vec<u8>,
) -> Result<Response<Vec<u8>>, warp::http::Error> {
    let (csp, frame_options) = frame_headers(served_path);
    let mut response = Response::builder()
        .status(StatusCode::OK)
        .header(ACCEPT_RANGES, HeaderValue::from_static("bytes"))
        .header(ACCESS_CONTROL_ALLOW_ORIGIN, HeaderValue::from_static("*"))
        .header(CONTENT_TYPE, content_type)
        .header(CONTENT_SECURITY_POLICY, HeaderValue::from_static(csp))
        .header(REFERRER_POLICY, HeaderValue::from_static(REFERRER_POLICY_VALUE));
    if let Some(frame_options) = frame_options {
        response = response.header(X_FRAME_OPTIONS, HeaderValue::from_static(frame_options));
    }
    if let Some(cache_control) = cache_control {
        response = response.header(CACHE_CONTROL, HeaderValue::from_static(cache_control));
    }
    response.body(bytes)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn interactive_pages_are_not_framable_by_other_sites() {
        for page in ["/mic.html", "/inputfield.html", "/MIC.HTML"] {
            assert_eq!(frame_headers(page), ("frame-ancestors 'self'", Some("SAMEORIGIN")), "{page}");
        }
        for page in ["/obs-captions.html", "/index.html", "/assets/index-abc.js"] {
            assert_eq!(frame_headers(page), ("frame-ancestors *", None), "{page}");
        }
    }

    #[test]
    fn referrer_policy_never_leaks_page_urls_cross_site() {
        // `same-origin`: a full Referer only to this server, nothing to other sites.
        assert_eq!(REFERRER_POLICY_VALUE, "same-origin");
        assert!(HeaderValue::from_static(REFERRER_POLICY_VALUE).to_str().is_ok());
    }

    #[test]
    fn captions_aliases() {
        for p in ["/obs-captions", "/obs-captions/", "/obs-captions.html", "/obs-captions.html/"] {
            assert_eq!(asset_path(p), "/obs-captions.html");
        }
        assert_eq!(asset_path("/client"), "/client");
    }

    #[test]
    fn only_the_exact_sw_path_is_the_kill_switch() {
        assert!(is_kill_switch("/sw.js"));
        for other in ["/SW.JS", "/sw.js/", "/sw.js.map", "/assets/sw.js", "/workbox-3f2a1b.js", "/client", "/index.html", "/"] {
            assert!(!is_kill_switch(other), "{other}");
        }
        // The path aliases never turn another path into it.
        assert_eq!(asset_path("/sw.js"), "/sw.js");
    }

    #[test]
    fn javascript_types_are_recognised_with_or_without_parameters() {
        for mime in ["text/javascript", "text/javascript; charset=utf-8", "Text/JavaScript", "application/javascript", " text/javascript ;charset=UTF-8"] {
            assert!(is_javascript_mime(mime), "{mime}");
        }
        for mime in ["text/html", "text/html; charset=utf-8", "application/json", "text/plain", "text/javascriptx", ""] {
            assert!(!is_javascript_mime(mime), "{mime}");
        }
    }

    #[test]
    fn the_kill_switch_is_javascript_and_never_cached() {
        // What the Tauri resolver gives for a `.js` asset (tauri-utils `MimeType::Js`).
        let expected = Some(("text/javascript; charset=utf-8".to_string(), Some("no-cache")));
        assert_eq!(delivery("/sw.js", "text/javascript", b"// kill switch"), expected);
        assert_eq!(delivery("/sw.js", "application/javascript", b"self.skipWaiting();"), expected);
    }

    #[test]
    fn the_kill_switch_is_never_the_index_html_fallback() {
        // Missing from the bundle: the resolver answers with the SPA page. That is a 404, not a 200 text/html.
        let page: &[u8] = b"<!doctype html><html><body></body></html>";
        assert_eq!(delivery("/sw.js", "text/html", page), None);
        assert_eq!(delivery("/sw.js", "text/html; charset=utf-8", page), None);
        assert_eq!(delivery("/sw.js", "", page), None);
        // Even typed as JavaScript (by its `.js` name), a body that is a page is not the worker.
        assert_eq!(delivery("/sw.js", "text/javascript", page), None);
        assert_eq!(delivery("/sw.js", "text/javascript", b"\n  <!DOCTYPE html>"), None);
    }

    #[test]
    fn scripts_do_not_look_like_markup() {
        for script in [&b"/* comment */\nself.skipWaiting();"[..], b"// comment", b"\xEF\xBB\xBF// bom", b"self.x = 1 < 2;", b""] {
            assert!(!looks_like_markup(script), "{:?}", String::from_utf8_lossy(script));
        }
        for page in [&b"<!doctype html>"[..], b"  \r\n<html>", b"\xEF\xBB\xBF<html>"] {
            assert!(looks_like_markup(page), "{:?}", String::from_utf8_lossy(page));
        }
    }

    #[test]
    fn other_assets_keep_the_resolver_type_and_get_no_cache_control() {
        for (path, mime) in [
            ("/index.html", "text/html"),
            ("/client", "text/html"),
            ("/assets/index-abc.js", "text/javascript"),
            ("/obs-captions.html", "text/html"),
            ("/images/favicon.ico", "image/vnd.microsoft.icon"),
        ] {
            assert_eq!(delivery(path, mime, b"<html>"), Some((mime.to_string(), None)), "{path}");
        }
    }

    const PAGE: &[u8] = b"<!doctype html><html><body></body></html>";
    const WORKER: &[u8] = b"// kill switch
self.skipWaiting();";

    type Files<'a> = &'a [(&'a str, &'a str, &'a [u8])];

    /// An asset lookup like the one in `file_response`: `path -> (mime type, bytes)`. With
    /// `spa_fallback` it behaves like Tauri's `AssetResolver::get`, which answers a path that is not
    /// in the bundle with `index.html` itself (so `file_response` never sees a miss for a route).
    fn lookup<'a>(files: Files<'a>, spa_fallback: bool) -> impl Fn(&str) -> Option<(String, Vec<u8>)> + 'a {
        move |path: &str| {
            let find = |wanted: &str| {
                files.iter().find(|(p, _, _)| *p == wanted).map(|(_, mime, bytes)| (mime.to_string(), bytes.to_vec()))
            };
            find(path).or_else(|| if spa_fallback { find("/index.html") } else { None })
        }
    }

    #[test]
    fn sw_js_in_the_bundle_is_served_as_the_uncached_worker() {
        let files: Files = &[("/index.html", "text/html", PAGE), ("/sw.js", "text/javascript", WORKER)];
        for spa_fallback in [true, false] {
            let hit = select_asset("/sw.js", lookup(files, spa_fallback)).expect("served");
            assert_eq!(hit.served_path, "/sw.js");
            assert_eq!(hit.content_type, "text/javascript; charset=utf-8");
            assert_eq!(hit.cache_control, Some("no-cache"));
            assert_eq!(hit.bytes, WORKER);
        }
    }

    #[test]
    fn sw_js_missing_from_the_bundle_is_a_404_and_never_index_html() {
        let files: Files = &[("/index.html", "text/html", PAGE)];
        // Tauri's own resolver answers the missing path with index.html (text/html); `delivery` refuses it.
        assert!(select_asset("/sw.js", lookup(files, true)).is_none());
        // A resolver without that fallback: the `None if is_kill_switch` arm refuses it before index.html is looked up.
        assert!(select_asset("/sw.js", lookup(files, false)).is_none());
        // No bundle at all.
        assert!(select_asset("/sw.js", lookup(&[], true)).is_none());
    }

    #[test]
    fn a_sw_js_that_is_really_a_page_is_a_404() {
        // Typed as JavaScript by its name, but the content is markup (a bad bundle).
        let files: Files = &[("/index.html", "text/html", PAGE), ("/sw.js", "text/javascript", PAGE)];
        assert!(select_asset("/sw.js", lookup(files, true)).is_none());
    }

    #[test]
    fn routes_still_get_the_page_and_other_assets_their_own_type() {
        let files: Files = &[
            ("/index.html", "text/html", PAGE),
            ("/mic.html", "text/html", b"<html>mic</html>"),
            ("/assets/index-abc.js", "text/javascript", b"export{}"),
        ];
        // With the SPA fallback inside the resolver (Tauri) the resolver's answer is used as it is.
        let route = select_asset("/client", lookup(files, true)).expect("route");
        assert_eq!((route.served_path, route.content_type.as_str(), route.cache_control), ("/client", "text/html", None));
        assert_eq!(route.bytes, PAGE);
        // Without it, the page is looked up under its own name.
        let route = select_asset("/client", lookup(files, false)).expect("route");
        assert_eq!((route.served_path, route.content_type.as_str(), route.cache_control), ("/index.html", "text/html", None));
        let script = select_asset("/assets/index-abc.js", lookup(files, true)).expect("script");
        assert_eq!((script.served_path, script.content_type.as_str(), script.cache_control), ("/assets/index-abc.js", "text/javascript", None));
        let mic = select_asset("/mic.html", lookup(files, true)).expect("mic");
        assert_eq!(frame_headers(mic.served_path), ("frame-ancestors 'self'", Some("SAMEORIGIN")));
    }

    #[test]
    fn nothing_is_served_without_a_bundle() {
        assert!(select_asset("/client", lookup(&[], true)).is_none());
        assert!(select_asset("/client", lookup(&[], false)).is_none());
    }

    fn header(response: &Response<Vec<u8>>, name: HeaderName) -> Option<&str> {
        response.headers().get(name).map(|v| v.to_str().unwrap())
    }

    #[test]
    fn kill_switch_response_headers() {
        let (content_type, cache_control) = delivery("/sw.js", "text/javascript", b"//").unwrap();
        let response = build_response("/sw.js", content_type, cache_control, b"//".to_vec()).unwrap();
        assert_eq!(response.status(), StatusCode::OK);
        assert_eq!(header(&response, CONTENT_TYPE), Some("text/javascript; charset=utf-8"));
        assert_eq!(header(&response, CACHE_CONTROL), Some("no-cache"));
        // Every other header of an asset stays.
        assert_eq!(header(&response, ACCEPT_RANGES), Some("bytes"));
        assert_eq!(header(&response, ACCESS_CONTROL_ALLOW_ORIGIN), Some("*"));
        assert_eq!(header(&response, CONTENT_SECURITY_POLICY), Some("frame-ancestors *"));
        assert_eq!(header(&response, REFERRER_POLICY), Some("same-origin"));
        assert_eq!(header(&response, X_FRAME_OPTIONS), None);
        assert_eq!(response.body(), b"//");
    }

    #[test]
    fn page_response_headers_are_unchanged() {
        let (content_type, cache_control) = delivery("/client", "text/html", b"<html>").unwrap();
        let response = build_response("/index.html", content_type, cache_control, b"<html>".to_vec()).unwrap();
        assert_eq!(header(&response, CONTENT_TYPE), Some("text/html"));
        assert_eq!(header(&response, CACHE_CONTROL), None);
        assert_eq!(header(&response, ACCEPT_RANGES), Some("bytes"));
        assert_eq!(header(&response, REFERRER_POLICY), Some("same-origin"));
        let mic = build_response("/mic.html", "text/html".to_string(), None, Vec::new()).unwrap();
        assert_eq!(header(&mic, CONTENT_SECURITY_POLICY), Some("frame-ancestors 'self'"));
        assert_eq!(header(&mic, X_FRAME_OPTIONS), Some("SAMEORIGIN"));
    }
}
