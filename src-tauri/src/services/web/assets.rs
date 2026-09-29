//! Static assets for `/client`, OBS and the helper pages (the app's embedded dist), with an
//! `index.html` fallback for SPA routes such as `/client`.
//!
//! Framing: display pages (`/client`, `obs-captions.html`, the SPA) may be framed by any page so
//! overlay tools can embed them; they only show what the host sends. Pages with controls that act
//! on the host (`mic.html` starts speech input, `inputfield.html` sends text) may only be framed by
//! their own origin, so another site cannot overlay them and trick clicks (P0-I).

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

/// `(Content-Security-Policy, X-Frame-Options)` for a served asset path.
fn frame_headers(asset: &str) -> (&'static str, Option<&'static str>) {
    match asset.to_ascii_lowercase().as_str() {
        "/mic.html" | "/inputfield.html" => ("frame-ancestors 'self'", Some("SAMEORIGIN")),
        _ => ("frame-ancestors *", None),
    }
}

async fn file_response<R: Runtime>(path: FullPath, resolver: Arc<AssetResolver<R>>) -> Result<impl Reply, Rejection> {
    let path = asset_path(path.as_str());

    let (asset, served_path) = match resolver.get(path.to_string()) {
        Some(asset) => (asset, path),
        None => match resolver.get("/index.html".to_string()) {
            Some(index) => (index, "/index.html"),
            None => return Err(warp::reject::not_found()),
        },
    };

    let (csp, frame_options) = frame_headers(served_path);
    let mut response = Response::builder()
        .status(StatusCode::OK)
        .header(ACCEPT_RANGES, HeaderValue::from_static("bytes"))
        .header(ACCESS_CONTROL_ALLOW_ORIGIN, HeaderValue::from_static("*"))
        .header(CONTENT_TYPE, asset.mime_type)
        .header(CONTENT_SECURITY_POLICY, HeaderValue::from_static(csp));
    if let Some(frame_options) = frame_options {
        response = response.header(X_FRAME_OPTIONS, HeaderValue::from_static(frame_options));
    }
    Ok(response.body(asset.bytes))
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
    fn captions_aliases() {
        for p in ["/obs-captions", "/obs-captions/", "/obs-captions.html", "/obs-captions.html/"] {
            assert_eq!(asset_path(p), "/obs-captions.html");
        }
        assert_eq!(asset_path("/client"), "/client");
    }
}
