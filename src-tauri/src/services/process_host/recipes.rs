//! Ready-made [`HelperSpec`]s for helpers Sigil ships or supports. A recipe encodes the security
//! and isolation choices (loopback bind, auto port, key in the environment, health route); the
//! owning feature (e.g. P4-B `llm_server`) picks the model and the performance knobs.
// The first consumer is P4-B (`llm_server`); until then the recipe is exercised by tests and by
// the manual `llama_server_smoke` test.
#![allow(dead_code)]

use super::{HealthCheck, HelperPriority, HelperSpec, PortRequest, RestartPolicy};
use std::path::{Path, PathBuf};

/// Helper name of the translation LLM server.
pub const LLAMA_SERVER_NAME: &str = "llama-server";
/// Executable inside a llama.cpp runtime pack (`runtime/llama-server-*` catalog entries).
pub const LLAMA_SERVER_EXE: &str = "llama-server.exe";

/// Performance knobs for [`llama_server`] (all optional; `None` keeps llama-server's default).
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct LlamaServerOptions {
    /// Helper name (one llama-server per name).
    pub name: String,
    /// `-c`: context size in tokens (translation lines are short; 4096 leaves room for glossary and
    /// rolling context).
    pub ctx_size: Option<u32>,
    /// `-ngl`: layers offloaded to the GPU (`999` = all; `0` for the CPU pack).
    pub gpu_layers: Option<u32>,
    /// `-fa on|off|auto`.
    pub flash_attn: Option<String>,
    /// `-np`: parallel slots (1 keeps VRAM and latency predictable for one caption stream).
    pub parallel: Option<u32>,
    /// `-t`: CPU threads for generation.
    pub threads: Option<u32>,
    /// `--alias`: the model name reported by `/v1/models`.
    pub alias: Option<String>,
    /// Extra arguments appended last (validated by the caller).
    pub extra_args: Vec<String>,
    /// Health timeout: model loading from disk (a 7B Q4 from a hard disk can take a minute).
    pub load_timeout_ms: u64,
    pub priority: HelperPriority,
    /// Job memory cap (committed memory; mmap'd model weights do not count).
    pub memory_limit_mb: Option<u64>,
}

impl Default for LlamaServerOptions {
    fn default() -> Self {
        Self {
            name: LLAMA_SERVER_NAME.into(),
            ctx_size: Some(4096),
            gpu_layers: None,
            flash_attn: None,
            parallel: Some(1),
            threads: None,
            alias: None,
            extra_args: Vec::new(),
            load_timeout_ms: 120_000,
            priority: HelperPriority::Normal,
            memory_limit_mb: None,
        }
    }
}

/// The llama.cpp `llama-server` binary inside an installed runtime pack folder.
pub fn llama_server_exe(pack_dir: &Path) -> PathBuf {
    pack_dir.join(LLAMA_SERVER_EXE)
}

/// Spec for `llama-server` serving `model` (a GGUF file) on `127.0.0.1:{port}`:
/// - `--host 127.0.0.1 --port {port}` on the command line (command-line arguments take precedence
///   over llama.cpp's `LLAMA_ARG_*` variables, which are removed from the inherited environment
///   anyway);
/// - the random key through `LLAMA_API_KEY` (never on the command line, which every local process
///   can read); requests need `Authorization: Bearer <key>` (llama-server keeps `/health` and the model
///   list public);
/// - health `GET /health` (503 while the model loads, 200 when ready);
/// - restart on failure (3 times, 1 s backoff doubling), EcoQoS off, immediate stop (no state).
pub fn llama_server(exe: &Path, model: &Path, opts: &LlamaServerOptions) -> HelperSpec {
    let mut args: Vec<String> = vec![
        "--host".into(),
        "127.0.0.1".into(),
        "--port".into(),
        super::spawn::PORT_PLACEHOLDER.into(),
        "-m".into(),
        model.display().to_string(),
    ];
    let mut push = |flag: &str, value: Option<String>| {
        if let Some(v) = value {
            args.push(flag.into());
            args.push(v);
        }
    };
    push("-c", opts.ctx_size.map(|v| v.to_string()));
    push("-ngl", opts.gpu_layers.map(|v| v.to_string()));
    push("-fa", opts.flash_attn.clone());
    push("-np", opts.parallel.map(|v| v.to_string()));
    push("-t", opts.threads.map(|v| v.to_string()));
    push("--alias", opts.alias.clone());
    args.extend(opts.extra_args.iter().cloned());
    HelperSpec {
        name: opts.name.clone(),
        program: exe.to_path_buf(),
        args,
        env: vec![("LLAMA_API_KEY".into(), super::spawn::API_KEY_PLACEHOLDER.into())],
        env_remove: vec!["LLAMA_ARG_*".into(), "LLAMA_API_KEY".into(), "LLAMA_API_KEY_FILE".into()],
        cwd: exe.parent().map(Path::to_path_buf),
        port: PortRequest::Auto,
        api_key: true,
        health: Some(HealthCheck {
            path: "/health".into(),
            timeout_ms: opts.load_timeout_ms,
            interval_ms: 250,
            send_api_key: false,
        }),
        memory_limit_mb: opts.memory_limit_mb,
        restart: RestartPolicy::OnFailure {
            max_restarts: 3,
            backoff_ms: 1000,
        },
        priority: opts.priority,
        high_qos: true,
        stop_grace_ms: 0,
        stdio: super::StdioMode::Log,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn llama_server_spec_binds_loopback_and_keeps_the_key_off_the_command_line() {
        let exe = Path::new("C:/rt/llama-server.exe");
        let spec = llama_server(
            exe,
            Path::new("C:/m/hy-mt2-1.8b-q4.gguf"),
            &LlamaServerOptions {
                gpu_layers: Some(999),
                flash_attn: Some("auto".into()),
                ..Default::default()
            },
        );
        assert_eq!(spec.name, "llama-server");
        let joined = spec.args.join(" ");
        assert!(joined.starts_with("--host 127.0.0.1 --port {port} -m "), "{joined}");
        assert!(joined.contains("-ngl 999") && joined.contains("-fa auto") && joined.contains("-c 4096"));
        assert!(!joined.contains("api"), "the key must not be on the command line: {joined}");
        assert_eq!(spec.env, [("LLAMA_API_KEY".to_string(), "{api_key}".to_string())]);
        assert!(spec.api_key && spec.port == PortRequest::Auto && spec.high_qos);
        assert_eq!(spec.health.as_ref().unwrap().path, "/health");
        assert!(spec.env_remove.iter().any(|p| p == "LLAMA_ARG_*"));
        assert_eq!(llama_server_exe(Path::new("C:/rt")), Path::new("C:/rt").join("llama-server.exe"));
        // Structurally valid (the program does not exist here, so only the path check fails).
        let err = super::super::spawn::validate(&spec).unwrap_err();
        assert!(err.contains("program not found"), "{err}");
    }
}
