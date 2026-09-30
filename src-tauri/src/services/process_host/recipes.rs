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

// ---------------------------------------------------------------------------------------------
// Bergamot (P4-S0 recipe type; the helper and its client are P4-E's)
// ---------------------------------------------------------------------------------------------

/// Helper name of the Bergamot CPU translator.
pub const BERGAMOT_NAME: &str = "bergamot";
/// Executable inside the Bergamot runtime pack (built by `scripts/runtime-packs/bergamot`, P4-E).
pub const BERGAMOT_EXE: &str = "sigil-bergamot.exe";
/// Catalog id of the runtime pack (`model-catalog/runtime-bergamot.json`; local-only, unpublished).
pub const BERGAMOT_RUNTIME_ID: &str = "runtime/bergamot";

/// Knobs for [`bergamot`] (frozen by P4-S0; fields may be added).
///
/// The helper's command line is `sigil-bergamot.exe --jsonl [--threads N] [--cache-size N]
/// [extra_args...]`: requests are JSON lines on stdin, replies JSON lines on stdout (one per
/// request, in any order, matched by id), diagnostics on stderr (piped to the log). EOF on stdin
/// means quit. The message schema is P4-E's (ADR 0008); model files are always passed by absolute
/// path in a load message, so the helper never searches folders itself.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct BergamotOptions {
    /// Helper name (one helper per name).
    pub name: String,
    /// `--threads`: translation worker threads (`None` = the helper's default, 1).
    pub threads: Option<u32>,
    /// `--cache-size`: translation cache entries (`None` = the helper's default).
    pub cache_size: Option<u32>,
    /// Extra arguments appended last (validated by the caller).
    pub extra_args: Vec<String>,
    /// Job memory cap: models are ~20–40 MB per direction, so 1.5 GB only stops a runaway.
    pub memory_limit_mb: Option<u64>,
    /// CPU helper next to a game and the voice changer: below normal by default.
    pub priority: HelperPriority,
}

impl Default for BergamotOptions {
    fn default() -> Self {
        Self {
            name: BERGAMOT_NAME.into(),
            threads: None,
            cache_size: None,
            extra_args: Vec::new(),
            memory_limit_mb: Some(1536),
            priority: HelperPriority::BelowNormal,
        }
    }
}

/// The Bergamot helper binary inside an installed runtime pack folder.
pub fn bergamot_exe(pack_dir: &Path) -> PathBuf {
    pack_dir.join(BERGAMOT_EXE)
}

/// Spec for the Bergamot JSON-lines helper:
/// - no port, no API key, no HTTP health check (`StdioMode::Protocol`: the client takes stdin and
///   stdout with `process_host::take_stdio` and must keep reading stdout);
/// - restart on failure (3 times, 500 ms backoff doubling), EcoQoS off (captions are latency
///   sensitive), below-normal priority and a job memory cap by default;
/// - 500 ms grace after stdin EOF before the job is terminated.
pub fn bergamot(exe: &Path, opts: &BergamotOptions) -> HelperSpec {
    let mut args: Vec<String> = vec!["--jsonl".into()];
    if let Some(n) = opts.threads {
        args.push("--threads".into());
        args.push(n.to_string());
    }
    if let Some(n) = opts.cache_size {
        args.push("--cache-size".into());
        args.push(n.to_string());
    }
    args.extend(opts.extra_args.iter().cloned());
    HelperSpec {
        name: opts.name.clone(),
        program: exe.to_path_buf(),
        args,
        env: Vec::new(),
        env_remove: Vec::new(),
        cwd: exe.parent().map(Path::to_path_buf),
        port: PortRequest::None,
        api_key: false,
        health: None,
        memory_limit_mb: opts.memory_limit_mb,
        restart: RestartPolicy::OnFailure {
            max_restarts: 3,
            backoff_ms: 500,
        },
        priority: opts.priority,
        high_qos: true,
        stop_grace_ms: 500,
        stdio: super::StdioMode::Protocol,
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

    #[test]
    fn bergamot_spec_is_a_stdio_helper_without_port_or_key() {
        let exe = Path::new("C:/rt/bergamot/sigil-bergamot.exe");
        let spec = bergamot(
            exe,
            &BergamotOptions {
                threads: Some(2),
                ..Default::default()
            },
        );
        assert_eq!(spec.name, BERGAMOT_NAME);
        assert_eq!(spec.args, ["--jsonl", "--threads", "2"]);
        assert_eq!(spec.stdio, crate::services::process_host::StdioMode::Protocol);
        assert_eq!(spec.port, PortRequest::None);
        assert!(!spec.api_key && spec.health.is_none() && spec.env.is_empty());
        assert_eq!(spec.priority, HelperPriority::BelowNormal);
        assert_eq!(spec.memory_limit_mb, Some(1536));
        assert!(spec.high_qos && spec.stop_grace_ms == 500);
        assert!(matches!(spec.restart, RestartPolicy::OnFailure { max_restarts: 3, .. }));
        assert_eq!(spec.cwd.as_deref(), Some(Path::new("C:/rt/bergamot")));
        assert_eq!(bergamot_exe(Path::new("C:/rt")), Path::new("C:/rt").join("sigil-bergamot.exe"));
        let err = super::super::spawn::validate(&spec).unwrap_err();
        assert!(err.contains("program not found"), "{err}");
    }
}
