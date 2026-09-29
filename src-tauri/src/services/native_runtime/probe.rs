//! `Sigil.exe --headless engine-probe <engine> [model] [wav] [--variant v] [--runtime-dir dir]
//! [--out file] [-- extra...]` for sigil_engine DLLs (echo, and any engine without its own probe).
//!
//! Loads the engine from `--runtime-dir` or its installed packs (variant fallback as in the app),
//! creates an instance, then exercises every capability the engine reports on the WAV (or a 1 s,
//! 440 Hz test tone at 16 kHz): `query {"op":"info"}`, `process`, `render`, a streaming session in
//! 20 ms chunks, and for test engines (`kind: "test"`) a cancelled long call. The JSON report
//! carries timings (load, create, per stage, real-time factor), every failed variant and the
//! engine's answers. For the echo engine the rendered audio must equal the input times its gain.
//!
//! Extra arguments after `--`: `--config <json object>` and/or `key=value` pairs merged into the
//! create config (values are parsed as JSON when they parse, else taken as strings), e.g.
//! `-- gain=0.5`.

use std::path::Path;
use std::time::{Duration, Instant};

use serde_json::{json, Map, Value};

use super::abi;
use super::loader::{AudioView, EngineInfo, EngineInstance};
use super::pack::{open_engine, OpenRequest};
use super::variant::RuntimeVariant;
use crate::services::speech_rt::EngineProbeArgs;

/// Longest WAV the probe reads.
const MAX_WAV_SECONDS: u64 = 600;
/// Stream chunk length.
const CHUNK_MS: u32 = 20;

fn ms(d: Duration) -> f64 {
    (d.as_secs_f64() * 1e6).round() / 1e3
}

/// The create config: `{"model": <model>}` (when given) plus the extra arguments.
pub(crate) fn build_config(model: Option<&Path>, extra: &[String]) -> Result<Map<String, Value>, String> {
    let mut config = Map::new();
    if let Some(m) = model {
        config.insert("model".into(), Value::String(m.display().to_string()));
    }
    let mut it = extra.iter();
    while let Some(arg) = it.next() {
        if arg == "--config" {
            let text = it.next().ok_or("--config needs a JSON object")?;
            match serde_json::from_str::<Value>(text) {
                Ok(Value::Object(o)) => config.extend(o),
                Ok(_) => return Err("--config must be a JSON object".into()),
                Err(e) => return Err(format!("--config is not valid JSON: {e}")),
            }
        } else if let Some((k, v)) = arg.split_once('=') {
            let key = k.trim().trim_start_matches("--");
            if key.is_empty() {
                return Err(format!("invalid extra argument {arg:?}"));
            }
            let value = serde_json::from_str::<Value>(v).unwrap_or_else(|_| Value::String(v.to_string()));
            config.insert(key.to_string(), value);
        } else {
            return Err(format!(
                "unknown extra argument {arg:?} (use --config <json> or key=value)"
            ));
        }
    }
    Ok(config)
}

/// Interleaved f32 audio from a WAV file (any PCM or float format hound reads).
pub(crate) fn read_wav(path: &Path) -> Result<(Vec<f32>, u32, u16), String> {
    let mut reader = hound::WavReader::open(path).map_err(|e| format!("cannot read {}: {e}", path.display()))?;
    let spec = reader.spec();
    if spec.channels == 0 || spec.sample_rate == 0 {
        return Err(format!("{} has no audio", path.display()));
    }
    let max = u64::from(spec.sample_rate) * u64::from(spec.channels) * MAX_WAV_SECONDS;
    if u64::from(reader.len()) > max {
        return Err(format!("{} is longer than {MAX_WAV_SECONDS} s", path.display()));
    }
    let samples: Result<Vec<f32>, hound::Error> = match spec.sample_format {
        hound::SampleFormat::Float => reader.samples::<f32>().collect(),
        hound::SampleFormat::Int => {
            let scale = 1.0 / (1u64 << (spec.bits_per_sample.clamp(1, 32) - 1)) as f32;
            reader.samples::<i32>().map(|s| s.map(|v| v as f32 * scale)).collect()
        }
    };
    let samples = samples.map_err(|e| format!("cannot decode {}: {e}", path.display()))?;
    Ok((samples, spec.sample_rate, spec.channels))
}

/// 1 s of a 440 Hz sine at 16 kHz mono, amplitude 0.5.
pub(crate) fn test_tone() -> (Vec<f32>, u32, u16) {
    let sr = 16_000u32;
    let samples = (0..sr)
        .map(|i| 0.5 * (2.0 * std::f32::consts::PI * 440.0 * i as f32 / sr as f32).sin())
        .collect();
    (samples, sr, 1)
}

fn parse_json(text: &str) -> Value {
    serde_json::from_str(text).unwrap_or_else(|_| Value::String(text.to_string()))
}

fn stage_error(e: impl std::fmt::Display) -> Value {
    json!({ "ok": false, "error": e.to_string() })
}

/// Largest |a - b| over two equally long signals (`None` when the lengths differ).
pub(crate) fn max_abs_diff(a: &[f32], b: &[f32]) -> Option<f32> {
    (a.len() == b.len()).then(|| a.iter().zip(b).fold(0.0f32, |m, (x, y)| m.max((x - y).abs())))
}

fn run_stages(
    info: &EngineInfo,
    instance: &mut EngineInstance,
    config: &Map<String, Value>,
    audio: &[f32],
    sample_rate: u32,
    channels: u16,
) -> (Map<String, Value>, bool) {
    let mut report = Map::new();
    let mut ok = true;
    let seconds = audio.len() as f64 / f64::from(sample_rate) / f64::from(channels);
    let view = AudioView {
        samples: audio,
        sample_rate,
        channels,
    };

    let t = Instant::now();
    match instance.query(r#"{"op":"info"}"#) {
        Ok(r) => {
            report.insert("query".into(), json!({ "ok": true, "ms": ms(t.elapsed()), "result": parse_json(&r) }));
        }
        Err(e) => {
            ok = false;
            report.insert("query".into(), stage_error(e));
        }
    }

    if info.has(abi::SIGIL_CAP_PROCESS) {
        let t = Instant::now();
        let stage = match instance.process(view, "{}") {
            Ok(r) => {
                let elapsed = t.elapsed();
                json!({ "ok": true, "ms": ms(elapsed), "rtf": elapsed.as_secs_f64() / seconds.max(1e-9), "result": parse_json(&r) })
            }
            Err(e) => {
                ok = false;
                stage_error(e)
            }
        };
        report.insert("process".into(), stage);
    }

    if info.has(abi::SIGIL_CAP_RENDER) {
        let t = Instant::now();
        let stage = match instance.render(Some(view), "{}") {
            Ok(out) => {
                let elapsed = t.elapsed();
                let mut stage = json!({
                    "ok": true,
                    "ms": ms(elapsed),
                    "rtf": elapsed.as_secs_f64() / seconds.max(1e-9),
                    "frames": out.frames(),
                    "sample_rate": out.sample_rate,
                    "channels": out.channels,
                });
                if info.name == "echo" {
                    // The echo engine returns the input times its gain.
                    let gain = config.get("gain").and_then(Value::as_f64).unwrap_or(1.0) as f32;
                    let expected: Vec<f32> = audio.iter().map(|s| s * gain).collect();
                    let diff = max_abs_diff(&out.samples, &expected);
                    let pass = diff.is_some_and(|d| d <= 1e-6) && out.sample_rate == sample_rate && out.channels == channels;
                    ok &= pass;
                    stage["echo_check"] = json!({ "pass": pass, "gain": gain, "max_abs_diff": diff });
                }
                stage
            }
            Err(e) => {
                ok = false;
                stage_error(e)
            }
        };
        report.insert("render".into(), stage);
    }

    if info.has(abi::SIGIL_CAP_STREAM) {
        let t = Instant::now();
        let stage = match instance.open_stream("{}") {
            Err(e) => {
                ok = false;
                stage_error(e)
            }
            Ok(mut stream) => {
                let chunk = (sample_rate * CHUNK_MS / 1000).max(1) as usize * usize::from(channels);
                let (mut chunks, mut partials, mut last_partial, mut failure) = (0usize, 0usize, Value::Null, None);
                for block in audio.chunks(chunk) {
                    let block = &block[..block.len() - block.len() % usize::from(channels)];
                    match stream.push(AudioView {
                        samples: block,
                        sample_rate,
                        channels,
                    }) {
                        Ok(p) => {
                            chunks += 1;
                            if let Some(p) = p {
                                partials += 1;
                                last_partial = parse_json(&p);
                            }
                        }
                        Err(e) => {
                            failure = Some(e);
                            break;
                        }
                    }
                }
                match failure {
                    Some(e) => {
                        ok = false;
                        stage_error(e)
                    }
                    None => match stream.finish() {
                        Ok(f) => {
                            let elapsed = t.elapsed();
                            json!({
                                "ok": true,
                                "ms": ms(elapsed),
                                "rtf": elapsed.as_secs_f64() / seconds.max(1e-9),
                                "chunks": chunks,
                                "chunk_ms": CHUNK_MS,
                                "partials": partials,
                                "last_partial": last_partial,
                                "final": parse_json(&f),
                            })
                        }
                        Err(e) => {
                            ok = false;
                            stage_error(e)
                        }
                    },
                }
            }
        };
        report.insert("stream".into(), stage);
    }

    if info.has(abi::SIGIL_CAP_CANCEL) && info.kind == "test" {
        // Test engines implement {"op":"sleep","ms":N}; cancel it from another thread.
        let canceller = instance.canceller();
        let fire = std::thread::spawn(move || {
            std::thread::sleep(Duration::from_millis(50));
            canceller.cancel();
        });
        let t = Instant::now();
        let result = instance.query_engine(r#"{"op":"sleep","ms":5000}"#);
        let elapsed = t.elapsed();
        let _ = fire.join();
        let pass = matches!(&result, Err(e) if e.is_cancelled()) && elapsed < Duration::from_secs(2);
        ok &= pass;
        report.insert(
            "cancel".into(),
            json!({ "pass": pass, "ms": ms(elapsed), "result": match result { Ok(r) => parse_json(&r), Err(e) => Value::String(e.message) } }),
        );
    }
    (report, ok)
}

/// Body of `Sigil.exe --headless engine-probe echo|<engine> ...` for sigil_engine DLLs: load the
/// engine (from `--runtime-dir` or its installed pack), run a smoke call, return a JSON report.
pub fn headless_probe(args: &EngineProbeArgs) -> Result<String, String> {
    let engine = args.engine.trim();
    let variant = args
        .variant
        .as_deref()
        .map(|v| {
            RuntimeVariant::parse(v.trim())
                .ok_or_else(|| format!("unknown variant {v:?} (cpu, cpu-avx2, cpu-avx512, vulkan, cuda13)"))
        })
        .transpose()?;
    let config = build_config(args.model.as_deref(), &args.extra)?;
    let (audio, sample_rate, channels, input) = match &args.wav {
        Some(p) => {
            let (a, sr, ch) = read_wav(p)?;
            (a, sr, ch, p.display().to_string())
        }
        None => {
            let (a, sr, ch) = test_tone();
            (a, sr, ch, "test tone (440 Hz, 1 s, 16 kHz mono)".to_string())
        }
    };
    let config_json = Value::Object(config.clone()).to_string();

    let started = Instant::now();
    let opened = open_engine(
        engine,
        &OpenRequest {
            variant,
            runtime_dir: args.runtime_dir.clone(),
            config_json: Some(config_json),
        },
    )?;
    let open_ms = ms(started.elapsed());
    let info = opened.library.info().clone();
    let mut instance = opened
        .instance
        .ok_or_else(|| format!("{engine}: no instance was created"))?;

    let (stages, ok) = run_stages(&info, &mut instance, &config, &audio, sample_rate, channels);
    drop(instance);

    let mut report = Map::new();
    report.insert("engine".into(), json!(engine));
    report.insert("ok".into(), json!(ok));
    report.insert("sigil_version".into(), json!(env!("CARGO_PKG_VERSION")));
    report.insert("host_abi_version".into(), json!(abi::SIGIL_ENGINE_ABI_VERSION));
    report.insert(
        "selected".into(),
        json!({
            "variant": opened.variant.map(RuntimeVariant::as_str),
            "pack_id": opened.pack_id,
            "dir": opened.dir.display().to_string(),
            "library": opened.library.path().display().to_string(),
        }),
    );
    report.insert("attempts".into(), serde_json::to_value(&opened.attempts).unwrap_or(Value::Null));
    report.insert(
        "info".into(),
        json!({
            "abi_version": info.abi_version,
            "name": info.name,
            "version": info.version,
            "variant": info.variant,
            "kind": info.kind,
            "build_info": info.build_info,
            "capabilities": info.capability_names(),
        }),
    );
    report.insert("config".into(), Value::Object(config));
    report.insert(
        "input".into(),
        json!({
            "source": input,
            "sample_rate": sample_rate,
            "channels": channels,
            "seconds": audio.len() as f64 / f64::from(sample_rate) / f64::from(channels),
        }),
    );
    report.insert("open_ms".into(), json!(open_ms));
    report.extend(stages);
    let text = serde_json::to_string_pretty(&Value::Object(report)).map_err(|e| e.to_string())?;
    if ok {
        Ok(text)
    } else {
        Err(format!("{engine}: a probe stage failed\n{text}"))
    }
}
