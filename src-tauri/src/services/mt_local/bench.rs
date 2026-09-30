//! MT benchmark: `Sigil.exe --headless mt-bench <spec.json> [--out <dir>]` runs a local sentence corpus (never
//! downloaded here) through translation engines and writes `report.json`, `report.md` and the hypotheses
//! (`hyps/<engine>.<src>-<tgt>.jsonl`, for listening-style blind comparisons) with corpus-level chrF and latency
//! (p50 / p95 / mean / max) per engine and direction.
//!
//! # Spec (JSON; paths relative to the spec file)
//! ```json
//! {
//!   "corpus": "corpus.sample.jsonl",
//!   "directions": ["en-ja", "ja-en"],
//!   "limit": 200,
//!   "warmup": 1,
//!   "engines": [
//!     { "engine": "nllb", "id": "nllb-600m", "model": "mt/nllb-200-distilled-600m-int8", "threads": 2, "beam": 1 },
//!     { "engine": "nllb", "id": "nllb-local", "path": "C:/models/nllb-600m" },
//!     { "engine": "openai", "id": "hy-mt2-1.8b", "baseUrl": "http://127.0.0.1:8080/v1", "model": "hy-mt2",
//!       "apiKeyEnv": "LLAMA_API_KEY", "prompt": "hy-mt2", "maxTokens": 512, "temperature": 0.2 },
//!     { "engine": "hypotheses", "id": "bergamot", "path": "bergamot-hyps.jsonl" }
//!   ]
//! }
//! ```
//! - `corpus`: JSON lines `{"id"?, "source": "en", "target": "ja", "src": "...", "ref": "..." | "refs": [...]}`
//!   (canonical BCP-47 codes). Lines without `id` get `<source>-<target>:<n>`.
//! - `directions` (optional) keeps only those `source-target` pairs; `limit` caps lines per direction; `warmup`
//!   translates that many lines per engine and direction untimed first.
//! - Engines: `nllb` (in process, the engine of the app: a catalog `model` installed in the model store of
//!   `SIGIL_APP_DATA_DIR`, or a folder `path`; `threads`, `beam`); `openai` (any OpenAI-compatible
//!   `/chat/completions` server: a llama-server started by hand or by Sigil, Ollama, LM Studio, a cloud API; the key
//!   is read from the environment variable `apiKeyEnv`, never from the spec; `prompt` = `generic`, `hy-mt2` or a
//!   template with `{source_lang}`, `{target_lang}` and `{text}`); `hypotheses` (a JSON-lines file of
//!   `{"id", "hyp"}` produced elsewhere, e.g. by the Bergamot helper; scored, no latency).
//!
//! # chrF
//! Corpus-level chrF2 exactly as sacrebleu 2.x's defaults (`CHRF(char_order=6, word_order=0, beta=2)`, whitespace
//! removed, effective-order averaging, best reference per line): see [`chrf`]. Failed lines count as empty
//! hypotheses and are reported as `errors`.
//!
//! Owned by P4-D. Placement note: the workstream named `bench/mt.rs`, but `services/bench/**` belongs to the
//! concurrently running P3 (voice engine), so the MT benchmark lives here as `mt_local::bench`; the CLI arm is in
//! `services/cli`.

use std::collections::HashMap;
use std::path::{Path, PathBuf};
use std::time::{Duration, Instant, SystemTime, UNIX_EPOCH};

use serde::{Deserialize, Serialize};

use super::flores;
use super::nllb::NllbModel;
use super::NLLB_DEFAULT_MODEL;

// ---------------------------------------------------------------------------------------------
// chrF (sacrebleu-compatible)
// ---------------------------------------------------------------------------------------------

/// Character n-gram order of chrF.
pub const CHAR_ORDER: usize = 6;
/// β of chrF2 (recall weighted twice as much as precision).
pub const BETA: f64 = 2.0;

/// Per-order `[hypothesis n-grams, reference n-grams, matches]`, summable over a corpus.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct ChrfStats([[u64; 3]; CHAR_ORDER]);

impl ChrfStats {
    pub fn add(&mut self, other: &ChrfStats) {
        for (a, b) in self.0.iter_mut().zip(other.0.iter()) {
            for (x, y) in a.iter_mut().zip(b) {
                *x += y;
            }
        }
    }
}

/// Character n-gram counts of orders 1..=[`CHAR_ORDER`], whitespace removed first (as sacrebleu does).
fn char_ngrams(text: &str) -> Vec<HashMap<String, u64>> {
    let chars: Vec<char> = text.split_whitespace().flat_map(str::chars).collect();
    (1..=CHAR_ORDER)
        .map(|n| {
            let mut counts: HashMap<String, u64> = HashMap::new();
            if chars.len() >= n {
                for w in chars.windows(n) {
                    *counts.entry(w.iter().collect()).or_insert(0) += 1;
                }
            }
            counts
        })
        .collect()
}

/// Statistics of one hypothesis against one reference.
pub fn chrf_stats(hyp: &str, reference: &str) -> ChrfStats {
    let h = char_ngrams(hyp);
    let r = char_ngrams(reference);
    let mut stats = ChrfStats::default();
    for ((slot, hg), rg) in stats.0.iter_mut().zip(&h).zip(&r) {
        let hyp_count: u64 = hg.values().sum();
        let ref_count: u64 = rg.values().sum();
        let matches: u64 = hg.iter().map(|(g, &c)| rg.get(g).map_or(0, |&rc| c.min(rc))).sum();
        *slot = [hyp_count, ref_count, matches];
    }
    stats
}

/// chrF (0–100) of summed statistics: sacrebleu 2.x `CHRF._compute_f_score` without eps smoothing, i.e. precision
/// and recall averaged over the orders where both the hypothesis and the reference have n-grams ("effective order").
pub fn chrf_score(stats: &ChrfStats) -> f64 {
    let factor = BETA * BETA;
    let (mut avg_prec, mut avg_rec, mut effective) = (0.0f64, 0.0f64, 0usize);
    for &[n_hyp, n_ref, n_match] in &stats.0 {
        if n_hyp > 0 && n_ref > 0 {
            avg_prec += n_match as f64 / n_hyp as f64;
            avg_rec += n_match as f64 / n_ref as f64;
            effective += 1;
        }
    }
    if effective == 0 {
        return 0.0;
    }
    avg_prec /= effective as f64;
    avg_rec /= effective as f64;
    if avg_prec + avg_rec == 0.0 {
        return 0.0;
    }
    100.0 * (1.0 + factor) * avg_prec * avg_rec / (factor * avg_prec + avg_rec)
}

/// Statistics of a hypothesis against the best of several references (the one with the highest sentence chrF).
pub fn chrf_stats_multi(hyp: &str, references: &[String]) -> ChrfStats {
    references
        .iter()
        .map(|r| chrf_stats(hyp, r))
        .max_by(|a, b| chrf_score(a).total_cmp(&chrf_score(b)))
        .unwrap_or_default()
}

/// Corpus chrF of `(hypothesis, references)` pairs.
pub fn chrf(pairs: &[(String, Vec<String>)]) -> f64 {
    let mut total = ChrfStats::default();
    for (hyp, refs) in pairs {
        total.add(&chrf_stats_multi(hyp, refs));
    }
    chrf_score(&total)
}

// ---------------------------------------------------------------------------------------------
// Spec and corpus
// ---------------------------------------------------------------------------------------------

#[derive(Clone, Debug, Deserialize, PartialEq)]
#[serde(rename_all = "camelCase", deny_unknown_fields)]
pub struct Spec {
    /// Free-form notes (ignored).
    #[serde(default, rename = "$comment")]
    pub _comment: Option<serde_json::Value>,
    pub corpus: PathBuf,
    #[serde(default)]
    pub directions: Vec<String>,
    #[serde(default)]
    pub limit: Option<usize>,
    #[serde(default = "default_warmup")]
    pub warmup: usize,
    pub engines: Vec<EngineSpec>,
    #[serde(default)]
    pub out: Option<PathBuf>,
}

fn default_warmup() -> usize {
    1
}

#[derive(Clone, Debug, Deserialize, PartialEq)]
#[serde(tag = "engine", rename_all = "camelCase", deny_unknown_fields)]
pub enum EngineSpec {
    #[serde(rename_all = "camelCase")]
    Nllb {
        id: Option<String>,
        model: Option<String>,
        path: Option<PathBuf>,
        threads: Option<u32>,
        beam: Option<u32>,
    },
    #[serde(rename_all = "camelCase")]
    Openai {
        id: String,
        base_url: String,
        model: String,
        api_key_env: Option<String>,
        prompt: Option<String>,
        max_tokens: Option<u32>,
        temperature: Option<f32>,
        timeout_ms: Option<u64>,
    },
    #[serde(rename_all = "camelCase")]
    Hypotheses { id: String, path: PathBuf },
}

impl EngineSpec {
    fn id(&self) -> String {
        match self {
            EngineSpec::Nllb { id, model, path, .. } => id.clone().unwrap_or_else(|| {
                path.as_ref()
                    .and_then(|p| p.file_name())
                    .map(|n| n.to_string_lossy().into_owned())
                    .or_else(|| model.clone())
                    .unwrap_or_else(|| NLLB_DEFAULT_MODEL.to_string())
            }),
            EngineSpec::Openai { id, .. } | EngineSpec::Hypotheses { id, .. } => id.clone(),
        }
    }

    fn kind(&self) -> &'static str {
        match self {
            EngineSpec::Nllb { .. } => "nllb",
            EngineSpec::Openai { .. } => "openai",
            EngineSpec::Hypotheses { .. } => "hypotheses",
        }
    }
}

/// One corpus line.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct CorpusLine {
    pub id: String,
    pub source: String,
    pub target: String,
    pub src: String,
    pub refs: Vec<String>,
}

impl CorpusLine {
    fn direction(&self) -> String {
        format!("{}-{}", self.source, self.target)
    }
}

#[derive(Deserialize)]
#[serde(rename_all = "camelCase")]
struct RawLine {
    id: Option<String>,
    source: String,
    target: String,
    src: String,
    #[serde(rename = "ref")]
    reference: Option<String>,
    refs: Option<Vec<String>>,
}

/// Parse a JSON-lines corpus (blank lines and `#` comments skipped).
pub fn parse_corpus(text: &str) -> Result<Vec<CorpusLine>, String> {
    let mut out = Vec::new();
    let mut counters: HashMap<String, usize> = HashMap::new();
    for (i, line) in text.lines().enumerate() {
        let line = line.trim();
        if line.is_empty() || line.starts_with('#') {
            continue;
        }
        let raw: RawLine = serde_json::from_str(line).map_err(|e| format!("corpus line {}: {e}", i + 1))?;
        let mut refs: Vec<String> = raw.refs.unwrap_or_default();
        refs.extend(raw.reference);
        refs.retain(|r| !r.trim().is_empty());
        if refs.is_empty() {
            return Err(format!("corpus line {}: no reference (\"ref\" or \"refs\")", i + 1));
        }
        if raw.src.trim().is_empty() {
            return Err(format!("corpus line {}: empty \"src\"", i + 1));
        }
        let direction = format!("{}-{}", raw.source.trim(), raw.target.trim());
        let n = counters.entry(direction.clone()).or_insert(0);
        *n += 1;
        out.push(CorpusLine {
            id: raw.id.unwrap_or_else(|| format!("{direction}:{n}")),
            source: raw.source.trim().to_string(),
            target: raw.target.trim().to_string(),
            src: raw.src,
            refs,
        });
    }
    let mut seen = std::collections::HashSet::new();
    for l in &out {
        if !seen.insert(l.id.as_str()) {
            return Err(format!("corpus: duplicate id {}", l.id));
        }
    }
    Ok(out)
}

/// Keep `directions` (all when empty) and at most `limit` lines each; returns the directions in first-seen order.
pub fn select_lines(lines: Vec<CorpusLine>, directions: &[String], limit: Option<usize>) -> Vec<(String, Vec<CorpusLine>)> {
    let wanted: Vec<String> = directions.iter().map(|d| d.trim().to_string()).collect();
    let mut groups: Vec<(String, Vec<CorpusLine>)> = Vec::new();
    for line in lines {
        let dir = line.direction();
        if !wanted.is_empty() && !wanted.contains(&dir) {
            continue;
        }
        let idx = match groups.iter().position(|(d, _)| *d == dir) {
            Some(i) => i,
            None => {
                groups.push((dir, Vec::new()));
                groups.len() - 1
            }
        };
        if limit.is_none_or(|l| groups[idx].1.len() < l) {
            groups[idx].1.push(line);
        }
    }
    groups
}

// ---------------------------------------------------------------------------------------------
// Engines
// ---------------------------------------------------------------------------------------------

/// Latency percentiles (milliseconds).
#[derive(Clone, Debug, Default, PartialEq, Serialize)]
pub struct Latency {
    pub p50: f64,
    pub p95: f64,
    pub mean: f64,
    pub max: f64,
}

/// Nearest-rank percentiles of `samples` (ms); zeros when empty.
pub fn latency(samples: &[f64]) -> Latency {
    if samples.is_empty() {
        return Latency::default();
    }
    let mut s = samples.to_vec();
    s.sort_by(f64::total_cmp);
    let rank = |p: f64| s[((p * s.len() as f64).ceil() as usize).clamp(1, s.len()) - 1];
    Latency {
        p50: rank(0.50),
        p95: rank(0.95),
        mean: s.iter().sum::<f64>() / s.len() as f64,
        max: s[s.len() - 1],
    }
}

trait Engine {
    /// Translate one line; `None` latency for precomputed hypotheses.
    fn translate(&mut self, line: &CorpusLine) -> Result<String, String>;
    fn timed(&self) -> bool {
        true
    }
    fn describe(&self) -> serde_json::Value;
}

struct NllbEngine {
    model: NllbModel,
    beam: usize,
    load_ms: u64,
}

impl Engine for NllbEngine {
    fn translate(&mut self, line: &CorpusLine) -> Result<String, String> {
        let tgt = flores::to_flores(&line.target).ok_or_else(|| format!("NLLB has no language {}", line.target))?;
        let src = flores::to_flores(&line.source);
        self.model
            .translate(&line.src, src, tgt, self.beam, &|| false)
            .map_err(|e| e.message)
    }

    fn describe(&self) -> serde_json::Value {
        serde_json::json!({
            "model": self.model.id(),
            "threads": self.model.threads(),
            "beam": self.beam,
            "device": "cpu",
            "loadMs": self.load_ms,
        })
    }
}

/// Prompt of the `openai` engine.
pub fn build_prompt(format: &str, source: &str, target: &str, text: &str) -> String {
    let name = |code: &str| flores::language_name(code).unwrap_or(code).to_string();
    match format {
        "" | "generic" => format!(
            "Translate the following {} text into {}. Output only the translation, nothing else.\n\n{text}",
            name(source),
            name(target)
        ),
        // Hunyuan-MT / Hy-MT2 model card prompts: Chinese instructions for pairs with Chinese.
        "hy-mt2" | "hunyuan" => {
            let chinese = |c: &str| c == "zh" || c.starts_with("zh-") || c == "yue";
            match chinese_name(target) {
                Some(zh_name) if chinese(source) || chinese(target) => format!("把下面的文本翻译成{zh_name}，不要额外解释。\n\n{text}"),
                _ => format!(
                    "Translate the following segment into {}, without additional explanation.\n\n{text}",
                    name(target)
                ),
            }
        }
        template => template
            .replace("{source_lang}", &name(source))
            .replace("{target_lang}", &name(target))
            .replace("{text}", text),
    }
}

fn chinese_name(code: &str) -> Option<&'static str> {
    Some(match code.split('-').next()? {
        "zh" if code.to_ascii_lowercase().contains("hant") => "繁体中文",
        "zh" => "中文",
        "yue" => "粤语",
        "en" => "英语",
        "ja" => "日语",
        "ko" => "韩语",
        "fr" => "法语",
        "de" => "德语",
        "es" => "西班牙语",
        "pt" => "葡萄牙语",
        "it" => "意大利语",
        "ru" => "俄语",
        "ar" => "阿拉伯语",
        "th" => "泰语",
        "vi" => "越南语",
        "id" => "印尼语",
        "ms" => "马来语",
        "tr" => "土耳其语",
        _ => return None,
    })
}

/// Model output without reasoning blocks and surrounding whitespace.
pub fn clean_llm_output(text: &str) -> String {
    let mut out = text.to_string();
    while let Some(start) = out.find("<think>") {
        match out[start..].find("</think>") {
            Some(end) => out.replace_range(start..start + end + "</think>".len(), ""),
            None => out.truncate(start),
        }
    }
    out.trim().to_string()
}

struct OpenAiEngine {
    runtime: tokio::runtime::Runtime,
    url: String,
    model: String,
    key: Option<String>,
    prompt: String,
    max_tokens: u32,
    temperature: f32,
    timeout: Duration,
}

impl Engine for OpenAiEngine {
    fn translate(&mut self, line: &CorpusLine) -> Result<String, String> {
        let body = serde_json::json!({
            "model": self.model,
            "messages": [{ "role": "user", "content": build_prompt(&self.prompt, &line.source, &line.target, &line.src) }],
            "max_tokens": self.max_tokens,
            "temperature": self.temperature,
            "stream": false,
        });
        let mut req = crate::services::http_client::http_client()
            .post(&self.url)
            .timeout(self.timeout)
            .json(&body);
        if let Some(key) = &self.key {
            req = req.bearer_auth(key);
        }
        let timeout = self.timeout;
        self.runtime.block_on(async move {
            let resp = req
                .send()
                .await
                .map_err(|e| crate::services::http_client::describe_request_error("LLM", &e, timeout))?;
            if !resp.status().is_success() {
                return Err(crate::services::http_client::describe_api_error("LLM", resp).await);
            }
            let v: serde_json::Value = resp.json().await.map_err(|e| format!("LLM: unexpected response ({e})"))?;
            let text = v["choices"][0]["message"]["content"]
                .as_str()
                .ok_or("LLM: the response has no message content")?;
            Ok(clean_llm_output(text))
        })
    }

    fn describe(&self) -> serde_json::Value {
        serde_json::json!({
            "url": self.url,
            "model": self.model,
            "prompt": self.prompt,
            "maxTokens": self.max_tokens,
            "temperature": self.temperature,
        })
    }
}

struct HypothesesEngine {
    path: PathBuf,
    hyps: HashMap<String, String>,
}

impl Engine for HypothesesEngine {
    fn translate(&mut self, line: &CorpusLine) -> Result<String, String> {
        self.hyps
            .get(&line.id)
            .cloned()
            .ok_or_else(|| format!("no hypothesis for {}", line.id))
    }

    fn timed(&self) -> bool {
        false
    }

    fn describe(&self) -> serde_json::Value {
        serde_json::json!({ "path": self.path.display().to_string(), "lines": self.hyps.len() })
    }
}

fn parse_hypotheses(text: &str) -> Result<HashMap<String, String>, String> {
    #[derive(Deserialize)]
    struct Row {
        id: String,
        hyp: String,
    }
    let mut out = HashMap::new();
    for (i, line) in text.lines().enumerate() {
        let line = line.trim();
        if line.is_empty() || line.starts_with('#') {
            continue;
        }
        let row: Row = serde_json::from_str(line).map_err(|e| format!("hypotheses line {}: {e}", i + 1))?;
        out.insert(row.id, row.hyp);
    }
    Ok(out)
}

fn resolve(base: &Path, p: &Path) -> PathBuf {
    if p.is_absolute() {
        p.to_path_buf()
    } else {
        base.join(p)
    }
}

fn open_engine(spec: &EngineSpec, base: &Path) -> Result<Box<dyn Engine>, String> {
    match spec {
        EngineSpec::Nllb {
            model, path, threads, beam, ..
        } => {
            let id = model.clone().unwrap_or_else(|| NLLB_DEFAULT_MODEL.to_string());
            let dir = match path {
                Some(p) => resolve(base, p),
                None => crate::services::model_store::installed_path(&id).ok_or_else(|| {
                    format!("{id} is not installed (Sigil.exe --headless fetch {id}, with SIGIL_APP_DATA_DIR if you use one)")
                })?,
            };
            let started = Instant::now();
            let model = NllbModel::load(&dir, &id, super::thread_count(*threads))?;
            Ok(Box::new(NllbEngine {
                model,
                beam: beam.unwrap_or(1).clamp(1, super::nllb::MAX_BEAM as u32) as usize,
                load_ms: started.elapsed().as_millis() as u64,
            }))
        }
        EngineSpec::Openai {
            base_url,
            model,
            api_key_env,
            prompt,
            max_tokens,
            temperature,
            timeout_ms,
            ..
        } => {
            let url = crate::services::http_client::api_url(base_url, "/chat/completions")?.to_string();
            let key = match api_key_env {
                Some(var) => Some(std::env::var(var).map_err(|_| format!("environment variable {var} is not set"))?),
                None => None,
            };
            let runtime = tokio::runtime::Builder::new_current_thread()
                .enable_all()
                .build()
                .map_err(|e| format!("tokio runtime: {e}"))?;
            Ok(Box::new(OpenAiEngine {
                runtime,
                url,
                model: model.clone(),
                key,
                prompt: prompt.clone().unwrap_or_else(|| "generic".into()),
                max_tokens: max_tokens.unwrap_or(512),
                temperature: temperature.unwrap_or(0.2),
                timeout: Duration::from_millis(timeout_ms.unwrap_or(60_000).clamp(1_000, 600_000)),
            }))
        }
        EngineSpec::Hypotheses { path, .. } => {
            let path = resolve(base, path);
            let text = std::fs::read_to_string(&path).map_err(|e| format!("{}: {e}", path.display()))?;
            Ok(Box::new(HypothesesEngine {
                hyps: parse_hypotheses(&text)?,
                path,
            }))
        }
    }
}

// ---------------------------------------------------------------------------------------------
// Running and reporting
// ---------------------------------------------------------------------------------------------

#[derive(Clone, Debug, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct DirectionResult {
    pub direction: String,
    pub lines: usize,
    pub errors: usize,
    pub chrf: f64,
    /// `None` for precomputed hypotheses.
    pub latency_ms: Option<Latency>,
    pub source_chars_per_sec: Option<f64>,
    /// First error message, for the report.
    pub first_error: Option<String>,
}

#[derive(Clone, Debug, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct EngineResult {
    pub id: String,
    pub engine: String,
    pub config: serde_json::Value,
    pub directions: Vec<DirectionResult>,
    /// chrF over every scored line of this engine.
    pub overall_chrf: f64,
    pub error: Option<String>,
}

/// UTC `YYYY-MM-DDTHH:MM:SSZ` of a Unix time.
fn utc_timestamp(secs: u64) -> String {
    let days = (secs / 86_400) as i64;
    let rem = secs % 86_400;
    // Civil from days (Howard Hinnant's algorithm).
    let z = days + 719_468;
    let era = z.div_euclid(146_097);
    let doe = z - era * 146_097;
    let yoe = (doe - doe / 1_460 + doe / 36_524 - doe / 146_096) / 365;
    let doy = doe - (365 * yoe + yoe / 4 - yoe / 100);
    let mp = (5 * doy + 2) / 153;
    let d = doy - (153 * mp + 2) / 5 + 1;
    let m = if mp < 10 { mp + 3 } else { mp - 9 };
    let y = yoe + era * 400 + i64::from(m <= 2);
    format!("{y:04}-{m:02}-{d:02}T{:02}:{:02}:{:02}Z", rem / 3600, rem % 3600 / 60, rem % 60)
}

fn run_engine(
    engine: &mut dyn Engine,
    groups: &[(String, Vec<CorpusLine>)],
    warmup: usize,
    hyps_dir: &Path,
    id: &str,
) -> (Vec<DirectionResult>, Vec<(String, Vec<String>)>) {
    let mut results = Vec::new();
    let mut all = Vec::new();
    for (direction, lines) in groups {
        if engine.timed() {
            for line in lines.iter().take(warmup) {
                let _ = engine.translate(line);
            }
        }
        let mut pairs = Vec::with_capacity(lines.len());
        let mut times = Vec::with_capacity(lines.len());
        let mut chars = 0usize;
        let mut errors = 0usize;
        let mut first_error = None;
        let mut rows = String::new();
        for line in lines {
            let started = Instant::now();
            let hyp = match engine.translate(line) {
                Ok(h) => h,
                Err(e) => {
                    errors += 1;
                    first_error.get_or_insert(e);
                    String::new()
                }
            };
            times.push(started.elapsed().as_secs_f64() * 1000.0);
            chars += line.src.chars().count();
            rows.push_str(&serde_json::json!({ "id": line.id, "src": line.src, "hyp": hyp, "ref": line.refs[0] }).to_string());
            rows.push('\n');
            pairs.push((hyp, line.refs.clone()));
        }
        let safe_id: String = id.chars().map(|c| if c.is_ascii_alphanumeric() || c == '-' || c == '.' { c } else { '_' }).collect();
        let _ = std::fs::write(hyps_dir.join(format!("{safe_id}.{direction}.jsonl")), rows);
        let total_s = times.iter().sum::<f64>() / 1000.0;
        results.push(DirectionResult {
            direction: direction.clone(),
            lines: lines.len(),
            errors,
            chrf: chrf(&pairs),
            latency_ms: engine.timed().then(|| latency(&times)),
            source_chars_per_sec: (engine.timed() && total_s > 0.0).then(|| chars as f64 / total_s),
            first_error,
        });
        all.extend(pairs);
    }
    (results, all)
}

fn markdown(report: &serde_json::Value, engines: &[EngineResult]) -> String {
    let mut md = String::new();
    md.push_str("# MT benchmark\n\n");
    md.push_str(&format!(
        "- Created: {}\n- Corpus: `{}` ({} lines)\n- Host: {} logical CPUs, {}\n- Metric: corpus chrF2 (sacrebleu defaults: char order 6, β 2, whitespace removed); latency = wall time per line\n\n",
        report["createdAt"].as_str().unwrap_or(""),
        report["corpus"]["path"].as_str().unwrap_or(""),
        report["corpus"]["lines"],
        report["host"]["logicalCpus"],
        report["host"]["os"].as_str().unwrap_or(""),
    ));
    md.push_str("| Engine | Direction | Lines | chrF | p50 ms | p95 ms | mean ms | Errors |\n|---|---|---:|---:|---:|---:|---:|---:|\n");
    for e in engines {
        if let Some(err) = &e.error {
            md.push_str(&format!("| {} | – | – | – | – | – | – | {} |\n", e.id, err.replace('|', "/")));
            continue;
        }
        for d in &e.directions {
            let (p50, p95, mean) = d
                .latency_ms
                .as_ref()
                .map(|l| (format!("{:.0}", l.p50), format!("{:.0}", l.p95), format!("{:.0}", l.mean)))
                .unwrap_or_else(|| ("–".into(), "–".into(), "–".into()));
            md.push_str(&format!(
                "| {} | {} | {} | {:.1} | {p50} | {p95} | {mean} | {} |\n",
                e.id, d.direction, d.lines, d.chrf, d.errors
            ));
        }
        md.push_str(&format!("| {} | **all** | {} | **{:.1}** | | | | |\n", e.id, e.directions.iter().map(|d| d.lines).sum::<usize>(), e.overall_chrf));
    }
    md
}

/// Body of `--headless mt-bench`: `Ok(report JSON)` or an error message.
pub fn headless_mt_bench(spec_path: &Path, out: Option<&Path>) -> Result<String, String> {
    let text = std::fs::read_to_string(spec_path).map_err(|e| format!("{}: {e}", spec_path.display()))?;
    let spec: Spec = serde_json::from_str(&text).map_err(|e| format!("{}: {e}", spec_path.display()))?;
    let base = spec_path.parent().map(Path::to_path_buf).unwrap_or_default();
    run(&spec, &base, out)
}

/// Run a parsed spec (paths relative to `base`).
pub fn run(spec: &Spec, base: &Path, out: Option<&Path>) -> Result<String, String> {
    if spec.engines.is_empty() {
        return Err("mt-bench: the spec lists no engines".into());
    }
    let corpus_path = resolve(base, &spec.corpus);
    let corpus_text = std::fs::read_to_string(&corpus_path).map_err(|e| format!("{}: {e}", corpus_path.display()))?;
    let groups = select_lines(parse_corpus(&corpus_text)?, &spec.directions, spec.limit);
    let total_lines: usize = groups.iter().map(|(_, l)| l.len()).sum();
    if total_lines == 0 {
        return Err("mt-bench: no corpus lines match the spec's directions".into());
    }
    let now = SystemTime::now().duration_since(UNIX_EPOCH).map(|d| d.as_secs()).unwrap_or(0);
    let created = utc_timestamp(now);
    let out_dir = match out.map(Path::to_path_buf).or_else(|| spec.out.as_ref().map(|p| resolve(base, p))) {
        Some(d) => d,
        None => PathBuf::from("bench-results")
            .join("mt")
            .join(created.replace([':', '-'], "").replace('T', "-").trim_end_matches('Z')),
    };
    let hyps_dir = out_dir.join("hyps");
    std::fs::create_dir_all(&hyps_dir).map_err(|e| format!("{}: {e}", hyps_dir.display()))?;

    let mut engines = Vec::new();
    for spec_engine in &spec.engines {
        let id = spec_engine.id();
        eprintln!("[mt-bench] {id}: loading");
        let mut result = EngineResult {
            id: id.clone(),
            engine: spec_engine.kind().to_string(),
            config: serde_json::Value::Null,
            directions: Vec::new(),
            overall_chrf: 0.0,
            error: None,
        };
        match open_engine(spec_engine, base) {
            Ok(mut engine) => {
                result.config = engine.describe();
                let (dirs, all) = run_engine(engine.as_mut(), &groups, spec.warmup, &hyps_dir, &id);
                result.overall_chrf = chrf(&all);
                for d in &dirs {
                    eprintln!(
                        "[mt-bench] {id} {}: chrF {:.1}, {} lines, {} errors{}",
                        d.direction,
                        d.chrf,
                        d.lines,
                        d.errors,
                        d.latency_ms.as_ref().map(|l| format!(", p50 {:.0} ms, p95 {:.0} ms", l.p50, l.p95)).unwrap_or_default()
                    );
                }
                result.directions = dirs;
            }
            Err(e) => {
                eprintln!("[mt-bench] {id}: {e}");
                result.error = Some(e);
            }
        }
        engines.push(result);
    }

    let report = serde_json::json!({
        "kind": "mt-bench",
        "version": 1,
        "createdAt": created,
        "sigilVersion": env!("CARGO_PKG_VERSION"),
        "corpus": {
            "path": corpus_path.display().to_string(),
            "lines": total_lines,
            "directions": groups.iter().map(|(d, l)| serde_json::json!({ "direction": d, "lines": l.len() })).collect::<Vec<_>>(),
        },
        "host": {
            "logicalCpus": std::thread::available_parallelism().map(|n| n.get()).unwrap_or(0),
            "os": std::env::consts::OS,
            "arch": std::env::consts::ARCH,
        },
        "metric": { "name": "chrF2", "charOrder": CHAR_ORDER, "beta": BETA, "whitespace": false },
        "engines": engines,
    });
    let json = serde_json::to_string_pretty(&report).map_err(|e| e.to_string())?;
    std::fs::write(out_dir.join("report.json"), &json).map_err(|e| format!("report.json: {e}"))?;
    std::fs::write(out_dir.join("report.md"), markdown(&report, &engines)).map_err(|e| format!("report.md: {e}"))?;
    eprintln!("[mt-bench] wrote {}", out_dir.display());
    Ok(json)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn close(a: f64, b: f64) -> bool {
        (a - b).abs() < 1e-5
    }

    fn pairs(rows: &[(&str, &[&str])]) -> Vec<(String, Vec<String>)> {
        rows.iter()
            .map(|(h, refs)| (h.to_string(), refs.iter().map(|r| r.to_string()).collect()))
            .collect()
    }

    /// Expected values computed with sacrebleu 2.6.0 (`sacrebleu.metrics.CHRF()`, sentence_score / corpus_score).
    #[test]
    fn chrf_matches_sacrebleu() {
        let sentence = |h: &str, r: &str| chrf_score(&chrf_stats(h, r));
        assert!(close(sentence("abc", "abc"), 100.0));
        assert!(close(sentence("ab", "abc"), 63.636363636363626));
        assert!(close(sentence("xyz", "abc"), 0.0));
        assert!(close(sentence("", "abc"), 0.0));
        assert!(close(sentence("a b c", "abc"), 100.0), "whitespace does not count");
        assert!(close(sentence("こんにちわ", "こんにちは"), 54.333333333333336));
        assert!(close(sentence("Hallo, wie geht es dir heute?", "Hallo, wie geht es dir?"), 89.30408));
        assert!(close(sentence("Ich bin gleich zurück.", "Ich bin gleich wieder da."), 53.675698));
        assert!(close(sentence("Das war schwer!", "Das war wirklich schwer."), 39.788327));
        let de = chrf(&pairs(&[
            ("Hallo, wie geht es dir heute?", &["Hallo, wie geht es dir?"]),
            ("Ich bin gleich zurück.", &["Ich bin gleich wieder da."]),
            ("Das war schwer!", &["Das war wirklich schwer."]),
        ]));
        assert!(close(de, 61.08569097846003), "{de}");
        let with_empty = chrf(&pairs(&[("abc", &["abc"]), ("", &["abcdef"])]));
        assert!(close(with_empty, 31.946508172362552), "{with_empty}");
        // The best reference wins.
        assert!(close(chrf(&pairs(&[("the cat", &["a dog", "the cat"])])), 100.0));
    }

    #[test]
    fn corpus_parsing_and_selection() {
        let text = r#"
# comment
{"source":"en","target":"ja","src":"Hello","ref":"こんにちは"}
{"id":"x","source":"en","target":"ja","src":"Bye","refs":["さようなら","じゃあね"]}
{"source":"ja","target":"en","src":"はい","ref":"Yes"}
{"source":"en","target":"ja","src":"Thanks","ref":"ありがとう"}
"#;
        let lines = parse_corpus(text).unwrap();
        assert_eq!(lines.len(), 4);
        assert_eq!(lines[0].id, "en-ja:1");
        assert_eq!(lines[1].id, "x");
        assert_eq!(lines[1].refs.len(), 2);
        assert_eq!(lines[3].id, "en-ja:3");
        let groups = select_lines(lines.clone(), &[], Some(2));
        assert_eq!(groups.len(), 2);
        assert_eq!((groups[0].0.as_str(), groups[0].1.len()), ("en-ja", 2));
        let groups = select_lines(lines, &["ja-en".into()], None);
        assert_eq!(groups.len(), 1);
        assert!(parse_corpus(r#"{"source":"en","target":"ja","src":"x"}"#).unwrap_err().contains("reference"));
        assert!(parse_corpus(r#"{"source":"en","target":"ja","src":" ","ref":"y"}"#).is_err());
        assert!(parse_corpus("{").is_err());
        let dup = r#"{"id":"a","source":"en","target":"ja","src":"x","ref":"y"}
{"id":"a","source":"en","target":"ja","src":"x","ref":"y"}"#;
        assert!(parse_corpus(dup).unwrap_err().contains("duplicate"));
    }

    #[test]
    fn spec_parsing() {
        let spec: Spec = serde_json::from_str(
            r#"{"corpus":"c.jsonl","engines":[
                {"engine":"nllb","model":"mt/nllb-200-distilled-600m-int8","threads":2},
                {"engine":"openai","id":"hy","baseUrl":"http://127.0.0.1:8080/v1","model":"m","apiKeyEnv":"K","prompt":"hy-mt2"},
                {"engine":"hypotheses","id":"bergamot","path":"h.jsonl"}]}"#,
        )
        .unwrap();
        assert_eq!(spec.warmup, 1);
        assert_eq!(spec.engines.len(), 3);
        assert_eq!(spec.engines[0].id(), "mt/nllb-200-distilled-600m-int8");
        assert_eq!(spec.engines[1].kind(), "openai");
        assert!(serde_json::from_str::<Spec>(r#"{"corpus":"c","engines":[{"engine":"bogus"}]}"#).is_err());
        assert!(serde_json::from_str::<Spec>(r#"{"corpus":"c","engines":[],"typo":1}"#).is_err());
    }

    #[test]
    fn prompts_and_output_cleaning() {
        assert_eq!(
            build_prompt("hy-mt2", "en", "ja", "Hi"),
            "Translate the following segment into Japanese, without additional explanation.\n\nHi"
        );
        assert_eq!(build_prompt("hy-mt2", "en", "zh", "Hi"), "把下面的文本翻译成中文，不要额外解释。\n\nHi");
        assert_eq!(build_prompt("hy-mt2", "zh", "en", "你好"), "把下面的文本翻译成英语，不要额外解释。\n\n你好");
        assert!(build_prompt("generic", "de", "fr", "Hallo").starts_with("Translate the following German text into French."));
        assert_eq!(build_prompt("{source_lang}>{target_lang}: {text}", "en", "ko", "x"), "English>Korean: x");
        assert_eq!(clean_llm_output("<think>hmm</think>\n Hola "), "Hola");
        assert_eq!(clean_llm_output("Hola<think>unfinished"), "Hola");
    }

    #[test]
    fn percentiles_and_timestamps() {
        let l = latency(&[10.0, 20.0, 30.0, 40.0]);
        assert_eq!((l.p50, l.p95, l.mean, l.max), (20.0, 40.0, 25.0, 40.0));
        assert_eq!(latency(&[]), Latency::default());
        assert_eq!(utc_timestamp(0), "1970-01-01T00:00:00Z");
        assert_eq!(utc_timestamp(1_790_000_000), "2026-09-21T14:13:20Z");
        assert_eq!(utc_timestamp(951_782_400), "2000-02-29T00:00:00Z");
    }

    #[test]
    fn a_hypotheses_run_writes_reports() {
        let dir = std::env::temp_dir().join(format!("sigil-mt-bench-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&dir);
        std::fs::create_dir_all(&dir).unwrap();
        std::fs::write(
            dir.join("corpus.jsonl"),
            "{\"id\":\"1\",\"source\":\"en\",\"target\":\"de\",\"src\":\"Hello\",\"ref\":\"Hallo\"}\n{\"id\":\"2\",\"source\":\"en\",\"target\":\"de\",\"src\":\"Bye\",\"ref\":\"Tschüss\"}\n",
        )
        .unwrap();
        std::fs::write(dir.join("hyps.jsonl"), "{\"id\":\"1\",\"hyp\":\"Hallo\"}\n").unwrap();
        std::fs::write(
            dir.join("spec.json"),
            r#"{"corpus":"corpus.jsonl","engines":[{"engine":"hypotheses","id":"fixed","path":"hyps.jsonl"},{"engine":"nllb","id":"missing","path":"no-such-folder"}]}"#,
        )
        .unwrap();
        let out = dir.join("out");
        let json = headless_mt_bench(&dir.join("spec.json"), Some(&out)).unwrap();
        let v: serde_json::Value = serde_json::from_str(&json).unwrap();
        let fixed = &v["engines"][0];
        assert_eq!(fixed["directions"][0]["lines"], 2);
        assert_eq!(fixed["directions"][0]["errors"], 1, "line 2 has no hypothesis");
        assert!(fixed["directions"][0]["latencyMs"].is_null());
        let chrf = fixed["overallChrf"].as_f64().unwrap();
        assert!(chrf > 0.0 && chrf < 100.0);
        assert!(v["engines"][1]["error"].as_str().unwrap().contains("encoder"));
        assert!(out.join("report.md").is_file());
        assert!(out.join("hyps/fixed.en-de.jsonl").is_file());
        let md = std::fs::read_to_string(out.join("report.md")).unwrap();
        assert!(md.contains("| fixed | en-de | 2 |"), "{md}");
        let _ = std::fs::remove_dir_all(&dir);
    }
}
