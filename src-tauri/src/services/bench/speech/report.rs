//! `sigil-speech-bench-report/1`: the measurement (frozen by P7-S0,
//! `scripts/speech-bench/schemas/speech-bench-report.schema.json`), its aggregation and its Markdown.
//!
//! The report holds hypotheses, renderings, timings, resources and the machine's state. Accuracy and
//! quality scores (WER / CER, MetricX, listening results) are never in it: the Python scorers compute
//! them from the report plus the corpus manifest, so one normalizer exists. chrF2 is the exception the
//! schema allows as a quick look (`chrf`), computed with mt-bench's sacrebleu-equal implementation.
//!
//! Percentiles are nearest-rank, `sorted[ceil(p / 100 * n) - 1]`, in integer arithmetic so a float
//! product such as `0.07 * 100` cannot move a rank.

use std::collections::BTreeMap;

use serde::de::Error as _;
use serde::ser::SerializeMap;
use serde::{Deserialize, Deserializer, Serialize, Serializer};
use serde_json::Value;

use super::spec::{Limit, Mode, Task};

/// Schema id of the report.
pub const REPORT_SCHEMA: &str = "sigil-speech-bench-report/1";

// ---------------------------------------------------------------------------------------------
// Statistics
// ---------------------------------------------------------------------------------------------

/// Nearest-rank percentile `p` (1-100) of ascending `sorted`: `sorted[ceil(p * n / 100) - 1]`.
pub fn percentile(sorted: &[f64], p: u32) -> Option<f64> {
    let n = sorted.len();
    if n == 0 {
        return None;
    }
    let rank = (p as usize * n).div_ceil(100).clamp(1, n);
    Some(sorted[rank - 1])
}

/// `values` sorted ascending (non-finite values dropped).
pub fn sorted_finite(values: impl IntoIterator<Item = f64>) -> Vec<f64> {
    let mut v: Vec<f64> = values.into_iter().filter(|x| x.is_finite()).collect();
    v.sort_by(f64::total_cmp);
    v
}

/// Arithmetic mean; `None` when empty.
pub fn mean(values: &[f64]) -> Option<f64> {
    (!values.is_empty()).then(|| values.iter().sum::<f64>() / values.len() as f64)
}

/// Median of an already measured list (the middle value, the mean of the two middle ones for an even
/// count): per-item timings of repeated passes.
pub fn median(values: &[f64]) -> Option<f64> {
    let s = sorted_finite(values.iter().copied());
    match s.len() {
        0 => None,
        n if n % 2 == 1 => Some(s[n / 2]),
        n => Some((s[n / 2 - 1] + s[n / 2]) / 2.0),
    }
}

/// Round to `digits` decimals (compact, diff-friendly reports).
pub fn round_to(x: f64, digits: i32) -> f64 {
    let m = 10f64.powi(digits);
    (x * m).round() / m
}

// ---------------------------------------------------------------------------------------------
// Aggregate
// ---------------------------------------------------------------------------------------------

/// The frozen keys of an aggregate, in report order (`items` and `errors` come first).
pub const FROZEN_KEYS: [&str; 27] = [
    "audio_s",
    "wall_ms",
    "rtf_mean",
    "rtf_p50",
    "rtf_p95",
    "compute_ms_p50",
    "compute_ms_p95",
    "compute_ms_p99",
    "final_latency_ms_p50",
    "final_latency_ms_p95",
    "final_latency_ms_p99",
    "first_partial_ms_p50",
    "first_partial_ms_p95",
    "latency_ms_mean",
    "latency_ms_p50",
    "latency_ms_p95",
    "latency_ms_p99",
    "chars_per_s",
    "ttfa_ms_p50",
    "ttfa_ms_p95",
    "cpu_cores_mean",
    "cpu_ms_per_audio_s",
    "vram_peak_mb",
    "vram_delta_mb",
    "ram_peak_mb",
    "chrf",
    // Not frozen by name but part of every TTS / pipeline summary.
    "partials_mean",
];

/// A summary of a set of items: counts plus numeric keys that are absent when the mode / task does not
/// measure them. Serialized with `items`, `errors`, then the frozen keys in order, then any other.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct Aggregate {
    pub items: u64,
    pub errors: u64,
    pub values: BTreeMap<String, f64>,
}

impl Aggregate {
    pub fn new(items: usize, errors: usize) -> Aggregate {
        Aggregate {
            items: items as u64,
            errors: errors as u64,
            values: BTreeMap::new(),
        }
    }

    /// Set a key (non-finite values are dropped; `None` leaves it absent).
    pub fn set(&mut self, key: &str, value: Option<f64>) {
        if let Some(v) = value.filter(|v| v.is_finite()) {
            self.values.insert(key.to_string(), v);
        }
    }

    /// Any aggregate key (`items` and `errors` included) as a number.
    pub fn get(&self, key: &str) -> Option<f64> {
        match key {
            "items" => Some(self.items as f64),
            "errors" => Some(self.errors as f64),
            other => self.values.get(other).copied(),
        }
    }
}

impl Serialize for Aggregate {
    fn serialize<S: Serializer>(&self, s: S) -> Result<S::Ok, S::Error> {
        let mut map = s.serialize_map(Some(2 + self.values.len()))?;
        map.serialize_entry("items", &self.items)?;
        map.serialize_entry("errors", &self.errors)?;
        for key in FROZEN_KEYS {
            if let Some(v) = self.values.get(key) {
                map.serialize_entry(key, v)?;
            }
        }
        for (key, v) in &self.values {
            if !FROZEN_KEYS.contains(&key.as_str()) {
                map.serialize_entry(key, v)?;
            }
        }
        map.end()
    }
}

impl<'de> Deserialize<'de> for Aggregate {
    fn deserialize<D: Deserializer<'de>>(d: D) -> Result<Self, D::Error> {
        let raw = serde_json::Map::<String, Value>::deserialize(d)?;
        let count = |key: &str| -> Result<u64, D::Error> {
            raw.get(key)
                .and_then(Value::as_u64)
                .ok_or_else(|| D::Error::custom(format!("aggregate: {key} must be a non-negative integer")))
        };
        let mut agg = Aggregate::new(count("items")? as usize, count("errors")? as usize);
        for (key, v) in &raw {
            if key == "items" || key == "errors" {
                continue;
            }
            match v {
                Value::Null => {}
                Value::Number(n) => {
                    if let Some(f) = n.as_f64() {
                        agg.values.insert(key.clone(), f);
                    }
                }
                other => return Err(D::Error::custom(format!("aggregate: {key} must be a number or null, not {other}"))),
            }
        }
        Ok(agg)
    }
}

// ---------------------------------------------------------------------------------------------
// Items
// ---------------------------------------------------------------------------------------------

/// One STT item: the hypothesis and its timings (no scores).
#[derive(Clone, Debug, Default, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct SttItem {
    /// Corpus item id.
    pub item: String,
    pub language: String,
    pub condition: String,
    pub audio_s: f64,
    /// The hypothesis (finals joined); `None` on error.
    pub text: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub language_detected: Option<String>,
    /// Pipeline / stream feeds: every final with its audio span.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub segments: Option<Vec<Segment>>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub partials: Option<u64>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub first_partial_ms: Option<f64>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub final_latency_ms: Option<f64>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub compute_ms: Option<f64>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub rtf: Option<f64>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub cpu_ms: Option<f64>,
    pub error: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub tags: Option<Vec<String>>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub extra: Option<Value>,
}

/// One final of a stream / pipeline item.
#[derive(Clone, Debug, Default, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Segment {
    pub start_ms: f64,
    pub end_ms: f64,
    pub text: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub lang: Option<String>,
    /// Which engine of a two-pass route produced it.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub engine: Option<String>,
    /// VAD end (or end of audio) to this final.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub latency_ms: Option<f64>,
}

/// One MT item: the translation and its latency.
#[derive(Clone, Debug, Default, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct MtItem {
    /// MT set line id.
    pub item: String,
    pub direction: String,
    pub text: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub latency_ms: Option<f64>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub src_chars: Option<u64>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub out_tokens: Option<u64>,
    pub error: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub extra: Option<Value>,
}

/// One TTS item: the rendering and its timings.
#[derive(Clone, Debug, Default, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct TtsItem {
    /// Prompt id.
    pub item: String,
    pub language: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub voice: Option<String>,
    /// `audio/<engine>/<prompt id>.wav`; `None` on error or when audio was not written.
    pub audio: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub audio_s: Option<f64>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub sample_rate: Option<u32>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub ttfa_ms: Option<f64>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub total_ms: Option<f64>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub rtf: Option<f64>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub asr_text: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub dnsmos_sig: Option<f64>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub dnsmos_bak: Option<f64>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub dnsmos_ovrl: Option<f64>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub speaker_cosine: Option<f64>,
    pub error: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub extra: Option<Value>,
}

/// An item of any task. Untagged: the three shapes differ by required members and unknown members
/// are refused, exactly as the schema's `oneOf`.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(untagged)]
pub enum ReportItem {
    Stt(SttItem),
    Mt(MtItem),
    Tts(TtsItem),
}

impl ReportItem {
    pub fn error(&self) -> Option<&str> {
        match self {
            ReportItem::Stt(i) => i.error.as_deref(),
            ReportItem::Mt(i) => i.error.as_deref(),
            ReportItem::Tts(i) => i.error.as_deref(),
        }
    }

    #[cfg(test)]
    pub fn id(&self) -> &str {
        match self {
            ReportItem::Stt(i) => &i.item,
            ReportItem::Mt(i) => &i.item,
            ReportItem::Tts(i) => &i.item,
        }
    }
}

// ---------------------------------------------------------------------------------------------
// Aggregation of items
// ---------------------------------------------------------------------------------------------

/// Engine-level measurements that are not per item.
#[derive(Clone, Debug, Default)]
pub struct EngineTotals {
    /// Wall time of the timed passes (ms).
    pub wall_ms: Option<f64>,
    /// Process CPU time / wall time over the timed passes, the load simulator's CPU subtracted.
    pub cpu_cores_mean: Option<f64>,
    pub vram_peak_mb: Option<f64>,
    pub vram_delta_mb: Option<f64>,
    pub ram_peak_mb: Option<f64>,
}

impl EngineTotals {
    fn apply(&self, agg: &mut Aggregate) {
        agg.set("cpu_cores_mean", self.cpu_cores_mean);
        agg.set("vram_peak_mb", self.vram_peak_mb);
        agg.set("vram_delta_mb", self.vram_delta_mb);
        agg.set("ram_peak_mb", self.ram_peak_mb);
    }
}

fn stats3(agg: &mut Aggregate, prefix: &str, values: &[f64]) {
    let sorted = sorted_finite(values.iter().copied());
    agg.set(&format!("{prefix}_p50"), percentile(&sorted, 50).map(|v| round_to(v, 3)));
    agg.set(&format!("{prefix}_p95"), percentile(&sorted, 95).map(|v| round_to(v, 3)));
    agg.set(&format!("{prefix}_p99"), percentile(&sorted, 99).map(|v| round_to(v, 3)));
}

/// Aggregate of STT items. `totals` (engine level) is `None` for a group: its wall time is then the
/// sum of its compute times and its CPU use the ratio of the items' CPU and compute times.
pub fn stt_aggregate(items: &[&SttItem], totals: Option<&EngineTotals>) -> Aggregate {
    let errors = items.iter().filter(|i| i.error.is_some()).count();
    let mut agg = Aggregate::new(items.len(), errors);
    agg.set("audio_s", Some(round_to(items.iter().map(|i| i.audio_s).sum(), 3)));
    let ok: Vec<&&SttItem> = items.iter().filter(|i| i.error.is_none()).collect();
    let compute: Vec<f64> = ok.iter().filter_map(|i| i.compute_ms).collect();
    let rtfs: Vec<f64> = ok.iter().filter_map(|i| i.rtf).collect();
    let wall = totals.and_then(|t| t.wall_ms).or_else(|| (!compute.is_empty()).then(|| compute.iter().sum()));
    agg.set("wall_ms", wall.map(|v| round_to(v, 1)));
    agg.set("rtf_mean", mean(&rtfs).map(|v| round_to(v, 5)));
    let sorted_rtf = sorted_finite(rtfs.iter().copied());
    agg.set("rtf_p50", percentile(&sorted_rtf, 50).map(|v| round_to(v, 5)));
    agg.set("rtf_p95", percentile(&sorted_rtf, 95).map(|v| round_to(v, 5)));
    let sorted_compute = sorted_finite(compute.iter().copied());
    agg.set("compute_ms_p50", percentile(&sorted_compute, 50).map(|v| round_to(v, 3)));
    agg.set("compute_ms_p95", percentile(&sorted_compute, 95).map(|v| round_to(v, 3)));
    agg.set("compute_ms_p99", percentile(&sorted_compute, 99).map(|v| round_to(v, 3)));
    let finals: Vec<f64> = ok.iter().filter_map(|i| i.final_latency_ms).collect();
    stats3(&mut agg, "final_latency_ms", &finals);
    let firsts = sorted_finite(ok.iter().filter_map(|i| i.first_partial_ms));
    agg.set("first_partial_ms_p50", percentile(&firsts, 50).map(|v| round_to(v, 3)));
    agg.set("first_partial_ms_p95", percentile(&firsts, 95).map(|v| round_to(v, 3)));
    let with_cpu: Vec<&&&SttItem> = ok.iter().filter(|i| i.cpu_ms.is_some()).collect();
    let cpu_ms: f64 = with_cpu.iter().filter_map(|i| i.cpu_ms).sum();
    let cpu_audio: f64 = with_cpu.iter().map(|i| i.audio_s).sum();
    if !with_cpu.is_empty() && cpu_audio > 0.0 {
        agg.set("cpu_ms_per_audio_s", Some(round_to(cpu_ms / cpu_audio, 3)));
    }
    let partials: Vec<f64> = ok.iter().filter_map(|i| i.partials.map(|p| p as f64)).collect();
    agg.set("partials_mean", mean(&partials).map(|v| round_to(v, 2)));
    match totals {
        Some(t) => t.apply(&mut agg),
        None => {
            let compute_total: f64 = with_cpu.iter().filter_map(|i| i.compute_ms).sum();
            if compute_total > 0.0 && !with_cpu.is_empty() {
                agg.set("cpu_cores_mean", Some(round_to(cpu_ms / compute_total, 3)));
            }
        }
    }
    agg
}

/// Aggregate of MT items; `chrf` is the corpus chrF2 the caller computed, if any.
pub fn mt_aggregate(items: &[&MtItem], chrf: Option<f64>, totals: Option<&EngineTotals>) -> Aggregate {
    let errors = items.iter().filter(|i| i.error.is_some()).count();
    let mut agg = Aggregate::new(items.len(), errors);
    let ok: Vec<&&MtItem> = items.iter().filter(|i| i.error.is_none()).collect();
    let lat: Vec<f64> = ok.iter().filter_map(|i| i.latency_ms).collect();
    agg.set("latency_ms_mean", mean(&lat).map(|v| round_to(v, 3)));
    stats3(&mut agg, "latency_ms", &lat);
    let chars: f64 = ok.iter().filter(|i| i.latency_ms.is_some()).filter_map(|i| i.src_chars).map(|c| c as f64).sum();
    let secs: f64 = lat.iter().sum::<f64>() / 1000.0;
    if secs > 0.0 && chars > 0.0 {
        agg.set("chars_per_s", Some(round_to(chars / secs, 2)));
    }
    agg.set("wall_ms", totals.and_then(|t| t.wall_ms).or_else(|| (!lat.is_empty()).then(|| lat.iter().sum())).map(|v| round_to(v, 1)));
    agg.set("chrf", chrf.map(|v| round_to(v, 3)));
    if let Some(t) = totals {
        t.apply(&mut agg);
    }
    agg
}

/// Aggregate of TTS items (`audio_s` is the output audio).
pub fn tts_aggregate(items: &[&TtsItem], totals: Option<&EngineTotals>) -> Aggregate {
    let errors = items.iter().filter(|i| i.error.is_some()).count();
    let mut agg = Aggregate::new(items.len(), errors);
    let ok: Vec<&&TtsItem> = items.iter().filter(|i| i.error.is_none()).collect();
    agg.set("audio_s", Some(round_to(ok.iter().filter_map(|i| i.audio_s).sum(), 3)));
    let totals_ms: Vec<f64> = ok.iter().filter_map(|i| i.total_ms).collect();
    agg.set("wall_ms", totals.and_then(|t| t.wall_ms).or_else(|| (!totals_ms.is_empty()).then(|| totals_ms.iter().sum())).map(|v| round_to(v, 1)));
    let rtfs: Vec<f64> = ok.iter().filter_map(|i| i.rtf).collect();
    agg.set("rtf_mean", mean(&rtfs).map(|v| round_to(v, 5)));
    let sorted_rtf = sorted_finite(rtfs.iter().copied());
    agg.set("rtf_p50", percentile(&sorted_rtf, 50).map(|v| round_to(v, 5)));
    agg.set("rtf_p95", percentile(&sorted_rtf, 95).map(|v| round_to(v, 5)));
    let ttfa = sorted_finite(ok.iter().filter_map(|i| i.ttfa_ms));
    agg.set("ttfa_ms_p50", percentile(&ttfa, 50).map(|v| round_to(v, 3)));
    agg.set("ttfa_ms_p95", percentile(&ttfa, 95).map(|v| round_to(v, 3)));
    if let Some(t) = totals {
        t.apply(&mut agg);
    }
    agg
}

// ---------------------------------------------------------------------------------------------
// The report
// ---------------------------------------------------------------------------------------------

/// A slice of an engine's items: by language and condition (stt), direction (mt) or language (tts).
#[derive(Clone, Debug, Default, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Group {
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub language: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub condition: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub direction: Option<String>,
    pub aggregate: Aggregate,
}

/// One threshold of a run against an engine's aggregate.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ThresholdResult {
    pub metric: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub min: Option<f64>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub max: Option<f64>,
    pub value: Option<f64>,
    pub passed: bool,
}

/// Check `limits` against `aggregate`; a metric the run did not produce fails as not measured.
pub fn evaluate_thresholds(aggregate: &Aggregate, limits: &BTreeMap<String, Limit>) -> Vec<ThresholdResult> {
    limits
        .iter()
        .map(|(metric, limit)| {
            let value = aggregate.get(metric);
            let passed = value.is_some_and(|v| limit.min.is_none_or(|lo| v >= lo) && limit.max.is_none_or(|hi| v <= hi));
            ThresholdResult {
                metric: metric.clone(),
                min: limit.min,
                max: limit.max,
                value,
                passed,
            }
        })
        .collect()
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum ResultStatus {
    Passed,
    Failed,
    Error,
    Cancelled,
    Skipped,
}

impl ResultStatus {
    pub fn as_str(self) -> &'static str {
        match self {
            ResultStatus::Passed => "passed",
            ResultStatus::Failed => "failed",
            ResultStatus::Error => "error",
            ResultStatus::Cancelled => "cancelled",
            ResultStatus::Skipped => "skipped",
        }
    }
}

/// One engine's result in a run.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct EngineResult {
    pub engine: String,
    pub status: ResultStatus,
    #[serde(default)]
    pub error: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub execution_provider_used: Option<String>,
    pub aggregate: Aggregate,
    pub groups: Vec<Group>,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub thresholds: Vec<ThresholdResult>,
    pub items: Vec<ReportItem>,
}

impl EngineResult {
    /// A result for an engine that never ran.
    pub fn failed_to_load(engine: &str, error: impl Into<String>) -> EngineResult {
        EngineResult {
            engine: engine.to_string(),
            status: ResultStatus::Error,
            error: Some(error.into()),
            execution_provider_used: None,
            aggregate: Aggregate::new(0, 0),
            groups: Vec::new(),
            thresholds: Vec::new(),
            items: Vec::new(),
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum RunStatus {
    Passed,
    Failed,
    Error,
    Cancelled,
}

impl RunStatus {
    pub fn as_str(self) -> &'static str {
        match self {
            RunStatus::Passed => "passed",
            RunStatus::Failed => "failed",
            RunStatus::Error => "error",
            RunStatus::Cancelled => "cancelled",
        }
    }
}

/// The GPU lock as the run found it.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct LockBlock {
    pub required: bool,
    pub tag: String,
    pub held_by: Option<String>,
    pub verified: bool,
}

/// One run.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct RunReport {
    pub name: String,
    pub mode: Mode,
    pub status: RunStatus,
    #[serde(default)]
    pub error: Option<String>,
    pub duration_ms: f64,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub feed: Option<Value>,
    /// Profile, the LoadSpec applied, external load text and what the load achieved.
    #[serde(default)]
    pub load: Option<Value>,
    /// The voice changer co-load's own paced metrics.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub co_load: Option<Value>,
    #[serde(default)]
    pub gpu_lock: Option<LockBlock>,
    /// `bench::gpu_state::RunState` of the run.
    #[serde(default)]
    pub state: Option<Value>,
    #[serde(default)]
    pub warnings: Vec<String>,
    pub results: Vec<EngineResult>,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub files: Vec<String>,
}

/// A native runtime the engines loaded.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct RuntimePack {
    pub id: String,
    #[serde(default)]
    pub version: Option<String>,
    #[serde(default)]
    pub variant: Option<String>,
}

/// Where and on what the benchmark ran (the bench's `Environment` plus the speech fields). Host and
/// user names never appear.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct EnvironmentBlock {
    pub sigil_version: String,
    pub build: String,
    pub os: String,
    pub arch: String,
    pub cpu: Option<String>,
    pub logical_cores: u64,
    #[serde(default)]
    pub physical_cores: Option<u64>,
    #[serde(default)]
    pub ram_mb: Option<u64>,
    #[serde(default)]
    pub onnx_runtime: String,
    pub gpus: Vec<Value>,
    #[serde(default)]
    pub power_plan: Option<String>,
    #[serde(default)]
    pub power_mode: Option<String>,
    #[serde(default)]
    pub voice_changer_running: bool,
    #[serde(default)]
    pub runtime_packs: Vec<RuntimePack>,
    #[serde(default)]
    pub sigil_rig: Option<String>,
}

/// The corpus the STT runs read.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct CorpusStamp {
    pub name: String,
    pub content_sha256: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub recipe_sha256: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub manifest_sha256: Option<String>,
    pub items_selected: u64,
}

/// What an engine was and where it ran.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct EngineInfo {
    pub id: String,
    pub task: Task,
    pub backend: String,
    pub model: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub model_version: Option<String>,
    pub execution_provider_requested: String,
    pub execution_provider_used: Option<String>,
    #[serde(default)]
    pub ep_reason: Option<String>,
    #[serde(default)]
    pub variant_used: Option<String>,
    #[serde(default)]
    pub threads: Option<i64>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub load_ms: Option<f64>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub vram_load_mb: Option<f64>,
    #[serde(default)]
    pub error: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub extra: Option<Value>,
}

/// `report.json`.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Report {
    pub schema: String,
    pub name: String,
    #[serde(default)]
    pub description: String,
    pub spec_sha256: String,
    /// The spec as written (paths as written, `$comment` members included).
    pub spec: Value,
    pub started_unix_ms: u64,
    pub duration_ms: f64,
    pub passed: bool,
    pub failures: Vec<String>,
    pub environment: EnvironmentBlock,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub corpus: Option<CorpusStamp>,
    pub engines: Vec<EngineInfo>,
    pub runs: Vec<RunReport>,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub files: Vec<String>,
}

// ---------------------------------------------------------------------------------------------
// Markdown
// ---------------------------------------------------------------------------------------------

fn cell(v: Option<f64>, digits: usize) -> String {
    v.map_or_else(|| "-".to_string(), |x| format!("{x:.digits$}"))
}

fn md_escape(s: &str) -> String {
    s.replace('|', "/").replace(['\r', '\n'], " ")
}

/// `report.md`: one table per run.
pub fn markdown(report: &Report) -> String {
    let mut md = String::new();
    md.push_str(&format!("# Speech benchmark: {}\n\n", report.name));
    if !report.description.is_empty() {
        md.push_str(&format!("{}\n\n", report.description));
    }
    let env = &report.environment;
    md.push_str(&format!(
        "- Result: **{}** ({:.1} s)\n- Sigil {} ({}), {} {}, {} ({} logical cores{})\n- Spec sha256: `{}`\n",
        if report.passed { "PASSED" } else { "FAILED" },
        report.duration_ms / 1000.0,
        env.sigil_version,
        env.build,
        env.os,
        env.arch,
        env.cpu.as_deref().unwrap_or("unknown CPU"),
        env.logical_cores,
        env.physical_cores.map(|c| format!(", {c} physical")).unwrap_or_default(),
        report.spec_sha256,
    ));
    if let Some(rig) = &env.sigil_rig {
        md.push_str(&format!("- Rig: {rig}\n"));
    }
    if let Some(c) = &report.corpus {
        md.push_str(&format!("- Corpus: {} (content sha256 `{}`, {} items selected)\n", c.name, c.content_sha256, c.items_selected));
    }
    md.push('\n');
    if !report.failures.is_empty() {
        md.push_str("## Failures\n\n");
        for f in &report.failures {
            md.push_str(&format!("- {}\n", md_escape(f)));
        }
        md.push('\n');
    }
    md.push_str("Scores (WER / CER, chrF with intervals, MetricX, listening results) come from `scripts/speech-bench/score/`; this report is the measurement.\n\n");
    for run in &report.runs {
        md.push_str(&format!("## {} ({}, {})\n\n", run.name, run.mode.as_str(), run.status.as_str()));
        if let Some(e) = &run.error {
            md.push_str(&format!("Error: {}\n\n", md_escape(e)));
        }
        if let Some(load) = run.load.as_ref().and_then(|l| l.get("profile")).and_then(Value::as_str) {
            if load != "none" {
                md.push_str(&format!("Load profile: {load}\n\n"));
            }
        }
        for w in &run.warnings {
            md.push_str(&format!("- warning: {}\n", md_escape(w)));
        }
        if !run.warnings.is_empty() {
            md.push('\n');
        }
        let stt = run.results.iter().any(|r| r.items.iter().any(|i| matches!(i, ReportItem::Stt(_))));
        let mt = run.results.iter().any(|r| r.items.iter().any(|i| matches!(i, ReportItem::Mt(_))));
        if stt {
            md.push_str("| Engine | Device | Items | Errors | Audio s | RTF p50 | RTF p95 | Compute p50 ms | Compute p95 ms | Final p95 ms | CPU cores | VRAM peak MB |\n|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|\n");
            for r in &run.results {
                let a = &r.aggregate;
                md.push_str(&format!(
                    "| {} | {} | {} | {} | {} | {} | {} | {} | {} | {} | {} | {} |\n",
                    r.engine,
                    r.execution_provider_used.as_deref().unwrap_or("-"),
                    a.items,
                    a.errors,
                    cell(a.get("audio_s"), 1),
                    cell(a.get("rtf_p50"), 3),
                    cell(a.get("rtf_p95"), 3),
                    cell(a.get("compute_ms_p50"), 0),
                    cell(a.get("compute_ms_p95"), 0),
                    cell(a.get("final_latency_ms_p95"), 0),
                    cell(a.get("cpu_cores_mean"), 2),
                    cell(a.get("vram_peak_mb"), 0),
                ));
            }
            md.push('\n');
        } else if mt {
            md.push_str("| Engine | Device | Lines | Errors | chrF | Latency p50 ms | p95 ms | p99 ms | chars/s | VRAM peak MB |\n|---|---|---:|---:|---:|---:|---:|---:|---:|---:|\n");
            for r in &run.results {
                let a = &r.aggregate;
                md.push_str(&format!(
                    "| {} | {} | {} | {} | {} | {} | {} | {} | {} | {} |\n",
                    r.engine,
                    r.execution_provider_used.as_deref().unwrap_or("-"),
                    a.items,
                    a.errors,
                    cell(a.get("chrf"), 1),
                    cell(a.get("latency_ms_p50"), 0),
                    cell(a.get("latency_ms_p95"), 0),
                    cell(a.get("latency_ms_p99"), 0),
                    cell(a.get("chars_per_s"), 1),
                    cell(a.get("vram_peak_mb"), 0),
                ));
            }
            md.push('\n');
        } else {
            md.push_str("| Engine | Device | Prompts | Errors | Audio s | TTFA p50 ms | TTFA p95 ms | RTF p50 | RTF p95 | VRAM peak MB |\n|---|---|---:|---:|---:|---:|---:|---:|---:|---:|\n");
            for r in &run.results {
                let a = &r.aggregate;
                md.push_str(&format!(
                    "| {} | {} | {} | {} | {} | {} | {} | {} | {} | {} |\n",
                    r.engine,
                    r.execution_provider_used.as_deref().unwrap_or("-"),
                    a.items,
                    a.errors,
                    cell(a.get("audio_s"), 1),
                    cell(a.get("ttfa_ms_p50"), 0),
                    cell(a.get("ttfa_ms_p95"), 0),
                    cell(a.get("rtf_p50"), 3),
                    cell(a.get("rtf_p95"), 3),
                    cell(a.get("vram_peak_mb"), 0),
                ));
            }
            md.push('\n');
        }
        for r in &run.results {
            if let Some(e) = &r.error {
                md.push_str(&format!("- {}: {}\n", r.engine, md_escape(e)));
            }
            for t in &r.thresholds {
                md.push_str(&format!(
                    "- {} {}: {} {}{}{}\n",
                    r.engine,
                    if t.passed { "PASS" } else { "FAIL" },
                    t.metric,
                    t.value.map_or_else(|| "not measured".to_string(), |v| format!("{v:.3}")),
                    t.min.map(|m| format!(" (min {m})")).unwrap_or_default(),
                    t.max.map(|m| format!(" (max {m})")).unwrap_or_default(),
                ));
            }
        }
        md.push('\n');
    }
    md
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn nearest_rank_percentiles_follow_the_formula() {
        let v: Vec<f64> = (1..=10).map(f64::from).collect();
        // ceil(p * n / 100) - 1: 5th, 10th, 10th and 1st elements.
        assert_eq!(percentile(&v, 50), Some(5.0));
        assert_eq!(percentile(&v, 95), Some(10.0));
        assert_eq!(percentile(&v, 99), Some(10.0));
        assert_eq!(percentile(&v, 1), Some(1.0));
        assert_eq!(percentile(&[7.0], 99), Some(7.0));
        assert_eq!(percentile(&[], 50), None);
        let hundred: Vec<f64> = (1..=100).map(f64::from).collect();
        assert_eq!(percentile(&hundred, 95), Some(95.0));
        assert_eq!(percentile(&hundred, 99), Some(99.0));
        // 0.07 * 100 is 7.000000000000001 in floating point; the integer rank stays 7.
        assert_eq!(percentile(&hundred, 7), Some(7.0));
        // Twenty items: p95 is the 19th (ceil(19.0)), not the 20th.
        let twenty: Vec<f64> = (1..=20).map(f64::from).collect();
        assert_eq!(percentile(&twenty, 95), Some(19.0));
    }

    #[test]
    fn mean_median_and_rounding() {
        assert_eq!(mean(&[1.0, 2.0, 6.0]), Some(3.0));
        assert_eq!(mean(&[]), None);
        assert_eq!(median(&[3.0, 1.0, 2.0]), Some(2.0));
        assert_eq!(median(&[4.0, 1.0, 2.0, 3.0]), Some(2.5));
        assert_eq!(median(&[]), None);
        assert_eq!(round_to(1.23456, 2), 1.23);
        assert_eq!(sorted_finite([3.0, f64::NAN, 1.0, f64::INFINITY, 2.0]), vec![1.0, 2.0, 3.0]);
    }

    fn stt(item: &str, lang: &str, cond: &str, audio: f64, compute: Option<f64>, error: Option<&str>) -> SttItem {
        SttItem {
            item: item.into(),
            language: lang.into(),
            condition: cond.into(),
            audio_s: audio,
            text: error.is_none().then(|| "hello".to_string()),
            compute_ms: compute,
            rtf: compute.map(|c| c / 1000.0 / audio),
            cpu_ms: compute.map(|c| c * 2.0),
            error: error.map(str::to_string),
            ..SttItem::default()
        }
    }

    #[test]
    fn stt_aggregates_count_errors_and_use_nearest_rank() {
        let items = [
            stt("a", "en", "clean", 4.0, Some(100.0), None),
            stt("b", "en", "clean", 5.0, Some(200.0), None),
            stt("c", "en", "clean", 5.0, Some(300.0), None),
            stt("d", "en", "clean", 4.0, Some(400.0), None),
            stt("e", "en", "clean", 2.0, None, Some("timeout after 120 s")),
        ];
        let refs: Vec<&SttItem> = items.iter().collect();
        let agg = stt_aggregate(&refs, None);
        assert_eq!((agg.items, agg.errors), (5, 1));
        assert_eq!(agg.get("audio_s"), Some(20.0), "errored items still count their input audio");
        assert_eq!(agg.get("compute_ms_p50"), Some(200.0));
        assert_eq!(agg.get("compute_ms_p95"), Some(400.0));
        assert_eq!(agg.get("wall_ms"), Some(1000.0));
        // 2 ms of CPU per ms of compute: two cores.
        assert_eq!(agg.get("cpu_cores_mean"), Some(2.0));
        assert!(agg.get("final_latency_ms_p95").is_none(), "a key the mode does not measure stays absent");
        let engine_level = stt_aggregate(
            &refs,
            Some(&EngineTotals {
                wall_ms: Some(1500.0),
                cpu_cores_mean: Some(1.25),
                vram_peak_mb: Some(900.0),
                ..EngineTotals::default()
            }),
        );
        assert_eq!(engine_level.get("wall_ms"), Some(1500.0));
        assert_eq!(engine_level.get("cpu_cores_mean"), Some(1.25));
        assert_eq!(engine_level.get("vram_peak_mb"), Some(900.0));
    }

    #[test]
    fn mt_and_tts_aggregates() {
        let mk = |id: &str, lat: f64, chars: u64, err: Option<&str>| MtItem {
            item: id.into(),
            direction: "en-ja".into(),
            text: err.is_none().then(|| "x".to_string()),
            latency_ms: err.is_none().then_some(lat),
            src_chars: Some(chars),
            error: err.map(str::to_string),
            ..MtItem::default()
        };
        let items = [mk("1", 100.0, 50, None), mk("2", 300.0, 150, None), mk("3", 0.0, 10, Some("boom"))];
        let refs: Vec<&MtItem> = items.iter().collect();
        let agg = mt_aggregate(&refs, Some(41.23456), None);
        assert_eq!((agg.items, agg.errors), (3, 1));
        assert_eq!(agg.get("latency_ms_mean"), Some(200.0));
        assert_eq!(agg.get("latency_ms_p50"), Some(100.0));
        assert_eq!(agg.get("latency_ms_p99"), Some(300.0));
        assert_eq!(agg.get("chars_per_s"), Some(500.0));
        assert_eq!(agg.get("chrf"), Some(41.235));

        let t = |id: &str, ttfa: f64, audio: f64, total: f64| TtsItem {
            item: id.into(),
            language: "en".into(),
            audio: Some(format!("audio/e/{id}.wav")),
            audio_s: Some(audio),
            ttfa_ms: Some(ttfa),
            total_ms: Some(total),
            rtf: Some(total / 1000.0 / audio),
            ..TtsItem::default()
        };
        let items = [t("a", 300.0, 2.0, 600.0), t("b", 500.0, 4.0, 800.0)];
        let refs: Vec<&TtsItem> = items.iter().collect();
        let agg = tts_aggregate(&refs, None);
        assert_eq!(agg.get("audio_s"), Some(6.0));
        assert_eq!(agg.get("ttfa_ms_p50"), Some(300.0));
        assert_eq!(agg.get("ttfa_ms_p95"), Some(500.0));
        assert_eq!(agg.get("rtf_mean"), Some(0.25));
    }

    #[test]
    fn thresholds_fail_when_a_metric_is_not_measured() {
        let mut agg = Aggregate::new(3, 0);
        agg.set("rtf_p95", Some(0.4));
        let mut limits = BTreeMap::new();
        limits.insert("rtf_p95".to_string(), Limit { min: None, max: Some(0.5) });
        limits.insert("ttfa_ms_p95".to_string(), Limit { min: None, max: Some(800.0) });
        limits.insert("items".to_string(), Limit { min: Some(3.0), max: None });
        let r = evaluate_thresholds(&agg, &limits);
        let by = |m: &str| r.iter().find(|t| t.metric == m).unwrap();
        assert!(by("rtf_p95").passed);
        assert!(!by("ttfa_ms_p95").passed && by("ttfa_ms_p95").value.is_none());
        assert!(by("items").passed);
        limits.insert("rtf_p95".to_string(), Limit { min: None, max: Some(0.3) });
        assert!(!evaluate_thresholds(&agg, &limits).iter().find(|t| t.metric == "rtf_p95").unwrap().passed);
    }

    #[test]
    fn aggregate_serializes_in_report_order_and_round_trips() {
        let mut agg = Aggregate::new(2, 1);
        agg.set("rtf_p95", Some(0.5));
        agg.set("audio_s", Some(3.0));
        agg.set("zzz_custom", Some(1.0));
        agg.set("nan", Some(f64::NAN));
        let text = serde_json::to_string(&agg).unwrap();
        assert_eq!(text, r#"{"items":2,"errors":1,"audio_s":3.0,"rtf_p95":0.5,"zzz_custom":1.0}"#);
        let back: Aggregate = serde_json::from_str(&text).unwrap();
        assert_eq!(back, agg);
        assert!(serde_json::from_str::<Aggregate>(r#"{"items":1}"#).is_err());
        assert!(serde_json::from_str::<Aggregate>(r#"{"items":1,"errors":0,"rtf_p95":"x"}"#).is_err());
        let with_null: Aggregate = serde_json::from_str(r#"{"items":1,"errors":0,"rtf_p95":null}"#).unwrap();
        assert!(with_null.get("rtf_p95").is_none());
    }

}
