//! Minimal, bounded GGUF header reader: the metadata the server plan needs (architecture, layer
//! count, attention shape for the KV-cache size, trained context, vocabulary size, model name), read
//! without loading tensors. Used for user-supplied `.gguf` files (validation and VRAM estimate) and
//! to refine catalog models once they are installed.
//!
//! GGUF v2/v3 layout: magic `GGUF`, u32 version, u64 tensor count, u64 metadata count, then
//! `key: string, type: u32, value` pairs (little endian; strings are u64 length + UTF-8 bytes).
//! Every length is checked against the file size and fixed caps, so a hostile or truncated file
//! yields an error, never a huge allocation or a hang.

use std::collections::HashMap;
use std::fs::File;
use std::io::{BufReader, Read, Seek, SeekFrom};
use std::path::{Path, PathBuf};
use std::sync::{Mutex, OnceLock, PoisonError};
use std::time::SystemTime;

/// `GGUF` read as a little-endian u32.
const MAGIC: u32 = 0x4655_4747;
/// Metadata pairs accepted (real models have a few dozen, plus tokenizer arrays).
const MAX_KV: u64 = 100_000;
/// Longest key.
const MAX_KEY_LEN: u64 = 64 * 1024;
/// Longest string value that is read into memory (longer ones, e.g. chat templates, are skipped
/// unless wanted; wanted strings are capped here too).
const MAX_STRING_LEN: u64 = 4 * 1024 * 1024;
/// Longest array (tokenizer vocabularies are a few hundred thousand entries).
const MAX_ARRAY_LEN: u64 = 50_000_000;
/// Nested arrays deeper than this are refused.
const MAX_DEPTH: u32 = 3;
/// Skips up to this many bytes read through the buffer instead of seeking.
const SMALL_SKIP: u64 = 16 * 1024;
/// Most layers accepted (the largest models have a few hundred; this also bounds the per-layer
/// KV-cache loop of a hostile header).
pub const MAX_BLOCK_COUNT: u64 = 4096;

/// Metadata of one GGUF file.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct GgufInfo {
    pub version: u32,
    pub tensor_count: u64,
    /// `general.architecture` (e.g. `llama`, `qwen3`, `gemma3`, `hunyuan-dense`, `t5`).
    pub architecture: String,
    /// `general.name`.
    pub name: Option<String>,
    /// `<arch>.block_count`.
    pub block_count: Option<u32>,
    /// `<arch>.context_length` (trained context).
    pub context_length: Option<u64>,
    /// `<arch>.embedding_length`.
    pub embedding_length: Option<u64>,
    /// `<arch>.attention.head_count` (per layer when the model varies it; else one value).
    pub head_count: Vec<u64>,
    /// `<arch>.attention.head_count_kv`.
    pub head_count_kv: Vec<u64>,
    /// `<arch>.attention.key_length` / `value_length`.
    pub key_length: Option<u64>,
    pub value_length: Option<u64>,
    /// `<arch>.expert_count` (mixture of experts).
    pub expert_count: Option<u64>,
    /// Entries of `tokenizer.ggml.tokens`.
    pub vocab_size: Option<u64>,
    /// `split.count` (a model split into several files).
    pub split_count: Option<u32>,
    /// The model has an encoder (T5-style encoder-decoder).
    pub has_encoder: bool,
}

impl GgufInfo {
    /// f16 K+V cache bytes per context token (all layers), when the header gives the shape.
    pub fn kv_bytes_per_token(&self) -> Option<u64> {
        let layers = u64::from(self.block_count?);
        if layers == 0 || layers > MAX_BLOCK_COUNT {
            return None;
        }
        let per_layer = |v: &[u64], i: u64| -> Option<u64> {
            match v.len() {
                0 => None,
                1 => Some(v[0]),
                n => v.get(i as usize).copied().or_else(|| v.get(n - 1).copied()),
            }
        };
        let layer_bytes = |i: u64| -> Option<u64> {
            let heads = per_layer(&self.head_count, i)?;
            let kv_heads = per_layer(&self.head_count_kv, i).unwrap_or(heads);
            if kv_heads == 0 {
                return Some(0); // recurrent / attention-free layer
            }
            let head_dim_k = self
                .key_length
                .or_else(|| self.embedding_length.map(|e| e / heads.max(1)))?;
            let head_dim_v = self.value_length.unwrap_or(head_dim_k);
            // K and V, 2 bytes per element (f16 cache).
            Some(
                kv_heads
                    .saturating_mul(head_dim_k.saturating_add(head_dim_v))
                    .saturating_mul(2),
            )
        };
        let total = if self.head_count.len() <= 1 && self.head_count_kv.len() <= 1 {
            // Every layer has the same shape.
            layer_bytes(0)?.saturating_mul(layers)
        } else {
            let mut total: u64 = 0;
            for i in 0..layers {
                total = total.saturating_add(layer_bytes(i)?);
            }
            total
        };
        (total > 0).then_some(total)
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum ValueType {
    U8,
    I8,
    U16,
    I16,
    U32,
    I32,
    F32,
    Bool,
    Str,
    Array,
    U64,
    I64,
    F64,
}

impl ValueType {
    fn from_u32(v: u32) -> Option<Self> {
        Some(match v {
            0 => Self::U8,
            1 => Self::I8,
            2 => Self::U16,
            3 => Self::I16,
            4 => Self::U32,
            5 => Self::I32,
            6 => Self::F32,
            7 => Self::Bool,
            8 => Self::Str,
            9 => Self::Array,
            10 => Self::U64,
            11 => Self::I64,
            12 => Self::F64,
            _ => return None,
        })
    }

    fn fixed_size(self) -> Option<u64> {
        Some(match self {
            Self::U8 | Self::I8 | Self::Bool => 1,
            Self::U16 | Self::I16 => 2,
            Self::U32 | Self::I32 | Self::F32 => 4,
            Self::U64 | Self::I64 | Self::F64 => 8,
            Self::Str | Self::Array => return None,
        })
    }
}

/// A value kept for the fields we read.
#[derive(Clone, Debug, PartialEq)]
enum Value {
    Int(u64),
    Str(String),
    Ints(Vec<u64>),
    /// Array whose items were skipped (only its length is known).
    Len(u64),
    Other,
}

struct Reader<R> {
    r: R,
    pos: u64,
    len: u64,
}

impl<R: Read + Seek> Reader<R> {
    fn bytes<const N: usize>(&mut self) -> Result<[u8; N], String> {
        if self.pos.saturating_add(N as u64) > self.len {
            return Err("the file ends inside its header (truncated or not a GGUF file)".into());
        }
        let mut b = [0u8; N];
        self.r
            .read_exact(&mut b)
            .map_err(|e| format!("read failed: {e}"))?;
        self.pos += N as u64;
        Ok(b)
    }

    fn u32(&mut self) -> Result<u32, String> {
        Ok(u32::from_le_bytes(self.bytes::<4>()?))
    }

    fn u64(&mut self) -> Result<u64, String> {
        Ok(u64::from_le_bytes(self.bytes::<8>()?))
    }

    fn skip(&mut self, n: u64) -> Result<(), String> {
        if self.pos.saturating_add(n) > self.len {
            return Err("the file ends inside its header (truncated or not a GGUF file)".into());
        }
        if n <= SMALL_SKIP {
            // Tokenizer vocabularies are ~150k short strings: read through the buffer instead of
            // one seek (which discards a BufReader's buffer) per string.
            let copied = std::io::copy(&mut (&mut self.r).take(n), &mut std::io::sink()).map_err(|e| format!("read failed: {e}"))?;
            if copied != n {
                return Err("the file ends inside its header (truncated or not a GGUF file)".into());
            }
        } else {
            let n_i64 = i64::try_from(n).map_err(|_| "header length out of range".to_string())?;
            self.r
                .seek(SeekFrom::Current(n_i64))
                .map_err(|e| format!("seek failed: {e}"))?;
        }
        self.pos += n;
        Ok(())
    }

    fn string_len(&mut self, cap: u64) -> Result<u64, String> {
        let n = self.u64()?;
        if n > cap || self.pos.saturating_add(n) > self.len {
            return Err(format!("a header string is {n} bytes long (corrupt file?)"));
        }
        Ok(n)
    }

    fn string(&mut self, cap: u64) -> Result<String, String> {
        let n = self.string_len(cap)?;
        let mut buf = vec![0u8; n as usize];
        self.r
            .read_exact(&mut buf)
            .map_err(|e| format!("read failed: {e}"))?;
        self.pos += n;
        Ok(String::from_utf8_lossy(&buf).into_owned())
    }

    fn scalar_int(&mut self, t: ValueType) -> Result<Option<u64>, String> {
        Ok(match t {
            ValueType::U8 | ValueType::Bool => Some(u64::from(self.bytes::<1>()?[0])),
            ValueType::I8 => Some(i8::from_le_bytes(self.bytes::<1>()?).max(0) as u64),
            ValueType::U16 => Some(u64::from(u16::from_le_bytes(self.bytes::<2>()?))),
            ValueType::I16 => Some(i16::from_le_bytes(self.bytes::<2>()?).max(0) as u64),
            ValueType::U32 => Some(u64::from(self.u32()?)),
            ValueType::I32 => Some(i32::from_le_bytes(self.bytes::<4>()?).max(0) as u64),
            ValueType::U64 => Some(self.u64()?),
            ValueType::I64 => Some(i64::from_le_bytes(self.bytes::<8>()?).max(0) as u64),
            ValueType::F32 => {
                self.skip(4)?;
                None
            }
            ValueType::F64 => {
                self.skip(8)?;
                None
            }
            ValueType::Str | ValueType::Array => None,
        })
    }

    /// Read (`keep`) or skip one value of type `t`.
    fn value(&mut self, t: ValueType, keep: bool, depth: u32) -> Result<Value, String> {
        match t {
            ValueType::Str => {
                if keep {
                    Ok(Value::Str(self.string(MAX_STRING_LEN)?))
                } else {
                    let n = self.string_len(u64::MAX)?;
                    self.skip(n)?;
                    Ok(Value::Other)
                }
            }
            ValueType::Array => {
                if depth >= MAX_DEPTH {
                    return Err("nested arrays are too deep (corrupt file?)".into());
                }
                let raw = self.u32()?;
                let item = ValueType::from_u32(raw).ok_or_else(|| format!("unknown array item type {raw}"))?;
                let n = self.u64()?;
                if n > MAX_ARRAY_LEN {
                    return Err(format!("a header array has {n} entries (corrupt file?)"));
                }
                if let Some(size) = item.fixed_size() {
                    let bytes = n.saturating_mul(size);
                    let int_item = !matches!(item, ValueType::F32 | ValueType::F64);
                    // Small integer arrays (per-layer head counts) are kept.
                    if keep && int_item && n <= 4096 {
                        let mut out = Vec::with_capacity(n as usize);
                        for _ in 0..n {
                            out.push(self.scalar_int(item)?.unwrap_or(0));
                        }
                        return Ok(Value::Ints(out));
                    }
                    self.skip(bytes)?;
                    return Ok(Value::Len(n));
                }
                for _ in 0..n {
                    self.value(item, false, depth + 1)?;
                }
                Ok(Value::Len(n))
            }
            _ => Ok(match self.scalar_int(t)? {
                Some(v) if keep => Value::Int(v),
                _ => Value::Other,
            }),
        }
    }
}

/// Read the header of the GGUF file at `path` (cached per path, size and modification time: plans
/// are recomputed often and big vocabularies take a few milliseconds to skip).
pub fn read(path: &Path) -> Result<GgufInfo, String> {
    let file = File::open(path).map_err(|e| format!("cannot open {}: {e}", path.display()))?;
    let meta = file
        .metadata()
        .map_err(|e| format!("cannot read {}: {e}", path.display()))?;
    let key = (path.to_path_buf(), meta.len(), meta.modified().ok());
    if let Some(info) = cache()
        .lock()
        .unwrap_or_else(PoisonError::into_inner)
        .get(&key)
    {
        return Ok(info.clone());
    }
    let info = read_from(BufReader::with_capacity(64 * 1024, file), meta.len())?;
    let mut c = cache().lock().unwrap_or_else(PoisonError::into_inner);
    if c.len() >= 64 {
        c.clear();
    }
    c.insert(key, info.clone());
    Ok(info)
}

type CacheKey = (PathBuf, u64, Option<SystemTime>);

fn cache() -> &'static Mutex<HashMap<CacheKey, GgufInfo>> {
    static C: OnceLock<Mutex<HashMap<CacheKey, GgufInfo>>> = OnceLock::new();
    C.get_or_init(Default::default)
}

/// Read a GGUF header from `r` (a stream of `len` bytes, positioned at the start).
pub fn read_from<R: Read + Seek>(r: R, len: u64) -> Result<GgufInfo, String> {
    let mut rd = Reader { r, pos: 0, len };
    if len < 24 {
        return Err("not a GGUF file (too short)".into());
    }
    if rd.u32()? != MAGIC {
        return Err("not a GGUF file (wrong magic)".into());
    }
    let version = rd.u32()?;
    if !(2..=3).contains(&version) {
        return Err(format!("unsupported GGUF version {version} (2 or 3 expected)"));
    }
    let tensor_count = rd.u64()?;
    let kv_count = rd.u64()?;
    if kv_count > MAX_KV {
        return Err(format!("the header lists {kv_count} metadata entries (corrupt file?)"));
    }

    let mut values: HashMap<String, Value> = HashMap::new();
    for _ in 0..kv_count {
        let key = rd.string(MAX_KEY_LEN)?;
        let raw = rd.u32()?;
        let t = ValueType::from_u32(raw).ok_or_else(|| format!("unknown metadata type {raw} for {key}"))?;
        let wanted = wanted_key(&key);
        let v = rd.value(t, wanted, 0)?;
        if wanted || key == "tokenizer.ggml.tokens" {
            values.insert(key, v);
        }
    }

    let architecture = match values.get("general.architecture") {
        Some(Value::Str(s)) if !s.is_empty() => s.clone(),
        _ => return Err("the GGUF header has no general.architecture".into()),
    };
    let int = |k: &str| -> Option<u64> {
        match values.get(k) {
            Some(Value::Int(v)) => Some(*v),
            _ => None,
        }
    };
    let ints = |k: &str| -> Vec<u64> {
        match values.get(k) {
            Some(Value::Int(v)) => vec![*v],
            Some(Value::Ints(v)) => v.clone(),
            _ => Vec::new(),
        }
    };
    let a = |suffix: &str| format!("{architecture}.{suffix}");
    let block_count = match int(&a("block_count")) {
        Some(n) if n > MAX_BLOCK_COUNT => {
            return Err(format!("the header says the model has {n} layers (corrupt file?)"));
        }
        n => n.and_then(|v| u32::try_from(v).ok()),
    };
    let vocab_size = match values.get("tokenizer.ggml.tokens") {
        Some(Value::Len(n)) => Some(*n),
        _ => None,
    };
    let name = match values.get("general.name") {
        Some(Value::Str(s)) if !s.trim().is_empty() => Some(s.trim().to_string()),
        _ => None,
    };
    Ok(GgufInfo {
        version,
        tensor_count,
        name,
        block_count,
        context_length: int(&a("context_length")),
        embedding_length: int(&a("embedding_length")),
        head_count: ints(&a("attention.head_count")),
        head_count_kv: ints(&a("attention.head_count_kv")),
        key_length: int(&a("attention.key_length")),
        value_length: int(&a("attention.value_length")),
        expert_count: int(&a("expert_count")),
        vocab_size,
        split_count: int("split.count").and_then(|v| u32::try_from(v).ok()),
        has_encoder: values.contains_key(&a("decoder_start_token_id")) || architecture == "t5" || architecture == "t5encoder",
        architecture,
    })
}

fn wanted_key(key: &str) -> bool {
    matches!(key, "general.architecture" | "general.name" | "split.count")
        || key.ends_with(".block_count")
        || key.ends_with(".context_length")
        || key.ends_with(".embedding_length")
        || key.ends_with(".attention.head_count")
        || key.ends_with(".attention.head_count_kv")
        || key.ends_with(".attention.key_length")
        || key.ends_with(".attention.value_length")
        || key.ends_with(".expert_count")
        || key.ends_with(".decoder_start_token_id")
}

/// Test helper: build a GGUF header in memory.
#[cfg(test)]
pub(crate) mod build {
    pub enum V<'a> {
        U32(u32),
        U64(u64),
        Str(&'a str),
        F32(f32),
        U32s(&'a [u32]),
        Strs(&'a [&'a str]),
    }

    pub fn header(version: u32, kvs: &[(&str, V)]) -> Vec<u8> {
        let mut b = Vec::new();
        b.extend_from_slice(b"GGUF");
        b.extend_from_slice(&version.to_le_bytes());
        b.extend_from_slice(&3u64.to_le_bytes()); // tensors
        b.extend_from_slice(&(kvs.len() as u64).to_le_bytes());
        let s = |b: &mut Vec<u8>, s: &str| {
            b.extend_from_slice(&(s.len() as u64).to_le_bytes());
            b.extend_from_slice(s.as_bytes());
        };
        for (k, v) in kvs {
            s(&mut b, k);
            match v {
                V::U32(x) => {
                    b.extend_from_slice(&4u32.to_le_bytes());
                    b.extend_from_slice(&x.to_le_bytes());
                }
                V::U64(x) => {
                    b.extend_from_slice(&10u32.to_le_bytes());
                    b.extend_from_slice(&x.to_le_bytes());
                }
                V::F32(x) => {
                    b.extend_from_slice(&6u32.to_le_bytes());
                    b.extend_from_slice(&x.to_le_bytes());
                }
                V::Str(x) => {
                    b.extend_from_slice(&8u32.to_le_bytes());
                    s(&mut b, x);
                }
                V::U32s(xs) => {
                    b.extend_from_slice(&9u32.to_le_bytes());
                    b.extend_from_slice(&4u32.to_le_bytes());
                    b.extend_from_slice(&(xs.len() as u64).to_le_bytes());
                    for x in xs.iter() {
                        b.extend_from_slice(&x.to_le_bytes());
                    }
                }
                V::Strs(xs) => {
                    b.extend_from_slice(&9u32.to_le_bytes());
                    b.extend_from_slice(&8u32.to_le_bytes());
                    b.extend_from_slice(&(xs.len() as u64).to_le_bytes());
                    for x in xs.iter() {
                        s(&mut b, x);
                    }
                }
            }
        }
        // Some tensor-info bytes after the header, as in a real file.
        b.extend_from_slice(&[0u8; 64]);
        b
    }

    /// A Hy-MT2 1.8B-shaped header (32 layers, 16 heads, 4 KV heads, head dim 128).
    pub fn hunyuan_like() -> Vec<u8> {
        header(
            3,
            &[
                ("general.architecture", V::Str("hunyuan-dense")),
                ("general.name", V::Str("Hy MT2 1.8B")),
                ("hunyuan-dense.block_count", V::U32(32)),
                ("hunyuan-dense.context_length", V::U32(262_144)),
                ("hunyuan-dense.embedding_length", V::U32(2048)),
                ("hunyuan-dense.attention.head_count", V::U32(16)),
                ("hunyuan-dense.attention.head_count_kv", V::U32(4)),
                ("hunyuan-dense.attention.key_length", V::U32(128)),
                ("hunyuan-dense.rope.freq_base", V::F32(10_000.0)),
                ("tokenizer.chat_template", V::Str("{% for m in messages %}{{ m.content }}{% endfor %}")),
                ("tokenizer.ggml.tokens", V::Strs(&["<s>", "</s>", "a", "b", "c"])),
                ("general.file_type", V::U32(15)),
            ],
        )
    }
}

#[cfg(test)]
mod tests {
    use super::build::{header, hunyuan_like, V};
    use super::*;
    use std::io::Cursor;

    fn parse(bytes: &[u8]) -> Result<GgufInfo, String> {
        read_from(Cursor::new(bytes.to_vec()), bytes.len() as u64)
    }

    #[test]
    fn reads_the_fields_the_plan_needs() {
        let info = parse(&hunyuan_like()).unwrap();
        assert_eq!(info.version, 3);
        assert_eq!(info.architecture, "hunyuan-dense");
        assert_eq!(info.name.as_deref(), Some("Hy MT2 1.8B"));
        assert_eq!(info.block_count, Some(32));
        assert_eq!(info.context_length, Some(262_144));
        assert_eq!(info.head_count, vec![16]);
        assert_eq!(info.head_count_kv, vec![4]);
        assert_eq!(info.vocab_size, Some(5));
        assert!(!info.has_encoder);
        // 32 layers x 4 KV heads x (128 + 128) x 2 bytes = the catalog's 65,536 B per token.
        assert_eq!(info.kv_bytes_per_token(), Some(65_536));
    }

    #[test]
    fn per_layer_head_counts_and_default_head_dim() {
        let bytes = header(
            3,
            &[
                ("general.architecture", V::Str("gemma3")),
                ("gemma3.block_count", V::U32(3)),
                ("gemma3.embedding_length", V::U32(1024)),
                ("gemma3.attention.head_count", V::U32(8)),
                ("gemma3.attention.head_count_kv", V::U32s(&[2, 0, 4])),
            ],
        );
        let info = parse(&bytes).unwrap();
        assert_eq!(info.head_count_kv, vec![2, 0, 4]);
        // head dim 1024 / 8 = 128; layers: 2*256*2 + 0 + 4*256*2.
        assert_eq!(info.kv_bytes_per_token(), Some(2 * 256 * 2 + 4 * 256 * 2));
    }

    #[test]
    fn encoder_decoder_models_are_flagged() {
        let bytes = header(
            3,
            &[
                ("general.architecture", V::Str("t5")),
                ("t5.block_count", V::U32(24)),
                ("t5.decoder_start_token_id", V::U32(0)),
            ],
        );
        assert!(parse(&bytes).unwrap().has_encoder);
    }

    #[test]
    fn rejects_other_files_and_corrupt_headers() {
        assert!(parse(b"PK\x03\x04 not a gguf at all, just a zip header....")
            .unwrap_err()
            .contains("magic"));
        assert!(parse(b"GGUF").unwrap_err().contains("too short"));
        let mut v1 = hunyuan_like();
        v1[4] = 1;
        assert!(parse(&v1).unwrap_err().contains("version 1"));
        // Truncated anywhere inside the metadata: an error, never a panic.
        let full = hunyuan_like();
        for cut in [24, 30, 60, 100, 200, full.len() - 70] {
            assert!(parse(&full[..cut]).is_err(), "cut at {cut}");
        }
        // Absurd counts and lengths are refused before allocating.
        let mut huge_kv = hunyuan_like();
        huge_kv[16..24].copy_from_slice(&u64::MAX.to_le_bytes());
        assert!(parse(&huge_kv).unwrap_err().contains("metadata entries"));
        let mut huge_key = hunyuan_like();
        huge_key[24..32].copy_from_slice(&(1u64 << 40).to_le_bytes());
        assert!(parse(&huge_key).is_err());
        // No architecture.
        let bytes = header(3, &[("general.name", V::Str("x"))]);
        assert!(parse(&bytes).unwrap_err().contains("architecture"));
        // Unknown value type.
        let mut bad_type = header(3, &[("general.architecture", V::Str("llama"))]);
        let type_at = 24 + 8 + "general.architecture".len();
        bad_type[type_at..type_at + 4].copy_from_slice(&99u32.to_le_bytes());
        assert!(parse(&bad_type)
            .unwrap_err()
            .contains("unknown metadata type"));
    }

    #[test]
    fn absurd_layer_counts_are_refused_quickly() {
        // A crafted header claiming 4 billion layers: an error, not billions of loop iterations.
        let bytes = header(
            3,
            &[
                ("general.architecture", V::Str("llama")),
                ("llama.block_count", V::U32(4_000_000_000)),
                ("llama.attention.head_count", V::U32(32)),
                ("llama.embedding_length", V::U32(4096)),
            ],
        );
        let t0 = std::time::Instant::now();
        assert!(parse(&bytes).unwrap_err().contains("layers"));
        let ok = header(
            3,
            &[
                ("general.architecture", V::Str("llama")),
                ("llama.block_count", V::U64(MAX_BLOCK_COUNT)),
                ("llama.attention.head_count", V::U32(32)),
                ("llama.attention.head_count_kv", V::U32(8)),
                ("llama.attention.key_length", V::U32(128)),
            ],
        );
        let info = parse(&ok).unwrap();
        assert_eq!(info.kv_bytes_per_token(), Some(MAX_BLOCK_COUNT * 8 * 256 * 2));
        // A struct built elsewhere with a huge count never loops: scalar shapes use the closed form,
        // and counts above the cap give no estimate.
        let huge = GgufInfo {
            block_count: Some(u32::MAX),
            head_count: vec![32],
            head_count_kv: vec![8],
            key_length: Some(128),
            ..Default::default()
        };
        assert_eq!(huge.kv_bytes_per_token(), None);
        let scalar = GgufInfo {
            block_count: Some(40),
            ..huge.clone()
        };
        assert_eq!(scalar.kv_bytes_per_token(), Some(40 * 8 * 256 * 2));
        let per_layer = GgufInfo {
            block_count: Some(2),
            head_count_kv: vec![8, 0],
            ..huge
        };
        assert_eq!(per_layer.kv_bytes_per_token(), Some(8 * 256 * 2));
        assert!(t0.elapsed() < std::time::Duration::from_secs(1), "{:?}", t0.elapsed());
    }

    #[test]
    fn missing_shape_gives_no_kv_estimate() {
        let bytes = header(3, &[("general.architecture", V::Str("llama"))]);
        let info = parse(&bytes).unwrap();
        assert_eq!(info.kv_bytes_per_token(), None);
        let bytes = header(2, &[("general.architecture", V::Str("llama")), ("llama.block_count", V::U64(32))]);
        let info = parse(&bytes).unwrap();
        assert_eq!((info.version, info.block_count), (2, Some(32)));
        assert_eq!(info.kv_bytes_per_token(), None);
    }

    #[test]
    fn large_vocabularies_are_skipped_quickly_and_headers_cached() {
        let words: Vec<String> = (0..200_000).map(|i| format!("tok{i}")).collect();
        let refs: Vec<&str> = words.iter().map(String::as_str).collect();
        let bytes = header(
            3,
            &[
                ("general.architecture", V::Str("qwen3")),
                ("tokenizer.ggml.tokens", V::Strs(&refs)),
                ("qwen3.block_count", V::U32(36)),
            ],
        );
        let dir = std::env::temp_dir().join(format!("sigil-gguf-vocab-{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        let path = dir.join("big.gguf");
        std::fs::write(&path, &bytes).unwrap();
        let t0 = std::time::Instant::now();
        let info = read(&path).unwrap();
        assert!(t0.elapsed() < std::time::Duration::from_secs(2), "{:?}", t0.elapsed());
        assert_eq!((info.vocab_size, info.block_count), (Some(200_000), Some(36)));
        // Cached: same answer without re-reading.
        assert_eq!(read(&path).unwrap(), info);
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn reads_a_real_file_from_disk() {
        let dir = std::env::temp_dir().join(format!("sigil-gguf-test-{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        let path = dir.join("m.gguf");
        std::fs::write(&path, hunyuan_like()).unwrap();
        assert_eq!(read(&path).unwrap().block_count, Some(32));
        assert!(read(&dir.join("missing.gguf"))
            .unwrap_err()
            .contains("cannot open"));
        let _ = std::fs::remove_dir_all(&dir);
    }
}
