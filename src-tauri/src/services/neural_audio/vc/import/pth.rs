//! PyTorch checkpoint reader (the zip serialization format of `torch.save`, torch >= 1.6).
//!
//! Layout: `<prefix>/data.pkl` (the object graph, parsed by the allow-list [`pickle`] VM),
//! `<prefix>/data/<key>` (one raw little-endian storage per key), `<prefix>/byteorder`,
//! `<prefix>/version`. Tensors are views into storages (offset, size, stride), bounds-checked
//! against the storage before any element is read. Legacy (pre-zip) pickles and TorchScript
//! archives are refused with their own error codes.
//!
//! Three shapes of checkpoint hold an RVC generator ([`TorchCheckpoint::kind`]):
//! - `inference`: RVC / Applio / Mangio / w-okada exports, a dict with `weight` (parameter name ->
//!   tensor, usually fp16) and metadata (`config`, `sr`, `f0`, `version`, `info`; Applio adds
//!   `embedder_model`, `vocoder`, `speakers_id`, ...);
//! - `training`: a training checkpoint (`G_*.pth`: `model`, `iteration`, `optimizer`,
//!   `learning_rate`), whose `model` state dict is the generator (no config: the template is
//!   matched by shapes);
//! - `state_dict`: a bare dict of generator tensors.
//!
//! A state dict saved by `torch.save` carries its `_metadata` version table through a `BUILD`,
//! which the [`pickle`] VM accepts in that shape only and discards.
//!
//! Keys written by `torch.nn.utils.parametrizations.weight_norm`
//! (`X.parametrizations.weight.original0/1`) are normalised to RVC's `X.weight_g` / `X.weight_v`,
//! and a `module.` prefix on every key (a DataParallel save) is dropped. Only the metadata entries
//! the importer reads ([`meta::USED_KEYS`]) are converted, all under one byte-counting
//! [`JsonBudget`]; other entries (an optimizer state, say) are never read.

use std::collections::{BTreeMap, HashMap};
use std::io::{Read, Seek};
use std::sync::Arc;

use half::slice::HalfFloatSliceExt;

use super::meta;
use super::pickle::{self, JsonBudget, PickleLimits, Pickled, StorageDtype, TensorRef, Value};
use super::zipguard::{GuardedZip, ZipLimits};
use super::{ErrorCode, ImportError};

/// Largest single tensor accepted (elements).
const MAX_TENSOR_ELEMENTS: u64 = 1 << 30;
/// Largest `data.pkl` accepted.
const MAX_PICKLE_BYTES: u64 = 64 << 20;
/// Most weights accepted (RVC v2: 457; a training checkpoint's generator: ~560) and top-level
/// entries (RVC: ~10; a bare state dict holds its weights at the top level).
const MAX_WEIGHTS: usize = 16_384;
const MAX_TOP_LEVEL: usize = MAX_WEIGHTS;
/// The tensor that makes a dict an RVC generator (every version has it).
const GENERATOR_KEY: &str = "enc_p.emb_phone.weight";
/// Longest weight name accepted (RVC's are under 64 bytes).
const MAX_WEIGHT_NAME: usize = 512;

/// A parsed checkpoint: tensor views plus plain metadata; storages are read on demand.
pub struct TorchCheckpoint<R: Read + Seek> {
    zip: GuardedZip<R>,
    prefix: String,
    /// Parameter name (normalised) -> tensor view.
    tensors: BTreeMap<String, TensorRef>,
    /// The top-level entries the importer reads ([`meta::USED_KEYS`]), as JSON (non-plain values,
    /// and values over the metadata budget, are dropped; long strings are cut).
    pub meta: serde_json::Map<String, serde_json::Value>,
    /// `inference`, `training` or `state_dict` (see the module docs).
    pub kind: &'static str,
    storages: HashMap<String, Arc<Vec<u8>>>,
}

/// Whether every value of the dict `id` is a tensor and it holds an RVC generator's
/// [`GENERATOR_KEY`] (possibly under a `module.` prefix).
fn is_generator_dict(root: &Pickled, id: pickle::Id) -> bool {
    let Some(items) = root.str_items(id) else {
        return false;
    };
    let Value::Dict(all) = root.get(id) else {
        return false;
    };
    items.len() == all.len()
        && !items.is_empty()
        && items.iter().all(|(_, v)| matches!(root.get(*v), Value::Tensor(_)))
        && items.iter().any(|(k, _)| *k == GENERATOR_KEY || k.strip_prefix("module.") == Some(GENERATOR_KEY))
}

fn invalid(msg: impl Into<String>) -> ImportError {
    ImportError::new(ErrorCode::TensorInvalid, msg)
}

/// Entries of a dict value (0 for anything else), checked before its items are collected.
fn dict_len(root: &Pickled, id: pickle::Id) -> usize {
    match root.get(id) {
        Value::Dict(d) => d.len(),
        _ => 0,
    }
}

/// `X.parametrizations.<name>.original0/1` -> `X.<name>_g` / `X.<name>_v`.
pub fn normalize_key(key: &str) -> String {
    if let Some(stem) = key.strip_suffix(".original0").or_else(|| key.strip_suffix(".original1")) {
        if let Some(at) = stem.rfind(".parametrizations.") {
            let base = &stem[..at];
            let name = &stem[at + ".parametrizations.".len()..];
            if !name.is_empty() && !name.contains('.') {
                let which = if key.ends_with('0') { "g" } else { "v" };
                return format!("{base}.{name}_{which}");
            }
        }
    }
    key.to_string()
}

/// Classify a file that is not a zip: legacy torch pickle, or something else.
pub fn reject_non_zip(head: &[u8]) -> ImportError {
    if head.first() == Some(&0x80) {
        ImportError::new(
            ErrorCode::LegacyFormat,
            "this checkpoint uses PyTorch's legacy (pre-1.6) serialization; re-save it with a current PyTorch \
             (torch.save(torch.load(f), f)) or re-export it from RVC / Applio",
        )
    } else {
        ImportError::new(ErrorCode::NotACheckpoint, "not a PyTorch checkpoint (.pth) file")
    }
}

impl<R: Read + Seek> TorchCheckpoint<R> {
    pub fn open(reader: R) -> Result<Self, ImportError> {
        let mut zip = GuardedZip::open(reader, ZipLimits::TORCH_CHECKPOINT)?;
        let pkl: Vec<String> = zip
            .entries()
            .iter()
            .filter(|e| e.name.ends_with("/data.pkl") && e.name.matches('/').count() == 1)
            .map(|e| e.name.clone())
            .collect();
        let name = match pkl.as_slice() {
            [one] => one.clone(),
            [] => {
                if zip.entries().iter().any(|e| e.name.ends_with("constants.pkl") || e.name.contains("/code/")) {
                    return Err(ImportError::new(
                        ErrorCode::TorchScript,
                        "this is a TorchScript module, not a checkpoint of weights",
                    ));
                }
                return Err(ImportError::new(ErrorCode::NotACheckpoint, "zip archive without a torch data.pkl"));
            }
            _ => return Err(ImportError::new(ErrorCode::NotACheckpoint, "more than one data.pkl in the archive")),
        };
        let prefix = name.trim_end_matches("data.pkl").to_string();
        if zip.entries().iter().any(|e| e.name.starts_with(&format!("{prefix}code/")) || e.name == format!("{prefix}constants.pkl")) {
            return Err(ImportError::new(ErrorCode::TorchScript, "this is a TorchScript module, not a checkpoint of weights"));
        }
        if zip.find(&format!("{prefix}byteorder")).is_some() {
            let order = zip.read(&format!("{prefix}byteorder"), 16)?;
            if order != b"little" {
                return Err(invalid("big-endian checkpoints are not supported"));
            }
        }
        let data = zip.read(&name, MAX_PICKLE_BYTES)?;
        let root = pickle::unpickle(&data, &PickleLimits::default())
            .map_err(|e| ImportError::new(ErrorCode::PickleRejected, e.to_string()))?;
        if dict_len(&root, root.root) > MAX_TOP_LEVEL {
            return Err(ImportError::new(ErrorCode::NotACheckpoint, "too many top-level entries for a voice checkpoint"));
        }
        let items = root
            .str_items(root.root)
            .ok_or_else(|| ImportError::new(ErrorCode::NotACheckpoint, "the checkpoint is not a dict"))?;

        let (weights_id, kind) = match items.iter().find(|(k, _)| *k == "weight") {
            Some((_, id)) => (*id, "inference"),
            None => match items.iter().find(|(k, _)| *k == "model") {
                Some((_, id)) if is_generator_dict(&root, *id) => (*id, "training"),
                Some(_) => {
                    return Err(ImportError::new(
                        ErrorCode::NotACheckpoint,
                        "a training checkpoint without an RVC generator (a discriminator D_*.pth?); import the voice's \
                         G_*.pth or the small model RVC / Applio export to the weights folder",
                    ))
                }
                None if is_generator_dict(&root, root.root) => (root.root, "state_dict"),
                None => return Err(ImportError::new(ErrorCode::NotACheckpoint, "no 'weight' entry: not an RVC voice checkpoint")),
            },
        };
        let weight_count = dict_len(&root, weights_id);
        if weight_count > MAX_WEIGHTS {
            return Err(invalid(format!("{weight_count} weights (at most {MAX_WEIGHTS})")));
        }
        let weight_items = root
            .str_items(weights_id)
            .ok_or_else(|| ImportError::new(ErrorCode::NotACheckpoint, "'weight' is not a dict"))?;
        // A DataParallel save prefixes every key with `module.`.
        let strip_module = !weight_items.is_empty() && weight_items.iter().all(|(k, _)| k.starts_with("module."));
        let mut tensors = BTreeMap::new();
        for (key, id) in weight_items {
            if key.len() > MAX_WEIGHT_NAME {
                return Err(invalid(format!("a weight name of {} bytes", key.len())));
            }
            let Value::Tensor(t) = root.get(id) else {
                return Err(invalid(format!("weight {key} is not a tensor")));
            };
            let key = if strip_module { &key["module.".len()..] } else { key };
            let k = normalize_key(key);
            if tensors.insert(k.clone(), TensorRef::clone(t)).is_some() {
                return Err(invalid(format!("duplicate weight {k}")));
            }
        }
        // Only what meta.rs reads, under one budget for the whole checkpoint.
        let mut budget = JsonBudget::METADATA;
        let mut meta = serde_json::Map::new();
        if kind != "state_dict" {
            for (key, id) in items {
                if !meta::USED_KEYS.contains(&key) {
                    continue;
                }
                if let Some(v) = root.to_json(id, &mut budget) {
                    meta.insert(key.to_string(), v);
                }
            }
        }
        Ok(Self { zip, prefix, tensors, meta, kind, storages: HashMap::new() })
    }

    pub fn tensor_names(&self) -> impl Iterator<Item = &str> {
        self.tensors.keys().map(String::as_str)
    }

    pub fn len(&self) -> usize {
        self.tensors.len()
    }

    pub fn has(&self, key: &str) -> bool {
        self.tensors.contains_key(key)
    }

    pub fn shape(&self, key: &str) -> Option<Vec<usize>> {
        self.tensors.get(key).map(|t| t.shape.iter().map(|&d| d as usize).collect())
    }

    pub fn dtype(&self, key: &str) -> Option<StorageDtype> {
        self.tensors.get(key).map(|t| t.storage.dtype)
    }

    fn storage(&mut self, t: &TensorRef) -> Result<Arc<Vec<u8>>, ImportError> {
        if let Some(s) = self.storages.get(&t.storage.key) {
            return Ok(s.clone());
        }
        let elem = t.storage.dtype.size() as u64;
        let want = t
            .storage
            .numel
            .checked_mul(elem)
            .filter(|&b| b <= ZipLimits::TORCH_CHECKPOINT.max_entry_bytes)
            .ok_or_else(|| invalid(format!("storage {} is too large", t.storage.key)))?;
        let name = format!("{}data/{}", self.prefix, t.storage.key);
        let bytes = self.zip.read(&name, want)?;
        if bytes.len() as u64 != want {
            return Err(invalid(format!(
                "storage {} holds {} bytes, its {} {} elements need {want}",
                t.storage.key,
                bytes.len(),
                t.storage.numel,
                t.storage.dtype.name()
            )));
        }
        let arc = Arc::new(bytes);
        self.storages.insert(t.storage.key.clone(), arc.clone());
        Ok(arc)
    }

    /// The tensor as a contiguous row-major f32 vector (fp16 / bf16 / f32 storages).
    pub fn read_f32(&mut self, key: &str) -> Result<Vec<f32>, ImportError> {
        let t = self
            .tensors
            .get(key)
            .cloned()
            .ok_or_else(|| ImportError::new(ErrorCode::MissingTensor, format!("the checkpoint has no {key}")))?;
        if !t.storage.dtype.is_float() {
            return Err(invalid(format!("{key} is a {} tensor, weights must be floating point", t.storage.dtype.name())));
        }
        // Consistent dtype for storages shared by several views.
        for other in self.tensors.values() {
            if other.storage.key == t.storage.key && (other.storage.dtype != t.storage.dtype || other.storage.numel != t.storage.numel) {
                return Err(invalid(format!("storage {} is declared with two types", t.storage.key)));
            }
        }
        let numel = t
            .shape
            .iter()
            .try_fold(1u64, |acc, &d| acc.checked_mul(d))
            .filter(|&n| n <= MAX_TENSOR_ELEMENTS)
            .ok_or_else(|| invalid(format!("{key}: tensor too large")))?;
        if numel > 0 {
            let last = t
                .shape
                .iter()
                .zip(&t.stride)
                .try_fold(t.offset, |acc, (&d, &s)| (d - 1).checked_mul(s).and_then(|x| acc.checked_add(x)));
            match last {
                Some(last) if last < t.storage.numel => {}
                _ => return Err(invalid(format!("{key}: tensor view reaches outside its storage"))),
            }
        }
        let storage = self.storage(&t)?;
        let elem = t.storage.dtype.size();
        let n = numel as usize;
        // Element indices of the view in row-major order.
        let contiguous = {
            let mut expect = 1u64;
            let mut ok = true;
            for (&d, &s) in t.shape.iter().zip(&t.stride).rev() {
                if d > 1 && s != expect {
                    ok = false;
                    break;
                }
                expect *= d;
            }
            ok
        };
        let mut raw = vec![0u8; n * elem];
        if contiguous {
            let start = t.offset as usize * elem;
            raw.copy_from_slice(&storage[start..start + n * elem]);
        } else {
            let rank = t.shape.len();
            let mut idx = vec![0u64; rank];
            for out in 0..n {
                let src = t.offset + idx.iter().zip(&t.stride).map(|(i, s)| i * s).sum::<u64>();
                let s = src as usize * elem;
                raw[out * elem..(out + 1) * elem].copy_from_slice(&storage[s..s + elem]);
                for d in (0..rank).rev() {
                    idx[d] += 1;
                    if idx[d] < t.shape[d] {
                        break;
                    }
                    idx[d] = 0;
                }
            }
        }
        let mut out = vec![0f32; n];
        match t.storage.dtype {
            StorageDtype::F32 => {
                for (o, c) in out.iter_mut().zip(raw.as_chunks::<4>().0) {
                    *o = f32::from_le_bytes(*c);
                }
            }
            StorageDtype::F16 => {
                let bits: Vec<half::f16> = raw.as_chunks::<2>().0.iter().map(|c| half::f16::from_bits(u16::from_le_bytes([c[0], c[1]]))).collect();
                bits.convert_to_f32_slice(&mut out);
            }
            StorageDtype::Bf16 => {
                let bits: Vec<half::bf16> = raw.as_chunks::<2>().0.iter().map(|c| half::bf16::from_bits(u16::from_le_bytes([c[0], c[1]]))).collect();
                bits.convert_to_f32_slice(&mut out);
            }
            StorageDtype::I64 | StorageDtype::I32 => unreachable!("checked above"),
        }
        Ok(out)
    }
}

/// Test hooks: edit a parsed checkpoint in place (architecture-mismatch tests in `import::tests`).
#[cfg(test)]
impl<R: Read + Seek> TorchCheckpoint<R> {
    /// Add `new_key` as another view of `existing`'s tensor.
    pub(crate) fn test_alias(&mut self, new_key: &str, existing: &str) {
        let t = self.tensors[existing].clone();
        self.tensors.insert(new_key.to_string(), t);
    }

    pub(crate) fn test_remove(&mut self, key: &str) {
        self.tensors.remove(key).expect("tensor to remove");
    }

    /// Declare `key` with another (contiguous) shape.
    pub(crate) fn test_set_shape(&mut self, key: &str, shape: &[u64]) {
        let t = self.tensors.get_mut(key).expect("tensor to reshape");
        t.shape = shape.to_vec();
        let mut stride = vec![1u64; shape.len()];
        for d in (0..shape.len().saturating_sub(1)).rev() {
            stride[d] = stride[d + 1] * shape[d + 1];
        }
        t.stride = stride;
    }

    /// Overwrite every byte of `key`'s storage with `byte`.
    pub(crate) fn test_fill_storage(&mut self, key: &str, byte: u8) {
        let t = self.tensors[key].clone();
        let len = self.storage(&t).expect("storage").len();
        self.storages.insert(t.storage.key.clone(), Arc::new(vec![byte; len]));
    }
}

#[cfg(test)]
pub(crate) mod tests {
    use super::super::pickle::tests::P;
    use super::super::zipguard::tests::build_zip;
    use super::*;
    use std::io::Cursor;
    use std::path::Path;
    use zip::CompressionMethod;

    /// A torch-format zip holding `pickle` and the given storages.
    pub(crate) fn torch_zip(pickle: &[u8], storages: &[(&str, Vec<u8>)]) -> Vec<u8> {
        let mut files: Vec<(String, Vec<u8>)> = vec![("ckpt/data.pkl".into(), pickle.to_vec()), ("ckpt/byteorder".into(), b"little".to_vec())];
        for (k, v) in storages {
            files.push((format!("ckpt/data/{k}"), v.clone()));
        }
        files.push(("ckpt/version".into(), b"3\n".to_vec()));
        let refs: Vec<(&str, &[u8], CompressionMethod)> =
            files.iter().map(|(n, d)| (n.as_str(), d.as_slice(), CompressionMethod::Stored)).collect();
        build_zip(&refs)
    }

    fn f16_bytes(v: &[f32]) -> Vec<u8> {
        v.iter().flat_map(|x| half::f16::from_f32(*x).to_bits().to_le_bytes()).collect()
    }

    fn checkpoint(weight_entries: P) -> P {
        P::new()
            .global("collections", "OrderedDict")
            .op(b')')
            .op(b'R')
            .op(b'(')
            .str("weight")
            .op(b'}')
            .op(b'(')
            .concat(weight_entries)
            .op(b'u')
            .str("sr")
            .str("40k")
            .str("f0")
            .op(b'K')
            .op(1)
            .op(b'u')
    }

    impl P {
        fn concat(mut self, other: P) -> P {
            self.0.extend_from_slice(&other.0[2..]); // drop the other's PROTO header
            self
        }
    }

    fn open(bytes: Vec<u8>) -> Result<TorchCheckpoint<Cursor<Vec<u8>>>, ImportError> {
        TorchCheckpoint::open(Cursor::new(bytes))
    }

    #[test]
    fn reads_contiguous_and_strided_views() {
        let w = P::new()
            .str("a.weight")
            .tensor("HalfStorage", "0", 6, 0, &[2, 3], &[3, 1])
            .str("a.parametrizations.weight.original0")
            .tensor("FloatStorage", "1", 4, 1, &[3], &[1])
            .str("t")
            .tensor("HalfStorage", "0", 6, 0, &[3, 2], &[1, 3]); // transposed view of storage 0
        let pkl = checkpoint(w).stop();
        let s0 = f16_bytes(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);
        let s1: Vec<u8> = [9.0f32, 10.0, 11.0, 12.0].iter().flat_map(|x| x.to_le_bytes()).collect();
        let mut ck = open(torch_zip(&pkl, &[("0", s0), ("1", s1)])).unwrap();
        assert_eq!(ck.len(), 3);
        assert!(ck.has("a.weight_g"), "parametrization key normalised");
        assert_eq!(ck.read_f32("a.weight").unwrap(), vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);
        assert_eq!(ck.read_f32("a.weight_g").unwrap(), vec![10.0, 11.0, 12.0]);
        assert_eq!(ck.read_f32("t").unwrap(), vec![1.0, 4.0, 2.0, 5.0, 3.0, 6.0]);
        assert_eq!(ck.meta.get("sr"), Some(&serde_json::json!("40k")));
        assert_eq!(ck.meta.get("f0"), Some(&serde_json::json!(1)));
        assert_eq!(ck.read_f32("missing").unwrap_err().code, ErrorCode::MissingTensor);
    }

    #[test]
    fn rejects_views_outside_storage_and_short_storages() {
        let w = P::new().str("x").tensor("FloatStorage", "0", 4, 2, &[2, 2], &[2, 1]);
        let s0: Vec<u8> = vec![0u8; 16];
        let mut ck = open(torch_zip(&checkpoint(w).stop(), &[("0", s0)])).unwrap();
        assert_eq!(ck.read_f32("x").unwrap_err().code, ErrorCode::TensorInvalid);

        let w = P::new().str("x").tensor("FloatStorage", "0", 4, 0, &[4], &[1]);
        let mut ck = open(torch_zip(&checkpoint(w).stop(), &[("0", vec![0u8; 12])])).unwrap();
        assert!(ck.read_f32("x").is_err());

        let w = P::new().str("x").tensor("LongStorage", "0", 1, 0, &[1], &[1]);
        let mut ck = open(torch_zip(&checkpoint(w).stop(), &[("0", vec![0u8; 8])])).unwrap();
        assert_eq!(ck.read_f32("x").unwrap_err().code, ErrorCode::TensorInvalid);
    }

    #[test]
    fn rejects_malicious_and_foreign_files() {
        // Code execution attempt inside an otherwise valid checkpoint.
        let evil = P::new().global("os", "system").str("calc.exe").op(0x85).op(b'R').stop();
        let err = open(torch_zip(&evil, &[])).err().unwrap();
        assert_eq!(err.code, ErrorCode::PickleRejected);
        assert!(err.message.contains("os.system"), "{}", err.message);

        // Legacy pickle and random bytes.
        assert_eq!(reject_non_zip(&[0x80, 0x02, 0x8a]).code, ErrorCode::LegacyFormat);
        assert_eq!(reject_non_zip(b"RIFF").code, ErrorCode::NotACheckpoint);

        // TorchScript archive.
        let ts = build_zip(&[
            ("m/constants.pkl", b"x", CompressionMethod::Stored),
            ("m/code/__torch__.py", b"x", CompressionMethod::Stored),
        ]);
        assert_eq!(open(ts).err().unwrap().code, ErrorCode::TorchScript);

        // Training checkpoint, and a dict without weights.
        let training = P::new().op(b'}').str("model").op(b'}').op(b's').stop();
        assert_eq!(open(torch_zip(&training, &[])).err().unwrap().code, ErrorCode::NotACheckpoint);
        let other = P::new().op(b'}').str("x").op(b'K').op(1).op(b's').stop();
        assert_eq!(open(torch_zip(&other, &[])).err().unwrap().code, ErrorCode::NotACheckpoint);
    }

    #[test]
    fn metadata_is_limited_to_used_keys_and_bounded() {
        let w = P::new().str("x").tensor("FloatStorage", "0", 1, 0, &[1], &[1]);
        // After the standard entries: an unused key holding a large list, and a long info string.
        let mut junk = P::new().str("junk").op(b']');
        for i in 0..50_000 {
            junk = junk.int(i).op(b'a');
        }
        let info = "i".repeat(10_000);
        let pkl = checkpoint(w).concat(junk).op(b's').str("info").str(&info).op(b's').stop();
        let ck = open(torch_zip(&pkl, &[("0", vec![0u8; 4])])).unwrap();
        assert!(!ck.meta.contains_key("junk"), "unused entries are not converted");
        assert_eq!(ck.meta["info"].as_str().unwrap().len(), 4096, "long strings are cut");
        assert_eq!(ck.meta["sr"], serde_json::json!("40k"));

        // More weights than any voice has: refused before they are collected.
        let mut w = P::new().str("w0").tensor("FloatStorage", "0", 1, 0, &[1], &[1]).put(0);
        for i in 1..=MAX_WEIGHTS {
            w = w.str(&format!("w{i}")).get(0);
        }
        let err = open(torch_zip(&checkpoint(w).stop(), &[("0", vec![0u8; 4])])).err().unwrap();
        assert_eq!(err.code, ErrorCode::TensorInvalid, "{err}");
    }

    /// Two generator tensors (the second a parametrized weight_norm key), optionally prefixed.
    fn generator_entries(prefix: &str) -> P {
        P::new()
            .str(&format!("{prefix}enc_p.emb_phone.weight"))
            .tensor("FloatStorage", "0", 4, 0, &[2, 2], &[2, 1])
            .str(&format!("{prefix}dec.ups.0.parametrizations.weight.original0"))
            .tensor("FloatStorage", "1", 2, 0, &[2], &[1])
    }

    fn storages() -> Vec<(&'static str, Vec<u8>)> {
        let f32s = |v: &[f32]| v.iter().flat_map(|x| x.to_le_bytes()).collect::<Vec<u8>>();
        vec![("0", f32s(&[1.0, 2.0, 3.0, 4.0])), ("1", f32s(&[5.0, 6.0]))]
    }

    #[test]
    fn training_checkpoints_and_bare_state_dicts() {
        // G_*.pth: {"model": {...}, "iteration": 5, "optimizer": {...}, "sr": "40k"}; the optimizer
        // state (tensors of its own) is never read.
        let pkl = P::new()
            .op(b'}')
            .op(b'(')
            .str("model")
            .op(b'}')
            .op(b'(')
            .concat(generator_entries(""))
            .op(b'u')
            .str("iteration")
            .int(5)
            .str("optimizer")
            .op(b'}')
            .str("exp_avg")
            .tensor("FloatStorage", "9", 1, 0, &[1], &[1])
            .op(b's')
            .str("sr")
            .str("40k")
            .op(b'u')
            .stop();
        let mut ck = open(torch_zip(&pkl, &storages())).unwrap();
        assert_eq!(ck.kind, "training");
        assert_eq!(ck.len(), 2);
        assert!(ck.has("dec.ups.0.weight_g"));
        assert_eq!(ck.read_f32("enc_p.emb_phone.weight").unwrap(), vec![1.0, 2.0, 3.0, 4.0]);
        assert_eq!(ck.meta.get("sr"), Some(&serde_json::json!("40k")));

        // A bare state dict saved from DataParallel: the `module.` prefix is dropped.
        let pkl = P::new().op(b'}').op(b'(').concat(generator_entries("module.")).op(b'u').stop();
        let mut ck = open(torch_zip(&pkl, &storages())).unwrap();
        assert_eq!(ck.kind, "state_dict");
        assert!(ck.meta.is_empty());
        assert_eq!(ck.read_f32("dec.ups.0.weight_g").unwrap(), vec![5.0, 6.0]);

        // A discriminator checkpoint, and a dict of tensors that is not a generator.
        let pkl = P::new().op(b'}').str("model").op(b'}').str("disc.w").tensor("FloatStorage", "0", 4, 0, &[4], &[1]).op(b's').op(b's').stop();
        let err = open(torch_zip(&pkl, &storages())).err().unwrap();
        assert_eq!(err.code, ErrorCode::NotACheckpoint);
        assert!(err.message.contains("discriminator"), "{err}");
        let pkl = P::new().op(b'}').str("other.weight").tensor("FloatStorage", "0", 4, 0, &[4], &[1]).op(b's').stop();
        assert_eq!(open(torch_zip(&pkl, &storages())).err().unwrap().code, ErrorCode::NotACheckpoint);
        // An inference export keeps working with a `weight` entry next to `model`-like keys.
        let pkl = checkpoint(generator_entries("")).stop();
        assert_eq!(open(torch_zip(&pkl, &storages())).unwrap().kind, "inference");
    }

    /// Real `torch.save` output (`test-fixtures/vc/fuzz/torch-made`, made with PyTorch 2.3.1): a
    /// state dict pickles its `_metadata` table with `BUILD`, which every training checkpoint and
    /// bare state dict carries; all three shapes are read, and every tensor matches PyTorch's.
    #[test]
    fn real_torch_saves_are_read() {
        let dir = Path::new(env!("CARGO_MANIFEST_DIR")).join("test-fixtures").join("vc").join("fuzz").join("torch-made");
        let expected: serde_json::Value = serde_json::from_slice(&std::fs::read(dir.join("expected.json")).unwrap()).unwrap();
        for (file, kind, table) in [("state-dict.pth", "state_dict", "state_dict"), ("data-parallel.pth", "state_dict", "state_dict"), ("G_7.pth", "training", "G_7")] {
            let mut ck = TorchCheckpoint::open(std::fs::File::open(dir.join(file)).unwrap()).unwrap_or_else(|e| panic!("{file}: {e}"));
            assert_eq!(ck.kind, kind, "{file}");
            let want = expected[table].as_object().unwrap();
            assert_eq!(ck.len(), want.len(), "{file}: {:?}", ck.tensor_names().collect::<Vec<_>>());
            for (name, values) in want {
                let values: Vec<f32> = values.as_array().unwrap().iter().map(|v| v.as_f64().unwrap() as f32).collect();
                assert_eq!(ck.read_f32(name).unwrap_or_else(|e| panic!("{file} {name}: {e}")), values, "{file} {name}");
            }
            assert!(ck.has("dec.conv_post.weight_g") && ck.has("dec.ups.0.weight_v"), "{file}: weight_norm keys normalised");
        }
    }

    #[test]
    fn key_normalisation() {
        assert_eq!(normalize_key("dec.ups.0.parametrizations.weight.original0"), "dec.ups.0.weight_g");
        assert_eq!(normalize_key("dec.ups.0.parametrizations.weight.original1"), "dec.ups.0.weight_v");
        assert_eq!(normalize_key("dec.ups.0.weight_v"), "dec.ups.0.weight_v");
        assert_eq!(normalize_key("x.original0"), "x.original0");
    }
}
