//! A DDSP-SVC training checkpoint (`model_*.pt`, P5-A): the torch zip format read with the
//! importer's allow-list pickle VM (`import::pickle`) and zip guard (`import::zipguard`), so a
//! hostile file can neither run code nor exhaust memory. Only ever opened in the isolated import
//! worker (`Sigil.exe --headless vc-import-worker`) or, where the worker is unavailable, under the
//! same limits in process.
//!
//! Upstream's trainer saves `{"global_step": n, "model": state_dict}` (plus `"optimizer"` with
//! `save_opt: true`); a bare state dict is accepted too. The state dict holds `ddsp_model.*` and
//! `reflow_model.*` tensors. Weight-norm parametrizations written by newer PyTorch
//! (`X.parametrizations.weight.original0/1`) are read as `X.weight_g` / `X.weight_v`, and a
//! `module.` prefix on every key (a DataParallel save) is dropped. Other top-level entries (the
//! optimizer state) are never converted or read.

use std::collections::{BTreeMap, HashMap};
use std::io::{Read, Seek};
use std::sync::Arc;

use half::slice::HalfFloatSliceExt;

use crate::services::neural_audio::vc::import::pickle::{self, PickleLimits, Pickled, StorageDtype, TensorRef, Value};
use crate::services::neural_audio::vc::import::pth::normalize_key;
use crate::services::neural_audio::vc::import::zipguard::{GuardedZip, ZipLimits};
use crate::services::neural_audio::vc::import::{ErrorCode, ImportError};

/// Largest `data.pkl` accepted (an optimizer state makes it bigger than the model's).
const MAX_PICKLE_BYTES: u64 = 64 << 20;
/// Most tensors in the state dict (the 6.3 default model: about 320).
const MAX_TENSORS: usize = 16_384;
/// Largest single tensor (elements).
const MAX_TENSOR_ELEMENTS: u64 = 1 << 30;
const MAX_KEY: usize = 512;
/// What makes a dict a DDSP-SVC state dict.
const PREFIXES: [&str; 2] = ["ddsp_model.", "reflow_model."];

fn invalid(msg: impl Into<String>) -> ImportError {
    ImportError::new(ErrorCode::TensorInvalid, msg)
}

/// A parsed DDSP-SVC checkpoint: tensor views and the training step; storages are read on demand.
pub struct DdspCheckpoint<R: Read + Seek> {
    zip: GuardedZip<R>,
    prefix: String,
    tensors: BTreeMap<String, TensorRef>,
    storages: HashMap<String, Arc<Vec<u8>>>,
    /// `global_step` of a training checkpoint.
    pub global_step: Option<i64>,
    /// `training` (`{"model": ...}`) or `state_dict`.
    pub kind: &'static str,
}

/// Whether `id` is a dict of tensors that looks like a DDSP-SVC state dict.
fn is_state_dict(root: &Pickled, id: pickle::Id) -> bool {
    let Some(items) = root.str_items(id) else {
        return false;
    };
    !items.is_empty()
        && items.iter().any(|(k, v)| {
            let k = k.strip_prefix("module.").unwrap_or(k);
            PREFIXES.iter().any(|p| k.starts_with(p)) && matches!(root.get(*v), Value::Tensor(_))
        })
}

impl<R: Read + Seek> DdspCheckpoint<R> {
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
            [] if zip.entries().iter().any(|e| e.name.ends_with("constants.pkl") || e.name.contains("/code/")) => {
                return Err(ImportError::new(ErrorCode::TorchScript, "this is a TorchScript module, not a checkpoint of weights"));
            }
            [] => return Err(ImportError::new(ErrorCode::NotACheckpoint, "zip archive without a torch data.pkl")),
            _ => return Err(ImportError::new(ErrorCode::NotACheckpoint, "more than one data.pkl in the archive")),
        };
        let prefix = name.trim_end_matches("data.pkl").to_string();
        if zip.entries().iter().any(|e| e.name.starts_with(&format!("{prefix}code/")) || e.name == format!("{prefix}constants.pkl")) {
            return Err(ImportError::new(ErrorCode::TorchScript, "this is a TorchScript module, not a checkpoint of weights"));
        }
        if zip.find(&format!("{prefix}byteorder")).is_some() && zip.read(&format!("{prefix}byteorder"), 16)? != b"little" {
            return Err(invalid("big-endian checkpoints are not supported"));
        }
        let data = zip.read(&name, MAX_PICKLE_BYTES)?;
        let root = pickle::unpickle(&data, &PickleLimits::default()).map_err(|e| ImportError::new(ErrorCode::PickleRejected, e.to_string()))?;
        let items = root
            .str_items(root.root)
            .ok_or_else(|| ImportError::new(ErrorCode::NotACheckpoint, "the checkpoint is not a dict"))?;
        let (dict, kind) = match items.iter().find(|(k, _)| *k == "model") {
            Some((_, id)) if is_state_dict(&root, *id) => (*id, "training"),
            Some(_) => {
                return Err(ImportError::new(
                    ErrorCode::NotACheckpoint,
                    "its `model` entry is not a DDSP-SVC model (no ddsp_model.* / reflow_model.* weights)",
                ))
            }
            None if is_state_dict(&root, root.root) => (root.root, "state_dict"),
            None if items.iter().any(|(k, _)| *k == "weight" || *k == "params") => {
                return Err(ImportError::new(
                    ErrorCode::UnsupportedModel,
                    "this looks like an RVC or so-vits-svc voice, not DDSP-SVC: import it as that kind of voice",
                ))
            }
            None => return Err(ImportError::new(ErrorCode::NotACheckpoint, "no `model` entry: not a DDSP-SVC checkpoint")),
        };
        let global_step = items.iter().find(|(k, _)| *k == "global_step").and_then(|(_, id)| match root.get(*id) {
            Value::Int(n) => Some(*n),
            _ => None,
        });
        let entries = root.str_items(dict).unwrap_or_default();
        if entries.len() > MAX_TENSORS {
            return Err(invalid(format!("{} tensors (at most {MAX_TENSORS})", entries.len())));
        }
        let strip_module = entries.iter().all(|(k, _)| k.starts_with("module."));
        let mut tensors = BTreeMap::new();
        for (key, id) in entries {
            if key.len() > MAX_KEY {
                return Err(invalid(format!("a weight name of {} bytes", key.len())));
            }
            let Value::Tensor(t) = root.get(id) else {
                return Err(invalid(format!("{key} is not a tensor")));
            };
            let key = if strip_module { &key["module.".len()..] } else { key };
            let k = normalize_key(key);
            if tensors.insert(k.clone(), TensorRef::clone(t)).is_some() {
                return Err(invalid(format!("duplicate weight {k}")));
            }
        }
        Ok(Self { zip, prefix, tensors, storages: HashMap::new(), global_step, kind })
    }

    pub fn keys(&self) -> impl Iterator<Item = &str> {
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
        let want = t
            .storage
            .numel
            .checked_mul(t.storage.dtype.size() as u64)
            .filter(|&b| b <= ZipLimits::TORCH_CHECKPOINT.max_entry_bytes)
            .ok_or_else(|| invalid(format!("storage {} is too large", t.storage.key)))?;
        let name = format!("{}data/{}", self.prefix, t.storage.key);
        let bytes = self.zip.read(&name, want)?;
        if bytes.len() as u64 != want {
            return Err(invalid(format!("storage {} holds {} bytes, {want} expected", t.storage.key, bytes.len())));
        }
        let arc = Arc::new(bytes);
        self.storages.insert(t.storage.key.clone(), arc.clone());
        Ok(arc)
    }

    /// Drop cached storages (they are read again on demand).
    pub fn release_storages(&mut self) {
        self.storages.clear();
    }

    /// The tensor as contiguous row-major f32 (float32 / float16 / bfloat16 storages).
    pub fn read_f32(&mut self, key: &str) -> Result<Vec<f32>, ImportError> {
        let t = self
            .tensors
            .get(key)
            .cloned()
            .ok_or_else(|| ImportError::new(ErrorCode::MissingTensor, format!("the checkpoint has no {key}")))?;
        if !t.storage.dtype.is_float() {
            return Err(invalid(format!("{key} is a {} tensor; weights must be floating point", t.storage.dtype.name())));
        }
        for other in self.tensors.values() {
            if other.storage.key == t.storage.key && (other.storage.dtype != t.storage.dtype || other.storage.numel != t.storage.numel) {
                return Err(invalid(format!("storage {} is declared with two types", t.storage.key)));
            }
        }
        if t.shape.len() != t.stride.len() {
            return Err(invalid(format!("{key}: shape and stride ranks differ")));
        }
        let numel = t
            .shape
            .iter()
            .try_fold(1u64, |acc, &d| acc.checked_mul(d))
            .filter(|&n| n <= MAX_TENSOR_ELEMENTS)
            .ok_or_else(|| invalid(format!("{key}: tensor too large")))?;
        // A saved weight never holds more values than its storage: a larger view repeats storage
        // values (a stride-0 broadcast), which no trained weight is, and would expand a few stored
        // bytes into gigabytes here.
        if numel > t.storage.numel {
            return Err(invalid(format!("{key}: a view of {numel} values over a storage of {} (a broadcast, not a weight)", t.storage.numel)));
        }
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
        let mut raw = vec![0u8; n * elem];
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
                for (o, c) in out.iter_mut().zip(raw.chunks_exact(4)) {
                    *o = f32::from_le_bytes([c[0], c[1], c[2], c[3]]);
                }
            }
            StorageDtype::F16 => {
                let bits: Vec<half::f16> = raw.chunks_exact(2).map(|c| half::f16::from_bits(u16::from_le_bytes([c[0], c[1]]))).collect();
                bits.convert_to_f32_slice(&mut out);
            }
            StorageDtype::Bf16 => {
                let bits: Vec<half::bf16> = raw.chunks_exact(2).map(|c| half::bf16::from_bits(u16::from_le_bytes([c[0], c[1]]))).collect();
                bits.convert_to_f32_slice(&mut out);
            }
            StorageDtype::I64 | StorageDtype::I32 => unreachable!("checked above"),
        }
        Ok(out)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::services::neural_audio::vc::import::pickle::tests::P;
    use crate::services::neural_audio::vc::import::pth::tests::torch_zip;
    use std::io::Cursor;

    fn f32_bytes(v: &[f32]) -> Vec<u8> {
        v.iter().flat_map(|x| x.to_le_bytes()).collect()
    }

    /// `{"global_step": 7, "model": {key: tensor}}` with one f32 storage per key.
    fn training(keys: &[(&str, &[i32])]) -> Vec<u8> {
        let mut p = P::new().op(b'}').op(b'(').str("global_step").int(7).str("model").op(b'}').op(b'(');
        let mut storages = Vec::new();
        for (i, (k, shape)) in keys.iter().enumerate() {
            let numel: i32 = shape.iter().product();
            let mut stride = vec![1i32; shape.len()];
            for d in (0..shape.len().saturating_sub(1)).rev() {
                stride[d] = stride[d + 1] * shape[d + 1];
            }
            p = p.str(k).tensor("FloatStorage", &i.to_string(), numel, 0, shape, &stride);
            storages.push((i.to_string(), f32_bytes(&(0..numel).map(|x| x as f32 + i as f32).collect::<Vec<_>>())));
        }
        let pickle = p.op(b'u').op(b'u').stop();
        let st: Vec<(&str, Vec<u8>)> = storages.iter().map(|(k, v)| (k.as_str(), v.clone())).collect();
        torch_zip(&pickle, &st)
    }

    #[test]
    fn reads_a_training_checkpoint() {
        let zip = training(&[("ddsp_model.unit2ctrl.norm.weight", &[3]), ("reflow_model.velocity_fn.norm.bias", &[2, 2])]);
        let mut c = DdspCheckpoint::open(Cursor::new(zip)).unwrap();
        assert_eq!(c.kind, "training");
        assert_eq!(c.global_step, Some(7));
        assert_eq!(c.len(), 2);
        assert_eq!(c.shape("reflow_model.velocity_fn.norm.bias"), Some(vec![2, 2]));
        assert_eq!(c.read_f32("ddsp_model.unit2ctrl.norm.weight").unwrap(), vec![0.0, 1.0, 2.0]);
        assert_eq!(c.read_f32("reflow_model.velocity_fn.norm.bias").unwrap(), vec![1.0, 2.0, 3.0, 4.0]);
        assert_eq!(c.read_f32("nope").unwrap_err().code, ErrorCode::MissingTensor);
    }

    /// A view with more values than its storage (stride 0) is refused before anything is
    /// allocated for it; a view inside its storage reads.
    #[test]
    fn refuses_broadcast_views() {
        let p = P::new()
            .op(b'}')
            .op(b'(')
            .str("global_step")
            .int(1)
            .str("model")
            .op(b'}')
            .op(b'(')
            .str("ddsp_model.unit2ctrl.spk_embed.weight")
            .tensor("FloatStorage", "0", 4, 0, &[1 << 24, 4], &[0, 1])
            .str("ddsp_model.unit2ctrl.norm.weight")
            .tensor("FloatStorage", "0", 4, 1, &[3], &[1])
            .op(b'u')
            .op(b'u')
            .stop();
        let zip = torch_zip(&p, &[("0", f32_bytes(&[1.0, 2.0, 3.0, 4.0]))]);
        let mut c = DdspCheckpoint::open(Cursor::new(zip)).unwrap();
        let e = c.read_f32("ddsp_model.unit2ctrl.spk_embed.weight").unwrap_err();
        assert!(e.code == ErrorCode::TensorInvalid && e.message.contains("broadcast"), "{e}");
        assert_eq!(c.read_f32("ddsp_model.unit2ctrl.norm.weight").unwrap(), vec![2.0, 3.0, 4.0]);
    }

    #[test]
    fn refuses_other_checkpoints() {
        // An RVC-shaped `model` (no DDSP keys) and a code-running global.
        let rvc = training(&[("enc_p.emb_phone.weight", &[2])]);
        assert_eq!(DdspCheckpoint::open(Cursor::new(rvc)).err().unwrap().code, ErrorCode::NotACheckpoint);
        let evil = P::new().global("os", "system").str("echo").op(0x85).op(b'R').stop();
        let zip = torch_zip(&evil, &[]);
        assert_eq!(DdspCheckpoint::open(Cursor::new(zip)).err().unwrap().code, ErrorCode::PickleRejected);
        assert_eq!(DdspCheckpoint::open(Cursor::new(b"not a zip".to_vec())).err().unwrap().code, ErrorCode::ArchiveRejected);
    }

    #[test]
    fn reads_the_tiny_fixture_checkpoint() {
        let dir = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("test-fixtures/vc/ddsp_svc");
        let f = std::fs::File::open(dir.join("model_100.pt")).unwrap();
        let mut c = DdspCheckpoint::open(std::io::BufReader::new(f)).unwrap();
        assert_eq!((c.kind, c.global_step), ("training", Some(100)));
        assert!(c.has("ddsp_model.unit2ctrl.stack.0.weight_g") && c.has("ddsp_model.unit2ctrl.stack.0.weight_v"));
        assert!(c.has("reflow_model.velocity_fn.output_projection.weight"));
        let w = c.read_f32("ddsp_model.unit2ctrl.norm.weight").unwrap();
        assert_eq!(w.len(), 8);
        assert!(w.iter().all(|x| x.is_finite() && (*x - 1.0).abs() < 0.1), "{w:?}");
    }
}
