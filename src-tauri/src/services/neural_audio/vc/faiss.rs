//! RVC retrieval index (`.index`) reader + k-NN feature blend.
//!
//! Reads the binary layout written by `faiss::write_index` (faiss/impl/index_write.cpp, read back
//! by faiss/impl/index_read.cpp) for the index types RVC produces:
//!
//! - `IwFl` = `IndexIVFFlat` (RVC's `added_IVF<n>_Flat_nprobe_1_*.index`), whose coarse quantizer
//!   is an `IndexFlat` and whose inverted lists are `ilar` (`ArrayInvertedLists`, "full" or
//!   "sprs" sizes);
//! - `IxF2` / `IxFI` / `IxFl` = `IndexFlatL2` / `IndexFlatIP` / `IndexFlat`, either as the
//!   quantizer or as a top-level index.
//!
//! Every value is little-endian: `int` = i32, `idx_t` = i64, `size_t` = u64, `bool` = u8,
//! fourcc = 4 ASCII bytes. The legacy raw dump (`u32 dim, u32 n, n*dim f32`) is still accepted.
//!
//! Retrieval mirrors RVC (`infer/modules/vc/pipeline.py`): `k = 8` neighbours, weights
//! `(1 / score)^2` with `score` = FAISS squared-L2 distance, normalized per frame, then
//! `feats = neighbours * index_rate + feats * (1 - index_rate)`.
//! When no index is loaded or `index_rate == 0`, features pass through unchanged.

use std::fs;
use std::path::Path;

/// Neighbours per frame (RVC `index.search(npy, k=8)`).
const K_NEIGHBORS: usize = 8;
/// FAISS rejects `d` above this (`FAISS_CHECK_RANGE(idx.d, 0, (1 << 20) + 1)`).
const MAX_DIM: usize = 1 << 20;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Metric {
    InnerProduct,
    L2,
}

impl Metric {
    fn from_faiss(metric_type: i32) -> Result<Self, String> {
        match metric_type {
            0 => Ok(Self::InnerProduct),
            1 => Ok(Self::L2),
            other => Err(format!("unsupported FAISS metric type {other} (only L2 and inner product)")),
        }
    }

    /// Smaller is closer.
    fn badness(self, query: &[f32], row: &[f32]) -> f32 {
        match self {
            Self::L2 => squared_l2(query, row),
            Self::InnerProduct => -dot(query, row),
        }
    }
}

/// Row-major `n × dim` float vectors.
struct Vectors {
    dim: usize,
    data: Vec<f32>,
}

impl Vectors {
    fn len(&self) -> usize {
        if self.dim == 0 {
            0
        } else {
            self.data.len() / self.dim
        }
    }

    fn rows(&self) -> std::slice::ChunksExact<'_, f32> {
        self.data.chunks_exact(self.dim.max(1))
    }
}

enum Store {
    Flat {
        metric: Metric,
        vectors: Vectors,
    },
    IvfFlat {
        metric: Metric,
        nprobe: usize,
        quantizer_metric: Metric,
        centroids: Vectors,
        lists: Vec<Vectors>,
    },
}

/// Retrieval index over HuBERT feature rows (dim = the model's embedding channels).
pub struct FaissIndex {
    dim: usize,
    ntotal: usize,
    store: Store,
}

impl FaissIndex {
    pub fn load(path: &Path) -> Result<Self, String> {
        let bytes = fs::read(path).map_err(|e| format!("Cannot read index {}: {}", path.display(), e))?;
        Self::from_bytes(&bytes)
    }

    pub fn from_bytes(bytes: &[u8]) -> Result<Self, String> {
        let index = if looks_like_faiss(bytes) {
            let mut r = Reader::new(bytes);
            let parsed = read_index(&mut r, 0)?;
            let dim = parsed.dim;
            let ntotal = parsed.ntotal;
            if ntotal == 0 {
                return Err("FAISS index holds no vectors (use RVC's added_*.index, not trained_*.index)".into());
            }
            FaissIndex {
                dim,
                ntotal,
                store: parsed.store,
            }
        } else {
            read_raw_dump(bytes)?
        };
        Ok(index)
    }

    pub fn dim(&self) -> usize {
        self.dim
    }

    /// Up to `k` nearest stored rows for `query` (closest first).
    fn search<'a>(&'a self, query: &[f32], k: usize) -> Vec<&'a [f32]> {
        let mut top = TopK::new(k);
        match &self.store {
            Store::Flat { metric, vectors } => {
                for row in vectors.rows() {
                    top.push(metric.badness(query, row), row);
                }
            }
            Store::IvfFlat {
                metric,
                nprobe,
                quantizer_metric,
                centroids,
                lists,
            } => {
                let mut probe = TopK::new((*nprobe).max(1));
                for (list_no, centroid) in centroids.rows().enumerate() {
                    probe.push_id(quantizer_metric.badness(query, centroid), list_no);
                }
                for list_no in probe.ids() {
                    if let Some(list) = lists.get(list_no) {
                        for row in list.rows() {
                            top.push(metric.badness(query, row), row);
                        }
                    }
                }
            }
        }
        top.into_rows()
    }

    /// RVC retrieval blend over `frames` rows of `features` (`frames × dim`, row-major).
    pub fn blend_inplace(&self, features: &mut [f32], frames: usize, index_rate: f32) {
        let rate = index_rate.clamp(0.0, 1.0);
        if rate <= 0.0 || self.ntotal == 0 || self.dim == 0 || frames == 0 {
            return;
        }
        let dim = self.dim;
        let mut blended = vec![0.0f32; dim];
        let mut weights: Vec<f64> = Vec::with_capacity(K_NEIGHBORS);
        for t in 0..frames {
            let start = t * dim;
            if start + dim > features.len() {
                break;
            }
            let neighbours = self.search(&features[start..start + dim], K_NEIGHBORS);
            if neighbours.is_empty() {
                continue;
            }
            let query = &features[start..start + dim];
            weights.clear();
            let mut exact: Option<&[f32]> = None;
            for row in &neighbours {
                let score = f64::from(squared_l2(query, row));
                if score <= 1e-12 {
                    exact = Some(row);
                    break;
                }
                weights.push(1.0 / (score * score));
            }
            blended.fill(0.0);
            if let Some(row) = exact {
                blended.copy_from_slice(row);
            } else {
                let sum: f64 = weights.iter().sum();
                if !(sum.is_finite() && sum > 0.0) {
                    continue;
                }
                for (row, w) in neighbours.iter().zip(weights.iter()) {
                    let w = (w / sum) as f32;
                    for (b, &v) in blended.iter_mut().zip(row.iter()) {
                        *b += w * v;
                    }
                }
            }
            let out = &mut features[start..start + dim];
            for (o, &b) in out.iter_mut().zip(blended.iter()) {
                *o = b * rate + *o * (1.0 - rate);
            }
        }
    }
}

/// Smallest-`badness` selection for small `k` (sorted insertion).
struct TopK<'a> {
    k: usize,
    items: Vec<(f32, &'a [f32], usize)>,
}

impl<'a> TopK<'a> {
    fn new(k: usize) -> Self {
        Self {
            k: k.max(1),
            items: Vec::with_capacity(k.max(1) + 1),
        }
    }

    fn insert(&mut self, badness: f32, row: &'a [f32], id: usize) {
        if badness.is_nan() {
            return;
        }
        if self.items.len() == self.k {
            match self.items.last() {
                Some(last) if badness < last.0 => {
                    self.items.pop();
                }
                _ => return,
            }
        }
        let pos = self.items.partition_point(|it| it.0 <= badness);
        self.items.insert(pos, (badness, row, id));
    }

    fn push(&mut self, badness: f32, row: &'a [f32]) {
        self.insert(badness, row, 0);
    }

    fn push_id(&mut self, badness: f32, id: usize) {
        self.insert(badness, &[], id);
    }

    fn ids(&self) -> Vec<usize> {
        self.items.iter().map(|it| it.2).collect()
    }

    fn into_rows(self) -> Vec<&'a [f32]> {
        self.items.into_iter().map(|it| it.1).collect()
    }
}

fn squared_l2(a: &[f32], b: &[f32]) -> f32 {
    let mut acc = [0.0f32; 8];
    let mut ca = a.chunks_exact(8);
    let mut cb = b.chunks_exact(8);
    for (x, y) in (&mut ca).zip(&mut cb) {
        for ((acc_j, &xj), &yj) in acc.iter_mut().zip(x).zip(y) {
            let d = xj - yj;
            *acc_j += d * d;
        }
    }
    let mut sum: f32 = acc.iter().sum();
    for (x, y) in ca.remainder().iter().zip(cb.remainder().iter()) {
        let d = x - y;
        sum += d * d;
    }
    sum
}

fn dot(a: &[f32], b: &[f32]) -> f32 {
    let mut acc = [0.0f32; 8];
    let mut ca = a.chunks_exact(8);
    let mut cb = b.chunks_exact(8);
    for (x, y) in (&mut ca).zip(&mut cb) {
        for ((acc_j, &xj), &yj) in acc.iter_mut().zip(x).zip(y) {
            *acc_j += xj * yj;
        }
    }
    let mut sum: f32 = acc.iter().sum();
    for (x, y) in ca.remainder().iter().zip(cb.remainder().iter()) {
        sum += x * y;
    }
    sum
}

/// FAISS files start with an index fourcc such as `IwFl` / `IxF2`; the legacy raw dump starts
/// with a small little-endian `u32` dimension, whose high bytes are zero.
fn looks_like_faiss(bytes: &[u8]) -> bool {
    bytes.len() >= 4 && bytes[0] == b'I' && bytes[1..4].iter().all(|b| b.is_ascii_alphanumeric())
}

/// Legacy Sigil layout: little-endian `u32 dim, u32 n`, then `n * dim` `f32`.
fn read_raw_dump(bytes: &[u8]) -> Result<FaissIndex, String> {
    let mut r = Reader::new(bytes);
    let dim = r
        .u32()
        .map_err(|_| "RVC index file too small".to_string())? as usize;
    let n = r
        .u32()
        .map_err(|_| "RVC index file too small".to_string())? as usize;
    let count = n.saturating_mul(dim);
    if dim == 0 || n == 0 || count > r.remaining() / 4 {
        return Err(format!(
            "RVC index is neither a FAISS index nor a raw dump (dim={}, n={}, bytes={})",
            dim,
            n,
            bytes.len()
        ));
    }
    let data = r.f32_vec(count)?;
    Ok(FaissIndex {
        dim,
        ntotal: n,
        store: Store::Flat {
            metric: Metric::L2,
            vectors: Vectors { dim, data },
        },
    })
}

struct Reader<'a> {
    buf: &'a [u8],
    pos: usize,
}

impl<'a> Reader<'a> {
    fn new(buf: &'a [u8]) -> Self {
        Self { buf, pos: 0 }
    }

    fn remaining(&self) -> usize {
        self.buf.len() - self.pos
    }

    fn take(&mut self, n: usize) -> Result<&'a [u8], String> {
        if n > self.remaining() {
            return Err(format!(
                "FAISS index truncated: need {} bytes at offset {}, {} left",
                n,
                self.pos,
                self.remaining()
            ));
        }
        let s = &self.buf[self.pos..self.pos + n];
        self.pos += n;
        Ok(s)
    }

    fn array<const N: usize>(&mut self) -> Result<[u8; N], String> {
        let mut out = [0u8; N];
        out.copy_from_slice(self.take(N)?);
        Ok(out)
    }

    fn u8(&mut self) -> Result<u8, String> {
        Ok(self.take(1)?[0])
    }

    fn u32(&mut self) -> Result<u32, String> {
        self.array().map(u32::from_le_bytes)
    }

    fn i32(&mut self) -> Result<i32, String> {
        self.array().map(i32::from_le_bytes)
    }

    fn i64(&mut self) -> Result<i64, String> {
        self.array().map(i64::from_le_bytes)
    }

    fn f32(&mut self) -> Result<f32, String> {
        self.array().map(f32::from_le_bytes)
    }

    /// A `size_t` that counts `elem_bytes`-sized items still to be read from this buffer.
    fn count(&mut self, elem_bytes: usize) -> Result<usize, String> {
        let n = self.array().map(u64::from_le_bytes)?;
        let n = usize::try_from(n).map_err(|_| format!("FAISS count {n} out of range"))?;
        if elem_bytes > 0 && n > self.remaining() / elem_bytes {
            return Err(format!("FAISS count {} exceeds the remaining {} bytes", n, self.remaining()));
        }
        Ok(n)
    }

    fn fourcc(&mut self) -> Result<[u8; 4], String> {
        self.array()
    }

    fn skip(&mut self, n: usize) -> Result<(), String> {
        self.take(n).map(|_| ())
    }

    fn f32_vec(&mut self, count: usize) -> Result<Vec<f32>, String> {
        let bytes = self.take(
            count
                .checked_mul(4)
                .ok_or_else(|| "FAISS vector size overflows".to_string())?,
        )?;
        Ok(bytes
            .chunks_exact(4)
            .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
            .collect())
    }
}

fn fourcc_str(h: &[u8; 4]) -> String {
    String::from_utf8_lossy(h).into_owned()
}

struct IndexHeader {
    dim: usize,
    ntotal: usize,
    metric: Metric,
}

/// `read_index_header`: d (int), ntotal (idx_t), two idx_t dummies, is_trained (bool),
/// metric_type (int), metric_arg (float, only when metric_type > 1).
fn read_header(r: &mut Reader<'_>) -> Result<IndexHeader, String> {
    let d = r.i32()?;
    let ntotal = r.i64()?;
    let _dummy = r.i64()?;
    let _dummy = r.i64()?;
    let _is_trained = r.u8()?;
    let metric_type = r.i32()?;
    if metric_type > 1 {
        let _metric_arg = r.f32()?;
    }
    if d <= 0 || d as usize > MAX_DIM {
        return Err(format!("FAISS index has invalid dimension {d}"));
    }
    if ntotal < 0 {
        return Err(format!("FAISS index has invalid ntotal {ntotal}"));
    }
    Ok(IndexHeader {
        dim: d as usize,
        ntotal: usize::try_from(ntotal).map_err(|_| "FAISS ntotal out of range".to_string())?,
        metric: Metric::from_faiss(metric_type)?,
    })
}

struct ParsedIndex {
    dim: usize,
    ntotal: usize,
    store: Store,
}

fn read_index(r: &mut Reader<'_>, depth: usize) -> Result<ParsedIndex, String> {
    let h = r.fourcc()?;
    match &h {
        b"IxF2" | b"IxFI" | b"IxFl" => {
            let hdr = read_header(r)?;
            // READXBVECTOR: size_t count of 4-byte groups (= floats), then the codes.
            let n_floats = r.count(4)?;
            let expected = hdr
                .ntotal
                .checked_mul(hdr.dim)
                .ok_or_else(|| "FAISS flat index size overflows".to_string())?;
            if n_floats != expected {
                return Err(format!("FAISS flat index stores {} floats, expected ntotal*d = {}", n_floats, expected));
            }
            let data = r.f32_vec(n_floats)?;
            let metric = match &h {
                b"IxF2" => Metric::L2,
                b"IxFI" => Metric::InnerProduct,
                _ => hdr.metric,
            };
            Ok(ParsedIndex {
                dim: hdr.dim,
                ntotal: hdr.ntotal,
                store: Store::Flat {
                    metric,
                    vectors: Vectors { dim: hdr.dim, data },
                },
            })
        }
        b"IwFl" => {
            if depth > 0 {
                return Err("nested FAISS IVF indexes are not supported".into());
            }
            // read_ivf_header: index header, nlist, nprobe, quantizer, direct map.
            let hdr = read_header(r)?;
            let nlist = r.count(0)?;
            let nprobe = r.count(0)?;
            let quantizer = read_index(r, depth + 1)?;
            let (quantizer_metric, centroids) = match quantizer.store {
                Store::Flat { metric, vectors } => (metric, vectors),
                Store::IvfFlat { .. } => {
                    return Err("FAISS IVF quantizer must be a flat index".into());
                }
            };
            if quantizer.dim != hdr.dim || centroids.len() != nlist {
                return Err(format!(
                    "FAISS IVF quantizer mismatch (d {} vs {}, {} centroids vs nlist {})",
                    quantizer.dim,
                    hdr.dim,
                    centroids.len(),
                    nlist
                ));
            }
            read_direct_map(r)?;
            let lists = read_array_inverted_lists(r, nlist, hdr.dim)?;
            let stored: usize = lists.iter().map(Vectors::len).sum();
            if stored != hdr.ntotal {
                return Err(format!("FAISS IVF lists hold {} vectors but ntotal is {}", stored, hdr.ntotal));
            }
            Ok(ParsedIndex {
                dim: hdr.dim,
                ntotal: hdr.ntotal,
                store: Store::IvfFlat {
                    metric: hdr.metric,
                    nprobe: nprobe.clamp(1, nlist.max(1)),
                    quantizer_metric,
                    centroids,
                    lists,
                },
            })
        }
        b"IvFl" | b"IvFL" => Err(format!(
            "legacy FAISS '{}' index; re-save it with a current faiss (write_index)",
            fourcc_str(&h)
        )),
        _ => Err(format!(
            "unsupported FAISS index type '{}' (expected IndexIVFFlat 'IwFl' or IndexFlat)",
            fourcc_str(&h)
        )),
    }
}

/// `read_direct_map`: type (char), `idx_t` array, plus `(idx_t, idx_t)` pairs for a hashtable.
fn read_direct_map(r: &mut Reader<'_>) -> Result<(), String> {
    let kind = r.u8()?;
    let n = r.count(8)?;
    r.skip(n * 8)?;
    if kind == 2 {
        let pairs = r.count(16)?;
        r.skip(pairs * 16)?;
    }
    Ok(())
}

/// `ilar` = ArrayInvertedLists: nlist, code_size, "full"/"sprs" list sizes, then per non-empty
/// list its codes (`n * code_size` bytes = raw f32 vectors for IVFFlat) and ids (`n` idx_t).
fn read_array_inverted_lists(r: &mut Reader<'_>, nlist: usize, dim: usize) -> Result<Vec<Vectors>, String> {
    let h = r.fourcc()?;
    if &h == b"il00" {
        return Err("FAISS index has no inverted lists (use RVC's added_*.index, not trained_*.index)".into());
    }
    if &h != b"ilar" {
        return Err(format!("unsupported FAISS inverted-list format '{}'", fourcc_str(&h)));
    }
    let il_nlist = r.count(0)?;
    let code_size = r.count(0)?;
    if il_nlist != nlist {
        return Err(format!("FAISS inverted lists: nlist {} != index nlist {}", il_nlist, nlist));
    }
    if code_size != dim * 4 {
        return Err(format!(
            "FAISS inverted lists: code_size {} is not {}*4 (IVFFlat stores raw f32)",
            code_size, dim
        ));
    }
    if nlist > r.remaining() {
        return Err(format!("FAISS nlist {nlist} is implausible for this file"));
    }
    let mut sizes = vec![0usize; nlist];
    let list_type = r.fourcc()?;
    match &list_type {
        b"full" => {
            let count = r.count(8)?;
            if count != nlist {
                return Err(format!("FAISS inverted lists: {} sizes for {} lists", count, nlist));
            }
            for s in sizes.iter_mut() {
                *s = r.count(0)?;
            }
        }
        b"sprs" => {
            let count = r.count(8)?;
            if count % 2 != 0 {
                return Err(format!("FAISS sparse list sizes: odd length {count}"));
            }
            for _ in 0..count / 2 {
                let list_no = r.count(0)?;
                let n = r.count(0)?;
                let slot = sizes
                    .get_mut(list_no)
                    .ok_or_else(|| format!("FAISS sparse list id {list_no} >= nlist {nlist}"))?;
                *slot = n;
            }
        }
        _ => return Err(format!("FAISS inverted lists: unknown size layout '{}'", fourcc_str(&list_type))),
    }
    let mut lists = Vec::with_capacity(nlist);
    for &n in &sizes {
        let data = if n > 0 {
            let floats = n
                .checked_mul(dim)
                .ok_or_else(|| "FAISS inverted list size overflows".to_string())?;
            let data = r.f32_vec(floats)?;
            r.skip(
                n.checked_mul(8)
                    .ok_or_else(|| "FAISS inverted list size overflows".to_string())?,
            )?;
            data
        } else {
            Vec::new()
        };
        lists.push(Vectors { dim, data });
    }
    Ok(lists)
}

#[cfg(test)]
mod tests {
    use super::*;

    // Writers mirroring faiss/impl/index_write.cpp.
    fn header(buf: &mut Vec<u8>, d: i32, ntotal: i64, metric: i32) {
        buf.extend_from_slice(&d.to_le_bytes());
        buf.extend_from_slice(&ntotal.to_le_bytes());
        buf.extend_from_slice(&(1i64 << 20).to_le_bytes());
        buf.extend_from_slice(&(1i64 << 20).to_le_bytes());
        buf.push(1); // is_trained
        buf.extend_from_slice(&metric.to_le_bytes());
    }

    fn flat(buf: &mut Vec<u8>, fourcc: &[u8; 4], d: usize, rows: &[Vec<f32>], metric: i32) {
        buf.extend_from_slice(fourcc);
        header(buf, d as i32, rows.len() as i64, metric);
        buf.extend_from_slice(&((rows.len() * d) as u64).to_le_bytes());
        for row in rows {
            for v in row {
                buf.extend_from_slice(&v.to_le_bytes());
            }
        }
    }

    fn u64s(buf: &mut Vec<u8>, vals: &[u64]) {
        buf.extend_from_slice(&(vals.len() as u64).to_le_bytes());
        for v in vals {
            buf.extend_from_slice(&v.to_le_bytes());
        }
    }

    /// `IwFl` with an `IxF2` quantizer. `lists[i]` = (ids, vectors) of list `i`.
    fn ivf_flat(
        d: usize,
        centroids: &[Vec<f32>],
        lists: &[(Vec<i64>, Vec<Vec<f32>>)],
        nprobe: u64,
        sparse: bool,
        hashtable_direct_map: bool,
    ) -> Vec<u8> {
        let ntotal: usize = lists.iter().map(|l| l.0.len()).sum();
        let mut buf = Vec::new();
        buf.extend_from_slice(b"IwFl");
        header(&mut buf, d as i32, ntotal as i64, 1);
        buf.extend_from_slice(&(centroids.len() as u64).to_le_bytes()); // nlist
        buf.extend_from_slice(&nprobe.to_le_bytes());
        flat(&mut buf, b"IxF2", d, centroids, 1);
        // direct map
        if hashtable_direct_map {
            buf.push(2);
            u64s(&mut buf, &[]);
            // WRITEVECTOR of pair<idx_t, idx_t>: size_t pair count, then the pairs.
            buf.extend_from_slice(&1u64.to_le_bytes());
            buf.extend_from_slice(&7i64.to_le_bytes());
            buf.extend_from_slice(&0i64.to_le_bytes());
        } else {
            buf.push(0);
            u64s(&mut buf, &[]);
        }
        // ArrayInvertedLists
        buf.extend_from_slice(b"ilar");
        buf.extend_from_slice(&(lists.len() as u64).to_le_bytes());
        buf.extend_from_slice(&((d * 4) as u64).to_le_bytes());
        if sparse {
            buf.extend_from_slice(b"sprs");
            let mut pairs = Vec::new();
            for (i, l) in lists.iter().enumerate() {
                if !l.0.is_empty() {
                    pairs.push(i as u64);
                    pairs.push(l.0.len() as u64);
                }
            }
            u64s(&mut buf, &pairs);
        } else {
            buf.extend_from_slice(b"full");
            let sizes: Vec<u64> = lists.iter().map(|l| l.0.len() as u64).collect();
            u64s(&mut buf, &sizes);
        }
        for (ids, vecs) in lists {
            for v in vecs {
                for x in v {
                    buf.extend_from_slice(&x.to_le_bytes());
                }
            }
            for id in ids {
                buf.extend_from_slice(&id.to_le_bytes());
            }
        }
        buf
    }

    fn v(x: f32) -> Vec<f32> {
        vec![x, x, x, x]
    }

    #[test]
    fn parses_ivf_flat_full_and_searches_nearest_list() {
        let bytes = ivf_flat(
            4,
            &[v(0.0), v(10.0)],
            &[(vec![0, 1], vec![v(0.1), v(-0.2)]), (vec![2, 3, 4], vec![v(9.0), v(10.5), v(12.0)])],
            1,
            false,
            false,
        );
        let index = FaissIndex::from_bytes(&bytes).unwrap();
        assert_eq!(index.dim(), 4);
        assert_eq!(index.ntotal, 5);
        let hits = index.search(&v(10.4), 8);
        // nprobe = 1: only the list around centroid 10 is scanned, closest first.
        assert_eq!(hits, vec![&v(10.5)[..], &v(9.0)[..], &v(12.0)[..]]);
        let hits = index.search(&v(0.05), 1);
        assert_eq!(hits, vec![&v(0.1)[..]]);
    }

    #[test]
    fn parses_sparse_lists_and_hashtable_direct_map() {
        let bytes = ivf_flat(
            4,
            &[v(0.0), v(5.0), v(100.0), v(-50.0)],
            &[
                (vec![], vec![]),
                (vec![0, 1], vec![v(4.0), v(6.5)]),
                (vec![], vec![]),
                (vec![2], vec![v(-49.0)]),
            ],
            2,
            true,
            true,
        );
        let index = FaissIndex::from_bytes(&bytes).unwrap();
        assert_eq!(index.ntotal, 3);
        // nprobe = 2 around -30: centroids -50 and 0 → lists 3 and 0 (empty).
        assert_eq!(index.search(&v(-30.0), 8), vec![&v(-49.0)[..]]);
        assert_eq!(index.search(&v(6.0), 8), vec![&v(6.5)[..], &v(4.0)[..]]);
    }

    #[test]
    fn parses_top_level_flat_l2_and_ip() {
        let rows = vec![v(1.0), v(2.0), v(3.0)];
        let mut bytes = Vec::new();
        flat(&mut bytes, b"IxF2", 4, &rows, 1);
        let index = FaissIndex::from_bytes(&bytes).unwrap();
        assert_eq!(index.search(&v(2.9), 2), vec![&v(3.0)[..], &v(2.0)[..]]);

        let mut bytes = Vec::new();
        flat(&mut bytes, b"IxFI", 4, &rows, 0);
        let index = FaissIndex::from_bytes(&bytes).unwrap();
        // Inner product: largest dot first.
        assert_eq!(index.search(&v(1.0), 1), vec![&v(3.0)[..]]);
    }

    #[test]
    fn rejects_empty_trained_and_truncated_indexes() {
        // Trained-only IVF: ntotal 0 and empty lists.
        let bytes = ivf_flat(4, &[v(0.0)], &[(vec![], vec![])], 1, false, false);
        let err = FaissIndex::from_bytes(&bytes).err().unwrap();
        assert!(err.contains("added_"), "{err}");
        // Truncation anywhere is an error, never a panic.
        let full = ivf_flat(
            4,
            &[v(0.0), v(10.0)],
            &[(vec![0], vec![v(0.5)]), (vec![1], vec![v(9.5)])],
            1,
            false,
            false,
        );
        for cut in 1..full.len() {
            assert!(FaissIndex::from_bytes(&full[..cut]).is_err(), "cut {cut}");
        }
        assert!(FaissIndex::from_bytes(b"IwPQ....").is_err());
    }

    #[test]
    fn raw_dump_fallback_still_loads() {
        let mut bytes = Vec::new();
        bytes.extend_from_slice(&2u32.to_le_bytes());
        bytes.extend_from_slice(&2u32.to_le_bytes());
        for x in [0.0f32, 0.0, 1.0, 1.0] {
            bytes.extend_from_slice(&x.to_le_bytes());
        }
        let index = FaissIndex::from_bytes(&bytes).unwrap();
        assert_eq!(index.dim(), 2);
        assert_eq!(index.search(&[0.9, 0.9], 1), vec![&[1.0f32, 1.0][..]]);
    }

    #[test]
    fn blend_uses_rvc_inverse_square_score_weights() {
        let rows = vec![vec![1.0f32, 0.0], vec![0.0, 2.0]];
        let mut bytes = Vec::new();
        bytes.extend_from_slice(b"IxF2");
        header(&mut bytes, 2, 2, 1);
        bytes.extend_from_slice(&4u64.to_le_bytes());
        for row in &rows {
            for x in row {
                bytes.extend_from_slice(&x.to_le_bytes());
            }
        }
        let index = FaissIndex::from_bytes(&bytes).unwrap();
        let mut feats = vec![0.0f32, 0.0];
        index.blend_inplace(&mut feats, 1, 1.0);
        // scores (squared L2) 1 and 4 → weights 1 and 1/16, normalized.
        let w0 = 1.0 / (1.0 + 1.0 / 16.0);
        let w1 = (1.0 / 16.0) / (1.0 + 1.0 / 16.0);
        assert!((feats[0] - w0).abs() < 1e-6, "{feats:?}");
        assert!((feats[1] - 2.0 * w1).abs() < 1e-6, "{feats:?}");

        // index_rate mixes: feats = nn * rate + feats * (1 - rate).
        let mut feats = vec![0.0f32, 0.0];
        index.blend_inplace(&mut feats, 1, 0.5);
        assert!((feats[0] - 0.5 * w0).abs() < 1e-6);

        // An exact match returns that row (no 1/0).
        let mut feats = vec![1.0f32, 0.0];
        index.blend_inplace(&mut feats, 1, 1.0);
        assert_eq!(feats, vec![1.0, 0.0]);

        // Rate 0 is a no-op.
        let mut feats = vec![0.3f32, 0.7];
        index.blend_inplace(&mut feats, 1, 0.0);
        assert_eq!(feats, vec![0.3, 0.7]);
    }
}
