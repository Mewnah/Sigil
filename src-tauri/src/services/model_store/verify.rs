//! Hashing: streaming sha256, and git blob ids (SHA-1 over `"blob <size>\0" + content`) for small
//! non-LFS files whose only published digest is the git object id.
//!
//! SHA-1 is implemented here (about 60 lines, checked against the FIPS 180 vectors and real git
//! blob ids in the tests) so the store needs no new crate. It is only used for git ids, which the
//! catalog takes from the Hub tree listing; sha256 is preferred wherever it exists.

use super::CatalogFile;
use sha2::{Digest as _, Sha256};
use std::fs::File;
use std::io::{self, Read};
use std::path::Path;

/// Read size of streaming hash / copy loops; cancellation is checked once per chunk.
pub(crate) const HASH_CHUNK: usize = 1 << 20;

/// What a file must hash to.
#[derive(Clone, Debug, PartialEq, Eq)]
pub(crate) enum Expected {
    Sha256(String),
    GitOid(String),
    /// Unpinned (trust on first use): the sha256 is computed and recorded.
    None,
}

impl Expected {
    pub fn of(file: &CatalogFile) -> Expected {
        match (&file.sha256, &file.git_oid) {
            (Some(h), _) => Expected::Sha256(h.to_ascii_lowercase()),
            (None, Some(o)) => Expected::GitOid(o.to_ascii_lowercase()),
            (None, None) => Expected::None,
        }
    }

    pub fn is_pinned(&self) -> bool {
        !matches!(self, Expected::None)
    }

    /// Whether `digest` satisfies this expectation (always true when unpinned).
    pub fn matches(&self, digest: &Digest) -> bool {
        match (self, digest) {
            (Expected::Sha256(want), Digest::Sha256(got)) => want == got,
            (Expected::GitOid(want), Digest::GitOid(got)) => want == got,
            (Expected::None, _) => true,
            _ => false,
        }
    }

    /// Short form for messages: `sha256 1a2b3c4d…`.
    pub fn describe(&self) -> String {
        match self {
            Expected::Sha256(h) => format!("sha256 {}…", &h[..h.len().min(12)]),
            Expected::GitOid(o) => format!("git {}…", &o[..o.len().min(12)]),
            Expected::None => "unpinned".into(),
        }
    }
}

/// A computed digest.
#[derive(Clone, Debug, PartialEq, Eq)]
pub(crate) enum Digest {
    Sha256(String),
    GitOid(String),
}

impl Digest {
    pub fn describe(&self) -> String {
        match self {
            Digest::Sha256(h) => format!("sha256 {}…", &h[..h.len().min(12)]),
            Digest::GitOid(o) => format!("git {}…", &o[..o.len().min(12)]),
        }
    }

    pub fn sha256(&self) -> Option<&str> {
        match self {
            Digest::Sha256(h) => Some(h),
            Digest::GitOid(_) => None,
        }
    }

    pub fn git_oid(&self) -> Option<&str> {
        match self {
            Digest::GitOid(o) => Some(o),
            Digest::Sha256(_) => None,
        }
    }
}

/// Streaming hasher for one file: sha256 unless the expectation is a git blob id.
pub(crate) enum FileHasher {
    Sha256(Box<Sha256>),
    Git(Box<Sha1>),
}

impl FileHasher {
    /// `size` is the full file size, needed up front for the git blob header.
    pub fn new(expected: &Expected, size: u64) -> Self {
        match expected {
            Expected::GitOid(_) => {
                let mut h = Sha1::new();
                h.update(format!("blob {size}\0").as_bytes());
                FileHasher::Git(Box::new(h))
            }
            _ => FileHasher::Sha256(Box::default()),
        }
    }

    pub fn update(&mut self, data: &[u8]) {
        match self {
            FileHasher::Sha256(h) => h.update(data),
            FileHasher::Git(h) => h.update(data),
        }
    }

    pub fn finish(self) -> Digest {
        match self {
            FileHasher::Sha256(h) => Digest::Sha256(hex::encode(h.finalize())),
            FileHasher::Git(h) => Digest::GitOid(hex::encode(h.finalize())),
        }
    }
}

/// Error of [`hash_file`].
#[derive(Debug)]
pub(crate) enum HashError {
    Cancelled,
    Io(io::Error),
}

/// Hash the first `len` bytes of `reader` (or all of it when `len` is `None`) into `hasher`,
/// polling `cancelled` once per chunk. Returns the number of bytes hashed.
pub(crate) fn hash_reader(
    reader: &mut dyn Read,
    len: Option<u64>,
    hasher: &mut FileHasher,
    cancelled: &dyn Fn() -> bool,
) -> Result<u64, HashError> {
    let mut buf = vec![0u8; HASH_CHUNK];
    let mut done: u64 = 0;
    loop {
        if cancelled() {
            return Err(HashError::Cancelled);
        }
        let want = match len {
            Some(l) if done >= l => break,
            Some(l) => ((l - done) as usize).min(buf.len()),
            None => buf.len(),
        };
        let n = match reader.read(&mut buf[..want]) {
            Ok(0) => break,
            Ok(n) => n,
            Err(e) if e.kind() == io::ErrorKind::Interrupted => continue,
            Err(e) => return Err(HashError::Io(e)),
        };
        hasher.update(&buf[..n]);
        done += n as u64;
    }
    Ok(done)
}

/// Hash a whole file against `expected` (the git header uses the file's real size).
pub(crate) fn hash_file(path: &Path, expected: &Expected, cancelled: &dyn Fn() -> bool) -> Result<Digest, HashError> {
    let mut f = File::open(path).map_err(HashError::Io)?;
    let size = f.metadata().map_err(HashError::Io)?.len();
    let mut hasher = FileHasher::new(expected, size);
    hash_reader(&mut f, None, &mut hasher, cancelled)?;
    Ok(hasher.finish())
}

// ---------------------------------------------------------------------------------------------
// SHA-1 (FIPS 180-4), only for git blob ids
// ---------------------------------------------------------------------------------------------

/// Minimal streaming SHA-1.
#[derive(Clone)]
pub(crate) struct Sha1 {
    state: [u32; 5],
    block: [u8; 64],
    block_len: usize,
    total_len: u64,
}

impl Sha1 {
    pub fn new() -> Self {
        Self {
            state: [0x6745_2301, 0xEFCD_AB89, 0x98BA_DCFE, 0x1032_5476, 0xC3D2_E1F0],
            block: [0; 64],
            block_len: 0,
            total_len: 0,
        }
    }

    pub fn update(&mut self, data: &[u8]) {
        self.total_len = self.total_len.wrapping_add(data.len() as u64);
        self.absorb(data);
    }

    fn absorb(&mut self, mut data: &[u8]) {
        if self.block_len > 0 {
            let take = (64 - self.block_len).min(data.len());
            self.block[self.block_len..self.block_len + take].copy_from_slice(&data[..take]);
            self.block_len += take;
            data = &data[take..];
            if self.block_len < 64 {
                return;
            }
            let block = self.block;
            compress(&mut self.state, &block);
            self.block_len = 0;
        }
        let mut chunks = data.chunks_exact(64);
        for block in &mut chunks {
            compress(&mut self.state, block.try_into().expect("64-byte chunk"));
        }
        let rest = chunks.remainder();
        self.block[..rest.len()].copy_from_slice(rest);
        self.block_len = rest.len();
    }

    pub fn finalize(mut self) -> [u8; 20] {
        let bit_len = self.total_len.wrapping_mul(8);
        let pad_len = if self.block_len < 56 { 56 - self.block_len } else { 120 - self.block_len };
        let mut pad = [0u8; 64];
        pad[0] = 0x80;
        self.absorb(&pad[..pad_len]);
        self.absorb(&bit_len.to_be_bytes());
        debug_assert_eq!(self.block_len, 0);
        let mut out = [0u8; 20];
        for (i, word) in self.state.iter().enumerate() {
            out[i * 4..i * 4 + 4].copy_from_slice(&word.to_be_bytes());
        }
        out
    }
}

fn compress(state: &mut [u32; 5], block: &[u8; 64]) {
    let mut w = [0u32; 80];
    for (i, word) in w.iter_mut().take(16).enumerate() {
        *word = u32::from_be_bytes([block[i * 4], block[i * 4 + 1], block[i * 4 + 2], block[i * 4 + 3]]);
    }
    for i in 16..80 {
        w[i] = (w[i - 3] ^ w[i - 8] ^ w[i - 14] ^ w[i - 16]).rotate_left(1);
    }
    let [mut a, mut b, mut c, mut d, mut e] = *state;
    for (i, wi) in w.iter().enumerate() {
        let (f, k) = match i {
            0..=19 => ((b & c) | (!b & d), 0x5A82_7999),
            20..=39 => (b ^ c ^ d, 0x6ED9_EBA1),
            40..=59 => ((b & c) | (b & d) | (c & d), 0x8F1B_BCDC),
            _ => (b ^ c ^ d, 0xCA62_C1D6),
        };
        let t = a
            .rotate_left(5)
            .wrapping_add(f)
            .wrapping_add(e)
            .wrapping_add(k)
            .wrapping_add(*wi);
        e = d;
        d = c;
        c = b.rotate_left(30);
        b = a;
        a = t;
    }
    state[0] = state[0].wrapping_add(a);
    state[1] = state[1].wrapping_add(b);
    state[2] = state[2].wrapping_add(c);
    state[3] = state[3].wrapping_add(d);
    state[4] = state[4].wrapping_add(e);
}

/// Git blob id of `data` (what `git hash-object` prints).
#[cfg(test)]
pub(crate) fn git_blob_id(data: &[u8]) -> String {
    let mut h = FileHasher::new(&Expected::GitOid(String::new()), data.len() as u64);
    h.update(data);
    match h.finish() {
        Digest::GitOid(o) => o,
        Digest::Sha256(_) => unreachable!(),
    }
}

/// sha256 hex of `data`.
#[cfg(test)]
pub(crate) fn sha256_hex(data: &[u8]) -> String {
    hex::encode(Sha256::digest(data))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn sha1_hex(data: &[u8]) -> String {
        let mut h = Sha1::new();
        h.update(data);
        hex::encode(h.finalize())
    }

    #[test]
    fn sha1_matches_fips_vectors_at_every_split() {
        assert_eq!(sha1_hex(b""), "da39a3ee5e6b4b0d3255bfef95601890afd80709");
        assert_eq!(sha1_hex(b"abc"), "a9993e364706816aba3e25717850c26c9cd0d89d");
        let two_blocks = b"abcdbcdecdefdefgefghfghighijhijkijkljklmklmnlmnomnopnopq";
        assert_eq!(sha1_hex(two_blocks), "84983e441c3bd26ebaae4aa1f95129e5e54670f1");
        // Streaming must not depend on how the input is split (block and padding boundaries).
        for split in 0..=two_blocks.len() {
            let mut h = Sha1::new();
            h.update(&two_blocks[..split]);
            h.update(&two_blocks[split..]);
            assert_eq!(hex::encode(h.finalize()), "84983e441c3bd26ebaae4aa1f95129e5e54670f1", "split {split}");
        }
        for len in [55usize, 56, 57, 63, 64, 65, 119, 120, 128] {
            let data = vec![b'x'; len];
            let mut h = Sha1::new();
            for b in &data {
                h.update(std::slice::from_ref(b));
            }
            assert_eq!(hex::encode(h.finalize()), sha1_hex(&data), "len {len}");
        }
        let million = vec![b'a'; 1_000_000];
        assert_eq!(sha1_hex(&million), "34aa973cd4c4daa4f61eeb2bdbad27316534016f");
    }

    #[test]
    fn git_blob_ids_match_git() {
        // `git hash-object` of an empty file and of "hello world\n".
        assert_eq!(git_blob_id(b""), "e69de29bb2d1d6434b8b29ae775ad8c2e48c5391");
        assert_eq!(git_blob_id(b"hello world\n"), "3b18e512dba79e4c8300dd08aeb37f8e728b8dad");
    }

    #[test]
    fn expectations_match_only_their_own_algorithm() {
        let data = b"model bytes";
        let sha = Expected::Sha256(sha256_hex(data));
        let git = Expected::GitOid(git_blob_id(data));
        let mut h = FileHasher::new(&sha, data.len() as u64);
        h.update(data);
        let d = h.finish();
        assert!(sha.matches(&d));
        assert!(!git.matches(&d));
        assert!(Expected::None.matches(&d));
        let mut h = FileHasher::new(&git, data.len() as u64);
        h.update(data);
        assert!(git.matches(&h.finish()));
        assert!(sha.describe().starts_with("sha256 "));
    }

    #[test]
    fn hash_reader_stops_at_len_and_honours_cancel() {
        let data = vec![7u8; 3 * HASH_CHUNK + 5];
        let mut h = FileHasher::new(&Expected::None, 0);
        let n = hash_reader(&mut &data[..], Some(HASH_CHUNK as u64 + 3), &mut h, &|| false).unwrap();
        assert_eq!(n, HASH_CHUNK as u64 + 3);
        let mut want = FileHasher::new(&Expected::None, 0);
        want.update(&data[..HASH_CHUNK + 3]);
        assert_eq!(h.finish(), want.finish());
        let mut h = FileHasher::new(&Expected::None, 0);
        assert!(matches!(hash_reader(&mut &data[..], None, &mut h, &|| true), Err(HashError::Cancelled)));
    }
}
