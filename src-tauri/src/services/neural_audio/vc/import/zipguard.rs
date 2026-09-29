//! Guarded zip reading for untrusted archives: community voice `.zip` files and the torch
//! checkpoint format (a `.pth` is itself a zip).
//!
//! Checked before anything is decompressed, from the archive's own directory:
//! - structure and entry count, pre-scanned before the zip crate parses anything: the end record
//!   the zip crate will use (zip64 included) must be unambiguous and end the file, and the
//!   central directory headers are walked and counted here, so a million-entry directory is
//!   refused before it is parsed and a repeated name is caught byte for byte;
//! - names: relative, no `..`, no drive or root, no backslash / NUL / control characters, unique;
//! - no symlinks, no encryption, only stored or deflated entries;
//! - declared sizes: per entry and in total, and the compression ratio of large entries.
//!
//! Reading an entry then enforces the *declared* size (a lying header cannot inflate past it) and
//! the CRC (checked by the zip crate at the end of the entry).

use std::collections::{HashMap, HashSet};
use std::io::{Read, Seek, SeekFrom};

use zip::{CompressionMethod, ZipArchive};

use super::{ErrorCode, ImportError};

/// Limits for one archive.
#[derive(Clone, Copy, Debug)]
pub struct ZipLimits {
    pub max_entries: u64,
    pub max_total_bytes: u64,
    pub max_entry_bytes: u64,
    /// Largest uncompressed / compressed ratio for entries of at least `ratio_min_bytes`.
    pub max_ratio: u64,
    pub ratio_min_bytes: u64,
}

impl ZipLimits {
    /// A community voice archive (`.pth` + `added_*.index` + extras).
    pub const VOICE_ARCHIVE: ZipLimits = ZipLimits {
        max_entries: 64,
        max_total_bytes: 2 << 30,
        max_entry_bytes: 2 << 30,
        max_ratio: 200,
        ratio_min_bytes: 1 << 20,
    };

    /// The torch checkpoint format: one entry per storage (RVC v2: ~460 entries).
    pub const TORCH_CHECKPOINT: ZipLimits = ZipLimits {
        max_entries: 8192,
        max_total_bytes: 2 << 30,
        max_entry_bytes: 2 << 30,
        max_ratio: 200,
        ratio_min_bytes: 1 << 20,
    };
}

/// One (file) entry that passed the checks.
#[derive(Clone, Debug)]
pub struct Entry {
    pub index: usize,
    pub name: String,
    pub size: u64,
}

/// A zip archive whose directory passed [`ZipLimits`].
pub struct GuardedZip<R: Read + Seek> {
    archive: ZipArchive<R>,
    entries: Vec<Entry>,
    by_name: HashMap<String, usize>,
}

fn rejected(msg: impl Into<String>) -> ImportError {
    ImportError::new(ErrorCode::ArchiveRejected, msg)
}

/// Whether `head` starts like a zip file (local header, or an empty archive's end record).
pub fn looks_like_zip(head: &[u8]) -> bool {
    head.starts_with(b"PK\x03\x04") || head.starts_with(b"PK\x05\x06")
}

/// A name that stays inside the extraction root on every platform.
fn safe_name(name: &str) -> bool {
    !name.is_empty()
        && name.len() <= 512
        && !name.starts_with('/')
        && !name.chars().any(|c| c == '\\' || c == ':' || c.is_control())
        && name.split('/').all(|part| part != ".." && part != ".")
        && !name.contains("//")
}

const EOCD_SIG: [u8; 4] = [0x50, 0x4b, 0x05, 0x06];
const ZIP64_LOCATOR_SIG: [u8; 4] = [0x50, 0x4b, 0x06, 0x07];
const ZIP64_EOCD_SIG: [u8; 4] = [0x50, 0x4b, 0x06, 0x06];
const CDFH_SIG: [u8; 4] = [0x50, 0x4b, 0x01, 0x02];
/// Fixed part of a central directory header.
const CDFH_LEN: usize = 46;
/// Most central-directory bytes accepted per allowed entry (header, name, extra fields, comment).
const CD_BYTES_PER_ENTRY: u64 = 4096;

/// What the pre-scan read from the archive's own end records and central directory.
#[derive(Debug)]
struct DirScan {
    /// Central directory headers, counted one by one (names unique byte for byte).
    count: u64,
    /// Where the central directory starts (the archive has no prepended data).
    cd_offset: u64,
}

fn read_at<R: Read + Seek>(r: &mut R, offset: u64, len: usize, what: &str) -> Result<Vec<u8>, ImportError> {
    r.seek(SeekFrom::Start(offset)).map_err(ImportError::io)?;
    let mut buf = vec![0u8; len];
    r.read_exact(&mut buf).map_err(|_| rejected(format!("truncated {what}")))?;
    Ok(buf)
}

fn le_u16(b: &[u8], o: usize) -> u16 {
    u16::from_le_bytes([b[o], b[o + 1]])
}

fn le_u32(b: &[u8], o: usize) -> u32 {
    u32::from_le_bytes([b[o], b[o + 1], b[o + 2], b[o + 3]])
}

fn le_u64(b: &[u8], o: usize) -> u64 {
    let mut a = [0u8; 8];
    a.copy_from_slice(&b[o..o + 8]);
    u64::from_le_bytes(a)
}

fn multi_disk() -> ImportError {
    rejected("multi-disk zip archives are not supported")
}

/// The zip64 end record announced by a zip64 locator right before the end record at `eocd_at`:
/// (its offset, its fixed 56 bytes), `None` when there is no locator.
fn zip64_end<R: Read + Seek>(r: &mut R, eocd_at: u64) -> Result<Option<(u64, Vec<u8>)>, ImportError> {
    let Some(locator_at) = eocd_at.checked_sub(20) else {
        return Ok(None);
    };
    let loc = read_at(r, locator_at, 20, "zip64 locator")?;
    if loc[0..4] != ZIP64_LOCATOR_SIG {
        return Ok(None);
    }
    if le_u32(&loc, 4) != 0 || le_u32(&loc, 16) > 1 {
        return Err(multi_disk());
    }
    let z64 = le_u64(&loc, 8);
    if z64 >= locator_at || locator_at - z64 < 56 {
        return Err(rejected("zip64 end record outside the archive"));
    }
    let rec = read_at(r, z64, 56, "zip64 end record")?;
    if rec[0..4] != ZIP64_EOCD_SIG {
        return Err(rejected("bad zip64 end record"));
    }
    let record_size = le_u64(&rec, 4);
    if record_size < 44 || record_size.checked_add(12) != Some(locator_at - z64) {
        return Err(rejected("the zip64 end record's size disagrees with its position"));
    }
    Ok(Some((z64, rec)))
}

/// Pre-scan of the directory before the zip crate parses it. The end record is picked the way the
/// zip crate picks it (the last signature from the end; zip64 when the zip32 entry total or
/// directory offset is saturated), anything that could make the two readers disagree is refused,
/// and the central directory headers are walked here, so the entry count and duplicate names never
/// depend on the declared counts:
/// - the end record's comment must end exactly at the end of the file (no forged record in a
///   comment, no appended data);
/// - single disk only, and "entries on this disk" must equal "total entries" (the zip crate reads
///   the first for zip32 and the second for zip64);
/// - the directory must sit right before the (zip64) end records, at its declared offset (no
///   prepended data), and hold exactly the declared number of headers.
fn scan_directory<R: Read + Seek>(r: &mut R, max_entries: u64) -> Result<DirScan, ImportError> {
    let len = r.seek(SeekFrom::End(0)).map_err(ImportError::io)?;
    if len < 22 {
        return Err(rejected("not a zip archive (too short)"));
    }
    let tail_len = len.min(22 + 65_535);
    let tail_start = len - tail_len;
    let tail = read_at(r, tail_start, tail_len as usize, "zip end record")?;
    let p = (0..=tail.len() - 22)
        .rev()
        .find(|&p| tail[p..p + 4] == EOCD_SIG)
        .ok_or_else(|| rejected("not a zip archive (no end-of-central-directory record)"))?;
    if p + 22 + le_u16(&tail, p + 20) as usize != tail.len() {
        return Err(rejected(
            "the end-of-central-directory record does not end the file (appended data, or a forged record in the comment)",
        ));
    }
    let eocd_at = tail_start + p as u64;
    let (disk, cd_disk) = (le_u16(&tail, p + 4), le_u16(&tail, p + 6));
    let (on_disk, total) = (le_u16(&tail, p + 8), le_u16(&tail, p + 10));
    let (cd_size32, cd_offset32) = (le_u32(&tail, p + 12), le_u32(&tail, p + 16));
    let counts_differ = || rejected("the end record's entry counts disagree (entries on this disk vs total)");
    // The zip crate's `may_be_zip64` test.
    let zip64 = total == 0xFFFF || cd_offset32 == 0xFFFF_FFFF;
    let (count, cd_size, cd_offset, cd_end) = if !zip64 {
        if disk != 0 || cd_disk != 0 {
            return Err(multi_disk());
        }
        if on_disk != total {
            return Err(counts_differ());
        }
        // Writers such as PyTorch's also put zip64 records between the directory and the end
        // record when they are not needed; the directory then ends where they begin.
        let cd_end = zip64_end(r, eocd_at)?.map_or(eocd_at, |(z64, _)| z64);
        (total as u64, cd_size32 as u64, cd_offset32 as u64, cd_end)
    } else {
        let (z64, rec) = zip64_end(r, eocd_at)?.ok_or_else(|| rejected("zip64 archive without a zip64 locator"))?;
        if le_u16(&rec, 14) > le_u16(&rec, 12) {
            return Err(rejected("the zip64 end record needs a newer version than the one that wrote it"));
        }
        if le_u32(&rec, 16) != 0 || le_u32(&rec, 20) != 0 {
            return Err(multi_disk());
        }
        let (on_disk, total) = (le_u64(&rec, 24), le_u64(&rec, 32));
        if on_disk != total {
            return Err(counts_differ());
        }
        (total, le_u64(&rec, 40), le_u64(&rec, 48), z64)
    };
    if count > max_entries {
        return Err(rejected(format!("{count} entries (at most {max_entries} allowed)")));
    }
    if cd_offset.checked_add(cd_size) != Some(cd_end) {
        return Err(rejected(
            "the central directory is not where the end record says (prepended data, or a gap before the end record)",
        ));
    }
    if cd_size > max_entries.saturating_mul(CD_BYTES_PER_ENTRY) || cd_size < count * CDFH_LEN as u64 {
        return Err(rejected(format!("a central directory of {cd_size} bytes for {count} entries")));
    }
    let cd = read_at(r, cd_offset, cd_size as usize, "central directory")?;
    let malformed = || rejected("malformed central directory");
    let mut names: HashSet<&[u8]> = HashSet::new();
    let mut seen = 0u64;
    let mut pos = 0usize;
    while pos < cd.len() {
        if cd.len() - pos < CDFH_LEN || cd[pos..pos + 4] != CDFH_SIG {
            return Err(malformed());
        }
        let name_len = le_u16(&cd, pos + 28) as usize;
        let end = pos + CDFH_LEN + name_len + le_u16(&cd, pos + 30) as usize + le_u16(&cd, pos + 32) as usize;
        if end > cd.len() {
            return Err(malformed());
        }
        seen += 1;
        if seen > count {
            break;
        }
        let name = &cd[pos + CDFH_LEN..pos + CDFH_LEN + name_len];
        if !names.insert(name) {
            return Err(rejected(format!("duplicate entry {}", String::from_utf8_lossy(name))));
        }
        let local = le_u32(&cd, pos + 42);
        if local != u32::MAX && local as u64 + 30 > cd_offset {
            return Err(rejected(format!("{}: local header outside the archive data", String::from_utf8_lossy(name))));
        }
        pos = end;
    }
    if seen != count {
        return Err(rejected(format!("the directory holds {seen} entries but the end record declares {count}")));
    }
    Ok(DirScan { count, cd_offset })
}

impl<R: Read + Seek> GuardedZip<R> {
    pub fn open(mut reader: R, limits: ZipLimits) -> Result<Self, ImportError> {
        let scan = scan_directory(&mut reader, limits.max_entries)?;
        reader.seek(SeekFrom::Start(0)).map_err(ImportError::io)?;
        let mut archive = ZipArchive::new(reader).map_err(|e| rejected(format!("unreadable zip: {e}")))?;
        // The zip crate must have read the directory that was checked (it falls back to earlier
        // end records when one fails its own checks).
        if archive.offset() != 0 || archive.central_directory_start() != scan.cd_offset {
            return Err(rejected("the archive has more than one central directory"));
        }
        // The zip crate keys entries by (decoded) name, so a repeated name silently hides one of
        // them; names that differ in bytes but decode alike (code page vs UTF-8) are caught here.
        if archive.len() as u64 != scan.count {
            return Err(rejected(format!(
                "the directory lists {} entries but only {} distinct names (duplicate entries)",
                scan.count,
                archive.len()
            )));
        }
        let mut entries = Vec::new();
        let mut by_name = HashMap::new();
        let mut total: u64 = 0;
        for index in 0..archive.len() {
            let f = archive
                .by_index_raw(index)
                .map_err(|e| rejected(format!("entry {index}: {e}")))?;
            let name = f.name().to_string();
            if !safe_name(&name) || f.enclosed_name().is_none() {
                return Err(rejected(format!("unsafe entry name {name:?}")));
            }
            if f.is_symlink() {
                return Err(rejected(format!("{name}: symbolic links are not allowed")));
            }
            if f.encrypted() {
                return Err(rejected(format!("{name}: encrypted entries are not supported")));
            }
            if by_name.contains_key(&name) {
                return Err(rejected(format!("duplicate entry {name}")));
            }
            if f.is_dir() {
                by_name.insert(name, usize::MAX);
                continue;
            }
            match f.compression() {
                CompressionMethod::Stored | CompressionMethod::Deflated => {}
                other => return Err(rejected(format!("{name}: compression {other:?} is not supported"))),
            }
            let (size, compressed) = (f.size(), f.compressed_size());
            if size > limits.max_entry_bytes {
                return Err(rejected(format!("{name}: {size} bytes uncompressed")));
            }
            if size >= limits.ratio_min_bytes && size / compressed.max(1) > limits.max_ratio {
                return Err(rejected(format!("{name}: compression ratio above {}x (zip bomb?)", limits.max_ratio)));
            }
            total = total.saturating_add(size);
            if total > limits.max_total_bytes {
                return Err(rejected(format!("more than {} bytes uncompressed in total", limits.max_total_bytes)));
            }
            by_name.insert(name.clone(), entries.len());
            entries.push(Entry { index, name, size });
        }
        by_name.retain(|_, v| *v != usize::MAX);
        Ok(Self { archive, entries, by_name })
    }

    pub fn entries(&self) -> &[Entry] {
        &self.entries
    }

    pub fn find(&self, name: &str) -> Option<&Entry> {
        self.by_name.get(name).map(|&i| &self.entries[i])
    }

    /// The entry's bytes; the decompressed length must equal the declared size and the CRC must
    /// match. `max` caps the size the caller is prepared to hold.
    pub fn read(&mut self, name: &str, max: u64) -> Result<Vec<u8>, ImportError> {
        let entry = self.find(name).cloned().ok_or_else(|| {
            ImportError::new(ErrorCode::NotACheckpoint, format!("{name} is missing from the archive"))
        })?;
        if entry.size > max {
            return Err(rejected(format!("{name}: {} bytes (at most {max} expected)", entry.size)));
        }
        let mut f = self
            .archive
            .by_index(entry.index)
            .map_err(|e| rejected(format!("{name}: {e}")))?;
        let mut out = Vec::with_capacity(entry.size as usize);
        (&mut f)
            .take(entry.size + 1)
            .read_to_end(&mut out)
            .map_err(|e| rejected(format!("{name}: {e}")))?;
        if out.len() as u64 != entry.size {
            return Err(rejected(format!(
                "{name}: {} bytes decompressed but {} declared",
                out.len(),
                entry.size
            )));
        }
        // Reach the end of the entry so the zip crate verifies the CRC.
        let mut probe = [0u8; 1];
        if f.read(&mut probe).map_err(|e| rejected(format!("{name}: {e}")))? != 0 {
            return Err(rejected(format!("{name}: more data than declared")));
        }
        Ok(out)
    }
}

#[cfg(test)]
pub(crate) mod tests {
    use super::*;
    use std::io::{Cursor, Write};
    use zip::write::SimpleFileOptions;

    pub(crate) fn build_zip(files: &[(&str, &[u8], CompressionMethod)]) -> Vec<u8> {
        let mut w = zip::ZipWriter::new(Cursor::new(Vec::new()));
        for (name, data, method) in files {
            w.start_file(*name, SimpleFileOptions::default().compression_method(*method)).unwrap();
            w.write_all(data).unwrap();
        }
        w.finish().unwrap().into_inner()
    }

    fn open(bytes: Vec<u8>, limits: ZipLimits) -> Result<GuardedZip<Cursor<Vec<u8>>>, ImportError> {
        GuardedZip::open(Cursor::new(bytes), limits)
    }

    #[test]
    fn reads_good_entries() {
        let z = build_zip(&[
            ("voice/model.pth", b"hello", CompressionMethod::Stored),
            ("voice/added_IVF1_Flat.index", &[7u8; 5000], CompressionMethod::Deflated),
        ]);
        let mut g = open(z, ZipLimits::VOICE_ARCHIVE).unwrap();
        assert_eq!(g.entries().len(), 2);
        assert_eq!(g.read("voice/model.pth", 100).unwrap(), b"hello");
        assert_eq!(g.read("voice/added_IVF1_Flat.index", 10_000).unwrap(), vec![7u8; 5000]);
        assert_eq!(g.read("voice/model.pth", 2).unwrap_err().code, ErrorCode::ArchiveRejected);
    }

    #[test]
    fn rejects_traversal_and_odd_names() {
        for name in ["../evil.pth", "a/../../evil", "/abs.pth", "C:/x.pth", "a\\..\\b", "./x", "a//b", "ok\u{1}.pth"] {
            let z = build_zip(&[(name, b"x", CompressionMethod::Stored)]);
            let err = open(z, ZipLimits::VOICE_ARCHIVE).err().unwrap_or_else(|| panic!("{name:?} accepted"));
            assert_eq!(err.code, ErrorCode::ArchiveRejected, "{name:?}");
        }
    }

    #[test]
    fn rejects_too_many_entries_before_parsing() {
        let names: Vec<String> = (0..65).map(|i| format!("f{i}")).collect();
        let files: Vec<(&str, &[u8], CompressionMethod)> =
            names.iter().map(|n| (n.as_str(), &b""[..], CompressionMethod::Stored)).collect();
        let err = open(build_zip(&files), ZipLimits::VOICE_ARCHIVE).err().unwrap();
        assert!(err.message.contains("65 entries"), "{}", err.message);
    }

    #[test]
    fn rejects_zip_bombs_and_oversize() {
        let zeros = vec![0u8; 4 << 20];
        let z = build_zip(&[("bomb.bin", &zeros, CompressionMethod::Deflated)]);
        let err = open(z, ZipLimits::VOICE_ARCHIVE).err().unwrap();
        assert!(err.message.contains("ratio"), "{}", err.message);
        let small = ZipLimits { max_total_bytes: 10, ..ZipLimits::VOICE_ARCHIVE };
        let z = build_zip(&[("a", b"123456", CompressionMethod::Stored), ("b", b"123456", CompressionMethod::Stored)]);
        assert!(open(z, small).is_err());
    }

    #[test]
    fn rejects_symlinks_and_duplicates() {
        let mut w = zip::ZipWriter::new(Cursor::new(Vec::new()));
        w.add_symlink("link.pth", "../../etc/passwd", SimpleFileOptions::default()).unwrap();
        let z = w.finish().unwrap().into_inner();
        assert!(open(z, ZipLimits::VOICE_ARCHIVE).err().unwrap().message.contains("symbolic"));

        // Duplicate names: patch the second name of a two-entry archive in place.
        let mut z = build_zip(&[("aa", b"1", CompressionMethod::Stored), ("ab", b"2", CompressionMethod::Stored)]);
        for i in 0..z.len() - 1 {
            if &z[i..i + 2] == b"ab" {
                z[i + 1] = b'a';
            }
        }
        assert!(open(z, ZipLimits::VOICE_ARCHIVE).is_err());
    }

    #[test]
    fn rejects_non_zip_and_lying_sizes() {
        assert!(open(b"definitely not a zip file at all".to_vec(), ZipLimits::VOICE_ARCHIVE).is_err());
        assert!(open(vec![], ZipLimits::VOICE_ARCHIVE).is_err());
        // Corrupt the stored payload: the CRC check fails on read.
        let mut z = build_zip(&[("x.bin", b"abcdefgh", CompressionMethod::Stored)]);
        let at = z.windows(8).position(|w| w == b"abcdefgh").unwrap();
        z[at] = b'Z';
        let mut g = open(z, ZipLimits::VOICE_ARCHIVE).unwrap();
        assert!(g.read("x.bin", 100).is_err());
    }

    /// Byte offset of the (last) end-of-central-directory record.
    fn eocd(z: &[u8]) -> usize {
        z.windows(4).rposition(|w| w == EOCD_SIG).unwrap()
    }

    fn two_entries() -> Vec<u8> {
        build_zip(&[("a", b"1", CompressionMethod::Stored), ("b", b"2", CompressionMethod::Stored)])
    }

    #[test]
    fn end_records_must_be_unambiguous() {
        assert_eq!(open(two_entries(), ZipLimits::VOICE_ARCHIVE).unwrap().entries().len(), 2);

        // A forged end record (one entry) at the end of the real record's comment, with a comment
        // length that does not end the file: the zip crate skips it, a naive pre-scan would not.
        let mut z = two_entries();
        let e = eocd(&z);
        let mut fake = z[e..e + 22].to_vec();
        fake[8..12].copy_from_slice(&[1, 0, 1, 0]);
        fake[20..22].copy_from_slice(&100u16.to_le_bytes());
        z[e + 20..e + 22].copy_from_slice(&22u16.to_le_bytes());
        z.extend_from_slice(&fake);
        let err = open(z, ZipLimits::VOICE_ARCHIVE).err().unwrap();
        assert!(err.message.contains("end-of-central-directory"), "{}", err.message);

        // Data appended after the end record, or prepended before the archive.
        let mut z = two_entries();
        z.extend_from_slice(b"trailing");
        assert!(open(z, ZipLimits::VOICE_ARCHIVE).is_err());
        let mut z = b"MZ-prefix".to_vec();
        z.extend_from_slice(&two_entries());
        let err = open(z, ZipLimits::VOICE_ARCHIVE).err().unwrap();
        assert!(err.message.contains("central directory"), "{}", err.message);
    }

    #[test]
    fn entry_counts_come_from_the_directory_itself() {
        // Two entries named "aa"; the end record says 2 on this disk (what the zip crate reads for
        // zip32) but 1 in total (what a naive pre-scan read): the duplicate would go unnoticed.
        let dup = || {
            let mut z = build_zip(&[("aa", b"1", CompressionMethod::Stored), ("ab", b"2", CompressionMethod::Stored)]);
            for i in 0..z.len() - 1 {
                if &z[i..i + 2] == b"ab" {
                    z[i + 1] = b'a';
                }
            }
            z
        };
        let mut z = dup();
        let e = eocd(&z);
        z[e + 10..e + 12].copy_from_slice(&1u16.to_le_bytes());
        let err = open(z, ZipLimits::VOICE_ARCHIVE).err().unwrap();
        assert!(err.message.contains("disagree"), "{}", err.message);
        // Consistent counts: walking the directory finds the repeated name.
        let err = open(dup(), ZipLimits::VOICE_ARCHIVE).err().unwrap();
        assert!(err.message.contains("duplicate entry aa"), "{}", err.message);
        // Declared counts that do not match the headers in the directory.
        let mut z = two_entries();
        let e = eocd(&z);
        z[e + 8..e + 12].copy_from_slice(&[1, 0, 1, 0]);
        let err = open(z, ZipLimits::VOICE_ARCHIVE).err().unwrap();
        assert!(err.message.contains("declares 1"), "{}", err.message);
    }

    #[test]
    fn torch_zip64_records_are_read_like_the_zip_crate() {
        // PyTorch writes zip64 end records even when the zip32 record suffices (the tiny fixture).
        let pth = std::fs::read(std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("test-fixtures/vc/tiny/tiny.pth")).unwrap();
        assert!(pth.windows(4).any(|w| w == ZIP64_LOCATOR_SIG));
        let n = open(pth.clone(), ZipLimits::TORCH_CHECKPOINT).unwrap().entries().len();
        assert!(n > 100, "{n} entries");
        // Saturated zip32 fields: the zip64 record is read instead, with the same result.
        let mut z = pth.clone();
        let e = eocd(&z);
        z[e + 8..e + 12].copy_from_slice(&[0xFF; 4]);
        z[e + 16..e + 20].copy_from_slice(&[0xFF; 4]);
        assert_eq!(open(z.clone(), ZipLimits::TORCH_CHECKPOINT).unwrap().entries().len(), n);
        // ...where "entries on this disk" and "total" must agree too.
        let z64 = z.windows(4).rposition(|w| w == ZIP64_EOCD_SIG).unwrap();
        let on_disk = u64::from_le_bytes(z[z64 + 24..z64 + 32].try_into().unwrap());
        z[z64 + 24..z64 + 32].copy_from_slice(&(on_disk + 1).to_le_bytes());
        let err = open(z, ZipLimits::TORCH_CHECKPOINT).err().unwrap();
        assert!(err.message.contains("disagree"), "{}", err.message);
    }

    /// Random mutations of small archives (zip32, and PyTorch's with zip64 records) never panic.
    #[test]
    fn mutated_archives_never_panic() {
        let tiny = std::fs::read(std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("test-fixtures/vc/tiny/tiny.pth")).unwrap();
        let bases = [two_entries(), build_zip(&[("v/x.pth", &[5u8; 300][..], CompressionMethod::Deflated)])];
        let mut state = 0x2545f4914f6cdd1du64;
        let mut next = || {
            state ^= state << 13;
            state ^= state >> 7;
            state ^= state << 17;
            state
        };
        for round in 0..3000 {
            let (mut d, limits) = match round % 3 {
                0 => (bases[0].clone(), ZipLimits::VOICE_ARCHIVE),
                1 => (bases[1].clone(), ZipLimits::VOICE_ARCHIVE),
                _ => (tiny.clone(), ZipLimits::TORCH_CHECKPOINT),
            };
            // Mutate the last 16 KB: the directory and end records, where the pre-scan reads.
            let tail = d.len().min(16_000);
            for _ in 0..(1 + next() % 4) {
                let i = d.len() - 1 - (next() as usize) % tail;
                d[i] = next() as u8;
            }
            if let Ok(mut g) = open(d, limits) {
                let names: Vec<String> = g.entries().iter().map(|e| e.name.clone()).collect();
                for n in names.iter().take(4) {
                    let _ = g.read(n, 1 << 20);
                }
            }
        }
    }
}
