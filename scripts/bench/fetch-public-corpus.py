#!/usr/bin/env python3
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Build the public T6 corpus for Sigil's compare mode from freely licensed real speech.

The user has no recordings of their own for the head-to-head (plan T6), so the corpus is made from
public corpora, as a fixed, documented, seeded subset:

- **LibriSpeech test-clean** (OpenSLR 12, CC BY 4.0; read English, 40 speakers, 20 female and 20
  male, 16 kHz FLAC): per speaker, ``--libri-per-speaker`` clips (default 2), each the consecutive
  utterances of one chapter joined with 300 ms gaps until at least ``--min-seconds`` (10 s; DNSMOS
  needs 9.01 s), at most ``--max-seconds`` (20 s). Chapter and starting utterance are drawn with
  ``--seed`` from the sorted lists, so the same archive always gives the same clips.
- **CMU ARCTIC bdl, jmk, slt** (festvox, BSD-style license; US English; the ``orig/`` recordings
  carry a simultaneous EGG channel): ``--arctic-per-speaker`` clips per speaker (default 4) of
  consecutive utterances joined the same way, each with tracker-independent F0 truth from its EGG
  (``scripts/vc-bench/egg_f0.py``: glottal closure instants), written as ``<clip>.f0.csv``.

Every clip is mono, loudness-normalised to ``--lufs`` (-26 LUFS, ITU-R BS.1770-4; a voice changer's
silence gate sees a sensible mic level) and written as 16-bit PCM WAV. ``corpus.json``
(``sigil-compare-corpus/1``) lists every clip with its sha256, speaker, sex, source utterances and
license; the drivers then fill in the competitors' points (``scripts/vc-bench/README.md``).

NEVER COMMIT THE AUDIO. The default output folder is outside the repository
(``%LOCALAPPDATA%/sigil-dev-cache/corpus/public-v1``); writing inside it needs ``--force``.

Downloads (a manual step; about 350 MB for LibriSpeech test-clean and about 100 MB per ARCTIC
speaker) run ONE AT A TIME, resume an interrupted file (HTTP Range into ``<file>.partial``) and try
the mirrors in turn; ``--archives DIR`` uses archives you already have instead. The archives are
not verified against hashes this script does not know: compare OpenSLR's ``md5sum.txt`` if you
want to.

    python scripts/bench/fetch-public-corpus.py --dry-run
    python scripts/bench/fetch-public-corpus.py [--out DIR] [--archives DIR] [--voice default.pth]
    python scripts/bench/fetch-public-corpus.py --self-test        # synthetic archives, no network

Needs numpy and soundfile (FLAC support).
"""

from __future__ import annotations

import argparse
import io
import json
import os
import random
import shutil
import sys
import tarfile
import time
import urllib.error
import urllib.request
import zlib

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(os.path.dirname(HERE))
sys.path.insert(0, os.path.join(REPO, "scripts", "vc-bench"))
import vcbench_common as vb  # noqa: E402

SOURCES = {
    "librispeech": {
        "file": "test-clean.tar.gz",
        "urls": [
            "https://www.openslr.org/resources/12/test-clean.tar.gz",
            "https://us.openslr.org/resources/12/test-clean.tar.gz",
            "https://openslr.elda.org/resources/12/test-clean.tar.gz",
        ],
        "approx_mb": 346,
        "license": "CC BY 4.0 (LibriSpeech ASR corpus, Panayotov et al., OpenSLR 12)",
    },
}
for spk in ("bdl", "jmk", "slt"):
    SOURCES[f"arctic_{spk}"] = {
        "file": f"cmu_us_{spk}_arctic.tar.bz2",
        "urls": [f"http://festvox.org/cmu_arctic/packed/cmu_us_{spk}_arctic.tar.bz2"],
        "approx_mb": 100,
        "license": "CMU ARCTIC, BSD-style (free use with attribution; Kominek and Black, CMU LTI)",
    }
ARCTIC_SEX = {"bdl": "M", "jmk": "M", "slt": "F"}
DEFAULT_OUT = os.path.join(os.environ.get("LOCALAPPDATA", os.path.expanduser("~")), "sigil-dev-cache", "corpus", "public-v1")


def parse_args(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", default=DEFAULT_OUT)
    ap.add_argument("--archives", default="", help="folder with the archives (downloaded there when missing)")
    ap.add_argument("--seed", type=int, default=20261003)
    ap.add_argument("--libri-per-speaker", type=int, default=2)
    ap.add_argument("--arctic-per-speaker", type=int, default=4)
    ap.add_argument("--min-seconds", type=float, default=10.0)
    ap.add_argument("--max-seconds", type=float, default=20.0)
    ap.add_argument("--gap-ms", type=float, default=300.0)
    ap.add_argument("--lufs", type=float, default=-26.0)
    ap.add_argument("--voice", default="", help="the voice every system will use (its sha256 goes into the manifest)")
    ap.add_argument("--skip", default="", help="sources to leave out, e.g. arctic_jmk,librispeech")
    ap.add_argument("--dry-run", action="store_true", help="print the plan; no network, no files")
    ap.add_argument("--force", action="store_true", help="allow an output folder inside the repository")
    ap.add_argument("--self-test", action="store_true")
    return ap.parse_args(argv)


def inside_repo(path: str) -> bool:
    p = os.path.abspath(path)
    try:
        return os.path.commonpath([p, REPO]) == REPO
    except ValueError:  # another drive (Windows): it cannot be inside the repository
        return False


class ShortDownload(IOError):
    """The server closed the connection before the announced length arrived."""


def download(url: str, dest: str, approx_mb: int) -> None:
    """One file, resumable (Range into ``dest.partial``); raises on failure. The file is moved to
    ``dest`` only when every announced byte arrived: a connection the router dropped mid-file
    leaves the ``.partial`` for the next run to resume (``http.client`` returns a short body
    without an error, so the byte count is checked here)."""
    part = dest + ".partial"
    have = os.path.getsize(part) if os.path.isfile(part) else 0
    req = urllib.request.Request(url, headers={"User-Agent": "sigil-fetch-public-corpus/1"})
    if have:
        req.add_header("Range", f"bytes={have}-")
    try:
        r = urllib.request.urlopen(req, timeout=30)
    except urllib.error.HTTPError as e:
        if e.code == 416 and have:  # the partial already holds the whole file
            os.replace(part, dest)
            return
        raise
    with r:
        if have and r.status != 206:
            have = 0  # the server ignored the range: start over
        mode = "ab" if have else "wb"
        length = r.headers.get("Content-Length")
        expected = have + int(length) if length and length.isdigit() else None
        with open(part, mode) as f:
            done = have
            last = time.time()
            while True:
                chunk = r.read(1 << 20)
                if not chunk:
                    break
                f.write(chunk)
                done += len(chunk)
                if time.time() - last > 2:
                    last = time.time()
                    print(f"  {done / 1e6:.0f} / {(expected or approx_mb * 1e6) / 1e6:.0f} MB", flush=True)
    if expected is not None and done != expected:
        raise ShortDownload(f"got {done} of {expected} bytes; the partial file is kept, run again to resume")
    os.replace(part, dest)


def archive_problem(path: str) -> str:
    """Why ``path`` is not a complete tar archive (read to its end), or ``""`` when it is."""
    try:
        with tarfile.open(path) as tar:
            for _ in tar:  # walking the members reads the whole compressed stream
                pass
    except (tarfile.TarError, EOFError, OSError, zlib.error) as e:
        return f"{type(e).__name__}: {e}"
    return ""


def fetch(name: str, folder: str) -> str:
    src = SOURCES[name]
    dest = os.path.join(folder, src["file"])
    if os.path.isfile(dest):
        why = archive_problem(dest)
        if why:
            raise SystemExit(f"{dest} is truncated or corrupt ({why}): move it away and run again to download it anew")
        return dest
    os.makedirs(folder, exist_ok=True)
    errors = []
    for url in src["urls"]:
        print(f"downloading {url} (about {src['approx_mb']} MB) -> {dest}")
        try:
            download(url, dest, src["approx_mb"])
            why = archive_problem(dest)
            if why:  # a bad file is never kept as the archive
                os.replace(dest, dest + ".bad")
                raise ShortDownload(f"the archive does not read to its end ({why}); kept as {os.path.basename(dest)}.bad")
            return dest
        except Exception as e:  # noqa: BLE001 (try the next mirror)
            errors.append(f"{url}: {e}")
            print(f"  failed: {e}")
    raise SystemExit(f"{name}: every mirror failed:\n" + "\n".join(errors))


def safe_members(tar: tarfile.TarFile, want):
    for m in tar.getmembers():
        n = m.name.replace("\\", "/")
        if n.startswith("/") or ".." in n.split("/") or not (m.isfile() or m.isdir()):
            continue
        if want(n):
            yield m


def extract(archive: str, dest: str, want, name: str = "") -> None:
    """Extract the wanted members of ``archive`` into ``dest``, all or nothing: they go to a
    temporary folder first and move into place only after the whole archive was read, then a
    ``.complete-<name>`` marker is written. A run that stopped half way leaves no folder that a
    later run could mistake for a finished extraction."""
    name = name or os.path.basename(archive)
    tmp = os.path.join(dest, f".extracting-{name}")
    shutil.rmtree(tmp, ignore_errors=True)
    os.makedirs(tmp, exist_ok=True)
    try:
        with tarfile.open(archive) as tar:
            members = list(safe_members(tar, want))
            try:
                tar.extractall(tmp, members=members, filter="data")
            except TypeError:  # Python < 3.12
                tar.extractall(tmp, members=members)
    except BaseException:
        shutil.rmtree(tmp, ignore_errors=True)  # nothing half done is left behind
        raise
    for entry in os.listdir(tmp):
        target = os.path.join(dest, entry)
        shutil.rmtree(target, ignore_errors=True)  # an earlier, unmarked (possibly partial) extraction
        os.replace(os.path.join(tmp, entry), target)
    shutil.rmtree(tmp, ignore_errors=True)
    with open(os.path.join(dest, f".complete-{name}"), "w", encoding="utf-8") as f:
        f.write(os.path.basename(archive) + "\n")


def extracted(dest: str, name: str) -> bool:
    """Whether source ``name`` was extracted completely (the marker ``extract`` writes last)."""
    return os.path.isfile(os.path.join(dest, f".complete-{name}"))


def join_utterances(paths, gap_ms: float, min_s: float, max_s: float):
    """Consecutive utterances joined with gaps until ``min_s`` (never beyond ``max_s`` unless one
    utterance is longer): ``(mono, rate, used paths)`` or ``None`` when they cannot reach it."""
    import numpy as np
    import soundfile as sf

    parts, used, rate, total = [], [], None, 0.0
    for p in paths:
        x, sr = sf.read(p, dtype="float32", always_2d=True)
        x = x[:, 0] if x.shape[1] > 1 else x.mean(axis=1)
        if rate is None:
            rate = sr
        elif sr != rate:
            continue
        add = len(x) / rate + (gap_ms / 1000.0 if parts else 0.0)
        if parts and total + add > max_s:
            break
        if parts:
            parts.append(np.zeros(int(gap_ms / 1000.0 * rate), dtype="float32"))
        parts.append(x)
        used.append(p)
        total += add
        if total >= min_s:
            return np.concatenate(parts), rate, used
    return None


def normalize(x, rate: int, lufs: float):
    """``(audio at the target loudness, gain dB, note)``; the gain is lowered to keep the peak at
    -1 dBFS (noted)."""
    import numpy as np

    loud = vb.loudness_lufs(x, rate)
    if loud is None:
        return x, 0.0, "loudness not measurable"
    gain = lufs - loud
    peak = float(np.max(np.abs(x))) * 10 ** (gain / 20)
    note = ""
    if peak > 10 ** (-1 / 20):
        gain -= 20 * np.log10(peak / 10 ** (-1 / 20))
        note = f"peak-limited: {lufs - gain:.1f} dB under the loudness target"
    return (x * 10 ** (gain / 20)).astype("float32"), gain, note


def librispeech_plan(root: str, per_speaker: int, seed: int, a):
    """Clips of LibriSpeech test-clean under ``root`` (the extracted ``LibriSpeech`` folder)."""
    speakers = {}
    with open(os.path.join(root, "SPEAKERS.TXT"), encoding="utf-8") as f:
        for line in f:
            if line.startswith(";"):
                continue
            parts = [p.strip() for p in line.split("|")]
            if len(parts) >= 3 and parts[2] == "test-clean":
                speakers[parts[0]] = parts[1]
    base = os.path.join(root, "test-clean")
    rng = random.Random(seed)
    clips = []
    for spk in sorted(os.listdir(base), key=lambda s: int(s) if s.isdigit() else s):
        chapters = sorted(os.listdir(os.path.join(base, spk)))
        order = chapters[:]
        rng.shuffle(order)
        got = 0
        for ch in order:
            utts = sorted(f for f in os.listdir(os.path.join(base, spk, ch)) if f.endswith(".flac"))
            starts = list(range(len(utts)))
            rng.shuffle(starts)
            for st in starts:
                if got >= per_speaker:
                    break
                j = join_utterances([os.path.join(base, spk, ch, u) for u in utts[st:]], a.gap_ms, a.min_seconds, a.max_seconds)
                if j is None:
                    continue
                x, rate, used = j
                ids = [os.path.splitext(os.path.basename(u))[0] for u in used]
                if any(c["_ids"] & set(ids) for c in clips):
                    continue
                clips.append({"id": f"libri-{spk}-{got + 1}", "x": x, "rate": rate, "speaker": f"librispeech-{spk}", "sex": speakers.get(spk, "?"),
                              "language": "en", "source": "LibriSpeech test-clean: " + ", ".join(ids), "license": SOURCES["librispeech"]["license"], "_ids": set(ids)})
                got += 1
            if got >= per_speaker:
                break
    return clips


def arctic_plan(root: str, spk: str, per_speaker: int, a):
    """Clips of one ARCTIC speaker (``root`` = the extracted ``cmu_us_<spk>_arctic`` folder) with
    EGG truth."""
    import numpy as np
    import soundfile as sf

    sys.path.insert(0, os.path.join(REPO, "scripts", "vc-bench"))
    import egg_f0

    orig = os.path.join(root, "orig")
    files = sorted(f for f in os.listdir(orig) if f.endswith(".wav"))
    clips, i = [], 0
    while len(clips) < per_speaker and i < len(files):
        speech, f0s, used, rate, total = [], [], [], None, 0.0
        while i < len(files) and total < a.min_seconds:
            p = os.path.join(orig, files[i])
            i += 1
            x, sr = sf.read(p, dtype="float32", always_2d=True)
            if x.shape[1] < 2:
                print(f"{p}: no EGG channel; skipped")
                continue
            f0, _pol, _n = egg_f0.egg_truth(x[:, 1], sr)
            hop = int(0.01 * sr)
            s = x[: len(f0) * hop, 0]
            if speech:
                gap = int(round(a.gap_ms / 10.0))
                speech.append(np.zeros(gap * hop, dtype="float32"))
                f0s.append(np.zeros(gap))
                total += a.gap_ms / 1000.0
            speech.append(s)
            f0s.append(f0)
            used.append(os.path.splitext(files[i - 1])[0])
            rate = sr
            total += len(s) / sr
        if total >= a.min_seconds:
            clips.append({"id": f"arctic-{spk}-{len(clips) + 1}", "x": np.concatenate(speech), "rate": rate, "f0": np.concatenate(f0s),
                          "speaker": f"arctic-{spk}", "sex": ARCTIC_SEX[spk], "language": "en",
                          "source": f"CMU ARCTIC {spk}: " + ", ".join(used), "license": SOURCES[f"arctic_{spk}"]["license"]})
    return clips


def write_corpus(clips, out: str, a, voice_sha: str, archives: dict | None = None) -> str:
    import soundfile as sf

    sys.path.insert(0, os.path.join(REPO, "scripts", "vc-bench"))
    import egg_f0

    os.makedirs(os.path.join(out, "clips"), exist_ok=True)
    entries = []
    for c in clips:
        x, gain, note = normalize(c["x"], c["rate"], a.lufs)
        rel = f"clips/{c['id']}.wav"
        path = os.path.join(out, rel)
        sf.write(path, x, c["rate"], subtype="PCM_16")
        e = {"id": c["id"], "input": rel, "sha256": vb.sha256_file(path), "speaker": c["speaker"], "sex": c["sex"],
             "language": c["language"], "seconds": round(len(x) / c["rate"], 3), "source": c["source"], "license": c["license"],
             "notes": f"loudness-normalised to {a.lufs} LUFS (gain {gain:+.1f} dB){'; ' + note if note else ''}"}
        if c.get("f0") is not None:
            truth = f"clips/{c['id']}.f0.csv"
            egg_f0.write_csv(os.path.join(out, truth), c["f0"])
            e["f0_truth"] = truth
        entries.append(e)
    manifest = {
        "schema": vb.CORPUS_SCHEMA,
        "name": os.path.basename(os.path.normpath(out)),
        "description": f"Public T6 corpus (scripts/bench/fetch-public-corpus.py, seed {a.seed}): LibriSpeech test-clean and CMU ARCTIC bdl/jmk/slt with EGG truth, clips of at least {a.min_seconds} s, {a.lufs} LUFS. Never commit the audio.",
        "voice": {"name": "MIT default voice (PhoenixStormJr/RVC-V2-default-voice)", "sha256": voice_sha},
        "transpose": 0,
        "clips": entries,
        "points": [],
        "min_clips": 20,
    }
    if archives:
        # The archives are not verified against published hashes (there are none for ARCTIC, and the
        # festvox mirror is plain http): their hashes are recorded so a later run can be compared.
        manifest["$comment"] = {"archives": archives}
    path = os.path.join(out, "corpus.json")
    with open(path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=1)
    return path


def plan_text(a) -> str:
    skip = {s.strip() for s in a.skip.split(",") if s.strip()}
    lines = [f"output: {os.path.abspath(a.out)}{'  (INSIDE the repository: needs --force)' if inside_repo(a.out) else ''}",
             f"archives: {os.path.abspath(a.archives or os.path.join(a.out, 'archives'))}", "sources (downloaded one at a time, resumable):"]
    total = 0
    for name, s in SOURCES.items():
        if name in skip:
            continue
        total += s["approx_mb"]
        lines.append(f"  {name}: {s['file']} (about {s['approx_mb']} MB) from {s['urls'][0]} (+{len(s['urls']) - 1} mirror(s)); {s['license']}")
    n_l = 0 if "librispeech" in skip else 40 * a.libri_per_speaker
    n_a = sum(a.arctic_per_speaker for s in ("bdl", "jmk", "slt") if f"arctic_{s}" not in skip)
    lines += [f"download total: about {total} MB",
              f"clips: about {n_l} LibriSpeech (40 speakers x {a.libri_per_speaker}, seed {a.seed}) + {n_a} ARCTIC with EGG truth = {n_l + n_a}",
              f"each {a.min_seconds}-{a.max_seconds} s of consecutive utterances, {a.gap_ms:g} ms gaps, {a.lufs} LUFS, 16-bit PCM",
              "the T6 DNSMOS row needs about 68 clips at a paired SD of 0.21 (50 % power), about 139 for 80 %"]
    return "\n".join(lines)


def run(a) -> str:
    if inside_repo(a.out) and not a.force:
        raise SystemExit(f"{a.out} is inside the repository: corpus audio is never committed (use --force to write there anyway)")
    skip = {s.strip() for s in a.skip.split(",") if s.strip()}
    arch = a.archives or os.path.join(a.out, "archives")
    work = os.path.join(a.out, "extracted")
    clips = []
    if "librispeech" not in skip:
        root = os.path.join(work, "LibriSpeech")
        if not extracted(work, "librispeech"):
            extract(fetch("librispeech", arch), work, lambda n: n.startswith("LibriSpeech/test-clean/") or n == "LibriSpeech/SPEAKERS.TXT", "librispeech")
        clips += librispeech_plan(root, a.libri_per_speaker, a.seed, a)
    for spk in ("bdl", "jmk", "slt"):
        if f"arctic_{spk}" in skip:
            continue
        root = os.path.join(work, f"cmu_us_{spk}_arctic")
        if not extracted(work, f"arctic_{spk}"):
            extract(
                fetch(f"arctic_{spk}", arch),
                work,
                lambda n, s=spk: n.startswith(f"cmu_us_{s}_arctic/orig/") or n.startswith(f"cmu_us_{s}_arctic/COPYING") or n.startswith(f"cmu_us_{s}_arctic/README"),
                f"arctic_{spk}",
            )
        clips += arctic_plan(root, spk, a.arctic_per_speaker, a)
    voice_sha = vb.sha256_file(a.voice) if a.voice else ""
    archives = {}
    for name, src in SOURCES.items():
        archive = os.path.join(arch, src["file"])
        if name not in skip and os.path.isfile(archive):
            archives[name] = {"file": src["file"], "bytes": os.path.getsize(archive), "sha256": vb.sha256_file(archive)}
            print(f"{src['file']}: sha256 {archives[name]['sha256']} (compare OpenSLR's md5sum.txt for LibriSpeech if you want to verify it)")
    path = write_corpus(clips, a.out, a, voice_sha, archives)
    print(f"{len(clips)} clips -> {path}")
    return path


def transfer_checks(d: str) -> bool:
    """The download and extraction guards, against a local HTTP server on 127.0.0.1 (no network): a
    connection that closes early is not an archive and the next run resumes it, a truncated archive
    is recognised, a failed extraction leaves nothing that reads as done, and a corpus folder on
    another drive does not crash the repository check."""
    import http.server
    import threading

    ok = True
    src = os.path.join(d, "src")
    os.makedirs(src)
    for i in range(3):
        with open(os.path.join(src, f"f{i}.bin"), "wb") as f:
            f.write(os.urandom(200_000))  # incompressible, so a cut lands inside the stream
    full_path = os.path.join(d, "full.tar.gz")
    with tarfile.open(full_path, "w:gz") as tar:
        tar.add(src, arcname="LibriSpeech")
    full = open(full_path, "rb").read()

    class Handler(http.server.BaseHTTPRequestHandler):
        protocol_version = "HTTP/1.1"

        def log_message(self, *_args):
            pass

        def do_GET(self):  # noqa: N802
            start = 0
            rng = self.headers.get("Range")
            if rng and self.path == "/full":
                start = int(rng.split("=")[1].split("-")[0])
            body = full[start:]
            self.send_response(206 if start else 200)
            if start:
                self.send_header("Content-Range", f"bytes {start}-{len(full) - 1}/{len(full)}")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            if self.path == "/cut":  # announce everything, send half, hang up
                self.wfile.write(body[: len(body) // 2])
                self.wfile.flush()
                self.close_connection = True
                return
            self.wfile.write(body)

    server = http.server.ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    base = f"http://127.0.0.1:{server.server_address[1]}"
    try:
        dest = os.path.join(d, "dl", "test-clean.tar.gz")
        os.makedirs(os.path.dirname(dest))
        try:
            download(base + "/cut", dest, 1)
            ok = False  # a short body was taken as the archive
        except ShortDownload:
            pass
        ok &= not os.path.exists(dest) and os.path.getsize(dest + ".partial") == len(full) // 2
        download(base + "/full", dest, 1)  # resumes from the partial
        ok &= open(dest, "rb").read() == full and not os.path.exists(dest + ".partial") and archive_problem(dest) == ""
    finally:
        server.shutdown()
        server.server_close()
    cut = os.path.join(d, "cut.tar.gz")
    with open(cut, "wb") as f:
        f.write(full[: len(full) // 2])
    ok &= bool(archive_problem(cut))
    work = os.path.join(d, "work")
    try:
        extract(cut, work, lambda n: True, "librispeech")
        ok = False
    except Exception:  # noqa: BLE001 (the truncated archive must fail)
        pass
    ok &= not extracted(work, "librispeech") and not any(e.startswith(".extracting") or e == "LibriSpeech" for e in os.listdir(work))
    extract(full_path, work, lambda n: True, "librispeech")
    ok &= extracted(work, "librispeech") and len(os.listdir(os.path.join(work, "LibriSpeech"))) == 3
    if os.name == "nt":
        other = "Z:\\corpus-x" if REPO[:1].upper() != "Z" else "Y:\\corpus-x"
        ok &= inside_repo(other) is False
    ok &= inside_repo(os.path.join(REPO, "x")) is True
    return bool(ok)


def self_test() -> bool:
    """Synthetic LibriSpeech and ARCTIC trees (no network): selection, joining, loudness, truth."""
    import tempfile

    import numpy as np
    import soundfile as sf

    sys.path.insert(0, os.path.join(REPO, "scripts", "vc-bench"))
    import egg_f0

    d = tempfile.mkdtemp()
    rng = np.random.default_rng(3)
    libri = os.path.join(d, "extracted", "LibriSpeech")
    os.makedirs(os.path.join(libri, "test-clean"))
    with open(os.path.join(libri, "SPEAKERS.TXT"), "w", encoding="utf-8") as f:
        f.write(";ID|SEX|SUBSET|MINUTES|NAME\n1 | F | test-clean | 8 | A\n2 | M | test-clean | 8 | B\n")
    for spk in ("1", "2"):
        for ch in ("10", "11"):
            p = os.path.join(libri, "test-clean", spk, ch)
            os.makedirs(p)
            for u in range(6):
                x = (0.05 * rng.standard_normal(int(16000 * rng.uniform(2.5, 4.5)))).astype("float32")
                sf.write(os.path.join(p, f"{spk}-{ch}-{u:04d}.flac"), x, 16000)
    arctic = os.path.join(d, "extracted", "cmu_us_slt_arctic", "orig")
    os.makedirs(arctic)
    for u in range(8):
        egg, _ = egg_f0.synthetic_egg(16000, 3.0, seed=u)
        speech = 0.1 * egg + 0.01 * rng.standard_normal(len(egg)).astype("float32")
        sf.write(os.path.join(arctic, f"arctic_a{u:04d}.wav"), np.stack([speech, egg], axis=1), 16000)
    a = parse_args(["--out", os.path.join(d, "out"), "--archives", os.path.join(d, "none"), "--skip", "arctic_bdl,arctic_jmk", "--arctic-per-speaker", "2"])
    a.out = os.path.join(d, "out")
    clips = librispeech_plan(libri, a.libri_per_speaker, a.seed, a)
    clips += arctic_plan(os.path.join(d, "extracted", "cmu_us_slt_arctic"), "slt", 2, a)
    again = librispeech_plan(libri, a.libri_per_speaker, a.seed, a)
    ok = len(clips) == 6 and [c["source"] for c in again] == [c["source"] for c in clips[:4]]
    path = write_corpus(clips, a.out, a, "")
    m = json.load(open(path, encoding="utf-8"))
    ok &= m["schema"] == vb.CORPUS_SCHEMA and len(m["clips"]) == 6
    ok &= all(c["seconds"] >= a.min_seconds for c in m["clips"])
    ok &= sum(1 for c in m["clips"] if c.get("f0_truth")) == 2
    for c in m["clips"]:
        x, sr = sf.read(os.path.join(a.out, c["input"]), dtype="float32")
        ok &= abs(vb.loudness_lufs(x, sr) - a.lufs) < 0.3 and vb.sha256_file(os.path.join(a.out, c["input"])) == c["sha256"]
    ok &= "INSIDE" in plan_text(parse_args(["--out", os.path.join(REPO, "x")]))
    t = os.path.join(d, "transfer")
    os.makedirs(t)
    ok &= transfer_checks(t)
    shutil.rmtree(d, ignore_errors=True)
    print("self-test", "passed" if ok else "FAILED")
    return ok


def main(argv=None):
    a = parse_args(argv)
    if a.self_test:
        sys.exit(0 if self_test() else 1)
    if a.dry_run:
        print(plan_text(a))
        return
    run(a)


if __name__ == "__main__":
    main()
