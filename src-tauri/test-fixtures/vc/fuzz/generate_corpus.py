#!/usr/bin/env python3
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Generate the voice importer's malformed-input corpus (plan T8: a 200-file .pth / .zip fuzz
corpus with 0 crashes or hangs).

    python src-tauri/test-fixtures/vc/fuzz/generate_corpus.py

writes ``corpus/*`` and ``corpus.json`` next to this script. Deterministic (fixed seeds, no
timestamps) and stdlib-only: pickles are assembled opcode by opcode and zips byte by byte, so the
generator needs neither torch nor a zip library that "fixes" hostile names. The only input is the
committed tiny test voice (``../tiny/tiny.pth``), whose tensor table the valid controls reuse.

``corpus.json`` lists every file with the importer result the Rust test
(``vc::import::corpus_tests``) expects against the tiny templates (``../tiny``):

- an error code of ``vc::import::ErrorCode`` (``pickle_rejected``, ``archive_rejected``, ...);
- ``ok`` for the few valid controls (a training checkpoint and a bare state dict, both with the
  ``_metadata`` ``BUILD`` ``torch.save`` writes for a state dict, and voice archives);
- ``any`` for random mutations: any result is fine, as long as the importer neither panics nor
  hangs.

Categories: hostile zip structure, hostile torch archives, hostile pickles (code-execution
globals, object-construction opcodes, malformed programs, resource bombs), tensor views and
storages, checkpoint semantics (metadata, shapes, non-finite weights), valid controls, and seeded
random mutations of small checkpoints.
"""

import collections
import json
import os
import pickle
import random
import shutil
import struct
import zlib

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "corpus")
TINY = os.path.join(HERE, "..", "tiny", "tiny.pth")
TOTAL = 200

# The tiny template's RVC config (scripts/vc-templates/rvc_arch/configs.py, "v2-16k-f0-hifigan").
TINY_CONFIG = [1025, 32, 8, 8, 16, 2, 2, 3, 0, "1", [3, 5], [[1, 3, 5], [1, 3, 5]], [4, 2], 16, [8, 4], 3, 8, 16000]

# ---------------------------------------------------------------------------------------------
# Zip writer (byte level: names, flags, methods, sizes and offsets exactly as given)
# ---------------------------------------------------------------------------------------------


def deflate(data):
    c = zlib.compressobj(9, zlib.DEFLATED, -15)
    return c.compress(data) + c.flush()


def zip_bytes(entries, comment=b"", n_override=None, disk=0, cd_offset_delta=0, prefix=b"", suffix=b""):
    """entries: dicts with name (str or bytes), data, and optional method (0 stored / 8 deflate /
    other), flags, create_system, attr, crc, csize, usize, local_offset, comp."""
    out = bytearray(prefix)
    central = bytearray()
    for e in entries:
        name = e["name"].encode("utf-8") if isinstance(e["name"], str) else e["name"]
        data = e.get("data", b"")
        method = e.get("method", 0)
        comp = e.get("comp", deflate(data) if method == 8 else data)
        crc = e.get("crc", zlib.crc32(data) & 0xFFFFFFFF)
        usize = e.get("usize", len(data))
        csize = e.get("csize", len(comp))
        flags = e.get("flags", 0)
        off = len(out)
        out += struct.pack("<IHHHHHIIIHH", 0x04034B50, 20, flags, method, 0, 0x21, crc, csize, usize, len(name), 0) + name + comp
        made = (e.get("create_system", 0) << 8) | 20
        central += struct.pack(
            "<IHHHHHHIIIHHHHHII",
            0x02014B50, made, 20, flags, method, 0, 0x21, crc, csize, usize, len(name), 0, 0, 0, 0,
            e.get("attr", 0), e.get("local_offset", off),
        ) + name
    cd_off = len(out)
    out += central
    n = len(entries) if n_override is None else n_override
    on_disk, total = (n, n) if not isinstance(n, tuple) else n
    eocd = struct.pack("<IHHHHIIH", 0x06054B50, disk, disk, on_disk, total, len(central), cd_off + cd_offset_delta, len(comment)) + comment
    return bytes(out + eocd + suffix)


def entry(name, data=b"", **kw):
    d = {"name": name, "data": data}
    d.update(kw)
    return d


# ---------------------------------------------------------------------------------------------
# Pickle assembler (protocol 2 opcodes, like the Rust tests' `P`)
# ---------------------------------------------------------------------------------------------


class Tensor:
    def __init__(self, key, stype, numel, offset, shape, stride, hooks=False):
        self.key, self.stype, self.numel, self.offset, self.shape, self.stride, self.hooks = key, stype, numel, offset, shape, stride, hooks


class StateDict(collections.OrderedDict):
    """A module's ``state_dict()`` as ``torch.save`` pickles it: the OrderedDict, then ``BUILD``
    with ``{"_metadata": OrderedDict({prefix: {"version": 1}})}`` (PyTorch's per-module version
    table, one entry per module prefix of the keys)."""

    def prefixes(self):
        out = [""]
        for key in self:
            parts = key.split(".")[:-1]
            for i in range(1, len(parts) + 1):
                prefix = ".".join(parts[:i])
                if prefix not in out:
                    out.append(prefix)
        return out


class Raw:
    """Opcodes inserted verbatim."""

    def __init__(self, b):
        self.b = b


class P:
    def __init__(self, proto=2):
        self.b = bytearray(b"\x80" + bytes([proto]))

    def op(self, *ops):
        for o in ops:
            self.b += bytes([o]) if isinstance(o, int) else o
        return self

    def glob(self, module, name):
        self.b += b"c" + module.encode() + b"\n" + name.encode() + b"\n"
        return self

    def s(self, text):
        d = text.encode("utf-8")
        self.b += b"X" + struct.pack("<I", len(d)) + d
        return self

    def i(self, v):
        self.b += b"J" + struct.pack("<i", v)
        return self

    def long(self, v):
        n = max(1, (v.bit_length() + 8) // 8)
        self.b += b"\x8a" + bytes([n]) + v.to_bytes(n, "little", signed=True)
        return self

    def f(self, v):
        self.b += b"G" + struct.pack(">d", v)
        return self

    def val(self, v):
        if isinstance(v, Raw):
            self.b += v.b
        elif isinstance(v, Tensor):
            self.tensor(v)
        elif v is None:
            self.op(b"N")
        elif isinstance(v, bool):
            self.op(0x88 if v else 0x89)
        elif isinstance(v, int):
            if -(2**31) <= v < 2**31:
                self.i(v)
            else:
                self.long(v)
        elif isinstance(v, float):
            self.f(v)
        elif isinstance(v, str):
            self.s(v)
        elif isinstance(v, StateDict):
            self.glob("collections", "OrderedDict").op(b")", b"R")
            if v:
                self.op(b"(")
                for k, x in v.items():
                    self.val(k).val(x)
                self.op(b"u")
            self.metadata_build(v.prefixes())
        elif isinstance(v, collections.OrderedDict):
            self.glob("collections", "OrderedDict").op(b")", b"R")
            if v:
                self.op(b"(")
                for k, x in v.items():
                    self.val(k).val(x)
                self.op(b"u")
        elif isinstance(v, dict):
            self.op(b"}")
            if v:
                self.op(b"(")
                for k, x in v.items():
                    self.val(k).val(x)
                self.op(b"u")
        elif isinstance(v, list):
            self.op(b"]")
            if v:
                self.op(b"(")
                for x in v:
                    self.val(x)
                self.op(b"e")
        elif isinstance(v, tuple):
            self.op(b"(")
            for x in v:
                self.val(x)
            self.op(b"t")
        else:
            raise TypeError(type(v))
        return self

    def metadata_state(self, prefixes, extra=()):
        """Push ``{"_metadata": OrderedDict({prefix: {"version": 1}}), **extra}``."""
        self.op(b"}").s("_metadata").glob("collections", "OrderedDict").op(b")", b"R")
        if prefixes:
            self.op(b"(")
            for prefix in prefixes:
                self.s(prefix).op(b"}").s("version").i(1).op(b"s")
            self.op(b"u")
        self.op(b"s")
        for k, x in extra:
            self.val(k).val(x).op(b"s")
        return self

    def metadata_build(self, prefixes, extra=()):
        """``BUILD`` the dict on the stack with torch's state-dict metadata (as torch.save does)."""
        return self.metadata_state(prefixes, extra).op(b"b")

    def tensor(self, t):
        self.glob("torch._utils", "_rebuild_tensor_v2").op(b"(")
        self.op(b"(").s("storage").glob("torch", t.stype).s(t.key).s("cpu").val(t.numel).op(b"t", b"Q")
        self.val(t.offset).val(tuple(t.shape)).val(tuple(t.stride)).op(0x89)
        if t.hooks:
            self.op(b"}").s("hook").i(1).op(b"s")
        else:
            self.glob("collections", "OrderedDict").op(b")", b"R")
        return self.op(b"t", b"R")

    def stop(self):
        return bytes(self.b + b".")


def pkl(value):
    return P().val(value).stop()


def torch_zip(pickle_bytes, storages=None, prefix="archive", byteorder=b"little", extra=()):
    files = [entry(f"{prefix}/data.pkl", pickle_bytes)]
    if byteorder is not None:
        files.append(entry(f"{prefix}/byteorder", byteorder))
    for k, v in (storages or {}).items():
        files.append(entry(f"{prefix}/data/{k}", v))
    files.append(entry(f"{prefix}/version", b"3\n"))
    files.extend(extra)
    return zip_bytes(files)


# ---------------------------------------------------------------------------------------------
# The tiny voice (valid controls) and a micro checkpoint (semantic cases)
# ---------------------------------------------------------------------------------------------


class TinyReader(pickle.Unpickler):
    """Reads tiny.pth's data.pkl into plain Python (nothing but the tensor table is built)."""

    def find_class(self, module, name):
        if (module, name) == ("collections", "OrderedDict"):
            return collections.OrderedDict
        if (module, name) == ("torch._utils", "_rebuild_tensor_v2"):
            return lambda storage, offset, size, stride, *rest: Tensor(storage[0], storage[1], storage[2], offset, list(size), list(stride))
        if module == "torch" and name.endswith("Storage"):
            return name
        raise pickle.UnpicklingError(f"{module}.{name}")

    def persistent_load(self, pid):
        _, stype, key, _location, numel = pid
        return (key, stype, numel)


def read_tiny():
    import io
    import zipfile

    with zipfile.ZipFile(TINY) as z:
        names = z.namelist()
        pkl_name = next(n for n in names if n.endswith("/data.pkl"))
        prefix = pkl_name[: -len("data.pkl")]
        root = TinyReader(io.BytesIO(z.read(pkl_name))).load()
        storages = {n[len(prefix) + len("data/"):]: z.read(n) for n in names if n.startswith(prefix + "data/")}
    return root, storages


def f16(x):
    return struct.pack("<e", x)


MICRO_STORAGES = {"0": f16(0.5), "1": f16(0.25), "2": f16(0.125)}


def micro_weights():
    return collections.OrderedDict(
        [
            # Expanded views (stride 0) keep the files tiny while the shapes match the template.
            ("enc_p.emb_phone.weight", Tensor("0", "HalfStorage", 1, 0, [8, 768], [0, 0])),
            ("enc_p.emb_pitch.weight", Tensor("1", "HalfStorage", 1, 0, [256, 8], [0, 0])),
            ("emb_g.weight", Tensor("2", "HalfStorage", 1, 0, [3, 8], [0, 0])),
        ]
    )


def micro_meta():
    return collections.OrderedDict([("config", list(TINY_CONFIG)), ("info", "fuzz"), ("sr", "16k"), ("f0", 1), ("version", "v2")])


def micro(weights=None, meta=None, storages=None, **zkw):
    """An RVC inference export shaped for the tiny template that holds only the first few
    weights: the importer gets through metadata and template matching and stops at the writer."""
    root = collections.OrderedDict([("weight", micro_weights() if weights is None else weights)])
    root.update(micro_meta() if meta is None else meta)
    return torch_zip(pkl(root), MICRO_STORAGES if storages is None else storages, **zkw)


# ---------------------------------------------------------------------------------------------
# Corpus
# ---------------------------------------------------------------------------------------------

corpus = []


def add(name, data, expect, note):
    assert all(c.isalnum() or c in "-_." for c in name), name
    assert name not in {c["file"] for c in corpus}, name
    corpus.append({"file": name, "expect": expect, "note": note, "_data": data})


def zip_attacks():
    ok_pth = micro()
    for i, (name, note) in enumerate(
        [
            (b"../evil.pth", "parent traversal"),
            (b"a/../../evil.pth", "nested traversal"),
            (b"/abs/evil.pth", "absolute path"),
            (b"C:/evil.pth", "drive letter"),
            (b"a\\..\\evil.pth", "backslash traversal"),
            (b"./evil.pth", "dot component"),
            (b"a//evil.pth", "empty component"),
            (b"evil\x01.pth", "control character"),
            (b"evil\x00.pth", "NUL in name"),
            (b"a/" + b"x" * 600 + b".pth", "600-byte name"),
        ]
    ):
        add(f"zip-name-{i:02d}.zip", zip_bytes([entry(name, ok_pth)]), "archive_rejected", note)
    add("zip-symlink.zip", zip_bytes([entry("voice.pth", b"../../etc/passwd", create_system=3, attr=(0o120777 << 16))]), "archive_rejected", "symbolic link entry")
    add("zip-duplicate.zip", zip_bytes([entry("v.pth", ok_pth), entry("v.pth", ok_pth)]), "archive_rejected", "duplicate names")
    add("zip-encrypted.zip", zip_bytes([entry("v.pth", ok_pth, flags=1)]), "archive_rejected", "encrypted flag")
    add("zip-bzip2.zip", zip_bytes([entry("v.pth", ok_pth, method=12, comp=ok_pth)]), "archive_rejected", "bzip2 method")
    add("zip-lzma.zip", zip_bytes([entry("v.pth", ok_pth, method=14, comp=ok_pth)]), "archive_rejected", "lzma method")
    add("zip-bomb.zip", zip_bytes([entry("bomb.pth", b"\0" * (8 << 20), method=8)]), "archive_rejected", "8 MB of zeros, ratio > 200")
    add("zip-65-entries.zip", zip_bytes([entry(f"f{i}.txt", b"") for i in range(65)]), "archive_rejected", "65 entries (voice archives allow 64)")
    add("zip-huge-declared.zip", zip_bytes([entry("v.pth", ok_pth, usize=3 << 30)]), "archive_rejected", "3 GB declared size")
    whole = zip_bytes([entry("v.pth", ok_pth)])
    add("zip-truncated.zip", whole[: len(whole) // 2], "archive_rejected", "cut in half")
    add("zip-appended.zip", whole + b"trailing bytes", "archive_rejected", "data after the end record")
    add("zip-prepended.zip", b"MZ\x90\x00" + whole, "not_a_checkpoint", "an executable stub before the archive")
    fake = struct.pack("<IHHHHIIH", 0x06054B50, 0, 0, 1, 1, 46, 0, 0)
    add("zip-forged-eocd.zip", zip_bytes([entry("v.pth", ok_pth)], comment=b"x" * 10 + fake), "archive_rejected", "a second end record inside the comment")
    add("zip-empty.zip", zip_bytes([]), "not_a_checkpoint", "an empty archive")
    add("zip-zero-bytes.zip", b"", "not_a_checkpoint", "an empty file")
    add("zip-multidisk.zip", zip_bytes([entry("v.pth", ok_pth)], disk=1), "archive_rejected", "multi-disk end record")
    add("zip-count-mismatch.zip", zip_bytes([entry("a.pth", ok_pth), entry("b.txt", b"x")], n_override=(2, 1)), "archive_rejected", "entries on disk != total")
    add("zip-count-lie.zip", zip_bytes([entry("a.pth", ok_pth), entry("b.txt", b"x")], n_override=1), "archive_rejected", "declares fewer entries than the directory holds")
    add("zip-cd-offset.zip", zip_bytes([entry("v.pth", ok_pth)], cd_offset_delta=-7), "archive_rejected", "central directory offset off by 7")
    add("zip-local-offset.zip", zip_bytes([entry("v.pth", ok_pth, local_offset=0x7FFFFFF0)]), "archive_rejected", "local header far outside the archive")
    add("zip-bad-crc.zip", zip_bytes([entry("v.pth", ok_pth, crc=0x12345678)]), "archive_rejected", "CRC mismatch")
    bad_deflate = entry("v.pth", ok_pth, method=8)
    bad_deflate["comp"] = b"\xff" * 64
    add("zip-bad-deflate.zip", zip_bytes([bad_deflate]), "archive_rejected", "corrupt deflate stream")
    add("zip-nested.zip", zip_bytes([entry("inner.zip", zip_bytes([entry("v.pth", ok_pth)]))]), "not_a_checkpoint", "a zip inside a zip")
    add("zip-two-voices.zip", zip_bytes([entry("a/A.pth", ok_pth), entry("b/B.pth", ok_pth)]), "ambiguous_archive", "two voices")
    add("zip-metadata-only.zip", zip_bytes([entry("__MACOSX/._v.pth", b"\0" * 64), entry("._v.pth", b"\0" * 64)]), "not_a_checkpoint", "only macOS metadata")
    add("zip-onnx-only.zip", zip_bytes([entry("voice/model.onnx", b"\x08\x07"), entry("voice/added.index", b"x")]), "not_a_checkpoint", "an ONNX voice")
    add("zip-discriminator-only.zip", zip_bytes([entry("D_2333.pth", ok_pth)]), "not_a_checkpoint", "only a training by-product")
    add("zip-pth-not-zip.zip", zip_bytes([entry("v.pth", b"RIFF....WAVEfmt ")]), "not_a_checkpoint", "the .pth is a WAV")
    add("zip-pth-legacy.zip", zip_bytes([entry("v.pth", b"\x80\x02\x8a\x0al\xfc\x9cF\xf9 j\xa8")]), "legacy_format", "the .pth is a legacy pickle")
    for i, size in enumerate([1, 3, 4, 21, 22, 100, 4096]):
        data = bytearray(random.Random(1000 + i).randbytes(size))
        if data[0] in (0x80, ord("P")):
            data[0] = 0x7F  # neither a pickle (legacy_format) nor a zip signature
        add(f"random-{i}.pth", bytes(data), "not_a_checkpoint", f"{size} random bytes")
    add("random-pk.pth", b"PK\x03\x04" + random.Random(7).randbytes(300), "archive_rejected", "a zip signature, then noise")


FORBIDDEN_GLOBALS = [
    ("os", "system"), ("posix", "system"), ("nt", "system"), ("builtins", "eval"), ("builtins", "exec"),
    ("builtins", "getattr"), ("builtins", "__import__"), ("__builtin__", "eval"), ("subprocess", "Popen"),
    ("subprocess", "call"), ("shutil", "rmtree"), ("webbrowser", "open"), ("pickle", "loads"), ("torch", "load"),
    ("torch._utils", "_rebuild_tensor"), ("numpy.core.multiarray", "_reconstruct"), ("collections", "OrderedDictX"),
    ("codecs", "encode"), ("runpy", "_run_code"), ("ctypes", "CDLL"),
]


def pickle_attacks():
    for module, name in FORBIDDEN_GLOBALS:
        evil = P().glob(module, name).s("calc.exe").op(0x85, b"R").stop()
        add(f"global-{module.replace('.', '_')}-{name.strip('_')}.pth", torch_zip(evil), "pickle_rejected", f"GLOBAL {module}.{name}")
    for module, name in [("os", "system"), ("builtins", "eval"), ("subprocess", "Popen")]:
        evil = P().s(module).s(name).op(0x93).s("calc.exe").op(0x85, b"R").stop()
        add(f"stack-global-{module}-{name}.pth", torch_zip(evil), "pickle_rejected", f"STACK_GLOBAL {module}.{name}")
    # A forbidden global hidden inside an otherwise valid-looking checkpoint (after the weights).
    root = collections.OrderedDict([("weight", micro_weights()), ("config", list(TINY_CONFIG)), ("info", Raw(P().glob("os", "system").s("x").op(0x85, b"R").b[2:]))])
    add("global-inside-checkpoint.pth", torch_zip(pkl(root), MICRO_STORAGES), "pickle_rejected", "os.system in the metadata")
    for op, label in [
        (0x81, "NEWOBJ"), (0x92, "NEWOBJ_EX"), (b"o", "OBJ"), (b"i", "INST"), (b"b", "BUILD"), (0x82, "EXT1"),
        (0x83, "EXT2"), (0x84, "EXT4"), (b"P", "PERSID"), (b"I", "INT"), (b"S", "STRING"), (b"V", "UNICODE"),
        (b"F", "FLOAT"), (b"L", "LONG"), (b"p", "PUT"), (b"g", "GET"), (0x97, "NEXT_BUFFER"), (0x98, "READONLY_BUFFER"),
    ]:
        data = P().op(b"N").op(op).stop()
        add(f"opcode-{label.lower()}.pth", torch_zip(data), "pickle_rejected", f"opcode {label}")
    # BUILD runs only in torch's state-dict shape (an OrderedDict, the `_metadata` state, once).
    one = Tensor("0", "HalfStorage", 1, 0, [1], [1])
    od = lambda p: p.glob("collections", "OrderedDict").op(b")", b"R")  # noqa: E731
    for name, data, note in [
        ("pickle-build-tensor.pth", P().val(one).metadata_build([""]).stop(), "BUILD on a tensor"),
        ("pickle-build-list.pth", P().op(b"]").metadata_build([""]).stop(), "BUILD on a list"),
        ("pickle-build-plain-dict.pth", P().op(b"}").metadata_build([""]).stop(), "BUILD on a plain dict"),
        ("pickle-build-extra-state.pth", od(P()).metadata_build([""], [("__class__", "x")]).stop(), "a state with another key"),
        ("pickle-build-tensor-metadata.pth", od(P()).op(b"}").s("_metadata").op(b"}").s("").op(b"}").s("version").val(one).op(b"s", b"s", b"s", b"b").stop(), "a tensor in the metadata"),
        ("pickle-build-twice.pth", od(P()).op(b"q", 0).metadata_build([""]).op(b"0", b"h", 0).metadata_build([""]).stop(), "a second BUILD on one dict"),
    ]:
        add(name, torch_zip(data, {"0": f16(1.0)}), "pickle_rejected", note)
    base = pkl(collections.OrderedDict([("weight", micro_weights()), ("config", list(TINY_CONFIG))]))
    add("pickle-truncated.pth", torch_zip(base[: len(base) // 2], MICRO_STORAGES), "pickle_rejected", "data.pkl cut in half")
    add("pickle-no-stop.pth", torch_zip(base[:-1], MICRO_STORAGES), "pickle_rejected", "no STOP opcode")
    add("pickle-empty.pth", torch_zip(b"", MICRO_STORAGES), "pickle_rejected", "an empty data.pkl")
    add("pickle-underflow.pth", torch_zip(P().op(b"a").stop()), "pickle_rejected", "APPEND on an empty stack")
    add("pickle-memo-unset.pth", torch_zip(P().op(b"h", 9).stop()), "pickle_rejected", "memo key never set")
    add("pickle-proto-9.pth", torch_zip(b"\x80\x09N."), "pickle_rejected", "unknown protocol")
    add("pickle-huge-string.pth", torch_zip(b"\x80\x02X\xff\xff\xff\xff" + b"x" * 16 + b"."), "pickle_rejected", "4 GB string length")
    add("pickle-long1-9.pth", torch_zip(b"\x80\x02\x8a\x09" + b"\x01" * 9 + b"."), "pickle_rejected", "a 9-byte LONG1")
    add("pickle-bad-utf8.pth", torch_zip(b"\x80\x02X\x02\x00\x00\x00\xff\xfe."), "pickle_rejected", "invalid UTF-8")
    add("pickle-mark-underflow.pth", torch_zip(P().i(1).op(b"(", b"0").stop()), "pickle_rejected", "POP below a MARK")
    add("pickle-odd-setitems.pth", torch_zip(P().op(b"}", b"(").s("a").op(b"u").stop()), "pickle_rejected", "odd SETITEMS")
    add("pickle-reduce-non-global.pth", torch_zip(P().s("x").op(b")", b"R").stop()), "pickle_rejected", "REDUCE on a string")
    add("pickle-call-storage.pth", torch_zip(P().glob("torch", "FloatStorage").op(b")", b"R").stop()), "pickle_rejected", "calling a storage type")
    add("pickle-ordereddict-args.pth", torch_zip(P().glob("collections", "OrderedDict").op(b"]").op(0x85, b"R").stop()), "pickle_rejected", "OrderedDict(list)")
    add("pickle-persid-module.pth", torch_zip(P().s("module").op(0x85, b"Q").stop()), "pickle_rejected", "persistent id that is not a storage")
    hooks = collections.OrderedDict([("weight", collections.OrderedDict([("x", Tensor("0", "HalfStorage", 1, 0, [1], [1], hooks=True))]))])
    add("pickle-tensor-hooks.pth", torch_zip(pkl(hooks), {"0": f16(1.0)}), "pickle_rejected", "tensor backward hooks")
    for i, (key, location) in enumerate([("../x", "cpu"), ("0", "c" * 65), ("a/b", "cpu"), ("0", "cpu/../x"), ("", "cpu")]):
        data = P().op(b"(").s("storage").glob("torch", "HalfStorage").s(key).s(location).i(1).op(b"t", b"Q").stop()
        add(f"pickle-storage-field-{i}.pth", torch_zip(data), "pickle_rejected", f"storage key {key!r} location {location[:12]!r}")
    rank = collections.OrderedDict([("weight", collections.OrderedDict([("x", Tensor("0", "HalfStorage", 4, 0, [2, 2], [2]))]))])
    add("pickle-rank-mismatch.pth", torch_zip(pkl(rank), {"0": f16(1.0) * 4}), "pickle_rejected", "size and stride ranks differ")
    neg = collections.OrderedDict([("weight", collections.OrderedDict([("x", Tensor("0", "HalfStorage", 1, -1, [1], [1]))]))])
    add("pickle-negative-offset.pth", torch_zip(pkl(neg), {"0": f16(1.0)}), "pickle_rejected", "negative storage offset")
    many_dims = collections.OrderedDict([("weight", collections.OrderedDict([("x", Tensor("0", "HalfStorage", 1, 0, [1] * 20, [1] * 20))]))])
    add("pickle-20-dims.pth", torch_zip(pkl(many_dims), {"0": f16(1.0)}), "pickle_rejected", "a 20-dimensional tensor")
    # Resource bombs: deep nesting (no recursion), a memo blow-up (no copies), a copy bomb.
    deep = P()
    for _ in range(20_000):
        deep.op(b"]")
    for _ in range(19_999):
        deep.op(b"a")
    add("bomb-deep-nesting.pth", torch_zip(deep.stop()), "not_a_checkpoint", "20,000 nested lists")
    memo = P().op(b"]", b"q", 0)
    for _ in range(64):
        memo.i(7).op(b"a")
    for _ in range(60):
        memo.op(b"0", b"]", b"h", 0, b"a", b"h", 0, b"a", b"q", 0)
    add("bomb-memo.pth", torch_zip(memo.stop()), "not_a_checkpoint", "2^60 leaves if references were copied")
    text = "y" * (64 << 10)
    copies = P().glob("_codecs", "encode").op(b"q", 0, b"0").s(text).s("latin1").op(0x86, b"q", 1, b"0", b"]")
    for _ in range(5000):
        copies.op(b"h", 0, b"h", 1, b"R", b"a")
    add("bomb-codecs-copies.pth", torch_zip(copies.stop()), "pickle_rejected", "5,000 x 64 KB bytes copies (320 MB)")
    shared = P().s("k" * (64 << 10)).op(b"q", 0, b"0", b"]")
    for _ in range(8000):
        shared.op(b"}", b"h", 0, b"h", 0, b"s", b"a")
    add("bomb-shared-keys.pth", torch_zip(shared.stop()), "not_a_checkpoint", "one 64 KB key shared by 8,000 dicts (500 MB if copied)")


def torch_archive_attacks():
    base = micro()
    add("torch-legacy-proto2.pth", b"\x80\x02\x8a\x0al\xfc\x9cF\xf9 j\xa8P\x19.\x80\x02M\xe9\x03.", "legacy_format", "PyTorch < 1.6 serialization")
    add("torch-legacy-proto4.pth", b"\x80\x04\x95\x10\x00\x00\x00\x00\x00\x00\x00N.", "legacy_format", "a bare protocol-4 pickle")
    add("torch-torchscript.pth", zip_bytes([entry("m/constants.pkl", b"\x80\x02N."), entry("m/code/__torch__.py", b"def f(): pass"), entry("m/data.pkl", b"\x80\x02N.")]), "torchscript", "a TorchScript module")
    add("torch-no-data-pkl.pth", zip_bytes([entry("archive/data/0", b"\0\0"), entry("archive/version", b"3\n")]), "not_a_checkpoint", "storages without data.pkl")
    add("torch-two-data-pkl.pth", zip_bytes([entry("a/data.pkl", pkl({})), entry("b/data.pkl", pkl({}))]), "not_a_checkpoint", "two data.pkl")
    add("torch-deep-data-pkl.pth", zip_bytes([entry("a/b/data.pkl", pkl({}))]), "not_a_checkpoint", "data.pkl two folders deep")
    add("torch-big-endian.pth", micro(byteorder=b"big"), "tensor_invalid", "big-endian byteorder")
    add("torch-root-list.pth", torch_zip(pkl([1, 2, 3])), "not_a_checkpoint", "the root is a list")
    add("torch-root-none.pth", torch_zip(pkl(None)), "not_a_checkpoint", "the root is None")
    add("torch-no-weight.pth", torch_zip(pkl({"config": TINY_CONFIG})), "not_a_checkpoint", "no weight entry")
    add("torch-weight-list.pth", torch_zip(pkl({"weight": [1, 2]})), "not_a_checkpoint", "weight is a list")
    add("torch-weight-int.pth", torch_zip(pkl({"weight": {"enc_p.emb_phone.weight": 5}})), "tensor_invalid", "a weight that is an int")
    long_key = collections.OrderedDict([("x" * 600, Tensor("0", "HalfStorage", 1, 0, [1], [1]))])
    add("torch-long-weight-name.pth", micro(weights=long_key), "tensor_invalid", "a 600-byte weight name")
    dup = collections.OrderedDict([
        ("a.parametrizations.weight.original0", Tensor("0", "HalfStorage", 1, 0, [1], [1])),
        ("a.weight_g", Tensor("1", "HalfStorage", 1, 0, [1], [1])),
    ])
    add("torch-duplicate-after-normalising.pth", micro(weights=dup), "tensor_invalid", "a parametrization key and its legacy twin")
    add("torch-training-discriminator.pth", torch_zip(pkl({"model": {"disc.w": Tensor("0", "HalfStorage", 1, 0, [1], [1])}, "iteration": 3}), {"0": f16(1.0)}), "not_a_checkpoint", "a discriminator training checkpoint")
    add("torch-training-micro.pth", torch_zip(pkl({"model": micro_weights(), "iteration": 3, "optimizer": {}}), MICRO_STORAGES), "unsupported_model", "a generator training checkpoint missing most weights")
    mod = collections.OrderedDict((f"module.{k}", v) for k, v in micro_weights().items())
    add("torch-state-dict-micro.pth", torch_zip(pkl(mod), MICRO_STORAGES), "unsupported_model", "a bare state dict missing most weights")
    mixed = micro_weights()
    mixed["step"] = 5
    add("torch-state-dict-mixed.pth", torch_zip(pkl(mixed), MICRO_STORAGES), "not_a_checkpoint", "a dict of tensors and an int")
    _ = base


def tensor_and_storage_cases():
    def with_phone(t, storages=None):
        w = micro_weights()
        w["enc_p.emb_phone.weight"] = t
        return micro(weights=w, storages=storages)

    add("tensor-view-outside.pth", with_phone(Tensor("0", "HalfStorage", 1, 3, [8, 768], [0, 0])), "tensor_invalid", "view starts past its storage")
    add("tensor-stride-overflow.pth", with_phone(Tensor("0", "HalfStorage", 1, 0, [8, 768], [1 << 62, 1])), "tensor_invalid", "strides that overflow")
    add("tensor-short-storage.pth", with_phone(Tensor("0", "HalfStorage", 2, 0, [8, 768], [0, 0])), "tensor_invalid", "storage shorter than declared")
    add("tensor-long-storage.pth", with_phone(Tensor("0", "HalfStorage", 1, 0, [8, 768], [0, 0]), dict(MICRO_STORAGES, **{"0": f16(1.0) * 4})), "archive_rejected", "storage longer than declared")
    add("tensor-missing-storage.pth", with_phone(Tensor("9", "HalfStorage", 1, 0, [8, 768], [0, 0])), "not_a_checkpoint", "storage entry missing from the archive")
    add("tensor-huge-numel.pth", with_phone(Tensor("0", "HalfStorage", 1 << 40, 0, [8, 768], [0, 0])), "tensor_invalid", "a 2 TB storage")
    add("tensor-long-dtype.pth", with_phone(Tensor("0", "LongStorage", 1, 0, [8, 768], [0, 0]), dict(MICRO_STORAGES, **{"0": b"\0" * 8})), "tensor_invalid", "int64 weights")
    add("tensor-int-dtype.pth", with_phone(Tensor("0", "IntStorage", 1, 0, [8, 768], [0, 0]), dict(MICRO_STORAGES, **{"0": b"\0" * 4})), "tensor_invalid", "int32 weights")
    add("tensor-two-dtypes.pth", with_phone(Tensor("1", "FloatStorage", 1, 0, [8, 768], [0, 0])), "tensor_invalid", "one storage declared as fp16 and fp32")
    add("tensor-bf16.pth", with_phone(Tensor("0", "BFloat16Storage", 1, 0, [8, 768], [0, 0])), "missing_tensor", "bf16 weights read, then a weight is missing")
    add("tensor-fp32.pth", with_phone(Tensor("0", "FloatStorage", 1, 0, [8, 768], [0, 0]), dict(MICRO_STORAGES, **{"0": struct.pack("<f", 0.5)})), "missing_tensor", "fp32 weights read, then a weight is missing")
    add("tensor-nan.pth", micro(storages=dict(MICRO_STORAGES, **{"0": b"\x00\x7e"})), "non_finite", "a NaN weight")
    add("tensor-inf.pth", micro(storages=dict(MICRO_STORAGES, **{"0": b"\x00\x7c"})), "non_finite", "an infinite weight")


def semantic_cases():
    add("semantic-missing-tensor.pth", micro(), "missing_tensor", "only the first weights of the template")
    w = micro_weights()
    w["enc_p.emb_phone.bias"] = Tensor("0", "HalfStorage", 1, 0, [9], [0])
    add("semantic-shape-mismatch.pth", micro(weights=w), "shape_mismatch", "a bias of 9 values where the template has 8")
    w = micro_weights()
    w["dec.refine.extra.weight"] = Tensor("0", "HalfStorage", 1, 0, [4], [0])
    add("semantic-extra-tensor.pth", micro(weights=w), "unsupported_model", "a tensor the template does not model")
    for name, shape, expect, note in [
        ("semantic-content-300.pth", [8, 300], "unsupported_model", "300-channel content input"),
        ("semantic-v1.pth", [8, 256], "no_template", "a v1 voice (no v1 tiny template)"),
    ]:
        w = micro_weights()
        w["enc_p.emb_phone.weight"] = Tensor("0", "HalfStorage", 1, 0, shape, [0, 0])
        add(name, micro(weights=w), expect, note)
    w = micro_weights()
    del w["enc_p.emb_pitch.weight"]
    add("semantic-nof0.pth", micro(weights=w), "no_template", "a voice without pitch guidance (no nof0 tiny template)")
    w = micro_weights()
    del w["emb_g.weight"]
    add("semantic-no-speakers.pth", micro(weights=w), "not_a_checkpoint", "no speaker table")
    w = micro_weights()
    w["emb_g.weight"] = Tensor("2", "HalfStorage", 1, 0, [3, 9], [0, 0])
    add("semantic-speaker-width.pth", micro(weights=w), "unsupported_model", "9-wide speaker table")
    w = micro_weights()
    w["emb_g.weight"] = Tensor("2", "HalfStorage", 1, 0, [0, 8], [0, 0])
    add("semantic-zero-speakers.pth", micro(weights=w), "not_a_checkpoint", "an empty speaker table")
    w = micro_weights()
    del w["enc_p.emb_phone.weight"]
    add("semantic-no-content.pth", micro(weights=w), "not_a_checkpoint", "no enc_p.emb_phone.weight")

    def meta(**changes):
        m = micro_meta()
        for k, v in changes.items():
            if v is None:
                m.pop(k, None)
            else:
                m[k] = v
        return m

    cfg12 = list(TINY_CONFIG)
    cfg12[12] = [2, 4]
    cfg_sr = list(TINY_CONFIG)
    cfg_sr[17] = 48000
    cfg_bad_sr = list(TINY_CONFIG)
    cfg_bad_sr[17] = "7"
    cfg_heads = list(TINY_CONFIG)
    cfg_heads[5] = 4
    for name, m, expect, note in [
        ("meta-no-config.pth", meta(config=None), "not_a_checkpoint", "an export without its config"),
        ("meta-config-17.pth", meta(config=TINY_CONFIG[:17]), "unsupported_model", "a 17-entry config"),
        ("meta-config-string.pth", meta(config="1025,32"), "unsupported_model", "a config that is a string"),
        ("meta-config-upsample.pth", meta(config=cfg12), "unsupported_model", "non-standard upsample rates"),
        ("meta-config-heads.pth", meta(config=cfg_heads), "unsupported_model", "non-standard attention heads"),
        ("meta-config-48k.pth", meta(config=cfg_sr), "no_template", "a 48 kHz voice (no tiny template)"),
        ("meta-config-bad-sr.pth", meta(config=cfg_bad_sr, sr=None), "unsupported_model", "no usable sample rate"),
        ("meta-vocoder-unknown.pth", meta(vocoder="WaveNet"), "unsupported_model", "an unknown vocoder"),
        ("meta-vocoder-refinegan.pth", meta(vocoder="RefineGAN"), "no_template", "RefineGAN declared (no tiny RefineGAN template)"),
        ("meta-embedder-spin-v1.pth", meta(embedder_model="spin"), "missing_tensor", "Applio's SPIN v1 embedder accepted (vc/embedder-spin, P3-I0)"),
        ("meta-embedder-custom.pth", meta(embedder_model="custom"), "unsupported_model", "a custom embedder"),
        ("meta-embedder-unknown.pth", meta(embedder_model="wavlm-large"), "unsupported_model", "an unknown embedder"),
        ("meta-embedder-japanese.pth", meta(embedder_model="japanese-hubert-base"), "missing_tensor", "Japanese HuBERT accepted"),
        ("meta-version-lie.pth", meta(version="v1", f0=0, sr="48k"), "missing_tensor", "declared fields that disagree (warnings only)"),
        ("meta-huge-info.pth", meta(info="i" * 20_000), "missing_tensor", "a 20 KB info string (cut)"),
        ("meta-odd-types.pth", meta(sr=[1, 2], f0="maybe", version=5, author=None), "missing_tensor", "metadata of the wrong types"),
    ]:
        add(name, micro(meta=m), expect, note)


def valid_controls(root, storages):
    weights = root["weight"]
    meta = collections.OrderedDict((k, v) for k, v in root.items() if k != "weight")
    # torch.save(net_g.state_dict()) writes the `_metadata` BUILD; so do these.
    training = collections.OrderedDict([("model", StateDict(weights)), ("iteration", 2333), ("optimizer", {"state": {}, "param_groups": []}), ("learning_rate", 0.0001)])
    add("valid-training-G_2333.pth", torch_zip(pkl(training), storages, prefix="G_2333"), "ok", "a training checkpoint (config-less, torch's state-dict metadata)")
    state = StateDict((f"module.{k}", v) for k, v in weights.items())
    add("valid-state-dict.pth", torch_zip(pkl(state), storages, prefix="state"), "ok", "a bare DataParallel state dict (torch's state-dict metadata)")
    voice = torch_zip(pkl(collections.OrderedDict([("weight", weights)] + list(meta.items()))), storages, prefix="Tiny")
    raw_index = struct.pack("<II", 768, 2) + struct.pack("<f", 0.5) * (768 * 2)
    add(
        "valid-voice-archive.zip",
        zip_bytes([
            entry("Tiny/Tiny.pth", voice, method=8),
            entry("Tiny/added_IVF2_Flat_nprobe_1_Tiny_v2.index", raw_index, method=8),
            entry("Tiny/trained_IVF2_Flat_nprobe_1_Tiny_v2.index", b"\0" * 16),
            entry("Tiny/D_2333.pth", b"not a voice"),
            entry("__MACOSX/Tiny/._Tiny.pth", b"\0" * 64),
        ]),
        "ok",
        "a voice archive with its index, a trained index, a training by-product and macOS metadata",
    )
    add(
        "valid-epochs-archive.zip",
        zip_bytes([
            entry("Tiny/Tiny_100e_2000s.pth", voice, method=8),
            entry("Tiny/Tiny_300e_6000s.pth", voice, method=8),
            entry("Tiny/Tiny_e20_s400.pth", voice, method=8),
            entry("Tiny/added_IVF2_Flat_nprobe_1_Tiny_v2.index", raw_index, method=8),
        ]),
        "ok",
        "one voice saved at several epochs (Applio / RVC names): the latest step is imported",
    )


def mutations(count):
    rng = random.Random(0x5161)
    bases = [micro(), torch_zip(pkl({"model": micro_weights(), "iteration": 1}), MICRO_STORAGES), zip_bytes([entry("v/v.pth", micro(), method=8), entry("v/added.index", b"\0" * 32)])]
    for i in range(count):
        d = bytearray(bases[i % len(bases)])
        kind = rng.randrange(4)
        if kind == 3:
            d = d[: rng.randrange(1, len(d))]
            note = "truncated"
        else:
            region = rng.choice(["anywhere", "tail"])
            for _ in range(1 + rng.randrange(6)):
                if region == "tail":
                    pos = len(d) - 1 - rng.randrange(min(len(d), 400))
                else:
                    pos = rng.randrange(len(d))
                if kind == 0:
                    d[pos] = rng.randrange(256)
                elif kind == 1 and len(d) > 1:
                    del d[pos]
                else:
                    d.insert(pos, rng.randrange(256))
            note = f"{['flipped', 'deleted', 'inserted'][kind]} bytes ({region})"
        add(f"mutated-{i:03d}.{'zip' if i % len(bases) == 2 else 'pth'}", bytes(d), "any", note)


def main():
    root, storages = read_tiny()
    zip_attacks()
    pickle_attacks()
    torch_archive_attacks()
    tensor_and_storage_cases()
    semantic_cases()
    valid_controls(root, storages)
    mutations(TOTAL - len(corpus))
    assert len(corpus) == TOTAL, len(corpus)
    if os.path.isdir(OUT):
        shutil.rmtree(OUT)
    os.makedirs(OUT)
    for c in corpus:
        with open(os.path.join(OUT, c["file"]), "wb") as f:
            f.write(c["_data"])
    manifest = {
        "$comment": "Generated by generate_corpus.py (deterministic). expect: an import error code, ok, or any (no panic / hang). Templates: ../tiny.",
        "templates": "../tiny",
        "files": [{k: v for k, v in c.items() if k != "_data"} for c in corpus],
    }
    with open(os.path.join(HERE, "corpus.json"), "w", encoding="utf-8", newline="\n") as f:
        json.dump(manifest, f, indent=1)
        f.write("\n")
    total = sum(len(c["_data"]) for c in corpus)
    by = collections.Counter(c["expect"] for c in corpus)
    print(f"wrote {len(corpus)} files, {total} bytes; expectations: {dict(sorted(by.items()))}")


if __name__ == "__main__":
    main()
