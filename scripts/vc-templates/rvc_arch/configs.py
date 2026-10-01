# SPDX-License-Identifier: AGPL-3.0-or-later
"""Template configurations: the RVC ``config`` list each Sigil template is exported for.

The list order is RVC's checkpoint ``config`` (``SynthesizerTrnMs768NSFsid(*cpt["config"])``):

    0 spec_channels  1 segment_size  2 inter_channels  3 hidden_channels  4 filter_channels
    5 n_heads  6 n_layers  7 kernel_size  8 p_dropout  9 resblock  10 resblock_kernel_sizes
    11 resblock_dilation_sizes  12 upsample_rates  13 upsample_initial_channel
    14 upsample_kernel_sizes  15 spk_embed_dim  16 gin_channels  17 sr

Only the *architecture* fields (``ARCH_FIELDS``) must match a checkpoint for a template to apply:
``spec_channels`` only sizes the training-time posterior encoder, ``segment_size`` and ``p_dropout``
are training-only, and ``spk_embed_dim`` sizes the speaker table, which Sigil keeps outside the
graphs (the speaker vector ``g`` is an input), so any speaker count works with one template.
"""

import math

CONFIG_FIELDS = (
    "spec_channels",
    "segment_size",
    "inter_channels",
    "hidden_channels",
    "filter_channels",
    "n_heads",
    "n_layers",
    "kernel_size",
    "p_dropout",
    "resblock",
    "resblock_kernel_sizes",
    "resblock_dilation_sizes",
    "upsample_rates",
    "upsample_initial_channel",
    "upsample_kernel_sizes",
    "spk_embed_dim",
    "gin_channels",
    "sr",
)

# Indices of CONFIG_FIELDS that define the exported graphs (see the module docs).
ARCH_FIELDS = (2, 3, 4, 5, 6, 7, 9, 10, 11, 12, 13, 14, 16, 17)

TEMPLATES = {
    # RVC v2, 40 kHz, pitch-guided, HiFi-GAN NSF vocoder (RVC WebUI / Applio default v2 40k).
    "v2-40k-f0-hifigan": {
        "version": "v2",
        "f0": True,
        "vocoder": "hifigan",
        "emb_channels": 768,
        "config": [
            1025, 32, 192, 192, 768, 2, 6, 3, 0, "1", [3, 7, 11],
            [[1, 3, 5], [1, 3, 5], [1, 3, 5]], [10, 10, 2, 2], 512, [16, 16, 4, 4], 109, 256, 40000,
        ],
    },
    # Tiny *test* configuration (same code paths, a few thousand parameters) for the Rust unit tests
    # and CI: exported only into src-tauri/test-fixtures/vc/tiny, never bundled. Its id follows the
    # importer's naming (a 16 kHz v2 voice) because the importer finds templates by that id.
    "v2-16k-f0-hifigan": {
        "test_only": True,
        "version": "v2",
        "f0": True,
        "vocoder": "hifigan",
        "emb_channels": 768,
        "config": [
            1025, 32, 8, 8, 16, 2, 2, 3, 0, "1", [3, 5], [[1, 3, 5], [1, 3, 5]], [4, 2], 16, [8, 4], 3, 8, 16000,
        ],
    },
}

# Tiny *test* configurations of the layout-version-2 kinds (exported only into
# src-tauri/test-fixtures/vc/tiny/v2 by `parity_test.py --write-fixture-v2`). RefineGAN's channel plan
# is fixed by its architecture (the down path's 16 * 2^stages channels must equal half the initial
# channels, and its speaker conditioning takes 256), so its tiny config keeps 128 / 256 there.
TINY_V2 = {
    "v2-16k-nof0-hifigan": {"test_only": True, "version": "v2", "f0": False, "vocoder": "hifigan", "emb_channels": 768,
                            "config": [1025, 32, 8, 8, 16, 2, 2, 3, 0, "1", [3, 5], [[1, 3, 5], [1, 3, 5]], [4, 2], 16, [8, 4], 3, 8, 16000]},
    "v2-16k-f0-mrf-hifigan": {"test_only": True, "version": "v2", "f0": True, "vocoder": "mrf-hifigan", "emb_channels": 768,
                              "config": [1025, 32, 8, 8, 16, 2, 2, 3, 0, "1", [3, 5], [[1, 3, 5], [1, 3, 5]], [4, 2], 16, [8, 4], 3, 8, 16000]},
    "v2-16k-f0-refinegan": {"test_only": True, "version": "v2", "f0": True, "vocoder": "refinegan", "emb_channels": 768, "adain_pool": 16384,
                            "config": [1025, 32, 8, 8, 16, 2, 2, 3, 0, "1", [3, 5], [[1, 3, 5], [1, 3, 5]], [4, 2], 128, [8, 4], 3, 256, 16000]},
}
TEMPLATES.update(TINY_V2)

# ---------------------------------------------------------------------------------------------
# The P3 template matrix
# ---------------------------------------------------------------------------------------------
#
# The decoder architecture per (version, sample rate) is what the trainers write into `config`:
# - RVC WebUI (configs/v1/*.json and configs/v2/*.json at commit 81eed5e8; a v2 40 kHz voice is
#   trained with v1/40k.json);
# - Applio (rvc/configs/<sr>.json at commit c7665ac9; v2 only; its 32 / 40 / 48 kHz configs equal
#   RVC's v2 ones, and it adds 24 kHz).
# Everything else (channels, heads, layers, resblocks, gin) is the same in every config.
_RESBLOCK_KERNELS = [3, 7, 11]
_RESBLOCK_DILATIONS = [[1, 3, 5], [1, 3, 5], [1, 3, 5]]
UPSAMPLING = {
    # (version, sample rate): (spec_channels, upsample_rates, upsample_kernel_sizes)
    ("v1", 32000): (513, [10, 4, 2, 2, 2], [16, 16, 4, 4, 4]),
    ("v1", 40000): (1025, [10, 10, 2, 2], [16, 16, 4, 4]),
    ("v1", 48000): (1025, [10, 6, 2, 2, 2], [16, 16, 4, 4, 4]),
    ("v2", 24000): (513, [10, 6, 2, 2], [20, 12, 4, 4]),
    ("v2", 32000): (513, [10, 8, 2, 2], [20, 16, 4, 4]),
    ("v2", 40000): (1025, [10, 10, 2, 2], [16, 16, 4, 4]),
    ("v2", 48000): (1025, [12, 10, 2, 2], [24, 20, 4, 4]),
}
VOCODERS = ("hifigan", "refinegan", "mrf-hifigan")
# Matrix cells no trainer produces, so no template exists (the importer answers `no_template`,
# naming the reason from `matrix.json`).
UNSUPPORTED = {
    "nof0-refinegan": "Applio trains RefineGAN only with pitch guidance (its Synthesizer builds no f0-less RefineGAN).",
    "nof0-mrf-hifigan": "Applio trains MRF HiFi-GAN only with pitch guidance (its Synthesizer builds no f0-less MRF generator).",
    "v1-refinegan": "RefineGAN voices come from Applio, which trains RVC v2 (768-channel ContentVec) models only.",
    "v1-mrf-hifigan": "MRF HiFi-GAN voices come from Applio, which trains RVC v2 (768-channel ContentVec) models only.",
    "v1-24k": "No trainer has an RVC v1 24 kHz configuration (24 kHz is Applio's, v2 only).",
}


def matrix_config(version, sample_rate):
    spec, rates, kernels = UPSAMPLING[(version, sample_rate)]
    return [
        spec, 32, 192, 192, 768, 2, 6, 3, 0, "1", list(_RESBLOCK_KERNELS), [list(d) for d in _RESBLOCK_DILATIONS],
        list(rates), 512, list(kernels), 109, 256, sample_rate,
    ]


def template_id(version, sample_rate, f0, vocoder):
    return f"{version}-{sample_rate // 1000}k-{'f0' if f0 else 'nof0'}-{vocoder}"


def unsupported_reason(version, sample_rate, f0, vocoder):
    """Why a matrix cell has no template, or None when it has one."""
    if (version, sample_rate) not in UPSAMPLING:
        return UNSUPPORTED["v1-24k"] if version == "v1" and sample_rate == 24000 else f"no {version} {sample_rate} Hz configuration"
    if vocoder != "hifigan" and not f0:
        return UNSUPPORTED[f"nof0-{vocoder}"]
    if vocoder != "hifigan" and version == "v1":
        return UNSUPPORTED[f"v1-{vocoder}"]
    return None


def matrix_cells():
    """[(template id, version, sample rate, f0, vocoder, unsupported reason or None)] for the whole
    v1/v2 x 24/32/40/48 kHz x f0/nof0 x vocoder matrix."""
    cells = []
    for version in ("v1", "v2"):
        for sr in (24000, 32000, 40000, 48000):
            for f0 in (True, False):
                for vocoder in VOCODERS:
                    cells.append((template_id(version, sr, f0, vocoder), version, sr, f0, vocoder, unsupported_reason(version, sr, f0, vocoder)))
    return cells


for _tid, _version, _sr, _f0, _vocoder, _why in matrix_cells():
    if _why is None and _tid not in TEMPLATES:
        TEMPLATES[_tid] = {
            "version": _version,
            "f0": _f0,
            "vocoder": _vocoder,
            "emb_channels": 768 if _version == "v2" else 256,
            "config": matrix_config(_version, _sr),
        }
assert TEMPLATES["v2-40k-f0-hifigan"]["config"] == matrix_config("v2", 40000), "the P1 template left the matrix"
MATRIX = sorted(t for t, spec in TEMPLATES.items() if not spec.get("test_only"))


def config_dict(config):
    """The RVC config list as a name -> value dict."""
    if len(config) != len(CONFIG_FIELDS):
        raise ValueError(f"RVC config has {len(config)} entries, expected {len(CONFIG_FIELDS)}")
    return dict(zip(CONFIG_FIELDS, config))


def sample_rate(config):
    sr = config[17]
    if isinstance(sr, str):
        sr = int(sr.lower().rstrip("k")) * 1000
    return int(sr)


def upp(config):
    """Output samples per 10 ms content frame (the NSF upsampling factor)."""
    return int(math.prod(config[12]))


def arch_key(config):
    """The architecture fields of a config, normalised for comparison."""
    out = []
    for i in ARCH_FIELDS:
        v = config[i]
        if i == 17:
            v = sample_rate(config)
        out.append(v)
    return out
