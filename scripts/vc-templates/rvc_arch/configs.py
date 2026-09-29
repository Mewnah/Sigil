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
