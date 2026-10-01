# SPDX-License-Identifier: AGPL-3.0-or-later
"""Applio model code (IAHispano/Applio, MIT): the upstream reference for Sigil's RefineGAN and
MRF HiFi-GAN templates, which mainline RVC does not have.

Every module in this package except this file and ``generators/__init__.py`` is vendored unchanged
(except for package-relative imports) from IAHispano/Applio; see ``LICENSE`` in this folder and the
header of each file for the pinned upstream commit and the upstream file's sha256.

Sigil's own ports of these generators (``rvc_arch/vocoders.py``) are what the exporter and the
default parity run use. This package is only imported by the maintainer cross-check
``parity_test.py --upstream-applio``, which compares those ports with Applio's
``Synthesizer.infer`` on the same weights and noise (a manual step, see the README).
"""
