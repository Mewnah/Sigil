# SPDX-License-Identifier: AGPL-3.0-or-later
"""RVC model code for Sigil's voice-changer template pipeline.

- ``models``, ``modules``, ``attentions``, ``commons``, ``transforms``: vendored unchanged (except
  for package-relative imports) from RVC-Project/Retrieval-based-Voice-Conversion-WebUI, MIT; see
  ``LICENSE`` in this folder and the header of each file for the pinned upstream commit.
- ``split``: Sigil's split-graph wrappers (enc / flowdec / knn) exported by ``export_rvc.py``.
- ``configs``: the template configurations Sigil exports (the RVC ``config`` list per template).
"""
