# SPDX-License-Identifier: AGPL-3.0-or-later
"""Unit tests of scripts/speech-bench and tools/listening-test (P7).

Run with the P7 venv (scripts/speech-bench/setup-venv.ps1):

    & "$env:LOCALAPPDATA\\sigil-dev-cache\\p7\\venv\\Scripts\\python.exe" -m unittest discover -s scripts/speech-bench -p "test_*.py"

No test touches the network, the GPU or the real P7 data; corpus tests build tiny synthetic inputs
in a temporary folder. The page test drives the two HTML pages with the repo's Playwright
(Chromium, headless, file://) and is skipped when Node or the browser is missing.
"""
