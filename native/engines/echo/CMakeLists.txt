# SPDX-License-Identifier: AGPL-3.0-only
# sigil_echo.dll: the sigil_engine ABI v1 test engine (no model, no GPU). Version and variants are in
# engine.json (read here and by scripts/native/build-runtime-packs.ps1).
file(READ "${CMAKE_CURRENT_SOURCE_DIR}/engine.json" _echo_json)
string(JSON _echo_version GET "${_echo_json}" version)

sigil_add_engine(echo
  VERSION "${_echo_version}"
  SOURCES echo.cpp
)
