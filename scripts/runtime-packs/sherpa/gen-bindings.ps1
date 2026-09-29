<#
.SYNOPSIS
  Regenerate src-tauri/src/services/speech_rt/sherpa/bindings.rs from sherpa-onnx's c-api.h.

.DESCRIPTION
  Maintainer tool (never part of the app build). Downloads the pinned header once into the dev
  cache (%LOCALAPPDATA%\sigil-dev-cache\sherpa-onnx), verifies its sha256, and runs the bindgen
  generator in ./bindgen (offline: bindgen 0.72.1 is already in the cargo registry because the app's
  dependency tree uses it). Needs LLVM (libclang.dll); pass -LibClang if it is not in
  C:\Program Files\LLVM\bin.

  Upgrading sherpa-onnx: change -Version and -HeaderSha256 (the header's sha256 at that tag), run
  this script, bump EXPECTED_VERSION in speech_rt/sherpa/lib.rs, update the struct-size table in
  speech_rt/sherpa/tests.rs from the regenerated layout asserts, re-pin
  src-tauri/model-catalog/runtime-sherpa.json (./pin-pack.ps1), and re-run the probe matrix in
  docs/adr/0002-sherpa-onnx.md.

.EXAMPLE
  pwsh scripts/runtime-packs/sherpa/gen-bindings.ps1
#>
[CmdletBinding()]
param(
  [string]$Version = "1.13.8",
  [string]$HeaderSha256 = "2a1b95084be8fd1deb3228fcad2fd3f7f0258b64582f7402281ec174c7b7f4ce",
  [string]$LibClang = "C:\Program Files\LLVM\bin",
  [string]$Header = ""
)
$ErrorActionPreference = "Stop"
$repo = Resolve-Path (Join-Path $PSScriptRoot "..\..\..")
$out = Join-Path $repo "src-tauri\src\services\speech_rt\sherpa\bindings.rs"
$cache = Join-Path $env:LOCALAPPDATA "sigil-dev-cache\sherpa-onnx"
New-Item -ItemType Directory -Force $cache | Out-Null

if (-not $Header) {
  $Header = Join-Path $cache "c-api-v$Version.h"
  if (-not (Test-Path $Header)) {
    $url = "https://raw.githubusercontent.com/k2-fsa/sherpa-onnx/v$Version/sherpa-onnx/c-api/c-api.h"
    Write-Host "Downloading $url"
    Invoke-WebRequest -Uri $url -OutFile "$Header.partial" -UseBasicParsing
    Move-Item -Force "$Header.partial" $Header
  }
}
$actual = (Get-FileHash -Algorithm SHA256 $Header).Hash.ToLowerInvariant()
if ($actual -ne $HeaderSha256.ToLowerInvariant()) {
  throw "c-api.h sha256 is $actual, expected $HeaderSha256 (wrong version or a corrupted download: delete $Header)"
}
if (-not (Test-Path (Join-Path $LibClang "libclang.dll"))) {
  throw "libclang.dll not found in $LibClang (install LLVM or pass -LibClang)"
}

$env:LIBCLANG_PATH = $LibClang
$env:CARGO_TARGET_DIR = Join-Path $env:LOCALAPPDATA "sigil-target\sherpa-bindgen"
Push-Location (Join-Path $PSScriptRoot "bindgen")
try {
  cargo run --offline --quiet -- $Header $out $Version $HeaderSha256
  if ($LASTEXITCODE -ne 0) { throw "the generator failed ($LASTEXITCODE)" }
} finally {
  Pop-Location
}
Write-Host "Regenerated $out. Now run: cargo test --manifest-path src-tauri/Cargo.toml speech_rt::sherpa"
