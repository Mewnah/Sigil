<#
.SYNOPSIS
  Run the ADR 0002 sherpa-onnx probe matrix and save every JSON report.

.DESCRIPTION
  Runs `Sigil.exe --headless engine-probe sherpa` over the models in a models folder laid out
  like the dev cache (%LOCALAPPDATA%\sigil-dev-cache\sherpa-onnx\models\<package>), one probe at a
  time, and writes each report to bench-results\sherpa\<timestamp>\<name>.json (gitignored) plus a
  summary table on stdout. Nothing is downloaded: install the pack with
  `Sigil.exe --headless fetch runtime/sherpa-onnx` (or pass -RuntimeDir) and extract the model
  packages yourself.

  Rows (each skipped when its package folder is missing):
  - SenseVoice int8 (2024-07-17) en.wav / ja.wav at 1, 2 and 4 threads (RTF, cores, idle spin)
  - the same en.wav with Sigil's own ONNX Runtime keeping a GPU busy at 70% duty (-CudaPreload
    lets a PC without the NVIDIA runtime pack use CUDA: pass the CUDA 13 bin\x64 and a cuDNN 9
    folder)
  - Kokoro int8 multi-lang v1.0: English (sid 0) and Japanese (lang=ja, sid 37) at 2 threads
  - Nemotron 3.5 streaming int8 (160 ms) en.wav at 1 thread, when present

  Close games, browsers and builds first: the probe measures wall-clock RTF.

.EXAMPLE
  pwsh scripts/runtime-packs/sherpa/probe-matrix.ps1 -Exe src-tauri\target\release\sigil.exe
#>
[CmdletBinding()]
param(
  [string]$Exe = "src-tauri\target\release\sigil.exe",
  [string]$Models = (Join-Path $env:LOCALAPPDATA "sigil-dev-cache\sherpa-onnx\models"),
  [string]$RuntimeDir = "",
  [string]$CudaPreload = "",
  [int]$Repeat = 5
)
$ErrorActionPreference = "Stop"
$repo = Resolve-Path (Join-Path $PSScriptRoot "..\..\..")
if (-not [IO.Path]::IsPathRooted($Exe)) { $Exe = Join-Path $repo $Exe }
if (-not (Test-Path $Exe)) { throw "$Exe not found (build it first)" }
$out = Join-Path $repo ("bench-results\sherpa\" + (Get-Date -Format "yyyyMMdd-HHmmss"))
New-Item -ItemType Directory -Force $out | Out-Null

$sv = Join-Path $Models "sherpa-onnx-sense-voice-zh-en-ja-ko-yue-int8-2024-07-17"
$kokoro = Join-Path $Models "kokoro-int8-multi-lang-v1_0"
$nemotron = Join-Path $Models "sherpa-onnx-nemotron-3.5-asr-streaming-0.6b-160ms-int8-2026-06-11"

$rows = @()
foreach ($t in 1, 2, 4) {
  foreach ($lang in "en", "ja") {
    $rows += , @("sensevoice-$lang-t$t", $sv, (Join-Path $sv "test_wavs\$lang.wav"), @("threads=$t", "idle_ms=2000"))
  }
}
$load = @("threads=2", "cuda_load=0.7")
if ($CudaPreload) { $load += "cuda_preload=$CudaPreload" }
$rows += , @("sensevoice-en-t2-gpu-load", $sv, (Join-Path $sv "test_wavs\en.wav"), $load)
$rows += , @("kokoro-en-t2", $kokoro, $null, @("threads=2", "idle_ms=2000"))
$rows += , @("kokoro-ja-t2", $kokoro, $null, @("threads=2", "lang=ja", "sid=37", "text=こんにちは、シギルです。今日はいい天気ですね。"))
$rows += , @("nemotron-en-t1", $nemotron, (Join-Path $sv "test_wavs\en.wav"), @("threads=1", "chunk_ms=160", "idle_ms=2000"))

$summary = @()
foreach ($r in $rows) {
  $name, $model, $wav, $extra = $r
  if (-not (Test-Path $model)) { Write-Host "skip $name ($model missing)"; continue }
  $argv = @("--headless", "engine-probe", "sherpa", $model)
  if ($wav) { $argv += $wav }
  if ($RuntimeDir) { $argv += @("--runtime-dir", $RuntimeDir) }
  $file = Join-Path $out "$name.json"
  $argv += @("--out", $file, "--", "repeat=$Repeat") + $extra
  Write-Host "probe $name"
  & $Exe @argv | Out-Null
  if ($LASTEXITCODE -ne 0) { Write-Warning "$name failed ($LASTEXITCODE)"; continue }
  $j = Get-Content -Raw $file | ConvertFrom-Json
  $summary += [pscustomobject]@{
    row         = $name
    rtf_median  = $j.decode.rtf_median
    cores       = ($j.decode.runs | ForEach-Object { $_.cpu_cores } | Measure-Object -Average).Average
    idle_ms_s   = $j.idle_cpu_ms_per_s_after_decode
    model_ms    = $j.model_load_ms
    rss_mb      = $j.memory_mb.after_decode.rss_mb
    loaded_rtf  = $j.contention.decode.rtf_median
    same_output = $j.contention.output_matches_unloaded
  }
}
$summary | Format-Table -AutoSize
$summary | ConvertTo-Json | Set-Content (Join-Path $out "summary.json")
Write-Host "Reports: $out"
