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

  -Catalog adds the P2 rows: every STT model of model-catalog/stt-sherpa.json, probed by catalog id
  (Nemotron 3.5 560 / 160 ms per language, Parakeet Ultra and v3 with and without hotwords, Qwen3-ASR
  0.6B with a language, auto and context, SenseVoice, Moonshine base, Whisper-base language ID). The
  models come from the model store that SIGIL_APP_DATA_DIR points at (install them with
  `Sigil.exe --headless fetch <id>`); the clips from -Clips (en.wav, ja.wav, ja-nemotron.wav,
  ja-qwen.wav, zh.wav, yue.wav, ko.wav, ru.wav, de.wav, es.wav, fr.wav; P2-B copied them from the
  packages' test_wavs). A row whose clip is missing is skipped; a model that is not installed fails
  its row with a warning.

  Close games, browsers and builds first: the probe measures wall-clock RTF.

.EXAMPLE
  pwsh scripts/runtime-packs/sherpa/probe-matrix.ps1 -Exe src-tauri\target\release\sigil.exe

.EXAMPLE
  $env:SIGIL_APP_DATA_DIR = "$env:LOCALAPPDATA\sigil-dev-cache\appdata-p2b"
  pwsh scripts/runtime-packs/sherpa/probe-matrix.ps1 -Exe src-tauri\target\release\sigil.exe -Catalog -Repeat 3
#>
[CmdletBinding()]
param(
  [string]$Exe = "src-tauri\target\release\sigil.exe",
  [string]$Models = (Join-Path $env:LOCALAPPDATA "sigil-dev-cache\sherpa-onnx\models"),
  [string]$Clips = (Join-Path $env:LOCALAPPDATA "sigil-dev-cache\sherpa-onnx\clips"),
  [string]$RuntimeDir = "",
  [string]$CudaPreload = "",
  [int]$Repeat = 5,
  [switch]$Catalog
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

if ($Catalog) {
  # P2 rows by catalog id (the model store resolves them; see -Catalog above).
  $clip = { param($f) Join-Path $Clips $f }
  foreach ($ms in 560, 160) {
    foreach ($l in @(@("en", "en.wav"), @("ja", "ja-nemotron.wav"), @("zh", "zh.wav"), @("ru", "ru.wav"), @("de", "de.wav"))) {
      $rows += , @("nemotron$ms-$($l[0])-t1", "stt/nemotron-3.5-streaming-${ms}ms-int8", (& $clip $l[1]), @("lang=$($l[0])", "chunk_ms=$ms", "threads=1", "idle_ms=1000"))
    }
  }
  $rows += , @("nemotron560-en-t2", "stt/nemotron-3.5-streaming-560ms-int8", (& $clip "en.wav"), @("lang=en", "chunk_ms=560", "threads=2", "idle_ms=1000"))
  foreach ($id in "parakeet-ultra-int8", "parakeet-tdt-0.6b-v3-int8") {
    foreach ($l in "en", "de", "es", "fr", "ru") {
      $rows += , @("$id-$l-t2", "stt/$id", (& $clip "$l.wav"), @("threads=2", "idle_ms=1000"))
    }
    $rows += , @("$id-ru-t2-hotwords", "stt/$id", (& $clip "ru.wav"), @("threads=2", "beam=1", "hotwords=вольера;зоопарке", "idle_ms=1000"))
  }
  foreach ($l in @(@("ja", "ja-qwen.wav"), @("zh", "zh.wav"), @("ru", "ru.wav"), @("en", "en.wav"), @("ko", "ko.wav"))) {
    $rows += , @("qwen3-$($l[0])-t2", "stt/qwen3-asr-0.6b-int8", (& $clip $l[1]), @("lang=$($l[0])", "threads=2", "idle_ms=1000"))
  }
  $rows += , @("qwen3-ru-t2-context", "stt/qwen3-asr-0.6b-int8", (& $clip "ru.wav"), @("lang=ru", "threads=2", "hotwords=зоопарке;вольера", "idle_ms=1000"))
  $rows += , @("qwen3-ja-t2-auto", "stt/qwen3-asr-0.6b-int8", (& $clip "ja-qwen.wav"), @("threads=2", "idle_ms=1000"))
  foreach ($l in "zh", "yue", "ja", "en", "ko") {
    $rows += , @("sensevoice-$l-t2", "stt/sense-voice-small-int8", (& $clip "$l.wav"), @("threads=2", "idle_ms=1000"))
  }
  foreach ($t in 2, 1) {
    $rows += , @("moonshine-en-t$t", "stt/moonshine-base-en-int8", (& $clip "en.wav"), @("threads=$t", "idle_ms=1000"))
  }
  foreach ($l in "en", "ja", "zh", "ru", "de") {
    $rows += , @("lid-$l-t2", "stt/lid-whisper-base-int8", (& $clip "$l.wav"), @("threads=2", "idle_ms=1000"))
  }
}

$summary = @()
foreach ($r in $rows) {
  $name, $model, $wav, $extra = $r
  $isCatalogId = $model -match '^[a-z0-9]+/[A-Za-z0-9._-]+$' -and -not (Test-Path $model)
  if ($isCatalogId) {
    if ($wav -and -not (Test-Path $wav)) { Write-Host "skip $name ($wav missing)"; continue }
  }
  elseif (-not (Test-Path $model)) { Write-Host "skip $name ($model missing)"; continue }
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
