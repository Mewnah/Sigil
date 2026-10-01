<#
.SYNOPSIS
  Run a voice through a matrix of engine configurations and tabulate the results.

.DESCRIPTION
  Materializes scripts/bench/specs/p3/matrix.template.json (or -Template) once per combination of
  -Eps x -Pitch x -BlockMs x -Precision x -ExtraMs, runs each with

      Sigil.exe --headless bench <cell spec> --out <OutRoot>/<cell>

  one after the other (never in parallel: the cells would compete for the GPU), and writes
  <OutRoot>/matrix.csv and <OutRoot>/matrix.md with, per cell: the EP it asked for and the one it
  ran on, compute p50 / p95 / p99, p99 RTF (over the engine block), the paced latency p50 / p99,
  over-budget ratio, process VRAM delta, CPU, cold load time and the per-component compute of
  template voices built directly (`-Build direct`). A cell that fails to load keeps its error.

  The template's variables: NAME, SLOT, CLIP, CONTENT_ENCODER, EP, PITCH, BLOCK_MS, BLOCK_FRAMES,
  EXTRA_MS, PRECISION, PRESET, BUILD, LEVERS (text) and LEVERS_JSON (inserted as JSON; variables
  ending in _JSON are not quoted). `-BlockFrames 0` uses the engine block as the device block
  (40 ms = 1920 frames at 48 kHz): the engine's latency floor, which the voice changer cannot run
  (its device blocks are its chunk sizes; any other setting runs 512 frames), so such cells are
  marked "benchmark only" and their latency is not one the app reproduces. `-Levers
  io_binding=on,cuda_graphs=on` sets engine levers (the rest stay `auto`).

      pwsh scripts/bench/run-matrix.ps1 -AppData <app data> -Slot <voice folder> -Clip C:/clips/speech.wav `
          -Eps dml,cuda,cpu -Pitch rmvpe,fcpe,swiftf0 -BlockMs 20,40,80

.PARAMETER Exe
  Sigil executable (default as in run-perf-gate.ps1).
#>
[CmdletBinding()]
param(
    [string]$Exe,
    [Parameter(Mandatory = $true)][string]$Slot,
    [Parameter(Mandatory = $true)][string]$Clip,
    [string]$AppData,
    [string]$ContentEncoder = '',
    [string[]]$Eps = @('dml'),
    [string[]]$Pitch = @('rmvpe'),
    [string[]]$BlockMs = @('40'),
    [string[]]$Precision = @('auto'),
    [string[]]$ExtraMs = @('64'),
    [int]$BlockFrames = 512,
    [string]$Preset = 'low_latency',
    [string[]]$Levers = @(),
    [string]$Build = 'auto',
    [string]$Template,
    [string]$OutRoot
)

$ErrorActionPreference = 'Stop'
Set-StrictMode -Version 3.0
$repo = (Resolve-Path (Join-Path $PSScriptRoot '..\..')).Path

function Split-List {
    param([string[]]$Items)
    return @($Items | Where-Object { $_ } | ForEach-Object { $_ -split ',' } | ForEach-Object { $_.Trim() } | Where-Object { $_ })
}

function Get-Prop {
    param($Object, [string]$Name)
    if ($null -eq $Object) { return $null }
    $p = $Object.PSObject.Properties[$Name]
    if ($null -eq $p) { return $null }
    return $p.Value
}

if (-not $Exe) {
    foreach ($c in @((Join-Path $repo 'src-tauri\target\release\sigil.exe'), $(if ($env:CARGO_TARGET_DIR) { Join-Path $env:CARGO_TARGET_DIR 'release\sigil.exe' }), (Join-Path $repo 'src-tauri\target\debug\sigil.exe'))) {
        if ($c -and (Test-Path -LiteralPath $c -PathType Leaf)) { $Exe = $c; break }
    }
}
if (-not $Exe -or -not (Test-Path -LiteralPath $Exe -PathType Leaf)) { Write-Error 'No Sigil executable found; pass -Exe.'; exit 2 }
$Exe = (Resolve-Path -LiteralPath $Exe).Path
if (-not $Template) { $Template = Join-Path $repo 'scripts\bench\specs\p3\matrix.template.json' }
$templateText = Get-Content -LiteralPath $Template -Raw
if (-not $OutRoot) { $OutRoot = Join-Path $repo ('bench-results\matrix-' + (Get-Date -Format 'yyyyMMdd-HHmmss')) }
New-Item -ItemType Directory -Force -Path (Join-Path $OutRoot 'specs') | Out-Null
$OutRoot = (Resolve-Path -LiteralPath $OutRoot).Path
if ($AppData) { $env:SIGIL_APP_DATA_DIR = (Resolve-Path -LiteralPath $AppData).Path }

function Get-JsonString {
    param([string]$Value)
    return (($Value -replace '\\', '/') | ConvertTo-Json).Trim('"')
}

$leverPairs = @(Split-List $Levers)
$leverMap = [ordered]@{}
foreach ($pair in $leverPairs) {
    $i = $pair.IndexOf('=')
    if ($i -lt 1) { Write-Error "-Levers expects name=mode, got '$pair'"; exit 2 }
    $leverMap[$pair.Substring(0, $i).Trim()] = $pair.Substring($i + 1).Trim()
}
$leversJson = if ($leverMap.Count -gt 0) { $leverMap | ConvertTo-Json -Compress } else { '{}' }
$leversText = if ($leverPairs.Count -gt 0) { $leverPairs -join ' ' } else { 'auto' }

$rows = @()
foreach ($ep in (Split-List $Eps)) {
    foreach ($f0 in (Split-List $Pitch)) {
        foreach ($block in (Split-List $BlockMs)) {
            foreach ($prec in (Split-List $Precision)) {
                foreach ($extra in (Split-List $ExtraMs)) {
                    $frames = if ($BlockFrames -gt 0) { $BlockFrames } else { [int]([double]$block * 48) }
                    $short = @{ io_binding = 'io'; cuda_graphs = 'graphs'; gpu_priority = 'prio'; partial_decode = 'pd'; anchored_noise = 'an'; nsf_phase0 = 'ph0'; f0_cache = 'f0c'; gpu_knn = 'knn'; governor = 'gov' }
                    $leverTag = if ($leverMap.Count -gt 0) { '-' + (($leverMap.Keys | ForEach-Object { $(if ($short.ContainsKey($_)) { $short[$_] } else { $_ }) + $(if ($leverMap[$_] -eq 'off') { '0' } else { '' }) }) -join '_') } else { '' }
                    $name = ("{0}-{1}-{2}ms-{3}-ctx{4}-dev{5}{6}" -f $ep, $f0, $block, $prec, $extra, $frames, $leverTag)
                    if ($name.Length -gt 64) { $name = $name.Substring(0, 64) }
                    $vars = @{
                        NAME = $name; SLOT = $Slot; CLIP = $Clip; CONTENT_ENCODER = $ContentEncoder; EP = $ep; PITCH = $f0
                        BLOCK_MS = $block; BLOCK_FRAMES = "$frames"; EXTRA_MS = $extra; PRECISION = $prec; PRESET = $Preset; BUILD = $Build
                        LEVERS = $leversText; LEVERS_JSON = $leversJson
                    }
                    $text = [regex]::Replace($templateText, '\$\{([A-Z0-9_]+)\}', {
                            param($m)
                            $k = $m.Groups[1].Value
                            if ($k.EndsWith('_JSON')) { [string]$vars[$k] } else { Get-JsonString ([string]$vars[$k]) }
                        })
                    $specPath = Join-Path $OutRoot ("specs\" + $name + '.json')
                    Set-Content -LiteralPath $specPath -Value $text -Encoding utf8
                    $out = Join-Path $OutRoot $name
                    Write-Host "== $name"
                    $prev = $ErrorActionPreference
                    $ErrorActionPreference = 'Continue'
                    & $Exe --headless bench $specPath --out $out 2>&1 | ForEach-Object { Write-Host "$_" }
                    $ErrorActionPreference = $prev
                    $row = [ordered]@{
                        cell = $name; ep = $ep; pitch = $f0; block_ms = $block; device_frames = $frames; precision = $prec; extra_ms = $extra; levers = $leversText
                        ep_active = ''; build = ''; load_ms = ''; compute_p50 = ''; compute_p95 = ''; compute_p99 = ''; rtf_p99 = ''
                        latency_p50 = ''; latency_p99 = ''; over_budget = ''; vram_mb = ''; cpu_cores = ''
                        content_p50 = ''; f0_p50 = ''; enc_p50 = ''; flowdec_p50 = ''; error = ''
                    }
                    $reportPath = Join-Path $out 'report.json'
                    if (Test-Path -LiteralPath $reportPath) {
                        $r = Get-Content -LiteralPath $reportPath -Raw | ConvertFrom-Json
                        $e = $r.engine
                        if ($null -ne $e) {
                            $row.ep_active = [string](Get-Prop -Object $e -Name 'execution_provider_active')
                            $row.build = [string](Get-Prop -Object $e -Name 'build')
                            $row.load_ms = [math]::Round([double](Get-Prop -Object $e -Name 'load_ms'), 1)
                            # A device block the voice changer cannot run (the engine's floor only).
                            $warns = @(Get-Prop -Object $e -Name 'warnings') | Where-Object { "$_" -like '*not a voice-changer device block*' }
                            if (@($warns).Count -gt 0) { $row.device_frames = "$frames (benchmark only)" }
                        }
                        $off = $r.runs | Where-Object { $_.name -eq 'offline' } | Select-Object -First 1
                        $pac = $r.runs | Where-Object { $_.name -eq 'paced' } | Select-Object -First 1
                        $g = { param($run, $k) $v = Get-Prop -Object (Get-Prop -Object $run -Name 'metrics') -Name $k; if ($null -ne $v) { [math]::Round([double]$v, 3) } else { '' } }
                        # Per-block compute = the device blocks that ran the model (`call_ms`; older
                        # reports: every device block).
                        $cm = if ("$(& $g $off 'call_ms_p50')" -ne '') { 'call_ms' } else { 'block_ms' }
                        $row.compute_p50 = & $g $off "$($cm)_p50"
                        $row.compute_p95 = & $g $off "$($cm)_p95"
                        $row.compute_p99 = & $g $off "$($cm)_p99"
                        $row.rtf_p99 = & $g $off 'rtf_p99'
                        $row.vram_mb = & $g $off 'vram_engine_mb'
                        $row.content_p50 = & $g $off 'content_ms_p50'
                        $row.f0_p50 = & $g $off 'f0_ms_p50'
                        $row.enc_p50 = & $g $off 'enc_ms_p50'
                        $row.flowdec_p50 = & $g $off 'flowdec_ms_p50'
                        $row.latency_p50 = & $g $pac 'latency_ms_p50'
                        $row.latency_p99 = & $g $pac 'latency_ms_p99'
                        $row.over_budget = & $g $pac 'over_budget_ratio'
                        $row.cpu_cores = & $g $pac 'cpu_cores'
                        $errs = @($r.runs | Where-Object { $null -ne (Get-Prop -Object $_ -Name 'error') } | ForEach-Object { $_.error })
                        if ($errs.Count -gt 0) { $row.error = ($errs -join '; ') }
                    } else {
                        $row.error = 'no report'
                    }
                    $rows += [pscustomobject]$row
                }
            }
        }
    }
}

$rows | Export-Csv -LiteralPath (Join-Path $OutRoot 'matrix.csv') -NoTypeInformation -Encoding utf8
$md = New-Object System.Text.StringBuilder
[void]$md.AppendLine("# Matrix: $(Split-Path -Leaf $Slot)")
[void]$md.AppendLine('')
[void]$md.AppendLine("Executable: ``$Exe``; clip: ``$Clip``; preset ``$Preset``; build ``$Build``.")
[void]$md.AppendLine('')
[void]$md.AppendLine('| EP (active) | F0 | Block ms | Device frames | Precision | Context ms | Levers | Load ms | Compute p50 / p95 / p99 ms | RTF p99 | Latency p50 / p99 ms | Over budget | VRAM MB | CPU cores | content / f0 / enc / flowdec p50 ms | Error |')
[void]$md.AppendLine('|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|')
foreach ($r in $rows) {
    [void]$md.AppendLine(("| {0} ({1}) | {2} | {3} | {4} | {5} | {6} | {22} | {7} | {8} / {9} / {10} | {11} | {12} / {13} | {14} | {15} | {16} | {17} / {18} / {19} / {20} | {21} |" -f `
        $r.ep, $r.ep_active, $r.pitch, $r.block_ms, $r.device_frames, $r.precision, $r.extra_ms, $r.load_ms, $r.compute_p50, $r.compute_p95, $r.compute_p99,
        $r.rtf_p99, $r.latency_p50, $r.latency_p99, $r.over_budget, $r.vram_mb, $r.cpu_cores, $r.content_p50, $r.f0_p50, $r.enc_p50, $r.flowdec_p50, $r.error, $r.levers))
}
if (@($rows | Where-Object { "$($_.device_frames)" -like '*benchmark only*' }).Count -gt 0) {
    [void]$md.AppendLine('')
    [void]$md.AppendLine('"Benchmark only": a device block the voice changer does not run (it runs its chunk sizes; any other setting becomes 512 frames). Those cells show the engine''s latency floor, not a configuration the app reproduces.')
}
$md.ToString() | Set-Content -LiteralPath (Join-Path $OutRoot 'matrix.md') -Encoding utf8
Write-Host ""
Write-Host "Matrix: $($rows.Count) cell(s). $(Join-Path $OutRoot 'matrix.md')"
