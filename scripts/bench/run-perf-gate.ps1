<#
.SYNOPSIS
  Dev-rig performance gate: run benchmark specs with Sigil's harness and compare against a baseline.

.DESCRIPTION
  For every spec (default: scripts/bench/specs/*.json) this runs

      Sigil.exe --headless bench <spec> --out <OutRoot>/<spec stem>

  and collects report.json. A spec fails the gate when:
    - the harness exits non-zero (a run errored or a threshold in the spec failed), or
    - with -Baseline: a gated metric (default: every *_p95 metric) regressed by more than
      -MaxRegressionPct percent (default 5, the plan's rule for voice-changer changes) and by at
      least -MinDelta in absolute terms (default 0.25; keeps sub-quarter-millisecond noise on fast
      stub runs from blocking). A metric whose baseline is 0 has no percentage: any worsening of
      at least -MinDelta regresses. A gated metric the baseline has and the current report lacks
      (the run errored, measured nothing, or is gone) fails as MISSING.
  Lower is better for every gated metric except throughput_x, e2e_detected and the load duties.

  The output folder of one run is the -Baseline of the next: record a baseline before a change,
  then gate the change against it.

      pwsh scripts/bench/run-perf-gate.ps1 -OutRoot bench-results/baseline
      # ... change the voice changer, rebuild ...
      pwsh scripts/bench/run-perf-gate.ps1 -Baseline bench-results/baseline

  Writes <OutRoot>/gate-summary.md and <OutRoot>/gate-summary.json. Exit code 0 = pass, 1 = fail,
  2 = usage error (no executable, no specs).

  Windows PowerShell 5.1 blocks unsigned scripts by default; run it with pwsh (PowerShell 7) or
  `powershell -ExecutionPolicy Bypass -File scripts/bench/run-perf-gate.ps1`.

.PARAMETER Exe
  Sigil executable. Default: the first of src-tauri/target/release/sigil.exe,
  $env:CARGO_TARGET_DIR/release/sigil.exe, src-tauri/target/debug/sigil.exe. Release builds give
  representative numbers; a debug build's numbers are only comparable with another debug build.

.PARAMETER Specs
  Spec files to run. Default: scripts/bench/specs/*.json.

.PARAMETER OutRoot
  Output folder. Default: bench-results/perf-gate-<yyyyMMdd-HHmmss>.

.PARAMETER Baseline
  A previous gate output folder (<Baseline>/<spec stem>/report.json).

.PARAMETER Metrics
  Wildcard patterns of the metrics to compare against the baseline, as an array or one
  comma-separated string (what `pwsh -File ... -Metrics a,b` passes). Default: '*_p95'. A baseline
  gate whose patterns match no metric fails (it would otherwise pass without comparing anything).
#>
[CmdletBinding()]
param(
    [string]$Exe,
    [string[]]$Specs,
    [string]$OutRoot,
    [string]$Baseline,
    [double]$MaxRegressionPct = 5.0,
    [double]$MinDelta = 0.25,
    [string[]]$Metrics = @('*_p95')
)

$ErrorActionPreference = 'Stop'
Set-StrictMode -Version 3.0

$repo = (Resolve-Path (Join-Path $PSScriptRoot '..\..')).Path
$higherIsBetter = @('throughput_x', 'e2e_detected', 'gpu_load_duty', 'cpu_load_duty')
# `pwsh -File run-perf-gate.ps1 -Metrics a,b` passes one string 'a,b': split it.
$Metrics = @($Metrics | ForEach-Object { $_ -split ',' } | ForEach-Object { $_.Trim() } | Where-Object { $_ })
if ($Metrics.Count -eq 0) {
    Write-Error 'No -Metrics patterns.'
    exit 2
}

function Resolve-Exe {
    param([string]$Given)
    if ($Given) {
        if (-not (Test-Path -LiteralPath $Given -PathType Leaf)) { throw "executable not found: $Given" }
        return (Resolve-Path -LiteralPath $Given).Path
    }
    $candidates = @(Join-Path $repo 'src-tauri\target\release\sigil.exe')
    if ($env:CARGO_TARGET_DIR) { $candidates += Join-Path $env:CARGO_TARGET_DIR 'release\sigil.exe' }
    $candidates += Join-Path $repo 'src-tauri\target\debug\sigil.exe'
    foreach ($c in $candidates) {
        if (Test-Path -LiteralPath $c -PathType Leaf) { return (Resolve-Path -LiteralPath $c).Path }
    }
    return $null
}

function Test-Gated {
    param([string]$Name)
    foreach ($p in $Metrics) { if ($Name -like $p) { return $true } }
    return $false
}

function Get-Prop {
    param($Object, [string]$Name)
    if ($null -eq $Object) { return $null }
    $p = $Object.PSObject.Properties[$Name]
    if ($null -eq $p) { return $null }
    return $p.Value
}

$exePath = Resolve-Exe -Given $Exe
if (-not $exePath) {
    Write-Error 'No Sigil executable found. Build one (cargo build --release --manifest-path src-tauri/Cargo.toml) or pass -Exe.'
    exit 2
}
if (-not $Specs -or $Specs.Count -eq 0) {
    $Specs = @(Get-ChildItem -Path (Join-Path $repo 'scripts\bench\specs') -Filter '*.json' | Sort-Object Name | ForEach-Object { $_.FullName })
}
if ($Specs.Count -eq 0) {
    Write-Error 'No spec files.'
    exit 2
}
if (-not $OutRoot) {
    $OutRoot = Join-Path $repo ('bench-results\perf-gate-' + (Get-Date -Format 'yyyyMMdd-HHmmss'))
}
New-Item -ItemType Directory -Force -Path $OutRoot | Out-Null
$OutRoot = (Resolve-Path -LiteralPath $OutRoot).Path
if ($Baseline -and -not (Test-Path -LiteralPath $Baseline -PathType Container)) {
    Write-Error "baseline folder not found: $Baseline"
    exit 2
}

Write-Host "Sigil perf gate"
Write-Host "  exe:      $exePath"
Write-Host "  out:      $OutRoot"
if ($Baseline) { Write-Host "  baseline: $Baseline (max regression $MaxRegressionPct %, min delta $MinDelta)" }

$results = @()
$comparisons = @()
$warnings = @()
foreach ($spec in $Specs) {
    $specPath = (Resolve-Path -LiteralPath $spec).Path
    $stem = [System.IO.Path]::GetFileNameWithoutExtension($specPath)
    $out = Join-Path $OutRoot $stem
    Write-Host ""
    Write-Host "== $stem"
    # Release builds use the GUI subsystem: PowerShell only waits for them (and sets
    # $LASTEXITCODE) when their output is piped. Stderr carries progress lines, which must not
    # become terminating errors under Windows PowerShell 5.1's 'Stop' preference.
    $prevPreference = $ErrorActionPreference
    $ErrorActionPreference = 'Continue'
    $global:LASTEXITCODE = $null
    & $exePath --headless bench $specPath --out $out 2>&1 | ForEach-Object { Write-Host "$_" }
    $code = $LASTEXITCODE
    $ErrorActionPreference = $prevPreference
    if ($null -eq $code) { $code = -1 }
    $reportPath = Join-Path $out 'report.json'
    $report = $null
    if (Test-Path -LiteralPath $reportPath) {
        $report = Get-Content -LiteralPath $reportPath -Raw | ConvertFrom-Json
    }
    $status = if ($null -eq $report) { 'ERROR' } elseif ($code -eq 0) { 'PASS' } else { 'FAIL' }
    $results += [pscustomobject]@{ spec = $stem; status = $status; exit_code = $code; report = $reportPath }

    if ($Baseline -and $null -ne $report) {
        $basePath = Join-Path $Baseline (Join-Path $stem 'report.json')
        if (-not (Test-Path -LiteralPath $basePath)) {
            $warnings += "${stem}: no baseline report ($basePath)"
            continue
        }
        $base = Get-Content -LiteralPath $basePath -Raw | ConvertFrom-Json
        if ($base.spec_sha256 -ne $report.spec_sha256) {
            $warnings += "${stem}: the spec changed since the baseline; comparing anyway"
        }
        if ($base.environment.build -ne $report.environment.build) {
            $warnings += "${stem}: baseline is a $($base.environment.build) build, this is $($report.environment.build)"
        }
        foreach ($run in $report.runs) {
            if ($null -eq ($base.runs | Where-Object { $_.name -eq $run.name } | Select-Object -First 1)) {
                $warnings += "${stem}/$($run.name): not in the baseline"
            }
        }
        # Every gated metric the baseline has must still be measured: a metric that disappears
        # (a run that errored, stopped playing, or was removed) fails instead of being skipped.
        foreach ($baseRun in $base.runs) {
            $run = $report.runs | Where-Object { $_.name -eq $baseRun.name } | Select-Object -First 1
            foreach ($prop in $baseRun.metrics.PSObject.Properties) {
                $name = $prop.Name
                if (-not (Test-Gated -Name $name)) { continue }
                $old = [double]$prop.Value
                $current = if ($null -ne $run) { Get-Prop -Object $run.metrics -Name $name } else { $null }
                if ($null -eq $current) {
                    $comparisons += [pscustomobject]@{
                        spec = $stem; run = $baseRun.name; metric = $name
                        baseline = [math]::Round($old, 4); current = $null
                        change_pct = $null; regressed = $true
                        note = $(if ($null -eq $run) { 'run missing' } else { 'not measured' })
                    }
                    continue
                }
                $new = [double]$current
                $better = $higherIsBetter -contains $name
                $delta = if ($better) { $old - $new } else { $new - $old }
                if ($old -ne 0) {
                    $pct = [math]::Round(100.0 * $delta / [math]::Abs($old), 2)
                    $regressed = ($pct -gt $MaxRegressionPct) -and ([math]::Abs($delta) -ge $MinDelta)
                    $note = ''
                } else {
                    # No percentage from a zero baseline (0 underruns -> 50 is not "0 %"): any
                    # worsening of at least -MinDelta regresses.
                    $pct = $null
                    $regressed = ($delta -gt 0) -and ($delta -ge $MinDelta)
                    $note = 'baseline 0'
                }
                $comparisons += [pscustomobject]@{
                    spec = $stem; run = $baseRun.name; metric = $name
                    baseline = [math]::Round($old, 4); current = [math]::Round($new, 4)
                    change_pct = $pct; regressed = $regressed; note = $note
                }
            }
        }
    }
}

$regressions = @($comparisons | Where-Object { $_.regressed })
$failedSpecs = @($results | Where-Object { $_.status -ne 'PASS' })
# A baseline gate that compared nothing (a -Metrics pattern that matches no metric) must not pass.
$comparedNothing = $Baseline -and ($comparisons.Count -eq 0)
if ($comparedNothing) {
    $warnings += "no baseline metric matched -Metrics '$($Metrics -join "', '")': nothing was compared"
}
$passed = ($failedSpecs.Count -eq 0) -and ($regressions.Count -eq 0) -and -not $comparedNothing

$md = New-Object System.Text.StringBuilder
[void]$md.AppendLine('# Perf gate: ' + $(if ($passed) { 'PASSED' } else { 'FAILED' }))
[void]$md.AppendLine('')
[void]$md.AppendLine("Executable: ``$exePath``")
if ($Baseline) { [void]$md.AppendLine("Baseline: ``$Baseline`` (max regression $MaxRegressionPct %, min delta $MinDelta)") }
[void]$md.AppendLine('')
[void]$md.AppendLine('| Spec | Harness | Report |')
[void]$md.AppendLine('|---|---|---|')
foreach ($r in $results) { [void]$md.AppendLine("| $($r.spec) | $($r.status) (exit $($r.exit_code)) | $($r.report) |") }
if ($comparisons.Count -gt 0) {
    [void]$md.AppendLine('')
    [void]$md.AppendLine('| Spec / run | Metric | Baseline | Current | Change | Result |')
    [void]$md.AppendLine('|---|---|---|---|---|---|')
    foreach ($c in $comparisons) {
        $res = if ($null -eq $c.current) { 'MISSING' } elseif ($c.regressed) { 'REGRESSED' } else { 'ok' }
        $cur = if ($null -eq $c.current) { $c.note } else { "$($c.current)" }
        $chg = if ($null -ne $c.change_pct) { "$($c.change_pct) %" } elseif ($null -eq $c.current) { '-' } else { "n/a ($($c.note))" }
        [void]$md.AppendLine("| $($c.spec) / $($c.run) | $($c.metric) | $($c.baseline) | $cur | $chg | $res |")
    }
}
if ($warnings.Count -gt 0) {
    [void]$md.AppendLine('')
    [void]$md.AppendLine('Warnings:')
    foreach ($w in $warnings) { [void]$md.AppendLine("- $w") }
}
$md.ToString() | Set-Content -LiteralPath (Join-Path $OutRoot 'gate-summary.md') -Encoding utf8
[pscustomobject]@{
    passed = $passed; exe = $exePath; baseline = $Baseline
    max_regression_pct = $MaxRegressionPct; min_delta = $MinDelta
    specs = $results; comparisons = $comparisons; warnings = $warnings
} | ConvertTo-Json -Depth 6 | Set-Content -LiteralPath (Join-Path $OutRoot 'gate-summary.json') -Encoding utf8

Write-Host ""
foreach ($w in $warnings) { Write-Warning $w }
foreach ($c in $regressions) {
    if ($null -eq $c.current) {
        Write-Host ("MISSING {0}/{1} {2}: {3} in the baseline, {4} now" -f $c.spec, $c.run, $c.metric, $c.baseline, $c.note)
    } elseif ($null -eq $c.change_pct) {
        Write-Host ("REGRESSED {0}/{1} {2}: {3} -> {4} ({5})" -f $c.spec, $c.run, $c.metric, $c.baseline, $c.current, $c.note)
    } else {
        Write-Host ("REGRESSED {0}/{1} {2}: {3} -> {4} ({5} %)" -f $c.spec, $c.run, $c.metric, $c.baseline, $c.current, $c.change_pct)
    }
}
foreach ($r in $failedSpecs) { Write-Host ("{0} {1} (see {2})" -f $r.status, $r.spec, $r.report) }
Write-Host ("Perf gate {0}: {1} spec(s), {2} comparison(s), {3} regression(s). Summary: {4}" -f `
    $(if ($passed) { 'PASSED' } else { 'FAILED' }), $results.Count, $comparisons.Count, $regressions.Count, (Join-Path $OutRoot 'gate-summary.md'))
if ($passed) { exit 0 } else { exit 1 }
