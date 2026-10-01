<#
.SYNOPSIS
  Dev-rig performance gate: run benchmark specs with Sigil's harness, compare against a baseline,
  and (with -Profile) check the plan's T-targets.

.DESCRIPTION
  For every spec (default: scripts/bench/specs/*.json, or the profile's list) this runs

      Sigil.exe --headless bench <spec> --out <OutRoot>/<spec stem>

  and collects report.json. A spec fails the gate when:
    - the harness exits non-zero (a run errored or a threshold in the spec failed), or
    - with -Baseline: a gated metric (default: every *_p95 metric) regressed by more than
      -MaxRegressionPct percent (default 5, the plan's rule for voice-changer changes) and by at
      least -MinDelta in absolute terms (default 0.25; keeps sub-quarter-millisecond noise on fast
      stub runs from blocking). A metric whose baseline is 0 has no percentage: any worsening of
      at least -MinDelta regresses. A gated metric the baseline has and the current report lacks
      (the run errored, measured nothing, or is gone) fails as MISSING.
  Lower is better for every gated metric except those a report lists in a run's
  `higher_is_better` (speaker similarity, DNSMOS, compare margins, throughput_x, ...), plus
  throughput_x, e2e_detected and the load duties for reports that predate that field.

  -Profile <name> reads scripts/bench/profiles/<name>.json (see profiles/p3.json):
    - `specs`: spec files run as they are;
    - `templates`: spec templates with ${VAR} placeholders (`*_JSON` variables are inserted as
      JSON, the others as string contents), materialized into <OutRoot>/specs/ with
      the profile's `vars`, the template's own `vars`, then (highest first) -Var NAME=value,
      -Slot / -Clip / -ContentEncoder, and SIGIL_BENCH_<NAME> environment variables. A template
      whose `requires` variables are empty is skipped with a warning (-Strict: an error);
    - `metrics`: the -Metrics default of the profile;
    - `targets`: the plan's T-rows: a metric of a spec / run with a min / max (or a `ratio` of two
      metrics). A missed or unmeasured `gate: true` target fails the gate (-TargetsAdvisory:
      reported only); `gate: false` rows are reported only. Targets of a skipped template are
      `skipped`.

  The output folder of one run is the -Baseline of the next: record a baseline before a change,
  then gate the change against it.

      pwsh scripts/bench/run-perf-gate.ps1 -OutRoot bench-results/baseline
      # ... change the voice changer, rebuild ...
      pwsh scripts/bench/run-perf-gate.ps1 -Baseline bench-results/baseline

      # P3: the voice engine with an imported template voice and a speech clip
      pwsh scripts/bench/run-perf-gate.ps1 -Profile p3 -AppData <app data with the models> `
          -Slot <app data>/voice-changer/models/default -Clip C:/clips/speech.wav

  Writes <OutRoot>/gate-summary.md and <OutRoot>/gate-summary.json. Exit code 0 = pass, 1 = fail,
  2 = usage error (no executable, no specs, bad profile).

  Windows PowerShell 5.1 blocks unsigned scripts by default; run it with pwsh (PowerShell 7) or
  `powershell -ExecutionPolicy Bypass -File scripts/bench/run-perf-gate.ps1`.

.PARAMETER Exe
  Sigil executable. Default: the first of src-tauri/target/release/sigil.exe,
  $env:CARGO_TARGET_DIR/release/sigil.exe, src-tauri/target/debug/sigil.exe. Release builds give
  representative numbers; a debug build's numbers are only comparable with another debug build.

.PARAMETER Specs
  Spec files to run. Default: the profile's specs and templates, else scripts/bench/specs/*.json.

.PARAMETER Profile
  A profile name (scripts/bench/profiles/<name>.json) or path.

.PARAMETER Var
  Template variables, NAME=value (repeatable, or one comma-separated string).

.PARAMETER Slot
  Shorthand for -Var SLOT=<folder> (an imported voice).

.PARAMETER Clip
  Shorthand for -Var CLIP=<wav> (the input clip of the real-voice templates).

.PARAMETER ContentEncoder
  Shorthand for -Var CONTENT_ENCODER=<onnx> (empty: the model store's per-EP variant).

.PARAMETER AppData
  SIGIL_APP_DATA_DIR for the harness (where the model store and voices are).

.PARAMETER Strict
  A template with missing variables is an error instead of a warning.

.PARAMETER TargetsAdvisory
  Report the profile's targets without failing the gate.

.PARAMETER OutRoot
  Output folder. Default: bench-results/perf-gate-<yyyyMMdd-HHmmss>.

.PARAMETER Baseline
  A previous gate output folder (<Baseline>/<spec stem>/report.json).

.PARAMETER Metrics
  Wildcard patterns of the metrics to compare against the baseline, as an array or one
  comma-separated string (what `pwsh -File ... -Metrics a,b` passes). Default: the profile's, else
  '*_p95'. A baseline gate whose patterns match no metric fails (it would otherwise pass without
  comparing anything).
#>
[CmdletBinding()]
param(
    [string]$Exe,
    [string[]]$Specs,
    [string]$Profile,
    [string[]]$Var,
    [string]$Slot,
    [string]$Clip,
    [string]$ContentEncoder,
    [string]$AppData,
    [switch]$Strict,
    [switch]$TargetsAdvisory,
    [string]$OutRoot,
    [string]$Baseline,
    [double]$MaxRegressionPct = 5.0,
    [double]$MinDelta = 0.25,
    [string[]]$Metrics
)

$ErrorActionPreference = 'Stop'
Set-StrictMode -Version 3.0

$repo = (Resolve-Path (Join-Path $PSScriptRoot '..\..')).Path
$benchDir = Join-Path $repo 'scripts\bench'
$fallbackHigherIsBetter = @('throughput_x', 'e2e_detected', 'gpu_load_duty', 'cpu_load_duty')

function Split-List {
    param([string[]]$Items)
    # `pwsh -File run-perf-gate.ps1 -Metrics a,b` passes one string 'a,b': split it.
    return @($Items | Where-Object { $_ } | ForEach-Object { $_ -split ',' } | ForEach-Object { $_.Trim() } | Where-Object { $_ })
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

function Get-Prop {
    param($Object, [string]$Name)
    if ($null -eq $Object) { return $null }
    $p = $Object.PSObject.Properties[$Name]
    if ($null -eq $p) { return $null }
    return $p.Value
}

function Test-Gated {
    param([string]$Name)
    foreach ($p in $script:gatedPatterns) { if ($Name -like $p) { return $true } }
    return $false
}

# ---- profile ------------------------------------------------------------------------------------
$profileDoc = $null
if ($Profile) {
    $profilePath = if (Test-Path -LiteralPath $Profile -PathType Leaf) { $Profile } else { Join-Path $benchDir ("profiles\" + $Profile + '.json') }
    if (-not (Test-Path -LiteralPath $profilePath -PathType Leaf)) {
        Write-Error "profile not found: $Profile ($profilePath)"
        exit 2
    }
    $profileDoc = Get-Content -LiteralPath $profilePath -Raw | ConvertFrom-Json
}

# Template variables: profile defaults < template vars < environment < parameters.
$cliVars = @{}
foreach ($pair in (Split-List $Var)) {
    $i = $pair.IndexOf('=')
    if ($i -lt 1) { Write-Error "-Var expects NAME=value, got '$pair'"; exit 2 }
    $cliVars[$pair.Substring(0, $i).Trim().ToUpperInvariant()] = $pair.Substring($i + 1)
}
if ($Slot) { $cliVars['SLOT'] = $Slot }
if ($Clip) { $cliVars['CLIP'] = $Clip }
if ($ContentEncoder) { $cliVars['CONTENT_ENCODER'] = $ContentEncoder }

function Resolve-Var {
    param([string]$Name, [hashtable]$TemplateVars)
    if ($cliVars.ContainsKey($Name)) { return $cliVars[$Name] }
    $envValue = [Environment]::GetEnvironmentVariable('SIGIL_BENCH_' + $Name)
    if ($envValue) { return $envValue }
    if ($TemplateVars.ContainsKey($Name)) { return $TemplateVars[$Name] }
    if ($null -ne $profileDoc) {
        $v = Get-Prop -Object (Get-Prop -Object $profileDoc -Name 'vars') -Name $Name
        if ($null -ne $v) { return [string]$v }
    }
    return ''
}

function Get-JsonString {
    # A value as the inside of a JSON string (paths use forward slashes).
    param([string]$Value)
    $v = $Value -replace '\\', '/'
    return ($v | ConvertTo-Json).Trim('"')
}

$exePath = Resolve-Exe -Given $Exe
if (-not $exePath) {
    Write-Error 'No Sigil executable found. Build one (cargo build --release --manifest-path src-tauri/Cargo.toml) or pass -Exe.'
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

$warnings = @()
$skippedTemplates = @()
if (-not $Specs -or $Specs.Count -eq 0) {
    $Specs = @()
    if ($null -ne $profileDoc) {
        foreach ($s in @(Get-Prop -Object $profileDoc -Name 'specs')) {
            if ($s) { $Specs += (Join-Path $benchDir $s) }
        }
        $templateDir = Join-Path $OutRoot 'specs'
        foreach ($t in @(Get-Prop -Object $profileDoc -Name 'templates')) {
            if ($null -eq $t) { continue }
            $tvars = @{}
            $own = Get-Prop -Object $t -Name 'vars'
            if ($null -ne $own) { foreach ($p in $own.PSObject.Properties) { $tvars[$p.Name.ToUpperInvariant()] = [string]$p.Value } }
            $tvars['NAME'] = [string]$t.name
            $missing = @(@(Get-Prop -Object $t -Name 'requires') | Where-Object { $_ -and -not (Resolve-Var -Name $_ -TemplateVars $tvars) })
            if ($missing.Count -gt 0) {
                $msg = "template $($t.name): no value for $($missing -join ', ') (-Var, -Slot / -Clip, or SIGIL_BENCH_<NAME>); skipped"
                if ($Strict) { Write-Error $msg; exit 2 }
                $warnings += $msg
                $skippedTemplates += [string]$t.name
                continue
            }
            $text = Get-Content -LiteralPath (Join-Path $benchDir $t.file) -Raw
            # `*_JSON` variables are inserted as JSON (objects, numbers); the rest as string contents.
            $text = [regex]::Replace($text, '\$\{([A-Z0-9_]+)\}', {
                    param($m)
                    $k = $m.Groups[1].Value
                    $v = Resolve-Var -Name $k -TemplateVars $tvars
                    if ($k.EndsWith('_JSON')) { if ($v) { $v } else { 'null' } } else { Get-JsonString $v }
                })
            New-Item -ItemType Directory -Force -Path $templateDir | Out-Null
            $path = Join-Path $templateDir ($t.name + '.json')
            Set-Content -LiteralPath $path -Value $text -Encoding utf8
            $Specs += $path
        }
    } else {
        $Specs = @(Get-ChildItem -Path (Join-Path $benchDir 'specs') -Filter '*.json' | Sort-Object Name | ForEach-Object { $_.FullName })
    }
}
if ($Specs.Count -eq 0) {
    Write-Error 'No spec files.'
    exit 2
}
$script:gatedPatterns = @(Split-List $Metrics)
if ($script:gatedPatterns.Count -eq 0 -and $null -ne $profileDoc) { $script:gatedPatterns = @(Split-List @(Get-Prop -Object $profileDoc -Name 'metrics')) }
if ($script:gatedPatterns.Count -eq 0) { $script:gatedPatterns = @('*_p95') }

if ($AppData) {
    New-Item -ItemType Directory -Force -Path $AppData | Out-Null
    $env:SIGIL_APP_DATA_DIR = (Resolve-Path -LiteralPath $AppData).Path
}

Write-Host "Sigil perf gate"
Write-Host "  exe:      $exePath"
Write-Host "  out:      $OutRoot"
if ($Profile) { Write-Host "  profile:  $Profile" }
if ($env:SIGIL_APP_DATA_DIR) { Write-Host "  app data: $($env:SIGIL_APP_DATA_DIR)" }
if ($Baseline) { Write-Host "  baseline: $Baseline (max regression $MaxRegressionPct %, min delta $MinDelta)" }

$results = @()
$comparisons = @()
$reports = @{}
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
        $reports[$stem] = $report
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
            $higher = @($fallbackHigherIsBetter) + @(Get-Prop -Object $run -Name 'higher_is_better') + @(Get-Prop -Object $baseRun -Name 'higher_is_better')
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
                $better = $higher -contains $name
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

# ---- targets --------------------------------------------------------------------------------------
function Get-Metric {
    param($Ref)
    $r = $reports[[string]$Ref.spec]
    if ($null -eq $r) { return $null }
    $run = $r.runs | Where-Object { $_.name -eq $Ref.run } | Select-Object -First 1
    if ($null -eq $run) { return $null }
    $v = Get-Prop -Object $run.metrics -Name $Ref.metric
    if ($null -eq $v) { return $null }
    return [double]$v
}

$targets = @()
if ($null -ne $profileDoc) {
    foreach ($t in @(Get-Prop -Object $profileDoc -Name 'targets')) {
        if ($null -eq $t) { continue }
        $ratio = Get-Prop -Object $t -Name 'ratio'
        if ($null -ne $ratio) {
            $num = Get-Metric -Ref $ratio.num
            $den = Get-Metric -Ref $ratio.den
            $value = if ($null -ne $num -and $null -ne $den -and $den -ne 0) { $num / $den } else { $null }
            $source = "$($ratio.num.spec)/$($ratio.num.run) $($ratio.num.metric) / $($ratio.den.spec)/$($ratio.den.run)"
        } else {
            $value = Get-Metric -Ref $t
            $source = "$($t.spec)/$($t.run) $($t.metric)"
        }
        $min = Get-Prop -Object $t -Name 'min'
        $max = Get-Prop -Object $t -Name 'max'
        $gate = [bool](Get-Prop -Object $t -Name 'gate')
        $specsOfTarget = if ($null -ne $ratio) { @($ratio.num.spec, $ratio.den.spec) } else { @($t.spec) }
        $skipped = @($specsOfTarget | Where-Object { $skippedTemplates -contains $_ }).Count -gt 0
        $status = if ($skipped) { 'skipped' }
            elseif ($null -eq $value) { 'not measured' }
            elseif (($null -ne $min -and $value -lt [double]$min) -or ($null -ne $max -and $value -gt [double]$max)) { 'MISSED' }
            else { 'met' }
        $limit = if ($null -ne $min -and $null -ne $max) { "$min .. $max" } elseif ($null -ne $min) { ">= $min" } else { "<= $max" }
        $targets += [pscustomobject]@{
            id = $t.id; label = $t.label; source = $source; limit = $limit
            value = $(if ($null -ne $value) { [math]::Round($value, 4) } else { $null })
            status = $status; gate = $gate
        }
    }
}

$regressions = @($comparisons | Where-Object { $_.regressed })
$failedSpecs = @($results | Where-Object { $_.status -ne 'PASS' })
$failedTargets = @($targets | Where-Object { $_.gate -and $_.status -ne 'met' -and $_.status -ne 'skipped' })
# A baseline gate that compared nothing (a -Metrics pattern that matches no metric) must not pass.
$comparedNothing = $Baseline -and ($comparisons.Count -eq 0)
if ($comparedNothing) {
    $warnings += "no baseline metric matched -Metrics '$($script:gatedPatterns -join "', '")': nothing was compared"
}
$passed = ($failedSpecs.Count -eq 0) -and ($regressions.Count -eq 0) -and -not $comparedNothing -and ($TargetsAdvisory -or $failedTargets.Count -eq 0)

$md = New-Object System.Text.StringBuilder
[void]$md.AppendLine('# Perf gate: ' + $(if ($passed) { 'PASSED' } else { 'FAILED' }))
[void]$md.AppendLine('')
[void]$md.AppendLine("Executable: ``$exePath``")
if ($Profile) { [void]$md.AppendLine("Profile: ``$Profile``") }
if ($Baseline) { [void]$md.AppendLine("Baseline: ``$Baseline`` (max regression $MaxRegressionPct %, min delta $MinDelta)") }
[void]$md.AppendLine('')
[void]$md.AppendLine('| Spec | Harness | Report |')
[void]$md.AppendLine('|---|---|---|')
foreach ($r in $results) { [void]$md.AppendLine("| $($r.spec) | $($r.status) (exit $($r.exit_code)) | $($r.report) |") }
if ($targets.Count -gt 0) {
    [void]$md.AppendLine('')
    [void]$md.AppendLine('| Target | What | Source | Limit | Value | Result |')
    [void]$md.AppendLine('|---|---|---|---|---|---|')
    foreach ($t in $targets) {
        $res = if ($t.status -eq 'met' -or $t.status -eq 'skipped') { $t.status } elseif ($t.gate) { "**$($t.status)**" } else { "$($t.status) (advisory)" }
        $val = if ($null -ne $t.value) { "$($t.value)" } else { '-' }
        [void]$md.AppendLine("| $($t.id) | $($t.label) | $($t.source) | $($t.limit) | $val | $res |")
    }
}
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
    passed = $passed; exe = $exePath; baseline = $Baseline; profile = $Profile
    max_regression_pct = $MaxRegressionPct; min_delta = $MinDelta
    specs = $results; comparisons = $comparisons; targets = $targets; warnings = $warnings
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
foreach ($t in $targets) {
    if ($t.status -ne 'met' -and $t.status -ne 'skipped') {
        Write-Host ("{0} {1} {2}: {3} (limit {4}){5}" -f $t.status.ToUpperInvariant(), $t.id, $t.label, $(if ($null -ne $t.value) { $t.value } else { '-' }), $t.limit, $(if ($t.gate) { '' } else { ' [advisory]' }))
    }
}
foreach ($r in $failedSpecs) { Write-Host ("{0} {1} (see {2})" -f $r.status, $r.spec, $r.report) }
Write-Host ("Perf gate {0}: {1} spec(s), {2} comparison(s), {3} regression(s), {4} target(s) ({5} gating missed). Summary: {6}" -f `
    $(if ($passed) { 'PASSED' } else { 'FAILED' }), $results.Count, $comparisons.Count, $regressions.Count, $targets.Count, $failedTargets.Count, (Join-Path $OutRoot 'gate-summary.md'))
if ($passed) { exit 0 } else { exit 1 }
