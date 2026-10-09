<#
.SYNOPSIS
  Dev-rig performance gate: run benchmark specs with Sigil's harness, compare against a baseline or
  (P3B) interleave two builds / configurations with paired statistics, and (with -Profile) check
  the plan's T-targets.

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

  A single before / after pair is noise on a desktop GPU (P3B measured 2-19 % per-pair spread of
  p95 between two runs of the same exe). Before/after claims use the interleaved mode:

  INTERLEAVED A/B (P3B-A; -CompareExe, -CompareVars and/or -CompareEnv)
    A = -CompareExe (the "before" build; default -Exe) with the profile's variables;
    B = -Exe with -CompareVars applied to the templates and -CompareEnv set in its environment.
    Each spec runs A and B in ABBA order (round 1: A then B, round 2: B then A, ...) for -Rounds
    rounds (default 7), each pair holding the GPU timing lock, so slow drifts of the PC hit both
    sides alike. Per gated metric (-Metrics, default *_p95) and run:
      r_i = ln(B_i / A_i)   (ln(A_i / B_i) for higher-is-better metrics; r > 0 = B is worse)
    and an interval on their centre: the exact Wilcoxon signed-rank / Hodges-Lehmann interval
    from 6 pairs on (coverage >= 95 %), the Student t-interval below. No bootstrap.
      REGRESSED     median above +MaxRegressionPct and the interval's lower bound above 0;
      INCONCLUSIVE  median above +MaxRegressionPct, lower bound at or below 0: two more rounds are
                    added automatically up to -MaxRounds (default 15);
      IMPROVED      the interval's upper bound below 0 and the median at least -ImprovedFloorPct
                    (default 1 %) better: a statistically clear but negligible shift (the A/A's false
                    IMPROVED was -0.87 %) is NEUTRAL, not evidence;
      NEUTRAL       otherwise.
    A row whose interval is wider than the regression limit (half-width above -MaxRegressionPct)
    cannot tell a real regression from noise at this round count: the gate warns for it (paced rows
    and stage metrics such as enc_ms_p95 are the usual ones; on this rig paced rows resolve to
    +-8..13 % at 7 rounds). Size their rounds from an A/A session (-AASummary), or pass
    -AllowInconclusive when a stage metric is gated through -Metrics.
    The gate fails on REGRESSED, on INCONCLUSIVE left at -MaxRounds (unless -AllowInconclusive)
    and on REFUSED: F0 metrics (pitch error, V/UV, cents, fed-F0) are compared only between reports
    stamped with the same `analyzer_version` (P3B-B; a report without a stamp counts as its own
    analyzer). Metrics with a value <= 0 in a pair use the difference relative to A's median
    (max of it and -MinDelta) instead of the log ratio.
    -AASummary <gate-summary.json of an A/A session> sizes each spec's rounds from its measured
    per-pair spread (enough pairs for a +-TargetHalfWidthPct interval, within -Rounds..-MaxRounds).
    -Only runs just the named specs / templates (interleave only what a change touches).
    Every report's GPU / host state (`state_*` metrics: SM / memory clock, speed index, host probe,
    effective CPU clock) is compared with the session's median for the same spec and run; runs more
    than 10 % off are flagged in gate-summary.json / .md and stamped into their report.json
    (`gate_state_flags`). A pair whose two sides ran in different GPU clock regimes (median SM clock,
    or the speed index without NVML, more than -RegimeTolerancePct apart: the driver held the GPU
    at a lower clock for a while, e.g. while another process used it) compares two GPU states, not
    two builds; each paired row also carries its statistics without such pairs (`matched_*`, the
    gate's resolution while the GPU holds one state; information only, the verdict uses every pair).
    A/B reports whose process QoS differs (an exe without the QoS opt-out vs
    one with it) are warned about: use -Var QOS=none (both sides) against such an exe.

      # A/A (the gate's resolution): the same exe on both sides
      pwsh scripts/bench/run-perf-gate.ps1 -Profile p3b -Exe new.exe -CompareExe new.exe -Only sve-dml,sve-cuda
      # A change against the P3B baseline build, sized from that A/A
      pwsh scripts/bench/run-perf-gate.ps1 -Profile p3b -Exe new.exe -CompareExe exe-s0/sigil.exe `
          -Var QOS=none -AASummary bench-results/aa/gate-summary.json
      # A kill switch on one exe
      pwsh scripts/bench/run-perf-gate.ps1 -Profile p3b -CompareEnv 'SIGIL_VC_CONTENT_PLAN=0'
      # A spec variable (e.g. levers) on one exe
      pwsh scripts/bench/run-perf-gate.ps1 -Profile p3b -CompareVars 'LEVERS_JSON={"cuda_graphs":"off"}'

  -Profile <name> reads scripts/bench/profiles/<name>.json (see profiles/p3.json, p3b.json):
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
      `skipped`. In the interleaved mode targets read the median of B's rounds (per metric), never
      one round: a single round's p95 carries the 11-19 % per-pair spread of the rig.

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

  Every harness run holds the dev rig's GPU timing lock (scripts/bench/gpu-lock.ps1, P3B rule R2):
  it waits up to -GpuLockWait minutes for another timing run (another agent, P6) to finish.
  -NoGpuLock skips the lock (stub specs that measure no timing). In the interleaved mode the lock
  is held for each A/B pair (the timed section only, never across a build).

  -SelfTest checks the statistics against the golden values of the Rust implementation
  (`bench::metrics::paired` tests), the verdict rule, the round extension and the analyzer refusal,
  without running anything.

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

.PARAMETER NoGpuLock
  Do not take the GPU timing lock (stub specs only).

.PARAMETER GpuLockWait
  Minutes to wait for the GPU timing lock before failing (default 180).

.PARAMETER GpuLockTag
  Owner tag written into the lock (default 'perf-gate').

.PARAMETER CompareExe
  Interleaved mode: the "before" executable (side A).

.PARAMETER CompareVars
  Interleaved mode: template variables of side B, 'NAME=value;NAME2=value2'.

.PARAMETER CompareEnv
  Interleaved mode: environment variables of side B's harness runs, 'NAME=value;NAME2=value2'.

.PARAMETER Rounds
  Interleaved mode: A/B pairs per spec (default 7; at least 2).

.PARAMETER MaxRounds
  Interleaved mode: the most pairs an INCONCLUSIVE spec grows to (default 15).

.PARAMETER AASummary
  Interleaved mode: an A/A session's gate-summary.json; sizes the rounds per spec.

.PARAMETER TargetHalfWidthPct
  Interleaved mode with -AASummary: the interval half-width the rounds aim for (default 3).

.PARAMETER RegimeTolerancePct
  Interleaved mode: two sides of a pair whose run's median SM clock (or, without NVML, GPU speed
  index) differ by more than this many percent ran in different GPU clock regimes (default 25; a
  real-time cadence moves the SM clock by up to about 20 % within one regime). Each paired row also
  reports its statistics without such pairs (`matched_*`, information only: the verdict always uses
  every pair).

.PARAMETER Only
  Spec or template names (wildcards) to run; the others are skipped.

.PARAMETER ImprovedFloorPct
  Interleaved mode: the median improvement a row needs, besides an interval entirely below 0, to be
  IMPROVED (default 1 %). Smaller shifts are NEUTRAL.

.PARAMETER AllowInconclusive
  Interleaved mode: INCONCLUSIVE at -MaxRounds does not fail the gate (reported only).

.PARAMETER Resume
  Interleaved mode: reuse the rounds whose A and B reports already exist in -OutRoot (a session
  that was interrupted continues where it stopped; rerun the same command with the same -OutRoot).
  A reused round's harness exit codes are not known and count as 0.

.PARAMETER SelfTest
  Check the statistics and rules, run nothing.
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
    [string[]]$Metrics,
    [switch]$NoGpuLock,
    [double]$GpuLockWait = 180,
    [string]$GpuLockTag = 'perf-gate',
    [string]$CompareExe,
    [string]$CompareVars,
    [string]$CompareEnv,
    [int]$Rounds = 7,
    [int]$MaxRounds = 15,
    [string]$AASummary,
    [double]$TargetHalfWidthPct = 3.0,
    [double]$RegimeTolerancePct = 25.0,
    [double]$ImprovedFloorPct = 1.0,
    [string[]]$Only,
    [switch]$AllowInconclusive,
    [switch]$Resume,
    [switch]$SelfTest
)

$ErrorActionPreference = 'Stop'
Set-StrictMode -Version 3.0

$repo = (Resolve-Path (Join-Path $PSScriptRoot '..\..')).Path
$benchDir = Join-Path $repo 'scripts\bench'
$fallbackHigherIsBetter = @('throughput_x', 'e2e_detected', 'gpu_load_duty', 'cpu_load_duty')
# Session-level slow-state check (P3B-A): these run-state metrics, compared with the session median.
$stateMetrics = @('state_sm_clock_mhz', 'state_mem_clock_mhz', 'state_speed_index_ms', 'state_host_probe_ms', 'state_cpu_effective_mhz')
$stateTolerance = 0.10

# ---- GPU timing lock (P3B rule R2; scripts/bench/gpu-lock.ps1) ----------------------------------
$lockScript = Join-Path $PSScriptRoot 'gpu-lock.ps1'
function Invoke-Timed {
    # Run `Body` (one timed harness run) while holding the dev rig's GPU timing lock; returns what
    # `Body` returns. -NoGpuLock (stub specs) runs it unlocked.
    param([string]$Label, [scriptblock]$Body)
    if ($NoGpuLock) { return & $Body }
    $tag = "$GpuLockTag $Label"
    & $lockScript -Acquire $tag -Wait $GpuLockWait -ExpectedMinutes 15 -OwnerPid $PID | Out-Host
    if ($LASTEXITCODE -ne 0) { throw "the GPU timing lock was not acquired within $GpuLockWait min ($tag)" }
    try {
        return & $Body
    } finally {
        & $lockScript -Release $tag | Out-Host
    }
}

function Split-List {
    param([string[]]$Items)
    # `pwsh -File run-perf-gate.ps1 -Metrics a,b` passes one string 'a,b': split it.
    return @($Items | Where-Object { $_ } | ForEach-Object { $_ -split ',' } | ForEach-Object { $_.Trim() } | Where-Object { $_ })
}

function Split-Pairs {
    # 'NAME=value;NAME2=value2' -> ordered hashtable (values may contain ',' and '=').
    param([string]$Text, [string]$What)
    $map = [ordered]@{}
    if (-not $Text) { return $map }
    foreach ($pair in ($Text -split ';')) {
        $p = $pair.Trim()
        if (-not $p) { continue }
        $i = $p.IndexOf('=')
        if ($i -lt 1) { throw "$What expects NAME=value;..., got '$p'" }
        $map[$p.Substring(0, $i).Trim()] = $p.Substring($i + 1)
    }
    return $map
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

# ---- paired statistics (mirror of bench::metrics::paired; golden values in -SelfTest) -----------
$script:T975Table = @(12.706205, 4.302653, 3.182446, 2.776445, 2.570582, 2.446912, 2.364624, 2.306004, 2.262157, 2.228139, 2.200985,
    2.178813, 2.160369, 2.144787, 2.131450, 2.119905, 2.109816, 2.100922, 2.093024, 2.085963, 2.079614, 2.073873,
    2.068658, 2.063899, 2.059539, 2.055529, 2.051831, 2.048407, 2.045230, 2.042272)

function Get-T975 {
    param([int]$Df)
    if ($Df -le 0) { return [double]::PositiveInfinity }
    if ($Df -le 30) { return [double]$script:T975Table[$Df - 1] }
    $z = 1.959963985
    $v = [double]$Df
    return $z + ([Math]::Pow($z, 3) + $z) / (4.0 * $v) + (5.0 * [Math]::Pow($z, 5) + 16.0 * [Math]::Pow($z, 3) + 3.0 * $z) / (96.0 * $v * $v)
}

function Get-SignedRankCdf {
    # P(T+ <= k), k = 0..n(n+1)/2, of the Wilcoxon signed-rank statistic (exact).
    param([int]$N)
    $m = [int]($N * ($N + 1) / 2)
    $c = New-Object 'double[]' ($m + 1)
    $c[0] = 1.0
    for ($rank = 1; $rank -le $N; $rank++) {
        for ($s = $m; $s -ge $rank; $s--) { $c[$s] += $c[$s - $rank] }
    }
    $total = [Math]::Pow(2.0, $N)
    $out = New-Object 'double[]' ($m + 1)
    $acc = 0.0
    for ($i = 0; $i -le $m; $i++) { $acc += $c[$i]; $out[$i] = $acc / $total }
    return , $out
}

function Get-Median {
    param([double[]]$V)
    $a = [double[]]$V.Clone()
    [Array]::Sort($a)
    $n = $a.Length
    if ($n -eq 0) { return [double]::NaN }
    if ($n % 2 -eq 1) { return $a[[int][Math]::Floor($n / 2)] }
    return 0.5 * ($a[[int]($n / 2) - 1] + $a[[int]($n / 2)])
}

function Get-ReportMetricMedian {
    # The median over `Reports` of one run's metric: the interleaved mode's targets read the median
    # of B's rounds, not one round (a single round's p95 carries the rig's 11-19 % per-pair spread).
    # Reports without the run or the metric (a round whose harness errored) are skipped; $null when
    # none has it.
    param([object[]]$Reports, [string]$Run, [string]$Metric)
    $vals = New-Object 'System.Collections.Generic.List[double]'
    foreach ($rep in $Reports) {
        if ($null -eq $rep) { continue }
        $r = @($rep.runs | Where-Object { $_.name -eq $Run }) | Select-Object -First 1
        if ($null -eq $r) { continue }
        $v = Get-Prop -Object $r.metrics -Name $Metric
        if ($null -ne $v) { $vals.Add([double]$v) }
    }
    if ($vals.Count -eq 0) { return $null }
    return Get-Median $vals.ToArray()
}

function Get-Walsh {
    param([double[]]$R)
    $w = New-Object 'System.Collections.Generic.List[double]'
    for ($i = 0; $i -lt $R.Length; $i++) {
        for ($j = $i; $j -lt $R.Length; $j++) { $w.Add(0.5 * ($R[$i] + $R[$j])) }
    }
    $a = $w.ToArray()
    [Array]::Sort($a)
    return , $a
}

function Get-PairedStats {
    # The statistics and verdict of the log ratios `R` (see the help). $null for an empty set.
    param([double[]]$R, [double]$LimitPct, [double]$ImprovedFloorPct = 1.0)
    $n = $R.Length
    if ($n -eq 0) { return $null }
    $sum = 0.0
    foreach ($x in $R) { $sum += $x }
    $mean = $sum / $n
    $sd = 0.0
    if ($n -gt 1) {
        $ss = 0.0
        foreach ($x in $R) { $ss += ($x - $mean) * ($x - $mean) }
        $sd = [Math]::Sqrt($ss / ($n - 1))
    }
    $half = if ($n -gt 1) { (Get-T975 ($n - 1)) * $sd / [Math]::Sqrt($n) } else { [double]::PositiveInfinity }
    $median = Get-Median $R
    $w = Get-Walsh $R
    $hl = Get-Median $w
    $cdf = Get-SignedRankCdf $n
    $k = 0
    while ($k -lt $cdf.Length -and $cdf[$k] -le (0.025 + 1e-12)) { $k++ }
    $wlo = $null; $whi = $null; $cov = $null
    if ($k -ge 1) {
        $crit = $k - 1
        $wlo = $w[$crit]; $whi = $w[$w.Length - 1 - $crit]; $cov = 1.0 - 2.0 * $cdf[$crit]
        $method = 'wilcoxon'; $lo = $wlo; $hi = $whi
    } else {
        $method = 't'; $lo = $mean - $half; $hi = $mean + $half
    }
    $pct = { param($x) ([Math]::Exp($x) - 1.0) * 100.0 }
    $medianPct = & $pct $median
    $verdict = if ($medianPct -gt $LimitPct -and $lo -gt 0) { 'REGRESSED' }
        elseif ($medianPct -gt $LimitPct) { 'INCONCLUSIVE' }
        elseif ($hi -lt 0 -and $medianPct -lt -$ImprovedFloorPct) { 'IMPROVED' }
        else { 'NEUTRAL' }
    $loPct = & $pct $lo
    $hiPct = & $pct $hi
    $halfWidth = if ([double]::IsInfinity($loPct) -or [double]::IsInfinity($hiPct)) { [double]::PositiveInfinity } else { 0.5 * ($hiPct - $loPct) }
    return [pscustomobject]@{
        n = $n; median = $median; median_pct = $medianPct; mean = $mean; sd = $sd
        t_lo = $mean - $half; t_hi = $mean + $half; hl = $hl
        w_lo = $wlo; w_hi = $whi; w_coverage = $cov
        method = $method; lo = $lo; hi = $hi; lo_pct = $loPct; hi_pct = $hiPct; half_width_pct = $halfWidth
        verdict = $verdict
    }
}

function Get-RoundsFor {
    # Pairs for a t-interval half-width of TargetPct at per-pair spread Sd, within Min..Max.
    param([double]$Sd, [double]$TargetPct, [int]$Min, [int]$Max)
    $target = [Math]::Log(1.0 + $TargetPct / 100.0)
    if (-not ($Sd -gt 0) -or [double]::IsNaN($Sd) -or -not ($target -gt 0)) { return [Math]::Min($Min, $Max) }
    $n = [Math]::Max($Min, 2)
    while ($n -lt $Max -and (Get-T975 ($n - 1)) * $Sd / [Math]::Sqrt($n) -gt $target) { $n++ }
    return [Math]::Min($n, $Max)
}

function Test-F0Metric {
    # Scored by a pitch analyzer (mirror of report::is_f0_metric); compute times are not.
    param([string]$Name)
    $n = $Name.ToLowerInvariant()
    foreach ($k in @('f0_rmse', 'pitch_error', 'vuv', 'cents', 'gpe', 'fpe', 'fed_f0', 'onset_lag')) {
        if ($n.Contains($k)) { return $true }
    }
    return $false
}

function Get-AnalyzerVersions {
    # Every analyzer version stamp in a report object (any depth; `analyzer_version` or a prefixed
    # one such as P3B-B's `details.scorers.f0_analyzer_version`, mirror of report::is_analyzer_stamp),
    # as one '+'-joined key.
    param($Obj)
    $found = New-Object 'System.Collections.Generic.SortedSet[string]'
    $stack = New-Object System.Collections.Stack
    $stack.Push($Obj)
    while ($stack.Count -gt 0) {
        $o = $stack.Pop()
        if ($null -eq $o -or $o -is [string] -or $o -is [ValueType]) { continue }
        if ($o -is [System.Management.Automation.PSCustomObject]) {
            foreach ($p in $o.PSObject.Properties) {
                if ($p.Name -eq 'analyzer_version' -or $p.Name.EndsWith('_analyzer_version')) {
                    if ($null -ne $p.Value) { [void]$found.Add([string]$p.Value) }
                } else {
                    $stack.Push($p.Value)
                }
            }
        } elseif ($o -is [System.Collections.IEnumerable]) {
            foreach ($x in $o) { $stack.Push($x) }
        }
    }
    if ($found.Count -eq 0) { return '<none>' }
    return ($found -join '+')
}

function Get-F0Refusal {
    # Why the F0 metrics of report sets A and B cannot be compared (mirror of
    # report::f0_comparison_refusal), or $null.
    param([object[]]$A, [object[]]$B)
    $sa = @($A | ForEach-Object { Get-AnalyzerVersions $_ } | Sort-Object -Unique)
    $sb = @($B | ForEach-Object { Get-AnalyzerVersions $_ } | Sort-Object -Unique)
    if ($sa.Count -gt 1 -or ($sa -join '|') -ne ($sb -join '|')) {
        return "F0 metrics come from different analyzers (before: $($sa -join ', '); after: $($sb -join ', ')): rescore both with one analyzer"
    }
    return $null
}

function Test-RegimeMismatch {
    # Whether the two sides of one pair ran in different GPU clock regimes (P3B-A): their run's
    # median SM clock (NVML), or without it the GPU speed index (DirectML / non-NVIDIA), more than
    # -RegimeTolerancePct apart. Such a pair compares two GPU states, not two builds: the regime-
    # matched statistics leave it out (information only; the verdict uses every pair).
    param($A, $B, [string]$Run, [double]$TolerancePct)
    $metric = {
        param($Report, [string]$Name)
        if ($null -eq $Report) { return $null }
        $r = $Report.runs | Where-Object { $_.name -eq $Run } | Select-Object -First 1
        if ($null -eq $r) { return $null }
        $v = Get-Prop -Object $r.metrics -Name $Name
        if ($null -eq $v) { return $null }
        return [double]$v
    }
    foreach ($m in @('state_sm_clock_mhz', 'state_speed_index_ms')) {
        $x = & $metric $A $m
        $y = & $metric $B $m
        if ($null -ne $x -and $null -ne $y -and $x -gt 0 -and $y -gt 0) {
            return ([Math]::Abs([Math]::Log($y / $x)) -gt [Math]::Log(1.0 + $TolerancePct / 100.0))
        }
    }
    return $false
}

function Get-PairedRows {
    # The interleaved comparison of one spec: every gated metric of every run of A's reports,
    # paired by round with B's (see the help for the statistics, the verdicts and REFUSED).
    param([string]$Stem, $Side)
    $as = @($Side.A)
    $bs = @($Side.B)
    $n = [Math]::Min($as.Count, $bs.Count)
    $firstA = $as | Where-Object { $null -ne $_ } | Select-Object -First 1
    if ($null -eq $firstA) { return @() }
    $higher = Get-HigherIsBetter -Reports ($as + $bs)
    $refusal = Get-F0Refusal -A @($as | Where-Object { $null -ne $_ }) -B @($bs | Where-Object { $null -ne $_ })
    $out = @()
    foreach ($run in $firstA.runs) {
        $names = @($run.metrics.PSObject.Properties | ForEach-Object { $_.Name } | Where-Object { Test-Gated -Name $_ })
        # Pairs (rounds) whose sides ran in different GPU clock regimes (Test-RegimeMismatch).
        $mismatched = @{}
        for ($i = 0; $i -lt $n; $i++) {
            if (Test-RegimeMismatch -A $as[$i] -B $bs[$i] -Run $run.name -TolerancePct $RegimeTolerancePct) { $mismatched[$i] = $true }
        }
        foreach ($m in $names) {
            $av = New-Object 'System.Collections.Generic.List[double]'
            $bv = New-Object 'System.Collections.Generic.List[double]'
            $pairIndex = New-Object 'System.Collections.Generic.List[int]'
            $missing = 0
            for ($i = 0; $i -lt $n; $i++) {
                $x = Get-RunMetric -Report $as[$i] -Run $run.name -Metric $m
                $y = Get-RunMetric -Report $bs[$i] -Run $run.name -Metric $m
                if ($null -ne $x -and $null -ne $y) { $av.Add($x); $bv.Add($y); $pairIndex.Add($i) }
                elseif ($null -ne $x) { $missing++ }
            }
            $row = [ordered]@{
                spec = $Stem; run = $run.name; metric = $m; gated = $true; n = $av.Count
                a_median = $null; b_median = $null; method = $null; median_pct = $null; lo_pct = $null; hi_pct = $null
                half_width_pct = $null; sd = $null; verdict = $null; note = ''
                regime_mismatched = 0; matched_n = $null; matched_median_pct = $null; matched_lo_pct = $null; matched_hi_pct = $null
                matched_half_width_pct = $null; matched_verdict = $null
            }
            if ($av.Count -gt 0) {
                $row.a_median = [math]::Round((Get-Median $av.ToArray()), 4)
                $row.b_median = [math]::Round((Get-Median $bv.ToArray()), 4)
            }
            $isHigher = $higher -contains $m
            if ((Test-F0Metric $m) -and $null -ne $refusal) {
                $row.verdict = 'REFUSED'; $row.note = $refusal
            } elseif ($missing -gt 0 -and $av.Count -lt 2) {
                $row.verdict = 'MISSING'; $row.note = "B did not measure it in $missing round(s)"
            } elseif ($av.Count -lt 2) {
                $row.verdict = 'MISSING'; $row.note = 'fewer than 2 pairs'
            } else {
                $r = New-Object 'System.Collections.Generic.List[double]'
                $positive = (@($av | Where-Object { $_ -le 0 }).Count -eq 0) -and (@($bv | Where-Object { $_ -le 0 }).Count -eq 0)
                if ($positive) {
                    for ($i = 0; $i -lt $av.Count; $i++) {
                        if ($isHigher) { $r.Add([Math]::Log($av[$i] / $bv[$i])) } else { $r.Add([Math]::Log($bv[$i] / $av[$i])) }
                    }
                } else {
                    # Zeros: the difference relative to A's median (at least -MinDelta).
                    $scale = [Math]::Max([Math]::Abs((Get-Median $av.ToArray())), $MinDelta)
                    for ($i = 0; $i -lt $av.Count; $i++) {
                        if ($isHigher) { $r.Add(($av[$i] - $bv[$i]) / $scale) } else { $r.Add(($bv[$i] - $av[$i]) / $scale) }
                    }
                    $row.note = "linear (values <= 0; relative to max(|A median|, $MinDelta))"
                }
                $s = Get-PairedStats -R $r.ToArray() -LimitPct $MaxRegressionPct -ImprovedFloorPct $ImprovedFloorPct
                $row.method = $s.method
                $row.median_pct = [math]::Round($s.median_pct, 2)
                $row.lo_pct = [math]::Round($s.lo_pct, 2)
                $row.hi_pct = [math]::Round($s.hi_pct, 2)
                $row.half_width_pct = if ([double]::IsInfinity($s.half_width_pct)) { $null } else { [math]::Round($s.half_width_pct, 2) }
                $row.sd = [math]::Round($s.sd, 5)
                $row.verdict = $s.verdict
                if ($missing -gt 0) { $row.note = ($row.note + " B did not measure it in $missing round(s)").Trim() }
                # Information only: the same statistics without the pairs whose sides ran in
                # different GPU clock regimes (the gate's resolution while the GPU holds one state).
                $keep = New-Object 'System.Collections.Generic.List[double]'
                for ($j = 0; $j -lt $r.Count; $j++) { if (-not $mismatched.ContainsKey($pairIndex[$j])) { $keep.Add($r[$j]) } }
                $row.regime_mismatched = $r.Count - $keep.Count
                if ($row.regime_mismatched -gt 0 -and $keep.Count -ge 2) {
                    $sm = Get-PairedStats -R $keep.ToArray() -LimitPct $MaxRegressionPct -ImprovedFloorPct $ImprovedFloorPct
                    $row.matched_n = $keep.Count
                    $row.matched_median_pct = [math]::Round($sm.median_pct, 2)
                    $row.matched_lo_pct = [math]::Round($sm.lo_pct, 2)
                    $row.matched_hi_pct = [math]::Round($sm.hi_pct, 2)
                    $row.matched_half_width_pct = if ([double]::IsInfinity($sm.half_width_pct)) { $null } else { [math]::Round($sm.half_width_pct, 2) }
                    $row.matched_verdict = $sm.verdict
                }
            }
            $out += [pscustomobject]$row
        }
    }
    return , $out
}

function Test-NeedsMoreRounds {
    # An INCONCLUSIVE gated row below the round cap asks for two more rounds (ABBA balance).
    param([object[]]$Rows, [int]$Round, [int]$Max)
    if ($Round -ge $Max) { return $false }
    return @($Rows | Where-Object { $_.gated -and $_.verdict -eq 'INCONCLUSIVE' }).Count -gt 0
}

function Invoke-SelfTest {
    $failed = $false
    $check = {
        param([bool]$Ok, [string]$What)
        if ($Ok) { Write-Host "  ok   $What" } else { Write-Host "  FAIL $What"; $script:selfTestFailed = $true }
    }
    $script:selfTestFailed = $false
    $close = { param($a, $b) [Math]::Abs([double]$a - [double]$b) -lt 1e-8 }
    $sets = [ordered]@{
        aa    = @(0.010, -0.020, 0.015, -0.005, 0.000, 0.020, -0.010)
        reg   = @(0.090, 0.100, 0.110, 0.095, 0.105, 0.098, 0.102)
        inc   = @(0.080, 0.070, -0.050, 0.060, -0.040, 0.090, 0.065)
        imp   = @(-0.100, -0.080, -0.120, -0.090, -0.110, -0.095, -0.105)
        small = @(0.08, 0.09, 0.07, 0.085, 0.075)
    }
    $golden = @{
        aa    = @{ verdict = 'NEUTRAL'; method = 'wilcoxon'; lo = -0.0125; hi = 0.015; hl = 0.00125; median_pct = 0.0; sd = 0.014351041 }
        reg   = @{ verdict = 'REGRESSED'; method = 'wilcoxon'; lo = 0.094; hi = 0.106; hl = 0.1; median_pct = 10.517091808; sd = 0.006557439 }
        inc   = @{ verdict = 'INCONCLUSIVE'; method = 'wilcoxon'; lo = -0.04; hi = 0.08; hl = 0.06125; median_pct = 6.715902438; sd = 0.058482802 }
        imp   = @{ verdict = 'IMPROVED'; method = 'wilcoxon'; lo = -0.1125; hi = -0.0875; hl = -0.1; median_pct = -9.516258196; sd = 0.013228757 }
        small = @{ verdict = 'REGRESSED'; method = 't'; lo = 0.070183785; hi = 0.089816215; hl = 0.08; median_pct = 8.328706767; sd = 0.007905694 }
    }
    foreach ($k in $sets.Keys) {
        $s = Get-PairedStats -R ([double[]]$sets[$k]) -LimitPct 5.0
        $g = $golden[$k]
        & $check ($s.verdict -eq $g.verdict -and $s.method -eq $g.method) "$k : $($s.verdict) by $($s.method)"
        foreach ($f in @('lo', 'hi', 'hl', 'median_pct', 'sd')) {
            & $check (& $close $s.$f $g[$f]) ("$k : {0} = {1:N9} (golden {2})" -f $f, $s.$f, $g[$f])
        }
    }
    $aa = Get-PairedStats -R ([double[]]$sets.aa) -LimitPct 5.0
    & $check (& $close $aa.w_coverage 0.953125) 'exact coverage at 7 pairs = 0.953125'
    # IMPROVED needs an effect-size floor: an interval entirely below 0 around a -0.5 % median is NEUTRAL.
    $tiny = [double[]]@(-0.004, -0.006, -0.005, -0.0045, -0.0055, -0.005, -0.0048)
    $tinyStats = Get-PairedStats -R $tiny -LimitPct 5.0
    & $check ($tinyStats.hi -lt 0 -and $tinyStats.verdict -eq 'NEUTRAL') "a clear but negligible -0.5 % shift is NEUTRAL ($($tinyStats.verdict), hi $($tinyStats.hi))"
    & $check ((Get-PairedStats -R $tiny -LimitPct 5.0 -ImprovedFloorPct 0.0).verdict -eq 'IMPROVED') 'without the floor the same pairs read IMPROVED'
    & $check ((Get-PairedStats -R ([double[]]$sets.imp) -LimitPct 5.0).verdict -eq 'IMPROVED') 'a -9.5 % shift stays IMPROVED'
    # The interleaved targets read the median of B's rounds, not the last one.
    $round = { param($v) [pscustomobject]@{ runs = @([pscustomobject]@{ name = 'offline'; metrics = [pscustomobject]@{ call_ms_p95 = $v } }, [pscustomobject]@{ name = 'paced'; metrics = [pscustomobject]@{ call_ms_p95 = 99.0 } }) } }
    $bRounds = @((& $round 5.4), (& $round 6.6), $null, (& $round 5.6), (& $round 5.8), (& $round 9.0))
    & $check ((& $close (Get-ReportMetricMedian -Reports $bRounds -Run 'offline' -Metric 'call_ms_p95') 5.8)) 'the target value is the median of the B rounds (5.8), not the last round (9.0)'
    & $check ($null -eq (Get-ReportMetricMedian -Reports $bRounds -Run 'offline' -Metric 'nope')) 'a metric no round has is not measured'
    & $check ($null -eq (Get-ReportMetricMedian -Reports @($null) -Run 'offline' -Metric 'call_ms_p95')) 'no report: not measured'
    & $check ((& $close (Get-ReportMetricMedian -Reports @((& $round 7.5)) -Run 'offline' -Metric 'call_ms_p95') 7.5)) 'a single report is its own median'
    $c7 = Get-SignedRankCdf 7
    & $check ((& $close $c7[2] 0.0234375) -and (& $close $c7[3] 0.0390625) -and $c7.Length -eq 29) 'signed-rank distribution n = 7'
    & $check ((& $close (Get-T975 6) 2.446912) -and [Math]::Abs((Get-T975 40) - 2.021075) -lt 1e-3) 't quantiles'
    $n = Get-RoundsFor -Sd 0.06 -TargetPct 3 -Min 5 -Max 40
    & $check ($n -ge 15 -and $n -le 19) "rounds for a 6 % spread at +-3 %: $n"
    & $check ((Get-RoundsFor -Sd 0.06 -TargetPct 3 -Min 5 -Max 10) -eq 10) 'rounds are capped'
    # INCONCLUSIVE grows the rounds until the cap; REFUSED / ungated rows do not.
    $rows = @([pscustomobject]@{ gated = $true; verdict = 'INCONCLUSIVE' }, [pscustomobject]@{ gated = $true; verdict = 'NEUTRAL' })
    & $check (Test-NeedsMoreRounds -Rows $rows -Round 7 -Max 15) 'an inconclusive row adds rounds'
    & $check (-not (Test-NeedsMoreRounds -Rows $rows -Round 15 -Max 15)) 'never past -MaxRounds'
    & $check (-not (Test-NeedsMoreRounds -Rows @([pscustomobject]@{ gated = $false; verdict = 'INCONCLUSIVE' }) -Round 7 -Max 15)) 'ungated rows do not'
    # Analyzer refusal (mirror of report::f0_comparison_refusal).
    foreach ($m in @('f0_rmse_cents', 'gross_pitch_error', 'vuv_error', 'fine_rmse_cents', 'fed_gpe')) { & $check (Test-F0Metric $m) "$m is an F0 metric" }
    foreach ($m in @('f0_ms_p95', 'call_ms_p95', 'speaker_cosine', 'gpu_pitch_ms_p50')) { & $check (-not (Test-F0Metric $m)) "$m is not" }
    $v1 = '{"runs":[{"details":{"scoring":{}}}]}' | ConvertFrom-Json
    $v2a = '{"runs":[{"details":{"scoring":{"analyzer_version":"rmvpe-7f3a"}}}]}' | ConvertFrom-Json
    $v2b = '{"engine":{"analyzer_version":"rmvpe-91c0"}}' | ConvertFrom-Json
    & $check ((Get-AnalyzerVersions $v2a) -eq 'rmvpe-7f3a') 'the stamp is found at any depth'
    $v2c = '{"runs":[{"details":{"scorers":{"f0_analyzer_version":"rmvpe-7f3a"}}}]}' | ConvertFrom-Json
    & $check ($null -eq (Get-F0Refusal -A @($v2a) -B @($v2c))) "P3B-B's f0_analyzer_version is the same stamp"
    & $check ($null -ne (Get-F0Refusal -A @($v1) -B @($v2c))) 'unstamped vs f0_analyzer_version is refused'
    & $check ($null -eq (Get-F0Refusal -A @($v1, $v1) -B @($v1))) 'two unstamped sides compare'
    & $check ($null -eq (Get-F0Refusal -A @($v2a) -B @($v2a, $v2a))) 'one analyzer compares'
    $why = Get-F0Refusal -A @($v1) -B @($v2a)
    & $check ($null -ne $why -and $why.Contains('<none>') -and $why.Contains('rmvpe-7f3a')) 'unstamped vs stamped is refused'
    & $check ($null -ne (Get-F0Refusal -A @($v2a) -B @($v2b))) 'two analyzers are refused'
    & $check ($null -ne (Get-F0Refusal -A @($v2a, $v2b) -B @($v2a, $v2b))) 'mixed stamps within a side are refused'
    $flags = Get-OffMedian -Values @(10.0, 10.4, $null, 9.8, 12.0, 10.1) -Tolerance 0.10
    & $check (($flags -join ',') -eq '4') "off-median flags: $($flags -join ',')"
    # GPU clock regimes of a pair: the SM clock decides, the speed index stands in without NVML.
    $rep = { param($sm, $idx) [pscustomobject]@{ runs = @([pscustomobject]@{ name = 'r'; metrics = [pscustomobject]@{ state_sm_clock_mhz = $sm; state_speed_index_ms = $idx } }) } }
    & $check (Test-RegimeMismatch -A (& $rep 2790 2.3) -B (& $rep 1455 2.3) -Run 'r' -TolerancePct 25) 'SM 2790 vs 1455 MHz: two regimes'
    & $check (-not (Test-RegimeMismatch -A (& $rep 1342 2.3) -B (& $rep 1117 5.0) -Run 'r' -TolerancePct 25)) 'SM 1342 vs 1117 MHz: one regime (the SM clock decides over the speed index)'
    & $check (Test-RegimeMismatch -A (& $rep $null 2.3) -B (& $rep $null 4.6) -Run 'r' -TolerancePct 25) 'without NVML the speed index decides'
    & $check (-not (Test-RegimeMismatch -A (& $rep $null $null) -B (& $rep 1455 2.3) -Run 'r' -TolerancePct 25)) 'no state on one side: no mismatch'
    if ($script:selfTestFailed) { Write-Host 'perf-gate self-test FAILED'; return $false }
    Write-Host 'perf-gate self-test passed'
    return $true
}

function Get-OffMedian {
    # Indices of values more than Tolerance off the median of the non-null values.
    param([object[]]$Values, [double]$Tolerance)
    $vals = @($Values | Where-Object { $null -ne $_ } | ForEach-Object { [double]$_ })
    if ($vals.Count -eq 0) { return @() }
    $med = Get-Median ([double[]]$vals)
    if ($med -eq 0) { return @() }
    $out = @()
    for ($i = 0; $i -lt $Values.Count; $i++) {
        if ($null -ne $Values[$i] -and [Math]::Abs(([double]$Values[$i] - $med) / $med) -gt $Tolerance) { $out += $i }
    }
    return , $out
}

if ($SelfTest) {
    if (Invoke-SelfTest) { exit 0 } else { exit 1 }
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
$script:overrideVars = @{}

function Resolve-Var {
    param([string]$Name, [hashtable]$TemplateVars)
    if ($script:overrideVars.ContainsKey($Name)) { return $script:overrideVars[$Name] }
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

function Test-Selected {
    param([string]$Name)
    $only = @(Split-List $Only)
    if ($only.Count -eq 0) { return $true }
    foreach ($p in $only) { if ($Name -like $p) { return $true } }
    return $false
}

function Get-ProfileSpecs {
    # The profile's specs and materialized templates in `Dir` (with `Overrides` on top of every
    # other variable source). Returns @{ specs; skipped; warnings }.
    param([string]$Dir, [hashtable]$Overrides)
    $script:overrideVars = $Overrides
    $list = @(); $skipped = @(); $warn = @()
    foreach ($s in @(Get-Prop -Object $profileDoc -Name 'specs')) {
        if ($s -and (Test-Selected ([System.IO.Path]::GetFileNameWithoutExtension($s)))) { $list += $(if ([System.IO.Path]::IsPathRooted($s)) { $s } else { Join-Path $benchDir $s }) }
    }
    foreach ($t in @(Get-Prop -Object $profileDoc -Name 'templates')) {
        if ($null -eq $t) { continue }
        if (-not (Test-Selected ([string]$t.name))) { continue }
        $tvars = @{}
        $own = Get-Prop -Object $t -Name 'vars'
        if ($null -ne $own) { foreach ($p in $own.PSObject.Properties) { $tvars[$p.Name.ToUpperInvariant()] = [string]$p.Value } }
        $tvars['NAME'] = [string]$t.name
        $missing = @(@(Get-Prop -Object $t -Name 'requires') | Where-Object { $_ -and -not (Resolve-Var -Name $_ -TemplateVars $tvars) })
        if ($missing.Count -gt 0) {
            $msg = "template $($t.name): no value for $($missing -join ', ') (-Var, -Slot / -Clip, or SIGIL_BENCH_<NAME>); skipped"
            if ($Strict) { throw $msg }
            $warn += $msg
            $skipped += [string]$t.name
            continue
        }
        $text = Get-Content -LiteralPath $(if ([System.IO.Path]::IsPathRooted($t.file)) { $t.file } else { Join-Path $benchDir $t.file }) -Raw
        # `*_JSON` variables are inserted as JSON (objects, numbers); the rest as string contents.
        $text = [regex]::Replace($text, '\$\{([A-Z0-9_]+)\}', {
                param($m)
                $k = $m.Groups[1].Value
                $v = Resolve-Var -Name $k -TemplateVars $tvars
                if ($k.EndsWith('_JSON')) { if ($v) { $v } else { 'null' } } else { Get-JsonString $v }
            })
        New-Item -ItemType Directory -Force -Path $Dir | Out-Null
        $path = Join-Path $Dir ($t.name + '.json')
        Set-Content -LiteralPath $path -Value $text -Encoding utf8
        $list += $path
    }
    $script:overrideVars = @{}
    return @{ specs = $list; skipped = $skipped; warnings = $warn }
}

$exePath = Resolve-Exe -Given $Exe
if (-not $exePath) {
    Write-Error 'No Sigil executable found. Build one (cargo build --release --manifest-path src-tauri/Cargo.toml) or pass -Exe.'
    exit 2
}
$compareMode = [bool]($CompareExe -or $CompareVars -or $CompareEnv)
$compareExePath = $exePath
if ($CompareExe) {
    $compareExePath = Resolve-Exe -Given $CompareExe
    if (-not $compareExePath) { Write-Error "compare executable not found: $CompareExe"; exit 2 }
}
if ($compareMode -and $Baseline) { Write-Error '-Baseline and the interleaved mode (-CompareExe / -CompareVars / -CompareEnv) are exclusive'; exit 2 }
if ($Rounds -lt 2) { $Rounds = 2 }
if ($MaxRounds -lt $Rounds) { $MaxRounds = $Rounds }
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
$specsB = $null
if (-not $Specs -or $Specs.Count -eq 0) {
    $Specs = @()
    if ($null -ne $profileDoc) {
        try {
            $r = Get-ProfileSpecs -Dir (Join-Path $OutRoot 'specs') -Overrides @{}
        } catch {
            Write-Error "$_"
            exit 2
        }
        $Specs = @($r.specs); $skippedTemplates = @($r.skipped); $warnings += @($r.warnings)
        if ($compareMode -and $CompareVars) {
            $over = @{}
            $pairs = Split-Pairs -Text $CompareVars -What '-CompareVars'
            foreach ($k in $pairs.Keys) { $over[$k.ToUpperInvariant()] = $pairs[$k] }
            $rb = Get-ProfileSpecs -Dir (Join-Path $OutRoot 'specs-b') -Overrides $over
            $specsB = @($rb.specs)
        }
    } else {
        $Specs = @(Get-ChildItem -Path (Join-Path $benchDir 'specs') -Filter '*.json' | Sort-Object Name | ForEach-Object { $_.FullName } | Where-Object { Test-Selected ([System.IO.Path]::GetFileNameWithoutExtension($_)) })
    }
} else {
    $Specs = @($Specs | Where-Object { Test-Selected ([System.IO.Path]::GetFileNameWithoutExtension($_)) })
}
if ($compareMode -and $CompareVars -and $null -eq $specsB) {
    Write-Error '-CompareVars needs a -Profile with templates (the variables are template variables)'
    exit 2
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
if ($compareMode) {
    Write-Host "  mode:     interleaved A/B ($Rounds..$MaxRounds rounds, ABBA)"
    Write-Host "  A:        $compareExePath"
    Write-Host "  B:        $exePath$(if ($CompareVars) { "  vars: $CompareVars" })$(if ($CompareEnv) { "  env: $CompareEnv" })"
}
Write-Host "  out:      $OutRoot"
if ($Profile) { Write-Host "  profile:  $Profile" }
if ($env:SIGIL_APP_DATA_DIR) { Write-Host "  app data: $($env:SIGIL_APP_DATA_DIR)" }
if ($Baseline) { Write-Host "  baseline: $Baseline (max regression $MaxRegressionPct %, min delta $MinDelta)" }

function Invoke-Harness {
    # One harness run; returns its exit code (-1 when none). Release builds use the GUI subsystem:
    # PowerShell only waits for them (and sets $LASTEXITCODE) when their output is piped. Stderr
    # carries progress lines, which must not become terminating errors under Windows PowerShell
    # 5.1's 'Stop' preference.
    param([string]$ExePath, [string]$SpecPath, [string]$Out, [System.Collections.IDictionary]$Env)
    $saved = @{}
    if ($null -ne $Env) {
        foreach ($k in $Env.Keys) {
            $saved[$k] = [Environment]::GetEnvironmentVariable($k)
            [Environment]::SetEnvironmentVariable($k, [string]$Env[$k])
        }
    }
    $prevPreference = $ErrorActionPreference
    $ErrorActionPreference = 'Continue'
    try {
        $global:LASTEXITCODE = $null
        & $ExePath --headless bench $SpecPath --out $Out 2>&1 | ForEach-Object { Write-Host "$_" }
        $code = $LASTEXITCODE
    } finally {
        $ErrorActionPreference = $prevPreference
        foreach ($k in $saved.Keys) { [Environment]::SetEnvironmentVariable($k, $saved[$k]) }
    }
    if ($null -eq $code) { $code = -1 }
    return $code
}

function Read-Report {
    param([string]$Path)
    if (-not (Test-Path -LiteralPath $Path)) { return $null }
    return Get-Content -LiteralPath $Path -Raw | ConvertFrom-Json
}

function Get-RunMetric {
    param($Report, [string]$Run, [string]$Metric)
    if ($null -eq $Report) { return $null }
    $r = $Report.runs | Where-Object { $_.name -eq $Run } | Select-Object -First 1
    if ($null -eq $r) { return $null }
    $v = Get-Prop -Object $r.metrics -Name $Metric
    if ($null -eq $v) { return $null }
    return [double]$v
}

function Get-HigherIsBetter {
    param($Reports)
    $h = @($fallbackHigherIsBetter)
    foreach ($rep in $Reports) {
        if ($null -eq $rep) { continue }
        foreach ($run in $rep.runs) { $h += @(Get-Prop -Object $run -Name 'higher_is_better') }
    }
    return $h
}

$results = @()
$comparisons = @()
$reports = @{}
$pairedRows = @()
$stateFlags = @()
$roundsRun = [ordered]@{}

if ($compareMode) {
    # ---- interleaved A/B ----------------------------------------------------------------------------
    $envB = Split-Pairs -Text $CompareEnv -What '-CompareEnv'
    $aa = $null
    if ($AASummary) {
        if (-not (Test-Path -LiteralPath $AASummary -PathType Leaf)) { Write-Error "A/A summary not found: $AASummary"; exit 2 }
        $aa = Get-Content -LiteralPath $AASummary -Raw | ConvertFrom-Json
    }
    $sideReports = @{}
    for ($si = 0; $si -lt $Specs.Count; $si++) {
        $specA = (Resolve-Path -LiteralPath $Specs[$si]).Path
        $specB = if ($null -ne $specsB) { (Resolve-Path -LiteralPath $specsB[$si]).Path } else { $specA }
        $stem = [System.IO.Path]::GetFileNameWithoutExtension($specA)
        # Rounds from the A/A spread of this spec's gated rows.
        $target = $Rounds
        if ($null -ne $aa) {
            $aaRows = @(@(Get-Prop -Object $aa -Name 'paired') | Where-Object { $null -ne $_ -and $_.gated -and $null -ne $_.sd })
            $sds = @($aaRows | Where-Object { $_.spec -eq $stem } | ForEach-Object { [double]$_.sd })
            # A spec the A/A did not run takes the widest spread of the A/A (conservative).
            if ($sds.Count -eq 0) { $sds = @($aaRows | ForEach-Object { [double]$_.sd }) }
            if ($sds.Count -gt 0) {
                $maxSd = ($sds | Measure-Object -Maximum).Maximum
                $target = Get-RoundsFor -Sd $maxSd -TargetPct $TargetHalfWidthPct -Min $Rounds -Max $MaxRounds
                if ($target % 2 -eq 1 -and $target -lt $MaxRounds) { $target++ }
            }
        }
        Write-Host ""
        Write-Host "== $stem (A/B, $target round(s))"
        $sideReports[$stem] = @{ A = @(); B = @(); codes = @(); reused = 0 }
        $round = 0
        while ($true) {
            while ($round -lt $target) {
                $round++
                $order = if ($round % 2 -eq 1) { @('A', 'B') } else { @('B', 'A') }
                $reuse = $Resume -and @('A', 'B' | Where-Object { Test-Path -LiteralPath (Join-Path $OutRoot (Join-Path $stem (Join-Path "$_-r$round" 'report.json'))) }).Count -eq 2
                if ($reuse) {
                    Write-Host "-- $stem round ${round}: reusing the existing A and B reports (-Resume)"
                    $sideReports[$stem].reused = $sideReports[$stem].reused + 1
                }
                $pairCodes = if ($reuse) { @{ A = 0; B = 0 } } else { Invoke-Timed -Label "$stem r$round" -Body {
                    $codes = @{}
                    foreach ($side in $order) {
                        $out = Join-Path $OutRoot (Join-Path $stem "$side-r$round")
                        Write-Host "-- $stem round $round side $side"
                        if ($side -eq 'A') { $codes[$side] = Invoke-Harness -ExePath $compareExePath -SpecPath $specA -Out $out -Env $null }
                        else { $codes[$side] = Invoke-Harness -ExePath $exePath -SpecPath $specB -Out $out -Env $envB }
                    }
                    $codes
                } }
                foreach ($side in @('A', 'B')) {
                    $rep = Read-Report -Path (Join-Path $OutRoot (Join-Path $stem (Join-Path "$side-r$round" 'report.json')))
                    $sideReports[$stem][$side] += , $rep
                }
                $sideReports[$stem].codes += , $pairCodes
            }
            $rowsNow = Get-PairedRows -Stem $stem -Side $sideReports[$stem]
            if (Test-NeedsMoreRounds -Rows $rowsNow -Round $round -Max $MaxRounds) {
                $target = [Math]::Min($round + 2, $MaxRounds)
                Write-Host "   INCONCLUSIVE rows: $round -> $target rounds"
                continue
            }
            break
        }
        $roundsRun[$stem] = $round
        if ($sideReports[$stem].reused -gt 0) { $warnings += "${stem}: $($sideReports[$stem].reused) round(s) reused from an earlier run (-Resume)" }
        $pairedRows += @($rowsNow)
        # Rows whose interval is wider than the regression limit cannot tell a real regression from
        # noise at this round count (a paced row, a stage metric): say so rather than let a NEUTRAL
        # read as "no change".
        $wide = @($rowsNow | Where-Object { $_.gated -and $null -ne $_.half_width_pct -and $_.half_width_pct -gt $MaxRegressionPct } | Sort-Object -Property half_width_pct -Descending)
        if ($wide.Count -gt 0) {
            $worst = $wide[0]
            $warnings += ("${stem}: {0} gated row(s) have a 95 % interval wider than +-{1} % after {2} rounds (worst: {3} {4} +-{5} %): the gate cannot resolve a {1} % change there; size the rounds from an A/A session (-AASummary), or read those rows as unresolved" -f $wide.Count, $MaxRegressionPct, $round, $worst.run, $worst.metric, $worst.half_width_pct)
        }
        $missingReports = @($sideReports[$stem].A + $sideReports[$stem].B | Where-Object { $null -eq $_ }).Count
        $failedHarness = @($sideReports[$stem].codes | ForEach-Object { $_.Values } | Where-Object { $_ -ne 0 }).Count
        $status = if ($missingReports -gt 0) { 'ERROR' } elseif ($failedHarness -gt 0) { 'FAIL' } else { 'PASS' }
        $results += [pscustomobject]@{ spec = $stem; status = $status; exit_code = $failedHarness; report = (Join-Path $OutRoot $stem) }
        if ($missingReports -gt 0) { $warnings += "${stem}: $missingReports report(s) missing (the harness errored)" }
        if ($failedHarness -gt 0) { $warnings += "${stem}: $failedHarness harness run(s) exited non-zero (spec thresholds); the A/B comparison stands" }
        # Every B round feeds the targets (their value is the median over the rounds, Get-Metric).
        $allB = @($sideReports[$stem].B | Where-Object { $null -ne $_ })
        if ($allB.Count -gt 0) { $reports[$stem] = $allB }
        # Process QoS of the two sides.
        $qa = @($sideReports[$stem].A | Where-Object { $null -ne $_ } | ForEach-Object { Get-Prop (Get-Prop (Get-Prop (Get-Prop $_.engine 'attribution') 'host') 'process_qos') 'execution_speed_throttled' } | ForEach-Object { "$_" } | Sort-Object -Unique)
        $qb = @($sideReports[$stem].B | Where-Object { $null -ne $_ } | ForEach-Object { Get-Prop (Get-Prop (Get-Prop (Get-Prop $_.engine 'attribution') 'host') 'process_qos') 'execution_speed_throttled' } | ForEach-Object { "$_" } | Sort-Object -Unique)
        if (($qa -join ',') -ne ($qb -join ',')) {
            $warnings += "${stem}: the sides ran under different process QoS (A: $($qa -join ',' ) / B: $($qb -join ',')); an exe without the QoS opt-out needs -Var QOS=none on both sides"
        }
    }
    # Session-level slow-state flags: each state metric of every report against the session median
    # of the same spec and run (A and B together).
    foreach ($stem in $sideReports.Keys) {
        $all = @()
        foreach ($side in @('A', 'B')) {
            $k = 0
            foreach ($rep in $sideReports[$stem][$side]) { $k++; if ($null -ne $rep) { $all += [pscustomobject]@{ side = $side; round = $k; report = $rep } } }
        }
        if ($all.Count -eq 0) { continue }
        $runNames = @($all[0].report.runs | ForEach-Object { $_.name })
        foreach ($runName in $runNames) {
            foreach ($sm in $stateMetrics) {
                $vals = @($all | ForEach-Object { Get-RunMetric -Report $_.report -Run $runName -Metric $sm })
                if (@($vals | Where-Object { $null -ne $_ }).Count -lt 3) { continue }
                $med = Get-Median ([double[]]@($vals | Where-Object { $null -ne $_ }))
                foreach ($i in (Get-OffMedian -Values $vals -Tolerance $stateTolerance)) {
                    $stateFlags += [pscustomobject]@{
                        spec = $stem; side = $all[$i].side; round = $all[$i].round; run = $runName; metric = $sm
                        value = [math]::Round([double]$vals[$i], 3); session_median = [math]::Round($med, 3)
                        off_pct = [math]::Round(100.0 * ([double]$vals[$i] - $med) / $med, 1)
                    }
                }
            }
        }
    }
    # Stamp the flags into the flagged reports (report.json gains `gate_state_flags`).
    foreach ($g in ($stateFlags | Group-Object -Property spec, side, round)) {
        $f = $g.Group[0]
        $path = Join-Path $OutRoot (Join-Path $f.spec (Join-Path "$($f.side)-r$($f.round)" 'report.json'))
        try {
            $doc = Get-Content -LiteralPath $path -Raw | ConvertFrom-Json
            $doc | Add-Member -NotePropertyName 'gate_state_flags' -NotePropertyValue @($g.Group | Select-Object run, metric, value, session_median, off_pct) -Force
            $doc | ConvertTo-Json -Depth 100 | Set-Content -LiteralPath $path -Encoding utf8
        } catch {
            $warnings += "cannot stamp the state flags into ${path}: $_"
        }
    }
} else {
    foreach ($spec in $Specs) {
        $specPath = (Resolve-Path -LiteralPath $spec).Path
        $stem = [System.IO.Path]::GetFileNameWithoutExtension($specPath)
        $out = Join-Path $OutRoot $stem
        Write-Host ""
        Write-Host "== $stem"
        $code = Invoke-Timed -Label $stem -Body { Invoke-Harness -ExePath $exePath -SpecPath $specPath -Out $out -Env $null }
        $reportPath = Join-Path $out 'report.json'
        $report = Read-Report -Path $reportPath
        if ($null -ne $report) { $reports[$stem] = @($report) }
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
}

# ---- targets --------------------------------------------------------------------------------------
function Get-Metric {
    # A target's metric: the median over the spec's reports (the interleaved mode's B rounds; the
    # one report of a plain run is its own median).
    param($Ref)
    $r = $reports[[string]$Ref.spec]
    if ($null -eq $r) { return $null }
    return Get-ReportMetricMedian -Reports @($r) -Run ([string]$Ref.run) -Metric ([string]$Ref.metric)
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
        $skipped = @($specsOfTarget | Where-Object { $skippedTemplates -contains $_ -or -not (Test-Selected $_) }).Count -gt 0
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
$pairedFailures = @($pairedRows | Where-Object { $_.gated -and (@('REGRESSED', 'REFUSED', 'MISSING') -contains $_.verdict -or ($_.verdict -eq 'INCONCLUSIVE' -and -not $AllowInconclusive)) })
# A gate that compared nothing (a -Metrics pattern that matches no metric) must not pass.
$comparedNothing = ($Baseline -and ($comparisons.Count -eq 0)) -or ($compareMode -and $pairedRows.Count -eq 0)
if ($comparedNothing) {
    $warnings += "no metric matched -Metrics '$($script:gatedPatterns -join "', '")': nothing was compared"
}
# In the interleaved mode a spec threshold failing (exit 1) does not decide the A/B gate; a spec
# that produced no report does.
$specFailuresCount = if ($compareMode) { @($results | Where-Object { $_.status -eq 'ERROR' }).Count } else { $failedSpecs.Count }
$passed = ($specFailuresCount -eq 0) -and ($regressions.Count -eq 0) -and ($pairedFailures.Count -eq 0) -and -not $comparedNothing -and ($TargetsAdvisory -or $failedTargets.Count -eq 0)

$md = New-Object System.Text.StringBuilder
[void]$md.AppendLine('# Perf gate: ' + $(if ($passed) { 'PASSED' } else { 'FAILED' }))
[void]$md.AppendLine('')
[void]$md.AppendLine("Executable: ``$exePath``")
if ($compareMode) {
    $line = "Interleaved A/B (ABBA): A = ``$compareExePath``, B = ``$exePath``"
    if ($CompareVars) { $line += " with vars ``$CompareVars``" }
    if ($CompareEnv) { $line += " with env ``$CompareEnv``" }
    $line += "; max regression $MaxRegressionPct %, rounds $Rounds..$MaxRounds"
    if ($AASummary) { $line += " (sized from ``$AASummary``)" }
    [void]$md.AppendLine($line + '.')
}
if ($Profile) { [void]$md.AppendLine("Profile: ``$Profile``") }
if ($Baseline) { [void]$md.AppendLine("Baseline: ``$Baseline`` (max regression $MaxRegressionPct %, min delta $MinDelta)") }
[void]$md.AppendLine('')
[void]$md.AppendLine('| Spec | Harness | Report |')
[void]$md.AppendLine('|---|---|---|')
foreach ($r in $results) {
    $roundsText = if ($roundsRun.Contains($r.spec)) { " ($($roundsRun[$r.spec]) rounds)" } else { '' }
    [void]$md.AppendLine("| $($r.spec)$roundsText | $($r.status) (exit $($r.exit_code)) | $($r.report) |")
}
if ($pairedRows.Count -gt 0) {
    [void]$md.AppendLine('')
    [void]$md.AppendLine('| Spec / run | Metric | Pairs | A median | B median | Median change | 95 % interval | Half-width | Method | Verdict | Note |')
    [void]$md.AppendLine('|---|---|---|---|---|---|---|---|---|---|---|')
    foreach ($c in $pairedRows) {
        $iv = if ($null -ne $c.lo_pct) { "$($c.lo_pct) .. $($c.hi_pct) %" } else { '-' }
        $chg = if ($null -ne $c.median_pct) { "$($c.median_pct) %" } else { '-' }
        $hw = if ($null -ne $c.half_width_pct) { "+-$($c.half_width_pct) %" } else { '-' }
        $v = if (@('REGRESSED', 'REFUSED', 'MISSING', 'INCONCLUSIVE') -contains $c.verdict) { "**$($c.verdict)**" } else { $c.verdict }
        $noteText = $c.note
        if ($c.regime_mismatched -gt 0) {
            $matched = if ($null -ne $c.matched_n) { "; without them ($($c.matched_n) pairs): $($c.matched_median_pct) %, $($c.matched_lo_pct) .. $($c.matched_hi_pct) %, +-$($c.matched_half_width_pct) %, $($c.matched_verdict)" } else { '' }
            $noteText = ("$noteText $($c.regime_mismatched) pair(s) across GPU clock regimes$matched").Trim()
        }
        [void]$md.AppendLine("| $($c.spec) / $($c.run) | $($c.metric) | $($c.n) | $($c.a_median) | $($c.b_median) | $chg | $iv | $hw | $($c.method) | $v | $noteText |")
    }
}
if ($stateFlags.Count -gt 0) {
    [void]$md.AppendLine('')
    [void]$md.AppendLine("Runs more than $([int]($stateTolerance * 100)) % off the session median of their GPU / host state (stamped into their report.json as ``gate_state_flags``):")
    [void]$md.AppendLine('')
    [void]$md.AppendLine('| Spec | Side | Round | Run | State | Value | Session median | Off |')
    [void]$md.AppendLine('|---|---|---|---|---|---|---|---|')
    foreach ($f in $stateFlags) { [void]$md.AppendLine("| $($f.spec) | $($f.side) | $($f.round) | $($f.run) | $($f.metric) | $($f.value) | $($f.session_median) | $($f.off_pct) % |") }
}
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
$modeName = if ($compareMode) { 'interleaved' } elseif ($Baseline) { 'baseline' } else { 'single' }
[pscustomobject]@{
    passed = $passed; exe = $exePath; baseline = $Baseline; profile = $Profile
    max_regression_pct = $MaxRegressionPct; min_delta = $MinDelta
    mode = $modeName
    compare_exe = $(if ($compareMode) { $compareExePath } else { $null }); compare_vars = $CompareVars; compare_env = $CompareEnv
    rounds = $roundsRun; paired = $pairedRows; state_flags = $stateFlags
    specs = $results; comparisons = $comparisons; targets = $targets; warnings = $warnings
} | ConvertTo-Json -Depth 8 | Set-Content -LiteralPath (Join-Path $OutRoot 'gate-summary.json') -Encoding utf8

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
foreach ($c in $pairedRows) {
    if ($c.verdict -ne 'NEUTRAL') {
        $noteText = if ($c.note) { " $($c.note)" } else { '' }
        Write-Host ("{0} {1}/{2} {3}: {4} -> {5} (median {6} %, interval {7} .. {8} %, {9} pairs){10}" -f $c.verdict, $c.spec, $c.run, $c.metric, $c.a_median, $c.b_median, $c.median_pct, $c.lo_pct, $c.hi_pct, $c.n, $noteText)
    }
}
foreach ($t in $targets) {
    if ($t.status -ne 'met' -and $t.status -ne 'skipped') {
        Write-Host ("{0} {1} {2}: {3} (limit {4}){5}" -f $t.status.ToUpperInvariant(), $t.id, $t.label, $(if ($null -ne $t.value) { $t.value } else { '-' }), $t.limit, $(if ($t.gate) { '' } else { ' [advisory]' }))
    }
}
foreach ($r in $failedSpecs) { Write-Host ("{0} {1} (see {2})" -f $r.status, $r.spec, $r.report) }
Write-Host ("Perf gate {0}: {1} spec(s), {2} comparison(s), {3} failing comparison(s), {4} target(s) ({5} gating missed). Summary: {6}" -f `
    $(if ($passed) { 'PASSED' } else { 'FAILED' }), $results.Count, ($comparisons.Count + $pairedRows.Count), ($regressions.Count + $pairedFailures.Count), $targets.Count, $failedTargets.Count, (Join-Path $OutRoot 'gate-summary.md'))
if ($passed) { exit 0 } else { exit 1 }
