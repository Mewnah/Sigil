<#
.SYNOPSIS
  P3B-F: fed-F0 fidelity of two streaming F0 configurations of ONE executable, A against B, on a
  clip corpus (scorer v2, three offsets per clip, leads of at least 4 s).

.DESCRIPTION
  Materializes fid-f0.template.json for every clip and runs `Sigil.exe --headless bench` for side A
  (-EnvA in its environment) and side B (-EnvB), ABBA across the clips (clip 1: A then B, clip 2: B
  then A, ...), holding the GPU timing lock per batch of clips (rule R2) and logging the background
  load (rustc processes, CPU %) of every batch. Both sides run the same executable, so the only
  difference is the switches (rule R5; P3B-G's concurrent execution-layer changes are on both
  sides).

  Per clip it reads the run's metrics (each the median over the three offsets) and prints, per
  metric, A, B and B - A; then the corpus summary: the mean difference with its paired Student t
  95 % interval over the clips, the share of clips where B is not worse and where it is better, and
  a verdict for the fed-F0 rows (lower is better; speaker cosine and DNSMOS higher). Writes
  rows.json / summary.json / summary.md to -Out.

  Switches (scripts/bench/specs/p3b/f/README.md): SIGIL_VC_F0_GEOMETRY=legacy|primed|filled,
  SIGIL_VC_F0_LEAD, SIGIL_VC_F0_PRIME, SIGIL_VC_F0_CONTINUITY, SIGIL_VC_F0_EDGE=zero,
  SIGIL_VC_F0_DECODE=1, SIGIL_RMVPE_PLAN_FRAMES=32|64, SIGIL_VC_FUSION_PARALLEL=0.

  The shipped default of the F0 cache is P3's (`legacy`: priming and silence continuity are
  opt-in, rule R6), so a side with no F0 switches IS P3's cache. A side that tests the v2 cache
  must opt in, with SIGIL_VC_F0_GEOMETRY=primed (or SIGIL_VC_F0_PRIME=1;SIGIL_VC_F0_CONTINUITY=1);
  SIGIL_VC_F0_GEOMETRY=filled adds the filled calls and is the measured combination
  (SIGIL_VC_F0_LEAD=fill alone changes only the lead).

.EXAMPLE
  # The F0 cache v2 (priming, continuity, replicated edges; opt-in) against P3's (decoder off on both
  # sides), DirectML. Side B must opt in: an empty side B is P3's cache again, the same as side A.
  pwsh scripts/bench/specs/p3b/f/run_fidelity.ps1 -Exe <exe> -AppData <appdata> -Ep dml -Out <dir> `
      -EnvA 'SIGIL_VC_F0_GEOMETRY=legacy;SIGIL_VC_F0_EDGE=zero' -EnvB 'SIGIL_VC_F0_GEOMETRY=primed'
  # The decoder (the f0_decode lever's override) on v2
  pwsh scripts/bench/specs/p3b/f/run_fidelity.ps1 -Exe <exe> -AppData <appdata> -Ep cuda -Out <dir> `
      -EnvA 'SIGIL_VC_F0_GEOMETRY=primed' -EnvB 'SIGIL_VC_F0_GEOMETRY=primed;SIGIL_VC_F0_DECODE=1'
#>
param(
    [Parameter(Mandatory = $true)][string]$Exe,
    [Parameter(Mandatory = $true)][string]$AppData,
    [string]$Slot = '',
    [ValidateSet('cpu', 'dml', 'cuda')][string]$Ep = 'dml',
    [string]$Pitch = 'rmvpe',
    [string]$Preset = 'low_latency',
    [double]$ExtraMs = 64,
    [double]$BlockMs = 40,
    [double]$LookaheadMs = 0,
    [string]$LeversJson = '{}',
    [string[]]$Clips = @(),
    [string]$EnvA = '',
    [string]$EnvB = '',
    [Parameter(Mandatory = $true)][string]$Out,
    [string]$LockTag = 'P3B-F fidelity',
    [int]$LockWait = 240,
    [int]$Batch = 4
)

$ErrorActionPreference = 'Stop'
$here = Split-Path -Parent $MyInvocation.MyCommand.Path
$repo = (Resolve-Path (Join-Path $here '..\..\..\..\..')).Path
$lock = Join-Path $repo 'scripts\bench\gpu-lock.ps1'
$devCache = Join-Path $env:LOCALAPPDATA 'sigil-dev-cache'
if (-not $Slot) { $Slot = Join-Path $AppData 'voice-changer\models\default' }
if ($Clips.Count -eq 1 -and $Clips[0] -match '[;,]') { $Clips = @($Clips[0] -split '[;,]' | Where-Object { $_.Trim() } | ForEach-Object { $_.Trim() }) }
if ($Clips.Count -eq 0) {
    # Eight clips of the dev cache: P3B-E's five onset-heavy clips plus three more languages.
    $Clips = @('models\transcribe\samples\ko-short.wav', 'sherpa-onnx\clips\ja-nemotron.wav', 'models\transcribe\samples\ru-short.wav',
        'sherpa-onnx\clips\en.wav', 'sherpa-onnx\clips\yue.wav', 'sherpa-onnx\clips\zh.wav', 'sherpa-onnx\clips\fr.wav',
        'sherpa-onnx\clips\en-jfk.wav') | ForEach-Object { Join-Path $devCache $_ }
}
New-Item -ItemType Directory -Force -Path $Out | Out-Null
$template = Get-Content -Raw (Join-Path $here 'fid-f0.template.json')
$env:SIGIL_APP_DATA_DIR = (Resolve-Path -LiteralPath $AppData).Path
$env:SIGIL_TEST_MODELS_DIR = $devCache

function Parse-Env([string]$Spec) {
    # A list of (name, value) pairs, built explicitly: a pipeline that emits one pair would be
    # unrolled into its two strings.
    $pairs = [System.Collections.Generic.List[object]]::new()
    foreach ($kv in ($Spec -split ';')) {
        if ($kv.Trim()) {
            $k, $v = $kv.Split('=', 2)
            $pairs.Add([string[]]@($k.Trim(), $v))
        }
    }
    return , $pairs.ToArray()
}
$sides = @{ A = (Parse-Env $EnvA); B = (Parse-Env $EnvB) }
@{ exe = $Exe; ep = $Ep; pitch = $Pitch; preset = $Preset; block_ms = $BlockMs; extra_ms = $ExtraMs; lookahead_ms = $LookaheadMs; levers = $LeversJson; env_a = $EnvA; env_b = $EnvB; clips = $Clips } |
    ConvertTo-Json | Set-Content -LiteralPath (Join-Path $Out 'config.json') -Encoding utf8

function Invoke-Side([string]$clip, [string]$side) {
    $stem = [IO.Path]::GetFileNameWithoutExtension($clip)
    $dir = Join-Path $Out "$side\$stem"
    if (Test-Path (Join-Path $dir 'report.json')) { Write-Host "$side $stem kept"; return }
    New-Item -ItemType Directory -Force -Path $dir | Out-Null
    $spec = $template.Replace('${NAME}', "fid-f0-$Ep-$side-$stem").Replace('${CLIP}', ($clip -replace '\\', '/')).
        Replace('${SLOT}', ($Slot -replace '\\', '/')).Replace('${EP}', $Ep).Replace('${PITCH}', $Pitch).
        Replace('${PRESET}', $Preset).Replace('${EXTRA_MS}', "$ExtraMs").Replace('${BLOCK_MS}', "$BlockMs").Replace('${LOOKAHEAD_MS}', "$LookaheadMs").Replace('${LEVERS_JSON}', $LeversJson)
    $specPath = Join-Path $dir 'spec.json'
    Set-Content -LiteralPath $specPath -Value $spec -Encoding utf8
    $saved = @{}
    foreach ($p in $sides[$side]) { $saved[$p[0]] = [Environment]::GetEnvironmentVariable($p[0]); [Environment]::SetEnvironmentVariable($p[0], $p[1]) }
    try {
        # Piped, so PowerShell waits for the (GUI-subsystem) executable to exit.
        & $Exe --headless bench $specPath --out $dir 2>&1 | Out-File -LiteralPath (Join-Path $dir 'bench.log') -Encoding utf8
        Write-Host "$side $stem exit $LASTEXITCODE"
    }
    finally {
        foreach ($k in $saved.Keys) { [Environment]::SetEnvironmentVariable($k, $saved[$k]) }
    }
}

$loads = @()
for ($b = 0; $b -lt $Clips.Count; $b += $Batch) {
    $chunk = @($Clips[$b..([Math]::Min($b + $Batch, $Clips.Count) - 1)])
    $pending = @($chunk | Where-Object { $s = [IO.Path]::GetFileNameWithoutExtension($_); -not ((Test-Path (Join-Path $Out "A\$s\report.json")) -and (Test-Path (Join-Path $Out "B\$s\report.json"))) })
    if ($pending.Count -eq 0) { continue }
    & pwsh -NoProfile -File $lock -Acquire $LockTag -Wait $LockWait -ExpectedMinutes (5 * $chunk.Count) -OwnerPid $PID
    if ($LASTEXITCODE -ne 0) { throw 'the GPU timing lock was not acquired' }
    try {
        $rustc = @(Get-Process rustc -ErrorAction SilentlyContinue).Count
        $cpu = (Get-CimInstance Win32_Processor | Measure-Object -Property LoadPercentage -Average).Average
        Write-Host "batch $($b / $Batch): background load rustc $rustc, CPU $cpu %"
        $loads += [pscustomobject]@{ batch = $b / $Batch; time = (Get-Date -Format o); rustc = $rustc; cpu_pct = $cpu }
        for ($i = 0; $i -lt $chunk.Count; $i++) {
            $order = if ((($b + $i) % 2) -eq 0) { @('A', 'B') } else { @('B', 'A') }
            foreach ($side in $order) { Invoke-Side $chunk[$i] $side }
        }
    }
    finally {
        & pwsh -NoProfile -File $lock -Release $LockTag
    }
}
if ($loads.Count) { $loads | ConvertTo-Json | Set-Content -LiteralPath (Join-Path $Out 'load.json') -Encoding utf8 }

# metric -> lower is better?
$keys = [ordered]@{
    fed_input_fpe_cents = $true; fed_input_gpe = $true; fed_input_vuv_error = $true; fed_input_onset_lag_frames = $true; fed_input_onsets_missed = $true
    fed_fpe_cents = $true; fed_gpe = $true; fed_vuv_error = $true
    env_mcd_db = $true; mcd_db = $true; speaker_cosine = $false; speaker_cosine_v2 = $false; dnsmos_ovrl = $false
    f0_rmse_cents = $true; vuv_error = $true
}
$rows = @()
foreach ($clip in $Clips) {
    $stem = [IO.Path]::GetFileNameWithoutExtension($clip)
    $ra = Join-Path $Out "A\$stem\report.json"; $rb = Join-Path $Out "B\$stem\report.json"
    if (-not ((Test-Path $ra) -and (Test-Path $rb))) { Write-Host "$stem : missing report"; continue }
    $ma = (Get-Content -Raw $ra | ConvertFrom-Json).runs[0].metrics
    $mb = (Get-Content -Raw $rb | ConvertFrom-Json).runs[0].metrics
    foreach ($k in $keys.Keys) {
        $a = $ma.$k; $bv = $mb.$k
        # The onset lag is signed (negative: the engine voiced before the reference); its size is what is better when smaller.
        if ($k -eq 'fed_input_onset_lag_frames' -and $null -ne $a -and $null -ne $bv) { $a = [Math]::Abs([double]$a); $bv = [Math]::Abs([double]$bv) }
        if ($null -ne $a -and $null -ne $bv) { $rows += [pscustomobject]@{ clip = $stem; metric = $k; A = [double]$a; B = [double]$bv; d = [double]$bv - [double]$a } }
    }
}
$rows | Sort-Object metric, clip | Format-Table -AutoSize | Out-String -Width 200 | Write-Host

# Student t quantile (two-sided 95 %) for small samples.
function T975([int]$df) {
    $t = @{ 1 = 12.706; 2 = 4.303; 3 = 3.182; 4 = 2.776; 5 = 2.571; 6 = 2.447; 7 = 2.365; 8 = 2.306; 9 = 2.262; 10 = 2.228; 11 = 2.201; 12 = 2.179; 13 = 2.160; 14 = 2.145; 15 = 2.131; 16 = 2.120; 17 = 2.110; 18 = 2.101; 19 = 2.093; 20 = 2.086 }
    if ($t.ContainsKey($df)) { return $t[$df] } else { return 1.96 }
}
$summary = foreach ($g in ($rows | Group-Object metric)) {
    $lower = $keys[$g.Name]
    $d = @($g.Group | ForEach-Object { $_.d })
    $n = $d.Count
    $mean = ($d | Measure-Object -Average).Average
    $sd = if ($n -gt 1) { [Math]::Sqrt((($d | ForEach-Object { ($_ - $mean) * ($_ - $mean) }) | Measure-Object -Sum).Sum / ($n - 1)) } else { 0 }
    $half = if ($n -gt 1) { (T975 ($n - 1)) * $sd / [Math]::Sqrt($n) } else { [double]::NaN }
    $eps = 1e-9
    $notWorse = @($d | Where-Object { if ($lower) { $_ -le $eps } else { $_ -ge -$eps } }).Count
    $better = @($d | Where-Object { if ($lower) { $_ -lt -$eps } else { $_ -gt $eps } }).Count
    $verdict = if ($n -lt 2) { 'n/a' }
        elseif (($lower -and ($mean + $half) -lt 0) -or (-not $lower -and ($mean - $half) -gt 0)) { 'B better' }
        elseif (($lower -and ($mean - $half) -gt 0) -or (-not $lower -and ($mean + $half) -lt 0)) { 'B worse' }
        else { 'within noise' }
    [pscustomobject]@{
        metric = $g.Name; clips = $n
        A = ($g.Group | Measure-Object A -Average).Average; B = ($g.Group | Measure-Object B -Average).Average
        mean_d = $mean; ci_lo = $mean - $half; ci_hi = $mean + $half
        not_worse_share = $notWorse / [Math]::Max(1, $n); better_share = $better / [Math]::Max(1, $n); verdict = $verdict
    }
}
$summary | Format-Table -AutoSize | Out-String -Width 220 | Write-Host
$summary | ConvertTo-Json | Set-Content -LiteralPath (Join-Path $Out 'summary.json') -Encoding utf8
$rows | ConvertTo-Json | Set-Content -LiteralPath (Join-Path $Out 'rows.json') -Encoding utf8
$md = @("| metric | clips | A | B | B - A (mean) | 95 % CI | B not worse | B better | verdict |", '|---|---|---|---|---|---|---|---|---|')
foreach ($s in $summary) {
    $md += ('| {0} | {1} | {2:G5} | {3:G5} | {4:+0.#####;-0.#####;0} | [{5:G4}, {6:G4}] | {7:P0} | {8:P0} | {9} |' -f $s.metric, $s.clips, $s.A, $s.B, $s.mean_d, $s.ci_lo, $s.ci_hi, $s.not_worse_share, $s.better_share, $s.verdict)
}
$md -join "`n" | Set-Content -LiteralPath (Join-Path $Out 'summary.md') -Encoding utf8
