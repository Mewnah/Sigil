<#
.SYNOPSIS
  P3B-E: whole-clip and onset fidelity of RMVPE configurations, A against B, on clips with many
  onsets (scorer v2, three offsets, leads of at least 4 s).

.DESCRIPTION
  Materializes fid-onsets.template.json for every clip and runs `Sigil.exe --headless bench` for
  side A (no switch) and side B (-EnvB, e.g. 'SIGIL_RMVPE_PLAN_FRAMES=64') in ABBA order across the
  clips (clip 1: A then B, clip 2: B then A, ...), holding the GPU timing lock per batch of clips
  (rule R2). Prints the background load (rustc processes, CPU %) per batch, then per clip and as a
  mean over the clips the fed-F0 rows (fine pitch error, GPE, V/UV against the input's whole-clip
  F0, onset lag and missed onsets), the speaker cosine (v1 and v2), the envelope and STFT MCD, and
  DNSMOS, side by side with B - A.

.EXAMPLE
  pwsh scripts/bench/specs/p3b/e/run_fidelity.ps1 -Exe <exe> -AppData <appdata> -Ep dml -Out <dir> `
      -EnvB 'SIGIL_RMVPE_PLAN_FRAMES=64'
#>
param(
    [Parameter(Mandatory = $true)][string]$Exe,
    [Parameter(Mandatory = $true)][string]$AppData,
    [string]$Slot = '',
    [ValidateSet('cpu', 'dml', 'cuda')][string]$Ep = 'dml',
    [string]$Pitch = 'rmvpe',
    [string]$Preset = 'low_latency',
    [double]$ExtraMs = 64,
    [double]$LookaheadMs = 0,
    [string[]]$Clips = @(),
    [Parameter(Mandatory = $true)][string]$EnvB,
    [Parameter(Mandatory = $true)][string]$Out,
    [string]$LockTag = 'P3B-E fidelity',
    [int]$LockWait = 180,
    [int]$Batch = 3
)

$ErrorActionPreference = 'Stop'
$here = Split-Path -Parent $MyInvocation.MyCommand.Path
$repo = (Resolve-Path (Join-Path $here '..\..\..\..\..')).Path
$lock = Join-Path $repo 'scripts\bench\gpu-lock.ps1'
$devCache = Join-Path $env:LOCALAPPDATA 'sigil-dev-cache'
if (-not $Slot) { $Slot = Join-Path $AppData 'voice-changer\models\default' }
if ($Clips.Count -eq 1 -and $Clips[0] -match '[;,]') { $Clips = @($Clips[0] -split '[;,]' | Where-Object { $_.Trim() } | ForEach-Object { $_.Trim() }) }
if ($Clips.Count -eq 0) {
    # The clips with the most onsets per minute of speech (measure-first step 1 of P3B-E).
    $Clips = @('models\transcribe\samples\ko-short.wav', 'sherpa-onnx\clips\ja-nemotron.wav', 'models\transcribe\samples\ru-short.wav',
        'sherpa-onnx\clips\en.wav', 'sherpa-onnx\clips\yue.wav') | ForEach-Object { Join-Path $devCache $_ }
}
New-Item -ItemType Directory -Force -Path $Out | Out-Null
$template = Get-Content -Raw (Join-Path $here 'fid-onsets.template.json')
$env:SIGIL_APP_DATA_DIR = (Resolve-Path -LiteralPath $AppData).Path
$env:SIGIL_TEST_MODELS_DIR = $devCache
$pairs = @($EnvB -split ';' | Where-Object { $_.Trim() } | ForEach-Object { $k, $v = $_.Split('=', 2); , @($k.Trim(), $v) })

function Invoke-Side([string]$clip, [string]$side) {
    $stem = [IO.Path]::GetFileNameWithoutExtension($clip)
    $dir = Join-Path $Out "$side\$stem"
    if (Test-Path (Join-Path $dir 'report.json')) { Write-Host "$side $stem kept"; return }
    New-Item -ItemType Directory -Force -Path $dir | Out-Null
    $spec = $template.Replace('${NAME}', "fid-onsets-$Ep-$side-$stem").Replace('${CLIP}', ($clip -replace '\\', '/')).
        Replace('${SLOT}', ($Slot -replace '\\', '/')).Replace('${CONTENT_ENCODER}', '').Replace('${EP}', $Ep).Replace('${PITCH}', $Pitch).
        Replace('${PRESET}', $Preset).Replace('${EXTRA_MS}', "$ExtraMs").Replace('${LOOKAHEAD_MS}', "$LookaheadMs").Replace('${ANALYZER}', 'v2')
    $specPath = Join-Path $dir 'spec.json'
    Set-Content -LiteralPath $specPath -Value $spec -Encoding utf8
    $saved = @{}
    if ($side -eq 'B') { foreach ($p in $pairs) { $saved[$p[0]] = [Environment]::GetEnvironmentVariable($p[0]); [Environment]::SetEnvironmentVariable($p[0], $p[1]) } }
    try {
        # Piped, so PowerShell waits for the (GUI-subsystem) executable to exit.
        & $Exe --headless bench $specPath --out $dir 2>&1 | Out-File -LiteralPath (Join-Path $dir 'bench.log') -Encoding utf8
        Write-Host "$side $stem exit $LASTEXITCODE"
    }
    finally {
        foreach ($k in $saved.Keys) { [Environment]::SetEnvironmentVariable($k, $saved[$k]) }
    }
}

for ($b = 0; $b -lt $Clips.Count; $b += $Batch) {
    $chunk = @($Clips[$b..([Math]::Min($b + $Batch, $Clips.Count) - 1)])
    $rustc = @(Get-Process rustc -ErrorAction SilentlyContinue).Count
    $cpu = (Get-CimInstance Win32_Processor | Measure-Object -Property LoadPercentage -Average).Average
    Write-Host "batch $($b / $Batch): background load rustc $rustc, CPU $cpu %"
    & pwsh -NoProfile -File $lock -Acquire $LockTag -Wait $LockWait -ExpectedMinutes (6 * $chunk.Count) -OwnerPid $PID
    if ($LASTEXITCODE -ne 0) { throw 'the GPU timing lock was not acquired' }
    try {
        for ($i = 0; $i -lt $chunk.Count; $i++) {
            $order = if ((($b + $i) % 2) -eq 0) { @('A', 'B') } else { @('B', 'A') }
            foreach ($side in $order) { Invoke-Side $chunk[$i] $side }
        }
    }
    finally {
        & pwsh -NoProfile -File $lock -Release $LockTag
    }
}

$keys = @('fed_input_fpe_cents', 'fed_input_gpe', 'fed_input_vuv_error', 'fed_input_onset_lag_frames', 'fed_input_onsets_missed',
    'speaker_cosine', 'speaker_cosine_v2', 'env_mcd_db', 'mcd_db', 'dnsmos_ovrl', 'f0_rmse_cents', 'vuv_error')
$rows = @()
foreach ($clip in $Clips) {
    $stem = [IO.Path]::GetFileNameWithoutExtension($clip)
    $ra = Join-Path $Out "A\$stem\report.json"; $rb = Join-Path $Out "B\$stem\report.json"
    if (-not ((Test-Path $ra) -and (Test-Path $rb))) { Write-Host "$stem : missing report"; continue }
    $ma = (Get-Content -Raw $ra | ConvertFrom-Json).runs[0].metrics
    $mb = (Get-Content -Raw $rb | ConvertFrom-Json).runs[0].metrics
    foreach ($k in $keys) {
        $a = $ma.$k; $bv = $mb.$k
        if ($null -ne $a -and $null -ne $bv) { $rows += [pscustomobject]@{ clip = $stem; metric = $k; A = [double]$a; B = [double]$bv; d = [double]$bv - [double]$a } }
    }
}
$rows | Sort-Object metric, clip | Format-Table -AutoSize | Out-String -Width 200 | Write-Host
$summary = foreach ($g in ($rows | Group-Object metric)) {
    [pscustomobject]@{ metric = $g.Name; clips = $g.Count; A = ($g.Group | Measure-Object A -Average).Average; B = ($g.Group | Measure-Object B -Average).Average; mean_d = ($g.Group | Measure-Object d -Average).Average }
}
$summary | Format-Table -AutoSize | Out-String -Width 200 | Write-Host
$summary | ConvertTo-Json | Set-Content -LiteralPath (Join-Path $Out 'summary.json') -Encoding utf8
$rows | ConvertTo-Json | Set-Content -LiteralPath (Join-Path $Out 'rows.json') -Encoding utf8
