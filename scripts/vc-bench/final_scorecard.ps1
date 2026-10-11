<#
.SYNOPSIS
  P7-B: drive the final scorecard of the shipped voice-changer defaults (T1-T5, T7, T9 on CUDA,
  DirectML and the CPU) with a quiet-window check and a load log for every step.

.DESCRIPTION
  One campaign folder (-Root) holds everything `final_scorecard.py` reads:

      env.json                  the executable's hash, the GPU / CPU / OS / power plan
      steps.jsonl               one line per step and sub-step: start / end, exit code, the quiet check
      load/<step>.csv           the load sampled every 2 s while the step ran (total CPU, Sigil's own
                                cores, compilers, other busy programs, the GPU timing lock's owner)
      gate/<pass>/<spec>/       run-perf-gate.ps1 -Profile final sessions
      corpus/<ep>/              scripts/bench/specs/p3b/b/run_corpus.ps1 (T5, scorer v2) + summary.json
      decisions/<name>/         interleaved (ABBA) sessions that decide a default
      final-scorecard.{json,md} the tables

  Steps (-Step, several allowed; every step is resumable: a finished (sub-)step leaves
  <Root>/<DoneDir>/<name> (-DoneDir, default `done`) and is skipped unless -Force; two instances of the
  script with different -Step lists and -DoneDir values can wait for the lock side by side):
    env      record the environment; write the synthetic kNN index the T3 row needs
    trial    plumbing check with the stub specs only (no models, no GPU): a locked step with two rows
             and its load log; point SIGIL_GPU_LOCK at a scratch file to keep it off the shared lock
    smoke    time-compressed run of the new templates (the soak, the app-shape rows, the CPU
             thread-policy row, the silent rows, the shared-queue row, a device-block pace row):
             validates the specs and the machine before the long steps
    gate     -Passes passes of the timing rows (sve-dml, sve-cuda, the three CPU rows, the app-shape
             rows in the first two passes), the order of the execution-provider families rotated per
             pass so a slow GPU or host state hits every one alike; a pass holds the GPU timing lock
             from its quiet check to its last row. Targets are judged on the median of the passes
    once     the rows that need one pass: the plumbing stubs, the five-process cold loads, VRAM, the
             T4 silent rows (once-a); context scaling and the kNN row (once-b)
    corpus   T5 (scorer v2) over the local clips on cuda, dml and cpu, each in a locked hold of 25-45
             minutes (the CPU in two halves); -CorpusEps names more: `cpu-fcpe` runs the CPU with FCPE,
             the F0 method the Streamer button applies, on the first half of the clips
    soak     the 30 minute T7 soak on CUDA under a game-like load
    decide   the interleaved decision sessions (-Decide names): devblock-cuda, devblock-dml,
             devblock-cpu, ctx512 (64 against 512 ms of context), cpu-int8, cpu-f0 (SwiftF0 against the
             FCPE the Streamer button applies, one thread), cpu-t6-f0 (the same on 6 threads below
             normal, the thread policy P7-I0 ships), dml-shared. A session's summary counts as its result
             only when this run wrote it and it lists every round of every spec (a stale summary of an
             earlier run is deleted before the gate starts)
    summary  final_scorecard.py

  Quiet window: a locked step takes the GPU timing lock itself (scripts/bench/gpu-lock.ps1), then
  waits up to -QuietWaitMin for no compiler and total CPU under -QuietCpuPct before it starts and
  between its rows, so its measurement window is checked while nobody else may time anything. The
  corpus runner and the decision sessions run inside such a hold too (the runner gets a private lock
  file through SIGIL_GPU_LOCK for its own acquisitions; the sessions pass -NoGpuLock), so the whole
  campaign queues once per hold, not once per run or per pair. A step that started outside the quiet
  window is recorded as such and the scorecard marks it `disturbed`.

.EXAMPLE
  pwsh scripts/vc-bench/final_scorecard.ps1 -Exe C:\bench\exe-final\sigil.exe -AppData C:\bench\appdata -Root C:\bench\final -Step env,smoke,gate,once,corpus,summary
#>
[CmdletBinding()]
param(
    [Parameter(Mandatory = $true)][string]$Exe,
    [Parameter(Mandatory = $true)][string]$AppData,
    [Parameter(Mandatory = $true)][string]$Root,
    [string]$Slot,
    [string]$Clip,
    [string[]]$Step = @('env', 'gate', 'once', 'corpus', 'summary'),
    [string[]]$Decide = @(),
    [string[]]$CorpusEps = @('cuda', 'dml', 'cpu'),
    [int]$Passes = 3,
    [string]$Tag = 'P7-B',
    [double]$LockWaitMin = 480,
    [double]$QuietCpuPct = 25,
    [double]$QuietWaitMin = 10,
    [int]$DecideRounds = 4,
    [string]$KnnIndex,
    [string]$DoneDir = 'done',
    [switch]$Force
)

$ErrorActionPreference = 'Stop'
Set-StrictMode -Version 3.0

# `pwsh -File ... -Step a,b` passes one string 'a,b': split every list parameter.
function Split-List { param([string[]]$Items) return @($Items | Where-Object { $_ } | ForEach-Object { $_ -split ',' } | ForEach-Object { $_.Trim() } | Where-Object { $_ }) }
$Step = @(Split-List $Step)
$Decide = @(Split-List $Decide)
$CorpusEps = @(Split-List $CorpusEps)
foreach ($s in $Step) {
    if (@('env', 'trial', 'smoke', 'gate', 'once', 'corpus', 'soak', 'decide', 'summary') -notcontains $s) { throw "unknown -Step '$s'" }
}

$here = Split-Path -Parent $MyInvocation.MyCommand.Path
$repo = (Resolve-Path (Join-Path $here '..\..')).Path
$gate = Join-Path $repo 'scripts\bench\run-perf-gate.ps1'
$lockScript = Join-Path $repo 'scripts\bench\gpu-lock.ps1'
$corpusScript = Join-Path $repo 'scripts\bench\specs\p3b\b\run_corpus.ps1'
$devCache = Join-Path $env:LOCALAPPDATA 'sigil-dev-cache'

$Exe = (Resolve-Path -LiteralPath $Exe).Path
$AppData = (Resolve-Path -LiteralPath $AppData).Path
New-Item -ItemType Directory -Force -Path $Root | Out-Null
$Root = (Resolve-Path -LiteralPath $Root).Path
if (-not $Slot) { $Slot = Join-Path $AppData 'voice-changer\models\default' }
if (-not $Clip) { $Clip = Join-Path $devCache 'sherpa-onnx\clips\en-jfk.wav' }
if (-not $KnnIndex) { $KnnIndex = Join-Path $Root 'knn30k.index' }
foreach ($p in @($Slot, $Clip)) { if (-not (Test-Path -LiteralPath $p)) { throw "not found: $p" } }
New-Item -ItemType Directory -Force -Path (Join-Path $Root 'load'), (Join-Path $Root $DoneDir) | Out-Null
$stepsFile = Join-Path $Root 'steps.jsonl'
$logical = [Environment]::ProcessorCount

# ---- CPU load (GetSystemTimes) -------------------------------------------------------------------
if (-not ('Sigil.SysTimes' -as [type])) {
    Add-Type -Namespace Sigil -Name SysTimes -MemberDefinition @'
[System.Runtime.InteropServices.DllImport("kernel32.dll")]
public static extern bool GetSystemTimes(out long idle, out long kernel, out long user);
'@
}
function Get-SysTimes {
    $i = 0L; $k = 0L; $u = 0L
    [void][Sigil.SysTimes]::GetSystemTimes([ref]$i, [ref]$k, [ref]$u)
    return @{ idle = $i; busy = ($k - $i) + $u; total = $k + $u }
}
function Measure-CpuPct {
    param([double]$Seconds = 6)
    $a = Get-SysTimes
    Start-Sleep -Milliseconds ([int]($Seconds * 1000))
    $b = Get-SysTimes
    $t = $b.total - $a.total
    if ($t -le 0) { return 0.0 }
    return [math]::Round(100.0 * ($b.busy - $a.busy) / $t, 1)
}

function Get-ProcCount {
    param([string[]]$Names)
    return @(Get-Process -Name $Names -ErrorAction SilentlyContinue).Count
}

function Test-QuietNow {
    $cpu = Measure-CpuPct -Seconds 6
    $rustc = Get-ProcCount -Names 'rustc'
    $cargo = Get-ProcCount -Names 'cargo', 'cl', 'link', 'lld-link', 'ninja', 'cmake'
    return [pscustomobject]@{ cpu_pct = $cpu; rustc = $rustc; cargo = $cargo; quiet = (($cpu -lt $QuietCpuPct) -and ($rustc -eq 0) -and ($cargo -eq 0)) }
}

function Wait-Quiet {
    # Wait (holding whatever lock the caller holds, for at most -QuietWaitMin) for the quiet window.
    $deadline = (Get-Date).AddMinutes($QuietWaitMin)
    $first = Get-Date
    do {
        $q = Test-QuietNow
        if ($q.quiet) { break }
        Write-Host ("  not quiet: CPU {0} %, rustc {1}, other compilers {2}; checking again" -f $q.cpu_pct, $q.rustc, $q.cargo)
        Start-Sleep -Seconds 10
    } while ((Get-Date) -lt $deadline)
    return [pscustomobject]@{ cpu_pct = $q.cpu_pct; rustc = $q.rustc; cargo = $q.cargo; quiet = $q.quiet; waited_s = [math]::Round(((Get-Date) - $first).TotalSeconds, 1) }
}

# ---- the load sampler (a background job per step) --------------------------------------------------
$samplerBlock = {
    param($csv, $stop, $lockPath)
    Add-Type -Namespace Sigil -Name SysTimes2 -MemberDefinition @'
[System.Runtime.InteropServices.DllImport("kernel32.dll")]
public static extern bool GetSystemTimes(out long idle, out long kernel, out long user);
'@
    $watch = @('cargo', 'cl', 'link', 'lld-link', 'ninja', 'cmake', 'msbuild')
    'ts,t_unix,cpu_total_pct,sigil_cores,rustc,cargo,heavy_other,lock_owner' | Set-Content -LiteralPath $csv -Encoding utf8
    $i = 0L; $k = 0L; $u = 0L
    [void][Sigil.SysTimes2]::GetSystemTimes([ref]$i, [ref]$k, [ref]$u)
    $prevBusy = ($k - $i) + $u; $prevTotal = $k + $u
    $prevWall = [DateTime]::UtcNow
    $prevCpu = @{}
    foreach ($p in [System.Diagnostics.Process]::GetProcesses()) {
        try { $prevCpu[$p.Id] = @($p.ProcessName, $p.TotalProcessorTime.TotalSeconds) } catch { }
        $p.Dispose()
    }
    while (-not (Test-Path -LiteralPath $stop)) {
        Start-Sleep -Milliseconds 2000
        $now = [DateTime]::UtcNow
        $wall = ($now - $prevWall).TotalSeconds
        [void][Sigil.SysTimes2]::GetSystemTimes([ref]$i, [ref]$k, [ref]$u)
        $busy = ($k - $i) + $u; $total = $k + $u
        $cpuPct = if ($total -gt $prevTotal) { [math]::Round(100.0 * ($busy - $prevBusy) / ($total - $prevTotal), 1) } else { 0 }
        $prevBusy = $busy; $prevTotal = $total; $prevWall = $now
        $sigil = 0.0; $rustc = 0; $cargo = 0
        $heavy = @{}
        $cur = @{}
        foreach ($p in [System.Diagnostics.Process]::GetProcesses()) {
            try {
                $name = $p.ProcessName
                $cpu = $p.TotalProcessorTime.TotalSeconds
                $cur[$p.Id] = @($name, $cpu)
                $before = if ($prevCpu.ContainsKey($p.Id)) { $prevCpu[$p.Id][1] } else { 0.0 }
                $d = [math]::Max(0.0, $cpu - $before)
                if ($name -ieq 'sigil') { $sigil += $d }
                elseif ($name -ieq 'rustc') { $rustc++ }
                elseif ($watch -contains $name.ToLowerInvariant()) { $cargo++ }
                if ($name -ine 'sigil' -and $name -ine 'pwsh' -and $wall -gt 0 -and ($d / $wall) -ge 0.4) { $heavy[$name] = $true }
            } catch { }
            $p.Dispose()
        }
        $prevCpu = $cur
        $owner = ''
        try {
            if ($lockPath -and (Test-Path -LiteralPath $lockPath)) {
                $txt = [System.IO.File]::ReadAllText($lockPath)
                try { $owner = [string]((ConvertFrom-Json $txt).tag) } catch { $owner = $txt.Trim() }
            }
        } catch { }
        $owner = ($owner -replace '[,\r\n]', ' ')
        $line = '{0},{1},{2},{3},{4},{5},{6},{7}' -f $now.ToString('o'), [DateTimeOffset]::new($now).ToUnixTimeSeconds(), $cpuPct, [math]::Round($sigil / [math]::Max($wall, 0.001), 3), $rustc, $cargo, (($heavy.Keys | Sort-Object) -join ';'), $owner
        Add-Content -LiteralPath $csv -Value $line -Encoding utf8
    }
}

$lockPath = (& pwsh -NoProfile -File $lockScript -PrintPath | Select-Object -First 1)

function Start-Sampler {
    param([string]$Name)
    $csv = Join-Path $Root "load\$Name.csv"
    $stop = Join-Path $Root "load\$Name.stop"
    Remove-Item -LiteralPath $stop -ErrorAction SilentlyContinue
    $job = Start-Job -ScriptBlock $samplerBlock -ArgumentList $csv, $stop, $lockPath
    return @{ job = $job; stop = $stop; csv = $csv }
}
function Stop-Sampler {
    param($S)
    New-Item -ItemType File -Force -Path $S.stop | Out-Null
    [void](Wait-Job -Job $S.job -Timeout 20)
    Remove-Job -Job $S.job -Force -ErrorAction SilentlyContinue
    Remove-Item -LiteralPath $S.stop -ErrorAction SilentlyContinue
}

function Add-StepRecord {
    param([hashtable]$Rec)
    ($Rec | ConvertTo-Json -Compress -Depth 6) | Add-Content -LiteralPath $stepsFile -Encoding utf8
}

function Get-DoneFile {
    param([string]$Name)
    return (Join-Path $Root ($DoneDir + '\' + ($Name -replace '[\\/:]', '_')))
}
function Test-Done {
    param([string]$Name)
    if ($Force) { return $false }
    return (Test-Path -LiteralPath (Get-DoneFile $Name))
}
function Set-Done {
    param([string]$Name)
    Set-Content -LiteralPath (Get-DoneFile $Name) -Value (Get-Date -Format 'o')
}

$script:CurrentStep = ''

function Invoke-Step {
    # kind 'locked': this script holds the GPU timing lock around the quiet check and the body.
    # kind 'sampled': the body's harness takes the lock itself; only the samples while it is ours count.
    param([string]$Name, [ValidateSet('locked', 'sampled')][string]$Kind, [double]$ExpectedMin, [scriptblock]$Body)
    if (Test-Done $Name) { Write-Host "[$Name] done before, kept"; return }
    Write-Host "[$Name] $Kind"
    $rec = [ordered]@{ name = $Name; kind = $Kind; lock_tag = $Tag; start = (Get-Date).ToString('o') }
    $lockTag = "$Tag-$Name"
    $held = $false
    $sampler = $null
    $code = 1
    $script:CurrentStep = $Name
    try {
        if ($Kind -eq 'locked') {
            & pwsh -NoProfile -File $lockScript -Acquire $lockTag -Wait $LockWaitMin -PollSeconds 5 -ExpectedMinutes $ExpectedMin -OwnerPid $PID | Out-Host
            if ($LASTEXITCODE -ne 0) { throw "the GPU timing lock was not acquired for $Name" }
            $held = $true
            $rec['pre'] = Wait-Quiet
        } else {
            $rec['pre'] = Test-QuietNow
        }
        $rec['acquired'] = (Get-Date).ToString('o')
        $sampler = Start-Sampler -Name $Name
        $res = @(& $Body)
        $code = [int]$res[-1]
    } catch {
        Write-Warning "[$Name] $_"
        $rec['error'] = "$_"
        $code = 1
    } finally {
        if ($null -ne $sampler) { Stop-Sampler $sampler }
        if ($held) { & pwsh -NoProfile -File $lockScript -Release $lockTag | Out-Host }
        $script:CurrentStep = ''
    }
    $rec['end'] = (Get-Date).ToString('o')
    $rec['exit'] = $code
    Add-StepRecord $rec
    if ($code -eq 0) { Set-Done $Name } else { Write-Warning "[$Name] exit $code (not marked done)" }
}

function Invoke-Sub {
    # A row of a locked step: skipped when done, a quiet re-check before it, its own record (the parent's
    # load file, its own time window) so the load of one row can be told from the next.
    param([string]$Name, [scriptblock]$Body, [string]$Pass = '', [string[]]$Specs = @())
    $full = "$($script:CurrentStep)/$Name"
    if (Test-Done $full) { Write-Host "  [$full] done before, kept"; return 0 }
    Write-Host "  [$full]"
    $rec = [ordered]@{ name = $full; kind = 'sub'; csv = $script:CurrentStep; pass = $Pass; specs = @($Specs); lock_tag = $Tag; start = (Get-Date).ToString('o') }
    $rec['pre'] = Wait-Quiet
    $code = 1
    try {
        $res = @(& $Body)
        $code = [int]$res[-1]
    } catch {
        Write-Warning "  [$full] $_"
        $rec['error'] = "$_"
    }
    $rec['end'] = (Get-Date).ToString('o')
    $rec['exit'] = $code
    Add-StepRecord $rec
    if ($code -eq 0) { Set-Done $full } else { Write-Warning "  [$full] exit $code (not marked done)" }
    return $code
}

function Invoke-Gate {
    # One run-perf-gate.ps1 session; returns its exit code. -TargetsAdvisory: a miss of a revised target
    # is a result, not a failure of the session.
    param([string]$OutDir, [string[]]$Only, [string[]]$Extra = @(), [string[]]$Vars = @(), [switch]$NoLock)
    New-Item -ItemType Directory -Force -Path $OutDir | Out-Null
    # `pwsh -File` rejects a repeated parameter: every -Var goes in one comma-separated string (the gate
    # splits it; none of these values contains a comma - run lists with commas travel through the
    # SIGIL_BENCH_<NAME> environment variables instead).
    $allVars = (@("KNN_INDEX=$KnnIndex") + $Vars) -join ','
    $gArgs = @('-NoProfile', '-File', $gate, '-Profile', 'final', '-Exe', $Exe, '-AppData', $AppData, '-Slot', $Slot, '-Clip', $Clip,
        '-OutRoot', $OutDir, '-Only', ($Only -join ','), '-TargetsAdvisory', '-Var', $allVars) + $Extra
    if ($NoLock) { $gArgs += '-NoGpuLock' }
    & pwsh @gArgs *> (Join-Path $OutDir 'gate.log')
    return $LASTEXITCODE
}

# ---- steps -----------------------------------------------------------------------------------------
function Step-Env {
    $envPath = Join-Path $Root 'env.json'
    $exeInfo = Get-Item -LiteralPath $Exe
    $sha = (Get-FileHash -LiteralPath $Exe -Algorithm SHA256).Hash.ToLowerInvariant()
    $gpu = ''
    try { $gpu = [string](& nvidia-smi --query-gpu=name,driver_version,clocks.max.sm,power.limit,pstate --format=csv,noheader 2>$null | Select-Object -First 1) } catch { $gpu = '' }
    $cpu = (Get-CimInstance Win32_Processor | Select-Object -First 1).Name
    $plan = ''
    try { $plan = (& powercfg /getactivescheme) -join ' ' } catch { $plan = '' }
    $os = (Get-CimInstance Win32_OperatingSystem)
    $head = ''
    try { $head = [string](& git -C $repo rev-parse --short HEAD) } catch { $head = '' }
    $doc = [ordered]@{
        exe = $Exe; exe_sha256 = $sha; exe_bytes = $exeInfo.Length; exe_time = $exeInfo.LastWriteTime.ToString('o')
        git_head = $head; app_data = $AppData; slot = $Slot; clip = $Clip
        clip_sha256 = (Get-FileHash -LiteralPath $Clip -Algorithm SHA256).Hash.ToLowerInvariant()
        gpu = $gpu; cpu = $cpu; logical_processors = $logical; os = "$($os.Caption) $($os.BuildNumber)"; power_plan = $plan
        memory_gb = [math]::Round($os.TotalVisibleMemorySize / 1MB, 1); started = (Get-Date).ToString('o'); lock_path = $lockPath
    }
    ($doc | ConvertTo-Json -Depth 4) | Set-Content -LiteralPath $envPath -Encoding utf8
    Write-Host "environment: $envPath"
    if (-not (Test-Path -LiteralPath $KnnIndex)) {
        & python (Join-Path $repo 'scripts\bench\specs\p3b\a\make-synthetic-index.py') --out $KnnIndex
    }
}

function Step-Trial {
    $out = Join-Path $Root 'trial'
    Invoke-Step -Name 'trial' -Kind 'locked' -ExpectedMin 2 -Body {
        $bad = 0
        if ((Invoke-Sub -Name 'stub-smoke' -Pass 'trial' -Specs @('stub-smoke') -Body { return (Invoke-Gate -OutDir (Join-Path $out 'a') -Only @('stub-smoke') -NoLock) }) -ne 0) { $bad++ }
        if ((Invoke-Sub -Name 'stub-p3' -Pass 'trial' -Specs @('stub-p3') -Body { return (Invoke-Gate -OutDir (Join-Path $out 'b') -Only @('stub-p3') -NoLock) }) -ne 0) { $bad++ }
        return $bad
    }
}

function Step-Smoke {
    $smokeVars = @('OFFLINE_REPEAT_JSON=1', 'PACED_S_JSON=6', 'CPU_REPEAT_JSON=1', 'SOAK_S_JSON=45', 'SOAK=1', 'DECIDE=1')
    $quickRuns = '[{"mode":"offline","name":"offline","repeat":1},{"mode":"paced","name":"paced","seconds":6,"clock":"wall","warmup_s":1},{"mode":"contention","name":"h3-game","profile":"h3_cpu_game","seconds":6,"warmup_s":1}]'
    $only = 'app-dml', 'app-cuda', 'sve-cpu-t6', 'silent-dml', 'silent-cpu', 'soak-cuda', 'dml-shared', 'cpu-t6-lean', 'devblock-cuda', 'knn30k-cuda', 'vram-cuda'
    $out = Join-Path $Root 'smoke'
    Invoke-Step -Name 'smoke' -Kind 'locked' -ExpectedMin 15 -Body {
        $env:SIGIL_BENCH_RUNS_JSON = $quickRuns
        try { return (Invoke-Gate -OutDir $out -Only $only -Vars $smokeVars -NoLock) } finally { Remove-Item Env:SIGIL_BENCH_RUNS_JSON -ErrorAction SilentlyContinue }
    }
}

function Get-PassOrder {
    param([int]$Pass)
    $families = @(
        @('sve-dml', 'app-dml'),
        @('sve-cuda', 'app-cuda'),
        @('sve-cpu', 'sve-cpu-pool', 'sve-cpu-t6')
    )
    $shift = ($Pass - 1) % $families.Count
    $list = @()
    for ($i = 0; $i -lt $families.Count; $i++) {
        foreach ($s in $families[($i + $shift) % $families.Count]) {
            if ($s -like 'app-*' -and $Pass -gt 2) { continue }
            $list += $s
        }
    }
    return $list
}

function Step-Gate {
    for ($p = 1; $p -le $Passes; $p++) {
        $pass = "pass$p"
        $order = Get-PassOrder -Pass $p
        Invoke-Step -Name $pass -Kind 'locked' -ExpectedMin 35 -Body {
            $bad = 0
            foreach ($spec in $order) {
                $outDir = Join-Path $Root "gate\$pass\$spec"
                $c = Invoke-Sub -Name $spec -Pass $pass -Specs @($spec) -Body { return (Invoke-Gate -OutDir $outDir -Only @($spec) -NoLock) }
                if ($c -ne 0) { $bad++ }
            }
            return $bad
        }
    }
}

function Step-Once {
    $once = Join-Path $Root 'gate\once'
    $loadExtra = @('-CompareExe', $Exe, '-Rounds', '5', '-MaxRounds', '5', '-AllowInconclusive', '-Metrics', 'engine_load_ms,generator_builds')
    Invoke-Step -Name 'once-a' -Kind 'locked' -ExpectedMin 30 -Body {
        $bad = 0
        if ((Invoke-Sub -Name 'stubs' -Pass 'once' -Specs @('stub-smoke', 'stub-p3') -Body { return (Invoke-Gate -OutDir (Join-Path $once 'stubs') -Only @('stub-smoke', 'stub-p3') -NoLock) }) -ne 0) { $bad++ }
        # Five fresh processes per load row: the interleaved mode with the same executable on both sides,
        # five rounds (the targets read the median of side B's rounds).
        if ((Invoke-Sub -Name 'load' -Pass 'once' -Specs @('load-dml', 'load-cuda', 'load-cpu', 'load-quality-cuda') -Body { return (Invoke-Gate -OutDir (Join-Path $once 'load') -Only @('load-dml', 'load-cuda', 'load-cpu', 'load-quality-cuda') -Extra $loadExtra -NoLock) }) -ne 0) { $bad++ }
        if ((Invoke-Sub -Name 'vram-dml' -Pass 'once' -Specs @('vram-dml') -Body { return (Invoke-Gate -OutDir (Join-Path $once 'vram-dml') -Only @('vram-dml') -NoLock) }) -ne 0) { $bad++ }
        if ((Invoke-Sub -Name 'vram-cuda' -Pass 'once' -Specs @('vram-cuda') -Body { return (Invoke-Gate -OutDir (Join-Path $once 'vram-cuda') -Only @('vram-cuda') -NoLock) }) -ne 0) { $bad++ }
        if ((Invoke-Sub -Name 'silent' -Pass 'once' -Specs @('silent-dml', 'silent-cuda', 'silent-cpu') -Body { return (Invoke-Gate -OutDir (Join-Path $once 'silent') -Only @('silent-dml', 'silent-cuda', 'silent-cpu') -NoLock) }) -ne 0) { $bad++ }
        return $bad
    }
    Invoke-Step -Name 'once-b' -Kind 'locked' -ExpectedMin 20 -Body {
        $bad = 0
        if ((Invoke-Sub -Name 'ctx512' -Pass 'once' -Specs @('sve-dml-ctx512') -Body { return (Invoke-Gate -OutDir (Join-Path $once 'ctx512') -Only @('sve-dml-ctx512') -NoLock) }) -ne 0) { $bad++ }
        if ((Invoke-Sub -Name 'knn' -Pass 'once' -Specs @('knn30k-cuda') -Body { return (Invoke-Gate -OutDir (Join-Path $once 'knn') -Only @('knn30k-cuda') -NoLock) }) -ne 0) { $bad++ }
        return $bad
    }
}

function Get-CorpusClips {
    # run_corpus.ps1's default clips (the sherpa-onnx test clips and the transcribe samples of the dev cache).
    return @(Get-ChildItem (Join-Path $devCache 'sherpa-onnx\clips') -Filter *.wav | Sort-Object Name | ForEach-Object FullName) +
           @(Get-ChildItem (Join-Path $devCache 'models\transcribe\samples') -Filter *.wav | Sort-Object Name | ForEach-Object FullName)
}

function Clear-StalePrivateLock {
    # The corpus runner's private lock file (<Root>/corpus.lock). Only the holder of the global GPU timing lock
    # runs the corpus, so a private lock found when a corpus step starts belongs to nobody who is still working:
    # a run that was stopped mid-way (a killed driver, an orchestrator pause) leaves it behind, and gpu-lock.ps1
    # breaks a dead owner's lock only after 30 minutes, which made the rerun wait out -LockWait and exit 1.
    # It is left alone when the process it names is alive (an unexpected second runner). Returns $true when a
    # lock file was removed.
    param([string]$Path)
    if (-not (Test-Path -LiteralPath $Path -PathType Leaf)) { return $false }
    $ownerPid = 0
    try {
        $rec = [System.IO.File]::ReadAllText($Path) | ConvertFrom-Json
        if ($null -ne $rec -and $rec -isnot [string] -and $rec.PSObject.Properties['pid'] -and $rec.pid) { $ownerPid = [int]$rec.pid }
    } catch { $ownerPid = 0 }
    if ($ownerPid -gt 0 -and $null -ne (Get-Process -Id $ownerPid -ErrorAction SilentlyContinue)) { return $false }
    Remove-Item -LiteralPath $Path -Force -ErrorAction SilentlyContinue
    return (-not (Test-Path -LiteralPath $Path))
}

function Invoke-Corpus {
    # One run_corpus.ps1 call inside a locked step. The step already holds the GPU timing lock, so the
    # runner (which takes the lock per batch and for its rescore) gets a private lock file through
    # SIGIL_GPU_LOCK (documented in gpu-lock.ps1) instead of waiting for ours.
    param([string]$Ep, [string]$Pitch, [string]$Preset, [string]$Out, [string[]]$Clips = @(), [switch]$SkipRescore, [string]$Log)
    $priv = Join-Path $Root 'corpus.lock'
    if (Clear-StalePrivateLock -Path $priv) { Write-Host "  removed the private corpus lock a stopped run left behind: $priv" }
    $old = $env:SIGIL_GPU_LOCK
    $env:SIGIL_GPU_LOCK = $priv
    try {
        $cArgs = @('-NoProfile', '-File', $corpusScript, '-Exe', $Exe, '-AppData', $AppData, '-Slot', $Slot, '-Ep', $Ep, '-Pitch', $Pitch, '-Preset', $Preset,
            '-Out', $Out, '-LockTag', "$Tag-corpus", '-LockWait', '5', '-Batch', '99')
        if ($Clips.Count -gt 0) { $cArgs += @('-Clips', ($Clips -join ';')) }
        if ($SkipRescore) { $cArgs += '-SkipRescore' }
        & pwsh @cArgs *> $Log
        return $LASTEXITCODE
    } finally {
        if ($null -eq $old) { Remove-Item Env:SIGIL_GPU_LOCK -ErrorAction SilentlyContinue } else { $env:SIGIL_GPU_LOCK = $old }
    }
}

function Step-Corpus {
    # T5 (scorer v2) per execution provider. Fidelity is a function of the audio, not of the clock, but the
    # runs are heavy: they hold the lock like every other benchmark run, in holds of 25-45 minutes (the CPU
    # corpus in two halves; the FCPE comparison on the first half only).
    $clips = Get-CorpusClips
    $half = [int][math]::Ceiling($clips.Count / 2)
    $first = @($clips | Select-Object -First $half)
    $second = @($clips | Select-Object -Skip $half)
    $plans = @()
    foreach ($name in $CorpusEps) {
        switch ($name) {
            'cuda' { $plans += , @{ step = 'corpus-cuda'; out = 'cuda'; ep = 'cuda'; pitch = 'rmvpe'; preset = 'low_latency'; clips = @(); skip = $false; min = 40 } }
            'dml' { $plans += , @{ step = 'corpus-dml'; out = 'dml'; ep = 'dml'; pitch = 'rmvpe'; preset = 'low_latency'; clips = @(); skip = $false; min = 40 } }
            'cpu' {
                $plans += , @{ step = 'corpus-cpu-a'; out = 'cpu'; ep = 'cpu'; pitch = 'swiftf0'; preset = 'streamer'; clips = $first; skip = $true; min = 45 }
                $plans += , @{ step = 'corpus-cpu-b'; out = 'cpu'; ep = 'cpu'; pitch = 'swiftf0'; preset = 'streamer'; clips = $second; skip = $false; min = 45 }
            }
            'cpu-fcpe' { $plans += , @{ step = 'corpus-cpu-fcpe'; out = 'cpu-fcpe'; ep = 'cpu'; pitch = 'fcpe'; preset = 'streamer'; clips = $first; skip = $false; min = 45 } }
            default { throw "unknown -CorpusEps '$name' (cuda, dml, cpu, cpu-fcpe)" }
        }
    }
    foreach ($pl in $plans) {
        $out = Join-Path $Root ("corpus\" + $pl.out)
        $log = Join-Path $Root ($pl.step + '.log')
        $cEp = $pl.ep; $cPitch = $pl.pitch; $cPreset = $pl.preset; $cClips = @($pl.clips); $cSkip = [bool]$pl.skip
        Invoke-Step -Name $pl.step -Kind 'locked' -ExpectedMin $pl.min -Body {
            return (Invoke-Corpus -Ep $cEp -Pitch $cPitch -Preset $cPreset -Out $out -Clips $cClips -SkipRescore:$cSkip -Log $log)
        }
    }
}

function Step-Soak {
    $out = Join-Path $Root 'gate\once\soak'
    Invoke-Step -Name 'soak-cuda' -Kind 'locked' -ExpectedMin 40 -Body {
        return (Invoke-Gate -OutDir $out -Only @('soak-cuda') -Vars @('SOAK=1') -NoLock)
    }
}

function Remove-DecisionSummary {
    # A decision session's output folder is reused by a rerun (-Force, or a retry after a crash): the summary
    # of the earlier run must not stand in for the new one, so the driver deletes it before the gate starts.
    param([string]$OutDir)
    foreach ($n in @('gate-summary.json', 'gate-summary.md')) {
        Remove-Item -LiteralPath (Join-Path $OutDir $n) -Force -ErrorAction SilentlyContinue
    }
}

function Test-DecisionComplete {
    # The session ran to its end: the gate wrote a FRESH summary (not older than -Since, the moment the session
    # started) with paired rows, and it lists every expected spec with at least the expected rounds (it writes
    # the summary last, after the final round). -Specs / -Rounds default to "any": a call without them only
    # checks freshness and paired rows.
    param([string]$OutDir, [datetime]$Since = [datetime]::MinValue, [string[]]$Specs = @(), [int]$Rounds = 0)
    $f = Join-Path $OutDir 'gate-summary.json'
    if (-not (Test-Path -LiteralPath $f)) { return $false }
    if ((Get-Item -LiteralPath $f).LastWriteTime -lt $Since) { return $false }
    try {
        $doc = Get-Content -LiteralPath $f -Raw | ConvertFrom-Json
        $pairedProp = $doc.PSObject.Properties['paired']
        if ($null -eq $pairedProp -or @($pairedProp.Value).Count -eq 0) { return $false }
        $roundsProp = $doc.PSObject.Properties['rounds']
        foreach ($s in $Specs) {
            $prop = if ($null -ne $roundsProp -and $null -ne $roundsProp.Value) { $roundsProp.Value.PSObject.Properties[$s] } else { $null }
            if ($null -eq $prop -or [int]$prop.Value -lt $Rounds) { return $false }
        }
        return $true
    } catch { return $false }
}

function Get-DecisionDefs {
    # name -> @{ only; vars (-CompareVars); env (-CompareEnv); metrics; rounds }
    $gpuMetrics = 'latency_ms_p50,latency_ms_p99,call_ms_p95,cpu_cores,underruns,over_budget_ratio'
    return [ordered]@{
        'devblock-cuda' = @{ only = 'devblock-cuda'; vars = 'BLOCK_FRAMES=480'; env = ''; metrics = $gpuMetrics; rounds = $DecideRounds }
        'devblock-dml'  = @{ only = 'devblock-dml'; vars = 'BLOCK_FRAMES=480'; env = ''; metrics = $gpuMetrics; rounds = $DecideRounds }
        'devblock-cpu'  = @{ only = 'devblock-cpu'; vars = 'BLOCK_FRAMES=480'; env = ''; metrics = 'latency_ms_p50,latency_ms_p99,call_ms_p95,cpu_cores,underruns,rtf_p99'; rounds = $DecideRounds }
        'cpu-int8'      = @{ only = 'cpu-streamer-lean,cpu-t6-lean'; vars = ''; env = 'SIGIL_VC_INT8=0'; metrics = 'rtf_p50,rtf_p99,call_ms_p95,cpu_cores,core_s_per_audio_s,content_ms_p50'; rounds = $DecideRounds }
        'cpu-f0'        = @{ only = 'cpu-streamer-lean'; vars = 'PITCH=fcpe'; env = ''; metrics = 'rtf_p50,rtf_p99,call_ms_p95,cpu_cores,f0_ms_p50'; rounds = $DecideRounds }
        # The same question on the thread policy P7-I0 ships (6 threads, below normal): FCPE is what the Streamer
        # button applies, the benchmark's CPU rows run SwiftF0. Delivery metrics ride along (what the output
        # queue played and trimmed), because a CPU row that is "fast enough" can still lose audio.
        # `runs` replaces the template's run list for this session (SIGIL_BENCH_RUNS_JSON, both sides alike): the
        # offline and paced rows of cpu-t6-lean plus the H3 game simulation, the load the thread policy was judged on.
        'cpu-t6-f0'     = @{ only = 'cpu-t6-lean'; vars = 'PITCH=fcpe'; env = ''; metrics = 'rtf_p50,rtf_p99,call_ms_p95,cpu_cores,f0_ms_p50,latency_ms_p99,underruns,output_trimmed_ms,underrun_ms,over_budget_ratio'; rounds = $DecideRounds
                          runs = '[{"mode":"offline","name":"offline","repeat":4},{"mode":"paced","name":"paced","seconds":45,"clock":"wall","warmup_s":1},{"mode":"contention","name":"h3-game","profile":"h3_cpu_game","seconds":45,"warmup_s":1}]' }
        'dml-shared'    = @{ only = 'dml-shared'; vars = ''; env = 'SIGIL_VC_DML_SHARED=1'; metrics = 'call_ms_p95,call_ms_p99,content_ms_p95'; rounds = $DecideRounds }
        'ctx512'        = @{ only = 'ctx-dml,ctx-cuda'; vars = 'EXTRA_MS=512'; env = ''; metrics = 'call_ms_p50,call_ms_p95'; rounds = 6 }
    }
}

function Step-Decide {
    # Each decision session is an ABBA run of one executable against itself with one switch on side B. The
    # session holds the lock for its whole length (the gate's per-pair locking would queue behind other
    # agents' runs once per pair): sessions of 10-30 minutes are grouped into holds of at most about 35.
    $defs = Get-DecisionDefs
    $groups = @(
        @('devblock-cuda'),
        @('devblock-dml'),
        @('devblock-cpu', 'ctx512'),
        @('dml-shared', 'cpu-f0'),
        @('cpu-int8'),
        @('cpu-t6-f0')
    )
    foreach ($n in $Decide) {
        if (-not $defs.Contains($n)) { throw "unknown decision session '$n' (known: $(@($defs.Keys) -join ', '))" }
    }
    foreach ($g in $groups) {
        $names = @($g | Where-Object { $Decide.Count -eq 0 -or $Decide -contains $_ })
        if ($names.Count -eq 0) { continue }
        Invoke-Step -Name ("decide-" + ($names -join '+')) -Kind 'locked' -ExpectedMin (15 * $names.Count + 10) -Body {
            $bad = 0
            foreach ($n in $names) {
                $d = $defs[$n]
                $out = Join-Path $Root "decisions\$n"
                $extra = @('-CompareExe', $Exe, '-Rounds', "$($d.rounds)", '-MaxRounds', "$($d.rounds)", '-AllowInconclusive', '-Metrics', $d.metrics)
                if ($d.vars) { $extra += @('-CompareVars', $d.vars) }
                if ($d.env) { $extra += @('-CompareEnv', $d.env) }
                $only = @($d.only -split ',')
                $wantRounds = [int]$d.rounds
                $c = Invoke-Sub -Name $n -Pass 'decide' -Specs $only -Body {
                    # The output folder may hold an earlier run's summary (a -Force rerun, a retry after a crash): clear it
                    # and remember when this session began, so only a summary this run wrote can count as its result.
                    Remove-DecisionSummary -OutDir $out
                    $began = (Get-Date).AddSeconds(-2)
                    if ($d.Contains('runs') -and $d.runs) { $env:SIGIL_BENCH_RUNS_JSON = $d.runs }
                    try { $gc = Invoke-Gate -OutDir $out -Only $only -Extra $extra -Vars @('DECIDE=1') -NoLock }
                    finally { if ($d.Contains('runs')) { Remove-Item Env:SIGIL_BENCH_RUNS_JSON -ErrorAction SilentlyContinue } }
                    # A candidate that REGRESSED makes the gate exit 1, but that is the answer of the session, not a
                    # failure: the session counts as done when this run wrote its summary with the paired rows and every
                    # expected round of every spec.
                    if ($gc -ne 0 -and (Test-DecisionComplete -OutDir $out -Since $began -Specs $only -Rounds $wantRounds)) { Write-Host "  [$n] the candidate regressed or was inconclusive on some rows (gate exit $gc): kept as a result"; return 0 }
                    return $gc
                }
                if ($c -ne 0) { $bad++ }
            }
            return $bad
        }
    }
}

function Step-Summary {
    & python (Join-Path $here 'final_scorecard.py') $Root
}

foreach ($s in $Step) {
    switch ($s) {
        'env' { Step-Env }
        'trial' { Step-Trial }
        'smoke' { Step-Smoke }
        'gate' { Step-Gate }
        'once' { Step-Once }
        'corpus' { Step-Corpus }
        'soak' { Step-Soak }
        'decide' { Step-Decide }
        'summary' { Step-Summary }
    }
}
