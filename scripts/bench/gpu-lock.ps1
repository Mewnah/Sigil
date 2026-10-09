<#
.SYNOPSIS
  The dev rig's GPU timing lock: one timing-sensitive run at a time (P3B rule R2).

.DESCRIPTION
  Every timing run (any `Sigil.exe --headless bench`, ignored GPU cargo tests, ORT profiling,
  Python microbenchmarks) holds this lock, so two agents or a build never share the GPU or the CPU
  during a measurement. The lock is a file created atomically (CreateNew) holding the owner tag,
  the owning process id (when one lives for the whole run), the start time and the expected
  duration:

      { "tag": "P3B-A", "pid": 1234, "start": "2026-10-01T18:00:00.0000000+02:00",
        "start_unix_ms": 1790870400000, "expected_minutes": 30, "host": "PC" }

  A lock written by hand (any text, e.g. "P6-B 18:02") is honoured too: its owner is the text and
  its start the file's last write time.

  Modes:
    -Acquire <tag>   take the lock (waiting up to -Wait minutes while another owner holds it,
                     polling every -PollSeconds); exit 0 when held, 1 when it timed out.
    -Release [<tag>] delete the lock (a tag that does not match the owner's is refused unless
                     -Force).
    -Run <tag> -Command '<command line>'
                     acquire, run the command in this session, release (also on failure); the
                     exit code is the command's (1 when the lock was not acquired).
    -Status          print the current owner (exit 0 free, 1 held).
    -PrintPath       print the lock file's path (the resolved one) and exit 0; for other scripts that
                     need to find the lock (no lock is taken).
    -SelfTest        exercise acquire / contention / release / the stale rule on a scratch lock.

  Fair queue (P3B-A): a waiting -Acquire (or -Run) takes a ticket in <lock path>.queue/ and takes
  the lock only when no live waiter's ticket is older than its own, so an owner that releases and
  re-acquires at once (an ABBA series) cannot starve the others: its next acquire queues behind
  them. A ticket of a process that is gone (or older than 4 h) is ignored and removed. An
  -Acquire with -Wait 0 takes no ticket and does not jump the queue: it fails while anyone waits.
  A waiter polls every -PollSeconds while the lock is held, and every second while the lock is
  free but it is not yet its turn.

  Stale rule: a lock may be broken (with a log line) only when its recorded pid is gone and it is
  older than 30 minutes, or when it is older than 4 hours whatever its pid. A lock without a pid
  (taken by -Acquire, whose process exits right away, or written by hand) is broken only by the
  4-hour rule. Never break a lock by deleting the file by hand while its owner may be measuring.
  Breaking is race-free against another breaker: only the waiter at the head of the queue (or one
  that finds nobody waiting) breaks, and it moves the observed file aside atomically, checks that
  what it moved is the record it judged stale, and puts a fresh lock back when it is not (it never
  deletes the lock another waiter just created in the stale one's place).

  Lock path: -LockPath, else $env:SIGIL_GPU_LOCK, else the path in the one-line file
  %LOCALAPPDATA%\sigil-bench\gpu_lock_path.txt (how the concurrent agent phases of a shared rig
  agree on one lock without a path in the repository; ignored, with a log line, while that path's
  folder does not exist), else %LOCALAPPDATA%\sigil-bench\gpu_timing.lock. Export
  SIGIL_GPU_LOCK for a session that should use another lock. Never hold the lock across a build.

.EXAMPLE
  pwsh scripts/bench/gpu-lock.ps1 -Acquire P3B-A -Wait 60
  & $exe --headless bench spec.json --out out
  pwsh scripts/bench/gpu-lock.ps1 -Release P3B-A

.EXAMPLE
  pwsh scripts/bench/gpu-lock.ps1 -Run P3B-E -Wait 60 -Command "cargo test --release rmvpe_plan -- --ignored"
#>
[CmdletBinding(DefaultParameterSetName = 'Status')]
param(
    [Parameter(ParameterSetName = 'Acquire', Mandatory = $true)]
    [string]$Acquire,
    [Parameter(ParameterSetName = 'Release')]
    [switch]$Release,
    [Parameter(ParameterSetName = 'Release', Position = 0)]
    [string]$ReleaseTag,
    [Parameter(ParameterSetName = 'Run', Mandatory = $true)]
    [string]$Run,
    [Parameter(ParameterSetName = 'Run', Mandatory = $true)]
    [string]$Command,
    [Parameter(ParameterSetName = 'Status')]
    [switch]$Status,
    [Parameter(ParameterSetName = 'PrintPath')]
    [switch]$PrintPath,
    [Parameter(ParameterSetName = 'SelfTest')]
    [switch]$SelfTest,
    # Minutes to wait while another owner holds the lock (0 = do not wait).
    [double]$Wait = 0,
    [int]$PollSeconds = 5,
    # Expected duration of the run (written into the lock for whoever waits on it).
    [double]$ExpectedMinutes = 30,
    # The process that owns the lock for its whole duration (-Run records its own pid).
    [int]$OwnerPid = 0,
    [string]$LockPath,
    [switch]$Force
)

$ErrorActionPreference = 'Stop'
Set-StrictMode -Version 3.0

# No machine- or session-specific path lives in this script. The per-user default is under
# %LOCALAPPDATA%; a shared rig points every agent at one lock with SIGIL_GPU_LOCK or the one-line
# pointer file (the lock path as its first line).
$BenchDataDir = Join-Path ([Environment]::GetFolderPath('LocalApplicationData')) 'sigil-bench'
$DefaultLockPath = Join-Path $BenchDataDir 'gpu_timing.lock'
$PointerFile = Join-Path $BenchDataDir 'gpu_lock_path.txt'
$StalePidGoneMinutes = 30
$StaleAnyMinutes = 240

function Resolve-LockPath {
    param([string]$Given)
    if ($Given) { return $Given }
    if ($env:SIGIL_GPU_LOCK) { return $env:SIGIL_GPU_LOCK }
    if (Test-Path -LiteralPath $PointerFile -PathType Leaf) {
        $first = ''
        try { $first = (([System.IO.File]::ReadAllText($PointerFile) -split "`r?`n")[0]).Trim() } catch { $first = '' }
        if ($first) {
            $parent = Split-Path -Parent $first
            if (-not $parent -or (Test-Path -LiteralPath $parent)) { return $first }
            Write-Log "$PointerFile points at '$first' but its folder does not exist; using $DefaultLockPath"
        }
    }
    return $DefaultLockPath
}

function Write-Log {
    param([string]$Message)
    Write-Host ("[gpu-lock {0}] {1}" -f (Get-Date -Format 'HH:mm:ss'), $Message)
}

function Read-Lock {
    # The current lock's owner record, or $null when there is no lock. A lock written by hand (not
    # JSON) reads as an owner without pid whose start is the file's last write time.
    param([string]$Path)
    if (-not (Test-Path -LiteralPath $Path -PathType Leaf)) { return $null }
    $item = Get-Item -LiteralPath $Path -ErrorAction SilentlyContinue
    if ($null -eq $item) { return $null }
    $text = ''
    try { $text = [System.IO.File]::ReadAllText($Path) } catch { $text = '' }
    $doc = $null
    try { if ($text.Trim()) { $doc = $text | ConvertFrom-Json } } catch { $doc = $null }
    $start = $item.LastWriteTime
    $tag = $text.Trim()
    $ownerPid = 0
    $expected = $null
    if ($null -ne $doc -and $doc -isnot [string]) {
        $p = $doc.PSObject.Properties
        if ($p['tag']) { $tag = [string]$doc.tag }
        if ($p['pid'] -and $doc.pid) { $ownerPid = [int]$doc.pid }
        if ($p['start_unix_ms'] -and $doc.start_unix_ms) {
            # The machine-readable start (ConvertFrom-Json turns ISO strings into DateTime values
            # whose kind varies between PowerShell versions).
            $start = [DateTimeOffset]::FromUnixTimeMilliseconds([long]$doc.start_unix_ms).LocalDateTime
        } elseif ($p['start'] -and $doc.start) {
            $s = $doc.start
            try {
                $start = if ($s -is [datetime]) { $s.ToLocalTime() } elseif ($s -is [DateTimeOffset]) { $s.LocalDateTime } else { [DateTimeOffset]::Parse([string]$s, [Globalization.CultureInfo]::InvariantCulture).LocalDateTime }
            } catch { }
        }
        if ($p['expected_minutes']) { $expected = $doc.expected_minutes }
    }
    if ($tag.Length -gt 80) { $tag = $tag.Substring(0, 80) }
    return [pscustomobject]@{ tag = $tag; pid = $ownerPid; start = $start; expected_minutes = $expected; path = $Path }
}

function Test-PidAlive {
    param([int]$Id)
    if ($Id -le 0) { return $true }
    return $null -ne (Get-Process -Id $Id -ErrorAction SilentlyContinue)
}

function Get-StaleReason {
    # Why the lock may be broken, or $null when it must be respected.
    param($Owner, [datetime]$Now)
    $age = ($Now - $Owner.start).TotalMinutes
    if ($age -gt $StaleAnyMinutes) {
        return ("older than {0} h ({1:N0} min)" -f ($StaleAnyMinutes / 60), $age)
    }
    if ($Owner.pid -gt 0 -and -not (Test-PidAlive -Id $Owner.pid) -and $age -gt $StalePidGoneMinutes) {
        return ("its process {0} is gone and it is older than {1} min ({2:N0} min)" -f $Owner.pid, $StalePidGoneMinutes, $age)
    }
    return $null
}

function Remove-StaleLock {
    # Break the stale lock `Owner` was read from without removing a lock that replaced it. Moving
    # the file aside is atomic, so of several breakers of the same record only one's move succeeds;
    # the moved file is then checked against the record that was judged stale, and put back when it
    # is a different (fresh) lock that a faster waiter created in between. $true when the stale
    # record was removed.
    param([string]$Path, $Owner)
    $aside = '{0}.stale-{1}-{2}' -f $Path, $PID, ([guid]::NewGuid().ToString('N').Substring(0, 8))
    try { [System.IO.File]::Move($Path, $aside) } catch { return $false }
    $moved = Read-Lock -Path $aside
    $same = $null -ne $moved -and $moved.tag -eq $Owner.tag -and $moved.pid -eq $Owner.pid -and
        [Math]::Abs(([datetime]$moved.start - [datetime]$Owner.start).TotalMilliseconds) -lt 1000
    if ($same) {
        Remove-Item -LiteralPath $aside -Force -ErrorAction SilentlyContinue
        return $true
    }
    # Someone else's fresh lock: put it back (a move never overwrites).
    try {
        [System.IO.File]::Move($aside, $Path)
    } catch {
        Write-Log "could not put back the lock moved aside as '$aside' (another lock took its place); it is a live lock: keep it until its owner is done"
    }
    return $false
}

function New-LockFile {
    # Atomic create; $true when this call created the lock.
    param([string]$Path, [string]$Tag, [int]$Id, [double]$Minutes)
    $dir = Split-Path -Parent $Path
    if ($dir -and -not (Test-Path -LiteralPath $dir)) { New-Item -ItemType Directory -Force -Path $dir | Out-Null }
    $record = [ordered]@{
        tag = $Tag
        pid = $(if ($Id -gt 0) { $Id } else { $null })
        start = (Get-Date).ToString('o')
        start_unix_ms = [DateTimeOffset]::Now.ToUnixTimeMilliseconds()
        expected_minutes = $Minutes
        host = $env:COMPUTERNAME
    } | ConvertTo-Json -Compress
    try {
        $fs = [System.IO.File]::Open($Path, [System.IO.FileMode]::CreateNew, [System.IO.FileAccess]::Write, [System.IO.FileShare]::None)
    } catch [System.IO.IOException] {
        return $false
    }
    try {
        $bytes = [System.Text.Encoding]::UTF8.GetBytes($record)
        $fs.Write($bytes, 0, $bytes.Length)
    } finally {
        $fs.Dispose()
    }
    return $true
}

# ---- fair queue -----------------------------------------------------------------------------------

function Get-QueueDir {
    param([string]$Path)
    return "$Path.queue"
}

function New-Ticket {
    # A waiter's place in the queue: <queue>/<unix ms, 15 digits>-<pid>-<random>.ticket (names sort
    # by arrival). $null when the queue folder cannot be written (the lock then works unqueued).
    param([string]$Path, [string]$Tag)
    $dir = Get-QueueDir -Path $Path
    try {
        if (-not (Test-Path -LiteralPath $dir)) { New-Item -ItemType Directory -Force -Path $dir | Out-Null }
        $name = '{0:D15}-{1}-{2}.ticket' -f [DateTimeOffset]::Now.ToUnixTimeMilliseconds(), $PID, ([guid]::NewGuid().ToString('N').Substring(0, 8))
        $file = Join-Path $dir $name
        [System.IO.File]::WriteAllText($file, (([ordered]@{ tag = $Tag; pid = $PID }) | ConvertTo-Json -Compress))
        return $file
    } catch {
        return $null
    }
}

function Get-Waiters {
    # The live waiters' tickets, oldest first (tickets of processes that are gone, or older than
    # 4 h, are removed).
    param([string]$Path)
    $dir = Get-QueueDir -Path $Path
    if (-not (Test-Path -LiteralPath $dir)) { return , @() }
    $nowMs = [DateTimeOffset]::Now.ToUnixTimeMilliseconds()
    $live = @()
    foreach ($f in @(Get-ChildItem -LiteralPath $dir -Filter '*.ticket' -ErrorAction SilentlyContinue | Sort-Object Name)) {
        $parts = $f.BaseName.Split('-')
        $ms = [long]0
        $tpid = 0
        $ok = $parts.Count -ge 2 -and [long]::TryParse($parts[0], [ref]$ms) -and [int]::TryParse($parts[1], [ref]$tpid)
        $tooOld = $ok -and ($nowMs - $ms) -gt ($StaleAnyMinutes * 60000)
        if (-not $ok -or $tooOld -or -not (Test-PidAlive -Id $tpid)) {
            Remove-Item -LiteralPath $f.FullName -Force -ErrorAction SilentlyContinue
            continue
        }
        $live += $f
    }
    return , $live
}

function Test-MyTurn {
    # No live waiter is ahead of `Ticket` (without a ticket: nobody waits at all).
    param([string]$Path, [string]$Ticket)
    # Get-Waiters returns its array as one object (`return , $live`): no @() around the call.
    $waiters = Get-Waiters -Path $Path
    if ($waiters.Count -eq 0) { return $true }
    if (-not $Ticket) { return $false }
    $mine = Split-Path -Leaf $Ticket
    foreach ($w in $waiters) {
        if ($w.Name -eq $mine) { return $true }
        if ([string]::CompareOrdinal($w.Name, $mine) -lt 0) { return $false }
    }
    return $true
}

function Invoke-Acquire {
    param([string]$Path, [string]$Tag, [double]$WaitMinutes, [int]$Poll, [int]$Id, [double]$Minutes)
    if (-not $Tag.Trim()) { throw 'a tag is required' }
    $deadline = (Get-Date).AddMinutes([math]::Max(0.0, [double]$WaitMinutes))
    $announced = $false
    $ticket = $null
    if ($WaitMinutes -gt 0) { $ticket = New-Ticket -Path $Path -Tag $Tag }
    try {
        while ($true) {
            $turn = Test-MyTurn -Path $Path -Ticket $ticket
            if ($turn -and (New-LockFile -Path $Path -Tag $Tag -Id $Id -Minutes $Minutes)) {
                Write-Log "acquired by $Tag ($Path)"
                return $true
            }
            $owner = Read-Lock -Path $Path
            if ($null -eq $owner -and $turn) { continue } # released between the two calls: try again
            if ($null -ne $owner) {
                $why = Get-StaleReason -Owner $owner -Now (Get-Date)
                # Only the waiter at the head of the queue (or one that finds nobody waiting) breaks:
                # two waiters judging the same stale record must not both remove a lock.
                if ($null -ne $why -and $turn) {
                    Write-Log ("breaking the stale lock of '{0}' (started {1:o}): {2}" -f $owner.tag, $owner.start, $why)
                    if (Remove-StaleLock -Path $Path -Owner $owner) { continue }
                }
            }
            $holder = if ($null -ne $owner) { "held by '{0}' since {1:HH:mm:ss}" -f $owner.tag, $owner.start } else { 'free, another waiter is first' }
            if ((Get-Date) -ge $deadline) {
                $expected = if ($null -ne $owner -and $owner.expected_minutes) { " (expected $($owner.expected_minutes) min)" } else { '' }
                Write-Log ("{0}{1}; not acquired" -f $holder, $expected)
                return $false
            }
            if (-not $announced) {
                $queuedText = if ($ticket) { ' (queued)' } else { '' }
                Write-Log ("{0}; waiting up to {1} min{2}" -f $holder, $WaitMinutes, $queuedText)
                $announced = $true
            }
            $left = ($deadline - (Get-Date)).TotalSeconds
            $step = if ($null -eq $owner) { 1 } else { $Poll }
            Start-Sleep -Seconds ([math]::Max(1, [math]::Min($step, [math]::Ceiling($left))))
        }
    } finally {
        if ($ticket) { Remove-Item -LiteralPath $ticket -Force -ErrorAction SilentlyContinue }
    }
}

function Invoke-Release {
    param([string]$Path, [string]$Tag, [switch]$Forced)
    $owner = Read-Lock -Path $Path
    if ($null -eq $owner) {
        Write-Log 'not held'
        return $true
    }
    if ($Tag -and $owner.tag -ne $Tag -and -not $Forced) {
        Write-Log "held by '$($owner.tag)', not '$Tag'; not released (-Force to release anyway)"
        return $false
    }
    Remove-Item -LiteralPath $Path -Force
    Write-Log "released ($($owner.tag))"
    return $true
}

function Invoke-SelfTest {
    $tmp = Join-Path ([System.IO.Path]::GetTempPath()) ("sigil-gpu-lock-selftest-{0}.lock" -f $PID)
    $failures = @()
    $check = {
        param([bool]$Ok, [string]$What)
        if ($Ok) { Write-Host "  ok   $What" } else { Write-Host "  FAIL $What"; $script:selfTestFailed = $true }
    }
    $script:selfTestFailed = $false
    Remove-Item -LiteralPath $tmp -Force -ErrorAction SilentlyContinue
    try {
        & $check (Invoke-Acquire -Path $tmp -Tag 'self-a' -WaitMinutes 0 -Poll 1 -Id 0 -Minutes 1) 'acquire a free lock'
        $owner = Read-Lock -Path $tmp
        & $check ($null -ne $owner -and $owner.tag -eq 'self-a') 'the lock records its owner'
        $t0 = Get-Date
        $second = Invoke-Acquire -Path $tmp -Tag 'self-b' -WaitMinutes 0.05 -Poll 1 -Id 0 -Minutes 1
        $waited = ((Get-Date) - $t0).TotalSeconds
        & $check ((-not $second) -and $waited -ge 2) ("a second acquire waits, then times out ({0:N1} s)" -f $waited)
        & $check (-not (Invoke-Release -Path $tmp -Tag 'self-b')) 'another tag cannot release it'
        & $check (Invoke-Release -Path $tmp -Tag 'self-a') 'the owner releases it'
        & $check (-not (Test-Path -LiteralPath $tmp)) 'the lock file is gone'

        # Stale rule: a dead pid older than 30 min is broken; a live pid is not; no pid needs 4 h.
        $dead = 0
        foreach ($candidate in 999990..999999) { if (-not (Test-PidAlive -Id $candidate)) { $dead = $candidate; break } }
        $write = {
            param([int]$Id, [double]$AgeMinutes)
            $at = [DateTimeOffset]::Now.AddMinutes(-$AgeMinutes)
            $rec = [ordered]@{ tag = 'old'; pid = $(if ($Id -gt 0) { $Id } else { $null }); start = $at.ToString('o'); start_unix_ms = $at.ToUnixTimeMilliseconds(); expected_minutes = 5 }
            [System.IO.File]::WriteAllText($tmp, ($rec | ConvertTo-Json -Compress))
        }
        & $write $dead 31
        & $check (Invoke-Acquire -Path $tmp -Tag 'self-c' -WaitMinutes 0 -Poll 1 -Id 0 -Minutes 1) 'a lock whose pid is gone and older than 30 min is broken'
        [void](Invoke-Release -Path $tmp -Tag 'self-c')
        & $write $dead 10
        & $check (-not (Invoke-Acquire -Path $tmp -Tag 'self-d' -WaitMinutes 0 -Poll 1 -Id 0 -Minutes 1)) 'a dead pid younger than 30 min is respected'
        & $write $PID 60
        & $check (-not (Invoke-Acquire -Path $tmp -Tag 'self-e' -WaitMinutes 0 -Poll 1 -Id 0 -Minutes 1)) 'a live pid is respected'
        & $write 0 60
        & $check (-not (Invoke-Acquire -Path $tmp -Tag 'self-f' -WaitMinutes 0 -Poll 1 -Id 0 -Minutes 1)) 'a lock without pid is respected for 4 h'
        & $write 0 241
        & $check (Invoke-Acquire -Path $tmp -Tag 'self-g' -WaitMinutes 0 -Poll 1 -Id 0 -Minutes 1) 'any lock older than 4 h is broken'
        [void](Invoke-Release -Path $tmp -Tag 'self-g')
        # A hand-written lock (not JSON) counts from its file time.
        [System.IO.File]::WriteAllText($tmp, 'P6-X timing run')
        $owner = Read-Lock -Path $tmp
        & $check ($null -ne $owner -and $owner.tag -eq 'P6-X timing run' -and $owner.pid -eq 0) 'a hand-written lock is read as an owner without pid'
        & $check (-not (Invoke-Acquire -Path $tmp -Tag 'self-h' -WaitMinutes 0 -Poll 1 -Id 0 -Minutes 1)) 'a fresh hand-written lock is respected'
        Remove-Item -LiteralPath $tmp -Force

        # Fair queue: a live waiter ahead keeps a free lock from a newcomer; a waiter that is gone
        # does not; a queued acquire takes the lock once it is first.
        $q = Get-QueueDir -Path $tmp
        New-Item -ItemType Directory -Force -Path $q | Out-Null
        $older = Join-Path $q ('{0:D15}-{1}-aaaaaaaa.ticket' -f ([DateTimeOffset]::Now.ToUnixTimeMilliseconds() - 5000), $PID)
        [System.IO.File]::WriteAllText($older, '{}')
        & $check (-not (Invoke-Acquire -Path $tmp -Tag 'self-i' -WaitMinutes 0 -Poll 1 -Id 0 -Minutes 1)) 'a free lock is not taken past a live waiter'
        & $check (-not (Invoke-Acquire -Path $tmp -Tag 'self-j' -WaitMinutes 0.04 -Poll 1 -Id 0 -Minutes 1)) 'a queued acquire behind a live waiter times out'
        & $check (-not (Test-Path -LiteralPath $tmp)) 'and leaves the lock free'
        & $check (@(Get-ChildItem -LiteralPath $q -Filter '*.ticket').Count -eq 1) 'a timed-out waiter removes its ticket'
        Remove-Item -LiteralPath $older -Force
        $ghost = Join-Path $q ('{0:D15}-{1}-bbbbbbbb.ticket' -f ([DateTimeOffset]::Now.ToUnixTimeMilliseconds() - 5000), $dead)
        [System.IO.File]::WriteAllText($ghost, '{}')
        & $check (Invoke-Acquire -Path $tmp -Tag 'self-k' -WaitMinutes 0.04 -Poll 1 -Id 0 -Minutes 1) 'the ticket of a waiter that is gone does not block'
        & $check (-not (Test-Path -LiteralPath $ghost)) 'and is removed'
        [void](Invoke-Release -Path $tmp -Tag 'self-k')
        & $check (Test-MyTurn -Path $tmp -Ticket $null) 'an empty queue: anyone may take the lock'

        # Breaking a stale lock never removes the lock that replaced it.
        & $write $dead 31
        $stale = Read-Lock -Path $tmp
        Remove-Item -LiteralPath $tmp -Force
        [void](New-LockFile -Path $tmp -Tag 'fresh' -Id 0 -Minutes 1)
        & $check (-not (Remove-StaleLock -Path $tmp -Owner $stale)) 'a fresh lock that replaced the stale one is not removed'
        $kept = Read-Lock -Path $tmp
        & $check ($null -ne $kept -and $kept.tag -eq 'fresh') 'and stays in place'
        $leftovers = @(Get-ChildItem -LiteralPath (Split-Path -Parent $tmp) -Filter ((Split-Path -Leaf $tmp) + '.stale-*') -ErrorAction SilentlyContinue)
        & $check ($leftovers.Count -eq 0) 'leaving no file moved aside'
        Remove-Item -LiteralPath $tmp -Force
        & $write $dead 31
        $stale = Read-Lock -Path $tmp
        & $check (Remove-StaleLock -Path $tmp -Owner $stale) 'the stale record that was read is removed'
        & $check (-not (Test-Path -LiteralPath $tmp)) 'and the lock is free'
        & $check (-not (Remove-StaleLock -Path $tmp -Owner $stale)) 'a second breaker of the same record finds nothing to move'
        # Only the head of the queue breaks.
        New-Item -ItemType Directory -Force -Path $q | Out-Null
        $ahead = Join-Path $q ('{0:D15}-{1}-cccccccc.ticket' -f ([DateTimeOffset]::Now.ToUnixTimeMilliseconds() - 5000), $PID)
        [System.IO.File]::WriteAllText($ahead, '{}')
        & $write $dead 31
        & $check (-not (Invoke-Acquire -Path $tmp -Tag 'self-m' -WaitMinutes 0.04 -Poll 1 -Id 0 -Minutes 1)) 'a waiter that is not first does not break a stale lock'
        $stillThere = Read-Lock -Path $tmp
        & $check ($null -ne $stillThere -and $stillThere.tag -eq 'old') 'the stale lock is untouched'
        Remove-Item -LiteralPath $ahead -Force
        & $check (Invoke-Acquire -Path $tmp -Tag 'self-n' -WaitMinutes 0 -Poll 1 -Id 0 -Minutes 1) 'with nobody ahead the stale lock is broken and taken'
        [void](Invoke-Release -Path $tmp -Tag 'self-n')

        # Lock path resolution: -LockPath, SIGIL_GPU_LOCK, the pointer file (while its folder exists), the per-user default.
        $savedEnv = $env:SIGIL_GPU_LOCK
        $savedPointer = $script:PointerFile
        $pointerTmp = Join-Path ([System.IO.Path]::GetTempPath()) ("sigil-gpu-lock-pointer-{0}.txt" -f $PID)
        try {
            $env:SIGIL_GPU_LOCK = $null
            $script:PointerFile = $pointerTmp
            & $check ((Resolve-LockPath -Given '') -eq $DefaultLockPath) 'no pointer: the per-user default'
            [System.IO.File]::WriteAllText($pointerTmp, ((Join-Path ([System.IO.Path]::GetTempPath()) 'shared.lock') + "`r`nignored"))
            & $check ((Resolve-LockPath -Given '') -eq (Join-Path ([System.IO.Path]::GetTempPath()) 'shared.lock')) 'the pointer file names the lock'
            [System.IO.File]::WriteAllText($pointerTmp, 'C:\sigil-no-such-folder-9f3a\gpu_timing.lock')
            & $check ((Resolve-LockPath -Given '') -eq $DefaultLockPath) 'a pointer into a missing folder falls back to the default'
            [System.IO.File]::WriteAllText($pointerTmp, (Join-Path ([System.IO.Path]::GetTempPath()) 'shared.lock'))
            $env:SIGIL_GPU_LOCK = 'X:\env.lock'
            & $check ((Resolve-LockPath -Given '') -eq 'X:\env.lock') 'SIGIL_GPU_LOCK wins over the pointer'
            & $check ((Resolve-LockPath -Given 'Y:\given.lock') -eq 'Y:\given.lock') '-LockPath wins over everything'
            & $check ($DefaultLockPath -notlike '*claude*' -and $DefaultLockPath -like '*sigil-bench*') 'the default carries no session path'
        } finally {
            $env:SIGIL_GPU_LOCK = $savedEnv
            $script:PointerFile = $savedPointer
            Remove-Item -LiteralPath $pointerTmp -Force -ErrorAction SilentlyContinue
        }
    } finally {
        Remove-Item -LiteralPath $tmp -Force -ErrorAction SilentlyContinue
        Remove-Item -LiteralPath (Get-QueueDir -Path $tmp) -Recurse -Force -ErrorAction SilentlyContinue
    }
    if ($script:selfTestFailed) { Write-Host 'gpu-lock self-test FAILED'; return $false }
    Write-Host 'gpu-lock self-test passed'
    return $true
}

$path = Resolve-LockPath -Given $LockPath
switch ($PSCmdlet.ParameterSetName) {
    'PrintPath' {
        Write-Output $path
        exit 0
    }
    'Acquire' {
        if (Invoke-Acquire -Path $path -Tag $Acquire -WaitMinutes $Wait -Poll $PollSeconds -Id $OwnerPid -Minutes $ExpectedMinutes) { exit 0 } else { exit 1 }
    }
    'Release' {
        if (Invoke-Release -Path $path -Tag $ReleaseTag -Forced:$Force) { exit 0 } else { exit 1 }
    }
    'Run' {
        if (-not (Invoke-Acquire -Path $path -Tag $Run -WaitMinutes $Wait -Poll $PollSeconds -Id $PID -Minutes $ExpectedMinutes)) { exit 1 }
        $code = 0
        try {
            $global:LASTEXITCODE = 0
            Invoke-Expression $Command
            $code = if ($null -ne $LASTEXITCODE) { $LASTEXITCODE } else { 0 }
            if (-not $?) { $code = [math]::Max(1, $code) }
        } catch {
            Write-Log "the command failed: $_"
            $code = 1
        } finally {
            [void](Invoke-Release -Path $path -Tag $Run)
        }
        exit $code
    }
    'SelfTest' {
        if (Invoke-SelfTest) { exit 0 } else { exit 1 }
    }
    default {
        $owner = Read-Lock -Path $path
        $queued = Get-Waiters -Path $path
        if ($queued.Count -gt 0) { Write-Log ("{0} waiter(s) queued" -f $queued.Count) }
        if ($null -eq $owner) { Write-Log "free ($path)"; exit 0 }
        $why = Get-StaleReason -Owner $owner -Now (Get-Date)
        Write-Log ("held by '{0}' since {1:o}{2}{3}" -f $owner.tag, $owner.start, $(if ($owner.pid -gt 0) { ", pid $($owner.pid)" } else { '' }), $(if ($why) { " (stale: $why)" } else { '' }))
        exit 1
    }
}
