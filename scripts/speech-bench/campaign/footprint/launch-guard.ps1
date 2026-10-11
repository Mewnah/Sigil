# Launch guard of the footprint harness (dot-sourced by footprint.ps1 and diagnose.ps1; the user works on this PC while they run).
#
#   . (Join-Path $PSScriptRoot 'launch-guard.ps1')
#   Assert-StartsMinimized -Exe $Exe -ShowWindow:$ShowWindow        # before anything is written or started
#   ...
#   $env:SIGIL_START_MINIMIZED = '1'                                # (Set-StartMinimizedEnv does this)
#
# 1. A Sigil.exe built before commit dcf3bca ignores SIGIL_START_MINIMIZED and opens its window in front of whatever the user is
#    doing. Assert-StartsMinimized refuses such an exe (the same check as scripts/release/exit-probe.ps1) unless -ShowWindow
#    says the visible start is wanted.
# 2. The clean-up must stop only what THIS run started. Get-OwnedProcess / Stop-OwnedProcess pick the processes that descend from
#    the app process (parent pid chain, created after their parent), never a llama-server the user started themselves.

if (-not ('FootprintLaunch.Probe' -as [type])) {
  Add-Type -TypeDefinition @'
using System;
using System.IO;
using System.Text;
namespace FootprintLaunch {
  public static class Probe {
    // True when the file holds the ASCII text (the exe of a build that knows an environment switch).
    public static bool FileContains(string path, string text) {
      byte[] needle = Encoding.ASCII.GetBytes(text);
      byte[] buf = new byte[4 * 1024 * 1024 + needle.Length];
      using (FileStream fs = new FileStream(path, FileMode.Open, FileAccess.Read, FileShare.ReadWrite)) {
        int keep = 0;
        while (true) {
          int n = fs.Read(buf, keep, buf.Length - keep);
          if (n <= 0) return false;
          int total = keep + n;
          for (int i = 0; i + needle.Length <= total; i++) {
            if (buf[i] != needle[0]) continue;
            int j = 1;
            while (j < needle.Length && buf[i + j] == needle[j]) j++;
            if (j == needle.Length) return true;
          }
          keep = Math.Min(needle.Length - 1, total);
          Buffer.BlockCopy(buf, total - keep, buf, 0, keep);
        }
      }
    }
  }
}
'@
}

function Test-KnowsStartMinimized {
  param([Parameter(Mandatory)][string]$Exe)
  return [FootprintLaunch.Probe]::FileContains($Exe, 'SIGIL_START_MINIMIZED')
}

function Assert-StartsMinimized {
  param([Parameter(Mandatory)][string]$Exe, [switch]$ShowWindow)
  if ($ShowWindow) { return }
  if (-not (Test-KnowsStartMinimized -Exe $Exe)) {
    throw "this Sigil.exe predates SIGIL_START_MINIMIZED: starting it would put a window in front of whatever you are doing (rebuild it, or pass -ShowWindow to start it visibly anyway)"
  }
}

# The environment switch of the child (restored by the caller's finally block: it must be in the caller's saved list).
function Set-StartMinimizedEnv {
  param([switch]$ShowWindow)
  if ($ShowWindow) { Remove-Item Env:SIGIL_START_MINIMIZED -ErrorAction SilentlyContinue } else { $env:SIGIL_START_MINIMIZED = '1' }
}

# Pure selection (tested; callers wrap the call in @() so that zero or one rows stay an array): the rows of $Table (ProcessId, ParentProcessId, Name, CreationDate) that descend from $RootPid.
# A row counts as a child only when it was created no earlier than its parent (a recycled parent pid would otherwise adopt
# a stranger), and the root itself is not returned. An orphan whose parent has exited still names that pid, so it is found.
function Select-OwnedProcess {
  param([object[]]$Table, [Parameter(Mandatory)][int]$RootPid, [Parameter(Mandatory)][datetime]$RootStart, [string]$Name = '*')
  $start = @{ $RootPid = $RootStart }
  $queue = [System.Collections.Generic.Queue[int]]::new()
  $queue.Enqueue($RootPid)
  $found = [System.Collections.Generic.List[object]]::new()
  while ($queue.Count -gt 0) {
    $parent = $queue.Dequeue()
    foreach ($row in $Table) {
      if ([int]$row.ParentProcessId -ne $parent -or $start.ContainsKey([int]$row.ProcessId)) { continue }
      if ($row.CreationDate -lt $start[$parent].AddSeconds(-1)) { continue }
      $start[[int]$row.ProcessId] = $row.CreationDate
      $queue.Enqueue([int]$row.ProcessId)
      if ($row.Name -like $Name) { $found.Add($row) }
    }
  }
  return $found.ToArray()
}

function Get-OwnedProcess {
  param([Parameter(Mandatory)][int]$RootPid, [Parameter(Mandatory)][datetime]$RootStart, [string]$Name = '*')
  $table = @(Get-CimInstance -ClassName Win32_Process -ErrorAction SilentlyContinue)
  return Select-OwnedProcess -Table $table -RootPid $RootPid -RootStart $RootStart -Name $Name
}

# Stops the processes named $Name that descend from the app, plus the ones in $Known (a Get-OwnedProcess snapshot taken while the
# app was still alive, for children whose parent chain is gone by now). A pid whose start time differs from the snapshot's was recycled.
function Stop-OwnedProcess {
  param([Parameter(Mandatory)][int]$RootPid, [Parameter(Mandatory)][datetime]$RootStart, [string]$Name = '*', [object[]]$Known = @())
  $targets = @{}
  foreach ($r in @($Known) + @(Get-OwnedProcess -RootPid $RootPid -RootStart $RootStart -Name $Name)) {
    if ($null -ne $r) { $targets[[int]$r.ProcessId] = $r }
  }
  foreach ($r in $targets.Values) {
    try {
      $p = Get-Process -Id ([int]$r.ProcessId) -ErrorAction Stop
      if ([math]::Abs(($p.StartTime - $r.CreationDate).TotalSeconds) -lt 2) { Stop-Process -Id $p.Id -Force -ErrorAction SilentlyContinue }
    } catch {}
  }
}
