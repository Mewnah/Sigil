#Requires -Version 7.2
<#
.SYNOPSIS
  Self-test of build-runtime-packs.ps1's helpers (no build, no CMake, no network).

.DESCRIPTION
  Loads the helper functions of build-runtime-packs.ps1 without running the script and checks:
    - the build output is taken from the path CMake recorded for -Config, never "the newest DLL"
      (a stale Debug build next to an up-to-date Release one);
    - a checker that dies without printing JSON (illegal instruction, crash) is reported with its
      exit status instead of a JSON parse error, and --variant / --dll-dir / --no-create are passed;
    - debug CRT and OpenMP imports are recognized (and release CRT / CUDA / system DLLs are not);
    - -NoCreate reaches the checker.

  pwsh scripts/native/test-build-runtime-packs.ps1     # exit code 0 = all passed
#>
[CmdletBinding()]
param()

$ErrorActionPreference = 'Stop'
Set-StrictMode -Version 3.0

$script = Join-Path $PSScriptRoot 'build-runtime-packs.ps1'
$errors = $null
$ast = [System.Management.Automation.Language.Parser]::ParseFile($script, [ref]$null, [ref]$errors)
if ($errors) { throw "build-runtime-packs.ps1 does not parse: $($errors -join '; ')" }
foreach ($fn in $ast.FindAll({ param($n) $n -is [System.Management.Automation.Language.FunctionDefinitionAst] }, $false)) {
    . ([scriptblock]::Create($fn.Extent.Text))
}
foreach ($assign in $ast.EndBlock.Statements | Where-Object {
        $_ -is [System.Management.Automation.Language.AssignmentStatementAst] -and
        $_.Left.Extent.Text -in '$DebugCrtPattern', '$OpenMpPattern'
    }) {
    . ([scriptblock]::Create($assign.Extent.Text))
}

# Script parameters the helpers read.
$Config = 'Release'
$Generator = $null
$NoCreate = $false

$failures = [System.Collections.Generic.List[string]]::new()
function Check([bool] $Condition, [string] $What) {
    if ($Condition) { Write-Host "  ok   $What" } else { Write-Host "  FAIL $What" -ForegroundColor Red; $failures.Add($What) }
}
function Throws([scriptblock] $Body) {
    try { & $Body | Out-Null; return $null } catch { return $_.Exception.Message }
}

$tmp = Join-Path ([IO.Path]::GetTempPath()) "sigil-brp-test-$PID"
New-Item -ItemType Directory -Force -Path $tmp | Out-Null
try {
    Write-Host 'Get-BuildOutput'
    $tree = Join-Path $tmp 'tree'
    foreach ($cfg in 'Release', 'Debug') {
        New-Item -ItemType Directory -Force -Path (Join-Path $tree 'engines' 'x' $cfg), (Join-Path $tree 'sigil-outputs' $cfg) | Out-Null
        $dll = Join-Path $tree 'engines' 'x' $cfg 'sigil_x.dll'
        Set-Content -LiteralPath $dll -Value $cfg
        # CMake writes forward slashes and no newline.
        [IO.File]::WriteAllText((Join-Path $tree 'sigil-outputs' $cfg 'sigil_x.path'), ($dll -replace '\\', '/'))
    }
    # The Debug DLL is newer (MSBuild did not relink the up-to-date Release one).
    (Get-Item (Join-Path $tree 'engines' 'x' 'Release' 'sigil_x.dll')).LastWriteTime = (Get-Date).AddHours(-1)
    $picked = Get-BuildOutput $tree 'sigil_x'
    Check ($picked.Directory.Name -eq 'Release') "Release is picked although Debug is newer ($($picked.FullName))"
    $Config = 'Debug'
    Check ((Get-BuildOutput $tree 'sigil_x').Directory.Name -eq 'Debug') 'Debug when -Config Debug'
    $Config = 'RelWithDebInfo'
    $msg = Throws { Get-BuildOutput $tree 'sigil_x' }
    Check ($msg -and $msg -match 'did not record the RelWithDebInfo output') "no record: an error, not a search ($msg)"
    $Config = 'Release'
    Remove-Item -LiteralPath (Join-Path $tree 'engines' 'x' 'Release' 'sigil_x.dll')
    $msg = Throws { Get-BuildOutput $tree 'sigil_x' }
    Check ($msg -and $msg -match 'produced no sigil_x') "recorded but not built: an error ($msg)"

    Write-Host 'Invoke-EngineCheck'
    $argLog = Join-Path $tmp 'args.txt'
    $fake = {
        param([string] $Name, [string] $Body)
        $p = Join-Path $tmp $Name
        Set-Content -LiteralPath $p -Value "@echo off`r`necho %*> `"$argLog`"`r`n$Body" -Encoding ascii
        return $p
    }
    $illegal = & $fake 'illegal.cmd' "echo [engine info] starting 1>&2`r`nexit /b -1073741795"
    $msg = Throws { Invoke-EngineCheck -Checker $illegal -Dll 'C:\x\sigil_x.dll' -V 'cpu-avx512' -DllDirs @() -NoCreateReason $null }
    Check ($msg -and $msg -match 'illegal instruction \(0xC000001D\)' -and $msg -match '-NoCreate' -and $msg -match 'starting') "illegal instruction is explained, with stderr ($msg)"
    Check ((Get-Content -Raw $argLog) -match '--variant cpu-avx512') '--variant is passed'
    $garbage = & $fake 'garbage.cmd' "echo not json`r`nexit /b 3"
    $msg = Throws { Invoke-EngineCheck -Checker $garbage -Dll 'C:\x\sigil_x.dll' -V 'cpu' -DllDirs @() -NoCreateReason $null }
    Check ($msg -and $msg -match 'code 3' -and $msg -notmatch 'Conversion from JSON') "non-JSON output is not a parse error ($msg)"
    $silent = & $fake 'silent.cmd' 'exit /b -1073741819'
    $msg = Throws { Invoke-EngineCheck -Checker $silent -Dll 'C:\x\sigil_x.dll' -V 'cpu' -DllDirs @() -NoCreateReason $null }
    Check ($msg -and $msg -match 'access violation') "a silent crash is explained ($msg)"
    $good = & $fake 'good.cmd' 'echo {"ok":true,"create":{"skipped":"--no-create"},"imports":["kernel32.dll"]}'
    $check = Invoke-EngineCheck -Checker $good -Dll 'C:\x\sigil_x.dll' -V 'cuda13' -DllDirs @('C:\cuda\bin\x64', 'C:\cuda\bin') -NoCreateReason 'no driver'
    $argsSeen = Get-Content -Raw $argLog
    Check ($check.ok -eq $true) 'a JSON result is returned'
    Check ($argsSeen -match '--dll-dir C:\\cuda\\bin\\x64 --dll-dir C:\\cuda\\bin') "--dll-dir for every folder ($argsSeen)"
    Check ($argsSeen -match '--no-create') '--no-create when there is a reason'
    Check ((Get-JsonProperty (Get-JsonProperty $check 'create') 'skipped') -eq '--no-create') 'Get-JsonProperty reads nested values'
    Check ($null -eq (Get-JsonProperty $check 'missing')) 'Get-JsonProperty: a missing property is $null under StrictMode'

    Write-Host 'Import patterns'
    foreach ($d in 'ucrtbased.dll', 'vcruntime140d.dll', 'vcruntime140_1d.dll', 'msvcp140d.dll', 'msvcp140_1d.dll', 'msvcp140d_atomic_wait.dll', 'concrt140d.dll') {
        Check ($d -match $DebugCrtPattern) "debug CRT: $d"
    }
    foreach ($d in 'ucrtbase.dll', 'vcruntime140.dll', 'vcruntime140_1.dll', 'msvcp140.dll', 'msvcp140_1.dll', 'msvcp140_atomic_wait.dll',
        'kernel32.dll', 'cudart64_13.dll', 'cublas64_13.dll', 'cublaslt64_13.dll', 'vulkan-1.dll', 'api-ms-win-crt-runtime-l1-1-0.dll') {
        Check ($d -notmatch $DebugCrtPattern -and $d -notmatch $OpenMpPattern) "allowed: $d"
    }
    foreach ($d in 'vcomp140.dll', 'libomp140.x86_64.dll', 'libiomp5md.dll') { Check ($d -match $OpenMpPattern) "OpenMP: $d" }

    Write-Host 'Assert-VariantPrerequisites'
    $p = Assert-VariantPrerequisites 'cpu'
    Check ($null -eq $p.NoCreate -and $p.DllDirs.Count -eq 0) 'cpu: create runs, no DLL folders'
    $NoCreate = $true
    Check ((Assert-VariantPrerequisites 'cpu-avx512').NoCreate -eq 'requested with -NoCreate') '-NoCreate reaches the checker'
    $NoCreate = $false
} finally {
    Remove-Item -LiteralPath $tmp -Recurse -Force -ErrorAction SilentlyContinue
}

if ($failures.Count) {
    Write-Host "$($failures.Count) check(s) failed" -ForegroundColor Red
    exit 1
}
Write-Host 'All checks passed' -ForegroundColor Green
