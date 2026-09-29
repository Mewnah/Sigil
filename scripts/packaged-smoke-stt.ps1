# Packaged STT smoke helper — prints checklist and optionally launches release Sigil.exe
$ErrorActionPreference = "Stop"
$repoRoot = Split-Path -Parent $PSScriptRoot
$exe = Join-Path $repoRoot "src-tauri\target\release\Sigil.exe"

Write-Host ""
Write-Host "=== Packaged STT smoke (Whisper + Vosk) ===" -ForegroundColor Cyan
Write-Host "Use the RELEASE build exe, not pnpm tauri dev."
Write-Host ""
Write-Host "1. Mic: STT inspector -> Rust input device (not Settings browser UUID)"
Write-Host "2. Whisper: Start -> download/load interim -> Connected -> speak -> segment finals"
Write-Host "3. Vosk: Start -> zip ensure -> WASM load -> Connected -> speak -> partials/finals"
Write-Host "4. Stop during model load/download, then restart immediately (no ghost status)"
Write-Host "5. DevTools: no CSP worker-src / asset: errors"
Write-Host ""
Write-Host "Voice changer ONNX (same build): CPU baseline; optional CUDA/DirectML on Windows."
Write-Host "See docs/RELEASE_AND_ENGINEERING.md#packaged-stt-smoke-whisper--vosk"
Write-Host ""

if (Test-Path $exe) {
    Write-Host "Launching: $exe" -ForegroundColor Green
    Start-Process -FilePath $exe
} else {
    Write-Host "Release exe not found. Build first:" -ForegroundColor Yellow
    Write-Host "  pnpm tauri build"
    Write-Host "  Expected: $exe"
}
