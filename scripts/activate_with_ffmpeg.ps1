Param()

$ErrorActionPreference = "Stop"

# Resolve project root (parent of this scripts folder)
$projRoot = Split-Path $PSScriptRoot -Parent

Write-Host "[env] Activating venv at .venv311" -ForegroundColor Cyan
$venvActivate = Join-Path $projRoot ".venv311/Scripts/Activate.ps1"
if (Test-Path $venvActivate) {
    . $venvActivate
} else {
    Write-Warning "Venv not found. Create with: py -3.11 -m venv .venv311"
}

# Try to locate a local FFmpeg under tools/*essentials_build/bin
$ffmpegBin = $null
$toolsDir = Join-Path $projRoot "tools"
if (Test-Path $toolsDir) {
    $cand = Get-ChildItem $toolsDir -Directory -Filter "ffmpeg-*essentials_build" -ErrorAction SilentlyContinue | Select-Object -First 1
    if ($cand) {
        $bin = Join-Path $cand.FullName "bin"
        if (Test-Path (Join-Path $bin "ffmpeg.exe")) { $ffmpegBin = $bin }
    }
}

if ($ffmpegBin) {
    Write-Host "[env] Prepending FFmpeg to PATH: $ffmpegBin" -ForegroundColor Green
    $env:PATH = "$ffmpegBin;" + $env:PATH
} else {
    Write-Warning "Local FFmpeg not found in tools/. Falling back to system PATH."
}

Write-Host "[env] Python: $(python -V)" -ForegroundColor DarkGray
if (Get-Command ffmpeg -ErrorAction SilentlyContinue) {
    Write-Host "[env] FFmpeg: $((ffmpeg -version | Select-Object -First 1))" -ForegroundColor DarkGray
} else {
    Write-Warning "ffmpeg not available in PATH."
}

