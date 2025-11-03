Param()

$ErrorActionPreference = "Stop"

Write-Host "[setup] Creating venv .venv311 (Python 3.11 assumed installed)" -ForegroundColor Cyan
python -V
py -3.11 -m venv .venv311

Write-Host "[setup] Activating venv" -ForegroundColor Cyan
if (Test-Path .\.venv311\Scripts\Activate.ps1) {
    . .\.venv311\Scripts\Activate.ps1
} elseif (Test-Path .\.venv311\Scripts\activate) {
    . .\.venv311\Scripts\activate
}

python -m pip install --upgrade pip
pip install -r requirements.txt

Write-Host "[setup] Checking FFmpeg in PATH..." -ForegroundColor Cyan
$ffmpeg = (Get-Command ffmpeg -ErrorAction SilentlyContinue)
if (-not $ffmpeg) {
    Write-Warning "FFmpeg not found in PATH.\nDownload a static build from https://www.gyan.dev/ffmpeg/builds/ (Essentials).\nExtract and add the 'bin' directory (with ffmpeg.exe) to your PATH, then restart the terminal."
} else {
    Write-Host "[setup] FFmpeg found: $($ffmpeg.Path)" -ForegroundColor Green
}

Write-Host "[setup] Launching Streamlit UI" -ForegroundColor Cyan
streamlit run app/ui_streamlit.py

