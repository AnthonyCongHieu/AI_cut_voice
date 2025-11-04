Param()

$ErrorActionPreference = "Stop"

# Activate venv and ensure FFmpeg in PATH
. "$PSScriptRoot/activate_with_ffmpeg.ps1"

Write-Host "[run] Launching Streamlit UI" -ForegroundColor Cyan
streamlit run "$PSScriptRoot/../app/ui_streamlit.py"

