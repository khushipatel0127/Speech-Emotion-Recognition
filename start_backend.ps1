# SpeechSense — Backend Launcher
Write-Host "==============================================" -ForegroundColor Cyan
Write-Host "  Starting SpeechSense FastAPI Backend Server " -ForegroundColor Cyan
Write-Host "==============================================" -ForegroundColor Cyan

$ScriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
Set-Location $ScriptDir

$VenvPython = Join-Path $ScriptDir ".venv\Scripts\python.exe"

if (-not (Test-Path $VenvPython)) {
    Write-Host "Virtual environment python not found at $VenvPython" -ForegroundColor Red
    exit 1
}

Write-Host "Running on http://localhost:8000" -ForegroundColor Green
Write-Host "API Documentation available at: http://localhost:8000/docs" -ForegroundColor Yellow
Write-Host "Press Ctrl+C to stop the backend server.`n" -ForegroundColor Gray

& $VenvPython -m uvicorn main:app --app-dir backend --host 127.0.0.1 --port 8000 --reload
