@echo off
setlocal EnableDelayedExpansion
title Local 4K Anime Dubbing Pipeline (Portable)

set "ROOT_DIR=%~dp0"
set "VENV_PYTHON=%ROOT_DIR%animegan2-pytorch\venv_win\Scripts\python.exe"
set "PATH=%ROOT_DIR%tools;%PATH%"

set COQUI_TOS_AGREED=1

echo ===================================================
echo   Local Offline 4K Anime Styling & Voice Dubbing
echo ===================================================
echo.

if not exist "%ROOT_DIR%input_videos" mkdir "%ROOT_DIR%input_videos"
if not exist "%ROOT_DIR%pipeline_output" mkdir "%ROOT_DIR%pipeline_output"

echo [*] Launching via wrapper script...
"%VENV_PYTHON%" -u "%ROOT_DIR%run_pipeline.py"

echo.
echo ===================================================
echo [+] Pipeline completed successfully!
echo [+] Check the 'pipeline_output' folder.
echo ===================================================
pause
