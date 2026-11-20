@echo off
setlocal

set "SCRIPT_DIR=%~dp0"
cd /d "%SCRIPT_DIR%"

set "PATH=%SCRIPT_DIR%bin\ffmpeg;%SCRIPT_DIR%.venv\Scripts;%PATH%"
if not exist "%SCRIPT_DIR%scratch\tmp" mkdir "%SCRIPT_DIR%scratch\tmp"
set "TEMP=%SCRIPT_DIR%scratch\tmp"
set "TMP=%SCRIPT_DIR%scratch\tmp"
set "TMPDIR=%SCRIPT_DIR%scratch\tmp"
set "HF_HOME=%SCRIPT_DIR%.cache\huggingface"
set "HUGGINGFACE_HUB_CACHE=%SCRIPT_DIR%.cache\huggingface\hub"
set "TORCH_HOME=%SCRIPT_DIR%.cache\torch"
set "XDG_CACHE_HOME=%SCRIPT_DIR%.cache"
set "PIP_CACHE_DIR=%SCRIPT_DIR%.cache\pip"
set "CUDA_CACHE_PATH=%SCRIPT_DIR%.cache\cuda_compute_cache"
set "TRITON_CACHE_DIR=%SCRIPT_DIR%.cache\triton_cache"
set "TORCHINDUCTOR_CACHE_DIR=%SCRIPT_DIR%.cache\inductor_cache"

:: Prefer local .venv or venv
if exist "%SCRIPT_DIR%.venv\Scripts\python.exe" (
    set "PYTHON_EXEC=%SCRIPT_DIR%.venv\Scripts\python.exe"
) else if exist "%SCRIPT_DIR%venv\Scripts\python.exe" (
    set "PYTHON_EXEC=%SCRIPT_DIR%venv\Scripts\python.exe"
) else (
    set "PYTHON_EXEC=python"
)

"%PYTHON_EXEC%" web_gui.py %*
exit /b %ERRORLEVEL%
