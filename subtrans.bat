@echo off
setlocal

set "SCRIPT_DIR=%~dp0"
cd /d "%SCRIPT_DIR%"

if exist "%SCRIPT_DIR%.venv\Scripts\python.exe" (
    set "PYTHON_EXEC=%SCRIPT_DIR%.venv\Scripts\python.exe"
) else if exist "%SCRIPT_DIR%venv\Scripts\python.exe" (
    set "PYTHON_EXEC=%SCRIPT_DIR%venv\Scripts\python.exe"
) else (
    set "PYTHON_EXEC=python"
)

if "%~1"=="" (
    echo [Subtrans AI] CLI Translation Tool
    echo Usage: subtrans.bat [check^|models^|run] [options]
    echo.
    "%PYTHON_EXEC%" subtrans.py --help
    exit /b 0
)

"%PYTHON_EXEC%" subtrans.py %*
exit /b %ERRORLEVEL%
