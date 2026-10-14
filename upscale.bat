@echo off
setlocal

set "SCRIPT_DIR=%~dp0"
cd /d "%SCRIPT_DIR%"

:: Ưu tiên môi trường ảo .venv hoặc venv
if exist "%SCRIPT_DIR%.venv\Scripts\python.exe" (
    set "PYTHON_EXEC=%SCRIPT_DIR%.venv\Scripts\python.exe"
) else if exist "%SCRIPT_DIR%venv\Scripts\python.exe" (
    set "PYTHON_EXEC=%SCRIPT_DIR%venv\Scripts\python.exe"
) else (
    set "PYTHON_EXEC=python"
)

if "%~1"=="" (
    echo ❌ Vui lòng cung cấp đường dẫn video!
    echo 💡 Cách dùng: upscale.bat ^<đường_dẫn_video.mkv^> [tùy_chọn]
    echo.
    "%PYTHON_EXEC%" upscale.py --help
    exit /b 1
)

"%PYTHON_EXEC%" upscale.py %*
exit /b %ERRORLEVEL%
