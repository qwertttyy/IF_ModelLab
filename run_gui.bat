@echo off
setlocal EnableExtensions

set "LAB_ROOT=%~dp0"
for %%I in ("%LAB_ROOT%.") do set "LAB_ROOT=%%~fI"
set "VENV_DIR=%LAB_ROOT%\.venv"
set "ENV_NAME=IRONFLOW_VISION"

title IronFlow Vision Experiment Platform

if not exist "%LAB_ROOT%\src\ironflow_exp" (
    echo [IronFlow] ERROR: src\ironflow_exp was not found.
    echo [IronFlow] This file must be inside the vision_experiment_platform folder.
    pause
    exit /b 1
)

set "PYTHON_EXE=python"
set "ENV_LABEL=system python"

if exist "%VENV_DIR%\Scripts\activate.bat" (
    call "%VENV_DIR%\Scripts\activate.bat"
    set "ENV_LABEL=project .venv"
) else (
    call :FIND_CONDA
    if defined CONDA_BAT if not exist "%CONDA_BAT%" set "CONDA_BAT="
    if defined CONDA_BAT (
        call "%CONDA_BAT%" env list | findstr /R /C:"^%ENV_NAME%[ ]" >nul 2>nul
        if not errorlevel 1 (
            call "%CONDA_BAT%" activate "%ENV_NAME%"
            set "ENV_LABEL=conda %ENV_NAME%"
        )
    )
)

cd /d "%LAB_ROOT%"
if errorlevel 1 goto :FAIL

set "PYTHONPATH=%LAB_ROOT%\src;%PYTHONPATH%"

python -c "import ironflow_exp.engine.ui.tk_app" >nul 2>nul
if errorlevel 1 (
    echo [IronFlow] ERROR: GUI module import failed.
    echo [IronFlow] Python environment: %ENV_LABEL%
    python --version
    echo.
    echo [IronFlow] Run setup first:
    echo [IronFlow] %LAB_ROOT%\setup.bat
    pause
    exit /b 1
)

if /i "%~1"=="--check" (
    echo [IronFlow] GUI environment check passed.
    echo [IronFlow] Python environment: %ENV_LABEL%
    echo [IronFlow] Lab root: %LAB_ROOT%
    exit /b 0
)

echo [IronFlow] Starting Vision Experiment Platform GUI...
echo [IronFlow] Python environment: %ENV_LABEL%
echo [IronFlow] Lab root: %LAB_ROOT%
python -m ironflow_exp.engine.ui.tk_app
if errorlevel 1 goto :FAIL

exit /b 0

:FIND_CONDA
set "CONDA_BAT="
for %%P in (
    "%USERPROFILE%\miniconda3\condabin\conda.bat"
    "%USERPROFILE%\anaconda3\condabin\conda.bat"
    "%USERPROFILE%\miniforge3\condabin\conda.bat"
    "%LOCALAPPDATA%\miniconda3\condabin\conda.bat"
    "%LOCALAPPDATA%\anaconda3\condabin\conda.bat"
    "%LOCALAPPDATA%\miniforge3\condabin\conda.bat"
    "%ProgramData%\miniconda3\condabin\conda.bat"
    "%ProgramData%\anaconda3\condabin\conda.bat"
    "%ProgramData%\miniforge3\condabin\conda.bat"
) do (
    if exist "%%~fP" (
        set "CONDA_BAT=%%~fP"
        goto :EOF
    )
)
for /f "delims=" %%C in ('where conda.bat 2^>nul') do (
    set "CONDA_BAT=%%C"
    goto :EOF
)
for /f "delims=" %%C in ('where conda 2^>nul') do (
    set "CONDA_EXE=%%C"
    goto :FOUND_CONDA_EXE
)
goto :EOF

:FOUND_CONDA_EXE
for %%I in ("%CONDA_EXE%") do set "CONDA_EXE_DIR=%%~dpI"
if exist "%CONDA_EXE_DIR%..\condabin\conda.bat" (
    for %%I in ("%CONDA_EXE_DIR%..\condabin\conda.bat") do set "CONDA_BAT=%%~fI"
    goto :EOF
)
if exist "%CONDA_EXE_DIR%..\..\condabin\conda.bat" (
    for %%I in ("%CONDA_EXE_DIR%..\..\condabin\conda.bat") do set "CONDA_BAT=%%~fI"
    goto :EOF
)
goto :EOF

:FAIL
echo.
echo [IronFlow] GUI exited with an error.
pause
exit /b 1
