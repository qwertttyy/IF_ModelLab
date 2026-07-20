@echo off
setlocal EnableExtensions

set "LAB_ROOT=%~dp0"
for %%I in ("%LAB_ROOT%.") do set "LAB_ROOT=%%~fI"
set "VENV_DIR=%LAB_ROOT%\.venv"
set "ENV_NAME=IRONFLOW_VISION"
set "PYTHON_VERSION=3.12"
set "PYTHON_INSTALLER_VERSION=3.12.10"
set "PYTHON_MAJOR=3"
set "PYTHON_MINOR=12"
set "PYPROJECT_FILE=%LAB_ROOT%\pyproject.toml"
set "PYPROJECT_HASH_FILE=%LAB_ROOT%\.vision-pyproject.sha256"
set "PYTHON_VERSION_FILE=%LAB_ROOT%\.python-version"
set "EXTRAS=test,classification,timm,yolo,augmentation"

title IronFlow Vision Setup

echo [IronFlow] Vision lab root: %LAB_ROOT%
echo.

if not exist "%LAB_ROOT%\src\ironflow_exp" (
    echo [IronFlow] ERROR: src\ironflow_exp was not found.
    echo [IronFlow] This file must be inside the vision_experiment_platform folder.
    pause
    exit /b 1
)

if not exist "%PYPROJECT_FILE%" (
    echo [IronFlow] ERROR: pyproject.toml was not found.
    echo [IronFlow] Expected: %PYPROJECT_FILE%
    pause
    exit /b 1
)

call :FIND_CONDA
if defined CONDA_BAT if not exist "%CONDA_BAT%" set "CONDA_BAT="
if defined CONDA_BAT (
    call :SETUP_CONDA_ENV
) else (
    call :SETUP_VENV
)
if errorlevel 1 goto :FAIL

python -c "import sys; raise SystemExit(0 if sys.version_info[:2] == (%PYTHON_MAJOR%, %PYTHON_MINOR%) else 1)" >nul 2>nul
if errorlevel 1 (
    echo [IronFlow] ERROR: Active Python is not %PYTHON_VERSION%.
    python --version
    echo [IronFlow] Recreate the environment with Python %PYTHON_VERSION%, then run setup again.
    pause
    exit /b 1
)

for /f "usebackq delims=" %%H in (`powershell -NoProfile -ExecutionPolicy Bypass -Command "(Get-FileHash -Algorithm SHA256 -LiteralPath '%PYPROJECT_FILE%').Hash"`) do set "PYPROJECT_HASH=%%H"
set "OLD_PYPROJECT_HASH="
if exist "%PYPROJECT_HASH_FILE%" (
    set /p OLD_PYPROJECT_HASH=<"%PYPROJECT_HASH_FILE%"
)

set "NEED_INSTALL=1"
if /i "%PYPROJECT_HASH%"=="%OLD_PYPROJECT_HASH%" set "NEED_INSTALL=0"
if /i "%~1"=="--force" set "NEED_INSTALL=1"
if /i "%~1"=="--reinstall" set "NEED_INSTALL=1"
python -c "import ironflow_exp, yaml, pandas, polars, PIL, albumentations, torchvision" >nul 2>nul
if errorlevel 1 set "NEED_INSTALL=1"

if "%NEED_INSTALL%"=="0" (
    echo [IronFlow] Python package definition unchanged. Skipping dependency install.
) else (
    echo [IronFlow] Installing IronFlow vision dependencies...
    python -m pip install --upgrade pip
    if errorlevel 1 goto :FAIL
    python -m pip install -e "%LAB_ROOT%[%EXTRAS%]"
    if errorlevel 1 goto :FAIL
    python -m pip install gdown
    if errorlevel 1 goto :FAIL
    >"%PYPROJECT_HASH_FILE%" echo %PYPROJECT_HASH%
    >"%PYTHON_VERSION_FILE%" echo %PYTHON_VERSION%
)

echo [IronFlow] Verifying imports...
python -c "import ironflow_exp, yaml, pandas, polars, PIL, albumentations, torchvision; print('IronFlow vision import OK')"
if errorlevel 1 goto :FAIL

echo.
echo [IronFlow] Setup complete.
echo [IronFlow] Check GUI: %LAB_ROOT%\run_gui.bat --check
echo [IronFlow] Start GUI: %LAB_ROOT%\run_gui.bat
echo.
pause
exit /b 0

:SETUP_CONDA_ENV
echo [IronFlow] Conda: %CONDA_BAT%
call "%CONDA_BAT%" env list | findstr /R /C:"^%ENV_NAME%[ ]" >nul 2>nul
if errorlevel 1 (
    echo [IronFlow] Creating conda env: %ENV_NAME% Python %PYTHON_VERSION%
    call "%CONDA_BAT%" create -y -n "%ENV_NAME%" python=%PYTHON_VERSION%
    if errorlevel 1 exit /b 1
) else (
    echo [IronFlow] Conda env already exists: %ENV_NAME%
)
call "%CONDA_BAT%" activate "%ENV_NAME%"
exit /b %errorlevel%

:SETUP_VENV
echo [IronFlow] Conda was not found. Using lab-local venv:
echo [IronFlow] %VENV_DIR%
call :FIND_PYTHON312
if not defined PYTHON312_EXE (
    call :INSTALL_PYTHON312
    if errorlevel 1 exit /b 1
    call :FIND_PYTHON312
)
if not defined PYTHON312_EXE (
    echo [IronFlow] ERROR: Python %PYTHON_VERSION% was not found after install.
    exit /b 1
)
if not exist "%VENV_DIR%\Scripts\python.exe" (
    echo [IronFlow] Creating .venv with Python %PYTHON_VERSION%...
    call "%PYTHON312_EXE%" %PYTHON312_ARGS% -m venv "%VENV_DIR%"
    if errorlevel 1 exit /b 1
)
call "%VENV_DIR%\Scripts\activate.bat"
exit /b %errorlevel%

:FIND_PYTHON312
set "PYTHON312_EXE="
set "PYTHON312_ARGS="
py -%PYTHON_VERSION% -c "import sys; raise SystemExit(0 if sys.version_info[:2] == (%PYTHON_MAJOR%, %PYTHON_MINOR%) else 1)" >nul 2>nul
if not errorlevel 1 (
    set "PYTHON312_EXE=py"
    set "PYTHON312_ARGS=-%PYTHON_VERSION%"
    goto :EOF
)
python -c "import sys; raise SystemExit(0 if sys.version_info[:2] == (%PYTHON_MAJOR%, %PYTHON_MINOR%) else 1)" >nul 2>nul
if not errorlevel 1 (
    set "PYTHON312_EXE=python"
    goto :EOF
)
python3 -c "import sys; raise SystemExit(0 if sys.version_info[:2] == (%PYTHON_MAJOR%, %PYTHON_MINOR%) else 1)" >nul 2>nul
if not errorlevel 1 (
    set "PYTHON312_EXE=python3"
    goto :EOF
)
for %%P in (
    "%LOCALAPPDATA%\Programs\Python\Python312\python.exe"
    "%ProgramFiles%\Python312\python.exe"
    "%ProgramFiles(x86)%\Python312\python.exe"
) do (
    if exist "%%~fP" (
        "%%~fP" -c "import sys; raise SystemExit(0 if sys.version_info[:2] == (%PYTHON_MAJOR%, %PYTHON_MINOR%) else 1)" >nul 2>nul
        if not errorlevel 1 (
            set "PYTHON312_EXE=%%~fP"
            goto :EOF
        )
    )
)
goto :EOF

:INSTALL_PYTHON312
echo [IronFlow] Python %PYTHON_VERSION% was not found. Installing Python %PYTHON_VERSION%...
where winget >nul 2>nul
if not errorlevel 1 (
    echo [IronFlow] Trying winget package: Python.Python.3.12
    winget install --id Python.Python.3.12 -e --silent --accept-package-agreements --accept-source-agreements
    call :FIND_PYTHON312
    if defined PYTHON312_EXE exit /b 0
    echo [IronFlow] winget did not expose Python %PYTHON_VERSION% yet. Trying direct installer...
)
set "PYTHON_INSTALLER=%TEMP%\python-%PYTHON_INSTALLER_VERSION%-amd64.exe"
powershell -NoProfile -ExecutionPolicy Bypass -Command "$ErrorActionPreference='Stop'; Invoke-WebRequest -Uri 'https://www.python.org/ftp/python/%PYTHON_INSTALLER_VERSION%/python-%PYTHON_INSTALLER_VERSION%-amd64.exe' -OutFile '%PYTHON_INSTALLER%'"
if errorlevel 1 exit /b 1
powershell -NoProfile -ExecutionPolicy Bypass -Command "$p = Start-Process -FilePath '%PYTHON_INSTALLER%' -ArgumentList '/quiet InstallAllUsers=0 PrependPath=1 Include_launcher=1 Include_pip=1' -Wait -PassThru; exit $p.ExitCode"
if errorlevel 1 exit /b 1
call :FIND_PYTHON312
if defined PYTHON312_EXE exit /b 0
exit /b 1

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
echo [IronFlow] Setup failed. Read the message above, then run setup.bat again.
pause
exit /b 1