@echo off
rem Garvis Desktop launcher: checks Python 3.14, creates .venv, installs the
rem dependencies, prepares .env and starts Garvis. No admin rights needed; nothing
rem outside this folder is changed. Usage:
rem   run.bat               set up (first time / after an update) and start Garvis
rem   run.bat --setup-only  set up, don't start
setlocal EnableExtensions
cd /d "%~dp0"
set "VENV_PY=.venv\Scripts\python.exe"
set "STAMP_DIR=.venv\garvis-installed"
set "PYCHECK=import sys, struct; sys.exit(0 if sys.version_info[:2] == (3, 14) and struct.calcsize('P') == 8 else 1)"

echo.
echo === Garvis Desktop ===
echo.

rem ---------------------------------------------------------------- 1. Python
if exist "%VENV_PY%" goto :check_venv
set "PY="
py -3.14 -c "%PYCHECK%" >nul 2>&1 && set "PY=py -3.14"
if not defined PY python -c "%PYCHECK%" >nul 2>&1 && set "PY=python"
if not defined PY goto :no_python
echo [1/5] Found Python 3.14 (64-bit).

rem ---------------------------------------------------------------- 2. .venv
echo [2/5] Creating the local virtual environment in .venv ...
%PY% -m venv .venv
if errorlevel 1 goto :venv_failed
goto :deps

:check_venv
"%VENV_PY%" -c "%PYCHECK%" >nul 2>&1
if errorlevel 1 goto :venv_wrong_python
echo [1/5] Using the existing .venv (Python 3.14).
echo [2/5] Virtual environment already exists.

rem ---------------------------------------------------------------- 3. dependencies
:deps
if not exist "%STAMP_DIR%\requirements.txt" goto :install
fc /b requirements.txt "%STAMP_DIR%\requirements.txt" >nul 2>&1 || goto :install
fc /b constraints.txt "%STAMP_DIR%\constraints.txt" >nul 2>&1 || goto :install
echo [3/5] Dependencies are up to date.
goto :config

:install
rem PySide6 puts files up to ~175 characters deep inside .venv; without Windows long
rem paths, the folder itself must stay within ~80 characters.
set "HERE=%CD%"
if "%HERE:~80,1%"=="" goto :path_ok
reg query "HKLM\SYSTEM\CurrentControlSet\Control\FileSystem" /v LongPathsEnabled 2>nul | find "0x1" >nul && goto :path_ok
goto :path_too_long
:path_ok
echo [3/5] Installing dependencies - the first time this downloads about 1 GB and takes several minutes ...
"%VENV_PY%" -m pip install --disable-pip-version-check -r requirements.txt -c constraints.txt
if errorlevel 1 goto :deps_failed
if not exist "%STAMP_DIR%" mkdir "%STAMP_DIR%"
copy /y requirements.txt "%STAMP_DIR%\requirements.txt" >nul
copy /y constraints.txt "%STAMP_DIR%\constraints.txt" >nul

rem ---------------------------------------------------------------- 4. configuration
:config
if exist ".env" goto :check_key
copy ".env.example" ".env" >nul
if errorlevel 1 goto :env_failed
echo [4/5] Created .env from .env.example - your own settings go there; it is never committed.
goto :check_key_done_msg
:check_key
echo [4/5] Using your existing .env - it is never overwritten.
:check_key_done_msg
if defined ANTHROPIC_API_KEY goto :voice
findstr /r /c:"^ANTHROPIC_API_KEY=..*" ".env" >nul 2>&1 && goto :voice
echo.
echo       No Anthropic API key is configured yet. Garvis starts without one, but every
echo       request will fail until you add your own key.
set "ANSWER="
set /p "ANSWER=      Enter your API key now? It is typed hidden and saved only in .env [Y/N]: "
if /i not "%ANSWER%"=="Y" goto :key_later
"%VENV_PY%" scripts\check_anthropic.py --set-key
goto :voice
:key_later
echo       Later: run  .venv\Scripts\python.exe scripts\check_anthropic.py --set-key
echo       or put ANTHROPIC_API_KEY=your-key into .env.

rem ---------------------------------------------------------------- 5. voice model
:voice
if exist "data\voices\en_US-lessac-medium.onnx" goto :ready
echo [5/5] Downloading the default Piper voice for spoken replies - about 60 MB ...
"%VENV_PY%" -m piper.download_voices --download-dir data\voices en_US-lessac-medium
if errorlevel 1 echo       WARNING: the voice download failed. Garvis still works; spoken replies need the voice - see README.
goto :ready_done
:ready
echo [5/5] Piper voice found.
:ready_done

if /i "%~1"=="--setup-only" goto :setup_done

rem ---------------------------------------------------------------- start
echo.
echo Starting Garvis Desktop ... keep this window open while Garvis runs.
"%VENV_PY%" run_garvis.py
if errorlevel 1 goto :app_failed
exit /b 0

:setup_done
echo.
echo Setup complete. Start Garvis with run.bat
exit /b 0

rem ---------------------------------------------------------------- errors
:no_python
echo ERROR: Python 3.14 (64-bit) was not found.
echo        Install it from https://www.python.org/downloads/ - "Install for current user" is enough,
echo        no admin rights needed. Then run run.bat again.
goto :fail
:venv_failed
echo ERROR: Could not create the virtual environment in .venv.
echo        Check that this folder is writable, then run run.bat again.
goto :fail
:venv_wrong_python
echo ERROR: The existing .venv was not made with Python 3.14 (64-bit).
echo        Delete the .venv folder and run run.bat again.
goto :fail
:deps_failed
echo ERROR: Installing the dependencies failed - see the messages above.
echo        Check your internet connection and free disk space, then run run.bat again.
goto :fail
:path_too_long
echo ERROR: This folder's path is too long for Windows to hold Garvis's Python packages:
echo        "%HERE%"
echo        Move the folder somewhere shorter, for example C:\garvis or your user folder,
echo        delete .venv if it exists, and run run.bat again.
goto :fail
:env_failed
echo ERROR: Could not create .env from .env.example.
goto :fail
:app_failed
echo.
echo Garvis exited with an error - see the messages above and data\logs\garvis.log.
goto :fail
:fail
echo.
pause
exit /b 1
