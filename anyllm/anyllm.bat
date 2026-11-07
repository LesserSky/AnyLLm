@echo off
setlocal EnableExtensions
pushd "%~dp0" >nul

set "noargs="
if "%~1"=="" set "noargs=1"

rem Use a real Python, not a broken Microsoft Store alias.
python -c "import sys" >nul 2>nul
if not errorlevel 1 goto run_python
py -3 -c "import sys" >nul 2>nul
if not errorlevel 1 goto run_py

echo.
echo AnyLLM needs Python 3.10 or newer.
echo Install it from https://www.python.org/downloads/ and run this again.
set "code=127"
goto finish

:run_python
python -m anyllm %*
set "code=%errorlevel%"
goto finish

:run_py
py -3 -m anyllm %*
set "code=%errorlevel%"
goto finish

:finish
popd >nul
if not "%code%"=="0" if defined noargs (
    echo.
    echo AnyLLM exited with code %code%.
    pause
)
exit /b %code%
