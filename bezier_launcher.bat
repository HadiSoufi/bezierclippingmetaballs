@echo off
setlocal

py -3.10 -c "import sys" >nul 2>&1
if errorlevel 1 (
    echo Python 3.10 not found.
    echo Install it from https://www.python.org/downloads/release/python-31011/ and try again.
    exit /b 1
)

py -3.10 -m pip install -r requirements.txt
if errorlevel 1 (
    echo Failed to install dependencies.
    exit /b 1
)

py -3.10 BezierMetaballsLab.py