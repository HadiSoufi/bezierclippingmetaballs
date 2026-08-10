#!/usr/bin/env bash
# Linux/macOS launcher for the Bezier-clipped metaballs lab.
# Usage: ./bezier_launcher.sh
set -e

if ! command -v python3.10 >/dev/null 2>&1; then
    echo "Python 3.10 not found -> install it with your package manager, e.g."
    echo "  Debian/Ubuntu: sudo apt install python3.10 python3.10-tk python3.10-venv"
    echo "  macOS:         brew install python@3.10"
    echo "  then re-run this script."
    exit 1
fi

if ! python3.10 -c "import tkinter" >/dev/null 2>&1; then
    echo "tkinter is missing -> install it, e.g."
    if command -v apt-get >/dev/null 2>&1; then
        echo "  sudo apt install python3.10-tk"
    else
        echo "  brew install python-tk@3.10"
    fi
    echo "  then re-run this script."
    exit 1
fi

python3.10 -m venv .venv
source .venv/bin/activate

python -m pip install -r requirements.txt

python BezierMetaballsLab.py