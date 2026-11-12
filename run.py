"""Create a machine-local environment, install dependencies, and open the GUI."""
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parent


def main():
    if sys.version_info < (3, 11):
        raise RuntimeError("Please install Python 3.11 or newer (64-bit). Python 3.12 is recommended.")
    import tkinter  # Check system Tk before downloading dependencies.

    environment = ROOT / ".venv-run"
    python = environment / ("Scripts/python.exe" if sys.platform == "win32" else "bin/python")
    if not python.is_file():
        subprocess.run([sys.executable, "-m", "venv", str(environment)], check=True)
    requirements = ROOT / "requirements.txt"
    marker = environment / ".installed-requirements"
    content = requirements.read_text(encoding="utf-8")
    if not marker.is_file() or marker.read_text(encoding="utf-8") != content:
        subprocess.run([str(python), "-m", "pip", "install", "-r", str(requirements)], check=True)
        marker.write_text(content, encoding="utf-8")
    return subprocess.call([str(python), str(ROOT / "app_gui.py")], cwd=ROOT)


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (OSError, RuntimeError, subprocess.CalledProcessError, ImportError) as exc:
        print(f"Cannot start: {exc}\nSee README.md for installation and troubleshooting.", file=sys.stderr)
        raise SystemExit(1)
