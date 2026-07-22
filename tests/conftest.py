"""Pytest setup shared by every test module.

Puts src/ on sys.path so `import FFT` works without installing the project
as a package, and forces the Qt offscreen platform *before* FFT.py (and
therefore PyQt6) gets imported anywhere -- this suite runs the same way in
a CI runner with no display as it does on a dev machine with one.
"""
import os
import sys
from pathlib import Path

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

SRC_DIR = Path(__file__).resolve().parent.parent / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))
