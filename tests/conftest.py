"""Pytest setup shared by every test module.

Puts src/ (for `import FFT`) and src/FFT_Visualize/ (for `import ui`, and
for ui.py's/FFT.py's own sibling `from dsp import ...`-style imports) on
sys.path so those work without installing the project as a package, and
forces the Qt offscreen platform *before* FFT.py (and therefore PyQt6) gets
imported anywhere -- this suite runs the same way in a CI runner with no
display as it does on a dev machine with one.
"""
import os
import sys
from pathlib import Path

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

SRC_DIR = Path(__file__).resolve().parent.parent / "src"
APP_DIR = SRC_DIR / "FFT_Visualize"
for _dir in (SRC_DIR, APP_DIR):
    if str(_dir) not in sys.path:
        sys.path.insert(0, str(_dir))
