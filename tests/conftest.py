"""Shared test setup.

PySide6 links against libGL, libEGL, libxkbcommon and libdbus-1 whether or not
it ever calls them, and a bare container has none of them. tests/qtstubs holds
stand-ins that satisfy the loader (see its README), but LD_LIBRARY_PATH is
read by ld.so once at process start -- setting it from Python is too late.

So the test process re-executes itself once with the stubs on the path. The
re-exec happens at most once, guarded by an environment marker, and only when
PySide6 actually fails to import; on a normal desktop nothing happens at all.
"""
from __future__ import annotations

import os
import pathlib
import sys

ROOT = pathlib.Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

STUBS = ROOT / "tests" / "qtstubs"
_RELAUNCHED = "GTAFBX_QT_STUBS_ACTIVE"

if STUBS.is_dir() and not os.environ.get(_RELAUNCHED):
    try:
        import PySide6.QtWidgets  # noqa: F401
    except ImportError:
        existing = os.environ.get("LD_LIBRARY_PATH", "")
        os.environ["LD_LIBRARY_PATH"] = (
            f"{STUBS}:{existing}" if existing else str(STUBS))
        os.environ[_RELAUNCHED] = "1"
        # execv, not a subprocess: the tests share this interpreter's state
        # and a child process would collect results the parent never sees.
        os.execv(sys.executable, [sys.executable, *sys.argv])
