#!/usr/bin/env python3
"""Start the converter's window.

Kept as a script rather than only a console entry point because that is what
a user double-clicks on Windows, and a stack trace in a console window is
useless to somebody who just wanted to convert a file.
"""
from __future__ import annotations

import os
import sys
import traceback

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))


def main() -> int:
    try:
        from gui.app import main as run
    except ImportError as exc:
        print(f"The window could not start: {exc}\n", file=sys.stderr)
        print("PySide6 is what draws it:", file=sys.stderr)
        print("    pip install PySide6-Essentials\n", file=sys.stderr)
        print("The command line converter does not need it:", file=sys.stderr)
        print("    python cli.py convert --fbx dance.fbx --dff male01.dff"
              " --out dance.ifp", file=sys.stderr)
        return 1
    return run(sys.argv)


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except SystemExit:
        raise
    except Exception:
        traceback.print_exc()
        input("\nPress Enter to close...")
        raise SystemExit(1)
