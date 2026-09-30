"""The converter's window.

`gui.app` is the window; `gui.pipeline` is the only thing that touches the
converter, and it hands back plain dicts.  Keeping the two apart is what lets
the pages be tested without a display and the converter be tested without a
window.
"""
from __future__ import annotations

__all__ = ["app", "jobs", "pipeline", "preview"]


def main(argv=None) -> int:
    """Start the window.  Imported lazily so `import gui` stays cheap."""
    from .app import main as _main

    return _main(argv)
