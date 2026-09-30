"""GTA San Andreas / MTA:SA FBX -> IFP retarget engine.

Pipeline::

    FBX  ->  source rig  ->  bone mapping  ->  rest pose + axis solve
          ->  retarget    ->  bake         ->  ANP3 writer  ->  .ifp
          ->  validator / round-trip metrics

The GTA-side DFF/HAnim/IFP handling is delegated to the MIT licensed
``rwfury`` package; this project adds the retargeting, validation and tooling
around it.  See ``README.md`` for the architecture and the known limitations.
"""

from __future__ import annotations

__version__ = "1.0.0"
__all__ = ["__version__"]
