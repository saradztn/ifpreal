"""GTA-side I/O: the target DFF/HAnim skeleton and the IFP format.

The target DFF is the authority for this project.  Every bone id written into
an IFP is resolved through :mod:`.dff_reader` and :mod:`.tag_resolve` from
the file the user picked -- never from a hard coded table.
"""

from __future__ import annotations

from .bones import (
    SaBoneTag,
    bone_name_from_tag,
    bone_tag_from_name,
    describe_tag,
    is_ped_body_bone,
)
from .dff_reader import DffHanimError, GtaBone, GtaSkeleton, load_skeleton
from .tag_resolve import TagDiagnostic, TagResolution, resolve_tags

__all__ = [
    "DffHanimError",
    "GtaBone",
    "GtaSkeleton",
    "SaBoneTag",
    "TagDiagnostic",
    "TagResolution",
    "bone_name_from_tag",
    "bone_tag_from_name",
    "describe_tag",
    "is_ped_body_bone",
    "load_skeleton",
    "resolve_tags",
]
