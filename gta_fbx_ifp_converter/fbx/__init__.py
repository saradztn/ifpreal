"""FBX front-end: a real binary + ASCII reader and the source-rig model."""

from __future__ import annotations

from .binary_reader import FbxParseError, read_fbx, sniff_format
from .document import FbxDocument, FbxNode, FbxProperty
from .source_rig import (
    BoneKind,
    Clip,
    KeyframeChannel,
    SourceBone,
    SourceRig,
    load_source_rig,
)

__all__ = [
    "BoneKind",
    "Clip",
    "FbxDocument",
    "FbxNode",
    "FbxProperty",
    "FbxParseError",
    "KeyframeChannel",
    "SourceBone",
    "SourceRig",
    "load_source_rig",
    "read_fbx",
    "sniff_format",
]
