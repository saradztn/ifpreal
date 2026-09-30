"""Read a GTA ped DFF and expose its HAnim skeleton as the export target.

The DFF is the authority for this project (see README "Target DFF is the
ultimate authority").  Everything the retarget solver needs about the target
comes from :class:`GtaSkeleton`:

* frame hierarchy and parent indices
* per-frame local bind transform (RenderWare right-handed, Z up)
* the HAnim plugin's node id for every frame -- this is the *bone id* that
  gets written into the ANP3 object header
* the skin bone index each frame corresponds to, when a skin is present

Frame names are whatever the DFF happens to use.  A canonical GTA SA name is
attached *alongside* the DFF name for the GUI, but resolution always goes
through the DFF.
"""

from __future__ import annotations

import math
import os
from dataclasses import dataclass, field
from typing import Iterable, Sequence

import numpy as np

from .bones import SaBoneTag, bone_tag_from_name, cross_check_frame, describe_tag

__all__ = [
    "GtaBone",
    "GtaSkeleton",
    "DffHanimError",
    "load_skeleton",
]


class DffHanimError(RuntimeError):
    """The DFF is not a usable animated ped target."""


def _normalize(name: str) -> str:
    return "".join(ch for ch in name.casefold() if ch.isalnum())


@dataclass
class GtaBone:
    """One HAnim frame of the target skeleton."""

    index: int
    name: str
    parent: int
    #: Local bind transform in DFF space, as a 4x4 column-vector matrix.
    bind_local: np.ndarray
    #: Local bind transform converted into the pipeline's working space
    #: (GTA ped model space: Z up, left handed).  See :mod:`..core.space`.
    bind_local_gta: np.ndarray = field(default_factory=lambda: np.eye(4))
    bind_world_gta: np.ndarray = field(default_factory=lambda: np.eye(4))
    bind_local_quat: np.ndarray = field(default_factory=lambda: np.array([0.0, 0.0, 0.0, 1.0]))
    bind_local_translation: np.ndarray = field(default_factory=lambda: np.zeros(3))
    #: HAnim node id from the DFF plugin, i.e. the IFP bone id.
    hanim_id: int | None = None
    #: Index inside the DFF skin's bone array, when the model has a skin.
    skin_bone_index: int | None = None
    hanim_flags: int = 0
    canonical_tag: SaBoneTag | None = None
    has_non_identity_bind_rotation: bool = False

    @property
    def is_root(self) -> bool:
        return self.parent < 0

    @property
    def depth(self) -> int:
        return self._depth

    _depth: int = 0

    @property
    def bone_id(self) -> int:
        """Bone id to write into the IFP, or ``-1`` when the DFF has no HAnim."""
        return int(self.hanim_id) if self.hanim_id is not None else -1

    @property
    def label(self) -> str:
        parts = [self.name]
        if self.canonical_tag is not None:
            parts.append(f"({describe_tag(self.canonical_tag)})")
        return " ".join(parts)

    @property
    def side(self) -> str:
        if self.canonical_tag is not None:
            return self.canonical_tag.side
        low = self.name.lower()
        if "left" in low or "_l" in low or low.startswith("l ") or ".l" in low:
            return "left"
        if "right" in low or "_r" in low or low.startswith("r ") or ".r" in low:
            return "right"
        return "center"


def _frame_matrix(rotation: Sequence[float], position: Sequence[float]) -> np.ndarray:
    """Build a RenderWare local matrix from the DFF's row-major 3x3 + position.

    RenderWare stores the 3x3 basis in *row-major* order inside the DFF
    (``frame.right/upt/at``), whereas the rest of this project uses the
    column-vector convention, hence the transpose.
    """
    m = np.eye(4, dtype=np.float64)
    m[:3, :3] = np.asarray(rotation, dtype=np.float64).reshape(3, 3).T
    m[:3, 3] = np.asarray(position, dtype=np.float64).reshape(3)
    return m


class GtaSkeleton:
    """Target ped skeleton resolved from a DFF."""

    def __init__(
        self,
        bones: Sequence[GtaBone],
        source_path: str = "",
        model_names: Sequence[str] = (),
        has_skin: bool = False,
    ) -> None:
        self.bones: list[GtaBone] = list(bones)
        self.source_path = source_path
        self.model_names = list(model_names)
        self.has_skin = has_skin
        self._by_name: dict[str, GtaBone] = {}
        self._by_hanim_id: dict[int, GtaBone] = {}
        for bone in self.bones:
            self._by_name.setdefault(_normalize(bone.name), bone)
            if bone.hanim_id is not None:
                self._by_hanim_id.setdefault(int(bone.hanim_id), bone)
        self._compute_world()

    # -- construction ------------------------------------------------------- #
    def _compute_world(self) -> None:
        order = self.topological_order()
        for bone in order:
            if bone.parent < 0:
                bone.bind_world_gta = bone.bind_local_gta.copy()
                bone.depth = 0
            else:
                parent = self.bones[bone.parent]
                bone.bind_world_gta = parent.bind_world_gta @ bone.bind_local_gta
                bone.depth = parent.depth + 1

    def topological_order(self) -> list[GtaBone]:
        """Bones ordered so that every parent precedes its children."""
        result: list[GtaBone] = []
        visiting: set[int] = set()
        done: set[int] = set()

        def visit(bone: GtaBone) -> None:
            if bone.index in done:
                return
            if bone.index in visiting:
                raise DffHanimError(
                    f"Cycle in the target DFF frame hierarchy at {bone.name!r}"
                )
            visiting.add(bone.index)
            if 0 <= bone.parent < len(self.bones):
                visit(self.bones[bone.parent])
            visiting.discard(bone.index)
            done.add(bone.index)
            result.append(bone)

        for bone in self.bones:
            visit(bone)
        return result

    # -- lookups ------------------------------------------------------------ #
    def by_name(self, name: str) -> GtaBone | None:
        return self._by_name.get(_normalize(name))

    def by_hanim_id(self, hanim_id: int) -> GtaBone | None:
        return self._by_hanim_id.get(int(hanim_id))

    def find(self, name: str) -> GtaBone:
        bone = self.by_name(name)
        if bone is None:
            raise KeyError(f"No frame named {name!r} in the target DFF")
        return bone

    def find_tag(self, tag: SaBoneTag | int) -> GtaBone | None:
        """Find the frame that carries an HAnim tag in *this* DFF.

        Prefers the HAnim plugin id, then the canonical name, so a custom
        ped that renames a frame still resolves.
        """
        if isinstance(tag, SaBoneTag):
            bone = self.by_hanim_id(int(tag))
            if bone is not None:
                return bone
        else:
            bone = self.by_hanim_id(int(tag))
            if bone is not None:
                return bone
        canonical = SaBoneTag(tag) if not isinstance(tag, SaBoneTag) else tag
        for candidate in (canonical.dff_frame_name, canonical.ifp_name):
            if candidate:
                bone = self.by_name(candidate)
                if bone is not None:
                    return bone
        return None

    @property
    def root(self) -> GtaBone | None:
        for bone in self.bones:
            if bone.parent < 0:
                return bone
        return self.bones[0] if self.bones else None

    def children_of(self, bone: GtaBone | int) -> list[GtaBone]:
        index = bone.index if isinstance(bone, GtaBone) else bone
        return [b for b in self.bones if b.parent == index]

    def descendants(self, bone: GtaBone | int) -> list[GtaBone]:
        start = bone.index if isinstance(bone, GtaBone) else bone
        out: list[GtaBone] = []
        frontier = [start]
        while frontier:
            current = frontier.pop()
            for child in self.children_of(current):
                out.append(child)
                frontier.append(child.index)
        return out

    def chain(self, bone: GtaBone | int) -> list[GtaBone]:
        """Root-to-bone path including ``bone`` itself."""
        index = bone.index if isinstance(bone, GtaBone) else bone
        chain: list[GtaBone] = []
        guard = 0
        while index >= 0 and guard <= len(self.bones):
            current = self.bones[index]
            chain.append(current)
            index = current.parent
            guard += 1
        chain.reverse()
        return chain

    @property
    def animated_tags(self) -> list[int]:
        """HAnim ids present in the DFF, ascending."""
        return sorted(self._by_hanim_id)

    @property
    def has_hanim(self) -> bool:
        return bool(self._by_hanim_id)

    # -- reporting ---------------------------------------------------------- #
    def cross_check(self) -> list:
        """Compare every frame's HAnim id with the canonical GTA table."""
        return [
            cross_check_frame(bone.name, bone.hanim_id)
            for bone in self.bones
            if bone.hanim_id is not None
        ]

    def summary(self) -> dict:
        return {
            "path": self.source_path,
            "frames": len(self.bones),
            "hanim_bones": len(self._by_hanim_id),
            "has_skin": self.has_skin,
            "models": list(self.model_names),
            "roots": [b.name for b in self.bones if b.parent < 0],
            "max_depth": max((b.depth for b in self.bones), default=0),
            "bones_with_bind_rotation": [
                b.name for b in self.bones if b.has_non_identity_bind_rotation
            ],
        }

    def describe(self) -> str:
        lines = [
            f"Target skeleton: {self.source_path or '<memory>'}",
            f"  frames      : {len(self.bones)}",
            f"  HAnim bones : {len(self._by_hanim_id)}",
            f"  roots       : {', '.join(b.name for b in self.bones if b.parent < 0)}",
            "  hierarchy:",
        ]
        for bone in self.topological_order():
            indent = "    " + "  " * bone.depth
            tag = describe_tag(bone.hanim_id) if bone.hanim_id is not None else "no-HAnim"
            lines.append(f"{indent}- {bone.name}  [{tag}]")
        return "\n".join(lines)


def load_skeleton(path: str, *, validate: bool = True) -> GtaSkeleton:
    """Load a target ped DFF and return its HAnim skeleton.

    Raises :class:`DffHanimError` with the message
    ``Target DFF has no valid ped HAnim skeleton`` when the file has frames
    but none of them carry an HAnim plugin.
    """
    from rwfury import Dff  # imported lazily so import errors point at the cause

    if not os.path.isfile(path):
        raise FileNotFoundError(path)

    dff = Dff.from_file(path)
    if not dff.frames:
        raise DffHanimError(f"{path} contains no frames")

    # The HAnim plugin lives on the *top* frame; rwfury attaches it there.
    root_hanim = None
    for frame in dff.frames:
        if frame.hanim is not None and frame.hanim.bones:
            root_hanim = frame.hanim
            break

    if root_hanim is None:
        raise DffHanimError(
            "Target DFF has no valid ped HAnim skeleton "
            f"({path}: {len(dff.frames)} frames, no HAnim plugin)"
        )

    skin_bone_for_hanim: dict[int, int] = {}
    for geometry in dff.geometries:
        skin = getattr(geometry, "skin", None)
        if skin is None:
            continue
        used = list(getattr(skin, "used_bone_indices", []) or [])
        for hanim_bone in root_hanim.bones:
            if hanim_bone.node_index < len(used):
                skin_bone_for_hanim.setdefault(hanim_bone.node_id, used[hanim_bone.node_index])

    from ..core.space import dff_to_gta  # local import: avoids a cycle

    hanim_by_frame: dict[int, tuple[int, int]] = {}
    for hb in root_hanim.bones:
        hanim_by_frame[hb.node_index] = (hb.node_id, hb.flags)

    bones: list[GtaBone] = []
    for index, frame in enumerate(dff.frames):
        matrix = _frame_matrix(frame.rotation_matrix, frame.position)
        gta_matrix = dff_to_gta(matrix)
        translation, rotation, scale = _safe_decompose(gta_matrix)
        hanim_id, hanim_flags = hanim_by_frame.get(index, (None, 0))
        tag = bone_tag_from_name(frame.name)
        bone = GtaBone(
            index=index,
            name=frame.name,
            parent=frame.parent,
            bind_local=matrix,
            bind_local_gta=gta_matrix,
            bind_local_quat=rotation,
            bind_local_translation=translation,
            hanim_id=hanim_id,
            hanim_flags=hanim_flags,
            skin_bone_index=skin_bone_for_hanim.get(hanim_id) if hanim_id is not None else None,
            canonical_tag=tag,
            has_non_identity_bind_rotation=bool(
                np.max(np.abs(rotation - np.array([0.0, 0.0, 0.0, 1.0]))) > 1e-6
            ),
        )
        bone._scale = scale
        bones.append(bone)

    model_names = [getattr(a, "name", "") for a in getattr(dff, "atomics", [])]
    skeleton = GtaSkeleton(
        bones,
        source_path=os.path.abspath(path),
        model_names=[n for n in model_names if n],
        has_skin=bool(skin_bone_for_hanim),
    )

    if validate and not skeleton.has_hanim:
        raise DffHanimError("Target DFF has no valid ped HAnim skeleton")
    return skeleton


def _safe_decompose(matrix: np.ndarray):
    from ..core.mathx import mat_decompose

    try:
        return mat_decompose(matrix)
    except ValueError:
        # A non-orthonormal bind basis (some third-party peds) must not abort
        # the import: keep the translation and fall back to the identity
        # rotation, which is what the engine effectively does too.
        from ..core.mathx import orthonormalize, quat_from_matrix

        translation = matrix[:3, 3].copy()
        try:
            basis = orthonormalize(matrix)
        except ValueError:
            basis = np.eye(3)
        quat = quat_from_matrix(basis)
        scales = np.linalg.norm(matrix[:3, :3], axis=0)
        return translation, quat, scales
