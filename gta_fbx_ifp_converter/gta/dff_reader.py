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
from .tag_resolve import TagDiagnostic, TagResolution, resolve_tags

#: Every tag the FBX/IFP pipeline can address.  A DFF may legitimately carry
#: extras (jaw variants, lids); those stay as plain ints rather than failing.
_KNOWN_TAGS = frozenset(int(t) for t in SaBoneTag)

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
    #: Tag taken from the DFF's own HAnim plugin -- the authority.
    canonical_tag: SaBoneTag | None = None
    #: Tag guessed from the frame name, kept only for cross-checking.
    name_derived_tag: SaBoneTag | None = None
    #: True for the frame that carries the HAnim plugin (the skeleton root).
    is_hanim_root: bool = False
    has_non_identity_bind_rotation: bool = False

    @property
    def is_root(self) -> bool:
        return self.parent < 0

    @property
    def depth(self) -> int:
        return self._depth

    _depth: int = 0

    #: Set by :func:`load_skeleton` from the tag resolution: the value that
    #: actually goes into the IFP object's ``BoneID`` field.
    resolved_tag: int | None = None

    @property
    def bone_id(self) -> int:
        """Bone id to write into the IFP, or ``-1`` when the frame has none.

        This is the *resolved* tag.  The DFF's raw HAnim ``node_id`` stays on
        :attr:`hanim_id` for diagnostics but is not what the engine is fed.
        """
        if self.resolved_tag is not None:
            return int(self.resolved_tag)
        return int(self.hanim_id) if self.hanim_id is not None else -1

    @property
    def is_addressable(self) -> bool:
        """True when this frame can carry an IFP track."""
        return self.bone_id >= 0

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
    """Build a RenderWare local matrix from the DFF's 3x3 basis + position.

    RenderWare serialises ``RwMatrix`` as five ``RwV3d`` values -- ``right``,
    ``up``, ``at``, ``pos``, ``pad`` -- and ``RwMatrixTransformPoint`` computes
    ``p' = right*p.x + up*p.y + at*p.z + pos``.  The first three vectors are
    therefore the *rows* of the matrix that acts on column vectors, so the
    nine floats reshape straight into the 3x3 block with no transpose.

    Verified against ``testdata/male01.dff``: read as rows, the bind pose
    puts the head at z = +0.684 and the toes at z = -1.032 (a 1.72 m ped,
    Z up, +Y to the character's left); transposing instead mirrors the
    skeleton and makes the head the lowest point.
    """
    m = np.eye(4, dtype=np.float64)
    m[:3, :3] = np.asarray(rotation, dtype=np.float64).reshape(3, 3)
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
        #: Filled in by :func:`load_skeleton` once the DFF's HAnim has been
        #: reconciled; see :mod:`.tag_resolve`.
        self.tag_resolution: TagResolution | None = None
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
                bone._depth = 0
            else:
                parent = self.bones[bone.parent]
                bone.bind_world_gta = parent.bind_world_gta @ bone.bind_local_gta
                bone._depth = parent.depth + 1

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

        The tag resolution from :mod:`.tag_resolve` is authoritative: it is
        the value that will be written into the IFP object's ``BoneID``.
        The raw HAnim ``node_id`` map is only a last resort, because on the
        shipped SA peds it points at the wrong frames entirely.
        """
        value = int(tag)
        if self.tag_resolution is not None:
            frame = self.tag_resolution.frame_of_tag(value)
            if frame is not None:
                return self.bones[frame]
        bone = self.by_hanim_id(value)
        if bone is not None:
            return bone
        canonical = SaBoneTag(value) if value in _KNOWN_TAGS else None
        if canonical is not None:
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

    # The HAnim plugin rides on the skeleton's root frame.  It carries the
    # authoritative tag for every animatable bone, and its ``node_index`` is
    # the index into the clump's frame list.  Both relations were verified
    # against testdata/male01.dff: skinning the mesh with
    # ``inv(world[node_index])`` reproduces a 1.9 m Z-up ped, whereas an
    # off-by-one frame mapping lays the model down along Y.
    root_hanim = None
    hanim_frame = -1
    for index, frame in enumerate(dff.frames):
        if frame.hanim is not None and frame.hanim.bones:
            root_hanim = frame.hanim
            hanim_frame = index
            break

    if root_hanim is None:
        raise DffHanimError(
            "Target DFF has no valid ped HAnim skeleton "
            f"({path}: {len(dff.frames)} frames, no HAnim plugin)"
        )

    # ``used_bone_indices[k]`` is the *HAnim bone array* index of the k-th
    # skin bone, not a frame index.  For male01.dff this resolves to the 28
    # body tags, correctly leaving Root/Belly/breasts unskinned.
    skin_position_of_hanim: dict[int, int] = {}
    for geometry in dff.geometries:
        skin = getattr(geometry, "skin", None)
        if skin is None:
            continue
        for position, used in enumerate(getattr(skin, "used_bone_indices", []) or []):
            skin_position_of_hanim.setdefault(int(used), position)

    from ..core.space import dff_to_gta  # local import: avoids a cycle

    hanim_by_frame: dict[int, tuple[int, int]] = {}
    hanim_position_by_frame: dict[int, int] = {}
    hanim_id_of_frame: dict[int, int | None] = {}
    for position, hb in enumerate(root_hanim.bones):
        hanim_by_frame[hb.node_index] = (hb.node_id, hb.flags)
        hanim_position_by_frame[hb.node_index] = position

    bones: list[GtaBone] = []
    for index, frame in enumerate(dff.frames):
        matrix = _frame_matrix(frame.rotation_matrix, frame.position)
        gta_matrix = dff_to_gta(matrix)
        translation, rotation, scale = _safe_decompose(gta_matrix)
        hanim_id, hanim_flags = hanim_by_frame.get(index, (None, 0))
        name_tag = bone_tag_from_name(frame.name)
        hanim_id_of_frame[index] = None if hanim_id is None else int(hanim_id)
        position_in_hanim = hanim_position_by_frame.get(index)
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
            skin_bone_index=(
                skin_position_of_hanim.get(position_in_hanim)
                if position_in_hanim is not None
                else None
            ),
            name_derived_tag=name_tag,
            is_hanim_root=(index == hanim_frame),
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
        has_skin=bool(skin_position_of_hanim),
    )

    # Decide which HAnim tag every frame really carries.  The DFF decides the
    # tag *set* and the hierarchy; see :mod:`.tag_resolve` for why the
    # ``node_id`` array cannot be trusted on its own.
    resolution = resolve_tags(
        [frame.name for frame in dff.frames],
        {bone.index: bone.parent for bone in bones},
        hanim_id_of_frame,
        hanim_tags_available=sorted({b.node_id for b in root_hanim.bones}),
    )
    for bone in bones:
        tag = resolution.tag_of_frame.get(bone.index)
        # ``-1`` means "this frame has no addressable bone", which is what
        # :attr:`GtaBone.bone_id` must report -- falling back to the stale
        # HAnim node id here would claim a bone the frame does not own.
        bone.resolved_tag = -1 if tag is None else int(tag)
        bone.canonical_tag = SaBoneTag(tag) if tag is not None and tag in _KNOWN_TAGS else None
    skeleton.tag_resolution = resolution

    # The skin indexes the HAnim bone array, so it inherits the same stale
    # enumeration as ``node_id`` does.  Report the damage rather than let it
    # pass silently: the IFP never needs these indices, but a user reading
    # the validation report should know the DFF disagrees with itself.
    skinned_tags = {
        bone.resolved_tag for bone in bones
        if bone.skin_bone_index is not None and bone.resolved_tag is not None
    }
    if skinned_tags and not {1, 2, 3, 4, 5, 6, 7, 8, 21, 22, 23, 24, 25, 26,
                             31, 32, 33, 34, 35, 36, 41, 42, 43, 44,
                             51, 52, 53, 54} >= skinned_tags:
        resolution.diagnostics.append(
            TagDiagnostic(
                "warning",
                "skin-index-mismatch",
                -1,
                "the DFF skin references bones that do not form the ped body "
                f"set under the resolved tags ({len(skinned_tags)} bones); the "
                "DFF's skin and HAnim arrays use a different bone order than "
                "its frame list",
            )
        )

    if validate:
        if not skeleton.has_hanim:
            raise DffHanimError("Target DFF has no valid ped HAnim skeleton")
        for problem in resolution.errors():
            raise DffHanimError(f"Target DFF has no valid ped HAnim skeleton ({problem})")
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
