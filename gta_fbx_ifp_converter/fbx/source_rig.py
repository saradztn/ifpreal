"""Turn a parsed FBX document into a bind-pose rig plus animation clips.

This is where the two easy-to-get-wrong parts of FBX live:

1. **Object composition.** A frame's children are attached through
   ``ConnectionProperty("child", id)`` stored in the *parent* object.  The
   loader inverts that map, so the rig is a real tree.

2. **Local transform composition.** FBX's evaluation order is

   .. code-block:: text

       L = T . Roff . Rpiv . Rpre . R . Rpost . Rpiv^-1 . Roff^-1
               . Soff . Spiv . S . Spiv^-1 . Gt . Gr . Gs

   where the trailing ``G`` terms are the *geometric transform*.  Blender and
   assimp both fold the geometric transform into the node's local matrix, and
   so do we -- but it is optional here (``include_geometric_transform``) and
   the choice is reported, because rigs that were pre-baked with it disabled
   need the other behaviour.

3. **Curve binding.** A curve only belongs to a model through an ``OP``
   (object-property) connection whose ``Src`` is the model id and whose
   ``Property`` is e.g. ``"|Lcl Translation"``.  Nothing is inferred from
   ordering or from names.
"""

from __future__ import annotations

import bisect
import os
from dataclasses import dataclass, field
from enum import Enum
from typing import Iterable, Iterator, Sequence

import numpy as np

from ..core.mathx import (
    euler_to_quat,
    mat_decompose,
    mat_from_trs,
    mat_identity,
    quat_normalize,
)
from .binary_reader import FbxParseError, read_fbx
from .document import FbxDocument, FbxNode

__all__ = [
    "BoneKind",
    "KeyframeChannel",
    "SourceBone",
    "SourceRig",
    "Clip",
    "load_source_rig",
    "build_source_rig",
    "FBX_TIME_UNITS_PER_SECOND",
]

#: Sentinel for lazily-populated caches: ``None`` is a meaningful value there.
_UNSET = object()

#: FBX stores times in 1/46186158000 s units (``FBXSDK_TIME_ONE_SECOND``).
FBX_TIME_UNITS_PER_SECOND = 46186158000.0

#: Key under which an :class:`AnimationCurve`'s single value channel is stored.
VALUE_KEY = "__value__"

#: **FBX stores every Euler angle in degrees**, both in a node's
#: ``Properties70`` (``Lcl Rotation``, ``Lcl PreRotation``, ``Lcl
#: PostRotation``) and in the ``Lcl Rotation``/``PreRotation`` animation
#: curves.  This is a property of the format, not of a particular exporter:
#: the file's ``GlobalSettings`` does not record it, and treating the values
#: as radians silently scrambles any rig with rotations past ~2pi (a value of
#: 765 degrees reads as 13.4 rad).  The conversion is applied exactly once,
#: at the boundary, so the rest of the pipeline works in radians.
FBX_EULER_UNITS_ARE_DEGREES = True
DEGREES_TO_RADIANS = 3.141592653589793 / 180.0

#: Euler order assumed when a ``Model`` record carries no ``RotationOrder``.
#: **This is ``ZYX``, not ``XYZ``** -- it is the Autodesk FBX SDK's own default
#: and the reference loaders' fallback.  It matters: the two orders differ by
#: a transposition of the X and Z components, so reading a Mixamo clip as XYZ
#: tilts the pelvis by ~90 degrees on average and puts the head below the hips
#: for a third of the clip.  A file that *does* declare an order always wins.
DEFAULT_ROTATION_ORDER = "ZYX"

#: How close a sampled time must be to a key time to count *as* that key.
#: FBX stores key times in seconds, and float32 seconds (what most FBX tools
#: hand back) resolve to about 1e-7 s near 10 s.  Without this tolerance a
#: key time read back from such a tool interpolates from the previous key and
#: then rounds to the previous key's value.
KEY_TIME_SNAP_SECONDS = 1e-6

#: Default clip rate when the stack does not declare one.  FBX's own default.
DEFAULT_FPS = 30.0


class BoneKind(str, Enum):
    """How a source bone participates in the retarget."""

    MAP = "MAP"
    """Directly retargeted onto a GTA HAnim bone."""

    MERGE = "MERGE"
    """Blended into a GTA chain (spine, twist bones, ...)."""

    CONTROL_ONLY = "CONTROL_ONLY"
    """Never exported: IK handles, pole targets, root gizmos."""

    IGNORE = "IGNORE"
    """Not part of the body (facial controls, costume bones, ...)."""


@dataclass
class KeyframeChannel:
    """One animation curve: key times (FBX units) and values for a component."""

    times_ms: list[float] = field(default_factory=list)
    values: list[float] = field(default_factory=list)
    interpolation: list[int] = field(default_factory=list)

    def __len__(self) -> int:
        return len(self.times_ms)

    @property
    def count(self) -> int:
        return len(self.times_ms)

    @property
    def is_empty(self) -> bool:
        return not self.times_ms

    def value_at(self, time_ms: float) -> float:
        """Sample with linear interpolation and constant extrapolation.

        FBX curve interpolation codes: 0 constant, 1 linear, 2 cubic,
        3 mixed, 4 weighted.  A true cubic evaluation needs the tangent
        channels, which this reader does not read; see the README's
        "Known limitations".  ``interpolation`` is preserved so the
        validator can report how many keys were non-linear.
        """
        if not self.times_ms:
            raise ValueError("Cannot sample an empty curve")
        if time_ms <= self.times_ms[0]:
            return self.values[0]
        if time_ms >= self.times_ms[-1]:
            return self.values[-1]
        index = bisect.bisect_right(self.times_ms, time_ms) - 1
        index = max(0, min(index, len(self.times_ms) - 2))
        lo, hi = index, index + 1
        span = self.times_ms[hi] - self.times_ms[lo]
        if span <= 0.0:
            return self.values[lo]
        # Snap to an exact key before anything else.  Key times are stored in
        # seconds, so a caller passing a key time back -- or a float32 time
        # read from another tool -- can land a few microseconds either side of
        # it.  ``(t - t_lo) / span`` then loses most of its significant digits
        # to cancellation: at 633.3333 ms the factor comes out 0.9999976 and
        # ``v_lo + factor * (v_hi - v_lo)`` rounds straight back to ``v_lo``,
        # silently returning the *previous* key's value.  The tolerance is set
        # to what a float32 second actually resolves to (~1e-7 s) rather than
        # to float64 epsilon, because that is the error being corrected.
        # This must run *before* the CONSTANT check below: a CONSTANT key
        # adjacent to the requested key would otherwise swallow it.
        tolerance = max(span * 1e-9, KEY_TIME_SNAP_SECONDS * 1000.0)
        if abs(time_ms - self.times_ms[hi]) <= tolerance:
            return self.values[hi]
        if self.interpolation and lo < len(self.interpolation):
            if self.interpolation[lo] == 0:  # CONSTANT
                return self.values[lo]
        factor = (time_ms - self.times_ms[lo]) / span
        return self.values[lo] + (self.values[hi] - self.values[lo]) * factor

    def sample(self, times_ms: Sequence[float], default: float = 0.0) -> np.ndarray:
        """Vectorised :meth:`value_at` over a list of times.

        Same rules as the scalar path -- linear between keys, constant before
        the first and after the last, and CONSTANT-interpolated keys hold
        their left value -- but evaluated for a whole clip in one pass.
        """
        times = np.asarray(times_ms, dtype=np.float64)
        if not self.times_ms:
            return np.full(times.shape, float(default), dtype=np.float64)
        keys = np.asarray(self.times_ms, dtype=np.float64)
        values = np.asarray(self.values, dtype=np.float64)
        lower = np.clip(np.searchsorted(keys, times, side="right") - 1, 0, keys.size - 1)
        upper = np.minimum(lower + 1, keys.size - 1)
        span = keys[upper] - keys[lower]
        safe = np.where(span > 0.0, span, 1.0)
        factor = (times - keys[lower]) / safe
        out = values[lower] + (values[upper] - values[lower]) * factor
        if self.interpolation:
            flags = np.asarray(self.interpolation, dtype=np.int64)
            out = np.where(flags[lower] == 0, values[lower], out)
        out = np.where(times <= keys[0], values[0], out)
        out = np.where(times >= keys[-1], values[-1], out)
        # Same key-snapping as :meth:`value_at`: a time that is a hair below a
        # key would otherwise interpolate ~99.9999% of the way and then round
        # back to the previous key's value.
        tolerance = np.maximum(span * 1e-9, KEY_TIME_SNAP_SECONDS * 1000.0)
        exact = np.abs(times - keys[lower]) <= tolerance
        out = np.where(exact, values[lower], out)
        return out

    @property
    def time_range_ms(self) -> tuple[float, float]:
        if not self.times_ms:
            return (0.0, 0.0)
        return (self.times_ms[0], self.times_ms[-1])


@dataclass
class SourceBone:
    """One node of the source rig."""

    index: int = -1
    name: str = ""
    fbx_id: int = 0
    parent: int = -1
    children: list[int] = field(default_factory=list)
    model_type: str = "Null"

    bind_local: np.ndarray = field(default_factory=mat_identity)
    bind_world: np.ndarray = field(default_factory=mat_identity)
    bind_local_quat: np.ndarray = field(default_factory=lambda: np.array([0.0, 0.0, 0.0, 1.0]))
    bind_local_translation: np.ndarray = field(default_factory=lambda: np.zeros(3))
    bind_local_scale: np.ndarray = field(default_factory=lambda: np.ones(3))

    rotation_order: str = "XYZ"
    pre_rotation: np.ndarray = field(default_factory=lambda: np.array([0.0, 0.0, 0.0, 1.0]))
    post_rotation: np.ndarray = field(default_factory=lambda: np.array([0.0, 0.0, 0.0, 1.0]))
    geometric_transform: np.ndarray = field(default_factory=mat_identity)
    include_geometric_transform: bool = True

    #: Rest values for channels the clip does not key.  A node with no
    #: ``Lcl Translation`` curve keeps the translation its ``Model`` record
    #: declares -- substituting zero would move every unanimated limb joint
    #: onto its parent's origin and silently distort the rest pose.
    rest_translation: np.ndarray = field(default_factory=lambda: np.zeros(3))
    rest_scale: np.ndarray = field(default_factory=lambda: np.ones(3))

    #: Channel name (``"Lcl Rotation X"``) -> keyframe curve.
    curves: dict[str, KeyframeChannel] = field(default_factory=dict)

    kind: BoneKind = BoneKind.IGNORE
    depth: int = 0
    notes: list[str] = field(default_factory=list)

    #: Derived-value caches, populated lazily by :meth:`_fixed_rotation` and
    #: :meth:`_effective_geometric`.  Both are constant for the life of a
    #: bone, so they are folded once instead of once per key.
    _fixed_cache: object = field(
        default=_UNSET, repr=False, compare=False
    )
    _geometric_cache: object = field(
        default=_UNSET, repr=False, compare=False
    )

    @property
    def is_root(self) -> bool:
        return self.parent < 0

    @property
    def has_animation(self) -> bool:
        return any(not c.is_empty for c in self.curves.values())

    @property
    def animated_channels(self) -> set[str]:
        return {name for name, curve in self.curves.items() if not curve.is_empty}

    @property
    def key_time_range_ms(self) -> tuple[float, float] | None:
        ranges = [c.time_range_ms for c in self.curves.values() if not c.is_empty]
        if not ranges:
            return None
        return (min(r[0] for r in ranges), max(r[1] for r in ranges))

    def curve(self, channel: str) -> KeyframeChannel | None:
        return self.curves.get(channel)

    def component(self, time_ms: float, channel: str, default: float = 0.0) -> float:
        curve = self.curves.get(channel)
        if curve is None or curve.is_empty:
            return default
        return curve.value_at(time_ms)

    def euler_at(self, time_ms: float, prefix: str) -> list[float]:
        return [self.component(time_ms, f"{prefix} {axis}") for axis in ("X", "Y", "Z")]

    def trs_at(self, time_ms: float) -> tuple[list[float], list[float], list[float]]:
        """``(rotation, translation, scale)`` at a time, with rest fallbacks.

        The fallback is the node's own ``Properties70`` value, never a bare
        zero/one: an unkeyed channel is *not* an identity channel.  Rotation
        is converted from FBX degrees to radians on the way out.
        """
        scale = DEGREES_TO_RADIANS
        return (
            [v * scale for v in self.euler_at(time_ms, "Lcl Rotation")],
            [
                self.component(time_ms, f"Lcl Translation {axis}", self.rest_translation[i])
                for i, axis in enumerate("XYZ")
            ],
            [
                self.component(time_ms, f"Lcl Scaling {axis}", self.rest_scale[i])
                for i, axis in enumerate("XYZ")
            ],
        )

    def sample(self, channel: str, times_ms: Sequence[float], default: float = 0.0) -> np.ndarray:
        """Evaluate one channel at every time in one vectorised pass.

        ``np.interp`` is only a valid shortcut between the *first* and *last*
        key; outside that window FBX holds the terminal value, which is what
        the frame range of an animation stack actually extends to.  Inside the
        window the caller must use :meth:`KeyframeChannel.value_at`, so this
        method reports a keyframe curve as unusable rather than lying about it.
        """
        curve = self.curves.get(channel)
        times = np.asarray(times_ms, dtype=np.float64)
        if curve is None or curve.is_empty:
            return np.full(times.shape, float(default), dtype=np.float64)
        first, last = curve.time_range_ms
        if (times < first - 1e-9).any() or (times > last + 1e-9).any():
            raise ValueError(
                f"cannot batch-sample {channel!r}: frames span "
                f"[{times.min():.4f}, {times.max():.4f}] ms but the curve only "
                f"covers [{first:.4f}, {last:.4f}] ms; use per-frame evaluation"
            )
        return curve.sample(times, default)

    def local_matrix(self, time_ms: float | None = None) -> np.ndarray:
        """Rebuild the node's local matrix at a time from its curves."""
        if time_ms is None:
            return self.bind_local
        rotation, translation, scale = self.trs_at(time_ms)
        return self.compose(rotation, translation, scale)

    def compose(
        self,
        euler: Sequence[float],
        translation: Sequence[float],
        scale: Sequence[float],
    ) -> np.ndarray:
        """Full FBX MABB/ROS + geometric transform composition."""
        return self.compose_many(
            np.asarray([euler], dtype=np.float64),
            np.asarray([translation], dtype=np.float64),
            np.asarray([scale], dtype=np.float64),
        )[0]

    def compose_many(
        self,
        euler: np.ndarray,
        translation: np.ndarray,
        scale: np.ndarray,
    ) -> np.ndarray:
        """Batched :meth:`compose` for ``(n, 3)`` inputs.

        Baking a clip evaluates one local transform per key per bone, which
        is tens of thousands of quaternions.  The pre/post rotation pair and
        the geometric transform are constant per bone, so they are folded
        once and the per-key work is vectorised instead of running a Python
        loop over every key.
        """
        from ..core.mathx import euler_to_quat_many, mat_from_trs_many

        euler = np.atleast_2d(np.asarray(euler, dtype=np.float64))
        translation = np.atleast_2d(np.asarray(translation, dtype=np.float64))
        scale = np.atleast_2d(np.asarray(scale, dtype=np.float64))
        rotation = euler_to_quat_many(euler, self.rotation_order)
        fixed = self._fixed_rotation()
        if fixed is not None:
            rotation = _quat_multiply_many(fixed[None, :], rotation)
        matrices = mat_from_trs_many(translation, rotation, scale)
        geometric = self._effective_geometric()
        if geometric is not None:
            matrices = matrices @ geometric
        return matrices

    def _fixed_rotation(self) -> np.ndarray | None:
        """``pre * post^-1``, cached, or ``None`` when both are identity."""
        if self._fixed_cache is _UNSET:
            pre = np.asarray(self.pre_rotation, dtype=np.float64)
            post = np.asarray(self.post_rotation, dtype=np.float64)
            identity = np.array([0.0, 0.0, 0.0, 1.0])
            if (np.allclose(pre, identity, atol=1e-12)
                    and np.allclose(post, identity, atol=1e-12)):
                self._fixed_cache = None
            else:
                from ..core.mathx import quat_inverse, quat_multiply

                self._fixed_cache = quat_multiply(pre, quat_inverse(post))
        return self._fixed_cache

    def _effective_geometric(self) -> np.ndarray | None:
        """The geometric transform when it has to be applied, else ``None``."""
        if self._geometric_cache is _UNSET:
            value = None
            if self.include_geometric_transform and self.geometric_transform is not None:
                if not np.allclose(self.geometric_transform, mat_identity(), atol=1e-12):
                    value = self.geometric_transform
            self._geometric_cache = value
        return self._geometric_cache

    def __repr__(self) -> str:  # pragma: no cover - debugging aid
        return f"SourceBone({self.name!r}, parent={self.parent}, children={self.children})"


@dataclass
class Clip:
    """One animation stack, sampled at the source's own key times."""

    name: str
    rig_index: int = 0
    #: Key times in **seconds**, relative to the clip start, sorted ascending.
    times_s: list[float] = field(default_factory=list)
    fps: float = DEFAULT_FPS
    start_ms: float = 0.0
    stop_ms: float = 0.0
    #: bone index -> ``(n_frames, 7)`` array of ``[qx, qy, qz, qw, tx, ty, tz]``
    #: in FBX local space (geometric transform already applied).
    poses: dict[int, np.ndarray] = field(default_factory=dict)
    source_animated_bones: list[int] = field(default_factory=list)
    #: Count of source keys whose interpolation was not linear/constant.
    non_linear_keys: int = 0

    @property
    def duration(self) -> float:
        return (self.stop_ms - self.start_ms) / 1000.0

    @property
    def frame_count(self) -> int:
        return len(self.times_s)

    def is_empty(self) -> bool:
        return not self.times_s or not self.poses

    def bone_local(self, bone_index: int, frame: int) -> np.ndarray:
        return self.poses[bone_index][frame]

    def describe(self) -> str:
        return (
            f"{self.name!r}: {self.frame_count} keys, {self.duration:.3f}s, "
            f"{len(self.poses)} animated bones, source range "
            f"[{self.start_ms:.0f}, {self.stop_ms:.0f}] ms @ {self.fps:g} fps"
        )


@dataclass
class SourceRig:
    """A whole FBX: one or more armatures with their clips."""

    bones: list[SourceBone] = field(default_factory=list)
    clips: list[Clip] = field(default_factory=list)
    rigs: list[int] = field(default_factory=list)
    document: FbxDocument | None = None
    path: str = ""
    up_axis: str | None = None
    unit_scale_factor: float | None = None
    creator: str = ""
    fbx_version: int = 0
    include_geometric_transform: bool = True
    scale_to_metres: float = 0.01
    include_unconnected_geometry: bool = False

    # -- lookups ------------------------------------------------------------ #
    def by_name(self, name: str) -> SourceBone | None:
        target = _normalize(name)
        for bone in self.bones:
            if _normalize(bone.name) == target:
                return bone
        return None

    def find(self, name: str) -> SourceBone:
        bone = self.by_name(name)
        if bone is None:
            raise KeyError(f"No source bone named {name!r}")
        return bone

    def root_of(self, bone: SourceBone | int) -> SourceBone:
        index = bone.index if isinstance(bone, SourceBone) else bone
        guard = 0
        while self.bones[index].parent >= 0 and guard <= len(self.bones):
            index = self.bones[index].parent
            guard += 1
        return self.bones[index]

    def rig_bones(self, rig_root: int) -> list[SourceBone]:
        """All bones in one armature, parents before children."""
        out: list[SourceBone] = []
        frontier = [rig_root]
        while frontier:
            current = frontier.pop(0)
            out.append(self.bones[current])
            frontier.extend(self.bones[current].children)
        return out

    def clip(self, name: str) -> Clip | None:
        target = name.casefold()
        for clip in self.clips:
            if clip.name.casefold() == target:
                return clip
        return None

    @property
    def animated_clips(self) -> list[Clip]:
        return [c for c in self.clips if not c.is_empty()]

    def clip_of_rig(self, rig_root: int) -> Clip | None:
        for clip in self.clips:
            if clip.rig_index == rig_root and not clip.is_empty():
                return clip
        return None

    def describe(self) -> str:
        lines = [
            f"Source rig: {self.path or '<memory>'}",
            f"  FBX {self.fbx_version} creator={self.creator!r}",
            f"  up axis={self.up_axis} unit scale={self.unit_scale_factor} "
            f"-> {self.scale_to_metres:g} m/unit",
            f"  geometric transform: "
            f"{'included' if self.include_geometric_transform else 'excluded'}",
            f"  bones: {len(self.bones)}  armatures: {len(self.rigs)}  "
            f"clips: {len(self.clips)}",
        ]
        for root_index in self.rigs:
            root = self.bones[root_index]
            lines.append(f"  armature {root.name!r}: {len(self.rig_bones(root_index))} bones")
        for clip in self.clips:
            lines.append(f"    {clip.describe()}")
        return "\n".join(lines)


def _normalize(name: str) -> str:
    return "".join(ch for ch in name.casefold() if ch.isalnum())


# --------------------------------------------------------------------------- #
# loader
# --------------------------------------------------------------------------- #
def load_source_rig(
    path: str,
    *,
    include_geometric_transform: bool = True,
    rig_index: int | None = None,
) -> SourceRig:
    """Load an FBX and return its rigs, bind pose and clips."""
    document = read_fbx(path)
    return build_source_rig(
        document,
        path=os.path.abspath(path),
        include_geometric_transform=include_geometric_transform,
        rig_index=rig_index,
    )


def build_source_rig(
    document: FbxDocument,
    *,
    path: str = "",
    include_geometric_transform: bool = True,
    rig_index: int | None = None,
    resample: str = "keys",
) -> SourceRig:
    """Assemble a :class:`SourceRig` from an already parsed document.

    ``resample``:
        ``"keys"``  keep the exact union of the source's key times (default,
                    nothing is invented and nothing is dropped),
        ``"grid"``   resample onto a uniform ``fps`` grid.
    """
    objects = document.objects
    if objects is None:
        raise FbxParseError("FBX has no Objects section")

    models = objects.find_all("Model")
    if not models:
        raise FbxParseError("FBX contains no Model objects")

    by_fbx_id: dict[int, FbxNode] = {node.object_id: node for node in models}
    parent_of, children_of = _hierarchy_from_connections(document, set(by_fbx_id))

    binding = _bind_curves(document, objects)
    curves_by_model = binding["curves_by_model"]
    stack_of_curve_node = binding["stack_of_curve_node"]
    curve_nodes_of_model = binding["curve_nodes_of_model"]

    bones: list[SourceBone] = []
    index_of_id: dict[int, int] = {}
    for node in models:
        fbx_id = node.object_id
        index_of_id[fbx_id] = len(bones)
        bones.append(_make_bone(node, fbx_id, include_geometric_transform))
    for bone in bones:
        bone.index = index_of_id[bone.fbx_id]

    for bone in bones:
        bone.parent = index_of_id.get(parent_of.get(bone.fbx_id), -1)
    for bone in bones:
        bone.children = sorted(
            index_of_id[cid] for cid in children_of.get(bone.fbx_id, []) if cid in index_of_id
        )

    order = _topological_order(bones)
    for index in order:
        bone = bones[index]
        bone.depth = 0 if bone.parent < 0 else bones[bone.parent].depth + 1
    for index in order:
        bone = bones[index]
        bone.bind_world = (
            bone.bind_local if bone.parent < 0
            else bones[bone.parent].bind_world @ bone.bind_local
        )

    for bone in bones:
        channels = curves_by_model.get(bone.fbx_id, {})
        if channels:
            bone.curves = channels

    rig = SourceRig(
        bones=bones,
        document=document,
        path=path,
        up_axis=document.up_axis,
        unit_scale_factor=document.unit_scale_factor,
        creator=document.creator,
        fbx_version=document.version,
        include_geometric_transform=include_geometric_transform,
    )
    rig.scale_to_metres = _unit_scale_to_metres(document.unit_scale_factor)
    rig.rigs = _detect_rigs(bones)
    rig.clips = _build_clips(
        rig, objects, stack_of_curve_node,
        resample=resample, curve_nodes_of_model=curve_nodes_of_model,
    )
    if rig_index is not None and 0 <= rig_index < len(rig.rigs):
        _restrict_to_rig(rig, rig.rigs[rig_index])
    return rig


def _unit_scale_to_metres(unit_scale: float | None) -> float:
    """FBX ``UnitScaleFactor``: 1 = cm, 100 = m, 10 = dm, 0.01 = mm."""
    if not unit_scale:
        return 0.01
    metres = float(unit_scale) / 100.0
    if not (0.5 <= metres <= 100.0):
        return 0.01
    return metres


def _connection_type(connection: FbxNode) -> str:
    """``OO`` / ``OP`` / ``PO`` / ``PP``, read positionally or by name."""
    value = connection.named("Type", 0, "")
    return str(value).strip().casefold() if value is not None else ""


def _connection_field(connection: FbxNode, index: int) -> int:
    """``Src`` (1) / ``Dst`` (2) of a connection record."""
    name = {1: "Src", 2: "Dst"}.get(index)
    if name is not None:
        prop = connection.get_prop(name)
        if prop is not None:
            return prop.as_int(-1)
    return connection.prop_int(index, -1)


def _hierarchy_from_connections(document: FbxDocument, valid_ids: set[int]):
    parent_of: dict[int, int] = {}
    children_of: dict[int, list[int]] = {mid: [] for mid in valid_ids}
    for connection in document.connections:
        if _connection_type(connection) != "oo":
            continue
        src = _connection_field(connection, 1)
        dst = _connection_field(connection, 2)
        if src not in valid_ids or dst not in valid_ids:
            continue
        # FBX writes ``C: "OO", <child id>, <parent id>``: ``Src`` is the
        # child, ``Dst`` is the parent it hangs off.
        parent_of[src] = dst
        children_of.setdefault(dst, []).append(src)
    return parent_of, children_of


def _make_bone(node: FbxNode, fbx_id: int, include_geometric: bool) -> SourceBone:
    """Build a source bone from a ``Model`` record.

    FBX 7.4 keeps the local TRS in the node's ``Properties70`` block, not in
    its property list.  Reading the legacy names positionally instead makes
    every bone of a rig collapse onto the origin -- silent, and wrong -- so
    both layouts are accepted and Properties70 wins when it is present.
    """
    order = (
        node.string("RotationOrder", "DefaultRotationOrder")
        or node.prop70("RotationOrder", 4, "")
        or DEFAULT_ROTATION_ORDER
    )
    if order not in ("XYZ", "XZY", "YXZ", "YZX", "ZXY", "ZYX"):
        order = DEFAULT_ROTATION_ORDER

    def local(name: str, *legacy: str, default: Sequence[float] = (0.0, 0.0, 0.0)):
        if node.find("Properties70") is not None:
            return node.prop70_vec3(name, default)
        return node.vec3(legacy[0] if legacy else name, default)

    bone = SourceBone(
        name=node.primary_name or node.string("Name") or f"node_{fbx_id}",
        fbx_id=fbx_id,
        model_type=node.subclass or node.string("Type", "Null") or "Null",
        rotation_order=order,
        pre_rotation=_euler_prop(node, "Lcl PreRotation", "PreRotation", order),
        post_rotation=_euler_prop(node, "Lcl PostRotation", "PostRotation", order),
        geometric_transform=_geometric_transform(node),
        include_geometric_transform=include_geometric,
    )
    translation = local("Lcl Translation", "Translation")
    rotation_euler = [v * DEGREES_TO_RADIANS for v in local("Lcl Rotation", "Rotation")]
    scale = local("Lcl Scaling", "Scaling", default=(1.0, 1.0, 1.0))
    bone.bind_local = bone.compose(rotation_euler, translation, scale)
    try:
        t, q, s = mat_decompose(bone.bind_local)
    except ValueError:
        t = np.array(translation, dtype=np.float64)
        q = euler_to_quat(rotation_euler, order)
        s = np.array(scale, dtype=np.float64)
    bone.bind_local_translation = t
    bone.bind_local_quat = q
    bone.bind_local_scale = s
    # Unkeyed channels hold these rest values for the whole clip.
    bone.rest_translation = np.array(translation, dtype=np.float64)
    bone.rest_scale = np.array(scale, dtype=np.float64)
    return bone


def _euler_prop(node: FbxNode, name: str, legacy: str, order: str) -> np.ndarray:
    """A pre/post rotation as a quaternion, from either FBX layout."""
    if node.find("Properties70") is not None:
        values = node.prop70_vec3(name)
    else:
        values = node.get(legacy)
    if values is None:
        return np.array([0.0, 0.0, 0.0, 1.0])
    try:
        vector = [float(v) for v in values]
    except (TypeError, ValueError):
        return np.array([0.0, 0.0, 0.0, 1.0])
    if len(vector) == 3:
        # FBX stores pre/post rotation in degrees like every other Euler.
        return euler_to_quat([v * DEGREES_TO_RADIANS for v in vector[:3]], order)
    if len(vector) == 4:
        return quat_normalize(np.array(vector, dtype=np.float64))
    return np.array([0.0, 0.0, 0.0, 1.0])


def _geometric_transform(node: FbxNode) -> np.ndarray:
    has = node.get("HasGeometricTransform", "Has Geometric Transform")
    if has is not None and str(has).strip().lower() in {"f", "false", "0"}:
        return mat_identity()
    local = node.get("GeometricTransform", "Geometric Transform")
    if not isinstance(local, (list, tuple)) or len(local) < 16:
        return mat_identity()
    values = [float(v) for v in local]
    return _matrix_from_flat(values)


def _matrix_from_flat(values: Sequence[float]) -> np.ndarray:
    """FBX stores 4x4 matrices row-major; this project uses column vectors."""
    rows = [values[0:4], values[4:8], values[8:12], values[12:16]]
    return np.array(rows, dtype=np.float64).T.copy()


def _topological_order(bones: list[SourceBone]) -> list[int]:
    order: list[int] = []
    seen: set[int] = set()

    def visit(index: int, stack: set[int]) -> None:
        if index in seen:
            return
        if index in stack:
            raise FbxParseError("Cycle detected in the FBX model hierarchy")
        stack.add(index)
        parent = bones[index].parent
        if 0 <= parent < len(bones):
            visit(parent, stack)
        stack.discard(index)
        seen.add(index)
        order.append(index)

    for bone in bones:
        visit(bone.index, set())
    return order


# --------------------------------------------------------------------------- #
# animation binding
# --------------------------------------------------------------------------- #
def _bind_curves(document: FbxDocument, objects: FbxNode) -> dict:
    """Resolve every animated channel into ``model id -> channel -> curve``.

    Two bindings exist in the wild and both are supported:

    * FBX 7.0 - 7.3 put the component curves on the ``AnimationCurveNode``
      as ``Props`` / ``AttrDataComposite`` properties.
    * FBX 7.4 replaced those with ``OP`` connections.  One of them reads
      ``C: "OP", <curve node>, <model>, "|Lcl Rotation"`` and three more read
      ``C: "OP", <curve>, <curve node>, "d|X"`` for the X/Y/Z components.
    """
    curve_node_ids = {node.object_id for node in objects.find_all("AnimationCurveNode")}
    curve_by_node: dict[int, dict[str, FbxNode]] = {}
    op_component: list[tuple[int, int, str]] = []   # (curve id, node id, comp)
    op_property: list[tuple[int, int, str]] = []    # (node id, model id, prop)

    for connection in document.connections:
        kind = _connection_type(connection)
        if kind != "op":
            continue
        src = _connection_field(connection, 1)
        dst = _connection_field(connection, 2)
        prop = str(connection.prop(3, "") or "").strip()
        if prop.startswith("|"):
            prop = prop[1:]
        if prop in ("d|X", "d|Y", "d|Z", "X", "Y", "Z"):
            component = prop.split("|")[-1]
            if src in curve_node_ids:
                op_component.append((dst, src, component))
            else:
                op_component.append((src, dst, component))
        else:
            op_property.append((src, dst, prop))

    for curve_id, node_id, component in op_component:
        curve_by_node.setdefault(node_id, {})[component] = _find_curve(
            objects, curve_id
        )

    # Legacy layout.
    legacy: dict[int, dict[str, FbxNode]] = {}
    for node in objects.find_all("AnimationCurveNode"):
        props = node.get_prop("Props")
        attr = node.get_prop("AttrDataComposite")
        if props is None or not props.is_array or attr is None or not attr.is_array:
            continue
        curve_ids = [int(v) for v in attr.value]
        comp_names = [str(p) for p in props.value[2:]]
        holder = _find_curve(objects, curve_ids[-1] if curve_ids else -1)
        if holder is None:
            continue
        mapped = {}
        for comp in comp_names:
            name = _channel_name(comp)
            if name:
                mapped[name] = holder
        legacy[node.object_id] = mapped

    channels_by_node: dict[int, dict[str, KeyframeChannel]] = {}
    for node_id, mapping in curve_by_node.items():
        resolved: dict[str, KeyframeChannel] = {}
        for component, curve_node in mapping.items():
            channel = _read_curve(curve_node).get(VALUE_KEY) if curve_node else None
            if channel is not None:
                resolved[component] = channel
        if resolved:
            channels_by_node[node_id] = resolved
    for node_id, mapping in legacy.items():
        target = channels_by_node.setdefault(node_id, {})
        for name, curve_node in mapping.items():
            channel = _read_curve(curve_node).get(VALUE_KEY)
            if channel is not None and name not in target:
                target[name] = channel

    curves_by_model: dict[int, dict[str, KeyframeChannel]] = {}
    for node_id, model_id, prop in op_property:
        channels = channels_by_node.get(node_id)
        if not channels:
            continue
        target = curves_by_model.setdefault(model_id, {})
        for component, channel in channels.items():
            # The OP connection names the animated property ("Lcl Rotation"),
            # the curve node names the component ("X"); the two combine into
            # the channel name the rest of the pipeline looks up.
            if prop in ("Lcl Translation", "Lcl Rotation", "Lcl Scaling",
                        "PreRotation", "PostRotation", "Visibility"):
                name = f"{prop} {component}" if component in "XYZR" else f"{prop} {component}"
            else:
                name = component if prop in ("", component) else f"{prop} {component}"
            target[name] = channel

    layer_of_node: dict[int, int] = {}
    layer_ids = {node.object_id for node in objects.find_all("AnimationLayer")}
    stack_of_layer: dict[int, int] = {}
    for connection in document.connections:
        if _connection_type(connection) != "oo":
            continue
        src = _connection_field(connection, 1)
        dst = _connection_field(connection, 2)
        if src in curve_node_ids:
            layer_of_node[src] = dst
        elif src in layer_ids:
            # FBX 7.4 links a layer to its stack with a plain object
            # connection; older files carry a ``Stack`` property instead.
            stack_of_layer[src] = dst
    for node in objects.find_all("AnimationLayer"):
        declared = node.prop_int(2, -1)
        if node.object_id not in stack_of_layer and declared >= 0:
            stack_of_layer[node.object_id] = declared
    return {
        "curves_by_model": curves_by_model,
        "curve_nodes_of_model": _group(op_property, 1, 0),
        "stack_of_curve_node": {
            node_id: stack_of_layer[layer]
            for node_id, layer in layer_of_node.items()
            if layer in stack_of_layer
        },
        "curve_node_ids": curve_node_ids,
    }


def _group(rows: Sequence[tuple], key: int, value: int) -> dict:
    out: dict = {}
    for row in rows:
        out.setdefault(row[key], []).append(row[value])
    return out


def _array_payload(node: FbxNode, child_name: str) -> list | None:
    """Read an array payload stored under ``child_name``.

    FBX writes ``KeyTime`` / ``KeyValueFloat`` as child records whose single
    property is unnamed and holds the array, so the value is taken
    positionally with a named lookup as a fallback.
    """
    child = node.find(child_name)
    if child is not None:
        for prop in child.properties:
            if isinstance(prop.value, (list, tuple)):
                return list(prop.value)
        return []
    value = node.get(child_name)
    if isinstance(value, (list, tuple)):
        return list(value)
    return None


def _find_curve(objects: FbxNode, curve_id: int) -> FbxNode | None:
    if curve_id is None or curve_id < 0:
        return None
    for node in objects.find_all("AnimationCurve"):
        if node.object_id == curve_id:
            return node
    return None


def _read_curve(node: FbxNode) -> dict[str, KeyframeChannel]:
    """Read one ``AnimationCurve`` record.

    ``KeyTime`` / ``KeyValueFloat`` are properties on the curve record, not
    child nodes, and the times are in FBX time units (1 s = 46186158000).
    """
    times = _array_payload(node, "KeyTime")
    values = _array_payload(node, "KeyValueFloat")
    if times is None or values is None:
        return {}
    try:
        time_values = [float(v) for v in times]
        value_values = [float(v) for v in values]
    except TypeError:
        return {}
    count = min(len(time_values), len(value_values))
    if count == 0:
        return {}
    flags_raw = _array_payload(node, "KeyAttrFlags")
    flags: list[int] = []
    if isinstance(flags_raw, (list, tuple)):
        flags = [int(v) for v in flags_raw]
    channel = KeyframeChannel(
        times_ms=[time_values[i] / FBX_TIME_UNITS_PER_SECOND * 1000.0 for i in range(count)],
        values=value_values[:count],
        interpolation=(list(flags) + [0] * count)[:count],
    )
    order = sorted(range(count), key=lambda i: channel.times_ms[i])
    channel.times_ms = [channel.times_ms[i] for i in order]
    channel.values = [channel.values[i] for i in order]
    channel.interpolation = [channel.interpolation[i] for i in order]
    return {VALUE_KEY: channel}


def _channel_name(prop_name: str) -> str:
    """``d|X`` -> ``"X"``; ``Lcl Rotation`` -> ``"Lcl Rotation "``."""
    name = prop_name.strip()
    if "|" in name:
        name = name.split("|", 1)[1]
    name = name.strip()
    if name in ("X", "Y", "Z", "R", "W"):
        return name
    if name and name[-1].upper() in "XYZR" and " " in name:
        return name
    return f"{name} " if name else ""


# --------------------------------------------------------------------------- #
# clips
# --------------------------------------------------------------------------- #
#: Model sub-classes that can act as a skeleton root.  Geometry, lights and
#: cameras are excluded so a DCC scene with meshes does not look like a rig.
SKELETON_MODEL_TYPES = frozenset(
    {"LimbNode", "Root", "Limb", "Null", "Skeleton", "Armature", "Bone", "Joint"}
)
#: Model sub-classes that are never part of a skeleton.
GEOMETRY_MODEL_TYPES = frozenset({"Mesh", "Light", "Camera", "Limb", "Patch"})


def _subtree(bones: list[SourceBone], index: int, seen: set[int] | None = None) -> int:
    """Number of bones in the subtree rooted at ``index`` (cycle safe)."""
    seen = set() if seen is None else seen
    if index in seen or not (0 <= index < len(bones)):
        return 0
    seen.add(index)
    return 1 + sum(_subtree(bones, child, seen) for child in bones[index].children)


def _detect_rigs(bones: list[SourceBone]) -> list[int]:
    """Find the armature roots of an FBX, most significant first.

    A rig root is a skeleton node with no parent.  Scenes that ship more than
    one armature (a character plus a prop, say) keep them all so the user can
    choose, but the rig carrying the animation and the most bones sorts first
    so index 0 is the character.
    """
    if not bones:
        return []
    skeleton = [
        b.index for b in bones
        if b.model_type in SKELETON_MODEL_TYPES
        and b.model_type not in GEOMETRY_MODEL_TYPES
    ]
    skeleton_set = set(skeleton)
    roots = [b.index for b in bones if b.parent < 0 and b.index in skeleton_set]
    if not roots:
        # A file whose skeleton is not parented at all: the node with the
        # largest subtree is the closest thing to a root.
        best = max(
            skeleton or list(range(len(bones))),
            key=lambda i: (_subtree(bones, i), len(bones[i].curves)),
        )
        return [best]

    animated = {b.index for b in bones if b.curves}

    def _animated_in_subtree(index: int) -> int:
        seen: set[int] = set()
        stack = [index]
        total = 0
        while stack:
            current = stack.pop()
            if current in seen or not (0 <= current < len(bones)):
                continue
            seen.add(current)
            total += current in animated
            stack.extend(bones[current].children)
        return total

    return sorted(
        roots,
        key=lambda i: (_animated_in_subtree(i), _subtree(bones, i), -i),
        reverse=True,
    )


def _build_clips(
    rig: SourceRig,
    objects: FbxNode,
    stack_of_curve_node: dict[int, int],
    *,
    resample: str = "keys",
    curve_nodes_of_model: dict[int, list[int]] | None = None,
) -> list[Clip]:
    stacks = objects.find_all("AnimationStack")
    if not stacks:
        return []
    if curve_nodes_of_model is None:
        curve_nodes_of_model = {}
        for bone in rig.bones:
            curve_nodes_of_model[bone.fbx_id] = _curve_nodes_of_model(rig, bone.fbx_id)
    clips: list[Clip] = []
    for stack in stacks:
        name = stack.primary_name or f"Take {len(clips)}"
        # ``LocalStart`` / ``LocalStop`` / ``CustomFrameRate`` live in the
        # stack's ``Properties70`` block, not in its property list.
        start_ms = _fbx_time_ms(stack.prop70("LocalStart", 4, stack.prop70("LocalStart", -1, 0)))
        stop_ms = _fbx_time_ms(stack.prop70("LocalStop", 4, stack.prop70("LocalStop", -1, 0)))
        fps_value = stack.prop70("CustomFrameRate", 4, None)
        fps = float(fps_value) if isinstance(fps_value, (int, float)) and fps_value else DEFAULT_FPS
        stack_id = stack.object_id

        members: list[SourceBone] = []
        for bone in rig.bones:
            node_ids = curve_nodes_of_model.get(bone.fbx_id, ())
            if any(stack_of_curve_node.get(nid) == stack_id for nid in node_ids):
                members.append(bone)
        if not members:
            continue
        if stop_ms <= start_ms:
            bounds = [b.key_time_range_ms for b in members if b.key_time_range_ms]
            if not bounds:
                continue
            start_ms = min(r[0] for r in bounds)
            stop_ms = max(r[1] for r in bounds)

        times = _clip_times(members, start_ms, stop_ms, fps, resample)
        clip = Clip(
            name=name,
            fps=fps,
            start_ms=start_ms,
            stop_ms=stop_ms,
            times_s=[(t - start_ms) / 1000.0 for t in times],
        )
        non_linear = 0
        for bone in members:
            # Sample every curve in one vectorised pass, then fold the
            # constant per-bone parts (pre/post rotation, geometric) once.
            euler = np.empty((len(times), 3), dtype=np.float64)
            translation = np.empty((len(times), 3), dtype=np.float64)
            scale = np.empty((len(times), 3), dtype=np.float64)
            for column, axis in enumerate(("X", "Y", "Z")):
                euler[:, column] = (
                    bone.sample(f"Lcl Rotation {axis}", times, 0.0) * DEGREES_TO_RADIANS
                )
                translation[:, column] = bone.sample(
                    f"Lcl Translation {axis}", times, bone.rest_translation[column]
                )
                scale[:, column] = bone.sample(
                    f"Lcl Scaling {axis}", times, bone.rest_scale[column]
                )
            matrices = bone.compose_many(euler, translation, scale)
            rows = np.zeros((len(times), 7), dtype=np.float64)
            for frame in range(len(times)):
                try:
                    t, q, _ = mat_decompose(matrices[frame])
                except ValueError:
                    t = bone.bind_local_translation
                    q = bone.bind_local_quat
                rows[frame, 0:4] = q
                rows[frame, 4:7] = t
            for curve in bone.curves.values():
                if any(flag not in (0, 1, 3, 4) for flag in curve.interpolation):
                    non_linear += 1
                    break
            clip.poses[bone.index] = rows
        clip.source_animated_bones = sorted(clip.poses)
        clip.non_linear_keys = non_linear
        clips.append(clip)
    return clips


def _curve_nodes_of_model(rig: SourceRig, model_fbx_id: int) -> list[int]:
    """AnimationCurveNode ids that target this model (needs the document)."""
    document = rig.document
    assert document is not None
    out: list[int] = []
    for connection in document.connections:
        if _connection_type(connection) != "op":
            continue
        if _connection_field(connection, 2) == model_fbx_id:
            out.append(_connection_field(connection, 1))
    return out


def _clip_times(
    bones: Sequence[SourceBone],
    start_ms: float,
    stop_ms: float,
    fps: float,
    resample: str,
) -> list[float]:
    if resample == "grid":
        step = 1000.0 / max(fps, 1e-6)
        count = max(1, int(round((stop_ms - start_ms) / step)) + 1)
        return [start_ms + i * step for i in range(count)]
    collected: set[float] = {start_ms}
    for bone in bones:
        for curve in bone.curves.values():
            if curve.is_empty:
                continue
            for time_ms in curve.times_ms:
                if start_ms - 1e-6 <= time_ms <= stop_ms + 1e-6:
                    collected.add(float(time_ms))
    if stop_ms > start_ms:
        collected.add(stop_ms)
    return sorted(collected)


def _fbx_time_ms(value) -> float:
    if value is None:
        return 0.0
    try:
        return float(value) / FBX_TIME_UNITS_PER_SECOND * 1000.0
    except (TypeError, ValueError):
        return 0.0


def _restrict_to_rig(rig: SourceRig, root_index: int) -> None:
    keep = {b.index for b in rig.rig_bones(root_index)}
    for bone in rig.bones:
        if bone.index not in keep:
            bone.kind = BoneKind.IGNORE
    for clip in rig.clips:
        clip.rig_index = root_index
        clip.poses = {k: v for k, v in clip.poses.items() if k in keep}
        clip.source_animated_bones = [b for b in clip.source_animated_bones if b in keep]
