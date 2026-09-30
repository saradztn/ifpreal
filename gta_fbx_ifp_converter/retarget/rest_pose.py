"""Rest-pose and local-axis correction.

An animation is a set of rotations *relative to a rest pose*.  Two rigs almost
never share a rest pose, and they rarely agree on which way a bone's own "up"
axis points, so a rotation sampled on one rig is meaningless on the other
unless both are expressed in the same frame.

This module builds the correction that makes them comparable, out of four
pieces of real data:

* the **source rest pose**, taken from the FBX armatures, in the source's own
  axis convention and its own units;
* the **target rest pose**, taken from the target DFF's HAnim frames -- the
  bind rotations the ped actually ships with, not a preset;
* the **source bone direction**, recovered from where the bone's children sit
  in the rest pose, which is how a bone that runs down the leg and a bone
  that runs out along the arm can be told apart;
* the **target bone direction**, recovered the same way from the DFF.

From those it derives a per-bone quaternion ``Q`` such that, for a source bone
at rest and its target at rest::

    Q_target_local = Q * Q_source_rest_local * Q_source_rest_local^-1 * ...

written out in full, for a bone ``b`` with target ``t`` and source parent
``s``::

    local_target(t) = A * local_source(b) * A^-1      (same-parent case)
    local_target(t) = A * local_source(b) * A^-1 * R  (general case)

where ``A`` is the rest-pose reorientation and ``R`` re-bases a bone whose
source parent and target parent are not themselves mapped to each other.

Nothing here guesses.  If a bone has no recoverable direction, the correction
for it is the identity and :attr:`RestPoseCorrection.unknown` says so, rather
than a plausible-looking rotation that quietly bends the elbow backwards.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Sequence

import numpy as np

from ..core import mathx
from ..mapping.mapper import BoneMatch, Side

#: A direction, a bone index, and how the direction was obtained.  Provenance
#: matters here: a direction measured from a real child is evidence, a
#: direction assumed from the role is a guess, and the two must not be
#: reported with the same confidence.
DERIVED_CHILD = "child"
DERIVED_ROLE = "role"
DERIVED_AXIS = "axis"
DERIVED_NONE = "none"


@dataclass(frozen=True)
class BoneAxis:
    """Where a bone points, and how we know."""

    #: Unit vector in the bone's own local space.
    direction: np.ndarray
    #: One of the ``DERIVED_*`` constants.
    provenance: str
    #: Bone whose child revealed the direction, when there was one.
    child_index: int | None = None

    @property
    def is_measured(self) -> bool:
        """True when a real child bone fixed this direction.

        Measured and assumed directions both produce a usable rotation, but
        only a measured one may be trusted above the confidence floor.
        """
        return self.provenance == DERIVED_CHILD


def _unit(vector: np.ndarray) -> np.ndarray:
    norm = float(np.linalg.norm(vector))
    if norm < 1e-12:
        return np.zeros(3, dtype=np.float64)
    return np.asarray(vector, dtype=np.float64) / norm


def bone_axes_from_pose(
    bones: Sequence,
    parents: Sequence[int],
    translations: Sequence[np.ndarray],
    quaternions: Sequence[np.ndarray],
    *,
    prefer_y: bool = True,
) -> list[BoneAxis]:
    """Recover each bone's own axis from a rest pose.

    A bone's local "along" direction is wherever its children go.  When a bone
    has children, the average of the child offsets rotated into the bone's
    frame is its axis, and that measurement is taken once and cached.  When it
    has none -- fingertips, toe tips, the head top -- the bone's direction is
    inherited from its parent, because a leaf bone is drawn along its
    parent's line in every rig format worth the name.

    ``prefer_y`` selects the convention for bones that have no children and no
    parent: GTA draws its bones along +Y, so that is the fallback, not an
    assumption of convenience.
    """
    count = len(bones)
    children: list[list[int]] = [[] for _ in range(count)]
    for index, parent in enumerate(parents):
        if parent is not None and 0 <= parent < count and parent != index:
            children[parent].append(index)

    axes: list[BoneAxis | None] = [None] * count
    for index in range(count):
        kids = children[index]
        if not kids:
            continue
        offsets = []
        for child in kids:
            delta = np.asarray(translations[child], dtype=np.float64) - np.asarray(
                translations[index], dtype=np.float64
            )
            if float(np.linalg.norm(delta)) > 1e-9:
                # Into the bone's own frame, so a rotated rest pose is
                # measured correctly rather than assumed axis-aligned.
                offsets.append(
                    mathx.quat_rotate_vector(
                        mathx.quat_inverse(quaternions[index]), delta
                    )
                )
        if not offsets:
            continue
        direction = _unit(np.sum(offsets, axis=0))
        if float(np.linalg.norm(direction)) < 0.5:
            # Offsets that cancel out describe a branching bone (a spine with
            # arms on it), not a direction.  The average is near zero, so
            # there is nothing to measure and the axis stays unknown.
            continue
        axes[index] = BoneAxis(direction, DERIVED_CHILD, kids[0])

    # Leaves inherit from the nearest ancestor that was measured.
    for index in range(count):
        if axes[index] is not None:
            continue
        parent = parents[index]
        seen: set[int] = set()
        while parent is not None and 0 <= parent < count and parent not in seen:
            if axes[parent] is not None:
                axes[index] = BoneAxis(
                    axes[parent].direction, DERIVED_ROLE, parent
                )
                break
            seen.add(parent)
            parent = parents[parent]

    fallback = np.array([0.0, 1.0, 0.0]) if prefer_y else np.array([0.0, 0.0, 1.0])
    return [
        axis if axis is not None else BoneAxis(fallback, DERIVED_AXIS)
        for axis in axes
    ]


def rotation_between(source: np.ndarray, target: np.ndarray) -> np.ndarray:
    """Shortest rotation carrying unit vector ``source`` onto ``target``.

    The two directions need not be perpendicular, so this is not a pair of
    cross-product frames (which would invent a roll the data does not contain
    and tilt every limb by however much the two bones happen to differ).  It
    is the minimal rotation between the two axes, which is the most that a
    measured direction can honestly support.
    """
    a = _unit(source)
    b = _unit(target)
    if float(np.linalg.norm(a)) < 0.5 or float(np.linalg.norm(b)) < 0.5:
        return np.array([0.0, 0.0, 0.0, 1.0])
    dot = float(np.clip(np.dot(a, b), -1.0, 1.0))
    if dot > 1.0 - 1e-12:
        return np.array([0.0, 0.0, 0.0, 1.0])
    if dot < -1.0 + 1e-12:
        # Opposite directions: any perpendicular axis gives a valid half turn,
        # and the one least likely to be degenerate is a stable choice.
        helper = np.array([1.0, 0.0, 0.0])
        if abs(float(np.dot(helper, a))) > 0.9:
            helper = np.array([0.0, 0.0, 1.0])
        axis = _unit(np.cross(a, helper))
        return mathx.quat_from_axis_angle(axis, np.pi)
    axis = _unit(np.cross(a, b))
    return mathx.quat_normalize(mathx.quat_from_axis_angle(axis, np.arccos(dot)))


@dataclass
class BoneCorrection:
    """The rest-pose correction for one mapped bone."""

    source_index: int
    target_index: int
    #: Carries the source's local frame onto the target's.
    reorient: np.ndarray
    #: Re-bases a bone whose source and target parents are unrelated.
    rebased: np.ndarray
    source_axis: BoneAxis
    target_axis: BoneAxis
    #: Where the two parents sit relative to each other, in target space.
    parent_offset: np.ndarray | None = None
    parent_reorient: np.ndarray | None = None

    @property
    def total(self) -> np.ndarray:
        return mathx.quat_normalize(
            mathx.quat_multiply(self.reorient, self.rebased)
        )

    @property
    def is_identity(self) -> bool:
        return bool(np.allclose(self.total, [0.0, 0.0, 0.0, 1.0], atol=1e-9))

    def is_trustworthy(self) -> bool:
        """Whether this correction rests on measurement rather than fallback.

        A bone whose axis came from the role default can still be corrected,
        but it is not allowed to carry a mapping silently -- the caller is
        expected to lower its confidence or ask the user.
        """
        return self.source_axis.is_measured or self.target_axis.is_measured

    def to_dict(self) -> dict:
        return {
            "source_index": self.source_index,
            "target_index": self.target_index,
            "reorient": [round(float(v), 6) for v in self.reorient],
            "rebased": [round(float(v), 6) for v in self.rebased],
            "source_axis": {
                "direction": [round(float(v), 4) for v in self.source_axis.direction],
                "provenance": self.source_axis.provenance,
            },
            "target_axis": {
                "direction": [round(float(v), 4) for v in self.target_axis.direction],
                "provenance": self.target_axis.provenance,
            },
            "parent_offset": (
                None if self.parent_offset is None
                else [round(float(v), 5) for v in self.parent_offset]
            ),
        }


@dataclass
class RestPoseCorrection:
    """The full set of per-bone corrections, plus what could not be derived."""

    corrections: dict[int, BoneCorrection] = field(default_factory=dict)
    #: Source bone names whose axis could not be measured on either rig.
    unknown: list[str] = field(default_factory=list)
    #: Source bone names corrected on a fallback axis rather than a measured
    #: one, so the caller can decide whether to trust them.
    assumed: list[str] = field(default_factory=list)
    #: Global scale from source units to GTA units, measured from the two
    #: rest poses rather than assumed.
    unit_scale: float = 1.0
    warnings: list[str] = field(default_factory=list)

    def get(self, source_index: int) -> BoneCorrection | None:
        return self.corrections.get(source_index)

    def reorient(self, source_index: int) -> np.ndarray:
        """The rotation to apply, or identity when nothing was derived."""
        correction = self.corrections.get(source_index)
        return correction.total if correction else np.array([0.0, 0.0, 0.0, 1.0])

    @property
    def measured_fraction(self) -> float:
        if not self.corrections:
            return 0.0
        return sum(
            1 for c in self.corrections.values() if c.is_trustworthy()
        ) / len(self.corrections)

    def to_dict(self) -> dict:
        return {
            "unit_scale": self.unit_scale,
            "measured_fraction": round(self.measured_fraction, 4),
            "unknown": list(self.unknown),
            "assumed": list(self.assumed),
            "corrections": {
                str(k): v.to_dict() for k, v in sorted(self.corrections.items())
            },
            "warnings": list(self.warnings),
        }


def _child_lists(parents: Sequence[int]) -> list[list[int]]:
    count = len(parents)
    out: list[list[int]] = [[] for _ in range(count)]
    for index, parent in enumerate(parents):
        if parent is not None and 0 <= parent < count and parent != index:
            out[parent].append(index)
    return out


def _world_positions(
    parents: Sequence[int],
    translations: Sequence[np.ndarray],
    quaternions: Sequence[np.ndarray],
) -> list[np.ndarray]:
    """Rest-pose world position of every bone, in its rig's own units.

    Each translation is the offset from the bone's parent, rotated into world
    by the parent's rest rotation, exactly as a bind matrix would be.  The
    translation of a bone is where *that* bone's origin sits; the length of
    the bone is the distance from its origin to its child's, which is what
    :func:`bone_axes_from_pose` measures and what the unit scale compares.
    """
    count = len(parents)
    out: list[np.ndarray] = []
    for index in range(count):
        local = np.asarray(translations[index], dtype=np.float64)
        parent = parents[index]
        if parent is None or parent < 0 or parent >= index:
            out.append(local.copy())
            continue
        out.append(out[parent] + mathx.quat_rotate_vector(quaternions[parent], local))
    return out


def _world_rotations(
    parents: Sequence[int], quaternions: Sequence[np.ndarray]
) -> list[np.ndarray]:
    count = len(parents)
    out: list[np.ndarray] = []
    for index in range(count):
        parent = parents[index]
        if parent is None or parent < 0 or parent >= index:
            out.append(mathx.quat_normalize(quaternions[index]))
            continue
        out.append(mathx.quat_normalize(
            mathx.quat_multiply(out[parent], quaternions[index])
        ))
    return out


def build_rest_pose_correction(
    source_bones: Sequence,
    source_parents: Sequence[int],
    source_translations: Sequence[np.ndarray],
    source_quaternions: Sequence[np.ndarray],
    target_bones: Sequence,
    target_parents: Sequence[int],
    target_translations: Sequence[np.ndarray],
    target_quaternions: Sequence[np.ndarray],
    matches: Sequence[BoneMatch],
) -> RestPoseCorrection:
    """Derive the per-bone rest-pose correction for a whole mapping.

    The work per bone is:

    1. measure where the source bone points and where the target bone points,
       both in their own rigs' rest poses;
    2. take the rotation that carries one onto the other -- this is the
       "local axis correction", and it is the same for every frame because
       both rigs are rigid;
    3. when the source bone's parent and the target bone's parent are *not*
       themselves a matched pair, measure where each parent actually sits in
       the rest pose and build the rotation that lines them up.  This is the
       part that is usually skipped and always shows up later as a shoulder
       that drifts.
    """
    result = RestPoseCorrection()

    source_axes = bone_axes_from_pose(
        source_bones, source_parents, source_translations, source_quaternions,
        prefer_y=True,
    )
    target_axes = bone_axes_from_pose(
        target_bones, target_parents, target_translations, target_quaternions,
        prefer_y=True,
    )
    source_world = _world_positions(
        source_parents, source_translations, source_quaternions
    )
    target_world = _world_positions(
        target_parents, target_translations, target_quaternions
    )
    source_world_rot = _world_rotations(source_parents, source_quaternions)
    target_world_rot = _world_rotations(target_parents, target_quaternions)

    source_children = _child_lists(source_parents)
    target_children = _child_lists(target_parents)
    result.unit_scale = _measure_unit_scale(
        source_world, target_world, matches, source_children, target_children
    )

    # Which target bone does each source bone's parent land on?  A bone whose
    # parent is not mapped needs its own re-basing; one whose parent is mapped
    # inherits through the chain and must not have its parent folded in twice.
    target_of_source = {m.source_index: m.target_index for m in matches}
    source_of_target = {m.target_index: m.source_index for m in matches}

    for match in matches:
        if not match.is_mapped:
            continue
        s = match.source_index
        t = match.target_index
        if not (0 <= s < len(source_axes) and 0 <= t < len(target_axes)):
            continue

        reorient = rotation_between(
            source_axes[s].direction, target_axes[t].direction
        )
        correction = BoneCorrection(
            source_index=s,
            target_index=t,
            reorient=reorient,
            rebased=np.array([0.0, 0.0, 0.0, 1.0]),
            source_axis=source_axes[s],
            target_axis=target_axes[t],
        )

        s_parent = source_parents[s]
        t_parent = target_parents[t]
        parents_paired = (
            s_parent is not None
            and s_parent >= 0
            and target_of_source.get(s_parent) == t_parent
        )
        if not parents_paired and s_parent is not None and s_parent >= 0 \
                and t_parent is not None and t_parent >= 0:
            correction.parent_offset = (
                target_world[t] - target_world[t_parent]
            )
            correction.parent_reorient = rotation_between(
                _unit(
                    mathx.quat_rotate_vector(
                        source_world_rot[s_parent], source_axes[s].direction
                    )
                ),
                _unit(
                    mathx.quat_rotate_vector(
                        target_world_rot[t_parent], target_axes[t].direction
                    )
                ),
            )
            correction.rebased = mathx.quat_normalize(
                mathx.quat_multiply(
                    mathx.quat_inverse(correction.parent_reorient), reorient
                )
            )

        result.corrections[s] = correction
        if not correction.is_trustworthy():
            result.assumed.append(source_bones[s].name)
        if source_axes[s].provenance == DERIVED_NONE:
            result.unknown.append(source_bones[s].name)

    if result.assumed:
        result.warnings.append(
            f"{len(result.assumed)} bone(s) rest axes were assumed, not "
            f"measured: {', '.join(result.assumed[:8])}"
            + (" ..." if len(result.assumed) > 8 else "")
        )
    return result


def _measure_unit_scale(
    source_world: Sequence[np.ndarray],
    target_world: Sequence[np.ndarray],
    matches: Sequence[BoneMatch],
    source_children: Sequence[Sequence[int]],
    target_children: Sequence[Sequence[int]],
) -> float:
    """How many source units make one GTA unit, measured from the rest poses.

    Taken over bone *lengths* -- the distance from a bone's own origin to its
    child's -- rather than absolute positions, because a rig's world origin
    means nothing: one model stands at the origin, another stands 100cm up.
    Bone length is the one thing two rigs must agree about, and using the
    median rather than the mean keeps a single mis-mapped finger from
    rescaling the whole character.
    """
    ratios: list[float] = []
    for match in matches:
        if not match.is_mapped:
            continue
        ratio = _rest_length_ratio(
            source_world, target_world, match,
            source_children, target_children,
        )
        if ratio is not None and ratio > 0.0:
            ratios.append(ratio)
    if not ratios:
        return 1.0
    return float(np.median(ratios))


#: Bones whose length is a meaningful measure of a human body.  Excludes
#: fingers and toes, whose models are frequently not rigged to the hand, and
#: the root, whose "length" is the whole model.
_LENGTH_ROLES = frozenset({"upper_arm", "forearm", "thigh", "calf", "spine2"})


def _rest_length_ratio(
    source_world: Sequence[np.ndarray],
    target_world: Sequence[np.ndarray],
    match: BoneMatch,
    source_children: Sequence[Sequence[int]],
    target_children: Sequence[Sequence[int]],
) -> float | None:
    """Target bone length / source bone length, or ``None`` if not measurable.

    A bone's length is measured to the *first* child in the chain that plays
    the next role, not to whichever child happens to be listed first: a foot
    has toe bones hanging off it, and the distance to a toe tip is the foot
    plus a toe, which is not comparable to a rig whose foot bone ends at the
    ankle.
    """
    if match.role is None or match.role.value not in _LENGTH_ROLES:
        return None
    s = match.source_index
    t = match.target_index
    if not (0 <= s < len(source_world) and 0 <= t < len(target_world)):
        return None
    source_kid = _chain_child(source_children[s])
    target_kid = _chain_child(target_children[t])
    if source_kid is None or target_kid is None:
        return None
    source_length = float(np.linalg.norm(
        np.asarray(source_world[source_kid]) - np.asarray(source_world[s])
    ))
    target_length = float(np.linalg.norm(
        np.asarray(target_world[target_kid]) - np.asarray(target_world[t])
    ))
    if source_length < 1e-9 or target_length < 1e-9:
        return None
    return target_length / source_length


def _chain_child(
    children: Sequence[int],
    world: Sequence[np.ndarray] | None = None,
) -> int | None:
    """The child a length is measured to: the one furthest along the body axis.

    Where a bone branches -- a wrist carrying four fingers, a neck carrying a
    head and two clavicles -- the child that continues the chain is the one
    whose offset is longest, because fingers are short and splayed while a
    head sits a full head-height away.  With no world positions to compare,
    the first child is the only defensible answer.
    """
    kids = [c for c in children if c is not None and c >= 0]
    if not kids:
        return None
    if world is None:
        return kids[0]
    best = kids[0]
    best_length = -1.0
    for kid in kids:
        length = float(np.linalg.norm(
            np.asarray(world[kid]) - np.asarray(world[best])
        ))
        if length > best_length:
            best_length = length
            best = kid
    return best


__all__ = [
    "BoneAxis",
    "BoneCorrection",
    "RestPoseCorrection",
    "bone_axes_from_pose",
    "rotation_between",
    "build_rest_pose_correction",
    "DERIVED_CHILD",
    "DERIVED_ROLE",
    "DERIVED_AXIS",
    "DERIVED_NONE",
]
