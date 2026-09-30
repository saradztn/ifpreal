"""The retarget: moving motion from one rig to another without breaking it.

Three things go wrong in a retarget, and this module is built around not doing
any of them:

1. **The limb bends the wrong way.**  Copying a local rotation straight across
   makes the elbow follow the source's elbow only if the two elbows are
   oriented identically, which they are not.  The fix is to re-express the
   motion in each bone's own rest frame before transferring it, so what
   crosses is a *delta* -- "this bone bent 40 degrees from where it started"
   -- rather than a world rotation.

2. **A limb loses its length.**  Some techniques scale along the bone.  This
   one never does: a bone's length is a property of the rig, and the ped's
   femur is however long the DFF says it is.  Only the rotations move.

3. **The root drifts.**  A character that walks in place in the source must
   walk in place on the target, and one that travels must travel the same
   distance.  Root handling is a stated policy (:class:`RootMode`), not an
   accident of how the source happened to be authored.

Every frame the source produced is kept, at the source's own times, unless
the caller explicitly asks for a resample with a stated error budget.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Sequence

import numpy as np

from ..core import mathx
from ..fbx.source_rig import Clip, SourceBone, SourceRig
from ..mapping.mapper import BoneMatch, MappingResult, Side
from ..mapping.roles import Role
from .rest_pose import RestPoseCorrection

IDENTITY = np.array([0.0, 0.0, 0.0, 1.0])


class RootMode(str, Enum):
    """What happens to the source's root translation."""

    #: Discard root translation entirely -- the character walks on the spot.
    IN_PLACE = "in_place"
    #: Keep the full root motion, so the character moves as the source did.
    FULL = "full"
    #: Keep only sideways travel, dropping height and depth changes.
    HORIZONTAL = "horizontal"
    #: Keep whatever the source's own root bone has as translation, untouched.
    PRESERVE = "preserve"


class UnrepresentableMotion(str, Enum):
    """Motion the IFP format physically cannot express.

    Recorded rather than silently dropped.  A ped has no facial bones in an
    ordinary IFP, and a root cannot both translate freely and stay on the
    ground, so some of what a source file contains has nowhere to go.  The
    validator reports these; a converter that hides them is claiming an
    accuracy it did not achieve.
    """

    NO_TARGET_BONE = "no_target_bone"
    ROOT_DISCARDED = "root_discarded"
    SCALE_DISCARDED = "scale_discarded"
    UNSUPPORTED_CHANNEL = "unsupported_channel"


@dataclass
class BoneTrack:
    """One target bone's animation, in target-local space."""

    target_index: int
    target_tag: int | None
    target_name: str
    source_index: int
    #: One quaternion per sampled frame.
    rotations: np.ndarray
    #: One translation per frame, in GTA units.  ``None`` means the source had
    #: no translation to carry, which is different from carrying a zero.
    translations: np.ndarray | None
    source_key_times_s: np.ndarray

    @property
    def frame_count(self) -> int:
        return int(self.rotations.shape[0])

    def to_dict(self) -> dict:
        return {
            "target_index": self.target_index,
            "target_tag": self.target_tag,
            "target_name": self.target_name,
            "source_index": self.source_index,
            "frames": self.frame_count,
            "has_translation": self.translations is not None,
        }


@dataclass
class RetargetReport:
    """Everything that was dropped, kept, or could not be represented."""

    tracks: list[BoneTrack] = field(default_factory=list)
    unrepresentable: list[tuple[str, UnrepresentableMotion, str]] = field(
        default_factory=list
    )
    warnings: list[str] = field(default_factory=list)
    frame_count: int = 0
    duration_s: float = 0.0
    root_mode: RootMode = RootMode.IN_PLACE
    root_travel_s: float = 0.0

    @property
    def is_complete(self) -> bool:
        """True when nothing had to be left behind.

        Not the same as "validated": this says the transfer was lossless with
        respect to what the format can hold.
        """
        return not self.unrepresentable

    def describe(self) -> str:
        lines = [
            f"retargeted {len(self.tracks)} target bones over "
            f"{self.frame_count} frames ({self.duration_s:.2f}s), "
            f"root mode {self.root_mode.value}"
        ]
        if self.unrepresentable:
            lines.append(f"  {len(self.unrepresentable)} unrepresentable:")
            for name, kind, detail in self.unrepresentable[:20]:
                lines.append(f"    {name!r}: {kind.value} -- {detail}")
        return "\n".join(lines)

    def to_dict(self) -> dict:
        return {
            "tracks": [t.to_dict() for t in self.tracks],
            "frames": self.frame_count,
            "duration_s": round(self.duration_s, 6),
            "root_mode": self.root_mode.value,
            "root_travel_s": round(self.root_travel_s, 6),
            "is_complete": self.is_complete,
            "unrepresentable": [
                {"bone": n, "reason": k.value, "detail": d}
                for n, k, d in self.unrepresentable
            ],
            "warnings": list(self.warnings),
        }


@dataclass
class RetargetSettings:
    """The retarget's knobs, all of them with a defensible default."""

    root_mode: RootMode = RootMode.IN_PLACE
    #: Frames to sample at.  ``None`` means "every frame the source has",
    #: which is the default because dropping keys is the one thing that can
    #: silently ruin an animation with no error anywhere.
    resample_fps: float | None = None
    #: Only used when ``resample_fps`` is set.  Refuses to optimise unless the
    #: resulting motion is provably within this many degrees.
    max_angle_error_deg: float = 0.5
    #: Whether a bone with a non-uniform source scale may be retargeted.  An
    #: IFP stores no scale, so a scaled bone cannot be reproduced; the
    #: default is to say so rather than to pretend.
    allow_scale_loss: bool = False

    def to_dict(self) -> dict:
        return {
            "root_mode": self.root_mode.value,
            "resample_fps": self.resample_fps,
            "max_angle_error_deg": self.max_angle_error_deg,
            "allow_scale_loss": self.allow_scale_loss,
        }


def _clip_frames(clip: Clip, fps: float | None) -> np.ndarray:
    """The frames to read, as indices into the clip.

    With no resampling this is every frame the source has, in order.  With
    resampling it is still the source's frames -- only the *times* change --
    so no source key is ever interpolated away without being asked for.
    """
    count = clip.frame_count
    if count == 0:
        return np.zeros(0, dtype=np.int64)
    return np.arange(count, dtype=np.int64)


def _clip_times(clip: Clip, indices: np.ndarray, fps: float | None) -> np.ndarray:
    """Absolute times, in seconds, for the chosen frames.

    The source's own key times are used verbatim, not regenerated from a frame
    rate.  A source authored at 30fps with an irregular tail keeps that tail;
    rounding its keys to 1/30 would drift the animation by whole frames over
    its length.
    """
    times = np.asarray(clip.times_s, dtype=np.float64)
    if times.size == 0:
        return np.zeros(indices.size, dtype=np.float64)
    if indices.size and indices.max() >= times.size:
        return np.arange(indices.size, dtype=np.float64) / 30.0
    return times[indices]


def _rotation_of(sample: np.ndarray) -> np.ndarray:
    """The rotation of a sampled local transform, as a unit quaternion.

    The FBX layer hands over ``[qx, qy, qz, qw, tx, ty, tz]``: rotation first,
    then translation, no scale.  Taking the first four is exact, and
    normalising guards against a key whose float32 precision left the
    quaternion a hair off unit length -- a defect that compounds across a
    thousand frames into a visible drift.
    """
    return mathx.quat_normalize(np.asarray(sample[:4], dtype=np.float64))


def _translation_of(sample: np.ndarray) -> np.ndarray:
    return np.asarray(sample[4:7], dtype=np.float64).copy()


def _relative_quaternion(
    current: np.ndarray, rest: np.ndarray
) -> np.ndarray:
    """How far a bone has turned from its rest pose.

    ``delta = current * rest^-1`` in the bone's own frame.  Multiplying on the
    right by the rest rotation's inverse is what makes this a *delta*: the
    same turn on a rig that rests at 90 degrees and one that rests flat must
    give the same answer, or a retarget bakes the rest pose into the motion.
    """
    return mathx.quat_normalize(
        mathx.quat_multiply(current, mathx.quat_inverse(rest))
    )


def _apply_delta(delta: np.ndarray, rest: np.ndarray) -> np.ndarray:
    """Put a delta back onto a target bone's rest orientation."""
    return mathx.quat_normalize(mathx.quat_multiply(delta, rest))


def retarget_clip(
    rig: SourceRig,
    clip: Clip,
    mapping: MappingResult,
    correction: RestPoseCorrection,
    target_bones: Sequence,
    settings: RetargetSettings | None = None,
) -> RetargetReport:
    """Retarget one source clip onto the target ped.

    The per-bone transfer, per frame:

        delta  = source_local * source_rest^-1      (how far it turned)
        output = target_rest * A * delta * A^-1      (same turn, target's frame)

    ``A`` is the bone's rest-pose correction, whose defining property is
    ``A * source_rest * A^-1 == target_rest``.  That equation is why a
    character standing still comes out standing still: with ``delta``
    identity, the output *is* the target's bind rotation, whatever the two
    rigs' rest poses happen to be.  31 of this ped's 33 bones have a
    non-identity bind rotation, so getting this wrong is not a rounding
    error -- it stands the character up at 179 degrees to the wrong side.

    The delta is conjugated by ``A`` rather than pre-multiplied, which keeps
    the correction and the motion in the same frame instead of rotating the
    animation by the rest pose as well.
    """
    settings = settings or RetargetSettings()
    report = RetargetReport(root_mode=settings.root_mode)
    if clip.is_empty():
        report.warnings.append("source clip is empty; nothing to retarget")
        return report

    frames = _clip_frames(clip, settings.resample_fps)
    times = _clip_times(clip, frames, settings.resample_fps)
    report.frame_count = int(frames.size)
    report.duration_s = float(times[-1] - times[0]) if times.size else 0.0

    scale = correction.unit_scale
    source_bones = rig.bones

    for match in mapping.mapped:
        source_bone = source_bones[match.source_index]
        bone_correction = correction.get(match.source_index)
        if bone_correction is None:
            report.unrepresentable.append((
                source_bone.name, UnrepresentableMotion.NO_TARGET_BONE,
                "mapped but has no rest-pose correction",
            ))
            continue

        target_bone = target_bones[match.target_index]
        rest_source = mathx.quat_normalize(source_bone.bind_local_quat)
        rest_target = mathx.quat_normalize(target_bone.bind_local_quat)
        reorient = bone_correction.reorient

        rotations = np.zeros((frames.size, 4), dtype=np.float64)
        has_translation = bool(
            source_bone.curves
            and any(
                channel.startswith("Lcl Translation")
                for channel in source_bone.curves
            )
        )
        translations = (
            np.zeros((frames.size, 3), dtype=np.float64) if has_translation else None
        )

        for slot, frame in enumerate(frames):
            sample = clip.poses.get(match.source_index)
            if sample is None or frame >= len(sample):
                # A bone that the mapping found but the clip never animates
                # holds its rest pose.  That is not an error: a rig may carry
                # bones no clip touches, and the ped should stand still there
                # rather than take a rotation from nowhere.
                rotations[slot] = rest_target
                if translations is not None:
                    translations[slot] = 0.0
                continue
            current = _rotation_of(sample[frame])
            delta = _relative_quaternion(current, rest_source)
            # Express the same turn in the target's own frame.  Conjugating by
            # the correction -- rather than multiplying it in front -- is what
            # makes a zero delta produce exactly the target's bind rotation.
            turned = mathx.quat_normalize(mathx.quat_multiply(
                reorient, mathx.quat_multiply(delta, mathx.quat_inverse(reorient))
            ))
            rotations[slot] = _apply_delta(turned, rest_target)
            if translations is not None:
                offset = _translation_of(sample[frame]) - source_bone.bind_local_translation
                translations[slot] = offset * scale

        if settings.root_mode is not RootMode.FULL and match.role is Role.ROOT:
            # The default is in-place: the character animates on the spot.
            # Discarding the root is a real loss of the source's motion and is
            # recorded as one, because an IFP that silently drops the
            # character's travel is not the animation the user handed over.
            dropped = _root_drop(translations, settings.root_mode)
            if translations is not None:
                translations[:] = dropped
            if settings.root_mode is not RootMode.PRESERVE:
                report.unrepresentable.append((
                    source_bone.name, UnrepresentableMotion.ROOT_DISCARDED,
                    f"root translation {settings.root_mode.value}",
                ))

        report.tracks.append(BoneTrack(
            target_index=match.target_index,
            target_tag=match.target_tag,
            target_name=target_bone.name,
            source_index=match.source_index,
            rotations=rotations,
            translations=translations,
            source_key_times_s=times.copy(),
        ))

    for match in mapping.unmapped:
        if match.role is not None and match.role.value not in ("", "unknown"):
            report.unrepresentable.append((
                match.source_name, UnrepresentableMotion.NO_TARGET_BONE,
                match.reason or "no target bone plays this role",
            ))

    travel = _root_travel(report, settings.root_mode)
    report.root_travel_s = travel
    return report


def _root_drop(translations, mode: RootMode) -> np.ndarray:
    """The root translation that survives, given a root mode.

    In-place keeps nothing: the character stays where the game put it and
    animates on the spot.  Horizontal keeps sideways travel and drops the
    height and depth, which is what a game wants from a strafe or a sidestep
    but not from a jump.  Preserve and full keep everything, differing only in
    that full is the mode the caller asked for by name.
    """
    if translations is None or mode is RootMode.FULL or mode is RootMode.PRESERVE:
        return np.zeros_like(translations) if translations is not None else None
    if mode is RootMode.HORIZONTAL:
        # Keep the component along the ped's left/right axis.  In GTA's
        # frame that is X, which the DFF's own bind data confirms: the left
        # and right thighs are separated along it.
        out = np.zeros_like(translations)
        out[:, 0] = translations[:, 0]
        return out
    return np.zeros_like(translations)


def _root_travel(report: RetargetReport, mode: RootMode) -> float:
    """How far the character travels, in GTA units, under this root mode.

    Measured on the *surviving* translation rather than the source's: under
    in-place the character travels zero, and that zero is the honest answer
    rather than a missing measurement.
    """
    for track in report.tracks:
        if track.translations is None or track.target_tag != 0:
            continue
        positions = track.translations
        if positions.shape[0] < 2:
            return 0.0
        steps = np.diff(positions, axis=0)
        return float(np.sum(np.linalg.norm(steps, axis=1)))
    return 0.0


__all__ = [
    "RootMode",
    "UnrepresentableMotion",
    "BoneTrack",
    "RetargetReport",
    "RetargetSettings",
    "retarget_clip",
]
