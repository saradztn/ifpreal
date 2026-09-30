"""Turning a retargeted animation into the bytes of an IFP.

The writer in :mod:`.ifp_writer` deals in structures; this module deals in
what the rest of the pipeline produces, and in the two things that go wrong at
this boundary:

* **Bone names.**  An IFP object is found by its *name*, and the name must be
  the one the target DFF actually uses.  A track knows its bone's index, so
  the name comes from the DFF -- never from the source rig, whose naming
  convention has nothing to do with the game's.
* **Which bone is the root.**  Exactly one object per animation is a root
  frame, and it is the one that carries translation.  It is found from the
  target skeleton's own hierarchy rather than assumed to be whichever track
  happened to come first.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Sequence

import numpy as np

from ..retarget.transfer import BoneTrack, RetargetReport
from .ifp_writer import Animation, BoneFrames, IfpWriteError, TimeFitting

#: The ped bones an IFP object may name.  Taken from the target DFF, never
#: from a hard-coded list -- a skin with a bone this project has never heard
#: of still has a tag, and the tag is what the game matches on.


@dataclass
class BuildResult:
    """The animation, plus what could not go into it."""

    animation: Animation | None = None
    skipped: list[tuple[str, str]] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    #: What the format's 1/50 s clock cost, once the animation is built.
    time_fitting: "TimeFitting | None" = None

    @property
    def is_complete(self) -> bool:
        return self.animation is not None and not self.skipped

    def describe(self) -> str:
        if self.animation is None:
            return "nothing to build"
        lines = [
            f"{self.animation.name!r}: {len(self.animation.bones)} objects, "
            f"{self.animation.frame_data_size} bytes of key data"
        ]
        if self.time_fitting is not None:
            lines.append(f"  {self.time_fitting.describe()}")
        for name, reason in self.skipped:
            lines.append(f"  skipped {name!r}: {reason}")
        return "\n".join(lines)


def _root_index(skeleton, tracks: Sequence[BoneTrack]) -> int | None:
    """The track that is the animation's root, by the target's own flag.

    The DFF resolver marks the HAnim root explicitly -- on male01.dff that is
    ``'Root'`` with tag 0, and it is *not* the same bone as the one with no
    parent, because the DFF hangs the whole skeleton off an unnamed frame
    above it.  Inferring the root from the hierarchy picks the unnamed frame,
    which has no HAnim tag, and the animation is then written with no root
    frame at all and does not play.

    Picking "the first track" instead is worse: the first track is usually
    the hips, and a hips translation in a ped animation is a body offset
    rather than the character's position, so the whole animation swings.
    """
    addressed = {t.target_index for t in tracks}
    for bone in skeleton.bones:
        if bone.is_hanim_root and bone.index in addressed:
            return bone.index
    # Some skins do not set the flag.  Fall back to the addressable bone
    # nearest the top of the hierarchy, which is the root by construction.
    candidates = [
        t for t in tracks
        if skeleton.bones[t.target_index].is_addressable
        and (skeleton.bones[t.target_index].parent is None
             or skeleton.bones[t.target_index].parent < 0
             or not skeleton.bones[skeleton.bones[t.target_index].parent].is_addressable)
    ]
    if candidates:
        return min(
            candidates, key=lambda t: skeleton.bones[t.target_index].index
        ).target_index
    return None


def build_animation(
    name: str,
    report: RetargetReport,
    skeleton,
    internal_name: str | None = None,
) -> BuildResult:
    """Assemble one IFP animation from a retarget report.

    The result is deliberately partial-aware: a track whose bone has no usable
    HAnim tag, or whose times do not fit the format, is reported and left out
    rather than written as something the game will misread.
    """
    result = BuildResult()
    tracks = [t for t in report.tracks if t.target_tag is not None]
    if not tracks:
        result.warnings.append(
            "no retargeted track has a usable HAnim tag, so there is nothing "
            "to write; check that the target DFF resolved"
        )
        return result

    root_target = _root_index(skeleton, tracks)
    if root_target is None:
        result.warnings.append(
            "no track drives a root bone of the target skeleton, so the "
            "animation would have no root frame and the game cannot play it"
        )
        return result

    bones: list[BoneFrames] = []
    for track in tracks:
        bone = skeleton.bones[track.target_index]
        is_root = track.target_index == root_target
        try:
            bones.append(BoneFrames(
                # The name must be the DFF's: the game looks the bone up by
                # this string, and a source-rig name would find nothing.
                name=bone.name,
                bone_id=int(track.target_tag),
                rotations=track.rotations,
                times_s=track.source_key_times_s,
                translations=(
                    track.translations if is_root and track.translations is not None
                    else None
                ),
                is_root=is_root,
            ))
        except IfpWriteError as error:
            result.skipped.append((track.target_name, str(error)))

    if not bones:
        result.warnings.append(
            "every track was skipped; the file would have been empty"
        )
        return result

    try:
        result.animation = Animation(
            name=name, bones=bones, internal_name=internal_name
        )
    except IfpWriteError as error:
        result.warnings.append(str(error))
        return result

    # What the format's 1/50 s clock costs this animation, measured rather
    # than assumed.  A 30 fps source has more keys per second than the format
    # can address, so something has to give, and the user is told what rather
    # than being handed a file that quietly runs slow.
    fitting = result.animation.fit()
    result.time_fitting = fitting
    if not fitting.fits_without_loss:
        result.warnings.append(
            f"time keys had to be reduced: {fitting.describe()}. The "
            f"animation keeps its length and its motion; the keys it lost "
            f"were {1 - fitting.stored_keys / max(1, fitting.source_keys):.0%} "
            f"of the source's, inside a {1 / 50:.0f} second slot"
        )
    return result


__all__ = ["BuildResult", "build_animation"]
