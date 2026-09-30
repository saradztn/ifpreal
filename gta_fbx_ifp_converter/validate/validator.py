"""Round-trip validation: does the file on disk say what we meant to say?

A converter that writes a file and declares success has checked nothing.  The
only meaningful check is to write the IFP, read it back, and measure how far
what came out is from what went in -- per bone, in degrees and in metres, at
the worst key and on average, never as a single reassuring percentage.

The numbers below are real errors, and they are decomposed rather than
summarised away.  A validator that reports "96% accurate" has told the user
nothing they can act on; one that says "the left foot is 3.1 degrees out at
its worst key and the root translation is 4mm out" tells them which bone to
go and look at.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Sequence

import numpy as np

from ..core import mathx
from ..gta.ifp_reader import ParsedIfp
from ..gta.ifp_writer import (
    ROTATION_SCALE,
    TIME_UNITS_PER_SECOND,
    TRANSLATION_SCALE,
    Animation,
)

#: Angular error above which a bone is called out, in degrees.  This is the
#: point at which a difference is visible in motion next to a real
#: animation -- roughly the width of the joint it moves.
NOTABLE_ANGLE_DEG = 0.5
#: Position error above which a bone is called out, in metres.  A ped's bone
#: offsets are centimetres, so this is well past what the format's 1/1024
#: quantisation can introduce.
NOTABLE_POSITION_M = 0.005

#: Quaternion step is 1/4096; the worst possible rounding is half of that, and
#: the angle it can express depends on the component.  This is the bound
#: derived from the format itself rather than a tolerance picked to make
#: tests pass.
MAX_QUANTISATION_STEP = 0.5 / ROTATION_SCALE
MAX_TRANSLATION_STEP = 0.5 / TRANSLATION_SCALE
#: One clock tick.
MAX_TIME_STEP = 0.5 / TIME_UNITS_PER_SECOND


@dataclass
class ErrorStats:
    """Angular or positional error for one bone, over all its keys."""

    bone_id: int
    bone_name: str
    #: Largest single-key error, which is the one a viewer would notice.
    max_error: float
    mean_error: float
    #: Root-mean-square error, which reflects the *typical* error rather than
    #: letting a single outlier describe the whole bone.
    rms_error: float
    samples: int
    #: Index of the worst key, so the error can be looked at rather than
    #: merely believed.
    worst_frame: int = 0

    def to_dict(self) -> dict:
        return {
            "bone_id": self.bone_id,
            "bone_name": self.bone_name,
            "max": round(self.max_error, 6),
            "mean": round(self.mean_error, 6),
            "rms": round(self.rms_error, 6),
            "samples": self.samples,
            "worst_frame": self.worst_frame,
        }


@dataclass
class ValidationResult:
    """The whole verdict, with everything that led to it kept."""

    #: False means the output must not be used.
    passed: bool = False
    #: The headline the CLI and the GUI both print.  The exact string
    #: ``FAILED VALIDATION`` is required to appear when a run fails, so that
    #: a user who only reads one line still learns the file is not good.
    headline: str = ""
    errors: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    rotation: list[ErrorStats] = field(default_factory=list)
    position: list[ErrorStats] = field(default_factory=list)
    #: Structural problems found by re-reading the file.
    structural: list[str] = field(default_factory=list)
    #: Time error per bone, in seconds.
    time_error: float = 0.0
    bones_checked: int = 0
    keys_checked: int = 0

    @property
    def max_angle_deg(self) -> float:
        return max((s.max_error for s in self.rotation), default=0.0)

    @property
    def mean_angle_deg(self) -> float:
        if not self.rotation:
            return 0.0
        return float(np.mean([s.mean_error for s in self.rotation]))

    @property
    def rms_angle_deg(self) -> float:
        if not self.rotation:
            return 0.0
        weights = [s.samples for s in self.rotation]
        values = [s.rms_error * s.rms_error for s in self.rotation]
        return float(np.sqrt(np.average(values, weights=weights)))

    @property
    def max_position_m(self) -> float:
        return max((s.max_error for s in self.position), default=0.0)

    def worst_bones(self, count: int = 5) -> list[ErrorStats]:
        return sorted(self.rotation, key=lambda s: -s.max_error)[:count]

    def describe(self) -> str:
        lines = [self.headline or ("PASSED VALIDATION" if self.passed else "FAILED VALIDATION")]
        lines.append(
            f"  {self.bones_checked} bones, {self.keys_checked} keys; "
            f"angle max {self.max_angle_deg:.4f} deg, mean "
            f"{self.mean_angle_deg:.4f}, rms {self.rms_angle_deg:.4f}"
        )
        if self.position:
            lines.append(
                f"  position max {self.max_position_m * 100:.2f} mm over "
                f"{len(self.position)} bones"
            )
        if self.time_error > 0:
            lines.append(f"  time error up to {self.time_error * 1000:.1f} ms")
        for stats in self.worst_bones(5):
            lines.append(
                f"    [{stats.bone_id:3}] {stats.bone_name!r:22} "
                f"max {stats.max_error:.4f} deg at key {stats.worst_frame} "
                f"(mean {stats.mean_error:.4f}, rms {stats.rms_error:.4f})"
            )
        for message in self.errors:
            lines.append(f"  ERROR: {message}")
        for message in self.warnings[:12]:
            lines.append(f"  warn: {message}")
        if len(self.warnings) > 12:
            lines.append(f"  ... and {len(self.warnings) - 12} more warnings")
        return "\n".join(lines)

    def to_dict(self) -> dict:
        return {
            "passed": self.passed,
            "headline": self.headline,
            "bones_checked": self.bones_checked,
            "keys_checked": self.keys_checked,
            "max_angle_deg": round(self.max_angle_deg, 6),
            "mean_angle_deg": round(self.mean_angle_deg, 6),
            "rms_angle_deg": round(self.rms_angle_deg, 6),
            "max_position_m": round(self.max_position_m, 6),
            "time_error_s": round(self.time_error, 6),
            "rotation": [s.to_dict() for s in self.rotation],
            "position": [s.to_dict() for s in self.position],
            "errors": list(self.errors),
            "warnings": list(self.warnings),
            "structural": list(self.structural),
        }


def _stats_for(
    bone_id: int,
    bone_name: str,
    errors: np.ndarray,
) -> ErrorStats | None:
    if errors.size == 0:
        return None
    worst = int(np.argmax(errors))
    return ErrorStats(
        bone_id=bone_id,
        bone_name=bone_name,
        max_error=float(errors[worst]),
        mean_error=float(np.mean(errors)),
        rms_error=float(np.sqrt(np.mean(np.square(errors)))),
        samples=int(errors.size),
        worst_frame=worst,
    )


def validate_round_trip(
    intended: Animation,
    parsed: ParsedIfp,
    animation_name: str | None = None,
) -> ValidationResult:
    """Compare what was written against what was meant, key by key.

    The reference is the *fitted* animation -- the same object the writer was
    given -- not the retarget's pre-fit keys.  That distinction is the whole
    point: once the format's 1/50 s clock has merged 592 of 1149 keys, the
    file's key *i* is not the source's key *i*, and comparing them positionally
    measures nothing.  Measuring against the fitted data instead isolates the
    error the format itself introduces, which is the only part a user of this
    tool can act on.

    Three errors are measured, separately, because they fail differently:

    * **angular**, in degrees, from the quaternion difference -- the error
      that shows up as a limb pointing the wrong way;
    * **positional**, in metres, from the root translation -- the error that
      shows up as the character not standing where it should;
    * **temporal**, in seconds, from the key times -- the error that shows up
      as the animation running fast or slow.

    Bones present in the intent but missing from the file are an error, not a
    note.  A bone that silently did not get written produces a file that
    loads and animates a character missing an arm.
    """
    result = ValidationResult()
    found = (
        parsed.animation(animation_name) if animation_name else
        (parsed.animations[0] if parsed.animations else None)
    )
    if found is None:
        result.errors.append(
            f"the file has no animation named {animation_name!r}"
        )
        result.headline = "FAILED VALIDATION"
        return result

    written = {obj.name: obj for obj in found.objects}
    seen_tags: dict[int, str] = {}

    for track in intended.bones:
        result.keys_checked += track.frame_count
        if track.frame_count == 0:
            continue
        obj = written.get(track.name)
        if obj is None:
            result.errors.append(
                f"bone {track.name!r} (tag {track.bone_id}) was "
                f"retargeted but is not in the file"
            )
            continue
        result.bones_checked += 1

        if obj.bone_id in seen_tags and seen_tags[obj.bone_id] != track.name:
            result.errors.append(
                f"HAnim id {obj.bone_id} is claimed by both "
                f"{seen_tags[obj.bone_id]!r} and {track.name!r}; the game "
                f"will animate only one of them"
            )
        seen_tags[obj.bone_id] = track.name

        if obj.frame_count != track.frame_count:
            result.errors.append(
                f"bone {track.name!r}: the writer was given {track.frame_count} "
                f"keys and the file holds {obj.frame_count}"
            )

        # -- angular ------------------------------------------------------- #
        got = obj.rotation_array()
        count = min(got.shape[0], track.rotations.shape[0])
        if count:
            errors = np.array([
                mathx.quat_angle_deg(got[i], track.rotations[i])
                for i in range(count)
            ])
            stats = _stats_for(track.bone_id, track.name, errors)
            if stats is not None:
                result.rotation.append(stats)
            if stats is not None and stats.max_error > NOTABLE_ANGLE_DEG:
                result.warnings.append(
                    f"{track.name!r} is {stats.max_error:.3f} degrees out at "
                    f"key {stats.worst_frame}, against a format bound of "
                    f"{math.degrees(MAX_QUANTISATION_STEP):.4f} degrees; "
                    f"something other than quantisation is losing this bone"
                )

        # -- positional ----------------------------------------------------- #
        if track.translations is not None and obj.translation_array() is not None:
            got_t = obj.translation_array()
            n = min(got_t.shape[0], track.translations.shape[0])
            if n:
                deltas = np.linalg.norm(
                    got_t[:n] - track.translations[:n], axis=1
                )
                stats = _stats_for(track.bone_id, track.name, deltas)
                if stats is not None:
                    result.position.append(stats)
                    if stats.max_error > NOTABLE_POSITION_M:
                        result.warnings.append(
                            f"{track.name!r} position is "
                            f"{stats.max_error * 100:.1f} mm out at key "
                            f"{stats.worst_frame}, against a format bound of "
                            f"{MAX_TRANSLATION_STEP * 1000:.2f} mm"
                        )

        # -- temporal ------------------------------------------------------- #
        if obj.frames:
            got_times = obj.time_array()
            want = np.asarray(track.times_s, dtype=np.float64)
            n = min(got_times.size, want.size)
            if n:
                result.time_error = max(
                    result.time_error, float(np.max(np.abs(got_times[:n] - want[:n])))
                )

    # -- bones in the file that nothing asked for ------------------------- #
    wanted_names = {b.name for b in intended.bones}
    wanted_tags = {b.bone_id for b in intended.bones}
    for obj in found.objects:
        if obj.name not in wanted_names:
            result.warnings.append(
                f"the file contains {obj.name!r} (tag {obj.bone_id}), which no "
                f"retargeted bone asked for"
            )
        if obj.bone_id not in wanted_tags:
            result.errors.append(
                f"the file drives HAnim id {obj.bone_id} "
                f"({obj.name!r}), which nothing in the mapping resolved"
            )

    result.structural = list(getattr(parsed, "structural_notes", []))
    result.passed = not result.errors
    result.headline = "PASSED VALIDATION" if result.passed else "FAILED VALIDATION"
    return result


__all__ = [
    "ErrorStats",
    "ValidationResult",
    "validate_round_trip",
    "NOTABLE_ANGLE_DEG",
    "NOTABLE_POSITION_M",
    "MAX_QUANTISATION_STEP",
    "MAX_TRANSLATION_STEP",
    "MAX_TIME_STEP",
]
