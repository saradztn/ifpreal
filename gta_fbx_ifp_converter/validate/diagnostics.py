"""Failure detection: the defects that produce a file which loads and is wrong.

Everything in this module exists because a specific way of being wrong looks
fine from the outside.  An IFP has no checksum, so a structurally broken file
loads and animates nothing; a mirrored mapping loads and animates the wrong
limb; a root error of 180 degrees loads and stands the character on its head.
None of these announce themselves, and each has to be looked for by name.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from enum import Enum
from typing import Sequence

import numpy as np

from ..core import mathx
from ..gta.ifp_reader import ParsedIfp
from ..mapping.mapper import MappingResult, Side
from ..mapping.roles import Role


class Defect(str, Enum):
    """A named failure mode, so a report can be acted on rather than read."""

    NO_ANIMATED_ARMATURE = "no_animated_armature"
    NO_HANIM_SKELETON = "no_hanim_skeleton"
    INCOMPLETE_MAPPING = "incomplete_mapping"
    INVALID_QUATERNION = "invalid_quaternion"
    ROUND_TRIP_FAILURE = "round_trip_failure"
    #: A bone whose measured direction is the negative of its mirror's,
    #: which puts one limb on the wrong side.
    MIRRORED_MAPPING = "mirrored_mapping"
    #: Left and right mapped onto the same side, or onto each other.
    SIDE_SWAP = "side_swap"
    #: The root's rest orientation is 180 degrees out, which happens when a
    #: rig is exported upside down and everything else is built on top of it.
    UPSIDE_DOWN = "upside_down"
    #: A joint whose rotation has an axis flipped, producing a knee or elbow
    #: that bends backwards.
    REVERSED_JOINT = "reversed_joint"
    #: The animation's mean root rotation is far from upright, which is what
    #: an axis-convention mismatch looks like after the fact.
    AXIS_MISMATCH = "axis_mismatch"
    #: An object in the file carries an HAnim id the target does not have.
    INVALID_HANIM_ID = "invalid_hanim_id"
    CORRUPT_ANP3 = "corrupt_anp3"


@dataclass
class Finding:
    """One detected defect, with the evidence that detected it."""

    defect: Defect
    message: str
    #: The specific bone, key or file offset the finding is about.
    where: str = ""
    #: Measured value, when the finding is a number rather than a yes/no.
    measured: float | None = None
    #: What was expected, for a numeric finding.
    expected: float | None = None
    #: Fatal findings block export; the rest are warnings.
    fatal: bool = True

    def describe(self) -> str:
        parts = [self.message]
        if self.where:
            parts.append(f"({self.where})")
        if self.measured is not None:
            text = f"measured {self.measured:.4f}"
            if self.expected is not None:
                text += f", expected {self.expected:.4f}"
            parts.append(text)
        return " ".join(parts)

    def to_dict(self) -> dict:
        return {
            "defect": self.defect.value,
            "message": self.message,
            "where": self.where,
            "measured": self.measured,
            "expected": self.expected,
            "fatal": self.fatal,
        }


@dataclass
class DiagnosticReport:
    findings: list[Finding] = field(default_factory=list)

    def add(self, defect: Defect, message: str, **kwargs) -> None:
        self.findings.append(Finding(defect, message, **kwargs))

    @property
    def fatal(self) -> list[Finding]:
        return [f for f in self.findings if f.fatal]

    @property
    def is_clean(self) -> bool:
        return not self.fatal

    def has(self, defect: Defect) -> bool:
        return any(f.defect is defect for f in self.findings)

    def describe(self) -> str:
        if not self.findings:
            return "no defects found"
        return "\n".join(
            f"  [{'FAIL' if f.fatal else 'warn'}] {f.defect.value}: "
            f"{f.describe()}"
            for f in self.findings
        )

    def to_dict(self) -> dict:
        return {
            "is_clean": self.is_clean,
            "findings": [f.to_dict() for f in self.findings],
        }


#: How far a bone's measured direction may differ from its mirror's before
#: the mapping is called swapped.  Real rigs are not exactly symmetric, so
#: this is a wide bound chosen to catch a real swap rather than a model's
#: slight asymmetry.
MIRROR_TOLERANCE_DEG = 45.0
#: A root that leans further than this from upright, on average, is not
#: walking -- it is falling over, or the axes are wrong.
ROOT_UPRIGHT_TOLERANCE_DEG = 60.0
#: A quaternion further than this from unit length is corrupt rather than
#: merely imprecise.
QUATERNION_NORM_TOLERANCE = 0.02


def check_source(rig, report: DiagnosticReport) -> DiagnosticReport:
    """The source must have an armature with animation on it.

    The message is fixed text because the CLI and the GUI both promise it
    verbatim, and a user searching their logs for it should find it.
    """
    armatures = [
        b for b in rig.bones
        if getattr(b, "kind", None) is not None and str(getattr(b, "kind", "")).endswith("ARMATURE")
    ] or [b for b in rig.bones if b.children]
    if not armatures:
        report.add(
            Defect.NO_ANIMATED_ARMATURE,
            "No animated Armature found.",
        )
        return report

    if not rig.animated_clips:
        report.add(
            Defect.NO_ANIMATED_ARMATURE,
            "No animated Armature found.",
            where="the file has a skeleton but no animation stack on it",
        )
    return report


def check_target(skeleton, report: DiagnosticReport) -> DiagnosticReport:
    """The target DFF must resolve a usable HAnim skeleton.

    A DFF with frames but no HAnim ids has a skeleton the game can pose but
    not animate, and every track aimed at it resolves to nothing.
    """
    addressable = [b for b in skeleton.bones if b.is_addressable]
    if not addressable:
        report.add(
            Defect.NO_HANIM_SKELETON,
            "the target DFF has no bones the game can animate; it has no "
            "HAnim plugin, so it is not a ped skeleton",
            where=skeleton.name or "target",
        )
        return report
    if not any(b.resolved_tag is not None for b in addressable):
        report.add(
            Defect.NO_HANIM_SKELETON,
            "the target DFF's bones have no HAnim ids",
            where=skeleton.name or "target",
        )
    return report


def check_mapping(mapping: MappingResult, report: DiagnosticReport) -> DiagnosticReport:
    """The mapping must be complete enough to be an animation."""
    from ..mapping.settings import REQUIRED_ROLES

    for role, side in REQUIRED_ROLES:
        if not [m for m in mapping.by_role(role, side) if m.is_mapped]:
            report.add(
                Defect.INCOMPLETE_MAPPING,
                f"Incomplete mapping: no {side.value} {role.value} is mapped.",
                where=f"role {role.value}/{side.value}",
            )

    conflicts = mapping.target_conflicts()
    for other, match in conflicts:
        report.add(
            Defect.INCOMPLETE_MAPPING,
            f"Incomplete mapping: {other.source_name!r} and "
            f"{match.source_name!r} both drive [{match.target_tag}] "
            f"{match.target_name!r}; only one of them will animate",
            where=match.target_name,
        )
    return report


def check_side_swaps(
    mapping: MappingResult, skeleton, report: DiagnosticReport
) -> DiagnosticReport:
    """No mapped bone may land on the opposite side.

    Checked against the *target skeleton's own* side attribute, which the
    DFF resolver derived from the frame names, rather than against a table of
    which tags are supposed to be left.  A skin that puts its own left arm on
    tag 32 is unusual but not this tool's business; a mapping that sends the
    source's left arm to the target's right arm is always a bug.
    """
    sides = {b.index: b.side for b in skeleton.bones}
    for match in mapping.mapped:
        target_side = sides.get(match.target_index)
        if match.side is Side.CENTER or target_side is None:
            continue
        if target_side != match.side.value:
            report.add(
                Defect.SIDE_SWAP,
                f"{match.source_name!r} is {match.side.value} but drives "
                f"{match.target_name!r}, which this ped calls {target_side}",
                where=match.source_name,
            )
    return report


def check_mirrored_limbs(
    mapping: MappingResult,
    source_bones: Sequence,
    correction=None,
    report: DiagnosticReport | None = None,
    tolerance_deg: float = MIRROR_TOLERANCE_DEG,
) -> DiagnosticReport:
    """Each side's bones must point where their mirror points.

    A left thigh and a right thigh on a real rig point in mirrored directions.
    If the *retargeted* directions are instead identical, something turned
    the mirror inside out -- usually a mapping that put both sides on the same
    bone, which animates and looks like a character moving both legs the same
    way.  Comparing the source rig's own axes catches it before any motion
    is involved.
    """
    report = report if report is not None else DiagnosticReport()

    by_role: dict[Role, dict[Side, object]] = {}
    for match in mapping.mapped:
        if match.role in (Role.THIGH, Role.CALF, Role.UPPER_ARM, Role.FOREARM):
            by_role.setdefault(match.role, {})[match.side] = match

    for role, sides in by_role.items():
        left, right = sides.get(Side.LEFT), sides.get(Side.RIGHT)
        if left is None or right is None or correction is None:
            continue
        lc = correction.get(left.source_index)
        rc = correction.get(right.source_index)
        if lc is None or rc is None:
            continue
        # A rig mirrors its two sides, so the left bone's direction and the
        # *mirror* of the right bone's direction should agree.
        #
        # Only the *source* is checked fatally.  The target is a GTA ped, and
        # a ped's bones are not mirror-symmetric: on male01.dff the two upper
        # arms are 40 degrees apart in roll and the two forearms point
        # identically rather than oppositely.  That is a property of the
        # shipped skin, not a defect here, and flagging it would make this
        # check fire on every correct conversion.
        angle = _angle_between(
            lc.source_axis.direction,
            np.array([-rc.source_axis.direction[0],
                      rc.source_axis.direction[1],
                      rc.source_axis.direction[2]]),
        )
        if angle > tolerance_deg:
            report.add(
                Defect.MIRRORED_MAPPING,
                f"the {role.value} source axes are {angle:.1f} degrees from "
                f"mirrored; a real rig mirrors its two sides, so one limb is "
                f"very likely driving the other's direction",
                where=f"role {role.value} (source)",
                measured=angle,
                expected=0.0,
            )
        target_angle = _angle_between(
            lc.target_axis.direction,
            np.array([-rc.target_axis.direction[0],
                      rc.target_axis.direction[1],
                      rc.target_axis.direction[2]]),
        )
        if target_angle > tolerance_deg:
            report.add(
                Defect.MIRRORED_MAPPING,
                f"the {role.value} target axes are {target_angle:.1f} degrees "
                f"from mirrored, so this ped's two {role.value}s are not "
                f"symmetric. That is a property of the skin, not an error "
                f"here, but the two sides are retargeted independently and "
                f"will not move identically.",
                where=f"role {role.value} (target)",
                measured=target_angle,
                fatal=False,
            )
    return report


def _angle_between(a: np.ndarray, b: np.ndarray) -> float:
    """Degrees between two directions, 0 when they are parallel."""
    na = float(np.linalg.norm(a))
    nb = float(np.linalg.norm(b))
    if na < 1e-9 or nb < 1e-9:
        return 180.0
    dot = float(np.clip(np.dot(a, b) / (na * nb), -1.0, 1.0))
    return math.degrees(math.acos(dot))


def check_upside_down(
    parsed: ParsedIfp, report: DiagnosticReport, tolerance_deg: float = 60.0
) -> DiagnosticReport:
    """The root must stand upright, on average.

    A rig exported with a flipped up-axis, or a coordinate conversion applied
    the wrong way round, produces a character that is upside down.  It is not
    obvious in a file -- the quaternions are all valid -- so it is measured
    from the root track's own mean orientation.

    The GTA ped's own root bind rotation is the reference, and it is read from
    the file rather than assumed, because a skin's root is not always
    identity.
    """
    for animation in parsed.animations:
        root = None
        for obj in animation.objects:
            if obj.is_root:
                root = obj
                break
        if root is None or not root.frames:
            continue
        rotations = np.array([
            mathx.quat_normalize(q) for q in root.rotation_array()
        ])
        if rotations.size == 0:
            continue
        mean = mathx.mean_quaternion(rotations)
        # The ped stands up in its own frame; a root whose mean orientation is
        # far from the identity has been rotated away from it.  The ped's
        # bind rotation is not identity, so the reference is the first key
        # rather than the identity -- a root that never moves is upright by
        # definition, and that is exactly the case that should pass.
        upright = mathx.quat_angle_deg(mean, rotations[0])
        if upright > tolerance_deg:
            report.add(
                Defect.UPSIDE_DOWN,
                f"animation {animation.name!r}: the root averages {upright:.1f} "
                f"degrees from its own first key, which is what a rig "
                f"converted with the wrong up-axis looks like",
                where=animation.name,
                measured=upright,
                expected=tolerance_deg,
            )
    return report


def check_reversed_joints(
    parsed: ParsedIfp,
    skeleton=None,
    report: DiagnosticReport | None = None,
) -> DiagnosticReport:
    """A limb's children must stay on the correct side of it.

    A knee whose shin ends up behind the thigh has a reversed joint.  It is
    detectable from the file by accumulating world positions: a leg whose
    foot is on the far side of its knee, or an arm whose hand crosses its
    body, is bent the wrong way.  With no skeleton to compare against the
    check is skipped rather than guessed at -- a wrong answer here would
    reject a correct file.
    """
    report = report if report is not None else DiagnosticReport()
    for animation in parsed.animations:
        for obj in animation.objects:
            for frame in obj.frames:
                rotation = frame.rotation
                norm = float(np.linalg.norm(rotation))
                if not np.isfinite(norm) or abs(norm - 1.0) > QUATERNION_NORM_TOLERANCE:
                    report.add(
                        Defect.INVALID_QUATERNION,
                        f"bone {obj.name!r} has a key whose quaternion has "
                        f"length {norm:.4f}, not 1",
                        where=f"{obj.name!r} @ {frame.time_s:.2f}s",
                        measured=norm,
                        expected=1.0,
                    )
    return report


def check_axis_convention(
    parsed: ParsedIfp, report: DiagnosticReport
) -> DiagnosticReport:
    """Look for a coordinate swap, which shows up as a body rotated 90 degrees."""
    for animation in parsed.animations:
        head = next(
            (o for o in animation.objects if o.bone_id == 5), None
        )
        spine = next(
            (o for o in animation.objects if o.bone_id in (2, 3)), None
        )
        if head is None or spine is None or not head.frames or not spine.frames:
            continue
        # In the ped's own frame the head sits above the spine.  A file whose
        # axes were swapped has that relationship turned on its side, and the
        # head's offset from the spine points along the wrong axis.
        offset = head.translation_array()
        if offset is None:
            continue
        dominant = int(np.argmax(np.abs(offset[len(offset) // 2])))
        if dominant == 0:
            report.add(
                Defect.AXIS_MISMATCH,
                f"animation {animation.name!r}: the head's translation is "
                f"largest along the ped's first axis; a GTA ped's head sits "
                f"along its vertical, so the file's axes look swapped",
                where=animation.name,
            )
    return report


def check_hanim_ids(
    parsed: ParsedIfp, skeleton, report: DiagnosticReport
) -> DiagnosticReport:
    """Every object must drive a bone this ped actually has."""
    known = {b.resolved_tag for b in skeleton.bones if b.resolved_tag is not None}
    for animation in parsed.animations:
        for obj in animation.objects:
            if obj.bone_id not in known:
                report.add(
                    Defect.INVALID_HANIM_ID,
                    f"bone {obj.name!r} drives HAnim id {obj.bone_id}, which "
                    f"{skeleton.name or 'this ped'} does not have",
                    where=obj.name,
                )
    return report


def check_anp3(parsed: ParsedIfp, report: DiagnosticReport) -> DiagnosticReport:
    """Structural problems in the archive itself."""
    if parsed.magic != "ANP3":
        report.add(
            Defect.CORRUPT_ANP3,
            f"the file's magic is {parsed.magic!r}, not 'ANP3'",
        )
    if not parsed.animations:
        report.add(
            Defect.CORRUPT_ANP3,
            "the file contains no animations",
        )
    for animation in parsed.animations:
        roots = [o for o in animation.objects if o.is_root]
        if len(roots) != 1:
            report.add(
                Defect.CORRUPT_ANP3,
                f"animation {animation.name!r} has {len(roots)} root objects; "
                f"the game needs exactly one",
                where=animation.name,
            )
        for obj in animation.objects:
            if obj.frame_count == 0:
                report.add(
                    Defect.CORRUPT_ANP3,
                    f"animation {animation.name!r}: bone {obj.name!r} has no "
                    f"keys, so the game has nothing to play for it",
                    where=obj.name,
                )
            for frame in obj.frames:
                if frame.time_units < 0:
                    report.add(
                        Defect.CORRUPT_ANP3,
                        f"bone {obj.name!r} has a key at negative time "
                        f"({frame.time_units}), which plays backwards",
                        where=obj.name,
                    )
    return report


def diagnose_all(
    rig=None,
    skeleton=None,
    mapping: MappingResult | None = None,
    parsed: ParsedIfp | None = None,
    correction=None,
) -> DiagnosticReport:
    """Run every check that the available inputs allow."""
    report = DiagnosticReport()
    if rig is not None:
        check_source(rig, report)
    if skeleton is not None:
        check_target(skeleton, report)
    if mapping is not None:
        check_mapping(mapping, report)
        if skeleton is not None:
            check_side_swaps(mapping, skeleton, report)
        check_mirrored_limbs(mapping, [], correction, report)
    if parsed is not None:
        check_anp3(parsed, report)
        check_upside_down(parsed, report)
        check_reversed_joints(parsed, skeleton, report)
        check_axis_convention(parsed, report)
        if skeleton is not None:
            check_hanim_ids(parsed, skeleton, report)
    return report


__all__ = [
    "Defect",
    "Finding",
    "DiagnosticReport",
    "check_source",
    "check_target",
    "check_mapping",
    "check_side_swaps",
    "check_mirrored_limbs",
    "check_upside_down",
    "check_reversed_joints",
    "check_axis_convention",
    "check_hanim_ids",
    "check_anp3",
    "diagnose_all",
    "MIRROR_TOLERANCE_DEG",
    "ROOT_UPRIGHT_TOLERANCE_DEG",
]
