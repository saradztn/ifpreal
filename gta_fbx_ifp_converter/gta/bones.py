"""Authoritative GTA San Andreas HAnim bone tags.

Sources of truth, in order of precedence
----------------------------------------
1. The HAnim plugin inside the **target DFF** the user supplies.  Every IFP
   track this tool writes is resolved through ``DffHAnim.id_of(frame_name)``
   -- never through a hard coded table.  See :mod:`.dff_reader`.
2. :class:`SaBoneTag` below, which reproduces the tag numbering used by
   ``rwfury.sa_bones`` and by MTA:SA's ``CClientIFP::eBoneType``.

A note on the two bone numbering schemes
---------------------------------------
GTA San Andreas exposes *two* different bone enumerations and they are easy
to confuse:

``eBoneType`` (HAnim tag / IFP bone ID)
    ``0`` Root, ``1`` Pelvis, ``2`` Spine, ``3`` Spine1, ``4`` Neck,
    ``5`` Head, ``6`` L Brow, ``7`` R Brow, ``8`` Jaw, ``21`` R Clavicle,
    ``22`` R UpperArm, ... ``41`` L Thigh, ``51`` R Thigh, ``201`` Belly.
    This is the number stored in the third ``int32`` of an ANP3 object
    header and it is what ``engineLoadIFP``/``setPedAnimation`` end up using.

``ePedBones`` (script-facing ``BONE_*`` enum)
    A *different* numbering (e.g. ``4`` there is ``BONE_UPPERTORSO`` while
    ``4`` here is ``Neck``).  It is used by ``getCharBoneCoord``-style
    scripting helpers, not by the IFP format.

Spec documents that quote ``0 ROOT / 1 PELVIS1 / 2 PELVIS / 4 UPPERTORSO``
are quoting the script enum.  Mixing the two up produces animations that load
without error but move the wrong bones, so this module keeps both, exposes
:meth:`SaBoneTag.script_enum_name`, and :mod:`..core.validator` emits an
explicit warning when a mapping was written against the wrong scheme.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import IntEnum
from typing import Iterable, Iterator

__all__ = [
    "SaBoneTag",
    "CUTSCENE_BONE_TAGS",
    "bone_tag_from_name",
    "bone_name_from_tag",
    "is_ped_body_bone",
    "describe_tag",
    "BoneTagConflict",
]


class SaBoneTag(IntEnum):
    """HAnim bone tag written into IFP object headers (the ``eBoneType`` set)."""

    UNKNOWN = -1

    ROOT = 0
    PELVIS = 1
    SPINE = 2
    SPINE1 = 3
    NECK = 4
    HEAD = 5
    L_BROW = 6
    R_BROW = 7
    JAW = 8

    R_CLAVICLE = 21
    R_UPPER_ARM = 22
    R_FOREARM = 23
    R_HAND = 24
    R_FINGER = 25
    R_FINGER_01 = 26

    L_CLAVICLE = 31
    L_UPPER_ARM = 32
    L_FOREARM = 33
    L_HAND = 34
    L_FINGER = 35
    L_FINGER_01 = 36

    L_THIGH = 41
    L_CALF = 42
    L_FOOT = 43
    L_TOE = 44

    R_THIGH = 51
    R_CALF = 52
    R_FOOT = 53
    R_TOE = 54

    BELLY = 201
    R_BREAST = 301
    L_BREAST = 302

    # --- cutscene-only bones, present in player.ifp / cutscene ifps only ------
    R_THUMB_1 = 28
    R_THUMB_2 = 29
    LIP_L_11 = 30
    L_THUMB_1 = 38
    L_THUMB_2 = 39
    JAW_22 = 40
    HEAD_NUB = 303
    L_FINGER_0_NUB = 304
    R_FINGER_0_NUB = 305
    L_TOE_0_NUB = 306
    R_TOE_0_NUB = 307
    R_BROW_1 = 5001
    R_BROW_2 = 5002
    L_BROW_2 = 5003
    L_BROW_1 = 5004
    R_LID = 5005
    L_LID = 5006
    R_TLIP_3 = 5007
    L_TLIP_3 = 5008
    R_TLIP_1 = 5009
    R_TLIP_2 = 5010
    L_TLIP_1 = 5011
    L_TLIP_2 = 5012
    R_CORNER = 5013
    L_CORNER = 5014
    JAW_1 = 5015
    JAW_2 = 5016
    L_LIP_1 = 5017
    R_EYE = 5018
    L_EYE = 5019
    R_CHEEK = 5020
    L_CHEEK = 5021

    @property
    def ifp_name(self) -> str | None:
        """Canonical object name as written in an IFP track header."""
        return _IFP_NAMES.get(self)

    @property
    def dff_frame_name(self) -> str | None:
        """Canonical frame name as found in GTA ped DFF HAnim hierarchies."""
        return _DFF_NAMES.get(self)

    @property
    def side(self) -> str:
        if self.name.startswith("L_") or self in (
            SaBoneTag.L_CLAVICLE, SaBoneTag.L_UPPER_ARM, SaBoneTag.L_FOREARM,
            SaBoneTag.L_HAND, SaBoneTag.L_FINGER, SaBoneTag.L_FINGER_01,
            SaBoneTag.L_THIGH, SaBoneTag.L_CALF, SaBoneTag.L_FOOT, SaBoneTag.L_TOE,
            SaBoneTag.L_BREAST, SaBoneTag.L_BROW,
        ):
            return "left"
        if self.name.startswith("R_") or self in (
            SaBoneTag.R_CLAVICLE, SaBoneTag.R_UPPER_ARM, SaBoneTag.R_FOREARM,
            SaBoneTag.R_HAND, SaBoneTag.R_FINGER, SaBoneTag.R_FINGER_01,
            SaBoneTag.R_THIGH, SaBoneTag.R_CALF, SaBoneTag.R_FOOT, SaBoneTag.R_TOE,
            SaBoneTag.R_BREAST, SaBoneTag.R_BROW,
        ):
            return "right"
        return "center"

    @property
    def script_enum_name(self) -> str:
        """Name this tag has in the *other* (scripting) enumeration.

        Only the handful of tags where the two schemes disagree are listed;
        everything else shares its HAnim name.
        """
        return _SCRIPT_ENUM_NAMES.get(self, self.name)

    @property
    def is_cutscene_only(self) -> bool:
        return self in CUTSCENE_BONE_TAGS


#: Tags that only exist in player/cutscene IFPs, never in plain ped models.
CUTSCENE_BONE_TAGS = frozenset(
    {
        SaBoneTag.R_THUMB_1, SaBoneTag.R_THUMB_2, SaBoneTag.LIP_L_11,
        SaBoneTag.L_THUMB_1, SaBoneTag.L_THUMB_2, SaBoneTag.JAW_22,
        SaBoneTag.HEAD_NUB, SaBoneTag.L_FINGER_0_NUB, SaBoneTag.R_FINGER_0_NUB,
        SaBoneTag.L_TOE_0_NUB, SaBoneTag.R_TOE_0_NUB,
    }
)

_IFP_NAMES: dict[SaBoneTag, str] = {
    SaBoneTag.ROOT: "Normal",
    SaBoneTag.PELVIS: "Pelvis",
    SaBoneTag.SPINE: "Spine",
    SaBoneTag.SPINE1: "Spine1",
    SaBoneTag.NECK: "Neck",
    SaBoneTag.HEAD: "Head",
    SaBoneTag.L_BROW: "L Brow",
    SaBoneTag.R_BROW: "R Brow",
    SaBoneTag.JAW: "Jaw",
    SaBoneTag.R_CLAVICLE: "Bip01 R Clavicle",
    SaBoneTag.R_UPPER_ARM: "R UpperArm",
    SaBoneTag.R_FOREARM: "R Forearm",
    SaBoneTag.R_HAND: "R Hand",
    SaBoneTag.R_FINGER: "R Finger",
    SaBoneTag.R_FINGER_01: "R Finger01",
    SaBoneTag.L_CLAVICLE: "Bip01 L Clavicle",
    SaBoneTag.L_UPPER_ARM: "L UpperArm",
    SaBoneTag.L_FOREARM: "L Forearm",
    SaBoneTag.L_HAND: "L Hand",
    SaBoneTag.L_FINGER: "L Finger",
    SaBoneTag.L_FINGER_01: "L Finger01",
    SaBoneTag.L_THIGH: "L Thigh",
    SaBoneTag.L_CALF: "L Calf",
    SaBoneTag.L_FOOT: "L Foot",
    SaBoneTag.L_TOE: "L Toe0",
    SaBoneTag.R_THIGH: "R Thigh",
    SaBoneTag.R_CALF: "R Calf",
    SaBoneTag.R_FOOT: "R Foot",
    SaBoneTag.R_TOE: "R Toe0",
    SaBoneTag.BELLY: "Belly",
    SaBoneTag.R_BREAST: "R Breast",
    SaBoneTag.L_BREAST: "L Breast",
}

_DFF_NAMES: dict[SaBoneTag, str] = {
    SaBoneTag.ROOT: "Root",
    SaBoneTag.PELVIS: "Pelvis",
    SaBoneTag.SPINE: "Bip01 Spine",
    SaBoneTag.SPINE1: "Bip01 Spine1",
    SaBoneTag.NECK: "Bip01 Neck",
    SaBoneTag.HEAD: "Bip01 Head",
    SaBoneTag.L_BROW: "Bip01 L Eyelidup",
    SaBoneTag.R_BROW: "Bip01 R Eyelidup",
    SaBoneTag.JAW: "Bip01 Jaw",
    SaBoneTag.R_CLAVICLE: "Bip01 R Clavicle",
    SaBoneTag.R_UPPER_ARM: "Bip01 R UpperArm",
    SaBoneTag.R_FOREARM: "Bip01 R Forearm",
    SaBoneTag.R_HAND: "Bip01 R Hand",
    SaBoneTag.R_FINGER: "Bip01 R Finger",
    SaBoneTag.R_FINGER_01: "Bip01 R Finger01",
    SaBoneTag.L_CLAVICLE: "Bip01 L Clavicle",
    SaBoneTag.L_UPPER_ARM: "Bip01 L UpperArm",
    SaBoneTag.L_FOREARM: "Bip01 L Forearm",
    SaBoneTag.L_HAND: "Bip01 L Hand",
    SaBoneTag.L_FINGER: "Bip01 L Finger",
    SaBoneTag.L_FINGER_01: "Bip01 L Finger01",
    SaBoneTag.L_THIGH: "Bip01 L Thigh",
    SaBoneTag.L_CALF: "Bip01 L Calf",
    SaBoneTag.L_FOOT: "Bip01 L Foot",
    SaBoneTag.L_TOE: "Bip01 L Toe0",
    SaBoneTag.R_THIGH: "Bip01 R Thigh",
    SaBoneTag.R_CALF: "Bip01 R Calf",
    SaBoneTag.R_FOOT: "Bip01 R Foot",
    SaBoneTag.R_TOE: "Bip01 R Toe0",
    SaBoneTag.BELLY: "Belly",
    SaBoneTag.R_BREAST: "Bip01 R Breast",
    SaBoneTag.L_BREAST: "Bip01 L Breast",
}

#: Only populated where the script enum and the HAnim enum actually differ.
_SCRIPT_ENUM_NAMES: dict[SaBoneTag, str] = {
    SaBoneTag.PELVIS: "PELVIS1",
    SaBoneTag.SPINE: "PELVIS",
    SaBoneTag.SPINE1: "SPINE1",
    SaBoneTag.NECK: "UPPERTORSO",
    SaBoneTag.HEAD: "HEAD",
    SaBoneTag.L_BROW: "HEAD2",
    SaBoneTag.R_BROW: "HEAD1",
    SaBoneTag.JAW: "HEAD3",
    SaBoneTag.R_CLAVICLE: "RIGHTUPPERTORSO",
    SaBoneTag.R_UPPER_ARM: "RIGHTSHOULDER",
    SaBoneTag.R_FOREARM: "RIGHTELBOW",
    SaBoneTag.R_HAND: "RIGHTWRIST",
    SaBoneTag.R_FINGER: "RIGHTHAND",
    SaBoneTag.R_FINGER_01: "RIGHTTHUMB",
    SaBoneTag.L_CLAVICLE: "LEFTUPPERTORSO",
    SaBoneTag.L_UPPER_ARM: "LEFTSHOULDER",
    SaBoneTag.L_FOREARM: "LEFTELBOW",
    SaBoneTag.L_HAND: "LEFTWRIST",
    SaBoneTag.L_FINGER: "LEFTHAND",
    SaBoneTag.L_FINGER_01: "LEFTTHUMB",
    SaBoneTag.L_THIGH: "LEFTHIP",
    SaBoneTag.L_CALF: "LEFTKNEE",
    SaBoneTag.L_FOOT: "LEFTANKLE",
    SaBoneTag.L_TOE: "LEFTFOOT",
    SaBoneTag.R_THIGH: "RIGHTHIP",
    SaBoneTag.R_CALF: "RIGHTKNEE",
    SaBoneTag.R_FOOT: "RIGHTANKLE",
    SaBoneTag.R_TOE: "RIGHTFOOT",
}


def _normalize(name: str) -> str:
    return "".join(ch for ch in name.casefold() if ch.isalnum())


_NAME_INDEX: dict[str, SaBoneTag] = {}
for _tag, _ifp in _IFP_NAMES.items():
    _NAME_INDEX.setdefault(_normalize(_ifp), _tag)
for _tag, _dff in _DFF_NAMES.items():
    _NAME_INDEX.setdefault(_normalize(_dff), _tag)

# Explicit aliases seen in real-world ifps and DFFs.
_NAME_INDEX.update({
    _normalize("root"): SaBoneTag.ROOT,
    _normalize("normal"): SaBoneTag.ROOT,
    _normalize("bip01"): SaBoneTag.PELVIS,
    _normalize("bip01ltoe"): SaBoneTag.L_TOE,
    _normalize("bip01rtoe"): SaBoneTag.R_TOE,
    _normalize("ltoe0"): SaBoneTag.L_TOE,
    _normalize("rtoe0"): SaBoneTag.R_TOE,
    _normalize("lfingers"): SaBoneTag.L_FINGER,
    _normalize("rfingers"): SaBoneTag.R_FINGER,
    _normalize("lthigh"): SaBoneTag.L_THIGH,
    _normalize("rthigh"): SaBoneTag.R_THIGH,
    _normalize("lbreast"): SaBoneTag.L_BREAST,
    _normalize("rbreast"): SaBoneTag.R_BREAST,
    _normalize("belly"): SaBoneTag.BELLY,
    _normalize("bip01head"): SaBoneTag.HEAD,
    _normalize("bip01jaw"): SaBoneTag.JAW,
    _normalize("bip01leylelidup"): SaBoneTag.L_BROW,
    _normalize("bip01reylelidup"): SaBoneTag.R_BROW,
    _normalize("bip01leyelidup"): SaBoneTag.L_BROW,
    _normalize("bip01reyelidup"): SaBoneTag.R_BROW,
    _normalize("bip01spine"): SaBoneTag.SPINE,
    _normalize("bip01spine1"): SaBoneTag.SPINE1,
    _normalize("bip01lclavicle"): SaBoneTag.L_CLAVICLE,
    _normalize("bip01rclavicle"): SaBoneTag.R_CLAVICLE,
})

# Aliases for cutscene face bones, keyed by the names MTA:SA writes.
for _tag, _mta in {
    SaBoneTag.R_THUMB_1: "RThumb1", SaBoneTag.R_THUMB_2: "RThumb2",
    SaBoneTag.LIP_L_11: "llip11", SaBoneTag.L_THUMB_1: "LThumb1",
    SaBoneTag.L_THUMB_2: "LThumb2", SaBoneTag.JAW_22: "jaw22",
    SaBoneTag.HEAD_NUB: "HeadNub", SaBoneTag.L_FINGER_0_NUB: "L Finger0Nub",
    SaBoneTag.R_FINGER_0_NUB: "R Finger0Nub", SaBoneTag.L_TOE_0_NUB: "L Toe0Nub",
    SaBoneTag.R_TOE_0_NUB: "R Toe0Nub", SaBoneTag.R_BROW_1: "rbrow1",
    SaBoneTag.R_BROW_2: "rbrow2", SaBoneTag.L_BROW_2: "lbrow2",
    SaBoneTag.L_BROW_1: "lbrow1", SaBoneTag.R_LID: "rlid",
    SaBoneTag.L_LID: "llid", SaBoneTag.R_TLIP_3: "rtlip3",
    SaBoneTag.L_TLIP_3: "ltlip3", SaBoneTag.R_TLIP_1: "rtlip1",
    SaBoneTag.R_TLIP_2: "rtlip2", SaBoneTag.L_TLIP_1: "ltlip1",
    SaBoneTag.L_TLIP_2: "ltlip2", SaBoneTag.R_CORNER: "rcorner",
    SaBoneTag.L_CORNER: "lcorner", SaBoneTag.JAW_1: "jaw1",
    SaBoneTag.JAW_2: "jaw2", SaBoneTag.L_LIP_1: "llip1",
    SaBoneTag.R_EYE: "reye", SaBoneTag.L_EYE: "leye",
    SaBoneTag.R_CHEEK: "rcheek", SaBoneTag.L_CHEEK: "lcheek",
}.items():
    _NAME_INDEX.setdefault(_normalize(_mta), _tag)


class BoneTagConflict(ValueError):
    """Raised when a name resolves to a tag but contradicts an explicit one."""


def bone_tag_from_name(name: str) -> SaBoneTag | None:
    """Resolve a DFF frame name or IFP object name to its HAnim tag."""
    return _NAME_INDEX.get(_normalize(name))


def bone_name_from_tag(tag: int | SaBoneTag) -> str | None:
    """Canonical IFP object name for a tag."""
    try:
        return _IFP_NAMES[SaBoneTag(tag)]
    except (ValueError, KeyError):
        return None


def is_ped_body_bone(tag: int | SaBoneTag) -> bool:
    """True for tags that a plain ped model (not a cutscene) can animate."""
    try:
        value = SaBoneTag(tag)
    except ValueError:
        return False
    return not value.is_cutscene_only and value in _IFP_NAMES


def describe_tag(tag: int | SaBoneTag) -> str:
    """Human readable ``id name`` string used by reports and the GUI."""
    try:
        value = SaBoneTag(tag)
    except ValueError:
        return f"{int(tag)} <unknown tag>"
    name = value.ifp_name or value.name
    return f"{int(value)} {name}"


def iter_body_tags() -> Iterator[SaBoneTag]:
    for tag in SaBoneTag:
        if is_ped_body_bone(tag):
            yield tag


@dataclass(frozen=True)
class TagCrossCheck:
    """Result of validating a name/tag pair discovered in a DFF."""

    frame_name: str
    dff_tag: int | None
    canonical_tag: int | None
    matches_canonical: bool

    @property
    def message(self) -> str:
        return (
            f"frame {self.frame_name!r}: DFF HAnim id="
            f"{self.dff_tag if self.dff_tag is not None else '?'} vs canonical "
            f"{self.canonical_tag if self.canonical_tag is not None else '?'} "
            f"({'ok' if self.matches_canonical else 'MISMATCH'})"
        )


def cross_check_frame(frame_name: str, dff_tag: int | None) -> TagCrossCheck:
    """Compare a DFF frame's HAnim id against the canonical table."""
    canonical = bone_tag_from_name(frame_name)
    canonical_id = int(canonical) if canonical is not None else None
    return TagCrossCheck(
        frame_name=frame_name,
        dff_tag=dff_tag,
        canonical_tag=canonical_id,
        matches_canonical=dff_tag is None or canonical_id is None or dff_tag == canonical_id,
    )


def tags_to_table(tags: Iterable[int | SaBoneTag]) -> str:
    return "\n".join(describe_tag(t) for t in tags)
