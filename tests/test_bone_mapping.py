"""Source-to-target bone mapping, checked against the real rig and real peds.

The mapping's job is to be right about *which GTA bone each source bone
drives*.  Getting that wrong produces an animation that plays, looks
plausible, and animates the wrong limb -- so the tests below assert the exact
target tag for every load-bearing bone rather than a count or a score.
"""

from __future__ import annotations

import os
import sys

import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from gta_fbx_ifp_converter.fbx import source_rig as sr
from gta_fbx_ifp_converter.gta import dff_reader as dff_reader
from gta_fbx_ifp_converter.mapping import map_rig, Role, Side
from gta_fbx_ifp_converter.mapping.aliases import normalize, role_from_name, side_of, load_aliases

FBX = os.path.join(ROOT, "testdata", "samba_dancing.fbx")
DFF = os.path.join(ROOT, "testdata", "male01.dff")

#: The bones that carry the animation, and the GTA tag each must drive.
#: Left and right are listed separately on purpose: a swap is invisible in a
#: frame count and obvious only when a character waves the wrong arm.
EXPECTED = {
    "mixamorig:Hips":         (Role.ROOT,      Side.CENTER, 0),
    "mixamorig:Spine":        (Role.SPINE1,   Side.CENTER, 2),
    "mixamorig:Head":         (Role.HEAD,      Side.CENTER, 5),
    "mixamorig:LeftShoulder": (Role.CLAVICLE,  Side.LEFT,   31),
    "mixamorig:LeftArm":      (Role.UPPER_ARM, Side.LEFT,   32),
    "mixamorig:LeftForeArm":  (Role.FOREARM,   Side.LEFT,   33),
    "mixamorig:LeftHand":     (Role.HAND,      Side.LEFT,   34),
    "mixamorig:RightShoulder":(Role.CLAVICLE,  Side.RIGHT,  21),
    "mixamorig:RightArm":     (Role.UPPER_ARM, Side.RIGHT,  22),
    "mixamorig:RightForeArm": (Role.FOREARM,   Side.RIGHT,  23),
    "mixamorig:RightHand":    (Role.HAND,      Side.RIGHT,  24),
    "mixamorig:LeftUpLeg":    (Role.THIGH,     Side.LEFT,   41),
    "mixamorig:LeftLeg":      (Role.CALF,      Side.LEFT,   42),
    "mixamorig:LeftFoot":     (Role.FOOT,      Side.LEFT,   43),
    "mixamorig:LeftToeBase":  (Role.TOE,       Side.LEFT,   44),
    "mixamorig:RightUpLeg":   (Role.THIGH,     Side.RIGHT,  51),
    "mixamorig:RightLeg":     (Role.CALF,      Side.RIGHT,  52),
    "mixamorig:RightFoot":    (Role.FOOT,      Side.RIGHT,  53),
    "mixamorig:RightToeBase": (Role.TOE,       Side.RIGHT,  54),
}


#: Tags a ped actually animates in an IFP: the body, not the face detail.
_ANIMATABLE_TAGS = {
    0, 1, 2, 3, 4, 5, 21, 31, 22, 32, 23, 33, 24, 34, 25, 35,
    41, 51, 42, 52, 43, 53, 44, 54, 201, 301, 302,
}

#: Ped bones this particular source rig has nothing to drive.  The Mixamo
#: Samba rig has one pelvis, no belly and no breasts, so these four GTA tags
#: cannot be animated from it by any mapping at all.  The converter must leave
#: them at rest and say so, not invent motion for them.  Keyed by tag, not by
#: hierarchy index, because the index is a property of this DFF's frame order
#: and means nothing in another ped.
_BONES_THE_SOURCE_RIG_LACKS = {1, 201, 301, 302}


@pytest.fixture(scope="module")
def mapping():
    skeleton = dff_reader.load_skeleton(DFF)
    rig = sr.load_source_rig(FBX)
    return map_rig(rig.bones, skeleton), skeleton, rig


@pytest.mark.parametrize("source_name,expected", sorted(EXPECTED.items()))
def test_each_bone_drives_the_right_gta_tag(mapping, source_name, expected):
    """Every load-bearing source bone drives its own GTA tag, correct side."""
    result, _skeleton, _rig = mapping
    role, side, tag = expected
    match = next((m for m in result.matches if m.source_name == source_name), None)
    assert match is not None, f"{source_name} missing from the mapping"
    assert match.role is role, f"{source_name}: role {match.role}, want {role}"
    assert match.side is side, f"{source_name}: side {match.side}, want {side}"
    assert match.target_tag == tag, (
        f"{source_name} drives GTA tag {match.target_tag} "
        f"({match.target_name!r}), want {tag}"
    )


def test_no_left_right_swap(mapping):
    """A mapped left bone never lands on a right target, and vice versa.

    Checked over every mapping rather than the table above, so a bone added
    later cannot swap sides without this failing.
    """
    result, skeleton, _rig = mapping
    sides = {b.index: b.side for b in skeleton.bones}
    for match in result.mapped:
        target_side = sides.get(match.target_index)
        if match.side is Side.CENTER or target_side is None:
            continue
        assert target_side == match.side.value, (
            f"{match.source_name!r} ({match.side.value}) drives "
            f"{match.target_name!r} ({target_side})"
        )


def test_no_two_source_bones_fight_over_one_target(mapping):
    result, _skeleton, _rig = mapping
    conflicts = result.target_conflicts()
    assert not conflicts, [
        (a.source_name, b.source_name, a.target_name) for a, b in conflicts
    ]


def test_every_target_bone_is_reachable(mapping):
    """The mapping drives every ped bone the source rig can actually drive.

    A retarget that silently drops the right arm still passes a bone-count
    check; this is the check that notices.
    """
    result, skeleton, _rig = mapping
    addressed = {skeleton.bones[m.target_index].resolved_tag for m in result.mapped}
    expected = {
        b.resolved_tag for b in skeleton.bones
        if b.is_addressable and b.resolved_tag in _ANIMATABLE_TAGS
    } - _BONES_THE_SOURCE_RIG_LACKS
    missing = expected - addressed
    assert not missing, [
        f"no source bone drives tag {tag} "
        f"({next(b.name for b in skeleton.bones if b.resolved_tag == tag)!r})"
        for tag in sorted(missing)
    ]



def test_bones_the_source_cannot_drive_stay_at_rest(mapping):
    """Unreachable ped bones are reported, never faked.

    A missing pelvis or breast must not be filled with a copy of the nearest
    bone's rotation: the ped would breathe where it should not, and the
    exported IFP would look validated while being invented.
    """
    result, skeleton, _rig = mapping
    for tag in _BONES_THE_SOURCE_RIG_LACKS:
        bone = next((b for b in skeleton.bones if b.resolved_tag == tag), None)
        assert bone is not None, f"fixture no longer has a bone with tag {tag}"
        driven = [m for m in result.mapped if m.target_index == bone.index]
        assert not driven, (
            f"[{tag}] {bone.name!r} is driven by a source bone that "
            f"does not exist in this rig"
        )


def test_unmapped_bones_are_reported_not_guessed(mapping):
    """Bones with no target counterpart are listed with a reason.

    The Mixamo rig carries face, eye and fingertip bones the ped has no tags
    for.  They must be visible in the report, never silently dropped.
    """
    result, _skeleton, _rig = mapping
    assert result.unmapped, "fixture no longer exercises unmapped bones"
    for match in result.unmapped:
        assert match.reason, f"{match.source_name!r} unmapped with no reason"
        assert match.confidence == 0.0


def test_confidence_is_real_not_decorative(mapping):
    """A mapped bone scores above an unmapped one, and never reaches 1.0 by
    accident.

    ``check`` blocks export below a threshold, so a name-only match that
    scored 1.0 would let a wrong mapping through.
    """
    result, _skeleton, _rig = mapping
    mapped = [m for m in result.mapped if m.kind.value == "alias"]
    assert mapped
    assert all(0.5 < m.confidence < 1.0 for m in mapped)
    assert all(m.confidence == 0.0 for m in result.unmapped)
    assert 0.0 < result.quality_score() <= 1.0


def test_mapping_is_deterministic(mapping):
    result, _skeleton, _rig = mapping
    skeleton = dff_reader.load_skeleton(DFF)
    rig = sr.load_source_rig(FBX)
    again = map_rig(rig.bones, skeleton)
    assert [m.to_dict() for m in again.matches] == [m.to_dict() for m in result.matches]


# --------------------------------------------------------------------------- #
# name normalisation
# --------------------------------------------------------------------------- #
def test_names_reduce_to_their_bare_bone_name():
    """Namespaces and side markers come off; the bone word stays.

    These are the exact spellings the two real test files use, plus the
    other exporters' conventions.
    """
    assert normalize("mixamorig:LeftUpLeg") == ["upleg"]
    assert normalize("Bip01 L UpperArm") == ["upperarm"]
    assert normalize("upper_arm.L") == ["upperarm"]
    assert normalize("armature:LeftForeArm") == ["forearm"]


def test_leg_is_never_read_as_an_arm():
    """``LeftUpLeg`` is a thigh.  It contains the substring "leg".

    Substring matching sends a Mixamo thigh to the GTA upper-arm tag, which
    then rotates the arm when the leg should move.
    """
    aliases = load_aliases()
    assert role_from_name("mixamorig:LeftUpLeg", aliases) is Role.THIGH
    assert role_from_name("mixamorig:RightUpLeg", aliases) is Role.THIGH
    assert role_from_name("mixamorig:LeftArm", aliases) is Role.UPPER_ARM


def test_side_is_read_from_the_name_exactly():
    assert side_of("mixamorig:LeftArm") == "left"
    assert side_of("mixamorig:RightArm") == "right"
    assert side_of(" L ForeArm") == "left"
    assert side_of("R Thigh") == "right"
    assert side_of("mixamorig:Head") == "center"
    # "right" must not be read as containing a leading "r" only.
    assert side_of("mixamorig:RightUpLeg") == "right"
    assert side_of("mixamorig:Hips") == "center"
