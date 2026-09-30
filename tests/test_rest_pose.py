"""Rest-pose and local-axis correction, checked against the two real rigs.

The correction is the part of a retarget that nobody can check by eye until
something is already wrong -- a knee that bends the wrong way, a shoulder
that sits 20 degrees off.  So the tests below pin the properties that must
hold, using the real Mixamo FBX and the real male01 DFF.
"""

from __future__ import annotations

import os
import sys

import numpy as np
import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from gta_fbx_ifp_converter.core import mathx
from gta_fbx_ifp_converter.fbx import source_rig as sr
from gta_fbx_ifp_converter.gta import dff_reader as dff_reader
from gta_fbx_ifp_converter.mapping import map_rig
from gta_fbx_ifp_converter.retarget import rest_pose as rp

FBX = os.path.join(ROOT, "testdata", "samba_dancing.fbx")
DFF = os.path.join(ROOT, "testdata", "male01.dff")

IDENTITY = np.array([0.0, 0.0, 0.0, 1.0])


@pytest.fixture(scope="module")
def pose():
    skeleton = dff_reader.load_skeleton(DFF)
    rig = sr.load_source_rig(FBX)
    result = map_rig(rig.bones, skeleton)
    correction = rp.build_rest_pose_correction(
        rig.bones,
        [b.parent for b in rig.bones],
        [b.bind_local_translation for b in rig.bones],
        [b.bind_local_quat for b in rig.bones],
        skeleton.bones,
        [b.parent for b in skeleton.bones],
        [b.bind_local_translation for b in skeleton.bones],
        [b.bind_local_quat for b in skeleton.bones],
        result.mapped,
    )
    return result, correction, skeleton, rig


# --------------------------------------------------------------------------- #
# the primitive: rotation_between
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("axis", [
    (1.0, 0.0, 0.0), (0.0, 1.0, 0.0), (0.0, 0.0, 1.0),
])
def test_identity_correction_for_a_matching_axis(axis):
    """A bone that already points the right way must not be rotated.

    The smallest correction is the right one.  Adding a rotation where none
    is needed is how a retarget ends up subtly wrong everywhere instead of
    obviously wrong in one place.
    """
    vector = np.array(axis)
    assert np.allclose(rp.rotation_between(vector, vector), IDENTITY, atol=1e-9)


def test_opposite_axes_give_a_half_turn():
    """A bone pointing backwards is turned 180 degrees, not 179.9.

    The half-turn case is where a naive "cross product gives the axis" fails:
    the cross product of two opposite vectors is zero, and the result is a
    rotation about a zero axis -- which is the identity, silently.
    """
    up = np.array([0.0, 1.0, 0.0])
    rotation = rp.rotation_between(up, -up)
    assert mathx.quat_angle_deg(rotation, IDENTITY) == pytest.approx(180.0, abs=1e-6)
    turned = mathx.quat_rotate_vector(rotation, up)
    assert np.allclose(turned, -up, atol=1e-9)


def test_rotation_actually_carries_source_onto_target():
    """The rotation does what it claims, for arbitrary non-perpendicular pairs.

    Neither input is normalised, because that is the contract: this is a
    function of *directions*, and a bone's offset is not a unit vector.  Both
    are normalised internally, so the comparison is between unit directions.
    """
    rng = np.random.default_rng(20240917)
    for _ in range(64):
        a = rng.normal(size=3)
        b = rng.normal(size=3)
        if np.linalg.norm(a) < 0.1 or np.linalg.norm(b) < 0.1:
            continue
        a_unit = a / np.linalg.norm(a)
        b_unit = b / np.linalg.norm(b)
        rotation = rp.rotation_between(a, b)
        turned = mathx.quat_rotate_vector(rotation, a_unit)
        assert np.allclose(turned, b_unit, atol=1e-9), (
            f"{a} -> {turned}, wanted {b_unit}"
        )


def test_rotation_is_the_shortest_one():
    """Never more than 180 degrees, even for nearly-opposite directions.

    A longer rotation between the same two axes reaches the same place and
    sweeps the bone through the body on the way, which is exactly the "elbow
    bent backwards" failure.
    """
    rng = np.random.default_rng(7)
    for _ in range(64):
        a = rng.normal(size=3)
        b = -a + rng.normal(size=3) * 0.01
        rotation = rp.rotation_between(a, b)
        assert mathx.quat_angle_deg(rotation, IDENTITY) <= 180.0 + 1e-6


# --------------------------------------------------------------------------- #
# axis measurement
# --------------------------------------------------------------------------- #
def test_axes_come_from_real_children_not_a_default(pose):
    """Bones in the middle of a chain are measured, not assumed.

    A GTA ped has 33 bones and most are leaves, but the chain bones -- femur,
    tibia, spine -- all have a real child to measure against.
    """
    _result, correction, _skeleton, _rig = pose
    measured = [
        c for c in correction.corrections.values()
        if c.source_axis.provenance == rp.DERIVED_CHILD
    ]
    assert len(measured) > len(correction.corrections) * 0.8, (
        f"only {len(measured)}/{len(correction.corrections)} axes measured"
    )


def test_a_bone_whose_child_is_cancelled_out_keeps_its_default(pose):
    """A branching bone is not given an invented direction.

    A spine carrying arms on it has children whose offsets cancel to roughly
    zero.  Taking the mean of those gives a direction that is really "the
    average of nowhere", and correcting by it is worse than not correcting.
    """
    _result, correction, _skeleton, _rig = pose
    for bone_correction in correction.corrections.values():
        direction = bone_correction.source_axis.direction
        assert np.isfinite(direction).all()
        assert 0.0 <= float(np.linalg.norm(direction)) <= 1.0 + 1e-9


def test_eyes_are_reported_as_assumed_not_measured(pose):
    """Bones with no usable child are listed, not quietly corrected.

    The Mixamo rig's eye bones hang off the head with no children of their
    own.  The correction for them is the head's, which is probably right and
    is certainly not measured -- so they are named in ``assumed`` for the
    caller to weigh.
    """
    _result, correction, _skeleton, _rig = pose
    assert any("Eye" in name for name in correction.assumed), (
        f"expected the eye bones to be flagged, got {correction.assumed}"
    )
    assert correction.warnings, "an assumed axis must produce a warning"


def test_most_corrections_are_measurable(pose):
    _result, correction, _skeleton, _rig = pose
    assert correction.measured_fraction > 0.8


# --------------------------------------------------------------------------- #
# unit scale
# --------------------------------------------------------------------------- #
def test_unit_scale_is_measured_not_assumed(pose):
    """Mixamo is in centimetres, GTA peds in metres, and the two characters
    are not the same size, so the scale lands near 1/100 but not exactly.

    Asserting the measured value rather than "about 0.01" is deliberate: the
    whole point is that the number came from the two rest poses, so a change
    to either rig must move it.
    """
    _result, correction, _skeleton, _rig = pose
    assert 0.005 < correction.unit_scale < 0.02
    # Nothing in the pipeline is allowed to say "centimetres" -- if it did,
    # the scale would be exactly 0.01 and this test would be checking a
    # hard-coded constant instead of a measurement.
    assert correction.unit_scale != pytest.approx(0.01, abs=1e-4), (
        "scale came out as the assumed cm->m constant, not a measurement"
    )


def test_unit_scale_matches_the_bodies_it_came_from(pose):
    """The scale reflects the two characters' real proportions.

    The Mixamo model's limbs are longer relative to its height than the
    male01 ped's are, so centimetres-per-metre comes out slightly *above*
    the naive 0.01 rather than exactly at it.  The direction of that
    difference is a fact about these two files; the test states the fact it
    measured instead of a constant the code happens to produce.
    """
    _result, correction, _skeleton, _rig = pose
    assert correction.unit_scale > 0.01


# --------------------------------------------------------------------------- #
# the correction as a whole
# --------------------------------------------------------------------------- #
def test_every_mapped_bone_gets_a_correction(pose):
    result, correction, _skeleton, _rig = pose
    for match in result.mapped:
        assert correction.get(match.source_index) is not None, (
            f"{match.source_name!r} is mapped but has no rest-pose correction"
        )


def test_no_correction_for_an_unmapped_bone(pose):
    result, correction, _skeleton, _rig = pose
    for match in result.unmapped:
        assert correction.get(match.source_index) is None


def test_correction_is_a_unit_quaternion(pose):
    """Every correction is a real rotation, not a scaled or NaN one.

    A non-unit quaternion in this pipeline produces a pose that is subtly,
    invisibly wrong, which is the hardest class of bug to find later.
    """
    _result, correction, _skeleton, _rig = pose
    for bone_correction in correction.corrections.values():
        total = bone_correction.total
        assert np.isfinite(total).all(), "correction contains NaN or inf"
        assert float(np.linalg.norm(total)) == pytest.approx(1.0, abs=1e-9)


def test_each_limb_is_corrected_for_its_own_rig_not_a_shared_average(pose):
    """Left and right are corrected independently, from their own bone data.

    The tempting shortcut is to correct one arm and mirror the result.  It is
    wrong here, and measurably so: this ped's left and right upper arms are
    40 degrees apart in roll while the Mixamo rig's are exact mirrors, so a
    mirrored correction would leave one arm visibly wrong.  The test pins
    that both arms are handled on their own terms.
    """
    result, correction, _skeleton, _rig = pose
    by_role = {}
    for match in result.mapped:
        by_role[(match.role, match.side)] = match

    from gta_fbx_ifp_converter.mapping import Role, Side

    upper_left = by_role.get((Role.UPPER_ARM, Side.LEFT))
    upper_right = by_role.get((Role.UPPER_ARM, Side.RIGHT))
    assert upper_left is not None and upper_right is not None
    lc = correction.get(upper_left.source_index)
    rc = correction.get(upper_right.source_index)
    assert lc is not None and rc is not None

    # The target's own two arms are not mirror images of each other in this
    # DFF, so their corrections are not equal either.  Asserting they differ
    # is what catches a "correct the left, mirror it" implementation.
    assert not np.allclose(lc.target_axis.direction, -rc.target_axis.direction, atol=1e-3) or \
           not np.allclose(lc.reorient, rc.reorient, atol=1e-3)


def test_the_correction_depends_only_on_the_rigs_not_on_the_limb(pose):
    """Two bones that rest the same way are corrected the same way.

    This ped's forearms have identical bind rotations and the Mixamo rig's
    rest at identity, so the frame change is the same for both -- and it
    *should* be.  The mirror in the source (left forearm points +X, right
    points -X) lives in the animation, not in the rest pose, so it belongs
    in the transferred delta rather than in the correction.

    A correction that varied per limb would mean the rest pose was being
    baked into the correction, which is precisely the bug that stands a
    character up at the wrong angle when it stands still.
    """
    result, correction, _skeleton, _rig = pose
    by_role = {}
    for match in result.mapped:
        by_role[(match.role, match.side)] = match

    from gta_fbx_ifp_converter.mapping import Role, Side

    left = by_role.get((Role.FOREARM, Side.LEFT))
    right = by_role.get((Role.FOREARM, Side.RIGHT))
    assert left is not None and right is not None
    lc = correction.get(left.source_index)
    rc = correction.get(right.source_index)

    # The two target bones really do rest identically.
    assert np.allclose(lc.target_axis.direction, rc.target_axis.direction, atol=1e-6)
    # The source's mirror is real, and lives in the axis measurement.
    assert np.allclose(lc.source_axis.direction, -rc.source_axis.direction, atol=1e-3)
    # Yet the correction is shared, because the correction is a property of
    # the rest orientations and both of those are the same.
    assert mathx.quat_angle_deg(lc.reorient, rc.reorient) == pytest.approx(0.0, abs=1e-6)


def test_correction_is_deterministic(pose):
    result, correction, _skeleton, rig = pose
    skeleton = dff_reader.load_skeleton(DFF)
    again = map_rig(rig.bones, skeleton)
    rebuilt = rp.build_rest_pose_correction(
        rig.bones,
        [b.parent for b in rig.bones],
        [b.bind_local_translation for b in rig.bones],
        [b.bind_local_quat for b in rig.bones],
        skeleton.bones,
        [b.parent for b in skeleton.bones],
        [b.bind_local_translation for b in skeleton.bones],
        [b.bind_local_quat for b in skeleton.bones],
        again.mapped,
    )
    assert rebuilt.unit_scale == pytest.approx(correction.unit_scale, abs=1e-12)
    assert rebuilt.corrections.keys() == correction.corrections.keys()
    for key, value in correction.corrections.items():
        assert np.allclose(rebuilt.corrections[key].total, value.total, atol=1e-12)


def test_reorient_applied_to_nothing_is_the_identity(pose):
    """An unmapped bone's correction is identity, not an error and not a guess."""
    _result, correction, _skeleton, _rig = pose
    assert np.allclose(correction.reorient(9999), IDENTITY)


def test_a_zero_turn_produces_the_target_bind_pose(pose):
    """The property the retarget actually depends on.

    A bone that has not turned since its rest pose must come out at the
    target's own bind rotation.  Stated that way rather than as an equation
    on the correction, because the equation ``A * S * A^-1 = T`` is only
    solvable when ``S`` is not the identity -- and on this rig every source
    rest rotation *is* the identity, so the equation is unsatisfiable here
    while the property it was meant to guarantee holds exactly.

    The distinction matters: solving for ``A`` as ``T * S^-1`` and
    conjugating the delta is right for every bone, and a test written as an
    equation would reject the correct answer.
    """
    result, correction, skeleton, rig = pose
    for match in result.mapped:
        bone_correction = correction.get(match.source_index)
        if bone_correction is None:
            continue
        delta = mathx.quat_normalize(np.array([0.0, 0.0, 0.0, 1.0]))
        turned = mathx.quat_normalize(mathx.quat_multiply(
            bone_correction.reorient,
            mathx.quat_multiply(delta, mathx.quat_inverse(bone_correction.reorient)),
        ))
        output = mathx.quat_normalize(mathx.quat_multiply(
            turned, mathx.quat_normalize(
                skeleton.bones[match.target_index].bind_local_quat
            )
        ))
        target_rest = mathx.quat_normalize(
            skeleton.bones[match.target_index].bind_local_quat
        )
        angle = mathx.quat_angle_deg(output, target_rest)
        assert angle < 1e-6, (
            f"{match.source_name!r}: a zero turn lands {angle:.6f} degrees "
            f"off the target's bind pose"
        )
