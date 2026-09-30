"""The FBX source layer, measured against the real binary fixture."""

from __future__ import annotations

import json
import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from gta_fbx_ifp_converter.fbx import read_fbx
from gta_fbx_ifp_converter.fbx import source_rig as sr

FBX = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "testdata", "samba_dancing.fbx",
)


@pytest.fixture(scope="module")
def document():
    return read_fbx(FBX)


def test_parses_without_resync(document):
    assert document.version == 7400
    assert document.resync_count == 0


def test_object_counts(document):
    objects = document.objects
    assert len(objects.find_all("Model")) == 69
    assert len(objects.find_all("AnimationCurveNode")) == 54
    assert len(objects.find_all("AnimationCurve")) == 315


def test_every_curve_exposes_key_times(document):
    """The 12-vs-9 byte array header must be picked per record.

    A mis-picked header makes the compressed KeyTime payload undecodable, so
    this is the regression test for the record-framing fix.
    """
    curves = document.objects.find_all("AnimationCurve")
    with_times = [c for c in curves if sr._array_payload(c, "KeyTime")]
    assert len(with_times) == len(curves)


def test_global_settings(document):
    assert document.up_axis == "Y"
    assert document.unit_scale_factor == 1.0


def test_hierarchy_direction(document):
    """FBX ``OO`` records read child -> parent."""
    rig = sr.build_source_rig(document)
    by_name = {bone.name: bone for bone in rig.bones}
    # The Mixamo spine chain is Hips -> Spine -> Spine1 -> Spine2 -> Neck.
    chain = ["mixamorig:Hips", "mixamorig:Spine", "mixamorig:Spine1",
             "mixamorig:Spine2", "mixamorig:Neck", "mixamorig:Head"]
    assert by_name[chain[0]].parent == -1
    for parent, child in zip(chain, chain[1:]):
        assert by_name[child].parent == by_name[parent].index, child


def test_one_armature_not_fifteen_leaves():
    rig = sr.load_source_rig(FBX)
    assert len(rig.rigs) == 1
    assert rig.bones[rig.rigs[0]].name == "mixamorig:Hips"
    assert len(rig.rig_bones(rig.rigs[0])) == 67


def test_clip_is_built_from_the_animated_stack():
    rig = sr.load_source_rig(FBX)
    assert len(rig.clips) == 1
    clip = rig.clips[0]
    assert clip.name == "mixamo.com"
    assert clip.start_ms == 0.0
    assert abs(clip.stop_ms - 18200.0) < 1e-3
    assert len(clip.source_animated_bones) == 52
    # The key grid is the union of the source keys, not a resampled one.
    assert len(clip.times_s) == 1149


# --------------------------------------------------------------------------
# Regression tests for three silent-corruption bugs found against this fixture.
# Each of these produced plausible-looking output rather than an error, which
# is exactly why they need a test that states the expected physical result.
# --------------------------------------------------------------------------


def test_transform_comes_from_properties70_not_the_legacy_list(document):
    """FBX 7.4 keeps local TRS in ``Properties70``.

    Reading the legacy property names positionally makes every bone sit on
    its parent's origin, so the whole rig collapses to a point.
    """
    rig = sr.build_source_rig(document)
    by_name = {bone.name: bone for bone in rig.bones}
    hips = by_name["mixamorig:Hips"]
    model = next(
        m for m in document.objects.find_all("Model")
        if m.primary_name == "mixamorig:Hips"
    )
    # The legacy property list has no `Lcl Translation` entry at all, so it
    # reads back as the zero default -- which is exactly the value that used
    # to collapse the whole rig onto the origin.
    assert np.allclose(model.vec3("Lcl Translation"), [0.0, 0.0, 0.0])
    assert not np.allclose(model.prop70_vec3("Lcl Translation"), [0.0, 0.0, 0.0])
    assert np.allclose(hips.rest_translation, [0.0, 99.672, 0.247], atol=1e-3)
    leg = by_name["mixamorig:LeftLeg"]
    assert np.allclose(leg.rest_translation, [0.245, -40.595, -0.517], atol=1e-3)


def test_bind_pose_is_a_standing_skeleton(document):
    """A 1.7 m Mixamo character: head up, feet on the ground, limbs apart."""
    rig = sr.build_source_rig(document)
    by_name = {bone.name: bone for bone in rig.bones}
    height = by_name["mixamorig:Head"].bind_world[1, 3]
    assert 150.0 < height < 175.0, height
    # Toes are the lowest joints and sit at the origin plane.
    for side in ("Left", "Right"):
        assert abs(by_name[f"mixamorig:{side}ToeBase"].bind_world[1, 3]) < 5.0
    # The head must be a real child of the hips, not stacked on the origin.
    assert abs(by_name["mixamorig:Spine"].bind_world[1, 3] - height) > 5.0
    # Left/right must actually be mirrored.
    left = by_name["mixamorig:LeftHand"].bind_world[0, 3]
    right = by_name["mixamorig:RightHand"].bind_world[0, 3]
    assert left > 0.0 and right < 0.0
    assert abs(left + right) < 1e-3  # the rig is only mirrored to float32


def test_rotation_curves_are_degrees_not_radians():
    """FBX stores every Euler angle in degrees, in curves and Properties70.

    Verified against three.js's ``FBXLoader`` -- the reference importer, and
    the renderer for this exact fixture -- which applies ``degToRad`` to the
    same values.  Read as radians, a 765-degree spin becomes 13.4 rad and the
    animation folds the skeleton through itself.
    """
    rig = sr.load_source_rig(FBX)
    clip = rig.clips[0]
    degrees = [
        value
        for bone in rig.bones
        for axis in "XYZ"
        for value in (
            [bone.curve(f"Lcl Rotation {axis}").values]
            if bone.curve(f"Lcl Rotation {axis}") and not bone.curve(f"Lcl Rotation {axis}").is_empty
            else []
        )
        for value in value
    ]
    assert max(abs(v) for v in degrees) > 180.0, "fixture no longer exercises degrees"
    # With the conversion applied the pelvis stays upright, matching the
    # 10.5 deg mean tilt measured from three.js's own evaluation of this file.
    tilts = []
    for frame in range(clip.frame_count):
        matrix = _clip_world_positions(rig, clip, frame)["mixamorig:Hips"]
        up = matrix[:3, :3] @ np.array([0.0, 1.0, 0.0])
        tilts.append(np.degrees(np.arccos(np.clip(up[1], -1.0, 1.0))))
    assert np.mean(tilts) < 15.0, np.mean(tilts)


def test_default_euler_order_is_zyx():
    """``RotationOrder`` is absent here, and the fallback is ``ZYX``.

    The FBX SDK default (and the reference loaders' fallback) is ZYX, not the
    XYZ that reads naturally.  Pairing the order letters with the wrong
    components tilts the pelvis ~90 deg on average and puts the head below
    the hips for a third of the clip.
    """
    rig = sr.load_source_rig(FBX)
    assert {bone.rotation_order for bone in rig.bones} == {"ZYX"}
    clip = rig.clips[0]
    by_index = {bone.index: bone for bone in rig.bones}
    world = _clip_world_positions(rig, clip)
    heads = world["mixamorig:Head"][:, 1]
    hips = world["mixamorig:Hips"][:, 1]
    # three.js reports 100.00% of frames with the head above the hips.
    assert (heads > hips).mean() > 0.99, (heads > hips).mean()


def test_local_rotations_match_the_reference_loader():
    """Per-bone local rotations, checked against three.js's own numbers.

    The reference data in ``testdata/reference`` was produced by parsing this
    fixture with ``three@0.160`` ``FBXLoader`` and dumping each bone's local
    rotation at its own key times.
    """
    reference = os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        "testdata", "reference", "threejs_local_rotations.json",
    )
    if not os.path.exists(reference):
        pytest.skip("reference dump not present")
    rig = sr.load_source_rig(FBX)
    by_name = {bone.name: bone for bone in rig.bones}
    from gta_fbx_ifp_converter.core import mathx

    with open(reference, "r", encoding="utf-8") as handle:
        data = json.load(handle)
    errors = []
    for name, track in data.items():
        bone = by_name.get(name)
        if bone is None:
            continue
        for time_s, expected in zip(track["times"], track["quaternions"]):
            rotation, _, _ = bone.trs_at(time_s * 1000.0)
            ours = mathx.euler_to_quat(rotation, bone.rotation_order)
            dot = min(abs(float(np.dot(ours, expected))), 1.0)
            errors.append(mathx.quat_angle_deg(ours, expected))
    assert errors, "reference dump did not match any bone"
    # Agreement is limited only by three.js storing its track values as
    # float32, so the bound is a few times 1e-5 degrees, not a real tolerance.
    assert max(errors) < 1e-3, max(errors)


def test_unkeyed_translation_keeps_its_bind_value():
    """An absent curve means "hold the rest value", not "snap to zero".

    The Mixamo rig keys translation on 3 of 69 bones only; the other limbs
    must keep the offsets their ``Model`` record declares.
    """
    rig = sr.load_source_rig(FBX)
    clip = rig.clips[0]
    unkeyed = [
        bone for bone in rig.bones
        if bone.index in clip.poses
        and not any(
            bone.curve(f"Lcl Translation {a}") and not bone.curve(f"Lcl Translation {a}").is_empty
            for a in "XYZ"
        )
    ]
    assert unkeyed, "fixture no longer has unkeyed translation channels"
    for bone in unkeyed:
        assert np.allclose(clip.poses[bone.index][:, 4:7], bone.rest_translation, atol=1e-9)


def test_batch_path_matches_the_scalar_path(document):
    """The vectorised bake must be bit-identical to the per-key reference.

    It is ~2.5x faster, and it is the only path production code uses.
    """
    rig = sr.load_source_rig(FBX)
    clip = rig.clips[0]
    times_ms = [t * 1000.0 for t in clip.times_s[:25]]
    for bone in rig.bones:
        triples = [bone.trs_at(t) for t in times_ms]
        batched = bone.compose_many(
            np.array([t[0] for t in triples]),
            np.array([t[1] for t in triples]),
            np.array([t[2] for t in triples]),
        )
        for index, (rotation, translation, scale) in enumerate(triples):
            assert np.array_equal(
                batched[index], bone.compose(rotation, translation, scale)
            ), bone.name


def _clip_world_positions(rig, clip, only_frame=None):
    """World matrix of every named bone, for one frame or all of them."""
    from gta_fbx_ifp_converter.core import mathx

    frames = range(clip.frame_count) if only_frame is None else (only_frame,)
    out: dict[str, list[np.ndarray]] = {}
    for frame in frames:
        acc: dict[int, np.ndarray] = {}
        for bone in rig.bones:
            pose = clip.poses.get(bone.index)
            if pose is None:
                continue
            rotation, translation = pose[frame][:4], pose[frame][4:7]
            matrix = np.eye(4)
            matrix[:3, :3] = mathx.quat_to_matrix(rotation)[:3, :3]
            matrix[:3, 3] = translation
            parent = np.eye(4) if bone.parent < 0 else acc[bone.parent]
            acc[bone.index] = parent @ matrix
        for bone in rig.bones:
            if bone.index in acc:
                out.setdefault(bone.name, []).append(acc[bone.index])
    if only_frame is not None:
        return {name: values[0] for name, values in out.items()}
    return {
        name: np.array([matrix[:3, 3] for matrix in values])
        for name, values in out.items()
    }
