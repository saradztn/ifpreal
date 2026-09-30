"""Tests for the failure checks.

A diagnostic that fires on correct files is worse than one that stays quiet:
a user learns to ignore it, and then it is not there when a real defect
ships. So most of these are two-sided -- the check must stay quiet on the
bundled conversion *and* fire on a file that has been deliberately broken.
"""
from __future__ import annotations

import math

import numpy as np
import pytest

from gta_fbx_ifp_converter.core import mathx
from gta_fbx_ifp_converter.fbx import source_rig as sr
from gta_fbx_ifp_converter.gta import dff_reader as dr
from gta_fbx_ifp_converter.gta.ifp_reader import read_ifp
from gta_fbx_ifp_converter.validate import diagnostics
from gta_fbx_ifp_converter.validate.diagnostics import Defect

FBX = "testdata/samba_dancing.fbx"
DFF = "testdata/male01.dff"


@pytest.fixture(scope="module")
def converted(tmp_path_factory):
    """Returns a function returning a *fresh* parsed file.

    A module-scoped fixture that hands out one parsed object would let a test
    that deliberately corrupts it leak that corruption into every test after
    it -- which is how "the conversion is clean" passes or fails depending on
    alphabetical order rather than on what the code does.
    """
    """The bundled FBX converted against the bundled DFF, written to disk."""
    from gta_fbx_ifp_converter.fbx import source_rig as source_rig_module
    from gta_fbx_ifp_converter.gta.ifp_build import build_animation
    from gta_fbx_ifp_converter.gta.ifp_writer import write_ifp
    from gta_fbx_ifp_converter.mapping import evaluate, map_rig
    from gta_fbx_ifp_converter.retarget import (
        build_rest_pose_correction,
        retarget_clip,
    )
    from gta_fbx_ifp_converter.retarget.transfer import RetargetSettings, RootMode

    rig = source_rig_module.load_source_rig(FBX)
    skeleton = dr.load_skeleton(DFF)
    mapping = map_rig(rig.bones, skeleton)
    evaluate(mapping, skeleton)
    correction = build_rest_pose_correction(
        rig.bones, [b.parent for b in rig.bones],
        [b.bind_local_translation for b in rig.bones],
        [b.bind_local_quat for b in rig.bones],
        skeleton.bones, [b.parent for b in skeleton.bones],
        [b.bind_local_translation for b in skeleton.bones],
        [b.bind_local_quat for b in skeleton.bones],
        mapping.mapped)
    retarget = retarget_clip(rig, rig.animated_clips[0], mapping, correction,
                             skeleton.bones,
                             RetargetSettings(root_mode=RootMode.IN_PLACE))
    built = build_animation("TEST", retarget, skeleton, "TEST")
    path = tmp_path_factory.mktemp("ifp") / "test.ifp"
    write_ifp(str(path), [built.animation], "TEST")
    raw = path.read_bytes()

    def parse() -> "read_ifp":
        """A freshly parsed copy, safe to break."""
        tmp = tmp_path_factory.mktemp("copy") / "copy.ifp"
        tmp.write_bytes(raw)
        return read_ifp(str(tmp))

    return rig, skeleton, parse, correction, mapping


def _rotated(parsed, bone_name, extra, skeleton):
    """Rotate one bone's every key by `extra` -- a driven-wrong joint."""
    bone = {b.name: b for b in skeleton.bones}[bone_name]
    for animation in parsed.animations:
        for obj in animation.objects:
            if obj.bone_id != bone.bone_id:
                continue
            obj.frames = [
                type(f)(
                    rotation_raw=tuple(
                        int(round(v * 4096))
                        for v in mathx.quat_multiply(
                            extra,
                            np.asarray(f.rotation, float)
                            / float(np.linalg.norm(f.rotation)))),
                    time_units=f.time_units,
                    translation_raw=None)
                for f in obj.frames
            ]
    return parsed


class TestReversedJoints:
    def test_the_real_conversion_is_not_flagged(self, converted):
        _, skeleton, parse, _, _ = converted
        report = diagnostics.check_reversed_joints(parse(), skeleton)
        assert report.is_clean, [f.describe() for f in report.findings]

    def test_a_knee_driven_the_long_way_round_is_caught(self, converted):
        """The check has to actually fire, or it is decoration."""
        _, skeleton, parse, _, _ = converted
        # 180 degrees is a full flip: the knee now bends the opposite way
        # and its travel from the rest pose passes a half turn, which is the
        # only thing that distinguishes it from a deep but correct bend.
        broken = _rotated(parse(), " L Calf",
                          mathx.quat_from_axis_angle([0, 0, 1.0],
                                                      math.radians(180)),
                          skeleton)
        report = diagnostics.check_reversed_joints(broken, skeleton)
        assert not report.is_clean, "an over-traveling knee went unreported"
        assert any(f.defect is Defect.REVERSED_JOINT for f in report.findings)

    def test_it_stays_quiet_without_a_skeleton(self, converted):
        """There is no rest pose to measure against, so it must not guess."""
        _, _, parse, _, _ = converted
        assert diagnostics.check_reversed_joints(parse()).is_clean

    def test_the_threshold_has_headroom_on_real_motion(self, converted):
        """A correct conversion must not sit just under the limit.

        The Samba clip's elbows reach 145.8 degrees. A threshold at 150 leaves
        4 degrees, which is a check waiting to reject the next ped. This
        records the real figure so a change in the retarget cannot quietly
        push valid motion over the line.
        """
        _, skeleton, parse, _, _ = converted
        parsed = parse()
        bones = {b.name: b for b in skeleton.bones}
        worst = 0.0
        for name in (" R Calf", " L Calf", " R ForeArm", " L ForeArm"):
            bone = bones[name]
            obj = diagnostics._find_object(parsed, bone.bone_id)
            rest = np.asarray(bone.bind_local_quat, float)
            rest /= np.linalg.norm(rest)
            conjugate = np.array([-rest[0], -rest[1], -rest[2], rest[3]])
            for frame in obj.frames:
                q = np.asarray(frame.rotation, float)
                q /= np.linalg.norm(q)
                relative = mathx.quat_multiply(q, conjugate)
                angle = 2 * math.degrees(
                    math.acos(min(1.0, abs(float(relative[3])))))
                worst = max(worst, angle)
        assert worst < diagnostics.MAX_JOINT_SWING_DEG
        assert diagnostics.MAX_JOINT_SWING_DEG - worst > 20.0, (
            f"valid motion reaches {worst:.1f} degrees against a "
            f"{diagnostics.MAX_JOINT_SWING_DEG} limit; too little headroom")


class TestInvalidQuaternions:
    def test_a_non_unit_quaternion_is_caught(self, converted):
        _, _, parse, _, _ = converted
        parsed = parse()
        for animation in parsed.animations:
            for obj in animation.objects:
                if obj.frames:
                    obj.frames[10] = type(obj.frames[10])(
                        rotation_raw=(4000, 4000, 4000, 4000),
                        time_units=obj.frames[10].time_units,
                        translation_raw=None)
        report = diagnostics.check_quaternions(parsed)
        assert report.has(Defect.INVALID_QUATERNION), \
            "a quaternion stored at twice unit length is not a rotation"


class TestHanimIds:
    def test_every_written_id_exists_in_the_target_dff(self, converted):
        """The rule the whole project is built on."""
        _, skeleton, parse, _, _ = converted
        parsed = parse()
        real = {b.bone_id for b in skeleton.bones if b.is_addressable}
        for animation in parsed.animations:
            for obj in animation.objects:
                assert obj.bone_id in real, (
                    f"{obj.name!r} carries id {obj.bone_id}, which is not in "
                    f"the DFF")

    def test_an_invented_id_is_reported(self, converted):
        _, skeleton, parse, _, _ = converted
        parsed = parse()
        for animation in parsed.animations:
            for obj in animation.objects:
                if obj.name == " L Calf":
                    obj.bone_id = 61        # no such bone
        report = diagnostics.check_hanim_ids(parsed, skeleton, diagnostics.DiagnosticReport())
        assert not report.is_clean
        assert any(f.defect is Defect.INVALID_HANIM_ID
                   for f in report.findings)


class TestFullDiagnosticRun:
    def test_the_whole_conversion_is_clean(self, converted):
        rig, skeleton, parse, correction, mapping = converted
        report = diagnostics.diagnose_all(
            rig=rig, skeleton=skeleton, mapping=mapping,
            parsed=parse(), correction=correction)
        assert report.is_clean, [f.describe() for f in report.findings]

    def test_a_mirrored_target_skin_warns_rather_than_fails(self, converted):
        """male01.dff's own upper arms are 179 degrees from mirrored.

        That is a property of the shipped skin. A check that failed on it
        would fire on every correct conversion, and a user would learn to
        ignore it.
        """
        rig, skeleton, parse, correction, mapping = converted
        report = diagnostics.diagnose_all(
            rig=rig, skeleton=skeleton, mapping=mapping,
            parsed=parse(), correction=correction)
        mirrored = [f for f in report.findings
                    if f.defect is Defect.MIRRORED_MAPPING]
        assert mirrored, "the known ped asymmetry should still be reported"
        assert all(not f.fatal for f in mirrored)
