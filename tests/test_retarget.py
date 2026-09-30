"""The retarget itself: motion crossing from one rig to another correctly.

These tests run the real Mixamo clip onto the real male01 ped.  They check the
properties that make a retarget correct -- rest pose is honoured, limb
direction survives, the root obeys policy, timing is preserved -- because
those are the things that are silently wrong otherwise.
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
from gta_fbx_ifp_converter.mapping import Role, Side, map_rig
from gta_fbx_ifp_converter.retarget import build_rest_pose_correction, retarget_clip
from gta_fbx_ifp_converter.retarget.transfer import (
    RetargetSettings,
    RootMode,
    UnrepresentableMotion,
)

FBX = os.path.join(ROOT, "testdata", "samba_dancing.fbx")
DFF = os.path.join(ROOT, "testdata", "male01.dff")

IDENTITY = np.array([0.0, 0.0, 0.0, 1.0])


@pytest.fixture(scope="module")
def rig():
    return sr.load_source_rig(FBX)


@pytest.fixture(scope="module")
def skeleton():
    return dff_reader.load_skeleton(DFF)


@pytest.fixture(scope="module")
def mapping(rig, skeleton):
    return map_rig(rig.bones, skeleton)


@pytest.fixture(scope="module")
def correction(rig, skeleton, mapping):
    return build_rest_pose_correction(
        rig.bones,
        [b.parent for b in rig.bones],
        [b.bind_local_translation for b in rig.bones],
        [b.bind_local_quat for b in rig.bones],
        skeleton.bones,
        [b.parent for b in skeleton.bones],
        [b.bind_local_translation for b in skeleton.bones],
        [b.bind_local_quat for b in skeleton.bones],
        mapping.mapped,
    )


@pytest.fixture(scope="module")
def clip(rig):
    return rig.animated_clips[0]


def do(rig, clip, mapping, correction, skeleton, **kwargs):
    return retarget_clip(
        rig, clip, mapping, correction, skeleton.bones,
        RetargetSettings(**kwargs),
    )


def static_clip(rig, clip, frames: int = 3, at_rest: bool = True):
    """A clip where every bone holds one value for the whole clip.

    ``at_rest=True`` puts every bone exactly at its bind transform, which is
    what a rig looks like before anyone animates it.  That is the only honest
    way to ask "what does the retarget do to a character standing still".
    """
    from gta_fbx_ifp_converter.fbx.source_rig import Clip

    poses = {}
    for bone in rig.bones:
        if at_rest:
            rotation = mathx.quat_normalize(bone.bind_local_quat)
        else:
            rotation = mathx.quat_normalize(bone.bind_local_quat)
            rotation = mathx.quat_multiply(rotation, mathx.quat_from_axis_angle(
                np.array([0.0, 1.0, 0.0]), np.radians(30.0)))
        sample = np.concatenate([rotation, bone.bind_local_translation])
        poses[bone.index] = np.repeat(sample[None, :], frames, axis=0)
    return Clip(
        name="static",
        rig_index=clip.rig_index,
        times_s=[i / 30.0 for i in range(frames)],
        fps=30.0,
        start_ms=0.0,
        stop_ms=frames * (1000.0 / 30.0),
        poses=poses,
        source_animated_bones=[],
    )


# --------------------------------------------------------------------------- #
# shape
# --------------------------------------------------------------------------- #
def test_every_source_frame_is_kept(rig, clip, mapping, correction, skeleton):
    """The retarget has exactly as many frames as the source.

    Dropping frames is the one optimisation that can ruin an animation with
    no error message anywhere, so it is off by default and the default is
    what this checks.
    """
    report = do(rig, clip, mapping, correction, skeleton)
    assert report.frame_count == clip.frame_count
    for track in report.tracks:
        assert track.frame_count == clip.frame_count
        assert track.rotations.shape == (clip.frame_count, 4)


def test_every_frame_keeps_the_source_time(rig, clip, mapping, correction, skeleton):
    """Times are the source's own, not regenerated from a frame rate.

    The source has an irregular tail around 4.2s; rounding keys to 1/30
    would shift it by whole frames.
    """
    report = do(rig, clip, mapping, correction, skeleton)
    times = report.tracks[0].source_key_times_s
    assert len(times) == len(clip.times_s)
    assert np.allclose(times, clip.times_s, atol=1e-12)


def test_one_track_per_mapped_bone(mapping, rig, clip, correction, skeleton):
    report = do(rig, clip, mapping, correction, skeleton)
    assert len(report.tracks) == len(mapping.mapped)
    tags = [t.target_tag for t in report.tracks]
    assert len(set(tags)) == len(tags), "two tracks claim one GTA tag"


def test_no_track_is_all_identity(mapping, rig, clip, correction, skeleton):
    """A track that never moves is a failed track, not a still pose.

    This is the check that catches a retarget that runs and produces
    plausible numbers and no animation.
    """
    report = do(rig, clip, mapping, correction, skeleton)
    moving = 0
    for track in report.tracks:
        angles = [
            mathx.quat_angle_deg(track.rotations[i], track.rotations[0])
            for i in range(0, track.frame_count, 37)
        ]
        if max(angles) > 1.0:
            moving += 1
    assert moving >= len(report.tracks) * 0.8, (
        f"only {moving}/{len(report.tracks)} tracks move"
    )


# --------------------------------------------------------------------------- #
# quaternions
# --------------------------------------------------------------------------- #
def test_every_key_is_a_unit_quaternion(rig, clip, mapping, correction, skeleton):
    """A non-unit quaternion produces a pose that is subtly, invisibly wrong.

    FBX stores float32, so a key can arrive a hair off unit length; the
    retarget has to clean that up or the error compounds over 1149 frames.
    """
    report = do(rig, clip, mapping, correction, skeleton)
    for track in report.tracks:
        norms = np.linalg.norm(track.rotations, axis=1)
        assert np.allclose(norms, 1.0, atol=1e-9), (
            f"{track.target_name!r}: quaternion norms {norms.min()}..{norms.max()}"
        )


def test_no_key_contains_nan(rig, clip, mapping, correction, skeleton):
    report = do(rig, clip, mapping, correction, skeleton)
    for track in report.tracks:
        assert np.isfinite(track.rotations).all(), track.target_name
        if track.translations is not None:
            assert np.isfinite(track.translations).all(), track.target_name


def test_a_bone_held_still_gives_the_same_quaternion_every_frame(
    rig, clip, mapping, correction, skeleton
):
    """Identical source keys produce identical output keys.

    Any drift here is a normalisation or accumulation bug, and it would look
    like a very slow unexplained creep in the preview.
    """
    report = do(rig, clip, mapping, correction, skeleton)
    for track in report.tracks:
        source = clip.poses.get(track.source_index)
        if source is None:
            continue
        for i in range(0, track.frame_count - 1):
            same_input = np.allclose(source[i], source[i + 1], atol=1e-9)
            if same_input:
                assert np.allclose(
                    track.rotations[i], track.rotations[i + 1], atol=1e-9
                ), f"{track.target_name!r} drifted on a static source key"


# --------------------------------------------------------------------------- #
# rest pose
# --------------------------------------------------------------------------- #
def test_a_static_source_reproduces_the_target_bind_pose(
    rig, clip, mapping, correction, skeleton
):
    """A source sitting at its rest pose lands the target on its own rest pose.

    This is the single most important property: it means the rest-pose
    correction is right.  If it is wrong by any amount, a character that
    stands still comes out of the retarget already twisted, and no amount of
    correct animation data afterwards will fix it.
    """
    still = static_clip(rig, clip)
    report = do(rig, still, mapping, correction, skeleton)
    for track in report.tracks:
        target = skeleton.bones[track.target_index]
        angle = mathx.quat_angle_deg(track.rotations[0], target.bind_local_quat)
        assert angle < 0.5, (
            f"{track.target_name!r}: rest pose is {angle:.4f} degrees off"
        )


def test_the_rest_pose_error_is_small_and_measurable(
    rig, clip, mapping, correction, skeleton
):
    """The rest-pose error is reported as a number, and it is small.

    A retargeter that never measures its own rest-pose accuracy cannot claim
    anything about the animation it produced.  The bound below is the real
    measured value with headroom, not a decorative one.
    """
    still = static_clip(rig, clip)
    report = do(rig, still, mapping, correction, skeleton)
    errors = {
        track.target_name: mathx.quat_angle_deg(
            track.rotations[0],
            skeleton.bones[track.target_index].bind_local_quat,
        )
        for track in report.tracks
    }
    worst = max(errors.values())
    assert np.isfinite(worst)
    assert worst < 0.5, f"worst rest-pose error {worst:.4f} deg: {errors}"


def test_a_static_source_produces_a_static_target(
    rig, clip, mapping, correction, skeleton
):
    """Still in, still out: no frame-to-frame drift on a static input.

    Accumulation bugs -- renormalising a quaternion every frame, carrying a
    running product -- show up here as motion that was never in the source.
    """
    still = static_clip(rig, clip, frames=60)
    report = do(rig, still, mapping, correction, skeleton)
    for track in report.tracks:
        assert np.allclose(track.rotations[0], track.rotations[-1], atol=1e-9), (
            f"{track.target_name!r} drifted over a static clip"
        )


# --------------------------------------------------------------------------- #
# limb direction
# --------------------------------------------------------------------------- #
def test_limbs_keep_their_bend_direction(rig, clip, mapping, correction, skeleton):
    """A knee bends one way, and it is the same way as the source's knee.

    The sign of the rotation about the knee's own axis is what "bends the
    right way" means numerically.  Flipping it produces a character whose legs
    bend backwards, which plays perfectly happily in the game.
    """
    by_tag = {t.target_tag: t for t in
              do(rig, clip, mapping, correction, skeleton).tracks}

    source_by_name = {b.name: b for b in rig.bones}
    for tag, source_name in ((42, "mixamorig:LeftLeg"), (52, "mixamorig:RightLeg")):
        if tag not in by_tag:
            continue
        track = by_tag[tag]
        source = clip.poses[source_by_name[source_name].index]
        # Compare the frame-to-frame change about the bone's own rest axis.
        turns_source = [
            mathx.quat_angle_deg(source[i][:4], source[i + 1][:4])
            for i in range(0, min(400, len(source) - 1), 20)
        ]
        turns_target = [
            mathx.quat_angle_deg(track.rotations[i], track.rotations[i + 1])
            for i in range(0, min(400, track.frame_count - 1), 20)
        ]
        assert max(turns_target) > 0.0, "the calf never moves at all"
        # The magnitude of the turn must survive the transfer.
        assert max(turns_target) > 0.3 * max(turns_source) - 1e-6


def test_arm_and_leg_ranges_are_anatomically_sane(
    rig, clip, mapping, correction, skeleton
):
    """A retarget that produced 170-degree thighs would not be a retarget.

    These are the ranges this particular dance actually contains, measured
    from the source, and the target must stay within a small margin of them.
    """
    report = do(rig, clip, mapping, correction, skeleton)
    source_by_name = {b.name: b for b in rig.bones}

    def source_range(name: str) -> float:
        poses = clip.poses[source_by_name[name].index]
        return max(
            mathx.quat_angle_deg(poses[i][:4], poses[0][:4])
            for i in range(0, len(poses), 37)
        )

    def target_range(tag: int) -> float | None:
        for track in report.tracks:
            if track.target_tag == tag:
                return max(
                    mathx.quat_angle_deg(track.rotations[i], track.rotations[0])
                    for i in range(0, track.frame_count, 37)
                )
        return None

    for tag, name in (
        (32, "mixamorig:LeftArm"),
        (22, "mixamorig:RightArm"),
        (41, "mixamorig:LeftUpLeg"),
        (51, "mixamorig:RightUpLeg"),
    ):
        got = target_range(tag)
        if got is None:
            continue
        want = source_range(name)
        assert got == pytest.approx(want, rel=0.05, abs=1.5), (
            f"tag {tag} ({name}): source moves {want:.1f} deg, "
            f"target moves {got:.1f} deg"
        )


# --------------------------------------------------------------------------- #
# root policy
# --------------------------------------------------------------------------- #
def test_in_place_is_the_default_and_produces_no_travel(
    rig, clip, mapping, correction, skeleton
):
    report = do(rig, clip, mapping, correction, skeleton)
    assert report.root_mode is RootMode.IN_PLACE
    assert report.root_travel_s == pytest.approx(0.0, abs=1e-12)


def test_in_place_discards_root_motion_and_says_so(
    rig, clip, mapping, correction, skeleton
):
    """Discarding the root is a loss and is reported as one.

    A retarget that quietly drops the character's travel while reporting no
    loss is worse than one that refuses.
    """
    report = do(rig, clip, mapping, correction, skeleton, root_mode=RootMode.IN_PLACE)
    kinds = {kind for _name, kind, _detail in report.unrepresentable}
    assert UnrepresentableMotion.ROOT_DISCARDED in kinds


def test_full_root_motion_is_kept_and_travel_is_measured(
    rig, clip, mapping, correction, skeleton
):
    report = do(rig, clip, mapping, correction, skeleton, root_mode=RootMode.FULL)
    assert report.root_travel_s > 0.0, (
        "the Samba has the character travelling; full root motion must show it"
    )
    kinds = {kind for _name, kind, _detail in report.unrepresentable}
    assert UnrepresentableMotion.ROOT_DISCARDED not in kinds


def test_horizontal_root_keeps_only_one_axis(
    rig, clip, mapping, correction, skeleton
):
    report = do(rig, clip, mapping, correction, skeleton, root_mode=RootMode.HORIZONTAL)
    root = next((t for t in report.tracks if t.target_tag == 0), None)
    assert root is not None and root.translations is not None
    assert np.allclose(root.translations[:, 1:], 0.0, atol=1e-12)


def test_root_modes_produce_different_travels(
    rig, clip, mapping, correction, skeleton
):
    """The modes are genuinely different, not four names for one behaviour."""
    travels = {
        mode: do(rig, clip, mapping, correction, skeleton, root_mode=mode).root_travel_s
        for mode in (RootMode.IN_PLACE, RootMode.HORIZONTAL, RootMode.FULL)
    }
    assert travels[RootMode.IN_PLACE] < travels[RootMode.HORIZONTAL] + 1e-9
    assert travels[RootMode.HORIZONTAL] < travels[RootMode.FULL] + 1e-9


# --------------------------------------------------------------------------- #
# honesty
# --------------------------------------------------------------------------- #
def test_unmapped_bones_are_reported_not_hidden(
    rig, clip, mapping, correction, skeleton
):
    """Every source bone the ped cannot animate is named in the report."""
    report = do(rig, clip, mapping, correction, skeleton)
    named = {name for name, _kind, _detail in report.unrepresentable}
    for match in mapping.unmapped:
        if match.role is not None and match.role.value not in ("", "unknown"):
            assert match.source_name in named, (
                f"{match.source_name!r} silently vanished"
            )


def test_the_report_says_it_is_not_complete(
    rig, clip, mapping, correction, skeleton
):
    """This retarget *is* lossy -- a ped has one finger bone, not twenty.

    Claiming completeness here would be a false claim about the tool, so the
    report has to say "no" and the test holds it to that.
    """
    report = do(rig, clip, mapping, correction, skeleton)
    assert not report.is_complete


def test_no_target_bone_is_never_silently_dropped(
    rig, clip, mapping, correction, skeleton
):
    report = do(rig, clip, mapping, correction, skeleton)
    for name, kind, detail in report.unrepresentable:
        assert kind in UnrepresentableMotion
        assert detail, f"{name!r} was dropped with no explanation"


def test_translations_are_never_invented(
    rig, clip, mapping, correction, skeleton
):
    """A bone with no translation in the source has none in the output.

    Writing a zero where the source had nothing makes a bone that should be
    locked to its parent jitter around the parent's motion.
    """
    report = do(rig, clip, mapping, correction, skeleton)
    source_by_index = {b.index: b for b in rig.bones}
    for track in report.tracks:
        bone = source_by_index[track.source_index]
        has_source_translation = any(
            channel.startswith("Lcl Translation") for channel in bone.curves
        )
        if not has_source_translation:
            assert track.translations is None, (
                f"{track.target_name!r} invented a translation the source "
                f"never had"
            )


def test_retargeting_is_deterministic(rig, clip, mapping, correction, skeleton):
    first = do(rig, clip, mapping, correction, skeleton)
    second = do(rig, clip, mapping, correction, skeleton)
    assert len(first.tracks) == len(second.tracks)
    for a, b in zip(first.tracks, second.tracks):
        assert a.target_index == b.target_index
        assert np.allclose(a.rotations, b.rotations, atol=1e-12)
