"""The FBX source layer, measured against the real binary fixture."""

from __future__ import annotations

import os
import sys

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
