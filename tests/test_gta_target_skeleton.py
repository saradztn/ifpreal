"""The target DFF layer, measured against real GTA San Andreas ped DFFs."""

from __future__ import annotations

import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from gta_fbx_ifp_converter.gta import dff_reader as dr
from gta_fbx_ifp_converter.gta.bones import SaBoneTag

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DFFS = ["male01.dff", "claude.dff", "army.dff", "ballas1.dff", "copgrl3.dff"]


@pytest.fixture(scope="module")
def skeleton():
    return dr.load_skeleton(os.path.join(ROOT, "testdata", "male01.dff"))


@pytest.mark.parametrize("name", DFFS)
def test_every_shipped_skin_loads(name):
    skel = dr.load_skeleton(os.path.join(ROOT, "testdata", name))
    assert skel.has_hanim
    assert len(skel.bones) == 33


def test_bind_pose_is_a_standing_z_up_ped(skeleton):
    """Rows-are-basis, no transpose: head high, toes low, ~1.9 m tall."""
    zs = [b.bind_world_gta[2, 3] for b in skeleton.bones]
    assert min(zs) < -0.9      # the toes
    assert max(zs) > 0.85      # the top of the head
    assert abs(max(zs) - min(zs) - 1.9) < 0.2
    # The left arm must be on +Y and the right arm on -Y.
    left = skeleton.find_tag(SaBoneTag.L_UPPER_ARM)
    right = skeleton.find_tag(SaBoneTag.R_UPPER_ARM)
    assert left.bind_world_gta[1, 3] > 0.0
    assert right.bind_world_gta[1, 3] < 0.0


def test_hanim_node_id_is_reported_not_trusted(skeleton):
    """The file's own ids disagree with the frames; both are kept."""
    resolution = skeleton.tag_resolution
    assert len(resolution.hanim_id_of_frame) == 32
    conflicts = [d for d in resolution.diagnostics if d.code == "hanim-name-conflict"]
    # Every frame that has both a plugin id and a real GTA name disagrees.
    assert len(conflicts) == 31
    assert skeleton.bones[0].resolved_tag is None   # unnamed container root
    assert skeleton.bones[3].hanim_id == 3          # stale, as measured
    assert skeleton.bones[3].resolved_tag == 51     # what actually gets written


def test_every_canonical_tag_resolves_to_a_frame(skeleton):
    """Resolution is DFF-driven but has to cover the whole ped."""
    expected = {
        0, 1, 2, 3, 4, 5, 6, 7, 8,
        21, 22, 23, 24, 25, 26, 31, 32, 33, 34, 35, 36,
        41, 42, 43, 44, 51, 52, 53, 54, 201, 301, 302,
    }
    assert set(skeleton.tag_resolution.animated_tags) == expected
    for tag in expected:
        bone = skeleton.find_tag(tag)
        assert bone is not None, f"tag {tag} has no frame"
        assert bone.bone_id == tag


def test_resolved_tags_match_the_frame_geometry(skeleton):
    """Left/right must be right: a wrong side shows up in the bind pose."""
    for tag, expected_y in ((SaBoneTag.L_FOOT, 1), (SaBoneTag.R_FOOT, -1)):
        bone = skeleton.find_tag(tag)
        assert np.sign(bone.bind_world_gta[1, 3]) == expected_y
    head = skeleton.find_tag(SaBoneTag.HEAD)
    pelvis = skeleton.find_tag(SaBoneTag.PELVIS)
    assert head.bind_world_gta[2, 3] > pelvis.bind_world_gta[2, 3]


def test_skin_inconsistency_is_reported(skeleton):
    """The DFF's skin array follows the stale order too -- say so.

    ``used_bone_indices`` is a list of HAnim *array* positions.  Under the
    canonical enumeration that is exactly the 28 body bones, but under the
    tags the frames actually resolve to it is a different set, so the reader
    has to raise a diagnostic instead of quietly exporting a wrong skin.
    """
    skinned = {
        b.resolved_tag for b in skeleton.bones
        if b.skin_bone_index is not None and b.resolved_tag is not None
    }
    assert len(skinned) == 28
    codes = {d.code for d in skeleton.tag_resolution.diagnostics}
    assert "skin-index-mismatch" in codes
