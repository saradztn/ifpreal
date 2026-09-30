"""The basis conversion is derived, not hard coded, and must not flip."""

from __future__ import annotations

import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from gta_fbx_ifp_converter.core import space


def apply(vector):
    matrix = space.FBX_TO_GTA
    return matrix @ np.array([*vector, 1.0])


def test_fbx_axes_map_to_the_declared_gta_axes():
    """FBX: +X right, +Y up, -Z forward.  GTA: +X right, +Z up, -Y forward."""
    assert np.allclose(apply((1, 0, 0))[:3], (1, 0, 0))
    assert np.allclose(apply((0, 1, 0))[:3], (0, 0, 1))
    assert np.allclose(apply((0, 0, -1))[:3], (0, -1, 0))


def test_handedness_change_is_a_reflection_not_a_roation():
    """A (right, up, forward) triple is left handed; det = -1 is correct."""
    assert np.linalg.det(space.FBX_TO_GTA[:3, :3]) < 0.0


def test_unit_scale_is_reported_not_guessed():
    assert space.detect_fbx_system("Y", 1.0).unit_in_metres == 0.01
    assert space.detect_fbx_system("Y", 100.0).unit_in_metres == 1.0
    assert space.detect_fbx_system("Y", 1.0).warnings == []
    assert space.detect_fbx_system("Y", None).warnings      # missing -> warn
    assert space.detect_fbx_system("Y", 3.7).warnings      # unknown -> warn
