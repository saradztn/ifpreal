"""Retargeting: rest pose, local axes, and the transfer of motion itself."""

from .rest_pose import (
    BoneAxis,
    BoneCorrection,
    RestPoseCorrection,
    bone_axes_from_pose,
    build_rest_pose_correction,
    rotation_between,
)

__all__ = [
    "BoneAxis",
    "BoneCorrection",
    "RestPoseCorrection",
    "bone_axes_from_pose",
    "build_rest_pose_correction",
    "rotation_between",
]
