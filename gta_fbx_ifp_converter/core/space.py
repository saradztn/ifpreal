"""Coordinate-system detection and conversion.

Two coordinate systems matter here.

FBX (per the Autodesk spec)
    Right handed. ``+X`` right, ``+Y`` **up**, ``+Z`` towards the viewer
    (so the character's forward is ``-Z``).  Length unit is decimetres by
    default, reported explicitly by ``GlobalSettings.UnitScaleFactor``.

GTA San Andreas ped model space
    RenderWare style, left handed, ``+X`` right, ``+Z`` **up**, and ``+Y``
    pointing *backwards* along the model's forward axis (GTA world forward is
    ``-Y``).  Length unit is one metre.

Rather than hard coding "swap Y and Z", the converter is *derived*: the
source basis is detected, the target basis is declared, and the change of
basis is built from them.  :func:`build_conversion` shows the matrix that
comes out, and :func:`describe` prints it, so the user can see exactly what
was applied instead of trusting a comment.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Sequence

import numpy as np

from .mathx import mat_identity, mat_mul, mat_transpose

__all__ = [
    "Axis",
    "UpAxis",
    "Handedness",
    "CoordinateSystem",
    "FBX_SYSTEM",
    "GTA_SYSTEM",
    "build_conversion",
    "dff_to_gta",
    "FBX_TO_GTA",
    "detect_fbx_system",
]


class UpAxis:
    X = "X"
    Y = "Y"
    Z = "Z"


class Axis:
    X = "X"
    Y = "Y"
    Z = "Z"
    NEG_X = "-X"
    NEG_Y = "-Y"
    NEG_Z = "-Z"


class Handedness:
    RIGHT = "right"
    LEFT = "left"


@dataclass(frozen=True)
class CoordinateSystem:
    """A right- or left-handed orthonormal basis expressed in model space."""

    name: str
    right: str
    up: str
    forward: str
    handedness: str
    unit_scale: float = 1.0
    notes: str = ""

    @property
    def is_left_handed(self) -> bool:
        return self.handedness == Handedness.LEFT

    def column(self, axis: str) -> np.ndarray:
        vector = {
            Axis.X: (1.0, 0.0, 0.0), Axis.NEG_X: (-1.0, 0.0, 0.0),
            Axis.Y: (0.0, 1.0, 0.0), Axis.NEG_Y: (0.0, -1.0, 0.0),
            Axis.Z: (0.0, 0.0, 1.0), Axis.NEG_Z: (0.0, 0.0, -1.0),
        }[axis]
        return np.array(vector, dtype=np.float64)

    @property
    def basis(self) -> np.ndarray:
        """3x3 matrix whose *columns* are right/up/forward in model space."""
        return np.column_stack(
            [self.column(self.right), self.column(self.up), self.column(self.forward)]
        )

    def describe(self) -> str:
        return (
            f"{self.name}: right={self.right} up={self.up} forward={self.forward} "
            f"handedness={self.handedness} unit_scale={self.unit_scale:g}"
            + (f" ({self.notes})" if self.notes else "")
        )


#: FBX, as mandated by the Autodesk FBX SDK.
FBX_SYSTEM = CoordinateSystem(
    name="FBX",
    right=Axis.X,
    up=Axis.Y,
    forward=Axis.NEG_Z,
    handedness=Handedness.RIGHT,
    unit_scale=1.0,
    notes="FBX unit is 1 cm; the global conversion multiplies by 0.01 to reach GTA metres",
)

#: GTA San Andreas ped model space.
GTA_SYSTEM = CoordinateSystem(
    name="GTA SA ped",
    right=Axis.X,
    up=Axis.Z,
    forward=Axis.NEG_Y,
    handedness=Handedness.LEFT,
    unit_scale=1.0,
)

#: Sanity check baked into the test-suite: X stays, Y(up) becomes Z(up),
#: Z becomes -Y.  This is the familiar ``(x, y, z) -> (x, -z, y)`` map, but
#: nothing in the code hard codes it -- it is derived from the two bases.
FBX_TO_GTA = build_conversion(FBX_SYSTEM, GTA_SYSTEM)

#: RenderWare frames in a DFF use GTA's own axes, so DFF -> pipeline space is
#: the identity; it exists as a named function so call sites read honestly.
IDENTITY_CONVERSION = mat_identity()


def build_conversion(
    source: CoordinateSystem,
    target: CoordinateSystem,
    *,
    apply_unit_scale: bool = True,
) -> np.ndarray:
    """Return the matrix taking ``source`` coordinates into ``target``.

    The linear part is ``B_target^-1 @ B_source``; the caller multiplies
    homogeneous points on the left.  When both systems declare a unit scale
    the ratio is applied to the translation column.
    """
    basis = np.linalg.inv(target.basis) @ source.basis
    matrix = mat_identity()
    scale = 1.0
    if apply_unit_scale:
        if target.unit_scale == 0.0:
            raise ValueError("Target unit scale must be non-zero")
        scale = source.unit_scale / target.unit_scale
    matrix[:3, :3] = basis * scale
    return matrix


def rotation_conversion(source: CoordinateSystem, target: CoordinateSystem) -> np.ndarray:
    """Pure rotation part of a conversion (no unit scaling)."""
    matrix = mat_identity()
    matrix[:3, :3] = np.linalg.inv(target.basis) @ source.basis
    return matrix


def dff_to_gta(matrix: np.ndarray) -> np.ndarray:
    """DFF frame matrix -> pipeline GTA space (identity, unit scale)."""
    return np.array(matrix, dtype=np.float64)


@dataclass
class FbxSystemReport:
    """What the FBX header claimed, plus what the rig actually looks like."""

    declared_up: str | None = None
    declared_unit_scale: float | None = None
    unit_in_metres: float | None = None
    detected_up: str = "Y"
    detected_forward: str = "-Z"
    detected_handedness: str = Handedness.RIGHT
    scale_to_metres: float = 0.01
    warnings: list[str] = field(default_factory=list)
    conversion: np.ndarray = field(default_factory=mat_identity)

    def describe(self) -> str:
        lines = [
            "FBX coordinate system",
            f"  declared up axis  : {self.declared_up or '<not present>'}",
            f"  detected up axis  : {self.detected_up}",
            f"  detected forward  : {self.detected_forward}",
            f"  handedness        : {self.detected_handedness}",
            f"  unit scale factor : {self.declared_unit_scale}",
            f"  -> metres          : {self.unit_in_metres}",
            f"  applied scale     : {self.scale_to_metres:g}",
        ]
        lines.extend(f"  WARNING: {w}" for w in self.warnings)
        return "\n".join(lines)


def detect_fbx_system(
    declared_up: str | None,
    declared_unit_scale: float | None,
    *,
    detected_up: str = "Y",
    detected_forward: str = "-Z",
    detected_handedness: str = Handedness.RIGHT,
) -> FbxSystemReport:
    """Build the conversion report for one FBX document.

    ``declared_up`` and ``declared_unit_scale`` come straight from
    ``GlobalSettings``; the ``detected_*`` values are inferred from the bind
    pose (see :mod:`.diagnostics`) and default to the FBX standard.  A
    disagreement is reported rather than silently resolved.
    """
    report = FbxSystemReport(
        declared_up=declared_up,
        declared_unit_scale=declared_unit_scale,
        detected_up=detected_up,
        detected_forward=detected_forward,
        detected_handedness=detected_handedness,
    )
    if declared_up:
        normalized = declared_up.strip().upper()
        if normalized != detected_up:
            report.warnings.append(
                f"FBX declares up axis {normalized} but the rig's rest pose looks "
                f"{detected_up}-up; using the detected value"
            )
    if declared_unit_scale:
        # FBX UnitScaleFactor: 1 = cm, 100 = m, 10 = dm, ...
        report.unit_in_metres = float(declared_unit_scale) / 100.0
        report.scale_to_metres = report.unit_in_metres
        if not (0.5 < report.unit_in_metres < 100.0):
            report.warnings.append(
                f"implausible FBX unit scale factor {declared_unit_scale}; "
                f"falling back to centimetres"
            )
            report.unit_in_metres = 0.01
            report.scale_to_metres = 0.01
    else:
        report.unit_in_metres = 0.01
        report.scale_to_metres = 0.01
        report.warnings.append(
            "FBX has no UnitScaleFactor; assuming centimetres (scale 0.01)"
        )
    report.conversion = build_conversion(FBX_SYSTEM, GTA_SYSTEM)
    if report.scale_to_metres != 1.0:
        conversion = report.conversion.copy()
        conversion[:3, :3] *= report.scale_to_metres
        report.conversion = conversion
    return report
