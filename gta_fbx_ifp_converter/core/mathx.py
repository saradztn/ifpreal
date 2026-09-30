"""Deterministic quaternion / matrix math used by the whole pipeline.

Conventions used everywhere in this project
-------------------------------------------
* Quaternions are ``numpy`` arrays shaped ``(4,)`` ordered ``(x, y, z, w)``.
  That is the order IFP/ANP3 stores on disk, so no reordering is ever needed
  at the writer boundary.
* Matrices are ``(4, 4)`` ``float64`` arrays using the *column vector*
  convention: a point is a column and transforms as ``p' = M @ p``.
  Composition therefore reads left to right, outer transform first:
  ``world = parent_world @ local``.
* ``from_trs`` builds ``T @ R @ S``; ``decompose`` is its exact inverse for
  rigid + uniform-scale transforms and raises for sheared input.

Nothing here guesses: every function documents the space it operates in.
"""

from __future__ import annotations

import math
from typing import Iterable, Sequence

import numpy as np

__all__ = [
    "EPS",
    "as_quat",
    "quat_to_matrix",
    "matrix_to_quat",
    "quat_multiply",
    "quat_multiply_many",
    "quat_normalize_many",
    "quat_conjugate",
    "quat_inverse",
    "quat_normalize",
    "quat_rotate_vector",
    "quat_from_axis_angle",
    "quat_angle_between",
    "quat_slerp",
    "quat_angle_deg",
    "quat_from_matrix",
    "mat_identity",
    "mat_from_trs",
    "mat_from_trs_many",
    "mat_translation",
    "mat_rotation",
    "mat_scale",
    "mat_mul",
    "mat_inverse",
    "mat_transpose",
    "mat_decompose",
    "mat_xform_point",
    "mat_xform_vector",
    "mat_to_quat",
    "orthonormalize",
    "euler_to_quat",
    "euler_to_quat_many",
    "quat_to_euler",
    "remap_unit_interval",
]

#: Tolerance used when deciding whether two rotations are "the same".
EPS = 1e-9


# --------------------------------------------------------------------------- #
# quaternions
# --------------------------------------------------------------------------- #
def as_quat(value: Sequence[float] | np.ndarray) -> np.ndarray:
    """Return ``value`` as a ``(4,)`` float64 ``(x, y, z, w)`` array."""
    array = np.asarray(value, dtype=np.float64).reshape(-1)
    if array.size != 4:
        raise ValueError(f"A quaternion needs exactly 4 components, got {array.size}")
    return array


def quat_normalize(q: Sequence[float] | np.ndarray) -> np.ndarray:
    """Return the unit quaternion with the same rotation."""
    q = as_quat(q)
    length = float(np.linalg.norm(q))
    if length < EPS:
        raise ValueError("Cannot normalise a zero-length quaternion")
    return q / length


def quat_conjugate(q: Sequence[float] | np.ndarray) -> np.ndarray:
    q = as_quat(q)
    return np.array([-q[0], -q[1], -q[2], q[3]], dtype=np.float64)


def quat_inverse(q: Sequence[float] | np.ndarray) -> np.ndarray:
    """Inverse of a unit quaternion (falls back to the general formula)."""
    q = as_quat(q)
    norm_sq = float(q @ q)
    if norm_sq < EPS:
        raise ValueError("Cannot invert a zero-length quaternion")
    return quat_conjugate(q) / norm_sq


def quat_multiply(
    a: Sequence[float] | np.ndarray,
    b: Sequence[float] | np.ndarray,
) -> np.ndarray:
    """Hamilton product ``a * b``.

    With the column-vector convention this means "apply ``b`` first, then
    ``a``", exactly like matrix multiplication.
    """
    ax, ay, az, aw = (float(v) for v in as_quat(a))
    bx, by, bz, bw = (float(v) for v in as_quat(b))
    return np.array(
        [
            aw * bx + ax * bw + ay * bz - az * by,
            aw * by - ax * bz + ay * bw + az * bx,
            aw * bz + ax * by - ay * bx + az * bw,
            aw * bw - ax * bx - ay * by - az * bz,
        ],
        dtype=np.float64,
    )


def quat_multiply_many(
    a: np.ndarray,
    b: np.ndarray,
) -> np.ndarray:
    """Row-wise Hamilton product for ``(n, 4)`` XYZW quaternion stacks."""
    a = np.atleast_2d(np.asarray(a, dtype=np.float64))
    b = np.atleast_2d(np.asarray(b, dtype=np.float64))
    if a.shape[0] == 1 and b.shape[0] > 1:
        a = np.broadcast_to(a, b.shape)
    elif b.shape[0] == 1 and a.shape[0] > 1:
        b = np.broadcast_to(b, a.shape)
    ax, ay, az, aw = a[:, 0], a[:, 1], a[:, 2], a[:, 3]
    bx, by, bz, bw = b[:, 0], b[:, 1], b[:, 2], b[:, 3]
    return np.stack(
        [
            aw * bx + ax * bw + ay * bz - az * by,
            aw * by - ax * bz + ay * bw + az * bx,
            aw * bz + ax * by - ay * bx + az * bw,
            aw * bw - ax * bx - ay * by - az * bz,
        ],
        axis=-1,
    )


def quat_from_axis_angle(axis: Sequence[float], angle_radians: float) -> np.ndarray:
    axis = np.asarray(axis, dtype=np.float64).reshape(-1)
    length = float(np.linalg.norm(axis))
    if length < EPS:
        raise ValueError("Rotation axis must not be a zero vector")
    axis = axis / length
    half = float(angle_radians) * 0.5
    s = math.sin(half)
    return np.array([axis[0] * s, axis[1] * s, axis[2] * s, math.cos(half)])


def quat_to_matrix(q: Sequence[float] | np.ndarray) -> np.ndarray:
    """Return the ``(4, 4)`` homogeneous rotation matrix for a unit quaternion."""
    x, y, z, w = (float(v) for v in quat_normalize(q))
    xx, yy, zz = x * x, y * y, z * z
    xy, xz, yz = x * y, x * z, y * z
    wx, wy, wz = w * x, w * y, w * z
    return np.array(
        [
            [1.0 - 2.0 * (yy + zz), 2.0 * (xy - wz), 2.0 * (xz + wy), 0.0],
            [2.0 * (xy + wz), 1.0 - 2.0 * (xx + zz), 2.0 * (yz - wx), 0.0],
            [2.0 * (xz - wy), 2.0 * (yz + wx), 1.0 - 2.0 * (xx + yy), 0.0],
            [0.0, 0.0, 0.0, 1.0],
        ],
        dtype=np.float64,
    )


def quat_from_matrix(m: np.ndarray) -> np.ndarray:
    """Shepperd's method: pick the largest component for numerical stability."""
    m = np.asarray(m, dtype=np.float64)
    if m.shape != (4, 4):
        raise ValueError("quat_from_matrix expects a 4x4 matrix")
    r = m[:3, :3]
    trace = r[0, 0] + r[1, 1] + r[2, 2]
    if trace > 0.0:
        s = math.sqrt(trace + 1.0) * 2.0
        w = 0.25 * s
        x = (r[2, 1] - r[1, 2]) / s
        y = (r[0, 2] - r[2, 0]) / s
        z = (r[1, 0] - r[0, 1]) / s
    elif r[0, 0] > r[1, 1] and r[0, 0] > r[2, 2]:
        s = math.sqrt(1.0 + r[0, 0] - r[1, 1] - r[2, 2]) * 2.0
        w = (r[2, 1] - r[1, 2]) / s
        x = 0.25 * s
        y = (r[0, 1] + r[1, 0]) / s
        z = (r[0, 2] + r[2, 0]) / s
    elif r[1, 1] > r[2, 2]:
        s = math.sqrt(1.0 + r[1, 1] - r[0, 0] - r[2, 2]) * 2.0
        w = (r[0, 2] - r[2, 0]) / s
        x = (r[0, 1] + r[1, 0]) / s
        y = 0.25 * s
        z = (r[1, 2] + r[2, 1]) / s
    else:
        s = math.sqrt(1.0 + r[2, 2] - r[0, 0] - r[1, 1]) * 2.0
        w = (r[1, 0] - r[0, 1]) / s
        x = (r[0, 2] + r[2, 0]) / s
        y = (r[1, 2] + r[2, 1]) / s
        z = 0.25 * s
    return quat_normalize(np.array([x, y, z, w], dtype=np.float64))


def quat_rotate_vector(q: Sequence[float] | np.ndarray, v: Sequence[float]) -> np.ndarray:
    """Rotate a 3-vector by a unit quaternion."""
    return quat_to_matrix(q)[:3, :3] @ np.asarray(v, dtype=np.float64).reshape(3)


def quat_angle_between(
    a: Sequence[float] | np.ndarray,
    b: Sequence[float] | np.ndarray,
) -> float:
    """Absolute geodesic angle between two rotations, in radians.

    ``q`` and ``-q`` describe the same rotation, so the dot product is made
    non-negative before the ``arccos``.  This is the number every accuracy
    claim in the validation report is expressed in.
    """
    qa = quat_normalize(a)
    qb = quat_normalize(b)
    dot = float(abs(qa @ qb))
    dot = min(1.0, max(-1.0, dot))
    return 2.0 * math.acos(dot)


def quat_angle_deg(
    a: Sequence[float] | np.ndarray,
    b: Sequence[float] | np.ndarray,
) -> float:
    """Smallest rotation angle in degrees between two orientations.

    Sign-agnostic: ``q`` and ``-q`` are the same rotation, so the dot product
    is taken in absolute value.
    """
    dot = abs(float(np.dot(quat_normalize(a), quat_normalize(b))))
    return math.degrees(2.0 * math.acos(min(1.0, max(-1.0, dot))))


def quat_slerp(
    a: Sequence[float] | np.ndarray,
    b: Sequence[float] | np.ndarray,
    t: float,
) -> np.ndarray:
    """Shortest-arc spherical interpolation between two rotations."""
    qa = quat_normalize(a)
    qb = quat_normalize(b)
    dot = float(qa @ qb)
    if dot < 0.0:
        qb = -qb
        dot = -dot
    dot = min(1.0, max(-1.0, dot))
    if dot > 1.0 - 1e-9:
        return quat_normalize(qa + t * (qb - qa))
    theta = math.acos(dot)
    sin_theta = math.sin(theta)
    s_a = math.sin((1.0 - t) * theta) / sin_theta
    s_b = math.sin(t * theta) / sin_theta
    return quat_normalize(s_a * qa + s_b * qb)


def euler_to_quat(
    rotation: Sequence[float],
    order: str = "XYZ",
) -> np.ndarray:
    """Convert ``[x, y, z]`` Euler angles in radians to a quaternion.

    ``order`` follows the FBX/Maya convention: the letters list the axes in
    *application* order, so ``"ZYX"`` means "rotate about Z, then Y, then X".
    ``rotation`` is always indexed by axis (``[X, Y, Z]``), never by position
    in ``order`` -- pairing the two positionally is what silently swaps X and
    Z for every non-``XYZ`` rig.
    """
    rotation = np.asarray(rotation, dtype=np.float64).reshape(3)
    if len(order) != 3 or set(order) != {"X", "Y", "Z"}:
        raise ValueError(f"Unsupported Euler order: {order!r}")
    axis_index = {"X": 0, "Y": 1, "Z": 2}
    result = np.array([0.0, 0.0, 0.0, 1.0], dtype=np.float64)
    for letter in order:
        result = quat_multiply(
            result,
            quat_from_axis_angle(
                _unit_axis(axis_index[letter]), float(rotation[axis_index[letter]])
            ),
        )
    return quat_normalize(result)


def euler_to_quat_many(
    rotation: np.ndarray,
    order: str = "XYZ",
) -> np.ndarray:
    """Vectorised :func:`euler_to_quat` for an ``(n, 3)`` stack of Euler angles.

    Identical maths, evaluated for every key at once: baking a clip needs one
    rotation per key per bone, and a Python loop over that is the single
    hottest path in the FBX stage.
    """
    rotation = np.atleast_2d(np.asarray(rotation, dtype=np.float64))
    if len(order) != 3 or set(order) != {"X", "Y", "Z"}:
        raise ValueError(f"Unsupported Euler order: {order!r}")
    axis_index = {"X": 0, "Y": 1, "Z": 2}
    result = np.zeros((rotation.shape[0], 4), dtype=np.float64)
    result[:, 3] = 1.0
    for letter in order:
        # ``rotation`` is indexed by axis, not by position in ``order``; see
        # :func:`euler_to_quat`.  The two differ whenever ``order != "XYZ"``.
        angle = rotation[:, axis_index[letter]]
        axis = np.zeros(3, dtype=np.float64)
        axis[axis_index[letter]] = 1.0
        half = angle * 0.5
        sin = np.sin(half)
        step = np.empty((rotation.shape[0], 4), dtype=np.float64)
        step[:, 0] = axis[0] * sin
        step[:, 1] = axis[1] * sin
        step[:, 2] = axis[2] * sin
        step[:, 3] = np.cos(half)
        result = quat_multiply_many(result, step)
    return quat_normalize_many(result)


def quat_normalize_many(quats: np.ndarray) -> np.ndarray:
    quats = np.atleast_2d(np.asarray(quats, dtype=np.float64))
    norms = np.linalg.norm(quats, axis=1, keepdims=True)
    norms[norms < 1e-12] = 1.0
    return quats / norms


def _unit_axis(index: int) -> list[float]:
    axis = [0.0, 0.0, 0.0]
    axis[index] = 1.0
    return axis


def quat_to_euler(q: Sequence[float] | np.ndarray, order: str = "XYZ") -> np.ndarray:
    """Inverse of :func:`euler_to_quat`; only used for diagnostics/UI."""
    x, y, z, w = (float(v) for v in quat_normalize(q))
    m = quat_to_matrix([x, y, z, w])
    if order == "XYZ":
        # R = Rz * Ry * Rx  (Z applied last with the column-vector convention)
        sy = -m[2, 0]
        sy = min(1.0, max(-1.0, sy))
        pitch = math.asin(sy)
        if abs(sy) < 1.0 - 1e-9:
            return np.array(
                [math.atan2(m[2, 1], m[2, 2]), pitch, math.atan2(m[1, 0], m[0, 0])]
            )
        return np.array([math.atan2(-m[1, 2], m[1, 1]), pitch, 0.0])
    if order == "ZYX":
        sy = m[0, 2]
        sy = min(1.0, max(-1.0, sy))
        pitch = math.asin(sy)
        if abs(sy) < 1.0 - 1e-9:
            return np.array(
                [math.atan2(m[0, 1], m[0, 0]), pitch, math.atan2(m[1, 2], m[2, 2])]
            )
        return np.array([0.0, pitch, math.atan2(-m[2, 1], m[1, 1])])
    raise ValueError(f"Unsupported Euler order: {order!r}")


# --------------------------------------------------------------------------- #
# matrices
# --------------------------------------------------------------------------- #
def mat_identity() -> np.ndarray:
    return np.eye(4, dtype=np.float64)


def mat_translation(t: Sequence[float]) -> np.ndarray:
    m = mat_identity()
    m[:3, 3] = np.asarray(t, dtype=np.float64).reshape(3)
    return m


def mat_scale(s: Sequence[float]) -> np.ndarray:
    m = mat_identity()
    s = np.asarray(s, dtype=np.float64).reshape(-1)
    if s.size == 1:
        s = np.repeat(s, 3)
    if s.size != 3:
        raise ValueError("Scale must have 1 or 3 components")
    m[0, 0], m[1, 1], m[2, 2] = s
    return m


def mat_rotation(q: Sequence[float] | np.ndarray) -> np.ndarray:
    return quat_to_matrix(q)


#: Backwards-compatible alias used by several call sites.
mat_from_quat = mat_rotation


def mat_from_trs(
    translation: Sequence[float],
    rotation: Sequence[float] | np.ndarray,
    scale: Sequence[float] | np.ndarray = (1.0, 1.0, 1.0),
) -> np.ndarray:
    """Compose ``T @ R @ S``."""
    return mat_translation(translation) @ mat_rotation(rotation) @ mat_scale(scale)


def mat_from_trs_many(
    translation: np.ndarray,
    rotation: np.ndarray,
    scale: np.ndarray | None = None,
) -> np.ndarray:
    """Batched ``T @ R @ S`` for ``(n, 3)`` / ``(n, 4)`` / ``(n, 3)`` stacks.

    Returns an ``(n, 4, 4)`` array of column-vector matrices.
    """
    translation = np.atleast_2d(np.asarray(translation, dtype=np.float64))
    rotation = np.atleast_2d(np.asarray(rotation, dtype=np.float64))
    count = max(translation.shape[0], rotation.shape[0])
    if scale is None:
        scale = np.ones((count, 3), dtype=np.float64)
    else:
        scale = np.atleast_2d(np.asarray(scale, dtype=np.float64))
    x, y, z, w = rotation[:, 0], rotation[:, 1], rotation[:, 2], rotation[:, 3]
    xx, yy, zz = x * x, y * y, z * z
    xy, xz, yz = x * y, x * z, y * z
    wx, wy, wz = w * x, w * y, w * z
    # T @ R @ S scales the *columns* of R, exactly as the scalar
    # ``mat_translation @ mat_rotation @ mat_scale`` does.
    out = np.empty((count, 4, 4), dtype=np.float64)
    out[:, :, 0] = np.stack(
        [1.0 - 2.0 * (yy + zz), 2.0 * (xy + wz), 2.0 * (xz - wy), np.zeros(count)],
        axis=-1,
    ) * scale[:, 0:1]
    out[:, :, 1] = np.stack(
        [2.0 * (xy - wz), 1.0 - 2.0 * (xx + zz), 2.0 * (yz + wx), np.zeros(count)],
        axis=-1,
    ) * scale[:, 1:2]
    out[:, :, 2] = np.stack(
        [2.0 * (xz + wy), 2.0 * (yz - wx), 1.0 - 2.0 * (xx + yy), np.zeros(count)],
        axis=-1,
    ) * scale[:, 2:3]
    out[:, :3, 3] = translation[:, :3]
    out[:, 3, 3] = 1.0
    return out


def mat_mul(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    return np.asarray(a, dtype=np.float64) @ np.asarray(b, dtype=np.float64)


def mat_transpose(m: np.ndarray) -> np.ndarray:
    return np.asarray(m, dtype=np.float64).T.copy()


def mat_inverse(m: np.ndarray) -> np.ndarray:
    """Full 4x4 inverse (no rigid-transform assumption)."""
    return np.linalg.inv(np.asarray(m, dtype=np.float64))


def mat_xform_point(m: np.ndarray, p: Sequence[float]) -> np.ndarray:
    m = np.asarray(m, dtype=np.float64)
    p = np.asarray(p, dtype=np.float64).reshape(3)
    return m[:3, :3] @ p + m[:3, 3]


def mat_xform_vector(m: np.ndarray, v: Sequence[float]) -> np.ndarray:
    m = np.asarray(m, dtype=np.float64)
    v = np.asarray(v, dtype=np.float64).reshape(3)
    return m[:3, :3] @ v


def orthonormalize(m: np.ndarray) -> np.ndarray:
    """Return the closest orthonormal rotation matrix (modified Gram-Schmidt)."""
    r = np.asarray(m, dtype=np.float64)[:3, :3].copy()
    for col in range(3):
        for prev in range(col):
            r[:, col] -= float(r[:, prev] @ r[:, col]) * r[:, prev]
        norm = float(np.linalg.norm(r[:, col]))
        if norm < EPS:
            raise ValueError("Degenerate basis while orthonormalising")
        r[:, col] /= norm
    if float(np.linalg.det(r)) < 0.0:
        r[:, 0] = -r[:, 0]
    return r


def mat_decompose(m: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Split a matrix into ``(translation, quaternion, scale)``.

    Raises ``ValueError`` when the matrix is sheared, because a silently
    wrong decomposition is exactly the class of bug this project must not
    ship.
    """
    m = np.asarray(m, dtype=np.float64)
    if m.shape != (4, 4):
        raise ValueError("mat_decompose expects a 4x4 matrix")
    translation = m[:3, 3].copy()
    basis = m[:3, :3]
    lengths = np.linalg.norm(basis, axis=0)
    if np.any(lengths < EPS):
        raise ValueError("Cannot decompose a matrix with a zero-length axis")
    normalized = basis / lengths
    for a in range(3):
        for b in range(a + 1, 3):
            if abs(float(normalized[:, a] @ normalized[:, b])) > 1e-6:
                raise ValueError("Cannot decompose a sheared matrix")
    if float(np.linalg.det(normalized)) < 0.0:
        # Negative determinant: keep the scale sign rather than the rotation.
        lengths = lengths.copy()
        lengths[0] = -lengths[0]
        normalized = basis / lengths
    rotation = quat_from_matrix(
        np.block(
            [
                [normalized, np.zeros((3, 1), dtype=np.float64)],
                [np.zeros((1, 3), dtype=np.float64), np.ones((1, 1), dtype=np.float64)],
            ]
        )
    )
    return translation, rotation, lengths


#: Backwards-compatible alias.
mat_to_quat = quat_from_matrix


def remap_unit_interval(x: float, in_min: float, in_max: float,
                        out_min: float, out_max: float) -> float:
    """Linear remap, guarded against a degenerate source range."""
    if in_max - in_min == 0.0:
        return out_min
    t = (x - in_min) / (in_max - in_min)
    return out_min + t * (out_max - out_min)


def quats_equal(a: Sequence[float], b: Sequence[float], tolerance: float = 1e-7) -> bool:
    """Sign-insensitive quaternion comparison."""
    qa, qb = quat_normalize(a), quat_normalize(b)
    return bool(
        np.allclose(qa, qb, atol=tolerance) or np.allclose(qa, -qb, atol=tolerance)
    )


def mean_quaternion(quats: Iterable[Sequence[float]]) -> np.ndarray:
    """Average a set of rotations, as close to the geodesic mean as is cheap.

    Quaternions double-cover SO(3): ``q`` and ``-q`` are the same rotation.
    Averaging their components directly therefore cancels a perfectly
    consistent set to near zero, and the "mean" comes out as whatever the
    normalisation leaves behind.  So every quaternion is first flipped into
    the hemisphere of the running mean before being averaged, and the mean is
    iterated until it stops moving.

    This is the *chordal* mean, not the geodesic one.  For two rotations it
    is biased: the midpoint of 90 degrees about X and 90 degrees about Z
    comes out at 70.5 degrees where the geodesic answer is 60.  That matters
    for anyone reading a diagnostic built on this, and the bias is stated
    here rather than left for them to discover.  It is accurate enough for
    the use it has -- deciding whether a ped is broadly upright -- and it
    never returns something outside the span of the inputs, which a
    component-wise average can.

    The sign-flipping is what makes it correct in the cases that matter: a
    set of keys that cross the double cover averages to the rotation they
    represent instead of to its negation.
    """
    normalised = [quat_normalize(q) for q in quats]
    if not normalised:
        raise ValueError("mean_quaternion needs at least one rotation")
    if len(normalised) == 1:
        return normalised[0]

    reference = normalised[0]
    aligned = [
        q if float(np.dot(q, reference)) >= 0.0 else -q for q in normalised
    ]
    current = np.mean(np.stack(aligned, axis=0), axis=0)
    current = quat_normalize(current)
    for _ in range(32):
        aligned = [
            q if float(np.dot(q, current)) >= 0.0 else -q for q in normalised
        ]
        updated = quat_normalize(np.mean(np.stack(aligned, axis=0), axis=0))
        if np.allclose(updated, current, atol=1e-12):
            return updated
        current = updated
    return current
