"""Drawing a skeleton, for the side-by-side preview.

Two things this has to get right.

It draws the *ped's* bone hierarchy for both sides, not the FBX's.  The two
rigs have different bone counts and different names; drawing each in its own
hierarchy would make a correct conversion look wrong and an incorrect one
look right, because the shapes would not correspond at all.

It draws a *posed* skeleton, from the bind pose and the rotations on top of
it, with the same camera on both sides.  A preview that shows the source's
skeleton and the output's skeleton in isolation cannot show that the arm is
rotating about the wrong axis, which is the failure worth catching.
"""
from __future__ import annotations

import math

import numpy as np
from PySide6.QtCore import QPointF, QRectF, Qt, QTimer
from PySide6.QtGui import QBrush, QColor, QFont, QPainter, QPen, QPolygonF
from PySide6.QtWidgets import QWidget

from . import pipeline

#: The ped's own bone rest positions, in the order the HAnim ids are used.
#: Filled in by `set_reference` from the loaded DFF; these are only the
#: fallback colours and proportions.
DEFAULT_COLOURS = {
    "left": QColor("#3f7fd0"),
    "right": QColor("#d0763f"),
    "centre": QColor("#8f8f8f"),
    "selected": QColor("#e8c547"),
}


def quat_to_matrix(q) -> np.ndarray:
    """XYZW quaternion to a 3x3 rotation matrix."""
    x, y, z, w = (float(v) for v in q)
    norm = math.sqrt(x * x + y * y + z * z + w * w)
    if norm < 1e-12:
        return np.eye(3)
    x, y, z, w = x / norm, y / norm, z / norm, w / norm
    return np.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - z * w),     2 * (x * z + y * w)],
        [2 * (x * y + z * w),     1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
        [2 * (x * z - y * w),     2 * (y * z + x * w),     1 - 2 * (x * x + y * y)],
    ])


class SkeletonView(QWidget):
    """One posed skeleton, drawn from a fixed camera."""

    def __init__(self, label: str, parent=None):
        super().__init__(parent)
        self.label = label
        self.setMinimumHeight(320)
        self.setAutoFillBackground(True)

        #: bone id -> (rest world position, parent bone id, side)
        self.reference: dict[int, tuple[np.ndarray, int, str]] = {}
        self.rotations: dict[int, np.ndarray] = {}
        self.key_times: list[float] = []
        self._index = 0
        self._playing = False
        self._fps = 30.0
        self._last = 0.0

        self._timer = QTimer(self)
        self._timer.timeout.connect(self._tick)

    # -- data ------------------------------------------------------------ #
    def set_reference(self, reference: dict, key_times: list[float]) -> None:
        """Set the rest pose and the times the keys happen at."""
        self.reference = reference
        self.key_times = list(key_times)
        self._index = 0
        self.update()

    def set_rotations(self, rotations: dict) -> None:
        self.rotations = rotations
        self.update()

    def key_count(self) -> int:
        return max(1, len(self.key_times))

    def key_index(self) -> int:
        return self._index

    def set_key(self, index: int) -> None:
        self._index = min(max(0, index), self.key_count() - 1)
        self.update()

    def set_time(self, seconds: float) -> None:
        if not self.key_times:
            return
        # Nearest key rather than an interpolated pose: the IFP stores
        # discrete keys and interpolating between them here would show a
        # smoothness the file does not contain.
        nearest = min(range(len(self.key_times)),
                      key=lambda i: abs(self.key_times[i] - seconds))
        self.set_key(nearest)

    def toggle_play(self) -> bool:
        self._playing = not self._playing
        if self._playing:
            self._last = 0.0
            self._timer.start(16)
        else:
            self._timer.stop()
        self.update()
        return self._playing

    def set_playing(self, playing: bool) -> None:
        if playing != self._playing:
            self.toggle_play()

    def _tick(self) -> None:
        import time

        now = time.monotonic()
        if self._last:
            # Advance by real elapsed time, so the preview plays at the
            # source's speed regardless of how fast the widget repaints.
            self._index = int(self._index + (now - self._last) * self._fps)
        self._last = now
        if self.key_times and self._index >= len(self.key_times) - 1:
            self._index = 0
        self.set_key(self._index)

    # -- posing ---------------------------------------------------------- #
    def posed(self) -> dict[int, np.ndarray]:
        """World positions of every bone with the current rotations applied."""
        posed: dict[int, np.ndarray] = {}
        order = sorted(self.reference,
                       key=lambda b: self._depth(b))
        for bone_id in order:
            rest, parent, _ = self.reference[bone_id]
            local = rest.copy()
            rotation = self.rotations.get(bone_id)
            if rotation is not None:
                local = quat_to_matrix(rotation) @ rest
            if parent in posed:
                posed[bone_id] = posed[parent] + local
            else:
                posed[bone_id] = local
        return posed

    def _depth(self, bone_id: int) -> int:
        depth, cursor = 0, bone_id
        while cursor in self.reference:
            _, parent, _ = self.reference[cursor]
            if parent < 0 or parent not in self.reference:
                break
            cursor, depth = parent, depth + 1
        return depth

    # -- drawing --------------------------------------------------------- #
    def paintEvent(self, event) -> None:
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing)
        painter.fillRect(self.rect(), QColor("#1c1c1c"))

        font = QFont()
        font.setPointSize(10)
        painter.setFont(font)
        painter.setPen(QPen(QColor("#c8c8c8")))
        painter.drawText(10, 20, self.label)
        painter.setPen(QPen(QColor("#6a6a6a")))
        painter.drawText(10, 36, f"key {self._index + 1} of {self.key_count()}")

        if not self.reference:
            painter.setPen(QPen(QColor("#777777")))
            painter.drawText(self.rect(), Qt.AlignCenter,
                             "nothing to show")
            return

        posed = self.posed()
        points = self._project(posed.values())

        painter.setPen(QPen(QColor("#9a9a9a"), 1.4))
        for bone_id, (rest, parent, side) in self.reference.items():
            if parent not in points or bone_id not in points:
                continue
            colour = DEFAULT_COLOURS.get(side, DEFAULT_COLOURS["centre"])
            painter.setPen(QPen(colour.lighter(130), 1.6))
            painter.drawLine(points[parent], points[bone_id])

        # The root, drawn as a small ring so the view has an orientation.
        root = min(self.reference)
        if root in points:
            painter.setBrush(QBrush(QColor("#e8c547")))
            painter.setPen(Qt.NoPen)
            painter.drawEllipse(points[root], 3.5, 3.5)

    def _project(self, positions) -> dict[int, QPointF]:
        """Fit the skeleton to the widget with a fixed camera.

        The scale comes from the rest pose so the two views are directly
        comparable: a conversion that leaves the character twice the size
        should look twice the size on both sides, not be auto-fitted away.
        """
        if not self.reference:
            return {}
        rest = np.array([r for r, _, _ in self.reference.values()])
        span = float(np.max(np.ptp(rest, axis=0)).max()) or 1.0
        current = np.array(list(positions))

        width, height = self.width(), self.height()
        margin = 46
        scale = min(width - 2 * margin, height - 2 * margin - 20) / span

        centre_x = float(np.mean(rest[:, 0]))
        centre_y = float(np.mean(rest[:, 2]))     # GTA is Z-up
        origin = QPointF(width / 2, height / 2 - 10)

        return {
            bone_id: QPointF(
                origin.x() + (position[0] - centre_x) * scale,
                # Screen y grows downward, GTA z grows upward.
                origin.y() - (position[2] - centre_y) * scale,
            )
            for bone_id, position in zip(self.reference, current)
        }


def build_reference(target: dict) -> dict[int, tuple[np.ndarray, int, str]]:
    """The ped's rest pose, keyed by HAnim id, with parents resolved.

    `target` is what `pipeline.load_target` returned.  Parents are looked up
    through the bone list rather than assumed to be the id below, because in
    a DFF the parent of bone 5 is bone 4, the parent of bone 2 is bone 1,
    and the root's parent is nothing at all.

    Positions come from the DFF's own `bind_world_gta` matrices, so the
    preview shows the ped the user actually picked rather than a built-in
    standing pose that happens to be about the right height.
    """
    reference: dict[int, tuple[np.ndarray, int, str]] = {}
    by_index = target["bones"]
    for index, bone in enumerate(by_index):
        if bone["hanim_id"] is None or bone["hanim_id"] in reference:
            continue
        parent_id = -1
        for candidate in by_index:
            if (candidate["tag"] is not None and candidate["tag"] == bone["parent"]
                    and candidate["hanim_id"] is not None):
                parent_id = candidate["hanim_id"]
                break
        side = bone["side"] if bone["side"] in DEFAULT_COLOURS else "centre"
        reference[bone["hanim_id"]] = (
            np.asarray(bone["rest"], dtype=float), parent_id, side)
    return reference


def reference_from_animation(animation) -> tuple[dict, list[float]]:
    """The reference pose and key times of a written animation.

    The bind translation of each track is the bone's offset from its parent,
    so placing those along the ped's real hierarchy gives a posed skeleton
    without needing the source rig -- which is the point: the preview shows
    the file that was written, not a re-derivation of it.
    """
    reference: dict[int, tuple[np.ndarray, int, str]] = {}
    times: list[float] = []
    for track in animation.tracks:
        if not track.times_s:
            continue
        times = max(times, track.times_s) if not times else times
        offset = (np.asarray(track.bind_translation, dtype=float)
                  if getattr(track, "bind_translation", None) is not None
                  else np.zeros(3))
        reference[track.bone_id] = (offset, -1, "centre")
    # Times are unioned across tracks and sorted, so the scrubber lands on
    # keys that exist in the file rather than in any one bone.
    merged = sorted({round(t, 5) for track in animation.tracks
                     for t in track.times_s})
    return reference, merged


def rotations_at(animation, key: int) -> dict[int, np.ndarray]:
    """Every bone's stored rotation at one key index."""
    out: dict[int, np.ndarray] = {}
    for track in animation.tracks:
        if not track.times_s:
            continue
        index = min(max(0, key), len(track.times_s) - 1)
        if index < len(track.rotations):
            out[track.bone_id] = quat_to_matrix(track.rotations[index])
    return out
