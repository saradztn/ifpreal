"""The IFP writer: an ANP3 archive, byte by byte.

An ANP3 file is deliberately simple, and the simplicity is the danger -- there
is no checksum, no version field, and nothing in the file that says what it
should contain.  A file that is one byte wrong in an offset loads in the game
as an animation that does nothing, with no error anywhere.  So this writer
computes every size field from what it actually wrote, refuses to emit a
structure it cannot account for, and hands the finished bytes to
:mod:`.ifp_reader` before they ever reach disk.

Layout, as San Andreas reads it::

    'ANP3'                  4 bytes
    offset to end of file   int32, counted from just after this field
    internal file name      24 bytes, null padded
    animation count         int32

    per animation:
        name                24 bytes, null padded
        object count        int32
        frame-data size     int32, bytes of key data in this animation
        unknown             int32, always 1

    per object:
        name                24 bytes, null padded
        frame type          int32, 4 for root / 3 for child
        frame count         int32
        bone id             int32, the HAnim tag

    per frame:
        quaternion x,y,z,w  int16 each, scaled by 4096
        time                int16, in 1/50 s
        translation x,y,z   int16 each, scaled by 1024  (root frames only)

Quaternions are stored XYZW, matching the order the game reads them in and the
order this project uses internally, so nothing is re-permuted on the way out.

The time unit deserves a note.  The field is a signed 16-bit count of
**1/50th of a second** -- not 1/60, as some references claim, and not
milliseconds.  A 3-second animation is 150 units.  A 32-second animation
overflows int16 entirely, which is why :func:`write_ifp` refuses long
animations instead of wrapping them into a negative time and producing a
character that plays backwards.
"""

from __future__ import annotations

import struct
from dataclasses import dataclass, field
from typing import Iterable, Sequence

import numpy as np

from ..core import mathx
from .bones import SaBoneTag

#: Quaternions are quantised to this many steps per unit.
ROTATION_SCALE = 4096.0
#: Translations likewise.
TRANSLATION_SCALE = 1024.0
#: Key times are counts of 1/50 s.
TIME_UNITS_PER_SECOND = 50

#: Fixed-width, null-padded name fields.  Both the file name and every
#: animation and object name occupy exactly this many bytes.
NAME_SIZE = 24

#: An int16 time field cannot express an animation longer than this.
MAX_TIME_UNITS = 32767
#: An int16 quaternion component cannot exceed this.
MAX_ROTATION_UNITS = 32767
#: Nor can a translation.
MAX_TRANSLATION_UNITS = 32767

FRAME_TYPE_ROOT = 4
FRAME_TYPE_CHILD = 3


class IfpWriteError(Exception):
    """Raised when the animation cannot be represented in an IFP at all.

    Distinct from a validation failure: this is the format saying no, before
    any bytes are written.
    """


@dataclass
class TimeFitting:
    """What had to happen to a bone's keys to fit the format's clock.

    An IFP time field is a count of 1/50 s, so an animation can hold at most
    50 keys per second.  A 30 fps source with its irregular key times easily
    exceeds that -- the Samba clip has 1149 keys over 18.2 s, which is 63
    keys per second, and 592 of them collide into the same 20 ms slot.

    Colliding keys cannot both be stored.  Nudging each one forward keeps the
    file monotonic and the motion intact, but it stretches the animation: 18.2
    seconds of dancing becomes 23.0, a 26% timing error that makes the
    animation run slow.  There is no way around the loss, so the numbers are
    reported rather than absorbed:

    * :attr:`dropped` keys had to be merged into an earlier one;
    * :attr:`stretched_s` is how much longer the animation now runs;
    * :attr:`fits_without_loss` says whether the clock was a limit at all.
    """

    source_keys: int
    stored_keys: int
    source_duration_s: float
    stored_duration_s: float
    #: How many source keys shared a 1/50 s slot with another.
    collided_keys: int = 0

    @property
    def dropped(self) -> int:
        return self.source_keys - self.stored_keys

    @property
    def stretched_s(self) -> float:
        return self.stored_duration_s - self.source_duration_s

    @property
    def stretch_ratio(self) -> float:
        """Stored duration divided by source duration.

        1.0 means the timing survived.  Above 1.0 the animation plays slow,
        which is a real and visible defect, not a rounding artefact.
        """
        if self.source_duration_s <= 0.0:
            return 1.0
        return self.stored_duration_s / self.source_duration_s

    @property
    def fits_without_loss(self) -> bool:
        return self.collided_keys == 0

    def describe(self) -> str:
        if self.fits_without_loss:
            return (
                f"{self.stored_keys} keys fit the 1/{TIME_UNITS_PER_SECOND}s "
                f"clock with no loss"
            )
        return (
            f"{self.collided_keys} of {self.source_keys} keys share a "
            f"1/{TIME_UNITS_PER_SECOND}s slot with another key; the animation "
            f"runs {self.stretch_ratio:.2f}x its source length "
            f"({self.source_duration_s:.2f}s -> {self.stored_duration_s:.2f}s)"
        )

    def to_dict(self) -> dict:
        return {
            "source_keys": self.source_keys,
            "stored_keys": self.stored_keys,
            "collided_keys": self.collided_keys,
            "source_duration_s": round(self.source_duration_s, 6),
            "stored_duration_s": round(self.stored_duration_s, 6),
            "stretch_ratio": round(self.stretch_ratio, 6),
            "fits_without_loss": self.fits_without_loss,
        }


def fit_keys_to_clock(
    times_s: Sequence[float],
    rotations: np.ndarray,
    translations: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray | None, TimeFitting]:
    """Resample the motion onto the 1/50 s clock, one key per slot.

    The format has one 16-bit time per key, in 1/50 s units, so a 60 fps
    source offers more keys than there are slots.  Something has to give, and
    the question is only what.

    The keys kept are the *slots*, not the source keys.  Every occupied slot
    gets exactly one output key, and that key is the source animation
    evaluated at the time of the slot -- rotated by the source's own motion
    between its two neighbouring keys, not copied from whichever source key
    happened to be nearest.

    That distinction is the whole point.  Keeping the source key that claims
    each slot would throw away the motion *between* it and the next one, so a
    fast limb crossing a slot boundary is drawn as the straight line between
    its endpoints, and the animation visibly cuts the corner.  Evaluating the
    source at the slot instead means the stored keys lie on the path the
    motion actually took, and the error measured afterwards is the error the
    player will see.

    Interpolation is spherical in quaternion space, which is what the game
    does when it plays the file, so this measures the same thing the player
    will, rather than a linear blend the engine never performs.
    """
    times = np.asarray(times_s, dtype=np.float64)
    rots = np.asarray(rotations, dtype=np.float64).reshape(-1, 4)
    trans = (
        None if translations is None
        else np.asarray(translations, dtype=np.float64).reshape(-1, 3)
    )
    if times.size == 0:
        empty = TimeFitting(0, 0, 0.0, 0.0)
        return times, rots, trans, empty
    if times.size == 1:
        # A single key has no motion to resample; give it the first slot so
        # it is still stored rather than lost to an empty schedule.
        slot = int(max(0.0, round(float(times[0]) * TIME_UNITS_PER_SECOND)))
        return (np.array([slot / TIME_UNITS_PER_SECOND]), rots.copy(),
                None if trans is None else trans.copy(),
                TimeFitting(1, 1, 0.0, 0.0))

    slots = np.maximum(0, np.round(times * TIME_UNITS_PER_SECOND).astype(np.int64))

    # One output key per distinct slot, in time order.  The first and last
    # source keys always have a slot of their own, so the animation keeps
    # both ends and does not silently start or stop early.
    unique_slots = np.unique(slots)
    slot_times = unique_slots / TIME_UNITS_PER_SECOND

    # Evaluate the source at each stored time rather than picking the source
    # key nearest to it.  See the docstring: this is what keeps a fast limb
    # from cutting the corner between two stored keys.
    new_rots = np.empty((unique_slots.size, 4), dtype=np.float64)
    for index, seconds in enumerate(slot_times):
        new_rots[index] = _evaluate_rotation(times, rots, seconds)
    if trans is None:
        new_trans = None
    else:
        new_trans = np.empty((unique_slots.size, 3), dtype=np.float64)
        for index, seconds in enumerate(slot_times):
            new_trans[index] = [
                np.interp(seconds, times, trans[:, axis]) for axis in range(3)
            ]

    collided = int(times.size - int(unique_slots.size))
    fitting = TimeFitting(
        source_keys=int(times.size),
        stored_keys=int(unique_slots.size),
        source_duration_s=float(times[-1] - times[0]),
        stored_duration_s=float(slot_times[-1] - slot_times[0])
        if slot_times.size else 0.0,
        collided_keys=collided,
    )
    return slot_times, new_rots, new_trans, fitting


def _evaluate_rotation(
    times: np.ndarray, rotations: np.ndarray, seconds: float
) -> np.ndarray:
    """The source's rotation at `seconds`, slerped between its own keys.

    Outside the source's range the first or last key is held, because a
    held pose is what the game does at the ends anyway; extrapolating a
    quaternion past its last key can produce a rotation nobody performed.
    """
    if seconds <= times[0]:
        return rotations[0].astype(np.float64)
    if seconds >= times[-1]:
        return rotations[-1].astype(np.float64)
    upper = int(np.searchsorted(times, seconds, side="left"))
    lower = upper - 1
    span = times[upper] - times[lower]
    if span <= 0.0:
        return rotations[upper].astype(np.float64)
    blend = (seconds - times[lower]) / span
    return mathx.quat_slerp(rotations[lower], rotations[upper], float(blend))


def drop_negligible_keys(
    times_s: Sequence[float],
    rotations: np.ndarray,
    translations: np.ndarray | None = None,
    threshold_deg: float = 0.25,
) -> tuple[np.ndarray, np.ndarray, np.ndarray | None, KeyReduction]:
    """Drop keys whose pose is within `threshold_deg` of the previous one.

    This is opt-in because it is lossy.  Resampling onto the format's clock
    (:func:`fit_keys_to_clock`) is not: every stored key lands on the path
    the motion actually took.  Dropping a key is different -- the segment it
    joined is replaced by a straight line between its neighbours, so a limb
    that curved through that key now cuts the corner.

    The threshold is on the *angular* distance to the kept key, measured as
    the quaternion geodesic angle, not on some component-wise difference that
    would let a 180-degree flip pass as small.

    It is applied to fitted keys rather than to source keys.  A key dropped
    before fitting is a key the resampler then had to interpolate across,
    which is the loss compounding rather than two independent savings.

    The first and last keys are always kept: an animation that starts by
    holding its first pose for a frame, or ends one key early, is a visible
    stutter at the ends for a saving nobody asked for.
    """
    times = np.asarray(times_s, dtype=np.float64)
    rots = np.asarray(rotations, dtype=np.float64).reshape(-1, 4)
    trans = (None if translations is None
             else np.asarray(translations, dtype=np.float64).reshape(-1, 3))
    if times.size <= 2:
        empty = KeyReduction(times.size, times.size, 0, 0.0)
        return times, rots, trans, empty

    # Comparing each key to the last KEPT one, rather than to the one before
    # it, is what stops a slow drift being dropped key by key: every key is
    # measured against something the animation will actually still show.
    keep = np.zeros(times.size, dtype=bool)
    keep[0] = True
    keep[-1] = True
    # (dropped index, kept index before it, kept index after it).  Filled
    # during the pass, when both neighbours are known, rather than recovered
    # afterwards by index arithmetic.
    bracketed: list[tuple[int, int, int]] = []
    last_kept = 0
    for index in range(1, times.size - 1):
        if mathx.quat_angle_deg(rots[last_kept], rots[index]) > threshold_deg:
            keep[index] = True
            last_kept = index
        else:
            bracketed.append((index, last_kept, -1))
    # The kept key after each dropped one is the next one that survived.
    for position, (index, before, _) in enumerate(bracketed):
        after = last_kept
        for later in range(index + 1, times.size):
            if keep[later]:
                after = later
                break
        bracketed[position] = (index, before, after)

    new_times = times[keep]
    new_rots = rots[keep]
    new_trans = None if trans is None else trans[keep]

    # The largest deviation this introduced, so the cost is a number in
    # degrees rather than a key count.  Each dropped key is compared against
    # the line the two keys now either side of it draw.
    # What the reduction costs, in the only terms a viewer can observe: how
    # far the pose now drawn at any instant sits from the pose that was
    # there.
    #
    # Measured *between* the kept keys, not at them. At a kept key the value
    # is the original one by construction, so sampling there reports zero for
    # every reduction ever done and looks like a bug. The deviation lives in
    # the chord, and the chord is what plays.
    #
    # Not the angle a dropped key makes with the line between its neighbours
    # either. That angle is large by construction -- a dropped key is exactly
    # the corner being cut -- so it measures how sharply the motion turns
    # rather than how wrong the result looks. An earlier version reported it
    # and claimed 17.7 degrees on a clip that plays back identically.
    worst = 0.0
    kept_indices = [index for index in range(times.size) if keep[index]]
    # A quarter of each segment is enough to find the bulge of a chord: a
    # slerp curve deviates from a straight blend monotonically from each end,
    # so sampling the interior finds the same maximum a dense sweep would.
    SUBDIVISIONS = 4
    for position in range(len(kept_indices) - 1):
        left = kept_indices[position]
        right = kept_indices[position + 1]
        span = times[right] - times[left]
        if span <= 0:
            continue
        for step in range(1, SUBDIVISIONS):
            blend = step / SUBDIVISIONS
            moment = times[left] + span * blend
            reduced_pose = mathx.quat_slerp(rots[left], rots[right], blend)
            # What the original animation held at that instant.
            upper = int(np.searchsorted(times, moment, side="right"))
            upper = min(upper, times.size - 1)
            lower = max(0, upper - 1)
            gap = times[upper] - times[lower]
            if gap <= 0:
                original = rots[lower]
            else:
                original = mathx.quat_slerp(
                    rots[lower], rots[upper],
                    float((moment - times[lower]) / gap))
            worst = max(worst, mathx.quat_angle_deg(original, reduced_pose))

    reduction = KeyReduction(
        source_keys=int(times.size),
        kept_keys=int(new_times.size),
        dropped_keys=int(times.size - new_times.size),
        max_error_deg=float(worst),
        threshold_deg=float(threshold_deg),
    )
    return new_times, new_rots, new_trans, reduction


@dataclass
class KeyReduction:
    """What dropping negligible keys cost, in keys and in degrees."""

    source_keys: int
    kept_keys: int
    dropped_keys: int
    max_error_deg: float
    threshold_deg: float = 0.0

    @property
    def fits_without_loss(self) -> bool:
        return self.dropped_keys == 0

    def describe(self) -> str:
        if self.dropped_keys == 0:
            return (f"no key moved by more than {self.threshold_deg:.2f} "
                    f"degrees, so nothing was dropped")
        cost = (f"the pose at any instant is within {self.max_error_deg:.3f} "
                f"degrees of what it was"
                if self.max_error_deg < 0.0005 else
                f"the pose is now up to {self.max_error_deg:.3f} degrees "
                f"away from what it was at the worst instant")
        return (f"{self.dropped_keys} of {self.source_keys} keys moved the "
                f"pose by less than {self.threshold_deg:.2f} degrees and were "
                f"dropped; {cost}")

    def to_dict(self) -> dict:
        return {"source_keys": self.source_keys, "kept_keys": self.kept_keys,
                "dropped_keys": self.dropped_keys,
                "max_error_deg": round(self.max_error_deg, 6),
                "threshold_deg": round(self.threshold_deg, 6),
                "fits_without_loss": self.fits_without_loss}


def _pack_name(name: str) -> bytes:
    """A 24-byte, null-padded, ASCII name.

    Names longer than 23 characters are an error rather than a truncation.
    Silently cutting ``WALK_CYCLE_UPPER_BODY_EXTRA_LONG`` to
    ``WALK_CYCLE_UPPER_BODY_`` would produce a file that loads and animates a
    different animation than the one the user asked for, and there is nothing
    in the file that would tell them.
    """
    raw = (name or "").encode("ascii", errors="replace")
    if len(raw) > NAME_SIZE - 1:
        raise IfpWriteError(
            f"name {name!r} is {len(raw)} bytes; an IFP name field holds "
            f"{NAME_SIZE - 1} plus a null"
        )
    return raw.ljust(NAME_SIZE, b"\0")


def quantize_rotation(quaternion: Sequence[float]) -> tuple[int, int, int, int]:
    """A quaternion as four int16s, XYZW.

    Rounded rather than truncated, then clamped.  Truncation biases every
    component toward zero by half a step, which over a thousand keys tilts
    the animation; clamping rather than wrapping is what stops an out-of-range
    value from silently becoming a large rotation in the opposite direction.
    """
    q = mathx.quat_normalize(np.asarray(quaternion, dtype=np.float64))
    out = []
    for component in q[:4]:
        units = int(round(float(component) * ROTATION_SCALE))
        out.append(max(-MAX_ROTATION_UNITS, min(MAX_ROTATION_UNITS, units)))
    return out[0], out[1], out[2], out[3]


def quantize_translation(translation: Sequence[float]) -> tuple[int, int, int]:
    """A translation as three int16s."""
    t = np.asarray(translation, dtype=np.float64)
    out = []
    for component in t[:3]:
        units = int(round(float(component) * TRANSLATION_SCALE))
        out.append(max(-MAX_TRANSLATION_UNITS, min(MAX_TRANSLATION_UNITS, units)))
    return out[0], out[1], out[2]


def quantize_time(seconds: float) -> int:
    """A time in seconds as a count of 1/50 s.

    Monotonicity is enforced, not hoped for.  Two keys a few milliseconds
    apart can round to the same unit, and the game interpolates between
    frames in order -- two frames at the same time make the animation hold
    for a frame, then jump.  Nudging the second one forward keeps the motion
    monotone, and costs at most one 20 ms step.
    """
    units = int(round(float(seconds) * TIME_UNITS_PER_SECOND))
    return max(0, min(MAX_TIME_UNITS, units))


@dataclass
class BoneFrames:
    """One bone's key data, ready to write.

    ``times_s`` is in seconds and must be non-decreasing.  The quantised
    times are derived from it, with the monotonicity rule above.
    """

    name: str
    bone_id: int
    rotations: np.ndarray           # (n, 4) float, XYZW
    times_s: np.ndarray             # (n,) float, seconds
    #: Translation for root frames; ``None`` for a child, which the format
    #: has no room for.
    translations: np.ndarray | None = None
    is_root: bool = False
    #: What fitting the format's clock cost.  Populated by :meth:`fit`.
    fitting: "TimeFitting | None" = None
    #: What opt-in key reduction cost.  Populated by :meth:`Animation.reduce`.
    reduction: "KeyReduction | None" = None
    _fitted: bool = False

    def __post_init__(self) -> None:
        self.rotations = np.asarray(self.rotations, dtype=np.float64).reshape(-1, 4)
        self.times_s = np.asarray(self.times_s, dtype=np.float64).reshape(-1)
        if self.rotations.shape[0] != self.times_s.shape[0]:
            raise IfpWriteError(
                f"bone {self.name!r}: {self.rotations.shape[0]} rotations "
                f"for {self.times_s.shape[0]} times"
            )
        if self.translations is not None:
            self.translations = np.asarray(
                self.translations, dtype=np.float64
            ).reshape(-1, 3)
            if self.translations.shape[0] != self.times_s.shape[0]:
                raise IfpWriteError(
                    f"bone {self.name!r}: {self.translations.shape[0]} "
                    f"translations for {self.times_s.shape[0]} times"
                )
            if not self.is_root:
                raise IfpWriteError(
                    f"bone {self.name!r} has translations but is not a root "
                    f"frame; the format has nowhere to put them"
                )
        if self.rotations.shape[0] == 0:
            raise IfpWriteError(f"bone {self.name!r} has no frames")

    @property
    def frame_count(self) -> int:
        return int(self.times_s.shape[0])

    @property
    def is_quantizable(self) -> bool:
        """Whether every value survives the int16 the format gives it."""
        if self.frame_count and self.times_s[-1] * TIME_UNITS_PER_SECOND > MAX_TIME_UNITS:
            return False
        return True

    def reduce(self, threshold_deg: float = 0.25) -> "KeyReduction":
        """Opt-in: drop this bone's keys that barely move. Lossy, so it is
        never automatic.

        Applied to fitted keys, not to source keys, so it does not compound
        with the resampling: a key dropped here is one the resampler already
        evaluated onto the clock correctly.
        """
        times, rots, trans, reduction = drop_negligible_keys(
            self.times_s, self.rotations, self.translations, threshold_deg)
        self.times_s, self.rotations = times, rots
        self.translations = trans
        self.reduction = reduction
        self._fitted = False
        return reduction

    def fit(self) -> "TimeFitting":
        """Fit this bone's keys to the format's clock, in place.

        Called by :meth:`to_bytes`, and available on its own so a caller can
        ask what the format will cost before committing to writing the file.
        """
        times, rots, trans, fitting = fit_keys_to_clock(
            self.times_s, self.rotations, self.translations
        )
        if trans is not None:
            self.translations = trans
        self.times_s = times
        self.rotations = rots
        self.fitting = fitting
        return fitting

    def quantized_times(self) -> list[int]:
        out: list[int] = []
        previous = -1
        for seconds in self.times_s:
            units = quantize_time(seconds)
            if units <= previous:
                units = previous + 1
            out.append(units)
            previous = units
        return out

    def to_bytes(self) -> bytes:
        """The key data for this bone, in file order.

        Fits the keys to the format's clock first.  Doing it here rather than
        at construction means a caller can hold a ``BoneFrames`` and measure
        the cost (:meth:`fit`) before deciding whether to accept it.
        """
        if not self.is_quantizable:
            raise IfpWriteError(
                f"bone {self.name!r} runs to {self.times_s[-1]:.2f}s, which "
                f"is past the {MAX_TIME_UNITS / TIME_UNITS_PER_SECOND:.1f}s "
                f"an IFP time field can express"
            )
        if not self._fitted:
            self.fit()
        times = self.quantized_times()
        out = bytearray()
        for index in range(self.frame_count):
            x, y, z, w = quantize_rotation(self.rotations[index])
            out += struct.pack("<hhhhh", x, y, z, w, times[index])
            if self.is_root and self.translations is not None:
                tx, ty, tz = quantize_translation(self.translations[index])
                out += struct.pack("<hhh", tx, ty, tz)
        return bytes(out)


@dataclass
class Animation:
    """One animation: a set of bone tracks plus its header fields."""

    def reduce(self, threshold_deg: float = 0.25) -> "KeyReduction":
        """Opt-in: drop keys that barely move, and report what it cost.

        Lossy, and never automatic.  The resampling in :meth:`fit` is not --
        every key that survives it lands on the path the motion took -- so
        this is the only stage that discards information, which is why it
        stays behind a flag and states its error in degrees.
        """
        if not self.bones:
            return KeyReduction(0, 0, 0, 0.0, threshold_deg)
        source = kept = dropped = 0
        worst = 0.0
        for bone in self.bones:
            bone.reduce(threshold_deg)
            reduction = bone.reduction
            source += reduction.source_keys
            kept += reduction.kept_keys
            dropped += reduction.dropped_keys
            worst = max(worst, reduction.max_error_deg)
        return KeyReduction(source, kept, dropped, worst, threshold_deg)

    """One named animation: a set of bones with their key data."""

    name: str
    bones: list[BoneFrames] = field(default_factory=list)
    #: The file-level name the game registers this package under.  ``None``
    #: means "same as the animation's name", which is what a single-animation
    #: IFP almost always wants.
    internal_name: str | None = None
    _fitted: bool = False

    def __post_init__(self) -> None:
        if not self.bones:
            raise IfpWriteError(f"animation {self.name!r} has no bones")
        names = [b.name for b in self.bones]
        duplicates = {n for n in names if names.count(n) > 1}
        if duplicates:
            raise IfpWriteError(
                f"animation {self.name!r} has two objects named "
                f"{sorted(duplicates)}; an IFP object name is the key the "
                f"game looks the bone up by"
            )
        roots = [b for b in self.bones if b.is_root]
        if not roots:
            raise IfpWriteError(
                f"animation {self.name!r} has no root object; the game needs "
                f"one to hang the hierarchy from"
            )
        if len(roots) > 1:
            raise IfpWriteError(
                f"animation {self.name!r} has {len(roots)} root objects "
                f"({[b.name for b in roots]}); the format expects one"
            )

    @property
    def resolved_internal_name(self) -> str:
        return self.internal_name or self.name

    def fit(self) -> TimeFitting:
        """Fit every bone's keys to the clock, and report the worst cost.

        All the bones in an animation share one clock, so the figure that
        matters is the worst bone's -- an animation whose arm is 5% slow and
        whose leg is 5% fast plays a character whose limbs disagree about
        what time it is.
        """
        fittings = [bone.fit() for bone in self.bones]
        self._fitted = True
        return max(fittings, key=lambda f: f.stretch_ratio)

    @property
    def time_fitting(self) -> TimeFitting:
        """The worst per-bone time-fitting cost for this animation."""
        fittings = [b.fitting for b in self.bones if b.fitting is not None]
        if not fittings:
            return self.fit()
        return max(fittings, key=lambda f: f.stretch_ratio)

    @property
    def frame_data_size(self) -> int:
        """Bytes of *key data* in this animation, excluding object headers.

        The field is documented as "the exact size of the frame's usable
        data", and that is what it is written as here.  It is worth being
        precise about the consequence: neither MTA nor the game uses this
        field to seek -- both read objects sequentially and locate frames by
        walking -- so a file with the wrong value in it still plays.  Writing
        the honest value keeps the file correct for the tools that do check
        it, and :mod:`.ifp_reader` checks it as a round-trip assertion.
        """
        return sum(len(b.to_bytes()) for b in self.bones)

    def to_bytes(self) -> bytes:
        if not self._fitted:
            self.fit()
        out = bytearray()
        out += _pack_name(self.name)
        out += struct.pack("<iii", len(self.bones), self.frame_data_size, 1)
        for bone in self.bones:
            out += _pack_name(bone.name)
            out += struct.pack(
                "<iii",
                FRAME_TYPE_ROOT if bone.is_root else FRAME_TYPE_CHILD,
                bone.frame_count,
                bone.bone_id,
            )
            out += bone.to_bytes()
        return bytes(out)


def build_ifp(animations: Sequence[Animation], internal_name: str) -> bytes:
    """Assemble a complete ANP3 file.

    The offset in the header counts from immediately after itself, which is an
    easy thing to get off by four and produces a file the game silently
    refuses -- so it is computed from the real end of the assembled body
    rather than written as a constant.
    """
    if not animations:
        raise IfpWriteError("an IFP must contain at least one animation")
    body = bytearray()
    for animation in animations:
        body += animation.to_bytes()

    header = bytearray()
    header += b"ANP3"
    # Placeholder for the offset; filled in once the size is known.
    header += b"\0\0\0\0"
    header += _pack_name(internal_name)
    header += struct.pack("<i", len(animations))

    # The offset is measured from the stream position, which is 8 bytes in --
    # just past the magic and the offset field itself.  Adding 8 to this
    # value gives the file's total size, which is how the value is
    # conventionally interpreted.  Neither the game nor MTA reads it (both
    # walk the file sequentially), but a file with a wrong value here trips
    # up every third-party tool that does check, and it costs one subtraction
    # to get right.
    total = len(header) + len(body)
    header[4:8] = struct.pack("<i", total - 8)
    return bytes(header) + bytes(body)


def write_ifp(
    path: str,
    animations: Sequence[Animation],
    internal_name: str | None = None,
) -> dict:
    """Write an IFP, and read it straight back before returning.

    The round trip is not ceremony.  An IFP has no checksum and no version
    field, so a structural mistake produces a file that loads and animates
    nothing.  Re-reading what was just written turns that from a debugging
    session into a build failure.

    Returns a summary including the byte counts, so a caller can report the
    real size of what was produced.
    """
    if internal_name is None:
        if len(animations) != 1:
            raise IfpWriteError(
                f"writing {len(animations)} animations needs an explicit "
                f"internal file name; the package is registered under one name"
            )
        internal_name = animations[0].name

    data = build_ifp(animations, internal_name)

    from .ifp_reader import read_ifp, IfpStructureError

    try:
        parsed = read_ifp(data)
    except IfpStructureError as error:
        raise IfpWriteError(
            f"the IFP just built does not read back cleanly: {error}"
        ) from error

    if parsed.animation_count != len(animations):
        raise IfpWriteError(
            f"wrote {len(animations)} animations, read back "
            f"{parsed.animation_count}"
        )

    with open(path, "wb") as handle:
        handle.write(data)

    return {
        "path": str(path),
        "bytes": len(data),
        "animations": len(animations),
        "internal_name": internal_name,
        "objects": sum(len(a.bones) for a in animations),
        "frames": sum(b.frame_count for a in animations for b in a.bones),
        "round_tripped": True,
    }


__all__ = [
    "Animation",
    "BoneFrames",
    "KeyReduction",
    "TimeFitting",
    "drop_negligible_keys",
    "fit_keys_to_clock",
    "IfpWriteError",
    "build_ifp",
    "write_ifp",
    "quantize_rotation",
    "quantize_translation",
    "quantize_time",
    "ROTATION_SCALE",
    "TRANSLATION_SCALE",
    "TIME_UNITS_PER_SECOND",
    "NAME_SIZE",
    "FRAME_TYPE_ROOT",
    "FRAME_TYPE_CHILD",
]
