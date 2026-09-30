"""The IFP reader: parse an ANP3 file back into quaternions and times.

This exists so the writer can check its own work, and so a file produced
elsewhere can be inspected without trusting it.  It is deliberately strict:
an ANP3 file has no checksum and no version field, so a truncated or
mis-offset file does not announce itself.  Every length here is checked
against the bytes actually remaining, and a file that does not account for
itself exactly is an error rather than a partial parse.

Frames are returned as they are stored -- quantised int16s, plus the float
values they represent -- so the validator can measure the quantisation error
instead of assuming it away.
"""

from __future__ import annotations

import struct
from dataclasses import dataclass, field
from typing import BinaryIO, Sequence

import numpy as np

from .ifp_writer import (
    FRAME_TYPE_CHILD,
    FRAME_TYPE_ROOT,
    NAME_SIZE,
    ROTATION_SCALE,
    TIME_UNITS_PER_SECOND,
    TRANSLATION_SCALE,
)


class IfpStructureError(Exception):
    """The bytes are not a well-formed ANP3 file."""


@dataclass
class ParsedFrame:
    """One key, as stored and as decoded."""

    rotation_raw: tuple[int, int, int, int]
    time_units: int
    translation_raw: tuple[int, int, int] | None

    @property
    def rotation(self) -> np.ndarray:
        return np.array(self.rotation_raw, dtype=np.float64) / ROTATION_SCALE

    @property
    def translation(self) -> np.ndarray | None:
        if self.translation_raw is None:
            return None
        return np.array(self.translation_raw, dtype=np.float64) / TRANSLATION_SCALE

    @property
    def time_s(self) -> float:
        return self.time_units / TIME_UNITS_PER_SECOND


@dataclass
class ParsedObject:
    name: str
    frame_type: int
    frame_count: int
    bone_id: int
    frames: list[ParsedFrame] = field(default_factory=list)

    @property
    def is_root(self) -> bool:
        return self.frame_type == FRAME_TYPE_ROOT

    @property
    def duration_s(self) -> float:
        return self.frames[-1].time_s if self.frames else 0.0

    def rotation_array(self) -> np.ndarray:
        if not self.frames:
            return np.zeros((0, 4), dtype=np.float64)
        return np.array([f.rotation for f in self.frames], dtype=np.float64)

    def translation_array(self) -> np.ndarray | None:
        if not self.frames or self.frames[0].translation is None:
            return None
        return np.array(
            [f.translation for f in self.frames], dtype=np.float64
        )

    def time_array(self) -> np.ndarray:
        return np.array([f.time_s for f in self.frames], dtype=np.float64)


@dataclass
class ParsedAnimation:
    name: str
    objects: list[ParsedObject] = field(default_factory=list)
    declared_frame_data_size: int = 0
    unknown: int = 1

    @property
    def duration_s(self) -> float:
        return max((o.duration_s for o in self.objects), default=0.0)

    def object_by_name(self, name: str) -> ParsedObject | None:
        for obj in self.objects:
            if obj.name == name:
                return obj
        return None


@dataclass
class ParsedIfp:
    magic: str
    offset_to_end: int
    internal_name: str
    animations: list[ParsedAnimation] = field(default_factory=list)
    total_bytes: int = 0

    @property
    def animation_count(self) -> int:
        return len(self.animations)

    def animation(self, name: str) -> ParsedAnimation | None:
        for anim in self.animations:
            if anim.name == name:
                return anim
        return None

    @property
    def all_objects(self) -> list[ParsedObject]:
        return [o for a in self.animations for o in a.objects]

    def bone_ids_used(self) -> set[int]:
        return {o.bone_id for o in self.all_objects}


class _Reader:
    """A bounds-checked cursor over the file's bytes."""

    def __init__(self, data: bytes):
        self.data = data
        self.pos = 0

    def remaining(self) -> int:
        return len(self.data) - self.pos

    def need(self, count: int, what: str) -> None:
        if self.remaining() < count:
            raise IfpStructureError(
                f"{what} needs {count} bytes at offset {self.pos}, but only "
                f"{self.remaining()} remain in a {len(self.data)}-byte file"
            )

    def take(self, count: int, what: str) -> bytes:
        self.need(count, what)
        out = self.data[self.pos:self.pos + count]
        self.pos += count
        return out

    def i32(self, what: str) -> int:
        return struct.unpack("<i", self.take(4, what))[0]

    def i16(self, what: str) -> int:
        return struct.unpack("<h", self.take(2, what))[0]

    def name(self, what: str) -> str:
        raw = self.take(NAME_SIZE, what)
        # Names are null-padded; a field with no null at all is malformed
        # rather than a 24-character name, and silently accepting it hides a
        # writer bug.
        terminator = raw.find(b"\0")
        if terminator < 0:
            raise IfpStructureError(
                f"{what} at offset {self.pos - NAME_SIZE} is 24 bytes with no "
                f"null terminator: {raw!r}"
            )
        return raw[:terminator].decode("ascii", errors="replace")


def read_ifp(source: bytes | str | BinaryIO) -> ParsedIfp:
    """Parse an ANP3 file from bytes, a path, or an open binary file."""
    if isinstance(source, (bytes, bytearray)):
        data = bytes(source)
    elif isinstance(source, str):
        with open(source, "rb") as handle:
            data = handle.read()
    else:
        data = source.read()

    reader = _Reader(data)
    magic = reader.take(4, "file magic")
    if magic != b"ANP3":
        raise IfpStructureError(
            f"expected an ANP3 file, found magic {magic!r}; an ANP2 package "
            f"has a different layout and this reader does not claim it"
        )

    offset_to_end = reader.i32("file offset")
    internal_name = reader.name("internal file name")
    animation_count = reader.i32("animation count")

    if animation_count < 0 or animation_count > 4096:
        raise IfpStructureError(
            f"animation count {animation_count} is not plausible; the file "
            f"may be misaligned"
        )

    animations: list[ParsedAnimation] = []
    for _ in range(animation_count):
        animations.append(_read_animation(reader))

    # The header's offset is the file's own claim about its size, counted
    # from byte 8.  Adding 8 gives the total, which is the convention every
    # tool that reads this field uses.  It is checked rather than trusted:
    # neither the game nor MTA reads it, so a file with a wrong value in it
    # plays perfectly and is silently malformed, which is exactly the kind
    # of defect worth catching here.
    if offset_to_end + 8 != len(data):
        raise IfpStructureError(
            f"the header says the file is {offset_to_end + 8} bytes but it "
            f"is {len(data)}; parsed {animation_count} animations first"
        )

    return ParsedIfp(
        magic="ANP3",
        offset_to_end=offset_to_end,
        internal_name=internal_name,
        animations=animations,
        total_bytes=len(data),
    )


def _read_animation(reader: _Reader) -> ParsedAnimation:
    name = reader.name("animation name")
    object_count = reader.i32("object count")
    frame_data_size = reader.i32("frame data size")
    unknown = reader.i32("unknown field")

    if object_count < 0 or object_count > 4096:
        raise IfpStructureError(
            f"animation {name!r} declares {object_count} objects"
        )

    start = reader.pos
    objects: list[ParsedObject] = []
    for _ in range(object_count):
        objects.append(_read_object(reader))

    # The declared size counts key data only.  Each object's header is a
    # 24-byte name plus three int32 fields, and the key data is interleaved
    # with those headers, so the key bytes are the total less the headers.
    # It is checked because a file whose own bookkeeping disagrees with its
    # contents is the one that has been written wrong, even though neither
    # the game nor MTA would notice.
    object_header_bytes = object_count * (NAME_SIZE + 12)
    key_bytes = reader.pos - start - object_header_bytes
    if key_bytes != frame_data_size:
        raise IfpStructureError(
            f"animation {name!r} declares {frame_data_size} bytes of frame "
            f"data but its frames occupy {key_bytes}"
        )

    return ParsedAnimation(
        name=name,
        objects=objects,
        declared_frame_data_size=frame_data_size,
        unknown=unknown,
    )


def _read_object(reader: _Reader) -> ParsedObject:
    name = reader.name("object name")
    frame_type = reader.i32("frame type")
    frame_count = reader.i32("frame count")
    bone_id = reader.i32("bone id")

    if frame_type not in (FRAME_TYPE_ROOT, FRAME_TYPE_CHILD):
        raise IfpStructureError(
            f"object {name!r} has frame type {frame_type}; expected "
            f"{FRAME_TYPE_ROOT} (root) or {FRAME_TYPE_CHILD} (child)"
        )
    if frame_count < 0:
        raise IfpStructureError(f"object {name!r} has frame count {frame_count}")

    has_translation = frame_type == FRAME_TYPE_ROOT
    frames: list[ParsedFrame] = []
    for _ in range(frame_count):
        x = reader.i16("quaternion x")
        y = reader.i16("quaternion y")
        z = reader.i16("quaternion z")
        w = reader.i16("quaternion w")
        time_units = reader.i16("time")
        translation = None
        if has_translation:
            translation = (
                reader.i16("translation x"),
                reader.i16("translation y"),
                reader.i16("translation z"),
            )
        frames.append(ParsedFrame((x, y, z, w), time_units, translation))

    return ParsedObject(
        name=name,
        frame_type=frame_type,
        frame_count=frame_count,
        bone_id=bone_id,
        frames=frames,
    )


__all__ = [
    "ParsedIfp",
    "ParsedAnimation",
    "ParsedObject",
    "ParsedFrame",
    "IfpStructureError",
    "read_ifp",
]
