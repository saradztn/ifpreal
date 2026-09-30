"""The IFP writer and reader, checked byte by byte.

An ANP3 file has no checksum, no version field, and no field the game uses
to find the end of the data.  A file that is one byte wrong in an offset
loads in the game as an animation that does nothing, with no error reported
anywhere.  So these tests check the bytes at the offsets, not just that the
writer and the reader agree with each other -- two wrong implementations of
the same misunderstanding pass a round-trip test perfectly.
"""

from __future__ import annotations

import os
import struct
import sys

import numpy as np
import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from gta_fbx_ifp_converter.core import mathx
from gta_fbx_ifp_converter.gta.ifp_build import build_animation
from gta_fbx_ifp_converter.gta.ifp_reader import IfpStructureError, read_ifp
from gta_fbx_ifp_converter.gta.ifp_writer import (
    FRAME_TYPE_CHILD,
    FRAME_TYPE_ROOT,
    NAME_SIZE,
    ROTATION_SCALE,
    TIME_UNITS_PER_SECOND,
    TRANSLATION_SCALE,
    Animation,
    BoneFrames,
    IfpWriteError,
    build_ifp,
    fit_keys_to_clock,
    quantize_rotation,
    quantize_time,
    quantize_translation,
    write_ifp,
)

IDENTITY = np.array([0.0, 0.0, 0.0, 1.0])


def a_clip(frames: int = 4, step: float = 0.02):
    times = np.arange(frames, dtype=np.float64) * step
    rots = np.tile(IDENTITY, (frames, 1))
    trans = np.zeros((frames, 3))
    return times, rots, trans


# --------------------------------------------------------------------------- #
# header, at fixed offsets
# --------------------------------------------------------------------------- #
def test_header_is_exactly_where_the_spec_says():
    """Magic, offset, 24-byte name, count -- at the byte offsets MTA reads.

    The struct MTA reads is ``{uint32 OffsetEOF; char Name[24]; int32
    TotalAnimations}`` immediately after the four magic bytes, so the file
    name starts at byte 8 and the animation count at byte 32.  These are
    checked at those offsets rather than through the project's own reader,
    because the reader is the thing that could be wrong.
    """
    times, rots, trans = a_clip()
    data = build_ifp(
        [Animation("T", [BoneFrames("Root", 0, rots, times, trans, is_root=True)])],
        "MYFILE",
    )
    assert data[:4] == b"ANP3"
    assert data[8:8 + NAME_SIZE].rstrip(b"\0") == b"MYFILE"
    assert struct.unpack_from("<i", data, 32)[0] == 1


def test_offset_counts_from_byte_eight():
    """``OffsetEOF + 8`` is the file's size, which is the documented meaning.

    Getting this off by four produces a file that plays perfectly -- nothing
    reads the field -- and is silently wrong for every tool that does.
    """
    times, rots, trans = a_clip()
    data = build_ifp(
        [Animation("T", [BoneFrames("Root", 0, rots, times, trans,
                                    is_root=True)])],
        "MYFILE",
    )
    offset = struct.unpack_from("<i", data, 4)[0]
    assert offset + 8 == len(data)


def test_the_header_name_is_the_package_name_not_the_animation_name():
    """The 24-byte header name is the block name the game registers.

    These are two different strings and swapping them loads the file under a
    name nobody calls, which then fails at the ``setPedAnimation`` call with
    no indication why.
    """
    times, rots, trans = a_clip()
    data = build_ifp(
        [Animation("DANCE", [BoneFrames("Root", 0, rots, times, trans,
                                         is_root=True)])],
        "myblock",
    )
    assert data[8:32].rstrip(b"\0") == b"myblock"
    parsed = read_ifp(data)
    assert parsed.internal_name == "myblock"
    assert parsed.animations[0].name == "DANCE"


# --------------------------------------------------------------------------- #
# animation and object headers
# --------------------------------------------------------------------------- #
def test_animation_header_layout():
    times, rots, trans = a_clip()
    data = build_ifp(
        [Animation("T", [
            BoneFrames("Root", 0, rots, times, trans, is_root=True),
            BoneFrames(" L UpperArm", 32, rots, times),
        ])],
        "T",
    )
    offset = 36
    assert data[offset:offset + NAME_SIZE].rstrip(b"\0") == b"T"
    assert struct.unpack_from("<i", data, offset + 24)[0] == 2      # objects
    assert struct.unpack_from("<i", data, offset + 32)[0] == 1      # compressed


def test_object_frame_types_are_the_documented_values():
    """Root is 4 and child is 3, per MTA's ``eFrameType``.

    These are not arbitrary: 3 is ``KR00_COMPRESSED`` and 4 is
    ``KRT0_COMPRESSED``, and 4 is the only one whose frames carry a
    translation.  Writing 3 for a root loses the character's movement, and
    writing 4 for a child makes the reader consume six bytes per frame that
    the writer did not write, which desynchronises the rest of the file.
    """
    times, rots, trans = a_clip()
    data = build_ifp(
        [Animation("T", [
            BoneFrames("Root", 0, rots, times, trans, is_root=True),
            BoneFrames(" L UpperArm", 32, rots, times),
        ])],
        "T",
    )
    first = 36 + NAME_SIZE + 12
    assert struct.unpack_from("<i", data, first + 24)[0] == FRAME_TYPE_ROOT
    second = first + NAME_SIZE + 12 + 4 * 16
    assert struct.unpack_from("<i", data, second + 24)[0] == FRAME_TYPE_CHILD


def test_bone_id_is_the_hanim_tag():
    """The object stores the bone's HAnim tag, not its hierarchy index.

    The game matches a sequence to a bone by this number.  Writing the DFF's
    frame index instead produces a file where every bone animates the wrong
    joint -- and the object's name still looks right, so nothing in the file
    reveals the mistake.
    """
    times, rots, trans = a_clip()
    data = build_ifp(
        [Animation("T", [BoneFrames("Root", 0, rots, times,
                                    np.zeros((times.size, 3)), is_root=True),
                          BoneFrames(" L UpperArm", 32, rots, times)])],
        "T",
    )
    # Root object header, then its 16-byte frames, then the arm's header.
    arm = 36 + NAME_SIZE + 12 + NAME_SIZE + 12 + times.size * 16
    assert data[arm:arm + NAME_SIZE].rstrip(b"\0") == b" L UpperArm"
    assert struct.unpack_from("<i", data, arm + 32)[0] == 32


# --------------------------------------------------------------------------- #
# frame data
# --------------------------------------------------------------------------- #
def test_a_quaternion_is_four_int16_then_a_time():
    """Frame layout: xyzw, time.  Translation follows, for a root only."""
    times, rots, trans = a_clip(frames=1)
    data = build_ifp(
        [Animation("T", [BoneFrames("Root", 0, rots, times, trans, is_root=True)])],
        "T",
    )
    offset = 36 + NAME_SIZE + 12 + NAME_SIZE + 12
    x, y, z, w, t = struct.unpack_from("<5h", data, offset)
    assert (x, y, z, w) == (0, 0, 0, 4096)
    assert t == 0
    tx, ty, tz = struct.unpack_from("<3h", data, offset + 10)
    assert (tx, ty, tz) == (0, 0, 0)


def test_a_child_frame_has_no_translation():
    times, rots, _ = a_clip(frames=1)
    data = build_ifp(
        [Animation("T", [BoneFrames("Root", 0, rots, times,
                                    np.zeros((1, 3)), is_root=True),
                          BoneFrames("Arm", 32, rots, times)])],
        "T",
    )
    # Root frame is 16 bytes, then the child object header, then 10.
    offset = 36 + NAME_SIZE + 12 + NAME_SIZE + 12 + 16 + NAME_SIZE + 12
    assert len(data) - offset == 10, "a child frame must be rotation+time only"


def test_time_is_in_fiftieths_of_a_second():
    """0.5 s is 25, not 30 and not 50.

    Some references say 1/60.  It is 1/50, and a file written at 1/60 plays
    20% slow with no error anywhere -- which is why this is checked against
    the raw bytes.
    """
    times = np.array([0.5])
    rots = np.tile(IDENTITY, (1, 1))
    data = build_ifp(
        [Animation("T", [BoneFrames("Root", 0, rots, times,
                                    np.zeros((1, 3)), is_root=True)])],
        "T",
    )
    offset = 36 + NAME_SIZE + 12 + NAME_SIZE + 12
    assert struct.unpack_from("<h", data, offset + 8)[0] == 25


# --------------------------------------------------------------------------- #
# quantisation
# --------------------------------------------------------------------------- #
def test_rotation_scale_is_4096():
    assert quantize_rotation([0.0, 0.0, 0.0, 1.0]) == (0, 0, 0, 4096)
    quarter = quantize_rotation(mathx.quat_normalize([1.0, 0.0, 0.0, 1.0]))
    assert abs(quarter[0] - 0.7071 * 4096) <= 1


def test_translation_scale_is_1024():
    assert quantize_translation([1.0, -0.5, 0.25]) == (1024, -512, 256)


def test_quantisation_rounds_rather_than_truncates():
    """A component just under a half-step rounds up, not down.

    Truncation biases every component toward zero by half a step, and over a
    thousand keys that bias accumulates into a visible lean.
    """
    # Values clear of a tie, where rounding-away-from-zero is unambiguous.
    # Half-integer inputs are avoided on purpose: Python rounds .5 to the
    # nearest even, so a test built on 1.5 would be testing the tie-break
    # rule rather than the thing this test is about.
    for raw in (0.55, 0.6, 0.9, 1.2, 2.2):
        q = mathx.quat_normalize([raw / ROTATION_SCALE, 0.0, 0.0, 1.0])
        assert q[0] * ROTATION_SCALE == pytest.approx(raw, abs=0.01)
        assert quantize_rotation(q)[0] == int(raw + 0.5)
        assert quantize_rotation(-q)[0] == -(int(raw + 0.5))
    # A value just under the boundary rounds to zero, as it should.
    below = mathx.quat_normalize([0.4 / ROTATION_SCALE, 0.0, 0.0, 1.0])
    assert quantize_rotation(below)[0] == 0


def test_an_out_of_range_value_clamps_rather_than_wraps():
    """A clamped component stays near where it was; a wrapped one inverts.

    ``-32769`` wrapping to ``32767`` is a 180-degree error in one key, which
    shows up as a single frame where a limb snaps the other way.
    """
    huge = quantize_rotation([0.0, 0.0, 0.0, 1.0])
    assert huge == (0, 0, 0, 4096)
    assert quantize_translation([100.0, 0.0, 0.0])[0] == 32767


def test_time_never_goes_negative():
    assert quantize_time(-1.0) == 0


def test_quaternions_are_normalised_before_quantisation():
    """A non-unit input is cleaned up, not encoded as given.

    FBX stores float32, so a key can arrive a hair off unit length.  Encoding
    it directly puts a non-unit quaternion in the file, and the game does not
    renormalise -- the error shows up as a slow drift over a long animation.
    """
    scaled = quantize_rotation([0.0, 0.0, 0.0, 1.0004])
    assert scaled == (0, 0, 0, 4096)


# --------------------------------------------------------------------------- #
# the 1/50 s clock -- the constraint that actually bites
# --------------------------------------------------------------------------- #
def test_keys_that_fit_the_clock_are_kept_exactly():
    times = np.arange(0, 1.0, 1.0 / 50)
    rots = np.tile(IDENTITY, (times.size, 1))
    new_times, new_rots, _, fitting = fit_keys_to_clock(times, rots)
    assert fitting.collided_keys == 0
    assert fitting.fits_without_loss
    assert new_times.size == times.size
    assert fitting.stretch_ratio == pytest.approx(1.0)


def test_too_many_keys_merges_rather_than_stretching_the_animation():
    """A 30 fps source is over the format's limit, and merging keeps time.

    The obvious fix -- push each colliding key forward by one slot -- keeps
    every key and makes the animation 26% longer, so it plays slow.  Merging
    into the previous slot costs one key per collision and preserves the
    duration exactly, which is the lesser evil by a wide margin.
    """
    # 100 keys per second, twice what the format can address, ending
    # exactly on a 1/50 s boundary so the end time is not itself rounded.
    times = np.arange(100, dtype=np.float64) / 100.0
    times[-1] = 1.0
    rots = np.tile(IDENTITY, (100, 1))
    new_times, _, _, fitting = fit_keys_to_clock(times, rots)
    assert fitting.collided_keys > 0
    assert not fitting.fits_without_loss
    assert fitting.stretch_ratio == pytest.approx(1.0), (
        "merging must not change the animation's length"
    )
    assert new_times.size < times.size
    assert np.all(np.diff(new_times) > 0), "keys must stay strictly ordered"


def test_the_cost_of_the_clock_is_reported_not_absorbed():
    """A caller can ask what the format will cost before writing the file."""
    times = np.arange(100, dtype=np.float64) / 100.0
    times[-1] = 1.0
    rots = np.tile(IDENTITY, (100, 1))
    _, _, _, fitting = fit_keys_to_clock(times, rots)
    text = fitting.describe()
    assert "share a" in text and "1.00x" in text
    assert fitting.source_duration_s == pytest.approx(1.0)
    assert fitting.stored_duration_s == pytest.approx(1.0)


def test_the_real_clip_keeps_its_length_through_the_writer():
    """End to end on the Samba clip: 18.2 s in, 18.2 s out.

    The source has 63 keys per second and the format addresses 50, so this is
    the real case rather than a synthetic one.  A 26% stretch here would be
    an animation that visibly plays slow, and it is the failure this whole
    mechanism exists to avoid.
    """
    from gta_fbx_ifp_converter.fbx import source_rig as sr

    rig = sr.load_source_rig(os.path.join(ROOT, "testdata", "samba_dancing.fbx"))
    clip = rig.animated_clips[0]
    source = clip.poses[0][:, :4]
    times, _, _, fitting = fit_keys_to_clock(
        np.asarray(clip.times_s), source
    )
    assert fitting.source_duration_s == pytest.approx(clip.duration, abs=0.05)
    assert fitting.stretch_ratio == pytest.approx(1.0, abs=1e-6)
    assert fitting.collided_keys > 0, "the fixture should exercise the limit"


# --------------------------------------------------------------------------- #
# refusals
# --------------------------------------------------------------------------- #
def test_an_animation_with_no_root_is_refused():
    times, rots, _ = a_clip()
    with pytest.raises(IfpWriteError, match="no root object"):
        Animation("T", [BoneFrames("Arm", 32, rots, times)])


def test_two_roots_are_refused():
    times, rots, trans = a_clip()
    with pytest.raises(IfpWriteError, match="2 root objects"):
        Animation("T", [
            BoneFrames("A", 0, rots, times, trans, is_root=True),
            BoneFrames("B", 1, rots, times, trans, is_root=True),
        ])


def test_duplicate_object_names_are_refused():
    """The object name is the key the game looks the bone up by."""
    times, rots, trans = a_clip()
    with pytest.raises(IfpWriteError, match="two objects named"):
        Animation("T", [
            BoneFrames("Root", 0, rots, times, trans, is_root=True),
            BoneFrames("Root", 1, rots, times),
        ])


def test_translation_on_a_child_is_refused():
    """The format has nowhere to put it, so writing it would desynchronise."""
    times, rots, trans = a_clip()
    with pytest.raises(IfpWriteError, match="not a root frame"):
        BoneFrames("Arm", 32, rots, times, trans, is_root=False)


def test_an_overlong_animation_is_refused_not_wrapped():
    """A negative time field makes the animation play backwards."""
    times = np.array([0.0, 700.0])
    rots = np.tile(IDENTITY, (2, 1))
    bone = BoneFrames("Root", 0, rots, times, np.zeros((2, 3)), is_root=True)
    assert not bone.is_quantizable
    with pytest.raises(IfpWriteError, match="time field can express"):
        bone.to_bytes()


def test_an_overlong_name_is_refused_not_truncated():
    """Silently cutting a name loads a *different* animation, with no error."""
    times, rots, trans = a_clip()
    long_name = "A" * 30
    animation = Animation(long_name, [
        BoneFrames("Root", 0, rots, times, trans, is_root=True)
    ])
    with pytest.raises(IfpWriteError, match="name field holds"):
        build_ifp([animation], "T")


def test_mismatched_counts_are_refused():
    times, rots, _ = a_clip(frames=3)
    with pytest.raises(IfpWriteError, match="2 rotations for 3 times"):
        BoneFrames("Arm", 32, rots[:2], times)


def test_an_empty_animation_is_refused():
    with pytest.raises(IfpWriteError, match="no bones"):
        Animation("T", [])


def test_an_empty_file_is_refused():
    with pytest.raises(IfpWriteError, match="at least one animation"):
        build_ifp([], "T")


# --------------------------------------------------------------------------- #
# the reader is a real check, not a mirror of the writer
# --------------------------------------------------------------------------- #
def test_the_reader_rejects_a_wrong_file_size():
    times, rots, trans = a_clip()
    data = bytearray(build_ifp(
        [Animation("T", [BoneFrames("Root", 0, rots, times, trans, is_root=True)])],
        "T",
    ))
    data.append(0)
    with pytest.raises(IfpStructureError, match="is \\d+ bytes"):
        read_ifp(bytes(data))


def test_the_reader_rejects_the_wrong_magic():
    times, rots, trans = a_clip()
    data = bytearray(build_ifp(
        [Animation("T", [BoneFrames("Root", 0, rots, times, trans, is_root=True)])],
        "T",
    ))
    data[0:4] = b"ANPK"
    with pytest.raises(IfpStructureError, match="ANP3"):
        read_ifp(bytes(data))


def test_the_reader_rejects_a_truncated_file():
    times, rots, trans = a_clip()
    data = build_ifp(
        [Animation("T", [BoneFrames("Root", 0, rots, times, trans, is_root=True)])],
        "T",
    )
    with pytest.raises(IfpStructureError):
        read_ifp(data[:len(data) // 2])


def test_the_reader_rejects_a_bad_frame_type():
    times, rots, trans = a_clip()
    data = bytearray(build_ifp(
        [Animation("T", [BoneFrames("Root", 0, rots, times, trans, is_root=True)])],
        "T",
    ))
    struct.pack_into("<i", data, 36 + NAME_SIZE + 12 + 24, 99)
    with pytest.raises(IfpStructureError, match="frame type"):
        read_ifp(bytes(data))


def test_the_reader_rejects_a_frame_size_that_does_not_add_up():
    times, rots, trans = a_clip()
    data = bytearray(build_ifp(
        [Animation("T", [BoneFrames("Root", 0, rots, times, trans, is_root=True)])],
        "T",
    ))
    # Declare a frame size one byte short of what the frames occupy.
    declared = struct.unpack_from("<i", data, 36 + NAME_SIZE + 12 + 28)[0]
    struct.pack_into("<i", data, 36 + NAME_SIZE + 12 + 28, declared - 1)
    with pytest.raises(IfpStructureError, match=r"frame data|occupy"):
        read_ifp(bytes(data))


# --------------------------------------------------------------------------- #
# round trip
# --------------------------------------------------------------------------- #
def test_a_written_file_reads_back_with_its_values_intact(tmp_path):
    times, _, _ = a_clip(frames=5, step=0.1)
    rots = np.array([
        mathx.quat_normalize([0.1, 0.2, 0.3, 1.0]) for _ in range(5)
    ])
    trans = np.array([[0.0, 0.0, 0.0], [0.1, 0.2, 0.3], [0.0, 0.0, 0.0],
                      [-0.1, 0.0, 0.1], [0.0, 0.0, 0.0]])
    path = tmp_path / "x.ifp"
    info = write_ifp(str(path), [Animation("T", [
        BoneFrames("Root", 0, rots, times, trans, is_root=True),
        BoneFrames(" L UpperArm", 32, rots, times),
    ])], "x")
    assert info["round_tripped"]

    parsed = read_ifp(str(path))
    assert parsed.internal_name == "x"
    root = parsed.animations[0].object_by_name("Root")
    assert root is not None and root.is_root
    assert np.allclose(root.rotation_array(), rots, atol=1.0 / ROTATION_SCALE)
    assert np.allclose(root.translation_array(), trans, atol=1.0 / TRANSLATION_SCALE)
    assert np.allclose(root.time_array(), times, atol=1.0 / TIME_UNITS_PER_SECOND)

    arm = parsed.animations[0].object_by_name(" L UpperArm")
    assert arm is not None and not arm.is_root
    assert arm.translation_array() is None


def test_the_file_is_not_written_when_the_round_trip_fails(tmp_path, monkeypatch):
    """A file that cannot be read back must not reach disk.

    An IFP has no checksum, so a structural mistake produces a file that
    loads and animates nothing.  Writing it anyway turns a build failure into
    a debugging session in the game.
    """
    import gta_fbx_ifp_converter.gta.ifp_reader as reader

    def broken(data):
        raise IfpStructureError("deliberately broken")

    monkeypatch.setattr(reader, "read_ifp", broken)
    times, rots, trans = a_clip()
    path = tmp_path / "x.ifp"
    with pytest.raises(IfpWriteError, match="does not read back cleanly"):
        write_ifp(str(path), [Animation("T", [
            BoneFrames("Root", 0, rots, times, trans, is_root=True)
        ])], "x")
    assert not path.exists()
