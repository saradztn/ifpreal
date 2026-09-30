"""Real binary and ASCII FBX readers.

Binary layout (FBX 7.x, the format every modern exporter writes)::

    "Kaydara FBX Binary  \\x00"  21 bytes, magic+ 0x1A 0x00
    uint32                    version
    <node record>*            until end_offset
    <footer>                  unknown footer, safely skippable

An ASCII file starts with the comment ``; FBX 7.x.0 project file``.

Both readers produce the same :class:`~..fbx.document.FbxNode` tree, so the
rest of the pipeline never learns which encoding it came from.
"""

from __future__ import annotations

import io
import re
import struct
import zlib
from pathlib import Path
from typing import Any, BinaryIO

from .document import FbxDocument, FbxNode, FbxProperty, TYPED_ARRAY_RE

__all__ = ["FbxParseError", "read_fbx", "sniff_format"]


class FbxParseError(RuntimeError):
    """The file is not a readable FBX."""


class _NodeTruncated(Exception):
    """Internal: a child record points outside the file; resynchronise."""


_BINARY_MAGIC = b"Kaydara FBX Binary  \x00\x1a\x00"  # 23-byte FBX header
_ASCII_MAGIC = b"; FBX"

#: FBX array encoding byte: 0 = stored, 1 = zlib deflate.
ENCODING_DEFLATE = 1


def sniff_format(data: bytes) -> str:
    if data.startswith(_BINARY_MAGIC):
        return "binary"
    if data[:8].strip().startswith(b"; FBX"):
        return "ascii"
    raise FbxParseError(
        "Not an FBX file: expected the 'Kaydara FBX Binary' magic or an ASCII "
        f"'; FBX ...' comment, got {data[:24]!r}"
    )


# --------------------------------------------------------------------------- #
# binary
# --------------------------------------------------------------------------- #
class _BinaryReader:
    def __init__(self, data: bytes) -> None:
        self.data = data
        self.offset = 0

    def eof(self) -> bool:
        return self.offset >= len(self.data)

    def read(self, count: int) -> bytes:
        if count < 0:
            raise FbxParseError(f"Negative read length {count} at {self.offset}")
        chunk = self.data[self.offset:self.offset + count]
        if len(chunk) != count:
            raise FbxParseError(
                f"Truncated FBX: wanted {count} bytes at offset {self.offset}, "
                f"got {len(chunk)}"
            )
        self.offset += count
        return chunk

    def read_u8(self) -> int:
        return struct.unpack("<B", self.read(1))[0]

    def read_u32(self) -> int:
        return struct.unpack("<I", self.read(4))[0]

    def read_i32(self) -> int:
        return struct.unpack("<i", self.read(4))[0]

    def read_u64(self) -> int:
        return struct.unpack("<Q", self.read(8))[0]

    def read_double(self) -> float:
        return struct.unpack("<d", self.read(8))[0]

    def read_raw(self) -> bytes:
        length = self.read_u32()
        return self.read(length)

    def read_string(self) -> str:
        return self.read_raw().decode("utf-8", errors="replace")

    def read_array(self, encoding: str, length: int, byte_size: int) -> list:
        if length == 0:
            return []
        data = self.read(length * byte_size)
        if encoding == "d":
            return list(struct.unpack(f"<{length}d", data))
        if encoding == "f":
            return list(struct.unpack(f"<{length}f", data))
        if encoding == "l":
            return list(struct.unpack(f"<{length}q", data))
        if encoding == "i":
            return list(struct.unpack(f"<{length}i", data))
        if encoding == "b":
            return list(struct.unpack(f"<{length}b", data[:length]))
        raise FbxParseError(f"Unsupported FBX array encoding {encoding!r}")


def _unpack_array(payload: bytes, type_code: str, count: int) -> list:
    formats = {"f": "f", "d": "d", "l": "q", "i": "i", "b": "b"}
    return list(struct.unpack(f"<{count}{formats[type_code]}", payload))


def _is_zlib_header(data: bytes, offset: int) -> bool:
    """Validate a zlib (RFC 1950) stream header at ``offset``."""
    if offset + 2 > len(data):
        return False
    cmf, flg = data[offset], data[offset + 1]
    if cmf & 0x0F != 8 or (cmf >> 4) > 7:
        return False
    return ((cmf << 8) | flg) % 31 == 0


def _find_zlib_header(data: bytes, offset: int, window: int = 8) -> int | None:
    """Locate the next plausible zlib header within ``window`` bytes.

    Some FBX writers pad the array header to 12 bytes, others use the 9 byte
    form; rather than guessing, the first position that actually starts a
    valid zlib stream wins.
    """
    last = min(offset + window, len(data) - 2)
    for candidate in range(offset, last + 1):
        if _is_zlib_header(data, candidate):
            return candidate
    return None


def _inflate(reader: "_BinaryReader", expected: int) -> tuple[bytes, int]:
    """Inflate one FBX deflate array, returning ``(payload, bytes_consumed)``."""
    decompressor = zlib.decompressobj()
    produced = bytearray()
    fed = 0
    chunk_size = 1 << 20
    while not decompressor.eof:
        chunk = reader.data[reader.offset + fed:reader.offset + fed + chunk_size]
        if not chunk:
            raise _NodeTruncated("truncated compressed array")
        produced += decompressor.decompress(chunk)
        fed += len(chunk)
        if len(produced) > expected + 8:
            raise FbxParseError(
                f"Compressed FBX array overflows its declared size ({expected} bytes)"
            )
    unused = len(decompressor.unused_data)
    return bytes(produced), fed - unused


def _read_property(reader: _BinaryReader, array_header_size: int = 12) -> FbxProperty:
    type_code = chr(reader.read_u8())
    if type_code == "Y":
        return FbxProperty("", reader.read_i32(), type_code)
    if type_code == "C":
        value = reader.read(1)
        return FbxProperty("", bool(value[0]), type_code)
    if type_code == "I":
        return FbxProperty("", reader.read_i32(), type_code)
    if type_code == "F":
        return FbxProperty("", reader.read_double(), type_code)
    if type_code == "D":
        return FbxProperty("", reader.read_double(), type_code)
    if type_code == "L":
        return FbxProperty("", reader.read_u64(), type_code)
    if type_code in ("f", "d", "l", "i", "b"):
        array_length = struct.unpack("<I", reader.read(4))[0]
        encoding = reader.read_u8()
        # ``array_header_size`` covers the 3 reserved bytes the FBX SDK pads
        # after the encoding byte plus the uint32 compressed length (12
        # bytes total); the 9 byte form omits the padding.  Which one a file
        # uses is decided per record by :func:`_read_properties`.
        reader.read(array_header_size - 5)
        byte_size = {"f": 4, "d": 8, "l": 8, "i": 4, "b": 1}[type_code]
        expected = array_length * byte_size
        if encoding == ENCODING_DEFLATE:
            # The zlib stream is self-delimiting, so it is inflated
            # incrementally and the reader is advanced by the number of input
            # bytes the stream actually consumed.  The declared compressed
            # length is ignored: some exporters write 1 there, others the
            # true size.
            start = _find_zlib_header(reader.data, reader.offset)
            if start is None:
                raise _NodeTruncated("missing zlib stream")
            reader.offset = start
            payload, consumed = _inflate(reader, expected)
            reader.offset += consumed
            return FbxProperty("", _unpack_array(payload, type_code, array_length), type_code)
        return FbxProperty(
            "", reader.read_array(type_code, array_length, byte_size), type_code
        )
    if type_code in ("S", "R"):
        return FbxProperty("", reader.read_string(), type_code)
    raise _NodeTruncated(f"unknown property type {type_code!r}")


#: Array-header widths to try, in preference order.  The 12 byte form is what
#: the Autodesk SDK writes; the 9 byte form is what most third-party readers
#: assume.  The choice is validated per record against ``PropertyListLen``.
ARRAY_HEADER_SIZES = (12, 9)


def _read_properties(
    reader: _BinaryReader,
    count: int,
    property_list_len: int,
) -> list[FbxProperty]:
    """Read a record's properties, choosing the array-header width that fits.

    Every FBX record declares the exact byte length of its property list, so
    the correct array-header variant is whichever one lands the cursor exactly
    on ``property_list_len``.  No guessing, and no drift.
    """
    start = reader.offset
    fallback: list[FbxProperty] | None = None
    fallback_end = start
    for header_size in ARRAY_HEADER_SIZES:
        reader.offset = start
        try:
            properties = [_read_property(reader, header_size) for _ in range(count)]
        except (_NodeTruncated, FbxParseError):
            continue
        if reader.offset == start + property_list_len:
            return properties
        if fallback is None:
            # Keep the widest interpretation around as a safety net, but
            # remember where it actually ended: the cursor has to stay past
            # the properties, never back at their first byte, or every
            # following record in the file is read from a wrong offset.
            fallback = properties
            fallback_end = reader.offset
    if fallback is None:
        raise _NodeTruncated("no readable property layout")
    reader.offset = fallback_end
    return fallback


class _NodeHeader:
    """Version-dependent widths of an FBX node record header.

    ``EndOffset``, ``NumProperties`` and ``PropertyListLen`` all widen from
    32 to 64 bits in FBX 7.5, and the *absolute* end offset then lives in
    the same record, so the three widths must move together.
    """

    __slots__ = ("num_props_format", "prop_len_format")

    def __init__(self, version: int) -> None:
        if version >= 7500:
            self.num_props_format = "<Q"
            self.prop_len_format = "<Q"
        else:
            self.num_props_format = "<I"
            self.prop_len_format = "<I"

    @property
    def size(self) -> int:
        return struct.calcsize(self.num_props_format) + struct.calcsize(self.prop_len_format) + 1


MAX_RECORD_PROPERTIES = 1 << 16
MAX_PROPERTY_LIST_BYTES = 512 << 20


def _looks_like_record(data: bytes, offset: int, header: "_NodeHeader") -> bool:
    """Validate a node record header against FBX's record invariant.

    ``end_offset`` must lie inside the file and must be at or after the end
    of the record's own property list.  For a leaf node the two are equal;
    for a container ``end_offset`` points past its children.  Combined with a
    printable ASCII name and a bounded property count this is a strong enough
    signature to reject damaged records and to re-find the next record after
    a damaged region.
    """
    start = offset
    size = header.size
    if start + size > len(data):
        return False
    end_offset = struct.unpack("<I", data[start:start + 4])[0]
    if end_offset <= start or end_offset > len(data):
        return False
    num_properties = struct.unpack(
        header.num_props_format,
        data[start + 4:start + 4 + struct.calcsize(header.num_props_format)],
    )[0]
    prop_len_off = start + 4 + struct.calcsize(header.num_props_format)
    property_list_len = struct.unpack(
        header.prop_len_format, data[prop_len_off:prop_len_off + struct.calcsize(header.prop_len_format)]
    )[0]
    if num_properties > MAX_RECORD_PROPERTIES or property_list_len > MAX_PROPERTY_LIST_BYTES:
        return False
    name_len = data[prop_len_off + struct.calcsize(header.prop_len_format)]
    if name_len == 0 or name_len > 128:
        return False
    name_start = prop_len_off + struct.calcsize(header.prop_len_format) + 1
    if name_start + name_len > len(data):
        return False
    if any(byte < 32 or byte > 126 for byte in data[name_start:name_start + name_len]):
        return False
    if end_offset == 0 and num_properties == 0:
        return False
    return name_start + name_len + property_list_len <= end_offset


def _find_next_record(
    data: bytes, offset: int, header: "_NodeHeader", limit: int
) -> int | None:
    """Scan forward for the next byte position that starts a valid record."""
    stop = min(len(data) - header.size, offset + limit)
    for candidate in range(offset, stop):
        if _looks_like_record(data, candidate, header):
            return candidate
    return None


def _read_node(reader: _BinaryReader, header: _NodeHeader) -> FbxNode:
    end_offset = reader.read_u32()
    num_properties = struct.unpack(header.num_props_format, reader.read(
        struct.calcsize(header.num_props_format)))[0]
    property_list_len = struct.unpack(
        header.prop_len_format,
        reader.read(struct.calcsize(header.prop_len_format)),
    )[0]
    name_len = reader.read_u8()
    if end_offset == 0 and num_properties == 0 and name_len == 0:
        # FBX NULL record: 13 (or 25) bytes of padding, no payload.
        return FbxNode("")
    name = reader.read(name_len).decode("utf-8", errors="replace")

    properties = _read_properties(reader, num_properties, property_list_len)
    for index, prop in enumerate(properties):
        if index == 0 and isinstance(prop.value, str):
            # The first raw property of a node is the property's own name.
            prop.name = prop.value

    node = FbxNode(name, properties=properties)
    while reader.offset < end_offset and not reader.eof():
        try:
            child = _read_node(reader, header)
        except _NodeTruncated:
            # A child record claims to live outside the file.  Real FBX
            # exporters do emit a few of these around geometry element
            # blocks, so resynchronise on the next valid record instead of
            # aborting the whole document.
            nxt = _find_next_record(reader.data, reader.offset, header, RESYNC_LIMIT)
            if nxt is None or nxt >= end_offset:
                reader.offset = min(end_offset, len(reader.data))
                break
            reader.offset = nxt
            continue
        if child.name:
            node.children.append(child)
    if reader.offset < end_offset:
        reader.offset = end_offset
    return node


#: How far the reader will scan for the next valid record after damage.
RESYNC_LIMIT = 1 << 20


def read_binary(data: bytes) -> FbxDocument:
    reader = _BinaryReader(data)
    magic = reader.read(len(_BINARY_MAGIC))
    if magic != _BINARY_MAGIC:
        raise FbxParseError("Bad binary FBX magic")
    version = reader.read_u32()
    document = FbxDocument(version=version, is_binary=True)
    header = _NodeHeader(version)
    document.resync_count = 0
    while reader.offset + header.size < len(data):
        if not _looks_like_record(reader.data, reader.offset, header):
            nxt = _find_next_record(reader.data, reader.offset, header, RESYNC_LIMIT)
            if nxt is None:
                break
            document.resync_count += 1
            reader.offset = nxt
            continue
        try:
            document.root.children.append(_read_node(reader, header))
        except _NodeTruncated:
            nxt = _find_next_record(reader.data, reader.offset, header, RESYNC_LIMIT)
            if nxt is None:
                break
            document.resync_count += 1
            reader.offset = nxt
    return document


# --------------------------------------------------------------------------- #
# ascii
# --------------------------------------------------------------------------- #
_ASCII_NODE_START = re.compile(r"^([A-Za-z][\w]*):\s*(.*)$")


def read_ascii(text: str) -> FbxDocument:
    from .document import FbxNode as _Node  # local alias, keeps typing simple

    lines = text.splitlines()
    version = 0
    creator = ""
    for line in lines[:20]:
        stripped = line.strip()
        if stripped.startswith("; FBX"):
            parts = stripped[2:].split()
            if len(parts) >= 2:
                try:
                    version = int(parts[1].rstrip(":"))
                except ValueError:
                    pass
            if len(parts) >= 4:
                creator = parts[3]
            break

    root = _Node("__root__")
    stack: list[_Node] = [root]
    document = FbxDocument(version=version, creator=creator, is_binary=False, root=root)

    index = 0
    total = len(lines)
    while index < total:
        raw = lines[index]
        line = raw.rstrip()
        if not line.strip() or line.strip().startswith(";"):
            index += 1
            continue
        match = _ASCII_NODE_START.match(line)
        if not match:
            index += 1
            continue
        name, rest = match.group(1), match.group(2).strip()
        if name in ("Objects", "Connections", "GlobalSettings", "Definitions", "Takes"):
            node = _Node(name)
        else:
            node = _Node(name)

        if rest:
            _parse_ascii_inline(node, rest)
            stack[-1].children.append(node)
            index += 1
            continue

        # Block form: gather until the matching close brace.
        depth = 1
        index += 1
        while index < total:
            candidate = lines[index].strip()
            if candidate.endswith("{"):
                depth += 1
            elif candidate == "}":
                depth -= 1
                if depth == 0:
                    index += 1
                    break
            elif candidate:
                _parse_ascii_inline(node, candidate)
            index += 1
        stack[-1].children.append(node)
        del stack  # the ASCII parser flattens exactly one level of nesting
    return document


_ASCII_ARRAY_RE = re.compile(r"^\*?\d*\s*\{?([\d.eE+-]+)\}?(?:,.*)?$")


def _parse_ascii_inline(node: _Node, text: str) -> None:
    """Parse ``"Name", "Type", value`` triples and ``a,b,c`` vector literals."""
    text = text.strip()
    if not text:
        return
    if text.startswith("*") or "," in text and not text.startswith('"'):
        _append_ascii_value(node, text)
        return
    _append_ascii_value(node, text)


def _append_ascii_value(node: _Node, text: str) -> None:
    text = text.strip().rstrip(",").strip()
    if not text:
        return

    if text.startswith('"'):
        try:
            value: Any = text[1:text.index('"', 1)]
        except ValueError:
            value = text.strip('"')
        _push_property(node, value)
        return

    if text.startswith("{"):
        inner = text[1:-1] if text.endswith("}") else text[1:]
        values = [v for v in inner.split(",") if v.strip() != ""]
        _push_property(node, [_ascii_number(v) for v in values])
        return

    if text.startswith("*"):
        tokens = text.split(":", 2)
        if len(tokens) == 3:
            _push_property(node, [_ascii_number(t) for t in tokens[2].split(",") if t.strip()])
        return

    if "," in text:
        # "Name", "Type", v1, v2, v3   (unquoted vector, very common)
        _push_property(node, [_ascii_number(t) for t in text.split(",")])
        return

    _push_property(node, _ascii_number(text))


def _push_property(node: _Node, value: Any) -> None:
    name = value if isinstance(value, str) and not node.properties else ""
    if not node.properties and isinstance(value, str):
        name = value
    node.properties.append(FbxProperty(name, value))


def _ascii_number(token: str) -> Any:
    token = token.strip().strip("{}").strip()
    if not token:
        return 0.0
    try:
        if any(c in token for c in ".eE") and not token.lower().startswith("0x"):
            return float(token)
        return int(token)
    except ValueError:
        return token


# --------------------------------------------------------------------------- #
# public entry point
# --------------------------------------------------------------------------- #
def read_fbx(path: str | Path) -> FbxDocument:
    """Parse an FBX file (binary or ASCII) into an :class:`FbxDocument`."""
    data = Path(path).read_bytes()
    kind = sniff_format(data)
    document = read_binary(data) if kind == "binary" else read_ascii(data.decode("utf-8", "replace"))
    document.root.name = "__root__"
    if not document.version:
        # Very old ASCII files: sniff the version from the comment.
        head = data[:120].decode("utf-8", "replace")
        for token in head.replace(";", " ").split():
            if token.replace(".", "").isdigit() and len(token) >= 5:
                document.version = int(float(token))
                break
    return document
