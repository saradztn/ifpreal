"""In-memory representation of an FBX document (a tree of typed properties).

The reader modules turn bytes into this; everything above this layer works on
:class:`FbxNode` and never touches the file format again.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any, Iterator, Sequence

__all__ = [
    "FbxProperty",
    "FbxNode",
    "FbxDocument",
    "decode_fbx_string",
    "TYPED_ARRAY_RE",
]


def decode_fbx_string(raw) -> str:
    """Decode an FBX 7.4 ``S`` property.

    Since 7.4 the FBX SDK appends ``\x00 <version byte> <SubClass name>`` to
    plain strings, so a Model's name property arrives as
    ``"mixamorig:Hips\x00\x01Model"``.  Everything from the first NUL is
    metadata, not part of the name.
    """
    if raw is None:
        return ""
    if isinstance(raw, bytes):
        text = raw.decode("utf-8", errors="replace")
    else:
        text = str(raw)
    nul = text.find("\x00")
    return text if nul < 0 else text[:nul]

TYPED_ARRAY_RE = re.compile(rb"^(?P<code>[dDfFiIlLbBcS][RTA]?)(?P<length>\d+):(?P<data>.*)$", re.S)

#: FBX property type codes mapped to the struct format character they carry.
_TYPE_CODES = {
    "Y": "b", "C": "b", "I": "h", "F": "f", "D": "d", "L": "q",
}

_ARRAY_CODES = {
    "d": "d", "f": "f", "l": "q", "i": "i", "b": "?", "c": "?", "d?": "d",
}


@dataclass
class FbxProperty:
    """A single typed property, e.g. ``"Name", "Lcl Translation", (0, 0, 0)``."""

    name: str
    value: Any
    type_code: str = ""

    @property
    def is_array(self) -> bool:
        return isinstance(self.value, (list, tuple))

    def as_float(self, default: float = 0.0) -> float:
        try:
            return float(self.value)
        except (TypeError, ValueError):
            return default

    def as_int(self, default: int = 0) -> int:
        try:
            return int(self.value)
        except (TypeError, ValueError):
            return default

    def as_str(self, default: str = "") -> str:
        if isinstance(self.value, bytes):
            return self.value.decode("utf-8", errors="replace")
        return str(self.value)

    def as_fbx_string(self, default: str = "") -> str:
        return decode_fbx_string(self.value) or default

    def as_vector(self, size: int, default: float = 0.0) -> list[float]:
        if not isinstance(self.value, (list, tuple)):
            return [default] * size
        values = [float(v) for v in self.value]
        if len(values) < size:
            values.extend([default] * (size - len(values)))
        return values[:size]

    def as_bool(self, default: bool = False) -> bool:
        if isinstance(self.value, (int, float)):
            return bool(self.value)
        if isinstance(self.value, bool):
            return self.value
        if isinstance(self.value, str):
            return self.value.strip().lower() in {"t", "true", "y", "yes", "1"}
        return default

    def __repr__(self) -> str:  # pragma: no cover - debugging aid
        return f"FbxProperty({self.name!r}, {self.value!r})"


@dataclass
class FbxNode:
    """One FBX node: a name, ordered children and an ordered property list."""

    name: str
    properties: list[FbxProperty] = field(default_factory=list)
    children: list["FbxNode"] = field(default_factory=list)

    # -- construction ------------------------------------------------------- #
    def add(self, child: "FbxNode") -> "FbxNode":
        self.children.append(child)
        return child

    # -- lookup ------------------------------------------------------------- #
    @property
    def attrs(self) -> list[str]:
        return [p.name for p in self.properties]

    def get(self, *names: str, default: Any = None) -> Any:
        """First value whose property name matches (case-insensitive)."""
        wanted = {n.casefold() for n in names}
        for prop in self.properties:
            if prop.name.casefold() in wanted:
                return prop.value
        return default

    def get_prop(self, *names: str) -> FbxProperty | None:
        wanted = {n.casefold() for n in names}
        for prop in self.properties:
            if prop.name.casefold() in wanted:
                return prop
        return None

    def find(self, *path: str) -> "FbxNode | None":
        """Walk a child path, matching names case-insensitively."""
        node: FbxNode | None = self
        for segment in path:
            if node is None:
                return None
            target = segment.casefold()
            node = next((c for c in node.children if c.name.casefold() == target), None)
        return node

    def find_all(self, name: str) -> list["FbxNode"]:
        target = name.casefold()
        return [c for c in self.children if c.name.casefold() == target]

    def walk(self) -> Iterator["FbxNode"]:
        yield self
        for child in self.children:
            yield from child.walk()

    def walk_named(self, name: str) -> Iterator["FbxNode"]:
        target = name.casefold()
        for node in self.walk():
            if node.name.casefold() == target:
                yield node

    # -- common shapes ------------------------------------------------------ #
    def int64(self, name: str, default: int = -1) -> int:
        prop = self.get_prop(name)
        return prop.as_int(default) if prop else default

    def int32(self, name: str, default: int = 0) -> int:
        return self.int64(name, default)

    def float(self, name: str, default: float = 0.0) -> float:
        prop = self.get_prop(name)
        return prop.as_float(default) if prop else default

    def string(self, name: str, default: str = "") -> str:
        prop = self.get_prop(name)
        return prop.as_str(default) if prop else default

    def prop(self, index: int, default: Any = None) -> Any:
        """Value of the n-th property, whatever it is called.

        FBX object records (Model, Deformer, AnimationStack, Connection, ...)
        store their fields positionally with empty property names, so
        positional access is the only reliable way to read them.
        """
        if 0 <= index < len(self.properties):
            return self.properties[index].value
        return default

    def prop_int(self, index: int, default: int = -1) -> int:
        if 0 <= index < len(self.properties):
            return self.properties[index].as_int(default)
        return default

    @property
    def object_id(self) -> int:
        """Object id: the first property of every FBX object record."""
        if self.properties:
            value = self.properties[0].value
            if isinstance(value, int):
                return value
            if isinstance(value, float):
                return int(value)
        return -1

    def named(self, name: str, index: int, default: Any = None) -> Any:
        """Named property if present, otherwise the n-th one."""
        prop = self.get_prop(name)
        if prop is not None:
            return prop.value
        return self.prop(index, default)

    @property
    def primary_name(self) -> str:
        """Object name: the first string-valued property of the record.

        FBX 7.4 stores ``Model``/``Deformer`` names as a property rather than
        as a ``Name`` child node, so the name has to be recovered from the
        property list.
        """
        for prop in self.properties:
            if isinstance(prop.value, (str, bytes)):
                name = decode_fbx_string(prop.value)
                if name:
                    return name
        return ""

    @property
    def subclass(self) -> str:
        """Sub-class string, e.g. ``"LimbNode"`` or ``"Mesh"``."""
        names = [
            decode_fbx_string(p.value)
            for p in self.properties
            if isinstance(p.value, (str, bytes))
        ]
        if len(names) >= 2:
            return names[1]
        return ""

    def bool(self, name: str, default: bool = False) -> bool:
        prop = self.get_prop(name)
        return prop.as_bool(default) if prop else default

    def prop70(self, name: str, index: int = -1, default: Any = None) -> Any:
        """Read a ``Properties70/P`` entry by name.

        FBX 6+ keeps the optional record properties in a ``Properties70``
        child, one ``P`` node each::

            P: "LocalStart", "KTime", "Time", "", <value>

        The trailing ``index`` selects the N-th property of the match, so
        ``prop70("Lcl Translation", 3)`` reads the X component while
        ``prop70("UpAxis", 4)`` reads the enum value itself.
        """
        holder = self.find("Properties70")
        if holder is None:
            return default
        target = name.casefold()
        for entry in holder.children:
            if not entry.properties:
                continue
            key = entry.properties[0]
            if not isinstance(key.value, (str, bytes)):
                continue
            if decode_fbx_string(key.value).casefold() != target:
                continue
            if 0 <= index < len(entry.properties):
                return entry.properties[index].value
            return entry.properties[-1].value
        return default

    def prop70_names(self) -> list[str]:
        """Every ``Properties70/P`` key declared on this record."""
        holder = self.find("Properties70")
        if holder is None:
            return []
        out: list[str] = []
        for entry in holder.children:
            if entry.properties and isinstance(entry.properties[0].value, (str, bytes)):
                out.append(decode_fbx_string(entry.properties[0].value))
        return out

    def prop70_vec3(
        self, name: str, default: Sequence[float] = (0.0, 0.0, 0.0)
    ) -> list[float]:
        """Read a ``Vector3D`` / ``Vector`` / ``Color`` Properties70 entry.

        FBX 7.4 writes ``Lcl Translation``, ``Lcl Rotation``, ``Lcl Scaling``
        and the pre/post rotations as ``P`` records with the three components
        at positions 4, 5, 6.  They are *not* properties of the node itself,
        so reading them positionally yields zeros for a whole rig.
        """
        holder = self.find("Properties70")
        if holder is None:
            return list(default)
        target = name.casefold()
        for entry in holder.children:
            if not entry.properties:
                continue
            key = entry.properties[0]
            if not isinstance(key.value, (str, bytes)):
                continue
            if decode_fbx_string(key.value).casefold() != target:
                continue
            values = [p.value for p in entry.properties[1:]]
            numbers: list[float] = []
            for value in values:
                if isinstance(value, (int, float)):
                    numbers.append(float(value))
                elif isinstance(value, (list, tuple)):
                    numbers.extend(float(v) for v in value)
            if len(numbers) >= 3:
                return numbers[:3]
            if numbers:
                return [numbers[0], 0.0, 0.0]
            return list(default)
        return list(default)

    def prop70_transform(
        self, *names: str, default: Sequence[float] = (0.0, 0.0, 0.0)
    ) -> list[float]:
        """First of ``names`` that resolves to a vector in Properties70."""
        for name in names:
            holder = self.find("Properties70")
            if holder is None:
                return list(default)
            target = name.casefold()
            for entry in holder.children:
                if not entry.properties:
                    continue
                key = entry.properties[0]
                if not isinstance(key.value, (str, bytes)):
                    continue
                if decode_fbx_string(key.value).casefold() == target:
                    return self.prop70_vec3(name, default)
        return list(default)

    def vec3(self, name: str, default: Sequence[float] = (0.0, 0.0, 0.0)) -> list[float]:
        prop = self.get_prop(name)
        return prop.as_vector(3) if prop else list(default)

    def __repr__(self) -> str:  # pragma: no cover - debugging aid
        return f"FbxNode({self.name!r}, props={self.attrs}, children={len(self.children)})"


@dataclass
class FbxDocument:
    """A parsed FBX file."""

    version: int = 0
    creator: str = ""
    is_binary: bool = True
    root: FbxNode = field(default_factory=lambda: FbxNode("__root__"))
    #: Number of damaged regions the binary reader had to resynchronise past.
    #: Non-zero means some nodes were skipped; the caller may want to warn.
    resync_count: int = 0

    # -- convenience accessors ---------------------------------------------- #
    @property
    def objects(self) -> FbxNode | None:
        return self.root.find("Objects")

    @property
    def connections(self) -> list[FbxNode]:
        node = self.root.find("Connections")
        return node.children if node else []

    @property
    def global_settings(self) -> FbxNode | None:
        return self.root.find("GlobalSettings")

    def _global_setting(self, name: str):
        """Look up a ``GlobalSettings`` template property.

        FBX 6+ stores these as ``Properties70/P`` children whose first
        property is the setting name; older files put them on the node
        itself.  Both shapes are handled.
        """
        settings = self.global_settings
        if settings is None:
            return None
        value = settings.get(name)
        if value is not None:
            return value
        properties = settings.find("Properties70")
        if properties is None:
            return None
        for prop in properties.find_all("P"):
            if prop.primary_name.strip().casefold() == name.casefold():
                # Layout: "Name", "Type", "TypeFlags", "Flags", value...
                candidates = prop.properties[4:]
                return candidates[0].value if candidates else None
        return None

    @property
    def up_axis(self) -> str | None:
        """Declared up axis (``"Y"``, ``"-Z"``, ...)."""
        axis = self._global_setting("UpAxis")
        if axis is None:
            return None
        text = str(axis).strip()
        if text.isdigit():
            # FBX 7.x stores the axis as an enum: 0 = X, 1 = Y, 2 = Z.
            text = {"0": "X", "1": "Y", "2": "Z"}.get(text, text)
        sign = self._global_setting("UpAxisSign")
        if sign is not None:
            try:
                if int(sign) < 0 and not text.startswith("-"):
                    text = "-" + text
            except (TypeError, ValueError):
                pass
        return text

    @property
    def unit_scale_factor(self) -> float | None:
        value = self._global_setting("UnitScaleFactor")
        if value is None:
            return None
        try:
            return float(value)
        except (TypeError, ValueError):
            return None

    def describe(self) -> str:
        nodes = sum(1 for _ in self.root.walk())
        objects = self.objects
        model_count = len(objects.find_all("Model")) if objects else 0
        deformer_count = len(objects.find_all("Deformer")) if objects else 0
        return (
            f"FBX {self.version} ({'binary' if self.is_binary else 'ascii'}) "
            f"creator={self.creator!r} nodes={nodes} models={model_count} "
            f"deformers={deformer_count} resyncs={self.resync_count}"
        )
