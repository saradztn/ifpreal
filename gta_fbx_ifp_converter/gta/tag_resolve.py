"""Decide which GTA HAnim tag each frame of the target DFF really is.

Why this module exists
======================
A DFF frame carries three independent pieces of information:

* its **geometry** (local matrix, parent index) -- always meaningful;
* its **name** -- a label a DCC tool wrote, usually a real GTA bone name;
* the **HAnim plugin's** ``node_id`` for it.

On paper ``node_id`` is the authority: it is the field the IFP object's
``BoneID`` has to match.  In practice the shipped GTA San Andreas ped DFFs
disagree with themselves.  Measured on ``testdata/male01.dff`` (identical in
``army.dff``, ``claude.dff``, ``ballas1.dff``, ``copgrl3.dff``)::

    pos frame  dff name        node_id   canonical name of node_id
      2     2  ' Pelvis'           2     Spine
      3     3  'R Thigh'           3     Spine1
     17    17  'L UpperArm'       23     R Forearm
     24    24  'R Hand'           41     L Thigh

0 of 32 frames agree.  The ``node_id`` array is simply the canonical GTA
bone list in canonical order, left over from a differently ordered skeleton,
while the frames themselves are stored in an authoring order that is
*isomorphic* to the canonical ped hierarchy (verified: the parent tree has
the same shape and the same left/right assignment as the canonical ped).

So blindly trusting ``node_id`` would write IFP tracks that animate the
thigh when the head track fires.  What this module does instead:

1. The **DFF still decides** the set of tags that exist (the HAnim array is
   the only place the tag set is written down) and the frame hierarchy and
   bind transforms, which are always trustworthy.
2. Each frame's identity is resolved from, in order of trust:
   ``node_id`` when it agrees with the name, the canonical **name** table,
   then **structural** matching (position in the hierarchy + left/right
   side) for frames whose name is missing or misleading.
3. Every disagreement is recorded as a diagnostic.  Nothing is silently
   dropped and nothing is silently rewritten.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Iterable, Sequence

from .bones import SaBoneTag, bone_tag_from_name, describe_tag, is_ped_body_bone

__all__ = [
    "TagResolution",
    "TagDiagnostic",
    "CANONICAL_PED_TREE",
    "resolve_tags",
    "canonical_tree_tags",
]


# --------------------------------------------------------------------------- #
# the canonical GTA SA ped hierarchy, used only as a *fallback* matcher
# --------------------------------------------------------------------------- #
#: ``tag -> (parent tag, side)`` for the 32 body bones of a GTA SA ped.
#: ``side`` is ``"left"``, ``"right"`` or ``"center"``.
CANONICAL_PED_TREE: dict[int, tuple[int | None, str]] = {
    0: (None, "center"),      # Root
    1: (0, "center"),         # Pelvis
    2: (1, "center"),         # Spine
    3: (2, "center"),         # Spine1
    4: (3, "center"),         # Neck
    5: (4, "center"),         # Head
    6: (5, "left"),           # L Brow
    7: (5, "right"),          # R Brow
    8: (5, "center"),         # Jaw
    21: (4, "right"),         # R Clavicle
    22: (21, "right"),        # R UpperArm
    23: (22, "right"),        # R Forearm
    24: (23, "right"),        # R Hand
    25: (24, "right"),        # R Finger
    26: (25, "right"),        # R Finger01
    31: (4, "left"),          # L Clavicle
    32: (31, "left"),         # L UpperArm
    33: (32, "left"),         # L Forearm
    34: (33, "left"),         # L Hand
    35: (34, "left"),         # L Finger
    36: (35, "left"),         # L Finger01
    41: (1, "left"),          # L Thigh
    42: (41, "left"),         # L Calf
    43: (42, "left"),         # L Foot
    44: (43, "left"),         # L Toe0
    51: (1, "right"),         # R Thigh
    52: (51, "right"),        # R Calf
    53: (52, "right"),        # R Foot
    54: (53, "right"),        # R Toe0
    201: (2, "center"),       # Belly
    301: (3, "right"),        # R breast
    302: (3, "left"),         # L breast
}


def canonical_tree_tags() -> list[int]:
    """Canonical tags in a stable, parents-before-children order."""
    out: list[int] = []

    def visit(tag: int) -> None:
        out.append(tag)
        for child, (parent, _side) in CANONICAL_PED_TREE.items():
            if parent == tag:
                visit(child)

    visit(0)
    return out


# --------------------------------------------------------------------------- #
# results
# --------------------------------------------------------------------------- #
@dataclass
class TagDiagnostic:
    """One disagreement or gap found while resolving tags."""

    severity: str          # "error" | "warning" | "info"
    code: str
    frame_index: int
    message: str

    def __str__(self) -> str:  # pragma: no cover - formatting only
        return f"{self.severity.upper()}: {self.message}"


@dataclass
class TagResolution:
    """The outcome of resolving every frame of a DFF to an HAnim tag."""

    #: frame index -> effective HAnim tag (the value written into the IFP).
    tag_of_frame: dict[int, int] = field(default_factory=dict)
    #: frame index -> how the tag was decided.
    source_of_frame: dict[int, str] = field(default_factory=dict)
    #: frame index -> the DFF's own HAnim ``node_id`` (``None`` if no plugin).
    hanim_id_of_frame: dict[int, int | None] = field(default_factory=dict)
    diagnostics: list[TagDiagnostic] = field(default_factory=list)

    def bone_id(self, frame_index: int) -> int:
        """IFP bone id for a frame, or ``-1`` when the frame is not addressable."""
        return self.tag_of_frame.get(frame_index, -1)

    def frame_of_tag(self, tag: int) -> int | None:
        for frame, value in self.tag_of_frame.items():
            if value == tag:
                return frame
        return None

    @property
    def addressable_frames(self) -> list[int]:
        return sorted(self.tag_of_frame)

    @property
    def animated_tags(self) -> list[int]:
        return sorted(self.tag_of_frame.values())

    def errors(self) -> list[TagDiagnostic]:
        return [d for d in self.diagnostics if d.severity == "error"]

    def warnings(self) -> list[TagDiagnostic]:
        return [d for d in self.diagnostics if d.severity == "warning"]

    def describe(self) -> str:
        lines = [f"Tag resolution: {len(self.tag_of_frame)} addressable frames"]
        counts: dict[str, int] = {}
        for source in self.source_of_frame.values():
            counts[source] = counts.get(source, 0) + 1
        for source, count in sorted(counts.items()):
            lines.append(f"  {source:<22} {count}")
        for diagnostic in self.diagnostics:
            lines.append(f"  {diagnostic}")
        return "\n".join(lines)


# --------------------------------------------------------------------------- #
# resolution
# --------------------------------------------------------------------------- #
#: How confident we are in a decision, ordered worst to best.
SOURCE_PRIORITY = {
    "hanim-plugin-agrees": 4,
    "canonical-name": 3,
    "structure+name": 2,
    "structure": 1,
    "unresolved": 0,
}


def _side_from_name(name: str) -> str | None:
    """Left/right read from a frame name, or ``None`` when ambiguous."""
    key = _compact(name)
    if not key:
        return None
    left = any(token in key for token in ("left", "lhand", "lfoot", "ltoe", "lcalf",
                                          "lthigh", "lupperarm", "lforearm", "lclavicle",
                                          "lfinger", "lbrow", "lbreast", "larm", "lleg"))
    right = any(token in key for token in ("right", "rhand", "rfoot", "rtoe", "rcalf",
                                           "rthigh", "rupperarm", "rforearm", "rclavicle",
                                           "rfinger", "rbrow", "rbreast", "rarm", "rleg"))
    if "bip01l" in key or "bip01r" in key:
        left = left or "bip01l" in key
        right = right or "bip01r" in key
    if left and not right:
        return "left"
    if right and not left:
        return "right"
    return None


def _compact(name: str) -> str:
    return "".join(ch for ch in (name or "").casefold() if ch.isalnum())


def _side_of_tag(tag: int) -> str:
    entry = CANONICAL_PED_TREE.get(int(tag))
    return entry[1] if entry else "center"


def _shape_of(
    frame: int,
    children: dict[int, list[int]],
    sides: dict[int, str],
    depth: int = 24,
) -> tuple:
    """Side-labelled subtree shape, used to match a frame to a canonical tag."""

    def build(node: int, seen: frozenset[int]) -> tuple:
        if depth <= 0 or node in seen:
            return ()
        kids = []
        for child in children.get(node, []):
            kids.append((sides.get(child, "center"), build(child, seen | {node})))
        return tuple(sorted(kids))

    return build(frame, frozenset())


def _canonical_shape(tag: int, depth: int = 24) -> tuple:
    if depth <= 0:
        return ()

    def build(node: int, seen: frozenset[int]) -> tuple:
        if node in seen:
            return ()
        kids = []
        for child, (_parent, side) in CANONICAL_PED_TREE.items():
            if _parent == node:
                kids.append((side, build(child, seen | {node})))
        return tuple(sorted(kids))

    return build(tag, frozenset())


def resolve_tags(
    frame_names: Sequence[str],
    parents: dict[int, int],
    hanim_id_of_frame: dict[int, int | None],
    *,
    hanim_tags_available: Iterable[int] = (),
) -> TagResolution:
    """Resolve every frame of a DFF to the HAnim tag it should carry.

    ``parents`` maps frame index to parent frame index (``-1``/absent for a
    root).  ``hanim_id_of_frame`` is the DFF's own HAnim ``node_id`` per
    frame.  ``hanim_tags_available`` is the set of tags the DFF's HAnim
    array declares; a resolved tag outside that set is flagged.
    """
    children: dict[int, list[int]] = {i: [] for i in range(len(frame_names))}
    for frame, parent in parents.items():
        if 0 <= parent < len(frame_names) and parent != frame:
            children.setdefault(parent, []).append(frame)
    for key in children:
        children[key].sort()

    # Side of each frame, from its own name and inherited down the tree.
    sides: dict[int, str] = {}
    for frame, name in enumerate(frame_names):
        side = _side_from_name(name)
        if side:
            sides[frame] = side
    # A frame that has no side of its own inherits its parent's: the L/R of
    # an arm chain is stated once, at the clavicle.
    for frame in range(len(frame_names)):
        if frame in sides:
            continue
        parent = parents.get(frame, -1)
        if parent in sides:
            sides[frame] = sides[parent]

    resolution = TagResolution()
    available = {int(t) for t in hanim_tags_available}
    name_tags: dict[int, int] = {}
    for frame, name in enumerate(frame_names):
        tag = bone_tag_from_name(name)
        if tag is not None:
            name_tags[frame] = int(tag)

    # Pass 1: the DFF's HAnim id, but only where the name agrees with it.
    for frame in range(len(frame_names)):
        hanim = hanim_id_of_frame.get(frame)
        if hanim is None:
            continue
        resolution.hanim_id_of_frame[frame] = int(hanim)
        named = name_tags.get(frame)
        if named is not None and named == int(hanim):
            resolution.tag_of_frame[frame] = int(hanim)
            resolution.source_of_frame[frame] = "hanim-plugin-agrees"

    # Pass 2: names alone, for frames the plugin did not settle.
    for frame in range(len(frame_names)):
        if frame in resolution.tag_of_frame:
            continue
        named = name_tags.get(frame)
        if named is None:
            continue
        hanim = hanim_id_of_frame.get(frame)
        if hanim is not None and int(hanim) != named:
            resolution.diagnostics.append(
                TagDiagnostic(
                    "warning",
                    "hanim-name-conflict",
                    frame,
                    f"frame {frame} {frame_names[frame]!r}: the DFF HAnim plugin "
                    f"says id {int(hanim)} ({describe_tag(int(hanim))}) but the "
                    f"frame name is {frame_names[frame]!r} which is "
                    f"{named} ({describe_tag(named)}); using the name",
                )
            )
            resolution.source_of_frame[frame] = "structure+name"
        else:
            resolution.source_of_frame[frame] = "canonical-name"
        resolution.tag_of_frame[frame] = named

    # Pass 3: structure, for frames with no usable name.
    for frame in range(len(frame_names)):
        if frame in resolution.tag_of_frame:
            continue
        best = _match_by_structure(
            frame, children, sides, available | set(name_tags.values())
        )
        if best is None:
            resolution.diagnostics.append(
                TagDiagnostic(
                    "warning",
                    "unresolved-frame",
                    frame,
                    f"frame {frame} {frame_names[frame]!r}: no HAnim tag could be "
                    f"resolved; the frame will not be animated",
                )
            )
            resolution.source_of_frame.setdefault(frame, "unresolved")
            continue
        resolution.tag_of_frame[frame] = best
        resolution.source_of_frame[frame] = "structure"
        resolution.diagnostics.append(
            TagDiagnostic(
                "info",
                "structure-match",
                frame,
                f"frame {frame} {frame_names[frame]!r}: resolved to {best} "
                f"({describe_tag(best)}) from its position in the hierarchy",
            )
        )

    # A tag used twice means the DFF has two frames claiming one bone.
    seen: dict[int, int] = {}
    for frame, tag in sorted(resolution.tag_of_frame.items()):
        if tag in seen:
            resolution.diagnostics.append(
                TagDiagnostic(
                    "error",
                    "duplicate-tag",
                    frame,
                    f"tag {tag} ({describe_tag(tag)}) is claimed by both frame "
                    f"{seen[tag]} and frame {frame}; only the first will animate",
                )
            )
        else:
            seen[tag] = frame

    if available:
        missing = sorted(available - set(resolution.tag_of_frame.values()))
        if missing:
            resolution.diagnostics.append(
                TagDiagnostic(
                    "info",
                    "unused-hanim-tag",
                    -1,
                    "HAnim declares tags no frame claimed: "
                    + ", ".join(f"{t} {describe_tag(t)}" for t in missing),
                )
            )
    return resolution


def _match_by_structure(
    frame: int,
    children: dict[int, list[int]],
    sides: dict[int, str],
    candidates: set[int],
) -> int | None:
    """Find the canonical tag whose subtree shape matches this frame."""
    target = _shape_of(frame, children, sides)
    best: tuple[int, float] | None = None
    for tag in sorted(candidates):
        entry = CANONICAL_PED_TREE.get(int(tag))
        if entry is None:
            continue
        shape = _canonical_shape(int(tag))
        distance = _shape_distance(target, shape)
        if best is None or distance < best[1]:
            best = (int(tag), distance)
    if best is None:
        return None
    if best[1] > 0.0:
        return None
    # Two tags can share a shape only if they are the same bone mirrored;
    # the side label in the shape already separates those.
    return best[0]


def _shape_distance(a: tuple, b: tuple) -> float:
    """0 when two side-labelled subtree shapes are identical."""
    if len(a) != len(b):
        return float(abs(len(a) - len(b))) * 10.0
    total = 0.0
    for (side_a, child_a), (side_b, child_b) in zip(sorted(a), sorted(b)):
        if side_a != side_b:
            total += 1.0
        total += _shape_distance(child_a, child_b)
    return total
