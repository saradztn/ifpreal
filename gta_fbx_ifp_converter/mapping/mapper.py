"""Semantic + geometry-aware bone mapping.

Maps a source rig onto a target GTA ped skeleton.  The rules run in order of
decreasing trust, and every match records *why* it was made, because a mapping
the user cannot audit is a mapping they cannot trust:

1. **Exact role + side.**  Both the anatomical role and the side agree.
2. **Chain position.**  Geometry decides: which source bone lies between the
   clavicle and the hand, for instance.
3. **Geometric role.**  Where a name is missing or misleading, the bone's
   position and direction in the rest pose identify it -- a bone whose parent
   is the pelvis and which points downward is a thigh, whatever it is called.
4. **Unmapped.**  Explicitly recorded, never guessed.

Side is never inferred from geometry alone.  A left/right swap is the worst
failure this tool can produce and it is the one that survives visual review
most easily, so sides come from the name and are checked, not deduced.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from enum import Enum
from typing import Sequence

import numpy as np

from ..core import mathx
from .aliases import (
    is_terminal_finger,
    load_aliases,
    normalize_key,
    role_from_name,
    side_of,
)
from .roles import ARM_ORDER, LEG_ORDER, SPINE_ORDER, Role


class MatchKind(str, Enum):
    """How a source bone was matched, strongest first."""

    EXACT = "exact"
    ALIAS = "alias"
    CHAIN = "chain"
    GEOMETRY = "geometry"
    MANUAL = "manual"
    UNMAPPED = "unmapped"


#: Confidence attached to each match kind.  ``check`` refuses to export below
#: the threshold the caller sets; nothing here is allowed to reach 1.0
#: implicitly.
CONFIDENCE: dict[MatchKind, float] = {
    MatchKind.EXACT: 1.00,
    MatchKind.MANUAL: 1.00,
    MatchKind.ALIAS: 0.90,
    MatchKind.CHAIN: 0.75,
    MatchKind.GEOMETRY: 0.60,
    MatchKind.UNMAPPED: 0.00,
}


class Side(str, Enum):
    LEFT = "left"
    RIGHT = "right"
    CENTER = "center"

    def opposite(self) -> "Side":
        if self is Side.LEFT:
            return Side.RIGHT
        if self is Side.RIGHT:
            return Side.LEFT
        return Side.CENTER


def _side_from_name(name: str) -> Side:
    text = side_of(name)
    return Side(text) if text in ("left", "right") else Side.CENTER


@dataclass
class BoneMatch:
    """One source bone's assignment."""

    source_index: int
    source_name: str
    role: Role
    side: Side
    target_index: int | None
    target_name: str = ""
    target_tag: int | None = None
    kind: MatchKind = MatchKind.UNMAPPED
    confidence: float = 0.0
    reason: str = ""
    user_edited: bool = False
    #: The bone above this one in the source rig, or -1 for a root.  Kept on
    #: the match because the retarget stage needs to know whether a bone's
    #: source parent and target parent are themselves a matched pair, and that
    #: question can only be answered here.
    source_parent: int = -1
    target_parent: int = -1

    @property
    def is_mapped(self) -> bool:
        return self.target_index is not None

    def to_dict(self) -> dict:
        return {
            "source_index": self.source_index,
            "source_name": self.source_name,
            "role": self.role.value,
            "side": self.side.value,
            "target_index": self.target_index,
            "target_name": self.target_name,
            "target_tag": self.target_tag,
            "kind": self.kind.value,
            "confidence": round(self.confidence, 4),
            "reason": self.reason,
            "user_edited": self.user_edited,
            "source_parent": self.source_parent,
            "target_parent": self.target_parent,
        }


@dataclass
class MappingResult:
    """The whole mapping, plus what the user changed."""

    matches: list[BoneMatch] = field(default_factory=list)
    aliases: dict[str, tuple[str, ...]] = field(default_factory=dict)
    warnings: list[str] = field(default_factory=list)

    def by_source(self) -> dict[int, BoneMatch]:
        return {m.source_index: m for m in self.matches}

    @property
    def mapped(self) -> list[BoneMatch]:
        return [m for m in self.matches if m.is_mapped]

    @property
    def unmapped(self) -> list[BoneMatch]:
        return [m for m in self.matches if not m.is_mapped]

    @property
    def mean_confidence(self) -> float:
        """Mean confidence over the source bones that carry animation.

        Weighting by the whole rig would let a correct 20-bone spine inflate
        the score past the one bone that is actually wrong.
        """
        return self.quality_score()

    def quality_score(self) -> float:
        scores = [m.confidence for m in self.matches if m.confidence > 0.0]
        return float(sum(scores) / len(scores)) if scores else 0.0

    def by_role(self, role: Role, side: Side) -> list[BoneMatch]:
        return [m for m in self.matches if m.role is role and m.side is side]

    def target_conflicts(self) -> list[tuple[BoneMatch, BoneMatch]]:
        """Pairs of source bones fighting over the same target bone."""
        seen: dict[int, BoneMatch] = {}
        conflicts = []
        for match in self.mapped:
            other = seen.get(match.target_index)
            if other is not None:
                conflicts.append((other, match))
            else:
                seen[match.target_index] = match
        return conflicts

    def describe(self) -> str:
        lines = [
            f"mapped {len(self.mapped)}/{len(self.matches)} source bones, "
            f"mean confidence {self.mean_confidence:.3f}"
        ]
        for match in self.matches:
            if not match.is_mapped:
                continue
            lines.append(
                f"  {match.source_name!r:28s} -> [{match.target_tag}] "
                f"{match.target_name!r:18s} {match.kind.value:9s} "
                f"{match.confidence:.2f} {match.reason}"
            )
        for match in self.unmapped:
            lines.append(
                f"  {match.source_name!r:28s} -> (unmapped) {match.reason}"
            )
        return "\n".join(lines)

    def to_dict(self) -> dict:
        return {
            "matches": [m.to_dict() for m in self.matches],
            "mapped": len(self.mapped),
            "total": len(self.matches),
            "mean_confidence": round(self.quality_score(), 4),
            "warnings": list(self.warnings),
        }


# --------------------------------------------------------------------------- #
# geometry helpers
# --------------------------------------------------------------------------- #
def _unit(vector: np.ndarray) -> np.ndarray:
    norm = float(np.linalg.norm(vector))
    return vector / norm if norm > 1e-12 else np.array([0.0, 0.0, 0.0])


def bone_direction(bind_world: np.ndarray) -> np.ndarray:
    """Where a bone points, in world space: its local Y axis.

    FBX bones run down their own +Y, and the GTA ped DFFs are laid out the
    same way (a thigh's bind offset is along its local Y).  Using the bone
    axis rather than "parent to child" keeps the direction meaningful even
    for a chain whose rest offsets are all zero.
    """
    return _unit(bind_world[:3, 1])


def _chain_indices(bones: Sequence, start: int) -> list[int]:
    """Indices from ``start`` down its subtree, depth first, excluding start."""
    out: list[int] = []
    stack = [start]
    while stack:
        current = stack.pop(0)
        for child in bones[current].children:
            out.append(child)
            stack.append(child)
    return out


def _ancestors(bones: Sequence, index: int) -> list[int]:
    out: list[int] = []
    current = bones[index].parent
    while current >= 0 and len(out) < 32:
        out.append(current)
        current = bones[current].parent
    return out


def _role_chain(
    bones: Sequence,
    role: Role,
    side: Side,
    start_index: int,
) -> list[int]:
    """Source bones playing a limb chain, in order, starting after the start.

    ``start_index`` is the bone whose *children* begin the chain -- the
    shoulder for an arm, the pelvis for a leg.
    """
    found: list[int] = []
    current = start_index
    guard = 0
    while current >= 0 and guard < 16:
        guard += 1
        child = -1
        for candidate in bones[current].children:
            if _side_from_name(bones[candidate].name) is not side:
                continue
            if role_from_name(bones[candidate].name, _CURRENT_ALIASES) is role:
                child = candidate
                break
        if child < 0:
            break
        found.append(child)
        current = child
    return found


#: Set by :func:`map_rig` so the chain helper can see the active alias table
#: without threading it through every call.  Module-level state is normally a
#: smell; here the alternative is a closure per call, and the value is set
#: once at the start of a mapping and never mutated.
_CURRENT_ALIASES: dict[str, tuple[str, ...]] = {}


def map_rig(
    source_bones: Sequence,
    target_skeleton,
    aliases_path: str | None = None,
) -> MappingResult:
    """Map a source rig onto a target skeleton.

    Returns every source bone, mapped or not, with the reason for each.
    """
    global _CURRENT_ALIASES
    aliases = load_aliases(aliases_path)
    _CURRENT_ALIASES = aliases

    matches: list[BoneMatch] = []
    warnings: list[str] = []

    # -- pass 1: names ----------------------------------------------------- #
    for bone in source_bones:
        side = _side_from_name(bone.name)
        role = role_from_name(bone.name, aliases)
        if role is None:
            # A finger chain's tip has no role of its own; inherit "finger".
            if is_terminal_finger(bone.name):
                role = Role.FINGER
        matches.append(BoneMatch(
            source_index=bone.index,
            source_name=bone.name,
            role=role or Role.UNKNOWN,
            side=side,
            target_index=None,
            kind=MatchKind.UNMAPPED,
            confidence=0.0,
            source_parent=bone.parent if bone.parent is not None else -1,
            target_parent=-1,
        ))

    # -- pass 1b: disambiguate within each limb chain ---------------------- #
    # "LeftUpLeg" -> thigh and "LeftLeg" -> calf are unambiguous, but a rig
    # that names both legs "Leg"/"Leg" is not.  Within a chain, once a role
    # is used the next bone cannot be it again; that turns an ambiguous name
    # into a definite one without guessing.
    _resolve_ambiguous_leg_roles(source_bones, matches)

    by_name = {m.source_name: m for m in matches}
    claimed: dict[int, BoneMatch] = {}

    def assign(match: BoneMatch, target_bone, kind: MatchKind, reason: str) -> bool:
        if target_bone is None or not target_bone.is_addressable:
            return False
        if target_bone.index in claimed:
            return False
        claimed[target_bone.index] = match
        match.target_index = target_bone.index
        match.target_name = target_bone.name
        match.target_tag = target_bone.resolved_tag
        match.target_parent = (
            target_bone.parent if target_bone.parent is not None else -1
        )
        match.kind = kind
        match.confidence = CONFIDENCE[kind]
        match.reason = reason
        return True

    # -- pass 2: role+side onto the target --------------------------------- #
    for match in matches:
        if match.role is Role.UNKNOWN:
            continue
        target_bone = _find_role_target(target_skeleton, match, aliases)
        if target_bone is not None:
            # Every target bone may carry only one source bone.
            conflict = claimed.get(target_bone.index)
            if conflict is not None:
                match.reason = (
                    f"role {match.role.value}/{match.side.value} already taken by "
                    f"{conflict.source_name!r}"
                )
                continue
            claimed[target_bone.index] = match
            match.target_index = target_bone.index
            match.target_name = target_bone.name
            match.target_tag = target_bone.resolved_tag
            match.target_parent = (
                target_bone.parent if target_bone.parent is not None else -1
            )
            match.kind = MatchKind.ALIAS
            match.confidence = CONFIDENCE[MatchKind.ALIAS]
            match.reason = f"role {match.role.value} + {match.side.value}"

    # -- pass 3: chains, for roles the target has but names did not fill -- #
    _fill_from_chains(source_bones, matches, by_name, target_skeleton, claimed, warnings)

    # -- pass 4: geometry for whatever is still unmapped ------------------- #
    _fill_from_geometry(source_bones, matches, target_skeleton, claimed, warnings)

    # -- report ------------------------------------------------------------- #
    for match in matches:
        if not match.is_mapped and not match.reason:
            match.reason = (
                f"no target bone plays {match.role.value}/{match.side.value}"
                if match.role is not Role.UNKNOWN
                else f"unknown role for {match.source_name!r}"
            )
    for first, second in MappingResult(matches).target_conflicts():
        warnings.append(
            f"{first.source_name!r} and {second.source_name!r} both claim "
            f"[{first.target_tag}] {first.target_name!r}"
        )

    return MappingResult(matches=matches, aliases=aliases, warnings=warnings)


def _find_role_target(skeleton, match: BoneMatch, aliases) -> object | None:
    """The target bone that should play this source bone's role.

    Asks the target for a bone of the right **role**, which the resolver
    already derived from the DFF's own HAnim tag, rather than assuming a tag
    number.
    """
    wanted = target_role_of(skeleton, match.role, match.side)
    if wanted is None:
        return None
    for bone in skeleton.bones:
        if not bone.is_addressable:
            continue
        if bone.side != match.side.value:
            continue
        if getattr(bone, "canonical_tag", None) == wanted:
            return bone
    return None


def target_role_of(skeleton, role: Role, side: Side) -> int | None:
    """Which GTA tag plays ``role`` on ``side`` in this particular DFF.

    Side matters: the GTA ped numbering is mirrored (``41 L Thigh`` /
    ``51 R Thigh``), so asking for the left tag and filtering by side finds
    nothing and the right leg silently goes unmapped.  The candidate list is
    therefore filtered by the skeleton's own ``side``, which the DFF resolver
    derived from the frame names.

    Returns ``None`` when the ped has no such bone, which is the honest
    answer for a ped with no fingers or no breasts.
    """
    candidates = _role_tag_candidates(role)
    if side is not Side.CENTER:
        matching = [
            b.resolved_tag for b in skeleton.bones
            if b.is_addressable and b.side == side.value
            and b.resolved_tag in candidates
        ]
        if matching:
            return matching[0]
        return None
    matching = [
        b.resolved_tag for b in skeleton.bones
        if b.is_addressable and b.side == "center"
        and b.resolved_tag in candidates
    ]
    return matching[0] if matching else None


def _role_tag_candidates(role: Role) -> list[int]:
    """GTA tags that can play a role, most specific first.

    Built from the SA ped bone list; a side is applied by the caller through
    the 1/2, 3/4 ... mirroring in the actual tag numbers.
    """
    return _TAG_CANDIDATES.get(role, [])


#: role -> GTA tag numbers that can play it.  Pairs are ``(center, +50)``
#: style mirrors resolved by the resolver, so both are listed and the
#: caller filters on the bone's own side.
_TAG_CANDIDATES: dict[Role, list[int]] = {
    Role.ROOT: [0],
    Role.PELVIS: [1],
    Role.SPINE1: [2],
    Role.SPINE2: [3],
    Role.SPINE3: [2, 3],
    Role.NECK: [4],
    Role.HEAD: [5],
    Role.JAW: [8],
    Role.BROW: [6, 7],
    Role.EYE: [6, 7],
    Role.CLAVICLE: [21, 31],
    Role.UPPER_ARM: [22, 32],
    Role.FOREARM: [23, 33],
    Role.HAND: [24, 34],
    Role.FINGER: [25, 35],
    Role.THIGH: [41, 51],
    Role.CALF: [42, 52],
    Role.FOOT: [43, 53],
    Role.TOE: [44, 54],
    Role.BREAST: [301, 302],
    Role.BELLY: [201],
}


def _fill_from_chains(
    source_bones: Sequence,
    matches: list[BoneMatch],
    by_name: dict[str, BoneMatch],
    skeleton,
    claimed: dict[int, BoneMatch],
    warnings: list[str],
) -> None:
    """Use chain position to place bones the name pass could not.

    A ped with one spine bone has one tag for the whole torso; a source rig
    with three spine bones then has to be reduced, and which bones survive is
    a question of geometry rather than names.
    """
    by_source = {m.source_index: m for m in matches}

    def spine_chain(side: Side) -> list[int]:
        pelvis = by_source.get(_find_pelvis(source_bones, by_source))
        if pelvis is None:
            return []
        return _role_chain(source_bones, Role.SPINE1, side, pelvis.source_index)

    # Torso: a source chain longer than the target's spine gets reduced to
    # the bones the target actually has, keeping the evenly-spaced subset
    # that best matches the target's proportions.
    torso = [m for m in matches if m.role in SPINE_ORDER and m.side is Side.CENTER]
    free = [m for m in torso if not m.is_mapped]
    if free:
        targets = [
            b for b in skeleton.bones
            if b.is_addressable and b.side == "center"
            and b.resolved_tag in _role_tag_candidates(Role.SPINE1)
            + _role_tag_candidates(Role.SPINE2)
            + _role_tag_candidates(Role.SPINE3)
        ]
        free.sort(key=lambda m: m.source_index)
        if targets and free:
            keep = _spread(len(free), len(targets))
            for slot, (match, target_bone) in enumerate(zip(free, targets)):
                if slot not in keep and len(free) > len(targets):
                    match.reason = (
                        f"reduced: {len(free)} source spine bones, "
                        f"{len(targets)} target"
                    )
                    continue
                if target_bone.index in claimed:
                    continue
                claimed[target_bone.index] = match
                match.target_index = target_bone.index
                match.target_name = target_bone.name
                match.target_tag = target_bone.resolved_tag
                match.kind = MatchKind.CHAIN
                match.confidence = CONFIDENCE[MatchKind.CHAIN]
                match.reason = "spine chain position"

    # Limbs: walk the source chain and hand out the target's bones in order.
    for side in (Side.LEFT, Side.RIGHT):
        root = _find_limb_root(source_bones, by_source, side, Role.CLAVICLE)
        if root is None:
            root = _find_limb_root(source_bones, by_source, side, Role.THIGH)
        if root is None:
            continue
        chain_indices = [root] + _chain_indices(source_bones, root)
        free = [
            by_source[i] for i in chain_indices
            if i in by_source and not by_source[i].is_mapped
            and by_source[i].role in ARM_ORDER + LEG_ORDER
        ]
        if not free:
            continue
        target_bones = [
            b for b in skeleton.bones
            if b.is_addressable and b.side == side.value
            and any(
                b.resolved_tag in _role_tag_candidates(r)
                for r in ARM_ORDER + LEG_ORDER
            )
        ]
        if not target_bones:
            continue
        # Greedy in chain order, respecting the target's own depth order.
        target_bones.sort(key=lambda b: b.depth)
        used: set[int] = set()
        for match in free:
            want = match.role
            choice = None
            for target_bone in target_bones:
                if target_bone.index in used or target_bone.index in claimed:
                    continue
                if want in _role_tag_candidates(Role.UNKNOWN):
                    continue
                if target_bone.resolved_tag in _role_tag_candidates(want):
                    choice = target_bone
                    break
            if choice is None:
                match.reason = (
                    f"no target bone in the {side.value} chain plays "
                    f"{want.value}"
                )
                continue
            used.add(choice.index)
            claimed[choice.index] = match
            match.target_index = choice.index
            match.target_name = choice.name
            match.target_tag = choice.resolved_tag
            match.kind = MatchKind.CHAIN
            match.confidence = CONFIDENCE[MatchKind.CHAIN]
            match.reason = f"{side.value} chain position for {want.value}"


#: Roles a leg chain is made of, in order from the hip outwards.
_LEG_ROLE_SET: frozenset[Role] = frozenset(LEG_ORDER)


def _resolve_ambiguous_leg_roles(
    source_bones: Sequence,
    matches: list[BoneMatch],
) -> None:
    """Assign thigh/calf/foot/toe by position when the name is ambiguous.

    A bone named ``LeftLeg`` matches both ``thigh`` and ``calf``, and
    whichever the alias table happens to list first would put the knee on the
    GTA thigh tag.  Within one leg chain the roles are strictly ordered from
    the hip outwards, so a bone takes the role the chain has not used yet --
    the only reading that makes the chain anatomically possible.

    Bones whose names are unambiguous keep what the name gave them.
    """
    for side in (Side.LEFT, Side.RIGHT):
        leg = [m for m in matches if m.side is side and m.role in _LEG_ROLE_SET]
        if len(leg) < 2:
            continue
        leg.sort(key=lambda m: source_bones[m.source_index].depth)
        # "LeftUpLeg" is a thigh and "LeftLeg" is a calf; neither is
        # ambiguous.  Only re-assign when the chain contains a repeat.
        repeated = [r for r in LEG_ORDER if sum(1 for m in leg if m.role is r) > 1]
        if not repeated:
            continue
        used: set[Role] = set()
        for match in leg:
            if match.role in used:
                # This bone repeats a role; give it the next unused one.
                free = [r for r in LEG_ORDER if r not in used]
                if free:
                    match.role = free[0]
            used.add(match.role)


def _spread(count: int, keep: int) -> set[int]:
    """Which of ``count`` items to keep when reducing to ``keep`` items.

    Evenly spaced and anchored at both ends, so a three-bone spine reduced
    to one keeps the pelvis-side bone rather than an arbitrary middle one.
    """
    if keep >= count:
        return set(range(count))
    if keep <= 0:
        return set()
    if keep == 1:
        return {0}
    return {round(i * (count - 1) / (keep - 1)) for i in range(keep)}


def _find_pelvis(source_bones: Sequence, by_source: dict[int, BoneMatch]) -> int | None:
    for match in by_source.values():
        if match.role is Role.PELVIS and match.side is Side.CENTER:
            return match.source_index
    for bone in source_bones:
        if bone.index == 0 and bone.is_root:
            return bone.index
    return None


def _find_limb_root(
    source_bones: Sequence,
    by_source: dict[int, BoneMatch],
    side: Side,
    first: Role,
) -> int | None:
    candidates = [
        m for m in by_source.values()
        if m.role is first and m.side is side
    ]
    if not candidates:
        # A rig may have no clavicle bone; start from the arm instead.
        for role in (Role.UPPER_ARM, Role.THIGH):
            candidates = [m for m in by_source.values() if m.role is role and m.side is side]
            if candidates:
                break
    if not candidates:
        return None
    return min(candidates, key=lambda m: m.source_index).source_index


def _fill_from_geometry(
    source_bones: Sequence,
    matches: list[BoneMatch],
    skeleton,
    claimed: dict[int, BoneMatch],
    warnings: list[str],
) -> None:
    """Last resort: identify a bone by where it is and which way it points.

    Only ever used for bones whose name said nothing.  The result is marked
    low-confidence so that ``check`` will ask the user before exporting.
    """
    if not matches:
        return
    hip = None
    for match in matches:
        if match.role is Role.PELVIS:
            hip = match
            break
    if hip is None:
        return
    hip_bone = source_bones[hip.source_index]
    hip_world = hip_bone.bind_world[:3, 3]
    hip_up = _unit(hip_bone.bind_world[:3, 1])

    for match in matches:
        if match.is_mapped:
            continue
        bone = source_bones[match.source_index]
        world = bone.bind_world[:3, 3]
        direction = bone_direction(bone.bind_world)
        relative = world - hip_world
        below = float(np.dot(relative, hip_up)) < 0.0
        lateral = abs(float(relative[0])) > 0.35 * max(
            1e-6, float(np.linalg.norm(relative))
        )
        inferred: Role | None = None
        if below and lateral:
            inferred = Role.THIGH
        elif below:
            inferred = Role.CALF
        elif relative[2] > 0.25 * max(1e-6, float(np.linalg.norm(relative))):
            inferred = Role.HEAD
        if inferred is None:
            continue
        side = Side.LEFT if relative[0] > 0 else Side.RIGHT
        if inferred in (Role.HEAD,):
            side = Side.CENTER
        match.role = inferred
        match.side = side
        target_bone = _find_role_target(skeleton, match, _CURRENT_ALIASES)
        if target_bone is not None and target_bone.index not in claimed:
            claimed[target_bone.index] = match
            match.target_index = target_bone.index
            match.target_name = target_bone.name
            match.target_tag = target_bone.resolved_tag
            match.kind = MatchKind.GEOMETRY
            match.confidence = CONFIDENCE[MatchKind.GEOMETRY]
            match.reason = f"rest-pose geometry inferred {inferred.value}/{side.value}"
        else:
            match.reason = (
                f"geometry suggests {inferred.value}/{side.value} but the target "
                "has no such bone"
            )


__all__ = [
    "Role",
    "Side",
    "MatchKind",
    "BoneMatch",
    "MappingResult",
    "map_rig",
    "target_role_of",
    "bone_direction",
    "CONFIDENCE",
]
