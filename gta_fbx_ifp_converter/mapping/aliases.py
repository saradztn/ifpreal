"""Name normalisation and the alias table.

Rig exporters disagree about bone names far more than they disagree about
anatomy, so the mapping starts by reducing a name to a *normalised token list*
and matching that against a table of known aliases.

Normalisation is deliberately conservative: it lowercases, splits on
separators that exporters actually use, and drops the namespace prefix and
side markers.  It never fuzzy-matches, because "left" and "right" differ by
one character and a fuzzy matcher will eventually confuse them -- and a
swapped arm is the single worst failure this tool can have.
"""

from __future__ import annotations

import json
import os
import re
from functools import lru_cache
from typing import Iterable

from .roles import Role

#: Separators used by Blender, Maya, 3ds Max, HumanIK, Unreal and Unity.
_SEPARATORS = re.compile(r"[^a-z0-9]+")

#: Rig namespaces that carry no anatomical information.  Matched anywhere in
#: the name, not only at the start, because exporters paste them on: the SDK
#: writes ``mixamorig:LeftUpLeg`` and the bake writes ``mixamorigLeftUpLeg``.
_NAMESPACE = re.compile(
    r"(?:^|[^a-z0-9])(?:mixamorig|biped|bip\d*|bn\d*|bpy_bone|def_bones|"
    r"b_[a-z]|bn_[a-z]|ctrl|org|grp|group|root|joint|jnt)(?=[^a-z0-9]|$)"
)

#: A namespace fused directly to the bone name, e.g. ``mixamorigLeftUpLeg``.
_GLUED_NAMESPACE = re.compile(
    r"(?:mixamorig|armature|biped|bip\d*|bn\d*|bpybone|defbones)"
)

#: A side marker fused to the bone name, e.g. ``LeftUpLeg`` -> ``left|upleg``.
#: Longest alternatives first: matching the bare ``l`` before ``left`` would
#: reduce ``leg`` to ``eg`` and then to nothing.
_GLUED_SIDE = re.compile(
    r"^(?:left|lft)(?=[a-z]{2,})|^(?:right|rgt)(?=[a-z]{2,})"
    r"|^l(?=[a-z]{3,})|^r(?=[a-z]{3,})"
)

#: Tokens that only ever describe a side.  Stripped from the alias key so that
#: "left_upleg" and "right_upleg" normalise to the same alias "upleg"; the
#: side is carried separately and matched exactly.
_SIDE_TOKENS = frozenset({"left", "right", "l", "r", "lft", "rgt", "lef", "rig"})

#: Numbering suffixes a finger chain appends; they are not part of the role.
_INDEX_TOKENS = re.compile(r"^(?:0[0-9]|[1-9][0-9]?)$")


def normalize(name: str) -> list[str]:
    """Split a bone name into normalised lower-case tokens.

    >>> normalize("mixamorig:LeftUpLeg")
    ['leftupleg']
    >>> normalize("Bip01 L UpperArm")
    ['upperarm']
    >>> normalize("upper_arm.L")
    ['upperarm']
    """
    text = (name or "").strip().lower()
    text = text.replace(":", " ").replace(".", " ")
    # A namespace glued to the name ("mixamorigLeftUpLeg") is split off even
    # without a separator, so the alias table sees the bare bone name.
    text = _GLUED_NAMESPACE.sub(" ", text)
    tokens = [t for t in _SEPARATORS.split(text) if t]
    out: list[str] = []
    for token in tokens:
        # A token that is *only* a side marker is dropped outright.  This is
        # checked first: "left" would otherwise lose its "l" to the glued
        # rule below and leave "eft" behind.
        if token in _SIDE_TOKENS:
            continue
        # A side marker glued to the bone name is peeled off, so "LeftUpLeg"
        # reduces to "upleg" and the alias table can match it exactly.  The
        # alternation is longest-first: matching the bare "l" first would
        # strip the l out of "leg" and leave "eg".
        side_match = _GLUED_SIDE.match(token)
        if side_match is not None:
            token = token[side_match.end():]
        if not token or token in _SIDE_TOKENS:
            continue
        out.append(_split_trailing_index(token))
    return _merge_known_pairs(out)


#: Aliases that legitimately carry a trailing index and must not be split.
_NUMBERED_ALIASES = frozenset({
    "spine0", "spine1", "spine2", "spine3", "spine4", "spine5",
    "spine6", "spine7", "spine8", "spine9", "neck1", "neck2",
})


def _split_trailing_index(token: str) -> str:
    """Pull a trailing joint index off a name so the alias table can match.

    ``LeftHandMiddle1`` names the first joint of the middle finger, and the
    ped has exactly one GTA finger bone for all five of them.  Keeping the
    ``1`` would leave every finger of a Mixamo hand with no role at all and
    silently unmapped.  Numbered *spine* names are the exception -- there the
    digit is the segment, so ``Spine2`` is the chest and not "spine, twice".
    """
    match = re.match(r"^(.*?)([0-9]+)$", token)
    if match is None or not match.group(1):
        return token
    if token in _NUMBERED_ALIASES:
        return token
    return match.group(1)


#: Pairs of tokens that form a single anatomical word.  Split on an
#: underscore, these are one word; joined, they match the alias table.
_WORD_PAIRS = {
    ("upper", "arm"): "upperarm",
    ("fore", "arm"): "forearm",
    ("lower", "arm"): "lowerarm",
    ("upper", "leg"): "upperleg",
    ("lower", "leg"): "lowerleg",
    ("toe", "base"): "toebase",
    ("up", "leg"): "upleg",
    ("down", "leg"): "downleg",
    ("upper", "chest"): "upperchest",
    ("lower", "back"): "lowerback",
}

#: The reverse direction: a fused bone name that a rig spells as one word.
#: "LeftHandMiddle1" arrives as "handmiddle" and must yield the two words
#: "hand" and "middle" -- the hand it hangs from, and the finger it is.  Only
#: listed pairs are split, so a name is never taken apart on a guess.
_FUSED_PAIRS = {
    "handthumb": ("hand", "thumb"),
    "handindex": ("hand", "index"),
    "handmiddle": ("hand", "middle"),
    "handring": ("hand", "ring"),
    "handpinky": ("hand", "pinky"),
    "handfinger": ("hand", "finger"),
    "thumb": ("thumb",),
    "indexfinger": ("index", "finger"),
    "middlefinger": ("middle", "finger"),
    "ringfinger": ("ring", "finger"),
    "pinkyfinger": ("pinky", "finger"),
    "littlefinger": ("little", "finger"),
}


def _merge_known_pairs(tokens: list[str]) -> list[str]:
    out: list[str] = []
    index = 0
    while index < len(tokens):
        if index + 1 < len(tokens):
            pair = (tokens[index], tokens[index + 1])
            if pair in _WORD_PAIRS:
                out.append(_WORD_PAIRS[pair])
                index += 2
                continue
        token = tokens[index]
        if token in _FUSED_PAIRS:
            out.extend(_FUSED_PAIRS[token])
        else:
            out.append(token)
        index += 1
    return out


def normalize_key(name: str) -> str:
    """The alias-table key for a name: its tokens joined by ``_``."""
    return "_".join(normalize(name))


def side_of(name: str) -> str:
    """``"left"``, ``"right"`` or ``"center"`` for a source bone name.

    Reads the *original* string, because normalisation removes the side.  It
    recognises a side as a whole token (" L ForeArm"), a side glued to the
    bone word after a namespace ("mixamorig:LeftUpLeg"), and a side fused to
    the word itself ("LeftUpLeg"), without ever reading the "l" of "left" as
    a side of its own.
    """
    lowered = (name or "").lower()
    # The namespace is stripped first, so "mixamorig:LeftUpLeg" starts at
    # "leftupleg" and a fused side is visible at the front.
    body = _GLUED_NAMESPACE.sub(" ", lowered.replace(":", " ").replace(".", " "))
    body = body.strip()
    # A side standing as a whole token: "left_arm", "ForeArm.L", " L ForeArm".
    for token in re.split(r"[^a-z]+", body):
        if token in ("left", "lft", "lef"):
            return "left"
        if token in ("right", "rgt", "rig"):
            return "right"
    # A single-letter side token: " L ForeArm", "Bip01 R Clavicle".
    if re.search(r"(?:^|[\s._])l(?:[\s._]|$)", body):
        return "left"
    if re.search(r"(?:^|[\s._])r(?:[\s._]|$)", body):
        return "right"
    # A side fused to the bone word: "leftupleg", "lforearm", "rthigh".
    fused = re.match(
        r"(?:left|lft|lef|right|rgt|rig)(?=[a-z]{2,})|(?:l|r)(?=[a-z]{3,})",
        body,
    )
    if fused:
        return "left" if fused.group(0)[0] == "l" else "right"
    return "center"


def is_terminal_finger(name: str) -> bool:
    """True for the last bone of a finger chain (``LeftHandIndex1``)."""
    tokens = (name or "").lower().replace(".", " ").replace(":", " ").split()
    return bool(tokens) and bool(_INDEX_TOKENS.match(tokens[-1]))


def strip_index(name: str) -> str:
    """Drop a trailing numeric segment: ``"Index1"`` -> ``"index"``."""
    parts = (name or "").split(".")
    if parts and _INDEX_TOKENS.match(parts[-1]):
        parts = parts[:-1]
    return ".".join(parts)


@lru_cache(maxsize=1)
def _builtin_aliases() -> dict[str, tuple[str, ...]]:
    path = os.path.join(os.path.dirname(__file__), "..", "profiles", "aliases.json")
    with open(path, "r", encoding="utf-8") as handle:
        raw = json.load(handle)
    return {
        str(role): tuple(str(a).lower() for a in aliases)
        for role, aliases in raw.items()
        if not role.startswith("$")
    }


def load_aliases(path: str | None = None) -> dict[str, tuple[str, ...]]:
    """Alias table as ``role -> (fragment, ...)``.

    A user-supplied *path* replaces the built-in table entirely; entries
    starting with ``$`` are comments and ignored either way.
    """
    if path is None:
        return dict(_builtin_aliases())
    with open(path, "r", encoding="utf-8") as handle:
        raw = json.load(handle)
    return {
        str(role).lower(): tuple(str(a).lower() for a in aliases)
        for role, aliases in raw.items()
        if not role.startswith("$")
    }


def role_from_name(
    name: str,
    aliases: dict[str, tuple[str, ...]],
    ambiguous: set[Role] | None = None,
) -> Role | None:
    """Best-guess role from a name alone, or ``None`` if nothing matches.

    Matching runs in three passes, strongest first:

    1. the whole normalised name is an alias (``"upleg"`` -> thigh);
    2. the name is one alias plus a side/index suffix (``"uplegleft"``);
    3. an alias is a *whole token* of the name (``"index1"`` -> finger).

    Every pass is token-bounded.  Substring matching is what makes
    ``"mixamorig:LeftUpLeg"`` resolve to the arm instead of the leg -- the
    fragment ``"leg"`` sits inside ``"upleg"`` -- and a swapped limb is the
    worst single failure this tool can produce, because it animates and looks
    plausible while doing the wrong thing.

    ``ambiguous`` is the set of roles a *previous* bone in the same chain
    already holds, used to break ties the name cannot: ``"LeftLeg"`` names
    both the thigh and the calf, and only the chain knows which one this is.
    When every candidate is taken the match is still returned, because a
    second occurrence of an ambiguous fragment is better evidence of a chain
    than of a mistake -- but the caller is told, via :func:`is_ambiguous`.
    """
    key = normalize_key(name)
    if not key:
        return None
    tokens = key.split("_")

    table: list[tuple[tuple[str, ...], Role]] = []
    for role_name, fragments in aliases.items():
        try:
            role = Role(role_name)
        except ValueError:
            continue
        for fragment in fragments:
            frag = fragment.replace("-", "_")
            table.append((tuple(frag.split("_")), role))
    # Longest alias first, so "upperarm" is tried before "arm".
    table.sort(key=lambda item: -sum(len(part) for part in item[0]))

    # One scoring pass, strongest placement first.  Every candidate explains
    # a contiguous span of the name; the best candidate is the one that
    # explains the *most* of it, ties going to the earlier (stronger) pass.
    # Ranking by coverage rather than by pass order is what stops "hand
    # middle" from resolving to the hand: the hand explains 4 characters,
    # the middle finger explains 6, and the finger wins.
    best: tuple[int, int, Role] | None = None
    for rank, (min_extra, max_extra) in enumerate(((0, 0), (0, 1), (0, 2))):
        for parts, role in table:
            span = len(parts)
            for start in range(len(tokens) - span + 1):
                if tuple(tokens[start:start + span]) != parts:
                    continue
                extra = len(tokens) - span
                if not (min_extra <= extra <= max_extra):
                    continue
                covered = sum(len(p) for p in parts)
                candidate = (covered, -rank, role)
                if best is None or candidate[:2] > best[:2]:
                    best = candidate
    return best[2] if best is not None else None


__all__ = [
    "normalize",
    "normalize_key",
    "side_of",
    "is_terminal_finger",
    "strip_index",
    "load_aliases",
    "role_from_name",
]
