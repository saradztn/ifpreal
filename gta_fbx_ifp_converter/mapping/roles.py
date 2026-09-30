"""Semantic bone roles.

A *role* is what a joint does in a human body -- "the knee", "the wrist".
It is deliberately separate from a GTA HAnim tag, because the tag that plays a
given role is a property of the **target DFF**, not of the role.  A ped without
fingers has no tag for ``finger``; a ped with two spines puts a different tag
on ``spine2`` than one with three.  Deciding that is the retargeter's job, and
this module only says what the source bones mean.

The role vocabulary is small and fixed so that a mapping report can be read
without a lookup table.
"""

from __future__ import annotations

from enum import Enum


class Role(str, Enum):
    """What a source bone is, anatomically."""

    ROOT = "root"
    PELVIS = "pelvis"
    SPINE1 = "spine1"
    SPINE2 = "spine2"
    SPINE3 = "spine3"
    NECK = "neck"
    HEAD = "head"
    JAW = "jaw"
    EYE = "eye"
    BROW = "brow"

    CLAVICLE = "clavicle"
    UPPER_ARM = "upper_arm"
    FOREARM = "forearm"
    HAND = "hand"
    FINGER = "finger"

    THIGH = "thigh"
    CALF = "calf"
    FOOT = "foot"
    TOE = "toe"

    BREAST = "breast"
    BELLY = "belly"

    UNKNOWN = "unknown"

    @property
    def is_arm(self) -> bool:
        return self in (
            Role.CLAVICLE, Role.UPPER_ARM, Role.FOREARM, Role.HAND, Role.FINGER,
        )

    @property
    def is_leg(self) -> bool:
        return self in (Role.THIGH, Role.CALF, Role.FOOT, Role.TOE)

    @property
    def is_spine(self) -> bool:
        return self in (
            Role.PELVIS, Role.SPINE1, Role.SPINE2, Role.SPINE3,
            Role.NECK, Role.HEAD,
        )


#: Roles in body order, used when a chain has to be reduced or split.
SPINE_ORDER: tuple[Role, ...] = (
    Role.PELVIS, Role.SPINE1, Role.SPINE2, Role.SPINE3, Role.NECK, Role.HEAD,
)

ARM_ORDER: tuple[Role, ...] = (
    Role.CLAVICLE, Role.UPPER_ARM, Role.FOREARM, Role.HAND, Role.FINGER,
)

LEG_ORDER: tuple[Role, ...] = (Role.THIGH, Role.CALF, Role.FOOT, Role.TOE)

#: GTA peds have a single finger bone, so only these two can ever reach one.
FINGER_ROLES: tuple[Role, ...] = (Role.FINGER,)

__all__ = [
    "Role",
    "SPINE_ORDER",
    "ARM_ORDER",
    "LEG_ORDER",
    "FINGER_ROLES",
]
