"""Source-to-target bone mapping."""

from .aliases import load_aliases, normalize, normalize_key, side_of
from .mapper import (
    BoneMatch,
    MappingResult,
    MatchKind,
    Side,
    map_rig,
    target_role_of,
)
from .roles import ARM_ORDER, LEG_ORDER, SPINE_ORDER, Role
from .settings import (
    DEFAULT_CONFIDENCE_FLOOR,
    REQUIRED_ROLES,
    MappingBlocked,
    MappingOverrides,
    MappingReport,
    MappingSettings,
    evaluate,
    require_exportable,
)

__all__ = [
    "Role",
    "Side",
    "MatchKind",
    "BoneMatch",
    "MappingResult",
    "map_rig",
    "target_role_of",
    "load_aliases",
    "normalize",
    "normalize_key",
    "side_of",
    "SPINE_ORDER",
    "ARM_ORDER",
    "LEG_ORDER",
    "MappingSettings",
    "MappingOverrides",
    "MappingReport",
    "MappingBlocked",
    "REQUIRED_ROLES",
    "DEFAULT_CONFIDENCE_FLOOR",
    "evaluate",
    "require_exportable",
]
