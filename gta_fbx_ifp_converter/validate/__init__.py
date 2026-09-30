"""Validation: re-reading what was written and measuring the difference."""

from .diagnostics import (
    Defect,
    DiagnosticReport,
    Finding,
    check_anp3,
    check_axis_convention,
    check_hanim_ids,
    check_mapping,
    check_mirrored_limbs,
    check_reversed_joints,
    check_side_swaps,
    check_source,
    check_target,
    check_upside_down,
    diagnose_all,
)
from .validator import ErrorStats, ValidationResult, validate_round_trip

__all__ = [
    "ErrorStats",
    "ValidationResult",
    "validate_round_trip",
    "Defect",
    "Finding",
    "DiagnosticReport",
    "check_source",
    "check_target",
    "check_mapping",
    "check_side_swaps",
    "check_mirrored_limbs",
    "check_upside_down",
    "check_reversed_joints",
    "check_axis_convention",
    "check_hanim_ids",
    "check_anp3",
    "diagnose_all",
]
