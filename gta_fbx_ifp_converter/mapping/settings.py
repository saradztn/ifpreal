"""Mapping policy: how confident a match must be, and what the user changed.

The mapper makes a proposal.  This module is where that proposal is judged and
where a human can overrule it, because no name-matching rule survives every
rig in the world and a tool that insists it is right is worse than one that
says "I am not sure about this bone, look at it".

Two things live here:

* :class:`MappingSettings` -- the confidence floor, the roles that must be
  covered before anything may be exported, and the deliberate choices (root
  mode, in-place, extra roles) that the retarget stage needs to know about.
* :class:`MappingOverrides` -- the user's edits, keyed by source bone name so
  they survive a re-run of the mapper.  A re-derivation that silently throws
  away the corrections a user spent ten minutes making is the single most
  annoying thing a tool like this can do.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable

from .roles import Role
from .mapper import BoneMatch, MappingResult, Side

#: Roles a retargeted ped animation cannot be missing.  Everything else is
#: optional: a ped with no fingers still walks, a ped with no right arm does
#: not.  These are the bones whose absence changes what the animation *is*.
REQUIRED_ROLES: tuple[tuple[Role, Side], ...] = (
    (Role.ROOT, Side.CENTER),
    (Role.SPINE1, Side.CENTER),
    (Role.HEAD, Side.CENTER),
    (Role.UPPER_ARM, Side.LEFT),
    (Role.FOREARM, Side.LEFT),
    (Role.UPPER_ARM, Side.RIGHT),
    (Role.FOREARM, Side.RIGHT),
    (Role.THIGH, Side.LEFT),
    (Role.CALF, Side.LEFT),
    (Role.FOOT, Side.LEFT),
    (Role.THIGH, Side.RIGHT),
    (Role.CALF, Side.RIGHT),
    (Role.FOOT, Side.RIGHT),
)

#: A match below this is a guess.  Alias matches on a well-known rig score
#: around 0.9, a geometry match lower; anything under the floor is surfaced to
#: the user rather than exported.
DEFAULT_CONFIDENCE_FLOOR = 0.70


class MappingBlocked(Exception):
    """Raised when a mapping is not good enough to export from.

    Carries the reasons, because "export refused" with no explanation is the
    most useless message a converter can print.
    """

    def __init__(self, reasons: Iterable[str]):
        self.reasons = list(reasons)
        super().__init__("; ".join(self.reasons))


@dataclass
class MappingSettings:
    """Everything the user can decide about the mapping itself."""

    #: A match scoring below this is treated as a guess and must be confirmed.
    confidence_floor: float = DEFAULT_CONFIDENCE_FLOOR
    #: Roles that must be mapped before an export is allowed.  Defaults to
    #: :data:`REQUIRED_ROLES`.
    required_roles: tuple[tuple[Role, Side], ...] = REQUIRED_ROLES
    #: Extra roles this particular job needs, on top of the required set.
    extra_roles: tuple[tuple[Role, Side], ...] = ()
    #: Source bones the user wants kept even if nothing maps them, by name.
    keep_source_bones: tuple[str, ...] = ()
    #: Source bones the user wants dropped, by name.  Checked *after* mapping,
    #: so it can veto a confident match as well as silence a noisy bone.
    ignore_source_bones: tuple[str, ...] = ()

    def to_dict(self) -> dict:
        return {
            "confidence_floor": self.confidence_floor,
            "required_roles": [[r.value, s.value] for r, s in self.required_roles],
            "extra_roles": [[r.value, s.value] for r, s in self.extra_roles],
            "keep_source_bones": list(self.keep_source_bones),
            "ignore_source_bones": list(self.ignore_source_bones),
        }

    @classmethod
    def from_dict(cls, data: dict) -> "MappingSettings":
        def roles(key: str) -> tuple[tuple[Role, Side], ...]:
            return tuple(
                (Role(r), Side(s)) for r, s in data.get(key, [])
            )

        return cls(
            confidence_floor=float(
                data.get("confidence_floor", DEFAULT_CONFIDENCE_FLOOR)
            ),
            required_roles=roles("required_roles") or REQUIRED_ROLES,
            extra_roles=roles("extra_roles"),
            keep_source_bones=tuple(data.get("keep_source_bones", ())),
            ignore_source_bones=tuple(data.get("ignore_source_bones", ())),
        )


@dataclass
class MappingOverrides:
    """The user's corrections to an auto-derived mapping.

    Keyed by *source bone name*, not by index: indices shift when a rig is
    re-exported, names are what the user saw on screen.  An override is also
    remembered with the reason it was made, so a later run can say "this bone
    was corrected by hand" instead of quietly re-deriving the wrong answer.
    """

    #: source bone name -> {"target": "<exact target bone name>", ...}
    assignments: dict[str, dict] = field(default_factory=dict)
    #: source bone names the user turned off.
    ignored: set[str] = field(default_factory=set)
    #: source bone names the user forced on, even below the confidence floor.
    confirmed: set[str] = field(default_factory=set)

    def is_empty(self) -> bool:
        return not (self.assignments or self.ignored or self.confirmed)

    def set_target(self, source_name: str, target_name: str) -> None:
        self.assignments[source_name] = {"target": target_name}

    def clear_target(self, source_name: str) -> None:
        self.assignments.pop(source_name, None)

    def to_dict(self) -> dict:
        return {
            "assignments": {
                name: dict(value) for name, value in self.assignments.items()
            },
            "ignored": sorted(self.ignored),
            "confirmed": sorted(self.confirmed),
        }

    @classmethod
    def from_dict(cls, data: dict) -> "MappingOverrides":
        return cls(
            assignments={
                name: dict(value)
                for name, value in data.get("assignments", {}).items()
            },
            ignored=set(data.get("ignored", ())),
            confirmed=set(data.get("confirmed", ())),
        )

    def save(self, path: str | Path) -> None:
        Path(path).write_text(
            json.dumps(self.to_dict(), indent=2, ensure_ascii=False),
            encoding="utf-8",
        )

    @classmethod
    def load(cls, path: str | Path) -> "MappingOverrides":
        text = Path(path).read_text(encoding="utf-8")
        return cls.from_dict(json.loads(text))


@dataclass
class MappingReport:
    """The outcome of judging a mapping, ready for a CLI line or a GUI panel."""

    result: MappingResult
    settings: MappingSettings
    overrides: MappingOverrides
    errors: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)

    @property
    def is_exportable(self) -> bool:
        return not self.errors

    @property
    def low_confidence(self) -> list[BoneMatch]:
        """Mapped bones the mapper is not sure about."""
        return [
            m for m in self.result.mapped
            if m.confidence < self.settings.confidence_floor
            and m.source_name not in self.overrides.confirmed
        ]

    def _apply_overrides(self, result: MappingResult, skeleton) -> None:
        """Fold the user's corrections into a mapping, in place.

        Applied to the live result so the retarget stage sees exactly what the
        user confirmed, and so the confidence numbers reported afterwards are
        the corrected ones rather than the mapper's first guess.
        """
        by_name = {b.name: b for b in skeleton.bones}
        for match in result.matches:
            if match.source_name in self.overrides.ignored:
                match.target_index = -1
                match.target_tag = None
                match.target_name = None
                match.confidence = 0.0
                match.reason = "ignored by user"
                match.user_edited = True
                continue
            choice = self.overrides.assignments.get(match.source_name)
            if not choice:
                continue
            wanted = choice.get("target")
            if not wanted:
                continue
            bone = by_name.get(wanted)
            if bone is None or not bone.is_addressable:
                match.reason = (
                    f"user override {wanted!r} is not an addressable bone "
                    f"of this ped"
                )
                match.user_edited = True
                continue
            match.target_index = bone.index
            match.target_tag = bone.resolved_tag
            match.target_name = bone.name
            # A hand-made choice is as certain as the person who made it.
            match.confidence = 1.0
            match.reason = f"user override -> {bone.name!r}"
            match.user_edited = True

    def _judge(self, result: MappingResult):
        errors: list[str] = []
        warnings: list[str] = list(result.warnings)

        required = self.settings.required_roles + self.settings.extra_roles
        for role, side in required:
            hits = [
                m for m in result.by_role(role, side)
                if m.is_mapped or m.source_name in self.settings.keep_source_bones
            ]
            if not hits:
                errors.append(
                    f"Incomplete mapping: no {side.value} {role.value} is mapped."
                )

        for match in self.low_confidence:
            warnings.append(
                f"low confidence ({match.confidence:.2f}) for "
                f"{match.source_name!r} -> [{match.target_tag}] "
                f"{match.target_name!r}; confirm it before exporting"
            )

        for other, match in result.target_conflicts():
            warnings.append(
                f"{other.source_name!r} and {match.source_name!r} both drive "
                f"[{match.target_tag}] {match.target_name!r}"
            )

        for match in result.unmapped:
            if match.source_name in self.overrides.ignored:
                continue
            if match.role in (None, Role.UNKNOWN):
                continue
            warnings.append(
                f"{match.source_name!r} ({match.role.value}) is unmapped: "
                f"{match.reason}"
            )

        if result.quality_score() < self.settings.confidence_floor:
            errors.append(
                f"mapping quality {result.quality_score():.3f} is below the "
                f"required {self.settings.confidence_floor:.2f}."
            )
        return errors, warnings


def evaluate(
    result: MappingResult,
    skeleton,
    settings: MappingSettings | None = None,
    overrides: MappingOverrides | None = None,
) -> MappingReport:
    """Judge a mapping and return a report; never raises.

    Use :func:`require_exportable` when the answer must be yes or no.
    """
    report = MappingReport(
        result=result,
        settings=settings or MappingSettings(),
        overrides=overrides or MappingOverrides(),
    )
    report._apply_overrides(result, skeleton)
    report.errors, report.warnings = report._judge(result)
    return report


def require_exportable(report: MappingReport) -> None:
    """Raise :class:`MappingBlocked` unless the mapping may be exported."""
    if report.errors:
        raise MappingBlocked(report.errors)


__all__ = [
    "MappingSettings",
    "MappingOverrides",
    "MappingReport",
    "MappingBlocked",
    "REQUIRED_ROLES",
    "DEFAULT_CONFIDENCE_FLOOR",
    "evaluate",
    "require_exportable",
]
