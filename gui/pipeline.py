"""Everything the GUI needs from the pipeline, as plain dictionaries.

The window never imports `gta_fbx_ifp_converter`.  It calls these functions
and gets dicts and lists of dicts back.  That keeps numpy and rwfury out of
the UI thread, keeps the widgets testable without a display, and means a
refactor inside the converter cannot silently break a layout file.

Anything the user must be able to edit or override is a plain value in these
dicts, never an object that has to be understood to be displayed.
"""
from __future__ import annotations

from typing import Any, Callable

Progress = Callable[[str, str, float], None]


# --------------------------------------------------------------------------- #
# loading
# --------------------------------------------------------------------------- #
def load_source(path: str, report: Progress) -> dict[str, Any]:
    """Parse an FBX into a description of its rig and clips."""
    report("Loading FBX", path, 0.05)
    from gta_fbx_ifp_converter.fbx import source_rig as sr

    rig = sr.load_source_rig(path)
    report("Loading FBX", f"{len(rig.bones)} bones", 0.6)
    return {
        "path": path,
        "name": path.rsplit("/", 1)[-1].rsplit("\\", 1)[-1],
        "bones": [
            {
                "index": index,
                "name": bone.name,
                "parent": (rig.bones[bone.parent].name
                           if bone.parent is not None and bone.parent >= 0 else ""),
                "depth": bone.depth,
                "animated": bone.has_animation,
            }
            for index, bone in enumerate(rig.bones)
        ],
        "clips": [
            {
                "name": clip.name,
                "frames": clip.frame_count,
                "duration_s": round(clip.duration, 3),
                "bones": len(clip.source_animated_bones),
            }
            for clip in rig.animated_clips
        ],
        "up_axis": rig.up_axis,
        "scale_to_metres": rig.scale_to_metres,
        "unit_scale_factor": round(rig.unit_scale_factor, 9),
    }


def load_target(path: str, report: Progress) -> dict[str, Any]:
    """Read a ped DFF and resolve every bone to a HAnim id."""
    report("Loading target DFF", path, 0.05)
    from gta_fbx_ifp_converter.gta import dff_reader as dr

    skeleton = dr.load_skeleton(path)
    report("Loading target DFF", f"{len(skeleton.bones)} bones", 0.6)
    if not skeleton.has_hanim:
        raise ValueError(
            "this DFF has no HAnim skeleton, so no bone has an animation id "
            "and nothing could be written to an IFP"
        )
    return {
        "path": path,
        "name": path.rsplit("/", 1)[-1].rsplit("\\", 1)[-1],
        "bones": [
            {
                # A bone with no canonical tag has no HAnim role and can
                # never be driven; showing it as "0" would look like a real
                # tag and invite a mapping to it.
                "tag": (int(bone.canonical_tag)
                        if bone.canonical_tag is not None else None),
                "name": bone.name,
                "parent": (int(skeleton.bones[bone.parent].canonical_tag)
                           if bone.parent is not None and bone.parent >= 0
                           and skeleton.bones[bone.parent].canonical_tag is not None
                           else None),
                "parent_name": skeleton.bones[bone.parent].name
                               if bone.parent is not None and bone.parent >= 0 else "",
                "hanim_id": bone.hanim_id,
                "bone_id": bone.bone_id,          # resolved IFP id
                "addressable": bone.is_addressable,
                "side": bone.side,
                "depth": bone.depth,
                # The ped's own rest position, read out of the 4x4 world
                # matrix in the DFF.  The preview needs it to draw the same
                # body for both sides of the comparison.
                "rest": [round(float(v), 6)
                         for v in bone.bind_world_gta[:3, 3]],
            }
            for bone in skeleton.bones
        ],
        "addressable": sum(1 for b in skeleton.bones if b.is_addressable),
        "has_skin": skeleton.has_skin,
    }


# --------------------------------------------------------------------------- #
# mapping
# --------------------------------------------------------------------------- #
def build_mapping(source: dict[str, Any], target: dict[str, Any],
                  report: Progress,
                  overrides: dict[str, int] | None = None) -> dict[str, Any]:
    """Map source bones to target HAnim ids and say how sure it is."""
    report("Mapping", "matching bones", 0.2)
    from gta_fbx_ifp_converter.fbx import source_rig as sr
    from gta_fbx_ifp_converter.gta import dff_reader as dr
    from gta_fbx_ifp_converter.mapping import map_rig, evaluate, require_exportable
    from gta_fbx_ifp_converter.mapping import MappingBlocked

    rig = sr.load_source_rig(source["path"])
    skeleton = dr.load_skeleton(target["path"])
    report("Mapping", "scoring", 0.6)
    mapping = map_rig(rig.bones, skeleton)
    if overrides:
        # Corrections the user typed into the table have to survive into the
        # converter, not just the display, or fixing a row changes nothing.
        apply_overrides(mapping, skeleton, overrides)
    result = evaluate(mapping, skeleton)
    report("Mapping", "checking confidence", 0.9)

    rows = []
    for match in result.result.matches:
        wanted = int(match.target_tag) if match.target_tag is not None else None
        target_bone = next(
            (b for b in skeleton.bones
             if b.canonical_tag is not None and int(b.canonical_tag) == wanted),
            None)
        rows.append({
            "source": match.source_name,
            "role": match.role.value if match.role else "",
            "target_tag": match.target_tag or "",
            "target_name": match.target_name or "",
            # bone_id is what the writer will actually put in the frame;
            # hanim_id is the raw HAnim node id and can legitimately be None
            # on a bone that still resolves through its tag.
            "hanim_id": target_bone.hanim_id if target_bone else None,
            "bone_id": target_bone.bone_id if target_bone else None,
            "addressable": bool(target_bone and target_bone.is_addressable),
            "confidence": round(match.confidence, 4),
            "how": match.kind.value if match.kind else "",
            "reason": match.reason,
            "is_mapped": match.is_mapped,
            "side": match.side.value if match.side else "",
            "source_index": match.source_index,
            "user_edited": match.user_edited,
        })

    try:
        require_exportable(result)
        exportable, blocked = True, []
    except MappingBlocked as exc:
        exportable, blocked = False, list(exc.reasons)

    return {
        "rows": rows,
        "mapped": len(result.result.mapped),
        "total": len(result.result.matches),
        "quality": round(result.result.quality_score(), 4),
        "mean_confidence": round(result.result.mean_confidence, 4),
        "warnings": list(result.warnings),
        "errors": list(result.errors),
        "exportable": exportable,
        "blocked_by": blocked,
        "unmapped": [m.source_name for m in result.result.unmapped],
        "conflicts": list(result.result.target_conflicts()),
    }


def apply_overrides(mapping, skeleton, overrides: dict[str, int]) -> None:
    """Force named source bones onto named target tags.

    An override is recorded as user-edited so the mapping report stops
    warning about the bone it displaced, and so the export page can show
    that the row was corrected by hand rather than guessed.  Rows that were
    never unmapped are left alone: overriding a row that was already right
    is how a good automatic mapping gets broken by a stray click.
    """
    for match in mapping.matches:
        tag = overrides.get(match.source_name)
        if tag is None or match.user_edited or match.is_mapped:
            continue
        target_bone = next(
            (b for b in skeleton.bones
             if b.canonical_tag is not None and int(b.canonical_tag) == tag), None)
        if target_bone is None:
            continue
        match.target_index = target_bone.index
        match.target_tag = tag
        match.target_name = target_bone.name
        match.is_mapped = True
        match.user_edited = True
        match.confidence = 1.0
        match.reason = "set by hand on the Skeleton Mapping page"


# --------------------------------------------------------------------------- #
# conversion
# --------------------------------------------------------------------------- #
def convert(source: dict[str, Any], target: dict[str, Any],
            mapping_rows: list[dict[str, Any]], settings: dict[str, Any],
            out_path: str, report: Progress) -> dict[str, Any]:
    """Run the whole pipeline and write a validated IFP.

    `report` is called at each stage so the window can show where the time
    went; a conversion that appears to hang is indistinguishable from one
    that has, and the stages here take very different amounts of time.
    """
    report("Retargeting", "solving the rest pose", 0.1)
    from gta_fbx_ifp_converter.fbx import source_rig as sr
    from gta_fbx_ifp_converter.gta import dff_reader as dr
    from gta_fbx_ifp_converter.mapping import map_rig
    from gta_fbx_ifp_converter.retarget import build_rest_pose_correction, retarget_clip
    from gta_fbx_ifp_converter.retarget.transfer import RetargetSettings, RootMode
    from gta_fbx_ifp_converter.gta.ifp_build import build_animation
    from gta_fbx_ifp_converter.gta.ifp_writer import write_ifp
    from gta_fbx_ifp_converter.gta.ifp_reader import read_ifp
    from gta_fbx_ifp_converter.validate import validate_round_trip, diagnose_all

    rig = sr.load_source_rig(source["path"])
    skeleton = dr.load_skeleton(target["path"])
    clip = _pick_clip(rig, settings["clip"])
    mapping = map_rig(rig.bones, skeleton)

    report("Retargeting", "solving local axes", 0.25)
    correction = build_rest_pose_correction(
        rig.bones,
        [b.parent for b in rig.bones],
        [b.bind_local_translation for b in rig.bones],
        [b.bind_local_quat for b in rig.bones],
        skeleton.bones,
        [b.parent for b in skeleton.bones],
        [b.bind_local_translation for b in skeleton.bones],
        [b.bind_local_quat for b in skeleton.bones],
        mapping.mapped,
    )

    root_mode = {"in_place": RootMode.IN_PLACE,
                 "preserve": RootMode.PRESERVE,
                 "horizontal": RootMode.HORIZONTAL,
                 "full": RootMode.FULL}[settings["root_mode"]]
    report("Retargeting", f"{clip.frame_count} frames", 0.45)
    retarget = retarget_clip(
        rig, clip, mapping, correction, skeleton.bones,
        RetargetSettings(root_mode=root_mode),
    )

    report("Writing IFP", settings["block_name"], 0.7)
    built = build_animation(settings["block_name"], retarget, skeleton,
                            settings["block_name"])
    if built.animation is None:
        raise ValueError("nothing could be written: " + "; ".join(built.warnings))
    written = write_ifp(out_path, [built.animation], settings["block_name"])

    report("Validating", "reading the file back", 0.9)
    parsed = read_ifp(out_path)
    validation = validate_round_trip(built.animation, parsed, settings["block_name"])
    diagnostics = diagnose_all(rig=rig, skeleton=skeleton, mapping=mapping,
                              parsed=parsed, correction=correction)

    report("Done", "", 1.0)
    return {
        "path": out_path,
        "bytes": written["bytes"],
        "objects": written["objects"],
        "frames": written["frames"],
        "warnings": list(built.warnings),
        "validation": validation.to_dict(),
        "validation_text": validation.describe(),
        "passed": validation.passed,
        "diagnostics": [f.to_dict() for f in diagnostics.findings],
        "clean": diagnostics.is_clean,
        "unit_scale": correction.unit_scale,
        "measured_fraction": correction.measured_fraction,
        "root_travel_s": retarget.root_travel_s,
        "unrepresentable": [
            {"bone": u.bone, "reason": u.reason} if hasattr(u, "bone")
            else {"bone": str(u), "reason": ""}
            for u in retarget.unrepresentable
        ],
        # Kept so the preview can play what was written, not a re-retarget
        # of the same inputs that might differ from the file on disk.
        "animation": built.animation,
        "parsed": parsed,
    }


def _pick_clip(rig, wanted: str):
    if not wanted:
        return rig.animated_clips[0]
    for clip in rig.animated_clips:
        if clip.name == wanted:
            return clip
    raise ValueError(f"no clip named {wanted!r}")


# --------------------------------------------------------------------------- #
# preview
# --------------------------------------------------------------------------- #
def sample_pose(animation, bone_id: int, key: int) -> dict[str, Any]:
    """One bone's stored key, for the side-by-side view."""
    track = next((t for t in animation.tracks if t.bone_id == bone_id), None)
    if track is None or not track.times_s:
        return {"bone_id": bone_id, "present": False}
    index = min(max(0, key), len(track.times_s) - 1)
    rot = track.rotations[index] if index < len(track.rotations) else (1, 0, 0, 0)
    pos = track.translations[index] if index < len(track.translations) else None
    return {
        "bone_id": bone_id,
        "present": True,
        "name": track.name,
        "index": index,
        "time_s": round(track.times_s[index], 4),
        "rotation": [float(v) for v in rot],
        "position": [float(v) for v in pos] if pos is not None else None,
        "keys": len(track.times_s),
    }
