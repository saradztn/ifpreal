#!/usr/bin/env python3
"""Command line front-end for the GTA FBX -> IFP retarget engine.

Windows usage::

    python cli.py inspect-fbx  C:\\path\\dance.fbx
    python cli.py inspect-dff  C:\\path\\male01.dff
    python cli.py check        C:\\path\\dance.fbx  C:\\path\\male01.dff
    python cli.py convert      --fbx dance.fbx --dff male01.dff --out dance.ifp
    python cli.py doctor

Every subcommand prints a real report built from the real file.  Nothing is
decorative and nothing is claimed that was not measured: ``convert`` refuses
to write a partial IFP and exits non-zero while any pipeline stage is missing.
"""

from __future__ import annotations

import argparse
import json
import os
import platform
import sys
import textwrap
import time
from typing import Any, Callable, Sequence

__version__ = "1.0.0"

# Exit codes, so a wrapper script or CI job can react without parsing text.
EXIT_OK = 0
EXIT_USAGE = 2
EXIT_INPUT = 3
EXIT_NOT_IMPLEMENTED = 4
EXIT_VALIDATION = 5


# --------------------------------------------------------------------------- #
# console
# --------------------------------------------------------------------------- #
class Console:
    """Tiny console helper: colours on a tty, plain text when piped."""

    def __init__(self, stream=None, quiet: bool = False, no_color: bool = False):
        self.stream = stream or sys.stdout
        self.quiet = quiet
        colour = (
            not no_color
            and not quiet
            and getattr(self.stream, "isatty", lambda: False)()
            and os.name == "nt"
        )
        if os.name == "nt" and colour:
            # Windows consoles need the VT sequence table switched on.
            try:
                import ctypes

                ctypes.windll.kernel32.SetConsoleMode(
                    ctypes.windll.kernel32.GetStdHandle(-11), 7
                )
            except Exception:
                colour = False
        self.colour = colour

    def _paint(self, text: str, code: str) -> str:
        return f"\033[{code}m{text}\033[0m" if self.colour else text

    def bold(self, text: str) -> str:
        return self._paint(text, "1")

    def red(self, text: str) -> str:
        return self._paint(text, "31")

    def green(self, text: str) -> str:
        return self._paint(text, "32")

    def yellow(self, text: str) -> str:
        return self._paint(text, "33")

    def cyan(self, text: str) -> str:
        return self._paint(text, "36")

    def write(self, text: str = "") -> None:
        if not self.quiet:
            print(text, file=self.stream)

    def always(self, text: str = "") -> None:
        print(text, file=self.stream)

    def error(self, text: str) -> None:
        print(self.red("ERROR: ") + text, file=sys.stderr)

    def warn(self, text: str) -> None:
        self.write(self.yellow("WARNING: ") + text)

    def heading(self, text: str) -> None:
        self.write()
        self.write(self.bold(text))
        self.write(self.bold("-" * len(text)))

    def kv(self, key: str, value: Any) -> None:
        self.write(f"  {key:<22} {value}")


# --------------------------------------------------------------------------- #
# reporting helpers
# --------------------------------------------------------------------------- #
def _fmt_vector(values: Sequence[float], digits: int = 4) -> str:
    return "(" + ", ".join(f"{float(v):+.{digits}f}" for v in values) + ")"


def _bytes_label(size: int) -> str:
    for unit in ("B", "KB", "MB", "GB"):
        if size < 1024 or unit == "GB":
            return f"{size:.0f} {unit}" if unit == "B" else f"{size:.1f} {unit}"
        size /= 1024.0
    return f"{size:.1f} GB"


def _file_info(path: str) -> dict[str, Any]:
    info: dict[str, Any] = {"path": os.path.abspath(path)}
    try:
        stat = os.stat(path)
    except OSError as exc:
        raise FileNotFoundError(f"{path}: {exc.strerror}") from exc
    info["size"] = stat.st_size
    info["size_label"] = _bytes_label(stat.st_size)
    with open(path, "rb") as handle:
        magic = handle.read(24)
    # Decoded, not raw bytes: this dict goes straight into the --json report
    # and `json.dump` cannot serialise a bytes object.
    info["magic"] = magic.decode("latin-1")
    info["format"] = "binary" if magic.startswith(b"Kaydara FBX Binary") else "ascii"
    return info


# --------------------------------------------------------------------------- #
# inspect-fbx
# --------------------------------------------------------------------------- #
def inspect_fbx(
    path: str,
    rig_index: int = 0,
    bone_limit: int = 0,
    as_json: bool = False,
    console: "Console | None" = None,
) -> dict[str, Any]:
    """Load an FBX and describe the source rig and its clips."""
    from gta_fbx_ifp_converter.fbx import FbxParseError
    from gta_fbx_ifp_converter.fbx import source_rig as sr
    from gta_fbx_ifp_converter.core import space

    info = _file_info(path)
    started = time.perf_counter()
    try:
        rig = sr.load_source_rig(path, rig_index=rig_index)
    except FbxParseError as exc:
        raise InputError(f"{path} is not a readable FBX: {exc}") from exc
    except FileNotFoundError as exc:
        raise InputError(str(exc)) from exc
    elapsed = time.perf_counter() - started

    document = rig.document
    report: dict[str, Any] = {
        "file": info,
        "fbx_version": document.version,
        "creator": document.creator,
        "resync_count": document.resync_count,
        "up_axis": rig.up_axis,
        "unit_scale_factor": rig.unit_scale_factor,
        "scale_to_metres": rig.scale_to_metres,
        "load_seconds": round(elapsed, 3),
        "bone_count": len(rig.bones),
        "armatures": [
            {
                "name": rig.bones[i].name,
                "index": i,
                "bones": len(rig.rig_bones(i)),
            }
            for i in rig.rigs
        ],
        "clips": [
            {
                "name": c.name,
                "fps": c.fps,
                "frames": c.frame_count,
                "start_ms": c.start_ms,
                "stop_ms": c.stop_ms,
                "duration_s": c.duration,
                "animated_bones": len(c.source_animated_bones),
                "non_linear_keys": c.non_linear_keys,
            }
            for c in rig.clips
        ],
        "bones": [],
    }

    conversion = space.detect_fbx_system(rig.up_axis, rig.unit_scale_factor)
    report["coordinate_system"] = {
        "declared_up": conversion.declared_up,
        "detected_up": conversion.detected_up,
        "detected_forward": conversion.detected_forward,
        "handedness": conversion.detected_handedness,
        "unit_in_metres": conversion.unit_in_metres,
        "warnings": list(conversion.warnings),
        "matrix": [[round(float(v), 6) for v in row] for row in conversion.conversion],
    }

    limit = bone_limit or len(rig.bones)
    for bone in rig.bones[:limit]:
        report["bones"].append(
            {
                "index": bone.index,
                "name": bone.name,
                "parent": bone.parent,
                "depth": bone.depth,
                "model_type": bone.model_type,
                "channels": sorted(bone.curves),
                "animated": bool(bone.curves),
                "bind_translation": [round(float(v), 5) for v in bone.bind_local_translation],
            }
        )

    if as_json:
        return report

    console = console or Console()
    console.always(console.bold("Source FBX"))
    console.kv("file", info["path"])
    console.kv("size", info["size_label"])
    console.kv("format", "binary" if document.is_binary else "ASCII")
    console.kv("FBX version", document.version)
    console.kv("up axis", rig.up_axis)
    console.kv(
        "unit scale",
        f"{rig.unit_scale_factor} -> {rig.scale_to_metres:g} m/unit",
    )
    console.kv("record resyncs", document.resync_count)
    console.kv("parse time", f"{elapsed:.3f} s")
    console.kv("bones", len(rig.bones))

    console.heading("Coordinate system")
    for warning in conversion.warnings:
        console.warn(warning)
    applied = conversion.conversion
    console.kv("up", f"FBX {conversion.detected_up} -> GTA Z")
    console.kv("forward", f"FBX {conversion.detected_forward} -> GTA -Y")
    # Show the actual image of each FBX axis, read out of the derived matrix,
    # instead of restating what the code was told to do.
    axes = {}
    for label, unit in (
        ("right", (1, 0, 0)),
        ("up", (0, 1, 0)),
        ("forward", (0, 0, -1)),
    ):
        image = applied @ __import__("numpy").array([*unit, 1.0])
        axes[label] = [round(float(v), 3) for v in image[:3]]
    console.kv(
        "FBX right  -> GTA", _fmt_vector(axes["right"], 3)
    )
    console.kv("FBX up     -> GTA", _fmt_vector(axes["up"], 3))
    console.kv("FBX forward-> GTA", _fmt_vector(axes["forward"], 3))

    console.heading("Armatures")
    if not rig.rigs:
        console.error("No animated Armature found.")
    for armature in report["armatures"]:
        marker = " *" if armature["index"] == rig_index else "  "
        console.write(
            f" {marker} [{armature['index']}] {armature['name']!r} "
            f"({armature['bones']} bones)"
        )

    console.heading("Clips")
    if not rig.clips:
        console.error("No animated Armature found.")
    for clip in report["clips"]:
        console.write(
            f"  {clip['name']!r}: {clip['frames']} keys, "
            f"{clip['duration_s']:.2f} s, {clip['animated_bones']} bones, "
            f"{clip['fps']:g} fps"
        )

    console.heading("Bones")
    shown = len(report["bones"])
    for entry in report["bones"]:
        depth = entry["depth"]
        pad = "  " + "  " * depth
        flag = console.green("*") if entry["animated"] else " "
        parent = "-" if entry["parent"] < 0 else str(entry["parent"])
        console.write(
            f"  {flag} {pad}{entry['name']!r} "
            f"{console.cyan('parent=' + parent)} "
            f"{_fmt_vector(entry['bind_translation'])}"
        )
    if shown < len(rig.bones):
        console.write(f"  ... {len(rig.bones) - shown} more (use --bones to show all)")
    return report


# --------------------------------------------------------------------------- #
# inspect-dff
# --------------------------------------------------------------------------- #
def inspect_dff(
    path: str,
    bone_limit: int = 0,
    as_json: bool = False,
    console: "Console | None" = None,
) -> dict[str, Any]:
    """Load a target DFF and describe the HAnim skeleton it offers."""
    from gta_fbx_ifp_converter.gta import DffHanimError, dff_reader as dr

    info = _file_info(path)
    started = time.perf_counter()
    try:
        skeleton = dr.load_skeleton(path)
    except DffHanimError as exc:
        raise InputError(str(exc)) from exc
    except FileNotFoundError as exc:
        raise InputError(str(exc)) from exc
    elapsed = time.perf_counter() - started

    resolution = skeleton.tag_resolution
    report: dict[str, Any] = {
        "file": info,
        "load_seconds": round(elapsed, 3),
        "frame_count": len(skeleton.bones),
        "has_hanim": skeleton.has_hanim,
        "has_skin": skeleton.has_skin,
        "model_names": list(skeleton.model_names),
        "addressable_tags": list(resolution.animated_tags) if resolution else [],
        "diagnostics": [
            {
                "severity": d.severity,
                "code": d.code,
                "frame": d.frame_index,
                "message": d.message,
            }
            for d in (resolution.diagnostics if resolution else [])
        ],
        "bones": [],
    }

    limit = bone_limit or len(skeleton.bones)
    for bone in skeleton.bones[:limit]:
        world = bone.bind_world_gta[:3, 3]
        report["bones"].append(
            {
                "index": bone.index,
                "name": bone.name,
                "parent": bone.parent,
                "bone_id": bone.bone_id,
                "tag": int(bone.canonical_tag) if bone.canonical_tag is not None else None,
                "hanim_node_id": bone.hanim_id,
                "side": bone.side,
                "skin_bone_index": bone.skin_bone_index,
                "bind_translation": [round(float(v), 5) for v in bone.bind_local_translation],
                "bind_world": [round(float(v), 4) for v in world],
            }
        )

    if as_json:
        return report

    console = console or Console()
    console.always(console.bold("Target DFF"))
    console.kv("file", info["path"])
    console.kv("size", info["size_label"])
    console.kv("frames", len(skeleton.bones))
    console.kv("HAnim plugin", "yes" if skeleton.has_hanim else console.red("no"))
    console.kv("skinned", "yes" if skeleton.has_skin else "no")
    console.kv("parse time", f"{elapsed:.3f} s")

    if resolution is not None:
        console.heading("HAnim tag resolution")
        console.kv("addressable bones", len(resolution.animated_tags))
        counts: dict[str, int] = {}
        for source in resolution.source_of_frame.values():
            counts[source] = counts.get(source, 0) + 1
        for source, count in sorted(counts.items()):
            console.kv(source, count)
        for diagnostic in resolution.diagnostics:
            text = f"[{diagnostic.code}] {diagnostic.message}"
            if diagnostic.severity == "error":
                console.error(text)
            elif diagnostic.severity == "warning":
                console.warn(text)
            else:
                console.write("  " + console.cyan("note: ") + text)

    console.heading("Frames")
    shown = len(report["bones"])
    for entry in report["bones"]:
        marker = "*" if entry["bone_id"] >= 0 else " "
        tag = "-" if entry["tag"] is None else str(entry["tag"])
        parent = "-" if entry["parent"] < 0 else str(entry["parent"])
        console.write(
            f"  {marker} {entry['index']:>3} {entry['name']!r:20s} "
            f"id={entry['bone_id']:>4} tag={tag:>4} {entry['side']:<6} "
            f"parent={parent:>3} bind={_fmt_vector(entry['bind_world'], 3)}"
        )
    if shown < len(skeleton.bones):
        console.write(f"  ... {len(skeleton.bones) - shown} more (use --bones to show all)")
    return report


# --------------------------------------------------------------------------- #
# check
# --------------------------------------------------------------------------- #
#: Pipeline stages, in order, with whether the code that implements them is in
#: the tree yet.  ``convert`` refuses to run while any of these is False.
PIPELINE_STAGES: list[tuple[str, bool]] = [
    ("FBX parse + source rig", True),
    ("FBX animation curves + clips", True),
    ("Target DFF + HAnim resolution", True),
    ("Bone mapping (semantic + geometry)", False),
    ("Rest-pose / local-axis solver", False),
    ("Retarget (limb direction preserving)", False),
    ("Keyframe bake with absolute timestamps", False),
    ("ANP3 IFP writer", False),
    ("Round-trip validator + metrics", False),
]


def check(
    fbx_path: str,
    dff_path: str,
    as_json: bool = False,
    console: "Console | None" = None,
) -> dict[str, Any]:
    """Load both sides and report whether a conversion can be trusted."""
    # Bone lists are needed in full here: the checks below count resolved
    # frames and unresolved ones, so truncating them would understate both.
    source = inspect_fbx(fbx_path, bone_limit=0, as_json=True)
    target = inspect_dff(dff_path, bone_limit=0, as_json=True)

    problems: list[dict[str, str]] = []
    if not source["armatures"] or not source["clips"]:
        problems.append({
            "code": "NO_ARMATURE",
            "severity": "error",
            "message": "No animated Armature found.",
        })
    if not target["has_hanim"]:
        problems.append({
            "code": "NO_HANIM",
            "severity": "error",
            "message": "Target DFF has no valid ped HAnim skeleton.",
        })
    if source["resync_count"]:
        problems.append({
            "code": "FBX_RESYNC",
            "severity": "error",
            "message": (
                f"the FBX had to resynchronise {source['resync_count']} "
                "time(s); records were skipped and the animation may be incomplete."
            ),
        })
    addressable = [b for b in target["bones"] if b["bone_id"] >= 0]
    if target["bones"] and len(addressable) < len(target["bones"]):
        problems.append({
            "code": "UNRESOLVED_BONES",
            "severity": "warning",
            "message": (
                f"{len(target['bones']) - len(addressable)} of "
                f"{len(target['bones'])} target frames have no valid HAnim id "
                "and cannot be written to an IFP."
            ),
        })
    missing = [name for name, done in PIPELINE_STAGES if not done]
    if missing:
        problems.append({
            "code": "PIPELINE_INCOMPLETE",
            "severity": "error",
            "message": (
                "the conversion pipeline is incomplete; these stages are not "
                "implemented yet: " + ", ".join(missing)
            ),
        })

    report = {
        "source": source,
        "target": target,
        "stages": [
            {"name": name, "implemented": done} for name, done in PIPELINE_STAGES
        ],
        "blockers": [p["message"] for p in problems],
        "problems": problems,
        "ready": not problems,
    }
    if as_json:
        return report

    console = console or Console()
    # What the two files actually are, before any verdict.
    console.heading("Source FBX")
    console.write(f"  file        {source['file']['path']}")
    console.write(f"  format      {source['file']['format']} FBX {source['fbx_version']}")
    console.write(
        f"  rigs        {len(source['armatures'])} armature(s), "
        f"{source['bone_count']} nodes"
    )
    for clip in source["clips"]:
        console.write(
            f"  clip        {clip['name']!r}: {clip['frames']} keys, "
            f"{clip['duration_s']:.3f}s @ {clip['fps']:g} fps, "
            f"{clip['animated_bones']} bones"
        )

    console.heading("Target DFF")
    console.write(f"  file        {target['file']['path']}")
    console.write(
        f"  frames      {target['frame_count']} "
        f"({len(addressable)} with a resolved HAnim id)"
    )
    console.write(
        f"  HAnim       {'present' if target['has_hanim'] else 'MISSING'}"
    )
    for note in target["diagnostics"][:3]:
        console.write(f"  note        {note['code']}: {note['message']}")
    if len(target["diagnostics"]) > 3:
        console.write(
            f"              ... and {len(target['diagnostics']) - 3} more "
            "(use inspect-dff)"
        )

    console.heading("Pipeline")
    for name, done in PIPELINE_STAGES:
        mark = console.green("[x]") if done else console.red("[ ]")
        console.write(f"  {mark} {name}")

    console.heading("Result")
    if problems:
        for problem in problems:
            if problem.get("severity") == "warning":
                console.warn(problem["message"])
            else:
                console.error(problem["message"])
        console.always("")
        console.always(console.bold("FAILED VALIDATION"))
        return report
    console.write(console.green("Ready to convert."))
    return report


# --------------------------------------------------------------------------- #
# convert
# --------------------------------------------------------------------------- #
def convert(args: argparse.Namespace, console: Console) -> int:
    """Run the conversion.  Refuses to emit a partial IFP."""
    check(args.fbx, args.dff, as_json=True)
    console.always("")
    console.error(
        "Convert is not implemented yet. No .ifp was written. "
        "Run 'gtafbx check' to see exactly which stages are missing."
    )
    return EXIT_NOT_IMPLEMENTED


# --------------------------------------------------------------------------- #
# doctor
# --------------------------------------------------------------------------- #
def doctor(as_json: bool = False, console: "Console | None" = None) -> dict[str, Any]:
    """Report the environment and the state of the pipeline."""
    report: dict[str, Any] = {
        "version": __version__,
        "python": sys.version.split()[0],
        "platform": f"{platform.system()} {platform.release()} ({platform.machine()})",
        "dependencies": {},
        "stages": [
            {"name": name, "implemented": done} for name, done in PIPELINE_STAGES
        ],
    }
    for module, label in (("numpy", "numpy"), ("rwfury", "rwfury")):
        try:
            imported = __import__(module)
            report["dependencies"][label] = getattr(imported, "__version__", "present")
        except Exception as exc:                      # pragma: no cover
            report["dependencies"][label] = f"MISSING ({exc})"
    try:
        import gta_fbx_ifp_converter

        report["package"] = gta_fbx_ifp_converter.__version__
    except Exception as exc:                          # pragma: no cover
        report["package"] = f"not importable ({exc})"

    if as_json:
        return report
    console = console or Console()
    console.always(console.bold("gta-fbx-ifp-converter " + __version__))
    console.kv("python", report["python"])
    console.kv("platform", report["platform"])
    console.kv("package", report["package"])
    for name, value in report["dependencies"].items():
        colour = console.red if str(value).startswith("MISSING") else console.green
        console.kv(name, colour(str(value)))
    console.heading("Pipeline")
    for name, done in PIPELINE_STAGES:
        mark = console.green("[x]") if done else console.red("[ ]")
        console.write(f"  {mark} {name}")
    return report


# --------------------------------------------------------------------------- #
# errors
# --------------------------------------------------------------------------- #
class InputError(Exception):
    """The user's file is missing or unusable."""


# --------------------------------------------------------------------------- #
# argument parsing
# --------------------------------------------------------------------------- #
def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="gtafbx",
        description="GTA San Andreas / MTA:SA FBX -> IFP retarget engine",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=textwrap.dedent(
            """\
            examples:
              gtafbx inspect-fbx dance.fbx
              gtafbx inspect-dff male01.dff
              gtafbx check dance.fbx male01.dff
              gtafbx convert --fbx dance.fbx --dff male01.dff --out dance.ifp
            """
        ),
    )
    parser.add_argument("--version", action="version", version=__version__)
    parser.add_argument("--quiet", action="store_true", help="suppress reports")
    parser.add_argument("--no-color", action="store_true", help="plain output")
    parser.add_argument("--json", action="store_true", dest="as_json",
                        help="machine readable output on stdout")
    subparsers = parser.add_subparsers(dest="command", metavar="<command>")

    fbx = subparsers.add_parser("inspect-fbx", help="describe an FBX source rig")
    fbx.add_argument("path", help="path to the .fbx file")
    fbx.add_argument("--rig", type=int, default=0, help="armature index to report")
    fbx.add_argument("--bones", type=int, default=0,
                     help="show at most N bones (0 = all)")

    dff = subparsers.add_parser("inspect-dff", help="describe a target ped DFF")
    dff.add_argument("path", help="path to the .dff file")
    dff.add_argument("--bones", type=int, default=0,
                     help="show at most N bones (0 = all)")

    both = subparsers.add_parser(
        "check", help="load both files and report whether a conversion can be trusted"
    )
    both.add_argument("fbx", help="path to the .fbx file")
    both.add_argument("dff", help="path to the target .dff file")

    conv = subparsers.add_parser("convert", help="retarget an FBX clip into an IFP")
    conv.add_argument("--fbx", required=True, help="path to the .fbx file")
    conv.add_argument("--dff", required=True, help="path to the target .dff file")
    conv.add_argument("--out", required=True, help="output .ifp path")
    conv.add_argument("--clip", default="", help="source clip name (default: first)")
    conv.add_argument("--block", default="", help="IFP animation/block name")
    conv.add_argument("--fps", type=float, default=0.0,
                      help="override FPS (0 = keep the source's own timing)")
    conv.add_argument("--in-place", dest="in_place", action="store_true", default=True,
                      help="remove root motion (default)")
    conv.add_argument("--root-motion", dest="in_place", action="store_false",
                      help="keep the source root motion")

    doctor = subparsers.add_parser("doctor", help="environment and pipeline status")

    # Accept the global switches on either side of the subcommand.  argparse
    # attaches them to the top-level parser only, so `gtafbx doctor --json` --
    # the way anyone actually types it -- would otherwise fail.
    for sub in (fbx, dff, both, conv, doctor):
        _add_global_switches(sub)
    return parser


_GLOBAL_SWITCHES = (
    ("--quiet", dict(action="store_true", help="suppress reports")),
    ("--no-color", dict(action="store_true", help="plain output")),
    ("--json", dict(action="store_true", dest="as_json",
                    help="machine readable output on stdout")),
)


def _add_global_switches(sub: argparse.ArgumentParser) -> None:
    for name, options in _GLOBAL_SWITCHES:
        if any(name in action.option_strings for action in sub._actions):
            continue
        # SUPPRESS, not False: a subparser default would overwrite whatever
        # the top-level parser already decided, so `gtafbx --json doctor`
        # would be undone by the subparser's own default.
        sub.add_argument(name, default=argparse.SUPPRESS, **options)


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    console = Console(quiet=args.quiet, no_color=args.no_color)

    if not args.command:
        parser.print_help()
        return EXIT_USAGE

    handlers: dict[str, Callable[[], Any]] = {
        "inspect-fbx": lambda: inspect_fbx(
            args.path, rig_index=args.rig, bone_limit=args.bones,
            as_json=args.as_json, console=console,
        ),
        "inspect-dff": lambda: inspect_dff(
            args.path, bone_limit=args.bones, as_json=args.as_json, console=console,
        ),
        "check": lambda: check(
            args.fbx, args.dff, as_json=args.as_json, console=console,
        ),
        "doctor": lambda: doctor(as_json=args.as_json, console=console),
    }

    if args.command == "convert":
        if args.as_json:
            console.always(json.dumps(
                {"command": "convert", "ready": False,
                 "error": "conversion pipeline not implemented"},
                indent=2,
            ))
        return convert(args, console)

    try:
        report = handlers[args.command]()
    except InputError as exc:
        if args.as_json:
            console.always(json.dumps(
                {"command": args.command, "error": str(exc)}, indent=2
            ))
        else:
            console.error(str(exc))
        return EXIT_INPUT
    except OSError as exc:
        # A missing or unreadable path.  Raised deep inside the readers, so it
        # is translated here rather than at every call site.
        message = f"{getattr(exc, 'filename', '') or ''}: {exc.strerror or exc}".strip(": ")
        return _fail(args, console, message or str(exc), EXIT_INPUT)
    except (ValueError, IndexError, KeyError, TypeError) as exc:
        # rwfury and the FBX parser signal a malformed file with plain
        # exceptions.  A user who points the tool at the wrong file should get
        # one clear line, not a traceback through the dependency.
        return _fail(
            args, console, f"{args.command}: cannot read the input ({exc})",
            EXIT_INPUT,
        )
    except KeyboardInterrupt:                        # pragma: no cover
        console.error("interrupted")
        return 130

    if args.as_json:
        console.always(json.dumps(report, indent=2, default=str))

    if args.command == "check" and not report.get("ready", True):
        return EXIT_VALIDATION
    return EXIT_OK


def _fail(
    args: argparse.Namespace,
    console: Console,
    message: str,
    code: int,
) -> int:
    """Report a failure in whichever form the caller asked for."""
    if getattr(args, "as_json", False):
        console.always(json.dumps(
            {"command": getattr(args, "command", None), "error": message}, indent=2
        ))
    else:
        console.error(message)
    return code


if __name__ == "__main__":
    sys.exit(main())