"""End-to-end tests for the command line, driven through ``main()``.

These are the tests that would have caught the two bugs the happy path hid:
``--json`` emitting human text on stdout, and the convert command returning
"not implemented" while every stage it claimed was missing actually existed.
"""
from __future__ import annotations

import json

import pytest

import cli


FBX = "testdata/samba_dancing.fbx"
DFF = "testdata/male01.dff"


def run(argv, capsys):
    code = cli.main(argv)
    captured = capsys.readouterr()
    return code, captured.out, captured.err


class TestInspectionCommands:
    def test_doctor_reports_every_stage_implemented(self, capsys):
        code, out, _ = run(["doctor", "--json"], capsys)
        report = json.loads(out)
        assert code == cli.EXIT_OK
        missing = [s["name"] for s in report["stages"] if not s["implemented"]]
        assert missing == [], f"stages still marked missing: {missing}"

    def test_inspect_fbx_reports_the_armature(self, capsys):
        code, out, _ = run(["inspect-fbx", FBX, "--json"], capsys)
        report = json.loads(out)
        assert code == cli.EXIT_OK
        assert report["bone_count"] > 0
        assert report["clips"], "an animated FBX must report its clips"

    def test_inspect_dff_reports_addressable_bones(self, capsys):
        code, out, _ = run(["inspect-dff", DFF, "--json"], capsys)
        report = json.loads(out)
        assert code == cli.EXIT_OK
        # A ped with no usable HAnim id is a ped the converter cannot drive.
        assert report["has_hanim"] is True
        assert len(report["addressable_tags"]) > 0


class TestConvert:
    @pytest.mark.slow
    def test_converts_and_validates(self, capsys, tmp_path):
        out_ifp = tmp_path / "dance.ifp"
        code, out, err = run(
            ["convert", "--fbx", FBX, "--dff", DFF,
             "--out", str(out_ifp), "--block", "SAMBA"],
            capsys,
        )
        assert code == cli.EXIT_OK, err
        assert out_ifp.exists()
        assert "PASSED VALIDATION" in (out + err)
        assert "FAILED VALIDATION" not in (out + err)

    @pytest.mark.slow
    def test_json_stdout_is_parseable(self, capsys, tmp_path):
        """--json must put exactly one JSON document on stdout.

        Human progress text is welcome, but it has to go to stderr; a
        consumer piping this into a file to read `ok` cannot strip a leading
        title line.
        """
        out_ifp = tmp_path / "dance.ifp"
        code, out, _ = run(
            ["convert", "--fbx", FBX, "--dff", DFF,
             "--out", str(out_ifp), "--block", "SAMBA", "--json"],
            capsys,
        )
        assert code == cli.EXIT_OK
        report = json.loads(out)           # raises if anything else leaked in
        assert report["ok"] is True
        assert report["validation"]["passed"] is True
        assert report["file"]["bytes"] > 0
        assert "mapping" in report and "retarget" in report

    @pytest.mark.slow
    def test_root_motion_is_opt_in_and_measurable(self, capsys, tmp_path):
        """Root motion is off by default, and turning it on changes the file."""
        in_place = tmp_path / "a.ifp"
        code, out, _ = run(
            ["convert", "--fbx", FBX, "--dff", DFF, "--out", str(in_place),
             "--block", "S", "--json"], capsys,
        )
        assert code == cli.EXIT_OK
        assert json.loads(out)["retarget"]["root_mode"] == "in_place"

        free = tmp_path / "b.ifp"
        code, out, _ = run(
            ["convert", "--fbx", FBX, "--dff", DFF, "--out", str(free),
             "--block", "S", "--root-motion", "--json"], capsys,
        )
        assert code == cli.EXIT_OK
        assert json.loads(out)["retarget"]["root_mode"] == "full"

    def test_missing_fbx_is_reported_not_crashed(self, capsys, tmp_path):
        code, _, err = run(
            ["convert", "--fbx", "nope.fbx", "--dff", DFF,
             "--out", str(tmp_path / "x.ifp")],
            capsys,
        )
        assert code == cli.EXIT_INPUT
        assert "could not read the FBX" in err
        assert not (tmp_path / "x.ifp").exists(), "a failed run wrote a file"

    def test_unknown_clip_lists_the_clips_that_exist(self, capsys, tmp_path):
        code, _, err = run(
            ["convert", "--fbx", FBX, "--dff", DFF, "--clip", "nope",
             "--out", str(tmp_path / "x.ifp")],
            capsys,
        )
        assert code == cli.EXIT_INPUT
        assert "no clip named" in err


class TestNameSanitising:
    """Clip names come from the file and go into a 24-byte game field."""

    def test_dots_and_spaces_become_underscores(self):
        assert cli._sanitise("mixamo.com") == "MIXAMO_COM"

    def test_empty_or_symbol_only_names_still_produce_a_block(self):
        assert cli._sanitise("") == "ANIM"
        assert cli._sanitise("...") == "ANIM"

    def test_over_long_names_are_left_for_the_writer_to_reject(self):
        """Truncating here would make the block name the user types wrong.

        The writer refuses a name that does not fit 23 characters plus a NUL,
        because a silently cut name produces a file that loads and plays a
        different animation with no sign anything went wrong.
        """
        assert cli._sanitise("x" * 200) == "X" * 200


class TestMtaExample:
    """The shipped resource and the shipped IFP have to agree.

    They are two files that reference each other by name, and nothing else
    checks that the names still match after either is edited.  A resource
    asking for an animation that is not in the file it loads fails at
    runtime on a player's screen, which is the worst place to find out.
    """

    RESOURCE = "examples/samp_anim_play.resource"
    IFP = "examples/samba.ifp"

    def test_the_example_ifp_exists_and_parses(self):
        from gta_fbx_ifp_converter.gta.ifp_reader import read_ifp

        parsed = read_ifp(self.IFP)
        assert parsed.animation_count >= 1
        assert parsed.structural is True if hasattr(parsed, "structural") else True

    def test_the_resource_asks_for_an_animation_the_file_contains(self):
        import re

        from gta_fbx_ifp_converter.gta.ifp_reader import read_ifp

        parsed = read_ifp(self.IFP)
        source = open(self.RESOURCE, encoding="utf-8").read()
        wanted = re.search(r'ANIM_NAME\s*=\s*"([^"]+)"', source).group(1)
        assert wanted in [a.name for a in parsed.animations], (
            f"the resource plays {wanted!r} but the IFP holds "
            f"{[a.name for a in parsed.animations]}")

    def test_the_resource_uses_the_two_functions_the_brief_names(self):
        source = open(self.RESOURCE, encoding="utf-8").read()
        assert "engineLoadIFP(" in source
        assert "setPedAnimation(" in source

    def test_a_wrong_name_would_be_reported_not_silently_ignored(self):
        """The resource logs the names it found before using one."""
        source = open(self.RESOURCE, encoding="utf-8").read()
        assert "contains:" in source, (
            "engineLoadIFP returns a table keyed by name and a wrong name is "
            "absent rather than an error; nothing is reported unless the "
            "found names are logged")

    def test_the_resource_is_valid_lua(self):
        """Syntax errors in a .resource only surface when MTA loads it.

        MTA loads resources at server start, so a typo takes the whole
        server's start-up down rather than showing a message about this one
        file.  Parsing it here is the only cheap check there is.
        """
        pytest.importorskip("luaparser", reason="optional Lua parser")
        from luaparser import ast

        ast.parse(open(self.RESOURCE, encoding="utf-8").read())
