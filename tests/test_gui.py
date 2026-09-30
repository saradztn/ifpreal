"""Tests for the window, driven headlessly.

These build the real `MainWindow`, push real files through it, and assert on
what the user would see.  They are the only tests that would have caught a
table sorting 0.9 above 0.11, a page showing a stale path after a reload, or
the validation page claiming success on a file that failed.

They need a Qt platform plugin but not a display, and they skip cleanly when
PySide6 is not installed -- the converter itself has no GUI dependency, and
its test suite has to pass on a machine with no Qt at all.
"""
from __future__ import annotations

import os

import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

pytest.importorskip("PySide6", reason="the window needs PySide6")


@pytest.fixture(scope="module")
def app():
    from PySide6.QtWidgets import QApplication

    existing = QApplication.instance()
    yield existing or QApplication([])


@pytest.fixture
def window(app):
    from gui.app import MainWindow

    win = MainWindow()
    yield win
    win.close()


FBX = "testdata/samba_dancing.fbx"
DFF = "testdata/male01.dff"


def pump(app, times=50):
    """Let queued signals and repaints settle."""
    from PySide6.QtCore import QCoreApplication

    for _ in range(times):
        QCoreApplication.processEvents()


def run_sync(app, window, job, timeout_ms=120_000):
    """Run a job the way the window does, but block until it is done.

    The window's own jobs run on a QThread; waiting on the real signal keeps
    the threading under test, where a job helper that called the function
    directly would leave the most interesting code untested.
    """
    from PySide6.QtCore import QEventLoop, QTimer

    from gui.jobs import Job

    handle = Job(job)
    loop = QEventLoop()
    outcome = {}

    def done(result):
        outcome["result"] = result
        loop.quit()

    handle.finished.connect(done)
    handle.start()
    QTimer.singleShot(timeout_ms, loop.quit)
    loop.exec()
    handle.finished.disconnect(done)
    assert "result" in outcome, "the job never finished"
    result = outcome["result"]
    assert result.ok, f"the job failed: {result.error}\n{result.traceback}"
    return result.payload


class TestStructure:
    def test_the_six_pages_are_present_and_ordered(self, window):
        from gui.app import STAGES

        titles = [window.tabs.tabText(i) for i in range(window.tabs.count())]
        assert titles == STAGES
        assert titles == ["Input", "Skeleton Mapping", "Retarget Settings",
                          "Preview", "Export", "Validation"]

    def test_it_starts_with_nothing_loaded(self, window):
        window.refresh()
        assert window.source is None
        assert window.target is None
        assert window.conversion is None
        assert "no FBX chosen" in window.input_page.source_edit.placeholderText()

    def test_export_is_blocked_before_anything_is_loaded(self, window):
        window.refresh()
        assert not window.export_page.go.isEnabled()


class TestMappingTable:
    def test_rows_show_the_hanim_id_from_the_dff(self, app, window):
        from gui import pipeline

        payload = run_sync(app, window, lambda r: dict(
            pipeline.load_source(FBX, r), _kind="source"))
        window._handle(payload)
        window.refresh()
        target = run_sync(app, window, lambda r: dict(
            pipeline.load_target(DFF, r), _kind="target"))
        window._handle(target)
        mapping = run_sync(app, window, lambda r: dict(
            pipeline.build_mapping(window.source, window.target, r),
            _kind="mapping"))
        window._handle(mapping)
        window.refresh()

        assert window.mapping["mapped"] == 25
        assert window.mapping_page.table.rowCount() == window.mapping["total"]

        # Every row that names a target bone must name a bone id that really
        # exists in the DFF.  A role with no bone at all -- spine3 on this
        # ped -- carries no id and must not be given one, because an id
        # invented here is one the game would animate as a different bone.
        real = {b["bone_id"] for b in window.target["bones"]
                if b["addressable"]}
        for row in window.mapping["rows"]:
            if row["target_name"]:
                assert row["bone_id"] in real, \
                    f"{row['source']} claims id {row['bone_id']}, not in the DFF"
            else:
                assert row["bone_id"] is None, \
                    f"{row['source']} has no target bone but claims an id"
                assert row["hanim_id"] is None

    def test_sorting_confidence_uses_the_number_not_the_text(self, app, window):
        """A formatted 0.9 must sort above 0.11, not below it."""
        from gui.app import _KeyTable, _item

        table = _KeyTable([("x", 50, lambda r: _item(f"{r['v']:.2f}"))])
        rows = [{"v": 0.11}, {"v": 0.9}, {"v": 0.5}]
        table.load(rows, lambda r: r["v"], table_columns := [
            ("x", 50, lambda r: _item(f"{r['v']:.2f}"))])
        table.sortItems(0, QtAscending())
        assert [table.item(r, 0).text() for r in range(3)] == ["0.11", "0.50", "0.90"]


def QtAscending():
    from PySide6.QtCore import Qt

    return Qt.AscendingOrder


class TestSettings:
    def test_root_is_in_place_by_default(self, window):
        assert window.settings_page.values()["root_mode"] == "in_place", \
            "an IFP that walks the ped off on its own is the wrong default"

    def test_optimisation_is_off_by_default(self, window):
        values = window.settings_page.values()
        assert values["reduce"] is False
        assert not window.settings_page.threshold.isEnabled(), \
            "an enabled threshold invites reducing keys nobody asked to reduce"

    def test_every_root_mode_is_offered(self, window):
        modes = [window.settings_page.root_mode.itemData(i)
                 for i in range(window.settings_page.root_mode.count())]
        assert set(modes) == {"in_place", "horizontal", "full", "preserve"}


@pytest.mark.slow
class TestFullRun:
    def test_converting_through_the_window_produces_a_validated_file(
        self, app, window, tmp_path
    ):
        from gui import pipeline

        for job in (lambda r: dict(pipeline.load_source(FBX, r), _kind="source"),
                    lambda r: dict(pipeline.load_target(DFF, r), _kind="target")):
            window._handle(run_sync(app, window, job))
        window._handle(run_sync(app, window, lambda r: dict(
            pipeline.build_mapping(window.source, window.target, r),
            _kind="mapping")))

        out = tmp_path / "dance.ifp"
        settings = window.export_page.settings()
        settings.update(out_path=str(out), block_name="DANCE", clip="mixamo.com")
        window._handle(run_sync(app, window, lambda r: dict(
            pipeline.convert(window.source, window.target,
                             window.mapping["rows"], settings,
                             str(out), r), _kind="conversion")))
        window.refresh()

        assert out.exists()
        assert window.conversion["passed"] is True
        assert window.conversion["clean"] is True

        # The validation page must show the real numbers, not a green tick.
        window.tabs.setCurrentWidget(window.validation_page)
        window.refresh()
        headline = window.validation_page.headline.text()
        assert "PASSED VALIDATION" in headline
        assert "FAILED VALIDATION" not in headline
        assert window.validation_page.table.rowCount() == 25
        worst = window.conversion["validation"]["max_angle_deg"]
        assert 0.0 <= worst < 1.0, f"{worst} degrees is not a converted file"

    def test_changing_a_setting_invalidates_the_earlier_result(self, window):
        window.conversion = {"passed": True, "validation": {}, "diagnostics": []}
        window.settings_page._emit()
        assert window.conversion is None, \
            "a stale PASSED VALIDATION would be shown for a different run"
