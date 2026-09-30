"""The converter's window.

Six pages, in the order they are used: input, skeleton mapping, retarget
settings, preview, export, validation.  Every page is a `QWidget` that talks
to the window only through signals and attributes, so a page can be built and
inspected in a test with no display and no FBX on disk.

The window holds the state.  Pages read it; they do not cache their own copy
of a file path or a setting, because a cached copy is how "I changed the DFF
and it still used the old one" happens.
"""
from __future__ import annotations

import os
import sys

from PySide6.QtCore import Qt, Signal
from PySide6.QtGui import QColor, QFont
from PySide6.QtWidgets import (
    QAbstractItemView,
    QApplication,
    QCheckBox,
    QComboBox,
    QDoubleSpinBox,
    QFileDialog,
    QFormLayout,
    QGroupBox,
    QHBoxLayout,
    QHeaderView,
    QLabel,
    QLineEdit,
    QMainWindow,
    QMessageBox,
    QProgressBar,
    QPushButton,
    QSpinBox,
    QSplitter,
    QStatusBar,
    QTableWidget,
    QTableWidgetItem,
    QTabWidget,
    QTextEdit,
    QTreeWidget,
    QTreeWidgetItem,
    QVBoxLayout,
    QWidget,
)

from . import pipeline
from .jobs import Job, Progress, Result

APP_NAME = "GTA FBX to IFP Converter"

#: The stages, in the order the tool runs them.  The status bar shows the
#: current one, and so does the progress bar's label.
STAGES = [
    "Input", "Skeleton Mapping", "Retarget Settings",
    "Preview", "Export", "Validation",
]

PASS_GREEN = QColor("#1b7f3b")
FAIL_RED = QColor("#b3261e")
WARN_AMBER = QColor("#a86a00")
MUTED = QColor("#666666")


class Page(QWidget):
    """Base for the six pages: a title, a blurb, and a body to fill in."""

    def __init__(self, window: "MainWindow"):
        super().__init__()
        self.win = window
        outer = QVBoxLayout(self)
        outer.setContentsMargins(14, 12, 14, 12)
        heading = QLabel(self.title_text)
        heading.setStyleSheet("font-size: 15pt; font-weight: 600;")
        outer.addWidget(heading)
        note = QLabel(self.blurb)
        note.setWordWrap(True)
        note.setStyleSheet(f"color: {MUTED.name()};")
        outer.addWidget(note)
        self.body = QVBoxLayout()
        outer.addLayout(self.body)
        outer.addStretch(1)

    title_text = "Page"
    blurb = ""

    def refresh(self) -> None:
        """Called when the page is shown or when shared state changes."""


def _item(text: str, colour: QColor | None = None, tip: str = "") -> QTableWidgetItem:
    cell = QTableWidgetItem(text)
    if colour is not None:
        cell.setForeground(colour)
    if tip:
        cell.setToolTip(tip)
    cell.setFlags(cell.flags() & ~Qt.ItemIsEditable)
    return cell


class _KeyTable(QTableWidget):
    """A table that keeps its rows sorted by a hidden key.

    Re-sorting a QTableWidget by the text of a formatted float column puts
    0.9 before 0.11, which reads as a bug in the confidence column.  The
    underlying value is what gets sorted, and the text is only a display of
    it.
    """

    def __init__(self, columns):
        super().__init__(0, len(columns))
        self.setHorizontalHeaderLabels([c[0] for c in columns])
        self._keys = []
        for index, (_, width, _) in enumerate(columns):
            self.setColumnWidth(index, width)
        self.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeToContents)
        self.verticalHeader().setVisible(False)
        self.setSelectionBehavior(QAbstractItemView.SelectRows)
        self.setEditTriggers(QAbstractItemView.NoEditTriggers)
        self.setSortingEnabled(False)
        self.setAlternatingRowColors(True)

    def load(self, rows: list[dict], key, columns) -> None:
        self._keys = [key(r) for r in rows]
        self.setSortingEnabled(False)
        self.setRowCount(len(rows))
        for row, data in enumerate(rows):
            for column, (name, _, render) in enumerate(columns):
                self.setItem(row, column, render(data))
        self.setSortingEnabled(True)

    def sortItems(self, column: int, order=Qt.AscendingOrder) -> None:
        # Numeric sort on the underlying value, not on the rendered string.
        rows = range(self.rowCount())
        descending = order == Qt.DescendingOrder
        ordered = sorted(rows, key=lambda r: self._keys[r], reverse=descending)
        values = [self.takeItem(r, column) for r in range(self.rowCount())]
        for new_row, old_row in enumerate(ordered):
            self.setItem(new_row, column, values[old_row])
        super().sortItems(column, order)


# --------------------------------------------------------------------------- #
# 1. Input
# --------------------------------------------------------------------------- #
class InputPage(Page):
    choose_source = Signal()
    choose_target = Signal()

    title_text = "Input"
    blurb = ("The FBX is the animation to convert. The DFF is the ped that will "
             "play it, and it is the file that decides which bones exist and "
             "what each one's animation id is.")

    def __init__(self, window):
        super().__init__(window)

        self.source_edit = QLineEdit()
        self.source_edit.setReadOnly(True)
        self.source_edit.setPlaceholderText("no FBX chosen")
        button = QPushButton("Choose FBX...")
        button.clicked.connect(self.choose_source)
        row = QHBoxLayout()
        row.addWidget(self.source_edit, 1)
        row.addWidget(button)
        self.body.addLayout(row)

        self.target_edit = QLineEdit()
        self.target_edit.setReadOnly(True)
        self.target_edit.setPlaceholderText("no DFF chosen")
        button = QPushButton("Choose DFF...")
        button.clicked.connect(self.choose_target)
        row = QHBoxLayout()
        row.addWidget(self.target_edit, 1)
        row.addWidget(button)
        self.body.addLayout(row)

        self.summary = QTextEdit()
        self.summary.setReadOnly(True)
        self.summary.setMaximumHeight(190)
        self.body.addWidget(QLabel("What was read"))
        self.body.addWidget(self.summary)

    def refresh(self) -> None:
        lines = []
        source = self.win.source
        if source:
            animated = sum(1 for b in source["bones"] if b["animated"])
            lines.append(
                f"<b>{source['name']}</b><br>"
                f"{len(source['bones'])} bones, {animated} animated, "
                f"up axis {source['up_axis']}, "
                f"1 unit = {source['scale_to_metres']} m<br>"
                f"{len(source['clips'])} clip(s): "
                + ", ".join(f"{c['name']} ({c['frames']} keys, "
                            f"{c['duration_s']:.2f}s)" for c in source["clips"])
            )
        target = self.win.target
        if target:
            untagged = sum(1 for b in target["bones"] if b["tag"] is None)
            lines.append(
                f"<b>{target['name']}</b><br>"
                f"{len(target['bones'])} bones, {target['addressable']} with an "
                f"animation id"
                + (f", {untagged} with no HAnim tag" if untagged else "")
                + (", has a skin" if target["has_skin"] else "")
            )
        self.summary.setHtml("<hr>".join(lines) if lines else
                             "<i>nothing loaded yet</i>")
        self.source_edit.setText(source["path"] if source else "")
        self.target_edit.setText(target["path"] if target else "")


# --------------------------------------------------------------------------- #
# 2. Skeleton Mapping
# --------------------------------------------------------------------------- #
class MappingPage(Page):
    remap = Signal(str, int)          # source bone name, target tag
    rebuild = Signal()

    title_text = "Skeleton Mapping"
    blurb = ("Every target id comes from the DFF itself, resolved through its "
             "HAnim skeleton. Nothing is inferred from a name. Fix anything "
             "here that is wrong -- the rest of the run is driven by this table.")

    COLUMNS = [
        ("Source bone", 200, lambda r: _item(r["source"])),
        ("Role", 90, lambda r: _item(
            r["role"],
            None if r["is_mapped"] else MUTED)),
        ("Target bone", 150, lambda r: _item(
            r["target_name"] or "-- unmapped --",
            None if r["is_mapped"] else FAIL_RED)),
        ("Tag", 50, lambda r: _item(str(r["target_tag"] or ""))),
        # The id the writer puts in the frame.  It comes from the DFF's tag
        # resolution, which is the only source allowed to produce one -- the
        # raw HAnim node id is shown as a tooltip because it can be absent on
        # a bone that still resolves correctly through its tag.
        ("Anim id", 60, lambda r: _item(
            str(r["bone_id"]) if r["bone_id"] is not None else "--",
            None if r["addressable"] else WARN_AMBER,
            f"HAnim node id: "
            f"{r['hanim_id'] if r['hanim_id'] is not None else 'absent'}")),
        ("Conf", 55, lambda r: _item(
            f"{r['confidence']:.2f}",
            FAIL_RED if r["confidence"] < 0.5 else
            (WARN_AMBER if r["confidence"] < 0.8 else PASS_GREEN),
            r.get("reason", ""))),
        ("Matched by", 90, lambda r: _item(r["how"] or "")),
    ]

    def __init__(self, window):
        super().__init__(window)
        self.table = _KeyTable(self.COLUMNS)
        self.table.itemSelectionChanged.connect(self._selection_changed)
        self.body.addWidget(self.table, 1)

        form = QFormLayout()
        self.target_choice = QComboBox()
        self.target_choice.currentIndexChanged.connect(self._apply)
        apply = QPushButton("Map to selected")
        apply.clicked.connect(self._apply)
        row = QHBoxLayout()
        row.addWidget(QLabel("Map selected source bone to"))
        row.addWidget(self.target_choice, 1)
        row.addWidget(apply)
        self.body.addLayout(row)
        form = QFormLayout()
        self.notes = QTextEdit()
        self.notes.setReadOnly(True)
        self.notes.setMaximumHeight(110)
        self.body.addWidget(QLabel("Mapping notes"))
        self.body.addWidget(self.notes)

    def refresh(self) -> None:
        mapping = self.win.mapping
        self.target_choice.blockSignals(True)
        self.target_choice.clear()
        if self.win.target:
            for bone in self.win.target["bones"]:
                if bone["tag"] is None:
                    continue
                label = (f"{bone['name']} (tag {bone['tag']}, "
                         f"id {bone['bone_id']})")
                self.target_choice.addItem(label, bone["tag"])
        self.target_choice.blockSignals(False)
        if not mapping:
            self.table.setRowCount(0)
            self.notes.setHtml("<i>load an FBX and a DFF to build the mapping</i>")
            return
        self.table.load(mapping["rows"],
                        lambda r: (not r["is_mapped"], -r["confidence"],
                                   r["source"]),
                        self.COLUMNS)
        lines = [
            f"<b>{mapping['mapped']} of {mapping['total']} source bones "
            f"mapped</b>, quality {mapping['quality']:.2f}, "
            f"mean confidence {mapping['mean_confidence']:.2f}."
        ]
        if not mapping["exportable"]:
            lines.append(f"<span style='color:{FAIL_RED.name()}'>"
                         f"<b>Incomplete mapping.</b></span>")
            lines += [f"&#8226; {r}" for r in mapping["blocked_by"][:6]]
        elif mapping["warnings"]:
            lines.append(f"{len(mapping['warnings'])} warning(s); "
                         f"the first few:")
            lines += [f"&#8226; {w}" for w in mapping["warnings"][:6]]
        if mapping["unmapped"]:
            lines.append(
                f"{len(mapping['unmapped'])} bones have no IFP equivalent and "
                f"will not appear in the output: "
                f"{', '.join(mapping['unmapped'][:6])}"
                f"{' ...' if len(mapping['unmapped']) > 6 else ''}"
            )
        self.notes.setHtml("<br>".join(lines))

    def _selection_changed(self) -> None:
        row = self.table.currentRow()
        if row >= 0 and self.target_choice.count():
            self.target_choice.setCurrentIndex(0)

    def _apply(self) -> None:
        row = self.table.currentRow()
        if row < 0:
            QMessageBox.information(
                self, "Select a bone first",
                "Click a source bone in the table, then choose the target bone "
                "it should drive.")
            return
        tag = self.target_choice.currentData()
        if tag is None:
            return
        source = self.table.item(row, 0).text()
        self.remap.emit(source, tag)


# --------------------------------------------------------------------------- #
# 3. Retarget Settings
# --------------------------------------------------------------------------- #
class SettingsPage(Page):
    changed = Signal()

    title_text = "Retarget Settings"
    blurb = ("How the source's motion is transferred. The defaults keep every "
             "source key and its timing; the two options that change that say "
             "so and are off until asked for.")

    def __init__(self, window):
        super().__init__(window)

        group = QGroupBox("Root")
        form = QFormLayout(group)
        self.root_mode = QComboBox()
        # Default is in-place: an IFP animation is usually wanted to play
        # where the ped is standing, and shipping a root that walks off on
        # its own is the most common way to get an unusable animation.
        self.root_mode.addItem("In place -- the ped does not move", "in_place")
        self.root_mode.addItem("Keep horizontal travel only", "horizontal")
        self.root_mode.addItem("Keep the full root track", "full")
        self.root_mode.addItem("Preserve the source root exactly", "preserve")
        self.root_mode.currentIndexChanged.connect(self._emit)
        form.addRow("Root motion", self.root_mode)
        self.body.addWidget(group)

        group = QGroupBox("Timing")
        form = QFormLayout(group)
        self.fps = QDoubleSpinBox()
        self.fps.setRange(0.0, 240.0)
        self.fps.setDecimals(3)
        self.fps.setValue(0.0)
        self.fps.setSpecialValueText("from the FBX")
        self.fps.valueChanged.connect(self._emit)
        form.addRow("Frame rate override", self.fps)
        self.timing_note = QLabel()
        self.timing_note.setWordWrap(True)
        self.timing_note.setStyleSheet(f"color: {MUTED.name()};")
        form.addRow("", self.timing_note)
        self.body.addWidget(group)

        group = QGroupBox("Optimisation")
        self.reduce = QCheckBox(
            "Drop keys that barely change the pose (opt-in)")
        self.reduce.toggled.connect(self._emit)
        self.threshold = QDoubleSpinBox()
        self.threshold.setRange(0.0, 10.0)
        self.threshold.setDecimals(3)
        self.threshold.setValue(0.25)
        self.threshold.setSuffix(" degrees")
        self.threshold.valueChanged.connect(self._emit)
        self.threshold.setEnabled(False)
        self.reduce.toggled.connect(self.threshold.setEnabled)
        form2 = QFormLayout(group)
        form2.addRow(self.reduce)
        form2.addRow("Ignore a change below", self.threshold)
        note = QLabel(
            "Off by default. IFP stores time in 1/50 s steps, so a 60 fps "
            "source cannot keep every key regardless -- the keys that share a "
            "slot are merged and the export says how many. This option is "
            "about dropping keys the format could have kept.")
        note.setWordWrap(True)
        note.setStyleSheet(f"color: {MUTED.name()};")
        form2.addRow(note)
        self.body.addWidget(group)

    def _emit(self) -> None:
        self.changed.emit()

    def values(self) -> dict:
        return {
            "root_mode": self.root_mode.currentData(),
            "fps": self.fps.value(),
            "reduce": self.reduce.isChecked(),
            "threshold": self.threshold.value(),
        }

    def refresh(self) -> None:
        source = self.win.source
        if source and source["clips"]:
            clip = source["clips"][0]
            self.timing_note.setText(
                f"the source is {clip['frames']} keys over {clip['duration_s']:.2f} s. "
                f"IFP stores time in 1/50 s steps, so a key is kept exactly "
                f"when it lands on one of those steps and merged into its "
                f"neighbour when it does not."
            )


# --------------------------------------------------------------------------- #
# 4. Preview
# --------------------------------------------------------------------------- #
class PreviewPage(Page):
    """Source and output side by side, scrubbed on one clock.

    The two views are the same skeleton drawn from the same camera.  Seeing
    them side by side is the only way to notice that a whole arm is
    rotating about the wrong axis, and it is much faster to see than to
    compute.
    """
    title_text = "Preview"
    blurb = ("The source pose and the converted pose at the same moment. Both "
             "are drawn with the same camera and scale so they can be compared "
             "by eye before anything is written.")

    def __init__(self, window):
        super().__init__(window)
        from .preview import SkeletonView

        self.source_view = SkeletonView("FBX (source)")
        self.target_view = SkeletonView("IFP (converted)")
        splitter = QSplitter(Qt.Horizontal)
        splitter.addWidget(self.source_view)
        splitter.addWidget(self.target_view)
        self.body.addWidget(splitter, 1)

        controls = QHBoxLayout()
        self.play_button = QPushButton("Play")
        self.play_button.clicked.connect(self.toggle)
        controls.addWidget(self.play_button)
        controls.addWidget(QLabel("Time"))
        self.scrub = QDoubleSpinBox()
        self.scrub.setRange(0.0, 100.0)
        self.scrub.setDecimals(3)
        self.scrub.valueChanged.connect(self._scrubbed)
        controls.addWidget(self.scrub, 1)
        controls.addWidget(QLabel("Key"))
        self.key_label = QLabel("0")
        controls.addWidget(self.key_label)
        self.body.addLayout(controls)

        self.timing = QLabel()
        self.timing.setWordWrap(True)
        self.timing.setStyleSheet(f"color: {MUTED.name()};")
        self.body.addWidget(self.timing)

    def toggle(self) -> None:
        running = self.source_view.toggle_play()
        self.target_view.set_playing(running)
        self.play_button.setText("Pause" if running else "Play")

    def _scrubbed(self, seconds: float) -> None:
        self.source_view.set_time(seconds)
        self.target_view.set_time(seconds)
        self.key_label.setText(str(self.source_view.key_index()))

    def refresh(self) -> None:
        if self.win.conversion is None:
            self.timing.setText(
                "Convert on the Export page to preview the result.")
            return
        self.timing.setText(self.win.conversion.get("timing_note", ""))


# --------------------------------------------------------------------------- #
# 5. Export
# --------------------------------------------------------------------------- #
class ExportPage(Page):
    run = Signal()
    open_folder = Signal()

    title_text = "Export"
    blurb = ("Runs the conversion, writes the IFP, then reads the file back "
             "off disk and measures it. Nothing here claims success without a "
             "number to back it up.")

    def __init__(self, window):
        super().__init__(window)
        row = QHBoxLayout()
        self.path_edit = QLineEdit()
        row.addWidget(QLabel("Output"))
        row.addWidget(self.path_edit, 1)
        browse = QPushButton("Browse...")
        browse.clicked.connect(self._browse)
        row.addWidget(browse)
        self.body.addLayout(row)

        self.block_edit = QLineEdit()
        self.block_edit.setPlaceholderText("animation block name, e.g. DANCE")
        self.body.addLayout(_labelled("Block name", self.block_edit))

        self.clip_box = QComboBox()
        self.body.addLayout(_labelled("Source clip", self.clip_box))

        self.go = QPushButton("Convert and validate")
        self.go.setDefault(True)
        self.go.clicked.connect(self.run)
        self.body.addWidget(self.go)

        self.status = QLabel()
        self.status.setWordWrap(True)
        self.body.addWidget(self.status)

    def _browse(self) -> None:
        start = os.path.dirname(self.path_edit.text()) or os.getcwd()
        path, _ = QFileDialog.getSaveFileName(
            self, "Write the IFP to", start, "IFP files (*.ifp)")
        if path:
            self.path_edit.setText(path)

    def refresh(self) -> None:
        if not self.path_edit.text() and self.win.source:
            base = os.path.splitext(os.path.basename(self.win.source["path"]))[0]
            block = base.upper().replace(" ", "_")[:23] or "ANIM"
            self.path_edit.setText(os.path.join(os.getcwd(), block + ".ifp"))
            self.block_edit.setText(block)
        self.clip_box.clear()
        for clip in (self.win.source or {}).get("clips", []):
            self.clip_box.addItem(f"{clip['name']}  "
                                  f"({clip['frames']} keys, {clip['duration_s']:.2f}s)")
        ready = bool(self.win.source and self.win.target
                     and (self.win.mapping or {}).get("exportable"))
        self.go.setEnabled(ready)
        if not self.win.source or not self.win.target:
            self.status.setText("Load an FBX and a DFF on the Input page first.")
        elif self.win.mapping and not self.win.mapping["exportable"]:
            self.status.setText(
                f"<span style='color:{FAIL_RED.name()}'>Incomplete mapping.</span> "
                f"Fix the mapping on the Skeleton Mapping page before exporting.")
        elif not self.win.mapping:
            self.status.setText("Build the mapping on the Skeleton Mapping page.")
        else:
            self.status.setText("Ready. The file is validated after it is written.")

    def settings(self) -> dict:
        values = dict(self.win.settings_page.values())
        values["clip"] = self.clip_box.currentText().split("  ")[0] \
            if self.clip_box.currentText() else ""
        values["block_name"] = self.block_edit.text().strip() or "ANIM"
        values["out_path"] = self.path_edit.text().strip()
        return values


def _labelled(text: str, widget: QWidget) -> QHBoxLayout:
    row = QHBoxLayout()
    row.addWidget(QLabel(text))
    row.addWidget(widget, 1)
    return row


# --------------------------------------------------------------------------- #
# 6. Validation
# --------------------------------------------------------------------------- #
class ValidationPage(Page):
    title_text = "Validation"
    blurb = ("What the file on disk actually contains, measured by reading it "
             "back. The errors below are the difference between the intended "
             "animation and the one that was written.")

    COLUMNS = [
        ("Bone", 150, lambda r: _item(r["bone_name"] or str(r["bone_id"]))),
        ("id", 40, lambda r: _item(str(r["bone_id"]))),
        ("Keys", 55, lambda r: _item(str(r["samples"]))),
        ("Max angle", 85, lambda r: _item(
            f"{r['max']:.4f}\u00b0", _grade(r["max"], 0.05, 0.5))),
        ("Mean", 75, lambda r: _item(f"{r['mean']:.4f}\u00b0")),
        ("RMS", 75, lambda r: _item(f"{r['rms']:.4f}\u00b0")),
        ("Worst key", 70, lambda r: _item(str(r["worst_frame"]))),
    ]

    def __init__(self, window):
        super().__init__(window)
        self.headline = QLabel()
        self.headline.setWordWrap(True)
        self.headline.setStyleSheet("font-size: 13pt; font-weight: 600;")
        self.body.addWidget(self.headline)

        self.table = _KeyTable(self.COLUMNS)
        self.body.addWidget(self.table, 1)

        self.findings = QTableWidget(0, 3)
        self.findings.setHorizontalHeaderLabels(["", "Defect", "Detail"])
        self.findings.horizontalHeader().setSectionResizeMode(1, QHeaderView.ResizeToContents)
        self.findings.horizontalHeader().setSectionResizeMode(2, QHeaderView.Stretch)
        self.findings.verticalHeader().setVisible(False)
        self.findings.setEditTriggers(QAbstractItemView.NoEditTriggers)
        self.body.addWidget(QLabel("Checks"))
        self.body.addWidget(self.findings, 1)

    def refresh(self) -> None:
        conversion = self.win.conversion
        if not conversion:
            self.headline.setText("Nothing has been converted yet.")
            self.headline.setStyleSheet("")
            self.table.setRowCount(0)
            self.findings.setRowCount(0)
            return
        validation = conversion["validation"]
        colour = PASS_GREEN if conversion["passed"] else FAIL_RED
        word = "PASSED VALIDATION" if conversion["passed"] else "FAILED VALIDATION"
        self.headline.setText(
            f"<span style='color:{colour.name()}'>{word}</span> &mdash; "
            f"{conversion['objects']} bones, {conversion['frames']} keys, "
            f"{conversion['bytes']} bytes")
        self.headline.setStyleSheet("font-size: 13pt; font-weight: 600;")

        # `rotation` and `position` are separate lists because only the root
        # carries a translation; joining them here would show a row per bone
        # twice, and positionless bones would claim a 0 mm error they never
        # measured.
        rows = []
        for entry in validation.get("rotation", []):
            rows.append(dict(entry))
        self.table.load(rows, lambda r: r["max"], self.COLUMNS)

        self.findings.setRowCount(len(conversion["diagnostics"]))
        for row, finding in enumerate(conversion["diagnostics"]):
            fatal = finding.get("fatal", True)
            self.findings.setItem(row, 0, _item(
                "FAIL" if fatal else "warn",
                FAIL_RED if fatal else WARN_AMBER))
            self.findings.setItem(row, 1, _item(finding.get("defect", "")))
            detail = finding.get("message", "")
            if finding.get("where"):
                detail += f"  ({finding['where']})"
            if finding.get("measured") is not None:
                detail += f"  measured {finding['measured']:.4g}"
            self.findings.setItem(row, 2, _item(detail))


def _grade(value: float, good: float, bad: float) -> QColor:
    if value <= good:
        return PASS_GREEN
    if value <= bad:
        return WARN_AMBER
    return FAIL_RED


# --------------------------------------------------------------------------- #
# the window
# --------------------------------------------------------------------------- #
class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle(APP_NAME)
        self.resize(1180, 780)

        self.source: dict | None = None
        self.target: dict | None = None
        self.mapping: dict | None = None
        self.conversion: dict | None = None
        self._job: Job | None = None

        self.tabs = QTabWidget()
        self.input_page = InputPage(self)
        self.mapping_page = MappingPage(self)
        self.settings_page = SettingsPage(self)
        self.preview_page = PreviewPage(self)
        self.export_page = ExportPage(self)
        self.validation_page = ValidationPage(self)
        for page, name in zip(
            (self.input_page, self.mapping_page, self.settings_page,
             self.preview_page, self.export_page, self.validation_page),
            STAGES,
        ):
            self.tabs.addTab(page, name)
        self.setCentralWidget(self.tabs)

        self.progress = QProgressBar()
        self.progress.setMaximumWidth(280)
        self.progress.setVisible(False)
        self.status = QStatusBar()
        self.status.addPermanentWidget(self.progress)
        self.setStatusBar(self.status)

        self.input_page.choose_source.connect(self.choose_source)
        self.input_page.choose_target.connect(self.choose_target)
        self.mapping_page.rebuild.connect(self.run_mapping)
        self.mapping_page.remap.connect(self.apply_override)
        self.settings_page.changed.connect(self._settings_changed)
        self.export_page.run.connect(self.run_conversion)
        self.tabs.currentChanged.connect(lambda _: self.refresh())

    # -- state ---------------------------------------------------------- #
    def refresh(self) -> None:
        for page in (self.input_page, self.mapping_page, self.settings_page,
                     self.preview_page, self.export_page, self.validation_page):
            page.refresh()

    def _settings_changed(self) -> None:
        self.conversion = None
        self.refresh()

    def _busy(self, message: str) -> None:
        self.progress.setVisible(True)
        self.progress.setRange(0, 0)          # indeterminate; the real work
        self.status.showMessage(message)
        for page in (self.input_page, self.mapping_page, self.export_page):
            page.setEnabled(False)

    def _idle(self, message: str = "") -> None:
        self.progress.setVisible(False)
        self.progress.setRange(0, 100)
        self.status.showMessage(message)
        for page in (self.input_page, self.mapping_page, self.export_page):
            page.setEnabled(True)
        if message:
            self.refresh()

    # -- jobs ------------------------------------------------------------ #
    def _launch(self, label: str, job) -> None:
        """Run `job` on a thread, with the window disabled while it runs."""
        if self._job is not None and self._job.is_running():
            return
        self._busy(label)
        self._job = Job(job)
        self._job.progressed.connect(self._on_progress)
        self._job.finished.connect(self._on_finished)
        self._job.start()

    def _on_progress(self, progress: Progress) -> None:
        self.progress.setRange(0, 100)
        self.progress.setValue(int(progress.fraction * 100))
        self.status.showMessage(f"{progress.stage}: {progress.detail}"
                                if progress.detail else progress.stage)

    def _on_finished(self, result: Result) -> None:
        self._job = None
        if not result.ok:
            self._idle("")
            QMessageBox.critical(self, "Failed", result.error)
            self.status.showMessage("failed")
            if result.traceback:
                print(result.traceback, file=sys.stderr)
            return
        self._handle(result.payload)
        self._idle()

    def _handle(self, payload: dict) -> None:
        """Route a finished job's payload to whichever page asked for it."""
        kind = payload.pop("_kind", "")
        if kind == "source":
            self.source = payload
            self.mapping = None
            self.conversion = None
            self.status.showMessage(
                f"loaded {len(payload['bones'])} bones, "
                f"{len(payload['clips'])} clip(s)")

        elif kind == "target":
            self.target = payload
            self.mapping = None
            self.conversion = None
            if not payload["addressable"]:
                QMessageBox.warning(
                    self, "No usable bones",
                    "This DFF has no HAnim skeleton, so none of its bones have "
                    "an animation id and nothing could be written to an IFP.")
            self.status.showMessage(
                f"loaded {payload['addressable']} addressable bones")

        elif kind == "mapping":
            self.mapping = payload
            self.conversion = None
            self.status.showMessage(
                f"{payload['mapped']}/{payload['total']} bones mapped, "
                f"quality {payload['quality']:.2f}")
            if not payload["exportable"]:
                QMessageBox.warning(
                    self, "Incomplete mapping",
                    "The mapping is not confident enough to export.\n\n"
                    + "\n".join(payload["blocked_by"][:8])
                    + "\n\nFix it on the Skeleton Mapping page.")

        elif kind == "conversion":
            self.conversion = payload
            self.tabs.setCurrentIndex(5)
            message = ("PASSED VALIDATION" if payload["passed"]
                       else "FAILED VALIDATION")
            self.status.showMessage(
                f"{message} -- {payload['objects']} bones, "
                f"{payload['frames']} keys, {payload['bytes']} bytes")
            if not payload["passed"]:
                QMessageBox.critical(
                    self, "FAILED VALIDATION",
                    f"{payload['path']} was written but did not validate.\n\n"
                    "It is not safe to ship. The Validation page has the "
                    "per-bone numbers.")
            elif not payload["clean"]:
                QMessageBox.warning(
                    self, "Written, with warnings",
                    f"{payload['path']} validated, but the checks on the "
                    f"Validation page found {len(payload['diagnostics'])} "
                    f"thing(s) worth reading before you ship it.")

    # -- file pickers ---------------------------------------------------- #
    def choose_source(self) -> None:
        path, _ = QFileDialog.getOpenFileName(
            self, "Choose the source animation", "",
            "FBX files (*.fbx);;All files (*)")
        if not path:
            return
        self._launch("Loading FBX", lambda report: _tag(
            pipeline.load_source(path, report), "source"))

    def choose_target(self) -> None:
        path, _ = QFileDialog.getOpenFileName(
            self, "Choose the target ped", "",
            "GTA model files (*.dff);;All files (*)")
        if not path:
            return
        self._launch("Loading target DFF", lambda report: _tag(
            pipeline.load_target(path, report), "target"))

    # -- mapping --------------------------------------------------------- #
    def run_mapping(self) -> None:
        if not (self.source and self.target):
            QMessageBox.information(
                self, "Nothing to map",
                "Load an FBX and a DFF on the Input page first.")
            return
        self._launch("Mapping", lambda report: _tag(
            pipeline.build_mapping(self.source, self.target, report), "mapping"))

    def apply_override(self, source_bone: str, tag: int) -> None:
        """Re-map one bone the user corrected by hand.

        The override has to survive into the conversion, not just the table,
        so it is stored on the window and replayed onto every mapping built
        from these two files.
        """
        if not hasattr(self, "_overrides"):
            self._overrides: dict[str, int] = {}
        self._overrides[source_bone] = tag
        self._launch("Applying override", lambda report: _tag(
            pipeline.build_mapping(self.source, self.target, report,
                                   overrides=self._overrides), "mapping"))

    # -- conversion ------------------------------------------------------ #
    def run_conversion(self) -> None:
        settings = self.export_page.settings()
        if not settings["out_path"]:
            QMessageBox.information(self, "No output path",
                                    "Choose where to write the IFP.")
            return
        if not (self.mapping and self.mapping["exportable"]):
            QMessageBox.warning(self, "Incomplete mapping",
                                "The mapping must be complete before exporting.")
            return
        self._launch("Converting", lambda report: _tag(
            pipeline.convert(self.source, self.target, self.mapping["rows"],
                             settings, settings["out_path"], report),
            "conversion"))


def _tag(payload: dict, kind: str) -> dict:
    payload["_kind"] = kind
    return payload


def main(argv=None) -> int:
    app = QApplication.instance() or QApplication(argv or sys.argv)
    app.setApplicationName(APP_NAME)
    window = MainWindow()
    window.show()
    return app.exec()


if __name__ == "__main__":
    raise SystemExit(main())
