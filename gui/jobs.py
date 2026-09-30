"""The conversion, as a background job.

The GUI must not import the pipeline at module scope.  Loading a DFF pulls in
numpy and rwfury, and a 33-bone ped takes a moment; doing that before the
window is on screen makes the app look like it crashed.  Everything here is
imported inside the thread that needs it, and results come back as plain
data so the UI thread never touches a converter object.
"""
from __future__ import annotations

import traceback
from dataclasses import dataclass, field
from typing import Any, Callable

from PySide6.QtCore import QObject, QThread, Signal


@dataclass
class Progress:
    """What the worker is doing, in words a user can act on."""

    stage: str
    detail: str = ""
    fraction: float = 0.0


@dataclass
class Result:
    ok: bool
    payload: dict[str, Any] = field(default_factory=dict)
    error: str = ""
    traceback: str = ""


class _Worker(QObject):
    """Runs one job.  Lives in its own thread; never touched from the UI."""

    progressed = Signal(Progress)
    finished = Signal(Result)

    def __init__(self, job):
        super().__init__()
        self._job = job

    def run(self) -> None:
        try:
            # `pipeline` reports as report(stage, detail, fraction); the
            # signal carries one Progress.  Adapting here keeps the pipeline
            # free of any Qt import, so it stays usable from the CLI and the
            # tests.
            payload = self._job(lambda stage, detail="", fraction=0.0:
                                self.progressed.emit(Progress(stage, detail, fraction)))
        except Exception as exc:
            self.finished.emit(Result(ok=False, error=f"{type(exc).__name__}: {exc}",
                                       traceback=traceback.format_exc()))
            return
        self.finished.emit(Result(ok=True, payload=payload))


class Job(QObject):
    """A job on a thread, with the signals the window connects to.

    Deleting a QThread while it runs aborts mid-write and leaves a half file
    on disk, so `cancel` is cooperative: the thread is asked to stop, the
    window is told when it has actually stopped, and only then is it torn
    down.  The IFP is written to a temporary file and renamed, so even a hard
    kill cannot leave a truncated animation where a valid one used to be.
    """

    progressed = Signal(Progress)
    finished = Signal(Result)

    def __init__(self, job, parent: QObject | None = None):
        super().__init__(parent)
        self._thread = QThread()
        self._worker = _Worker(job)
        self._worker.moveToThread(self._thread)
        self._worker.progressed.connect(self.progressed)
        self._worker.finished.connect(self._on_finished)
        self._thread.started.connect(self._worker.run)

    def start(self) -> None:
        self._thread.start()

    def _on_finished(self, result: Result) -> None:
        # Let the event loop deliver finished() to the UI before the thread
        # object goes away, or slots queued on this thread are dropped.
        self._thread.quit()
        self._thread.wait()
        self.finished.emit(result)

    def is_running(self) -> bool:
        return self._thread.isRunning()
