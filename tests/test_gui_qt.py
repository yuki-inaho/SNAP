"""Real Qt widget tests for ``iterative_gui_demo`` (real Qt + VTK).

Run this module on its own in a fresh process, with a display available:

    SNAP_QT_GUI_TESTS=1 xvfb-run -a python -m pytest tests/test_gui_qt.py -q

It is opt-in because a Qt OpenGL widget and an off-screen VTK render window
cannot be mixed reliably in a single VTK process on this stack.  The rest of
the GUI behaviour is covered by ``test_gui_interaction.py`` with a real
off-screen VTK plotter; only the model is faked in both modules.
"""

from __future__ import annotations

import os
import sys
from contextlib import contextmanager

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from gui_test_support import FakeModel, make_cloud  # noqa: E402

pytestmark = pytest.mark.skipif(
    os.environ.get("SNAP_QT_GUI_TESTS") != "1",
    reason=(
        "Qt widget tests are opt-in; run "
        "`SNAP_QT_GUI_TESTS=1 xvfb-run -a python -m pytest tests/test_gui_qt.py`"
    ),
)


@pytest.fixture(scope="module")
def gui():
    pytest.importorskip("PyQt6")
    pytest.importorskip("pyvistaqt")
    pytest.importorskip("pyvista")
    import iterative_gui_demo

    original = iterative_gui_demo.SegmentationModel
    iterative_gui_demo.SegmentationModel = FakeModel
    yield iterative_gui_demo
    iterative_gui_demo.SegmentationModel = original


@pytest.fixture(scope="module")
def qapp(gui):
    if not os.environ.get("DISPLAY") or os.environ.get("QT_QPA_PLATFORM") == "offscreen":
        pytest.skip("Qt widget tests need a display; run them under `xvfb-run -a`")
    from PyQt6.QtWidgets import QApplication

    app = QApplication.instance() or QApplication([])
    yield app


@contextmanager
def demo_console(window, capsys):
    """Send ``print`` output to the GUI console while capture is suspended."""

    window.console_output.clear()
    with capsys.disabled():
        previous = sys.stdout
        sys.stdout = window.stream
        try:
            yield
        finally:
            sys.stdout = previous


@pytest.fixture(scope="module")
def window(gui, qapp):
    pv = pytest.importorskip("pyvista")
    # QtInteractor only creates a picking interactor when it is not off-screen.
    pv.OFF_SCREEN = False
    gui.clip.tokenize = lambda text: torch.zeros((1, 77), dtype=torch.long)

    args = gui.get_args_parser().parse_args(
        ["--domain", "Indoor", "--resume", "unused.pth"]
    )
    original_stdout = sys.stdout
    try:
        win = gui.MainWindow(args, make_cloud())
    except Exception as exc:  # pragma: no cover - depends on the GL stack
        sys.stdout = original_stdout
        pytest.skip(f"QtInteractor/VTK could not initialise: {exc!r}")

    win.show()
    qapp.processEvents()
    try:
        yield win
    finally:
        win.close()
        win.deleteLater()
        qapp.processEvents()
        sys.stdout = original_stdout


def test_window_widgets_and_shortcuts(window):
    assert window.windowTitle() == "SNAP: Interactive 3D Segmentation"
    assert window.btn_new_object.text() == "New Object (N)"
    assert window.btn_new_object.shortcut().toString() == "N"
    assert window.btn_segment_points.shortcut().toString() == "F"
    assert window.btn_segment_everything.shortcut().toString() == "E"
    assert window.btn_segment_text.shortcut().toString() == "T"
    assert window.btn_clear_clicks.shortcut().toString() == "C"
    assert window.btn_clear_masks.shortcut().toString() == "M"
    assert window.plotter_handler.pc.n_points == len(make_cloud()["coord"])


def test_button_click_flows_through_handler(window):
    handler = window.plotter_handler
    handler.point_picking_callback(handler.model_data["coord"][0].cpu().numpy(), 0)

    window.btn_new_object.click()
    assert len(handler.all_click_sets) == 1

    window.btn_segment_points.click()
    assert len(handler.finalized_objects) == 1
    assert handler.actor_names["labels"]

    window.btn_clear_masks.click()
    assert handler.finalized_objects == []


def test_new_object_button_without_clicks_logs_to_console(window, capsys):
    with demo_console(window, capsys):
        window.btn_new_object.click()
        assert "No new clicks" in window.console_output.toPlainText()


def test_empty_text_prompt_warns_in_console(window, capsys):
    with demo_console(window, capsys):
        window.btn_segment_text.click()
        assert "Text prompt is empty" in window.console_output.toPlainText()


def test_text_prompt_runs_text_search(window, capsys):
    with demo_console(window, capsys):
        window.text_prompt_input.setText("chair")
        window.btn_segment_text.click()
        assert window.plotter_handler.cached_masks is not None
        assert "Best mask" in window.console_output.toPlainText()


def test_console_captures_stdout(window, capsys):
    with demo_console(window, capsys):
        print("hello from the demo")
        assert "hello from the demo" in window.console_output.toPlainText()


def test_segment_everything_button(window):
    window.btn_segment_everything.click()
    assert window.plotter_handler.cached_masks is not None
    assert window.plotter_handler.actor_names["labels"]
