"""Plotter interaction tests for ``iterative_gui_demo`` (real VTK, fake model).

These tests construct the real :class:`InteractivePlotter` with a real
off-screen VTK render window and drive the click/segment state machine directly.
The Qt widget layer is covered separately in ``test_gui_qt.py`` so that a VTK
render window and a Qt OpenGL widget never have to coexist in one process.
"""

from __future__ import annotations

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from gui_test_support import FakeModel, make_cloud  # noqa: E402


@pytest.fixture(scope="module")
def gui():
    pytest.importorskip("pyvista")
    import iterative_gui_demo

    return iterative_gui_demo


@pytest.fixture(scope="module", autouse=True)
def _patch_global_model(gui):
    """InteractivePlotter instantiates the model through this module global."""

    original = gui.SegmentationModel
    gui.SegmentationModel = FakeModel
    yield
    gui.SegmentationModel = original


@pytest.fixture
def args(gui):
    return gui.get_args_parser().parse_args(
        ["--domain", "Indoor", "--resume", "unused.pth"]
    )


@pytest.fixture
def plotter():
    pv = pytest.importorskip("pyvista")
    try:
        plotter = pv.Plotter(off_screen=True)
    except Exception as exc:  # pragma: no cover - depends on the GL stack
        pytest.skip(f"VTK cannot create an off-screen plotter: {exc!r}")
    yield plotter
    plotter.close()


@pytest.fixture
def handler(gui, args, plotter):
    instance = gui.InteractivePlotter(args, make_cloud(), plotter)
    # ``enable_point_picking`` registers an observer, so the render window
    # interactor has to exist before ``Plotter.update`` is called.
    plotter.show(auto_close=False, interactive=False)
    return instance


def test_plotter_initialises_scene(handler):
    assert handler.pc.n_points == len(make_cloud()["coord"])
    assert handler.vis_colors.shape == (handler.pc.n_points, 3)
    assert handler.current_clicks == []
    assert handler.all_click_sets == []


def test_click_adds_prompt_and_preview(handler):
    point = handler.model_data["coord"][10].cpu().numpy()
    handler.point_picking_callback(point, 10)

    assert len(handler.current_clicks) == 1
    assert len(handler.actor_names["spheres"]) == 1
    rgb = np.asarray(handler.pc["RGB"])
    assert not np.allclose(rgb, handler.vis_colors)


def test_new_object_finalises_click_set(handler):
    handler.point_picking_callback(handler.model_data["coord"][0].cpu().numpy(), 0)
    handler.new_object()
    assert handler.all_click_sets and len(handler.all_click_sets) == 1
    assert handler.current_clicks == []


def test_segment_from_points_populates_masks_and_labels(handler):
    handler.point_picking_callback(handler.model_data["coord"][0].cpu().numpy(), 0)
    handler.segment_from_points()

    assert len(handler.finalized_objects) == 1
    assert handler.finalized_objects[0]["mask"].shape == (handler.pc.n_points,)
    assert handler.actor_names["labels"], "a text label actor should be added"
    assert handler.all_click_sets == []
    rgb = np.asarray(handler.pc["RGB"])
    assert not np.allclose(rgb, handler.vis_colors)


def test_segment_without_clicks_is_a_noop(handler, capsys):
    handler.segment_from_points()
    assert "No prompt sets created" in capsys.readouterr().out
    assert handler.finalized_objects == []


def test_clear_masks_restores_colors(handler):
    handler.point_picking_callback(handler.model_data["coord"][0].cpu().numpy(), 0)
    handler.segment_from_points()
    handler.clear_masks()

    assert handler.finalized_objects == []
    np.testing.assert_allclose(np.asarray(handler.pc["RGB"]), handler.vis_colors)
    assert handler.actor_names["labels"] == []


def test_clear_clicks_resets_prompt_state(handler):
    handler.point_picking_callback(handler.model_data["coord"][0].cpu().numpy(), 0)
    handler.new_object()
    handler.point_picking_callback(handler.model_data["coord"][1].cpu().numpy(), 1)
    handler.clear_clicks()

    assert handler.current_clicks == []
    assert handler.all_click_sets == []
    assert handler.actor_names["spheres"] == []
    np.testing.assert_allclose(np.asarray(handler.pc["RGB"]), handler.vis_colors)


def test_segment_everything_caches_masks_and_labels(handler, gui, monkeypatch):
    monkeypatch.setattr(
        gui.clip, "tokenize", lambda text: torch.zeros((1, 77), dtype=torch.long)
    )
    handler.segment_everything()

    assert handler.cached_masks is not None and len(handler.cached_masks) == 2
    assert handler.cached_tokens is not None and len(handler.cached_tokens) == 2
    assert handler.actor_names["labels"], "labels from segment everything should be shown"
    rgb = np.asarray(handler.pc["RGB"])
    assert not np.allclose(rgb, handler.vis_colors)


def test_text_search_selects_cached_mask(handler, gui, monkeypatch):
    monkeypatch.setattr(
        gui.clip, "tokenize", lambda text: torch.zeros((1, 77), dtype=torch.long)
    )
    handler.visualize_best_mask("chair")

    assert handler.cached_masks is not None
    rgb = np.asarray(handler.pc["RGB"])
    red = np.all(np.isclose(rgb, [1.0, 0.2, 0.2], atol=1e-5), axis=1)
    assert red.any()


def test_gui_loader_returns_coord_and_color(gui, examples):
    data = gui.PointCloudData().load_point_cloud(
        str(examples / "ScanNet" / "scene0011_00")
    )
    assert data["coord"].shape[1] == 3
    assert data["color"].max() <= 255.0
    assert data["normal"].shape == data["coord"].shape


# ---------------------------------------------------------------------------
# Pure helpers
# ---------------------------------------------------------------------------
def test_voxel_downsample_returns_original_points(gui):
    points = np.array(
        [[0.01, 0.01, 0.01], [0.02, 0.02, 0.02], [1.01, 1.01, 1.01]], dtype=np.float64
    )
    downsampled = gui.voxel_downsample(points, 1.0)
    assert downsampled.shape == (2, 3)
    for point in downsampled:
        assert np.any(np.all(np.isclose(points, point), axis=1))


def test_nms_suppresses_overlapping_masks(gui):
    masks = torch.tensor(
        [[1, 1, 0, 0], [1, 1, 1, 0], [0, 0, 0, 1]], dtype=torch.float32
    )
    scores = torch.tensor([0.9, 0.8, 0.7])
    keep = gui.non_maximum_suppression_masks(masks, scores, 0.3)
    assert keep.tolist() == [0, 2]
