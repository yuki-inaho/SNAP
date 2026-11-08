"""Integration tests: both demos must delegate to the shared loader."""

from __future__ import annotations

import importlib
from pathlib import Path

import numpy as np
import pytest

import pointcloud_loader


def _import_demo(module_name: str):
    try:
        return importlib.import_module(module_name)
    except Exception as exc:  # pragma: no cover - depends on the environment
        pytest.skip(
            f"{module_name} cannot be imported in this environment ({exc!r}); "
            "the loader itself is covered by test_pointcloud_loader.py"
        )


@pytest.fixture(scope="module")
def python_demo():
    return _import_demo("python_demo")


@pytest.fixture(scope="module")
def gui_demo():
    return _import_demo("iterative_gui_demo")


def test_demos_import_the_same_loader_function(python_demo, gui_demo):
    assert python_demo.load_point_cloud_file is pointcloud_loader.load_point_cloud
    assert gui_demo.load_point_cloud_file is pointcloud_loader.load_point_cloud


def test_python_demo_wrapper_matches_shared_loader(python_demo, examples: Path):
    expected = pointcloud_loader.load_point_cloud(examples / "KITTI" / "000000.bin")
    actual = python_demo.PointCloudData().load_point_cloud(
        str(examples / "KITTI" / "000000.bin")
    )
    assert set(actual) == set(expected)
    np.testing.assert_array_equal(actual["coord"], expected["coord"])
    np.testing.assert_array_equal(actual["intensity"], expected["intensity"])


def test_python_demo_wrapper_keeps_intensity_log(python_demo, examples: Path, capsys):
    python_demo.PointCloudData().load_point_cloud(
        str(examples / "KITTI" / "000000.bin")
    )
    out = capsys.readouterr().out
    assert "Intensity data found" in out
    assert "Min and max are:" in out


def test_python_demo_wrapper_prints_color_and_normal(python_demo, examples: Path, capsys):
    python_demo.PointCloudData().load_point_cloud(
        str(examples / "ScanNet" / "scene0011_00")
    )
    out = capsys.readouterr().out
    assert "Color data found" in out
    assert "Normal data found" in out


def test_gui_wrapper_loads_stpls3d_pth(gui_demo, examples: Path):
    """The GUI previously could not open the documented STPLS3D examples."""

    path = examples / "STPLS3D" / "example_1.pth"
    data = gui_demo.PointCloudData().load_point_cloud(str(path))
    expected = pointcloud_loader.load_point_cloud(path)
    np.testing.assert_array_equal(data["coord"], expected["coord"])
    np.testing.assert_array_equal(data["color"], expected["color"])


def test_wrappers_propagate_trust_pickle(python_demo, gui_demo, examples: Path):
    path = str(examples / "STPLS3D" / "example_1.pth")
    with pytest.raises(pointcloud_loader.PointCloudLoadError, match="trust_pickle=False"):
        python_demo.PointCloudData(trust_pickle=False).load_point_cloud(path)
    with pytest.raises(pointcloud_loader.PointCloudLoadError, match="trust_pickle=False"):
        gui_demo.PointCloudData(trust_pickle=False).load_point_cloud(path)


def test_stpls3d_colors_survive_demo_dataset_normalization(python_demo, examples: Path):
    """The loader's 0..255 convention is what DemoDatset's NormalizeColor expects."""

    data = python_demo.PointCloudData().load_point_cloud(
        str(examples / "STPLS3D" / "example_1.pth")
    )
    dataset = python_demo.DemoDatset(domain="Aerial")
    processed = dataset.process_data(
        data["coord"], data["color"], data.get("normal"), data.get("intensity")
    )

    rgb = processed["color"]
    assert float(rgb.min()) >= -1.001
    assert float(rgb.max()) <= 1.001
    # Guard against the historic double-normalisation that collapsed RGB to -1.
    assert float(rgb.max()) > 0.5
    assert processed["coord"].shape[1] == 3


@pytest.mark.parametrize(
    "relative,domain",
    [
        ("KITTI/000000.bin", "Outdoor"),
        ("ScanNet/scene0011_00", "Indoor"),
    ],
)
def test_shared_loader_feeds_demo_dataset(python_demo, examples: Path, relative: str, domain: str):
    """Loaded arrays must satisfy the transforms documented for each domain."""

    data = python_demo.PointCloudData().load_point_cloud(str(examples / relative))
    dataset = python_demo.DemoDatset(domain=domain)
    processed = dataset.process_data(
        data["coord"], data.get("color"), data.get("normal"), data.get("intensity")
    )

    assert processed["coord"].shape[1] == 3
    assert processed["grid_coord"].shape == processed["coord"].shape
    if domain == "Indoor":
        assert processed["color"].shape == processed["coord"].shape
        assert float(processed["color"].min()) >= -1.001
        assert float(processed["color"].max()) <= 1.001
        assert float(processed["normal"].abs().max()) <= 1.001
    else:
        assert "color" not in processed
        strength = processed["strength"]
        assert strength.shape[1] == 1
        assert float(strength.min()) >= 0.0
        assert float(strength.max()) <= 1.0


def test_gui_arg_parser_exposes_trust_pickle(gui_demo):
    parser = gui_demo.get_args_parser()
    args = parser.parse_args(["--no-trust_pickle"])
    assert args.trust_pickle is False
    args = parser.parse_args([])
    assert args.trust_pickle is True
