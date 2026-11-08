"""Tests for :mod:`pointcloud_loader`.

The suite loads the real files shipped in ``data_examples/`` so that the
documented example commands keep working, and it exercises malformed inputs so
that failures are explicit instead of silently producing empty clouds.
"""

from __future__ import annotations

import warnings
from pathlib import Path

import numpy as np
import pytest

from pointcloud_loader import (
    STPLS3D_COORD_SCALE,
    PointCloudLoadError,
    load_point_cloud,
)

pytestmark = pytest.mark.filterwarnings("ignore::RuntimeWarning")


# ---------------------------------------------------------------------------
# Real examples (documented demo inputs)
# ---------------------------------------------------------------------------
def test_kitti_bin_matches_raw_file(examples: Path):
    path = examples / "KITTI" / "000000.bin"
    data = load_point_cloud(path)

    raw = np.fromfile(path, dtype=np.float32).reshape(-1, 4)
    assert data["coord"].shape == (raw.shape[0], 3)
    assert data["coord"].dtype == np.float32
    np.testing.assert_array_equal(data["coord"], raw[:, :3])
    assert data["intensity"].shape == (raw.shape[0], 1)
    np.testing.assert_array_equal(
        data["intensity"].ravel(), raw[:, 3]
    )
    assert 0.0 <= float(data["intensity"].min())
    assert float(data["intensity"].max()) <= 1.0
    assert "color" not in data and "normal" not in data


def test_nuscenes_pcd_bin_matches_raw_file(examples: Path):
    path = examples / "nuScenes" / "example_3.pcd.bin"
    data = load_point_cloud(path)

    raw = np.fromfile(path, dtype=np.float32).reshape(-1, 5)
    np.testing.assert_array_equal(data["coord"], raw[:, :3])
    np.testing.assert_allclose(
        data["intensity"].ravel(), raw[:, 3] / 255.0, rtol=1e-6
    )
    assert data["coord"].shape[0] == raw.shape[0]
    assert float(data["intensity"].min()) >= 0.0


def test_pandaset_pkl_keeps_360_sweep(examples: Path):
    pd = pytest.importorskip("pandas")
    path = examples / "PandaSet" / "30.pkl"

    data = load_point_cloud(path)
    frame = pd.read_pickle(path)
    valid = np.where(frame.d == 0)[0]
    values = np.asarray(frame.values).astype(np.float32)

    assert data["coord"].shape == (len(valid), 3)
    np.testing.assert_allclose(data["coord"], values[valid][:, :3], rtol=1e-6)
    np.testing.assert_allclose(
        data["intensity"].ravel(), values[valid][:, 3] / 255.0, rtol=1e-6
    )
    # The 360 degree sweep is a strict subset for this example file.
    assert 0 < len(valid) < len(frame)


def test_scannet_directory_round_trips_arrays(examples: Path):
    path = examples / "ScanNet" / "scene0011_00"
    data = load_point_cloud(path)

    coord = np.load(path / "coord.npy")
    color = np.load(path / "color.npy")
    normal = np.load(path / "normal.npy")

    np.testing.assert_array_equal(data["coord"], coord.astype(np.float32))
    np.testing.assert_array_equal(data["color"], color.astype(np.float32))
    np.testing.assert_array_equal(data["normal"], normal.astype(np.float32))
    assert data["color"].max() <= 255.0
    assert data["color"].min() >= 0.0


def test_s3dis_directory_loads_large_cloud(examples: Path):
    path = examples / "S3DIS" / "conferenceRoom_1"
    data = load_point_cloud(path)

    assert data["coord"].shape == (1_047_554, 3)
    assert data["color"].shape == data["coord"].shape
    assert data["color"].max() == 255.0
    # S3DIS exports also carry normals in this repository.
    assert "normal" in data


@pytest.mark.parametrize(
    "filename",
    ["example_1.pth", "example_2.pth", "example_3.pth"],
)
def test_stpls3d_pth_loads_gui_compatible_colors(examples: Path, filename: str):
    torch = pytest.importorskip("torch")
    path = examples / "STPLS3D" / filename
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        data = load_point_cloud(path)

    try:
        raw = torch.load(path, weights_only=False)
    except TypeError:  # torch < 2.0
        raw = torch.load(path)
    raw_coord, raw_color = raw[:2]
    raw_coord = np.asarray(raw_coord, dtype=np.float32)
    raw_color = np.asarray(raw_color, dtype=np.float32)

    np.testing.assert_allclose(
        data["coord"], raw_coord * np.float32(STPLS3D_COORD_SCALE), rtol=1e-6
    )
    assert data["coord"].shape == (raw_coord.shape[0], 3)
    # STPLS3D stores colours in [-1, 1]; DemoDatset expects 0..255 because it
    # applies ``color / 127.5 - 1`` itself.
    assert data["color"].shape == raw_color.shape
    assert data["color"].min() >= 0.0
    assert data["color"].max() <= 255.0
    np.testing.assert_allclose(
        data["color"], (raw_color + 1.0) * 127.5, atol=1e-3
    )
    assert data["color"].max() > 1.0


def test_stpls3d_pth_coord_scale_is_configurable(examples: Path):
    pytest.importorskip("torch")
    path = examples / "STPLS3D" / "example_1.pth"
    scaled = load_point_cloud(path, pth_coord_scale=0.1)
    unscaled = load_point_cloud(path, pth_coord_scale=1.0)
    np.testing.assert_allclose(scaled["coord"] * 10.0, unscaled["coord"], rtol=1e-6)


def test_dales_npy_uses_first_three_columns(tmp_path: Path):
    array = np.array(
        [[1.0, 2.0, 3.0, 7], [4.0, 5.0, 6.0, 8]], dtype=np.float64
    )
    path = tmp_path / "cloud.npy"
    np.save(path, array)

    data = load_point_cloud(path)
    np.testing.assert_array_equal(data["coord"], array[:, :3].astype(np.float32))
    assert set(data) == {"coord"}


# ---------------------------------------------------------------------------
# Open3D formats: colours must reach the 0..255 convention
# ---------------------------------------------------------------------------
ASCII_PLY = """ply
format ascii 1.0
element vertex 3
property float x
property float y
property float z
property uchar red
property uchar green
property uchar blue
end_header
0 0 0 255 0 0
1 0 0 0 255 0
0 1 0 0 0 255
"""

ASCII_PCD = """# .PCD v0.7 - Point Cloud Data file format
VERSION 0.7
FIELDS x y z rgb
SIZE 4 4 4 4
TYPE F F F F
COUNT 1 1 1 1
WIDTH 2
HEIGHT 1
VIEWPOINT 0 0 0 1 0 0 0
POINTS 2
DATA ascii
0 0 0 2.3418052e-38
1 0 0 2.3418052e-38
"""


def _require_open3d():
    return pytest.importorskip("open3d", reason="open3d is needed for ply/pcd")


def test_ply_colors_are_scaled_to_byte_range(tmp_path: Path):
    _require_open3d()
    path = tmp_path / "cloud.ply"
    path.write_text(ASCII_PLY)

    data = load_point_cloud(path)
    assert data["coord"].shape == (3, 3)
    assert data["color"].shape == (3, 3)
    assert data["color"].max() > 1.0
    np.testing.assert_allclose(data["color"], [[255, 0, 0], [0, 255, 0], [0, 0, 255]], atol=1)


def test_pcd_cloud_is_loadable(tmp_path: Path):
    _require_open3d()
    path = tmp_path / "cloud.pcd"
    path.write_text(ASCII_PCD)

    data = load_point_cloud(path)
    assert data["coord"].shape == (2, 3)
    assert set(data["coord"][0]) == {0.0}


# ---------------------------------------------------------------------------
# Malformed inputs
# ---------------------------------------------------------------------------
def test_missing_path_raises(tmp_path: Path):
    with pytest.raises(PointCloudLoadError, match="does not exist"):
        load_point_cloud(tmp_path / "missing.bin")


def test_unsupported_extension_lists_formats(tmp_path: Path):
    path = tmp_path / "cloud.xyz"
    path.write_text("0 0 0\n")
    with pytest.raises(PointCloudLoadError, match="Unsupported point cloud format") as info:
        load_point_cloud(path)
    assert ".pth" in str(info.value)


def test_empty_bin_raises(tmp_path: Path):
    path = tmp_path / "empty.bin"
    path.write_bytes(b"")
    with pytest.raises(PointCloudLoadError, match="Empty KITTI"):
        load_point_cloud(path)


def test_truncated_kitti_bin_raises(tmp_path: Path):
    path = tmp_path / "truncated.bin"
    np.arange(7, dtype=np.float32).tofile(path)
    with pytest.raises(PointCloudLoadError, match="multiple of 4"):
        load_point_cloud(path)


def test_truncated_nuscenes_bin_raises(tmp_path: Path):
    path = tmp_path / "truncated.pcd.bin"
    np.arange(9, dtype=np.float32).tofile(path)
    with pytest.raises(PointCloudLoadError, match="multiple of 5"):
        load_point_cloud(path)


def test_nuscenes_extension_is_checked_before_bin(tmp_path: Path):
    path = tmp_path / "cloud.PCD.BIN"
    raw = np.array([[1, 2, 3, 255, 7]], dtype=np.float32)
    raw.tofile(path)
    data = load_point_cloud(path)
    assert data["coord"].shape == (1, 3)
    assert data["intensity"][0, 0] == pytest.approx(1.0)


def test_npy_needs_three_columns(tmp_path: Path):
    path = tmp_path / "bad.npy"
    np.save(path, np.zeros((4, 2), dtype=np.float32))
    with pytest.raises(PointCloudLoadError, match=r"\(N, >=3\)"):
        load_point_cloud(path)


def test_directory_without_coord_lists_npy_files(tmp_path: Path):
    (tmp_path / "color.npy").write_bytes(b"")
    with pytest.raises(PointCloudLoadError, match="coord.npy") as info:
        load_point_cloud(tmp_path)
    assert "color.npy" in str(info.value)


def test_directory_color_length_mismatch_raises(tmp_path: Path):
    np.save(tmp_path / "coord.npy", np.zeros((4, 3), dtype=np.float32))
    np.save(tmp_path / "color.npy", np.zeros((3, 3), dtype=np.uint8))
    with pytest.raises(PointCloudLoadError, match="color has 3 entries"):
        load_point_cloud(tmp_path)


def test_directory_intensity_is_not_silently_used_as_color(tmp_path: Path):
    np.save(tmp_path / "coord.npy", np.zeros((2, 3), dtype=np.float32))
    np.save(tmp_path / "color.npy", np.zeros((2, 2), dtype=np.uint8))
    with pytest.raises(PointCloudLoadError, match=r"Color array must have shape"):
        load_point_cloud(tmp_path)


def test_empty_coordinate_array_raises(tmp_path: Path):
    np.save(tmp_path / "coord.npy", np.zeros((0, 3), dtype=np.float32))
    with pytest.raises(PointCloudLoadError, match="contains no points"):
        load_point_cloud(tmp_path)


def test_nonfinite_coordinates_are_dropped(tmp_path: Path):
    np.save(
        tmp_path / "coord.npy",
        np.array([[0, 0, 0], [np.nan, 1, 1], [1, 1, 1]], dtype=np.float32),
    )
    np.save(tmp_path / "color.npy", np.zeros((3, 3), dtype=np.uint8))

    with pytest.warns(RuntimeWarning, match="dropping 1 point"):
        data = load_point_cloud(tmp_path)
    assert data["coord"].shape == (2, 3)
    assert data["color"].shape == (2, 3)


def test_nonfinite_coordinates_can_be_rejected(tmp_path: Path):
    np.save(tmp_path / "coord.npy", np.array([[np.inf, 0, 0]], dtype=np.float32))
    with pytest.raises(PointCloudLoadError, match="non-finite"):
        load_point_cloud(tmp_path, on_nonfinite="error")


def test_nonfinite_color_is_zeroed_not_dropped(tmp_path: Path):
    np.save(tmp_path / "coord.npy", np.zeros((2, 3), dtype=np.float32))
    np.save(
        tmp_path / "color.npy",
        np.array([[np.nan, 1, 1], [2, 3, 4]], dtype=np.float32),
    )
    with pytest.warns(RuntimeWarning, match="non-finite color"):
        data = load_point_cloud(tmp_path)
    assert np.isfinite(data["color"]).all()
    assert data["color"][0, 0] == 0.0


def test_out_of_range_color_is_clipped(tmp_path: Path):
    np.save(tmp_path / "coord.npy", np.zeros((2, 3), dtype=np.float32))
    np.save(tmp_path / "color.npy", np.array([[-5, 1, 1], [300, 3, 4]], dtype=np.float32))
    with pytest.warns(RuntimeWarning, match="clipping"):
        data = load_point_cloud(tmp_path)
    assert data["color"].min() >= 0.0
    assert data["color"].max() <= 255.0


def test_bad_on_nonfinite_mode_is_rejected():
    with pytest.raises(ValueError, match="on_nonfinite"):
        load_point_cloud("data_examples/KITTI/000000.bin", on_nonfinite="ignore")


def test_pth_byte_colors_are_not_rescaled(tmp_path: Path):
    """Some STPLS3D exports store raw 0..255 bytes instead of [-1, 1]."""

    torch = pytest.importorskip("torch")
    color = np.array([[255, 0, 0], [0, 128, 255]], dtype=np.uint8)
    path = tmp_path / "bytecolor.pth"
    torch.save((np.zeros((2, 3), dtype=np.float32), color), path)

    data = load_point_cloud(path)
    np.testing.assert_array_equal(data["color"], color.astype(np.float32))


def test_directory_rgba_colors_drop_alpha(tmp_path: Path):
    np.save(tmp_path / "coord.npy", np.zeros((4, 3), dtype=np.float32))
    rgba = np.zeros((4, 4), dtype=np.uint8)
    rgba[:, 0] = 200
    np.save(tmp_path / "color.npy", rgba)

    with pytest.warns(RuntimeWarning, match="alpha"):
        data = load_point_cloud(tmp_path)
    assert data["color"].shape == (4, 3)
    np.testing.assert_array_equal(data["color"][:, 0], [200, 200, 200, 200])


def test_pth_rejects_non_tuple_payload(tmp_path: Path):
    torch = pytest.importorskip("torch")
    path = tmp_path / "weights.pth"
    torch.save({"state_dict": {}}, path)
    with pytest.raises(PointCloudLoadError, match="tuple/list"):
        load_point_cloud(path)


def test_pth_rejects_missing_color(tmp_path: Path):
    torch = pytest.importorskip("torch")
    path = tmp_path / "one.pth"
    torch.save((np.zeros((4, 3), dtype=np.float32),), path)
    with pytest.raises(PointCloudLoadError, match="at least"):
        load_point_cloud(path)


def test_pth_rejects_bad_coord_shape(tmp_path: Path):
    torch = pytest.importorskip("torch")
    path = tmp_path / "bad.pth"
    torch.save((np.zeros((4, 2), dtype=np.float32), np.zeros((4, 3), dtype=np.float32)), path)
    with pytest.raises(PointCloudLoadError, match=r"coord must have shape \(N, 3\)"):
        load_point_cloud(path)


def test_pth_rejects_point_count_mismatch(tmp_path: Path):
    torch = pytest.importorskip("torch")
    path = tmp_path / "mismatch.pth"
    torch.save((np.zeros((4, 3), dtype=np.float32), np.zeros((3, 3), dtype=np.float32)), path)
    with pytest.raises(PointCloudLoadError, match="disagree on point count"):
        load_point_cloud(path)


def test_pth_can_be_refused(tmp_path: Path):
    torch = pytest.importorskip("torch")
    path = tmp_path / "cloud.pth"
    torch.save((np.zeros((4, 3), dtype=np.float32), np.zeros((4, 3), dtype=np.float32)), path)
    with pytest.raises(PointCloudLoadError, match="trust_pickle=False"):
        load_point_cloud(path, trust_pickle=False)


def test_pkl_can_be_refused(tmp_path: Path):
    path = tmp_path / "cloud.pkl"
    path.write_bytes(b"not a pickle")
    with pytest.raises(PointCloudLoadError, match="trust_pickle=False"):
        load_point_cloud(path, trust_pickle=False)


def test_pkl_without_sweep_column_raises(tmp_path: Path):
    pd = pytest.importorskip("pandas")
    frame = pd.DataFrame({"x": [0.0], "y": [0.0], "z": [0.0], "i": [1]})
    path = tmp_path / "nosweep.pkl"
    frame.to_pickle(path)
    with pytest.raises(PointCloudLoadError, match="'d' column"):
        load_point_cloud(path)


def test_pkl_without_matching_sweep_raises(tmp_path: Path):
    pd = pytest.importorskip("pandas")
    frame = pd.DataFrame(
        {"x": [0.0], "y": [0.0], "z": [0.0], "i": [1], "d": [1]}
    )
    path = tmp_path / "nosweeprows.pkl"
    frame.to_pickle(path)
    with pytest.raises(PointCloudLoadError, match="d == 0"):
        load_point_cloud(path)


def test_corrupt_ply_raises_loader_error(tmp_path: Path):
    _require_open3d()
    path = tmp_path / "broken.ply"
    path.write_text("this is not a ply file")
    with pytest.raises(PointCloudLoadError):
        load_point_cloud(path)


def test_empty_cloud_is_rejected_not_returned_as_none(tmp_path: Path):
    path = tmp_path / "zero.pcd.bin"
    np.zeros((0,), dtype=np.float32).tofile(path)
    with pytest.raises(PointCloudLoadError, match="Empty"):
        load_point_cloud(path)
