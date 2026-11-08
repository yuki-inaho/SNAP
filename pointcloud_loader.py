"""Shared point cloud loading utilities for the SNAP interactive demos.

Both ``python_demo.py`` and ``iterative_gui_demo.py`` used to carry their own
copy of :class:`PointCloudData`.  The two copies drifted apart (the GUI copy
never learned the STPLS3D ``.pth`` format) and neither one validated its input.
This module is the single implementation used by both demos.

Supported inputs
----------------
``.ply``, ``.pcd``
    Read with Open3D.  Open3D returns colours as floats in ``[0, 1]``; they are
    converted to the ``0..255`` convention described below.
``.pcd.bin``
    nuScenes format: ``float32`` records of ``x, y, z, intensity, ring``.
    Intensity is divided by 255 (the nuScenes convention), exactly like the
    original demo loaders.
``.bin``
    KITTI format: ``float32`` records of ``x, y, z, intensity``.  Intensity is
    left untouched (KITTI stores it in ``[0, 1]``).
``.pkl``
    PandaSet format: a pickled :class:`pandas.DataFrame`.  Only rows with
    ``d == 0`` (the 360 degree sweep) are kept.  Columns ``x, y, z`` are used
    for coordinates and ``i`` (falling back to the fourth column) for
    intensity, divided by 255 like the original loaders.
``.pth``
    STPLS3D demo format: a tuple ``(coord, color[, semantic, instance])`` of
    numpy arrays or torch tensors.  Coordinates are multiplied by
    ``0.1`` (``STPLS3D_COORD_SCALE``) to preserve the legacy demo behaviour.
    Stored colours are expected in ``[-1, 1]`` (the STPLS3D pre-processing
    result) and are converted to ``0..255``.
``.npy``
    DALES style array of shape ``(N, >=3)``; the first three columns are used
    as coordinates.
directory
    A folder containing ``coord.npy`` and optionally ``color.npy`` /
    ``normal.npy`` (ScanNet, S3DIS, HM3D exports).

Output conventions
------------------
Every loader returns a plain ``dict`` (never ``None`` values):

``coord``
    ``(N, 3)`` ``float32`` coordinates in the coordinate system stored in the
    file.  For ``.pth`` the legacy ``0.1`` scale is applied.  Prompt points
    clicked in the GUI therefore live in the same frame.
``color``
    Optional ``(N, 3)`` ``float32`` RGB in ``0..255``.  This is the convention
    consumed by :class:`datasets.demo.DemoDatset`, which applies
    ``NormalizeColor`` (``color / 127.5 - 1``) for the Indoor and Aerial
    domains.  Byte colour arrays (``uint8`` ``0..255``) are passed through,
    Open3D's ``[0, 1]`` floats are scaled, and STPLS3D's ``[-1, 1]`` colours are
    converted back to ``0..255``.
``normal``
    Optional ``(N, 3)`` ``float32`` surface normals, passed through unchanged.
``intensity``
    Optional ``(N, 1)`` ``float32`` intensity.  KITTI is ``[0, 1]``, nuScenes
    and PandaSet are ``0..255 / 255``.  ``DemoDatset`` maps this to the
    ``strength`` feature, so the scale is part of the model input contract.

Robustness
----------
* Missing, empty, truncated and structurally invalid files raise
  :class:`PointCloudLoadError` with the offending path and a hint.
* Non-finite coordinates are dropped (default) or rejected; non-finite
  colour/normal/intensity values are replaced with zeros.  Both are reported.
* Colour arrays must agree with the coordinate array in length and are clipped
  to ``0..255``.

Trust constraints for pickle based formats
------------------------------------------
``.pth`` (``torch.load``) and ``.pkl`` (``pandas.read_pickle``) execute
arbitrary Python during unpickling.  Only load files from a trusted source.
``load_point_cloud(..., trust_pickle=False)`` refuses unsafe loads and the
``.pth`` path first tries ``torch.load(weights_only=True)`` where the installed
torch version supports it; the unsafe fallback is announced with a warning.
"""

from __future__ import annotations

import warnings
from pathlib import Path
from typing import Any, Mapping

import numpy as np

__all__ = [
    "KITTI_POINT_DIM",
    "NUSCENES_POINT_DIM",
    "PointCloudLoadError",
    "STPLS3D_COORD_SCALE",
    "SUPPORTED_FORMATS",
    "load_dales_npy",
    "load_kitti_bin",
    "load_npy_directory",
    "load_nuscenes_bin",
    "load_pandaset_pkl",
    "load_point_cloud",
    "load_stpls3d_pth",
]

KITTI_POINT_DIM = 4
NUSCENES_POINT_DIM = 5
STPLS3D_COORD_SCALE = 0.1

SUPPORTED_FORMATS = (
    ".ply",
    ".pcd",
    ".pcd.bin",
    ".bin",
    ".pkl",
    ".pth",
    ".npy",
    "directory{coord.npy,color.npy,normal.npy}",
)

_COLOR_EPS = 1e-4
_NONFINITE_MODES = ("drop", "error")


class PointCloudLoadError(ValueError):
    """Raised when a point cloud cannot be loaded or is structurally invalid."""


def load_point_cloud(
    filename: str | Path,
    *,
    trust_pickle: bool = True,
    pth_coord_scale: float = STPLS3D_COORD_SCALE,
    on_nonfinite: str = "drop",
) -> dict[str, np.ndarray]:
    """Load a point cloud and return ``coord`` plus optional feature arrays.

    Parameters
    ----------
    filename:
        File or directory to load.  ``~`` is expanded.
    trust_pickle:
        Allow ``.pth`` / ``.pkl`` files to be unpickled.  Set to ``False`` to
        reject them with :class:`PointCloudLoadError`.  See the module
        docstring for the security implications.
    pth_coord_scale:
        Scale applied to STPLS3D ``.pth`` coordinates.  Defaults to the legacy
        demo value ``0.1``; pass ``1.0`` to disable it.
    on_nonfinite:
        ``"drop"`` (default) removes points with non-finite coordinates,
        ``"error"`` raises instead.
    """

    if on_nonfinite not in _NONFINITE_MODES:
        raise ValueError(
            f"on_nonfinite must be one of {_NONFINITE_MODES}, got {on_nonfinite!r}"
        )

    path = Path(filename).expanduser()
    if not path.exists():
        raise PointCloudLoadError(f"Point cloud path does not exist: {path}")

    if path.is_dir():
        result = load_npy_directory(path)
    else:
        name = path.name.lower()
        # ``.pcd.bin`` has to be checked before ``.bin``.
        if name.endswith(".pcd.bin"):
            result = load_nuscenes_bin(path)
        elif name.endswith(".bin"):
            result = load_kitti_bin(path)
        elif name.endswith(".ply") or name.endswith(".pcd"):
            result = _load_open3d(path)
        elif name.endswith(".pkl"):
            result = load_pandaset_pkl(path, trust_pickle=trust_pickle)
        elif name.endswith(".pth"):
            result = load_stpls3d_pth(
                path, trust_pickle=trust_pickle, coord_scale=pth_coord_scale
            )
        elif name.endswith(".npy"):
            result = load_dales_npy(path)
        else:
            raise PointCloudLoadError(
                f"Unsupported point cloud format for {path}. "
                f"Supported formats: {', '.join(SUPPORTED_FORMATS)}"
            )

    return _finalize(result, path=path, on_nonfinite=on_nonfinite)


# ---------------------------------------------------------------------------
# Format specific loaders
# ---------------------------------------------------------------------------
def load_kitti_bin(path: str | Path) -> dict[str, np.ndarray]:
    """Read a KITTI ``velodyne`` ``.bin`` file (``x, y, z, intensity``)."""

    path = Path(path)
    raw = np.fromfile(path, dtype=np.float32)
    if raw.size == 0:
        raise PointCloudLoadError(f"Empty KITTI .bin file: {path}")
    if raw.size % KITTI_POINT_DIM != 0:
        raise PointCloudLoadError(
            f"Truncated KITTI .bin file: {path} has {raw.size} float32 values, "
            f"which is not a multiple of {KITTI_POINT_DIM}."
        )
    pcdata = raw.reshape(-1, KITTI_POINT_DIM)
    return {"coord": pcdata[:, :3], "intensity": pcdata[:, 3:4]}


def load_nuscenes_bin(path: str | Path) -> dict[str, np.ndarray]:
    """Read a nuScenes ``.pcd.bin`` file (``x, y, z, intensity, ring``)."""

    path = Path(path)
    raw = np.fromfile(path, dtype=np.float32)
    if raw.size == 0:
        raise PointCloudLoadError(f"Empty nuScenes .pcd.bin file: {path}")
    if raw.size % NUSCENES_POINT_DIM != 0:
        raise PointCloudLoadError(
            f"Truncated nuScenes .pcd.bin file: {path} has {raw.size} float32 "
            f"values, which is not a multiple of {NUSCENES_POINT_DIM}."
        )
    pcdata = raw.reshape(-1, NUSCENES_POINT_DIM)
    return {"coord": pcdata[:, :3], "intensity": pcdata[:, 3:4] / 255.0}


def load_pandaset_pkl(
    path: str | Path, *, trust_pickle: bool = True
) -> dict[str, np.ndarray]:
    """Read a PandaSet ``.pkl`` DataFrame and keep the 360 degree sweep."""

    path = Path(path)
    if not trust_pickle:
        raise PointCloudLoadError(
            f"Refusing to unpickle {path}: trust_pickle=False. "
            "PandaSet .pkl files can execute arbitrary code when loaded."
        )

    try:
        import pandas as pd  # imported lazily: only this format needs pandas
    except ImportError as exc:  # pragma: no cover - depends on the environment
        raise PointCloudLoadError(
            f"Loading {path.name} requires pandas, which is not installed."
        ) from exc

    try:
        frame = pd.read_pickle(path)
    except Exception as exc:  # noqa: BLE001 - surface a loader specific error
        raise PointCloudLoadError(
            f"Could not unpickle PandaSet file {path}: {exc}"
        ) from exc

    if not hasattr(frame, "columns") or not hasattr(frame, "values"):
        raise PointCloudLoadError(
            f"PandaSet .pkl must contain a pandas DataFrame, got "
            f"{type(frame).__name__} in {path}"
        )
    if "d" not in frame.columns:
        raise PointCloudLoadError(
            f"PandaSet .pkl is missing the 'd' column required to select the "
            f"360 degree sweep: {path}"
        )

    values = np.asarray(frame.values)
    if values.ndim != 2 or values.shape[1] < 4:
        raise PointCloudLoadError(
            f"PandaSet .pkl needs at least 4 columns (x, y, z, intensity), "
            f"got shape {values.shape} in {path}"
        )

    try:
        values = values.astype(np.float32)
    except (TypeError, ValueError) as exc:
        raise PointCloudLoadError(
            f"PandaSet .pkl contains non numeric columns: {path}"
        ) from exc

    sweep = np.asarray(frame["d"]).astype(np.float64)
    valid = np.where(sweep == 0)[0]
    if valid.size == 0:
        raise PointCloudLoadError(
            f"PandaSet .pkl contains no points with d == 0 (360 degree sweep): {path}"
        )

    index_of = {str(column): i for i, column in enumerate(frame.columns)}
    try:
        if {"x", "y", "z"} <= index_of.keys():
            coord_cols = [index_of["x"], index_of["y"], index_of["z"]]
        else:
            coord_cols = [0, 1, 2]
        intensity_col = index_of.get("i", 3)
    except (KeyError, IndexError) as exc:
        raise PointCloudLoadError(
            f"PandaSet .pkl has an unexpected column layout: {list(frame.columns)}"
        ) from exc

    coord = values[valid][:, coord_cols]
    intensity = values[valid][:, intensity_col].reshape(-1, 1) / 255.0
    return {"coord": coord, "intensity": intensity}


def load_stpls3d_pth(
    path: str | Path,
    *,
    trust_pickle: bool = True,
    coord_scale: float = STPLS3D_COORD_SCALE,
) -> dict[str, np.ndarray]:
    """Read an STPLS3D demo ``.pth`` tuple ``(coord, color[, ...])``."""

    path = Path(path)
    payload = _torch_load(path, trust_pickle=trust_pickle)

    if not isinstance(payload, (tuple, list)):
        raise PointCloudLoadError(
            f"STPLS3D .pth must contain a tuple/list (coord, color, ...), got "
            f"{type(payload).__name__} in {path}"
        )
    if len(payload) < 2:
        raise PointCloudLoadError(
            f"STPLS3D .pth must contain at least (coord, color); got "
            f"{len(payload)} element(s) in {path}"
        )

    try:
        coord = _to_numpy(payload[0])
        color = _to_numpy(payload[1])
    except (TypeError, ValueError) as exc:
        raise PointCloudLoadError(
            f"STPLS3D .pth coord/color cannot be converted to numpy arrays: {path}"
        ) from exc

    if coord.ndim != 2 or coord.shape[1] != 3:
        raise PointCloudLoadError(
            f"STPLS3D .pth coord must have shape (N, 3), got {coord.shape} in {path}"
        )
    if color.ndim != 2 or color.shape[1] != 3:
        raise PointCloudLoadError(
            f"STPLS3D .pth color must have shape (N, 3), got {color.shape} in {path}"
        )
    if coord.shape[0] != color.shape[0]:
        raise PointCloudLoadError(
            f"STPLS3D .pth coord and color disagree on point count: "
            f"{coord.shape[0]} vs {color.shape[0]} in {path}"
        )
    if not np.isfinite(coord_scale) or coord_scale == 0:
        raise ValueError(f"pth_coord_scale must be a non-zero number, got {coord_scale}")

    return {
        "coord": coord.astype(np.float32) * np.float32(coord_scale),
        "color": _coerce_color_to_0_255(
            color, assume=_stpls3d_color_convention(color), path=path
        ),
    }


def load_dales_npy(path: str | Path) -> dict[str, np.ndarray]:
    """Read a DALES style ``.npy`` array, keeping its first three columns."""

    path = Path(path)
    try:
        array = np.load(path, allow_pickle=False)
    except Exception as exc:  # noqa: BLE001
        raise PointCloudLoadError(f"Could not read .npy file {path}: {exc}") from exc
    if array.ndim != 2 or array.shape[1] < 3:
        raise PointCloudLoadError(
            f"DALES .npy must have shape (N, >=3), got {array.shape} in {path}"
        )
    return {"coord": array[:, :3]}


def load_npy_directory(path: str | Path) -> dict[str, np.ndarray]:
    """Read a ``coord.npy`` / ``color.npy`` / ``normal.npy`` directory."""

    path = Path(path)
    coord_path = path / "coord.npy"
    if not coord_path.exists():
        available = sorted(p.name for p in path.glob("*.npy"))
        raise PointCloudLoadError(
            f"Directory {path} does not contain coord.npy "
            f"(found npy files: {available or 'none'})."
        )

    result: dict[str, np.ndarray] = {"coord": _np_load(coord_path)}

    color_path = path / "color.npy"
    if color_path.exists():
        result["color"] = _coerce_color_to_0_255(
            _np_load(color_path), assume=None, path=color_path
        )

    normal_path = path / "normal.npy"
    if normal_path.exists():
        result["normal"] = _np_load(normal_path)

    return result


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------
def _np_load(path: Path) -> np.ndarray:
    try:
        return np.load(path, allow_pickle=False)
    except Exception as exc:  # noqa: BLE001
        raise PointCloudLoadError(f"Could not read numpy array {path}: {exc}") from exc


def _load_open3d(path: Path) -> dict[str, np.ndarray]:
    try:
        import open3d as o3d
    except ImportError as exc:  # pragma: no cover - depends on the environment
        raise PointCloudLoadError(
            f"Loading {path.name} requires open3d, which is not installed."
        ) from exc

    point_cloud = o3d.io.read_point_cloud(str(path))
    points = np.asarray(point_cloud.points)
    if points.size == 0:
        raise PointCloudLoadError(
            f"Open3D could not read any point from {path}. "
            "The file may be empty, truncated or not a valid point cloud."
        )

    result: dict[str, np.ndarray] = {"coord": points}
    colors = np.asarray(point_cloud.colors)
    if colors.size:
        result["color"] = _coerce_color_to_0_255(colors, assume="unit", path=path)
    normals = np.asarray(point_cloud.normals)
    if normals.size:
        result["normal"] = normals
    return result


def _to_numpy(value: Any) -> np.ndarray:
    if hasattr(value, "detach"):  # torch tensor
        value = value.detach().cpu().numpy()
    array = np.asarray(value)
    if array.dtype == object:
        # e.g. a list of tensors; let numpy raise a useful conversion error.
        array = np.stack([_to_numpy(item) for item in value])
    return array


def _torch_load(path: Path, *, trust_pickle: bool) -> Any:
    try:
        import torch
    except ImportError as exc:  # pragma: no cover - depends on the environment
        raise PointCloudLoadError(
            f"Loading {path.name} requires torch, which is not installed."
        ) from exc

    # Prefer the restricted unpickler when the installed torch supports it.
    try:
        return torch.load(str(path), map_location="cpu", weights_only=True)
    except TypeError:
        # torch < 2.0 has no weights_only argument.
        if not trust_pickle:
            raise PointCloudLoadError(
                f"Refusing to unpickle {path}: this torch version has no "
                "weights_only support and trust_pickle=False. .pth files can "
                "execute arbitrary code when loaded."
            ) from None
        warnings.warn(
            f"{path.name}: torch {torch.__version__} has no weights_only "
            "support; falling back to an unsafe torch.load(). Only load trusted "
            ".pth files.",
            RuntimeWarning,
            stacklevel=3,
        )
    except Exception as safe_error:  # noqa: BLE001 - any restriction failure
        if not trust_pickle:
            raise PointCloudLoadError(
                f"Refusing unsafe unpickle of {path}: torch.load(weights_only=True) "
                f"failed ({safe_error}) and trust_pickle=False."
            ) from safe_error
        warnings.warn(
            f"{path.name}: torch.load(weights_only=True) failed ({safe_error}); "
            "falling back to an unsafe torch.load(). Only load trusted .pth files.",
            RuntimeWarning,
            stacklevel=3,
        )

    try:
        try:
            # Be explicit on torch >= 2.x so the load does not depend on the
            # version specific default of ``weights_only``.
            return torch.load(str(path), map_location="cpu", weights_only=False)
        except TypeError:
            return torch.load(str(path), map_location="cpu")
    except Exception as exc:  # noqa: BLE001
        raise PointCloudLoadError(f"Could not load torch file {path}: {exc}") from exc


def _stpls3d_color_convention(color: Any) -> str:
    """Guess whether STPLS3D colours are ``[-1, 1]`` or raw bytes.

    STPLS3D pre-processing stores colours in ``[-1, 1]`` and the demo examples
    follow that convention.  Some third-party exports store raw ``0..255``
    bytes instead; values above 1 unambiguously identify that case.
    """

    array = np.asarray(color, dtype=np.float64)
    finite = array[np.isfinite(array)]
    if finite.size == 0:
        return "byte"
    if float(finite.min()) < -_COLOR_EPS:
        return "signed_unit"
    if float(finite.max()) > 1.0 + _COLOR_EPS:
        return "byte"
    # Ambiguous (all values in 0..1): trust the documented STPLS3D convention.
    return "signed_unit"


def _coerce_color_to_0_255(
    color: Any, *, assume: str | None, path: Path
) -> np.ndarray:
    """Return ``(N, 3)`` float32 RGB clipped to ``0..255``.

    ``assume`` documents the source convention: ``"byte"`` (0..255),
    ``"unit"`` (0..1), ``"signed_unit"`` (-1..1) or ``None`` for auto-detection.
    """

    array = _to_numpy(color)
    if array.ndim != 2 or array.shape[1] not in (3, 4):
        raise PointCloudLoadError(
            f"Color array must have shape (N, 3), got {array.shape} in {path}"
        )
    if array.dtype == object:
        raise PointCloudLoadError(f"Color array has object dtype in {path}")
    if array.shape[1] == 4:
        warnings.warn(
            f"{path.name}: dropping the alpha channel from RGBA colors.",
            RuntimeWarning,
            stacklevel=3,
        )
        array = array[:, :3]

    array = array.astype(np.float32)
    finite = np.isfinite(array)
    if not finite.all():
        warnings.warn(
            f"{path.name}: replacing {int((~finite).sum())} non-finite color "
            "value(s) with 0.",
            RuntimeWarning,
            stacklevel=3,
        )
        array = np.where(finite, array, 0.0)

    if assume == "unit":
        array = array * 255.0
    elif assume == "signed_unit":
        array = (array + 1.0) * 127.5
    elif assume is None:
        maximum = float(array.max()) if array.size else 0.0
        minimum = float(array.min()) if array.size else 0.0
        if minimum >= -_COLOR_EPS and maximum <= 1.0 + _COLOR_EPS:
            # Ambiguous 0..1 and all-black 0..255: follow Open3D/demo history.
            array = array * 255.0
        elif minimum >= -1.0 - _COLOR_EPS and maximum <= 1.0 + _COLOR_EPS:
            array = (array + 1.0) * 127.5
    elif assume != "byte":
        raise ValueError(f"Unknown color convention {assume!r}")

    if array.size:
        out_of_range = (array < -_COLOR_EPS) | (array > 255.0 + _COLOR_EPS)
        if out_of_range.any():
            warnings.warn(
                f"{path.name}: clipping {int(out_of_range.sum())} color value(s) "
                "outside 0..255.",
                RuntimeWarning,
                stacklevel=3,
            )
    return np.clip(array, 0.0, 255.0)


def _finalize(
    result: Mapping[str, Any], *, path: Path, on_nonfinite: str
) -> dict[str, np.ndarray]:
    if "coord" not in result or result["coord"] is None:
        raise PointCloudLoadError(f"No coordinates were decoded from {path}")

    coord = np.asarray(result["coord"], dtype=np.float32)
    if coord.ndim != 2 or coord.shape[1] != 3:
        raise PointCloudLoadError(
            f"Coordinates must have shape (N, 3), got {coord.shape} in {path}"
        )
    if coord.shape[0] == 0:
        raise PointCloudLoadError(f"Point cloud {path} contains no points")

    finite_rows = np.isfinite(coord).all(axis=1)
    original_num_points = coord.shape[0]
    num_nonfinite = int((~finite_rows).sum())
    if num_nonfinite:
        if on_nonfinite == "error":
            raise PointCloudLoadError(
                f"{path} contains {num_nonfinite} point(s) with non-finite "
                "coordinates. Pass on_nonfinite='drop' to skip them."
            )
        warnings.warn(
            f"{path.name}: dropping {num_nonfinite} point(s) with non-finite "
            "coordinates.",
            RuntimeWarning,
            stacklevel=3,
        )
        coord = coord[finite_rows]

    output: dict[str, np.ndarray] = {"coord": np.ascontiguousarray(coord)}

    for key in ("color", "normal", "intensity"):
        if result.get(key) is None:
            continue
        array = np.asarray(result[key], dtype=np.float32)
        if array.ndim == 1 and key == "intensity":
            array = array.reshape(-1, 1)
        if array.ndim != 2 or array.shape[1] != (1 if key == "intensity" else 3):
            raise PointCloudLoadError(
                f"{key} must have shape (N, {1 if key == 'intensity' else 3}), "
                f"got {array.shape} in {path}"
            )
        # Compare against the pre-drop point count: an array that matches the
        # file is valid even when non-finite rows are dropped below.
        if array.shape[0] != original_num_points:
            raise PointCloudLoadError(
                f"{key} has {array.shape[0]} entries but coord has "
                f"{original_num_points} in {path}"
            )
        if num_nonfinite:
            array = array[finite_rows]

        finite = np.isfinite(array)
        if not finite.all():
            warnings.warn(
                f"{path.name}: replacing {int((~finite).sum())} non-finite "
                f"{key} value(s) with 0.",
                RuntimeWarning,
                stacklevel=3,
            )
            array = np.where(finite, array, 0.0)
        output[key] = np.ascontiguousarray(array)

    return output
