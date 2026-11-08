#!/usr/bin/env python3
"""Regression test for export metadata consistency.

Covers the bug found in the real UI export:
  * mask labels are offset (part 1 -> label 101) but ``objects[].id`` is 1
  * background clicks were serialized as ``is_positive: true``
  * metadata object colour was computed from id 1 while the PLY used label 101

Run with the locked env:
    env/.pixi/envs/default/bin/python scripts/test_export_metadata.py
"""
from __future__ import annotations

import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import open3d as o3d
import torch

WS = Path(__file__).resolve().parents[1]
BACKEND = WS / "interSeg3D-Studio" / "src" / "backend"
sys.path.insert(0, str(BACKEND))

from core.utils.app_utils import (  # noqa: E402
    create_colored_ply,
    generate_metadata_json,
    get_obj_color,
)
from infrastructure.inference.engine import Click  # noqa: E402


@dataclass
class FakeClick:
    obj_idx: int
    part_idx: int

    def to_dict(self) -> dict:
        return {
            "position": [0.0, 0.0, 0.0],
            "obj_idx": self.obj_idx,
            "part_idx": self.part_idx,
            "obj_name": "background" if self.obj_idx == 0 else "object_1",
            "time_idx": 0,
            "is_positive": self.obj_idx != 0,
            "id": 1,
            "cube_size": 0.02,
        }


@dataclass
class FakeHandler:
    clicks: list


@dataclass
class FakeInference:
    click_handler: FakeHandler


def main() -> int:
    mask = np.zeros(10, dtype=np.int32)
    mask[:3] = 101
    object_info = [
        {
            "obj_id": 1,
            "label": "chair",
            "description": "",
            "type": "object",
            "position": [],
        }
    ]
    clicks = [FakeClick(obj_idx=1, part_idx=1), FakeClick(obj_idx=0, part_idx=0)]

    with tempfile.TemporaryDirectory() as td:
        metadata = generate_metadata_json(
            mask=mask,
            new_ply_path=str(Path(td) / "scene_with_colored_objects.ply"),
            original_file_path="scene0025_00.ply",
            object_info=object_info,
            inference_obj=FakeInference(FakeHandler(clicks)),
            get_obj_color_func=get_obj_color,
        )

        obj = metadata["objects"][0]
        assert obj["id"] == 1, obj
        assert obj["part_id"] == 1, obj
        assert obj["mask_label"] == 101, obj
        assert obj["points"] == 3, obj
        assert obj["color"] == list(get_obj_color(101, normalize=True)), obj
        assert metadata["object_counts"] == {1: 3}, metadata["object_counts"]
        assert metadata["mask_label_counts"] == {"101": 3}, metadata["mask_label_counts"]
        assert metadata["object_part_counts"] == [
            {"object_id": 1, "part_id": 1, "mask_label": 101, "points": 3}
        ], metadata["object_part_counts"]
        assert metadata["click_data"][0]["is_positive"] is True
        assert metadata["click_data"][0]["part_idx"] == 1
        assert metadata["click_data"][1]["is_positive"] is False, metadata["click_data"][1]
        assert metadata["click_data"][1]["part_idx"] == 0

        # The exported PLY must use the same colour as metadata (mask label 101).
        coords = np.zeros((10, 3), dtype=np.float64)
        coords[:, 0] = np.arange(10)
        ply_path = Path(td) / "scene_with_colored_objects.ply"
        create_colored_ply(
            coords=coords,
            colors=np.zeros((10, 3)),
            mask=mask,
            is_point_cloud=True,
            original_geometry_path="unused.ply",
            output_path=str(ply_path),
            get_obj_color_func=get_obj_color,
        )
        pc = o3d.io.read_point_cloud(str(ply_path))
        colors = np.asarray(pc.colors)
        # PLY stores uchar RGB, so compare within one 8-bit step.
        assert np.allclose(colors[0], obj["color"], atol=1.0 / 255.0 + 1e-6), (colors[0], obj["color"])
        assert np.allclose(colors[3], [0.7, 0.7, 0.7], atol=1.0 / 255.0 + 1e-6), colors[3]

        # Without recognition, ids are still canonical object ids (101 -> 1),
        # while the raw mask label remains available for diagnostics.
        bare = generate_metadata_json(
            mask=mask,
            new_ply_path=str(Path(td) / "bare.ply"),
            original_file_path="scene0025_00.ply",
            object_info=[],
            inference_obj=None,
            get_obj_color_func=get_obj_color,
        )
        assert bare["objects"][0]["id"] == 1, bare["objects"][0]
        assert bare["objects"][0]["mask_label"] == 101
        assert bare["object_counts"] == {1: 3}, bare["object_counts"]

    # Engine click serialization keeps the true part index.
    click = Click(
        position=torch.zeros(3, dtype=torch.float32),
        obj_idx=0,
        part_idx=0,
        obj_name="background",
        time_idx=0,
        is_positive=False,
        id=7,
        cube_size=0.02,
    )
    raw = click.to_dict()
    assert raw["part_idx"] == 0, raw
    assert raw["is_positive"] is False, raw

    print("PASS: export metadata ids, colours, raw labels and negative-click flags are consistent")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
