#!/usr/bin/env python3
"""Outdoor KITTI smoke test through the public ``python_demo`` API.

This complements ``tools/verify_gpu_inference.py`` (which uses labelled ScanNet
data) by exercising the same classes the demo/GUI use: point-cloud loading,
DemoDatset preprocessing, SNAP checkpoint loading, CUDA backbone and prompt
segmentation.  ``data_examples/KITTI`` contains no ground-truth labels, so the
reported IoU is the model's IoU-token prediction, not a measured GT accuracy.

Example:
    pixi run python tools/verify_kitti_public_api.py
"""

from __future__ import annotations

import argparse
import json
import sys
import traceback
from pathlib import Path

import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

# The public demo classes live at the repository root.
from python_demo import PointCloudData, SegmentationModel  # noqa: E402


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", default="checkpoints/SNAP_C.pth")
    parser.add_argument("--data", default="data_examples/KITTI/000000.bin")
    parser.add_argument("--domain", default="Outdoor")
    parser.add_argument("--grid-size", type=float, default=0.05)
    parser.add_argument(
        "--prompt",
        default="9.4854517,7.34119511,-0.40044212",
        help="x,y,z prompt in the input point-cloud coordinate system",
    )
    parser.add_argument("--output", default=None)
    args = parser.parse_args()

    checkpoint = Path(args.checkpoint)
    if not checkpoint.is_absolute():
        checkpoint = REPO_ROOT / checkpoint
    if not checkpoint.exists():
        raise FileNotFoundError(f"checkpoint not found: {checkpoint}")

    report = {
        "nonce": "gpu-20261108-01",
        "status": "FAIL",
        "checkpoint": str(checkpoint),
        "data": str(args.data),
        "domain": args.domain,
        "grid_size": args.grid_size,
        "prompt": [float(v) for v in args.prompt.split(",")],
        "ground_truth": None,
        "iou_note": (
            "data_examples/KITTI has no labels; the IoU value is the model's "
            "IoU-token prediction, not a measured ground-truth IoU"
        ),
        "failures": [],
    }

    pcd = PointCloudData()
    raw = pcd.load_point_cloud(str(REPO_ROOT / args.data) if not Path(args.data).is_absolute() else args.data)
    report["raw_points"] = int(raw["coord"].shape[0])

    model = SegmentationModel(
        checkpoint_path=str(checkpoint), domain=args.domain, grid_size=args.grid_size
    )
    data = model.intialize_pointcloud(raw)
    report["processed_points"] = int(data["coord"].shape[0])
    report["offset"] = [int(v) for v in data["offset"].tolist()]

    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
    model.extract_backbone_features(data)
    feature = model.point_features["feat"]
    report["backbone"] = {
        "feature_shape": list(feature.shape),
        "feature_finite": bool(torch.isfinite(feature).all().item()),
        "device": str(feature.device),
        "mid_points": int(model.mid_points_features["coord"].shape[0]),
    }
    if not report["backbone"]["feature_finite"]:
        report["failures"].append("backbone produced non-finite features")

    prompt_points = [[report["prompt"]]]
    masks, text_labels, iou_scores = model.segment(data, prompt_points, text_prompt=None)
    report["segmentation"] = {
        "masks": len(masks),
        "labels": [str(label) for label in text_labels],
        "predicted_iou": [
            round(float(np.asarray(score).reshape(-1)[0]), 4) for score in iou_scores
        ],
        "mask_points": [int(np.asarray(mask).sum()) for mask in masks],
        "mask_fraction": [
            round(float(np.asarray(mask).mean()), 4) for mask in masks
        ],
    }
    if len(masks) != 1 or not report["segmentation"]["mask_points"][0]:
        report["failures"].append("prompt did not produce a non-empty mask")

    if torch.cuda.is_available():
        report["memory"] = {
            "max_allocated_mib": round(torch.cuda.max_memory_allocated() / 2**20, 1),
            "max_reserved_mib": round(torch.cuda.max_memory_reserved() / 2**20, 1),
        }

    report["status"] = "PASS" if not report["failures"] else "FAIL"
    text = json.dumps(report, indent=2, ensure_ascii=False)
    print(text)
    if args.output:
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(text + "\n")
    return 0 if report["status"] == "PASS" else 1


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except SystemExit:
        raise
    except Exception:
        traceback.print_exc()
        raise SystemExit(2)
