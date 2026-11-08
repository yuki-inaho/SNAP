#!/usr/bin/env python3
"""Render a before/after comparison of raw PinPoint3D masks (matplotlib, Agg).

Left column: dataset instance mask (diagnostic only).
Then one column per run (pos-only, pos+neg, ...), same viewpoint.
Row 1: top view (x-y).  Row 2: side view (x-z).
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", nargs="+", required=True)
    ap.add_argument("--titles", nargs="+", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--bg-points", type=int, default=60000)
    ap.add_argument("--dpi", type=int, default=150)
    ns = ap.parse_args()
    assert len(ns.runs) == len(ns.titles)

    panels = []
    for run in ns.runs:
        d = Path(run)
        z = np.load(d / "masks.npz")
        info = json.loads((d / "run_info.json").read_text())
        data_dir = Path(info["data"])
        prompts = json.loads((d / "prompts.json").read_text())
        panels.append({
            "pred": z["pred_full"],
            "gt": z["gt_instance"].astype(bool),
            "instance": z["instance"],
            "prompts": prompts,
            "metrics": json.loads((d / "metrics.json").read_text()),
        })

    # per-point coordinates (the masks are per input point, not per voxel)
    data_dir = Path(json.loads((Path(ns.runs[0]) / "run_info.json").read_text())["data"])
    xyz = np.load(data_dir / "coord.npy").astype(np.float32)
    rng = np.random.default_rng(0)
    bg_idx = rng.choice(len(xyz), size=min(ns.bg_points, len(xyz)), replace=False)
    bg = xyz[bg_idx]

    fig, axes = plt.subplots(2, 1 + len(panels), figsize=(5.2 * (1 + len(panels)), 9.5))
    cols = ["dataset instance 13 / seg20 class 4 (diagnostic)"] + ns.titles

    for c, title in enumerate(cols):
        for r, (ax, view) in enumerate(zip(axes[:, c], ["xy", "xz"])):
            ax.scatter(bg[:, 0], bg[:, 1 if view == "xy" else 2], s=0.15, c="#c9c9c9", alpha=0.35,
                       linewidths=0, rasterized=True)
            if c == 0:
                gt = panels[0]["gt"]
                g = xyz[gt]
                ax.scatter(g[:, 0], g[:, 1 if view == "xy" else 2], s=0.6, c="#2ca02c",
                           linewidths=0, rasterized=True, label="dataset mask")
            else:
                p = panels[c - 1]
                gt = p["gt"]
                pred = p["pred"] > 0
                g = xyz[gt & ~pred]
                o = xyz[gt & pred]
                x = xyz[pred & ~gt]
                ax.scatter(g[:, 0], g[:, 1 if view == "xy" else 2], s=0.6, c="#2ca02c",
                           linewidths=0, rasterized=True, label="dataset only")
                ax.scatter(o[:, 0], o[:, 1 if view == "xy" else 2], s=0.6, c="#7f4fbf",
                           linewidths=0, rasterized=True, label="prediction ∩ dataset")
                ax.scatter(x[:, 0], x[:, 1 if view == "xy" else 2], s=0.6, c="#d62728",
                           linewidths=0, rasterized=True, label="prediction only")
                for pr in p["prompts"]:
                    vx, vy, vz = pr["voxel_xyz"]
                    ax.scatter([vx], [vy if view == "xy" else vz], marker="*" if pr["is_positive"] else "X",
                               s=180 if pr["is_positive"] else 120,
                               c="#1f77b4" if pr["is_positive"] else "#ff7f0e",
                               edgecolors="k", linewidths=0.5, zorder=5)
                m = p["metrics"]
                ax.text(0.02, 0.02,
                        f"pred pts {m['pred_points_foreground']:,}\n"
                        f"IoU(GT) {m['iou_vs_gt_instance']:.3f}  "
                        f"P {m['precision_vs_gt_instance']:.3f}  R {m['recall_vs_gt_instance']:.3f}",
                        transform=ax.transAxes, fontsize=8.5, va="bottom",
                        bbox=dict(fc="white", ec="#999999", alpha=0.85))
            ax.set_title(f"{title}\n{'top view (x-y)' if view == 'xy' else 'side view (x-z)'}", fontsize=9)
            ax.set_aspect("equal")
            ax.set_xlabel("x [m]")
            ax.set_ylabel("y [m]" if view == "xy" else "z [m]")
            ax.tick_params(labelsize=7)
        axes[0, c].legend(markerscale=6, fontsize=7, loc="upper right", framealpha=0.9)

    fig.suptitle("PinPoint3D native clicks on ScanNet scene0025_00 chair (instance 13) — star = positive click, X = negative click",
                 fontsize=11)
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    fig.savefig(ns.out, dpi=ns.dpi)
    print(f"wrote {ns.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
