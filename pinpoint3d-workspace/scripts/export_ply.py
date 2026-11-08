#!/usr/bin/env python3
"""Write a colored PLY for one run: grey scene, red = predicted foreground."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np


def write_ply(path: Path, xyz: np.ndarray, rgb: np.ndarray) -> None:
    colors = np.clip(rgb * 255.0, 0, 255).astype(np.uint8)
    with path.open("w") as f:
        f.write("ply\nformat ascii 1.0\n")
        f.write(f"element vertex {len(xyz)}\n")
        f.write("property float x\nproperty float y\nproperty float z\n")
        f.write("property uchar red\nproperty uchar green\nproperty uchar blue\n")
        f.write("end_header\n")
        np.savetxt(f, np.hstack([xyz, colors]), fmt="%.6f %.6f %.6f %d %d %d")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True)
    ap.add_argument("--data", required=True)
    ap.add_argument("--out", required=True)
    ns = ap.parse_args()

    z = np.load(Path(ns.run) / "masks.npz")
    coord = np.load(Path(ns.data) / "coord.npy").astype(np.float32)
    pred = z["pred_full"] > 0
    rgb = np.full((len(coord), 3), 0.72, dtype=np.float32)
    rgb[pred] = [0.85, 0.10, 0.10]
    write_ply(Path(ns.out), coord, rgb)
    print(f"wrote {ns.out} ({pred.sum()} red points)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
