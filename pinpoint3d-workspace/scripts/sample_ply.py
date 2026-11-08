#!/usr/bin/env python3
"""Convert the bundled ScanNet example to a coloured PLY for Studio."""
import argparse
from pathlib import Path

import numpy as np

from export_ply import write_ply

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--data", type=Path, required=True)
parser.add_argument("--out", type=Path, required=True)
args = parser.parse_args()
xyz = np.load(args.data / "coord.npy")
rgb = np.load(args.data / "color.npy").astype(np.float32)
if rgb.max() > 1:
    rgb /= 255.0
args.out.parent.mkdir(parents=True, exist_ok=True)
write_ply(args.out, xyz, rgb)
