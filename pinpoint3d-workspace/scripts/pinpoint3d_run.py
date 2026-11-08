#!/usr/bin/env python3
"""Run PinPoint3D with native positive *and* negative clicks on one ScanNet chair.

Negative clicks are not a bolt-on: PinPoint3D's mask decoder takes
``click_idx['0']`` as background (negative) click indices and fuses them with the
learned background queries (models/pinpoint3d.py, forward_mask).  This script
feeds that native path, using the same preprocessing as the official
interSeg3D-Studio backend (ME.sparse_quantize -> SparseTensor -> forward_backbone).

Everything is derived from the official artefacts:
  * code:      https://github.com/Quit123/PinPoint3D (commit recorded in README)
  * weights:   Google Drive id 1Rg2JDjh8iFGKwzP0UMLBCkce7bCvO5-D
  * reference: https://github.com/zh-plus/interSeg3D-Studio (backend/inference.py)

Outputs are raw model masks (no click forcing, no cube postprocess) so the
positive/negative effect is measurable.  Dataset labels are only used as a
diagnostic.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch

WORKSPACE = Path(__file__).resolve().parents[1]
REPO = WORKSPACE / "PinPoint3D"
sys.path.insert(0, str(REPO))

import MinkowskiEngine as ME  # noqa: E402

from models import build_model  # noqa: E402


def log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def load_cloud(data_dir: Path):
    coord = np.ascontiguousarray(np.load(data_dir / "coord.npy").astype(np.float32))
    color = np.load(data_dir / "color.npy")
    if color.dtype == np.uint8:
        color = color.astype(np.float32) / 255.0
    else:
        color = color.astype(np.float32)
        if color.max() > 1.0:
            color /= 255.0
    instance = np.load(data_dir / "instance.npy").astype(np.int64)
    seg20 = np.load(data_dir / "segment20.npy").astype(np.int64)
    return coord, color, instance, seg20


def write_ply(path: Path, xyz: np.ndarray, rgb: np.ndarray) -> None:
    """Minimal binary-free PLY so the cloud can be re-opened by open3d/Studio."""
    colors = np.clip(rgb * 255.0, 0, 255).astype(np.uint8)
    with path.open("w") as f:
        f.write("ply\nformat ascii 1.0\n")
        f.write(f"element vertex {len(xyz)}\n")
        f.write("property float x\nproperty float y\nproperty float z\n")
        f.write("property uchar red\nproperty uchar green\nproperty uchar blue\n")
        f.write("end_header\n")
        np.savetxt(f, np.hstack([xyz, colors]), fmt="%.6f %.6f %.6f %d %d %d")


def build_config(args, device):
    cfg = type("cfg", (), {})()
    for key, value in vars(args).items():
        setattr(cfg, key, value)
    cfg.model_type = "pinpoint3D"
    cfg.device = device
    return cfg


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True, help="directory with coord/color/instance npy")
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--instance", type=int, required=True, help="ScanNet instance id of the target chair")
    ap.add_argument("--prompts", required=True, help="JSON list of {position:[x,y,z], is_positive:bool}")
    ap.add_argument("--out", required=True)
    ap.add_argument("--voxel-size", type=float, default=0.05)
    args = ap.parse_args()

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    data_dir = Path(args.data)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    log(f"device={device} ({torch.cuda.get_device_name(0) if device.type == 'cuda' else 'cpu'}) "
        f"sm{''.join(map(str, torch.cuda.get_device_capability(0)))} torch={torch.__version__} cuda={torch.version.cuda}")

    ckpt = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    cfg = build_config(ckpt["args"], str(device))
    cfg.voxel_size = args.voxel_size
    log(f"checkpoint epoch={ckpt.get('epoch')} model_type(cfg)={cfg.model_type} voxel={cfg.voxel_size}")

    model = build_model(cfg).to(device).eval()
    missing, unexpected = model.load_state_dict(ckpt["model"], strict=False)
    missing = [k for k in missing if not k.endswith(("total_params", "total_ops"))]
    unexpected = [k for k in unexpected if not k.endswith(("total_params", "total_ops"))]
    if missing:
        raise RuntimeError(f"missing keys: {missing[:5]} (n={len(missing)})")
    log(f"weights loaded: {args.checkpoint} (unexpected keys: {len(unexpected)})")
    log(f"params: {sum(p.numel() for p in model.parameters()):,}")

    xyz, rgb, instance, seg20 = load_cloud(data_dir)
    log(f"cloud: {xyz.shape[0]} points, target instance {args.instance} "
        f"({int((instance == args.instance).sum())} points, seg20 class {np.unique(seg20[instance == args.instance]).tolist()})")

    t0 = time.time()
    coords_qv, unique_map, inverse_map = ME.utils.sparse_quantize(
        coordinates=xyz, quantization_size=args.voxel_size, return_index=True, return_inverse=True
    )
    unique_np = unique_map.cpu().numpy()
    raw_qv = torch.from_numpy(xyz[unique_np]).float().to(device)
    feats = torch.from_numpy(rgb[unique_np]).float().to(device)
    qv = coords_qv.int()
    log(f"voxelized {xyz.shape[0]} -> {len(coords_qv)} voxels ({time.time()-t0:.1f}s)")

    data = ME.SparseTensor(coordinates=ME.utils.batched_coordinates([qv]), features=feats, device=device)
    with torch.no_grad():
        t0 = time.time()
        pcd_features, aux, coordinates, pos_encodings = model.forward_backbone(data, raw_coordinates=raw_qv)
        torch.cuda.synchronize()
        log(f"backbone done in {time.time()-t0:.1f}s, peak {torch.cuda.max_memory_allocated()/2**20:.0f} MiB")

    prompts = json.loads(Path(args.prompts).read_text())
    pos_ids, neg_ids = [], []
    resolved = []
    for p in prompts:
        pos = torch.tensor(p["position"], dtype=torch.float32, device=device)
        dist = torch.cdist(raw_qv, pos.unsqueeze(0), p=2)
        idx = int(dist.argmin().item())
        (pos_ids if p["is_positive"] else neg_ids).append(idx)
        resolved.append({**p, "voxel_index": idx, "voxel_xyz": raw_qv[idx].cpu().tolist(),
                         "snap_to_click": float(dist[idx].item())})
        log(f"prompt {'POS' if p['is_positive'] else 'NEG'} {p['position']} -> voxel {idx} {raw_qv[idx].tolist()} (d={dist[idx].item():.4f})")

    if not pos_ids:
        raise SystemExit("need at least one positive click")

    # Native prompt layout of PinPoint3D: '0' == background (negative) clicks,
    # {'1': {'1': [...]}} == part 1 of the target object 1.
    click_idx = [{"0": neg_ids, "1": {"1": pos_ids}}]
    click_time_idx = [{"0": list(range(len(neg_ids))), "1": {"1": list(range(len(neg_ids), len(neg_ids) + len(pos_ids)))}}]
    log(f"click_idx={click_idx}")

    with torch.no_grad():
        t0 = time.time()
        outputs = model.forward_mask(
            pcd_features, aux, coordinates, pos_encodings,
            click_idx=click_idx, click_time_idx=click_time_idx, target_object_id=[1],
        )
        torch.cuda.synchronize()
        dt = time.time() - t0
    if outputs == "ignore this test":
        raise SystemExit("model returned 'ignore this test' (bad prompt layout)")
    log(f"mask decoder done in {dt:.1f}s")

    part_logits = outputs["part_predictions_mask"][0]  # (V, P+1)
    pred_vox = part_logits.argmax(-1).cpu().numpy().astype(np.int16)
    pred_full = pred_vox[inverse_map.cpu().numpy()].astype(np.int16)
    gt = (instance == args.instance).astype(np.uint8)
    positive = (pred_full > 0).astype(np.uint8)

    inter = int((positive & gt).sum())
    union = int((positive | gt).sum())
    metrics = {
        "prompt_count": {"positive": len(pos_ids), "negative": len(neg_ids)},
        "pred_voxels_foreground": int((pred_vox > 0).sum()),
        "pred_points_foreground": int(positive.sum()),
        "gt_points": int(gt.sum()),
        "iou_vs_gt_instance": inter / union if union else 0.0,
        "precision_vs_gt_instance": inter / int(positive.sum()) if positive.sum() else 0.0,
        "recall_vs_gt_instance": inter / int(gt.sum()) if gt.sum() else 0.0,
        "unique_pred_labels": np.unique(pred_vox).tolist(),
        "decoder_seconds": dt,
        "peak_mib": torch.cuda.max_memory_allocated() / 2**20,
    }

    np.savez_compressed(
        out_dir / "masks.npz",
        pred_full=pred_full,
        pred_vox=pred_vox,
        gt_instance=gt,
        click_pos_ids=np.array(pos_ids, dtype=np.int64),
        click_neg_ids=np.array(neg_ids, dtype=np.int64),
        voxel_xyz=raw_qv.cpu().numpy(),
        instance=instance,
        seg20=seg20,
    )
    (out_dir / "metrics.json").write_text(json.dumps(metrics, indent=2))
    (out_dir / "prompts.json").write_text(json.dumps(resolved, indent=2))
    (out_dir / "run_info.json").write_text(json.dumps({
        "data": str(data_dir),
        "checkpoint": str(Path(args.checkpoint).resolve()),
        "checkpoint_epoch": ckpt.get("epoch"),
        "voxel_size": args.voxel_size,
        "device": str(device),
        "gpu": torch.cuda.get_device_name(0),
        "capability": torch.cuda.get_device_capability(0),
        "torch": torch.__version__,
        "cuda": torch.version.cuda,
        "points": int(xyz.shape[0]),
        "voxels": int(len(coords_qv)),
        "metrics": metrics,
    }, indent=2))

    log(f"metrics: {json.dumps(metrics)}")
    log(f"wrote {out_dir/'masks.npz'} and {out_dir/'metrics.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
