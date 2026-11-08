#!/usr/bin/env python3
"""Verify SNAP real-checkpoint inference on the local GPU.

This is a GPU/compatibility smoke test, not a test-suite replacement.  It:

* resolves the attention backend and reports why FlashAttention-2 is or is not
  used (RTX 2070 / Turing sm75 has no FlashAttention-2 kernels);
* builds ``SNAP`` with the released pretrained checkpoint (no random weights);
* preprocesses a real point cloud with the same transforms used by the demo;
* runs the CUDA backbone and the mask decoder with real prompt points;
* reports mask IoU against ground-truth classes and measured CUDA memory.

Example:
    pixi run python tools/verify_gpu_inference.py \
        --checkpoint checkpoints/SNAP_C.pth \
        --data data_examples/ScanNet/scene0011_00 \
        --attention-backend auto
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import math
import os
import sys
import time
import traceback
from collections import Counter
from pathlib import Path

import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.ptv3_backbone_domain import (  # noqa: E402
    SerializedAttention,
    SerializedPooling,
    get_flash_attention_status,
    resolve_attention_backend,
)
from src.snap import SNAP  # noqa: E402


def _load_transforms_module():
    """Load datasets/transforms.py without importing the datasets package."""
    path = REPO_ROOT / "datasets" / "transforms.py"
    spec = importlib.util.spec_from_file_location("_snap_verify_transforms", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _load_checkpoint(path: Path):
    try:
        checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:  # torch < 2.0
        checkpoint = torch.load(path, map_location="cpu")
    if isinstance(checkpoint, dict) and "model" in checkpoint:
        state_dict = checkpoint["model"]
    elif isinstance(checkpoint, dict) and "state_dict" in checkpoint:
        state_dict = checkpoint["state_dict"]
    else:
        state_dict = checkpoint
    if not isinstance(state_dict, dict):
        raise TypeError(f"checkpoint does not contain a state dict: {type(state_dict)}")
    cleaned = {}
    for key, value in state_dict.items():
        new_key = key[len("module."):] if key.startswith("module.") else key
        if new_key in ("seg_head.weight", "seg_head.bias"):
            continue
        cleaned[new_key] = value
    return cleaned


def _load_scan(args):
    data_dir = Path(args.data)
    if data_dir.is_dir():
        coord = np.load(data_dir / "coord.npy").astype(np.float32)
        color = np.load(data_dir / "color.npy").astype(np.float32)
        normal = np.load(data_dir / "normal.npy").astype(np.float32)
        segment_path = data_dir / args.segment_file
        if not segment_path.exists():
            raise FileNotFoundError(f"missing ground-truth labels: {segment_path}")
        segment = np.load(segment_path)
        instance_path = data_dir / "instance.npy"
        if instance_path.exists():
            instance = np.load(instance_path)
        else:
            instance = np.full(segment.shape, -1, dtype=np.int64)
    else:
        raise ValueError(
            "--data must be a ScanNet-style directory containing coord.npy, "
            f"color.npy, normal.npy, and {args.segment_file}; got {data_dir}"
        )

    strength = np.zeros((coord.shape[0], 1), dtype=np.float32)
    return coord, color, normal, strength, segment, instance


def _build_input(coord, color, normal, strength, segment, instance, args):
    transforms = _load_transforms_module()
    pipeline = [
        transforms.CenterShift(apply_z=True),
        transforms.GridSample(
            grid_size=args.grid_size,
            hash_type="fnv",
            mode="train",
            keys=("coord", "color", "strength", "normal", "segment20", "instance"),
            return_grid_coord=True,
        ),
        transforms.CenterShift(apply_z=False),
        transforms.NormalizeColor(),
        transforms.Add(keys_dict={"condition": "ScanNet", "domain": args.domain}),
        transforms.ToTensor(),
        transforms.Collect(
            keys=(
                "coord",
                "grid_coord",
                "color",
                "normal",
                "strength",
                "condition",
                "domain",
                "segment20",
                "instance",
            )
        ),
    ]
    data = {
        "coord": coord,
        "color": color,
        "normal": normal,
        "strength": strength,
        "segment20": segment,
        "instance": instance,
    }
    for transform in pipeline:
        data = transform(data)
    return data


def _select_prompt_classes(segment: np.ndarray, args):
    values, counts = np.unique(segment, return_counts=True)
    excluded = set(args.exclude_classes)
    candidates = [
        (int(v), int(c))
        for v, c in zip(values, counts)
        if int(v) >= 0 and int(v) not in excluded and int(c) >= args.min_class_points
    ]
    candidates.sort(key=lambda item: item[1], reverse=True)
    if args.prompt_classes:
        wanted = [int(v) for v in args.prompt_classes.split(",") if v.strip()]
        selected = [(v, int(np.sum(segment == v))) for v in wanted if np.any(segment == v)]
    else:
        selected = candidates[: args.max_prompts]
    if not selected:
        raise RuntimeError(
            "no prompt class with at least "
            f"{args.min_class_points} points found in {np.unique(segment).tolist()}"
        )
    return selected


def _prompt_point(data, cls):
    class_mask = data["segment20"] == cls
    if not bool(class_mask.any()):
        raise RuntimeError(f"class {cls} has no sampled points")

    # Prefer a single instance so the prompt mask is evaluated against the
    # object that was clicked rather than every object of the same class.
    instance = data["instance"]
    instance_ids = instance[class_mask]
    instance_ids = instance_ids[instance_ids > 0]
    if instance_ids.numel() > 0:
        values, counts = torch.unique(instance_ids, return_counts=True)
        instance_id = int(values[torch.argmax(counts)].item())
        mask = class_mask & (instance == instance_id)
    else:
        instance_id = None
        mask = class_mask

    coords = data["coord"][mask]
    centroid = coords.mean(dim=0)
    distance = torch.linalg.norm(coords - centroid, dim=1)
    index = int(torch.argmin(distance))
    return coords[index].unsqueeze(0).unsqueeze(0), mask, instance_id


def _memory_mib():
    if not torch.cuda.is_available():
        return {}
    return {
        "allocated_mib": round(torch.cuda.memory_allocated() / 2**20, 1),
        "reserved_mib": round(torch.cuda.memory_reserved() / 2**20, 1),
        "max_allocated_mib": round(torch.cuda.max_memory_allocated() / 2**20, 1),
        "max_reserved_mib": round(torch.cuda.max_memory_reserved() / 2**20, 1),
    }


def run(args):
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but torch.cuda.is_available() is False")

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()

    report = {
        "nonce": "gpu-20261108-01",
        "status": "FAIL",
        "command": " ".join(sys.argv),
        "environment": {
            "python": sys.version.split()[0],
            "torch": torch.__version__,
            "torch_cuda": torch.version.cuda,
            "cuda_available": torch.cuda.is_available(),
            "device": str(device),
            "device_name": None,
            "compute_capability": None,
        },
        "attention": {},
        "checkpoint": {},
        "data": {},
        "backbone": {},
        "segmentation": {},
        "memory": {},
        "failures": [],
    }

    if torch.cuda.is_available():
        index = torch.cuda.current_device()
        props = torch.cuda.get_device_properties(index)
        report["environment"]["device_name"] = props.name
        report["environment"]["compute_capability"] = f"{props.major}.{props.minor}"
        report["environment"]["total_memory_mib"] = round(props.total_memory / 2**20, 1)

    flash_available, flash_reason = get_flash_attention_status(device)
    resolved_backend = resolve_attention_backend(args.attention_backend, device)
    backend_checks = {
        "cpu_reports_flash_unavailable": get_flash_attention_status("cpu")[0] is False,
        "normal_forces_fallback": resolve_attention_backend("normal", device) == "normal",
        "auto_matches_status": resolved_backend
        == ("flash" if flash_available else "normal"),
    }
    if not flash_available:
        try:
            resolve_attention_backend("flash", device)
            backend_checks["flash_strict_rejected"] = False
            report.setdefault("failures", []).append(
                "explicit flash backend did not fail on an unsupported device"
            )
        except RuntimeError as exc:
            backend_checks["flash_strict_rejected"] = "FlashAttention-2" in str(exc)
    report["attention"] = {
        "flash_attn_available": flash_available,
        "reason": flash_reason,
        "requested_backend": args.attention_backend,
        "resolved_backend": resolved_backend,
        "checks": backend_checks,
    }

    checkpoint_path = Path(args.checkpoint)
    if not checkpoint_path.is_absolute():
        checkpoint_path = REPO_ROOT / checkpoint_path
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"checkpoint not found: {checkpoint_path}")
    state_dict = _load_checkpoint(checkpoint_path)
    report["checkpoint"] = {
        "path": str(checkpoint_path),
        "size_mib": round(checkpoint_path.stat().st_size / 2**20, 1),
        "tensors": len(state_dict),
    }

    model = SNAP(
        num_points=1,
        num_merge_blocks=1,
        use_pdnorm=True,
        return_mid_points=True,
        attention_backend=args.attention_backend,
        attention_device=args.device,
    ).to(device)
    model.eval()

    load_result = model.load_state_dict(state_dict, strict=False)
    model_state = model.state_dict()
    missing = list(load_result.missing_keys)
    unexpected = list(load_result.unexpected_keys)
    matched = len(set(model_state).intersection(state_dict))
    sample_keys = [k for k in model_state if k in state_dict][:5]
    sample_matches = {
        key: bool(torch.equal(model_state[key].cpu(), state_dict[key].cpu()))
        for key in sample_keys
    }
    report["checkpoint"].update(
        {
            "model_tensors": len(model_state),
            "matched_tensors": matched,
            "missing_keys": missing,
            "unexpected_keys": unexpected,
            "sample_weight_matches": sample_matches,
        }
    )
    if missing:
        report["failures"].append(
            f"checkpoint does not provide {len(missing)} model tensors: {missing[:10]}"
        )
    if not all(sample_matches.values()):
        report["failures"].append("sample checkpoint tensors differ from loaded model weights")

    attention_modules = [
        module for module in model.modules() if isinstance(module, SerializedAttention)
    ]
    report["attention"].update(
        {
            "model_backend": model.attention_backend,
            "backbone_backend": model.backbone.attention_backend,
            "attention_layers": len(attention_modules),
            "layers_using_flash": sum(m.enable_flash for m in attention_modules),
            "layers_using_normal": sum(not m.enable_flash for m in attention_modules),
            "parameter_devices": sorted({str(p.device) for p in model.parameters()}),
            "parameter_count": sum(p.numel() for p in model.parameters()),
        }
    )
    if report["attention"]["layers_using_flash"] and report["attention"]["layers_using_normal"]:
        report["failures"].append("attention backend is mixed across layers")
    expected_flash = model.attention_backend == "flash"
    if attention_modules and any(m.enable_flash != expected_flash for m in attention_modules):
        report["failures"].append(
            "attention layer configuration does not match the resolved backend"
        )

    coord, color, normal, strength, segment, instance = _load_scan(args)
    if args.max_points and coord.shape[0] > args.max_points:
        rng = np.random.default_rng(args.seed)
        indices = np.sort(rng.choice(coord.shape[0], args.max_points, replace=False))
        coord = coord[indices]
        color = color[indices]
        normal = normal[indices]
        strength = strength[indices]
        segment = segment[indices]
        instance = instance[indices]
    report["data"].update(
        {
            "path": str(Path(args.data)),
            "original_points": int(coord.shape[0]),
            "segment_file": args.segment_file,
            "has_instance_labels": bool((instance > 0).any()),
            "classes": {
                int(v): int(c)
                for v, c in zip(*np.unique(segment, return_counts=True))
            },
        }
    )
    data = _build_input(coord, color, normal, strength, segment, instance, args)
    report["data"]["sampled_points"] = int(data["coord"].shape[0])
    report["data"]["grid_size"] = args.grid_size
    report["data"]["domain"] = args.domain
    report["data"]["sampled_class_counts"] = {
        int(v): int(c)
        for v, c in zip(*np.unique(data["segment20"].numpy(), return_counts=True))
    }

    data = {
        key: value.to(device) if isinstance(value, torch.Tensor) else value
        for key, value in data.items()
    }

    if torch.cuda.is_available():
        torch.cuda.synchronize()
    start = time.perf_counter()
    with torch.no_grad():
        data = model.run_input_encoder(data)
        backbone_out = model.run_backbone(data)
        if model.backbone.return_mid_points:
            point, mid_points = backbone_out
            report["backbone"]["mid_points"] = int(mid_points["coord"].shape[0])
        else:
            point = backbone_out

        torch.cuda.synchronize() if torch.cuda.is_available() else None
        backbone_seconds = time.perf_counter() - start
        feat = point.feat
        finite = bool(torch.isfinite(feat).all().item())
        report["backbone"].update(
            {
                "seconds": round(backbone_seconds, 3),
                "input_points": int(data["coord"].shape[0]),
                "output_points": int(point.coord.shape[0]),
                "feature_shape": list(feat.shape),
                "feature_finite": finite,
                "feature_min": float(feat.min().item()),
                "feature_max": float(feat.max().item()),
                "feature_mean": float(feat.mean().item()),
                "output_device": str(feat.device),
            }
        )
        if not finite:
            report["failures"].append("backbone produced non-finite features")
        if point.coord.shape[0] != data["coord"].shape[0]:
            report["failures"].append("backbone output point count differs from input")

        if args.check_chunk_equivalence:
            old_budget = os.environ.get("SNAP_ATTN_CHUNK_BYTES")
            old_shuffle = model.backbone.shuffle_orders
            pooling_modules = [
                module
                for module in model.modules()
                if isinstance(module, SerializedPooling)
            ]
            old_pooling_shuffle = [module.shuffle_orders for module in pooling_modules]
            model.backbone.shuffle_orders = False
            for module in pooling_modules:
                module.shuffle_orders = False
            try:
                os.environ["SNAP_ATTN_CHUNK_BYTES"] = str(256 * 1024 * 1024)
                with torch.no_grad():
                    chunked_out = model.run_backbone(dict(data))
                os.environ["SNAP_ATTN_CHUNK_BYTES"] = str(10**12)
                with torch.no_grad():
                    unchunked_out = model.run_backbone(dict(data))
                if isinstance(chunked_out, tuple):
                    chunked_out = chunked_out[0]
                if isinstance(unchunked_out, tuple):
                    unchunked_out = unchunked_out[0]
                difference = (chunked_out.feat - unchunked_out.feat).abs()
                equivalence = {
                    "max_abs_diff": float(difference.max().item()),
                    "mean_abs_diff": float(difference.mean().item()),
                    "allclose_1e-4": bool(
                        torch.allclose(
                            chunked_out.feat, unchunked_out.feat, atol=1e-4, rtol=1e-4
                        )
                    ),
                }
            finally:
                model.backbone.shuffle_orders = old_shuffle
                for module, value in zip(pooling_modules, old_pooling_shuffle):
                    module.shuffle_orders = value
                if old_budget is None:
                    os.environ.pop("SNAP_ATTN_CHUNK_BYTES", None)
                else:
                    os.environ["SNAP_ATTN_CHUNK_BYTES"] = old_budget
            report["backbone"]["chunk_equivalence"] = equivalence
            if not equivalence["allclose_1e-4"]:
                report["failures"].append(
                    "chunked and unchunked attention differ by more than 1e-4"
                )
            del chunked_out, unchunked_out, difference

        selected = _select_prompt_classes(data["segment20"].cpu().numpy(), args)
        report["data"]["prompt_classes"] = [cls for cls, _ in selected]
        segmentation = []
        for cls, gt_count in selected:
            prompt, instance_mask, instance_id = _prompt_point(data, cls)
            prompt_data = dict(data)
            prompt_data["point"] = prompt.to(device)
            prompt_data["point_offset"] = [1]
            torch.cuda.synchronize() if torch.cuda.is_available() else None
            seg_start = time.perf_counter()
            (
                seg_logits_list,
                text_logits_list,
                iou_out_list,
                _aux,
                _weights,
                _confidence,
                _confidence_gt,
            ) = model.run_mask_decoder(point, prompt_data, val=True)
            torch.cuda.synchronize() if torch.cuda.is_available() else None
            seg_seconds = time.perf_counter() - seg_start

            logits = seg_logits_list[0]
            pred = logits.sigmoid() > model.mask_threshold
            gt = instance_mask
            class_mask = prompt_data["segment20"] == cls
            intersection = int((pred & gt).sum().item())
            union = int((pred | gt).sum().item())
            iou = intersection / union if union else float("nan")
            class_intersection = int((pred & class_mask).sum().item())
            class_union = int((pred | class_mask).sum().item())
            class_iou = (
                class_intersection / class_union if class_union else float("nan")
            )
            precision = intersection / int(pred.sum().item()) if pred.any() else float("nan")
            recall = intersection / int(gt.sum().item()) if gt.any() else float("nan")
            pred_fraction = float(pred.float().mean().item())
            gt_fraction = float(gt.float().mean().item())
            random_iou = (
                pred_fraction * gt_fraction
                / max(pred_fraction + gt_fraction - pred_fraction * gt_fraction, 1e-12)
            )
            text_logits = text_logits_list[0]
            iou_pred = iou_out_list[0]
            segmentation.append(
                {
                    "class": cls,
                    "gt_instance": instance_id,
                    "gt_points": int(gt_count),
                    "gt_instance_points": int(gt.sum().item()),
                    "prompt_coord": [round(float(v), 4) for v in prompt.flatten().tolist()],
                    "seconds": round(seg_seconds, 3),
                    "iou": round(float(iou), 4),
                    "class_iou": round(float(class_iou), 4),
                    "precision": round(float(precision), 4),
                    "recall": round(float(recall), 4),
                    "predicted_fraction": round(pred_fraction, 4),
                    "gt_fraction": round(gt_fraction, 4),
                    "random_baseline_iou": round(float(random_iou), 4),
                    "predicted_points": int(pred.sum().item()),
                    "logits_finite": bool(torch.isfinite(logits).all().item()),
                    "logits_min": round(float(logits.min().item()), 4),
                    "logits_max": round(float(logits.max().item()), 4),
                    "text_logits_shape": list(text_logits.shape),
                    "text_logits_finite": bool(torch.isfinite(text_logits).all().item()),
                    "iou_prediction_shape": list(iou_pred.shape),
                    "iou_prediction_finite": bool(torch.isfinite(iou_pred).all().item()),
                }
            )
            if not torch.isfinite(logits).all():
                report["failures"].append(f"non-finite segmentation logits for class {cls}")
            if not torch.isfinite(text_logits).all():
                report["failures"].append(f"non-finite text logits for class {cls}")
            if not torch.isfinite(iou_pred).all():
                report["failures"].append(f"non-finite IoU prediction for class {cls}")
            if not math.isnan(iou) and iou <= 0.0:
                report["failures"].append(f"empty mask for class {cls}")

    report["segmentation"] = {
        "metric": "mask IoU against the prompted ground-truth instance",
        "results": segmentation,
        "mean_iou": round(
            float(np.mean([item["iou"] for item in segmentation])), 4
        )
        if segmentation
        else None,
        "mean_class_iou": round(
            float(np.mean([item["class_iou"] for item in segmentation])), 4
        )
        if segmentation
        else None,
        "mean_random_baseline_iou": round(
            float(np.mean([item["random_baseline_iou"] for item in segmentation])), 4
        )
        if segmentation
        else None,
    }
    report["memory"] = _memory_mib()

    # A random-weight model would not beat the class-prior baseline.  Require the
    # real checkpoint to show clearly non-random overlap on at least one prompt.
    best_iou = max((item["iou"] for item in segmentation), default=0.0)
    best_baseline = max(
        (item["random_baseline_iou"] for item in segmentation), default=0.0
    )
    report["segmentation"]["best_iou"] = best_iou
    report["segmentation"]["best_random_baseline_iou"] = best_baseline
    if best_iou <= best_baseline + 0.05:
        report["failures"].append(
            "segmentation IoU does not clearly exceed the random-mask baseline; "
            "checkpoint may not have loaded correctly"
        )

    if args.min_iou is not None and best_iou < args.min_iou:
        report["failures"].append(
            f"best IoU {best_iou:.4f} is below --min-iou {args.min_iou:.4f}"
        )

    report["status"] = "PASS" if not report["failures"] else "FAIL"
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--checkpoint", default="checkpoints/SNAP_C.pth", help="pretrained checkpoint"
    )
    parser.add_argument(
        "--data",
        default="data_examples/ScanNet/scene0011_00",
        help="ScanNet-style directory with coord/color/normal/segment npy files",
    )
    parser.add_argument("--segment-file", default="segment20.npy")
    parser.add_argument("--domain", default="Indoor", choices=("Indoor", "Outdoor", "Aerial"))
    parser.add_argument("--grid-size", type=float, default=0.02)
    parser.add_argument("--device", default="cuda")
    parser.add_argument(
        "--attention-backend",
        default="auto",
        choices=("auto", "flash", "normal"),
        help="attention backend selection (default: auto)",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--max-points",
        type=int,
        default=0,
        help="deterministically subsample input points (0 = use all)",
    )
    parser.add_argument(
        "--check-chunk-equivalence",
        action="store_true",
        help="compare chunked and unchunked attention on the same input",
    )
    parser.add_argument("--max-prompts", type=int, default=3)
    parser.add_argument("--min-class-points", type=int, default=1000)
    parser.add_argument("--exclude-classes", default="0,1")
    parser.add_argument("--prompt-classes", default=None, help="comma-separated class ids")
    parser.add_argument("--min-iou", type=float, default=None)
    parser.add_argument("--output", default=None, help="optional JSON output path")
    args = parser.parse_args()
    args.exclude_classes = [
        int(value) for value in args.exclude_classes.split(",") if value.strip()
    ]
    if args.check_chunk_equivalence and not args.max_points:
        parser.error("--check-chunk-equivalence requires --max-points (e.g. 40000)")
    if args.check_chunk_equivalence and args.max_points > 60000:
        parser.error("--check-chunk-equivalence needs --max-points <= 60000 on 8 GiB")

    report = run(args)
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
