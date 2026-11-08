#!/usr/bin/env python3
"""Benchmark FP32 vs FP16 autocast on SNAP real-checkpoint inference.

The same checkpoint, ScanNet scene, grid sampling, prompt classes and prompt
coordinates are used for both precisions.  Order shuffling is disabled so the
comparison measures precision effects rather than the model's built-in
serialization randomness.  Reports timing, peak CUDA memory, finite outputs,
instance IoU and mask/logit agreement.

This is a measurement tool; it does not change the model default precision.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import statistics
import sys
import time
import traceback
from pathlib import Path

import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.ptv3_backbone_domain import SerializedPooling  # noqa: E402
from src.snap import SNAP  # noqa: E402


def _load_verify_module():
    path = REPO_ROOT / "tools" / "verify_gpu_inference.py"
    spec = importlib.util.spec_from_file_location("_snap_verify_gpu", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _sync():
    if torch.cuda.is_available():
        torch.cuda.synchronize()


def run(args):
    vg = _load_verify_module()
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
        torch.cuda.empty_cache()

    report = {
        "nonce": "gpu-20261108-01",
        "status": "FAIL",
        "environment": {
            "torch": torch.__version__,
            "torch_cuda": torch.version.cuda,
            "device": str(device),
            "device_name": torch.cuda.get_device_name(device)
            if torch.cuda.is_available()
            else None,
            "compute_capability": list(torch.cuda.get_device_capability(device))
            if torch.cuda.is_available()
            else None,
        },
        "data": {},
        "precision": {},
        "comparison": {},
        "failures": [],
    }

    data_args = argparse.Namespace(
        data=args.data,
        segment_file=args.segment_file,
        domain=args.domain,
        grid_size=args.grid_size,
        exclude_classes=args.exclude_classes,
        min_class_points=args.min_class_points,
        max_prompts=args.max_prompts,
        prompt_classes=None,
    )
    coord, color, normal, strength, segment, instance = vg._load_scan(data_args)
    if args.max_points and coord.shape[0] > args.max_points:
        rng = np.random.default_rng(args.seed)
        indices = np.sort(rng.choice(coord.shape[0], args.max_points, replace=False))
        coord, color, normal, strength, segment, instance = [
            value[indices] for value in (coord, color, normal, strength, segment, instance)
        ]
    data = vg._build_input(
        coord, color, normal, strength, segment, instance, data_args
    )
    data = {
        key: value.to(device) if isinstance(value, torch.Tensor) else value
        for key, value in data.items()
    }
    report["data"] = {
        "path": str(Path(args.data)),
        "original_points": int(coord.shape[0]),
        "sampled_points": int(data["coord"].shape[0]),
        "grid_size": args.grid_size,
        "domain": args.domain,
    }

    checkpoint = Path(args.checkpoint)
    if not checkpoint.is_absolute():
        checkpoint = REPO_ROOT / checkpoint
    state_dict = vg._load_checkpoint(checkpoint)
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
    if load_result.missing_keys:
        report["failures"].append(
            f"checkpoint missing model keys: {load_result.missing_keys[:10]}"
        )
    # Remove the model's serialization randomness for a controlled comparison.
    model.backbone.shuffle_orders = False
    for module in model.modules():
        if isinstance(module, SerializedPooling):
            module.shuffle_orders = False

    selected = vg._select_prompt_classes(data["segment20"].cpu().numpy(), data_args)
    prompts = []
    for cls, gt_count in selected:
        prompt, instance_mask, instance_id = vg._prompt_point(data, cls)
        prompts.append(
            {
                "class": int(cls),
                "gt_points": int(gt_count),
                "instance": instance_id,
                "instance_mask": instance_mask.detach().cpu(),
                "point": prompt.to(device),
            }
        )
    report["prompts"] = [
        {
            "class": prompt["class"],
            "gt_instance": prompt["instance"],
            "gt_instance_points": int(prompt["instance_mask"].sum()),
            "coord": [round(float(v), 4) for v in prompt["point"].flatten().tolist()],
        }
        for prompt in prompts
    ]

    def one_run(precision):
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats()
        enabled = precision == "fp16"
        _sync()
        total_start = time.perf_counter()
        with torch.no_grad(), torch.autocast(
            device_type=device.type, dtype=torch.float16, enabled=enabled
        ):
            encoded = model.run_input_encoder(dict(data))
            backbone_start = time.perf_counter()
            backbone_out = model.run_backbone(encoded)
            _sync()
            backbone_seconds = time.perf_counter() - backbone_start
            point = backbone_out[0] if isinstance(backbone_out, tuple) else backbone_out
            outputs = []
            decoder_start = time.perf_counter()
            for prompt in prompts:
                prompt_data = dict(encoded)
                prompt_data["point"] = prompt["point"]
                prompt_data["point_offset"] = [1]
                seg_logits, text_logits, iou_out, _aux, _w, _c, _cg = (
                    model.run_mask_decoder(point, prompt_data, val=True)
                )
                outputs.append(
                    {
                        "logits": seg_logits[0].detach().float().cpu(),
                        "text_shape": list(text_logits[0].shape),
                        "iou_shape": list(iou_out[0].shape),
                        "fine": bool(
                            torch.isfinite(seg_logits[0]).all().item()
                            and torch.isfinite(text_logits[0]).all().item()
                            and torch.isfinite(iou_out[0]).all().item()
                        ),
                    }
                )
            _sync()
            decoder_seconds = time.perf_counter() - decoder_start
        total_seconds = time.perf_counter() - total_start
        memory = {
            "max_allocated_mib": round(torch.cuda.max_memory_allocated() / 2**20, 1)
            if torch.cuda.is_available()
            else None,
            "max_reserved_mib": round(torch.cuda.max_memory_reserved() / 2**20, 1)
            if torch.cuda.is_available()
            else None,
        }
        return {
            "backbone_seconds": backbone_seconds,
            "decoder_seconds": decoder_seconds,
            "total_seconds": total_seconds,
            "memory": memory,
            "outputs": outputs,
            "feature_finite": bool(torch.isfinite(point.feat).all().item()),
            "feature_dtype": str(point.feat.dtype),
        }

    precision_results = {}
    for precision in ("fp32", "fp16"):
        # Warm up algorithm selection/caches, then measure.
        one_run(precision)
        runs = []
        for _ in range(args.repeats):
            runs.append(one_run(precision))
        last = runs[-1]
        per_prompt = []
        for index, prompt in enumerate(prompts):
            logits = last["outputs"][index]["logits"]
            pred = logits.sigmoid() > model.mask_threshold
            gt = prompt["instance_mask"]
            intersection = int((pred & gt).sum().item())
            union = int((pred | gt).sum().item())
            iou = intersection / union if union else float("nan")
            precision_value = (
                intersection / int(pred.sum().item()) if pred.any() else float("nan")
            )
            recall = intersection / int(gt.sum().item()) if gt.any() else float("nan")
            per_prompt.append(
                {
                    "class": prompt["class"],
                    "gt_instance": prompt["instance"],
                    "iou": round(float(iou), 4),
                    "precision": round(float(precision_value), 4),
                    "recall": round(float(recall), 4),
                    "predicted_points": int(pred.sum().item()),
                    "logits_finite": last["outputs"][index]["fine"],
                    "logits_min": round(float(logits.min().item()), 4),
                    "logits_max": round(float(logits.max().item()), 4),
                }
            )
            if not last["outputs"][index]["fine"]:
                report["failures"].append(
                    f"{precision} produced non-finite output for class {prompt['class']}"
                )
        if not last["feature_finite"]:
            report["failures"].append(f"{precision} produced non-finite backbone features")
        precision_results[precision] = {
            "backbone_seconds_median": round(
                statistics.median(run["backbone_seconds"] for run in runs), 4
            ),
            "decoder_seconds_median": round(
                statistics.median(run["decoder_seconds"] for run in runs), 4
            ),
            "total_seconds_median": round(
                statistics.median(run["total_seconds"] for run in runs), 4
            ),
            "memory": last["memory"],
            "feature_dtype": last["feature_dtype"],
            "mean_iou": round(
                float(np.mean([item["iou"] for item in per_prompt])), 4
            ),
            "prompts": per_prompt,
            "run_times": [round(run["total_seconds"], 4) for run in runs],
            "peak_memory_each_run": [run["memory"] for run in runs],
        }

    # Compare the last measured run of each precision.
    fp32_outputs = one_run("fp32")["outputs"]
    fp16_outputs = one_run("fp16")["outputs"]
    comparisons = []
    for index, prompt in enumerate(prompts):
        logits32 = fp32_outputs[index]["logits"]
        logits16 = fp16_outputs[index]["logits"]
        pred32 = logits32.sigmoid() > model.mask_threshold
        pred16 = logits16.sigmoid() > model.mask_threshold
        intersection = int((pred32 & pred16).sum().item())
        union = int((pred32 | pred16).sum().item())
        diff = (logits16 - logits32).abs()
        comparisons.append(
            {
                "class": prompt["class"],
                "mask_iou": round(intersection / union, 4) if union else None,
                "mask_agreement": round(
                    float((pred32 == pred16).float().mean().item()), 4
                ),
                "logits_max_abs_diff": round(float(diff.max().item()), 6),
                "logits_mean_abs_diff": round(float(diff.mean().item()), 6),
            }
        )
    report["comparison"] = {
        "prompts": comparisons,
        "mean_mask_iou": round(
            float(np.mean([item["mask_iou"] for item in comparisons])), 4
        )
        if comparisons and all(item["mask_iou"] is not None for item in comparisons)
        else None,
        "all_outputs_finite": all(
            precision_results[p]["prompts"][i]["logits_finite"]
            for p in ("fp32", "fp16")
            for i in range(len(prompts))
        ),
    }
    report["precision"] = precision_results

    fp32 = precision_results["fp32"]
    fp16 = precision_results["fp16"]
    report["summary"] = {
        "fp32_mean_iou": fp32["mean_iou"],
        "fp16_mean_iou": fp16["mean_iou"],
        "iou_delta_fp16_minus_fp32": round(fp16["mean_iou"] - fp32["mean_iou"], 4),
        "fp32_total_seconds": fp32["total_seconds_median"],
        "fp16_total_seconds": fp16["total_seconds_median"],
        "speedup_fp16": round(fp32["total_seconds_median"] / fp16["total_seconds_median"], 3)
        if fp16["total_seconds_median"]
        else None,
        "fp32_max_allocated_mib": fp32["memory"]["max_allocated_mib"],
        "fp16_max_allocated_mib": fp16["memory"]["max_allocated_mib"],
        "memory_reduction_mib": round(
            fp32["memory"]["max_allocated_mib"] - fp16["memory"]["max_allocated_mib"], 1
        )
        if fp32["memory"]["max_allocated_mib"] is not None
        and fp16["memory"]["max_allocated_mib"] is not None
        else None,
    }
    report["status"] = "PASS" if not report["failures"] else "FAIL"
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", default="checkpoints/SNAP_C.pth")
    parser.add_argument("--data", default="data_examples/ScanNet/scene0011_00")
    parser.add_argument("--segment-file", default="segment20.npy")
    parser.add_argument("--domain", default="Indoor")
    parser.add_argument("--grid-size", type=float, default=0.02)
    parser.add_argument("--device", default="cuda")
    parser.add_argument(
        "--attention-backend", default="auto", choices=("auto", "flash", "normal")
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-points", type=int, default=0)
    parser.add_argument("--max-prompts", type=int, default=3)
    parser.add_argument("--min-class-points", type=int, default=1000)
    parser.add_argument("--exclude-classes", default="0,1")
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--output", default=None)
    args = parser.parse_args()
    args.exclude_classes = [
        int(value) for value in args.exclude_classes.split(",") if value.strip()
    ]

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
