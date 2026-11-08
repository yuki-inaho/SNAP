#!/usr/bin/env python3
"""Reference planner for the synthetic Harness² fixture.

This file is a test oracle, not part of the editable base harness. It is never copied to an
editor workspace, so the editor must derive the contract from the task prompt.
"""

from __future__ import annotations

import argparse
import json
import math

COMMON_CHECKS = [
    "pixi run just imports",
    "QT_QPA_PLATFORM=offscreen PYVISTA_OFF_SCREEN=true pixi run pytest -q tests/test_gui_interaction.py",
]
CUDA_CHECKS = ["pixi run just validate", "pixi run just model-smoke"]


def plan(scenario: dict) -> dict:
    gpu = scenario.get("gpu") or {}
    device = "cuda" if gpu.get("available") else "cpu"
    rounds = scenario.get("prompt_rounds") or []
    points = int(scenario["points"])
    batch = int(scenario["batch_points"])
    if device == "cuda":
        capability = float(gpu.get("compute_capability") or 0.0)
        backend = (
            "flash"
            if bool(scenario.get("attention_installed")) and capability >= 8.0
            else "normal"
        )
    else:
        backend = "normal"
    per_batch = min(points, batch)
    estimate = 512 + 8 * math.ceil(points / 1024) + (256 if device == "cuda" else 0)
    return {
        "scene_id": scenario["scene_id"],
        "device": device,
        "attention_backend": backend,
        "seed": int(scenario["seed"]),
        "click_rounds": len(rounds),
        "total_clicks": sum(int(r.get("clicks", 0)) for r in rounds),
        "points_per_batch": per_batch,
        "num_batches": math.ceil(points / per_batch),
        "memory_estimate_mb": estimate,
        "memory_ok": estimate <= int(scenario["memory_budget_mb"]),
        "gui_model_mode": "fake",
        "checks": COMMON_CHECKS + (CUDA_CHECKS if device == "cuda" else []),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--scenario", required=True)
    args = parser.parse_args()
    with open(args.scenario, encoding="utf-8") as handle:
        scenario = json.load(handle)
    print(json.dumps(plan(scenario), sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
