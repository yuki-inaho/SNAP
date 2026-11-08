#!/usr/bin/env python3
"""Emit a bounded GPU GUI test plan for one synthetic point-cloud scenario.

BASE, deliberately incomplete. It reports the obvious device but ignores the
FlashAttention capability/installation gate, treats every prompt round as one click,
uses an un-derived memory number, and always emits the same command list.
"""

from __future__ import annotations

import argparse
import json
import math


def plan(scenario: dict) -> dict:
    gpu = scenario.get("gpu") or {}
    device = "cuda" if gpu.get("available") else "cpu"
    rounds = scenario.get("prompt_rounds") or []
    points = int(scenario["points"])
    batch = int(scenario["batch_points"])
    return {
        "scene_id": scenario["scene_id"],
        "device": device,
        "attention_backend": "flash",
        "seed": int(scenario["seed"]),
        "click_rounds": len(rounds),
        "total_clicks": len(rounds),
        "points_per_batch": batch,
        "num_batches": math.ceil(points / batch),
        "memory_estimate_mb": 512,
        "memory_ok": True,
        "gui_model_mode": "real",
        "checks": [
            "pixi run just imports",
            "pixi run pytest -q tests/test_gui_interaction.py",
            "pixi run just validate",
            "pixi run just model-smoke",
        ],
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
