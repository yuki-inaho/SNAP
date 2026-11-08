#!/usr/bin/env python3
"""Independent judge: recompute the contract from the task scenario.

The judge sees a copy of the task prompt/source and a copy of the answer only. It does
not read the harness, a workspace, or an expected-answer file. Expected values are
recomputed here; a missing or malformed answer grades as failed criteria, not a crash.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

COMMON_CHECKS = [
    "pixi run just imports",
    "QT_QPA_PLATFORM=offscreen PYVISTA_OFF_SCREEN=true pixi run pytest -q tests/test_gui_interaction.py",
]
CUDA_CHECKS = ["pixi run just validate", "pixi run just model-smoke"]


def expected(scenario: dict) -> dict[str, Any]:
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
    checks = list(COMMON_CHECKS) + (list(CUDA_CHECKS) if device == "cuda" else [])
    return {
        "scene_id": scenario.get("scene_id"),
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
        "checks": checks,
    }


def _int(value: Any) -> int | None:
    return value if isinstance(value, int) and not isinstance(value, bool) else None


def _criteria(answer: dict, want: dict) -> list[dict]:
    checks = answer.get("checks")
    return [
        {
            "name": "scene_id",
            "passed": answer.get("scene_id") == want["scene_id"],
        },
        {
            "name": "device",
            "passed": answer.get("device") == want["device"],
        },
        {
            "name": "attention_backend",
            "passed": answer.get("attention_backend") == want["attention_backend"],
        },
        {
            "name": "clicks",
            "passed": _int(answer.get("click_rounds")) == want["click_rounds"]
            and _int(answer.get("total_clicks")) == want["total_clicks"],
        },
        {
            "name": "seed",
            "passed": _int(answer.get("seed")) == want["seed"],
        },
        {
            "name": "batching",
            "passed": _int(answer.get("points_per_batch")) == want["points_per_batch"]
            and _int(answer.get("num_batches")) == want["num_batches"],
        },
        {
            "name": "memory",
            "passed": _int(answer.get("memory_estimate_mb"))
            == want["memory_estimate_mb"]
            and answer.get("memory_ok") is want["memory_ok"],
        },
        {
            "name": "gui_model_mode",
            "passed": answer.get("gui_model_mode") == want["gui_model_mode"],
        },
        {
            "name": "checks",
            "passed": isinstance(checks, list) and checks == want["checks"],
        },
    ]


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", type=Path, required=True)
    parser.add_argument("--answer", type=Path, required=True)
    args = parser.parse_args()

    scenario = json.loads(
        (args.task / "source" / "scenario.json").read_text(encoding="utf-8")
    )
    try:
        answer = json.loads(args.answer.read_text(encoding="utf-8"))
        if not isinstance(answer, dict):
            answer = {}
    except (OSError, json.JSONDecodeError):
        answer = {}

    criteria = _criteria(answer, expected(scenario))
    n_passed = sum(1 for c in criteria if c["passed"])
    print(
        json.dumps(
            {
                "criteria": criteria,
                "n_passed": n_passed,
                "n_criteria": len(criteria),
                "all_pass": n_passed == len(criteria),
            }
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
