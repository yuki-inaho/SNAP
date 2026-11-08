#!/usr/bin/env python3
"""Executor: run the candidate planner and write its JSON stdout as the answer.

The harness tree is supplied by Harness²; this script never reads the judge or an
expected-answer file. A missing planner, a non-zero exit, or non-JSON output is a
failure, not an empty success.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", type=Path, required=True)
    parser.add_argument("--harness", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()

    planner = args.harness / "scripts" / "plan_gui_ml_test.py"
    scenario = args.task / "source" / "scenario.json"
    started = time.time()
    try:
        proc = subprocess.run(
            [sys.executable, str(planner), "--scenario", str(scenario)],
            capture_output=True,
            text=True,
            timeout=30,
        )
    except (OSError, subprocess.TimeoutExpired) as exc:
        print(f"planner failed to start: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 1

    parsed = None
    parse_error = ""
    if proc.returncode == 0:
        try:
            parsed = json.loads(proc.stdout)
        except json.JSONDecodeError as exc:
            parse_error = f"{type(exc).__name__}: {exc}"
    else:
        parse_error = f"planner exited {proc.returncode}"

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(
        json.dumps(parsed, indent=2, sort_keys=True) + "\n"
        if parsed is not None
        else json.dumps({"error": parse_error}, indent=2) + "\n",
        encoding="utf-8",
    )
    (args.out.parent / "trajectory.md").write_text(
        "[ACTION] run scripts/plan_gui_ml_test.py on source/scenario.json\n"
        f"[OBSERVATION] {proc.stdout.strip()[:4000] or '(no stdout)'}\n"
        f"[STDERR] {proc.stderr.strip()[:2000] or '(none)'}\n"
        f"[ACTION] wrote {args.out.name} after {time.time() - started:.2f}s\n",
        encoding="utf-8",
    )
    if parsed is None:
        print(f"no usable planner JSON: {parse_error}", file=sys.stderr)
        if proc.stdout.strip():
            print(proc.stdout[-2000:], file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
